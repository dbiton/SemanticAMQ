import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Optional
from filters.filter import Filter, InsufficientMemoryError


class DistanceRegressor(nn.Module):
    """Two hidden layers, not one: "distance to the nearest of many anchors"
    is a genuinely piecewise, non-convex function of the query (its argmin
    changes across Voronoi-like regions), and a single hidden layer measurably
    can't represent it tightly enough to recover near-zero distances at real
    anchors -- widening a single layer from 128 to 512 units didn't help
    (self-recall stuck around 0.4 either way in the measurements that led to
    this), while a second layer with the same total width reliably drove it
    to ~1.0. See _choose_hidden_dim for the matching two-layer parameter
    count."""
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),  # distance is non-negative
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MLPFilter(Filter):
    """
    A small MLP is trained to regress the distance from a query vector to its
    nearest construction vector, rather than to classify duplicate-or-not at
    one fixed threshold. Training pairs are (vector, log1p of the true
    nearest-neighbor distance to the construct set -- see
    _generate_training_data for why log-space); at query time a vector is
    accepted iff the predicted (log-space) distance is <= log1p(distance_
    threshold), equivalent to predicted_distance <= distance_threshold since
    log1p is strictly monotonic.

    This replaces an earlier design that trained a binary classifier on
    synthetic samples labelled by a fixed tau: every training example lived
    within ~1.5*tau of some construct vector, so the classifier never saw a
    genuinely far-away point and its behaviour out there was pure
    extrapolation -- unconstrained by anything in the loss, and empirically
    flipped from "mostly negative" to "mostly positive" as hidden_dim grew
    (more capacity fit the same narrow, mildly positive-skewed local signal
    more confidently, and that confidence generalised the wrong way across
    the vast region of embedding space real queries actually come from).

    Regressing distance instead fixes this at the source: distance-to-
    nearest-anchor is a real, smooth, non-negative function defined
    everywhere (the min of Euclidean distances to the construct set), so
    training on a spread of radii -- including points sampled uniformly over
    the whole embedding space, not just close to some anchor -- gives the
    network an actual target to fit far from the data too, instead of an
    open decision boundary whose sign was never specified there. It also
    decouples the model from tau entirely: one trained regressor supports
    any distance_threshold at query time, rather than needing to be retrained
    per threshold.

    Three sample families make up the training set, all labelled with the
    exact nearest distance to `vectors` (self-distances included, so every
    construct vector itself is a zero-distance example). None assumes
    unit-normalised input -- every scale below is derived from `vectors`
    itself, so the same code trains correctly whether embeddings sit on the
    unit sphere or not. Together they cover the distance axis continuously
    from 0 outwards, which is the property that matters: a band with no
    training samples in it is a band the regressor extrapolates into, and
    the sign of that extrapolation is not specified by anything in the loss.
      - anchor samples: each construct vector jittered by a random direction
        and a radius drawn uniformly up to ANCHOR_RADIUS_MULTIPLIER times the
        near-field scale (distance_threshold when known, else the data's own
        median nearest-neighbor spacing -- see _near_field_scale). This is
        where resolution matters most, since it straddles the actual
        accept/reject boundary.
      - mid-field samples: also anchor-centered, radii log-spaced from the
        anchor family's outer radius (or the set's own median spacing, if
        that is wider) out to its ambient spread (_ambient_scale), and only
        when those two edges are far enough apart to be worth sampling
        (MIDFIELD_MIN_BAND_RATIO). This is the band real queries actually
        fall in and the one the other two families left empty -- see
        MIDFIELD_SAMPLES_PER_VECTOR for what that cost.
      - global samples: points scattered around the data's centroid at radii
        log-spaced from the data's own ambient spread (_ambient_scale) out to
        GLOBAL_RADIUS_SPAN times that, so the regressor sees the distance
        target keep growing far past the construct set's own extent -- fixing
        the old design's undefined behaviour out there -- without training on
        an arbitrary absolute scale that may not match the input at all.
    """
    HIDDEN_DIM = 128
    # Gradient steps, not epochs: an epoch is one pass over a DataLoader that
    # re-batches at BATCH_SIZE, so its step count scales with dataset size --
    # fixing a step BUDGET instead keeps training cost roughly constant
    # whether the construct set is 20 vectors (one batch/epoch, so this
    # becomes the epoch count directly) or 20,000 (many batches/epoch, so
    # fewer epochs reach the same number of updates). 900 was the smallest
    # budget that reliably reached ~1.0 self/neighbor recall across five
    # random seeds at this architecture (measured; 700 already got most of
    # the way, 500 was inconsistent).
    TARGET_GRADIENT_STEPS = 900
    # Compute-time safety valve on _choose_hidden_dim's budget-driven width,
    # not an accuracy cap -- see its docstring. Generous relative to HIDDEN_DIM
    # so the equal-memory sweep's realistic budget range actually drives
    # hidden_dim instead of saturating against this.
    MAX_HIDDEN_DIM = 1024
    LR_DECAY_STAGES = 5
    LR_DECAY_GAMMA = 0.6
    BATCH_SIZE = 256
    LEARNING_RATE = 5e-3
    ANCHOR_RADIUS_MULTIPLIER = 3.0
    GLOBAL_RADIUS_SPAN = 100.0
    ANCHOR_SAMPLES_PER_VECTOR = 4
    GLOBAL_SAMPLES_PER_VECTOR = 2
    # Mid-field samples per construct vector: the band between the anchor
    # jitter's outer radius (ANCHOR_RADIUS_MULTIPLIER * near_scale) and the
    # construct set's own ambient spread. This is where real query vectors
    # actually live, and before this family existed nothing in the training
    # set did. Measured on the Quora blocklist workload (|B|=20,000,
    # tau=0.1): 69% of real queries have a nearest-construct distance in
    # [0.3, 1.0), a band that held 0.00% of the training samples -- 71% sat
    # below 0.3 and 29% above 1.0. The regressor had to extrapolate every
    # one of those queries from the near-field side, which is why it called
    # them duplicates and reported FPR between 0.3 and 0.9 with FNR 0
    # regardless of how much memory it was given. Sized to match the anchor
    # family so the gap is covered as densely as the region around tau.
    MIDFIELD_SAMPLES_PER_VECTOR = 4
    # The mid-field family exists to bridge a GAP between the near field and
    # the ambient scale, so it only fires when there is one: the band must
    # span at least this ratio. On a sparse construct set the two edges nearly
    # touch (30 gaussian points in 32-d: band [6.35, 7.14], ratio 1.12) and
    # sampling it just piles mass at one radius, right where the far field
    # already starts -- which measurably cost both self-recall and
    # neighbor-recall there. The blocklist reference sets span 3.6x
    # ([0.3, 1.07]), comfortably clear of this.
    MIDFIELD_MIN_BAND_RATIO = 1.5
    # Down-weights the far-field samples' contribution to the loss: without
    # this, SmoothL1 spends its gradient budget fitting the (less important)
    # far field as accurately as the near field, which measurably starved the
    # near-threshold precision that the real accept/reject decision needs.
    FAR_FIELD_LOSS_WEIGHT = 0.2
    RANDOM_SEED = 42

    def __init__(self, dim: int, distance_threshold: Optional[float] = None, max_memory: Optional[int] = None):
        super().__init__(dim=dim, max_memory=max_memory)
        self.distance_threshold = distance_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_dim = self._choose_hidden_dim()
        self.model: Optional[DistanceRegressor] = None
        self.rng = np.random.default_rng(self.RANDOM_SEED)
        torch.manual_seed(self.RANDOM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.RANDOM_SEED)


    def _choose_hidden_dim(self) -> int:
        """Largest hidden width whose two-hidden-layer parameter count fits
        max_memory. DistanceRegressor's params: dim*h + h (layer 1) + h*h + h
        (layer 2) + h + 1 (output) = h^2 + h*(dim+3) + 1 -- quadratic in h,
        unlike the single hidden layer's linear h*(dim+2)+1 this replaces
        (see DistanceRegressor for why the second layer is load-bearing)."""
        if self.max_memory is None:
            return self.HIDDEN_DIM

        max_params = self.max_memory // 4
        a, b, c = 1.0, float(self.dim + 3), float(1 - max_params)
        discriminant = b * b - 4.0 * a * c
        hidden_dim = int((-b + discriminant ** 0.5) / (2.0 * a)) if discriminant >= 0 else 0
        # The quadratic root can round up past the budget; step down to the
        # largest integer that actually fits it.
        while hidden_dim > 0 and hidden_dim * hidden_dim + hidden_dim * (self.dim + 3) + 1 > max_params:
            hidden_dim -= 1
        # MAX_HIDDEN_DIM is a compute-time safety valve only (training cost
        # is quadratic-ish in hidden_dim), not an accuracy cap -- clamping to
        # HIDDEN_DIM (the empirically-tuned no-budget default, 128) here used
        # to silently defeat the whole point of this method: every budget
        # above the byte cost of a 128-wide network built the exact same
        # model, so an equal-memory sweep measured a flat, non-improving MLP
        # past that point instead of one that uses the memory it's given.
        hidden_dim = min(hidden_dim, self.MAX_HIDDEN_DIM)

        if hidden_dim < 1:
            # InsufficientMemoryError, not ValueError: this is the shared
            # "budget too tight to build at all" signal that
            # runner.equal_memory_worker turns into an explicit infeasible
            # row. As a ValueError it escaped that handler and took the whole
            # per-budget sweep job down -- the same failure mode DSHFilter's
            # bare IndexError had.
            raise InsufficientMemoryError(
                f"max_memory={self.max_memory} bytes is too small for an MLPFilter with "
                f"dim={self.dim}: the smallest two-hidden-layer network (hidden_dim=1) needs "
                f"{(1 + (self.dim + 3) + 1) * 4} bytes of float32 parameters."
            )

        return hidden_dim


    def _sample_unit_directions(self, num_samples: int) -> np.ndarray:
        raw = self.rng.standard_normal((num_samples, self.dim), dtype=np.float32)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        return raw / (norms + 1e-8)


    def _nearest_construct_distances(self, queries: np.ndarray, construct_vectors: np.ndarray) -> np.ndarray:
        construct_norms = np.sum(construct_vectors ** 2, axis=1)
        nearest_distances = np.empty(len(queries), dtype=np.float32)
        batch_size = 1024

        for start in range(0, len(queries), batch_size):
            end = min(start + batch_size, len(queries))
            batch = queries[start:end]
            batch_norms = np.sum(batch ** 2, axis=1)
            squared_distances = batch_norms[:, None] + construct_norms[None, :] - 2.0 * (batch @ construct_vectors.T)
            nearest_distances[start:end] = np.sqrt(np.maximum(np.min(squared_distances, axis=1), 0.0))

        return nearest_distances


    def _median_nn_spacing(self, vectors: np.ndarray) -> float:
        """How far apart `vectors` typically sit: the median
        nearest-*other*-neighbor distance within the set, from a bounded
        subsample so this stays cheap even for a large construct set.

        This is the construct set's own resolution. Two callers need it:
        _near_field_scale, as its fallback when no threshold was given, and
        _generate_training_data, to keep the mid-field family from sampling
        inside it (see MIDFIELD_SAMPLES_PER_VECTOR)."""
        n = len(vectors)
        if n < 2:
            return 0.0
        sample_size = min(256, n)
        idx = self.rng.choice(n, size=sample_size, replace=False)
        sample = vectors[idx]
        d2 = (np.sum(sample ** 2, axis=1)[:, None] + np.sum(vectors ** 2, axis=1)[None, :]
              - 2.0 * (sample @ vectors.T))
        d2 = np.maximum(d2, 0.0)
        d2[np.arange(sample_size), idx] = np.inf  # exclude self-match
        return float(np.median(np.sqrt(d2.min(axis=1))))

    def _near_field_scale(self, vectors: np.ndarray) -> float:
        """The radius that matters most for the real accept/reject decision:
        distance_threshold when the caller gave one (every real caller does),
        else the construct set's own median nearest-neighbor spacing."""
        if self.distance_threshold is not None:
            return max(float(self.distance_threshold), 1e-6)
        if len(vectors) < 2:
            return 1.0
        return max(self._median_nn_spacing(vectors), 1e-6)


    def _ambient_scale(self, vectors: np.ndarray, centroid: np.ndarray) -> float:
        """How far the construct set itself spreads from its centroid --
        the scale "just past the data" starts at, for global far-field
        sampling. Derived from `vectors`, never assumed, so this adapts to
        any input scale (unit-normalised embeddings, unnormalised vectors,
        an arbitrarily shifted dataset, ...)."""
        spread = np.linalg.norm(vectors - centroid[None, :], axis=1)
        return max(float(spread.max()) if len(spread) else 0.0, 1e-3)


    def _generate_training_data(self, vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        num_anchors = len(vectors)
        centroid = vectors.mean(axis=0)

        # Near-field: dense coverage around the actual operating threshold,
        # where the real duplicate decision is made.
        near_scale = self._near_field_scale(vectors)
        num_anchor_samples = num_anchors * self.ANCHOR_SAMPLES_PER_VECTOR
        base_vectors = np.repeat(vectors, self.ANCHOR_SAMPLES_PER_VECTOR, axis=0)
        radii = self.rng.uniform(
            0.0, self.ANCHOR_RADIUS_MULTIPLIER * near_scale, size=num_anchor_samples
        ).astype(np.float32).reshape(-1, 1)
        directions = self._sample_unit_directions(num_anchor_samples)
        anchor_samples = base_vectors + directions * radii

        far_scale = self._ambient_scale(vectors, centroid)

        # Mid-field: anchor-centered, radii log-spaced across the band between
        # the anchor family's outer radius and the construct set's own ambient
        # spread -- the band real queries occupy (see
        # MIDFIELD_SAMPLES_PER_VECTOR). Anchor-centered rather than
        # centroid-centered so the samples land on the data's own shell, where
        # queries come from, instead of on a sphere around the centroid.
        # Never inside the construct set's own typical spacing: within that
        # radius the near-field family already owns the resolution, and the
        # regressor's job there is the sharp dip to 0 at each anchor. Putting
        # mid-field mass inside it competes with that dip instead of adding
        # information, which measurably cost self-recall on a sparse construct
        # set (30 gaussian points in 32-d, median spacing 6.3, tau 0.3: 0.97
        # -> 0.77). On the blocklist reference sets the two regimes are far
        # apart -- they are built from duplicate clusters, so their median
        # spacing is 0.0 and the band still starts at 3*tau as intended.
        near_outer = max(self.ANCHOR_RADIUS_MULTIPLIER * near_scale,
                         self._median_nn_spacing(vectors))
        midfield_samples = np.empty((0, self.dim), dtype=np.float32)
        if far_scale >= near_outer * self.MIDFIELD_MIN_BAND_RATIO:
            num_midfield_samples = num_anchors * self.MIDFIELD_SAMPLES_PER_VECTOR
            midfield_base = np.repeat(vectors, self.MIDFIELD_SAMPLES_PER_VECTOR, axis=0)
            midfield_log = self.rng.uniform(
                np.log10(near_outer), np.log10(far_scale), size=num_midfield_samples)
            midfield_radii = (10.0 ** midfield_log).astype(np.float32).reshape(-1, 1)
            midfield_directions = self._sample_unit_directions(num_midfield_samples)
            midfield_samples = midfield_base + midfield_directions * midfield_radii

        # Far-field: log-spaced radii from the data's own ambient scale out to
        # GLOBAL_RADIUS_SPAN times that, centered on the data's centroid --
        # this is what teaches the regressor that distance keeps growing well
        # past the construct set, instead of leaving that region undefined.
        num_global_samples = num_anchors * self.GLOBAL_SAMPLES_PER_VECTOR
        log_span = self.rng.uniform(0.0, np.log10(self.GLOBAL_RADIUS_SPAN), size=num_global_samples)
        global_radii = (far_scale * (10.0 ** log_span)).astype(np.float32).reshape(-1, 1)
        global_directions = self._sample_unit_directions(num_global_samples)
        global_samples = centroid[None, :] + global_directions * global_radii

        X = np.vstack([vectors, anchor_samples, midfield_samples, global_samples]).astype(np.float32)
        # Self-distances included: every anchor's nearest neighbor in
        # `vectors` (which contains it) is itself, at distance 0 -- exactly
        # the label a duplicate-of-itself example should carry.
        distances = self._nearest_construct_distances(X, vectors).astype(np.float32)
        # log1p, not raw distance: far-field samples can be ~GLOBAL_RADIUS_SPAN
        # times the near-field scale, and training directly on raw distance
        # let those few huge targets dominate the loss, degrading precision
        # exactly where it matters -- near the real accept/reject threshold
        # (measured: construct vectors regressed to ~1.2 instead of ~0).
        # log1p compresses that range (log1p(0)=0, log1p(near_scale) stays
        # small, log1p(far_scale * GLOBAL_RADIUS_SPAN) stays a modest single
        # digit) while remaining strictly monotonic, so thresholding in
        # log-space (see query()) is equivalent to thresholding raw distance.
        y = np.log1p(distances).astype(np.float32)

        # Near and mid-field samples get full weight -- between them they
        # cover every distance a real query is at, so both sides of the
        # accept/reject decision are theirs. Only the far field is
        # down-weighted (see FAR_FIELD_LOSS_WEIGHT): it still teaches
        # "distance keeps growing out here" without competing for gradient
        # budget against the region the decision is actually made in.
        weights = np.full(len(X), self.FAR_FIELD_LOSS_WEIGHT, dtype=np.float32)
        weights[:num_anchors + num_anchor_samples + len(midfield_samples)] = 1.0

        return X, y, weights


    def _train(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> None:
        self.model = DistanceRegressor(self.dim, self.hidden_dim).to(self.device)

        X_tensor = torch.from_numpy(X).to(self.device)
        y_tensor = torch.from_numpy(y).unsqueeze(1).to(self.device)
        w_tensor = torch.from_numpy(weights).unsqueeze(1).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor, w_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.BATCH_SIZE, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=self.LEARNING_RATE)
        # Unreduced so each sample's loss can be scaled by its weight before
        # averaging -- see FAR_FIELD_LOSS_WEIGHT.
        criterion = nn.SmoothL1Loss(reduction="none")

        # See TARGET_GRADIENT_STEPS: fix the number of gradient updates, not
        # the number of passes over the data, so training cost doesn't scale
        # with how many batches one epoch happens to contain. TARGET_GRADIENT_STEPS
        # itself was only measured at HIDDEN_DIM=128 (see its docstring); a
        # wider network built by a generous memory budget has more parameters
        # to fit and empirically needs proportionally more updates to reach
        # the same convergence, or the extra capacity a bigger budget buys
        # goes untrained rather than translating into lower error.
        target_steps = self.TARGET_GRADIENT_STEPS * max(1.0, self.hidden_dim / self.HIDDEN_DIM)
        steps_per_epoch = max(1, -(-len(X) // self.BATCH_SIZE))  # ceil division
        epochs = max(1, -(-int(target_steps) // steps_per_epoch))
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, epochs // self.LR_DECAY_STAGES), gamma=self.LR_DECAY_GAMMA)

        self.model.train()

        for _ in range(epochs):
            for batch_x, batch_y, batch_w in loader:
                optimizer.zero_grad()
                loss = (criterion(self.model(batch_x), batch_y) * batch_w).mean()
                loss.backward()
                optimizer.step()
            scheduler.step()

        self.model.eval()


    def construct(self, vectors: np.ndarray) -> None:
        if len(vectors) == 0:
            return

        vectors = np.ascontiguousarray(vectors, dtype=np.float32)

        X, y, weights = self._generate_training_data(vectors)
        self._train(X, y, weights)


    def query(self, vectors: np.ndarray) -> List[bool]:
        if self.model is None:
            raise ValueError("MLPFilter has not been constructed.")
        if self.distance_threshold is None:
            raise ValueError("MLPFilter.query() requires distance_threshold to be set.")

        if len(vectors) == 0:
            return [False] * len(vectors)

        X_tensor = torch.from_numpy(np.ascontiguousarray(vectors, dtype=np.float32)).to(self.device)

        with torch.no_grad():
            predicted_log_distance = self.model(X_tensor).squeeze(1).cpu().numpy()

        # Compare in the same log1p space the model was trained on (see
        # _generate_training_data) -- log1p is strictly monotonic, so this is
        # equivalent to predicted_distance <= distance_threshold without ever
        # needing to invert the transform.
        return (predicted_log_distance <= np.log1p(self.distance_threshold)).tolist()


    def insert(self, vectors: np.ndarray) -> None:
        raise NotImplementedError("MLPFilter does not support insert operation.")


    def add(self, vectors: np.ndarray) -> None:
        self.insert(vectors)


    def remove(self, vectors: np.ndarray) -> List[bool]:
        raise NotImplementedError("MLPFilter does not support remove operation.")


    def get_bytesize(self) -> int:
        if self.model is None:
            return 0
        model_bytes = sum(p.nelement() * p.element_size() for p in self.model.parameters())
        buffer_bytes = sum(b.nelement() * b.element_size() for b in self.model.buffers())
        return model_bytes + buffer_bytes
