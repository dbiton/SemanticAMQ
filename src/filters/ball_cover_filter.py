from enum import Enum
from typing import Optional

import faiss
import numpy as np


class Quantize(Enum):
    NONE = "None"
    BF16 = "SQbf16"
    FP16 = "SQfp16"
    SQ8 = "SQ8"
    SQ6 = "SQ6"
    SQ4 = "SQ4"
    SQ4T = "SQ4_trained"
    SQ8T = "SQ8_trained"
    PQ = "PQ"

PQ_MAX_NBITS = 8

_QTYPES = {
    Quantize.BF16: faiss.ScalarQuantizer.QT_bf16,
    Quantize.FP16: faiss.ScalarQuantizer.QT_fp16,
    Quantize.SQ8: faiss.ScalarQuantizer.QT_8bit,
    Quantize.SQ6: faiss.ScalarQuantizer.QT_6bit,
    Quantize.SQ4: faiss.ScalarQuantizer.QT_4bit,
    Quantize.SQ4T: faiss.ScalarQuantizer.QT_4bit_tqmse,
    Quantize.SQ8T: faiss.ScalarQuantizer.QT_8bit_tqmse,
}

_INSERTION_LEVELS = ("nearest", "least_inflation", "merge", "doubling")


class BallCoverFilter:
    def __init__(
        self,
        dim: int,
        max_memory: int | None  = None,
        quantize: Quantize = Quantize.NONE,
        construction_level: int = 3,
        insertion_level: str | None = None,
    ) -> None:
        self.dim = dim
        self.quantize = quantize
        self.construction_level = construction_level
        self.insertion_level = insertion_level
        self.index = self._make_index(quantize)
        self.max_count_balls = self.calculate_max_balls(max_memory)
        self.count_balls = 0
        self.radii = np.zeros(self.max_count_balls, dtype=np.float32)
        self._doubling_R = 0.0  # lazily initialized on first add_doubling() insertion

    @property
    def all_radiuses(self) -> np.ndarray:
        return self.radii[: self.count_balls]

    @staticmethod
    def _choose_pq_m(dim: int) -> int:
        for m in range(min(dim, PQ_MAX_NBITS), 0, -1):
            if dim % m == 0:
                return m
        return 1

    @staticmethod
    def _pq_nbits_for(n: int) -> Optional[int]:
        """Largest nbits (<= 8) whose 2**nbits codebook can be trained on n
        points, or None if there aren't even enough points for a 1-bit
        codebook (n < 2) -- IndexPQ.search segfaults at nbits=0, so that case
        must be treated as "PQ unavailable", not built with nbits=0."""
        nbits = 0
        while nbits < PQ_MAX_NBITS and (1 << (nbits + 1)) <= n:
            nbits += 1
        return nbits if nbits >= 1 else None

    def _make_index(self, quantize: Quantize, pq_nbits: int = PQ_MAX_NBITS) -> faiss.Index:
        if quantize == Quantize.NONE:
            return faiss.IndexFlatL2(self.dim)
        if quantize == Quantize.PQ:
            pq_m = self._choose_pq_m(self.dim)
            return faiss.IndexPQ(self.dim, pq_m, pq_nbits, faiss.METRIC_L2)
        return faiss.IndexScalarQuantizer(self.dim, _QTYPES[quantize], faiss.METRIC_L2)

    def bytes_per_ball(self) -> int:
        if isinstance(self.index, faiss.IndexPQ):
            return self.index.pq.code_size + 4
        if isinstance(self.index, faiss.IndexScalarQuantizer):
            return self.index.sq.code_size + 4
        return (self.index.d + 1) * 4

    def calculate_max_balls(self, max_memory: int) -> int:
        return max(1, max_memory // self.bytes_per_ball())

    def get_bytesize(self) -> int:
        return self.count_balls * self.bytes_per_ball()

    def construct_enclosing_balls(
        self, vectors: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Gonzalez farthest-first, then exact radii. O(n*k*d)."""
        X = np.ascontiguousarray(vectors, dtype=np.float32)
        n = len(X)
        k = min(self.max_count_balls, n)
        if n <= k:
            return X.copy(), np.zeros(n, dtype=np.float32)

        chosen = [0]
        min_d2 = ((X - X[0]) ** 2).sum(1)
        for _ in range(k - 1):
            chosen.append(int(min_d2.argmax()))
            min_d2 = np.minimum(min_d2, ((X - X[chosen[-1]]) ** 2).sum(1))
        centers = np.ascontiguousarray(X[chosen])

        index = faiss.IndexFlatL2(self.dim)
        index.add(centers)
        d2, labels = index.search(X, 1)
        radii = np.zeros(k, dtype=np.float32)
        np.maximum.at(radii, labels.ravel(), np.sqrt(np.maximum(d2.ravel(), 0)))
        return centers, radii

    def _build_quantizer(
        self, quantize: Quantize, centers: np.ndarray
    ) -> Optional[tuple[faiss.Index, np.ndarray]]:
        """Train an index of the given quantizer type over centers (not yet
        populated -- caller adds centers to the returned index) and return the
        per-center reconstruction error ||center - decode(encode(center))||.
        Returns None if this quantizer type can't be built for this many
        centers (only possible for PQ: too few centers for even a 1-bit
        codebook)."""
        if quantize == Quantize.PQ:
            nbits = self._pq_nbits_for(len(centers))
            if nbits is None:
                return None
            index = self._make_index(quantize, pq_nbits=nbits)
            index.pq.cp.min_points_per_centroid = 1  # our "training set" is just the ball centers
            index.train(centers)
            decoded = index.pq.decode(index.pq.compute_codes(centers))
        else:
            index = self._make_index(quantize)
            index.train(centers)
            decoded = index.sq.decode(index.sq.compute_codes(centers))
        error = np.linalg.norm(centers - decoded, axis=1).astype(np.float32)
        return index, error

    def construct(self, vectors: np.ndarray) -> None:
        centers, radii = self.construct_enclosing_balls(vectors)
        centers = np.ascontiguousarray(centers, dtype=np.float32)
        self._rebuild_from_centers(centers, radii)

    def _rebuild_from_centers(self, centers: np.ndarray, radii: np.ndarray) -> None:
        """Re-derive self.index/self.radii/self.count_balls from an explicit
        (centers, radii) pair, applying the configured quantizer exactly like
        construct() does. Used both by construct() itself and by the
        insertion methods that restructure the ball set (add_merge,
        add_doubling) rather than just growing an existing ball's radius."""
        centers = np.ascontiguousarray(centers, dtype=np.float32)
        radii = np.asarray(radii, dtype=np.float32)
        n = len(centers)

        if n == 0:
            index = self._make_index(Quantize.NONE)
        elif self.quantize == Quantize.NONE:
            index = self._make_index(Quantize.NONE)
            index.add(centers)
        else:
            built = self._build_quantizer(self.quantize, centers)
            if built is None:
                index = self._make_index(Quantize.NONE)
                index.add(centers)
            else:
                index, error = built
                radii = radii + error
                index.add(centers)

        self.index = index

        self.radii[:] = 0.0
        self.radii[:n] = radii
        self.count_balls = n

    def distance_to_balls_surfaces(self, query_vectors: np.ndarray) -> np.ndarray:
        """(nq, k) signed distance to each ball surface. Negative = inside."""
        Q = np.ascontiguousarray(query_vectors, dtype=np.float32)
        d2, labels = self.index.search(Q, self.count_balls)
        return np.sqrt(np.maximum(d2, 0.0)) - self.radii[labels]

    def query(self, query_vectors: np.ndarray, distance_threshold: float) -> list[bool]:
        surface = self.distance_to_balls_surfaces(query_vectors)
        return (surface.min(axis=1) < distance_threshold).tolist()

    def insert(self, vectors: np.ndarray) -> None:
        """Alias for add(), kept for compatibility with the evaluation harness."""
        self.add(vectors)

    def add(self, vectors: np.ndarray) -> None:
        """Dispatches to the method named by self.insertion_level (default
        "nearest" when None, matching the pre-insertion_level behavior)."""
        level = self.insertion_level or "nearest"
        if level == "nearest":
            self.add_nearest(vectors)
        elif level == "least_inflation":
            self.add_least_inflation(vectors)
        elif level == "merge":
            self.add_merge(vectors)
        elif level == "doubling":
            self.add_doubling(vectors)
        else:
            raise ValueError(f"Unknown insertion_level {level!r}; expected one of {_INSERTION_LEVELS}")

    def add_nearest(self, vectors: np.ndarray) -> None:
        """Method 1 (baseline): absorb each vector into the nearest existing
        ball by raw distance, inflating it if needed. O(n*k*d), no extra
        state. Cheap and predictable, but nearest != cheapest to grow: a
        far-but-huge ball may need no growth at all while the nearest one is
        a tight singleton about to blow up by a factor of dist**d. Bootstraps
        via construct() when there's nothing to absorb into yet."""
        if self.count_balls == 0:
            self.construct(vectors)
            return
        X = np.ascontiguousarray(vectors, dtype=np.float32)
        d2, labels = self.index.search(X, self.count_balls)
        dist = np.sqrt(np.maximum(d2, 0.0))
        surface = dist - self.radii[labels]

        cost = np.where(surface <= 0, -np.inf, dist)
        col = cost.argmin(axis=1)
        row = np.arange(len(X))
        ball = labels[row, col]
        np.maximum.at(self.radii, ball, dist[row, col])

    def add_least_inflation(self, vectors: np.ndarray) -> None:
        """Method 2: same O(n*k*d) search as add_nearest, but absorb into
        whichever ball minimizes the actual objective delta
        max(r_i, dist_i)**d - r_i**d, not whichever is nearest. Containment
        (delta=0) is preferred for free, and a far-but-huge ball can beat a
        near-but-tiny one whenever growing the tiny one would cost more
        volume -- strictly better covering volume than add_nearest at
        identical asymptotics and no new memory.

        Computed in log space, scaled by each point's own largest candidate
        radius: r**d alone overflows float64 well inside float32's range at
        d ~ 384, but the scaled terms are always in (0, 1] so they safely
        underflow to 0 instead of overflowing, and dividing every candidate
        by the same positive per-point scale preserves the argmin exactly."""
        if self.count_balls == 0:
            self.construct(vectors)
            return
        X = np.ascontiguousarray(vectors, dtype=np.float32)
        d2, labels = self.index.search(X, self.count_balls)
        dist = np.sqrt(np.maximum(d2, 0.0)).astype(np.float64)
        r = self.radii[labels].astype(np.float64)
        r_new = np.maximum(r, dist)

        M = r_new.max(axis=1, keepdims=True)
        safe_M = np.where(M > 0, M, 1.0)
        with np.errstate(divide="ignore"):
            scaled_new = np.exp(self.dim * np.log(np.maximum(r_new, 0.0) / safe_M))
            scaled_old = np.exp(self.dim * np.log(np.maximum(r, 0.0) / safe_M))
        delta = np.where(M > 0, scaled_new - scaled_old, 0.0)

        col = delta.argmin(axis=1)
        row = np.arange(len(X))
        ball = labels[row, col]
        np.maximum.at(self.radii, ball, dist[row, col].astype(np.float32))

    @staticmethod
    def _merge_two_balls(c1: np.ndarray, r1: float, c2: np.ndarray, r2: float) -> tuple:
        """Smallest enclosing ball of two balls: whichever contains the
        other if one does, else centered on the line between them with
        radius (dist + r1 + r2) / 2."""
        diff = c2 - c1
        dist = float(np.sqrt(max(float(np.dot(diff, diff)), 0.0)))
        if dist + min(r1, r2) <= max(r1, r2):
            return (c1, r1) if r1 >= r2 else (c2, r2)
        new_r = 0.5 * (dist + r1 + r2)
        new_c = c1 if dist < 1e-12 else c1 + ((new_r - r1) / dist) * diff
        return new_c, new_r

    def _cheapest_pair(self, centers: np.ndarray, radii: np.ndarray) -> tuple:
        """argmin over all pairs (i, j) of the merge objective delta
        r_merged**d - r_i**d - r_j**d, log-space scaled the same way as
        add_least_inflation (shared M = max radius among these centers)."""
        n = len(centers)
        ci, cj = centers[:, None, :], centers[None, :, :]
        ri, rj = radii[:, None], radii[None, :]
        diff = ci - cj
        dist = np.sqrt(np.maximum(np.sum(diff * diff, axis=-1), 0.0))
        contains = dist + np.minimum(ri, rj) <= np.maximum(ri, rj)
        merged = 0.5 * (dist + ri + rj)
        r_merged = np.where(contains, np.maximum(ri, rj), merged)

        # M must cover r_merged too, not just the existing radii -- a merged
        # ball can be larger than every current ball, and scaling by a
        # smaller M would let (r_merged/M)**d exceed 1 and overflow exp().
        M = float(max(radii.max(), r_merged.max())) if n else 0.0
        safe_M = M if M > 0 else 1.0
        with np.errstate(divide="ignore"):
            scaled_i = np.exp(self.dim * np.log(np.maximum(ri, 0.0) / safe_M))
            scaled_j = np.exp(self.dim * np.log(np.maximum(rj, 0.0) / safe_M))
            scaled_m = np.exp(self.dim * np.log(np.maximum(r_merged, 0.0) / safe_M))
        delta = (scaled_m - scaled_i - scaled_j) if M > 0 else np.zeros((n, n))

        idx = np.arange(n)
        delta[idx, idx] = np.inf  # exclude self-pairs
        i, j = np.unravel_index(np.argmin(delta), delta.shape)
        return int(i), int(j)

    def add_merge(self, vectors: np.ndarray) -> None:
        """Method 3: insert each vector as its own zero-radius ball; once at
        capacity, evict by merging whichever pair of balls -- existing-
        existing or the new singleton against an existing ball -- grows the
        covering objective least. The choice is global: a point landing in
        genuinely new territory gets to found a ball rather than deforming
        one, and redundancy gets given up wherever it's cheapest, not
        wherever the new point happened to land. Under distribution shift
        that difference is large.

        Sequential per point (each merge changes the ball set the next
        point's decision depends on, unlike add_nearest/add_least_inflation
        which can absorb a whole batch in one vectorized pass). Recomputes
        the full k x k merge-cost matrix on every insertion at capacity --
        O(k**2*d) per point, correct but not the O(k*d)-amortized version an
        incremental row/column-cached heap would give; that cache is a
        natural follow-up if this becomes a throughput bottleneck at large k."""
        if self.count_balls == 0:
            self.construct(vectors)
            return
        X = np.ascontiguousarray(vectors, dtype=np.float32)
        for x in X:
            self._merge_insert_one(x)

    def _merge_insert_one(self, x: np.ndarray) -> None:
        centers = self.index.reconstruct_n(0, self.count_balls).astype(np.float64)
        radii = self.radii[: self.count_balls].astype(np.float64)
        centers = np.vstack([centers, x.astype(np.float64)[None, :]])
        radii = np.append(radii, 0.0)

        if len(centers) > self.max_count_balls:
            i, j = self._cheapest_pair(centers, radii)
            c_new, r_new = self._merge_two_balls(centers[i], radii[i], centers[j], radii[j])
            keep = [idx for idx in range(len(centers)) if idx not in (i, j)]
            centers = np.vstack([centers[keep], c_new[None, :]])
            radii = np.append(radii[keep], r_new)

        self._rebuild_from_centers(centers, radii)

    def add_doubling(self, vectors: np.ndarray) -> None:
        """Method 4: Charikar-Chekuri-Feder-Motwani incremental k-center.
        Maintains a working threshold self._doubling_R; a point farther than
        R from every ball's surface becomes a new ball, and once balls
        exceed the budget, R doubles and any two balls within 2R of each
        other are greedily merged until back at budget. The textbook
        8-approximation bound is about the *maximum* radius, the wrong
        objective for this filter's FPR (which depends on sum of r**d) --
        useful mainly if a certificate matters (it would pair naturally with
        a Gonzalez-style lower bound), otherwise add_merge usually wins
        empirically on the actual objective.

        Unlike the textbook version (a uniform radius R for every ball,
        correct only up to the asymptotic approximation proof), every merge
        here uses the exact smallest enclosing ball of the two source balls
        (_merge_two_balls, same geometry as add_merge) rather than assuming
        the uniform-R bound directly -- R only drives *when* to add a new
        ball vs. merge, not what radius gets stored, so the covering
        invariant (FNR=0) holds unconditionally rather than only
        asymptotically. Cheap in the common case (a point within R of some
        ball: O(k*d), no restructuring), with occasional O(k**2*d)
        doubling/merge phases."""
        if self.count_balls == 0:
            self.construct(vectors)
            self._doubling_R = self._initial_doubling_R()
            return
        X = np.ascontiguousarray(vectors, dtype=np.float32)
        for x in X:
            self._doubling_insert_one(x)

    def _initial_doubling_R(self) -> float:
        """R_0 = the smallest pairwise distance among the bootstrap centers
        (the classic CCFM initialization), not just the largest bootstrap
        radius -- construct()'s trivial n<=k branch leaves every radius at
        0, and starting R near 0 in a high-dimensional space where every
        pairwise distance is large forces many doublings (each only 2x) on
        the very first insertion, overshooting the data's real scale and
        collapsing far more balls than necessary in one "big bang" merge."""
        if self.count_balls <= 1:
            return max(float(self.radii[: self.count_balls].max()) if self.count_balls else 0.0, 1e-12)
        centers = self.index.reconstruct_n(0, self.count_balls).astype(np.float64)
        diff = centers[:, None, :] - centers[None, :, :]
        dist = np.sqrt(np.maximum(np.sum(diff * diff, axis=-1), 0.0))
        idx = np.arange(self.count_balls)
        dist[idx, idx] = np.inf
        return max(float(dist.min()), float(self.radii[: self.count_balls].max()), 1e-12)

    def _doubling_insert_one(self, x: np.ndarray) -> None:
        if self._doubling_R <= 0:
            # Lazily initialized here too, not just in add_doubling()'s own
            # bootstrap branch: a filter constructed via construct() directly
            # (not add()) reaches add_doubling() with count_balls already >0,
            # skipping that branch entirely.
            self._doubling_R = self._initial_doubling_R()

        centers = self.index.reconstruct_n(0, self.count_balls).astype(np.float64)
        radii = self.radii[: self.count_balls].astype(np.float64)
        x64 = x.astype(np.float64)

        dist = np.sqrt(np.maximum(np.sum((centers - x64) ** 2, axis=1), 0.0))
        surface = dist - radii
        nearest = int(np.argmin(surface))

        if surface[nearest] <= self._doubling_R:
            radii[nearest] = max(radii[nearest], dist[nearest])
            self._rebuild_from_centers(centers, radii)
            return

        centers = np.vstack([centers, x64[None, :]])
        radii = np.append(radii, 0.0)

        if len(centers) > self.max_count_balls:
            while len(centers) > self.max_count_balls:
                self._doubling_R *= 2.0
                centers, radii = self._doubling_merge_within(
                    centers, radii, 2.0 * self._doubling_R, self.max_count_balls
                )

        self._rebuild_from_centers(centers, radii)

    def _doubling_merge_within(
        self, centers: np.ndarray, radii: np.ndarray, threshold: float, target_count: int
    ) -> tuple:
        """Greedily merge the globally nearest pair of centers (exact
        smallest-enclosing-ball merge) as long as they're within `threshold`
        of each other, stopping as soon as target_count is reached -- not
        once no more qualifying pairs exist. Every pair being within
        `threshold` after a couple of doublings is common (threshold grows
        multiplicatively, so it can jump from "too small to merge anything"
        to "bigger than the whole point cloud" in very few steps); merging
        every such pair regardless of target_count collapsed the entire
        cover into 1-2 balls on the first doubling in testing, instead of
        the intended "just enough to get back under budget"."""
        while len(centers) > target_count:
            n = len(centers)
            diff = centers[:, None, :] - centers[None, :, :]
            dist = np.sqrt(np.maximum(np.sum(diff * diff, axis=-1), 0.0))
            idx = np.arange(n)
            dist[idx, idx] = np.inf
            i, j = np.unravel_index(np.argmin(dist), dist.shape)
            if dist[i, j] > threshold:
                break
            c_new, r_new = self._merge_two_balls(centers[i], radii[i], centers[j], radii[j])
            keep = [k for k in range(n) if k not in (i, j)]
            centers = np.vstack([centers[keep], c_new[None, :]])
            radii = np.append(radii[keep], r_new)
        return centers, radii 
