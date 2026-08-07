import time
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
from scipy.special import logsumexp

from filters.filter import Filter
from hyperspheres_volume import estimate_hypersphere_volume


class GonzalezMEBFilter(Filter):
    """
    Duplicate filter that covers the input vectors with a bounded number of
    hyperspheres and answers "is anything indexed within tau of this query?"
    without storing the vectors (tau is supplied at query time, not baked in).

    Construction is a cumulative, ablatable ladder of up to 5 levels
    (`set_complexity`, default 4), each one refining the cover produced by the
    previous level while never breaking the covering invariant (every indexed
    vector lies within its ball's radius):

      1. Initial cover: k-means++ seeding (probability proportional to squared
         distance from the nearest existing center -- far less sensitive to
         outliers than farthest-point/Gonzalez seeding) + a few Lloyd
         iterations, then each center is recomputed as the exact centroid of
         its final assignment and the radius is the exact max distance to any
         member. Cheapest level; O(n*k*d) for the Lloyd/assignment passes.
      2. Ball refinement: each cluster's ball is replaced by its Badoiu-
         Clarkson coreset MEB, vectors are reassigned to the nearest refined
         center, and the process repeats until log J = log sum(r_i^d) stops
         decreasing (or a small iteration cap is hit). Costs a small constant
         multiple of level 1.
      3. Boundary refinement: points near a cluster boundary (close second-
         nearest center, or realizing their ball's own radius) are shortlisted
         each iteration by a cheap distance-only bound, and only that bounded
         batch pays for real MEB recomputes to test whether migrating the
         point to its second-nearest ball strictly lowers log J. Coverage is
         preserved because both affected balls are always exact MEBs of their
         true (post-move) member sets.
      4. Radius tightening: one more exact max-distance pass per current
         assignment, squeezing out any slack left by the approximate MEB
         steps. Cheap and always safe.
      5. Feedback-driven refinement: opt-in, driven by externally verified
         false positives supplied after construction via
         `set_feedback_policy` + `refine_with_feedback` (there is nothing to
         do for this level *inside* `construct()` itself, since the feedback
         set doesn't exist yet -- level 5 only forces construction to retain
         the vectors/assignment that splitting needs).

    Streaming updates (`add`/`insert`) insert a zero-radius singleton ball and,
    if that exceeds `max_spheres`, merge the pair whose combined footprint
    grows least (in log-space, since r^d overflows/underflows float64 well
    inside float32's dynamic range at d ~ 384). An incremental sign/log-
    magnitude cost cache means only the merged ball's row/column needs
    recomputing per insertion, not the full k^2 pair matrix.

    Storage: centers/radii/counts live in a single preallocated buffer
    (capacity grows by amortized doubling up to max_spheres), so add()/
    remove() never reallocate or copy the whole structure. `get_bytesize()`
    covers only that buffer; construction state retained for levels >= 3
    (the original vectors + assignment, needed by boundary/feedback
    refinement) is reported separately by `get_construction_bytesize()`.
    """

    _BYTES_PER_FLOAT32 = 4
    _MEB_CORESET_ITERATIONS = 30
    _MEB_REFINE_ITERATIONS = 200
    _LLOYD_ITERATIONS = 10
    _BALL_REFINE_ITERATIONS = 10
    _BOUNDARY_REFINE_ITERATIONS = 5
    _BOUNDARY_BATCH_SIZE = 256
    _BOUNDARY_MARGIN = 0.15
    _KMEANS2_ITERATIONS = 10
    _CHUNK_SIZE = 20_000
    _LOG_J_EPS = 1e-9
    _RADIUS_SAFETY_REL = 1e-6
    _RADIUS_SAFETY_ABS = 1e-6
    _DEFAULT_SEED = 42
    _DEFAULT_COMPLEXITY = 4
    _FEEDBACK_MAX_CANDIDATE_BALLS = 50
    _FEEDBACK_MAX_MERGE_CANDIDATES = 20
    _LEVEL_NAMES = {1: "initial_cover", 2: "ball_refine", 3: "boundary_refine", 4: "radius_tighten"}
    MERGE_CRITERIA = ("log_space", "r_pow_d", "sum_radii", "min_radius_increase")

    def __init__(self, dim: int, distance_threshold: float, max_memory: Optional[int] = None,
                 max_spheres: Optional[int] = None, seed: int = _DEFAULT_SEED,
                 merge_criterion: str = "log_space"):
        super().__init__(dim, lsh=None, max_memory=max_memory)
        self.distance_threshold = distance_threshold
        self.max_spheres = (max_spheres if max_spheres is not None
                            else self._calculate_max_spheres(self.max_memory))
        self._seed = seed
        if merge_criterion not in self.MERGE_CRITERIA:
            raise ValueError(f"Unknown merge_criterion {merge_criterion!r}; expected one of {self.MERGE_CRITERIA}")
        self.merge_criterion = merge_criterion
        self._constructed = False
        self._complexity_level = self._DEFAULT_COMPLEXITY

        self._center_buf = np.empty((0, dim), dtype=np.float32)
        self._radius_buf = np.empty((0,), dtype=np.float32)
        self._count_buf = np.empty((0,), dtype=np.int32)
        self._capacity = 0
        self._size = 0

        # construction state (vectors + assignment), retained for level >= 3
        self._retain_assignments = False
        self._X: Optional[np.ndarray] = None
        self._assignment: Optional[np.ndarray] = None

        # feedback policy (level 5)
        self._feedback_mode = "reserved"
        self._feedback_reserved_fraction = 0.10
        self._feedback_margin = 0.0
        self._feedback_max_iterations = 100

        # instrumentation
        self._stage_stats: List[Dict[str, Any]] = []
        self._feedback_stats: Optional[Dict[str, Any]] = None

        # incremental merge-cost cache for streaming add()
        self._merge_bucket: Optional[np.ndarray] = None
        self._merge_key: Optional[np.ndarray] = None
        self._merge_cache_valid = False
        self._merge_cache_size = 0

    # ------------------------------------------------------------------ properties

    @property
    def all_spheres(self) -> np.ndarray:
        return self._center_buf[:self._size]

    @property
    def all_radiuses(self) -> np.ndarray:
        return self._radius_buf[:self._size]

    @property
    def sphere_counts(self) -> np.ndarray:
        return self._count_buf[:self._size]

    # ------------------------------------------------------------------ complexity / feedback policy

    def set_complexity(self, level: int) -> None:
        if level not in (1, 2, 3, 4, 5):
            raise ValueError(f"complexity level must be in 1..5, got {level}")
        if self._constructed:
            self._constructed = False
        self._complexity_level = level

    def set_feedback_policy(self, mode: str = "reserved", reserved_fraction: float = 0.10,
                            margin: float = 0.0, max_iterations: int = 100) -> None:
        if mode not in ("reserved", "split_merge"):
            raise ValueError(f"Unknown feedback mode {mode!r}; expected 'reserved' or 'split_merge'")
        self._feedback_mode = mode
        self._feedback_reserved_fraction = reserved_fraction
        self._feedback_margin = margin
        self._feedback_max_iterations = max_iterations

    # ------------------------------------------------------------------ sizing / buffers

    def _calculate_max_spheres(self, max_memory: int, gamma: float = 4.0) -> int:
        bytes_per_sphere = (self.dim + 1) * self._BYTES_PER_FLOAT32
        return max(1, int(max_memory // (bytes_per_sphere * gamma)))

    def _reset_buffers(self, capacity: int) -> None:
        self._center_buf = np.empty((capacity, self.dim), dtype=np.float32)
        self._radius_buf = np.empty(capacity, dtype=np.float32)
        self._count_buf = np.empty(capacity, dtype=np.int32)
        self._capacity = capacity
        self._size = 0
        self._merge_bucket = None
        self._merge_key = None
        self._merge_cache_valid = False
        self._merge_cache_size = 0

    def _grow_capacity(self) -> None:
        # Allowed to grow one past max_spheres: add() always inserts a singleton
        # first and only then merges the cheapest pair back down to max_spheres,
        # so a transient k+1 must fit.
        hard_cap = self.max_spheres + 1
        new_capacity = min(hard_cap, max(self._capacity * 2, self._capacity + 1))
        if new_capacity <= self._capacity:
            return
        center_buf = np.empty((new_capacity, self.dim), dtype=np.float32)
        radius_buf = np.empty(new_capacity, dtype=np.float32)
        count_buf = np.empty(new_capacity, dtype=np.int32)
        center_buf[:self._size] = self._center_buf[:self._size]
        radius_buf[:self._size] = self._radius_buf[:self._size]
        count_buf[:self._size] = self._count_buf[:self._size]
        self._center_buf, self._radius_buf, self._count_buf = center_buf, radius_buf, count_buf
        if self._merge_bucket is not None:
            merge_bucket = np.full((new_capacity, new_capacity), 2, dtype=np.int8)
            merge_key = np.full((new_capacity, new_capacity), np.inf, dtype=np.float64)
            merge_bucket[:self._capacity, :self._capacity] = self._merge_bucket
            merge_key[:self._capacity, :self._capacity] = self._merge_key
            self._merge_bucket, self._merge_key = merge_bucket, merge_key
        self._capacity = new_capacity

    def _append_sphere(self, center: np.ndarray, radius: float, count: int) -> None:
        if self._size == self._capacity:
            self._grow_capacity()
            if self._size == self._capacity:
                return  # at hard capacity already, nothing to grow into
        self._center_buf[self._size] = center
        self._radius_buf[self._size] = radius
        self._count_buf[self._size] = count
        self._size += 1

    def _compact(self) -> None:
        """Swap out zero-count spheres against the tail, in place."""
        i = 0
        n = self._size
        while i < n:
            if self._count_buf[i] == 0:
                n -= 1
                if i != n:
                    self._center_buf[i] = self._center_buf[n]
                    self._radius_buf[i] = self._radius_buf[n]
                    self._count_buf[i] = self._count_buf[n]
            else:
                i += 1
        self._size = n
        self._merge_cache_valid = False

    # ------------------------------------------------------------------ numerics (log-space)

    def _safe_radius(self, r: float) -> float:
        """
        Inflate a freshly computed radius by a tiny margin before it is cast to
        float32 for storage. Query-time containment is tested with the "expand"
        distance formula (||x||^2 - 2 x.c + ||c||^2, needed for BLAS-batched n x k
        matmuls), which suffers catastrophic cancellation for points close to
        their center -- so a radius computed via a direct sqrt(sum((x-c)^2)) can
        come out a few ULPs *larger* than what the expand formula later measures
        for that exact same point, silently breaking the zero-false-negative
        invariant at tau=0. This margin is many orders of magnitude below any
        realistic distance_threshold, so it never affects observable recall.
        """
        r = float(r)
        return r * (1.0 + self._RADIUS_SAFETY_REL) + self._RADIUS_SAFETY_ABS

    def _log_J(self, radii: np.ndarray) -> float:
        """log sum_i r_i^d, safe for d ~ hundreds where r^d over/underflows float64."""
        r = np.asarray(radii, dtype=np.float64)
        if r.size == 0:
            return -np.inf
        with np.errstate(divide='ignore'):
            logr = np.log(r)
        return float(logsumexp(self.dim * logr))

    def _log_J_tau(self, radii: np.ndarray, tau: float) -> float:
        r = np.asarray(radii, dtype=np.float64) + float(tau)
        if r.size == 0:
            return -np.inf
        with np.errstate(divide='ignore'):
            logr = np.log(r)
        return float(logsumexp(self.dim * logr))

    def _pairwise_merge_keys(self, ci: np.ndarray, ri: np.ndarray,
                             cj: np.ndarray, rj: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Vectorized (bucket, key, merged_radius) for merging ball(s) (ci, ri) with
        ball(s) (cj, rj). bucket 0 = beneficial merge (cost < 0), key ascending
        = most beneficial first; bucket 1 = neutral/harmful, key ascending =
        least harmful first. Cost function is pluggable via self.merge_criterion
        (see MERGE_CRITERIA and the _merge_cost_* helpers below); geometry
        (r_merged) is identical across criteria.
        """
        diff = ci - cj
        delta = np.sqrt(np.maximum(np.sum(diff * diff, axis=-1), 0.0))
        contain_j_in_i = delta + rj <= ri
        contain_i_in_j = delta + ri <= rj
        r_general = (delta + ri + rj) / 2.0
        r_merged = np.where(contain_i_in_j, rj, np.where(contain_j_in_i, ri, r_general))

        if self.merge_criterion == "log_space":
            bucket, key = self._merge_cost_log_space(ri, rj, r_merged)
        elif self.merge_criterion == "r_pow_d":
            bucket, key = self._merge_cost_r_pow_d(ri, rj, r_merged)
        elif self.merge_criterion == "sum_radii":
            bucket, key = self._merge_cost_sum_radii(ri, rj, r_merged)
        elif self.merge_criterion == "min_radius_increase":
            bucket, key = self._merge_cost_min_radius_increase(ri, rj, r_merged)
        else:
            raise ValueError(f"Unknown merge_criterion {self.merge_criterion!r}")

        return bucket, key, r_merged

    def _merge_cost_log_space(self, ri: np.ndarray, rj: np.ndarray,
                              r_merged: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Delta = M^d * (a_ij - a_i - a_j) with M = max(r_i, r_j, r_ij) and
        a_x = exp(d * log(r_x / M)) (safe underflow to 0); never materializes r^d."""
        M = np.maximum(np.maximum(ri, rj), r_merged)

        def a_of(r):
            # r<=0 (zero-radius ball) substitutes M for r, not 1.0: numpy still
            # eagerly evaluates the discarded branch, and 1.0 >> M would overflow
            # exp(); M makes it exp(0) = 1 instead, always safe.
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                safe_M = np.where(M > 0, M, 1.0)
                safe_r = np.where(r > 0, r, safe_M)
                ratio = np.where((r > 0) & (M > 0), np.exp(self.dim * (np.log(safe_r) - np.log(safe_M))), 0.0)
            return ratio

        a_i, a_j, a_ij = a_of(ri), a_of(rj), a_of(r_merged)
        diff_a = a_ij - a_i - a_j
        sign = np.sign(diff_a)
        with np.errstate(divide='ignore'):
            log_abs = self.dim * np.log(np.where(M > 0, M, 1.0)) + np.log(np.where(diff_a != 0, np.abs(diff_a), 1.0))
        log_abs = np.where(diff_a != 0, log_abs, -np.inf)
        bucket = np.where(sign < 0, 0, 1).astype(np.int8)
        key = np.where(sign < 0, -log_abs, log_abs)
        return bucket, key

    def _merge_cost_r_pow_d(self, ri: np.ndarray, rj: np.ndarray,
                            r_merged: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Delta = r_merged^d - r_i^d - r_j^d, computed directly (no log-domain
        protection) -- overflows to +-inf well within realistic radii at d ~ 384,
        intentionally, as a naive baseline."""
        with np.errstate(over='ignore', invalid='ignore'):
            delta = r_merged ** self.dim - ri ** self.dim - rj ** self.dim
        delta = np.where(np.isnan(delta), np.inf, delta)
        sign = np.sign(delta)
        bucket = np.where(sign < 0, 0, 1).astype(np.int8)
        key = np.where(sign < 0, -delta, delta)
        return bucket, key

    def _merge_cost_sum_radii(self, ri: np.ndarray, rj: np.ndarray,
                              r_merged: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Delta = r_merged - r_i - r_j under the linear objective J = sum r_i."""
        delta = r_merged - ri - rj
        sign = np.sign(delta)
        bucket = np.where(sign < 0, 0, 1).astype(np.int8)
        key = np.where(sign < 0, -delta, delta)
        return bucket, key

    def _merge_cost_min_radius_increase(self, ri: np.ndarray, rj: np.ndarray,
                                        r_merged: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Cost = growth of the merged radius past the larger constituent ball;
        always >= 0 by construction of r_merged, so bucket is always 1 (never
        "strictly beneficial") -- smallest growth first."""
        delta = r_merged - np.maximum(ri, rj)
        bucket = np.ones_like(delta, dtype=np.int8)
        key = delta
        return bucket, key

    def _all_pairs_merge_keys(self, centers: np.ndarray, radii: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        c64 = centers.astype(np.float64)
        r64 = radii.astype(np.float64)
        bucket, key, _ = self._pairwise_merge_keys(c64[:, None, :], r64[:, None], c64[None, :, :], r64[None, :])
        n = len(centers)
        idx = np.arange(n)
        bucket[idx, idx] = 2
        key[idx, idx] = np.inf
        return bucket, key

    @staticmethod
    def _merged_ball(ci: np.ndarray, ri: float, cj: np.ndarray, rj: float) -> Tuple[np.ndarray, float]:
        ci64 = np.asarray(ci, dtype=np.float64)
        cj64 = np.asarray(cj, dtype=np.float64)
        ri = float(ri)
        rj = float(rj)
        delta = float(np.sqrt(np.maximum(np.sum((ci64 - cj64) ** 2), 0.0)))
        if delta + ri <= rj:
            return cj64, rj
        if delta + rj <= ri:
            return ci64, ri
        r_new = (delta + ri + rj) / 2.0
        c_new = ci64 + ((r_new - ri) / delta) * (cj64 - ci64)
        return c_new, r_new

    # ------------------------------------------------------------------ chunked distances

    def _sq_dists_to_centers(self, vectors: np.ndarray, sq_norms: np.ndarray, centers: np.ndarray) -> np.ndarray:
        # Accumulate in float64: with threshold-free (possibly zero-slack) radii,
        # float32 accumulation over d ~ hundreds of dims can push a member point
        # just outside its own ball's stored radius.
        centers64 = centers.astype(np.float64)
        vectors64 = vectors.astype(np.float64)
        sq_norms64 = np.asarray(sq_norms, dtype=np.float64)
        c_sq = np.einsum('ij,ij->i', centers64, centers64)
        cross = vectors64 @ centers64.T
        return sq_norms64[:, None] - 2.0 * cross + c_sq[None, :]

    def _chunked_assign(self, vectors: np.ndarray, sq_norms: np.ndarray, centers: np.ndarray,
                        chunk: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Nearest-center assignment, chunked so the n x k distance matrix is bounded."""
        chunk = chunk or self._CHUNK_SIZE
        n = len(vectors)
        assignment = np.empty(n, dtype=np.int64)
        nearest_sq = np.empty(n, dtype=np.float64)
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            sq_d = self._sq_dists_to_centers(vectors[start:end], sq_norms[start:end], centers)
            a = np.argmin(sq_d, axis=1)
            assignment[start:end] = a
            nearest_sq[start:end] = sq_d[np.arange(end - start), a]
        return assignment, nearest_sq

    def _chunked_top2(self, vectors: np.ndarray, sq_norms: np.ndarray, centers: np.ndarray,
                      chunk: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Nearest + second-nearest center index/distance per row, chunked over n."""
        chunk = chunk or self._CHUNK_SIZE
        n = len(vectors)
        k = len(centers)
        nearest = np.empty(n, dtype=np.int64)
        second = np.empty(n, dtype=np.int64)
        d_nearest = np.empty(n, dtype=np.float64)
        d_second = np.empty(n, dtype=np.float64)
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            sq_d = self._sq_dists_to_centers(vectors[start:end], sq_norms[start:end], centers)
            if k >= 2:
                part = np.argpartition(sq_d, 1, axis=1)[:, :2]
                row_vals = np.take_along_axis(sq_d, part, axis=1)
                order = np.argsort(row_vals, axis=1)
                idx_sorted = np.take_along_axis(part, order, axis=1)
                val_sorted = np.take_along_axis(row_vals, order, axis=1)
                nearest[start:end] = idx_sorted[:, 0]
                second[start:end] = idx_sorted[:, 1]
                d_nearest[start:end] = np.sqrt(np.maximum(val_sorted[:, 0], 0.0))
                d_second[start:end] = np.sqrt(np.maximum(val_sorted[:, 1], 0.0))
            else:
                nearest[start:end] = 0
                second[start:end] = 0
                d_nearest[start:end] = np.sqrt(np.maximum(sq_d[:, 0], 0.0))
                d_second[start:end] = np.inf
        return nearest, second, d_nearest, d_second

    @staticmethod
    def _groups_from_assignment(assignment: np.ndarray, num_labels: int) -> List[np.ndarray]:
        """Split point indices into per-label groups; assumes labels are a
        contiguous 0..num_labels-1 range with no empty labels (true right after
        np.unique(..., return_inverse=True))."""
        if num_labels == 0:
            return []
        order = np.argsort(assignment, kind='stable')
        sorted_assignment = assignment[order]
        boundaries = np.flatnonzero(np.diff(sorted_assignment)) + 1
        return np.split(order, boundaries)

    # ------------------------------------------------------------------ clustering

    def _kmeanspp_seed(self, vectors: np.ndarray, sq_norms: np.ndarray, k: int,
                       stop_sq: float, rng: np.random.Generator) -> Tuple[np.ndarray, bool]:
        n = len(vectors)
        mean = vectors.mean(axis=0)
        first = int((sq_norms - 2.0 * (vectors @ mean)).argmax())

        idx = [first]
        c = vectors[first]
        nearest_sq = np.maximum(sq_norms - 2.0 * (vectors @ c) + c @ c, 0.0)

        for _ in range(1, k):
            if float(nearest_sq.max()) <= stop_sq:
                return np.array(idx, dtype=np.int64), True
            total = float(nearest_sq.sum())
            if total <= 0.0:
                break
            probs = nearest_sq.astype(np.float64) / total
            probs /= probs.sum()  # guard against float64 sum drifting off 1.0
            next_idx = int(rng.choice(n, p=probs))
            idx.append(next_idx)
            c = vectors[next_idx]
            new_sq = np.maximum(sq_norms - 2.0 * (vectors @ c) + c @ c, 0.0)
            np.minimum(nearest_sq, new_sq, out=nearest_sq)

        return np.array(idx, dtype=np.int64), False

    def _lloyd_refine(self, vectors: np.ndarray, sq_norms: np.ndarray, centers: np.ndarray) -> np.ndarray:
        k = len(centers)

        for _ in range(self._LLOYD_ITERATIONS):
            assignment, nearest_sq = self._chunked_assign(vectors, sq_norms, centers)

            new_centers = np.empty_like(centers)
            farthest_order = None
            for i in range(k):
                members = assignment == i
                if np.any(members):
                    new_centers[i] = vectors[members].mean(axis=0)
                else:
                    if farthest_order is None:
                        farthest_order = list(np.argsort(-nearest_sq))
                    new_centers[i] = vectors[farthest_order.pop(0)]

            converged = np.allclose(new_centers, centers, atol=1e-5)
            centers = new_centers
            if converged:
                break

        return centers

    def _minimum_enclosing_ball(self, points: np.ndarray, sq_norms: np.ndarray) -> Tuple[np.ndarray, float]:
        if len(points) == 1:
            return points[0].astype(np.float32), 0.0

        center = points.mean(axis=0, dtype=np.float64)
        coreset_idx = set()
        for t in range(1, self._MEB_CORESET_ITERATIONS + 1):
            sq_d = sq_norms - 2.0 * (points @ center) + center @ center
            far = int(sq_d.argmax())
            coreset_idx.add(far)
            center += (points[far] - center) / (t + 1.0)

        coreset_idx = list(coreset_idx)
        coreset = points[coreset_idx]
        coreset_sq = sq_norms[coreset_idx]
        for t in range(1, self._MEB_REFINE_ITERATIONS + 1):
            sq_d = coreset_sq - 2.0 * (coreset @ center) + center @ center
            far = int(sq_d.argmax())
            center += (coreset[far] - center) / (t + self._MEB_CORESET_ITERATIONS + 1.0)

        center32 = center.astype(np.float32)
        center64 = center32.astype(np.float64)  # radius must match the center as actually stored
        # Direct subtraction (not the ||x||^2 - 2x.c + ||c||^2 expansion): avoids
        # catastrophic cancellation for points close to the center, which is
        # exactly the regime the final radius has to be exact in.
        points64 = points.astype(np.float64)
        diff = points64 - center64
        d = np.sqrt(np.maximum(np.einsum('ij,ij->i', diff, diff), 0.0))
        return center32, self._safe_radius(d.max())

    def _log_volume_bounds(self) -> Tuple[float, float]:
        if self._size == 0:
            return -np.inf, -np.inf
        return estimate_hypersphere_volume(self.all_spheres, self.all_radiuses, return_log=True)

    # ------------------------------------------------------------------ level 1: initial cover

    def _level1_initial_cover(self, vectors: np.ndarray, sq_norms: np.ndarray,
                              k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        stop_sq = float(self.distance_threshold) ** 2
        rng = np.random.default_rng(self._seed)

        seed_idx, early_stop = self._kmeanspp_seed(vectors, sq_norms, k, stop_sq, rng)
        centers = vectors[seed_idx].copy()
        if not early_stop:
            centers = self._lloyd_refine(vectors, sq_norms, centers)

        raw_assignment, _ = self._chunked_assign(vectors, sq_norms, centers)
        unique_labels, assignment = np.unique(raw_assignment, return_inverse=True)
        num_spheres = len(unique_labels)
        groups = self._groups_from_assignment(assignment, num_spheres)

        centers_out = np.empty((num_spheres, self.dim), dtype=np.float32)
        radii_out = np.empty(num_spheres, dtype=np.float32)
        counts_out = np.empty(num_spheres, dtype=np.int32)
        for i, idx in enumerate(groups):
            pts = vectors[idx]
            c32 = pts.mean(axis=0, dtype=np.float64).astype(np.float32)
            c = c32.astype(np.float64)  # radius must match the center as actually stored
            d = np.sqrt(np.maximum(np.einsum('ij,ij->i', pts.astype(np.float64) - c, pts.astype(np.float64) - c), 0.0))
            centers_out[i] = c32
            radii_out[i] = self._safe_radius(d.max())
            counts_out[i] = len(idx)

        return centers_out, radii_out, counts_out, assignment.astype(np.int32)

    # ------------------------------------------------------------------ level 2: ball refinement

    def _level2_ball_refine(self, vectors: np.ndarray, sq_norms: np.ndarray, centers: np.ndarray,
                            radii: np.ndarray, counts: np.ndarray,
                            assignment: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        prev_log_J = self._log_J(radii)
        accepted = 0

        for _ in range(self._BALL_REFINE_ITERATIONS):
            raw_assignment, _ = self._chunked_assign(vectors, sq_norms, centers)
            unique_labels, new_assignment = np.unique(raw_assignment, return_inverse=True)
            num_spheres = len(unique_labels)
            groups = self._groups_from_assignment(new_assignment, num_spheres)

            new_centers = np.empty((num_spheres, self.dim), dtype=np.float32)
            new_radii = np.empty(num_spheres, dtype=np.float32)
            new_counts = np.empty(num_spheres, dtype=np.int32)
            for i, idx in enumerate(groups):
                c, r = self._minimum_enclosing_ball(vectors[idx], sq_norms[idx])
                new_centers[i] = c
                new_radii[i] = r
                new_counts[i] = len(idx)

            new_log_J = self._log_J(new_radii)
            if new_log_J < prev_log_J - self._LOG_J_EPS:
                centers, radii, counts, assignment = new_centers, new_radii, new_counts, new_assignment.astype(np.int32)
                prev_log_J = new_log_J
                accepted += 1
            else:
                break

        return centers, radii, counts, assignment, accepted

    # ------------------------------------------------------------------ level 3: boundary refinement

    def _level3_boundary_refine(self, vectors: np.ndarray, sq_norms: np.ndarray, centers: np.ndarray,
                                radii: np.ndarray, counts: np.ndarray,
                                assignment: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        centers = centers.copy()
        radii = radii.copy()
        counts = counts.copy()
        assignment = assignment.copy()
        accepted_total = 0

        for _ in range(self._BOUNDARY_REFINE_ITERATIONS):
            if len(centers) < 2:
                break
            nearest, second, d_nearest, d_second = self._chunked_top2(vectors, sq_norms, centers)
            r_nearest = radii[nearest].astype(np.float64)
            r_second = radii[second].astype(np.float64)

            realizes = d_nearest >= r_nearest * (1.0 - self._BOUNDARY_MARGIN)
            margin_close = d_second <= d_nearest * (1.0 + self._BOUNDARY_MARGIN)
            eligible = realizes & margin_close & (nearest != second) & np.isfinite(d_second)
            cand_idx = np.flatnonzero(eligible)
            if cand_idx.size == 0:
                break

            shrink_bound = r_nearest[cand_idx]
            growth_lower_bound = np.maximum(0.0, d_second[cand_idx] - r_second[cand_idx])
            score = shrink_bound - growth_lower_bound
            order = np.argsort(-score)
            batch = cand_idx[order[:self._BOUNDARY_BATCH_SIZE]]

            accepted_this_iter = 0
            for x in batch:
                x = int(x)
                i = int(assignment[x])
                j = int(second[x])
                if i == j:
                    continue
                members_i = np.flatnonzero(assignment == i)
                if len(members_i) <= 1:
                    continue  # never migrate away a ball's last member here
                members_i_wo_x = members_i[members_i != x]
                new_c_i, new_r_i = self._minimum_enclosing_ball(vectors[members_i_wo_x], sq_norms[members_i_wo_x])
                if new_r_i >= radii[i] - self._LOG_J_EPS:
                    continue  # removing x doesn't actually shrink ball i; skip before the 2nd MEB

                members_j = np.flatnonzero(assignment == j)
                members_j_with_x = np.concatenate([members_j, [x]])
                new_c_j, new_r_j = self._minimum_enclosing_ball(vectors[members_j_with_x], sq_norms[members_j_with_x])

                old_log_J = self._log_J(radii)
                trial_radii = radii.copy()
                trial_radii[i] = new_r_i
                trial_radii[j] = new_r_j
                new_log_J = self._log_J(trial_radii)
                if new_log_J < old_log_J - self._LOG_J_EPS:
                    centers[i], radii[i] = new_c_i, new_r_i
                    centers[j], radii[j] = new_c_j, new_r_j
                    counts[i] -= 1
                    counts[j] += 1
                    assignment[x] = j
                    accepted_this_iter += 1
                    accepted_total += 1

            if accepted_this_iter == 0:
                break

        return centers, radii, counts, assignment, accepted_total

    # ------------------------------------------------------------------ level 4: radius tightening

    def _level4_tighten(self, vectors: np.ndarray, sq_norms: np.ndarray, centers: np.ndarray,
                        assignment: np.ndarray, num_spheres: int) -> Tuple[np.ndarray, np.ndarray]:
        radii = np.zeros(num_spheres, dtype=np.float32)
        counts = np.zeros(num_spheres, dtype=np.int32)
        groups = self._groups_from_assignment(assignment, num_spheres)
        for i, idx in enumerate(groups):
            pts = vectors[idx]
            c = centers[i].astype(np.float64)
            d = np.sqrt(np.maximum(np.einsum('ij,ij->i', pts - c, pts - c), 0.0))
            radii[i] = self._safe_radius(d.max()) if len(d) else 0.0
            counts[i] = len(idx)
        return radii, counts

    # ------------------------------------------------------------------ instrumentation

    def _record_stage(self, level: int, name: str, t_start: float, radii: np.ndarray, accepted_ops: int) -> None:
        self._stage_stats.append({
            "level": level,
            "name": name,
            "wall_time": time.perf_counter() - t_start,
            "ball_count": len(radii),
            "log_J": self._log_J(radii),
            "radii_snapshot": np.asarray(radii, dtype=np.float32).copy(),
            "accepted_ops": accepted_ops,
        })

    def get_stats(self, taus: Optional[List[float]] = None) -> Dict[str, Any]:
        taus = list(taus) if taus is not None else []
        levels = []
        for stage in self._stage_stats:
            levels.append({
                "level": stage["level"],
                "name": stage["name"],
                "wall_time": stage["wall_time"],
                "ball_count": stage["ball_count"],
                "log_J": stage["log_J"],
                "log_J_tau": {t: self._log_J_tau(stage["radii_snapshot"], t) for t in taus},
                "accepted_ops": stage["accepted_ops"],
            })
        return {"levels": levels, "feedback": self._feedback_stats}

    # ------------------------------------------------------------------ construct

    def construct(self, vectors: np.ndarray, retain_assignments: bool = False) -> Tuple[float, float]:
        self._stage_stats = []
        self._feedback_stats = None
        self._merge_cache_valid = False
        retain = bool(retain_assignments) or self._complexity_level >= 3
        levels_to_run = min(self._complexity_level, 4)

        if len(vectors) == 0 or self.max_spheres == 0:
            self._reset_buffers(0)
            vectors_arr = np.ascontiguousarray(vectors, dtype=np.float32)
            self._retain_assignments = retain
            self._X = vectors_arr if retain else None
            self._assignment = np.empty(0, dtype=np.int32) if retain else None
            for lvl in range(1, levels_to_run + 1):
                t0 = time.perf_counter()
                self._record_stage(lvl, self._LEVEL_NAMES[lvl], t0, np.empty(0, dtype=np.float32), 0)
            self._constructed = True
            return self._log_volume_bounds()

        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        n = len(vectors)
        sq_norms = np.einsum('ij,ij->i', vectors, vectors, dtype=np.float64)

        if self.max_spheres >= n:
            self._reset_buffers(n)
            self._center_buf[:n] = vectors
            self._radius_buf[:n] = 0.0
            self._count_buf[:n] = 1
            self._size = n
            assignment = np.arange(n, dtype=np.int32)
            for lvl in range(1, levels_to_run + 1):
                t0 = time.perf_counter()
                self._record_stage(lvl, self._LEVEL_NAMES[lvl], t0, self._radius_buf[:n], 0)
            self._retain_assignments = retain
            self._X = vectors if retain else None
            self._assignment = assignment if retain else None
            self._constructed = True
            return self._log_volume_bounds()

        # --- level 1: initial cover ---
        t0 = time.perf_counter()
        centers, radii, counts, assignment = self._level1_initial_cover(vectors, sq_norms, self.max_spheres)
        self._record_stage(1, "initial_cover", t0, radii, 0)

        # --- level 2: ball refinement loop ---
        if levels_to_run >= 2:
            t0 = time.perf_counter()
            centers, radii, counts, assignment, accepted = self._level2_ball_refine(
                vectors, sq_norms, centers, radii, counts, assignment)
            self._record_stage(2, "ball_refine", t0, radii, accepted)

        # --- level 3: boundary refinement ---
        if levels_to_run >= 3:
            t0 = time.perf_counter()
            centers, radii, counts, assignment, accepted = self._level3_boundary_refine(
                vectors, sq_norms, centers, radii, counts, assignment)
            self._record_stage(3, "boundary_refine", t0, radii, accepted)

        # --- level 4: radius tightening ---
        if levels_to_run >= 4:
            t0 = time.perf_counter()
            radii, counts = self._level4_tighten(vectors, sq_norms, centers, assignment, len(centers))
            self._record_stage(4, "radius_tighten", t0, radii, 0)

        num_spheres = len(centers)
        self._reset_buffers(num_spheres)
        self._center_buf[:num_spheres] = centers
        self._radius_buf[:num_spheres] = radii
        self._count_buf[:num_spheres] = counts
        self._size = num_spheres

        self._retain_assignments = retain
        self._X = vectors if retain else None
        self._assignment = assignment.astype(np.int32) if retain else None
        self._constructed = True
        return self._log_volume_bounds()

    # ------------------------------------------------------------------ query

    def query(self, vectors: np.ndarray, tau: Optional[float] = None) -> List[bool]:
        if not self._constructed:
            raise ValueError("GonzalezMEBFilter has not been constructed.")
        if tau is None:
            tau = self.distance_threshold

        if self._size == 0 or len(vectors) == 0:
            return [False] * len(vectors)

        q = np.ascontiguousarray(vectors, dtype=np.float32)
        n = len(q)
        centers = self.all_spheres
        eff_r = self.all_radiuses.astype(np.float64) + float(tau)
        q_sq_norms = np.einsum('ij,ij->i', q, q, dtype=np.float64)

        result = np.zeros(n, dtype=bool)
        for start in range(0, n, self._CHUNK_SIZE):
            end = min(start + self._CHUNK_SIZE, n)
            sq_d = self._sq_dists_to_centers(q[start:end], q_sq_norms[start:end], centers)
            result[start:end] = np.any(sq_d <= (eff_r[None, :]) ** 2, axis=1)
        return result.tolist()

    # ------------------------------------------------------------------ streaming add/remove

    def _ensure_merge_cache(self) -> None:
        if self._merge_cache_valid and self._merge_cache_size == self._size:
            return
        if self._merge_bucket is None or self._merge_bucket.shape[0] < self._capacity:
            cap = max(self._capacity, 1)
            self._merge_bucket = np.full((cap, cap), 2, dtype=np.int8)
            self._merge_key = np.full((cap, cap), np.inf, dtype=np.float64)
        n = self._size
        if n > 0:
            bucket, key = self._all_pairs_merge_keys(self.all_spheres, self.all_radiuses)
            self._merge_bucket[:n, :n] = bucket
            self._merge_key[:n, :n] = key
        self._merge_cache_valid = True
        self._merge_cache_size = n

    def _merge_cache_recompute_slot(self, idx: int) -> None:
        """Recompute the row/column of the merge-cost cache for slot `idx` only
        (O(k*d)), instead of rebuilding the whole k x k matrix."""
        n = self._size
        centers = self.all_spheres[:n].astype(np.float64)
        radii = self.all_radiuses[:n].astype(np.float64)
        ci = centers[idx]
        ri = radii[idx]
        bucket, key, _ = self._pairwise_merge_keys(ci[None, :], ri, centers, radii)
        bucket[idx] = 2
        key[idx] = np.inf
        self._merge_bucket[idx, :n] = bucket
        self._merge_key[idx, :n] = key
        self._merge_bucket[:n, idx] = bucket
        self._merge_key[:n, idx] = key
        self._merge_bucket[idx, idx] = 2
        self._merge_key[idx, idx] = np.inf

    def _merge_cache_best_pair(self) -> Tuple[int, int]:
        n = self._size
        bucket = self._merge_bucket[:n, :n]
        key = self._merge_key[:n, :n]
        mask0 = bucket == 0
        if mask0.any():
            key_eff = np.where(mask0, key, np.inf)
        else:
            key_eff = np.where(bucket == 1, key, np.inf)
        flat_idx = int(np.argmin(key_eff))
        i, j = divmod(flat_idx, n)
        return i, j

    def _apply_merge_in_place(self, i: int, j: int) -> None:
        if i > j:
            i, j = j, i
        c_new, r_new = self._merged_ball(self._center_buf[i], float(self._radius_buf[i]),
                                         self._center_buf[j], float(self._radius_buf[j]))
        count_new = int(self._count_buf[i]) + int(self._count_buf[j])
        self._center_buf[i] = c_new.astype(np.float32)
        self._radius_buf[i] = r_new
        self._count_buf[i] = count_new

        last = self._size - 1
        if j != last:
            self._center_buf[j] = self._center_buf[last]
            self._radius_buf[j] = self._radius_buf[last]
            self._count_buf[j] = self._count_buf[last]
            if self._merge_cache_valid:
                self._merge_bucket[j, :self._size] = self._merge_bucket[last, :self._size]
                self._merge_key[j, :self._size] = self._merge_key[last, :self._size]
                self._merge_bucket[:self._size, j] = self._merge_bucket[:self._size, last]
                self._merge_key[:self._size, j] = self._merge_key[:self._size, last]

        self._size -= 1
        if self._merge_cache_valid:
            self._merge_cache_size = self._size
            self._merge_cache_recompute_slot(i)
            if j != last:
                self._merge_cache_recompute_slot(j)

    def add(self, vectors: np.ndarray) -> None:
        if not self._constructed:
            self.construct(vectors)
            return
        if len(vectors) == 0:
            return
        if self.max_spheres == 0:
            return

        new_vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        for q in new_vectors:
            self._append_sphere(q, 0.0, 1)
            if self._size <= self.max_spheres:
                # cheap incremental update: new ball's row/col only, if cache is live
                if self._merge_cache_valid and self._merge_bucket is not None and self._merge_bucket.shape[0] >= self._size:
                    self._merge_cache_size = self._size
                    self._merge_cache_recompute_slot(self._size - 1)
                continue

            # over budget: merge the globally cheapest pair back down to max_spheres
            self._ensure_merge_cache()
            if not self._merge_cache_valid:
                # size didn't actually grow (hard capacity reached without room);
                # nothing meaningful to merge
                continue
            i, j = self._merge_cache_best_pair()
            self._apply_merge_in_place(i, j)

    def insert(self, vectors: np.ndarray) -> None:
        self.add(vectors)

    def remove(self, vectors: np.ndarray, tau: Optional[float] = None) -> List[bool]:
        if tau is None:
            tau = self.distance_threshold
        if self._size == 0 or len(vectors) == 0:
            return [False] * len(vectors)

        q = np.ascontiguousarray(vectors, dtype=np.float32)
        centers = self.all_spheres
        sq_d = self._sq_dists_to_centers(q, np.einsum('ij,ij->i', q, q, dtype=np.float64), centers)
        nearest = np.argmin(sq_d, axis=1)
        nearest_d = np.sqrt(np.maximum(sq_d[np.arange(len(q)), nearest], 0.0))
        in_sphere = nearest_d <= self._radius_buf[nearest].astype(np.float64) + float(tau)

        if np.any(in_sphere):
            decrements = np.bincount(nearest[in_sphere], minlength=self._size)
            self._count_buf[:self._size] -= decrements.astype(self._count_buf.dtype)
            self._compact()

        return in_sphere.tolist()

    # ------------------------------------------------------------------ memory / misc

    def get_volume(self) -> float:
        return 0.0

    def get_exact_matching_fpr(self) -> float:
        return 0.0

    def get_bytesize(self) -> int:
        if not self._constructed:
            return 0
        return self.all_spheres.nbytes + self.all_radiuses.nbytes + self.sphere_counts.nbytes

    def get_construction_bytesize(self) -> int:
        if not self._constructed or not self._retain_assignments or self._X is None or self._assignment is None:
            return 0
        return int(self._X.nbytes) + int(self._assignment.nbytes)

    # ------------------------------------------------------------------ level 5: split candidates

    def _two_means_split(self, X: np.ndarray) -> Optional[np.ndarray]:
        n = len(X)
        if n < 2:
            return None
        sqn = np.einsum('ij,ij->i', X, X, dtype=np.float64)
        i1 = int(np.argmax(sqn - 2.0 * (X @ X[0]) + sqn[0]))
        d1 = sqn - 2.0 * (X @ X[i1]) + sqn[i1]
        i2 = int(np.argmax(d1))
        centers = np.stack([X[i1], X[i2]]).astype(np.float64)
        membership = None
        for _ in range(self._KMEANS2_ITERATIONS):
            sq_d = sqn[:, None] - 2.0 * (X @ centers.T) + np.einsum('ij,ij->i', centers, centers)[None, :]
            membership = np.argmin(sq_d, axis=1)
            if len(np.unique(membership)) < 2:
                return None
            for c in range(2):
                centers[c] = X[membership == c].mean(axis=0)
        return membership.astype(np.int32)

    def _pca_split(self, X: np.ndarray) -> Optional[np.ndarray]:
        if len(X) < 2:
            return None
        mean = X.mean(axis=0)
        Xc = (X - mean).astype(np.float64)
        cov = Xc.T @ Xc
        _, vecs = np.linalg.eigh(cov)
        top = vecs[:, -1]
        proj = Xc @ top
        median = np.median(proj)
        membership = (proj > median).astype(np.int32)
        if len(np.unique(membership)) < 2:
            return None
        return membership

    def _witness_split(self, X: np.ndarray, witness: np.ndarray) -> Optional[np.ndarray]:
        d = np.linalg.norm(X.astype(np.float64) - witness.astype(np.float64), axis=1)
        median = np.median(d)
        membership = (d > median).astype(np.int32)
        if len(np.unique(membership)) < 2:
            return None
        return membership

    def _split_candidates_for_ball(self, ball_idx: int, assignment: np.ndarray, accept_col: np.ndarray,
                                   weights: np.ndarray, X_q: np.ndarray) -> List[Dict[str, Any]]:
        members_global_idx = np.flatnonzero(assignment == ball_idx)
        if len(members_global_idx) < 2:
            return []
        Xi = self._X[members_global_idx]
        memberships = []
        m2 = self._two_means_split(Xi)
        if m2 is not None:
            memberships.append(m2)
        mp = self._pca_split(Xi)
        if mp is not None:
            memberships.append(mp)
        witness_idx = np.flatnonzero(accept_col)
        if witness_idx.size > 0:
            top_witness = X_q[witness_idx[np.argmax(weights[witness_idx])]]
            mw = self._witness_split(Xi, top_witness)
            if mw is not None:
                memberships.append(mw)
        return [{"ball_idx": ball_idx, "members": members_global_idx, "membership": m} for m in memberships]

    def _apply_split_to_cover(self, centers: np.ndarray, radii: np.ndarray, counts: np.ndarray,
                              assignment: np.ndarray, ball_idx: int, members_global_idx: np.ndarray,
                              membership: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx0 = members_global_idx[membership == 0]
        idx1 = members_global_idx[membership == 1]
        sqn0 = np.einsum('ij,ij->i', self._X[idx0], self._X[idx0], dtype=np.float64)
        sqn1 = np.einsum('ij,ij->i', self._X[idx1], self._X[idx1], dtype=np.float64)
        c0, r0 = self._minimum_enclosing_ball(self._X[idx0], sqn0)
        c1, r1 = self._minimum_enclosing_ball(self._X[idx1], sqn1)

        new_centers = np.vstack([centers, c1[None, :]])
        new_radii = np.append(radii, np.float32(r1))
        new_counts = np.append(counts, np.int32(len(idx1)))
        new_centers[ball_idx] = c0
        new_radii[ball_idx] = r0
        new_counts[ball_idx] = len(idx0)

        new_assignment = assignment.copy()
        new_assignment[idx1] = len(centers)  # new slot appended at the old length
        return new_centers, new_radii, new_counts, new_assignment

    def _per_ball_accept(self, centers: np.ndarray, radii: np.ndarray, X_q: np.ndarray, q_sq_norms: np.ndarray,
                         thresholds: np.ndarray, gamma: float) -> np.ndarray:
        m = len(X_q)
        k = len(centers)
        accept = np.zeros((m, k), dtype=bool)
        if m == 0 or k == 0:
            return accept
        eff_r = radii[None, :].astype(np.float64) + gamma
        for start in range(0, m, self._CHUNK_SIZE):
            end = min(start + self._CHUNK_SIZE, m)
            sq_d = self._sq_dists_to_centers(X_q[start:end], q_sq_norms[start:end], centers)
            r_total = eff_r + thresholds[start:end, None]
            accept[start:end] = sq_d <= r_total ** 2
        return accept

    # ------------------------------------------------------------------ level 5: merge candidates

    def _apply_merge_to_cover(self, centers: np.ndarray, radii: np.ndarray, counts: np.ndarray,
                              assignment: np.ndarray, i: int, j: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if i > j:
            i, j = j, i
        c_new, r_new = self._merged_ball(centers[i], radii[i], centers[j], radii[j])

        new_centers = centers.copy()
        new_radii = radii.copy()
        new_counts = counts.copy()
        new_centers[i] = c_new.astype(np.float32)
        new_radii[i] = r_new
        new_counts[i] = int(counts[i]) + int(counts[j])

        last = len(new_centers) - 1
        new_assignment = assignment.copy()
        new_assignment[assignment == j] = i
        if j != last:
            new_centers[j] = new_centers[last]
            new_radii[j] = new_radii[last]
            new_counts[j] = new_counts[last]
            new_assignment[assignment == last] = j
        new_centers = np.delete(new_centers, last, axis=0)
        new_radii = np.delete(new_radii, last)
        new_counts = np.delete(new_counts, last)
        return new_centers, new_radii, new_counts, new_assignment

    def _top_merge_candidates(self, centers: np.ndarray, radii: np.ndarray, top_n: int) -> List[Tuple[int, int]]:
        n = len(centers)
        if n < 2:
            return []
        bucket, key = self._all_pairs_merge_keys(centers, radii)
        mask0 = bucket == 0
        key_eff = np.where(mask0, key, np.inf) if mask0.any() else np.where(bucket == 1, key, np.inf)

        iu = np.triu_indices(n, k=1)
        pair_keys = np.minimum(key_eff[iu], key_eff[(iu[1], iu[0])])
        order = np.argsort(pair_keys)
        take = min(top_n, len(order))
        result = []
        for t in range(take):
            oi = order[t]
            if not np.isfinite(pair_keys[oi]):
                break
            result.append((int(iu[0][oi]), int(iu[1][oi])))
        return result

    def _merge_down(self, centers: np.ndarray, radii: np.ndarray, counts: np.ndarray,
                    assignment: np.ndarray, target_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        while len(centers) > target_size:
            pairs = self._top_merge_candidates(centers, radii, top_n=1)
            if not pairs:
                break
            i, j = pairs[0]
            centers, radii, counts, assignment = self._apply_merge_to_cover(centers, radii, counts, assignment, i, j)
        return centers, radii, counts, assignment

    # ------------------------------------------------------------------ level 5: reserved mode

    def _best_split_candidate(self, centers, radii, counts, assignment, X_q, q_sq_norms, thresholds, weights,
                              gamma, loss_and_accept):
        accept = self._per_ball_accept(centers, radii, X_q, q_sq_norms, thresholds, gamma)
        ball_has_witness = np.flatnonzero(accept.any(axis=0))
        if ball_has_witness.size > self._FEEDBACK_MAX_CANDIDATE_BALLS:
            witness_weight = np.array([weights[accept[:, b]].sum() for b in ball_has_witness])
            keep = np.argsort(-witness_weight)[:self._FEEDBACK_MAX_CANDIDATE_BALLS]
            ball_has_witness = ball_has_witness[keep]

        base_loss, _ = loss_and_accept(centers, radii)
        best = None
        for b in ball_has_witness:
            b = int(b)
            for cand in self._split_candidates_for_ball(b, assignment, accept[:, b], weights, X_q):
                new_centers, new_radii, new_counts, new_assignment = self._apply_split_to_cover(
                    centers, radii, counts, assignment, cand["ball_idx"], cand["members"], cand["membership"])
                new_loss, _ = loss_and_accept(new_centers, new_radii)
                gain = base_loss - new_loss
                if best is None or gain > best[0]:
                    best = (gain, new_centers, new_radii, new_counts, new_assignment)
        return best

    def _feedback_reserved(self, centers, radii, counts, assignment, X_q, q_sq_norms, thresholds, weights,
                           gamma, loss_and_accept):
        k = len(centers)
        s = min(int(np.ceil(self._feedback_reserved_fraction * k)), max(k - 1, 0))
        centers, radii, counts, assignment = self._merge_down(centers, radii, counts, assignment, k - s)

        ops = 0
        for _ in range(s):
            best = self._best_split_candidate(centers, radii, counts, assignment, X_q, q_sq_norms, thresholds,
                                              weights, gamma, loss_and_accept)
            if best is None or best[0] <= 0:
                break
            _, centers, radii, counts, assignment = best
            ops += 1
        return centers, radii, counts, assignment, ops

    # ------------------------------------------------------------------ level 5: split_merge mode

    def _generate_split_candidates(self, centers, radii, assignment, X_q, q_sq_norms, thresholds, weights, gamma):
        accept = self._per_ball_accept(centers, radii, X_q, q_sq_norms, thresholds, gamma)
        ball_has_witness = np.flatnonzero(accept.any(axis=0))
        if ball_has_witness.size > self._FEEDBACK_MAX_CANDIDATE_BALLS:
            witness_weight = np.array([weights[accept[:, b]].sum() for b in ball_has_witness])
            keep = np.argsort(-witness_weight)[:self._FEEDBACK_MAX_CANDIDATE_BALLS]
            ball_has_witness = ball_has_witness[keep]
        candidates = []
        for b in ball_has_witness:
            b = int(b)
            candidates.extend(self._split_candidates_for_ball(b, assignment, accept[:, b], weights, X_q))
        return candidates

    def _feedback_split_merge(self, centers, radii, counts, assignment, X_q, q_sq_norms, thresholds, weights,
                              gamma, loss_and_accept):
        ops = 0
        for _ in range(self._feedback_max_iterations):
            loss_current, _ = loss_and_accept(centers, radii)
            if loss_current <= 0:
                break
            split_candidates = self._generate_split_candidates(
                centers, radii, assignment, X_q, q_sq_norms, thresholds, weights, gamma)
            if not split_candidates:
                break
            merge_candidates = self._top_merge_candidates(centers, radii, top_n=self._FEEDBACK_MAX_MERGE_CANDIDATES)
            if not merge_candidates:
                break

            best = None
            for sc in split_candidates:
                centers_s, radii_s, counts_s, assignment_s = self._apply_split_to_cover(
                    centers, radii, counts, assignment, sc["ball_idx"], sc["members"], sc["membership"])
                for (mi, mj) in merge_candidates:
                    if mi == sc["ball_idx"] or mj == sc["ball_idx"]:
                        continue
                    centers_sm, radii_sm, counts_sm, assignment_sm = self._apply_merge_to_cover(
                        centers_s, radii_s, counts_s, assignment_s, mi, mj)
                    loss_sm, _ = loss_and_accept(centers_sm, radii_sm)
                    gain = loss_current - loss_sm
                    if best is None or gain > best[0]:
                        best = (gain, centers_sm, radii_sm, counts_sm, assignment_sm)

            if best is None or best[0] <= 0:
                break
            _, centers, radii, counts, assignment = best
            ops += 1
        return centers, radii, counts, assignment, ops

    # ------------------------------------------------------------------ level 5: entry point

    def refine_with_feedback(self, queries: np.ndarray, thresholds=None,
                             weights: Optional[np.ndarray] = None) -> Dict[str, Any]:
        if not self._constructed:
            raise ValueError("GonzalezMEBFilter has not been constructed.")
        if self._X is None or self._assignment is None:
            raise ValueError(
                "refine_with_feedback requires construction state (original vectors + "
                "assignment); call construct() with complexity level >= 3, or "
                "construct(vectors, retain_assignments=True)."
            )

        t_start = time.perf_counter()
        X_q = np.ascontiguousarray(queries, dtype=np.float32)
        m = len(X_q)

        if thresholds is None:
            thr = np.full(m, self.distance_threshold, dtype=np.float64)
        elif np.isscalar(thresholds):
            thr = np.full(m, float(thresholds), dtype=np.float64)
        else:
            thr = np.asarray(thresholds, dtype=np.float64)
        w = np.ones(m, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
        gamma = self._feedback_margin
        q_sq_norms = np.einsum('ij,ij->i', X_q, X_q, dtype=np.float64) if m else np.empty(0)

        def loss_and_accept(centers, radii):
            if len(centers) == 0 or m == 0:
                return 0.0, np.zeros(m, dtype=bool)
            accepted = np.zeros(m, dtype=bool)
            radii64 = radii.astype(np.float64)
            for start in range(0, m, self._CHUNK_SIZE):
                end = min(start + self._CHUNK_SIZE, m)
                sq_d = self._sq_dists_to_centers(X_q[start:end], q_sq_norms[start:end], centers)
                eff_r = radii64[None, :] + thr[start:end, None] + gamma
                accepted[start:end] = np.any(sq_d <= eff_r ** 2, axis=1)
            return float(w[accepted].sum()), accepted

        centers = self.all_spheres.copy()
        radii = self.all_radiuses.copy()
        counts = self.sphere_counts.copy()
        assignment = self._assignment.copy()

        log_J_before = self._log_J(radii)
        loss_before, _ = loss_and_accept(centers, radii)

        accepted_ops = 0
        if loss_before > 0.0:
            if self._feedback_mode == "reserved":
                centers, radii, counts, assignment, accepted_ops = self._feedback_reserved(
                    centers, radii, counts, assignment, X_q, q_sq_norms, thr, w, gamma, loss_and_accept)
            elif self._feedback_mode == "split_merge":
                centers, radii, counts, assignment, accepted_ops = self._feedback_split_merge(
                    centers, radii, counts, assignment, X_q, q_sq_norms, thr, w, gamma, loss_and_accept)
            else:
                raise ValueError(f"Unknown feedback mode {self._feedback_mode!r}")

        loss_after, _ = loss_and_accept(centers, radii)
        log_J_after = self._log_J(radii)

        num_spheres = len(centers)
        self._reset_buffers(num_spheres)
        self._center_buf[:num_spheres] = centers
        self._radius_buf[:num_spheres] = radii
        self._count_buf[:num_spheres] = counts
        self._size = num_spheres
        self._assignment = assignment.astype(np.int32)

        fraction_eliminated = (1.0 - loss_after / loss_before) if loss_before > 0.0 else 0.0
        result = {
            "accepted_operations": accepted_ops,
            "fraction_eliminated": fraction_eliminated,
            "log_J_before": log_J_before,
            "log_J_after": log_J_after,
            "elapsed_seconds": time.perf_counter() - t_start,
        }
        self._feedback_stats = result
        return result
