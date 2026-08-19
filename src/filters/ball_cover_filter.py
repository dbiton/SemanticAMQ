import heapq
import time
from typing import List, Tuple, Optional, Dict, Any

import faiss
import numpy as np
import faiss
from typing import List, Tuple, Optional, Dict, Any
from filters.filter import Filter
from filters import ball_geometry
from hyperspheres_volume import estimate_hypersphere_volume


class BallCoverFilter(Filter):
    """
    Duplicate filter that covers the input vectors with a bounded number of
    hyperspheres and answers "is anything indexed within tau of this query?"
    without storing the vectors (tau is supplied at query time, not baked in).

    Construction is a fixed two-stage pipeline. Stage I builds an initial
    cover using one of three selectable strategies (`level`, set at
    construction time, default 3 -- see _init_streaming_merge/
    _init_hierarchical_split/_init_gonzalez); Stage II then always refines
    that cover: each cluster's ball is replaced by its Badoiu-Clarkson
    coreset MEB, vectors are reassigned to the nearest refined center, and
    the process repeats until log J = log sum(r_i^d) stops decreasing (or a
    small iteration cap is hit). Stage II is level-agnostic -- it consumes
    the same (centers, radii, counts, assignment) shape regardless of which
    level produced it.

    A further refinement is opt-in and runs *after* `construct()` rather
    than as a pipeline stage:

      - Quantization: compresses the stored centroids with a FAISS
        ScalarQuantizer (`quantize=True` at construction, or a manual
        `quantize_centers()` call). Every candidate in `_QUANTIZER_CANDIDATES`
        is tried; each candidate's own per-center reconstruction error is
        folded into that ball's radius (never subtracted), and whichever
        candidate yields the smallest resulting total covering volume
        log10 J(C) = log10 sum_i r_i^d is kept -- so the covering invariant
        holds regardless of which candidate wins, and the choice is driven
        directly by the same geometric objective construction itself
        minimizes.

    Streaming updates (`add`/`insert`) evaluate incoming vectors and create new
    spheres with radius `distance_threshold` for uncovered points. If the sphere
    count exceeds `max_spheres`, the filter iteratively merges candidate pairs from
    a minimal-pair min-heap. Volume growth is evaluated in log-space to prevent
    float64 overflow/underflow at high dimensions. To avoid full neighbor search, 
    candidate merges are restricted to nearest neighbors (`_MAX_NEIGHBORS`), and obsolete 
    heap entries are lazily invalidated via a per-sphere version array (`_sphere_versions`).

    Storage: centers/radii live in a single preallocated buffer
    (capacity grows by amortized doubling up to max_spheres), so add()/
    remove() never reallocate or copy the whole structure. `get_bytesize()`
    covers only that buffer (using the compressed per-vector size once
    quantized); construction state retained for quantize_centers()'s
    rebuild-from-vectors path (the original vectors + assignment) is
    reported separately by `get_construction_bytesize()`.
    """

    _BYTES_PER_FLOAT32 = 4
    _BYTES_PER_FLOAT64 = 8
    _MAX_NEIGHBORS = 50
    _BATCH_SIZE = 2048
    _MEB_CORESET_ITERATIONS = 30
    _MEB_REFINE_ITERATIONS = 200
    _BALL_REFINE_ITERATIONS = 10
    _CHUNK_SIZE = 20_000
    _LOG_J_EPS = 1e-9
    _DEFAULT_SEED = 42
    LEVELS = (1, 2, 3)
    _DEFAULT_LEVEL = 3
    _STAGE2_NAME = "ball_refine"
    _STAGE1_NAMES = {1: "streaming_merge_init", 2: "hierarchical_split_init", 3: "gonzalez_init"}
    _LN10 = float(np.log(10.0))
    _QUANTIZER_CANDIDATES = ("QT_fp16", "QT_8bit", "QT_6bit", "QT_4bit")

    def __init__(self, dim: int, distance_threshold: float, max_memory: Optional[int] = None,
                 max_spheres: Optional[int] = None, seed: int = _DEFAULT_SEED,
                 merge_criterion: str = "log_space", quantize: bool = False,
                 quantizer_candidates: Optional[Tuple[str, ...]] = None,
                 level: int = _DEFAULT_LEVEL, streaming_shuffle: bool = False,
                 streaming_n_candidates: int = 0):
        super().__init__(dim, lsh=None, max_memory=max_memory)
        self.distance_threshold = distance_threshold
        self.max_spheres = (max_spheres if max_spheres is not None
                            else self._calculate_max_spheres(self.max_memory))
        self._seed = seed
        if merge_criterion not in self.MERGE_CRITERIA:
            raise ValueError(f"Unknown merge_criterion {merge_criterion!r}; expected one of {self.MERGE_CRITERIA}")
        self.merge_criterion = merge_criterion
        if level not in self.LEVELS:
            raise ValueError(f"level must be in {self.LEVELS}, got {level}")
        self.level = level
        self.streaming_shuffle = bool(streaming_shuffle)
        self.streaming_n_candidates = int(streaming_n_candidates)
        # certificate: only level 3 (Gonzalez) populates this, see _init_gonzalez
        self.lower_bound: Optional[float] = None
        self._constructed = False

        # main filter sphere buffers
        self._num_spheres = 0
        self._all_spheres = np.arange(self.max_spheres, dtype=np.int32)
        self._all_centers = np.empty((self.max_spheres, dim), dtype=np.float32)
        self._all_radiuses = np.empty((self.max_spheres,), dtype=np.float32)

        # streaming merge data structures, minimal pair heap and sphere version array
        self._sphere_versions = np.zeros((self.max_spheres,), dtype=np.int32)
        self._minimal_pairs: List[Tuple[float, int, int, int, int]] = []

        # construction state
        self._retain_assignments = False
        self._X: Optional[np.ndarray] = None
        self._assignment: Optional[np.ndarray] = None

        # quantization policy
        self._quantize = bool(quantize)
        self._quantizer_candidates = (tuple(quantizer_candidates) if quantizer_candidates is not None
                                      else self._QUANTIZER_CANDIDATES)
        self._quantizer_name: Optional[str] = None
        self._quantizer_bytes_per_vector: Optional[int] = None
        self._quantizer_stats: Optional[Dict[str, Any]] = None
        self._stage_stats: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ properties

    @property
    def all_spheres(self) -> np.ndarray:
        return self._all_spheres[:self._num_spheres]

    @property
    def all_centers(self) -> np.ndarray:
        return self._all_centers[self.all_spheres]

    @property
    def all_radiuses(self) -> np.ndarray:
        return self._all_radiuses[self.all_spheres]

    # ------------------------------------------------------------------ sizing / buffers

    def _calculate_max_spheres(self, max_memory: int, gamma: float = 4.0) -> int:
        bytes_per_sphere = (self.dim + 2) * self._BYTES_PER_FLOAT32
        return max(1, int(max_memory // (bytes_per_sphere * gamma)))

    def _ensure_capacity(self, required_cap: int) -> None:
        current_cap = len(self._all_centers)
        if required_cap <= current_cap:
            return

        new_cap = max(current_cap * 2, required_cap)
        new_spheres = np.empty((new_cap,), dtype=np.int32)
        new_spheres[:current_cap] = self._all_spheres
        new_spheres[current_cap:] = np.arange(current_cap, new_cap, dtype=np.int32)
        self._all_spheres = new_spheres

        new_centers = np.empty((new_cap, self.dim), dtype=np.float32)
        new_centers[:current_cap] = self._all_centers
        self._all_centers = new_centers

        new_radiuses = np.empty((new_cap,), dtype=np.float32)
        new_radiuses[:current_cap] = self._all_radiuses
        self._all_radiuses = new_radiuses

        new_versions = np.zeros(new_cap, dtype=np.int32)
        new_versions[:current_cap] = self._sphere_versions
        self._sphere_versions = new_versions

    def _get_new_index(self) -> int:
        if self._num_spheres == len(self._all_centers):
            self._ensure_capacity(len(self._all_centers) * 2)
        return self._all_spheres[self._num_spheres]

    def _reset_buffers(self, capacity: int) -> None:
        self._ensure_capacity(capacity)
        self._num_spheres = 0
        self._minimal_pairs = []
        self._sphere_versions.fill(0)

    # ------------------------------------------------------------------ numerics (log-space)

    def _safe_radius(self, r: float) -> float:
        return ball_geometry.safe_radius(r)

    def _log_J(self, radii: np.ndarray) -> float:
        return ball_geometry.log_J(radii, self.dim)

    def _log_J_tau(self, radii: np.ndarray, tau: float) -> float:
        return ball_geometry.log_J_tau(radii, self.dim, tau)

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
        return ball_geometry.merge_cost_log_space(ri, rj, r_merged, self.dim)

    def _merge_cost_r_pow_d(self, ri: np.ndarray, rj: np.ndarray,
                            r_merged: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Delta = r_merged^d - r_i^d - r_j^d under the direct (non-log)
        objective J = sum r_i^d. Same log-domain evaluation as
        _merge_cost_log_space (never materializes r^d, which overflows well
        within realistic radii at d ~ 384) since it's the same real-valued
        Delta either way -- this criterion differs from "log_space" only in
        name, kept as its own MERGE_CRITERIA entry for callers that select
        by that name."""
        return ball_geometry.merge_cost_log_space(ri, rj, r_merged, self.dim)

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
        return ball_geometry.merged_ball(ci, ri, cj, rj)

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
        if len(self.all_centers) == 0:
            return -np.inf, -np.inf
        return estimate_hypersphere_volume(self.all_centers, self.all_radiuses, return_log=True)

    # ------------------------------------------------------------------ init: shared helpers

    def _exact_centroids_from_groups(self, vectors: np.ndarray, groups: List[np.ndarray]
                                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Center = exact centroid of a group, radius = exact max distance to
        any member -- never carried forward from whatever produced the
        grouping (invariant #1). Shared final-recompute step for all three
        init strategies."""
        m = len(groups)
        centers = np.empty((m, self.dim), dtype=np.float32)
        radii = np.empty(m, dtype=np.float32)
        counts = np.empty(m, dtype=np.int32)
        for i, idx in enumerate(groups):
            pts = vectors[idx].astype(np.float64)
            c64 = pts.mean(axis=0)
            c32 = c64.astype(np.float32)
            c64 = c32.astype(np.float64)  # radius must match the center as actually stored
            d = np.sqrt(np.maximum(np.einsum('ij,ij->i', pts - c64, pts - c64), 0.0))
            centers[i] = c32
            radii[i] = self._safe_radius(d.max())
            counts[i] = len(idx)
        return centers, radii, counts

    def _assign_and_recompute(self, vectors: np.ndarray, sq_norms: np.ndarray, raw_centers: np.ndarray
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """One nearest-center pass against `raw_centers`, then centers/radii
        recomputed exactly (never carried forward) from that assignment.
        Used by the streaming-merge (level 1) and Gonzalez (level 3) init
        strategies, whose raw output is a set of centers with no exact
        per-point assignment yet."""
        raw_assignment, _ = self._chunked_assign(vectors, sq_norms, raw_centers)
        unique_labels, assignment = np.unique(raw_assignment, return_inverse=True)
        groups = self._groups_from_assignment(assignment, len(unique_labels))
        centers, radii, counts = self._exact_centroids_from_groups(vectors, groups)
        return centers, radii, counts, assignment.astype(np.int32)

    def _farthest_pair_split(self, idx: np.ndarray, vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Split point-index group `idx` into two: farthest-from-centroid
        point A, then farthest-from-A point B, then assign every member to
        whichever of {A, B} is nearer. Falls back to an index bisection when
        all members are coincident (A==B at distance 0) or the geometric
        split would strand one side empty, so a degenerate cluster still
        yields two non-empty children."""
        pts = vectors[idx].astype(np.float64)
        centroid = pts.mean(axis=0)
        d_centroid = np.einsum('ij,ij->i', pts - centroid, pts - centroid)
        a_local = int(d_centroid.argmax())
        a = pts[a_local]
        d_a = np.einsum('ij,ij->i', pts - a, pts - a)
        b_local = int(d_a.argmax())
        if d_a[b_local] <= 0.0:
            half = len(idx) // 2
            return idx[:half], idx[half:]
        b = pts[b_local]
        d_b = np.einsum('ij,ij->i', pts - b, pts - b)
        nearer_a = d_a <= d_b
        left, right = idx[nearer_a], idx[~nearer_a]
        if len(left) == 0 or len(right) == 0:
            half = len(idx) // 2
            return idx[:half], idx[half:]
        return left, right

    def _saturate_to_k(self, groups: List[np.ndarray], vectors: np.ndarray, k: int) -> List[np.ndarray]:
        """Invariant #2: repeatedly split the largest-population group until
        exactly k groups exist (a no-op if a level already produced k).
        Population, not radius, is the split key here since callers reach
        this after already discarding whatever intermediate radii they had."""
        groups = list(groups)
        while len(groups) < k:
            sizes = [len(g) for g in groups]
            largest = int(np.argmax(sizes))
            if sizes[largest] < 2:
                break  # fewer distinct points than k; cannot split further
            left, right = self._farthest_pair_split(groups[largest], vectors)
            groups[largest] = left
            groups.append(right)
        return groups

    # ------------------------------------------------------------------ level 1: streaming merge

    def _init_streaming_merge(self, vectors: np.ndarray, sq_norms: np.ndarray, k: int
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Single pass, O(kd) working memory: insert every point as a
        zero-radius singleton (arrival order, optionally shuffled) and, each
        time the cover exceeds k, merge the pair minimizing the existing
        merge cost (self._pairwise_merge_keys / self._merged_ball -- the
        same cost function and geometry BallCoverFilter.add() uses, just
        driven over local arrays instead of the instance's incremental
        O(1)-amortized cache, since that cache is tied to the persistent
        instance buffers and construction shouldn't mutate those as scratch
        space). Produces no assignment during the pass; recovered by one
        nearest-center pass at the end via _assign_and_recompute, which also
        recomputes radii exactly."""
        n = len(vectors)
        order = np.arange(n)
        if self.streaming_shuffle:
            order = np.random.default_rng(self._seed).permutation(n)

        cap = min(k, n) + 1
        centers = np.empty((cap, self.dim), dtype=np.float64)
        radii = np.zeros(cap, dtype=np.float64)
        size = 0
        n_candidates = self.streaming_n_candidates

        for i in order:
            new_idx = size
            centers[size] = vectors[i].astype(np.float64)
            radii[size] = 0.0
            size += 1
            if size <= k:
                continue

            active_c, active_r = centers[:size], radii[:size]
            if n_candidates > 0 and size > n_candidates + 1:
                # restricted: only the new singleton vs. its n_candidates
                # nearest existing balls -- O(kd) distances + O(kappa*d) cost
                # evaluation, instead of the O(k^2 d) all-pairs scan below.
                others = np.delete(np.arange(size), new_idx)
                d = np.linalg.norm(active_c[others] - active_c[new_idx], axis=1)
                kappa = min(n_candidates, len(others))
                nearest = others[np.argpartition(d, kappa - 1)[:kappa]]
                ii = np.full(kappa, new_idx)
                jj = nearest
            else:
                ii, jj = np.triu_indices(size, k=1)

            bucket, key, _ = self._pairwise_merge_keys(active_c[ii], active_r[ii], active_c[jj], active_r[jj])
            best = int(ball_geometry.argmin_bucket_key(bucket, key))
            a, b = int(ii[best]), int(jj[best])
            c_new, r_new = self._merged_ball(active_c[a], active_r[a], active_c[b], active_r[b])
            centers[a] = c_new
            radii[a] = self._safe_radius(r_new)
            last = size - 1
            if b != last:
                centers[b] = centers[last]
                radii[b] = radii[last]
            size -= 1

        raw_centers = centers[:size].astype(np.float32)
        return self._assign_and_recompute(vectors, sq_norms, raw_centers)

    # ------------------------------------------------------------------ level 2: hierarchical split

    def _init_hierarchical_split(self, vectors: np.ndarray, sq_norms: np.ndarray, k: int
                                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """log k passes: a max-heap of clusters keyed by (approximate)
        radius, starting from one cluster holding every point. Each of k-1
        iterations pops the largest-radius cluster and 2-center splits it
        (_farthest_pair_split); children are pushed back keyed by their own
        approximate radius. If splitting plateaus (a cluster of coincident
        points, farthest-pair distance 0) _farthest_pair_split's index
        bisection keeps the loop progressing toward k without further
        distance work. Exact centers/radii are only computed once, at the
        end, from the final k leaf groups (invariant #1)."""
        n = len(vectors)
        all_idx = np.arange(n, dtype=np.int64)
        vectors64 = vectors.astype(np.float64)
        centroid = vectors64.mean(axis=0)
        d0 = np.sqrt(np.maximum(np.einsum('ij,ij->i', vectors64 - centroid, vectors64 - centroid), 0.0))
        # heap entries: (-approx_radius, insertion_counter, index_array) --
        # the counter breaks ties without numpy ever comparing two arrays.
        heap: List[Tuple[float, int, np.ndarray]] = [(-float(d0.max()), 0, all_idx)]
        counter = 1

        while len(heap) < k:
            neg_r, _, idx = heapq.heappop(heap)
            if len(idx) < 2:
                heapq.heappush(heap, (neg_r, -1, idx))
                break  # unreachable when n > k (pigeonhole), kept as a guard
            left, right = self._farthest_pair_split(idx, vectors)
            for child in (left, right):
                pts = vectors64[child]
                c = pts.mean(axis=0)
                r = float(np.sqrt(np.maximum(np.einsum('ij,ij->i', pts - c, pts - c), 0.0)).max())
                heapq.heappush(heap, (-r, counter, child))
                counter += 1

        groups = self._saturate_to_k([idx for _, _, idx in heap], vectors, k)
        centers, radii, counts = self._exact_centroids_from_groups(vectors, groups)
        assignment = np.empty(n, dtype=np.int32)
        for label, idx in enumerate(groups):
            assignment[idx] = label
        return centers, radii, counts, assignment

    # ------------------------------------------------------------------ level 3: gonzalez

    def _init_gonzalez(self, vectors: np.ndarray, sq_norms: np.ndarray, k: int
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """k passes of farthest-point (Gonzalez) traversal: the first center
        is the point farthest from the mean, each subsequent center is the
        point farthest from its nearest already-chosen center. One extra
        (k+1-th) selection step beyond the k needed records D, the
        (k+1)-th insertion distance -- by pigeonhole any k-ball cover must
        have some ball of radius >= D/2 (the (k+1)-th farthest point and its
        nearest of the first k centers are each >= D apart from every other
        chosen center, so no single ball of radius < D/2 can cover two of
        them), certifying lower_bound = d * log(D/2) as a lower bound on
        log J -- only this level can report it (see class docstring)."""
        n = len(vectors)
        mean = vectors.astype(np.float64).mean(axis=0)
        first = int((sq_norms - 2.0 * (vectors @ mean)).argmax())

        chosen = [first]
        c = vectors[first]
        nearest_sq = np.maximum(sq_norms - 2.0 * (vectors @ c) + c @ c, 0.0)

        n_steps = min(k + 1, n)
        farthest_d = 0.0
        for step in range(1, n_steps):
            far_idx = int(nearest_sq.argmax())
            farthest_d = float(np.sqrt(max(float(nearest_sq[far_idx]), 0.0)))
            if step >= k:
                break  # certificate-only (k+1)-th selection; not added as a center
            chosen.append(far_idx)
            c = vectors[far_idx]
            new_sq = np.maximum(sq_norms - 2.0 * (vectors @ c) + c @ c, 0.0)
            np.minimum(nearest_sq, new_sq, out=nearest_sq)

        self.lower_bound = float(self.dim * np.log(farthest_d / 2.0)) if farthest_d > 0 else float('-inf')

        raw_centers = vectors[np.array(chosen, dtype=np.int64)]
        return self._assign_and_recompute(vectors, sq_norms, raw_centers)

    def _run_init(self, vectors: np.ndarray, sq_norms: np.ndarray, k: int
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self.lower_bound = None
        if self.level == 1:
            return self._init_streaming_merge(vectors, sq_norms, k)
        elif self.level == 2:
            return self._init_hierarchical_split(vectors, sq_norms, k)
        elif self.level == 3:
            return self._init_gonzalez(vectors, sq_norms, k)
        raise ValueError(f"Unknown level {self.level!r}; expected one of {self.LEVELS}")

    # ------------------------------------------------------------------ stage II: ball refinement

    def _meb_refine(self, vectors: np.ndarray, sq_norms: np.ndarray, centers: np.ndarray,
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
            for i, idx in enumerate(groups):
                c, r = self._minimum_enclosing_ball(vectors[idx], sq_norms[idx])
                new_centers[i] = c
                new_radii[i] = r

            new_log_J = self._log_J(new_radii)
            if new_log_J < prev_log_J - self._LOG_J_EPS:
                centers, radii, assignment = new_centers, new_radii, new_assignment.astype(np.int32)
                prev_log_J = new_log_J
                accepted += 1
            else:
                break

        return centers, radii, assignment, accepted

    # ------------------------------------------------------------------ instrumentation

    def _make_stage_record(self, level: int, name: str, wall_time: float,
                           radii: np.ndarray, accepted_ops: int) -> Dict[str, Any]:
        return {
            "level": level,
            "name": name,
            "wall_time": wall_time,
            "ball_count": len(radii),
            "log_J": self._log_J(radii),
            "radii_snapshot": np.asarray(radii, dtype=np.float32).copy(),
            "accepted_ops": accepted_ops,
        }

    def _record_stage(self, level: int, name: str, t_start: float, radii: np.ndarray, accepted_ops: int) -> None:
        self._stage_stats.append(
            self._make_stage_record(level, name, time.perf_counter() - t_start, radii, accepted_ops))

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
        return {"levels": levels, "quantization": self._quantizer_stats}

    # ------------------------------------------------------------------ construct

    def construct(self, vectors: np.ndarray, retain_assignments: bool = False) -> Tuple[float, float]:
        self._stage_stats = []
        self._quantizer_name = None
        self._quantizer_bytes_per_vector = None
        self._quantizer_stats = None
        self.lower_bound = None
        retain = bool(retain_assignments)

        if len(vectors) == 0 or self.max_spheres == 0:
            self._reset_buffers(0)
            vectors_arr = np.ascontiguousarray(vectors, dtype=np.float32)
            self._retain_assignments = retain
            self._X = vectors_arr if retain else None
            self._assignment = np.empty(0, dtype=np.int32) if retain else None
            t0 = time.perf_counter()
            self._record_stage(1, self._STAGE1_NAMES[self.level], t0, np.empty(0, dtype=np.float32), 0)
            self._record_stage(2, self._STAGE2_NAME, t0, np.empty(0, dtype=np.float32), 0)
            self._constructed = True
            return self._log_volume_bounds()

        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        sq_norms = np.einsum('ij,ij->i', vectors, vectors, dtype=np.float64)

        if self._quantize:
            centers, radii, assignment, chosen_k, stage_records, quant = \
                self._construct_with_quantization(vectors, sq_norms)
            self.max_spheres = chosen_k
            self._stage_stats = stage_records
            if quant is not None:
                self._quantizer_name = quant["quantizer"]
                self._quantizer_bytes_per_vector = quant["bytes_per_vector"]
                self._quantizer_stats = quant
        else:
            centers, radii, assignment, stage_records = self._build_cover(
                vectors, sq_norms, self.max_spheres)
            self._stage_stats = stage_records

        num_spheres = len(centers)
        self._reset_buffers(num_spheres)
        for center, radius in zip(centers, radii):
            self._add_sphere(center, radius)

        self._retain_assignments = retain
        self._X = vectors if retain else None
        self._assignment = assignment.astype(np.int32) if retain else None
        self._constructed = True
        return self._log_volume_bounds()

    def _build_cover(self, vectors: np.ndarray, sq_norms: np.ndarray,
                     k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Run the fixed two-stage construction pipeline (init strategy
        self.level, then MEB refinement) for an arbitrary ball budget k
        against already-preprocessed vectors, touching no instance state.
        Used both for a plain (non-quantized) construct() and, once per
        quantization candidate, to rebuild at that candidate's own larger
        effective_k (see _construct_with_quantization). Returns
        centers/radii/counts/assignment plus this build's own stage records,
        the same shape _record_stage would append to self._stage_stats for a
        normal top-level build.
        """
        n = len(vectors)
        stage_records: List[Dict[str, Any]] = []

        if k >= n:
            centers = vectors.copy()
            radii = np.zeros(n, dtype=np.float32)
            assignment = np.arange(n, dtype=np.int32)
            stage_records.append(self._make_stage_record(1, self._STAGE1_NAMES[self.level], 0.0, radii, 0))
            stage_records.append(self._make_stage_record(2, self._STAGE2_NAME, 0.0, radii, 0))
            return centers, radii, counts, assignment, stage_records

        t0 = time.perf_counter()
        centers, radii, counts, assignment = self._run_init(vectors, sq_norms, k)
        stage_records.append(
            self._make_stage_record(1, self._STAGE1_NAMES[self.level], time.perf_counter() - t0, radii, 0))

        t0 = time.perf_counter()
        centers, radii, counts, assignment, accepted = self._meb_refine(
            vectors, sq_norms, centers, radii, counts, assignment)
        stage_records.append(
            self._make_stage_record(2, self._STAGE2_NAME, time.perf_counter() - t0, radii, accepted))

        return centers, radii, assignment, stage_records

    # ------------------------------------------------------------------ quantization

    def _construct_with_quantization(self, vectors: np.ndarray, sq_norms: np.ndarray):
        """
        K (self.max_spheres) is treated as a memory *budget* -- "how many
        float32-precision spheres this much memory could hold" -- not a
        literal ball count. A quantizer that halves the per-sphere footprint
        is only worth using if that freed memory is spent on twice as many,
        finer spheres, so every candidate here is rebuilt from scratch at its
        own larger effective_k = budget_k * (float32 bytes/sphere) /
        (candidate bytes/sphere) -- never just compressed in place at the
        original k, which could only ever grow J(C).

        The unquantized float32-at-budget_k build is included as its own
        baseline candidate: more, coarser spheres isn't guaranteed to beat
        fewer, exact ones, since covering radius shrinks slowly with k until
        points/ball approaches ~1 (curse of dimensionality), so a modest
        compression ratio can lose more to quantization noise than it gains
        from extra spheres. Whichever configuration ends up with the
        smallest total covering volume log10 J(C) = log10 sum_i r_i^d is
        kept -- each candidate's own per-center reconstruction error is
        folded into that ball's radius (never subtracted), so whichever one
        wins still holds the zero-false-negative covering invariant.

        Returns (centers, radii, counts, assignment, chosen_k, stage_records,
        quantizer_stats_or_None).
        """
        budget_k = self.max_spheres
        # Per-sphere cost as get_bytesize() actually accounts it: center +
        # radius (float32) + count (int32) -- omitting the count term here
        # would let effective_k overshoot the true byte budget once it's
        # multiplied out over a much larger ball count.
        bytes_per_sphere_f32 = (self.dim + 2) * self._BYTES_PER_FLOAT32

        centers0, radii0, assignment0, stages0 = self._build_cover(vectors, sq_norms, budget_k)
        best = {
            "log10_J": self._log_J(radii0.astype(np.float64)) / self._LN10,
            "k": budget_k, "centers": centers0, "radii": radii0,
            "assignment": assignment0, "stages": stages0, "quant": None,
        }

        for name in self._quantizer_candidates:
            qtype = getattr(faiss.ScalarQuantizer, name)
            bytes_per_vector = faiss.ScalarQuantizer(self.dim, qtype).code_size
            bytes_per_sphere_candidate = bytes_per_vector + 2 * self._BYTES_PER_FLOAT32  
            effective_k = max(1, budget_k * bytes_per_sphere_f32 // bytes_per_sphere_candidate)

            centers, radii, assignment, stages = self._build_cover(vectors, sq_norms, effective_k)
            centers32 = np.ascontiguousarray(centers, dtype=np.float32)
            radii64 = radii.astype(np.float64)

            sq = faiss.ScalarQuantizer(self.dim, qtype)
            sq.train(centers32)
            codes = sq.compute_codes(centers32)
            decoded = sq.decode(codes).astype(np.float64)
            # Triangle inequality: for any point x, |d(x,center) - d(x,decoded)|
            # <= ||center - decoded||, so inflating a ball's own radius by its
            # own reconstruction error keeps every point the true-center ball
            # covered still covered once the center is replaced by its decoded
            # (quantized) approximation.
            error = np.linalg.norm(centers32.astype(np.float64) - decoded, axis=1)
            quantized_radii = radii64 + error
            log10_j = self._log_J(quantized_radii) / self._LN10

            if log10_j < best["log10_J"]:
                best = {
                    "log10_J": log10_j, "k": effective_k,
                    "centers": decoded.astype(np.float32),
                    "radii": quantized_radii.astype(np.float32),
                    "assignment": assignment, "stages": stages,
                    "quant": {
                        "quantizer": name, "bytes_per_vector": int(codes.shape[1]),
                        "log10_J": log10_j, "log10_J_unquantized": self._log_J(radii64) / self._LN10,
                        "mean_center_error": float(error.mean()), "max_center_error": float(error.max()),
                        "budget_k": budget_k, "effective_k": effective_k,
                    },
                }

        return (best["centers"], best["radii"], best["assignment"], best["k"], best["stages"], best["quant"])

    def _apply_quantization_inplace(self) -> Optional[Dict[str, Any]]:
        """
        Compress the *currently stored* centers in place, with no k
        rescaling -- used by quantize_centers() when the original vectors
        aren't available to rebuild from (construct() wasn't called with
        retain_assignments=True). Since this can only ever grow radii, it
        compares every candidate's inflated-radius volume against the
        unquantized baseline and leaves the cover untouched (returns None)
        if nothing beats staying float32; see _construct_with_quantization
        for the k-aware version used when the original vectors are on hand.
        """
        if self._num_spheres == 0:
            return None

        centers = self.all_centers.astype(np.float32)
        radii = self.all_radiuses.astype(np.float64)
        log10_j_plain = self._log_J(radii) / self._LN10

        best = None
        for name in self._quantizer_candidates:
            qtype = getattr(faiss.ScalarQuantizer, name)
            sq = faiss.ScalarQuantizer(self.dim, qtype)
            sq.train(centers)
            codes = sq.compute_codes(centers)
            decoded = sq.decode(codes).astype(np.float64)
            error = np.linalg.norm(centers.astype(np.float64) - decoded, axis=1)

            log10_j = self._log_J(radii + error) / self._LN10
            if best is None or log10_j < best["log10_J"]:
                best = {
                    "name": name, "log10_J": log10_j,
                    "bytes_per_vector": int(codes.shape[1]),
                    "decoded": decoded.astype(np.float32),
                    "error": error,
                }

        if best is None or log10_j_plain <= best["log10_J"]:
            return None  # unquantized float32 already minimizes volume; leave centers untouched

        current_spheres = self.all_spheres
        self._all_centers[current_spheres] = best["decoded"]
        self._all_radiuses[current_spheres] = (radii + best["error"]).astype(np.float32)

        return {
            "quantizer": best["name"],
            "bytes_per_vector": best["bytes_per_vector"],
            "log10_J": best["log10_J"],
            "log10_J_unquantized": log10_j_plain,
            "mean_center_error": float(best["error"].mean()),
            "max_center_error": float(best["error"].max()),
        }

    def quantize_centers(self) -> Optional[Dict[str, Any]]:
        """
        Manually (re-)apply center quantization outside of construct()'s
        auto-quantize path -- e.g. after streaming add()/remove() calls have
        left centers at full float32 precision. If construction state was
        retained (construct(..., retain_assignments=True)), this
        re-runs the same budget-vs-quantizer search construct() uses when
        quantize=True, rebuilding each candidate from the original vectors
        (and may grow self.max_spheres, same as construct()); otherwise it
        falls back to compressing the currently stored centers in place
        (_apply_quantization_inplace), which cannot add spheres. Returns the
        same diagnostic dict as get_stats()["quantization"], or None if
        nothing was quantized (empty cover, or float32 already minimizes
        total volume).
        """
        if not self._constructed:
            raise ValueError("BallCoverFilter has not been constructed.")
        if self._num_spheres == 0:
            return None

        if self._X is not None and len(self._X) > 0:
            sq_norms = np.einsum('ij,ij->i', self._X, self._X, dtype=np.float64)
            centers, radii, assignment, chosen_k, stage_records, quant = \
                self._construct_with_quantization(self._X, sq_norms)
            self.max_spheres = chosen_k
            num_spheres = len(centers)
            self._reset_buffers(num_spheres)
            self._all_centers[:num_spheres] = centers
            self._all_radiuses[:num_spheres] = radii
            self._all_spheres[:num_spheres] = np.arange(num_spheres, dtype=np.int32)
            self._num_spheres = num_spheres
            self._assignment = assignment.astype(np.int32)
            self._stage_stats = stage_records
        else:
            quant = self._apply_quantization_inplace()

        self._quantizer_name = quant["quantizer"] if quant else None
        self._quantizer_bytes_per_vector = quant["bytes_per_vector"] if quant else None
        self._quantizer_stats = quant
        return quant

    # ------------------------------------------------------------------ query

    def query(self, vectors: np.ndarray, tau: Optional[float] = None) -> List[bool]:
        if not self._constructed:
            raise ValueError("BallCoverFilter has not been constructed.")
        if tau is None:
            tau = self.distance_threshold

        if len(self.all_centers) == 0 or len(vectors) == 0:
            return [False] * len(vectors)

        q = np.ascontiguousarray(vectors, dtype=np.float32)
        n = len(q)
        centers = self.all_centers
        eff_r = self.all_radiuses.astype(np.float64) + float(tau)
        q_sq_norms = np.einsum('ij,ij->i', q, q, dtype=np.float64)

        result = np.zeros(n, dtype=bool)
        for start in range(0, n, self._CHUNK_SIZE):
            end = min(start + self._CHUNK_SIZE, n)
            sq_d = self._sq_dists_to_centers(q[start:end], q_sq_norms[start:end], centers)
            result[start:end] = np.any(sq_d <= (eff_r[None, :]) ** 2, axis=1)
        return result.tolist()

    def query_multi_tau(self, vectors: np.ndarray, taus: List[float]) -> Dict[float, List[bool]]:
        """Same containment test as query(), evaluated at several tau at
        once. The query-to-center distance pass (O(n*k*d), the expensive
        part) doesn't depend on tau -- only the per-ball threshold (radius +
        tau) does -- so it's computed once per chunk and reused for every
        tau's containment test, instead of re-querying once per tau. Results
        are identical to calling query(vectors, tau=t) separately for each
        t in taus."""
        if not self._constructed:
            raise ValueError("BallCoverFilter has not been constructed.")
        taus = list(taus)
        q = np.ascontiguousarray(vectors, dtype=np.float32)
        n = len(q)

        if self._size == 0 or n == 0:
            return {t: [False] * n for t in taus}

        centers = self.all_spheres
        radii64 = self.all_radiuses.astype(np.float64)
        q_sq_norms = np.einsum('ij,ij->i', q, q, dtype=np.float64)

        result = {t: np.zeros(n, dtype=bool) for t in taus}
        for start in range(0, n, self._CHUNK_SIZE):
            end = min(start + self._CHUNK_SIZE, n)
            sq_d = self._sq_dists_to_centers(q[start:end], q_sq_norms[start:end], centers)
            for t in taus:
                eff_r = radii64 + float(t)
                result[t][start:end] = np.any(sq_d <= (eff_r[None, :]) ** 2, axis=1)
        return {t: arr.tolist() for t, arr in result.items()}

    # ------------------------------------------------------------------ streaming add/remove

    def _add_sphere(self, center: np.ndarray, radius: float) -> None:
        current_spheres = self.all_spheres

        if self._num_spheres > 0:
            current_centers = self._all_centers[current_spheres]
            current_radiuses = self._all_radiuses[current_spheres]
            center_sq = np.einsum('ij,ij->i', center[None, :], center[None, :], dtype=np.float64)

            sq_center_distances = self._sq_dists_to_centers(
                center[None, :], center_sq, current_centers)[0]
            center_distances = np.sqrt(np.maximum(sq_center_distances, 0.0))

            if np.any(center_distances + radius <= current_radiuses):
                return

            contained_indices = (center_distances + current_radiuses) <= radius
            if np.any(contained_indices):
                contained_spheres = current_spheres[contained_indices]
                current_spheres = current_spheres[~contained_indices]
                self._all_spheres[:self._num_spheres] = np.concatenate([current_spheres, contained_spheres])
                self._num_spheres -= len(contained_spheres)
                self._sphere_versions[contained_spheres] += 1
                current_radiuses = current_radiuses[~contained_indices]
                center_distances = center_distances[~contained_indices]

        sphere_idx = self._get_new_index()
        sphere_version = self._sphere_versions[sphere_idx]
        self._all_centers[sphere_idx] = center
        self._all_radiuses[sphere_idx] = radius
        self._num_spheres += 1

        if len(current_spheres) > 0:
            max_neighbors = min(self._MAX_NEIGHBORS, len(current_spheres))
            if max_neighbors == len(current_spheres):
                neighbors_indices = np.arange(len(current_spheres))
            else:
                neighbors_indices = np.argpartition(center_distances, max_neighbors - 1)[:max_neighbors]

            closest_neighbors = current_spheres[neighbors_indices]
            neighbors_center_dists = center_distances[neighbors_indices]
            neighbors_radiuses = current_radiuses[neighbors_indices]
            merged_radius = np.maximum((neighbors_radiuses + neighbors_center_dists + radius) / 2.0, 1e-6)

            # log(R_merged^D - (R_1^D + R_2^D)) = D*log(R_merged) + log(1 - ((R_1/R_merged)^D + (R_2/R_merged)^D))
            merged_volumes = self.dim * np.log(merged_radius) + np.log(np.maximum(1.0 - ((radius / merged_radius) ** self.dim + (neighbors_radiuses / merged_radius) ** self.dim), 1e-6))

            for volume, neighbor_idx in zip(merged_volumes, closest_neighbors):
                neighbor_version = self._sphere_versions[neighbor_idx]
                heapq.heappush(self._minimal_pairs, (volume, sphere_idx, sphere_version, neighbor_idx, neighbor_version))

    def _merge_spheres(self) -> None:
        while self._num_spheres > self.max_spheres and self._minimal_pairs:
            vol, i, i_version, j, j_version = heapq.heappop(self._minimal_pairs)

            if self._sphere_versions[i] != i_version or self._sphere_versions[j] != j_version:
                continue

            center_i, radius_i = self._all_centers[i], self._all_radiuses[i]
            center_j, radius_j = self._all_centers[j], self._all_radiuses[j]

            centers_diff = center_i - center_j
            distance_ij = float(np.sqrt(np.dot(centers_diff, centers_diff)))
            new_radius = (distance_ij + radius_i + radius_j) / 2.0

            if distance_ij > 0.0:
                weight_j = ((radius_j - radius_i) / (2.0 * distance_ij)) + 0.5
                weight_i = ((radius_i - radius_j) / (2.0 * distance_ij)) + 0.5
                new_center = weight_j * center_j + weight_i * center_i
            else:
                new_center = center_i

            current_spheres = self.all_spheres[(self.all_spheres != i) & (self.all_spheres != j)]
            self._all_spheres[:self._num_spheres] = np.concatenate([current_spheres, [i, j]])
            self._sphere_versions[[i, j]] += 1
            self._num_spheres -= 2

            self._add_sphere(new_center.astype(np.float32), new_radius)

    def add(self, vectors: np.ndarray) -> None:
        if not self._constructed:
            self.construct(vectors)
            return

        if len(vectors) == 0 or self.max_spheres == 0:
            return

        for i in range(0, len(vectors), self._BATCH_SIZE):
            vector_batch = vectors[i:i + self._BATCH_SIZE]
            vector_batch = np.ascontiguousarray(vector_batch, dtype=np.float32)
            sq_vector_batch = np.einsum('ij,ij->i', vector_batch, vector_batch, dtype=np.float64)

            if self._num_spheres > 0:
                sq_batch_distances = self._sq_dists_to_centers(
                    vector_batch, sq_vector_batch, self.all_centers)
                batch_distances = np.sqrt(np.maximum(sq_batch_distances, 0.0))

                is_already_covered = np.any(batch_distances <= (self.all_radiuses[None, :] - self.distance_threshold), axis=1)
                vector_batch = vector_batch[~is_already_covered]

            for vec in vector_batch:
                self._add_sphere(vec, self.distance_threshold)

            self._merge_spheres()

    def insert(self, vectors: np.ndarray) -> None:
        self.add(vectors)

    def remove(self, vectors: np.ndarray) -> List[bool]:
        raise NotImplementedError("BallCoverFilter does not support remove operation.")

    # ------------------------------------------------------------------ memory / misc

    def get_volume(self) -> float:
        return 0.0

    def get_exact_matching_fpr(self) -> float:
        return 0.0

    def get_bytesize(self) -> int:
        if not self._constructed:
            return 0
        if self._quantizer_name is not None:
            center_bytes = self._num_spheres * self._quantizer_bytes_per_vector
        else:
            center_bytes = self.all_centers.nbytes
        return center_bytes + self.all_radiuses.nbytes + self._all_spheres.nbytes + sys.getsizeof(self._minimal_pairs)

    def get_construction_bytesize(self) -> int:
        if not self._constructed or not self._retain_assignments or self._X is None or self._assignment is None:
            return 0
        return int(self._X.nbytes) + int(self._assignment.nbytes)