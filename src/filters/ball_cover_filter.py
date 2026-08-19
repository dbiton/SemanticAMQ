import sys, heapq, time
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

    Construction is a cumulative, ablatable ladder of up to 2 levels
    (`set_complexity`, default 2), each one refining the cover produced by the
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

    A further refinement is opt-in and runs *after* `construct()` rather
    than as a rung of the complexity ladder:

      - Quantization: compresses the stored centroids with a FAISS
        ScalarQuantizer (`quantize=True` at construction, or a manual
        `quantize_centers()` call). Every candidate in `_QUANTIZER_CANDIDATES`
        is tried; each candidate's own per-center reconstruction error is
        folded into that ball's radius (never subtracted), and whichever
        candidate yields the smallest resulting total covering volume
        log10 J(C) = log10 sum_i r_i^d is kept -- so the covering invariant
        holds regardless of which level wins, and the choice is driven
        directly by the same geometric objective the construction ladder
        itself minimizes.

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
    _LLOYD_ITERATIONS = 10
    _BALL_REFINE_ITERATIONS = 10
    _KMEANS2_ITERATIONS = 10
    _CHUNK_SIZE = 20_000
    _LOG_J_EPS = 1e-9
    _DEFAULT_SEED = 42
    _DEFAULT_COMPLEXITY = 2
    _LEVEL_NAMES = {1: "initial_cover", 2: "ball_refine"}
    _LN10 = float(np.log(10.0))
    _QUANTIZER_CANDIDATES = ("QT_fp16", "QT_8bit", "QT_6bit", "QT_4bit")

    def __init__(self, dim: int, distance_threshold: float, max_memory: Optional[int] = None,
                 max_spheres: Optional[int] = None, seed: int = _DEFAULT_SEED,
                 quantize: bool = False, quantizer_candidates: Optional[Tuple[str, ...]] = None):
        super().__init__(dim, lsh=None, max_memory=max_memory)
        self.distance_threshold = distance_threshold
        self.max_spheres = (max_spheres if max_spheres is not None
                            else self._calculate_max_spheres(self.max_memory))
        self._seed = seed
        self._constructed = False
        self._complexity_level = self._DEFAULT_COMPLEXITY

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

    # ------------------------------------------------------------------ complexity policy

    def set_complexity(self, level: int) -> None:
        if level not in (1, 2):
            raise ValueError(f"complexity level must be in 1..2, got {level}")
        if self._constructed:
            self._constructed = False
        self._complexity_level = level

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
        if len(self.all_centers) == 0:
            return -np.inf, -np.inf
        return estimate_hypersphere_volume(self.all_centers, self.all_radiuses, return_log=True)

    # ------------------------------------------------------------------ level 1: initial cover

    def _level1_initial_cover(self, vectors: np.ndarray, sq_norms: np.ndarray,
                              k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        for i, idx in enumerate(groups):
            pts = vectors[idx]
            c32 = pts.mean(axis=0, dtype=np.float64).astype(np.float32)
            c = c32.astype(np.float64)  # radius must match the center as actually stored
            d = np.sqrt(np.maximum(np.einsum('ij,ij->i', pts.astype(np.float64) - c, pts.astype(np.float64) - c), 0.0))
            centers_out[i] = c32
            radii_out[i] = self._safe_radius(d.max())

        return centers_out, radii_out, assignment.astype(np.int32)

    # ------------------------------------------------------------------ level 2: ball refinement

    def _level2_ball_refine(self, vectors: np.ndarray, sq_norms: np.ndarray, centers: np.ndarray,
                            radii: np.ndarray, assignment: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
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
        retain = bool(retain_assignments)
        levels_to_run = min(self._complexity_level, 2)

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
        Run the (up to 2-level) construction ladder for an arbitrary ball
        budget k against already-preprocessed vectors, touching no instance
        state. Used both for a plain (non-quantized) construct() and, once
        per quantization candidate, to rebuild at that candidate's own larger
        effective_k (see _construct_with_quantization). Returns
        centers/radii/counts/assignment plus this build's own stage records,
        the same shape _record_stage would append to self._stage_stats for a
        normal top-level build.
        """
        n = len(vectors)
        levels_to_run = min(self._complexity_level, 2)
        stage_records: List[Dict[str, Any]] = []

        if k >= n:
            centers = vectors.copy()
            radii = np.zeros(n, dtype=np.float32)
            assignment = np.arange(n, dtype=np.int32)
            for lvl in range(1, levels_to_run + 1):
                stage_records.append(self._make_stage_record(lvl, self._LEVEL_NAMES[lvl], 0.0, radii, 0))
            return centers, radii, assignment, stage_records

        t0 = time.perf_counter()
        centers, radii, assignment = self._level1_initial_cover(vectors, sq_norms, k)
        stage_records.append(self._make_stage_record(1, "initial_cover", time.perf_counter() - t0, radii, 0))

        if levels_to_run >= 2:
            t0 = time.perf_counter()
            centers, radii, assignment, accepted = self._level2_ball_refine(
                vectors, sq_norms, centers, radii, assignment)
            stage_records.append(
                self._make_stage_record(2, "ball_refine", time.perf_counter() - t0, radii, accepted))

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

    # ------------------------------------------------------------------ streaming / dynamic merge

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