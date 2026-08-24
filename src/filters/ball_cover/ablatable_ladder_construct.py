import time, faiss
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from filters import ball_geometry
from filters.ball_cover.ball_interfaces import ConstructInterface, Filter


class AblatableLadderConstruct(ConstructInterface):
    """
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
    """

    _BYTES_PER_FLOAT32 = 4
    _BYTES_PER_FLOAT64 = 8
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

    def __init__(self, seed: int = _DEFAULT_SEED, complexity: int = _DEFAULT_COMPLEXITY,
                 quantize: bool = False, quantizer_candidates: Optional[Tuple[str, ...]] = None,
                 retain_assignments: bool = False):
        self._seed = seed
        self._complexity_level = complexity
        self._retain_assignments = retain_assignments

        # construction state
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

    # ------------------------------------------------------------------ complexity policy

    def set_complexity(self, level: int) -> None:
        if level not in (1, 2):
            raise ValueError(f"complexity level must be in 1..2, got {level}")
        self._complexity_level = level

    # ------------------------------------------------------------------ numerics (log-space)

    def _safe_radius(self, r: float) -> float:
        return ball_geometry.safe_radius(r)

    def _log_J(self, radii: np.ndarray, dim: int) -> float:
        return ball_geometry.log_J(radii, dim)

    def _log_J_tau(self, radii: np.ndarray, dim: int, tau: float) -> float:
        return ball_geometry.log_J_tau(radii, dim, tau)

    # ------------------------------------------------------------------ chunked distances

    def _chunked_assign(self, filter: Filter, vectors: np.ndarray, sq_norms: np.ndarray, centers: np.ndarray,
                        chunk: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Nearest-center assignment, chunked so the n x k distance matrix is bounded."""
        chunk = chunk or self._CHUNK_SIZE
        n = len(vectors)
        assignment = np.empty(n, dtype=np.int64)
        nearest_sq = np.empty(n, dtype=np.float64)
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            sq_d = filter._sq_dists_to_centers(vectors[start:end], sq_norms[start:end], centers)
            a = np.argmin(sq_d, axis=1)
            assignment[start:end] = a
            nearest_sq[start:end] = sq_d[np.arange(end - start), a]
        return assignment, nearest_sq

    def _chunked_top2(self, filter: Filter, vectors: np.ndarray, sq_norms: np.ndarray, centers: np.ndarray,
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
            sq_d = filter._sq_dists_to_centers(vectors[start:end], sq_norms[start:end], centers)
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

    def _lloyd_refine(self, filter: Filter, vectors: np.ndarray, sq_norms: np.ndarray, centers: np.ndarray) -> np.ndarray:
        k = len(centers)

        for _ in range(self._LLOYD_ITERATIONS):
            assignment, nearest_sq = self._chunked_assign(filter, vectors, sq_norms, centers)

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

    # ------------------------------------------------------------------ level 1: initial cover

    def _level1_initial_cover(self, filter: Filter, vectors: np.ndarray, sq_norms: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        stop_sq = float(filter.distance_threshold) ** 2
        dim = filter.dim
        rng = np.random.default_rng(self._seed)

        seed_idx, early_stop = self._kmeanspp_seed(vectors, sq_norms, k, stop_sq, rng)
        centers = vectors[seed_idx].copy()
        if not early_stop:
            centers = self._lloyd_refine(filter, vectors, sq_norms, centers)

        raw_assignment, _ = self._chunked_assign(filter, vectors, sq_norms, centers)
        unique_labels, assignment = np.unique(raw_assignment, return_inverse=True)
        num_spheres = len(unique_labels)
        groups = self._groups_from_assignment(assignment, num_spheres)

        centers_out = np.empty((num_spheres, dim), dtype=np.float32)
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

    def _level2_ball_refine(self, filter: Filter, vectors: np.ndarray, sq_norms: np.ndarray, centers: np.ndarray,
                            radii: np.ndarray, assignment: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        dim = filter.dim
        prev_log_J = self._log_J(radii, dim)
        accepted = 0

        for _ in range(self._BALL_REFINE_ITERATIONS):
            raw_assignment, _ = self._chunked_assign(filter, vectors, sq_norms, centers)
            unique_labels, new_assignment = np.unique(raw_assignment, return_inverse=True)
            num_spheres = len(unique_labels)
            groups = self._groups_from_assignment(new_assignment, num_spheres)

            new_centers = np.empty((num_spheres, dim), dtype=np.float32)
            new_radii = np.empty(num_spheres, dtype=np.float32)
            for i, idx in enumerate(groups):
                c, r = self._minimum_enclosing_ball(vectors[idx], sq_norms[idx])
                new_centers[i] = c
                new_radii[i] = r

            new_log_J = self._log_J(new_radii, dim)
            if new_log_J < prev_log_J - self._LOG_J_EPS:
                centers, radii, assignment = new_centers, new_radii, new_assignment.astype(np.int32)
                prev_log_J = new_log_J
                accepted += 1
            else:
                break

        return centers, radii, assignment, accepted

    # ------------------------------------------------------------------ instrumentation

    def _make_stage_record(self, level: int, name: str, wall_time: float,
                           radii: np.ndarray, accepted_ops: int, dim: int) -> Dict[str, Any]:
        return {
            "level": level,
            "name": name,
            "wall_time": wall_time,
            "ball_count": len(radii),
            "log_J": self._log_J(radii, dim),
            "radii_snapshot": np.asarray(radii, dtype=np.float32).copy(),
            "accepted_ops": accepted_ops,
        }

    def _record_stage(self, level: int, name: str, t_start: float, radii: np.ndarray, accepted_ops: int, dim: int) -> None:
        self._stage_stats.append(
            self._make_stage_record(level, name, time.perf_counter() - t_start, radii, accepted_ops, dim))

    def get_stats(self, dim: int, taus: Optional[List[float]] = None) -> Dict[str, Any]:
        taus = list(taus) if taus is not None else []
        levels = []
        for stage in self._stage_stats:
            levels.append({
                "level": stage["level"],
                "name": stage["name"],
                "wall_time": stage["wall_time"],
                "ball_count": stage["ball_count"],
                "log_J": stage["log_J"],
                "log_J_tau": {t: self._log_J_tau(stage["radii_snapshot"], dim, t) for t in taus},
                "accepted_ops": stage["accepted_ops"],
            })
        return {"levels": levels, "quantization": self._quantizer_stats}

    # ------------------------------------------------------------------ filter construct

    def construct_filter(self, filter: Filter, vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self._stage_stats = []
        self._quantizer_name = None
        self._quantizer_bytes_per_vector = None
        self._quantizer_stats = None
        levels_to_run = min(self._complexity_level, 2)

        if len(vectors) == 0 or filter.max_spheres == 0:
            vectors_arr = np.ascontiguousarray(vectors, dtype=np.float32)
            self._X = vectors_arr if self._retain_assignments else None
            self._assignment = np.empty(0, dtype=np.int32) if self._retain_assignments else None
            for lvl in range(1, levels_to_run + 1):
                t0 = time.perf_counter()
                self._record_stage(lvl, self._LEVEL_NAMES[lvl], t0, np.empty(0, dtype=np.float32), 0, filter.dim)
            return np.empty((0, filter.dim), dtype=np.float32), np.empty(0, dtype=np.float32)

        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        sq_norms = np.einsum('ij,ij->i', vectors, vectors, dtype=np.float64)

        if self._quantize:
            centers, radii, assignment, chosen_k, stage_records, quant = \
                self._construct_with_quantization(filter, vectors, sq_norms)
            filter.max_spheres = chosen_k
            self._stage_stats = stage_records
            if quant is not None:
                self._quantizer_name = quant["quantizer"]
                self._quantizer_bytes_per_vector = quant["bytes_per_vector"]
                self._quantizer_stats = quant
        else:
            centers, radii, assignment, stage_records = self._build_cover(filter, vectors, sq_norms, filter.max_spheres)
            self._stage_stats = stage_records

        self._X = vectors if self._retain_assignments else None
        self._assignment = assignment.astype(np.int32) if self._retain_assignments else None
        
        return centers, radii

    def _build_cover(self, filter: Filter, vectors: np.ndarray, sq_norms: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
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
                stage_records.append(self._make_stage_record(lvl, self._LEVEL_NAMES[lvl], 0.0, radii, 0, filter.dim))
            return centers, radii, assignment, stage_records

        t0 = time.perf_counter()
        centers, radii, assignment = self._level1_initial_cover(filter, vectors, sq_norms, k)
        stage_records.append(self._make_stage_record(1, "initial_cover", time.perf_counter() - t0, radii, 0, filter.dim))

        if levels_to_run >= 2:
            t0 = time.perf_counter()
            centers, radii, assignment, accepted = self._level2_ball_refine(filter, vectors, sq_norms, centers, radii, assignment)
            stage_records.append(
                self._make_stage_record(2, "ball_refine", time.perf_counter() - t0, radii, accepted, filter.dim))

        return centers, radii, assignment, stage_records

    # ------------------------------------------------------------------ quantization

    def _construct_with_quantization(self, filter: Filter, vectors: np.ndarray, sq_norms: np.ndarray):
        """
        K (max_spheres) is treated as a memory *budget* -- "how many
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
        budget_k = filter.max_spheres
        dim = filter.dim
        # Per-sphere cost as get_bytesize() actually accounts it: center +
        # radius (float32) + count (int32) -- omitting the count term here
        # would let effective_k overshoot the true byte budget once it's
        # multiplied out over a much larger ball count.
        bytes_per_sphere_f32 = (dim + 2) * self._BYTES_PER_FLOAT32

        centers0, radii0, assignment0, stages0 = self._build_cover(filter, vectors, sq_norms, budget_k)
        best = {
            "log10_J": self._log_J(radii0.astype(np.float64), dim) / self._LN10,
            "k": budget_k, "centers": centers0, "radii": radii0,
            "assignment": assignment0, "stages": stages0, "quant": None,
        }

        for name in self._quantizer_candidates:
            qtype = getattr(faiss.ScalarQuantizer, name)
            bytes_per_vector = faiss.ScalarQuantizer(dim, qtype).code_size
            bytes_per_sphere_candidate = bytes_per_vector + 2 * self._BYTES_PER_FLOAT32  
            effective_k = max(1, budget_k * bytes_per_sphere_f32 // bytes_per_sphere_candidate)

            centers, radii, assignment, stages = self._build_cover(filter, vectors, sq_norms, effective_k)
            centers32 = np.ascontiguousarray(centers, dtype=np.float32)
            radii64 = radii.astype(np.float64)

            sq = faiss.ScalarQuantizer(dim, qtype)
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
            log10_j = self._log_J(quantized_radii, dim) / self._LN10

            if log10_j < best["log10_J"]:
                best = {
                    "log10_J": log10_j, "k": effective_k,
                    "centers": decoded.astype(np.float32),
                    "radii": quantized_radii.astype(np.float32),
                    "assignment": assignment, "stages": stages,
                    "quant": {
                        "quantizer": name, "bytes_per_vector": int(codes.shape[1]),
                        "log10_J": log10_j, "log10_J_unquantized": self._log_J(radii64, dim) / self._LN10,
                        "mean_center_error": float(error.mean()), "max_center_error": float(error.max()),
                        "budget_k": budget_k, "effective_k": effective_k,
                    },
                }

        return (best["centers"], best["radii"], best["assignment"], best["k"], best["stages"], best["quant"])

    def _apply_quantization_inplace(self, filter: Filter) -> Optional[Dict[str, Any]]:
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
        if filter._num_spheres == 0:
            return None

        centers = filter.all_centers.astype(np.float32)
        radii = filter.all_radiuses.astype(np.float64)
        log10_j_plain = self._log_J(radii, filter.dim) / self._LN10

        best = None
        for name in self._quantizer_candidates:
            qtype = getattr(faiss.ScalarQuantizer, name)
            sq = faiss.ScalarQuantizer(filter.dim, qtype)
            sq.train(centers)
            codes = sq.compute_codes(centers)
            decoded = sq.decode(codes).astype(np.float64)
            error = np.linalg.norm(centers.astype(np.float64) - decoded, axis=1)

            log10_j = self._log_J(radii + error, filter.dim) / self._LN10
            if best is None or log10_j < best["log10_J"]:
                best = {
                    "name": name, "log10_J": log10_j,
                    "bytes_per_vector": int(codes.shape[1]),
                    "decoded": decoded.astype(np.float32),
                    "error": error,
                }

        if best is None or log10_j_plain <= best["log10_J"]:
            return None  # unquantized float32 already minimizes volume; leave centers untouched

        current_spheres = filter.all_spheres
        filter._all_centers[current_spheres] = best["decoded"]
        filter._all_radiuses[current_spheres] = (radii + best["error"]).astype(np.float32)

        return {
            "quantizer": best["name"],
            "bytes_per_vector": best["bytes_per_vector"],
            "log10_J": best["log10_J"],
            "log10_J_unquantized": log10_j_plain,
            "mean_center_error": float(best["error"].mean()),
            "max_center_error": float(best["error"].max()),
        }

    def quantize_centers(self, filter: Filter) -> Optional[Dict[str, Any]]:
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
        if filter._num_spheres == 0:
            return None

        if self._X is not None and len(self._X) > 0:
            sq_norms = np.einsum('ij,ij->i', self._X, self._X, dtype=np.float64)
            centers, radii, assignment, chosen_k, stage_records, quant = \
                self._construct_with_quantization(filter, self._X, sq_norms)
            filter.max_spheres = chosen_k
            num_spheres = len(centers)
            filter._reset_buffers(num_spheres)
            filter._all_centers[:num_spheres] = centers
            filter._all_radiuses[:num_spheres] = radii + filter.distance_threshold
            filter._all_spheres[:num_spheres] = np.arange(num_spheres, dtype=np.int32)
            filter._num_spheres = num_spheres
            self._assignment = assignment.astype(np.int32)
            self._stage_stats = stage_records
        else:
            quant = self._apply_quantization_inplace(filter)

        self._quantizer_name = quant["quantizer"] if quant else None
        self._quantizer_bytes_per_vector = quant["bytes_per_vector"] if quant else None
        self._quantizer_stats = quant
        return quant

    # ------------------------------------------------------------------ memory / misc

    def get_bytesize(self) -> int:
        if not self._retain_assignments or self._X is None or self._assignment is None:
            return 0
        return int(self._X.nbytes) + int(self._assignment.nbytes)