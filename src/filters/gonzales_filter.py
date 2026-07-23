import numpy as np
from typing import List, Tuple, Optional
from filters.filter import Filter
from hyperspheres_volume import estimate_hypersphere_volume


class GonzalezMEBFilter(Filter):
    """
    Duplicate filter that covers the input vectors with a bounded number of
    hyperspheres, each admitting a (1+eps)-approximate Minimum Enclosing
    Ball (MEB) of its cluster plus distance_threshold. Radii are exact max
    distances from the final centers to every member point, so the filter
    admits false positives but never false negatives.

    Clustering: centers are seeded with k-means++ (probability proportional
    to squared distance from the nearest existing center), which is far
    less sensitive to outliers than farthest-point (Gonzalez) seeding, then
    refined with a few Lloyd iterations. Seeding stops early once every
    point already lies within distance_threshold of a center.

    MEB: each cluster's ball is fit with a coreset-based Badoiu-Clarkson
    approximation — a cheap pass over the full cluster builds a small
    coreset of "farthest" points touched during the iteration, then many
    more (nearly free) refinement passes run over just that coreset. The
    final radius is always the exact max distance from the refined center
    to every point in the full cluster, so approximation only affects how
    tight the ball is, never correctness.

    Storage: centers/radii/counts live in a single preallocated buffer
    (capacity grows by amortized doubling up to max_spheres, exactly like
    a dynamic array), so add()/remove() never reallocate or copy the whole
    structure. There is no cached FAISS index: containment and nearest-
    center queries are plain broadcasted matmuls against the buffer, which
    keeps FAISS from holding a second copy of the same data in memory.
    """

    _BYTES_PER_FLOAT32 = 4
    _MEB_CORESET_ITERATIONS = 30
    _MEB_REFINE_ITERATIONS = 200
    _LLOYD_ITERATIONS = 10
    _DEFAULT_SEED = 42

    def __init__(self, dim: int, distance_threshold: float, max_memory: Optional[int] = None,
                 max_spheres: Optional[int] = None, seed: int = _DEFAULT_SEED):
        super().__init__(dim, lsh=None, max_memory=max_memory)
        self.distance_threshold = distance_threshold
        self.max_spheres = (max_spheres if max_spheres is not None
                            else self._calculate_max_spheres(self.max_memory))
        self._seed = seed
        self._constructed = False

        self._center_buf = np.empty((0, dim), dtype=np.float32)
        self._radius_buf = np.empty((0,), dtype=np.float32)
        self._count_buf = np.empty((0,), dtype=np.int32)
        self._capacity = 0
        self._size = 0

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

    # ------------------------------------------------------------------ sizing

    def _calculate_max_spheres(self, max_memory: int, gamma: float = 4.0) -> int:
        bytes_per_sphere = (self.dim + 1) * self._BYTES_PER_FLOAT32
        return max(1, int(max_memory // (bytes_per_sphere * gamma)))

    def _reset_buffers(self, capacity: int) -> None:
        self._center_buf = np.empty((capacity, self.dim), dtype=np.float32)
        self._radius_buf = np.empty(capacity, dtype=np.float32)
        self._count_buf = np.empty(capacity, dtype=np.int32)
        self._capacity = capacity
        self._size = 0

    def _grow_capacity(self) -> None:
        new_capacity = min(self.max_spheres, max(self._capacity * 2, self._capacity + 1))
        if new_capacity <= self._capacity:
            return
        center_buf = np.empty((new_capacity, self.dim), dtype=np.float32)
        radius_buf = np.empty(new_capacity, dtype=np.float32)
        count_buf = np.empty(new_capacity, dtype=np.int32)
        center_buf[:self._size] = self._center_buf[:self._size]
        radius_buf[:self._size] = self._radius_buf[:self._size]
        count_buf[:self._size] = self._count_buf[:self._size]
        self._center_buf, self._radius_buf, self._count_buf = center_buf, radius_buf, count_buf
        self._capacity = new_capacity

    def _append_sphere(self, center: np.ndarray, radius: float, count: int) -> None:
        if self._size == self._capacity:
            self._grow_capacity()
            if self._size == self._capacity:
                return  # at max_spheres capacity already, nothing to grow into
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

    # ------------------------------------------------------------------ clustering

    def _sq_dists_to_centers(self, vectors: np.ndarray, sq_norms: np.ndarray, centers: np.ndarray) -> np.ndarray:
        c_sq = np.einsum('ij,ij->i', centers, centers)
        cross = vectors @ centers.T
        return sq_norms[:, None] - 2.0 * cross + c_sq[None, :]

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
        n = len(vectors)
        k = len(centers)

        for _ in range(self._LLOYD_ITERATIONS):
            sq_d = self._sq_dists_to_centers(vectors, sq_norms, centers)
            assignment = np.argmin(sq_d, axis=1)
            nearest_sq = sq_d[np.arange(n), assignment]

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

    def _kmeans_clustering(self, vectors: np.ndarray, sq_norms: np.ndarray,
                           k: int) -> Tuple[np.ndarray, int]:
        stop_sq = float(self.distance_threshold) ** 2
        rng = np.random.default_rng(self._seed)

        seed_idx, early_stop = self._kmeanspp_seed(vectors, sq_norms, k, stop_sq, rng)
        centers = vectors[seed_idx].copy()
        if not early_stop:
            centers = self._lloyd_refine(vectors, sq_norms, centers)

        sq_d = self._sq_dists_to_centers(vectors, sq_norms, centers)
        raw_assignment = np.argmin(sq_d, axis=1)
        # collapse to a contiguous 0..num_clusters-1 range in case any seed
        # ended up with zero members after refinement
        unique_labels, assignment = np.unique(raw_assignment, return_inverse=True)
        return assignment.astype(np.int32), len(unique_labels)

    # ------------------------------------------------------------------ MEB

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

        sq_d = np.maximum(sq_norms - 2.0 * (points @ center) + center @ center, 0.0)
        return center.astype(np.float32), float(np.sqrt(sq_d.max()))

    def _log_volume_bounds(self) -> Tuple[float, float]:
        if self._size == 0:
            return -np.inf, -np.inf
        return estimate_hypersphere_volume(self.all_spheres, self.all_radiuses, return_log=True)

    # ------------------------------------------------------------------ public API

    def construct(self, vectors: np.ndarray) -> Tuple[float, float]:
        if len(vectors) == 0 or self.max_spheres == 0:
            self._reset_buffers(0)
            self._constructed = True
            return self._log_volume_bounds()

        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        n = len(vectors)

        if self.max_spheres >= n:
            self._reset_buffers(n)
            self._center_buf[:] = vectors
            self._radius_buf[:] = self.distance_threshold
            self._count_buf[:] = 1
            self._size = n
            self._constructed = True
            return self._log_volume_bounds()

        sq_norms = np.einsum('ij,ij->i', vectors, vectors)
        assignment, num_spheres = self._kmeans_clustering(vectors, sq_norms, self.max_spheres)

        order = np.argsort(assignment, kind='stable')
        sorted_assignment = assignment[order]
        boundaries = np.flatnonzero(np.diff(sorted_assignment)) + 1
        groups = np.split(order, boundaries)

        self._reset_buffers(num_spheres)
        for i, member_idx in enumerate(groups):
            center, meb_radius = self._minimum_enclosing_ball(vectors[member_idx], sq_norms[member_idx])
            self._center_buf[i] = center
            self._radius_buf[i] = meb_radius + self.distance_threshold
            self._count_buf[i] = len(member_idx)
        self._size = num_spheres

        self._constructed = True
        return self._log_volume_bounds()

    def query(self, vectors: np.ndarray) -> List[bool]:
        if not self._constructed:
            raise ValueError("GonzalezMEBFilter has not been constructed.")

        if self._size == 0 or len(vectors) == 0:
            return [False] * len(vectors)

        q = np.ascontiguousarray(vectors, dtype=np.float32)
        centers = self.all_spheres
        radii = self.all_radiuses

        sq_d = self._sq_dists_to_centers(q, np.einsum('ij,ij->i', q, q), centers)
        within = sq_d <= (radii[None, :].astype(np.float64) ** 2)
        return np.any(within, axis=1).tolist()

    def add(self, vectors: np.ndarray) -> None:
        if not self._constructed:
            self.construct(vectors)
            return

        if len(vectors) == 0:
            return

        new_vectors = np.ascontiguousarray(vectors, dtype=np.float32)

        for q in new_vectors:
            if self.max_spheres == 0:
                continue

            if self._size == 0:
                self._append_sphere(q, self.distance_threshold, 1)
                continue

            centers = self.all_spheres
            diff = centers - q
            sq_d = np.einsum('ij,ij->i', diff, diff)
            best = int(sq_d.argmin())
            best_dist = float(np.sqrt(sq_d[best]))
            r_best = float(self._radius_buf[best])

            if best_dist <= r_best:
                self._count_buf[best] += 1
            elif best_dist <= r_best + 2.0 * self.distance_threshold or self._size >= self.max_spheres:
                c_old = self._center_buf[best].copy()
                count = self._count_buf[best]
                c_new = ((c_old * count) + q) / (count + 1)
                shift = float(np.linalg.norm(c_old - c_new))
                dist_q = float(np.linalg.norm(q - c_new))
                self._center_buf[best] = c_new
                self._radius_buf[best] = max(r_best + shift, dist_q + self.distance_threshold)
                self._count_buf[best] = count + 1
            else:
                self._append_sphere(q, self.distance_threshold, 1)

    def get_volume(self) -> float:
        return 0.0

    def insert(self, vectors: np.ndarray) -> None:
        self.add(vectors)

    def remove(self, vectors: np.ndarray) -> List[bool]:
        if self._size == 0 or len(vectors) == 0:
            return [False] * len(vectors)

        q = np.ascontiguousarray(vectors, dtype=np.float32)
        centers = self.all_spheres
        sq_d = self._sq_dists_to_centers(q, np.einsum('ij,ij->i', q, q), centers)
        nearest = np.argmin(sq_d, axis=1)
        nearest_sq_d = np.maximum(sq_d[np.arange(len(q)), nearest], 0.0)
        in_sphere = nearest_sq_d <= self._radius_buf[nearest] ** 2

        if np.any(in_sphere):
            decrements = np.bincount(nearest[in_sphere], minlength=self._size)
            self._count_buf[:self._size] -= decrements.astype(self._count_buf.dtype)
            self._compact()

        return in_sphere.tolist()

    def get_exact_matching_fpr(self) -> float:
        return 0.0

    def get_bytesize(self) -> int:
        if not self._constructed:
            return 0
        return self.all_spheres.nbytes + self.all_radiuses.nbytes + self.sphere_counts.nbytes
