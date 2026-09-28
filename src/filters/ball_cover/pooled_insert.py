import numpy as np
import faiss
from typing import List, Tuple, Optional
from filters.filter import Filter
from filters.ball_cover.ball_interfaces import InsertInterface


class PooledInsert(InsertInterface):
    """Pooled/Buffered Gonzales Insertion Strategy.

    Maintains incoming vectors inside a designated pool budget within the filter
    (e.g., reserving ~20% of max_spheres for insertions). Incoming uncovered
    vectors are added immediately as zero-radius spheres so they are immediately
    queryable without violating the zero-false-negative guarantee.

    When the total sphere count reaches or exceeds max_spheres, a
    Gonzales re-clustering step compresses all active spheres back down to
    target_spheres (by default max_construct_spheres), preserving
    sphere radius uncertainty during the re-clustering pass.
    """

    def __init__(self, target_spheres: Optional[int] = None) -> None:
        self.target_spheres = target_spheres


    def initialize_buffers(self, filter: Filter, spheres_indices: np.ndarray, centers: np.ndarray, radii: np.ndarray) -> None:
        pass


    def ensure_capacity(self, required_cap: int) -> None:
        pass


    def reset_buffers(self, capacity: int) -> None:
        pass


    @staticmethod
    def _gonzalez_select_with_uncertainty(points: np.ndarray, uncertainty: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Mirrors GonzalesConstruct's greedy-selection-then-exact-radii
        algorithm; the only change is `reach = distance-to-center + that
        point's own uncertainty` instead of plain distance for each
        cluster's final radius (see class docstring)."""
        n = points.shape[0]
        dim = points.shape[1]
        k = min(k, n)
        if k <= 0:
            return np.empty((0, dim), dtype=np.float32), np.empty(0, dtype=np.float32)
        if n <= k:
            return points.copy(), uncertainty.copy()

        sq = np.einsum("ij,ij->i", points, points)

        chosen = np.empty(k, dtype=np.int64)
        min_d2 = np.empty(n, dtype=np.float32)
        d2 = np.empty(n, dtype=np.float32)

        chosen[0] = 0
        np.dot(points, points[0], out=min_d2)
        min_d2 *= -2.0
        min_d2 += sq
        min_d2 += sq[0]

        for j in range(1, k):
            c = int(min_d2.argmax())
            if min_d2[c] <= 0.0:
                chosen = chosen[:j]
                k = j
                break
            chosen[j] = c

            np.dot(points, points[c], out=d2)
            d2 *= -2.0
            d2 += sq
            d2 += sq[c]
            np.minimum(min_d2, d2, out=min_d2)

        centers = np.ascontiguousarray(points[chosen])

        index = faiss.IndexFlatL2(dim)
        index.add(centers)
        d2_exact, labels = index.search(points, 1)
        d_exact = np.sqrt(np.maximum(d2_exact.ravel(), 0.0))
        labels = labels.ravel()

        reach = (d_exact + uncertainty).astype(np.float32)

        order = np.argsort(labels, kind="stable")
        counts = np.bincount(labels, minlength=k)
        starts = np.zeros(k, dtype=np.int64)
        np.cumsum(counts[:-1], out=starts[1:])
        nonempty = counts > 0

        radii = np.zeros(k, dtype=np.float32)
        radii[nonempty] = np.maximum.reduceat(reach[order], starts[nonempty])
        return centers, radii


    def _get_target_spheres(self, filter: Filter) -> int:
        if self.target_spheres is not None:
            return min(self.target_spheres, filter.max_spheres)
        return max(1, filter.max_construct_spheres)


    def _rebuild(self, filter: Filter) -> None:
        target_k = self._get_target_spheres(filter)
        points = np.ascontiguousarray(filter.all_centers, dtype=np.float32)
        uncertainty = np.asarray(filter.all_radiuses, dtype=np.float32)

        new_centers, new_radii = self._gonzalez_select_with_uncertainty(
            points, uncertainty, target_k
        )

        filter._remove_spheres(filter.all_spheres)
        filter._add_spheres(new_centers, new_radii)


    @staticmethod
    def _exhaustive_search(centers: np.ndarray, queries: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(centers) == 0:
            return np.empty((len(queries), 0), dtype=np.float32), np.empty((len(queries), 0), dtype=np.int64)
        dim = centers.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(centers)
        d2, labels = index.search(queries, len(centers))
        return np.sqrt(np.maximum(d2, 0.0)), labels


    def insert(self, filter: Filter, vectors: np.ndarray) -> None:
        if len(vectors) == 0:
            return

        X = np.ascontiguousarray(vectors, dtype=np.float32)

        # Filter out vectors already covered by active spheres
        if filter._num_spheres > 0:
            centers = filter.all_centers
            radii = filter.all_radiuses
            dist, labels = self._exhaustive_search(centers, X)
            is_already_covered = np.any(dist <= radii[labels], axis=1)
            X = X[~is_already_covered]

        if len(X) == 0:
            return

        # Add uncovered vectors and rebuild when max capacity is hit
        pool_capacity = max(1, filter.max_spheres - self._get_target_spheres(filter))

        offset = 0
        while offset < len(X):
            # Rebuild if capacity limit is reached
            if filter._num_spheres >= filter.max_spheres:
                self._rebuild(filter)

            available_cap = filter.max_spheres - filter._num_spheres
            if available_cap <= 0:
                break

            chunk_size = min(pool_capacity, available_cap, len(X) - offset)
            chunk = X[offset : offset + chunk_size]
            filter._add_spheres(chunk, np.zeros(len(chunk), dtype=np.float32))
            offset += chunk_size


    def remove(self, filter: Filter, vectors: np.ndarray) -> List[bool]:
        raise NotImplementedError("PooledInsert does not support remove operation.")


    def get_bytesize(self, filter: Filter) -> int:
        return 0