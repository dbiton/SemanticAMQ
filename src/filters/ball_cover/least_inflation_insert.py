import numpy as np
import faiss
from typing import List, Tuple
from filters.filter import Filter
from filters.ball_cover.ball_interfaces import InsertInterface


class LeastInflationInsert(InsertInterface):
    """Like NearestInsert, but absorbs into whichever ball minimizes the
    actual objective delta max(r_i, dist_i)**d - r_i**d, not whichever is
    nearest -- a far-but-huge ball can beat a near-but-tiny one whenever
    growing the tiny one would cost more volume.

    Computed in log space, scaled by each point's own largest candidate
    radius: r**d alone overflows float64 well inside float32's range at
    d ~ 384, but the scaled terms stay in (0, 1] and underflow safely
    instead, preserving the argmin exactly."""

    def ensure_capacity(self, required_cap: int) -> None:
        pass


    def reset_buffers(self, capacity: int) -> None:
        pass


    def add_sphere(self, filter: Filter, center: np.ndarray, radius: float) -> None:
        idx = filter._get_new_index()
        filter._all_centers[idx] = center
        filter._all_radiuses[idx] = radius
        filter._num_spheres += 1


    @staticmethod
    def _exhaustive_search(centers: np.ndarray, queries: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Via a throwaway faiss index, not a raw (m, k, d) numpy broadcast
        -- the latter materializes a full dense tensor (~150GB at
        20000x5000x384) where this holds only the (m, k) result."""
        dim = centers.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(centers)
        d2, labels = index.search(queries, len(centers))
        return np.sqrt(np.maximum(d2, 0.0)), labels


    def insert(self, filter: Filter, vectors: np.ndarray) -> None:
        dim = filter.dim
        X = np.ascontiguousarray(vectors, dtype=np.float32)
        centers = filter.all_centers
        radii = filter.all_radiuses

        dist, labels = self._exhaustive_search(centers, X)
        dist_64 = dist.astype(np.float64)
        r_64 = radii[labels].astype(np.float64)
        r_new = np.maximum(r_64, dist_64)

        M = r_new.max(axis=1, keepdims=True)
        safe_M = np.where(M > 0, M, 1.0)
        with np.errstate(divide="ignore"):
            scaled_new = np.exp(dim * np.log(np.maximum(r_new, 0.0) / safe_M))
            scaled_old = np.exp(dim * np.log(np.maximum(r_64, 0.0) / safe_M))
        delta = np.where(M > 0, scaled_new - scaled_old, 0.0)

        col = delta.argmin(axis=1)
        row = np.arange(len(X))
        ball = labels[row, col]

        new_radii = radii.copy()
        np.maximum.at(new_radii, ball, dist[row, col].astype(np.float32))

        filter._all_radiuses[:filter._num_spheres] = new_radii


    def remove(self, filter: Filter, vectors: np.ndarray) -> List[bool]:
        raise NotImplementedError("LeastInflationInsert does not support remove operation.")


    def get_bytesize(self, filter: Filter) -> int:
        return 0