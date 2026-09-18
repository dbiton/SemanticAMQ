import numpy as np
import faiss
from typing import List, Tuple
from filters.filter import Filter
from filters.ball_cover.ball_interfaces import InsertInterface


class NearestInsert(InsertInterface):
    """Absorb each new vector into the nearest existing ball by raw
    distance, growing that ball's radius if needed. O(m*k*d), no new balls
    ever created -- max_count_balls is accepted but unused (there's nothing
    to cap). Cheap and predictable, but nearest != cheapest to grow: a
    far-but-huge ball may need no growth at all while the nearest one is a
    tight singleton about to blow up by a factor of dist**d (see
    insert_least_inflation)."""

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
        """Exact distance from every query to every center, via a throwaway
        faiss index rather than a raw (m, k, d) numpy broadcast -- the latter
        materializes a full dense tensor (a real problem at scale: e.g.
        20000 x 5000 x 384 floats is ~150GB) where this is O(m*k*d) compute
        with no more than a (m, k) result ever held in memory."""
        dim = centers.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(centers)
        d2, labels = index.search(queries, len(centers))
        return np.sqrt(np.maximum(d2, 0.0)), labels


    def insert(self, filter: Filter, vectors: np.ndarray) -> None:
        X = np.ascontiguousarray(vectors, dtype=np.float32)
        centers = filter.all_centers
        radii = filter.all_radiuses

        dist, labels = self._exhaustive_search(centers, X)
        surface = dist - radii[labels]

        cost = np.where(surface <= 0, -np.inf, dist)
        col = cost.argmin(axis=1)
        row = np.arange(len(X))
        ball = labels[row, col]

        new_radii = radii.copy()
        np.maximum.at(new_radii, ball, dist[row, col])

        filter._all_radiuses[:filter._num_spheres] = new_radii


    def remove(self, filter: Filter, vectors: np.ndarray) -> List[bool]:
        raise NotImplementedError("NearestInsert does not support remove operation.")


    def get_bytesize(self, filter: Filter) -> int:
        return 0