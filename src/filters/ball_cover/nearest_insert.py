import numpy as np
import faiss
from typing import List, Tuple
from filters.filter import Filter
from filters.ball_cover.ball_interfaces import InsertInterface


class NearestInsert(InsertInterface):
    """Absorb each new vector into the nearest existing ball by raw
    distance, growing its radius if needed -- never creates new balls.
    Cheap and predictable, but nearest != cheapest to grow: a far-but-huge
    ball may need no growth while the nearest one is a tight singleton about
    to blow up by dist**d (see LeastInflationInsert)."""

    def initialize_buffers(self, filter: Filter, spheres_indices: np.ndarray, centers: np.ndarray, radii: np.ndarray) -> None:
        pass


    def ensure_capacity(self, required_cap: int) -> None:
        pass


    def reset_buffers(self, capacity: int) -> None:
        pass


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

        if filter._num_spheres == 0:
            filter.construct(vectors)
            return

        X = np.ascontiguousarray(vectors, dtype=np.float32)
        active_indices = filter.all_spheres
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

        filter._all_radiuses[active_indices] = new_radii


    def remove(self, filter: Filter, vectors: np.ndarray) -> List[bool]:
        raise NotImplementedError("NearestInsert does not support remove operation.")


    def get_bytesize(self, filter: Filter) -> int:
        return 0