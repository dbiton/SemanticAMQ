from typing import List

from filters.filter import Filter
import numpy as np
import faiss


class NaiveFilter(Filter):
    def __init__(self, dim: int, distance_threshold: float):
        super().__init__(dim)
        self.distance_threshold = distance_threshold
        self.index = None

    def construct(self, vectors: np.ndarray):
        self.index = faiss.IndexFlatL2(self.dim)
        self.index.add(vectors)

    def get_exact_matching_fpr(self) -> float:
        return 0

    def query(self, vectors: np.ndarray) -> List[bool]:
        if self.index is None:
            raise ValueError("Filter has not been constructed with any vectors.")

        D, I = self.index.search(vectors, 1)
        return np.logical_and(I[:, 0] != -1, D[:, 0] < self.distance_threshold)
