from typing import List
import sys
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

    def get_bytesize(self) -> int:
        """
        Returns the approximate memory size in bytes of the FAISS index.
        
        Uses sys.getsizeof for the index object and stored vectors.
        """
        if self.index is None:
            return 0
        # Get the size of the FAISS index object
        index_size = sys.getsizeof(self.index)
        # Estimate size of stored vectors
        # Each vector is dim floats (4 bytes per float)
        num_vectors = self.index.ntotal
        vector_size = num_vectors * self.dim * 4  # 4 bytes per float32
        return index_size + vector_size
