from typing import List
import sys
import numpy as np
from filters.filter import Filter
import faiss


class NaiveFilter(Filter):
    def __init__(self, dim: int, distance_threshold: float):
        super().__init__(dim, None)
        self.distance_threshold = distance_threshold
        self.index = None
        self.vectors = None  # Store vectors for potential re-indexing

    def construct(self, vectors: np.ndarray):
        self.index = faiss.IndexFlatL2(self.dim)
        self.vectors = np.array(vectors, dtype=np.float32)
        self.index.add(self.vectors)

    def get_exact_matching_fpr(self) -> float:
        return 0

    def query(self, vectors: np.ndarray) -> List[bool]:
        if self.index is None:
            raise ValueError("Filter has not been constructed with any vectors.")

        D, I = self.index.search(vectors, 1)
        return np.logical_and(I[:, 0] != -1, D[:, 0] < self.distance_threshold)

    def add(self, vectors: np.ndarray) -> None:
        """
        Add vectors to the FAISS index after construction.
        
        Appends new vectors to the index.
        """
        if self.index is None:
            raise ValueError("Filter must be constructed before adding vectors.")
        
        new_vectors = np.array(vectors, dtype=np.float32)
        self.index.add(new_vectors)
        if self.vectors is not None:
            self.vectors = np.vstack([self.vectors, new_vectors])
        else:
            self.vectors = new_vectors

    def remove(self, vectors: np.ndarray) -> List[bool]:
        """
        Remove vectors from the filter by rebuilding the index.
        
        Since FAISS doesn't natively support deletion, this method identifies
        and removes matching vectors by reconstructing the index.
        
        Returns a list of booleans indicating whether each vector was present before removal.
        """
        if self.index is None or self.vectors is None:
            raise ValueError("Filter has not been constructed with any vectors.")
        
        query_vectors = np.array(vectors, dtype=np.float32)
        results: List[bool] = []
        
        # For each query vector, find and mark matching vectors for removal
        remaining_mask = np.ones(len(self.vectors), dtype=bool)
        
        D, I = self.index.search(query_vectors, 1)
        for i, (dist, idx) in enumerate(zip(D, I)):
            if idx[0] != -1 and dist[0] < self.distance_threshold:
                remaining_mask[idx[0]] = False
                results.append(True)
            else:
                results.append(False)
        
        # Rebuild the index with remaining vectors
        self.vectors = self.vectors[remaining_mask]
        self.index = faiss.IndexFlatL2(self.dim)
        if len(self.vectors) > 0:
            self.index.add(self.vectors)
        
        return results

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
        # Each vector is dim floats (4 bytes per float32)
        num_vectors = self.index.ntotal
        vector_size = num_vectors * self.dim * 4  # 4 bytes per float32
        return index_size + vector_size
