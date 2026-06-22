from ctypes import sizeof
from typing import List, Optional
import numpy as np


class KMEBFilter:
    def __init__(self, dim: int, max_memory: Optional[int] = None, max_balls: Optional[int] = None):
        self.dim = dim
        self.lsh = None
        if max_memory is None and max_balls is None:
            raise ValueError("Either max_memory or max_balls must be specified.")
        elif max_memory is not None and max_balls is not None:
            raise ValueError("Only one of max_memory and max_balls can be specified.")
        elif max_balls is not None:
            self.max_balls = max_balls
        else:
            self.max_balls = self.calculate_max_balls(max_memory)

    def calculate_max_balls(self, max_memory: int) -> int:
        radius_memory_usage = sizeof(float)
        centroid_memory_usage = sizeof(float) * self.dim
        ball_memory_usage = radius_memory_usage + centroid_memory_usage
        return max_memory // ball_memory_usage

    def get_exact_matching_fpr(self) -> float:
        """
        Dummy implementation: assumes 0 false positives for exact matching.
        """
        return 0.0

    def construct(self, vectors: np.ndarray) -> None:
        """
        Construct the dummy filter by simply keeping the first max_balls vectors.
        """
        limit = self.max_balls if self.max_balls is not None else len(vectors)
        self.stored_vectors = np.copy(vectors[:limit])

    def insert(self, vectors: np.ndarray) -> None:
        """
        Dummy implementation: deliberately ignore new inserts.
        """
        pass

    def query(self, vectors: np.ndarray) -> List[bool]:
        """
        Dummy query implementation: checks for exact vector matches 
        against the stored vectors.
        """
        if len(self.stored_vectors) == 0:
            return [False] * len(vectors)
            
        # Broadcast equality check across vectors and stored_vectors
        # Shape: (num_queries, num_stored, dim) -> all on axis 2 -> any on axis 1
        matches = np.any(np.all(vectors[:, None, :] == self.stored_vectors[None, :, :], axis=2), axis=1)
        return matches.tolist()

    def get_bytesize(self) -> int:
        """
        Returns the memory size of the stored numpy array.
        """
        return self.stored_vectors.nbytes