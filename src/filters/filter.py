from typing import List, Optional
import numpy as np


class Filter:
    def __init__(self, dim: int, max_memory: Optional[int] = None):
        self.dim = dim
        self.max_memory = max_memory

    def insert(self, vectors: np.ndarray) -> None:
        raise NotImplementedError("Subclasses must implement the insert method.")

    def construct(self, vectors: np.ndarray) -> None:
        raise NotImplementedError("Subclasses must implement the construct method.")

    def query(self, vectors: np.ndarray) -> List[bool]:
        raise NotImplementedError("Subclasses must implement the query method.")

    def add(self, vectors: np.ndarray) -> None:
        """
        Add vectors to an already-constructed filter.

        Not all filters support dynamic insertion. Filters that don't should raise NotImplementedError.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support dynamic insertion.")

    def get_bytesize(self) -> int:
        """
        Returns the approximate memory size in bytes used by the filter.

        Returns:
            The estimated number of bytes used by the filter structure.
        """
        raise NotImplementedError("Subclasses must implement get_bytesize method.")
