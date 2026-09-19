from typing import List, Optional
import numpy as np


class Filter:
    def __init__(self, dim: int, max_memory: Optional[int] = None):
        self.dim = dim
        self.max_memory = max_memory

    def insert(self, vectors: np.ndarray) -> None:
        """Not all filters support this; those that don't raise NotImplementedError."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support dynamic insertion.")

    def construct(self, vectors: np.ndarray) -> None:
        raise NotImplementedError("Subclasses must implement the construct method.")

    def query(self, vectors: np.ndarray) -> List[bool]:
        raise NotImplementedError("Subclasses must implement the query method.")

    def get_bytesize(self) -> int:
        raise NotImplementedError("Subclasses must implement get_bytesize method.")
