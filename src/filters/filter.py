from typing import List

from git import Optional
import numpy as np
from lsh import LSH


class Filter:
    def __init__(self, dim: int, lsh: Optional[LSH] = None):
        self.dim = dim
        self.lsh = lsh

    def construct(self, vectors: np.ndarray) -> None:
        raise NotImplementedError("Subclasses must implement the construct method.")

    def query(self, vectors: np.ndarray) -> List[bool]:
        raise NotImplementedError("Subclasses must implement the query method.")
