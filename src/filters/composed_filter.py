from typing import List, Tuple, Optional
import numpy as np
from lsh import LSH
from filters.filter import Filter

class ComposedFilter(Filter):
    def __init__(self, filters: List[Filter], lsh: Optional[LSH] = None):
        dims = [f.dim for f in filters]
        assert len(set(dims)) == 1, "All filters must have the same dimensionality."
        self.dim = dims[0]
        super().__init__(self.dim)
        self.filters = filters
        self.lsh = lsh

    def get_exact_matching_fpr(self) -> float:
        epsilon_total = 1.0
        for f in self.filters:
            epsilon_total *= f.get_exact_matching_fpr()
                
        return float(epsilon_total)

    def get_criteria(self, n: int, r1: float, r2: float) -> Tuple[float, float]:
        joint_fpr = 1.0
        joint_tpr = 1.0

        for f in self.filters:
            layer_fpr, layer_fnr = f.get_criteria(n, r1, r2)
            joint_fpr *= layer_fpr
            layer_tpr = 1.0 - layer_fnr
            joint_tpr *= layer_tpr
        joint_fnr = 1.0 - joint_tpr

        return float(joint_fpr), float(joint_fnr)

    def construct(self, vectors: np.ndarray) -> None:
        for f in self.filters:
            f.construct(vectors)

    def query(self, vectors: np.ndarray) -> List[bool]:
        queries = [f.query(vectors) for f in self.filters]
        return np.all(queries, axis=0)

    def get_bytesize(self) -> int:
        """
        Returns the total memory size in bytes of all composed filters.
        
        Sums the get_bytesize() of all component filters.
        """
        total_size = 0
        for f in self.filters:
            total_size += f.get_bytesize()
        return total_size