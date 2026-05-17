from typing import List, Tuple, Optional
import numpy as np
from lsh import LSH


class Filter:
    def __init__(self, dim: int, lsh: Optional[LSH] = None):
        self.dim = dim
        self.lsh = lsh

    def get_exact_matching_fpr(self) -> float:
        """
        Returns the standard exact-matching False Positive Rate (epsilon) 
        of the structural filter, ignoring the LSH layer.
        """
        raise NotImplementedError("Subclasses must implement get_exact_matching_fpr method.")

    def get_criteria(self, n: int, r1: float, r2: float) -> Tuple[float, float]:
        """
        Calculates and returns the expected global semantic error criteria.

        Args:
            n:  The number of elements currently stored in the filter.
            r1: The target close-distance semantic threshold.
            r2: The target far-distance semantic threshold.

        Returns:
            A tuple of floats containing (Expected FPR, Expected FNR).
        """
        if self.lsh is not None:
            # Query the underlying LSH implementation for the p1, p2 probabilities
            p1, p2 = self.lsh.get_criteria(r1, r2)
        else:
            # Fallback to standard exact matching if no LSH is attached
            p1, p2 = 1.0, 0.0

        # FNR is dictated by the LSH layer failing to collide a valid neighbor pair
        fnr = 1.0 - p1

        # FPR combines random LSH collisions over n elements with structural filter errors
        epsilon = self.get_exact_matching_fpr()
        fpr = 1.0 - (1.0 - epsilon) * ((1.0 - p2) ** n)

        return float(fpr), float(fnr)

    def construct(self, vectors: np.ndarray) -> None:
        raise NotImplementedError("Subclasses must implement the construct method.")

    def query(self, vectors: np.ndarray) -> List[bool]:
        raise NotImplementedError("Subclasses must implement the query method.")