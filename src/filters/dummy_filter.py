from typing import List, Optional
import numpy as np
from filters.filter import Filter
from lsh import LSH


class DummyFilter(Filter):
    """
    Simple in-memory filter that stores all normalized LSH signature keys in a set.

    This filter is intended for testing and debugging. It treats each LSH
    signature as a key and keeps every OR branch separately. A query succeeds
    if any OR signature key is present.
    """

    def __init__(self, dim: int, lsh: LSH):
        super().__init__(dim, lsh)
        self.hash_set = set()

    def get_exact_matching_fpr(self) -> float:
        # Exact set membership is assumed to be perfect for inserted keys.
        return 0.0

    def construct(self, vectors: np.ndarray) -> None:
        """Build the set of all normalized LSH signature keys."""
        self.hash_set = set()
        hashes = self._normalize_hashes(self.lsh.hash(vectors))

        for signature_keys in hashes:
            for key in signature_keys:
                self.hash_set.add(key)

    def query(self, vectors: np.ndarray) -> List[bool]:
        hashes = self._normalize_hashes(self.lsh.hash(vectors))
        return [any(key in self.hash_set for key in signature_keys) for signature_keys in hashes]

    def add(self, vectors: np.ndarray) -> None:
        """
        Add vectors to the filter after construction.
        
        Simply inserts the normalized LSH signature keys into the set.
        """
        hashes = self._normalize_hashes(self.lsh.hash(vectors))
        for signature_keys in hashes:
            for key in signature_keys:
                self.hash_set.add(key)

    def remove(self, vectors: np.ndarray) -> List[bool]:
        """
        Remove vectors from the filter.
        
        Returns a list of booleans indicating whether each vector was present before removal.
        """
        hashes = self._normalize_hashes(self.lsh.hash(vectors))
        results: List[bool] = []
        for signature_keys in hashes:
            removed_any = False
            for key in signature_keys:
                if key in self.hash_set:
                    self.hash_set.discard(key)
                    removed_any = True
            results.append(removed_any)
        return results

    def get_bytesize(self) -> int:
        # Approximate bytes used by the set contents; this is a rough estimate.
        return sum(len(str(key).encode("utf-8")) for key in self.hash_set)
