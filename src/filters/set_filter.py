import sys
from typing import List
import numpy as np
from filters.filter import Filter

class SetFilter(Filter):
    """
    A simple baseline filter using a standard Python set.
    Serves as a sanity check for LSH hashing and exact matching logic.
    """
    def __init__(self, dim: int, lsh):
        super().__init__(dim, lsh)
        self._storage = set()

    def get_exact_matching_fpr(self) -> float:
        """
        A standard Python set has no structural false positives.
        Any false positives are purely due to LSH collisions.
        """
        return 0.0

    def _get_keys(self, vectors: np.ndarray) -> List[List[str]]:
        """Helper to hash vectors and normalize them into keys."""
        if self.lsh is None:
            raise ValueError("LSH instance is required to generate keys.")
        # Assuming the LSH instance exposes a `hash` method
        hashes = self.lsh.hash(vectors)
        return self._normalize_hashes(hashes)

    def construct(self, vectors: np.ndarray) -> None:
        """Clear the current set and initialize it with the given vectors."""
        self._storage.clear()
        self.insert(vectors)

    def insert(self, vectors: np.ndarray) -> None:
        """Insert hashed vectors into the set."""
        keys_list = self._get_keys(vectors)
        for keys in keys_list:
            for key in keys:
                self._storage.add(key)

    def add(self, vectors: np.ndarray) -> None:
        """Alias for insert, as dynamic insertion is natively supported."""
        self.insert(vectors)

    def query(self, vectors: np.ndarray) -> List[bool]:
        """
        Check if the hashed vectors are in the set.
        Returns True if ANY of the item's signatures match an entry in the set.
        """
        keys_list = self._get_keys(vectors)
        results = []
        for keys in keys_list:
            found = any(key in self._storage for key in keys)
            results.append(found)
        return results

    def remove(self, vectors: np.ndarray) -> List[bool]:
        """
        Remove vectors from the set.
        Returns True if the item had at least one signature removed.
        """
        keys_list = self._get_keys(vectors)
        results = []
        for keys in keys_list:
            removed_any = False
            for key in keys:
                if key in self._storage:
                    self._storage.remove(key)
                    removed_any = True
            results.append(removed_any)
        return results

    def get_bytesize(self) -> int:
        """
        Calculates the approximate memory footprint of the underlying set.
        Includes both the set structure overhead and the stored string objects.
        """
        size = sys.getsizeof(self._storage)
        # Deep size accumulation for the string objects stored inside
        for item in self._storage:
            size += sys.getsizeof(item)
        return size