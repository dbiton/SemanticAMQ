from cuckoofilter import CuckooFilter as _CuckooFilter
from lsh import LSH
from typing import Optional, List
from filters.filter import Filter
import numpy as np


class CuckooFilter(Filter):
    """
    Wrapper around cuckoofilter.CuckooFilter.

    Unlike Bloom filters, Cuckoo filters support deletion.  Call ``delete``
    with vectors that were previously inserted to remove them.

    Args:
        dim:              Dimensionality of the input vectors.
        capacity:         Number of buckets in the underlying table.  The
                          usable element count is roughly 0.95 × capacity.
        fingerprint_size: Fingerprint size in bytes (1–4).  Larger values give
                          lower false-positive rates at the cost of memory.
                          fingerprint_size=1  → ~3 % FPR
                          fingerprint_size=2  → ~0.01 % FPR
        lsh:              Optional pre-built LSH instance.
    """

    def __init__(
        self,
        dim: int,
        capacity: int = 1_000,
        fingerprint_size: int = 2,
        lsh: Optional[LSH] = None,
    ):
        super().__init__(dim, lsh)
        self._capacity = capacity
        self._fingerprint_size = fingerprint_size
        self.filter: _CuckooFilter = _CuckooFilter(
            capacity=capacity, fingerprint_size=fingerprint_size
        )

    def get_exact_matching_fpr(self) -> float:
        """
        Returns the theoretical upper bound of the exact-matching False Positive Rate (FPR)
        of the underlying Cuckoo Filter, ignoring the LSH layer.
        
        Assumes a standard architecture of 4 slots per bucket and 2 target bucket probes.
        """
        # Convert fingerprint size from bytes to bits (1 byte = 8 bits)
        bits = 8 * self._fingerprint_size
        # Formula: (2 * slots_per_bucket) / 2^bits where slots_per_bucket = 4
        return float(8.0 / (2 ** bits))

    def construct(self, vectors: np.ndarray) -> None:
        """Populate the filter from a batch of vectors."""
        # Re-create so that construct() is idempotent.
        self.filter = _CuckooFilter(
            capacity=self._capacity, fingerprint_size=self._fingerprint_size
        )
        hashes = self.lsh.hash(vectors)
        for h in hashes:
            self.filter.insert(h)

    def query(self, vectors: np.ndarray) -> List[bool]:
        """Return a per-vector list of membership booleans."""
        hashes = self.lsh.hash(vectors)
        return [self.filter.contains(h) for h in hashes]

    def delete(self, vectors: np.ndarray) -> List[bool]:
        """
        Remove vectors from the filter.

        Returns a boolean list indicating whether each vector was present
        (and successfully deleted) before the call.
        """
        hashes = self.lsh.hash(vectors)
        results: List[bool] = []
        for h in hashes:
            if self.filter.contains(h):
                self.filter.delete(h)
                results.append(True)
            else:
                results.append(False)
        return results