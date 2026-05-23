import numpy as np
import math
from lsh import LSH
from typing import Optional, List
from filters.filter import Filter
from probables import QuotientFilter


class QuotientFilterWrapper(Filter):
    def __init__(self, dim: int, lsh: Optional[LSH] = None, remainder_bits: int = 8):
        """
        A wrapper for pyprobables.QuotientFilter integrating with the Filter interface.
        
        Args:
            dim: The expected number of elements to store.
            lsh: The locality-sensitive hashing object.
            remainder_bits: Number of bits for the fingerprint remainder. 
                            Determines the exact matching FPR (1 / 2^remainder_bits).
        """
        super().__init__(dim, lsh)
        
        # Calculate the required quotient (q) size such that 2^q can safely fit 'dim' elements
        # Adding 1 to ensure the load factor stays safely under 100%
        self.quotient: int = max(3, math.ceil(math.log2(dim)) + 1)
        
        # Initialize the underlying filter
        self.filter = QuotientFilter(quotient=self.quotient)
        
        # Exact-matching false positive rate for a quotient filter is roughly 1 / 2^r
        self.epsilon: float = 1.0 / (2**remainder_bits)

    def get_exact_matching_fpr(self) -> float:
        """
        Returns the standard exact-matching False Positive Rate (FPR) 
        of the underlying AMQ structure, completely ignoring the LSH layer.
        """
        return float(self.epsilon)

    def construct(self, vectors: np.ndarray) -> None:
        """
        Hashes the input vectors via LSH and populates the quotient filter.
        """
        hashes = self.lsh.hash(vectors)
        for h in hashes:
            # pyprobables expects strings or bytes to internalize its hashing logic
            self.filter.add(str(h))

    def query(self, vectors: np.ndarray) -> List[bool]:
        """
        Queries the quotient filter for membership of the provided vectors.
        """
        hashes = self.lsh.hash(vectors)
        return [self.filter.check(str(h)) for h in hashes]

    def get_bytesize(self) -> int:
        """
        Returns the approximate memory size in bytes of the Quotient Filter.
        
        Each entry in a quotient filter stores quotient (q bits) and remainder,
        plus is_occupied, is_continuation, and is_shifted flags (3 bits each per slot).
        """
        # The number of slots is 2^quotient
        num_slots = 2 ** self.quotient
        # Each slot needs bits for: quotient (quotient bits) + remainder (3 bits) + 3 flags (is_occupied, is_continuation, is_shifted)
        # Approximate: 1 byte per entry plus some overhead
        bytes_per_entry = 1
        total_bytes = num_slots * bytes_per_entry
        # Add overhead for metadata (~5%)
        overhead = int(total_bytes * 0.05)
        return total_bytes + overhead