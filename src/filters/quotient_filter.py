import numpy as np
import math
from lsh import LSH
from typing import Optional, List
from filters.filter import Filter
from probables import QuotientFilter


class QuotientFilterWrapper(Filter):
    def __init__(self, dim: int, lsh: Optional[LSH] = None, remainder_bits: int = 8, max_memory: Optional[int] = None):
        """
        A wrapper for pyprobables.QuotientFilter integrating with the Filter interface.
        
        Args:
            dim: The expected number of elements to store.
            lsh: The locality-sensitive hashing object.
            remainder_bits: Number of bits for the fingerprint remainder. 
                            Determines the exact matching FPR (1 / 2^remainder_bits).
            max_memory: Maximum memory usage in bytes. If set, the quotient size
                        is derived from the memory budget.
        """
        super().__init__(dim, lsh, max_memory=max_memory)
        
        if max_memory is not None:
            # Quotient filter: number of slots = 2^quotient, each slot ~1 byte + 3 flag bits.
            # Approx 11 bits per slot. So 2^quotient * 11 / 8 ≈ max_memory
            # quotient = floor(log2(max_memory * 8 / 11))
            slots_from_memory = max_memory * 8 // 11
            self.quotient: int = max(3, int(math.log2(max(slots_from_memory, 8))))
        else:
            # Calculate the required quotient (q) size such that 2^q can safely fit 'dim' elements
            # Adding 1 to ensure the load factor stays safely under 100%
            self.quotient: int = max(3, math.ceil(math.log2(dim)) + 1)
        
        # Initialize the underlying filter. When a memory budget was given, the
        # quotient size is a hard cap: disable auto-expansion so overflow raises
        # instead of silently growing the table past max_memory.
        self.filter = QuotientFilter(quotient=self.quotient, auto_expand=max_memory is None)
        
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
        hashes = self._normalize_hashes(self.lsh.hash(vectors))
        for signature_keys in hashes:
            for key in signature_keys:
                self.filter.add(str(key))

    def query(self, vectors: np.ndarray) -> List[bool]:
        """
        Queries the quotient filter for membership of the provided vectors.
        """
        hashes = self._normalize_hashes(self.lsh.hash(vectors))
        return [any(self.filter.check(str(key)) for key in signature_keys) for signature_keys in hashes]

    def add(self, vectors: np.ndarray) -> None:
        """
        Add vectors to the quotient filter after construction.
        
        Quotient filters support dynamic insertion.
        """
        hashes = self._normalize_hashes(self.lsh.hash(vectors))
        for signature_keys in hashes:
            for key in signature_keys:
                self.filter.add(str(key))

    def remove(self, vectors: np.ndarray) -> List[bool]:
        """
        Remove vectors from the quotient filter.
        
        Returns a list of booleans indicating whether each vector was present before removal.
        Note: The underlying pyprobables QuotientFilter may have limited deletion support.
        """
        hashes = self._normalize_hashes(self.lsh.hash(vectors))
        results: List[bool] = []
        for signature_keys in hashes:
            removed_any = False
            for key in signature_keys:
                key_str = str(key)
                if self.filter.check(key_str):
                    try:
                        self.filter.delete(key_str)
                        removed_any = True
                    except (AttributeError, NotImplementedError):
                        # If delete is not supported, we can't remove
                        pass
            results.append(removed_any)
        return results

    def get_bytesize(self) -> int:
        """
        Returns the approximate memory size in bytes of the Quotient Filter.
        
        Each entry in a quotient filter stores quotient (q bits) and remainder,
        plus is_occupied, is_continuation, and is_shifted flags (3 bits each per slot).
        """
        # Read the live quotient from the underlying filter rather than
        # self.quotient: when auto_expand is enabled (no max_memory set), the
        # underlying filter can grow its quotient at runtime, and self.quotient
        # would otherwise go stale.
        num_slots = 2 ** self.filter.quotient
        # Each slot needs bits for: quotient (quotient bits) + remainder (3 bits) + 3 flags (is_occupied, is_continuation, is_shifted)
        # Approximate: 1 byte per entry plus some overhead
        bytes_per_entry = 1
        total_bytes = num_slots * bytes_per_entry
        # Add overhead for metadata (~5%)
        overhead = int(total_bytes * 0.05)
        return total_bytes + overhead