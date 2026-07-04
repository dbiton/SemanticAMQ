import array
import numpy as np
import math
from lsh import LSH
from typing import Optional, List
from filters.filter import Filter
from probables import QuotientFilter

# pyprobables' QuotientFilter is valid for quotients in [3, 31].
_MIN_QUOTIENT = 3
_MAX_QUOTIENT = 31


def _quotient_backing_bytes(quotient: int) -> int:
    """
    Real number of bytes the pyprobables QuotientFilter allocates for a filter
    of the given quotient.

    The filter is a fixed table of ``2**quotient`` slots stored in a single
    ``array.array``. pyprobables picks the element typecode from the remainder
    width ``r = 32 - quotient`` ('B' for r<=8, 'I' for r<=16, else 'L'), and the
    *real* per-slot cost is that array's platform ``itemsize`` -- e.g. 'I' is 4
    bytes on Windows, not the 2 implied by its 16 "bits per element". Sizing and
    reporting must use this real cost; the previous ~11-bits-per-slot / 1-byte
    -per-slot estimates undercounted the true allocation by 2-4x, letting the
    filter blow past max_memory while claiming to fit.
    """
    r = 32 - quotient
    typecode = "B" if r <= 8 else ("I" if r <= 16 else "L")
    return (1 << quotient) * array.array(typecode).itemsize


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
            # Pick the largest quotient whose REAL backing-store allocation fits
            # the budget, so the table never exceeds max_memory. Sizing against
            # the true per-slot cost (not a bits-per-slot guess) is what keeps
            # get_bytesize() honest and the capacity limit enforced.
            self.quotient: int = _MIN_QUOTIENT
            for candidate in range(_MIN_QUOTIENT, _MAX_QUOTIENT + 1):
                if _quotient_backing_bytes(candidate) <= max_memory:
                    self.quotient = candidate
                else:
                    break
        else:
            # Calculate the required quotient (q) size such that 2^q can safely fit 'dim' elements
            # Adding 1 to ensure the load factor stays safely under 100%
            self.quotient = max(_MIN_QUOTIENT, math.ceil(math.log2(dim)) + 1)

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

    def insert(self, vectors: np.ndarray) -> None:
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
        Returns the memory size in bytes of the Quotient Filter's backing store.

        The dominant (and budget-relevant) cost is the fixed slot table, an
        ``array.array`` of ``2**quotient`` elements at the platform's real
        per-slot itemsize. This is measured exactly rather than estimated.
        """
        # Read the live quotient from the underlying filter rather than
        # self.quotient: when auto_expand is enabled (no max_memory set), the
        # underlying filter can grow its quotient at runtime, and self.quotient
        # would otherwise go stale.
        return _quotient_backing_bytes(self.filter.quotient)