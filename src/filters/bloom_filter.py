import numpy as np
import math
from pybloom_live import BloomFilter as _BloomFilter, ScalableBloomFilter as _ScalableBloomFilter
from lsh import LSH
from typing import Optional, List
from filters.filter import Filter


class BloomFilter(Filter):
    """
    Wrapper around pybloom_live.BloomFilter.

    The underlying filter operates on integer hashes produced by the LSH layer,
    so there is no need to serialise raw vectors into the filter itself.

    Args:
        dim:         Dimensionality of the input vectors (passed to Filter).
        capacity:    Maximum number of elements before false-positive rate rises.
        error_rate:  Desired maximum false-positive probability (default 0.1 %).
        lsh:         Optional pre-built LSH instance; one is created if omitted.
        max_memory:  Maximum memory usage in bytes. If set, capacity is derived
                     from the memory budget instead of being specified directly.
    """

    def __init__(
        self,
        dim: int,
        capacity: int = 1_000,
        error_rate: float = 0.001,
        lsh: Optional[LSH] = None,
        max_memory: Optional[int] = None,
    ):
        super().__init__(dim, lsh, max_memory=max_memory)
        self._error_rate = error_rate

        if max_memory is not None:
            # Derive capacity from the memory budget.
            # Bloom filter memory (bits) = capacity * -ln(error_rate) / (ln(2)^2)
            # So capacity = memory_bytes * 8 * (ln(2)^2) / -ln(error_rate)
            bits_available = max_memory * 8
            self._capacity = int(
                bits_available * (math.log(2) ** 2) / (-math.log(error_rate))
            )
            self._capacity = max(self._capacity, 1)
        else:
            self._capacity = capacity

        self.filter: _BloomFilter = _BloomFilter(
            capacity=self._capacity, error_rate=error_rate
        )

    def get_exact_matching_fpr(self) -> float:
        """
        Returns the configured exact-matching False Positive Rate (FPR)
        of the underlying Bloom Filter, completely ignoring the LSH layer.
        """
        return float(self._error_rate)

    def construct(self, vectors: np.ndarray, is_hashed: bool = False) -> None:
        """Populate the filter from a batch of vectors."""
        # Re-initialise so that construct() is idempotent.
        self.filter = _BloomFilter(
            capacity=max(self._capacity, len(vectors)),
            error_rate=self._error_rate,
        )
        hashes = self._normalize_hashes(vectors) if is_hashed else self._normalize_hashes(self.lsh.hash(vectors))
        for signature_keys in hashes:
            for key in signature_keys:
                self.filter.add(key)

    def query(self, vectors: np.ndarray, is_hashed: bool = False) -> List[bool]:
        """Return a per-vector list of membership booleans."""
        hashes = self._normalize_hashes(vectors) if is_hashed else self._normalize_hashes(self.lsh.hash(vectors))
        return [any(key in self.filter for key in signature_keys) for signature_keys in hashes]

    def insert(self, vectors: np.ndarray, is_hashed: bool = False) -> None:
        """
        Insert vectors into the filter after construction.

        Bloom filters support dynamic insertion; automatically scales capacity if needed.
        """
        hashes = self._normalize_hashes(vectors) if is_hashed else self._normalize_hashes(self.lsh.hash(vectors))
        for signature_keys in hashes:
            for key in signature_keys:
                self.filter.add(key)

    def add(self, vectors: np.ndarray, is_hashed: bool = False) -> None:
        """
        Add vectors to the filter after construction.

        Bloom filters support dynamic insertion; automatically scales capacity if needed.
        """
        hashes = self._normalize_hashes(vectors) if is_hashed else self._normalize_hashes(self.lsh.hash(vectors))
        for signature_keys in hashes:
            for key in signature_keys:
                self.filter.add(key)

    def get_bytesize(self) -> int:
        """
        Returns the approximate memory size in bytes of the Bloom Filter.
        
        Uses the formula: size = (capacity * -ln(error_rate)) / (ln(2)^2) bits
        """
        # Calculate the number of bits needed
        bits_needed = -self._capacity * math.log(self._error_rate) / (math.log(2) ** 2)
        # Convert bits to bytes
        return int(math.ceil(bits_needed / 8))


class ScalableBloomFilter(Filter):
    """
    Wrapper around pybloom_live.ScalableBloomFilter.

    Grows automatically as more elements are inserted, so no capacity needs to
    be specified up front.

    Args:
        dim:              Dimensionality of the input vectors.
        initial_capacity: Starting capacity before the first scale-up.
        error_rate:       Desired false-positive probability (default 0.1 %).
        lsh:              Optional pre-built LSH instance.
        max_memory:       Maximum memory usage in bytes. If set, the initial
                          capacity is derived from the memory budget.
    """

    def __init__(
        self,
        dim: int,
        initial_capacity: int = 1_000,
        error_rate: float = 0.001,
        lsh: Optional[LSH] = None,
        max_memory: Optional[int] = None,
    ):
        super().__init__(dim, lsh, max_memory=max_memory)
        self._error_rate = error_rate

        if max_memory is not None:
            # Derive initial capacity from the memory budget (same formula as BloomFilter).
            bits_available = max_memory * 8
            self._initial_capacity = int(
                bits_available * (math.log(2) ** 2) / (-math.log(error_rate))
            )
            self._initial_capacity = max(self._initial_capacity, 1)
        else:
            self._initial_capacity = initial_capacity

        self.filter: _ScalableBloomFilter = _ScalableBloomFilter(
            initial_capacity=self._initial_capacity, error_rate=error_rate
        )

    def get_exact_matching_fpr(self) -> float:
        """
        Returns the bounded exact-matching False Positive Rate (FPR)
        of the Scalable Bloom Filter, completely ignoring the LSH layer.
        
        The scalable variant dynamically adds internal filters with tighter 
        bounds to ensure the cumulative exact matching FPR stays below this limit.
        """
        return float(self._error_rate)

    def construct(self, vectors: np.ndarray, is_hashed: bool = False) -> None:
        """Populate the filter from a batch of vectors."""
        self.filter = _ScalableBloomFilter(
            initial_capacity=self._initial_capacity,
            error_rate=self._error_rate,
        )
        hashes = vectors if is_hashed else self.lsh.hash(vectors)
        for h in hashes:
            self.filter.add(h)

    def query(self, vectors: np.ndarray, is_hashed: bool = False) -> List[bool]:
        """Return a per-vector list of membership booleans."""
        hashes = vectors if is_hashed else self.lsh.hash(vectors)
        return [h in self.filter for h in hashes]

    def add(self, vectors: np.ndarray, is_hashed: bool = False) -> None:
        """
        Add vectors to the scalable Bloom filter after construction.

        The scalable filter automatically grows as needed.
        """
        hashes = vectors if is_hashed else self.lsh.hash(vectors)
        for h in hashes:
            self.filter.add(h)

    def get_bytesize(self) -> int:
        """
        Returns the approximate memory size in bytes of the Scalable Bloom Filter.
        
        Estimates based on the initial capacity and error rate.
        The actual size may grow as the filter scales up.
        """
        # Calculate the number of bits needed for initial filter
        bits_needed = -self._initial_capacity * math.log(self._error_rate) / (math.log(2) ** 2)
        # Convert bits to bytes, add some overhead for scalability structure
        return int(math.ceil(bits_needed / 8)) + 1024  # 1KB overhead for scalability metadata