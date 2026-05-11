import numpy as np
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
        dim:        Dimensionality of the input vectors (passed to Filter).
        capacity:   Maximum number of elements before false-positive rate rises.
        error_rate: Desired maximum false-positive probability (default 0.1 %).
        lsh:        Optional pre-built LSH instance; one is created if omitted.
    """

    def __init__(
        self,
        dim: int,
        capacity: int = 1_000,
        error_rate: float = 0.001,
        lsh: Optional[LSH] = None,
    ):
        super().__init__(dim, lsh)
        self._capacity = capacity
        self._error_rate = error_rate
        self.filter: _BloomFilter = _BloomFilter(
            capacity=capacity, error_rate=error_rate
        )

    def construct(self, vectors: np.ndarray) -> None:
        """Populate the filter from a batch of vectors."""
        # Re-initialise so that construct() is idempotent.
        self.filter = _BloomFilter(
            capacity=max(self._capacity, len(vectors)),
            error_rate=self._error_rate,
        )
        hashes = self.lsh.hash(vectors)
        for h in hashes:
            self.filter.add(h)

    def query(self, vectors: np.ndarray) -> List[bool]:
        """Return a per-vector list of membership booleans."""
        hashes = self.lsh.hash(vectors)
        return [h in self.filter for h in hashes]


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
    """

    def __init__(
        self,
        dim: int,
        initial_capacity: int = 1_000,
        error_rate: float = 0.001,
        lsh: Optional[LSH] = None,
    ):
        super().__init__(dim, lsh)
        self._initial_capacity = initial_capacity
        self._error_rate = error_rate
        self.filter: _ScalableBloomFilter = _ScalableBloomFilter(
            initial_capacity=initial_capacity, error_rate=error_rate
        )

    def construct(self, vectors: np.ndarray) -> None:
        """Populate the filter from a batch of vectors."""
        self.filter = _ScalableBloomFilter(
            initial_capacity=self._initial_capacity,
            error_rate=self._error_rate,
        )
        hashes = self.lsh.hash(vectors)
        for h in hashes:
            self.filter.add(h)

    def query(self, vectors: np.ndarray) -> List[bool]:
        """Return a per-vector list of membership booleans."""
        hashes = self.lsh.hash(vectors)
        return [h in self.filter for h in hashes]


