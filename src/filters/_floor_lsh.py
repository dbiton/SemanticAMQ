"""Shared machinery for the zero-false-negative floor-projection LSH baselines
(PacukLSHFilter, SankowskiWygockiFilter).

Both hash a point to one integer signature per table, each coordinate a
scalar projection floored to a bucket, built so that ANY pair within the
radius lands in the same or an adjacent bucket on every coordinate
(|floor(a/w) - floor(b/w)| <= 1 whenever |a - b| <= w). A stored point is
therefore always reachable from a query within the radius through one of the
3^L signatures within +-1 of the query's own, L the signature length. That
gives the papers' two variants (Pacuk et al., Theorem 1/2 vs. Section 4.1):

  replicate="store": every point's key is written into all 3^L neighbouring
                     cells at build time; a query is 1 lookup per table.
  replicate="probe": each point's own signature is written once; a query
                     probes the 3^L cells around its own signature per table.

The two give identical answers (up to Bloom error) and trade memory for query
time. As an AMQ the filter answers "yes" on any hit: the papers' last step,
re-checking candidates against the stored points, needs the raw vectors, which
a filter does not keep. A Bloom filter only ever adds false positives, so the
no-false-negative guarantee survives the max_memory path intact.

The projection vectors are a deterministic function of `seed`, so they are
counted as seed bytes, like BallCoverFilter's rotation (_ROTATION_SEED_BYTES)
-- not as a stored matrix.
"""

import math
from typing import List, Optional, Union

import numpy as np
from pybloom_live import BloomFilter as _BloomFilter

from filters.filter import Filter, InsufficientMemoryError


_REPLICATE_MODES = ("store", "probe")


class FloorLSHFilter(Filter):
    """Base class: subclasses define the tables and the signature function;
    this class owns key layout, replication, the key store and memory
    accounting.

    distance_threshold is baked in (the bucket widths are derived from it);
    query(..., tau=t) only accepts t == distance_threshold or None, as in
    DSHFilter. `slack` widens the radius the buckets are sized for, so that
    float32 rounding in the ground truth's own distance computation cannot
    turn a boundary pair into a false negative (same role as
    BallCoverFilter._COVERAGE_EPSILON).
    """

    _SEED_BYTES = 8
    # Cap on exact-set keys (max_memory=None only): a mistyped parameter
    # (3^L grows fast) should fail loudly, not exhaust the machine's RAM.
    _MAX_EXACT_KEYS = 20_000_000
    # Bound on the int32 working array built per chunk in _cell_keys.
    _CHUNK_ELEMENTS = 4_000_000
    # Cells one point may expand to (store: keys written; probe: lookups per
    # query). Past this, a single point's key array alone runs to hundreds of
    # MB and a Python-level lookup loop to seconds per query.
    _MAX_CELLS_PER_POINT = 2_000_000

    def __init__(
        self,
        dim: int,
        distance_threshold: float,
        max_memory: Optional[int] = None,
        replicate: str = "store",
        error_rate: float = 0.01,
        slack: float = 1e-3,
        seed: int = 42,
    ):
        super().__init__(dim, max_memory=max_memory)
        if replicate not in _REPLICATE_MODES:
            raise ValueError(f"replicate must be one of {_REPLICATE_MODES}, got {replicate!r}")
        if distance_threshold <= 0:
            raise ValueError(f"distance_threshold must be positive, got {distance_threshold}")
        self.distance_threshold = float(distance_threshold)
        self.replicate = replicate
        self._error_rate = error_rate
        self.slack = float(slack)
        self.seed = seed
        self._keys: Optional[Union[set, _BloomFilter]] = None
        self._offsets_cache: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Subclass interface
    # ------------------------------------------------------------------

    @property
    def num_tables(self) -> int:
        raise NotImplementedError

    @property
    def signature_length(self) -> int:
        raise NotImplementedError

    def _signatures(self, vectors: np.ndarray) -> np.ndarray:
        """(n, num_tables, signature_length) int32 bucket indices."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Derived sizes
    # ------------------------------------------------------------------

    @property
    def radius(self) -> float:
        """The radius the buckets are sized for: tau plus the float slack."""
        return self.distance_threshold + self.slack

    @property
    def cells_per_signature(self) -> int:
        return 3 ** self.signature_length

    def keys_per_point(self) -> int:
        per_table = self.cells_per_signature if self.replicate == "store" else 1
        return self.num_tables * per_table

    def probes_per_query(self) -> int:
        per_table = 1 if self.replicate == "store" else self.cells_per_signature
        return self.num_tables * per_table

    @property
    def _key_bytes(self) -> int:
        return 4 * (self.signature_length + 1)  # int32 table id + int32 signature

    # ------------------------------------------------------------------
    # Keys
    # ------------------------------------------------------------------

    def _offsets(self) -> np.ndarray:
        """(3^L, L) every {-1, 0, +1}^L displacement."""
        if self._offsets_cache is None:
            L = self.signature_length
            grids = np.meshgrid(*([np.array([-1, 0, 1], dtype=np.int32)] * L), indexing="ij")
            self._offsets_cache = np.stack([g.ravel() for g in grids], axis=1)
        return self._offsets_cache

    def _cell_keys(self, sig: np.ndarray, expand: bool) -> np.ndarray:
        """(n, T*M) array of fixed-width byte keys, M = 3^L if expand else 1."""
        n, T, L = sig.shape
        if expand and T * self.cells_per_signature > self._MAX_CELLS_PER_POINT:
            raise ValueError(
                f"{self.__class__.__name__}: {T} tables x 3^{L} cells = "
                f"{T * self.cells_per_signature:,} cells per point exceeds "
                f"_MAX_CELLS_PER_POINT={self._MAX_CELLS_PER_POINT:,}."
            )
        cells = sig[:, :, None, :] + self._offsets()[None, None] if expand else sig[:, :, None, :]
        M = cells.shape[2]
        table_ids = np.broadcast_to(np.arange(T, dtype=np.int32)[None, :, None, None], (n, T, M, 1))
        full = np.ascontiguousarray(np.concatenate([table_ids, cells.astype(np.int32)], axis=-1))
        return full.view(np.dtype((np.void, self._key_bytes))).reshape(n, T * M)

    def _point_chunks(self, n: int, expand: bool):
        per_point = self.num_tables * (self.cells_per_signature if expand else 1)
        step = max(1, self._CHUNK_ELEMENTS // (per_point * (self.signature_length + 1)))
        for start in range(0, n, step):
            yield start, min(n, start + step)

    # ------------------------------------------------------------------
    # Filter interface
    # ------------------------------------------------------------------

    def _check_vectors(self, vectors: np.ndarray) -> np.ndarray:
        vecs = np.asarray(vectors)
        if vecs.ndim != 2 or vecs.shape[1] != self.dim:
            raise ValueError(f"expected shape (n, {self.dim}), got {vecs.shape}")
        return vecs

    def construct(self, vectors: np.ndarray) -> None:
        vecs = self._check_vectors(vectors)
        needed = len(vecs) * self.keys_per_point()
        if self.max_memory is None:
            if needed > self._MAX_EXACT_KEYS:
                raise ValueError(
                    f"{self.__class__.__name__} would store {needed:,} exact keys "
                    f"({len(vecs)} points x {self.keys_per_point():,} keys/point), above "
                    f"_MAX_EXACT_KEYS={self._MAX_EXACT_KEYS:,}; pass max_memory to use a "
                    f"Bloom filter, or use replicate='probe'."
                )
            self._keys = set()
        else:
            bits_available = max(0, self.max_memory - self._SEED_BYTES) * 8
            capacity = int(bits_available * (math.log(2) ** 2) / (-math.log(self._error_rate)))
            if capacity < max(1, needed):
                raise InsufficientMemoryError(
                    f"max_memory={self.max_memory} bytes holds a Bloom filter of "
                    f"capacity {capacity:,} at error_rate={self._error_rate}, but "
                    f"{self.__class__.__name__} needs {needed:,} keys ({len(vecs)} points x "
                    f"{self.keys_per_point():,} keys/point, replicate={self.replicate!r})."
                )
            self._keys = _BloomFilter(capacity=capacity, error_rate=self._error_rate)
        self._insert_batch(vecs)

    def _insert_batch(self, vecs: np.ndarray) -> None:
        expand = self.replicate == "store"
        try:
            for lo, hi in self._point_chunks(len(vecs), expand):
                keys = self._cell_keys(self._signatures(vecs[lo:hi]), expand)
                if isinstance(self._keys, set):
                    self._keys.update(k.tobytes() for k in keys.ravel())
                else:
                    for k in keys.ravel():
                        self._keys.add(k.tobytes())
        except IndexError as e:
            # pybloom_live raises a bare IndexError once the bit array is at
            # capacity -- only reachable through insert(), since construct()
            # checks capacity up front. Same translation as DSHFilter.
            raise InsufficientMemoryError(
                f"{self.__class__.__name__}'s Bloom filter filled up on insert "
                f"(max_memory={self.max_memory})."
            ) from e

    def insert(self, vectors: np.ndarray) -> None:
        vecs = self._check_vectors(vectors)
        if self._keys is None:
            self.construct(vecs)
            return
        if isinstance(self._keys, set):
            if len(self._keys) + len(vecs) * self.keys_per_point() > self._MAX_EXACT_KEYS:
                raise ValueError(f"insert would exceed _MAX_EXACT_KEYS={self._MAX_EXACT_KEYS:,}")
        self._insert_batch(vecs)

    def query(self, vectors: np.ndarray, tau: Optional[float] = None) -> List[bool]:
        if self._keys is None:
            raise ValueError(f"{self.__class__.__name__} has not been constructed.")
        if tau is not None and abs(tau - self.distance_threshold) > 1e-9:
            raise ValueError(
                f"{self.__class__.__name__} bakes tau into construction (distance_threshold="
                f"{self.distance_threshold}); query-time tau={tau} is not supported."
            )
        vecs = self._check_vectors(vectors)
        expand = self.replicate == "probe"
        store = self._keys
        results: List[bool] = []
        for lo, hi in self._point_chunks(len(vecs), expand):
            for row in self._cell_keys(self._signatures(vecs[lo:hi]), expand):
                results.append(any(k.tobytes() in store for k in row))
        return results

    @staticmethod
    def _bloom_bytes_for(capacity: int, error_rate: float) -> int:
        bits_needed = -capacity * math.log(error_rate) / (math.log(2) ** 2)
        return int(math.ceil(bits_needed / 8))

    def get_bytesize(self) -> int:
        if self._keys is None:
            return self._SEED_BYTES
        if isinstance(self._keys, set):
            return self._SEED_BYTES + len(self._keys) * self._key_bytes
        return self._SEED_BYTES + self._bloom_bytes_for(self._keys.capacity, self._error_rate)
