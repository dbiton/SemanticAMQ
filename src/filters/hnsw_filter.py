"""HNSWFilter -- semantic AMQ baseline built directly on FAISS's HNSW
approximate nearest-neighbor index. Unlike every other filter here, it
stores full float32 vectors plus the HNSW graph and answers via a real
approximate NN search: no structural false positives, but it trades away
BallCoverFilter's zero-false-negative guarantee (HNSW can miss the true
neighbor, governed by ef_search/ef_construction/M).

Memory is not budget-*tunable* -- IndexHNSWFlat stores full uncompressed
vectors, with no way to shrink short of a quantized HNSW variant (out of
scope). It is still budget-*checked*: construct() raises
InsufficientMemoryError when n vectors plus their graph cannot fit in
max_memory, so an equal-memory sweep records those budgets as infeasible
instead of plotting an over-budget point (a 1MB budget silently holding a
39MB index) as if it were a legitimate memory/accuracy tradeoff."""

from typing import List, Optional

import faiss
import numpy as np

from filters.filter import Filter, InsufficientMemoryError


class HNSWFilter(Filter):
    """FAISS IndexHNSWFlat wrapped as a semantic AMQ filter: query is a real
    1-NN search, "duplicate" iff the nearest neighbor found is within tau.
    tau is a pure query-time knob -- nothing about the threshold is baked
    into the graph, unlike DSHFilter/DSBFFilter/LSBFFilter."""

    def __init__(
        self,
        dim: int,
        distance_threshold: float,
        max_memory: Optional[int] = None,
        M: int = 32,
        ef_construction: int = 200,
        ef_search: int = 64,
        max_spheres: Optional[int] = None,
    ):
        super().__init__(dim, max_memory=max_memory)
        self.distance_threshold = float(distance_threshold)
        self.M = M
        self.ef_construction = ef_construction
        self._ef_search = ef_search
        self._max_spheres = max_spheres  # unused; accepted for _instantiate_filter parity
        self.index: Optional[faiss.IndexHNSWFlat] = None

    @property
    def ef_search(self) -> int:
        """A property, not a plain attribute, so the equal-memory
        calibration dial's `setattr(filter_instance, "ef_search", value)` on
        an already-built instance actually reaches the underlying faiss
        index (mirrors IVFFilter.nprobe). As a plain attribute this silently
        did nothing: _new_index() copies it into the index once at
        construction and never looks at it again."""
        return self.index.hnsw.efSearch if self.index is not None else self._ef_search

    @ef_search.setter
    def ef_search(self, value: int) -> None:
        self._ef_search = int(value)
        if self.index is not None:
            self.index.hnsw.efSearch = int(value)

    def _min_footprint_bytes(self, n: int) -> int:
        """Strict lower bound on the built index's size, knowable before
        construction: the full float32 vectors, the level-0 neighbor table
        (every node has 2*M level-0 slots), and the per-node offsets/levels
        arrays. Real usage is higher by the upper-level links, whose count
        depends on HNSW's random level assignment -- so this bound only
        rejects budgets that are certainly infeasible; _footprint_bytes()
        re-checks exactly once the graph exists."""
        return n * self.dim * 4 + n * (2 * self.M) * 4 + (n + 1) * 8 + n * 4

    def _footprint_bytes(self) -> int:
        """Exact resident size of the built index, summed from the faiss
        arrays directly rather than via serialize_index(): identical to
        within a ~200-byte header (verified against faiss 1.15.1) but O(1)
        instead of allocating a second full copy of the index to measure it."""
        if self.index is None:
            return 0
        hnsw = self.index.hnsw
        return (
            self.index.ntotal * self.dim * 4      # IndexFlat storage, float32
            + hnsw.neighbors.size() * 4           # graph links, int32
            + hnsw.offsets.size() * 8             # per-node link offsets, int64
            + hnsw.levels.size() * 4              # per-node level, int32
        )

    def _new_index(self) -> faiss.IndexHNSWFlat:
        index = faiss.IndexHNSWFlat(self.dim, self.M)
        index.hnsw.efConstruction = self.ef_construction
        index.hnsw.efSearch = self._ef_search
        return index

    def construct(self, vectors: np.ndarray) -> None:
        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        n = len(vecs)

        if self.max_memory is not None and n > 0:
            lower_bound = self._min_footprint_bytes(n)
            if lower_bound > self.max_memory:
                raise InsufficientMemoryError(
                    f"max_memory={self.max_memory} bytes is too small for HNSWFilter to hold "
                    f"n={n} vectors of dim={self.dim} at M={self.M}: uncompressed vectors plus "
                    f"the level-0 graph alone need at least {lower_bound} bytes. IndexHNSWFlat "
                    f"has no compressed variant here, so no configuration fits this budget."
                )

        self.index = self._new_index()
        if n > 0:
            self.index.add(vecs)

        if self.max_memory is not None:
            actual = self._footprint_bytes()
            if actual > self.max_memory:
                self.index = None
                raise InsufficientMemoryError(
                    f"max_memory={self.max_memory} bytes is too small for HNSWFilter: the built "
                    f"index over n={n} vectors of dim={self.dim} at M={self.M} occupies "
                    f"{actual} bytes."
                )

    def query(self, vectors: np.ndarray, tau: Optional[float] = None) -> List[bool]:
        if self.index is None:
            raise ValueError("HNSWFilter has not been constructed.")
        if tau is None:
            tau = self.distance_threshold
        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        if len(vecs) == 0 or self.index.ntotal == 0:
            return [False] * len(vecs)
        sq_dist, ids = self.index.search(vecs, 1)
        # faiss fills an unfilled slot with id=-1 and distance=FLT_MAX, which
        # already fails the tau test; the explicit id check mirrors IVFFilter
        # so neither filter relies on that sentinel's exact value.
        found = ids[:, 0] != -1
        return (found & (sq_dist[:, 0] <= float(tau) ** 2)).tolist()

    def insert(self, vectors: np.ndarray) -> None:
        if self.index is None:
            self.construct(vectors)
            return
        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        if len(vecs) > 0:
            self.index.add(vecs)

    def get_bytesize(self) -> int:
        return self._footprint_bytes()
