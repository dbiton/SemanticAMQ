"""HNSWFilter -- semantic AMQ baseline built directly on FAISS's HNSW
approximate nearest-neighbor index. Unlike every other filter here (besides
IVFFilter), it answers via a real approximate NN search: no structural false
positives, but it trades away BallCoverFilter's zero-false-negative guarantee
(HNSW can miss the true neighbor, governed by ef_search/ef_construction/M).

Memory *is* budget-tunable, via the vector storage's precision: `quantize`
picks among IndexHNSWFlat (float32, lossless) and IndexHNSWSQ at fp16/8bit/
6bit/4bit-per-dimension (lossy, faiss's ScalarQuantizer), same idea as
BallCoverFilter's quantization ladder and IVFFilter's PQ codes. Left at the
default "auto", the finest (least lossy) precision that fits max_memory is
chosen automatically -- so, like IVFFilter, a tight budget degrades the
index's accuracy rather than the filter simply refusing to build at all.
Without this, HNSW's memory floor was IVF's floor plus float32-vs-PQ-code
overhead, so an equal-memory sweep could show HNSW going infeasible at
budgets where IVF (and BallCoverFilter) had plenty of room to spare -- not a
real accuracy comparison, just a quantization gap between baselines that had
nothing to do with either algorithm's merit at that budget.

construct() still raises InsufficientMemoryError when not even the coarsest
quantization (4-bit) fits max_memory, so an equal-memory sweep records that
budget as infeasible instead of plotting an over-budget point (a 1MB budget
silently holding a 39MB index) as if it were a legitimate memory/accuracy
tradeoff."""

from typing import List, Optional

import faiss
import numpy as np

from filters.filter import Filter, InsufficientMemoryError

# Ladder from finest (most bytes/dim, least lossy) to coarsest, searched by
# _choose_quantize in that order so the first one that fits wins. None means
# IndexHNSWFlat (full float32, faiss has no "QT_fp32" scalar quantizer type).
_QUANTIZE_LADDER = (None, "fp16", "8bit", "6bit", "4bit")

_QTYPE_BY_NAME = {
    "fp16": faiss.ScalarQuantizer.QT_fp16,
    "8bit": faiss.ScalarQuantizer.QT_8bit,
    "6bit": faiss.ScalarQuantizer.QT_6bit,
    "4bit": faiss.ScalarQuantizer.QT_4bit,
}


class HNSWFilter(Filter):
    """FAISS IndexHNSWFlat/IndexHNSWSQ wrapped as a semantic AMQ filter:
    query is a real 1-NN search, "duplicate" iff the nearest neighbor found
    is within tau. tau is a pure query-time knob -- nothing about the
    threshold is baked into the graph, unlike DSHFilter/DSBFFilter/LSBFFilter."""

    def __init__(
        self,
        dim: int,
        distance_threshold: float,
        max_memory: Optional[int] = None,
        M: int = 32,
        ef_construction: int = 200,
        ef_search: int = 64,
        max_spheres: Optional[int] = None,
        quantize: Optional[str] = "auto",
    ):
        super().__init__(dim, max_memory=max_memory)
        self.distance_threshold = float(distance_threshold)
        self.M = M
        self.ef_construction = ef_construction
        self._ef_search = ef_search
        self._max_spheres = max_spheres  # unused; accepted for _instantiate_filter parity
        if quantize != "auto" and quantize not in _QUANTIZE_LADDER:
            raise ValueError(
                f"quantize={quantize!r} not recognized; expected 'auto' or one of "
                f"{_QUANTIZE_LADDER} (None means unquantized IndexHNSWFlat)."
            )
        self._quantize_override = None if quantize == "auto" else quantize
        self._quantize_auto = quantize == "auto"
        self.quantize: Optional[str] = self._quantize_override
        self.index: Optional[faiss.Index] = None  # IndexHNSWFlat or IndexHNSWSQ

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

    @staticmethod
    def _code_bytes_per_vector(dim: int, quantize: Optional[str]) -> int:
        """Bytes one vector costs in storage at this quantization level,
        matching exactly what faiss's ScalarQuantizer.code_size reports (no
        need to hardcode bits/dim and get 6-bit/4-bit packing wrong): None is
        unquantized float32."""
        if quantize is None:
            return dim * 4
        return faiss.ScalarQuantizer(dim, _QTYPE_BY_NAME[quantize]).code_size

    @staticmethod
    def _trained_bytes_per_vector_ceiling(dim: int, quantize: Optional[str]) -> int:
        """Upper bound on the scalar quantizer's own trained parameters
        (per-dimension vmin/vdiff, 2 floats/dim for every QT_* type this
        filter uses): tiny next to n*code_size at realistic n, but real, so
        the pre-construction floor counts it rather than assuming it away."""
        return 0 if quantize is None else dim * 2 * 4

    def _min_footprint_bytes(self, n: int, quantize: Optional[str]) -> int:
        """Strict lower bound on the built index's size at this quantization
        level, knowable before construction: the vector codes, the level-0
        neighbor table (every node has 2*M level-0 slots), the per-node
        offsets/levels arrays, and the quantizer's trained parameters. Real
        usage is higher by the upper-level links, whose count depends on
        HNSW's random level assignment -- so this bound only rejects budgets
        that are certainly infeasible; _footprint_bytes() re-checks exactly
        once the graph exists."""
        return (
            n * self._code_bytes_per_vector(self.dim, quantize)
            + n * (2 * self.M) * 4 + (n + 1) * 8 + n * 4
            + self._trained_bytes_per_vector_ceiling(self.dim, quantize)
        )

    def _choose_quantize(self, n: int) -> Optional[str]:
        """The finest (least lossy) quantization in _QUANTIZE_LADDER whose
        _min_footprint_bytes fits max_memory, mirroring IVFFilter._choose_params:
        a tight budget degrades the index rather than refusing to build.
        Skipped entirely when max_memory is None (nothing to size against,
        so stay at the unquantized default) or the caller pinned a specific
        level via the constructor's `quantize` argument."""
        if self._quantize_override is not None or not self._quantize_auto:
            return self._quantize_override
        if self.max_memory is None:
            return None
        for candidate in _QUANTIZE_LADDER:
            if self._min_footprint_bytes(n, candidate) <= self.max_memory:
                return candidate
        return _QUANTIZE_LADDER[-1]  # coarsest; construct()'s post-build check reports the real shortfall

    def _footprint_bytes(self) -> int:
        """Exact resident size of the built index, summed from the faiss
        arrays directly rather than via serialize_index(): identical to
        within a ~200-byte header (verified against faiss 1.15.1) but O(1)
        instead of allocating a second full copy of the index to measure it."""
        if self.index is None:
            return 0
        hnsw = self.index.hnsw
        storage = faiss.downcast_index(self.index.storage)
        sq = getattr(storage, "sq", None)
        code_size = sq.code_size if sq is not None else self.dim * 4
        trained_bytes = sq.trained.size() * 4 if sq is not None else 0
        return (
            self.index.ntotal * code_size         # vector codes (float32 if unquantized)
            + hnsw.neighbors.size() * 4           # graph links, int32
            + hnsw.offsets.size() * 8             # per-node link offsets, int64
            + hnsw.levels.size() * 4              # per-node level, int32
            + trained_bytes                       # scalar quantizer's per-dim vmin/vdiff, if any
        )

    def _new_index(self, quantize: Optional[str]) -> faiss.Index:
        if quantize is None:
            index = faiss.IndexHNSWFlat(self.dim, self.M)
        else:
            index = faiss.IndexHNSWSQ(self.dim, _QTYPE_BY_NAME[quantize], self.M)
        index.hnsw.efConstruction = self.ef_construction
        index.hnsw.efSearch = self._ef_search
        return index

    def construct(self, vectors: np.ndarray) -> None:
        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        n = len(vecs)

        quantize = self._choose_quantize(n)

        if self.max_memory is not None and n > 0:
            lower_bound = self._min_footprint_bytes(n, quantize)
            if lower_bound > self.max_memory:
                raise InsufficientMemoryError(
                    f"max_memory={self.max_memory} bytes is too small for HNSWFilter to hold "
                    f"n={n} vectors of dim={self.dim} at M={self.M}: even at the coarsest "
                    f"quantization ({quantize!r}), the vector codes plus the level-0 graph alone "
                    f"need at least {lower_bound} bytes."
                )

        self.quantize = quantize
        self.index = self._new_index(quantize)
        if n > 0:
            self.index.train(vecs)
            self.index.add(vecs)

        if self.max_memory is not None:
            actual = self._footprint_bytes()
            if actual > self.max_memory:
                self.index = None
                raise InsufficientMemoryError(
                    f"max_memory={self.max_memory} bytes is too small for HNSWFilter: the built "
                    f"index over n={n} vectors of dim={self.dim} at M={self.M}, quantize={quantize!r} "
                    f"occupies {actual} bytes."
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
