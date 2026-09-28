"""HNSWFilter -- semantic AMQ baseline built directly on FAISS's HNSW
approximate nearest-neighbor index. Unlike every other filter here (besides
IVFFilter), it answers via a real approximate NN search: no structural false
positives, but it trades away BallCoverFilter's zero-false-negative guarantee
(HNSW can miss the true neighbor, governed by ef_search/ef_construction/M).

Memory *is* budget-tunable on both of HNSW's two axes:

  - the vector storage's precision: `quantize` picks among IndexHNSWFlat
    (float32, lossless), IndexHNSWSQ at fp16/8bit/6bit/4bit-per-dimension
    (faiss's ScalarQuantizer) and, below the scalar quantizer's floor,
    IndexHNSWPQ at "pq<m>" (m product-quantizer sub-vectors, 8 bits each,
    so m bytes per vector) -- the same codes IVFFilter stores. The scalar
    ladder alone stopped at 4 bits/dim = dim/2 bytes per vector, which at
    dim=384 is 192 bytes and, at n=80,000 construct vectors, 15.4MB of codes
    before a single graph link. That was above the *entire* top budget of the
    equal-memory sweep (12.3MB), so HNSW reported infeasible at all 85
    budget x dataset points no matter what else was tuned -- a quantization
    gap between baselines (IVFFilter's PQ codes reach ~1 byte/vector), not a
    statement about HNSW's merit at those budgets. The PQ rungs close it.
  - the graph's degree, `M`: the level-0 neighbour table alone costs
    n * 2M * 4 bytes, 20.5MB at M=32 and n=80,000 -- again more than the
    whole top budget, and untouchable by any amount of quantization. M is
    therefore a construction-time config dial in the equal-memory sweep (see
    runner.py::_CONSTRUCTION_TIME_CONFIG_DIALS), not a fixed 32.

Left at the default "auto", the finest (least lossy) precision that fits
max_memory is chosen automatically -- so, like IVFFilter, a tight budget
degrades the index's accuracy rather than the filter simply refusing to
build at all.

construct() still raises InsufficientMemoryError when not even the coarsest
rung of the ladder fits max_memory, so an equal-memory sweep records that
budget as infeasible instead of plotting an over-budget point (a 1MB budget
silently holding a 39MB index) as if it were a legitimate memory/accuracy
tradeoff."""

from typing import List, Optional

import faiss
import numpy as np

from filters.filter import Filter, InsufficientMemoryError

# Scalar-quantizer rungs, finest (most bytes/dim, least lossy) to coarsest.
# None means IndexHNSWFlat (full float32, faiss has no "QT_fp32" scalar
# quantizer type). _quantize_ladder() appends the product-quantizer rungs
# below these; _choose_quantize walks the whole thing in order so the first
# rung that fits wins.
_SQ_LADDER = (None, "fp16", "8bit", "6bit", "4bit")

_QTYPE_BY_NAME = {
    "fp16": faiss.ScalarQuantizer.QT_fp16,
    "8bit": faiss.ScalarQuantizer.QT_8bit,
    "6bit": faiss.ScalarQuantizer.QT_6bit,
    "4bit": faiss.ScalarQuantizer.QT_4bit,
}

# faiss's IndexHNSWPQ always uses 8-bit sub-quantizer indices (one byte per
# sub-vector), so a "pq<m>" rung costs exactly m bytes per vector.
_PQ_NBITS = 8
# A PQ codebook is 2**_PQ_NBITS centroids per subspace, and faiss's k-means
# refuses to fit more centroids than it has training points. Below this many
# construct vectors the PQ rungs are dropped from the ladder rather than
# attempted and crashed on.
_PQ_MIN_TRAIN = 1 << _PQ_NBITS


class HNSWFilter(Filter):
    """FAISS IndexHNSWFlat/IndexHNSWSQ/IndexHNSWPQ wrapped as a semantic AMQ
    filter: query is a real 1-NN search, "duplicate" iff the nearest neighbor
    found is within tau (times accept_radius_scale -- see the module
    docstring's note on the PQ rungs' inflated distance estimates). tau is a
    pure query-time knob -- nothing about the threshold is baked into the
    graph, unlike DSHFilter/DSBFFilter/LSBFFilter."""

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
        accept_radius_scale: float = 1.0,
    ):
        super().__init__(dim, max_memory=max_memory)
        self.distance_threshold = float(distance_threshold)
        # Multiplies tau to give the radius query() tests. Needed for the same
        # reason IVFFilter has it: on the PQ rungs the returned distance is a
        # reconstruction-error-inflated ESTIMATE, so testing it against the
        # exact tau rejects nearly every true duplicate. runner.py calibrates
        # it on held-out data; 1.0 is the old exact-tau rule, so callers that
        # do not calibrate it are unaffected.
        self.accept_radius_scale = float(accept_radius_scale)
        self.M = M
        self.ef_construction = ef_construction
        self._ef_search = ef_search
        self._max_spheres = max_spheres  # unused; accepted for _instantiate_filter parity
        if quantize != "auto" and quantize not in self._quantize_ladder():
            raise ValueError(
                f"quantize={quantize!r} not recognized; expected 'auto' or one of "
                f"{self._quantize_ladder()} (None means unquantized IndexHNSWFlat)."
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
    def _pq_m(quantize: Optional[str]) -> Optional[int]:
        """The sub-vector count of a "pq<m>" rung, or None for a scalar rung."""
        if isinstance(quantize, str) and quantize.startswith("pq"):
            return int(quantize[2:])
        return None

    def _pq_m_ladder(self) -> tuple:
        """Sub-vector counts for the product-quantizer rungs, finest first.

        Only divisors of dim (faiss's ProductQuantizer hard-asserts
        dim % pq_m == 0), and only those strictly cheaper per vector than the
        coarsest scalar rung, so the combined ladder stays monotonically
        decreasing in bytes/vector and _choose_quantize's "first one that
        fits" stays equivalent to "finest one that fits"."""
        sq_floor = self._code_bytes_per_vector(self.dim, "4bit")
        return tuple(
            m for m in range(self.dim, 0, -1)
            if self.dim % m == 0 and self._code_bytes_per_vector(self.dim, f"pq{m}") < sq_floor
        )

    def _quantize_ladder(self, n: Optional[int] = None) -> tuple:
        """The full rung ladder, finest to coarsest: scalar quantization down
        to 4 bits/dim, then product quantization below that. n, when given,
        drops the PQ rungs for a construct set too small to fit a PQ codebook
        (see _PQ_MIN_TRAIN)."""
        if n is not None and n < _PQ_MIN_TRAIN:
            return _SQ_LADDER
        return _SQ_LADDER + tuple(f"pq{m}" for m in self._pq_m_ladder())

    @classmethod
    def _code_bytes_per_vector(cls, dim: int, quantize: Optional[str]) -> int:
        """Bytes one vector costs in storage at this rung, matching exactly
        what faiss reports for it (no need to hardcode bits/dim and get
        6-bit/4-bit packing wrong): None is unquantized float32, "pq<m>" is m
        8-bit sub-quantizer indices."""
        if quantize is None:
            return dim * 4
        pq_m = cls._pq_m(quantize)
        if pq_m is not None:
            return (pq_m * _PQ_NBITS + 7) // 8
        return faiss.ScalarQuantizer(dim, _QTYPE_BY_NAME[quantize]).code_size

    @classmethod
    def _trained_bytes_per_vector_ceiling(cls, dim: int, quantize: Optional[str]) -> int:
        """Upper bound on the quantizer's own trained parameters: a scalar
        quantizer's per-dimension vmin/vdiff (2 floats/dim for every QT_*
        type this filter uses), or a PQ codebook's 2**_PQ_NBITS centroids per
        subspace (2**nbits * dim floats in total, independent of pq_m).

        Tiny next to n*code_size at realistic n for the scalar rungs, but the
        PQ codebook is 393KB at dim=384 -- a real share of a tight budget --
        so the pre-construction floor counts it rather than assuming it away."""
        if quantize is None:
            return 0
        if cls._pq_m(quantize) is not None:
            return (1 << _PQ_NBITS) * dim * 4
        return dim * 2 * 4

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

    def _expected_upper_level_link_bytes(self, n: int) -> int:
        """What the graph's levels ABOVE level 0 add to the neighbor table.

        faiss assigns node levels geometrically, so ~n/M**l nodes reach level
        l and each holds M slots there: sum over l >= 1 of (n/M**l)*M =
        n*M/(M-1) int32 slots. Measured at n=20,000, M=4: predicted 26,667
        slots, actual 26,752.

        An expectation, not a bound -- level assignment is random -- which is
        why construct() still re-checks the built index exactly and steps down
        the ladder if it overran. Its purpose is to stop _choose_quantize
        picking a rung that misses by ~17% (the M=4 case) and turning a budget
        that a coarser rung fits comfortably into an infeasible row."""
        return int(n * self.M / max(self.M - 1, 1)) * 4

    def _expected_footprint_bytes(self, n: int, quantize: Optional[str]) -> int:
        """_min_footprint_bytes plus the expected upper-level links: what the
        built index should actually weigh, for choosing a rung with."""
        return self._min_footprint_bytes(n, quantize) + self._expected_upper_level_link_bytes(n)

    def _choose_quantize(self, n: int) -> Optional[str]:
        """The finest (least lossy) rung of _quantize_ladder() whose
        _expected_footprint_bytes fits max_memory, mirroring
        IVFFilter._choose_params: a tight budget degrades the index rather
        than refusing to build. Skipped entirely when max_memory is None
        (nothing to size against, so stay at the unquantized default) or the
        caller pinned a specific level via the constructor's `quantize`
        argument."""
        if self._quantize_override is not None or not self._quantize_auto:
            return self._quantize_override
        if self.max_memory is None:
            return None
        ladder = self._quantize_ladder(n)
        for candidate in ladder:
            if self._expected_footprint_bytes(n, candidate) <= self.max_memory:
                return candidate
        return ladder[-1]  # coarsest; construct()'s post-build check reports the real shortfall

    def _footprint_bytes(self) -> int:
        """Exact resident size of the built index, summed from the faiss
        arrays directly rather than via serialize_index(): identical to
        within a ~200-byte header (verified against faiss 1.15.1) but O(1)
        instead of allocating a second full copy of the index to measure it."""
        if self.index is None:
            return 0
        hnsw = self.index.hnsw
        storage = faiss.downcast_index(self.index.storage)
        # IndexScalarQuantizer exposes .sq, IndexPQ exposes .pq, IndexFlat
        # (the unquantized rung) neither.
        sq = getattr(storage, "sq", None)
        pq = getattr(storage, "pq", None)
        if sq is not None:
            code_size, trained_bytes = sq.code_size, sq.trained.size() * 4
        elif pq is not None:
            code_size, trained_bytes = pq.code_size, pq.centroids.size() * 4
        else:
            code_size, trained_bytes = self.dim * 4, 0
        return (
            self.index.ntotal * code_size         # vector codes (float32 if unquantized)
            + hnsw.neighbors.size() * 4           # graph links, int32
            + hnsw.offsets.size() * 8             # per-node link offsets, int64
            + hnsw.levels.size() * 4              # per-node level, int32
            + trained_bytes                       # quantizer's trained params (SQ vmin/vdiff or PQ codebook)
        )

    def _new_index(self, quantize: Optional[str]) -> faiss.Index:
        pq_m = self._pq_m(quantize)
        if quantize is None:
            index = faiss.IndexHNSWFlat(self.dim, self.M)
        elif pq_m is not None:
            # The graph is built over PQ-reconstructed distances too, not just
            # searched with them, so a coarse pq_m costs link quality on top
            # of code precision. That is the honest price of the rung, and the
            # same tradeoff IVFFilter's coarse-quantizer assignment makes.
            index = faiss.IndexHNSWPQ(self.dim, pq_m, self.M)
        else:
            index = faiss.IndexHNSWSQ(self.dim, _QTYPE_BY_NAME[quantize], self.M)
        index.hnsw.efConstruction = self.ef_construction
        index.hnsw.efSearch = self._ef_search
        return index

    def _construct_candidates(self, n: int) -> list:
        """The rungs construct() may build, coarsest-ward from the one
        _choose_quantize picked: a pinned `quantize` gives exactly that rung
        (the caller asked for it and gets it or an error), "auto" gives the
        chosen rung and every coarser one as fallbacks for the post-build
        check to fall through to."""
        chosen = self._choose_quantize(n)
        if self._quantize_override is not None or not self._quantize_auto or self.max_memory is None:
            return [chosen]
        ladder = self._quantize_ladder(n)
        return list(ladder[ladder.index(chosen):])

    def construct(self, vectors: np.ndarray) -> None:
        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        n = len(vecs)
        candidates = self._construct_candidates(n)

        if self.max_memory is not None and n > 0:
            # The strict lower bound of the COARSEST rung available: if even
            # that overruns, no amount of stepping down the ladder helps, so
            # report infeasible without building anything.
            lower_bound = min(self._min_footprint_bytes(n, c) for c in candidates)
            if lower_bound > self.max_memory:
                raise InsufficientMemoryError(
                    f"max_memory={self.max_memory} bytes is too small for HNSWFilter to hold "
                    f"n={n} vectors of dim={self.dim} at M={self.M}: even at the coarsest "
                    f"available rung ({candidates[-1]!r}), the vector codes plus the level-0 "
                    f"graph alone need at least {lower_bound} bytes."
                )

        # _choose_quantize sizes rungs against an ESTIMATE of the upper-level
        # links (their count is random), so the rung it picks can still overrun
        # once built. Step to the next coarser rung and rebuild rather than
        # reporting the whole budget infeasible -- without this, a budget that
        # e.g. pq4 fits with room to spare was recorded as an infeasible row
        # because pq8 missed it by 3%.
        overruns = []
        for quantize in candidates:
            if self.max_memory is not None and n > 0 and \
                    self._min_footprint_bytes(n, quantize) > self.max_memory:
                continue
            self.quantize = quantize
            self.index = self._new_index(quantize)
            if n > 0:
                self.index.train(vecs)
                self.index.add(vecs)
            if self.max_memory is None:
                return
            actual = self._footprint_bytes()
            if actual <= self.max_memory:
                return
            overruns.append(f"{quantize!r}:{actual}")
            self.index = None

        raise InsufficientMemoryError(
            f"max_memory={self.max_memory} bytes is too small for HNSWFilter: every rung it "
            f"could build over n={n} vectors of dim={self.dim} at M={self.M} overran it "
            f"(rung:bytes = {', '.join(overruns) if overruns else 'none buildable'})."
        )

    def query_squared_nn_distance(self, vectors: np.ndarray) -> np.ndarray:
        """The index's ESTIMATE of each query's squared nearest-neighbor
        distance, np.inf where the search found nothing.

        Split out of query() so runner.py's calibration can score every
        accept radius from one search pass instead of re-searching per radius
        -- which matters more here than for IVFFilter, since the radius grid
        would otherwise multiply against both the ef_search grid and the M
        grid. faiss fills an unfilled slot with id=-1 and distance=FLT_MAX;
        the explicit id check mirrors IVFFilter so neither filter relies on
        that sentinel's exact value."""
        if self.index is None:
            raise ValueError("HNSWFilter has not been constructed.")
        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        if len(vecs) == 0 or self.index.ntotal == 0:
            return np.full(len(vecs), np.inf, dtype=np.float64)
        sq_dist, ids = self.index.search(vecs, 1)
        out = sq_dist[:, 0].astype(np.float64)
        return np.where(ids[:, 0] != -1, out, np.inf)

    def query(self, vectors: np.ndarray, tau: Optional[float] = None) -> List[bool]:
        if tau is None:
            tau = self.distance_threshold
        radius = float(tau) * self.accept_radius_scale
        return (self.query_squared_nn_distance(vectors) <= radius ** 2).tolist()

    def insert(self, vectors: np.ndarray) -> None:
        if self.index is None:
            self.construct(vectors)
            return
        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        if len(vecs) > 0:
            self.index.add(vecs)

    def get_bytesize(self) -> int:
        return self._footprint_bytes()
