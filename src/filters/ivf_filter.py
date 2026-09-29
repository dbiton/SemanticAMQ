"""IVFFilter -- semantic AMQ baseline built on FAISS's IVF (Inverted File)
index with product-quantized codes (IndexIVFPQ).

Like BallCoverFilter, a memory budget picks a compact per-point encoding
(`pq_m` sub-quantizers at `nbits` each); unlike it, construct vectors are
first assigned to one of `nlist` Voronoi cells, and a query only
decodes/compares codes in the `nprobe` nearest cells -- sublinear query, at
the cost of two lossy steps (coarse assignment + PQ residual) instead of
BallCoverFilter's single exact covering-radius test. Because both steps are
lossy, the estimated nearest-neighbor distance can quietly under-report the
true distance -- an FPR mode with no closed-form "structural" formula.

Memory sizing (construct-time, since it needs n): nlist centroids and the PQ
codebook are fixed overhead independent of pq_m; nlist is shrunk first, then
nbits, until per-vector codes get at least half the budget. If even the
smallest configuration overruns max_memory, construct() raises
InsufficientMemoryError rather than shipping an over-budget index.

Accept radius: query() thresholds the index's ESTIMATED nearest-neighbor
distance, and both of IVFPQ's lossy steps bias that estimate upward -- the
PQ code cannot represent the residual exactly, so a pair at true distance 0
comes back at the codes' reconstruction error. At tau=0.1 on 384-d
embeddings that bias is larger than tau itself: measured on the Quora
blocklist workload at |B|=4,000 and 10% of raw storage (nlist=63, pq_m=64,
nbits=7), pairs whose true distance was <= 0.1 came back with a median
estimate of 0.294. Thresholding those at tau flagged 3 of 38 true
duplicates -- FNR 0.92, FPR 0.00, i.e. a filter that answers "not a
duplicate" to almost everything, which is exactly what the equal-memory
sweep reported for IVFFilter on three of its five corpora.

The estimate is still strongly discriminative there (true duplicates
median 0.294, true non-duplicates median 0.994), so what was broken was the
decision rule, not the index: `accept_radius_scale` multiplies tau to give
the radius query() actually tests, and runner.py calibrates it on held-out
labeled data exactly as it calibrates nprobe. At the same budget that scale
reaches FPR 0.000 / FNR 0.263 (F1 0.836 against 0.146 at scale 1.0). It buys
no memory and FPR/FNR are still scored against the true tau, so too generous
a scale is paid for in FPR. Default 1.0 = the old exact-tau rule.

FAISS's IVFPQ precomputed ADC table is explicitly disabled (see
_PRECOMPUTED_TABLE_DISABLED): faiss switches it on during train(), it costs
nlist * pq_m * 2**nbits * 4 bytes of resident memory -- tens of MB, often
several times the whole budget -- and write_index() does *not* serialize it,
so it is invisible to a serialize_index()-based measurement. It buys only
query speed, which is not a trade an equal-memory comparison may make
silently. get_bytesize() counts it anyway, so re-enabling it could never
reintroduce a silent undercount."""

from typing import List, Optional

import faiss
import numpy as np

from filters.filter import Filter, InsufficientMemoryError

_PQ_MAX_NBITS = 8

# What an IndexIVFPQ costs beyond its centroids/codebook/codes/ids: a fixed
# index header plus one int64 length field per inverted list. Measured
# against faiss 1.15.1, where predicted == serialize_index() exactly across
# nlist in 1..1024 and every (pq_m, nbits) tried. Small, but at a tight
# budget it is the difference between _choose_params believing a config fits
# and construct()'s post-build check rejecting it -- which made feasibility
# non-monotonic in the budget (a 300KB budget failing where 250KB succeeded).
# If a future faiss adds to this, that post-build check still catches it.
_SERIALIZED_HEADER_BYTES = 180
_SERIALIZED_BYTES_PER_LIST = 8

# faiss's sentinel for "never build the precomputed table". 0 means "decide
# for me", and faiss then decides yes for METRIC_L2 with a flat quantizer.
_PRECOMPUTED_TABLE_DISABLED = -1


class IVFFilter(Filter):
    """FAISS IndexIVFPQ wrapped as a semantic AMQ filter: query is an
    approximate 1-NN search over PQ-compressed codes within the nprobe
    nearest coarse cells, "duplicate" iff the estimated nearest-neighbor
    distance is within tau.

    tau is a pure query-time knob (like HNSWFilter, unlike
    DSBFFilter/LSBFFilter/DSHFilter): nothing about IVFPQ's structure depends
    on the target distance, only on dim/n/max_memory.
    """

    def __init__(
        self,
        dim: int,
        distance_threshold: float,
        max_memory: Optional[int] = None,
        nlist: Optional[int] = None,
        nprobe: int = 8,
        pq_m: Optional[int] = None,
        max_spheres: Optional[int] = None,
        seed: int = 42,
        accept_radius_scale: float = 1.0,
    ):
        super().__init__(dim, max_memory=max_memory)
        self.distance_threshold = float(distance_threshold)
        # See the module docstring's "Accept radius" section. 1.0 reproduces
        # the exact-tau rule, so every caller that does not calibrate this
        # (every evaluation outside the equal-memory sweep) is unaffected.
        self.accept_radius_scale = float(accept_radius_scale)
        self._nlist_override = nlist
        self._nprobe = nprobe
        self._pq_m_override = pq_m
        self._max_spheres = max_spheres  # unused; accepted for _instantiate_filter parity
        self.seed = seed

        self.nlist: Optional[int] = None
        self.pq_m: Optional[int] = None
        self.pq_nbits: Optional[int] = None
        self.index: Optional[faiss.Index] = None  # IndexIVFPQ, or IndexIVFFlat for the n<2 fallback
        self._constructed = False

    @property
    def nprobe(self) -> int:
        """A property, not a plain attribute, so the equal-memory
        calibration dial's `setattr(filter_instance, "nprobe", value)` on an
        already-built instance actually reaches the underlying faiss index."""
        return self.index.nprobe if self.index is not None else self._nprobe

    @nprobe.setter
    def nprobe(self, value: int) -> None:
        self._nprobe = value
        if self.index is not None:
            self.index.nprobe = value

    @staticmethod
    def _code_size(pq_m: int, nbits: int) -> int:
        """Bytes per PQ code, as faiss packs it: pq_m sub-quantizer indices
        of nbits each, rounded up to a whole byte."""
        return (pq_m * nbits + 7) // 8

    @classmethod
    def _choose_pq_m(cls, dim: int, byte_budget: int, nbits: int) -> int:
        """Largest divisor of dim whose packed code fits byte_budget, i.e.
        the finest PQ code (most sub-quantizers) the per-vector budget
        allows. nbits matters: at nbits=4 one byte holds two sub-quantizers,
        so charging a byte each -- as this did before -- threw away half the
        per-vector budget whenever the nbits shrink loop had fired."""
        ceiling = max(1, min(dim, (max(1, int(byte_budget)) * 8) // max(nbits, 1)))
        for m in range(ceiling, 0, -1):
            if dim % m == 0:
                return m
        return 1

    def _next_smaller_pq_m(self, pq_m: int) -> int:
        """The next divisor of dim strictly below pq_m (1 always qualifies).
        Strictly decreasing, so callers stepping down through it terminate."""
        return next(m for m in range(pq_m - 1, 0, -1) if self.dim % m == 0)

    @staticmethod
    def _pq_nbits_for(n: int) -> Optional[int]:
        """Largest nbits (<= 8) whose 2**nbits codebook can be trained on n
        points, or None if there aren't even enough for a 1-bit codebook."""
        nbits = 0
        while nbits < _PQ_MAX_NBITS and (1 << (nbits + 1)) <= n:
            nbits += 1
        return nbits if nbits >= 1 else None

    def _overhead_bytes(self, nlist: int, nbits: int) -> int:
        """Fixed cost independent of pq_m: nlist coarse-quantizer centroids
        plus the 2**nbits-entry PQ codebook, as full-dimension float32s."""
        return (nlist * self.dim + (1 << nbits) * self.dim) * 4

    def _predicted_bytes(self, n: int, nlist: int, pq_m: int, nbits: int) -> int:
        """Total footprint of the IndexIVFPQ this configuration would build,
        matching get_bytesize() exactly: fixed overhead, per vector its
        packed PQ code and the int64 id faiss stores alongside it in the
        inverted list, and faiss's own header/per-list bookkeeping."""
        return (
            self._overhead_bytes(nlist, nbits)
            + n * (self._code_size(pq_m, nbits) + 8)
            + _SERIALIZED_HEADER_BYTES
            + _SERIALIZED_BYTES_PER_LIST * nlist
        )

    def _per_vector_floor_bytes(self, n: int) -> int:
        """Bytes no configuration can avoid spending on the inverted lists:
        one code byte (the smallest a packed PQ code ever gets) plus the
        int64 id faiss stores beside it, per vector. Overhead must be sized
        against what is left over after this, not against the raw budget --
        at tight budgets this floor is most of the budget."""
        return n * (1 + 8)

    def _min_footprint_bytes(self, n: int) -> int:
        """Cheapest index this filter can possibly build for n vectors: one
        coarse cell, a 1-bit codebook, a single sub-quantizer. Almost
        entirely _per_vector_floor_bytes, which no shrinking touches."""
        return self._predicted_bytes(n, nlist=1, pq_m=1, nbits=1)

    def _affordable_nlist(self, nbits: int, overhead_cap: int) -> int:
        """Largest nlist whose overhead at this nbits fits overhead_cap
        (may be < 1, meaning the codebook alone already overruns it).
        Closed form, rather than decrementing one cell at a time from
        O(sqrt(n))."""
        return overhead_cap // (self.dim * 4) - (1 << nbits)

    def _choose_params(self, n: int) -> None:
        nbits = self._pq_nbits_for(n) or 1
        nlist = self._nlist_override
        if nlist is None:
            nlist = max(1, round(n ** 0.5))
        target_nlist = max(1, min(nlist, n))
        nlist = target_nlist

        if self.max_memory is not None:
            # Overhead gets at most half the budget, and never the part of
            # it the per-vector floor and faiss's own bookkeeping have
            # already spoken for -- the second cap is what keeps a tight
            # budget feasible at all.
            overhead_cap = min(
                self.max_memory // 2,
                self.max_memory - self._per_vector_floor_bytes(n)
                - _SERIALIZED_HEADER_BYTES - _SERIALIZED_BYTES_PER_LIST,
            )
            # Shrink nlist first, dropping nbits only once no nlist >= 1
            # fits, then re-deriving nlist at the smaller codebook. nlist
            # goes first because it is the cheaper concession: fewer coarse
            # cells costs selectivity and query speed, whereas every nbits
            # step *halves* the codebook resolution every code is quantized
            # against -- the most accuracy-destructive dial available. The
            # old order (nbits first) reached configurations like nbits=1,
            # where PQ has two centroids per subspace and is effectively
            # noise. Re-deriving nlist afterwards matters too: shrinking it
            # against the *original* nbits left it crushed at 1 (an
            # exhaustive scan, IVF's whole point gone) when the smaller
            # codebook would comfortably have paid for a few dozen cells.
            while True:
                affordable = self._affordable_nlist(nbits, overhead_cap)
                if affordable >= 1:
                    nlist = max(1, min(target_nlist, int(affordable)))
                    break
                if nbits <= 1:
                    nlist = 1
                    break
                nbits -= 1

        if self._pq_m_override is not None:
            pq_m = self._pq_m_override
        elif self.max_memory is not None:
            remaining = self.max_memory - self._overhead_bytes(nlist, nbits)
            # 8 bytes/vector for the int64 id faiss stores next to each code.
            per_vector_budget = max(1, int(remaining / max(n, 1)) - 8)
            pq_m = self._choose_pq_m(self.dim, per_vector_budget, nbits)
            # _choose_pq_m sizes one vector at a time, so integer rounding
            # there can still leave the total a hair over. Step down through
            # divisors of dim until the whole index fits, so construct()
            # only ever reports infeasible when it genuinely is.
            while pq_m > 1 and self._predicted_bytes(n, nlist, pq_m, nbits) > self.max_memory:
                pq_m = self._next_smaller_pq_m(pq_m)
            if nlist > 1 and self._predicted_bytes(n, nlist, pq_m, nbits) > self.max_memory:
                # pq_m has bottomed out; buy the remaining bytes back from
                # nlist, whose cost is linear (one centroid plus one list
                # length field each). The floor check above already proved
                # nlist=1 fits, so this always converges to a feasible
                # config rather than to a spurious InsufficientMemoryError.
                slack = self.max_memory - self._predicted_bytes(n, 0, pq_m, nbits)
                per_list = self.dim * 4 + _SERIALIZED_BYTES_PER_LIST
                nlist = max(1, min(nlist, slack // per_list))
        else:
            pq_m = self._choose_pq_m(self.dim, self.dim, nbits)

        self.nlist, self.pq_m, self.pq_nbits = nlist, pq_m, nbits

    def construct(self, vectors: np.ndarray) -> None:
        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        n = len(vecs)
        self._constructed = True
        if n == 0:
            # nlist=1 rather than None: runner.py's nprobe calibration dial
            # reads self.nlist on the built instance, and None there raises
            # a TypeError that takes down the whole sweep job.
            self.nlist, self.pq_m, self.pq_nbits = 1, None, None
            self.index = None
            return

        if self.max_memory is not None:
            floor_bytes = self._min_footprint_bytes(n)
            if floor_bytes > self.max_memory:
                raise InsufficientMemoryError(
                    f"max_memory={self.max_memory} bytes is too small for IVFFilter to index "
                    f"n={n} vectors: even one coarse cell with a 1-bit, single-sub-quantizer PQ "
                    f"code needs {floor_bytes} bytes, dominated by the unavoidable {n * 9} bytes "
                    f"of 1-byte codes plus int64 inverted-list ids. Only a geometric summary "
                    f"(e.g. BallCoverFilter) can operate under this budget."
                )

        quantizer = faiss.IndexFlatL2(self.dim)
        if n < 2:
            # PQ needs >= 2 training points for even a 1-bit codebook (faiss
            # hard-errors below that); fall back to an uncompressed IndexIVFFlat.
            self.nlist, self.pq_m, self.pq_nbits = 1, None, None
            index = faiss.IndexIVFFlat(quantizer, self.dim, 1)
            index.cp.min_points_per_centroid = 1
            index.cp.seed = self.seed
        else:
            self._choose_params(n)
            index = faiss.IndexIVFPQ(quantizer, self.dim, self.nlist, self.pq_m, self.pq_nbits)
            index.cp.min_points_per_centroid = 1
            index.cp.seed = self.seed
            index.pq.cp.min_points_per_centroid = 1
            index.pq.cp.seed = self.seed
            # Must be set before train(), which is where faiss would
            # otherwise allocate the table. See the module docstring.
            index.use_precomputed_table = _PRECOMPUTED_TABLE_DISABLED
        index.train(vecs)
        index.add(vecs)
        index.nprobe = self._nprobe
        self.index = index

        if self.max_memory is not None:
            actual = self.get_bytesize()
            if actual > self.max_memory:
                self.index = None
                raise InsufficientMemoryError(
                    f"max_memory={self.max_memory} bytes is too small for IVFFilter: the "
                    f"smallest index it could build over n={n} vectors of dim={self.dim} "
                    f"(nlist={self.nlist}, pq_m={self.pq_m}, nbits={self.pq_nbits}) occupies "
                    f"{actual} bytes."
                )

    def query_squared_nn_distance(self, vectors: np.ndarray) -> np.ndarray:
        """The index's ESTIMATE of each query's squared nearest-neighbor
        distance, np.inf where the search found nothing (faiss's id=-1).

        Split out of query() so runner.py's calibration can score every
        accept radius from one search pass instead of re-searching per
        radius -- the radius is a comparison against these numbers, it does
        not change the search that produced them."""
        if not self._constructed:
            raise ValueError("IVFFilter has not been constructed.")
        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        if len(vecs) == 0 or self.index is None or self.index.ntotal == 0:
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

    def _precomputed_table_bytes(self) -> int:
        """Resident size of faiss's IVFPQ precomputed ADC table. Zero while
        _PRECOMPUTED_TABLE_DISABLED is in force, but measured rather than
        assumed: serialize_index() cannot see this table, and assuming it
        away is exactly how it went uncounted before."""
        table = getattr(self.index, "precomputed_table", None)
        return table.size() * 4 if table is not None else 0

    def get_bytesize(self) -> int:
        if self.index is None:
            return 0
        return len(faiss.serialize_index(self.index)) + self._precomputed_table_bytes()
