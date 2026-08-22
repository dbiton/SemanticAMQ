"""
IVFFilter -- semantic AMQ baseline built on FAISS's IVF (Inverted File) index
with product-quantized codes (IndexIVFPQ).

Unlike HNSWFilter (full-precision vectors, memory not budget-tunable) and like
BallCoverFilter (a memory budget picks a compact per-point encoding), IVFPQ
gives each construct vector a lossy `pq_m`-byte product-quantized code, so its
footprint scales with the memory budget the way the other compact filters'
does. What it adds over BallCoverFilter is coarse partitioning: construct
vectors are first assigned to one of `nlist` Voronoi cells (a k-means
quantizer over the *raw* vectors), and a query only decodes/compares codes in
the `nprobe` nearest cells rather than the whole construct set -- a
sublinear-query ANN structure, at the cost of two lossy steps (coarse
assignment, then PQ residual quantization) instead of BallCoverFilter's
single exact-covering-radius test.

Because both steps are lossy, "duplicate" here means "the nearest neighbor
found via IVF search of a PQ-decoded (ADC) distance is within tau" -- an
approximate distance to an approximate candidate set. Unlike HNSWFilter,
where the returned neighbor's distance is exact (so HNSWFilter's only error
mode is a missed true neighbor, i.e. false negatives from limited graph
recall), IVFPQ's *distance itself* is an estimate that can quietly
under-report the true distance to a genuine non-duplicate, producing a false
positive that isn't visible as a hash collision the way DSBF/LSBF/DSH's
errors are. There is no simple closed-form "structural" FPR for that ADC
error the way there is for a Bloom filter's load factor.

Memory-budget sizing (construct-time, since it needs n): the coarse
quantizer's `nlist` centroids and the PQ codebook are a fixed, budget-
independent overhead (nlist*dim*4 and 2**nbits*dim*4 bytes respectively,
regardless of pq_m -- pq_m only reshuffles that codebook total across
sub-quantizers); `nbits` is picked as large as the construct set can train
(same "largest nbits whose 2**nbits codebook has enough training points"
rule as BallCoverFilter._pq_nbits_for), then whatever budget remains per
vector, after that fixed overhead and an 8-byte-per-vector id-storage
allowance, sizes `pq_m` -- the largest divisor of `dim` that fits, i.e. the
finest (least lossy) code the remaining budget affords.
"""

from typing import List, Optional

import faiss
import numpy as np

from filters.filter import Filter

_PQ_MAX_NBITS = 8


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
    ):
        super().__init__(dim, max_memory=max_memory)
        self.distance_threshold = float(distance_threshold)
        self._nlist_override = nlist
        self._nprobe = nprobe
        self._pq_m_override = pq_m
        self._max_spheres = max_spheres  # unused; accepted for _instantiate_filter parity
        self.seed = seed

        self.nlist: Optional[int] = None
        self.pq_m: Optional[int] = None
        self.pq_nbits: Optional[int] = None
        self.index: Optional[faiss.IndexIVFPQ] = None
        self._constructed = False

    @property
    def nprobe(self) -> int:
        """Query-time cell count. A property (not a plain attribute) so the
        equal-memory calibration dial (_POST_CONSTRUCTION_CONFIG_DIALS, which
        works by `setattr(filter_instance, "nprobe", value)` on an already-
        built instance) actually reaches the underlying faiss index, which
        reads its own `index.nprobe` field at search time rather than
        anything on this wrapper."""
        return self.index.nprobe if self.index is not None else self._nprobe

    @nprobe.setter
    def nprobe(self, value: int) -> None:
        self._nprobe = value
        if self.index is not None:
            self.index.nprobe = value

    # ------------------------------------------------------------------ sizing

    @staticmethod
    def _choose_pq_m(dim: int, byte_budget: int) -> int:
        """Largest divisor of dim, capped at min(dim, byte_budget), i.e. the
        finest PQ code (most sub-quantizers) that fits the per-vector budget."""
        ceiling = max(1, min(dim, int(byte_budget)))
        for m in range(ceiling, 0, -1):
            if dim % m == 0:
                return m
        return 1

    @staticmethod
    def _pq_nbits_for(n: int) -> Optional[int]:
        """Largest nbits (<= 8) whose 2**nbits codebook can be trained on n
        points (with a 2x safety margin, mirroring
        BallCoverFilter._pq_nbits_for), or None if there aren't even enough
        points for a 1-bit codebook."""
        nbits = 0
        while nbits < _PQ_MAX_NBITS and (1 << (nbits + 1)) <= n:
            nbits += 1
        return nbits if nbits >= 1 else None

    def _overhead_bytes(self, nlist: int, nbits: int) -> int:
        """Fixed cost independent of pq_m: nlist coarse-quantizer centroids
        plus the 2**nbits-entry PQ codebook, both stored as float32 vectors
        of the full dimension (see module docstring -- codebook total size
        doesn't depend on how it's split across pq_m sub-quantizers)."""
        return (nlist * self.dim + (1 << nbits) * self.dim) * 4

    def _choose_params(self, n: int) -> None:
        nbits = self._pq_nbits_for(n) or 1
        nlist = self._nlist_override
        if nlist is None:
            nlist = max(1, round(n ** 0.5))
        nlist = max(1, min(nlist, n))

        if self.max_memory is not None:
            # nbits and nlist are themselves fixed overhead, independent of
            # pq_m -- at small n*dim they can be picked large enough to blow
            # the whole budget before a single vector is even encoded (the
            # PQ codebook alone is 2**nbits*dim*4 bytes). Shrink nbits first
            # (exponentially cheaper per step than nlist) and then nlist
            # until at least half the budget is left over for actual
            # per-vector codes, so a small budget still gets a usably-sized
            # (if coarse) index instead of silently overshooting max_memory.
            half = self.max_memory // 2
            while nbits > 1 and self._overhead_bytes(nlist, nbits) > half:
                nbits -= 1
            while nlist > 1 and self._overhead_bytes(nlist, nbits) > half:
                nlist -= 1

        if self._pq_m_override is not None:
            pq_m = self._pq_m_override
        elif self.max_memory is not None:
            remaining = self.max_memory - self._overhead_bytes(nlist, nbits)
            per_vector_budget = max(1, int(remaining / max(n, 1)) - 8)
            pq_m = self._choose_pq_m(self.dim, per_vector_budget)
        else:
            pq_m = self._choose_pq_m(self.dim, self.dim)

        self.nlist, self.pq_m, self.pq_nbits = nlist, pq_m, nbits

    # ------------------------------------------------------------------ Filter interface

    def construct(self, vectors: np.ndarray) -> None:
        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        n = len(vecs)
        self._constructed = True
        if n == 0:
            self.index = None
            return

        self._choose_params(n)
        quantizer = faiss.IndexFlatL2(self.dim)
        index = faiss.IndexIVFPQ(quantizer, self.dim, self.nlist, self.pq_m, self.pq_nbits)
        index.cp.min_points_per_centroid = 1
        index.cp.seed = self.seed
        index.pq.cp.min_points_per_centroid = 1
        index.pq.cp.seed = self.seed
        index.train(vecs)
        index.add(vecs)
        index.nprobe = self._nprobe
        self.index = index

    def query(self, vectors: np.ndarray, tau: Optional[float] = None) -> List[bool]:
        if not self._constructed:
            raise ValueError("IVFFilter has not been constructed.")
        if tau is None:
            tau = self.distance_threshold
        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        if len(vecs) == 0 or self.index is None or self.index.ntotal == 0:
            return [False] * len(vecs)
        sq_dist, ids = self.index.search(vecs, 1)
        found = ids[:, 0] != -1
        is_dup = found & (sq_dist[:, 0] <= float(tau) ** 2)
        return is_dup.tolist()

    def add(self, vectors: np.ndarray) -> None:
        if self.index is None:
            self.construct(vectors)
            return
        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        if len(vecs) > 0:
            self.index.add(vecs)

    def insert(self, vectors: np.ndarray) -> None:
        self.add(vectors)

    def get_bytesize(self) -> int:
        if self.index is None:
            return 0
        return len(faiss.serialize_index(self.index))
