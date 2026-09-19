import numpy as np
import faiss
import sys
from enum import Enum
from typing import List, Tuple, Optional
from hyperspheres_volume import estimate_hypersphere_volume
from filters.filter import Filter
from filters.ball_cover.ball_interfaces import ConstructInterface, InsertInterface
from filters.ball_cover.gonzales_construct import GonzalesConstruct
from filters.ball_cover.nearest_insert import NearestInsert
from filters.ball_cover.least_inflation_insert import LeastInflationInsert
from filters.ball_cover.exhaustive_merge_insert import ExhaustiveMergeInsert


class Quantize(Enum):
    NONE = "None"
    BF16 = "SQbf16"
    FP16 = "SQfp16"
    SQ8 = "SQ8"
    SQ6 = "SQ6"
    SQ4 = "SQ4"
    SQ4T = "SQ4_trained"
    SQ8T = "SQ8_trained"
    SQ2T = "SQ2_trained"
    PQ = "PQ"

PQ_MAX_NBITS = 8


class InsufficientMemoryError(ValueError):
    """Raised when max_memory is too small to construct the requested
    quantizer -- currently only reachable for Quantize.PQ, whose k-means
    codebook training needs at least 2 ball centers for even a 1-bit
    codebook (see BallCoverFilter._pq_nbits_for)."""


_QTYPES = {
    Quantize.BF16: faiss.ScalarQuantizer.QT_bf16,
    Quantize.FP16: faiss.ScalarQuantizer.QT_fp16,
    Quantize.SQ8: faiss.ScalarQuantizer.QT_8bit,
    Quantize.SQ6: faiss.ScalarQuantizer.QT_6bit,
    Quantize.SQ4: faiss.ScalarQuantizer.QT_4bit,
    Quantize.SQ4T: faiss.ScalarQuantizer.QT_4bit_tqmse,
    Quantize.SQ8T: faiss.ScalarQuantizer.QT_8bit_tqmse,
    # No plain (min-max) QT_2bit exists in faiss -- 2 bits/component is only
    # available as the MSE-trained TurboQuant variant, so SQ2T has no
    # untrained SQ2 counterpart the way SQ4T/SQ8T do.
    Quantize.SQ2T: faiss.ScalarQuantizer.QT_2bit_tqmse,
}


class BallCoverFilter(Filter):
    _BYTES_PER_FLOAT32 = 4
    _BYTES_PER_FLOAT64 = 8

    # Coverage-comparison safety margin: FAISS's IndexFlatL2.search() isn't
    # bit-reproducible across different query-batch shapes, so a ball whose
    # radius was grown to exactly match one measured distance can see a
    # slightly different distance later when re-queried in a bigger batch
    # (e.g. a no-false-negative sweep). sqrt() of a near-zero squared
    # distance (near-duplicate points, ~0 radius) amplifies that error --
    # measured up to ~9.1e-4 on real data at dim=384. 1e-3 covers that with
    # margin, matches this project's own tol=1e-3 precedent
    # (test_ball_cover_construction.py's _covering_ok), and stays far below
    # any real distance_threshold used in this project's sweeps.
    _COVERAGE_EPSILON = 1e-3

    def __init__(
        self,
        dim: int,
        construct_method: Optional[ConstructInterface] = None,
        insert_method: Optional[InsertInterface] = None,
        quantize: Quantize = Quantize.NONE,
        max_memory: Optional[int] = None,
        max_spheres: Optional[int] = None,
        insert_memory_percent: Optional[float] = 0.0
    ) -> None:
        super().__init__(dim, max_memory=max_memory)
        self.quantize = quantize
        # Constructed fresh here, not as parameter defaults: a mutable default
        # argument is evaluated once at def-time and shared by every call that
        # doesn't override it. Both defaults are stateless (harmless to
        # share), but strategies with real per-filter state (e.g.
        # MinHeapInsert's _sphere_versions/_minimal_pairs) must not be shared
        # this way if passed explicitly across filters in the same process.
        self.construct_method = construct_method if construct_method is not None else GonzalesConstruct()
        self.insert_method = insert_method if insert_method is not None else NearestInsert()
        self.insert_memory_percent = insert_memory_percent
        # Codebook state. Both are fixed once, by construct(), and never
        # retrained: insert() encodes new centers with whatever construct()
        # fit. Retraining would re-encode every stored center under a new
        # codebook while their radii still carry the old codebook's
        # inflation -- and the unquantized centers needed to recompute that
        # inflation no longer exist.
        self._pq_nbits: Optional[int] = None
        self.all_centers_index = self._make_faiss_index(self.quantize)
        self.max_spheres = (max_spheres if max_spheres is not None 
                    else self._calculate_max_spheres(self.max_memory))
        self.max_construct_spheres = self._calculate_max_construct_spheres(self.insert_memory_percent)
        self._constructed = False

        self._num_spheres = 0
        self._all_spheres = np.arange(self.max_spheres, dtype=np.int32)
        self._all_centers = None  # lazily rebuilt cache, not persistent state
        self._all_radiuses = np.empty((self.max_spheres,), dtype=np.float32)
        # Per-ball reconstruction error (||center - decode(encode(center))||),
        # 0 for Quantize.NONE: added onto each radius by _store_spheres, once,
        # to preserve the no-false-negative guarantee under lossy encoding.
        self._radius_inflation = np.zeros((self.max_spheres,), dtype=np.float32)

        self.insert_method.ensure_capacity(self.max_spheres)

    @property
    def all_spheres(self) -> np.ndarray:
        return self._all_spheres[:self._num_spheres]

    @property
    def all_centers(self) -> np.ndarray:
        if self._all_centers is None:
            self._all_centers = np.empty((len(self._all_radiuses), self.dim), dtype=np.float32)
            if self._num_spheres > 0:
                sphere_indices = np.ascontiguousarray(self.all_spheres, dtype=np.int64)
                self._all_centers[self.all_spheres] = self.all_centers_index.reconstruct_batch(sphere_indices)
        return self._all_centers[self.all_spheres]

    @property
    def all_radiuses(self) -> np.ndarray:
        return self._all_radiuses[self.all_spheres]

    @property
    def all_radius_inflation(self) -> np.ndarray:
        """Per-ball radius inflation (0 for Quantize.NONE) -- see
        _radius_inflation's docstring in __init__."""
        return self._radius_inflation[self.all_spheres]

    def _ensure_capacity(self, required_cap: int) -> None:
        current_cap = len(self._all_radiuses)
        if required_cap <= current_cap:
            return

        new_cap = max(current_cap * 2, required_cap)
        new_spheres = np.empty((new_cap,), dtype=np.int32)
        new_spheres[:current_cap] = self._all_spheres
        new_spheres[current_cap:] = np.arange(current_cap, new_cap, dtype=np.int32)
        self._all_spheres = new_spheres

        if self._all_centers is not None:
            new_centers = np.empty((new_cap, self.dim), dtype=np.float32)
            new_centers[:current_cap] = self._all_centers
            self._all_centers = new_centers

        new_radiuses = np.empty((new_cap,), dtype=np.float32)
        new_radiuses[:current_cap] = self._all_radiuses
        self._all_radiuses = new_radiuses

        new_inflation = np.zeros((new_cap,), dtype=np.float32)
        new_inflation[:current_cap] = self._radius_inflation
        self._radius_inflation = new_inflation

        self.insert_method.ensure_capacity(new_cap)

    def _get_new_index(self) -> int:
        if self._num_spheres == len(self._all_radiuses):
            self._ensure_capacity(len(self._all_radiuses) * 2)
        return self._all_spheres[self._num_spheres]

    def _reset_buffers(self, capacity: int) -> None:
        self._ensure_capacity(capacity)
        self._num_spheres = 0
        self._all_centers = None
        # Back to an untrained index: construct() has to run again to refit
        # the codebook before any sphere can be stored.
        self.all_centers_index = self._make_faiss_index(self.quantize)
        self._pq_nbits = None
        self._constructed = False
        self.insert_method.reset_buffers(capacity)

    def _calculate_max_spheres(self, max_memory: Optional[int]) -> int:
        if max_memory is None:
            # When unconstrained, fallback to standard integer bounds to avoid crash
            return sys.maxsize
        usable = max(0, max_memory - self._codebook_bytes())
        return max(1, usable // self.bytes_per_ball())

    def _calculate_max_construct_spheres(self, insertion_memory_percent: float) -> int:
        if insertion_memory_percent <= 0.0:
            return self.max_spheres

        if self.max_memory is not None:
            construct_memory = max(0, int(self.max_memory * (1.0 - insertion_memory_percent / 100.0)))
            return self._calculate_max_spheres(construct_memory)

        return max(1, int(self.max_spheres * (1.0 - insertion_memory_percent / 100.0)))

    @staticmethod
    def _choose_pq_m(dim: int) -> int:
        for m in range(dim, 0, -1):
            if dim % m == 0:
                return m
        return 1

    @staticmethod
    def _pq_nbits_for(n: int) -> Optional[int]:
        nbits = 0
        while nbits < PQ_MAX_NBITS and (1 << (nbits + 1)) <= n:
            nbits += 1
        return nbits if nbits >= 1 else None

    def _make_faiss_index(self, quantize: Quantize, pq_nbits: int = PQ_MAX_NBITS) -> faiss.IndexIDMap2:
        if quantize == Quantize.NONE:
            index = faiss.IndexFlatL2(self.dim)
        elif quantize == Quantize.PQ:
            pq_m = self._choose_pq_m(self.dim)
            index = faiss.IndexPQ(self.dim, pq_m, pq_nbits, faiss.METRIC_L2)
            # ST_PQ is asymmetric distance computation (ADC): the raw float32
            # query is scored against the *decoded* stored codes, so the query
            # is never encoded. That is what makes the one-sided radius
            # inflation in _store_spheres correct and tight -- see query().
            # ST_SDC/ST_polysemous encode the query too and would silently
            # break the zero-false-negative guarantee, so pin it.
            assert index.search_type == faiss.IndexPQ.ST_PQ, (
                f"BallCoverFilter requires ADC (ST_PQ); got search_type={index.search_type}"
            )
        else:
            index = faiss.IndexScalarQuantizer(self.dim, _QTYPES[quantize], faiss.METRIC_L2)
        return faiss.IndexIDMap2(index)

    def bytes_per_ball(self) -> int:
        index = self.all_centers_index
        if isinstance(index, faiss.IndexIDMap2):
            index = faiss.downcast_index(self.all_centers_index.index)
        if isinstance(index, faiss.IndexPQ):
            return index.pq.code_size + self._BYTES_PER_FLOAT32
        if isinstance(index, faiss.IndexScalarQuantizer):
            return index.sq.code_size + self._BYTES_PER_FLOAT32
        return (self.dim + 1) * self._BYTES_PER_FLOAT32

    def _codebook_bytes(self) -> int:
        index = self.all_centers_index
        if isinstance(index, faiss.IndexIDMap2):
            index = faiss.downcast_index(self.all_centers_index.index)
        if isinstance(index, faiss.IndexPQ):
            return int(faiss.vector_to_array(index.pq.centroids).nbytes)
        if isinstance(index, faiss.IndexScalarQuantizer):
            trained = faiss.vector_to_array(index.sq.trained)
            if len(trained) == 0:
                probe = faiss.IndexScalarQuantizer(self.dim, index.sq.qtype, faiss.METRIC_L2)
                # Use random probes to avoid crashing MSE-trained quantizer training loops
                probe.train(np.random.rand(2, self.dim).astype(np.float32))
                trained = faiss.vector_to_array(probe.sq.trained)
            return int(trained.nbytes)
        return 0

    def _train_quantizer(self, centers: np.ndarray) -> None:
        """Fits the one codebook this filter will ever use, on the ball
        centers themselves, and fixes PQ's nbits. Called exactly once, by
        construct().

        Trained on `centers`, not on the construct set, because the centers
        are the *only* vectors this index ever encodes: queries go in raw
        (see query()) and the construct vectors are never stored. Fitting to
        a wider sample is therefore a pure loss -- an SQ's per-dimension
        min/max widens and its buckets coarsen, and PQ's k-means centroids
        drift off the centers toward the bulk distribution. Measured at
        dim=32 over 32 centers, training on the full construct set instead
        cost ~1.22x the reconstruction error for SQ4/SQ8, nothing for the
        TurboQuant variants, and turned PQ from exact (dsub=1, one codeword
        per center) into ~1.8 mean error. All of that lands on every radius
        as inflation, and so straight on the false-positive rate.

        nbits is chosen here, from the construct-time center count, and
        frozen: it sets pq.code_size, hence bytes_per_ball() and the
        codebook size, so letting it move later would make the filter's
        memory footprint jump mid-stream.
        """
        if self.quantize == Quantize.NONE:
            self.all_centers_index = self._make_faiss_index(Quantize.NONE)
            return

        if self.quantize == Quantize.PQ:
            # Must be checked before the index is constructed: IndexPQ.search
            # segfaults if built with nbits=0.
            self._pq_nbits = self._pq_nbits_for(len(centers))
            if self._pq_nbits is None:
                raise InsufficientMemoryError(
                    f"Not enough memory to construct a PQ index: {len(centers)} ball "
                    "center(s) is too few for even a 1-bit codebook (need >= 2 centers, "
                    "which requires a larger max_memory or a smaller ball budget)."
                )
            index = self._make_faiss_index(self.quantize, pq_nbits=self._pq_nbits)
            faiss.downcast_index(index.index).pq.cp.min_points_per_centroid = 1
        else:
            index = self._make_faiss_index(self.quantize)

        index.train(centers)
        self.all_centers_index = index

    def _encoding_error(self, centers: np.ndarray) -> np.ndarray:
        """Per-center ||c - decode(encode(c))||, measured against the trained
        codebook -- never a probabilistic bound. sa_encode/sa_decode are used
        as the uniform binding fallback across index types."""
        if self.quantize == Quantize.NONE:
            return np.zeros(len(centers), dtype=np.float32)
        base = faiss.downcast_index(self.all_centers_index.index)
        decoded = base.sa_decode(base.sa_encode(centers))
        return np.linalg.norm(centers - decoded, axis=1).astype(np.float32)

    def _store_spheres(self, sphere_indices: np.ndarray, centers: np.ndarray, radii: np.ndarray) -> None:
        """Writes (center, radius) pairs into the already-trained index,
        inflating each radius by that center's own encoding error.

        The inflation is applied exactly once, here, at the moment a center
        is written -- which is the whole reason nothing rebuilds. A radius
        only needs inflating when it was measured against a center that is
        about to be replaced by its lossy encoding (construct()'s radii, and
        any center an insert strategy relocates). Once stored, `all_centers`
        hands back the *decoded* center, so every radius an insert strategy
        derives from it is already correct with respect to what query()
        scores against, and must not be inflated again. Re-inflating on each
        touch is what made the old rebuild path grow radii without bound.
        """
        sphere_indices = np.ascontiguousarray(sphere_indices, dtype=np.int64)
        if len(sphere_indices) == 0:
            return
        centers = np.ascontiguousarray(centers, dtype=np.float32)
        radii = np.asarray(radii, dtype=np.float32)

        error = self._encoding_error(centers)
        # add_with_ids appends rather than overwrites, so an id being
        # rewritten (a relocated/merged ball) has to be dropped first.
        if self.all_centers_index.ntotal > 0:
            self.all_centers_index.remove_ids(faiss.IDSelectorBatch(sphere_indices))
        self.all_centers_index.add_with_ids(centers, sphere_indices)
        self._all_radiuses[sphere_indices] = radii + error
        self._radius_inflation[sphere_indices] = error

    def _drop_spheres(self, sphere_indices: np.ndarray) -> None:
        """Evicts ids an insert strategy retired (merged away, absorbed) from
        the index, so ntotal keeps tracking _num_spheres."""
        sphere_indices = np.ascontiguousarray(sphere_indices, dtype=np.int64)
        if len(sphere_indices) == 0 or self.all_centers_index.ntotal == 0:
            return
        self.all_centers_index.remove_ids(faiss.IDSelectorBatch(sphere_indices))
        self._radius_inflation[sphere_indices] = 0.0

    def construct(self, vectors: np.ndarray) -> Tuple[float, float]:
        """Builds directly from construct_method's output, independent of
        insert_method: routing through insert_method.add_sphere() here was
        redundant (its result was immediately overwritten below) and, for
        strategies with per-ball bookkeeping (e.g. MinHeapInsert's
        merge-candidate heap), unintentionally seeded that state from
        construction -- a cost only the streaming policy needs, paid by every
        one-shot construct-then-query use of this filter.

        This is also the only place the quantizer is ever trained, and the
        only place PQ's nbits is chosen (see _train_quantizer)."""
        centers, radii = self.construct_method.construct_filter(self, vectors)
        centers = np.ascontiguousarray(centers, dtype=np.float32)
        radii = np.asarray(radii, dtype=np.float32)
        n = len(centers)

        self._ensure_capacity(max(n, 1))
        self._num_spheres = 0
        self._all_centers = None
        self._radius_inflation[:] = 0.0

        if n == 0:
            self.all_centers_index = self._make_faiss_index(Quantize.NONE)
            self._constructed = True
            return self._log_volume_bounds()

        self._train_quantizer(centers)

        sphere_indices = np.arange(n, dtype=np.int64)
        self._all_spheres[:n] = sphere_indices.astype(np.int32)
        self._num_spheres = n
        # construct_method measures radii against the true fp32 centers, so
        # this is where those radii pick up their encoding error.
        self._store_spheres(sphere_indices, centers, radii)
        self._constructed = True

        return self._log_volume_bounds()

    def query(self, vectors: np.ndarray, distance_threshold: float) -> list[bool]:
        if not self._constructed:
            raise ValueError("BallCoverFilter has not been constructed.")

        if self._num_spheres == 0 or len(vectors) == 0:
            return [False] * len(vectors)

        # Q goes in raw and is never encoded: both IndexPQ (search_type
        # ST_PQ, pinned in _make_faiss_index) and IndexScalarQuantizer score
        # with asymmetric distance computation, comparing the float32 query
        # against the *decoded* stored codes. Verified to match an exact
        # float64 ||q - decode(c)|| to 2e-6 at dim=32 and 2e-5 at dim=384 --
        # float32 accumulation noise, well inside _COVERAGE_EPSILON.
        #
        # That is what makes the single-sided inflation in _store_spheres
        # both correct and tight:
        #     d(q, c) <= d(q, decode(c)) + ||c - decode(c)||
        # so d(q, c) <= tau implies d(q, decode(c)) <= tau + error_c, and
        # only the center side of the triangle inequality is lossy. Adding a
        # query-side term on top would be over-inflation, pure false
        # positives for no gain. (For reference, if the query *were* encoded,
        # SQ4 at dim=384 would need another ~2.7 on every radius.)
        Q = np.ascontiguousarray(vectors, dtype=np.float32)
        # The raw ID-mapped index, not the unwrapped inner one -- the latter
        # returns labels keyed by insertion position, which stops matching
        # sphere ID after a merge reassigns some, silently indexing
        # self._all_radiuses[labels] by the wrong sphere.
        d2, labels = self.all_centers_index.search(Q, self._num_spheres)
        surface = np.sqrt(np.maximum(d2, 0.0)) - self._all_radiuses[labels]
        return (surface.min(axis=1) <= distance_threshold + self._COVERAGE_EPSILON).tolist()

    def insert(self, vectors: np.ndarray) -> None:
        """Streaming insertion into an already-constructed filter. Nothing is
        retrained and nothing is rebuilt: whatever centers the strategy
        leaves behind are encoded once, with construct()'s codebook, and
        written straight into the index.

        insert_method strategies mutate _all_centers/_all_radiuses/
        _all_spheres in place, so the sync below is by sphere id: ids whose
        center the strategy actually changed (relocated, merged, or newly
        allocated) get re-encoded and re-inflated; ids it only grew the
        radius of are left alone, because a radius measured against an
        already-decoded center needs no inflation; ids it retired are dropped
        from the index."""
        if not self._constructed:
            raise ValueError(
                "BallCoverFilter.insert() requires a constructed filter -- call construct() first. "
                "insert() reuses the codebook and PQ nbits that construct() fixed, and has no "
                "centers to fit them from on its own."
            )

        if len(vectors) == 0:
            return

        prev_ids = self.all_spheres.copy()
        # Materializes the _all_centers cache the strategies write through.
        prev_centers = self.all_centers.copy()

        self.insert_method.insert(self, vectors)

        new_ids = self.all_spheres.copy()
        new_centers = self._all_centers[new_ids]

        # Position of each still-active id in the pre-insert snapshot, -1 for
        # ids the strategy newly allocated.
        prev_pos = np.full(len(self._all_radiuses), -1, dtype=np.int64)
        prev_pos[prev_ids] = np.arange(len(prev_ids), dtype=np.int64)
        pos = prev_pos[new_ids]

        unchanged = pos >= 0
        if unchanged.any():
            unchanged[unchanged] = np.all(
                new_centers[unchanged] == prev_centers[pos[unchanged]], axis=1
            )
        changed = ~unchanged

        alive = np.zeros(len(self._all_radiuses), dtype=bool)
        alive[new_ids] = True
        self._drop_spheres(prev_ids[~alive[prev_ids]])

        if changed.any():
            changed_ids = new_ids[changed]
            self._store_spheres(changed_ids, new_centers[changed], self._all_radiuses[changed_ids])

        # Drop the cache so the next read comes back decoded, i.e. equal to
        # what query() actually scores against.
        self._all_centers = None

    def add(self, vectors: np.ndarray) -> None:
        self.insert(vectors)

    def remove(self, vectors: np.ndarray) -> List[bool]:
        return self.insert_method.remove(self, vectors)

    def _log_volume_bounds(self) -> Tuple[float, float]:
        if self._num_spheres == 0:
            return -np.inf, -np.inf
        current_centers = self.all_centers
        self._all_centers = None
        return estimate_hypersphere_volume(current_centers, self.all_radiuses, return_log=True)

    def get_bytesize(self) -> int:
        if not self._constructed:
            return 0
        filter_bytes = self._num_spheres * self.bytes_per_ball() + self._codebook_bytes()
        if self.insert_method:
            filter_bytes += self.insert_method.get_bytesize(self)
        return filter_bytes

    def get_construction_bytesize(self) -> int:
        if not self.construct_method:
            return 0
        return self.construct_method.get_bytesize()