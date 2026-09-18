import numpy as np
import faiss
from enum import Enum
from typing import List, Tuple, Optional
from hyperspheres_volume import estimate_hypersphere_volume
from filters.filter import Filter
from filters.ball_cover.ball_interfaces import ConstructInterface, InsertInterface
from filters.ball_cover.gonzales_construct import GonzalesConstruct
from filters.ball_cover.ivf_construct import IVFConstruct
from filters.ball_cover.rptree_construct import RPTreeConstruct
from filters.ball_cover.min_heap_insert import MinHeapInsert
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

    # Coverage-comparison safety margin, not a knob to tune per call: every
    # insertion strategy grows a ball's radius to exactly match a distance it
    # measured itself (often via its own throwaway, single-vector FAISS
    # search), with zero slack. FAISS's IndexFlatL2.search() isn't bit-
    # reproducible across different query-batch shapes -- re-querying that
    # same point later as part of a larger batch (e.g. a no-false-negative
    # sweep over every point seen so far) computes a different squared
    # distance for the identical point/center pair, which flips a boundary
    # point from covered to uncovered whenever the caller's own
    # distance_threshold has no margin of its own (distance_threshold=0.0,
    # as the streaming evaluation's tau calibration can produce on datasets
    # with real near-duplicate content). The error is a roughly fixed
    # absolute amount in *squared*-distance space (consistent with float32
    # rounding in a batched sum-of-384-terms), so taking sqrt() of a
    # near-zero squared distance (a ball whose radius is itself ~0, i.e. an
    # exact or near-exact duplicate) amplifies it hugely: reproduced directly
    # against real MMLU embeddings at dim=384, this measured up to ~9.1e-4 in
    # `surface` for an exact self-match (single-query search: exactly 0.0;
    # the same point re-searched inside a several-hundred-row batch, as
    # query()/the no-false-negative check actually do: ~9e-4), across k up to
    # 1024 and batch sizes up to 4000. 1e-3 sits just above that observed
    # worst case -- matching this project's own established tolerance for
    # exactly this class of check (see test_ball_cover_construction.py's
    # _covering_ok helper, tol=1e-3) -- and stays far below the smallest
    # distance_threshold used anywhere in this project's actual sweeps
    # (percentile-based tau values are on the order of 1e-2 or larger), so it
    # cannot mask a real false negative.
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
        # Must construct fresh instances here, not as the parameter defaults
        # above: a mutable object used as a default argument value is
        # evaluated exactly once, at function-definition time, and then
        # shared by every call that doesn't override it -- so every
        # BallCoverFilter built without an explicit construct_method/
        # insert_method used to share the *same* GonzalesConstruct/
        # MinHeapInsert instance. GonzalesConstruct is stateless (harmless
        # to share), but MinHeapInsert carries real per-filter state
        # (_sphere_versions, _minimal_pairs) that must not leak between
        # instances: every filter built in the same process (e.g. one
        # ProcessPoolExecutor worker handling many Runner jobs) was silently
        # accumulating merge-candidate bookkeeping from every filter built
        # before it, inflating get_bytesize() without bound and risking
        # stale-version merge decisions once two live filters shared it.
        self.construct_method = construct_method if construct_method is not None else GonzalesConstruct()
        self.insert_method = insert_method if insert_method is not None else MinHeapInsert()
        self.insert_memory_percent = insert_memory_percent
        self.all_centers_index = self._make_faiss_index(self.quantize)
        self.max_spheres = (max_spheres if max_spheres is not None 
                    else self._calculate_max_spheres(self.max_memory))
        self.max_construct_spheres = self._calculate_max_construct_spheres(self.insert_memory_percent)
        self._constructed = False

        # main filter sphere buffers
        self._num_spheres = 0
        self._all_spheres = np.arange(self.max_spheres, dtype=np.int32)
        self._all_centers = None # temp array used only during insertions
        self._all_radiuses = np.empty((self.max_spheres,), dtype=np.float32)
        # Per-ball radius inflation: 0 for Quantize.NONE, otherwise the
        # ||center - decode(encode(center))|| reconstruction error added onto
        # that ball's radius by _rebuild_from_centers to preserve the
        # no-false-negative guarantee under a lossy center encoding.
        self._radius_inflation = np.zeros((self.max_spheres,), dtype=np.float32)

        # initialize insertion method buffers
        self.insert_method.ensure_capacity(self.max_spheres)

    # ------------------------------------------------------------------ properties

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

    # ------------------------------------------------------------------ buffer operations

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
        self.all_centers_index = self._make_faiss_index(self.quantize)
        self.insert_method.reset_buffers(capacity)


    def _calculate_max_spheres(self, max_memory: int) -> int:
        usable = max(0, max_memory - self._codebook_bytes())
        return max(1, usable // self.bytes_per_ball())


    def _calculate_max_construct_spheres(self, insertion_memory_percent: float) -> int:
        if insertion_memory_percent <= 0.0:
            return self.max_spheres

        if self.max_memory is not None:
            construct_memory = max(0, int(self.max_memory * (1.0 - insertion_memory_percent / 100.0)))
            return self._calculate_max_spheres(construct_memory)

        return max(1, int(self.max_spheres * (1.0 - insertion_memory_percent / 100.0)))

    # ------------------------------------------------------------------ quantization & helpers

    @staticmethod
    def _choose_pq_m(dim: int) -> int:
        for m in range(min(dim, PQ_MAX_NBITS), 0, -1):
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
                probe.train(np.zeros((2, self.dim), dtype=np.float32))
                trained = faiss.vector_to_array(probe.sq.trained)
            return int(trained.nbytes)
        return 0

    # ------------------------------------------------------------------ center quantization & rebuilding

    def _build_quantizer(self, quantize: Quantize, centers: np.ndarray) -> Tuple[faiss.IndexIDMap2, np.ndarray]:
        """Train an index of the given quantizer type over centers (not yet
        populated -- caller adds centers to the returned index) and return the
        per-center reconstruction error ||center - decode(encode(center))||.
        Raises InsufficientMemoryError if this quantizer type can't be built
        for this many centers (only possible for PQ: too few centers for even
        a 1-bit codebook -- IndexPQ.search segfaults if built with nbits=0, so
        this check must happen before ever constructing that index, not after)."""
        if quantize == Quantize.PQ:
            nbits = self._pq_nbits_for(len(centers))
            if nbits is None:
                raise InsufficientMemoryError(
                    f"Not enough memory to construct a PQ index: {len(centers)} ball "
                    "center(s) is too few for even a 1-bit codebook (need >= 2 centers, "
                    "which requires a larger max_memory or a smaller ball budget)."
                )
            index = self._make_faiss_index(quantize, pq_nbits=nbits)
            base_idx = faiss.downcast_index(index.index)
            base_idx.pq.cp.min_points_per_centroid = 1
            index.train(centers)
            decoded = base_idx.pq.decode(base_idx.pq.compute_codes(centers))
        else:
            index = self._make_faiss_index(quantize)
            base_idx = faiss.downcast_index(index.index)
            index.train(centers)
            decoded = base_idx.sq.decode(base_idx.sq.compute_codes(centers))
        error = np.linalg.norm(centers - decoded, axis=1).astype(np.float32)
        return index, error


    def _rebuild_from_centers(self, sphere_indices: np.ndarray, centers: np.ndarray, radii: np.ndarray) -> None:
        """Re-derive our ball data structures from an explicit
        (centers, radii) pair, applying the configured quantizer exactly like
        construct() does. Used both by construct() itself and by insert() when
        an insertion strategy relocates/adds balls (e.g. "merge") rather
        than just growing an existing ball's radius in place."""
        sphere_indices = np.ascontiguousarray(sphere_indices, dtype=np.int64)
        centers = np.ascontiguousarray(centers, dtype=np.float32)
        radii = np.asarray(radii, dtype=np.float32)
        n = len(centers)

        inflation = np.zeros(n, dtype=np.float32)
        if n == 0:
            index = self._make_faiss_index(Quantize.NONE)
        elif self.quantize == Quantize.NONE:
            index = self._make_faiss_index(Quantize.NONE)
            index.add_with_ids(centers, sphere_indices)
        else:
            index, error = self._build_quantizer(self.quantize, centers)
            radii = radii + error
            inflation = error
            index.add_with_ids(centers, sphere_indices)

        self._ensure_capacity(n)
        self._num_spheres = n
        self._all_spheres[:n] = sphere_indices.astype(np.int32)
        self._all_centers = None
        self.all_centers_index = index
        self._all_radiuses[sphere_indices] = radii
        self._radius_inflation[sphere_indices] = inflation

    # ------------------------------------------------------------------ construct & insert

    def construct(self, vectors: np.ndarray) -> Tuple[float, float]:
        centers, radii = self.construct_method.construct_filter(self, vectors)
        _ = self.all_centers

        for center, radius in zip(centers, radii):
            self.insert_method.add_sphere(self, center, radius)

        self._rebuild_from_centers(self.all_spheres, self.all_centers, self.all_radiuses)
        self._constructed = True

        return self._log_volume_bounds()

    # ------------------------------------------------------------------ query

    def query(self, vectors: np.ndarray, distance_threshold: float) -> list[bool]:
        if not self._constructed:
            raise ValueError("BallCoverFilter has not been constructed.")

        if self._num_spheres == 0 or len(vectors) == 0:
            return [False] * len(vectors)

        Q = np.ascontiguousarray(vectors, dtype=np.float32)
        # self._all_centers_index (the raw ID-mapped index), not the
        # all_centers_index *property* -- that property unwraps to the
        # plain inner index for bytes_per_ball()/_codebook_bytes()'s
        # isinstance checks, and searching the unwrapped index returns
        # labels keyed by raw insertion *position*, not sphere ID. Sphere
        # IDs stop matching their insertion position after any merge
        # retires/reassigns some of them, so self._all_radiuses[labels]
        # below would silently index by the wrong sphere's radius,
        # producing false negatives on exactly those merged points.
        d2, labels = self.all_centers_index.search(Q, self._num_spheres)
        surface = np.sqrt(np.maximum(d2, 0.0)) - self._all_radiuses[labels]
        return (surface.min(axis=1) <= distance_threshold + self._COVERAGE_EPSILON).tolist()

    # ------------------------------------------------------------------ streaming / dynamic insertion

    def insert(self, vectors: np.ndarray) -> None:
        if not self._constructed:
            self.construct(vectors)
            return

        if len(vectors) == 0:
            return

        # Whether the FAISS center index needs rebuilding must be decided by
        # BOTH the *set of active sphere ids* and the center *values* --
        # neither alone is sufficient, because different insertion
        # strategies key their bookkeeping differently:
        #
        # - MinHeapInsert tracks spheres *by id* (filter._all_spheres), and
        #   its merge step can set new_center = center_i verbatim when two
        #   centers being merged are exactly identical (distance_ij == 0.0
        #   -- genuinely common with real embeddings, since near-duplicate
        #   text produces near-duplicate vectors). That can make the merged
        #   ball's value at its (possibly different) sphere id bit-for-bit
        #   equal to some other, now-discarded id's old value at a different
        #   row position -- so a value-only check can see "no change" even
        #   though the id set genuinely changed, silently skipping the
        #   rebuild FAISS needs.
        # - ExhaustiveMergeInsert instead tracks spheres *purely
        #   positionally*: it always writes filter._all_centers[:n] and
        #   never touches filter._all_spheres at all, so its active id set
        #   stays constant (e.g. always {0,...,n-1}) even while the values
        #   at those same ids change substantially from merges -- so an
        #   id-set-only check sees "no change" and skips a rebuild that's
        #   very much needed.
        #
        # Either kind of false negative leaves the FAISS index silently out
        # of sync with the filter's own bookkeeping. That drift stays
        # invisible for a while (the numpy _all_centers cache, written
        # directly by add_sphere/insert, still has correct data for
        # querying) until some later, unrelated no-op insertion invalidates
        # that cache and forces a refetch from the now-stale FAISS index --
        # surfacing as either a "key not found" reconstruct error or,
        # worse, silently wrong (stale) query answers. Checking both catches
        # both strategies' failure modes.
        old_spheres = set(self.all_spheres.tolist())
        old_centers = self.all_centers.copy()
        self.insert_method.insert(self, vectors)
        new_spheres = set(self.all_spheres.tolist())
        new_centers = self.all_centers

        spheres_changed = old_spheres != new_spheres
        values_changed = len(old_centers) != len(new_centers) or not np.array_equal(old_centers, new_centers)

        if spheres_changed or values_changed:
            self._rebuild_from_centers(self.all_spheres, self.all_centers, self.all_radiuses)
        else:
            self._all_centers = None


    def add(self, vectors: np.ndarray) -> None:
        self.insert(vectors)


    def remove(self, vectors: np.ndarray) -> List[bool]:
        return self.insert_method.remove(self, vectors)

    # ------------------------------------------------------------------ volume & memory accounting

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