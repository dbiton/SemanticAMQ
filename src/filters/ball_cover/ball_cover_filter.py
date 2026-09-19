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
        # 0 for Quantize.NONE: added onto each radius by _rebuild_from_centers
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


    def _build_quantizer(self, quantize: Quantize, centers: np.ndarray) -> Tuple[faiss.IndexIDMap2, np.ndarray]:
        """Returns (untrained index over `centers`, per-center reconstruction
        error). Raises InsufficientMemoryError for PQ with too few centers
        for even a 1-bit codebook -- must check before constructing that
        index, since IndexPQ.search segfaults if built with nbits=0."""
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
        """Re-derives ball state from an explicit (centers, radii) pair,
        applying the configured quantizer -- used by construct() and by
        insert() whenever a strategy relocates/adds balls (e.g. merge)
        rather than just growing an existing radius in place."""
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


    def construct(self, vectors: np.ndarray) -> Tuple[float, float]:
        """Builds directly from construct_method's output, independent of
        insert_method: routing through insert_method.add_sphere() here was
        redundant (its result was immediately overwritten below) and, for
        strategies with per-ball bookkeeping (e.g. MinHeapInsert's
        merge-candidate heap), unintentionally seeded that state from
        construction -- a cost only the streaming policy needs, paid by every
        one-shot construct-then-query use of this filter."""
        centers, radii = self.construct_method.construct_filter(self, vectors)
        sphere_indices = np.arange(len(centers), dtype=np.int64)
        self._rebuild_from_centers(sphere_indices, centers, radii)
        self._constructed = True

        return self._log_volume_bounds()


    def query(self, vectors: np.ndarray, distance_threshold: float) -> list[bool]:
        if not self._constructed:
            raise ValueError("BallCoverFilter has not been constructed.")

        if self._num_spheres == 0 or len(vectors) == 0:
            return [False] * len(vectors)

        Q = np.ascontiguousarray(vectors, dtype=np.float32)
        # The raw ID-mapped index, not the unwrapped inner one -- the latter
        # returns labels keyed by insertion position, which stops matching
        # sphere ID after a merge reassigns some, silently indexing
        # self._all_radiuses[labels] by the wrong sphere.
        d2, labels = self.all_centers_index.search(Q, self._num_spheres)
        surface = np.sqrt(np.maximum(d2, 0.0)) - self._all_radiuses[labels]
        return (surface.min(axis=1) <= distance_threshold + self._COVERAGE_EPSILON).tolist()


    def insert(self, vectors: np.ndarray) -> None:
        if not self._constructed:
            self.construct(vectors)
            return

        if len(vectors) == 0:
            return

        # Must check BOTH the sphere-id set and the center values -- neither
        # alone is sufficient, since strategies key bookkeeping differently:
        # MinHeapInsert tracks by id but can leave a merged ball's value
        # bit-identical to some other (now-discarded) id's old value, so a
        # value-only check misses the id change; ExhaustiveMergeInsert tracks
        # purely positionally, so its id set never changes even when values
        # do, and an id-only check misses that. Either false negative leaves
        # the FAISS index silently stale until a later, unrelated insertion
        # forces a refetch.
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