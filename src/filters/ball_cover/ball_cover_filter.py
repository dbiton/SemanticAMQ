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
    PQ = "PQ"


_QTYPES = {
    Quantize.BF16: faiss.ScalarQuantizer.QT_bf16,
    Quantize.FP16: faiss.ScalarQuantizer.QT_fp16,
    Quantize.SQ8: faiss.ScalarQuantizer.QT_8bit,
    Quantize.SQ6: faiss.ScalarQuantizer.QT_6bit,
    Quantize.SQ4: faiss.ScalarQuantizer.QT_4bit,
    Quantize.SQ4T: faiss.ScalarQuantizer.QT_4bit_tqmse,
    Quantize.SQ8T: faiss.ScalarQuantizer.QT_8bit_tqmse,
}
PQ_MAX_NBITS = 8


class BallCoverFilter(Filter):
    _BYTES_PER_FLOAT32 = 4
    _BYTES_PER_FLOAT64 = 8

    def __init__(
        self,
        dim: int,
        construct_method: Optional[ConstructInterface] = GonzalesConstruct(),
        insert_method: Optional[InsertInterface] = MinHeapInsert(),
        quantize: Quantize = Quantize.NONE,
        max_memory: Optional[int] = None,
        max_spheres: Optional[int] = None,
        insert_memory_percent: Optional[float] = 0.0
    ) -> None:
        super().__init__(dim, lsh=None, max_memory=max_memory)
        self.quantize = quantize
        self.construct_method = construct_method
        self.insert_method = insert_method
        self.insert_memory_percent = insert_memory_percent
        self.index = self._make_index(self.quantize)
        self.max_spheres = (max_spheres if max_spheres is not None 
                    else self._calculate_max_spheres(self.max_memory))
        self.max_construct_spheres = self._calculate_max_construct_spheres(self.insert_memory_percent)
        self._constructed = False

        # main filter sphere buffers
        self._num_spheres = 0
        self._all_spheres = np.arange(self.max_spheres, dtype=np.int32)
        self._all_centers = np.empty((self.max_spheres, dim), dtype=np.float32)
        self._all_radiuses = np.empty((self.max_spheres,), dtype=np.float32)

        # initialize insertion method buffers
        self.insert_method.ensure_capacity(self.max_spheres)

    # ------------------------------------------------------------------ properties

    @property
    def all_spheres(self) -> np.ndarray:
        return self._all_spheres[:self._num_spheres]

    @property
    def all_centers(self) -> np.ndarray:
        return self._all_centers[self.all_spheres]

    @property
    def all_radiuses(self) -> np.ndarray:
        return self._all_radiuses[self.all_spheres]

    # ------------------------------------------------------------------ buffer operations

    def _ensure_capacity(self, required_cap: int) -> None:
        current_cap = len(self._all_centers)
        if required_cap <= current_cap:
            return

        new_cap = max(current_cap * 2, required_cap)
        new_spheres = np.empty((new_cap,), dtype=np.int32)
        new_spheres[:current_cap] = self._all_spheres
        new_spheres[current_cap:] = np.arange(current_cap, new_cap, dtype=np.int32)
        self._all_spheres = new_spheres

        new_centers = np.empty((new_cap, self.dim), dtype=np.float32)
        new_centers[:current_cap] = self._all_centers
        self._all_centers = new_centers

        new_radiuses = np.empty((new_cap,), dtype=np.float32)
        new_radiuses[:current_cap] = self._all_radiuses
        self._all_radiuses = new_radiuses
        self.insert_method.ensure_capacity(new_cap)


    def _get_new_index(self) -> int:
        if self._num_spheres == len(self._all_centers):
            self._ensure_capacity(len(self._all_centers) * 2)
        return self._all_spheres[self._num_spheres]


    def _reset_buffers(self, capacity: int) -> None:
        self._ensure_capacity(capacity)
        self._num_spheres = 0
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


    def _make_index(self, quantize: Quantize, pq_nbits: int = PQ_MAX_NBITS) -> faiss.Index:
        if quantize == Quantize.NONE:
            return faiss.IndexFlatL2(self.dim)
        if quantize == Quantize.PQ:
            pq_m = self._choose_pq_m(self.dim)
            return faiss.IndexPQ(self.dim, pq_m, pq_nbits, faiss.METRIC_L2)
        return faiss.IndexScalarQuantizer(self.dim, _QTYPES[quantize], faiss.METRIC_L2)


    def bytes_per_ball(self) -> int:
        if isinstance(self.index, faiss.IndexPQ):
            return self.index.pq.code_size + self._BYTES_PER_FLOAT32
        if isinstance(self.index, faiss.IndexScalarQuantizer):
            return self.index.sq.code_size + self._BYTES_PER_FLOAT32
        return (self.dim + 1) * self._BYTES_PER_FLOAT32


    def _codebook_bytes(self) -> int:
        if isinstance(self.index, faiss.IndexPQ):
            return int(faiss.vector_to_array(self.index.pq.centroids).nbytes)
        if isinstance(self.index, faiss.IndexScalarQuantizer):
            trained = faiss.vector_to_array(self.index.sq.trained)
            if len(trained) == 0:
                probe = faiss.IndexScalarQuantizer(self.dim, self.index.sq.qtype, faiss.METRIC_L2)
                probe.train(np.zeros((2, self.dim), dtype=np.float32))
                trained = faiss.vector_to_array(probe.sq.trained)
            return int(trained.nbytes)
        return 0

    # ------------------------------------------------------------------ center quantization & rebuilding

    def _build_quantizer(self, quantize: Quantize, centers: np.ndarray) -> Optional[Tuple[faiss.Index, np.ndarray]]:
        """Train an index of the given quantizer type over centers (not yet
        populated -- caller adds centers to the returned index) and return the
        per-center reconstruction error ||center - decode(encode(center))||.
        Returns None if this quantizer type can't be built for this many
        centers (only possible for PQ: too few centers for even a 1-bit
        codebook)."""
        if quantize == Quantize.PQ:
            nbits = self._pq_nbits_for(len(centers))
            if nbits is None:
                return None
            index = self._make_index(quantize, pq_nbits=nbits)
            index.pq.cp.min_points_per_centroid = 1
            index.train(centers)
            decoded = index.pq.decode(index.pq.compute_codes(centers))
        else:
            index = self._make_index(quantize)
            index.train(centers)
            decoded = index.sq.decode(index.sq.compute_codes(centers))
        error = np.linalg.norm(centers - decoded, axis=1).astype(np.float32)
        return index, error


    def _rebuild_from_centers(self, centers: np.ndarray, radii: np.ndarray) -> None:
        """Re-derive ball data structures from an explicit
        (centers, radii) pair, applying the configured quantizer exactly like
        construct() does. Used both by construct() itself and by insert() when
        an insertion strategy relocates/adds balls (e.g. "merge") rather
        than just growing an existing ball's radius in place."""
        centers = np.ascontiguousarray(centers, dtype=np.float32)
        radii = np.asarray(radii, dtype=np.float32)
        n = len(centers)

        if n == 0:
            index = self._make_index(Quantize.NONE)
        elif self.quantize == Quantize.NONE:
            index = self._make_index(Quantize.NONE)
            index.add(centers)
        else:
            built = self._build_quantizer(self.quantize, centers)
            if built is None:
                index = self._make_index(Quantize.NONE)
                index.add(centers)
            else:
                index, error = built
                radii = radii + error
                index.add(centers)

        self._ensure_capacity(n)
        self.index = index
        self._all_centers[:n] = centers
        self._all_radiuses[:n] = radii
        self._num_spheres = n

    # ------------------------------------------------------------------ construct

    def construct(self, vectors: np.ndarray) -> Tuple[float, float]:
        centers, radii = self.construct_method.construct_filter(self, vectors)
        for center, radius in zip(centers, radii):
            self.insert_method.add_sphere(self, center, radius)

        self._rebuild_from_centers(self.all_centers, self.all_radiuses)
        self._constructed = True

        return self._log_volume_bounds()

    # ------------------------------------------------------------------ query

    def query(self, vectors: np.ndarray, distance_threshold: float) -> list[bool]:
        if not self._constructed:
            raise ValueError("BallCoverFilter has not been constructed.")

        if self._num_spheres == 0 or len(vectors) == 0:
            return [False] * len(vectors)

        Q = np.ascontiguousarray(vectors, dtype=np.float32)
        d2, labels = self.index.search(Q, self._num_spheres)
        surface = np.sqrt(np.maximum(d2, 0.0)) - self.all_radiuses[labels]
        return (surface.min(axis=1) < distance_threshold).tolist()

    # ------------------------------------------------------------------ streaming / dynamic insertion

    def insert(self, vectors: np.ndarray) -> None:
        if not self._constructed:
            self.construct(vectors)
            return

        if len(vectors) == 0:
            return

        self.insert_method.insert(self, vectors)
        self._rebuild_from_centers(self.all_centers, self.all_radiuses)


    def add(self, vectors: np.ndarray) -> None:
        self.insert(vectors)


    def remove(self, vectors: np.ndarray) -> List[bool]:
        return self.insert_method.remove(self, vectors)

    # ------------------------------------------------------------------ volume & memory accounting

    def _log_volume_bounds(self) -> Tuple[float, float]:
        if len(self.all_centers) == 0:
            return -np.inf, -np.inf
        return estimate_hypersphere_volume(self.all_centers, self.all_radiuses, return_log=True)


    def get_bytesize(self) -> int:
        if not self._constructed:
            return 0
        filter_bytes = self._num_spheres * self.bytes_per_ball() + self._codebook_bytes()
        if self.insert_method:
            filter_bytes += self.insert_method.get_bytesize()
        return filter_bytes


    def get_construction_bytesize(self) -> int:
        if not self.construct_method:
            return 0
        return self.construct_method.get_bytesize()