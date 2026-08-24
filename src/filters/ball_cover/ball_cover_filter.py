from enum import Enum
from typing import Optional

import faiss
import numpy as np

from filters.ball_cover.ball_cover_construction import gonzales_construct_enclosing_balls
from filters.ball_cover.ball_cover_insertion import INSERTION_STRATEGIES


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

PQ_MAX_NBITS = 8

_QTYPES = {
    Quantize.BF16: faiss.ScalarQuantizer.QT_bf16,
    Quantize.FP16: faiss.ScalarQuantizer.QT_fp16,
    Quantize.SQ8: faiss.ScalarQuantizer.QT_8bit,
    Quantize.SQ6: faiss.ScalarQuantizer.QT_6bit,
    Quantize.SQ4: faiss.ScalarQuantizer.QT_4bit,
    Quantize.SQ4T: faiss.ScalarQuantizer.QT_4bit_tqmse,
    Quantize.SQ8T: faiss.ScalarQuantizer.QT_8bit_tqmse,
}



class BallCoverFilter:
    def __init__(
        self,
        dim: int,
        max_memory: int | None  = None,
        quantize: Quantize = Quantize.NONE,
        construction_level: int = 3,
        insertion_level: str | None = None,
    ) -> None:
        self.dim = dim
        self.quantize = quantize
        self.construction_level = construction_level
        self.insertion_level = insertion_level
        self.index = self._make_index(quantize)
        self.max_count_balls = self.calculate_max_balls(max_memory)
        self.count_balls = 0
        self.radii = np.zeros(self.max_count_balls, dtype=np.float32)

    @property
    def all_radiuses(self) -> np.ndarray:
        return self.radii[: self.count_balls]

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
            return self.index.pq.code_size + 4
        if isinstance(self.index, faiss.IndexScalarQuantizer):
            return self.index.sq.code_size + 4
        return (self.index.d + 1) * 4

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

    def calculate_max_balls(self, max_memory: int) -> int:
        usable = max(0, max_memory - self._codebook_bytes())
        return max(1, usable // self.bytes_per_ball())

    def get_bytesize(self) -> int:
        return self.count_balls * self.bytes_per_ball() + self._codebook_bytes()

    def construct_enclosing_balls(
        self, vectors: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return gonzales_construct_enclosing_balls(self.max_count_balls, vectors)

    def _build_quantizer(
        self, quantize: Quantize, centers: np.ndarray
    ) -> Optional[tuple[faiss.Index, np.ndarray]]:
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
            index.pq.cp.min_points_per_centroid = 1  # our "training set" is just the ball centers
            index.train(centers)
            decoded = index.pq.decode(index.pq.compute_codes(centers))
        else:
            index = self._make_index(quantize)
            index.train(centers)
            decoded = index.sq.decode(index.sq.compute_codes(centers))
        error = np.linalg.norm(centers - decoded, axis=1).astype(np.float32)
        return index, error

    def construct(self, vectors: np.ndarray) -> None:
        centers, radii = self.construct_enclosing_balls(vectors)
        centers = np.ascontiguousarray(centers, dtype=np.float32)
        self._rebuild_from_centers(centers, radii)

    def _rebuild_from_centers(self, centers: np.ndarray, radii: np.ndarray) -> None:
        """Re-derive self.index/self.radii/self.count_balls from an explicit
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

        self.index = index

        self.radii[:] = 0.0
        self.radii[:n] = radii
        self.count_balls = n

    def query(self, query_vectors: np.ndarray, distance_threshold: float) -> list[bool]:
        Q = np.ascontiguousarray(query_vectors, dtype=np.float32)
        d2, labels = self.index.search(Q, self.count_balls)
        surface = np.sqrt(np.maximum(d2, 0.0)) - self.radii[labels]
        return (surface.min(axis=1) < distance_threshold).tolist()

    def insert(self, vectors: np.ndarray) -> None:
        if self.count_balls == 0:
            self.construct(vectors)
            return

        level = self.insertion_level or "nearest"
        if level not in INSERTION_STRATEGIES:
            raise ValueError(
                f"Unknown insertion_level {level!r}; expected one of {tuple(INSERTION_STRATEGIES)}"
            )

        X = np.ascontiguousarray(vectors, dtype=np.float32)
        centers = self.index.reconstruct_n(0, self.count_balls)
        radii = self.radii[: self.count_balls]

        new_centers, new_radii = INSERTION_STRATEGIES[level](centers, radii, X, self.max_count_balls)

        if new_centers is centers:
            # "nearest"/"least_inflation" never relocate or add a ball (see
            # ball_cover_insertion.py) -- only radii grew, so there's nothing
            # to re-quantize or re-index.
            self.radii[: self.count_balls] = new_radii
        else:
            self._rebuild_from_centers(new_centers, new_radii)
