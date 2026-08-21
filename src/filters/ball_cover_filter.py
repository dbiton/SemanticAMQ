from enum import Enum
from typing import Optional

import faiss
import numpy as np


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
        """Largest nbits (<= 8) whose 2**nbits codebook can be trained on n
        points, or None if there aren't even enough points for a 1-bit
        codebook (n < 2) -- IndexPQ.search segfaults at nbits=0, so that case
        must be treated as "PQ unavailable", not built with nbits=0."""
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

    def calculate_max_balls(self, max_memory: int) -> int:
        return max(1, max_memory // self.bytes_per_ball())

    def get_bytesize(self) -> int:
        return self.count_balls * self.bytes_per_ball()

    def construct_enclosing_balls(
        self, vectors: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Gonzalez farthest-first, then exact radii. O(n*k*d)."""
        X = np.ascontiguousarray(vectors, dtype=np.float32)
        n = len(X)
        k = min(self.max_count_balls, n)
        if n <= k:
            return X.copy(), np.zeros(n, dtype=np.float32)

        chosen = [0]
        min_d2 = ((X - X[0]) ** 2).sum(1)
        for _ in range(k - 1):
            chosen.append(int(min_d2.argmax()))
            min_d2 = np.minimum(min_d2, ((X - X[chosen[-1]]) ** 2).sum(1))
        centers = np.ascontiguousarray(X[chosen])

        index = faiss.IndexFlatL2(self.dim)
        index.add(centers)
        d2, labels = index.search(X, 1)
        radii = np.zeros(k, dtype=np.float32)
        np.maximum.at(radii, labels.ravel(), np.sqrt(np.maximum(d2.ravel(), 0)))
        return centers, radii

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

    def distance_to_balls_surfaces(self, query_vectors: np.ndarray) -> np.ndarray:
        """(nq, k) signed distance to each ball surface. Negative = inside."""
        Q = np.ascontiguousarray(query_vectors, dtype=np.float32)
        d2, labels = self.index.search(Q, self.count_balls)
        return np.sqrt(np.maximum(d2, 0.0)) - self.radii[labels]

    def query(self, query_vectors: np.ndarray, distance_threshold: float) -> list[bool]:
        surface = self.distance_to_balls_surfaces(query_vectors)
        return (surface.min(axis=1) < distance_threshold).tolist()

    def insert(self, vectors: np.ndarray) -> None:
        """Alias for add(), kept for compatibility with the evaluation harness."""
        self.add(vectors)
    
    def add(self, vectors: np.ndarray) -> None:
        self.add_nearest(vectors)
    
    def add_nearest(self, vectors: np.ndarray) -> None:
        """Absorb each vector into one existing ball, inflating it if needed.
        Bootstraps via construct() when there's nothing to absorb into yet."""
        if self.count_balls == 0:
            self.construct(vectors)
            return
        X = np.ascontiguousarray(vectors, dtype=np.float32)
        d2, labels = self.index.search(X, self.count_balls)
        dist = np.sqrt(np.maximum(d2, 0.0))
        surface = dist - self.radii[labels]

        cost = np.where(surface <= 0, -np.inf, dist)
        col = cost.argmin(axis=1)
        row = np.arange(len(X))
        ball = labels[row, col]
        np.maximum.at(self.radii, ball, dist[row, col]) 
