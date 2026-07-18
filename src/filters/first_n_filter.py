import faiss
import numpy as np
from typing import List, Optional
from filters.filter import Filter


class FirstNFilter(Filter):
    """
    Baseline semantic filter that simply stores the first N vectors that fit its memory budget.

    Each stored vector v implicitly represents a hypersphere of radius `distance_threshold`
    centred at v, exactly like the trivial case of KDBallFilter/KMEBFilter — but instead of
    compressing the vectors into fewer spheres, vectors beyond the budget are silently dropped.

    A query matches if its L2 distance to the nearest stored vector is within
    `distance_threshold` (computed with a FAISS flat index). Dropped vectors are never
    matched, so unlike the sphere filters this baseline can produce false negatives once
    the budget is exceeded; it serves as the lower bound the compressing filters must beat.
    """

    _BYTES_PER_FLOAT32 = 4

    def __init__(
        self,
        dim: int,
        distance_threshold: float,
        max_memory: Optional[int] = None,
        max_spheres: Optional[int] = None,
    ):
        if max_memory is None and max_spheres is None:
            raise ValueError("Either max_memory or max_spheres must be specified.")

        super().__init__(dim, lsh=None, max_memory=max_memory)

        self.distance_threshold = distance_threshold
        # a "sphere" is one stored vector of dim float32s
        self.max_spheres = (
            max_spheres
            if max_spheres is not None
            else max(1, max_memory // (dim * self._BYTES_PER_FLOAT32))
        )
        
        self._index: Optional[faiss.IndexFlatL2] = None
        self._constructed = False

    def get_exact_matching_fpr(self) -> float:
        return 0.0

    def construct(self, vectors: np.ndarray) -> None:
        """Store the first `max_spheres` vectors in a FAISS flat index; drop the rest."""
        self._index = faiss.IndexFlatL2(self.dim)
        self._constructed = True
        self.add(vectors)

    def query(self, vectors: np.ndarray) -> List[bool]:
        if not self._constructed:
            raise ValueError("FirstNFilter has not been constructed.")
        if self._index.ntotal == 0 or len(vectors) == 0:
            return [False] * len(vectors)

        q = np.ascontiguousarray(vectors, dtype=np.float32)
        # IndexFlatL2 returns squared L2 distances, so compare against threshold^2
        sq_dists, _ = self._index.search(q, 1)
        return (sq_dists[:, 0] <= self.distance_threshold ** 2).tolist()

    def add(self, vectors: np.ndarray) -> None:
        if not self._constructed:
            self.construct(vectors)
            return

        remaining = self.max_spheres - self._index.ntotal
        if remaining <= 0 or len(vectors) == 0:
            return
        new_vectors = np.ascontiguousarray(vectors[:remaining], dtype=np.float32)
        self._index.add(new_vectors)

    def insert(self, vectors: np.ndarray) -> None:
        self.add(vectors)

    def get_bytesize(self) -> int:
        if not self._constructed:
            return 0
        return self._index.ntotal * self.dim * self._BYTES_PER_FLOAT32
