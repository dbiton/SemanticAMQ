"""
HNSWFilter -- semantic AMQ baseline built directly on FAISS's HNSW
approximate nearest-neighbor index.

Unlike every other filter here, it stores full float32 vectors plus the HNSW
graph and answers via a real (approximate) nearest-neighbor search: it
cannot produce a structural false positive (distance is measured exactly
against the real stored vector), but trades away BallCoverFilter's
zero-false-negative guarantee -- HNSW search can miss the true neighbor and
silently under-report duplicates, governed by ef_search/ef_construction/M.

Memory is not budget-tunable: IndexHNSWFlat stores full uncompressed
vectors plus the graph's neighbor lists, with no way to shrink to an
arbitrary target short of a quantized HNSW variant (out of scope here).
max_memory/max_spheres are accepted only for _instantiate_filter parity.
"""

from typing import List, Optional

import faiss
import numpy as np

from filters.filter import Filter


class HNSWFilter(Filter):
    """FAISS IndexHNSWFlat wrapped as a semantic AMQ filter: query is a real
    1-NN search, "duplicate" iff the nearest neighbor found is within tau.
    tau is a pure query-time knob -- nothing about the threshold is baked
    into the graph, unlike DSHFilter/DSBFFilter/LSBFFilter."""

    def __init__(
        self,
        dim: int,
        distance_threshold: float,
        max_memory: Optional[int] = None,
        M: int = 32,
        ef_construction: int = 200,
        ef_search: int = 64,
        max_spheres: Optional[int] = None,
    ):
        super().__init__(dim, max_memory=max_memory)
        self.distance_threshold = float(distance_threshold)
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self._max_spheres = max_spheres  # unused; accepted for _instantiate_filter parity
        self.index: Optional[faiss.IndexHNSWFlat] = None

    def _new_index(self) -> faiss.IndexHNSWFlat:
        index = faiss.IndexHNSWFlat(self.dim, self.M)
        index.hnsw.efConstruction = self.ef_construction
        index.hnsw.efSearch = self.ef_search
        return index

    def construct(self, vectors: np.ndarray) -> None:
        self.index = self._new_index()
        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        if len(vecs) > 0:
            self.index.add(vecs)

    def query(self, vectors: np.ndarray, tau: Optional[float] = None) -> List[bool]:
        if self.index is None:
            raise ValueError("HNSWFilter has not been constructed.")
        if tau is None:
            tau = self.distance_threshold
        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        if len(vecs) == 0 or self.index.ntotal == 0:
            return [False] * len(vecs)
        sq_dist, _ = self.index.search(vecs, 1)
        return (sq_dist[:, 0] <= float(tau) ** 2).tolist()

    def insert(self, vectors: np.ndarray) -> None:
        if self.index is None:
            self.construct(vectors)
            return
        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        if len(vecs) > 0:
            self.index.add(vecs)

    def get_bytesize(self) -> int:
        if self.index is None:
            return 0
        return len(faiss.serialize_index(self.index))
