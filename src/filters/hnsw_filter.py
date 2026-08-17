"""
HNSWFilter -- semantic AMQ baseline built directly on FAISS's HNSW
(Hierarchical Navigable Small World) approximate nearest-neighbor index.

Unlike every other filter in this module, HNSWFilter makes no attempt at a
compact/hashed representation: it stores full float32 vectors plus the HNSW
graph and answers "is anything within tau?" via a real (approximate)
nearest-neighbor search on real vectors. That gives it a property no other
filter here has: it cannot produce a structural false positive (a returned
neighbor's distance is measured exactly against the real stored vector, not
inferred from a hash collision), so get_exact_matching_fpr() is 0.0. What it
trades away is BallCoverFilter's zero-false-negative guarantee -- HNSW
search is approximate, so it can miss the true nearest neighbor and silently
under-report duplicates; how often that happens is governed by
ef_search/ef_construction/M, not bounded by construction the way a covering
structure is.

Memory is not budget-tunable the way max_memory works for the other filters
here: IndexHNSWFlat stores full uncompressed vectors (n * dim * 4 bytes)
plus the graph's neighbor lists, so at typical embedding dimensions the raw
vector storage dominates M's contribution and there is no way to shrink it
to an arbitrary target without dropping to a quantized HNSW variant (out of
scope here -- this filter answers "what does a real ANN index cost as-is",
not "how small can an ANN index get"). max_memory/max_spheres are accepted
only for _instantiate_filter parity with the other direct-tau filters.
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
        super().__init__(dim, lsh=None, max_memory=max_memory)
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

    def get_exact_matching_fpr(self) -> float:
        return 0.0

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
