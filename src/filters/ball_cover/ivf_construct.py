import numpy as np
import faiss
from faiss.contrib.inspect_tools import get_invlist
from typing import Tuple
from filters.filter import Filter
from filters.ball_cover.ball_interfaces import ConstructInterface


class IVFConstruct(ConstructInterface):
    """Inverted File (IVF) k-means partitioning strategy for enclosing ball construction."""

    KMEANS_ITERS = 8
    HNSW_QUANTIZER_MIN_K = 2048
    HNSW_M = 32
    HNSW_EF_CONSTRUCTION = 80
    HNSW_EF_SEARCH = 64
    KMEANS_SEED = 4263
    CHUNK = 1 << 16

    def construct_filter(self, filter: Filter, vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = np.ascontiguousarray(vectors, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"expected a 2-D array, got shape {X.shape}")
        n, dim = X.shape

        k = min(filter.max_spheres, n)
        if k <= 0:
            return np.empty((0, dim), dtype=np.float32), np.empty(0, dtype=np.float32)
        if n <= k:
            return X.copy(), np.zeros(n, dtype=np.float32)
        if k == 1:
            center = X.mean(0, dtype=np.float32).reshape(1, dim)
            diff = X - center
            radius = np.sqrt(np.einsum("ij,ij->i", diff, diff).max())
            return center, np.array([radius], dtype=np.float32)

        if k >= self.HNSW_QUANTIZER_MIN_K:
            quantizer = faiss.IndexHNSWFlat(dim, self.HNSW_M)
            quantizer.hnsw.efConstruction = self.HNSW_EF_CONSTRUCTION
            quantizer.hnsw.efSearch = self.HNSW_EF_SEARCH
        else:
            quantizer = faiss.IndexFlatL2(dim)

        index = faiss.IndexIVFFlat(quantizer, dim, k, faiss.METRIC_L2)
        index.cp.niter = self.KMEANS_ITERS
        index.cp.seed = self.KMEANS_SEED
        index.cp.min_points_per_centroid = 1  # silences the under-training warning
        index.train(X)
        index.add(X)

        centroids = quantizer.reconstruct_n(0, k)

        sizes = np.zeros(k, dtype=np.int64)
        id_lists = []
        for i in range(k):
            ids, _codes = get_invlist(index.invlists, i)
            sizes[i] = ids.size
            if ids.size:
                id_lists.append(ids.astype(np.int64, copy=False))

        rows = np.concatenate(id_lists)  # grouped by list, so segments are contiguous
        cell_of = np.repeat(np.arange(k, dtype=np.int64), sizes)

        # Exact distances (no norm expansion), each point against its own centroid.
        d2 = np.empty(rows.size, dtype=np.float32)
        for start in range(0, rows.size, self.CHUNK):
            stop = min(start + self.CHUNK, rows.size)
            diff = X[rows[start:stop]] - centroids[cell_of[start:stop]]
            np.einsum("ij,ij->i", diff, diff, out=d2[start:stop])

        starts = np.zeros(k, dtype=np.int64)
        np.cumsum(sizes[:-1], out=starts[1:])
        nonempty = sizes > 0

        radii = np.zeros(k, dtype=np.float32)
        radii[nonempty] = np.maximum.reduceat(d2, starts[nonempty])
        np.sqrt(radii, out=radii)

        centers = np.ascontiguousarray(centroids[nonempty])
        return centers, radii[nonempty]


    def get_bytesize(self) -> int:
        return 0