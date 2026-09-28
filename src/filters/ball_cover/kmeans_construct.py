import numpy as np
import faiss
from typing import Tuple
from filters.filter import Filter
from filters.ball_cover.ball_interfaces import ConstructInterface
from filters.ball_cover.exact_cover import exact_cover


class KMeansConstruct(ConstructInterface):
    """k-means++ seeding, a few Lloyd iterations (faiss.Kmeans), then exact
    radii. Centers become centroids rather than data points, which pulls
    each ball's center toward the bulk of its members. The catch is that
    k-means minimizes squared error, not the farthest member, so an outlier
    can still leave its ball wide. O(n*k*d) per iteration."""

    def __init__(self, n_iter: int = 20, seed: int = 0) -> None:
        self.n_iter = n_iter
        self.seed = seed


    def construct_filter(self, filter: Filter, vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = np.ascontiguousarray(vectors, dtype=np.float32)
        n = X.shape[0]
        dim = X.shape[1]
        k = min(filter.max_construct_spheres, n)
        if k <= 0:
            return np.empty((0, dim), dtype=np.float32), np.empty(0, dtype=np.float32)
        if n <= k:
            return X.copy(), np.zeros(n, dtype=np.float32)

        seeds = X[self._kmeans_pp(X, k)]
        if len(seeds) < k:  # every vector already sits on a seed
            return exact_cover(X, seeds)

        # min_points_per_centroid=1 silences faiss's "too few points"
        # warning, which the small k-to-n ratios here would trip constantly.
        # max_points_per_centroid=n stops it subsampling the training set.
        kmeans = faiss.Kmeans(dim, k, niter=self.n_iter, seed=self.seed, verbose=False,
                              min_points_per_centroid=1, max_points_per_centroid=n)
        kmeans.train(X, init_centroids=seeds)

        # Radii from the exact re-search, never from k-means' own distances.
        return exact_cover(X, kmeans.centroids)


    def _kmeans_pp(self, X: np.ndarray, k: int) -> np.ndarray:
        """Row indices of up to k seeds, each drawn with probability
        proportional to its squared distance from the seeds so far. Stops
        early once every row coincides with a seed."""
        rng = np.random.default_rng(self.seed)
        n = X.shape[0]
        sq = np.einsum("ij,ij->i", X, X)

        chosen = np.empty(k, dtype=np.int64)
        min_d2 = np.empty(n, dtype=np.float32)
        d2 = np.empty(n, dtype=np.float32)

        # The same expanded d2 as GonzalesConstruct, clipped at 0: float32
        # cancellation only perturbs sampling weights here, never a radius.
        chosen[0] = rng.integers(n)
        np.dot(X, X[chosen[0]], out=min_d2)
        min_d2 *= -2.0
        min_d2 += sq
        min_d2 += sq[chosen[0]]
        np.maximum(min_d2, 0.0, out=min_d2)

        for j in range(1, k):
            cdf = np.cumsum(min_d2, dtype=np.float64)
            if cdf[-1] <= 0.0:
                return chosen[:j]
            c = int(np.searchsorted(cdf, rng.random() * cdf[-1], side="right"))
            chosen[j] = min(c, n - 1)

            np.dot(X, X[chosen[j]], out=d2)
            d2 *= -2.0
            d2 += sq
            d2 += sq[chosen[j]]
            np.maximum(d2, 0.0, out=d2)
            np.minimum(min_d2, d2, out=min_d2)
        return chosen


    def get_bytesize(self) -> int:
        return 0
