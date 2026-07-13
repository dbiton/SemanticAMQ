import faiss
import numpy as np
from typing import List, Optional
from filters.filter import Filter


class KMEBFilter(Filter):
    """
    K-Means Embedded Balls (KMEB) filter — a static semantic filter.

    Each vector v passed to construct() implicitly represents a hypersphere of radius
    `distance_threshold` centred at v.  The filter compresses the union of those
    hyperspheres into at most K balls via K-Means clustering:

        ball_radius_i = max_dist(centroid_i, cluster_members_i) + distance_threshold

    This guarantees that every point within `distance_threshold` of any stored vector
    lies inside some ball (zero false negatives).  Because K-Means minimises
    within-cluster variance, each cluster is as compact as possible, so the total
    volume covered by all balls is minimised for a given K.

    Balls have per-cluster radii (varying radii), which avoids over-covering sparse
    regions that would arise from a uniform-radius scheme.
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
        self.max_spheres = (
            max_spheres
            if max_spheres is not None
            else self._calculate_max_spheres(max_memory)
        )

        self._centroids: Optional[np.ndarray] = None  # shape (K, dim), float32
        self._radii: Optional[np.ndarray] = None      # shape (K,), float32
        self._constructed = False

    def _calculate_max_spheres(self, max_memory: int, gamma: float = 4.0) -> int:
        bytes_per_sphere = (self.dim + 1) * self._BYTES_PER_FLOAT32
        # calculate expected memeory size for spheres with gamma factor
        return max(1, int(max_memory // (bytes_per_sphere * gamma)))

    def get_exact_matching_fpr(self) -> float:
        return 0.0

    def construct(self, vectors: np.ndarray) -> None:
        """
        Build K balls from vectors using K-Means clustering.

        Ball radius for cluster i = max distance from centroid to any cluster member
        plus `distance_threshold`, so that the ball covers the full semantic
        neighbourhood of every stored vector in that cluster.

        Varying radii minimise total covered volume: tight clusters get small balls,
        spread clusters get larger balls, each fitted exactly to their contents.
        """
        n = len(vectors)
        k = self.max_spheres

        if n == 0 or k == 0:
            self._centroids = np.empty((0, self.dim), dtype=np.float32)
            self._radii = np.empty(0, dtype=np.float32)
            self._constructed = True
            return

        if k >= n:
            # Trivial case: each vector is its own centroid; radius = distance_threshold
            self._centroids = np.array(vectors, dtype=np.float32)
            self._radii = np.full(n, self.distance_threshold, dtype=np.float32)
            self._constructed = True
            return

        # K-Means minimises within-cluster variance → compact clusters → minimal ball volume
        # Faiss K-Means uses fast multi-threaded operation for fast vector clustering
        vecs_f32 = np.ascontiguousarray(vectors, dtype=np.float32)
        kmeans = faiss.Kmeans(d=self.dim, k=k, niter=10, seed=42, min_points_per_centroid=1, verbose=False, spherical=False)
        kmeans.train(vecs_f32)
        centroids_f32 = kmeans.centroids

        _, labels_matrix = kmeans.index.search(vecs_f32, 1)
        labels = labels_matrix.flatten()

        # Compute each ball's radius in float32 (consistent with query arithmetic).
        # radius_i = cluster covering radius + distance_threshold
        # A small epsilon guards against boundary vectors being missed by rounding.
        radii = np.zeros(k, dtype=np.float32)
        for i in range(k):
            members = vectors[labels == i].astype(np.float32)
            if len(members) == 0:
                radii[i] = self.distance_threshold
                continue
            diff = members - centroids_f32[i]
            sq_dists = np.sum(diff ** 2, axis=1)
            radii[i] = float(np.sqrt(np.max(sq_dists))) + self.distance_threshold + 1e-6

        self._centroids = centroids_f32
        self._radii = radii
        self._constructed = True

    def query(self, vectors: np.ndarray) -> List[bool]:
        if not self._constructed:
            raise ValueError("KMEBFilter has not been constructed.")
        if len(self._centroids) == 0:
            return [False] * len(vectors)

        q = np.array(vectors, dtype=np.float32)
        # ||q - c||^2 = ||q||^2 - 2 q·c + ||c||^2  (avoids (N, K, dim) broadcast tensor)
        q_sq = np.sum(q ** 2, axis=1, keepdims=True)        # (N, 1)
        c_sq = np.sum(self._centroids ** 2, axis=1)          # (K,)
        cross = q @ self._centroids.T                         # (N, K)
        sq_dists = np.maximum(q_sq + c_sq[None, :] - 2 * cross, 0.0)  # (N, K)

        in_any_ball = np.any(sq_dists <= self._radii[None, :] ** 2, axis=1)
        return in_any_ball.tolist()

    def insert(self, vectors: np.ndarray) -> None:  # noqa: ARG002
        raise NotImplementedError("KMEBFilter is a static filter.")

    def add(self, vectors: np.ndarray) -> None:  # noqa: ARG002
        raise NotImplementedError("KMEBFilter is a static filter.")

    def remove(self, vectors: np.ndarray) -> List[bool]:  # noqa: ARG002
        raise NotImplementedError("KMEBFilter is a static filter.")

    def get_bytesize(self) -> int:
        if not self._constructed:
            return 0
        return self._centroids.nbytes + self._radii.nbytes
