import sys, heapq
import numpy as np
from typing import List, Tuple, Optional
from filters.filter import Filter
from hyperspheres_volume import estimate_hypersphere_volume


class GonzalezMEBFilter(Filter):
    """
    A high-performance spatial filter that partitions high-dimensional vector data into bounding spheres.

    The GonzalezMEBFilter combines greedy spatial clustering with core-set optimization to efficiently 
    approximate complex vector distributions using a bounded collection of hyperspheres.

    The partitioning strategy leverages:
    - Gonzalez's algorithm for initial greedy k-center clustering, which provides a fast
      initialization and a 2-approximation of the cluster radius.
    - Badoiu-Clarkson iterative core-set refinement to compute a tight Minimum 
      Enclosing Ball (MEB) for each cluster, optimizing the spatial boundaries.

    The filter is designed with a strict one-sided error guarantee: it allows false positives 
    for fast pruning, but guarantees zero false negatives (no FN), ensuring that any matching 
    vector is never incorrectly excluded.

    Attributes:
        dim (int): The dimensionality of the vector space.
        distance_threshold (float): Additional padding (margin of error) added to the radius of each
                                   bounding sphere to ensure boundary vectors are fully enclosed.
        max_memory (Optional[int]): The maximum memory limit in bytes allocated for storing spheres.
        max_spheres (int): The maximum number of space partitions (spheres) allowed, automatically
                           calculated based on the memory limit to prevent overflow.
        all_spheres (np.ndarray): A 1D array of shape (K,) containing the active indices of the spheres.
        all_centers (np.ndarray): A 2D array of shape (K, D) storing the centroids of all spheres.
        all_radiuses (np.ndarray): A 1D array of shape (K,) storing the bounding radiuses.
    """
    _BYTES_PER_FLOAT32 = 4
    _BYTES_PER_FLOAT64 = 8
    _MEB_ITERATIONS = 100
    _MAX_NEIGHBORS = 50
    _COMPACTION_RATIO = 2
    _BATCH_SIZE = 2048

    def __init__(self, dim: int, distance_threshold: float, max_memory: Optional[int] = None, max_spheres: Optional[int] = None):
        super().__init__(dim, max_memory=max_memory)
        self.distance_threshold = distance_threshold
        self.max_spheres = (max_spheres if max_spheres is not None 
                    else self._calculate_max_spheres(self.max_memory))
        self._num_spheres = 0
        self._all_spheres = np.arange(self.max_spheres, dtype=np.int32)
        self._all_centers = np.empty((self.max_spheres, dim), dtype=np.float32)
        self._all_radiuses = np.empty((self.max_spheres,), dtype=np.float32)
        self._indices_count = np.zeros((self.max_spheres,), dtype=np.int32)
        self._minimal_pairs = []
        self._constructed = False

    @property
    def all_spheres(self) -> np.ndarray:
        return self._all_spheres[:self._num_spheres]

    @property
    def all_centers(self) -> np.ndarray:
        return self._all_centers[self.all_spheres]

    @property
    def all_radiuses(self) -> np.ndarray:
        return self._all_radiuses[self.all_spheres]


    def _calculate_max_spheres(self, max_memory: int, gamma: float = 4.0) -> int:
        bytes_per_sphere = (self.dim + 1) * self._BYTES_PER_FLOAT32
        return max(1, int(max_memory // (bytes_per_sphere * gamma)))


    def _log_volume_bounds(self) -> Tuple[float, float]:
        if len(self.all_centers) == 0:
            return -np.inf, -np.inf
        return estimate_hypersphere_volume(self.all_centers, self.all_radiuses, return_log=True)


    def _compute_squared_distances(self, a: np.ndarray, a_sq: np.ndarray, b: np.ndarray, b_sq: np.ndarray) -> np.ndarray:
        a_b_cross = a @ b.T
        sq_distances = a_sq[:, None] - 2.0 * a_b_cross + b_sq[None, :]
        return np.maximum(sq_distances, 0.0)


    def _get_new_index(self) -> int:
        if self._num_spheres == len(self._all_centers):
            old_cap = len(self._all_centers)
            new_cap = old_cap * 2

            new_spheres = np.empty((new_cap,), dtype=np.int32)
            new_spheres[:old_cap] = self._all_spheres
            new_spheres[old_cap:] = np.arange(old_cap, new_cap, dtype=np.int32)
            self._all_spheres = new_spheres

            new_centers = np.empty((new_cap, self.dim), dtype=np.float32)
            new_centers[:old_cap] = self._all_centers
            self._all_centers = new_centers

            new_radiuses = np.empty((new_cap,), dtype=np.float32)
            new_radiuses[:old_cap] = self._all_radiuses
            self._all_radiuses = new_radiuses

            new_indices_count = np.zeros(new_cap, dtype=np.int32)
            new_indices_count[:old_cap] = self._indices_count
            self._indices_count = new_indices_count

        return self._all_spheres[self._num_spheres]


    def _gonzalez_clustering(self, vectors: np.ndarray, sq_norms: np.ndarray, k: int) -> np.ndarray:
        n = len(vectors)
        mean = vectors.mean(axis=0)
        first = int((sq_norms - 2.0 * (vectors @ mean)).argmax())

        c = vectors[first]
        nearest_sq = np.maximum(sq_norms - 2.0 * (vectors @ c) + c @ c, 0.0)
        assignment = np.zeros(n, dtype=np.int32)

        for center_id in range(1, k):
            far_idx = int(nearest_sq.argmax())
            if nearest_sq[far_idx] <= 1e-12:
                break
            c = vectors[far_idx]
            new_sq = np.maximum(sq_norms - 2.0 * (vectors @ c) + c @ c, 0.0)
            closer = new_sq < nearest_sq
            np.copyto(nearest_sq, new_sq, where=closer)
            assignment[closer] = center_id

        return assignment


    def _minimum_enclosing_ball(self, points: np.ndarray, sq_norms: np.ndarray) -> Tuple[np.ndarray, float]:
        if len(points) == 1:
            return points[0].astype(np.float32), 0.0

        center = points.mean(axis=0, dtype=np.float64)
        for t in range(1, self._MEB_ITERATIONS + 1):
            sq_d = sq_norms - 2.0 * (points @ center) + center @ center
            center += (points[int(sq_d.argmax())] - center) / (t + 1.0)

        sq_d = np.maximum(sq_norms - 2.0 * (points @ center) + center @ center, 0.0)
        radius = np.sqrt(sq_d.max())

        return center.astype(np.float32), radius.astype(np.float32)


    def _remove_invalid_pairs(self) -> None:
        if len(self._minimal_pairs) > self._COMPACTION_RATIO * self.max_spheres * self._MAX_NEIGHBORS:
            versions = self._indices_count
            dtype = [('vol', 'f8'), ('i', 'i4'), ('ver_i', 'i4'), ('j', 'i4'), ('ver_j', 'i4')]
            pairs = np.asarray(self._minimal_pairs, dtype=dtype)

            valid_pairs = (versions[pairs['i']] == pairs['ver_i']) & (versions[pairs['j']] == pairs['ver_j'])
            self._minimal_pairs = pairs[valid_pairs].tolist()
            heapq.heapify(self._minimal_pairs)


    def _add_sphere(self, center: np.ndarray, radius: float) -> None:
        current_spheres = self.all_spheres

        if self._num_spheres > 0:
            current_centers = self._all_centers[current_spheres]
            current_radiuses = self._all_radiuses[current_spheres]
            center_sq = np.sum(center**2, keepdims=True)
            current_centers_sq = np.sum(current_centers**2, axis=1)

            sq_center_distances = self._compute_squared_distances(
                center[None, :], center_sq, current_centers, current_centers_sq)[0]
            center_distances = np.sqrt(sq_center_distances)

            if np.any(center_distances + radius <= current_radiuses):
                return

            contained_indices = (center_distances + current_radiuses) <= radius
            if np.any(contained_indices):
                contained_spheres = current_spheres[contained_indices]
                current_spheres = current_spheres[~contained_indices]
                self._all_spheres[:self._num_spheres] = np.concatenate([current_spheres, contained_spheres])
                self._num_spheres -= len(contained_spheres)
                self._indices_count[contained_spheres] += 1
                current_radiuses = current_radiuses[~contained_indices]
                center_distances = center_distances[~contained_indices]

        sphere_idx = self._get_new_index()
        sphere_count = self._indices_count[sphere_idx]
        self._all_centers[sphere_idx] = center
        self._all_radiuses[sphere_idx] = radius
        self._num_spheres += 1

        if len(current_spheres) > 0:
            max_neighbors = min(self._MAX_NEIGHBORS, len(current_spheres))
            if max_neighbors == len(current_spheres):
                neighbors_indices = np.arange(len(current_spheres))
            else:
                neighbors_indices = np.argpartition(center_distances, max_neighbors - 1)[:max_neighbors]

            closest_neighbors = current_spheres[neighbors_indices]
            neighbors_center_dists = center_distances[neighbors_indices]
            neighbors_radiuses = current_radiuses[neighbors_indices]
            merged_radius = np.maximum((neighbors_radiuses + neighbors_center_dists + radius) / 2.0, 1e-6)

            # log(R_merged^D - (R_1^D + R_2^D)) = D*log(R_merged) + log(1 - ((R_1/R_merged)^D + (R_2/R_merged)^D))
            merged_volumes = self.dim * np.log(merged_radius) + np.log(np.maximum(1.0 - ((radius / merged_radius) ** self.dim + (neighbors_radiuses / merged_radius) ** self.dim), 1e-6))

            for volume, neighbor_idx in zip(merged_volumes, closest_neighbors):
                neighbor_count = self._indices_count[neighbor_idx]
                heapq.heappush(self._minimal_pairs, (volume, sphere_idx, sphere_count, neighbor_idx, neighbor_count))


    def _merge_spheres(self) -> None:
        while self._num_spheres > self.max_spheres and self._minimal_pairs:
            vol, i, i_count, j, j_count = heapq.heappop(self._minimal_pairs)

            if self._indices_count[i] != i_count or self._indices_count[j] != j_count:
                continue

            center_i, radius_i = self._all_centers[i], self._all_radiuses[i]
            center_j, radius_j = self._all_centers[j], self._all_radiuses[j]
            centers_diff = center_i - center_j
            distance_ij = float(np.sqrt(np.dot(centers_diff, centers_diff)))

            if distance_ij > 0.0:
                weight_j = ((radius_j - radius_i) / (2.0 * distance_ij)) + 0.5
                weight_i = ((radius_i - radius_j) / (2.0 * distance_ij)) + 0.5
                new_center = weight_j * center_j + weight_i * center_i
                new_radius = max((distance_ij + radius_i + radius_j) / 2.0, 1e-6)
            else:
                new_center = center_i
                new_radius = max(radius_i, radius_j)

            current_spheres = self.all_spheres[(self.all_spheres != i) & (self.all_spheres != j)]
            self._all_spheres[:self._num_spheres] = np.concatenate([current_spheres, [i, j]])
            self._indices_count[[i, j]] += 1
            self._num_spheres -= 2

            self._add_sphere(new_center, new_radius)


    def construct(self, vectors: np.ndarray) -> Tuple[float, float]:
        if len(vectors) == 0 or self.max_spheres == 0:
            self._constructed = True
            return self._log_volume_bounds()

        vectors = np.ascontiguousarray(vectors, dtype=np.float32)

        if self.max_spheres >= len(vectors):
            for vec in vectors:
                self._add_sphere(vec, 0.0)
            self._constructed = True
            return self._log_volume_bounds()

        sq_norms = np.sum(vectors * vectors, axis=1)
        assignment = self._gonzalez_clustering(vectors, sq_norms, self.max_spheres)

        num_of_clusters = int(assignment.max()) + 1
        clusters = [[] for _ in range(num_of_clusters)]
        for idx, cluster_id in enumerate(assignment):
            clusters[cluster_id].append(idx)

        for cluster_id in clusters:
            center, radius = self._minimum_enclosing_ball(vectors[cluster_id], sq_norms[cluster_id])
            self._add_sphere(center, radius)
        self._constructed = True

        return self._log_volume_bounds()


    def query(self, vectors: np.ndarray, distance_threshold: Optional[float] = None) -> List[bool]:
        if not self._constructed:
            raise ValueError("GonzalezMEBFilter has not been constructed.")

        if len(self.all_centers) == 0 or len(vectors) == 0:
            return [False] * len(vectors)

        if distance_threshold is None:
            distance_threshold = self.distance_threshold

        q = np.ascontiguousarray(vectors, dtype=np.float32)
        q_sq = np.sum(q**2, axis=1)

        c = self.all_centers
        c_sq = np.sum(self.all_centers**2, axis=1)

        sq_distances = self._compute_squared_distances(q, q_sq, c, c_sq)
        all_radiuses = self.all_radiuses + distance_threshold
        in_any_sphere = np.any(sq_distances <= (all_radiuses[None, :] ** 2), axis=1)

        return in_any_sphere.tolist()


    def insert(self, vectors: np.ndarray) -> None:
        if not self._constructed:
            self.construct(vectors)
            return

        if len(vectors) == 0:
            return

        for i in range(0, len(vectors), self._BATCH_SIZE):
            vector_batch = vectors[i:i + self._BATCH_SIZE]
            vector_batch = np.ascontiguousarray(vector_batch, dtype=np.float32)
            sq_vector_batch = np.sum(vector_batch**2, axis=1)

            if self._num_spheres > 0:
                sq_centers = np.sum(self.all_centers**2, axis=1)
                sq_batch_distances = self._compute_squared_distances(vector_batch, sq_vector_batch, self.all_centers, sq_centers)
                batch_distances = np.sqrt(sq_batch_distances)

                is_already_covered = np.any(batch_distances <= self.all_radiuses[None, :], axis=1)
                vector_batch = vector_batch[~is_already_covered]

            for vec in vector_batch:
                self._add_sphere(vec, 0.0)

            self._merge_spheres()
            self._remove_invalid_pairs()


    def add(self, vectors: np.ndarray) -> None:
        self.insert(vectors)


    def remove(self, vectors: np.ndarray) -> List[bool]:
        raise NotImplementedError("GonzalezMEBFilter does not support remove operation.")


    def get_exact_matching_fpr(self) -> float:
        return 0.0


    def get_volume(self) -> float:
        return 0.0


    def get_bytesize(self) -> int:
        if not self._constructed:
            return 0
        filter_bytes = self._all_spheres.nbytes + self._all_centers.nbytes + self._all_radiuses.nbytes
        pair_size = 216 # PyTuple (80B) + PyFloat (24B) + 4 * PyInt (28B)
        heap_bytes = self._indices_count.nbytes + sys.getsizeof(self._minimal_pairs) + pair_size * len(self._minimal_pairs)
        return filter_bytes + heap_bytes