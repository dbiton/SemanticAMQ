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
        all_spheres (np.ndarray): A 2D array of shape (K, D) storing the centroids of all spheres.
        all_radiuses (np.ndarray): A 1D array of shape (K,) storing the bounding radiuses.
    """
    _BYTES_PER_FLOAT32 = 4
    _BYTES_PER_FLOAT64 = 8
    _MEB_ITERATIONS = 100
    _MAX_NEIGHBORS = 50
    _BATCH_SIZE = 2048


    def __init__(self, dim: int, distance_threshold: float, max_memory: Optional[int] = None, max_spheres: Optional[int] = None):
        super().__init__(dim, lsh=None, max_memory=max_memory)
        self.distance_threshold = distance_threshold
        self.all_spheres = np.empty((0, dim), dtype=np.float32)
        self.all_radiuses = np.empty((0,), dtype=np.float32)
        self._constructed = False

        self.max_spheres = (max_spheres if max_spheres is not None 
                           else self._calculate_max_spheres(self.max_memory))


    def _calculate_max_spheres(self, max_memory: int, gamma: float = 4.0) -> int:
        bytes_per_sphere = (self.dim + 1) * self._BYTES_PER_FLOAT32
        return max(1, int(max_memory // (bytes_per_sphere * gamma)))


    def _log_volume_bounds(self) -> Tuple[float, float]:
        if len(self.all_spheres) == 0:
            return -np.inf, -np.inf
        return estimate_hypersphere_volume(self.all_spheres, self.all_radiuses, return_log=True)


    def _compute_squared_distances(self, a: np.ndarray, a_sq: np.ndarray, b: np.ndarray, b_sq: np.ndarray) -> np.ndarray:
        a_b_cross = a @ b.T
        sq_distances = a_sq[:, None] - 2.0 * a_b_cross + b_sq[None, :]

        return np.maximum(sq_distances, 0.0)


    def _get_nearest_centers(self, vectors: np.ndarray, sq_norms: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        num_spheres = len(self.all_spheres)
        
        if num_spheres == 0:
            return np.empty((len(vectors), 0), dtype=np.float32), np.empty((len(vectors), 0), dtype=np.int64)

        k = min(k, num_spheres)
        centers = self.all_spheres
        sq_centers = np.sum(centers**2, axis=1)
        sq_distances = self._compute_squared_distances(vectors, sq_norms, centers, sq_centers)

        if k == num_spheres:
            center_indices = np.argsort(sq_distances, axis=1)

        else:
            center_indices = np.argpartition(sq_distances, k - 1, axis=1)[:, :k]
            top_k_dists = np.take_along_axis(sq_distances, center_indices, axis=1)
            sorted_top_k = np.argsort(top_k_dists, axis=1)
            center_indices = np.take_along_axis(center_indices, sorted_top_k, axis=1)

        center_sq_dists = np.take_along_axis(sq_distances, center_indices, axis=1)

        return center_sq_dists, center_indices


    def _gonzalez_clustering(self, vectors: np.ndarray, sq_norms: np.ndarray, k: int) -> np.ndarray:
        n = len(vectors)
        mean = vectors.mean(axis=0)
        first = int((sq_norms - 2.0 * (vectors @ mean)).argmax())

        c = vectors[first]
        nearest_sq = np.maximum(sq_norms - 2.0 * (vectors @ c) + c @ c, 0.0)
        assignment = np.zeros(n, dtype=np.int32)
        stop_sq = float(self.distance_threshold) ** 2

        for center_id in range(1, k):
            far_idx = int(nearest_sq.argmax())
            if nearest_sq[far_idx] <= stop_sq:
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


    def _merge_spheres(self) -> None:
        while len(self.all_spheres) > self.max_spheres:
            current_num_spheres = len(self.all_spheres)
            pairs_to_merge = current_num_spheres - self.max_spheres
            centers = self.all_spheres
            sq_centers = np.sum(self.all_spheres**2, axis=1)

            max_neighbors = min(self._MAX_NEIGHBORS, current_num_spheres)
            neighbor_dists, neighbor_indices = self._get_nearest_centers(centers, sq_centers, max_neighbors)

            pair_idx_i = np.repeat(np.arange(current_num_spheres), max_neighbors - 1)
            pair_idx_j = neighbor_indices[:, 1:].ravel()
            pair_distances = np.sqrt(np.maximum(neighbor_dists[:, 1:].ravel(), 0.0))

            valid_pairs = pair_idx_i < pair_idx_j
            pair_idx_i, pair_idx_j = pair_idx_i[valid_pairs], pair_idx_j[valid_pairs]
            pair_distances = pair_distances[valid_pairs]

            if len(pair_idx_i) == 0:
                break

            max_radius = max(np.max(self.all_radiuses), 1.0)
            pair_radius_i, pair_radius_j = self.all_radiuses[pair_idx_i], self.all_radiuses[pair_idx_j]
            pair_merged_radius = (pair_distances + pair_radius_i + pair_radius_j) / 2.0

            scaled_merged = (pair_merged_radius / max_radius).astype(np.float64)
            scaled_i = (pair_radius_i / max_radius).astype(np.float64)
            scaled_j = (pair_radius_j / max_radius).astype(np.float64)
            pair_volumes = (scaled_merged ** self.dim) - (scaled_i ** self.dim + scaled_j ** self.dim)
            sorted_pair_indices = np.argsort(pair_volumes)

            sphere_merged_indices = np.zeros(current_num_spheres, dtype=bool)
            merged_centers, merged_radiuses = [], []
            merges_done = 0

            for best_pair_idx in sorted_pair_indices:
                if merges_done >= pairs_to_merge or pair_volumes[best_pair_idx] == np.inf:
                    break

                idx_i, idx_j = int(pair_idx_i[best_pair_idx]), int(pair_idx_j[best_pair_idx])

                if sphere_merged_indices[idx_i] or sphere_merged_indices[idx_j]:
                    continue

                center_i, radius_i = self.all_spheres[idx_i], self.all_radiuses[idx_i]
                center_j, radius_j = self.all_spheres[idx_j], self.all_radiuses[idx_j]
                dist_ij = pair_distances[best_pair_idx]

                if dist_ij > 0.0:
                    weight_j = ((radius_j - radius_i) / (2.0 * dist_ij)) + 0.5
                    weight_i = ((radius_i - radius_j) / (2.0 * dist_ij)) + 0.5
                    new_center = weight_j * center_j + weight_i * center_i
                else:
                    new_center = center_i

                merged_centers.append(new_center)
                merged_radiuses.append(pair_merged_radius[best_pair_idx])
                sphere_merged_indices[idx_i], sphere_merged_indices[idx_j] = True, True
                merges_done += 1

            if not merged_centers:
                break

            self.all_spheres = np.vstack([self.all_spheres[~sphere_merged_indices], np.array(merged_centers, dtype=np.float32)])
            self.all_radiuses = np.concatenate([self.all_radiuses[~sphere_merged_indices], np.array(merged_radiuses, dtype=np.float32)])


    def _remove_spheres(self) -> None:
        num_spheres = len(self.all_spheres)

        if num_spheres < 2:
            return

        contained_spheres = np.zeros(num_spheres, dtype=bool)
        center_sq_norms = np.sum(self.all_spheres**2, axis=1)

        for i in range(0, num_spheres, self._BATCH_SIZE):
            end = min(i + self._BATCH_SIZE, num_spheres)

            container_centers = self.all_spheres[i:end]
            container_radiuses = self.all_radiuses[i:end]
            container_sq_norms = np.sum(container_centers**2, axis=1)

            pairwise_sq_distances = self._compute_squared_distances(
                self.all_spheres, center_sq_norms, container_centers, container_sq_norms
            )
            pairwise_distances = np.sqrt(np.maximum(pairwise_sq_distances, 0.0))

            is_contained_sphere = (pairwise_distances + self.all_radiuses[:, None]) <= container_radiuses[None, :]
            match_rows, match_cols = np.arange(i, end), np.arange(end - i)
            is_contained_sphere[match_rows, match_cols] = False

            contained_spheres |= np.any(is_contained_sphere, axis=1)

        if np.any(contained_spheres):
            self.all_spheres = self.all_spheres[~contained_spheres]
            self.all_radiuses = self.all_radiuses[~contained_spheres]


    def construct(self, vectors: np.ndarray) -> Tuple[float, float]:
        if len(vectors) == 0 or self.max_spheres == 0:
            self._constructed = True
            return self._log_volume_bounds()

        vectors = np.ascontiguousarray(vectors, dtype=np.float32)

        if self.max_spheres >= len(vectors):
            self.all_spheres = np.array(vectors, dtype=np.float32)
            self.all_radiuses = np.full(len(vectors), self.distance_threshold, dtype=np.float32)
            self._constructed = True
            return self._log_volume_bounds()

        sq_norms = np.sum(vectors * vectors, axis=1)
        assignment = self._gonzalez_clustering(vectors, sq_norms, self.max_spheres)

        num_of_clusters = int(assignment.max()) + 1
        clusters = [[] for _ in range(num_of_clusters)]
        for idx, cluster_id in enumerate(assignment):
            clusters[cluster_id].append(idx)

        centers, radiuses = [], []

        for cluster_id in clusters:
            center, radius = self._minimum_enclosing_ball(vectors[cluster_id], sq_norms[cluster_id])
            centers.append(center)
            radiuses.append(radius + self.distance_threshold)

        self.all_spheres = np.array(centers, dtype=np.float32)
        self.all_radiuses = np.array(radiuses, dtype=np.float32)
        self._constructed = True
        
        return self._log_volume_bounds()


    def query(self, vectors: np.ndarray) -> List[bool]:
        if not self._constructed:
            raise ValueError("GonzalezMEBFilter has not been constructed.")

        if len(self.all_spheres) == 0 or len(vectors) == 0:
            return [False] * len(vectors)

        q = np.ascontiguousarray(vectors, dtype=np.float32)
        q_sq = np.sum(q**2, axis=1)

        c = self.all_spheres
        c_sq = np.sum(self.all_spheres**2, axis=1)

        sq_distances = self._compute_squared_distances(q, q_sq, c, c_sq)
        in_any_sphere = np.any(sq_distances <= self.all_radiuses[None, :]**2, axis=1)

        return in_any_sphere.tolist()


    def add(self, vectors: np.ndarray) -> None:
        if not self._constructed:
            self.construct(vectors)
            return

        if len(vectors) == 0:
            return

        for i in range(0, len(vectors), self._BATCH_SIZE):
            vector_batch = vectors[i:i + self._BATCH_SIZE]
            vector_batch = np.ascontiguousarray(vector_batch, dtype=np.float32)
            sq_vector_batch = np.sum(vector_batch**2, axis=1)
            
            if len(self.all_spheres) > 0:
                max_neighbors = min(self._MAX_NEIGHBORS, len(self.all_spheres))
                neighbor_sq_dists, neighbor_indices = self._get_nearest_centers(vector_batch, sq_vector_batch, max_neighbors)
                neighbor_distances = np.sqrt(np.maximum(neighbor_sq_dists, 0.0))
                neighbor_radiuses = self.all_radiuses[neighbor_indices]

                is_already_covered = np.any(neighbor_distances <= (neighbor_radiuses - self.distance_threshold), axis=1)
                vector_batch = vector_batch[~is_already_covered]

            if len(vector_batch) == 0:
                continue

            self.all_spheres = np.vstack([self.all_spheres, vector_batch])
            self.all_radiuses = np.append(self.all_radiuses, np.full(len(vector_batch), self.distance_threshold, dtype=np.float32))

            if len(self.all_spheres) > self.max_spheres:
                self._merge_spheres()
                self._remove_spheres()


    def insert(self, vectors: np.ndarray) -> None:
        self.add(vectors)


    def remove(self, vectors: np.ndarray) -> List[bool]:
        raise NotImplementedError("GonzalezMEBFilter does not support remove operation.")


    def get_exact_matching_fpr(self) -> float:
        return 0.0


    def get_volume(self) -> float:
        return 0.0


    def get_bytesize(self) -> int:
        if not self._constructed:
            return 0
        return self.all_spheres.nbytes + self.all_radiuses.nbytes