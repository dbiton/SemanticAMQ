import numpy as np
from heapq import heappop, heappush
from typing import List, Tuple, Optional
from filters.filter import Filter


class KDBallFilter(Filter):
    """
    A high-performance spatial filter that partitions high-dimensional vector data into bounding spheres.

    The KDBallFilter uses a KD-tree-inspired top-down splitting strategy to recursively partition the vector space
    along the widest empty margins, forming axis-aligned hyper-rectangular regions (KD-style partitions).

    Each resulting rectangular region is then approximated using a bounding sphere (via centroid + radius),
    creating a relaxed geometric representation that enables efficient similarity and containment queries.

    This filter design combines:
    - KD-like recursive space partitioning (rectangular decomposition)
    - Spherical relaxation of each partition for fast Euclidean distance checks

    The filter is designed with a one-sided error guarantee:
    it allows false positives but guarantees no false negatives (no FN),
    ensuring that any vector that should match is never incorrectly excluded.

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
    _MAX_NEIGHBORS = 50
    _BATCH_SIZE = 2048


    # constructor of our KDBallFilter class, initializes class attributes and max amount of spheres
    def __init__(self, dim: int, distance_threshold: float, max_memory: Optional[int] = None, max_spheres: Optional[int] = None):
        # initialize class attributes and call parent class ctor
        super().__init__(dim, lsh=None, max_memory=max_memory)
        self.distance_threshold = distance_threshold
        self.all_spheres = np.empty((0, dim), dtype=np.float32) # represents matrix of spheres center of shape (K, D)
        self.all_radiuses = np.empty((0,), dtype=np.float32) # represents vector of radiuses of shape (K,)
        self._constructed = False

        # handle memory limit based on max memory limit given
        self.max_spheres = (max_spheres if max_spheres is not None 
                           else self._calculate_max_spheres(self.max_memory))


    # method for calculating optimal amount of spheres we allow based on max memory size
    def _calculate_max_spheres(self, max_memory: int, gamma: float = 4.0) -> int:
        bytes_per_sphere = (self.dim + 1) * self._BYTES_PER_FLOAT32
        # calculate expected memeory size for spheres with gamma factor
        return max(1, int(max_memory // (bytes_per_sphere * gamma)))


    # method for computing pairwise squared euclidean distances between two arrays of vectors efficiently
    def _compute_squared_distances(self, a: np.ndarray, a_sq: np.ndarray, b: np.ndarray, b_sq: np.ndarray) -> np.ndarray:
        a_b_cross = a @ b.T
        sq_distances = a_sq[:, None] - 2.0 * a_b_cross + b_sq[None, :]

        return np.maximum(sq_distances, 0.0)


    # method that finds the k-nearest sphere centers for given vectors using partitioned distance queries
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


    # method for finding the widest margin in high dimensional space for giving us best split best on given vector group
    def _find_widest_margin(self, vector_group: np.ndarray) -> Tuple[float, int, float]:
        # if less then two vectors we dont split
        if len(vector_group) < 2:
            return -1.0, None, None

        # get min and max values for all dimensions
        group_min = vector_group.min(axis=0)
        group_max = vector_group.max(axis=0)

        # calculate the overall margin per dimension
        margins = group_max - group_min

        # find the axis with the largest overall margin
        best_axis = int(np.argmax(margins))
        best_margin = float(margins[best_axis])
        best_split_val = float(group_min[best_axis] + (best_margin / 2.0))

        return best_margin, best_axis, best_split_val


    # method for merging overlapping spheres by finding the pair that produces the minimal volume increase
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


    # method for removing spheres contained in other larger spheres
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


    # method for constructing the filter attributes and finding optimal spheres to cover the vector space 
    # using a KD-style approach that splits the space into rectangles and creates bounding spheres
    def construct(self, vectors: np.ndarray) -> None:
        # if empty array we return empty filter
        if len(vectors) == 0 or self.max_spheres == 0:
            self._constructed = True
            return

        # if we have enough spheres we create one for each vector, essentially creating trivial case
        elif self.max_spheres >= len(vectors):
            self.all_spheres = np.array(vectors, dtype=np.float32)
            self.all_radiuses = np.full(len(vectors), self.distance_threshold, dtype=np.float32)
            self._constructed = True
            return

        # represnets max heap for rectangles, stores each rectangle properties
        rectangles: List[Tuple[float, int, float, np.ndarray]] = []

        # create initial rectangle and find its best margin and push into heap
        initial_indices = np.arange(len(vectors), dtype=np.int32)
        margin, axis, split_val = self._find_widest_margin(vectors)
        heappush(rectangles, (-margin, axis, split_val, initial_indices))

        # each iteration split until we reached max amount of spheres
        while len(rectangles) < self.max_spheres and len(rectangles) < len(vectors):
            # get best rectangle from heap
            p_margin, p_axis, p_split_val, parent_indices = heappop(rectangles)
            p_margin = -p_margin

            # if margin is smaller then our threshold we dont split and finish
            if p_margin <= self.distance_threshold:
                heappush(rectangles, (-p_margin, p_axis, p_split_val, parent_indices))
                break

            # split parent into two child rectangles in best axis and split value
            current_vectors = vectors[parent_indices]
            left_mask = current_vectors[:, p_axis] < p_split_val
            right_mask = ~left_mask
            left_indices = parent_indices[left_mask]
            right_indices = parent_indices[right_mask]

            # append left child if not empty
            if len(left_indices) > 0:
                l_margin, l_axis, l_split_val = self._find_widest_margin(vectors[left_indices])
                heappush(rectangles, (-l_margin, l_axis, l_split_val, left_indices))

            # append right child if not empty
            if len(right_indices) > 0:
                r_margin, r_axis, r_split_val = self._find_widest_margin(vectors[right_indices])
                heappush(rectangles, (-r_margin, r_axis, r_split_val, right_indices))

        # convert final rectangle groups into spheres
        sphere_centers, sphere_radiuses = [], []

        # iterate over each group we saved and compute sphere properties like center and radius
        for rectangle in rectangles:
            rectangle_indices = rectangle[3]

            # if group is empty we skip
            if len(rectangle_indices) == 0:
                continue

            # get rectangle vectors for calculating sphere properties
            sphere_vectors = vectors[rectangle_indices]
            sphere_vectors_sq = np.sum(sphere_vectors**2, axis=1)

            # center of the sphere is the average vector of this group
            center = sphere_vectors.mean(axis=0, dtype=np.float32)
            center_sq = np.array([np.sum(center**2)], dtype=np.float32)

            # compute the maximum distance from center to any vector in the group
            sq_distances = self._compute_squared_distances(sphere_vectors, sphere_vectors_sq, center[None, :], center_sq)
            max_distance = np.sqrt(sq_distances.max())

            # compute the radius based on max distance and threshold
            radius = np.float32(max_distance + self.distance_threshold)

            # append center and radius to our lists
            sphere_centers.append(center)
            sphere_radiuses.append(radius)

        # initialize arrays with our final spheres properties
        self.all_spheres = np.asarray(sphere_centers, dtype=np.float32)
        self.all_radiuses = np.asarray(sphere_radiuses, dtype=np.float32)
        self._constructed = True
        # print(f"Number of spheres found: {self.all_spheres.shape[0]}, max_spheres: {self.max_spheres}")


    # method to query our filter and check if query vectors fall inside our spheres
    def query(self, vectors: np.ndarray) -> List[bool]:
        # check if filter is not initialized
        if not self._constructed:
            raise ValueError("KDBallFilter has not been constructed.")

        # if no spheres exist or no vectors given return results
        if len(self.all_spheres) == 0 or len(vectors) == 0:
            return [False] * len(vectors)

        # cast queries to float32 array and compute squared norm
        q = np.ascontiguousarray(vectors, dtype=np.float32)
        q_sq = np.sum(q**2, axis=1)

        # get sphere centers and compute squared norm
        c = self.all_spheres
        c_sq = np.sum(self.all_spheres**2, axis=1)

        # vectorized comparison against squared radiuses
        sq_distances = self._compute_squared_distances(q, q_sq, c, c_sq)
        in_any_sphere = np.any(sq_distances <= self.all_radiuses[None, :]**2, axis=1)

        return in_any_sphere.tolist()


    # method for adding new vectors into filter
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


    # method for inserting new vectors into filter
    def insert(self, vectors: np.ndarray) -> None:
        self.add(vectors)


    # method for removing vectors from filter
    def remove(self, vectors: np.ndarray) -> List[bool]:
        raise NotImplementedError("KDBallFilter does not support remove operation.")


    # method to get exact fpr
    def get_exact_matching_fpr(self) -> float:
        return 0.0


    # method to get bytesize of overall filter
    def get_bytesize(self) -> int:
        if not self._constructed:
            return 0
        # return the overall spheres size including size of radiuses
        return self.all_spheres.nbytes + self.all_radiuses.nbytes