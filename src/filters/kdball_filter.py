import sys, heapq
import numpy as np
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
        all_spheres (np.ndarray): A 1D array of shape (K,) containing the active indices of the spheres.
        all_centers (np.ndarray): A 2D array of shape (K, D) storing the centroids of all spheres.
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
        self.max_spheres = (max_spheres if max_spheres is not None 
                    else self._calculate_max_spheres(self.max_memory))
        self._num_spheres = 0
        self._all_spheres = np.arange(self.max_spheres, dtype=np.int32) # represents vector of sphere indices of shape (K,)
        self._all_centers = np.empty((self.max_spheres, dim), dtype=np.float32) # represents matrix of spheres center of shape (K, D)
        self._all_radiuses = np.empty((self.max_spheres,), dtype=np.float32) # represents vector of radiuses of shape (K,)
        self._indices_count = np.zeros((self.max_spheres,), dtype=np.int32) # represents vector of indices counters of shape (K,)
        self._minimal_pairs = [] # represents min-heap that stores tuples (volume, i, i_count, j, j_count) representing pairs with minimal volume merge increase
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


    # method for obtaining new sphere index, increases the size of sphere data structures when capacity exceeded
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


    # method for adding a new sphere into filter, checking if it contains or is contained by existing spheres and adding best pairs into heap
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
            merged_radius = (neighbors_radiuses + neighbors_center_dists + radius) / 2.0

            # log(R_merged^D - (R_1^D + R_2^D)) = D*log(R_merged) + log(1 - ((R_1/R_merged)^D + (R_2/R_merged)^D))
            merged_volumes = self.dim * np.log(merged_radius) + np.log(1.0 - ((radius / merged_radius) ** self.dim + (neighbors_radiuses / merged_radius) ** self.dim))

            for volume, neighbor_idx in zip(merged_volumes, closest_neighbors):
                neighbor_count = self._indices_count[neighbor_idx]
                heapq.heappush(self._minimal_pairs, (volume, sphere_idx, sphere_count, neighbor_idx, neighbor_count))


    # method for merging overlapping spheres by finding the pair that produces the minimal volume increase
    def _merge_spheres(self) -> None:
        while self._num_spheres > self.max_spheres and self._minimal_pairs:
            vol, i, i_count, j, j_count = heapq.heappop(self._minimal_pairs)

            if self._indices_count[i] != i_count or self._indices_count[j] != j_count:
                continue

            center_i, radius_i = self._all_centers[i], self._all_radiuses[i]
            center_j, radius_j = self._all_centers[j], self._all_radiuses[j]

            centers_diff = center_i - center_j
            distance_ij = float(np.sqrt(np.dot(centers_diff, centers_diff)))
            new_radius = (distance_ij + radius_i + radius_j) / 2.0

            if distance_ij > 0.0:
                weight_j = ((radius_j - radius_i) / (2.0 * distance_ij)) + 0.5
                weight_i = ((radius_i - radius_j) / (2.0 * distance_ij)) + 0.5
                new_center = weight_j * center_j + weight_i * center_i
            else:
                new_center = center_i

            current_spheres = self.all_spheres[(self.all_spheres != i) & (self.all_spheres != j)]
            self._all_spheres[:self._num_spheres] = np.concatenate([current_spheres, [i, j]])
            self._indices_count[[i, j]] += 1
            self._num_spheres -= 2

            self._add_sphere(new_center, new_radius)


    # method for constructing the filter attributes and finding optimal spheres to cover the vector space 
    # using a KD-style approach that splits the space into rectangles and creates bounding spheres
    def construct(self, vectors: np.ndarray) -> None:
        # if empty array we return empty filter
        if len(vectors) == 0 or self.max_spheres == 0:
            self._constructed = True
            return

        # if we have enough spheres we create one for each vector, essentially creating trivial case
        elif self.max_spheres >= len(vectors):
            for vec in vectors:
                self._add_sphere(vec, self.distance_threshold)
            self._constructed = True
            return

        # represnets max heap for rectangles, stores each rectangle properties
        rectangles: List[Tuple[float, int, float, np.ndarray]] = []

        # create initial rectangle and find its best margin and push into heap
        initial_indices = np.arange(len(vectors), dtype=np.int32)
        margin, axis, split_val = self._find_widest_margin(vectors)
        heapq.heappush(rectangles, (-margin, axis, split_val, initial_indices))

        # each iteration split until we reached max amount of spheres
        while len(rectangles) < self.max_spheres and len(rectangles) < len(vectors):
            # get best rectangle from heap
            p_margin, p_axis, p_split_val, parent_indices = heapq.heappop(rectangles)
            p_margin = -p_margin

            # if margin is smaller then our threshold we dont split and finish
            if p_margin <= self.distance_threshold:
                heapq.heappush(rectangles, (-p_margin, p_axis, p_split_val, parent_indices))
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
                heapq.heappush(rectangles, (-l_margin, l_axis, l_split_val, left_indices))

            # append right child if not empty
            if len(right_indices) > 0:
                r_margin, r_axis, r_split_val = self._find_widest_margin(vectors[right_indices])
                heapq.heappush(rectangles, (-r_margin, r_axis, r_split_val, right_indices))

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

            # add new sphere to our current spheres
            self._add_sphere(center, radius)

        self._constructed = True
        # print(f"Number of spheres found: {self.all_spheres.shape[0]}, max_spheres: {self.max_spheres}")


    # method to query our filter and check if query vectors fall inside our spheres
    def query(self, vectors: np.ndarray) -> List[bool]:
        # check if filter is not initialized
        if not self._constructed:
            raise ValueError("KDBallFilter has not been constructed.")

        # if no spheres exist or no vectors given return results
        if len(self.all_centers) == 0 or len(vectors) == 0:
            return [False] * len(vectors)

        # cast queries to float32 array and compute squared norm
        q = np.ascontiguousarray(vectors, dtype=np.float32)
        q_sq = np.sum(q**2, axis=1)

        # get sphere centers and compute squared norm
        c = self.all_centers
        c_sq = np.sum(self.all_centers**2, axis=1)

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

            if self._num_spheres > 0:
                sq_centers = np.sum(self.all_centers**2, axis=1)
                sq_batch_distances = self._compute_squared_distances(vector_batch, sq_vector_batch, self.all_centers, sq_centers)
                batch_distances = np.sqrt(sq_batch_distances)

                is_already_covered = np.any(batch_distances <= (self.all_radiuses[None, :] - self.distance_threshold), axis=1)
                vector_batch = vector_batch[~is_already_covered]

            for vec in vector_batch:
                self._add_sphere(vec, self.distance_threshold)

            self._merge_spheres()


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
        return (self._all_spheres.nbytes + self._all_centers.nbytes + self._all_radiuses.nbytes + 
                self._indices_count.nbytes + sys.getsizeof(self._minimal_pairs))