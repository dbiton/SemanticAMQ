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
        # define number of spheres and number of dimentions
        K, D = self.all_spheres.shape

        # calculate global max radius for scaling (needed for computing volume correctly)
        max_r = max(np.max(self.all_radiuses), 1.0)

        # define best pair of spheres to merge with best minimal combined volume
        best_volume = np.inf
        best_pair = (-1, -1)
        best_r_new = 0.0
        best_distance = 0.0

        # iterate over each sphere i and compare against all subsequent spheres j
        for i in range(K - 1):
            j = i + 1 # next subsequent sphere

            # get current sphere i and subsequent spheres j
            c_i, r_i = self.all_spheres[i], self.all_radiuses[i]
            c_j, r_j = self.all_spheres[j:], self.all_radiuses[j:]

            # calculate the new merged sphere radius
            distance_ci_cj = np.linalg.norm(c_j - c_i, axis=1)
            r_new = (distance_ci_cj + r_i + r_j) / 2.0

            # scale our radiuses for computing volume with D dimensions
            r_new_scaled = (r_new / max_r).astype(np.float64)
            r_i_scaled = (r_i / max_r).astype(np.float64)
            r_j_scaled = (r_j / max_r).astype(np.float64)

            # calculate the volume of the current merged spheres
            volumes = (r_new_scaled ** D) - (r_i_scaled ** D + r_j_scaled ** D)

            # get the index and value of the minimum merge volume
            min_volume_idx = int(np.argmin(volumes))
            min_volume = volumes[min_volume_idx]

            # if current volume is smaller, update best pair so far
            if min_volume < best_volume:
                best_volume = min_volume
                best_pair = (i, j + min_volume_idx)
                best_r_new = r_new[min_volume_idx]
                best_distance = distance_ci_cj[min_volume_idx]

        # get best pair of spheres and their centers and radiuses
        i, j = best_pair
        c_i, r_i = self.all_spheres[i], self.all_radiuses[i]
        c_j, r_j = self.all_spheres[j], self.all_radiuses[j]

        # calculate the new center using the formula we've defined
        if best_distance > 0.0:
            best_c_new =  (((r_j - r_i) / (2.0 * best_distance)) + 0.5) * c_j + (((r_i - r_j) / (2.0 * best_distance)) + 0.5) * c_i
        else:
            best_c_new = c_i

        # update our spheres and radiuses with new merged pair with minimal volume
        self.all_spheres[i] = np.float32(best_c_new)
        self.all_radiuses[i] = np.float32(best_r_new)
        self.all_spheres = np.delete(self.all_spheres, j, axis=0)
        self.all_radiuses = np.delete(self.all_radiuses, j)
        # print(f"Found best volume {best_volume} with radius {best_r_new} and distance {best_distance}.")
    

    # method for removing spheres contained in other larger spheres
    def _remove_spheres(self) -> None:
        # define number of spheres
        K = self.all_spheres.shape[0]

        # if less then two sphere finsih
        if K < 2:
            return

        # represents spheres that are contained in other spheres
        contained_spheres = []

        # iterate over spheres and find contained spheres
        for i in range(K):
            # get current sphere center and radius
            c_i, r_i = self.all_spheres[i], self.all_radiuses[i]

            # represents all spheres big enough to contain current sphere
            contained_mask = self.all_radiuses >= r_i
            contained_mask[i] = False

            # if no other spheres are big enough we skip
            if not np.any(contained_mask):
                continue

            # get all potential spheres j that can contain current sphere i
            c_j, r_j = self.all_spheres[contained_mask], self.all_radiuses[contained_mask]
            distance_ci_cj = np.linalg.norm(c_j - c_i, axis=1)

            # check if any sphere contains current sphere d(c_i, c_j) + r_i <= r_j
            if np.any(distance_ci_cj + r_i <= r_j):
                contained_spheres.append(i)

        # remove all spheres we found that are already contained in other spheres
        if contained_spheres:
            self.all_spheres = np.delete(self.all_spheres, contained_spheres, axis=0)
            self.all_radiuses = np.delete(self.all_radiuses, contained_spheres)
        # print(f"Found {len(contained_spheres)} contained spheres.")


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
            sphere_data = vectors[rectangle_indices]

            # center of the sphere is the average vector of this group
            center = sphere_data.mean(axis=0, dtype=np.float32)

            # compute the maximum distance from center to any vector in the group
            max_distance = np.linalg.norm(sphere_data - center, axis=1).max(initial=0.0)

            # compute the radius based on max distance and threshold
            radius = np.float32(max_distance + self.distance_threshold)

            # get min and max coordinates across all dimensions
            # min_vec, max_vec = sphere_data.min(axis=0), sphere_data.max(axis=0)
            # center = np.float32((max_vec + min_vec) / 2.0)
            # radius = np.float32(np.linalg.norm((max_vec - min_vec) / 2.0) + self.distance_threshold)

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
        # if not initialized call construct
        if not self._constructed:
            self.construct(vectors)

        # if no spheres exist or no vectors given return results
        if len(self.all_spheres) == 0 or len(vectors) == 0:
            return [False] * len(vectors)

        # cast queries to float32 array for better performance
        q = np.asarray(vectors, dtype=np.float32)

        # compute components for the squared distance identity
        # ||q - c||^2 = ||q||^2 - 2(q·c) + ||c||^2
        q_sq = np.sum(q ** 2, axis=1, keepdims=True)        
        c_sq = np.sum(self.all_spheres ** 2, axis=1)          
        cross = q @ self.all_spheres.T                     

        # combine terms via broadcasting to get the total squared distance matrix
        sq_distances = np.maximum(q_sq + c_sq[None, :] - 2 * cross, 0.0) # shape (N, K)

        # vectorized comparison against squared radiuses
        in_any_sphere = np.any(sq_distances <= self.all_radiuses[None, :] ** 2, axis=1)

        return in_any_sphere.tolist()


    # method for adding new vectors into filter
    def add(self, vectors: np.ndarray) -> None:
        # if filter is not initialized we call construct
        if not self._constructed:
            self.construct(vectors)
            return

        # if vectors array is empty we return
        if len(vectors) == 0:
            return

        # cast new vector queries to float32 array for better performance
        new_vectors = np.asarray(vectors, dtype=np.float32)

        # process each new query vector and add it to filter
        for q in new_vectors:
            # if we have spheres we check if given query is inside one
            if len(self.all_spheres) > 0:
                # calculate the sphere distances from query and get best sphere
                sphere_distances = np.linalg.norm(self.all_spheres - q, axis=1)
                best_sphere = int(np.argmin(sphere_distances))

                # check if query inside sphere d(c, q) <= r - threshold
                if sphere_distances[best_sphere] <= (self.all_radiuses[best_sphere] - self.distance_threshold):
                    continue

            # create new sphere for the query using threshold as its initial radius
            self.all_spheres = np.vstack([self.all_spheres, q])
            self.all_radiuses = np.append(self.all_radiuses, np.float32(self.distance_threshold))

            # merge overlapping spheres if max sphere count is exceeded
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