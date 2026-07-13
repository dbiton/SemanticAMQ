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
        all_spheres (np.ndarray): A 2D array of shape (K, dim) storing the centroids of all spheres.
        all_radiuses (np.ndarray): A 1D array of shape (K,) storing the bounding radiuses.
        sphere_counts (np.ndarray): A 1D array of shape (K,) storing the count of membership vectors per sphere.
    """
    _BYTES_PER_FLOAT32 = 4
    _BYTES_PER_FLOAT64 = 8


    # constructor of our KDBallFilter class, initializes class attributes and max amount of spheres
    def __init__(self, dim: int, distance_threshold: float, max_memory: Optional[int] = None, max_spheres: Optional[int] = None):
        # initialize class attributes and call parent class ctor
        super().__init__(dim, lsh=None, max_memory=max_memory)
        self.distance_threshold = distance_threshold
        self.all_spheres = np.empty((0, dim), dtype=np.float32) # represents matrix of spheres center of shape (K, dim)
        self.all_radiuses = np.empty((0,), dtype=np.float32) # represents vector of radiuses of shape (K,)
        self.sphere_counts = np.empty((0,), dtype=np.int32) # represents vector of sphere counts of shape (K,)
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
        best_axis = int(margins.argmax())
        best_margin = float(margins[best_axis])
        best_split_val = float(group_min[best_axis] + (best_margin / 2.0))

        return best_margin, best_axis, best_split_val


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
            if p_margin <= self.distance_threshold * 2.0:
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
        sphere_centers, sphere_radiuses, sphere_counts = [], [], []

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

            # compute distance from center to every vector in group
            distances = np.linalg.norm(sphere_data - center, axis=1)
            max_distance = distances.max() if len(distances) else 0.0

            # compute the radius based on max distance and threshold
            radius = np.float32(max_distance + self.distance_threshold)

            # append center, radius and number of vectors to our lists
            sphere_centers.append(center)
            sphere_radiuses.append(radius)
            sphere_counts.append(len(rectangle_indices))

        # initialize arrays with our final spheres properties
        self.all_spheres = np.asarray(sphere_centers, dtype=np.float32)
        self.all_radiuses = np.asarray(sphere_radiuses, dtype=np.float32)
        self.sphere_counts = np.asarray(sphere_counts, dtype=np.int32)
        self._constructed = True
        # print(f"Number of spheres found: {self.all_spheres.shape[0]}, max_spheres: {self.max_spheres}")


    # method to query our filter and check if query vectors fall inside our spheres
    def query(self, vectors: np.ndarray) -> List[bool]:
        if not self._constructed:
            raise ValueError("KDBallFilter has not been constructed.")

        # if no spheres exist return results
        if len(self.all_spheres) == 0:
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
            # find the closest sphere index and calculate vector distance from it
            sphere_distances = np.linalg.norm(self.all_spheres - q, axis=1)
            best_sphere_idx = int(np.argmin(sphere_distances))
            best_sphere_dist = sphere_distances[best_sphere_idx]

            # define distance limit to consider a vector close enough to an existing sphere
            distance_limit = self.all_radiuses[best_sphere_idx] + (2.0 * self.distance_threshold)

            # if vector is inside the existing sphere, we increment sphere counter
            if best_sphere_dist <= self.all_radiuses[best_sphere_idx]:
                self.sphere_counts[best_sphere_idx] += 1

            # if vector is close enough to existing sphere or we reached max spheres we expand it
            # to include new vector and maintain coverage of existing vectors in sphere
            elif best_sphere_dist <= distance_limit or len(self.all_spheres) >= self.max_spheres:
                # get sphere center, radius and count
                c_old = self.all_spheres[best_sphere_idx]
                r_old = self.all_radiuses[best_sphere_idx]
                count = self.sphere_counts[best_sphere_idx]

                # compute the new center including new vector with our current count
                c_new = ((c_old * count) + q) / (count + 1)

                # calculate the distance from new center to old center and new vector
                dist_to_c_old = float(np.linalg.norm(c_old - c_new))
                dist_to_q = float(np.linalg.norm(q - c_new))

                # choose our new radius that cover all existing vectors including new one
                # r_new >= max(r_old + ||c_old - c_new||, ||q - c_new|| + threshold)
                r_new = max(r_old + dist_to_c_old, dist_to_q + self.distance_threshold)

                # update our center and radius in existing sphere
                self.all_spheres[best_sphere_idx] = c_new
                self.all_radiuses[best_sphere_idx] = r_new
                self.sphere_counts[best_sphere_idx] += 1

            # else vector is far from our nearest sphere, create a new sphere because we have enough memory
            else:
                self.all_spheres = np.vstack([self.all_spheres, q])
                self.all_radiuses = np.append(self.all_radiuses, np.float32(self.distance_threshold))
                self.sphere_counts = np.append(self.sphere_counts, np.int32(1))


    # method for inserting new vectors into filter
    def insert(self, vectors: np.ndarray) -> None:
        self.add(vectors)


    # method for removing vectors from filter
    def remove(self, vectors: np.ndarray) -> List[bool]:
        # if not initialized or vectors array is empty we return results
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

        # find the closest sphere index and calculate vector distance from it
        best_sphere_idx = np.argmin(sq_distances, axis=1)
        best_sphere_dist = sq_distances[np.arange(len(q)), best_sphere_idx]

        # vectorized comparison against squared radiuses
        in_any_sphere = best_sphere_dist <= (self.all_radiuses[best_sphere_idx] ** 2)

        # means we have vectors to remove from spheres
        if np.any(in_any_sphere):
            # update count of all sphere using bincount
            self.sphere_counts -= np.bincount(best_sphere_idx[in_any_sphere], minlength=len(self.sphere_counts))

            # remove empty spheres from our memory if their count reached zero
            non_empty_spheres = self.sphere_counts > 0
            if not np.all(non_empty_spheres):
                self.all_spheres = self.all_spheres[non_empty_spheres]
                self.all_radiuses = self.all_radiuses[non_empty_spheres]
                self.sphere_counts = self.sphere_counts[non_empty_spheres]

        return in_any_sphere.tolist()


    # method to get exact fpr
    def get_exact_matching_fpr(self) -> float:
        return 0.0


    # method to get bytesize of overall filter
    def get_bytesize(self) -> int:
        if not self._constructed:
            return 0
        # return the overall spheres size including size of radiuses
        return self.all_spheres.nbytes + self.all_radiuses.nbytes + self.sphere_counts.nbytes