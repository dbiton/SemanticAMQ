import sys, heapq
import numpy as np
from typing import List, Tuple, Optional
from filters.filter import Filter
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def box_query(queries: np.ndarray, boxes_min: np.ndarray, boxes_max: np.ndarray) -> np.ndarray:
    N = queries.shape[0]
    K = boxes_min.shape[0]
    D = boxes_min.shape[1]

    results = np.zeros(N, dtype=np.bool_)

    for i in prange(N):
        for j in range(K):
            in_this_box = True

            for k in range(D):
                if queries[i, k] < boxes_min[j, k] or queries[i, k] > boxes_max[j, k]:
                    in_this_box = False
                    break 

            if in_this_box:
                results[i] = True
                break
     
    return results


class GonzalezBoxFilter(Filter):
    """
    A high-performance spatial filter that partitions high-dimensional vector data into axis-aligned bounding boxes.

    The GonzalezBoxFilter combines greedy spatial clustering with hyperrectangular envelope fitting to efficiently 
    approximate complex vector distributions using a bounded collection of axis-aligned boxes.

    The partitioning strategy leverages:
    - Gonzalez's algorithm for initial greedy k-center clustering to efficiently partition vectors into 
      cohesive spatial clusters.
    - Axis-aligned spatial bounding, calculating tight per-dimension minimum and maximum coordinates 
      to encapsulate each vector cluster into a hyperrectangle.

    The filter is designed with a strict one-sided error guarantee: it allows false positives 
    for fast pruning, but guarantees zero false negatives (no FN), ensuring that any matching 
    vector is never incorrectly excluded.

    Attributes:
        dim (int): The dimensionality of the vector space.
        distance_threshold (float): Additional padding (margin of error) added to the minimum and maximum 
                                    bounds of each box to ensure boundary vectors are fully enclosed.
        max_memory (Optional[int]): The maximum memory limit in bytes allocated for storing boxes.
        max_boxes (int): The maximum number of space partitions (boxes) allowed, automatically
                         calculated based on the memory limit to prevent overflow.
        all_boxes (np.ndarray): A 1D array of shape (K,) containing the active indices of the boxes.
        all_boxes_min (np.ndarray): A 2D array of shape (K, D) storing the minimum coordinate bounds of all boxes.
        all_boxes_max (np.ndarray): A 2D array of shape (K, D) storing the maximum coordinate bounds of all boxes.
    """
    _BYTES_PER_FLOAT32 = 4
    _BYTES_PER_FLOAT64 = 8
    _MAX_NEIGHBORS = 50
    _BATCH_SIZE = 2048

    def __init__(self, dim: int, distance_threshold: float, max_memory: Optional[int] = None, max_spheres: Optional[int] = None):
        super().__init__(dim, max_memory=max_memory)
        self.distance_threshold = distance_threshold
        self.max_boxes = (max_spheres if max_spheres is not None 
                    else self._calculate_max_boxes(self.max_memory))
        self._num_boxes = 0
        self._all_boxes = np.arange(self.max_boxes, dtype=np.int32)
        self._all_boxes_min = np.empty((self.max_boxes, dim), dtype=np.float32)
        self._all_boxes_max = np.empty((self.max_boxes, dim), dtype=np.float32)
        self._indices_count = np.zeros((self.max_boxes,), dtype=np.int32)
        self._minimal_pairs: List[Tuple[float, int, int, int, int]] = []
        self._constructed = False

    @property
    def all_boxes(self) -> np.ndarray:
        return self._all_boxes[:self._num_boxes]

    @property
    def all_boxes_min(self) -> np.ndarray:
        return self._all_boxes_min[self.all_boxes]

    @property
    def all_boxes_max(self) -> np.ndarray:
        return self._all_boxes_max[self.all_boxes]


    def _calculate_max_boxes(self, max_memory: Optional[int], gamma: float = 4.0) -> int:
        bytes_per_box = (self.dim * 2) * self._BYTES_PER_FLOAT32
        return max(1, int(max_memory // (bytes_per_box * gamma)))


    def _get_new_index(self) -> int:
        if self._num_boxes == len(self._all_boxes_min):
            old_cap = len(self._all_boxes_min)
            new_cap = old_cap * 2

            new_boxes = np.empty((new_cap,), dtype=np.int32)
            new_boxes[:old_cap] = self._all_boxes
            new_boxes[old_cap:] = np.arange(old_cap, new_cap, dtype=np.int32)
            self._all_boxes = new_boxes

            new_min = np.empty((new_cap, self.dim), dtype=np.float32)
            new_min[:old_cap] = self._all_boxes_min
            self._all_boxes_min = new_min

            new_max = np.empty((new_cap, self.dim), dtype=np.float32)
            new_max[:old_cap] = self._all_boxes_max
            self._all_boxes_max = new_max

            new_indices_count = np.zeros(new_cap, dtype=np.int32)
            new_indices_count[:old_cap] = self._indices_count
            self._indices_count = new_indices_count

        return self._all_boxes[self._num_boxes]


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


    def _remove_invalid_pairs(self) -> None:
        counts = self._indices_count
        self._minimal_pairs = [p for p in self._minimal_pairs 
                if counts[p[1]] == p[2] and counts[p[3]] == p[4]]
        heapq.heapify(self._minimal_pairs)


    def _add_box(self, box_min: np.ndarray, box_max: np.ndarray) -> None:
        current_boxes = self.all_boxes

        if self._num_boxes > 0:
            current_mins = self._all_boxes_min[current_boxes]
            current_maxs = self._all_boxes_max[current_boxes]

            is_contained = np.all((current_mins <= box_min[None, :]) & (current_maxs >= box_max[None, :]), axis=1)
            if np.any(is_contained):
                return

            contained_indices = np.all((box_min[None, :] <= current_mins) & (box_max[None, :] >= current_maxs), axis=1)
            if np.any(contained_indices):
                contained_boxes = current_boxes[contained_indices]
                current_boxes = current_boxes[~contained_indices]
                self._all_boxes[: self._num_boxes] = np.concatenate([current_boxes, contained_boxes])
                self._num_boxes -= len(contained_boxes)
                self._indices_count[contained_boxes] += 1
                current_mins = current_mins[~contained_indices]
                current_maxs = current_maxs[~contained_indices]

        box_idx = self._get_new_index()
        box_count = self._indices_count[box_idx]
        self._all_boxes_min[box_idx] = box_min
        self._all_boxes_max[box_idx] = box_max
        self._num_boxes += 1

        if len(current_boxes) > 0:
            merged_mins = np.minimum(box_min[None, :], current_mins)
            merged_maxs = np.maximum(box_max[None, :], current_maxs)

            box_sides = np.maximum(box_max - box_min, 1e-6)
            current_sides = np.maximum(current_maxs - current_mins, 1e-6)
            merged_sides = np.maximum(merged_maxs - merged_mins, 1e-6)

            # log(V_merged - (V_1 + V_2)) = sum(log(L_merged)) + log(1 - (prod(L_1 / L_merged) + prod(L_2 / L_merged)))
            merged_volumes = np.sum(np.log(merged_sides), axis=1) + np.log(np.maximum(1.0 - (np.prod(box_sides / merged_sides, axis=1) + np.prod(current_sides / merged_sides, axis=1)), 1e-6))

            max_neighbors = min(self._MAX_NEIGHBORS, len(current_boxes))
            if max_neighbors == len(current_boxes):
                neighbors_indices = np.arange(len(current_boxes))
            else:
                neighbors_indices = np.argpartition(merged_volumes, max_neighbors - 1)[:max_neighbors]

            best_neighbors = current_boxes[neighbors_indices]
            best_merged_volumes = merged_volumes[neighbors_indices]

            for volume, neighbor_idx in zip(best_merged_volumes, best_neighbors):
                neighbor_count = self._indices_count[neighbor_idx]
                heapq.heappush(self._minimal_pairs, (volume, box_idx, box_count, neighbor_idx, neighbor_count))


    def _merge_boxes(self) -> None:
        while self._num_boxes > self.max_boxes and self._minimal_pairs:
            vol, i, i_count, j, j_count = heapq.heappop(self._minimal_pairs)

            if self._indices_count[i] != i_count or self._indices_count[j] != j_count:
                continue

            i_min, i_max = self._all_boxes_min[i], self._all_boxes_max[i]
            j_min, j_max = self._all_boxes_min[j], self._all_boxes_max[j]
            new_min, new_max = np.minimum(i_min, j_min), np.maximum(i_max, j_max)

            current_boxes = self.all_boxes[(self.all_boxes != i) & (self.all_boxes != j)]
            self._all_boxes[:self._num_boxes] = np.concatenate([current_boxes, [i, j]])
            self._indices_count[[i, j]] += 1
            self._num_boxes -= 2

            self._add_box(new_min, new_max)

        if len(self._minimal_pairs) > 2 * self.max_boxes * self._MAX_NEIGHBORS:
            self._remove_invalid_pairs()


    def construct(self, vectors: np.ndarray) -> Tuple[float, float]:
        if len(vectors) == 0 or self.max_boxes == 0:
            self._constructed = True
            return

        vectors = np.ascontiguousarray(vectors, dtype=np.float32)

        if self.max_boxes >= len(vectors):
            for vec in vectors:
                box_min = vec - self.distance_threshold
                box_max = vec + self.distance_threshold
                self._add_box(box_min, box_max)
            self._constructed = True
            return

        sq_norms = np.sum(vectors * vectors, axis=1)
        assignment = self._gonzalez_clustering(vectors, sq_norms, self.max_boxes)

        num_of_clusters = int(assignment.max()) + 1
        clusters = [[] for _ in range(num_of_clusters)]
        for idx, cluster_id in enumerate(assignment):
            clusters[cluster_id].append(idx)

        for cluster_indices in clusters:
            if not cluster_indices:
                continue

            cluster_vectors = vectors[cluster_indices]
            box_min = cluster_vectors.min(axis=0) - self.distance_threshold
            box_max = cluster_vectors.max(axis=0) + self.distance_threshold
            self._add_box(box_min, box_max)

        self._constructed = True


    def query(self, vectors: np.ndarray) -> List[bool]:
        if not self._constructed:
            raise ValueError("GonzalezBoxFilter has not been constructed.")

        if len(self.all_boxes_min) == 0 or len(vectors) == 0:
            return [False] * len(vectors)

        q = np.ascontiguousarray(vectors, dtype=np.float32)
        in_any_box = box_query(q, self.all_boxes_min, self.all_boxes_max)

        return in_any_box.tolist()


    def add(self, vectors: np.ndarray) -> None:
        if not self._constructed:
            self.construct(vectors)
            return

        if len(vectors) == 0:
            return

        for i in range(0, len(vectors), self._BATCH_SIZE):
            vector_batch = vectors[i : i + self._BATCH_SIZE]
            vector_batch = np.ascontiguousarray(vector_batch, dtype=np.float32)

            if self._num_boxes > 0:
                in_each_box = np.all((vector_batch[:, None, :] >= self.all_boxes_min[None, :, :] + self.distance_threshold) & 
                                     (vector_batch[:, None, :] <= self.all_boxes_max[None, :, :] - self.distance_threshold), axis=2)
                is_already_covered = np.any(in_each_box, axis=1)
                vector_batch = vector_batch[~is_already_covered]

            for vec in vector_batch:
                box_min = vec - self.distance_threshold
                box_max = vec + self.distance_threshold
                self._add_box(box_min, box_max)

            self._merge_boxes()


    def insert(self, vectors: np.ndarray) -> None:
        self.add(vectors)


    def remove(self, vectors: np.ndarray) -> List[bool]:
        raise NotImplementedError("GonzalezBoxFilter does not support remove operation.")


    def get_exact_matching_fpr(self) -> float:
        return 0.0


    def get_bytesize(self) -> int:
        if not self._constructed:
            return 0
        return (self._all_boxes.nbytes + self._all_boxes_min.nbytes + self._all_boxes_max.nbytes + 
                self._indices_count.nbytes + sys.getsizeof(self._minimal_pairs))