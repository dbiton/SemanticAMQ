import sys
import heapq
import numpy as np
from typing import List, Tuple
from filters.ball_cover.ball_interfaces import InsertInterface, Filter


class MinHeapInsert(InsertInterface):
    _MAX_NEIGHBORS = 50
    _BATCH_SIZE = 2048

    def __init__(self):
        # streaming merge data structures, minimal pair heap and sphere version array
        self._sphere_versions = np.zeros(0, dtype=np.int32)
        self._minimal_pairs: List[Tuple[float, int, int, int, int]] = []

    # ------------------------------------------------------------------ sizing / buffers

    def reset_buffers(self, capacity: int) -> None:
        self.ensure_capacity(capacity)
        self._minimal_pairs = []
        self._sphere_versions.fill(0)

    def ensure_capacity(self, required_cap: int) -> None:
        current_cap = len(self._sphere_versions)
        if required_cap <= current_cap:
            return

        new_versions = np.zeros(required_cap, dtype=np.int32)
        new_versions[:current_cap] = self._sphere_versions[:current_cap]
        self._sphere_versions = new_versions

    # ------------------------------------------------------------------ streaming / dynamic merge

    def add_sphere(self, filter: Filter, center: np.ndarray, radius: float) -> None:
        current_spheres = filter.all_spheres

        if filter._num_spheres > 0:
            current_centers = filter._all_centers[current_spheres]
            current_radiuses = filter._all_radiuses[current_spheres]
            center_sq = np.einsum('ij,ij->i', center[None, :], center[None, :], dtype=np.float64)

            sq_center_distances = filter._sq_dists_to_centers(
                center[None, :], center_sq, current_centers)[0]
            center_distances = np.sqrt(np.maximum(sq_center_distances, 0.0))

            if np.any(center_distances + radius <= current_radiuses):
                return

            contained_indices = (center_distances + current_radiuses) <= radius
            if np.any(contained_indices):
                contained_spheres = current_spheres[contained_indices]
                current_spheres = current_spheres[~contained_indices]
                filter._all_spheres[:filter._num_spheres] = np.concatenate([current_spheres, contained_spheres])
                filter._num_spheres -= len(contained_spheres)
                self._sphere_versions[contained_spheres] += 1
                current_radiuses = current_radiuses[~contained_indices]
                center_distances = center_distances[~contained_indices]

        sphere_idx = filter._get_new_index()
        sphere_version = self._sphere_versions[sphere_idx]
        filter._all_centers[sphere_idx] = center
        filter._all_radiuses[sphere_idx] = radius
        filter._num_spheres += 1

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
            merged_volumes = filter.dim * np.log(merged_radius) + np.log(np.maximum(1.0 - ((radius / merged_radius) ** filter.dim + (neighbors_radiuses / merged_radius) ** filter.dim), 1e-6))

            for volume, neighbor_idx in zip(merged_volumes, closest_neighbors):
                neighbor_version = self._sphere_versions[neighbor_idx]
                heapq.heappush(self._minimal_pairs, (volume, sphere_idx, sphere_version, neighbor_idx, neighbor_version))

    def _merge_spheres(self, filter: Filter) -> None:
        while filter._num_spheres > filter.max_spheres and self._minimal_pairs:
            vol, i, i_version, j, j_version = heapq.heappop(self._minimal_pairs)

            if self._sphere_versions[i] != i_version or self._sphere_versions[j] != j_version:
                continue

            center_i, radius_i = filter._all_centers[i], filter._all_radiuses[i]
            center_j, radius_j = filter._all_centers[j], filter._all_radiuses[j]

            centers_diff = center_i - center_j
            distance_ij = float(np.sqrt(np.dot(centers_diff, centers_diff)))
            new_radius = (distance_ij + radius_i + radius_j) / 2.0

            if distance_ij > 0.0:
                weight_j = ((radius_j - radius_i) / (2.0 * distance_ij)) + 0.5
                weight_i = ((radius_i - radius_j) / (2.0 * distance_ij)) + 0.5
                new_center = weight_j * center_j + weight_i * center_i
            else:
                new_center = center_i

            current_spheres = filter.all_spheres[(filter.all_spheres != i) & (filter.all_spheres != j)]
            filter._all_spheres[:filter._num_spheres] = np.concatenate([current_spheres, [i, j]])
            self._sphere_versions[[i, j]] += 1
            filter._num_spheres -= 2

            self.add_sphere(filter, new_center.astype(np.float32), new_radius)

    def add(self, filter: Filter, vectors: np.ndarray) -> None:
        if not filter._constructed:
            filter.construct(vectors)
            return

        if len(vectors) == 0 or filter.max_spheres == 0:
            return

        for i in range(0, len(vectors), self._BATCH_SIZE):
            vector_batch = vectors[i:i + self._BATCH_SIZE]
            vector_batch = np.ascontiguousarray(vector_batch, dtype=np.float32)
            sq_vector_batch = np.einsum('ij,ij->i', vector_batch, vector_batch, dtype=np.float64)

            if filter._num_spheres > 0:
                sq_batch_distances = filter._sq_dists_to_centers(vector_batch, sq_vector_batch, filter.all_centers)
                batch_distances = np.sqrt(np.maximum(sq_batch_distances, 0.0))

                is_already_covered = np.any(batch_distances <= (filter.all_radiuses[None, :] - filter.distance_threshold), axis=1)
                vector_batch = vector_batch[~is_already_covered]

            for vec in vector_batch:
                self.add_sphere(filter, vec, filter.distance_threshold)

            self._merge_spheres(filter)

    def insert(self, filter: Filter, vectors: np.ndarray) -> None:
        self.add(filter, vectors)

    def remove(self, filter: Filter, vectors: np.ndarray) -> List[bool]:
        raise NotImplementedError("MinHeapInsert does not support remove operation.")

    # ------------------------------------------------------------------ memory / misc

    def get_bytesize(self) -> int:
        return self._sphere_versions.nbytes + sys.getsizeof(self._minimal_pairs)