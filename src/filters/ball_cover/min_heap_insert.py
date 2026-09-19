import sys, heapq
import numpy as np
from typing import List, Tuple
from filters.ball_cover.ball_interfaces import InsertInterface, Filter


class MinHeapInsert(InsertInterface):
    _MAX_NEIGHBORS = 50
    _COMPACTION_RATIO = 2
    _BATCH_SIZE = 2048
    _COMPACT_THRESHOLD_MULTIPLIER = 4  # see _compact_if_needed

    def __init__(self):
        self._sphere_versions = np.zeros(0, dtype=np.int32)
        self._minimal_pairs: List[Tuple[float, int, int, int, int]] = []


    def ensure_capacity(self, required_cap: int) -> None:
        current_cap = len(self._sphere_versions)
        if required_cap <= current_cap:
            return

        new_versions = np.zeros(required_cap, dtype=np.int32)
        new_versions[:current_cap] = self._sphere_versions[:current_cap]
        self._sphere_versions = new_versions


    def reset_buffers(self, capacity: int) -> None:
        self.ensure_capacity(capacity)
        self._minimal_pairs = []
        self._sphere_versions.fill(0)


    @staticmethod
    def _compute_squared_distances(a: np.ndarray, a_sq: np.ndarray, b: np.ndarray, b_sq: np.ndarray) -> np.ndarray:
        a_b_cross = a @ b.T
        sq_distances = a_sq[:, None] - 2.0 * a_b_cross + b_sq[None, :]
        return np.maximum(sq_distances, 0.0)


    def _remove_invalid_pairs(self, filter: Filter) -> None:
        if len(self._minimal_pairs) > self._COMPACTION_RATIO * filter.max_spheres * self._MAX_NEIGHBORS:
            versions = self._sphere_versions
            dtype = [('vol', 'f8'), ('i', 'i4'), ('ver_i', 'i4'), ('j', 'i4'), ('ver_j', 'i4')]
            pairs = np.asarray(self._minimal_pairs, dtype=dtype)

            valid_pairs = (versions[pairs['i']] == pairs['ver_i']) & (versions[pairs['j']] == pairs['ver_j'])
            self._minimal_pairs = pairs[valid_pairs].tolist()
            heapq.heapify(self._minimal_pairs)


    def add_sphere(self, filter: Filter, center: np.ndarray, radius: float) -> None:
        current_spheres = filter.all_spheres

        if filter._num_spheres > 0:
            current_centers = filter._all_centers[current_spheres]
            current_radiuses = filter._all_radiuses[current_spheres]
            center_sq = np.einsum('ij,ij->i', center[None, :], center[None, :])
            current_centers_sq = np.einsum('ij,ij->i', current_centers, current_centers)

            sq_center_distances = self._compute_squared_distances(
                center[None, :], center_sq, current_centers, current_centers_sq)[0]
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

            if distance_ij > 0.0:
                weight_j = ((radius_j - radius_i) / (2.0 * distance_ij)) + 0.5
                weight_i = ((radius_i - radius_j) / (2.0 * distance_ij)) + 0.5
                new_center = weight_j * center_j + weight_i * center_i
                new_radius = max((distance_ij + radius_i + radius_j) / 2.0, 1e-6)
            else:
                new_center = center_i
                new_radius = max(radius_i, radius_j)

            current_spheres = filter.all_spheres[(filter.all_spheres != i) & (filter.all_spheres != j)]
            filter._all_spheres[:filter._num_spheres] = np.concatenate([current_spheres, [i, j]])
            self._sphere_versions[[i, j]] += 1
            filter._num_spheres -= 2

            self.add_sphere(filter, new_center, new_radius)


    def _compact_if_needed(self, filter: Filter) -> None:
        """Lazy deletion (a version bump marks a heap entry stale) avoids an
        O(n) removal per merge, but without a periodic sweep stale entries
        accumulate unboundedly (add_sphere pushes up to _MAX_NEIGHBORS per
        call, _merge_spheres pops one) -- a real unbounded-memory issue, not
        just wasted heap space. Rebuild once past the live footprint."""
        live_bound = max(
            64, filter.max_spheres * self._MAX_NEIGHBORS * self._COMPACT_THRESHOLD_MULTIPLIER
        )
        if len(self._minimal_pairs) <= live_bound:
            return
        alive = [
            entry for entry in self._minimal_pairs
            if self._sphere_versions[entry[1]] == entry[2] and self._sphere_versions[entry[3]] == entry[4]
        ]
        heapq.heapify(alive)
        self._minimal_pairs = alive


    def insert(self, filter: Filter, vectors: np.ndarray) -> None:
        for i in range(0, len(vectors), self._BATCH_SIZE):
            vector_batch = vectors[i:i + self._BATCH_SIZE]
            vector_batch = np.ascontiguousarray(vector_batch, dtype=np.float32)
            sq_vector_batch = np.einsum('ij,ij->i', vector_batch, vector_batch)

            if filter._num_spheres > 0:
                sq_centers = np.einsum('ij,ij->i', filter.all_centers, filter.all_centers)
                sq_batch_distances = self._compute_squared_distances(vector_batch, sq_vector_batch, filter.all_centers, sq_centers)
                batch_distances = np.sqrt(np.maximum(sq_batch_distances, 0.0))

                is_already_covered = np.any(batch_distances <= (filter.all_radiuses[None, :]), axis=1)
                vector_batch = vector_batch[~is_already_covered]

            for vec in vector_batch:
                self.add_sphere(filter, vec, 0.0)

            self._merge_spheres(filter)
            self._remove_invalid_pairs(filter)


    def remove(self, filter: Filter, vectors: np.ndarray) -> List[bool]:
        raise NotImplementedError("MinHeapInsert does not support remove operation.")


    def get_bytesize(self, filter: Filter) -> int:
        pair_size = 216  # PyTuple (80B) + PyFloat (24B) + 4 * PyInt (28B)
        # _sphere_versions is pre-allocated to filter.max_spheres (the budget's
        # theoretical capacity), which can hugely exceed actual usage for a
        # compact quantizer -- report bytes for slots actually in use instead.
        active_version_bytes = filter._num_spheres * self._sphere_versions.itemsize
        return active_version_bytes + sys.getsizeof(self._minimal_pairs) + pair_size * len(self._minimal_pairs)