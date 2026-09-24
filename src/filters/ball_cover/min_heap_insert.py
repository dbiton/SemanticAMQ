import sys, heapq
import numpy as np
from typing import List, Tuple
from filters.ball_cover.ball_interfaces import InsertInterface, Filter


class MinHeapInsert(InsertInterface):
    _MAX_NEIGHBORS = 10
    _COMPACTION_RATIO = 2
    _BATCH_SIZE = 2048

    def __init__(self):
        self._sphere_versions = np.zeros(0, dtype=np.int32)
        self._minimal_pairs: List[Tuple[float, int, int, int, int]] = []


    @staticmethod
    def _compute_squared_distances(a: np.ndarray, a_sq: np.ndarray, b: np.ndarray, b_sq: np.ndarray) -> np.ndarray:
        a_b_cross = a @ b.T
        sq_distances = a_sq[:, None] - 2.0 * a_b_cross + b_sq[None, :]
        return np.maximum(sq_distances, 0.0)


    @staticmethod
    def _compute_merged_radius(dim: int, radius_a: np.ndarray, radius_b: np.ndarray, center_distances: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        radius_a = np.asarray(radius_a, dtype=np.float64)
        radius_b = np.asarray(radius_b, dtype=np.float64)
        center_distances = np.asarray(center_distances, dtype=np.float64)

        # R_merged = max((R_1 + dist_1_2 + R_2) / 2, R_1, R_2)
        merged_radius = np.maximum(np.maximum((radius_a + center_distances + radius_b) / 2.0, np.maximum(radius_a, radius_b)), 1e-6)

        # log(R_merged^D - (R_1^D + R_2^D)) = D*log(R_merged) + log(1 - ((R_1/R_merged)^D + (R_2/R_merged)^D))
        merged_volume = dim * np.log(merged_radius) + np.log(np.maximum((1.0 - (radius_a / merged_radius) ** dim - (radius_b / merged_radius) ** dim), 1e-6))

        return merged_radius.astype(np.float32), merged_volume


    def initialize_buffers(self, filter: Filter, spheres_indices: np.ndarray, centers: np.ndarray, radii: np.ndarray) -> None:
        if len(centers) < 2:
            return

        centers_sq = np.einsum('ij,ij->i', centers, centers)
        max_neighbors = min(self._MAX_NEIGHBORS, len(centers) - 1)

        for i in range(0, len(centers), self._BATCH_SIZE):
            batch_centers = centers[i:i + self._BATCH_SIZE]
            batch_centers_sq = centers_sq[i:i + self._BATCH_SIZE]

            batch_sq_distances = self._compute_squared_distances(batch_centers, batch_centers_sq, centers, centers_sq)
            batch_distances = np.sqrt(np.maximum(batch_sq_distances, 0.0))
            np.fill_diagonal(batch_distances[:, i:i + len(batch_centers)], np.inf)
            neighbors_indices = np.argpartition(batch_distances, max_neighbors - 1, axis=1)[:, :max_neighbors]

            for j in range(len(batch_centers)):
                sphere_idx = spheres_indices[i + j]
                sphere_radius = radii[i + j]
                sphere_version = self._sphere_versions[sphere_idx]

                current_neighbors = neighbors_indices[j]
                closest_neighbors = spheres_indices[current_neighbors]
                neighbors_center_dists = batch_distances[j, current_neighbors]
                neighbors_radiuses = radii[current_neighbors]
                merged_volumes = self._compute_merged_radius(filter.dim, sphere_radius, neighbors_radiuses, neighbors_center_dists)[1]

                for volume, neighbor_idx in zip(merged_volumes, closest_neighbors):
                    neighbor_version = self._sphere_versions[neighbor_idx]
                    self._minimal_pairs.append((float(volume), int(sphere_idx), int(sphere_version), int(neighbor_idx), int(neighbor_version)))

        heapq.heapify(self._minimal_pairs)


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

            center_sq_distances = self._compute_squared_distances(
                center[None, :], center_sq, current_centers, current_centers_sq)[0]
            center_distances = np.sqrt(np.maximum(center_sq_distances, 0.0))

            if np.any(center_distances + radius <= current_radiuses):
                return

            contained_indices = (center_distances + current_radiuses) <= radius
            if np.any(contained_indices):
                contained_spheres = current_spheres[contained_indices]
                current_spheres = current_spheres[~contained_indices]
                filter._remove_spheres(contained_spheres)
                self._sphere_versions[contained_spheres] += 1
                current_radiuses = current_radiuses[~contained_indices]
                center_distances = center_distances[~contained_indices]

        sphere_idx = filter._add_spheres(center[None, :], radius)[0]
        sphere_version = self._sphere_versions[sphere_idx]

        if len(current_spheres) > 0:
            max_neighbors = min(self._MAX_NEIGHBORS, len(current_spheres))
            neighbors_indices = np.argpartition(center_distances, max_neighbors - 1)[:max_neighbors]

            closest_neighbors = current_spheres[neighbors_indices]
            neighbors_center_dists = center_distances[neighbors_indices]
            neighbors_radiuses = current_radiuses[neighbors_indices]
            merged_volumes = self._compute_merged_radius(filter.dim, radius, neighbors_radiuses, neighbors_center_dists)[1]

            for volume, neighbor_idx in zip(merged_volumes, closest_neighbors):
                neighbor_version = self._sphere_versions[neighbor_idx]
                heapq.heappush(self._minimal_pairs, (float(volume), int(sphere_idx), int(sphere_version), int(neighbor_idx), int(neighbor_version)))


    def _merge_spheres(self, filter: Filter) -> None:
        while filter._num_spheres > filter.max_spheres and self._minimal_pairs:
            vol, i, i_version, j, j_version = heapq.heappop(self._minimal_pairs)

            if self._sphere_versions[i] != i_version or self._sphere_versions[j] != j_version:
                continue

            center_i, radius_i = filter._all_centers[i], filter._all_radiuses[i]
            center_j, radius_j = filter._all_centers[j], filter._all_radiuses[j]
            centers_diff = center_i - center_j
            distance_ij = float(np.sqrt(np.dot(centers_diff, centers_diff)))

            if distance_ij + radius_j <= radius_i:
                new_center = center_i.copy()
                new_radius = float(radius_i)
            elif distance_ij + radius_i <= radius_j:
                new_center = center_j.copy()
                new_radius = float(radius_j)
            else:
                weight_j = ((radius_j - radius_i) / (2.0 * distance_ij)) + 0.5
                weight_i = ((radius_i - radius_j) / (2.0 * distance_ij)) + 0.5
                new_center = weight_j * center_j + weight_i * center_i
                new_radius = float(self._compute_merged_radius(filter.dim, radius_i, radius_j, distance_ij)[0])

            filter._remove_spheres(np.array([i, j]))
            self._sphere_versions[[i, j]] += 1

            self.add_sphere(filter, new_center, new_radius)


    def insert(self, filter: Filter, vectors: np.ndarray) -> None:
        for i in range(0, len(vectors), self._BATCH_SIZE):
            vector_batch = vectors[i:i + self._BATCH_SIZE]
            vector_batch = np.ascontiguousarray(vector_batch, dtype=np.float32)
            vector_batch_sq = np.einsum('ij,ij->i', vector_batch, vector_batch)

            if filter._num_spheres > 0:
                centers_sq = np.einsum('ij,ij->i', filter.all_centers, filter.all_centers)
                batch_sq_distances = self._compute_squared_distances(vector_batch, vector_batch_sq, filter.all_centers, centers_sq)
                batch_distances = np.sqrt(np.maximum(batch_sq_distances, 0.0))

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
        return self._sphere_versions.nbytes + sys.getsizeof(self._minimal_pairs) + pair_size * len(self._minimal_pairs)