import numpy as np
from typing import List, Optional, Tuple
from filters.ball_cover.ball_interfaces import InsertInterface, Filter


class MinHeapInsert(InsertInterface):
    """A not-already-covered vector becomes its own zero-radius ball; past
    capacity, the cheapest candidate pair of balls is merged into their
    enclosing ball, cheapest by the log volume the merge adds
    (_compute_merged_radius). Each ball proposes its _MAX_NEIGHBORS nearest
    balls by center distance when it is added, so candidates are local and a
    merge costs O(k * d) rather than ExhaustiveMergeInsert's k x k matrix.

    Candidate pairs live in three flat numpy arrays -- the cost (float32) and
    the two sphere ids (uint16 while ids fit, else uint32) -- 8 B per pair,
    where a heap of Python tuples took 216 B; the cheapest is one argmin. A
    pair is dropped the moment either of its balls is, so every stored pair
    is live and each live ball owns at most _MAX_NEIGHBORS of them. Vectors
    are merged in one at a time, so at most max_spheres + 1 balls are ever
    live, and the arrays are allocated once at _MAX_NEIGHBORS *
    (max_spheres + 1) slots: that allocation is what get_bytesize() charges
    and max_spheres_for_budget() pays for in balls."""
    _MAX_NEIGHBORS = 10
    _BATCH_SIZE = 2048

    def __init__(self):
        self._cost = np.empty(0, dtype=np.float32)
        self._lo = np.empty(0, dtype=np.uint16)
        self._hi = np.empty(0, dtype=np.uint16)
        self._n = 0


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


    @staticmethod
    def _id_dtype(n_ids: int) -> np.dtype:
        return np.dtype(np.uint16) if n_ids <= np.iinfo(np.uint16).max + 1 else np.dtype(np.uint32)


    def _pair_bytes(self, max_spheres: int) -> int:
        # Ids come from the filter's id pool, which grows to at most twice
        # max_spheres when the (max_spheres + 1)-th ball arrives.
        return np.dtype(np.float32).itemsize + 2 * self._id_dtype(2 * (max_spheres + 1)).itemsize


    def max_spheres_for_budget(self, usable_bytes: int, bytes_per_ball: int) -> int:
        n = self._MAX_NEIGHBORS

        def fits(k: int) -> bool:
            return k * bytes_per_ball + n * (k + 1) * self._pair_bytes(k) <= usable_bytes

        k = max(1, (usable_bytes - n * self._pair_bytes(1)) // (bytes_per_ball + n * self._pair_bytes(1)))
        while k > 1 and not fits(k):
            k -= 1
        while fits(k + 1):
            k += 1
        return k


    def _allocate(self, filter: Filter) -> None:
        slots = self._MAX_NEIGHBORS * (filter.max_spheres + 1)
        dtype = self._id_dtype(max(2 * (filter.max_spheres + 1), len(filter._all_radiuses)))
        if len(self._cost) != slots or self._lo.dtype != dtype:
            self._cost = np.empty(slots, dtype=np.float32)
            self._lo = np.empty(slots, dtype=dtype)
            self._hi = np.empty(slots, dtype=dtype)
        self._n = 0


    def _push(self, lo: np.ndarray, hi: np.ndarray, cost: np.ndarray) -> None:
        m = len(cost)
        if self._n + m > len(self._cost):
            # Unreachable while every pair is live (see the class docstring);
            # grown rather than dropped so a miscount can't lose candidates,
            # and get_bytesize() reports whatever is actually allocated.
            size = max(2 * len(self._cost), self._n + m)
            self._cost = np.resize(self._cost, size)
            self._lo = np.resize(self._lo, size)
            self._hi = np.resize(self._hi, size)
        end = self._n + m
        self._cost[self._n:end] = cost
        self._lo[self._n:end] = lo
        self._hi[self._n:end] = hi
        self._n = end


    def _drop(self, sphere_ids: np.ndarray) -> None:
        """Removes every pair that has one of `sphere_ids` as an endpoint."""
        n = self._n
        if n == 0 or len(sphere_ids) == 0:
            return
        ids = np.asarray(sphere_ids, dtype=self._lo.dtype)
        keep = ~(np.isin(self._lo[:n], ids) | np.isin(self._hi[:n], ids))
        kept = int(keep.sum())
        self._cost[:kept] = self._cost[:n][keep]
        self._lo[:kept] = self._lo[:n][keep]
        self._hi[:kept] = self._hi[:n][keep]
        self._n = kept


    def _pop_cheapest(self) -> Optional[Tuple[int, int]]:
        if self._n == 0:
            return None
        idx = int(np.argmin(self._cost[:self._n]))
        return int(self._lo[idx]), int(self._hi[idx])


    def initialize_buffers(self, filter: Filter, spheres_indices: np.ndarray, centers: np.ndarray, radii: np.ndarray,
                           vectors: Optional[np.ndarray] = None) -> None:
        self._allocate(filter)
        if len(centers) < 2:
            return

        centers = np.ascontiguousarray(centers, dtype=np.float32)
        radii = np.asarray(radii, dtype=np.float32)
        centers_sq = np.einsum('ij,ij->i', centers, centers)
        max_neighbors = min(self._MAX_NEIGHBORS, len(centers) - 1)

        for i in range(0, len(centers), self._BATCH_SIZE):
            batch_centers = centers[i:i + self._BATCH_SIZE]
            batch_centers_sq = centers_sq[i:i + self._BATCH_SIZE]

            batch_sq_distances = self._compute_squared_distances(batch_centers, batch_centers_sq, centers, centers_sq)
            batch_distances = np.sqrt(np.maximum(batch_sq_distances, 0.0))
            np.fill_diagonal(batch_distances[:, i:i + len(batch_centers)], np.inf)
            neighbors_indices = np.argpartition(batch_distances, max_neighbors - 1, axis=1)[:, :max_neighbors]

            rows = np.arange(len(batch_centers))[:, None]
            merged_volumes = self._compute_merged_radius(
                filter.dim, radii[i:i + len(batch_centers), None], radii[neighbors_indices],
                batch_distances[rows, neighbors_indices])[1]
            self._push(np.repeat(spheres_indices[i:i + len(batch_centers)], max_neighbors),
                       spheres_indices[neighbors_indices].ravel(), merged_volumes.ravel())


    def ensure_capacity(self, required_cap: int) -> None:
        # The filter's id pool grew: ids must still fit the id arrays.
        dtype = self._id_dtype(required_cap)
        if dtype.itemsize > self._lo.dtype.itemsize:
            self._lo = self._lo.astype(dtype)
            self._hi = self._hi.astype(dtype)


    def reset_buffers(self, capacity: int) -> None:
        self.ensure_capacity(capacity)
        self._n = 0


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
                self._drop(contained_spheres)
                current_radiuses = current_radiuses[~contained_indices]
                center_distances = center_distances[~contained_indices]

        sphere_idx = filter._add_spheres(center[None, :], radius)[0]

        if len(current_spheres) > 0:
            max_neighbors = min(self._MAX_NEIGHBORS, len(current_spheres))
            neighbors_indices = np.argpartition(center_distances, max_neighbors - 1)[:max_neighbors]

            closest_neighbors = current_spheres[neighbors_indices]
            neighbors_center_dists = center_distances[neighbors_indices]
            neighbors_radiuses = current_radiuses[neighbors_indices]
            merged_volumes = self._compute_merged_radius(filter.dim, radius, neighbors_radiuses, neighbors_center_dists)[1]
            self._push(np.full(max_neighbors, sphere_idx), closest_neighbors, merged_volumes)


    def _merge_spheres(self, filter: Filter) -> None:
        while filter._num_spheres > filter.max_spheres:
            pair = self._pop_cheapest()
            if pair is None:
                # Every candidate of every live ball has been merged away:
                # re-propose from scratch rather than stay over capacity.
                self.initialize_buffers(filter, filter.all_spheres, filter.all_centers, filter.all_radiuses)
                pair = self._pop_cheapest()
                if pair is None:
                    return
            i, j = pair

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
            self._drop(np.array([i, j]))

            self.add_sphere(filter, new_center, new_radius)


    def insert(self, filter: Filter, vectors: np.ndarray) -> None:
        if len(self._cost) == 0:
            self._allocate(filter)

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

            # One at a time -- genuinely sequential, since each merge changes
            # what the next vector finds -- which also keeps at most
            # max_spheres + 1 balls live. add_sphere re-tests coverage against
            # the balls as they are now.
            for vec in vector_batch:
                self.add_sphere(filter, vec, 0.0)
                self._merge_spheres(filter)


    def remove(self, filter: Filter, vectors: np.ndarray) -> List[bool]:
        raise NotImplementedError("MinHeapInsert does not support remove operation.")


    def get_bytesize(self, filter: Filter) -> int:
        return int(self._cost.nbytes + self._lo.nbytes + self._hi.nbytes)
