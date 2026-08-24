import numpy as np
from typing import List, Tuple, Optional
from filters.filter import Filter
from filters.ball_cover.ball_interfaces import ConstructInterface
from filters.ball_cover.ball_interfaces import InsertInterface
from hyperspheres_volume import estimate_hypersphere_volume
from filters.ball_cover.ablatable_ladder_construct import AblatableLadderConstruct
from filters.ball_cover.min_heap_insert import MinHeapInsert


class BallCoverFilter(Filter):
    _BYTES_PER_FLOAT32 = 4
    _BYTES_PER_FLOAT64 = 8

    def __init__(self, dim: int, distance_threshold: float,
                 construct_method: ConstructInterface = AblatableLadderConstruct(),
                 insert_method: InsertInterface = MinHeapInsert(),
                 max_memory: Optional[int] = None,
                 max_spheres: Optional[int] = None):
        super().__init__(dim, lsh=None, max_memory=max_memory)
        self.distance_threshold = distance_threshold
        self.max_spheres = (max_spheres if max_spheres is not None
                            else self._calculate_max_spheres(self.max_memory))
        self._constructed = False

        # main filter sphere buffers
        self._num_spheres = 0
        self._all_spheres = np.arange(self.max_spheres, dtype=np.int32)
        self._all_centers = np.empty((self.max_spheres, dim), dtype=np.float32)
        self._all_radiuses = np.empty((self.max_spheres,), dtype=np.float32)

        # main filter construct and insert strategies
        self.construct_method = construct_method
        self.insert_method = insert_method
        self.insert_method.ensure_capacity(self.max_spheres)

    # ------------------------------------------------------------------ properties

    @property
    def all_spheres(self) -> np.ndarray:
        return self._all_spheres[:self._num_spheres]

    @property
    def all_centers(self) -> np.ndarray:
        return self._all_centers[self.all_spheres]

    @property
    def all_radiuses(self) -> np.ndarray:
        return self._all_radiuses[self.all_spheres]

    # ------------------------------------------------------------------ sizing / buffers

    def _calculate_max_spheres(self, max_memory: int, gamma: float = 4.0) -> int:
        bytes_per_sphere = (self.dim + 2) * self._BYTES_PER_FLOAT32
        return max(1, int(max_memory // (bytes_per_sphere * gamma)))

    def _ensure_capacity(self, required_cap: int) -> None:
        current_cap = len(self._all_centers)
        if required_cap <= current_cap:
            return

        new_cap = max(current_cap * 2, required_cap)
        new_spheres = np.empty((new_cap,), dtype=np.int32)
        new_spheres[:current_cap] = self._all_spheres
        new_spheres[current_cap:] = np.arange(current_cap, new_cap, dtype=np.int32)
        self._all_spheres = new_spheres

        new_centers = np.empty((new_cap, self.dim), dtype=np.float32)
        new_centers[:current_cap] = self._all_centers
        self._all_centers = new_centers

        new_radiuses = np.empty((new_cap,), dtype=np.float32)
        new_radiuses[:current_cap] = self._all_radiuses
        self._all_radiuses = new_radiuses
        self.insert_method.ensure_capacity(new_cap)

    def _get_new_index(self) -> int:
        if self._num_spheres == len(self._all_centers):
            self._ensure_capacity(len(self._all_centers) * 2)
        return self._all_spheres[self._num_spheres]

    def _reset_buffers(self, capacity: int) -> None:
        self._ensure_capacity(capacity)
        self._num_spheres = 0
        self.insert_method.reset_buffers(capacity)

    # ------------------------------------------------------------------ chunked distances

    def _sq_dists_to_centers(self, vectors: np.ndarray, sq_norms: np.ndarray, centers: np.ndarray) -> np.ndarray:
        c_sq = np.einsum('ij,ij->i', centers, centers)
        cross = vectors @ centers.T
        return sq_norms[:, None] - 2.0 * cross + c_sq[None, :]

    def _log_volume_bounds(self) -> Tuple[float, float]:
        if len(self.all_centers) == 0:
            return -np.inf, -np.inf
        return estimate_hypersphere_volume(self.all_centers, self.all_radiuses, return_log=True)

    # ------------------------------------------------------------------ construct

    def construct(self, vectors: np.ndarray) -> Tuple[float, float]:
        centers, radii = self.construct_method.construct_filter(self, vectors)

        for center, radius in zip(centers, radii):
            self.insert_method.add_sphere(self, center, radius + self.distance_threshold)
        self._constructed = True

        return self._log_volume_bounds()

    # ------------------------------------------------------------------ query

    def query(self, vectors: np.ndarray) -> List[bool]:
        if not self._constructed:
            raise ValueError("BallCoverFilter has not been constructed.")

        if len(self.all_centers) == 0 or len(vectors) == 0:
            return [False] * len(vectors)

        q = np.ascontiguousarray(vectors, dtype=np.float32)
        q_sq = np.einsum('ij,ij->i', q, q)

        sq_distances = self._sq_dists_to_centers(q, q_sq, self.all_centers)
        in_any_sphere = np.any(sq_distances <= self.all_radiuses[None, :]**2, axis=1)

        return in_any_sphere.tolist()

    # ------------------------------------------------------------------ streaming / dynamic insertion

    def add(self, vectors: np.ndarray) -> None:
        self.insert_method.add(self, vectors)

    def insert(self, vectors: np.ndarray) -> None:
        self.insert_method.insert(self, vectors)

    def remove(self, vectors: np.ndarray) -> List[bool]:
        return self.insert_method.remove(self, vectors)

    # ------------------------------------------------------------------ memory / misc

    def get_volume(self) -> float:
        return 0.0

    def get_exact_matching_fpr(self) -> float:
        return 0.0

    def get_bytesize(self) -> int:
        if not self._constructed:
            return 0
        filter_bytes = self._all_spheres.nbytes + self.all_centers.nbytes + self.all_radiuses.nbytes
        if self.insert_method:
            filter_bytes += self.insert_method.get_bytesize()
        return filter_bytes

    def get_construction_bytesize(self) -> int:
        if not self.construct_method:
            return 0
        return self.construct_method.get_bytesize()