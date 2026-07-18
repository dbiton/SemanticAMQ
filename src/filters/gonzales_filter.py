import numpy as np
import faiss
from typing import List, Tuple, Optional
from filters.filter import Filter
from hyperspheres_volume import estimate_hypersphere_volume


class GonzalezMEBFilter(Filter):
    """
    Duplicate filter built from Gonzalez farthest-point clustering with a
    (1+eps)-approximate Minimum Enclosing Ball per cluster. Radii are exact
    max distances from the final centers plus distance_threshold, so the
    filter admits false positives but never false negatives. Batch query and
    remove run on FAISS flat indexes; ball containment with per-sphere radii
    is reduced to a single exact k=1 inner-product search over augmented
    centers [2c, r^2 - ||c||^2].
    """

    _BYTES_PER_FLOAT32 = 4
    _MEB_ITERATIONS = 100

    def __init__(self, dim: int, distance_threshold: float, max_memory: Optional[int] = None, max_spheres: Optional[int] = None):
        super().__init__(dim, lsh=None, max_memory=max_memory)
        self.distance_threshold = distance_threshold
        self.all_spheres = np.empty((0, dim), dtype=np.float32)
        self.all_radiuses = np.empty((0,), dtype=np.float32)
        self.sphere_counts = np.empty((0,), dtype=np.int32)
        self._constructed = False
        self._containment_index: Optional[faiss.IndexFlatIP] = None
        self._center_index: Optional[faiss.IndexFlatL2] = None
        self.max_spheres = (max_spheres if max_spheres is not None
                            else self._calculate_max_spheres(self.max_memory))

    def _calculate_max_spheres(self, max_memory: int, gamma: float = 4.0) -> int:
        bytes_per_sphere = (self.dim + 1) * self._BYTES_PER_FLOAT32
        return max(1, int(max_memory // (bytes_per_sphere * gamma)))

    def _invalidate_indexes(self) -> None:
        self._containment_index = None
        self._center_index = None

    def _get_containment_index(self) -> faiss.IndexFlatIP:
        if self._containment_index is None:
            c_sq = np.einsum('ij,ij->i', self.all_spheres, self.all_spheres)
            augmented = np.empty((len(self.all_spheres), self.dim + 1), dtype=np.float32)
            augmented[:, :self.dim] = 2.0 * self.all_spheres
            augmented[:, self.dim] = self.all_radiuses.astype(np.float64) ** 2 - c_sq
            index = faiss.IndexFlatIP(self.dim + 1)
            index.add(augmented)
            self._containment_index = index
        return self._containment_index

    def _get_center_index(self) -> faiss.IndexFlatL2:
        if self._center_index is None:
            index = faiss.IndexFlatL2(self.dim)
            index.add(np.ascontiguousarray(self.all_spheres))
            self._center_index = index
        return self._center_index

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

    def _minimum_enclosing_ball(self, points: np.ndarray, sq_norms: np.ndarray) -> Tuple[np.ndarray, float]:
        if len(points) == 1:
            return points[0].astype(np.float32), 0.0

        center = points.mean(axis=0, dtype=np.float64)
        for t in range(1, self._MEB_ITERATIONS + 1):
            sq_d = sq_norms - 2.0 * (points @ center) + center @ center
            center += (points[int(sq_d.argmax())] - center) / (t + 1.0)

        sq_d = np.maximum(sq_norms - 2.0 * (points @ center) + center @ center, 0.0)
        return center.astype(np.float32), float(np.sqrt(sq_d.max()))

    def _log_volume_bounds(self) -> Tuple[float, float]:
        if len(self.all_spheres) == 0:
            return -np.inf, -np.inf
        return estimate_hypersphere_volume(self.all_spheres, self.all_radiuses, return_log=True)

    def construct(self, vectors: np.ndarray) -> Tuple[float, float]:
        self._invalidate_indexes()

        if len(vectors) == 0 or self.max_spheres == 0:
            self._constructed = True
            return self._log_volume_bounds()

        vectors = np.ascontiguousarray(vectors, dtype=np.float32)

        if self.max_spheres >= len(vectors):
            self.all_spheres = vectors.copy()
            self.all_radiuses = np.full(len(vectors), self.distance_threshold, dtype=np.float32)
            self.sphere_counts = np.ones(len(vectors), dtype=np.int32)
            self._constructed = True
            return self._log_volume_bounds()

        sq_norms = np.einsum('ij,ij->i', vectors, vectors)
        assignment = self._gonzalez_clustering(vectors, sq_norms, self.max_spheres)

        order = np.argsort(assignment, kind='stable')
        sorted_assignment = assignment[order]
        boundaries = np.flatnonzero(np.diff(sorted_assignment)) + 1
        groups = np.split(order, boundaries)

        num_spheres = len(groups)
        centers = np.empty((num_spheres, self.dim), dtype=np.float32)
        radiuses = np.empty(num_spheres, dtype=np.float32)
        counts = np.empty(num_spheres, dtype=np.int32)

        for i, member_idx in enumerate(groups):
            center, meb_radius = self._minimum_enclosing_ball(vectors[member_idx], sq_norms[member_idx])
            centers[i] = center
            radiuses[i] = meb_radius + self.distance_threshold
            counts[i] = len(member_idx)

        self.all_spheres = centers
        self.all_radiuses = radiuses
        self.sphere_counts = counts
        self._constructed = True
        return self._log_volume_bounds()

    def query(self, vectors: np.ndarray) -> List[bool]:
        if not self._constructed:
            raise ValueError("GonzalezMEBFilter has not been constructed.")

        if len(self.all_spheres) == 0 or len(vectors) == 0:
            return [False] * len(vectors)

        q = np.ascontiguousarray(vectors, dtype=np.float32)
        augmented_q = np.empty((len(q), self.dim + 1), dtype=np.float32)
        augmented_q[:, :self.dim] = q
        augmented_q[:, self.dim] = 1.0

        scores, _ = self._get_containment_index().search(augmented_q, 1)
        q_sq = np.einsum('ij,ij->i', q, q)
        return (scores[:, 0] >= q_sq).tolist()

    def add(self, vectors: np.ndarray) -> None:
        if not self._constructed:
            self.construct(vectors)
            return

        if len(vectors) == 0:
            return

        self._invalidate_indexes()
        new_vectors = np.asarray(vectors, dtype=np.float32)

        for q in new_vectors:
            diff = self.all_spheres - q
            sq_d = np.einsum('ij,ij->i', diff, diff)
            best = int(sq_d.argmin())
            best_dist = float(np.sqrt(sq_d[best]))
            r_best = float(self.all_radiuses[best])

            if best_dist <= r_best:
                self.sphere_counts[best] += 1
            elif best_dist <= r_best + 2.0 * self.distance_threshold or len(self.all_spheres) >= self.max_spheres:
                c_old = self.all_spheres[best]
                count = self.sphere_counts[best]
                c_new = ((c_old * count) + q) / (count + 1)
                shift = float(np.linalg.norm(c_old - c_new))
                dist_q = float(np.linalg.norm(q - c_new))
                self.all_spheres[best] = c_new
                self.all_radiuses[best] = max(r_best + shift, dist_q + self.distance_threshold)
                self.sphere_counts[best] += 1
            else:
                self.all_spheres = np.vstack([self.all_spheres, q[None, :]])
                self.all_radiuses = np.append(self.all_radiuses, np.float32(self.distance_threshold))
                self.sphere_counts = np.append(self.sphere_counts, np.int32(1))

    def get_volume(self) -> float:
        return 0.0

    def insert(self, vectors: np.ndarray) -> None:
        self.add(vectors)

    def remove(self, vectors: np.ndarray) -> List[bool]:
        if len(self.all_spheres) == 0 or len(vectors) == 0:
            return [False] * len(vectors)

        q = np.ascontiguousarray(vectors, dtype=np.float32)
        sq_d, idx = self._get_center_index().search(q, 1)
        best_idx = idx[:, 0]
        in_sphere = sq_d[:, 0] <= self.all_radiuses[best_idx] ** 2

        if np.any(in_sphere):
            self.sphere_counts -= np.bincount(best_idx[in_sphere], minlength=len(self.sphere_counts)).astype(np.int32)
            keep = self.sphere_counts > 0
            if not np.all(keep):
                self.all_spheres = np.ascontiguousarray(self.all_spheres[keep])
                self.all_radiuses = self.all_radiuses[keep]
                self.sphere_counts = self.sphere_counts[keep]
            self._invalidate_indexes()

        return in_sphere.tolist()

    def get_exact_matching_fpr(self) -> float:
        return 0.0

    def get_bytesize(self) -> int:
        if not self._constructed:
            return 0
        return self.all_spheres.nbytes + self.all_radiuses.nbytes + self.sphere_counts.nbytes