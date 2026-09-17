import numpy as np
from typing import List, Tuple
from filters.filter import Filter
from filters.ball_cover.ball_interfaces import InsertInterface


class ExhaustiveMergeInsert(InsertInterface):
    """Insert each new vector as its own zero-radius ball; once at capacity,
    evict by merging whichever pair of balls grows the covering objective
    least, chosen globally rather than locally to the new point.

    Sequential per point (each merge changes the next point's decision),
    unlike insert_nearest/insert_least_inflation's vectorized batch pass.
    Recomputes the full k x k merge-cost matrix on every insertion at
    capacity -- O(k**2*d) per point, not the amortized O(k*d) an incremental
    cached heap would give.

    Quantizes once from the final (new_centers, new_radii), not after every
    point (which would compound quantization error)."""

    def ensure_capacity(self, required_cap: int) -> None:
        pass


    def reset_buffers(self, capacity: int) -> None:
        pass


    def add_sphere(self, filter: Filter, center: np.ndarray, radius: float) -> None:
        idx = filter._get_new_index()
        filter._all_centers[idx] = center
        filter._all_radiuses[idx] = radius
        filter._num_spheres += 1


    @staticmethod
    def _merge_two_balls(c1: np.ndarray, r1: float, c2: np.ndarray, r2: float) -> Tuple[np.ndarray, float]:
        """Smallest enclosing ball of two balls: whichever contains the other
        if one does, else centered on the line between them with radius
        (dist + r1 + r2) / 2."""
        diff = c2 - c1
        dist = float(np.sqrt(max(float(np.dot(diff, diff)), 0.0)))
        if dist + min(r1, r2) <= max(r1, r2):
            return (c1, r1) if r1 >= r2 else (c2, r2)
        new_r = 0.5 * (dist + r1 + r2)
        new_c = c1 if dist < 1e-12 else c1 + ((new_r - r1) / dist) * diff
        return new_c, new_r


    def insert(self, filter: Filter, vectors: np.ndarray) -> None:
        dim = filter.dim
        max_k = filter.max_spheres
        n = filter._num_spheres

        capacity = max(max_k, n) + 1
        c = np.empty((capacity, dim), dtype=np.float64)
        r = np.empty(capacity, dtype=np.float64)

        if n > 0:
            c[:n] = np.asarray(filter.all_centers[:n], dtype=np.float64)
            r[:n] = np.asarray(filter.all_radiuses[:n], dtype=np.float64)

        X = np.ascontiguousarray(vectors, dtype=np.float64)

        for x in X:
            if n > 0:
                diffs = c[:n] - x
                dists_sq = np.einsum("ij,ij->i", diffs, diffs)
                if np.any(dists_sq <= r[:n] * r[:n]):
                    continue

            c[n] = x
            r[n] = 0.0
            n += 1

            while n > max_k:
                C = c[:n]

                norms = np.einsum("ij,ij->i", C, C)
                gram = C @ C.T

                dist_sq = norms[:, None] + norms[None, :] - 2.0 * gram
                np.maximum(dist_sq, 0.0, out=dist_sq)
                np.sqrt(dist_sq, out=dist_sq)
                dist = dist_sq

                ri = r[:n, None]
                rj = r[:n][None, :]

                rmin = np.minimum(ri, rj)
                rmax = np.maximum(ri, rj)

                contains = dist + rmin <= rmax
                r_merged = 0.5 * (dist + ri + rj)
                np.maximum(r_merged, rmax, out=r_merged, where=contains)

                M = float(max(r[:n].max(), r_merged.max()))

                if M > 0.0:
                    inv_M = 1.0 / M
                    scaled_i = np.power(ri * inv_M, dim)
                    scaled_j = np.power(rj * inv_M, dim)
                    scaled_m = np.power(r_merged * inv_M, dim)
                    delta = scaled_m - scaled_i - scaled_j
                else:
                    delta = np.zeros((n, n), dtype=np.float64)

                np.fill_diagonal(delta, np.inf)

                i, j = np.unravel_index(np.argmin(delta), delta.shape)
                i, j = int(i), int(j)

                c_new, r_new = self._merge_two_balls(c[i], r[i], c[j], r[j])
                c[i] = c_new
                r[i] = r_new

                if j != n - 1:
                    c[j] = c[n - 1]
                    r[j] = r[n - 1]

                n -= 1

        filter._ensure_capacity(n)
        filter._all_centers[:n] = c[:n].astype(np.float32)
        filter._all_radiuses[:n] = r[:n].astype(np.float32)
        filter._num_spheres = n


    def remove(self, filter: Filter, vectors: np.ndarray) -> List[bool]:
        raise NotImplementedError("ExhaustiveMergeInsert does not support remove operation.")


    def get_bytesize(self) -> int:
        return 0