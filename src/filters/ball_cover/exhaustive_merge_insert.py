import numpy as np
from typing import List, Tuple
from filters.filter import Filter
from filters.ball_cover.ball_interfaces import InsertInterface


class ExhaustiveMergeInsert(InsertInterface):
    """Insert each new vector as its own zero-radius ball; once at capacity,
    evict by merging whichever pair of balls -- existing-existing or the
    new singleton against an existing ball -- grows the covering objective
    least. The choice is global: a point landing in genuinely new territory
    gets to found a ball rather than deforming one, and redundancy gets
    given up wherever it's cheapest, not wherever the new point happened to
    land.

    Sequential per point (each merge changes the ball set the next point's
    decision depends on, unlike insert_nearest/insert_least_inflation,
    which can absorb a whole batch in one vectorized pass). Recomputes the
    full k x k merge-cost matrix on every insertion at capacity --
    O(k**2*d) per point, correct but not the O(k*d)-amortized version an
    incremental row/column-cached heap would give.

    Runs the whole batch in float64 numpy with no intermediate
    re-quantization -- the caller quantizes once, from the final
    (new_centers, new_radii), rather than re-quantizing (and reading centers
    back out of the quantized index, compounding quantization error) after
    every single point."""

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


    @staticmethod
    def _cheapest_pair(centers: np.ndarray, radii: np.ndarray, dim: int) -> Tuple[int, int]:
        """argmin over all pairs (i, j) of the merge objective delta
        r_merged**d - r_i**d - r_j**d, log-space scaled the same way as
        insert_least_inflation (shared M = max radius among these centers)."""
        n = len(centers)
        ci, cj = centers[:, None, :], centers[None, :, :]
        ri, rj = radii[:, None], radii[None, :]
        diff = ci - cj
        dist = np.sqrt(np.maximum(np.sum(diff * diff, axis=-1), 0.0))
        contains = dist + np.minimum(ri, rj) <= np.maximum(ri, rj)
        merged = 0.5 * (dist + ri + rj)
        r_merged = np.where(contains, np.maximum(ri, rj), merged)

        M = float(max(radii.max(), r_merged.max())) if n else 0.0
        safe_M = M if M > 0 else 1.0
        with np.errstate(divide="ignore"):
            scaled_i = np.exp(dim * np.log(np.maximum(ri, 0.0) / safe_M))
            scaled_j = np.exp(dim * np.log(np.maximum(rj, 0.0) / safe_M))
            scaled_m = np.exp(dim * np.log(np.maximum(r_merged, 0.0) / safe_M))
        delta = (scaled_m - scaled_i - scaled_j) if M > 0 else np.zeros((n, n))

        idx = np.arange(n)
        delta[idx, idx] = np.inf
        i, j = np.unravel_index(np.argmin(delta), delta.shape)
        return int(i), int(j)


    def insert(self, filter: Filter, vectors: np.ndarray) -> None:
        dim = filter.dim
        c = np.ascontiguousarray(filter.all_centers, dtype=np.float64)
        r = np.asarray(filter.all_radiuses, dtype=np.float64)
        X = np.ascontiguousarray(vectors, dtype=np.float32).astype(np.float64)

        for x in X:
            c = np.vstack([c, x[None, :]])
            r = np.append(r, 0.0)

            if len(c) > filter.max_spheres:
                i, j = self._cheapest_pair(c, r, dim)
                c_new, r_new = self._merge_two_balls(c[i], r[i], c[j], r[j])
                keep = [idx for idx in range(len(c)) if idx not in (i, j)]
                c = np.vstack([c[keep], c_new[None, :]])
                r = np.append(r[keep], r_new)

        n_new = len(c)
        filter._ensure_capacity(n_new)
        filter._all_centers[:n_new] = c.astype(np.float32)
        filter._all_radiuses[:n_new] = r.astype(np.float32)
        filter._num_spheres = n_new


    def remove(self, filter: Filter, vectors: np.ndarray) -> List[bool]:
        raise NotImplementedError("ExhaustiveMergeInsert does not support remove operation.")


    def get_bytesize(self) -> int:
        return 0