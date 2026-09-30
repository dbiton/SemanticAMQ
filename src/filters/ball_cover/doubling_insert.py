import numpy as np
import faiss
from typing import List, Tuple, Optional
from filters.filter import Filter
from filters.ball_cover.ball_interfaces import InsertInterface


class DoublingInsert(InsertInterface):
    """Amortized-rebuild insertion: a not-already-covered vector is added
    immediately as its own zero-radius ball (no merge work), letting the
    active count grow past `filter.max_spheres` until it hits
    `max_spheres * (1 + growth_budget_fraction)`, at which point a
    Gonzalez-style re-clustering folds everything back down -- amortized
    O(1) per insertion, at the price of an infrequent rebuild spike, in
    exchange for the other strategies' steady per-point merge cost. Every
    newly-added ball is immediately queryable, so the zero-false-negative
    guarantee holds between rebuilds too, not just right after one.

    growth_budget_fraction (default 1.0, hence "doubling") is a multiplier
    rather than an absolute budget since the strategy is constructed before
    the filter that will own it, and doesn't yet know max_spheres.

    The rebuild can't call GonzalesConstruct directly: each active ball's
    own radius must carry into re-clustering as positional uncertainty (the
    points it covered could be anywhere within that radius) -- the same
    reconstruction-error technique BallCoverFilter._add_spheres uses for
    quantized centers -- so the farthest-first selection is mirrored here
    with that one addition."""

    def __init__(self, growth_budget_fraction: float = 1.0):
        if growth_budget_fraction <= 0.0:
            raise ValueError(
                f"growth_budget_fraction={growth_budget_fraction} must be > 0 "
                "(it sizes the rebuild-trigger threshold as a multiple of max_spheres)."
            )
        self.growth_budget_fraction = growth_budget_fraction


    def initialize_buffers(self, filter: Filter, spheres_indices: np.ndarray, centers: np.ndarray, radii: np.ndarray,
                           vectors: Optional[np.ndarray] = None) -> None:
        pass


    def ensure_capacity(self, required_cap: int) -> None:
        pass


    def reset_buffers(self, capacity: int) -> None:
        pass


    @staticmethod
    def _gonzalez_select_with_uncertainty(
        points: np.ndarray, uncertainty: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Mirrors GonzalesConstruct's greedy-selection-then-exact-radii
        algorithm; the only change is `reach = distance-to-center + that
        point's own uncertainty` instead of plain distance for each
        cluster's final radius (see class docstring)."""
        n = points.shape[0]
        dim = points.shape[1]
        k = min(k, n)
        if k <= 0:
            return np.empty((0, dim), dtype=np.float32), np.empty(0, dtype=np.float32)
        if n <= k:
            return points.copy(), uncertainty.copy()

        sq = np.einsum("ij,ij->i", points, points)

        chosen = np.empty(k, dtype=np.int64)
        min_d2 = np.empty(n, dtype=np.float32)
        d2 = np.empty(n, dtype=np.float32)

        chosen[0] = 0
        np.dot(points, points[0], out=min_d2)
        min_d2 *= -2.0
        min_d2 += sq
        min_d2 += sq[0]

        for j in range(1, k):
            c = int(min_d2.argmax())
            if min_d2[c] <= 0.0:
                chosen = chosen[:j]
                k = j
                break
            chosen[j] = c

            np.dot(points, points[c], out=d2)
            d2 *= -2.0
            d2 += sq
            d2 += sq[c]
            np.minimum(min_d2, d2, out=min_d2)

        centers = np.ascontiguousarray(points[chosen])

        index = faiss.IndexFlatL2(dim)
        index.add(centers)
        d2_exact, labels = index.search(points, 1)
        d_exact = np.sqrt(np.maximum(d2_exact.ravel(), 0.0))
        labels = labels.ravel()

        # Covers the region a point stands for, not just itself.
        reach = (d_exact + uncertainty).astype(np.float32)

        order = np.argsort(labels, kind="stable")
        counts = np.bincount(labels, minlength=k)
        starts = np.zeros(k, dtype=np.int64)
        np.cumsum(counts[:-1], out=starts[1:])
        nonempty = counts > 0

        radii = np.zeros(k, dtype=np.float32)
        radii[nonempty] = np.maximum.reduceat(reach[order], starts[nonempty])
        return centers, radii


    def _threshold_for(self, max_spheres: int) -> int:
        return max(max_spheres + 1, int(round(max_spheres * (1.0 + self.growth_budget_fraction))))


    def _rebuild_threshold(self, filter: Filter) -> int:
        return self._threshold_for(filter.max_spheres)


    def max_spheres_for_budget(self, usable_bytes: int, bytes_per_ball: int) -> int:
        """Between insert() calls the filter holds up to threshold - 1 balls
        (a call that reaches the threshold rebuilds before returning), so
        that peak is what has to fit -- max_spheres itself is only the
        post-rebuild size, at 1 / (1 + growth_budget_fraction) of it."""
        fit = usable_bytes // bytes_per_ball
        max_spheres = max(1, int(fit / (1.0 + self.growth_budget_fraction)))
        while max_spheres > 1 and self._threshold_for(max_spheres) - 1 > fit:
            max_spheres -= 1
        while self._threshold_for(max_spheres + 1) - 1 <= fit:
            max_spheres += 1
        return max_spheres


    def _rebuild(self, filter: Filter) -> None:
        points = np.ascontiguousarray(filter.all_centers, dtype=np.float32)
        uncertainty = np.asarray(filter.all_radiuses, dtype=np.float32)

        new_centers, new_radii = self._gonzalez_select_with_uncertainty(
            points, uncertainty, filter.max_spheres
        )

        filter._remove_spheres(filter.all_spheres)
        filter._add_spheres(new_centers, new_radii)


    @staticmethod
    def _exhaustive_search(centers: np.ndarray, queries: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(centers) == 0:
            return np.empty((len(queries), 0), dtype=np.float32), np.empty((len(queries), 0), dtype=np.int64)
        dim = centers.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(centers)
        d2, labels = index.search(queries, len(centers))
        return np.sqrt(np.maximum(d2, 0.0)), labels


    def insert(self, filter: Filter, vectors: np.ndarray) -> None:
        if len(vectors) == 0:
            return

        X = np.ascontiguousarray(vectors, dtype=np.float32)

        # Filter out covered vectors only if spheres exist
        if filter._num_spheres > 0:
            centers = filter.all_centers
            radii = filter.all_radiuses
            dist, labels = self._exhaustive_search(centers, X)
            is_already_covered = np.any(dist <= radii[labels], axis=1)
            X = X[~is_already_covered]

        if len(X) == 0:
            return

        # Batch-add uncovered vectors as zero-radius spheres
        zero_radii = np.zeros(len(X), dtype=np.float32)
        filter._add_spheres(X, zero_radii)

        # Rebuild if growth threshold exceeded
        if filter._num_spheres >= self._rebuild_threshold(filter):
            self._rebuild(filter)


    def remove(self, filter: Filter, vectors: np.ndarray) -> List[bool]:
        raise NotImplementedError("DoublingInsert does not support remove operation.")


    def get_bytesize(self, filter: Filter) -> int:
        return 0