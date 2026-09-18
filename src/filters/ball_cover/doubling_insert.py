import numpy as np
import faiss
from typing import List, Tuple
from filters.filter import Filter
from filters.ball_cover.ball_interfaces import InsertInterface


class DoublingInsert(InsertInterface):
    """Amortized-rebuild insertion: a newly-arrived, not-already-covered
    vector is added immediately as its own zero-radius ball (an array
    append, no merge work), letting the active ball count grow past
    `filter.max_spheres` between rebuilds instead of enforcing a hard cap on
    every insertion. Once the active count reaches
    `max_spheres * (1 + growth_budget_fraction)`, a Gonzalez-style
    re-clustering pass folds every active ball back down to at most
    `max_spheres` -- amortized O(1) per insertion, at the price of an
    infrequent rebuild spike, trading the other strategies' steady per-point
    merge cost for occasional latency spikes.

    Every newly-added ball is immediately part of the filter's own queryable
    structure, so the zero-false-negative guarantee holds at every
    checkpoint, not just right after a rebuild.

    growth_budget_fraction (default 1.0, hence "doubling": up to 2x
    max_spheres before a forced rebuild) is a multiplier, not an absolute
    budget, since the strategy object is constructed before the filter that
    will own it and doesn't yet know max_spheres.

    The rebuild can't just call GonzalesConstruct directly: each active
    ball's own current radius must be carried into the re-clustering as
    positional uncertainty (the points it once covered could be anywhere
    within that radius), the same reconstruction-error technique
    BallCoverFilter._build_quantizer uses for quantized centers -- so the
    greedy farthest-first selection is mirrored here with that one addition
    rather than reused as-is.
    """

    def __init__(self, growth_budget_fraction: float = 1.0):
        if growth_budget_fraction <= 0.0:
            raise ValueError(
                f"growth_budget_fraction={growth_budget_fraction} must be > 0 "
                "(it sizes the rebuild-trigger threshold as a multiple of max_spheres)."
            )
        self.growth_budget_fraction = growth_budget_fraction

    # ------------------------------------------------------------------ InsertInterface plumbing

    def ensure_capacity(self, required_cap: int) -> None:
        # No strategy-local state beyond the filter's own sphere arrays,
        # which already grow themselves via filter._get_new_index().
        pass

    def reset_buffers(self, capacity: int) -> None:
        pass

    def add_sphere(self, filter: Filter, center: np.ndarray, radius: float) -> None:
        idx = filter._get_new_index()
        filter._all_centers[idx] = center
        filter._all_radiuses[idx] = radius
        filter._num_spheres += 1

    # ------------------------------------------------------------------ Gonzalez re-clustering

    @staticmethod
    def _gonzalez_select_with_uncertainty(
        points: np.ndarray, uncertainty: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Gonzalez farthest-first center selection over `points`, each
        carrying its own pre-existing uncertainty radius (see class
        docstring). Mirrors GonzalesConstruct.construct_filter's own
        greedy-selection-then-exact-radii algorithm; the only change is
        `reach = distance-to-center + that point's own uncertainty` instead
        of plain distance when computing each cluster's final radius."""
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

        # Each assigned point's contribution to its cluster's radius must
        # cover the *region* it stands for, not just itself: distance to
        # the new center, plus that point's own prior uncertainty radius.
        reach = (d_exact + uncertainty).astype(np.float32)

        order = np.argsort(labels, kind="stable")
        counts = np.bincount(labels, minlength=k)
        starts = np.zeros(k, dtype=np.int64)
        np.cumsum(counts[:-1], out=starts[1:])
        nonempty = counts > 0

        radii = np.zeros(k, dtype=np.float32)
        radii[nonempty] = np.maximum.reduceat(reach[order], starts[nonempty])
        return centers, radii

    def _rebuild_threshold(self, filter: Filter) -> int:
        return max(
            filter.max_spheres + 1,
            int(round(filter.max_spheres * (1.0 + self.growth_budget_fraction))),
        )

    def _rebuild(self, filter: Filter) -> None:
        points = np.ascontiguousarray(filter.all_centers, dtype=np.float32)
        uncertainty = np.asarray(filter.all_radiuses, dtype=np.float32)

        new_centers, new_radii = self._gonzalez_select_with_uncertainty(
            points, uncertainty, filter.max_spheres
        )

        # Retire the whole active set, then re-add via the same add_sphere()
        # every other strategy uses for a from-scratch build.
        filter._num_spheres = 0
        for center, radius in zip(new_centers, new_radii):
            self.add_sphere(filter, center, radius)
        filter._rebuild_from_centers(filter.all_spheres, filter.all_centers, filter.all_radiuses)

    # ------------------------------------------------------------------ InsertInterface: insert/remove

    @staticmethod
    def _exhaustive_search(centers: np.ndarray, queries: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        dim = centers.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(centers)
        d2, labels = index.search(queries, len(centers))
        return np.sqrt(np.maximum(d2, 0.0)), labels

    def insert(self, filter: Filter, vectors: np.ndarray) -> None:
        X = np.ascontiguousarray(vectors, dtype=np.float32)

        centers = filter.all_centers
        radii = filter.all_radiuses
        # labels give each match's position in `centers`, not sorted-rank
        # order -- must index radii by labels, not compare against radii
        # directly, or points get paired with the wrong ball's radius.
        dist, labels = self._exhaustive_search(centers, X)
        is_already_covered = np.any(dist <= radii[labels], axis=1)
        X = X[~is_already_covered]

        for vec in X:
            self.add_sphere(filter, vec, 0.0)

        if filter._num_spheres >= self._rebuild_threshold(filter):
            self._rebuild(filter)

    def remove(self, filter: Filter, vectors: np.ndarray) -> List[bool]:
        raise NotImplementedError("DoublingInsert does not support remove operation.")

    def get_bytesize(self, filter: Filter) -> int:
        return 0
