import numpy as np
from typing import Tuple
from filters.filter import Filter
from filters.ball_cover.ball_interfaces import ConstructInterface
from filters.ball_cover.gonzales_construct import GonzalesConstruct
from filters.ball_cover.exact_cover import (exact_cover, nearest_centers, segments, segment_max,
                                            sphere_centers, sphere_radius)


class MinimaxConstruct(ConstructInterface):
    """Gonzalez's centers refined by local search on the cover's own
    objective: Lloyd's alternation, with each cluster's minimum-enclosing-
    ball center in place of its centroid. Every round moves each center
    toward its members' MEB center with Badoiu-Clarkson steps, which shrinks
    each ball's max radius (what k-means ignores), and then reassigns every
    vector to its nearest center.

    Never worse than Gonzalez on the sum of ball volumes (sum_i r_i^d):
    each cluster keeps whichever center gave it the smallest radius during
    the steps, and the returned centers are the best round's by that sum,
    with Gonzalez's own centers as round 0. Costs
    O(rounds * (n*k*d + bc_steps*n*d)) on top of Gonzalez.

    With `sphere` set and data that lies on a sphere, the chosen centers are
    finally moved onto one common sphere (exact_cover.sphere_centers) before
    the radii are measured. Measured 2026-09-29 on the blocklist workload:
    FPR -0.15 / -0.06 / -0.045 / -0.04 against the default at k = 5/10/15/20%
    of |B| for the same construction time -- while its cover volume gets
    worse, since the radii grow as the caps on the sphere shrink."""

    def __init__(self, rounds: int = 5, bc_steps: int = 64, sphere: bool = False) -> None:
        self.rounds = rounds
        self.bc_steps = bc_steps
        self.sphere = sphere


    def construct_filter(self, filter: Filter, vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = np.ascontiguousarray(vectors, dtype=np.float32)
        n = X.shape[0]
        dim = X.shape[1]
        k = min(filter.max_construct_spheres, n)
        if k <= 0:
            return np.empty((0, dim), dtype=np.float32), np.empty(0, dtype=np.float32)
        if n <= k:
            return X.copy(), np.zeros(n, dtype=np.float32)

        centers, _ = GonzalesConstruct().construct_filter(filter, X)
        best_centers, best_volume = centers, np.inf
        for round_ in range(self.rounds + 1):
            d2, labels = nearest_centers(X, centers)
            order, starts, counts = segments(labels, len(centers))
            r2 = segment_max(d2[order], starts, counts)
            # log sum_i r_i^d, from squared radii; empty clusters add nothing.
            with np.errstate(divide="ignore"):
                volume = np.logaddexp.reduce(0.5 * dim * np.log(r2[counts > 0]))
            if volume < best_volume:
                best_centers, best_volume = centers.copy(), volume
            if round_ == self.rounds:
                break

            centers = self._badoiu_clarkson(X[order], labels[order], centers, starts, counts)
            centers = self._reseed_empty(X, centers, counts, d2)

        if self.sphere:
            rho = sphere_radius(X)
            if rho:
                best_centers = sphere_centers(X, best_centers, rho)

        # Radii from the exact re-search, never from the ones tracked above.
        return exact_cover(X, best_centers)


    def _badoiu_clarkson(self, Xs: np.ndarray, labels: np.ndarray, centers: np.ndarray,
                         starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
        """bc_steps of c <- c + (farthest member - c) / (t + 1) on every
        cluster at once, with membership fixed. Xs and labels are in
        segments() order. Returns each cluster's best center seen, so no
        radius comes out larger than it went in."""
        centers = centers.copy()
        best = centers.copy()
        best_r2 = np.full(len(centers), np.inf, dtype=np.float32)
        nonempty = counts > 0

        for t in range(1, self.bc_steps + 1):
            delta = Xs - centers[labels]
            d2 = np.einsum("ij,ij->i", delta, delta)
            r2 = segment_max(d2, starts, counts)

            improved = nonempty & (r2 < best_r2)
            best[improved] = centers[improved]
            best_r2[improved] = r2[improved]

            # First row of each cluster that attains its max: rows are
            # sorted by label, so np.unique's first index picks it.
            hits = np.flatnonzero(d2 >= r2[labels])
            cluster, first = np.unique(labels[hits], return_index=True)
            farthest = hits[first]
            centers[cluster] += (Xs[farthest] - centers[cluster]) / np.float32(t + 1)
        return best


    @staticmethod
    def _reseed_empty(X: np.ndarray, centers: np.ndarray, counts: np.ndarray,
                      d2: np.ndarray) -> np.ndarray:
        """Moves centers that lost every member onto the vectors farthest
        from their own centers -- farthest-first again, for the balls that
        would otherwise be wasted."""
        empty = np.flatnonzero(counts == 0)
        if len(empty) == 0:
            return centers
        farthest = np.argpartition(-d2, len(empty) - 1)[:len(empty)]
        centers = centers.copy()
        centers[empty] = X[farthest]
        return centers


    def get_bytesize(self) -> int:
        return 0
