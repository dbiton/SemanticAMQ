import numpy as np
from typing import Tuple
from filters.filter import Filter
from filters.cap_cover.cap_interfaces import ConstructInterface
from filters.cap_cover.gonzales_construct import GonzalesConstruct
from filters.cap_cover.cap_geometry import (angle_between, exact_cap_cover, nearest_directions, segments,
                                            segment_max, unit_rows)


class MinimaxConstruct(ConstructInterface):
    """Gonzalez's directions refined by Lloyd's alternation with each
    cluster's minimum enclosing cap in place of its centroid: nearest-
    direction reassignment, then Badoiu-Clarkson steps toward each cluster's
    enclosing-ball center. For points on the unit sphere that center's
    direction is the minimum enclosing cap's (for caps under a hemisphere),
    so the steps run in Euclidean space and each cluster keeps whichever
    direction gave it the smallest cap angle.

    Rounds keep the best by sum over caps of theta^p (p = `exponent`), with
    Gonzalez's own directions as round 0."""

    def __init__(self, rounds: int = 5, bc_steps: int = 64, exponent: float = 8.0) -> None:
        self.rounds = rounds
        self.bc_steps = bc_steps
        self.exponent = exponent

    def construct_filter(self, filter: Filter, vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = np.ascontiguousarray(vectors, dtype=np.float64)
        n, dim = X.shape
        k = min(filter.max_construct_caps, n)
        if k <= 0:
            return np.empty((0, dim)), np.empty(0)
        if n <= k:
            return X.copy(), np.zeros(n)

        U, _ = GonzalesConstruct().construct_filter(filter, X)
        best_U, best_score = U, np.inf
        for round_ in range(self.rounds + 1):
            _, labels = nearest_directions(X, U)
            order, starts, counts = segments(labels, len(U))
            theta = segment_max(angle_between(X, U[labels])[order], starts, counts)
            score = float(np.sum(theta[counts > 0] ** self.exponent))
            if score < best_score:
                best_U, best_score = U.copy(), score
            if round_ == self.rounds:
                break
            U = badoiu_clarkson(X[order], labels[order], U, starts, counts, self.bc_steps)
            U = reseed_empty(X, U, counts, labels)
        return exact_cap_cover(X, best_U)

    def get_bytesize(self) -> int:
        return 0


def badoiu_clarkson(Xs: np.ndarray, labels: np.ndarray, U: np.ndarray, starts: np.ndarray,
                    counts: np.ndarray, steps: int) -> np.ndarray:
    """`steps` Badoiu-Clarkson steps c <- c + (farthest - c) / (t + 1) on every
    cluster at once, membership fixed, Xs and labels in segments() order.
    Returns each cluster's best *unit direction* seen, by its cap angle."""
    C = np.asarray(U, dtype=np.float64).copy()
    best = unit_rows(C)
    best_t = np.full(len(C), np.inf)
    nonempty = counts > 0
    for t in range(1, steps + 1):
        Un = unit_rows(C)
        ang = angle_between(Xs, Un[labels])
        theta = segment_max(ang, starts, counts)
        improved = nonempty & (theta < best_t)
        best[improved] = Un[improved]
        best_t[improved] = theta[improved]
        hits = np.flatnonzero(ang >= theta[labels])
        cluster, first = np.unique(labels[hits], return_index=True)
        far = hits[first]
        C[cluster] += (Xs[far] - C[cluster]) / (t + 1.0)
    return best


def reseed_empty(X: np.ndarray, U: np.ndarray, counts: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Directions that lost every member move onto the members farthest from
    their own directions."""
    empty = np.flatnonzero(counts == 0)
    if len(empty) == 0:
        return U
    d = angle_between(X, U[labels])
    far = np.argpartition(-d, len(empty) - 1)[:len(empty)]
    U = U.copy()
    U[empty] = X[far]
    return U
