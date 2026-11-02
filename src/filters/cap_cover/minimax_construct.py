import numpy as np
from typing import Tuple
from filters.filter import Filter
from filters.cap_cover.cap_interfaces import ConstructInterface
from filters.cap_cover.gonzales_construct import GonzalesConstruct
from filters.cap_cover.cap_geometry import (angle_between, exact_cap_cover, labelled_cap_cover,
                                            nearest_directions, segments, segment_max, unit_rows)

try:  # Rust kernels (CapCoverRustFilter/src/construct_minimax.rs); the NumPy path is the fallback
    from capcover_rs import (mm_assign as _rs_assign, mm_assign_cand as _rs_cand, mm_bc as _rs_bc, mm_lanes as _rs_lanes,
                             mm_stats as _rs_stats)
except ImportError:  # extension missing or built before the kernels existed
    _rs_assign = _rs_cand = _rs_bc = _rs_lanes = _rs_stats = None

FAST_AVAILABLE = _rs_assign is not None


class MinimaxConstruct(ConstructInterface):
    """Gonzalez's directions refined by Lloyd's alternation with each
    cluster's minimum enclosing cap in place of its centroid: nearest-
    direction reassignment, then Badoiu-Clarkson steps toward each cluster's
    enclosing-ball center. For points on the unit sphere that center's
    direction is the minimum enclosing cap's (for caps under a hemisphere),
    so the steps run in Euclidean space and each cluster keeps whichever
    direction gave it the smallest cap angle.

    Rounds keep the best by sum over caps of theta^p (p = `exponent`), with
    Gonzalez's own directions as round 0.

    fast: run the rounds on the Rust kernels (exact fused nearest-direction
    search, cluster-local Badoiu-Clarkson); same algorithm, results equal up
    to float32 summation-order ties. Default: whenever the extension has them.
    adaptive (opt-in, changes the result): stop after a round whose score
    improves on the previous round's by less than `tol` (relative).
    approx_assign (opt-in, changes the result): each reassignment ranks the
    directions in the top `proj_dim` principal components of X (a 16-way
    shortlist, one winner per class of directions), then picks the best
    shortlisted direction by the full inner product. The cover stays valid
    (every angle is measured against the direction actually assigned) but a
    point may join a near-best rather than the best direction; the final
    cover does not redo an exact assignment."""

    def __init__(self, rounds: int = 5, bc_steps: int = 64, exponent: float = 8.0, fast: bool = None,
                 adaptive: bool = False, tol: float = 0.01, approx_assign: bool = False,
                 proj_dim: int = 128) -> None:
        self.rounds = rounds
        self.bc_steps = bc_steps
        self.exponent = exponent
        self.fast = FAST_AVAILABLE if fast is None else (bool(fast) and FAST_AVAILABLE)
        self.adaptive = adaptive
        self.tol = tol
        self.approx_assign = approx_assign
        self.proj_dim = proj_dim

    def construct_filter(self, filter: Filter, vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = np.ascontiguousarray(vectors, dtype=np.float64)
        n, dim = X.shape
        k = min(filter.max_construct_caps, n)
        if k <= 0:
            return np.empty((0, dim)), np.empty(0)
        if n <= k:
            return X.copy(), np.zeros(n)

        U, _ = GonzalesConstruct().construct_filter(filter, X)
        return self.refine(X, U)

    def refine(self, X: np.ndarray, U: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """The Lloyd/Badoiu-Clarkson rounds from the unit directions U (e.g.
        Gonzalez's) over the float64 unit rows X; returns the exact cover."""
        if self.fast:
            return self._refine_fast(X, U)
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

    def _refine_fast(self, X: np.ndarray, U: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = np.ascontiguousarray(X, dtype=np.float64)
        X32 = X.astype(np.float32)
        U = np.ascontiguousarray(U, dtype=np.float64)
        best_U, best_score, best_labels, prev = U, np.inf, None, np.inf
        assign = self._assigner(X32)
        for round_ in range(self.rounds + 1):
            labels = assign(np.ascontiguousarray(U, dtype=np.float32))
            _, theta_c, counts = _rs_stats(X, U, labels)
            score = float(np.sum(theta_c[counts > 0] ** self.exponent))
            if score < best_score:
                best_U, best_score, best_labels = U.copy(), score, labels
            if round_ == self.rounds or (self.adaptive and score > prev * (1.0 - self.tol)):
                break
            prev = score
            order, starts, counts = segments(labels, len(U))
            U = _rs_bc(X32, X, order, starts, counts, U, self.bc_steps)
            if (counts == 0).any():
                U = reseed_empty(X, U, counts, labels, theta=_rs_stats(X, U, labels)[0])
        U_out, angles, _, _ = labelled_cap_cover(X, best_U, best_labels)
        return U_out, angles

    def _assigner(self, X32: np.ndarray):
        """labels = f(U32): nearest direction of every row of X32 (exact, or
        the projected shortlist of approx_assign)."""
        n, d = X32.shape
        r = min(self.proj_dim, d)
        if not self.approx_assign or r >= d:
            return lambda U32: _rs_assign(X32, U32)[0]
        sample = X32[:: max(1, n // 20000)].astype(np.float64)
        sample -= sample.mean(axis=0)
        _, V = np.linalg.eigh(sample.T @ sample)
        V = np.ascontiguousarray(V[:, ::-1][:, :r], dtype=np.float32)
        Xp = np.ascontiguousarray(X32 @ V)

        def assign(U32: np.ndarray) -> np.ndarray:
            cand = _rs_lanes(Xp, np.ascontiguousarray(U32 @ V))
            return _rs_cand(X32, U32, cand)[0]
        return assign

    def get_bytesize(self) -> int:
        return 0


def badoiu_clarkson(Xs: np.ndarray, labels: np.ndarray, U: np.ndarray, starts: np.ndarray,
                    counts: np.ndarray, steps: int, fast: bool = None) -> np.ndarray:
    """`steps` Badoiu-Clarkson steps c <- c + (farthest - c) / (t + 1) on every
    cluster at once, membership fixed, Xs and labels in segments() order.
    Returns each cluster's best *unit direction* seen, by its cap angle.

    The farthest member is ranked by float32 inner product and only its
    angle is measured in float64: the step needs a farthest member, not the
    exact one, and the caller re-measures every angle it keeps."""
    if _rs_bc is not None and fast is not False and Xs.ndim == 2 and len(Xs):
        Xs64 = np.ascontiguousarray(Xs, dtype=np.float64)
        return _rs_bc(Xs64.astype(np.float32), Xs64, np.arange(len(Xs), dtype=np.int64),
                      np.ascontiguousarray(starts, dtype=np.int64), np.ascontiguousarray(counts, dtype=np.int64),
                      np.ascontiguousarray(U, dtype=np.float64), int(steps))
    C = np.asarray(U, dtype=np.float64).copy()
    best = unit_rows(C)
    best_t = np.full(len(C), np.inf)
    # A one-member cluster's cap is its member, exactly: no steps needed.
    single = np.flatnonzero(counts == 1)
    best[single] = unit_rows(Xs[starts[single]])
    multi = counts > 1
    if not multi.any():
        return best
    if len(single):
        rows = np.flatnonzero(multi[labels])
        Xs, labels = Xs[rows], labels[rows]
        counts = np.where(multi, counts, 0)
        starts = np.zeros(len(counts), dtype=np.int64)
        np.cumsum(counts[:-1], out=starts[1:])
    nonempty = np.flatnonzero(counts > 0)
    Xs32 = np.ascontiguousarray(Xs, dtype=np.float32)
    for t in range(1, steps + 1):
        Un = unit_rows(C)
        ip = np.einsum("ij,ij->i", Xs32, Un.astype(np.float32)[labels])
        least = np.minimum.reduceat(ip, starts[nonempty])
        low = np.full(len(C), np.inf, dtype=np.float32)
        low[nonempty] = least
        hits = np.flatnonzero(ip <= low[labels])
        cluster, first = np.unique(labels[hits], return_index=True)
        far = hits[first]
        theta = angle_between(Xs[far], Un[cluster])
        improved = theta < best_t[cluster]
        best[cluster[improved]] = Un[cluster[improved]]
        best_t[cluster[improved]] = theta[improved]
        C[cluster] += (Xs[far] - C[cluster]) / (t + 1.0)
    return best


def reseed_empty(X: np.ndarray, U: np.ndarray, counts: np.ndarray, labels: np.ndarray,
                 return_rows: bool = False, theta: np.ndarray = None):
    """Directions that lost every member move onto the members farthest from
    their own directions (`theta`: those angles, if the caller has them; with return_rows, also returns those members' rows,
    aligned with the empty caps in index order)."""
    empty = np.flatnonzero(counts == 0)
    if len(empty) == 0:
        return (U, np.empty(0, dtype=np.int64)) if return_rows else U
    d = angle_between(X, U[labels]) if theta is None else theta
    far = np.argpartition(-d, len(empty) - 1)[:len(empty)]
    U = U.copy()
    U[empty] = X[far]
    return (U, far) if return_rows else U
