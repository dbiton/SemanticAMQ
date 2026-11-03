import os
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

try:  # construct_assign.rs: nearest direction plus runner-up score (for AssignState)
    from capcover_rs import as_top2 as _rs_top2, row_angles as _rs_row_angles
except ImportError:
    _rs_top2 = _rs_row_angles = None

REUSE_AVAILABLE = _rs_top2 is not None and FAST_AVAILABLE
# float32 slack of the bound test (see AssignState): inner products are padded by _IP_PAD, angles by _ANGLE_MARGIN.
_IP_PAD = 1e-5
_ANGLE_MARGIN = 2e-4


def _angle_ub_ip(ip: np.ndarray) -> np.ndarray:
    """Lower bound on the angle for a float32 inner product `ip` (true ip <= ip + _IP_PAD)."""
    return np.arccos(np.clip(ip.astype(np.float64) + _IP_PAD, -1.0, 1.0))


class AssignState:
    """Exact nearest-direction assignment of fixed rows X against directions that move a little between
    calls, reusing the previous call's work (Hamerly's bound on the sphere).

    Per row it keeps the owner a(x) and a lower bound L(x) on the angle to every *other* direction. When the
    directions move by delta_j = angle(old_j, new_j) (measured in float64):
      * the owner's angle u(x) is re-measured exactly (n x d);
      * a direction j can come closer by at most delta_j, so for the *stable* directions L -= max delta;
      * the few *movers* (largest delta, chosen per call by a cost estimate on a row sample) are scored
        exactly against all rows instead, which is what keeps one wandering direction from loosening the
        bound of everybody;
      * rows with u(x) <= L(x) - margin provably keep their owner; the others are searched in full.
    The result equals mm_assign's labels except where float32 inner products are within ~1e-4 rad of a tie
    (the margin makes the skip decision conservative; searched rows are ranked exactly as mm_assign does).

    reuse=False (or env CAPCOVER_ASSIGN_REUSE=0, or the kernel missing) searches everything every call.
    `history` holds, per call, the (fraction of rows searched, number of movers)."""

    def __init__(self, X32: np.ndarray, X64: np.ndarray = None, reuse: bool = None, sample: int = 256,
                 margin: float = _ANGLE_MARGIN, min_k: int = 400) -> None:
        self.min_k = min_k          # fewer directions than this: the bookkeeping costs more than it saves
        self.X32 = np.ascontiguousarray(X32, dtype=np.float32)
        self.X64 = None if X64 is None else np.ascontiguousarray(X64, dtype=np.float64)
        if reuse is None:
            reuse = os.environ.get("CAPCOVER_ASSIGN_REUSE", "1") not in ("0", "false", "False", "")
        self.reuse = bool(reuse) and REUSE_AVAILABLE and self.X64 is not None
        self.sample = sample
        self.margin = margin
        self.labels = None
        self.history = []
        self._U = None
        self._L = None

    def _full(self, U32: np.ndarray, U64: np.ndarray) -> np.ndarray:
        if self.reuse and len(U32) > 1:
            lab, _, v2 = _rs_top2(self.X32, U32)
            self._L = _angle_ub_ip(v2)
            self._U = np.array(U64, dtype=np.float64)
        else:
            lab = _rs_assign(self.X32, U32)[0]
        self.labels = lab
        self.history.append((1.0, len(U32)))
        return lab

    def assign(self, U32: np.ndarray, U64: np.ndarray = None) -> np.ndarray:
        """labels of the nearest direction of every row; U32 float32 (k x d), U64 the float64 directions it
        was cast from (unit rows)."""
        U32 = np.ascontiguousarray(U32, dtype=np.float32)
        if self._U is None and len(U32) < self.min_k:
            self.reuse = False
        if U64 is None:
            U64 = U32.astype(np.float64)
        U64 = np.ascontiguousarray(U64, dtype=np.float64)
        k, n = len(U32), len(self.X32)
        if not self.reuse or self._U is None or self._U.shape != U64.shape or k <= 1 or n == 0:
            return self._full(U32, U64)
        lab, L = self.labels, self._L
        delta = _rs_row_angles(U64, self._U, np.arange(k, dtype=np.int64))
        order = np.argsort(-delta, kind="stable")
        ds = delta[order]                                   # descending
        # choose the number of movers m on a row sample
        step = max(1, n // self.sample)
        sidx = np.arange(0, n, step)
        S = self.X32[sidx] @ U32[order].T                   # columns in descending-delta order
        pos = np.empty(k, dtype=np.int64)
        pos[order] = np.arange(k)
        S[np.arange(len(sidx)), pos[lab[sidx]]] = -2.0      # the owner is never a mover candidate
        cm = np.maximum.accumulate(S, axis=1)
        cands = np.unique(np.concatenate(([0], np.geomspace(1, max(1, k - 1), 24).astype(np.int64))))
        cands = cands[cands < k]
        best = None
        us = _rs_row_angles(self.X64[sidx].copy(), U64, lab[sidx])
        Ls = L[sidx]
        for m in cands:
            rest = ds[m]                                    # largest delta among the stable ones
            lp = Ls - rest
            if m > 0:
                lp = np.minimum(lp, _angle_ub_ip(cm[:, m - 1]))
            cost = m / k + float(np.mean(us > lp - self.margin))
            if best is None or cost < best[0]:
                best = (cost, int(m))
        if best[0] >= 0.75:                                  # nothing to gain: plain search
            return self._full(U32, U64)
        m = best[1]
        u = _rs_row_angles(self.X64, U64, lab)
        Lp = L - ds[m]
        if m > 0:
            mov = np.ascontiguousarray(order[:m])
            m1, v1, v2 = _rs_top2(self.X32, np.ascontiguousarray(U32[mov]))
            vm = np.where(mov[m1] == lab, v2, v1)
            Lp = np.minimum(Lp, _angle_ub_ip(vm))
        need = np.flatnonzero(u > Lp - self.margin)
        if len(need):
            nl, _, n2 = _rs_top2(np.ascontiguousarray(self.X32[need]), U32)
            lab = lab.copy()
            lab[need] = nl
            Lp[need] = _angle_ub_ip(n2)
        self.labels, self._L, self._U = lab, Lp, U64.copy()
        self.history.append((len(need) / n, m))
        return lab


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
    reuse_assign (default on; env CAPCOVER_ASSIGN_REUSE=0 turns it off): rounds after the first reuse the
    previous assignment through AssignState's exact bounds instead of searching every row again (only
    from `assign_min_k` directions up; fewer and the bookkeeping does not pay).
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
                 proj_dim: int = 128, reuse_assign: bool = None, assign_min_k: int = 400) -> None:
        self.assign_min_k = assign_min_k
        self.reuse_assign = reuse_assign        # None: env CAPCOVER_ASSIGN_REUSE (default on)
        self.assign_history = []
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
        self._state = None
        assign = self._assigner(X32, X)
        for round_ in range(self.rounds + 1):
            labels = assign(np.ascontiguousarray(U, dtype=np.float32), U)
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
        st = getattr(self, "_state", None)
        self.assign_history = list(st.history) if st is not None else []
        U_out, angles, _, _ = labelled_cap_cover(X, best_U, best_labels)
        return U_out, angles

    def _assigner(self, X32: np.ndarray, X64: np.ndarray = None):
        """labels = f(U32): nearest direction of every row of X32 (exact, or
        the projected shortlist of approx_assign)."""
        n, d = X32.shape
        r = min(self.proj_dim, d)
        if not self.approx_assign or r >= d:
            state = AssignState(X32, X64, reuse=self.reuse_assign, min_k=self.assign_min_k)
            self._state = state
            return state.assign
        sample = X32[:: max(1, n // 20000)].astype(np.float64)
        sample -= sample.mean(axis=0)
        _, V = np.linalg.eigh(sample.T @ sample)
        V = np.ascontiguousarray(V[:, ::-1][:, :r], dtype=np.float32)
        Xp = np.ascontiguousarray(X32 @ V)

        def assign(U32: np.ndarray, U64: np.ndarray = None) -> np.ndarray:
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
