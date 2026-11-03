"""Soft-anneal construction: minimax's cover, then a continuous relaxation of
J = ln sum_i sigma(theta_i) minimised by annealed Riemannian Adam on the directions.

Volume-only: no query threshold or member counts enter the annealed objective or the start cover.

Relaxation (all on the n x m top-m nearest-cap lists, m = `topm`):
  D_xi        angle(x, u_i)
  R_xi        softmax_i(-D_xi / Ta)                    soft membership  (Ta annealed to a small value)
  theta_i     Tm * [ LSE_x(ln R_xi + D_xi/Tm) - LSE_x(ln R_xi) ]   weighted soft-max of
              member angles (-> max over members as Tm -> 0, R -> hard)
  J_soft      (1/g) * LSE_i( g * ln sigma(theta_i) )       cap-level temperature:
              g = p_eff / (d-1); p_eff = 30 at the start (a flat power cost, like sphere
              merge's), annealed to d-1 (the true measure).
The analytic gradient was checked against finite differences and torch autograd.

SoftAnnealMinimax refines after the filter has chosen its encoding levels and trained
its codebooks. Gradients use decoded directions with a straight-through approximation
to the discrete encoding. Checkpoints fit codes, reassign members, fit again to those
assignments and score the actual rounded stored angles. The original encoded minimax
cover is a candidate, so the stored unexpanded-volume objective cannot get worse.
Cap count, encoding levels and codebooks stay fixed. Query tau is never used.

Speed. With the extension built (CapCoverRustFilter, src/construct_anneal.rs) SoftAnneal runs fast=True by
default; fast=False is the NumPy reference and stays the specification. The fast path computes the same
quantities: fused kernels for angles, softmax, per-cap online LSE, gradient coefficients and the
scatter G[c] += coef * x (SoftState._value_and_grad_fast); CapSearch, which answers the candidate and
nearest-cap searches from a cache of each vector's 48 nearest caps plus an exactness certificate
(triangle inequality on how far the caps moved) and falls back to a full IndexFlatIP search for the
rows it cannot certify; and, for TurboQuant checkpoints, the settling/code-fit loops in Rust
(tight_angles_fast, fit_scalar_codes_fast, equal to the filter's own). SoftAnneal(approx_age=a) is an
opt-in approximation (default off): candidate lists come from the cache without the certificate and
the cache is rebuilt only every `a` refreshes.

The standalone SoftAnneal.refine also supports continuous refinement: without projection
or a supplied objective it keeps the best exact nearest-assignment cap volume.
"""
import numpy as np
import faiss
import scipy.sparse as sp
from typing import Callable, Optional

try:
    import capcover_rs as _rs
    HAVE_FAST = all(hasattr(_rs, f) for f in ("anneal_stage1", "anneal_stage2", "assigned_theta", "cache_topm", "tight_angles", "fit_scalar_levels"))
except ImportError:  # extension not built: NumPy reference path only
    _rs, HAVE_FAST = None, False
HAVE_ADAM = HAVE_FAST and hasattr(_rs, "adam_step")

from filters.filter import Filter
from filters.cap_cover.cap_interfaces import ConstructInterface, EncodedCover
from filters.cap_cover.cap_geometry import (angle_between, log_cap_measure,
                                            nearest_directions, segments, segment_max, unit_rows)
from filters.cap_cover.minimax_construct import MinimaxConstruct, badoiu_clarkson, reseed_empty


def _lse(a: np.ndarray, axis=None):
    m = np.max(a, axis=axis, keepdims=True)
    out = np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True)) + m
    return out if axis is None else np.squeeze(out, axis=axis)


def lnsig_and_grad(psi: np.ndarray, dim: int):
    """ln sigma(psi) (table, float64) and its derivative (central difference of the table)."""
    psi = np.clip(psi, 1e-6, np.pi - 1e-6)
    h = 1e-4
    f = log_cap_measure(psi, dim)
    df = (log_cap_measure(np.minimum(psi + h, np.pi), dim) - log_cap_measure(np.maximum(psi - h, 1e-6), dim)) \
        / (np.minimum(psi + h, np.pi) - np.maximum(psi - h, 1e-6))
    return f, df


def hard_eval(X: np.ndarray, U: np.ndarray, dim: int, search: Optional["CapSearch"] = None):
    """(J_hard, theta, labels, counts): nearest-direction assignment, exact angles; empty caps ignored.

    With `search` (a CapSearch over the float32 copy of X) the nearest caps come from its certified
    cache and the per-cap maxima from a fused float64 kernel; the result is the same as without."""
    if search is not None and search.enabled and X.dtype == np.float64 and X.flags.c_contiguous:
        U = np.ascontiguousarray(U, dtype=np.float64)
        labels = search.nearest(U)
        counts = np.bincount(labels, minlength=len(U))
        theta = _rs.assigned_theta(X, U, labels)
    else:
        _, labels = nearest_directions(X, U)
        labels = labels.astype(np.int64)
        theta_x = angle_between(X, U[labels])
        order, starts, counts = segments(labels, len(U))
        theta = segment_max(theta_x[order], starts, counts)
    live = counts > 0
    J = float(_lse(log_cap_measure(theta[live], dim)).item())
    return J, theta, labels, counts


class CapSearch:
    """Exact nearest-cap lists for a slowly moving set of directions, without a full search each time.

    A full IndexFlatIP search of all n vectors against k caps is the dominant cost of annealing
    (n k d multiply-adds). After one search keeping the nearest `cache` caps of every vector (with
    the cap directions it used), a later query for moved directions U needs only those cached caps,
    re-scored, plus the few caps that moved the most: a cap outside both sets was at angle
    >= D_M(x) from x at search time, so it is now at angle >= D_M(x) - move(cap) >= D_M(x) - thr,
    thr = the largest move among the caps not re-scored explicitly. If the m-th best re-scored angle
    is below that bound, no uncached cap can be among the true top-m: the list is exact (up to
    ties). Vectors that fail the certificate are searched in full; when many fail, the cache is
    rebuilt. The result is the list a full search returns, so annealing is unchanged.

    exact=False is an opt-in approximation: no per-row certificate, the cache is rebuilt every
    `max_age` queries, and uncertified rows keep the best caps found among the cached and explicit
    ones.
    """

    def __init__(self, X32: np.ndarray, cache: int = 48, explicit_frac: float = 0.02,
                 rebuild_below: float = 0.9, margin: float = 2e-4, block: int = 16384,
                 enabled: bool = True, exact: bool = True, max_age: int = 3) -> None:
        self.X32 = X32
        self.cache = cache
        self.explicit_frac = explicit_frac
        self.rebuild_below = rebuild_below
        self.margin = margin
        self.block = block
        self.enabled = enabled and HAVE_FAST
        self.exact, self.max_age, self.age = exact, max_age, 0
        self.idx = None
        self.stats = {"full": 0, "rebuild": 0, "cached": 0, "rows_full": 0, "rows": 0}

    def _full(self, rows, U32: np.ndarray, width: int):
        index = faiss.IndexFlatIP(U32.shape[1])
        index.add(U32)
        Xq = self.X32 if rows is None else np.ascontiguousarray(self.X32[rows])
        ip, lab = index.search(Xq, min(width, len(U32)))
        return ip, lab.astype(np.int64)

    def _build(self, U: np.ndarray, U32: np.ndarray, m: int):
        ip, lab = self._full(None, U32, max(self.cache, m))
        self.stats["full"] += 1
        self.idx, self.Uref, self.age = lab, U.copy(), 0
        self.dm = np.arccos(np.clip(ip[:, -1].astype(np.float64), -1.0, 1.0))
        return np.ascontiguousarray(lab[:, :m]), np.ascontiguousarray(ip[:, :m])

    def topm(self, U: np.ndarray, m: int):
        """(cand n x m int64, ip n x m float32), caps by descending float32 inner product."""
        k = len(U)
        m = min(m, k)
        U32 = np.ascontiguousarray(U, dtype=np.float32)
        n = len(self.X32)
        self.stats["rows"] += n
        if not self.enabled or k <= self.cache + m:
            self.stats["rows_full"] += n
            ip, lab = self._full(None, U32, m)
            return lab, ip
        if self.idx is None or self.idx.shape[1] < m:
            self.stats["rows_full"] += n
            return self._build(U, U32, m)
        move = angle_between(U, self.Uref)
        nb = int(self.explicit_frac * k)
        if nb > 0:
            bids = np.argpartition(-move, nb - 1)[:nb].astype(np.int64)
            in_b = np.zeros(k, dtype=np.uint8)
            in_b[bids] = 1
            thr = float(np.max(np.where(in_b == 1, -np.inf, move)))
        else:
            bids = np.empty(0, dtype=np.int64)
            in_b = np.zeros(k, dtype=np.uint8)
            thr = float(move.max())
        cand = np.empty((n, m), dtype=np.int64)
        ip = np.empty((n, m), dtype=np.float32)
        ok = np.empty(n, dtype=np.uint8)
        UB = np.ascontiguousarray(U32[bids])
        for s in range(0, n, self.block):
            e = min(s + self.block, n)
            ipb = (self.X32[s:e] @ UB.T) if nb > 0 else np.empty((e - s, 0), dtype=np.float32)
            cand[s:e], ip[s:e], ok[s:e] = _rs.cache_topm(
                self.X32[s:e], U32, self.idx[s:e], bids, np.ascontiguousarray(ipb), in_b, m,
                np.ascontiguousarray(self.dm[s:e] - thr), self.margin)
        good = ok.astype(bool)
        if not self.exact:
            # Opt-in approximation: answer from the cache (uncertified rows included) until it is
            # `max_age` queries old; the lists may then miss a true top-m cap.
            self.age += 1
            if self.age > self.max_age:
                self.stats["rebuild"] += 1
                self.stats["rows_full"] += n
                return self._build(U, U32, m)
            self.stats["cached"] += 1
            return cand, ip
        if good.mean() < self.rebuild_below:
            self.stats["rebuild"] += 1
            self.stats["rows_full"] += n
            return self._build(U, U32, m)
        bad = np.flatnonzero(~good)
        if len(bad):
            self.stats["rows_full"] += len(bad)
            ipf, labf = self._full(bad, U32, m)
            cand[bad], ip[bad] = labf, ipf
        self.stats["cached"] += 1
        return cand, ip

    def nearest(self, U: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(self.topm(U, 1)[0][:, 0])


class SoftState:
    """Sparse relaxation on top-m cap lists. Holds no copies of X."""

    def __init__(self, X32: np.ndarray, dim: int, topm: int, Ta: float, Tm: float, p_eff: float,
                 fast: Optional[bool] = None, approx_age: Optional[int] = None):
        self.X32, self.dim, self.topm = X32, dim, topm
        self.fast = HAVE_FAST if fast is None else (fast and HAVE_FAST)
        self.searcher = CapSearch(X32, enabled=self.fast, exact=approx_age is None,
                                  max_age=approx_age if approx_age is not None else 3)
        self.Ta, self.Tm, self.p_eff = Ta, Tm, p_eff
        self.gemm_max_k = 500

    def search(self, U: np.ndarray) -> np.ndarray:
        if self.fast:
            return self.searcher.topm(U, self.topm)[0]
        index = faiss.IndexFlatIP(U.shape[1])
        index.add(np.ascontiguousarray(U, dtype=np.float32))
        _, cand = index.search(self.X32, min(self.topm, len(U)))
        return cand.astype(np.int64)

    def angles(self, U: np.ndarray, cand: np.ndarray, block: int = 2048) -> np.ndarray:
        U32 = U.astype(np.float32)
        n, m = cand.shape
        if len(U) <= self.gemm_max_k:
            ip = np.take_along_axis(self.X32 @ U32.T, cand, axis=1)
        else:
            ip = np.empty((n, m), dtype=np.float32)
            for s in range(0, n, block):
                ip[s:s + block] = np.einsum("bd,bmd->bm", self.X32[s:s + block], U32[cand[s:s + block]])
        return np.arccos(np.clip(ip.astype(np.float64), -1.0, 1.0))

    def value_and_grad(self, U: np.ndarray, cand: np.ndarray, want_grad: bool = True):
        """(J_soft, dJ/dU (Euclidean, k x d), theta_soft)."""
        if self.fast:
            return self._value_and_grad_fast(U, cand, want_grad)
        n, m = cand.shape
        k = len(U)
        Ta, Tm = self.Ta, self.Tm
        D = self.angles(U, cand)                               # n x m
        lnZ = _lse(-D / Ta, axis=1)                            # n
        lnR = -D / Ta - lnZ[:, None]
        R = np.exp(lnR)
        L = lnR + D / Tm
        # per-cap LSE over the (x, slot) entries with that cap: segment LSE via sorting by cap id
        flat_c = cand.ravel()
        order = np.argsort(flat_c, kind="stable")
        sc = flat_c[order]
        counts = np.bincount(sc, minlength=k)
        starts = np.zeros(k, dtype=np.int64)
        np.cumsum(counts[:-1], out=starts[1:])
        nonempty = counts > 0

        def seg_lse(v_sorted):
            mx = np.full(k, -np.inf)
            mx[nonempty] = np.maximum.reduceat(v_sorted, starts[nonempty])
            e = np.exp(v_sorted - mx[sc])
            s = np.zeros(k)
            s[nonempty] = np.add.reduceat(e, starts[nonempty])
            with np.errstate(divide="ignore"):
                return np.log(s) + mx, e, s

        Ls, lnRs = L.ravel()[order], lnR.ravel()[order]
        lseL, eL, sL = seg_lse(Ls)
        lseR, eR, sR = seg_lse(lnRs)
        theta = np.zeros(k)
        theta[nonempty] = Tm * (lseL[nonempty] - lseR[nonempty])
        theta = np.clip(theta, 0.0, np.pi - 1e-3)
        g = self.p_eff / (self.dim - 1.0)
        f, df = lnsig_and_grad(theta, self.dim)
        live = nonempty & (sR > 1e-12)
        fl = np.where(live, f, -np.inf)
        J = float((_lse(g * fl) / g).item())
        if not want_grad:
            return J, None, theta
        s_i = np.exp(g * fl - _lse(g * fl))                    # softmax over caps
        gth = np.where(live, s_i * df, 0.0)                    # dJ/dtheta_i
        a_s = eL / np.maximum(sL[sc], 1e-300)                  # a_xi (sorted order)
        b_s = eR / np.maximum(sR[sc], 1e-300)
        a = np.empty_like(a_s); b = np.empty_like(b_s)
        a[order] = a_s; b[order] = b_s
        a = a.reshape(n, m); b = b.reshape(n, m)
        E = gth[cand] * Tm * (a - b)                           # n x m
        dD = gth[cand] * a - E / Ta + R / Ta * E.sum(axis=1, keepdims=True)
        sinD = np.maximum(np.sin(D), 0.02)
        coef = -dD / sinD                                      # dJ/dU_j = sum coef_xj X_x
        M = sp.csr_matrix((coef.ravel().astype(np.float32), (flat_c, np.repeat(np.arange(n), m))), shape=(k, n))
        G = (M @ self.X32).astype(np.float64)
        return J, G, theta

    _NO_ORDER = np.empty(0, dtype=np.int64)   # kernels visit rows in order (a row permutation measured no faster)

    def _order(self, cand: np.ndarray) -> np.ndarray:
        return self._NO_ORDER

    def _value_and_grad_fast(self, U: np.ndarray, cand: np.ndarray, want_grad: bool = True):
        """Same maths as the reference, with the O(n m d) and O(n m) parts fused in Rust:
        angles, row softmax and per-cap online LSE in one pass; coefficients and the
        scatter G[c] += coef * x in a second (no sorting, no sparse matrix)."""
        k = len(U)
        Ta, Tm = self.Ta, self.Tm
        cand = np.ascontiguousarray(cand, dtype=np.int64)
        U32 = np.ascontiguousarray(U, dtype=np.float32)
        D, lnR, cap = _rs.anneal_stage1(self.X32, U32, cand, self._order(cand), Ta, Tm)
        mxl, sl, mxr, sr = cap
        nonempty = sl > 0
        theta = np.zeros(k)
        with np.errstate(divide="ignore"):
            lseL, lseR = np.log(sl) + mxl, np.log(sr) + mxr
        theta[nonempty] = Tm * (lseL[nonempty] - lseR[nonempty])
        theta = np.clip(theta, 0.0, np.pi - 1e-3)
        g = self.p_eff / (self.dim - 1.0)
        f, df = lnsig_and_grad(theta, self.dim)
        live = nonempty & (sr > 1e-12)
        fl = np.where(live, f, -np.inf)
        lse_fl = _lse(g * fl)
        J = float((lse_fl / g).item())
        if not want_grad:
            return J, None, theta
        gth = np.ascontiguousarray(np.where(live, np.exp(g * fl - lse_fl) * df, 0.0))
        G = _rs.anneal_stage2(self.X32, cand, self._order(cand), D, lnR, cap, gth, Ta, Tm)
        return J, G, theta


class SoftAnneal:
    """Annealing refinement with the original continuous annealing schedule.

    Quantized refinement uses this schedule with decoded-direction gradients
    and an exact encoded objective at checkpoints; see the module docstring.
    """

    def __init__(self, iters: int = 300, Ta0: float = 0.1, Ta1: float = 0.002, tm_ratio: float = 1.0,
                 p0: float = 30.0, p1: Optional[float] = None, lr0: float = 0.1, lr1: float = 0.005,
                 topm: int = 16, refresh: int = 4, eval_every: int = 10,
                 bc_steps: int = 64, beta1: float = 0.9, beta2: float = 0.99,
                 fast: Optional[bool] = None, approx_age: Optional[int] = None) -> None:
        """fast: use the Rust kernels and the certified neighbour cache (default: when the extension
        is built). fast=False is the NumPy reference. approx_age (default None = off): opt-in
        approximation, the annealing candidate lists come from a cache rebuilt only every
        `approx_age` refreshes, without the exactness certificate (hard objectives stay exact)."""
        self.approx_age = approx_age
        self.fast = HAVE_FAST if fast is None else (fast and HAVE_FAST)
        self.iters = iters
        self.Ta0, self.Ta1, self.tm_ratio = Ta0, Ta1, tm_ratio
        self.p0, self.p1 = p0, p1
        self.lr0, self.lr1 = lr0, lr1
        self.topm, self.refresh, self.eval_every = topm, refresh, eval_every
        self.bc_steps = bc_steps
        self.beta1, self.beta2 = beta1, beta2

    def refine(self, X: np.ndarray, U0: np.ndarray,
               objective: Optional[Callable[[np.ndarray], float]] = None,
               project: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> np.ndarray:
        """Anneal directions, retaining the best checkpoint under `objective`.

        With `project`, evaluate smooth gradients at decoded directions and pass
        them to the latent directions (straight-through); discrete acceptance
        still uses the supplied hard objective, not the smooth surrogate.
        """
        n, dim = X.shape
        U0 = unit_rows(U0)
        U = U0.copy()
        k = len(U)
        X32 = np.ascontiguousarray(X, dtype=np.float32)
        p1 = self.p1 if self.p1 is not None else dim - 1.0
        st = SoftState(X32, dim, self.topm, self.Ta0, self.Ta0 * self.tm_ratio, self.p0, fast=self.fast,
                       approx_age=self.approx_age)
        # Hard-objective cache over the latent directions (an exact cache; shared with the
        # annealing search when both look at the same directions).
        if X.dtype != np.float64 or not st.fast:
            ev = None
        elif project is None and self.approx_age is None:
            ev = st.searcher
        else:
            ev = CapSearch(X32)
        if objective is None:
            score = lambda V: hard_eval(X, V, dim, ev)[0]
        elif getattr(objective, "takes_search", False):
            score = lambda V: objective(V, ev)
        else:
            score = objective
        best_J = score(U0)
        best_U = U0.copy()
        m1 = np.zeros_like(U)
        v = np.zeros(k)
        cand = None
        for it in range(self.iters):
            f = it / max(self.iters - 1, 1)
            st.Ta = self.Ta0 * (self.Ta1 / self.Ta0) ** f
            st.Tm = st.Ta * self.tm_ratio
            st.p_eff = self.p0 * (p1 / self.p0) ** f
            lr = self.lr0 * (self.lr1 / self.lr0) ** f
            decoded = U if project is None else project(U)
            if cand is None or it % self.refresh == 0:
                cand = st.search(decoded)
            _, G, _ = st.value_and_grad(decoded, cand)
            if not np.isfinite(G).all():
                break  # preserve the best valid checkpoint if the relaxation fails
            # Riemannian Adam (per-cap scalar second moment): tangent gradient, step in radians
            if self.fast and HAVE_ADAM:
                U = _rs.adam_step(np.ascontiguousarray(G), U, m1, v, lr, self.beta1, self.beta2,
                                  1 - self.beta1 ** (it + 1), 1 - self.beta2 ** (it + 1))
            else:
                G = G - np.sum(G * U, axis=1, keepdims=True) * U
                gn2 = np.sum(G * G, axis=1)
                m1 = self.beta1 * m1 + (1 - self.beta1) * G
                v = self.beta2 * v + (1 - self.beta2) * gn2
                mhat = m1 / (1 - self.beta1 ** (it + 1))
                vhat = v / (1 - self.beta2 ** (it + 1))
                step = mhat / (np.sqrt(vhat)[:, None] + 1e-30)
                Un = unit_rows(U - lr * step)
                # parallel-transport the momentum approximately by re-projecting it
                m1 = m1 - np.sum(m1 * Un, axis=1, keepdims=True) * Un
                U = Un
            if (it + 1) % self.eval_every == 0 or it == self.iters - 1:
                Jh = score(U)
                if Jh < best_J:
                    best_J, best_U = Jh, U.copy()
        _, Up = self._polish(X, best_U, dim)
        Jp = score(Up)
        return Up if Jp < best_J else best_U

    def _polish(self, X, U, dim):
        """One minimax round: nearest reassignment, BC, reseed."""
        _, labels = nearest_directions(X, U)
        order, starts, counts = segments(labels, len(U))
        Up = badoiu_clarkson(X[order], labels[order], U, starts, counts, self.bc_steps)
        Up = reseed_empty(X, Up, counts, labels)
        Jp, _, _, _ = hard_eval(X, Up, dim)
        return Jp, Up


def tight_angles_fast(filter, X, U, decoded, baseline, home, search: CapSearch):
    """CapCoverFilter._tight_angles for a cover without `neighbors`: the nearest decoded caps come
    from `search` (certified cache) and the settling loop runs in Rust."""
    n_caps = len(decoded)
    if n_caps == 0 or len(X) == 0:
        return np.zeros(n_caps), np.zeros(len(X), dtype=np.int64)
    bound = np.asarray(baseline, dtype=np.float64) + angle_between(U, decoded)
    labels = search.topm(np.ascontiguousarray(decoded), min(filter._COVERAGE_NEIGHBORS, n_caps))[0]
    return _rs.tight_angles(X, np.ascontiguousarray(decoded), np.ascontiguousarray(labels),
                            np.ascontiguousarray(home, dtype=np.int64), np.ascontiguousarray(bound))


def fit_scalar_codes_fast(sq, U, X, labels, passes: int = 2) -> np.ndarray:
    """cap_quantizers._fit_scalar_codes with the coordinate search in Rust (identical arithmetic)."""
    D = np.ascontiguousarray(U, dtype=np.float32)
    C = sq.decode(sq.compute_codes(D)).astype(np.float64)
    if len(X) == 0 or len(C) == 0:
        return sq.compute_codes(np.ascontiguousarray(C, dtype=np.float32))
    n_levels = 1 << int(round(np.log2(len(faiss.vector_to_array(sq.trained)) + 1) - 1))
    levels = np.sort(faiss.vector_to_array(sq.trained)[:n_levels].astype(np.float64))
    idx = np.clip(np.searchsorted(levels, C - 1e-12), 0, n_levels - 1).astype(np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    order = np.argsort(labels, kind="stable")
    caps, starts = np.unique(labels[order], return_index=True)
    Xo, lo = np.ascontiguousarray(X[order], dtype=np.float64), labels[order]
    dot = np.einsum("nd,nd->n", Xo, C[lo])
    norm2 = np.einsum("kd,kd->k", C, C)
    C = _rs.fit_scalar_levels(Xo, np.ascontiguousarray(starts, dtype=np.int64), np.ascontiguousarray(caps, dtype=np.int64),
                              np.ascontiguousarray(C), np.ascontiguousarray(idx), np.ascontiguousarray(dot),
                              np.ascontiguousarray(norm2), np.ascontiguousarray(levels), passes)
    return sq.compute_codes(np.ascontiguousarray(C, dtype=np.float32))


def fit_codes_by_level_fast(filter, U, X, owner, levels) -> np.ndarray:
    """CapCoverFilter._fit_codes_by_level, with TurboQuant's scalar fit in Rust."""
    codes = np.zeros((len(U), int(filter._code_sizes.max())), dtype=np.uint8)
    for level in np.unique(levels):
        level = int(level)
        caps = np.flatnonzero(levels == level)
        local = np.full(len(U), -1, dtype=np.int64)
        local[caps] = np.arange(len(caps))
        rows = np.flatnonzero(local[owner] >= 0)
        sq = filter._quantizers[level]
        if filter.fit_codes and hasattr(sq, "fit_codes"):
            c = sq.fit_codes(U[caps], X[rows], local[owner[rows]], filter.fit_candidates, filter.fit_passes)
        elif filter.fit_codes and filter._is_turboquant(level):
            c = fit_scalar_codes_fast(sq, U[caps], X[rows], local[owner[rows]], filter.fit_passes)
        else:
            c = filter.encode(U[caps], level)
        codes[caps, :c.shape[1]] = c
    return codes


def encode_cover_fast(filter, X, U, theta, home, levels, search: CapSearch) -> EncodedCover:
    """CapCoverFilter._encode_cover (cover without neighbours) with the fast settling."""
    codes = fit_codes_by_level_fast(filter, U, X, home, levels)
    angles, owner = tight_angles_fast(filter, X, U, filter._decode_levels(codes, levels), theta, home, search)
    return EncodedCover(U, codes, angles, owner)


class SoftAnnealMinimax(ConstructInterface):
    """Minimax followed by annealing after encoding selection and reassignment."""

    def __init__(self, anneal: Optional[SoftAnneal] = None, minimax: Optional[MinimaxConstruct] = None,
                 reassign_passes: int = 2) -> None:
        if reassign_passes < 1:
            raise ValueError("reassign_passes must be positive")
        self.anneal = anneal or SoftAnneal()
        self.minimax = minimax or MinimaxConstruct()
        self.reassign_passes = reassign_passes

    def construct_filter(self, filter: Filter, vectors: np.ndarray):
        return self.minimax.construct_filter(filter, vectors)

    def iter_candidates(self, filter: Filter, vectors: np.ndarray, ks, weights=None):
        """Minimax candidates; annealing waits for the final encoding plan."""
        return self.minimax.iter_candidates(filter, vectors, ks, weights)

    def refine_encoded(self, filter: Filter, vectors: np.ndarray,
                       cover: EncodedCover, levels: np.ndarray) -> EncodedCover:
        if len(cover.directions) == 0 or len(cover.directions) >= len(vectors):
            return cover
        X = np.ascontiguousarray(vectors, dtype=np.float64)
        best, best_score = cover, filter._encoded_volume(cover.angles)
        # Decoded directions get their own certified neighbour cache (tight-angle settling needs
        # each member's nearest decoded caps).
        age = getattr(self.anneal, "approx_age", None)
        decoded_search = CapSearch(np.ascontiguousarray(X, dtype=np.float32), exact=age is None,
                                   max_age=age if age is not None else 3) \
            if getattr(self.anneal, "fast", False) and HAVE_FAST else None

        def objective(U, search=None):
            nonlocal best, best_score
            _, theta, owner, _ = hard_eval(X, U, X.shape[1], search)
            fast = search is not None and decoded_search is not None
            candidate_score = np.inf
            for _ in range(self.reassign_passes):
                if fast:
                    candidate = encode_cover_fast(filter, X, U, theta, owner, levels, decoded_search)
                else:
                    candidate = filter._encode_cover(X, U, theta, owner, levels)
                value = filter._encoded_volume(candidate.angles)
                candidate_score = min(candidate_score, value)
                if value < best_score:
                    best, best_score = candidate, value
                owner = candidate.owner
            return candidate_score

        objective.takes_search = True

        def project(U):
            codes = np.zeros((len(U), int(filter._code_sizes.max())), dtype=np.uint8)
            for level in np.unique(levels):
                caps = np.flatnonzero(levels == level)
                c = filter.encode(U[caps], int(level))
                codes[caps, :c.shape[1]] = c
            return filter._decode_levels(codes, levels)

        self.anneal.refine(X, cover.directions, objective=objective, project=project)
        return best

    def get_bytesize(self) -> int:
        return 0
