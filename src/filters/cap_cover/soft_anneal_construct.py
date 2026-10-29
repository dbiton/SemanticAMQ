"""Soft-anneal construction: minimax's cover, then a continuous relaxation of
J = ln sum_i sigma(theta_i + alpha) minimised by annealed Riemannian Adam on the directions.

Volume-only: no member counts enter the annealed objective or the start cover.

Relaxation (all on the n x m top-m nearest-cap lists, m = `topm`):
  D_xi        angle(x, u_i)
  R_xi        softmax_i(-D_xi / Ta)                    soft membership  (Ta annealed to a small value)
  theta_i     Tm * [ LSE_x(ln R_xi + D_xi/Tm) - LSE_x(ln R_xi) ]   weighted soft-max of
              member angles (-> max over members as Tm -> 0, R -> hard)
  J_soft      (1/g) * LSE_i( g * ln sigma(theta_i + alpha) )       cap-level temperature:
              g = p_eff / (d-1); p_eff = 30 at the start (a flat power cost, like sphere
              merge's), annealed to d-1 (the true measure).
The analytic gradient was checked against finite differences and torch autograd.

Safety: the hard objective J_hard(U) = ln sum_i sigma(max_{x -> i} angle(x,u_i) + alpha)
(nearest-direction assignment, exact float64 angles) is evaluated on the start and every
`eval_every` iterations; the best iterate (the start included) is kept, then polished with
a minimax round and kept only if its hard J is lower. The result is therefore
never worse than the start on the construct-set criterion, and its angles are measured from
the members (labelled_cap_cover), so the cover is valid.

SoftAnnealMinimax plugs this into CapCoverFilter (also through auto()): the candidates over the
k grid are minimax's covers and finish() anneals only the chosen one, once. The annealed objective is the unencoded cover's;
the filter re-measures angles after coding, so the gain has to survive quantization.
"""
import numpy as np
import faiss
import scipy.sparse as sp
from typing import Optional

from filters.filter import Filter
from filters.cap_cover.cap_interfaces import ConstructInterface, Cover
from filters.cap_cover.cap_geometry import (angle_between, labelled_cap_cover, log_cap_measure,
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


def hard_eval(X: np.ndarray, U: np.ndarray, alpha: float, dim: int):
    """(J_hard, theta, labels, counts): nearest-direction assignment, exact angles; empty caps ignored."""
    _, labels = nearest_directions(X, U)
    labels = labels.astype(np.int64)
    theta_x = angle_between(X, U[labels])
    order, starts, counts = segments(labels, len(U))
    theta = segment_max(theta_x[order], starts, counts)
    live = counts > 0
    psi = np.minimum(theta[live] + alpha, np.pi)
    J = float(_lse(log_cap_measure(psi, dim)).item())
    return J, theta, labels, counts


class SoftState:
    """Sparse relaxation on top-m cap lists. Holds no copies of X."""

    def __init__(self, X32: np.ndarray, dim: int, alpha: float, topm: int, Ta: float, Tm: float, p_eff: float):
        self.X32, self.dim, self.alpha, self.topm = X32, dim, alpha, topm
        self.Ta, self.Tm, self.p_eff = Ta, Tm, p_eff
        self.gemm_max_k = 500

    def search(self, U: np.ndarray) -> np.ndarray:
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
        theta[nonempty] = Tm * (lseL - lseR)[nonempty]
        theta = np.clip(theta, 0.0, np.pi - self.alpha - 1e-3)
        g = self.p_eff / (self.dim - 1.0)
        psi = theta + self.alpha
        f, df = lnsig_and_grad(psi, self.dim)
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


class SoftAnneal:
    """The annealing refinement; see the module docstring. Defaults are the frozen
    hyperparameters (chosen on the construct sets of five development cells, without queries)."""

    def __init__(self, iters: int = 300, Ta0: float = 0.1, Ta1: float = 0.002, tm_ratio: float = 1.0,
                 p0: float = 30.0, p1: Optional[float] = None, lr0: float = 0.1, lr1: float = 0.005,
                 topm: int = 16, refresh: int = 4, eval_every: int = 10, obj_tau: float = 0.1,
                 bc_steps: int = 64, beta1: float = 0.9, beta2: float = 0.99) -> None:
        self.iters = iters
        self.Ta0, self.Ta1, self.tm_ratio = Ta0, Ta1, tm_ratio
        self.p0, self.p1 = p0, p1
        self.lr0, self.lr1 = lr0, lr1
        self.topm, self.refresh, self.eval_every = topm, refresh, eval_every
        self.obj_tau, self.bc_steps = obj_tau, bc_steps
        self.beta1, self.beta2 = beta1, beta2

    def refine(self, filter: Filter, X: np.ndarray, U0: np.ndarray) -> np.ndarray:
        """Anneal from the unit directions U0 (k x d); returns directions whose hard
        construct-set objective is <= U0's."""
        n, dim = X.shape
        alpha = float(filter._beta(np.ones(1), self.obj_tau)[0][0])
        U0 = unit_rows(U0)
        U = U0.copy()
        k = len(U)
        X32 = np.ascontiguousarray(X, dtype=np.float32)
        p1 = self.p1 if self.p1 is not None else dim - 1.0
        best_J, _, _, _ = hard_eval(X, U0, alpha, dim)
        best_U = U0.copy()
        st = SoftState(X32, dim, alpha, self.topm, self.Ta0, self.Ta0 * self.tm_ratio, self.p0)
        m1 = np.zeros_like(U)
        v = np.zeros(k)
        cand = None
        for it in range(self.iters):
            f = it / max(self.iters - 1, 1)
            st.Ta = self.Ta0 * (self.Ta1 / self.Ta0) ** f
            st.Tm = st.Ta * self.tm_ratio
            st.p_eff = self.p0 * (p1 / self.p0) ** f
            lr = self.lr0 * (self.lr1 / self.lr0) ** f
            if cand is None or it % self.refresh == 0:
                cand = st.search(U)
            _, G, _ = st.value_and_grad(U, cand)
            # Riemannian Adam (per-cap scalar second moment): tangent gradient, step in radians
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
                Jh, _, _, _ = hard_eval(X, U, alpha, dim)
                if Jh < best_J:
                    best_J, best_U = Jh, U.copy()
        Jp, Up = self._polish(X, best_U, alpha, dim)
        return Up if Jp < best_J else best_U

    def _polish(self, X, U, alpha, dim):
        """One minimax round: nearest reassignment, BC, reseed."""
        _, labels = nearest_directions(X, U)
        order, starts, counts = segments(labels, len(U))
        Up = badoiu_clarkson(X[order], labels[order], U, starts, counts, self.bc_steps)
        Up = reseed_empty(X, Up, counts, labels)
        Jp, _, _, _ = hard_eval(X, Up, alpha, dim)
        return Jp, Up


class SoftAnnealMinimax(ConstructInterface):
    """Minimax's cover, annealed (see the module docstring). Takes anneal=SoftAnneal(...) and
    minimax=MinimaxConstruct(...)."""

    def __init__(self, anneal: Optional[SoftAnneal] = None, minimax: Optional[MinimaxConstruct] = None) -> None:
        self.anneal = anneal or SoftAnneal()
        self.minimax = minimax or MinimaxConstruct()

    def construct_filter(self, filter: Filter, vectors: np.ndarray):
        X = np.ascontiguousarray(vectors, dtype=np.float64)
        U, angles = self.minimax.construct_filter(filter, X)
        if len(U) == 0 or len(U) >= len(X):
            return U, angles                  # every vector is its own cap: nothing to anneal
        U, angles, _, _ = labelled_cap_cover(X, self.anneal.refine(filter, X, U))
        return U, angles

    def iter_candidates(self, filter: Filter, vectors: np.ndarray, ks, weights=None):
        """Minimax's covers at each k: only the chosen one is annealed, in finish()."""
        return self.minimax.iter_candidates(filter, vectors, ks, weights)

    def finish(self, filter: Filter, vectors: np.ndarray, cover: Cover) -> Cover:
        if len(cover.directions) == 0 or len(cover.directions) >= len(vectors):
            return cover
        X = np.ascontiguousarray(vectors, dtype=np.float64)
        U, angles, labels, kept = labelled_cap_cover(X, self.anneal.refine(filter, X, cover.directions))
        return Cover(U, angles, labels, kept=kept)    # neighbour lists dropped: the filter searches all caps

    def get_bytesize(self) -> int:
        return 0
