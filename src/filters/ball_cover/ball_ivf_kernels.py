"""
Compiled query kernels for BallIVFIndex.

The ball test, per (query q, ball j) with decoded center c_j and radius r_j:

    ||q - c_j||^2 <= R2_j = (r_j + tau + eps)^2  (no ball admits anything at a negative reach)

Everything is in the basis of BallCoverFilter's quantizer (past its rotation).

The index keeps no copy of the balls, only a compact sketch per ball: its
first h centred-PCA coordinates in int8 (one scale; every ball's rounding is
at most dl), its norm past them as a uint8 interval [t_lo, t_lo + st], and a
locator into BallCoverFilter's own code array (the radius is read from the
filter; per cell the largest is kept).  A batch of queries is answered in
chunks of CHUNK queries:

  1. per query: its head q_h (h*d multiply-adds, or one BLAS product for the
     batch), its norm past the head, and the IVF gate -- cell c is skipped
     if a lower bound on ||q - mu_c|| exceeds U_c + tau + eps,
     U_c = max_j (r_j + ||c_j - mu_c||) (triangle inequality);
  2. the head distances S = ||q_h - c^_h||^2 of every (query, ball) pair of
     the chunk, in one coordinate-major pass over the sketch: each int8 row
     is widened once and used by every query, four coordinates at a time
     (long, contiguous, vectorised loops);
  3. per query and cell, the lower bound
         ||q - c||^2 >= max(sqrt(S) - dl, 0)^2 + dist(||q_>=h||, [t_lo, t_lo + st])^2 (*)
     (reverse triangle inequality on the rest) against the cell's largest R2,
     vectorised; the few balls under it are then held against their own R2:
     a ball above it provably does not accept;
  4. each survivor is decided exactly on BallCoverFilter's own code (4-bit
     nibbles through the quantizer's decode table, or float32) as soon as
     step 3 finds it, in storage order -- largest ball first within a cell,
     so an accepting query is nearly always settled by its first survivor:
     accepted or rejected unless within the float margin, or the
     bound rq on the query's rotation error (see BallIVFIndex.build), of the
     boundary, which leaves the query undecided -- the caller
     (BallCoverFilter) then decides it with its own scan.

Every float comparison of steps 1-3 is widened by REL * R2 + ABS terms (and
the reach by rq |q|), towards "keep", so no ball that accepts q is ever
dropped.  Performance notes (numba): member loops index with uint64 offsets,
which skips the negative-index wraparound check that blocks vectorisation;
per-ball lookups (radius) stay out of the vectorised loops, which they would
serialise.
"""

import numpy as np
from numba import njit, prange

_FM = {"reassoc", "contract", "nsz"}     # lets the float32 sums vectorize; no nnan/ninf

REL = 1e-4                               # relative margin on float32 sums (their error is <= ~1e-5)
ABS = 1e-5                               # absolute margin, times the pair's norm scale
PAD = 8                                  # cells are padded to a multiple of PAD balls
CHUNK = 32                               # queries per chunk (shares each sketch row between them)
BLOCK = 1024                             # sketch rows per pass of step 2 (stays in L1/L2)
BLAS_MIN = 32                            # batches this large get their heads from one BLAS product
KIND_FLAT = 0                            # float32 codes (BallCoverFilter's FP32 centers, or a copy)
KIND_NIBBLE = 1                          # 4-bit scalar-quantizer codes and their decode table


# ---------------------------------------------------------------------------
# Exact test (step 4)
# ---------------------------------------------------------------------------

@njit(nogil=True, cache=True, inline="always")
def _verdict(d2, reach, dq, tol):
    """Accept iff ||q - c|| + dq <= reach, reject iff ||q - c|| - dq > reach
    (each with a relative margin tol on the squares), else undecided (2)."""
    lo = reach - dq
    hi = reach + dq
    if lo >= 0.0 and d2 <= lo * lo * (1.0 - tol):
        return 1
    if d2 > hi * hi * (1.0 + tol):
        return 0
    return 2


@njit(nogil=True, cache=True, fastmath=_FM, inline="always")
def _exact(q32, te, dq, g, kind, Xf, cbuf, T, rad, idm, tol):
    """1 = accept, 0 = reject, 2 = undecided, for code row g of the source:
    BallCoverFilter's rule ||q' - decode(c)|| <= r + tau + eps, for every q'
    within dq of q (the filter's own rotation of the query may differ from q
    by that much): decided only if the rule holds, or fails, for all of them.
    The distance is summed in float32 (vectorised); its rounding, in any
    order (d + 3) 2^-24 ||q - c||^2 at most (q and the decoded center are
    float32 exactly), is inside the margin passed to _verdict."""
    reach = np.float64(rad[idm[g]]) + te
    if reach < 0.0:
        return 0
    if reach == np.inf:
        return 1
    d = q32.shape[0]
    s2 = np.float32(0.0)
    if kind == KIND_FLAT:
        for t in range(np.uint64(d)):
            e = q32[t] - Xf[g, t]
            s2 += e * e
    else:
        row = cbuf[g]
        for t in range(np.uint64(d)):
            y = np.int64(row[t >> np.uint64(1)])
            e = q32[t] - T[t, (y >> (4 * np.int64(t & np.uint64(1)))) & 15]
            s2 += e * e
    return _verdict(np.float64(s2), reach, dq, tol + 2.0 * (d + 4) * 2.0 ** -24)


# ---------------------------------------------------------------------------
# Per-query setup (step 1)
# ---------------------------------------------------------------------------

@njit(nogil=True, cache=True, inline="always")
def _rsq(r, te, u2, am2):
    """(r + tau + eps)^2 in code units^2, widened by the float margin; -1 at
    a negative reach (admits nothing)."""
    v = np.float64(r) + te
    if v < 0.0:
        return np.float32(-1.0)
    r2 = v * v * u2
    return np.float32(r2 + REL * r2 + am2)


@njit(nogil=True, cache=True, fastmath=_FM, inline="always")
def _prep(q64, Eh, iu, qh, Yh, i):
    """The query's head (qh, code units, float32), from Yh[i] when the host
    computed the batch's heads, else here; returns (its norm past the head
    [code units], |q|^2)."""
    d = q64.shape[0]
    h = qh.shape[0]
    qn2 = 0.0
    for k in range(d):
        qn2 += q64[k] * q64[k]
    if Yh.shape[0] > 0:
        yh = Yh[i]
    else:
        yh = np.zeros(h, np.float64)
        for k in range(d):
            qk = q64[k]
            er = Eh[k]
            for t in range(h):
                yh[t] += qk * np.float64(er[t])
    run = 0.0
    for t in range(h):
        run += yh[t] * yh[t]
        qh[t] = yh[t] * iu
    return np.sqrt(max(qn2 - run, 0.0)) * iu, qn2


@njit(nogil=True, cache=True, fastmath=_FM, inline="always")
def _gate(qh, qg, u2, am2, Mh, Mt, cellU, te, gate):
    """IVF gate in code units: gate[c] = the lower bound on ||q - mu_c||^2 is
    within (U_c + tau + eps)^2.  Returns how many cells pass."""
    h = qh.shape[0]
    n = 0
    for c in range(Mh.shape[0]):
        u = cellU[c] + te # ||q - mu_c|| <= U_c + tau + eps
        ok = False
        if u >= 0.0:
            g2 = u * u * u2
            acc = np.float32(0.0)
            for t in range(h):
                z = qh[t] - Mh[c, t]
                acc += z * z
            dt = qg - Mt[c]
            ok = acc + dt * dt <= g2 + REL * g2 + am2
        gate[c] = ok
        n += np.int64(ok)
    return n


# ---------------------------------------------------------------------------
# One chunk of queries (steps 1-4)
# ---------------------------------------------------------------------------

@njit(nogil=True, cache=True, fastmath=_FM)
def _chunk(c0, c1, live, out, und, Q64, Yh, te, Eh, iu, stu, xmax2, XC, TQ, dl, RM, fmap, Mh, Mt,
           cellU, cs, cn, kind, Xf, cbuf, T, rad, idm, tol, rq, S):
    """Queries [c0, c1).  S: scratch of at least c1 - c0 rows of Np floats
    (the head distances)."""
    Tq = c1 - c0
    d = Q64.shape[1]
    h = Eh.shape[1]
    Np = XC.shape[1]
    K = Mh.shape[0]
    u2 = iu * iu
    zero = np.float32(0.0)
    stf = np.float32(stu)

    # 1. per query: head, tails, margins, gate
    qh = np.empty((Tq, h), np.float32)
    qg = np.empty(Tq, np.float32)
    am = np.empty(Tq) # absolute float margin (code units^2)
    dqa = np.empty(Tq) # bound on the query's rotation error
    gate = np.zeros((Tq, K), np.bool_)
    need = np.zeros(K, np.bool_)
    act = np.zeros(Tq, np.bool_)
    for a in range(Tq):
        i = c0 + a
        if not live[i]:
            continue
        g_, qn2 = _prep(Q64[i], Eh, iu, qh[a], Yh, i)
        qg[a] = g_
        am[a] = ABS * (qn2 + xmax2) * u2
        dqa[a] = rq * np.sqrt(qn2)
        if _gate(qh[a], g_, u2, am[a], Mh, Mt, cellU, te + dqa[a], gate[a]) > 0:
            act[a] = True
            for c in range(K):
                need[c] |= gate[a, c]

    # 2. S[a, j] = ||q_h - x^_h||^2 over the cells some query needs (runs of
    #    consecutive cells), four coordinates per pass, each row widened once
    xf = np.empty((4, BLOCK), np.float32)
    c = 0
    while c < K:
        if not need[c]:
            c += 1
            continue
        ce = c
        while ce < K and need[ce]:
            ce += 1
        for b0 in range(cs[c], cs[ce], BLOCK):
            o = np.uint64(b0)
            n = np.uint64(min(b0 + BLOCK, cs[ce]) - b0)
            for a in range(Tq):
                if act[a]:
                    for j in range(n):
                        S[a, o + j] = 0.0
            for t0 in range(0, h, 4):
                for u in range(4):
                    for j in range(n):
                        xf[u, j] = np.float32(XC[t0 + u, o + j])
                for a in range(Tq):
                    if not act[a]:
                        continue
                    q0 = qh[a, t0]
                    q1 = qh[a, t0 + 1]
                    q2 = qh[a, t0 + 2]
                    q3 = qh[a, t0 + 3]
                    for j in range(n):
                        z0 = q0 - xf[0, j]
                        z1 = q1 - xf[1, j]
                        z2 = q2 - xf[2, j]
                        z3 = q3 - xf[3, j]
                        S[a, o + j] += (z0 * z0 + z1 * z1) + (z2 * z2 + z3 * z3)
        c = ce

    # 3-4. per query and cell, the bound (*): vectorised against the cell's
    #      largest R2, then each real ball against its own, and each ball
    #      under it tested exactly at once, in storage order (largest first
    #      within a cell, so an accepting query is nearly always settled by
    #      its first survivor; a rejecting one tests every survivor)
    lb = np.empty(Np, np.float32)
    q32 = np.empty(d, np.float32)
    dlf = np.float32(dl)
    for a in range(Tq):
        if not act[a]:
            continue
        i = c0 + a
        ted = te + dqa[a]
        qk = qg[a]
        ready = False # q32 holds query i (set at its first survivor)
        for c in range(K):
            if not gate[a, c]:
                continue
            s0 = np.uint64(cs[c])
            m = np.uint64(cs[c + 1]) - s0
            hc = _rsq(RM[c], ted, u2, am[a])
            cnt = 0
            for jj in range(m):
                lo = np.float32(TQ[s0 + jj]) * stf
                td = max(max(lo - qk, qk - lo - stf), zero)
                hs = max(np.sqrt(S[a, s0 + jj]) - dlf, zero)
                v = hs * hs + td * td
                lb[s0 + jj] = v
                cnt += np.int64(v <= hc)
            if cnt == 0:
                continue
            for j in range(cs[c], cs[c] + cn[c]):
                if lb[j] <= hc and lb[j] <= _rsq(rad[idm[fmap[j]]], ted, u2, am[a]):
                    if not ready:
                        for t in range(d):
                            q32[t] = Q64[i, t]
                        ready = True
                    ex = _exact(q32, te, dqa[a], np.int64(fmap[j]), kind, Xf, cbuf, T, rad, idm, tol)
                    if ex == 1:
                        out[i] = True
                        break
                    if ex == 2:
                        und[i] = True
            if out[i]:
                break


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------

@njit(nogil=True, cache=True)
def _finish(out, und, nq, cnt):
    n = 0
    for i in range(nq):
        if out[i]:
            und[i] = False
        elif und[i]:
            n += 1
    cnt[0] = n


@njit(nogil=True, cache=True)
def accepts_serial(live, Q64, Yh, te, Eh, iu, stu, xmax2, XC, TQ, dl, RM, fmap, Mh, Mt, cellU, cs, cn,
                   kind, Xf, cbuf, T, rad, idm, tol, rq, und, cnt):
    """Accept mask at tau + eps = te; und[:nq] / cnt[0] get the undecided
    queries / their count."""
    nq = live.shape[0]
    out = np.zeros(nq, np.bool_)
    und[:nq] = False
    S = np.empty((min(CHUNK, nq), XC.shape[1]), np.float32)
    for c0 in range(0, nq, CHUNK):
        _chunk(c0, min(c0 + CHUNK, nq), live, out, und, Q64, Yh, te, Eh, iu, stu, xmax2, XC, TQ, dl, RM,
               fmap, Mh, Mt, cellU, cs, cn, kind, Xf, cbuf, T, rad, idm, tol, rq, S)
    _finish(out, und, nq, cnt)
    return out


@njit(nogil=True, cache=True, parallel=True)
def accepts_parallel(live, Q64, Yh, te, Eh, iu, stu, xmax2, XC, TQ, dl, RM, fmap, Mh, Mt, cellU, cs, cn,
                     kind, Xf, cbuf, T, rad, idm, tol, rq, und, cnt):
    nq = live.shape[0]
    out = np.zeros(nq, np.bool_)
    und[:nq] = False
    for c in prange((nq + CHUNK - 1) // CHUNK):
        c0 = c * CHUNK
        S = np.empty((min(CHUNK, nq), XC.shape[1]), np.float32)      # hoisted: once per thread
        _chunk(c0, min(c0 + CHUNK, nq), live, out, und, Q64, Yh, te, Eh, iu, stu, xmax2, XC, TQ, dl, RM,
               fmap, Mh, Mt, cellU, cs, cn, kind, Xf, cbuf, T, rad, idm, tol, rq, S)
    _finish(out, und, nq, cnt)
    return out


# ---------------------------------------------------------------------------
# Host side: build once per index
# ---------------------------------------------------------------------------

def pca_basis(X64: np.ndarray) -> np.ndarray:
    """d x d orthonormal basis, centred-PCA order (largest variance first):
    the coordinates in which ||q - x||^2 accumulates fastest."""
    Xc = X64 - X64.mean(axis=0, keepdims=True)
    w, E = np.linalg.eigh(Xc.T @ Xc)
    return E[:, ::-1]


def locator_dtype(n_ids: int):
    """Smallest unsigned type holding ids 0 .. n_ids - 1."""
    return np.uint16 if n_ids <= np.iinfo(np.uint16).max + 1 else np.uint32


def sketch_bytes(Np: int, n_ids: int, K: int, d: int, h: int) -> int:
    """Bytes KernelData holds at head width h, for balls padded to Np rows in
    K cells, with locators below n_ids."""
    per_row = h + 1 + np.dtype(locator_dtype(n_ids)).itemsize # head, tails, locator
    return Np * per_row + d * h * 4 + K * (h * 4 + 4 + 8 + 8 + 8 + 4) + 8


def fit_head(Np: int, n_ids: int, K: int, d: int, max_memory) -> int:
    """The widest head (a multiple of 4, at most d) whose sketch fits max_memory;
    64 coordinates without a budget; 0 if not even 4 fit."""
    top = -(-d // 4) * 4
    if max_memory is None:
        return min(64, top)
    for h in range(top, 0, -4):
        if sketch_bytes(Np, n_ids, K, d, h) <= max_memory:
            return h
    return 0


class KernelData:
    """The ball-side arrays every kernel call takes, built once per index.

    C64: decoded centers (float64, the queries' basis), radii -- in storage
    order, used at build time only (ball i of storage is code row gid[i]);
    centroids64, cellU, cell_starts: the IVF cells; src: the exact test's
    arguments (kind, Xf, cbuf, T, rad, idm, tol, rq; see BallIVFIndex), of
    which the index owns `owned` bytes; E: the PCA basis; h: head width.  Stored per ball: the int8 PCA head (coordinate-major, h x Np), the
    uint8 tail-norm code and the uint16/uint32 locator; per cell its largest radius
    and its number of real balls.  Storage is padded per cell (see PAD); pad
    rows are never compacted."""

    def __init__(self, C64: np.ndarray, radii: np.ndarray, gid: np.ndarray, centroids64: np.ndarray,
                 cellU: np.ndarray, cell_starts: np.ndarray, src: tuple, owned: int, E: np.ndarray, h: int):
        N, d = C64.shape
        self.d = d
        self.src = src
        self.owned = owned
        Eh = np.zeros((d, h))
        Eh[:, :min(h, d)] = E[:, :min(h, d)]
        self.Eh = Eh.astype(np.float32)
        Eh = self.Eh.astype(np.float64) # the basis the kernel actually uses
        Y = C64 @ Eh
        x2 = np.einsum("ij,ij->i", C64, C64)
        self.xmax2 = float(x2.max()) if N else 0.0
        tails = np.sqrt(np.maximum(x2 - np.einsum("ij,ij->i", Y, Y), 0.0))

        cs = np.asarray(cell_starts, dtype=np.int64)
        counts = np.diff(cs)
        padded = -(-counts // PAD) * PAD
        self.cs = np.concatenate(([0], np.cumsum(padded))).astype(np.int64)
        self.cn = counts.astype(np.int64)
        Np = int(self.cs[-1])
        cell_of = np.repeat(np.arange(len(counts)), counts)
        pos = self.cs[cell_of] + (np.arange(N) - cs[cell_of])

        # int8 head, one scale; dlk[k] bounds every ball's rounding over its
        # first ck[k] coordinates (code units)
        amax = float(np.abs(Y).max()) if Y.size else 1.0
        self.su = amax / 127.0 if amax > 0.0 else 1.0
        codes = np.clip(np.rint(Y / self.su), -127, 127)
        self.dl = float(np.sqrt(((Y / self.su - codes) ** 2).sum(1)).max()) * (1 + 1e-6) + 1e-6 if N else 1e-6
        self.XC = np.zeros((h, Np), np.int8)
        self.XC[:, pos] = codes.T.astype(np.int8)
        # uint8 tail norms: code c means the tail lies in [c st, (c + 1) st]
        tmax = float(tails.max()) if N else 1.0
        self.st = (tmax / 254.0 if tmax > 0.0 else 1.0) * (1 + 1e-9)
        self.TQ = np.zeros(Np, np.uint8)
        self.TQ[pos] = np.clip(np.floor(tails / self.st), 0, 254).astype(np.uint8)
        # each cell's largest radius
        self.RM = np.full(len(counts), -np.inf, np.float32)
        np.maximum.at(self.RM, cell_of, np.asarray(radii, dtype=np.float32))
        self.fmap = np.zeros(Np, locator_dtype(int(gid.max()) + 1 if N else 0))
        self.fmap[pos] = gid
        # IVF cells, in code units
        M = centroids64 @ Eh
        Mg = np.sqrt(np.maximum(np.einsum("ij,ij->i", centroids64, centroids64) - np.einsum("ij,ij->i", M, M), 0.0))
        self.Mh = np.ascontiguousarray(M / self.su, dtype=np.float32)
        self.Mt = (Mg / self.su).astype(np.float32)
        self.cellU = np.ascontiguousarray(cellU, dtype=np.float64)
        self._und = np.zeros(1, np.bool_) # undecided flags of the last batch
        self._cnt = np.zeros(1, np.int64)
        self._nq = 0
        for par in (False, True): # compile (or load) now, not in a timed query()
            for nq in (1, BLAS_MIN):
                self.run(np.zeros(nq, np.bool_), np.zeros((nq, d)), 0.0, parallel=par)

    def undecided(self):
        """Rows of the last batch left undecided (to be decided by the caller's
        exact test), or None."""
        return np.flatnonzero(self._und[:self._nq]) if self._cnt[0] else None

    def run(self, live, Q64, te, parallel):
        """Accept mask at tau + eps = te; see undecided() for the queries it
        could not decide.  From BLAS_MIN queries up, their heads come from one
        BLAS product."""
        nq = len(live)
        self._nq = nq
        if len(self._und) < nq:
            self._und = np.empty(max(nq, 2 * len(self._und)), np.bool_)
        Yh = Q64 @ self.Eh.astype(np.float64) if nq >= BLAS_MIN else np.zeros((0, 1))
        args = ((live, Q64, Yh, float(te), self.Eh, 1.0 / self.su, self.st / self.su, self.xmax2, self.XC,
                 self.TQ, self.dl, self.RM, self.fmap, self.Mh, self.Mt, self.cellU, self.cs, self.cn)
                + self.src + (self._und, self._cnt))
        return (accepts_parallel if parallel else accepts_serial)(*args)

    def layout(self):
        """(storage rows of the real balls, their cells, their source rows):
        recomputed from the sketch, for diagnostics."""
        pos = np.concatenate([np.arange(self.cs[c], self.cs[c] + self.cn[c]) for c in range(len(self.cn))])
        return pos, np.searchsorted(self.cs, pos, side="right") - 1, self.fmap[pos].astype(np.int64)

    def head(self, pos):
        """The stored int8 head of storage rows pos, in code units [len(pos), h]."""
        return self.XC[:, pos].T.astype(np.float64)

    def survivor_fraction(self, Q64, te, stage: str = "full", chunk: int = 256) -> np.ndarray:
        """Diagnostics (numpy, float64, on the sketch with its bounds): per query,
        the fraction of balls that pass the IVF gate (stage="gate") or the gate
        and the bound (*), i.e. reach the exact test
        (stage="full")."""
        su = self.su
        pos, cell_of, g = self.layout()
        rad, idm = self.src[4], self.src[5]
        X = self.head(pos) * su
        Mh = self.Mh.astype(np.float64) * su
        reach = rad[idm[g]].astype(np.float64) + te
        R2 = np.where(reach >= 0.0, reach * reach, -1.0)
        u = self.cellU + te
        g2 = np.where(u >= 0.0, u * u, -1.0)
        E64 = self.Eh.astype(np.float64)
        out = np.empty(len(Q64))
        for s in range(0, len(Q64), chunk):
            Q = np.asarray(Q64[s:s + chunk], dtype=np.float64)
            Y = Q @ E64
            qn2 = np.einsum("ij,ij->i", Q, Q)
            run = np.concatenate((np.zeros((len(Y), 1)), np.cumsum(Y * Y, axis=1)), axis=1)
            qtail = lambda p: np.sqrt(np.maximum(qn2 - run[:, p], 0.0))[:, None]
            lbc = np.maximum((Y * Y).sum(1)[:, None] + (Mh * Mh).sum(1)[None] - 2.0 * Y @ Mh.T, 0.0)
            lbc += (qtail(Y.shape[1]) - self.Mt[None].astype(np.float64) * su) ** 2
            alive = (lbc <= g2[None])[:, cell_of]
            if stage == "full":
                S = np.maximum((Y * Y).sum(1)[:, None] + (X * X).sum(1)[None] - 2.0 * Y @ X.T, 0.0)
                hs = np.maximum(np.sqrt(S) - self.dl * su, 0.0)
                lo = self.TQ[pos].astype(np.float64)[None] * self.st
                qt = qtail(Y.shape[1])
                td = np.maximum(np.maximum(lo - qt, qt - lo - self.st), 0.0)
                alive &= hs ** 2 + td ** 2 <= R2[None]
            out[s:s + len(Y)] = alive.mean(axis=1)
        return out

    def nbytes(self) -> int:
        """Bytes the index holds: the sketch, the PCA head basis, the cells, and
        the exact test's arrays it owns."""
        return int(sum(a.nbytes for a in (self.Eh, self.XC, self.TQ, self.RM, self.fmap,
                        self.Mh, self.Mt, self.cellU, self.cs, self.cn))) + self.owned