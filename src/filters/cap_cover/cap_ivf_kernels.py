"""
Compiled query kernels for CapIVFIndex.

The cap test, per (query q, cap j) with u_j = v_j / |v_j|:

    <q, u_j> >= cos(theta_j + beta)   <=>
    ||q - u_j||^2 <= R2_j = |q|^2 + 1 - 2 cos(theta_j) cos(beta) + 2 sin(theta_j) sin(beta)

(RA = |q|^2 + 1, RB_j = -2 cos theta_j, RC_j = 2 sin theta_j; theta_j + beta < pi
for every query that reaches the kernel -- CapIVFIndex accepts the rest outright.)

Per query the kernel
  1. rotates q into the first h (= 64) coordinates of a centred-PCA basis of
     the members (h*d multiply-adds; the remaining coordinates are never
     needed -- their norm is |q|^2 - |head|^2);
  2. IVF gate: drops every cell c whose lower bound on ||q - mu_c||^2 exceeds
     the cell's reach (triangle inequality: a cell can only hold an accepting
     member if ||q - mu_c|| <= max_j (reach_j + ||x_j - mu_c||)), and orders
     the surviving cells nearest-first so accepting queries stop early;
  3. early abandoning: for each member of a surviving cell, at checkpoints
     p = 16, 32, 48, h,
         ||q - x||^2  >=  sum_{t<p} (q_t - x_t)^2 + (||q_{>=p}|| - ||x_{>=p}||)^2      (*)
     (reverse triangle inequality on the remaining coordinates); the member is
     dropped as soon as (*) exceeds its reach -- it provably cannot accept.
     PCA puts most of ||q - x||^2 in the leading coordinates, so on the
     blocklist corpora 80-95% of members are dropped at p = 16 and ~99.9% by
     p = 64.  The first `dense` checkpoints are computed for the whole cell,
     coordinate-major and vectorised across its members (a cell with no
     survivor is left there); the few survivors are then compacted and
     continue row-major;
  4. the survivors of the last checkpoint are tested in full in the ORIGINAL
     basis (float32, with its rounding bound ruled out), and the rare one
     inside that bound is re-measured in float64 by CapCoverFilter's exact
     rule, <q, v> >= |v| cos(theta + beta), on the decoded vector v.

Every float32 comparison is widened by REL * |R2| + abs_i (far above its
rounding error), and every widening is towards "keep / re-measure", so no
member that the exact rule accepts is ever dropped.

Large batches run tiles of up to TILE queries per thread, cell by cell, so a
cell's data is read from memory once per tile instead of once per query.

Performance notes (numba): the per-cell helpers are inlined (a call passing
~25 arrays costs ~150 ns in reference counting alone); cells are padded to a
multiple of PAD members so vectorised loops never run a scalar remainder; and
hot loops index 1-D slices from 0 (xr = A[t, s:e]; xr[jj]), which lets LLVM
drop numba's negative-index wraparound check -- with A[t, s+jj] that check
alone blocks vectorisation (measured 4x).
"""

import numpy as np
from numba import get_num_threads, njit, prange

_FM = {"reassoc", "contract", "nsz"}     # lets the float32 sums vectorize; no nnan/ninf (inf reaches are legal)

HEAD = 64                                # PCA coordinates used for gating and abandoning
CHECKPOINTS = (16, 32, 48, 64, 80, 96, 112)   # abandon checks inside the head (h itself is always one)
DENSE = 2                                # default number of leading checkpoints computed for the
                                         # whole cell (the index may pass its own)
REL = 1e-4                               # relative margin on float32 sums (their error is <= ~2.3e-5)
ABS = 1e-5                               # absolute margin, times the pair's norm scale
TILE = 16                                # max queries per thread-tile in the batched kernel
PAD = 16                                 # cells are padded to a multiple of PAD members


@njit(nogil=True, cache=True)
def _exact_ok(j, q64, Xo, RB, RC, En, cb, sb, tiny):
    """CapCoverFilter's float64 rule, <q, v> >= |v| cos(theta + beta), on the
    original vector (accept-leaning by tiny |v|)."""
    d = Xo.shape[1]
    xo = Xo[j]
    ip = 0.0
    for t in range(d):
        ip += q64[t] * np.float64(xo[t])
    thr = En[j] * (-0.5 * RB[j] * cb - 0.5 * RC[j] * sb) # |v| cos(theta + beta)
    return ip - thr >= -tiny * En[j]


@njit(nogil=True, cache=True, fastmath=_FM, inline="always")
def _prep(q64, qpar_i, Eh, ck, xmax2, qh, qt, q32):
    """Fill the query's head coordinates qh, checkpoint tails qt and float32
    copy q32; return (tail past the head, RA, cos beta, sin beta, |q|, abs margin)."""
    d = q64.shape[0]
    h = qh.shape[0]
    nck = ck.shape[0]
    yh = np.zeros(h, np.float64)
    qn2 = 0.0
    for k in range(d):
        qk = q64[k]
        qn2 += qk * qk
        er = Eh[k]
        for t in range(h):
            yh[t] += qk * er[t]
    run = 0.0
    kk = 0
    for t in range(h):
        while kk < nck and ck[kk] == t:
            qt[kk] = np.sqrt(max(qn2 - run, 0.0))
            kk += 1
        run += yh[t] * yh[t]
        qh[t] = yh[t]
    while kk < nck: # checkpoint p = h
        qt[kk] = np.sqrt(max(qn2 - run, 0.0))
        kk += 1
    qg = np.sqrt(max(qn2 - run, 0.0))
    for k in range(d):
        q32[k] = q64[k]
    am = ABS * (qn2 + xmax2)
    return qg, qn2 + 1.0, np.cos(qpar_i), np.sin(qpar_i), np.sqrt(qn2), am


@njit(nogil=True, cache=True, fastmath=_FM, inline="always")
def _gate(qh, qg, RA, qn, qpar_i, am, rel, Mh, Mt, cellU, gate_add, keys, cells):
    """IVF gate: lower bound (*) on ||q - mu_c||^2 against each cell's reach.
    Writes the surviving cells and their bounds; returns how many survive."""
    K = Mh.shape[0]
    h = qh.shape[0]
    nl = 0
    for c in range(K):
        a = cellU[c] + gate_add + qpar_i # angle(q, mu_c) <= U_c + eta + beta
        if a >= np.pi:
            g2 = np.inf
        else:
            g2 = RA - 2.0 * qn * np.cos(a)
        acc = np.float32(0.0)
        mr = Mh[c]
        for t in range(h):
            z = qh[t] - mr[t]
            acc += z * z
        dt = qg - Mt[c]
        lb = acc + dt * dt
        if lb > g2 + rel * g2 + am:
            continue
        keys[nl] = lb
        cells[nl] = c
        nl += 1
    return nl


@njit(nogil=True, cache=True, fastmath=_FM)
def _query_tile(i0, i1, live, out, Q64, qpar, Eh, XT, XR, XtT, RBf, RCf, ck, Xo, RB, RC, En,
                xmax2, Mh, Mt, cellU, gate_add, cs, rel, tiny, eip, maxm, dense):
    """Queries [i0, i1) together (a tile of 1 is the single-query path): cells
    in the tile's nearest-first order (by the smallest bound any of its
    queries has), each cell scanned for every query that still needs it while
    its data is in cache.  Steps 3-4 of the module doc are written out in this
    one body on purpose: any helper taking the ~25 arrays costs ~190 ns per
    call in reference counting, i.e. per (query, cell).

    Member loops index with uint64 offsets: numba skips its negative-index
    wraparound check for unsigned indices, so they vectorise without slices."""
    d = Q64.shape[1]
    h = Mh.shape[1]
    K = Mh.shape[0]
    T = i1 - i0
    nck = ck.shape[0]
    qh = np.empty((T, h), np.float32)
    qt = np.empty((T, nck), np.float32)
    q32 = np.empty((T, d), np.float32)
    sc = np.empty((T, 4), np.float64) # RA, cos beta, sin beta, abs margin
    alive = np.zeros((T, K), np.bool_)
    best = np.full(K, np.inf, np.float32)
    keys = np.empty(K, np.float32)
    cells = np.empty(K, np.int64)
    todo = np.zeros(T, np.bool_)
    n_todo = 0
    for a in range(T):
        i = i0 + a
        if not live[i]:
            continue
        qg, RA, cb, sb, qn, am = _prep(Q64[i], qpar[i], Eh, ck, xmax2, qh[a], qt[a], q32[a])
        sc[a, 0] = RA
        sc[a, 1] = cb
        sc[a, 2] = sb
        sc[a, 3] = am
        nl = _gate(qh[a], np.float32(qg), RA, qn, qpar[i], am, rel, Mh, Mt, cellU, gate_add, keys, cells)
        for oi in range(nl):
            c = cells[oi]
            alive[a, c] = True
            if keys[oi] < best[c]:
                best[c] = keys[oi]
        if nl > 0:
            todo[a] = True
            n_todo += 1
    if n_todo == 0:
        return
    order = np.argsort(best)
    acc = np.empty(maxm, np.float32)
    gmin = np.empty(maxm, np.float32)
    hib = np.empty(maxm, np.float32)
    idx = np.empty(maxm, np.uint64)
    relf = np.float32(rel)
    one = np.uint64(1)

    for oi in range(K):
        c = order[oi]
        if best[c] == np.inf:
            break # no query of the tile needs this cell (nor later ones)
        s0 = np.uint64(cs[c])
        m = np.uint64(cs[c + 1]) - s0
        for a in range(T):
            if not todo[a] or not alive[a, c]:
                continue
            i = i0 + a
            RA = sc[a, 0]
            cb = sc[a, 1]
            sb = sc[a, 2]
            am = sc[a, 3]
            # float32 reach for the abandon tests: its rounding (~1e-6) is far
            # inside am, and the final test uses the float64 reach
            RAf = np.float32(RA)
            cbf = np.float32(cb)
            sbf = np.float32(sb)
            amf = np.float32(am)
            for jj in range(m):
                R2f = RAf + RBf[s0 + jj] * cbf + RCf[s0 + jj] * sbf
                hib[jj] = R2f + relf * abs(R2f) + amf
                acc[jj] = 0.0
                gmin[jj] = np.inf

            # the first `dense` checkpoints: the whole cell, coordinate-major, vectorised
            p = 0
            k = 0
            dead = False
            while k < nck:
                e = ck[k]
                for t in range(p, e):
                    qv = qh[a, t]
                    for jj in range(m):
                        z = qv - XT[t, s0 + jj]
                        acc[jj] += z * z
                qk = qt[a, k]
                mx = -np.inf
                for jj in range(m):
                    dt = qk - XtT[k, s0 + jj]
                    v = min(gmin[jj], hib[jj] - (acc[jj] + dt * dt))
                    gmin[jj] = v
                    mx = max(mx, v)
                p = e
                k += 1
                if mx < 0.0:
                    dead = True # no member of the cell survives
                    break
                if k >= dense:
                    break
            if dead:
                continue

            # compact the survivors (in place: ns <= jj)
            ns = np.uint64(0)
            for jj in range(m):
                idx[ns] = jj
                acc[ns] = acc[jj]
                if gmin[jj] >= 0.0:
                    ns += one

            # remaining checkpoints: survivors only, row-major
            for k in range(k, nck):
                if ns == 0:
                    break
                e = ck[k]
                qk = qt[a, k]
                n2 = np.uint64(0)
                for u in range(ns):
                    jj = idx[u]
                    r = s0 + jj
                    v = acc[u]
                    for t in range(p, e):
                        z = qh[a, t] - XR[r, t]
                        v += z * z
                    dt = qk - XtT[k, r]
                    idx[n2] = jj
                    acc[n2] = v
                    if v + dt * dt <= hib[jj]:
                        n2 += one
                ns = n2
                p = e

            # full test of the survivors in the original basis
            found = False
            for u in range(ns):
                j = s0 + idx[u]
                ip = np.float32(0.0)
                for t in range(d):
                    ip += q32[a, t] * Xo[j, t]
                scr = ip - En[j] * (-0.5 * RB[j] * cb - 0.5 * RC[j] * sb)
                er = eip * En[j]
                if scr >= er:
                    found = True
                    break
                if scr < -er - tiny * En[j]:
                    continue
                if _exact_ok(j, Q64[i], Xo, RB, RC, En, cb, sb, tiny):
                    found = True
                    break
            if found:
                out[i] = True
                todo[a] = False
                n_todo -= 1
        if n_todo == 0:
            return


@njit(nogil=True, cache=True)
def accepts_serial(live, Q64, qpar, Eh, XT, XR, XtT, RBf, RCf, ck, Xo, RB, RC, En,
                   xmax2, Mh, Mt, cellU, gate_add, cs, rel, tiny, eip, maxm, dense):
    out = np.zeros(live.shape[0], np.bool_)
    for i in range(live.shape[0]):
        _query_tile(i, i + 1, live, out, Q64, qpar, Eh, XT, XR, XtT, RBf, RCf, ck, Xo,
                    RB, RC, En, xmax2, Mh, Mt, cellU, gate_add, cs, rel, tiny, eip, maxm, dense)
    return out


@njit(nogil=True, cache=True, parallel=True)
def accepts_parallel(live, Q64, qpar, Eh, XT, XR, XtT, RBf, RCf, ck, Xo, RB, RC, En,
                     xmax2, Mh, Mt, cellU, gate_add, cs, rel, tiny, eip, maxm, dense, tile):
    nq = live.shape[0]
    out = np.zeros(nq, np.bool_)
    n_tiles = (nq + tile - 1) // tile
    for g in prange(n_tiles):
        i0 = g * tile
        i1 = min(i0 + tile, nq)
        _query_tile(i0, i1, live, out, Q64, qpar, Eh, XT, XR, XtT, RBf, RCf, ck, Xo,
                    RB, RC, En, xmax2, Mh, Mt, cellU, gate_add, cs, rel, tiny, eip, maxm, dense)
    return out


@njit(nogil=True, cache=True, fastmath=_FM)
def _gemm_one(i, gi, Q64, qpar, Yh, G, kp, xh2p, XR, XtT, RBf, RCf, ck, Xo, RB, RC, En,
              xmax2, rel, tiny, eip, gerr, gap, idx, acc):
    """Large-batch path for query i (row gi of this chunk's G = qh[:p] . XR[:, :p]^T,
    one BLAS GEMM for the chunk).  The first abandon test, at p = ck[kp], is
        ||q_h - x_h||^2 = ||q_h||^2 + ||x_h||^2 - 2 G   (+ the tail term of (*)),
    for every member at once (vectorised); G's float32 rounding is bounded by
    gerr (|q|^2 + max|x|^2) and added to the margin.  Survivors continue as in
    _query_tile.  Returns whether some member accepts."""
    d = Q64.shape[1]
    h = Yh.shape[1]
    nck = ck.shape[0]
    Np = G.shape[1]
    q64 = Q64[i]
    yh = Yh[gi]
    qn2 = 0.0
    for k in range(d):
        qn2 += q64[k] * q64[k]
    p = ck[kp]
    qh2p = 0.0
    for t in range(p):
        v = np.float64(np.float32(yh[t])) # the float32 value G was computed from
        qh2p += v * v
    run = 0.0
    for t in range(p):
        run += yh[t] * yh[t]
    qtp = np.float32(np.sqrt(max(qn2 - run, 0.0)))
    am = ABS * (qn2 + xmax2) + gerr * (qn2 + xmax2)
    cb = np.cos(qpar[i])
    sb = np.sin(qpar[i])
    RA = qn2 + 1.0
    RAf = np.float32(RA)
    cbf = np.float32(cb)
    sbf = np.float32(sb)
    amf = np.float32(am)
    relf = np.float32(rel)
    qh2pf = np.float32(qh2p)
    g = G[gi]
    xt = XtT[kp]
    # first test, every member, vectorised (uint64 index: no wraparound check)
    for j in range(np.uint64(Np)):
        R2f = RAf + RBf[j] * cbf + RCf[j] * sbf
        dt = qtp - xt[j]
        gap[j] = (R2f + relf * abs(R2f) + amf) - ((qh2pf + xh2p[j] - 2.0 * g[j]) + dt * dt)
    ns = 0
    for j in range(Np):
        idx[ns] = j
        if gap[j] >= 0.0:
            ns += 1
    if ns == 0:
        return False
    qh = np.empty(h, np.float32)
    for t in range(h):
        qh[t] = yh[t]
    for u in range(ns):
        j = idx[u]
        acc[u] = qh2pf + xh2p[j] - 2.0 * g[j]
    # remaining head checkpoints, survivors only, row-major
    for k in range(kp + 1, nck):
        if ns == 0:
            return False
        e = ck[k]
        for t in range(p, e):
            run += yh[t] * yh[t]
        qk = np.float32(np.sqrt(max(qn2 - run, 0.0)))
        n2 = 0
        for u in range(ns):
            j = idx[u]
            v = acc[u]
            for t in range(p, e):
                z = qh[t] - XR[j, t]
                v += z * z
            dt = qk - XtT[k, j]
            R2f = RAf + RBf[j] * cbf + RCf[j] * sbf
            idx[n2] = j
            acc[n2] = v
            if v + dt * dt <= R2f + relf * abs(R2f) + amf:
                n2 += 1
        ns = n2
        p = e
    # full test of the survivors in the original basis, then the exact rule
    for u in range(ns):
        j = idx[u]
        ip = np.float32(0.0)
        for t in range(d):
            ip += np.float32(q64[t]) * Xo[j, t]
        scr = ip - En[j] * (-0.5 * RB[j] * cb - 0.5 * RC[j] * sb)
        er = eip * En[j]
        if scr >= er:
            return True
        if scr < -er - tiny * En[j]:
            continue
        if _exact_ok(j, q64, Xo, RB, RC, En, cb, sb, tiny):
            return True
    return False


@njit(nogil=True, cache=True)
def accepts_gemm_serial(live, i0, out, Q64, qpar, Yh, G, kp, xh2p, XR, XtT, RBf, RCf, ck, Xo,
                        RB, RC, En, xmax2, rel, tiny, eip, gerr):
    Np = G.shape[1]
    gap = np.empty(Np, np.float32)
    idx = np.empty(Np, np.int64)
    acc = np.empty(Np, np.float32)
    for gi in range(G.shape[0]):
        i = i0 + gi
        if live[i]:
            out[i] = _gemm_one(i, gi, Q64, qpar, Yh, G, kp, xh2p, XR, XtT, RBf, RCf, ck, Xo,
                               RB, RC, En, xmax2, rel, tiny, eip, gerr, gap, idx, acc)


@njit(nogil=True, cache=True, parallel=True)
def accepts_gemm_parallel(live, i0, out, Q64, qpar, Yh, G, kp, xh2p, XR, XtT, RBf, RCf, ck, Xo,
                          RB, RC, En, xmax2, rel, tiny, eip, gerr, nthreads):
    Np = G.shape[1]
    nc = G.shape[0]
    nth = max(1, min(nthreads, nc))
    per = (nc + nth - 1) // nth
    for w in prange(nth):
        gap = np.empty(Np, np.float32)
        idx = np.empty(Np, np.int64)
        acc = np.empty(Np, np.float32)
        for gi in range(w * per, min((w + 1) * per, nc)):
            i = i0 + gi
            if live[i]:
                out[i] = _gemm_one(i, gi, Q64, qpar, Yh, G, kp, xh2p, XR, XtT, RBf, RCf, ck, Xo,
                                   RB, RC, En, xmax2, rel, tiny, eip, gerr, gap, idx, acc)


# ---------------------------------------------------------------------------
# Host side: build once per index
# ---------------------------------------------------------------------------

def pca_head(X64: np.ndarray, h: int) -> np.ndarray:
    """d x h orthonormal columns: the leading h centred-PCA directions of X,
    the coordinates in which ||q - x||^2 accumulates fastest."""
    Xc = X64 - X64.mean(axis=0, keepdims=True)
    w, E = np.linalg.eigh(Xc.T @ Xc)
    return np.ascontiguousarray(E[:, ::-1][:, :h])


def layout(d: int, head: int = HEAD):
    """(head width h, checkpoints ck): ck holds the checks inside the head
    plus h itself (the last check before the full test)."""
    h = min(head, d)
    ck = [c for c in CHECKPOINTS if c < h] + [h]
    return h, np.asarray(ck, dtype=np.int64)


def head_and_tails(X64: np.ndarray, Eh: np.ndarray, ck: np.ndarray):
    """For rows X64 [n,d]: (float64 head coordinates [n,h], float32 norms of
    everything past each checkpoint [n,len(ck)], float32 norm past the head [n])."""
    Y = X64 @ Eh
    x2 = np.einsum("ij,ij->i", X64, X64)
    run = np.concatenate((np.zeros((len(Y), 1)), np.cumsum(Y * Y, axis=1)), axis=1) # run[:, p] = sum_{t<p}
    T = np.sqrt(np.maximum(x2[:, None] - run[:, ck], 0.0))
    g = np.sqrt(np.maximum(x2 - run[:, Eh.shape[1]], 0.0))
    return Y, np.ascontiguousarray(T, dtype=np.float32), g.astype(np.float32)


class KernelData:
    """The member-side arrays every kernel call takes, built once per index.

    X64: members in storage order (float64; unit cap directions) -- what the kernel measures distances to; Xo: the original
    float32 vectors the exact rule is evaluated on; RB, RC, En: per-member
    reach coefficients and norms (see the module doc); centroids64, cellU,
    cell_starts: the IVF cells.  Storage is padded per cell (see PAD); pad
    rows have an infinite tail norm and a negative reach, so they are dropped
    at the first checkpoint and could never accept."""

    def __init__(self, X64: np.ndarray, Xo: np.ndarray, RB: np.ndarray, RC: np.ndarray,
                 En: np.ndarray, centroids64: np.ndarray, cellU: np.ndarray, cell_starts: np.ndarray,
                 head: int = HEAD, dense: int = DENSE, gemm_ck: int = 1):
        N, d = X64.shape
        h, ck = layout(d, head)
        self.ck = ck
        self.dense = int(dense)
        self.Eh = pca_head(X64, h)
        Y, Xt, _ = head_and_tails(X64, self.Eh, ck)

        cs = np.asarray(cell_starts, dtype=np.int64)
        counts = np.diff(cs)
        padded = -(-counts // PAD) * PAD
        self.cs = np.concatenate(([0], np.cumsum(padded))).astype(np.int64)
        Np = int(self.cs[-1])
        cell_of = np.repeat(np.arange(len(counts)), counts)
        self.pos = (self.cs[cell_of] + (np.arange(N) - cs[cell_of])).astype(np.int64) # row -> padded row
        self.maxm = int(padded.max())

        XR = np.zeros((Np, h), np.float32)
        XR[self.pos] = Y
        self.XR = XR # Np x h, row-major
        # coordinate-major copy, only of the coordinates the dense checkpoints read
        self.XT = np.ascontiguousarray(XR[:, :int(ck[min(self.dense, len(ck)) - 1])].T)
        self.cell_of = cell_of # row -> cell (diagnostics)
        XtT = np.full((len(ck), Np), 1e30, np.float32)
        XtT[:, self.pos] = Xt.T
        self.XtT = XtT # len(ck) x Np tails
        M, _, self.Mt = head_and_tails(centroids64, self.Eh, ck)
        self.Mh = np.ascontiguousarray(M, dtype=np.float32)
        self.Xo = np.zeros((Np, d), np.float32)
        self.Xo[self.pos] = Xo
        self.RC = np.zeros(Np)
        self.RC[self.pos] = RC
        self.RCf = self.RC.astype(np.float32)
        self.En = np.ones(Np)
        self.En[self.pos] = En
        self.set_reach(RB)
        self.cellU = np.ascontiguousarray(cellU, dtype=np.float64)
        X2 = np.einsum("ij,ij->i", X64, X64)
        self.xmax2 = float(X2.max())
        # large-batch path: the first abandon test (checkpoint ck[kp]) via one GEMM
        self.kp = min(int(gemm_ck), len(ck) - 1)
        pg = int(ck[self.kp])
        self.XRpT = np.ascontiguousarray(XR[:, :pg].T) # pg x Np
        self.xh2p = (XR[:, :pg].astype(np.float64) ** 2).sum(1).astype(np.float32)
        # float32 error of -2 <q_h, x_h> from the GEMM, per unit of |q|^2 + max|x|^2
        self.gerr = 2.0 * (pg + 4) * 2.0 ** -24
        self._warm_up(d)

    def _warm_up(self, d: int) -> None:
        """Compile (or load from numba's cache) both drivers now, at build time,
        so no timed query() pays for it: an all-dead batch runs no query."""
        dead = np.zeros(1, np.bool_)
        Q = np.zeros((1, d))
        z = np.zeros(1)
        for par in (False, True):
            self.run(dead, Q, z, 0.0, 0.0, 0.0, parallel=par)
            self.run_gemm(dead, Q, z, 0.0, 0.0, parallel=par)

    def set_reach(self, RB: np.ndarray) -> None:
        """(Re)set the per-member reach coefficient RB (-2 cos theta),
        given in unpadded storage order; pad rows get -1 (admits nothing)."""
        self.RB = np.full(len(self.Xo), -1.0)
        self.RB[self.pos] = RB
        self.RBf = self.RB.astype(np.float32)

    def run(self, live, Q64, qpar, gate_add, tiny, eip, parallel):
        args = (live, Q64, qpar, self.Eh, self.XT, self.XR, self.XtT, self.RBf, self.RCf, self.ck,
                self.Xo, self.RB, self.RC, self.En, self.xmax2, self.Mh, self.Mt, self.cellU, gate_add,
                self.cs, REL, tiny, eip, self.maxm, self.dense)
        if not parallel:
            return accepts_serial(*args)
        # tiles of up to TILE queries, but enough tiles to keep every thread busy
        tile = max(1, min(TILE, -(-len(live) // get_num_threads())))
        return accepts_parallel(*args, tile)

    GEMM_BYTES = 1 << 25 # scratch for one chunk's G (queries x members float32)

    def run_gemm(self, live, Q64, qpar, tiny, eip, parallel, chunk: int = 0):
        """Large batches: per chunk of queries, the PCA head of the queries
        (float64 GEMM) and the first abandon test's inner products G (float32
        GEMM), then _gemm_one per query.  Same decisions as run().  Chunks are
        as large as GEMM_BYTES allows (up to 1024 queries): larger chunks
        measured faster, single-threaded, on the blocklist covers."""
        nq = len(Q64)
        out = np.zeros(nq, np.bool_)
        pg = self.XRpT.shape[0]
        if chunk <= 0:
            chunk = int(max(64, min(1024, self.GEMM_BYTES // (4 * max(1, self.XRpT.shape[1])))))
        nth = get_num_threads()
        for s in range(0, nq, chunk):
            Yh = Q64[s:s + chunk] @ self.Eh
            G = np.ascontiguousarray(Yh[:, :pg], dtype=np.float32) @ self.XRpT
            args = (live, s, out, Q64, qpar, Yh, G, self.kp, self.xh2p, self.XR, self.XtT, self.RBf,
                    self.RCf, self.ck, self.Xo, self.RB, self.RC, self.En, self.xmax2, REL, tiny, eip, self.gerr)
            if parallel:
                accepts_gemm_parallel(*args, nth)
            else:
                accepts_gemm_serial(*args)
        return out

    def survivor_fraction(self, Q64, qpar, gate_add, stage: str = "full",
                          chunk: int = 256) -> np.ndarray:
        """Diagnostics (numpy, float64, no margins): per query, the fraction of
        members that pass the IVF cell gate (stage="gate") or the gate and every
        abandon checkpoint, i.e. reach the full test (stage="full")."""
        nq = len(Q64)
        X = self.XR[self.pos].astype(np.float64) # real rows, PCA head
        Xt = self.XtT[:, self.pos].astype(np.float64)
        x2 = np.einsum("ij,ij->i", X, X)
        Mh = self.Mh.astype(np.float64)
        out = np.empty(nq)
        for s in range(0, nq, chunk):
            Q = np.asarray(Q64[s:s + chunk], dtype=np.float64)
            b = np.asarray(qpar[s:s + chunk], dtype=np.float64)
            Y = Q @ self.Eh
            qn2 = np.einsum("ij,ij->i", Q, Q)
            run = np.concatenate((np.zeros((len(Y), 1)), np.cumsum(Y * Y, axis=1)), axis=1)
            qtail = lambda p: np.sqrt(np.maximum(qn2 - run[:, p], 0.0))
            # gate
            lbc = (Y * Y).sum(1)[:, None] + (Mh * Mh).sum(1)[None] - 2.0 * Y @ Mh.T
            lbc += (qtail(Y.shape[1])[:, None] - self.Mt[None].astype(np.float64)) ** 2
            A = self.cellU[None] + gate_add + b[:, None]
            with np.errstate(invalid="ignore"):
                g2 = np.where(A >= np.pi, np.inf, (qn2 + 1.0)[:, None] - 2.0 * np.sqrt(qn2)[:, None] * np.cos(A))
            alive = (lbc <= g2)[:, self.cell_of]
            if stage == "full":
                cb, sb, RA = np.cos(b), np.sin(b), qn2 + 1.0
                R2 = RA[:, None] + self.RB[self.pos][None] * cb[:, None] + self.RC[self.pos][None] * sb[:, None]
                for k, p in enumerate(self.ck):
                    p = int(p)
                    part = run[:, p][:, None] + (X[:, :p] ** 2).sum(1)[None] - 2.0 * Y[:, :p] @ X[:, :p].T
                    alive &= part + (qtail(p)[:, None] - Xt[k][None]) ** 2 <= R2
            out[s:s + len(Y)] = alive.mean(axis=1)
        return out

    def nbytes(self) -> int:
        return int(sum(a.nbytes for a in (self.Eh, self.XT, self.XR, self.XtT, self.Mh, self.Mt, self.Xo,
                                          self.RB, self.RC, self.RBf, self.RCf, self.En, self.cellU,
                                          self.cs, self.pos, self.cell_of, self.XRpT, self.xh2p)))