"""
Compiled query kernels for CapIVFIndex.

The cap test, per (query q, cap j) with u_j = v_j / |v_j| (v_j the decoded cap):

    <q, u_j> >= cos(theta_j + beta)   <=>   ||q - u_j||^2 <= R2(theta_j) = |q|^2 + 1 - 2 |q| cos(theta_j + beta)

R2 grows with theta on [0, pi - beta], so a cap may be bounded with any
theta^ >= theta_j (CapIVFIndex accepts every query with theta_max + beta >= pi
outright).  Everything is in CapCoverFilter's working basis.

The index keeps no copy of the caps, only a compact sketch per cap: its first
h centred-PCA coordinates in int8 (one scale; every cap's rounding is at most
dl), its norm past them as a uint8 interval [t_lo, t_lo + st], its angle as a
uint8 code (rounded up), and a locator into CapCoverFilter's own codes.  A
batch of queries is answered in chunks of CHUNK queries:

  1. per query: its head q_h (h*d multiply-adds, or one BLAS product for the
     batch), its norm past the head, a 255-entry table of R2 per angle code,
     and the IVF gate -- cell c is skipped if a lower bound on ||q - mu_c||
     exceeds the cell's reach (angle(q, mu_c) <= U_c + beta,
     U_c = max_j (theta_j + angle(u_j, mu_c)));
  2. the head distances S = ||q_h - u^_h||^2 of every (query, cap) pair of
     the chunk, in one coordinate-major pass over the sketch: each int8 row
     is widened once and used by every query, four coordinates at a time
     (long, contiguous, vectorised loops);
  3. per query and cell, the lower bound
         ||q - u||^2 >= max(sqrt(S) - dl, 0)^2 + dist(||q_>=h||, [t_lo, t_lo + st])^2 (*)
     (reverse triangle inequality on the rest) against the cell's widest R2,
     vectorised; the few caps under it are then held against their own R2:
     a cap above it provably does not accept;
  4. the survivors are decided exactly on CapCoverFilter's own codes
     (seeded-PQ codewords / TurboQuant levels, decoded in place) by the
     filter's rule <q, v> >= |v| cos(theta + beta), most relative slack
     first (an accepting query is then nearly always settled by the first
     decode).  A pair within the filter's float margin of its boundary leaves
     the query undecided -- the caller (CapCoverFilter) then decides it with
     its own scan.  In a batch of PAIRS_MIN or more queries, the remaining
     tests of GROUP queries are grouped by cap, each cap decoded once.
Chunks, and then the grouped tests cell by cell, are what accepts_parallel
spreads over the threads.

Every float comparison of steps 1-3 is widened by REL * |R2| + ABS terms,
towards "keep", so no cap that accepts q is ever dropped.  Performance notes
(numba): member loops index with uint64 offsets, which skips the
negative-index wraparound check that blocks vectorisation; per-cap table
lookups stay out of the vectorised loops, which they would serialise.
"""

import numpy as np
from numba import njit, prange

_FM = {"reassoc", "contract", "nsz"}     # lets the float32 sums vectorize; no nnan/ninf

REL = 1e-4                               # relative margin on float32 sums (their error is <= ~1e-5)
ABS = 1e-5                               # absolute margin, times the pair's norm scale
PAD = 8                                  # cells are padded to a multiple of PAD caps
CHUNK = 32                               # queries per chunk (shares each sketch row between them)
BLOCK = 1024                             # sketch rows per pass of step 2 (stays in L1/L2)
BLAS_MIN = 32                            # batches this large get their heads from one BLAS product
BEST_FIRST = 8                           # survivors tested in order of slack (the rest in storage order)
GROUP = 1024                             # queries whose remaining exact tests are grouped by cap
PAIRS_MIN = 16                           # smaller batches test their survivors one query at a time
PAIR_ROWS = 256                          # pair-list room per query (pooled per chunk)
N_CODES = 255                            # angle codes 0..254; 255 marks a pad row
PAD_CODE = 255
KIND_FLAT = 0                            # exact test on a float32 copy (standalone use, tests)
KIND_CODES = 1                           # exact test on CapCoverFilter's codes


# ---------------------------------------------------------------------------
# Exact test (step 4)
# ---------------------------------------------------------------------------

@njit(nogil=True, cache=True, inline="always")
def _verdict(dot, n2, th, qn, b, tol):
    n = np.sqrt(n2)
    thr = n * np.cos(th + b) # |v| cos(theta + beta)
    e = tol * n * (qn + 1.0)
    if dot >= thr + e:
        return 1
    if dot < thr - e:
        return 0
    return 2


@njit(nogil=True, cache=True, fastmath=_FM, inline="always")
def _exact(q32, qn, b, g, ixs, kind, Xf, thf, cbuf, kbuf, abuf, books, nbuf, L, alo, astep, tol):
    """1 = accept, 0 = reject, 2 = undecided, for cap g of the exact test's
    source (see CapIVFIndex.build): CapCoverFilter's rule
    <q, v> >= |v| cos(theta + beta) on the decoded v, undecided within
    tol * |v| (|q| + 1) of the boundary.
    q32: q in float32; ixs: scratch of >= d codeword indices.  The dot product
    is summed in float32, in six independent accumulators for the seeded-PQ
    slices of six dimensions (most caps; other slice widths vectorise well as
    they are); its rounding, (d + 5) 2^-24 |q| |v| at most, is inside that
    margin.  |v|^2 sums the codewords' squared norms (float32 table: its
    rounding, 2^-24 relative, is far inside the margin too)."""
    z = np.float32(0.0)
    a0 = z
    a1 = z
    a2 = z
    a3 = z
    a4 = z
    a5 = z
    n2 = 0.0
    if kind == KIND_FLAT:
        for t in range(q32.shape[0]):
            v = Xf[g, t]
            a0 += q32[t] * v
            n2 += np.float64(v) * np.float64(v)
        th = thf[g]
    else:
        lv = 0
        while g >= L[lv + 1, 0]:
            lv += 1
        rr = g - L[lv, 0]
        bits = L[lv, 5]
        m = L[lv, 6]
        cw = books[lv]
        sub = cw.shape[1]
        no = L[lv, 7]
        mask = (np.int64(1) << bits) - 1
        a = np.int64(0)
        nacc = 0
        if L[lv, 4] > 0:
            a = kbuf[L[lv, 3] + rr]
            nacc = L[lv, 4]
        bb = L[lv, 1] + rr * L[lv, 2]
        # the codeword of each slice first (the bit reader), then the dot
        # product in a loop of its own (twice as fast as interleaved)
        for s in range(m):
            while nacc < bits:
                a |= np.int64(cbuf[bb]) << nacc
                bb += 1
                nacc += 8
            ix = a & mask
            a >>= bits
            nacc -= bits
            ixs[s] = ix
            n2 += nbuf[no + ix]
        if sub == 6:
            for s in range(m):
                x = ixs[s]
                o = s * 6
                a0 += q32[o] * cw[x, 0]
                a1 += q32[o + 1] * cw[x, 1]
                a2 += q32[o + 2] * cw[x, 2]
                a3 += q32[o + 3] * cw[x, 3]
                a4 += q32[o + 4] * cw[x, 4]
                a5 += q32[o + 5] * cw[x, 5]
        else:
            for s in range(m):
                x = ixs[s]
                o = s * sub
                for t in range(sub):
                    a0 += q32[o + t] * cw[x, t]
        th = alo + np.float64(abuf[g]) * astep
    dot = np.float64(((a0 + a1) + (a2 + a3)) + (a4 + a5))
    return _verdict(dot, n2, th, qn, b, tol)


@njit(nogil=True, cache=True, fastmath=_FM)
def _decode(g, v, ixs, kind, Xf, thf, cbuf, kbuf, abuf, books, nbuf, L, alo, astep):
    """Cap g of the code source decoded into v (float32, exactly the stored
    codewords); returns (|v|^2, theta)."""
    n2 = 0.0
    if kind == KIND_FLAT:
        for t in range(v.shape[0]):
            x = Xf[g, t]
            v[t] = x
            n2 += np.float64(x) * np.float64(x)
        return n2, thf[g]
    lv = 0
    while g >= L[lv + 1, 0]:
        lv += 1
    rr = g - L[lv, 0]
    bits = L[lv, 5]
    m = L[lv, 6]
    cw = books[lv]
    sub = cw.shape[1]
    no = L[lv, 7]
    mask = (np.int64(1) << bits) - 1
    a = np.int64(0)
    nacc = 0
    if L[lv, 4] > 0:
        a = kbuf[L[lv, 3] + rr]
        nacc = L[lv, 4]
    bb = L[lv, 1] + rr * L[lv, 2]
    for s in range(m):
        while nacc < bits:
            a |= np.int64(cbuf[bb]) << nacc
            bb += 1
            nacc += 8
        ix = a & mask
        a >>= bits
        nacc -= bits
        ixs[s] = ix
        n2 += nbuf[no + ix]
    for s in range(m):
        x = ixs[s]
        o = s * sub
        for t in range(sub):
            v[o + t] = cw[x, t]
    return n2, alo + np.float64(abuf[g]) * astep


@njit(nogil=True, cache=True)
def _sort_pairs(pq, pr, npair, room, Np):
    """The pair lists (chunk c's: its npair[c] pairs from c * room; pq: query,
    pr: storage row) sorted by row (counting sort): row r's queries are
    qof[start[r]:start[r + 1]].  Returns (start, qof)."""
    start = np.zeros(Np + 1, np.int64)
    for c in range(len(npair)):
        for k in range(c * room, c * room + npair[c]):
            start[pr[k] + 1] += 1
    for r in range(Np):
        start[r + 1] += start[r]
    fill = start[:-1].copy()
    qof = np.empty(start[Np], np.int64)
    for c in range(len(npair)):
        for k in range(c * room, c * room + npair[c]):
            r = pr[k]
            qof[fill[r]] = pq[k]
            fill[r] += 1
    return start, qof


@njit(nogil=True, cache=True, fastmath=_FM)
def _pairs(r0, r1, start, qof, out, und, Q64, qpar, gqn, fmap, kind, Xf, thf, cbuf, kbuf, abuf, books, nbuf, L,
           alo, astep, tol, v, ixs):
    """The pair lists' exact tests at storage rows [r0, r1) (see _sort_pairs):
    each cap is decoded once and tested against every query still waiting
    for it, by one vectorised float32 dot product (its rounding is inside
    _verdict's margin, as in _exact).  v, ixs: d floats / ints of scratch.
    Calls on different rows may run at once: they only ever set out[i] /
    und[i] to True."""
    d = Q64.shape[1]
    for r in range(r0, r1):
        need = False
        for k in range(start[r], start[r + 1]):
            if not out[qof[k]]:
                need = True
                break
        if not need:
            continue
        n2, th = _decode(np.int64(fmap[r]), v, ixs, kind, Xf, thf, cbuf, kbuf, abuf, books, nbuf, L, alo, astep)
        for k in range(start[r], start[r + 1]):
            i = qof[k]
            if out[i]:
                continue
            q = Q64[i]
            dot = np.float32(0.0)
            for t in range(np.uint64(d)):
                dot += np.float32(q[t]) * v[t]
            ex = _verdict(np.float64(dot), n2, th, gqn[i], qpar[i], tol)
            if ex == 1:
                out[i] = True
            elif ex == 2:
                und[i] = True


# ---------------------------------------------------------------------------
# Per-query setup (step 1)
# ---------------------------------------------------------------------------

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


@njit(nogil=True, cache=True, inline="always")
def _reach_table(RA, qn, cb, sb, u2, am2, cth, sth, rt):
    """rt[c] = R2(theta^_c) in code units^2, widened by the float margin;
    rt[PAD_CODE] = -1 (pad rows never survive)."""
    for c in range(N_CODES):
        ca = cth[c] * cb - sth[c] * sb # cos(theta^_c + beta)
        if cth[c] * sb + sth[c] * cb < 0.0: # theta^_c + beta past pi
            ca = -1.0
        r2 = (RA - 2.0 * qn * ca) * u2
        rt[c] = np.float32(r2 + REL * abs(r2) + am2)
    rt[PAD_CODE] = -1.0


@njit(nogil=True, cache=True, fastmath=_FM, inline="always")
def _gate(qh, qg, RA, qn, b, u2, am2, Mh, Mt, cellU, gate_add, gate):
    """IVF gate in code units: gate[c] = the lower bound on ||q - mu_c||^2 is
    within cell c's reach.  Returns how many cells pass."""
    h = qh.shape[0]
    n = 0
    for c in range(Mh.shape[0]):
        a = cellU[c] + gate_add + b # angle(q, mu_c) <= U_c + eta + beta
        if a >= np.pi:
            g2 = np.inf
        else:
            g2 = (RA - 2.0 * qn * np.cos(a)) * u2
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
def _chunk(c0, c1, live, out, und, Q64, Yh, qpar, Eh, iu, stu, XC, TQ, dl, TC, CM, cth, sth, fmap,
           Mh, Mt, cellU, gate_add, cs, kind, Xf, thf, cbuf, kbuf, abuf, books, nbuf, L, alo, astep, tol,
           pq, pr, npair, gqn, S):
    """Queries [c0, c1).  Survivors left after the best-first tests go to the
    pair list (pq, pr; npair[0] used) while it has room, with the query's |q|
    in gqn, for _pairs; an empty list tests them here.  S: scratch of at
    least c1 - c0 rows of Np floats (the head distances)."""
    T = c1 - c0
    d = Q64.shape[1]
    h = Eh.shape[1]
    Np = XC.shape[1]
    K = Mh.shape[0]
    u2 = iu * iu
    zero = np.float32(0.0)
    stf = np.float32(stu)

    # 1. per query: head, tails, R2 table, gate
    qh = np.empty((T, h), np.float32)
    qg = np.empty(T, np.float32)
    qna = np.empty(T) # |q|
    rt = np.empty((T, N_CODES + 1), np.float32)
    gate = np.zeros((T, K), np.bool_)
    need = np.zeros(K, np.bool_)
    act = np.zeros(T, np.bool_)
    for a in range(T):
        i = c0 + a
        if not live[i]:
            continue
        b = qpar[i]
        g_, qn2 = _prep(Q64[i], Eh, iu, qh[a], Yh, i)
        qg[a] = g_
        RA = qn2 + 1.0
        qn = np.sqrt(qn2)
        am2 = ABS * RA * u2
        cb = np.cos(b)
        sb = np.sin(b)
        qna[a] = qn
        _reach_table(RA, qn, cb, sb, u2, am2, cth, sth, rt[a])
        if _gate(qh[a], g_, RA, qn, b, u2, am2, Mh, Mt, cellU, gate_add, gate[a]) > 0:
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
            for a in range(T):
                if act[a]:
                    for j in range(n):
                        S[a, o + j] = 0.0
            for t0 in range(0, h, 4):
                for u in range(4):
                    for j in range(n):
                        xf[u, j] = np.float32(XC[t0 + u, o + j])
                for a in range(T):
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

    # 3-4. per query: the bound (*), then the exact test
    lb = np.empty(Np, np.float32)
    rows = np.empty(Np, np.int64)
    hib = np.empty(Np, np.float32)
    slack = np.empty(Np, np.float32)
    ixs = np.empty(d, np.int64)
    q32 = np.empty(d, np.float32)
    dlf = np.float32(dl)
    for a in range(T):
        if not act[a]:
            continue
        i = c0 + a
        # the bound (*) by cell: vectorised against the cell's widest R2,
        # then each candidate against its own
        qk = qg[a]
        ns = 0
        for c in range(K):
            if not gate[a, c]:
                continue
            s0 = np.uint64(cs[c])
            m = np.uint64(cs[c + 1]) - s0
            hc = rt[a, CM[c]]
            cnt = 0
            for jj in range(m):
                lo = np.float32(TQ[s0 + jj]) * stf
                td = max(max(lo - qk, qk - lo - stf), zero)
                hs = max(np.sqrt(S[a, s0 + jj]) - dlf, zero)
                v = hs * hs + td * td
                lb[s0 + jj] = v
                cnt += np.int64(v <= hc)
            if cnt:
                for j in range(cs[c], cs[c + 1]):
                    hb = rt[a, TC[j]]
                    if lb[j] <= hb:
                        rows[ns] = j
                        hib[ns] = hb
                        slack[ns] = lb[j]
                        ns += 1
        if ns == 0:
            continue
        for uu in range(ns):
            slack[uu] = (hib[uu] - slack[uu]) / hib[uu] # relative slack: the best accept predictor
        b = qpar[i]
        qn = qna[a]
        q = Q64[i]
        for t in range(d):
            q32[t] = q[t]
        # the caps with the most slack first: an accepting query (whose
        # survivors can number thousands) is nearly always settled by the first
        # of them; a rejecting one has to test every survivor
        for _ in range(min(BEST_FIRST, ns)):
            jb = 0
            for uu in range(1, ns):
                if slack[uu] > slack[jb]:
                    jb = uu
            ex = _exact(q32, qn, b, np.int64(fmap[rows[jb]]), ixs, kind, Xf, thf, cbuf, kbuf, abuf, books, nbuf, L, alo, astep, tol)
            if ex == 1:
                out[i] = True
                break
            if ex == 2:
                und[i] = True
            ns -= 1
            rows[jb] = rows[ns]
            slack[jb] = slack[ns]
        if out[i] or ns == 0:
            continue
        if npair[0] + ns <= pq.shape[0]:
            k0 = npair[0]
            for uu in range(ns):
                pq[k0 + uu] = i
                pr[k0 + uu] = rows[uu]
            npair[0] = k0 + ns
            gqn[i] = qn
            continue
        for uu in range(ns):
            ex = _exact(q32, qn, b, np.int64(fmap[rows[uu]]), ixs, kind, Xf, thf, cbuf, kbuf, abuf, books, nbuf, L, alo, astep, tol)
            if ex == 1:
                out[i] = True
                break
            if ex == 2:
                und[i] = True


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
def accepts_serial(live, Q64, Yh, qpar, Eh, iu, stu, XC, TQ, dl, TC, CM, cth, sth, fmap, Mh, Mt, cellU,
                   gate_add, cs, kind, Xf, thf, cbuf, kbuf, abuf, books, nbuf, L, alo, astep, tol, und, cnt):
    """Accept mask; und[:nq] / cnt[0] get the undecided queries / their
    count.  Per GROUP queries: steps 1-4 chunk by chunk (each chunk leaving
    its remaining exact tests in its own slice of the pair list), then those
    tests grouped by cap, cell by cell."""
    nq = live.shape[0]
    out = np.zeros(nq, np.bool_)
    und[:nq] = False
    gqn = np.empty(nq)
    S = np.empty((min(CHUNK, nq), XC.shape[1]), np.float32)
    v = np.empty(Q64.shape[1], np.float32)
    ixs = np.empty(Q64.shape[1], np.int64)
    for g0 in range(0, nq, GROUP):
        g1 = min(g0 + GROUP, nq)
        nc = (g1 - g0 + CHUNK - 1) // CHUNK
        room = CHUNK * PAIR_ROWS if g1 - g0 >= PAIRS_MIN else 0
        pq = np.empty(nc * room, np.int64)
        pr = np.empty(nc * room, np.int32)
        npair = np.zeros(nc, np.int64)
        for c in range(nc):
            c0 = g0 + c * CHUNK
            _chunk(c0, min(c0 + CHUNK, g1), live, out, und, Q64, Yh, qpar, Eh, iu, stu, XC, TQ, dl, TC, CM,
                   cth, sth, fmap, Mh, Mt, cellU, gate_add, cs, kind, Xf, thf, cbuf, kbuf, abuf, books, nbuf, L,
                   alo, astep, tol, pq[c * room:(c + 1) * room], pr[c * room:(c + 1) * room], npair[c:c + 1], gqn,
                   S)
        if npair.sum():
            start, qof = _sort_pairs(pq, pr, npair, room, XC.shape[1])
            for c in range(Mh.shape[0]):
                _pairs(cs[c], cs[c + 1], start, qof, out, und, Q64, qpar, gqn, fmap, kind, Xf, thf, cbuf, kbuf,
                       abuf, books, nbuf, L, alo, astep, tol, v, ixs)
    _finish(out, und, nq, cnt)
    return out


@njit(nogil=True, cache=True, parallel=True)
def accepts_parallel(live, Q64, Yh, qpar, Eh, iu, stu, XC, TQ, dl, TC, CM, cth, sth, fmap, Mh, Mt, cellU,
                     gate_add, cs, kind, Xf, thf, cbuf, kbuf, abuf, books, nbuf, L, alo, astep, tol, und, cnt):
    """accepts_serial with its chunks, and then its cells, spread over the
    threads (each thread with scratch of its own)."""
    nq = live.shape[0]
    out = np.zeros(nq, np.bool_)
    und[:nq] = False
    gqn = np.empty(nq)
    for g0 in range(0, nq, GROUP):
        g1 = min(g0 + GROUP, nq)
        nc = (g1 - g0 + CHUNK - 1) // CHUNK
        room = CHUNK * PAIR_ROWS if g1 - g0 >= PAIRS_MIN else 0
        pq = np.empty(nc * room, np.int64)
        pr = np.empty(nc * room, np.int32)
        npair = np.zeros(nc, np.int64)
        for c in prange(nc):
            c0 = g0 + c * CHUNK
            S = np.empty((min(CHUNK, nq), XC.shape[1]), np.float32)  # hoisted: once per thread
            _chunk(c0, min(c0 + CHUNK, g1), live, out, und, Q64, Yh, qpar, Eh, iu, stu, XC, TQ, dl, TC, CM,
                   cth, sth, fmap, Mh, Mt, cellU, gate_add, cs, kind, Xf, thf, cbuf, kbuf, abuf, books, nbuf, L,
                   alo, astep, tol, pq[c * room:(c + 1) * room], pr[c * room:(c + 1) * room], npair[c:c + 1], gqn,
                   S)
        if npair.sum():
            start, qof = _sort_pairs(pq, pr, npair, room, XC.shape[1])
            for c in prange(Mh.shape[0]):
                v = np.empty(Q64.shape[1], np.float32)              # hoisted: once per thread
                ixs = np.empty(Q64.shape[1], np.int64)
                _pairs(cs[c], cs[c + 1], start, qof, out, und, Q64, qpar, gqn, fmap, kind, Xf, thf, cbuf, kbuf,
                       abuf, books, nbuf, L, alo, astep, tol, v, ixs)
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
    """Bytes KernelData holds at head width h, for caps padded to Np rows in K
    cells, with locators below n_ids."""
    per_row = h + 1 + 1 + np.dtype(locator_dtype(n_ids)).itemsize # head, tails, angle code, locator
    return Np * per_row + d * h * 4 + K * (h * 4 + 4 + 8 + 8 + 1) + 2 * N_CODES * 8 + 8


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
    """The cap-side arrays every kernel call takes, built once per index.

    U64: unit cap directions (float64, working basis), theta: their angles,
    gid: their rows in the exact test's source -- in storage order, used at
    build time only; centroids64, cellU, cell_starts: the IVF cells; src: the
    exact test's arguments (kind, Xf, thf, cbuf, kbuf, abuf, books, nbuf, L,
    alo, astep, tol; see CapIVFIndex), of which the index owns `owned` bytes;
    E: the PCA basis; h: head width.  Stored per cap: the int8 PCA head (coordinate-major, h x Np),
    the uint8 tail-norm code, a uint8 angle code (theta rounded up to a 255-step grid)
    and the uint16/uint32 locator; per cell its widest angle code.  Storage is
    padded per cell (see PAD); pad rows have angle code PAD_CODE."""

    def __init__(self, U64: np.ndarray, theta: np.ndarray, gid: np.ndarray, centroids64: np.ndarray,
                 cellU: np.ndarray, cell_starts: np.ndarray, src: tuple, owned: int, E: np.ndarray, h: int):
        N, d = U64.shape
        self.d = d
        self.src = src
        self.owned = owned
        Eh = np.zeros((d, h))
        Eh[:, :min(h, d)] = E[:, :min(h, d)]
        self.Eh = Eh.astype(np.float32)
        Eh = self.Eh.astype(np.float64) # the basis the kernel actually uses
        Y = U64 @ Eh
        tails = np.sqrt(np.maximum(np.einsum("ij,ij->i", U64, U64) - np.einsum("ij,ij->i", Y, Y), 0.0))

        cs = np.asarray(cell_starts, dtype=np.int64)
        counts = np.diff(cs)
        padded = -(-counts // PAD) * PAD
        self.cs = np.concatenate(([0], np.cumsum(padded))).astype(np.int64)
        Np = int(self.cs[-1])
        cell_of = np.repeat(np.arange(len(counts)), counts)
        pos = self.cs[cell_of] + (np.arange(N) - cs[cell_of])

        # int8 head, one scale; dlk[k] bounds every cap's rounding over its
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
        # uint8 angle codes: theta rounded up on a grid of N_CODES values over [min, max]
        theta = np.asarray(theta, dtype=np.float64)
        lo, hi = (float(theta.min()), float(theta.max())) if N else (0.0, 0.0)
        grid = lo + np.arange(N_CODES) * ((hi - lo) / (N_CODES - 1))
        grid[-1] = hi
        tc = np.minimum(np.searchsorted(grid, theta, side="left"), N_CODES - 1)
        assert (grid[tc] >= theta).all()
        self.TC = np.full(Np, PAD_CODE, np.uint8)
        self.TC[pos] = tc
        self.CM = np.zeros(len(counts), np.uint8) # each cell's widest angle code
        np.maximum.at(self.CM, cell_of, tc.astype(np.uint8))
        self.cth = np.cos(grid)
        self.sth = np.sin(grid)
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
                self.run(np.zeros(nq, np.bool_), np.zeros((nq, d)), np.zeros(nq), 0.0, parallel=par)

    def undecided(self):
        """Rows of the last batch left undecided (to be decided by the caller's
        exact test), or None."""
        return np.flatnonzero(self._und[:self._nq]) if self._cnt[0] else None

    def run(self, live, Q64, qpar, gate_add, parallel):
        """Accept mask (qpar: each query's beta); see undecided() for the
        queries it could not decide.  From BLAS_MIN queries up, their heads
        come from one BLAS product."""
        nq = len(live)
        self._nq = nq
        if len(self._und) < nq:
            self._und = np.empty(max(nq, 2 * len(self._und)), np.bool_)
        Yh = Q64 @ self.Eh.astype(np.float64) if nq >= BLAS_MIN else np.zeros((0, 1))
        args = ((live, Q64, Yh, qpar, self.Eh, 1.0 / self.su, self.st / self.su, self.XC, self.TQ,
                 self.dl, self.TC, self.CM, self.cth, self.sth, self.fmap, self.Mh, self.Mt, self.cellU,
                 gate_add, self.cs) + self.src + (self._und, self._cnt))
        return (accepts_parallel if parallel else accepts_serial)(*args)

    def layout(self):
        """(storage rows of the real caps, their cells, their source rows):
        recomputed from the sketch (pad rows carry PAD_CODE), for diagnostics."""
        pos = np.flatnonzero(self.TC != PAD_CODE)
        return pos, np.searchsorted(self.cs, pos, side="right") - 1, self.fmap[pos].astype(np.int64)

    def head(self, pos):
        """The stored int8 head of storage rows pos, in code units [len(pos), h]."""
        return self.XC[:, pos].T.astype(np.float64)

    def survivor_fraction(self, Q64, qpar, gate_add, stage: str = "full", chunk: int = 256) -> np.ndarray:
        """Diagnostics (numpy, float64, on the sketch with its bounds): per query,
        the fraction of caps that pass the IVF gate (stage="gate") or the gate
        and the bound (*), i.e. reach the exact test
        (stage="full")."""
        su = self.su
        pos, cell_of, _ = self.layout()
        X = self.head(pos) * su
        Mh = self.Mh.astype(np.float64) * su
        th = np.arctan2(self.sth, self.cth)[self.TC[pos]]
        E64 = self.Eh.astype(np.float64)
        out = np.empty(len(Q64))
        for s in range(0, len(Q64), chunk):
            Q = np.asarray(Q64[s:s + chunk], dtype=np.float64)
            b = np.asarray(qpar[s:s + chunk], dtype=np.float64)
            Y = Q @ E64
            qn2 = np.einsum("ij,ij->i", Q, Q)
            qn = np.sqrt(qn2)[:, None]
            run = np.concatenate((np.zeros((len(Y), 1)), np.cumsum(Y * Y, axis=1)), axis=1)
            qtail = lambda p: np.sqrt(np.maximum(qn2 - run[:, p], 0.0))[:, None]
            h = Y.shape[1]
            lbc = np.maximum((Y * Y).sum(1)[:, None] + (Mh * Mh).sum(1)[None] - 2.0 * Y @ Mh.T, 0.0)
            lbc += (qtail(h) - self.Mt[None].astype(np.float64) * su) ** 2
            A = self.cellU[None] + gate_add + b[:, None]
            with np.errstate(invalid="ignore"):
                g2 = np.where(A >= np.pi, np.inf, (qn2 + 1.0)[:, None] - 2.0 * qn * np.cos(A))
            alive = (lbc <= g2)[:, cell_of]
            if stage == "full":
                R2 = (qn2 + 1.0)[:, None] - 2.0 * qn * np.cos(np.minimum(th[None] + b[:, None], np.pi))
                S = np.maximum((Y * Y).sum(1)[:, None] + (X * X).sum(1)[None] - 2.0 * Y @ X.T, 0.0)
                hs = np.maximum(np.sqrt(S) - self.dl * su, 0.0)
                lo = self.TQ[pos].astype(np.float64)[None] * self.st
                qt = qtail(Y.shape[1])
                td = np.maximum(np.maximum(lo - qt, qt - lo - self.st), 0.0)
                alive &= hs ** 2 + td ** 2 <= R2
            out[s:s + len(Y)] = alive.mean(axis=1)
        return out

    def nbytes(self) -> int:
        """Bytes the index holds: the sketch, the PCA head basis, the cells, the
        angle grid, and the exact test's arrays it owns (none when it reads CapCoverFilter's codes)."""
        return int(sum(a.nbytes for a in (self.Eh, self.XC, self.TQ, self.TC, self.CM,
                        self.fmap, self.cth, self.sth, self.Mh, self.Mt, self.cellU, self.cs))) + self.owned