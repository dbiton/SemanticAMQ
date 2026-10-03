import numpy as np
from numba import njit


# Compiled pieces of CapCoverFilter's query scan. Each call decodes the caps
# it scans into a small float32 block (nothing decoded persists between
# calls: only codes are stored, which is what get_bytesize() charges),
# scores the block against the queries with one BLAS GEMM, and decides every
# (query, cap) pair in one compiled pass.
#
# Both code families the filter auto-selects are little-endian bit streams,
# code j at bits [j * bits, (j + 1) * bits):
#   - TurboQuant (faiss QT_<b>bit_tqmse): one code per dimension, decoded
#     through the quantizer's shared table of 2^b levels;
#   - seeded PQ (_SeededPQ): one code per `sub`-dimension slice, decoded to
#     that slice's codeword.
# Each decoder also returns every row's norm, accumulated in float64.
#
# Exactness. The GEMM scores q . raw_j in float32, so the rule
# q . raw_j >= |raw_j| cos(theta_j + beta_q) is decided outright only when
# the score clears the threshold by more than slack * |raw_j|, slack the
# float32 error bound the filter uses (CapCoverFilter._slack); pairs within
# it are re-measured in float64 exactly as CapCoverFilter.surface_angle()
# measures them -- angle(q, raw_j / |raw_j|) <= theta_j + beta_q.


@njit(nogil=True, cache=True)
def decode_levels(codes, levels, bits, dim, out, norms):
    """TurboQuant codes -> out[r, i] = levels[code of dimension i]."""
    mask = (1 << bits) - 1
    for r in range(codes.shape[0]):
        acc = 0
        nacc = 0
        b = 0
        ss = 0.0
        for i in range(dim):
            while nacc < bits:
                acc |= np.int64(codes[r, b]) << nacc
                b += 1
                nacc += 8
            v = levels[acc & mask]
            acc >>= bits
            nacc -= bits
            out[r, i] = v
            ss += np.float64(v) * np.float64(v)
        norms[r] = np.sqrt(ss)


@njit(nogil=True, cache=True)
def decode_codebook(codes, codebook, bits, m, out, norms):
    """Seeded-PQ codes -> out[r, j * sub : (j + 1) * sub] = codebook[code j]."""
    mask = (1 << bits) - 1
    sub = codebook.shape[1]
    for r in range(codes.shape[0]):
        acc = 0
        nacc = 0
        b = 0
        ss = 0.0
        for j in range(m):
            while nacc < bits:
                acc |= np.int64(codes[r, b]) << nacc
                b += 1
                nacc += 8
            c = acc & mask
            acc >>= bits
            nacc -= bits
            for t in range(sub):
                v = codebook[c, t]
                out[r, j * sub + t] = v
                ss += np.float64(v) * np.float64(v)
        norms[r] = np.sqrt(ss)


@njit(nogil=True, cache=True)
def _exact_angle(q, raw, norm):
    """angle(q, raw / norm) in float64 by the chord formula (angle_between)."""
    s = 0.0
    for i in range(q.shape[0]):
        d = q[i] - np.float64(raw[i]) / norm
        s += d * d
    half = 0.5 * np.sqrt(s)
    if half > 1.0:
        half = 1.0
    return 2.0 * np.arcsin(half)


@njit(nogil=True, cache=True)
def accept_tables(codes, bits, m, T, sq_norms, codebook, theta, rows, beta, Qn, slack, out):
    """query() over seeded-PQ codes without decoding: T[s, c, i] holds query
    i's slice s against codeword c (float32), so q . raw_j is the sum of m
    table rows -- one row read per slice for the whole batch of queries --
    and |raw_j|^2 the sum of the codewords' squared norms. Decides each
    pair as accept_block does; a pair within the error bound is decoded
    and re-measured in float64. The table sums carry at most
    (sub + m) * 2^-24 * |raw_j| of error for unit queries, inside slack."""
    nq = T.shape[2]
    sub = codebook.shape[1]
    dim = m * sub
    mask = (1 << bits) - 1
    cb = np.empty(nq)
    sb = np.empty(nq)
    live = 0
    for i in range(nq):
        cb[i] = np.cos(beta[rows[i]])
        sb[i] = np.sin(beta[rows[i]])
        if not out[rows[i]]:
            live += 1
    acc = np.empty(nq, dtype=np.float32)
    idx = np.empty(m, dtype=np.int64)
    raw = np.empty(dim, dtype=np.float32)
    for j in range(codes.shape[0]):
        if live == 0:
            break
        a = 0
        nacc = 0
        b = 0
        ss = 0.0
        for s in range(m):
            while nacc < bits:
                a |= np.int64(codes[j, b]) << nacc
                b += 1
                nacc += 8
            idx[s] = a & mask
            a >>= bits
            nacc -= bits
            ss += sq_norms[idx[s]]
        n = np.sqrt(ss)
        acc[:] = T[0, idx[0], :]
        for s in range(1, m):
            acc += T[s, idx[s], :]
        ct = np.cos(theta[j])
        st = np.sin(theta[j])
        decoded = False
        for i in range(nq):
            q = rows[i]
            if out[q]:
                continue
            if theta[j] + beta[q] >= np.pi:
                out[q] = True
                live -= 1
                continue
            thr = n * (cb[i] * ct - sb[i] * st)
            v = np.float64(acc[i])
            margin = slack * n
            if v >= thr + margin:
                out[q] = True
                live -= 1
            elif v > thr - margin:
                if not decoded:
                    for s in range(m):
                        for t in range(sub):
                            raw[s * sub + t] = codebook[idx[s], t]
                    decoded = True
                if _exact_angle(Qn[q], raw, n) <= theta[j] + beta[q]:
                    out[q] = True
                    live -= 1


@njit(nogil=True, cache=True, fastmath={"reassoc", "contract"})
def accept_direct(codes, bits, m, codebook, sq_norms, theta, rows, beta, Qn, slack, out):
    """query() over seeded-PQ codes for a few queries at a time, with no
    decode and no table: per query, each cap's slices are scored against
    the query's slices straight from the codebook (float64 sums of float32
    codewords times the float64 query), and a cap is given up once even a
    perfectly aligned rest could not reach its threshold -- the dot so far
    plus |q_rest| |c_rest| (Cauchy-Schwarz; |c_rest|^2 from the codewords'
    squared norms) below the threshold by more than the margin. Decides
    each pair as accept_block does: outright outside slack * |raw|, else
    by the float64 angle to the decoded vector."""
    sub = codebook.shape[1]
    dim = m * sub
    mask = (1 << bits) - 1
    idx = np.empty(m, dtype=np.int64)
    raw = np.empty(dim, dtype=np.float32)
    rest_q = np.empty(m + 1)
    nc = codes.shape[0]
    for i in range(rows.shape[0]):
        q = rows[i]
        if out[q]:
            continue
        b = beta[q]
        cb = np.cos(b)
        sb = np.sin(b)
        qv = Qn[q]
        # rest_q[s] = |q's slices s..m-1|
        acc = 0.0
        rest_q[m] = 0.0
        for s in range(m - 1, -1, -1):
            e = 0.0
            for t in range(sub):
                e += qv[s * sub + t] * qv[s * sub + t]
            acc += e
            rest_q[s] = np.sqrt(acc)
        for j in range(nc):
            if theta[j] + b >= np.pi:
                out[q] = True
                break
            a = 0
            nacc = 0
            bb = 0
            n2 = 0.0
            for s in range(m):
                while nacc < bits:
                    a |= np.int64(codes[j, bb]) << nacc
                    bb += 1
                    nacc += 8
                idx[s] = a & mask
                a >>= bits
                nacc -= bits
                n2 += sq_norms[idx[s]]
            n = np.sqrt(n2)
            thr = n * (cb * np.cos(theta[j]) - sb * np.sin(theta[j]))
            margin = slack * n
            dot = 0.0
            left = n2
            dropped = False
            for s in range(m):
                c = idx[s]
                d = 0.0
                for t in range(sub):
                    d += np.float64(codebook[c, t]) * qv[s * sub + t]
                dot += d
                left -= sq_norms[c]
                if (s & 3) == 3 and dot + rest_q[s + 1] * np.sqrt(max(left, 0.0)) < thr - margin:
                    dropped = True
                    break
            if dropped:
                continue
            if dot >= thr + margin:
                out[q] = True
                break
            if dot > thr - margin:
                for s in range(m):
                    for t in range(sub):
                        raw[s * sub + t] = codebook[idx[s], t]
                if _exact_angle(qv, raw, n) <= theta[j] + b:
                    out[q] = True
                    break


@njit(nogil=True, cache=True, fastmath={"reassoc", "contract"})
def accept_direct_levels(codes, levels, bits, dim, theta, rows, beta, Qn, slack, out):
    """accept_direct for TurboQuant codes: per query, each cap's q . raw and
    |raw|^2 accumulate in one pass over its codes (levels[code] per
    dimension, float64 sums), with no decoded block written out. Decides
    each pair as accept_block does; boundary pairs are decoded and
    re-measured in float64."""
    mask = (1 << bits) - 1
    raw = np.empty(dim, dtype=np.float32)
    nc = codes.shape[0]
    for i in range(rows.shape[0]):
        q = rows[i]
        if out[q]:
            continue
        b = beta[q]
        cb = np.cos(b)
        sb = np.sin(b)
        qv = Qn[q]
        for j in range(nc):
            if theta[j] + b >= np.pi:
                out[q] = True
                break
            acc = 0
            nacc = 0
            bb = 0
            dot = 0.0
            n2 = 0.0
            for t in range(dim):
                while nacc < bits:
                    acc |= np.int64(codes[j, bb]) << nacc
                    bb += 1
                    nacc += 8
                v = np.float64(levels[acc & mask])
                acc >>= bits
                nacc -= bits
                dot += v * qv[t]
                n2 += v * v
            n = np.sqrt(n2)
            thr = n * (cb * np.cos(theta[j]) - sb * np.sin(theta[j]))
            margin = slack * n
            if dot >= thr + margin:
                out[q] = True
                break
            if dot > thr - margin:
                acc = 0
                nacc = 0
                bb = 0
                for t in range(dim):
                    while nacc < bits:
                        acc |= np.int64(codes[j, bb]) << nacc
                        bb += 1
                        nacc += 8
                    raw[t] = levels[acc & mask]
                    acc >>= bits
                    nacc -= bits
                if _exact_angle(qv, raw, n) <= theta[j] + b:
                    out[q] = True
                    break


@njit(nogil=True, cache=True)
def accept_block(S, rows, raw, norms, theta, beta, Qn, slack, out):
    """Marks out[rows[i]] for each query i of the block S (queries x caps,
    float32 q . raw) that some cap accepts: theta_j + beta_q >= pi, or
    q . raw_j >= |raw_j| cos(theta_j + beta_q) with float32 error ruled
    out, else by the float64 angle. Stops at a query's first accepting cap."""
    nq, kb = S.shape
    ct = np.cos(theta)
    st = np.sin(theta)
    for i in range(nq):
        q = rows[i]
        if out[q]:
            continue
        b = beta[q]
        cb = np.cos(b)
        sb = np.sin(b)
        for j in range(kb):
            if theta[j] + b >= np.pi:
                out[q] = True
                break
            n = norms[j]
            thr = n * (cb * ct[j] - sb * st[j])
            s = np.float64(S[i, j])
            margin = slack * n
            if s >= thr + margin:
                out[q] = True
                break
            if s > thr - margin:
                if _exact_angle(Qn[q], raw[j], n) <= theta[j] + b:
                    out[q] = True
                    break
