import numpy as np
from numba import njit


# Compiled accept tests for BallCoverFilter.query(): every query against every
# ball, by exact squared L2 distance to the decoded center, with no decoded
# intermediate and no GEMM. Two things make them fast where faiss's scalar
# scans are not:
#
# - Early abandon. A squared distance is a sum of non-negative per-dimension
#   terms, so a partial sum past a ball's (r + tau + eps)^2 already rejects
#   it. Behind the quantizer's rotation the energy is spread evenly over the
#   dimensions, and a query sits ~1.35 from most balls, so on the real covers
#   a ball is rejected after 15-90 of 384 dimensions.
# - For 4-bit codes, a per-query lookup table: one 256-entry row per code
#   byte holding (q_2b - t_lo)^2 + (q_2b+1 - t_hi)^2, so a ball costs one
#   table load and one add per byte, with the code read 8 bytes at a time.
#
# Exactness. Every term is the square of a difference of two float32 values,
# formed in float64, so nothing cancels: a computed partial sum P of exact
# value D satisfies |P - D| <= _REL * D + _ABS, where _REL covers the float32
# rounding of each table entry (2^-24) plus float64 accumulation
# (code_size * 2^-53, negligible below millions of dimensions) and _ABS
# covers float32 underflow (at most 2^-150 per entry). A ball is therefore
#   - rejected only when P > reach^2 * (1 + _DELTA) + _ABS, which by that
#     bound implies D > reach^2 -- for a partial sum too, since the remaining
#     terms are >= 0;
#   - accepted outright only when the full sum is <= reach^2 * (1 - _DELTA) -
#     _ABS, which implies D <= reach^2;
#   - otherwise re-measured directly in float64, as _scan's boundary
#     refinement does.
# _DELTA = 2^-22 is twice the worst-case _REL, so accept decisions are those
# of the float64 distance to the decoded center, with no false negatives by
# construction rather than by measurement.
_DELTA = 2.0 ** -22
_ABS = 1e-36


@njit(nogil=True, cache=True)
def _reach_bounds(radii, tau, eps):
    """Per ball: reach^2 = (r + tau + eps)^2, the computed-sum threshold
    below which it accepts outright, and the one above which it is
    abandoned. All three are -1 for a negative reach, which admits nothing
    (squaring would lose the sign)."""
    n = radii.shape[0]
    reach_sq = np.empty(n, np.float64)
    lo = np.empty(n, np.float64)
    hi = np.empty(n, np.float64)
    for i in range(n):
        r = (np.float64(radii[i]) + tau) + eps
        if r < 0.0:
            reach_sq[i] = -1.0
            lo[i] = -1.0
            hi[i] = -1.0
        else:
            reach_sq[i] = r * r
            lo[i] = r * r * (1.0 - _DELTA) - _ABS
            hi[i] = r * r * (1.0 + _DELTA) + _ABS
    return reach_sq, lo, hi


@njit(nogil=True, cache=True)
def _nibble_exact(codes, i, q, T):
    x = 0.0
    for b in range(codes.shape[1]):
        c = codes[i, b]
        d0 = q[2 * b] - T[2 * b, c & 15]
        d1 = q[2 * b + 1] - T[2 * b + 1, c >> 4]
        x += d0 * d0 + d1 * d1
    return x


@njit(nogil=True, cache=True)
def nibble_accepts(codes, Qr, T, radii, tau, eps):
    """query() over 4-bit scalar-quantizer codes.

    codes: (n, nb) uint8, C-contiguous, nb a multiple of 8, in faiss's 4-bit
           layout -- dimension 2b in the low nibble of byte b, 2b + 1 in the
           high nibble.
    Qr:    (m, d) float32 queries, already in the quantizer's basis, d <= 2nb.
    T:     (2 * nb, 16) float64 decode table: T[j, v] is what faiss decodes
           nibble v of dimension j to. Rows past d must be zero; the query
           is zero there too, so those nibbles add exactly nothing.
    radii: (n,) float32 radius of each code's ball.

    Returns an (m,) bool mask: some ball has
    ||q - decode(c)||^2 <= (r + tau + eps)^2."""
    n, nb = codes.shape
    m, d = Qr.shape
    out = np.zeros(m, np.bool_)
    words = codes.view(np.uint64)
    reach_sq, lo, hi = _reach_bounds(radii, tau, eps)
    q = np.zeros(2 * nb, np.float64)
    A = np.zeros((2 * nb, 16), np.float64)
    L = np.empty((nb, 256), np.float32)
    for qi in range(m):
        for j in range(d):
            q[j] = Qr[qi, j]
        for j in range(2 * nb):
            for v in range(16):
                t = q[j] - T[j, v]
                A[j, v] = t * t
        for b in range(nb):
            for h in range(16):
                a = A[2 * b + 1, h]
                for v in range(16):
                    L[b, 16 * h + v] = a + A[2 * b, v]
        for i in range(n):
            h = hi[i]
            if h < 0.0:
                continue
            # Four accumulators break the add-latency chain; one abandon
            # check per 8-byte word.
            s0 = 0.0
            s1 = 0.0
            s2 = 0.0
            s3 = 0.0
            abandoned = False
            for w in range(nb // 8):
                x = words[i, w]
                b = 8 * w
                s0 += L[b, x & 255] + L[b + 4, (x >> 32) & 255]
                s1 += L[b + 1, (x >> 8) & 255] + L[b + 5, (x >> 40) & 255]
                s2 += L[b + 2, (x >> 16) & 255] + L[b + 6, (x >> 48) & 255]
                s3 += L[b + 3, (x >> 24) & 255] + L[b + 7, x >> 56]
                if (s0 + s1) + (s2 + s3) > h:
                    abandoned = True
                    break
            if abandoned:
                continue
            s = (s0 + s1) + (s2 + s3)
            if s <= lo[i] or _nibble_exact(codes, i, q, T) <= reach_sq[i]:
                out[qi] = True
                break
    return out


@njit(nogil=True, cache=True)
def _flat_exact(C, i, q):
    x = 0.0
    for j in range(C.shape[1]):
        t = q[j] - C[i, j]
        x += t * t
    return x


# Dimensions summed between two abandon checks in flat_accepts: one 64-byte
# cache line of a float32 center.
_FLAT_GROUP = 16


@njit(nogil=True, cache=True)
def flat_accepts(C, Qr, radii, tau, eps):
    """query() over raw float32 centers: C is (n, d) float32, the rest as in
    nibble_accepts. Each term (q_j - c_j)^2 is formed in float64 straight
    from the stored floats, so the same bounds hold with room to spare."""
    n, d = C.shape
    m = Qr.shape[0]
    out = np.zeros(m, np.bool_)
    reach_sq, lo, hi = _reach_bounds(radii, tau, eps)
    q = np.empty(d, np.float64)
    full = d - d % _FLAT_GROUP
    for qi in range(m):
        for j in range(d):
            q[j] = Qr[qi, j]
        for i in range(n):
            h = hi[i]
            if h < 0.0:
                continue
            s0 = 0.0
            s1 = 0.0
            s2 = 0.0
            s3 = 0.0
            abandoned = False
            for j0 in range(0, full, _FLAT_GROUP):
                for j in range(j0, j0 + _FLAT_GROUP, 4):
                    t0 = q[j] - C[i, j]
                    t1 = q[j + 1] - C[i, j + 1]
                    t2 = q[j + 2] - C[i, j + 2]
                    t3 = q[j + 3] - C[i, j + 3]
                    s0 += t0 * t0
                    s1 += t1 * t1
                    s2 += t2 * t2
                    s3 += t3 * t3
                if (s0 + s1) + (s2 + s3) > h:
                    abandoned = True
                    break
            if abandoned:
                continue
            for j in range(full, d):
                t = q[j] - C[i, j]
                s0 += t * t
            s = (s0 + s1) + (s2 + s3)
            if s > h:
                continue
            if s <= lo[i] or _flat_exact(C, i, q) <= reach_sq[i]:
                out[qi] = True
                break
    return out
