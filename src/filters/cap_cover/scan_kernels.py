import numpy as np
from numba import njit


# Compiled helper for construction planning (CapCoverFilter's query scan is Rust, see cap_scan.py).


@njit(nogil=True, cache=True, fastmath=True)
def nearest_codeword(S, CBt, out):
    """out[r] = the codeword (a column of CBt, sub x 2^bits) of largest inner
    product with row r of S (one slice of a direction per row): one row's
    2^bits scores stay in L1, where faiss's flat search over 6-48-dim
    slices runs well below GEMM speed (1.3-4.5x slower on one core). Ties
    and near-ties within float32 round-off may resolve differently from
    faiss (measured on 0-0.05% of directions); construct() uses it only to
    plan (CapCoverFilter._level_psi), never for the codes it stores."""
    K = CBt.shape[1]
    sub = S.shape[1]
    acc = np.empty(K, np.float32)
    for r in range(S.shape[0]):
        x0 = S[r, 0]
        for c in range(K):
            acc[c] = x0 * CBt[0, c]
        for t in range(1, sub):
            x = S[r, t]
            for c in range(K):
                acc[c] += x * CBt[t, c]
        best = 0
        bv = acc[0]
        for c in range(1, K):
            if acc[c] > bv:
                bv = acc[c]
                best = c
        out[r] = best
