"""Ball-cover geometry shared by BallCoverFilter and the streaming insertion
methods in streaming_insertion.py (Section 2.3's merge rule + the log-space
covering objective). Extracted as free functions parameterized by `dim`
directly (no instance state) so both call sites can share one implementation
without either depending on the other.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.special import logsumexp

_CHUNK_SIZE = 20_000
_RADIUS_SAFETY_REL = 1e-6
_RADIUS_SAFETY_ABS = 1e-6


def safe_radius(r: float) -> float:
    """Inflate a freshly computed radius by a tiny margin -- same rationale
    and constants as BallCoverFilter._safe_radius: a radius from a direct
    sqrt(sum((x-c)^2)) can come out a few ULPs smaller than what the
    "expand" distance formula (query-time, BLAS-batched) later measures for
    that exact same point, silently breaking FNR=0 at tau=0. Many orders of
    magnitude below any realistic tau, so it never affects observable
    recall."""
    r = float(r)
    return r * (1.0 + _RADIUS_SAFETY_REL) + _RADIUS_SAFETY_ABS


def merged_ball(ci: np.ndarray, ri: float, cj: np.ndarray, rj: float) -> Tuple[np.ndarray, float]:
    """Merge two balls B_i=(ci,ri), B_j=(cj,rj): if one contains the other,
    the merge is that ball unchanged; otherwise the externally-tangent ball
    covering both. Verbatim geometry from BallCoverFilter._merged_ball."""
    ci64 = np.asarray(ci, dtype=np.float64)
    cj64 = np.asarray(cj, dtype=np.float64)
    ri = float(ri)
    rj = float(rj)
    delta = float(np.sqrt(np.maximum(np.sum((ci64 - cj64) ** 2), 0.0)))
    if delta + ri <= rj:
        return cj64, rj
    if delta + rj <= ri:
        return ci64, ri
    r_new = (delta + ri + rj) / 2.0
    c_new = ci64 + ((r_new - ri) / delta) * (cj64 - ci64)
    return c_new, r_new


def log_J(radii: np.ndarray, dim: int) -> float:
    """log sum_i r_i^d, safe for d ~ hundreds where r^d over/underflows float64."""
    r = np.asarray(radii, dtype=np.float64)
    if r.size == 0:
        return -np.inf
    with np.errstate(divide='ignore'):
        logr = np.log(r)
    return float(logsumexp(dim * logr))


def log_J_tau(radii: np.ndarray, dim: int, tau: float) -> float:
    r = np.asarray(radii, dtype=np.float64) + float(tau)
    if r.size == 0:
        return -np.inf
    with np.errstate(divide='ignore'):
        logr = np.log(r)
    return float(logsumexp(dim * logr))


def merge_cost_log_space(ri: np.ndarray, rj: np.ndarray, r_merged: np.ndarray,
                         dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """Delta = M^d * (a_ij - a_i - a_j) with M = max(r_i, r_j, r_ij) and
    a_x = exp(d * log(r_x / M)) (safe underflow to 0); never materializes r^d.
    bucket 0 = beneficial merge (cost < 0), key ascending = most beneficial
    first; bucket 1 = neutral/harmful, key ascending = least harmful first."""
    M = np.maximum(np.maximum(ri, rj), r_merged)

    def a_of(r):
        # r<=0 (zero-radius ball) substitutes M for r, not 1.0: numpy still
        # eagerly evaluates the discarded branch, and 1.0 >> M would overflow
        # exp(); M makes it exp(0) = 1 instead, always safe.
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            safe_M = np.where(M > 0, M, 1.0)
            safe_r = np.where(r > 0, r, safe_M)
            ratio = np.where((r > 0) & (M > 0), np.exp(dim * (np.log(safe_r) - np.log(safe_M))), 0.0)
        return ratio

    a_i, a_j, a_ij = a_of(ri), a_of(rj), a_of(r_merged)
    diff_a = a_ij - a_i - a_j
    sign = np.sign(diff_a)
    with np.errstate(divide='ignore'):
        log_abs = dim * np.log(np.where(M > 0, M, 1.0)) + np.log(np.where(diff_a != 0, np.abs(diff_a), 1.0))
    log_abs = np.where(diff_a != 0, log_abs, -np.inf)
    bucket = np.where(sign < 0, 0, 1).astype(np.int8)
    key = np.where(sign < 0, -log_abs, log_abs)
    return bucket, key


def pairwise_merge_geometry(ci: np.ndarray, ri: np.ndarray, cj: np.ndarray, rj: np.ndarray,
                            dim: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized (bucket, key, r_merged) for merging ball(s) (ci,ri) with
    ball(s) (cj,rj) under the log-space objective -- the only merge-cost
    criterion the streaming methods use (see gonzales_filter.py's other
    MERGE_CRITERIA for the retired alternatives). Broadcasts like numpy: pass
    a single ball (shape (dim,)) against many (shape (k,dim)) for a one-vs-k
    row, or full (n,1,dim) x (1,n,dim) grids for an n x n matrix."""
    diff = ci - cj
    delta = np.sqrt(np.maximum(np.sum(diff * diff, axis=-1), 0.0))
    contain_j_in_i = delta + rj <= ri
    contain_i_in_j = delta + ri <= rj
    r_general = (delta + ri + rj) / 2.0
    r_merged = np.where(contain_i_in_j, rj, np.where(contain_j_in_i, ri, r_general))
    bucket, key = merge_cost_log_space(ri, rj, r_merged, dim)
    return bucket, key, r_merged


def argmin_bucket_key(bucket: np.ndarray, key: np.ndarray) -> int:
    """Flat index of the preferred (bucket, key) pair: bucket 0 (beneficial)
    before bucket 1 (neutral/harmful), ascending key within a bucket -- same
    tie-break convention as BallCoverFilter._merge_cache_best_pair."""
    bucket = np.asarray(bucket).ravel()
    key = np.asarray(key).ravel()
    mask0 = bucket == 0
    if np.any(mask0):
        key_eff = np.where(mask0, key, np.inf)
    else:
        key_eff = np.where(bucket == 1, key, np.inf)
    return int(np.argmin(key_eff))


def best_bucket_key(bucket: np.ndarray, key: np.ndarray) -> Tuple[int, float]:
    """(bucket, key) of the preferred entry -- a comparable summary of "the
    best available merge cost in this candidate set", usable to compare two
    different candidate sets (e.g. M3's restricted vs. unrestricted search)
    via plain tuple ordering."""
    i = argmin_bucket_key(bucket, key)
    bucket = np.asarray(bucket).ravel()
    key = np.asarray(key).ravel()
    return int(bucket[i]), float(key[i])


def delta_from_bucket_key(bucket: int, key: float) -> float:
    """Reconstruct the realized signed Delta = A(merged) - A(i) - A(j) from
    the (bucket, key) encoding merge_cost_log_space produces (magnitude kept
    in log space there specifically to avoid materializing r^d; this inverts
    it back to a real number for reporting/statistics only -- never for a
    decision, so overflow to +-inf here is an acceptable, informative
    outcome, not a bug)."""
    if bucket == 0:
        log_abs, sign = -key, -1.0
    else:
        log_abs, sign = key, 1.0
    if log_abs == -np.inf:
        return 0.0
    with np.errstate(over='ignore'):
        return float(sign * np.exp(log_abs))


def sq_dists_to_centers(vectors: np.ndarray, sq_norms: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """n x k squared L2 distances, accumulated in float64 (avoids catastrophic
    cancellation for points close to their center at d ~ hundreds)."""
    centers64 = centers.astype(np.float64)
    vectors64 = vectors.astype(np.float64)
    sq_norms64 = np.asarray(sq_norms, dtype=np.float64)
    c_sq = np.einsum('ij,ij->i', centers64, centers64)
    cross = vectors64 @ centers64.T
    return sq_norms64[:, None] - 2.0 * cross + c_sq[None, :]


def ball_cover_query(centers: np.ndarray, radii: np.ndarray, q: np.ndarray, tau: float,
                     chunk_size: int = _CHUNK_SIZE) -> np.ndarray:
    """Batched containment test at eff_r = radius + tau, chunked over q the
    same way BallCoverFilter.query is."""
    n = len(q)
    if len(centers) == 0 or n == 0:
        return np.zeros(n, dtype=bool)

    q = np.ascontiguousarray(q, dtype=np.float32)
    eff_r = np.asarray(radii, dtype=np.float64) + float(tau)
    q_sq_norms = np.einsum('ij,ij->i', q, q, dtype=np.float64)

    result = np.zeros(n, dtype=bool)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        sq_d = sq_dists_to_centers(q[start:end], q_sq_norms[start:end], centers)
        result[start:end] = np.any(sq_d <= (eff_r[None, :]) ** 2, axis=1)
    return result
