"""
Tight lower/upper bounds on the volume of a union of hyperspheres.

Improvements over the naive bounds (max single volume, sum of volumes):

1.  All work is done in log/scaled space. In dim up to 1000, pi^(d/2)/Gamma(d/2+1)
    and r^d overflow/underflow float64 catastrophically, so raw volumes are useless.

2.  Spheres fully contained in another sphere are removed: they change neither
    the union nor the bounds, but they inflate the naive upper bound.

3.  The exact set of intersecting pairs is found with a (radius-bucketed) faiss
    range search, and the exact d-dimensional "lens" volume of every intersecting
    pair is computed analytically via the regularized incomplete beta function.

4.  The intersection graph is split into connected components; components are
    independent, so bounds are computed per component and summed. Per component C:

      lower(C) = max( max_i V_i ,  sum_i V_i - sum_{(i,j) in C} V_ij )
                 (degree-2 Bonferroni inequality, valid because we have ALL pairs)

      upper(C) = sum_i V_i - weight(maximum spanning forest of C, edge weight V_ij)
                 (Hunter's bound: valid for any tree subgraph of the intersection
                  graph; the max-weight spanning tree gives the tightest one)

    Singleton components are exact, and 2-sphere components are exact as well
    (lower == upper == V1 + V2 - V12), which generalizes the special-casing of
    "non-intersecting" and "intersecting once" spheres.

Complexity: pair discovery is the dominant cost, O(n^2 * d) through faiss's
BLAS-optimized exhaustive range search (fine up to n ~ 1e5 with threads).
Everything downstream is O(P log n) in the number P of truly intersecting pairs.
"""

from typing import Tuple

import numpy as np
import faiss
from scipy.special import betainc, gammaln
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree


# ----------------------------------------------------------------------------- volumes

def log_hypersphere_volume(dim: int, radius: np.ndarray) -> np.ndarray:
    """log of the volume of a dim-ball; safe for dim up to (at least) 1000."""
    radius = np.asarray(radius, dtype=np.float64)
    return (dim / 2.0) * np.log(np.pi) + dim * np.log(radius) - gammaln(dim / 2.0 + 1.0)


def calculate_hypersphere_volume(dim: int, radius: float) -> float:
    """Kept for API compatibility. Prefer log_hypersphere_volume for dim >~ 300."""
    return float(np.exp(log_hypersphere_volume(dim, radius)))


def _cap_fraction(r: np.ndarray, h: np.ndarray, dim: int) -> np.ndarray:
    """
    Fraction of a dim-ball's volume contained in a spherical cap of height h,
    0 <= h <= 2r (vectorized). Uses V_cap = 1/2 V_ball * I_x((d+1)/2, 1/2),
    x = (2 r h - h^2) / r^2, mirrored for h > r.
    """
    h = np.clip(h, 0.0, 2.0 * r)
    minor = h <= r
    hm = np.where(minor, h, 2.0 * r - h)              # height of the minor cap
    x = np.clip((2.0 * r * hm - hm * hm) / (r * r), 0.0, 1.0)
    f = 0.5 * betainc((dim + 1) / 2.0, 0.5, x)
    return np.where(minor, f, 1.0 - f)


def _pair_lens_scaled(d, r1, r2, w1, w2, dim):
    """
    Scaled volume (same scale as w1/w2) of the intersection lens of two balls
    with center distance d < r1 + r2. Handles containment as min(w1, w2).
    """
    contained = d <= np.abs(r1 - r2)
    d_safe = np.where(d > 0, d, 1.0)
    c1 = (d_safe * d_safe + r1 * r1 - r2 * r2) / (2.0 * d_safe)
    h1 = r1 - c1                    # cap height on sphere 1
    h2 = r2 - (d_safe - c1)         # cap height on sphere 2
    lens = w1 * _cap_fraction(r1, h1, dim) + w2 * _cap_fraction(r2, h2, dim)
    return np.where(contained, np.minimum(w1, w2), lens)


# ----------------------------------------------------------------------------- pair discovery

def _find_intersecting_pairs(centroids: np.ndarray, radii: np.ndarray,
                             n_buckets: int = 8, exact: bool = True):
    """
    Return (i, j, d) for every pair with ||c_i - c_j|| < r_i + r_j, i < j.
    Exact mode: queries are bucketed by radius so each bucket uses the search
    threshold (bucket_r_max + global_r_max)^2, which cannot miss a pair; the
    coarse float32 candidates are then re-verified in float64.
    Approximate mode (exact=False): an IVF index prunes the O(n^2 d) scan; it
    may miss a few pairs, which keeps the upper bound valid (Hunter's bound
    holds for any subgraph) but makes the lower bound heuristic.
    """
    n, dim = centroids.shape
    xb = np.ascontiguousarray(centroids, dtype=np.float32)
    if exact or n < 20_000:
        index = faiss.IndexFlatL2(dim)
    else:
        nlist = int(np.sqrt(n)) * 4
        index = faiss.IndexIVFFlat(faiss.IndexFlatL2(dim), dim, nlist)
        index.train(xb)
        index.nprobe = max(8, nlist // 16)
    index.add(xb)

    r_max = float(radii.max())
    order = np.argsort(radii)                    # bucket queries by radius so small
    ii, jj = [], []                              # spheres don't pay r_max twice
    for chunk in np.array_split(order, min(n_buckets, n)):
        if chunk.size == 0:
            continue
        thr = (float(radii[chunk].max()) + r_max) ** 2
        thr = thr * 1.001 + 1e-9                 # slack for float32 rounding
        lims, _, I = index.range_search(xb[chunk], thr)
        q = np.repeat(chunk, np.diff(lims).astype(np.int64))
        keep = q < I                             # drop self-matches + dedupe (i < j)
        ii.append(q[keep].astype(np.int64))
        jj.append(I[keep].astype(np.int64))

    i = np.concatenate(ii) if ii else np.empty(0, np.int64)
    j = np.concatenate(jj) if jj else np.empty(0, np.int64)
    if i.size == 0:
        return i, j, np.empty(0, np.float64)

    # exact float64 distances, chunked to bound memory at high dim
    d = np.empty(i.size, dtype=np.float64)
    step = max(1, (1 << 25) // max(dim, 1))      # ~256 MB per chunk
    for s in range(0, i.size, step):
        e = min(s + step, i.size)
        diff = centroids[i[s:e]] - centroids[j[s:e]]
        d[s:e] = np.sqrt(np.einsum("ij,ij->i", diff, diff))

    keep = d < radii[i] + radii[j]
    return i[keep], j[keep], d[keep]


# ----------------------------------------------------------------------------- main

def estimate_hypersphere_volume(centroids: np.ndarray,
                                radii: np.ndarray,
                                return_log: bool = False,
                                exact: bool = True) -> Tuple[float, float]:
    """
    Tight (lower_bound, upper_bound) on the volume of the union of the spheres.

    centroids  : (n, dim) array
    radii      : (n,) array of positive radii
    return_log : if True, return (log_lower, log_upper). Strongly recommended for
                 dim >~ 300, where the true volume itself under/overflows float64
                 (e.g. a unit ball in dim=1000 has volume ~ 1e-711) and the linear
                 result would degenerate to 0.0 or inf.
    exact      : True guarantees both bounds. False switches pair discovery to an
                 IVF index, which is much faster at n ~ 1e5 x dim ~ 1e3; the upper
                 bound remains a true bound, the lower bound becomes heuristic
                 (missed pairs could overestimate it).
    """
    centroids = np.asarray(centroids, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64).ravel()
    n, dim = centroids.shape
    if n == 0:
        return (-np.inf, -np.inf) if return_log else (0.0, 0.0)

    # scaled volumes: w_i = V_i / V_max, all math stays in [0, n]
    logV = log_hypersphere_volume(dim, radii)
    scale = float(logV.max())
    w = np.exp(logV - scale)

    # 1) all truly intersecting pairs
    i, j, d = _find_intersecting_pairs(centroids, radii, exact=exact)

    # 2) drop spheres fully contained in another (tie-break identical spheres by index)
    dropped = np.zeros(n, dtype=bool)
    if i.size:
        ri, rj = radii[i], radii[j]
        j_in_i = (d + rj <= ri) & ((ri > rj) | ((ri == rj) & (i < j)))
        i_in_j = (d + ri <= rj) & ((rj > ri) | ((rj == ri) & (j < i)))
        dropped[j[j_in_i]] = True
        dropped[i[i_in_j]] = True
        keep = ~(dropped[i] | dropped[j])
        i, j, d = i[keep], j[keep], d[keep]
    active = ~dropped

    # 3) exact pairwise lens volumes (scaled)
    wij = _pair_lens_scaled(d, radii[i], radii[j], w[i], w[j], dim) if i.size \
        else np.empty(0, np.float64)

    # 4) connected components of the intersection graph
    if i.size:
        adj = coo_matrix((np.ones(i.size), (i, j)), shape=(n, n))
        n_comp, labels = connected_components(adj, directed=False)
    else:
        n_comp, labels = n, np.arange(n)

    S = np.bincount(labels[active], weights=w[active], minlength=n_comp)   # sum V_i
    M = np.zeros(n_comp)                                                   # max V_i
    np.maximum.at(M, labels[active], w[active])

    P = np.zeros(n_comp)                                                   # sum V_ij
    F = np.zeros(n_comp)                                                   # max spanning forest
    if i.size:
        P = np.bincount(labels[i], weights=wij, minlength=n_comp)
        mst = minimum_spanning_tree(coo_matrix((-wij, (i, j)), shape=(n, n))).tocoo()
        if mst.nnz:
            F = np.bincount(labels[mst.row.astype(np.int64)],
                            weights=-mst.data, minlength=n_comp)

    upper_c = S - F                              # Hunter's spanning-forest bound
    lower_c = np.maximum(M, S - P)               # Bonferroni-2, floored by max volume
    lower_c = np.minimum(lower_c, upper_c)       # numerical safety

    low = float(lower_c.sum())
    high = float(upper_c.sum())

    def unscale(x: float) -> float:
        logx = scale + np.log(x) if x > 0.0 else -np.inf
        return float(logx) if return_log else float(np.exp(logx))

    return unscale(low), unscale(high)