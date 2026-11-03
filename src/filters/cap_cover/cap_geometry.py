import os
import numpy as np
import faiss
from typing import Tuple


# Shared geometry for CapCoverFilter and its strategies. Everything here works
# on unit vectors and angles: a cap (u, theta) is the set of unit vectors
# within angle theta of the unit direction u. Angles that coverage rests on
# are always measured in float64 by angle_between(); float32 inner products
# only ever *rank* candidates.


def unit_rows(X: np.ndarray) -> np.ndarray:
    """Rows of X scaled to unit length, in float64. Zero rows stay zero."""
    X = np.asarray(X, dtype=np.float64)
    norms = np.sqrt(np.einsum("ij,ij->i", X, X))
    return X / np.maximum(norms, 1e-300)[:, None]


def angle_between(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Row-wise angle between unit vectors, 2 * arcsin(|a - b| / 2): well
    conditioned at small angles, where arccos(<a, b>) loses half its digits."""
    diff = np.asarray(A, dtype=np.float64) - np.asarray(B, dtype=np.float64)
    chord = np.sqrt(np.einsum("...j,...j->...", diff, diff))
    return 2.0 * np.arcsin(np.minimum(1.0, 0.5 * chord))


try:
    import capcover_rs as _rs
    _FUSED = hasattr(_rs, "row_angles")
except ImportError:  # pure-NumPy fallback
    _rs, _FUSED = None, False


def tail_rust() -> bool:
    """Whether the construct-tail Rust kernels (construct_tail.rs) are used;
    env CAPCOVER_TAIL=python selects the original NumPy/faiss code. Needs a Rust knn kernel
    (AVX-512 VNNI or AVX2+FMA); CAPCOVER_KNN only picks the kNN path, see nearest_directions."""
    return (_rs is not None and hasattr(_rs, "knn_ip") and os.environ.get("CAPCOVER_TAIL", "rust") != "python"
            and _rs.knn_available_py())


def row_angles(X: np.ndarray, U: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """angle_between(X, U[labels]) without the gathered copy and difference
    temporaries (fused Rust kernel; float64, summation order differs from
    numpy's at the 1e-16 level)."""
    if _FUSED:
        return _rs.row_angles(np.ascontiguousarray(X, dtype=np.float64),
                              np.ascontiguousarray(U, dtype=np.float64),
                              np.ascontiguousarray(labels, dtype=np.int64))
    return angle_between(X, U[labels])


def pair_angles(X: np.ndarray, U: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """out[i, j] = angle(X[i], U[idx[i, j]]), +inf where idx[i, j] < 0."""
    if _FUSED:
        return _rs.pair_angles(np.ascontiguousarray(X, dtype=np.float64),
                               np.ascontiguousarray(U, dtype=np.float64),
                               np.ascontiguousarray(idx, dtype=np.int64))
    out = np.full(idx.shape, np.inf)
    for s in range(0, len(X), 2048):
        lab = idx[s:s + 2048]
        d = angle_between(X[s:s + 2048, None, :], U[np.maximum(lab, 0)])
        out[s:s + 2048] = np.where(lab >= 0, d, np.inf)
    return out


def chord_to_angle(chord) -> np.ndarray:
    """Angle subtended by a chord of the unit sphere."""
    return 2.0 * np.arcsin(np.minimum(1.0, 0.5 * np.asarray(chord, dtype=np.float64)))


def nearest_directions(X: np.ndarray, U: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """(inner products, indices) of each row of X's k nearest unit directions
    by angle, from an exact IndexFlatIP search in float32. Ranks only."""
    Xf = np.ascontiguousarray(X, dtype=np.float32)
    Uf = np.ascontiguousarray(U, dtype=np.float32)
    if len(Uf) and len(Xf) and tail_rust() and _rs.knn_path_py() != "faiss" and np.isfinite(Xf).all() and np.isfinite(Uf).all():
        # Exact top-k (Rust, construct_tail.rs): AMX > AVX-512 VNNI (int8-screened) > AVX2+FMA (fused
        # GEMM + running top-k); CAPCOVER_KNN=amx|vnni|avx2|faiss forces one. Same index sets as faiss.
        ip, labels = _rs.knn_ip(Xf, Uf, min(k, len(Uf)), max(1, faiss.omp_get_max_threads()))
        if k == 1:
            return ip.ravel(), labels.ravel()
        return ip, labels
    index = faiss.IndexFlatIP(U.shape[1])
    index.add(Uf)
    ip, labels = index.search(Xf, min(k, len(U)))
    if k == 1:
        return ip.ravel(), labels.ravel()
    return ip, labels


# Above this many directions knn_directions() searches cells instead of all
# pairs: one exact search over 80K directions is ~2.5e12 multiply-adds.
KNN_EXACT_MAX = 16384


def knn_directions(U: np.ndarray, m: int, exact_max: int = KNN_EXACT_MAX, probes: int = 12,
                   seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """(neighbors, dots): each unit row's m nearest *other* rows by angle
    (-1 padded) and the inner products with them, recomputed in float64.

    Up to `exact_max` rows the search is exact (IndexFlatIP). Above it, rows
    are split into ~sqrt(a) spherical k-means cells and each cell's rows are
    scored by one GEMM against the rows of the `probes` cells whose
    centroids are nearest its own -- ~probes / sqrt(a) of the exact work.
    Callers only rank candidates with these lists, never decide coverage."""
    U32 = np.ascontiguousarray(U, dtype=np.float32)
    a = len(U32)
    m = min(m, a - 1)
    if m <= 0:
        return np.full((a, 0), -1, dtype=np.int64), np.zeros((a, 0))
    if a <= exact_max:
        index = faiss.IndexFlatIP(U32.shape[1])
        index.add(U32)
        _, nbr = index.search(U32, m + 1)
    else:
        nbr = _cell_knn(U32, m + 1, probes, seed)
    nbr = np.where(nbr == np.arange(a)[:, None], -1, nbr)
    # Drop each row's self-match (not always column 0 under ties).
    keep = np.argsort(nbr < 0, axis=1, kind="stable")[:, :m]
    nbr = np.take_along_axis(nbr, keep, axis=1)
    return nbr, pair_dots(U, np.repeat(np.arange(a), m).reshape(a, m), nbr)


def _cell_knn(U32: np.ndarray, kk: int, probes: int, seed: int) -> np.ndarray:
    """IVF search done cell by cell: every row probes its own `probes`
    nearest centroids, each cell answers all rows probing it with one
    blocked GEMM, and each row keeps the best kk over its probes."""
    a, d = U32.shape
    cells = max(2, int(round(np.sqrt(a))))
    probes = min(probes, cells)
    km = faiss.Kmeans(d, cells, niter=10, seed=seed, spherical=True, max_points_per_centroid=64,
                      min_points_per_centroid=1)
    km.train(U32)
    _, probe = km.index.search(U32, probes)
    order, starts, counts = segments(probe[:, 0], cells)
    by_probe, p_starts, p_counts = segments(probe.ravel(), cells)
    found = np.full((a, probes * kk), -1, dtype=np.int64)
    score = np.full((a, probes * kk), -np.inf, dtype=np.float32)
    for c in range(cells):
        members = order[starts[c]:starts[c] + counts[c]]
        slots = by_probe[p_starts[c]:p_starts[c] + p_counts[c]]
        if len(members) == 0 or len(slots) == 0:
            continue
        rows, j = slots // probes, slots % probes
        index = faiss.IndexFlatIP(d)
        index.add(U32[members])
        width = min(kk, len(members))
        S, top = index.search(U32[rows], width)
        cols = j[:, None] * kk + np.arange(width)[None, :]
        found[rows[:, None], cols] = np.where(top >= 0, members[np.maximum(top, 0)], -1)
        score[rows[:, None], cols] = np.where(top >= 0, S, -np.inf)
    best = np.argsort(-score, axis=1, kind="stable")[:, :kk]
    return np.take_along_axis(found, best, axis=1)


def pair_dots(U: np.ndarray, rows: np.ndarray, cols: np.ndarray, block: int = 1 << 15) -> np.ndarray:
    """float64 <U[rows], U[cols]> elementwise; 0 where cols < 0."""
    rows, cols = np.asarray(rows), np.asarray(cols)
    out = np.zeros(cols.shape)
    r, c = rows.ravel(), cols.ravel()
    flat = out.ravel()
    for s in range(0, len(c), block):
        cc = c[s:s + block]
        ok = cc >= 0
        flat[s:s + block][ok] = np.einsum("ij,ij->i", U[r[s:s + block][ok]], U[cc[ok]])
    return flat.reshape(cols.shape)


def nearest_in_candidates(X: np.ndarray, U: np.ndarray, candidates: np.ndarray, labels: np.ndarray,
                          block: int = 1024) -> np.ndarray:
    """For each row x of X, the direction of largest inner product among
    candidates[labels[x]] (cap ids, -1 padded), ranked in float32. A cover
    whose caps know their nearby caps reassigns members this way in
    O(n * width * d) rather than O(n * caps * d)."""
    X32 = np.ascontiguousarray(X, dtype=np.float32)
    U32 = np.ascontiguousarray(U, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    out = labels.copy()
    for s in range(0, len(X32), block):
        C = candidates[labels[s:s + block]]
        ip = np.einsum("bd,bwd->bw", X32[s:s + block], U32[np.maximum(C, 0)])
        ip[C < 0] = -np.inf
        out[s:s + block] = C[np.arange(len(C)), np.argmax(ip, axis=1)]
    return out


def log10_cap_measure(psi: np.ndarray, dim: int) -> np.ndarray:
    """log10 of the share of the unit sphere S^{dim-1} a cap of angle psi
    covers, sigma(psi) = I_{sin^2 psi}((dim-1)/2, 1/2) / 2 for psi <= pi/2
    (1 - sigma(pi - psi) above). In log space throughout: at dim = 384 a cap
    of 0.15 rad is ~1e-316. I_x(a, b) = x^a (1-x)^b / (a B(a, b)) *
    2F1(a+b, 1; a+1; x), the series summed until its terms fall below
    1e-17 of the sum, for x <= 0.99; above, 1 - I_{1-x}(b, a), which no
    longer cancels there (sigma >= ~1e-2 at dim = 384)."""
    from scipy.special import betaln
    psi = np.clip(np.atleast_1d(np.asarray(psi, dtype=np.float64)), 1e-300, np.pi)
    a, b = 0.5 * (dim - 1), 0.5
    folded = np.minimum(psi, np.pi - psi)

    def log_I(x, p, q):
        term, total = np.ones_like(x), np.ones_like(x)
        for k in range(200000):
            term = term * x * ((p + q + k) / (p + 1.0 + k))
            total += term
            if k > 20 and np.all(term <= 1e-17 * total):
                break
        return p * np.log(x) + q * np.log1p(-x) - np.log(p) - betaln(p, q) + np.log(total)

    x = np.sin(folded) ** 2
    out = np.empty(len(x))
    small = x <= 0.99
    if small.any():
        out[small] = log_I(np.maximum(x[small], 1e-300), a, b)
    if (~small).any():
        rest = np.exp(log_I(np.maximum(1.0 - x[~small], 1e-300), b, a))
        out[~small] = np.log1p(-np.minimum(rest, 1.0))
    log_sigma = out - np.log(2.0)
    wide = psi > np.pi / 2
    log_sigma[wide] = np.log1p(-np.exp(log_sigma[wide]))
    return log_sigma / np.log(10.0)


# Tables of ln sigma(psi) by dimension, for log_cap_measure.
_MEASURE_TABLES = {}
_MEASURE_GRID = 1 << 16


def log_cap_measure(psi: np.ndarray, dim: float) -> np.ndarray:
    """Natural log of sigma(psi) on S^{dim-1} (log10_cap_measure's), fast:
    ln sigma(psi) - (dim - 1) ln psi, which is smooth down to psi = 0 (a
    small cap's measure grows as psi^(dim-1)), interpolated linearly in a
    cached table of 2^16 angles over (0, pi]. For ranking many caps at once
    (merge costs, encoding selection); within 1e-4 of exact at dim = 384
    (tests/test_cap_cover_filter.py). `dim` may be fractional: a cap
    measure in any number of dimensions."""
    key = float(dim)
    table = _MEASURE_TABLES.get(key)
    if table is None:
        grid = np.linspace(0.0, np.pi, _MEASURE_GRID + 1)[1:]
        smooth = log10_cap_measure(grid, key) * np.log(10.0) - (key - 1.0) * np.log(grid)
        table = _MEASURE_TABLES[key] = (grid, smooth)
    grid, smooth = table
    psi = np.clip(np.atleast_1d(np.asarray(psi, dtype=np.float64)), 1e-300, np.pi)
    return np.interp(psi, grid, smooth) + (key - 1.0) * np.log(psi)


def segments(labels: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(order, starts, counts) such that cluster c's rows are
    order[starts[c] : starts[c] + counts[c]]."""
    order = np.argsort(labels, kind="stable")
    counts = np.bincount(labels, minlength=k)
    starts = np.zeros(k, dtype=np.int64)
    np.cumsum(counts[:-1], out=starts[1:])
    return order, starts, counts


def segment_max(values_sorted: np.ndarray, starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Per-cluster max of values already in segments() order; 0 for an empty
    cluster."""
    out = np.zeros(len(starts), dtype=values_sorted.dtype)
    nonempty = counts > 0
    if nonempty.any():
        out[nonempty] = np.maximum.reduceat(values_sorted, starts[nonempty])
    return out


def exact_cap_cover(X: np.ndarray, U: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(directions, angles) making the unit directions U a cover of the unit
    rows X: every row joins its nearest direction and each angle is its
    farthest member's, measured exactly. Directions left with no member are
    dropped -- they cover nothing and would still cost a cap."""
    U, angles, _, _ = labelled_cap_cover(X, U)
    return U, angles


def labelled_cap_cover(X: np.ndarray, U: np.ndarray, labels: np.ndarray = None
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """exact_cap_cover with the assignment: (directions, angles, labels,
    kept), labels[i] the cap row i joined (its nearest direction unless
    `labels` gives the assignment) and kept[j] the input index of output
    cap j."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    U = unit_rows(U)
    if labels is None:
        _, labels = nearest_directions(X, U)
    labels = np.asarray(labels, dtype=np.int64)
    theta = row_angles(X, U, labels)
    order, starts, counts = segments(labels, len(U))
    angles = segment_max(theta[order], starts, counts)
    kept = np.flatnonzero(counts > 0)
    new_id = np.full(len(U), -1, dtype=np.int64)
    new_id[kept] = np.arange(len(kept))
    return np.ascontiguousarray(U[kept]), angles[kept], new_id[labels], kept


def remap_neighbors(neighbors: np.ndarray, kept: np.ndarray, n_before: int) -> np.ndarray:
    """Neighbor lists of the caps `kept` (input indices) re-indexed to their
    output positions, dropped caps removed (-1, moved to the row's end)."""
    new_id = np.full(n_before + 1, -1, dtype=np.int64)
    new_id[kept] = np.arange(len(kept))
    out = new_id[np.where(neighbors[kept] >= 0, neighbors[kept], n_before)]
    keep = np.argsort(out < 0, axis=1, kind="stable")
    return np.take_along_axis(out, keep, axis=1)


def enclosing_cap(ua: np.ndarray, ta: float, ub: np.ndarray, tb: float) -> Tuple[np.ndarray, float]:
    """Smallest cap holding caps (ua, ta) and (ub, tb): angle
    (phi + ta + tb) / 2 around the point on the great circle between them.
    The returned angle is re-measured from the direction actually computed,
    so rounding can only widen it."""
    phi = float(angle_between(ua[None, :], ub[None, :])[0])
    if phi + tb <= ta:
        return ua.copy(), ta
    if phi + ta <= tb:
        return ub.copy(), tb
    theta = 0.5 * (phi + ta + tb)
    if theta >= np.pi:
        return ua.copy(), np.pi
    if phi < 1e-12:
        u = ua.copy()
    else:
        f = (theta - ta) / phi
        u = (np.sin((1.0 - f) * phi) * ua + np.sin(f * phi) * ub) / np.sin(phi)
        u /= np.linalg.norm(u)
    theta = max(theta, float(angle_between(u[None, :], ua[None, :])[0]) + ta,
                float(angle_between(u[None, :], ub[None, :])[0]) + tb)
    return u, min(theta, np.pi)
