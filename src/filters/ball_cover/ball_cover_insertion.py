from __future__ import annotations

import faiss
import numpy as np


def _exhaustive_search(centers: np.ndarray, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact distance from every query to every center, via a throwaway
    faiss index rather than a raw (m, k, d) numpy broadcast -- the latter
    materializes a full dense tensor (a real problem at scale: e.g.
    20000 x 5000 x 384 floats is ~150GB) where this is O(m*k*d) compute
    with no more than a (m, k) result ever held in memory."""
    dim = centers.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(centers)
    d2, labels = index.search(queries, len(centers))
    return np.sqrt(np.maximum(d2, 0.0)), labels


def insert_nearest(
    centers: np.ndarray,
    radii: np.ndarray,
    new_vectors: np.ndarray,
    max_count_balls: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Absorb each new vector into the nearest existing ball by raw
    distance, growing that ball's radius if needed. O(m*k*d), no new balls
    ever created -- max_count_balls is accepted but unused (there's nothing
    to cap). Cheap and predictable, but nearest != cheapest to grow: a
    far-but-huge ball may need no growth at all while the nearest one is a
    tight singleton about to blow up by a factor of dist**d (see
    insert_least_inflation)."""
    X = np.ascontiguousarray(new_vectors, dtype=np.float32)
    dist, labels = _exhaustive_search(centers, X)
    surface = dist - radii[labels]

    cost = np.where(surface <= 0, -np.inf, dist)
    col = cost.argmin(axis=1)
    row = np.arange(len(X))
    ball = labels[row, col]

    new_radii = radii.copy()
    np.maximum.at(new_radii, ball, dist[row, col])
    return centers, new_radii


def insert_least_inflation(
    centers: np.ndarray,
    radii: np.ndarray,
    new_vectors: np.ndarray,
    max_count_balls: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Same O(m*k*d) exhaustive search as insert_nearest, but absorb into
    whichever ball minimizes the actual objective delta
    max(r_i, dist_i)**d - r_i**d, not whichever is nearest. Containment
    (delta=0) is preferred for free, and a far-but-huge ball can beat a
    near-but-tiny one whenever growing the tiny one would cost more volume
    -- strictly better covering volume than insert_nearest at identical
    asymptotics and no new memory. max_count_balls is accepted but unused,
    same reason as insert_nearest.

    Computed in log space, scaled by each point's own largest candidate
    radius: r**d alone overflows float64 well inside float32's range at
    d ~ 384, but the scaled terms are always in (0, 1] so they safely
    underflow to 0 instead of overflowing, and dividing every candidate by
    the same positive per-point scale preserves the argmin exactly."""
    dim = centers.shape[1]
    X = np.ascontiguousarray(new_vectors, dtype=np.float32)
    dist, labels = _exhaustive_search(centers, X)
    dist = dist.astype(np.float64)
    r = radii[labels].astype(np.float64)
    r_new = np.maximum(r, dist)

    M = r_new.max(axis=1, keepdims=True)
    safe_M = np.where(M > 0, M, 1.0)
    with np.errstate(divide="ignore"):
        scaled_new = np.exp(dim * np.log(np.maximum(r_new, 0.0) / safe_M))
        scaled_old = np.exp(dim * np.log(np.maximum(r, 0.0) / safe_M))
    delta = np.where(M > 0, scaled_new - scaled_old, 0.0)

    col = delta.argmin(axis=1)
    row = np.arange(len(X))
    ball = labels[row, col]

    new_radii = radii.copy()
    np.maximum.at(new_radii, ball, dist[row, col].astype(np.float32))
    return centers, new_radii


def _merge_two_balls(c1: np.ndarray, r1: float, c2: np.ndarray, r2: float) -> tuple:
    """Smallest enclosing ball of two balls: whichever contains the other
    if one does, else centered on the line between them with radius
    (dist + r1 + r2) / 2."""
    diff = c2 - c1
    dist = float(np.sqrt(max(float(np.dot(diff, diff)), 0.0)))
    if dist + min(r1, r2) <= max(r1, r2):
        return (c1, r1) if r1 >= r2 else (c2, r2)
    new_r = 0.5 * (dist + r1 + r2)
    new_c = c1 if dist < 1e-12 else c1 + ((new_r - r1) / dist) * diff
    return new_c, new_r


def _cheapest_pair(centers: np.ndarray, radii: np.ndarray, dim: int) -> tuple:
    """argmin over all pairs (i, j) of the merge objective delta
    r_merged**d - r_i**d - r_j**d, log-space scaled the same way as
    insert_least_inflation (shared M = max radius among these centers)."""
    n = len(centers)
    ci, cj = centers[:, None, :], centers[None, :, :]
    ri, rj = radii[:, None], radii[None, :]
    diff = ci - cj
    dist = np.sqrt(np.maximum(np.sum(diff * diff, axis=-1), 0.0))
    contains = dist + np.minimum(ri, rj) <= np.maximum(ri, rj)
    merged = 0.5 * (dist + ri + rj)
    r_merged = np.where(contains, np.maximum(ri, rj), merged)

    # M must cover r_merged too, not just the existing radii -- a merged
    # ball can be larger than every current ball, and scaling by a smaller
    # M would let (r_merged/M)**d exceed 1 and overflow exp().
    M = float(max(radii.max(), r_merged.max())) if n else 0.0
    safe_M = M if M > 0 else 1.0
    with np.errstate(divide="ignore"):
        scaled_i = np.exp(dim * np.log(np.maximum(ri, 0.0) / safe_M))
        scaled_j = np.exp(dim * np.log(np.maximum(rj, 0.0) / safe_M))
        scaled_m = np.exp(dim * np.log(np.maximum(r_merged, 0.0) / safe_M))
    delta = (scaled_m - scaled_i - scaled_j) if M > 0 else np.zeros((n, n))

    idx = np.arange(n)
    delta[idx, idx] = np.inf  # exclude self-pairs
    i, j = np.unravel_index(np.argmin(delta), delta.shape)
    return int(i), int(j)


def insert_merge(
    centers: np.ndarray,
    radii: np.ndarray,
    new_vectors: np.ndarray,
    max_count_balls: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Insert each new vector as its own zero-radius ball; once at capacity,
    evict by merging whichever pair of balls -- existing-existing or the
    new singleton against an existing ball -- grows the covering objective
    least. The choice is global: a point landing in genuinely new territory
    gets to found a ball rather than deforming one, and redundancy gets
    given up wherever it's cheapest, not wherever the new point happened to
    land.

    Sequential per point (each merge changes the ball set the next point's
    decision depends on, unlike insert_nearest/insert_least_inflation,
    which can absorb a whole batch in one vectorized pass). Recomputes the
    full k x k merge-cost matrix on every insertion at capacity --
    O(k**2*d) per point, correct but not the O(k*d)-amortized version an
    incremental row/column-cached heap would give.

    Runs the whole batch in float64 numpy with no intermediate
    re-quantization -- the caller quantizes once, from the final
    (new_centers, new_radii), rather than re-quantizing (and reading centers
    back out of the quantized index, compounding quantization error) after
    every single point."""
    dim = centers.shape[1]
    c = np.ascontiguousarray(centers, dtype=np.float64)
    r = np.asarray(radii, dtype=np.float64)
    X = np.ascontiguousarray(new_vectors, dtype=np.float32).astype(np.float64)

    for x in X:
        c = np.vstack([c, x[None, :]])
        r = np.append(r, 0.0)

        if len(c) > max_count_balls:
            i, j = _cheapest_pair(c, r, dim)
            c_new, r_new = _merge_two_balls(c[i], r[i], c[j], r[j])
            keep = [idx for idx in range(len(c)) if idx not in (i, j)]
            c = np.vstack([c[keep], c_new[None, :]])
            r = np.append(r[keep], r_new)

    return c.astype(np.float32), r.astype(np.float32)


# Uniform dispatch table -- every value has the exact same call signature,
# so a caller can select a strategy by name without special-casing any of
# them (see module docstring).
INSERTION_STRATEGIES = {
    "nearest": insert_nearest,
    "least_inflation": insert_least_inflation,
    "merge": insert_merge,
}
