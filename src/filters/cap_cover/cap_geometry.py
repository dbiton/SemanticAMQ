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


def chord_to_angle(chord) -> np.ndarray:
    """Angle subtended by a chord of the unit sphere."""
    return 2.0 * np.arcsin(np.minimum(1.0, 0.5 * np.asarray(chord, dtype=np.float64)))


def nearest_directions(X: np.ndarray, U: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """(inner products, indices) of each row of X's k nearest unit directions
    by angle, from an exact IndexFlatIP search in float32. Ranks only."""
    index = faiss.IndexFlatIP(U.shape[1])
    index.add(np.ascontiguousarray(U, dtype=np.float32))
    ip, labels = index.search(np.ascontiguousarray(X, dtype=np.float32), min(k, len(U)))
    if k == 1:
        return ip.ravel(), labels.ravel()
    return ip, labels


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
    X = np.ascontiguousarray(X, dtype=np.float64)
    U = unit_rows(U)
    _, labels = nearest_directions(X, U)
    theta = angle_between(X, U[labels])
    order, starts, counts = segments(labels, len(U))
    angles = segment_max(theta[order], starts, counts)
    keep = counts > 0
    return np.ascontiguousarray(U[keep]), angles[keep]


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
