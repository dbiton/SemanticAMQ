import numpy as np
import faiss
from typing import Tuple


# Shared by the construct methods that choose centers some other way than
# farthest-first (MergeConstruct, MinimaxConstruct, SplitConstruct,
# SphereMergeConstruct).
# exact_cover() is GonzalesConstruct's radius step unchanged -- an exact
# IndexFlatL2 re-search plus a per-cluster max -- because that step, not the
# center selection, is what the zero-false-negative guarantee rests on. See
# the comment in gonzales_construct.py for why a radius must never come from
# distances tracked while choosing centers.


def nearest_centers(X: np.ndarray, centers: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(squared distance, index) of each row of X's nearest center, from an
    exact IndexFlatL2 search."""
    index = faiss.IndexFlatL2(X.shape[1])
    index.add(np.ascontiguousarray(centers, dtype=np.float32))
    d2, labels = index.search(X, 1)
    return np.maximum(d2.ravel(), 0.0), labels.ravel()


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
    cluster. Segmented reduce, since np.maximum.at is much slower."""
    out = np.zeros(len(starts), dtype=values_sorted.dtype)
    nonempty = counts > 0
    if nonempty.any():
        out[nonempty] = np.maximum.reduceat(values_sorted, starts[nonempty])
    return out


def sphere_radius(X: np.ndarray, rtol: float = 1e-3) -> float:
    """The common norm of X's rows if they all lie on one sphere (relative
    spread <= rtol, which float32 normalization clears by four orders of
    magnitude), else 0.0."""
    if len(X) == 0:
        return 0.0
    norms = np.sqrt(np.einsum("ij,ij->i", X, X, dtype=np.float64))
    rho = float(norms.mean())
    return rho if rho > 0.0 and float(norms.max() - norms.min()) <= rtol * rho else 0.0


def sphere_centers(X: np.ndarray, centers: np.ndarray, rho: float, max_norm: float = 5.0) -> np.ndarray:
    """Centers moved along their own directions onto one common sphere, for
    data X lying on the sphere of radius rho.

    A ball only matters through the cap it cuts from the data's sphere when
    every query lies on that sphere too, as unit-normalized embeddings do. For
    a member cap of angle theta around the center's direction, the Euclidean
    MEB center sits inside the sphere at norm cos(theta), and its
    tau-expanded ball captures a cap ~tau*tan(theta) wider than the members'.
    A center at norm s holding the same cap needs R^2 = 1 + s^2 - 2 s cos(theta)
    (in units of rho) and its tau-expansion grows the cap by ~tau*R/s: tan at
    s = cos(theta), chord at s = 1, and least, sin(theta), at s = 1/cos(theta).

    All centers get one norm -- the median over balls of 1/cos(theta),
    clipped at max_norm for caps near a hemisphere -- not each its own: with
    unequal norms a vector's nearest center stops being the one whose cap
    holds it, which exact_cover's reassignment and
    BallCoverFilter._tight_radii both rely on. A common norm keeps Euclidean
    and angular nearest the same thing. Cover volume (sum of r^dim) gets
    worse under this move while the false-positive rate improves: radii grow,
    caps on the sphere shrink, and the sphere is where queries live.

    Only centers move; radii still come from exact_cover afterwards."""
    C = np.asarray(centers, dtype=np.float64)
    U = C / np.maximum(np.linalg.norm(C, axis=1), 1e-12)[:, None]
    unit = np.ascontiguousarray(rho * U, dtype=np.float32)
    # On a common norm, nearest center is nearest by angle.
    _, labels = nearest_centers(X, unit)
    cos = np.einsum("ij,ij->i", X, unit[labels], dtype=np.float64) / (rho * rho)
    order, starts, counts = segments(labels, len(unit))
    nonempty = counts > 0
    cos_cap = np.minimum.reduceat(cos[order], starts[nonempty])
    s = float(np.median(1.0 / np.clip(cos_cap, 1.0 / max_norm, 1.0)))
    return np.ascontiguousarray(rho * s * U, dtype=np.float32)


def exact_cover(X: np.ndarray, centers: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Radii that make `centers` a cover of X: every row joins its nearest
    center and each radius is its farthest member's distance.

    Unlike Gonzalez, whose centers are data points and so never end up
    empty, a centroid can be left with no members. Such centers are dropped:
    they cover nothing, and a zero-radius ball would still cost a ball's
    memory and flag every query within tau of it."""
    X = np.ascontiguousarray(X, dtype=np.float32)
    centers = np.ascontiguousarray(centers, dtype=np.float32)
    d2, labels = nearest_centers(X, centers)
    order, starts, counts = segments(labels, len(centers))
    r2 = segment_max(d2[order], starts, counts)
    keep = counts > 0
    radii = np.sqrt(r2[keep]).astype(np.float32)
    return np.ascontiguousarray(centers[keep]), radii
