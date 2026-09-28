import numpy as np
import faiss
from typing import Tuple


# Shared by the construct methods that choose centers some other way than
# farthest-first (MergeConstruct, KMeansConstruct, MinimaxConstruct).
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
