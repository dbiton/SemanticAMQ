import numpy as np
from typing import Optional, Tuple
from filters.filter import Filter
from filters.cap_cover.cap_interfaces import ConstructInterface
from filters.cap_cover.cap_geometry import exact_cap_cover, labelled_cap_cover

try:
    from capcover_rs import gonzalez_chosen as _rust_gonzalez
    from capcover_rs import gonzalez_chosen_labels as _rust_gonzalez_labels
except Exception:   # extension not built (or built without the kernel)
    _rust_gonzalez = None
    _rust_gonzalez_labels = None


def gonzalez_chosen_numpy(X32: np.ndarray, k: int) -> np.ndarray:
    """Reference farthest-first sequence (float32 BLAS), k >= 1, n > k."""
    n = len(X32)
    chosen = np.empty(k, dtype=np.int64)
    chosen[0] = 0
    best_ip = X32 @ X32[0]
    ip = np.empty(n, dtype=np.float32)
    for j in range(1, k):
        c = int(best_ip.argmin())
        if best_ip[c] >= 1.0:   # every point already a chosen direction
            return chosen[:j]
        chosen[j] = c
        np.dot(X32, X32[c], out=ip)
        np.maximum(best_ip, ip, out=best_ip)
    return chosen


class GonzalesConstruct(ConstructInterface):
    """Gonzalez farthest-first by angle (on unit vectors, the farthest point
    by angle is the one of least inner product with its nearest chosen
    direction), then exact cap angles. O(n*k*d).

    fast=True (default when capcover_rs exposes gonzalez_chosen) runs the
    selection in a fused Rust kernel (dot + max-update + next argmin in one
    pass over X, GIL released). The index sequence equals the NumPy path's
    except where float32 summation order (BLAS vs the kernel's SIMD lanes)
    flips a comparison between inner products within a few ulp (~1e-7); ties
    between bit-identical rows resolve to the lowest index in both. Cap angles
    always come from the float64 exact_cap_cover, so coverage is exact
    either way. fast=False forces the NumPy reference. In fast mode the kernel
    also returns each row's best chosen direction, so the n*k*d nearest-direction
    search of exact_cap_cover is skipped; a row whose two best directions are within
    float32 noise may join the other one than faiss would pick (angles stay exact)."""

    def __init__(self, fast: Optional[bool] = None):
        if fast is None:
            fast = _rust_gonzalez is not None
        if fast and _rust_gonzalez is None:
            raise RuntimeError("capcover_rs.gonzalez_chosen is not available; build CapCoverRustFilter or use fast=False")
        self.fast = fast

    def construct_filter(self, filter: Filter, vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = np.ascontiguousarray(vectors, dtype=np.float64)
        n, dim = X.shape
        k = min(filter.max_construct_caps, n)
        if k <= 0:
            return np.empty((0, dim)), np.empty(0)
        if n <= k:
            return X.copy(), np.zeros(n)

        # float32 view of the input: no extra copy when it already is float32
        # (float64 -> float32 of float32 data is lossless, so identical to X.astype).
        X32 = np.ascontiguousarray(vectors, dtype=np.float32)
        if self.fast:
            chosen, labels = _rust_gonzalez_labels(X32, k)
            chosen = np.asarray(chosen, dtype=np.int64)
            # The kernel already knows each row's best chosen direction (float32 argmax), so
            # skip exact_cap_cover's n*k*d nearest-direction search. The labels only pick which
            # cap a row joins; every angle below is still the float64 angle to that cap's
            # direction, so coverage is exact for any labelling.
            U, ang, _, _ = labelled_cap_cover(X, X[chosen], np.asarray(labels, dtype=np.int64))
            return U, ang
        chosen = gonzalez_chosen_numpy(X32, k)
        # Angles from the exact float64 re-measurement, never from best_ip.
        return exact_cap_cover(X, X[chosen])

    def get_bytesize(self) -> int:
        return 0
