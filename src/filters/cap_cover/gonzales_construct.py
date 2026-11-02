import os
import numpy as np
from typing import Optional, Tuple
from filters.filter import Filter
from filters.cap_cover.cap_interfaces import ConstructInterface
from filters.cap_cover.cap_geometry import exact_cap_cover, labelled_cap_cover

try:
    import capcover_rs as _rs
except Exception:   # extension not built
    _rs = None

_rust_gonzalez = getattr(_rs, "gonzalez_chosen", None)                       # int8 / fused kernel
_rust_gonzalez_labels = getattr(_rs, "gonzalez_chosen_labels", None)
_rust_lazy = getattr(_rs, "gonzalez_chosen_pruned", None)                    # lazy exact kernel
_rust_lazy_labels = getattr(_rs, "gonzalez_chosen_pruned_labels", None)
_screening_available = getattr(_rs, "gonzalez_screening_available", lambda: False)

KERNELS = ("int8", "lazy", "numpy")


def resolve_kernel(kernel: Optional[str] = None) -> str:
    """Which Gonzalez selection kernel to use: 'int8' (fused AVX512 kernel with int8 screening),
    'lazy' (exact lazy/CELF kernel, AVX2 with scalar fallback) or 'numpy' (reference).
    Explicit argument > env CAPCOVER_GONZALEZ_KERNEL (int8|lazy|numpy|auto) > auto:
    int8 when the CPU has AVX512 VNNI+VL+BW and CAPCOVER_GONZALEZ_NOQ is unset, else lazy,
    else numpy when the extension lacks the kernels."""
    k = kernel if kernel is not None else os.environ.get("CAPCOVER_GONZALEZ_KERNEL", "auto")
    k = k.lower()
    if k == "auto":
        if _rust_gonzalez_labels is not None and _screening_available():
            return "int8"
        return "lazy" if _rust_lazy_labels is not None else "numpy"
    if k not in KERNELS:
        raise ValueError(f"unknown Gonzalez kernel {k!r}; expected one of {KERNELS} or 'auto'")
    if k == "int8" and _rust_gonzalez_labels is None:
        raise RuntimeError("capcover_rs.gonzalez_chosen_labels is not available; build CapCoverRustFilter")
    if k == "lazy" and _rust_lazy_labels is None:
        raise RuntimeError("capcover_rs.gonzalez_chosen_pruned_labels is not available; build CapCoverRustFilter")
    return k


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
    direction), then exact cap angles. O(n*k*d) with the NumPy reference.

    Kernels (see resolve_kernel; env CAPCOVER_GONZALEZ_KERNEL=int8|lazy|numpy):
      int8  fused Rust pass (dot + max-update + next argmin), int8 VNNI screening that skips
            rows which provably cannot change their best direction; needs AVX512 VNNI+VL.
      lazy  exact lazy ("CELF") Rust kernel: stale best_ip lower bounds, only candidates for
            the argmin are brought up to date; AVX2/FMA tiles, scalar fallback. Works on any CPU.
      numpy reference loop (fast=False).
    The default picks int8 when the CPU supports it (and CAPCOVER_GONZALEZ_NOQ is unset), else lazy.
    The index sequence equals the reference's except where float32 summation order (BLAS vs SIMD
    lanes) flips a comparison between inner products within a few ulp (~1e-7); ties between
    bit-identical rows resolve to the lowest index in all paths. Both Rust kernels also return each
    row's best chosen direction (float32 argmax), so the n*k*d nearest-direction search of
    exact_cap_cover is skipped; a row whose two best directions are within float32 noise may join
    the other one than faiss would pick. Cap angles always come from float64 re-measurement
    (labelled_cap_cover / exact_cap_cover), so coverage is exact either way.

    fast=False forces 'numpy'; `kernel=` overrides fast and the environment."""

    def __init__(self, fast: Optional[bool] = None, kernel: Optional[str] = None):
        if kernel is None and fast is False:
            kernel = "numpy"
        self._kernel = kernel
        self.fast = self.kernel != "numpy"
        if fast and not self.fast:
            raise RuntimeError("capcover_rs Gonzalez kernels are not available; build CapCoverRustFilter or use fast=False")

    @property
    def kernel(self) -> str:
        return resolve_kernel(self._kernel)

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
        kern = self.kernel
        if kern != "numpy":
            fn = _rust_gonzalez_labels if kern == "int8" else _rust_lazy_labels
            chosen, labels = fn(X32, k)
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
