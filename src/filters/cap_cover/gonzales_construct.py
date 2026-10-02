import numpy as np
from typing import Tuple
from filters.filter import Filter
from filters.cap_cover.cap_interfaces import ConstructInterface
from filters.cap_cover.cap_geometry import exact_cap_cover


class GonzalesConstruct(ConstructInterface):
    """Gonzalez farthest-first by angle (on unit vectors, the farthest point
    by angle is the one of least inner product with its nearest chosen
    direction), then exact cap angles. O(n*k*d)."""

    def construct_filter(self, filter: Filter, vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = np.ascontiguousarray(vectors, dtype=np.float64)
        n, dim = X.shape
        k = min(filter.max_construct_caps, n)
        if k <= 0:
            return np.empty((0, dim)), np.empty(0)
        if n <= k:
            return X.copy(), np.zeros(n)

        X32 = X.astype(np.float32)
        chosen = np.empty(k, dtype=np.int64)
        chosen[0] = 0
        best_ip = X32 @ X32[0]
        ip = np.empty(n, dtype=np.float32)
        for j in range(1, k):
            c = int(best_ip.argmin())
            if best_ip[c] >= 1.0:   # every point already a chosen direction
                chosen = chosen[:j]
                break
            chosen[j] = c
            np.dot(X32, X32[c], out=ip)
            np.maximum(best_ip, ip, out=best_ip)
        # Angles from the exact float64 re-measurement, never from best_ip.
        return exact_cap_cover(X, X[chosen])

    def get_bytesize(self) -> int:
        return 0
