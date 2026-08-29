import numpy as np
import faiss
from typing import Tuple
from filters.filter import Filter
from filters.ball_cover.ball_interfaces import ConstructInterface


class GonzalesConstruct(ConstructInterface):
    """Gonzalez farthest-first, then exact radii. O(n*k*d), single pass for
    center selection plus one exact search for radii."""

    def construct_filter(self, filter: Filter, vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = np.ascontiguousarray(vectors, dtype=np.float32)
        n = X.shape[0]
        dim = X.shape[1]
        k = min(filter.max_construct_spheres, n)
        if k <= 0:
            return np.empty((0, dim), dtype=np.float32), np.empty(0, dtype=np.float32)
        if n <= k:
            return X.copy(), np.zeros(n, dtype=np.float32)

        # ||x||^2, one pass, no temporaries
        sq = np.einsum("ij,ij->i", X, X)

        chosen = np.empty(k, dtype=np.int64)
        min_d2 = np.empty(n, dtype=np.float32)
        d2 = np.empty(n, dtype=np.float32)

        # d2(x, c) = ||x||^2 - 2 <x, c> + ||c||^2  -> one sgemv per center.
        # Approximate (float32 cancellation for near-duplicate x, c), but that's
        # fine here: min_d2 only drives *which* point to pick next, not any
        # radius that ends up returned -- see the exact re-search below.
        chosen[0] = 0
        np.dot(X, X[0], out=min_d2)
        min_d2 *= -2.0
        min_d2 += sq
        min_d2 += sq[0]

        for j in range(1, k):
            c = int(min_d2.argmax())
            if min_d2[c] <= 0.0:  # all points already covered exactly
                chosen = chosen[:j]
                k = j
                break
            chosen[j] = c

            np.dot(X, X[c], out=d2)
            d2 *= -2.0
            d2 += sq
            d2 += sq[c]
            np.minimum(min_d2, d2, out=min_d2)

        centers = np.ascontiguousarray(X[chosen])

        # Exact final assignment + radii via a real nearest-center search, not
        # the approximate min_d2/labels tracked during selection above: reusing
        # that running minimum directly as the squared radius let the same
        # float32 cancellation error that only needed to be "good enough" to
        # pick reasonable centers actually *understate* a returned radius by a
        # tiny margin -- rare, but a real, verified violation of the
        # zero-false-negative covering invariant every construction strategy
        # promises (caught by test_ball_cover_construction.py on clustered
        # data). The exact search costs one more O(n*k*d) pass, same order as
        # selection, and is what rptree/ivf's own radius computations already do.
        index = faiss.IndexFlatL2(dim)
        index.add(centers)
        d2_exact, labels = index.search(X, 1)
        d2_exact = np.maximum(d2_exact.ravel(), 0.0)
        labels = labels.ravel()

        # Per-cluster max via segmented reduce (np.maximum.at is much slower).
        order = np.argsort(labels, kind="stable")
        counts = np.bincount(labels, minlength=k)
        starts = np.zeros(k, dtype=np.int64)
        np.cumsum(counts[:-1], out=starts[1:])
        nonempty = counts > 0

        radii = np.zeros(k, dtype=np.float32)
        radii[nonempty] = np.maximum.reduceat(d2_exact[order], starts[nonempty])
        np.sqrt(radii, out=radii)
        return centers, radii


    def get_bytesize(self) -> int:
        return 0