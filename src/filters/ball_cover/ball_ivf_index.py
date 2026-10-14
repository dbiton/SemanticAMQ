"""
BallIVFIndex -- exact IVF index for BallCoverFilter (zero false negatives).

Accept test
-----------
    accept(q)  <=>  exists ball j :  ||q - c_j||  <=  r_j + tau + eps

c_j is ball j's decoded center, r_j its radius, tau the distance threshold and
eps BallCoverFilter's coverage margin.  A negative reach admits nothing.

Index
-----
1. IVF cells: K = sqrt(N) k-means cells, visited nearest-first.
   Cell c is skipped only if ||q - mu_c|| > U_c + tau + eps, where
   U_c = max_j (r_j + ||c_j - mu_c||) over its balls.
2. Early abandoning: centers are stored in a centred-PCA basis; a ball is
   dropped once a lower bound on ||q - c_j||^2 (partial sum over the leading
   coordinates + reverse triangle inequality on the rest) exceeds its reach.
   ~98% of balls are dropped within the first 64 of 384 coordinates.
3. Exact decision: survivors are tested in full; pairs within float32 rounding
   of the boundary are re-measured in float64 by BallCoverFilter's own rule.
Large batches compute step 2's first bound for all pairs as one BLAS GEMM.
Compiled kernels: ball_ivf_kernels.py.

Correctness
-----------
Steps 1-2 discard a ball only via a valid lower bound (triangle inequality),
widened by its float rounding error, so no ball that accepts q is discarded.
Step 3 leans towards accept by _TINY, so it never rejects a pair the scan
accepts.

Threshold
---------
Dynamic: tau is passed at query time; each new tau recomputes the per-ball
reach (r_j + tau + eps)^2 in O(N), nothing else depends on it.

Cost
----
Build: k-means + PCA, O(N d (K + d)).  Memory: a float32 copy of the centers
plus their 64-coordinate PCA head, outside the filter's budget (get_bytesize()).
"""

import numpy as np
from filters.ball_cover import ball_ivf_kernels as IVF_K


# ---------------------------------------------------------------------------
# Euclidean k-means (build time only)
# ---------------------------------------------------------------------------

def _euclidean_kmeans(X: np.ndarray, K: int, n_iter: int = 20, seed: int = 0):
    """(centroids float32 [K,d], labels int32 [N]); strictly L2."""
    rng = np.random.default_rng(seed)
    N, d = X.shape
    K = min(K, N)
    if K == 0:
        return np.empty((0, d), np.float32), np.empty(0, np.int32)

    # k-means++ seeding keeps cells balanced
    centroids = np.empty((K, d), np.float32)
    first = int(rng.integers(N))
    centroids[0] = X[first]
    xn = np.einsum("ij,ij->i", X, X)
    best = np.maximum(xn + xn[first] - 2.0 * (X @ X[first]), 0.0)
    for k in range(1, K):
        p = best / best.sum() if best.sum() > 0 else None
        nxt = int(rng.choice(N, p=p)) if p is not None else int(rng.integers(N))
        centroids[k] = X[nxt]
        best = np.minimum(best, np.maximum(xn + xn[nxt] - 2.0 * (X @ X[nxt]), 0.0))

    labels = np.full(N, -1, np.int32)
    for it in range(n_iter):
        cn = np.einsum("ij,ij->i", centroids, centroids)
        new_labels = (2.0 * (X @ centroids.T) - cn[None, :]).argmax(axis=1).astype(np.int32)

        # re-seat empty cells to random data points
        counts = np.bincount(new_labels, minlength=K)
        empty = np.flatnonzero(counts == 0)
        if len(empty):
            # re-seat on the points farthest from their centroid
            dist = xn - 2.0 * np.einsum("ij,ij->i", X, centroids[new_labels]) + cn[new_labels]
            far = np.argsort(-dist)[:len(empty)]
            new_labels[far] = empty
            counts = np.bincount(new_labels, minlength=K)

        order = np.argsort(new_labels, kind="stable")
        starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
        sums = np.add.reduceat(X[order].astype(np.float64), starts, axis=0)
        centroids = (sums / counts[:, None]).astype(np.float32)

        if (new_labels == labels).all():
            break
        labels = new_labels
    return centroids, labels


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

class BallIVFIndex:
    """
    Exact IVF index for BallCoverFilter.  Zero false negatives.

    Parameters
    ----------
    n_cells : int
        Upper bound on the number of cells; K = min(n_cells, sqrt(N)).
    n_iter, seed : k-means iterations / RNG seed.
    gemm_min_batch : int
        Batches with at least this many queries take the large-batch path:
        the first abandon test for every (query, member) pair as one BLAS GEMM
        over the leading PCA coordinates (see ivf_kernels.KernelData.run_gemm).
    parallel_min_batch : int
        Batches with at least this many queries run multi-threaded.
    """

    # The exact test accepts d^2 <= reach^2 + _TINY (|q|^2 + |c|^2): float64
    # rounding plus the ~1e-6 by which BallCoverFilter._scan's rotated float32
    # distances may differ from the input-basis ones here -- towards accept, so
    # the index never rejects a query the scan accepts; it can only accept a pair
    # within ~1e-5 of the boundary, far inside the filter's 1e-3 coverage margin.
    _TINY = 2e-6

    def __init__(self, n_cells: int = 4096, n_iter: int = 12, seed: int = 0,
                 parallel_min_batch: int = 8, gemm_min_batch: int = 16):
        self.n_cells = n_cells
        self.n_iter = n_iter
        self.seed = seed
        self.parallel_min_batch = int(parallel_min_batch)
        self.gemm_min_batch = int(gemm_min_batch)
        self._built = False
        self._tau_key = None

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, centers: np.ndarray, radii: np.ndarray) -> None:
        """centers: [N,d] decoded centers (in the basis queries will be passed
        in); radii: [N]."""
        centers = np.ascontiguousarray(centers, dtype=np.float32)
        radii = np.asarray(radii, dtype=np.float32).ravel()
        N, d = centers.shape
        self.N, self.d = N, d
        self._tau_key = None
        self.K = 0
        if N == 0:
            self._built = True
            return

        # --- cells: K = sqrt(N) k-means ---
        K = max(1, min(int(self.n_cells), int(np.sqrt(N))))
        mu, labels = _euclidean_kmeans(centers, K, self.n_iter, self.seed)
        used = np.unique(labels)
        remap = np.full(len(mu), -1, np.int64)
        remap[used] = np.arange(len(used))
        labels = remap[labels]
        mu64 = mu[used].astype(np.float64)
        K = len(mu64)

        # per-ball distance to its own centroid (float64, rounded up); cell reach U_c
        c64 = centers.astype(np.float64)
        delta = np.linalg.norm(c64 - mu64[labels], axis=1)
        delta = delta * (1.0 + 1e-9) + 1e-9
        r64 = radii.astype(np.float64)
        cell_U = np.full(K, -np.inf)
        np.maximum.at(cell_U, labels, r64 + delta)

        # storage: by cell, widest ball first within a cell
        order = np.lexsort((-radii, labels))
        self._order = order.astype(np.int64)
        counts = np.bincount(labels, minlength=K)
        self._cell_starts = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
        self._cell_sizes = counts.astype(np.float64)
        self._cell_U = cell_U
        self._r64 = r64[order]
        self.K = K

        n = self.N
        self._kd = IVF_K.KernelData(c64[order], centers[order], np.full(n, -1.0), np.zeros(n), np.ones(n),
                                 mu64, cell_U, self._cell_starts, dense=3, gemm_ck=3)
        self._built = True

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def _prepare(self, tau_eps: float) -> None:
        """Per-tau reach^2 of every ball (-1 for a negative reach: admits nothing)."""
        if self._tau_key == tau_eps:
            return
        reach = self._r64 + tau_eps
        with np.errstate(over="ignore", invalid="ignore"):
            rsq = np.where(reach >= 0.0, reach * reach, -1.0)
        self._kd.set_reach(rsq)
        self._tau_key = tau_eps

    def query(self, Q: np.ndarray, tau: float, eps: float) -> np.ndarray:
        """bool[nq]: does some ball accept the query?  Exact, 0-FNR.
        Q: queries [nq,d] in the centers' basis (float64; float32 is widened)."""
        nq = len(Q)
        if not self._built or self.N == 0 or nq == 0:
            return np.zeros(nq, dtype=bool)
        tau_eps = float(tau) + float(eps)
        self._prepare(tau_eps)
        Q64 = np.ascontiguousarray(Q, dtype=np.float64)
        live = np.ones(nq, dtype=np.bool_)
        parallel = nq >= self.parallel_min_batch and IVF_K.get_num_threads() > 1
        if nq >= self.gemm_min_batch:
            return self._kd.run_gemm(live, Q64, np.zeros(nq), self._TINY, 0.0, parallel)
        return self._kd.run(live, Q64, np.zeros(nq), tau_eps, self._TINY, 0.0, parallel)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def pruning_fraction(self, Q: np.ndarray, tau: float = 0.5, eps: float = 1e-3,
                         stage: str = "full") -> np.ndarray:
        """Per query, the fraction of balls still alive after the IVF cell gate
        (stage="gate") or after the gate and every early-abandon checkpoint,
        i.e. measured in full (stage="full", the default).  0.0 = everything
        pruned, 1.0 = nothing."""
        if not self._built or self.N == 0:
            return np.ones(len(Q))
        self._prepare(float(tau) + float(eps))
        Q64 = np.asarray(Q, dtype=np.float64)
        return self._kd.survivor_fraction(Q64, np.zeros(len(Q64)), float(tau) + float(eps), stage)

    def get_bytesize(self) -> int:
        if not self._built or self.N == 0:
            return 0
        return self._kd.nbytes() + int(sum(a.nbytes for a in (
            self._cell_U, self._cell_starts, self._cell_sizes, self._r64, self._order)))

    def __repr__(self):
        if not self._built:
            return f"BallIVFIndex(n_cells={self.n_cells}, not built)"
        return (f"BallIVFIndex(K={self.K}, N={self.N}, d={self.d}, "
                f"~{self.get_bytesize() // 1024} KB)")