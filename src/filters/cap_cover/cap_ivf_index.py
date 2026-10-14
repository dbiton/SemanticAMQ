"""
CapIVFIndex -- exact IVF index for CapCoverFilter (zero false negatives).

Accept test
-----------
    accept(q)  <=>  feasible(q)  and  exists cap j :  angle(q, u_j) <= theta_j + beta

u_j = v_j / |v_j| is cap j's decoded direction, theta_j its angle, beta the
per-query margin CapCoverFilter derives from tau and |q|.  Equivalently
||q - u_j||^2 <= |q|^2 + 1 - 2 cos(theta_j + beta): a ball test on the sphere.

Index
-----
1. IVF cells: K = sqrt(N) spherical k-means cells, visited nearest-first.
   Cell c is skipped only if angle(q, mu_c) > U_c + beta, where
   U_c = max_j (theta_j + angle(u_j, mu_c)) over its caps.
2. Early abandoning: caps are stored in a centred-PCA basis; a cap is dropped
   once a lower bound on ||q - u_j||^2 (partial sum over the leading
   coordinates + reverse triangle inequality on the rest) exceeds its reach.
   ~99.8% of caps are dropped within the first 64 of 384 coordinates.
3. Exact decision: survivors are tested in full; pairs within float32 rounding
   of the boundary are re-measured in float64 by CapCoverFilter's own rule.
Large batches compute step 2's first bound for all pairs as one BLAS GEMM.
Compiled kernels: cap_ivf_kernels.py.

Correctness
-----------
Steps 1-2 discard a cap only via a valid lower bound (triangle inequality),
widened by its float rounding error, so no cap that accepts q is discarded.
Step 3 leans towards accept by _TINY, so it never rejects a pair the scan
accepts.  Queries with theta_max + beta >= pi are accepted outright (the
widest cap covers the sphere); infeasible queries are rejected.

Threshold
---------
Dynamic: beta is passed per query at query time; nothing depends on it at
build time.

Cost
----
Build: k-means + PCA, O(N d (K + d)).  Memory: a float32 copy of the caps plus
their 64-coordinate PCA head, outside the filter's budget (get_bytesize()).
"""

import numpy as np
from filters.cap_cover import cap_ivf_kernels as IVF_K


# ---------------------------------------------------------------------------
# Spherical k-means (build time only)
# ---------------------------------------------------------------------------

def _spherical_kmeans(X: np.ndarray, K: int, n_iter: int = 12, seed: int = 0):
    """(unit float32 centroids [K,d], int32 labels [N]).  X: unit float32 rows."""
    rng = np.random.default_rng(seed)
    N, d = X.shape
    K = min(K, N)
    if K == 0:
        return np.empty((0, d), np.float32), np.empty(0, np.int32)

    # k-means++ seeding (chord distance^2 = 2 - 2<x,c>) keeps cells balanced
    centroids = np.empty((K, d), np.float32)
    first = int(rng.integers(N))
    centroids[0] = X[first]
    best = np.maximum(2.0 - 2.0 * (X @ X[first]), 0.0).astype(np.float64)
    for k in range(1, K):
        tot = best.sum()
        nxt = int(rng.choice(N, p=best / tot)) if tot > 0 else int(rng.integers(N))
        centroids[k] = X[nxt]
        best = np.minimum(best, np.maximum(2.0 - 2.0 * (X @ X[nxt]), 0.0))

    labels = np.full(N, -1, np.int32)
    for _ in range(n_iter):
        ip = X @ centroids.T
        new_labels = ip.argmax(axis=1).astype(np.int32)

        # re-seat empty cells to random data points
        counts = np.bincount(new_labels, minlength=K)
        empty = np.flatnonzero(counts == 0)
        if len(empty):
            # re-seat on the points least well served by their centroid
            fit = ip[np.arange(N), new_labels]
            for c, p in zip(empty, np.argsort(fit)[:len(empty)]):
                new_labels[p] = c
            counts = np.bincount(new_labels, minlength=K)

        order = np.argsort(new_labels, kind="stable")
        starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
        sums = np.add.reduceat(X[order].astype(np.float64), starts, axis=0)
        nrm = np.linalg.norm(sums, axis=1, keepdims=True)
        centroids = (sums / np.maximum(nrm, 1e-300)).astype(np.float32)

        if (new_labels == labels).all():
            break
        labels = new_labels
    return centroids, labels


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

class CapIVFIndex:
    """
    Exact IVF index for CapCoverFilter.  Zero false negatives.

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
        Batches with at least this many live queries run multi-threaded.
    """

    _TINY = 4e-7 # accept-side slack of the exact rule, per unit norm

    def __init__(self, n_cells: int = 4096, n_iter: int = 12, seed: int = 0,
                 parallel_min_batch: int = 8, gemm_min_batch: int = 16):
        self.n_cells = n_cells
        self.n_iter = n_iter
        self.seed = seed
        self.parallel_min_batch = int(parallel_min_batch)
        self.gemm_min_batch = int(gemm_min_batch)
        self._built = False

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, directions: np.ndarray, angles: np.ndarray) -> None:
        """directions: [N,d] decoded cap vectors (norm ~1), in the basis queries
        will be passed in (CapCoverFilter passes the input basis); angles: [N]
        cap angles in radians (kept in float64)."""
        V = np.ascontiguousarray(directions, dtype=np.float32)
        theta = np.clip(np.asarray(angles, dtype=np.float64).ravel(), 0.0, np.pi)
        N, d = V.shape
        self.N, self.d = N, d
        self._degenerate = False
        self.K = 0

        if N == 0:
            self._built = True
            return

        V64 = V.astype(np.float64)
        n = np.linalg.norm(V64, axis=1)
        if (n < 1e-12).any():
            # A zero decoded vector satisfies <q,v> >= |v| cos(.) for every q:
            # it accepts every feasible query.  Keep that behaviour, trivially.
            self._degenerate = True
            self._built = True
            return
        U64 = V64 / n[:, None]

        # --- cells: K = sqrt(N) spherical k-means on the unit directions ---
        K = max(1, min(int(self.n_cells), int(np.sqrt(N))))
        mu, labels = _spherical_kmeans(U64.astype(np.float32), K, self.n_iter, self.seed)
        used = np.unique(labels)
        remap = np.full(len(mu), -1, np.int64)
        remap[used] = np.arange(len(used))
        labels = remap[labels]
        mu64 = mu[used].astype(np.float64)
        mu64 /= np.linalg.norm(mu64, axis=1, keepdims=True)
        K = len(mu64)

        # per-cap angle to its own centroid, rounded up; cell reach U_c
        chord = np.linalg.norm(U64 - mu64[labels], axis=1)
        delta = 2.0 * np.arcsin(np.minimum(chord * 0.5, 1.0))
        delta = delta * (1.0 + 1e-9) + 1e-9
        cell_U = np.full(K, -np.inf)
        np.maximum.at(cell_U, labels, theta + delta)

        # storage: by cell, widest cap first within a cell
        order = np.lexsort((-theta, labels))
        self._order = order.astype(np.int64) # storage row -> input row
        counts = np.bincount(labels, minlength=K)
        self._cell_starts = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
        self._cell_sizes = counts.astype(np.float64)
        self._cell_U = cell_U
        self.K = K

        th_s = theta[order]
        self._theta_max = float(theta.max())
        # kernel data: unit directions in storage order; reach of cap j at margin beta
        #   ||q - u_j||^2 <= |q|^2 + 1 + RB_j cos(beta) + RC_j sin(beta)
        self._eip = 2.0 * (d + 6) * 2.0 ** -24 + 1e-6 # float32 error of the full <q,v> test, per |v|
        self._kd = IVF_K.KernelData(U64[order], V[order], -2.0 * np.cos(th_s), 2.0 * np.sin(th_s),
                                 n[order], mu64, cell_U, self._cell_starts, gemm_ck=2)
        # cos-domain slack of the exact rule -> angle slack for the gate
        # (arccos is 1/2-Holder with constant pi/sqrt2): cells are never cut by it
        t = (self._TINY + 2.0 * self._eip) / float(n.min())
        self._eta = 2.3 * float(np.sqrt(t)) + 1e-9
        self._built = True

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, Qn: np.ndarray, beta: np.ndarray, feasible: np.ndarray) -> np.ndarray:
        """bool[nq]: does some cap accept the query?  Exact, 0-FNR.

        Qn: unit queries [nq,d] (float64 -- the exact boundary rule is
        evaluated on them; float32 input is widened); beta: per-query margin
        (radians); feasible: bool mask (infeasible queries are rejected)."""
        nq = len(Qn)
        if not self._built or self.N == 0 or nq == 0:
            return np.zeros(nq, dtype=bool)
        feasible = np.asarray(feasible, dtype=bool)
        if self._degenerate:
            return feasible.copy()
        beta = np.ascontiguousarray(beta, dtype=np.float64).reshape(-1)

        # the widest cap already covers the sphere -> accept; no scan needed
        full = beta >= np.pi - self._theta_max
        live = feasible & ~full
        Q64 = np.ascontiguousarray(Qn, dtype=np.float64)
        n_live = int(live.sum())
        parallel = n_live >= self.parallel_min_batch and IVF_K.get_num_threads() > 1
        if n_live >= self.gemm_min_batch:
            out = self._kd.run_gemm(live, Q64, beta, self._TINY, self._eip, parallel)
        else:
            out = self._kd.run(live, Q64, beta, self._eta, self._TINY, self._eip, parallel)
        out |= feasible & full
        return out

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def pruning_fraction(self, Q: np.ndarray, beta: np.ndarray = None, stage: str = "full") -> np.ndarray:
        """Per query, the fraction of caps still alive after the IVF cell gate
        (stage="gate") or after the gate and every early-abandon checkpoint,
        i.e. measured in full (stage="full", the default).  0.0 = everything
        pruned, 1.0 = nothing; queries with theta_max + beta >= pi count as 1.0
        (accepted outright, no pruning needed).  Q is normalised here."""
        if not self._built or self.N == 0 or self._degenerate:
            return np.ones(len(Q))
        Q64 = np.asarray(Q, dtype=np.float64)
        Q64 = Q64 / np.maximum(np.linalg.norm(Q64, axis=1, keepdims=True), 1e-300)
        b = np.zeros(len(Q64)) if beta is None else np.asarray(beta, dtype=np.float64).ravel()
        out = self._kd.survivor_fraction(Q64, b, self._eta, stage)
        out[b >= np.pi - self._theta_max] = 1.0
        return out

    def get_bytesize(self) -> int:
        if not self._built or self.N == 0 or self._degenerate:
            return 0
        return self._kd.nbytes() + int(sum(a.nbytes for a in (
            self._cell_U, self._cell_starts, self._cell_sizes, self._order)))

    def __repr__(self):
        if not self._built:
            return f"CapIVFIndex(n_cells={self.n_cells}, not built)"
        return (f"CapIVFIndex(K={self.K}, N={self.N}, d={self.d}, "
                f"~{self.get_bytesize() // 1024} KB)")