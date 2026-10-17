"""
BallIVFIndex -- exact IVF index for BallCoverFilter (zero false negatives).

Accept test
-----------
    accept(q)  <=>  exists ball j :  ||q - c_j||  <=  r_j + tau + eps

c_j is ball j's decoded center, r_j its radius, tau the distance threshold and
eps BallCoverFilter's coverage margin.  A negative reach admits nothing.

Index
-----
1. IVF cells: K = sqrt(N) k-means cells.  Cell c is skipped only if
   ||q - mu_c|| > U_c + tau + eps, where U_c = max_j (r_j + ||c_j - mu_c||).
2. Lower bound on a compact sketch of each center: its first h centred-PCA
   coordinates in int8 and its norm past them in uint8.  A ball is dropped
   when the sketch's head distance (minus its rounding radius) plus the
   reverse triangle inequality on the rest exceeds its reach.  A chunk of
   queries scans the sketch together, in one pass.
3. Exact test: each survivor is decoded from BallCoverFilter's own code
   (4-bit nibbles through the quantizer's decode table, or float32), with the
   filter's own radius, as soon as step 2 finds it (largest ball first within
   a cell); a query within the float margin of a boundary is returned
   undecided and decided by BallCoverFilter's own scan.
Compiled kernels: ball_ivf_kernels.py.

Correctness
-----------
Steps 1-2 discard a ball only via a valid lower bound (triangle inequality,
with the int8/uint8 rounding bounded), widened by its float rounding error and
by the bound on the query's rotation error, so no ball that accepts q is
discarded.  Step 3 is the filter's own test on the filter's own codes and
radii; pairs it cannot settle go to the filter's scan.

Threshold
---------
Dynamic: tau is passed at query time; nothing depends on it at build time.

Cost
----
Build: k-means + PCA, O(N d (K + d)).  Memory (get_bytesize(), on top of the
filter's): h + 3 bytes per ball (head, tail, locator), plus the d x h
PCA basis, the cells and the 4-bit decode table; no copy of the centers.  h is
the widest head whose sketch fits the budget the caller passes --
BallCoverFilter passes its own size, so filter + index <= 2x filter.
"""

import numpy as np
from numba import get_num_threads
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
    parallel_min_batch : int
        Batches with at least this many queries run multi-threaded.
    """

    def __init__(self, n_cells: int = 4096, n_iter: int = 12, seed: int = 0, parallel_min_batch: int = 8):
        self.n_cells = n_cells
        self.n_iter = n_iter
        self.seed = seed
        self.parallel_min_batch = int(parallel_min_batch)
        self._built = False

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, centers: np.ndarray, radii: np.ndarray, rotation=None, codes: np.ndarray = None,
              table: np.ndarray = None, ids: np.ndarray = None, max_memory: int = None) -> None:
        """centers: [N,d] decoded centers; radii: ball i's radius is
        radii[ids[i]] (ids: None = 0 .. N-1, i.e. one radius per center).
        rotation: (A, b) taking float32 queries into the centers' basis,
        q @ A.T + b (b may be None), or None if they are already in it; the
        index allows for the float32 rounding of that product.  max_memory:
        the index's memory budget; its PCA head is the widest that fits (64
        coordinates without a budget), and no index is built if none fits
        (_built stays False).

        codes: where the exact test reads ball i.  None keeps a float32 copy
        of the centers, radii and ids (standalone use; counted in the index's
        bytes).  BallCoverFilter passes its own code rows instead, read in
        place with its radii and id map: FP32 codes (row i: center i's bytes),
        or 4-bit codes with the quantizer's decode table (row i, byte b: the
        nibbles of dimensions 2b (low) and 2b + 1; table[t, v]: what nibble v
        of dimension t decodes to)."""
        centers = np.ascontiguousarray(centers, dtype=np.float32)
        N, d = centers.shape
        rad = np.asarray(radii).ravel()
        idm = np.arange(N, dtype=np.int64) if ids is None else ids
        radii = rad[idm].astype(np.float32) # ball i's radius
        self.N, self.d, self.K, self.h = N, d, 0, 0
        self._rotation = rotation
        self._built = N == 0
        if self._built:
            return

        # cells: K = sqrt(N) k-means, stored by cell (widest ball first within
        # a cell); centroids = their members' mean (float64), so that a
        # cell's reach U_c = max over its balls of r + ||c - mu_c||, rounded
        # up, is exact for them
        _, labels = _euclidean_kmeans(centers, max(1, min(int(self.n_cells), int(np.sqrt(N)))),
                                      self.n_iter, self.seed)
        _, labels = np.unique(labels, return_inverse=True) # without empty cells
        order = np.lexsort((-radii, labels))
        counts = np.bincount(labels)
        cell_starts = np.concatenate(([0], np.cumsum(counts)))
        self.K = K = len(counts)
        c64 = centers.astype(np.float64)
        mu64 = np.add.reduceat(c64[order], cell_starts[:-1]) / counts[:, None]
        delta = np.linalg.norm(c64 - mu64[labels], axis=1)
        cell_U = np.full(K, -np.inf)
        np.maximum.at(cell_U, labels, radii.astype(np.float64) + delta * (1.0 + 1e-9) + 1e-9)

        # where the exact test reads ball i: the kernels take plain arrays
        # (numba cannot read the filter's objects)
        rq = 0.0
        if rotation is not None:
            # the filter rotates its queries with the same float32 matrix but
            # possibly another summation order: each product is within
            # gamma_d |q| per coordinate (rows of A are unit), so the two
            # within 2 sqrt(d) gamma_d |q| (gamma_d = d u / (1 - d u), u = 2^-24)
            u = 2.0 ** -24
            rq = 2.0 * np.sqrt(d) * (d * u / (1.0 - d * u)) * (1.0 + 1e-6)
        kind, owned = IVF_K.KIND_FLAT, 0
        Xf, cbuf, T = np.zeros((1, 1), np.float32), np.zeros((1, 1), np.uint8), np.zeros((1, 16), np.float32)
        if codes is None: # standalone: a copy of its own
            Xf, rad = centers, radii
            owned = centers.nbytes + rad.nbytes + idm.nbytes
        elif table is None: # the filter's FP32 codes
            Xf = codes.view(np.float32)
        else: # its 4-bit codes and their decode table
            kind, cbuf, T = IVF_K.KIND_NIBBLE, codes, np.ascontiguousarray(table, dtype=np.float32)
            owned = T.nbytes
        src = (kind, Xf, cbuf, T, rad, idm, 0.0, rq)

        # the sketch, with the widest head that fits the budget
        Np = int((-(-counts // IVF_K.PAD) * IVF_K.PAD).sum())
        self.h = IVF_K.fit_head(Np, N, K, d, None if max_memory is None else max_memory - owned)
        if self.h == 0:
            return
        self._kd = IVF_K.KernelData(c64[order], radii[order], order, mu64, cell_U, cell_starts, src, owned,
                                    IVF_K.pca_basis(c64), self.h)
        self._built = True

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, Q: np.ndarray, tau: float, eps: float, exact=None) -> np.ndarray:
        """bool[nq]: does some ball accept the query?  Zero false negatives.

        Q: queries [nq,d] (rotated here into the centers' basis, see build).
        exact: callable(rows) -> bool[len(rows)] deciding the queries `rows`
        exactly (BallCoverFilter passes its own scan).  It is called only for
        the rare query the index leaves undecided (within float rounding, or
        the rotation bound, of a ball's boundary); without it such a query is
        accepted (no false negative)."""
        if not self._built:
            raise RuntimeError("BallIVFIndex is not built (no build(), or its budget fits no sketch)")
        nq = len(Q)
        if self.N == 0 or nq == 0:
            return np.zeros(nq, dtype=bool)
        if self._rotation is not None:
            A, b = self._rotation
            Q = np.asarray(Q, dtype=np.float32) @ A.T
            if b is not None:
                Q += b
        Q64 = np.ascontiguousarray(Q, dtype=np.float64)
        parallel = nq >= self.parallel_min_batch and get_num_threads() > 1
        out = self._kd.run(np.ones(nq, dtype=np.bool_), Q64, float(tau) + float(eps), parallel)
        rows = self._kd.undecided()
        if rows is not None:
            out[rows] = True if exact is None else np.asarray(exact(rows), dtype=bool)
        return out

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def pruning_fraction(self, Q: np.ndarray, tau: float = 0.5, eps: float = 1e-3,
                         stage: str = "full") -> np.ndarray:
        """Per query, the fraction of balls still alive after the IVF cell gate
        (stage="gate") or after the gate and the sketch's lower bound, i.e.
        reaching the exact test (stage="full", the default).  0.0 =
        everything pruned, 1.0 = nothing."""
        if not self._built or self.N == 0:
            return np.ones(len(Q))
        Q = np.asarray(Q, dtype=np.float64)
        if self._rotation is not None:
            A, b = self._rotation
            Q = Q @ A.T + (0.0 if b is None else b)
        return self._kd.survivor_fraction(Q, float(tau) + float(eps), stage)

    def layout(self):
        """(order, cell_starts, cell_U): input row of each ball in storage
        order, each cell's first position in that order, each cell's reach
        U_c -- recomputed from the kernel data (diagnostics and tests)."""
        _, cell_of, order = self._kd.layout()
        starts = np.concatenate(([0], np.cumsum(np.bincount(cell_of, minlength=self.K))))
        return order, starts, self._kd.cellU

    def get_bytesize(self) -> int:
        if not self._built or self.N == 0:
            return 0
        return self._kd.nbytes()

    def __repr__(self):
        if not self._built:
            return f"BallIVFIndex(n_cells={self.n_cells}, not built)"
        return (f"BallIVFIndex(K={self.K}, N={self.N}, d={self.d}, h={self.h}, "
                f"~{self.get_bytesize() // 1024} KB)")