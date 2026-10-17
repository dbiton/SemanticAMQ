"""
CapIVFIndex -- exact IVF index for CapCoverFilter (zero false negatives).

Accept test
-----------
    accept(q)  <=>  feasible(q)  and  exists cap j :  angle(q, u_j) <= theta_j + beta

u_j = v_j / |v_j| is cap j's decoded direction, theta_j its angle, beta the
per-query margin CapCoverFilter derives from tau and |q|.  Equivalently
||q - u_j||^2 <= |q|^2 + 1 - 2 |q| cos(theta_j + beta): a ball test on the sphere.

Index
-----
1. IVF cells: K = sqrt(N) spherical k-means cells.  Cell c is skipped only if
   angle(q, mu_c) > U_c + beta, where U_c = max_j (theta_j + angle(u_j, mu_c)).
2. Lower bound on a compact sketch of each cap: its first h centred-PCA
   coordinates in int8, its norm past them in uint8, its angle in uint8
   (rounded up).  A cap is dropped when the sketch's head distance (minus its
   rounding radius) plus the reverse triangle inequality on the rest exceeds
   its reach.  A chunk of queries scans the sketch together, in one pass.
3. Exact test: each survivor is decoded from CapCoverFilter's own codes
   (seeded PQ / TurboQuant, in place) and decided by the filter's rule, most
   slack first; a query within the filter's float margin of a boundary is
   returned undecided and decided by CapCoverFilter's own scan.  In larger
   batches each surviving cap is decoded once for all queries that kept it.
Compiled kernels: cap_ivf_kernels.py.

Correctness
-----------
Steps 1-2 discard a cap only via a valid lower bound (triangle inequality,
with the int8/uint8 rounding bounded and theta rounded up), widened by its
float rounding error, so no cap that accepts q is discarded.  Step 3 is the
filter's own test on the filter's own codes; pairs it cannot settle go to the
filter's scan.  Queries with theta_max + beta >= pi are accepted outright
(the widest cap covers the sphere); infeasible queries are rejected.

Threshold
---------
Dynamic: beta is passed per query at query time; nothing depends on it at
build time.

Cost
----
Build: k-means + PCA, O(N d (K + d)).  Memory (get_bytesize(), on top of the
filter's): h + 4 bytes per cap (head, tail, angle, locator), plus the
d x h PCA basis and the cells; no copy of the caps.  h is the widest head
whose sketch fits the budget the caller passes -- CapCoverFilter passes its
own size, so filter + index <= 2x filter.
"""

import numpy as np
from numba import get_num_threads
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
# Caps storage for the exact test (build time only)
# ---------------------------------------------------------------------------

def _exact_source(V: np.ndarray, theta: np.ndarray, codes, angle_grid):
    """Prepare the caps storage for the exact test, which runs in numba and
    so needs plain arrays.  Returns those arrays and the bytes the index owns.

    codes None: the index keeps its own float32 copy of the caps and their
    angles (counted).  codes given: CapCoverFilter's per-level storage, read
    in place (not counted) -- each level's code bytes, angle codes and keys
    are joined into one array and the filter's groups point at views of it
    (shared, not duplicated); codebooks are passed as is; the index adds a
    small level table L and the codewords' squared norms.  angle_grid =
    (lo, step) decodes an angle code: lo + code * step."""
    d = V.shape[1]
    tol = 2.0 * (d + 4) * 2.0 ** -24 # float32 dot product error, doubled
    z = lambda dt: np.zeros(1, dt)
    if codes is None:
        Xf = np.ascontiguousarray(V, dtype=np.float32)
        thf = np.ascontiguousarray(theta, dtype=np.float64)
        return ((IVF_K.KIND_FLAT, Xf, thf, z(np.uint8), z(np.int64), z(np.uint16), (np.zeros((1, 1), np.float32),),
                 z(np.float32), np.full((2, 8), np.iinfo(np.int64).max), 0.0, 0.0, tol), Xf.nbytes + thf.nbytes)
    groups = [g for g, _, _, _ in codes]
    books = tuple(np.ascontiguousarray(cw, dtype=np.float32) for _, cw, _, _ in codes)
    cbuf = np.concatenate([g.rem.ravel() for g in groups])
    abuf = np.concatenate([g.ang for g in groups])
    kbuf = np.concatenate([g.keys() for g in groups if g.w] or [z(np.int64)])
    nbuf = np.concatenate([np.einsum("ij,ij->i", cw.astype(np.float64), cw.astype(np.float64)) for cw in books]
                          ).astype(np.float32)
    L = np.zeros((len(groups) + 1, 8), dtype=np.int64)
    L[-1, 0] = np.iinfo(np.int64).max
    first = co = ko = no = 0
    for lv, (g, cw, bits, m) in enumerate(codes):
        n, stride = g.rem.shape
        L[lv] = (first, co, stride, ko, g.w, bits, m, no)
        g.rem = cbuf[co:co + n * stride].reshape(n, stride)
        g.ang = abuf[first:first + n]
        if g.w:
            g._keys_cache = kbuf[ko:ko + n]
            ko += n
        first += n
        co += n * stride
        no += len(cw)
    return (IVF_K.KIND_CODES, np.zeros((1, 1), np.float32), z(np.float64), cbuf, kbuf, abuf, books, nbuf, L,
            float(angle_grid[0]), float(angle_grid[1]), tol), 0


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
    parallel_min_batch : int
        Batches with at least this many queries run multi-threaded.
    """

    _ETA = 1e-6 # angle slack of the cell gate (float rounding)

    def __init__(self, n_cells: int = 4096, n_iter: int = 12, seed: int = 0, parallel_min_batch: int = 8):
        self.n_cells = n_cells
        self.n_iter = n_iter
        self.seed = seed
        self.parallel_min_batch = int(parallel_min_batch)
        self._built = False

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, directions: np.ndarray, angles: np.ndarray, rotation: np.ndarray = None,
             codes: list = None, angle_grid: tuple = None,  max_memory: int = None) -> None:
        """directions: [N,d] decoded cap vectors (norm ~1); angles: [N] cap
        angles in radians (kept in float64).  rotation: d x d matrix taking
        queries into the directions' basis (q @ rotation), None if they are
        already in it.  max_memory: the index's memory budget; its PCA head is
        the widest that fits (64 coordinates without a budget), and no index
        is built if none fits (_built stays False).

        codes: where the exact test reads cap i.  None keeps a float32 copy of
        the directions (standalone use; counted in the index's bytes).
        CapCoverFilter passes its set storage instead, read in place: per
        encoding level (group, codewords [2^bits, sub] float32, bits, slices),
        caps numbered level by level in group order, with angle_grid =
        (lo, step) decoding the groups' angle codes (lo + code * step)."""
        V = np.ascontiguousarray(directions, dtype=np.float32)
        theta = np.clip(np.asarray(angles, dtype=np.float64).ravel(), 0.0, np.pi)
        N, d = V.shape
        self.N, self.d, self.K, self.h = N, d, 0, 0
        self._rotation = rotation
        n = np.linalg.norm(V.astype(np.float64), axis=1)
        # a zero decoded vector satisfies <q,v> >= |v| cos(.) for every q: it
        # accepts every feasible query, and query() keeps that, trivially
        self._degenerate = bool((n < 1e-12).any())
        self._built = N == 0 or self._degenerate
        if self._built:
            return
        U64 = V / n[:, None]
        self._theta_max = float(theta.max())

        # cells: K = sqrt(N) spherical k-means on the unit directions, stored
        # by cell (widest cap first within a cell); a cell's reach
        # U_c = max over its caps of theta + angle(u, mu_c), rounded up
        mu, labels = _spherical_kmeans(U64.astype(np.float32), max(1, min(int(self.n_cells), int(np.sqrt(N)))),
                                       self.n_iter, self.seed)
        used, labels = np.unique(labels, return_inverse=True) # without empty cells
        order = np.lexsort((-theta, labels))
        counts = np.bincount(labels)
        self.K = K = len(counts)
        mu64 = mu[used].astype(np.float64)
        mu64 /= np.linalg.norm(mu64, axis=1, keepdims=True)
        delta = 2.0 * np.arcsin(np.minimum(np.linalg.norm(U64 - mu64[labels], axis=1) * 0.5, 1.0))
        cell_U = np.full(K, -np.inf)
        np.maximum.at(cell_U, labels, theta + delta * (1.0 + 1e-9) + 1e-9)

        # the sketch, with the widest head that fits the budget
        src, owned = _exact_source(V, theta, codes, angle_grid)
        Np = int((-(-counts // IVF_K.PAD) * IVF_K.PAD).sum())
        self.h = IVF_K.fit_head(Np, N, K, d, None if max_memory is None else max_memory - owned)
        if self.h == 0:
            return
        self._kd = IVF_K.KernelData(U64[order], theta[order], order, mu64, cell_U,
                                    np.concatenate(([0], np.cumsum(counts))), src, owned, IVF_K.pca_basis(U64), self.h)
        self._built = True

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, Qn: np.ndarray, beta: np.ndarray, feasible: np.ndarray,
              exact=None) -> np.ndarray:
        """bool[nq]: does some cap accept the query?  Zero false negatives.

        Qn: unit queries [nq,d] (rotated here into the caps' basis, see
        build); beta: per-query margin (radians); feasible: bool mask
        (infeasible queries are rejected); exact: callable(rows) ->
        bool[len(rows)] deciding the queries `rows` exactly (CapCoverFilter
        passes its own scan).  It is called only for the rare query the index
        leaves undecided; without it such a query is accepted (no false
        negative, at most a false positive within float rounding of a cap's
        boundary)."""
        if not self._built:
            raise RuntimeError("CapIVFIndex is not built (no build(), or its budget fits no sketch)")
        nq = len(Qn)
        if self.N == 0 or nq == 0:
            return np.zeros(nq, dtype=bool)
        feasible = np.asarray(feasible, dtype=bool)
        if self._degenerate:
            return feasible.copy()
        beta = np.ascontiguousarray(beta, dtype=np.float64).reshape(-1)
        # the widest cap already covers the sphere: accept, no scan needed
        full = feasible & (beta >= np.pi - self._theta_max)
        Q64 = np.asarray(Qn, dtype=np.float64)
        if self._rotation is not None:
            Q64 = Q64 @ self._rotation
        parallel = nq >= self.parallel_min_batch and get_num_threads() > 1
        out = self._kd.run(feasible & ~full, np.ascontiguousarray(Q64), beta, self._ETA, parallel)
        rows = self._kd.undecided()
        if rows is not None:
            out[rows] = True if exact is None else np.asarray(exact(rows), dtype=bool)
        return out | full

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def pruning_fraction(self, Q: np.ndarray, beta: np.ndarray = None, stage: str = "full") -> np.ndarray:
        """Per query, the fraction of caps still alive after the IVF cell gate
        (stage="gate") or after the gate and the sketch's lower bound, i.e.
        reaching the exact test (stage="full", the default).  0.0 = everything
        pruned, 1.0 = nothing; queries with theta_max + beta >= pi count as 1.0
        (accepted outright, no pruning needed).  Q is normalised here."""
        if not self._built or self.N == 0 or self._degenerate:
            return np.ones(len(Q))
        Q64 = np.asarray(Q, dtype=np.float64)
        Q64 = Q64 / np.maximum(np.linalg.norm(Q64, axis=1, keepdims=True), 1e-300)
        if self._rotation is not None:
            Q64 = Q64 @ self._rotation
        b = np.zeros(len(Q64)) if beta is None else np.asarray(beta, dtype=np.float64).ravel()
        out = self._kd.survivor_fraction(Q64, b, self._ETA, stage)
        out[b >= np.pi - self._theta_max] = 1.0
        return out

    def layout(self):
        """(order, cell_starts, cell_U): input row of each cap in storage order,
        each cell's first position in that order, each cell's reach U_c --
        recomputed from the kernel data (diagnostics and tests)."""
        _, cell_of, order = self._kd.layout()
        starts = np.concatenate(([0], np.cumsum(np.bincount(cell_of, minlength=self.K))))
        return order, starts, self._kd.cellU

    def get_bytesize(self) -> int:
        if not self._built or self.N == 0 or self._degenerate:
            return 0
        return self._kd.nbytes()

    def __repr__(self):
        if not self._built:
            return f"CapIVFIndex(n_cells={self.n_cells}, not built)"
        return (f"CapIVFIndex(K={self.K}, N={self.N}, d={self.d}, h={self.h}, "
                f"~{self.get_bytesize() // 1024} KB)")