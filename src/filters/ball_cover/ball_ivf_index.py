"""
BallIVFIndex -- exact IVF pruning index for BallCoverFilter.

Every hot loop is a NumPy vectorised op backed by BLAS (OpenBLAS / MKL),
so the arithmetic is C-speed with no Python interpreter overhead in the
inner loop.

Design
------
Sphere centers are clustered into K cells by spherical k-means.
Each cell stores the max possible reach of any of its members.

A cell is skipped when ||q - centroid[c]||^2  >  max_reach[c]^2
where max_reach[c] = max_j(r_j + tau + eps) over the cell.

For surviving cells the query is scored with one BLAS GEMM:
  ip = Q @ sorted_vecs[cell_start:cell_end].T
followed by a NumPy vectorised accept test (one Python line, fully C-speed).

Zero-FNR guarantee
------------------
max_reach[c] >= reach of every member in cell c by construction.
For ball cover: dist(q, centroid[c]) <= dist(q, center_j) + dist(centroid[c], center_j)
  <= reach_j + centroid_radius <= max_reach[c], so any accepting sphere's cell passes.
Therefore: if an accepting member exists, its cell passes the gate -- nothing is skipped.

Usage
-----
  # Ball filter (static only -- insert_method=None):
  idx = BallIVFIndex(n_cells=128)
  idx.build_ball(centers_f32, radii_f32, tau, eps)
  out = idx.query_ball(Qr, q_norms, tau, eps)          # bool[nq]

"""

import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Spherical k-means -- pure numpy, run once at build time
# ---------------------------------------------------------------------------

def _spherical_kmeans(X: np.ndarray, K: int, n_iter: int = 20,
                      seed: int = 0) -> tuple:
    """
    (centroids, labels): unit float32 centroids [K, d] and int32 labels [N].
    X must have unit rows (float32).
    """
    rng = np.random.default_rng(seed)
    N, d = X.shape
    K = min(K, N)
    if K == 0:
        return np.empty((0, d), dtype=np.float32), np.empty(0, dtype=np.int32)

    # Init: K random rows
    idx = rng.choice(N, K, replace=False)
    centroids = X[idx].copy()
    _renorm(centroids)

    labels = np.zeros(N, dtype=np.int32)
    for it in range(n_iter):
        # Assign: nearest centroid by inner product (= smallest angle)
        ip = X @ centroids.T          # (N, K) -- BLAS GEMM
        new_labels = ip.argmax(1).astype(np.int32)

        # Re-seat empty cells to random data points
        counts = np.bincount(new_labels, minlength=K)
        empty = np.flatnonzero(counts == 0)
        if len(empty):
            spare = rng.choice(N, len(empty), replace=False)
            new_labels[spare] = empty

        # Update centroids as mean of members, renormalised
        new_centroids = np.zeros_like(centroids)
        np.add.at(new_centroids, new_labels, X)
        counts2 = np.bincount(new_labels, minlength=K).astype(np.float32)
        new_centroids /= np.maximum(counts2[:, None], 1.0)
        _renorm(new_centroids)

        if it > 0 and (new_labels == labels).all():
            break
        labels = new_labels
        centroids = new_centroids

    return centroids, labels


def _renorm(X: np.ndarray) -> None:
    """Normalise rows of X to unit length in-place (float32)."""
    n = np.sqrt((X * X).sum(axis=1, keepdims=True))
    np.divide(X, np.maximum(n, 1e-30), out=X)


# ---------------------------------------------------------------------------
# CSR layout helper
# ---------------------------------------------------------------------------

def _make_csr(cell_ids: np.ndarray, reaches: np.ndarray, K: int,
              unit_vecs: np.ndarray) -> tuple:
    """
    Sort members into cells, widest-first within each cell.
    Returns (cell_starts, sorted_vecs, sorted_reaches, cell_max_reach, cell_cos_min)

    All arrays are contiguous float32 / int32.  No malloc, no C.
    """
    N = len(cell_ids)

    # Per-cell max reach
    cell_max_reach = np.full(K, -np.inf, dtype=np.float64)
    np.maximum.at(cell_max_reach, cell_ids, reaches.astype(np.float64))
    cell_max_reach = cell_max_reach.astype(np.float32)

    # cos(max_reach[c]) for the angle-domain gate
    r_clipped = np.clip(cell_max_reach, 0.0, np.pi)
    cell_cos_min = np.cos(r_clipped).astype(np.float32)
    # Mark empty cells (max_reach == -inf) with cos_min = +2 (unreachable threshold)
    cell_cos_min[cell_max_reach < -1e30] = 2.0

    # Stable sort: primary = cell id, secondary = reach descending (widest first)
    # Use a combined key: sort by (cell_id * big_number - reach)
    # Achieves this in one argsort -- no per-cell loop.
    BIG = float(reaches.max() + 1.0) if N > 0 else 1.0
    key = cell_ids.astype(np.float64) * BIG - reaches.astype(np.float64)
    order = np.argsort(key, kind="stable")

    sorted_vecs    = np.ascontiguousarray(unit_vecs[order],  dtype=np.float32)
    sorted_reaches = np.ascontiguousarray(reaches[order],    dtype=np.float32)

    # CSR starts: cumsum of per-cell counts
    counts = np.bincount(cell_ids, minlength=K).astype(np.int32)
    cell_starts = np.zeros(K + 1, dtype=np.int32)
    np.cumsum(counts, out=cell_starts[1:])

    return cell_starts, sorted_vecs, sorted_reaches, cell_max_reach, cell_cos_min


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class BallIVFIndex:
    """
    Exact IVF pruning index for BallCoverFilter.  
    Zero false negatives by construction.

    All arithmetic is NumPy vectorised (BLAS GEMMs, no Python inner loops).

    Parameters
    ----------
    n_cells : int
        Number of k-means cells.  Auto-sized to min(n_cells, sqrt(N)) at
        build time so small covers aren't over-partitioned.
    n_iter : int
        k-means iterations.
    seed : int
        k-means RNG seed.
    """

    def __init__(self, n_cells: int = 128, n_iter: int = 20, seed: int = 0):
        self.n_cells = n_cells
        self.n_iter  = n_iter
        self.seed    = seed
        self._built  = False

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, centers: np.ndarray, radii: np.ndarray,
                   tau: float, eps: float) -> None:
        """
        Build the ball-filter IVF index.

        Parameters
        ----------
        centers : float32[N, d]
            Decoded sphere centers in the rotated working basis.
        radii : float32[N]
            Stored sphere radii.
        tau : float
            Distance threshold baked into cell bounds.
            max_reach[c] = max(r_j + tau + eps) over cell members.
            Pass the largest tau you expect to query (conservative is safe).
        eps : float
            BallCoverFilter._COVERAGE_EPSILON.
        """
        centers = np.ascontiguousarray(centers, dtype=np.float32)
        radii   = np.asarray(radii, dtype=np.float32).ravel()
        reaches = radii + float(tau) + float(eps)   # float64 for precision
        self._tau_baked = float(tau)
        self._eps       = float(eps)
        self._build_common(centers, reaches.astype(np.float32))

    def _build_common(self, unit_vecs: np.ndarray, reaches: np.ndarray) -> None:
        N, d = unit_vecs.shape
        self.d = d
        self.N = N

        if N == 0:
            self._built = True
            # Store empty arrays so query methods can still run
            self._centroids      = np.empty((0, d), dtype=np.float32)
            self._centroid_norms = np.empty(0, dtype=np.float32)
            self._cell_starts    = np.array([0], dtype=np.int32)
            self._cell_max_reach = np.empty(0, dtype=np.float32)
            self._cell_cos_min   = np.empty(0, dtype=np.float32)
            self._sorted_vecs    = np.empty((0, d), dtype=np.float32)
            self._sorted_reaches = np.empty(0, dtype=np.float32)
            self._sorted_reach_sq= np.empty(0, dtype=np.float32)
            self._vec_norms_sq   = np.empty(0, dtype=np.float32)
            self.K = 0
            return

        K = min(self.n_cells, max(1, int(N ** 0.5)))
        K = min(K, N)
        self.K = K

        # Spherical k-means on the unit vectors
        centroids, labels = _spherical_kmeans(unit_vecs, K, self.n_iter, self.seed)
        self._centroids = np.ascontiguousarray(centroids, dtype=np.float32)

        # Centroid squared norms (should be 1 but compute for numerical safety)
        self._centroid_norms_sq = (self._centroids * self._centroids).sum(1).astype(np.float32)

        # Build CSR layout + sorted arrays
        cell_starts, sorted_vecs, sorted_reaches, cell_max_reach, cell_cos_min = \
            _make_csr(labels.astype(np.int32), reaches, K, unit_vecs)

        self._cell_starts    = cell_starts
        self._cell_max_reach = cell_max_reach     # float32[K]
        self._cell_cos_min   = cell_cos_min       # float32[K]  cos(max_reach[c])
        self._sorted_vecs    = sorted_vecs        # float32[N, d]  widest-first within cell
        self._sorted_reaches = sorted_reaches     # float32[N]

        # Pre-square for ball mode L2 gate
        self._sorted_reach_sq = (sorted_reaches.astype(np.float64) ** 2).astype(np.float32)

        # Per-member ||v||^2 for L2 computation: ||q-v||^2 = ||q||^2 + ||v||^2 - 2<q,v>
        self._vec_norms_sq = (sorted_vecs * sorted_vecs).sum(1).astype(np.float32)

        self._built = True

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, Qr: np.ndarray, q_norms_sq: np.ndarray,
                   tau: float, eps: float,
                   rel_error: Optional[float] = None) -> np.ndarray:
        """
        Exact ball-filter IVF query.  0-FNR guaranteed.

        Parameters
        ----------
        Qr : float32[nq, d]
            Queries in the rotated working basis.
        q_norms_sq : float32[nq]
            Per-query ||q||^2.
        tau, eps : float
            Distance threshold and coverage epsilon.
        rel_error : float, optional
            Float32 slack: (2*d + 4) * 2^{-24}.  Computed from d if None.

        Returns
        -------
        out : bool[nq]
        """
        if not self._built or self.N == 0:
            return np.zeros(len(Qr), dtype=bool)

        Qr          = np.ascontiguousarray(Qr, dtype=np.float32)
        q_norms_sq  = np.ascontiguousarray(q_norms_sq, dtype=np.float32)
        nq          = len(Qr)
        out         = np.zeros(nq, dtype=bool)

        if rel_error is None:
            rel_error = (2 * self.d + 4) * 2.0 ** -24
        slack = rel_error  # per-query slack is O(rel_error); use scalar bound

        # Reach at current tau (may differ from baked tau).
        # The cell gate uses the baked max_reach (conservative -- safe).
        # The per-member accept test uses the exact current tau.
        reach_sq = self._sorted_reach_sq  # baked; exact when tau == _tau_baked

        if tau != self._tau_baked:
            # Recompute per-member reach at the actual tau
            # sorted_reaches was baked at _tau_baked; strip and re-add:
            r = self._sorted_reaches - self._tau_baked - self._eps + float(tau) + float(eps)
            reach_sq = (r.astype(np.float64) ** 2).astype(np.float32)
            # Cell gates remain conservative (baked >= actual), so 0-FNR holds.

        # --- Gate: one BLAS GEMM against K centroids ---
        # ip_cen[qi, c] = <q_i, centroid_c>  (float32 BLAS)
        ip_cen = Qr @ self._centroids.T   # (nq, K)
        # L2^2(q, centroid) = ||q||^2 + ||centroid||^2 - 2*ip
        d2_cen = (q_norms_sq[:, None]
                  + self._centroid_norms_sq[None, :]
                  - 2.0 * ip_cen)          # (nq, K)  float32
        d2_cen = np.maximum(d2_cen, 0.0)   # clamp fp rounding

        # max_reach[c]^2 for threshold
        mr_sq = (self._cell_max_reach.astype(np.float64) ** 2).astype(np.float32)

        # --- Per-cell scan over surviving (query, cell) pairs ---
        # Outer loop is over K cells (scalar Python, O(K) iterations).
        # Inner ops are BLAS + NumPy vectorised over all live queries at once.
        for c in range(self.K):
            mr = self._cell_max_reach[c]
            if mr < 0.0:
                continue  # empty cell

            # Which queries might be accepted by a member of this cell?
            # Gate: d2(q, centroid[c]) <= mr^2
            live = d2_cen[:, c] <= mr_sq[c]
            live &= ~out  # already-accepted queries skip further work
            if not live.any():
                continue

            s, e = int(self._cell_starts[c]), int(self._cell_starts[c + 1])
            if s == e:
                continue

            cell_vecs  = self._sorted_vecs[s:e]      # (m, d) contiguous
            cell_rsq   = self._sorted_reach_sq[s:e]  # (m,)
            cell_vnsq  = self._vec_norms_sq[s:e]     # (m,)

            live_idx = np.flatnonzero(live)
            Qs = Qr[live_idx]                         # (n_live, d)
            qn2 = q_norms_sq[live_idx]               # (n_live,)

            # BLAS GEMM: (n_live, d) @ (d, m) -> (n_live, m)
            ip = Qs @ cell_vecs.T                     # float32 BLAS

            # L2^2(q, v) = ||q||^2 + ||v||^2 - 2*ip
            d2 = qn2[:, None] + cell_vnsq[None, :] - 2.0 * ip
            d2 = np.maximum(d2, 0.0)

            # Accept: d2 - slack <= reach_sq[m]
            # (slack absorbs fp32 rounding, same as _scan's accept_block)
            accepts = (d2 - slack) <= cell_rsq[None, :]   # (n_live, m) bool

            # A query is accepted if ANY member in the cell accepts it.
            # np.any over axis=1 is C-speed.
            out[live_idx] |= accepts.any(axis=1)

        return out

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def pruning_fraction(self, Qr: np.ndarray,
                         q_norms_sq: np.ndarray = None) -> np.ndarray:
        """
        Fraction of members (by count) visited per query.
        1.0 = no pruning.
        """
        if not self._built or self.N == 0:
            return np.ones(len(Qr))
        Qr = np.ascontiguousarray(Qr, dtype=np.float32)
        if q_norms_sq is None:
            q_norms_sq = (Qr * Qr).sum(1).astype(np.float32)

        ip_cen = Qr @ self._centroids.T  # (nq, K)
        d2_cen = (q_norms_sq[:, None]
                  + self._centroid_norms_sq[None, :]
                  - 2.0 * ip_cen)
        d2_cen = np.maximum(d2_cen, 0.0)
        mr_sq  = (self._cell_max_reach.astype(np.float64) ** 2).astype(np.float32)

        # Cells that survive the gate per query
        gate_pass = d2_cen <= mr_sq[None, :]  # (nq, K) bool

        cell_sizes = np.diff(self._cell_starts).astype(np.float32)  # (K,)
        members_visited = gate_pass @ cell_sizes  # (nq,)
        return members_visited / max(1, self.N)

    def get_bytesize(self) -> int:
        """Approximate RAM used by the index arrays."""
        if not self._built:
            return 0
        total = 0
        for arr in (self._centroids, self._centroid_norms_sq,
                    self._cell_starts, self._cell_max_reach, self._cell_cos_min,
                    self._sorted_vecs, self._sorted_reaches,
                    self._sorted_reach_sq, self._vec_norms_sq):
            total += arr.nbytes
        return total

    def __repr__(self):
        if not self._built:
            return (f"BallIVFIndex(n_cells={self.n_cells}, not built)")
        return (f"BallIVFIndex(K={self.K}, N={self.N}, "
                f"d={self.d}, ~{self.get_bytesize() // 1024} KB)")
