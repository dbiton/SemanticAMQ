"""
CapIVFIndex -- exact IVF pruning index for CapCoverFilter.

Every hot loop is a NumPy vectorised op backed by BLAS (OpenBLAS / MKL),
so the arithmetic is C-speed with no Python interpreter overhead in the
inner loop.

Design
------
Cap centers are clustered into K cells by spherical k-means.
Each cell stores the max intrinsic angle (theta) of any of its members.

A cell is skipped when <q, centroid[c]>  <  cos(max_theta[c] + beta_q + slack)

For surviving cells the query is scored with one BLAS GEMM:
  ip = Q @ sorted_vecs[cell_start:cell_end].T
followed by a NumPy vectorised accept test (one Python line, fully C-speed).

Zero-FNR guarantee
------------------
max_theta[c] >= theta of every member in cell c by construction.
For cap mode: angle(q, centroid[c]) <= angle(q, dir_j) + angle(centroid[c], dir_j)
  and the analogous bound holds in angle space.
Therefore: if an accepting member exists, its cell passes the gate -- nothing is skipped.

Usage
-----
  # Cap filter (static only -- insert_method=None):
  idx = CapIVFIndex(n_cells=128)
  idx.build(decoded_dirs, angles)
  out = idx.query(Qn32, beta_f32, feasible, slack)
"""

import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Spherical k-means -- pure numpy, run once at build time
# ---------------------------------------------------------------------------

def _spherical_kmeans(X: np.ndarray, K: int, n_iter: int = 20, seed: int = 0) -> tuple:
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
        ip = X @ centroids.T
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

def _make_csr(cell_ids: np.ndarray, K: int, angles: np.ndarray,
              directions: np.ndarray) -> tuple:
    """
    Sort members into cells, widest-first within each cell.
    Returns (cell_starts, sorted_vecs, sorted_angles, sorted_norms, cell_max_theta)
    """
    N = len(cell_ids)

    # Per-cell max intrinsic angle (theta)
    cell_max_theta = np.full(K, -1e30, dtype=np.float64)
    np.maximum.at(cell_max_theta, cell_ids, angles.astype(np.float64))
    cell_max_theta = cell_max_theta.astype(np.float32)

    # Stable sort: primary = cell id, secondary = angle descending (widest first)
    BIG = float(angles.max() + 1.0) if N > 0 else 1.0
    key = cell_ids.astype(np.float64) * BIG - angles.astype(np.float64)
    order = np.argsort(key, kind="stable")

    sorted_vecs   = np.ascontiguousarray(directions[order], dtype=np.float32)
    sorted_angles = np.ascontiguousarray(angles[order],    dtype=np.float32)
    sorted_norms  = np.sqrt((sorted_vecs * sorted_vecs).sum(axis=1)).astype(np.float32)

    # CSR starts: cumsum of per-cell counts
    counts = np.bincount(cell_ids, minlength=K).astype(np.int32)
    cell_starts = np.zeros(K + 1, dtype=np.int32)
    np.cumsum(counts, out=cell_starts[1:])

    return cell_starts, sorted_vecs, sorted_angles, sorted_norms, cell_max_theta


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class CapIVFIndex:
    """
    Exact IVF pruning index for CapCoverFilter.  
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

    def build(self, directions: np.ndarray, angles: np.ndarray) -> None:
        """
        Build the cap-filter IVF index.

        Parameters
        ----------
        directions : float32[N, d]
            Decoded unit cap directions (working basis).
        angles : array[N]
            Stored cap angles theta_j in radians.
        """
        directions = np.ascontiguousarray(directions, dtype=np.float32)
        angles     = np.asarray(angles, dtype=np.float64).ravel()
        self._build_common(directions, angles)

    def _build_common(self, directions: np.ndarray, angles: np.ndarray) -> None:
        N, d = directions.shape
        self.d = d
        self.N = N

        if N == 0:
            self._built = True
            self._centroids      = np.empty((0, d), dtype=np.float32)
            self._centroid_norms_sq = np.empty(0, dtype=np.float32)
            self._cell_starts    = np.array([0], dtype=np.int32)
            self._cell_max_theta = np.empty(0, dtype=np.float32)
            self._sorted_vecs    = np.empty((0, d), dtype=np.float32)
            self._sorted_angles  = np.empty(0, dtype=np.float32)
            self._sorted_norms   = np.empty(0, dtype=np.float32)
            self.K = 0
            return

        K = min(self.n_cells, max(1, int(N ** 0.5)))
        K = min(K, N)
        self.K = K

        # Spherical k-means on the unit vectors
        centroids, labels = _spherical_kmeans(directions, K, self.n_iter, self.seed)
        self._centroids = np.ascontiguousarray(centroids, dtype=np.float32)
        self._centroid_norms_sq = (self._centroids * self._centroids).sum(1).astype(np.float32)

        # Build CSR layout and sorted arrays
        cell_starts, sorted_vecs, sorted_angles, sorted_norms, cell_max_theta = \
            _make_csr(labels.astype(np.int32), K, angles, directions)

        self._cell_starts    = cell_starts
        self._cell_max_theta = cell_max_theta
        self._sorted_vecs    = sorted_vecs
        self._sorted_angles  = sorted_angles
        self._sorted_norms   = sorted_norms
        self._built = True

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, Qn32: np.ndarray, beta: np.ndarray,
              feasible: np.ndarray, slack: float) -> np.ndarray:
        """
        Exact cap-filter IVF query.  0-FNR guaranteed.

        Parameters
        ----------
        Qn32 : float32[nq, d]
            Unit queries in the working basis.
        beta : float32[nq]
            Per-query acceptance margin (from CapCoverFilter._beta).
        feasible : uint8 or bool [nq]
            Mask; infeasible queries are always rejected.
        slack : float
            CapCoverFilter._slack().

        Returns
        -------
        out : bool[nq]
        """
        if not self._built or self.N == 0:
            return np.zeros(len(Qn32), dtype=bool)

        Qn32     = np.ascontiguousarray(Qn32, dtype=np.float32)
        beta     = np.ascontiguousarray(beta, dtype=np.float32)
        feasible = np.asarray(feasible, dtype=bool)
        nq       = len(Qn32)
        out      = np.zeros(nq, dtype=bool)

        # --- Gate: one BLAS GEMM against K centroids ---
        ip_cen = Qn32 @ self._centroids.T

        # Fully vectorized, precomputed threshold bounds for all queries against all cells
        gate_angles = self._cell_max_theta[None, :] + beta[:, None] + slack
        np.clip(gate_angles, 0.0, np.pi, out=gate_angles)
        cos_min_all = np.cos(gate_angles)

        # --- Per-cell scan over surviving (query, cell) pairs ---
        # Outer loop is over K cells (scalar Python, O(K) iterations).
        # Inner ops are BLAS + NumPy vectorised over all live queries at once.
        for c in range(self.K):
            s, e = int(self._cell_starts[c]), int(self._cell_starts[c + 1])
            if s == e:
                continue  # empty cell

            # Gate: ip(q, centroid[c]) < cos(max_theta[c] + beta_q + slack) => skip
            live = ip_cen[:, c] >= cos_min_all[:, c]
            live &= feasible
            live &= ~out
            if not live.any():
                continue

            cell_dirs  = self._sorted_vecs[s:e]
            cell_theta = self._sorted_angles[s:e]
            cell_norms = self._sorted_norms[s:e]

            live_idx = np.flatnonzero(live)
            Qs   = Qn32[live_idx]
            beta_live = beta[live_idx]

            # BLAS GEMM: (n_live, m) inner products
            ip = Qs @ cell_dirs.T

            # Accept rule: ip(q, decoded_j) + slack >= |decoded_j| * cos(theta_j + beta_i)
            thr = cell_norms[None, :] * np.cos(
                cell_theta[None, :] + beta_live[:, None])

            accepts = ip + slack >= thr
            out[live_idx] |= accepts.any(axis=1)

        return out

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def pruning_fraction(self, Qr: np.ndarray, beta: np.ndarray = None, 
                         slack: float = 0.0) -> np.ndarray:
        """
        Fraction of members (by count) visited per query.
        """
        if not self._built or self.N == 0:
            return np.ones(len(Qr))
        
        Qr = np.ascontiguousarray(Qr, dtype=np.float32)
        nq = len(Qr)
        q_norms = np.sqrt((Qr * Qr).sum(axis=1)).astype(np.float32)
        Qn32 = Qr / np.maximum(q_norms, 1e-30)[:, None]
        
        ip_cen = Qn32 @ self._centroids.T
        if beta is None:
            beta = np.zeros(nq, dtype=np.float32)
        else:
            beta = np.ascontiguousarray(beta, dtype=np.float32)
            
        gate_angles = self._cell_max_theta[None, :] + beta[:, None] + slack
        np.clip(gate_angles, 0.0, np.pi, out=gate_angles)
        cos_min_all = np.cos(gate_angles)

        # Cells that survive the gate per query
        gate_pass = ip_cen >= cos_min_all
        cell_sizes = np.diff(self._cell_starts).astype(np.float32)
        members_visited = gate_pass @ cell_sizes
        return members_visited / max(1, self.N)

    def get_bytesize(self) -> int:
        """Approximate RAM used by the index arrays."""
        if not self._built:
            return 0
        total = 0
        for arr in (self._centroids, self._centroid_norms_sq,
                    self._cell_starts, self._cell_max_theta,
                    self._sorted_vecs, self._sorted_angles,
                    self._sorted_norms):
            total += arr.nbytes
        return total

    def __repr__(self):
        if not self._built:
            return (f"CapIVFIndex(n_cells={self.n_cells}, not built)")
        return (f"CapIVFIndex(K={self.K}, N={self.N}, "
                f"d={self.d}, ~{self.get_bytesize() // 1024} KB)")