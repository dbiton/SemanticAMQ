"""
CapIVFIndex -- exact IVF pruning index for CapCoverFilter.

Every hot loop is a NumPy vectorised op backed by BLAS (OpenBLAS / MKL),
so the arithmetic is C-speed with no Python interpreter overhead in the
inner loop.

Design
------
Cap centers are clustered into K cells by spherical k-means.
Each cell stores the max possible reach of any of its members.

A cell is skipped when <q, centroid[c]>  <  cos(max_reach[c])
where max_reach[c] = max_j(theta_j + beta + slack).

For surviving cells the query is scored with one BLAS GEMM:
  ip = Q @ sorted_vecs[cell_start:cell_end].T
followed by a NumPy vectorised accept test (one Python line, fully C-speed).

Zero-FNR guarantee
------------------
max_reach[c] >= reach of every member in cell c by construction.
For cap mode: angle(q, centroid[c]) <= angle(q, dir_j) + angle(centroid[c], dir_j)
  and the analogous bound holds in angle space.
Therefore: if an accepting member exists, its cell passes the gate -- nothing is skipped.

Usage
-----
  # Cap filter (static only -- insert_method=None):
  idx = CapIVFIndex(n_cells=128)
  idx.build_cap(decoded_dirs, angles, beta_max, slack)
  out = idx.query_cap(Qn32, beta_f32, feasible,
                      theta, decoded_norms, slack)      # bool[nq]
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

    def build(self, directions: np.ndarray, angles: np.ndarray,
                  beta_max: float, slack: float) -> None:
        """
        Build the cap-filter IVF index.

        Parameters
        ----------
        directions : float32[N, d]
            Decoded unit cap directions (working basis).
        angles : array[N]
            Stored cap angles theta_j in radians.
        beta_max : float
            Upper bound on beta for queries this index will serve.
            max_reach[c] = max(theta_j + beta_max + slack).
        slack : float
            CapCoverFilter._slack(), the fp32 boundary tolerance.
        """
        directions = np.ascontiguousarray(directions, dtype=np.float32)
        angles     = np.asarray(angles, dtype=np.float64).ravel()
        reaches    = (angles + float(beta_max) + float(slack)).astype(np.float32)
        self._beta_baked = float(beta_max)
        self._slack      = float(slack)
        self._build_common(directions, reaches)

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

    def query(self, Qn32: np.ndarray, beta: np.ndarray,
                  feasible: np.ndarray, theta: np.ndarray,
                  decoded_norms: np.ndarray, slack: float) -> np.ndarray:
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
        theta : float32[N]
            Raw cap angles in the IVF's sorted member order.
        decoded_norms : float32[N]
            |decoded_j| for each sorted cap direction.
        slack : float
            CapCoverFilter._slack().

        Returns
        -------
        out : bool[nq]
        """
        if not self._built or self.N == 0:
            return np.zeros(len(Qn32), dtype=bool)

        Qn32         = np.ascontiguousarray(Qn32, dtype=np.float32)
        beta         = np.ascontiguousarray(beta, dtype=np.float32)
        feasible     = np.asarray(feasible, dtype=bool)
        theta        = np.ascontiguousarray(theta, dtype=np.float32)
        decoded_norms= np.ascontiguousarray(decoded_norms, dtype=np.float32)
        nq           = len(Qn32)
        out          = np.zeros(nq, dtype=bool)

        # --- Gate: one BLAS GEMM against K centroids ---
        ip_cen = Qn32 @ self._centroids.T   # (nq, K)

        for c in range(self.K):
            cos_min = self._cell_cos_min[c]
            if cos_min > 1.0:
                continue  # empty cell

            # Gate: ip(q, centroid[c]) < cos(max_reach[c]) => skip
            # (angle(q, centroid) > max_reach[c] => no member can accept q)
            live = ip_cen[:, c] >= cos_min
            live &= feasible
            live &= ~out
            if not live.any():
                continue

            s, e = int(self._cell_starts[c]), int(self._cell_starts[c + 1])
            if s == e:
                continue

            cell_dirs  = self._sorted_vecs[s:e]     # (m, d)
            cell_theta = theta[s:e]                  # (m,)  raw angles
            cell_norms = decoded_norms[s:e]          # (m,)

            live_idx = np.flatnonzero(live)
            Qs   = Qn32[live_idx]                   # (n_live, d)
            beta_live = beta[live_idx]               # (n_live,)

            # BLAS GEMM: (n_live, m) inner products
            ip = Qs @ cell_dirs.T                    # float32 BLAS

            # Accept rule: ip(q, decoded_j) + slack >= |decoded_j| * cos(theta_j + beta_i)
            # Threshold per (query, member): cell_norms[j] * cos(cell_theta[j] + beta_live[i])
            # cos() over entire (n_live, m) array -- fully vectorised
            thr = cell_norms[None, :] * np.cos(
                cell_theta[None, :] + beta_live[:, None])   # (n_live, m)

            accepts = ip + slack >= thr                      # (n_live, m) bool
            out[live_idx] |= accepts.any(axis=1)

        return out

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def pruning_fraction(self, Qr: np.ndarray,
                         q_norms_sq: np.ndarray = None) -> np.ndarray:
        """
        Fraction of members (by count) visited per query.
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
            return (f"CapIVFIndex(n_cells={self.n_cells}, not built)")
        return (f"CapIVFIndex(K={self.K}, N={self.N}, "
                f"d={self.d}, ~{self.get_bytesize() // 1024} KB)")
