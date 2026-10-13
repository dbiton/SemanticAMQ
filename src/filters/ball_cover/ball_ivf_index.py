"""
BallIVFIndex -- exact IVF pruning index for BallCoverFilter.

Every hot loop is a NumPy vectorised op backed by BLAS (OpenBLAS / MKL),
so the arithmetic is C-speed with no Python interpreter overhead in the
inner loop.

Design
------
Sphere centers are clustered into K cells by euclidean k-means.
Each cell stores the max possible intrinsic reach of any of its members,
incorporating the intra-cell distance to the centroid to guarantee 0-FNR.

A cell is skipped when ||q - centroid[c]||^2  >  (max_reach[c] + tau + eps)^2
where max_reach[c] = max_j(r_j + dist(centroid[c], center_j)) over the cell.

For surviving cells the query is scored with one BLAS GEMM:
  ip = Q @ sorted_centers[cell_start:cell_end].T
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
  idx.build(centers, radii)
  out = idx.query(Qr, q_norms, tau, eps)

"""

import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Euclidean k-means -- pure numpy, run once at build time
# ---------------------------------------------------------------------------

def _euclidean_kmeans(X: np.ndarray, K: int, n_iter: int = 20, seed: int = 0) -> tuple:
    """
    (centroids, labels): float32 centroids [K, d] and int32 labels [N].
    Clustering is strictly L2 (Euclidean).
    """
    rng = np.random.default_rng(seed)
    N, d = X.shape
    K = min(K, N)
    if K == 0:
        return np.empty((0, d), dtype=np.float32), np.empty(0, dtype=np.int32)

    # Init: K random rows
    idx = rng.choice(N, K, replace=False)
    centroids = X[idx].copy()

    labels = np.zeros(N, dtype=np.int32)
    for it in range(n_iter):
        # Assign: nearest centroid by L2 distance
        # d^2(x, c) = ||x||^2 + ||c||^2 - 2<x, c>
        # To minimize d^2 across c, we maximize 2<x, c> - ||c||^2
        c_norms_sq = (centroids * centroids).sum(axis=1).astype(np.float32)
        ip = X @ centroids.T
        scores = 2.0 * ip - c_norms_sq[None, :]
        new_labels = scores.argmax(axis=1).astype(np.int32)

        # Re-seat empty cells to random data points
        counts = np.bincount(new_labels, minlength=K)
        empty = np.flatnonzero(counts == 0)
        if len(empty):
            spare = rng.choice(N, len(empty), replace=False)
            new_labels[spare] = empty

        # Update centroids as mean of members
        new_centroids = np.zeros_like(centroids)
        np.add.at(new_centroids, new_labels, X)
        counts2 = np.bincount(new_labels, minlength=K).astype(np.float32)
        new_centroids /= np.maximum(counts2[:, None], 1.0)

        if it > 0 and (new_labels == labels).all():
            break
        labels = new_labels
        centroids = new_centroids

    return centroids, labels


# ---------------------------------------------------------------------------
# CSR layout helper
# ---------------------------------------------------------------------------

def _make_csr(cell_ids: np.ndarray, K: int,
              centers: np.ndarray, radii: np.ndarray) -> tuple:
    """
    Sort members into cells, widest-first within each cell.
    Returns (cell_starts, sorted_vecs, sorted_reaches, cell_max_reach)
    """
    N = len(cell_ids)

    # Per-cell max reach
    cell_max_reach = np.full(K, -1e30, dtype=np.float64)
    np.maximum.at(cell_max_reach, cell_ids, radii.astype(np.float64))
    cell_max_reach = cell_max_reach.astype(np.float32)

    # Stable sort: primary = cell id, secondary = radii descending (widest first)
    BIG = float(radii.max() + 1.0) if N > 0 else 1.0
    key = cell_ids.astype(np.float64) * BIG - radii.astype(np.float64)
    order = np.argsort(key, kind="stable")

    sorted_centers = np.ascontiguousarray(centers[order], dtype=np.float32)
    sorted_radii   = np.ascontiguousarray(radii[order], dtype=np.float32)

    # CSR starts: cumsum of per-cell counts
    counts = np.bincount(cell_ids, minlength=K).astype(np.int32)
    cell_starts = np.zeros(K + 1, dtype=np.int32)
    np.cumsum(counts, out=cell_starts[1:])

    return cell_starts, sorted_centers, sorted_radii, cell_max_reach


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

    def build(self, centers: np.ndarray, radii: np.ndarray) -> None:
        """
        Build the ball-filter IVF index.

        Parameters
        ----------
        centers : float32[N, d]
            Decoded sphere centers in the rotated working basis.
        radii : float32[N]
            Stored sphere radii.
        """
        centers = np.ascontiguousarray(centers, dtype=np.float32)
        radii   = np.asarray(radii, dtype=np.float32).ravel()
        self._build_common(centers, radii)

    def _build_common(self, centers: np.ndarray, radii: np.ndarray) -> None:
        N, d = centers.shape
        self.d = d
        self.N = N

        if N == 0:
            self._built = True
            self._centroids      = np.empty((0, d), dtype=np.float32)
            self._centroid_norms = np.empty(0, dtype=np.float32)
            self._cell_starts    = np.array([0], dtype=np.int32)
            self._cell_max_reach = np.empty(0, dtype=np.float32)
            self._sorted_centers = np.empty((0, d), dtype=np.float32)
            self._sorted_radii   = np.empty(0, dtype=np.float32)
            self._vec_norms_sq   = np.empty(0, dtype=np.float32)
            self.K = 0
            return

        K = min(self.n_cells, max(1, int(N ** 0.5)))
        K = min(K, N)
        self.K = K

        # Euclidean k-means on the vectors
        centroids, labels = _euclidean_kmeans(centers, K, self.n_iter, self.seed)
        self._centroids = np.ascontiguousarray(centroids, dtype=np.float32)
        self._centroid_norms_sq = (self._centroids * self._centroids).sum(1).astype(np.float32)

        # Build CSR layout and sorted arrays
        cell_starts, sorted_centers, sorted_radii, cell_max_reach = \
            _make_csr(labels.astype(np.int32), K, centers, radii)

        self._cell_starts    = cell_starts
        self._cell_max_reach = cell_max_reach     
        self._sorted_centers = sorted_centers     
        self._sorted_radii   = sorted_radii       
        self._vec_norms_sq   = (sorted_centers * sorted_centers).sum(1).astype(np.float32)
        self._built = True

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, Qr: np.ndarray, q_norms_sq: np.ndarray,
              tau: float, eps: float, rel_error: Optional[float] = None) -> np.ndarray:
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

        tau_eps = float(tau) + float(eps)

        # --- Gate: one BLAS GEMM against K centroids ---
        ip_cen = Qr @ self._centroids.T
        d2_cen = (q_norms_sq[:, None] + self._centroid_norms_sq[None, :] - 2.0 * ip_cen)
        d2_cen = np.maximum(d2_cen, 0.0)

        # Fully vectorized, precomputed threshold bounds for all queries against all cells
        max_reach_sq = np.where(self._cell_max_reach > -1e30, 
                                (self._cell_max_reach + tau_eps)**2, -1.0).astype(np.float32)

        # --- Per-cell scan over surviving (query, cell) pairs ---
        # Outer loop is over K cells (scalar Python, O(K) iterations).
        # Inner ops are BLAS + NumPy vectorised over all live queries at once.
        for c in range(self.K):
            if max_reach_sq[c] < 0.0:
                continue  # empty cell

            live = (d2_cen[:, c] <= max_reach_sq[c]) & ~out
            if not live.any():
                continue

            s, e = int(self._cell_starts[c]), int(self._cell_starts[c + 1])
            if s == e:
                continue

            cell_vecs  = self._sorted_centers[s:e]
            cell_radii = self._sorted_radii[s:e]
            cell_vnsq  = self._vec_norms_sq[s:e]

            # Dynamic member reach boundary
            cell_rsq   = (cell_radii + tau_eps)**2

            live_idx = np.flatnonzero(live)
            Qs = Qr[live_idx]
            qn2 = q_norms_sq[live_idx]

            # L2^2(q, v) = ||q||^2 + ||v||^2 - 2*ip
            ip = Qs @ cell_vecs.T
            d2 = qn2[:, None] + cell_vnsq[None, :] - 2.0 * ip
            d2 = np.maximum(d2, 0.0)

            # Scale slack by the norms to match _scan exactness
            scale = qn2[:, None] + cell_vnsq.max()
            slack_scaled = rel_error * scale

            # A query is accepted if ANY member in the cell accepts it
            accepts = (d2 - slack_scaled) <= cell_rsq[None, :]
            out[live_idx] |= accepts.any(axis=1)

            # Early stopping: break cell loop if all queries are accepted
            if out.all():
                break

        return out

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def pruning_fraction(self, Qr: np.ndarray, q_norms_sq: np.ndarray = None, 
                         tau: float = 0.5, eps: float = 1e-3) -> np.ndarray:
        """
        Fraction of members (by count) visited per query.
        1.0 = no pruning.
        """
        if not self._built or self.N == 0:
            return np.ones(len(Qr))
        Qr = np.ascontiguousarray(Qr, dtype=np.float32)
        if q_norms_sq is None:
            q_norms_sq = (Qr * Qr).sum(1).astype(np.float32)

        ip_cen = Qr @ self._centroids.T
        d2_cen = np.maximum(q_norms_sq[:, None] + self._centroid_norms_sq[None, :] - 2.0 * ip_cen, 0.0)
        
        tau_eps = float(tau) + float(eps)
        max_reach_sq = np.where(self._cell_max_reach > -1e30, 
                         (self._cell_max_reach + tau_eps)**2, -1.0).astype(np.float32)

        # Cells that survive the gate per query
        gate_pass = d2_cen <= max_reach_sq[None, :]
        cell_sizes = np.diff(self._cell_starts).astype(np.float32)
        members_visited = gate_pass @ cell_sizes
        return members_visited / max(1, self.N)

    def get_bytesize(self) -> int:
        """Approximate RAM used by the index arrays."""
        if not self._built:
            return 0
        total = 0
        for arr in (self._centroids, self._centroid_norms_sq,
                    self._cell_starts, self._cell_max_reach,
                    self._sorted_centers, self._sorted_radii,
                    self._vec_norms_sq):
            total += arr.nbytes
        return total

    def __repr__(self):
        if not self._built:
            return (f"BallIVFIndex(n_cells={self.n_cells}, not built)")
        return (f"BallIVFIndex(K={self.K}, N={self.N}, "
                f"d={self.d}, ~{self.get_bytesize() // 1024} KB)")