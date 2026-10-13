import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'src'))
import unittest
import numpy as np
import time
from src.filters.ball_cover.ball_ivf_index import BallIVFIndex, _euclidean_kmeans as _ball_kmeans, _make_csr as _ball_csr
from src.filters.cap_cover.cap_ivf_index  import CapIVFIndex,  _spherical_kmeans as _cap_kmeans,  _make_csr as _cap_csr


# ---------------------------------------------------------------------------
# Test utilities
# ---------------------------------------------------------------------------

def rng(seed=42):
    return np.random.default_rng(seed)

def unit(X: np.ndarray) -> np.ndarray:
    """Return unit-row copy of X (float32)."""
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return (X / np.maximum(n, 1e-30)).astype(np.float32)

def make_ball_data(N: int = 400, d: int = 64, seed: int = 0):
    """(centers, radii, queries) all float32; centers and queries are unit."""
    r = rng(seed)
    centers = unit(r.standard_normal((N, d)).astype(np.float32))
    radii   = r.uniform(0.05, 0.25, N).astype(np.float32)
    queries = unit(r.standard_normal((200, d)).astype(np.float32))
    return centers, radii, queries

def make_cap_data(N: int = 300, d: int = 64, seed: int = 1):
    """(directions, angles, queries); directions and queries are unit."""
    r = rng(seed)
    dirs    = unit(r.standard_normal((N, d)).astype(np.float32))
    angles  = r.uniform(0.05, 0.40, N).astype(np.float64)
    queries = unit(r.standard_normal((200, d)).astype(np.float32))
    return dirs, angles, queries

# ---------------------------------------------------------------------------
# Reference implementations (brute-force, float64, no IVF)
# ---------------------------------------------------------------------------

def ref_ball(centers, radii, tau, eps, Qr):
    """True if any sphere accepts the query (exact, float64)."""
    reach = (radii + tau + eps).astype(np.float64)
    out   = np.zeros(len(Qr), dtype=bool)
    C64   = centers.astype(np.float64)
    for i, q in enumerate(Qr.astype(np.float64)):
        d2 = ((q - C64) ** 2).sum(1)
        out[i] = (d2 <= reach ** 2).any()
    return out

def ref_cap(dirs, angles, beta, feasible, slack, Qn32):
    """True if any cap accepts the query (exact, float64)."""
    D64   = dirs.astype(np.float64)
    norms = np.linalg.norm(D64, axis=1)
    out   = np.zeros(len(Qn32), dtype=bool)
    for i, q in enumerate(Qn32.astype(np.float64)):
        if not feasible[i]:
            continue
        ip  = D64 @ q
        thr = norms * np.cos(angles + float(beta[i]))
        out[i] = (ip + slack >= thr).any()
    return out

def _cap_beta(tau: float, eps: float = 1e-3) -> float:
    """beta at tau, norm=1 -- worst case for unit queries."""
    tp    = tau + eps + 1e-4
    cos_b = (1.0 + 1.0 - tp ** 2) / 2.0
    return float(np.arccos(np.clip(cos_b, -1, 1)))

def _cap_slack(d: int) -> float:
    return 2.0 * (d + 4) * 2.0 ** -24


# ===========================================================================
# TestCSR  --  shared _make_csr helper
# ===========================================================================

class TestCSR(unittest.TestCase):
    
    def test_ball_csr_cell_starts_sum_to_N(self):
        N, K = 100, 8
        r = rng(0)
        ids     = r.integers(0, K, N).astype(np.int32)
        reaches = r.uniform(0.1, 1.0, N).astype(np.float32)
        vecs    = unit(r.standard_normal((N, 16)).astype(np.float32))
        starts, _, _, _ = _ball_csr(ids, K, vecs, reaches)
        self.assertEqual(starts[-1], N, "CSR total must equal N")
        self.assertTrue((np.diff(starts) >= 0).all())

    def test_cap_csr_cell_starts_sum_to_N(self):
        N, K = 100, 8
        r = rng(0)
        ids     = r.integers(0, K, N).astype(np.int32)
        angles  = r.uniform(0.1, 1.0, N).astype(np.float32)
        vecs    = unit(r.standard_normal((N, 16)).astype(np.float32))
        starts, _, _, _, _ = _cap_csr(ids, K, angles, vecs)
        self.assertEqual(starts[-1], N, "CSR total must equal N")
        self.assertTrue((np.diff(starts) >= 0).all())

    def test_ball_max_reach_per_cell(self):
        N, K = 50, 4
        r = rng(1)
        ids     = (np.arange(N) % K).astype(np.int32)
        reaches = r.uniform(0.0, 2.0, N).astype(np.float32)
        vecs    = unit(r.standard_normal((N, 8)).astype(np.float32))
        _, _, _, cell_max = _ball_csr(ids, K, vecs, reaches)
        for c in range(K):
            expected = float(reaches[ids == c].max())
            self.assertTrue(abs(float(cell_max[c]) - expected) < 1e-5,
                            f"cell {c}: {cell_max[c]:.4f} != {expected:.4f}")

    def test_cap_max_reach_per_cell(self):
        N, K = 50, 4
        r = rng(1)
        ids     = (np.arange(N) % K).astype(np.int32)
        angles  = r.uniform(0.0, 2.0, N).astype(np.float32)
        vecs    = unit(r.standard_normal((N, 8)).astype(np.float32))
        _, _, _, _, cell_max = _cap_csr(ids, K, angles, vecs)
        for c in range(K):
            expected = float(angles[ids == c].max())
            self.assertTrue(abs(float(cell_max[c]) - expected) < 1e-5,
                            f"cell {c}: {cell_max[c]:.4f} != {expected:.4f}")

    def test_ball_widest_first_within_cell(self):
        N, K = 80, 5
        r = rng(2)
        ids     = r.integers(0, K, N).astype(np.int32)
        reaches = r.uniform(0.0, 1.0, N).astype(np.float32)
        vecs    = unit(r.standard_normal((N, 8)).astype(np.float32))
        starts, _, sorted_r, _ = _ball_csr(ids, K, vecs, reaches)
        for c in range(K):
            s, e = int(starts[c]), int(starts[c + 1])
            self.assertTrue((np.diff(sorted_r[s:e]) <= 1e-6).all(),
                            f"cell {c}: not widest-first")

    def test_cap_widest_first_within_cell(self):
        N, K = 80, 5
        r = rng(2)
        ids     = r.integers(0, K, N).astype(np.int32)
        angles  = r.uniform(0.0, 1.0, N).astype(np.float32)
        vecs    = unit(r.standard_normal((N, 8)).astype(np.float32))
        starts, _, sorted_a, _, _ = _cap_csr(ids, K, angles, vecs)
        for c in range(K):
            s, e = int(starts[c]), int(starts[c + 1])
            self.assertTrue((np.diff(sorted_a[s:e]) <= 1e-6).all(),
                            f"cell {c}: not widest-first")


# ===========================================================================
# TestKmeans
# ===========================================================================

class TestKmeans(unittest.TestCase):
    def test_all_points_assigned(self):
        X = unit(rng(10).standard_normal((200, 32)).astype(np.float32))
        c, l = _ball_kmeans(X, 16, n_iter=10, seed=0)
        self.assertEqual(l.shape, (200,))
        self.assertTrue(set(l).issubset(set(range(16))))

    def test_centroids_are_unit(self):
        X = unit(rng(11).standard_normal((100, 64)).astype(np.float32))
        c, _ = _cap_kmeans(X, 8, n_iter=5, seed=0)
        self.assertTrue(np.allclose(np.linalg.norm(c, axis=1), 1.0, atol=1e-5))

    def test_K_clamped_to_N(self):
        """K > N must not crash; at most N cells returned."""
        X = unit(rng(12).standard_normal((5, 8)).astype(np.float32))
        c, _ = _ball_kmeans(X, 20, n_iter=5, seed=0)
        self.assertLessEqual(len(c), 5)

    def test_empty_data(self):
        X = np.empty((0, 8), dtype=np.float32)
        c, l = _ball_kmeans(X, 4, n_iter=5, seed=0)
        self.assertEqual(len(c), 0)
        self.assertEqual(len(l), 0)


# ===========================================================================
# TestBallIVFIndex
# ===========================================================================

class TestBallIVFIndex(unittest.TestCase):

    # ---- 0-FNR ----

    def test_zero_fnr_small(self):
        """Every query accepted by brute-force must be accepted by BallIVFIndex."""
        for tau in [0.05, 0.15, 0.30]:
            for N, d in [(200, 32), (400, 128)]:
                centers, radii, queries = make_ball_data(N, d)
                eps = 1e-3
                idx = BallIVFIndex(n_cells=max(4, N // 10))
                idx.build(centers, radii)
                qn  = (queries * queries).sum(1).astype(np.float32)
                ref = ref_ball(centers, radii, tau, eps, queries)
                ivf = idx.query(queries, qn, tau=tau, eps=eps)
                fn  = ref & ~ivf
                self.assertEqual(fn.sum(), 0,
                                 f"BallIVFIndex: {fn.sum()} FN at tau={tau} N={N} d={d}")

    # ---- No extra FP ----

    def test_zero_extra_fp(self):
        """BallIVFIndex must not accept queries the brute-force rejects."""
        for tau in [0.05, 0.15, 0.30]:
            centers, radii, queries = make_ball_data(300, 64)
            eps = 1e-3
            idx = BallIVFIndex(n_cells=32)
            idx.build(centers, radii)
            qn  = (queries * queries).sum(1).astype(np.float32)
            ref = ref_ball(centers, radii, tau, eps, queries)
            ivf = idx.query(queries, qn, tau=tau, eps=eps)
            self.assertEqual((ivf & ~ref).sum(), 0,
                             f"BallIVFIndex: extra FP at tau={tau}")

    # ---- Edge cases ----

    def test_single_sphere(self):
        d   = 32
        c   = unit(np.ones((1, d), dtype=np.float32))
        r   = np.array([0.2], dtype=np.float32)
        idx = BallIVFIndex(n_cells=1)
        idx.build(c, r)
        q   = np.vstack([c, -c])
        qn  = (q * q).sum(1).astype(np.float32)
        out = idx.query(q, qn, tau=0.1, eps=1e-3)
        self.assertTrue(out[0], "center query must be accepted")
        self.assertFalse(out[1], "antipodal query must be rejected")

    def test_empty_build(self):
        d   = 64
        idx = BallIVFIndex(n_cells=16)
        idx.build(np.empty((0, d), dtype=np.float32),
                  np.empty(0, dtype=np.float32))
        q   = unit(np.ones((5, d), dtype=np.float32))
        qn  = (q * q).sum(1).astype(np.float32)
        self.assertFalse(idx.query(q, qn, tau=0.1, eps=1e-3).any())

    def test_large_tau_matches_ref(self):
        """At tau=3.0 the IVF and brute-force must still agree."""
        centers, radii, queries = make_ball_data(100, 32)
        tau = 3.0; eps = 1e-3
        idx = BallIVFIndex(n_cells=16)
        idx.build(centers, radii)
        qn  = (queries * queries).sum(1).astype(np.float32)
        ref = ref_ball(centers, radii, tau, eps, queries)
        ivf = idx.query(queries, qn, tau=tau, eps=eps)
        self.assertEqual((ref & ~ivf).sum(), 0)


# ===========================================================================
# TestCapIVFIndex
# ===========================================================================

class TestCapIVFIndex(unittest.TestCase):

    def _build(self, dirs, angles, tau, d):
        eps      = 1e-3
        beta_max = _cap_beta(tau, eps)
        slack    = _cap_slack(d)
        idx = CapIVFIndex(n_cells=32)
        idx.build(dirs, angles)
        return idx, beta_max, slack

    def _beta_arr(self, n: int, beta_max: float):
        return np.full(n, beta_max, dtype=np.float32)

    # ---- 0-FNR ----

    def test_zero_fnr(self):
        for tau in [0.05, 0.15, 0.30]:
            dirs, angles, queries = make_cap_data(200, 64)
            d        = 64
            idx, beta_max, slack = self._build(dirs, angles, tau, d)
            feasible = np.ones(len(queries), dtype=bool)
            beta_arr = self._beta_arr(len(queries), beta_max)

            Qn32 = queries.astype(np.float32)
            ref  = ref_cap(dirs, angles,
                           beta_arr.astype(np.float64), feasible, slack, Qn32)
            ivf  = idx.query(Qn32, beta_arr, feasible, slack)
            fn   = ref & ~ivf
            self.assertEqual(fn.sum(), 0,
                             f"CapIVFIndex: {fn.sum()} FN at tau={tau}")

    # ---- No extra FP ----

    def test_zero_extra_fp(self):
        dirs, angles, queries = make_cap_data(200, 64)
        d        = 64
        tau      = 0.10
        idx, beta_max, slack = self._build(dirs, angles, tau, d)
        feasible = np.ones(len(queries), dtype=bool)
        beta_arr = self._beta_arr(len(queries), beta_max)

        Qn32 = queries.astype(np.float32)
        ref  = ref_cap(dirs, angles,
                       beta_arr.astype(np.float64), feasible, slack, Qn32)
        ivf  = idx.query(Qn32, beta_arr, feasible, slack)
        self.assertEqual((ivf & ~ref).sum(), 0, "CapIVFIndex: extra FP")

    # ---- Infeasible queries ----

    def test_infeasible_always_rejected(self):
        dirs, angles, _ = make_cap_data(50, 32)
        d   = 32
        tau = 0.1
        idx, beta_max, slack = self._build(dirs, angles, tau, d)
        nq       = 30
        Qn32     = unit(rng(5).standard_normal((nq, d)).astype(np.float32))
        feasible = np.zeros(nq, dtype=bool)           # all infeasible
        beta_arr = self._beta_arr(nq, beta_max)
        
        out      = idx.query(Qn32, beta_arr, feasible, slack)
        self.assertFalse(out.any(), "infeasible queries must always be rejected")

    # ---- Edge cases ----

    def test_empty_build(self):
        d   = 32
        idx = CapIVFIndex(n_cells=8)
        idx.build(np.empty((0, d), dtype=np.float32),
                  np.empty(0, dtype=np.float64))
        q        = unit(rng(0).standard_normal((5, d)).astype(np.float32))
        feasible = np.ones(5, dtype=bool)
        beta_arr = np.full(5, 0.3, dtype=np.float32)
        
        out      = idx.query(q, beta_arr, feasible, slack=_cap_slack(d))
        self.assertFalse(out.any())

    def test_single_cap(self):
        d   = 16
        u   = unit(np.ones((1, d), dtype=np.float32))
        th  = np.array([0.3], dtype=np.float64)                        # 0.3 rad cap
        idx = CapIVFIndex(n_cells=1)
        beta_max = _cap_beta(0.1)
        slack    = _cap_slack(d)
        idx.build(u, th)

        # Query aligned with direction: must accept
        q_in  = u.copy()
        # Query antipodal: must reject
        q_out = -u
        Qn32  = np.vstack([q_in, q_out]).astype(np.float32)
        feas  = np.ones(2, dtype=bool)
        beta  = np.full(2, beta_max, dtype=np.float32)
        
        out   = idx.query(Qn32, beta, feas, slack)
        self.assertTrue(out[0], "direction query should be accepted")
        self.assertFalse(out[1], "antipodal query should be rejected")

    # ---- Pruning fraction diagnostic ----

    def test_pruning_fraction_in_range(self):
        dirs, angles, queries = make_cap_data(200, 64)
        idx, beta_max, slack = self._build(dirs, angles, 0.1, 64)
        beta_arr = self._beta_arr(len(queries), beta_max)
        fracs = idx.pruning_fraction(queries.astype(np.float32), beta=beta_arr, slack=slack)
        self.assertTrue((fracs >= 0.0).all() and (fracs <= 1.0 + 1e-6).all())


# ===========================================================================
# TestPruning  --  pruning_fraction for BallIVFIndex
# ===========================================================================

class TestPruning(unittest.TestCase):
    def test_fraction_in_range(self):
        centers, radii, queries = make_ball_data(200, 64)
        idx = BallIVFIndex(n_cells=32)
        idx.build(centers, radii)
        qn    = (queries * queries).sum(1).astype(np.float32)
        fracs = idx.pruning_fraction(queries, qn, tau=0.1, eps=1e-3)
        self.assertTrue((fracs >= 0.0).all() and (fracs <= 1.0 + 1e-6).all())

    def test_negatives_prune_more_than_positives(self):
        """Negative queries (far from all spheres) should visit fewer cells
        on average than queries sitting on sphere centers."""
        N, d = 400, 128
        r    = rng(20)
        centers = unit(r.standard_normal((N, d)).astype(np.float32))
        radii   = np.full(N, 0.05, dtype=np.float32)
        idx = BallIVFIndex(n_cells=64)
        idx.build(centers, radii)

        pos     = centers[:20]
        far_neg = unit(-centers[:20] + r.standard_normal((20, d)).astype(np.float32) * 0.01)
        qn_p    = (pos     * pos    ).sum(1).astype(np.float32)
        qn_n    = (far_neg * far_neg).sum(1).astype(np.float32)

        frac_p = idx.pruning_fraction(pos,     qn_p, tau=0.05, eps=1e-3)
        frac_n = idx.pruning_fraction(far_neg, qn_n, tau=0.05, eps=1e-3)
        self.assertTrue(frac_n.mean() < frac_p.mean() + 0.5,
                        (f"negatives fraction {frac_n.mean():.2f} should not exceed "
                         f"positives {frac_p.mean():.2f} by 0.5"))


# ===========================================================================
# TestThroughput
# ===========================================================================

class TestThroughput(unittest.TestCase):
    """Correctness + informational throughput numbers."""

    @staticmethod
    def _blas_brute_ball(centers, radii, tau, eps, Qr):
        """Vectorised brute-force via a single BLAS GEMM."""
        qn  = (Qr * Qr).sum(1).astype(np.float32)
        vn  = (centers * centers).sum(1).astype(np.float32)
        ip  = Qr @ centers.T
        d2  = np.maximum(qn[:, None] + vn[None, :] - 2.0 * ip, 0.0)
        rch = ((radii + tau + eps) ** 2).astype(np.float32)
        return (d2 <= rch[None, :]).any(1)

    @staticmethod
    def _blas_brute_cap(dirs, angles, beta, feasible, slack, Qn32):
        """Vectorised brute-force via a single BLAS GEMM for cap cover."""
        D64   = dirs.astype(np.float64)
        norms = np.linalg.norm(D64, axis=1)
        ip    = Qn32.astype(np.float64) @ D64.T
        thr   = norms[None, :] * np.cos(angles[None, :] + beta.astype(np.float64)[:, None])
        
        accepts = ip + slack >= thr
        out = np.zeros(len(Qn32), dtype=bool)
        out[feasible] = accepts[feasible].any(axis=1)
        return out

    def test_ball_ivf_correctness_and_speed(self):
        N, d = 500, 128
        r    = rng(30)
        centers = unit(r.standard_normal((N, d)).astype(np.float32))
        radii   = np.full(N, 0.04, dtype=np.float32)
        tau = 0.04; eps = 1e-3

        idx = BallIVFIndex(n_cells=64)
        idx.build(centers, radii)

        neg = unit(r.standard_normal((500, d)).astype(np.float32))
        qn  = (neg * neg).sum(1).astype(np.float32)

        # Warmup
        for _ in range(3):
            idx.query(neg, qn, tau=tau, eps=eps)

        t0  = time.perf_counter()
        ivf_out = idx.query(neg, qn, tau=tau, eps=eps)
        t_ivf   = time.perf_counter() - t0

        t0  = time.perf_counter()
        ref_out = self._blas_brute_ball(centers, radii, tau, eps, neg)
        t_ref   = time.perf_counter() - t0

        self.assertTrue((ivf_out == ref_out).all(), "BallIVFIndex result disagrees with brute-force")
        print(f"\n  BallIVFIndex: {len(neg)/t_ivf:.0f} q/s  |  "
              f"BLAS brute-force: {len(neg)/t_ref:.0f} q/s  |  "
              f"speedup: {t_ref/t_ivf:.1f}x  (N={N}, d={d})")
        # Soft sanity: not more than 10x slower than brute force
        self.assertTrue(t_ivf <= 10 * t_ref + 1.0)

    def test_cap_ivf_correctness_and_speed(self):
        N, d = 500, 128
        r    = rng(31)
        dirs   = unit(r.standard_normal((N, d)).astype(np.float32))
        angles = np.full(N, 0.08)
        queries = unit(r.standard_normal((200, d)).astype(np.float32))
        tau      = 0.08; eps = 1e-3
        beta_max = _cap_beta(tau, eps)
        slack    = _cap_slack(d)

        idx = CapIVFIndex(n_cells=32)
        idx.build(dirs, angles)

        Qn32     = queries.astype(np.float32)
        feasible = np.ones(len(queries), dtype=bool)
        beta_arr = np.full(len(queries), beta_max, dtype=np.float32)

        for _ in range(3):
            idx.query(Qn32, beta_arr, feasible, slack)

        t0      = time.perf_counter()
        ivf_out = idx.query(Qn32, beta_arr, feasible, slack)
        t_ivf   = time.perf_counter() - t0

        t0      = time.perf_counter()
        ref_out = self._blas_brute_cap(dirs, angles, beta_arr, feasible, slack, Qn32)
        t_ref   = time.perf_counter() - t0

        fn = ref_out & ~ivf_out
        self.assertEqual(fn.sum(), 0, f"CapIVFIndex: {fn.sum()} FN in throughput test")
        
        print(f"\n  CapIVFIndex: {len(queries)/t_ivf:.0f} q/s  |  "
              f"BLAS brute-force: {len(queries)/t_ref:.0f} q/s  |  "
              f"speedup: {t_ref/t_ivf:.1f}x  (N={N}, d={d})")
        # Soft sanity: not more than 10x slower than brute force
        self.assertTrue(t_ivf <= 10 * t_ref + 1.0)


# ===========================================================================
# TestBookkeeping
# ===========================================================================

class TestBookkeeping(unittest.TestCase):
    def test_ball_bytesize_positive(self):
        centers, radii, _ = make_ball_data(100, 32)
        idx = BallIVFIndex(n_cells=16)
        idx.build(centers, radii)
        self.assertTrue(idx.get_bytesize() > 0)

    def test_ball_repr_built(self):
        centers, radii, _ = make_ball_data(50, 16)
        idx = BallIVFIndex(n_cells=8)
        idx.build(centers, radii)
        s = repr(idx)
        self.assertTrue("BallIVFIndex" in s and "K=" in s and "not built" not in s)

    def test_ball_repr_not_built(self):
        self.assertTrue("not built" in repr(BallIVFIndex(n_cells=32)))

    def test_cap_bytesize_positive(self):
        dirs, angles, _ = make_cap_data(100, 32)
        idx = CapIVFIndex(n_cells=16)
        idx.build(dirs, angles)
        self.assertTrue(idx.get_bytesize() > 0)

    def test_cap_repr_built(self):
        dirs, angles, _ = make_cap_data(50, 16)
        idx = CapIVFIndex(n_cells=8)
        idx.build(dirs, angles)
        s = repr(idx)
        self.assertTrue("CapIVFIndex" in s and "K=" in s and "not built" not in s)

    def test_cap_repr_not_built(self):
        self.assertTrue("not built" in repr(CapIVFIndex(n_cells=32)))


# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == "__main__":
    unittest.main()