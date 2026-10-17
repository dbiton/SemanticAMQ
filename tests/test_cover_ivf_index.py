import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'src'))
import unittest
import numpy as np
import time
from src.filters.ball_cover.ball_ivf_index import BallIVFIndex, _euclidean_kmeans as _ball_kmeans
from src.filters.cap_cover.cap_ivf_index  import CapIVFIndex,  _spherical_kmeans as _cap_kmeans


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
    """True if any sphere accepts the query (exact, float64; the reach in
    float64 too, as BallCoverFilter computes it)."""
    reach = radii.astype(np.float64) + tau + eps
    out   = np.zeros(len(Qr), dtype=bool)
    C64   = centers.astype(np.float64)
    for i, q in enumerate(Qr.astype(np.float64)):
        d2 = ((q - C64) ** 2).sum(1)
        out[i] = (d2 <= reach ** 2).any()
    return out

def ref_cap(dirs, angles, beta, feasible, Qn):
    """True if any cap accepts the query: CapCoverFilter._scan's float64 rule,
    theta_j + beta >= pi or angle(q, u_j) <= theta_j + beta (chord formula)."""
    U   = dirs.astype(np.float64)
    U  /= np.linalg.norm(U, axis=1, keepdims=True)
    Q   = Qn.astype(np.float64)
    d2  = (Q * Q).sum(1)[:, None] + (U * U).sum(1)[None, :] - 2.0 * (Q @ U.T)
    ang = 2.0 * np.arcsin(np.minimum(0.5 * np.sqrt(np.maximum(d2, 0.0)), 1.0))
    lim = np.asarray(angles, np.float64)[None, :] + np.asarray(beta, np.float64)[:, None]
    return np.asarray(feasible, bool) & ((ang <= lim) | (lim >= np.pi)).any(1)

def make_clustered_caps(N: int = 400, d: int = 32, n_q: int = 2000, seed: int = 0):
    """Clustered directions and queries (like embeddings): the regime where a
    cell gate that ignores each cap's angle to its centroid drops accepts."""
    r    = rng(seed)
    base = r.standard_normal((8, d))
    dirs = unit(base[r.integers(8, size=N)] + 0.6 * r.standard_normal((N, d)))
    Q    = unit(base[r.integers(8, size=n_q)] + 0.6 * r.standard_normal((n_q, d)))
    return dirs, r.uniform(0.05, 0.4, N), Q

def _cap_beta(tau: float, eps: float = 1e-3) -> float:
    """beta at tau, norm=1 -- worst case for unit queries."""
    tp    = tau + eps + 1e-4
    cos_b = (1.0 + 1.0 - tp ** 2) / 2.0
    return float(np.arccos(np.clip(cos_b, -1, 1)))


# ===========================================================================
# TestLayout  --  cells, storage order, cell reach U_c
# ===========================================================================

class TestLayout(unittest.TestCase):

    def test_ball_cells_partition_and_bound_members(self):
        centers, radii, _ = make_ball_data(300, 32)
        idx = BallIVFIndex()
        idx.build(centers, radii)
        order, cs, cell_U = idx.layout()
        self.assertEqual(cs[-1], 300)
        self.assertTrue((np.diff(cs) >= 0).all())
        self.assertTrue((np.sort(order) == np.arange(300)).all())
        C = centers[order].astype(np.float64)
        r = radii[order].astype(np.float64)
        for c in range(idx.K):
            s_, e_ = int(cs[c]), int(cs[c + 1])
            mu = C[s_:e_].mean(axis=0) # each centroid is its members' mean
            reach = r[s_:e_] + np.linalg.norm(C[s_:e_] - mu, axis=1)
            self.assertTrue((reach <= cell_U[c] + 1e-6).all(), f"cell {c}: U_c below a member's reach")
            self.assertTrue((np.diff(r[s_:e_]) <= 0).all(), f"cell {c}: not widest-first")

    def test_cap_cells_partition_and_bound_members(self):
        dirs, angles, _ = make_cap_data(300, 32)
        idx = CapIVFIndex()
        idx.build(dirs, angles)
        self.assertEqual(idx.K, int(np.sqrt(300)), "K must be sqrt(N)")
        order, cs, cell_U = idx.layout()
        self.assertEqual(cs[-1], 300)
        self.assertTrue((np.diff(cs) >= 0).all())
        self.assertTrue((np.sort(order) == np.arange(300)).all())
        U = dirs[order].astype(np.float64)
        U /= np.linalg.norm(U, axis=1, keepdims=True)
        theta = angles[order]
        for c in range(idx.K):
            s_, e_ = int(cs[c]), int(cs[c + 1])
            mu = U[s_:e_].sum(axis=0) # spherical k-means: normalised member mean
            mu /= np.linalg.norm(mu)
            delta = np.arccos(np.clip(U[s_:e_] @ mu, -1.0, 1.0))
            self.assertTrue((theta[s_:e_] + delta <= cell_U[c] + 1e-6).all(),
                            f"cell {c}: U_c below a member's theta + delta")
            self.assertTrue((np.diff(theta[s_:e_]) <= 0).all(), f"cell {c}: not widest-first")


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
                ref = ref_ball(centers, radii, tau, eps, queries)
                ivf = idx.query(queries, tau=tau, eps=eps)
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
            ref = ref_ball(centers, radii, tau, eps, queries)
            ivf = idx.query(queries, tau=tau, eps=eps)
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
        out = idx.query(q, tau=0.1, eps=1e-3)
        self.assertTrue(out[0], "center query must be accepted")
        self.assertFalse(out[1], "antipodal query must be rejected")

    def test_empty_build(self):
        d   = 64
        idx = BallIVFIndex(n_cells=16)
        idx.build(np.empty((0, d), dtype=np.float32),
                  np.empty(0, dtype=np.float32))
        q   = unit(np.ones((5, d), dtype=np.float32))
        self.assertFalse(idx.query(q, tau=0.1, eps=1e-3).any())

    def test_zero_fnr_high_dim_both_paths(self):
        """d = 384 (> the 64-coordinate PCA head), anisotropic clustered centers
        off the unit sphere (as the blocklist covers' decoded centers), unit
        queries, several tau in a row (the per-tau reach is reset between
        calls): exact on the serial path and on parallel tiles."""
        r    = rng(22)
        d    = 384
        spec = np.exp(-np.arange(d) / 40.0)
        base = r.standard_normal((10, d)) * spec
        dirs = base[r.integers(10, size=1200)] + 0.35 * r.standard_normal((1200, d)) * spec
        centers = (2.5 * dirs / np.linalg.norm(dirs, axis=1, keepdims=True)).astype(np.float32)
        Q    = unit(base[r.integers(10, size=500)] + 0.35 * r.standard_normal((500, d)) * spec)
        radii = r.uniform(1.6, 1.8, 1200).astype(np.float32)
        for pmb in (1, 10 ** 9):                      # parallel tiles / serial
            idx = BallIVFIndex(parallel_min_batch=pmb)
            idx.build(centers, radii)
            for tau in (0.05, 0.10, 0.20, 0.10):
                ref = ref_ball(centers, radii, tau, 1e-3, Q)
                ivf = idx.query(Q, tau=tau, eps=1e-3)
                self.assertEqual((ref & ~ivf).sum(), 0, f"FN tau={tau} parallel_min_batch={pmb}")
                self.assertEqual((ivf & ~ref).sum(), 0, f"extra FP tau={tau} parallel_min_batch={pmb}")

    def test_exact_callback_decides_undecided_queries(self):
        """Queries placed exactly on a ball's boundary may fall inside the exact
        test's float margin: the index must hand them (and only queries it
        cannot decide) to `exact`, and use its answer -- one query at a time
        (survivors tested per query) and as one batch (tested grouped by ball)."""
        r = rng(24)
        centers, radii, _ = make_ball_data(300, 48)
        tau, eps = 0.1, 1e-3
        j = r.integers(300, size=40)
        dirs = unit(r.standard_normal((40, 48)).astype(np.float32)).astype(np.float64)
        reach = radii[j].astype(np.float64) + tau + eps
        Q = (centers[j].astype(np.float64) + dirs * reach[:, None]).astype(np.float32)
        Q = np.vstack([Q, unit(r.standard_normal((40, 48)).astype(np.float32))])
        ref = ref_ball(centers, radii, tau, eps, Q)
        idx = BallIVFIndex()
        idx.build(centers, radii)
        for bs in (1, len(Q)):
            asked = []
            out = []
            for s0 in range(0, len(Q), bs):
                rows0 = np.arange(s0, min(s0 + bs, len(Q)))
                def exact(rows, rows0=rows0):
                    asked.extend(rows0[rows].tolist())
                    return ref[rows0[rows]]
                out.append(idx.query(Q[rows0], tau=tau, eps=eps, exact=exact))
            self.assertTrue((np.concatenate(out) == ref).all(), f"batch size {bs}")
            self.assertLessEqual(len(asked), 40, "only boundary queries may need the exact test")

    def test_sketch_fits_its_budget(self):
        """The sketch (int8 PCA head, uint8 tail, locator, cells) fits the
        byte budget it is given -- the head narrows to fit -- and the index
        stays exact at every width."""
        for cls, (X, w, Q), per_member in ((BallIVFIndex, make_ball_data(2000, 384), 194),
                                           (CapIVFIndex, make_cap_data(2000, 384), 75),
                                           (CapIVFIndex, make_cap_data(2000, 384), 40)):
            idx = cls()
            copy = 2000 * 384 * 4 + 2000 * 8 # the standalone float32 copy (+ angles / radii, ids)
            budget = copy + per_member * 2000
            idx.build(X, w, max_memory=budget)
            self.assertEqual(idx._kd.XC.dtype, np.int8)
            self.assertLessEqual(idx.get_bytesize() - idx._kd.owned, per_member * 2000,
                                 f"{cls.__name__} h={idx.h}")
            if cls is BallIVFIndex:
                self.assertTrue((idx.query(Q, 0.1, 1e-3) == ref_ball(X, w, 0.1, 1e-3, Q)).all())
            else:
                b = np.full(len(Q), 0.1)
                f = np.ones(len(Q), bool)
                self.assertTrue((idx.query(Q, b, f) == ref_cap(X, w, b, f, Q)).all(), f"h={idx.h}")

    def test_no_index_when_nothing_fits(self):
        """A budget too small for any sketch builds no index, and an unbuilt
        index refuses to answer rather than answer wrongly."""
        X, w, Q = make_ball_data(500, 64)
        idx = BallIVFIndex()
        idx.build(X, w, max_memory=1000)
        self.assertFalse(idx._built)
        self.assertEqual(idx.get_bytesize(), 0)
        with self.assertRaises(RuntimeError):
            idx.query(Q, 0.1, 1e-3)

    def test_mixed_batch_sizes(self):
        centers, radii, queries = make_ball_data(500, 96)
        ref = ref_ball(centers, radii, 0.15, 1e-3, queries)
        idx = BallIVFIndex()
        idx.build(centers, radii)
        out = np.concatenate([idx.query(queries[s:s + bs], tau=0.15, eps=1e-3)
                              for bs, s in zip((1, 7, 32, 160), (0, 1, 8, 40))])
        self.assertTrue((out == ref).all())

    def test_infinite_radius_accepts_everything(self):
        centers, radii, queries = make_ball_data(100, 32)
        radii = radii.copy()
        radii[37] = np.inf
        idx = BallIVFIndex()
        idx.build(centers, radii)
        self.assertTrue(idx.query(queries, tau=0.0, eps=1e-3).all())

    def test_negative_reach_admits_nothing(self):
        """r + tau + eps < 0 for every ball: nothing is accepted, even at the center."""
        centers, radii, _ = make_ball_data(100, 32)
        idx = BallIVFIndex()
        idx.build(centers, radii)
        self.assertFalse(idx.query(centers[:20], tau=-1.0, eps=1e-3).any())

    def test_large_tau_matches_ref(self):
        """At tau=3.0 the IVF and brute-force must still agree."""
        centers, radii, queries = make_ball_data(100, 32)
        tau = 3.0; eps = 1e-3
        idx = BallIVFIndex(n_cells=16)
        idx.build(centers, radii)
        ref = ref_ball(centers, radii, tau, eps, queries)
        ivf = idx.query(queries, tau=tau, eps=eps)
        self.assertEqual((ref & ~ivf).sum(), 0)


# ===========================================================================
# TestCapIVFIndex
# ===========================================================================

class TestCapIVFIndex(unittest.TestCase):

    def _build(self, dirs, angles, tau, d):
        eps      = 1e-3
        beta_max = _cap_beta(tau, eps)
        idx = CapIVFIndex(n_cells=32)
        idx.build(dirs, angles)
        return idx, beta_max

    def _beta_arr(self, n: int, beta_max: float):
        return np.full(n, beta_max, dtype=np.float32)

    # ---- 0-FNR ----

    def test_zero_fnr(self):
        for tau in [0.05, 0.15, 0.30]:
            dirs, angles, queries = make_cap_data(200, 64)
            d        = 64
            idx, beta_max = self._build(dirs, angles, tau, d)
            feasible = np.ones(len(queries), dtype=bool)
            beta_arr = self._beta_arr(len(queries), beta_max)

            Qn32 = queries.astype(np.float32)
            ref  = ref_cap(dirs, angles, beta_arr, feasible, Qn32)
            ivf  = idx.query(Qn32, beta_arr, feasible)
            fn   = ref & ~ivf
            self.assertEqual(fn.sum(), 0,
                             f"CapIVFIndex: {fn.sum()} FN at tau={tau}")

    # ---- No extra FP ----

    def test_zero_extra_fp(self):
        dirs, angles, queries = make_cap_data(200, 64)
        d        = 64
        tau      = 0.10
        idx, beta_max = self._build(dirs, angles, tau, d)
        feasible = np.ones(len(queries), dtype=bool)
        beta_arr = self._beta_arr(len(queries), beta_max)

        Qn32 = queries.astype(np.float32)
        ref  = ref_cap(dirs, angles, beta_arr, feasible, Qn32)
        ivf  = idx.query(Qn32, beta_arr, feasible)
        self.assertEqual((ivf & ~ref).sum(), 0, "CapIVFIndex: extra FP")

    def test_zero_fnr_clustered(self):
        """Regression: the first gate (max theta per cell, no member-to-centroid
        angle) rejected a large share of the accepted queries here.  Exact on
        both kernel paths: serial (one query at a time) and parallel tiles."""
        for seed in range(5):
            dirs, angles, Q = make_clustered_caps(seed=seed)
            beta = np.full(len(Q), 0.15)
            feas = np.ones(len(Q), dtype=bool)
            ref  = ref_cap(dirs, angles, beta, feas, Q)
            for pmb in (1, 10 ** 9):
                idx = CapIVFIndex(parallel_min_batch=pmb)
                idx.build(dirs, angles)
                ivf = idx.query(Q, beta, feas)
                self.assertEqual((ref & ~ivf).sum(), 0, f"FN seed={seed} parallel_min_batch={pmb}")
                self.assertEqual((ivf & ~ref).sum(), 0, f"extra FP seed={seed} parallel_min_batch={pmb}")

    def test_high_dim_embedding_like(self):
        """d = 384 > the 64-coordinate PCA head, anisotropic clustered data (as
        sentence embeddings): exercises every abandon checkpoint and the full
        test in the original basis, at mixed batch sizes and per-query beta."""
        r    = rng(21)
        d    = 384
        spec = np.exp(-np.arange(d) / 40.0)
        base = r.standard_normal((12, d)) * spec
        dirs = unit(base[r.integers(12, size=1500)] + 0.35 * r.standard_normal((1500, d)) * spec)
        Q    = unit(base[r.integers(12, size=600)] + 0.35 * r.standard_normal((600, d)) * spec)
        angles = r.uniform(0.2, 0.5, 1500)
        beta = r.uniform(0.0, 0.2, len(Q))
        feas = np.ones(len(Q), dtype=bool)
        ref  = ref_cap(dirs, angles, beta, feas, Q)
        self.assertTrue(0 < ref.sum() < len(Q), "test data should mix accepts and rejects")
        idx = CapIVFIndex()
        idx.build(dirs, angles)
        out = np.concatenate([idx.query(Q[s:s + bs], beta[s:s + bs], feas[s:s + bs])
                              for bs, s in zip((1, 3, 16, 64, 516), (0, 1, 4, 20, 84))])
        self.assertTrue((out == ref).all(), f"FN={(ref & ~out).sum()} FP={(out & ~ref).sum()}")

    def test_abandon_bound_is_a_lower_bound(self):
        """(*) of cap_ivf_kernels on the stored sketch: (||q_h - x^_h|| - dl)_+^2
        (x^ the int8 head) plus the distance from |q past the head| to the
        cap's uint8 tail interval never exceeds ||q - x||^2."""
        dirs, angles, Q = make_clustered_caps(N=300, d=96, n_q=200, seed=3)
        idx = CapIVFIndex()
        idx.build(dirs, angles)
        kd  = idx._kd
        pos, _, g = kd.layout()
        X   = dirs[g].astype(np.float64)
        X  /= np.linalg.norm(X, axis=1, keepdims=True)
        Q64 = Q.astype(np.float64)
        Yq  = Q64 @ kd.Eh.astype(np.float64)
        Xh  = kd.head(pos) * kd.su
        d2  = ((Q64[:, None, :] - X[None, :, :]) ** 2).sum(-1)
        part = ((Yq[:, None, :] - Xh[None, :, :]) ** 2).sum(-1)
        hs = np.maximum(np.sqrt(part) - kd.dl * kd.su, 0.0)
        qt = np.sqrt(np.maximum((Q64 * Q64).sum(1) - (Yq ** 2).sum(1), 0.0))[:, None]
        lo = kd.TQ[pos].astype(np.float64)[None, :] * kd.st
        td = np.maximum(np.maximum(lo - qt, qt - lo - kd.st), 0.0)
        self.assertTrue((hs ** 2 + td ** 2 <= d2 + 1e-9).all(), "bound exceeds the distance")

    def test_boundary_queries_accepted(self):
        """Queries placed at exactly theta + beta from a cap (float64) are accepted."""
        d = 48
        r = rng(9)
        dirs, angles, _ = make_cap_data(300, d)
        idx  = CapIVFIndex()
        idx.build(dirs, angles)
        U    = dirs.astype(np.float64) / np.linalg.norm(dirs.astype(np.float64), axis=1, keepdims=True)
        j    = r.integers(300, size=200)
        beta = r.uniform(0.0, 0.3, 200)
        v    = r.standard_normal((200, d))
        v   -= (v * U[j]).sum(1, keepdims=True) * U[j]
        v   /= np.linalg.norm(v, axis=1, keepdims=True)
        a    = angles[j] + beta - 1e-9
        Q64  = np.cos(a)[:, None] * U[j] + np.sin(a)[:, None] * v
        feas = np.ones(200, dtype=bool)
        out  = idx.query(Q64, beta, feas)
        self.assertTrue(out.all(), f"{(~out).sum()} boundary queries rejected")

    def test_theta_plus_beta_past_pi_accepts(self):
        """theta + beta >= pi accepts any feasible query (cos is not monotone past pi)."""
        d   = 8
        u   = unit(np.ones((1, d), dtype=np.float32))
        idx = CapIVFIndex()
        idx.build(u, np.array([3.0]))
        Q   = np.vstack([-u, u]).astype(np.float32)
        out = idx.query(Q, np.array([0.5, 0.5]), np.ones(2, dtype=bool))
        self.assertTrue(out.all())

    # ---- Infeasible queries ----

    def test_infeasible_always_rejected(self):
        dirs, angles, _ = make_cap_data(50, 32)
        d   = 32
        tau = 0.1
        idx, beta_max = self._build(dirs, angles, tau, d)
        nq       = 30
        Qn32     = unit(rng(5).standard_normal((nq, d)).astype(np.float32))
        feasible = np.zeros(nq, dtype=bool) # all infeasible
        beta_arr = self._beta_arr(nq, beta_max)
        
        out      = idx.query(Qn32, beta_arr, feasible)
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
        
        out      = idx.query(q, beta_arr, feasible)
        self.assertFalse(out.any())

    def test_single_cap(self):
        d   = 16
        u   = unit(np.ones((1, d), dtype=np.float32))
        th  = np.array([0.3], dtype=np.float64)
        idx = CapIVFIndex(n_cells=1)
        beta_max = _cap_beta(0.1)
        idx.build(u, th)

        # Query aligned with direction: must accept
        q_in  = u.copy()
        # Query antipodal: must reject
        q_out = -u
        Qn32  = np.vstack([q_in, q_out]).astype(np.float32)
        feas  = np.ones(2, dtype=bool)
        beta  = np.full(2, beta_max, dtype=np.float32)
        
        out   = idx.query(Qn32, beta, feas)
        self.assertTrue(out[0], "direction query should be accepted")
        self.assertFalse(out[1], "antipodal query should be rejected")

    # ---- Pruning fraction diagnostic ----

    def test_pruning_fraction_in_range(self):
        dirs, angles, queries = make_cap_data(200, 64)
        idx, beta_max = self._build(dirs, angles, 0.1, 64)
        beta_arr = self._beta_arr(len(queries), beta_max)
        fracs = idx.pruning_fraction(queries.astype(np.float32), beta=beta_arr)
        self.assertTrue((fracs >= 0.0).all() and (fracs <= 1.0 + 1e-6).all())


# ===========================================================================
# TestPruning  --  pruning_fraction diagnostics (both indices)
# ===========================================================================

class TestPruning(unittest.TestCase):
    def test_fraction_in_range(self):
        centers, radii, queries = make_ball_data(200, 64)
        idx = BallIVFIndex(n_cells=32)
        idx.build(centers, radii)
        fracs = idx.pruning_fraction(queries, tau=0.1, eps=1e-3)
        self.assertTrue((fracs >= 0.0).all() and (fracs <= 1.0 + 1e-6).all())

    def test_negatives_prune_more_than_positives(self):
        """Negative queries (far from all spheres) should keep no more balls
        alive on average than queries sitting on sphere centers."""
        N, d = 400, 128
        r    = rng(20)
        centers = unit(r.standard_normal((N, d)).astype(np.float32))
        radii   = np.full(N, 0.05, dtype=np.float32)
        idx = BallIVFIndex(n_cells=64)
        idx.build(centers, radii)

        pos     = centers[:20]
        far_neg = unit(-centers[:20] + r.standard_normal((20, d)).astype(np.float32) * 0.01)

        frac_p = idx.pruning_fraction(pos, tau=0.05, eps=1e-3)
        frac_n = idx.pruning_fraction(far_neg, tau=0.05, eps=1e-3)
        self.assertTrue(frac_n.mean() < frac_p.mean() + 0.5,
                        (f"negatives fraction {frac_n.mean():.2f} should not exceed "
                         f"positives {frac_p.mean():.2f} by 0.5"))


    def test_full_stage_is_consistent_with_exactness(self):
        """stage="full" never keeps more than stage="gate"; every accepted query
        keeps at least one member alive to the full test (the accepting one --
        nothing that can accept is pruned); and on anisotropic d = 384 data the
        early-abandon stages prune most members."""
        r    = rng(23)
        d    = 384
        spec = np.exp(-np.arange(d) / 40.0)
        base = r.standard_normal((10, d)) * spec
        dirs = unit(base[r.integers(10, size=1000)] + 0.35 * r.standard_normal((1000, d)) * spec)
        Q    = unit(base[r.integers(10, size=300)] + 0.35 * r.standard_normal((300, d)) * spec)
        angles = r.uniform(0.2, 0.4, 1000)
        beta = np.full(len(Q), 0.1)
        cap = CapIVFIndex()
        cap.build(dirs, angles)
        ref = ref_cap(dirs, angles, beta, np.ones(len(Q), bool), Q)
        g, f = cap.pruning_fraction(Q, beta, stage="gate"), cap.pruning_fraction(Q, beta)
        self.assertTrue((f <= g + 1e-12).all())
        self.assertTrue((f[ref] >= 1.0 / 1000 - 1e-12).all(), "an accepted query lost its accepting cap")
        self.assertLess(f.mean(), 0.5, "early abandoning should prune most caps here")

        centers = (2.5 * dirs).astype(np.float32)
        radii = r.uniform(1.6, 1.8, 1000).astype(np.float32)
        ball = BallIVFIndex()
        ball.build(centers, radii)
        refb = ref_ball(centers, radii, 0.1, 1e-3, Q)
        g, f = ball.pruning_fraction(Q, 0.1, 1e-3, stage="gate"), ball.pruning_fraction(Q, 0.1, 1e-3)
        self.assertTrue((f <= g + 1e-12).all())
        self.assertTrue((f[refb] >= 1.0 / 1000 - 1e-12).all(), "an accepted query lost its accepting ball")


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
    def _blas_brute_cap(dirs, angles, beta, feasible, Qn32):
        """Brute force: one float64 GEMM, the exact rule (theta + beta < pi here)."""
        D64 = dirs.astype(np.float64)
        ip  = Qn32.astype(np.float64) @ D64.T
        thr = np.linalg.norm(D64, axis=1)[None, :] * np.cos(angles[None, :] + beta.astype(np.float64)[:, None])
        out = np.zeros(len(Qn32), dtype=bool)
        out[feasible] = (ip >= thr)[feasible].any(axis=1)
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

        # Warmup
        for _ in range(3):
            idx.query(neg, tau=tau, eps=eps)

        t0  = time.perf_counter()
        ivf_out = idx.query(neg, tau=tau, eps=eps)
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

        idx = CapIVFIndex(n_cells=32)
        idx.build(dirs, angles)

        Qn32     = queries.astype(np.float32)
        feasible = np.ones(len(queries), dtype=bool)
        beta_arr = np.full(len(queries), beta_max, dtype=np.float32)

        for _ in range(3):
            idx.query(Qn32, beta_arr, feasible)

        t0      = time.perf_counter()
        ivf_out = idx.query(Qn32, beta_arr, feasible)
        t_ivf   = time.perf_counter() - t0

        t0      = time.perf_counter()
        ref_out = self._blas_brute_cap(dirs, angles, beta_arr, feasible, Qn32)
        t_ref   = time.perf_counter() - t0

        fn = ref_out & ~ivf_out
        self.assertEqual(fn.sum(), 0, f"CapIVFIndex: {fn.sum()} FN in throughput test")
        
        print(f"\n  CapIVFIndex: {len(queries)/t_ivf:.0f} q/s  |  "
              f"BLAS brute-force: {len(queries)/t_ref:.0f} q/s  |  "
              f"speedup: {t_ref/t_ivf:.1f}x  (N={N}, d={d})")
        # Soft sanity: not more than 10x slower than brute force
        self.assertTrue(t_ivf <= 10 * t_ref + 1.0)


# ===========================================================================
# TestFilterMemory  --  the index next to the real filters
# ===========================================================================

class TestFilterMemory(unittest.TestCase):
    """Each filter gives its index its own size as the budget: with the index
    the filter takes at most twice its memory (the index about as much as
    the filter, never more), and answers exactly as without it."""

    @staticmethod
    def _data(n=3000, d=96, seed=40):
        r = rng(seed)
        base = r.standard_normal((30, d))
        X = unit(base[r.integers(30, size=n)] + 0.6 * r.standard_normal((n, d)))
        Q = unit(base[r.integers(30, size=400)] + 0.6 * r.standard_normal((400, d)))
        return X, Q

    def _check(self, filt, X, Q, tau):
        filt.construct(X)
        idx = filt._ivf_index
        self.assertTrue(idx._built, "the index should fit the filter's own size")
        fb, ib = filt.get_bytesize(), idx.get_bytesize()
        self.assertGreater(ib, 0)
        self.assertLessEqual(ib, fb, f"index {ib} B > filter {fb} B")
        self.assertLessEqual((fb + ib) / fb, 2.0)
        with_index = np.asarray(filt.query(Q, tau))
        filt._ivf_index = None
        without = np.asarray(filt.query(Q, tau))
        filt._ivf_index = idx
        self.assertTrue((with_index == without).all(),
                        f"FN={(without & ~with_index).sum()} FP={(with_index & ~without).sum()}")
        return ib / fb

    def test_cap_cover_filter(self):
        from src.filters.cap_cover.cap_cover_filter import CapCoverFilter
        X, Q = self._data()
        ratio = self._check(CapCoverFilter.auto(96, 40_000, ivf_index="auto"), X, Q, 0.3)
        self.assertGreater(ratio, 0.5, "the index should use most of its budget for its head")

    def test_ball_cover_filter(self):
        from src.filters.ball_cover.ball_cover_filter import BallCoverFilter, Quantize
        from src.filters.ball_cover.gonzales_construct import GonzalesConstruct
        X, Q = self._data()
        filt = BallCoverFilter(96, construct_method=GonzalesConstruct(), quantize=Quantize.TQ4,
                               max_spheres=1000, ivf_index="auto")
        ratio = self._check(filt, X, Q, 0.3)
        self.assertGreater(ratio, 0.5, "the index should use most of its budget for its head")


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