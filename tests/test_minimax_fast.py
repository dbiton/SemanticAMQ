"""MinimaxConstruct's Rust path (capcover_rs mm_* kernels) against the NumPy reference.

Tolerance, by design: the kernels rank by float32 inner product like the reference but sum in a
different order, so a near-tie can pick another member / direction; Badoiu-Clarkson then takes a
different (equally valid) trajectory for that cluster. Tests therefore assert (a) the same best
direction on all but a small share of clusters and caps no looser than the reference's beyond a
small slack, and (b) that every final cover is valid, which no tolerance applies to."""
import unittest

import numpy as np

from filters.cap_cover.cap_geometry import angle_between, segments
from filters.cap_cover.minimax_construct import FAST_AVAILABLE, MinimaxConstruct, badoiu_clarkson, reseed_empty


def _unit(a):
    return a / np.linalg.norm(a, axis=1, keepdims=True)


class _Stub:
    def __init__(self, k):
        self.max_construct_caps = k


def assert_cover(test, X, U, theta, slack=1e-9):
    """Every row of X lies within theta[j] of some direction U[j] (float64, no float32 involved)."""
    X = np.asarray(X, dtype=np.float64)
    test.assertEqual(len(U), len(theta))
    test.assertTrue(np.allclose(np.linalg.norm(U, axis=1), 1.0, atol=1e-9))
    best = np.full(len(X), np.inf)
    for s in range(0, len(X), 512):
        ang = angle_between(X[s:s + 512, None, :], U[None, :, :])      # (b, k)
        best[s:s + 512] = (ang - theta[None, :]).min(axis=1)
    test.assertLessEqual(best.max(), slack)


@unittest.skipUnless(FAST_AVAILABLE, "capcover_rs built without the minimax kernels")
class TestMinimaxFast(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(3)
        centers = _unit(rng.normal(size=(30, 64)))
        cls.X = _unit(centers[rng.integers(0, 30, 900)] + 0.35 * rng.normal(size=(900, 64))).astype(np.float32)

    # --- kernels ---------------------------------------------------------------------------
    def test_assign_matches_numpy(self):
        import capcover_rs as R
        rng = np.random.default_rng(0)
        for n, k, d in [(1, 1, 5), (7, 3, 4), (101, 37, 33), (64, 16, 16), (50, 17, 384), (200, 130, 100)]:
            X = _unit(rng.normal(size=(n, d))).astype(np.float32)
            U = _unit(rng.normal(size=(k, d))).astype(np.float32)
            lab, ip = R.mm_assign(X, U)
            S = X @ U.T
            self.assertTrue(np.all(S[np.arange(n), lab] >= S.max(axis=1) - 1e-5))
            self.assertTrue(np.allclose(ip, S.max(axis=1), atol=1e-5))
            self.assertTrue(np.all((lab >= 0) & (lab < k)))

    def test_assign_ties_take_lowest_index(self):
        import capcover_rs as R
        rng = np.random.default_rng(1)
        U = _unit(rng.normal(size=(40, 20))).astype(np.float32)
        U = np.vstack([U, U, U])                      # every direction three times
        X = _unit(rng.normal(size=(60, 20))).astype(np.float32)
        lab, _ = R.mm_assign(X, np.ascontiguousarray(U))
        self.assertTrue(np.all(lab < 40))

    def test_lanes_and_candidates(self):
        import capcover_rs as R
        rng = np.random.default_rng(2)
        X = _unit(rng.normal(size=(80, 24))).astype(np.float32)
        for k in (1, 5, 16, 47):
            U = _unit(rng.normal(size=(k, 24))).astype(np.float32)
            cand = R.mm_lanes(X, U)
            self.assertEqual(cand.shape, (80, 16))
            self.assertTrue(np.all(cand < k))
            S = X @ U.T
            lab_c, _ = R.mm_assign_cand(X, U, cand)
            lab, _ = R.mm_assign(X, U)
            # every direction is some lane's winner when full-dimensional rows rank it
            self.assertTrue(np.array_equal(lab_c, lab) or np.all(S[np.arange(80), lab_c] >= S.max(1) - 1e-5))
        with self.assertRaises(ValueError):
            R.mm_assign_cand(X, U, np.full((80, 4), k + 3, dtype=np.int32))

    def test_stats(self):
        import capcover_rs as R
        X = _unit(np.random.default_rng(4).normal(size=(50, 8)))
        U = _unit(np.random.default_rng(5).normal(size=(6, 8)))
        lab = np.arange(50) % 5                       # direction 5 stays empty
        th, tc, cnt = R.mm_stats(X, U, lab.astype(np.int64))
        self.assertTrue(np.allclose(th, angle_between(X, U[lab]), atol=1e-12))
        self.assertEqual(list(cnt), [10, 10, 10, 10, 10, 0])
        self.assertEqual(tc[5], 0.0)
        self.assertTrue(np.allclose(tc[:5], [th[lab == c].max() for c in range(5)]))

    # --- Badoiu-Clarkson against the reference -----------------------------------------------
    def _bc_pair(self, X, labels, U, steps=64):
        order, starts, counts = segments(labels, len(U))
        Xs, ls = X[order].astype(np.float64), labels[order]
        a = badoiu_clarkson(Xs, ls, U, starts, counts, steps, fast=True)
        b = badoiu_clarkson(Xs, ls, U, starts, counts, steps, fast=False)
        return Xs, ls, counts, a, b

    def test_bc_same_directions(self):
        X = self.X
        rng = np.random.default_rng(7)
        U = _unit(X[rng.choice(len(X), 25, replace=False)].astype(np.float64))
        labels = np.argmax(X @ U.T.astype(np.float32), axis=1)
        Xs, ls, counts, a, b = self._bc_pair(X, labels, U)
        self.assertTrue(np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-12))
        same = np.max(np.abs(a - b), axis=1) < 1e-6
        self.assertGreaterEqual(same.mean(), 0.9)       # float32 summation-order ties may fork a cluster
        # cap angle of every cluster under fast directions is no looser than the reference's + slack
        ta = np.zeros(len(U)); tb = np.zeros(len(U))
        np.maximum.at(ta, ls, angle_between(Xs, a[ls]))
        np.maximum.at(tb, ls, angle_between(Xs, b[ls]))
        self.assertTrue(np.all(ta <= tb * 1.02 + 1e-9))

    def test_bc_singletons_empty_and_pairs(self):
        rng = np.random.default_rng(8)
        X = _unit(rng.normal(size=(9, 16)))
        U = _unit(rng.normal(size=(6, 16)))
        labels = np.array([0, 0, 0, 1, 2, 2, 4, 4, 4])     # 3 empty, 5 empty, 1 singleton
        Xs, ls, counts, a, b = self._bc_pair(X, labels, U)
        self.assertTrue(np.allclose(a[1], Xs[3]))           # singleton: the member itself
        self.assertTrue(np.allclose(a[[3, 5]], U[[3, 5]]))  # empty: direction unchanged
        self.assertTrue(np.allclose(a, b, atol=1e-6))

    def test_bc_duplicates_and_coincident(self):
        rng = np.random.default_rng(9)
        base = _unit(rng.normal(size=(1, 12)))
        for noise in (0.0, 1e-9, 1e-6):
            X = _unit(base + noise * rng.normal(size=(20, 12)))
            U = _unit(rng.normal(size=(2, 12)))
            labels = np.zeros(20, dtype=np.int64)
            Xs, ls, counts, a, b = self._bc_pair(X, labels, U)
            ang = angle_between(Xs, a[0][None, :]).max()
            ref = angle_between(Xs, b[0][None, :]).max()
            self.assertLess(ref, 0.1)                       # 64 steps of 1/(t+1) from a random start
            self.assertLessEqual(ang, ref * 1.01 + 1e-9)

    def test_bc_zero_steps(self):
        rng = np.random.default_rng(10)
        X = _unit(rng.normal(size=(12, 6)))
        U = _unit(rng.normal(size=(3, 6)))
        labels = np.arange(12) % 3
        Xs, ls, counts, a, b = self._bc_pair(X, labels, U, steps=0)
        self.assertTrue(np.allclose(a, b, atol=1e-12))

    def test_reseed_with_given_theta(self):
        rng = np.random.default_rng(11)
        X = _unit(rng.normal(size=(30, 5)))
        U = _unit(rng.normal(size=(4, 5)))
        labels = np.arange(30) % 3
        counts = np.bincount(labels, minlength=4)
        a = reseed_empty(X, U, counts, labels)
        b = reseed_empty(X, U, counts, labels, theta=angle_between(X, U[labels]))
        self.assertTrue(np.array_equal(a, b))

    # --- whole construction ----------------------------------------------------------------
    def _variants(self):
        return {"ref": MinimaxConstruct(fast=False), "fast": MinimaxConstruct(fast=True),
                "adaptive": MinimaxConstruct(fast=True, adaptive=True),
                "approx": MinimaxConstruct(fast=True, approx_assign=True, proj_dim=16)}

    def test_construct_covers_every_vector(self):
        for k in (3, 12, 60, 400):
            for name, m in self._variants().items():
                U, th = m.construct_filter(_Stub(k), self.X)
                self.assertLessEqual(len(U), k, name)
                assert_cover(self, self.X, U, th)

    def test_construct_quality_close_to_reference(self):
        out = {n: m.construct_filter(_Stub(60), self.X) for n, m in self._variants().items()}
        ref = np.sum(out["ref"][1] ** 8)
        self.assertLessEqual(np.sum(out["fast"][1] ** 8), ref * 1.05)
        self.assertLessEqual(np.sum(out["approx"][1] ** 8), ref * 1.10)
        self.assertLessEqual(out["adaptive"][1].mean(), out["ref"][1].mean() * 1.05)

    def test_tiny_and_degenerate_inputs(self):
        rng = np.random.default_rng(12)
        X = _unit(rng.normal(size=(5, 8))).astype(np.float32)
        for m in self._variants().values():
            U, th = m.construct_filter(_Stub(5), X)             # n <= k: every vector its own cap
            self.assertEqual(len(U), 5)
            self.assertTrue(np.all(th == 0))
            U, th = m.construct_filter(_Stub(0), X)
            self.assertEqual(len(U), 0)
            for k in (1, 2, 4):                                  # k near n
                U, th = m.construct_filter(_Stub(k), X)
                assert_cover(self, X, U, th)
            D = np.repeat(X[:2], 6, axis=0)                      # exact duplicates, k between distinct and n
            for k in (1, 2, 5):
                U, th = m.construct_filter(_Stub(k), D)
                assert_cover(self, D, U, th)
            near = _unit(X[:1] + 1e-9 * rng.normal(size=(10, 8))).astype(np.float32)
            U, th = m.construct_filter(_Stub(3), near)           # nearly coincident
            assert_cover(self, near, U, th)

    def test_k_close_to_n(self):
        rng = np.random.default_rng(13)
        X = _unit(rng.normal(size=(40, 10))).astype(np.float32)
        for k in (38, 39):
            for m in self._variants().values():
                U, th = m.construct_filter(_Stub(k), X)
                assert_cover(self, X, U, th)

    def test_default_uses_fast_path(self):
        self.assertTrue(MinimaxConstruct().fast)
        self.assertFalse(MinimaxConstruct().adaptive)
        self.assertFalse(MinimaxConstruct().approx_assign)
        self.assertFalse(MinimaxConstruct(fast=False).fast)


if __name__ == "__main__":
    unittest.main()
