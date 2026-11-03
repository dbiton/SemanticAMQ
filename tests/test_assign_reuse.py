"""AssignState (bound-based reuse of the nearest-direction search) against a full re-search.

The kernel ranks by float32 inner product; AssignState skips a row only when its owner beats every other
direction by more than a conservative float32 margin (~2e-4 rad), so its labels must equal mm_assign's
exactly. (Rows within that margin of a tie are re-searched by the same kernel, hence also identical.)"""
import unittest

import numpy as np

from filters.cap_cover.minimax_construct import REUSE_AVAILABLE, AssignState, MinimaxConstruct, _rs_assign


def _unit(a):
    return a / np.linalg.norm(a, axis=1, keepdims=True)


class _Stub:
    def __init__(self, k):
        self.max_construct_caps = k


def _full(X32, U):
    return _rs_assign(X32, np.ascontiguousarray(U, dtype=np.float32))[0]


@unittest.skipUnless(REUSE_AVAILABLE, "capcover_rs built without as_top2")
class TestAssignState(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(7)

    def data(self, n=600, k=40, d=24, clusters=12):
        centers = _unit(self.rng.standard_normal((clusters, d)))
        X = _unit(centers[self.rng.integers(0, clusters, n)] + 0.3 * self.rng.standard_normal((n, d)))
        U = X[self.rng.choice(n, k, replace=False)].copy()
        return X, U

    def state(self, X):
        return AssignState(X.astype(np.float32), X, reuse=True, min_k=0)

    def check(self, st, X32, U):
        lab = st.assign(U.astype(np.float32), U)
        np.testing.assert_array_equal(lab, _full(X32, U))
        return lab

    def test_small_moves_equal_full_search_and_prune(self):
        X, U = self.data(n=2000, k=60)
        X32 = X.astype(np.float32)
        st = self.state(X)
        self.check(st, X32, U)
        for _ in range(8):
            U = _unit(U + 0.0005 * self.rng.standard_normal(U.shape))
            self.check(st, X32, U)
        self.assertLess(max(f for f, _ in st.history[1:]), 0.2)   # the bounds do prune on small moves

    def test_large_moves(self):
        X, U = self.data()
        X32 = X.astype(np.float32)
        st = self.state(X)
        for scale in (0.05, 0.5, 3.0, 0.01, 10.0):
            U = _unit(U + scale * self.rng.standard_normal(U.shape))
            self.check(st, X32, U)

    def test_mixed_movers_and_static(self):
        X, U = self.data(n=1500, k=80)
        X32 = X.astype(np.float32)
        st = self.state(X)
        self.check(st, X32, U)
        for _ in range(6):
            move = self.rng.random(len(U)) < 0.1
            U = U.copy()
            U[move] = _unit(U[move] + 1.0 * self.rng.standard_normal(U[move].shape))
            U[~move] = _unit(U[~move] + 0.002 * self.rng.standard_normal(U[~move].shape))
            self.check(st, X32, U)

    def test_duplicates_and_ties(self):
        X, U = self.data(n=300, k=30)
        X = np.concatenate([X, X[:100], X[:100]])
        U = np.concatenate([U, U[:10], U[:10]])             # duplicate directions: exact ties
        X32 = X.astype(np.float32)
        st = self.state(X)
        self.check(st, X32, U)
        for _ in range(5):
            U = U.copy()
            U[:30] = _unit(U[:30] + 0.003 * self.rng.standard_normal(U[:30].shape))
            U[30:] = U[:len(U) - 30]                        # keep copies equal to their originals
            self.check(st, X32, U)

    def test_empty_cluster_reseed_jump(self):
        X, U = self.data(n=500, k=50)
        X32 = X.astype(np.float32)
        st = self.state(X)
        self.check(st, X32, U)
        for _ in range(4):
            U = _unit(U + 0.005 * self.rng.standard_normal(U.shape))
            U[self.rng.choice(len(U), 5, replace=False)] = X[self.rng.choice(len(X), 5, replace=False)]
            self.check(st, X32, U)

    def test_k_near_n_and_small(self):
        for n, k in ((60, 59), (60, 60), (50, 70), (30, 2), (30, 1), (200, 150)):
            X, _ = self.data(n=n, k=1, clusters=6)
            U = X[self.rng.choice(n, min(k, n), replace=False)]
            if k > n:
                U = np.concatenate([U, _unit(self.rng.standard_normal((k - n, X.shape[1])))])
            X32 = X.astype(np.float32)
            st = self.state(X)
            self.check(st, X32, U)
            for _ in range(4):
                U = _unit(U + 0.004 * self.rng.standard_normal(U.shape))
                self.check(st, X32, U)

    def test_reuse_off_is_plain_search(self):
        X, U = self.data()
        st = AssignState(X.astype(np.float32), X, reuse=False)
        self.check(st, X.astype(np.float32), U)
        self.check(st, X.astype(np.float32), _unit(U + 0.01))
        self.assertTrue(all(f == 1.0 for f, _ in st.history))


@unittest.skipUnless(REUSE_AVAILABLE, "capcover_rs built without as_top2")
class TestRefineParity(unittest.TestCase):
    def test_refine_covers_identical(self):
        rng = np.random.default_rng(3)
        centers = _unit(rng.standard_normal((30, 48)))
        X = _unit(centers[rng.integers(0, 30, 3000)] + 0.35 * rng.standard_normal((3000, 48)))
        k = 120
        U0 = X[rng.choice(len(X), k, replace=False)].copy()
        a = MinimaxConstruct(reuse_assign=True, assign_min_k=0)
        b = MinimaxConstruct(reuse_assign=False)
        Ua, ta = a.refine(X, U0)
        Ub, tb = b.refine(X, U0)
        np.testing.assert_array_equal(Ua, Ub)
        np.testing.assert_array_equal(ta, tb)
        self.assertTrue(any(f < 1.0 for f, _ in a.assign_history))


if __name__ == "__main__":
    unittest.main()
