import os
import pickle
import unittest
import numpy as np

try:
    import capcover_rs
    HAVE = hasattr(capcover_rs, "gonzalez_chosen_pruned") and hasattr(capcover_rs, "gonzalez_chosen_pruned_labels")
except ImportError:
    HAVE = False

from filters.cap_cover.gonzales_construct import GonzalesConstruct
from filters.cap_cover.cap_geometry import angle_between

ELI5 = "/tmp/claude-0/-home-user/faeeddb3-925b-5717-b1cb-caa0100e8a36/scratchpad/eli5/eli5_10k.pkl"


def _unit32(X):
    X = np.asarray(X, dtype=np.float64)
    return (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.float32)


def reference_chosen(X32, k):
    """The NumPy loop of GonzalesConstruct(fast=False)."""
    n = len(X32)
    chosen = [0]
    best = X32 @ X32[0]
    ip = np.empty(n, dtype=np.float32)
    for _ in range(1, k):
        c = int(best.argmin())
        if best[c] >= 1.0:
            break
        chosen.append(c)
        np.dot(X32, X32[c], out=ip)
        np.maximum(best, ip, out=best)
    return np.array(chosen, dtype=np.int64)


class _F:
    def __init__(self, k):
        self.max_construct_caps = k


@unittest.skipUnless(HAVE, "capcover_rs.gonzalez_chosen_pruned not built")
class TestGonzalezPruned(unittest.TestCase):
    """The lazy exact kernel (chosen sequence only)."""

    def fn(self, X, k):
        return capcover_rs.gonzalez_chosen_pruned(X, k)

    def _same(self, X, k, allow_tail_swap=False):
        X = np.ascontiguousarray(X, dtype=np.float32)
        a = self.fn(X, k)
        b = reference_chosen(X, k)
        np.testing.assert_array_equal(a, b)

    def test_random_parity(self):
        rng = np.random.default_rng(0)
        for n, d, k in [(500, 32, 40), (2000, 64, 200), (1500, 384, 100), (300, 17, 120)]:
            with self.subTest(n=n, d=d, k=k):
                self._same(_unit32(rng.standard_normal((n, d))), k)

    def test_clustered_parity(self):
        rng = np.random.default_rng(1)
        C = rng.standard_normal((20, 48))
        X = _unit32(np.repeat(C, 30, axis=0) + 0.2 * rng.standard_normal((600, 48)))
        self._same(X, 60)

    def test_real_eli5(self):
        if not os.path.exists(ELI5):
            self.skipTest("ELI5 sample not available")
        X = pickle.load(open(ELI5, "rb"))[:3000].astype(np.float32)
        self._same(X, 300)

    def test_k_edges(self):
        rng = np.random.default_rng(2)
        X = _unit32(rng.standard_normal((200, 24)))
        np.testing.assert_array_equal(self.fn(X, 1), [0])
        self._same(X, 199)
        self._same(X, 2)

    def test_n_le_k_selects_every_point_once(self):
        rng = np.random.default_rng(3)
        X = _unit32(rng.standard_normal((50, 16)))
        a = self.fn(X, 80)
        self.assertEqual(sorted(a.tolist()), list(range(50)))
        np.testing.assert_array_equal(a, reference_chosen(X, 80)[:50])   # reference then re-picks a center

    def test_duplicates_ties_and_early_stop(self):
        rng = np.random.default_rng(4)
        base = _unit32(rng.standard_normal((10, 32)))
        X = np.repeat(base, 5, axis=0)            # 10 distinct points, 5 exact copies each
        a = self.fn(X, 50)
        # exact copies give bit-identical dots, so a duplicate never beats its twin
        self.assertEqual(len(a), 10)
        self.assertEqual(len(set((X[a] @ X[0]).tolist())), 10)
        # the reference may re-pick an already chosen center whose float32 self inner
        # product rounds to 0.99999994 (< 1.0); the kernel stops instead. Same prefix.
        b = reference_chosen(X, 50)
        np.testing.assert_array_equal(a, b[:10])
        # all-identical rows: one center, then stop
        Z = np.tile(base[:1], (40, 1))
        self.assertEqual(self.fn(Z, 10).tolist(), [0])

    def test_nearly_coincident(self):
        rng = np.random.default_rng(5)
        base = _unit32(rng.standard_normal((40, 32)))
        X = _unit32(np.repeat(base, 4, axis=0) + 1e-6 * rng.standard_normal((160, 32)))
        a = self.fn(X, 160)
        self.assertEqual(len(set(a.tolist())), len(a))      # no index chosen twice
        # first 40 picks: the sequence of distinct directions must agree with the reference
        b = reference_chosen(X, 160)
        np.testing.assert_array_equal(a[:40], b[:40])

    def test_non_contiguous_rejected(self):
        X = _unit32(np.random.default_rng(6).standard_normal((20, 8)))
        with self.assertRaises(ValueError):
            self.fn(X[:, ::2], 3)

    def test_construct_filter_fast_vs_slow_and_coverage(self):
        rng = np.random.default_rng(7)
        X = _unit32(rng.standard_normal((1200, 48)))
        k = 80
        Uf, tf = GonzalesConstruct(kernel="lazy").construct_filter(_F(k), X)
        Us, ts = GonzalesConstruct(fast=False).construct_filter(_F(k), X)
        np.testing.assert_array_equal(Uf, Us)
        np.testing.assert_allclose(tf, ts, rtol=0, atol=1e-12)
        # coverage: every row within some cap's angle (angles are exact float64 re-measurements)
        th = angle_between(X.astype(np.float64)[:, None, :], Uf[None, :, :])
        self.assertTrue((th <= tf[None, :] + 1e-12).any(axis=1).all())


@unittest.skipUnless(HAVE, "capcover_rs.gonzalez_chosen_pruned_labels not built")
class TestGonzalezPrunedLabelsSequence(TestGonzalezPruned):
    """The same nine checks through the labelled entry point (its chosen sequence)."""

    def fn(self, X, k):
        return np.asarray(capcover_rs.gonzalez_chosen_pruned_labels(X, k)[0], dtype=np.int64)


@unittest.skipUnless(HAVE, "capcover_rs.gonzalez_chosen_pruned_labels not built")
class TestGonzalezPrunedLabels(unittest.TestCase):
    """labels[i] = position of row i's best chosen direction (float32 argmax, earliest on ties)."""

    def _check(self, X, k):
        X = np.ascontiguousarray(X, dtype=np.float32)
        chosen, lab = capcover_rs.gonzalez_chosen_pruned_labels(X, k)
        chosen = np.asarray(chosen); lab = np.asarray(lab, dtype=np.int64)
        np.testing.assert_array_equal(chosen, capcover_rs.gonzalez_chosen_pruned(X, k))
        self.assertEqual(lab.shape, (len(X),))
        self.assertTrue(lab.min() >= 0 and lab.max() < len(chosen))
        ip = X.astype(np.float64) @ X[chosen].astype(np.float64).T
        got = ip[np.arange(len(X)), lab]
        # the label's inner product is the row maximum up to float32 summation noise
        self.assertLessEqual(float((ip.max(axis=1) - got).max()), 2e-6)
        # a chosen row belongs to its own cap
        np.testing.assert_array_equal(lab[chosen], np.arange(len(chosen)))
        return chosen, lab, ip

    def test_random_and_clustered(self):
        rng = np.random.default_rng(10)
        for n, d, k in [(500, 32, 40), (1500, 384, 150), (300, 17, 120), (200, 24, 199)]:
            with self.subTest(n=n, d=d, k=k):
                self._check(_unit32(rng.standard_normal((n, d))), k)
        C = rng.standard_normal((20, 48))
        self._check(_unit32(np.repeat(C, 30, axis=0) + 0.2 * rng.standard_normal((600, 48))), 60)

    def test_real_eli5(self):
        if not os.path.exists(ELI5):
            self.skipTest("ELI5 sample not available")
        X = pickle.load(open(ELI5, "rb"))[:3000].astype(np.float32)
        chosen, lab, ip = self._check(X, 300)
        # same assignment as the exact nearest-direction search, except float32-near ties
        am = ip.argmax(axis=1)
        far = np.flatnonzero(am != lab)
        self.assertTrue(np.all(ip[far, am[far]] - ip[far, lab[far]] <= 2e-6))

    def test_duplicates_ties_early_stop_and_k1(self):
        rng = np.random.default_rng(4)
        base = _unit32(rng.standard_normal((10, 32)))
        X = np.repeat(base, 5, axis=0)
        chosen, lab, _ = self._check(X, 50)
        self.assertEqual(len(chosen), 10)
        # exact copies of a center share its label
        for j, c in enumerate(chosen):
            self.assertTrue(np.all(lab[(np.arange(50) // 5) == c // 5] == j))
        chosen, lab = capcover_rs.gonzalez_chosen_pruned_labels(X, 1)
        self.assertEqual(list(chosen), [0])
        self.assertTrue(np.all(np.asarray(lab) == 0))

    def test_construct_filter_exact_coverage_float64(self):
        rng = np.random.default_rng(12)
        for X in (_unit32(rng.standard_normal((1200, 48))),
                  _unit32(np.repeat(rng.standard_normal((25, 40)), 40, axis=0) + 0.1 * rng.standard_normal((1000, 40)))):
            for k in (1, 9, 90):
                U, t = GonzalesConstruct(kernel="lazy").construct_filter(_F(k), X)
                X64 = X.astype(np.float64)
                th = angle_between(X64[:, None, :], U[None, :, :])
                self.assertTrue((th <= t[None, :] + 1e-12).any(axis=1).all())
                U0, t0 = GonzalesConstruct(fast=False).construct_filter(_F(k), X)
                self.assertEqual(U.shape, U0.shape)
                np.testing.assert_allclose(t, t0, rtol=0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
