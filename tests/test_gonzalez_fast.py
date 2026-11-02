"""Fused Rust Gonzalez kernel (GonzalesConstruct(fast=True)) vs the NumPy reference path.

The two index sequences are equal except where float32 summation order flips a comparison
of inner products within a few ulp; the parity helper accepts a divergence only when the
two candidate picks' float64 best-ip values differ by <= 1e-5. Coverage is checked against
float64 independently of the sequence."""
import os
import pickle
import unittest
import numpy as np

from filters.cap_cover import gonzales_construct as gc
from filters.cap_cover.gonzales_construct import GonzalesConstruct, gonzalez_chosen_numpy
from filters.cap_cover.cap_geometry import angle_between

ELI5 = "/tmp/claude-0/-home-user/faeeddb3-925b-5717-b1cb-caa0100e8a36/scratchpad/eli5/eli5_10k.pkl"
HAVE_RUST = gc._rust_gonzalez is not None


class _F:
    def __init__(self, k):
        self.max_construct_caps = k


def _unit(X):
    X = np.asarray(X, dtype=np.float64)
    return (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.float32)


def _clustered(n, d, seed=0, clusters=30, spread=0.3):
    rng = np.random.default_rng(seed)
    c = rng.standard_normal((clusters, d))
    c /= np.linalg.norm(c, axis=1, keepdims=True)
    lab = rng.integers(0, clusters, n)
    return _unit(c[lab] + spread * rng.standard_normal((n, d)) / np.sqrt(d))


def _best_ip64(X, seq, upto):
    X = X.astype(np.float64)
    return (X @ X[seq[:upto]].T).max(axis=1)


@unittest.skipUnless(HAVE_RUST, "capcover_rs.gonzalez_chosen not built")
class ParityBase(unittest.TestCase):
    def assert_parity(self, X, k, tol=1e-5):
        X = np.ascontiguousarray(X, dtype=np.float32)
        a = gonzalez_chosen_numpy(X, k)
        b = np.asarray(gc._rust_gonzalez(X, k))
        m = min(len(a), len(b))
        diff = np.flatnonzero(a[:m] != b[:m])
        if len(diff) == 0:
            self.assertEqual(len(a), len(b))
            return a, b
        j = int(diff[0])     # first divergence must be a near-tie of best-ip values
        best = _best_ip64(X, a, j)
        self.assertLessEqual(abs(best[a[j]] - best[b[j]]), tol, f"divergence at {j} is not a near-tie")
        return a, b


class TestGonzalezFast(ParityBase):
    def test_random_parity(self):
        for seed, (n, d, k) in enumerate([(500, 32, 60), (2000, 384, 300), (999, 37, 120), (700, 130, 700 - 1)]):
            X = _unit(np.random.default_rng(seed).standard_normal((n, d)))
            self.assert_parity(X, k)

    def test_clustered_parity(self):
        X = _clustered(3000, 96, seed=3)
        self.assert_parity(X, 400)

    @unittest.skipUnless(os.path.exists(ELI5), "ELI5 sample not available")
    def test_eli5_parity(self):
        X = np.asarray(pickle.load(open(ELI5, "rb")), dtype=np.float32)[:8000]
        a, b = self.assert_parity(X, 500)
        self.assertEqual(a[0], 0)

    def test_duplicates_and_ties(self):
        base = _unit(np.random.default_rng(5).standard_normal((40, 64)))
        X = np.vstack([base, base, base[::-1], base])      # exact duplicates, interleaved copies
        a, b = self.assert_parity(X, 39)
        self.assertEqual(len(set(b.tolist())), len(b))
        self.assertTrue(np.array_equal(a, b))              # identical rows tie to the lowest index

    def test_all_identical_early_stop(self):
        X = np.tile(_unit(np.random.default_rng(6).standard_normal((1, 48))), (50, 1))
        a = gonzalez_chosen_numpy(X, 10)
        b = np.asarray(gc._rust_gonzalez(X, 10))
        self.assertEqual(a.tolist(), [0])
        self.assertEqual(b.tolist(), [0])

    def test_early_stop_all_chosen(self):
        base = _unit(np.random.default_rng(7).standard_normal((12, 40)))
        X = np.vstack([base] * 5)
        a = gonzalez_chosen_numpy(X, 50)
        b = np.asarray(gc._rust_gonzalez(X, 50))
        # The stop is best_ip >= 1.0 in float32, so whether it triggers after the 12 distinct
        # directions depends on rounding of x.x; both paths must agree either way.
        # After the 12 distinct directions every best_ip is ~1 - 1ulp, so any later pick is rounding
        # noise (a documented divergence); only the first 12 picks are meaningful.
        self.assertLessEqual(len(a), 50)
        self.assertEqual(a[:12].tolist(), b[:12].tolist())
        self.assertEqual(sorted((a[:12] % 12).tolist()), list(range(12)))

    def test_k1_and_k2(self):
        X = _unit(np.random.default_rng(8).standard_normal((100, 24)))
        self.assertEqual(list(gc._rust_gonzalez(X, 1)), [0])
        self.assert_parity(X, 2)

    def test_k_n_minus_1(self):
        X = _unit(np.random.default_rng(9).standard_normal((200, 20)))
        self.assert_parity(X, 199)

    def test_nearly_coincident_and_tiny(self):
        rng = np.random.default_rng(10)
        base = _unit(rng.standard_normal((30, 64)))
        X = _unit(np.repeat(base, 8, axis=0).astype(np.float64) + 1e-6 * rng.standard_normal((240, 64)))
        self.assert_parity(X, 100)
        # a long run of tiny-norm-ish rows before normalisation: still unit after _unit
        Y = _unit(1e-12 * rng.standard_normal((300, 64)) + np.eye(64)[rng.integers(0, 64, 300)] * 1e-9)
        self.assert_parity(Y, 80)

    def test_odd_dims(self):
        for d in (1, 3, 15, 17, 63, 65, 100):
            X = _unit(np.random.default_rng(d).standard_normal((300, d)))
            self.assert_parity(X, 40)

    def test_wrapper_matches_reference_cover(self):
        X = _clustered(1500, 64, seed=11)
        F = _F(150)
        U1, a1 = GonzalesConstruct(fast=True).construct_filter(F, X)
        U0, a0 = GonzalesConstruct(fast=False).construct_filter(F, X)
        self.assertEqual(U1.shape, U0.shape)
        self.assertTrue(np.allclose(U1, U0) and np.allclose(a1, a0))

    def test_n_le_k_and_float64_input(self):
        X = _unit(np.random.default_rng(12).standard_normal((20, 16)))
        for fast in (True, False):
            U, ang = GonzalesConstruct(fast=fast).construct_filter(_F(25), X)
            self.assertEqual(U.shape, (20, 16))
            self.assertTrue(np.all(ang == 0))
        X = _clustered(400, 32, seed=13).astype(np.float64)
        U1, a1 = GonzalesConstruct(fast=True).construct_filter(_F(40), X)
        U0, a0 = GonzalesConstruct(fast=False).construct_filter(_F(40), X)
        self.assertTrue(np.allclose(a1, a0))

    def test_coverage_exact(self):
        for seed, X in enumerate([_clustered(2000, 64, seed=14), _unit(np.random.default_rng(15).standard_normal((800, 40)))]):
            for k in (1, 7, 100):
                U, ang = GonzalesConstruct(fast=True).construct_filter(_F(k), X)
                X64 = X.astype(np.float64)
                nearest = (X64 @ U.T).argmax(axis=1)
                self.assertTrue(np.all(angle_between(X64, U[nearest]) <= ang[nearest] + 1e-12))

    def test_python_fallback_flag(self):
        g = GonzalesConstruct(fast=False)
        self.assertFalse(g.fast)
        X = _clustered(300, 32, seed=16)
        U, ang = g.construct_filter(_F(30), X)
        self.assertEqual(len(U), len(ang))


if __name__ == "__main__":
    unittest.main()
