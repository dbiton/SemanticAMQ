"""Fast soft-anneal path (Rust kernels + certified neighbour cache) against the NumPy reference."""
import os
import pickle
import unittest

import numpy as np

import filters.cap_cover.soft_anneal_construct as sac
from filters.cap_cover.cap_cover_filter import CapCoverFilter, Quantize
from filters.cap_cover.minimax_construct import MinimaxConstruct
from filters.cap_cover.soft_anneal_construct import (CapSearch, SoftAnneal, SoftAnnealMinimax, SoftState,
                                                     hard_eval)

DIM = 384


def _unit(a):
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def _clusters(n, rng, centers=20, noise=0.05):
    c = _unit(rng.normal(size=(centers, DIM)))
    return _unit(c[rng.integers(0, centers, n)] + noise * rng.normal(size=(n, DIM))).astype(np.float32)


def _eli5(n):
    path = os.environ.get("ELI5_PKL")
    if path and os.path.exists(path):
        return pickle.load(open(path, "rb")).astype(np.float32)[:n]
    return None


@unittest.skipUnless(sac.HAVE_FAST, "capcover_rs.anneal_* not built (maturin develop --release)")
class TestFastKernel(unittest.TestCase):
    def _compare(self, X, U, Ta, p_eff=30.0, topm=16, J_tol=2e-4):
        ref = SoftState(X, DIM, topm, Ta, Ta, p_eff, fast=False)
        fast = SoftState(X, DIM, topm, Ta, Ta, p_eff, fast=True)
        cand = ref.search(U)
        Jr, Gr, tr = ref.value_and_grad(U, cand)
        Jf, Gf, tf = fast.value_and_grad(U, cand)
        self.assertAlmostEqual(Jr, Jf, delta=J_tol * max(1.0, abs(Jr)))
        np.testing.assert_allclose(tf, tr, atol=3e-4)
        scale = max(np.abs(Gr).max(), 1e-12)
        self.assertLess(np.abs(Gf - Gr).max() / scale, 5e-3)
        self.assertTrue(np.isfinite(Gf).all())

    def test_random_and_extreme_temperatures(self):
        rng = np.random.default_rng(0)
        X = _clusters(1500, rng)
        U = _unit(X[rng.choice(1500, 60, replace=False)].astype(np.float64) + 0.02 * rng.normal(size=(60, DIM)))
        for Ta in (1e-4, 0.002, 0.05, 1.0):
            with self.subTest(Ta=Ta):
                self._compare(X, U, Ta)

    def test_duplicates_ties_and_coincident_vectors(self):
        rng = np.random.default_rng(1)
        base = _clusters(300, rng)
        X = np.vstack([base, base, base[:50] + 1e-6 * rng.normal(size=(50, DIM)).astype(np.float32)])
        X = _unit(X.astype(np.float64)).astype(np.float32)
        U = _unit(base[:40].astype(np.float64))
        U = np.vstack([U, U[:5]])                      # duplicate caps tie exactly
        for Ta in (1e-4, 0.05, 1.0):
            with self.subTest(Ta=Ta):
                self._compare(X, U, Ta)

    def test_empty_singleton_caps_and_k_near_n(self):
        rng = np.random.default_rng(2)
        X = _clusters(60, rng)
        for k in (50, 59, 60):                          # k near n: many singleton or empty caps
            U = _unit(X[:k].astype(np.float64) + 0.01 * rng.normal(size=(k, DIM)))
            with self.subTest(k=k):
                self._compare(X, U, 0.05, topm=16)
        far = _unit(rng.normal(size=(8, DIM)))          # caps far from every vector: usually empty lists
        self._compare(X, np.vstack([_unit(X[:10].astype(np.float64)), far]), 0.02)

    def test_eli5_data(self):
        V = _eli5(2000)
        if V is None:
            self.skipTest("ELI5_PKL not set")
        U = _unit(V[:200].astype(np.float64))
        for Ta in (1e-4, 0.05, 1.0):
            with self.subTest(Ta=Ta):
                self._compare(V, U, Ta)

    def test_gradient_matches_finite_differences(self):
        rng = np.random.default_rng(3)
        X = _clusters(400, rng)
        U = _unit(X[rng.choice(400, 20, replace=False)].astype(np.float64) + 0.03 * rng.normal(size=(20, DIM)))
        st = SoftState(X, DIM, 8, 0.08, 0.08, 30.0, fast=True)
        cand = st.search(U)
        J, G, _ = st.value_and_grad(U, cand)
        # Directional derivatives along random tangent directions of random caps. The relaxation
        # evaluates float32 inner products, so use a step large enough to dominate that noise.
        eps = 2e-3
        for _ in range(8):
            c = int(rng.integers(0, len(U)))
            v = rng.normal(size=DIM)
            v -= (v @ U[c]) * U[c]
            v /= np.linalg.norm(v)
            Up, Um = U.copy(), U.copy()
            Up[c] = _unit((U[c] + eps * v)[None])[0]
            Um[c] = _unit((U[c] - eps * v)[None])[0]
            fd = (st.value_and_grad(Up, cand, want_grad=False)[0] - st.value_and_grad(Um, cand, want_grad=False)[0]) / (2 * eps)
            self.assertAlmostEqual(float(G[c] @ v), fd, delta=0.02 * max(abs(fd), 0.05) + 5e-3)


    def test_adam_step_matches_numpy(self):
        rng = np.random.default_rng(8)
        U = _unit(rng.normal(size=(30, DIM)))
        G = rng.normal(size=(30, DIM))
        m1, v = rng.normal(size=(30, DIM)) * 0.1, np.abs(rng.normal(size=30))
        m1f, vf = m1.copy(), v.copy()
        b1, b2, lr, it = 0.9, 0.99, 0.05, 7
        Gt = G - np.sum(G * U, axis=1, keepdims=True) * U
        m1n = b1 * m1 + (1 - b1) * Gt
        vn = b2 * v + (1 - b2) * np.sum(Gt * Gt, axis=1)
        step = (m1n / (1 - b1 ** (it + 1))) / (np.sqrt(vn / (1 - b2 ** (it + 1)))[:, None] + 1e-30)
        Un = _unit(U - lr * step)
        m1n = m1n - np.sum(m1n * Un, axis=1, keepdims=True) * Un
        out = sac._rs.adam_step(G, U, m1f, vf, lr, b1, b2, 1 - b1 ** (it + 1), 1 - b2 ** (it + 1))
        np.testing.assert_allclose(out, Un, atol=1e-12)
        np.testing.assert_allclose(m1f, m1n, atol=1e-12)
        np.testing.assert_allclose(vf, vn, atol=1e-12)


@unittest.skipUnless(sac.HAVE_FAST, "capcover_rs.anneal_* not built")
class TestCapSearch(unittest.TestCase):
    def test_certified_cache_equals_full_search(self):
        rng = np.random.default_rng(4)
        X = _clusters(3000, rng, centers=40)
        U = _unit(X[rng.choice(3000, 300, replace=False)].astype(np.float64))
        s = CapSearch(X)
        s.topm(U, 16)                                    # builds the cache
        for step in range(6):
            U = _unit(U + 0.002 * (step + 1) * rng.normal(size=U.shape) / np.sqrt(DIM) * 3)
            cand, _ = s.topm(U, 16)
            ip = X @ U.astype(np.float32).T
            ref = np.argsort(-ip, axis=1)[:, :16]
            # same sets, up to float32 ties at the 16th place
            for i in range(len(X)):
                if set(cand[i]) != set(ref[i]):
                    diff = set(cand[i]) ^ set(ref[i])
                    vals = [ip[i, c] for c in diff]
                    self.assertLess(max(vals) - min(vals), 1e-5)
        self.assertGreater(s.stats["cached"], 0)

    def test_nearest_matches_hard_eval(self):
        rng = np.random.default_rng(5)
        X = _clusters(1200, rng).astype(np.float64)
        U = _unit(X[rng.choice(1200, 90, replace=False)])
        s = CapSearch(X.astype(np.float32))
        for _ in range(4):
            U = _unit(U + 0.02 * rng.normal(size=U.shape) / np.sqrt(DIM) * 3)
            Jf, tf, lf, cf = hard_eval(X, U, DIM, s)
            Jr, tr, lr, cr = hard_eval(X, U, DIM)
            self.assertAlmostEqual(Jf, Jr, places=9)
            np.testing.assert_allclose(tf, tr, atol=1e-12)


@unittest.skipUnless(sac.HAVE_FAST, "capcover_rs.anneal_* not built")
class TestFastRefine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(6)
        cls.X = _clusters(700, rng, centers=25)

    def test_refine_never_worse_than_start_and_close_to_reference(self):
        X = self.X.astype(np.float64)
        filt = CapCoverFilter(DIM, max_caps=60)
        U0, _ = MinimaxConstruct().construct_filter(filt, X)
        J0 = hard_eval(X, U0, DIM)[0]
        Jf = hard_eval(X, SoftAnneal(iters=40, fast=True).refine(X, U0), DIM)[0]
        Jr = hard_eval(X, SoftAnneal(iters=40, fast=False).refine(X, U0), DIM)[0]
        self.assertLessEqual(Jf, J0 + 1e-9)
        self.assertLess(abs(Jf - Jr), 0.05 * max(1.0, abs(Jr)))

    def test_refine_encoded_covers_every_vector(self):
        for q in (Quantize.FP32, Quantize.TQ2, Quantize.TQ4):
            with self.subTest(q=q):
                base = CapCoverFilter(DIM, max_caps=40, quantize=q, angle_bits=8, construct_method=MinimaxConstruct())
                filt = CapCoverFilter(DIM, max_caps=40, quantize=q, angle_bits=8,
                                      construct_method=SoftAnnealMinimax(anneal=SoftAnneal(iters=20, eval_every=5, fast=True)))
                base.construct(self.X)
                filt.construct(self.X)
                self.assertTrue(all(filt.query(self.X, distance_threshold=0.0)))
                self.assertLessEqual(float(np.logaddexp.reduce(sac.log_cap_measure(filt.all_angles, DIM))),
                                     float(np.logaddexp.reduce(sac.log_cap_measure(base.all_angles, DIM))) + 1e-9)
                self.assertEqual(base.get_bytesize(), filt.get_bytesize())

    def test_fast_tight_angles_equals_reference(self):
        rng = np.random.default_rng(7)
        filt = CapCoverFilter(DIM, max_caps=40, quantize=Quantize.TQ2, construct_method=MinimaxConstruct())
        X = filt._prepare_members(self.X[:300])
        U, theta = MinimaxConstruct().construct_filter(filt, X)
        _, _, home, _ = hard_eval(X, U, DIM)
        levels = np.zeros(len(U), dtype=np.int64)
        filt._train_quantizers(U)
        ref = filt._encode_cover(X, U, theta, home, levels)
        fast = sac.encode_cover_fast(filt, X, U, theta, home, levels, CapSearch(X.astype(np.float32)))
        np.testing.assert_array_equal(ref.codes, fast.codes)
        np.testing.assert_allclose(ref.angles, fast.angles, atol=1e-12)
        np.testing.assert_array_equal(ref.owner, fast.owner)


if __name__ == "__main__":
    unittest.main()
