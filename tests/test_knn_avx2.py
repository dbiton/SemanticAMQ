"""The AVX2+FMA exact top-k kernel (construct_knn_avx2.rs) against faiss IndexFlatIP.

CAPCOVER_KNN=avx2 forces the path on any CPU with AVX2+FMA (also AVX-512 ones). Index sets must
match faiss except where float32 scores are tied/near-tied (callers re-measure angles in float64).
ELI5_PKL (float32 (N, 384) pickle) overrides the default data location; the ELI5 tests skip without it.
"""
import os
import pickle
import unittest

import numpy as np

import faiss
from filters.cap_cover import cap_geometry as cg

ELI5 = os.environ.get("ELI5_PKL", "/tmp/claude-0/-home-user/faeeddb3-925b-5717-b1cb-caa0100e8a36/scratchpad/eli5/eli5_10k.pkl")
HAVE = cg._rs is not None and hasattr(cg._rs, "knn_path_py")
TOL = 2e-5


def unit(a):
    a = np.asarray(a, dtype=np.float32)
    return np.ascontiguousarray(a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-30), dtype=np.float32)


def faiss_knn(x, u, k):
    ix = faiss.IndexFlatIP(u.shape[1])
    ix.add(u)
    return ix.search(x, min(k, len(u)))


class Forced:
    def __init__(self, mode):
        self.mode = mode

    def __enter__(self):
        self.old = os.environ.get("CAPCOVER_KNN")
        os.environ["CAPCOVER_KNN"] = self.mode

    def __exit__(self, *a):
        if self.old is None:
            os.environ.pop("CAPCOVER_KNN", None)
        else:
            os.environ["CAPCOVER_KNN"] = self.old


@unittest.skipUnless(HAVE and cg._rs.knn_available_py(), "Rust knn kernels unavailable")
class TestKnnAvx2(unittest.TestCase):
    def check(self, x, u, k):
        with Forced("avx2"):
            self.assertEqual(cg._rs.knn_path_py(), "avx2")
            ip, ix = cg.nearest_directions(x, u, k)
        fi, fx = faiss_knn(x, u, k)
        kk = min(k, len(u))
        ip, ix = ip.reshape(len(x), kk), ix.reshape(len(x), kk)
        self.assertEqual(ix.shape, fx.shape)
        # scores of the returned indices (float64) equal faiss's, sorted descending
        s64 = np.einsum("nd,nkd->nk", x.astype(np.float64), u[ix].astype(np.float64))
        f64 = np.einsum("nd,nkd->nk", x.astype(np.float64), u[fx].astype(np.float64))
        self.assertLess(np.abs(np.sort(s64, 1) - np.sort(f64, 1)).max(), TOL)
        self.assertTrue((np.diff(ip, axis=1) <= 1e-7).all())
        self.assertLess(np.abs(ip - fi).max(), TOL)
        # rows whose index set differs must be float32 near-ties at the k-th boundary
        for r in np.nonzero([set(a) != set(b) for a, b in zip(ix, fx)])[0]:
            self.assertLess(abs(f64[r].min() - s64[r].min()), TOL)
            all_s = u.astype(np.float64) @ x[r].astype(np.float64)
            kth = np.sort(all_s)[-kk]
            diff = set(ix[r]) ^ set(fx[r])
            for j in diff:
                self.assertLess(abs(all_s[j] - kth), TOL)
        return ix, fx

    def test_random(self):
        rng = np.random.RandomState(0)
        x, u = unit(rng.randn(500, 384)), unit(rng.randn(300, 384))
        for k in (1, 16):
            ix, fx = self.check(x, u, k)
            self.assertTrue((ix == fx).all())

    def test_sizes_not_multiple(self):
        rng = np.random.RandomState(1)
        for d in (1, 5, 13, 33, 100, 384):
            for n, m in ((1, 1), (1, 40), (7, 17), (101, 33), (97, 1)):
                x, u = unit(rng.randn(n, d)), unit(rng.randn(m, d))
                for k in (1, 16):
                    self.check(x, u, k)

    def test_m_less_than_k(self):
        rng = np.random.RandomState(2)
        x, u = unit(rng.randn(20, 384)), unit(rng.randn(5, 384))
        self.check(x, u, 16)
        with Forced("avx2"):
            ip, ix = cg._rs.knn_ip(x, u, 16, 1)
        self.assertEqual(ix.shape, (20, 5))

    def test_duplicates_and_ties(self):
        rng = np.random.RandomState(3)
        u0 = unit(rng.randn(40, 64))
        u = np.ascontiguousarray(np.concatenate([u0, u0, u0[:7]]))
        x = unit(rng.randn(50, 64))
        for k in (1, 16):
            self.check(x, u, k)
        # exact duplicate directions: the lowest index is returned for k=1
        with Forced("avx2"):
            _, lab = cg.nearest_directions(u0, np.ascontiguousarray(np.concatenate([u0, u0])), 1)
        self.assertTrue((lab == np.arange(40)).all())
        # all-equal scores
        z = np.ascontiguousarray(np.tile(unit(rng.randn(1, 32)), (30, 1)))
        with Forced("avx2"):
            ip, ix = cg._rs.knn_ip(z[:3], z, 16, 1)
        self.assertEqual(ix.shape, (3, 16))
        self.assertTrue(((ix >= 0) & (ix < 30)).all())
        self.assertTrue(all(len(set(r)) == 16 for r in ix))

    def test_threads_match_single(self):
        rng = np.random.RandomState(4)
        x, u = unit(rng.randn(700, 384)), unit(rng.randn(900, 384))
        with Forced("avx2"):
            a = cg._rs.knn_ip(x, u, 16, 1)
            b = cg._rs.knn_ip(x, u, 16, 3)
        self.assertTrue((a[1] == b[1]).all() and (a[0] == b[0]).all())

    def test_non_finite_falls_back(self):
        rng = np.random.RandomState(5)
        x, u = unit(rng.randn(30, 384)), unit(rng.randn(50, 384))
        xb = x.copy()
        xb[3, 7] = np.nan
        ub = u.copy()
        ub[4, 0] = np.inf
        with Forced("avx2"):
            for xx, uu in ((xb, u), (x, ub)):
                ip, ix = cg.nearest_directions(xx, uu, 16)  # must not panic; faiss path
                self.assertEqual(ix.shape, (30, 16))
            with self.assertRaises(ValueError):
                cg._rs.knn_ip(xb, u, 1, 1)

    def test_forced_paths(self):
        with Forced("faiss"):
            self.assertEqual(cg._rs.knn_path_py(), "faiss")
        with Forced("avx2"):
            self.assertEqual(cg._rs.knn_path_py(), "avx2")

    @unittest.skipUnless(os.path.exists(ELI5), "ELI5 data missing")
    def test_eli5(self):
        X = unit(pickle.load(open(ELI5, "rb")))[:3000]
        rng = np.random.RandomState(6)
        for m in (50, 400, 1200):
            u = np.ascontiguousarray(X[rng.choice(len(X), m, replace=False)])
            for k in (1, 16):
                self.check(X, u, k)
        Xd = np.ascontiguousarray(np.concatenate([X[:1000], X[:1000]]))  # duplicated rows
        self.check(Xd, np.ascontiguousarray(X[:200]), 16)

    @unittest.skipUnless(os.path.exists(ELI5), "ELI5 data missing")
    def test_filter_parity(self):
        """Whole construct(): n_caps, angles, codes, directions identical, CAPCOVER_KNN=avx2 vs faiss."""
        from filters.cap_cover.cap_cover_filter import CapCoverFilter
        from filters.cap_cover.cap_quantizers import Quantize
        from filters.cap_cover.gonzales_construct import GonzalesConstruct
        C = pickle.load(open(ELI5, "rb")).astype(np.float32)[:8000]
        for q in ("FP32", "TQ2", "TQ4"):
            for b in (0.01, 0.03, 0.1):
                fs = {}
                for mode in ("avx2", "faiss"):
                    with Forced(mode):
                        f = CapCoverFilter(384, construct_method=GonzalesConstruct(), max_memory=int(b * len(C) * 384 * 4),
                                           quantize=Quantize(q), insert_method=None)
                        f.construct(C)
                        fs[mode] = f
                a, c = fs["avx2"], fs["faiss"]
                self.assertEqual(len(a.all_caps), len(c.all_caps))
                np.testing.assert_allclose(a.all_angles, c.all_angles, atol=1e-12, rtol=0)
                np.testing.assert_array_equal(a._codes[a.all_caps], c._codes[c.all_caps])
                np.testing.assert_allclose(a.all_directions, c.all_directions, atol=1e-12, rtol=0)


if __name__ == "__main__":
    unittest.main()
