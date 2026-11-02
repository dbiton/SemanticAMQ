"""Parity of the construct-tail Rust kernels (construct_tail.rs) with the NumPy/faiss reference.
CAPCOVER_TAIL=python selects the reference inside the filter; ELI5_PKL (float32 (N, 384) pickle)
enables the ELI5 cases."""
import os
import pickle
import unittest

import faiss
import numpy as np

import capcover_rs as rs
from filters.cap_cover.cap_cover_filter import CapCoverFilter
from filters.cap_cover.cap_geometry import pair_angles, row_angles, tail_rust
from filters.cap_cover.cap_quantizers import Quantize, _fit_scalar_codes
from filters.cap_cover.gonzales_construct import GonzalesConstruct

ELI5 = os.environ.get("ELI5_PKL", "/tmp/claude-0/-home-user/faeeddb3-925b-5717-b1cb-caa0100e8a36/scratchpad/eli5/eli5_10k.pkl")
TAU = 10 ** -0.5


def unit(a):
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def reference_settle(labels, dist, home, home_d, bound):
    """The original Python loop of CapCoverFilter._tight_angles."""
    n_caps = len(bound)
    angles = np.zeros(n_caps)
    owner = np.zeros(len(labels), dtype=np.int64)
    nearest_d = dist.min(axis=1)
    for i in np.argsort(-nearest_d, kind="stable"):
        row_l, row_d = labels[i], dist[i]
        held = np.flatnonzero(row_d <= angles[row_l])
        if len(held):
            owner[i] = row_l[held[0]]
            continue
        if home_d[i] <= angles[home[i]]:
            owner[i] = home[i]
            continue
        for j in np.argsort(row_d, kind="stable"):
            if row_d[j] <= bound[row_l[j]]:
                angles[row_l[j]] = row_d[j]
                owner[i] = row_l[j]
                break
        else:
            angles[home[i]] = home_d[i]
            owner[i] = home[i]
    return angles, owner


def settle_case(X, D, home, bound, L, neighbors=None):
    if neighbors is None:
        ip = X.astype(np.float32) @ D.astype(np.float32).T
        labels = np.argsort(-ip, axis=1, kind="stable")[:, :min(L, len(D))]
    else:
        labels = np.concatenate([home[:, None], neighbors[home]], axis=1)
    dist = pair_angles(X, D, labels)
    home_d = row_angles(X, D, home)
    b = bound.copy()
    np.maximum.at(b, home, home_d)
    return (np.ascontiguousarray(labels, dtype=np.int64), np.ascontiguousarray(dist), home, home_d, b)


class SettleParity(unittest.TestCase):
    def check(self, args):
        a0, o0 = reference_settle(*args)
        a1, o1 = rs.settle(*args)
        np.testing.assert_array_equal(a0, a1)
        np.testing.assert_array_equal(o0, o1)

    def test_random(self):
        rng = np.random.default_rng(0)
        for trial in range(20):
            n, m, d = int(rng.integers(1, 300)), int(rng.integers(1, 80)), 16
            X = unit(rng.standard_normal((n, d)))
            D = unit(X[rng.integers(0, n, m)] + 0.3 * rng.standard_normal((m, d)))
            home = rng.integers(0, m, n)                      # mostly wrong homes
            bound = rng.uniform(0, 1.5, m) * (rng.random(m) < 0.7)   # some caps with bound 0
            self.check(settle_case(X, D, home, bound, 16))

    def test_duplicates_and_ties(self):
        rng = np.random.default_rng(1)
        base = unit(rng.standard_normal((20, 8)))
        X = np.repeat(base, 5, axis=0)                          # exact duplicates -> equal distances
        D = np.repeat(base[:10], 2, axis=0)                     # duplicate caps -> ties between candidates
        home = np.arange(len(X)) // 5 * 2 % len(D)
        bound = np.full(len(D), 0.5)
        self.check(settle_case(X, D, home, bound, 6))

    def test_empty_and_singleton(self):
        rng = np.random.default_rng(2)
        X = unit(rng.standard_normal((1, 8)))
        D = unit(rng.standard_normal((1, 8)))
        self.check(settle_case(X, D, np.zeros(1, dtype=np.int64), np.zeros(1), 16))
        X = unit(rng.standard_normal((30, 8)))
        D = unit(rng.standard_normal((40, 8)))                  # more caps than members: empty caps
        self.check(settle_case(X, D, rng.integers(0, 40, 30), np.full(40, 0.8), 16))

    def test_given_neighbors_with_padding(self):
        rng = np.random.default_rng(3)
        X = unit(rng.standard_normal((200, 12)))
        D = unit(X[:50] + 0.2 * rng.standard_normal((50, 12)))
        home = rng.integers(0, 50, 200)
        nbr = np.argsort(-(D @ D.T), axis=1)[:, 1:6].astype(np.int64)
        nbr[rng.random(nbr.shape) < 0.3] = -1
        self.check(settle_case(X, D, home, np.full(50, 0.9), 0, neighbors=nbr))

    @unittest.skipUnless(os.path.exists(ELI5), "ELI5 data not found")
    def test_eli5(self):
        V = pickle.load(open(ELI5, "rb")).astype(np.float64)[:3000]
        X = unit(V)
        rng = np.random.default_rng(4)
        for m in (60, 600, 2500):
            D = unit(X[rng.choice(len(X), m, replace=False)] + 0.05 * rng.standard_normal((m, X.shape[1])))
            home = np.argmax(X @ D.T, axis=1)
            self.check(settle_case(X, D, home, np.full(m, 0.7), 16))


class KnnParity(unittest.TestCase):
    def check(self, X, D, k):
        X, D = np.ascontiguousarray(X, np.float32), np.ascontiguousarray(D, np.float32)
        ix = faiss.IndexFlatIP(D.shape[1]); ix.add(D)
        ip, lab = ix.search(X, min(k, len(D)))
        ip2, lab2 = rs.knn_ip(X, D, k, 1)
        self.assertEqual(lab.shape, lab2.shape)
        # same index sets unless a pair is within float32 summation noise of the k-th score
        for i in range(len(X)):
            if set(lab[i]) != set(lab2[i]):
                self.assertLess(abs(float(ip[i, -1]) - float(ip2[i, -1])), 1e-6)
        np.testing.assert_allclose(ip, ip2, atol=1e-6)

    @unittest.skipUnless(rs.knn_available_py(), "needs AVX-512 VNNI")
    def test_random(self):
        rng = np.random.default_rng(0)
        for n, m, d, k in [(1, 1, 8, 1), (17, 5, 8, 16), (100, 31, 384, 16), (100, 33, 384, 1),
                           (500, 3000, 384, 16), (333, 1000, 100, 4)]:
            self.check(unit(rng.standard_normal((n, d))), unit(rng.standard_normal((m, d))), k)

    @unittest.skipUnless(rs.knn_available_py(), "needs AVX-512 VNNI")
    def test_duplicates(self):
        rng = np.random.default_rng(1)
        D = unit(rng.standard_normal((50, 64)))
        self.check(np.vstack([D, D]), np.vstack([D, D[:10]]), 16)

    @unittest.skipUnless(rs.knn_available_py() and os.path.exists(ELI5), "needs AVX-512 VNNI and ELI5")
    def test_eli5(self):
        V = unit(pickle.load(open(ELI5, "rb")).astype(np.float64))
        for k in (1, 16):
            self.check(V[:2000], V[2000:6000], k)


class FitScalarParity(unittest.TestCase):
    @unittest.skipUnless(tail_rust(), "tail kernels unavailable")
    def test_fit_scalar(self):
        rng = np.random.default_rng(0)
        d = 64
        for q in (Quantize.TQ2, Quantize.TQ4):
            f = CapCoverFilter(d, quantize=q, max_memory=10 ** 6)
            U = unit(rng.standard_normal((120, d)))
            X = unit(np.repeat(U[:100], 3, axis=0) + 0.2 * rng.standard_normal((300, d)))
            X = np.vstack([X, X[:5]])                           # duplicate members
            labels = np.concatenate([np.repeat(np.arange(100), 3), np.arange(5) * 0 + 7])
            f._train_quantizers(U)
            sq = f._quantizers[f._uniform]
            os.environ["CAPCOVER_TAIL"] = "python"
            try:
                ref = _fit_scalar_codes(sq, U, X, labels)
            finally:
                os.environ.pop("CAPCOVER_TAIL")
            np.testing.assert_array_equal(ref, _fit_scalar_codes(sq, U, X, labels))   # caps 100..119 empty


class FilterParity(unittest.TestCase):
    def build(self, mode, C, q, budget):
        old = os.environ.get("CAPCOVER_TAIL")
        os.environ["CAPCOVER_TAIL"] = mode
        try:
            n, d = C.shape
            f = CapCoverFilter(d, construct_method=GonzalesConstruct(), max_memory=int(budget * n * d * 4),
                               quantize=Quantize(q), insert_method=None)
            f.construct(C)
            return f
        finally:
            if old is None:
                os.environ.pop("CAPCOVER_TAIL", None)
            else:
                os.environ["CAPCOVER_TAIL"] = old

    def compare(self, C, q, budget):
        a, b = self.build("python", C, q, budget), self.build("rust", C, q, budget)
        self.assertEqual(len(a.all_caps), len(b.all_caps))
        np.testing.assert_allclose(a.all_angles, b.all_angles, atol=1e-12, rtol=0)
        np.testing.assert_array_equal(a._codes[a.all_caps], b._codes[b.all_caps])
        np.testing.assert_allclose(a.all_directions, b.all_directions, atol=1e-12, rtol=0)

    def test_random(self):
        rng = np.random.default_rng(0)
        C = unit(np.repeat(unit(rng.standard_normal((300, 384))), 2, axis=0) + 0.05 * rng.standard_normal((600, 384))).astype(np.float32)
        C = np.vstack([C, C[:20]])                                 # exact duplicates
        for q in ("FP32", "TQ2", "TQ4"):
            for b in (0.03, 0.1):
                self.compare(C, q, b)

    @unittest.skipUnless(os.path.exists(ELI5), "ELI5 data not found")
    def test_eli5_8k(self):
        C = pickle.load(open(ELI5, "rb")).astype(np.float32)[:8000]
        for q in ("FP32", "TQ2", "TQ4"):
            for b in (0.01, 0.03, 0.1):
                self.compare(C, q, b)


if __name__ == "__main__":
    unittest.main()
