"""Red-team tests for the Rust construction kernels (Gonzalez lazy/int8, minimax, fused angles).

Invariants attacked: every construct vector inside a cap (float64, exact), zero false negatives
for queries within tau, get_bytesize() <= max_memory, no crash. RT_FULL=1 adds the ELI5 8k matrix
(also scripted in bench_construct/redteam/fpr_cell.py)."""
import os
import threading
import unittest

import numpy as np

from filters.cap_cover import gonzales_construct as gc
from filters.cap_cover import minimax_construct as mmc
from filters.cap_cover.cap_cover_filter import CapCoverFilter
from filters.cap_cover.cap_quantizers import Quantize
from filters.cap_cover.cap_geometry import angle_between, row_angles, pair_angles
from filters.cap_cover.gonzales_construct import GonzalesConstruct, gonzalez_chosen_numpy
from filters.cap_cover.minimax_construct import MinimaxConstruct, badoiu_clarkson
from runner import CAP_CONSTRUCT_METHODS

ELI5 = "/tmp/claude-0/-home-user/faeeddb3-925b-5717-b1cb-caa0100e8a36/scratchpad/eli5/eli5_10k.pkl"
TAU = 10 ** -0.5
KERNELS = [k for k in ("numpy", "lazy", "int8")
           if k == "numpy" or (k == "lazy" and gc._rust_lazy_labels is not None)
           or (k == "int8" and gc._rust_gonzalez_labels is not None)]


class _F:
    def __init__(self, k):
        self.max_construct_caps = k


def unit(a, dtype=np.float32):
    a = np.asarray(a, dtype=np.float64)
    return (a / np.linalg.norm(a, axis=1, keepdims=True)).astype(dtype)


def clustered(n, d, seed=0, clusters=20, spread=0.3):
    r = np.random.default_rng(seed)
    c = unit(r.standard_normal((clusters, d)), np.float64)
    return unit(c[r.integers(0, clusters, n)] + spread * r.standard_normal((n, d)) / np.sqrt(d))


def assert_cover(tc, X, U, theta, slack=1e-9):
    """float64 exact: each row inside some cap."""
    X = np.asarray(X, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)
    tc.assertEqual(len(U), len(theta))
    tc.assertTrue(np.all(np.isfinite(theta)))
    tc.assertTrue(np.allclose(np.linalg.norm(U, axis=1), 1.0, atol=1e-9))
    worst = -np.inf
    for s in range(0, len(X), 256):
        ang = angle_between(X[s:s + 256, None, :], U[None, :, :])
        worst = max(worst, float((ang - theta[None, :]).min(axis=1).max()))
    tc.assertLessEqual(worst, slack)


def methods():
    out = [("gonzalez-" + k, GonzalesConstruct(kernel=k)) for k in KERNELS]
    out.append(("minimax-fast", MinimaxConstruct(fast=True)))
    out.append(("minimax-numpy", MinimaxConstruct(fast=False)))
    return out


def check_filter(tc, C, method, q, max_memory=None, max_caps=None, taus=(TAU,), near=None):
    f = CapCoverFilter(C.shape[1], construct_method=method, quantize=q, max_memory=max_memory,
                       max_caps=max_caps, insert_method=None)
    f.construct(C)
    if max_memory is not None:
        tc.assertLessEqual(f.get_bytesize(), max_memory)
    acc = np.asarray(f.query(C, 0.0), dtype=bool)
    tc.assertTrue(acc.all(), f"{(~acc).sum()} construct vectors rejected at tau=0")
    if near is not None:
        tc.assertTrue(np.asarray(f.query(near, TAU), dtype=bool).all())
    return f


# ----------------------------------------------------------------------------------- (1) numerical
class TestNumerical(unittest.TestCase):
    D = 32

    def cover_all(self, X, k):
        for name, m in methods():
            with self.subTest(method=name):
                U, th = m.construct_filter(_F(k), X)
                assert_cover(self, X, U, th)

    def test_exact_duplicates(self):
        v = unit(np.random.default_rng(0).standard_normal((1, self.D)))
        X = np.vstack([np.repeat(v, 500, axis=0), clustered(100, self.D, 1)])
        self.cover_all(X, 10)
        self.cover_all(np.repeat(v, 300, axis=0), 5)      # only duplicates

    def test_filter_many_copies(self):
        v = unit(np.random.default_rng(0).standard_normal((1, self.D)))
        X = np.vstack([np.repeat(v, 20000, axis=0), clustered(200, self.D, 2)])
        for name in ("gonzalez", "minimax"):
            with self.subTest(name):
                check_filter(self, X, CAP_CONSTRUCT_METHODS[name](), Quantize.FP32, max_caps=8)

    def test_antipodal(self):
        B = clustered(60, self.D, 3)
        X = np.vstack([B, -B])
        self.cover_all(X, 8)
        self.cover_all(X[:2] * np.array([[1], [-1]], dtype=np.float32) * 0 + np.vstack([B[:1], -B[:1]]), 1)

    def test_near_coincident(self):
        r = np.random.default_rng(4)
        base = clustered(40, self.D, 5)
        for eps in (1e-12, 1e-9, 1e-7, 1e-5):
            X = np.vstack([base, unit(base.astype(np.float64) + eps * r.standard_normal(base.shape))])
            with self.subTest(eps=eps):
                self.cover_all(X, 12)

    def test_not_exactly_unit(self):
        r = np.random.default_rng(6)
        X = clustered(300, self.D, 6).astype(np.float64)
        X *= (1 + 1e-6 * r.choice([-1, 1], (300, 1)))
        X = X.astype(np.float32)
        for name in ("gonzalez", "minimax"):
            with self.subTest(name):
                check_filter(self, X, CAP_CONSTRUCT_METHODS[name](), Quantize.FP32, max_caps=20)
        # direct construct methods (unit assumption violated by 1e-6): angles still must cover
        # in the sense of angle_between on the same vectors
        self.cover_all(X, 20)

    def test_not_unit_beyond_tolerance_is_rejected_cleanly(self):
        X = clustered(50, self.D, 7) * np.float32(1.01)
        f = CapCoverFilter(self.D, construct_method=GonzalesConstruct(), max_caps=5, insert_method=None)
        with self.assertRaises(ValueError):
            f.construct(X)

    def test_tiny_components(self):
        r = np.random.default_rng(8)
        X = np.zeros((200, self.D), dtype=np.float64)
        X[:, 0] = 1.0
        X[:, 1:] = r.standard_normal((200, self.D - 1)) * r.choice([1e-30, 1e-38, 1e-45, 1e-3], (200, 1))
        X = unit(X)
        self.cover_all(X, 6)
        check_filter(self, X, GonzalesConstruct(), Quantize.FP32, max_caps=6)

    def test_zero_vectors_documented(self):
        """Documented behaviour: a zero row violates the unit-norm precondition -> ValueError from
        CapCoverFilter._prepare_members (not a crash/segfault, not a silent bad cover)."""
        X = clustered(50, self.D, 9)
        X[7] = 0
        for name in ("gonzalez", "minimax"):
            f = CapCoverFilter(self.D, construct_method=CAP_CONSTRUCT_METHODS[name](), max_caps=5, insert_method=None)
            with self.assertRaises(ValueError):
                f.construct(X)
        # at the strategy level zero rows must at least not crash
        for name, m in methods():
            with self.subTest(method=name):
                U, th = m.construct_filter(_F(5), X)
                self.assertTrue(np.all(np.isfinite(th)))


# ------------------------------------------------------------------------------- (2) combinatorial
class TestCombinatorial(unittest.TestCase):
    D = 24

    def test_budgets(self):
        for n in (1, 2, 3, 10):
            X = clustered(n, self.D, n, clusters=max(1, n))
            for k in sorted({0, 1, max(1, n - 1), n, n + 1, 3 * n}):
                for name, m in methods():
                    with self.subTest(n=n, k=k, method=name):
                        U, th = m.construct_filter(_F(k), X)
                        if k == 0:
                            self.assertEqual(len(U), 0)
                        else:
                            assert_cover(self, X, U, th)
                            self.assertLessEqual(len(U), k)

    def test_single_cluster_and_two_points(self):
        X = clustered(80, self.D, 1, clusters=1, spread=0.05)
        for name, m in methods():
            for k in (1, 2, 40):
                with self.subTest(method=name, k=k):
                    U, th = m.construct_filter(_F(k), X)
                    assert_cover(self, X, U, th)
        P = unit(np.eye(self.D)[:2])
        for name, m in methods():
            with self.subTest(method=name):
                U, th = m.construct_filter(_F(1), P)
                assert_cover(self, P, U, th)

    def test_empty_clusters_minimax(self):
        """k >> number of distinct directions: Lloyd rounds leave empty clusters."""
        base = clustered(5, self.D, 2, clusters=5)
        X = np.repeat(base, 40, axis=0)
        for k in (6, 20, 100):
            for fast in (True, False):
                with self.subTest(k=k, fast=fast):
                    U, th = MinimaxConstruct(fast=fast).construct_filter(_F(k), X)
                    assert_cover(self, X, U, th)

    def test_filter_budget_edges(self):
        X = clustered(120, 64, 3)
        for name in ("gonzalez", "minimax"):
            for caps in (1, 2, 119, 120, 500):
                with self.subTest(method=name, max_caps=caps):
                    check_filter(self, X, CAP_CONSTRUCT_METHODS[name](), Quantize.FP32, max_caps=caps)
            with self.subTest(method=name, what="tiny max_memory"):
                f = CapCoverFilter(64, construct_method=CAP_CONSTRUCT_METHODS[name](), quantize=Quantize.TQ2,
                                   max_memory=96 + 64, insert_method=None)
                f.construct(X)
                self.assertLessEqual(f.get_bytesize(), 160)
                self.assertTrue(np.asarray(f.query(X, 0.0)).all())

    def test_empty_input(self):
        f = CapCoverFilter(self.D, construct_method=GonzalesConstruct(), max_caps=4, insert_method=None)
        f.construct(np.zeros((0, self.D), dtype=np.float32))
        self.assertEqual(list(f.query(clustered(3, self.D), TAU)), [False] * 3)


# -------------------------------------------------------------------------- (2b) odd shapes at scale
class TestShapes(unittest.TestCase):
    def test_odd_dims_large_k_all_kernels(self):
        r = np.random.default_rng(21)
        for d in (1, 5, 7, 63, 65, 127, 129, 383, 385):
            n = int(r.integers(200, 1500)); k = int(r.choice([1, n // 7, n // 2, n - 1]))
            X = unit(r.standard_normal((n, d)) * (r.random(d) ** 4 + 1e-3))     # anisotropic
            for kern in KERNELS:
                with self.subTest(d=d, n=n, k=k, kernel=kern):
                    U, th = GonzalesConstruct(kernel=kern).construct_filter(_F(k), X)
                    assert_cover(self, X, U, th)
            with self.subTest(d=d, method="minimax"):
                U, th = MinimaxConstruct(fast=True, rounds=2).construct_filter(_F(k), X)
                assert_cover(self, X, U, th)

    def test_options_adaptive_and_approx_assign(self):
        X = clustered(900, 64, 5, clusters=15)
        for kw in ({"adaptive": True}, {"approx_assign": True, "proj_dim": 16}, {"approx_assign": True, "proj_dim": 64},
                   {"rounds": 0}, {"bc_steps": 0}, {"bc_steps": 1}):
            with self.subTest(**kw):
                U, th = MinimaxConstruct(**kw).construct_filter(_F(30), X)
                assert_cover(self, X, U, th)

    def test_norm_within_tolerance(self):
        r = np.random.default_rng(22)
        X = clustered(400, 48, 3).astype(np.float64) * (1 + r.uniform(-9e-5, 9e-5, (400, 1)))
        for name in ("gonzalez", "minimax"):
            with self.subTest(name):
                check_filter(self, X.astype(np.float32), CAP_CONSTRUCT_METHODS[name](), Quantize.FP32, max_caps=25)
                check_filter(self, X.astype(np.float32), CAP_CONSTRUCT_METHODS[name](), Quantize.TQ4, max_caps=25)

    def test_nan_input_documented(self):
        """A NaN row slips past _prepare_members' norm check (abs(nan-1).max() > tol is False).
        Pre-existing, not a kernel bug; this test only documents the outcome (no hang/segfault)."""
        X = clustered(60, 32, 4); X[3, 0] = np.nan
        f = CapCoverFilter(32, construct_method=GonzalesConstruct(), max_caps=5, insert_method=None)
        try:
            f.construct(X)
            outcome = "accepted"
        except Exception as e:
            outcome = type(e).__name__
        print(f"\n[nan] filter.construct with a NaN row -> {outcome}")


# ---------------------------------------------------------------------------------- (3) near-ties
class TestTies(unittest.TestCase):
    def lattice(self, d=16, side=3):
        g = np.stack(np.meshgrid(*[np.arange(side)] * 4, indexing="ij"), -1).reshape(-1, 4).astype(np.float64)
        X = np.zeros((len(g), d))
        X[:, :4] = g - (side - 1) / 2
        X[:, 4] = 0.7
        return unit(X)

    def datasets(self):
        d = 16
        yield "lattice", self.lattice()
        yield "orthonormal", np.eye(d, dtype=np.float32)
        yield "signs", unit(np.array([[1 if (i >> j) & 1 else -1 for j in range(10)] for i in range(1024)]))
        yield "all-equal-ip", unit(np.eye(40) + 0.5)
        yield "dup+lattice", np.vstack([self.lattice(), self.lattice()])

    def test_all_kernels_valid_cover(self):
        for dn, X in self.datasets():
            for k in (1, 2, 5, 17, 40, len(X) - 1):
                if k >= len(X):
                    continue
                for kern in KERNELS:
                    with self.subTest(data=dn, k=k, kernel=kern):
                        U, th = GonzalesConstruct(kernel=kern).construct_filter(_F(k), X)
                        assert_cover(self, X, U, th)

    def test_kernel_env_filter_level(self):
        X = self.lattice()
        for kern in KERNELS:
            os.environ["CAPCOVER_GONZALEZ_KERNEL"] = kern
            try:
                check_filter(self, X, GonzalesConstruct(), Quantize.FP32, max_caps=12)
            finally:
                os.environ.pop("CAPCOVER_GONZALEZ_KERNEL", None)

    def test_chosen_distinct_and_in_range(self):
        for dn, X in self.datasets():
            for fn in (gc._rust_lazy, gc._rust_gonzalez):
                if fn is None:
                    continue
                with self.subTest(data=dn, fn=fn.__name__):
                    ch = np.asarray(fn(X, min(30, len(X) - 1)))
                    self.assertTrue(((ch >= 0) & (ch < len(X))).all())
                    self.assertEqual(len(set(ch.tolist())), len(ch))


# ----------------------------------------------------------------------------- (4) full matrix
@unittest.skipUnless(os.environ.get("RT_FULL") == "1" and os.path.exists(ELI5), "set RT_FULL=1 (slow)")
class TestEli5(unittest.TestCase):
    def test_matrix(self):
        import pickle, faiss
        V = pickle.load(open(ELI5, "rb")).astype(np.float32)
        C, Q = V[:8000], V[8000:]
        idx = faiss.IndexFlatL2(384)
        idx.add(C)
        actual = np.sqrt(np.maximum(idx.search(Q, 1)[0][:, 0], 0)) <= TAU
        for method in ("gonzalez", "minimax"):
            for q in ("FP32", "TQ2", "TQ4"):
                for b in (0.01, 0.03, 0.10):
                    with self.subTest(method=method, q=q, b=b):
                        mm = int(b * 8000 * 384 * 4)
                        f = check_filter(self, C, CAP_CONSTRUCT_METHODS[method](), Quantize[q], max_memory=mm)
                        pred = np.asarray(f.query(Q, TAU), dtype=bool)
                        self.assertEqual(int((~pred & actual).sum()), 0)


# ---------------------------------------------------------------------------------- (5) fuzz
class TestFuzz(unittest.TestCase):
    N = 320

    def rand_data(self, r):
        n = int(r.integers(2, 400)); d = int(r.choice([2, 3, 8, 17, 33, 64]))
        kind = r.integers(0, 4)
        if kind == 0:
            X = r.standard_normal((n, d))
        elif kind == 1:
            X = clustered(n, d, int(r.integers(1e9)), clusters=int(r.integers(1, 12)), spread=float(r.uniform(0.01, 1))).astype(np.float64)
        elif kind == 2:   # duplicates
            X = r.standard_normal((max(2, n // 10), d))[r.integers(0, max(2, n // 10), n)]
        else:             # near-coincident
            b = r.standard_normal((1, d)); X = b + 10.0 ** r.uniform(-12, -3) * r.standard_normal((n, d))
        return unit(X)

    def test_gonzalez_lazy_vs_numpy_centers(self):
        if gc._rust_lazy is None:
            self.skipTest("no lazy kernel")
        r = np.random.default_rng(11); worst = 0.0; diverged = 0
        for _ in range(self.N):
            X = self.rand_data(r); n = len(X)
            k = int(r.integers(1, n)) if n > 1 else 1
            a = np.asarray(gc._rust_lazy(X, k)); b = gonzalez_chosen_numpy(X, k)
            m = min(len(a), len(b))
            if not np.array_equal(a[:m], b[:m]):
                j = int(np.flatnonzero(a[:m] != b[:m])[0]); diverged += 1
                X64 = X.astype(np.float64)
                va = (X64 @ X64[a[:j]].T).max(1)[a[j]]; vb = (X64 @ X64[a[:j]].T).max(1)[b[j]]
                worst = max(worst, abs(va - vb))
        print(f"\n[fuzz] lazy vs numpy: diverged {diverged}/{self.N}, max best-ip gap at divergence {worst:.3g}")
        self.assertLess(worst, 1e-5)

    def test_all_gonzalez_kernels_cover(self):
        r = np.random.default_rng(12)
        for _ in range(self.N):
            X = self.rand_data(r); k = int(r.integers(1, len(X) + 2))
            for kern in KERNELS:
                U, th = GonzalesConstruct(kernel=kern).construct_filter(_F(k), X)
                assert_cover(self, X, U, th)

    def test_minimax_covers(self):
        r = np.random.default_rng(13)
        for _ in range(self.N // 2):
            X = self.rand_data(r); k = int(r.integers(1, len(X) + 1))
            for fast in (True, False):
                U, th = MinimaxConstruct(fast=fast, rounds=int(r.integers(0, 4))).construct_filter(_F(k), X)
                assert_cover(self, X, U, th)

    def test_bc_rust_vs_numpy(self):
        if mmc._rs_bc is None:
            self.skipTest("no kernels")
        r = np.random.default_rng(14); worst = 0.0; N = 0
        from filters.cap_cover.cap_geometry import segments
        for _ in range(self.N):
            X = self.rand_data(r).astype(np.float64); n, d = X.shape
            k = int(r.integers(1, min(n, 12) + 1))
            U = unit(X[r.choice(n, k, replace=False)], np.float64)
            lab = np.argmax(X @ U.T, 1)
            order, starts, counts = segments(lab, k)
            Xs, ls = X[order], lab[order]
            steps = int(r.integers(1, 80))
            A = badoiu_clarkson(Xs, ls, U, starts, counts, steps, fast=True)
            B = badoiu_clarkson(Xs, ls, U, starts, counts, steps, fast=False)
            for c in np.flatnonzero(counts > 0):
                mem = Xs[starts[c]:starts[c] + counts[c]]
                ta = angle_between(mem, np.repeat(A[c:c + 1], len(mem), 0)).max()
                tb = angle_between(mem, np.repeat(B[c:c + 1], len(mem), 0)).max()
                worst = max(worst, ta - tb); N += 1
        print(f"\n[fuzz] BC rust-minus-numpy final cap angle: max {worst:.3g} over {N} clusters")
        self.assertLess(worst, 0.05)

    def test_angles_vs_angle_between(self):
        r = np.random.default_rng(15); w1 = w2 = 0.0
        for _ in range(self.N):
            X = self.rand_data(r).astype(np.float64); n, d = X.shape
            U = unit(r.standard_normal((int(r.integers(1, 20)), d)), np.float64)
            if r.random() < .3:
                U[0] = X[0]
            lab = r.integers(0, len(U), n)
            w1 = max(w1, float(np.abs(row_angles(X, U, lab) - angle_between(X, U[lab])).max()))
            idx = r.integers(-1, len(U), (n, 5))
            P = pair_angles(X, U, idx)
            ref = angle_between(X[:, None, :], U[np.maximum(idx, 0)])
            self.assertTrue((np.isinf(P) == (idx < 0)).all())
            w2 = max(w2, float(np.abs(np.where(idx >= 0, P - ref, 0)).max()))
        print(f"\n[fuzz] row_angles max dev {w1:.3g}, pair_angles max dev {w2:.3g}")
        self.assertLess(max(w1, w2), 1e-12)

    def test_nearest_kernel_vs_faiss(self):
        if mmc._rs_assign is None:
            self.skipTest("no kernels")
        r = np.random.default_rng(16); bad = 0; gap = 0.0
        for _ in range(self.N):
            X = self.rand_data(r); n, d = X.shape
            U = unit(r.standard_normal((int(r.integers(1, 40)), d)))
            if r.random() < .5:
                U[: min(len(U), n)] = X[: min(len(U), n)]
            lab = np.asarray(mmc._rs_assign(X, U)[0])
            ip = X @ U.T
            ref = ip.max(1); got = ip[np.arange(n), lab]
            gap = max(gap, float((ref - got).max()))
        print(f"\n[fuzz] nearest kernel: max ip gap to argmax {gap:.3g}")
        self.assertLess(gap, 1e-6)


# --------------------------------------------------------------------------------- (6) threads
class TestThreads(unittest.TestCase):
    def test_concurrent_construct(self):
        res, errs = {}, []

        def work(i):
            try:
                X = clustered(1500, 64, i, clusters=25)
                for name in ("gonzalez", "minimax"):
                    f = check_filter(self, X, CAP_CONSTRUCT_METHODS[name](), Quantize.FP32, max_caps=40)
                    res[(i, name)] = f.all_angles.copy()
            except BaseException as e:   # noqa
                errs.append(repr(e))
        ts = [threading.Thread(target=work, args=(i,)) for i in (1, 2)]
        [t.start() for t in ts]; [t.join() for t in ts]
        self.assertEqual(errs, [])
        # same result as serial
        for i in (1, 2):
            X = clustered(1500, 64, i, clusters=25)
            f = check_filter(self, X, CAP_CONSTRUCT_METHODS["gonzalez"](), Quantize.FP32, max_caps=40)
            np.testing.assert_allclose(np.sort(res[(i, "gonzalez")]), np.sort(f.all_angles), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
