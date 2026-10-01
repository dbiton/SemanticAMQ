import unittest

import numpy as np

from filters.ivf_radii_filter import IVFRadiiFilter, InsufficientMemoryError

DIM = 32
N = 600
TAU = 0.3


def _rand(n: int = N, dim: int = DIM, seed: int = 0) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal((n, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _clustered(n: int = N, dim: int = DIM, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((10, dim))
    x = centers[rng.integers(0, len(centers), n)] + 0.15 * rng.standard_normal((n, dim))
    x = x.astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _near(x: np.ndarray, scale: float, seed: int) -> np.ndarray:
    """Perturbations of x, renormalized: a mix of in- and out-of-tau queries."""
    rng = np.random.default_rng(seed)
    q = x + scale * rng.standard_normal(x.shape).astype(np.float32)
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def _true_nn(x: np.ndarray, q: np.ndarray) -> np.ndarray:
    d2 = (q ** 2).sum(1)[:, None] + (x ** 2).sum(1)[None, :] - 2.0 * q @ x.T
    return np.sqrt(np.maximum(d2, 0.0)).min(axis=1)


def _built(x: np.ndarray, budget_lists: int) -> IVFRadiiFilter:
    probe = IVFRadiiFilter(DIM, TAU, max_memory=1)
    budget = probe._header_bytes() + budget_lists * probe.bytes_per_list()
    f = IVFRadiiFilter(DIM, TAU, max_memory=budget)
    f.construct(x)
    return f


class TestIVFRadiiZeroFN(unittest.TestCase):

    def _assert_no_fn(self, f: IVFRadiiFilter, x: np.ndarray, q: np.ndarray) -> None:
        truth = _true_nn(x, q) <= TAU
        self.assertGreater(truth.sum(), 0, "workload has no positives")
        predicted = np.array(f.query(q))
        self.assertFalse((truth & ~predicted).any(), f"{(truth & ~predicted).sum()} false negatives")

    def test_zero_fn_across_budgets(self):
        x = _clustered()
        q = np.vstack([_near(x, s, seed=i) for i, s in enumerate((0.01, 0.05, 0.1))])
        for lists in (1, 5, 30, 150, N):
            with self.subTest(lists=lists):
                self._assert_no_fn(_built(x, lists), x, q)

    def test_zero_fn_on_construct_set(self):
        x = _rand()
        f = _built(x, 40)
        self.assertTrue(all(f.query(x)))

    def test_zero_fn_after_insert(self):
        x = _clustered(seed=1)
        f = _built(x[:300], 20)
        bytes_before = f.get_bytesize()
        f.insert(x[300:])
        self.assertEqual(f.get_bytesize(), bytes_before, "insert must not grow the footprint")
        self._assert_no_fn(f, x, _near(x, 0.05, seed=9))

    def test_surface_distance_lower_bounds_true_nn(self):
        x = _rand(seed=2)
        f = _built(x, 25)
        q = _rand(200, seed=3)
        self.assertTrue((f.surface_distance(q) <= _true_nn(x, q) + 1e-4).all())


class TestIVFRadiiSelectivity(unittest.TestCase):

    def test_more_lists_reject_more_far_queries(self):
        x = _clustered(seed=4)
        far = _rand(400, seed=5)
        far = far[_true_nn(x, far) > TAU]
        rates = [np.mean(_built(x, lists).query(far)) for lists in (2, 40, N)]
        self.assertGreaterEqual(rates[0], rates[1])
        self.assertGreaterEqual(rates[1], rates[2])
        self.assertEqual(rates[2], 0.0, "one list per vector is exact")


class TestIVFRadiiMemory(unittest.TestCase):

    def test_respects_budget(self):
        x = _rand()
        for budget in (2_000, 20_000, 200_000):
            with self.subTest(budget=budget):
                f = IVFRadiiFilter(DIM, TAU, max_memory=budget)
                f.construct(x)
                self.assertLessEqual(f.get_bytesize(), budget)

    def test_budget_sets_nlist(self):
        f = _built(_rand(), 37)
        self.assertLessEqual(f.nlist, 37)
        self.assertGreaterEqual(f.nlist, 30, "empty lists only, never a smaller budget")

    def test_nlist_capped_at_n(self):
        f = IVFRadiiFilter(DIM, TAU, max_memory=10_000_000)
        f.construct(_rand(50))
        self.assertEqual(f.nlist, 50)

    def test_bytesize_matches_layout(self):
        f = _built(_rand(), 20)
        self.assertEqual(f.get_bytesize(), f._header_bytes() + f.nlist * f.bytes_per_list())

    def test_too_small_budget_raises(self):
        f = IVFRadiiFilter(DIM, TAU, max_memory=100)
        with self.assertRaises(InsufficientMemoryError):
            f.construct(_rand())


class TestIVFRadiiInterface(unittest.TestCase):

    def test_query_before_construct_raises(self):
        with self.assertRaises(ValueError):
            IVFRadiiFilter(DIM, TAU, max_memory=10_000).query(_rand(1))

    def test_empty_construct_then_insert(self):
        f = IVFRadiiFilter(DIM, TAU, max_memory=10_000)
        f.construct(np.zeros((0, DIM), dtype=np.float32))
        self.assertEqual(f.get_bytesize(), 0)
        self.assertEqual(f.query(_rand(3)), [False, False, False])
        x = _rand(20, seed=6)
        f.insert(x)
        self.assertTrue(all(f.query(x)))

    def test_no_budget_uses_sqrt_n(self):
        f = IVFRadiiFilter(DIM, TAU)
        f.construct(_rand(400))
        self.assertLessEqual(f.nlist, 20)

    def test_explicit_tau_overrides_default(self):
        x = _rand()
        f = _built(x, N)
        q = _near(x[:50], 0.05, seed=7)
        self.assertEqual(f.query(q, tau=0.0), (f.surface_distance(q) <= f._COVERAGE_EPSILON).tolist())

    def test_same_seed_same_result(self):
        x = _clustered(seed=8)
        a, b = _built(x, 15), _built(x, 15)
        np.testing.assert_array_equal(a.radii, b.radii)


if __name__ == "__main__":
    unittest.main()
