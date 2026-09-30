import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'src'))
import unittest

import numpy as np

from src.filters.pacuk_lsh_filter import PacukLSHFilter
from filters.filter import InsufficientMemoryError

DIM = 16
R = 0.2


def _rand(n: int, dim: int = DIM, seed: int = 0) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal((n, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _at_distance(base: np.ndarray, directions: np.ndarray, r: float) -> np.ndarray:
    """base + r * unit(directions): points at L2 distance exactly r."""
    d = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    return (base + r * d).astype(np.float32)


class TestPacukNoFalseNegatives(unittest.TestCase):

    def _assert_all_found(self, f, queries):
        self.assertTrue(all(f.query(queries)), "false negative within tau")

    def test_random_directions_at_tau(self):
        base = _rand(100, seed=1)
        f = PacukLSHFilter(DIM, R, num_hashes=5, seed=3)
        f.construct(base)
        dirs = np.random.default_rng(2).standard_normal(base.shape)
        self._assert_all_found(f, _at_distance(base, dirs, R))

    def test_adversarial_direction_along_each_hash_vector(self):
        """z along sign(v) maximises |<z, v>| at tau*||v||_1/sqrt(d), which is
        tau*sqrt(d) for Rademacher v -- the exact case Observation 3's width is
        sized for. z along v itself is the Cauchy-Schwarz extreme."""
        base = _rand(40, seed=4)
        for distribution in ("rademacher", "uniform"):
            f = PacukLSHFilter(DIM, R, num_hashes=4, distribution=distribution, seed=5)
            f.construct(base)
            for v in f._projections:
                for direction in (v, -v, np.sign(v), -np.sign(v)):
                    dirs = np.broadcast_to(direction, base.shape)
                    self._assert_all_found(f, _at_distance(base, dirs, R))

    def test_probe_mode_and_bloom_path(self):
        base = _rand(80, seed=6)
        dirs = np.random.default_rng(7).standard_normal(base.shape)
        queries = _at_distance(base, dirs, R)
        for kw in (dict(replicate="probe"), dict(max_memory=200_000),
                   dict(replicate="probe", max_memory=50_000)):
            f = PacukLSHFilter(DIM, R, num_hashes=5, seed=8, **kw)
            f.construct(base)
            self._assert_all_found(f, queries)

    def test_insert_then_recall(self):
        f = PacukLSHFilter(DIM, R, num_hashes=4, seed=9)
        f.construct(_rand(20, seed=1))
        probe = _rand(1, seed=77)
        f.insert(probe)
        self._assert_all_found(f, _at_distance(probe, _rand(1, seed=78), R))


class TestPacukBehaviour(unittest.TestCase):

    def test_store_and_probe_agree(self):
        base, queries = _rand(60, seed=10), _rand(200, seed=11)
        a = PacukLSHFilter(DIM, R, num_hashes=5, replicate="store", seed=12)
        b = PacukLSHFilter(DIM, R, num_hashes=5, replicate="probe", seed=12)
        a.construct(base)
        b.construct(base)
        self.assertEqual(a.query(queries), b.query(queries))

    def test_rejects_far_points_in_low_dimension(self):
        """In low d the sqrt(d) width is small enough to separate far points --
        guards against a filter that trivially says yes to everything."""
        dim = 4
        base, queries = _rand(50, dim=dim, seed=13), _rand(300, dim=dim, seed=14)
        f = PacukLSHFilter(dim, 0.02, num_hashes=6, seed=15)
        f.construct(base)
        self.assertLess(np.mean(f.query(queries)), 0.5)

    def test_deterministic_given_seed(self):
        a = PacukLSHFilter(DIM, R, seed=123)
        b = PacukLSHFilter(DIM, R, seed=123)
        np.testing.assert_array_equal(a._projections, b._projections)

    def test_key_counts(self):
        f = PacukLSHFilter(DIM, R, num_hashes=4, replicate="store")
        self.assertEqual((f.keys_per_point(), f.probes_per_query()), (81, 1))
        f = PacukLSHFilter(DIM, R, num_hashes=4, replicate="probe")
        self.assertEqual((f.keys_per_point(), f.probes_per_query()), (1, 81))


class TestPacukInterface(unittest.TestCase):

    def test_query_before_construct_raises(self):
        with self.assertRaises(ValueError):
            PacukLSHFilter(DIM, R).query(_rand(1))

    def test_query_with_mismatched_tau_raises(self):
        f = PacukLSHFilter(DIM, R, num_hashes=3)
        f.construct(_rand(10))
        with self.assertRaises(ValueError):
            f.query(_rand(1), tau=0.9)

    def test_too_small_budget_raises(self):
        with self.assertRaises(InsufficientMemoryError):
            PacukLSHFilter(DIM, R, num_hashes=6, max_memory=1_000).construct(_rand(100))

    def test_bytesize_within_budget(self):
        f = PacukLSHFilter(DIM, R, num_hashes=4, max_memory=100_000)
        f.construct(_rand(50))
        self.assertLessEqual(f.get_bytesize(), 100_000)

    def test_invalid_arguments_raise(self):
        with self.assertRaises(ValueError):
            PacukLSHFilter(DIM, R, num_hashes=0)
        with self.assertRaises(ValueError):
            PacukLSHFilter(DIM, R, distribution="gaussian")
        with self.assertRaises(ValueError):
            PacukLSHFilter(DIM, R, replicate="both")


if __name__ == "__main__":
    unittest.main()
