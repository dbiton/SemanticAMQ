import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'src'))
import unittest

import numpy as np

from src.filters.dsh_filter import DSHFilter

DIM = 32
R = 0.3


def _rand(n: int, dim: int = DIM, seed: int = 0) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal((n, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# Small calibration grid / projection count everywhere below to keep test
# runtime bounded -- calibration cost scales with calibration_grid and dim.
_FAST_KW = dict(num_projections_per_table=3, calibration_grid=9)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

class TestDSHCalibration(unittest.TestCase):

    def test_calibration_finds_a_usable_operating_point(self):
        for tau in (0.1, 0.3, 0.6):
            f = DSHFilter(dim=DIM, distance_threshold=tau, max_memory=200_000, seed=1, **_FAST_KW)
            self.assertGreater(f.z, 0.0)
            self.assertGreaterEqual(f.num_tables, 1)

    def test_unreasonable_min_gap_raises(self):
        """An impossibly strict calibration_min_gap must fail loudly rather
        than silently deploy a non-functional filter."""
        with self.assertRaises(RuntimeError):
            DSHFilter(dim=DIM, distance_threshold=R, max_memory=200_000,
                      calibration_min_gap=0.999, seed=1, **_FAST_KW)

    def test_deterministic_given_seed(self):
        f1 = DSHFilter(dim=DIM, distance_threshold=R, max_memory=200_000, seed=123, **_FAST_KW)
        f2 = DSHFilter(dim=DIM, distance_threshold=R, max_memory=200_000, seed=123, **_FAST_KW)
        self.assertEqual(f1.z, f2.z)
        self.assertEqual(f1.num_tables, f2.num_tables)
        np.testing.assert_array_equal(f1._projections, f2._projections)


# ---------------------------------------------------------------------------
# Exact-recall invariant
# ---------------------------------------------------------------------------

class TestDSHExactRecall(unittest.TestCase):

    def test_self_recall_after_construct(self):
        vecs = _rand(80, seed=5)
        f = DSHFilter(dim=DIM, distance_threshold=R, max_memory=300_000, seed=9, **_FAST_KW)
        f.construct(vecs)
        predicted = f.query(vecs)
        self.assertTrue(all(predicted), "false negative on exact re-query")

    def test_insert_then_recall(self):
        f = DSHFilter(dim=DIM, distance_threshold=R, max_memory=300_000, seed=9, **_FAST_KW)
        f.construct(_rand(20, seed=1))
        probe = _rand(1, seed=77)
        f.insert(probe)
        self.assertTrue(f.query(probe)[0])


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

class TestDSHInterface(unittest.TestCase):

    def test_query_before_construct_raises(self):
        f = DSHFilter(dim=DIM, distance_threshold=R, max_memory=200_000, seed=1, **_FAST_KW)
        with self.assertRaises(ValueError):
            f.query(_rand(1))

    def test_query_with_mismatched_tau_raises(self):
        f = DSHFilter(dim=DIM, distance_threshold=0.3, max_memory=200_000, seed=1, **_FAST_KW)
        f.construct(_rand(10))
        with self.assertRaises(ValueError):
            f.query(_rand(1), tau=0.9)

    def test_bytesize_includes_projection_storage_before_construct(self):
        f = DSHFilter(dim=DIM, distance_threshold=R, max_memory=200_000, seed=1, **_FAST_KW)
        self.assertEqual(f.get_bytesize(), f._projections.nbytes)

    def test_bytesize_grows_after_construct(self):
        f = DSHFilter(dim=DIM, distance_threshold=R, max_memory=200_000, seed=1, **_FAST_KW)
        before = f.get_bytesize()
        f.construct(_rand(50))
        self.assertGreaterEqual(f.get_bytesize(), before)


if __name__ == "__main__":
    unittest.main()
