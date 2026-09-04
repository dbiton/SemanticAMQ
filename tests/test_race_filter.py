import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'src'))
import unittest

import numpy as np

from src.filters.race_filter import RaceFilter

DIM = 32
R = 0.3


def _rand(n: int, dim: int = DIM, seed: int = 0) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal((n, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

class TestRaceCalibration(unittest.TestCase):

    def test_calibration_finds_a_usable_operating_point(self):
        for tau in (0.1, 0.3, 0.6):
            f = RaceFilter(dim=DIM, distance_threshold=tau, max_memory=400_000, seed=1)
            f.construct(_rand(200, seed=2))
            self.assertIsNotNone(f.theta)
            self.assertGreater(f.theta, 0.0)
            self.assertGreaterEqual(f.num_tables, 1)

    def test_k_theory_diagnostic_is_computed_but_not_gating(self):
        """Theorem 5 (arXiv:1902.06687) gives a worst-case-sufficient K that
        is typically far larger than hashes_per_table (see _k_theory's
        docstring) -- it must be computed and exposed, but must never block
        construction on its own."""
        f = RaceFilter(dim=DIM, distance_threshold=R, max_memory=400_000,
                        hashes_per_table=8, seed=1)
        f.construct(_rand(300, seed=2))
        self.assertIsNotNone(f.k_theory)
        self.assertGreater(f.k_theory, f.hashes_per_table)

    def test_unreasonable_min_snr_raises(self):
        """An impossibly strict calibration_min_snr must fail loudly rather
        than silently deploy a non-functional filter."""
        f = RaceFilter(dim=DIM, distance_threshold=R, max_memory=400_000,
                        calibration_min_snr=1e6, seed=1)
        with self.assertRaises(RuntimeError):
            f.construct(_rand(200, seed=2))

    def test_deterministic_given_seed(self):
        f1 = RaceFilter(dim=DIM, distance_threshold=R, max_memory=400_000, seed=123)
        f2 = RaceFilter(dim=DIM, distance_threshold=R, max_memory=400_000, seed=123)
        self.assertEqual(f1.num_tables, f2.num_tables)
        np.testing.assert_array_equal(f1._projections, f2._projections)
        vecs = _rand(100, seed=7)
        f1.construct(vecs)
        f2.construct(vecs)
        self.assertEqual(f1.theta, f2.theta)
        np.testing.assert_array_equal(f1._counts, f2._counts)

    def test_empty_construct_never_matches(self):
        f = RaceFilter(dim=DIM, distance_threshold=R, max_memory=400_000, seed=1)
        f.construct(np.empty((0, DIM), dtype=np.float32))
        self.assertEqual(f.theta, float('inf'))
        self.assertEqual(f.query(_rand(5, seed=3)), [False] * 5)


# ---------------------------------------------------------------------------
# Recall / discriminative power (probabilistic, not a hard invariant)
# ---------------------------------------------------------------------------

class TestRaceRecall(unittest.TestCase):

    def test_self_recall_rate_after_construct(self):
        """Every inserted vector's own bucket is incremented by that vector
        itself in every table (a deterministic floor of >= 1 per table), so
        self-recall should hold with high (not perfect -- RACE is a
        probabilistic sketch, unlike DSBFFilter/LSBFFilter's structural
        exact-recall) probability."""
        vecs = _rand(300, seed=5)
        f = RaceFilter(dim=DIM, distance_threshold=R, max_memory=500_000, seed=9)
        f.construct(vecs)
        predicted = f.query(vecs)
        recall_rate = sum(predicted) / len(predicted)
        self.assertGreaterEqual(recall_rate, 0.9, f"self-recall rate too low: {recall_rate}")

    def test_far_queries_mostly_rejected(self):
        """Queries far from every stored vector (near-orthogonal on the unit
        sphere) should mostly be predicted negative -- the whole point of
        calibrating theta above the background level."""
        construct = _rand(300, seed=5)
        # A separate random batch on the unit sphere in dim=32 is, in
        # expectation, close to orthogonal to the construct set and far
        # outside distance_threshold=R.
        far_queries = _rand(200, seed=999)
        f = RaceFilter(dim=DIM, distance_threshold=R, max_memory=500_000, seed=9)
        f.construct(construct)
        predicted = f.query(far_queries)
        fpr = sum(predicted) / len(predicted)
        self.assertLessEqual(fpr, 0.1, f"false positive rate too high on far queries: {fpr}")

    def test_insert_then_recall(self):
        f = RaceFilter(dim=DIM, distance_threshold=R, max_memory=500_000, seed=9)
        f.construct(_rand(50, seed=1))
        probe = _rand(1, seed=77)
        f.insert(probe)
        self.assertTrue(f.query(probe)[0])


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

class TestRaceInterface(unittest.TestCase):

    def test_query_before_construct_raises(self):
        f = RaceFilter(dim=DIM, distance_threshold=R, max_memory=400_000, seed=1)
        with self.assertRaises(ValueError):
            f.query(_rand(1))

    def test_query_with_mismatched_tau_raises(self):
        f = RaceFilter(dim=DIM, distance_threshold=0.3, max_memory=400_000, seed=1)
        f.construct(_rand(50, seed=2))
        with self.assertRaises(ValueError):
            f.query(_rand(1), tau=0.9)

    def test_bytesize_before_construct_is_nominal_preallocated_size(self):
        """Unlike DSHFilter (whose Bloom filter genuinely doesn't exist
        until construct()), RaceFilter's num_tables/table_range are fixed
        at __init__ from the memory budget alone, so get_bytesize() reports
        the full nominal counts-array size even before any data arrives."""
        f = RaceFilter(dim=DIM, distance_threshold=R, max_memory=400_000, seed=1)
        expected_counts_bytes = f.num_tables * f.table_range * np.dtype(np.uint32).itemsize
        self.assertEqual(f.get_bytesize(), expected_counts_bytes + f._projections.nbytes)

    def test_bytesize_grows_after_construct(self):
        f = RaceFilter(dim=DIM, distance_threshold=R, max_memory=400_000, seed=1)
        before = f.get_bytesize()
        f.construct(_rand(50, seed=2))
        self.assertGreaterEqual(f.get_bytesize(), before)

    def test_num_tables_scales_with_memory_budget(self):
        small = RaceFilter(dim=DIM, distance_threshold=R, max_memory=20_000, seed=1)
        large = RaceFilter(dim=DIM, distance_threshold=R, max_memory=400_000, seed=1)
        self.assertLess(small.num_tables, large.num_tables)

    def test_invalid_hashes_per_table_raises(self):
        with self.assertRaises(ValueError):
            RaceFilter(dim=DIM, distance_threshold=R, max_memory=400_000, hashes_per_table=0)

    def test_invalid_num_means_raises(self):
        with self.assertRaises(ValueError):
            RaceFilter(dim=DIM, distance_threshold=R, max_memory=400_000, num_tables=4, num_means=5)


if __name__ == "__main__":
    unittest.main()
