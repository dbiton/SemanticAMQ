import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'src'))
import unittest

import h5py
import numpy as np

from filters.covering_lsh_fallback_filter import CoveringLSHFallbackFilter
from src.util.dataset_paths import dataset_filenames, HDF5_DATASET_KEY

DIM = 32
R = 0.3


def _rand(n: int, dim: int = DIM, seed: int = 0) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal((n, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _neighbors(v: np.ndarray, r: float, n: int = 5, seed: int = 1) -> np.ndarray:
    """Return n vectors within Euclidean distance r of each row of v."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((len(v), n, v.shape[1])).astype(np.float32)
    norms = np.linalg.norm(noise, axis=2, keepdims=True)
    scales = rng.uniform(0, r * 0.95, (len(v), n, 1)).astype(np.float32)
    unit_noise = noise / (norms + 1e-9) * scales
    return (v[:, None, :] + unit_noise).reshape(-1, v.shape[1])


def _load_small(dataset_name: str, n: int) -> np.ndarray:
    with h5py.File(dataset_filenames[dataset_name], "r") as f:
        return np.array(f[HDF5_DATASET_KEY][:n], dtype=np.float32)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestCoveringLSHInit(unittest.TestCase):

    def test_distance_threshold_stored(self):
        f = CoveringLSHFallbackFilter(dim=DIM, distance_threshold=0.5, max_memory=100_000)
        self.assertAlmostEqual(f.distance_threshold, 0.5)

    def test_default_cell_width_exceeds_twice_threshold(self):
        f = CoveringLSHFallbackFilter(dim=DIM, distance_threshold=0.3, max_memory=100_000)
        self.assertGreater(f.cell_width, 2 * 0.3)

    def test_directions_are_unit_vectors(self):
        f = CoveringLSHFallbackFilter(dim=DIM, distance_threshold=R, num_directions=5, max_memory=100_000)
        norms = np.linalg.norm(f._directions, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_num_directions_over_ceiling_raises(self):
        with self.assertRaises(ValueError):
            CoveringLSHFallbackFilter(dim=DIM, distance_threshold=R, num_directions=11, max_memory=100_000)

    def test_fanout_is_3_pow_L(self):
        f = CoveringLSHFallbackFilter(dim=DIM, distance_threshold=R, num_directions=4, max_memory=100_000)
        self.assertEqual(f._fanout, 3 ** 4)


# ---------------------------------------------------------------------------
# Zero-false-negative guarantee -- the mandatory test for this baseline.
# ---------------------------------------------------------------------------

class TestCoveringLSHZeroFalseNegatives(unittest.TestCase):
    """
    This is the highest-priority correctness property of CoveringLSHFallbackFilter:
    the task brief requires FNR == 0, verified on synthetic data (below) and on
    every real dataset present on disk, across a small tau sweep.
    """

    def test_exact_self_recall_synthetic(self):
        for tau in (0.1, 0.3, 0.6):
            vecs = _rand(150, seed=hash(tau) % 1000)
            f = CoveringLSHFallbackFilter(dim=DIM, distance_threshold=tau, num_directions=4,
                                           max_memory=2_000_000, seed=7)
            f.construct(vecs)
            predicted = f.query(vecs)
            self.assertTrue(all(predicted), f"false negative on exact re-query at tau={tau}")

    def test_near_neighbor_recall_synthetic(self):
        for tau in (0.1, 0.3, 0.6):
            vecs = _rand(100, seed=hash(tau) % 1000 + 1)
            f = CoveringLSHFallbackFilter(dim=DIM, distance_threshold=tau, num_directions=4,
                                           max_memory=2_000_000, seed=11)
            f.construct(vecs)
            probes = _neighbors(vecs, tau, n=5)
            predicted = f.query(probes)
            fn = sum(1 for p in predicted if not p)
            self.assertEqual(fn, 0, f"{fn} false negatives among within-tau probes at tau={tau}")

    def test_query_tau_exceeding_threshold_raises(self):
        f = CoveringLSHFallbackFilter(dim=DIM, distance_threshold=0.2, num_directions=4, max_memory=500_000)
        f.construct(_rand(20))
        with self.assertRaises(ValueError):
            f.query(_rand(1), tau=0.5)

    def test_query_tau_below_threshold_still_safe(self):
        """A tighter tau than the one baked in at construction is still covered
        (the cover was built for the larger threshold, so it over-approximates)."""
        f = CoveringLSHFallbackFilter(dim=DIM, distance_threshold=0.4, num_directions=4, max_memory=2_000_000, seed=3)
        vecs = _rand(80, seed=5)
        f.construct(vecs)
        predicted = f.query(vecs, tau=0.1)
        self.assertTrue(all(predicted))

    def test_zero_false_negatives_on_every_dataset_on_disk(self):
        n_sample = 250
        num_directions = 4  # kept small (3**4 = 81 keys/vector) for test runtime
        tau_values = (0.1, 0.3, 0.6)
        available = [name for name, path in dataset_filenames.items() if os.path.exists(path)]
        self.assertGreater(len(available), 0, "no datasets found on disk to test against")

        for dataset in available:
            vecs = _load_small(dataset, n_sample)
            if len(vecs) < 10:
                continue
            dim = vecs.shape[1]
            for tau in tau_values:
                f = CoveringLSHFallbackFilter(dim=dim, distance_threshold=tau, num_directions=num_directions,
                                               max_memory=8_000_000, seed=13)
                f.construct(vecs)

                exact_pred = f.query(vecs)
                exact_fn = sum(1 for p in exact_pred if not p)
                self.assertEqual(exact_fn, 0,
                                  f"{exact_fn} exact-recall false negatives on {dataset} at tau={tau}")

                probes = _neighbors(vecs[:100], tau, n=3)
                near_pred = f.query(probes)
                near_fn = sum(1 for p in near_pred if not p)
                self.assertEqual(near_fn, 0,
                                  f"{near_fn} near-neighbor false negatives on {dataset} at tau={tau}")


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

class TestCoveringLSHMemory(unittest.TestCase):

    def test_bytesize_includes_direction_vectors_before_construct(self):
        f = CoveringLSHFallbackFilter(dim=DIM, distance_threshold=R, num_directions=5, max_memory=100_000)
        self.assertEqual(f.get_bytesize(), f._directions.nbytes)

    def test_bytesize_grows_after_construct(self):
        f = CoveringLSHFallbackFilter(dim=DIM, distance_threshold=R, num_directions=4, max_memory=200_000)
        before = f.get_bytesize()
        f.construct(_rand(50))
        self.assertGreaterEqual(f.get_bytesize(), before)

    def test_respects_memory_budget(self):
        budget = 50_000
        f = CoveringLSHFallbackFilter(dim=DIM, distance_threshold=R, num_directions=4, max_memory=budget)
        f.construct(_rand(500))
        self.assertLessEqual(f.get_bytesize(), budget + f._directions.nbytes)


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

class TestCoveringLSHInterface(unittest.TestCase):

    def test_insert_is_alias_for_add(self):
        f = CoveringLSHFallbackFilter(dim=DIM, distance_threshold=R, num_directions=4, max_memory=500_000, seed=1)
        f.construct(_rand(20))
        probe = _rand(1, seed=99)
        f.insert(probe)
        self.assertTrue(f.query(probe)[0])

    def test_query_before_construct_raises(self):
        f = CoveringLSHFallbackFilter(dim=DIM, distance_threshold=R, num_directions=4, max_memory=500_000)
        with self.assertRaises(ValueError):
            f.query(_rand(1))

    def test_get_exact_matching_fpr_returns_configured_error_rate(self):
        f = CoveringLSHFallbackFilter(dim=DIM, distance_threshold=R, num_directions=4,
                                       max_memory=500_000, error_rate=0.02)
        self.assertAlmostEqual(f.get_exact_matching_fpr(), 0.02)


if __name__ == "__main__":
    unittest.main()
