import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'src'))

import unittest
import numpy as np
import torch
from src.filters.mlp_filter import MLPFilter

DIM = 32
R = 0.3  # semantic distance threshold (tau) used across most tests


def _rand(n: int, dim: int = DIM, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, dim)).astype(np.float32)


def _neighbors(v: np.ndarray, r: float, n: int = 5, seed: int = 1) -> np.ndarray:
    """Return n vectors that are within Euclidean distance r of each row of v."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((len(v), n, v.shape[1])).astype(np.float32)
    norms = np.linalg.norm(noise, axis=2, keepdims=True)
    scales = rng.uniform(0, r * 0.8, (len(v), n, 1)).astype(np.float32)
    unit_noise = noise / (norms + 1e-9) * scales
    return (v[:, None, :] + unit_noise).reshape(-1, v.shape[1])


# ---------------------------------------------------------------------------
# Initialization & Memory Allocation
# ---------------------------------------------------------------------------

class TestMLPInit(unittest.TestCase):

    def test_default_hidden_dim(self):
        f = MLPFilter(dim=DIM, distance_threshold=R)
        self.assertEqual(f.hidden_dim, f.HIDDEN_DIM)

    def test_max_memory_derives_hidden_dim(self):
        dim = 16
        # DistanceRegressor has two hidden layers (see its docstring for why
        # one isn't enough): params = h^2 + h*(dim+3) + 1, not the single
        # layer's linear h*(dim+2)+1.
        # For dim=16: h^2 + 19h + 1 <= max_params. At budget=724 bytes,
        # max_params = 724 // 4 = 181. h=6 -> 36+114+1=151 <= 181 (fits);
        # h=7 -> 49+133+1=183 > 181 (doesn't) -- so the largest fitting h is 6.
        budget = 724
        f = MLPFilter(dim=dim, distance_threshold=R, max_memory=budget)
        self.assertEqual(f.hidden_dim, 6)

    def test_max_memory_too_small_raises_value_error(self):
        # Requires at least 4 * (DIM + 3) bytes for h=1
        insufficient_budget = 10
        with self.assertRaises(ValueError):
            MLPFilter(dim=DIM, distance_threshold=R, max_memory=insufficient_budget)

    def test_distance_threshold_stored(self):
        f = MLPFilter(dim=DIM, distance_threshold=0.5)
        self.assertAlmostEqual(f.distance_threshold, 0.5)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestMLPConstruct(unittest.TestCase):

    def test_construct_initializes_and_trains_model(self):
        vecs = _rand(40)
        f = MLPFilter(dim=DIM, distance_threshold=R)
        self.assertIsNone(f.model)
        f.construct(vecs)
        self.assertIsNotNone(f.model)

    def test_empty_input(self):
        f = MLPFilter(dim=DIM, distance_threshold=R)
        f.construct(np.empty((0, DIM), dtype=np.float32))
        self.assertIsNone(f.model)

    def test_reconstruct_replaces_previous_state(self):
        f = MLPFilter(dim=DIM, distance_threshold=R)
        first_vecs = _rand(20, seed=1)
        second_vecs = _rand(20, seed=2) + 100.0

        f.construct(first_vecs)
        first_model = f.model

        f.construct(second_vecs)
        self.assertIsNot(f.model, first_model)
        self.assertGreaterEqual(sum(f.query(second_vecs)), 18)


# ---------------------------------------------------------------------------
# Query — Classifier Predictions
# ---------------------------------------------------------------------------

class TestMLPQuery(unittest.TestCase):

    def test_query_before_construct_raises(self):
        f = MLPFilter(dim=DIM, distance_threshold=R)
        with self.assertRaises(ValueError):
            f.query(np.zeros((1, DIM), dtype=np.float32))

    def test_query_empty_batch_returns_empty_list(self):
        f = MLPFilter(dim=DIM, distance_threshold=R)
        f.construct(_rand(20))
        self.assertEqual(f.query(np.empty((0, DIM), dtype=np.float32)), [])

    def test_query_returns_list_of_bools(self):
        vecs = _rand(20)
        f = MLPFilter(dim=DIM, distance_threshold=R)
        f.construct(vecs)
        results = f.query(vecs)
        self.assertIsInstance(results, list)
        self.assertTrue(all(isinstance(r, bool) for r in results))

    def test_stored_vectors_recalled(self):
        """Construction vectors should be classified as positive (True)."""
        vecs = _rand(30, seed=42)
        f = MLPFilter(dim=DIM, distance_threshold=R)
        f.construct(vecs)
        results = f.query(vecs)
        positive_predictions = sum(results)
        # Expect high recall on training construction vectors
        self.assertGreaterEqual(positive_predictions / len(vecs), 0.90)

    def test_neighbors_within_threshold_recalled(self):
        """Close neighbors well within tau should be identified as duplicates."""
        vecs = _rand(20, seed=10)
        f = MLPFilter(dim=DIM, distance_threshold=R)
        f.construct(vecs)

        neighbors = _neighbors(vecs, r=R, n=2, seed=11)
        results = f.query(neighbors)
        positive_predictions = sum(results)
        self.assertGreaterEqual(positive_predictions / len(neighbors), 0.80)

    def test_distant_vectors_rejected(self):
        """Faraway query points should be classified as negative (False)."""
        vecs = _rand(30, seed=5)
        f = MLPFilter(dim=DIM, distance_threshold=R)
        f.construct(vecs)

        distant_queries = vecs + 500.0
        results = f.query(distant_queries)
        self.assertFalse(any(results))

    def test_batch_equals_individual(self):
        """Batch query must give identical results to per-vector queries."""
        vecs = _rand(30)
        queries = _rand(10, seed=99)
        f = MLPFilter(dim=DIM, distance_threshold=R)
        f.construct(vecs)

        batch_res = f.query(queries)
        indiv_res = [f.query(q[None, :])[0] for q in queries]
        self.assertEqual(batch_res, indiv_res)


# ---------------------------------------------------------------------------
# Dynamic Operations (Unsupported)
# ---------------------------------------------------------------------------

class TestMLPUnsupportedOps(unittest.TestCase):

    def test_insert_raises_not_implemented(self):
        f = MLPFilter(dim=DIM, distance_threshold=R)
        f.construct(_rand(10))
        with self.assertRaises(NotImplementedError):
            f.insert(_rand(2))

    def test_add_raises_not_implemented(self):
        f = MLPFilter(dim=DIM, distance_threshold=R)
        f.construct(_rand(10))
        with self.assertRaises(NotImplementedError):
            f.add(_rand(2))

    def test_remove_raises_not_implemented(self):
        f = MLPFilter(dim=DIM, distance_threshold=R)
        f.construct(_rand(10))
        with self.assertRaises(NotImplementedError):
            f.remove(_rand(2))


# ---------------------------------------------------------------------------
# Memory Reporting
# ---------------------------------------------------------------------------

class TestMLPMemory(unittest.TestCase):

    def test_bytesize_before_construct_is_zero(self):
        f = MLPFilter(dim=DIM, distance_threshold=R)
        self.assertEqual(f.get_bytesize(), 0)

    def test_bytesize_after_construct(self):
        f = MLPFilter(dim=DIM, distance_threshold=R)
        f.construct(_rand(30))
        bytesize = f.get_bytesize()
        self.assertGreater(bytesize, 0)

    def test_bytesize_respects_max_memory(self):
        budget = 2000
        f = MLPFilter(dim=DIM, distance_threshold=R, max_memory=budget)
        f.construct(_rand(30))
        self.assertLessEqual(f.get_bytesize(), budget)


if __name__ == "__main__":
    unittest.main(verbosity=2)