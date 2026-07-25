import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'src'))
import unittest
import numpy as np
from src.filters.kdball_filter import KDBallFilter

DIM = 32
R = 0.3  # semantic distance threshold used across most tests


def _rand(n: int, dim: int = DIM, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, dim)).astype(np.float32)


def _clustered(n_clusters: int, per_cluster: int, dim: int = DIM,
               spread: float = 0.05, sep: float = 15.0, seed: int = 0) -> np.ndarray:
    """Generate vectors with a clear cluster structure."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, dim)) * sep
    parts = [rng.standard_normal((per_cluster, dim)) * spread + c for c in centers]
    return np.vstack(parts).astype(np.float32)


def _neighbors(v: np.ndarray, r: float, n: int = 5, seed: int = 1) -> np.ndarray:
    """Return n vectors that are within Euclidean distance r of each row of v."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((len(v), n, v.shape[1])).astype(np.float32)
    norms = np.linalg.norm(noise, axis=2, keepdims=True)
    scales = rng.uniform(0, r * 0.95, (len(v), n, 1)).astype(np.float32)
    unit_noise = noise / (norms + 1e-9) * scales
    return (v[:, None, :] + unit_noise).reshape(-1, v.shape[1])


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestKDBallInit(unittest.TestCase):

    def test_max_spheres_stored(self):
        f = KDBallFilter(dim=DIM, distance_threshold=R, max_spheres=10)
        self.assertEqual(f.max_spheres, 10)

    def test_max_memory_derives_sphere_count(self):
        dim = 16
        # bytes_per_sphere = (16 + 1) * 4 = 68
        # gamma is hardcoded to 4.0 in _calculate_max_spheres
        # adjusted bytes = 68 * 4.0 = 272
        # max_memory = 19000 bytes
        # max_spheres = 19000 // 272 = 69
        budget = 19000
        f = KDBallFilter(dim=dim, distance_threshold=R, max_memory=budget)
        self.assertEqual(f.max_spheres, 69)

    def test_distance_threshold_stored(self):
        f = KDBallFilter(dim=DIM, distance_threshold=0.5, max_spheres=5)
        self.assertAlmostEqual(f.distance_threshold, 0.5)


# ---------------------------------------------------------------------------
# KD-Tree Splitting Logic
# ---------------------------------------------------------------------------

class TestKDBallWidestGap(unittest.TestCase):

    def test_finds_correct_gap_and_axis(self):
        f = KDBallFilter(dim=2, distance_threshold=R, max_spheres=5)

        # vectors have the largest spread on axis 1 (Y axis)
        # Y range = 13.0 - 0.0 = 13.0
        # split value = (0.0 + 13.0) / 2 = 6.5
        vecs = np.array([
            [1.0, 0.0],
            [1.5, 2.0],
            [0.5, 12.0],
            [1.2, 13.0]
        ], dtype=np.float32)

        best_margin, best_axis, best_split_val = f._find_widest_margin(vecs)

        self.assertEqual(best_axis, 1)
        self.assertAlmostEqual(best_margin, 13.0)
        self.assertAlmostEqual(best_split_val, 6.5)

    def test_no_split_for_single_vector(self):
        f = KDBallFilter(dim=2, distance_threshold=R, max_spheres=5)
        vecs = np.array([[1.0, 1.0]], dtype=np.float32)
        
        best_margin, best_axis, best_split_val = f._find_widest_margin(vecs)
        
        self.assertIsNone(best_axis)
        self.assertIsNone(best_split_val)
        self.assertEqual(best_margin, -1)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestKDBallConstruct(unittest.TestCase):

    def test_produces_correct_number_of_spheres(self):
        f = KDBallFilter(dim=DIM, distance_threshold=R, max_spheres=8)
        f.construct(_rand(100))
        self.assertTrue(f._constructed)
        self.assertEqual(len(f.all_spheres), 8)
        self.assertEqual(len(f.all_radiuses), 8)

    def test_trivial_case_fewer_vectors_than_spheres(self):
        """Construction succeeds even when there are fewer vectors than available spheres."""
        vecs = _rand(3)
        f = KDBallFilter(dim=DIM, distance_threshold=R, max_spheres=20)
        f.construct(vecs)
        self.assertTrue(f._constructed)
        self.assertLessEqual(len(f.all_spheres), len(vecs))
        self.assertTrue(np.all(f.all_radiuses >= R - 1e-5))

    def test_radii_at_least_distance_threshold(self):
        """Every sphere radius must be >= distance_threshold."""
        f = KDBallFilter(dim=DIM, distance_threshold=R, max_spheres=10)
        f.construct(_rand(200))
        self.assertTrue(np.all(f.all_radiuses >= R - 1e-5),
                        f"Some radii < distance_threshold: {f.all_radiuses[f.all_radiuses < R]}")


# ---------------------------------------------------------------------------
# Query — semantic correctness
# ---------------------------------------------------------------------------

class TestKDBallQuery(unittest.TestCase):

    def test_stored_vectors_always_recalled(self):
        """Every vector used in construct() must be returned True by query()."""
        vecs = _rand(60)
        f = KDBallFilter(dim=DIM, distance_threshold=R, max_spheres=8)
        f.construct(vecs)
        results = f.query(vecs)
        fn = sum(1 for r in results if not r)
        self.assertEqual(fn, 0, f"{fn} stored vectors produced false negatives")

    def test_neighbors_within_threshold_recalled(self):
        """Vectors within distance_threshold of a stored vector must be found."""
        vecs = _rand(20)
        f = KDBallFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        f.construct(vecs)
        neighbors = _neighbors(vecs, r=R, n=3)
        results = f.query(neighbors)
        fn = sum(1 for r in results if not r)
        self.assertEqual(fn, 0,
                         f"{fn}/{len(neighbors)} close neighbors not recalled")

    def test_distant_vectors_rejected(self):
        """Vectors far beyond distance_threshold of all stored vectors must not match."""
        vecs = _rand(30)
        f = KDBallFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        f.construct(vecs)
        # shift by 1000 units, enormously beyond any sphere radius
        self.assertFalse(any(f.query(vecs + 1000.0)))

    def test_query_before_construct_raises(self):
        f = KDBallFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        with self.assertRaises(ValueError):
            f.query(np.zeros((1, DIM), dtype=np.float32))

    def test_query_returns_list_of_bools(self):
        vecs = _rand(20)
        f = KDBallFilter(dim=DIM, distance_threshold=R, max_spheres=4)
        f.construct(vecs)
        results = f.query(vecs)
        self.assertIsInstance(results, list)
        self.assertTrue(all(isinstance(r, (bool, np.bool_)) for r in results))

    def test_batch_equals_individual(self):
        """Batch query must give identical results to per-vector queries."""
        vecs = _rand(50)
        queries = _rand(15, seed=99)
        f = KDBallFilter(dim=DIM, distance_threshold=R, max_spheres=6)
        f.construct(vecs)
        self.assertEqual(f.query(queries),
                         [f.query(q[None, :])[0] for q in queries])


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

class TestKDBallMemory(unittest.TestCase):
    dim = 16

    def test_bytesize_before_construct_is_zero(self):
        self.assertEqual(KDBallFilter(dim=self.dim, distance_threshold=R, max_spheres=10).get_bytesize(), 0)

    def test_bytesize_after_construct(self):
        k, dim = 6, self.dim
        f = KDBallFilter(dim=dim, distance_threshold=R, max_spheres=k)
        f.construct(_rand(100, dim))
        # k centroids * dim float32 + k radiuses * float32
        self.assertEqual(f.get_bytesize(), k * dim * 4 + k * 4)


# ---------------------------------------------------------------------------
# Static filter interface
# ---------------------------------------------------------------------------

class TestKDBallStaticInterface(unittest.TestCase):

    def setUp(self):
        self.f = KDBallFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        self.f.construct(_rand(40))
        self.dummy = np.zeros((1, DIM), dtype=np.float32)

    def test_insert_exists(self):
        self.assertTrue(callable(self.f.insert))

    def test_add_exists(self):
        self.assertTrue(callable(self.f.add))

    def test_remove_exists(self):
        self.assertTrue(callable(self.f.remove))

    def test_exact_matching_fpr_is_zero(self):
        self.assertEqual(self.f.get_exact_matching_fpr(), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)