import unittest

import numpy as np

from src.filters.kmeb_filter import KMEBFilter

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
    # Scale each noise vector to a random length in (0, r)
    scales = rng.uniform(0, r * 0.95, (len(v), n, 1)).astype(np.float32)
    unit_noise = noise / (norms + 1e-9) * scales
    # Return shape (len(v)*n, dim)
    return (v[:, None, :] + unit_noise).reshape(-1, v.shape[1])


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestKMEBInit(unittest.TestCase):

    def test_max_balls_stored(self):
        f = KMEBFilter(dim=DIM, distance_threshold=R, max_balls=10)
        self.assertEqual(f.max_balls, 10)

    def test_max_memory_derives_ball_count(self):
        dim = 16
        budget = (dim + 1) * 4 * 7      # exactly 7 balls
        f = KMEBFilter(dim=dim, distance_threshold=R, max_memory=budget)
        self.assertEqual(f.max_balls, 7)

    def test_distance_threshold_stored(self):
        f = KMEBFilter(dim=DIM, distance_threshold=0.5, max_balls=5)
        self.assertAlmostEqual(f.distance_threshold, 0.5)

    def test_neither_size_param_raises(self):
        with self.assertRaises(ValueError):
            KMEBFilter(dim=DIM, distance_threshold=R)

    def test_both_size_params_raise(self):
        with self.assertRaises(ValueError):
            KMEBFilter(dim=DIM, distance_threshold=R, max_memory=1000, max_balls=5)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestKMEBConstruct(unittest.TestCase):

    def test_produces_correct_number_of_balls(self):
        f = KMEBFilter(dim=DIM, distance_threshold=R, max_balls=8)
        f.construct(_rand(100))
        self.assertTrue(f._constructed)
        self.assertEqual(len(f._centroids), 8)
        self.assertEqual(len(f._radii), 8)

    def test_empty_input(self):
        f = KMEBFilter(dim=DIM, distance_threshold=R, max_balls=5)
        f.construct(np.empty((0, DIM), dtype=np.float32))
        self.assertTrue(f._constructed)
        self.assertEqual(len(f._centroids), 0)

    def test_trivial_case_fewer_vectors_than_balls(self):
        """When n <= max_balls each vector becomes its own centroid."""
        vecs = _rand(3)
        f = KMEBFilter(dim=DIM, distance_threshold=R, max_balls=20)
        f.construct(vecs)
        self.assertEqual(len(f._centroids), 3)
        # In the trivial case radius must equal distance_threshold (not zero)
        np.testing.assert_allclose(f._radii, R, atol=1e-5)

    def test_trivial_case_n_equals_max_balls(self):
        vecs = _rand(5)
        f = KMEBFilter(dim=DIM, distance_threshold=R, max_balls=5)
        f.construct(vecs)
        self.assertEqual(len(f._centroids), 5)
        np.testing.assert_allclose(f._radii, R, atol=1e-5)

    def test_radii_at_least_distance_threshold(self):
        """Every ball radius must be >= distance_threshold."""
        f = KMEBFilter(dim=DIM, distance_threshold=R, max_balls=10)
        f.construct(_rand(200))
        self.assertTrue(np.all(f._radii >= R - 1e-5),
                        f"Some radii < distance_threshold: {f._radii[f._radii < R]}")

    def test_radii_vary_between_tight_and_spread_clusters(self):
        """A tight cluster has a smaller ball than a spread cluster."""
        rng = np.random.default_rng(0)
        tight = rng.standard_normal((50, DIM)).astype(np.float32) * 0.01
        spread = rng.standard_normal((50, DIM)).astype(np.float32) * 2.0
        spread[:, 0] += 100.0           # shift far from tight cluster

        f = KMEBFilter(dim=DIM, distance_threshold=R, max_balls=2)
        f.construct(np.vstack([tight, spread]))

        # Tight cluster: radius ≈ R (very small intra-cluster distance + R)
        self.assertLess(np.min(f._radii), R + 0.5,
                        "Tight cluster ball should be close to distance_threshold")
        # Spread cluster: radius >> R due to large intra-cluster distances
        self.assertGreater(np.max(f._radii), R + 1.0,
                           "Spread cluster ball should be much larger than distance_threshold")


# ---------------------------------------------------------------------------
# Query — semantic correctness
# ---------------------------------------------------------------------------

class TestKMEBQuery(unittest.TestCase):

    def test_stored_vectors_always_recalled(self):
        """Every vector used in construct() must be returned True by query()."""
        vecs = _rand(60)
        f = KMEBFilter(dim=DIM, distance_threshold=R, max_balls=8)
        f.construct(vecs)
        results = f.query(vecs)
        fn = sum(1 for r in results if not r)
        self.assertEqual(fn, 0, f"{fn} stored vectors produced false negatives")

    def test_neighbors_within_threshold_recalled(self):
        """Vectors within distance_threshold of a stored vector must be found."""
        vecs = _rand(20)
        f = KMEBFilter(dim=DIM, distance_threshold=R, max_balls=5)
        f.construct(vecs)
        neighbors = _neighbors(vecs, r=R, n=3)
        results = f.query(neighbors)
        fn = sum(1 for r in results if not r)
        self.assertEqual(fn, 0,
                         f"{fn}/{len(neighbors)} close neighbors not recalled")

    def test_distant_vectors_rejected(self):
        """Vectors far beyond distance_threshold of all stored vectors must not match."""
        vecs = _rand(30)
        f = KMEBFilter(dim=DIM, distance_threshold=R, max_balls=5)
        f.construct(vecs)
        # Shift by 1000 units — enormously beyond any ball radius
        self.assertFalse(any(f.query(vecs + 1000.0)))

    def test_query_before_construct_raises(self):
        f = KMEBFilter(dim=DIM, distance_threshold=R, max_balls=5)
        with self.assertRaises(ValueError):
            f.query(np.zeros((1, DIM), dtype=np.float32))

    def test_query_on_empty_filter_returns_all_false(self):
        f = KMEBFilter(dim=DIM, distance_threshold=R, max_balls=5)
        f.construct(np.empty((0, DIM), dtype=np.float32))
        self.assertFalse(any(f.query(_rand(10))))

    def test_query_returns_list_of_bools(self):
        vecs = _rand(20)
        f = KMEBFilter(dim=DIM, distance_threshold=R, max_balls=4)
        f.construct(vecs)
        results = f.query(vecs)
        self.assertIsInstance(results, list)
        self.assertTrue(all(isinstance(r, (bool, np.bool_)) for r in results))

    def test_batch_equals_individual(self):
        """Batch query must give identical results to per-vector queries."""
        vecs = _rand(50)
        queries = _rand(15, seed=99)
        f = KMEBFilter(dim=DIM, distance_threshold=R, max_balls=6)
        f.construct(vecs)
        self.assertEqual(f.query(queries),
                         [f.query(q[None, :])[0] for q in queries])

    def test_single_ball_covers_all_stored_and_neighbors(self):
        vecs = _rand(20)
        neighbors = _neighbors(vecs, r=R, n=2)
        f = KMEBFilter(dim=DIM, distance_threshold=R, max_balls=1)
        f.construct(vecs)
        self.assertTrue(all(f.query(vecs)))
        self.assertTrue(all(f.query(neighbors)))

    def test_larger_threshold_increases_recall(self):
        """A bigger distance_threshold should recall more vectors near the boundary."""
        rng = np.random.default_rng(42)
        stored = _rand(10, seed=0)
        # Queries at distance ~0.6 from stored vectors
        queries = _neighbors(stored, r=0.6 * 0.95, n=2)

        f_small = KMEBFilter(dim=DIM, distance_threshold=0.1, max_balls=10)
        f_small.construct(stored)

        f_large = KMEBFilter(dim=DIM, distance_threshold=0.7, max_balls=10)
        f_large.construct(stored)

        recall_small = sum(f_small.query(queries))
        recall_large = sum(f_large.query(queries))
        self.assertGreater(recall_large, recall_small,
                           "Larger threshold should recall more near-boundary queries")


# ---------------------------------------------------------------------------
# Volume minimisation
# ---------------------------------------------------------------------------

class TestKMEBVolumeMinimisation(unittest.TestCase):
    """
    Verify that K-Means clustering produces tight balls, minimising total
    covered volume compared to coarser representations.
    """
    dim = 16

    def _volume_proxy(self, f: KMEBFilter) -> float:
        """Sum of r^dim — proportional to the total d-ball volume covered."""
        return float(np.sum(f._radii ** self.dim))

    def test_tight_clusters_produce_balls_close_to_threshold(self):
        """
        For well-separated, compact clusters the intra-cluster radius should be
        tiny, so each ball radius should be close to distance_threshold.
        """
        r = 0.2
        vecs = _clustered(5, 30, self.dim, spread=0.02, sep=20.0)
        f = KMEBFilter(dim=self.dim, distance_threshold=r, max_balls=5)
        f.construct(vecs)
        # Each ball radius = tiny cluster spread + r ≈ r
        self.assertTrue(np.all(f._radii < r + 0.5),
                        f"Expected radii close to {r}, got {f._radii}")

    def test_more_balls_does_not_increase_max_radius(self):
        """More balls → smaller clusters → no individual ball needs a larger radius."""
        r = 0.3
        vecs = _rand(300, self.dim)

        f_coarse = KMEBFilter(dim=self.dim, distance_threshold=r, max_balls=5)
        f_coarse.construct(vecs)

        f_fine = KMEBFilter(dim=self.dim, distance_threshold=r, max_balls=50)
        f_fine.construct(vecs)

        self.assertLessEqual(
            np.max(f_fine._radii), np.max(f_coarse._radii),
            f"k=50 max_r={np.max(f_fine._radii):.4f} should not exceed "
            f"k=5  max_r={np.max(f_coarse._radii):.4f}"
        )

    def test_more_balls_reduces_total_covered_volume(self):
        """Total covered volume (sum of r^dim) decreases as K increases.

        Each individual ball gets smaller as K grows, and in high dimensions
        r^dim shrinks faster than the number of balls grows, so the sum falls.
        """
        r = 0.3
        vecs = _rand(400, self.dim)
        ks = [2, 5, 10, 40]
        volumes = []
        for k in ks:
            f = KMEBFilter(dim=self.dim, distance_threshold=r, max_balls=k)
            f.construct(vecs)
            volumes.append(self._volume_proxy(f))

        for i in range(len(ks) - 1):
            self.assertLessEqual(
                volumes[i + 1], volumes[i],
                f"Total volume increased from k={ks[i]} ({volumes[i]:.4e}) "
                f"to k={ks[i + 1]} ({volumes[i + 1]:.4e})"
            )

    def test_zero_false_negatives_for_stored_and_close_neighbors(self):
        """Coverage guarantee holds end-to-end with non-trivial clusters."""
        r = 0.2
        vecs = _clustered(4, 25, self.dim, spread=0.05, sep=10.0)
        f = KMEBFilter(dim=self.dim, distance_threshold=r, max_balls=4)
        f.construct(vecs)
        neighbors = _neighbors(vecs, r=r, n=2)
        self.assertTrue(all(f.query(vecs)))
        self.assertTrue(all(f.query(neighbors)))


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

class TestKMEBMemory(unittest.TestCase):
    dim = 16

    def test_bytesize_before_construct_is_zero(self):
        self.assertEqual(KMEBFilter(dim=self.dim, distance_threshold=R, max_balls=10).get_bytesize(), 0)

    def test_bytesize_after_construct(self):
        k, dim = 6, self.dim
        f = KMEBFilter(dim=dim, distance_threshold=R, max_balls=k)
        f.construct(_rand(100, dim))
        # k centroids × dim float32 + k radii × float32
        self.assertEqual(f.get_bytesize(), k * dim * 4 + k * 4)

    def test_bytesize_within_max_memory_budget(self):
        dim = self.dim
        budget = (dim + 1) * 4 * 8
        f = KMEBFilter(dim=dim, distance_threshold=R, max_memory=budget)
        f.construct(_rand(200, dim))
        self.assertLessEqual(f.get_bytesize(), budget)


# ---------------------------------------------------------------------------
# Static filter interface
# ---------------------------------------------------------------------------

class TestKMEBStaticInterface(unittest.TestCase):

    def setUp(self):
        self.f = KMEBFilter(dim=DIM, distance_threshold=R, max_balls=5)
        self.f.construct(_rand(40))
        self.dummy = np.zeros((1, DIM), dtype=np.float32)

    def test_insert_raises(self):
        with self.assertRaises(NotImplementedError):
            self.f.insert(self.dummy)

    def test_add_raises(self):
        with self.assertRaises(NotImplementedError):
            self.f.add(self.dummy)

    def test_remove_raises(self):
        with self.assertRaises(NotImplementedError):
            self.f.remove(self.dummy)

    def test_exact_matching_fpr_is_zero(self):
        self.assertEqual(self.f.get_exact_matching_fpr(), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
