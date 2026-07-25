import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'src'))
import unittest
import numpy as np
from src.filters.gonzales_filter import GonzalezMEBFilter

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


def _uniform_ball(v: np.ndarray, r: float, n: int = 20, seed: int = 2) -> np.ndarray:
    """Return n points drawn uniformly from the *volume* of the r-ball around
    each row of v (radii scaled by u**(1/dim), not just uniform direction)."""
    rng = np.random.default_rng(seed)
    dim = v.shape[1]
    directions = rng.standard_normal((len(v), n, dim)).astype(np.float64)
    directions /= np.linalg.norm(directions, axis=2, keepdims=True)
    u = rng.uniform(0, 1, (len(v), n, 1))
    radii = (r * 0.95) * u ** (1.0 / dim)
    points = v[:, None, :] + (directions * radii).astype(np.float32)
    return points.reshape(-1, dim)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestGonzalezInit(unittest.TestCase):

    def test_max_spheres_stored(self):
        f = GonzalezMEBFilter(dim=DIM, distance_threshold=R, max_spheres=10)
        self.assertEqual(f.max_spheres, 10)

    def test_max_memory_derives_sphere_count(self):
        dim = 16
        # bytes_per_sphere = (dim + 1) * 4, gamma = 4 -> exactly 7 spheres
        budget = (dim + 1) * 4 * 4 * 7
        f = GonzalezMEBFilter(dim=dim, distance_threshold=R, max_memory=budget)
        self.assertEqual(f.max_spheres, 7)

    def test_max_memory_yields_at_least_one_sphere(self):
        f = GonzalezMEBFilter(dim=DIM, distance_threshold=R, max_memory=1)
        self.assertEqual(f.max_spheres, 1)

    def test_max_spheres_overrides_max_memory(self):
        f = GonzalezMEBFilter(dim=DIM, distance_threshold=R,
                              max_memory=10 ** 9, max_spheres=3)
        self.assertEqual(f.max_spheres, 3)

    def test_distance_threshold_stored(self):
        f = GonzalezMEBFilter(dim=DIM, distance_threshold=0.5, max_spheres=5)
        self.assertAlmostEqual(f.distance_threshold, 0.5)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestGonzalezConstruct(unittest.TestCase):

    def test_respects_max_spheres(self):
        f = GonzalezMEBFilter(dim=DIM, distance_threshold=R, max_spheres=8)
        f.construct(_rand(100))
        self.assertTrue(f._constructed)
        self.assertLessEqual(len(f.all_spheres), 8)
        self.assertEqual(len(f.all_spheres), len(f.all_radiuses))
        self.assertEqual(len(f.all_spheres), len(f.sphere_counts))

    def test_empty_input(self):
        f = GonzalezMEBFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        f.construct(np.empty((0, DIM), dtype=np.float32))
        self.assertTrue(f._constructed)
        self.assertEqual(len(f.all_spheres), 0)

    def test_trivial_case_fewer_vectors_than_spheres(self):
        """When n <= max_spheres each vector becomes its own sphere."""
        vecs = _rand(3)
        f = GonzalezMEBFilter(dim=DIM, distance_threshold=R, max_spheres=20)
        f.construct(vecs)
        self.assertEqual(len(f.all_spheres), 3)
        np.testing.assert_allclose(f.all_radiuses, R, atol=1e-5)
        np.testing.assert_array_equal(f.sphere_counts, 1)

    def test_trivial_case_n_equals_max_spheres(self):
        vecs = _rand(5)
        f = GonzalezMEBFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        f.construct(vecs)
        self.assertEqual(len(f.all_spheres), 5)
        np.testing.assert_allclose(f.all_radiuses, R, atol=1e-5)

    def test_all_spheres_contain_original_vectors_in_trivial_case(self):
        """When max_spheres >= n, each sphere center must literally be one
        of the original vectors, not a cluster average."""
        vecs = _rand(6, seed=17)
        f = GonzalezMEBFilter(dim=DIM, distance_threshold=R, max_spheres=20)
        f.construct(vecs)
        stored = f.all_spheres
        self.assertEqual(len(stored), len(vecs))
        for v in vecs:
            self.assertTrue(any(np.allclose(v, s) for s in stored),
                            f"original vector {v} not found among stored centers")

    def test_all_vectors_within_radius_of_some_sphere_in_nontrivial_case(self):
        """When max_spheres < n, stored centers are cluster MEB centers
        rather than the original vectors — but every original vector must
        still fall within the radius of at least one stored sphere."""
        vecs = _rand(100, seed=23)
        f = GonzalezMEBFilter(dim=DIM, distance_threshold=R, max_spheres=10)
        f.construct(vecs)
        self.assertLess(len(f.all_spheres), len(vecs),
                        "expected clustering to collapse spheres below vector count")

        centers = f.all_spheres
        radii = f.all_radiuses
        dists = np.linalg.norm(vecs[:, None, :] - centers[None, :, :], axis=2)
        within = dists <= radii[None, :] + 1e-5
        uncovered = np.flatnonzero(~np.any(within, axis=1))
        self.assertEqual(len(uncovered), 0,
                         f"{len(uncovered)} vectors not within radius of any stored sphere: "
                         f"{uncovered.tolist()}")

    def test_radii_at_least_distance_threshold(self):
        f = GonzalezMEBFilter(dim=DIM, distance_threshold=R, max_spheres=10)
        f.construct(_rand(200))
        self.assertTrue(np.all(f.all_radiuses >= R - 1e-5),
                        f"Some radii < distance_threshold: {f.all_radiuses[f.all_radiuses < R]}")

    def test_sphere_counts_sum_to_input_size(self):
        f = GonzalezMEBFilter(dim=DIM, distance_threshold=R, max_spheres=10)
        f.construct(_rand(200))
        self.assertEqual(int(f.sphere_counts.sum()), 200)

    def test_separated_clusters_get_one_sphere_each(self):
        """Well-separated tight clusters should each collapse to a small sphere."""
        vecs = _clustered(4, 25, spread=0.02, sep=20.0)
        f = GonzalezMEBFilter(dim=DIM, distance_threshold=R, max_spheres=4)
        f.construct(vecs)
        self.assertEqual(len(f.all_spheres), 4)
        # tiny intra-cluster radius + R ≈ R
        self.assertTrue(np.all(f.all_radiuses < R + 0.5),
                        f"Expected radii close to {R}, got {f.all_radiuses}")
        np.testing.assert_array_equal(np.sort(f.sphere_counts), 25)

    def test_clustering_stops_early_when_threshold_covered(self):
        """If every point is within distance_threshold of a center, Gonzalez
        stops adding centers even when max_spheres allows more."""
        vecs = _clustered(3, 30, spread=0.01, sep=20.0)
        f = GonzalezMEBFilter(dim=DIM, distance_threshold=1.0, max_spheres=50)
        f.construct(vecs)
        self.assertLess(len(f.all_spheres), 50)

    def test_construct_returns_log_volume_bounds(self):
        f = GonzalezMEBFilter(dim=DIM, distance_threshold=R, max_spheres=8)
        low, high = f.construct(_rand(100))
        self.assertTrue(np.isfinite(low))
        self.assertTrue(np.isfinite(high))
        self.assertLessEqual(low, high + 1e-9)

    def test_construct_empty_returns_neg_inf_bounds(self):
        f = GonzalezMEBFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        low, high = f.construct(np.empty((0, DIM), dtype=np.float32))
        self.assertEqual(low, -np.inf)
        self.assertEqual(high, -np.inf)

    def test_max_spheres_zero_builds_empty_filter(self):
        f = GonzalezMEBFilter(dim=DIM, distance_threshold=R, max_spheres=0)
        f.construct(_rand(50))
        self.assertTrue(f._constructed)
        self.assertEqual(len(f.all_spheres), 0)
        self.assertFalse(any(f.query(_rand(10))))

    def test_reconstruct_replaces_previous_state(self):
        f = GonzalezMEBFilter(dim=DIM, distance_threshold=R, max_spheres=10)
        first = _rand(10, seed=0)
        # Offset keeps the datasets far apart while staying inside the
        # float32 precision envelope of the inner-product containment check.
        second = _rand(10, seed=7) + 50.0
        f.construct(first)
        f.construct(second)
        self.assertTrue(all(f.query(second)))
        self.assertFalse(any(f.query(first)))


# ---------------------------------------------------------------------------
# Query — semantic correctness
# ---------------------------------------------------------------------------

class TestGonzalezQuery(unittest.TestCase):

    def test_stored_vectors_always_recalled(self):
        """Every vector used in construct() must be returned True by query()."""
        vecs = _rand(60)
        f = GonzalezMEBFilter(dim=DIM, distance_threshold=R, max_spheres=8)
        f.construct(vecs)
        results = f.query(vecs)
        fn = sum(1 for r in results if not r)
        self.assertEqual(fn, 0, f"{fn} stored vectors produced false negatives")

    def test_neighbors_within_threshold_recalled(self):
        """Vectors within distance_threshold of a stored vector must be found."""
        vecs = _rand(20)
        f = GonzalezMEBFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        f.construct(vecs)
        neighbors = _neighbors(vecs, r=R, n=3)
        results = f.query(neighbors)
        fn = sum(1 for r in results if not r)
        self.assertEqual(fn, 0,
                         f"{fn}/{len(neighbors)} close neighbors not recalled")

    def test_distant_vectors_rejected(self):
        vecs = _rand(30)
        f = GonzalezMEBFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        f.construct(vecs)
        # Shift by 1000 units — enormously beyond any sphere radius
        self.assertFalse(any(f.query(vecs + 1000.0)))

    def test_query_before_construct_raises(self):
        f = GonzalezMEBFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        with self.assertRaises(ValueError):
            f.query(np.zeros((1, DIM), dtype=np.float32))

    def test_query_on_empty_filter_returns_all_false(self):
        f = GonzalezMEBFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        f.construct(np.empty((0, DIM), dtype=np.float32))
        self.assertFalse(any(f.query(_rand(10))))

    def test_query_empty_batch_returns_empty_list(self):
        f = GonzalezMEBFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        f.construct(_rand(20))
        self.assertEqual(f.query(np.empty((0, DIM), dtype=np.float32)), [])

    def test_query_returns_list_of_bools(self):
        vecs = _rand(20)
        f = GonzalezMEBFilter(dim=DIM, distance_threshold=R, max_spheres=4)
        f.construct(vecs)
        results = f.query(vecs)
        self.assertIsInstance(results, list)
        self.assertTrue(all(isinstance(r, (bool, np.bool_)) for r in results))

    def test_batch_equals_individual(self):
        """Batch query must give identical results to per-vector queries."""
        vecs = _rand(50)
        queries = _rand(15, seed=99)
        f = GonzalezMEBFilter(dim=DIM, distance_threshold=R, max_spheres=6)
        f.construct(vecs)
        self.assertEqual(f.query(queries),
                         [f.query(q[None, :])[0] for q in queries])

    def test_single_sphere_covers_all_stored_and_neighbors(self):
        vecs = _rand(20)
        neighbors = _neighbors(vecs, r=R, n=2)
        f = GonzalezMEBFilter(dim=DIM, distance_threshold=R, max_spheres=1)
        f.construct(vecs)
        self.assertTrue(all(f.query(vecs)))
        self.assertTrue(all(f.query(neighbors)))

    def test_single_point_just_outside_threshold_rejected(self):
        """A filter built from a single point should reject a query that
        sits just beyond distance_threshold from it."""
        vec = _rand(1, seed=42)
        f = GonzalezMEBFilter(dim=DIM, distance_threshold=R, max_spheres=1)
        f.construct(vec)
        self.assertTrue(f.query(vec)[0])

        direction = np.random.default_rng(43).standard_normal((1, DIM)).astype(np.float32)
        direction /= np.linalg.norm(direction)
        just_outside = vec + direction * (R + 0.05)
        self.assertFalse(f.query(just_outside)[0])

    def test_uniform_points_around_stored_vectors_recalled(self):
        """Points sampled uniformly from within the distance_threshold ball
        around each stored vector must always be recalled."""
        vecs = _rand(15, seed=11)
        f = GonzalezMEBFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        f.construct(vecs)

        samples = _uniform_ball(vecs, r=R, n=30)
        results = f.query(samples)
        fn = sum(1 for r_ in results if not r_)
        self.assertEqual(fn, 0,
                         f"{fn}/{len(samples)} uniformly sampled points not recalled")

    def test_larger_threshold_increases_recall(self):
        stored = _rand(10, seed=0)
        # Queries at distance ~0.6 from stored vectors
        queries = _neighbors(stored, r=0.6 * 0.95, n=2)

        f_small = GonzalezMEBFilter(dim=DIM, distance_threshold=0.1, max_spheres=10)
        f_small.construct(stored)

        f_large = GonzalezMEBFilter(dim=DIM, distance_threshold=0.7, max_spheres=10)
        f_large.construct(stored)

        recall_small = sum(f_small.query(queries))
        recall_large = sum(f_large.query(queries))
        self.assertGreater(recall_large, recall_small,
                           "Larger threshold should recall more near-boundary queries")


# ---------------------------------------------------------------------------
# Dynamic insertion (add / insert)
# ---------------------------------------------------------------------------

class TestGonzalezAdd(unittest.TestCase):

    def test_add_before_construct_constructs(self):
        vecs = _rand(30)
        f = GonzalezMEBFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        f.add(vecs)
        self.assertTrue(f._constructed)
        self.assertTrue(all(f.query(vecs)))

    def test_added_vectors_are_recalled(self):
        base = _rand(20, seed=0)
        extra = _rand(20, seed=5) + 50.0
        f = GonzalezMEBFilter(dim=DIM, distance_threshold=R, max_spheres=10)
        f.construct(base)
        f.add(extra)
        self.assertTrue(all(f.query(base)), "base vectors lost after add")
        self.assertTrue(all(f.query(extra)), "added vectors not recalled")

    def test_add_far_vector_creates_new_sphere(self):
        f = GonzalezMEBFilter(dim=DIM, distance_threshold=R, max_spheres=30)
        f.construct(_rand(5))
        before = len(f.all_spheres)
        f.add(np.full((1, DIM), 500.0, dtype=np.float32))
        self.assertEqual(len(f.all_spheres), before + 1)

    def test_add_duplicate_increments_count_only(self):
        vecs = _rand(5)
        f = GonzalezMEBFilter(dim=DIM, distance_threshold=R, max_spheres=30)
        f.construct(vecs)
        before = len(f.all_spheres)
        f.add(vecs[:1])
        self.assertEqual(len(f.all_spheres), before)
        self.assertEqual(int(f.sphere_counts.sum()), 6)

    def test_add_never_exceeds_max_spheres(self):
        """Once at capacity, far vectors are merged instead of creating spheres."""
        k = 5
        f = GonzalezMEBFilter(dim=DIM, distance_threshold=R, max_spheres=k)
        f.construct(_rand(k))
        self.assertEqual(len(f.all_spheres), k)
        extra = _rand(50, seed=3) * 20.0
        f.add(extra)
        self.assertLessEqual(len(f.all_spheres), k)
        self.assertTrue(all(f.query(extra)),
                        "merged vectors must still be recalled")

    def test_add_preserves_no_false_negatives_over_many_batches(self):
        f = GonzalezMEBFilter(dim=DIM, distance_threshold=R, max_spheres=8)
        all_vecs = []
        f.construct(_rand(20, seed=0))
        all_vecs.append(_rand(20, seed=0))
        for seed in range(1, 6):
            batch = _rand(15, seed=seed) * (1.0 + seed)
            f.add(batch)
            all_vecs.append(batch)
        stored = np.vstack(all_vecs)
        self.assertTrue(all(f.query(stored)),
                        "no-false-negative invariant broken by add()")

    def test_add_counts_track_total_insertions(self):
        f = GonzalezMEBFilter(dim=DIM, distance_threshold=R, max_spheres=6)
        f.construct(_rand(30, seed=0))
        f.add(_rand(12, seed=1))
        self.assertEqual(int(f.sphere_counts.sum()), 42)

    def test_add_empty_is_noop(self):
        f = GonzalezMEBFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        f.construct(_rand(10))
        spheres = f.all_spheres.copy()
        f.add(np.empty((0, DIM), dtype=np.float32))
        np.testing.assert_array_equal(f.all_spheres, spheres)

    def test_insert_delegates_to_add(self):
        vecs = _rand(10)
        f = GonzalezMEBFilter(dim=DIM, distance_threshold=R, max_spheres=20)
        f.insert(vecs)
        self.assertTrue(f._constructed)
        self.assertTrue(all(f.query(vecs)))


# ---------------------------------------------------------------------------
# Removal
# ---------------------------------------------------------------------------

class TestGonzalezRemove(unittest.TestCase):

    def test_remove_stored_vector_returns_true(self):
        vecs = _rand(10)
        f = GonzalezMEBFilter(dim=DIM, distance_threshold=R, max_spheres=30)
        f.construct(vecs)
        self.assertEqual(f.remove(vecs[:1]), [True])

    def test_remove_far_vector_returns_false(self):
        vecs = _rand(10)
        f = GonzalezMEBFilter(dim=DIM, distance_threshold=R, max_spheres=30)
        f.construct(vecs)
        far = vecs[:1] + 1000.0
        self.assertEqual(f.remove(far), [False])
        self.assertEqual(len(f.all_spheres), 10)

    def test_remove_deletes_emptied_sphere(self):
        """In the trivial case each sphere holds one vector, so removing the
        vector must delete the sphere and stop it matching queries."""
        vecs = _rand(10)
        f = GonzalezMEBFilter(dim=DIM, distance_threshold=R, max_spheres=30)
        f.construct(vecs)
        f.remove(vecs[:1])
        self.assertEqual(len(f.all_spheres), 9)
        self.assertFalse(f.query(vecs[:1])[0])
        self.assertTrue(all(f.query(vecs[1:])), "unrelated vectors lost by remove")

    def test_remove_twice_second_returns_false(self):
        vecs = _rand(10)
        f = GonzalezMEBFilter(dim=DIM, distance_threshold=R, max_spheres=30)
        f.construct(vecs)
        self.assertEqual(f.remove(vecs[:1]), [True])
        self.assertEqual(f.remove(vecs[:1]), [False])

    def test_remove_decrements_count_without_deleting_shared_sphere(self):
        """A sphere holding several vectors survives until its count reaches 0."""
        vecs = _clustered(1, 5, spread=0.01)
        f = GonzalezMEBFilter(dim=DIM, distance_threshold=R, max_spheres=1)
        f.construct(vecs)
        self.assertEqual(len(f.all_spheres), 1)
        self.assertEqual(int(f.sphere_counts[0]), 5)
        f.remove(vecs[:2])
        self.assertEqual(len(f.all_spheres), 1)
        self.assertEqual(int(f.sphere_counts[0]), 3)
        f.remove(vecs[2:])
        self.assertEqual(len(f.all_spheres), 0)

    def test_remove_on_empty_filter_returns_all_false(self):
        f = GonzalezMEBFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        f.construct(np.empty((0, DIM), dtype=np.float32))
        self.assertEqual(f.remove(_rand(3)), [False, False, False])

    def test_remove_empty_batch_returns_empty_list(self):
        f = GonzalezMEBFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        f.construct(_rand(10))
        self.assertEqual(f.remove(np.empty((0, DIM), dtype=np.float32)), [])

    def test_query_reflects_state_after_remove_then_add(self):
        """Cached FAISS indexes must be invalidated across mutations."""
        vecs = _rand(10)
        f = GonzalezMEBFilter(dim=DIM, distance_threshold=R, max_spheres=30)
        f.construct(vecs)
        self.assertTrue(f.query(vecs[:1])[0])
        f.remove(vecs[:1])
        self.assertFalse(f.query(vecs[:1])[0])
        f.add(vecs[:1])
        self.assertTrue(f.query(vecs[:1])[0])


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

class TestGonzalezMemory(unittest.TestCase):
    dim = 16

    def test_bytesize_before_construct_is_zero(self):
        f = GonzalezMEBFilter(dim=self.dim, distance_threshold=R, max_spheres=10)
        self.assertEqual(f.get_bytesize(), 0)

    def test_bytesize_after_construct(self):
        f = GonzalezMEBFilter(dim=self.dim, distance_threshold=R, max_spheres=6)
        f.construct(_rand(100, self.dim))
        k = len(f.all_spheres)
        # k centers × dim float32 + k radii × float32 + k counts × int32
        self.assertEqual(f.get_bytesize(), k * self.dim * 4 + k * 4 + k * 4)

    def test_bytesize_within_max_memory_budget(self):
        budget = (self.dim + 1) * 4 * 4 * 8
        f = GonzalezMEBFilter(dim=self.dim, distance_threshold=R, max_memory=budget)
        f.construct(_rand(200, self.dim))
        self.assertLessEqual(f.get_bytesize(), budget)


# ---------------------------------------------------------------------------
# Misc interface
# ---------------------------------------------------------------------------

class TestGonzalezInterface(unittest.TestCase):

    def test_exact_matching_fpr_is_zero(self):
        f = GonzalezMEBFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        self.assertEqual(f.get_exact_matching_fpr(), 0.0)

    def test_get_volume_is_zero(self):
        f = GonzalezMEBFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        self.assertEqual(f.get_volume(), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
