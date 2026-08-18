import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'src'))
import unittest
import numpy as np
from filters.ball_cover_filter import BallCoverFilter

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


def _covering_ok(vecs: np.ndarray, f: BallCoverFilter, tau: float) -> bool:
    """Every vector must be within radius+tau of at least one stored sphere."""
    centers = f.all_spheres
    radii = f.all_radiuses
    if len(centers) == 0:
        return len(vecs) == 0
    dists = np.linalg.norm(vecs[:, None, :] - centers[None, :, :], axis=2)
    within = dists <= (radii[None, :] + tau + 1e-4)
    return bool(np.all(np.any(within, axis=1)))


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestGonzalezInit(unittest.TestCase):

    def test_max_spheres_stored(self):
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=10)
        self.assertEqual(f.max_spheres, 10)

    def test_max_memory_derives_sphere_count(self):
        dim = 16
        # bytes_per_sphere = (dim + 1) * 4, gamma = 4 -> exactly 7 spheres
        budget = (dim + 1) * 4 * 4 * 7
        f = BallCoverFilter(dim=dim, distance_threshold=R, max_memory=budget)
        self.assertEqual(f.max_spheres, 7)

    def test_max_memory_yields_at_least_one_sphere(self):
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_memory=1)
        self.assertEqual(f.max_spheres, 1)

    def test_max_spheres_overrides_max_memory(self):
        f = BallCoverFilter(dim=DIM, distance_threshold=R,
                              max_memory=10 ** 9, max_spheres=3)
        self.assertEqual(f.max_spheres, 3)

    def test_distance_threshold_stored(self):
        f = BallCoverFilter(dim=DIM, distance_threshold=0.5, max_spheres=5)
        self.assertAlmostEqual(f.distance_threshold, 0.5)

    def test_default_complexity_is_2(self):
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        self.assertEqual(f._complexity_level, 2)


# ---------------------------------------------------------------------------
# set_complexity
# ---------------------------------------------------------------------------

class TestGonzalezComplexity(unittest.TestCase):

    def test_valid_levels_accepted(self):
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        for level in (1, 2):
            f.set_complexity(level)
            self.assertEqual(f._complexity_level, level)

    def test_invalid_level_raises(self):
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        with self.assertRaises(ValueError):
            f.set_complexity(0)
        with self.assertRaises(ValueError):
            f.set_complexity(3)

    def test_changing_complexity_after_construct_requires_rebuild(self):
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        f.construct(_rand(20))
        self.assertTrue(f._constructed)
        f.set_complexity(2)
        self.assertFalse(f._constructed)
        with self.assertRaises(ValueError):
            f.query(_rand(1))


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestGonzalezConstruct(unittest.TestCase):

    def test_respects_max_spheres(self):
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=8)
        f.construct(_rand(100))
        self.assertTrue(f._constructed)
        self.assertLessEqual(len(f.all_spheres), 8)
        self.assertEqual(len(f.all_spheres), len(f.all_radiuses))
        self.assertEqual(len(f.all_spheres), len(f.sphere_counts))

    def test_empty_input(self):
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        f.construct(np.empty((0, DIM), dtype=np.float32))
        self.assertTrue(f._constructed)
        self.assertEqual(len(f.all_spheres), 0)

    def test_trivial_case_fewer_vectors_than_spheres(self):
        """When n <= max_spheres each vector becomes its own zero-radius sphere
        (radii are threshold-free now; distance_threshold is applied at query time)."""
        vecs = _rand(3)
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=20)
        f.construct(vecs)
        self.assertEqual(len(f.all_spheres), 3)
        np.testing.assert_allclose(f.all_radiuses, 0.0, atol=1e-6)
        np.testing.assert_array_equal(f.sphere_counts, 1)

    def test_trivial_case_n_equals_max_spheres(self):
        vecs = _rand(5)
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        f.construct(vecs)
        self.assertEqual(len(f.all_spheres), 5)
        np.testing.assert_allclose(f.all_radiuses, 0.0, atol=1e-6)

    def test_all_spheres_contain_original_vectors_in_trivial_case(self):
        """When max_spheres >= n, each sphere center must literally be one
        of the original vectors, not a cluster average."""
        vecs = _rand(6, seed=17)
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=20)
        f.construct(vecs)
        stored = f.all_spheres
        self.assertEqual(len(stored), len(vecs))
        for v in vecs:
            self.assertTrue(any(np.allclose(v, s) for s in stored),
                            f"original vector {v} not found among stored centers")

    def test_all_vectors_within_radius_of_some_sphere_in_nontrivial_case(self):
        """When max_spheres < n, stored centers are cluster centers rather than
        the original vectors -- but every original vector must still fall
        within the (threshold-free) radius of at least one stored sphere."""
        vecs = _rand(100, seed=23)
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=10)
        f.construct(vecs)
        self.assertLess(len(f.all_spheres), len(vecs),
                        "expected clustering to collapse spheres below vector count")
        self.assertTrue(_covering_ok(vecs, f, tau=0.0))

    def test_sphere_counts_sum_to_input_size(self):
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=10)
        f.construct(_rand(200))
        self.assertEqual(int(f.sphere_counts.sum()), 200)

    def test_separated_clusters_get_one_sphere_each(self):
        """Well-separated tight clusters should each collapse to a small sphere
        whose (threshold-free) radius is close to the cluster's own spread."""
        vecs = _clustered(4, 25, spread=0.02, sep=20.0)
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=4)
        f.construct(vecs)
        self.assertEqual(len(f.all_spheres), 4)
        self.assertTrue(np.all(f.all_radiuses < 0.5),
                        f"Expected small intra-cluster radii, got {f.all_radiuses}")
        np.testing.assert_array_equal(np.sort(f.sphere_counts), 25)

    def test_clustering_stops_early_when_threshold_covered(self):
        """If every point is within distance_threshold of a center, k-means++
        seeding stops adding centers even when max_spheres allows more."""
        vecs = _clustered(3, 30, spread=0.01, sep=20.0)
        f = BallCoverFilter(dim=DIM, distance_threshold=1.0, max_spheres=50)
        f.construct(vecs)
        self.assertLess(len(f.all_spheres), 50)

    def test_construct_returns_log_volume_bounds(self):
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=8)
        low, high = f.construct(_rand(100))
        self.assertTrue(np.isfinite(low))
        self.assertTrue(np.isfinite(high))
        self.assertLessEqual(low, high + 1e-9)

    def test_construct_empty_returns_neg_inf_bounds(self):
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        low, high = f.construct(np.empty((0, DIM), dtype=np.float32))
        self.assertEqual(low, -np.inf)
        self.assertEqual(high, -np.inf)

    def test_max_spheres_zero_builds_empty_filter(self):
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=0)
        f.construct(_rand(50))
        self.assertTrue(f._constructed)
        self.assertEqual(len(f.all_spheres), 0)
        self.assertFalse(any(f.query(_rand(10))))

    def test_reconstruct_replaces_previous_state(self):
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=10)
        first = _rand(10, seed=0)
        second = _rand(10, seed=7) + 50.0
        f.construct(first)
        f.construct(second)
        self.assertTrue(all(f.query(second)))
        self.assertFalse(any(f.query(first)))


# ---------------------------------------------------------------------------
# Query — semantic correctness / tau handling
# ---------------------------------------------------------------------------

class TestGonzalezQuery(unittest.TestCase):

    def test_stored_vectors_always_recalled(self):
        vecs = _rand(60)
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=8)
        f.construct(vecs)
        results = f.query(vecs)
        fn = sum(1 for r in results if not r)
        self.assertEqual(fn, 0, f"{fn} stored vectors produced false negatives")

    def test_neighbors_within_threshold_recalled(self):
        vecs = _rand(20)
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        f.construct(vecs)
        neighbors = _neighbors(vecs, r=R, n=3)
        results = f.query(neighbors)
        fn = sum(1 for r in results if not r)
        self.assertEqual(fn, 0, f"{fn}/{len(neighbors)} close neighbors not recalled")

    def test_distant_vectors_rejected(self):
        vecs = _rand(30)
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        f.construct(vecs)
        self.assertFalse(any(f.query(vecs + 1000.0)))

    def test_query_before_construct_raises(self):
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        with self.assertRaises(ValueError):
            f.query(np.zeros((1, DIM), dtype=np.float32))

    def test_query_on_empty_filter_returns_all_false(self):
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        f.construct(np.empty((0, DIM), dtype=np.float32))
        self.assertFalse(any(f.query(_rand(10))))

    def test_query_empty_batch_returns_empty_list(self):
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        f.construct(_rand(20))
        self.assertEqual(f.query(np.empty((0, DIM), dtype=np.float32)), [])

    def test_query_returns_list_of_bools(self):
        vecs = _rand(20)
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=4)
        f.construct(vecs)
        results = f.query(vecs)
        self.assertIsInstance(results, list)
        self.assertTrue(all(isinstance(r, (bool, np.bool_)) for r in results))

    def test_batch_equals_individual(self):
        vecs = _rand(50)
        queries = _rand(15, seed=99)
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=6)
        f.construct(vecs)
        self.assertEqual(f.query(queries),
                         [f.query(q[None, :])[0] for q in queries])

    def test_single_sphere_covers_all_stored_and_neighbors(self):
        vecs = _rand(20)
        neighbors = _neighbors(vecs, r=R, n=2)
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=1)
        f.construct(vecs)
        self.assertTrue(all(f.query(vecs)))
        self.assertTrue(all(f.query(neighbors)))

    def test_single_point_just_outside_threshold_rejected(self):
        vec = _rand(1, seed=42)
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=1)
        f.construct(vec)
        self.assertTrue(f.query(vec)[0])

        direction = np.random.default_rng(43).standard_normal((1, DIM)).astype(np.float32)
        direction /= np.linalg.norm(direction)
        just_outside = vec + direction * (R + 0.05)
        self.assertFalse(f.query(just_outside)[0])

    def test_uniform_points_around_stored_vectors_recalled(self):
        vecs = _rand(15, seed=11)
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        f.construct(vecs)
        samples = _uniform_ball(vecs, r=R, n=30)
        results = f.query(samples)
        fn = sum(1 for r_ in results if not r_)
        self.assertEqual(fn, 0, f"{fn}/{len(samples)} uniformly sampled points not recalled")

    def test_tau_defaults_to_distance_threshold(self):
        vecs = _rand(20)
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        f.construct(vecs)
        neighbors = _neighbors(vecs, r=R, n=2)
        self.assertEqual(f.query(neighbors), f.query(neighbors, tau=R))

    def test_larger_tau_increases_recall_on_one_constructed_filter(self):
        """A single constructed filter must serve multiple tau at query time."""
        stored = _rand(10, seed=0)
        queries = _neighbors(stored, r=0.6 * 0.95, n=2)
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=10)
        f.construct(stored)
        recall_small = sum(f.query(queries, tau=0.1))
        recall_large = sum(f.query(queries, tau=0.7))
        self.assertGreaterEqual(recall_large, recall_small,
                                "Larger tau should recall at least as many near-boundary queries")

    def test_zero_radius_sphere_still_matches_at_tau(self):
        vec = _rand(1, seed=5)
        f = BallCoverFilter(dim=DIM, distance_threshold=0.0, max_spheres=1)
        f.construct(vec)
        self.assertAlmostEqual(float(f.all_radiuses[0]), 0.0, places=5)
        self.assertTrue(f.query(vec, tau=0.0)[0])


# ---------------------------------------------------------------------------
# Threshold independence
# ---------------------------------------------------------------------------

class TestGonzalezThresholdIndependence(unittest.TestCase):

    def test_one_structure_answers_several_taus_consistently(self):
        vecs = _clustered(4, 20, spread=0.05, sep=10.0)
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=4)
        f.construct(vecs)

        # every stored vector must be recalled at every tau >= 0
        for tau in (0.0, 0.1, 0.3, 1.0):
            self.assertTrue(all(f.query(vecs, tau=tau)),
                            f"stored vectors not recalled at tau={tau}")

        # recall is monotone non-decreasing in tau for a fixed probe set
        probes = _neighbors(vecs, r=0.5, n=3)
        taus = [0.0, 0.2, 0.5, 1.0, 2.0]
        recalls = [sum(f.query(probes, tau=t)) for t in taus]
        for a, b in zip(recalls, recalls[1:]):
            self.assertLessEqual(a, b)

    def test_rebuild_is_unnecessary_across_taus(self):
        """A filter built once must match one rebuilt per tau in stored-vector recall."""
        vecs = _rand(40, seed=3)
        for tau in (0.0, 0.2, 0.5):
            f_once = BallCoverFilter(dim=DIM, distance_threshold=tau, max_spheres=6, seed=1)
            f_once.construct(vecs)
            f_shared = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=6, seed=1)
            f_shared.construct(vecs)
            self.assertTrue(all(f_shared.query(vecs, tau=tau)))
            self.assertTrue(all(f_once.query(vecs)))


# ---------------------------------------------------------------------------
# Complexity levels: coverage, monotonicity, budget, determinism
# ---------------------------------------------------------------------------

class TestGonzalezLevels(unittest.TestCase):

    def test_no_false_negatives_at_every_level(self):
        vecs = _clustered(5, 30, spread=0.1, sep=12.0, seed=4)
        for level in (1, 2):
            f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=8, seed=1)
            f.set_complexity(level)
            f.construct(vecs)
            for tau in (0.0, R, 1.0):
                self.assertTrue(all(f.query(vecs, tau=tau)),
                                f"false negative at level {level}, tau={tau}")

    def test_log_J_non_increasing_across_levels(self):
        vecs = _rand(150, seed=6)
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=10, seed=1)
        f.set_complexity(2)
        f.construct(vecs)
        stats = f.get_stats()
        log_Js = [lvl["log_J"] for lvl in stats["levels"]]
        self.assertEqual(len(log_Js), 2)
        for a, b in zip(log_Js, log_Js[1:]):
            self.assertLessEqual(b, a + 1e-6, f"log_J increased: {log_Js}")

    def test_ball_count_never_exceeds_max_spheres(self):
        vecs = _rand(300, seed=8)
        for level in (1, 2):
            f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=12, seed=2)
            f.set_complexity(level)
            f.construct(vecs)
            self.assertLessEqual(len(f.all_spheres), 12)

    def test_determinism_same_seed_same_result(self):
        vecs = _rand(120, seed=9)
        f1 = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=10, seed=7)
        f2 = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=10, seed=7)
        f1.construct(vecs)
        f2.construct(vecs)
        np.testing.assert_array_equal(f1.all_spheres, f2.all_spheres)
        np.testing.assert_array_equal(f1.all_radiuses, f2.all_radiuses)
        np.testing.assert_array_equal(f1.sphere_counts, f2.sphere_counts)

    def test_get_stats_reports_per_level_entries(self):
        vecs = _rand(80, seed=10)
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=6, seed=1)
        f.construct(vecs)
        stats = f.get_stats(taus=[0.1, 0.5])
        self.assertEqual([lvl["level"] for lvl in stats["levels"]], [1, 2])
        for lvl in stats["levels"]:
            self.assertIn(0.1, lvl["log_J_tau"])
            self.assertIn(0.5, lvl["log_J_tau"])
            self.assertGreaterEqual(lvl["wall_time"], 0.0)
        self.assertIsNone(stats["feedback"])


# ---------------------------------------------------------------------------
# Dynamic insertion (add / insert)
# ---------------------------------------------------------------------------

class TestGonzalezAdd(unittest.TestCase):

    def test_add_before_construct_constructs(self):
        vecs = _rand(30)
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        f.add(vecs)
        self.assertTrue(f._constructed)
        self.assertTrue(all(f.query(vecs)))

    def test_added_vectors_are_recalled(self):
        base = _rand(20, seed=0)
        extra = _rand(20, seed=5) + 50.0
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=10)
        f.construct(base)
        f.add(extra)
        self.assertTrue(all(f.query(base)), "base vectors lost after add")
        self.assertTrue(all(f.query(extra)), "added vectors not recalled")

    def test_add_far_vector_creates_new_sphere(self):
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=30)
        f.construct(_rand(5))
        before = len(f.all_spheres)
        f.add(np.full((1, DIM), 500.0, dtype=np.float32))
        self.assertEqual(len(f.all_spheres), before + 1)

    def test_add_singleton_has_zero_radius(self):
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=30)
        f.construct(_rand(5))
        far = np.full((1, DIM), 500.0, dtype=np.float32)
        f.add(far)
        self.assertAlmostEqual(float(f.all_radiuses[-1]), 0.0, places=4)

    def test_add_never_exceeds_max_spheres(self):
        """Once at capacity, far vectors are merged instead of creating spheres."""
        k = 5
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=k)
        f.construct(_rand(k))
        self.assertEqual(len(f.all_spheres), k)
        extra = _rand(50, seed=3) * 20.0
        f.add(extra)
        self.assertLessEqual(len(f.all_spheres), k)
        self.assertTrue(all(f.query(extra)), "merged vectors must still be recalled")

    def test_add_preserves_no_false_negatives_over_many_batches(self):
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=8)
        all_vecs = []
        f.construct(_rand(20, seed=0))
        all_vecs.append(_rand(20, seed=0))
        for seed in range(1, 6):
            batch = _rand(15, seed=seed) * (1.0 + seed)
            f.add(batch)
            all_vecs.append(batch)
        stored = np.vstack(all_vecs)
        self.assertTrue(all(f.query(stored)), "no-false-negative invariant broken by add()")
        self.assertLessEqual(len(f.all_spheres), 8)

    def test_add_counts_track_total_insertions(self):
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=6)
        f.construct(_rand(30, seed=0))
        f.add(_rand(12, seed=1))
        self.assertEqual(int(f.sphere_counts.sum()), 42)

    def test_add_empty_is_noop(self):
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        f.construct(_rand(10))
        spheres = f.all_spheres.copy()
        f.add(np.empty((0, DIM), dtype=np.float32))
        np.testing.assert_array_equal(f.all_spheres, spheres)

    def test_insert_delegates_to_add(self):
        vecs = _rand(10)
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=20)
        f.insert(vecs)
        self.assertTrue(f._constructed)
        self.assertTrue(all(f.query(vecs)))

    def test_duplicate_vector_merges_cheaply_when_at_capacity(self):
        """When at capacity, re-adding an exact duplicate of a stored vector
        should be an (almost) free merge: J shouldn't grow and the duplicate
        must remain recalled."""
        k = 6
        vecs = _rand(k, seed=2)
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=k)
        f.construct(vecs)
        log_J_before = f._log_J(f.all_radiuses)
        f.add(vecs[:1])
        self.assertLessEqual(len(f.all_spheres), k)
        self.assertTrue(f.query(vecs[:1])[0])
        log_J_after = f._log_J(f.all_radiuses)
        self.assertLessEqual(log_J_after, log_J_before + 1e-6)


# ---------------------------------------------------------------------------
# Removal
# ---------------------------------------------------------------------------

class TestGonzalezRemove(unittest.TestCase):

    def test_remove_stored_vector_returns_true(self):
        vecs = _rand(10)
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=30)
        f.construct(vecs)
        self.assertEqual(f.remove(vecs[:1]), [True])

    def test_remove_far_vector_returns_false(self):
        vecs = _rand(10)
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=30)
        f.construct(vecs)
        far = vecs[:1] + 1000.0
        self.assertEqual(f.remove(far), [False])
        self.assertEqual(len(f.all_spheres), 10)

    def test_remove_deletes_emptied_sphere(self):
        vecs = _rand(10)
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=30)
        f.construct(vecs)
        f.remove(vecs[:1])
        self.assertEqual(len(f.all_spheres), 9)
        self.assertFalse(f.query(vecs[:1])[0])
        self.assertTrue(all(f.query(vecs[1:])), "unrelated vectors lost by remove")

    def test_remove_twice_second_returns_false(self):
        vecs = _rand(10)
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=30)
        f.construct(vecs)
        self.assertEqual(f.remove(vecs[:1]), [True])
        self.assertEqual(f.remove(vecs[:1]), [False])

    def test_remove_decrements_count_without_deleting_shared_sphere(self):
        vecs = _clustered(1, 5, spread=0.01)
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=1)
        f.construct(vecs)
        self.assertEqual(len(f.all_spheres), 1)
        self.assertEqual(int(f.sphere_counts[0]), 5)
        f.remove(vecs[:2])
        self.assertEqual(len(f.all_spheres), 1)
        self.assertEqual(int(f.sphere_counts[0]), 3)
        f.remove(vecs[2:])
        self.assertEqual(len(f.all_spheres), 0)

    def test_remove_on_empty_filter_returns_all_false(self):
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        f.construct(np.empty((0, DIM), dtype=np.float32))
        self.assertEqual(f.remove(_rand(3)), [False, False, False])

    def test_remove_empty_batch_returns_empty_list(self):
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        f.construct(_rand(10))
        self.assertEqual(f.remove(np.empty((0, DIM), dtype=np.float32)), [])

    def test_remove_unseen_vector_on_nonempty_filter_returns_false(self):
        vecs = _rand(10, seed=21)
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=30)
        f.construct(vecs)
        unseen = vecs[:1] + 1000.0
        self.assertEqual(f.remove(unseen), [False])

    def test_query_reflects_state_after_remove_then_add(self):
        vecs = _rand(10)
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=30)
        f.construct(vecs)
        self.assertTrue(f.query(vecs[:1])[0])
        f.remove(vecs[:1])
        self.assertFalse(f.query(vecs[:1])[0])
        f.add(vecs[:1])
        self.assertTrue(f.query(vecs[:1])[0])

    def test_remove_respects_tau(self):
        vecs = _rand(10, seed=31)
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=30)
        f.construct(vecs)
        neighbor = _neighbors(vecs[:1], r=0.6, n=1)
        self.assertEqual(f.remove(neighbor, tau=0.05), [False])
        self.assertEqual(f.remove(neighbor, tau=0.8), [True])


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

class TestGonzalezMemory(unittest.TestCase):
    dim = 16

    def test_bytesize_before_construct_is_zero(self):
        f = BallCoverFilter(dim=self.dim, distance_threshold=R, max_spheres=10)
        self.assertEqual(f.get_bytesize(), 0)

    def test_bytesize_after_construct(self):
        f = BallCoverFilter(dim=self.dim, distance_threshold=R, max_spheres=6)
        f.construct(_rand(100, self.dim))
        k = len(f.all_spheres)
        self.assertEqual(f.get_bytesize(), k * self.dim * 4 + k * 4 + k * 4)

    def test_bytesize_within_max_memory_budget(self):
        budget = (self.dim + 1) * 4 * 4 * 8
        f = BallCoverFilter(dim=self.dim, distance_threshold=R, max_memory=budget)
        f.construct(_rand(200, self.dim))
        self.assertLessEqual(f.get_bytesize(), budget)

    def test_construction_bytesize_zero_when_not_retained(self):
        f = BallCoverFilter(dim=self.dim, distance_threshold=R, max_spheres=6)
        f.construct(_rand(100, self.dim))  # retain_assignments defaults to False
        self.assertEqual(f.get_construction_bytesize(), 0)

    def test_construction_bytesize_nonzero_when_retained(self):
        f = BallCoverFilter(dim=self.dim, distance_threshold=R, max_spheres=6)
        vecs = _rand(100, self.dim)
        f.construct(vecs, retain_assignments=True)
        self.assertGreater(f.get_construction_bytesize(), 0)
        self.assertEqual(f.get_construction_bytesize(), vecs.nbytes + f._assignment.nbytes)


# ---------------------------------------------------------------------------
# Misc interface
# ---------------------------------------------------------------------------

class TestGonzalezInterface(unittest.TestCase):

    def test_exact_matching_fpr_is_zero(self):
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        self.assertEqual(f.get_exact_matching_fpr(), 0.0)

    def test_get_volume_is_zero(self):
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        self.assertEqual(f.get_volume(), 0.0)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestGonzalezEdgeCases(unittest.TestCase):

    def test_all_identical_vectors(self):
        """Guards the k-means++ zero-total-distance path."""
        v = _rand(1, seed=44)
        vecs = np.repeat(v, 50, axis=0)
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        f.construct(vecs)
        self.assertGreaterEqual(len(f.all_spheres), 1)
        self.assertTrue(all(f.query(vecs)))

    def test_singleton_cluster(self):
        vecs = np.vstack([_rand(1, seed=1), _clustered(1, 20, spread=0.01, seed=2)])
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=2)
        f.construct(vecs)
        self.assertTrue(all(f.query(vecs)))

    def test_k_greater_equal_n_gives_zero_radius_singletons(self):
        vecs = _rand(4, seed=50)
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=100)
        f.construct(vecs)
        self.assertEqual(len(f.all_spheres), 4)
        np.testing.assert_allclose(f.all_radiuses, 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Level 5: feedback-driven refinement
# ---------------------------------------------------------------------------

def _slack_dataset(dim=8, seed=0):
    """Five tight, well-separated clusters with only 4 spheres allowed, so
    clustering is forced to merge two of them into one large-radius ball --
    exactly the kind of slack feedback refinement should be able to exploit."""
    rng = np.random.default_rng(seed)
    vals = [-60.0, -40.0, -20.0, 20.0, 40.0]
    centers = [np.full(dim, v, dtype=np.float32) for v in vals]
    clusters = [c + rng.standard_normal((25, dim)).astype(np.float32) * 0.2 for c in centers]
    return np.vstack(clusters), rng


class TestGonzalezFeedback(unittest.TestCase):

    def test_set_feedback_policy_rejects_unknown_mode(self):
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        with self.assertRaises(ValueError):
            f.set_feedback_policy(mode="bogus")

    def test_refine_without_construct_raises(self):
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        with self.assertRaises(ValueError):
            f.refine_with_feedback(_rand(5))

    def test_refine_without_retained_assignments_raises(self):
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        f.construct(_rand(30))  # retain_assignments defaults to False
        with self.assertRaises(ValueError):
            f.refine_with_feedback(_rand(5))

    def test_reserved_mode_eliminates_false_positives_without_false_negatives(self):
        dim = 8
        X, rng = _slack_dataset(dim=dim, seed=0)
        f = BallCoverFilter(dim=dim, distance_threshold=0.1, max_spheres=4, seed=3)
        f.construct(X, retain_assignments=True)

        big = int(np.argmax(f.all_radiuses))
        mid = f.all_spheres[big].copy()
        fp_queries = mid + rng.standard_normal((30, dim)).astype(np.float32) * 0.3
        self.assertEqual(sum(f.query(fp_queries)), 30)  # verified false positives

        f.set_feedback_policy(mode="reserved", reserved_fraction=0.25)
        result = f.refine_with_feedback(fp_queries)

        self.assertGreater(result["fraction_eliminated"], 0.0)
        self.assertLessEqual(len(f.all_spheres), f.max_spheres)
        self.assertTrue(all(f.query(X)), "feedback refinement introduced a false negative")

    def test_split_merge_mode_preserves_ball_count(self):
        dim = 8
        X, rng = _slack_dataset(dim=dim, seed=1)
        f = BallCoverFilter(dim=dim, distance_threshold=0.1, max_spheres=4, seed=3)
        f.construct(X, retain_assignments=True)
        k_before = len(f.all_spheres)

        big = int(np.argmax(f.all_radiuses))
        mid = f.all_spheres[big].copy()
        fp_queries = mid + rng.standard_normal((30, dim)).astype(np.float32) * 0.3

        f.set_feedback_policy(mode="split_merge")
        f.refine_with_feedback(fp_queries)

        self.assertEqual(len(f.all_spheres), k_before)
        self.assertTrue(all(f.query(X)), "feedback refinement introduced a false negative")

    def test_get_stats_feedback_populated_after_refine(self):
        dim = 8
        X, rng = _slack_dataset(dim=dim, seed=2)
        f = BallCoverFilter(dim=dim, distance_threshold=0.1, max_spheres=4, seed=3)
        f.construct(X, retain_assignments=True)
        self.assertIsNone(f.get_stats()["feedback"])

        big = int(np.argmax(f.all_radiuses))
        mid = f.all_spheres[big].copy()
        fp_queries = mid + rng.standard_normal((30, dim)).astype(np.float32) * 0.3
        f.set_feedback_policy(mode="reserved", reserved_fraction=0.25)
        f.refine_with_feedback(fp_queries)

        stats = f.get_stats()
        self.assertIsNotNone(stats["feedback"])
        self.assertIn("fraction_eliminated", stats["feedback"])


# ---------------------------------------------------------------------------
# Centroid quantization
# ---------------------------------------------------------------------------

class TestGonzalezQuantization(unittest.TestCase):

    def test_quantization_disabled_by_default(self):
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=8)
        f.construct(_rand(100))
        self.assertIsNone(f.get_stats()["quantization"])
        self.assertIsNone(f._quantizer_name)

    def test_quantize_respects_the_original_byte_budget_and_uses_more_spheres(self):
        """K is a memory budget, not a literal ball count: quantize=True must
        spend the memory a smaller per-sphere footprint frees on *more*
        spheres, not just compress the same budget_k spheres in place."""
        f_plain = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=8, seed=1)
        f_plain.construct(_rand(100))

        f_quant = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=8, seed=1, quantize=True)
        f_quant.construct(_rand(100))

        stats = f_quant.get_stats()["quantization"]
        self.assertIsNotNone(stats)
        self.assertIn(stats["quantizer"], BallCoverFilter._QUANTIZER_CANDIDATES)
        self.assertLessEqual(f_quant.get_bytesize(), f_plain.get_bytesize())
        self.assertGreater(len(f_quant.all_spheres), len(f_plain.all_spheres))
        self.assertEqual(f_quant.max_spheres, stats["effective_k"])

    def test_quantized_total_volume_never_exceeds_unquantized_baseline(self):
        """The unquantized float32-at-budget_k build is always included as its
        own candidate, so whichever configuration quantize=True settles on
        must be at least as good (by total covering volume) as not
        quantizing at all -- ball count differs, so this compares the
        aggregate objective, not per-ball radii."""
        f_plain = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=8, seed=1)
        f_plain.construct(_rand(100))
        plain_log10_J = f_plain._log_J(f_plain.all_radiuses.astype(np.float64)) / BallCoverFilter._LN10

        f_quant = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=8, seed=1, quantize=True)
        f_quant.construct(_rand(100))
        quant_log10_J = f_quant._log_J(f_quant.all_radiuses.astype(np.float64)) / BallCoverFilter._LN10

        self.assertLessEqual(quant_log10_J, plain_log10_J + 1e-6)

    def test_quantization_preserves_no_false_negatives(self):
        vecs = _rand(60)
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=8, quantize=True)
        f.construct(vecs)
        self.assertTrue(all(f.query(vecs)), "quantization introduced a false negative on stored vectors")

        neighbors = _neighbors(vecs, r=R, n=3)
        self.assertTrue(all(f.query(neighbors)), "quantization introduced a false negative on near neighbors")

    def test_quantize_centers_rebuilds_with_more_spheres_when_vectors_retained(self):
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=8, seed=1)
        vecs = _rand(200)
        f.construct(vecs, retain_assignments=True)
        self.assertIsNone(f.get_stats()["quantization"])

        result = f.quantize_centers()
        self.assertIsNotNone(result)
        self.assertEqual(f.get_stats()["quantization"], result)
        self.assertGreaterEqual(len(f.all_spheres), 8)
        self.assertTrue(all(f.query(vecs)), "quantize_centers introduced a false negative")

    def test_quantize_centers_without_retained_vectors_cannot_improve_and_returns_none(self):
        """Without the original vectors there's no way to rebuild at a larger
        k, and pure in-place precision loss can only grow radii -- so this
        must leave the cover untouched and report no quantization applied."""
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=8, seed=1)
        f.construct(_rand(100))  # retain_assignments defaults to False
        spheres_before = f.all_spheres.copy()
        radii_before = f.all_radiuses.copy()

        result = f.quantize_centers()
        self.assertIsNone(result)
        np.testing.assert_array_equal(f.all_spheres, spheres_before)
        np.testing.assert_array_equal(f.all_radiuses, radii_before)

    def test_quantize_centers_before_construct_raises(self):
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=8)
        with self.assertRaises(ValueError):
            f.quantize_centers()

    def test_quantize_on_empty_filter_is_noop(self):
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=5, quantize=True)
        f.construct(np.empty((0, DIM), dtype=np.float32))
        self.assertIsNone(f.get_stats()["quantization"])
        self.assertEqual(f.get_bytesize(), 0)

    def test_quantizer_candidates_can_be_restricted(self):
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=8, seed=1,
                              quantize=True, quantizer_candidates=("QT_4bit",))
        f.construct(_rand(100))
        self.assertEqual(f.get_stats()["quantization"]["quantizer"], "QT_4bit")


if __name__ == "__main__":
    unittest.main(verbosity=2)
