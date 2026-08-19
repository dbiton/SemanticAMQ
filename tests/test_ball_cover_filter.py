import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'src'))
import unittest
import numpy as np
from src.filters.ball_cover_filter import BallCoverFilter

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
    centers = f.all_centers
    radii = f.all_radiuses
    if len(centers) == 0:
        return len(vecs) == 0
    dists = np.linalg.norm(vecs[:, None, :] - centers[None, :, :], axis=2)
    within = dists <= (radii[None, :] + tau + 1e-4)
    return bool(np.all(np.any(within, axis=1)))


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestBallCoverInit(unittest.TestCase):

    def test_max_spheres_stored(self):
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=10)
        self.assertEqual(f.max_spheres, 10)

    def test_max_memory_derives_sphere_count(self):
        dim = 16
        # bytes_per_sphere = (dim + 1) * 4, gamma = 4
        budget = (dim + 1) * 4 * 4 * 7
        f = BallCoverFilter(dim=dim, distance_threshold=R, max_memory=budget)
        self.assertEqual(f.max_spheres, 6)

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

    def test_default_level_is_3(self):
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        self.assertEqual(f.level, 3)


# ---------------------------------------------------------------------------
# level (init strategy, set at construction time)
# ---------------------------------------------------------------------------

class TestBallCoverLevel(unittest.TestCase):

    def test_valid_levels_accepted(self):
        for level in (1, 2, 3):
            f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=5, level=level)
            self.assertEqual(f.level, level)

    def test_invalid_level_raises(self):
        with self.assertRaises(ValueError):
            BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=5, level=0)
        with self.assertRaises(ValueError):
            BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=5, level=4)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestBallCoverConstruct(unittest.TestCase):

    def test_respects_max_spheres(self):
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=8)
        f.construct(_rand(100))
        self.assertTrue(f._constructed)
        self.assertLessEqual(len(f.all_centers), 8)
        self.assertEqual(len(f.all_centers), len(f.all_radiuses))
        self.assertEqual(len(f.all_centers), f._num_spheres)

    def test_empty_input(self):
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        f.construct(np.empty((0, DIM), dtype=np.float32))
        self.assertTrue(f._constructed)
        self.assertEqual(len(f.all_centers), 0)

    def test_trivial_case_fewer_vectors_than_spheres(self):
        """When n <= max_spheres each vector becomes its own zero-radius sphere
        (radii are threshold-free now; distance_threshold is applied at query time)."""
        vecs = _rand(3)
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=20)
        f.construct(vecs)
        self.assertEqual(len(f.all_centers), 3)
        np.testing.assert_allclose(f.all_radiuses, 0.0, atol=1e-6)
        self.assertEqual(f._num_spheres, 3)

    def test_trivial_case_n_equals_max_spheres(self):
        vecs = _rand(5)
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        f.construct(vecs)
        self.assertEqual(len(f.all_centers), 5)
        np.testing.assert_allclose(f.all_radiuses, 0.0, atol=1e-6)

    def test_all_centers_contain_original_vectors_in_trivial_case(self):
        """When max_spheres >= n, each sphere center must literally be one
        of the original vectors, not a cluster average."""
        vecs = _rand(6, seed=17)
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=20)
        f.construct(vecs)
        stored = f.all_centers
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
        self.assertLess(len(f.all_centers), len(vecs),
                        "expected clustering to collapse spheres below vector count")
        self.assertTrue(_covering_ok(vecs, f, tau=0.0))

    def test_separated_clusters_get_one_sphere_each(self):
        """Well-separated tight clusters should each collapse to a small sphere
        whose (threshold-free) radius is close to the cluster's own spread."""
        vecs = _clustered(4, 25, spread=0.02, sep=20.0)
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=4)
        f.construct(vecs)
        self.assertEqual(len(f.all_centers), 4)
        self.assertTrue(np.all(f.all_radiuses < 0.5),
                        f"Expected small intra-cluster radii, got {f.all_radiuses}")
        self.assertEqual(f._num_spheres, 4)

    def test_budget_saturates_even_when_tightly_clustered(self):
        """Budget saturation (invariant #2): all three init levels return
        exactly max_spheres balls whenever max_spheres < n, even when the
        data is one tight, well-under-threshold cluster -- unlike the
        retired k-means++ path, none of them early-stop under
        distance_threshold."""
        vecs = _clustered(3, 30, spread=0.01, sep=20.0)
        for level in (1, 2, 3):
            f = BallCoverFilter(dim=DIM, distance_threshold=1.0, max_spheres=50, level=level, seed=1)
            f.construct(vecs)
            self.assertEqual(len(f.all_spheres), 50, f"level {level} did not saturate to max_spheres")

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
        self.assertEqual(len(f.all_centers), 0)
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

class TestBallCoverQuery(unittest.TestCase):

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

    def test_query_multi_tau_matches_per_tau_query(self):
        vecs = _rand(60, seed=12)
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=8)
        f.construct(vecs)
        probes = _neighbors(vecs, r=R, n=3)
        taus = [0.0, 0.1, R, 0.5, 1.0]

        result = f.query_multi_tau(probes, taus)
        self.assertEqual(sorted(result.keys()), sorted(taus))
        for t in taus:
            self.assertEqual(result[t], f.query(probes, tau=t), f"mismatch at tau={t}")

    def test_query_multi_tau_before_construct_raises(self):
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        with self.assertRaises(ValueError):
            f.query_multi_tau(_rand(3), [0.0, 0.5])

    def test_query_multi_tau_on_empty_filter_returns_all_false(self):
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        f.construct(np.empty((0, DIM), dtype=np.float32))
        result = f.query_multi_tau(_rand(4), [0.0, 1.0])
        self.assertEqual(result[0.0], [False] * 4)
        self.assertEqual(result[1.0], [False] * 4)

    def test_query_multi_tau_empty_batch_returns_empty_lists(self):
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        f.construct(_rand(20))
        result = f.query_multi_tau(np.empty((0, DIM), dtype=np.float32), [0.0, 0.5])
        self.assertEqual(result[0.0], [])
        self.assertEqual(result[0.5], [])


# ---------------------------------------------------------------------------
# Threshold independence
# ---------------------------------------------------------------------------

class TestBallCoverThresholdIndependence(unittest.TestCase):

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

class TestBallCoverLevels(unittest.TestCase):

    def test_no_false_negatives_at_every_level(self):
        vecs = _clustered(5, 30, spread=0.1, sep=12.0, seed=4)
        for level in (1, 2, 3):
            f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=8, seed=1, level=level)
            f.construct(vecs)
            for tau in (0.0, R, 1.0):
                self.assertTrue(all(f.query(vecs, tau=tau)),
                                f"false negative at level {level}, tau={tau}")

    def test_log_J_non_increasing_across_stages(self):
        vecs = _rand(150, seed=6)
        for level in (1, 2, 3):
            f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=10, seed=1, level=level)
            f.construct(vecs)
            stats = f.get_stats()
            log_Js = [lvl["log_J"] for lvl in stats["levels"]]
            self.assertEqual(len(log_Js), 2)
            for a, b in zip(log_Js, log_Js[1:]):
                self.assertLessEqual(b, a + 1e-6, f"level {level}: log_J increased: {log_Js}")

    def test_ball_count_never_exceeds_max_spheres(self):
        vecs = _rand(300, seed=8)
        for level in (1, 2, 3):
            f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=12, seed=2, level=level)
            f.construct(vecs)
            self.assertLessEqual(len(f.all_centers), 12)

    def test_exactly_k_balls_when_n_greater_than_k(self):
        """Budget saturation invariant #2: exactly k balls, not merely <= k,
        whenever n > k."""
        vecs = _rand(300, seed=8)
        for level in (1, 2, 3):
            f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=12, seed=2, level=level)
            f.construct(vecs)
            self.assertEqual(len(f.all_spheres), 12, f"level {level} did not saturate to k")

    def test_determinism_same_seed_same_result(self):
        vecs = _rand(120, seed=9)
        for level in (1, 2, 3):
            f1 = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=10, seed=7, level=level)
            f2 = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=10, seed=7, level=level)
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

    def test_n_less_than_k_gives_n_singletons_at_every_level(self):
        vecs = _rand(4, seed=50)
        for level in (1, 2, 3):
            f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=100, level=level)
            f.construct(vecs)
            self.assertEqual(len(f.all_spheres), 4, f"level {level}")
            np.testing.assert_allclose(f.all_radiuses, 0.0, atol=1e-6)

    def test_coverage_invariant_at_every_level(self):
        vecs = _rand(150, seed=17)
        for level in (1, 2, 3):
            f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=10, seed=1, level=level)
            f.construct(vecs)
            self.assertTrue(_covering_ok(vecs, f, tau=0.0), f"level {level} covering invariant violated")

    def test_log_J_finite_and_negative_at_high_dim(self):
        """log J must stay finite (never overflow via a raw r**d) at d ~ 384,
        where r**d itself would over/underflow float64."""
        dim = 384
        rng = np.random.default_rng(3)
        vecs = rng.standard_normal((100, dim)).astype(np.float32) * 0.01
        for level in (1, 2, 3):
            f = BallCoverFilter(dim=dim, distance_threshold=0.05, max_spheres=8, seed=1, level=level)
            f.construct(vecs)
            log_J = f.get_stats()["levels"][-1]["log_J"]
            self.assertTrue(np.isfinite(log_J), f"level {level}: log_J={log_J}")
            self.assertLess(log_J, 0.0, f"level {level}: expected negative-large log_J, got {log_J}")

    def test_gonzalez_lower_bound_only_set_for_level_3(self):
        vecs = _rand(150, seed=6)
        for level in (1, 2):
            f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=10, seed=1, level=level)
            f.construct(vecs)
            self.assertIsNone(f.lower_bound, f"level {level} should not report a certificate")

        f3 = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=10, seed=1, level=3)
        f3.construct(vecs)
        self.assertIsNotNone(f3.lower_bound)
        self.assertTrue(np.isfinite(f3.lower_bound))
        log_J = f3.get_stats()["levels"][-1]["log_J"]
        self.assertGreaterEqual(log_J - f3.lower_bound, -1e-6,
                                "log_J fell below its own certified lower bound")

    def test_streaming_shuffle_and_n_candidates_preserve_coverage(self):
        vecs = _rand(200, seed=21)
        for shuffle in (False, True):
            for n_candidates in (0, 3):
                f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=10, seed=1, level=1,
                                     streaming_shuffle=shuffle, streaming_n_candidates=n_candidates)
                f.construct(vecs)
                self.assertEqual(len(f.all_spheres), 10)
                self.assertTrue(_covering_ok(vecs, f, tau=0.0),
                                f"shuffle={shuffle}, n_candidates={n_candidates}")


# ---------------------------------------------------------------------------
# Dynamic insertion (add / insert)
# ---------------------------------------------------------------------------

class TestBallCoverAdd(unittest.TestCase):

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
        before = len(f.all_centers)
        f.add(np.full((1, DIM), 500.0, dtype=np.float32))
        self.assertEqual(len(f.all_centers), before + 1)

    def test_add_singleton_has_zero_radius(self):
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=30)
        f.construct(_rand(5))
        far = np.full((1, DIM), 500.0, dtype=np.float32)
        f.add(far)
        self.assertAlmostEqual(float(f.all_radiuses[-1]), f.distance_threshold, places=4)

    def test_add_never_exceeds_max_spheres(self):
        """Once at capacity, far vectors are merged instead of creating spheres."""
        k = 5
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=k)
        f.construct(_rand(k))
        self.assertEqual(len(f.all_centers), k)
        extra = _rand(50, seed=3) * 20.0
        f.add(extra)
        self.assertLessEqual(len(f.all_centers), k)
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
        self.assertLessEqual(len(f.all_centers), 8)

    def test_add_empty_is_noop(self):
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        f.construct(_rand(10))
        spheres = f.all_centers.copy()
        f.add(np.empty((0, DIM), dtype=np.float32))
        np.testing.assert_array_equal(f.all_centers, spheres)

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
        self.assertLessEqual(len(f.all_centers), k)
        self.assertTrue(f.query(vecs[:1])[0])
        log_J_after = f._log_J(f.all_radiuses)
        if np.isneginf(log_J_before):
            self.assertLessEqual(log_J_after, -30.0)
        else:
            self.assertLessEqual(log_J_after, log_J_before + 1e-6)

    def test_add_forces_optimal_merge_heuristic(self):
        """When capacity is reached, the filter must merge the closest pair 
        (minimizing false positive volume) rather than arbitrary spheres."""
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=2)
        close_pts = np.array([[0.0] * DIM, [0.1] * DIM], dtype=np.float32)
        far_pt = np.array([[100.0] * DIM], dtype=np.float32)
        f.construct(close_pts)
        f.add(far_pt)
        self.assertEqual(len(f.all_centers), 2)
        radii = f.all_radiuses
        self.assertTrue(any(r < R + 1.0 for r in radii), "Did not merge the closest pair")
        self.assertTrue(all(f.query(close_pts)))
        self.assertTrue(f.query(far_pt)[0])

    def test_extreme_distances_do_not_cause_log_space_overflow(self):
        """Adding extremely distant vectors must safely handle high-dimensional
        volume growth calculations using log-space without yielding NaN/Inf."""
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=3)
        f.construct(_rand(3))
        extreme_vecs = np.array([[1e15] * DIM, [-1e15] * DIM], dtype=np.float32)
        f.add(extreme_vecs)
        self.assertFalse(np.any(np.isnan(f.all_centers)), "Centers contain NaN")
        self.assertFalse(np.any(np.isinf(f.all_centers)), "Centers contain Inf")
        self.assertFalse(np.any(np.isnan(f.all_radiuses)), "Radii contain NaN")
        self.assertFalse(np.any(np.isinf(f.all_radiuses)), "Radii contain Inf")
        self.assertTrue(all(f.query(extreme_vecs)))

    def test_min_heap_versioning_rejects_stale_candidate_pairs(self):
        """Ensure that _sphere_versions correctly increments during merges,
        forcing the candidate pair min-heap to ignore stale/invalidated pairs."""
        k = 4
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=k)
        f.construct(_rand(k, seed=42))
        f.add(_rand(15, seed=43) * 5.0)
        self.assertTrue(np.any(f._sphere_versions[:len(f.all_centers)] > 0), "Sphere versions did not increment after merges")
        self.assertTrue(all(f.query(_rand(k, seed=42))), "Original vectors lost during heap resolution")

    def test_add_batch_larger_than_capacity_merges_successfully(self):
        """Adding a single batch of vectors larger than max_spheres must trigger
        internal batch-merging down to max_spheres without exceeding buffer bounds."""
        k = 5
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=k)
        f.construct(_rand(2))
        massive_batch = _rand(50, seed=100) * 15.0
        f.add(massive_batch)

        self.assertLessEqual(len(f.all_centers), k)
        self.assertTrue(all(f.query(massive_batch)), "Vectors lost during massive batch merge")

    def test_add_zero_variance_batch_prevents_nan_corruption(self):
        """Adding a batch of identical vectors should merge cleanly without
        causing divide-by-zero or NaN center/radius calculations."""
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=4)
        f.construct(_rand(3))
        identical_batch = np.tile(np.array([[50.0] * DIM], dtype=np.float32), (20, 1))
        f.add(identical_batch)
        self.assertFalse(np.any(np.isnan(f.all_centers)), "Centers contain NaN")
        self.assertFalse(np.any(np.isnan(f.all_radiuses)), "Radii contain NaN")
        self.assertTrue(all(f.query(identical_batch)), "Identical batch not recalled")

    def test_internal_buffer_expansion_preserves_state(self):
        """Verify that _ensure_capacity correctly copies old data into newly
        allocated, larger NumPy arrays without corrupting existing spheres."""
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=100)
        initial_cap = 2
        f._all_spheres = np.arange(initial_cap, dtype=np.int32)
        f._all_centers = np.empty((initial_cap, DIM), dtype=np.float32)
        f._all_radiuses = np.empty((initial_cap,), dtype=np.float32)
        f._sphere_versions = np.zeros(initial_cap, dtype=np.int32)
        f._num_spheres = 0
        pts = _rand(5, seed=77)
        f.construct(pts)
        self.assertGreaterEqual(len(f._all_centers), 5)
        self.assertTrue(all(f.query(pts)), "Vectors lost during buffer expansion")

    def test_merging_spheres_monotonically_increases_volume(self):
        """When spheres merge, the new bounding sphere must have a radius large 
        enough to cover the volume of the original separate spheres."""
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=2)
        pts = np.array([[0.0] * DIM, [5.0] * DIM], dtype=np.float32)
        f.construct(pts)
        max_radius_before = np.max(f.all_radiuses)
        far_pt = np.array([[100.0] * DIM], dtype=np.float32)
        f.add(far_pt)
        max_radius_after = np.max(f.all_radiuses)
        self.assertGreater(max_radius_after, max_radius_before)
        self.assertTrue(all(f.query(pts)))


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

class TestBallCoverMemory(unittest.TestCase):
    dim = 16

    def test_bytesize_before_construct_is_zero(self):
        f = BallCoverFilter(dim=self.dim, distance_threshold=R, max_spheres=10)
        k = len(f.all_centers)
        self.assertEqual(f.get_bytesize(), k * self.dim * 4 + k * 4 + k * 4)

    def test_bytesize_after_construct(self):
        f = BallCoverFilter(dim=self.dim, distance_threshold=R, max_spheres=6)
        f.construct(_rand(100, self.dim))
        k = len(f.all_centers)
        heap_size = sys.getsizeof(f._minimal_pairs)
        self.assertEqual(f.get_bytesize(), k * self.dim * 4 + k * 4 + k * 4 + heap_size)

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

class TestBallCoverInterface(unittest.TestCase):

    def test_exact_matching_fpr_is_zero(self):
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        self.assertEqual(f.get_exact_matching_fpr(), 0.0)

    def test_get_volume_is_zero(self):
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        self.assertEqual(f.get_volume(), 0.0)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestBallCoverEdgeCases(unittest.TestCase):

    def test_all_identical_vectors(self):
        """Guards the k-means++ zero-total-distance path."""
        v = _rand(1, seed=44)
        vecs = np.repeat(v, 50, axis=0)
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        f.construct(vecs)
        self.assertGreaterEqual(len(f.all_centers), 1)
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
        self.assertEqual(len(f.all_centers), 4)
        np.testing.assert_allclose(f.all_radiuses, 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Centroid quantization
# ---------------------------------------------------------------------------

class TestBallCoverQuantization(unittest.TestCase):

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
        f_plain_bytesize = f_plain.get_bytesize() - sys.getsizeof(f_plain._minimal_pairs)

        f_quant = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=8, seed=1, quantize=True)
        f_quant.construct(_rand(100))
        f_quant_bytesize = f_quant.get_bytesize() - sys.getsizeof(f_quant._minimal_pairs)

        stats = f_quant.get_stats()["quantization"]
        self.assertIsNotNone(stats)
        self.assertIn(stats["quantizer"], BallCoverFilter._QUANTIZER_CANDIDATES)
        self.assertLessEqual(f_quant_bytesize, f_plain_bytesize)
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
        self.assertGreaterEqual(len(f.all_centers), 8)
        self.assertTrue(all(f.query(vecs)), "quantize_centers introduced a false negative")

    def test_quantize_centers_without_retained_vectors_cannot_improve_and_returns_none(self):
        """Without the original vectors there's no way to rebuild at a larger
        k, and pure in-place precision loss can only grow radii -- so this
        must leave the cover untouched and report no quantization applied."""
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=8, seed=1)
        f.construct(_rand(100))  # retain_assignments defaults to False
        spheres_before = f.all_centers.copy()
        radii_before = f.all_radiuses.copy()

        result = f.quantize_centers()
        self.assertIsNone(result)
        np.testing.assert_array_equal(f.all_centers, spheres_before)
        np.testing.assert_array_equal(f.all_radiuses, radii_before)

    def test_quantize_centers_before_construct_raises(self):
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=8)
        with self.assertRaises(ValueError):
            f.quantize_centers()

    def test_quantize_on_empty_filter_is_noop(self):
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=5, quantize=True)
        f.construct(np.empty((0, DIM), dtype=np.float32))
        self.assertIsNone(f.get_stats()["quantization"])
        self.assertEqual(f.get_bytesize(), 76)

    def test_quantizer_candidates_can_be_restricted(self):
        f = BallCoverFilter(dim=DIM, distance_threshold=R, max_spheres=8, seed=1,
                              quantize=True, quantizer_candidates=("QT_4bit",))
        f.construct(_rand(100))
        self.assertEqual(f.get_stats()["quantization"]["quantizer"], "QT_4bit")


if __name__ == "__main__":
    unittest.main(verbosity=2)
