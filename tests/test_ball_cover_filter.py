import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'src'))
import unittest
import numpy as np
from src.filters.ball_cover_filter import BallCoverFilter

DIM = 8
R = 0.3  # distance threshold used across most tests


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


def _stored_centers(f: BallCoverFilter) -> np.ndarray:
    return f.index.reconstruct_n(0, f.count_balls)


def _covering_ok(vecs: np.ndarray, f: BallCoverFilter, tol: float = 1e-4) -> bool:
    """Every vector must be within (radius + tol) of at least one stored ball."""
    centers = _stored_centers(f)
    radii = f.radii[:f.count_balls]
    if len(centers) == 0:
        return len(vecs) == 0
    dists = np.linalg.norm(vecs[:, None, :] - centers[None, :, :], axis=2)
    within = dists <= (radii[None, :] + tol)
    return bool(np.all(np.any(within, axis=1)))


# ---------------------------------------------------------------------------
# Initialisation / sizing
# ---------------------------------------------------------------------------

class TestBallCoverInit(unittest.TestCase):

    def test_bytes_per_ball(self):
        f = BallCoverFilter(dim=DIM, max_memory=10_000)
        self.assertEqual(f.bytes_per_ball(), (DIM + 1) * 4)

    def test_max_memory_derives_ball_count(self):
        dim = 16
        bytes_per_ball = (dim + 1) * 4
        budget = bytes_per_ball * 7
        f = BallCoverFilter(dim=dim, max_memory=budget)
        self.assertEqual(f.max_count_balls, 7)

    def test_max_memory_rounds_down(self):
        dim = 16
        bytes_per_ball = (dim + 1) * 4
        budget = bytes_per_ball * 7 + 1
        f = BallCoverFilter(dim=dim, max_memory=budget)
        self.assertEqual(f.max_count_balls, 7)

    def test_max_memory_yields_at_least_one_ball(self):
        f = BallCoverFilter(dim=DIM, max_memory=1)
        self.assertEqual(f.max_count_balls, 1)

    def test_initial_state_empty(self):
        f = BallCoverFilter(dim=DIM, max_memory=10_000)
        self.assertEqual(f.count_balls, 0)
        self.assertEqual(f.index.ntotal, 0)
        self.assertEqual(len(f.radii), f.max_count_balls)
        np.testing.assert_array_equal(f.radii, np.zeros(f.max_count_balls, dtype=np.float32))

    def test_dim_stored(self):
        f = BallCoverFilter(dim=DIM, max_memory=10_000)
        self.assertEqual(f.dim, DIM)
        self.assertEqual(f.index.d, DIM)


# ---------------------------------------------------------------------------
# construct_enclosing_balls
# ---------------------------------------------------------------------------

class TestConstructEnclosingBalls(unittest.TestCase):

    def test_trivial_case_fewer_vectors_than_balls(self):
        f = BallCoverFilter(dim=DIM, max_memory=10_000)  # max_count_balls large
        vecs = _rand(5)
        centers, radii = f.construct_enclosing_balls(vecs)
        np.testing.assert_array_equal(centers, vecs)
        np.testing.assert_allclose(radii, 0.0)

    def test_trivial_case_n_equals_max_balls(self):
        bytes_per_ball = (DIM + 1) * 4
        f = BallCoverFilter(dim=DIM, max_memory=bytes_per_ball * 5)
        vecs = _rand(5)
        centers, radii = f.construct_enclosing_balls(vecs)
        self.assertEqual(len(centers), 5)
        np.testing.assert_allclose(radii, 0.0)

    def test_nontrivial_case_returns_exactly_max_balls_centers(self):
        bytes_per_ball = (DIM + 1) * 4
        f = BallCoverFilter(dim=DIM, max_memory=bytes_per_ball * 5)
        vecs = _rand(50, seed=3)
        centers, radii = f.construct_enclosing_balls(vecs)
        self.assertEqual(len(centers), 5)
        self.assertEqual(len(radii), 5)

    def test_nontrivial_case_covers_all_vectors(self):
        bytes_per_ball = (DIM + 1) * 4
        f = BallCoverFilter(dim=DIM, max_memory=bytes_per_ball * 5)
        vecs = _rand(50, seed=3)
        centers, radii = f.construct_enclosing_balls(vecs)
        dists = np.linalg.norm(vecs[:, None, :] - centers[None, :, :], axis=2)
        within = dists <= (radii[None, :] + 1e-3)
        self.assertTrue(np.all(np.any(within, axis=1)))

    def test_empty_input(self):
        f = BallCoverFilter(dim=DIM, max_memory=10_000)
        centers, radii = f.construct_enclosing_balls(np.empty((0, DIM), dtype=np.float32))
        self.assertEqual(len(centers), 0)
        self.assertEqual(len(radii), 0)


# ---------------------------------------------------------------------------
# construct
# ---------------------------------------------------------------------------

class TestConstruct(unittest.TestCase):

    def test_respects_max_balls(self):
        bytes_per_ball = (DIM + 1) * 4
        f = BallCoverFilter(dim=DIM, max_memory=bytes_per_ball * 8)
        f.construct(_rand(100))
        self.assertLessEqual(f.count_balls, 8)
        self.assertEqual(f.index.ntotal, f.count_balls)

    def test_empty_input_leaves_zero_balls(self):
        f = BallCoverFilter(dim=DIM, max_memory=10_000)
        f.construct(np.empty((0, DIM), dtype=np.float32))
        self.assertEqual(f.count_balls, 0)
        self.assertEqual(f.index.ntotal, 0)

    def test_trivial_case_zero_radius_balls(self):
        vecs = _rand(5)
        f = BallCoverFilter(dim=DIM, max_memory=10_000)
        f.construct(vecs)
        self.assertEqual(f.count_balls, 5)
        np.testing.assert_allclose(f.radii[:5], 0.0, atol=1e-6)

    def test_radii_beyond_count_balls_are_zeroed(self):
        bytes_per_ball = (DIM + 1) * 4
        f = BallCoverFilter(dim=DIM, max_memory=bytes_per_ball * 20)
        f.construct(_rand(5))
        self.assertEqual(f.count_balls, 5)
        np.testing.assert_array_equal(f.radii[5:], np.zeros(len(f.radii) - 5, dtype=np.float32))

    def test_covering_invariant_nontrivial_case(self):
        bytes_per_ball = (DIM + 1) * 4
        f = BallCoverFilter(dim=DIM, max_memory=bytes_per_ball * 10)
        vecs = _rand(100, seed=23)
        f.construct(vecs)
        self.assertLess(f.count_balls, len(vecs))
        self.assertTrue(_covering_ok(vecs, f))

    def test_reconstruct_replaces_previous_state(self):
        bytes_per_ball = (DIM + 1) * 4
        f = BallCoverFilter(dim=DIM, max_memory=bytes_per_ball * 10)
        first = _rand(10, seed=0)
        second = _rand(10, seed=7) + 50.0
        f.construct(first)
        f.construct(second)
        self.assertTrue(all(f.query(second, distance_threshold=1e-3)))
        self.assertFalse(any(f.query(first, distance_threshold=1e-3)))
        self.assertEqual(f.index.ntotal, f.count_balls)


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------

class TestQuery(unittest.TestCase):

    def test_stored_vectors_always_recalled(self):
        vecs = _rand(60)
        bytes_per_ball = (DIM + 1) * 4
        f = BallCoverFilter(dim=DIM, max_memory=bytes_per_ball * 8)
        f.construct(vecs)
        results = f.query(vecs, distance_threshold=1e-3)
        fn = sum(1 for r in results if not r)
        self.assertEqual(fn, 0, f"{fn} stored vectors produced false negatives")

    def test_neighbors_within_threshold_recalled(self):
        vecs = _rand(20)
        bytes_per_ball = (DIM + 1) * 4
        f = BallCoverFilter(dim=DIM, max_memory=bytes_per_ball * 5)
        f.construct(vecs)
        neighbors = _neighbors(vecs, r=R, n=3)
        results = f.query(neighbors, distance_threshold=R)
        fn = sum(1 for r in results if not r)
        self.assertEqual(fn, 0, f"{fn}/{len(neighbors)} close neighbors not recalled")

    def test_distant_vectors_rejected(self):
        vecs = _rand(30)
        bytes_per_ball = (DIM + 1) * 4
        f = BallCoverFilter(dim=DIM, max_memory=bytes_per_ball * 5)
        f.construct(vecs)
        self.assertFalse(any(f.query(vecs + 1000.0, distance_threshold=R)))

    def test_query_returns_list_of_bools(self):
        vecs = _rand(20)
        f = BallCoverFilter(dim=DIM, max_memory=10_000)
        f.construct(vecs)
        results = f.query(vecs, distance_threshold=R)
        self.assertIsInstance(results, list)
        self.assertTrue(all(isinstance(r, (bool, np.bool_)) for r in results))

    def test_batch_equals_individual(self):
        vecs = _rand(50)
        queries = _rand(15, seed=99)
        bytes_per_ball = (DIM + 1) * 4
        f = BallCoverFilter(dim=DIM, max_memory=bytes_per_ball * 6)
        f.construct(vecs)
        self.assertEqual(f.query(queries, distance_threshold=R),
                         [f.query(q[None, :], distance_threshold=R)[0] for q in queries])

    def test_query_empty_batch_returns_empty_list(self):
        f = BallCoverFilter(dim=DIM, max_memory=10_000)
        f.construct(_rand(20))
        self.assertEqual(f.query(np.empty((0, DIM), dtype=np.float32), distance_threshold=R), [])

    def test_query_before_construct_raises(self):
        f = BallCoverFilter(dim=DIM, max_memory=10_000)
        with self.assertRaises(AssertionError):
            f.query(np.zeros((1, DIM), dtype=np.float32), distance_threshold=R)

    def test_query_on_empty_constructed_filter_raises(self):
        """count_balls == 0 means faiss has nothing to search against."""
        f = BallCoverFilter(dim=DIM, max_memory=10_000)
        f.construct(np.empty((0, DIM), dtype=np.float32))
        with self.assertRaises(AssertionError):
            f.query(_rand(10), distance_threshold=R)

    def test_single_point_just_outside_threshold_rejected(self):
        vec = _rand(1, seed=42)
        f = BallCoverFilter(dim=DIM, max_memory=10_000)
        f.construct(vec)
        self.assertTrue(f.query(vec, distance_threshold=R)[0])

        direction = np.random.default_rng(43).standard_normal((1, DIM)).astype(np.float32)
        direction /= np.linalg.norm(direction)
        just_outside = vec + direction * (R + 0.05)
        self.assertFalse(f.query(just_outside, distance_threshold=R)[0])

    def test_larger_threshold_increases_recall(self):
        stored = _rand(10, seed=0)
        queries = _neighbors(stored, r=0.6 * 0.95, n=2)
        f = BallCoverFilter(dim=DIM, max_memory=10_000)
        f.construct(stored)
        recall_small = sum(f.query(queries, distance_threshold=0.1))
        recall_large = sum(f.query(queries, distance_threshold=0.7))
        self.assertGreaterEqual(recall_large, recall_small)


# ---------------------------------------------------------------------------
# distance_to_balls_surfaces
# ---------------------------------------------------------------------------

class TestDistanceToBallsSurfaces(unittest.TestCase):

    def test_shape(self):
        bytes_per_ball = (DIM + 1) * 4
        f = BallCoverFilter(dim=DIM, max_memory=bytes_per_ball * 5)
        f.construct(_rand(50, seed=3))
        queries = _rand(7, seed=1)
        surface = f.distance_to_balls_surfaces(queries)
        self.assertEqual(surface.shape, (7, f.count_balls))

    def test_center_is_non_positive_on_its_own_surface(self):
        f = BallCoverFilter(dim=DIM, max_memory=10_000)
        vecs = _rand(20, seed=2)
        f.construct(vecs)
        centers = _stored_centers(f)
        surface = f.distance_to_balls_surfaces(centers)
        self.assertTrue(np.all(surface.min(axis=1) <= 1e-4))

    def test_far_point_is_positive(self):
        f = BallCoverFilter(dim=DIM, max_memory=10_000)
        f.construct(_rand(20, seed=2))
        far = np.full((1, DIM), 1000.0, dtype=np.float32)
        surface = f.distance_to_balls_surfaces(far)
        self.assertTrue(np.all(surface > 0))


# ---------------------------------------------------------------------------
# add
# ---------------------------------------------------------------------------

class TestAdd(unittest.TestCase):

    def test_add_absorbs_into_nearest_ball_without_growing_count(self):
        f = BallCoverFilter(dim=DIM, max_memory=10_000)
        f.construct(_rand(5))
        before = f.count_balls
        f.add(np.full((1, DIM), 500.0, dtype=np.float32))
        self.assertEqual(f.count_balls, before)

    def test_added_vector_inflates_ball_radius(self):
        f = BallCoverFilter(dim=4, max_memory=(4 + 1) * 4 * 3)
        base = np.zeros((1, 4), dtype=np.float32)
        f.construct(base)
        self.assertAlmostEqual(float(f.radii[0]), 0.0)
        far = np.array([[3.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        f.add(far)
        self.assertAlmostEqual(float(f.radii[0]), 3.0, places=4)

    def test_added_vector_is_recalled(self):
        base = _rand(20, seed=0)
        extra = _rand(20, seed=5) + 50.0
        bytes_per_ball = (DIM + 1) * 4
        f = BallCoverFilter(dim=DIM, max_memory=bytes_per_ball * 10)
        f.construct(base)
        f.add(extra)
        self.assertTrue(all(f.query(base, distance_threshold=1e-3)), "base vectors lost after add")
        self.assertTrue(all(f.query(extra, distance_threshold=1e-3)), "added vectors not recalled")

    def test_add_batch_updates_radius_to_max_distance(self):
        f = BallCoverFilter(dim=4, max_memory=(4 + 1) * 4 * 3)
        base = np.zeros((1, 4), dtype=np.float32)
        f.construct(base)
        batch = np.array([[1, 0, 0, 0], [3, 0, 0, 0]], dtype=np.float32)
        f.add(batch)
        self.assertAlmostEqual(float(f.radii[0]), 3.0, places=4)

    def test_point_already_covered_does_not_shrink_radius(self):
        f = BallCoverFilter(dim=4, max_memory=(4 + 1) * 4 * 3)
        base = np.zeros((1, 4), dtype=np.float32)
        f.construct(base)
        f.add(np.array([[5.0, 0.0, 0.0, 0.0]], dtype=np.float32))
        radius_after_first = float(f.radii[0])
        f.add(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32))
        self.assertAlmostEqual(float(f.radii[0]), radius_after_first, places=4)

    def test_add_before_construct_raises(self):
        f = BallCoverFilter(dim=DIM, max_memory=10_000)
        with self.assertRaises(AssertionError):
            f.add(_rand(3))


# ---------------------------------------------------------------------------
# get_bytesize
# ---------------------------------------------------------------------------

class TestBytesize(unittest.TestCase):

    def test_bytesize_before_construct_is_zero(self):
        f = BallCoverFilter(dim=DIM, max_memory=10_000)
        self.assertEqual(f.get_bytesize(), 0)

    def test_bytesize_after_construct(self):
        bytes_per_ball = (DIM + 1) * 4
        f = BallCoverFilter(dim=DIM, max_memory=bytes_per_ball * 6)
        f.construct(_rand(100))
        self.assertEqual(f.get_bytesize(), f.count_balls * f.bytes_per_ball())

    def test_bytesize_within_max_memory_budget(self):
        bytes_per_ball = (DIM + 1) * 4
        budget = bytes_per_ball * 8
        f = BallCoverFilter(dim=DIM, max_memory=budget)
        f.construct(_rand(200))
        self.assertLessEqual(f.get_bytesize(), budget)


if __name__ == "__main__":
    unittest.main(verbosity=2)
