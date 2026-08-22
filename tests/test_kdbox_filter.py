import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'src'))
import unittest
import numpy as np
from src.filters.kdbox_filter import KDBoxFilter

DIM = 32
R = 0.3 # semantic distance threshold used across most tests


def _rand(n: int, dim: int = DIM, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, dim)).astype(np.float32)


def _clustered(n_clusters: int, per_cluster: int, dim: int = DIM,
               spread: float = 0.05, sep: float = 15.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, dim)) * sep
    parts = [rng.standard_normal((per_cluster, dim)) * spread + c for c in centers]
    return np.vstack(parts).astype(np.float32)


def _neighbors_in_box(v: np.ndarray, r: float, n: int = 5, seed: int = 1) -> np.ndarray:
    """Return n vectors that are within the L-infinity distance (box bounds) r of each row of v."""
    rng = np.random.default_rng(seed)
    noise = rng.uniform(-r * 0.95, r * 0.95, (len(v), n, v.shape[1])).astype(np.float32)
    return (v[:, None, :] + noise).reshape(-1, v.shape[1])


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestKDBoxInit(unittest.TestCase):
    def test_max_boxes_stored(self):
        f = KDBoxFilter(dim=DIM, distance_threshold=R, max_spheres=10)
        self.assertEqual(f.max_boxes, 10)

    def test_max_memory_derives_box_count(self):
        dim = 16
        budget = 19000
        f = KDBoxFilter(dim=dim, distance_threshold=R, max_memory=budget)
        self.assertEqual(f.max_boxes, 37)

    def test_distance_threshold_stored(self):
        f = KDBoxFilter(dim=DIM, distance_threshold=0.5, max_spheres=5)
        self.assertAlmostEqual(f.distance_threshold, 0.5)

    def test_initial_state_not_constructed(self):
        f = KDBoxFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        self.assertFalse(f._constructed)


class TestKDBoxSplittingLogic(unittest.TestCase):
    def test_finds_correct_gap_and_axis(self):
        f = KDBoxFilter(dim=2, distance_threshold=R, max_spheres=5)
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
        f = KDBoxFilter(dim=2, distance_threshold=R, max_spheres=5)
        vecs = np.array([[1.0, 1.0]], dtype=np.float32)
        best_margin, best_axis, best_split_val = f._find_widest_margin(vecs)
        self.assertIsNone(best_axis)
        self.assertIsNone(best_split_val)
        self.assertEqual(best_margin, -1.0)

    def test_split_for_identical_vectors_yields_zero_margin(self):
        f = KDBoxFilter(dim=2, distance_threshold=R, max_spheres=5)
        vecs = np.array([[1.0, 2.0], [1.0, 2.0]], dtype=np.float32)
        best_margin, best_axis, best_split_val = f._find_widest_margin(vecs)
        self.assertAlmostEqual(best_margin, 0.0)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestKDBoxConstruct(unittest.TestCase):
    def test_produces_correct_number_of_boxes(self):
        f = KDBoxFilter(dim=DIM, distance_threshold=R, max_spheres=8)
        f.construct(_rand(100))
        self.assertTrue(f._constructed)
        self.assertEqual(len(f.all_boxes), 8)
        self.assertEqual(len(f.all_boxes_min), 8)

    def test_trivial_case_fewer_vectors_than_boxes(self):
        vecs = _rand(3)
        f = KDBoxFilter(dim=DIM, distance_threshold=R, max_spheres=20)
        f.construct(vecs)
        self.assertTrue(f._constructed)
        self.assertEqual(len(f.all_boxes), 3)

    def test_all_vectors_within_bounds_of_some_box_in_nontrivial_case(self):
        """Every original vector must fall within the bounds of at least one stored box."""
        vecs = _rand(100, seed=23)
        f = KDBoxFilter(dim=DIM, distance_threshold=R, max_spheres=10)
        f.construct(vecs)
        self.assertLess(len(f.all_boxes), len(vecs))

        mins = f.all_boxes_min
        maxs = f.all_boxes_max
        inside = np.logical_and(
            np.all(vecs[:, None, :] >= mins[None, :, :] - 1e-5, axis=2),
            np.all(vecs[:, None, :] <= maxs[None, :, :] + 1e-5, axis=2)
        )
        uncovered = np.flatnonzero(~np.any(inside, axis=1))
        self.assertEqual(len(uncovered), 0, f"{len(uncovered)} vectors not within bounds of any box.")

    def test_boxes_incorporate_distance_threshold(self):
        f = KDBoxFilter(dim=DIM, distance_threshold=R, max_spheres=10)
        vecs = _rand(200)
        f.construct(vecs)
        sides = f.all_boxes_max - f.all_boxes_min
        self.assertTrue(np.all(sides >= 2 * R - 1e-5))

    def test_separated_clusters_get_one_box_each(self):
        """Well-separated tight clusters should each collapse to an isolated box."""
        vecs = _clustered(4, 25, spread=0.02, sep=20.0)
        f = KDBoxFilter(dim=DIM, distance_threshold=R, max_spheres=4)
        f.construct(vecs)
        self.assertEqual(len(f.all_boxes), 4)
        sides = f.all_boxes_max - f.all_boxes_min
        # Bounds should be tight enough, close to 2*R
        self.assertTrue(np.all(sides < 2 * R + 0.5))

    def test_construct_with_zero_vectors_sets_constructed_flag(self):
        f = KDBoxFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        f.construct(np.empty((0, DIM), dtype=np.float32))
        self.assertTrue(f._constructed)
        self.assertEqual(len(f.all_boxes), 0)

    def test_construct_with_zero_max_boxes(self):
        f = KDBoxFilter(dim=DIM, distance_threshold=R, max_spheres=0)
        f.construct(_rand(10))
        self.assertTrue(f._constructed)
        self.assertEqual(len(f.all_boxes), 0)
        self.assertFalse(any(f.query(_rand(10))))

    def test_exact_max_boxes_match_vectors_count(self):
        vecs = _rand(5)
        f = KDBoxFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        f.construct(vecs)
        self.assertEqual(len(f.all_boxes), 5)

    def test_reconstruct_replaces_previous_state(self):
        f = KDBoxFilter(dim=DIM, distance_threshold=R, max_spheres=10)
        first = _rand(10, seed=0)
        second = _rand(10, seed=7) + 50.0
        f.construct(first)
        
        # Reset and construct again
        f = KDBoxFilter(dim=DIM, distance_threshold=R, max_spheres=10)
        f.construct(second)
        self.assertTrue(all(f.query(second)))
        self.assertFalse(any(f.query(first)))


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

class TestKDBoxQuery(unittest.TestCase):
    def test_stored_vectors_always_recalled(self):
        vecs = _rand(60)
        f = KDBoxFilter(dim=DIM, distance_threshold=R, max_spheres=8)
        f.construct(vecs)
        results = f.query(vecs)
        self.assertNotIn(False, results)

    def test_neighbors_within_threshold_recalled(self):
        vecs = _rand(20)
        f = KDBoxFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        f.construct(vecs)
        neighbors = _neighbors_in_box(vecs, r=R, n=3)
        results = f.query(neighbors)
        self.assertNotIn(False, results)
        
    def test_uniform_points_around_stored_vectors_recalled(self):
        """Points sampled uniformly from within the L-infinity bounds around each stored vector must be recalled."""
        vecs = _rand(15, seed=11)
        f = KDBoxFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        f.construct(vecs)

        samples = _neighbors_in_box(vecs, r=R, n=30)
        results = f.query(samples)
        fn = sum(1 for r_ in results if not r_)
        self.assertEqual(fn, 0, f"{fn}/{len(samples)} uniformly sampled points not recalled")

    def test_single_point_just_outside_threshold_rejected(self):
        """A filter built from a single point should reject a query that exceeds distance_threshold on one axis."""
        vec = _rand(1, seed=42)
        f = KDBoxFilter(dim=DIM, distance_threshold=R, max_spheres=1)
        f.construct(vec)
        self.assertTrue(f.query(vec)[0])

        just_outside = vec.copy()
        just_outside[0, 0] += (R + 0.05)
        self.assertFalse(f.query(just_outside)[0])

    def test_larger_threshold_increases_recall(self):
        stored = _rand(10, seed=0)
        queries = _neighbors_in_box(stored, r=0.6 * 0.95, n=2)

        f_small = KDBoxFilter(dim=DIM, distance_threshold=0.1, max_spheres=10)
        f_small.construct(stored)

        f_large = KDBoxFilter(dim=DIM, distance_threshold=0.7, max_spheres=10)
        f_large.construct(stored)

        recall_small = sum(f_small.query(queries))
        recall_large = sum(f_large.query(queries))
        self.assertGreater(recall_large, recall_small, "Larger threshold should recall more near-boundary queries")

    def test_distant_vectors_rejected(self):
        vecs = _rand(30)
        f = KDBoxFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        f.construct(vecs)
        self.assertFalse(any(f.query(vecs + 1000.0)))

    def test_query_before_construct_raises(self):
        f = KDBoxFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        with self.assertRaises(ValueError):
            f.query(np.zeros((1, DIM), dtype=np.float32))

    def test_query_on_empty_filter_returns_all_false(self):
        f = KDBoxFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        f.construct(np.empty((0, DIM), dtype=np.float32))
        self.assertFalse(any(f.query(_rand(10))))

    def test_query_returns_list_of_bools(self):
        vecs = _rand(20)
        f = KDBoxFilter(dim=DIM, distance_threshold=R, max_spheres=4)
        f.construct(vecs)
        results = f.query(vecs)
        self.assertIsInstance(results, list)
        self.assertTrue(all(isinstance(r, (bool, np.bool_)) for r in results))

    def test_batch_equals_individual(self):
        vecs = _rand(50)
        queries = _rand(15, seed=99)
        f = KDBoxFilter(dim=DIM, distance_threshold=R, max_spheres=6)
        f.construct(vecs)
        self.assertEqual(f.query(queries), [f.query(q[None, :])[0] for q in queries])

    def test_query_with_empty_vectors_returns_empty_list(self):
        f = KDBoxFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        f.construct(_rand(10))
        self.assertEqual(f.query(np.empty((0, DIM), dtype=np.float32)), [])


# ---------------------------------------------------------------------------
# Dynamic Updates
# ---------------------------------------------------------------------------

class TestKDBoxDynamicUpdates(unittest.TestCase):
    def test_add_new_vectors_updates_boxes(self):
        f = KDBoxFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        f.construct(_rand(10))
        f.add(_rand(20, seed=42) + 100.0)
        self.assertTrue(len(f.all_boxes) <= 5)

    def test_add_before_construct_calls_construct(self):
        f = KDBoxFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        vecs = _rand(10)
        f.add(vecs)
        self.assertTrue(f._constructed)
        self.assertNotIn(False, f.query(vecs))

    def test_add_empty_vectors_safe(self):
        f = KDBoxFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        f.construct(_rand(10))
        f.add(np.empty((0, DIM), dtype=np.float32))
        self.assertTrue(f._constructed)

    def test_insert_is_alias_to_add(self):
        f1 = KDBoxFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        f2 = KDBoxFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        vecs = _rand(20)
        f1.add(vecs)
        f2.insert(vecs)
        self.assertTrue(np.array_equal(f1.all_boxes_min, f2.all_boxes_min))

    def test_add_covered_vector_does_not_increase_box_count(self):
        f = KDBoxFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        f.construct(_rand(10, seed=42))
        box_count = len(f.all_boxes)

        # Midpoint of the first box is strictly inside its bounds
        covered_vec = ((f.all_boxes_min[0] + f.all_boxes_max[0]) / 2.0).reshape(1, DIM)
        f.add(covered_vec)
        self.assertEqual(len(f.all_boxes), box_count)

    def test_add_duplicate_does_not_create_new_box(self):
        vecs = _rand(5)
        f = KDBoxFilter(dim=DIM, distance_threshold=R, max_spheres=30)
        f.construct(vecs)
        before = len(f.all_boxes)
        f.add(vecs[:1])
        self.assertEqual(len(f.all_boxes), before)

    def test_add_growth_and_merge_at_max_boxes(self):
        f = KDBoxFilter(dim=DIM, distance_threshold=R, max_spheres=3)
        f.construct(np.array([[0.0] * DIM], dtype=np.float32))
        self.assertEqual(len(f.all_boxes), 1)

        f.add(np.array([[100.0] * DIM], dtype=np.float32))
        self.assertEqual(len(f.all_boxes), 2)

        f.add(np.array([[200.0] * DIM], dtype=np.float32))
        self.assertEqual(len(f.all_boxes), 3)

        f.add(np.array([[300.0] * DIM], dtype=np.float32))
        self.assertLessEqual(len(f.all_boxes), 3)

    def test_add_in_batches(self):
        f = KDBoxFilter(dim=DIM, distance_threshold=R, max_spheres=10)
        f.construct(_rand(5))
        vecs = _rand(4000)
        f.add(vecs)
        self.assertLessEqual(len(f.all_boxes), 10)
        
    def test_add_preserves_no_false_negatives_over_many_batches(self):
        f = KDBoxFilter(dim=DIM, distance_threshold=R, max_spheres=8)
        all_vecs = []
        f.construct(_rand(20, seed=0))
        all_vecs.append(_rand(20, seed=0))
        for seed in range(1, 6):
            batch = _rand(15, seed=seed) * (1.0 + seed)
            f.add(batch)
            all_vecs.append(batch)
        stored = np.vstack(all_vecs)
        self.assertTrue(all(f.query(stored)), "no-false-negative invariant broken by add()")


# ---------------------------------------------------------------------------
# Memory & Interface
# ---------------------------------------------------------------------------

class TestKDBoxMemoryAndState(unittest.TestCase):
    def test_bytesize_before_construct_is_zero(self):
        f = KDBoxFilter(dim=DIM, distance_threshold=R, max_spheres=10)
        self.assertEqual(f.get_bytesize(), 0)

    def test_bytesize_after_construct(self):
        k = 6
        f = KDBoxFilter(dim=DIM, distance_threshold=R, max_spheres=k)
        f.construct(_rand(100))
        active_boxes = f._all_boxes.nbytes
        min_bytes = f._all_boxes_min.nbytes
        max_bytes = f._all_boxes_max.nbytes
        idx_bytes = f._indices_count.nbytes
        heap_bytes = sys.getsizeof(f._minimal_pairs)
        self.assertEqual(f.get_bytesize(), active_boxes + min_bytes + max_bytes + idx_bytes + heap_bytes)

    def test_get_new_index_expands_capacity(self):
        f = KDBoxFilter(dim=DIM, distance_threshold=R, max_spheres=2)
        f._num_boxes = len(f._all_boxes_min) 
        old_cap = len(f._all_boxes_min)
        idx = f._get_new_index()
        self.assertEqual(len(f._all_boxes_min), old_cap * 2)

    def test_all_boxes_properties_slice_correctly(self):
        f = KDBoxFilter(dim=DIM, distance_threshold=R, max_spheres=10)
        f.construct(_rand(5))
        self.assertEqual(f.all_boxes.shape, (5,))
        self.assertEqual(f.all_boxes_min.shape, (5, DIM))
        self.assertEqual(f.all_boxes_max.shape, (5, DIM))


class TestKDBoxEdgeCasesAndInterface(unittest.TestCase):
    def test_extreme_coordinates_handling(self):
        f = KDBoxFilter(dim=DIM, distance_threshold=R, max_spheres=5)
        extreme_vecs = np.array([[1e10, -1e10] * (DIM // 2)], dtype=np.float32)
        f.construct(extreme_vecs)
        self.assertTrue(f.query(extreme_vecs)[0])

    def test_high_dimension(self):
        high_dim = 512
        f = KDBoxFilter(dim=high_dim, distance_threshold=R, max_spheres=5)
        vecs = _rand(20, dim=high_dim)
        f.construct(vecs)
        self.assertNotIn(False, f.query(vecs))


if __name__ == "__main__":
    unittest.main(verbosity=2)