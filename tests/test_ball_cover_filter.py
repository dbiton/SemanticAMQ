import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'src'))
import unittest
import faiss
import numpy as np
from src.filters.ball_cover_filter import BallCoverFilter, Quantize

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
        f = BallCoverFilter(dim=DIM, max_memory=10_000, quantize=Quantize.NONE)
        self.assertEqual(f.bytes_per_ball(), (DIM + 1) * 4)

    def test_max_memory_derives_ball_count(self):
        dim = 16
        bytes_per_ball = (dim + 1) * 4
        budget = bytes_per_ball * 7
        f = BallCoverFilter(dim=dim, max_memory=budget, quantize=Quantize.NONE)
        self.assertEqual(f.max_count_balls, 7)

    def test_max_memory_rounds_down(self):
        dim = 16
        bytes_per_ball = (dim + 1) * 4
        budget = bytes_per_ball * 7 + 1
        f = BallCoverFilter(dim=dim, max_memory=budget, quantize=Quantize.NONE)
        self.assertEqual(f.max_count_balls, 7)

    def test_max_memory_yields_at_least_one_ball(self):
        f = BallCoverFilter(dim=DIM, max_memory=1, quantize=Quantize.NONE)
        self.assertEqual(f.max_count_balls, 1)

    def test_initial_state_empty(self):
        f = BallCoverFilter(dim=DIM, max_memory=10_000, quantize=Quantize.NONE)
        self.assertEqual(f.count_balls, 0)
        self.assertEqual(f.index.ntotal, 0)
        self.assertEqual(len(f.radii), f.max_count_balls)
        np.testing.assert_array_equal(f.radii, np.zeros(f.max_count_balls, dtype=np.float32))

    def test_dim_stored(self):
        f = BallCoverFilter(dim=DIM, max_memory=10_000, quantize=Quantize.NONE)
        self.assertEqual(f.dim, DIM)
        self.assertEqual(f.index.d, DIM)


# ---------------------------------------------------------------------------
# construct_enclosing_balls
# ---------------------------------------------------------------------------

class TestConstructEnclosingBalls(unittest.TestCase):

    def test_trivial_case_fewer_vectors_than_balls(self):
        f = BallCoverFilter(dim=DIM, max_memory=10_000, quantize=Quantize.NONE)  # max_count_balls large
        vecs = _rand(5)
        centers, radii = f.construct_enclosing_balls(vecs)
        np.testing.assert_array_equal(centers, vecs)
        np.testing.assert_allclose(radii, 0.0)

    def test_trivial_case_n_equals_max_balls(self):
        bytes_per_ball = (DIM + 1) * 4
        f = BallCoverFilter(dim=DIM, max_memory=bytes_per_ball * 5, quantize=Quantize.NONE)
        vecs = _rand(5)
        centers, radii = f.construct_enclosing_balls(vecs)
        self.assertEqual(len(centers), 5)
        np.testing.assert_allclose(radii, 0.0)

    def test_nontrivial_case_returns_exactly_max_balls_centers(self):
        bytes_per_ball = (DIM + 1) * 4
        f = BallCoverFilter(dim=DIM, max_memory=bytes_per_ball * 5, quantize=Quantize.NONE)
        vecs = _rand(50, seed=3)
        centers, radii = f.construct_enclosing_balls(vecs)
        self.assertEqual(len(centers), 5)
        self.assertEqual(len(radii), 5)

    def test_nontrivial_case_covers_all_vectors(self):
        bytes_per_ball = (DIM + 1) * 4
        f = BallCoverFilter(dim=DIM, max_memory=bytes_per_ball * 5, quantize=Quantize.NONE)
        vecs = _rand(50, seed=3)
        centers, radii = f.construct_enclosing_balls(vecs)
        dists = np.linalg.norm(vecs[:, None, :] - centers[None, :, :], axis=2)
        within = dists <= (radii[None, :] + 1e-3)
        self.assertTrue(np.all(np.any(within, axis=1)))

    def test_empty_input(self):
        f = BallCoverFilter(dim=DIM, max_memory=10_000, quantize=Quantize.NONE)
        centers, radii = f.construct_enclosing_balls(np.empty((0, DIM), dtype=np.float32))
        self.assertEqual(len(centers), 0)
        self.assertEqual(len(radii), 0)


# ---------------------------------------------------------------------------
# construct
# ---------------------------------------------------------------------------

class TestConstruct(unittest.TestCase):

    def test_respects_max_balls(self):
        bytes_per_ball = (DIM + 1) * 4
        f = BallCoverFilter(dim=DIM, max_memory=bytes_per_ball * 8, quantize=Quantize.NONE)
        f.construct(_rand(100))
        self.assertLessEqual(f.count_balls, 8)
        self.assertEqual(f.index.ntotal, f.count_balls)

    def test_empty_input_leaves_zero_balls(self):
        f = BallCoverFilter(dim=DIM, max_memory=10_000, quantize=Quantize.NONE)
        f.construct(np.empty((0, DIM), dtype=np.float32))
        self.assertEqual(f.count_balls, 0)
        self.assertEqual(f.index.ntotal, 0)

    def test_trivial_case_zero_radius_balls(self):
        vecs = _rand(5)
        f = BallCoverFilter(dim=DIM, max_memory=10_000, quantize=Quantize.NONE)
        f.construct(vecs)
        self.assertEqual(f.count_balls, 5)
        np.testing.assert_allclose(f.radii[:5], 0.0, atol=1e-6)

    def test_radii_beyond_count_balls_are_zeroed(self):
        bytes_per_ball = (DIM + 1) * 4
        f = BallCoverFilter(dim=DIM, max_memory=bytes_per_ball * 20, quantize=Quantize.NONE)
        f.construct(_rand(5))
        self.assertEqual(f.count_balls, 5)
        np.testing.assert_array_equal(f.radii[5:], np.zeros(len(f.radii) - 5, dtype=np.float32))

    def test_covering_invariant_nontrivial_case(self):
        bytes_per_ball = (DIM + 1) * 4
        f = BallCoverFilter(dim=DIM, max_memory=bytes_per_ball * 10, quantize=Quantize.NONE)
        vecs = _rand(100, seed=23)
        f.construct(vecs)
        self.assertLess(f.count_balls, len(vecs))
        self.assertTrue(_covering_ok(vecs, f))

    def test_reconstruct_replaces_previous_state(self):
        bytes_per_ball = (DIM + 1) * 4
        f = BallCoverFilter(dim=DIM, max_memory=bytes_per_ball * 10, quantize=Quantize.NONE)
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
        f = BallCoverFilter(dim=DIM, max_memory=bytes_per_ball * 8, quantize=Quantize.NONE)
        f.construct(vecs)
        results = f.query(vecs, distance_threshold=1e-3)
        fn = sum(1 for r in results if not r)
        self.assertEqual(fn, 0, f"{fn} stored vectors produced false negatives")

    def test_neighbors_within_threshold_recalled(self):
        vecs = _rand(20)
        bytes_per_ball = (DIM + 1) * 4
        f = BallCoverFilter(dim=DIM, max_memory=bytes_per_ball * 5, quantize=Quantize.NONE)
        f.construct(vecs)
        neighbors = _neighbors(vecs, r=R, n=3)
        results = f.query(neighbors, distance_threshold=R)
        fn = sum(1 for r in results if not r)
        self.assertEqual(fn, 0, f"{fn}/{len(neighbors)} close neighbors not recalled")

    def test_distant_vectors_rejected(self):
        vecs = _rand(30)
        bytes_per_ball = (DIM + 1) * 4
        f = BallCoverFilter(dim=DIM, max_memory=bytes_per_ball * 5, quantize=Quantize.NONE)
        f.construct(vecs)
        self.assertFalse(any(f.query(vecs + 1000.0, distance_threshold=R)))

    def test_query_returns_list_of_bools(self):
        vecs = _rand(20)
        f = BallCoverFilter(dim=DIM, max_memory=10_000, quantize=Quantize.NONE)
        f.construct(vecs)
        results = f.query(vecs, distance_threshold=R)
        self.assertIsInstance(results, list)
        self.assertTrue(all(isinstance(r, (bool, np.bool_)) for r in results))

    def test_batch_equals_individual(self):
        vecs = _rand(50)
        queries = _rand(15, seed=99)
        bytes_per_ball = (DIM + 1) * 4
        f = BallCoverFilter(dim=DIM, max_memory=bytes_per_ball * 6, quantize=Quantize.NONE)
        f.construct(vecs)
        self.assertEqual(f.query(queries, distance_threshold=R),
                         [f.query(q[None, :], distance_threshold=R)[0] for q in queries])

    def test_query_empty_batch_returns_empty_list(self):
        f = BallCoverFilter(dim=DIM, max_memory=10_000, quantize=Quantize.NONE)
        f.construct(_rand(20))
        self.assertEqual(f.query(np.empty((0, DIM), dtype=np.float32), distance_threshold=R), [])

    def test_query_before_construct_raises(self):
        f = BallCoverFilter(dim=DIM, max_memory=10_000, quantize=Quantize.NONE)
        with self.assertRaises(AssertionError):
            f.query(np.zeros((1, DIM), dtype=np.float32), distance_threshold=R)

    def test_query_on_empty_constructed_filter_raises(self):
        """count_balls == 0 means faiss has nothing to search against."""
        f = BallCoverFilter(dim=DIM, max_memory=10_000, quantize=Quantize.NONE)
        f.construct(np.empty((0, DIM), dtype=np.float32))
        with self.assertRaises(AssertionError):
            f.query(_rand(10), distance_threshold=R)

    def test_single_point_just_outside_threshold_rejected(self):
        vec = _rand(1, seed=42)
        f = BallCoverFilter(dim=DIM, max_memory=10_000, quantize=Quantize.NONE)
        f.construct(vec)
        self.assertTrue(f.query(vec, distance_threshold=R)[0])

        direction = np.random.default_rng(43).standard_normal((1, DIM)).astype(np.float32)
        direction /= np.linalg.norm(direction)
        just_outside = vec + direction * (R + 0.05)
        self.assertFalse(f.query(just_outside, distance_threshold=R)[0])

    def test_larger_threshold_increases_recall(self):
        stored = _rand(10, seed=0)
        queries = _neighbors(stored, r=0.6 * 0.95, n=2)
        f = BallCoverFilter(dim=DIM, max_memory=10_000, quantize=Quantize.NONE)
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
        f = BallCoverFilter(dim=DIM, max_memory=bytes_per_ball * 5, quantize=Quantize.NONE)
        f.construct(_rand(50, seed=3))
        queries = _rand(7, seed=1)
        surface = f.distance_to_balls_surfaces(queries)
        self.assertEqual(surface.shape, (7, f.count_balls))

    def test_center_is_non_positive_on_its_own_surface(self):
        f = BallCoverFilter(dim=DIM, max_memory=10_000, quantize=Quantize.NONE)
        vecs = _rand(20, seed=2)
        f.construct(vecs)
        centers = _stored_centers(f)
        surface = f.distance_to_balls_surfaces(centers)
        self.assertTrue(np.all(surface.min(axis=1) <= 1e-4))

    def test_far_point_is_positive(self):
        f = BallCoverFilter(dim=DIM, max_memory=10_000, quantize=Quantize.NONE)
        f.construct(_rand(20, seed=2))
        far = np.full((1, DIM), 1000.0, dtype=np.float32)
        surface = f.distance_to_balls_surfaces(far)
        self.assertTrue(np.all(surface > 0))


# ---------------------------------------------------------------------------
# add
# ---------------------------------------------------------------------------

class TestAdd(unittest.TestCase):

    def test_add_absorbs_into_nearest_ball_without_growing_count(self):
        f = BallCoverFilter(dim=DIM, max_memory=10_000, quantize=Quantize.NONE)
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
        f = BallCoverFilter(dim=DIM, max_memory=bytes_per_ball * 10, quantize=Quantize.NONE)
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

    def test_add_before_construct_bootstraps(self):
        """With no existing balls to absorb into, add() must bootstrap via
        construct() rather than crashing on an empty index search."""
        f = BallCoverFilter(dim=DIM, max_memory=10_000, quantize=Quantize.NONE)
        vecs = _rand(3)
        f.add(vecs)
        self.assertEqual(f.count_balls, 3)
        self.assertTrue(all(f.query(vecs, distance_threshold=1e-3)))


# ---------------------------------------------------------------------------
# get_bytesize
# ---------------------------------------------------------------------------

class TestBytesize(unittest.TestCase):

    def test_bytesize_before_construct_is_zero(self):
        f = BallCoverFilter(dim=DIM, max_memory=10_000, quantize=Quantize.NONE)
        self.assertEqual(f.get_bytesize(), 0)

    def test_bytesize_after_construct(self):
        bytes_per_ball = (DIM + 1) * 4
        f = BallCoverFilter(dim=DIM, max_memory=bytes_per_ball * 6, quantize=Quantize.NONE)
        f.construct(_rand(100))
        self.assertEqual(f.get_bytesize(), f.count_balls * f.bytes_per_ball())

    def test_bytesize_within_max_memory_budget(self):
        bytes_per_ball = (DIM + 1) * 4
        budget = bytes_per_ball * 8
        f = BallCoverFilter(dim=DIM, max_memory=budget, quantize=Quantize.NONE)
        f.construct(_rand(200))
        self.assertLessEqual(f.get_bytesize(), budget)


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

_EXPLICIT_QUANTIZERS = (Quantize.SQ8, Quantize.SQ6, Quantize.SQ4, Quantize.FP16, Quantize.BF16,
                        Quantize.SQ4T, Quantize.SQ8T, Quantize.PQ)


_PQ_QUANTIZERS = (Quantize.PQ,)


def _assert_index_matches_quantizer(test, index, q):
    if q == Quantize.NONE:
        test.assertIsInstance(index, faiss.IndexFlatL2)
    elif q in _PQ_QUANTIZERS:
        test.assertIsInstance(index, faiss.IndexPQ)
    else:
        test.assertIsInstance(index, faiss.IndexScalarQuantizer)


class TestQuantize(unittest.TestCase):

    def test_none_is_unquantized_flat_index(self):
        f = BallCoverFilter(dim=DIM, max_memory=10_000, quantize=Quantize.NONE)
        f.construct(_rand(20))
        self.assertIsInstance(f.index, faiss.IndexFlatL2)
        np.testing.assert_allclose(f.radii[:f.count_balls], 0.0, atol=1e-6)

    def test_default_quantize_is_none(self):
        f = BallCoverFilter(dim=DIM, max_memory=10_000)
        self.assertEqual(f.quantize, Quantize.NONE)

    def test_explicit_quantizer_is_used(self):
        for q in _EXPLICIT_QUANTIZERS:
            f = BallCoverFilter(dim=DIM, max_memory=10_000, quantize=q)
            f.construct(_rand(30, seed=1))
            _assert_index_matches_quantizer(self, f.index, q)

    def test_quantization_inflates_radius_above_exact_geometry(self):
        """A quantized build's radii must be >= the exact (unquantized) radii,
        never smaller -- shrinking would risk false negatives. Both filters are
        pinned to the same ball budget k so construct_enclosing_balls (which
        does not know about quantize) picks identical Gonzalez centers."""
        vecs = _rand(40, seed=5)
        k = 6
        bpb_exact = BallCoverFilter(dim=DIM, max_memory=1, quantize=Quantize.NONE).bytes_per_ball()
        bpb_sq4 = BallCoverFilter(dim=DIM, max_memory=1, quantize=Quantize.SQ4).bytes_per_ball()

        f_exact = BallCoverFilter(dim=DIM, max_memory=bpb_exact * k, quantize=Quantize.NONE)
        f_quant = BallCoverFilter(dim=DIM, max_memory=bpb_sq4 * k, quantize=Quantize.SQ4)
        self.assertEqual(f_exact.max_count_balls, k)
        self.assertEqual(f_quant.max_count_balls, k)

        f_exact.construct(vecs)
        f_quant.construct(vecs)

        self.assertEqual(f_exact.count_balls, f_quant.count_balls)
        self.assertTrue(np.all(f_quant.radii[:f_quant.count_balls] >=
                               f_exact.radii[:f_exact.count_balls] - 1e-6))

    def test_quantization_introduces_no_false_negatives_on_stored_vectors(self):
        vecs = _rand(60, seed=6)
        for q in _EXPLICIT_QUANTIZERS:
            f = BallCoverFilter(dim=DIM, max_memory=10_000, quantize=q)
            f.construct(vecs)
            results = f.query(vecs, distance_threshold=1e-3)
            self.assertTrue(all(results), f"quantizer {q} produced a false negative on stored vectors")

    def test_quantization_introduces_no_false_negatives_on_neighbors(self):
        vecs = _rand(30, seed=7)
        neighbors = _neighbors(vecs, r=R, n=3)
        for q in _EXPLICIT_QUANTIZERS:
            f = BallCoverFilter(dim=DIM, max_memory=10_000, quantize=q)
            f.construct(vecs)
            results = f.query(neighbors, distance_threshold=R)
            self.assertTrue(all(results), f"quantizer {q} produced a false negative on a near neighbor")

    def test_smaller_codes_allow_more_balls_at_same_budget(self):
        dim = DIM
        budget = ((dim + 1) * 4) * 10  # sized for 10 unquantized (or FP16/BF16) balls
        f_sq4 = BallCoverFilter(dim=dim, max_memory=budget, quantize=Quantize.SQ4)
        f_fp16 = BallCoverFilter(dim=dim, max_memory=budget, quantize=Quantize.FP16)
        self.assertGreater(f_sq4.max_count_balls, f_fp16.max_count_balls)

    def test_bytesize_stays_within_budget_for_every_quantizer(self):
        bytes_per_ball_fp16 = 2 * DIM + 4  # largest code size among the candidates
        budget = bytes_per_ball_fp16 * 10
        for q in _EXPLICIT_QUANTIZERS + (Quantize.NONE,):
            f = BallCoverFilter(dim=DIM, max_memory=budget, quantize=q)
            f.construct(_rand(200, seed=8))
            self.assertLessEqual(f.get_bytesize(), budget, f"quantizer {q} exceeded its budget")

    def test_empty_construct_falls_back_to_none_for_every_quantizer(self):
        for q in _EXPLICIT_QUANTIZERS:
            f = BallCoverFilter(dim=DIM, max_memory=10_000, quantize=q)
            f.construct(np.empty((0, DIM), dtype=np.float32))
            self.assertEqual(f.count_balls, 0)
            self.assertIsInstance(f.index, faiss.IndexFlatL2)

    def test_add_after_quantized_construct_still_recalls(self):
        for q in _EXPLICIT_QUANTIZERS:
            f = BallCoverFilter(dim=DIM, max_memory=10_000, quantize=q)
            base = _rand(10, seed=9)
            f.construct(base)
            far = np.full((1, DIM), 500.0, dtype=np.float32)
            f.add(far)
            self.assertTrue(f.query(far, distance_threshold=1e-2)[0], f"quantizer {q} lost an added vector")
            self.assertTrue(all(f.query(base, distance_threshold=1e-3)), f"quantizer {q} lost a base vector after add")


# ---------------------------------------------------------------------------
# Product Quantization (a different quantization family from ScalarQuantizer)
# ---------------------------------------------------------------------------

class TestQuantizePQ(unittest.TestCase):

    def test_pq_m_divides_dim(self):
        for dim in (1, 7, 8, 17, 32, 384):
            pq_m = BallCoverFilter._choose_pq_m(dim)
            self.assertEqual(dim % pq_m, 0, f"dim={dim}, pq_m={pq_m}")

    def test_single_center_falls_back_to_none_instead_of_crashing(self):
        """Regression test: IndexPQ.search segfaults when built with nbits=0
        (which happens if n < 2 centers). A single stored vector must fall
        back to Quantize.NONE, not attempt a 0-bit PQ codebook."""
        f = BallCoverFilter(dim=DIM, max_memory=10_000, quantize=Quantize.PQ)
        f.construct(_rand(1, seed=5))
        self.assertIsInstance(f.index, faiss.IndexFlatL2)
        self.assertTrue(f.query(_rand(1, seed=5), distance_threshold=1e-3)[0])

    def test_two_centers_is_the_minimum_pq_can_use(self):
        f = BallCoverFilter(dim=DIM, max_memory=10_000, quantize=Quantize.PQ)
        vecs = _rand(2, seed=9)
        f.construct(vecs)
        self.assertIsInstance(f.index, faiss.IndexPQ)
        self.assertTrue(all(f.query(vecs, distance_threshold=1e-2)))

    def test_pq_bytesize_stays_within_budget(self):
        bytes_per_ball_fp16 = 2 * DIM + 4
        budget = bytes_per_ball_fp16 * 10
        f = BallCoverFilter(dim=DIM, max_memory=budget, quantize=Quantize.PQ)
        f.construct(_rand(200, seed=8))
        self.assertLessEqual(f.get_bytesize(), budget)

    def test_pq_introduces_no_false_negatives(self):
        vecs = _rand(60, seed=6)
        neighbors = _neighbors(vecs, r=R, n=3)
        f = BallCoverFilter(dim=DIM, max_memory=10_000, quantize=Quantize.PQ)
        f.construct(vecs)
        self.assertTrue(all(f.query(vecs, distance_threshold=1e-2)))
        self.assertTrue(all(f.query(neighbors, distance_threshold=R)))


# ---------------------------------------------------------------------------
# Misc interface: level/merge_criterion acceptance, insert/all_radiuses aliases
# ---------------------------------------------------------------------------

class TestMiscInterface(unittest.TestCase):

    def test_level_and_merge_criterion_are_accepted_and_ignored(self):
        bytes_per_ball = (DIM + 1) * 4
        f1 = BallCoverFilter(dim=DIM, max_memory=bytes_per_ball * 6, level=1, quantize=Quantize.NONE)
        f2 = BallCoverFilter(dim=DIM, max_memory=bytes_per_ball * 6, level=2,
                             merge_criterion="log_space", quantize=Quantize.NONE)
        vecs = _rand(40, seed=3)
        f1.construct(vecs)
        f2.construct(vecs)
        # same construction algorithm regardless of level/merge_criterion
        np.testing.assert_array_equal(f1.radii[:f1.count_balls], f2.radii[:f2.count_balls])

    def test_insert_is_an_alias_for_add(self):
        f = BallCoverFilter(dim=DIM, max_memory=10_000, quantize=Quantize.NONE)
        vecs = _rand(10)
        f.insert(vecs)
        self.assertEqual(f.count_balls, 10)
        self.assertTrue(all(f.query(vecs, distance_threshold=1e-3)))

    def test_all_radiuses_property(self):
        f = BallCoverFilter(dim=DIM, max_memory=10_000, quantize=Quantize.NONE)
        f.construct(_rand(20))
        np.testing.assert_array_equal(f.all_radiuses, f.radii[:f.count_balls])

    def test_add_bootstraps_a_streamed_filter_from_empty(self):
        """add() with no existing balls to absorb into (e.g. a filter
        constructed on zero vectors, then streamed one vector at a time)
        must bootstrap via construct(), not crash on an empty index search."""
        bytes_per_ball = (DIM + 1) * 4
        f = BallCoverFilter(dim=DIM, max_memory=bytes_per_ball * 8, quantize=Quantize.NONE)
        f.construct(np.empty((0, DIM), dtype=np.float32))
        stream = _rand(15, seed=4)
        for v in stream:
            f.add(v[np.newaxis, :])
        self.assertTrue(all(f.query(stream, distance_threshold=1e-2)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
