import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'src'))
import unittest
import faiss
import numpy as np
from src.filters.ball_cover.ball_cover_filter import BallCoverFilter, Quantize

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
# insert
# ---------------------------------------------------------------------------

class TestInsert(unittest.TestCase):

    def test_insert_absorbs_into_nearest_ball_without_growing_count(self):
        f = BallCoverFilter(dim=DIM, max_memory=10_000, quantize=Quantize.NONE)
        f.construct(_rand(5))
        before = f.count_balls
        f.insert(np.full((1, DIM), 500.0, dtype=np.float32))
        self.assertEqual(f.count_balls, before)

    def test_added_vector_inflates_ball_radius(self):
        f = BallCoverFilter(dim=4, max_memory=(4 + 1) * 4 * 3)
        base = np.zeros((1, 4), dtype=np.float32)
        f.construct(base)
        self.assertAlmostEqual(float(f.radii[0]), 0.0)
        far = np.array([[3.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        f.insert(far)
        self.assertAlmostEqual(float(f.radii[0]), 3.0, places=4)

    def test_added_vector_is_recalled(self):
        base = _rand(20, seed=0)
        extra = _rand(20, seed=5) + 50.0
        bytes_per_ball = (DIM + 1) * 4
        f = BallCoverFilter(dim=DIM, max_memory=bytes_per_ball * 10, quantize=Quantize.NONE)
        f.construct(base)
        f.insert(extra)
        self.assertTrue(all(f.query(base, distance_threshold=1e-3)), "base vectors lost after insert")
        self.assertTrue(all(f.query(extra, distance_threshold=1e-3)), "inserted vectors not recalled")

    def test_insert_batch_updates_radius_to_max_distance(self):
        f = BallCoverFilter(dim=4, max_memory=(4 + 1) * 4 * 3)
        base = np.zeros((1, 4), dtype=np.float32)
        f.construct(base)
        batch = np.array([[1, 0, 0, 0], [3, 0, 0, 0]], dtype=np.float32)
        f.insert(batch)
        self.assertAlmostEqual(float(f.radii[0]), 3.0, places=4)

    def test_point_already_covered_does_not_shrink_radius(self):
        f = BallCoverFilter(dim=4, max_memory=(4 + 1) * 4 * 3)
        base = np.zeros((1, 4), dtype=np.float32)
        f.construct(base)
        f.insert(np.array([[5.0, 0.0, 0.0, 0.0]], dtype=np.float32))
        radius_after_first = float(f.radii[0])
        f.insert(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32))
        self.assertAlmostEqual(float(f.radii[0]), radius_after_first, places=4)

    def test_insert_before_construct_bootstraps(self):
        """With no existing balls to absorb into, insert() must bootstrap via
        construct() rather than crashing on an empty index search."""
        f = BallCoverFilter(dim=DIM, max_memory=10_000, quantize=Quantize.NONE)
        vecs = _rand(3)
        f.insert(vecs)
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
        sq4_probe = BallCoverFilter(dim=DIM, max_memory=1, quantize=Quantize.SQ4)
        bpb_sq4 = sq4_probe.bytes_per_ball()
        # SQ4's own trained-parameter codebook is fixed overhead on top of its
        # per-ball code (see BallCoverFilter._codebook_bytes) -- omitting it
        # from the budget here would starve max_count_balls below k once
        # calculate_max_balls correctly reserves room for it.
        codebook_sq4 = sq4_probe._codebook_bytes()

        f_exact = BallCoverFilter(dim=DIM, max_memory=bpb_exact * k, quantize=Quantize.NONE)
        f_quant = BallCoverFilter(dim=DIM, max_memory=bpb_sq4 * k + codebook_sq4, quantize=Quantize.SQ4)
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
        bytes_per_ball_fp16 = 2 * DIM + 4  # largest per-ball code size among the candidates
        n_balls_budget = 10
        for q in _EXPLICIT_QUANTIZERS + (Quantize.NONE,):
            # Each quantizer's own fixed codebook overhead (0 for FP16/BF16,
            # up to ~2KB for SQ8T at this DIM) must fit in the budget too --
            # see BallCoverFilter._codebook_bytes -- on top of the same
            # per-ball allowance every quantizer gets.
            codebook = BallCoverFilter(dim=DIM, max_memory=1, quantize=q)._codebook_bytes()
            budget = bytes_per_ball_fp16 * n_balls_budget + codebook
            f = BallCoverFilter(dim=DIM, max_memory=budget, quantize=q)
            f.construct(_rand(200, seed=8))
            self.assertLessEqual(f.get_bytesize(), budget, f"quantizer {q} exceeded its budget")

    def test_empty_construct_falls_back_to_none_for_every_quantizer(self):
        for q in _EXPLICIT_QUANTIZERS:
            f = BallCoverFilter(dim=DIM, max_memory=10_000, quantize=q)
            f.construct(np.empty((0, DIM), dtype=np.float32))
            self.assertEqual(f.count_balls, 0)
            self.assertIsInstance(f.index, faiss.IndexFlatL2)

    def test_insert_after_quantized_construct_still_recalls(self):
        for q in _EXPLICIT_QUANTIZERS:
            f = BallCoverFilter(dim=DIM, max_memory=10_000, quantize=q)
            base = _rand(10, seed=9)
            f.construct(base)
            far = np.full((1, DIM), 500.0, dtype=np.float32)
            f.insert(far)
            self.assertTrue(f.query(far, distance_threshold=1e-2)[0], f"quantizer {q} lost an inserted vector")
            self.assertTrue(all(f.query(base, distance_threshold=1e-3)), f"quantizer {q} lost a base vector after insert")


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

    def test_construction_level_is_accepted_and_ignored(self):
        bytes_per_ball = (DIM + 1) * 4
        f1 = BallCoverFilter(dim=DIM, max_memory=bytes_per_ball * 6, construction_level=1, quantize=Quantize.NONE)
        f2 = BallCoverFilter(dim=DIM, max_memory=bytes_per_ball * 6, construction_level=2, quantize=Quantize.NONE)
        vecs = _rand(40, seed=3)
        f1.construct(vecs)
        f2.construct(vecs)
        # same construction algorithm regardless of construction_level
        np.testing.assert_array_equal(f1.radii[:f1.count_balls], f2.radii[:f2.count_balls])

    def test_insert_grows_count_balls_from_empty(self):
        f = BallCoverFilter(dim=DIM, max_memory=10_000, quantize=Quantize.NONE)
        vecs = _rand(10)
        f.insert(vecs)
        self.assertEqual(f.count_balls, 10)
        self.assertTrue(all(f.query(vecs, distance_threshold=1e-3)))

    def test_all_radiuses_property(self):
        f = BallCoverFilter(dim=DIM, max_memory=10_000, quantize=Quantize.NONE)
        f.construct(_rand(20))
        np.testing.assert_array_equal(f.all_radiuses, f.radii[:f.count_balls])

    def test_insert_bootstraps_a_streamed_filter_from_empty(self):
        """insert() with no existing balls to absorb into (e.g. a filter
        constructed on zero vectors, then streamed one vector at a time)
        must bootstrap via construct(), not crash on an empty index search."""
        bytes_per_ball = (DIM + 1) * 4
        f = BallCoverFilter(dim=DIM, max_memory=bytes_per_ball * 8, quantize=Quantize.NONE)
        f.construct(np.empty((0, DIM), dtype=np.float32))
        stream = _rand(15, seed=4)
        for v in stream:
            f.insert(v[np.newaxis, :])
        self.assertTrue(all(f.query(stream, distance_threshold=1e-2)))


# ---------------------------------------------------------------------------
# Insertion methods (insertion_level: nearest / least_inflation / merge)
# ---------------------------------------------------------------------------

_INSERTION_LEVELS = ("nearest", "least_inflation", "merge")


def _stream_insert(f: BallCoverFilter, base: np.ndarray, stream: np.ndarray, batch: int = 3) -> np.ndarray:
    f.construct(base)
    all_vecs = [base]
    for i in range(0, len(stream), batch):
        chunk = stream[i:i + batch]
        f.insert(chunk)
        all_vecs.append(chunk)
    return np.vstack(all_vecs)


class TestInsertionLevels(unittest.TestCase):

    def test_default_insertion_level_is_none_and_dispatches_to_nearest(self):
        f = BallCoverFilter(dim=DIM, max_memory=10_000, quantize=Quantize.NONE)
        self.assertIsNone(f.insertion_level)
        vecs = _rand(5, seed=1)
        f.construct(vecs)
        far = np.full((1, DIM), 500.0, dtype=np.float32)
        f.insert(far)  # must not raise, and must behave like insert_nearest
        self.assertTrue(f.query(far, distance_threshold=1e-2)[0])

    def test_unknown_insertion_level_raises(self):
        f = BallCoverFilter(dim=DIM, max_memory=10_000, quantize=Quantize.NONE, insertion_level="bogus")
        f.construct(_rand(5))
        with self.assertRaises(ValueError):
            f.insert(_rand(1, seed=9))

    def test_no_false_negatives_across_all_levels(self):
        bytes_per_ball = (DIM + 1) * 4
        for level in _INSERTION_LEVELS:
            f = BallCoverFilter(dim=DIM, max_memory=bytes_per_ball * 8, quantize=Quantize.NONE, insertion_level=level)
            base = _rand(8, seed=1)
            stream = _rand(60, seed=2)
            stored = _stream_insert(f, base, stream)
            recalled = f.query(stored, distance_threshold=1e-2)
            self.assertTrue(all(recalled), f"insertion_level={level!r} produced a false negative")

    def test_merge_never_exceeds_budget(self):
        bytes_per_ball = (DIM + 1) * 4
        f = BallCoverFilter(dim=DIM, max_memory=bytes_per_ball * 6, quantize=Quantize.NONE, insertion_level="merge")
        base = _rand(6, seed=3)
        stream = _rand(80, seed=4)
        _stream_insert(f, base, stream, batch=1)
        self.assertLessEqual(f.count_balls, f.max_count_balls)

    def test_no_false_negatives_under_quantization(self):
        for level in ("merge",):
            for q in (Quantize.SQ8, Quantize.PQ):
                f = BallCoverFilter(dim=DIM, max_memory=10_000, quantize=q, insertion_level=level)
                base = _rand(10, seed=1)
                stream = _rand(40, seed=2) * 3
                stored = _stream_insert(f, base, stream, batch=4)
                recalled = f.query(stored, distance_threshold=1e-2)
                self.assertTrue(all(recalled), f"insertion_level={level!r} quantize={q} produced a false negative")

    def test_least_inflation_diverges_from_nearest_when_cheaper(self):
        """A concrete case where the raw-nearest ball is NOT the cheapest to
        grow: ball A is nearer (dist=5.0) but a bare singleton (r=0), while
        ball B is fractionally farther (dist=5.01) but already almost as
        large as the new distance (r=4.05) -- growing B costs far less
        volume than blowing up A from 0. insertion_level="nearest" must
        grow A (raw nearest); insertion_level="least_inflation" must grow
        B instead."""
        dim = 16

        def make(level):
            f = BallCoverFilter(dim=dim, max_memory=10_000, quantize=Quantize.NONE, insertion_level=level)
            a = np.zeros(dim, dtype=np.float32)
            b = np.zeros(dim, dtype=np.float32)
            b[0] = 10.01
            f.construct(np.vstack([a, b]))
            f.radii[0] = 0.0
            f.radii[1] = 4.05
            return f

        x = np.zeros((1, dim), dtype=np.float32)
        x[0, 0] = 5.0  # dist to A = 5.0, dist to B = 5.01

        f_nearest = make("nearest")
        f_least = make("least_inflation")
        f_nearest.insert(x)
        f_least.insert(x)

        self.assertAlmostEqual(float(f_nearest.radii[0]), 5.0, places=4)
        self.assertAlmostEqual(float(f_nearest.radii[1]), 4.05, places=4)
        self.assertAlmostEqual(float(f_least.radii[0]), 0.0, places=4)
        self.assertAlmostEqual(float(f_least.radii[1]), 5.01, places=4)

    def test_merge_evicts_cheapest_pair_globally_not_just_the_new_point(self):
        """Two existing balls are near-duplicates (essentially free to
        merge) and a third is far away; a new point lands in genuinely new
        territory. The globally cheapest eviction is the two near-duplicate
        EXISTING balls, not anything involving the new point -- both the far
        ball and the new point must survive as their own balls."""
        dim = 8
        f = BallCoverFilter(dim=dim, max_memory=10_000, quantize=Quantize.NONE, insertion_level="merge")
        f.max_count_balls = 3
        f.radii = np.zeros(3, dtype=np.float32)
        near1 = np.zeros(dim, dtype=np.float32)
        near2 = np.full(dim, 0.05, dtype=np.float32)
        far = np.full(dim, 100.0, dtype=np.float32)
        f.construct(np.vstack([near1, near2, far]))

        new_point = np.full((1, dim), -100.0, dtype=np.float32)
        f.insert(new_point)

        self.assertEqual(f.count_balls, 3)
        self.assertTrue(f.query(new_point, distance_threshold=1e-2)[0], "new point in new territory was lost")
        self.assertTrue(f.query(far[None, :], distance_threshold=1e-2)[0], "far existing ball was lost")

    def test_cheapest_pair_scaling_handles_merged_radius_larger_than_any_existing(self):
        """Regression test: _cheapest_pair's log-space scale M used to be
        computed only from existing radii, but a merged ball's radius can
        exceed every existing radius -- scaling by too-small an M let
        (r_merged/M)**d exceed 1 and overflow exp() at high dim."""
        dim = 384
        f = BallCoverFilter(dim=dim, max_memory=(dim + 1) * 4 * 10, quantize=Quantize.NONE, insertion_level="merge")
        base = _rand(10, dim=dim, seed=0)
        f.construct(base)
        with np.errstate(over="raise"):
            f.insert(_rand(5, dim=dim, seed=1) * 3)
        self.assertFalse(np.any(np.isnan(f.radii[:f.count_balls])))
        self.assertFalse(np.any(np.isinf(f.radii[:f.count_balls])))


if __name__ == "__main__":
    unittest.main(verbosity=2)
