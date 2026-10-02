import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'src'))
import unittest
import numpy as np
from src.filters.ball_cover.ball_cover_filter import BallCoverFilter, Quantize
from src.filters.ball_cover.nearest_insert import NearestInsert
from src.filters.ball_cover.min_heap_insert import MinHeapInsert

DIM = 32
TAU = 0.3  # query distance threshold used across most tests


def _rand(n: int, dim: int = DIM, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, dim)).astype(np.float32)


def _clustered(n_clusters: int, per_cluster: int, dim: int = DIM,
               spread: float = 0.05, sep: float = 15.0, seed: int = 0) -> np.ndarray:
    """Vectors with a clear cluster structure."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, dim)) * sep
    parts = [rng.standard_normal((per_cluster, dim)) * spread + c for c in centers]
    return np.vstack(parts).astype(np.float32)


def _neighbors(v: np.ndarray, r: float, n: int = 5, seed: int = 1) -> np.ndarray:
    """n vectors strictly within Euclidean distance r of each row of v."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((len(v), n, v.shape[1])).astype(np.float32)
    noise /= np.linalg.norm(noise, axis=2, keepdims=True) + 1e-9
    scales = rng.uniform(0, r * 0.95, (len(v), n, 1)).astype(np.float32)
    return (v[:, None, :] + noise * scales).reshape(-1, v.shape[1])


def _static(max_spheres: int, **kw) -> BallCoverFilter:
    return BallCoverFilter(DIM, max_spheres=max_spheres, insert_method=None, **kw)


def _covering_ok(vecs: np.ndarray, f: BallCoverFilter, tol: float = 1e-3) -> bool:
    """Every vector lies within its radius (+tol) of some stored ball."""
    centers, radii = f.all_centers, f.all_radiuses
    if len(centers) == 0:
        return len(vecs) == 0
    dists = np.linalg.norm(vecs[:, None, :] - centers[None, :, :], axis=2)
    return bool(np.all((dists - radii[None, :]).min(axis=1) <= tol))


# ---------------------------------------------------------------------------
# Initialisation and budget
# ---------------------------------------------------------------------------

class TestBallCoverInit(unittest.TestCase):

    def test_max_spheres_stored(self):
        self.assertEqual(_static(17).max_spheres, 17)

    def test_max_memory_derives_sphere_count(self):
        f = BallCoverFilter(DIM, max_memory=10_000, insert_method=None)
        self.assertEqual(f.max_spheres, 10_000 // f.bytes_per_ball())

    def test_max_memory_yields_at_least_one_sphere(self):
        self.assertGreaterEqual(BallCoverFilter(DIM, max_memory=1, insert_method=None).max_spheres, 1)

    def test_max_spheres_overrides_max_memory(self):
        self.assertEqual(BallCoverFilter(DIM, max_memory=10_000, max_spheres=3, insert_method=None).max_spheres, 3)


# ---------------------------------------------------------------------------
# construct()
# ---------------------------------------------------------------------------

class TestBallCoverConstruct(unittest.TestCase):

    def test_respects_max_spheres(self):
        f = _static(20)
        f.construct(_rand(300))
        self.assertLessEqual(len(f.all_radiuses), 20)

    def test_empty_input(self):
        f = _static(10)
        f.construct(np.empty((0, DIM), dtype=np.float32))
        self.assertEqual(len(f.all_radiuses), 0)

    def test_fewer_vectors_than_spheres_gives_zero_radius_singletons(self):
        vecs = _rand(7)
        f = _static(20)
        f.construct(vecs)
        self.assertEqual(len(f.all_radiuses), 7)
        np.testing.assert_array_equal(f.all_radiuses, np.zeros(7, dtype=np.float32))
        self.assertTrue(_covering_ok(vecs, f))

    def test_every_vector_covered_in_nontrivial_case(self):
        vecs = _rand(400, seed=3)
        f = _static(25)
        f.construct(vecs)
        self.assertTrue(_covering_ok(vecs, f))

    def test_separated_clusters_get_tight_balls(self):
        vecs = _clustered(5, 40, seed=4)
        f = _static(5)
        f.construct(vecs)
        self.assertEqual(len(f.all_radiuses), 5)
        self.assertLess(float(f.all_radiuses.max()), 1.0)  # spread 0.05, separation 15

    def test_construct_returns_none(self):
        self.assertIsNone(_static(5).construct(_rand(50)))

    def test_max_spheres_zero_builds_empty_filter(self):
        f = _static(0)
        f.construct(_rand(50))
        self.assertEqual(len(f.all_radiuses), 0)
        self.assertEqual(f.query(_rand(4), distance_threshold=TAU), [False] * 4)

    def test_reconstruct_replaces_previous_state(self):
        f = _static(10)
        f.construct(_rand(100, seed=5))
        second = _clustered(2, 30, seed=6)
        f.construct(second)
        self.assertLessEqual(len(f.all_radiuses), 10)
        self.assertTrue(_covering_ok(second, f))
        self.assertTrue(all(f.query(second, distance_threshold=0.0)))


# ---------------------------------------------------------------------------
# query()
# ---------------------------------------------------------------------------

class TestBallCoverQuery(unittest.TestCase):

    def setUp(self):
        self.vecs = _rand(200, seed=7)
        self.f = _static(20)
        self.f.construct(self.vecs)

    def test_stored_vectors_always_recalled(self):
        self.assertTrue(all(self.f.query(self.vecs, distance_threshold=0.0)))

    def test_neighbors_within_threshold_recalled(self):
        self.assertTrue(all(self.f.query(_neighbors(self.vecs, TAU), distance_threshold=TAU)))

    def test_distant_vectors_rejected(self):
        far = _rand(20, seed=8) + 1000.0
        self.assertFalse(any(self.f.query(far, distance_threshold=TAU)))

    def test_query_before_construct_raises(self):
        with self.assertRaises(ValueError):
            _static(5).query(_rand(3), distance_threshold=TAU)

    def test_query_empty_batch_returns_empty_list(self):
        self.assertEqual(self.f.query(np.empty((0, DIM), dtype=np.float32), distance_threshold=TAU), [])

    def test_query_returns_list_of_bools(self):
        out = self.f.query(self.vecs[:5], distance_threshold=TAU)
        self.assertIsInstance(out, list)
        self.assertTrue(all(isinstance(x, bool) for x in out))

    def test_batch_equals_individual(self):
        q = _neighbors(self.vecs[:10], 2 * TAU, n=3)
        batch = self.f.query(q, distance_threshold=TAU)
        single = [self.f.query(q[i:i + 1], distance_threshold=TAU)[0] for i in range(len(q))]
        self.assertEqual(batch, single)

    def test_zero_radius_ball_matches_exactly_within_tau(self):
        v = _rand(1, seed=9)
        f = _static(5)
        f.construct(v)
        step = np.zeros((1, DIM), dtype=np.float32)
        step[0, 0] = 1.0
        self.assertTrue(f.query(v + 0.9 * TAU * step, distance_threshold=TAU)[0])
        self.assertFalse(f.query(v + 1.5 * TAU * step, distance_threshold=TAU)[0])


class TestBallCoverThresholdIndependence(unittest.TestCase):
    """One constructed filter answers any tau: nothing tau-specific is built."""

    def test_recall_is_monotone_in_tau(self):
        vecs = _rand(150, seed=10)
        f = _static(15)
        f.construct(vecs)
        q = _rand(300, seed=11) * 1.2
        hits = [sum(f.query(q, distance_threshold=t)) for t in (0.0, 0.5, 1.0, 2.0, 4.0)]
        self.assertEqual(hits, sorted(hits))

    def test_no_false_negatives_at_every_tau(self):
        vecs = _rand(150, seed=12)
        f = _static(15)
        f.construct(vecs)
        for tau in (0.05, 0.3, 1.0):
            self.assertTrue(all(f.query(_neighbors(vecs, tau, n=2), distance_threshold=tau)), tau)


# ---------------------------------------------------------------------------
# insert() -- streaming, through the filter (strategies are unit-tested in
# test_ball_cover_insertion.py)
# ---------------------------------------------------------------------------

class TestBallCoverInsert(unittest.TestCase):

    def test_insert_before_construct_raises(self):
        with self.assertRaises(ValueError):
            BallCoverFilter(DIM, max_spheres=10, insert_method=NearestInsert()).insert(_rand(3))

    def test_insert_without_strategy_raises(self):
        f = _static(10)
        f.construct(_rand(50))
        with self.assertRaises(ValueError):
            f.insert(_rand(3))

    def test_inserted_vectors_are_recalled(self):
        for strategy in (NearestInsert(), MinHeapInsert()):
            f = BallCoverFilter(DIM, max_spheres=20, insert_method=strategy)
            f.construct(_rand(100, seed=13))
            new = _rand(40, seed=14) * 2.0
            f.insert(new)
            name = strategy.__class__.__name__
            self.assertTrue(all(f.query(new, distance_threshold=0.0)), name)
            self.assertTrue(all(f.query(_neighbors(new, TAU, n=2), distance_threshold=TAU)), name)
            self.assertLessEqual(len(f.all_radiuses), f.max_spheres, name)

    def test_insert_empty_is_noop(self):
        f = BallCoverFilter(DIM, max_spheres=10, insert_method=NearestInsert())
        f.construct(_rand(50, seed=15))
        before = f.all_radiuses.copy()
        f.insert(np.empty((0, DIM), dtype=np.float32))
        np.testing.assert_array_equal(f.all_radiuses, before)

    def test_add_delegates_to_insert(self):
        f = BallCoverFilter(DIM, max_spheres=10, insert_method=NearestInsert())
        f.construct(_rand(50, seed=16))
        new = _rand(5, seed=17) + 3.0
        f.add(new)
        self.assertTrue(all(f.query(new, distance_threshold=0.0)))

    def test_many_batches_keep_every_vector_covered(self):
        f = BallCoverFilter(DIM, max_spheres=15, insert_method=MinHeapInsert())
        seen = _rand(60, seed=18)
        f.construct(seen)
        for s in range(19, 24):
            batch = _rand(30, seed=s)
            f.insert(batch)
            seen = np.vstack([seen, batch])
        self.assertTrue(all(f.query(seen, distance_threshold=0.0)))


# ---------------------------------------------------------------------------
# Memory accounting
# ---------------------------------------------------------------------------

class TestBallCoverMemory(unittest.TestCase):

    def test_bytesize_before_construct_is_zero(self):
        self.assertEqual(_static(10).get_bytesize(), 0)

    def test_bytesize_is_balls_times_bytes_per_ball(self):
        f = _static(10)
        f.construct(_rand(100))
        self.assertEqual(f.get_bytesize(), len(f.all_radiuses) * f.bytes_per_ball())

    def test_bytesize_within_max_memory_budget(self):
        f = BallCoverFilter(DIM, max_memory=5_000, insert_method=None)
        f.construct(_rand(500))
        self.assertLessEqual(f.get_bytesize(), 5_000)

    def test_construction_bytesize_is_zero_for_default_method(self):
        f = _static(10)
        f.construct(_rand(100))
        self.assertEqual(f.get_construction_bytesize(), 0)


# ---------------------------------------------------------------------------
# Quantized centers -- the zero-false-negative guarantee must survive encoding
# ---------------------------------------------------------------------------

class TestBallCoverQuantization(unittest.TestCase):

    def test_unquantized_by_default(self):
        self.assertEqual(_static(5).quantize, Quantize.NONE)

    def test_no_false_negatives_under_every_exact_encoding(self):
        vecs = _rand(300, seed=25)
        for q in (Quantize.FP16, Quantize.SQ8, Quantize.SQ4, Quantize.PQ4_R2):
            f = BallCoverFilter(DIM, max_spheres=40, insert_method=None, quantize=q)
            f.construct(vecs)
            self.assertTrue(all(f.query(vecs, distance_threshold=0.0)), q)
            self.assertTrue(all(f.query(_neighbors(vecs, TAU, n=2), distance_threshold=TAU)), q)

    def test_bulk_decode_matches_reconstruct(self):
        # all_centers decodes the stored codes in one pass; it must sit where
        # faiss's own per-id reconstruct puts each ball. dim=32 takes the
        # Hadamard rotation and dim=384 the dense one.
        for dim in (DIM, 384):
            vecs = _rand(600, dim=dim, seed=28)
            for q in (Quantize.FP32, Quantize.SQ4, Quantize.TQ4, Quantize.PQ4_R2):
                f = BallCoverFilter(dim, max_spheres=40, insert_method=None, quantize=q)
                f.construct(vecs)
                ids = f.all_spheres.astype(np.int64)
                np.testing.assert_allclose(
                    f.all_centers, f.all_centers_index.reconstruct_batch(ids), atol=1e-5,
                    err_msg=f"{q} at dim={dim}")

    def test_encoding_shrinks_bytes_per_ball(self):
        full = _static(5).bytes_per_ball()
        self.assertLess(_static(5, quantize=Quantize.SQ8).bytes_per_ball(), full)


# ---------------------------------------------------------------------------
# Blocked scan -- query()/surface_distance() against faiss's own search
# ---------------------------------------------------------------------------

def _unit(n: int, dim: int, seed: int) -> np.ndarray:
    v = _rand(n, dim=dim, seed=seed)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


class TestBallCoverScan(unittest.TestCase):

    TAUS = (0.0, 0.05, 0.1, 0.3, 0.6)

    def _check_against_search(self, f: BallCoverFilter, queries: np.ndarray) -> None:
        # Reference: float64 distance to every decoded center.
        C = f.all_centers.astype(np.float64)
        exact = (np.sqrt(((queries[:, None, :].astype(np.float64) - C[None]) ** 2).sum(-1))
                 - f.all_radiuses[None, :]).min(axis=1)
        surface = f.surface_distance(queries)
        np.testing.assert_allclose(surface, exact, atol=1e-5)
        # faiss's own search(k=all), which query() used to run, agrees to
        # within the coverage slack its float32 cancellation needed.
        np.testing.assert_allclose(f._estimator_surface_distance(queries), exact,
                                   atol=f._COVERAGE_EPSILON)
        for tau in self.TAUS:
            boundary = tau + f._COVERAGE_EPSILON
            got = np.asarray(f.query(queries, distance_threshold=tau))
            # Only queries within rounding of the boundary may go either way.
            settled = np.abs(exact - boundary) > 1e-5
            np.testing.assert_array_equal(got[settled], (exact <= boundary)[settled],
                                          err_msg=f"{f.quantize} tau={tau}")

    def test_matches_faiss_search_under_every_encoding(self):
        for dim in (DIM, 384):
            vecs = _unit(1200, dim, seed=29)
            queries = np.vstack([vecs[:200], _neighbors(vecs[:100], 0.2, n=2), _unit(200, dim, seed=30)])
            for q in Quantize:
                if q in (Quantize.RABITQ, Quantize.PQ8_R1, Quantize.PQ8_R2):
                    continue   # RaBitQ keeps faiss search; PQ8 needs >= 256 balls
                f = BallCoverFilter(dim, max_spheres=120, insert_method=None, quantize=q)
                f.construct(vecs)
                with self.subTest(dim=dim, quantize=q):
                    self._check_against_search(f, queries)

    def test_many_blocks_and_chunks_match_one(self):
        vecs = _unit(800, 64, seed=31)
        queries = np.vstack([vecs[:300], _unit(300, 64, seed=32)])
        f = BallCoverFilter(64, max_spheres=150, insert_method=None, quantize=Quantize.TQ4)
        f.construct(vecs)
        whole_q = [f.query(queries, tau) for tau in self.TAUS]
        whole_s = f.surface_distance(queries)
        f._SCAN_BLOCK, f._SCAN_QUERY_CHUNK = 7, 5
        for tau, expected in zip(self.TAUS, whole_q):
            self.assertEqual(f.query(queries, tau), expected, tau)
        np.testing.assert_allclose(f.surface_distance(queries), whole_s, atol=1e-5)

    def test_small_batches_match_scan(self):
        # Batches below _SCAN_MIN_BATCH take range_search instead of _scan.
        for q in (Quantize.FP32, Quantize.TQ4, Quantize.PQ4_R2):
            vecs = _unit(800, 64, seed=34)
            queries = np.vstack([vecs[:60], _neighbors(vecs[:30], 0.15, n=1), _unit(60, 64, seed=35)])
            f = BallCoverFilter(64, max_spheres=150, insert_method=None, quantize=q)
            f.construct(vecs)
            self.assertGreater(f._SCAN_MIN_BATCH, 1)
            for tau in self.TAUS:
                scanned = f._scan(queries, tau).tolist()
                one_by_one = [f.query(queries[i:i + 1], tau)[0] for i in range(len(queries))]
                self.assertEqual(one_by_one, scanned, (q, tau))

    def test_exact_duplicates_accepted_at_tau_zero(self):
        # Zero-radius balls queried by the very vectors they hold: the
        # ||q||^2 + ||c||^2 - 2<q,c> form cancels here, and only the float64
        # re-measure keeps these from depending on float32 rounding.
        vecs = _unit(500, 384, seed=33) * 3.0   # larger norms, larger cancellation
        f = BallCoverFilter(384, max_spheres=500, insert_method=None)
        f.construct(vecs)
        self.assertAlmostEqual(float(f.all_radiuses.max()), 0.0, places=6)
        self.assertTrue(all(f.query(vecs, distance_threshold=0.0)))
        self.assertLess(float(np.abs(f.surface_distance(vecs)).max()), 1e-5)

    def test_no_balls_rejects_everything(self):
        f = _static(5)
        f.construct(np.empty((0, DIM), dtype=np.float32))
        self.assertEqual(f.query(_rand(3), distance_threshold=1.0), [False] * 3)


class TestBallCoverKernels(unittest.TestCase):
    """scan_kernels, which answer small FP32 / 4-bit batches. Called through
    _kernel_accepts directly so every batch size and ball count reaches them,
    whatever query()'s cutoffs are."""

    KERNEL_ENCODINGS = (Quantize.FP32, Quantize.TQ4, Quantize.SQ4)
    TAUS = (-0.01, 0.0, 0.05, 0.1, 0.3, 0.6)

    def _check(self, f: BallCoverFilter, queries: np.ndarray) -> None:
        # Reference: float64 distance to every decoded center. Only queries
        # within rounding of some ball's boundary may go either way.
        C = f.all_centers.astype(np.float64)
        d = np.sqrt(((queries[:, None, :].astype(np.float64) - C[None]) ** 2).sum(-1))
        for tau in self.TAUS:
            reach = f.all_radiuses.astype(np.float64) + tau + f._COVERAGE_EPSILON
            expected = ((reach >= 0) & (d <= reach)).any(axis=1)
            settled = np.where(reach >= 0, np.abs(d - reach), np.inf).min(axis=1) > 1e-5
            got = f._kernel_accepts(queries, tau)
            np.testing.assert_array_equal(got[settled], expected[settled],
                                          err_msg=f"{f.quantize} dim={f.dim} tau={tau}")
            # And query-for-query what _scan decides, boundary pairs included.
            np.testing.assert_array_equal(got, f._scan(queries, tau), err_msg=f"{f.quantize} tau={tau}")

    def test_matches_exact_reference(self):
        # dim 21 and 20 exercise padding to whole 64-bit code words and the
        # FP32 kernel's tail; scale 3 un-normalizes the vectors.
        for dim, scale in ((32, 1.0), (384, 1.0), (64, 3.0), (21, 1.0), (20, 2.0)):
            vecs = _unit(900, dim, seed=41) * scale
            queries = np.vstack([vecs[:150], _neighbors(vecs[:100], 0.2 * scale, n=2),
                                 _unit(150, dim, seed=42) * scale])
            for q in self.KERNEL_ENCODINGS:
                f = BallCoverFilter(dim, max_spheres=120, insert_method=None, quantize=q)
                f.construct(vecs)
                with self.subTest(dim=dim, scale=scale, quantize=q):
                    self._check(f, queries)

    def test_nibble_table_is_sa_decode(self):
        rng = np.random.default_rng(43)
        for q in (Quantize.TQ4, Quantize.SQ4):
            for dim in (32, 21, 384):
                f = BallCoverFilter(dim, max_spheres=60, insert_method=None, quantize=q)
                f.construct(_unit(400, dim, seed=44))
                leaf = f._leaf_index()
                nb = int(leaf.code_size)
                codes = rng.integers(0, 256, (50, nb), dtype=np.uint8)
                nibbles = np.empty((50, 2 * nb), dtype=np.int64)
                nibbles[:, 0::2], nibbles[:, 1::2] = codes & 15, codes >> 4
                table = f._nibble_table(2 * nb)
                decoded = table[np.arange(2 * nb)[None, :], nibbles]
                with self.subTest(quantize=q, dim=dim):
                    np.testing.assert_array_equal(decoded[:, :dim].astype(np.float32), leaf.sa_decode(codes))
                    self.assertFalse(decoded[:, dim:].any())

    def test_exact_duplicates_accepted_at_tau_zero(self):
        # Zero-radius FP32 balls queried by the very vectors they hold, at a
        # larger norm; and 4-bit balls queried by their own decoded centers.
        vecs = _unit(500, 384, seed=45) * 3.0
        f = BallCoverFilter(384, max_spheres=500, insert_method=None)
        f.construct(vecs)
        self.assertAlmostEqual(float(f.all_radiuses.max()), 0.0, places=6)
        self.assertTrue(f._kernel_accepts(vecs, 0.0).all())
        for q in (Quantize.TQ4, Quantize.SQ4):
            f = BallCoverFilter(384, max_spheres=500, insert_method=None, quantize=q)
            f.construct(vecs)
            self.assertTrue(f._kernel_accepts(np.ascontiguousarray(f.all_centers), 0.0).all(), q)

    def test_infinite_radius_accepts_everything(self):
        # fp16 radii past 65504 are stored as +inf (see _fp16_ceil).
        for q in self.KERNEL_ENCODINGS:
            f = BallCoverFilter(DIM, max_spheres=20, insert_method=None, quantize=q)
            f.construct(_unit(200, DIM, seed=46))
            f._all_radiuses[f.all_spheres[3]] = np.inf
            self.assertTrue(f._kernel_accepts(_rand(30, seed=47) * 100.0, 0.0).all(), q)

    def test_storage_is_largest_first_and_otherwise_invisible(self):
        vecs = _unit(1500, 64, seed=50)
        queries = np.vstack([vecs[:200], _unit(200, 64, seed=51)])
        for q in (Quantize.FP32, Quantize.TQ4, Quantize.PQ4_R2, Quantize.RABITQ):
            built = []
            for reorder in (True, False):
                f = BallCoverFilter(64, max_spheres=200, insert_method=None, quantize=q)
                if not reorder:
                    f._store_largest_first = lambda: None
                f.construct(vecs)
                built.append(f)
            f, plain = built
            ids, _ = f._stored_codes()
            with self.subTest(quantize=q):
                self.assertTrue((np.diff(f._all_radiuses[ids]) <= 0).all())
                self.assertFalse(np.array_equal(ids, plain._stored_codes()[0]))
                # Same balls under the same ids, wherever they are stored.
                np.testing.assert_array_equal(f.all_spheres, plain.all_spheres)
                np.testing.assert_array_equal(f.all_radiuses, plain.all_radiuses)
                np.testing.assert_allclose(f.all_centers, plain.all_centers, atol=1e-6)
                np.testing.assert_allclose(f.all_centers_index.reconstruct_batch(f.all_spheres.astype(np.int64)),
                                           f.all_centers, atol=1e-5)
                for tau in (0.0, 0.1):
                    self.assertEqual(f.query(queries, tau), plain.query(queries, tau))
                np.testing.assert_allclose(f.surface_distance(queries), plain.surface_distance(queries), atol=1e-5)

    def test_streaming_filters_keep_construction_order(self):
        # Insert strategies decide from all_centers, whose un-rotation
        # rounds by storage position, so reordering would change their covers.
        vecs = _unit(1500, 64, seed=53)
        for insert in (NearestInsert(), MinHeapInsert()):
            f = BallCoverFilter(64, max_spheres=200, insert_method=insert, quantize=Quantize.TQ4)
            f.construct(vecs)
            np.testing.assert_array_equal(f._stored_codes()[0], f.all_spheres)

    def test_query_dispatch_matches_scan(self):
        # query() itself, on both sides of every kernel cutoff; the FP32
        # ball floor is lifted so a small cover reaches the kernel too.
        vecs = _unit(800, 32, seed=48)
        queries = np.vstack([vecs[:100], _unit(100, 32, seed=49)])
        for q in self.KERNEL_ENCODINGS:
            for floor in (0, BallCoverFilter._FLAT_KERNEL_MIN_BALLS):
                f = BallCoverFilter(32, max_spheres=150, insert_method=None, quantize=q)
                f._FLAT_KERNEL_MIN_BALLS = floor
                f.construct(vecs)
                for bs in (1, 3, 7, 8, 63, 64, 200):
                    got = np.concatenate([f.query(queries[s:s + bs], 0.05) for s in range(0, len(queries), bs)])
                    with self.subTest(quantize=q, floor=floor, batch=bs):
                        np.testing.assert_array_equal(got, f._scan(queries, 0.05))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestBallCoverEdgeCases(unittest.TestCase):

    def test_all_identical_vectors(self):
        vecs = np.repeat(_rand(1, seed=26), 50, axis=0)
        f = _static(5)
        f.construct(vecs)
        self.assertTrue(all(f.query(vecs, distance_threshold=0.0)))
        self.assertAlmostEqual(float(f.all_radiuses.max()), 0.0, places=3)

    def test_single_vector(self):
        v = _rand(1, seed=27)
        f = _static(5)
        f.construct(v)
        self.assertEqual(len(f.all_radiuses), 1)
        self.assertTrue(f.query(v, distance_threshold=0.0)[0])


if __name__ == "__main__":
    unittest.main()
