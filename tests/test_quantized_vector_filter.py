import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'src'))
import unittest

import numpy as np

from src.filters.quantized_vector_filter import (
    QuantizedVectorFilter, ErrorBudgetMode, InsufficientMemoryError, QuantizationLevel,
    _footprint_bytes, _cluster_count, _id_bytes,
)

DIM = 32
R = 0.3


def _rand(n: int, dim: int = DIM, seed: int = 0) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal((n, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _neighbors(v: np.ndarray, r: float, n: int = 5, seed: int = 1) -> np.ndarray:
    """Return n vectors within Euclidean distance r of each row of v."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((len(v), n, v.shape[1])).astype(np.float32)
    norms = np.linalg.norm(noise, axis=2, keepdims=True)
    scales = rng.uniform(0, r * 0.95, (len(v), n, 1)).astype(np.float32)
    unit_noise = noise / (norms + 1e-9) * scales
    return (v[:, None, :] + unit_noise).reshape(-1, v.shape[1])


# ---------------------------------------------------------------------------
# Zero-false-negative guarantee -- the mandatory property for this baseline
# (see module docstring), checked in every error-budget storage mode.
# ---------------------------------------------------------------------------

class TestQVFZeroFalseNegatives(unittest.TestCase):

    def _assert_no_fn(self, mode: ErrorBudgetMode, budget: int, seed_offset: int) -> None:
        for tau in (0.1, 0.3, 0.6):
            vecs = _rand(120, seed=seed_offset + hash(tau) % 1000)
            f = QuantizedVectorFilter(dim=DIM, distance_threshold=tau, max_memory=budget,
                                       error_budget_mode=mode, seed=3)
            f.construct(vecs)

            exact_pred = f.query(vecs)
            self.assertTrue(all(exact_pred), f"[{mode}] false negative on exact self-recall at tau={tau}")

            probes = _neighbors(vecs, tau, n=3)
            near_pred = f.query(probes)
            fn = sum(1 for p in near_pred if not p)
            self.assertEqual(fn, 0, f"[{mode}] {fn} false negatives among near-tau probes at tau={tau}")

    def test_global_mode(self):
        self._assert_no_fn(ErrorBudgetMode.GLOBAL, budget=200_000, seed_offset=1)

    def test_per_cluster_mode(self):
        self._assert_no_fn(ErrorBudgetMode.PER_CLUSTER, budget=200_000, seed_offset=2)

    def test_per_vector_mode(self):
        self._assert_no_fn(ErrorBudgetMode.PER_VECTOR, budget=200_000, seed_offset=3)

    def test_zero_false_negatives_at_coarsest_level(self):
        """Forces the RaBitQ rung specifically (a budget too tight for sq4
        but just big enough for rabitq) -- the level with the largest eps_i,
        and the only one that takes the non-SIMD fallback query path."""
        n = 200
        vecs = _rand(n, seed=11)
        budget = _footprint_bytes(QuantizationLevel.RABITQ, n, DIM, ErrorBudgetMode.PER_CLUSTER, 8) + 8
        f = QuantizedVectorFilter(dim=DIM, distance_threshold=0.3, max_memory=budget,
                                   error_budget_mode=ErrorBudgetMode.PER_CLUSTER, seed=3)
        f.construct(vecs)
        self.assertEqual(f.level, QuantizationLevel.RABITQ)

        self.assertTrue(all(f.query(vecs)), "false negative on exact self-recall at the RaBitQ level")
        probes = _neighbors(vecs, 0.3, n=3)
        fn = sum(1 for p in f.query(probes) if not p)
        self.assertEqual(fn, 0, f"{fn} false negatives among near-tau probes at the RaBitQ level")


# ---------------------------------------------------------------------------
# Budget infeasibility -- fail loudly, never silently drop/subsample.
# ---------------------------------------------------------------------------

class TestQVFBudgetInfeasibility(unittest.TestCase):

    def test_too_small_budget_raises(self):
        with self.assertRaises(InsufficientMemoryError):
            QuantizedVectorFilter(dim=DIM, distance_threshold=R, max_memory=1).construct(_rand(500))

    def test_exception_names_budget_n_dim(self):
        n = 500
        with self.assertRaises(InsufficientMemoryError) as ctx:
            QuantizedVectorFilter(dim=DIM, distance_threshold=R, max_memory=1).construct(_rand(n))
        msg = str(ctx.exception)
        self.assertIn("max_memory=1", msg)
        self.assertIn(f"n={n}", msg)
        self.assertIn(f"dim={DIM}", msg)

    def test_no_memory_budget_never_raises(self):
        """max_memory=None (no budget constraint) always picks the finest level."""
        f = QuantizedVectorFilter(dim=DIM, distance_threshold=R, max_memory=None)
        f.construct(_rand(50))
        self.assertEqual(f.level, QuantizationLevel.FP16)


# ---------------------------------------------------------------------------
# Memory accounting -- the ladder-selection formula and get_bytesize() must
# use the same code path (see _footprint_bytes' docstring); verified here
# against an independently hand-computed figure per error-budget mode.
# ---------------------------------------------------------------------------

class TestQVFMemoryAccounting(unittest.TestCase):

    def test_bytesize_zero_before_construct(self):
        f = QuantizedVectorFilter(dim=DIM, distance_threshold=R, max_memory=200_000)
        self.assertEqual(f.get_bytesize(), 0)

    def test_bytesize_matches_hand_computed_global(self):
        n = 100
        # sq8 code (dim bytes/vector) + sq8's trained min/diff table
        # (2 floats/dim) + one global float32 error budget. Budget set to
        # exactly this so sq8 is selected (fp16 needs n*dim*2, well over).
        expected = n * DIM + 2 * DIM * 4 + 4
        f = QuantizedVectorFilter(dim=DIM, distance_threshold=R, max_memory=expected,
                                   error_budget_mode=ErrorBudgetMode.GLOBAL)
        f.construct(_rand(n))
        self.assertEqual(f.level, QuantizationLevel.SQ8)
        self.assertEqual(f.get_bytesize(), expected)

    def test_bytesize_matches_hand_computed_per_cluster(self):
        n = 100
        k = _cluster_count(n)
        expected = n * DIM + 2 * DIM * 4 + k * 4 + n * _id_bytes(k)
        f = QuantizedVectorFilter(dim=DIM, distance_threshold=R, max_memory=expected,
                                   error_budget_mode=ErrorBudgetMode.PER_CLUSTER)
        f.construct(_rand(n))
        self.assertEqual(f.level, QuantizationLevel.SQ8)
        self.assertEqual(f.get_bytesize(), expected)

    def test_bytesize_matches_hand_computed_per_vector(self):
        n = 100
        eps_bits = 8
        expected = n * DIM + 2 * DIM * 4 + n  # eps_bits=8 -> exactly 1 byte/vector
        f = QuantizedVectorFilter(dim=DIM, distance_threshold=R, max_memory=expected,
                                   error_budget_mode=ErrorBudgetMode.PER_VECTOR, per_vector_eps_bits=eps_bits)
        f.construct(_rand(n))
        self.assertEqual(f.level, QuantizationLevel.SQ8)
        self.assertEqual(f.get_bytesize(), expected)

    def test_bytesize_grows_with_more_vectors(self):
        f = QuantizedVectorFilter(dim=DIM, distance_threshold=R, max_memory=5_000_000)
        f.construct(_rand(50))
        small = f.get_bytesize()
        g = QuantizedVectorFilter(dim=DIM, distance_threshold=R, max_memory=5_000_000)
        g.construct(_rand(500, seed=1))
        self.assertGreater(g.get_bytesize(), small)

    def test_larger_budget_yields_finer_or_equal_level(self):
        ladder = list(QuantizationLevel)
        small = QuantizedVectorFilter(dim=DIM, distance_threshold=R, max_memory=4_000)
        small.construct(_rand(100, seed=2))
        large = QuantizedVectorFilter(dim=DIM, distance_threshold=R, max_memory=5_000_000)
        large.construct(_rand(100, seed=2))
        # Finer levels sort earlier in the ladder, so a larger budget must
        # yield an index <= the smaller budget's.
        self.assertLessEqual(ladder.index(large.level), ladder.index(small.level))


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

class TestQVFInterface(unittest.TestCase):

    def test_query_before_construct_raises(self):
        f = QuantizedVectorFilter(dim=DIM, distance_threshold=R, max_memory=200_000)
        with self.assertRaises(ValueError):
            f.query(_rand(1))

    def test_empty_construct_is_safe(self):
        f = QuantizedVectorFilter(dim=DIM, distance_threshold=R, max_memory=200_000)
        f.construct(np.zeros((0, DIM), dtype=np.float32))
        self.assertEqual(f.get_bytesize(), 0)
        self.assertEqual(f.query(_rand(3)), [False, False, False])

    def test_insert_not_supported(self):
        f = QuantizedVectorFilter(dim=DIM, distance_threshold=R, max_memory=200_000)
        f.construct(_rand(10))
        with self.assertRaises(NotImplementedError):
            f.insert(_rand(1))

    def test_invalid_per_vector_eps_bits_raises(self):
        with self.assertRaises(ValueError):
            QuantizedVectorFilter(dim=DIM, distance_threshold=R, max_memory=200_000, per_vector_eps_bits=3)


if __name__ == "__main__":
    unittest.main()
