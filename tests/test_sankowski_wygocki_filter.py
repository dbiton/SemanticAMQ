import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'src'))
import unittest

import numpy as np

from src.filters.sankowski_wygocki_filter import SankowskiWygockiFilter
from filters.filter import InsufficientMemoryError

DIM = 16
R = 0.2


def _rand(n: int, dim: int = DIM, seed: int = 0) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal((n, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _at_distance(base: np.ndarray, directions: np.ndarray, r: float) -> np.ndarray:
    d = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    return (base + r * d).astype(np.float32)


class TestSWNoFalseNegatives(unittest.TestCase):

    def _assert_all_found(self, f, queries):
        self.assertTrue(all(f.query(queries)), "false negative within tau")

    def test_random_directions_at_tau(self):
        base = _rand(100, seed=1)
        dirs = np.random.default_rng(2).standard_normal(base.shape)
        for L in (1, 2):
            f = SankowskiWygockiFilter(DIM, R, block_dim=4, hashes_per_block=3, num_families=L, seed=3)
            f.construct(base)
            self._assert_all_found(f, _at_distance(base, dirs, R))

    def test_adversarial_direction_inside_one_block(self):
        """z confined to one block of family 0 and aligned with one of that
        table's hash directions: that block is stretched by sqrt(d/k), so only
        the pigeonhole over the OTHER blocks can save it."""
        base = _rand(30, seed=4)
        f = SankowskiWygockiFilter(DIM, R, block_dim=4, hashes_per_block=3, seed=5)
        f.construct(base)
        k = f.block_dim
        unscaled = f._bases[0] / np.sqrt(f.padded_dim / k)  # orthonormal rows
        for t, (i,) in enumerate(f._combinations):
            for u in f._hash_dirs[t, 0]:
                z = unscaled[i * k:(i + 1) * k].T @ u  # unit vector living in block i
                self._assert_all_found(f, _at_distance(base, np.broadcast_to(z, base.shape), R))

    def test_padding_when_dim_not_divisible(self):
        dim = 10
        base = _rand(60, dim=dim, seed=6)
        dirs = np.random.default_rng(7).standard_normal(base.shape)
        f = SankowskiWygockiFilter(dim, R, block_dim=4, hashes_per_block=3, seed=8)
        self.assertEqual((f.padded_dim, f.num_blocks), (12, 3))
        f.construct(base)
        self._assert_all_found(f, _at_distance(base, dirs, R))

    def test_probe_mode_and_bloom_path(self):
        base = _rand(60, seed=9)
        queries = _at_distance(base, np.random.default_rng(10).standard_normal(base.shape), R)
        for kw in (dict(replicate="probe"), dict(max_memory=200_000),
                   dict(replicate="probe", max_memory=20_000)):
            f = SankowskiWygockiFilter(DIM, R, block_dim=4, hashes_per_block=3, seed=11, **kw)
            f.construct(base)
            self._assert_all_found(f, queries)

    def test_insert_then_recall(self):
        f = SankowskiWygockiFilter(DIM, R, block_dim=4, hashes_per_block=3, seed=12)
        f.construct(_rand(20, seed=1))
        probe = _rand(1, seed=77)
        f.insert(probe)
        self._assert_all_found(f, _at_distance(probe, _rand(1, seed=78), R))


class TestSWBehaviour(unittest.TestCase):

    def test_table_count(self):
        for L in (1, 2, 3):
            f = SankowskiWygockiFilter(DIM, R, block_dim=8, hashes_per_block=2, num_families=L)
            self.assertEqual(f.num_tables, 2 ** L)
            self.assertEqual(f.signature_length, 2 * L)

    def test_blocks_split_norm_exactly(self):
        """Lemma 3's premise: the mean of ||A(i) z||^2 over blocks is ||z||^2."""
        f = SankowskiWygockiFilter(DIM, R, block_dim=4, hashes_per_block=2)
        z = np.random.default_rng(13).standard_normal(DIM)
        blocks = (f._bases[0] @ z).reshape(f.num_blocks, f.block_dim)
        np.testing.assert_allclose(np.mean(np.sum(blocks ** 2, axis=1)), z @ z, rtol=1e-10)

    def test_store_and_probe_agree(self):
        base, queries = _rand(50, seed=14), _rand(150, seed=15)
        kw = dict(block_dim=4, hashes_per_block=3, seed=16)
        a = SankowskiWygockiFilter(DIM, R, replicate="store", **kw)
        b = SankowskiWygockiFilter(DIM, R, replicate="probe", **kw)
        a.construct(base)
        b.construct(base)
        self.assertEqual(a.query(queries), b.query(queries))

    def test_rejects_far_points_in_low_dimension(self):
        dim = 4
        base, queries = _rand(50, dim=dim, seed=17), _rand(300, dim=dim, seed=18)
        f = SankowskiWygockiFilter(dim, 0.02, block_dim=2, hashes_per_block=4, seed=19)
        f.construct(base)
        self.assertLess(np.mean(f.query(queries)), 0.5)

    def test_deterministic_given_seed(self):
        a = SankowskiWygockiFilter(DIM, R, block_dim=4, seed=123)
        b = SankowskiWygockiFilter(DIM, R, block_dim=4, seed=123)
        np.testing.assert_array_equal(a._bases, b._bases)
        np.testing.assert_array_equal(a._hash_dirs, b._hash_dirs)


class TestSWInterface(unittest.TestCase):

    def test_query_before_construct_raises(self):
        with self.assertRaises(ValueError):
            SankowskiWygockiFilter(DIM, R, block_dim=4).query(_rand(1))

    def test_query_with_mismatched_tau_raises(self):
        f = SankowskiWygockiFilter(DIM, R, block_dim=4, hashes_per_block=2)
        f.construct(_rand(10))
        with self.assertRaises(ValueError):
            f.query(_rand(1), tau=0.9)

    def test_too_small_budget_raises(self):
        with self.assertRaises(InsufficientMemoryError):
            SankowskiWygockiFilter(DIM, R, block_dim=4, hashes_per_block=4,
                                   max_memory=1_000).construct(_rand(100))

    def test_oversized_expansion_raises_instead_of_allocating(self):
        f = SankowskiWygockiFilter(DIM, R, block_dim=4, hashes_per_block=12, replicate="probe")
        f.construct(_rand(5))
        with self.assertRaises(ValueError):
            f.query(_rand(1))

    def test_invalid_arguments_raise(self):
        for kw in (dict(block_dim=0), dict(hashes_per_block=0), dict(num_families=0)):
            with self.assertRaises(ValueError):
                SankowskiWygockiFilter(DIM, R, **kw)


if __name__ == "__main__":
    unittest.main()
