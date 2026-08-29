import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'src'))
import unittest
import copy

import numpy as np

from src.filters.dsbf_filter import DSBFFilter

DIM = 32
R = 0.3


def _rand(n: int, dim: int = DIM, seed: int = 0) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal((n, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestDSBFInit(unittest.TestCase):

    def test_explicit_num_tables_used(self):
        f = DSBFFilter(dim=DIM, distance_threshold=R, num_tables=6, hashes_per_table=3, max_memory=100_000)
        self.assertEqual(f.num_tables, 6)

    def test_max_memory_derives_bits_per_table(self):
        dim = 16
        budget = 8_000  # total bits = 64000
        f = DSBFFilter(dim=dim, distance_threshold=R, max_memory=budget, num_tables=4, hashes_per_table=2)
        self.assertEqual(f.num_tables, 4)
        self.assertEqual(f.bits_per_table, (budget * 8) // 4)

    def test_default_vote_threshold_is_majority(self):
        f = DSBFFilter(dim=DIM, distance_threshold=R, num_tables=5, hashes_per_table=2, max_memory=100_000)
        self.assertEqual(f.vote_threshold, 3)  # (5+1)//2

    def test_explicit_vote_threshold_used(self):
        f = DSBFFilter(dim=DIM, distance_threshold=R, num_tables=5, hashes_per_table=2,
                        vote_threshold=1, max_memory=100_000)
        self.assertEqual(f.vote_threshold, 1)

    def test_vote_threshold_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            DSBFFilter(dim=DIM, distance_threshold=R, num_tables=4, hashes_per_table=2,
                       vote_threshold=0, max_memory=100_000)
        with self.assertRaises(ValueError):
            DSBFFilter(dim=DIM, distance_threshold=R, num_tables=4, hashes_per_table=2,
                       vote_threshold=5, max_memory=100_000)

    def test_deterministic_given_seed(self):
        f1 = DSBFFilter(dim=DIM, distance_threshold=R, num_tables=4, hashes_per_table=3,
                         max_memory=100_000, seed=123)
        f2 = DSBFFilter(dim=DIM, distance_threshold=R, num_tables=4, hashes_per_table=3,
                         max_memory=100_000, seed=123)
        np.testing.assert_array_equal(f1._lsh.hyperplanes, f2._lsh.hyperplanes)


# ---------------------------------------------------------------------------
# Exact-recall invariant (holds for any valid vote_threshold)
# ---------------------------------------------------------------------------

class TestDSBFExactRecall(unittest.TestCase):

    def test_self_recall_at_every_vote_threshold(self):
        vecs = _rand(100, seed=5)
        for t in (1, 2, 3, 4):
            f = DSBFFilter(dim=DIM, distance_threshold=R, num_tables=4, hashes_per_table=3,
                            vote_threshold=t, max_memory=300_000, seed=9)
            f.construct(vecs)
            predicted = f.query(vecs)
            self.assertTrue(all(predicted), f"false negative on exact re-query at vote_threshold={t}")

    def test_insert_then_recall(self):
        f = DSBFFilter(dim=DIM, distance_threshold=R, num_tables=4, hashes_per_table=3,
                        vote_threshold=2, max_memory=300_000, seed=9)
        f.construct(_rand(20, seed=1))
        probe = _rand(1, seed=77)
        f.insert(probe)
        self.assertTrue(f.query(probe)[0])


# ---------------------------------------------------------------------------
# AND/OR/threshold semantics -- the defining B1 property.
# ---------------------------------------------------------------------------

class TestDSBFVoteSemantics(unittest.TestCase):
    """
    B1's whole point is the vote_threshold dial: t=1 behaves like an OR
    across tables (most permissive), t=num_tables behaves like an AND
    (strictest). This constructs two filters sharing the same underlying
    hash codes and bit-array state, differing only in vote_threshold, and
    checks a probe with exactly 1-of-num_tables bits set is accepted at
    t=1 but rejected at t=num_tables -- an accidental OR-only or AND-only
    implementation would fail one side of this.
    """

    def test_single_table_hit_passes_t1_fails_t_all(self):
        num_tables = 4
        f1 = DSBFFilter(dim=DIM, distance_threshold=R, num_tables=num_tables, hashes_per_table=2,
                         vote_threshold=1, max_memory=200_000, seed=42)
        f1._reset_storage()  # empty bit arrays, no data inserted

        probe = _rand(1, seed=999)
        code_row = f1._codes(probe)[0]  # (num_tables,)

        # Set only table 0's bit for this probe -- simulate exactly 1-of-num_tables hits.
        bit_idx = None
        from filters._hashing import stable_hash_to_int
        bit_idx = stable_hash_to_int(code_row[0]) % f1.bits_per_table
        f1._byte_arrays[0][bit_idx // 8] |= np.uint8(1 << (bit_idx % 8))

        self.assertTrue(f1.query(probe)[0], "t=1 (OR-like) must accept a 1-of-num_tables hit")

        f_all = DSBFFilter(dim=DIM, distance_threshold=R, num_tables=num_tables, hashes_per_table=2,
                            vote_threshold=num_tables, max_memory=200_000, seed=42)
        f_all._reset_storage()
        f_all._lsh = f1._lsh  # identical hash codes
        f_all._byte_arrays = copy.deepcopy(f1._byte_arrays)  # identical bit-array state

        self.assertFalse(f_all.query(probe)[0], "t=num_tables (AND-like) must reject a 1-of-num_tables hit")


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

class TestDSBFMemory(unittest.TestCase):

    def test_bytesize_matches_configured_tables(self):
        f = DSBFFilter(dim=DIM, distance_threshold=R, num_tables=4, hashes_per_table=2, max_memory=80_000)
        expected = f.num_tables * f._bytes_per_table
        self.assertEqual(f.get_bytesize(), expected)

    def test_respects_memory_budget(self):
        budget = 40_000
        f = DSBFFilter(dim=DIM, distance_threshold=R, num_tables=4, hashes_per_table=2, max_memory=budget)
        f.construct(_rand(200))
        self.assertLessEqual(f.get_bytesize(), budget + f.num_tables)  # +rounding to whole bytes/table


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

class TestDSBFInterface(unittest.TestCase):

    def test_query_before_construct_raises(self):
        f = DSBFFilter(dim=DIM, distance_threshold=R, num_tables=4, hashes_per_table=2, max_memory=100_000)
        with self.assertRaises(ValueError):
            f.query(_rand(1))

    def test_query_with_mismatched_tau_raises(self):
        f = DSBFFilter(dim=DIM, distance_threshold=0.3, num_tables=4, hashes_per_table=2, max_memory=100_000)
        f.construct(_rand(10))
        with self.assertRaises(ValueError):
            f.query(_rand(1), tau=0.9)

if __name__ == "__main__":
    unittest.main()
