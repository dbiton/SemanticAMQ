import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'src'))
import unittest

import numpy as np

from src.filters.lsbf_filter import LSBFFilter
from src.filters._hashing import stable_hash_to_int

DIM = 32
R = 0.3


def _rand(n: int, dim: int = DIM, seed: int = 0) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal((n, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestLSBFInit(unittest.TestCase):

    def test_verify_true_reserves_bit_budget(self):
        f_verify = LSBFFilter(dim=DIM, distance_threshold=R, max_memory=100_000,
                               num_hashes=4, verify=True, verify_budget_fraction=0.2)
        f_plain = LSBFFilter(dim=DIM, distance_threshold=R, max_memory=100_000,
                              num_hashes=4, verify=False)
        self.assertLess(f_verify.primary_bits, f_plain.primary_bits)

    def test_verify_false_has_no_verify_buckets(self):
        f = LSBFFilter(dim=DIM, distance_threshold=R, max_memory=100_000, verify=False)
        self.assertEqual(f.num_verify_buckets, 0)

    def test_deterministic_given_seed(self):
        f1 = LSBFFilter(dim=DIM, distance_threshold=R, max_memory=100_000, seed=123)
        f2 = LSBFFilter(dim=DIM, distance_threshold=R, max_memory=100_000, seed=123)
        np.testing.assert_array_equal(f1._lsh.hyperplanes, f2._lsh.hyperplanes)


# ---------------------------------------------------------------------------
# Exact-recall invariant
# ---------------------------------------------------------------------------

class TestLSBFExactRecall(unittest.TestCase):

    def test_self_recall_verify_true_and_false(self):
        vecs = _rand(100, seed=5)
        for verify in (True, False):
            f = LSBFFilter(dim=DIM, distance_threshold=R, max_memory=300_000,
                            num_hashes=4, sub_bits_per_hash=6, verify=verify, seed=9)
            f.construct(vecs)
            predicted = f.query(vecs)
            self.assertTrue(all(predicted), f"false negative on exact re-query (verify={verify})")

    def test_insert_then_recall(self):
        f = LSBFFilter(dim=DIM, distance_threshold=R, max_memory=300_000,
                        num_hashes=4, sub_bits_per_hash=6, verify=True, seed=9)
        f.construct(_rand(20, seed=1))
        probe = _rand(1, seed=77)
        f.insert(probe)
        self.assertTrue(f.query(probe)[0])


# ---------------------------------------------------------------------------
# AND-not-OR semantics -- the defining B3 property.
# ---------------------------------------------------------------------------

class TestLSBFAndSemantics(unittest.TestCase):
    """
    LSBF answers YES only if ALL num_hashes mapped bit positions are set.
    This constructs a probe with exactly (num_hashes - 1) of its positions
    set in the shared array and checks query() says NO -- an accidental OR
    implementation would say YES.
    """

    def test_all_but_one_bit_set_is_rejected(self):
        num_hashes = 4
        f = LSBFFilter(dim=DIM, distance_threshold=R, max_memory=200_000,
                        num_hashes=num_hashes, sub_bits_per_hash=4, verify=False, seed=42)
        f._reset_storage()

        probe = _rand(1, seed=999)
        codes = f._codes(np.ascontiguousarray(probe, dtype=np.float32))[0]  # (num_hashes,)

        # Set all bits except the last one.
        for h in range(num_hashes - 1):
            f._set_bit(f._primary, f._primary_bit_index(codes[h]))

        self.assertFalse(f.query(probe)[0], "AND semantics: (num_hashes-1)-of-num_hashes must be rejected")

        # Now set the final bit too -- must be accepted.
        f._set_bit(f._primary, f._primary_bit_index(codes[num_hashes - 1]))
        self.assertTrue(f.query(probe)[0], "all num_hashes bits set must be accepted")


# ---------------------------------------------------------------------------
# verify=True/False ablation
# ---------------------------------------------------------------------------

class TestLSBFVerifyAblation(unittest.TestCase):

    def test_bytesize_differs(self):
        f_verify = LSBFFilter(dim=DIM, distance_threshold=R, max_memory=100_000, verify=True)
        f_plain = LSBFFilter(dim=DIM, distance_threshold=R, max_memory=100_000, verify=False)
        self.assertNotEqual(f_verify.get_bytesize(), f_plain.get_bytesize())

    def test_verify_rejects_a_primary_false_positive(self):
        """
        Construct a probe that passes the primary AND-test (by directly
        setting its bits, simulating an LSH collision with unrelated stored
        data) but whose fingerprint was never OR'd into its verify bucket.
        verify=True must reject it; verify=False (no secondary check) must
        accept it -- demonstrating the ablation's FPR effect concretely.
        """
        num_hashes = 3
        f_v = LSBFFilter(dim=DIM, distance_threshold=R, max_memory=200_000,
                          num_hashes=num_hashes, sub_bits_per_hash=4, verify=True, seed=1)
        f_v._reset_storage()
        f_p = LSBFFilter(dim=DIM, distance_threshold=R, max_memory=200_000,
                          num_hashes=num_hashes, sub_bits_per_hash=4, verify=False, seed=1)
        f_p._lsh = f_v._lsh
        f_p._reset_storage()

        probe = _rand(1, seed=321)
        vecs = np.ascontiguousarray(probe, dtype=np.float32)
        codes = f_v._codes(vecs)[0]  # shared _lsh -> identical codes for both filters
        for h in range(num_hashes):
            # primary_bits differs between verify=True/False (budget split), so
            # each filter's bit index must be computed against its own modulus.
            f_v._set_bit(f_v._primary, f_v._primary_bit_index(codes[h]))
            f_p._set_bit(f_p._primary, f_p._primary_bit_index(codes[h]))
        # Deliberately do NOT populate f_v's verify bucket for this probe.

        self.assertTrue(f_p.query(probe)[0], "verify=False must accept a bare primary-AND pass")
        self.assertFalse(f_v.query(probe)[0], "verify=True must reject when the fingerprint was never recorded")


if __name__ == "__main__":
    unittest.main()
