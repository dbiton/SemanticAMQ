import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'src'))
import unittest

import numpy as np

from src.filters.ivf_filter import IVFFilter

DIM = 32
R = 0.3


def _rand(n: int, dim: int = DIM, seed: int = 0) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal((n, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------

class TestIVFSizing(unittest.TestCase):

    def test_pq_m_divides_dim(self):
        f = IVFFilter(dim=DIM, distance_threshold=R, max_memory=200_000)
        f.construct(_rand(200))
        self.assertEqual(DIM % f.pq_m, 0)

    def test_nlist_at_most_n(self):
        f = IVFFilter(dim=DIM, distance_threshold=R, max_memory=200_000)
        f.construct(_rand(60))
        self.assertLessEqual(f.nlist, 60)
        self.assertGreaterEqual(f.nlist, 1)

    def test_larger_budget_yields_finer_or_equal_code(self):
        small = IVFFilter(dim=DIM, distance_threshold=R, max_memory=5_000)
        small.construct(_rand(300, seed=1))
        large = IVFFilter(dim=DIM, distance_threshold=R, max_memory=5_000_000)
        large.construct(_rand(300, seed=1))
        self.assertGreaterEqual(large.pq_m, small.pq_m)

    def test_explicit_nlist_and_pq_m_respected(self):
        f = IVFFilter(dim=DIM, distance_threshold=R, max_memory=200_000, nlist=3, pq_m=4)
        f.construct(_rand(50))
        self.assertEqual(f.nlist, 3)
        self.assertEqual(f.pq_m, 4)

    def test_empty_construct_is_safe(self):
        f = IVFFilter(dim=DIM, distance_threshold=R, max_memory=200_000)
        f.construct(np.zeros((0, DIM), dtype=np.float32))
        self.assertEqual(f.get_bytesize(), 0)
        self.assertEqual(f.query(_rand(3)), [False, False, False])

    def test_singleton_construct_does_not_crash(self):
        """PQ training hard-errors below 2 points (faiss's "nx >= k" check on
        even a 1-bit codebook); a 1-vector construct set must fall back to an
        uncompressed index instead of raising."""
        f = IVFFilter(dim=DIM, distance_threshold=R, max_memory=200_000)
        vec = _rand(1, seed=3)
        f.construct(vec)
        self.assertTrue(f.query(vec)[0])


# ---------------------------------------------------------------------------
# Near-neighbor behavior (approximate, unlike CoveringLSHFallbackFilter's
# zero-FN guarantee -- IVFPQ can miss, so these assert high recall in
# practice rather than a hard invariant).
# ---------------------------------------------------------------------------

class TestIVFRecall(unittest.TestCase):

    def test_self_recall_with_full_nprobe(self):
        vecs = _rand(150, seed=5)
        f = IVFFilter(dim=DIM, distance_threshold=R, max_memory=500_000, nprobe=10_000)
        f.construct(vecs)
        predicted = f.query(vecs)
        fn = sum(1 for p in predicted if not p)
        self.assertLessEqual(fn, 1, f"unexpectedly high self-recall failures: {fn}/150")

    def test_dynamic_add_then_recall(self):
        f = IVFFilter(dim=DIM, distance_threshold=R, max_memory=500_000, nprobe=10_000)
        f.construct(_rand(80, seed=1))
        probe = _rand(5, seed=77)
        f.insert(probe)
        predicted = f.query(probe)
        fn = sum(1 for p in predicted if not p)
        self.assertLessEqual(fn, 1)


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

class TestIVFMemory(unittest.TestCase):

    def test_bytesize_zero_before_construct(self):
        f = IVFFilter(dim=DIM, distance_threshold=R, max_memory=200_000)
        self.assertEqual(f.get_bytesize(), 0)

    def test_bytesize_grows_after_construct(self):
        f = IVFFilter(dim=DIM, distance_threshold=R, max_memory=500_000)
        before = f.get_bytesize()
        f.construct(_rand(200))
        self.assertGreater(f.get_bytesize(), before)

    def test_bytesize_roughly_respects_budget(self):
        budget = 50_000
        f = IVFFilter(dim=DIM, distance_threshold=R, max_memory=budget)
        f.construct(_rand(2_000))
        # Fixed overhead (quantizer + PQ codebook) plus faiss's own index
        # bookkeeping means this is an approximate target, not a hard cap
        # (see module docstring); allow generous headroom.
        self.assertLessEqual(f.get_bytesize(), budget * 3)


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

class TestIVFInterface(unittest.TestCase):

    def test_query_before_construct_raises(self):
        f = IVFFilter(dim=DIM, distance_threshold=R, max_memory=200_000)
        with self.assertRaises(ValueError):
            f.query(_rand(1))

    def test_nprobe_setter_reaches_built_index(self):
        f = IVFFilter(dim=DIM, distance_threshold=R, max_memory=200_000, nprobe=1)
        f.construct(_rand(100))
        f.nprobe = 5
        self.assertEqual(f.index.nprobe, 5)
        self.assertEqual(f.nprobe, 5)

    def test_nprobe_readable_before_construct(self):
        f = IVFFilter(dim=DIM, distance_threshold=R, max_memory=200_000, nprobe=7)
        self.assertEqual(f.nprobe, 7)


if __name__ == "__main__":
    unittest.main()
