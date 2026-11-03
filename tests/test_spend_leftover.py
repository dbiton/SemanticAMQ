"""CapCoverFilter(spend_leftover=True): bytes stay within max_memory, no false negatives, flag off is the reference."""
import unittest
import numpy as np
from filters.cap_cover.cap_cover_filter import CapCoverFilter
from filters.cap_cover.cap_quantizers import Quantize

TAU = 10 ** -0.5
LADDER = (Quantize.TQ2, Quantize.TQ3, Quantize.TQ4)


def unit(x):
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def corpus(n, d=64, seed=0):
    rng = np.random.default_rng(seed)
    centers = unit(rng.standard_normal((12, d)))
    return unit(centers[rng.integers(0, 12, n)] + 0.3 * rng.standard_normal((n, d)) / np.sqrt(d) * 3)


def exact_member(C, Q, tau):
    return np.sqrt(np.maximum(2 - 2 * Q @ C.T, 0)).min(axis=1) <= tau


class TestSpendLeftover(unittest.TestCase):
    def build(self, C, frac, flag):
        mm = int(frac * len(C) * C.shape[1] * 4)
        f = CapCoverFilter.auto(C.shape[1], mm, candidates=LADDER, spend_leftover=flag)
        f.construct(C)
        return f, mm

    def test_budget_and_no_false_negatives(self):
        C = corpus(1500)
        Q = np.vstack([C[:300], unit(C[:300] + 0.05 * np.random.default_rng(1).standard_normal((300, 64)))])
        for frac in (0.02, 0.05, 0.10, 0.2):
            for flag in (False, True):
                f, mm = self.build(C, frac, flag)
                self.assertLessEqual(f.get_bytesize(), mm)
                pred = np.asarray(f.query(Q, TAU), bool)
                self.assertFalse((~pred & exact_member(C, Q, TAU)).any(), (frac, flag))

    def test_spends_more_and_never_loosens_coverage(self):
        C = corpus(1500)
        on, mm = self.build(C, 0.10, True)
        off, _ = self.build(C, 0.10, False)
        self.assertGreaterEqual(on.get_bytesize(), off.get_bytesize())
        self.assertLessEqual(on.get_bytesize(), mm)
        # every construct vector inside some cap in float64
        for f in (on, off):
            self.assertTrue(np.asarray(f.query(C, 1e-9), bool).all())

    def test_tiny_and_degenerate_inputs(self):
        d = 64
        base = corpus(1, d)
        for C in (base, np.repeat(base, 5, axis=0), corpus(2, d), corpus(7, d)):
            for frac in (4.0, 16.0):
                f, mm = self.build(C, frac, True)
                self.assertLessEqual(f.get_bytesize(), mm)
                self.assertTrue(np.asarray(f.query(C, 1e-9), bool).all())

    def test_off_is_reference(self):
        C = corpus(800)
        a, _ = self.build(C, 0.10, False)
        b = CapCoverFilter.auto(64, int(0.10 * 800 * 64 * 4), candidates=LADDER)
        b.construct(C)
        self.assertEqual(a.get_bytesize(), b.get_bytesize())
        self.assertTrue(np.array_equal(a.all_angles, b.all_angles))


if __name__ == "__main__":
    unittest.main()
