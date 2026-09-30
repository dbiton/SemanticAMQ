import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'src'))
import unittest

import numpy as np

from src.filters.va_plus_filter import (
    VAPlusFilter, InsufficientMemoryError, _allocate_bits, _footprint_bytes, _greedy_sequence,
)

DIM = 32
N = 600
TAU = 0.2
FRACTIONS = (0.1, 0.2, 0.5)


def _anisotropic(n: int = N, dim: int = DIM, seed: int = 0) -> np.ndarray:
    """Clustered, with variance decaying across dims -- the shape VA+'s
    non-uniform bit allocation is for."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((12, dim))
    scale = np.linspace(1.0, 0.05, dim)
    x = centers[rng.integers(0, len(centers), n)] + 0.4 * rng.standard_normal((n, dim)) * scale
    return x.astype(np.float32)


def _budget(frac: float, n: int = N) -> int:
    return int(n * DIM * 4 * frac)


def _built(frac: float = 0.5, x: np.ndarray = None, **kwargs) -> VAPlusFilter:
    x = _anisotropic() if x is None else x
    f = VAPlusFilter(DIM, TAU, _budget(frac, len(x)), **kwargs)
    f.construct(x)
    return f


def _basis(f: VAPlusFilter) -> np.ndarray:
    """Orthonormal basis of the stored fp32 rows (orthonormal only to ~1e-7),
    signs matched to the rows."""
    q, r = np.linalg.qr(f.components.astype(np.float64).T)
    return (q * np.sign(np.diag(r))).T


def _offsets(n: int, radius_lo: float, radius_hi: float, seed: int, basis: np.ndarray = None) -> np.ndarray:
    """Random offsets with norm in [radius_lo, radius_hi) * TAU, confined to
    the row span of `basis` when given."""
    rng = np.random.default_rng(seed)
    step = rng.standard_normal((n, DIM)) if basis is None else rng.standard_normal((n, len(basis))) @ basis
    step /= np.linalg.norm(step, axis=1, keepdims=True)
    return step * rng.uniform(radius_lo, radius_hi, (n, 1)) * TAU


# ---------------------------------------------------------------------------
# Zero-false-negative guarantee.
# ---------------------------------------------------------------------------

class TestVAPlusZeroFalseNegatives(unittest.TestCase):

    def _assert_all_accepted(self, f: VAPlusFilter, probes: np.ndarray, tau: float, what: str) -> None:
        got = np.asarray(f.query(probes.astype(np.float32), tau=tau))
        self.assertTrue(got.all(), f"[bits={f.bits.sum()}] {np.sum(~got)} of {len(got)} {what} rejected")

    def test_zero_fn(self):
        x = _anisotropic()
        for frac in FRACTIONS:
            f = _built(frac, x)
            with self.subTest(frac=frac):
                basis = _basis(f)
                self._assert_all_accepted(f, x + _offsets(N, 0.2, 0.99, 1), TAU, "near-duplicates")
                # Along the leading principal axes, then along the trailing
                # ones (which get few or no bits).
                self._assert_all_accepted(f, x + _offsets(N, 0.9, 0.99, 2, basis[:2]), TAU, "leading-axis offsets")
                self._assert_all_accepted(f, x + _offsets(N, 0.9, 0.99, 3, basis[-8:]), TAU, "trailing-axis offsets")
                # Exact duplicates at tau=0 sit on the box boundary.
                self._assert_all_accepted(f, x, 0.0, "exact duplicates at tau=0")
                # Past the data range on the first PCA dim (outer cells).
                coord = (x.astype(np.float64) - f.mean) @ basis[0]
                ext = np.concatenate([np.argsort(coord)[:5], np.argsort(coord)[-5:]])
                sign = np.where(coord[ext] > np.median(coord), 1.0, -1.0)[:, None]
                self._assert_all_accepted(f, x[ext] + sign * 0.95 * TAU * basis[0], TAU, "out-of-range queries")

    def test_query_matches_unpruned_lower_bound(self):
        """query()'s early pruning must decide exactly as thresholding the
        full min LB^2 does."""
        x = _anisotropic()
        rng = np.random.default_rng(7)
        probes = (x[rng.integers(0, N, 300)] + rng.standard_normal((300, DIM)) * 0.1).astype(np.float32)
        f = _built(0.3, x)
        thr = (TAU + VAPlusFilter._COVERAGE_EPSILON) ** 2
        np.testing.assert_array_equal(np.asarray(f.query(probes)), f.min_lower_bound_sq(probes) <= thr)

    def test_matches_direct_va_plus(self):
        """Direct VA+ from the stored pieces: project, box every x, and lower
        bound per dimension (0 inside the interval, else the gap)."""
        x = _anisotropic()
        f = _built(0.3, x)
        w = _basis(f)
        y = (x.astype(np.float64) - f.mean) @ w.T
        cells = f.cell_ids()
        lo = np.stack([f.boundaries[j][cells[:, j]] for j in range(DIM)], axis=1).astype(np.float64)
        hi = np.stack([f.boundaries[j][cells[:, j] + 1] for j in range(DIM)], axis=1).astype(np.float64)
        self.assertTrue(((lo <= y + 1e-9) & (y <= hi + 1e-9)).all(), "a vector is outside its own box")

        rng = np.random.default_rng(11)
        probes = (x[rng.integers(0, N, 100)] + rng.standard_normal((100, DIM)) * 0.2).astype(np.float32)
        yq = (probes.astype(np.float64) - f.mean) @ w.T
        gap = np.maximum(np.maximum(lo[None] - yq[:, None], yq[:, None] - hi[None]), 0.0)
        direct = (gap ** 2).sum(axis=2).min(axis=1)
        np.testing.assert_allclose(f.min_lower_bound_sq(probes), direct, rtol=1e-6, atol=1e-9)


# ---------------------------------------------------------------------------
# Bit allocation and memory accounting.
# ---------------------------------------------------------------------------

class TestVAPlusAllocation(unittest.TestCase):

    def test_greedy_sequence_is_vaplus_rule(self):
        """Each pick is the dim with the largest var_j / 4^b_j."""
        variances = np.sort(np.random.default_rng(0).exponential(size=24))[::-1]
        bits = np.zeros(24, dtype=np.int64)
        for j in _greedy_sequence(variances, 8):
            err = np.where(bits < 8, variances / 4.0 ** bits, -1.0)
            self.assertEqual(err[j], err.max())
            bits[j] += 1
        self.assertTrue((bits == 8).all())

    def test_allocation_is_largest_fitting_prefix(self):
        variances = np.sort(np.random.default_rng(0).exponential(size=24))[::-1]
        n, dim = 1_000, 24
        seq = _greedy_sequence(variances, 8)
        for budget in (3_000, 5_000, 20_000, 10 ** 7):
            with self.subTest(budget=budget):
                bits = _allocate_bits(variances, n, dim, 8, budget)
                total = int(bits.sum())
                self.assertTrue((bits == np.bincount(seq[:total], minlength=dim)).all(), "not the greedy prefix")
                self.assertLessEqual(_footprint_bytes(n, dim, bits), budget)
                self.assertTrue((np.diff(bits) <= 0).all(), f"higher-variance dim got fewer bits: {bits}")
                if total < len(seq):
                    more = bits.copy()
                    more[seq[total]] += 1
                    self.assertGreater(_footprint_bytes(n, dim, more), budget, "a further bit would fit")

    def test_bytesize_matches_hand_layout(self):
        x = _anisotropic()
        for frac in FRACTIONS:
            f = _built(frac, x)
            budget = _budget(frac)
            b = f.bits
            expected = (DIM * DIM + DIM) * 4                     # KLT matrix + mean
            expected += sum((2 ** int(bj) + 1) for bj in b) * 4  # cell boundaries
            expected += -(-N * int(b.sum()) // 8)                # packed codes
            self.assertEqual(f.get_bytesize(), expected)
            self.assertLessEqual(f.get_bytesize(), budget)
            self.assertEqual([len(e) for e in f.boundaries], [2 ** int(bj) + 1 for bj in b])

    def test_too_small_budget_raises(self):
        x = _anisotropic()
        f = VAPlusFilter(DIM, TAU, max_memory=4_000)  # below the 32x32 fp32 matrix + mean
        with self.assertRaises(InsufficientMemoryError) as ctx:
            f.construct(x)
        msg = str(ctx.exception)
        for part in ("4000", f"n={N}", f"dim={DIM}"):
            self.assertIn(part, msg)

    def test_same_seed_same_codes(self):
        x = _anisotropic()
        a, b = _built(0.3, x, seed=5), _built(0.3, x, seed=5)
        np.testing.assert_array_equal(a.cell_ids(), b.cell_ids())
        for ea, eb in zip(a.boundaries, b.boundaries):
            np.testing.assert_array_equal(ea, eb)

    def test_empty_and_insert(self):
        f = VAPlusFilter(DIM, TAU, max_memory=10_000)
        f.construct(np.zeros((0, DIM), dtype=np.float32))
        self.assertEqual(f.query(_anisotropic(3)), [False] * 3)
        self.assertEqual(f.get_bytesize(), 0)
        with self.assertRaises(NotImplementedError):
            f.insert(_anisotropic(3))


if __name__ == '__main__':
    unittest.main()
