import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'src'))
import unittest

import numpy as np

from src.filters.quantized_vector_filter import (
    QuantizedVectorFilter, ErrorBudgetMode, _LADDER, _footprint_bytes,
)

DIM = 64


def _clustered(n: int, dim: int = DIM, n_clusters: int = 40, spread: float = 0.25, seed: int = 0) -> np.ndarray:
    """Unit-norm vectors around n_clusters anchors -- tight enough that balls
    hold several points, the regime every radius argument here is about."""
    rng = np.random.default_rng(seed)
    anchors = rng.standard_normal((n_clusters, dim))
    anchors /= np.linalg.norm(anchors, axis=1, keepdims=True)
    x = anchors[rng.integers(0, n_clusters, n)] + spread / np.sqrt(dim) * rng.standard_normal((n, dim))
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


class TestQuantizedVectorFilterLevels(unittest.TestCase):
    """Every rung of QuantizedVectorFilter's ladder, each forced by a budget
    that fits it exactly, must recall every near-duplicate -- the eps_i
    argument rests on each rung's search() being exact w.r.t. decode()."""

    def test_every_level_has_no_false_negatives(self):
        dim = 128
        X = _clustered(1500, dim=dim)
        tau = 0.1
        rng = np.random.default_rng(3)
        step = rng.standard_normal(X.shape)
        step *= rng.uniform(0.2, 0.9, (len(X), 1)) * tau / np.linalg.norm(step, axis=1, keepdims=True)
        near = (X + step).astype(np.float32)
        for mode in ErrorBudgetMode:
            for level in _LADDER:
                with self.subTest(level=level.value, mode=mode.value):
                    budget = _footprint_bytes(level, len(X), dim, mode, 8)
                    f = QuantizedVectorFilter(dim=dim, distance_threshold=tau, max_memory=budget,
                                              error_budget_mode=mode)
                    f.construct(X)
                    self.assertEqual(f.level, level)
                    got = np.asarray(f.query(near, tau=tau))
                    self.assertTrue(got.all(), f"{np.sum(~got)} of {len(got)} near-duplicates rejected")
                    # Exact duplicates at tau=0 sit on the eps_i boundary --
                    # see QuantizedVectorFilter._COVERAGE_EPSILON.
                    got = np.asarray(f.query(X, tau=0.0))
                    self.assertTrue(got.all(), f"{np.sum(~got)} of {len(got)} stored vectors rejected at tau=0")


if __name__ == '__main__':
    unittest.main()
