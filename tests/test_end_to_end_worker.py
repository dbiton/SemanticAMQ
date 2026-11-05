"""runner.end_to_end_worker: the filter in front of a flat index vs the index alone."""
import unittest

import numpy as np

import runner
from tests.helpers import lazy

DIM = 32


def _unit(x):
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


class TestEndToEndWorker(unittest.TestCase):

    def test_rows_per_cell_with_consistent_improvement(self):
        rng = np.random.default_rng(0)
        centers = _unit(rng.normal(size=(10, DIM)))
        B = _unit(centers[rng.integers(0, 10, 300)] + 0.05 * rng.normal(size=(300, DIM)))
        Q = _unit(np.vstack([centers[rng.integers(0, 10, 40)] + 0.05 * rng.normal(size=(40, DIM)),
                             rng.normal(size=(40, DIM))]))
        cells = [(0.1, 0.1), (0.1, 0.3), (0.2, 0.1)]
        rows = runner.end_to_end_worker(DIM, "synthetic", 3, cells, lazy(B), lazy(Q), repeats=1)
        self.assertEqual([(r["memory_fraction"], r["tau"]) for r in rows], sorted(cells))
        for r in rows:
            self.assertEqual((r["dataset"], r["seed"], r["n_queries"]), ("synthetic", 3, 80))
            self.assertGreaterEqual(r["positive_rate_pct"], r["true_positive_rate_pct"] - 1e-9)  # zero FN
            want = 100.0 * (r["throughput_with_qps"] - r["throughput_without_qps"]) / r["throughput_without_qps"]
            self.assertAlmostEqual(r["throughput_improvement_pct"], want, places=6)
            self.assertGreater(r["n_caps"], 0)
        # one cover per memory_fraction, one no-filter timing per tau
        a, b, c = rows
        self.assertEqual(a["n_caps"], b["n_caps"])
        self.assertEqual(a["memory_bytes"], b["memory_bytes"])


if __name__ == "__main__":
    unittest.main()
