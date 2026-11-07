"""runner.end_to_end_worker: the filter in front of a flat index vs the index alone."""
import unittest
from unittest.mock import patch

import numpy as np

import runner
from tests.helpers import lazy

DIM = 32


def _unit(x):
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


class TestEndToEndWorker(unittest.TestCase):

    def test_timings_exclude_setup_and_remeasure_baseline_after_construction(self):
        clock = [0.0]
        builds = [0]

        class Index:
            def add(self, vectors):
                clock[0] += 100

            def search(self, vectors, k):
                clock[0] += 100
                return np.array([[0.0], [4.0]]), None

            def range_search(self, vector, radius):
                clock[0] += 2.0 if builds[0] else 5.0

        class Filter:
            _num_caps = 1
            encoding_label = "TQ2"

            def construct(self, vectors):
                builds[0] += 1
                self.cold = True
                clock[0] += 10

            def get_bytesize(self):
                clock[0] += 100
                return 8

            def query(self, vectors, distance_threshold):
                if self.cold:
                    clock[0] += 1000  # lazy Rust initialization
                    self.cold = False
                clock[0] += 0.25 * len(vectors)
                return [v[0] > 0 for v in vectors]

        vectors = np.array([[1., 0.], [-1., 0.]], dtype=np.float32)
        with patch.object(runner, "_pin_to_own_core"), \
             patch.object(runner, "_cap_cover", side_effect=lambda *args: Filter()), \
             patch.object(runner.faiss, "IndexFlatL2", return_value=Index()), \
             patch.object(runner.time, "perf_counter", side_effect=lambda: clock[0]):
            rows = runner.end_to_end_worker(2, "synthetic", 0,
                                           [(0.1, 0.1), (0.1, 0.2), (0.2, 0.1)],
                                           lazy(vectors), lazy(vectors), repeats=2)
        self.assertEqual(builds[0], 2)
        for row in rows:
            self.assertEqual(row["construct_s"], 10)
            self.assertEqual(row["prepare_s"], 1000.25)
            self.assertEqual(row["query_without_s"], 4)
            self.assertEqual(row["query_with_s"], 2.5)
            self.assertEqual(row["filter_query_s"], 0.5)
            self.assertEqual(row["throughput_without_qps"], 0.5)
            self.assertEqual(row["throughput_with_qps"], 0.8)

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
        # One cover per memory_fraction, fresh paired timings per cell.
        a, b, c = rows
        self.assertEqual(a["n_caps"], b["n_caps"])
        self.assertEqual(a["memory_bytes"], b["memory_bytes"])


if __name__ == "__main__":
    unittest.main()
