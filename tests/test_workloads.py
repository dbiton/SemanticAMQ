"""Positional fold selection (workloads.build_blocklist_workload) on a small HDF5 file."""
import os
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

import workloads
from util.dataset_paths import HDF5_DATASET_KEY


class TestBuildBlocklistWorkload(unittest.TestCase):

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = os.path.join(tmp.name, "toy.pkl")
        self.X = np.random.default_rng(0).normal(size=(1000, 8)).astype(np.float32)
        with h5py.File(self.path, "w") as f:
            f.create_dataset(HDF5_DATASET_KEY, data=self.X)
        patcher = patch.dict(workloads.dataset_filenames, {"Toy": self.path})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_folds_are_positional_and_disjoint(self):
        w = workloads.build_blocklist_workload("Toy", "toy", 200, 3)
        self.assertEqual(len(w.selections), 3)
        refs = [set(s["reference"]) for s in w.selections]
        qs = [set(s["queries"]) for s in w.selections]
        self.assertTrue(all(len(r) == 160 for r in refs) and all(len(q) == 40 for q in qs))
        for group in (refs, qs):
            self.assertEqual(len(set.union(*group)), 3 * len(group[0]))      # disjoint across folds
        self.assertLess(max(set.union(*refs)), min(set.union(*qs)))          # construct rows precede queries
        np.testing.assert_array_equal(w.rows(w.selections[0]["reference"]), self.X[:160])
        np.testing.assert_array_equal(w.query_vectors(w.selections[0]), self.X[-40:])

    def test_file_too_short_is_skipped(self):
        self.assertIsNone(workloads.build_blocklist_workload("Toy", "toy", 600, 2))

    def test_missing_file_is_skipped(self):
        os.remove(self.path)
        self.assertIsNone(workloads.build_blocklist_workload("Toy", "toy", 100, 1))

    def test_ground_truth_is_distance_at_tau(self):
        ref = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        q = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
        self.assertEqual(workloads.compute_static_actual_results(ref, q, 0.5), [True, False])


if __name__ == "__main__":
    unittest.main()
