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
        self.assertEqual(len(w.folds), 3)
        refs = [set(range(f.reference.start, f.reference.stop)) for f in w.folds]
        qs = [set(range(f.queries.start, f.queries.stop)) for f in w.folds]
        self.assertTrue(all(len(r) == 160 for r in refs) and all(len(q) == 40 for q in qs))
        for group in (refs, qs):
            self.assertEqual(len(set.union(*group)), 3 * len(group[0]))      # disjoint across folds
        self.assertLess(max(set.union(*refs)), min(set.union(*qs)))          # construct rows precede queries
        np.testing.assert_array_equal(w.folds[0].reference.to_array(), self.X[:160])
        np.testing.assert_array_equal(w.folds[0].queries.to_array(), self.X[-40:])

    def test_a_job_carries_handles_not_arrays(self):
        import pickle
        w = workloads.build_blocklist_workload("Toy", "toy", 200, 1)
        self.assertLess(len(pickle.dumps((w.folds[0].reference, w.folds[0].queries))), 1000)

    def test_stratified_split_returns_disjoint_indices_with_equal_rates(self):
        labels = np.array([True] * 20 + [False] * 180)
        cal, meas = workloads.stratified_calibration_split(labels, 0.25, seed=1)
        self.assertEqual(set(cal) & set(meas), set())
        self.assertEqual(len(cal) + len(meas), 200)
        self.assertAlmostEqual(labels[cal].mean(), labels[meas].mean(), places=2)

    def test_fold_ground_truth(self):
        w = workloads.build_blocklist_workload("Toy", "toy", 200, 1)
        truth = workloads.fold_ground_truth(w.folds[0], 0.0)
        self.assertEqual(truth.shape, (40,))
        self.assertFalse(truth.any())

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
