"""SoftAnnealMerge: sphere merge whose final cover is annealed; the cover stays valid and the
construct-set objective never gets worse than sphere merge's."""
import unittest

import numpy as np

from filters.cap_cover.cap_cover_filter import CapCoverFilter
from filters.cap_cover.cap_geometry import labelled_cap_cover
from filters.cap_cover.soft_anneal_construct import SoftAnneal, SoftAnnealMerge, hard_eval
from filters.cap_cover.sphere_merge_construct import SphereMergeConstruct
import runner


def _unit(a):
    return a / np.linalg.norm(a, axis=1, keepdims=True)


class TestSoftAnneal(unittest.TestCase):
    DIM = 384

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        centers = _unit(rng.normal(size=(25, cls.DIM)))
        cls.X = _unit(centers[rng.integers(0, 25, 700)] + 0.05 * rng.normal(size=(700, cls.DIM))).astype(np.float32)
        near = cls.X[rng.integers(0, 700, 100)] + 0.004 * rng.normal(size=(100, cls.DIM))   # within tau of an indexed vector
        cls.Q = _unit(np.vstack([near, rng.normal(size=(60, cls.DIM))])).astype(np.float32)

    def test_registered(self):
        self.assertIsInstance(runner.CAP_CONSTRUCT_METHODS["soft_anneal"](), SoftAnnealMerge)

    def test_valid_cover_and_objective_not_worse(self):
        k = 60
        X = self.X.astype(np.float64)
        filt = CapCoverFilter(self.DIM, max_caps=k)
        base_U, _ = SphereMergeConstruct().construct_filter(filt, X)
        U, theta = SoftAnnealMerge(anneal=SoftAnneal(iters=40)).construct_filter(filt, X)
        self.assertLessEqual(len(U), k)
        # every vector lies in some cap, within its angle
        ang = np.arccos(np.clip(X @ U.T, -1, 1))
        self.assertTrue(np.all((ang <= theta[None, :] + 1e-6).any(axis=1)))
        alpha = float(filt._beta(np.ones(1), 0.1)[0][0])
        j_base = hard_eval(X, base_U, alpha, self.DIM)[0]
        j_new = hard_eval(X, U, alpha, self.DIM)[0]
        self.assertLessEqual(j_new, j_base + 1e-9)

    def test_filter_has_no_false_negatives(self):
        filt = CapCoverFilter.auto(self.DIM, max_memory=runner._cover_max_memory(self.DIM, 70),
                                   construct_method=SoftAnnealMerge(anneal=SoftAnneal(iters=30)))
        filt.construct(self.X)
        tau = runner.tau_of_cosine_distance(0.05)
        actual = (2.0 - 2.0 * (self.Q @ self.X.T).max(axis=1)) <= tau * tau
        accepted = np.asarray(filt.query(self.Q, distance_threshold=tau), dtype=bool)
        self.assertTrue(actual.any())
        self.assertTrue(np.all(accepted[actual]))


if __name__ == "__main__":
    unittest.main()
