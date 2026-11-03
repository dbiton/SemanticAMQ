import unittest
import numpy as np
from filters.cap_cover.cap_geometry import angle_between, row_angles, pair_angles, labelled_cap_cover, unit_rows


def _unit(rng, n, d):
    return unit_rows(rng.standard_normal((n, d)))


class FusedAnglesTest(unittest.TestCase):
    def test_row_angles_match_numpy(self):
        rng = np.random.default_rng(0)
        for d in (1, 3, 7, 384):
            X, U = _unit(rng, 200, d), _unit(rng, 17, d)
            lab = rng.integers(0, 17, 200)
            np.testing.assert_allclose(row_angles(X, U, lab), angle_between(X, U[lab]), rtol=0, atol=1e-13)

    def test_pair_angles_match_numpy_and_inf_padding(self):
        rng = np.random.default_rng(1)
        X, U = _unit(rng, 50, 9), _unit(rng, 6, 9)
        idx = rng.integers(0, 6, (50, 4)); idx[:, 3] = -1
        out = pair_angles(X, U, idx)
        self.assertTrue(np.isinf(out[:, 3]).all())
        ref = angle_between(X[:, None, :], U[idx[:, :3]])
        np.testing.assert_allclose(out[:, :3], ref, atol=1e-13)

    def test_coincident_and_duplicate_vectors(self):
        rng = np.random.default_rng(2)
        U = _unit(rng, 4, 16)
        X = np.repeat(U, 3, axis=0)
        lab = np.repeat(np.arange(4), 3)
        self.assertTrue((row_angles(X, U, lab) == 0.0).all())
        near = unit_rows(U[lab] + 1e-12 * rng.standard_normal(X.shape))
        np.testing.assert_allclose(row_angles(near, U, lab), angle_between(near, U[lab]), atol=1e-15)

    def test_bad_labels_rejected(self):
        X = _unit(np.random.default_rng(3), 3, 4)
        with self.assertRaises(ValueError):
            row_angles(X, X[:2], np.array([0, 1, 2]))

    def test_cover_still_covers(self):
        rng = np.random.default_rng(4)
        X, U = _unit(rng, 500, 24), _unit(rng, 20, 24)
        Uk, ang, lab, _ = labelled_cap_cover(X, U)
        self.assertTrue((angle_between(X, Uk[lab]) <= ang[lab] + 1e-15).all())


if __name__ == "__main__":
    unittest.main()
