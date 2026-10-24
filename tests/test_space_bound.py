"""util.space_bound: the information-theoretic space bound is vacuous at the accept-region
measures of real covers and positive below sigma(alpha)/sigma(2 alpha)."""
import math
import unittest

from util.space_bound import space_lower_bound, alpha_of_tau


class TestSpaceBound(unittest.TestCase):

    def test_vacuous_at_measured_eps(self):
        b = space_lower_bound(384, 0.2, 80_000, 1e-40)
        self.assertTrue(b["vacuous"])
        self.assertEqual(b["bits"], 0.0)
        self.assertAlmostEqual(b["log10_eps_max"], -111.96, places=1)

    def test_positive_below_threshold_and_grows_with_k(self):
        eps = 10.0 ** (space_lower_bound(384, 0.2, 1000, 1e-40)["log10_eps_max"] - 5)
        small = space_lower_bound(384, 0.2, 1000, eps)
        big = space_lower_bound(384, 0.2, 2000, eps)
        self.assertFalse(small["vacuous"])
        self.assertGreater(small["bits"], 0)
        self.assertAlmostEqual(big["bits"] / small["bits"], 2.0, places=2)    # linear in k

    def test_alpha(self):
        self.assertAlmostEqual(alpha_of_tau(math.sqrt(2 * 0.02)), math.acos(0.98), places=9)


if __name__ == "__main__":
    unittest.main()
