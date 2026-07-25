import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'src'))
import unittest
import numpy as np
from numpy.testing import assert_allclose
from src.hyperspheres_volume import *

class TestHyperspheresVolume(unittest.TestCase):

    ## -----------------------------------------------------------------------------
    ## Math & Volume Tests
    ## -----------------------------------------------------------------------------

    def test_exact_volumes_2d_3d(self):
        """Test hypersphere volume formulas against known 2D (circle) and 3D (sphere) equations."""
        r = 2.5
        
        # 2D: pi * r^2
        vol_2d_expected = np.pi * (r ** 2)
        assert_allclose(calculate_hypersphere_volume(2, r), vol_2d_expected)
        
        # 3D: 4/3 * pi * r^3
        vol_3d_expected = (4.0 / 3.0) * np.pi * (r ** 3)
        assert_allclose(calculate_hypersphere_volume(3, r), vol_3d_expected)


    ## -----------------------------------------------------------------------------
    ## Estimation Tests
    ## -----------------------------------------------------------------------------

    def test_zero_spheres(self):
        """Test bounds when the input is empty."""
        centroids = np.empty((0, 3))
        radii = np.empty(0)
        
        # Linear scale
        low, high = estimate_hypersphere_volume(centroids, radii)
        self.assertEqual(low, 0.0)
        self.assertEqual(high, 0.0)
        
        # Log scale
        log_low, log_high = estimate_hypersphere_volume(centroids, radii, return_log=True)
        self.assertEqual(log_low, -np.inf)
        self.assertEqual(log_high, -np.inf)

    def test_single_sphere(self):
        """Test bounds on a single sphere (bounds should be exact and equal)."""
        centroids = np.array([[0.0, 0.0, 0.0]])
        radii = np.array([2.0])
        
        low, high = estimate_hypersphere_volume(centroids, radii)
        expected = calculate_hypersphere_volume(3, 2.0)
        
        assert_allclose(low, expected)
        assert_allclose(high, expected)

    def test_disjoint_spheres(self):
        """Test totally disconnected spheres (bounds should exactly equal the sum)."""
        centroids = np.array([
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 10.0]
        ])
        radii = np.array([1.0, 1.0, 1.0])
        
        low, high = estimate_hypersphere_volume(centroids, radii)
        expected_single = calculate_hypersphere_volume(2, 1.0)
        
        assert_allclose(low, expected_single * 3)
        assert_allclose(high, expected_single * 3)

    def test_identical_spheres(self):
        """Test heavily overlapping/identical spheres (should reduce to a single sphere)."""
        centroids = np.array([
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0]
        ])
        radii = np.array([3.0, 3.0, 3.0])
        
        low, high = estimate_hypersphere_volume(centroids, radii)
        expected = calculate_hypersphere_volume(2, 3.0)
        
        assert_allclose(low, expected)
        assert_allclose(high, expected)

    def test_contained_spheres(self):
        """Test smaller spheres entirely contained within a large one."""
        centroids = np.array([
            [0.0, 0.0],   # Big sphere center
            [0.5, 0.0],   # Small sphere strictly inside
            [-0.5, 0.0]   # Another small sphere inside
        ])
        radii = np.array([10.0, 1.0, 2.0])
        
        low, high = estimate_hypersphere_volume(centroids, radii)
        expected = calculate_hypersphere_volume(2, 10.0)
        
        assert_allclose(low, expected)
        assert_allclose(high, expected)

    def test_two_intersecting_spheres(self):
        """Test exact bounds for a single intersecting pair (V1 + V2 - V12)."""
        centroids = np.array([
            [0.0, 0.0],
            [2.0, 0.0]
        ])
        radii = np.array([2.0, 2.0])
        
        low, high = estimate_hypersphere_volume(centroids, radii)
        
        # For a 2-sphere component, the bounds should be perfectly tight (exact)
        assert_allclose(low, high)

    def test_three_intersecting_spheres(self):
        """Test valid bound inequalities for a complex intersection graph (n >= 3)."""
        centroids = np.array([
            [0.0, 0.0],
            [1.5, 0.0],
            [0.75, 1.25]
        ])
        radii = np.array([2.0, 2.0, 2.0])
        
        low, high = estimate_hypersphere_volume(centroids, radii)
        expected_sum = 3 * calculate_hypersphere_volume(2, 2.0)
        expected_max = calculate_hypersphere_volume(2, 2.0)
        
        # Lower bound must be <= Upper bound
        self.assertLessEqual(low, high)
        # Lower bound must be >= Max single volume
        self.assertGreaterEqual(low, expected_max)
        # Upper bound must be <= Sum of all volumes
        self.assertLessEqual(high, expected_sum)

    def test_high_dimension_underflow_protection(self):
        """Test dimensions where raw float64 volume catastrophically underflows."""
        dim = 1000
        centroids = np.zeros((2, dim))
        # Place them so they don't intersect
        centroids[1, 0] = 5.0  
        radii = np.array([1.0, 1.0])
        
        # The raw volume of a 1000-D unit sphere is ~ 1.7e-712, which is 0.0 in float64.
        # The linear call should yield exactly 0.0
        low_lin, high_lin = estimate_hypersphere_volume(centroids, radii, return_log=False)
        self.assertEqual(low_lin, 0.0)
        self.assertEqual(high_lin, 0.0)
        
        # The log calculation should succeed
        log_low, log_high = estimate_hypersphere_volume(centroids, radii, return_log=True)
        
        # Because they don't intersect, total volume is 2 * V_single
        # log(2 * V) = log(2) + log(V)
        expected_log_single = log_hypersphere_volume(dim, 1.0)
        expected_log_total = np.log(2.0) + expected_log_single
        
        assert_allclose(log_low, expected_log_total)
        assert_allclose(log_high, expected_log_total)

if __name__ == '__main__':
    unittest.main()