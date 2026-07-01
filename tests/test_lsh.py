import math
import unittest
import numpy as np
from typing import Tuple

from src.lsh import CrossPolytopeLSH, FaissSimHashLSH, SimHashLSH

# Assuming your LSH classes are saved in a file named `lsh_module.py`
# from lsh_module import SimHashLSH, CrossPolytopeLSH

def generate_pairs_at_distance(n_pairs: int, dim: int, r: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates pairs of unit vectors separated by an exact Euclidean distance 'r'.
    """
    # 1. Generate first set of random unit vectors
    v1 = np.random.randn(n_pairs, dim)
    v1 /= np.linalg.norm(v1, axis=1, keepdims=True)
    
    # 2. Calculate the required angle (theta) for the target Euclidean distance (r)
    # Euclidean distance r = sqrt(2 - 2*cos(theta)) => cos(theta) = 1 - r^2/2
    cos_theta = np.clip(1.0 - (r**2) / 2.0, -1.0, 1.0)
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    
    # 3. Generate a random orthogonal vector for each v1
    v_rand = np.random.randn(n_pairs, dim)
    proj = np.sum(v_rand * v1, axis=1, keepdims=True) * v1  # Projection of v_rand onto v1
    v_ortho = v_rand - proj                                 # Remove projection to make orthogonal
    v_ortho /= np.linalg.norm(v_ortho, axis=1, keepdims=True)
    
    # 4. Construct v2 by rotating v1 towards v_ortho by theta
    v2 = v1 * cos_theta + v_ortho * sin_theta
    
    return v1, v2

class TestLSHProbabilities(unittest.TestCase):
    
    def setUp(self):
        # High dimensionality ensures CrossPolytope approximations hold true
        self.dim = 128
        # Using a large sample size minimizes variance in empirical probability
        self.n_samples = 15000
        # Absolute tolerance for the probability check (3%)
        self.tolerance = 0.03  
        
        # Generate some test distances (S-curve validation points)
        self.test_distances = [0.2, 0.7, 1.1, 1.5]

        self.build_optimized_threshold_d = 0.7

    def verify_empirical_vs_theoretical(self, lsh_instance, r: float):
        """Helper method to execute the probabilistic test for a given LSH instance and distance."""
        # Generate vectors
        v1, v2 = generate_pairs_at_distance(self.n_samples, self.dim, r)
        
        # Hash both vector batches
        sig1 = lsh_instance.hash(v1)  # Shape: (N, L)
        sig2 = lsh_instance.hash(v2)  # Shape: (N, L)
        
        # Boolean matrix of matches: True if the K-tuple matches at table 'l'
        matches = (sig1 == sig2)
        
        # A collision is successful if there is a match in ANY of the L tables (OR condition)
        collisions = matches.any(axis=1)
        
        empirical_p = np.mean(collisions)
        theoretical_p, _ = lsh_instance.get_criteria(r, r)
        
        # Validate that empirical matches theoretical within the tolerance
        is_accurate = math.isclose(empirical_p, theoretical_p, abs_tol=self.tolerance)
        
        error_msg = (f"\nFailed at distance {r}:\n"
                     f"  Theoretical Prob: {theoretical_p:.4f}\n"
                     f"  Empirical Prob:   {empirical_p:.4f}\n"
                     f"  Difference:       {abs(theoretical_p - empirical_p):.4f} "
                     f"(Tolerance: {self.tolerance})")
        
        self.assertTrue(is_accurate, error_msg)

    def test_simhash_probabilities(self):
        """Test SimHash theoretical vs empirical probabilities."""
        # Using a standard configuration: 5 tables (OR), 12 hashes per table (AND)
        lsh = SimHashLSH(dim=self.dim, L=5, K=12)
        
        for r in self.test_distances:
            with self.subTest(r=r, method="SimHash"):
                self.verify_empirical_vs_theoretical(lsh, r)

    def test_crosspolytope_probabilities(self):
        """Test CrossPolytope theoretical vs empirical probabilities."""
        # Using a standard configuration: 8 tables (OR), 2 hashes per table (AND)
        lsh = CrossPolytopeLSH(dim=self.dim, L=8, K=2)
        
        for r in self.test_distances:
            with self.subTest(r=r, method="CrossPolytope"):
                self.verify_empirical_vs_theoretical(lsh, r)
        
    def test_simhash_build_optimized_explicit_k(self):
        """Test SimHash optimization when K is explicitly provided."""
        K_target = 8
        lsh = SimHashLSH.build_optimized(dim=self.dim, threshold_d=self.build_optimized_threshold_d, K=K_target)
        
        self.assertIsInstance(lsh, SimHashLSH)
        self.assertEqual(lsh.K, K_target)
        
        # Theoretical probability at threshold should be >= 0.5 (allowing slight float rounding)
        p, _ = lsh.get_criteria(self.build_optimized_threshold_d, self.build_optimized_threshold_d)
        self.assertGreaterEqual(p, 0.49)

    def test_simhash_build_optimized_auto_k(self):
        """Test SimHash optimization when K must be automatically discovered."""
        max_L_target = 150
        lsh = SimHashLSH.build_optimized(dim=self.dim, threshold_d=self.build_optimized_threshold_d, max_L=max_L_target)
        
        self.assertIsInstance(lsh, SimHashLSH)
        self.assertLessEqual(lsh.L, max_L_target)
        self.assertGreaterEqual(lsh.K, 1)
        
        # Theoretical probability at threshold should be >= 0.5
        p, _ = lsh.get_criteria(self.build_optimized_threshold_d, self.build_optimized_threshold_d)
        self.assertGreaterEqual(p, 0.49)

    def test_crosspolytope_build_optimized_explicit_k(self):
        """Test CrossPolytope optimization when K is explicitly provided."""
        K_target = 3
        lsh = CrossPolytopeLSH.build_optimized(dim=self.dim, threshold_d=self.build_optimized_threshold_d, K=K_target)
        
        self.assertIsInstance(lsh, CrossPolytopeLSH)
        self.assertEqual(lsh.K, K_target)
        
        p, _ = lsh.get_criteria(self.build_optimized_threshold_d, self.build_optimized_threshold_d)
        
        self.assertGreaterEqual(p, 0.49)

    def test_crosspolytope_build_optimized_auto_k(self):
        """Test CrossPolytope optimization when K must be automatically discovered."""
        max_L_target = 100
        lsh = CrossPolytopeLSH.build_optimized(dim=self.dim, threshold_d=self.build_optimized_threshold_d, max_L=max_L_target)
        
        self.assertIsInstance(lsh, CrossPolytopeLSH)
        self.assertLessEqual(lsh.L, max_L_target)
        self.assertGreaterEqual(lsh.K, 1)
        
        p, _ = lsh.get_criteria(self.build_optimized_threshold_d, self.build_optimized_threshold_d)
        
        # Account for Monte Carlo variance
        self.assertGreaterEqual(p, 0.49)

    def test_build_optimized_zero_probability_error(self):
        """Test that a threshold yielding <= 0% base probability raises a ValueError."""
        # For SimHash, a distance of 2.0 means vectors are perfectly opposite (180 degrees), 
        # resulting in a base collision probability of exactly 0.0.
        with self.assertRaisesRegex(ValueError, "yields a 0% base collision probability"):
            SimHashLSH.build_optimized(dim=self.dim, threshold_d=2.0)

    def test_build_optimized_max_l_error(self):
        """Test that failure to achieve the 50% threshold within max_L raises a ValueError."""
        # Force a failure by requiring an impossibly strict K for a relaxed distance, 
        # combined with a very small max_L allowance.
        with self.assertRaisesRegex(ValueError, "Cannot achieve 50% threshold"):
            SimHashLSH.build_optimized(dim=self.dim, threshold_d=1.5, K=50, max_L=5)

class TestFaissSimHashLSH(unittest.TestCase):

    def setUp(self):
        self.dim = 128
        self.n_samples = 15000
        self.tolerance = 0.04
        self.test_distances = [0.2, 0.7, 1.1, 1.5]
        self.build_optimized_threshold_d = 0.7

    # --- output shape ---

    def test_raw_hash_shape(self):
        for K in (1, 7, 8, 12, 16):
            with self.subTest(K=K):
                lsh = FaissSimHashLSH(dim=self.dim, L=5, K=K)
                vecs = np.random.randn(20, self.dim).astype(np.float32)
                raw = lsh._compute_raw_hashes(vecs)
                self.assertEqual(raw.shape, (20, 5, K))
                self.assertTrue(np.all((raw == 0) | (raw == 1)), "bits must be 0 or 1")

    def test_hash_shape(self):
        lsh = FaissSimHashLSH(dim=self.dim, L=5, K=12)
        vecs = np.random.randn(20, self.dim).astype(np.float32)
        h = lsh.hash(vecs)
        self.assertEqual(h.shape, (20, 5))

    def test_accepts_float64_input(self):
        """_compute_raw_hashes must handle float64 without error."""
        lsh = FaissSimHashLSH(dim=self.dim, L=3, K=8)
        vecs = np.random.randn(10, self.dim)  # float64
        raw = lsh._compute_raw_hashes(vecs)
        self.assertEqual(raw.shape, (10, 3, 8))

    # --- correctness: collision probabilities ---

    def test_probabilities(self):
        lsh = FaissSimHashLSH(dim=self.dim, L=5, K=12)
        for r in self.test_distances:
            with self.subTest(r=r):
                v1, v2 = generate_pairs_at_distance(self.n_samples, self.dim, r)
                sig1 = lsh.hash(v1)
                sig2 = lsh.hash(v2)
                empirical_p = float(np.mean((sig1 == sig2).any(axis=1)))
                theoretical_p, _ = lsh.get_criteria(r, r)
                self.assertTrue(
                    math.isclose(empirical_p, theoretical_p, abs_tol=self.tolerance),
                    f"r={r}: empirical={empirical_p:.4f} theoretical={theoretical_p:.4f} "
                    f"diff={abs(empirical_p - theoretical_p):.4f}",
                )

    def test_probabilities_non_multiple_of_8_K(self):
        """K=7 exercises the LSB-first bit extraction path for non-byte-aligned hashes."""
        lsh = FaissSimHashLSH(dim=self.dim, L=8, K=7)
        r = 0.7
        v1, v2 = generate_pairs_at_distance(self.n_samples, self.dim, r)
        sig1 = lsh.hash(v1)
        sig2 = lsh.hash(v2)
        empirical_p = float(np.mean((sig1 == sig2).any(axis=1)))
        theoretical_p, _ = lsh.get_criteria(r, r)
        self.assertTrue(
            math.isclose(empirical_p, theoretical_p, abs_tol=self.tolerance),
            f"K=7, r={r}: empirical={empirical_p:.4f} theoretical={theoretical_p:.4f}",
        )

    # --- build_optimized ---

    def test_build_optimized_explicit_k(self):
        K_target = 8
        lsh = FaissSimHashLSH.build_optimized(
            dim=self.dim, threshold_d=self.build_optimized_threshold_d, K=K_target
        )
        self.assertIsInstance(lsh, FaissSimHashLSH)
        self.assertEqual(lsh.K, K_target)
        p, _ = lsh.get_criteria(self.build_optimized_threshold_d, self.build_optimized_threshold_d)
        self.assertGreaterEqual(p, 0.49)

    def test_build_optimized_auto_k(self):
        max_L_target = 150
        lsh = FaissSimHashLSH.build_optimized(
            dim=self.dim, threshold_d=self.build_optimized_threshold_d, max_L=max_L_target
        )
        self.assertIsInstance(lsh, FaissSimHashLSH)
        self.assertLessEqual(lsh.L, max_L_target)
        self.assertGreaterEqual(lsh.K, 1)
        p, _ = lsh.get_criteria(self.build_optimized_threshold_d, self.build_optimized_threshold_d)
        self.assertGreaterEqual(p, 0.49)

    # --- determinism: same index, same hashes ---

    def test_deterministic(self):
        """The same index must produce identical hashes on repeated calls."""
        lsh = FaissSimHashLSH(dim=self.dim, L=4, K=8)
        vecs = np.random.randn(50, self.dim).astype(np.float32)
        h1 = lsh.hash(vecs)
        h2 = lsh.hash(vecs)
        self.assertTrue(np.all(h1 == h2))


if __name__ == '__main__':
    unittest.main(verbosity=2)