import math
import unittest
import numpy as np
from typing import Optional, Tuple

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


def simhash_base_p(r: float) -> float:
    """Closed-form SimHash base collision probability at distance r (unit vectors)."""
    return 1.0 - math.acos(max(-1.0, min(1.0, 1.0 - r * r / 2.0))) / math.pi


def reference_min_L(p: float, K: int, cap: int) -> int:
    """Smallest L with 1-(1-p^K)^L >= 0.5, by direct search. Returns cap+1 if none <= cap."""
    q = 1.0 - p ** K
    L = 1
    while q ** L > 0.5:
        L += 1
        if L > cap:
            return cap + 1
    return L


def reference_auto_KL(p: float, max_L: int) -> Optional[Tuple[int, int]]:
    """Largest K in [1, 255] whose minimal L fits max_L, via exhaustive scan.

    Independent of build_optimized's early-break loop and log-based L formula,
    so it cross-checks both. Returns None if even K=1 cannot fit.
    """
    best = None
    for K in range(1, 256):
        L = reference_min_L(p, K, cap=max_L)
        if L <= max_L:
            best = (K, L)
    return best


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
        
        # SimHash probability is closed-form, so the >= 0.5 guarantee is exact.
        p, _ = lsh.get_criteria(self.build_optimized_threshold_d, self.build_optimized_threshold_d)
        self.assertGreaterEqual(p, 0.5 - 1e-9)

    def test_simhash_build_optimized_auto_k(self):
        """Test SimHash optimization when K must be automatically discovered."""
        max_L_target = 150
        lsh = SimHashLSH.build_optimized(dim=self.dim, threshold_d=self.build_optimized_threshold_d, max_L=max_L_target)
        
        self.assertIsInstance(lsh, SimHashLSH)
        self.assertLessEqual(lsh.L, max_L_target)
        self.assertGreaterEqual(lsh.K, 1)
        
        # SimHash probability is closed-form, so the >= 0.5 guarantee is exact.
        p, _ = lsh.get_criteria(self.build_optimized_threshold_d, self.build_optimized_threshold_d)
        self.assertGreaterEqual(p, 0.5 - 1e-9)

    def test_crosspolytope_build_optimized_explicit_k(self):
        """Test CrossPolytope optimization when K is explicitly provided."""
        K_target = 3
        lsh = CrossPolytopeLSH.build_optimized(dim=self.dim, threshold_d=self.build_optimized_threshold_d, K=K_target)
        
        self.assertIsInstance(lsh, CrossPolytopeLSH)
        self.assertEqual(lsh.K, K_target)
        
        p, _ = lsh.get_criteria(self.build_optimized_threshold_d, self.build_optimized_threshold_d)

        # CrossPolytope's base probability is Monte Carlo estimated (SE ~0.0015) both in
        # build_optimized and again independently in get_criteria; the AND/OR construction
        # amplifies that error by roughly L*K*p^(K-1)*(1-p^K)^(L-1), so allow ~0.04 slack.
        self.assertGreaterEqual(p, 0.46)

    def test_crosspolytope_build_optimized_auto_k(self):
        """Test CrossPolytope optimization when K must be automatically discovered."""
        max_L_target = 100
        lsh = CrossPolytopeLSH.build_optimized(dim=self.dim, threshold_d=self.build_optimized_threshold_d, max_L=max_L_target)
        
        self.assertIsInstance(lsh, CrossPolytopeLSH)
        self.assertLessEqual(lsh.L, max_L_target)
        self.assertGreaterEqual(lsh.K, 1)
        
        p, _ = lsh.get_criteria(self.build_optimized_threshold_d, self.build_optimized_threshold_d)

        # Same Monte Carlo noise amplification as the explicit-K case above.
        self.assertGreaterEqual(p, 0.46)

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

class TestBuildOptimizedSelection(unittest.TestCase):
    """Validates the exact (K, L) chosen by build_optimized, not just loose bounds.

    Uses SimHash's closed-form base probability so expected values can be computed
    independently with reference_min_L / reference_auto_KL.
    """

    def setUp(self):
        # dim does not affect (K, L) selection for SimHash; keep small for speed.
        self.dim = 16

    def test_explicit_k_L_is_minimal(self):
        """For explicit K, L must be the *smallest* table count reaching 50% at the threshold."""
        for threshold_d in [0.5, 0.7, 1.1, 1.5]:
            p = simhash_base_p(threshold_d)
            for K in [1, 2, 8, 20]:
                expected_L = reference_min_L(p, K, cap=256)
                with self.subTest(threshold_d=threshold_d, K=K):
                    if expected_L > 256:
                        with self.assertRaisesRegex(ValueError, "Cannot achieve 50% threshold"):
                            SimHashLSH.build_optimized(dim=self.dim, threshold_d=threshold_d, K=K)
                        continue
                    lsh = SimHashLSH.build_optimized(dim=self.dim, threshold_d=threshold_d, K=K)
                    self.assertEqual(lsh.L, expected_L)
                    p_at, _ = lsh.get_criteria(threshold_d, threshold_d)
                    self.assertGreaterEqual(p_at, 0.5 - 1e-9)
                    if lsh.L > 1:
                        p_one_less = 1.0 - (1.0 - p ** K) ** (lsh.L - 1)
                        self.assertLess(p_one_less, 0.5, "L is not minimal: L-1 also reaches 50%")

    def test_auto_k_matches_bruteforce(self):
        """Auto-selected (K, L) must equal the largest K (and its minimal L) fitting max_L."""
        for threshold_d in [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]:
            p = simhash_base_p(threshold_d)
            for max_L in [4, 16, 64, 256]:
                expected = reference_auto_KL(p, max_L)
                with self.subTest(threshold_d=threshold_d, max_L=max_L):
                    lsh = SimHashLSH.build_optimized(dim=self.dim, threshold_d=threshold_d, max_L=max_L)
                    self.assertEqual((lsh.K, lsh.L), expected)
                    self.assertLessEqual(lsh.L, max_L)
                    p_at, _ = lsh.get_criteria(threshold_d, threshold_d)
                    self.assertGreaterEqual(p_at, 0.5 - 1e-9)

    def test_faiss_selects_same_parameters_as_simhash(self):
        """FaissSimHashLSH shares SimHash's closed form, so it must pick identical (K, L)."""
        for threshold_d, max_L in [(0.7, 64), (1.1, 256)]:
            with self.subTest(threshold_d=threshold_d, max_L=max_L):
                sim = SimHashLSH.build_optimized(dim=self.dim, threshold_d=threshold_d, max_L=max_L)
                fai = FaissSimHashLSH.build_optimized(dim=self.dim, threshold_d=threshold_d, max_L=max_L)
                self.assertEqual((fai.K, fai.L), (sim.K, sim.L))

    def test_threshold_zero_uses_single_table(self):
        """threshold_d=0 gives base probability 1.0: one table suffices; auto-K hits the 255 cap."""
        lsh_auto = SimHashLSH.build_optimized(dim=self.dim, threshold_d=0.0)
        self.assertEqual((lsh_auto.K, lsh_auto.L), (255, 1))
        lsh_explicit = SimHashLSH.build_optimized(dim=self.dim, threshold_d=0.0, K=4)
        self.assertEqual(lsh_explicit.L, 1)

    def test_auto_k_raises_when_even_k1_exceeds_max_l(self):
        """If K=1 already needs more than max_L tables, auto mode must raise, not silently clamp."""
        # At threshold 1.9 the base probability is ~0.20, so K=1 needs L=4 tables.
        with self.assertRaisesRegex(ValueError, "Cannot achieve 50% threshold"):
            SimHashLSH.build_optimized(dim=self.dim, threshold_d=1.9, max_L=3)

    def test_returned_instance_is_functional(self):
        """The built instance must hash with shapes consistent with its selected L."""
        lsh = SimHashLSH.build_optimized(dim=self.dim, threshold_d=0.7, max_L=32)
        self.assertEqual(lsh.dim, self.dim)
        vecs = np.random.randn(10, self.dim)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        self.assertEqual(lsh.hash(vecs).shape, (10, lsh.L))


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
        self.assertGreaterEqual(p, 0.5 - 1e-9)

    def test_build_optimized_auto_k(self):
        max_L_target = 150
        lsh = FaissSimHashLSH.build_optimized(
            dim=self.dim, threshold_d=self.build_optimized_threshold_d, max_L=max_L_target
        )
        self.assertIsInstance(lsh, FaissSimHashLSH)
        self.assertLessEqual(lsh.L, max_L_target)
        self.assertGreaterEqual(lsh.K, 1)
        p, _ = lsh.get_criteria(self.build_optimized_threshold_d, self.build_optimized_threshold_d)
        self.assertGreaterEqual(p, 0.5 - 1e-9)

    def test_build_optimized_empirical_guarantee(self):
        """End-to-end: an auto-built index must empirically reach ~50% collisions at the
        threshold distance while staying near 0% well beyond it (the point of maximizing K)."""
        np.random.seed(42)
        threshold_d = self.build_optimized_threshold_d
        lsh = FaissSimHashLSH.build_optimized(dim=self.dim, threshold_d=threshold_d, max_L=64)

        v1, v2 = generate_pairs_at_distance(self.n_samples, self.dim, threshold_d)
        empirical_at = float(np.mean((lsh.hash(v1) == lsh.hash(v2)).any(axis=1)))
        theoretical_at, _ = lsh.get_criteria(threshold_d, threshold_d)
        self.assertGreaterEqual(theoretical_at, 0.5 - 1e-9)
        self.assertTrue(
            math.isclose(empirical_at, theoretical_at, abs_tol=self.tolerance),
            f"at threshold: empirical={empirical_at:.4f} theoretical={theoretical_at:.4f}",
        )
        self.assertGreaterEqual(empirical_at, 0.5 - self.tolerance)

        far_d = 1.5
        f1, f2 = generate_pairs_at_distance(self.n_samples, self.dim, far_d)
        empirical_far = float(np.mean((lsh.hash(f1) == lsh.hash(f2)).any(axis=1)))
        self.assertLessEqual(empirical_far, 0.02, "collision rate far beyond threshold should be ~0")

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