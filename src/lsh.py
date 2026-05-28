import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional

class LSH(ABC):
    def __init__(self, dim: int, L: int, K: int):
        """
        Base class for Locality Sensitive Hashing families with AND-OR banding.
        
        Args:
            dim: Dimensionality of the input vectors.
            L: Number of independent hash tables (OR condition).
            K: Number of hashes concatenated per table (AND condition).
        """
        self.dim = dim
        self.L = L
        self.K = K
    
    @classmethod
    def build_optimized(cls, dim: int, threshold_d: float, K: Optional[int] = None, max_L: int = 256) -> "LSH":
        """
        Factory method to instantiate the LSH class with parameters tuned 
        to center the S-curve threshold at distance 'threshold_d'.
        
        Args:
            dim: Dimensionality of the embeddings.
            threshold_d: The target distance D where collision probability should be ~50%.
            K: Fixed number of hashes (AND). If None, it auto-searches for the best K.
            max_L: The maximum number of tables (OR) allowed in memory during auto-search.
        """
        # 1. Instantiate a dummy object to access the child class's specific probability math
        dummy = cls(dim=dim, L=1, K=1)
        p_threshold, _ = dummy._get_base_probabilities(threshold_d, threshold_d)
        
        if p_threshold <= 0:
            raise ValueError(f"Threshold D={threshold_d} yields a 0% base collision probability.")

        # 2. Mathematical function to calculate L given K for a 50% cutoff
        def calculate_L(k_val):
            p_k = np.clip(p_threshold ** k_val, 0.0, 0.999999)
            return int(np.ceil(np.log(0.5) / np.log(1.0 - p_k)))

        # 3. Optimize K and L
        if K is not None:
            # User defined K, just calculate L
            L = calculate_L(K)
        else:
            # Auto-tune: Find the highest K that keeps L <= max_L
            best_K, best_L = 1, calculate_L(1)
            
            for test_K in range(2, 256):
                test_L = calculate_L(test_K)
                if test_L > max_L:
                    break  # Exceeded memory limit, stop searching
                best_K, best_L = test_K, test_L
                
            K, L = best_K, best_L

        print(f"[{cls.__name__}] Tuned for D={threshold_d} (Base Prob = {p_threshold:.3f})")
        print(f"  -> Selected K (Hashes) = {K}")
        print(f"  -> Selected L (Tables) = {L}")
        
        # 4. Return a fully initialized instance of the child class
        return cls(dim=dim, L=L, K=K)

    @abstractmethod
    def _get_base_probabilities(self, r1: float, r2: float) -> Tuple[float, float]:
        """
        Calculates the single-hash collision probability for distances r1 and r2.
        Must be implemented by the child class.
        """
        pass

    def get_criteria(self, r1: float, r2: float) -> Tuple[float, float]:
        """
        Returns the final p1, p2 criteria given r1, r2 thresholds after 
        applying the AND-OR banding amplification.
        """
        base_p1, base_p2 = self._get_base_probabilities(r1, r2)
        
        # Apply the AND-OR S-Curve formula
        final_p1 = 1.0 - (1.0 - base_p1 ** self.K) ** self.L
        final_p2 = 1.0 - (1.0 - base_p2 ** self.K) ** self.L
        
        return float(final_p1), float(final_p2)

    @abstractmethod
    def _compute_raw_hashes(self, vectors: np.ndarray) -> np.ndarray:
        """
        Computes the raw integer/bit hashes for the batch.
        Must return an array of shape (N_samples, L, K).
        """
        pass

    def hash(self, vectors: np.ndarray) -> np.ndarray:
        """
        Hashes a batch of vectors and groups them into signature tuples.
        Returns an array of shape (N_samples, L) containing tuples of length K.
        """
        # Get the raw (N, L, K) hash matrix from the specific algorithm
        raw_hashes = self._compute_raw_hashes(vectors)
        
        n_samples = vectors.shape[0]
        signatures = np.empty((n_samples, self.L), dtype=object)

        # Combine the inner K dimension into a single tuple signature
        for l in range(self.L):
            for i in range(n_samples):
                signatures[i, l] = tuple(raw_hashes[i, l, :])

        return signatures


class SimHashLSH(LSH):
    def __init__(self, dim: int, L: int, K: int):
        super().__init__(dim, L, K)
        # Generate ALL hyperplanes for ALL tables in a single 3D tensor
        # Shape: (L, K, dim)
        self.hyperplanes = np.random.randn(self.L, self.K, self.dim)

    def _get_base_probabilities(self, r1: float, r2: float) -> Tuple[float, float]:
        theta1 = np.arccos(np.clip(1.0 - (r1 ** 2) / 2.0, -1.0, 1.0))
        theta2 = np.arccos(np.clip(1.0 - (r2 ** 2) / 2.0, -1.0, 1.0))

        p1 = 1.0 - (theta1 / np.pi)
        p2 = 1.0 - (theta2 / np.pi)

        return float(p1), float(p2)

    def _compute_raw_hashes(self, vectors: np.ndarray) -> np.ndarray:
        # Fully vectorized projection across all L tables and K bits
        # vectors: (N, dim), hyperplanes: (L, K, dim) -> output: (N, L, K)
        projections = np.einsum('nd, lkd -> nlk', vectors, self.hyperplanes)
        return (projections > 0).astype(int)


class CrossPolytopeLSH(LSH):
    def __init__(self, dim: int, L: int, K: int):
        super().__init__(dim, L, K)
        # Generate ALL rotations for ALL tables
        # Shape: (L, K, dim, dim)
        self.rotations = np.random.randn(self.L, self.K, self.dim, self.dim)

    def _get_base_probabilities(self, r1: float, r2: float) -> Tuple[float, float]:
        denom1 = 4.0 - (r1 ** 2)
        denom2 = 4.0 - (r2 ** 2)

        p1 = (self.dim ** (-(r1 ** 2) / denom1)) if denom1 > 0 else 0.0
        p2 = (self.dim ** (-(r2 ** 2) / denom2)) if denom2 > 0 else 0.0

        return float(p1), float(p2)

    def _compute_raw_hashes(self, vectors: np.ndarray) -> np.ndarray:
        n_samples = vectors.shape[0]
        raw_hashes = np.zeros((n_samples, self.L, self.K), dtype=int)

        # Loop through L and K to apply rotations. 
        # (Kept as a loop rather than einsum to manage memory spikes on large dims)
        for l in range(self.L):
            for k in range(self.K):
                rotated = vectors @ self.rotations[l, k].T
                max_idx = np.argmax(np.abs(rotated), axis=1)
                signs = np.sign(rotated[np.arange(n_samples), max_idx])
                
                raw_hashes[:, l, k] = max_idx * 2 + (signs > 0).astype(int)

        return raw_hashes