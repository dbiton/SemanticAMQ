import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional

class LSH(ABC):
    def __init__(self, dim: int, L: int, K: int):
        self.dim = dim
        self.L = L
        self.K = K

    @classmethod
    def build_optimized(cls, dim: int, threshold_d: float, K: Optional[int] = None, max_L: int = 256) -> "LSH":
        dummy = cls(dim=dim, L=1, K=1)
        p_threshold, _ = dummy._get_base_probabilities(threshold_d, threshold_d)

        if p_threshold <= 0:
            raise ValueError(f"Threshold D={threshold_d} yields a 0% base collision probability.")

        def calculate_L(k_val):
            p_k = np.clip(p_threshold ** k_val, 1e-300, 0.999999)
            return int(np.ceil(np.log(0.5) / np.log1p(-p_k)))

        if K is not None:
            L = calculate_L(K)
        else:
            best_K, best_L = 1, calculate_L(1)
            for test_K in range(2, 256):
                test_L = calculate_L(test_K)
                if test_L > max_L:
                    break
                best_K, best_L = test_K, test_L
            K, L = best_K, best_L

        if L > max_L:
            raise ValueError(f"Cannot achieve 50% threshold at D={threshold_d} within max_L={max_L}. Requires L={L}.")

        return cls(dim=dim, L=L, K=K)

    @abstractmethod
    def _get_base_probabilities(self, r1: float, r2: float) -> Tuple[float, float]:
        pass

    def get_criteria(self, r1: float, r2: float) -> Tuple[float, float]:
        base_p1, base_p2 = self._get_base_probabilities(r1, r2)
        final_p1 = 1.0 - (1.0 - base_p1 ** self.K) ** self.L
        final_p2 = 1.0 - (1.0 - base_p2 ** self.K) ** self.L
        return float(final_p1), float(final_p2)

    @abstractmethod
    def _compute_raw_hashes(self, vectors: np.ndarray) -> np.ndarray:
        pass

    def hash(self, vectors: np.ndarray) -> np.ndarray:
        raw_hashes = self._compute_raw_hashes(vectors)
        n_samples = vectors.shape[0]
        signatures = np.empty((n_samples, self.L), dtype=object)
        for l in range(self.L):
            for i in range(n_samples):
                signatures[i, l] = tuple(raw_hashes[i, l, :])
        return signatures


class SimHashLSH(LSH):
    def __init__(self, dim: int, L: int, K: int):
        super().__init__(dim, L, K)
        self.hyperplanes = np.random.randn(self.L, self.K, self.dim)

    def _get_base_probabilities(self, r1: float, r2: float) -> Tuple[float, float]:
        theta1 = np.arccos(np.clip(1.0 - (r1 ** 2) / 2.0, -1.0, 1.0))
        theta2 = np.arccos(np.clip(1.0 - (r2 ** 2) / 2.0, -1.0, 1.0))
        p1 = 1.0 - (theta1 / np.pi)
        p2 = 1.0 - (theta2 / np.pi)
        return float(p1), float(p2)

    def _compute_raw_hashes(self, vectors: np.ndarray) -> np.ndarray:
        projections = np.einsum('nd, lkd -> nlk', vectors, self.hyperplanes)
        return (projections > 0).astype(int)


class CrossPolytopeLSH(LSH):
    def __init__(self, dim: int, L: int, K: int):
        super().__init__(dim, L, K)
        self.rotations = np.zeros((self.L, self.K, self.dim, self.dim))
        for l in range(self.L):
            for k in range(self.K):
                H = np.random.randn(self.dim, self.dim)
                Q, R = np.linalg.qr(H)
                self.rotations[l, k] = Q
    
    # The close formula is only accurate when d->infinity
    def _get_base_probabilities(self, r1: float, r2: float) -> Tuple[float, float]:
        def estimate_p(r):
            if r <= 0.0: return 1.0
            if r >= 2.0: return 0.0
            
            # The theoretical formula d^(-r^2 / (4-r^2)) is an asymptotic bound. 
            # For finite dimensions, no simple closed form exists. We use a fast, 
            # vectorized Monte Carlo approximation of the exact underlying distribution.
            n_sim = 50000
            cos_theta = np.clip(1.0 - (r**2) / 2.0, -1.0, 1.0)
            sin_theta = np.sqrt(1.0 - cos_theta**2)
            
            # Generate bivariate Gaussians projected onto d dimensions
            X = np.random.randn(n_sim, self.dim)
            Z = np.random.randn(n_sim, self.dim)
            Y = X * cos_theta + Z * sin_theta
            
            max_idx_X = np.argmax(np.abs(X), axis=1)
            max_idx_Y = np.argmax(np.abs(Y), axis=1)
            
            signs_X = np.sign(X[np.arange(n_sim), max_idx_X])
            signs_Y = np.sign(Y[np.arange(n_sim), max_idx_Y])
            
            matches = (max_idx_X == max_idx_Y) & (signs_X == signs_Y)
            return float(np.mean(matches))
            
        return estimate_p(r1), estimate_p(r2)

    def _compute_raw_hashes(self, vectors: np.ndarray) -> np.ndarray:
        n_samples = vectors.shape[0]
        raw_hashes = np.zeros((n_samples, self.L, self.K), dtype=int)
        for l in range(self.L):
            for k in range(self.K):
                rotated = vectors @ self.rotations[l, k].T
                max_idx = np.argmax(np.abs(rotated), axis=1)
                signs = np.sign(rotated[np.arange(n_samples), max_idx])
                raw_hashes[:, l, k] = max_idx * 2 + (signs > 0).astype(int)
        return raw_hashes