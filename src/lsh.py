import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple


class LSH(ABC):
    def __init__(self, dim: int):
        """
        Base class for Locality Sensitive Hashing families.
        """
        self.dim = dim
    
    @abstractmethod
    def get_criteria(self, r1: float, r2: float) -> Tuple[float, float]:
        """
        Returns p1, p2 criteria given r1, r2 thresholds.
        """
        pass

    @abstractmethod
    def hash(self, vectors: np.ndarray) -> np.ndarray:
        """
        Hashes a batch of vectors.
        """
        pass


class SimHashLSH(LSH):
    def __init__(self, dim: int, num_bits: int):
        super().__init__(dim)
        self.num_bits = num_bits
        self.hyperplanes = np.random.randn(self.num_bits, self.dim)

    def get_criteria(self, r1: float, r2: float) -> Tuple[float, float]:
        # cos(theta) = 1 - r^2 / 2
        theta1 = np.arccos(np.clip(1.0 - (r1 ** 2) / 2.0, -1.0, 1.0))
        theta2 = np.arccos(np.clip(1.0 - (r2 ** 2) / 2.0, -1.0, 1.0))

        p1 = 1.0 - (theta1 / np.pi)
        p2 = 1.0 - (theta2 / np.pi)

        return float(p1), float(p2)

    def hash(self, vectors: np.ndarray) -> np.ndarray:
        projections = vectors @ self.hyperplanes.T
        return (projections > 0).astype(int)


class CrossPolytopeLSH(LSH):
    def __init__(self, dim: int, num_rotations: int):
        super().__init__(dim)
        self.num_rotations = num_rotations
        self.rotations = np.random.randn(self.num_rotations, self.dim, self.dim)

    def get_criteria(self, r1: float, r2: float) -> Tuple[float, float]:
        denom1 = 4.0 - (r1 ** 2)
        denom2 = 4.0 - (r2 ** 2)

        p1 = (self.dim ** (-(r1 ** 2) / denom1)) if denom1 > 0 else 0.0
        p2 = (self.dim ** (-(r2 ** 2) / denom2)) if denom2 > 0 else 0.0

        return float(p1), float(p2)

    def hash(self, vectors: np.ndarray) -> np.ndarray:
        n_samples = vectors.shape[0]
        signatures = np.zeros((n_samples, self.num_rotations), dtype=int)

        for i in range(self.num_rotations):
            rotated = vectors @ self.rotations[i].T
            max_idx = np.argmax(np.abs(rotated), axis=1)
            signs = np.sign(rotated[np.arange(n_samples), max_idx])
            signatures[:, i] = max_idx * 2 + (signs > 0).astype(int)

        return signatures