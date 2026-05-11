import numpy as np
from abc import ABC, abstractmethod


class LSH(ABC):
    def __init__(self, dim: int):
        """
        Base class for Locality Sensitive Hashing families.
        """
        self.dim = dim

    @abstractmethod
    def hash(self, vectors: np.ndarray) -> np.ndarray:
        """
        Hashes a batch of vectors.

        Args:
            vectors: np.ndarray of shape (n_samples, dim)

        Returns:
            np.ndarray of shape (n_samples, num_hashes) containing integer hashes.
        """
        pass


class SimHashLSH(LSH):
    def __init__(self, dim: int, num_bits: int):
        """
        SimHash uses random hyperplanes to partition the space.

        Args:
            dim: The dimensionality of the vectors.
            num_bits: The number of random hyperplanes (maps to CMS rows).
        """
        super().__init__(dim)
        self.num_bits = num_bits

        # Draw random hyperplanes from a standard normal distribution
        # Shape: (num_bits, dim)
        self.hyperplanes = np.random.randn(self.num_bits, self.dim)

    def hash(self, vectors: np.ndarray) -> np.ndarray:
        # Dot product with hyperplanes: shape (n_samples, num_bits)
        projections = vectors @ self.hyperplanes.T

        # Return binary signature (1 if positive, 0 if negative)
        return (projections > 0).astype(int)


class CrossPolytopeLSH(LSH):
    def __init__(self, dim: int, num_rotations: int):
        """
        Cross-Polytope finds the nearest vertex of a randomly rotated hyper-octahedron.
        It has better bounding properties for distant vectors compared to SimHash.

        Args:
            dim: The dimensionality of the vectors.
            num_rotations: The number of independent rotations (maps to CMS rows).
        """
        super().__init__(dim)
        self.num_rotations = num_rotations

        # Dense random Gaussian matrices to act as random rotations
        # Shape: (num_rotations, dim, dim)
        self.rotations = np.random.randn(self.num_rotations, self.dim, self.dim)

    def hash(self, vectors: np.ndarray) -> np.ndarray:
        n_samples = vectors.shape[0]
        # Output matrix
        signatures = np.zeros((n_samples, self.num_rotations), dtype=int)

        for i in range(self.num_rotations):
            # Rotate vectors: shape (n_samples, dim)
            rotated = vectors @ self.rotations[i].T

            # Find the coordinate with the maximum absolute value
            max_idx = np.argmax(np.abs(rotated), axis=1)

            # Extract the sign of that maximum value (+1 or -1)
            signs = np.sign(rotated[np.arange(n_samples), max_idx])

            # Map index and sign to a unique integer vertex ID
            # Since there are 'dim' axes, there are 2 * dim vertices.
            signatures[:, i] = max_idx * 2 + (signs > 0).astype(int)

        return signatures


class TurboQuantLSH(LSH):
    def __init__(self, dim: int, num_rows: int, num_buckets: int):
        """
        Proxy for a TurboQuant-style LSH (Random Orthogonal Preconditioning + Quantization).
        Uses orthogonal rotation matrices to isotropize the data, followed by 1-bit
        scalar quantization, which is then universally hashed into bucket IDs.

        Args:
            dim: The dimensionality of the vectors.
            num_rows: Number of independent hash tables (maps to CMS rows / K).
            num_buckets: The size of each hash table (maps to N).
        """
        super().__init__(dim)
        self.num_rows = num_rows
        self.num_buckets = num_buckets

        # 1. Generate random orthogonal matrices for each row
        self.rotations = np.zeros((num_rows, dim, dim))
        for i in range(num_rows):
            # Generate a random orthogonal matrix using QR decomposition
            # of a standard normal matrix (Haar measure proxy)
            H = np.random.randn(dim, dim)
            Q, R = np.linalg.qr(H)

            # Ensure a uniform distribution of signs to prevent reflection bias
            D = np.diag(np.sign(np.diag(R)))
            self.rotations[i] = Q @ D

        # 2. Random weights to hash the d-dimensional bit-vectors down to num_buckets
        # We use large primes/random integers for a simple universal hash step
        self.hash_weights = np.random.randint(1, 10000000, size=(num_rows, dim))

    def hash(self, vectors: np.ndarray) -> np.ndarray:
        n_samples = vectors.shape[0]
        signatures = np.zeros((n_samples, self.num_rows), dtype=int)

        for i in range(self.num_rows):
            # Step 1: Isotropization
            # Orthogonal rotation perfectly preserves L2 distances while forcing
            # the embedding into a uniform distribution across all axes.
            rotated = vectors @ self.rotations[i].T

            # Step 2: 1-bit Scalar Quantization
            # Converts the rotated floats into a discrete boolean signature
            bits = (rotated > 0).astype(int)

            # Step 3: Bucket Hashing
            # Reduce the dim-length bit vector to an integer bucket ID [0, num_buckets - 1]
            bucket_ids = (bits @ self.hash_weights[i]) % self.num_buckets

            signatures[:, i] = bucket_ids

        return signatures
