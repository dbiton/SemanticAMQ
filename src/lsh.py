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
        raw_hashes = self._compute_raw_hashes(vectors)  # (n_samples, L, K)
        # View each K-element row as a single opaque byte string so numpy's == compares
        # whole tuples without building Python objects.
        contiguous = np.ascontiguousarray(raw_hashes)
        void_dt = np.dtype((np.void, contiguous.dtype.itemsize * self.K))
        return contiguous.view(void_dt).reshape(vectors.shape[0], self.L)


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
        return (projections > 0).astype(np.uint8)


class CrossPolytopeLSH(LSH):
    def __init__(self, dim: int, L: int, K: int):
        super().__init__(dim, L, K)
        # Generate all L*K orthogonal matrices in one batched QR call.
        H = np.random.randn(self.L, self.K, self.dim, self.dim)
        Q, _ = np.linalg.qr(H)
        self.rotations = Q  # (L, K, dim, dim)

    # The close formula is only accurate when d->infinity
    def _get_base_probabilities(self, r1: float, r2: float) -> Tuple[float, float]:
        def estimate_p(r):
            if r <= 0.0: return 1.0
            if r >= 2.0: return 0.0

            cos_theta = np.clip(1.0 - (r**2) / 2.0, -1.0, 1.0)
            sin_theta = np.sqrt(1.0 - cos_theta**2)

            batch_size = 25000
            max_sims = 250000  # Hard ceiling to prevent infinite loops
            target_se = 0.0015  # Target Standard Error (0.15%)

            matches = 0
            total_sims = 0

            while total_sims < max_sims:
                X = np.random.randn(batch_size, self.dim)
                Z = np.random.randn(batch_size, self.dim)
                Y = X * cos_theta + Z * sin_theta

                max_idx_X = np.argmax(np.abs(X), axis=1)
                max_idx_Y = np.argmax(np.abs(Y), axis=1)

                signs_X = np.sign(X[np.arange(batch_size), max_idx_X])
                signs_Y = np.sign(Y[np.arange(batch_size), max_idx_Y])

                matches += np.sum((max_idx_X == max_idx_Y) & (signs_X == signs_Y))
                total_sims += batch_size

                p_hat = matches / total_sims
                se = np.sqrt(p_hat * (1.0 - p_hat) / total_sims)

                if se <= target_se:
                    break

            return float(p_hat)

        return estimate_p(r1), estimate_p(r2)

    def _compute_raw_hashes(self, vectors: np.ndarray) -> np.ndarray:
        n_samples = vectors.shape[0]
        raw_hashes = np.empty((n_samples, self.L, self.K), dtype=np.int32)
        # Pre-build index grids for advanced indexing inside the L loop.
        n_idx = np.arange(n_samples)[:, None]  # (n, 1)
        k_idx = np.arange(self.K)[None, :]     # (1, K)
        for l in range(self.L):
            # einsum: rotated[n,k,i] = sum_j vectors[n,j] * rotations[l,k,i,j]
            # Equivalent to vectors @ rotations[l,k].T for every k simultaneously.
            rotated = np.einsum('nj, kij -> nki', vectors, self.rotations[l])
            max_idx = np.argmax(np.abs(rotated), axis=-1)      # (n, K)
            selected = rotated[n_idx, k_idx, max_idx]           # (n, K)
            raw_hashes[:, l, :] = max_idx * 2 + (selected > 0).astype(np.int32)
        return raw_hashes

import faiss
class FaissSimHashLSH(LSH):
    def __init__(self, dim: int, L: int, K: int):
        super().__init__(dim, L, K)
        # One index with L*K hyperplanes is statistically equivalent to L separate
        # K-hyperplane tables because all hyperplanes are i.i.d. Gaussian. A single
        # FAISS call avoids the L-iteration Python loop and its per-call overhead.
        self._index = faiss.IndexLSH(dim, L * K)

    def _get_base_probabilities(self, r1: float, r2: float) -> Tuple[float, float]:
        theta1 = np.arccos(np.clip(1.0 - (r1 ** 2) / 2.0, -1.0, 1.0))
        theta2 = np.arccos(np.clip(1.0 - (r2 ** 2) / 2.0, -1.0, 1.0))
        p1 = 1.0 - (theta1 / np.pi)
        p2 = 1.0 - (theta2 / np.pi)
        return float(p1), float(p2)

    def _compute_raw_hashes(self, vectors: np.ndarray) -> np.ndarray:
        vecs_f32 = np.ascontiguousarray(vectors, dtype=np.float32)
        n_samples = vecs_f32.shape[0]
        # sa_encode packs bits LSB-first within each byte, so bitorder='little'
        # is required to correctly extract bit i from position i%8 of byte i//8.
        packed = self._index.sa_encode(vecs_f32)          # (n, ceil(L*K/8))
        bits = np.unpackbits(packed, axis=1, bitorder='little')[:, : self.L * self.K]
        return bits.reshape(n_samples, self.L, self.K)