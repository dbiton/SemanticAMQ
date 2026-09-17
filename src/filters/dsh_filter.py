"""
DSHFilter -- Distance-Sensitive Hashing (Aumuller, Christiani, Pagh &
Silvestri, PODS 2018), instantiated as h(x) = 1 iff <a,x> in [-z, z] for
Gaussian a (a window centered at 0, since this filter only needs sensitivity
to a ball around the query, not the paper's more general off-origin
annulus), concatenated `num_projections_per_table` times per table across
`num_tables` tables. Calibration reduces to the single free parameter z.

Calibration maximizes p(tau; z) - p(r_far; z) (r_far defaults to sqrt(2),
the expected distance between independent unit-sphere points) rather than
p(0) - p(tau): two points at distance 0 always collide regardless of z, so
p(0) carries no calibration signal. Both probabilities are estimated by
batched Monte Carlo (same pattern as
`src/lsh.py::CrossPolytopeLSH._get_base_probabilities`), and z is chosen
from a bounded grid.

If the best achievable gap falls below `calibration_min_gap`, the
constructor raises RuntimeError with the measured gap rather than silently
building a non-functional filter.
"""

import math
from typing import List, Optional, Tuple

import numpy as np
from pybloom_live import BloomFilter as _BloomFilter

from filters.filter import Filter
from filters._hashing import signature_to_key


class DSHFilter(Filter):
    """Distance-sensitive hashing (see module docstring).

    tau is baked in at construction via `distance_threshold` (z and
    num_tables are both calibrated around it); `query(..., tau=t)` only
    accepts t == distance_threshold or t=None.
    """

    def __init__(
        self,
        dim: int,
        distance_threshold: float,
        max_memory: Optional[int] = None,
        num_projections_per_table: int = 4,
        num_tables: Optional[int] = None,
        max_L: int = 256,
        error_rate: float = 0.01,
        calibration_grid: int = 13,
        calibration_min_gap: float = 0.05,
        far_reference_distance: Optional[float] = None,
        max_spheres: Optional[int] = None,
        seed: int = 42,
    ):
        super().__init__(dim, max_memory=max_memory)
        self.distance_threshold = float(distance_threshold)
        self.num_projections_per_table = num_projections_per_table
        self._error_rate = error_rate
        self._max_spheres = max_spheres  # unused; accepted for _instantiate_filter parity
        # Real embeddings cluster much more tightly than a uniform-random
        # unit sphere, so a fixed sqrt(2) reference can leave no calibration
        # margin -- scale off tau itself instead.
        self.far_reference_distance = (
            far_reference_distance if far_reference_distance is not None
            else max(math.sqrt(2.0), 2.5 * self.distance_threshold)
        )

        rng = np.random.default_rng(seed)

        self.z, best_gap, self._p_table_tau = self._calibrate(
            rng, dim, self.distance_threshold, self.far_reference_distance,
            num_projections_per_table, calibration_grid,
        )
        if best_gap < calibration_min_gap:
            raise RuntimeError(
                f"DSHFilter calibration did not converge to a usable operating point: "
                f"best achievable gap p(tau)-p(far) = {best_gap:.4f} at z={self.z:.4f} "
                f"is below calibration_min_gap={calibration_min_gap} "
                f"(dim={dim}, distance_threshold={self.distance_threshold}, "
                f"num_projections_per_table={num_projections_per_table}). "
                f"Try a smaller num_projections_per_table, a larger calibration_grid, "
                f"or accept a lower calibration_min_gap."
            )

        if num_tables is None:
            p = np.clip(self._p_table_tau, 1e-300, 0.999999)
            self.num_tables = int(np.ceil(np.log(0.5) / np.log1p(-p)))
            if self.num_tables > max_L:
                raise RuntimeError(
                    f"DSHFilter would need num_tables={self.num_tables} to reach 50% "
                    f"collision probability at tau, exceeding max_L={max_L}."
                )
        else:
            self.num_tables = num_tables

        self._projections = rng.standard_normal(
            (self.num_tables, num_projections_per_table, dim)
        ).astype(np.float32)

        if max_memory is not None:
            bits_available = max_memory * 8
            self._capacity = max(
                1, int(bits_available * (math.log(2) ** 2) / (-math.log(error_rate)))
            )
        else:
            self._capacity = 1_000 * self.num_tables

        self.filter: Optional[_BloomFilter] = None

    # ------------------------------------------------------------------ calibration

    @staticmethod
    def _match_probability(
        rng: np.random.Generator, dim: int, r: float, z: float,
        batch_size: int = 8_000, max_sims: int = 40_000, target_se: float = 0.004,
    ) -> float:
        """
        Monte Carlo estimate of P(bit(X) == bit(Q)) for a single Gaussian
        projection with acceptance window [-z, z], where X is a random unit
        vector and Q is at Euclidean distance r from X in a random direction
        -- batched with a target-standard-error stopping rule, mirroring
        `CrossPolytopeLSH._get_base_probabilities`.
        """
        if r <= 0.0:
            return 1.0
        matches = 0
        total = 0
        p_hat = 1.0
        while total < max_sims:
            X = rng.standard_normal((batch_size, dim))
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            noise = rng.standard_normal((batch_size, dim))
            noise /= np.linalg.norm(noise, axis=1, keepdims=True)
            Q = X + r * noise
            a = rng.standard_normal((batch_size, dim))
            proj_x = np.einsum('ij,ij->i', a, X)
            proj_q = np.einsum('ij,ij->i', a, Q)
            bit_x = np.abs(proj_x) <= z
            bit_q = np.abs(proj_q) <= z
            matches += int(np.sum(bit_x == bit_q))
            total += batch_size
            p_hat = matches / total
            se = math.sqrt(p_hat * (1.0 - p_hat) / total)
            if se <= target_se:
                break
        return float(p_hat)

    def _calibrate(
        self, rng: np.random.Generator, dim: int, tau: float, r_far: float,
        num_projections_per_table: int, calibration_grid: int,
    ) -> Tuple[float, float, float]:
        """Grid-search the window half-width z maximizing p(tau;z) - p(r_far;z)
        for a single projection, then raise the per-projection probabilities
        to num_projections_per_table (AND composition within a table)."""
        z_candidates = np.linspace(0.1 * tau, 3.0 * r_far, calibration_grid)
        best_z, best_gap, best_p_tau = float(z_candidates[0]), -1.0, 0.0
        for z in z_candidates:
            p_tau = self._match_probability(rng, dim, tau, float(z))
            p_far = self._match_probability(rng, dim, r_far, float(z))
            gap = p_tau - p_far
            if gap > best_gap:
                best_z, best_gap, best_p_tau = float(z), gap, p_tau
        p_table_tau = best_p_tau ** num_projections_per_table
        return best_z, best_gap, p_table_tau

    # ------------------------------------------------------------------ hashing / storage

    def _codes(self, vectors: np.ndarray) -> np.ndarray:
        """(n, num_tables) opaque per-table signature codes: each table's code
        is the tuple of num_projections_per_table window-membership bits."""
        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        # projections[t, k, :] . vecs[n, :] -> (n, num_tables, K)
        proj = np.einsum('nd,tkd->ntk', vecs, self._projections)
        bits = (np.abs(proj) <= self.z)  # (n, num_tables, K) bool
        packed = np.packbits(bits, axis=-1)  # (n, num_tables, ceil(K/8))
        contiguous = np.ascontiguousarray(packed)
        void_dt = np.dtype((np.void, contiguous.shape[-1]))
        return contiguous.view(void_dt).reshape(vecs.shape[0], self.num_tables)

    def _insert_batch(self, vectors: np.ndarray) -> None:
        if len(vectors) == 0:
            return
        codes = self._codes(vectors)
        for row in codes:
            for t in range(self.num_tables):
                key = signature_to_key((t, row[t].tobytes()))
                self.filter.add(key)

    # ------------------------------------------------------------------ Filter interface

    def construct(self, vectors: np.ndarray) -> None:
        capacity = (
            self._capacity if self.max_memory is not None
            else max(self._capacity, len(vectors) * self.num_tables)
        )
        self.filter = _BloomFilter(capacity=capacity, error_rate=self._error_rate)
        self._insert_batch(vectors)

    def query(self, vectors: np.ndarray, tau: Optional[float] = None) -> List[bool]:
        if self.filter is None:
            raise ValueError("DSHFilter has not been constructed.")
        if tau is not None and abs(tau - self.distance_threshold) > 1e-9:
            raise ValueError(
                f"DSHFilter bakes tau into construction (distance_threshold="
                f"{self.distance_threshold}); query-time tau={tau} is not supported."
            )
        if len(vectors) == 0:
            return []
        codes = self._codes(vectors)
        results = []
        for row in codes:
            hit = any(
                signature_to_key((t, row[t].tobytes())) in self.filter
                for t in range(self.num_tables)
            )
            results.append(hit)
        return results

    def insert(self, vectors: np.ndarray) -> None:
        if self.filter is None:
            self.construct(vectors)
            return
        self._insert_batch(vectors)

    def get_bytesize(self) -> int:
        proj_bytes = int(self._projections.nbytes)
        if self.filter is None:
            return proj_bytes
        bits_needed = -self.filter.capacity * math.log(self._error_rate) / (math.log(2) ** 2)
        bloom_bytes = int(math.ceil(bits_needed / 8))
        return bloom_bytes + proj_bytes
