"""RaceFilter -- RACE (Repeated Array-of-Counts Estimator) sketch, adapted
from Coleman & Shrivastava's KDE estimator (arXiv:1912.02283) into a
membership filter.

Unlike the bit-based LSH filters here (LSBFFilter/DSHFilter), RACE stores
per-bucket *counts*, so the query decision is a signal-to-noise problem --
does the true neighbor's expected contribution exceed the background counts
-- rather than a vote threshold (see _calibrate_theta). theta is calibrated
empirically on the real, already-populated counts array: a fixed-reference-
distance estimate was found to push FPR to ~98% (background is dominated by
points landing unusually close by chance, not by the mean).

k_theory (see _k_theory) is a data-dependent, worst-case-optimal K from a
follow-up paper (arXiv:1902.06687, Theorem 5), exposed as a diagnostic only
-- at realistic settings it comes out to K~18-21, impractical as a hard
requirement, and even the paper calls this bound overly pessimistic for
non-adversarial data."""

import math
from typing import List, Optional, Tuple

import numpy as np

from filters.filter import Filter


class RaceFilter(Filter):
    """RACE sketch (see module docstring): thresholds a median-of-means
    density estimate against a calibrated theta.

    tau is baked in at construction via `distance_threshold` (theta is
    calibrated around it and re-calibrated on every add()/insert(), since it
    depends on the current construct-set size); `query(..., tau=t)` only
    accepts t == distance_threshold or t=None.
    """

    _COUNTER_DTYPE = np.uint32

    def __init__(
        self,
        dim: int,
        distance_threshold: float,
        max_memory: Optional[int] = None,
        hashes_per_table: int = 8,
        num_tables: Optional[int] = None,
        num_means: Optional[int] = None,
        far_reference_distance: Optional[float] = None,
        calibration_min_snr: float = 3.0,
        calibration_probes: int = 200,
        max_spheres: Optional[int] = None,
        seed: int = 42,
    ):
        super().__init__(dim, max_memory=max_memory)
        if not (1 <= hashes_per_table <= 20):
            raise ValueError(
                f"hashes_per_table={hashes_per_table} must be in [1, 20] "
                "(table_range = 2**hashes_per_table)."
            )
        self.distance_threshold = float(distance_threshold)
        self.hashes_per_table = hashes_per_table
        self.table_range = 1 << hashes_per_table
        self._max_spheres = max_spheres  # unused; accepted for _instantiate_filter parity
        # Same reasoning as DSHFilter's far_reference_distance: scale off tau, not a fixed sqrt(2).
        self.far_reference_distance = (
            far_reference_distance if far_reference_distance is not None
            else max(math.sqrt(2.0), 2.5 * self.distance_threshold)
        )
        self.calibration_min_snr = calibration_min_snr
        self.calibration_probes = calibration_probes
        self._seed = seed

        counter_bytes = np.dtype(self._COUNTER_DTYPE).itemsize
        proj_bytes_per_table = hashes_per_table * dim * 4  # float32 projections
        bytes_per_table = self.table_range * counter_bytes + proj_bytes_per_table
        if num_tables is not None:
            self.num_tables = num_tables
        elif max_memory is not None:
            self.num_tables = max(1, max_memory // bytes_per_table)
        else:
            self.num_tables = 32

        # Trades off mean-vs-median aggregation; see _aggregate.
        self.num_means = num_means if num_means is not None else 1
        if not (1 <= self.num_means <= self.num_tables):
            raise ValueError(
                f"num_means={self.num_means} must be in [1, num_tables={self.num_tables}]"
            )

        rng = np.random.default_rng(seed)
        self._projections = rng.standard_normal(
            (self.num_tables, hashes_per_table, dim)
        ).astype(np.float32)
        self._bit_weights = (1 << np.arange(hashes_per_table)).astype(np.int64)

        self._counts: Optional[np.ndarray] = None
        self.theta: Optional[float] = None
        self.k_theory: Optional[int] = None  # diagnostic only -- see _k_theory
        self._n_inserted = 0

    @staticmethod
    def _base_collision_probability(r: float) -> float:
        """SRP/SimHash single-hyperplane collision probability for two
        points at Euclidean distance r on the unit sphere:
        1 - arccos(1 - r^2/2)/pi. Matches the closed form DSBFFilter/LSBFFilter cite."""
        cos_theta = float(np.clip(1.0 - (r ** 2) / 2.0, -1.0, 1.0))
        return 1.0 - math.acos(cos_theta) / math.pi

    def _k_theory(self, n: int) -> int:
        """Theorem 5 (arXiv:1902.06687): worst-case-sufficient amplification
        depth K = ceil(2 log(B) / log(1/Delta)). Diagnostic only
        (self.k_theory), never gates hashes_per_table -- at realistic
        settings K~18-21, needing table_range=2**20+."""
        if n <= 1:
            return 1
        p_near = self._base_collision_probability(self.distance_threshold)
        p_far = self._base_collision_probability(self.far_reference_distance)
        if p_far <= 0.0 or p_near <= 0.0:
            return 1
        delta = p_far / p_near
        if delta <= 0.0 or delta >= 1.0:
            return 1
        return max(1, math.ceil(2.0 * math.log(n) / math.log(1.0 / delta)))

    def _table_indices(self, vectors: np.ndarray) -> np.ndarray:
        """(n, num_tables) integer bucket index per vector, per table, in
        [0, table_range) -- each table's index is `hashes_per_table`
        independent hyperplane sign bits read as a binary number."""
        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        proj = np.einsum('nd,tkd->ntk', vecs, self._projections)  # (n, num_tables, K)
        bits = (proj > 0).astype(np.int64)
        return (bits * self._bit_weights).sum(axis=-1)  # (n, num_tables)

    def _insert_batch(self, vectors: np.ndarray) -> None:
        if len(vectors) == 0:
            return
        idx = self._table_indices(vectors)  # (n, num_tables)
        for l in range(self.num_tables):
            counts_l = np.bincount(idx[:, l], minlength=self.table_range)
            self._counts[l] += counts_l[: self.table_range].astype(self._COUNTER_DTYPE)
        self._n_inserted += len(vectors)

    def _aggregate(self, counts_per_table: np.ndarray) -> np.ndarray:
        """(n, num_tables) -> (n,) median-of-means: average within each of
        num_means groups, then median across groups. num_means=1 (default)
        reduces to a plain mean; num_means=num_tables is an ordinary
        median."""
        usable = (self.num_tables // self.num_means) * self.num_means
        reshaped = counts_per_table[:, :usable].reshape(
            counts_per_table.shape[0], self.num_means, -1
        )
        group_means = reshaped.mean(axis=2)
        return np.median(group_means, axis=1)

    def _calibrate_theta(self, vectors: np.ndarray, seed: int) -> float:
        """Empirical signal-vs-background calibration on the real,
        already-populated counts array (see module docstring): near-side
        probes from real construct vectors, far-side probes from random unit
        vectors, both through the same _table_indices/_aggregate path
        query() uses, theta set at the midpoint between their means. Raises
        RuntimeError if SNR falls below calibration_min_snr."""
        n = len(vectors)
        if n <= 0:
            return float('inf')

        rng = np.random.default_rng(seed)
        n_probes = min(n, self.calibration_probes)
        near_idx = rng.choice(n, size=n_probes, replace=False)
        near_probes = np.ascontiguousarray(vectors[near_idx], dtype=np.float32)
        far_probes = rng.standard_normal((n_probes, self.dim)).astype(np.float32)
        far_probes /= np.linalg.norm(far_probes, axis=1, keepdims=True)

        table_ids = np.arange(self.num_tables)[None, :]
        near_counts = self._counts[table_ids, self._table_indices(near_probes)].astype(np.float64)
        far_counts = self._counts[table_ids, self._table_indices(far_probes)].astype(np.float64)
        near_agg = self._aggregate(near_counts)
        far_agg = self._aggregate(far_counts)

        mu0, mu1 = float(far_agg.mean()), float(near_agg.mean())
        sigma0 = float(far_agg.std())
        snr = (mu1 - mu0) / max(sigma0, 1e-9)

        if snr < self.calibration_min_snr:
            raise RuntimeError(
                f"RaceFilter calibration did not converge to a usable operating point: "
                f"estimated SNR={snr:.3f} (background mean={mu0:.3f}, std={sigma0:.3f}, "
                f"near-probe mean={mu1:.3f}) at n={n} construct vectors is below "
                f"calibration_min_snr={self.calibration_min_snr} "
                f"(dim={self.dim}, distance_threshold={self.distance_threshold}, "
                f"hashes_per_table={self.hashes_per_table}, num_tables={self.num_tables}). "
                f"Try more num_tables, fewer hashes_per_table, or accept a lower "
                f"calibration_min_snr."
            )
        return mu0 + 0.5 * (mu1 - mu0)

    def get_exact_matching_fpr(self) -> float:
        # No structural collision-free layer beyond the calibrated threshold itself.
        return 0.0

    def get_criteria(self, n: int, r1: float, r2: float) -> Tuple[float, float]:
        """Approximate (FPR, FNR) via a CLT/Gaussian approximation of the
        median-of-means aggregate's sampling distribution (mu0 = background
        at n points, mu1 = background plus one true r1-neighbor). Approximate
        only; the comparison harness measures FPR/FNR empirically instead."""
        from scipy.stats import norm

        n = max(n, 0)
        p_r1_table = self._base_collision_probability(r1) ** self.hashes_per_table
        p_r2_table = self._base_collision_probability(r2) ** self.hashes_per_table
        mu0 = n * p_r2_table
        mu1 = mu0 + p_r1_table
        sigma = math.sqrt(max(n * p_r2_table * (1.0 - p_r2_table), 1e-12) / max(self.num_means, 1))
        theta = self.theta if self.theta is not None else mu0 + 0.5 * p_r1_table

        fpr = float(norm.sf((theta - mu0) / max(sigma, 1e-12)))
        fnr = float(norm.cdf((theta - mu1) / max(sigma, 1e-12)))
        return fpr, fnr

    def construct(self, vectors: np.ndarray) -> None:
        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        self.k_theory = self._k_theory(len(vecs))  # diagnostic only, see _k_theory
        self._counts = np.zeros((self.num_tables, self.table_range), dtype=self._COUNTER_DTYPE)
        self._n_inserted = 0
        self._insert_batch(vecs)
        self.theta = self._calibrate_theta(vecs, seed=self._seed)

    def query(self, vectors: np.ndarray, tau: Optional[float] = None) -> List[bool]:
        if self._counts is None:
            raise ValueError("RaceFilter has not been constructed.")
        if tau is not None and abs(tau - self.distance_threshold) > 1e-9:
            raise ValueError(
                f"RaceFilter bakes tau into construction (distance_threshold="
                f"{self.distance_threshold}); query-time tau={tau} is not supported."
            )
        if len(vectors) == 0:
            return []
        idx = self._table_indices(vectors)  # (n, num_tables)
        table_ids = np.arange(self.num_tables)[None, :]
        counts_per_table = self._counts[table_ids, idx]  # (n, num_tables)
        estimates = self._aggregate(counts_per_table.astype(np.float64))
        return (estimates >= self.theta).tolist()

    def add(self, vectors: np.ndarray) -> None:
        if self._counts is None:
            self.construct(vectors)
            return
        if len(vectors) == 0:
            return
        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        self._insert_batch(vecs)
        self.k_theory = self._k_theory(self._n_inserted)  # diagnostic only, see _k_theory
        # The newly-added batch qualifies as near-probes just as well as construct()'s did.
        self.theta = self._calibrate_theta(vecs, seed=self._seed)

    def insert(self, vectors: np.ndarray) -> None:
        self.add(vectors)

    def get_bytesize(self) -> int:
        counts_bytes = (
            int(self._counts.nbytes) if self._counts is not None
            else self.num_tables * self.table_range * np.dtype(self._COUNTER_DTYPE).itemsize
        )
        return counts_bytes + int(self._projections.nbytes)
