"""DSHFilter -- Distance-Sensitive Hashing (Aumuller, Christiani, Pagh &
Silvestri, PODS 2018), instantiated as h(x) = 1 iff <a,x> in [-z, z] for
Gaussian a (a window centered at 0, since this only needs sensitivity to a
ball around the query, not the paper's off-origin annulus), concatenated
`num_projections_per_table` times per table across `num_tables` tables.

Calibration has two free parameters: the window half-width z, and
`num_projections_per_table` (K), the number of projections AND-ed together
within a table. K is the real accuracy/memory dial -- larger K shrinks the
false-accept rate exponentially at a fixed z, at the cost of needing more
OR-tables (`num_tables`, auto-sized below) to hold recall up, i.e. more
bytes. Leaving K fixed (as earlier versions did, calibrating only z) means
memory only ever buys a bigger Bloom filter, whose own ~1% engineered error
rate is swamped by K=4's structural false-accept rate on real, non-uniform
embeddings -- runner.py's equal-memory sweep now grid-searches (K, z)
jointly for exactly this reason (see its _CONSTRUCTION_TIME_CONFIG_DIALS
entry for DSHFilter).

By default (z=None) z is chosen by maximizing p(tau; z) - p(r_far; z)
(r_far defaults to sqrt(2), the expected distance between independent
unit-sphere points) under a synthetic model: X uniform on the unit sphere,
Q at distance r from X in a random direction. That model is a real-data
*proxy*, not real data -- real embeddings cluster far more tightly than a
uniform sphere (see BallCoverFilter's docstring), so it can misjudge where
the best z is. HNSWFilter/IVFFilter's query-time dials (ef_search/nprobe)
are instead grid-searched empirically against real calibration data in
runner._build_calibrated_filter; z cannot be tuned that way post-hoc
because it is baked into every inserted point's hash codes, not a
query-time knob. `z_grid()` exposes the same candidate grid this synthetic
search would sweep, so a caller with real labeled calibration data (as
runner.py's equal-memory sweep has) can instead build one DSHFilter per
candidate z and pick whichever scores best on that real data -- the same
empirical-selection treatment IVF/HNSW get, applied at construction time
since that is the only time DSH's z can move.

If the best achievable gap falls below `calibration_min_gap`, the
constructor raises RuntimeError rather than silently building a
non-functional filter."""

import math
from typing import List, Optional, Tuple

import numpy as np
from pybloom_live import BloomFilter as _BloomFilter

from filters.filter import Filter, InsufficientMemoryError
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
        seed: int = 42,
        z: Optional[float] = None,
    ):
        super().__init__(dim, max_memory=max_memory)
        self.distance_threshold = float(distance_threshold)
        self.num_projections_per_table = num_projections_per_table
        self._error_rate = error_rate
        # Real embeddings cluster much more tightly than a uniform-random
        # unit sphere, so a fixed sqrt(2) reference can leave no calibration
        # margin -- scale off tau itself instead.
        self.far_reference_distance = (
            far_reference_distance if far_reference_distance is not None
            else max(math.sqrt(2.0), 2.5 * self.distance_threshold)
        )

        rng = np.random.default_rng(seed)

        if z is None:
            self.z, best_gap, self._p_table_tau = self._calibrate(
                rng, dim, self.distance_threshold, self.far_reference_distance,
                num_projections_per_table, calibration_grid,
            )
        else:
            # z chosen by the caller (e.g. an empirical grid-search against
            # real calibration data, see z_grid()'s docstring) rather than
            # this file's synthetic sphere model. p_table_tau still has to be
            # measured at that z to size num_tables below, and the gap check
            # still applies: a badly-chosen z (from any source) that cannot
            # separate tau from far_reference_distance would otherwise size
            # num_tables absurdly large or raise a confusing error later.
            self.z = float(z)
            p_tau = self._match_probability(rng, dim, self.distance_threshold, self.z)
            p_far = self._match_probability(rng, dim, self.far_reference_distance, self.z)
            best_gap = p_tau - p_far
            self._p_table_tau = p_tau ** num_projections_per_table
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

        proj_bytes = int(self._projections.nbytes)
        if max_memory is not None:
            # The projection matrix is part of the filter and get_bytesize()
            # counts it, so only what is left of the budget may go to the
            # Bloom filter. Sizing the Bloom against the *whole* budget put
            # every DSH point over its budget by exactly proj_bytes -- small
            # in relative terms, but it is the one thing an equal-memory
            # comparison may not get wrong.
            bits_available = max(0, max_memory - proj_bytes) * 8
            self._capacity = max(
                1, int(bits_available * (math.log(2) ** 2) / (-math.log(error_rate)))
            )
            total_bytes = proj_bytes + self._bloom_bytes_for(self._capacity, error_rate)
            if total_bytes > max_memory:
                raise InsufficientMemoryError(
                    f"max_memory={max_memory} bytes is too small for DSHFilter: its "
                    f"{self.num_tables}x{num_projections_per_table}x{dim} float32 projection "
                    f"matrix alone needs {proj_bytes} bytes, leaving too little for even a "
                    f"minimum-capacity Bloom filter ({total_bytes} bytes needed in total)."
                )
        else:
            self._capacity = 1_000 * self.num_tables

        self.filter: Optional[_BloomFilter] = None


    @staticmethod
    def _match_probability(
        rng: np.random.Generator, dim: int, r: float, z: float,
        batch_size: int = 8_000, max_sims: int = 40_000, target_se: float = 0.004,
    ) -> float:
        """Monte Carlo estimate of P(bit(X) == bit(Q)) for a single Gaussian
        projection with acceptance window [-z, z], X a random unit vector and
        Q at distance r from X in a random direction -- batched with a
        target-standard-error stopping rule."""
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

    @staticmethod
    def z_grid(
        distance_threshold: float,
        far_reference_distance: Optional[float] = None,
        calibration_grid: int = 13,
    ) -> np.ndarray:
        """The same bounded candidate grid the synthetic-model calibration
        (`_calibrate`) would search, exposed so a caller with real labeled
        calibration data can instead build one DSHFilter(z=candidate) per
        entry and pick whichever scores best empirically (see module
        docstring) -- rather than duplicating this formula at the call site."""
        tau = float(distance_threshold)
        r_far = (
            far_reference_distance if far_reference_distance is not None
            else max(math.sqrt(2.0), 2.5 * tau)
        )
        return np.linspace(0.1 * tau, 3.0 * r_far, calibration_grid)

    def _calibrate(
        self, rng: np.random.Generator, dim: int, tau: float, r_far: float,
        num_projections_per_table: int, calibration_grid: int,
    ) -> Tuple[float, float, float]:
        """Grid-search the window half-width z maximizing p(tau;z) - p(r_far;z)
        for a single projection, then raise the per-projection probabilities
        to num_projections_per_table (AND composition within a table)."""
        z_candidates = self.z_grid(tau, r_far, calibration_grid)
        best_z, best_gap, best_p_tau = float(z_candidates[0]), -1.0, 0.0
        for z in z_candidates:
            p_tau = self._match_probability(rng, dim, tau, float(z))
            p_far = self._match_probability(rng, dim, r_far, float(z))
            gap = p_tau - p_far
            if gap > best_gap:
                best_z, best_gap, best_p_tau = float(z), gap, p_tau
        p_table_tau = best_p_tau ** num_projections_per_table
        return best_z, best_gap, p_table_tau


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


    def construct(self, vectors: np.ndarray) -> None:
        capacity = (
            self._capacity if self.max_memory is not None
            else max(self._capacity, len(vectors) * self.num_tables)
        )
        self.filter = _BloomFilter(capacity=capacity, error_rate=self._error_rate)
        try:
            self._insert_batch(vectors)
        except IndexError as e:
            # pybloom_live raises a bare IndexError once its bit array fills
            # up. That happens here when max_memory bounds self._capacity
            # below the len(vectors) * num_tables distinct (table, code)
            # entries this construction actually inserts -- a memory-budget
            # infeasibility like any other, not a crash. Re-raise as the
            # shared InsufficientMemoryError so _build_calibrated_filter's
            # per-candidate except clause skips this dial candidate instead
            # of the raw IndexError propagating out and failing the whole
            # equal-memory worker job.
            raise InsufficientMemoryError(
                f"DSHFilter's Bloom filter (capacity={capacity}, "
                f"error_rate={self._error_rate}) filled up while inserting "
                f"{len(vectors)} vectors x {self.num_tables} tables at "
                f"num_projections_per_table={self.num_projections_per_table}: "
                f"max_memory={self.max_memory} is too small for this many "
                f"distinct (table, code) entries."
            ) from e

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

    @staticmethod
    def _bloom_bytes_for(capacity: int, error_rate: float) -> int:
        """Bytes pybloom_live's bit array needs at this capacity and error
        rate. Shared by the max_memory sizing in __init__ and by
        get_bytesize() so the budget the filter is built to and the size it
        reports cannot drift apart."""
        bits_needed = -capacity * math.log(error_rate) / (math.log(2) ** 2)
        return int(math.ceil(bits_needed / 8))

    def get_bytesize(self) -> int:
        proj_bytes = int(self._projections.nbytes)
        if self.filter is None:
            return proj_bytes
        return self._bloom_bytes_for(self.filter.capacity, self._error_rate) + proj_bytes
