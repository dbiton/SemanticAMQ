"""
DSBFFilter -- Distance-Sensitive Bloom Filter.

[VERIFY] Source and Hamming-to-Euclidean adaptation
-------------------------------------------------------
Kirsch & Mitzenmacher, "Distance-Sensitive Bloom Filters" (ALENEX 2006,
pp. 41-50). The paper's own construction: `m` disjoint bit arrays of `m'`
bits each; array i is indexed by a hash g_i formed by concatenating `L`
locality-sensitive hash functions (so a vector's bit in array i is only set
if all `L` sub-hashes agree with the LSH bucket the array was built for).
Insert(x) sets bit g_i(x) in every array i. Query(q) counts how many of the
`m` arrays have their bit g_i(q) set and answers YES iff that count is
>= a threshold `t <= m`. This is explicitly a two-sided-error structure --
`t` trades false positives against false negatives (`t=1` is maximally
permissive/OR-like across arrays; `t=m` is maximally strict/AND-like).

The paper's own analysis is for the Hamming metric. This codebase works
exclusively with L2-normalized (unit-sphere) real embeddings at d=384, so
this port substitutes the Euclidean/cosine analogue of Hamming-bit hashing
that the rest of this codebase already uses: random-hyperplane SimHash
(`src/lsh.py::SimHashLSH`), whose sign bit is the direct continuous-space
counterpart of a Hamming coordinate, and whose closed-form collision
probability `_get_base_probabilities` (the standard `1 - arccos(1-r^2/2)/pi`
formula relating Euclidean distance on the unit sphere to hyperplane-sign
agreement probability) is reused unchanged for `DSBFFilter.get_criteria`.
The paper's own `m`/`m'`/`L`/`t` notation is renamed here to `num_tables`
(m), `bits_per_table` (m'), `hashes_per_table` (L), `vote_threshold` (t) to
avoid clashing with this codebase's own L/K naming for LSH tables/sub-hashes.

Correctness note used by the mandatory unit tests: because Insert(x) always
sets one bit in *every* one of the `num_tables` arrays for x itself,
re-querying the exact same x always yields count == num_tables, which is
>= vote_threshold for any valid threshold -- an exact-recall invariant that
holds regardless of the (probabilistic, two-sided) near-neighbor FPR/FNR.
"""

import math
from typing import List, Optional

import numpy as np

from filters.filter import Filter
from filters._hashing import stable_hash_to_int
from lsh import SimHashLSH


class DSBFFilter(Filter):
    """
    Distance-sensitive Bloom filter (Kirsch & Mitzenmacher, ALENEX 2006),
    adapted to Euclidean/cosine distance on L2-normalized embeddings via
    random-hyperplane SimHash (see module docstring for the [VERIFY] note).

    tau is baked in at construction via `distance_threshold` (the SimHash
    table count/width is calibrated around it); `query(..., tau=t)` only
    accepts t == distance_threshold or t=None.
    """

    def __init__(
        self,
        dim: int,
        distance_threshold: float,
        max_memory: Optional[int] = None,
        num_tables: Optional[int] = None,
        hashes_per_table: int = 4,
        vote_threshold: Optional[int] = None,
        max_L: int = 256,
        max_spheres: Optional[int] = None,
        seed: int = 42,
    ):
        super().__init__(dim, lsh=None, max_memory=max_memory)
        self.distance_threshold = float(distance_threshold)
        self.hashes_per_table = hashes_per_table
        self._max_spheres = max_spheres  # unused; accepted for _instantiate_filter parity

        # SimHashLSH draws its hyperplanes from the numpy global RNG (no seed
        # param of its own) -- seed deterministically without permanently
        # disturbing the caller's global random state.
        rng_state = np.random.get_state()
        np.random.seed(seed)
        try:
            if num_tables is None:
                self._lsh = SimHashLSH.build_optimized(
                    dim, distance_threshold, K=hashes_per_table, max_L=max_L
                )
            else:
                self._lsh = SimHashLSH(dim=dim, L=num_tables, K=hashes_per_table)
        finally:
            np.random.set_state(rng_state)

        self.num_tables = self._lsh.L
        self.vote_threshold = (
            vote_threshold if vote_threshold is not None else (self.num_tables + 1) // 2
        )
        if not (1 <= self.vote_threshold <= self.num_tables):
            raise ValueError(
                f"vote_threshold={self.vote_threshold} must be in [1, num_tables={self.num_tables}]"
            )

        if max_memory is not None:
            self.bits_per_table = max(8, (max_memory * 8) // self.num_tables)
        else:
            self.bits_per_table = 8192
        self._bytes_per_table = (self.bits_per_table + 7) // 8

        self._byte_arrays: Optional[List[np.ndarray]] = None
        self._n_inserted = 0

    # ------------------------------------------------------------------ hashing / storage

    def _codes(self, vectors: np.ndarray) -> np.ndarray:
        """(n, num_tables) opaque per-table signature codes."""
        return self._lsh.hash(np.ascontiguousarray(vectors, dtype=np.float32))

    def _reset_storage(self) -> None:
        self._byte_arrays = [
            np.zeros(self._bytes_per_table, dtype=np.uint8) for _ in range(self.num_tables)
        ]
        self._n_inserted = 0

    def _insert_batch(self, vectors: np.ndarray) -> None:
        if len(vectors) == 0:
            return
        codes = self._codes(vectors)  # (n, num_tables)
        for i in range(self.num_tables):
            arr = self._byte_arrays[i]
            for code in codes[:, i]:
                bit_idx = stable_hash_to_int(code) % self.bits_per_table
                arr[bit_idx // 8] |= np.uint8(1 << (bit_idx % 8))
        self._n_inserted += len(vectors)

    def _bit_set(self, table: int, code) -> bool:
        bit_idx = stable_hash_to_int(code) % self.bits_per_table
        byte = self._byte_arrays[table][bit_idx // 8]
        return bool((byte >> (bit_idx % 8)) & 1)

    # ------------------------------------------------------------------ Filter interface

    def get_exact_matching_fpr(self) -> float:
        """
        Structural (LSH-collision-free) false positive estimate: probability
        that >= vote_threshold of num_tables independent bit arrays have a
        uniformly-random bit already set, given each array's observed fill
        fraction. 0.0 before construction.
        """
        if self._byte_arrays is None or self._n_inserted == 0:
            return 0.0
        from scipy.stats import binom

        fills = [
            float(np.unpackbits(arr).sum()) / self.bits_per_table for arr in self._byte_arrays
        ]
        avg_fill = sum(fills) / len(fills)
        return float(binom.sf(self.vote_threshold - 1, self.num_tables, avg_fill))

    def get_criteria(self, n: int, r1: float, r2: float):
        """
        Approximate expected (FPR, FNR) from the SimHash collision model,
        composed as: base per-hyperplane probability -> per-table AND of
        `hashes_per_table` hyperplanes -> vote-threshold-of-`num_tables`
        binomial composition. Documented as approximate/secondary -- the
        comparison harness (runner.py) measures FPR/FNR empirically against
        ground truth and does not call this method.
        """
        from scipy.stats import binom

        p1_base, p2_base = self._lsh._get_base_probabilities(r1, r2)
        p1_table = p1_base ** self.hashes_per_table
        p2_table = p2_base ** self.hashes_per_table
        p1_vote = binom.sf(self.vote_threshold - 1, self.num_tables, p1_table)
        p2_vote = binom.sf(self.vote_threshold - 1, self.num_tables, p2_table)
        fnr = 1.0 - p1_vote
        fpr = 1.0 - (1.0 - p2_vote) ** max(n, 0)
        return float(fpr), float(fnr)

    def construct(self, vectors: np.ndarray) -> None:
        self._reset_storage()
        self._insert_batch(vectors)

    def query(self, vectors: np.ndarray, tau: Optional[float] = None) -> List[bool]:
        if self._byte_arrays is None:
            raise ValueError("DSBFFilter has not been constructed.")
        if tau is not None and abs(tau - self.distance_threshold) > 1e-9:
            raise ValueError(
                f"DSBFFilter bakes tau into construction (distance_threshold="
                f"{self.distance_threshold}); query-time tau={tau} is not supported."
            )
        if len(vectors) == 0:
            return []
        codes = self._codes(vectors)
        results = []
        for row in codes:
            count = sum(1 for i, code in enumerate(row) if self._bit_set(i, code))
            results.append(count >= self.vote_threshold)
        return results

    def add(self, vectors: np.ndarray) -> None:
        if self._byte_arrays is None:
            self.construct(vectors)
            return
        self._insert_batch(vectors)

    def insert(self, vectors: np.ndarray) -> None:
        self.add(vectors)

    def get_bytesize(self) -> int:
        return self.num_tables * self._bytes_per_table
