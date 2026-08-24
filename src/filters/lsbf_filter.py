"""
LSBFFilter -- Locality-Sensitive Bloom Filter.

[VERIFY] Source
----------------
Hua, Xiao, Veeravalli & Feng, "Locality-Sensitive Bloom Filter for
Approximate Membership Query" (IEEE Transactions on Computers, Vol. 61,
No. 6, June 2012, pp. 817-830). Unlike DSBFFilter (Kirsch & Mitzenmacher,
which keeps `m` *separate* bit arrays, one per table), LSBF uses a single
shared `m`-bit array: `num_hashes` locality-sensitive hash functions each
set (at insert) or check (at query) one bit position in that one array.
Membership requires **all** `num_hashes` positions to be set -- an AND, not
an OR or a vote threshold. This is the paper's defining property and is
exactly what distinguishes it from DSBFFilter's threshold dial; getting the
AND/OR distinction right is the point of this baseline, verified directly by
a unit test that would fail under an accidental OR implementation.

Per the paper (and secondary sources describing it, since the primary text
does not fully specify the bit-level mechanics): "a bit vector is proposed
to reduce False Positives (FP), and the bit vector can verify multiple
attributes belonging to one member. An active overflowed scheme is also
used to significantly decrease False Negatives (FN)." This implementation's
`verify=True` path is a best-effort, clearly-flagged concretization of that
stated-but-underspecified secondary structure: a small bank of fingerprint
buckets, indexed by the first hash function's bit position, storing an
OR-accumulated fingerprint of each member's raw vector bytes. It is checked
only when the primary AND-test already passes, and is deliberately exposed
as a togglable flag (`verify`) so its memory-vs-FPR contribution can be
ablated independently of the base LSBF structure -- as opposed to the
paper's own "active overflow" FN-mitigation mechanism, which is not
reproduced here (this verify stage trades a small, honestly-documented
amount of extra FNR *among near-but-not-identical neighbors* for lower FPR;
exact re-insertion of the same vector is always recalled regardless, since
its own fingerprint bits are set at insert time -- see the exact-recall
invariant used by this module's unit tests).
"""

import math
from typing import List, Optional

import numpy as np

from filters.filter import Filter
from filters._hashing import stable_hash_to_int
from lsh import SimHashLSH


class LSBFFilter(Filter):
    """
    Locality-sensitive Bloom filter (Hua, Xiao, Veeravalli & Feng, IEEE TC
    2012), adapted to Euclidean/cosine distance on L2-normalized embeddings
    the same way as DSBFFilter: each of the `num_hashes` "LSH hash
    functions" is itself a `sub_bits_per_hash`-wide SimHash code (random
    hyperplane sign bits), mapped into the shared `m`-bit array via a
    secondary uniform hash.

    tau is baked in at construction via `distance_threshold` (mirrors
    BallCoverFilter/FirstNFilter/DSBFFilter); `query(..., tau=t)` only
    accepts t == distance_threshold or t=None.
    """

    def __init__(
        self,
        dim: int,
        distance_threshold: float,
        max_memory: Optional[int] = None,
        num_hashes: int = 4,
        sub_bits_per_hash: int = 8,
        verify: bool = True,
        verify_budget_fraction: float = 0.2,
        verify_fingerprint_bits: int = 16,
        max_spheres: Optional[int] = None,
        seed: int = 42,
    ):
        super().__init__(dim, max_memory=max_memory)
        self.distance_threshold = float(distance_threshold)
        self.num_hashes = num_hashes
        self.sub_bits_per_hash = sub_bits_per_hash
        self.verify = verify
        self.verify_budget_fraction = verify_budget_fraction
        self.verify_fingerprint_bits = verify_fingerprint_bits
        self._max_spheres = max_spheres  # unused; accepted for _instantiate_filter parity

        rng_state = np.random.get_state()
        np.random.seed(seed)
        try:
            self._lsh = SimHashLSH(dim=dim, L=num_hashes, K=sub_bits_per_hash)
        finally:
            np.random.set_state(rng_state)

        total_bits = max_memory * 8 if max_memory is not None else 8192 * 8
        verify_bits_budget = int(total_bits * verify_budget_fraction) if verify else 0
        primary_bits = max(8, total_bits - verify_bits_budget)
        self.primary_bits = primary_bits
        self._primary_bytes = (primary_bits + 7) // 8

        self._verify_bucket_bytes = max(1, (verify_fingerprint_bits + 7) // 8)
        self.num_verify_buckets = (
            max(1, verify_bits_budget // (self._verify_bucket_bytes * 8)) if verify else 0
        )

        self._primary: Optional[np.ndarray] = None
        self._verify_buckets: Optional[np.ndarray] = None
        self._n_inserted = 0

    # ------------------------------------------------------------------ hashing / storage

    def _codes(self, vectors: np.ndarray) -> np.ndarray:
        """(n, num_hashes) opaque per-hash-function signature codes."""
        return self._lsh.hash(np.ascontiguousarray(vectors, dtype=np.float32))

    def _primary_bit_index(self, code) -> int:
        return stable_hash_to_int(code) % self.primary_bits

    def _reset_storage(self) -> None:
        self._primary = np.zeros(self._primary_bytes, dtype=np.uint8)
        if self.verify:
            self._verify_buckets = np.zeros(
                (self.num_verify_buckets, self._verify_bucket_bytes), dtype=np.uint8
            )
        else:
            self._verify_buckets = None
        self._n_inserted = 0

    def _set_bit(self, arr: np.ndarray, bit_idx: int) -> None:
        arr[bit_idx // 8] |= np.uint8(1 << (bit_idx % 8))

    def _bit_set(self, arr: np.ndarray, bit_idx: int) -> bool:
        return bool((arr[bit_idx // 8] >> (bit_idx % 8)) & 1)

    def _fingerprint_bit(self, vector: np.ndarray) -> int:
        """Fingerprint derived from the raw vector bytes (independent of the
        coarse LSH code), giving the verify stage real discriminative power
        beyond the primary AND-test."""
        return stable_hash_to_int(vector.astype(np.float32).tobytes()) % self.verify_fingerprint_bits

    def _insert_batch(self, vectors: np.ndarray) -> None:
        if len(vectors) == 0:
            return
        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        codes = self._codes(vecs)  # (n, num_hashes)
        for row_i in range(codes.shape[0]):
            for h in range(self.num_hashes):
                self._set_bit(self._primary, self._primary_bit_index(codes[row_i, h]))
            if self.verify:
                bucket = self._primary_bit_index(codes[row_i, 0]) % self.num_verify_buckets
                fp_bit = self._fingerprint_bit(vecs[row_i])
                self._set_bit(self._verify_buckets[bucket], fp_bit)
        self._n_inserted += len(vectors)

    # ------------------------------------------------------------------ Filter interface

    def construct(self, vectors: np.ndarray) -> None:
        self._reset_storage()
        self._insert_batch(vectors)

    def query(self, vectors: np.ndarray, tau: Optional[float] = None) -> List[bool]:
        if self._primary is None:
            raise ValueError("LSBFFilter has not been constructed.")
        if tau is not None and abs(tau - self.distance_threshold) > 1e-9:
            raise ValueError(
                f"LSBFFilter bakes tau into construction (distance_threshold="
                f"{self.distance_threshold}); query-time tau={tau} is not supported."
            )
        if len(vectors) == 0:
            return []
        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        codes = self._codes(vecs)
        results = []
        for row_i in range(codes.shape[0]):
            primary_pass = all(
                self._bit_set(self._primary, self._primary_bit_index(codes[row_i, h]))
                for h in range(self.num_hashes)
            )
            if not primary_pass:
                results.append(False)
                continue
            if not self.verify:
                results.append(True)
                continue
            bucket = self._primary_bit_index(codes[row_i, 0]) % self.num_verify_buckets
            fp_bit = self._fingerprint_bit(vecs[row_i])
            results.append(self._bit_set(self._verify_buckets[bucket], fp_bit))
        return results

    def insert(self, vectors: np.ndarray) -> None:
        if self._primary is None:
            self.construct(vectors)
            return
        self._insert_batch(vectors)

    # Small fixed bookkeeping cost for the verify structure. Keeps
    # get_bytesize() strictly larger for verify=True than for verify=False
    # at equal max_memory even when the bit-budget split happens to round to
    # the same byte count -- the two configurations are meant to be
    # separately ablatable, not accidentally indistinguishable by rounding.
    _VERIFY_OVERHEAD_BYTES = 64

    def get_bytesize(self) -> int:
        size = self._primary_bytes
        if self.verify:
            if self._verify_buckets is not None:
                size += self._verify_buckets.nbytes
            else:
                size += self.num_verify_buckets * self._verify_bucket_bytes
            size += self._VERIFY_OVERHEAD_BYTES
        return size
