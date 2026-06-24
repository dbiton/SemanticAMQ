"""
Vacuum Filter implementation.

Based on the 2020 paper:
"Vacuum Filters: A Space-Efficient and Highly-Dynamic Alternative
 to Bloom and Cuckoo Filters" by Zhang, Chan, and Tan (2020).

Key properties:
- Supports insertion and deletion (DYNAMIC filter)
- Uses cuckoo hashing with "vacuuming" for better space utilization
- Higher load factor (~95%) than Cuckoo filters (~88%)
- Uses buckets with fingerprint storage and partial-key cuckoo hashing
"""

import hashlib
import struct
import sys
import numpy as np
from typing import List, Optional, Tuple
from filters.filter import Filter
from lsh import LSH


class VacuumFilter(Filter):
    """
    A Vacuum Filter: a dynamic approximate membership query structure.

    Vacuum filters extend Cuckoo filters with a "vacuuming" mechanism
    that evacuates and re-slots entries to maintain high load, achieving
    ~95% space utilization compared to ~88% for standard Cuckoo filters.

    This is a DYNAMIC filter – elements can be added, queried, and removed.

    Args:
        dim:              Dimensionality of the input vectors.
        lsh:              Pre-built LSH instance.
        capacity:         Maximum expected number of elements.
        fingerprint_size: Fingerprint size in bits (8, 16, or 32).
                          8 bits → ~0.4% FPR
                          16 bits → ~0.0015% FPR
        buckets_per_entry: Number of candidate buckets per entry (2, 3, or 4).
                           Larger values improve load factor at cost of more probes.
        max_memory:       Maximum memory usage in bytes. If set, capacity is
                          derived from the memory budget.
    """

    def __init__(
        self,
        dim: int,
        lsh: Optional[LSH] = None,
        capacity: int = 1_000,
        fingerprint_size: int = 8,
        buckets_per_entry: int = 2,
        max_memory: Optional[int] = None,
    ):
        super().__init__(dim, lsh, max_memory=max_memory)

        if max_memory is not None:
            load_factor = 0.95
            bits_available = max_memory * 8
            entries_possible = int(bits_available / fingerprint_size * load_factor)
            self._capacity = max(entries_possible, 1)
        else:
            self._capacity = max(capacity, 1)

        self._fingerprint_size = fingerprint_size
        self._buckets_per_entry = max(2, min(buckets_per_entry, 4))
        self._epsilon = 1.0 / (2 ** fingerprint_size)

        slots_per_bucket = 4
        self._num_buckets = max(
            2,
            int(self._capacity / (slots_per_bucket * 0.95)) + 1
        )

        self._buckets: List[List[Tuple[int, int]]] = [
            [] for _ in range(self._num_buckets)
        ]
        self._max_slots_per_bucket = slots_per_bucket
        self._size = 0
        self._max_kicks = 500

        # All string keys currently in the filter — needed to rebuild on expansion.
        self._all_keys: List[str] = []

    def get_exact_matching_fpr(self) -> float:
        """
        Returns the exact-matching FPR.

        FPR ≈ slots_per_bucket * buckets_per_entry / 2^fingerprint_size because
        each query probes buckets_per_entry buckets of slots_per_bucket entries each.
        """
        return float(self._epsilon * self._max_slots_per_bucket * self._buckets_per_entry)

    @staticmethod
    def _hash_to_int(key: str) -> int:
        """Deterministic 32-bit hash using SHA-256 (PYTHONHASHSEED-independent)."""
        digest = hashlib.sha256(key.encode()).digest()
        return struct.unpack('<I', digest[:4])[0]

    @staticmethod
    def _fingerprint(value: int, size: int) -> int:
        """Extract a non-zero fingerprint of `size` bits from a hash value."""
        mask = (1 << size) - 1
        fp = value & mask
        return fp if fp != 0 else 1

    @staticmethod
    def _alt_hash(fp: int, bucket: int, num_buckets: int) -> int:
        """
        Compute the alternate bucket for partial-key cuckoo hashing.

        alt_bucket = bucket XOR hash(fp) % num_buckets

        Guaranteed to differ from `bucket` so that each key always has two
        distinct candidate buckets and kick chains make progress.
        """
        h = fp ^ (fp << 16) ^ (fp >> 12)
        h = h & 0xFFFFFFFF
        delta = h % num_buckets
        if delta == 0:
            delta = 1  # prevent alt == bucket
        return (bucket ^ delta) % num_buckets

    def _get_buckets(self, key: str) -> Tuple[List[int], int]:
        """Return (candidate_bucket_list, fingerprint) for a key."""
        h = self._hash_to_int(key)
        fp = self._fingerprint(h, self._fingerprint_size)
        b0 = h % self._num_buckets
        buckets = [b0]
        b = b0
        for _ in range(1, self._buckets_per_entry):
            b = self._alt_hash(fp, b, self._num_buckets)
            buckets.append(b)
        return buckets, fp

    def construct(self, vectors: np.ndarray, is_hashed: bool = False) -> None:
        """Populate the filter from a batch of vectors."""
        self._buckets = [[] for _ in range(self._num_buckets)]
        self._size = 0
        self._all_keys = []
        self.insert(vectors, is_hashed=is_hashed)

    def insert(self, vectors: np.ndarray, is_hashed: bool = False) -> None:
        """Insert vectors into the vacuum filter."""
        hashes = self._normalize_hashes(vectors) if is_hashed else self._normalize_hashes(self.lsh.hash(vectors))
        for signature_keys in hashes:
            for key in signature_keys:
                self._insert_single(key)

    def _try_insert_no_expand(self, key: str) -> bool:
        """
        Attempt to insert `key` using direct slot or iterative cuckoo kicking.
        Returns True on success. On failure, undoes all kicks and returns False
        without modifying the table or calling _do_expand().

        This is the non-recursive primitive used by both normal insertion and
        the expand-rebuild loop.

        Duplicate-key optimisation: if the key's fingerprint is already present
        in one of the candidate buckets, a query for this key would return True
        regardless, so we skip the insertion entirely. This is essential when
        many vectors share the same LSH hash (common for small K values), because
        inserting duplicate fingerprints into the same bucket pair would saturate
        the slots and cause infinite kick loops.
        """
        buckets, fp = self._get_buckets(key)
        extra = self._hash_to_int(f"extra:{key}")

        # Skip if the fingerprint is already present — query already returns True.
        for b in buckets:
            for stored_fp, _ in self._buckets[b]:
                if stored_fp == fp:
                    return True

        # Phase 1: direct slot.
        for b in buckets:
            bkt = self._buckets[b]
            if len(bkt) < self._max_slots_per_bucket:
                bkt.append((fp, extra))
                self._size += 1
                self._all_keys.append(key)
                return True

        # Phase 2: iterative cuckoo kicking — bounded by _max_kicks.
        current_fp = fp
        current_bucket = buckets[np.random.randint(len(buckets))]
        kick_chain: List[Tuple[int, int, int, int]] = []

        for _ in range(self._max_kicks):
            bkt = self._buckets[current_bucket]
            if len(bkt) < self._max_slots_per_bucket:
                bkt.append((current_fp, 0))
                self._size += 1
                self._all_keys.append(key)
                return True

            evict_idx = np.random.randint(len(bkt))
            evicted_fp, evicted_extra = bkt[evict_idx]
            bkt[evict_idx] = (current_fp, 0)
            kick_chain.append((current_bucket, evict_idx, evicted_fp, evicted_extra))

            current_bucket = self._alt_hash(evicted_fp, current_bucket, self._num_buckets)
            current_fp = evicted_fp

        # Undo all kicks so the table remains consistent.
        for b_idx, slot_idx, orig_fp, orig_extra in reversed(kick_chain):
            self._buckets[b_idx][slot_idx] = (orig_fp, orig_extra)

        return False

    def _insert_single(self, key: str) -> bool:
        """Insert a key, expanding the table as needed (iterative, not recursive)."""
        if self._try_insert_no_expand(key):
            return True
        while True:
            self._do_expand()
            if self._try_insert_no_expand(key):
                return True

    def _do_expand(self) -> None:
        """
        Expand the table and rebuild all tracked keys without any recursion.

        Targets ≤50% load after expansion so that the rebuild completes without
        further expansion calls. If a rebuild key still fails (extremely unlikely),
        we loop and expand again rather than recursing.
        """
        while True:
            n = len(self._all_keys)
            # At least double; also target ≤50% load for current key count.
            min_slots = max(
                self._num_buckets * 2 * self._max_slots_per_bucket,
                n * 2 + self._max_slots_per_bucket,
            )
            self._num_buckets = (min_slots + self._max_slots_per_bucket - 1) // self._max_slots_per_bucket
            self._buckets = [[] for _ in range(self._num_buckets)]
            self._size = 0
            keys = list(self._all_keys)
            self._all_keys = []

            failed_at: int = -1
            for i, k in enumerate(keys):
                if not self._try_insert_no_expand(k):
                    # Put uninserted keys back and expand again (loop continues).
                    self._all_keys.extend(keys[i:])
                    failed_at = i
                    break

            if failed_at == -1:
                return  # All keys rebuilt successfully.

    def query(self, vectors: np.ndarray, is_hashed: bool = False) -> List[bool]:
        """Query the vacuum filter for membership of the provided vectors."""
        hashes = self._normalize_hashes(vectors) if is_hashed else self._normalize_hashes(self.lsh.hash(vectors))
        results: List[bool] = []

        for signature_keys in hashes:
            found = False
            for key in signature_keys:
                buckets, fp = self._get_buckets(key)
                for b in buckets:
                    for stored_fp, _ in self._buckets[b]:
                        if stored_fp == fp:
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            results.append(found)

        return results

    def add(self, vectors: np.ndarray, is_hashed: bool = False) -> None:
        """Add vectors to the vacuum filter (dynamic insertion supported)."""
        self.insert(vectors, is_hashed=is_hashed)

    def remove(self, vectors: np.ndarray, is_hashed: bool = False) -> List[bool]:
        """
        Remove vectors from the vacuum filter.

        Returns True per vector if it was present and removed.
        """
        hashes = self._normalize_hashes(vectors) if is_hashed else self._normalize_hashes(self.lsh.hash(vectors))
        results: List[bool] = []

        for signature_keys in hashes:
            removed_any = False
            for key in signature_keys:
                buckets, fp = self._get_buckets(key)
                for b in buckets:
                    bucket = self._buckets[b]
                    for i, (stored_fp, _) in enumerate(bucket):
                        if stored_fp == fp:
                            bucket.pop(i)
                            self._size -= 1
                            removed_any = True
                            try:
                                self._all_keys.remove(key)
                            except ValueError:
                                pass
                            break
                    if removed_any:
                        break
                if removed_any:
                    break
            results.append(removed_any)

        return results

    def get_bytesize(self) -> int:
        """Returns the approximate memory size in bytes of the vacuum filter."""
        bucket_overhead = sys.getsizeof(self._buckets)
        entries_size = self._size * (self._fingerprint_size // 8 + 4)
        return bucket_overhead + entries_size

    def __repr__(self) -> str:
        load = self._size / (self._num_buckets * self._max_slots_per_bucket)
        return (
            f"{self.__class__.__name__}("
            f"buckets={self._num_buckets}, "
            f"size={self._size}, "
            f"load_factor={load:.2f}, "
            f"fingerprint_size={self._fingerprint_size}, "
            f"epsilon={self._epsilon:.6g})"
        )
