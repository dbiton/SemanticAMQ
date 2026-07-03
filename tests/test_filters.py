import pytest
import numpy as np

from lsh import SimHashLSH
from filters.vacuum_filter import VacuumFilter
from filters.ribbon_filter import RibbonFilter

DIM = 32
N_INSERT = 200
N_NON_MEMBERS = 2000


def make_lsh(L: int = 4, K: int = 4) -> SimHashLSH:
    return SimHashLSH(dim=DIM, L=L, K=K)


def make_unit_vectors(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, DIM))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# VacuumFilter
# ---------------------------------------------------------------------------

class TestVacuumFilter:
    def test_no_false_negatives_after_insert(self):
        lsh = make_lsh()
        filt = VacuumFilter(dim=DIM, lsh=lsh, capacity=N_INSERT, fingerprint_size=16)
        vectors = make_unit_vectors(N_INSERT)
        filt.insert(vectors)
        results = filt.query(vectors)
        false_negs = sum(1 for r in results if not r)
        assert false_negs == 0, f"{false_negs} false negatives after insert"

    def test_no_false_negatives_after_construct(self):
        lsh = make_lsh()
        filt = VacuumFilter(dim=DIM, lsh=lsh, capacity=N_INSERT, fingerprint_size=16)
        vectors = make_unit_vectors(N_INSERT)
        filt.construct(vectors)
        results = filt.query(vectors)
        false_negs = sum(1 for r in results if not r)
        assert false_negs == 0, f"{false_negs} false negatives after construct"

    def test_removal_returns_true(self):
        lsh = make_lsh()
        filt = VacuumFilter(dim=DIM, lsh=lsh, capacity=50, fingerprint_size=16)
        vectors = make_unit_vectors(20)
        filt.insert(vectors)
        removed = filt.remove(vectors[:1])
        assert removed[0] is True

    def test_fpr_within_bounds(self):
        lsh = make_lsh()
        filt = VacuumFilter(dim=DIM, lsh=lsh, capacity=N_INSERT, fingerprint_size=16)
        filt.construct(make_unit_vectors(N_INSERT))

        # Probe with synthetic keys to measure the filter's structural FPR directly,
        # bypassing LSH so the result is deterministic and not inflated by hash collisions.
        rng = np.random.default_rng(42)
        n_probe = 50_000
        fp_count = 0
        for i in range(n_probe):
            key = f"probe_{i}_{int(rng.integers(0, 1 << 32))}"
            buckets, fp = filt._get_buckets(key)
            found = False
            for b in buckets:
                for stored_fp in filt._buckets[b]:
                    if stored_fp == fp:
                        found = True
                        break
                if found:
                    break
            if found:
                fp_count += 1

        fpr = fp_count / n_probe
        theoretical = filt.get_exact_matching_fpr()
        assert fpr < theoretical * 3, (
            f"FPR {fpr:.4f} exceeds 3× theoretical {theoretical:.4f}"
        )

    def test_expansion_correctness(self):
        """Force expansion by using a tiny initial capacity, then verify all keys found."""
        lsh = make_lsh()
        filt = VacuumFilter(
            dim=DIM, lsh=lsh, capacity=5, fingerprint_size=16, buckets_per_entry=2
        )
        vectors = make_unit_vectors(80)
        filt.insert(vectors)
        results = filt.query(vectors)
        false_negs = sum(1 for r in results if not r)
        assert false_negs == 0, (
            f"{false_negs}/{len(results)} false negatives after expansion"
        )

    def test_dynamic_add(self):
        lsh = make_lsh()
        filt = VacuumFilter(dim=DIM, lsh=lsh, capacity=N_INSERT, fingerprint_size=16)
        v1 = make_unit_vectors(50, seed=1)
        v2 = make_unit_vectors(50, seed=2)
        filt.construct(v1)
        filt.add(v2)
        assert all(filt.query(v1)), "False negatives in v1 after add(v2)"
        assert all(filt.query(v2)), "False negatives in v2 after add(v2)"

    def test_construct_resets_state(self):
        """A second construct() call replaces the previous state."""
        lsh = make_lsh()
        filt = VacuumFilter(dim=DIM, lsh=lsh, capacity=N_INSERT, fingerprint_size=16)
        v1 = make_unit_vectors(50, seed=1)
        v2 = make_unit_vectors(50, seed=2)
        filt.construct(v1)
        filt.construct(v2)
        assert all(filt.query(v2)), "False negatives in v2 after second construct"


# ---------------------------------------------------------------------------
# RibbonFilter
# ---------------------------------------------------------------------------

class TestRibbonFilter:
    def test_no_false_negatives_after_construct(self):
        lsh = make_lsh()
        filt = RibbonFilter(dim=DIM, lsh=lsh, bits_per_key=8, arity=3)
        vectors = make_unit_vectors(N_INSERT)
        filt.construct(vectors)
        results = filt.query(vectors)
        false_negs = sum(1 for r in results if not r)
        assert false_negs == 0, f"{false_negs} false negatives after construct"

    def test_fpr_within_bounds(self):
        lsh = make_lsh()
        filt = RibbonFilter(dim=DIM, lsh=lsh, bits_per_key=8, arity=3)
        filt.construct(make_unit_vectors(N_INSERT))

        # Probe with synthetic keys to measure the filter's structural FPR directly,
        # bypassing LSH so the result is deterministic and not inflated by hash collisions.
        rng = np.random.default_rng(42)
        member_keys = set(filt._keys)
        n_probe = 50_000
        fp_count = 0
        for i in range(n_probe):
            key = f"probe_{i}_{int(rng.integers(0, 1 << 32))}"
            if key in member_keys:
                continue
            fp, rows = RibbonFilter._hash_to_values(
                key, filt._num_slots, filt._arity, filt._bits_per_key
            )
            xor_sum = fp
            for r in rows:
                xor_sum ^= filt._solution[r]
            if xor_sum == 0:
                fp_count += 1

        fpr = fp_count / n_probe
        theoretical = filt.get_exact_matching_fpr()
        assert fpr < theoretical * 4, (
            f"FPR {fpr:.4f} exceeds 4× theoretical {theoretical:.4f}"
        )

    def test_static_insert_raises(self):
        lsh = make_lsh()
        filt = RibbonFilter(dim=DIM, lsh=lsh)
        with pytest.raises(NotImplementedError):
            filt.insert(make_unit_vectors(10))

    def test_static_add_raises(self):
        lsh = make_lsh()
        filt = RibbonFilter(dim=DIM, lsh=lsh)
        with pytest.raises(NotImplementedError):
            filt.add(make_unit_vectors(10))

    def test_static_remove_raises(self):
        lsh = make_lsh()
        filt = RibbonFilter(dim=DIM, lsh=lsh)
        filt.construct(make_unit_vectors(10))
        with pytest.raises(NotImplementedError):
            filt.remove(make_unit_vectors(5))

    def test_query_before_construct_raises(self):
        lsh = make_lsh()
        filt = RibbonFilter(dim=DIM, lsh=lsh)
        with pytest.raises(ValueError):
            filt.query(make_unit_vectors(5))

    def test_bits_per_key_16(self):
        lsh = SimHashLSH(dim=DIM, L=4, K=20)
        filt = RibbonFilter(dim=DIM, lsh=lsh, bits_per_key=16, arity=3)
        members = make_unit_vectors(N_INSERT, seed=0)
        non_members = make_unit_vectors(N_NON_MEMBERS, seed=99)
        filt.construct(members)
        assert all(filt.query(members)), "False negatives with 16-bit fingerprint"
        results = filt.query(non_members)
        fpr = sum(results) / len(results)
        # 1/65536 theoretical filter FPR; LSH collisions in 32D dominate.
        # With K=20 and DIM=32, cosine-similarity distribution causes SimHash
        # P(same hash) ≈ 7.5e-6 (vs naive 0.5^20 = 9.5e-7), so LSH adds ~0.6%.
        assert fpr < 0.02, f"16-bit FPR unexpectedly high: {fpr:.4f}"

    def test_construct_is_idempotent(self):
        lsh = make_lsh()
        filt = RibbonFilter(dim=DIM, lsh=lsh, bits_per_key=8)
        v1 = make_unit_vectors(50, seed=1)
        v2 = make_unit_vectors(50, seed=2)
        filt.construct(v1)
        filt.construct(v2)
        assert all(filt.query(v2)), "False negatives in v2 after second construct"
