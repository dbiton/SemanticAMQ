"""
Verify that filters honour the `max_memory` budget they were constructed with.

Each filter derives its internal sizing (capacity/quotient/num_slots/etc.) from
`max_memory`. These tests check the two failure modes that matter:

1. Under a workload that comfortably fits, `get_bytesize()` stays <= max_memory.
2. Under a workload that would naturally need far more space than the budget
   allows, the filter must not silently balloon past max_memory. It's
   acceptable to raise (the filter refuses to build something oversized), but
   it is not acceptable to construct successfully while exceeding the budget.

LSHBloom is excluded: it depends on the optional `pybloomfilter` package,
which isn't installed in this environment, so it can't be constructed at all.
"""

import pytest
import numpy as np

from lsh import SimHashLSH
from filters.bloom_filter import BloomFilter, ScalableBloomFilter
from filters.cuckoo_filter import CuckooFilter
from filters.gcs_filter import GCSFilter
from filters.quotient_filter import QuotientFilterWrapper
from filters.ribbon_filter import RibbonFilter
from filters.pyfusefilter_filters import Xor8Filter, Xor16Filter, Fuse8Filter, Fuse16Filter
from filters.kmeb_filter import KMEBFilter
from filters.vacuum_filter import VacuumFilter

DIM = 16


def make_unique_lsh(dim: int = DIM) -> SimHashLSH:
    """L=1, high K -> near-unique signatures per vector, to stress capacity."""
    return SimHashLSH(dim=dim, L=1, K=30)


def make_light_lsh(dim: int = DIM) -> SimHashLSH:
    return SimHashLSH(dim=dim, L=1, K=8)


def make_unit_vectors(n: int, dim: int = DIM, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# CuckooFilter's underlying structure is a fixed-size table that never grows:
# inserts past capacity are silently dropped rather than raising, so it must
# stay within budget even under heavy load without ever throwing.
FIXED_SIZE_FILTERS = [
    (CuckooFilter, dict(fingerprint_size=2)),
]

# Filters that size themselves from max_memory but must refuse (raise) rather
# than silently exceed the budget when the workload doesn't fit. This
# includes BloomFilter, whose underlying pybloom_live structure raises
# IndexError once its (budget-derived) capacity is exceeded, and VacuumFilter,
# which raises once its budget-derived entry cap is reached rather than
# expanding its table past max_memory.
BUDGET_ENFORCING_FILTERS = [
    (BloomFilter, dict(error_rate=0.01)),
    (GCSFilter, dict()),
    (QuotientFilterWrapper, dict()),
    (RibbonFilter, dict(bits_per_key=8)),
    (Xor8Filter, dict()),
    (Xor16Filter, dict()),
    (Fuse8Filter, dict()),
    (Fuse16Filter, dict()),
    (VacuumFilter, dict(fingerprint_size=8)),
]

ALL_MAX_MEMORY_FILTERS = FIXED_SIZE_FILTERS + BUDGET_ENFORCING_FILTERS


@pytest.mark.parametrize("filter_cls,kwargs", ALL_MAX_MEMORY_FILTERS)
def test_respects_budget_under_light_load(filter_cls, kwargs):
    """When the workload comfortably fits, bytesize must stay within budget."""
    lsh = make_light_lsh()
    budget = 100_000
    f = filter_cls(dim=DIM, lsh=lsh, max_memory=budget, **kwargs)
    f.construct(make_unit_vectors(100))
    assert f.get_bytesize() <= budget, (
        f"{filter_cls.__name__} used {f.get_bytesize()} bytes, "
        f"exceeding budget={budget}"
    )


@pytest.mark.parametrize("filter_cls,kwargs", FIXED_SIZE_FILTERS)
def test_fixed_size_filters_stay_bounded_under_heavy_load(filter_cls, kwargs):
    """
    Fixed-size structures must never exceed their budget no matter how much
    data is inserted -- excess items should be dropped/rejected internally,
    not grow the structure.
    """
    lsh = make_unique_lsh()
    budget = 500
    f = filter_cls(dim=DIM, lsh=lsh, max_memory=budget, **kwargs)
    vectors = make_unit_vectors(2000)
    f.construct(vectors)  # more unique keys than the budget can hold
    assert f.get_bytesize() <= budget, (
        f"{filter_cls.__name__} used {f.get_bytesize()} bytes after heavy load, "
        f"exceeding budget={budget}"
    )


@pytest.mark.parametrize("filter_cls,kwargs", BUDGET_ENFORCING_FILTERS)
def test_budget_enforcing_filters_never_exceed_budget_under_heavy_load(filter_cls, kwargs):
    """
    Static/self-sizing filters must not silently exceed max_memory when given
    more unique keys than the budget can support. Either they raise (refusing
    to build an oversized structure) or, if construction somehow succeeds,
    get_bytesize() must still respect the budget.
    """
    lsh = make_unique_lsh()
    budget = 200
    f = filter_cls(dim=DIM, lsh=lsh, max_memory=budget, **kwargs)
    vectors = make_unit_vectors(2000)
    try:
        f.construct(vectors)
    except Exception:
        return  # refusing to build an oversized structure is acceptable
    assert f.get_bytesize() <= budget, (
        f"{filter_cls.__name__} constructed successfully but used "
        f"{f.get_bytesize()} bytes, exceeding budget={budget}"
    )


def test_scalable_bloom_filter_respects_budget_within_initial_capacity():
    """
    ScalableBloomFilter is explicitly designed to grow past its initial
    capacity as more elements are inserted (that's the point of "scalable").
    max_memory only bounds the *initial* capacity, not a hard lifetime cap --
    so we only check the budget holds while under that initial capacity.
    """
    lsh = make_light_lsh()
    budget = 100_000
    f = ScalableBloomFilter(dim=DIM, lsh=lsh, max_memory=budget, error_rate=0.01)
    f.construct(make_unit_vectors(50))
    assert f.get_bytesize() <= budget


def test_kmeb_filter_respects_max_memory_with_many_vectors():
    """Cross-check against tests/test_kmeb_filter.py under a heavier load."""
    dim = 16
    budget = (dim + 1) * 4 * 8  # exactly 8 balls
    f = KMEBFilter(dim=dim, distance_threshold=0.3, max_memory=budget)
    f.construct(make_unit_vectors(500, dim=dim))
    assert f.get_bytesize() <= budget


# ---------------------------------------------------------------------------
# Real-allocation checks.
#
# The tests above trust each filter's own get_bytesize(). That is exactly the
# blind spot that let the memory-usage plots misrepresent reality: a filter
# could derive its size from a wrong per-slot cost and then report an equally
# wrong (too small) bytesize, so it silently allocated 2-4x past max_memory
# while its own accounting claimed it fit. These tests measure the memory each
# filter *actually* allocates -- independently of get_bytesize() -- and assert
# both that it honours the budget and that get_bytesize() reports it faithfully.
# ---------------------------------------------------------------------------

def _real_quotient_backing_bytes(f: QuotientFilterWrapper) -> int:
    """Ground-truth size of the quotient filter's slot table, read straight from
    the underlying array.array (2**quotient slots at the platform itemsize),
    with no reliance on get_bytesize()."""
    backing = f.filter._filter
    return backing.itemsize * len(backing)


@pytest.mark.parametrize("budget", [50_000, 100_000, 500_000, 1_000_000])
def test_quotient_filter_real_allocation_within_budget(budget):
    """The quotient filter's real slot table must fit max_memory, and
    get_bytesize() must report that real size exactly (never under-count it).

    This fails on the old implementation, which sized the quotient from a
    ~11-bits-per-slot guess and reported 1 byte/slot, so the real array.array
    (up to 4 bytes/slot) blew past the budget while get_bytesize() hid it.
    """
    lsh = make_light_lsh()
    f = QuotientFilterWrapper(dim=DIM, lsh=lsh, max_memory=budget)
    f.construct(make_unit_vectors(100))
    real = _real_quotient_backing_bytes(f)
    assert real <= budget, (
        f"QuotientFilter really allocated {real} bytes, exceeding budget={budget}"
    )
    assert f.get_bytesize() == real, (
        f"get_bytesize()={f.get_bytesize()} does not match real allocation {real}"
    )
    assert f.get_bytesize() <= budget


@pytest.mark.parametrize(
    "filter_cls", [Xor8Filter, Xor16Filter, Fuse8Filter, Fuse16Filter]
)
def test_pyfuse_bytesize_matches_real_serialized_size(filter_cls):
    """get_bytesize() must track the fingerprint table's real size.

    Ground truth is the library's serialized byte length. The old hand-tuned
    slots-per-key factors under-counted the Fuse variants by ~14%, which this
    5% tolerance rejects.
    """
    # Several LSH bands -> enough keys that the fixed header doesn't dominate.
    lsh = SimHashLSH(dim=DIM, L=8, K=8)
    f = filter_cls(dim=DIM, lsh=lsh, max_memory=10_000_000)
    f.construct(make_unit_vectors(500))
    real = len(f.filter.serialize())
    reported = f.get_bytesize()
    assert abs(reported - real) <= 0.05 * real, (
        f"{filter_cls.__name__} reported {reported} bytes vs real {real} "
        f"(off by more than 5%)"
    )
