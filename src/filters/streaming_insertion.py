"""Five streaming insertion methods (M1-M5) for the covering-ball filter,
compared in the streaming figure (see plots/plot_streaming.py). Each method
differs in decision rule and cost class, not just in a tuning knob -- see the
per-class docstrings below for the rule and its complexity.

All five share the same ball representation (BallCover) and the same merge
geometry (filters.ball_geometry, Section 2.3's rule), and all five reach
their first k balls the same way: bootstrap-then-diverge -- the first k
arrivals are appended as raw zero-radius singletons (identical across every
method and both references in runner.py), so checkpoint k+1 onward is a
genuine comparison of decision rules, not of bootstrap artifacts.

The two references described in the streaming spec (exhaustive O(k^2 d)
Algorithm 2, and from-scratch static rebuild) are NOT implemented here --
they're just BallCoverFilter used directly (see exhaustive_algo2_step
below, and runner.stream_rebuild_worker's policy="oracle").
"""

from __future__ import annotations

import heapq
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from filters import ball_geometry
from filters.ball_cover_filter import BallCoverFilter


class BallCover:
    """Fixed-capacity (centers, radii) array pair with in-place mutation.
    Unlike BallCoverFilter's own buffers, capacity here is always known up
    front (k, or k+1 where a method transiently needs a singleton slot) --
    nothing to grow into, so no amortized-doubling machinery."""

    def __init__(self, dim: int, capacity: int):
        self.dim = dim
        self.capacity = capacity
        self.centers = np.empty((capacity, dim), dtype=np.float32)
        self.radii = np.empty(capacity, dtype=np.float64)
        self.size = 0

    def append(self, center: np.ndarray, radius: float) -> int:
        if self.size >= self.capacity:
            raise IndexError("BallCover is at capacity")
        idx = self.size
        self.centers[idx] = center
        self.radii[idx] = radius
        self.size += 1
        return idx

    @property
    def active_centers(self) -> np.ndarray:
        return self.centers[:self.size]

    @property
    def active_radii(self) -> np.ndarray:
        return self.radii[:self.size]


def min_radius_increase_step(cover: BallCover, x: np.ndarray) -> None:
    """The M2 rule, factored out so PeriodicBatchRebuild (M5) can apply the
    identical rule between rebuilds, per spec. Selects the ball requiring the
    smallest radius growth; the max() guard is required (not just cosmetic
    for M1) -- without it, the "already contains x" zero-cost case could
    *shrink* the radius and evict a previously-covered member, breaking the
    covering invariant every method here must preserve (FNR=0)."""
    centers = cover.active_centers.astype(np.float64)
    x64 = np.asarray(x, dtype=np.float64)
    d = np.sqrt(np.maximum(np.sum((centers - x64) ** 2, axis=1), 0.0))
    increase = np.maximum(0.0, d - cover.active_radii)
    i = int(np.argmin(increase))
    if d[i] > cover.radii[i]:
        cover.radii[i] = ball_geometry.safe_radius(d[i])


class StreamingInsertionMethod(ABC):
    """Shared insert(x)/query(q, tau) contract. Bootstrap-then-diverge: the
    first k arrivals are appended as raw zero-radius singletons (same rule
    for every method); _insert_full(x) is the method-specific rule applied
    from arrival k+1 onward."""

    name: str = "base"

    def __init__(self, dim: int, k: int, seed: int = 0):
        self.dim = dim
        self.k = k
        self.seed = seed
        self.cover = BallCover(dim, capacity=self._capacity())
        # Populated only by merge-performing methods (M3, M4); left empty by
        # M1/M2 (no merges ever happen) and M5 (its own accepted_deltas stays
        # empty too -- its cost story is rebuild-vs-absorb time, not Delta).
        self.accepted_deltas: List[float] = []

    def _capacity(self) -> int:
        return self.k

    def insert(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float32)
        if self.cover.size < self.k:
            self.cover.append(x, 0.0)
        else:
            self._insert_full(x)

    @abstractmethod
    def _insert_full(self, x: np.ndarray) -> None:
        ...

    def query(self, q: np.ndarray, tau: float) -> np.ndarray:
        return ball_geometry.ball_cover_query(self.all_spheres, self.all_radiuses, q, tau)

    @property
    def all_spheres(self) -> np.ndarray:
        return self.cover.active_centers

    @property
    def all_radiuses(self) -> np.ndarray:
        return self.cover.active_radii


class NearestCenterAbsorb(StreamingInsertionMethod):
    """M1: absorb into the nearest center by raw distance. Center unchanged;
    radius expands only if needed. O(kd), one pass, no auxiliary state --
    the cheapest possible policy, the floor the other methods must beat."""

    name = "M1"

    def _insert_full(self, x: np.ndarray) -> None:
        centers = self.cover.active_centers.astype(np.float64)
        x64 = np.asarray(x, dtype=np.float64)
        d = np.sqrt(np.maximum(np.sum((centers - x64) ** 2, axis=1), 0.0))
        i = int(np.argmin(d))
        if d[i] > self.cover.radii[i]:
            self.cover.radii[i] = ball_geometry.safe_radius(d[i])


class MinRadiusIncrease(StreamingInsertionMethod):
    """M2: absorb into the ball requiring the smallest radius growth (rather
    than the nearest center). Isolates whether the *selection criterion*
    matters independently of any merge machinery -- if M1 and M2 coincide,
    nearest-center is sufficient and the objective is doing no work."""

    name = "M2"

    def _insert_full(self, x: np.ndarray) -> None:
        min_radius_increase_step(self.cover, x)


class RestrictedMerge(StreamingInsertionMethod):
    """M3: insert (x, 0) as a singleton, evaluate only the k merges involving
    the singleton, apply the beneficial-first argmin. O(kd), no auxiliary
    state. NOT equivalent to Algorithm 2: an already-nested pair of existing
    balls can merge for Delta = -min(A_i, A_j) < 0, cheaper than absorbing x
    (Delta=0 at best), so the *unrestricted* argmin frequently excludes x --
    M3 systematically misses those free consolidations. track_gap=True
    additionally runs that O(k^2 d) unrestricted check (existing-existing
    pairs too) to quantify how often that happens; only ever enabled by the
    dedicated diagnostic job, so the O(kd) benchmark runs stay honest."""

    name = "M3"

    def __init__(self, dim: int, k: int, seed: int = 0, track_gap: bool = False):
        super().__init__(dim, k, seed)
        self.track_gap = track_gap
        self.n_gap = 0
        self.n_total = 0

    def _insert_full(self, x: np.ndarray) -> None:
        centers = self.cover.active_centers.astype(np.float64)
        radii = self.cover.active_radii.astype(np.float64)
        x64 = np.asarray(x, dtype=np.float64)

        bucket, key, _ = ball_geometry.pairwise_merge_geometry(x64[None, :], 0.0, centers, radii, self.dim)
        i = ball_geometry.argmin_bucket_key(bucket, key)

        if self.track_gap:
            self.n_total += 1
            if self._unrestricted_excludes_singleton(centers, radii, bucket, key):
                self.n_gap += 1

        c_new, r_new = ball_geometry.merged_ball(x64, 0.0, centers[i], radii[i])
        self.accepted_deltas.append(ball_geometry.delta_from_bucket_key(int(bucket[i]), float(key[i])))
        self.cover.centers[i] = c_new.astype(np.float32)
        self.cover.radii[i] = ball_geometry.safe_radius(r_new)

    def _unrestricted_excludes_singleton(self, centers: np.ndarray, radii: np.ndarray,
                                         bucket_sx: np.ndarray, key_sx: np.ndarray) -> bool:
        """True iff the globally-best merge among {existing k} u {singleton}
        does NOT involve the singleton -- i.e. where M3's restricted search
        provably diverges from the unrestricted Algorithm 2 argmin. O(k^2 d),
        only ever called with track_gap=True."""
        k = len(centers)
        ci, cj = centers[:, None, :], centers[None, :, :]
        ri, rj = radii[:, None], radii[None, :]
        bucket_ee, key_ee, _ = ball_geometry.pairwise_merge_geometry(ci, ri, cj, rj, self.dim)
        idx = np.arange(k)
        bucket_ee[idx, idx] = 2  # exclude self-pairs (never the argmin, mirrors _all_pairs_merge_keys)
        key_ee[idx, idx] = np.inf

        best_ee = ball_geometry.best_bucket_key(bucket_ee, key_ee)
        best_sx = ball_geometry.best_bucket_key(bucket_sx, key_sx)
        # tuple comparison: bucket 0 < bucket 1, ascending key within a bucket;
        # exact ties favor existing-existing (i.e. count as "excludes the
        # singleton") -- an arbitrary but documented tie-break for this
        # diagnostic-only statistic.
        return best_ee <= best_sx


class CachedHeapMerge(StreamingInsertionMethod):
    """M4: semantically identical to Algorithm 2 (selects the same pair as
    exhaustive O(k^2 d) search) but maintains a priority queue over all
    pairwise Delta values with lazy deletion, so an insertion (which touches
    at most two balls) only recomputes those two balls' costs against the
    rest -- amortized O(kd) per insertion after an O(k^2 d) one-time heap
    build (reported separately, see heap_build_time_s).

    No compaction: capacity is k+1, a free-list of unused slot indices plus
    an occupied mask means a ball's slot index is a stable identity for its
    whole lifetime -- nothing else moves when a ball is removed. A version
    counter per slot lets stale heap entries (referencing a slot whose
    content has since changed) be recognized and discarded lazily on pop,
    rather than eagerly removed from the heap."""

    name = "M4"

    def _capacity(self) -> int:
        return self.k + 1

    def __init__(self, dim: int, k: int, seed: int = 0):
        super().__init__(dim, k, seed)
        self._occupied = np.zeros(self._capacity(), dtype=bool)
        self._version = np.zeros(self._capacity(), dtype=np.int64)
        self._heap: List[Tuple[int, float, int, int, int, int]] = []
        self._free_list: List[int] = []
        self.heap_build_time_s = 0.0
        self.heap_size_peak = 0

    def insert(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float32)
        if self.cover.size < self.k:
            self.cover.append(x, 0.0)
            if self.cover.size == self.k:
                self._occupied[:self.k] = True
                self._build_heap()
        else:
            self._insert_full(x)

    def _build_heap(self) -> None:
        t0 = time.perf_counter()
        centers = self.cover.active_centers.astype(np.float64)
        radii = self.cover.active_radii.astype(np.float64)
        n = self.k
        ci, cj = centers[:, None, :], centers[None, :, :]
        ri, rj = radii[:, None], radii[None, :]
        bucket, key, _ = ball_geometry.pairwise_merge_geometry(ci, ri, cj, rj, self.dim)
        heap: List[Tuple[int, float, int, int, int, int]] = []
        for i in range(n):
            for j in range(i + 1, n):
                heap.append((int(bucket[i, j]), float(key[i, j]), i, j, 0, 0))
        heapq.heapify(heap)
        self._heap = heap
        self._free_list = [n]  # the one extra (k+1-th) slot, unused until the first merge
        self.heap_build_time_s = time.perf_counter() - t0
        self.heap_size_peak = len(heap)

    def _push_row(self, idx: int, others: np.ndarray) -> None:
        if len(others) == 0:
            return
        center_idx = self.cover.centers[idx].astype(np.float64)
        radius_idx = float(self.cover.radii[idx])
        centers_o = self.cover.centers[others].astype(np.float64)
        radii_o = self.cover.radii[others].astype(np.float64)
        bucket, key, _ = ball_geometry.pairwise_merge_geometry(
            center_idx[None, :], radius_idx, centers_o, radii_o, self.dim)
        v_idx = int(self._version[idx])
        for b, ky, other in zip(bucket, key, others):
            other = int(other)
            i, j = (idx, other) if idx < other else (other, idx)
            v_other = int(self._version[other])
            vi, vj = (v_idx, v_other) if i == idx else (v_other, v_idx)
            heapq.heappush(self._heap, (int(b), float(ky), i, j, vi, vj))

    def _pop_best(self) -> Tuple[int, int]:
        while self._heap:
            bucket, key, i, j, vi, vj = heapq.heappop(self._heap)
            if not (self._occupied[i] and self._occupied[j]):
                continue
            if self._version[i] != vi or self._version[j] != vj:
                continue
            return i, j
        raise RuntimeError("CachedHeapMerge: heap exhausted without a valid pair")

    def _insert_full(self, x: np.ndarray) -> None:
        idx_new = self._free_list.pop()
        self.cover.centers[idx_new] = x
        self.cover.radii[idx_new] = 0.0
        self._occupied[idx_new] = True
        self._version[idx_new] += 1

        occ_others = np.flatnonzero(self._occupied)
        occ_others = occ_others[occ_others != idx_new]
        self._push_row(idx_new, occ_others)

        i, j = self._pop_best()

        ci, ri = self.cover.centers[i].astype(np.float64), float(self.cover.radii[i])
        cj, rj = self.cover.centers[j].astype(np.float64), float(self.cover.radii[j])
        bucket, key, _ = ball_geometry.pairwise_merge_geometry(ci, ri, cj, rj, self.dim)
        self.accepted_deltas.append(ball_geometry.delta_from_bucket_key(int(bucket), float(key)))
        c_new, r_new = ball_geometry.merged_ball(ci, ri, cj, rj)

        self.cover.centers[i] = c_new.astype(np.float32)
        self.cover.radii[i] = ball_geometry.safe_radius(r_new)
        self._version[i] += 1
        self._occupied[j] = False
        self._version[j] += 1
        self._free_list.append(j)

        occ_others2 = np.flatnonzero(self._occupied)
        occ_others2 = occ_others2[occ_others2 != i]
        self._push_row(i, occ_others2)
        self.heap_size_peak = max(self.heap_size_peak, len(self._heap))

    @property
    def all_spheres(self) -> np.ndarray:
        return self.cover.centers[self._occupied]

    @property
    def all_radiuses(self) -> np.ndarray:
        return self.cover.radii[self._occupied]


class _GrowingBuffer:
    """Amortized-doubling raw-vector history, used by PeriodicBatchRebuild --
    O(n_seen) memory by construction, since a rebuild needs every point seen
    so far, not just the current cover."""

    def __init__(self, dim: int, initial_capacity: int = 1024):
        self.dim = dim
        self._capacity = initial_capacity
        self.array = np.empty((initial_capacity, dim), dtype=np.float32)
        self.size = 0

    def append(self, x: np.ndarray) -> None:
        if self.size == self._capacity:
            new_capacity = self._capacity * 2
            new_array = np.empty((new_capacity, self.dim), dtype=np.float32)
            new_array[:self.size] = self.array[:self.size]
            self.array = new_array
            self._capacity = new_capacity
        self.array[self.size] = x
        self.size += 1


class PeriodicBatchRebuild(StreamingInsertionMethod):
    """M5: buffer arrivals; every b insertions, discard the cover and rerun
    BallCoverFilter.construct() (Gonzalez seeding -> Lloyd -> MEB
    refinement -> reassignment) on every point seen so far. Between rebuilds,
    absorb arrivals using the M2 rule directly (not BallCoverFilter.add(),
    which would silently substitute Algorithm 2's rule instead)."""

    name = "M5"

    def __init__(self, dim: int, k: int, seed: int = 0, b: int = 1000, construct_tau: float = 0.0):
        super().__init__(dim, k, seed)
        self.b = b
        self._filt = BallCoverFilter(dim=dim, distance_threshold=construct_tau, max_spheres=k, seed=seed)
        self._filt.construct(np.empty((0, dim), dtype=np.float32))
        self._buffer = _GrowingBuffer(dim)
        self.rebuild_time_s = 0.0
        self.absorb_time_s = 0.0

    def insert(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float32)
        self._buffer.append(x)
        super().insert(x)

    def _insert_full(self, x: np.ndarray) -> None:
        t0 = time.perf_counter()
        if self._buffer.size % self.b == 0:
            self._filt.construct(self._buffer.array[:self._buffer.size])
            self._sync_cover_from_filter()
            self.rebuild_time_s += time.perf_counter() - t0
        else:
            min_radius_increase_step(self.cover, x)
            self.absorb_time_s += time.perf_counter() - t0

    def _sync_cover_from_filter(self) -> None:
        centers = self._filt.all_spheres
        radii = self._filt.all_radiuses
        new_cover = BallCover(self.dim, capacity=max(len(centers), self.k))
        for c, r in zip(centers, radii):
            new_cover.append(c, float(r))
        self.cover = new_cover


def exhaustive_algo2_step(filt: BallCoverFilter, x: np.ndarray) -> Optional[float]:
    """Drive one arrival through a BallCoverFilter exactly as add() does
    (unrestricted exhaustive argmin over the cached k x k cost matrix), but
    also return the realized merge Delta -- None if this arrival didn't
    trigger a merge (e.g. during bootstrap). Used by the M4-exactness and
    Delta-distribution diagnostics, which need to observe what the exhaustive
    reference chose without re-deriving its selection logic. Mirrors
    BallCoverFilter.add()'s body; not a public method on that class because
    only these diagnostics need the realized Delta back."""
    x = np.ascontiguousarray(x, dtype=np.float32)[np.newaxis, :]
    if not filt._constructed:
        filt.construct(x)
        return None
    if filt.max_spheres == 0:
        return None

    filt._append_sphere(x[0], 0.0, 1)
    if filt._size <= filt.max_spheres:
        if filt._merge_cache_valid and filt._merge_bucket is not None and filt._merge_bucket.shape[0] >= filt._size:
            filt._merge_cache_size = filt._size
            filt._merge_cache_recompute_slot(filt._size - 1)
        return None

    filt._ensure_merge_cache()
    if not filt._merge_cache_valid:
        return None
    i, j = filt._merge_cache_best_pair()
    bucket = int(filt._merge_bucket[i, j])
    key = float(filt._merge_key[i, j])
    filt._apply_merge_in_place(i, j)
    return ball_geometry.delta_from_bucket_key(bucket, key)


_METHOD_CLASSES = {
    "M1": NearestCenterAbsorb,
    "M2": MinRadiusIncrease,
    "M3": RestrictedMerge,
    "M4": CachedHeapMerge,
    "M5": PeriodicBatchRebuild,
}


def make_method(name: str, dim: int, k: int, seed: int = 0, **kwargs: Any) -> StreamingInsertionMethod:
    if name not in _METHOD_CLASSES:
        raise ValueError(f"Unknown method {name!r}; expected one of {sorted(_METHOD_CLASSES)}")
    return _METHOD_CLASSES[name](dim=dim, k=k, seed=seed, **kwargs)

