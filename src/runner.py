from __future__ import annotations

import logging
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

# Must happen before numpy/faiss are imported (BLAS/OpenMP thread count x
# Runner's process count otherwise multiply into CPU/memory oversubscription
# -- see main.py's copy of this guard). Repeated here since each
# ProcessPoolExecutor worker re-imports this module fresh.
for _env_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env_var, "1")

import faiss
import numpy as np
import pandas as pd
import time
import inspect

faiss.omp_set_num_threads(1)

from scipy.special import logsumexp
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from logging_config import configure_logging
from filters.filter import InsufficientMemoryError
from filters.dsh_filter import DSHFilter
from filters.cap_cover.cap_cover_filter import (CapCoverFilter, Quantize as CapQuantize,
                                                AUTO_CANDIDATES as _CAP_AUTO_CANDIDATES,
                                                DENSE10_LADDER as _CAP_DENSE10_LADDER)
from filters.cap_cover.cap_geometry import log10_cap_measure, unit_rows
from filters.cap_cover.nearest_insert import NearestInsert as CapNearestInsert
from filters.cap_cover.cap_merge_insert import CapMergeInsert as CapCoverMergeInsert
from filters.cap_cover.gonzales_construct import GonzalesConstruct as CapGonzalesConstruct
from filters.cap_cover.minimax_construct import MinimaxConstruct as CapMinimaxConstruct
from filters.cap_cover.sphere_merge_construct import SphereMergeConstruct as CapSphereMergeConstruct
from filters.cap_cover.cap_interfaces import ConstructInterface as CapConstructInterface
from vector_source import VectorSource

configure_logging()
logger = logging.getLogger(__name__)

# Byte-budget unit for "k caps": the cost of k (centre code + fp16 radius)
# records at 4 bits/dim -- the equal-bytes convention every evaluation sizes
# its CapCoverFilter budget by (k fixes the comparison scale, bytes enforce it).
_BUDGET_BITS_PER_DIM = 4
_BUDGET_BYTES_PER_FLOAT16 = 2


def _cover_max_memory(dim: int, k: int, bits_per_dim: int = _BUDGET_BITS_PER_DIM) -> int:
    """max_memory in bytes corresponding to a nominal k covers at bits_per_dim
    bits per dimension (plus a 2-byte radius each). CapCoverFilter sizes its
    cap count from a byte budget, not a raw count."""
    return k * (dim * bits_per_dim // 8 + _BUDGET_BYTES_PER_FLOAT16)



def _cap_cover(dim: int, k: int) -> CapCoverFilter:
    """CapCoverFilter.auto() at the nominal k-cover byte budget
    (_cover_max_memory)."""
    return CapCoverFilter.auto(dim, max_memory=_cover_max_memory(dim, k))


def _cap_cover_insertable(dim: int, k: int) -> CapCoverFilter:
    """A CapCoverFilter that can insert(), unlike _cap_cover's static
    auto() (insert_method=None there): single-level TQ4 with CapNearestInsert, at the same equal-bytes budget as
    _cap_cover. Kept separate from _cap_cover rather than giving it a
    default insert_method, so cap_cover_system_tau's already-validated
    static-cover numbers don't shift (an insert strategy's bookkeeping
    reserves bytes from the cap budget even when insert() is never
    called -- see InsertInterface.max_caps_for_budget)."""
    return CapCoverFilter(dim=dim, max_memory=_cover_max_memory(dim, k),
                          quantize=CapQuantize.TQ4, insert_method=CapNearestInsert())


# CapCoverFilter's default construct method (what CapCoverFilter()/
# _cap_cover() already use when no construct_method is passed) -- a
# named default for call sites that need one rather than relying on the class's own default.
CAP_COVER_CONSTRUCT = "sphere_merge"

# CapCoverFilter's own construct-method registry: only the 3 strategies its
# package implements.
CAP_CONSTRUCT_METHODS = {
    "gonzalez": CapGonzalesConstruct,
    "minimax": CapMinimaxConstruct,
    "sphere_merge": CapSphereMergeConstruct,
}

# CapCoverFilter's own insert-strategy registry: only the 2 its package
# implements.
CAP_INSERT_STRATEGIES = {
    "cap_merge": CapCoverMergeInsert,
    "nearest": CapNearestInsert,
}

# The quantization evaluation's cap-cover sweep: the ~14 encodings
# CapCoverFilter.auto() actually chooses among (AUTO_CANDIDATES +
# DENSE10_LADDER), not every Quantize member -- auto()'s own candidate pool,
# confirmed by its docstring, so the sweep tests what auto() actually picks
# from rather than exotic opt-in-only rungs (BOOT_CANDIDATES, KSQ_CANDIDATES).
CAP_QUANTIZATION_SWEEP = tuple(_CAP_AUTO_CANDIDATES) + tuple(_CAP_DENSE10_LADDER)

VectorInput = Union[np.ndarray, VectorSource]
_QUERY_BATCH = 32


def _to_construct_array(vectors: VectorInput) -> np.ndarray:
    if isinstance(vectors, VectorSource):
        return vectors.to_array()
    return np.asarray(vectors)


def _iter_query_batches(vectors: VectorInput, batch_size: int) -> Iterator[np.ndarray]:
    if isinstance(vectors, VectorSource):
        yield from vectors.iter_batches(batch_size)
    else:
        arr = np.asarray(vectors)
        for base in range(0, len(arr), batch_size):
            yield arr[base:base + batch_size]


def _iter_query_vectors(vectors: VectorInput) -> Iterator[np.ndarray]:
    if isinstance(vectors, VectorSource):
        yield from vectors
    else:
        yield from np.asarray(vectors)


def _binary_confusion(predicted: Any, actual: Any) -> Tuple[int, int, int, int]:
    p = np.asarray(predicted, dtype=bool)
    a = np.asarray(actual, dtype=bool)
    tn, fp, fn, tp = confusion_matrix(a, p, labels=[False, True]).ravel()
    return int(tp), int(fp), int(fn), int(tn)


def _precision_recall_f1(predicted: Any, actual: Any) -> Tuple[float, float, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual, predicted, average='binary', zero_division=0
    )
    return float(precision), float(recall), float(f1)


_DIRECT_TAU_FILTERS = {
    "FirstNFilter", "CapCoverFilter", "DSBFFilter", "CoveringLSHFallbackFilter", "LSBFFilter",
    "DSHFilter", "HNSWFilter", "IVFFilter", "QuantizedVectorFilter", "MLPFilter",
    "PacukLSHFilter", "SankowskiWygockiFilter", "VAPlusFilter", "IVFRadiiFilter",
}






class _CapCoverFilterEvalAdapter:
    """CapCoverFilter.query() takes an explicit per-call distance_threshold
    rather than baking one in at construction time like the other
    _DIRECT_TAU_FILTERS -- this captures the threshold once so this module's
    generic `filter_instance.query(block)` call sites keep working. Builds via CapCoverFilter.
    auto() (not a plain constructor) so a budget gets auto()'s own
    multi-candidate encoding choice, matching how this evaluation uses
    CapCoverFilter everywhere else (_cap_cover). Renames _instantiate_filter's
    generic max_spheres/insert_mem_percent kwargs onto CapCoverFilter's own
    max_caps/insert_memory_percent."""

    def __init__(self, dim: int, max_memory: int, distance_threshold: float,
                 max_spheres: Optional[int] = None, insert_mem_percent: Optional[float] = 0.0, **extra_kwargs: Any) -> None:
        self.distance_threshold = distance_threshold
        extra_kwargs.setdefault("construct_method", CAP_CONSTRUCT_METHODS[CAP_COVER_CONSTRUCT]())
        self._filt = CapCoverFilter.auto(dim, max_memory=max_memory, max_caps=max_spheres,
                                         insert_memory_percent=insert_mem_percent, **extra_kwargs)

    def construct(self, vectors: np.ndarray) -> None:
        return self._filt.construct(vectors)

    def query(self, vectors: np.ndarray) -> List[bool]:
        return self._filt.query(vectors, distance_threshold=self.distance_threshold)

    def insert(self, vectors: np.ndarray) -> None:
        self._filt.insert(vectors)

    def get_bytesize(self) -> int:
        return self._filt.get_bytesize()

    @property
    def all_angles(self) -> np.ndarray:
        return self._filt.all_angles


def _instantiate_filter(filter_cls: Any, dim: int, memory_budget: int, distance_threshold: float = None, max_spheres: int = None,
                         insert_mem_percent: Optional[float] = 0.0, extra_kwargs: Optional[Dict[str, Any]] = None) -> Any:
    kwargs = {"dim": dim, "max_memory": memory_budget, **(extra_kwargs or {})}

    if filter_cls.__name__ in _DIRECT_TAU_FILTERS:
        kwargs.update(distance_threshold=distance_threshold)

        if filter_cls is CapCoverFilter:
            kwargs.update(max_spheres=max_spheres, insert_mem_percent=insert_mem_percent)
            return _CapCoverFilterEvalAdapter(**kwargs)

        if "max_spheres" in inspect.signature(filter_cls.__init__).parameters:
            kwargs.update(max_spheres=max_spheres)

        return filter_cls(**kwargs)
    else:
        try:
            return filter_cls(**kwargs)
        except TypeError:
            kwargs.pop("max_memory")
            return filter_cls(**kwargs)


def _f1_score(predicted: np.ndarray, actual: np.ndarray) -> float:
    return _precision_recall_f1(predicted, actual)[2]


_IVF_NPROBE_CANDIDATES = (1, 2, 4, 8, 16, 32, 64)
_HNSW_EF_SEARCH_CANDIDATES = (16, 32, 64, 128, 256, 512)
# HNSW graph degree. Unlike ef_search this is not free: the level-0 neighbour
# table alone is n * 2M * 4 bytes, which at M=32, n=80,000 is 20.5MB -- more
# than the whole top budget of the equal-memory sweep, and nothing
# quantization does to the vector codes can shrink it. Left fixed at 32, HNSW
# therefore reported infeasible at every budget in that sweep while IVF's
# structural overhead (nlist centroids) was being sized to the budget for it.
# Descending so the finest graph that fits is tried first, matching
# HNSWFilter._choose_quantize's finest-first order on the other axis.
_HNSW_M_CANDIDATES = (32, 16, 8, 4)


# Filters with a cheap post-construction query-time dial: score every candidate
# on one built instance. Each entry maps the class name to a callable taking the
# built instance and returning a list of SETTINGS DICTS -- each dict a complete
# assignment of that filter's query-time dials, applied together.
#
# Dicts rather than (attr, value) pairs because a pair list cannot express a
# filter with two dials: swept as a flat list, the instance accumulates state
# from earlier candidates and only one attribute gets restored at the end,
# leaving the other wherever the last candidate tried left it.
#
# Every dial here must take effect on an already-built instance, not just set
# an inert Python attribute -- see IVFFilter.nprobe / HNSWFilter.ef_search,
# both properties for exactly that reason (accept_radius_scale needs no
# property: query() reads it on every call).
_POST_CONSTRUCTION_CONFIG_DIALS = {
    "DSBFFilter": lambda f: [{"vote_threshold": t} for t in range(1, f.num_tables + 1)],
    # Swept on a log-ish grid up to nlist, not exhaustively (nlist can be in the hundreds).
    "IVFFilter": lambda f: [
        {"nprobe": p} for p in sorted({c for c in _IVF_NPROBE_CANDIDATES if c <= f.nlist} | {f.nlist})
    ],
    # efSearch is HNSW's accuracy/speed dial and costs no memory, exactly
    # like IVF's nprobe. Without it HNSW competed at a fixed efSearch=64
    # against an IVF whose nprobe was tuned per budget -- not an equal
    # comparison of the two baselines.
    "HNSWFilter": lambda f: [{"ef_search": e} for e in _HNSW_EF_SEARCH_CANDIDATES],
}

# Accept-radius multipliers on tau, swept for any filter exposing
# query_squared_nn_distance() (IVFFilter, HNSWFilter). Those two threshold an
# approximate index's ESTIMATE of the nearest-neighbor distance, and the lossy
# codes bias that estimate upward by more than tau itself at these budgets --
# testing it against the exact tau made both answer "not a duplicate" to
# nearly every true duplicate. See IVFFilter's module docstring for the
# measurement. Sweeping the radius costs no extra searches: the radius is a
# comparison against distances one search pass already returned, so
# _tune_post_construction scores the whole grid off each pass.
_ACCEPT_RADIUS_SCALES = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0)

# The floor-projection zero-FN LSH filters (PacukLSHFilter,
# SankowskiWygockiFilter; see filters._floor_lsh) expand every point to
# tables x 3^L cells -- written at build time with replicate="store", probed
# per query with replicate="probe". Longer signatures narrow the buckets (lower
# structural FPR) at 3^L the keys or lookups, so L is their accuracy/memory
# dial, just as K is DSHFilter's. Candidates past this many cells per point are
# never generated: in store mode they could not fit any equal-memory budget
# anyway, and in probe mode each query's Python-level Bloom lookups grow with it.
# Also keeps every candidate under FloorLSHFilter._MAX_CELLS_PER_POINT, whose
# ValueError the dial loop below does not catch.
_FLOOR_LSH_MAX_CELLS = 3 ** 8
_FLOOR_LSH_REPLICATE = ("store", "probe")


def _pacuk_lsh_candidates(dim: int, distance_threshold: float) -> List[Dict[str, Any]]:
    """num_hashes (one table, 3^L cells) x replicate mode."""
    return [
        {"num_hashes": num_hashes, "replicate": replicate}
        for num_hashes in range(1, 32) if 3 ** num_hashes <= _FLOOR_LSH_MAX_CELLS
        for replicate in _FLOOR_LSH_REPLICATE
    ]


def _sankowski_wygocki_candidates(dim: int, distance_threshold: float) -> List[Dict[str, Any]]:
    """block_dim x num_families x hashes_per_block x replicate mode. block_dim
    sets the table count ((d'/k)^L tables); block_dim=dim is a single table
    over one random rotation, where a second family would only duplicate
    hashes_per_block, so it gets num_families=1 only."""
    candidates = []
    for block_dim in (16, 32, 64, 128, dim):
        num_blocks = -(-dim // block_dim)
        for num_families in ((1,) if num_blocks == 1 else (1, 2)):
            num_tables = num_blocks ** num_families
            hashes_per_block = 1
            while num_tables * 3 ** (hashes_per_block * num_families) <= _FLOOR_LSH_MAX_CELLS:
                for replicate in _FLOOR_LSH_REPLICATE:
                    candidates.append({
                        "block_dim": block_dim, "num_families": num_families,
                        "hashes_per_block": hashes_per_block, "replicate": replicate,
                    })
                hashes_per_block += 1
    return candidates

# Filters whose configuration changes memory layout at construction time: one build per candidate.
# A value may be a static list of kwargs (LSBFFilter) or a callable
# (dim, distance_threshold) -> list of kwargs, for a dial whose candidates
# depend on the sweep's own parameters (DSHFilter's z grid).
_CONSTRUCTION_TIME_CONFIG_DIALS = {
    "LSBFFilter": [{"verify": True}, {"verify": False}],
    # M sizes the graph, which is allocated at construction time, so unlike
    # ef_search it cannot be tuned on a built instance -- see
    # _HNSW_M_CANDIDATES for why leaving it fixed made HNSW infeasible
    # everywhere. HNSWFilter is in BOTH dial registries: each M here is built
    # once and then scored at its own best ef_search (see
    # _tune_post_construction), so M is not chosen by an artifact of one
    # arbitrary ef_search. Candidates whose graph does not fit the budget
    # raise InsufficientMemoryError from construct()'s pre-build floor check,
    # before any graph is actually built, so the infeasible ones are cheap.
    "HNSWFilter": lambda dim, distance_threshold: [{"M": m} for m in _HNSW_M_CANDIDATES],
    # DSHFilter bakes z into every inserted point's hash codes, so unlike
    # IVFFilter.nprobe/HNSWFilter.ef_search it cannot be tuned post-hoc on a
    # single built instance -- it needs one build per candidate, exactly
    # like LSBFFilter.verify above. Without this, z came only from a
    # synthetic uniform-sphere model (see DSHFilter's module docstring),
    # while IVF/HNSW's dials were already selected by F1 on real calibration
    # data -- an asymmetry that understated DSH's achievable accuracy on
    # real, non-uniform embeddings. z_grid() reuses the same bounded
    # candidate range that synthetic model would have searched, so this
    # changes *how* the best candidate in that range is chosen, not the
    # range itself.
    #
    # num_projections_per_table (K) is swept too, not left at its default of
    # 4: K is DSHFilter's only real accuracy/memory dial (more per-projection
    # AND-composition shrinks the false-accept rate exponentially, at the
    # cost of needing more OR-tables to hold recall up, i.e. more bytes) --
    # with K fixed, DSHFilter's memory budget only ever bought a bigger Bloom
    # filter, whose own ~1% engineered error rate is swamped by K=4's
    # structural false-accept rate on real (non-uniform-sphere) embeddings,
    # which is why FPR measured flat at ~1.0 across the whole equal-memory
    # sweep regardless of budget. Candidates that don't fit memory_budget or
    # fail the calibration_min_gap guard at a given K are skipped by
    # _build_calibrated_filter's own exception handling below, same as
    # before.
    "DSHFilter": lambda dim, distance_threshold: [
        {"num_projections_per_table": k, "z": float(z)}
        for k in (4, 8, 16, 24, 32, 48, 64)
        for z in DSHFilter.z_grid(distance_threshold)
    ],
    # Same treatment as DSHFilter: every candidate is built at the budget and
    # scored by F1 on calibration; ones whose keys don't fit raise
    # InsufficientMemoryError before hashing anything, so they cost nothing.
    "PacukLSHFilter": _pacuk_lsh_candidates,
    "SankowskiWygockiFilter": _sankowski_wygocki_candidates,
}


def _tune_post_construction(
    filter_instance: Any,
    name: str,
    calib_arr: np.ndarray,
    calib_actual: np.ndarray,
) -> float:
    """Sets filter_instance's query-time dials to whichever combination scores
    best (F1) on the calibration set, and returns that F1 -- or the instance's
    as-built F1 when it has no tunable dial.

    Two dial sources, both free of memory cost:
      - the registered search-shaping dial (_POST_CONSTRUCTION_CONFIG_DIALS),
        one query pass per candidate settings dict;
      - the accept radius (_ACCEPT_RADIUS_SCALES), for any filter exposing
        query_squared_nn_distance(). The radius only reinterprets distances a
        search already returned, so the whole grid is scored off each search
        pass rather than costing a pass of its own.

    Split out of _build_calibrated_filter so the construction-time dial loop
    can use it too. It could not before: that loop `return`ed as soon as it
    picked a build, so a filter registered in BOTH dial tables (HNSWFilter now
    is) silently never got its query-time dial tuned at all."""
    if len(calib_arr) == 0:
        return 0.0

    dial = _POST_CONSTRUCTION_CONFIG_DIALS.get(name)
    settings_grid = dial(filter_instance) if dial is not None else [{}]
    tunes_radius = hasattr(filter_instance, "query_squared_nn_distance")

    def apply(settings: Dict[str, Any]) -> None:
        for attr, value in settings.items():
            setattr(filter_instance, attr, value)

    best_settings, best_f1 = None, -1.0
    for settings in settings_grid:
        apply(settings)
        if tunes_radius:
            sq_dist = np.asarray(filter_instance.query_squared_nn_distance(calib_arr), dtype=np.float64)
            for scale in _ACCEPT_RADIUS_SCALES:
                radius = filter_instance.distance_threshold * scale
                f1 = _f1_score(sq_dist <= radius * radius, calib_actual)
                if f1 > best_f1:
                    best_settings = {**settings, "accept_radius_scale": scale}
                    best_f1 = f1
        else:
            f1 = _f1_score(np.asarray(filter_instance.query(calib_arr), dtype=bool), calib_actual)
            if f1 > best_f1:
                best_settings, best_f1 = settings, f1

    if best_settings is not None:
        apply(best_settings)
        logger.debug(f"{name}: calibrated {best_settings} at F1={best_f1:.4f}")
    return best_f1


def _build_calibrated_filter(
    filter_cls: Any,
    dim: int,
    memory_budget: int,
    distance_threshold: float,
    construct_arr: np.ndarray,
    calib_arr: np.ndarray,
    calib_actual: np.ndarray,
    extra_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, float]:
    """Builds filter_cls at memory_budget, picking whichever registered config
    dial candidate scores best (F1) on the calibration set. Returns
    (filter_instance, construction_seconds) -- the latter sums only time
    inside .construct() calls, excluding the calibration .query() calls used
    to score candidates, so it reflects reaching a ready-to-query filter.
    extra_kwargs is forwarded to _instantiate_filter for filters outside the
    config-dial registries (e.g. a fixed quantize)."""
    name = filter_cls.__name__
    construction_time = 0.0

    if name in _CONSTRUCTION_TIME_CONFIG_DIALS:
        dial = _CONSTRUCTION_TIME_CONFIG_DIALS[name]
        candidates = dial(dim, distance_threshold) if callable(dial) else dial
        best_f1, best_instance = -1.0, None
        last_error: Optional[Exception] = None
        for kwargs in candidates:
            # Via _instantiate_filter's own dispatch, not a hardcoded
            # direct-tau call, so a dial-registry filter not in
            # _DIRECT_TAU_FILTERS still gets built correctly.
            try:
                candidate = _instantiate_filter(
                    filter_cls, dim, memory_budget, distance_threshold, extra_kwargs=kwargs
                )
                t0 = time.perf_counter()
                candidate.construct(construct_arr)
                construction_time += time.perf_counter() - t0
            except (InsufficientMemoryError, RuntimeError) as e:
                # A dial value can be infeasible at this budget (DSH's z
                # shifts num_tables, so its memory footprint moves with the
                # grid) or fail DSH's own calibration_min_gap guard -- either
                # way that is one bad candidate, not a reason to fail the
                # whole sweep row as long as some other candidate works.
                last_error = e
                continue
            # Each construction candidate gets its own query-time dial tuned
            # before being scored, so the comparison between them is between
            # their BEST achievable operating points -- HNSWFilter is in both
            # registries (M at construction, ef_search after it), and scoring
            # every M at one arbitrary ef_search would pick M by an artifact
            # of that choice.
            f1 = _tune_post_construction(candidate, name, calib_arr, calib_actual)
            if f1 > best_f1:
                best_f1, best_instance = f1, candidate
        if best_instance is None:
            raise InsufficientMemoryError(
                f"No candidate for {name} in its construction-time config dial fit "
                f"max_memory={memory_budget} bytes / passed its calibration guard "
                f"(last error: {last_error})."
            )
        # No re-tune needed: _tune_post_construction leaves each candidate
        # set to its own best dial value before returning, and the later
        # candidates it ran on are separate instances, so best_instance is
        # already at the operating point its winning F1 was measured at.
        return best_instance, construction_time

    filter_instance = _instantiate_filter(filter_cls, dim, memory_budget, distance_threshold, extra_kwargs=extra_kwargs)
    t0 = time.perf_counter()
    filter_instance.construct(construct_arr)
    construction_time = time.perf_counter() - t0

    _tune_post_construction(filter_instance, name, calib_arr, calib_actual)

    return filter_instance, construction_time


def equal_memory_worker(
    filter_cls: Any,
    dim: int,
    distance_threshold: float,
    memory_budget: int,
    vectors_construct: VectorInput,
    vectors_calibration: VectorInput,
    vectors_query: VectorInput,
    actual_calibration: List[bool],
    actual_query: List[bool],
    dataset: str = "",
    extra_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    # A filter can be submitted multiple times at one budget, once per
    # quantize variant (see main.generate_equal_memory) -- suffix the variant
    # onto Config so each becomes its own labelled line downstream rather
    # than being pooled into one line. The suffix uses
    # "(...)"  rather than plot_equal_memory.aggregate's " + " delimiter, so
    # its filter-identity split leaves the whole string intact.
    quantize = (extra_kwargs or {}).get("quantize")
    config = f"{filter_cls.__name__} ({quantize.value})" if quantize is not None else filter_cls.__name__
    construct_arr = _to_construct_array(vectors_construct)
    calib_arr = _to_construct_array(vectors_calibration)
    calib_actual = np.asarray(actual_calibration, dtype=bool)

    try:
        filter_instance, construction_time = _build_calibrated_filter(
            filter_cls, dim, memory_budget, distance_threshold,
            construct_arr, calib_arr, calib_actual, extra_kwargs=extra_kwargs,
        )
    except InsufficientMemoryError as e:
        # Expected at small budgets -- recorded as an explicit infeasible
        # row (NaN metrics) rather than dropped or crashing the sweep. Every
        # filter raises the one shared class from filters.filter; this used
        # to name only QuantizedVectorFilter's copy.
        logger.warning(f"[Infeasible] {config} at memory_budget={memory_budget}: {e}")
        return {
            'Config': config, 'Dataset': dataset, 'Memory Budget': memory_budget,
            'Memory Usage': float('nan'), 'Over Budget': False,
            'Threshold': distance_threshold,
            'FPR': float('nan'), 'FNR': float('nan'),
            'Latency (ms)': float('nan'), 'Throughput (qps)': float('nan'),
            'Infeasible': True,
            'Construct Size': len(construct_arr),
        }

    predicted: List[bool] = []
    start = time.perf_counter()
    for block in _iter_query_batches(vectors_query, _QUERY_BATCH):
        predicted.extend(filter_instance.query(block))
    elapsed = time.perf_counter() - start

    tp, fp, fn, tn = _binary_confusion(predicted, actual_query)
    n_q = max(len(actual_query), 1)

    # The premise of the whole experiment is that a point plotted at a given
    # budget respects it. Filters enforce that themselves (raising
    # InsufficientMemoryError above rather than shipping an over-budget
    # index), so anything caught here is a filter bug, not a tight budget --
    # hence a loud warning and a flagged row rather than a silent datapoint.
    memory_usage = filter_instance.get_bytesize()
    over_budget = memory_usage > memory_budget
    if over_budget:
        logger.warning(
            f"[Over budget] {config} at memory_budget={memory_budget} used {memory_usage} bytes "
            f"({memory_usage / max(memory_budget, 1):.2f}x). Its get_bytesize() and its "
            f"construct()-time budget check disagree; treat this row as unsound."
        )

    return {
        'Config': config,
        'Dataset': dataset,
        'Memory Budget': memory_budget,
        'Memory Usage': memory_usage,
        'Over Budget': over_budget,
        'Threshold': distance_threshold,
        'FPR': fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        'FNR': fn / (fn + tp) if (fn + tp) > 0 else 0.0,
        # Construction latency, not query latency (see
        # _build_calibrated_filter's docstring for what's included).
        'Latency (ms)': construction_time * 1000,
        'Throughput (qps)': n_q / max(elapsed, 1e-9),
        'Infeasible': False,
        # This corpus/fold's construct-set size, so plot_equal_memory can
        # express Memory Budget/Usage as a percentage of storing this set
        # raw (Construct Size * dim float32s) -- per row, since datasets
        # pooled into this sweep don't all share the same reference-set
        # size (see main.generate_equal_memory's min-over-corpora budget
        # scaling), unlike the shared budget grid itself.
        'Construct Size': len(construct_arr),
    }


_LN10 = float(np.log(10.0))










def _cap_log10_volume(filt: CapCoverFilter, tau: float) -> float:
    """log10 upper bound on the share of the unit sphere `filt` accepts at
    `tau`: log10 of the sum over caps of their tau-expanded measure
    (sigma(psi), psi = min(stored angle + beta, pi)) -- the cap-cover
    naive sum-of-volumes proxy, ported from
    compare_cap_cover.py's accept_angles/accept_volume. No overlap-aware
    union bound exists for caps (nothing in cap_geometry.py does
    angle-based overlap), so this returns
    only the upper bound -- callers pair it with NaN where a lower bound
    column is expected."""
    angles = filt.all_angles
    if len(angles) == 0:
        return float('-inf')
    beta, _ = filt._beta(np.ones(1), tau)
    psi = np.minimum(angles + float(beta[0]), np.pi)
    log10_measure = log10_cap_measure(psi, filt.dim)
    return float(logsumexp(log10_measure * _LN10)) / _LN10


def _confusion_rates(predicted: List[bool], actual: np.ndarray) -> Dict[str, float]:
    tp, fp, fn, tn = _binary_confusion(predicted, actual)
    return {
        'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        'fnr': fn / (fn + tp) if (fn + tp) > 0 else 0.0,
    }


# The blocklist workload's query stream is drawn from three sources, in this
# order (main.BlocklistWorkload.query_vectors): held-out members of the
# reference set's clusters, members of clusters withheld from it, and
# singletons. Sources are strata of the query distribution, never labels --
# every query is labeled by distance at tau, whichever source it came from.
QUERY_SOURCES = ("heldout", "withheld", "singleton")


def _per_source_rates(predicted: Any, actual: Any, n_heldout: int, n_withheld: int,
                      n_singleton: int) -> Dict[str, Any]:
    """FPR restricted to the negatives drawn from each source, those negative
    counts, the positive count, and neg_share_pct -- the share of negative
    queries, which bounds accesses saved for a zero-false-negative filter.
    predicted/actual follow the QUERY_SOURCES layout; a source with no
    negatives reports fpr NaN rather than 0."""
    predicted = np.asarray(predicted, dtype=bool)
    actual = np.asarray(actual, dtype=bool)
    sizes = (n_heldout, n_withheld, n_singleton)
    if sum(sizes) != len(actual) or len(predicted) != len(actual):
        raise ValueError(f"Source sizes {sizes} do not partition {len(actual)} queries "
                         f"({len(predicted)} predictions)")
    out: Dict[str, Any] = {}
    start = 0
    for source, n in zip(QUERY_SOURCES, sizes):
        neg = ~actual[start:start + n]
        out[f"fpr_{source}"] = float(predicted[start:start + n][neg].mean()) if neg.any() else float("nan")
        out[f"n_neg_{source}"] = int(neg.sum())
        start += n
    out["n_pos"] = int(actual.sum())
    out["neg_share_pct"] = 100.0 * (1.0 - actual.mean()) if len(actual) else float("nan")
    return out






def cap_quantization_worker(
    dim: int,
    dataset: str,
    k: int,
    quantize: CapQuantize,
    tau: float,
    seed: int,
    vectors_construct: VectorInput,
    vectors_query: VectorInput,
    actual_results: List[bool],
    n_heldout: int,
    n_withheld: int,
    n_singleton: int,
) -> List[Dict[str, Any]]:
    """One CapCoverFilter cover per
    (dataset, k, seed, quantize) over CAP_QUANTIZATION_SWEEP (AUTO_CANDIDATES
    + DENSE10_LADDER, the ~14 encodings CapCoverFilter.auto() actually
    chooses among -- not the full, much larger Quantize enum), evaluated at a single tau. Returns one
    row, tagged stage="quantizing" always (CAP_QUANTIZATION_SWEEP has no
    FP32/uncompressed member to single out as a baseline stage).

    max_memory is derived from k at 32 bits/dim (_cover_max_memory), held
    fixed across every quantize candidate, so the sweep varies only the
    encoding.

    log10_J/log10_J_tau use _cap_log10_volume. FNR is asserted (raises on
    violation), never plotted."""
    construct_arr = _to_construct_array(vectors_construct)
    query_arr = _to_construct_array(vectors_query)
    n_q = len(query_arr)
    if len(actual_results) != n_q:
        raise ValueError(
            f"vectors_query ({n_q}) and actual_results ({len(actual_results)}) length mismatch"
        )

    actual = np.asarray(actual_results, dtype=bool)
    if n_heldout + n_withheld + n_singleton != n_q:
        raise ValueError(f"Source sizes ({n_heldout}, {n_withheld}, {n_singleton}) do not "
                         f"partition n_queries ({n_q})")

    rows: List[Dict[str, Any]] = []

    max_memory = _cover_max_memory(dim, k, 32)
    t0 = time.perf_counter()
    try:
        filt = CapCoverFilter(dim=dim, max_memory=max_memory, quantize=quantize,
                              construct_method=CAP_CONSTRUCT_METHODS[CAP_COVER_CONSTRUCT]())
        filt.construct(construct_arr)
    except Exception as e:
        logger.warning(
            f"Skipping cap_quantization job (dataset={dataset}, k={k}, quantize={quantize.value}, "
            f"seed={seed}): construction raised {type(e).__name__}: {e}"
        )
        return []
    stage_time = time.perf_counter() - t0
    log10_J = _cap_log10_volume(filt, 0.0)

    memory_bytes = filt.get_bytesize()

    q0 = time.perf_counter()
    predicted = filt.query(query_arr, distance_threshold=tau)
    query_time = time.perf_counter() - q0

    predicted_arr = np.asarray(predicted, dtype=bool)
    n_index_accesses = int(predicted_arr.sum())

    rates = _confusion_rates(predicted, actual)
    rows.append({
        "dataset": dataset, "k": k, "quantize": quantize.value, "stage": "quantizing", "tau": tau,
        "log10_J": log10_J,
        "log10_J_tau": _cap_log10_volume(filt, tau),
        "fpr": rates["fpr"], "fnr": rates["fnr"],
        **_per_source_rates(predicted_arr, actual, n_heldout, n_withheld, n_singleton),
        "stage_time_s": stage_time, "cum_time_s": stage_time,
        "query_throughput_qps": n_q / max(query_time, 1e-9),
        "n_queries": n_q,
        "n_index_accesses": n_index_accesses,
        "index_accesses_saved": n_q - n_index_accesses,
        "index_accesses_saved_pct": (1.0 - n_index_accesses / max(n_q, 1)) * 100.0,
        "n_caps": filt._num_caps, "memory_bytes": memory_bytes, "seed": seed,
        "n_construct": len(construct_arr),
    })

    bad = [r for r in rows if r["fnr"] != 0]
    if bad:
        raise InvariantViolation(
            f"Zero-false-negative invariant violated (dataset={dataset}, k={k}, quantize={quantize}, seed={seed}): "
            f"{bad[:3]}"
        )

    return rows




class _TimedCapConstruct(CapConstructInterface):
    """_TimedConstruct's cap-cover analogue. cap_cover's ConstructInterface
    adds iter_candidates()/finish() to the ball-cover contract (for choosing
    among several encodings' cap counts at once), but construct_method_worker
    only ever asks for one cap count, so the inherited default
    iter_candidates (which calls construct_filter once) and finish
    (identity) are exactly what's wanted -- only construct_filter itself
    needs timing, same as _TimedConstruct."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.seconds = float("nan")

    def construct_filter(self, filter, vectors):
        start = time.perf_counter()
        out = self.inner.construct_filter(filter, vectors)
        self.seconds = time.perf_counter() - start
        return out

    def get_bytesize(self) -> int:
        return self.inner.get_bytesize()


def construct_method_worker(
    dim: int,
    dataset: str,
    method: str,
    k: int,
    k_fraction: float,
    tau: float,
    seed: int,
    vectors_construct: np.ndarray,
    vectors_query: np.ndarray,
    actual_results: np.ndarray,
    n_heldout: int,
    n_withheld: int,
    n_singleton: int,
) -> Dict[str, Any]:
    """One CapCoverFilter built by CAP_CONSTRUCT_METHODS[method] at k caps,
    with the filter's default (FP32) quantization and no insert method, so
    the construction method is the only thing that varies between rows.

    Returns one row: cover volume, angles, FPR/FNR against distance ground
    truth at tau, and construction time. vectors_query is the fold's
    [held-out, withheld, singleton] query stream
    (BlocklistWorkload.query_vectors), sized by n_heldout/n_withheld/
    n_singleton, so fpr splits by source as budget_worker's does (see
    _per_source_rates). Covers are unquantized, so a row is not comparable
    with results/cap_quantization.csv's, which is built at a swept encoding.
    seed is the fold, reported as "fold".

    A nonzero FNR is a bug in the construction method, not a data point, so
    it raises (the Runner logs and skips the job) instead of returning."""
    construct = np.ascontiguousarray(vectors_construct, dtype=np.float32)
    queries = np.ascontiguousarray(vectors_query, dtype=np.float32)
    actual = np.asarray(actual_results, dtype=bool)
    n = len(construct)

    timed = _TimedCapConstruct(CAP_CONSTRUCT_METHODS[method]())
    filt = CapCoverFilter(dim, construct_method=timed, max_caps=k, insert_method=None)
    start = time.perf_counter()
    filt.construct(construct)
    construct_s = time.perf_counter() - start

    angles = filt.all_angles
    start = time.perf_counter()
    vol_high = _cap_log10_volume(filt, tau)
    volume_s = time.perf_counter() - start

    predicted = np.asarray(filt.query(queries, distance_threshold=tau), dtype=bool)
    tp = int(np.sum(predicted & actual))
    fp = int(np.sum(predicted & ~actual))
    fn = int(np.sum(~predicted & actual))
    tn = int(np.sum(~predicted & ~actual))

    row = {
        "dataset": dataset, "fold": seed, "method": method,
        "k_fraction": float(k_fraction), "k": k, "n_caps": len(angles), "n_reference": n,
        "tau": tau,
        "log10_vol_low": float('nan'), "log10_vol_high": vol_high,
        "log10_j_tau": vol_high,
        "psi_mean": float(angles.mean()), "psi_med": float(np.median(angles)),
        "psi_max": float(angles.max()),
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "fnr": fn / (fn + tp) if fn + tp else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        **_per_source_rates(predicted, actual, n_heldout, n_withheld, n_singleton),
        "method_s": timed.seconds, "construct_s": construct_s, "volume_s": volume_s,
    }
    if fn:
        raise InvariantViolation(f"Zero-false-negative invariant violated -- a bug in construction method "
                         f"{method!r}: {row}")
    return row


def construct_method_scale_worker(dim: int, dataset: str, n: int, method: str, k: int) -> Dict[str, Any]:
    """construct_filter alone on the first n rows of a corpus, plus a coverage
    check of its output: how a construction method's time grows past the
    blocklist workload's size. Raises on a coverage violation, as
    construct_method_worker does on a false negative."""
    X = np.ascontiguousarray(VectorSource(dataset, 0, n).to_array(), dtype=np.float32)
    filt = CapCoverFilter(dim, max_caps=k, insert_method=None)
    start = time.perf_counter()
    directions, angles = CAP_CONSTRUCT_METHODS[method]().construct_filter(filt, X)
    seconds = time.perf_counter() - start

    # Angle from each vector to its 16 nearest cap directions (which include
    # the one it was assigned to), minus that cap's stored angle -- the
    # angle-cover analogue of construct_method_worker's radius-containment
    # check, at the filter's own coverage epsilon.
    directions = unit_rows(np.ascontiguousarray(directions, dtype=np.float64))
    index = faiss.IndexFlatIP(dim)
    index.add(directions.astype(np.float32))
    X_unit = unit_rows(X.astype(np.float64))
    sims, labels = index.search(X_unit.astype(np.float32), min(16, len(directions)))
    actual_angle = np.arccos(np.clip(sims, -1.0, 1.0).astype(np.float64))
    worst = float((actual_angle - angles[labels]).min(axis=1).max())
    row = {"dataset": dataset, "n": len(X), "method": method, "k": k, "n_caps": len(angles),
           "method_s": seconds, "psi_max": float(angles.max()), "psi_mean": float(angles.mean()),
           "worst_uncovered": worst, "covered": worst <= CapCoverFilter._COVERAGE_EPSILON}
    if not row["covered"]:
        raise InvariantViolation(f"Coverage violation at scale: {row}")
    return row


def system_tau_ground_truth_worker(
    dim: int,
    vectors_construct: VectorInput,
    vectors_query: VectorInput,
) -> np.ndarray:
    """Nearest-construct-set-neighbor distance for every query vector, via a
    brute-force flat index. Split into its own worker so main.py can run it
    across a ProcessPoolExecutor and cache the result to disk, rather than
    generate_system_tau() paying this cost serially per dataset."""
    construct_arr = np.ascontiguousarray(_to_construct_array(vectors_construct), dtype=np.float32)
    query_arr = np.ascontiguousarray(_to_construct_array(vectors_query), dtype=np.float32)
    index = faiss.IndexFlatL2(dim)
    index.add(construct_arr)
    nn_sq_dist, _ = index.search(query_arr, 1)
    return np.sqrt(np.clip(nn_sq_dist[:, 0], 0, None)).astype(np.float32)




def cap_cover_system_tau_worker(
    dim: int,
    dataset: str,
    k: int,
    tau_values: List[float],
    seed: int,
    vectors_construct: VectorInput,
    vectors_query: VectorInput,
    actual_results_by_tau: Dict[float, List[bool]],
) -> List[Dict[str, Any]]:
    """CapCoverFilter's analogue of system_tau_worker: one CapCoverFilter.auto()
    cover (_cap_cover's equal-byte budget for k balls) plus a flat L2 index
    over the same construct set, queried one at a time at every tau in
    tau_values without ever rebuilding -- the realistic online setting a
    byte-budgeted membership filter sits in front of a flat index in.
    Reports n_caps for the cover's size."""
    construct_arr = np.ascontiguousarray(_to_construct_array(vectors_construct), dtype=np.float32)
    query_arr = np.ascontiguousarray(_to_construct_array(vectors_query), dtype=np.float32)
    n_q = len(query_arr)
    for t, actual in actual_results_by_tau.items():
        if len(actual) != n_q:
            raise ValueError(
                f"vectors_query ({n_q}) and actual_results_by_tau[{t}] ({len(actual)}) length mismatch"
            )

    index = faiss.IndexFlatL2(dim)
    index.add(construct_arr)

    filt = _cap_cover(dim, k)
    filt.construct(construct_arr)
    n_caps = len(filt.all_caps)

    rows: List[Dict[str, Any]] = []
    for t in tau_values:
        actual = np.asarray(actual_results_by_tau[t], dtype=bool)
        redundancy_rate = float(np.mean(actual)) if len(actual) > 0 else 0.0

        start = time.perf_counter()
        for i in range(n_q):
            index.range_search(query_arr[i:i + 1], t * t)
        time_without = time.perf_counter() - start

        predicted = np.empty(n_q, dtype=bool)
        start = time.perf_counter()
        for i in range(n_q):
            v = query_arr[i:i + 1]
            hit = filt.query(v, distance_threshold=t)[0]
            predicted[i] = hit
            if hit:
                index.range_search(v, t * t)
        time_with = time.perf_counter() - start

        n_index_accesses = int(predicted.sum())
        rates = _confusion_rates(predicted.tolist(), actual)
        throughput_without = n_q / max(time_without, 1e-9)
        throughput_with = n_q / max(time_with, 1e-9)

        rows.append({
            "dataset": dataset, "k": k, "tau": t, "seed": seed,
            "redundancy_rate": redundancy_rate,
            "tp_rate_pct": redundancy_rate * 100.0,
            "throughput_improvement_pct": (throughput_with - throughput_without) / throughput_without * 100.0,
            "mean_index_accesses": n_index_accesses,
            "n_queries": n_q,
            "positive_rate_pct": n_index_accesses / max(n_q, 1) * 100.0,
            "fpr": rates["fpr"], "fnr": rates["fnr"],
            "n_caps": n_caps,
        })
    return rows


_SURFACE_CHUNK = 2048  # queries per all-balls search in _blocklist_cover_rows


def _blocklist_cover_rows(
    dim: int,
    dataset: str,
    seed: int,
    k_fractions: List[float],
    tau_values: List[float],
    vectors_reference: VectorInput,
    vectors_heldout: VectorInput,
    vectors_withheld: VectorInput,
    vectors_singleton: VectorInput,
    volume_bounds: bool,
) -> List[Dict[str, Any]]:
    """Shared core of blocklist_tau_worker and budget_worker: build a
    CapCoverFilter per k in k_fractions and emit one sweep="tau" row per
    (k, tau) pair. Both callers read only these rows -- see each one's own
    docstring for which axis it treats as fixed and which it sweeps.

    The query stream comes from three sources (QUERY_SOURCES): held-out
    members of the reference set's clusters, members of clusters withheld from
    it, and singletons. Sources decide what is asked, never the answer: a
    query is positive iff its exact distance to the reference set is <= tau,
    recomputed per tau, so these rows mean what quantization_worker's and
    equal_memory_worker's do. Predictions use the filter's own acceptance rule
    (CapCoverFilter.surface_angle, thresholded at 2*arcsin(tau'/2) -- search
    once, threshold per tau), so a row says what query() would.

    FPR is also split by source (_per_source_rates): withheld-cluster
    negatives test whether the cover localizes to B's clusters rather than to
    dense regions in general, singletons are background traffic, and held-out
    cluster-mates farther than tau are the hardest negatives of all, since
    they sit inside their own cluster's ball. A pooled FPR alone would be
    dominated by whichever source is larger.

    volume_bounds adds the log10_vol_high column (_cap_log10_volume). Off by
    default to match blocklist_tau_worker/budget_worker's existing cost
    tradeoff (a whole tau grid per k vs. a single tau per k), even though
    caps have no overlap-aware union bound to make expensive in the first
    place (see _cap_log10_volume's docstring) -- log10_vol_low stays NaN
    always, there being no cap counterpart of that lower bound.
    """
    B = np.ascontiguousarray(_to_construct_array(vectors_reference), dtype=np.float32)
    by_source = [np.ascontiguousarray(_to_construct_array(v), dtype=np.float32)
                 for v in (vectors_heldout, vectors_withheld, vectors_singleton)]
    sizes = dict(zip(("n_heldout", "n_withheld", "n_singleton"), map(len, by_source)))
    nonempty = [q for q in by_source if len(q)]
    Q = np.concatenate(nonempty) if nonempty else np.zeros((0, dim), dtype=np.float32)
    n_queries = len(Q)

    # Exact nearest-reference distance per query: the ground truth, and the
    # workload's own geometry, reported per source so a row is interpretable
    # without re-deriving it (the cover's margin vs d_withheld).
    bix = faiss.IndexFlatL2(dim)
    bix.add(B)
    if n_queries:
        d2, _ = bix.search(Q, 1)
        nn = np.sqrt(np.clip(d2[:, 0], 0, None))
    else:
        nn = np.zeros(0, dtype=np.float32)
    heldout_nn = nn[:sizes["n_heldout"]]
    withheld_nn = nn[sizes["n_heldout"]:sizes["n_heldout"] + sizes["n_withheld"]]
    d_heldout_med = float(np.median(heldout_nn)) if len(heldout_nn) else float("nan")
    d_withheld_med = float(np.median(withheld_nn)) if len(withheld_nn) else float("nan")

    rows: List[Dict[str, Any]] = []

    for kf in k_fractions:
        k = min(max(1, round(kf * len(B))), len(B))
        filt = _cap_cover(dim, k)
        filt.construct(B)
        angles = filt.all_angles

        # Searched once per cover and thresholded per tau; query() applies the
        # same comparison, so these are exactly its predictions. Chunked: the
        # search scores every cap, and all 20K queries against 16K caps at
        # once is ~7 GB per job, too much with a full pool of workers.
        surface = np.concatenate(
            [filt.surface_angle(Q[i:i + _SURFACE_CHUNK]) for i in range(0, n_queries, _SURFACE_CHUNK)]
        ) if n_queries else np.zeros(0)
        psi_med = float(np.median(angles)) if len(angles) else 0.0

        for tau in tau_values:
            tau = float(tau)
            # Labels are recomputed per tau (the paper's y(q, tau)), and so
            # are the geometric objective and the volume bound, since tau
            # expands every cap (eff. angle theta_i + beta(tau)).
            actual = nn <= tau
            tau_prime = tau + CapCoverFilter._COVERAGE_EPSILON + CapCoverFilter._NORM_TOLERANCE
            predicted = surface <= 2.0 * np.arcsin(np.clip(tau_prime / 2.0, 0.0, 1.0))
            fn = int(np.sum(actual & ~predicted))
            if fn:
                raise ValueError(
                    f"Zero-false-negative invariant violated: {fn} FN "
                    f"(dataset={dataset}, fold={seed}, k={len(angles)}, tau={tau})")
            rates = _confusion_rates(predicted.tolist(), actual)
            saved = 100.0 * (1.0 - predicted.mean()) if n_queries else 0.0

            # log10_vol is a logsumexp over the caps, i.e. O(k) -- the same
            # cost class as Ball-Cover's "always reported" log10_j_tau, which
            # it directly replaces; it sums each cap's measure independently
            # and double-counts overlap, same caveat as log10_j_tau always
            # had. Unlike Ball-Cover there is no overlap-aware union bound
            # for caps to opt into (_cap_log10_volume's docstring), so
            # volume_bounds here only gates whether this same number is also
            # exposed as log10_vol_high (preserving that column's "opt-in"
            # meaning for downstream consumers), not any extra computation.
            log10_vol = _cap_log10_volume(filt, tau)
            with np.errstate(over='ignore'):
                j_tau = 0.0 if log10_vol == float('-inf') else float(10.0 ** log10_vol)
            log10_vol_high = log10_vol if volume_bounds else float('nan')

            rows.append({
                "dataset": dataset, "seed": seed, "sweep": "tau",
                "k": len(angles), "n_reference": len(B),
                # k_fraction is the nominal knob and is identical across
                # datasets and folds; `compression` is what it realised here,
                # which is not (|B| differs per dataset and per fold, so k =
                # round(kf*|B|) lands on a different ratio each time). Group
                # and filter on k_fraction -- grouping on compression makes
                # every fold its own singleton group, which silently drops the
                # error bars and matches only one dataset when filtering.
                "k_fraction": float(kf),
                "compression": len(B) / max(len(angles), 1),
                "psi_med": psi_med, "psi_max": float(angles.max()) if len(angles) else 0.0,
                "tau": tau,
                "fpr": rates["fpr"], "fnr": rates["fnr"],
                # fpr_<source>, n_neg_<source>, n_pos, and neg_share_pct --
                # the ceiling on accesses_saved_pct for a zero-FN filter.
                **_per_source_rates(predicted, actual, **sizes),
                **sizes,
                "n_queries": n_queries,
                "accesses_saved_pct": saved,
                "j_tau": j_tau,
                "log10_j_tau": log10_vol,
                "log10_vol_low": float('nan'),
                "log10_vol_high": log10_vol_high,
                "n_caps": len(angles),
                "d_heldout_med": d_heldout_med, "d_withheld_med": d_withheld_med,
                "memory_bytes": filt.get_bytesize(),
            })
    return rows


def blocklist_tau_worker(
    dim: int,
    dataset: str,
    seed: int,
    k_fraction: float,
    tau_values: List[float],
    vectors_reference: VectorInput,
    vectors_heldout: VectorInput,
    vectors_withheld: VectorInput,
    vectors_singleton: VectorInput,
) -> List[Dict[str, Any]]:
    """Asymmetric blocklist workload (main.generate_blocklist): a tight
    reference set B and a broad query stream drawn from held-out members of
    B's own duplicate clusters, withheld clusters and singletons, labeled by
    distance to B at each tau.

    Cap budget k is fixed (a single k_fraction, not a sweep -- varying it is
    budget_worker's job); tau is swept over tau_values so FPR/accesses saved
    are read as one corpus's response curve at that one budget.
    """
    return _blocklist_cover_rows(
        dim, dataset, seed, [k_fraction], tau_values,
        vectors_reference, vectors_heldout, vectors_withheld, vectors_singleton,
        volume_bounds=False,
    )


def budget_worker(
    dim: int,
    dataset: str,
    seed: int,
    k_fractions: List[float],
    tau: float,
    vectors_reference: VectorInput,
    vectors_heldout: VectorInput,
    vectors_withheld: VectorInput,
    vectors_singleton: VectorInput,
) -> List[Dict[str, Any]]:
    """The same asymmetric blocklist workload as blocklist_tau_worker
    (main.generate_budget), read along the other axis: tau is fixed at a
    single value and the cap budget k is swept over k_fractions.

    Requests log10_vol_high (see _blocklist_cover_rows's volume_bounds):
    exposed here, where it is paid once per budget, rather than in
    blocklist_tau_worker, which would otherwise multiply the same O(k)
    computation out over a whole tau grid.
    """
    return _blocklist_cover_rows(
        dim, dataset, seed, k_fractions, [tau],
        vectors_reference, vectors_heldout, vectors_withheld, vectors_singleton,
        volume_bounds=True,
    )


def query_performance_worker(
    dim: int,
    dataset: str,
    k: int,
    seed: int,
    query_tau: float,
    vectors_construct: VectorInput,
    vectors_query: VectorInput,
    single_query_limit: int = 2000,
) -> List[Dict[str, Any]]:
    """Times one query at a time vs. batched queries against the same cover;
    single-query mode is capped at single_query_limit, batched is uncapped."""
    construct_arr = _to_construct_array(vectors_construct)

    filt = _cap_cover(dim, k)
    filt.construct(construct_arr)

    rows: List[Dict[str, Any]] = []

    n_single = 0
    start = time.perf_counter()
    for v in _iter_query_vectors(vectors_query):
        if n_single >= single_query_limit:
            break
        filt.query(v[np.newaxis, :], distance_threshold=query_tau)
        n_single += 1
    elapsed_single = time.perf_counter() - start
    rows.append({
        "dataset": dataset, "k": k, "seed": seed, "mode": "single",
        "n_queries": n_single,
        "latency_ms": elapsed_single / max(n_single, 1) * 1000,
        "throughput_qps": n_single / max(elapsed_single, 1e-9),
    })

    n_batched = 0
    start = time.perf_counter()
    for block in _iter_query_batches(vectors_query, _QUERY_BATCH):
        filt.query(block, distance_threshold=query_tau)
        n_batched += len(block)
    elapsed_batched = time.perf_counter() - start
    rows.append({
        "dataset": dataset, "k": k, "seed": seed, "mode": "batched",
        "n_queries": n_batched,
        "latency_ms": elapsed_batched / max(n_batched, 1) * 1000,
        "throughput_qps": n_batched / max(elapsed_batched, 1e-9),
    })

    return rows


def scaling_query_worker(
    dim: int,
    dataset: str,
    k: int,
    batch_size: int,
    seed: int,
    tau_mid: float,
    vectors_construct: VectorInput,
    vectors_query: VectorInput,
    n_query_vectors: int = 8192,
) -> Dict[str, Any]:
    """Measures batched-query throughput for a CapCoverFilter at filter
    size k, querying in batches of `batch_size`. Every batch_size times the
    same n_query_vectors, so the panel's lines are directly comparable.

    tau_mid is part of the measurement: query() abandons a ball once a
    partial distance passes r + tau, and stops a query at the first ball
    that accepts it (see CapCoverFilter's scan kernels), so a
    larger tau means more work per ball and earlier exits. Throughput is
    therefore reported at this one threshold, not for every tau."""
    construct_arr = _to_construct_array(vectors_construct)
    query_arr = _to_construct_array(vectors_query)
    if len(query_arr) < n_query_vectors:
        raise ValueError(
            f"scaling_query_worker needs {n_query_vectors} query vectors but the holdout "
            f"holds {len(query_arr)}. Widen the holdout in generate_scaling() (it is sized "
            f"max(n_query_vectors, n_insert_vectors)) or lower n_query_vectors."
        )

    filt = _cap_cover(dim, k)
    filt.construct(construct_arr)

    stream = query_arr[:n_query_vectors]

    start = time.perf_counter()
    for i in range(0, len(stream), batch_size):
        filt.query(stream[i:i + batch_size], distance_threshold=tau_mid)
    elapsed = time.perf_counter() - start

    return {
        "dataset": dataset, "k": k, "batch_size": batch_size, "seed": seed,
        "n_queries": len(stream),
        "throughput_qps": len(stream) / max(elapsed, 1e-9),
    }


def scaling_insert_worker(
    dim: int,
    dataset: str,
    k: int,
    batch_size: int,
    seed: int,
    vectors_construct: VectorInput,
    vectors_insert: VectorInput,
    n_insert_vectors: int = 8192,
) -> Dict[str, Any]:
    """Like scaling_query_worker, but times insert() in batches of
    `batch_size` after the initial construct()."""
    construct_arr = _to_construct_array(vectors_construct)
    insert_arr = _to_construct_array(vectors_insert)
    if len(insert_arr) < n_insert_vectors:
        raise ValueError(
            f"scaling_insert_worker needs {n_insert_vectors} insert vectors but the holdout "
            f"holds {len(insert_arr)}. Widen the holdout in generate_scaling() (it is sized "
            f"max(n_query_vectors, n_insert_vectors)) or lower n_insert_vectors."
        )

    filt = _cap_cover_insertable(dim, k)
    filt.construct(construct_arr)

    stream = insert_arr[:n_insert_vectors]

    start = time.perf_counter()
    for i in range(0, len(stream), batch_size):
        filt.insert(stream[i:i + batch_size])
    elapsed = time.perf_counter() - start

    return {
        "dataset": dataset, "k": k, "batch_size": batch_size, "seed": seed,
        "n_inserted": len(stream),
        "throughput_ips": len(stream) / max(elapsed, 1e-9),
    }


def scaling_construction_worker(
    dim: int,
    dataset: str,
    k: int,
    n: int,
    seed: int,
    vectors_construct: VectorInput,
) -> Dict[str, Any]:
    """Measures construction throughput for a CapCoverFilter at filter size
    k built over n indexed vectors -- swept over a (k, n) grid."""
    construct_arr = _to_construct_array(vectors_construct)[:n]

    start = time.perf_counter()
    filt = _cap_cover(dim, k)
    filt.construct(construct_arr)
    elapsed = time.perf_counter() - start

    return {
        "dataset": dataset, "k": k, "n": n, "seed": seed,
        "construct_time_s": elapsed,
        "throughput_vps": n / max(elapsed, 1e-9),
    }


def _rebuild_policy_tags(policy: str, insertion_level: Optional[str]) -> tuple:
    """(metric, method) tag used by the streaming figure. "oracle" always
    gets one fixed reference tag; "streaming"/"periodic" report whichever
    insertion_level they ran, so all methods plot as directly comparable
    lines rather than one needing a special reference tag of its own."""
    if policy == "oracle":
        return "static_reference", "static_rebuild"
    return "checkpoint", insertion_level or "nearest"


def stream_rebuild_worker(
    dim: int,
    dataset: str,
    k: int,
    seed: int,
    tau: float,
    policy: str,
    vectors_stream: VectorInput,
    vectors_query: VectorInput,
    actual_at_checkpoint: Dict[int, List[bool]],
    n_heldout: int,
    n_withheld: int,
    n_singleton: int,
    checkpoints: List[int],
    T: Optional[int] = None,
    insertion_level: Optional[str] = None,
    sweep: Optional[str] = None,
    fold_offset: Optional[int] = None,
    bootstrap_n: Optional[int] = None,
    construct_method: Optional[str] = None,
    stream_order: Optional[str] = None,
    insert_memory_percent: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Simulates one rebuild policy over vectors_stream (a blocklist fold's
    reference set B, in arrival order -- see main.generate_streaming),
    reporting query FPR and amortized cost per checkpoint.
      "streaming": CapCoverFilter.insert() per arrival, never rebuilt.
      "periodic": like "streaming", rebuilt from scratch every T arrivals.
      "oracle": rebuilt from scratch at every checkpoint.
    sweep/fold_offset/stream_order are pass-through tag columns only.

    construct_method names a CAP_CONSTRUCT_METHODS entry (default: the
    filter's own, SphereMergeConstruct) and is used for every construct()
    the policy makes: the oracle's rebuilds, and the streaming/periodic
    bootstrap. insertion_level sweeps CAP_INSERT_STRATEGIES, which only has
    2 members (cap_merge/nearest).

    Memory is held to one budget for every method: max_memory is the static
    filter's footprint at k caps, and CapCoverFilter sizes each insert
    strategy's cap count so that its peak get_bytesize() between insert()
    calls fits it (see InsertInterface.max_caps_for_budget).
    peak_memory_bytes is that peak as measured over the stream so far, next
    to budget_bytes. max_insert_s is the slowest single insert() so far, the
    spike a rebuilding strategy trades for its amortized cost, and
    insert_time_s the insert-only total (cumulative_time_s also counts the
    bootstrap construct()).

    vectors_query is the fold's [held-out, withheld, singleton] query stream
    (see main.BlocklistWorkload.query_vectors), sized by n_heldout/n_withheld/
    n_singleton, with fpr split by source exactly as quantization_worker
    does. actual_at_checkpoint[c] is its distance-based
    ground truth against vectors_stream[:c] -- computed once by the caller,
    since every policy and method on a fold shares it.

    bootstrap_n sizes the initial construct() window independently of the
    ball budget k -- default min(k, n), which for k <= n gives
    GonzalesConstruct one input point per center and hence zero-radius balls
    at position 0. Pass an explicit bootstrap_n (e.g. an order of magnitude
    larger than k) to start from a filter with real coverage instead. Both
    policies checkpoint only past it, and report position as arrivals since
    it, so their rows line up."""
    stream_arr = _to_construct_array(vectors_stream)
    query_arr = _to_construct_array(vectors_query)
    n = len(stream_arr)
    n_q = len(query_arr)
    if n_heldout + n_withheld + n_singleton != n_q:
        raise ValueError(f"Source sizes ({n_heldout}, {n_withheld}, {n_singleton}) do not "
                         f"partition n_queries ({n_q})")
    metric, method = _rebuild_policy_tags(policy, insertion_level)
    # By default pooled reserves a 20% pool and every other method constructs
    # at full capacity; insert_memory_percent overrides that for any method.
    if insert_memory_percent is None:
        insert_memory_percent = 20.0 if insertion_level == "pooled" else 0.0
    bootstrap_n = min(bootstrap_n if bootstrap_n is not None else k, n)
    checkpoint_set = sorted(set(c for c in checkpoints if bootstrap_n < c <= n))
    missing = [c for c in checkpoint_set if c not in actual_at_checkpoint]
    if missing:
        raise ValueError(f"actual_at_checkpoint has no ground truth for checkpoint(s) {missing}")
    if construct_method is not None and construct_method not in CAP_CONSTRUCT_METHODS:
        raise ValueError(f"Unknown construct_method {construct_method!r}; expected one of "
                         f"{sorted(CAP_CONSTRUCT_METHODS)}")
    budget_bytes = _cover_max_memory(dim, k)

    def _new_filter(insert_method: Any = None, **kwargs: Any) -> CapCoverFilter:
        if construct_method is not None:
            kwargs["construct_method"] = CAP_CONSTRUCT_METHODS[construct_method]()
        return CapCoverFilter(dim=dim, max_memory=budget_bytes, insert_method=insert_method, **kwargs)

    peak_memory_bytes = 0
    max_insert_s = 0.0
    insert_time_s = 0.0

    def _checkpoint_row(filt: CapCoverFilter, pos: int, cumulative_time_s: float) -> Dict[str, Any]:
        """Measures one checkpoint. Raises rather than recording a false
        negative, like every other FNR check here: every stream vector seen
        so far must still be recalled, and so must every query the ground
        truth says is within tau of them."""
        recalled = np.asarray(filt.query(stream_arr[:pos], distance_threshold=tau), dtype=bool)
        actual = np.asarray(actual_at_checkpoint[pos], dtype=bool)
        predicted = np.asarray(filt.query(query_arr, distance_threshold=tau), dtype=bool)
        rates = _confusion_rates(predicted, actual)
        if not recalled.all() or rates["fnr"] != 0.0:
            raise InvariantViolation(
                f"Zero-false-negative invariant violated (dataset={dataset}, policy={policy}, "
                f"insertion_level={insertion_level}, k={k}, seed={seed}, position={pos}): "
                f"stream fnr={1.0 - recalled.mean()}, query fnr={rates['fnr']}"
            )
        n_inserted = pos - bootstrap_n
        streamed = policy != "oracle" and n_inserted > 0
        return {
            "dataset": dataset, "policy": policy, "T": T, "k": k, "seed": seed,
            "position": pos - bootstrap_n, "tau": tau,
            "fpr": rates["fpr"], "fnr": rates["fnr"],
            # fpr_heldout: held-out cluster-mates farther than tau from what
            # has arrived, which sit inside their cluster's ball.
            **_per_source_rates(predicted, actual, n_heldout, n_withheld, n_singleton),
            "n_queries": n_q, "n_duplicates": int(actual.sum()),
            "log10_j_tau": _cap_log10_volume(filt, tau),
            "n_caps": filt._num_caps,
            "memory_bytes": filt.get_bytesize(),
            "peak_memory_bytes": max(peak_memory_bytes, filt.get_bytesize()),
            "budget_bytes": budget_bytes,
            "cumulative_time_s": cumulative_time_s,
            "amortized_cost_s": cumulative_time_s / pos,
            "insert_time_s": insert_time_s if streamed else float("nan"),
            "amortized_insert_s": insert_time_s / n_inserted if streamed else float("nan"),
            "max_insert_s": max_insert_s if streamed else float("nan"),
            "construct_method": construct_method or CAP_COVER_CONSTRUCT,
            "stream_order": stream_order,
            "insert_memory_percent": insert_memory_percent,
            "metric": metric, "method": method, "sweep": sweep, "fold_offset": fold_offset,
        }

    rows: List[Dict[str, Any]] = []
    cumulative_time_s = 0.0

    if policy == "oracle":
        for c in checkpoint_set:
            filt = _new_filter()
            t0 = time.perf_counter()
            filt.construct(stream_arr[:c])
            cumulative_time_s += time.perf_counter() - t0
            rows.append(_checkpoint_row(filt, c, cumulative_time_s))
        return rows

    if policy not in ("streaming", "periodic"):
        raise ValueError(f"Unknown policy {policy!r}; expected 'streaming', 'periodic', or 'oracle'")

    if insertion_level not in CAP_INSERT_STRATEGIES:
        raise ValueError(
            f"Unknown insertion_level {insertion_level!r}; expected one of "
            f"{sorted(CAP_INSERT_STRATEGIES)}"
        )
    rebuild_every = T if (policy == "periodic" and T is not None) else None
    filt = _new_filter(insert_method=CAP_INSERT_STRATEGIES[insertion_level](),
                       insert_memory_percent=insert_memory_percent)
    # Bootstrap with construct() on the same vectors and construct_method for
    # every insertion_level, so every checkpoint from bootstrap_n onward
    # compares each method's own decision rule -- at its own ball count, since
    # each strategy's bookkeeping takes its share of the one byte budget.
    t0 = time.perf_counter()
    filt.construct(stream_arr[:bootstrap_n])
    cumulative_time_s += time.perf_counter() - t0
    peak_memory_bytes = filt.get_bytesize()

    ci = 0
    for t in range(bootstrap_n, n):
        pos = t + 1
        t0 = time.perf_counter()
        if rebuild_every is not None and pos % rebuild_every == 0:
            filt.construct(stream_arr[:pos])
        else:
            filt.insert(stream_arr[t][np.newaxis, :])
        dt = time.perf_counter() - t0
        cumulative_time_s += dt
        insert_time_s += dt
        max_insert_s = max(max_insert_s, dt)
        peak_memory_bytes = max(peak_memory_bytes, filt.get_bytesize())

        if ci < len(checkpoint_set) and pos == checkpoint_set[ci]:
            rows.append(_checkpoint_row(filt, pos, cumulative_time_s))
            ci += 1

    return rows


_WORKER_FNS: Dict[str, Any] = {
    'cap_cover_system_tau': cap_cover_system_tau_worker,
    'blocklist': blocklist_tau_worker,
    # Same underlying cover-and-margin core (_blocklist_cover_rows), second
    # worker: generate_budget runs the identical blocklist measurement but
    # sweeps the ball budget at a fixed tau instead of sweeping tau at a fixed
    # budget (see main.generate_budget).
    'budget': budget_worker,
    'cap_quantization': cap_quantization_worker,
    'query_performance': query_performance_worker,
    'equal_memory': equal_memory_worker,
    'scaling_query': scaling_query_worker,
    'scaling_insert': scaling_insert_worker,
    'scaling_construction': scaling_construction_worker,
    'stream_rebuild': stream_rebuild_worker,
    'construct_method': construct_method_worker,
    'construct_method_scale': construct_method_scale_worker,
}


def _describe_job(kwargs: Dict[str, Any]) -> str:
    if 'filter_cls' in kwargs:
        return kwargs['filter_cls'].__name__
    if {'dataset', 'k', 'tau_values', 'seed'} <= kwargs.keys():
        return f"{kwargs['dataset']} k={kwargs['k']} tau_values={kwargs['tau_values']} seed={kwargs['seed']}"
    if {'dataset', 'k', 'tau', 'seed'} <= kwargs.keys():
        return f"{kwargs['dataset']} k={kwargs['k']} tau={kwargs['tau']:g} seed={kwargs['seed']}"
    if {'dataset', 'method', 'k', 'seed'} <= kwargs.keys():
        fold = f" fold_offset={kwargs['fold_offset']}" if 'fold_offset' in kwargs else ""
        return f"{kwargs['dataset']} method={kwargs['method']} k={kwargs['k']} seed={kwargs['seed']}{fold}"
    if {'dataset', 'policy', 'k', 'seed'} <= kwargs.keys():
        extra = f" T={kwargs['T']}" if kwargs.get('T') is not None else ""
        return f"{kwargs['dataset']} policy={kwargs['policy']}{extra} k={kwargs['k']} seed={kwargs['seed']}"
    if {'dataset', 'k', 'seed'} <= kwargs.keys():
        return f"{kwargs['dataset']} k={kwargs['k']} seed={kwargs['seed']}"
    return "job"


def _describe_result(result: Any) -> str:
    if isinstance(result, list):
        if not result:
            return "0 rows"
        first = result[0]
        return (f"{first.get('dataset', '?')} k={first.get('k', '?')} tau={first.get('tau', '?')} "
                f"seed={first.get('seed', '?')} ({len(result)} rows)")
    if isinstance(result, dict):
        config = result.get('Config', '?')
        pct = result.get('Percentage')
        return f"{config} @ {pct:.1f}%" if pct is not None else str(config)
    return str(result)


class InvariantViolation(ValueError):
    """A job broke the zero-false-negative / coverage invariant. Unlike an
    ordinary job failure (logged and skipped), Runner.run() re-raises these
    after all jobs finish. Plain message-only class so it pickles across
    ProcessPoolExecutor."""


class Runner:
    def __init__(self, num_workers: int = 1, truncate: bool = True) -> None:
        """truncate=False preserves an output_csv's existing rows (see
        submit()) instead of starting fresh, so a resumed run appends."""
        self.num_workers = num_workers
        self.truncate = truncate
        self._jobs: List[Dict[str, Any]] = []
        self._rows_by_csv: Dict[str, List[Dict[str, Any]]] = {}
        self._initialized_csvs: set = set()
        self._violations: List[str] = []

    def submit(self, mode: str, output_csv: Optional[str] = None, **kwargs: Any) -> None:
        """output_csv, if given, is this job's target CSV: rewritten from
        every row completed so far for that path each time one of its jobs
        completes, so a long-running sweep's progress is visible on disk
        before it finishes. output_csv, not mode, is the routing key --
        several jobs (even of different modes) can share one path, or each
        target its own. A job with no output_csv is only returned by run()."""
        if mode not in _WORKER_FNS:
            raise ValueError(f"Unknown mode {mode!r}. Valid: {list(_WORKER_FNS)}")
        if output_csv is not None and output_csv not in self._initialized_csvs:
            self._init_csv(output_csv)
        self._jobs.append({'mode': mode, 'output_csv': output_csv, **kwargs})

    def _init_csv(self, output_csv: str) -> None:
        """Runs once per distinct output_csv, at first submit(), so
        truncation happens before any job targeting that path has run."""
        self._initialized_csvs.add(output_csv)
        if not self.truncate and os.path.exists(output_csv):
            self._rows_by_csv[output_csv] = pd.read_csv(output_csv).to_dict("records")
        else:
            if os.path.exists(output_csv):
                os.remove(output_csv)
            self._rows_by_csv[output_csv] = []

    def shuffle(self, seed: Optional[int] = None) -> None:
        """Randomizes submitted-job order in place, so several generate_
        functions sharing one Runner interleave fast and slow jobs (e.g.
        PQ construction, tens of minutes each) rather than one worker
        draining an entire slow evaluation while others sit idle. Affects
        only wall-clock makespan, not correctness."""
        random.Random(seed).shuffle(self._jobs)

    def run(self) -> List[Tuple[str, Any]]:
        """Returns (mode, result) pairs, in completion order (parallel) or
        submission order (sequential) -- the mode tag lets a caller that
        shared this Runner across job types split results back apart."""
        self._violations: List[str] = []
        results = self._run_sequential() if self.num_workers == 1 else self._run_parallel()
        if self._violations:
            raise InvariantViolation(
                f"{len(self._violations)} job(s) violated the zero-FN invariant "
                "(other jobs' results were recorded):\n" + "\n".join(self._violations))
        return results

    def clear(self) -> None:
        self._jobs.clear()

    def _record_result(self, output_csv: Optional[str], result: Any) -> None:
        """Appends to output_csv's in-memory accumulator and rewrites that
        file -- called only from the single coordinating process (never a
        worker), so multiple jobs sharing an output_csv never race; no lock
        needed since there's never more than one writer."""
        if not output_csv:
            return
        self._rows_by_csv[output_csv].extend(result if isinstance(result, list) else [result])
        rows = self._rows_by_csv[output_csv]
        if not rows:
            return
        out_dir = os.path.dirname(output_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame(rows).to_csv(output_csv, index=False)

    def _run_sequential(self) -> List[Tuple[str, Any]]:
        results = []
        for job in self._jobs:
            mode = job['mode']
            output_csv = job.get('output_csv')
            fn = _WORKER_FNS[mode]
            kwargs = {k: v for k, v in job.items() if k not in ('mode', 'output_csv')}
            try:
                result = fn(**kwargs)
            except Exception as e:
                if isinstance(e, InvariantViolation):
                    self._violations.append(f"{_describe_job(kwargs)}: {e}")
                # One job's exception must not lose every other job's
                # already-completed results.
                logger.exception(f"Job failed, skipping: {_describe_job(kwargs)}")
                continue
            logger.info(f"Completed: {_describe_result(result)}")
            results.append((mode, result))
            self._record_result(output_csv, result)
        return results

    def _run_parallel(self) -> List[Tuple[str, Any]]:
        futures: Dict[Any, Tuple[str, str, Optional[str]]] = {}
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            for job in self._jobs:
                mode = job['mode']
                output_csv = job.get('output_csv')
                fn = _WORKER_FNS[mode]
                kwargs = {k: v for k, v in job.items() if k not in ('mode', 'output_csv')}
                f = executor.submit(fn, **kwargs)
                futures[f] = (mode, _describe_job(kwargs), output_csv)
            results = []
            for f in as_completed(futures):
                mode, desc, output_csv = futures[f]
                try:
                    result = f.result()
                except Exception as e:
                    if isinstance(e, InvariantViolation):
                        self._violations.append(f"{desc}: {e}")
                    # See _run_sequential's comment.
                    logger.exception(f"Job failed, skipping: {desc}")
                    continue
                logger.info(f"Completed: {desc}")
                results.append((mode, result))
                self._record_result(output_csv, result)
        return results