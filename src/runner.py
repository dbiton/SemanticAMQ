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
from filters.ball_cover.ball_cover_filter import BallCoverFilter, Quantize
from filters.quantized_vector_filter import QuantizedVectorFilter
from filters.ball_cover.nearest_insert import NearestInsert
from filters.ball_cover.least_inflation_insert import LeastInflationInsert
from filters.ball_cover.exhaustive_merge_insert import ExhaustiveMergeInsert
from filters.ball_cover.min_heap_insert import MinHeapInsert
from filters.ball_cover.doubling_insert import DoublingInsert
from hyperspheres_volume import estimate_hypersphere_volume
from vector_source import VectorSource

configure_logging()
logger = logging.getLogger(__name__)

# Classes, not instances -- each job needs its own fresh strategy object.
# "doubling" is registered but opt-in only (not in generate_streaming()'s
# sweep list) since it adds real compute.
_INSERT_STRATEGIES = {
    "nearest": NearestInsert,
    "least_inflation": LeastInflationInsert,
    "merge": ExhaustiveMergeInsert,
    "min_heap": MinHeapInsert,
    "doubling": DoublingInsert,
}

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
    "FirstNFilter", "BallCoverFilter", "KDBoxFilter", "GonzalezBoxFilter",
    "DSBFFilter", "CoveringLSHFallbackFilter", "LSBFFilter", "DSHFilter", "HNSWFilter",
    "IVFFilter", "QuantizedVectorFilter", "MLPFilter"
}


def _ball_cover_max_memory(dim: int, k: int, quantize: Quantize = Quantize.FP32) -> int:
    """BallCoverFilter sizes its ball budget from a byte budget (max_memory),
    not a raw ball count -- this converts a desired count k into the
    max_memory that yields exactly k balls for the given quantize type."""
    bytes_per_ball = BallCoverFilter(dim=dim, max_memory=1, quantize=quantize).bytes_per_ball()
    return bytes_per_ball * k


class _BallCoverFilterEvalAdapter:
    """BallCoverFilter.query() takes an explicit per-call distance_threshold
    rather than baking one in at construction time like the other
    _DIRECT_TAU_FILTERS -- this captures the threshold once so this module's
    generic `filter_instance.query(block)` call sites keep working."""

    def __init__(self, dim: int, max_memory: int, distance_threshold: float,
                 max_spheres: Optional[int] = None, insert_mem_percent: Optional[float] = 0.0, **extra_kwargs: Any) -> None:
        self.distance_threshold = distance_threshold
        quantize = extra_kwargs.get("quantize", Quantize.FP32)
        if max_spheres is not None:
            max_memory = _ball_cover_max_memory(dim, max_spheres, quantize)
        self._filt = BallCoverFilter(dim=dim, max_memory=max_memory, max_spheres=max_spheres, 
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
    def all_radiuses(self) -> np.ndarray:
        return self._filt.all_radiuses


def _instantiate_filter(filter_cls: Any, dim: int, memory_budget: int, distance_threshold: float = None, max_spheres: int = None,
                         insert_mem_percent: Optional[float] = 0.0, extra_kwargs: Optional[Dict[str, Any]] = None) -> Any:
    kwargs = {"dim": dim, "max_memory": memory_budget, **(extra_kwargs or {})}

    if filter_cls.__name__ in _DIRECT_TAU_FILTERS:
        kwargs.update(distance_threshold=distance_threshold)
    
        if filter_cls is BallCoverFilter:
            kwargs.update(max_spheres=max_spheres, insert_mem_percent=insert_mem_percent)
            return _BallCoverFilterEvalAdapter(**kwargs)

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


# Filters with a cheap post-construction query-time dial: score every candidate on one built instance.
# Every dial here must reach the underlying index, not just set a Python
# attribute -- see IVFFilter.nprobe / HNSWFilter.ef_search, both properties
# for exactly that reason.
_POST_CONSTRUCTION_CONFIG_DIALS = {
    "DSBFFilter": lambda f: [("vote_threshold", t) for t in range(1, f.num_tables + 1)],
    # Swept on a log-ish grid up to nlist, not exhaustively (nlist can be in the hundreds).
    "IVFFilter": lambda f: [
        ("nprobe", p) for p in sorted({c for c in _IVF_NPROBE_CANDIDATES if c <= f.nlist} | {f.nlist})
    ],
    # efSearch is HNSW's accuracy/speed dial and costs no memory, exactly
    # like IVF's nprobe. Without it HNSW competed at a fixed efSearch=64
    # against an IVF whose nprobe was tuned per budget -- not an equal
    # comparison of the two baselines.
    "HNSWFilter": lambda f: [("ef_search", e) for e in _HNSW_EF_SEARCH_CANDIDATES],
}

# Filters whose configuration changes memory layout at construction time: one build per candidate.
_CONSTRUCTION_TIME_CONFIG_DIALS = {
    "LSBFFilter": [{"verify": True}, {"verify": False}],
}


def _build_calibrated_filter(
    filter_cls: Any,
    dim: int,
    memory_budget: int,
    distance_threshold: float,
    construct_arr: np.ndarray,
    calib_arr: np.ndarray,
    calib_actual: np.ndarray,
) -> Tuple[Any, float]:
    """Builds filter_cls at memory_budget, picking whichever registered config
    dial candidate scores best (F1) on the calibration set. Returns
    (filter_instance, construction_seconds) -- the latter sums only time
    inside .construct() calls, excluding the calibration .query() calls used
    to score candidates, so it reflects reaching a ready-to-query filter."""
    name = filter_cls.__name__
    construction_time = 0.0

    if name in _CONSTRUCTION_TIME_CONFIG_DIALS:
        best_f1, best_instance = -1.0, None
        for kwargs in _CONSTRUCTION_TIME_CONFIG_DIALS[name]:
            # Via _instantiate_filter's own dispatch, not a hardcoded
            # direct-tau call, so a dial-registry filter not in
            # _DIRECT_TAU_FILTERS still gets built correctly.
            candidate = _instantiate_filter(
                filter_cls, dim, memory_budget, distance_threshold, extra_kwargs=kwargs
            )
            t0 = time.perf_counter()
            candidate.construct(construct_arr)
            construction_time += time.perf_counter() - t0
            f1 = (
                _f1_score(np.asarray(candidate.query(calib_arr), dtype=bool), calib_actual)
                if len(calib_arr) > 0 else 0.0
            )
            if f1 > best_f1:
                best_f1, best_instance = f1, candidate
        return best_instance, construction_time

    filter_instance = _instantiate_filter(filter_cls, dim, memory_budget, distance_threshold)
    t0 = time.perf_counter()
    filter_instance.construct(construct_arr)
    construction_time = time.perf_counter() - t0

    if name in _POST_CONSTRUCTION_CONFIG_DIALS and len(calib_arr) > 0:
        best_attr, best_value, best_f1 = None, None, -1.0
        for attr, value in _POST_CONSTRUCTION_CONFIG_DIALS[name](filter_instance):
            setattr(filter_instance, attr, value)
            f1 = _f1_score(np.asarray(filter_instance.query(calib_arr), dtype=bool), calib_actual)
            if f1 > best_f1:
                best_attr, best_value, best_f1 = attr, value, f1
        if best_attr is not None:
            setattr(filter_instance, best_attr, best_value)

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
) -> Dict[str, Any]:
    config = filter_cls.__name__
    construct_arr = _to_construct_array(vectors_construct)
    calib_arr = _to_construct_array(vectors_calibration)
    calib_actual = np.asarray(actual_calibration, dtype=bool)

    try:
        filter_instance, construction_time = _build_calibrated_filter(
            filter_cls, dim, memory_budget, distance_threshold,
            construct_arr, calib_arr, calib_actual,
        )
    except InsufficientMemoryError as e:
        # Expected at small budgets -- recorded as an explicit infeasible
        # row (NaN metrics) rather than dropped or crashing the sweep. Every
        # filter raises the one shared class from filters.filter; this used
        # to name only QuantizedVectorFilter's copy, so the same condition
        # from BallCoverFilter took the whole job down instead.
        logger.warning(f"[Infeasible] {config} at memory_budget={memory_budget}: {e}")
        return {
            'Config': config, 'Dataset': dataset, 'Memory Budget': memory_budget,
            'Memory Usage': float('nan'), 'Over Budget': False,
            'Threshold': distance_threshold,
            'FPR': float('nan'), 'FNR': float('nan'),
            'Latency (ms)': float('nan'), 'Throughput (qps)': float('nan'),
            'Infeasible': True,
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
    }


_LN10 = float(np.log(10.0))


def _log_effective_radii(radii: np.ndarray, tau: float) -> np.ndarray:
    """Natural log of each ball's tau-expanded radius, carried in log space --
    -inf wherever that radius is not positive.

    tau is not always positive: blocklist_worker solves its operating point as
    a percentile of the query margins (distance-to-centre minus radius), and a
    query strictly inside a ball has a negative margin, so the solved tau can
    come out slightly negative -- a stricter "inside by a margin" threshold.
    A ball whose radius that drives to zero or below covers nothing, and -inf
    is the log-space statement of exactly that: logsumexp maps it back to a
    zero contribution, so it drops out of the sum without ever being
    materialised as a radius.

    Kept in log space rather than clamping the radius to 0, because clamping
    rewrites the geometry (the sum would be over a cover that does not exist)
    to work around a representation limit. Taking the log first also avoids
    both failure modes of the linear form: np.log of a negative radius is NaN
    ("invalid value encountered in log"), and r**d silently returns a LARGE
    POSITIVE number when d is even -- which would make a ball that covers
    nothing the dominant term.
    """
    r = np.asarray(radii, dtype=np.float64) + float(tau)
    logr = np.full(r.shape, -np.inf, dtype=np.float64)
    # where= leaves the -inf in place for non-positive radii and never
    # evaluates the log there, so no invalid-value warning is raised.
    np.log(r, out=logr, where=r > 0.0)
    return logr


def _log10_J(radii: np.ndarray, dim: int, tau: float = 0.0) -> float:
    """log10 sum_i (r_i + tau)^d, safe for d in the hundreds where r^d over/underflows float64."""
    logr = _log_effective_radii(radii, tau)
    if logr.size == 0:
        return float('-inf')
    return float(logsumexp(dim * logr)) / _LN10


def _J_tau(radii: np.ndarray, dim: int, tau: float = 0.0) -> float:
    """Linear J_tau(C) = sum_i (r_i + tau)^d; may overflow to +inf at d~384, r>~1 -- use _log10_J to plot."""
    if len(radii) == 0:
        return 0.0
    # Derived from the log form rather than summed directly, so a ball with a
    # non-positive tau-expanded radius contributes nothing instead of a large
    # positive r**d (d is even). Still overflows to +inf at d~384, r>~1 --
    # that is what _log10_J is for.
    log10_j = _log10_J(radii, dim, tau)
    if log10_j == float('-inf'):
        return 0.0
    with np.errstate(over='ignore'):
        return float(10.0 ** log10_j)


def _tau_volume_bounds_log10(centers: np.ndarray, radii: np.ndarray, tau: float) -> Tuple[float, float]:
    """Log10 (lower, upper) bounds on the volume actually covered by the
    tau-expanded balls (radius + tau), via estimate_hypersphere_volume's
    overlap-aware union bound -- unlike _log10_J's naive sum-of-volumes proxy,
    which ignores overlap entirely. High is floored at low the same way
    _fpvr_bounds floored its log-ratio at 0: floating-point error in the
    lens/Bonferroni terms can otherwise invert them by a hair.

    This is the only home of that estimate. BallCoverFilter.construct() used
    to run it eagerly at tau=0 and return the bounds, so every construct() in
    the suite paid for an O(k^2) union bound that almost no caller read;
    callers that want it now ask for it here, from all_centers/all_radiuses,
    at whatever tau they are reporting (pass tau=0.0 for the cover's own
    volume). Only system_tau_worker does -- see blocklist_worker for a worker
    that deliberately skips it on cost grounds."""
    if len(radii) == 0:
        return float('nan'), float('nan')
    # This estimator takes radii, not log-radii, so the degenerate case is
    # expressed as a zero-radius ball -- which is the correct geometry for one
    # that covers nothing, not a clamp working around a representation limit.
    eff_radii = np.maximum(np.asarray(radii, dtype=np.float64) + float(tau), 0.0)
    log_low, log_high = estimate_hypersphere_volume(centers, eff_radii, return_log=True)
    if np.isnan(log_low) or np.isinf(log_low):
        return float('nan'), float('nan')
    log_low /= _LN10
    log_high = max(log_high / _LN10, log_low)
    return log_low, log_high


def _confusion_rates(predicted: List[bool], actual: np.ndarray) -> Dict[str, float]:
    tp, fp, fn, tn = _binary_confusion(predicted, actual)
    return {
        'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        'fnr': fn / (fn + tp) if (fn + tp) > 0 else 0.0,
    }


# NONE is the raw/unquantized baseline point in this sweep (tagged
# stage="ball"): one "ball" cover to compare every real quantizer against.
QUANTIZATION_SWEEP = (
    Quantize.FP32, Quantize.FP16, Quantize.SQ8, Quantize.SQ6, Quantize.SQ4,
    Quantize.TQ3, Quantize.PQ4_R2, Quantize.PQ8_R2,
    # The ladder stops at 2 b/dim: both 1 b/dim rungs are left out.
    # Quantize.PQ4_R1 halves PQ4_R2's subquantizer count at the same 4-bit
    # width, so it buys its extra balls with reconstruction error that
    # inflates every radius -- excluded as below the useful rate.
    # Quantize.RABITQ is the one rung whose search() is not exact ADC, so its
    # zero-false-negative property rests on a measured one-sided estimator
    # rather than a proof -- see the CAVEAT on Quantize.RABITQ.
    # Both stay constructible, just not swept.
)


def quantization_worker(
    dim: int,
    dataset: str,
    k: int,
    quantize: Quantize,
    tau: float,
    seed: int,
    vectors_construct: VectorInput,
    vectors_query: VectorInput,
    actual_results: List[bool],
) -> List[Dict[str, Any]]:
    """Builds one BallCoverFilter cover per (dataset, k, seed, quantize) and
    evaluates it at a single tau. Returns one row, tagged
    stage="ball"/"quantizing".

    Unlike system_tau_worker, this sweep is single-tau by design: it varies
    the quantizer at a fixed threshold (the paper's Figure 3 fixes tau so 5%
    of queries are duplicates), so the tau-dependent columns (fpr,
    log10_J_tau, index_accesses_saved_pct) are unambiguous without the plots
    having to pick one tau out of several.

    max_memory is derived from k using Quantize.FP32's byte-per-ball size and
    held fixed across every quantize candidate, so a more compact code packs
    more balls into the same budget -- a quantizer compact enough to
    saturate at n_construct balls before using the full budget (e.g. PQ)
    will show flat memory usage across the k sweep, which is real
    information, not a plotting artifact.

    FNR is asserted (raises on violation), never plotted. If construction
    itself raises (e.g. InsufficientMemoryError, too tight a budget for PQ's
    minimum codebook), that candidate is logged and contributes zero rows
    rather than aborting the whole sweep."""
    construct_arr = _to_construct_array(vectors_construct)
    query_arr = _to_construct_array(vectors_query)
    n_q = len(query_arr)
    if len(actual_results) != n_q:
        raise ValueError(
            f"vectors_query ({n_q}) and actual_results ({len(actual_results)}) length mismatch"
        )

    actual = np.asarray(actual_results, dtype=bool)
    stage = "ball" if quantize == Quantize.FP32 else "quantizing"

    rows: List[Dict[str, Any]] = []

    max_memory = _ball_cover_max_memory(dim, k, Quantize.FP32)
    t0 = time.perf_counter()
    try:
        filt = BallCoverFilter(dim=dim, max_memory=max_memory, quantize=quantize)
        filt.construct(construct_arr)
    except Exception as e:
        logger.warning(
            f"Skipping quantization job (dataset={dataset}, k={k}, quantize={quantize.value}, "
            f"seed={seed}): construction raised {type(e).__name__}: {e}"
        )
        return []
    stage_time = time.perf_counter() - t0
    radii = filt.all_radiuses
    log10_J = _log10_J(radii, dim)

    radius_inflation = filt.all_radius_inflation
    radius_inflation_mean = float(np.mean(radius_inflation)) if len(radius_inflation) else 0.0
    radius_inflation_std = float(np.std(radius_inflation)) if len(radius_inflation) else 0.0

    memory_bytes = filt.get_bytesize()

    q0 = time.perf_counter()
    predicted = filt.query(query_arr, distance_threshold=tau)
    query_time = time.perf_counter() - q0

    # What the filter buys a downstream system: a query it rejects never
    # reaches the main index, so accesses == positives and everything else
    # is saved. Same accounting as system_tau_worker's mean_index_accesses,
    # counted rather than timed -- no index is built here, since the count
    # is a property of the filter's decisions, not of what sits behind it.
    # Given the zero-false-negative guarantee this is determined by fpr and
    # the duplicate rate, but it is the number a system integrator reads,
    # and fpr alone doesn't give it without knowing the redundancy rate.
    predicted_arr = np.asarray(predicted, dtype=bool)
    n_index_accesses = int(predicted_arr.sum())

    rates = _confusion_rates(predicted, actual)
    rows.append({
        "dataset": dataset, "k": k, "quantize": quantize.value, "stage": stage, "tau": tau,
        "log10_J": log10_J,
        "log10_J_tau": _log10_J(radii, dim, tau=tau),
        "fpr": rates["fpr"], "fnr": rates["fnr"],
        "stage_time_s": stage_time, "cum_time_s": stage_time,
        "query_throughput_qps": n_q / max(query_time, 1e-9),
        "n_queries": n_q,
        "n_index_accesses": n_index_accesses,
        "index_accesses_saved": n_q - n_index_accesses,
        "index_accesses_saved_pct": (1.0 - n_index_accesses / max(n_q, 1)) * 100.0,
        "n_balls": filt._num_spheres, "memory_bytes": memory_bytes, "seed": seed,
        "radius_inflation_mean": radius_inflation_mean,
        "radius_inflation_std": radius_inflation_std,
    })

    bad = [r for r in rows if r["fnr"] != 0]
    if bad:
        raise ValueError(
            f"Zero-false-negative invariant violated (dataset={dataset}, k={k}, quantize={quantize}, seed={seed}): "
            f"{bad[:3]}"
        )

    return rows


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


def system_tau_worker(
    dim: int,
    dataset: str,
    k: int,
    tau_values: List[float],
    seed: int,
    vectors_construct: VectorInput,
    vectors_query: VectorInput,
    actual_results_by_tau: Dict[float, List[bool]],
) -> List[Dict[str, Any]]:
    """Builds one BallCoverFilter cover (fixed k) plus a flat L2 index over
    the same construct set, then sweeps the query-time threshold tau over
    that single construction, measuring filtered vs. unfiltered
    throughput/index-access/FPR/FNR at each tau. Queries are issued one at a
    time on both arms, reflecting a realistic online setting where the
    filtered path only falls through to index.search() on a possible match.
    redundancy_rate is the measured (not nominal) duplicate fraction.

    Also reports log10_vol_low/log10_vol_high: overlap-aware lower/upper
    bounds on the volume covered by the tau-expanded cover, from the same
    estimator BallCoverFilter.construct() uses for its own volume bounds
    (see _tau_volume_bounds_log10) -- a tighter picture than log10_j_tau's
    naive sum-of-ball-volumes, which double-counts overlapping regions."""
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

    filt = BallCoverFilter(dim=dim, max_memory=_ball_cover_max_memory(dim, k))
    filt.construct(construct_arr)
    radii = filt.all_radiuses
    centers = filt.all_centers

    rows: List[Dict[str, Any]] = []
    for t in tau_values:
        actual = np.asarray(actual_results_by_tau[t], dtype=bool)
        redundancy_rate = float(np.mean(actual)) if len(actual) > 0 else 0.0

        start = time.perf_counter()
        for i in range(n_q):
            index.search(query_arr[i:i + 1], 1)
        time_without = time.perf_counter() - start

        predicted = np.empty(n_q, dtype=bool)
        start = time.perf_counter()
        for i in range(n_q):
            v = query_arr[i:i + 1]
            hit = filt.query(v, distance_threshold=t)[0]
            predicted[i] = hit
            if hit:
                index.search(v, 1)
        time_with = time.perf_counter() - start

        n_index_accesses = int(predicted.sum())
        rates = _confusion_rates(predicted.tolist(), actual)
        throughput_without = n_q / max(time_without, 1e-9)
        throughput_with = n_q / max(time_with, 1e-9)
        vol_log10_low, vol_log10_high = _tau_volume_bounds_log10(centers, radii, t)

        rows.append({
            "dataset": dataset, "k": k, "tau": t, "seed": seed,
            "redundancy_rate": redundancy_rate,
            "tp_rate_pct": redundancy_rate * 100.0,
            "throughput_improvement_pct": (throughput_with - throughput_without) / throughput_without * 100.0,
            "mean_index_accesses": n_index_accesses,
            "n_queries": n_q,
            "positive_rate_pct": n_index_accesses / max(n_q, 1) * 100.0,
            "fpr": rates["fpr"], "fnr": rates["fnr"],
            "j_tau": _J_tau(radii, dim, tau=t),
            "log10_j_tau": _log10_J(radii, dim, tau=t),
            "log10_vol_low": vol_log10_low,
            "log10_vol_high": vol_log10_high,
            "n_balls": len(radii),
        })
    return rows


def blocklist_worker(
    dim: int,
    dataset: str,
    seed: int,
    k_fractions: List[float],
    target_recall: float,
    tau_values: List[float],
    vectors_reference: VectorInput,
    vectors_positive: VectorInput,
    vectors_hard_negative: VectorInput,
    vectors_easy_negative: VectorInput,
    volume_bounds: bool = False,
) -> List[Dict[str, Any]]:
    """Asymmetric blocklist workload (main.generate_blocklist): a tight
    reference set B, a broad query stream, positives that are held-out members
    of B's own duplicate clusters.

    Unlike system_tau_worker, tau is not swept over a grid -- it is solved per
    row for `target_recall` on the positives, so every compression ratio is
    compared at an equal-recall operating point and the interesting axis is
    FPR. Sweeping tau instead would let a row buy a better FPR by quietly
    giving up recall.

    Negatives are split in two and never pooled. Hard negatives are members of
    duplicate clusters that were NOT blocklisted: same tight local geometry as
    B, so a filter that merely detects "is this vector in a dense region"
    fails them and only one that localises B passes. Easy negatives are
    singletons, i.e. background traffic. A pooled FPR would be dominated by
    whichever class happened to be larger.

    volume_bounds adds the overlap-aware log10_vol_low/log10_vol_high columns
    (_tau_volume_bounds_log10) to the tau-sweep rows. Off by default because
    that bound is ~O(k^2) -- ~160s at k=9665 -- and generate_blocklist sweeps
    a whole tau grid per k, which multiplies that cost by the length of the
    grid and dominates the run. generate_budget passes a single tau, so it
    pays it once per k and turns this on; see the log10_j_tau comment in
    row_at for the cheap sum-of-volumes proxy that is always reported.
    """
    B = np.ascontiguousarray(_to_construct_array(vectors_reference), dtype=np.float32)
    pos = np.ascontiguousarray(_to_construct_array(vectors_positive), dtype=np.float32)
    hard = np.ascontiguousarray(_to_construct_array(vectors_hard_negative), dtype=np.float32)
    easy = np.ascontiguousarray(_to_construct_array(vectors_easy_negative), dtype=np.float32)

    # Exact nearest-reference distances: the workload's own geometry, reported
    # so a row is interpretable without re-deriving it (delta vs d_hard).
    bix = faiss.IndexFlatL2(dim)
    bix.add(B)

    def nn_dist(Q):
        if len(Q) == 0:
            return np.zeros(0, dtype=np.float32)
        d2, _ = bix.search(Q, 1)
        return np.sqrt(np.clip(d2[:, 0], 0, None))

    d_pos_med = float(np.median(nn_dist(pos))) if len(pos) else float("nan")
    d_hard_med = float(np.median(nn_dist(hard))) if len(hard) else float("nan")

    n_queries = len(pos) + len(hard) + len(easy)
    rows: List[Dict[str, Any]] = []

    for kf in k_fractions:
        k = min(max(1, round(kf * len(B))), len(B))
        filt = BallCoverFilter(dim=dim, max_memory=_ball_cover_max_memory(dim, k))
        filt.construct(B)
        radii = filt.all_radiuses
        centers = np.ascontiguousarray(filt.all_centers, dtype=np.float32)

        cix = faiss.IndexFlatL2(dim)
        cix.add(centers)

        def margin(Q):
            """Smallest (distance to a ball's centre - that ball's radius) over
            every ball: the query is flagged iff this is <= tau. Searching all
            balls rather than a top-m probe, since a query can sit nearest to
            one centre while another, fatter ball is the one that covers it."""
            if len(Q) == 0:
                return np.zeros(0, dtype=np.float32)
            D, I = cix.search(Q, len(radii))
            return (np.sqrt(np.clip(D, 0, None)) - radii[I]).min(axis=1)

        m_pos, m_hard, m_easy = margin(pos), margin(hard), margin(easy)
        r_med = float(np.median(radii)) if len(radii) else 0.0

        def row_at(tau: float, sweep: str) -> Dict[str, Any]:
            """One measurement of the whole query stream at this tau. The
            geometric objective and the volume bounds are recomputed per tau
            because tau expands every ball (eff. radius r_i + tau), so they are
            functions of tau at fixed k, not constants of the cover."""
            recall = float(np.mean(m_pos <= tau)) if len(m_pos) else 0.0
            fpr_hard = float(np.mean(m_hard <= tau)) if len(m_hard) else 0.0
            fpr_easy = float(np.mean(m_easy <= tau)) if len(m_easy) else 0.0
            # Pooled FPR over both negative classes, for the "FPR vs tau" view
            # where the split into hard/easy is not the axis of interest.
            n_neg = len(hard) + len(easy)
            fpr = ((fpr_hard * len(hard)) + (fpr_easy * len(easy))) / n_neg if n_neg else 0.0

            n_flagged = (recall * len(pos)) + (fpr_hard * len(hard)) + (fpr_easy * len(easy))
            saved = 100.0 * (1.0 - n_flagged / max(n_queries, 1))

            # Only for the fixed-tau rows, and only when asked: this is the
            # ~O(k^2) bound (see the docstring). The sweep="recall" row solves
            # its own tau and so would need its own second run of it, which no
            # figure reads -- NaN there is "not measured", not "zero volume".
            vol_low, vol_high = float('nan'), float('nan')
            if volume_bounds and sweep == "tau":
                vol_low, vol_high = _tau_volume_bounds_log10(centers, radii, tau)

            return {
                "dataset": dataset, "seed": seed, "sweep": sweep,
                "k": len(radii), "n_reference": len(B),
                # k_fraction is the nominal knob and is identical across
                # datasets and folds; `compression` is what it realised here,
                # which is not (|B| differs per dataset and per fold, so k =
                # round(kf*|B|) lands on a different ratio each time). Group
                # and filter on k_fraction -- grouping on compression makes
                # every fold its own singleton group, which silently drops the
                # error bars and matches only one dataset when filtering.
                "k_fraction": float(kf),
                "compression": len(B) / max(len(radii), 1),
                "r_med": r_med, "r_max": float(radii.max()) if len(radii) else 0.0,
                "tau": tau, "recall": recall,
                "fpr": fpr, "fpr_hard": fpr_hard, "fpr_easy": fpr_easy,
                "n_positive": len(pos), "n_hard": len(hard), "n_easy": len(easy),
                "n_queries": n_queries,
                "accesses_saved_pct": saved,
                # log10_J is a logsumexp over the balls, i.e. O(k), so it is
                # always reported -- but it sums each ball's volume
                # independently and double-counts overlap. The overlap-aware
                # union bound below is the tighter number, and is opt-in
                # because it is ~O(k^2) (see volume_bounds in the docstring).
                "j_tau": _J_tau(radii, dim, tau=tau),
                "log10_j_tau": _log10_J(radii, dim, tau=tau),
                "log10_vol_low": vol_low,
                "log10_vol_high": vol_high,
                "n_balls": len(radii),
                "d_pos_med": d_pos_med, "d_hard_med": d_hard_med,
                # The predictor: room left for a ball once the duplicate
                # distance delta (d_pos_med) is paid for.
                "margin_budget": d_hard_med - r_med,
                "memory_bytes": filt.get_bytesize(),
            }

        # sweep="recall": the equal-recall operating point, one row per k --
        # tau solved so every compression ratio is compared at the same recall
        # and FPR is the only thing that moves.
        tau_recall = float(np.percentile(m_pos, target_recall * 100.0)) if len(m_pos) else 0.0
        rows.append(row_at(tau_recall, "recall"))

        # sweep="tau": the same cover swept over a fixed, dataset-independent
        # tau grid. Absolute tau is NOT comparable across corpora (that is the
        # whole reason the blocklist workload solves for a target instead), so
        # these rows are for reading one corpus's response curve, not for
        # ranking corpora against each other at a given tau.
        for t in tau_values:
            rows.append(row_at(float(t), "tau"))
    return rows


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

    filt = BallCoverFilter(dim=dim, max_memory=_ball_cover_max_memory(dim, k))
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
    """Measures batched-query throughput for a BallCoverFilter at filter
    size k, querying in batches of `batch_size`. Every batch_size times the
    same n_query_vectors, so the panel's lines are directly comparable.

    tau_mid does not affect the measurement: query() scores every ball and
    only then compares against the threshold, so the work is identical at
    any tau. It is here because query() requires one, not as a variable."""
    construct_arr = _to_construct_array(vectors_construct)
    query_arr = _to_construct_array(vectors_query)
    if len(query_arr) < n_query_vectors:
        raise ValueError(
            f"scaling_query_worker needs {n_query_vectors} query vectors but the holdout "
            f"holds {len(query_arr)}. Widen the holdout in generate_scaling() (it is sized "
            f"max(n_query_vectors, n_insert_vectors)) or lower n_query_vectors."
        )

    filt = BallCoverFilter(dim=dim, max_memory=_ball_cover_max_memory(dim, k))
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

    filt = BallCoverFilter(dim=dim, max_memory=_ball_cover_max_memory(dim, k))
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
    """Measures construction throughput for a BallCoverFilter at filter size
    k built over n indexed vectors -- swept over a (k, n) grid."""
    construct_arr = _to_construct_array(vectors_construct)[:n]

    start = time.perf_counter()
    filt = BallCoverFilter(dim=dim, max_memory=_ball_cover_max_memory(dim, k))
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
    tau_mid: float,
    policy: str,
    vectors_stream: VectorInput,
    vectors_negative: VectorInput,
    checkpoints: List[int],
    T: Optional[int] = None,
    insertion_level: Optional[str] = None,
    sweep: Optional[str] = None,
    fold_offset: Optional[int] = None,
    bootstrap_n: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Simulates one rebuild policy over vectors_stream, reporting FPR and
    amortized cost per checkpoint.
      "streaming": BallCoverFilter.insert() per arrival, never rebuilt.
      "periodic": like "streaming", rebuilt from scratch every T arrivals.
      "oracle": rebuilt from scratch at every checkpoint.
    sweep/fold_offset are pass-through tag columns only.

    bootstrap_n (streaming/periodic only) sizes the initial construct()
    window independently of the ball budget k -- default min(k, n) reproduces
    the old behavior, but that ties the bootstrap set 1:1 to the ball count,
    which for k <= n gives GonzalesConstruct one input point per center and
    hence zero-radius balls at position 0. Pass an explicit bootstrap_n
    (e.g. an order of magnitude larger than k) to start from a filter with
    real coverage instead."""
    stream_arr = _to_construct_array(vectors_stream)
    negative_arr = _to_construct_array(vectors_negative)
    n = len(stream_arr)
    checkpoint_set = sorted(set(c for c in checkpoints if 0 < c <= n))
    metric, method = _rebuild_policy_tags(policy, insertion_level)

    rows: List[Dict[str, Any]] = []
    cumulative_time_s = 0.0

    def _fpr(filt: BallCoverFilter) -> float:
        predicted = filt.query(negative_arr, distance_threshold=tau_mid)
        return float(np.mean(predicted)) if len(predicted) else 0.0

    def _assert_fnr_zero(filt: BallCoverFilter, seen: np.ndarray, pos: int) -> None:
        """Every stream vector seen so far must still be recalled; raises
        rather than recording a violation, like every other FNR check here."""
        recalled = filt.query(seen, distance_threshold=tau_mid)
        fnr = float(np.mean(~np.asarray(recalled, dtype=bool))) if len(recalled) else 0.0
        if fnr != 0.0:
            raise ValueError(
                f"Zero-false-negative invariant violated (dataset={dataset}, policy={policy}, "
                f"insertion_level={insertion_level}, k={k}, seed={seed}, position={pos}): fnr={fnr}"
            )

    if policy == "oracle":
        for c in checkpoint_set:
            filt = BallCoverFilter(dim=dim, max_memory=_ball_cover_max_memory(dim, k))
            t0 = time.perf_counter()
            filt.construct(stream_arr[:c])
            cumulative_time_s += time.perf_counter() - t0
            _assert_fnr_zero(filt, stream_arr[:c], c)
            radii = filt.all_radiuses
            rows.append({
                "dataset": dataset, "policy": policy, "T": T, "k": k, "seed": seed,
                "position": c, "fpr": _fpr(filt),
                "log10_j_tau": _log10_J(radii, dim, tau=tau_mid),
                "n_balls": len(radii),
                "memory_bytes": filt.get_bytesize(),
                "cumulative_time_s": cumulative_time_s,
                "amortized_cost_s": cumulative_time_s / c,
                "metric": metric, "method": method, "sweep": sweep, "fold_offset": fold_offset,
            })
        return rows

    if policy not in ("streaming", "periodic"):
        raise ValueError(f"Unknown policy {policy!r}; expected 'streaming', 'periodic', or 'oracle'")

    if insertion_level not in _INSERT_STRATEGIES:
        raise ValueError(
            f"Unknown insertion_level {insertion_level!r}; expected one of "
            f"{sorted(_INSERT_STRATEGIES)}"
        )
    rebuild_every = T if (policy == "periodic" and T is not None) else None
    filt = BallCoverFilter(dim=dim, max_memory=_ball_cover_max_memory(dim, k),
                             insert_method=_INSERT_STRATEGIES[insertion_level]())
    # Bootstrap via one shared construct() (identical across insertion_level)
    # so every checkpoint from bootstrap_n onward compares each method's own
    # decision rule.
    bootstrap_n = min(bootstrap_n if bootstrap_n is not None else k, n)
    t0 = time.perf_counter()
    filt.construct(stream_arr[:bootstrap_n])
    cumulative_time_s += time.perf_counter() - t0

    ci = 0
    for t in range(bootstrap_n, n):
        pos = t + 1
        t0 = time.perf_counter()
        if rebuild_every is not None and pos % rebuild_every == 0:
            filt.construct(stream_arr[:pos])
        else:
            filt.insert(stream_arr[t][np.newaxis, :])
        cumulative_time_s += time.perf_counter() - t0

        if ci < len(checkpoint_set) and pos == checkpoint_set[ci]:
            _assert_fnr_zero(filt, stream_arr[:pos], pos)
            radii = filt.all_radiuses
            rows.append({
                "dataset": dataset, "policy": policy, "T": T, "k": k, "seed": seed,
                "position": pos, "fpr": _fpr(filt),
                "log10_j_tau": _log10_J(radii, dim, tau=tau_mid),
                "n_balls": len(radii),
                "memory_bytes": filt.get_bytesize(),
                "cumulative_time_s": cumulative_time_s,
                "amortized_cost_s": cumulative_time_s / pos,
                "metric": metric, "method": method, "sweep": sweep, "fold_offset": fold_offset,
            })
            ci += 1

    return rows


_WORKER_FNS: Dict[str, Any] = {
    'system_tau': system_tau_worker,
    'blocklist': blocklist_worker,
    # Same worker, second mode: generate_budget runs the identical blocklist
    # measurement but sweeps the ball budget at a fixed tau instead of solving
    # tau per budget (see main.generate_budget). A distinct mode tag only so
    # rows_for_mode can split the two evaluations' rows back apart.
    'budget': blocklist_worker,
    'quantization': quantization_worker,
    'query_performance': query_performance_worker,
    'equal_memory': equal_memory_worker,
    'scaling_query': scaling_query_worker,
    'scaling_insert': scaling_insert_worker,
    'scaling_construction': scaling_construction_worker,
    'stream_rebuild': stream_rebuild_worker,
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


class Runner:
    def __init__(self, num_workers: int = 1, truncate: bool = True) -> None:
        """truncate=False preserves an output_csv's existing rows (see
        submit()) instead of starting fresh, so a resumed run appends."""
        self.num_workers = num_workers
        self.truncate = truncate
        self._jobs: List[Dict[str, Any]] = []
        self._rows_by_csv: Dict[str, List[Dict[str, Any]]] = {}
        self._initialized_csvs: set = set()

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
        return self._run_sequential() if self.num_workers == 1 else self._run_parallel()

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
            except Exception:
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
                except Exception:
                    # See _run_sequential's comment.
                    logger.exception(f"Job failed, skipping: {desc}")
                    continue
                logger.info(f"Completed: {desc}")
                results.append((mode, result))
                self._record_result(output_csv, result)
        return results