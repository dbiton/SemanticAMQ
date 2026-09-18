from __future__ import annotations

import logging
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

# Must happen before numpy/faiss are imported -- see main.py's own copy of
# this guard for why (BLAS/OpenMP thread-count x Runner's multiprocess
# worker count multiply into CPU/memory oversubscription otherwise).
# Repeated here (not just in main.py) since each ProcessPoolExecutor worker
# re-imports this module fresh, and other entry points (tests, tradeoff.py)
# may import runner.py directly without going through main.py first.
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
from filters.ball_cover.ball_cover_filter import BallCoverFilter, Quantize, InsufficientMemoryError
from filters.quantized_vector_filter import (
    QuantizedVectorFilter, InsufficientMemoryError as QVFInsufficientMemoryError,
)
from filters.ball_cover.nearest_insert import NearestInsert
from filters.ball_cover.least_inflation_insert import LeastInflationInsert
from filters.ball_cover.exhaustive_merge_insert import ExhaustiveMergeInsert
from filters.ball_cover.min_heap_insert import MinHeapInsert
from filters.ball_cover.doubling_insert import DoublingInsert
from hyperspheres_volume import estimate_hypersphere_volume
from vector_source import VectorSource

# Configured at import time so worker functions still log with a timestamp
# when re-imported inside a ProcessPoolExecutor child process.
configure_logging()
logger = logging.getLogger(__name__)

# String name -> InsertInterface class (classes, not instances -- each job
# needs its own fresh strategy object with its own state). "doubling" is
# registered but not in generate_streaming()'s own sweep list, a deliberate
# opt-in since it adds real compute.
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


def _compute_metrics(
    predicted: List[bool],
    actual: List[bool],
    distance_threshold: float,
    config: str,
    percentage: float,
    memory_usage: int,
    memory_budget: int,
) -> Dict[str, Any]:
    tp, fp, fn, tn = _binary_confusion(predicted, actual)
    precision, recall, f1 = _precision_recall_f1(predicted, actual)
    return {
        'Config':       config,
        'Percentage':   percentage,
        'Threshold':    distance_threshold,
        'FPR':          fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        'FNR':          fn / (fn + tp) if (fn + tp) > 0 else 0.0,
        'Precision':    precision,
        'Recall':       recall,
        'F1-Score':     f1,
        'Memory Usage': memory_usage,
        'Memory Budget': memory_budget,
    }


def _radius_stats(filter_instance: Any) -> Dict[str, float]:
    radii = getattr(filter_instance, 'all_radiuses', None)
    if radii is None or len(radii) == 0:
        return {'Radius Mean': float('nan'), 'Radius Max': float('nan'),
                'Radius Min': float('nan'), 'Radius Std': float('nan')}
    return {
        'Radius Mean': float(np.mean(radii)), 'Radius Max': float(np.max(radii)),
        'Radius Min': float(np.min(radii)), 'Radius Std': float(np.std(radii)),
    }


_DIRECT_TAU_FILTERS = {
    "FirstNFilter", "BallCoverFilter",
    "DSBFFilter", "CoveringLSHFallbackFilter", "LSBFFilter", "DSHFilter", "HNSWFilter",
    "IVFFilter", "QuantizedVectorFilter",
}


def _ball_cover_max_memory(dim: int, k: int, quantize: Quantize = Quantize.NONE) -> int:
    """BallCoverFilter sizes its ball budget from a byte budget (max_memory),
    not a raw ball count -- this converts a desired count k into the
    max_memory that yields exactly k balls for the given quantize type."""
    bytes_per_ball = BallCoverFilter(dim=dim, max_memory=1, quantize=quantize).bytes_per_ball()
    return bytes_per_ball * k


class _BallCoverFilterEvalAdapter:
    """BallCoverFilter.query() takes an explicit per-call distance_threshold
    rather than baking one in at construction time, unlike the other
    _DIRECT_TAU_FILTERS below -- this adapter captures the threshold once at
    construction so the generic single-arg `filter_instance.query(block)`
    call sites elsewhere in this module (static/dynamic/system/equal_memory
    workers) keep working unchanged."""

    def __init__(self, dim: int, max_memory: int, distance_threshold: float,
                 max_spheres: Optional[int] = None, insert_mem_percent: Optional[float] = 0.0, **extra_kwargs: Any) -> None:
        self.distance_threshold = distance_threshold
        quantize = extra_kwargs.get("quantize", Quantize.NONE)
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
    extra_kwargs = extra_kwargs or {}
    if filter_cls.__name__ in _DIRECT_TAU_FILTERS or 'max_spheres' in inspect.signature(filter_cls.__init__).parameters:
        if filter_cls is BallCoverFilter:
            return _BallCoverFilterEvalAdapter(dim=dim, max_memory=memory_budget,distance_threshold=distance_threshold,
                                                max_spheres=max_spheres, insert_mem_percent=insert_mem_percent, **extra_kwargs)
        else:
            return filter_cls(dim=dim, max_memory=memory_budget, distance_threshold=distance_threshold,
                            max_spheres=max_spheres, **extra_kwargs)
    else:
        try:
            return filter_cls(dim=dim, max_memory=memory_budget, **extra_kwargs)
        except TypeError:
            return filter_cls(dim=dim, **extra_kwargs)


def static_worker(
    filter_cls: Any,
    dim: int,
    distance_threshold: float,
    memory_budget: int,
    vectors_construct: VectorInput,
    vectors_query: VectorInput,
    actual_results: List[bool],
    percentage: float,
) -> Dict[str, Any]:
    config = filter_cls.__name__
    filter_instance = _instantiate_filter(filter_cls, dim, memory_budget, distance_threshold)
    filter_instance.construct(_to_construct_array(vectors_construct))

    predicted: List[bool] = []
    for block in _iter_query_batches(vectors_query, _QUERY_BATCH):
        predicted.extend(filter_instance.query(block))

    memory_usage = filter_instance.get_bytesize()
    metrics = _compute_metrics(predicted, actual_results, distance_threshold, config, percentage, memory_usage, memory_budget)
    metrics.update(_radius_stats(filter_instance))
    return metrics


def dynamic_worker(
    filter_cls: Any,
    dim: int,
    distance_threshold: float,
    memory_budget: int,
    vectors_construct: VectorInput,
    vectors_query: VectorInput,
    actual_results: List[bool],
    percentage: float,
) -> Dict[str, Any]:
    config = filter_cls.__name__
    filter_instance = _instantiate_filter(filter_cls, dim, memory_budget, distance_threshold)
    filter_instance.construct(_to_construct_array(vectors_construct))

    predicted: List[bool] = []
    for v in _iter_query_vectors(vectors_query):
        is_dup = filter_instance.query(np.array([v]))[0]
        predicted.append(is_dup)
        if not is_dup:
            filter_instance.insert(np.array([v]))

    memory_usage = filter_instance.get_bytesize()
    return _compute_metrics(predicted, actual_results, distance_threshold, config, percentage, memory_usage, memory_budget)


def _f1_score(predicted: np.ndarray, actual: np.ndarray) -> float:
    return _precision_recall_f1(predicted, actual)[2]


_IVF_NPROBE_CANDIDATES = (1, 2, 4, 8, 16, 32, 64)


# Filters with a cheap post-construction query-time dial: score every candidate on one built instance.
_POST_CONSTRUCTION_CONFIG_DIALS = {
    "DSBFFilter": lambda f: [("vote_threshold", t) for t in range(1, f.num_tables + 1)],
    # nprobe is a pure query-time knob (IVFFilter.nprobe proxies to the built
    # index, see its property docstring) -- swept on a log-ish grid up to
    # nlist rather than exhaustively, since nlist can be in the hundreds.
    "IVFFilter": lambda f: [
        ("nprobe", p) for p in sorted({c for c in _IVF_NPROBE_CANDIDATES if c <= f.nlist} | {f.nlist})
    ],
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
    (filter_instance, construction_seconds) -- construction_seconds sums only
    the time spent inside .construct() calls (every candidate build, for a
    construction-time dial), excluding the calibration-set .query() calls used
    purely to score candidates, so it reflects the cost of reaching a
    ready-to-query filter rather than this function's own search overhead."""
    name = filter_cls.__name__
    construction_time = 0.0

    if name in _CONSTRUCTION_TIME_CONFIG_DIALS:
        best_f1, best_instance = -1.0, None
        for kwargs in _CONSTRUCTION_TIME_CONFIG_DIALS[name]:
            # Routed through _instantiate_filter's own dispatch (not a
            # hardcoded direct-tau constructor call) so a filter added to
            # this dial registry that isn't in _DIRECT_TAU_FILTERS still
            # gets built correctly.
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
    except QVFInsufficientMemoryError as e:
        # Expected outcome at small budgets (see QuantizedVectorFilter's
        # docstring) -- recorded as an explicit infeasible row (NaN metrics)
        # rather than dropped silently or crashing the sweep.
        logger.warning(f"[Infeasible] {config} at memory_budget={memory_budget}: {e}")
        return {
            'Config': config, 'Dataset': dataset, 'Memory Budget': memory_budget,
            'Memory Usage': float('nan'), 'Threshold': distance_threshold,
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

    return {
        'Config': config,
        'Dataset': dataset,
        'Memory Budget': memory_budget,
        'Memory Usage': filter_instance.get_bytesize(),
        'Threshold': distance_threshold,
        'FPR': fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        'FNR': fn / (fn + tp) if (fn + tp) > 0 else 0.0,
        # Construction latency, not query latency -- this comparison cares
        # about the one-time cost of building each filter at a given memory
        # budget (see _build_calibrated_filter's docstring for what's
        # included/excluded), not per-query serving cost.
        'Latency (ms)': construction_time * 1000,
        'Throughput (qps)': n_q / max(elapsed, 1e-9),
        'Infeasible': False,
    }


def _fpvr_bounds(
    orig_log_low: Any, orig_log_high: Any, filt_log_low: float, filt_log_high: float,
) -> tuple:
    """Bounds on log(V_filter / V_original) -- kept in log space since the raw
    ratio can overflow at high dimension. Always >= 0 since every true-duplicate
    ball is contained in its filter ball; lower bound floored at 0."""
    if (orig_log_low is None or orig_log_high is None
            or np.isnan(filt_log_low) or np.isinf(filt_log_low)):
        return float('nan'), float('nan')

    log_ratio_lower = max(filt_log_low - orig_log_high, 0.0)
    log_ratio_upper = max(filt_log_high - orig_log_low, log_ratio_lower)
    return log_ratio_lower, log_ratio_upper


def system_worker(
    filter_cls: Any,
    dim: int,
    distance_threshold: float,
    memory_budget: int,
    vectors_construct: VectorInput,
    vectors_query: VectorInput,
    percentage: float,
    dataset: str = "",
    orig_vol_log_low: float = None,
    orig_vol_log_high: float = None,
) -> Dict[str, Any]:
    """Measures how much a filter reduces flat-L2-index queries (one-sided:
    filter negatives are trusted, positives get searched) -- same query
    stream timed with and without the filter. orig_vol_log_* is the caller's
    precomputed true-duplicate-volume bound, combined with the filter's own
    covering volume into ground-truth-free FPVR bounds."""
    config = filter_cls.__name__

    construct = np.ascontiguousarray(_to_construct_array(vectors_construct), dtype=np.float32)

    index = faiss.IndexFlatL2(dim)
    index.add(construct)

    filter_instance = _instantiate_filter(filter_cls, dim, memory_budget, distance_threshold)
    construct_result = filter_instance.construct(construct)
    if isinstance(construct_result, tuple) and len(construct_result) == 2:
        filt_vol_log_low, filt_vol_log_high = construct_result
    else:
        filt_vol_log_low, filt_vol_log_high = float('nan'), float('nan')
    fpvr_lower, fpvr_upper = _fpvr_bounds(orig_vol_log_low, orig_vol_log_high, filt_vol_log_low, filt_vol_log_high)

    sq_threshold = distance_threshold ** 2

    n_queries = 0
    baseline_pred: List[bool] = []
    start = time.perf_counter()
    for block in _iter_query_batches(vectors_query, _QUERY_BATCH):
        block = np.ascontiguousarray(block, dtype=np.float32)
        n_queries += len(block)
        nn_sq_dist, _ = index.search(block, 1)
        baseline_pred.extend((nn_sq_dist[:, 0] <= sq_threshold).tolist())
    time_without = time.perf_counter() - start

    n_index_accesses = 0
    filtered_pred: List[bool] = []
    start = time.perf_counter()
    for block in _iter_query_batches(vectors_query, _QUERY_BATCH):
        block = np.ascontiguousarray(block, dtype=np.float32)
        positive_mask = np.array(filter_instance.query(block), dtype=bool)
        block_pred = np.zeros(len(block), dtype=bool)
        if positive_mask.any():
            n_index_accesses += int(positive_mask.sum())
            nn_sq_dist, _ = index.search(block[positive_mask], 1)
            block_pred[positive_mask] = nn_sq_dist[:, 0] <= sq_threshold
        filtered_pred.extend(block_pred.tolist())
    time_with = time.perf_counter() - start

    false_negatives = sum(1 for b, f in zip(baseline_pred, filtered_pred) if b and not f)

    n_q = max(n_queries, 1)
    return {
        'Config':                       config,
        'Dataset':                      dataset,
        'Percentage':                   percentage,
        'Threshold':                    distance_threshold,
        'Memory Budget':                memory_budget,
        'Filter Memory Usage':          filter_instance.get_bytesize(),
        'Total Queries':                n_queries,
        'Time Without Filter (s)':      time_without,
        'Time With Filter (s)':         time_with,
        'Latency Without Filter (ms)':  time_without / n_q * 1000,
        'Latency With Filter (ms)':     time_with / n_q * 1000,
        'Throughput Without Filter':    n_queries / max(time_without, 1e-9),
        'Throughput With Filter':       n_queries / max(time_with, 1e-9),
        'Speedup':                      time_without / max(time_with, 1e-9),
        'Index Accesses':               n_index_accesses,
        'Index Access Rate':            n_index_accesses / n_q,
        'Filter False Negatives':       false_negatives,
        'Original Volume Log Low':      orig_vol_log_low,
        'Original Volume Log High':     orig_vol_log_high,
        'Filter Volume Log Low':        filt_vol_log_low,
        'Filter Volume Log High':       filt_vol_log_high,
        'FPVR Lower':                   fpvr_lower,
        'FPVR Upper':                   fpvr_upper,
        **_radius_stats(filter_instance),
    }


def _compute_tradeoff_metrics(
    predicted: List[bool],
    actual: List[bool],
    config: str,
    max_spheres: int,
    memory_budget: int,
    memory_usage: int,
    construct_time: float,
    qps: float
) -> Dict[str, Any]:
    """
    function for computing memory and sphere tradeoff metrics
    """
    tp = sum(1 for p, a in zip(predicted, actual) if p and a)
    fp = sum(1 for p, a in zip(predicted, actual) if p and not a)
    fn = sum(1 for p, a in zip(predicted, actual) if not p and a)
    tn = sum(1 for p, a in zip(predicted, actual) if not p and not a)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "Config": config,
        "NumSpheres": max_spheres if max_spheres is not None else 0,
        "MemoryBudget": memory_budget,
        "MemoryUsage": memory_usage,
        "FPR": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        "FNR": fn / (fn + tp) if (fn + tp) > 0 else 0.0,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ConstructTime": construct_time,
        "QPS": qps
    }


def tradeoff_static_worker(
    filter_cls: Any,
    dim: int,
    distance_threshold: float,
    vectors_construct: np.ndarray,
    vectors_query: np.ndarray,
    actual_results: List[bool],
    max_spheres: int = None,
    memory_budget: int = None,
    insert_memory_percent: float = 0.0
) -> Dict[str, Any]:
    """
    Function for executing a single benchmarking trial of a static filter,
    constructing the filter once from the full dataset and measuring
    construction time, memory usage, query throughput, and accuracy metrics.
    """
    try:
        config = filter_cls.__name__
        vectors_construct, vectors_query = _to_construct_array(vectors_construct), _to_construct_array(vectors_query)

        construct_start = time.perf_counter() # start construct timer
        filter_instance = _instantiate_filter(filter_cls, dim, memory_budget, distance_threshold, max_spheres, insert_memory_percent)
        filter_instance.construct(vectors_construct)
        construct_latency = time.perf_counter() - construct_start # final construct time

        query_start = time.perf_counter() # start qps timer
        # stream the query set in bounded blocks for efficiency
        predicted: List[bool] = []
        for block in _iter_query_batches(vectors_query, _QUERY_BATCH):
            predicted.extend(filter_instance.query(block))
        query_latency = time.perf_counter() - query_start # final qps timer

        qps = len(vectors_query) / max(query_latency, 1e-9)
        memory_usage = filter_instance.get_bytesize()

    except Exception as e:
        logger.warning(f"[Skipped] {config} failed: {e}")
        return None

    return _compute_tradeoff_metrics(predicted, actual_results, config, max_spheres, memory_budget, memory_usage, construct_latency, qps)


def tradeoff_dynamic_worker(
    filter_cls: Any,
    dim: int,
    distance_threshold: float,
    vectors_construct: np.ndarray,
    vectors_query: np.ndarray,
    actual_results: List[bool],
    max_spheres: int = None,
    memory_budget: int = None,
    insert_memory_percent: float = 0.0
) -> Dict[str, Any]:
    """
    Function for executing a single benchmarking trial of a dynamic filter,
    where vectors are inserted incrementally while measuring insertion time,
    memory usage, query throughput, and accuracy metrics.
    """
    try:
        config = filter_cls.__name__
        vectors_construct, vectors_query = _to_construct_array(vectors_construct), _to_construct_array(vectors_query)

        construct_start = time.perf_counter() # start construct timer
        filter_instance = _instantiate_filter(filter_cls, dim, memory_budget, distance_threshold, max_spheres, insert_memory_percent)
        filter_instance.construct(vectors_construct)
        construct_latency = time.perf_counter() - construct_start # final construct time

        query_start = time.perf_counter() # start throughput timer
        predicted: List[bool] = []
        for block in _iter_query_batches(vectors_query, _QUERY_BATCH):
            block_preds = filter_instance.query(block)
            predicted.extend(block_preds)

            misses = [block[i] for i, is_dup in enumerate(block_preds) if not is_dup]
            if misses:
                filter_instance.insert(np.array(misses, dtype=np.float32))
        query_latency = time.perf_counter() - query_start # final throughput timer

        qps = len(vectors_query) / max(query_latency, 1e-9)
        memory_usage = filter_instance.get_bytesize()

    except Exception as e:
        logger.warning(f"[Skipped] {config} failed: {e}")
        return None

    return _compute_tradeoff_metrics(predicted, actual_results, config, max_spheres, memory_budget, memory_usage, construct_latency, qps)


_LN10 = float(np.log(10.0))


def _log10_J(radii: np.ndarray, dim: int, tau: float = 0.0) -> float:
    """log10 sum_i (r_i + tau)^d, safe for d in the hundreds where r^d over/underflows float64."""
    r = np.asarray(radii, dtype=np.float64) + float(tau)
    if r.size == 0:
        return float('-inf')
    with np.errstate(divide='ignore'):
        logr = np.log(r)
    return float(logsumexp(dim * logr)) / _LN10


def _J_tau(radii: np.ndarray, dim: int, tau: float = 0.0) -> float:
    """Linear J_tau(C) = sum_i (r_i + tau)^d; may overflow to +inf at d~384, r>~1 -- use _log10_J to plot."""
    r = np.asarray(radii, dtype=np.float64) + float(tau)
    if r.size == 0:
        return 0.0
    with np.errstate(over='ignore'):
        return float(np.sum(r ** dim))


def _confusion_rates(predicted: List[bool], actual: np.ndarray) -> Dict[str, float]:
    tp, fp, fn, tn = _binary_confusion(predicted, actual)
    return {
        'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        'fnr': fn / (fn + tp) if (fn + tp) > 0 else 0.0,
    }


# The quantization sweep varies BallCoverFilter's quantize choice rather than
# the retired per-level construction strategies. NONE is included as the
# raw/unquantized baseline point in this same sweep (tagged stage="ball")
# rather than a separate per-candidate stage: Gonzalez construction no
# longer varies by quantize choice, so there's exactly one "ball" cover to
# compare every real quantizer against, not one per candidate.
QUANTIZATION_SWEEP = (
    Quantize.NONE, Quantize.SQ8, Quantize.SQ4,
    Quantize.FP16, Quantize.BF16, Quantize.SQ4T, Quantize.SQ8T, Quantize.PQ,
)


def quantization_worker(
    dim: int,
    dataset: str,
    k: int,
    quantize: Quantize,
    tau_values: List[float],
    seed: int,
    construct_tau: float,
    vectors_construct: VectorInput,
    vectors_query: VectorInput,
    actual_results_by_tau: Dict[float, List[bool]],
) -> List[Dict[str, Any]]:
    """Builds one BallCoverFilter cover per (dataset, k, seed, quantize) and
    evaluates it at every tau. Returns one row per tau, tagged
    stage="ball"/"quantizing".

    max_memory is derived from k using Quantize.NONE's byte-per-ball size and
    held fixed across every quantize candidate, so a more compact code packs
    more balls into the same budget -- n_balls and memory_bytes then vary
    meaningfully per quantize choice instead of trivially matching k. A
    quantizer compact enough to saturate at n_construct balls before using
    the full budget (e.g. PQ) will show flat memory usage across the k
    sweep -- real information (construct-set-bound, not budget-bound), not a
    plotting artifact. construct_tau is unused: BallCoverFilter's
    construction has no distance-threshold knob.

    FNR is asserted (raises on violation), never plotted.

    If construction itself raises (e.g. InsufficientMemoryError -- too
    tight a budget for PQ's minimum codebook), that candidate is treated as
    non-viable at this (dataset, k) point: caught, logged, and this job
    contributes zero rows rather than aborting the whole sweep."""
    construct_arr = _to_construct_array(vectors_construct)
    query_arr = _to_construct_array(vectors_query)
    n_q = len(query_arr)
    for t, actual in actual_results_by_tau.items():
        if len(actual) != n_q:
            raise ValueError(
                f"vectors_query ({n_q}) and actual_results_by_tau[{t}] ({len(actual)}) length mismatch"
            )

    actual_by_tau = {t: np.asarray(a, dtype=bool) for t, a in actual_results_by_tau.items()}
    stage = "ball" if quantize == Quantize.NONE else "quantizing"

    rows: List[Dict[str, Any]] = []

    max_memory = _ball_cover_max_memory(dim, k, Quantize.NONE)
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

    # Per-ball radius inflation: the reconstruction error added onto each
    # ball's radius to preserve the no-false-negative guarantee under this
    # quantize choice's lossy center encoding (0 for Quantize.NONE by
    # construction -- see BallCoverFilter._radius_inflation).
    radius_inflation = filt.all_radius_inflation
    radius_inflation_mean = float(np.mean(radius_inflation)) if len(radius_inflation) else 0.0
    radius_inflation_std = float(np.std(radius_inflation)) if len(radius_inflation) else 0.0

    memory_bytes = filt.get_bytesize()

    predicted_by_tau: Dict[float, List[bool]] = {}
    query_throughput_by_tau: Dict[float, float] = {}
    for t in tau_values:
        q0 = time.perf_counter()
        predicted_by_tau[t] = filt.query(query_arr, distance_threshold=t)
        query_time = time.perf_counter() - q0
        query_throughput_by_tau[t] = n_q / max(query_time, 1e-9)

    for t in tau_values:
        rates = _confusion_rates(predicted_by_tau[t], actual_by_tau[t])
        rows.append({
            "dataset": dataset, "k": k, "quantize": quantize.value, "stage": stage, "tau": t,
            "log10_J": log10_J,
            "log10_J_tau": _log10_J(radii, dim, tau=t),
            "fpr": rates["fpr"], "fnr": rates["fnr"],
            "stage_time_s": stage_time, "cum_time_s": stage_time,
            "query_throughput_qps": query_throughput_by_tau[t],
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
    brute-force flat index -- the ground-truth step generate_system_tau()
    needs once per (dataset, split) before it can derive actual_results_by_tau
    at each tau. Split out into its own worker (rather than computed inline
    in the driver process, one dataset at a time) so main.py can run it
    across a ProcessPoolExecutor and cache the result to disk, instead of
    paying this cost serially on every call."""
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
    construct_tau: float,
    vectors_construct: VectorInput,
    vectors_query: VectorInput,
    actual_results_by_tau: Dict[float, List[bool]],
) -> List[Dict[str, Any]]:
    """Builds one BallCoverFilter cover (fixed k) per (dataset, seed) plus a
    flat L2 index over the same construct set, then sweeps the query-time
    threshold tau over that single construction, measuring filtered vs.
    unfiltered throughput/index-access/FPR/FNR at each tau. Queries are
    issued one at a time (not batched), on both arms of the comparison, to
    reflect a realistic online setting where each arrival is answered
    individually: the filtered path calls filt.query() on that one vector
    and only falls through to the backend index.search() when the filter
    reports a possible match, skipping the index entirely otherwise.
    redundancy_rate is the measured (not nominal) duplicate fraction of the
    query stream."""
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
            "n_balls": len(radii),
        })
    return rows


def query_performance_worker(
    dim: int,
    dataset: str,
    k: int,
    seed: int,
    construct_tau: float,
    query_tau: float,
    vectors_construct: VectorInput,
    vectors_query: VectorInput,
    single_query_limit: int = 2000,
) -> List[Dict[str, Any]]:
    """Times one query at a time vs. batched queries against the same cover;
    single-query mode is capped at single_query_limit (pays a Python call per
    query), batched mode is left uncapped. Returns one row per mode."""
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
    """Measures batched-query throughput (queries/sec) for a BallCoverFilter
    built at filter size k (over a fixed stream held constant across the
    whole batch_size sweep -- only k and batch_size vary here), querying in
    batches of `batch_size`. The query pool is cycled up to n_query_vectors
    so every batch_size times the same total query count."""
    construct_arr = _to_construct_array(vectors_construct)
    query_arr = _to_construct_array(vectors_query)

    filt = BallCoverFilter(dim=dim, max_memory=_ball_cover_max_memory(dim, k))
    filt.construct(construct_arr)

    stream = query_arr[np.arange(n_query_vectors) % len(query_arr)]

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
    """Measures batched-insertion throughput (vectors/sec) for a
    BallCoverFilter built at filter size k (same fixed-stream setup as
    scaling_query_worker), inserting via insert() in batches of `batch_size`
    after the initial construct(). The insert pool is cycled up to
    n_insert_vectors so every batch_size inserts the same total count."""
    construct_arr = _to_construct_array(vectors_construct)
    insert_arr = _to_construct_array(vectors_insert)

    filt = BallCoverFilter(dim=dim, max_memory=_ball_cover_max_memory(dim, k))
    filt.construct(construct_arr)

    stream = insert_arr[np.arange(n_insert_vectors) % len(insert_arr)]

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
    """Measures construction throughput (vectors/sec) for a BallCoverFilter
    at filter size k built over n indexed vectors -- the (k, n) grid this is
    swept over is the ablation-style basis for the construction-throughput-
    vs-filter-size plot's per-stream-size lines."""
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
    """(metric, method) tag used by the streaming figure. "oracle" is always
    a from-scratch rebuild regardless of insertion_level, so it gets one
    fixed reference tag; "streaming"/"periodic" report whichever
    insertion_level they actually ran (nearest/least_inflation/merge/
    min_heap -- see _INSERT_STRATEGIES) as the method tag, so all four are
    plotted as regular, directly comparable lines rather than one of them
    needing a special "exhaustive reference" tag of its own."""
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
) -> List[Dict[str, Any]]:
    """Simulates one rebuild policy over vectors_stream, reporting FPR and
    amortized cost per checkpoint.
      "streaming": BallCoverFilter.insert() per arrival (using the
        InsertInterface instance _INSERT_STRATEGIES[insertion_level]()
        selects), never rebuilt.
      "periodic": like "streaming", rebuilt from scratch every T arrivals.
      "oracle": rebuilt from scratch at every checkpoint (insertion_level
        doesn't apply -- there's nothing to insert into).
    sweep/fold_offset are pass-through tag columns only."""
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
        """The covering invariant is a correctness requirement, not a
        plotted quantity: every stream vector seen so far must still be
        recalled. Raises rather than recording a violation, matching every
        other FNR check in this module."""
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
    # Bootstrap-then-diverge: the first k arrivals become raw zero-radius
    # singletons via one batch construct() (identical across every
    # insertion_level), so checkpoint k+1 onward is a genuine comparison of
    # each method's own decision rule, not an artifact of some methods never
    # creating new balls on their own.
    bootstrap_n = min(k, n)
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
    'static':  static_worker,
    'dynamic': dynamic_worker,
    'tradeoff_static': tradeoff_static_worker,
    'tradeoff_dynamic': tradeoff_dynamic_worker,
    'system':  system_worker,
    'system_tau': system_tau_worker,
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
        """truncate controls what happens the first time a job targeting a
        given output_csv (see submit()) is submitted: True (default) starts
        that file fresh (removing any existing one); False loads and
        preserves its existing rows instead, so a resumed run appends to
        them rather than losing them."""
        self.num_workers = num_workers
        self.truncate = truncate
        self._jobs: List[Dict[str, Any]] = []
        self._rows_by_csv: Dict[str, List[Dict[str, Any]]] = {}
        self._initialized_csvs: set = set()

    def submit(self, mode: str, output_csv: Optional[str] = None, **kwargs: Any) -> None:
        """output_csv, if given, is this job's target CSV: rewritten from
        every row completed so far *for that path* (not appended -- job
        result schemas differ, and a rewrite union-handles that safely)
        each time one of its jobs completes -- so a long-running sweep's
        completed work is visible on disk well before the whole sweep
        finishes, and a crash mid-sweep doesn't lose it. output_csv, not
        mode, is the routing key several jobs (even of different modes)
        can share one path, or each target its own -- e.g. generate_scaling()
        submits three modes (scaling_query/insert/construction) to three
        separate files, while generate_quantization() submits one mode to
        one file. A job with no output_csv is only returned by run(), same
        as before this per-job routing existed."""
        if mode not in _WORKER_FNS:
            raise ValueError(f"Unknown mode {mode!r}. Valid: {list(_WORKER_FNS)}")
        if output_csv is not None and output_csv not in self._initialized_csvs:
            self._init_csv(output_csv)
        self._jobs.append({'mode': mode, 'output_csv': output_csv, **kwargs})

    def _init_csv(self, output_csv: str) -> None:
        """Runs once per distinct output_csv, at first submit() -- not at
        run() time -- so truncation happens before any job targeting that
        path has run, matching the old constructor-time truncation
        semantics."""
        self._initialized_csvs.add(output_csv)
        if not self.truncate and os.path.exists(output_csv):
            self._rows_by_csv[output_csv] = pd.read_csv(output_csv).to_dict("records")
        else:
            if os.path.exists(output_csv):
                os.remove(output_csv)
            self._rows_by_csv[output_csv] = []

    def shuffle(self, seed: Optional[int] = None) -> None:
        """Randomizes submitted-job order in place. Lets several generate_
        functions share one Runner (submit-all, then run-all once) without
        one evaluation's slow jobs (e.g. PQ construction, tens of minutes
        each) all landing on the same worker back-to-back while other
        workers drain a faster evaluation and sit idle -- shuffling
        interleaves fast and slow jobs across the whole combined queue so
        workers finish around the same time. No effect on correctness,
        only on job order and therefore wall-clock makespan."""
        random.Random(seed).shuffle(self._jobs)

    def run(self) -> List[Tuple[str, Any]]:
        """Returns (mode, result) pairs, one per submitted job, in
        completion order (parallel) or submission order (sequential) --
        the mode tag is what lets a caller that shared this Runner across
        several unrelated job types (see shuffle()'s docstring) split the
        combined results back apart afterward without guessing from each
        result's own schema."""
        return self._run_sequential() if self.num_workers == 1 else self._run_parallel()

    def clear(self) -> None:
        self._jobs.clear()

    def _record_result(self, output_csv: Optional[str], result: Any) -> None:
        """Appends result's rows to output_csv's in-memory accumulator (seeded
        by _init_csv at submit() time) and rewrites that one file from it --
        called from the single coordinating process only (never from inside a
        worker), both in the sequential loop and in _run_parallel's
        as_completed loop, so multiple jobs targeting the same output_csv
        never race to write it; no lock is needed because there is never more
        than one writer."""
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
                # already-completed results -- especially now that several
                # generate_* functions can share one Runner (see shuffle()),
                # so a single bad job would otherwise wipe out a fully
                # combined multi-evaluation batch instead of just its own
                # evaluation's rows.
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
                    # See _run_sequential's comment: one job's exception
                    # must not lose every other (possibly unrelated
                    # evaluation's) already-completed job's results.
                    logger.exception(f"Job failed, skipping: {desc}")
                    continue
                logger.info(f"Completed: {desc}")
                results.append((mode, result))
                self._record_result(output_csv, result)
        return results