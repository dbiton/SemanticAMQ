from __future__ import annotations

import copy
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import faiss
import numpy as np
import pandas as pd
import time

from scipy.special import logsumexp
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from src.filters.ball_cover_filter import BallCoverFilter
from filters import streaming_insertion
from hyperspheres_volume import estimate_hypersphere_volume
from vector_source import VectorSource

VectorInput = Union[np.ndarray, VectorSource]
_QUERY_BATCH = 4096


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
    "DSBFFilter", "CoveringLSHFallbackFilter", "LSBFFilter", "DSHFilter",
}


def _instantiate_filter(filter_cls: Any, dim: int, lsh: Any, memory_budget: int, distance_threshold: float = None, max_spheres: int = None) -> Any:
    if filter_cls.__name__ not in _DIRECT_TAU_FILTERS:
        try:
            return filter_cls(dim=dim, lsh=lsh, max_memory=memory_budget)
        except TypeError:
            return filter_cls(dim=dim, lsh=lsh)
    else:
        return filter_cls(dim=dim, max_memory=memory_budget, distance_threshold=distance_threshold, max_spheres=max_spheres)


def static_worker(
    filter_cls: Any,
    lsh_cls: Any,
    dim: int,
    distance_threshold: float,
    max_L: int,
    memory_budget: int,
    vectors_construct: VectorInput,
    vectors_query: VectorInput,
    actual_results: List[bool],
    percentage: float,
) -> Dict[str, Any]:
    config = f"{filter_cls.__name__} + {lsh_cls.__name__}"
    lsh = lsh_cls.build_optimized(dim, distance_threshold, None, max_L)
    filter_instance = _instantiate_filter(filter_cls, dim, lsh, memory_budget, distance_threshold)
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
    lsh_cls: Any,
    dim: int,
    distance_threshold: float,
    max_L: int,
    memory_budget: int,
    vectors_construct: VectorInput,
    vectors_query: VectorInput,
    actual_results: List[bool],
    percentage: float,
) -> Dict[str, Any]:
    config = f"{filter_cls.__name__} + {lsh_cls.__name__}"
    lsh = lsh_cls.build_optimized(dim, distance_threshold, None, max_L)
    filter_instance = _instantiate_filter(filter_cls, dim, lsh, memory_budget, distance_threshold)
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


# Filters with a cheap post-construction query-time dial: score every candidate on one built instance.
_POST_CONSTRUCTION_CONFIG_DIALS = {
    "DSBFFilter": lambda f: [("vote_threshold", t) for t in range(1, f.num_tables + 1)],
}

# Filters whose configuration changes memory layout at construction time: one build per candidate.
_CONSTRUCTION_TIME_CONFIG_DIALS = {
    "LSBFFilter": [{"verify": True}, {"verify": False}],
}


def _build_calibrated_filter(
    filter_cls: Any,
    dim: int,
    lsh: Any,
    memory_budget: int,
    distance_threshold: float,
    construct_arr: np.ndarray,
    calib_arr: np.ndarray,
    calib_actual: np.ndarray,
) -> Any:
    """Builds filter_cls at memory_budget, picking whichever registered config
    dial candidate scores best (F1) on the calibration set."""
    name = filter_cls.__name__

    if name in _CONSTRUCTION_TIME_CONFIG_DIALS:
        best_f1, best_instance = -1.0, None
        for kwargs in _CONSTRUCTION_TIME_CONFIG_DIALS[name]:
            candidate = filter_cls(
                dim=dim, max_memory=memory_budget, distance_threshold=distance_threshold, **kwargs
            )
            candidate.construct(construct_arr)
            f1 = (
                _f1_score(np.asarray(candidate.query(calib_arr), dtype=bool), calib_actual)
                if len(calib_arr) > 0 else 0.0
            )
            if f1 > best_f1:
                best_f1, best_instance = f1, candidate
        return best_instance

    filter_instance = _instantiate_filter(filter_cls, dim, lsh, memory_budget, distance_threshold)
    filter_instance.construct(construct_arr)

    if name in _POST_CONSTRUCTION_CONFIG_DIALS and len(calib_arr) > 0:
        best_attr, best_value, best_f1 = None, None, -1.0
        for attr, value in _POST_CONSTRUCTION_CONFIG_DIALS[name](filter_instance):
            setattr(filter_instance, attr, value)
            f1 = _f1_score(np.asarray(filter_instance.query(calib_arr), dtype=bool), calib_actual)
            if f1 > best_f1:
                best_attr, best_value, best_f1 = attr, value, f1
        if best_attr is not None:
            setattr(filter_instance, best_attr, best_value)

    return filter_instance


def equal_memory_worker(
    filter_cls: Any,
    lsh_cls: Any,
    dim: int,
    distance_threshold: float,
    max_L: int,
    memory_budget: int,
    vectors_construct: VectorInput,
    vectors_calibration: VectorInput,
    vectors_query: VectorInput,
    actual_calibration: List[bool],
    actual_query: List[bool],
    dataset: str = "",
) -> Dict[str, Any]:
    config = f"{filter_cls.__name__} + {lsh_cls.__name__}"
    lsh = lsh_cls.build_optimized(dim, distance_threshold, None, max_L)
    construct_arr = _to_construct_array(vectors_construct)
    calib_arr = _to_construct_array(vectors_calibration)
    calib_actual = np.asarray(actual_calibration, dtype=bool)

    filter_instance = _build_calibrated_filter(
        filter_cls, dim, lsh, memory_budget, distance_threshold,
        construct_arr, calib_arr, calib_actual,
    )

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
        'Latency (ms)': elapsed / n_q * 1000,
        'Throughput (qps)': n_q / max(elapsed, 1e-9),
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
    lsh_cls: Any,
    dim: int,
    distance_threshold: float,
    max_L: int,
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
    config = f"{filter_cls.__name__} + {lsh_cls.__name__}"
    lsh = lsh_cls.build_optimized(dim, distance_threshold, None, max_L)

    construct = np.ascontiguousarray(_to_construct_array(vectors_construct), dtype=np.float32)

    index = faiss.IndexFlatL2(dim)
    index.add(construct)

    filter_instance = _instantiate_filter(filter_cls, dim, lsh, memory_budget, distance_threshold)
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
    tp, fp, fn, tn = _binary_confusion(predicted, actual)
    precision, recall, f1 = _precision_recall_f1(predicted, actual)
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


def tradeoff_worker(
    filter_cls: Any,
    lsh_cls: Any,
    dim: int,
    distance_threshold: float,
    max_L: int,
    vectors_construct: np.ndarray,
    vectors_query: np.ndarray,
    actual_results: List[bool],
    max_spheres: int = None,
    memory_budget: int = None
) -> Dict[str, Any]:
    try:
        config = f"{filter_cls.__name__} + {lsh_cls.__name__}"
        lsh = lsh_cls.build_optimized(dim, distance_threshold, None, max_L)
        vectors_construct, vectors_query = _to_construct_array(vectors_construct), _to_construct_array(vectors_query)

        construct_start = time.perf_counter()
        filter_instance = _instantiate_filter(filter_cls, dim, lsh, memory_budget, distance_threshold, max_spheres)
        filter_instance.construct(vectors_construct)
        construct_latency = time.perf_counter() - construct_start

        query_start = time.perf_counter()
        predicted: List[bool] = []
        for block in _iter_query_batches(vectors_query, _QUERY_BATCH):
            predicted.extend(filter_instance.query(block))
        query_latency = time.perf_counter() - query_start

        qps = len(vectors_query) / max(query_latency, 1e-9)
        memory_usage = filter_instance.get_bytesize()

    except Exception as e:
        print(f"  [Skipped] {config} failed: {e}")
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


def ablation_worker(
    dim: int,
    dataset: str,
    k: int,
    tau_values: List[float],
    seed: int,
    construct_tau: float,
    vectors_construct: VectorInput,
    vectors_query: VectorInput,
    actual_results_by_tau: Dict[float, List[bool]],
    feedback_fraction: float = 0.1,
) -> List[Dict[str, Any]]:
    """Builds one BallCoverFilter cover per (dataset, k, seed) via the full
    construction ladder, then evaluates it at every tau. Returns one row per
    (stage, tau) for stages ["ball", "quantizing", "feedback"]; construct_tau
    is a sweep-independent construction knob, kept separate from the
    evaluated tau's so log10_J stays tau-independent for a given cover.
    Feedback is tau-specific (false positives depend on tau), so it branches
    per tau off a deep copy of the post-quantization cover."""
    construct_arr = _to_construct_array(vectors_construct)
    query_arr = _to_construct_array(vectors_query)
    n_q = len(query_arr)
    for t, actual in actual_results_by_tau.items():
        if len(actual) != n_q:
            raise ValueError(
                f"vectors_query ({n_q}) and actual_results_by_tau[{t}] ({len(actual)}) length mismatch"
            )

    n_witness = max(1, int(round(n_q * feedback_fraction))) if n_q > 0 else 0
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_q)
    witness_idx, eval_idx = perm[:n_witness], perm[n_witness:]
    witness_q, eval_q = query_arr[witness_idx], query_arr[eval_idx]
    actual_by_tau = {t: np.asarray(a, dtype=bool) for t, a in actual_results_by_tau.items()}

    rows: List[Dict[str, Any]] = []

    filt = BallCoverFilter(dim=dim, distance_threshold=construct_tau, max_spheres=k, seed=seed)
    filt.construct(construct_arr, retain_assignments=True)
    levels = filt.get_stats(taus=tau_values)["levels"]
    ball_level = levels[-1]
    ball_cum_time = sum(s["wall_time"] for s in levels)
    ball_log10_J = ball_level["log_J"] / _LN10

    for t in tau_values:
        predicted = filt.query(eval_q, tau=t)
        rates = _confusion_rates(predicted, actual_by_tau[t][eval_idx])
        rows.append({
            "dataset": dataset, "k": k, "stage": "ball", "stage_index": 1, "tau": t,
            "log10_J": ball_log10_J,
            "log10_J_tau": ball_level["log_J_tau"][t] / _LN10,
            "fpr": rates["fpr"], "fnr": rates["fnr"],
            "stage_time_s": ball_cum_time, "cum_time_s": ball_cum_time,
            "n_balls": ball_level["ball_count"], "seed": seed,
        })

    t0 = time.perf_counter()
    filt.quantize_centers()
    quant_time = time.perf_counter() - t0
    quant_cum_time = ball_cum_time + quant_time
    quant_radii = filt.all_radiuses
    quant_log10_J = _log10_J(quant_radii, dim)

    for t in tau_values:
        predicted = filt.query(eval_q, tau=t)
        rates = _confusion_rates(predicted, actual_by_tau[t][eval_idx])
        rows.append({
            "dataset": dataset, "k": k, "stage": "quantizing", "stage_index": 2, "tau": t,
            "log10_J": quant_log10_J,
            "log10_J_tau": _log10_J(quant_radii, dim, tau=t),
            "fpr": rates["fpr"], "fnr": rates["fnr"],
            "stage_time_s": quant_time, "cum_time_s": quant_cum_time,
            "n_balls": len(quant_radii), "seed": seed,
        })

    bad = [r for r in rows if r["fnr"] != 0]
    if bad:
        raise ValueError(
            f"Zero-false-negative invariant violated (dataset={dataset}, k={k}, seed={seed}): "
            f"{bad[:3]}"
        )

    for t in tau_values:
        filt_t = copy.deepcopy(filt)
        witness_actual_t = actual_by_tau[t][witness_idx]
        eval_actual_t = actual_by_tau[t][eval_idx]

        feedback_time = 0.0
        if n_witness > 0:
            witness_pred = np.asarray(filt_t.query(witness_q, tau=t), dtype=bool)
            fp_witnesses = witness_q[witness_pred & ~witness_actual_t]
            if len(fp_witnesses) > 0:
                feedback = filt_t.refine_with_feedback(fp_witnesses, thresholds=t)
                feedback_time = feedback["elapsed_seconds"]

        predicted = filt_t.query(eval_q, tau=t)
        rates = _confusion_rates(predicted, eval_actual_t)
        if rates["fnr"] != 0:
            raise ValueError(
                f"Zero-false-negative invariant violated at the feedback stage "
                f"(dataset={dataset}, k={k}, tau={t}, seed={seed}): fnr={rates['fnr']}"
            )

        radii = filt_t.all_radiuses
        rows.append({
            "dataset": dataset, "k": k, "stage": "feedback", "stage_index": 3, "tau": t,
            "log10_J": _log10_J(radii, dim),
            "log10_J_tau": _log10_J(radii, dim, tau=t),
            "fpr": rates["fpr"], "fnr": rates["fnr"],
            "stage_time_s": feedback_time, "cum_time_s": quant_cum_time + feedback_time,
            "n_balls": len(radii), "seed": seed,
        })
    return rows


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
    unfiltered throughput/index-access/FPR/FNR at each tau. redundancy_rate
    is the measured (not nominal) duplicate fraction of the query stream."""
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

    filt = BallCoverFilter(dim=dim, distance_threshold=construct_tau, max_spheres=k, seed=seed)
    filt.construct(construct_arr)
    radii = filt.all_radiuses

    rows: List[Dict[str, Any]] = []
    for t in tau_values:
        actual = np.asarray(actual_results_by_tau[t], dtype=bool)
        redundancy_rate = float(np.mean(actual)) if len(actual) > 0 else 0.0

        start = time.perf_counter()
        index.search(query_arr, 1)
        time_without = time.perf_counter() - start

        start = time.perf_counter()
        predicted = np.asarray(filt.query(query_arr, tau=t), dtype=bool)
        n_index_accesses = int(predicted.sum())
        if n_index_accesses > 0:
            index.search(query_arr[predicted], 1)
        time_with = time.perf_counter() - start

        rates = _confusion_rates(predicted.tolist(), actual)
        throughput_without = n_q / max(time_without, 1e-9)
        throughput_with = n_q / max(time_with, 1e-9)

        rows.append({
            "dataset": dataset, "k": k, "tau": t, "seed": seed,
            "redundancy_rate": redundancy_rate,
            "throughput_improvement_pct": (throughput_with - throughput_without) / throughput_without * 100.0,
            "mean_index_accesses": n_index_accesses,
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

    filt = BallCoverFilter(dim=dim, distance_threshold=construct_tau, max_spheres=k, seed=seed)
    filt.construct(construct_arr)

    rows: List[Dict[str, Any]] = []

    n_single = 0
    start = time.perf_counter()
    for v in _iter_query_vectors(vectors_query):
        if n_single >= single_query_limit:
            break
        filt.query(v[np.newaxis, :], tau=query_tau)
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
        filt.query(block, tau=query_tau)
        n_batched += len(block)
    elapsed_batched = time.perf_counter() - start
    rows.append({
        "dataset": dataset, "k": k, "seed": seed, "mode": "batched",
        "n_queries": n_batched,
        "latency_ms": elapsed_batched / max(n_batched, 1) * 1000,
        "throughput_qps": n_batched / max(elapsed_batched, 1e-9),
    })

    return rows


def scaling_worker(
    dim: int,
    dataset: str,
    k: int,
    seed: int,
    tau_mid: float,
    vectors_construct: VectorInput,
    vectors_query: VectorInput,
    actual_results: List[bool],
    single_query_limit: int = 2000,
) -> Dict[str, Any]:
    """Builds one BallCoverFilter at fixed k over n=len(vectors_construct)
    vectors; reports construction time, single-query latency, memory, FPR."""
    construct_arr = _to_construct_array(vectors_construct)
    query_arr = _to_construct_array(vectors_query)
    n = len(construct_arr)

    construct_start = time.perf_counter()
    filt = BallCoverFilter(dim=dim, distance_threshold=tau_mid, max_spheres=k, seed=seed)
    filt.construct(construct_arr)
    construct_time_s = time.perf_counter() - construct_start

    memory_bytes = filt.get_bytesize()

    n_single = min(single_query_limit, len(query_arr))
    start = time.perf_counter()
    for v in query_arr[:n_single]:
        filt.query(v[np.newaxis, :], tau=tau_mid)
    elapsed_single = time.perf_counter() - start
    query_latency_us = elapsed_single / max(n_single, 1) * 1e6

    predicted: List[bool] = []
    for block in _iter_query_batches(query_arr, _QUERY_BATCH):
        predicted.extend(filt.query(block, tau=tau_mid))
    rates = _confusion_rates(predicted, np.asarray(actual_results, dtype=bool))

    return {
        "dataset": dataset, "n": n, "k": k, "seed": seed,
        "construct_time_s": construct_time_s,
        "query_latency_us": query_latency_us,
        "fpr": rates["fpr"],
        "memory_bytes": memory_bytes,
    }


def streaming_method_worker(
    dim: int,
    dataset: str,
    k: int,
    seed: int,
    fold_offset: int,
    tau_mid: float,
    method: str,
    method_kwargs: Dict[str, Any],
    vectors_stream_pool: VectorInput,
    stream_len: int,
    vectors_negative: VectorInput,
    checkpoints: List[int],
    sweep: str,
) -> List[Dict[str, Any]]:
    """Streams one insertion method (M1-M5, see filters.streaming_insertion)
    over a fold_offset-windowed slice of vectors_stream_pool; at each
    checkpoint, reports FPR, log10_j_tau, ball count, and amortized cost per
    insertion. Serves both the position sweep (sweep="position") and the k
    sweep (sweep="k_sweep"); rows are tagged with sweep/method/fold_offset so
    they're self-describing regardless of completion order. FNR is asserted
    (raises on violation), never plotted."""
    stream_arr = _to_construct_array(vectors_stream_pool)[:stream_len]
    negative_arr = _to_construct_array(vectors_negative)
    n = len(stream_arr)
    checkpoint_set = sorted(set(c for c in checkpoints if 0 < c <= n))

    m = streaming_insertion.make_method(method, dim=dim, k=k, seed=seed, **method_kwargs)

    rows: List[Dict[str, Any]] = []
    cumulative_time_s = 0.0
    ci = 0
    for t in range(n):
        pos = t + 1
        t0 = time.perf_counter()
        m.insert(stream_arr[t])
        cumulative_time_s += time.perf_counter() - t0

        if ci < len(checkpoint_set) and pos == checkpoint_set[ci]:
            predicted_neg = m.query(negative_arr, tau=tau_mid)
            actual_neg = np.zeros(len(negative_arr), dtype=bool)
            rates = _confusion_rates(predicted_neg.tolist(), actual_neg)

            predicted_seen = np.asarray(m.query(stream_arr[:pos], tau=tau_mid), dtype=bool)
            fnr = float(np.mean(~predicted_seen)) if pos else 0.0
            if fnr != 0.0:
                raise ValueError(
                    f"Zero-false-negative invariant violated (method={method}, dataset={dataset}, "
                    f"k={k}, seed={seed}, fold_offset={fold_offset}, position={pos}): fnr={fnr}"
                )

            radii = m.all_radiuses
            rows.append({
                "dataset": dataset, "method": method, "k": k, "seed": seed, "fold_offset": fold_offset,
                "b": method_kwargs.get("b"), "position": pos,
                "fpr": rates["fpr"],
                "log10_j_tau": _log10_J(radii, dim, tau=tau_mid),
                "n_balls": len(radii),
                "cumulative_time_s": cumulative_time_s,
                "amortized_cost_s": cumulative_time_s / pos,
                "metric": "checkpoint", "sweep": sweep,
            })
            ci += 1

    return rows


def stream_m4_exactness_worker(
    dim: int,
    dataset: str,
    k: int,
    seed: int,
    tau_mid: float,
    vectors_stream_pool: VectorInput,
    stream_len: int,
) -> Dict[str, Any]:
    """Runs CachedHeapMerge (M4) and the exhaustive Algorithm 2 reference in
    lockstep, comparing resulting ball sets (sorted radii) after every
    post-bootstrap insertion. Feeds the paper's "M4 exactness" text sentence."""
    stream_arr = _to_construct_array(vectors_stream_pool)[:stream_len]
    n = len(stream_arr)

    m4 = streaming_insertion.make_method("M4", dim=dim, k=k, seed=seed)
    ref = BallCoverFilter(dim=dim, distance_threshold=tau_mid, max_spheres=k, seed=seed,
                            merge_criterion="log_space")
    ref.construct(np.empty((0, dim), dtype=np.float32))

    n_checked = 0
    n_mismatches = 0
    first_mismatch_position: Optional[int] = None
    for t in range(n):
        x = stream_arr[t]
        m4.insert(x)
        streaming_insertion.exhaustive_algo2_step(ref, x)

        pos = t + 1
        if pos > k:
            n_checked += 1
            r_m4 = np.sort(np.round(m4.all_radiuses.astype(np.float64), 6))
            r_ref = np.sort(np.round(ref.all_radiuses.astype(np.float64), 6))
            same = len(r_m4) == len(r_ref) and np.allclose(r_m4, r_ref, atol=1e-4, rtol=1e-4)
            if not same:
                n_mismatches += 1
                if first_mismatch_position is None:
                    first_mismatch_position = pos

    return {
        "dataset": dataset, "k": k, "seed": seed,
        "n_checked": n_checked, "n_mismatches": n_mismatches,
        "first_mismatch_position": first_mismatch_position,
    }


def stream_m3_gap_worker(
    dim: int,
    dataset: str,
    k: int,
    seed: int,
    tau_mid: float,
    vectors_stream_pool: VectorInput,
    stream_len: int,
) -> Dict[str, Any]:
    """Runs RestrictedMerge (M3, track_gap=True); returns the fraction of
    post-bootstrap insertions where the unrestricted argmin would have
    excluded the arriving singleton. Feeds the paper's "M3 gap" sentence."""
    stream_arr = _to_construct_array(vectors_stream_pool)[:stream_len]
    m3 = streaming_insertion.make_method("M3", dim=dim, k=k, seed=seed, track_gap=True)
    for x in stream_arr:
        m3.insert(x)
    return {
        "dataset": dataset, "k": k, "seed": seed,
        "n_checked": m3.n_total,
        "fraction_excluded": (m3.n_gap / m3.n_total) if m3.n_total else 0.0,
    }


def stream_delta_distribution_worker(
    dim: int,
    dataset: str,
    k: int,
    seed: int,
    tau_mid: float,
    method: str,
    vectors_stream_pool: VectorInput,
    stream_len: int,
) -> Dict[str, Any]:
    """Streams `method` in {"M3","M4","ref_exhaustive"} and summarizes the
    distribution of realized merge Deltas (mean/median/percentiles/beneficial
    fraction). Feeds the paper's Delta-distribution text statistics."""
    stream_arr = _to_construct_array(vectors_stream_pool)[:stream_len]

    if method == "ref_exhaustive":
        filt = BallCoverFilter(dim=dim, distance_threshold=tau_mid, max_spheres=k, seed=seed,
                                 merge_criterion="log_space")
        filt.construct(np.empty((0, dim), dtype=np.float32))
        deltas: List[float] = []
        for x in stream_arr:
            delta = streaming_insertion.exhaustive_algo2_step(filt, x)
            if delta is not None:
                deltas.append(delta)
    elif method in ("M3", "M4"):
        m = streaming_insertion.make_method(method, dim=dim, k=k, seed=seed)
        for x in stream_arr:
            m.insert(x)
        deltas = list(m.accepted_deltas)
    else:
        raise ValueError(f"Unknown method {method!r} for delta-distribution; expected 'M3', 'M4', or 'ref_exhaustive'")

    arr = np.asarray(deltas, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        finite = np.array([0.0])
    return {
        "dataset": dataset, "k": k, "seed": seed, "method": method,
        "n": len(arr),
        "mean": float(np.mean(finite)), "median": float(np.median(finite)),
        "p10": float(np.percentile(finite, 10)), "p90": float(np.percentile(finite, 90)),
        "min": float(np.min(finite)), "max": float(np.max(finite)),
        "frac_beneficial": float(np.mean(arr < 0)) if len(arr) else 0.0,
    }


# policy -> (metric, method) tag used by the streaming figure.
_REBUILD_POLICY_TAGS = {
    "oracle": ("static_reference", "static_reference"),
    "streaming": ("exhaustive_reference", "ref_exhaustive"),
    "periodic": ("exhaustive_reference", "ref_exhaustive"),
}


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
    merge_criterion: str = "log_space",
    sweep: Optional[str] = None,
    fold_offset: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Simulates one rebuild policy over vectors_stream, reporting FPR and
    amortized cost per checkpoint.
      "streaming": BallCoverFilter.add() per arrival, never rebuilt.
      "periodic": like "streaming", rebuilt from scratch every T arrivals.
      "oracle": rebuilt from scratch at every checkpoint.
    sweep/fold_offset are pass-through tag columns only."""
    stream_arr = _to_construct_array(vectors_stream)
    negative_arr = _to_construct_array(vectors_negative)
    n = len(stream_arr)
    checkpoint_set = sorted(set(c for c in checkpoints if 0 < c <= n))
    metric, method = _REBUILD_POLICY_TAGS.get(policy, (None, None))

    rows: List[Dict[str, Any]] = []
    cumulative_time_s = 0.0

    def _fpr(filt: BallCoverFilter) -> float:
        predicted = filt.query(negative_arr, tau=tau_mid)
        return float(np.mean(predicted)) if len(predicted) else 0.0

    if policy == "oracle":
        for c in checkpoint_set:
            filt = BallCoverFilter(dim=dim, distance_threshold=tau_mid, max_spheres=k, seed=seed,
                                     merge_criterion=merge_criterion)
            t0 = time.perf_counter()
            filt.construct(stream_arr[:c])
            cumulative_time_s += time.perf_counter() - t0
            radii = filt.all_radiuses
            rows.append({
                "dataset": dataset, "policy": policy, "T": T, "k": k, "seed": seed,
                "position": c, "fpr": _fpr(filt),
                "log10_j_tau": _log10_J(radii, dim, tau=tau_mid),
                "n_balls": len(radii),
                "cumulative_time_s": cumulative_time_s,
                "amortized_cost_s": cumulative_time_s / c,
                "metric": metric, "method": method, "sweep": sweep, "fold_offset": fold_offset,
            })
        return rows

    if policy not in ("streaming", "periodic"):
        raise ValueError(f"Unknown policy {policy!r}; expected 'streaming', 'periodic', or 'oracle'")

    rebuild_every = T if (policy == "periodic" and T is not None) else None
    filt = BallCoverFilter(dim=dim, distance_threshold=tau_mid, max_spheres=k, seed=seed,
                             merge_criterion=merge_criterion)
    t0 = time.perf_counter()
    filt.construct(np.empty((0, dim), dtype=np.float32))
    cumulative_time_s += time.perf_counter() - t0

    ci = 0
    for t in range(n):
        pos = t + 1
        t0 = time.perf_counter()
        if rebuild_every is not None and pos % rebuild_every == 0:
            filt.construct(stream_arr[:pos])
        else:
            filt.add(stream_arr[t][np.newaxis, :])
        cumulative_time_s += time.perf_counter() - t0

        if ci < len(checkpoint_set) and pos == checkpoint_set[ci]:
            radii = filt.all_radiuses
            rows.append({
                "dataset": dataset, "policy": policy, "T": T, "k": k, "seed": seed,
                "position": pos, "fpr": _fpr(filt),
                "log10_j_tau": _log10_J(radii, dim, tau=tau_mid),
                "n_balls": len(radii),
                "cumulative_time_s": cumulative_time_s,
                "amortized_cost_s": cumulative_time_s / pos,
                "metric": metric, "method": method, "sweep": sweep, "fold_offset": fold_offset,
            })
            ci += 1

    return rows


_WORKER_FNS: Dict[str, Any] = {
    'static':  static_worker,
    'dynamic': dynamic_worker,
    'tradeoff': tradeoff_worker,
    'system':  system_worker,
    'system_tau': system_tau_worker,
    'ablation': ablation_worker,
    'query_performance': query_performance_worker,
    'equal_memory': equal_memory_worker,
    'scaling': scaling_worker,
    'stream_rebuild': stream_rebuild_worker,
    'streaming_method': streaming_method_worker,
    'stream_m4_exactness': stream_m4_exactness_worker,
    'stream_m3_gap': stream_m3_gap_worker,
    'stream_delta_distribution': stream_delta_distribution_worker,
}


def _describe_job(kwargs: Dict[str, Any]) -> str:
    if 'filter_cls' in kwargs and 'lsh_cls' in kwargs:
        return f"{kwargs['filter_cls'].__name__} + {kwargs['lsh_cls'].__name__}"
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
    def __init__(self, num_workers: int = 1, output_csv: Optional[str] = None, truncate: bool = True) -> None:
        """output_csv, if set, is rewritten from every row completed so far
        after each job (not appended -- job result schemas differ, and a
        rewrite union-handles that safely), so a crash mid-sweep still
        leaves completed work on disk. truncate=False loads and preserves an
        existing file's rows instead of starting fresh."""
        self.num_workers = num_workers
        self.output_csv = output_csv
        self._jobs: List[Dict[str, Any]] = []
        self._prior_rows: List[Dict[str, Any]] = []
        if output_csv:
            if truncate:
                if os.path.exists(output_csv):
                    os.remove(output_csv)
            elif os.path.exists(output_csv):
                self._prior_rows = pd.read_csv(output_csv).to_dict("records")

    def submit(self, mode: str, **kwargs: Any) -> None:
        if mode not in _WORKER_FNS:
            raise ValueError(f"Unknown mode {mode!r}. Valid: {list(_WORKER_FNS)}")
        self._jobs.append({'mode': mode, **kwargs})

    def run(self) -> List[Dict[str, Any]]:
        return self._run_sequential() if self.num_workers == 1 else self._run_parallel()

    def clear(self) -> None:
        self._jobs.clear()

    def _rewrite_csv(self, results_so_far: List[Any]) -> None:
        if not self.output_csv:
            return
        rows: List[Dict[str, Any]] = list(self._prior_rows)
        for result in results_so_far:
            rows.extend(result if isinstance(result, list) else [result])
        if not rows:
            return
        os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)
        pd.DataFrame(rows).to_csv(self.output_csv, index=False)

    def _run_sequential(self) -> List[Dict[str, Any]]:
        results = []
        for job in self._jobs:
            fn = _WORKER_FNS[job['mode']]
            kwargs = {k: v for k, v in job.items() if k != 'mode'}
            result = fn(**kwargs)
            print(f"Completed: {_describe_result(result)}")
            results.append(result)
            self._rewrite_csv(results)
        return results

    def _run_parallel(self) -> List[Dict[str, Any]]:
        futures: Dict[Any, str] = {}
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            for job in self._jobs:
                fn = _WORKER_FNS[job['mode']]
                kwargs = {k: v for k, v in job.items() if k != 'mode'}
                f = executor.submit(fn, **kwargs)
                futures[f] = _describe_job(kwargs)
            results = []
            for f in as_completed(futures):
                result = f.result()
                print(f"Completed: {futures[f]}")
                results.append(result)
                self._rewrite_csv(results)
        return results
