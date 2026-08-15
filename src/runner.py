from __future__ import annotations

import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, Iterator, List, Optional, Union

import faiss
import numpy as np
import time

from scipy.special import logsumexp

from filters.gonzales_filter import GonzalezMEBFilter
from hyperspheres_volume import estimate_hypersphere_volume
from vector_source import VectorSource

# A worker accepts either an in-memory array or a lazy on-disk VectorSource.
VectorInput = Union[np.ndarray, VectorSource]

# Query stream is consumed in bounded-size blocks so nothing larger than this
# many rows is ever materialised at once, no matter how long the stream is.
_QUERY_BATCH = 4096


def _to_construct_array(vectors: VectorInput) -> np.ndarray:
    """
    Materialise a construct set into a single array.

    Static/immutable filters build from the full batch of keys at once, so the
    (bounded) construct slice must live in memory here -- but it is read lazily
    from disk only when the job actually runs, never copied into the job payload.
    """
    if isinstance(vectors, VectorSource):
        return vectors.to_array()
    return np.asarray(vectors)


def _iter_query_batches(vectors: VectorInput, batch_size: int) -> Iterator[np.ndarray]:
    """Yield the query stream as 2D blocks, lazily for a VectorSource."""
    if isinstance(vectors, VectorSource):
        yield from vectors.iter_batches(batch_size)
    else:
        arr = np.asarray(vectors)
        for base in range(0, len(arr), batch_size):
            yield arr[base:base + batch_size]


def _iter_query_vectors(vectors: VectorInput) -> Iterator[np.ndarray]:
    """Yield the query stream one vector at a time, lazily for a VectorSource."""
    if isinstance(vectors, VectorSource):
        yield from vectors
    else:
        yield from np.asarray(vectors)


def _compute_metrics(
    predicted: List[bool],
    actual: List[bool],
    distance_threshold: float,
    config: str,
    percentage: float,
    memory_usage: int,
    memory_budget: int,
) -> Dict[str, Any]:
    tp = sum(1 for p, a in zip(predicted, actual) if p and a)
    fp = sum(1 for p, a in zip(predicted, actual) if p and not a)
    fn = sum(1 for p, a in zip(predicted, actual) if not p and a)
    tn = sum(1 for p, a in zip(predicted, actual) if not p and not a)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

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
    """
    Mean/max/min/std of per-sphere radii. Only geometric filters (e.g.
    GonzalezMEBFilter) expose `all_radiuses`; everything else reports NaN so
    the radius plots simply skip that series.
    """
    radii = getattr(filter_instance, 'all_radiuses', None)
    if radii is None or len(radii) == 0:
        return {'Radius Mean': float('nan'), 'Radius Max': float('nan'),
                'Radius Min': float('nan'), 'Radius Std': float('nan')}
    return {
        'Radius Mean': float(np.mean(radii)), 'Radius Max': float(np.max(radii)),
        'Radius Min': float(np.min(radii)), 'Radius Std': float(np.std(radii)),
    }


_DIRECT_TAU_FILTERS = {
    "FirstNFilter", "GonzalezMEBFilter",
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

    # Stream the query set in bounded blocks so the full stream never has to be
    # resident (or copied into the job) at once.
    predicted: List[bool] = []
    for block in _iter_query_batches(vectors_query, _QUERY_BATCH):
        predicted.extend(filter_instance.query(block))

    # Checked at the end of the run: for static filters, memory is fixed after
    # construct(), but querying can't change it, so this reflects final usage.
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

    # Pull the query stream one vector at a time (dynamic filters insert after
    # each miss, so this is inherently sequential); disk reads stay chunked.
    predicted: List[bool] = []
    for v in _iter_query_vectors(vectors_query):
        is_dup = filter_instance.query(np.array([v]))[0]
        predicted.append(is_dup)
        if not is_dup:
            filter_instance.insert(np.array([v]))

    # Checked at the end of the run: dynamic filters grow via insert() during
    # the query loop, so memory usage is only meaningful once it's finished.
    memory_usage = filter_instance.get_bytesize()
    return _compute_metrics(predicted, actual_results, distance_threshold, config, percentage, memory_usage, memory_budget)


def _f1_score(predicted: np.ndarray, actual: np.ndarray) -> float:
    if len(predicted) == 0:
        return 0.0
    tp = int(np.sum(predicted & actual))
    fp = int(np.sum(predicted & ~actual))
    fn = int(np.sum(~predicted & actual))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


# Filters exposing a cheap, purely post-construction query-time dial: rebuild
# is unnecessary, so every candidate can be scored against the calibration
# set on the *same* constructed instance.
_POST_CONSTRUCTION_CONFIG_DIALS = {
    "DSBFFilter": lambda f: [("vote_threshold", t) for t in range(1, f.num_tables + 1)],
}

# Filters whose "configuration" choice changes memory layout at construction
# time (so each candidate needs its own build): mapped to the constructor
# kwarg(s) swept, one full (filter_cls, kwargs) build per candidate.
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
    """
    Build filter_cls at memory_budget and, for filters that expose a
    discrete "valid configuration" choice (see the two dial registries
    above), pick whichever candidate scores best (F1) on a held-out
    calibration workload -- distinct from both the construct set and the
    final evaluation query set. Filters without a registered dial just use
    their single memory-budget-derived default construction.
    """
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
    """
    Equal-memory comparison: every filter is given the same memory_budget,
    the best available "valid configuration" of each (see
    _build_calibrated_filter) is selected on a calibration workload disjoint
    from both the construct set and the final query set, and FPR/FNR/latency/
    throughput are then measured on that held-out query set.
    """
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

    predicted_arr = np.asarray(predicted, dtype=bool)
    actual_arr = np.asarray(actual_query, dtype=bool)
    tp = int(np.sum(predicted_arr & actual_arr))
    fp = int(np.sum(predicted_arr & ~actual_arr))
    fn = int(np.sum(~predicted_arr & actual_arr))
    tn = int(np.sum(~predicted_arr & ~actual_arr))
    n_q = max(len(actual_arr), 1)

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
    """
    Bounds on log(V_filter / V_original): how many (natural-log) times larger
    a filter's covered hypersphere volume is than the true duplicate region
    (the union of distance_threshold-balls around every construct vector).

    Kept in log space rather than exponentiated to a raw ratio: in high
    dimensions the log-volume gap can be large enough that exp() overflows to
    +inf, which is a legitimate value but useless for plotting/comparison.

    Every per-point true-duplicate ball is provably contained in the filter
    ball of its cluster (radius = MEB_radius + distance_threshold), so the
    ratio is always >= 1, i.e. log-ratio >= 0. Bounds combine the loosest
    pairing of the two (lower/upper) volume-bound estimates; the lower bound
    is floored at 0.0 to absorb bound looseness.
    """
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
    """
    Measure how much a semantic filter reduces the cost of answering
    "is any stored vector within distance_threshold of the query?" against a
    flat L2 index that explicitly stores all construct vectors.

    The filter acts like a bloom filter (one-sided error): a negative answer is
    trusted and the index is skipped; only filter-positive queries are searched
    in the flat index. The same query stream is timed with and without the filter.

    orig_vol_log_low/orig_vol_log_high are the caller-precomputed log-volume
    bounds of the union of true-duplicate balls (radius distance_threshold
    around every construct vector) -- geometry-only, so they depend on the
    dataset and threshold but not on the filter or memory budget, and are
    reused across the whole memory-budget sweep instead of being recomputed
    per job. Combined with the filter's own covering-sphere volume (returned
    by construct(), when the filter is geometric), they produce the FPVR
    (False Positive Volume Ratio) bounds below -- a ground-truth-free proxy
    for false-positive rate, since this worker has no per-query labels.
    """
    config = f"{filter_cls.__name__} + {lsh_cls.__name__}"
    lsh = lsh_cls.build_optimized(dim, distance_threshold, None, max_L)

    construct = np.ascontiguousarray(_to_construct_array(vectors_construct), dtype=np.float32)

    # Flat L2 index storing all vectors explicitly (squared L2 metric)
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

    # PASS 1 (baseline): every query goes to the flat index
    n_queries = 0
    baseline_pred: List[bool] = []
    start = time.perf_counter()
    for block in _iter_query_batches(vectors_query, _QUERY_BATCH):
        block = np.ascontiguousarray(block, dtype=np.float32)
        n_queries += len(block)
        nn_sq_dist, _ = index.search(block, 1)
        baseline_pred.extend((nn_sq_dist[:, 0] <= sq_threshold).tolist())
    time_without = time.perf_counter() - start

    # PASS 2 (filtered): trust filter negatives, search the index only for positives
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

    # sanity: a correct one-sided filter never loses a baseline positive
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
    """
    function for executing a single benchmarking trial for a specific filter configuration, 
    measuring initialization time, memory usage, query throughput, and accuracy metrics.
    """
    try:
        config = f"{filter_cls.__name__} + {lsh_cls.__name__}"
        lsh = lsh_cls.build_optimized(dim, distance_threshold, None, max_L)
        vectors_construct, vectors_query = _to_construct_array(vectors_construct), _to_construct_array(vectors_query)

        construct_start = time.perf_counter() # start construct timer
        filter_instance = _instantiate_filter(filter_cls, dim, lsh, memory_budget, distance_threshold, max_spheres)
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
    """Linear J_tau(C) = sum_i (r_i + tau)^d. Overflows to +inf for d in the
    hundreds (float64 maxes out around 1e308, and r**d blows past that for
    any r > ~1 at d=384) -- _log10_J is the version safe to actually plot;
    this is kept only for callers that want the raw value when it happens to
    be representable."""
    r = np.asarray(radii, dtype=np.float64) + float(tau)
    if r.size == 0:
        return 0.0
    with np.errstate(over='ignore'):
        return float(np.sum(r ** dim))


def _confusion_rates(predicted: List[bool], actual: np.ndarray) -> Dict[str, float]:
    predicted = np.asarray(predicted, dtype=bool)
    tp = int(np.sum(predicted & actual))
    fp = int(np.sum(predicted & ~actual))
    fn = int(np.sum(~predicted & actual))
    tn = int(np.sum(~predicted & ~actual))
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
    """
    Build ONE GonzalezMEBFilter cover per (dataset, k, seed) via the full
    (level 1-2) construction ladder, then evaluate that same cover at every
    tau in `tau_values`. Returns one row per (stage, tau) matching the schema
    plot_ablation.generate_ablation_figures expects: dataset, k, stage,
    stage_index, tau, log10_J, log10_J_tau, fpr, fnr, stage_time_s,
    cum_time_s, n_balls, seed. Reported stages are ["ball", "quantizing",
    "feedback"] -- the level-1 initial cover (k-means++ seeding, no radius
    refinement) never gets its own row since it holds no meaningful spheres
    on its own; its cost is folded into "ball"'s stage_time_s/cum_time_s as
    a necessary precursor instead.

    log10_J is a pure geometric quantity of the cover (sum r_i^d, no tau
    involved) -- panel (a) aggregates it "at the selected k" only, without
    conditioning on tau, so it must be genuinely tau-independent for a given
    (dataset, k, stage, seed); it would be wrong to rebuild the cover per tau
    (that bakes tau into the radii and turns different-tau rows into
    apples-to-oranges "replicates"). `construct_tau` is therefore a single,
    sweep-independent threshold used only for the level-1 k-means++ stopping
    criterion (a construction-tightness knob) -- distinct from the evaluated
    tau's, which only affect log10_J_tau/fpr/fnr (computed post hoc from the
    same radii) and the feedback branch below.

    "ball" is built via the full ladder (set_complexity defaults to 2) with
    retain_assignments=True, so the deep copy taken below for the feedback
    stage -- and quantize_centers()'s own rebuild-from-vectors path -- has
    the construction state they need.

    "quantizing" applies GonzalezMEBFilter.quantize_centers() to that same
    cover in place: it rebuilds each FAISS ScalarQuantizer candidate from the
    retained vectors at its own bytes-rescaled effective_k and keeps
    whichever configuration (including staying unquantized) minimizes
    log10 J(C), so n_balls can differ from k here. Its stage_time_s is the
    wall-clock cost of the quantize_centers() call itself, not the ladder's
    self-reported per-level times.

    Feedback *is* legitimately tau-specific -- which points count as a
    witnessed false positive depends on tau -- so it branches once per tau
    off a deep copy of the post-quantization cover.
    """
    construct_arr = _to_construct_array(vectors_construct)
    query_arr = _to_construct_array(vectors_query)
    n_q = len(query_arr)
    for t, actual in actual_results_by_tau.items():
        if len(actual) != n_q:
            raise ValueError(
                f"vectors_query ({n_q}) and actual_results_by_tau[{t}] ({len(actual)}) length mismatch"
            )

    # Single witness/eval split, shared across every tau for this (k, seed) --
    # a held-out workload disjoint from the points feedback saw.
    n_witness = max(1, int(round(n_q * feedback_fraction))) if n_q > 0 else 0
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_q)
    witness_idx, eval_idx = perm[:n_witness], perm[n_witness:]
    witness_q, eval_q = query_arr[witness_idx], query_arr[eval_idx]
    actual_by_tau = {t: np.asarray(a, dtype=bool) for t, a in actual_results_by_tau.items()}

    rows: List[Dict[str, Any]] = []

    filt = GonzalezMEBFilter(dim=dim, distance_threshold=construct_tau, max_spheres=k, seed=seed)
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

    # quantizing: compress the ball-refined cover's centers in place
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

    # feedback: branched per tau off a copy of the post-quantization cover
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
    """
    Build ONE GonzalezMEBFilter cover (fixed ball budget k) per (dataset,
    seed) against a flat L2 index storing the same construct set, then sweep
    the query-time similarity threshold tau over that single construction --
    mirrors ablation_worker's "build once, evaluate at every tau" shape, but
    measures system_worker's two-pass filtered-access throughput/index-access
    metrics at each tau instead of a memory-budget sweep at one fixed tau.

    tau is purely a query-time knob here (GonzalezMEBFilter.query(tau=...)),
    so it never touches the cover itself; `construct_tau` is a single,
    sweep-independent construction-tightness knob, same role it plays in
    ablation_worker. The query stream (vectors_query) is fixed-length and
    reused unchanged across the whole sweep -- only tau (and therefore how
    much of it counts as a ground-truth duplicate) changes.

    Returns one row per tau with:
      - redundancy_rate: the *measured* fraction of the fixed query stream
        that's an actual duplicate at that tau (mean of actual_results_by_tau[tau]),
        which is what the figure's shared x-axis plots against -- not the
        nominal percentage used upstream to pick tau, since
        calculate_nn_distance_threshold's sampled percentile and this
        exact nearest-construct-neighbour rate aren't identical.
      - throughput_improvement_pct: % throughput gain of the filtered
        pipeline (trust filter negatives, only hit the index on positives)
        over the unfiltered flat-index baseline, both timed on the same
        query stream.
      - mean_index_accesses: count of queries that reached the flat index
        (filter positives) -- "mean" once aggregated over the seed sweep at
        plot time via plot_common.mean_ci95.
      - fpr/fnr: empirical confusion rates against actual_results_by_tau[tau].
      - j_tau/log10_j_tau: the threshold-expanded geometric objective
        J_tau(C) = sum_i (r_i + tau)^d, linear (may be +inf at this
        dimensionality) and log10 (always finite) -- see _J_tau/_log10_J.
    """
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

    filt = GonzalezMEBFilter(dim=dim, distance_threshold=construct_tau, max_spheres=k, seed=seed)
    filt.construct(construct_arr)
    radii = filt.all_radiuses

    rows: List[Dict[str, Any]] = []
    for t in tau_values:
        actual = np.asarray(actual_results_by_tau[t], dtype=bool)
        redundancy_rate = float(np.mean(actual)) if len(actual) > 0 else 0.0

        # PASS 1 (baseline): every query goes to the flat index
        start = time.perf_counter()
        index.search(query_arr, 1)
        time_without = time.perf_counter() - start

        # PASS 2 (filtered): trust filter negatives, search the index only for positives
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
    """
    Build one GonzalezMEBFilter cover (max_spheres=k, default complexity=4,
    i.e. the full construction ladder) per (dataset, k, seed), then time
    querying that same cover two ways: one query vector at a time
    (_iter_query_vectors) vs. _QUERY_BATCH-sized blocks (_iter_query_batches).
    GonzalezMEBFilter.query() is fully vectorized with no separate
    single-query code path -- "single vs batched" is purely how many rows
    the caller passes per call. Single-mode timing is capped at
    single_query_limit queries since it pays one Python-level call per query;
    batched mode already covers the full stream cheaply and is left uncapped.
    Returns one row per mode ("single"/"batched").
    """
    construct_arr = _to_construct_array(vectors_construct)

    filt = GonzalezMEBFilter(dim=dim, distance_threshold=construct_tau, max_spheres=k, seed=seed)
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
    """Build one GonzalezMEBFilter at fixed k over n = len(vectors_construct)
    vectors; reports construction time, single-query latency, memory, and FPR
    against actual_results (ground truth for this n, from the caller)."""
    construct_arr = _to_construct_array(vectors_construct)
    query_arr = _to_construct_array(vectors_query)
    n = len(construct_arr)

    construct_start = time.perf_counter()
    filt = GonzalezMEBFilter(dim=dim, distance_threshold=tau_mid, max_spheres=k, seed=seed)
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


def stream_absorption_worker(
    dim: int,
    dataset: str,
    k: int,
    seed: int,
    tau_mid: float,
    vectors_stream: VectorInput,
    checkpoints: List[int],
    merge_criterion: str = "log_space",
) -> List[Dict[str, Any]]:
    """Stream vectors_stream through GonzalezMEBFilter.add() one at a time; at
    each checkpoint, report the fraction of arrivals since the last checkpoint
    that already landed inside an existing ball (tau=0) before insertion."""
    stream_arr = _to_construct_array(vectors_stream)
    n = len(stream_arr)

    filt = GonzalezMEBFilter(dim=dim, distance_threshold=tau_mid, max_spheres=k, seed=seed,
                             merge_criterion=merge_criterion)
    filt.construct(np.empty((0, dim), dtype=np.float32))

    checkpoint_set = sorted(set(c for c in checkpoints if 0 < c <= n))
    rows: List[Dict[str, Any]] = []
    window_hits = 0
    window_count = 0
    window_start = 1
    ci = 0

    for t in range(n):
        v = stream_arr[t]
        absorbed = filt.query(v[np.newaxis, :], tau=0.0)[0]
        window_hits += int(absorbed)
        window_count += 1
        filt.add(v[np.newaxis, :])

        pos = t + 1
        if ci < len(checkpoint_set) and pos == checkpoint_set[ci]:
            rows.append({
                "dataset": dataset, "k": k, "seed": seed,
                "position": pos, "window_start": window_start,
                "absorption_rate": window_hits / max(window_count, 1),
                "n_balls": len(filt.all_spheres),
            })
            window_start = pos + 1
            window_hits = 0
            window_count = 0
            ci += 1

    return rows


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
) -> List[Dict[str, Any]]:
    """
    Simulate one rebuild policy over vectors_stream; at each checkpoint,
    report FPR against a fixed negative set plus the amortized wall-clock
    cost per insertion (including rebuilds) spent so far.

      "streaming": GonzalezMEBFilter.add() per arrival, never rebuilt.
      "periodic": like "streaming", but every T arrivals the cover is fully
        reconstructed from scratch.
      "oracle": reconstructed from scratch at every checkpoint -- the best
        achievable cover for that many points.
    """
    stream_arr = _to_construct_array(vectors_stream)
    negative_arr = _to_construct_array(vectors_negative)
    n = len(stream_arr)
    checkpoint_set = sorted(set(c for c in checkpoints if 0 < c <= n))

    rows: List[Dict[str, Any]] = []
    cumulative_time_s = 0.0

    def _fpr(filt: GonzalezMEBFilter) -> float:
        predicted = filt.query(negative_arr, tau=tau_mid)
        return float(np.mean(predicted)) if len(predicted) else 0.0

    if policy == "oracle":
        for c in checkpoint_set:
            filt = GonzalezMEBFilter(dim=dim, distance_threshold=tau_mid, max_spheres=k, seed=seed,
                                     merge_criterion=merge_criterion)
            t0 = time.perf_counter()
            filt.construct(stream_arr[:c])
            cumulative_time_s += time.perf_counter() - t0
            rows.append({
                "dataset": dataset, "policy": policy, "T": T, "k": k, "seed": seed,
                "position": c, "fpr": _fpr(filt),
                "cumulative_time_s": cumulative_time_s,
                "amortized_cost_s": cumulative_time_s / c,
            })
        return rows

    if policy not in ("streaming", "periodic"):
        raise ValueError(f"Unknown policy {policy!r}; expected 'streaming', 'periodic', or 'oracle'")

    rebuild_every = T if (policy == "periodic" and T is not None) else None
    filt = GonzalezMEBFilter(dim=dim, distance_threshold=tau_mid, max_spheres=k, seed=seed,
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
            rows.append({
                "dataset": dataset, "policy": policy, "T": T, "k": k, "seed": seed,
                "position": pos, "fpr": _fpr(filt),
                "cumulative_time_s": cumulative_time_s,
                "amortized_cost_s": cumulative_time_s / pos,
            })
            ci += 1

    return rows


def stream_merge_criterion_worker(
    dim: int,
    dataset: str,
    k: int,
    seed: int,
    tau_mid: float,
    merge_criterion: str,
    vectors_stream: VectorInput,
    vectors_negative: VectorInput,
) -> Dict[str, Any]:
    """
    Stream the full vectors_stream through GonzalezMEBFilter.add() at a fixed
    k under a single merge_criterion (see GonzalezMEBFilter.MERGE_CRITERIA),
    and report the FINAL FPR against a fixed held-out negative set once the
    whole stream has been absorbed -- one (k, merge_criterion) point for
    panel (c)'s "final FPR vs k, one line per merge criterion".
    """
    stream_arr = _to_construct_array(vectors_stream)
    negative_arr = _to_construct_array(vectors_negative)

    filt = GonzalezMEBFilter(dim=dim, distance_threshold=tau_mid, max_spheres=k, seed=seed,
                             merge_criterion=merge_criterion)
    filt.construct(np.empty((0, dim), dtype=np.float32))
    for v in stream_arr:
        filt.add(v[np.newaxis, :])

    predicted = filt.query(negative_arr, tau=tau_mid)
    fpr = float(np.mean(predicted)) if len(predicted) else 0.0

    return {
        "dataset": dataset, "k": k, "seed": seed,
        "merge_criterion": merge_criterion,
        "fpr": fpr,
    }


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
    'stream_absorption': stream_absorption_worker,
    'stream_rebuild': stream_rebuild_worker,
    'stream_merge_criterion': stream_merge_criterion_worker,
}


def _describe_job(kwargs: Dict[str, Any]) -> str:
    if 'filter_cls' in kwargs and 'lsh_cls' in kwargs:
        return f"{kwargs['filter_cls'].__name__} + {kwargs['lsh_cls'].__name__}"
    if {'dataset', 'k', 'tau_values', 'seed'} <= kwargs.keys():
        return f"{kwargs['dataset']} k={kwargs['k']} tau_values={kwargs['tau_values']} seed={kwargs['seed']}"
    if {'dataset', 'policy', 'k', 'seed'} <= kwargs.keys():
        extra = f" T={kwargs['T']}" if kwargs.get('T') is not None else ""
        return f"{kwargs['dataset']} policy={kwargs['policy']}{extra} k={kwargs['k']} seed={kwargs['seed']}"
    if {'dataset', 'merge_criterion', 'k', 'seed'} <= kwargs.keys():
        return f"{kwargs['dataset']} merge_criterion={kwargs['merge_criterion']} k={kwargs['k']} seed={kwargs['seed']}"
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
    def __init__(self, num_workers: int = 1) -> None:
        self.num_workers = num_workers
        self._jobs: List[Dict[str, Any]] = []

    def submit(self, mode: str, **kwargs: Any) -> None:
        if mode not in _WORKER_FNS:
            raise ValueError(f"Unknown mode {mode!r}. Valid: {list(_WORKER_FNS)}")
        self._jobs.append({'mode': mode, **kwargs})

    def run(self) -> List[Dict[str, Any]]:
        return self._run_sequential() if self.num_workers == 1 else self._run_parallel()
    
    def clear(self) -> None:
        self._jobs.clear()

    def _run_sequential(self) -> List[Dict[str, Any]]:
        results = []
        for job in self._jobs:
            fn = _WORKER_FNS[job['mode']]
            kwargs = {k: v for k, v in job.items() if k != 'mode'}
            result = fn(**kwargs)
            print(f"Completed: {_describe_result(result)}")
            results.append(result)
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
        return results
