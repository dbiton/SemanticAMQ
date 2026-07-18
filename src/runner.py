from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, Iterator, List, Union

import faiss
import numpy as np
import time

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


def _instantiate_filter(filter_cls: Any, dim: int, lsh: Any, memory_budget: int, distance_threshold: float = None, max_spheres: int = None) -> Any:
    if filter_cls.__name__ not in ["KDBallFilter", "KMEBFilter", "FirstNFilter"]:
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
    return _compute_metrics(predicted, actual_results, distance_threshold, config, percentage, memory_usage, memory_budget)


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


def hnsw_worker(
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
    hnsw_m: int = 32,
    ef_search: int = 64,
) -> Dict[str, Any]:
    """
    Measure how much a semantic filter reduces the cost of answering
    "is any stored vector within distance_threshold of the query?" against an
    HNSW index that explicitly stores all construct vectors.

    The filter acts like a bloom filter (one-sided error): a negative answer is
    trusted and the index is skipped; only filter-positive queries are searched
    in the HNSW index. The same query stream is timed with and without the filter.
    """
    config = f"{filter_cls.__name__} + {lsh_cls.__name__}"
    lsh = lsh_cls.build_optimized(dim, distance_threshold, None, max_L)

    construct = np.ascontiguousarray(_to_construct_array(vectors_construct), dtype=np.float32)

    # HNSW index storing all vectors explicitly (squared L2 metric)
    index = faiss.IndexHNSWFlat(dim, hnsw_m)
    index.hnsw.efSearch = ef_search
    index.add(construct)

    filter_instance = _instantiate_filter(filter_cls, dim, lsh, memory_budget, distance_threshold)
    filter_instance.construct(construct)

    sq_threshold = distance_threshold ** 2

    # PASS 1 (baseline): every query goes to the HNSW index
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


_WORKER_FNS: Dict[str, Any] = {
    'static':  static_worker,
    'dynamic': dynamic_worker,
    'tradeoff': tradeoff_worker,
    'hnsw':    hnsw_worker,
}


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
            print(f"Completed: {result['Config']} @ {result['Percentage']:.1f}%")
            results.append(result)
        return results

    def _run_parallel(self) -> List[Dict[str, Any]]:
        futures: Dict[Any, str] = {}
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            for job in self._jobs:
                fn = _WORKER_FNS[job['mode']]
                kwargs = {k: v for k, v in job.items() if k != 'mode'}
                f = executor.submit(fn, **kwargs)
                futures[f] = f"{kwargs['filter_cls'].__name__} + {kwargs['lsh_cls'].__name__}"
            results = []
            for f in as_completed(futures):
                result = f.result()
                print(f"Completed: {futures[f]}")
                results.append(result)
        return results
