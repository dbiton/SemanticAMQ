from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, Iterator, List, Union

import numpy as np
import time
import inspect

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
    }


def _instantiate_filter(filter_cls: Any, dim: int, lsh_cls: Any, max_L: int, memory_budget: int, distance_threshold: float = None, max_shapes: int = None) -> Any:
    if 'max_shapes' in inspect.signature(filter_cls.__init__).parameters:
        return filter_cls(dim=dim, max_memory=memory_budget, distance_threshold=distance_threshold, max_shapes=max_shapes)
    else:
        lsh = lsh_cls.build_optimized(dim, distance_threshold, None, max_L)
        try:
            return filter_cls(dim=dim, lsh=lsh, max_memory=memory_budget)
        except TypeError:
            return filter_cls(dim=dim, lsh=lsh)


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
    filter_instance = _instantiate_filter(filter_cls, dim, lsh_cls, max_L, memory_budget, distance_threshold)
    filter_instance.construct(_to_construct_array(vectors_construct))

    # Stream the query set in bounded blocks so the full stream never has to be
    # resident (or copied into the job) at once.
    predicted: List[bool] = []
    for block in _iter_query_batches(vectors_query, _QUERY_BATCH):
        predicted.extend(filter_instance.query(block))

    # Checked at the end of the run: for static filters, memory is fixed after
    # construct(), but querying can't change it, so this reflects final usage.
    memory_usage = filter_instance.get_bytesize()
    return _compute_metrics(predicted, actual_results, distance_threshold, config, percentage, memory_usage)


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
    filter_instance = _instantiate_filter(filter_cls, dim, lsh_cls, max_L, memory_budget, distance_threshold)
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
    return _compute_metrics(predicted, actual_results, distance_threshold, config, percentage, memory_usage)


def _compute_tradeoff_metrics(
    predicted: List[bool],
    actual: List[bool],
    config: str,
    max_shapes: int,
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
        "NumShapes": max_shapes if max_shapes is not None else 0,
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
    lsh_cls: Any,
    dim: int,
    distance_threshold: float,
    max_L: int,
    vectors_construct: np.ndarray,
    vectors_query: np.ndarray,
    actual_results: List[bool],
    max_shapes: int = None,
    memory_budget: int = None
) -> Dict[str, Any]:
    """
    Function for executing a single benchmarking trial of a static filter,
    constructing the filter once from the full dataset and measuring
    construction time, memory usage, query throughput, and accuracy metrics.
    """
    try:
        config = f"{filter_cls.__name__} + {lsh_cls.__name__}"
        vectors_construct, vectors_query = _to_construct_array(vectors_construct), _to_construct_array(vectors_query)

        construct_start = time.perf_counter() # start construct timer
        filter_instance = _instantiate_filter(filter_cls, dim, lsh_cls, max_L, memory_budget, distance_threshold, max_shapes)
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

    return _compute_tradeoff_metrics(predicted, actual_results, config, max_shapes, memory_budget, memory_usage, construct_latency, qps)


def tradeoff_dynamic_worker(
    filter_cls: Any,
    lsh_cls: Any,
    dim: int,
    distance_threshold: float,
    max_L: int,
    vectors_construct: np.ndarray,
    vectors_query: np.ndarray,
    actual_results: List[bool],
    max_shapes: int = None,
    memory_budget: int = None
) -> Dict[str, Any]:
    """
    Function for executing a single benchmarking trial of a dynamic filter,
    where vectors are inserted incrementally while measuring insertion time,
    memory usage, query throughput, and accuracy metrics.
    """
    try:
        config = f"{filter_cls.__name__} + {lsh_cls.__name__}"
        vectors_construct, vectors_query = _to_construct_array(vectors_construct), _to_construct_array(vectors_query)

        construct_start = time.perf_counter() # start construct timer
        filter_instance = _instantiate_filter(filter_cls, dim, lsh_cls, max_L, memory_budget, distance_threshold, max_shapes)
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
        print(f"  [Skipped] {config} failed: {e}")
        return None

    return _compute_tradeoff_metrics(predicted, actual_results, config, max_shapes, memory_budget, memory_usage, construct_latency, qps)


_WORKER_FNS: Dict[str, Any] = {
    'static':  static_worker,
    'dynamic': dynamic_worker,
    'tradeoff_static': tradeoff_static_worker,
    'tradeoff_dynamic': tradeoff_dynamic_worker
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
