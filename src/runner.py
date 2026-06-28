from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List

import numpy as np


def _compute_metrics(
    predicted: List[bool],
    actual: List[bool],
    distance_threshold: float,
    config: str,
    percentage: float,
) -> Dict[str, Any]:
    tp = sum(1 for p, a in zip(predicted, actual) if p and a)
    fp = sum(1 for p, a in zip(predicted, actual) if p and not a)
    fn = sum(1 for p, a in zip(predicted, actual) if not p and a)
    tn = sum(1 for p, a in zip(predicted, actual) if not p and not a)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'Config':     config,
        'Percentage': percentage,
        'Threshold':  distance_threshold,
        'FPR':        fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        'FNR':        fn / (fn + tp) if (fn + tp) > 0 else 0.0,
        'Precision':  precision,
        'Recall':     recall,
        'F1-Score':   f1,
    }


def _instantiate_filter(filter_cls: Any, dim: int, lsh: Any, memory_budget: int, distance_threshold: float) -> Any:
    try:
        return filter_cls(dim=dim, lsh=lsh, max_memory=memory_budget)
    except TypeError:
        try:
            return filter_cls(dim=dim, lsh=lsh)
        except TypeError:
            return filter_cls(dim=dim, max_memory=memory_budget, distance_threshold=distance_threshold)


def static_worker(
    filter_cls: Any,
    lsh_cls: Any,
    dim: int,
    distance_threshold: float,
    max_L: int,
    memory_budget: int,
    vectors_construct: np.ndarray,
    vectors_query: np.ndarray,
    actual_results: List[bool],
    percentage: float,
) -> Dict[str, Any]:
    config = f"{filter_cls.__name__} + {lsh_cls.__name__}"
    lsh = lsh_cls.build_optimized(dim, distance_threshold, None, max_L)
    filter_instance = _instantiate_filter(filter_cls, dim, lsh, memory_budget, distance_threshold)
    filter_instance.construct(vectors_construct)
    predicted = filter_instance.query(vectors_query)
    return _compute_metrics(predicted, actual_results, distance_threshold, config, percentage)


def dynamic_worker(
    filter_cls: Any,
    lsh_cls: Any,
    dim: int,
    distance_threshold: float,
    max_L: int,
    memory_budget: int,
    vectors_construct: np.ndarray,
    vectors_query: np.ndarray,
    actual_results: List[bool],
    percentage: float,
) -> Dict[str, Any]:
    config = f"{filter_cls.__name__} + {lsh_cls.__name__}"
    lsh = lsh_cls.build_optimized(dim, distance_threshold, None, max_L)
    filter_instance = _instantiate_filter(filter_cls, dim, lsh, memory_budget)
    filter_instance.construct(vectors_construct)

    predicted: List[bool] = []
    for v in vectors_query:
        is_dup = filter_instance.query(np.array([v]))[0]
        predicted.append(is_dup)
        if not is_dup:
            filter_instance.insert(np.array([v]))

    return _compute_metrics(predicted, actual_results, distance_threshold, config, percentage)


_WORKER_FNS: Dict[str, Any] = {
    'static':  static_worker,
    'dynamic': dynamic_worker,
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
