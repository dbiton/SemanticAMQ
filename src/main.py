import os

import faiss
import time
import numpy as np
import pandas as pd

from filters.cuckoo_filter import CuckooFilter
from filters.gcs_filter import GCSFilter
from filters.pyfusefilter_filters import Fuse8Filter, Xor8Filter
from filters.quotient_filter import QuotientFilterWrapper
from lsh import FaissSimHashLSH, LSH, DummyLSH
from runner import Runner
from plots import plot_results, plot_throughput_results, plot_tradeoff_results, plot_hnsw_results, plot_equal_memory_results
from plot_ablation import generate_ablation_figures
from plot_query_performance import generate_query_performance_figures
from plot_equal_memory import generate_equal_memory_figures
from typing import Tuple, List
from filters.first_n_filter import FirstNFilter
from filters.gonzales_filter import GonzalezMEBFilter
from filters.dsbf_filter import DSBFFilter
from filters.covering_lsh_fallback_filter import CoveringLSHFallbackFilter
from filters.lsbf_filter import LSBFFilter
from filters.dsh_filter import DSHFilter
from hyperspheres_volume import estimate_hypersphere_volume
from util.dataset_paths import dataset_filenames
from util.fetch_datasets import load_embeds
from vector_source import VectorSource, ArrayVectorSource

MAX_WORKERS = 6
DATASET = 'ELI5'

dynamic_filters = [GonzalezMEBFilter, DSBFFilter, CoveringLSHFallbackFilter, LSBFFilter, DSHFilter]
static_filters  = []

def calculate_max_memory_budget_bytes(num_vectors: int, dimension: int, bytes_per_element: float) -> int:
    return num_vectors * dimension * bytes_per_element * 6

def calculate_memory_budgets(max_memory_budget: int, count_steps: int = 10, include_zero: bool = False) -> list[int]:
    """Uniformly spaced budgets from max/count_steps (or 0 if `include_zero`) up to the full max budget."""
    start = 0 if include_zero else max_memory_budget / count_steps
    return [int(b) for b in np.linspace(start, max_memory_budget, count_steps)]

def calculate_distance_threshold(
    vectors: np.ndarray,
    percentage: float,
    max_pairs: int = 20_000_000,
    seed: int = 42,
) -> float:
    """
    Return the distance threshold at which `percentage`% of all pairwise L2
    distances are considered duplicates (i.e., the `percentage`-th percentile).

    Computing every pairwise distance is O(n^2) in time and memory -- an n x n
    matrix plus an n(n-1)/2 distance array (~40 GB at n=100k), which is why the
    exact version blows up. Instead we sample up to `max_pairs // n` query rows
    and use faiss to compute their *exact* L2 distances against the full set
    (BLAS-accelerated, multithreaded). Each sampled query contributes a complete
    distance row, so the pooled distances are an unbiased sample of the pairwise
    distribution; the percentile estimate is within ~0.2% of exact even in the
    1-9% tail, at bounded (~`max_pairs`-sized) memory.
    """
    n = len(vectors)
    if n < 2:
        return 0.0

    vecs = np.ascontiguousarray(vectors, dtype=np.float32)

    # Sample enough query rows to pool ~max_pairs distances (all rows when n is
    # small enough that the full set already fits the budget).
    n_query = int(np.clip(max_pairs // n, 1, n))
    if n_query >= n:
        query_idx = np.arange(n)
    else:
        query_idx = np.random.default_rng(seed).choice(n, size=n_query, replace=False)

    # faiss returns squared L2 distances (shape: n_query x n).
    sq_dists = np.asarray(faiss.pairwise_distances(vecs[query_idx], vecs))
    np.clip(sq_dists, 0, None, out=sq_dists)
    distances = np.sqrt(sq_dists)

    # Drop each sampled query's distance to itself (the 0 on the diagonal),
    # matching the exact version's exclusion of self-pairs.
    keep = np.ones(distances.shape, dtype=bool)
    keep[np.arange(len(query_idx)), query_idx] = False
    return float(np.percentile(distances[keep], percentage))


def compute_actual_results(vectors: np.ndarray, distance_threshold: float) -> list[bool]:
    """
    Online ground-truth: the first vector is unique; each subsequent vector is a
    duplicate if it falls within `distance_threshold` of any previously accepted
    unique vector, otherwise it joins the reference set.
    """
    actual   = [False]
    ref_set  = [vectors[0]]

    for v in vectors[1:]:
        ref_array = np.array(ref_set)
        distances = np.sqrt(np.sum((ref_array - v) ** 2, axis=1))
        is_dup    = bool(np.any(distances <= distance_threshold))
        actual.append(is_dup)
        if not is_dup:
            ref_set.append(v)

    return actual


def compute_static_actual_results(
    construct_vectors: np.ndarray,
    query_vectors: np.ndarray,
    distance_threshold: float,
) -> list[bool]:
    """
    Static ground-truth: a query vector is a duplicate iff it falls within
    `distance_threshold` of any vector in the fixed construct set.

    "Within threshold of any construct vector" is exactly a nearest-neighbour
    test, so we let faiss find each query's closest construct vector in one
    batched, BLAS-accelerated call instead of a Python loop that builds a full
    (n_construct,) distance array per query. faiss returns *squared* L2, so we
    compare against distance_threshold**2 (equivalent to sqrt(...) <= threshold).
    """
    construct = np.ascontiguousarray(construct_vectors, dtype=np.float32)
    queries = np.ascontiguousarray(query_vectors, dtype=np.float32)

    if len(construct) == 0:
        return [False] * len(queries)

    index = faiss.IndexFlatL2(construct.shape[1])
    index.add(construct)
    nn_sq_dist, _ = index.search(queries, 1)  # squared L2 to nearest construct vector
    return (nn_sq_dist[:, 0] <= distance_threshold ** 2).tolist()


def static_evaluation() -> list:
    num_vectors              = 1_000
    dim                      = 384
    min_percentage_duplicates = 1
    max_percentage_duplicates = 9
    count_steps              = 9
    count_memory_steps       = 10
    percentage_construct     = 10
    max_L                    = 256

    dataset = DATASET
    vectors = load_embeds(dataset, num_vectors)[0]
    bytes_per_element = vectors.itemsize
    num_construct   = num_vectors * percentage_construct // 100
    vectors_construct = vectors[:num_construct]
    vectors_query     = vectors[num_construct:]

    # Jobs receive lazy on-disk pointers instead of array copies, so the query
    # stream length is bounded by disk, not by num_jobs * stream_length memory.
    construct_source = VectorSource(dataset, 0, num_construct)
    query_source     = VectorSource(dataset, num_construct, num_vectors)
    max_memory_budget = calculate_max_memory_budget_bytes(num_vectors, dim, bytes_per_element)
    memory_budgets    = calculate_memory_budgets(max_memory_budget, count_memory_steps)
    runner = Runner(num_workers=MAX_WORKERS)

    for percentage in np.linspace(min_percentage_duplicates, max_percentage_duplicates, count_steps):
        print(f"Processing {percentage:.1f}% duplicates...")
        print("Calculating distance threshold...")
        distance_threshold = calculate_distance_threshold(vectors, percentage)
        print("Calculating actual results...")
        actual_results     = compute_static_actual_results(vectors_construct, vectors_query, distance_threshold)

        for memory_budget in memory_budgets:
            for filter_cls in static_filters + dynamic_filters:
                runner.submit(
                    'static',
                    filter_cls=filter_cls, lsh_cls=DummyLSH,
                    dim=dim, distance_threshold=distance_threshold,
                    max_L=max_L, memory_budget=memory_budget,
                    vectors_construct=construct_source, vectors_query=query_source,
                    actual_results=actual_results, percentage=percentage,
                )

    results = runner.run()
    print("Static evaluation completed.")
    plot_results(results, "static")
    return results


EQUAL_MEMORY_BUDGETS_BYTES = [64 * 1024, 256 * 1024, 1 * 1024 * 1024, 4 * 1024 * 1024, 16 * 1024 * 1024]


def equal_memory_evaluation() -> list:
    """
    Equal-memory comparison: every registered filter (GonzalezMEBFilter --
    the paper's own method -- plus the four LSH/Bloom baselines) is given the
    exact same memory budget from EQUAL_MEMORY_BUDGETS_BYTES (clipped to a
    dataset-scaled maximum via calculate_max_memory_budget_bytes, so a small
    dataset never gets a budget wildly larger than its own raw vector size).

    For each budget, every filter's best available "valid configuration" is
    chosen on a calibration workload disjoint from both the construct set and
    the final query set (see runner.py::_build_calibrated_filter -- filters
    with a discrete config dial, e.g. DSBFFilter's vote_threshold or
    LSBFFilter's verify flag, pick whichever candidate scores best on
    calibration; filters without one just use their single memory-budget-
    derived default). FPR, FNR, query latency and query throughput are then
    measured on the held-out query set and plotted one series per budget.
    """
    num_vectors             = 10_000
    dim                     = 384
    percentage_construct    = 40
    percentage_calibration  = 20
    duplicate_percentage    = 5.0
    max_L                   = 256

    dataset = DATASET
    vectors = load_embeds(dataset, num_vectors)[0]
    bytes_per_element = vectors.itemsize
    num_construct   = num_vectors * percentage_construct // 100
    num_calibration = num_vectors * percentage_calibration // 100
    vectors_construct   = vectors[:num_construct]
    vectors_calibration = vectors[num_construct:num_construct + num_calibration]
    vectors_query       = vectors[num_construct + num_calibration:]

    # Jobs receive lazy on-disk pointers instead of array copies (same
    # reasoning as static_evaluation).
    construct_source   = VectorSource(dataset, 0, num_construct)
    calibration_source = VectorSource(dataset, num_construct, num_construct + num_calibration)
    query_source        = VectorSource(dataset, num_construct + num_calibration, num_vectors)

    print("Calculating distance threshold...")
    distance_threshold = calculate_distance_threshold(vectors, duplicate_percentage)
    print("Calculating calibration/query ground truth...")
    actual_calibration = compute_static_actual_results(vectors_construct, vectors_calibration, distance_threshold)
    actual_query       = compute_static_actual_results(vectors_construct, vectors_query, distance_threshold)

    max_memory_budget = calculate_max_memory_budget_bytes(num_vectors, dim, bytes_per_element)
    memory_budgets = sorted({min(b, max_memory_budget) for b in EQUAL_MEMORY_BUDGETS_BYTES})

    runner = Runner(num_workers=MAX_WORKERS)
    for memory_budget in memory_budgets:
        print(f"Submitting jobs for memory_budget={memory_budget} bytes...")
        for filter_cls in static_filters + dynamic_filters:
            runner.submit(
                'equal_memory',
                filter_cls=filter_cls, lsh_cls=DummyLSH,
                dim=dim, distance_threshold=distance_threshold, max_L=max_L,
                memory_budget=memory_budget,
                vectors_construct=construct_source, vectors_calibration=calibration_source,
                vectors_query=query_source,
                actual_calibration=actual_calibration, actual_query=actual_query,
                dataset=dataset,
            )

    results = runner.run()
    print("Equal-memory evaluation completed.")
    plot_equal_memory_results(results)

    results_df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    results_df.to_csv("results/equal_memory.csv", index=False)
    print("Saved results/equal_memory.csv")
    generate_equal_memory_figures(df=results_df)

    return results


def hnsw_evaluation() -> list:
    """
    Measure how much a semantic filter reduces accesses to an HNSW index that
    explicitly stores all construct vectors, across every dataset present on
    disk and across a sweep of filter memory budgets from 0 up to the full
    max memory budget. The filter is one-sided (bloom-like): filter negatives
    skip the index entirely; positives are confirmed against it. Reports mean
    throughput, mean per-query latency and HNSW index accesses per dataset,
    with and without the filter, averaged over the memory-budget sweep.
    """
    num_vectors              = 1_000
    dim                      = 384
    percentage_duplicates    = 50
    count_memory_steps       = 10
    percentage_construct     = 10
    max_L                    = 256

    datasets = [name for name, path in dataset_filenames.items() if os.path.exists(path)]
    print(f"Datasets found on disk: {', '.join(datasets)}")
    runner = Runner(num_workers=MAX_WORKERS)

    for dataset in datasets:
        vectors = load_embeds(dataset, num_vectors)[0]
        n = len(vectors)  # a dataset may hold fewer than num_vectors
        bytes_per_element = vectors.itemsize
        num_construct = n * percentage_construct // 100

        construct_source = VectorSource(dataset, 0, num_construct)
        query_source     = VectorSource(dataset, num_construct, n)
        max_memory_budget = calculate_max_memory_budget_bytes(num_construct, dim, bytes_per_element)
        memory_budgets    = calculate_memory_budgets(max_memory_budget, count_memory_steps, include_zero=True)

        print(f"{dataset}: processing...")
        distance_threshold = 0.1

        # Volume of the true duplicate region (union of distance_threshold-balls
        # around every construct vector) is a pure-geometry quantity: independent
        # of the filter and memory budget, so it's computed once per dataset
        # here and reused across the whole sweep (see FPVR in hnsw_worker).
        vectors_construct = vectors[:num_construct]
        orig_vol_log_low, orig_vol_log_high = estimate_hypersphere_volume(
            vectors_construct,
            np.full(len(vectors_construct), distance_threshold),
            return_log=True,
        )

        for memory_budget in memory_budgets:
            for filter_cls in static_filters + dynamic_filters:
                runner.submit(
                    'hnsw',
                    filter_cls=filter_cls, lsh_cls=DummyLSH,
                    dim=dim, distance_threshold=distance_threshold,
                    max_L=max_L, memory_budget=memory_budget,
                    vectors_construct=construct_source, vectors_query=query_source,
                    percentage=percentage_duplicates, dataset=dataset,
                    orig_vol_log_low=orig_vol_log_low, orig_vol_log_high=orig_vol_log_high,
                )

    results = runner.run()
    print("HNSW evaluation completed.")

    df = pd.DataFrame(results).sort_values(['Dataset', 'Config', 'Percentage'])
    print("\n=== HNSW Filtered-Access Evaluation (per run) ===")
    print(df.to_string(index=False))

    mean_cols = [
        'Throughput Without Filter', 'Throughput With Filter',
        'Latency Without Filter (ms)', 'Latency With Filter (ms)',
        'Index Accesses', 'Total Queries', 'Speedup',
    ]
    summary = df.groupby(['Dataset', 'Config'])[mean_cols].mean().reset_index()
    print("\n=== Mean per dataset (averaged over the memory-budget sweep) ===")
    print(summary.to_string(index=False))

    plot_hnsw_results(results)
    return results


# Registry name -> canonical dataset key used by the ablation figure (must
# match plot_ablation.DATASETS exactly).
_ABLATION_DATASET_KEYS = {"ELI5": "eli5", "MMLU": "mmlu", "WildChat": "wildchat"}


def evaluate_ablation() -> pd.DataFrame:
    """
    Generate the per-stage construction-ladder data for GonzalezMEBFilter
    (initial cover -> ball refine -> boundary refine -> radius tighten ->
    feedback) across the 3 datasets the ablation figure needs, write it to
    results/ablation.csv, and render the figure panels via
    plot_ablation.generate_ablation_figures -- the same pattern
    static_evaluation()/hnsw_evaluation() use with plot_results/plot_hnsw_results,
    just without discarding the in-memory results in favor of a re-read.
    """
    num_vectors           = 1000
    dim                   = 384
    percentage_construct  = 20
    k_values              = [64, 128, 256, 512]
    tau_values            = [0.20, 0.35, 0.50]
    seeds                 = [0, 1, 2]
    feedback_fraction     = 0.10
    output_csv            = "results/ablation.csv"

    datasets = [d for d in _ABLATION_DATASET_KEYS if os.path.exists(dataset_filenames[d])]
    missing = [d for d in _ABLATION_DATASET_KEYS if d not in datasets]
    if missing:
        print(f"Skipping dataset(s) not found on disk: {', '.join(missing)}")
    if set(datasets) != set(_ABLATION_DATASET_KEYS):
        raise RuntimeError(
            "evaluate_ablation needs all 3 datasets "
            f"({', '.join(_ABLATION_DATASET_KEYS)}) on disk; only found: {', '.join(datasets) or 'none'}"
        )

    # k >= a dataset's construct-set size makes GonzalezMEBFilter fall back to
    # one zero-radius ball per point (log10_J = -inf, unplottable), so clamp
    # the sweep to what the *smallest* construct set can support -- keeps the
    # same k values present for every dataset, which the figure requires.
    construct_sizes = {d: len(load_embeds(d, num_vectors)[0]) * percentage_construct // 100 for d in datasets}
    min_construct = min(construct_sizes.values())
    usable_k_values = [k for k in k_values if k < min_construct]
    dropped_k = sorted(set(k_values) - set(usable_k_values))
    if dropped_k:
        print(
            f"Dropping k values >= smallest construct set ({min_construct} vectors, "
            f"from {min(construct_sizes, key=construct_sizes.get)}): {dropped_k}"
        )
    if not usable_k_values:
        raise RuntimeError(
            f"No k value in {k_values} is smaller than the smallest construct set "
            f"({min_construct} vectors). Raise num_vectors/percentage_construct or lower k_values."
        )
    k_values = usable_k_values

    # A single, sweep-independent construction tightness knob (see
    # ablation_worker's docstring for why tau can't double as this and stay
    # tau-independent for log10_J). The sweep's median is a reasonable default.
    construct_tau = float(np.median(tau_values))

    runner = Runner(num_workers=MAX_WORKERS)

    for dataset in datasets:
        vectors = load_embeds(dataset, num_vectors)[0]
        n = len(vectors)  # a dataset may hold fewer than num_vectors
        num_construct = n * percentage_construct // 100
        vectors_construct = vectors[:num_construct]
        vectors_query = vectors[num_construct:]

        construct_source = VectorSource(dataset, 0, num_construct)
        query_source = VectorSource(dataset, num_construct, n)

        print(f"{dataset}: computing ground truth for tau in {tau_values}...")
        actual_results_by_tau = {
            tau: compute_static_actual_results(vectors_construct, vectors_query, tau)
            for tau in tau_values
        }
        for k in k_values:
            for seed in seeds:
                runner.submit(
                    'ablation',
                    dim=dim, dataset=_ABLATION_DATASET_KEYS[dataset],
                    k=k, tau_values=tau_values, seed=seed, construct_tau=construct_tau,
                    vectors_construct=construct_source, vectors_query=query_source,
                    actual_results_by_tau=actual_results_by_tau, feedback_fraction=feedback_fraction,
                )

    job_results = runner.run()
    print("Ablation evaluation completed.")

    df = pd.DataFrame([row for job_rows in job_results for row in job_rows])

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Wrote {output_csv}")

    # plot_ablation's own K default may not survive the k-clamping above;
    # the largest surviving budget is the closest match to the original intent.
    generate_ablation_figures(df, k=max(k_values))
    return df


def evaluate_query_performance() -> pd.DataFrame:
    """
    Measure single-query vs. batched-query latency/throughput of a
    GonzalezMEBFilter cover as a function of filter size k, across the same
    3 datasets used by evaluate_ablation(). Mirrors evaluate_ablation()'s
    structure: one job per (dataset, k, seed), flattened into a long-format
    DataFrame, written to results/query_performance.csv, then rendered via
    plot_query_performance.generate_query_performance_figures.
    """
    num_vectors           = 100_000
    dim                   = 384
    percentage_construct  = 20
    k_values              = [16, 32, 64, 128, 256, 512, 1024]
    query_tau             = 0.35
    seeds                 = [0, 1, 2]
    single_query_limit    = 2000
    output_csv            = "results/query_performance.csv"

    datasets = [d for d in _ABLATION_DATASET_KEYS if os.path.exists(dataset_filenames[d])]
    missing = [d for d in _ABLATION_DATASET_KEYS if d not in datasets]
    if missing:
        print(f"Skipping dataset(s) not found on disk: {', '.join(missing)}")
    if set(datasets) != set(_ABLATION_DATASET_KEYS):
        raise RuntimeError(
            "evaluate_query_performance needs all 3 datasets "
            f"({', '.join(_ABLATION_DATASET_KEYS)}) on disk; only found: {', '.join(datasets) or 'none'}"
        )

    # k >= a dataset's construct-set size makes GonzalezMEBFilter fall back to
    # one zero-radius ball per point, so clamp the sweep the same way
    # evaluate_ablation() does (kept defensive even though the 100K-row
    # datasets comfortably support k=1024 at 20% construct).
    construct_sizes = {d: len(load_embeds(d, num_vectors)[0]) * percentage_construct // 100 for d in datasets}
    min_construct = min(construct_sizes.values())
    usable_k_values = [k for k in k_values if k < min_construct]
    dropped_k = sorted(set(k_values) - set(usable_k_values))
    if dropped_k:
        print(
            f"Dropping k values >= smallest construct set ({min_construct} vectors, "
            f"from {min(construct_sizes, key=construct_sizes.get)}): {dropped_k}"
        )
    if not usable_k_values:
        raise RuntimeError(
            f"No k value in {k_values} is smaller than the smallest construct set "
            f"({min_construct} vectors). Raise num_vectors/percentage_construct or lower k_values."
        )
    k_values = usable_k_values

    runner = Runner(num_workers=MAX_WORKERS)

    for dataset in datasets:
        vectors = load_embeds(dataset, num_vectors)[0]
        n = len(vectors)  # a dataset may hold fewer than num_vectors
        num_construct = n * percentage_construct // 100

        construct_source = VectorSource(dataset, 0, num_construct)
        query_source = VectorSource(dataset, num_construct, n)

        for k in k_values:
            for seed in seeds:
                runner.submit(
                    'query_performance',
                    dim=dim, dataset=_ABLATION_DATASET_KEYS[dataset],
                    k=k, seed=seed, construct_tau=query_tau, query_tau=query_tau,
                    vectors_construct=construct_source, vectors_query=query_source,
                    single_query_limit=single_query_limit,
                )

    job_results = runner.run()
    print("Query performance evaluation completed.")

    df = pd.DataFrame([row for job_rows in job_results for row in job_rows])

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Wrote {output_csv}")

    generate_query_performance_figures(df)
    return df


def dynamic_evaluation() -> list:
    num_vectors              = 100_000
    dim                      = 384
    min_percentage_duplicates = 1
    max_percentage_duplicates = 9
    count_steps              = 10
    count_memory_steps       = 10
    percentage_construct     = 25
    max_L                    = 256

    dataset       = DATASET
    vectors       = load_embeds(dataset, num_vectors)[0]
    bytes_per_element = vectors.itemsize
    num_construct = num_vectors * percentage_construct // 100

    construct_source = VectorSource(dataset, 0, num_construct)
    query_source     = VectorSource(dataset, num_construct, num_vectors)
    max_memory_budget = calculate_max_memory_budget_bytes(num_construct, dim, bytes_per_element)
    memory_budgets    = calculate_memory_budgets(max_memory_budget, count_memory_steps)
    runner = Runner(num_workers=MAX_WORKERS)

    for percentage in np.linspace(min_percentage_duplicates, max_percentage_duplicates, count_steps):
        distance_threshold = calculate_distance_threshold(vectors, percentage)
        actual_results = compute_actual_results(vectors, distance_threshold)[num_construct:]
        for memory_budget in memory_budgets:
            for filter_cls in dynamic_filters:
                runner.submit(
                    'dynamic',
                    filter_cls=filter_cls, lsh_cls=DummyLSH,
                    dim=dim, distance_threshold=distance_threshold,
                    max_L=max_L, memory_budget=memory_budget,
                    vectors_construct=construct_source, vectors_query=query_source,
                    actual_results=actual_results, percentage=percentage,
                )

    results = runner.run()
    print("Dynamic evaluation completed.")
    plot_results(results, "dynamic")
    return results

def tradeoff_evaluation() -> Tuple[List, List]:
    num_vectors = 100_000
    dim = 384
    max_L = 256
    percentage_construct = 10
    target_percentile = 0.2
    dataset = DATASET
    random_test = False

    # load dataset arrays for ground-truth processing
    vectors = load_embeds(dataset, num_vectors)[0]
    num_construct = num_vectors * percentage_construct // 100
    vectors_construct = vectors[:num_construct]
    vectors_query = vectors[num_construct:]

    # create construct and query vector objects for workers
    construct_source = VectorSource(dataset, 0, num_construct)
    query_source = VectorSource(dataset, num_construct, num_vectors)

    # calculate constant reference distance threshold
    print(f"Pre-calculating evaluation distance baselines at {target_percentile}%...")
    distance_threshold = calculate_distance_threshold(vectors, target_percentile)

    # means we run test with randomly generated queires for testing generalization
    if random_test:
        print("\nPreparing Generalization Test (Random Vectors)...")
        # generate random samples from a uniform distribution and normalize them for dataset consistency
        random_queries = np.random.uniform(low=np.min(vectors), high=np.max(vectors), size=(len(vectors_query), dim)).astype(np.float32)
        random_queries_norms = np.linalg.norm(random_queries, axis=1, keepdims=True)
        random_queries = random_queries / np.where(random_queries_norms == 0, 1e-10, random_queries_norms)

        print("Calculating actual results for random queries...")
        actual_results = compute_static_actual_results(vectors_construct, random_queries, distance_threshold)
        query_source = ArrayVectorSource("random_queries", array=random_queries)

    # else we run the original dataset queries provided
    else:
        # compute static filter resuls
        print("Calculating actual results for test queries...")
        actual_results = compute_static_actual_results(vectors_construct, vectors_query, distance_threshold)

    runner = Runner(num_workers=MAX_WORKERS)

    # EXPERIMENT 1: Evaluate Geometric Filter Performance
    # Tests how the number of spheres (spatial partitioning) impacts 
    # search accuracy and speed for ball-based filters.
    sphere_sizes = [5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240]
    sphere_sizes = [s for s in sphere_sizes if s < len(vectors_construct)] + [len(vectors_construct)]
    sphere_filters = [KDBallFilter, KMEBFilter]

    for size in sphere_sizes:
        for filter_cls in sphere_filters:
            runner.submit(
                'tradeoff',
                filter_cls=filter_cls,
                lsh_cls=FaissSimHashLSH,
                dim=dim,
                distance_threshold=distance_threshold,
                max_L=max_L,
                vectors_construct=construct_source,
                vectors_query=query_source,
                actual_results=actual_results,
                max_spheres=size,
                memory_budget=None
            )

    print("\nExecuting sphere evaluation...")
    raw_sphere_results = runner.run()
    sphere_results = [r for r in raw_sphere_results if r is not None]
    runner.clear() # clear runner jobs
    print("Sphere evaluation completed.")

    # EXPERIMENT 2: Benchmark Resource Efficiency
    # Compares all available filter strategies under strict memory limits 
    # to identify which methods perform best when storage is constrained.
    memory_budgets = [500*1024, 1*1024*1024, 2*1024*1024, 4*1024*1024, 8*1024*1024, 16*1024*1024, 32*1024*1024, 64*1024*1024]

    for budget in memory_budgets:
        for filter_cls in static_filters + dynamic_filters:
            runner.submit(
                'tradeoff',
                filter_cls=filter_cls,
                lsh_cls=FaissSimHashLSH,
                dim=dim,
                distance_threshold=distance_threshold,
                max_L=max_L,
                vectors_construct=construct_source,
                vectors_query=query_source,
                actual_results=actual_results,
                max_spheres=None,
                memory_budget=budget
            )

    print("\nExecuting memory evaluation...")
    raw_memory_results = runner.run()
    memory_results = [r for r in raw_memory_results if r is not None]
    print("Memory evaluation completed.")
    plot_tradeoff_results(sphere_results, memory_results)

    return sphere_results, memory_results

if __name__ == "__main__":
    # static_evaluation()
    # dynamic_evaluation()
    equal_memory_evaluation()
    # hnsw_evaluation()
    # tradeoff_evaluation()
    # evaluate_ablation()
    # evaluate_query_performance()
