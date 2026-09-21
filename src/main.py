import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

# Must happen before numpy/faiss are imported: otherwise each of Runner's
# MAX_WORKERS processes also grabs every CPU core for its own internal
# threads, and process x thread parallelism multiply into severe
# oversubscription (observed: whole-system unresponsiveness).
for _env_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env_var, "1")

import faiss
import time
import inspect
import numpy as np
import pandas as pd

faiss.omp_set_num_threads(1)  # belt-and-suspenders alongside the env vars above

from logging_config import configure_logging
from runner import Runner, QUANTIZATION_SWEEP, system_tau_ground_truth_worker
from plots.plot_quantization import generate_quantization_figures
from plots.plot_query_performance import generate_query_performance_figures
from plots.plot_equal_memory import generate_equal_memory_figures
from plots.plot_scaling import generate_scaling_figures
from plots.plot_streaming import generate_streaming_figures
from plots.plot_system_tau import generate_system_tau_figures
from plots.plot_system import generate_system_figures
from plots.plot_common import DATASETS, DATASET_LABELS
from typing import Any, List, Tuple
from filters.ball_cover.ball_cover_filter import BallCoverFilter
from filters.dsh_filter import DSHFilter
from filters.hnsw_filter import HNSWFilter
from filters.ivf_filter import IVFFilter
from filters.quantized_vector_filter import QuantizedVectorFilter
from util.dataset_paths import dataset_filenames
from util.fetch_datasets import load_embeds
from vector_source import VectorSource

configure_logging()
logger = logging.getLogger(__name__)

MAX_WORKERS = 8
DATASET = 'ELI5'

# Vectors/dataset for the paper's Workloads protocol (Section 5.1): "100K
# queries per dataset, 80% construction / 20% queries". One knob for every
# evaluation: the ground-truth FPR/FNR ones (system_tau, quantization,
# equal_memory, streaming) and the throughput-only sweeps (scaling,
# query_performance), which used to read a separate STREAM_SIZE.
WORKLOAD_NUM_VECTORS = 10_000

# Registry name -> canonical dataset key, one source of truth via plot_common.
DATASET_KEYS = {DATASET_LABELS[d]: d for d in DATASETS}

# Output CSV path per evaluation, shared by each generate_*'s
# runner.submit(output_csv=...), each save_*'s df, and plot_all()'s reload.
EQUAL_MEMORY_CSV = "results/equal_memory.csv"
SYSTEM_TAU_CSV = "results/system_tau.csv"
SYSTEM_CSV = "results/system.csv"
QUANTIZATION_CSV = "results/quantization.csv"
SCALING_QUERY_CSV = "results/scaling_query.csv"
SCALING_INSERT_CSV = "results/scaling_insert.csv"
SCALING_CONSTRUCTION_CSV = "results/scaling_construction.csv"
STREAMING_CSV = "results/streaming.csv"

# generate_equal_memory()'s own comparison set: excludes DSBFFilter/
# CoveringLSHFallbackFilter/LSBFFilter, which degenerate to ~100% duplicate
# predictions at realistic construct-set sizes (still used elsewhere).
equal_memory_filters = [BallCoverFilter, HNSWFilter, IVFFilter, DSHFilter, QuantizedVectorFilter]

def calculate_nn_distance_threshold(
    construct_vectors: np.ndarray,
    percentage: float,
    sample_size: int = 5_000,
    seed: int = 42,
) -> float:
    """The `percentage`-th percentile of nearest-*other*-neighbor distance
    within a sample of construct_vectors -- ties the threshold to the actual
    test compute_static_actual_results performs, unlike a random-pairwise
    percentile: at large construct-set sizes a point's nearest neighbor is
    much closer than a random pair, so a random-pairwise percentile silently
    pushes the duplicate rate toward 100%, collapsing FPR/FNR to degenerate
    values."""
    n = len(construct_vectors)
    if n < 2:
        return 0.0

    vecs = np.ascontiguousarray(construct_vectors, dtype=np.float32)
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(n, size=sample_size, replace=False) if sample_size < n else np.arange(n)

    index = faiss.IndexFlatL2(vecs.shape[1])
    index.add(vecs)
    # k=2: rank 0 is the point itself (distance 0); rank 1 is the *other* nearest neighbor.
    sq_dist, _ = index.search(vecs[sample_idx], 2)
    nn_dist = np.sqrt(np.clip(sq_dist[:, 1], 0, None))
    return float(np.percentile(nn_dist, percentage))


def advancing_fold_starts(pool_len: int, k_folds: int, fold_stride: int, window_len: int) -> List[int]:
    """Non-wrapping K-fold-like sweep: fold f's window starts f*fold_stride
    into the pool and is window_len vectors wide, never wrapping back to the
    start. Raises if the pool isn't long enough for all k_folds windows,
    rather than silently reusing or wrapping a shorter one. Returns the
    fold-start offsets (0, fold_stride, 2*fold_stride, ...)."""
    max_start = (k_folds - 1) * fold_stride
    required = max_start + window_len
    if pool_len < required:
        raise ValueError(
            f"advancing_fold_starts: pool of {pool_len} vectors is too short for a "
            f"non-wrapping {k_folds}-fold sweep (window_len={window_len}, "
            f"fold_stride={fold_stride}); needs >= {required} = "
            "(k_folds-1)*fold_stride + window_len. Load/point at a longer stream, "
            "or lower window_len/k_folds/fold_stride."
        )
    return [f * fold_stride for f in range(k_folds)]


def rows_for_mode(job_results: List[Tuple[str, Any]], mode: str) -> list:
    """Pulls one Runner mode's rows out of a shared Runner.run()'s combined
    (mode, result) list -- result is either one dict per job or a list of
    dicts, flattened uniformly here. Used by save_* functions to recover
    just their own evaluation's rows from a Runner several share."""
    rows = []
    for job_mode, result in job_results:
        if job_mode != mode:
            continue
        rows.extend(result if isinstance(result, list) else [result])
    return rows


def compute_static_actual_results(
    construct_vectors: np.ndarray,
    query_vectors: np.ndarray,
    distance_threshold: float,
) -> list[bool]:
    """Static ground-truth: a query vector is a duplicate iff within
    `distance_threshold` of any vector in the fixed construct set. Uses one
    batched faiss nearest-neighbor call rather than a per-query Python loop;
    faiss returns squared L2, hence the distance_threshold**2 comparison."""
    construct = np.ascontiguousarray(construct_vectors, dtype=np.float32)
    queries = np.ascontiguousarray(query_vectors, dtype=np.float32)

    if len(construct) == 0:
        return [False] * len(queries)

    index = faiss.IndexFlatL2(construct.shape[1])
    index.add(construct)
    nn_sq_dist, _ = index.search(queries, 1)
    return (nn_sq_dist[:, 0] <= distance_threshold ** 2).tolist()


def generate_equal_memory(runner: Runner) -> None:
    """Equal-memory comparison across equal_memory_filters. Sweeps
    count_memory_steps budgets scaled off the construct set itself (one
    sphere per point, see bytes_per_sphere_at_1to1 below). For each budget,
    every filter's best config is chosen on a
    held-out calibration set (see runner.py::_build_calibrated_filter), then
    FPR/FNR/latency/throughput are measured on a separate held-out query
    set. Uses the paper's default single-dataset representative, MMLU
    (Section 5.6 names no specific dataset).

    Only submits 'equal_memory' jobs (output_csv=EQUAL_MEMORY_CSV) to the
    given `runner` -- does not run it. The Runner writes
    results/equal_memory.csv incrementally as each job completes."""
    num_vectors               = WORKLOAD_NUM_VECTORS
    dim                       = 384
    percentage_construct      = 80
    percentage_calibration    = 10
    duplicate_percentage      = 5.0
    count_memory_steps        = 10

    dataset = "MMLU"  # paper's default single-dataset representative
    vectors = load_embeds(dataset, num_vectors)[0]
    num_construct   = num_vectors * percentage_construct // 100
    num_calibration = num_vectors * percentage_calibration // 100
    vectors_construct   = vectors[:num_construct]
    vectors_calibration = vectors[num_construct:num_construct + num_calibration]
    vectors_query       = vectors[num_construct + num_calibration:]

    construct_source   = VectorSource(dataset, 0, num_construct)
    calibration_source = VectorSource(dataset, num_construct, num_construct + num_calibration)
    query_source        = VectorSource(dataset, num_construct + num_calibration, num_vectors)

    logger.info("Calculating distance threshold...")
    # A nearest-neighbor threshold, not a random-pairwise percentile: at this
    # construct-set size, a random-pairwise percentile is far more
    # permissive than the nearest-neighbor test actually used, pushing the
    # duplicate rate to ~100% and collapsing FPR/FNR to 0/0.
    distance_threshold = calculate_nn_distance_threshold(vectors_construct, duplicate_percentage)
    logger.info("Calculating calibration/query ground truth...")
    actual_calibration = compute_static_actual_results(vectors_construct, vectors_calibration, distance_threshold)
    actual_query       = compute_static_actual_results(vectors_construct, vectors_query, distance_threshold)

    # Scaled off the construct set itself: one sphere per point.
    bytes_per_sphere_at_1to1 = (dim + 1) * 4 * 4.0
    max_memory_budget = int(num_construct * bytes_per_sphere_at_1to1)
    memory_budgets = np.linspace(0, max_memory_budget, count_memory_steps + 1, dtype=int)[1:]

    for memory_budget in memory_budgets:
        logger.info(f"Submitting jobs for memory_budget={memory_budget} bytes...")
        for filter_cls in equal_memory_filters:
            runner.submit(
                'equal_memory',
                output_csv=EQUAL_MEMORY_CSV,
                filter_cls=filter_cls,
                dim=dim, distance_threshold=distance_threshold,
                memory_budget=memory_budget,
                vectors_construct=construct_source, vectors_calibration=calibration_source,
                vectors_query=query_source,
                actual_calibration=actual_calibration, actual_query=actual_query,
                dataset=dataset,
            )


def save_equal_memory(job_results: List[Tuple[str, Any]]) -> list:
    """Extracts this evaluation's rows from a shared Runner's combined
    run() output (see generate_equal_memory()). The Runner itself already
    wrote results/equal_memory.csv incrementally, one row-batch per
    completed job (see generate_equal_memory()'s output_csv=), so this no
    longer writes the file -- it only recovers this evaluation's rows for
    plot_equal_memory()."""
    return rows_for_mode(job_results, 'equal_memory')


def plot_equal_memory(results: list) -> None:
    generate_equal_memory_figures(df=pd.DataFrame(results))


def generate_system_tau(runner: Runner) -> None:
    """End-to-end filtering benefit (paper Section 5.2, Figure 2): one
    BallCoverFilter cover per (dataset, fold), queried at every tau in a
    fixed sweep without ever rebuilding. Ball budget k is a fixed fraction
    (k_fraction, 5%) of each fold's construct-set size rather than the
    paper's fixed k=400.

    Implements the paper's Workloads protocol (Section 5.1): 5 datasets x
    k_folds=5 non-wrapping folds via advancing_fold_starts (80%
    construct/20% query per fold); each dataset needs enough vectors on
    disk for that or it raises. plot_system_tau reports the mean +/- std
    over the 5 folds, matching the paper's "5 runs" error bars.

    Ground truth (nearest-construct-set distance per query) doesn't depend
    on k, so it's computed once per (dataset, fold) in parallel and cached
    to results/system_tau_ground_truth/, not redone per (dataset, fold, k).

    Only submits 'system_tau' jobs (output_csv=SYSTEM_TAU_CSV) -- does not
    run `runner`. The ground-truth precomputation above still runs eagerly,
    in its own dedicated ProcessPoolExecutor."""
    num_vectors      = WORKLOAD_NUM_VECTORS
    dim              = 384
    k_folds          = 5
    k_fraction       = 0.1
    tau_values       = [float(t) for t in np.logspace(-2, -0.5, num=10)]
    ground_truth_dir = "results/system_tau_ground_truth"

    datasets = [d for d in DATASET_KEYS if os.path.exists(dataset_filenames[d])]
    missing = [d for d in DATASET_KEYS if d not in datasets]
    if missing:
        logger.warning(f"Skipping dataset(s) not found on disk: {', '.join(missing)}")
    if set(datasets) != set(DATASET_KEYS):
        raise RuntimeError(
            f"generate_system_tau needs all {len(DATASET_KEYS)} datasets "
            f"({', '.join(DATASET_KEYS)}) on disk; only found: {', '.join(datasets) or 'none'}"
        )

    logger.info(f"Ball budget fixed as a fraction of the construct set: {k_fraction}")
    logger.info(f"Fixed tau sweep (shared across datasets): {tau_values}")
    logger.info(f"K-fold cross-validation: {k_folds} folds, non-wrapping contiguous 80/20 spans")

    query_len = num_vectors // k_folds
    construct_len = num_vectors - query_len

    # (dataset, fold, construct_source, query_source), reused by both passes
    # below -- lazy VectorSource pointers, no array materialized here.
    splits = []
    for dataset in datasets:
        total_rows = len(VectorSource(dataset))
        fold_starts = advancing_fold_starts(total_rows, k_folds, query_len, num_vectors)
        for fold, fold_start in enumerate(fold_starts):
            query_start, query_end = fold_start, fold_start + query_len
            construct_start, construct_end = query_end, query_end + construct_len
            construct_source = VectorSource(dataset, construct_start, construct_end)
            query_source = VectorSource(dataset, query_start, query_end)
            splits.append((dataset, fold, construct_source, query_source))

    # Ground truth (nearest-construct-set distance per query vector): load
    # whatever's already cached on disk, compute the rest in parallel.
    os.makedirs(ground_truth_dir, exist_ok=True)
    nn_dist_by_split = {}
    to_compute = []
    for dataset, fold, construct_source, query_source in splits:
        cache_path = os.path.join(ground_truth_dir, f"{dataset}_{num_vectors}_{k_folds}fold_{fold}.npy")
        if os.path.exists(cache_path):
            nn_dist_by_split[(dataset, fold)] = np.load(cache_path)
        else:
            to_compute.append((dataset, fold, cache_path, construct_source, query_source))

    if to_compute:
        logger.info(f"Computing ground truth for {len(to_compute)} (dataset, fold) pair(s) not yet cached...")
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(system_tau_ground_truth_worker, dim, cs, qs): (dataset, fold, cache_path)
                for dataset, fold, cache_path, cs, qs in to_compute
            }
            for future in as_completed(futures):
                dataset, fold, cache_path = futures[future]
                nn_dist = future.result()
                np.save(cache_path, nn_dist)
                nn_dist_by_split[(dataset, fold)] = nn_dist
    else:
        logger.info("All ground truth already cached on disk; skipping recomputation.")

    for dataset, fold, construct_source, query_source in splits:
        nn_dist = nn_dist_by_split[(dataset, fold)]
        actual_results_by_tau = {t: (nn_dist <= t).tolist() for t in tau_values}

        k = min(max(1, round(k_fraction * construct_len)), construct_len - 1)

        runner.submit(
            'system_tau',
            output_csv=SYSTEM_TAU_CSV,
            dim=dim, dataset=DATASET_KEYS[dataset],
            k=k, tau_values=tau_values, seed=fold,
            vectors_construct=construct_source,
            vectors_query=query_source,
            actual_results_by_tau=actual_results_by_tau,
            )


def save_system_tau(job_results: List[Tuple[str, Any]]) -> pd.DataFrame:
    """Extracts this evaluation's rows from a shared Runner's combined
    run() output (see generate_system_tau()). The Runner itself already
    wrote results/system_tau.csv incrementally (see generate_system_tau()'s
    output_csv=), so this only rebuilds the DataFrame for plot_system_tau()."""
    return pd.DataFrame(rows_for_mode(job_results, 'system_tau'))


def plot_system_tau(df: pd.DataFrame) -> None:
    generate_system_tau_figures(df)


def generate_system(runner: Runner) -> None:
    """Same end-to-end filtering benefit as generate_system_tau -- one
    BallCoverFilter cover per (dataset, fold), queried at every tau without
    rebuilding -- but tau is *solved for* a target redundancy percentage
    instead of swept over a fixed, dataset-independent grid.

    For each (dataset, fold), tau at target p is the p-th percentile of
    that split's query->nearest-construct-vector distances, so exactly p%
    of the query vectors sit within tau of some construct vector. Percentile
    is the inverse-ECDF of that exact population, so the measured
    redundancy rate lands on the target within a rounding hair -- unless the
    distribution has a mass spike (exact-duplicate embeddings) wide enough
    to straddle several targets, in which case they collapse onto the same
    tau and share a measured rate.

    That makes redundancy_pct -- not tau -- the shared x-axis across
    datasets (see plot_system), which is the point: absolute L2 distances
    aren't comparable between embedding sets, but "5% of queries are
    redundant" is.

    Reuses generate_system_tau's splits and its cached per-(dataset, fold)
    nearest-neighbor distances (same ground_truth_dir, same cache keys), so
    running both costs that brute-force pass only once.

    Only submits 'system' jobs (output_csv=SYSTEM_CSV) -- does not run
    `runner`. The ground-truth precomputation still runs eagerly, in its own
    dedicated ProcessPoolExecutor."""
    num_vectors      = WORKLOAD_NUM_VECTORS
    dim              = 384
    k_folds          = 5
    k_fraction       = 0.1
    min_redundancy_pct = 5.0
    max_redundancy_pct = 20.0
    count_steps        = 8
    ground_truth_dir = "results/system_tau_ground_truth"

    redundancy_pcts = [float(p) for p in np.linspace(min_redundancy_pct, max_redundancy_pct, count_steps)]

    datasets = [d for d in DATASET_KEYS if os.path.exists(dataset_filenames[d])]
    missing = [d for d in DATASET_KEYS if d not in datasets]
    if missing:
        logger.warning(f"Skipping dataset(s) not found on disk: {', '.join(missing)}")
    if set(datasets) != set(DATASET_KEYS):
        raise RuntimeError(
            f"generate_system needs all {len(DATASET_KEYS)} datasets "
            f"({', '.join(DATASET_KEYS)}) on disk; only found: {', '.join(datasets) or 'none'}"
        )

    logger.info(f"Ball budget fixed as a fraction of the construct set: {k_fraction}")
    logger.info(f"Target redundancy percentages (shared across datasets): {redundancy_pcts}")
    logger.info(f"K-fold cross-validation: {k_folds} folds, non-wrapping contiguous 80/20 spans")

    query_len = num_vectors // k_folds
    construct_len = num_vectors - query_len

    splits = []
    for dataset in datasets:
        total_rows = len(VectorSource(dataset))
        fold_starts = advancing_fold_starts(total_rows, k_folds, query_len, num_vectors)
        for fold, fold_start in enumerate(fold_starts):
            query_start, query_end = fold_start, fold_start + query_len
            construct_start, construct_end = query_end, query_end + construct_len
            construct_source = VectorSource(dataset, construct_start, construct_end)
            query_source = VectorSource(dataset, query_start, query_end)
            splits.append((dataset, fold, construct_source, query_source))

    os.makedirs(ground_truth_dir, exist_ok=True)
    nn_dist_by_split = {}
    to_compute = []
    for dataset, fold, construct_source, query_source in splits:
        cache_path = os.path.join(ground_truth_dir, f"{dataset}_{num_vectors}_{k_folds}fold_{fold}.npy")
        if os.path.exists(cache_path):
            nn_dist_by_split[(dataset, fold)] = np.load(cache_path)
        else:
            to_compute.append((dataset, fold, cache_path, construct_source, query_source))

    if to_compute:
        logger.info(f"Computing ground truth for {len(to_compute)} (dataset, fold) pair(s) not yet cached...")
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(system_tau_ground_truth_worker, dim, cs, qs): (dataset, fold, cache_path)
                for dataset, fold, cache_path, cs, qs in to_compute
            }
            for future in as_completed(futures):
                dataset, fold, cache_path = futures[future]
                nn_dist = future.result()
                np.save(cache_path, nn_dist)
                nn_dist_by_split[(dataset, fold)] = nn_dist
    else:
        logger.info("All ground truth already cached on disk; skipping recomputation.")

    for dataset, fold, construct_source, query_source in splits:
        nn_dist = nn_dist_by_split[(dataset, fold)]

        # tau solved per (dataset, fold) so that redundancy_pcts[i]% of this
        # split's queries are within tau_values[i] of a construct vector.
        tau_values = [float(np.percentile(nn_dist, p)) for p in redundancy_pcts]
        logger.info(
            f"{dataset} fold {fold}: tau per target % = "
            + ", ".join(f"{p:.1f}%->{t:.4f}" for p, t in zip(redundancy_pcts, tau_values))
        )
        # Keyed by tau, so colliding targets share one entry -- the worker
        # still emits one row per (tau, target) pair.
        actual_results_by_tau = {t: (nn_dist <= t).tolist() for t in tau_values}

        k = min(max(1, round(k_fraction * construct_len)), construct_len - 1)

        runner.submit(
            'system',
            output_csv=SYSTEM_CSV,
            dim=dim, dataset=DATASET_KEYS[dataset],
            k=k, tau_values=tau_values, redundancy_pcts=redundancy_pcts,
            seed=fold,
            vectors_construct=construct_source,
            vectors_query=query_source,
            actual_results_by_tau=actual_results_by_tau,
            )


def save_system(job_results: List[Tuple[str, Any]]) -> pd.DataFrame:
    """Extracts this evaluation's rows from a shared Runner's combined
    run() output (see generate_system()). The Runner itself already wrote
    results/system.csv incrementally, so this only rebuilds the DataFrame
    for plot_system()."""
    return pd.DataFrame(rows_for_mode(job_results, 'system'))


def plot_system(df: pd.DataFrame) -> None:
    generate_system_figures(df)


def generate_quantization(runner: Runner) -> None:
    """Generates per-quantize-variant construction data for BallCoverFilter
    (QUANTIZATION_SWEEP) on a single dataset. Every candidate shares one
    memory budget per k (derived from the unquantized cover's byte-per-ball
    size), so a more compact quantizer packs more balls into the same
    budget, letting plot_quantization.py use recorded memory usage as the
    x-axis. k_values/tau_values derive from num_vectors/percentage_construct
    rather than hardcoded absolutes, so those two knobs rescale the whole
    sweep. Dataset and duplicate percentage follow the paper's Figure 3
    caption ("MS MARCO, tau such that 5% are duplicates").

    K-fold cross-validation (Section 5.1): 5 non-wrapping folds via
    advancing_fold_starts, tagged via seed=fold for plot_quantization's
    seed-based mean +/- 95% CI aggregation. tau is a single value, calibrated
    once from fold 0's construct set so every fold lands on the same
    threshold; each fold's ground truth is still recomputed from its own
    construct/query split. This sweep varies the quantizer, not the
    threshold -- the tau sweep lives in generate_system_tau().

    Only submits 'quantization' jobs (output_csv=QUANTIZATION_CSV) -- does
    not run `runner`."""
    num_vectors            = WORKLOAD_NUM_VECTORS
    dim                    = 384
    k_folds                = 5
    k_fractions            = [0.01, 0.02, 0.04, 0.08, 0.16]
    duplicate_percentage   = 5

    dataset = "MsMarco"  # Figure 3: "We evaluate on MS MARCO"

    query_len = num_vectors // k_folds
    construct_len = num_vectors - query_len
    total_rows = len(VectorSource(dataset))
    fold_starts = advancing_fold_starts(total_rows, k_folds, query_len, num_vectors)

    k_values = sorted({max(1, round(construct_len * f)) for f in k_fractions})
    k_values = [k for k in k_values if k < construct_len]
    if not k_values:
        raise RuntimeError(
            f"No k_fraction in {k_fractions} produced a k below the construct set "
            f"({construct_len} vectors). Raise num_vectors or lower k_fractions."
        )
    logger.info(f"Scaling k_values to {k_values} ({construct_len} vectors in the construct set)")

    # Calibrated from fold 0's construct span only, loaded as a real array
    # just for this; job submissions below use lazy VectorSource instead.
    fold0_construct_start = fold_starts[0] + query_len
    fold0_construct = VectorSource(dataset, fold0_construct_start, fold0_construct_start + construct_len).to_array()
    tau = calculate_nn_distance_threshold(fold0_construct, duplicate_percentage)
    logger.info(f"Scaling tau to {tau} ({duplicate_percentage}-th percentile NN distance, fold 0)")

    for fold, fold_start in enumerate(fold_starts):
        query_start, query_end = fold_start, fold_start + query_len
        construct_start, construct_end = query_end, query_end + construct_len

        logger.info(f"Computing ground truth for fold {fold} (tau={tau})...")
        # Ground truth runs in this driver process (not a worker), so it
        # needs real arrays, loaded once per fold here.
        vectors_construct = VectorSource(dataset, construct_start, construct_end).to_array()
        vectors_query = VectorSource(dataset, query_start, query_end).to_array()
        actual_results = compute_static_actual_results(vectors_construct, vectors_query, tau)
        construct_source = VectorSource(dataset, construct_start, construct_end)
        query_source = VectorSource(dataset, query_start, query_end)
        for k in k_values:
            for quantize in QUANTIZATION_SWEEP:
                runner.submit(
                    'quantization',
                    output_csv=QUANTIZATION_CSV,
                    dim=dim, dataset=dataset,
                    k=k, quantize=quantize, tau=tau, seed=fold,
                    vectors_construct=construct_source, vectors_query=query_source,
                    actual_results=actual_results,
                )


def save_quantization(job_results: List[Tuple[str, Any]]) -> pd.DataFrame:
    """Extracts this evaluation's rows from a shared Runner's combined
    run() output (see generate_quantization()). The Runner itself already
    wrote results/quantization.csv incrementally (see generate_quantization()'s
    output_csv=), so this only rebuilds the DataFrame for plot_quantization()."""
    return pd.DataFrame(rows_for_mode(job_results, 'quantization'))


def plot_quantization(df: pd.DataFrame) -> None:
    generate_quantization_figures(df)


def generate_query_performance() -> pd.DataFrame:
    """Measures single-query vs. batched-query latency/throughput of a
    BallCoverFilter cover as a function of filter size k, across every
    dataset in DATASET_KEYS. One job per (dataset, k, seed), written to
    results/query_performance.csv."""
    num_vectors           = WORKLOAD_NUM_VECTORS
    dim                   = 384
    percentage_construct  = 20
    k_values              = [16, 32, 64, 128, 256, 512, 1024]
    query_tau             = 0.35
    seeds                 = [0, 1, 2]
    single_query_limit    = 2000
    output_csv            = "results/query_performance.csv"

    datasets = [d for d in DATASET_KEYS if os.path.exists(dataset_filenames[d])]
    missing = [d for d in DATASET_KEYS if d not in datasets]
    if missing:
        logger.warning(f"Skipping dataset(s) not found on disk: {', '.join(missing)}")
    if set(datasets) != set(DATASET_KEYS):
        raise RuntimeError(
            f"generate_query_performance needs all {len(DATASET_KEYS)} datasets "
            f"({', '.join(DATASET_KEYS)}) on disk; only found: {', '.join(datasets) or 'none'}"
        )

    # k >= a dataset's construct-set size makes BallCoverFilter fall back to
    # one zero-radius ball per point, so clamp the sweep like generate_quantization() does.
    construct_sizes = {d: len(load_embeds(d, num_vectors)[0]) * percentage_construct // 100 for d in datasets}
    min_construct = min(construct_sizes.values())
    usable_k_values = [k for k in k_values if k < min_construct]
    dropped_k = sorted(set(k_values) - set(usable_k_values))
    if dropped_k:
        logger.warning(
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
                    dim=dim, dataset=DATASET_KEYS[dataset],
                    k=k, seed=seed, query_tau=query_tau,
                    vectors_construct=construct_source, vectors_query=query_source,
                    single_query_limit=single_query_limit,
                )

    job_results = runner.run()
    logger.info("Query performance evaluation completed.")

    df = pd.DataFrame([row for job_rows in job_results for row in job_rows])

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info(f"Wrote {output_csv}")

    return df


def plot_query_performance(df: pd.DataFrame) -> None:
    generate_query_performance_figures(df)


# These panels compare across batch_size/stream_size, not datasets, so one
# representative dataset suffices (Figure 5: "no measurable effect"
# per-dataset). MsMarco, matching generate_quantization: the 5 non-wrapping
# folds below need ~1.8x the per-fold pool (180K rows at the Workloads
# protocol's sizes), which rules out the smaller registry entries -- MMLU
# holds 115,700, NaturalQuestions 91,535.
_SCALING_DATASET = "MsMarco"


def generate_scaling(runner: Runner) -> None:
    """Measures three BallCoverFilter throughput surfaces against filter
    size k: query (k x batch_size grid), insertion (same grid, timing add()
    instead), and construction (k x n, where indexed-set size itself
    varies). Runs on a single dataset (_SCALING_DATASET); stream_sizes (the
    6 construction lines) derive from `stream_size` alone.

    Sized to the paper's Workloads protocol (Section 5.1): 100K vectors per
    fold, 80% construct / 20% query, so stream_size is the 80K construct set
    and n_query_vectors the 20K holdout it is queried with. Both sweeps stay
    at 6 values because plot_scaling draws one line per value and defines 6
    line colors.

    K-fold cross-validation (Section 5.1): fold f's (construct, holdout)
    window starts f*fold_stride further into the dataset than fold 0's,
    non-wrapping via advancing_fold_starts, so each repeat measures
    throughput against genuinely different data. With fold_stride equal to
    the holdout width, the 5 query holdouts are disjoint -- which is what
    makes them 5 samples rather than 5 views of one. Tagged via seed=fold
    for plot_scaling's seed-based mean +/- 95% CI aggregation.

    Only submits 'scaling_query'/'scaling_insert'/'scaling_construction'
    jobs -- does not run `runner`. Each targets its own output_csv, so the
    Runner writes all three results/scaling_*.csv files incrementally."""
    # 80% of the protocol's 100K per fold. Not WORKLOAD_NUM_VECTORS itself:
    # that is the whole 100K, and using it here would make the construct set
    # the full workload and push the pool (construct + holdout) past it.
    stream_size      = WORKLOAD_NUM_VECTORS * 70 // 100
    dim              = 384
    n_steps          = 6
    k_percentages    = [1, 2, 4, 8, 16, 32]  # ball budget k, as a percentage of stream_size
    # Powers of 4 from a single query up to 1024: spans three orders of
    # magnitude in 6 lines, and the largest still cuts n_query_vectors into
    # ~20 batches, so no line is timed from a handful of calls.
    batch_sizes      = [1, 4, 16, 64, 256, 1024]
    # Throughput is threshold-independent here (query() scores every ball
    # before comparing), so this only has to be a valid tau, not a tuned one.
    tau_mid          = 0.35
    k_folds          = 5
    # The protocol's 20% query split. n_insert_vectors stays smaller: insert()
    # mutates the cover, so its cost per vector grows with the ball count and
    # a 20K stream would dominate the sweep's runtime for no extra resolution.
    n_query_vectors  = WORKLOAD_NUM_VECTORS * 20 // 100
    n_insert_vectors = WORKLOAD_NUM_VECTORS * 10 // 100

    if _SCALING_DATASET not in DATASET_KEYS:
        raise RuntimeError(f"_SCALING_DATASET={_SCALING_DATASET!r} must be one of {sorted(DATASET_KEYS)}")
    dataset_key = _SCALING_DATASET
    if not os.path.exists(dataset_filenames[dataset_key]):
        raise RuntimeError(
            f"generate_scaling needs {_SCALING_DATASET} on disk "
            f"(looked for {dataset_filenames[dataset_key]})."
        )

    stream_sizes = sorted({max(1, stream_size // (2 ** i)) for i in range(n_steps)})
    max_stream = max(stream_sizes)
    k_values = sorted({max(1, round(stream_size * p / 100)) for p in k_percentages})

    # Drop sweep entries that no longer fit, rather than raising, so
    # resizing the sweep never requires hand-editing the lists too. Each
    # list is bounded by the thing it actually indexes: k is a ball count,
    # so it is capped by the construct set (k >= n degenerates the cover
    # into one zero-radius ball per point); batch_size is a slice of the
    # query stream, so it is capped by that -- it has nothing to do with
    # how many vectors the filter was built over.
    def _bounded_by(values: list[int], limit: int, name: str, limit_name: str) -> list[int]:
        usable = sorted(v for v in values if v < limit)
        dropped = sorted(set(values) - set(usable))
        if dropped:
            logger.warning(f"Dropping {name} value(s) >= {limit_name} ({limit}): {dropped}")
        if not usable:
            raise RuntimeError(
                f"No {name} value in {values} is below {limit_name} ({limit}). "
                f"Raise {limit_name} or lower {name}."
            )
        return usable

    k_values = _bounded_by(k_values, max_stream, "k_values", "stream_size")
    batch_sizes = _bounded_by(batch_sizes, n_query_vectors, "batch_sizes", "n_query_vectors")

    # Pool shape (largest stream size + a disjoint holdout); fold f's copy
    # starts fold_stride*f further into the dataset.
    holdout_size = max(n_query_vectors, n_insert_vectors)
    pool_needed = max_stream + holdout_size
    total_rows = len(VectorSource(dataset_key))
    fold_stride = pool_needed // k_folds
    fold_starts = advancing_fold_starts(total_rows, k_folds, fold_stride, pool_needed)

    def _source(start: int, stop: int) -> VectorSource:
        return VectorSource(dataset_key, start, stop)

    for k in k_values:
        for batch_size in batch_sizes:
            for fold, fold_start in enumerate(fold_starts):
                construct_source = _source(fold_start, fold_start + max_stream)
                holdout_source = _source(fold_start + max_stream, fold_start + max_stream + holdout_size)
                runner.submit(
                    'scaling_query',
                    output_csv=SCALING_QUERY_CSV,
                    dim=dim, dataset=_SCALING_DATASET, k=k, batch_size=batch_size, seed=fold,
                    tau_mid=tau_mid,
                    vectors_construct=construct_source, vectors_query=holdout_source,
                    n_query_vectors=n_query_vectors,
                )
                runner.submit(
                    'scaling_insert',
                    output_csv=SCALING_INSERT_CSV,
                    dim=dim, dataset=_SCALING_DATASET, k=k, batch_size=batch_size, seed=fold,
                    vectors_construct=construct_source, vectors_insert=holdout_source,
                    n_insert_vectors=n_insert_vectors,
                )

    for k in k_values:
        for n in stream_sizes:
            if k >= n:
                continue
            for fold, fold_start in enumerate(fold_starts):
                construct_source_n = _source(fold_start, fold_start + n)
                runner.submit(
                    'scaling_construction',
                    output_csv=SCALING_CONSTRUCTION_CSV,
                    dim=dim, dataset=_SCALING_DATASET, k=k, n=n, seed=fold,
                    vectors_construct=construct_source_n,
                )


def save_scaling(job_results: List[Tuple[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Extracts this evaluation's rows from a shared Runner's combined
    run() output (see generate_scaling()). Each of the three query/insert/
    construction job types was submitted under its own mode name AND its own
    output_csv, so the Runner already wrote all three results/scaling_*.csv
    files incrementally as their jobs completed -- this only rebuilds the
    three DataFrames for plot_scaling()."""
    df_query = pd.DataFrame(rows_for_mode(job_results, 'scaling_query'))
    df_insert = pd.DataFrame(rows_for_mode(job_results, 'scaling_insert'))
    df_construction = pd.DataFrame(rows_for_mode(job_results, 'scaling_construction'))
    return df_query, df_insert, df_construction


def plot_scaling(df_query: pd.DataFrame, df_insert: pd.DataFrame, df_construction: pd.DataFrame) -> None:
    generate_scaling_figures(df_query, df_insert, df_construction)


# Panels (b)/(c)/(d) collapse to one representative dataset (Figure 4: "different datasets do not really matter").
_STREAMING_REPRESENTATIVE_DATASET = "MMLU"


def _build_streaming_negative_set(
    dataset: str, stream_vectors: np.ndarray, candidate_pool: np.ndarray, tau_mid: float, n_negative: int,
) -> np.ndarray:
    """A negative that's true against the FULL stream stays true against any
    prefix of it. `stream_vectors` must cover every fold's own stream
    window (not just fold 0's), since folds see genuinely different data."""
    is_duplicate = compute_static_actual_results(stream_vectors, candidate_pool, tau_mid)
    negatives = candidate_pool[~np.asarray(is_duplicate, dtype=bool)]
    if len(negatives) < n_negative:
        raise RuntimeError(
            f"{dataset}: only found {len(negatives)} true negatives against the full stream "
            f"(needed {n_negative}); shrink n_negative, raise tau_mid, or grow the candidate pool."
        )
    return negatives[:n_negative]


def generate_streaming(runner: Runner) -> None:
    """The paper's "dynamic evaluation" (Section 5.1/5.4, Figure 4): compares
    BallCoverFilter's 4 insertion strategies via stream_rebuild_worker's
    policy="streaming". Produces the streaming figure's panels: (a) FPR vs.
    position + a static-rebuild reference, (b) threshold-expanded objective
    inflation vs. position -- no k sweep: the ball budget k stays fixed for
    the whole run, and only the stream position varies.

    The ball budget k, the bootstrap (construct) window, and the subsequent
    insert-only window are each sized as their own fraction of num_vectors:
    k_percent_stream sets k itself, init_percent_stream controls how many
    vectors seed construct() before any insert() call, and
    insert_percent_stream controls how many are then streamed in one at a
    time. Decoupling the bootstrap window from k matters: the old scheme
    bootstrapped on exactly k vectors (bootstrap_n = min(k, n)), which for
    k <= n gives GonzalesConstruct one point per center and hence
    zero-radius balls at position 0 -- a degenerate starting point that
    saturates FPR to ~1 within a few dozen insertions regardless of method.
    Bootstrapping on a window an order of magnitude larger than k gives the
    filter genuinely non-trivial radii before streaming even starts.

    Implements the paper's Workloads protocol (Section 5.1, "80% construction
    / 20% queries"): init_percent_stream + insert_percent_stream is the
    construction share (default 0.4 + 0.4 = 0.8) and the remaining
    1 - init_percent_stream - insert_percent_stream of num_vectors, past
    every fold's window, is held out as the source of verified-negative
    queries. Each of k_folds=5 folds gets its own stream window, starting
    fold_stride further into the representative dataset than fold 0's --
    non-wrapping via advancing_fold_starts, so folds see genuinely different
    data rather than the same pool reordered. The dataset needs enough
    vectors on disk for both the folds' windows and the query span, or this
    raises.

    FNR=0 is asserted per checkpoint inside stream_rebuild_worker (raises,
    not recorded) -- a correctness requirement, not a plotted quantity.

    Only submits 'stream_rebuild' jobs (output_csv=STREAMING_CSV) -- does
    not run `runner`."""
    num_vectors = WORKLOAD_NUM_VECTORS
    dim = 384
    methods = ["nearest", "least_inflation", "merge", "min_heap"]
    n_negative = 500
    duplicate_percentage = 10.0  # calibrates tau_mid; see calculate_nn_distance_threshold
    k_folds = 5  # "we do this 5 times" (Section 5.1)
    seed = 0  # fixed; fold_offset supplies the stream-order variance instead
    k_percent_stream = 0.1  # ball budget k, as a fraction of num_vectors
    init_percent_stream = 0.4  # bootstrap/construct window, as a fraction of num_vectors
    insert_percent_stream = 0.4  # vectors streamed via insert() after bootstrap, as a fraction of num_vectors

    datasets = [d for d in DATASET_KEYS if os.path.exists(dataset_filenames[d])]
    missing = [d for d in DATASET_KEYS if d not in datasets]
    if missing:
        logger.warning(f"Skipping dataset(s) not found on disk: {', '.join(missing)}")
    if set(datasets) != set(DATASET_KEYS):
        raise RuntimeError(
            f"generate_streaming needs all {len(DATASET_KEYS)} datasets "
            f"({', '.join(DATASET_KEYS)}) on disk; only found: {', '.join(datasets) or 'none'}"
        )
    if _STREAMING_REPRESENTATIVE_DATASET not in datasets:
        raise RuntimeError(
            f"generate_streaming's representative dataset {_STREAMING_REPRESENTATIVE_DATASET!r} "
            "is not on disk."
        )

    rep_dataset = _STREAMING_REPRESENTATIVE_DATASET
    rep_key = DATASET_KEYS[rep_dataset]

    k_fixed = max(16, round(k_percent_stream * num_vectors))

    n_init = max(1, round(init_percent_stream * num_vectors))
    n_insert = max(1, round(insert_percent_stream * num_vectors))
    n_stream_window = n_init + n_insert
    if n_stream_window > num_vectors:
        raise RuntimeError(
            f"init_percent_stream={init_percent_stream} + insert_percent_stream="
            f"{insert_percent_stream} needs {n_stream_window} vectors per fold, but "
            f"num_vectors is only {num_vectors}. Lower either percentage or raise num_vectors."
        )

    # construct()+insert() consume n_stream_window vectors per fold; the
    # remainder of num_vectors, past every fold's window, is held out as the
    # source of verified-negative queries.
    fold_stride = n_stream_window // k_folds
    offsets = [f * fold_stride for f in range(k_folds)]
    negative_pool_size = num_vectors - n_stream_window

    # Placed past every fold's window (all contained in [0, offsets[-1] +
    # n_stream_window)) so it can never overlap any fold's stream.
    negative_pool_start = offsets[-1] + n_stream_window
    negative_pool_end = negative_pool_start + negative_pool_size

    total_rows = len(VectorSource(rep_dataset))
    if total_rows < negative_pool_end:
        raise RuntimeError(
            f"{rep_dataset}: has only {total_rows} vectors on disk; generate_streaming's "
            f"non-wrapping {k_folds}-fold sweep at num_vectors={num_vectors} needs at least "
            f"{negative_pool_end} (room for {k_folds} stream windows of {n_stream_window} "
            f"vectors each, advancing by {fold_stride}, plus a {negative_pool_size}-vector "
            "negative-candidate pool past the last one). Raise the dataset's available "
            "vectors, or lower num_vectors."
        )

    # The driver needs real arrays only for calibrating tau_mid and building
    # the negative set (one-time, not per-job) -- everything else stays
    # lazy. `full_span` covers every fold's window, not just fold 0's.
    full_span = VectorSource(rep_dataset, 0, negative_pool_start).to_array()
    negative_candidate_pool = VectorSource(rep_dataset, negative_pool_start, negative_pool_end).to_array()
    if len(negative_candidate_pool) < n_negative:
        raise RuntimeError(
            f"{rep_dataset}: held-out negative-candidate pool has only "
            f"{len(negative_candidate_pool)} vectors (needed >= {n_negative}). "
            "Raise num_vectors or lower n_negative."
        )

    # Calibrated from fold 0's window (full_span's first n_stream_window vectors).
    logger.info(f"{rep_dataset}: calibrating tau_mid against the stream window...")
    tau_mid = calculate_nn_distance_threshold(full_span[:n_stream_window], duplicate_percentage)
    logger.info(f"tau_mid={tau_mid:.4f}")

    logger.info(f"{rep_dataset}: building fixed streaming negative set from the held-out span...")
    negative_vectors = _build_streaming_negative_set(
        rep_dataset, full_span, negative_candidate_pool, tau_mid, n_negative,
    )

    # Bootstrap (construct()) covers positions [0, n_init); the position
    # sweep only checkpoints the subsequent insert-only span, since nothing
    # streams in before n_init.
    checkpoints = sorted(set(
        int(round(x)) for x in np.logspace(np.log10(n_init + 1), np.log10(n_stream_window), num=12)
    ))

    def _fold_source(fold_offset: int, length: int) -> VectorSource:
        return VectorSource(rep_dataset, fold_offset, fold_offset + length)

    # Each row is self-describing (method/sweep/metric/fold_offset), so no
    # positional reconstruction is needed despite Runner's completion-order results.

    # --- panels (a)/(b): position sweep at fixed k_fixed, decoupled bootstrap/insert windows ---
    for method in methods:
        for fold_offset in offsets:
            runner.submit(
                'stream_rebuild',
                output_csv=STREAMING_CSV,
                dim=dim, dataset=rep_key, k=k_fixed, seed=seed, tau_mid=tau_mid, policy="streaming", T=None,
                insertion_level=method, vectors_stream=_fold_source(fold_offset, n_stream_window),
                vectors_negative=negative_vectors, bootstrap_n=n_init,
                checkpoints=checkpoints, sweep="position", fold_offset=fold_offset,
            )
    for fold_offset in offsets:
        runner.submit(
            'stream_rebuild',
            output_csv=STREAMING_CSV,
            dim=dim, dataset=rep_key, k=k_fixed, seed=seed, tau_mid=tau_mid, policy="oracle", T=None,
            vectors_stream=_fold_source(fold_offset, n_stream_window), vectors_negative=negative_vectors,
            checkpoints=checkpoints, sweep="position", fold_offset=fold_offset,
        )


def save_streaming(job_results: List[Tuple[str, Any]]) -> Tuple[pd.DataFrame, int]:
    """Rebuilds this evaluation's DataFrame from a shared Runner's run()
    output, plus k_fixed (the smallest k among 'position'-sweep rows,
    matching plot_all()'s own derivation)."""
    df = pd.DataFrame(rows_for_mode(job_results, 'stream_rebuild'))
    k_fixed = int(df.loc[df["sweep"] == "position", "k"].iloc[0])
    return df, k_fixed


def plot_streaming(df: pd.DataFrame, k_fixed: int) -> None:
    generate_streaming_figures(df, k_fixed=k_fixed)


def generate_and_plot_all() -> None:
    """Submit-all, run-all, then plot-all: every generate_* function below
    only submits jobs to one shared Runner instead of running its own, so
    the combined job list executes as a single parallel batch and
    runner.shuffle() can interleave fast and slow jobs across evaluations
    rather than one worker draining a whole slow evaluation alone.

    Each generate_ function tags its jobs with its own output_csv, so the
    Runner writes each results/*.csv file incrementally as that
    evaluation's jobs complete, not after the whole batch finishes -- a
    crash mid-run only loses whichever jobs hadn't completed yet. There's
    no separate save-all step: save_* functions still exist for a caller
    that wants rows back as a DataFrame directly, but aren't needed here.

    Each generate_ call is independently try/except-guarded so one
    evaluation failing to submit doesn't take down the others. A failure
    inside runner.run() itself aborts the whole batch, but every job that
    did complete beforehand is still on disk."""
    runner = Runner(num_workers=MAX_WORKERS)
    
    try:
        generate_system(runner)
    except Exception:
        logger.exception("An error occurred")
    
    '''    
    try:
        generate_system_tau(runner)
    except Exception:
        logger.exception("An error occurred")

    try:
        generate_system(runner)
    except Exception:
        logger.exception("An error occurred")

    try:
        generate_quantization(runner)
    except Exception:
        logger.exception("An error occurred")
    try:
        generate_scaling(runner)
    except Exception:
        logger.exception("An error occurred")
    try:
        generate_streaming(runner)
    except Exception:
        logger.exception("An error occurred")
    try:
        generate_equal_memory(runner)
    except Exception:
        logger.exception("An error occurred")
    try:
        generate_system_tau(runner)
    except Exception:
        logger.exception("An error occurred")
    '''
    runner.shuffle()
    runner.run()
    logger.info("All submitted jobs completed.")

    plot_all()


def plot_all() -> None:
    """Re-renders every figure from data already persisted to disk, without
    repeating the evaluation sweeps -- handy for iterating on plot styling
    alone. Re-derives whatever generate-time metadata (k, k_fixed) each
    plot_ needs from the loaded data. Requires at least one prior
    generate_and_plot_all() (or generate_X + runner.run()) to have produced
    those files."""
    try:
        df = pd.read_csv(SYSTEM_TAU_CSV)
        plot_system_tau(df)
    except Exception:
        logger.exception("An error occurred")

    try:
        df = pd.read_csv(SYSTEM_CSV)
        plot_system(df)
    except Exception:
        logger.exception("An error occurred")

    try:
        # keep_default_na=False: stage/config strings are read verbatim; pandas
        # would otherwise read back as NaN (see plot_quantization.py::load_data()).
        df = pd.read_csv(QUANTIZATION_CSV, keep_default_na=False)
        plot_quantization(df)
    except Exception:
        logger.exception("An error occurred")

    try:
        results = pd.read_csv(EQUAL_MEMORY_CSV).to_dict("records")
        plot_equal_memory(results)
    except Exception:
        logger.exception("An error occurred")

    try:
        df = pd.read_csv("results/query_performance.csv")
        plot_query_performance(df)
    except Exception:
        logger.exception("An error occurred")

    try:
        df_query = pd.read_csv(SCALING_QUERY_CSV)
        df_insert = pd.read_csv(SCALING_INSERT_CSV)
        df_construction = pd.read_csv(SCALING_CONSTRUCTION_CSV)
        plot_scaling(df_query, df_insert, df_construction)
    except Exception:
        logger.exception("An error occurred")

    try:
        df = pd.read_csv(STREAMING_CSV)
        k_fixed = int(df.loc[df["sweep"] == "position", "k"].iloc[0])
        plot_streaming(df, k_fixed=k_fixed)
    except Exception:
        logger.exception("An error occurred")


if __name__ == "__main__":
    generate_and_plot_all()
    # plot_all()  # alternative: re-render figures from an existing results/*.csv run