import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

# Must happen before numpy/faiss are imported (their BLAS/OpenMP backends
# read these once at init and cache the thread count) -- otherwise each of
# Runner's MAX_WORKERS worker processes independently tries to grab every
# CPU core for its own internal matrix/distance-computation threads, and
# process-level x thread-level parallelism multiply into severe
# oversubscription (observed: escalating per-job latency under load, up to
# whole-system unresponsiveness). One thread per process is correct here --
# the parallelism already comes from running MAX_WORKERS processes.
for _env_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env_var, "1")

import faiss
import time
import inspect
import numpy as np
import pandas as pd

faiss.omp_set_num_threads(1)  # belt-and-suspenders: explicit, not just env-var-dependent

from logging_config import configure_logging
from runner import Runner, QUANTIZATION_SWEEP, system_tau_ground_truth_worker
from plots.plot_quantization import generate_quantization_figures
from plots.plot_query_performance import generate_query_performance_figures
from plots.plot_equal_memory import generate_equal_memory_figures
from plots.plot_scaling import generate_scaling_figures
from plots.plot_streaming import generate_streaming_figures
from plots.plot_system_tau import generate_system_tau_figures
from plots.plot_common import DATASETS, DATASET_LABELS
from typing import Tuple
from filters.ball_cover.ball_cover_filter import BallCoverFilter
from filters.lsbf_filter import LSBFFilter
from filters.dsh_filter import DSHFilter
from filters.hnsw_filter import HNSWFilter
from filters.ivf_filter import IVFFilter
from filters.quantized_vector_filter import QuantizedVectorFilter
from util.dataset_paths import dataset_filenames
from util.fetch_datasets import load_embeds
from vector_source import VectorSource, ArrayVectorSource

configure_logging()
logger = logging.getLogger(__name__)

MAX_WORKERS = 4
DATASET = 'ELI5'

# Single knob for how many vectors every generate_* evaluation pulls as its
# input stream. 1_000 is conservative -- some evaluations are documented to
# blow up or crash well below the 100_000 they used to hardcode; check a
# function's own docstring before raising this.
STREAM_SIZE = 10_000

# Vectors/dataset for the evaluations governed by the paper's Workloads
# protocol (Section 5.1): "We utilize 100K queries from each dataset for
# evaluation, with 80% used for construction of the filter and 20% for
# queries." Kept separate from STREAM_SIZE -- which throughput-only sweeps
# (scaling, query_performance) still use at a smaller, paper-independent
# scale -- so raising it only affects the ground-truth FPR/FNR evaluations
# (system_tau, quantization, equal_memory, streaming) the paper actually
# specifies this number for.
WORKLOAD_NUM_VECTORS = 10_000

# Registry name -> canonical dataset key, shared by every long-format
# evaluation (quantization, query_performance, scaling, streaming). Derived from
# plot_common so the dataset trio has one source of truth.
DATASET_KEYS = {DATASET_LABELS[d]: d for d in DATASETS}

dynamic_filters = [BallCoverFilter, LSBFFilter, DSHFilter, HNSWFilter, IVFFilter]
static_filters  = []

# generate_equal_memory()'s own comparison set. The other dynamic_filters
# entries (DSBFFilter, CoveringLSHFallbackFilter, LSBFFilter) degenerate to
# ~100% duplicate predictions at realistic construct-set sizes, making them
# uninformative for a budget-sensitivity comparison; they stay in
# dynamic_filters for the other evaluations.
equal_memory_filters = [BallCoverFilter, HNSWFilter, IVFFilter, DSHFilter, QuantizedVectorFilter]

def calculate_max_memory_budget_bytes(num_vectors: int, dimension: int, bytes_per_element: float) -> int:
    return num_vectors * dimension * bytes_per_element // 64

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

    Full pairwise distance is O(n^2) memory (~40 GB at n=100k), so this
    samples up to `max_pairs // n` query rows and computes exact L2 against
    the full set via faiss instead -- an unbiased sample of the pairwise
    distribution, within ~0.2% of the exact percentile at bounded memory.
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


def calculate_nn_distance_threshold(
    construct_vectors: np.ndarray,
    percentage: float,
    sample_size: int = 5_000,
    seed: int = 42,
) -> float:
    """
    Return the distance threshold at which `percentage`% of a sample of
    construct_vectors have their nearest *other* construct vector within that
    distance.

    Unlike calculate_distance_threshold (random pairwise distances), this
    ties the threshold to the actual nearest-neighbour test
    compute_static_actual_results performs -- at large construct-set sizes a
    point's nearest neighbour is much closer than a random pair, so the
    pairwise-percentile version silently pushes the duplicate rate toward
    100%, collapsing FPR/FNR to degenerate values. Used by
    generate_equal_memory(), whose construct set is large enough for this to
    matter.
    """
    n = len(construct_vectors)
    if n < 2:
        return 0.0

    vecs = np.ascontiguousarray(construct_vectors, dtype=np.float32)
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(n, size=sample_size, replace=False) if sample_size < n else np.arange(n)

    index = faiss.IndexFlatL2(vecs.shape[1])
    index.add(vecs)
    # k=2: a sampled point's own nearest neighbour within its own reference
    # set is itself (distance 0) at rank 0, so the *other* nearest neighbour
    # is rank 1.
    sq_dist, _ = index.search(vecs[sample_idx], 2)
    nn_dist = np.sqrt(np.clip(sq_dist[:, 1], 0, None))
    return float(np.percentile(nn_dist, percentage))


def kfold_wrapped_split(vectors: np.ndarray, fold: int, k_folds: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implements the paper's contiguous-span K-fold protocol (Section 5.1,
    "Workloads"): one contiguous span of the dataset (80% when k_folds=5) is
    the indexed set X, and the adjacent contiguous span (the remaining 20%)
    is the query set Q, wrapping around to the start of the dataset when a
    span extends past the end. The query span's start advances by one
    query-span-width per fold, so across k_folds folds every vector serves
    as a query exactly once and as a construct vector the rest of the time
    ("each part of the stream gets to be both in query and in the
    construction").

    Returns (construct_vectors, query_vectors).
    """
    n = len(vectors)
    query_len = n // k_folds
    construct_len = n - query_len
    query_start = fold * query_len
    construct_start = (query_start + query_len) % n
    query_idx = np.arange(query_start, query_start + query_len) % n
    construct_idx = np.arange(construct_start, construct_start + construct_len) % n
    return vectors[construct_idx], vectors[query_idx]


def compute_dynamic_actual_results(vectors: np.ndarray, distance_threshold: float) -> list[bool]:
    """
    Online ground-truth: the first vector is unique; each subsequent vector is a
    duplicate if it falls within `distance_threshold` of any previously accepted
    unique vector, otherwise it joins the reference set.
    """
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)

    actual = [False]
    thresh_sq = distance_threshold ** 2

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors[0:1])

    for i in range(1, len(vectors)):
        v = vectors[i:i+1]

        # find the single nearest accepted vector
        nn_sq_dist, _ = index.search(v, 1)

        # check if the closest vector is within the threshold
        is_dup = bool(nn_sq_dist[0, 0] <= thresh_sq)
        actual.append(is_dup)
        if not is_dup:
            index.add(v)

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


def generate_static() -> list:
    num_vectors              = STREAM_SIZE
    dim                      = 384
    min_percentage_duplicates = 1
    max_percentage_duplicates = 9
    count_steps              = 9
    count_memory_steps       = 10
    percentage_construct     = 10

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
        logger.info(f"Processing {percentage:.1f}% duplicates...")
        logger.info("Calculating distance threshold...")
        distance_threshold = calculate_distance_threshold(vectors, percentage)
        logger.info("Calculating actual results...")
        actual_results     = compute_static_actual_results(vectors_construct, vectors_query, distance_threshold)

        for memory_budget in memory_budgets:
            for filter_cls in static_filters + dynamic_filters:
                runner.submit(
                    'static',
                    filter_cls=filter_cls,
                    dim=dim, distance_threshold=distance_threshold,
                    memory_budget=memory_budget,
                    vectors_construct=construct_source, vectors_query=query_source,
                    actual_results=actual_results, percentage=percentage,
                )

    results = runner.run()
    logger.info("Static evaluation completed.")
    return results


def generate_equal_memory() -> list:
    """
    Equal-memory comparison across equal_memory_filters (see its own comment
    for why some dynamic_filters baselines are excluded here). Sweeps
    count_memory_steps budgets up to max_memory_budget, scaled off the
    construct set itself (not calculate_max_memory_budget_bytes, which is
    tuned for generate_static()/generate_dynamic()'s smaller num_vectors).

    For each budget, every filter's best config is chosen on a held-out
    calibration set (see runner.py::_build_calibrated_filter); FPR/FNR,
    construction latency, and query throughput are then measured on a
    separate held-out query set.

    num_vectors/percentage_construct follow the paper's Workloads protocol
    (Section 5.1): 100K vectors, 80% construct (the remaining 20% is then
    split between calibration and query). Section 5.6's SOTA-comparison
    figure names no specific dataset, so this uses the paper's default
    single-dataset representative, MMLU.
    """
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

    # Jobs receive lazy on-disk pointers instead of array copies (same
    # reasoning as generate_static).
    construct_source   = VectorSource(dataset, 0, num_construct)
    calibration_source = VectorSource(dataset, num_construct, num_construct + num_calibration)
    query_source        = VectorSource(dataset, num_construct + num_calibration, num_vectors)

    logger.info("Calculating distance threshold...")
    # Nearest-neighbour-based, not calculate_distance_threshold's random-pairwise
    # percentile: with a 40%-of-100k = 40k-vector construct set, a pairwise
    # percentile is far more permissive than its nominal percentage once
    # tested against "nearest of 40k candidates" (see calculate_nn_distance_threshold's
    # docstring) -- that mismatch was pushing the actual duplicate rate to ~100%,
    # collapsing both FPR and FNR to degenerate 0/0 values regardless of budget.
    distance_threshold = calculate_nn_distance_threshold(vectors_construct, duplicate_percentage)
    logger.info("Calculating calibration/query ground truth...")
    actual_calibration = compute_static_actual_results(vectors_construct, vectors_calibration, distance_threshold)
    actual_query       = compute_static_actual_results(vectors_construct, vectors_query, distance_threshold)

    # Scaled off the construct set itself (one sphere per point), not
    # calculate_max_memory_budget_bytes -- see docstring.
    bytes_per_sphere_at_1to1 = (dim + 1) * 4 * 4.0
    max_memory_budget = int(num_construct * bytes_per_sphere_at_1to1)
    memory_budgets = np.linspace(0, max_memory_budget, count_memory_steps + 1, dtype=int)[1:]

    runner = Runner(num_workers=MAX_WORKERS)
    for memory_budget in memory_budgets:
        logger.info(f"Submitting jobs for memory_budget={memory_budget} bytes...")
        for filter_cls in equal_memory_filters:
            runner.submit(
                'equal_memory',
                filter_cls=filter_cls,
                dim=dim, distance_threshold=distance_threshold,
                memory_budget=memory_budget,
                vectors_construct=construct_source, vectors_calibration=calibration_source,
                vectors_query=query_source,
                actual_calibration=actual_calibration, actual_query=actual_query,
                dataset=dataset,
            )

    results = runner.run()
    logger.info("Equal-memory evaluation completed.")

    results_df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    results_df.to_csv("results/equal_memory.csv", index=False)
    logger.info("Saved results/equal_memory.csv")

    return results


def plot_equal_memory(results: list) -> None:
    generate_equal_memory_figures(df=pd.DataFrame(results))


def generate_system_tau() -> pd.DataFrame:
    """
    End-to-end filtering benefit (paper Section 5.2, Figure 2): one
    BallCoverFilter cover built per (dataset, fold, k), then queried at
    every tau in a fixed, dataset-independent sweep (system_tau_worker) --
    the same constructed representation is reused across all thresholds,
    never rebuilt. Unlike the paper's own fixed k=400, k here is swept as a
    ball-budget fraction of each fold's construct-set size (k_fractions,
    1%-10%) so plot_system_tau can show how the filtering benefit varies
    with the memory budget, not just with tau.

    Implements the paper's Workloads protocol (Section 5.1) exactly: for
    each of the 5 datasets, WORKLOAD_NUM_VECTORS vectors are split into
    k_folds=5 contiguous-span folds via kfold_wrapped_split (80% construct /
    20% query per fold, wrapping around the dataset end, split point
    advanced by one query-span-width per fold so every vector serves as
    both query and construct across the 5 folds). plot_system_tau then
    reports the mean +/- std over these 5 folds (at whichever k it's
    rendering), matching the paper's "error bars represent the standard
    deviation during our 5 runs".

    Ground truth (nearest-construct-set distance per query vector, one flat
    L2 index build + search per (dataset, fold)) does not depend on k, so it
    is computed once per (dataset, fold) -- not per (dataset, fold, k) --
    in parallel across a ProcessPoolExecutor rather than serially in this
    process, and cached to results/system_tau_ground_truth/ so a repeat
    call with the same (dataset, num_vectors, k_folds, fold) doesn't redo
    that (expensive, at 100K-scale) nearest-neighbor search.

    Writes results/system_tau.csv. Returns the long-format df (k now varies
    per row, so there's no single fixed k left to return alongside it).
    """
    num_vectors      = WORKLOAD_NUM_VECTORS
    dim              = 384
    k_folds          = 5
    # Ball budget as a fraction of each fold's construct-set size (~80,000
    # vectors at num_vectors=100_000), swept 1%-10% instead of the paper's
    # fixed k=400 -- note this yields much larger ball counts (roughly
    # 800-8,000) than the paper's figure, so this sweep is significantly
    # more expensive to run than the paper's single-k evaluation.
    k_fractions      = [float(f) for f in np.linspace(0.01, 0.1, 5)]
    tau_values       = [float(t) for t in np.linspace(0.01, 0.10, 10)]
    output_csv       = "results/system_tau.csv"
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

    logger.info(f"Ball budget swept as fractions of the construct set: {k_fractions}")
    logger.info(f"Fixed tau sweep (shared across datasets): {tau_values}")
    logger.info(f"K-fold cross-validation: {k_folds} folds, contiguous 80/20 spans with wraparound")

    # tau is fixed up front (not solved for), so the construction-tightness
    # knob is a single global median instead of a per-dataset one.
    construct_tau = float(np.median(tau_values))

    # One entry per (dataset, fold), built once and reused by both passes
    # below: (dataset, fold, vectors_construct, vectors_query).
    splits = []
    for dataset in datasets:
        vectors = load_embeds(dataset, num_vectors)[0]
        if len(vectors) < num_vectors:
            logger.warning(f"{dataset}: only {len(vectors)} vectors on disk (< requested {num_vectors})")
        for fold in range(k_folds):
            vectors_construct, vectors_query = kfold_wrapped_split(vectors, fold, k_folds)
            splits.append((dataset, fold, vectors_construct, vectors_query))

    # Ground truth (nearest-construct-set distance per query vector): load
    # whatever's already cached on disk, compute the rest in parallel.
    os.makedirs(ground_truth_dir, exist_ok=True)
    nn_dist_by_split = {}
    to_compute = []
    for dataset, fold, vectors_construct, vectors_query in splits:
        cache_path = os.path.join(ground_truth_dir, f"{dataset}_{num_vectors}_{k_folds}fold_{fold}.npy")
        if os.path.exists(cache_path):
            nn_dist_by_split[(dataset, fold)] = np.load(cache_path)
        else:
            to_compute.append((dataset, fold, cache_path, vectors_construct, vectors_query))

    if to_compute:
        logger.info(f"Computing ground truth for {len(to_compute)} (dataset, fold) pair(s) not yet cached...")
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(system_tau_ground_truth_worker, dim, vc, vq): (dataset, fold, cache_path)
                for dataset, fold, cache_path, vc, vq in to_compute
            }
            for future in as_completed(futures):
                dataset, fold, cache_path = futures[future]
                nn_dist = future.result()
                np.save(cache_path, nn_dist)
                nn_dist_by_split[(dataset, fold)] = nn_dist
    else:
        logger.info("All ground truth already cached on disk; skipping recomputation.")

    runner = Runner(num_workers=MAX_WORKERS)

    for dataset, fold, vectors_construct, vectors_query in splits:
        nn_dist = nn_dist_by_split[(dataset, fold)]
        actual_results_by_tau = {t: (nn_dist <= t).tolist() for t in tau_values}

        construct_len = len(vectors_construct)
        k_values = sorted({max(1, round(f * construct_len)) for f in k_fractions})
        k_values = [k for k in k_values if k < construct_len]

        for k in k_values:
            runner.submit(
                'system_tau',
                dim=dim, dataset=DATASET_KEYS[dataset],
                k=k, tau_values=tau_values, seed=fold, construct_tau=construct_tau,
                vectors_construct=ArrayVectorSource(
                    dataset_name=dataset, start=0, stop=len(vectors_construct), array=vectors_construct
                ),
                vectors_query=ArrayVectorSource(
                    dataset_name=dataset, start=0, stop=len(vectors_query), array=vectors_query
                ),
                actual_results_by_tau=actual_results_by_tau,
                )

    job_results = runner.run()
    logger.info("System (tau-sweep x k-sweep, K-fold) evaluation completed.")

    df = pd.DataFrame([row for job_rows in job_results for row in job_rows])

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info(f"Wrote {output_csv}")

    return df


def plot_system_tau(df: pd.DataFrame) -> None:
    generate_system_tau_figures(df)


def generate_quantization() -> pd.DataFrame:
    """
    Generate per-quantize-variant construction data for BallCoverFilter
    (QUANTIZATION_SWEEP) on a single dataset, written to
    results/quantization.csv. Every candidate shares one memory budget per k
    (derived from the unquantized cover's byte-per-ball size), so a more
    compact quantizer packs more balls into the same budget -- letting
    plot_quantization.py use recorded memory usage as the x-axis. A
    quantizer compact enough to want more balls than the construct set has
    saturates at n_construct balls short of the nominal budget instead (see
    quantization_worker) -- real information, not a plotting bug.

    k_values/tau_values derive from num_vectors/percentage_construct rather
    than hardcoded absolutes, so those two knobs rescale the whole sweep:
    k_fractions stay < 1 (k >= the construct set makes BallCoverFilter fall
    back to one zero-radius ball per point), and tau_values is a percentile
    of the dataset's own nearest-neighbour distance rather than a raw float,
    so it stays meaningful as the construct set's scale changes.

    Dataset and duplicate percentage follow the paper's Figure 3 caption
    exactly ("We evaluate on MS MARCO, we select tau such that 5% of the
    documents are duplicates"), while num_vectors/percentage_construct
    follow the paper's general Workloads protocol (Section 5.1): 100K
    vectors, 80% construct / 20% query.

    K-fold cross-validation (Section 5.1): 5 contiguous-span folds via
    kfold_wrapped_split (wrapping around the dataset end), tagged via
    seed=fold so plot_quantization's existing seed-based mean +/- 95% CI
    aggregation reports variance across folds instead of always drawing
    0-width error bars from a single run. tau_values are calibrated once
    (not per fold, from fold 0's construct set) so every fold lands on the
    exact same tau and can be aggregated together; each fold's ground truth
    is still recomputed from that fold's own construct/query split.
    """
    num_vectors            = WORKLOAD_NUM_VECTORS
    dim                    = 384
    k_folds                = 5
    k_fractions            = [0.01, 0.02, 0.04, 0.08, 0.16]
    duplicate_percentages  = [5]
    output_csv             = "results/quantization.csv"

    dataset = "MsMarco"  # Figure 3: "We evaluate on MS MARCO"
    vectors = load_embeds(dataset, num_vectors)[0]
    if len(vectors) < num_vectors:
        logger.warning(f"{dataset}: only {len(vectors)} vectors on disk (< requested {num_vectors})")

    splits = [kfold_wrapped_split(vectors, fold, k_folds) for fold in range(k_folds)]
    construct_len = len(splits[0][0])  # identical across folds by construction

    k_values = sorted({max(1, round(construct_len * f)) for f in k_fractions})
    k_values = [k for k in k_values if k < construct_len]
    if not k_values:
        raise RuntimeError(
            f"No k_fraction in {k_fractions} produced a k below the construct set "
            f"({construct_len} vectors). Raise num_vectors or lower k_fractions."
        )
    logger.info(f"Scaling k_values to {k_values} ({construct_len} vectors in the construct set)")

    tau_values = [calculate_nn_distance_threshold(splits[0][0], p) for p in duplicate_percentages]
    logger.info(f"Scaling tau_values to {tau_values} ({duplicate_percentages}-th percentile NN distance, fold 0)")

    # Kept for interface parity with quantization_worker's signature; unused there
    # now that BallCoverFilter's Gonzalez construction has no distance-
    # threshold knob to feed (only quantize varies the cover, not tau).
    construct_tau = float(np.median(tau_values))

    runner = Runner(num_workers=MAX_WORKERS)
    for fold, (vectors_construct, vectors_query) in enumerate(splits):
        logger.info(f"Computing ground truth for fold {fold} (tau in {tau_values})...")
        actual_results_by_tau = {
            tau: compute_static_actual_results(vectors_construct, vectors_query, tau)
            for tau in tau_values
        }
        construct_source = ArrayVectorSource(
            dataset_name=dataset, start=0, stop=len(vectors_construct), array=vectors_construct
        )
        query_source = ArrayVectorSource(
            dataset_name=dataset, start=0, stop=len(vectors_query), array=vectors_query
        )
        for k in k_values:
            for quantize in QUANTIZATION_SWEEP:
                runner.submit(
                    'quantization',
                    dim=dim, dataset=dataset,
                    k=k, quantize=quantize, tau_values=tau_values, seed=fold, construct_tau=construct_tau,
                    vectors_construct=construct_source, vectors_query=query_source,
                    actual_results_by_tau=actual_results_by_tau,
                )

    job_results = runner.run()
    logger.info("Quantization evaluation completed.")

    df = pd.DataFrame([row for job_rows in job_results for row in job_rows])

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info(f"Wrote {output_csv}")

    return df


def plot_quantization(df: pd.DataFrame) -> None:
    generate_quantization_figures(df)


def generate_query_performance() -> pd.DataFrame:
    """
    Measure single-query vs. batched-query latency/throughput of a
    BallCoverFilter cover as a function of filter size k, across the 3
    datasets in DATASET_KEYS (unlike generate_quantization(), which now runs on
    a single dataset). One job per (dataset, k, seed), flattened into a
    long-format DataFrame, written to results/query_performance.csv. Render
    with plot_query_performance().
    """
    num_vectors           = STREAM_SIZE
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
    # one zero-radius ball per point, so clamp the sweep the same way
    # generate_quantization() does (kept defensive even though the 100K-row
    # datasets comfortably support k=1024 at 20% construct).
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
                    k=k, seed=seed, construct_tau=query_tau, query_tau=query_tau,
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


# These panels compare across batch_size/stream_size, not across datasets,
# so sweeping all 3 datasets would only add runtime, not a comparison axis --
# one representative dataset (same single-dataset pattern as
# _STREAMING_REPRESENTATIVE_DATASET) is enough. Figure 5's caption: "The
# specific dataset has no measurable effect on any of these values", so this
# uses the paper's default single-dataset representative, MMLU.
_SCALING_DATASET = "MMLU"


def generate_scaling() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Measures three BallCoverFilter throughput surfaces against filter size k:
    query throughput (k x batch_size grid), insertion throughput (same grid,
    timing add() instead), and construction throughput (k x n, the one panel
    where the indexed-set size itself varies). Runs on a single dataset
    (_SCALING_DATASET).

    stream_sizes (the 6 construction lines) derive from `stream_size` alone;
    k_values/batch_sizes are independent swept axes.

    K-fold cross-validation (paper Section 5.1): rather than 3 repeated
    timing trials on one fixed construct/holdout split, a shared pool is
    rotated (np.roll, wrapping around the pool's end) to k_folds=5 different
    starting offsets -- the same fold-offset pattern generate_streaming()
    uses -- so each of the 5 repeats also measures throughput against
    genuinely different data, not just the same slice timed again. Tagged
    via seed=fold so plot_scaling's existing seed-based mean +/- 95% CI
    aggregation reports variance across folds.
    """
    stream_size      = STREAM_SIZE
    dim              = 384
    n_steps          = 6
    k_values         = [16, 32, 64, 128, 256, 512]
    batch_sizes      = [1, 4, 16, 64, 256, 1024]
    tau_mid          = 0.35
    k_folds          = 5
    n_query_vectors  = 8192
    n_insert_vectors = 8192
    query_csv        = "results/scaling_query.csv"
    insert_csv       = "results/scaling_insert.csv"
    construction_csv = "results/scaling_construction.csv"

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

    # Drop k_values/batch_sizes entries that no longer fit stream_size,
    # rather than raising, so shrinking stream_size alone never requires
    # hand-editing the sweep lists too.
    def _below_stream(values: list[int], name: str) -> list[int]:
        usable = sorted(v for v in values if v < max_stream)
        dropped = sorted(set(values) - set(usable))
        if dropped:
            logger.warning(f"Dropping {name} value(s) >= stream_size ({max_stream}): {dropped}")
        if not usable:
            raise RuntimeError(
                f"No {name} value in {values} is below stream_size ({max_stream}). "
                f"Raise stream_size or lower {name}."
            )
        return usable

    k_values = _below_stream(k_values, "k_values")
    batch_sizes = _below_stream(batch_sizes, "batch_sizes")

    # Shared pool (the largest stream size + a disjoint holdout), rotated
    # per fold below -- only k/batch_size vary within a fold's (construct,
    # holdout) pair in the query/insert sweeps.
    holdout_size = max(n_query_vectors, n_insert_vectors)
    pool_needed = max_stream + holdout_size
    vectors = load_embeds(dataset_key, pool_needed)[0]
    total = len(vectors)
    if total < pool_needed:
        raise RuntimeError(
            f"{_SCALING_DATASET} only has {total} vectors on disk; need >= {pool_needed} "
            f"(stream_size's max {max_stream} + a {holdout_size}-vector holdout). "
            "Lower stream_size."
        )

    fold_stride = pool_needed // k_folds
    rotated_pools = [np.roll(vectors, -(f * fold_stride), axis=0) for f in range(k_folds)]

    def _source(array: np.ndarray) -> ArrayVectorSource:
        return ArrayVectorSource(dataset_name=dataset_key, start=0, stop=len(array), array=array)

    runner = Runner(num_workers=MAX_WORKERS)

    for k in k_values:
        for batch_size in batch_sizes:
            for fold in range(k_folds):
                pool = rotated_pools[fold]
                construct_source = _source(pool[:max_stream])
                holdout_source = _source(pool[max_stream:max_stream + holdout_size])
                runner.submit(
                    'scaling_query',
                    dim=dim, dataset=_SCALING_DATASET, k=k, batch_size=batch_size, seed=fold,
                    tau_mid=tau_mid,
                    vectors_construct=construct_source, vectors_query=holdout_source,
                    n_query_vectors=n_query_vectors,
                )
                runner.submit(
                    'scaling_insert',
                    dim=dim, dataset=_SCALING_DATASET, k=k, batch_size=batch_size, seed=fold,
                    vectors_construct=construct_source, vectors_insert=holdout_source,
                    n_insert_vectors=n_insert_vectors,
                )

    for k in k_values:
        for n in stream_sizes:
            if k >= n:
                continue
            for fold in range(k_folds):
                construct_source_n = _source(rotated_pools[fold][:n])
                runner.submit(
                    'scaling_construction',
                    dim=dim, dataset=_SCALING_DATASET, k=k, n=n, seed=fold,
                    vectors_construct=construct_source_n,
                )

    job_results = runner.run()
    logger.info("Scaling evaluation completed.")

    # Jobs interleave all three modes and Runner.run() returns results in
    # completion order, not submission order -- split by each mode's own
    # rate column (self-describing) rather than by position.
    df_query = pd.DataFrame([r for r in job_results if "throughput_qps" in r])
    df_insert = pd.DataFrame([r for r in job_results if "throughput_ips" in r])
    df_construction = pd.DataFrame([r for r in job_results if "throughput_vps" in r])

    for df, path in (
        (df_query, query_csv), (df_insert, insert_csv), (df_construction, construction_csv),
    ):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        logger.info(f"Wrote {path}")

    return df_query, df_insert, df_construction


def plot_scaling(df_query: pd.DataFrame, df_insert: pd.DataFrame, df_construction: pd.DataFrame) -> None:
    generate_scaling_figures(df_query, df_insert, df_construction)


# Panels (b)/(c)/(d) collapse to one representative dataset -- Figure 4's
# caption: "Different datasets do not really matter", so this uses the
# paper's default single-dataset representative, MMLU.
_STREAMING_REPRESENTATIVE_DATASET = "MMLU"


def _build_streaming_negative_set(
    dataset: str, stream_vectors: np.ndarray, candidate_pool: np.ndarray, tau_mid: float, n_negative: int,
) -> np.ndarray:
    """A negative that's true against the FULL stream stays true against any
    prefix of it, so this stays valid at every streaming checkpoint."""
    is_duplicate = compute_static_actual_results(stream_vectors, candidate_pool, tau_mid)
    negatives = candidate_pool[~np.asarray(is_duplicate, dtype=bool)]
    if len(negatives) < n_negative:
        raise RuntimeError(
            f"{dataset}: only found {len(negatives)} true negatives against the full stream "
            f"(needed {n_negative}); shrink n_negative, raise tau_mid, or grow the candidate pool."
        )
    return negatives[:n_negative]


def generate_streaming() -> Tuple[pd.DataFrame, int]:
    """
    The paper's "dynamic evaluation" (Section 5.1/5.4, Figure 4): compares
    BallCoverFilter's 4 insertion strategies (see runner._INSERT_STRATEGIES)
    via stream_rebuild_worker's policy="streaming". Produces the long-format
    data behind the streaming figure's four panels: (a) FPR vs. stream
    position + a static-rebuild reference, (b) threshold-expanded objective
    inflation vs. stream position, (c) final FPR vs. ball budget k, (d)
    Pareto of mean FPR vs. amortized cost/insertion.

    Implements the paper's Workloads protocol (Section 5.1) exactly: "The
    dynamic evaluation uses the same construction [as the static evaluation],
    varying the stream starting position across folds so that insertions
    arrive in the dataset order beginning at each fold offset." Of
    WORKLOAD_NUM_VECTORS vectors loaded from the representative dataset, 80%
    form the shared stream pool (mirroring the static evaluation's 80%
    construct share) and the remaining 20% are held out as the source of
    verified-negative queries. Each of the k_folds=5 folds rotates the pool
    to start at a different fold offset (np.roll, wrapping around the pool's
    end) -- true K-fold, not an overlapping-window approximation.

    FNR=0 is asserted per checkpoint inside stream_rebuild_worker (raises,
    not recorded) -- a correctness requirement, not a plotted quantity.

    Every other size-related knob (k_fixed, k_values) derives from the
    stream pool size alone; k_values are dropped (with a printed reason) if
    too large relative to it to be a meaningful sweep point.

    Writes results/streaming.csv. Returns the long-format df plus k_fixed,
    which plot_streaming() needs and can't derive from the df alone.
    """
    num_vectors = WORKLOAD_NUM_VECTORS
    dim = 384
    k_values = [16, 32, 64, 128, 256, 512, 1024]
    methods = ["nearest", "least_inflation", "merge", "min_heap"]
    n_negative = 500
    duplicate_percentage = 10.0  # calibrates tau_mid; see calculate_nn_distance_threshold
    k_folds = 5  # "we do this 5 times" (Section 5.1)
    stream_fraction = 0.8  # mirrors the static evaluation's 80% construct share
    seed = 0  # fixed; fold_offset supplies the stream-order variance instead
    output_csv = "results/streaming.csv"

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

    all_vectors = load_embeds(rep_dataset, num_vectors)[0]
    if len(all_vectors) < num_vectors:
        logger.warning(f"{rep_dataset}: only {len(all_vectors)} vectors on disk (< requested {num_vectors})")

    n_stream_pool = int(len(all_vectors) * stream_fraction)
    stream_pool = all_vectors[:n_stream_pool]
    negative_candidate_pool = all_vectors[n_stream_pool:]
    if len(negative_candidate_pool) < n_negative:
        raise RuntimeError(
            f"{rep_dataset}: held-out negative-candidate pool has only "
            f"{len(negative_candidate_pool)} vectors (needed >= {n_negative}). "
            "Raise num_vectors or lower n_negative."
        )

    k_fixed = max(16, n_stream_pool // 1000)
    n_stream_ab = min(k_fixed * 1000, n_stream_pool)
    n_stream_c = n_stream_pool

    # k must leave enough room in the k-sweep's shared stream to be a
    # meaningful sweep point (not just barely fit) -- dropping oversized k
    # values here means num_vectors alone can shrink the whole evaluation
    # without also having to hand-edit k_values to match.
    usable_k_values = [k for k in k_values if k * 10 < n_stream_c]
    dropped_k = sorted(set(k_values) - set(usable_k_values))
    if dropped_k:
        logger.warning(f"Dropping k value(s) too large for the stream pool ({n_stream_c}): {dropped_k}")
    if not usable_k_values:
        raise RuntimeError(
            f"No k value in {k_values} leaves room in a stream pool of {n_stream_c} vectors "
            "(k*10 < pool size required). Raise num_vectors or lower k_values."
        )
    k_values = usable_k_values

    # K-fold stream-starting-offset sweep over the shared pool, wrapping
    # around the pool's end -- the dynamic evaluation's fold offsets.
    fold_stride = n_stream_pool // k_folds
    offsets = [f * fold_stride for f in range(k_folds)]

    logger.info(f"{rep_dataset}: calibrating tau_mid against the stream pool...")
    tau_mid = calculate_nn_distance_threshold(stream_pool[:n_stream_ab], duplicate_percentage)
    logger.info(f"tau_mid={tau_mid:.4f}")

    logger.info(f"{rep_dataset}: building fixed streaming negative set from the held-out 20%...")
    negative_vectors = _build_streaming_negative_set(
        rep_dataset, stream_pool, negative_candidate_pool, tau_mid, n_negative,
    )

    checkpoints_ab = sorted(set(
        int(round(x)) for x in np.logspace(np.log10(k_fixed + 1), np.log10(n_stream_ab), num=12)
    ))
    checkpoints_cd = sorted(set(
        int(round(x)) for x in np.linspace(n_stream_c // 20, n_stream_c, num=20)
    ))

    def _rotated_source(fold_offset: int, length: int) -> ArrayVectorSource:
        rotated = np.roll(stream_pool, -fold_offset, axis=0)[:length]
        return ArrayVectorSource(dataset_name=rep_dataset, start=0, stop=length, array=rotated)

    # Each row is self-describing (method/sweep/metric/fold_offset), so no
    # positional reconstruction is needed even though Runner's parallel mode
    # returns results in completion, not submission, order. output_csv writes
    # each job's rows as soon as it finishes, so a mid-sweep crash doesn't
    # lose completed work.

    # --- Runner A: panels (a)/(b), position sweep at fixed k_fixed ---
    runner_a = Runner(num_workers=MAX_WORKERS, output_csv=output_csv, truncate=True)
    for method in methods:
        for fold_offset in offsets:
            runner_a.submit(
                'stream_rebuild',
                dim=dim, dataset=rep_key, k=k_fixed, seed=seed, tau_mid=tau_mid, policy="streaming", T=None,
                insertion_level=method, vectors_stream=_rotated_source(fold_offset, n_stream_ab),
                vectors_negative=negative_vectors,
                checkpoints=checkpoints_ab, sweep="position", fold_offset=fold_offset,
            )
    for fold_offset in offsets:
        runner_a.submit(
            'stream_rebuild',
            dim=dim, dataset=rep_key, k=k_fixed, seed=seed, tau_mid=tau_mid, policy="oracle", T=None,
            vectors_stream=_rotated_source(fold_offset, n_stream_ab), vectors_negative=negative_vectors,
            checkpoints=checkpoints_ab, sweep="position", fold_offset=fold_offset,
        )
    rows_ab = runner_a.run()
    logger.info("Streaming position-sweep (a/b) evaluation completed.")
    df_ab = pd.DataFrame([row for job_rows in rows_ab for row in job_rows])

    # --- Runner B: panels (c)/(d), k sweep over the shared stream pool ---
    runner_b = Runner(num_workers=MAX_WORKERS, output_csv=output_csv, truncate=False)
    for method in methods:
        for k in k_values:
            for fold_offset in offsets:
                runner_b.submit(
                    'stream_rebuild',
                    dim=dim, dataset=rep_key, k=k, seed=seed, tau_mid=tau_mid, policy="streaming", T=None,
                    insertion_level=method, vectors_stream=_rotated_source(fold_offset, n_stream_c),
                    vectors_negative=negative_vectors,
                    checkpoints=checkpoints_cd, sweep="k_sweep", fold_offset=fold_offset,
                )
    rows_cd = runner_b.run()
    logger.info("Streaming k-sweep (c/d) evaluation completed.")
    df_cd = pd.DataFrame([row for job_rows in rows_cd for row in job_rows])

    df = pd.concat([df_ab, df_cd], ignore_index=True)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info(f"Wrote {output_csv} (already written incrementally during the run; this is the final canonical pass)")

    return df, k_fixed


def plot_streaming(df: pd.DataFrame, k_fixed: int) -> None:
    generate_streaming_figures(df, k_fixed=k_fixed)


def generate_dynamic() -> list:
    num_vectors              = STREAM_SIZE
    dim                      = 384
    min_percentage_duplicates = 1
    max_percentage_duplicates = 9
    count_steps              = 10
    count_memory_steps       = 10
    percentage_construct     = 25

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
        actual_results = compute_dynamic_actual_results(vectors, distance_threshold)[num_construct:]
        for memory_budget in memory_budgets:
            for filter_cls in dynamic_filters:
                runner.submit(
                    'dynamic',
                    filter_cls=filter_cls,
                    dim=dim, distance_threshold=distance_threshold,
                    memory_budget=memory_budget,
                    vectors_construct=construct_source, vectors_query=query_source,
                    actual_results=actual_results, percentage=percentage,
                )

    results = runner.run()
    logger.info("Dynamic evaluation completed.")
    return results


def generate_and_plot_all() -> None:
    """Run each generate_/plot_ pair end-to-end. Each pair is independently
    try/except-guarded so one evaluation failing (or raising) doesn't take
    down the ones after it -- comment out a pair to skip it for standalone
    testing."""
    try:
        plot_quantization(generate_quantization())
    except Exception:
        logger.exception("An error occurred")
    try:
        plot_scaling(*generate_scaling())
    except Exception:
        logger.exception("An error occurred")
    try:
        plot_streaming(*generate_streaming())
    except Exception:
        logger.exception("An error occurred")
    try:
        plot_equal_memory(generate_equal_memory())
    except Exception:
        logger.exception("An error occurred")
    try:
        plot_system_tau(generate_system_tau())
    except Exception:
        logger.exception("An error occurred")

        


def plot_all() -> None:
    """
    Re-render every figure from data a prior generate_and_plot_all() run
    already persisted to disk, without repeating the (expensive) evaluation
    sweeps -- handy for iterating on plot styling alone. Reads back
    results/{quantization,query_performance,scaling,streaming,system_tau}.csv,
    re-deriving whatever generate-time metadata (k, k_fixed,
    T_representative) each plot_ needs directly from the loaded data instead
    of threading it through from the generate_ call. Requires at least one
    prior generate_and_plot_all() (or equivalent generate_X + plot_X) run to
    have produced those files.
    """
    try:
        df = pd.read_csv("results/system_tau.csv")
        plot_system_tau(df)
    except Exception:
        logger.exception("An error occurred")

    try:
        df = pd.read_csv("results/quantization.csv")
        plot_quantization(df)
    except Exception:
        logger.exception("An error occurred")

    try:
        df = pd.read_csv("results/query_performance.csv")
        plot_query_performance(df)
    except Exception:
        logger.exception("An error occurred")

    try:
        df_query = pd.read_csv("results/scaling_query.csv")
        df_insert = pd.read_csv("results/scaling_insert.csv")
        df_construction = pd.read_csv("results/scaling_construction.csv")
        # plot_scaling(df_query, df_insert, df_construction)
    except Exception:
        logger.exception("An error occurred")

    try:
        df = pd.read_csv("results/streaming.csv")
        k_fixed = int(df.loc[df["sweep"] == "position", "k"].iloc[0])
        # plot_streaming(df, k_fixed=k_fixed)
    except Exception:
        logger.exception("An error occurred")


if __name__ == "__main__":
    generate_and_plot_all()
    # plot_all()