import os

import faiss
import time
import inspect
import numpy as np
import pandas as pd

from runner import Runner, QUANTIZATION_SWEEP
from plots.plot_quantization import generate_quantization_figures
from plots.plot_query_performance import generate_query_performance_figures
from plots.plot_equal_memory import generate_equal_memory_figures
from plots.plot_scaling import generate_scaling_figures
from plots.plot_streaming import generate_streaming_figures
from plots.plot_system_tau import generate_system_tau_figures
from plots.plot_common import DATASETS, DATASET_LABELS
from typing import Tuple
from filters.ball_cover.ball_cover_filter import BallCoverFilter
from filters.dsbf_filter import DSBFFilter
from filters.covering_lsh_fallback_filter import CoveringLSHFallbackFilter
from filters.lsbf_filter import LSBFFilter
from filters.dsh_filter import DSHFilter
from filters.hnsw_filter import HNSWFilter
from filters.ivf_filter import IVFFilter
from util.dataset_paths import dataset_filenames
from util.fetch_datasets import load_embeds
from vector_source import VectorSource, ArrayVectorSource

MAX_WORKERS = 4
DATASET = 'ELI5'

# Registry name -> canonical dataset key, shared by every long-format
# evaluation (quantization, query_performance, scaling, streaming). Derived from
# plot_common so the dataset trio has one source of truth.
DATASET_KEYS = {DATASET_LABELS[d]: d for d in DATASETS}

dynamic_filters = [BallCoverFilter, DSBFFilter, CoveringLSHFallbackFilter, LSBFFilter, DSHFilter, HNSWFilter, IVFFilter]
static_filters  = []

# generate_equal_memory()'s own comparison set: BallCoverFilter (the paper's
# own method) against only the two FAISS-based ANN baselines. The other four
# dynamic_filters entries (DSBFFilter, CoveringLSHFallbackFilter, LSBFFilter,
# DSHFilter) degenerate to predicting ~100% of queries as duplicates at every
# memory budget on realistic construct-set sizes (their hash-code alphabets /
# grid cell width saturate regardless of budget -- see project memory), which
# makes them uninformative for a budget-sensitivity comparison specifically;
# they stay in dynamic_filters for the other evaluations.
equal_memory_filters = [BallCoverFilter, HNSWFilter, IVFFilter]

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

    Unlike calculate_distance_threshold (which percentiles *random pairwise*
    distances), this ties the threshold directly to the nearest-neighbour
    test compute_static_actual_results actually performs. That distinction
    matters once the construct set is large: with tens of thousands of
    candidate neighbours, a point's *nearest* neighbour is typically far
    closer than a *random* pair, so calculate_distance_threshold's percentile
    silently becomes far more permissive than its nominal `percentage`
    suggests -- at construct-set sizes in the tens of thousands this pushed
    the "duplicate" rate to ~100%, collapsing both FPR and FNR to degenerate
    0/0 default values (no actual negatives -> FPR defaults to 0; predicted
    matching the near-universal actual positives -> FNR ~0 too) regardless of
    memory budget. Used by generate_equal_memory(), whose construct set is
    large enough for this gap to matter.
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
    num_vectors              = 100_000
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
        print(f"Processing {percentage:.1f}% duplicates...")
        print("Calculating distance threshold...")
        distance_threshold = calculate_distance_threshold(vectors, percentage)
        print("Calculating actual results...")
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
    print("Static evaluation completed.")
    return results


def generate_equal_memory() -> list:
    """
    Equal-memory comparison: BallCoverFilter (the paper's own method) against
    equal_memory_filters' two FAISS-based ANN baselines, HNSWFilter and
    IVFFilter (see equal_memory_filters' own comment for why the other four
    dynamic_filters baselines are excluded here specifically).
    count_memory_steps evenly spaced budgets are swept from
    max_memory_budget/count_memory_steps up to max_memory_budget, where
    max_memory_budget is scaled off the construct set itself (num_construct *
    bytes for one float32 sphere per point) -- not
    calculate_max_memory_budget_bytes, which is tuned for
    generate_static()/generate_dynamic()'s much smaller num_vectors and
    would cap out at a couple MB here, far below the point past which more
    memory stops buying additional accuracy for a covering-based method.

    For each budget, every filter's best available "valid configuration" is
    chosen on a calibration workload disjoint from both the construct set and
    the final query set (see runner.py::_build_calibrated_filter --
    IVFFilter's nprobe is one such dial, registered in
    runner.py::_POST_CONSTRUCTION_CONFIG_DIALS, so it's picked per-budget on
    the calibration set the same way; BallCoverFilter and HNSWFilter have no
    registered dial and just use their single memory-budget-derived default).
    FPR and FNR are then measured on the held-out query set, construction
    latency is timed during calibration (see runner.py::_build_calibrated_filter),
    and query throughput is measured on the held-out query set; all four are
    plotted one series per budget.
    """
    num_vectors             = 100_000
    dim                     = 384
    percentage_construct    = 20
    percentage_calibration  = 10
    duplicate_percentage    = 5.0
    count_memory_steps        = 10

    dataset = DATASET
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

    print("Calculating distance threshold...")
    # Nearest-neighbour-based, not calculate_distance_threshold's random-pairwise
    # percentile: with a 40%-of-100k = 40k-vector construct set, a pairwise
    # percentile is far more permissive than its nominal percentage once
    # tested against "nearest of 40k candidates" (see calculate_nn_distance_threshold's
    # docstring) -- that mismatch was pushing the actual duplicate rate to ~100%,
    # collapsing both FPR and FNR to degenerate 0/0 values regardless of budget.
    distance_threshold = calculate_nn_distance_threshold(vectors_construct, duplicate_percentage)
    print("Calculating calibration/query ground truth...")
    actual_calibration = compute_static_actual_results(vectors_construct, vectors_calibration, distance_threshold)
    actual_query       = compute_static_actual_results(vectors_construct, vectors_query, distance_threshold)

    # Scaled off the construct set itself, not calculate_max_memory_budget_bytes:
    # that helper is tuned for generate_static()/generate_dynamic()'s much
    # smaller num_vectors and produces a ceiling of a couple MB here -- far
    # below the budget at which a covering-ball method can give every
    # construct point its own sphere (num_construct * bytes_per_sphere,
    # mirroring BallCoverFilter._calculate_max_spheres's own formula), which
    # is the point past which more memory stops buying additional accuracy for
    # any covering-based method. Below that point every filter is forced into
    # heavy compression and FPR stays pinned regardless of budget; sweeping up
    # to it is what makes "equal memory" actually vary with the budget.
    bytes_per_sphere_at_1to1 = (dim + 1) * 4 * 4.0
    max_memory_budget = int(num_construct * bytes_per_sphere_at_1to1)
    memory_budgets = np.linspace(0, max_memory_budget, count_memory_steps + 1, dtype=int)[1:]

    runner = Runner(num_workers=MAX_WORKERS)
    for memory_budget in memory_budgets:
        print(f"Submitting jobs for memory_budget={memory_budget} bytes...")
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
    print("Equal-memory evaluation completed.")

    results_df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    results_df.to_csv("results/equal_memory.csv", index=False)
    print("Saved results/equal_memory.csv")

    return results


def plot_equal_memory(results: list) -> None:
    generate_equal_memory_figures(df=pd.DataFrame(results))


def generate_system_tau() -> Tuple[pd.DataFrame, int]:
    """
    Measure how much a semantic filter reduces accesses to a flat L2 index
    that explicitly stores all construct vectors: fixed ball budget k, one
    BallCoverFilter cover built per (dataset, fold), then queried at every
    tau in a sweep (system_tau_worker). tau_values is one fixed,
    dataset-independent set of absolute L2 distances shared across all
    datasets, so redundancy_rate is a reported *outcome* of a given tau
    rather than a quantity tau is solved for -- see plot_system_tau's
    redundancy-rate-vs-tau panel.

    use_kfold selects between two construct/query splitting protocols:
      - True (default): k-fold cross-validation. Each dataset is cut into
        k_folds contiguous, equal-sized intervals, and every fold in turn
        plays the role a fixed prefix used to play -- that fold's interval
        becomes the construct set (the filter/index reference set), and
        every other point in the dataset becomes the query set. Each fold
        is submitted with seed=fold, so the existing per-seed aggregation
        in plot_system_tau.py (mean_std) turns into a per-fold mean +/- std
        across k_folds independent construct/query splits instead of a
        single point estimate.
      - False: a single run per dataset, with construct_size set directly
        instead of derived from k_folds. The first construct_size vectors
        become the construct set and every remaining vector becomes the
        query set (seed=0). There's only one split, so plot_system_tau's
        per-fold std collapses to 0 (its error bars are already skipped
        when std == 0 -- see _line_panel).
    Either way, the query set generally isn't contiguous on disk once its
    held-out interval sits in the middle of the dataset, so it's shipped to
    the worker as an ArrayVectorSource (materialized in memory) rather than
    a VectorSource (lazy, contiguous-range-only); the construct set stays a
    plain VectorSource since it's always a single contiguous interval.

    Writes results/system_tau.csv. Returns the long-format df plus k, since
    plot_system_tau() needs it for the per-panel annotation and there's
    nothing to derive it from -- k is fixed by construction, not swept.
    """
    num_vectors           = 100000
    dim                   = 384
    use_kfold             = False
    k_folds               = 5
    construct_percentage  = 0.95
    k_fraction              = 0.1
    tau_values              = [float(t) for t in np.linspace(0.01, 0.1, 10)]
    output_csv              = "results/system_tau.csv"

    datasets = [d for d in DATASET_KEYS if os.path.exists(dataset_filenames[d])]
    missing = [d for d in DATASET_KEYS if d not in datasets]
    if missing:
        print(f"Skipping dataset(s) not found on disk: {', '.join(missing)}")
    if set(datasets) != set(DATASET_KEYS):
        raise RuntimeError(
            f"generate_system_tau needs all {len(DATASET_KEYS)} datasets "
            f"({', '.join(DATASET_KEYS)}) on disk; only found: {', '.join(datasets) or 'none'}"
        )

    construct_size = int(num_vectors * construct_percentage)
    construct_sizes = {d: (num_vectors // k_folds if use_kfold else construct_size) for d in datasets}
    min_construct = min(construct_sizes.values())
    k = round(k_fraction * min_construct)
    print(f"Scaling k to {k} ({k_fraction:.0%} of the smallest construct set, {min_construct} vectors)")
    print(f"Fixed tau sweep (shared across datasets): {tau_values}")
    print(f"K-fold cross-validation: {k_folds} folds" if use_kfold else f"Single run: construct_size={construct_size}")

    # tau is fixed up front (not solved for), so the construction-tightness
    # knob is a single global median instead of a per-dataset one.
    construct_tau = float(np.median(tau_values))

    runner = Runner(num_workers=MAX_WORKERS)

    for dataset in datasets:
        vectors = load_embeds(dataset, num_vectors)[0]
        n = len(vectors)  # a dataset may hold fewer than num_vectors

        if use_kfold:
            fold_edges = [int(e) for e in np.linspace(0, n, k_folds + 1)]
            splits = [(fold_edges[fold], fold_edges[fold + 1], fold) for fold in range(k_folds)]
        else:
            splits = [(0, min(construct_size, n), 0)]

        for start, stop, seed in splits:
            vectors_construct = vectors[start:stop]
            vectors_query = np.concatenate([vectors[:start], vectors[stop:]])

            construct_source = VectorSource(dataset, start, stop)
            query_source = ArrayVectorSource(dataset_name=dataset, start=0, stop=len(vectors_query),
                                             array=vectors_query)

            split_label = f"fold {seed}/{k_folds}" if use_kfold else "single run"
            print(f"{dataset} {split_label}: computing ground truth at {len(tau_values)} fixed tau values...")
            construct_arr = np.ascontiguousarray(vectors_construct, dtype=np.float32)
            query_arr = np.ascontiguousarray(vectors_query, dtype=np.float32)
            index = faiss.IndexFlatL2(dim)
            index.add(construct_arr)
            nn_sq_dist, _ = index.search(query_arr, 1)
            nn_dist = np.sqrt(np.clip(nn_sq_dist[:, 0], 0, None))
            actual_results_by_tau = {t: (nn_dist <= t).tolist() for t in tau_values}

            runner.submit(
                'system_tau',
                dim=dim, dataset=DATASET_KEYS[dataset],
                k=k, tau_values=tau_values, seed=seed, construct_tau=construct_tau,
                vectors_construct=construct_source, vectors_query=query_source,
                actual_results_by_tau=actual_results_by_tau,
            )

    job_results = runner.run()
    print(f"System (fixed tau-sweep, {'K-fold' if use_kfold else 'single run'}) evaluation completed.")

    df = pd.DataFrame([row for job_rows in job_results for row in job_rows])

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Wrote {output_csv}")

    return df, k


def plot_system_tau(df: pd.DataFrame, k: int) -> None:
    generate_system_tau_figures(df, k_fixed=k)


def generate_quantization() -> pd.DataFrame:
    """
    Generate the per-quantize-variant construction data for BallCoverFilter
    (QUANTIZATION_SWEEP: the raw/unquantized cover plus every scalar/
    product quantizer) on a single dataset (DATASET), and write it to
    results/quantization.csv. Every candidate in the sweep shares one memory
    budget per k (derived from k against the unquantized cover's byte-per-
    ball size), so a more compact quantizer's realized ball count can pack
    more balls into that same budget -- which is what lets plot_quantization.py
    use each row's recorded memory usage as the x-axis with one line per
    quantize method, instead of faceting bars by dataset. That tracking is
    imperfect for the most compact quantizers (see quantization_worker's
    docstring): a quantizer compact enough to want more balls than the
    construct set actually has saturates at n_construct balls short of the
    nominal budget, so its memory-usage line goes flat across the k sweep
    instead of spanning it -- real information about that quantizer being
    construct-set-bound rather than a plotting bug.

    k and tau are both derived from num_vectors/percentage_construct rather
    than hardcoded absolute values, so editing those two knobs rescales
    the whole sweep instead of silently clamping/dropping stale constants:
      - k_values = k_fractions * the construct set size, guaranteed below it
        since every fraction is < 1 -- k >= the construct-set size makes
        BallCoverFilter fall back to one zero-radius ball per point
        (log10_J = -inf, unplottable).
      - tau_values = duplicate_percentages-th percentile of the dataset's own
        nearest-neighbour distance (calculate_nn_distance_threshold) -- an
        absolute L2 distance means something different at every
        construct-set scale, so the percentile is what stays meaningful as
        num_vectors/percentage_construct change, not a raw float.
    """
    num_vectors            = 50_000
    dim                    = 384
    percentage_construct   = 20
    k_fractions            = [0.03, 0.06, 0.125, 0.25]
    duplicate_percentages  = [20]
    seeds                  = [0]
    output_csv             = "results/quantization.csv"

    dataset = DATASET
    vectors = load_embeds(dataset, num_vectors)[0]
    num_construct = len(vectors) * percentage_construct // 100
    vectors_construct = vectors[:num_construct]
    vectors_query = vectors[num_construct:]

    construct_source = VectorSource(dataset, 0, num_construct)
    query_source = VectorSource(dataset, num_construct, len(vectors))

    k_values = sorted({max(1, round(num_construct * f)) for f in k_fractions})
    k_values = [k for k in k_values if k < num_construct]
    if not k_values:
        raise RuntimeError(
            f"No k_fraction in {k_fractions} produced a k below the construct set "
            f"({num_construct} vectors). Raise num_vectors/percentage_construct or lower k_fractions."
        )
    print(f"Scaling k_values to {k_values} ({num_construct} vectors in the construct set)")

    tau_values = [calculate_nn_distance_threshold(vectors_construct, p) for p in duplicate_percentages]
    print(f"Scaling tau_values to {tau_values} ({duplicate_percentages}-th percentile NN distance)")

    # Kept for interface parity with quantization_worker's signature; unused there
    # now that BallCoverFilter's Gonzalez construction has no distance-
    # threshold knob to feed (only quantize varies the cover, not tau).
    construct_tau = float(np.median(tau_values))

    print(f"Computing ground truth for tau in {tau_values}...")
    actual_results_by_tau = {
        tau: compute_static_actual_results(vectors_construct, vectors_query, tau)
        for tau in tau_values
    }

    runner = Runner(num_workers=MAX_WORKERS)
    for k in k_values:
        for seed in seeds:
            for quantize in QUANTIZATION_SWEEP:
                runner.submit(
                    'quantization',
                    dim=dim, dataset=dataset,
                    k=k, quantize=quantize, tau_values=tau_values, seed=seed, construct_tau=construct_tau,
                    vectors_construct=construct_source, vectors_query=query_source,
                    actual_results_by_tau=actual_results_by_tau,
                )

    job_results = runner.run()
    print("Quantization evaluation completed.")

    df = pd.DataFrame([row for job_rows in job_results for row in job_rows])

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Wrote {output_csv}")

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
    num_vectors           = 100_000
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
        print(f"Skipping dataset(s) not found on disk: {', '.join(missing)}")
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
                    dim=dim, dataset=DATASET_KEYS[dataset],
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

    return df


def plot_query_performance(df: pd.DataFrame) -> None:
    generate_query_performance_figures(df)


# These panels compare across batch_size/stream_size, not across datasets,
# so sweeping all 3 datasets would only add runtime, not a comparison axis --
# one representative dataset (same single-dataset pattern as
# _STREAMING_REPRESENTATIVE_DATASET) is enough.
_SCALING_DATASET = "ELI5"


def generate_scaling() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Measures three BallCoverFilter throughput surfaces, all plotted against
    filter size k on the x-axis:
      - query throughput: k x batch_size grid (batch_size -> one line each,
        6 lines), querying a fixed-size stream held constant across the
        whole batch_size sweep -- only k and batch_size vary here.
      - insertion throughput: the same k x batch_size grid/lines, timing
        add() instead of query().
      - construction throughput: k x n grid (n, the stream/indexed-set
        size -> one line each, 6 lines) -- this is the one panel where the
        indexed-set size itself varies instead of being held fixed.
    Runs on a single dataset (_SCALING_DATASET) -- see its comment above.

    Edit `stream_size` alone to rescale the construction sweep and the fixed
    stream query/insertion throughput measure against: stream_sizes (halved
    n_steps-1 times, the 6 construction lines) derives from it, the same
    single-knob pattern used elsewhere in this file. k_values/batch_sizes
    are the two other swept axes and stay independent knobs (they're not
    size-of-data parameters); dim/tau_mid/seeds are independent too.
    """
    stream_size      = 90_000
    dim              = 384
    n_steps          = 6
    k_values         = [16, 32, 64, 128, 256, 512]
    batch_sizes      = [1, 4, 16, 64, 256, 1024]
    tau_mid          = 0.35
    seeds            = [0, 1, 2]
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

    # k_values/batch_sizes must each stay below stream_size (max_stream):
    # k needs room to be a strictly smaller filter-size budget than the
    # indexed set, and batch_size chunks a stream sized off the same
    # stream_size, so shrinking stream_size can put either sweep list out of
    # range -- drop whatever no longer fits rather than raising, so changing
    # stream_size alone never requires hand-editing k_values/batch_sizes too.
    def _below_stream(values: list[int], name: str) -> list[int]:
        usable = sorted(v for v in values if v < max_stream)
        dropped = sorted(set(values) - set(usable))
        if dropped:
            print(f"Dropping {name} value(s) >= stream_size ({max_stream}): {dropped}")
        if not usable:
            raise RuntimeError(
                f"No {name} value in {values} is below stream_size ({max_stream}). "
                f"Raise stream_size or lower {name}."
            )
        return usable

    k_values = _below_stream(k_values, "k_values")
    batch_sizes = _below_stream(batch_sizes, "batch_sizes")

    # Fixed construct set (the largest stream size) + a disjoint holdout,
    # reused as both the query and insert stream for every (k, batch_size)
    # job below -- only k/batch_size vary in those two sweeps.
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

    fixed_construct_source = VectorSource(dataset_key, 0, max_stream)
    holdout_source = VectorSource(dataset_key, max_stream, max_stream + holdout_size)

    runner = Runner(num_workers=MAX_WORKERS)

    for k in k_values:
        for batch_size in batch_sizes:
            for seed in seeds:
                runner.submit(
                    'scaling_query',
                    dim=dim, dataset=_SCALING_DATASET, k=k, batch_size=batch_size, seed=seed,
                    tau_mid=tau_mid,
                    vectors_construct=fixed_construct_source, vectors_query=holdout_source,
                    n_query_vectors=n_query_vectors,
                )
                runner.submit(
                    'scaling_insert',
                    dim=dim, dataset=_SCALING_DATASET, k=k, batch_size=batch_size, seed=seed,
                    vectors_construct=fixed_construct_source, vectors_insert=holdout_source,
                    n_insert_vectors=n_insert_vectors,
                )

    for k in k_values:
        for n in stream_sizes:
            if k >= n:
                continue
            construct_source_n = VectorSource(dataset_key, 0, n)
            for seed in seeds:
                runner.submit(
                    'scaling_construction',
                    dim=dim, dataset=_SCALING_DATASET, k=k, n=n, seed=seed,
                    vectors_construct=construct_source_n,
                )

    job_results = runner.run()
    print("Scaling evaluation completed.")

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
        print(f"Wrote {path}")

    return df_query, df_insert, df_construction


def plot_scaling(df_query: pd.DataFrame, df_insert: pd.DataFrame, df_construction: pd.DataFrame) -> None:
    generate_scaling_figures(df_query, df_insert, df_construction)


# Panels (b)/(c)/(d) collapse to one representative dataset.
_STREAMING_REPRESENTATIVE_DATASET = "ELI5"


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
    Compare BallCoverFilter's 4 insertion methods (insertion_level:
    "nearest", "least_inflation", "merge", "doubling" -- see
    BallCoverFilter.insert()'s dispatch) via stream_rebuild_worker's
    policy="streaming", each sharing the identical bootstrap-then-diverge
    setup (see that worker's docstring). Produces the long-format data
    behind the streaming figure's four panels:
      (a) FPR vs. stream position (fixed k_fixed), + a dashed static-rebuild
          reference (stream_rebuild_worker policy="oracle");
      (b) threshold-expanded objective inflation vs. stream position (same
          x-axis/checkpoints as (a));
      (c) final FPR vs. ball budget k, one line per method;
      (d) Pareto: mean FPR (2nd half of stream) vs. amortized cost/insertion,
          one connected curve per method per k, + the same static-rebuild
          reference as a dashed curve (the expensive-but-best baseline every
          insertion method trades cost for).

    Every dataset on disk caps out around ~100k vectors, so the spec's
    "stream position from k to >= 10^3 * k" is only reachable at a small
    k_fixed -- see k_fixed's derivation below. Panel (c)'s k-sweep goes up to
    k=1024 but shares one fixed-length stream (n_stream_c = stream_size)
    across every k, not a literal 10^3*k-scaled length -- documented here
    rather than silently truncated; most consequential at k=1024 (~29*k at
    the default stream_size, not 1000*k).

    "Multiple starting offsets" (the paper's K-fold variance protocol) is
    operationalized as overlapping offset windows into one shared
    representative-dataset pool (true disjoint folds don't scale to a
    useful count at this vector budget) -- a sliding-window robustness
    check, not strict cross-validation.

    FNR=0 is asserted per checkpoint inside stream_rebuild_worker (raises,
    not recorded) for every method -- the covering invariant is a
    correctness requirement, not a plotted quantity.

    Edit `stream_size` alone to rescale the whole evaluation -- every other
    size-related knob below derives from it, the same single-knob pattern
    used by generate_scaling():
      - n_stream_c: stream_size itself (the k-sweep's stream length).
      - k_fixed: stream_size // 1000 (floored at 16), so the position-sweep
        stream n_stream_ab = k_fixed * 1000 stays proportioned the same way
        as the original 32-vs-32,000 relationship.
      - stride_ab/stride_cd: a small fraction of their respective stream
        lengths, so the K-fold offset windows stay proportioned regardless
        of scale instead of needing separate tuning.
      - k_values: dropped (with a printed reason) if too large relative to
        stream_size for the k-sweep to be meaningful at that k, mirroring
        generate_scaling()'s k_values/batch_sizes filtering.
    dim/methods/duplicate_percentage/n_negative/seed aren't size-related, so
    they stay independent knobs.

    Writes results/streaming.csv (panels a-d). Returns the long-format df
    plus k_fixed, which plot_streaming() needs and can't derive from the
    df alone.
    """
    stream_size = 30_000
    dim = 384
    k_values = [16, 32, 64, 128, 256, 512, 1024]
    methods = ["nearest", "least_inflation", "merge", "doubling"]
    n_negative, negative_candidate_multiplier = 500, 10
    duplicate_percentage = 2.0  # calibrates tau_mid; see calculate_nn_distance_threshold
    seed = 0  # fixed; fold_offset supplies the stream-order variance instead
    output_csv = "results/streaming.csv"

    k_fixed = max(16, stream_size // 1000)
    n_stream_ab, F_ab = k_fixed * 1000, 4
    stride_ab = max(1, n_stream_ab // 20)
    n_stream_c, F_cd = stream_size, 4
    stride_cd = max(1, n_stream_c // 30)

    # k must leave enough room in the k-sweep's shared stream to be a
    # meaningful sweep point (not just barely fit) -- dropping oversized k
    # values here means stream_size alone can shrink the whole evaluation
    # without also having to hand-edit k_values to match.
    usable_k_values = [k for k in k_values if k * 10 < n_stream_c]
    dropped_k = sorted(set(k_values) - set(usable_k_values))
    if dropped_k:
        print(f"Dropping k value(s) too large for stream_size={stream_size}: {dropped_k}")
    if not usable_k_values:
        raise RuntimeError(
            f"No k value in {k_values} leaves room in a stream_size={stream_size} stream "
            "(k*10 < stream_size required). Raise stream_size or lower k_values."
        )
    k_values = usable_k_values

    datasets = [d for d in DATASET_KEYS if os.path.exists(dataset_filenames[d])]
    missing = [d for d in DATASET_KEYS if d not in datasets]
    if missing:
        print(f"Skipping dataset(s) not found on disk: {', '.join(missing)}")
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

    pool_ab_len = n_stream_ab + (F_ab - 1) * stride_ab
    pool_cd_len = n_stream_c + (F_cd - 1) * stride_cd
    negative_candidates_len = n_negative * negative_candidate_multiplier
    cd_start = pool_ab_len
    neg_start = pool_ab_len + pool_cd_len
    total_needed = neg_start + negative_candidates_len

    rep_n_available = len(load_embeds(rep_dataset, total_needed)[0])
    if rep_n_available < total_needed:
        raise RuntimeError(
            f"{rep_dataset}: only {rep_n_available} vectors on disk, need {total_needed} "
            f"(pool_ab={pool_ab_len} + pool_cd={pool_cd_len} + negative_candidates="
            f"{negative_candidates_len}). Lower stream_size (or F_ab/F_cd/n_negative)."
        )

    offsets_ab = [f * stride_ab for f in range(F_ab)]
    offsets_cd = [cd_start + f * stride_cd for f in range(F_cd)]

    print(f"{rep_dataset}: calibrating tau_mid against the a/b stream...")
    calib_vectors = load_embeds(rep_dataset, n_stream_ab)[0][:n_stream_ab]
    tau_mid = calculate_nn_distance_threshold(calib_vectors, duplicate_percentage)
    print(f"tau_mid={tau_mid:.4f}")

    print(f"{rep_dataset}: building fixed streaming negative set...")
    stream_union = load_embeds(rep_dataset, neg_start)[0]
    candidate_pool = load_embeds(rep_dataset, neg_start + negative_candidates_len)[0][neg_start:]
    negative_vectors = _build_streaming_negative_set(rep_dataset, stream_union, candidate_pool, tau_mid, n_negative)

    checkpoints_ab = sorted(set(
        int(round(x)) for x in np.logspace(np.log10(k_fixed + 1), np.log10(n_stream_ab), num=12)
    ))
    checkpoints_cd = sorted(set(
        int(round(x)) for x in np.linspace(n_stream_c // 20, n_stream_c, num=20)
    ))

    # Every row each worker returns is now self-describing (method/sweep/
    # metric/fold_offset all included directly -- see stream_rebuild_worker's
    # docstring), so results don't need positional reconstruction here --
    # which matters because Runner's parallel mode returns results in
    # *completion* order, not submission order. output_csv makes each Runner
    # write every job's rows to disk as soon as that job finishes, so a
    # mid-sweep crash still leaves whatever completed so far on disk instead
    # of losing the whole run.

    # --- Runner A: panels (a)/(b), position sweep at fixed k_fixed ---
    runner_a = Runner(num_workers=MAX_WORKERS, output_csv=output_csv, truncate=True)
    for method in methods:
        for fold_offset in offsets_ab:
            src = VectorSource(rep_dataset, fold_offset, fold_offset + n_stream_ab)
            runner_a.submit(
                'stream_rebuild',
                dim=dim, dataset=rep_key, k=k_fixed, seed=seed, tau_mid=tau_mid, policy="streaming", T=None,
                insertion_level=method, vectors_stream=src, vectors_negative=negative_vectors,
                checkpoints=checkpoints_ab, sweep="position", fold_offset=fold_offset,
            )
    for fold_offset in offsets_ab:
        src = VectorSource(rep_dataset, fold_offset, fold_offset + n_stream_ab)
        runner_a.submit(
            'stream_rebuild',
            dim=dim, dataset=rep_key, k=k_fixed, seed=seed, tau_mid=tau_mid, policy="oracle", T=None,
            vectors_stream=src, vectors_negative=negative_vectors, checkpoints=checkpoints_ab,
            sweep="position", fold_offset=fold_offset,
        )
    rows_ab = runner_a.run()
    print("Streaming position-sweep (a/b) evaluation completed.")
    df_ab = pd.DataFrame([row for job_rows in rows_ab for row in job_rows])

    # --- Runner B: panels (c)/(d), k sweep over a shared stream length ---
    runner_b = Runner(num_workers=MAX_WORKERS, output_csv=output_csv, truncate=False)
    for method in methods:
        for k in k_values:
            for fold_offset in offsets_cd:
                src = VectorSource(rep_dataset, fold_offset, fold_offset + n_stream_c)
                runner_b.submit(
                    'stream_rebuild',
                    dim=dim, dataset=rep_key, k=k, seed=seed, tau_mid=tau_mid, policy="streaming", T=None,
                    insertion_level=method, vectors_stream=src, vectors_negative=negative_vectors,
                    checkpoints=checkpoints_cd, sweep="k_sweep", fold_offset=fold_offset,
                )
    rows_cd = runner_b.run()
    print("Streaming k-sweep (c/d) evaluation completed.")
    df_cd = pd.DataFrame([row for job_rows in rows_cd for row in job_rows])

    df = pd.concat([df_ab, df_cd], ignore_index=True)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Wrote {output_csv} (already written incrementally during the run; this is the final canonical pass)")

    return df, k_fixed


def plot_streaming(df: pd.DataFrame, k_fixed: int) -> None:
    generate_streaming_figures(df, k_fixed=k_fixed)


def generate_dynamic() -> list:
    num_vectors              = 1_000
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
    print("Dynamic evaluation completed.")
    return results


def generate_and_plot_all() -> None:
    """
    Run every active evaluation end-to-end (generate, then plot), the same
    5 evaluations __main__ used to run inline, plus system_tau.
    static/dynamic/equal_memory stay disabled -- they fail at the
    100K-vector scale this file is currently configured for (see comments
    below). The early return currently isolates system_tau for standalone
    testing; remove it to also run the other 5.
    """
    plot_quantization(generate_quantization())
    # plot_scaling(*generate_scaling())
    # plot_streaming(*generate_streaming())
    # plot_equal_memory(generate_equal_memory()) # failed on 100K
    # plot_system_tau(*generate_system_tau())
    return
    try:
        plot_quantization(generate_quantization())
    except Exception as e:
        print(f"An error occurred: {e}")
    return
    try:
        plot_query_performance(generate_query_performance())
    except Exception as e:
        print(f"An error occurred: {e}")

    try:
        plot_scaling(*generate_scaling())
    except Exception as e:
        print(f"An error occurred: {e}")

    try:
        plot_streaming(*generate_streaming())
    except Exception as e:
        print(f"An error occurred: {e}")


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
        plot_system_tau(df, k=int(df["k"].iloc[0]))
    except Exception as e:
        print(f"An error occurred: {e}")

    try:
        df = pd.read_csv("results/quantization.csv")
        plot_quantization(df)
    except Exception as e:
        print(f"An error occurred: {e}")

    try:
        df = pd.read_csv("results/query_performance.csv")
        plot_query_performance(df)
    except Exception as e:
        print(f"An error occurred: {e}")

    try:
        df_query = pd.read_csv("results/scaling_query.csv")
        df_insert = pd.read_csv("results/scaling_insert.csv")
        df_construction = pd.read_csv("results/scaling_construction.csv")
        plot_scaling(df_query, df_insert, df_construction)
    except Exception as e:
        print(f"An error occurred: {e}")

    try:
        df = pd.read_csv("results/streaming.csv")
        k_fixed = int(df.loc[df["sweep"] == "position", "k"].iloc[0])
        plot_streaming(df, k_fixed=k_fixed)
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    generate_and_plot_all()
    # plot_all()