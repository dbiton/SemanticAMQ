import json
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
from plots.legacy import plot_results, plot_throughput_results, plot_tradeoff_results, plot_equal_memory_results
from plots.plot_ablation import generate_ablation_figures
from plots.plot_query_performance import generate_query_performance_figures
from plots.plot_equal_memory import generate_equal_memory_figures
from plots.plot_scaling import generate_scaling_figures
from plots.plot_streaming import generate_streaming_figures
from plots.plot_system import generate_system_figures
from plots.plot_system_tau import generate_system_tau_figures
from plots.plot_common import DATASETS, DATASET_LABELS
from typing import Tuple, List
from filters.first_n_filter import FirstNFilter
from src.filters.ball_cover_filter import BallCoverFilter
from filters.dsbf_filter import DSBFFilter
from filters.covering_lsh_fallback_filter import CoveringLSHFallbackFilter
from filters.lsbf_filter import LSBFFilter
from filters.dsh_filter import DSHFilter
from filters.hnsw_filter import HNSWFilter
from util.dataset_paths import dataset_filenames
from util.fetch_datasets import load_embeds
from vector_source import VectorSource, ArrayVectorSource

MAX_WORKERS = 6
DATASET = 'ELI5'

# Registry name -> canonical dataset key, shared by every long-format
# evaluation (ablation, query_performance, scaling, streaming). Derived from
# plot_common so the dataset trio has one source of truth.
DATASET_KEYS = {DATASET_LABELS[d]: d for d in DATASETS}

dynamic_filters = [BallCoverFilter, DSBFFilter, CoveringLSHFallbackFilter, LSBFFilter, DSHFilter, HNSWFilter]
static_filters  = []

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


def generate_static() -> list:
    num_vectors              = 100_000
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
    return results


def plot_static(results: list) -> None:
    plot_results(results, "static")


def generate_equal_memory() -> list:
    """
    Equal-memory comparison: every registered filter (BallCoverFilter --
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
    num_vectors             = 100_000
    dim                     = 384
    percentage_construct    = 40
    percentage_calibration  = 20
    duplicate_percentage    = 5.0
    max_L                   = 256
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

    results_df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    results_df.to_csv("results/equal_memory.csv", index=False)
    print("Saved results/equal_memory.csv")

    return results


def plot_equal_memory(results: list) -> None:
    plot_equal_memory_results(results)
    generate_equal_memory_figures(df=pd.DataFrame(results))


def generate_system() -> Tuple[pd.DataFrame, int]:
    """
    Measure how much a semantic filter reduces accesses to a flat L2 index
    that explicitly stores all construct vectors, as a function of the
    threshold-driven redundancy rate of the query stream, at a fixed ball
    budget k -- not the memory-budget sweep this used to run. One
    BallCoverFilter cover is built per (dataset, seed) at that fixed k,
    then queried at every tau in a sweep (see system_tau_worker); tau is a
    pure query-time knob controlling how much of the fixed-length query
    stream counts as a ground-truth duplicate, which is the "redundancy
    rate" plot_system's figure actually plots against on its shared x-axis.

    Writes results/system.csv. Returns the long-format df plus k, since
    plot_system() needs it for the per-panel annotation and there's nothing
    to derive it from -- k is fixed by construction, not swept.
    """
    num_vectors               = 10_000
    dim                       = 384
    percentage_construct      = 20
    # k is scaled off the construct-set size rather than a fixed constant.
    # tau_values below are deliberately fine-grained (1-20th percentile of
    # single-nearest-neighbour distance), but a ball's own construction
    # radius is set by how many construct points it has to cover -- and at
    # 384 dimensions, that radius barely shrinks until points-per-ball drops
    # to nearly 1 (covering-ball volume is exponential in dimension, so an
    # 8x denser cover, e.g. k 100->800, only shaved the radius ~15%).
    # Below ~k_fraction=0.9 the base radius already exceeds the whole tau
    # sweep, so eff_r = radius + tau saturates to "covers everything" at
    # every tau and the filter can never report a negative -- verified
    # empirically on all 3 datasets, where 0.9 produces a smooth savings
    # gradient across the full redundancy sweep instead of an instant cliff.
    k_fraction                = 0.5
    min_percentage_duplicates = 1
    max_percentage_duplicates = 20
    count_steps               = 10
    seeds                     = [0]
    output_csv                = "results/system.csv"

    datasets = [d for d in DATASET_KEYS if os.path.exists(dataset_filenames[d])]
    missing = [d for d in DATASET_KEYS if d not in datasets]
    if missing:
        print(f"Skipping dataset(s) not found on disk: {', '.join(missing)}")
    if set(datasets) != set(DATASET_KEYS):
        raise RuntimeError(
            "generate_system needs all 3 datasets "
            f"({', '.join(DATASET_KEYS)}) on disk; only found: {', '.join(datasets) or 'none'}"
        )

    # A single, shared k (plot_system's figure assumes one global k across
    # all datasets) scaled off the *smallest* construct set, same clamping
    # pattern as generate_ablation's min_construct.
    construct_sizes = {d: len(load_embeds(d, num_vectors)[0]) * percentage_construct // 100 for d in datasets}
    min_construct = min(construct_sizes.values())
    k = round(k_fraction * min_construct)
    print(f"Scaling k to {k} ({k_fraction:.0%} of the smallest construct set, {min_construct} vectors)")

    runner = Runner(num_workers=MAX_WORKERS)

    for dataset in datasets:
        vectors = load_embeds(dataset, num_vectors)[0]
        n = len(vectors)  # a dataset may hold fewer than num_vectors
        num_construct = n * percentage_construct // 100
        vectors_construct = vectors[:num_construct]
        vectors_query = vectors[num_construct:]

        construct_source = VectorSource(dataset, 0, num_construct)
        query_source = VectorSource(dataset, num_construct, n)

        print(f"{dataset}: calculating tau sweep...")
        # Pick tau directly off the percentiles of the query stream's own
        # nearest-construct-neighbour distances, rather than a proxy percentile
        # (pairwise or construct-only NN -- see calculate_nn_distance_threshold's
        # docstring for why those drift from the nominal percentage). Percentile
        # is the inverse-ECDF of this exact population, so the p-th percentile
        # tau is guaranteed to land within a rounding hair of a p% redundancy
        # rate once measured -- unless the distribution has a mass spike (e.g.
        # exact-duplicate embeddings) wide enough to straddle multiple target
        # percentages, in which case they collapse onto the same tau/rate and
        # fewer than count_steps distinct points appear for that dataset.
        construct_arr = np.ascontiguousarray(vectors_construct, dtype=np.float32)
        query_arr = np.ascontiguousarray(vectors_query, dtype=np.float32)
        index = faiss.IndexFlatL2(dim)
        index.add(construct_arr)
        nn_sq_dist, _ = index.search(query_arr, 1)
        nn_dist = np.sqrt(np.clip(nn_sq_dist[:, 0], 0, None))

        redundancy_percentages = np.linspace(min_percentage_duplicates, max_percentage_duplicates, count_steps)
        tau_values = sorted({float(np.percentile(nn_dist, p)) for p in redundancy_percentages})
        print(f"{dataset}: computing ground truth at {len(tau_values)} tau values...")
        actual_results_by_tau = {t: (nn_dist <= t).tolist() for t in tau_values}
        # Sweep-independent construction-tightness knob (same role as
        # generate_ablation's construct_tau) -- tau only affects querying.
        construct_tau = float(np.median(tau_values))

        for seed in seeds:
            runner.submit(
                'system_tau',
                dim=dim, dataset=DATASET_KEYS[dataset],
                k=k, tau_values=tau_values, seed=seed, construct_tau=construct_tau,
                vectors_construct=construct_source, vectors_query=query_source,
                actual_results_by_tau=actual_results_by_tau,
            )

    job_results = runner.run()
    print("System (tau-sweep) evaluation completed.")

    df = pd.DataFrame([row for job_rows in job_results for row in job_rows])

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Wrote {output_csv}")

    return df, k


def plot_system(df: pd.DataFrame, k: int) -> None:
    generate_system_figures(df, k_fixed=k)


def generate_system_tau() -> Tuple[pd.DataFrame, int]:
    """
    Same measurement as generate_system() -- fixed ball budget k, one
    BallCoverFilter cover built per (dataset, seed), then queried at every
    tau in a sweep (system_tau_worker) -- but tau itself is the swept
    independent variable here, instead of being reverse-engineered per
    dataset from percentiles of that dataset's own nearest-construct-
    neighbour distances. tau_values is one fixed, dataset-independent set of
    absolute L2 distances shared across all 3 datasets, so redundancy_rate
    becomes a reported *outcome* of a given tau rather than the quantity tau
    is solved for -- see plot_system_tau's redundancy-rate-vs-tau panel.

    Writes results/system_tau.csv. Returns the long-format df plus k, same
    shape as generate_system().
    """
    num_vectors           = 100_000
    dim                   = 384
    percentage_construct  = 20
    # Same k-scaling rationale as generate_system (see its comment): k is a
    # fraction of the smallest construct set, not a fixed constant.
    k_fraction             = 0.5
    tau_values              = [float(t) for t in np.linspace(0.1, 0.5, 10)]
    seeds                   = [0]
    output_csv              = "results/system_tau.csv"

    datasets = [d for d in DATASET_KEYS if os.path.exists(dataset_filenames[d])]
    missing = [d for d in DATASET_KEYS if d not in datasets]
    if missing:
        print(f"Skipping dataset(s) not found on disk: {', '.join(missing)}")
    if set(datasets) != set(DATASET_KEYS):
        raise RuntimeError(
            "generate_system_tau needs all 3 datasets "
            f"({', '.join(DATASET_KEYS)}) on disk; only found: {', '.join(datasets) or 'none'}"
        )

    construct_sizes = {d: len(load_embeds(d, num_vectors)[0]) * percentage_construct // 100 for d in datasets}
    min_construct = min(construct_sizes.values())
    k = round(k_fraction * min_construct)
    print(f"Scaling k to {k} ({k_fraction:.0%} of the smallest construct set, {min_construct} vectors)")
    print(f"Fixed tau sweep (shared across datasets): {tau_values}")

    # tau is fixed up front (not solved for), so the construction-tightness
    # knob is a single global median instead of a per-dataset one.
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

        print(f"{dataset}: computing ground truth at {len(tau_values)} fixed tau values...")
        construct_arr = np.ascontiguousarray(vectors_construct, dtype=np.float32)
        query_arr = np.ascontiguousarray(vectors_query, dtype=np.float32)
        index = faiss.IndexFlatL2(dim)
        index.add(construct_arr)
        nn_sq_dist, _ = index.search(query_arr, 1)
        nn_dist = np.sqrt(np.clip(nn_sq_dist[:, 0], 0, None))
        actual_results_by_tau = {t: (nn_dist <= t).tolist() for t in tau_values}

        for seed in seeds:
            runner.submit(
                'system_tau',
                dim=dim, dataset=DATASET_KEYS[dataset],
                k=k, tau_values=tau_values, seed=seed, construct_tau=construct_tau,
                vectors_construct=construct_source, vectors_query=query_source,
                actual_results_by_tau=actual_results_by_tau,
            )

    job_results = runner.run()
    print("System (fixed tau-sweep) evaluation completed.")

    df = pd.DataFrame([row for job_rows in job_results for row in job_rows])

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Wrote {output_csv}")

    return df, k


def plot_system_tau(df: pd.DataFrame, k: int) -> None:
    generate_system_tau_figures(df, k_fixed=k)


def generate_ablation() -> Tuple[pd.DataFrame, int]:
    """
    Generate the per-stage construction-ladder data for BallCoverFilter
    (ball refine -> quantizing -> feedback) across the 3 datasets the
    ablation figure needs, and write it to results/ablation.csv. Returns the
    long-format df plus the largest k used.

    k and tau are both derived from num_vectors/percentage_construct rather
    than hardcoded absolute values, so editing those two knobs rescales
    the whole sweep instead of silently clamping/dropping stale constants:
      - k_values = k_fractions * the smallest construct set (same pattern as
        generate_system's k_fraction), guaranteed < that set's size since
        every fraction is < 1 -- k >= a construct set's size makes
        BallCoverFilter fall back to one zero-radius ball per point
        (log10_J = -inf, unplottable).
      - tau_values = duplicate_percentages-th percentile of each dataset's
        own nearest-neighbour distance (calculate_nn_distance_threshold),
        averaged across datasets into one shared float per percentage --
        every dataset's ground truth/job below needs the exact same literal
        tau (plot_ablation's shared-tau bar panels filter by equality), but
        an absolute L2 distance means something different at every
        construct-set scale, so the percentile (computed per-dataset, off
        that dataset's own distribution) is what actually stays meaningful
        as num_vectors/percentage_construct change -- not the raw float.
    """
    num_vectors            = 1_000
    dim                    = 384
    percentage_construct   = 20
    k_fractions            = [0.03, 0.06, 0.125, 0.25]
    duplicate_percentages  = [5, 10, 20]
    seeds                  = [0, 1, 2]
    feedback_fraction      = 0.10
    output_csv             = "results/ablation.csv"

    datasets = [d for d in DATASET_KEYS if os.path.exists(dataset_filenames[d])]
    missing = [d for d in DATASET_KEYS if d not in datasets]
    if missing:
        print(f"Skipping dataset(s) not found on disk: {', '.join(missing)}")
    if set(datasets) != set(DATASET_KEYS):
        raise RuntimeError(
            "generate_ablation needs all 3 datasets "
            f"({', '.join(DATASET_KEYS)}) on disk; only found: {', '.join(datasets) or 'none'}"
        )

    # Load once per dataset and reuse below (construct sizing, tau
    # calibration, and job submission all need the same vectors).
    dataset_vectors = {d: load_embeds(d, num_vectors)[0] for d in datasets}
    construct_sizes = {d: len(dataset_vectors[d]) * percentage_construct // 100 for d in datasets}
    min_construct = min(construct_sizes.values())

    k_values = sorted({max(1, round(min_construct * f)) for f in k_fractions})
    k_values = [k for k in k_values if k < min_construct]
    if not k_values:
        raise RuntimeError(
            f"No k_fraction in {k_fractions} produced a k below the smallest construct set "
            f"({min_construct} vectors, from {min(construct_sizes, key=construct_sizes.get)}). "
            "Raise num_vectors/percentage_construct or lower k_fractions."
        )
    print(f"Scaling k_values to {k_values} ({min_construct} vectors in the smallest construct set)")

    per_dataset_tau = {
        d: [calculate_nn_distance_threshold(dataset_vectors[d][:construct_sizes[d]], p) for p in duplicate_percentages]
        for d in datasets
    }
    tau_values = [float(np.mean([per_dataset_tau[d][i] for d in datasets])) for i in range(len(duplicate_percentages))]
    print(f"Scaling tau_values to {tau_values} (mean {duplicate_percentages}-th percentile NN distance across datasets)")

    # A single, sweep-independent construction tightness knob (see
    # ablation_worker's docstring for why tau can't double as this and stay
    # tau-independent for log10_J). The sweep's median is a reasonable default.
    construct_tau = float(np.median(tau_values))

    runner = Runner(num_workers=MAX_WORKERS)

    for dataset in datasets:
        vectors = dataset_vectors[dataset]
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
                    dim=dim, dataset=DATASET_KEYS[dataset],
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

    return df, max(k_values)


def plot_ablation(df: pd.DataFrame, k: int) -> None:
    generate_ablation_figures(df, k=k)


def generate_query_performance() -> pd.DataFrame:
    """
    Measure single-query vs. batched-query latency/throughput of a
    BallCoverFilter cover as a function of filter size k, across the same
    3 datasets used by generate_ablation(). Mirrors generate_ablation()'s
    structure: one job per (dataset, k, seed), flattened into a long-format
    DataFrame, written to results/query_performance.csv. Render with
    plot_query_performance().
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
            "generate_query_performance needs all 3 datasets "
            f"({', '.join(DATASET_KEYS)}) on disk; only found: {', '.join(datasets) or 'none'}"
        )

    # k >= a dataset's construct-set size makes BallCoverFilter fall back to
    # one zero-radius ball per point, so clamp the sweep the same way
    # generate_ablation() does (kept defensive even though the 100K-row
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


def generate_scaling() -> Tuple[pd.DataFrame, int]:
    """
    Fix filter size k and sweep indexed-set size n: memory and query latency
    should stay flat, construction time and FPR may grow with n. One job per
    (dataset, n, seed); a fixed query holdout is reused across every n.
    """
    num_vectors_pool = 100_000
    dim              = 384
    k_fixed          = 256
    n_values         = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
    query_holdout    = 10_000
    tau_mid          = 0.35
    seeds            = [0, 1, 2, 3, 4]
    single_query_limit = 2000
    output_csv       = "results/scaling.csv"

    datasets = [d for d in DATASET_KEYS if os.path.exists(dataset_filenames[d])]
    missing = [d for d in DATASET_KEYS if d not in datasets]
    if missing:
        print(f"Skipping dataset(s) not found on disk: {', '.join(missing)}")
    if set(datasets) != set(DATASET_KEYS):
        raise RuntimeError(
            "generate_scaling needs all 3 datasets "
            f"({', '.join(DATASET_KEYS)}) on disk; only found: {', '.join(datasets) or 'none'}"
        )

    # n must leave room for the query holdout, and k must stay below every n.
    pool_sizes = {d: len(load_embeds(d, num_vectors_pool)[0]) for d in datasets}
    max_usable_n = min(pool_sizes.values()) - query_holdout
    usable_n_values = [n for n in n_values if n <= max_usable_n]
    dropped_n = sorted(set(n_values) - set(usable_n_values))
    if dropped_n:
        print(
            f"Dropping n value(s) that don't leave room for a {query_holdout}-vector "
            f"query holdout (smallest dataset pool: {min(pool_sizes.values())} vectors, "
            f"from {min(pool_sizes, key=pool_sizes.get)}): {dropped_n}"
        )
    if not usable_n_values:
        raise RuntimeError(
            f"No n value in {n_values} leaves room for a {query_holdout}-vector query "
            f"holdout out of the smallest dataset pool ({min(pool_sizes.values())} vectors). "
            "Raise num_vectors_pool or lower n_values/query_holdout."
        )
    n_values = usable_n_values
    if k_fixed >= min(n_values):
        raise RuntimeError(
            f"k_fixed={k_fixed} must be smaller than every n in {n_values} "
            "(k stays fixed across the whole sweep)."
        )

    runner = Runner(num_workers=MAX_WORKERS)

    for dataset in datasets:
        vectors = load_embeds(dataset, num_vectors_pool)[0]
        total = len(vectors)  # a dataset may hold fewer than num_vectors_pool
        query_start = total - query_holdout
        vectors_query = vectors[query_start:]
        query_source = VectorSource(dataset, query_start, total)

        for n in n_values:
            vectors_construct_n = vectors[:n]
            construct_source_n = VectorSource(dataset, 0, n)

            print(f"{dataset}: computing ground truth at n={n}, tau_mid={tau_mid}...")
            actual_results_n = compute_static_actual_results(vectors_construct_n, vectors_query, tau_mid)

            for seed in seeds:
                runner.submit(
                    'scaling',
                    dim=dim, dataset=DATASET_KEYS[dataset],
                    k=k_fixed, seed=seed, tau_mid=tau_mid,
                    vectors_construct=construct_source_n, vectors_query=query_source,
                    actual_results=actual_results_n, single_query_limit=single_query_limit,
                )

    job_results = runner.run()
    print("Scaling evaluation completed.")

    df = pd.DataFrame(job_results)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Wrote {output_csv}")

    return df, k_fixed


def plot_scaling(df: pd.DataFrame, k_fixed: int) -> None:
    generate_scaling_figures(df, k_fixed=k_fixed)


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


def generate_streaming() -> Tuple[pd.DataFrame, int, int]:
    """
    Compare 5 streaming insertion methods (M1-M5, see
    filters.streaming_insertion) that differ in decision rule and cost
    class, not in a tuning knob -- the O(k^2 d) -> O(kd) merge-search
    optimization (M4) is the centerpiece result. Produces the long-format
    data behind the streaming figure's four panels:
      (a) FPR vs. stream position (fixed k_fixed), + a dashed static-rebuild
          reference (stream_rebuild_worker policy="oracle");
      (b) threshold-expanded objective inflation vs. stream position (same
          x-axis/checkpoints as (a));
      (c) final FPR vs. ball budget k, one line per method;
      (d) Pareto: mean FPR (2nd half of stream) vs. amortized cost/insertion,
          one connected curve per method per k (M5: per (k, b)), plus the
          exhaustive Algorithm 2 reference (stream_rebuild_worker
          policy="streaming", merge_criterion="log_space" -- no new code
          needed, add() already *is* that reference) as a 6th dashed curve,
          since "M4 matches Algorithm 2's quality at a fraction of the cost"
          is the paper's stated centerpiece claim and (d) is where it's made.

    Every dataset on disk caps out around ~100k vectors, so the spec's
    "stream position from k to >= 10^3 * k" is only reachable at a small
    k_fixed -- see k_fixed's comment below. Panel (c)'s k-sweep goes up to
    k=1024 but shares one fixed-length stream (n_stream_c) across every k,
    not a literal 10^3*k-scaled length -- documented here rather than
    silently truncated; most consequential at k=1024 (~29*k, not 1000*k).

    "Multiple starting offsets" (the paper's K-fold variance protocol) is
    operationalized as overlapping offset windows into one shared
    representative-dataset pool (true disjoint folds don't scale to a
    useful count at this vector budget) -- a sliding-window robustness
    check, not strict cross-validation.

    FNR=0 is asserted per checkpoint inside streaming_method_worker (raises,
    not recorded) for every method -- the covering invariant is a
    correctness requirement, not a plotted quantity.

    Writes results/streaming.csv (panels a-d) and results/streaming_checks.json
    (the M4-exactness / M3-gap / Delta-distribution text-report checks).
    Returns the long-format df plus k_fixed/b_representative, which
    plot_streaming() needs and can't derive from the df alone.
    """
    dim = 384
    # 10^3 * k_fixed = 32,000 fits the ~100k-vector budget alongside a held-out
    # negative pool and K-fold offset margin (see the pool arithmetic below);
    # k_fixed also sits inside the panel-(c) k-sweep, so (a)/(b) is
    # cross-checkable against (c) at k=32.
    k_fixed = 32
    k_values = [16, 32, 64, 128, 256, 512, 1024]
    methods = ["M1", "M2", "M3", "M4"]  # M5 handled separately below (its own b dimension)
    # The paper's illustrative b in {1e3,1e4,1e5} doesn't fit a ~30k-vector
    # stream (1e5 would mean "rebuild once, at the very end" -- a materially
    # different, less interesting policy); replaced with values that stay
    # meaningful at this budget while spanning a similar order of magnitude.
    b_values_sweep = [1_500, 6_000, 15_000]
    b_representative = 6_000  # panels a/b's single M5 curve
    n_stream_ab, F_ab, stride_ab = k_fixed * 1000, 4, 1_500
    n_stream_c, F_cd, stride_cd = 30_000, 4, 1_000
    n_negative, negative_candidate_multiplier = 500, 10
    duplicate_percentage = 2.0  # calibrates tau_mid; see calculate_nn_distance_threshold
    seed = 0  # fixed; fold_offset supplies the stream-order variance instead
    output_csv = "results/streaming.csv"
    checks_output = "results/streaming_checks.json"

    datasets = [d for d in DATASET_KEYS if os.path.exists(dataset_filenames[d])]
    missing = [d for d in DATASET_KEYS if d not in datasets]
    if missing:
        print(f"Skipping dataset(s) not found on disk: {', '.join(missing)}")
    if set(datasets) != set(DATASET_KEYS):
        raise RuntimeError(
            "generate_streaming needs all 3 datasets "
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
            f"{negative_candidates_len}). Lower F_ab/F_cd/n_stream_*/n_negative."
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
    # metric/fold_offset all included directly -- see streaming_method_worker
    # and stream_rebuild_worker's docstrings), so results don't need
    # positional reconstruction here -- which matters because Runner's
    # parallel mode returns results in *completion* order, not submission
    # order. output_csv makes each Runner write every job's rows to disk as
    # soon as that job finishes, so a mid-sweep crash still leaves whatever
    # completed so far on disk instead of losing the whole run.

    # --- Runner A: panels (a)/(b), position sweep at fixed k_fixed ---
    runner_a = Runner(num_workers=MAX_WORKERS, output_csv=output_csv, truncate=True)
    for method in methods + ["M5"]:
        method_kwargs = {"b": b_representative, "construct_tau": tau_mid} if method == "M5" else {}
        for fold_offset in offsets_ab:
            src = VectorSource(rep_dataset, fold_offset, fold_offset + n_stream_ab)
            runner_a.submit(
                'streaming_method',
                dim=dim, dataset=rep_key, k=k_fixed, seed=seed, fold_offset=fold_offset, tau_mid=tau_mid,
                method=method, method_kwargs=method_kwargs,
                vectors_stream_pool=src, stream_len=n_stream_ab,
                vectors_negative=negative_vectors, checkpoints=checkpoints_ab, sweep="position",
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
                    'streaming_method',
                    dim=dim, dataset=rep_key, k=k, seed=seed, fold_offset=fold_offset, tau_mid=tau_mid,
                    method=method, method_kwargs={},
                    vectors_stream_pool=src, stream_len=n_stream_c,
                    vectors_negative=negative_vectors, checkpoints=checkpoints_cd, sweep="k_sweep",
                )
    for k in k_values:
        for b in b_values_sweep:
            for fold_offset in offsets_cd:
                src = VectorSource(rep_dataset, fold_offset, fold_offset + n_stream_c)
                runner_b.submit(
                    'streaming_method',
                    dim=dim, dataset=rep_key, k=k, seed=seed, fold_offset=fold_offset, tau_mid=tau_mid,
                    method="M5", method_kwargs={"b": b, "construct_tau": tau_mid},
                    vectors_stream_pool=src, stream_len=n_stream_c,
                    vectors_negative=negative_vectors, checkpoints=checkpoints_cd, sweep="k_sweep",
                )
    for k in k_values:
        for fold_offset in offsets_cd:
            src = VectorSource(rep_dataset, fold_offset, fold_offset + n_stream_c)
            runner_b.submit(
                'stream_rebuild',
                dim=dim, dataset=rep_key, k=k, seed=seed, tau_mid=tau_mid, policy="streaming", T=None,
                vectors_stream=src, vectors_negative=negative_vectors, checkpoints=checkpoints_cd,
                sweep="k_sweep", fold_offset=fold_offset,
            )
    rows_cd = runner_b.run()
    print("Streaming k-sweep (c/d) evaluation completed.")
    df_cd = pd.DataFrame([row for job_rows in rows_cd for row in job_rows])

    df = pd.concat([df_ab, df_cd], ignore_index=True)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Wrote {output_csv} (already written incrementally during the run; this is the final canonical pass)")

    # --- Runner C: the three text-report checks (M4 exactness, M3 gap, Delta distribution) ---
    runner_c = Runner(num_workers=MAX_WORKERS)
    check_src = VectorSource(rep_dataset, 0, n_stream_ab)
    for k in (32, 64):
        runner_c.submit(
            'stream_m4_exactness',
            dim=dim, dataset=rep_key, k=k, seed=seed, tau_mid=tau_mid,
            vectors_stream_pool=VectorSource(rep_dataset, 0, k * 1000), stream_len=k * 1000,
        )
    runner_c.submit(
        'stream_m3_gap',
        dim=dim, dataset=rep_key, k=k_fixed, seed=seed, tau_mid=tau_mid,
        vectors_stream_pool=check_src, stream_len=n_stream_ab,
    )
    for method in ("M3", "M4", "ref_exhaustive"):
        runner_c.submit(
            'stream_delta_distribution',
            dim=dim, dataset=rep_key, k=k_fixed, seed=seed, tau_mid=tau_mid, method=method,
            vectors_stream_pool=check_src, stream_len=n_stream_ab,
        )
    check_results = runner_c.run()
    print("Streaming correctness/diagnostic checks completed.")

    # Routed by each result's own distinguishing field, not by submission
    # position -- Runner's parallel mode returns results in completion
    # order, which needn't match submission order.
    checks = {"m4_exactness": {}, "m3_gap": None, "delta_distribution": {}}
    for result in check_results:
        if "n_mismatches" in result:
            checks["m4_exactness"][result["k"]] = result
        elif "fraction_excluded" in result:
            checks["m3_gap"] = result
        elif "frac_beneficial" in result:
            checks["delta_distribution"][result["method"]] = result
        else:
            raise ValueError(f"Unrecognized streaming check result shape: {sorted(result.keys())}")

    os.makedirs(os.path.dirname(checks_output), exist_ok=True)
    with open(checks_output, "w") as f:
        json.dump(checks, f, indent=2)
    print(f"Wrote {checks_output}")

    return df, k_fixed, b_representative


def plot_streaming(df: pd.DataFrame, k_fixed: int, b_representative: int) -> None:
    generate_streaming_figures(df, k_fixed=k_fixed, b_representative=b_representative)


def generate_dynamic() -> list:
    num_vectors              = 1_000
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
    return results


def plot_dynamic(results: list) -> None:
    plot_results(results, "dynamic")


def generate_tradeoff() -> Tuple[List, List]:
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

    return sphere_results, memory_results


def plot_tradeoff(sphere_results: List, memory_results: List) -> None:
    plot_tradeoff_results(sphere_results, memory_results)


def _load_tradeoff() -> Tuple[List, List]:
    """Reconstructs sphere_results/memory_results from tradeoff_evaluations.csv
    (written as a side effect of plot_tradeoff -- see legacy.plot_tradeoff_results),
    splitting back on the 'Mode' column plot_tradeoff_results itself stamps onto
    each half before concatenating them for that CSV."""
    df = pd.read_csv("plots/tradeoff/tradeoff_evaluations.csv")
    sphere_results = df[df["Mode"] == "sphere_test"].to_dict("records")
    memory_results = df[df["Mode"] == "memory_test"].to_dict("records")
    return sphere_results, memory_results


def generate_and_plot_all() -> None:
    """
    Run every active evaluation end-to-end (generate, then plot), the same
    5 evaluations __main__ used to run inline, plus system (tau-sweep).
    static/dynamic/equal_memory stay disabled -- they fail at the
    100K-vector scale this file is currently configured for (see comments
    below). The early return currently isolates system for standalone
    testing; remove it to also run the other 5.
    """
    plot_streaming(*generate_streaming())
    #plot_static(generate_static())
    #plot_dynamic(generate_dynamic())
    #plot_equal_memory(generate_equal_memory()) // failed on 100K
    #plot_system(*generate_system())
    #plot_system_tau(*generate_system_tau())
    return
    try:
        pass
        #plot_tradeoff(*generate_tradeoff())
    except Exception as e:
        print(f"An error occurred: {e}")

    try:
        plot_ablation(*generate_ablation())
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
    results/{ablation,query_performance,scaling,streaming,system}.csv and
    plots/tradeoff/tradeoff_evaluations.csv, re-deriving whatever generate-time
    metadata (k, k_fixed, T_representative) each plot_ needs directly from the
    loaded data instead of threading it through from the generate_ call.
    Requires at least one prior generate_and_plot_all() (or equivalent
    generate_X + plot_X) run to have produced those files.
    """
    try:
        plot_tradeoff(*_load_tradeoff())
    except Exception as e:
        print(f"An error occurred: {e}")

    try:
        df = pd.read_csv("results/system.csv")
        plot_system(df, k=int(df["k"].iloc[0]))
    except Exception as e:
        print(f"An error occurred: {e}")

    try:
        df = pd.read_csv("results/system_tau.csv")
        plot_system_tau(df, k=int(df["k"].iloc[0]))
    except Exception as e:
        print(f"An error occurred: {e}")

    try:
        df = pd.read_csv("results/ablation.csv")
        plot_ablation(df, k=int(df["k"].max()))
    except Exception as e:
        print(f"An error occurred: {e}")

    try:
        df = pd.read_csv("results/query_performance.csv")
        plot_query_performance(df)
    except Exception as e:
        print(f"An error occurred: {e}")

    try:
        df = pd.read_csv("results/scaling.csv")
        plot_scaling(df, k_fixed=int(df["k"].iloc[0]))
    except Exception as e:
        print(f"An error occurred: {e}")

    try:
        df = pd.read_csv("results/streaming.csv")
        k_fixed = int(df.loc[df["sweep"] == "position", "k"].iloc[0])
        b_representative = int(df.loc[(df["sweep"] == "position") & (df["method"] == "M5"), "b"].dropna().iloc[0])
        plot_streaming(df, k_fixed=k_fixed, b_representative=b_representative)
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    generate_and_plot_all()
    # plot_all()