import numpy as np
import pandas as pd
import faiss
import inspect

from runner import Runner
from plots.plot_tradeoff import load_tradeoff_results, plot_tradeoff_results
from typing import Tuple, List
from filters.kdbox_filter import KDBoxFilter
from filters.gonzalesbox_filter import GonzalezBoxFilter
from filters.ball_cover.ball_cover_filter import BallCoverFilter
from filters.dsbf_filter import DSBFFilter
from filters.covering_lsh_fallback_filter import CoveringLSHFallbackFilter
from filters.lsbf_filter import LSBFFilter
from filters.dsh_filter import DSHFilter
from filters.hnsw_filter import HNSWFilter
from filters.ivf_filter import IVFFilter
from vector_source import VectorSource, ArrayVectorSource

MAX_WORKERS = 6
DATASET = 'ELI5'

dynamic_filters = [BallCoverFilter, DSBFFilter, CoveringLSHFallbackFilter, LSBFFilter, DSHFilter, HNSWFilter, IVFFilter]
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


def generate_tradeoff_static() -> Tuple[List, List]:
    """Orchestrates the tradeoff benchmarking pipeline: static ground truth,
    then filter evaluation across sphere-count and memory-budget sweeps."""
    num_vectors = 100_000
    dim = 384
    percentage_construct = 10
    distance_threshold = 0.2
    dataset = DATASET
    random_test = False
    synthetic_test = False

    num_construct = num_vectors * percentage_construct // 100
    num_query = num_vectors - num_construct

    if synthetic_test:
        print(f"Loading synthetic dataset from '{dataset}' using targeted keys...")
        construct_vectors = VectorSource(dataset, 0, num_construct, dataset_key="construct_vectors")
        query_vectors = VectorSource(dataset, 0, num_query, dataset_key="query_vectors")
    else:
        print(f"Loading real-world embeddings from dataset: '{dataset}'...")
        construct_vectors = VectorSource(dataset, 0, num_construct)
        query_vectors = VectorSource(dataset, num_construct, num_vectors)

        if random_test:
            print("\nPreparing Generalization Test (Random Vectors)...")
            all_vectors = np.vstack((construct_vectors.to_array(), query_vectors.to_array()))
            random_queries = np.random.uniform(low=np.min(all_vectors), high=np.max(all_vectors), size=(num_query, dim)).astype(np.float32)
            random_queries_norms = np.linalg.norm(random_queries, axis=1, keepdims=True)
            random_queries = random_queries / np.where(random_queries_norms == 0, 1e-10, random_queries_norms)
            query_vectors = ArrayVectorSource("random_queries", array=random_queries)

    print("Calculating actual results for test queries...")
    actual_results = compute_static_actual_results(construct_vectors.to_array(), query_vectors.to_array(), distance_threshold)

    runner = Runner(num_workers=MAX_WORKERS)

    # Sphere-count sweep: search accuracy/speed vs. spatial partitioning granularity.
    num_spheres = [5, 10, 20, 40, 80, 160, 320, 640, 1000, 1500, 2000, 2300, 2600, 2900, 3200, 3600,
                    4000, 4400, 4800, 5300, 5800, 6000, 6300, 6800, 7400, 8000, 8700, 9000, 9700, 10000]
    num_spheres = [s for s in num_spheres if s < num_construct] + [num_construct]
    geometric_filters = [f for f in dynamic_filters + static_filters if 'max_spheres' in inspect.signature(f.__init__).parameters]

    for size in num_spheres:
        for filter_cls in geometric_filters:
            runner.submit(
                'tradeoff_static', filter_cls=filter_cls, dim=dim, distance_threshold=distance_threshold,
                vectors_construct=construct_vectors, vectors_query=query_vectors,
                actual_results=actual_results, max_spheres=size, memory_budget=None
            )

    print("\nExecuting sphere size evaluation...")
    raw_sphere_results = runner.run()
    sphere_results = [r for r in raw_sphere_results if r is not None]
    runner.clear()
    print("Sphere size evaluation completed.")

    # Memory-budget sweep: all filter strategies under strict memory limits.
    MB = 1024 * 1024
    memory_budgets = [int(0.5 * MB), int(0.75 * MB), int(1 * MB), int(1.5 * MB), int(2 * MB), int(3 * MB), int(4.5 * MB), 
                      int(6.5 * MB), int(9 * MB), int(13 * MB), int(18 * MB), int(26 * MB), int(38 * MB), int(54 * MB), int(64 * MB)]

    for budget in memory_budgets:
        for filter_cls in static_filters + dynamic_filters:
            runner.submit(
                'tradeoff_static', filter_cls=filter_cls, dim=dim, distance_threshold=distance_threshold,
                vectors_construct=construct_vectors, vectors_query=query_vectors,
                actual_results=actual_results, max_spheres=None, memory_budget=budget
            )

    print("\nExecuting memory evaluation...")
    raw_memory_results = runner.run()
    memory_results = [r for r in raw_memory_results if r is not None]
    print("Memory evaluation completed.")
    plot_tradeoff_results(sphere_results, memory_results)

    return sphere_results, memory_results


def plot_tradeoff_static(sphere_results: List, memory_results: List) -> None:
    plot_tradeoff_results(sphere_results, memory_results, folder="plots/tradeoff_static")


def generate_tradeoff_dynamic() -> Tuple[List, List]:
    """Orchestrates the tradeoff benchmarking pipeline: dynamic ground truth,
    then filter evaluation across sphere-count and memory-budget sweeps."""
    num_vectors = 100_000
    dim = 384
    percentage_construct = 10
    distance_threshold = 0.2
    dataset = DATASET
    random_test = False
    synthetic_test = False

    num_construct = num_vectors * percentage_construct // 100
    num_query = num_vectors - num_construct

    if synthetic_test:
        print(f"Loading synthetic dataset from '{dataset}' using targeted keys...")
        construct_vectors = VectorSource(dataset, 0, num_construct, dataset_key="construct_vectors")
        query_vectors = VectorSource(dataset, 0, num_query, dataset_key="query_vectors")
    else:
        print(f"Loading real-world embeddings from dataset: '{dataset}'...")
        construct_vectors = VectorSource(dataset, 0, num_construct)
        query_vectors = VectorSource(dataset, num_construct, num_vectors)

        if random_test:
            print("\nPreparing Generalization Test (Random Vectors)...")
            all_vectors = np.vstack((construct_vectors.to_array(), query_vectors.to_array()))
            random_queries = np.random.uniform(low=np.min(all_vectors), high=np.max(all_vectors), size=(num_query, dim)).astype(np.float32)
            random_queries_norms = np.linalg.norm(random_queries, axis=1, keepdims=True)
            random_queries = random_queries / np.where(random_queries_norms == 0, 1e-10, random_queries_norms)
            query_vectors = ArrayVectorSource("random_queries", array=random_queries)

    print("Calculating actual results for test queries...")
    all_vectors = np.vstack((construct_vectors.to_array(), query_vectors.to_array()))
    actual_results = compute_dynamic_actual_results(all_vectors, distance_threshold)[num_construct:]

    runner = Runner(num_workers=MAX_WORKERS)

    # Sphere-count sweep: search accuracy/speed vs. spatial partitioning granularity.
    num_spheres = [5, 10, 20, 40, 80, 160, 320, 640, 1000, 1500, 2000, 2300, 2600, 2900, 3200, 3600,
                    4000, 4400, 4800, 5300, 5800, 6000, 6300, 6800, 7400, 8000, 8700, 9000, 9700, 10000]
    num_spheres = [s for s in num_spheres if s < num_construct] + [num_construct]
    geometric_filters = [f for f in dynamic_filters + static_filters if 'max_spheres' in inspect.signature(f.__init__).parameters]

    for size in num_spheres:
        for filter_cls in geometric_filters:
            runner.submit(
                'tradeoff_dynamic', filter_cls=filter_cls, dim=dim, distance_threshold=distance_threshold,
                vectors_construct=construct_vectors, vectors_query=query_vectors,
                actual_results=actual_results, max_spheres=size, memory_budget=None
            )

    print("\nExecuting sphere size evaluation...")
    raw_sphere_results = runner.run()
    sphere_results = [r for r in raw_sphere_results if r is not None]
    runner.clear()
    print("Spheres size evaluation completed.")

    # Memory-budget sweep: all dynamic filter strategies under strict memory limits.
    MB = 1024 * 1024
    memory_budgets = [int(0.5 * MB), int(0.75 * MB), int(1 * MB), int(1.5 * MB), int(2 * MB), int(3 * MB), int(4.5 * MB), 
                      int(6.5 * MB), int(9 * MB), int(13 * MB), int(18 * MB), int(26 * MB), int(38 * MB), int(54 * MB), int(64 * MB)]

    for budget in memory_budgets:
        for filter_cls in dynamic_filters:
            runner.submit(
                'tradeoff_dynamic', filter_cls=filter_cls, dim=dim, distance_threshold=distance_threshold,
                vectors_construct=construct_vectors, vectors_query=query_vectors,
                actual_results=actual_results, max_spheres=None, memory_budget=budget
            )

    print("\nExecuting memory evaluation...")
    raw_memory_results = runner.run()
    memory_results = [r for r in raw_memory_results if r is not None]
    print("Memory evaluation completed.")
    plot_tradeoff_results(sphere_results, memory_results, folder="plots/tradeoff_dynamic")

    return sphere_results, memory_results


def plot_tradeoff_dynamic(sphere_results: List, memory_results: List) -> None:
    plot_tradeoff_results(sphere_results, memory_results, folder="plots/tradeoff_dynamic")



if __name__ == "__main__":
    generate_tradeoff_static()
    # generate_tradeoff_dynamic()