import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import seaborn as sns

from filters.cuckoo_filter import CuckooFilter
from filters.gcs_filter import GCSFilter
from filters.kmeb_filter import KMEBFilter
from filters.pyfusefilter_filters import Fuse8Filter, Xor8Filter
from filters.quotient_filter import QuotientFilterWrapper
from lsh import CrossPolytopeLSH, LSH
from runner import Runner
from typing import List
from util.fetch_datasets import load_embeds

MAX_WORKERS = 6

dynamic_filters = [CuckooFilter, QuotientFilterWrapper]
static_filters  = [Xor8Filter, Fuse8Filter, GCSFilter, KMEBFilter]
lsh_families: List[LSH] = [CrossPolytopeLSH]


def calculate_memory_budget(num_vectors: int, bits_per_element: float = 12.0, overhead_bytes: int = 4096) -> int:
    return int(np.ceil(num_vectors * bits_per_element / 8.0)) + overhead_bytes


def calculate_distance_threshold(vectors: np.ndarray, percentage: float) -> float:
    """
    Return the distance threshold at which `percentage`% of all pairwise L2
    distances are considered duplicates (i.e., the `percentage`-th percentile).
    """
    sq_norms = np.sum(vectors ** 2, axis=1)
    dot      = vectors @ vectors.T
    sq_dists = sq_norms[:, None] + sq_norms[None, :] - 2 * dot
    np.clip(sq_dists, 0, None, out=sq_dists)
    upper_idx = np.triu_indices(len(vectors), k=1)
    distances = np.sqrt(sq_dists[upper_idx])
    return float(np.percentile(distances, percentage))


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
    """
    actual = []
    for v in query_vectors:
        distances = np.sqrt(np.sum((construct_vectors - v) ** 2, axis=1))
        actual.append(bool(np.any(distances <= distance_threshold)))
    return actual


def plot_results(results_list: list, folder: str) -> None:
    df = pd.DataFrame(results_list)
    df[['Filter', 'LSH']] = df['Config'].str.split(' + ', regex=False, expand=True)

    sns.set_theme(style="whitegrid")

    unique_filters = df['Filter'].unique()
    color_map = dict(zip(unique_filters, sns.color_palette("husl", len(unique_filters))))

    unique_lshs     = df['LSH'].unique()
    available_marks = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'X']
    marker_map      = dict(zip(unique_lshs, available_marks[:len(unique_lshs)]))

    configs = df['Config'].unique()

    for metric in ('F1-Score', 'Precision', 'Recall'):
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.lineplot(
            data=df, x='Percentage', y=metric,
            hue='Filter', style='LSH',
            markers=marker_map, dashes=False,
            palette=color_map, ax=ax, legend=False,
        )
        ax.set_xlabel('Percentage of Duplicates (%)', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_ylim(-0.05, 1.05)
        filename = f'{metric.lower().replace("-", "_")}_plot.png'
        plt.savefig(f"plots/{folder}/{filename}", bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved plots/{folder}/{filename}")

    fig_leg, ax_leg = plt.subplots(figsize=(10, 2))
    ax_leg.axis('off')
    handles = [
        plt.Line2D([0], [0], color=color_map[filter_name], lw=2, marker=marker_map[lsh_name])
        for filter_name, lsh_name in (c.split(' + ') for c in configs)
    ]
    ax_leg.legend(
        handles, configs,
        loc='center', ncol=min(3, len(configs)),
        title="Configuration (Color = Filter, Marker = LSH)",
        title_fontsize=12, fontsize=10, frameon=True,
    )
    plt.savefig(f"plots/{folder}/legend_only.png", bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved plots/{folder}/legend_only.png")


def static_evaluation() -> list:
    num_vectors              = 1000
    dim                      = 384
    min_percentage_duplicates = 1
    max_percentage_duplicates = 9
    count_steps              = 9
    percentage_construct     = 10
    max_L                    = 256

    vectors = load_embeds('ELI5', num_vectors)[0]
    split   = int(num_vectors * percentage_construct / 100)
    vectors_construct = vectors[:split]
    vectors_query     = vectors[split:]
    memory_budget = calculate_memory_budget(num_vectors, bits_per_element=12.0)

    runner = Runner(num_workers=MAX_WORKERS)

    for percentage in np.linspace(min_percentage_duplicates, max_percentage_duplicates, count_steps):
        distance_threshold = calculate_distance_threshold(vectors, percentage)
        actual_results     = compute_static_actual_results(vectors_construct, vectors_query, distance_threshold)

        for filter_cls in static_filters + dynamic_filters:
            for lsh_cls in lsh_families:
                runner.submit(
                    'static',
                    filter_cls=filter_cls, lsh_cls=lsh_cls,
                    dim=dim, distance_threshold=distance_threshold,
                    max_L=max_L, memory_budget=memory_budget,
                    vectors_construct=vectors_construct, vectors_query=vectors_query,
                    actual_results=actual_results, percentage=percentage,
                )

    results = runner.run()
    print("Static evaluation completed.")
    plot_results(results, "static")
    return results


def dynamic_evaluation() -> list:
    num_vectors              = 1000
    dim                      = 384
    min_percentage_duplicates = 1
    max_percentage_duplicates = 9
    count_steps              = 10
    percentage_construct     = 25
    max_L                    = 256

    vectors       = load_embeds('ELI5', num_vectors)[0]
    num_construct = num_vectors * percentage_construct // 100
    vectors_construct = vectors[:num_construct]
    vectors_query     = vectors[num_construct:]
    memory_budget = calculate_memory_budget(num_construct, bits_per_element=12.0)

    runner = Runner(num_workers=MAX_WORKERS)

    for percentage in np.linspace(min_percentage_duplicates, max_percentage_duplicates, count_steps):
        distance_threshold = calculate_distance_threshold(vectors, percentage)
        # compute_actual_results processes all vectors in sequence;
        # slice to query positions so ground truth aligns with the query phase.
        actual_results = compute_actual_results(vectors, distance_threshold)[num_construct:]

        for filter_cls in dynamic_filters:
            for lsh_cls in lsh_families:
                runner.submit(
                    'dynamic',
                    filter_cls=filter_cls, lsh_cls=lsh_cls,
                    dim=dim, distance_threshold=distance_threshold,
                    max_L=max_L, memory_budget=memory_budget,
                    vectors_construct=vectors_construct, vectors_query=vectors_query,
                    actual_results=actual_results, percentage=percentage,
                )

    results = runner.run()
    print("Dynamic evaluation completed.")
    plot_results(results, "dynamic")
    return results


def plot_throughput_results(results_list: list, folder: str, ylabel: str, title_suffix: str) -> None:
    df = pd.DataFrame(results_list)
    df[['Filter', 'LSH']] = df['Config'].str.split(' + ', regex=False, expand=True)

    sns.set_theme(style="whitegrid")

    unique_filters = df['Filter'].unique()
    color_map = dict(zip(unique_filters, sns.color_palette("husl", len(unique_filters))))

    unique_lshs     = df['LSH'].unique()
    available_marks = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'X']
    marker_map      = dict(zip(unique_lshs, available_marks[:len(unique_lshs)]))

    configs = df['Config'].unique()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.lineplot(
        data=df, x='FilterSize', y='Throughput',
        hue='Filter', style='LSH',
        markers=marker_map, dashes=False,
        palette=color_map, ax=ax, legend=False,
    )
    ax.set_xlabel('Number of Elements in Filter', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'Throughput vs Filter Size ({title_suffix})', fontsize=13)
    filename = f'throughput_{title_suffix.lower().replace(" ", "_")}_plot.png'
    plt.savefig(f"plots/{folder}/{filename}", bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved plots/{folder}/{filename}")

    fig_leg, ax_leg = plt.subplots(figsize=(10, 2))
    ax_leg.axis('off')
    handles = [
        plt.Line2D([0], [0], color=color_map[filter_name], lw=2, marker=marker_map[lsh_name])
        for filter_name, lsh_name in (c.split(' + ') for c in configs)
    ]
    ax_leg.legend(
        handles, configs,
        loc='center', ncol=min(3, len(configs)),
        title="Configuration (Color = Filter, Marker = LSH)",
        title_fontsize=12, fontsize=10, frameon=True,
    )
    plt.savefig(f"plots/{folder}/throughput_legend_only.png", bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved plots/{folder}/throughput_legend_only.png")


def throughput_evaluation():
    import time

    dim             = 384
    max_L           = 256
    memory_budget   = 10 * 1024 * 1024
    query_batch     = 100
    num_trials      = 3
    filter_sizes    = [100, 500, 1000, 2500, 5000, 10000]

    vectors = load_embeds('ELI5', max(filter_sizes) + query_batch)[0]
    distance_threshold = calculate_distance_threshold(vectors, 5.0)

    def _build(filter_cls, lsh_instance):
        try:
            return filter_cls(dim=dim, lsh=lsh_instance, max_memory=memory_budget)
        except TypeError:
            return filter_cls(dim=dim, lsh=lsh_instance)

    # --- Query Throughput ---
    print("\n=== Query Throughput Evaluation ===")
    query_results = []
    for filter_cls in static_filters + dynamic_filters:
        for lsh_cls in lsh_families:
            config = f"{filter_cls.__name__} + {lsh_cls.__name__}"
            print(f"  Evaluating {config} ...")
            lsh_instance = lsh_cls.build_optimized(dim, distance_threshold, None, max_L)
            for size in filter_sizes:
                construct_vecs = vectors[:size]
                query_vecs     = vectors[size:size + query_batch]
                fi = _build(filter_cls, lsh_instance)
                fi.construct(construct_vecs)
                fi.query(query_vecs)  # warm-up
                times = []
                for _ in range(num_trials):
                    t0 = time.perf_counter()
                    fi.query(query_vecs)
                    times.append(time.perf_counter() - t0)
                throughput = query_batch / np.mean(times)
                query_results.append({'Config': config, 'FilterSize': size, 'Throughput': throughput})
                print(f"    Size={size}: {throughput:.1f} queries/sec")
    plot_throughput_results(query_results, "throughput", "Queries per Second", "Query")

    # --- Construction Throughput ---
    print("\n=== Construction Throughput Evaluation ===")
    construct_results = []
    for filter_cls in static_filters + dynamic_filters:
        for lsh_cls in lsh_families:
            config = f"{filter_cls.__name__} + {lsh_cls.__name__}"
            print(f"  Evaluating {config} ...")
            lsh_instance = lsh_cls.build_optimized(dim, distance_threshold, None, max_L)
            for size in filter_sizes:
                construct_vecs = vectors[:size]
                times = []
                for _ in range(num_trials):
                    fi = _build(filter_cls, lsh_instance)
                    t0 = time.perf_counter()
                    fi.construct(construct_vecs)
                    times.append(time.perf_counter() - t0)
                throughput = size / np.mean(times)
                construct_results.append({'Config': config, 'FilterSize': size, 'Throughput': throughput})
                print(f"    Size={size}: {throughput:.1f} elements/sec")
    plot_throughput_results(construct_results, "throughput", "Elements per Second", "Construction")

    # --- Insertion Throughput (dynamic filters only) ---
    print("\n=== Insertion Throughput Evaluation ===")
    insert_results = []
    for filter_cls in dynamic_filters:
        for lsh_cls in lsh_families:
            config = f"{filter_cls.__name__} + {lsh_cls.__name__}"
            print(f"  Evaluating {config} ...")
            lsh_instance = lsh_cls.build_optimized(dim, distance_threshold, None, max_L)
            for size in filter_sizes:
                construct_vecs = vectors[:size]
                insert_vecs    = vectors[size:size + query_batch]
                times = []
                for _ in range(num_trials):
                    fi = _build(filter_cls, lsh_instance)
                    fi.construct(construct_vecs)
                    t0 = time.perf_counter()
                    fi.insert(insert_vecs)
                    times.append(time.perf_counter() - t0)
                throughput = query_batch / np.mean(times)
                insert_results.append({'Config': config, 'FilterSize': size, 'Throughput': throughput})
                print(f"    Size={size}: {throughput:.1f} inserts/sec")
    plot_throughput_results(insert_results, "throughput", "Elements per Second (Insertions)", "Insertion")

    return query_results, construct_results, insert_results


if __name__ == "__main__":
    static_evaluation()
    dynamic_evaluation()
    # throughput_evaluation()
