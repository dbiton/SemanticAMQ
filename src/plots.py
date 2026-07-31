import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List


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

    memory_budgets = sorted(df['Memory Budget'].unique()) if 'Memory Budget' in df.columns else []
    if len(memory_budgets) > 1:
        df['Memory Budget (MB)'] = (df['Memory Budget'] / (1024 * 1024)).round(2)
        df['Percentage'] = df['Percentage'].round(2)
        pct_palette = sns.color_palette("viridis", n_colors=df['Percentage'].nunique())

    ylabels = {'Memory Usage': 'Memory Usage (bytes)', 'FPR': 'False Positive Rate (FPR)', 'FNR': 'False Negative Rate (FNR)'}
    bounded_metrics = ('F1-Score', 'Precision', 'Recall', 'FPR', 'FNR')
    for metric in bounded_metrics + ('Memory Usage',):
        fig, ax = plt.subplots(figsize=(8, 6))
        if len(memory_budgets) > 1:
            # One curve per duplicate percentage, coloured dark-to-light from
            # smallest to largest; line style still distinguishes filters.
            sns.lineplot(
                data=df, x='Memory Budget (MB)', y=metric,
                hue='Percentage', style='Filter',
                markers=True, dashes=False,
                palette=pct_palette, ax=ax,
            )
            ax.legend(fontsize=8, ncol=2)
            ax.set_xlabel('Memory Budget (MB)', fontsize=12)
        else:
            sns.lineplot(
                data=df, x='Percentage', y=metric,
                hue='Filter', style='LSH',
                markers=marker_map, dashes=False,
                palette=color_map, ax=ax, legend=False,
            )
            ax.set_xlabel('Percentage of Duplicates (%)', fontsize=12)
        ax.set_ylabel(ylabels.get(metric, metric), fontsize=12)
        if metric in bounded_metrics:
            ax.set_ylim(-0.05, 1.05)
        filename = f'{metric.lower().replace("-", "_").replace(" ", "_")}_plot.png'
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

    _plot_radius_stats(df, len(memory_budgets) > 1, folder)


_RADIUS_COLUMNS = ['Radius Mean', 'Radius Max', 'Radius Min', 'Radius Std']


def _plot_radius_error_bars(
    df: pd.DataFrame, x_col: str, xlabel: str, out_folder: str, filename: str, group_col: str = None,
) -> None:
    """
    Single combined plot of sphere-radius statistics reported by geometric
    filters like GonzalezMEBFilter: Mean is drawn as a line with error bars
    of +/- one Std, Max/Min as dashed bounding lines around it. Values at a
    given x are averaged first (over whatever else was swept, e.g. duplicate
    percentage) since error bars need one point per x, not seaborn's implicit
    per-point aggregation. `group_col`, if given, draws one such triple per
    distinct value (e.g. one per dataset), each in its own color.
    """
    if not set(_RADIUS_COLUMNS).issubset(df.columns) or df['Radius Mean'].isna().all():
        return
    plot_df = df.dropna(subset=['Radius Mean'])
    if plot_df.empty:
        return

    groups = list(dict.fromkeys(plot_df[group_col])) if group_col else [None]
    palette = (dict(zip(groups, sns.color_palette("husl", len(groups))))
               if group_col else {None: '#2a78d6'})

    fig, ax = plt.subplots(figsize=(9 if group_col else 8, 6))
    for group in groups:
        sub = plot_df if group is None else plot_df[plot_df[group_col] == group]
        agg = sub.groupby(x_col)[_RADIUS_COLUMNS].mean().sort_index()
        if agg.empty:
            continue
        color = palette[group]
        prefix = f"{group} - " if group is not None else ""
        ax.errorbar(
            agg.index, agg['Radius Mean'], yerr=agg['Radius Std'],
            fmt='-o', color=color, capsize=4, linewidth=2, label=f"{prefix}Mean ± Std",
        )
        ax.plot(agg.index, agg['Radius Max'], color=color, alpha=0.6, marker='^', linestyle='--', label=f"{prefix}Max")
        ax.plot(agg.index, agg['Radius Min'], color=color, alpha=0.6, marker='v', linestyle='--', label=f"{prefix}Min")

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Sphere Radius', fontsize=12)
    ax.legend(fontsize=9, loc='best')
    plt.savefig(f"{out_folder}/{filename}", bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved {out_folder}/{filename}")


def _plot_radius_stats(df: pd.DataFrame, vs_memory: bool, folder: str) -> None:
    x_col = 'Memory Budget (MB)' if vs_memory else 'Percentage'
    xlabel = 'Memory Budget (MB)' if vs_memory else 'Percentage of Duplicates (%)'
    _plot_radius_error_bars(df, x_col, xlabel, f"plots/{folder}", 'radius_stats_plot.png')


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


def get_plot_mappings(all_filters: List[str]) -> Tuple[Dict, Dict, Dict]:
    """
    function to generate consistent plot mappings:
    - Colors: Grouped by the base filter.
    - Markers/Dashes: Unique based on individual config index.
    """
    # colors grouped by base filter
    available_marks = ['s', 'o', '^', 'D', 'v', 'p', '*', 'X', 'h', 'P', 'd', 'H', '<', '>']
    available_dashes = [(1, 1), (4, 1.5), (3, 1, 1.5, 1), (5, 2, 1, 2), (2, 2), (4, 1, 1, 1, 1, 1), (1, 3), (6, 1), (3, 3, 1, 3)]
    base_names = [c.split(' + ')[0] for c in all_filters]
    unique_bases = list(dict.fromkeys(base_names)) # Preserve order
    base_palette = sns.color_palette("tab10", n_colors=len(unique_bases))
    base_color_map = dict(zip(unique_bases, base_palette))

    color_map = {config: base_color_map[config.split(' + ')[0]] for config in all_filters}
    
    # markers and dashes cycle for uniqueness
    marker_map = {config: available_marks[i % len(available_marks)] for i, config in enumerate(all_filters)}
    dash_map = {config: available_dashes[i % len(available_dashes)] for i, config in enumerate(all_filters)}
    
    return color_map, marker_map, dash_map


def plot_hnsw_results(results_list: list, folder: str = "plots/hnsw") -> None:
    """
    Plots the HNSW filtered-access evaluation aggregated per dataset: mean
    throughput, mean per-query latency, and mean HNSW index accesses of the
    query stream, comparing the unfiltered baseline against each filter config
    placed in front of the index. Bars are means over the memory-budget sweep;
    whiskers show one standard deviation across it. Also plots index accesses,
    latency, and throughput (with filter) as a function of memory budget,
    one line per dataset.
    """
    os.makedirs(folder, exist_ok=True)
    df = pd.DataFrame(results_list)

    sns.set_theme(style="whitegrid")

    baseline = 'No Filter'
    configs  = list(dict.fromkeys(df['Config']))
    datasets = list(dict.fromkeys(df['Dataset']))
    # Neutral gray for the unfiltered baseline; fixed (never cycled) categorical
    # hues for the filter configs so a config keeps its color across plots.
    series_hues = ['#2a78d6', '#008300', '#e87ba4', '#eda100', '#1baf7a', '#eb6834', '#4a3aa7', '#e34948']
    palette = {baseline: '#898781', **{c: series_hues[i % len(series_hues)] for i, c in enumerate(configs)}}
    pipelines = [baseline] + configs

    metrics = [
        ('Throughput Without Filter', 'Throughput With Filter', 'Mean Throughput (queries/s)', 'throughput'),
        ('Latency Without Filter (ms)', 'Latency With Filter (ms)', 'Mean Latency per Query (ms)', 'latency'),
        ('Total Queries', 'Index Accesses', 'Mean HNSW Index Accesses', 'index_accesses'),
    ]
    for without_col, with_col, ylabel, name in metrics:
        # Long form: one row per (dataset, pipeline, run) so the barplot can
        # average over the duplicate-percentage sweep and draw spread whiskers.
        # Without the filter every query reaches the index, so 'Total Queries'
        # doubles as the baseline access count.
        base_df = df[['Dataset', without_col]].rename(columns={without_col: ylabel})
        base_df['Pipeline'] = baseline
        with_df = df[['Dataset', 'Config', with_col]].rename(columns={with_col: ylabel, 'Config': 'Pipeline'})
        long_df = pd.concat([base_df, with_df], ignore_index=True)

        fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(datasets)), 6))
        sns.barplot(
            data=long_df, x='Dataset', y=ylabel, hue='Pipeline',
            order=datasets, hue_order=pipelines, palette=palette,
            errorbar='sd', capsize=0.15, err_kws={'linewidth': 1.2}, ax=ax,
        )
        ax.set_xlabel('Dataset', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        # above the axes so it can never sit on top of a bar
        ax.legend(
            title=None, fontsize=9, loc='lower center',
            bbox_to_anchor=(0.5, 1.02), ncol=min(3, len(pipelines)), frameon=False,
        )
        filename = f'hnsw_{name}_plot.png'
        plt.savefig(f"{folder}/{filename}", bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved {folder}/{filename}")

    vs_memory_metrics = [
        ('Index Accesses', 'Index Accesses', 'index_accesses'),
        ('Latency With Filter (ms)', 'Mean Latency per Query (ms)', 'latency'),
        ('Throughput With Filter', 'Mean Throughput (queries/s)', 'throughput'),
    ]
    for col, ylabel, name in vs_memory_metrics:
        _plot_metric_vs_memory(df, datasets, col, ylabel, name, folder)

    _plot_radius_stats_hnsw(df, datasets, folder)
    _plot_fpvr_vs_memory(df, datasets, folder)


def _plot_radius_stats_hnsw(df: pd.DataFrame, datasets: List[str], folder: str) -> None:
    if 'Memory Budget' not in df.columns or df['Memory Budget'].nunique() <= 1:
        return
    plot_df = df.copy()
    plot_df['Memory Budget (MB)'] = plot_df['Memory Budget'] / (1024 * 1024)
    group_col = 'Dataset' if len(datasets) > 1 else None
    _plot_radius_error_bars(
        plot_df, 'Memory Budget (MB)', 'Memory Budget (MB)', folder, 'hnsw_radius_stats_plot.png',
        group_col=group_col,
    )


def _plot_metric_vs_memory(
    df: pd.DataFrame, datasets: List[str], col: str, ylabel: str, name: str, folder: str,
) -> None:
    """
    Plots a filtered-pipeline metric as a function of the filter's memory
    budget, with one line per dataset (styled by config when more than one
    is swept).
    """
    if 'Memory Budget' not in df.columns or df['Memory Budget'].nunique() <= 1:
        return

    plot_df = df.copy()
    plot_df['Memory Budget (MB)'] = plot_df['Memory Budget'] / (1024 * 1024)
    configs = list(dict.fromkeys(plot_df['Config']))
    palette = dict(zip(datasets, sns.color_palette("husl", len(datasets))))

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.lineplot(
        data=plot_df, x='Memory Budget (MB)', y=col,
        hue='Dataset', style='Config' if len(configs) > 1 else None,
        hue_order=datasets, palette=palette,
        markers=True, dashes=False, ax=ax,
    )
    ax.set_xlabel('Memory Budget (MB)', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(fontsize=9, loc='best')
    filename = f'hnsw_{name}_vs_memory_plot.png'
    plt.savefig(f"{folder}/{filename}", bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved {folder}/{filename}")


def _plot_fpvr_vs_memory(df: pd.DataFrame, datasets: List[str], folder: str) -> None:
    """
    Plots the volume-based False Positive Volume Ratio (FPVR = log(V_filter /
    V_original)) against memory budget, one line per dataset (styled by
    config when more than one is swept). FPVR estimates how many (natural-log)
    times larger a geometric filter's covered hypersphere volume is than the
    true duplicate region, using hypersphere-volume lower/upper bounds as a
    ground-truth-free stand-in for false-positive rate (hnsw_worker has no
    per-query labels to compute FPR directly). Kept in log space -- exponentiating
    to a raw ratio can overflow to +inf in high dimensions -- so values are
    always >= 0 and plotted on a linear axis. The lower/upper bound is drawn
    as a shaded band per series.
    """
    required = {'FPVR Lower', 'FPVR Upper', 'Memory Budget'}
    if not required.issubset(df.columns) or df['Memory Budget'].nunique() <= 1:
        return

    plot_df = df.dropna(subset=['FPVR Lower', 'FPVR Upper']).copy()
    if plot_df.empty:
        return
    plot_df['Memory Budget (MB)'] = plot_df['Memory Budget'] / (1024 * 1024)
    configs = list(dict.fromkeys(plot_df['Config']))
    palette = dict(zip(datasets, sns.color_palette("husl", len(datasets))))

    fig, ax = plt.subplots(figsize=(9, 6))
    for dataset in datasets:
        color = palette[dataset]
        for config in configs:
            sub = plot_df[(plot_df['Dataset'] == dataset) & (plot_df['Config'] == config)]
            sub = sub.sort_values('Memory Budget (MB)')
            if sub.empty:
                continue
            label = dataset if len(configs) <= 1 else f"{dataset} ({config})"
            mid = (sub['FPVR Lower'] + sub['FPVR Upper']) / 2
            ax.plot(sub['Memory Budget (MB)'], mid, color=color, marker='o', label=label)
            ax.fill_between(sub['Memory Budget (MB)'], sub['FPVR Lower'], sub['FPVR Upper'],
                             color=color, alpha=0.2, linewidth=0)

    ax.set_xlabel('Memory Budget (MB)', fontsize=12)
    ax.set_ylabel('False Positive Volume Ratio (log[V_filter / V_original])', fontsize=12)
    ax.legend(fontsize=9, loc='best')
    filename = 'hnsw_fpvr_vs_memory_plot.png'
    plt.savefig(f"{folder}/{filename}", bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved {folder}/{filename}")


def plot_equal_memory_results(results_list: list, folder: str = "plots/equal_memory") -> None:
    """
    Plots the equal-memory comparison: every filter is evaluated at the same
    fixed set of memory budgets (see main.py::equal_memory_evaluation), each
    having independently picked its best available configuration on a
    calibration workload. One grouped bar chart per metric (FPR, FNR,
    latency, throughput), x-axis = memory budget, one bar per filter config.
    """
    os.makedirs(folder, exist_ok=True)
    df = pd.DataFrame(results_list)

    sns.set_theme(style="whitegrid")

    df['Memory Budget (MB)'] = (df['Memory Budget'] / (1024 * 1024)).round(3)
    configs = list(dict.fromkeys(df['Config']))
    color_map, _, _ = get_plot_mappings(configs)
    budget_order = sorted(df['Memory Budget (MB)'].unique())

    metrics = [
        ('FPR', 'False Positive Rate (FPR)', 'fpr'),
        ('FNR', 'False Negative Rate (FNR)', 'fnr'),
        ('Latency (ms)', 'Query Latency (ms)', 'latency'),
        ('Throughput (qps)', 'Query Throughput (queries/s)', 'throughput'),
    ]
    for col, ylabel, name in metrics:
        fig, ax = plt.subplots(figsize=(max(8, 1.8 * len(budget_order)), 6))
        sns.barplot(
            data=df, x='Memory Budget (MB)', y=col, hue='Config',
            order=budget_order, hue_order=configs, palette=color_map, ax=ax,
        )
        ax.set_xlabel('Memory Budget (MB)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        if col in ('FPR', 'FNR'):
            ax.set_ylim(-0.05, 1.05)
        ax.legend(
            title=None, fontsize=9, loc='lower center',
            bbox_to_anchor=(0.5, 1.02), ncol=min(3, len(configs)), frameon=False,
        )
        filename = f'equal_memory_{name}_plot.png'
        plt.savefig(f"{folder}/{filename}", bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved {folder}/{filename}")

    csv_path = f"{folder}/equal_memory_evaluations.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")


def plot_tradeoff_results(sphere_results: list, memory_results: list, folder: str = "plots/tradeoff") -> None:
    """
    Plots tradeoff evaluation results, comparing FPR, FNR, Precision, F1-Score, QPS, 
    and ConstructTime across both geometric (spheres) and hardware (memory) sweeps.
    """
    os.makedirs(folder, exist_ok=True)
    
    # Initialize DataFrames and inject Mode labels
    df_spheres = pd.DataFrame(sphere_results) if sphere_results else pd.DataFrame()
    df_memory = pd.DataFrame(memory_results) if memory_results else pd.DataFrame()
    
    if not df_spheres.empty and 'Mode' not in df_spheres.columns:
        df_spheres.insert(0, 'Mode', 'sphere_test')
    if not df_memory.empty and 'Mode' not in df_memory.columns:
        df_memory.insert(0, 'Mode', 'memory_test')

    # Convert Bytes to MB for the memory plots
    if not df_memory.empty:
        df_memory['MemoryBudget_MB'] = df_memory['MemoryBudget'] / (1024 * 1024)

    sns.set_theme(style="whitegrid")

    # Generate unified maps for colors, markers, and dash patterns
    all_filters = list(dict.fromkeys(list(df_spheres['Config'].unique()) + list(df_memory['Config'].unique())))
    color_map, marker_map, dash_map = get_plot_mappings(all_filters)

    # Helper function to place a single legend below individual 1x1 plots
    def _add_bottom_legend(ax_obj):
        handles, labels = ax_obj.get_legend_handles_labels()
        if ax_obj.get_legend() is not None:
            ax_obj.get_legend().remove()
        ax_obj.legend(
            handles, labels, 
            loc='upper center', bbox_to_anchor=(0.5, -0.18), 
            ncol=min(3, len(labels)), fontsize=11, frameon=True
        )

    # Dictionary of standard plot styling args to keep calls clean
    plot_kwargs = {
        'palette': color_map,
        'markers': marker_map,
        'dashes': dash_map,
        'linewidth': 2.5,
        'markersize': 10,
        'markeredgewidth': 1.5,
        'alpha': 0.75
    }

    # CHART 1: Evaluate how increasing sphere count impacts overall filtering accuracy
    if not df_spheres.empty:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        (ax1, ax2), (ax3, ax4) = axes
        
        metrics = [
            ('FPR', 'False Positive Rate (FPR)', ax1, 'FPR across Sphere Count'),
            ('FNR', 'False Negative Rate (FNR)', ax2, 'FNR across Sphere Count'),
            ('Precision', 'Precision', ax3, 'Precision across Sphere Count'),
            ('F1-Score', 'F1-Score', ax4, 'F1-Score across Sphere Count')
        ]
        
        for col, ylabel, ax, title in metrics:
            sns.lineplot(
                data=df_spheres, x='NumSpheres', y=col, hue='Config', style='Config',
                ax=ax, **plot_kwargs
            )
            ax.set_xlabel('Number of Spheres', fontsize=12, fontweight='bold')
            ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
            ax.set_ylim(-0.05, 1.05)
            ax.set_title(title, fontsize=13)
            if ax.get_legend() is not None:
                ax.get_legend().remove()
        
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=min(3, len(labels)), fontsize=12, frameon=True)
        
        plt.suptitle('Filter Accuracy Metrics across Sphere Allocations', fontsize=16, y=0.96)
        fig.subplots_adjust(bottom=0.12, top=0.92, hspace=0.3)
        
        filename1 = "accuracy_vs_spheres_plot.png"
        plt.savefig(f"{folder}/{filename1}", bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved {folder}/{filename1}")

    # CHART 2: Analyze the trade-off between memory allocation and filtering accuracy
    if not df_memory.empty:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        (ax1, ax2), (ax3, ax4) = axes
        
        metrics = [
            ('FPR', 'False Positive Rate (FPR)', ax1, 'FPR across Memory Size'),
            ('FNR', 'False Negative Rate (FNR)', ax2, 'FNR across Memory Size'),
            ('Precision', 'Precision', ax3, 'Precision across Memory Size'),
            ('F1-Score', 'F1-Score', ax4, 'F1-Score across Memory Size')
        ]
        
        for col, ylabel, ax, title in metrics:
            sns.lineplot(
                data=df_memory, x='MemoryBudget_MB', y=col, hue='Config', style='Config',
                ax=ax, **plot_kwargs
            )
            ax.set_xlabel('Memory Allocation (MB)', fontsize=12, fontweight='bold')
            ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
            ax.set_ylim(-0.05, 1.05)
            ax.set_title(title, fontsize=13)
            if ax.get_legend() is not None:
                ax.get_legend().remove()
        
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=min(3, len(labels)), fontsize=12, frameon=True)
        
        plt.suptitle('Filter Accuracy Metrics across Memory Budgets', fontsize=16, y=0.96)
        fig.subplots_adjust(bottom=0.12, top=0.92, hspace=0.3)
        
        filename2 = "accuracy_vs_memory_plot.png"
        plt.savefig(f"{folder}/{filename2}", bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved {folder}/{filename2}")

    # CHART 3: Compare query performance scaling across different sphere densities
    if not df_spheres.empty and 'QPS' in df_spheres.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(
            data=df_spheres, x='NumSpheres', y='QPS', hue='Config', style='Config',
            ax=ax, **plot_kwargs
        )
        ax.set_xlabel('Number of Spheres', fontsize=12, fontweight='bold')
        ax.set_ylabel('Queries per Second (QPS)', fontsize=12, fontweight='bold')
        ax.set_title('Query Throughput across Sphere Count', fontsize=13)
        _add_bottom_legend(ax)
        
        filename3 = "qps_vs_spheres_plot.png"
        plt.savefig(f"{folder}/{filename3}", bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved {folder}/{filename3}")

    # CHART 4: Assess how memory budget constraints affect query throughput (QPS)
    if not df_memory.empty and 'QPS' in df_memory.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(
            data=df_memory, x='MemoryBudget_MB', y='QPS', hue='Config', style='Config',
            ax=ax, **plot_kwargs
        )
        ax.set_xlabel('Memory Allocation (MB)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Queries per Second (QPS)', fontsize=12, fontweight='bold')
        ax.set_title('Query Throughput across Memory Size', fontsize=13)
        _add_bottom_legend(ax)
        
        filename4 = "qps_vs_memory_plot.png"
        plt.savefig(f"{folder}/{filename4}", bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved {folder}/{filename4}")
        
    # CHART 5: Benchmark filter construction time relative to sphere complexity
    if not df_spheres.empty and 'ConstructTime' in df_spheres.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(
            data=df_spheres, x='NumSpheres', y='ConstructTime', hue='Config', style='Config',
            ax=ax, **plot_kwargs
        )
        ax.set_xlabel('Number of Spheres', fontsize=12, fontweight='bold')
        ax.set_ylabel('Construct Time (Seconds)', fontsize=12, fontweight='bold')
        ax.set_title('Filter Build Time across Sphere Count', fontsize=13)
        _add_bottom_legend(ax)
        
        filename5 = "construct_time_vs_spheres_plot.png"
        plt.savefig(f"{folder}/{filename5}", bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved {folder}/{filename5}")

    # CHART 6: Measure the impact of memory capacity on filter initialization speed
    if not df_memory.empty and 'ConstructTime' in df_memory.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(
            data=df_memory, x='MemoryBudget_MB', y='ConstructTime', hue='Config', style='Config',
            ax=ax, **plot_kwargs
        )
        ax.set_xlabel('Memory Allocation (MB)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Construct Time (Seconds)', fontsize=12, fontweight='bold')
        ax.set_title('Filter Build Time across Memory Size', fontsize=13)
        _add_bottom_legend(ax)
        
        filename6 = "construct_time_vs_memory_plot.png"
        plt.savefig(f"{folder}/{filename6}", bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved {folder}/{filename6}")

    # Final Result: Combine and save all experimental metrics for documentation
    if not df_spheres.empty and not df_memory.empty:
        final_df = pd.concat([df_spheres, df_memory], ignore_index=True)
        
        # Save to current folder
        filename7 = "tradeoff_evaluations.csv"
        final_df.to_csv(f"{folder}/{filename7}", index=False)
        print(f"Saved {folder}/{filename7}")

        print("\n=== Tradeoff Evaluation Results ===")
        print(final_df.to_string(index=False))