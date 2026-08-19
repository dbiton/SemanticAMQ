import math
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List

from plots.plot_common import apply_style, style_axes, save_figure, wrap_label

DPI = 300
GRID_FIGSIZE = (3.4, 3.0)  # 2x2 grids: two paper-style panels per side
LINEWIDTH = 0.8
MARKERSIZE = 2.5
LEGEND_LINEWIDTH = 1.0
LEGEND_MARKERSIZE = 3


def _legend_figsize(n_items: int, ncol: int) -> Tuple[float, float]:
    """Fixed-width legend strip whose height grows with the number of rows
    needed to fit n_items at ncol columns, keeping single-row legends the
    same size across every plot script."""
    rows = max(1, math.ceil(n_items / max(1, ncol)))
    return (4.5, 0.28 * rows + 0.05)


def plot_results(results_list: list, folder: str) -> None:
    df = pd.DataFrame(results_list)
    df[['Filter', 'LSH']] = df['Config'].str.split(' + ', regex=False, expand=True)

    apply_style(DPI)
    outdir = f"plots/{folder}"

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
        fig, ax = plt.subplots()
        style_axes(ax)
        if len(memory_budgets) > 1:
            sns.lineplot(
                data=df, x='Memory Budget (MB)', y=metric,
                hue='Percentage', style='Filter',
                markers=True, dashes=False, linewidth=LINEWIDTH, markersize=MARKERSIZE,
                palette=pct_palette, ax=ax,
            )
            ax.legend(ncol=2)
            ax.set_xlabel(wrap_label('Memory Budget (MB)'))
        else:
            sns.lineplot(
                data=df, x='Percentage', y=metric,
                hue='Filter', style='LSH',
                markers=marker_map, dashes=False, linewidth=LINEWIDTH, markersize=MARKERSIZE,
                palette=color_map, ax=ax, legend=False,
            )
            ax.set_xlabel(wrap_label('Percentage of Duplicates (%)'))
        ax.set_ylabel(wrap_label(ylabels.get(metric, metric)))
        if metric in bounded_metrics:
            ax.set_ylim(-0.05, 1.05)
        filename = f'{metric.lower().replace("-", "_").replace(" ", "_")}_plot.png'
        save_figure(fig, filename, outdir, DPI)

    fig_leg, ax_leg = plt.subplots(figsize=_legend_figsize(len(configs), min(3, len(configs))))
    ax_leg.axis('off')
    handles = [
        plt.Line2D([0], [0], color=color_map[filter_name], lw=LEGEND_LINEWIDTH, marker=marker_map[lsh_name],
                  markersize=LEGEND_MARKERSIZE)
        for filter_name, lsh_name in (c.split(' + ') for c in configs)
    ]
    ax_leg.legend(
        handles, configs,
        loc='center', ncol=min(3, len(configs)),
        title=wrap_label("Configuration (Color = Filter, Marker = LSH)"),
    )
    save_figure(fig_leg, 'legend_only.png', outdir, DPI)

    _plot_radius_stats(df, len(memory_budgets) > 1, folder)


_RADIUS_COLUMNS = ['Radius Mean', 'Radius Max', 'Radius Min', 'Radius Std']


def _plot_radius_error_bars(
    df: pd.DataFrame, x_col: str, xlabel: str, out_folder: str, filename: str, group_col: str = None,
) -> None:
    """
    Single combined plot of sphere-radius statistics reported by geometric
    filters like BallCoverFilter: Mean is drawn as a line with error bars
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

    fig, ax = plt.subplots()
    style_axes(ax)
    for group in groups:
        sub = plot_df if group is None else plot_df[plot_df[group_col] == group]
        agg = sub.groupby(x_col)[_RADIUS_COLUMNS].mean().sort_index()
        if agg.empty:
            continue
        color = palette[group]
        prefix = f"{group} - " if group is not None else ""
        ax.errorbar(
            agg.index, agg['Radius Mean'], yerr=agg['Radius Std'],
            fmt='-o', color=color, capsize=1.5, linewidth=LINEWIDTH, markersize=MARKERSIZE,
            label=f"{prefix}Mean ± Std",
        )
        ax.plot(agg.index, agg['Radius Max'], color=color, alpha=0.6, marker='^', linestyle='--',
                linewidth=LINEWIDTH, markersize=MARKERSIZE, label=f"{prefix}Max")
        ax.plot(agg.index, agg['Radius Min'], color=color, alpha=0.6, marker='v', linestyle='--',
                linewidth=LINEWIDTH, markersize=MARKERSIZE, label=f"{prefix}Min")

    ax.set_xlabel(wrap_label(xlabel))
    ax.set_ylabel('Sphere Radius')
    ax.legend(loc='best')
    save_figure(fig, filename, out_folder, DPI)


def _plot_radius_stats(df: pd.DataFrame, vs_memory: bool, folder: str) -> None:
    x_col = 'Memory Budget (MB)' if vs_memory else 'Percentage'
    xlabel = 'Memory Budget (MB)' if vs_memory else 'Percentage of Duplicates (%)'
    _plot_radius_error_bars(df, x_col, xlabel, f"plots/{folder}", 'radius_stats_plot.png')


def plot_throughput_results(results_list: list, folder: str, ylabel: str, title_suffix: str) -> None:
    df = pd.DataFrame(results_list)
    df[['Filter', 'LSH']] = df['Config'].str.split(' + ', regex=False, expand=True)

    apply_style(DPI)
    outdir = f"plots/{folder}"

    unique_filters = df['Filter'].unique()
    color_map = dict(zip(unique_filters, sns.color_palette("husl", len(unique_filters))))

    unique_lshs     = df['LSH'].unique()
    available_marks = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'X']
    marker_map      = dict(zip(unique_lshs, available_marks[:len(unique_lshs)]))

    configs = df['Config'].unique()

    fig, ax = plt.subplots()
    style_axes(ax)
    sns.lineplot(
        data=df, x='FilterSize', y='Throughput',
        hue='Filter', style='LSH',
        markers=marker_map, dashes=False, linewidth=LINEWIDTH, markersize=MARKERSIZE,
        palette=color_map, ax=ax, legend=False,
    )
    ax.set_xlabel(wrap_label('Number of Elements in Filter'))
    ax.set_ylabel(wrap_label(ylabel))
    ax.set_title(wrap_label(f'Throughput vs Filter Size ({title_suffix})'))
    filename = f'throughput_{title_suffix.lower().replace(" ", "_")}_plot.png'
    save_figure(fig, filename, outdir, DPI)

    fig_leg, ax_leg = plt.subplots(figsize=_legend_figsize(len(configs), min(3, len(configs))))
    ax_leg.axis('off')
    handles = [
        plt.Line2D([0], [0], color=color_map[filter_name], lw=LEGEND_LINEWIDTH, marker=marker_map[lsh_name],
                  markersize=LEGEND_MARKERSIZE)
        for filter_name, lsh_name in (c.split(' + ') for c in configs)
    ]
    ax_leg.legend(
        handles, configs,
        loc='center', ncol=min(3, len(configs)),
        title=wrap_label("Configuration (Color = Filter, Marker = LSH)"),
    )
    save_figure(fig_leg, 'throughput_legend_only.png', outdir, DPI)


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


def plot_system_results(results_list: list, folder: str = "plots/system") -> None:
    """
    Plots the flat-index filtered-access evaluation aggregated per dataset: mean
    throughput, mean per-query latency, and mean index accesses of the
    query stream, comparing the unfiltered baseline against each filter config
    placed in front of the index. Bars are means over the memory-budget sweep;
    whiskers show one standard deviation across it. Also plots index accesses,
    latency, and throughput (with filter) as a function of memory budget,
    one line per dataset.
    """
    apply_style(DPI)
    df = pd.DataFrame(results_list)

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
        ('Total Queries', 'Index Accesses', 'Mean Index Accesses', 'index_accesses'),
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

        fig, ax = plt.subplots()
        style_axes(ax)
        sns.barplot(
            data=long_df, x='Dataset', y=ylabel, hue='Pipeline',
            order=datasets, hue_order=pipelines, palette=palette,
            errorbar='sd', capsize=0.15, err_kws={'linewidth': LINEWIDTH}, ax=ax,
        )
        ax.set_xlabel('Dataset')
        ax.set_ylabel(wrap_label(ylabel))
        # above the axes so it can never sit on top of a bar
        ax.legend(
            title=None, loc='lower center',
            bbox_to_anchor=(0.5, 1.02), ncol=min(3, len(pipelines)),
        )
        filename = f'system_{name}_plot.png'
        save_figure(fig, filename, folder, DPI)

    vs_memory_metrics = [
        ('Index Accesses', 'Index Accesses', 'index_accesses'),
        ('Latency With Filter (ms)', 'Mean Latency per Query (ms)', 'latency'),
        ('Throughput With Filter', 'Mean Throughput (queries/s)', 'throughput'),
    ]
    for col, ylabel, name in vs_memory_metrics:
        _plot_metric_vs_memory(df, datasets, col, ylabel, name, folder)

    _plot_radius_stats_system(df, datasets, folder)
    _plot_fpvr_vs_memory(df, datasets, folder)


def _plot_radius_stats_system(df: pd.DataFrame, datasets: List[str], folder: str) -> None:
    if 'Memory Budget' not in df.columns or df['Memory Budget'].nunique() <= 1:
        return
    plot_df = df.copy()
    plot_df['Memory Budget (MB)'] = plot_df['Memory Budget'] / (1024 * 1024)
    group_col = 'Dataset' if len(datasets) > 1 else None
    _plot_radius_error_bars(
        plot_df, 'Memory Budget (MB)', 'Memory Budget (MB)', folder, 'system_radius_stats_plot.png',
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

    fig, ax = plt.subplots()
    style_axes(ax)
    sns.lineplot(
        data=plot_df, x='Memory Budget (MB)', y=col,
        hue='Dataset', style='Config' if len(configs) > 1 else None,
        hue_order=datasets, palette=palette,
        markers=True, dashes=False, linewidth=LINEWIDTH, markersize=MARKERSIZE, ax=ax,
    )
    ax.set_xlabel(wrap_label('Memory Budget (MB)'))
    ax.set_ylabel(wrap_label(ylabel))
    ax.legend(loc='best')
    filename = f'system_{name}_vs_memory_plot.png'
    save_figure(fig, filename, folder, DPI)


def _plot_fpvr_vs_memory(df: pd.DataFrame, datasets: List[str], folder: str) -> None:
    """
    Plots the volume-based False Positive Volume Ratio (FPVR = log(V_filter /
    V_original)) against memory budget, one line per dataset (styled by
    config when more than one is swept). FPVR estimates how many (natural-log)
    times larger a geometric filter's covered hypersphere volume is than the
    true duplicate region, using hypersphere-volume lower/upper bounds as a
    ground-truth-free stand-in for false-positive rate (system_worker has no
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

    fig, ax = plt.subplots()
    style_axes(ax)
    for dataset in datasets:
        color = palette[dataset]
        for config in configs:
            sub = plot_df[(plot_df['Dataset'] == dataset) & (plot_df['Config'] == config)]
            sub = sub.sort_values('Memory Budget (MB)')
            if sub.empty:
                continue
            label = dataset if len(configs) <= 1 else f"{dataset} ({config})"
            mid = (sub['FPVR Lower'] + sub['FPVR Upper']) / 2
            ax.plot(sub['Memory Budget (MB)'], mid, color=color, marker='o', linewidth=LINEWIDTH,
                    markersize=MARKERSIZE, label=label)
            ax.fill_between(sub['Memory Budget (MB)'], sub['FPVR Lower'], sub['FPVR Upper'],
                             color=color, alpha=0.2, linewidth=0)

    ax.set_xlabel(wrap_label('Memory Budget (MB)'))
    ax.set_ylabel(wrap_label('False Positive Volume Ratio (log[V_filter / V_original])'))
    ax.legend(loc='best')
    filename = 'system_fpvr_vs_memory_plot.png'
    save_figure(fig, filename, folder, DPI)


def plot_equal_memory_results(results_list: list, folder: str = "plots/equal_memory") -> None:
    """
    Plots the equal-memory comparison: every filter is evaluated at the same
    fixed set of memory budgets (see main.py::generate_equal_memory), each
    having independently picked its best available configuration on a
    calibration workload. One line plot per metric (FPR, FNR, latency,
    throughput), x-axis = memory budget (log scale), one line per filter
    config; the legend is a separate, shared image rather than embedded in
    each panel.
    """
    apply_style(DPI)
    df = pd.DataFrame(results_list)

    df['Memory Budget (MB)'] = (df['Memory Budget'] / (1024 * 1024)).round(4)
    configs = list(dict.fromkeys(df['Config']))
    color_map, marker_map, dash_map = get_plot_mappings(configs)

    metrics = [
        ('FPR', 'False Positive Rate (FPR)', 'fpr'),
        ('FNR', 'False Negative Rate (FNR)', 'fnr'),
        ('Latency (ms)', 'Query Latency (ms)', 'latency'),
        ('Throughput (qps)', 'Query Throughput (queries/s)', 'throughput'),
    ]
    for col, ylabel, name in metrics:
        fig, ax = plt.subplots()
        style_axes(ax)
        sns.lineplot(
            data=df, x='Memory Budget (MB)', y=col, hue='Config', style='Config',
            markers=marker_map, dashes=dash_map, palette=color_map,
            linewidth=LINEWIDTH, markersize=MARKERSIZE, ax=ax, legend=False,
        )
        ax.set_xscale('log', base=2)
        ax.set_xlabel(wrap_label('Memory Budget (MB)'))
        ax.set_ylabel(wrap_label(ylabel))
        if col in ('FPR', 'FNR'):
            ax.set_ylim(-0.05, 1.05)
        else:
            ax.set_yscale('log')
        filename = f'equal_memory_{name}_plot.png'
        save_figure(fig, filename, folder, DPI)

    fig_leg, ax_leg = plt.subplots(figsize=_legend_figsize(len(configs), min(3, len(configs))))
    ax_leg.axis('off')
    handles = [
        plt.Line2D(
            [0], [0], color=color_map[config], marker=marker_map[config],
            dashes=dash_map[config], lw=LEGEND_LINEWIDTH, markersize=LEGEND_MARKERSIZE,
        )
        for config in configs
    ]
    ax_leg.legend(
        handles, configs,
        loc='center', ncol=min(3, len(configs)),
        title="Configuration",
    )
    save_figure(fig_leg, 'legend_only.png', folder, DPI)

    csv_path = f"{folder}/equal_memory_evaluations.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")
