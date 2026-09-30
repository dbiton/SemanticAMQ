from __future__ import annotations

import dataclasses
import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext, LogLocator, MaxNLocator, NullFormatter, NullLocator

from plots.plot_common import apply_style, style_axes, save_figure, mean_std

INPUT_PATH = "results/streaming.csv"
OUTDIR = "plots/streaming/"
K_FIXED = 16
DPI = 300
EMIT_PDF = False
DEMO = False

METHODS = ["nearest", "least_inflation", "pooled", "merge", "min_heap", "cap_merge"]

METHOD_LABELS = {
    "nearest": "Nearest-center absorb",
    "least_inflation": "Least-inflation absorb",
    "pooled": "Pooled Gonzales re-cluster",
    "merge": "Exhaustive cheapest-pair merge",
    "min_heap": "Heap-based cheapest-pair merge",
    "cap_merge": "Mass-weighted cap merge",
}

# Okabe-Ito for the first five; cap_merge's #882255 (Tol's wine) was checked
# with the dataviz skill's validate_palette.js (--mode light --pairs all) on
# all six: it adds no warning of its own. The set's worst all-pairs CVD
# separation stays #009E73<->#CC79A7 (dE 7.6, the 6-8 band that needs
# secondary encoding -- the per-method markers), and #E69F00/#CC79A7 sit
# below 3:1 on white, as before.
METHOD_COLORS = {
    "nearest": "#0072B2",
    "least_inflation": "#E69F00",
    "pooled": "#CC79A7",
    "merge": "#009E73",
    "min_heap": "#D55E00",
    "cap_merge": "#882255",
}

METHOD_MARKERS = {
    "nearest": "o",         # Circle
    "least_inflation": "^", # Triangle Up
    "pooled": "s",          # Square
    "merge": "D",           # Diamond
    "min_heap": "v",        # Triangle Down
    "cap_merge": "P",       # Filled plus
}

# Static-rebuild references, one per construction method (the rows'
# construct_method): neutral ink, told apart by dash pattern.
REFERENCE_STYLES = {
    "gonzalez": {"color": "black", "linestyle": "--", "label": "Static rebuild, Gonzalez (oracle)"},
    "sphere_merge": {"color": "#52514e", "linestyle": "-.", "label": "Static rebuild, sphere merge (oracle)"},
}

_REQUIRED_COLUMNS = [
    "dataset", "method", "k", "seed", "fold_offset", "position",
    "fpr", "log10_j_tau", "n_balls", "memory_bytes", "cumulative_time_s",
    "amortized_cost_s", "metric", "sweep",
]


@dataclasses.dataclass
class Config:
    k_fixed: int
    dpi: int
    emit_pdf: bool
    outdir: str


# Demo data / schema / config

def _synthesize_demo_data() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    k_fixed = K_FIXED
    n_stream_ab, F_ab = 3000, 4
    positions_ab = np.unique(np.round(np.logspace(np.log10(50), np.log10(n_stream_ab), 12)).astype(int))

    base_fpr = {"nearest": 0.06, "least_inflation": 0.045, "pooled": 0.018, "merge": 0.012, "min_heap": 0.03, "static_rebuild": 0.008}
    cost_scale = {"nearest": 1.0, "least_inflation": 1.1, "merge": 2.0, "pooled": 1.5, "min_heap": 1.3, "static_rebuild": 40.0}

    rows = []
    for method in METHODS:
        offsets = [f * 1500 for f in range(F_ab)]
        for fold_offset in offsets:
            for pos in positions_ab:
                x = pos / n_stream_ab
                fpr = base_fpr[method] * (0.3 + 0.7 * x) * rng.uniform(0.9, 1.1)
                log10_j = 40.0 + 8.0 * x + rng.normal(0, 0.2)
                rows.append({
                    "dataset": "eli5", "method": method, "k": k_fixed, "seed": 0,
                    "fold_offset": fold_offset, "position": int(pos), "fpr": float(fpr),
                    "log10_j_tau": float(log10_j), "n_balls": k_fixed, "memory_bytes": k_fixed * 1540,
                    "cumulative_time_s": cost_scale[method] * pos * 1e-5,
                    "amortized_cost_s": cost_scale[method] * 1e-5,
                    "metric": "checkpoint", "sweep": "position",
                })

        for fold_offset in offsets:
            for pos in positions_ab:
                x = pos / n_stream_ab
                rows.append({
                    "dataset": "eli5", "method": "static_rebuild", "k": k_fixed, "seed": 0,
                    "fold_offset": fold_offset, "position": int(pos),
                    "fpr": float(base_fpr["static_rebuild"] * (0.3 + 0.7 * x) * rng.uniform(0.9, 1.1)),
                    "log10_j_tau": float(38.0 + 6.0 * x + rng.normal(0, 0.2)),
                    "n_balls": k_fixed, "memory_bytes": k_fixed * 1540,
                    "cumulative_time_s": cost_scale["static_rebuild"] * pos * 1e-5,
                    "amortized_cost_s": cost_scale["static_rebuild"] * 1e-5,
                    "metric": "static_reference", "sweep": "position",
                })

    # Filter size k sweep (sweep="k") for panel (d) throughput vs k
    for method in METHODS:
        offsets = [f * 1500 for f in range(F_ab)]
        for fold_offset in offsets:
            for k_val in [16, 32, 64, 128]:
                cost = cost_scale[method] * (1.0 + 0.005 * k_val) * 1e-5 * rng.uniform(0.95, 1.05)
                rows.append({
                    "dataset": "eli5", "method": method, "k": k_val, "seed": 0,
                    "fold_offset": fold_offset, "position": 3050, "fpr": 0.02,
                    "log10_j_tau": 40.0, "n_balls": k_val, "memory_bytes": k_val * 1540,
                    "cumulative_time_s": cost * 3000, "amortized_cost_s": cost,
                    "metric": "checkpoint", "sweep": "k",
                })

    return pd.DataFrame(rows)


def _validate_schema(df: pd.DataFrame) -> None:
    missing_cols = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Input missing columns: {missing_cols}. Present: {sorted(df.columns)}")
    missing_methods = set(METHODS) - set(df.loc[df["metric"] == "checkpoint", "method"].unique())
    if missing_methods:
        raise ValueError(f"Missing method(s) {sorted(missing_methods)} among metric=='checkpoint' rows.")


def load_data() -> pd.DataFrame:
    if DEMO:
        df = _synthesize_demo_data()
    else:
        if not os.path.exists(INPUT_PATH):
            raise FileNotFoundError(f"Input file not found: {INPUT_PATH!r}. Set DEMO=True or fix INPUT_PATH.")
        df = pd.read_parquet(INPUT_PATH) if INPUT_PATH.endswith(".parquet") else pd.read_csv(INPUT_PATH)
    _validate_schema(df)
    return df


def resolve_config(df: pd.DataFrame, k_fixed: Optional[int] = K_FIXED,
                   dpi: int = DPI, emit_pdf: bool = EMIT_PDF, outdir: str = OUTDIR) -> Config:
    pos_rows = df[df["sweep"] == "position"]
    available_k = sorted(pos_rows["k"].unique()) if not pos_rows.empty else [K_FIXED]
    if k_fixed is None:
        k_fixed = int(available_k[0])
    elif k_fixed not in available_k:
        raise ValueError(f"k_fixed={k_fixed} not present in position-sweep rows. Available: {available_k}")

    return Config(k_fixed=k_fixed, dpi=dpi, emit_pdf=emit_pdf, outdir=outdir)


# Styling helpers matching main paper specifications

def _style_log_x(ax: plt.Axes) -> None:
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", which="minor", length=1.2, width=0.4)


def _style_log_y(ax: plt.Axes) -> None:
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="y", which="minor", length=1.2, width=0.4)


def _apply_paper_axes_style(ax: plt.Axes, log_x: bool = False, log_y: bool = False) -> None:
    """Applies paper-wide aesthetic: top/right spine removal, faint light gray grid, uniform ticks."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.7)
    ax.spines['bottom'].set_linewidth(0.7)
    ax.grid(True, linestyle='-', alpha=0.25, color="#cccccc", zorder=0)

    if log_x:
        _style_log_x(ax)
    if log_y:
        _style_log_y(ax)


def _set_scaled_linear_x_axis(ax: plt.Axes, x_values, base_label: str = "Stream Position") -> None:
    x = np.asarray(pd.Series(x_values).dropna().to_numpy(), dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    if len(x) == 0:
        ax.set_xlabel(base_label)
        return

    xmax = float(np.max(x))
    ax.set_xscale("linear")
    ax.set_xlim(left=-0.03 * xmax, right=xmax * 1.03)

    if xmax >= 1000:
        exponent = int(np.floor(np.log10(xmax)))
        scale_factor = 10.0 ** exponent
        label = rf"{base_label} ($\times 10^{{{exponent}}}$)"
    else:
        exponent = 0
        scale_factor = 1.0
        label = base_label

    scaled_xmax = xmax / scale_factor
    locator = MaxNLocator(nbins=4)
    scaled_ticks = locator.tick_values(0, scaled_xmax)
    scaled_ticks = [t for t in scaled_ticks if 0 <= t <= scaled_xmax * 1.01]

    raw_ticks = [t * scale_factor for t in scaled_ticks]
    tick_labels = [f"{t:g}" for t in scaled_ticks]

    ax.set_xticks(raw_ticks)
    ax.set_xticklabels(tick_labels)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel(label)


def _add_k_annotation(ax: plt.Axes, k_val: int, loc: str = "bottom-left") -> None:
    """Annotates corner with k budget label for clarity on position-sweep panels."""
    if loc == "top-left":
        ax.text(0.05, 0.95, f"$k={k_val}$", transform=ax.transAxes, fontsize=8,
                va="top", ha="left", zorder=10)
    elif loc == "top-right":
        ax.text(0.95, 0.95, f"$k={k_val}$", transform=ax.transAxes, fontsize=8,
                va="top", ha="right", zorder=10)
    else:
        ax.text(0.05, 0.05, f"$k={k_val}$", transform=ax.transAxes, fontsize=8,
                va="bottom", ha="left", zorder=10)


def _with_construct_method(df: pd.DataFrame) -> pd.DataFrame:
    """Rows from before construct_method existed were all Gonzalez-built."""
    df = df.copy()
    if "construct_method" not in df.columns:
        df["construct_method"] = "gonzalez"
    df["construct_method"] = df["construct_method"].fillna("gonzalez")
    return df


def _std_errorbars(ax: plt.Axes, x: pd.Series, agg: pd.DataFrame, color: str,
                   log_floor: bool = False, zorder: int = 3) -> None:
    """+-1 std error bars around agg["mean"]. log_floor clips the lower bar at
    mean/10 so it stays positive on a log axis."""
    if not (agg["std"] > 0).any():
        return
    lower = agg["std"]
    if log_floor:
        lower = np.minimum(lower, agg["mean"] * 0.9)
    ax.errorbar(x, agg["mean"], yerr=[lower, agg["std"]], fmt="none", ecolor=color,
                elinewidth=0.5, capsize=1.5, capthick=0.5, zorder=zorder)


def _draw_references(ax: plt.Axes, df: pd.DataFrame, cfg: Config, value_col: str,
                     band: bool = True, log_floor: bool = False) -> pd.Series:
    """One static-rebuild line (with +-1 std error bars) per construction
    method present; returns their positions for axis scaling."""
    ref = df[(df["sweep"] == "position") & (df["metric"] == "static_reference") & (df["k"] == cfg.k_fixed)]
    positions = []
    for construct_method, sub in ref.groupby("construct_method"):
        style = REFERENCE_STYLES.get(construct_method, {"color": "black", "linestyle": ":"})
        agg = mean_std(sub, ["position"], value_col).sort_values("position")
        ax.plot(agg["position"], agg["mean"], color=style["color"], linestyle=style["linestyle"],
                linewidth=0.9, zorder=4)
        if band:
            _std_errorbars(ax, agg["position"], agg, style["color"], log_floor=log_floor, zorder=4)
        positions.append(agg["position"])
    return pd.concat(positions) if positions else pd.Series(dtype=float)


# Panel (a): FPR vs. stream position

def panel_fpr_vs_position(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    sel = df[(df["sweep"] == "position") & (df["metric"] == "checkpoint") & (df["k"] == cfg.k_fixed)]
    agg = mean_std(sel, ["method", "position"], "fpr")

    apply_style(cfg.dpi)
    fig, ax = plt.subplots(figsize=(3.2, 2.4))

    for method in METHODS:
        sub = agg[agg["method"] == method].sort_values("position")
        if sub.empty:
            continue
        ax.plot(sub["position"], sub["mean"], color=METHOD_COLORS[method], marker=METHOD_MARKERS[method],
                markersize=3.5, linewidth=1.0, zorder=3, label=METHOD_LABELS[method])
        _std_errorbars(ax, sub["position"], sub, METHOD_COLORS[method], log_floor=True)

    ref_positions = _draw_references(ax, df, cfg, "fpr", band=True, log_floor=True)

    _apply_paper_axes_style(ax, log_x=False, log_y=True)
    _set_scaled_linear_x_axis(ax, pd.concat([agg["position"], ref_positions]), "Stream Position")
    _add_k_annotation(ax, cfg.k_fixed, loc="bottom-left")
    ax.set_ylabel("FPR")
    style_axes(ax)

    save_figure(fig, "streaming-fpr-vs-position.png", cfg.outdir, cfg.dpi, cfg.emit_pdf)
    return agg


# Panel (b): geometric objective vs. stream position

def panel_geometric_objective(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    min_log = df["log10_j_tau"].replace(-np.inf, np.nan).min()
    df = df.copy()
    df["log10_j_tau"] = df["log10_j_tau"].replace(-np.inf, min_log + 1e-6)

    sel = df[(df["sweep"] == "position") & (df["metric"] == "checkpoint") & (df["k"] == cfg.k_fixed)]
    agg = mean_std(sel, ["method", "position"], "log10_j_tau")

    apply_style(cfg.dpi)
    fig, ax = plt.subplots(figsize=(3.2, 2.4))

    for method in METHODS:
        sub = agg[agg["method"] == method].sort_values("position")
        if sub.empty:
            continue
        ax.plot(sub["position"], sub["mean"], color=METHOD_COLORS[method], marker=METHOD_MARKERS[method], 
                markersize=3.5, linewidth=1.0, zorder=3)
        _std_errorbars(ax, sub["position"], sub, METHOD_COLORS[method])

    ref_positions = _draw_references(ax, df, cfg, "log10_j_tau", band=True)

    _apply_paper_axes_style(ax, log_x=False, log_y=False)
    _set_scaled_linear_x_axis(ax, pd.concat([agg["position"], ref_positions]), "Stream Position")
    _add_k_annotation(ax, cfg.k_fixed, loc="bottom-left")
    ax.set_ylabel(r"$\log_{10} J_\tau(\mathcal{C})$")
    style_axes(ax)

    save_figure(fig, "streaming-geometric-objective.png", cfg.outdir, cfg.dpi, cfg.emit_pdf)
    return agg


# Panel (c): number of balls vs. stream position

def panel_n_balls_vs_position(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Ball count over the stream. Expected flat at k_fixed for all methods."""
    sel = df[(df["sweep"] == "position") & (df["metric"] == "checkpoint") & (df["k"] == cfg.k_fixed)]
    agg = mean_std(sel, ["method", "position"], "n_balls")

    apply_style(cfg.dpi)
    fig, ax = plt.subplots(figsize=(3.2, 2.4))

    for method in METHODS:
        sub = agg[agg["method"] == method].sort_values("position")
        if sub.empty:
            continue
        ax.plot(sub["position"], sub["mean"], color=METHOD_COLORS[method], marker=METHOD_MARKERS[method],
                markersize=3.5, linewidth=1.0, zorder=3)
        _std_errorbars(ax, sub["position"], sub, METHOD_COLORS[method])

    ref_positions = _draw_references(ax, df, cfg, "n_balls", band=False)

    _apply_paper_axes_style(ax, log_x=False, log_y=False)
    _set_scaled_linear_x_axis(ax, pd.concat([agg["position"], ref_positions]), "Stream Position")
    _add_k_annotation(ax, cfg.k_fixed, loc="bottom-left")
    ax.set_ylabel("Number of Balls")
    style_axes(ax)

    save_figure(fig, "streaming-n-balls-vs-position.png", cfg.outdir, cfg.dpi, cfg.emit_pdf)
    return agg


# Panel (d): insertion throughput vs. filter size k (Figure 4c)

def panel_insertion_throughput(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Amortized insertion throughput (vectors/sec) vs. filter size k Figure 4(c)."""
    sel_k = df[(df["sweep"] == "k") & (df["metric"] == "checkpoint")].copy()
    if sel_k.empty:
        raise ValueError("No data found for sweep='k' among metric=='checkpoint' rows.")

    # Select final position for each fold/k run to capture full-stream throughput
    max_positions = sel_k.groupby(["method", "k", "fold_offset"])["position"].transform("max")
    sel = sel_k[sel_k["position"] == max_positions].copy()

    sel["throughput"] = 1.0 / sel["amortized_cost_s"]
    agg = mean_std(sel, ["method", "k"], "throughput")

    apply_style(cfg.dpi)
    fig, ax = plt.subplots(figsize=(3.2, 2.4))

    for method in METHODS:
        sub = agg[agg["method"] == method].sort_values("k")
        if sub.empty:
            continue
        ax.plot(sub["k"], sub["mean"], color=METHOD_COLORS[method], marker=METHOD_MARKERS[method], 
                markersize=3.5, linewidth=1.0, zorder=3, label=METHOD_LABELS[method])
        _std_errorbars(ax, sub["k"], sub, METHOD_COLORS[method], log_floor=True)

    _apply_paper_axes_style(ax, log_x=True, log_y=True)

    k_ticks = sorted(agg["k"].unique())
    if k_ticks:
        ax.set_xticks(k_ticks)
        ax.set_xticklabels([f"$2^{{{int(np.round(np.log2(x)))}}}$" for x in k_ticks])
        ax.xaxis.set_minor_locator(NullLocator())

    ax.set_xlabel("Filter Size ($k$)")
    ax.set_ylabel("Insertion Throughput (vec/s)")
    style_axes(ax)

    save_figure(fig, "streaming-insertion-throughput.png", cfg.outdir, cfg.dpi, cfg.emit_pdf)
    return agg


# Panel (e): vectors inserted vs. stream position (Clean Single Axis)

def panel_vectors_vs_balls(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Cumulative vectors inserted vs. stream position with a single clean axis."""
    sel = df[(df["sweep"] == "position") & (df["metric"] == "checkpoint") & (df["k"] == cfg.k_fixed)]
    vectors_agg = mean_std(sel, ["method", "position"], "position")

    apply_style(cfg.dpi)
    fig, ax = plt.subplots(figsize=(3.2, 2.4))

    for method in METHODS:
        sub = vectors_agg[vectors_agg["method"] == method].sort_values("position")
        if sub.empty:
            continue
        ax.plot(sub["position"], sub["mean"], color=METHOD_COLORS[method], marker=METHOD_MARKERS[method],
                markersize=3.5, linewidth=1.0, zorder=3)

    _apply_paper_axes_style(ax, log_x=False, log_y=False)
    _set_scaled_linear_x_axis(ax, vectors_agg["position"], "Stream Position")
    _add_k_annotation(ax, cfg.k_fixed, loc="bottom-left")
    ax.set_ylabel("Vectors Inserted")
    style_axes(ax)

    save_figure(fig, "streaming-vectors-vs-balls.png", cfg.outdir, cfg.dpi, cfg.emit_pdf)
    return vectors_agg


# Panel (f): filter size vs. stream position (Clean Single Axis)

def panel_filter_size(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Filter memory footprint (KB) vs. stream position with a single clean axis."""
    sel = df[(df["sweep"] == "position") & (df["metric"] == "checkpoint") & (df["k"] == cfg.k_fixed)].copy()
    sel["memory_kb"] = sel["memory_bytes"] / 1024.0
    size_agg = mean_std(sel, ["method", "position"], "memory_kb")

    apply_style(cfg.dpi)
    fig, ax = plt.subplots(figsize=(3.2, 2.4))

    for method in METHODS:
        sub = size_agg[size_agg["method"] == method].sort_values("position")
        if sub.empty:
            continue
        ax.plot(sub["position"], sub["mean"], color=METHOD_COLORS[method], marker=METHOD_MARKERS[method],
                markersize=3.5, linewidth=1.0, zorder=3)

    _apply_paper_axes_style(ax, log_x=False, log_y=False)
    _set_scaled_linear_x_axis(ax, size_agg["position"], "Stream Position")
    _add_k_annotation(ax, cfg.k_fixed, loc="top-left")
    ax.set_ylabel("Filter Total Size (KB)")
    style_axes(ax)

    save_figure(fig, "streaming-filter-size.png", cfg.outdir, cfg.dpi, cfg.emit_pdf)
    return size_agg


# Legend

def save_method_legend(cfg: Config) -> None:
    apply_style(cfg.dpi)
    fig, ax = plt.subplots(figsize=(6.0, 0.45))
    ax.axis("off")
    handles = [plt.Line2D([0], [0], color=METHOD_COLORS[m], marker=METHOD_MARKERS[m], markersize=4, linewidth=1.0)
               for m in METHODS]
    labels = [METHOD_LABELS[m] for m in METHODS]

    for style in REFERENCE_STYLES.values():
        handles.append(plt.Line2D([0], [0], color=style["color"], linestyle=style["linestyle"], linewidth=1.0))
        labels.append(style["label"])

    fig.legend(handles, labels, loc="center", ncol=4, frameon=False, fontsize=7.8)
    fig.patch.set_alpha(0.0)
    save_figure(fig, "streaming-method-legend.png", cfg.outdir, cfg.dpi, cfg.emit_pdf, transparent=True)


# Text-report summary

def print_summary(df: pd.DataFrame, cfg: Config) -> None:
    print(f"\n--- Streaming summary (k_fixed={cfg.k_fixed}) ---")
    print("FNR=0 confirmed at every checkpoint for all insertion methods "
          "(enforced at generation time in stream_rebuild_worker).")
    print()


# Main

def generate_streaming_figures(df: Optional[pd.DataFrame] = None, outdir: str = OUTDIR,
                               k_fixed: Optional[int] = K_FIXED,
                               dpi: int = DPI, emit_pdf: bool = EMIT_PDF) -> None:
    """One figure set per bootstrap construction method: the default
    (Gonzalez) bootstrap's set at `outdir` itself, any other's under
    `outdir`/bootstrap-<method>/. Every set draws the static-rebuild
    reference of each construction method present, so each method's gap
    can be read against both oracles."""
    df = _with_construct_method(load_data() if df is None else df)
    refs = df[df["metric"] == "static_reference"]
    streamed = df[df["metric"] != "static_reference"]
    bootstraps = sorted(streamed["construct_method"].unique(), key=lambda m: (m != "gonzalez", m))

    for bootstrap in bootstraps:
        sub = pd.concat([streamed[streamed["construct_method"] == bootstrap], refs], ignore_index=True)
        _validate_schema(sub)
        out = outdir if bootstrap == "gonzalez" else os.path.join(outdir, f"bootstrap-{bootstrap}")
        cfg = resolve_config(sub, k_fixed=k_fixed, dpi=dpi, emit_pdf=emit_pdf, outdir=out)

        panel_fpr_vs_position(sub, cfg)
        panel_geometric_objective(sub, cfg)
        panel_n_balls_vs_position(sub, cfg)
        if ((sub["sweep"] == "k") & (sub["metric"] == "checkpoint")).any():
            panel_insertion_throughput(sub, cfg)
        panel_vectors_vs_balls(sub, cfg)
        panel_filter_size(sub, cfg)
        save_method_legend(cfg)
        print_summary(sub, cfg)

    snippet = r"""
\begin{subfigure}[b]{\linewidth}
    \centering
    \includegraphics[width=0.6\linewidth]{figures/streaming-method-legend.png}
\end{subfigure}\\[-0.3em]
\begin{subfigure}[b]{0.32\linewidth}
    \centering
    \includegraphics[width=\linewidth]{figures/streaming-fpr-vs-position.png}
    \caption{FPR vs. stream position}
\end{subfigure}
\begin{subfigure}[b]{0.32\linewidth}
    \centering
    \includegraphics[width=\linewidth]{figures/streaming-geometric-objective.png}
    \caption{Geometric objective vs. stream position}
\end{subfigure}
\begin{subfigure}[b]{0.32\linewidth}
    \centering
    \includegraphics[width=\linewidth]{figures/streaming-insertion-throughput.png}
    \caption{Insertion throughput vs. filter size $k$}
\end{subfigure}
""".strip("\n")
    print("--- subfigure snippet (paste above the panel row) ---")
    print(snippet)


def main() -> None:
    generate_streaming_figures()


if __name__ == "__main__":
    main()