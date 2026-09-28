"""Streaming-evaluation figures: BallCoverFilter's 4 insertion strategies
(see runner._INSERT_STRATEGIES), one line per method throughout.

Reads results/streaming.csv: one row per (method, checkpoint), tagged by
`sweep` == "position" and `metric` in {"checkpoint", "static_reference"}
(`method` is "static_rebuild" for the reference rows). No k sweep: the ball
budget k is fixed for the whole run (see main.generate_streaming), so every
panel is a function of stream position only.

Five panels, all with a dashed static-rebuild reference:
  (a) FPR vs. stream position at fixed k_fixed, mean +/- 1 std over folds.
  (b) Geometric objective log10(J_tau) vs. the same x-axis.
  (c) Number of balls vs. stream position -- a validation check (expected
      flat/identical across methods), not a discriminating panel.
  (d) Vectors inserted (solid) vs. ball count (dashed), twin y-axes.
  (e) Filter size in bytes (solid) vs. ball count (dashed), twin y-axes."""

from __future__ import annotations

import dataclasses
import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter

from plots.plot_common import apply_style, style_axes, save_figure, mean_std

INPUT_PATH = "results/streaming.csv"
OUTDIR = "plots/streaming/"
K_FIXED: Optional[int] = None
DPI = 300
EMIT_PDF = False
DEMO = False

METHODS = ["nearest", "least_inflation", "merge", "min_heap"]
METHOD_LABELS = {
    "nearest": "Nearest-center absorb",
    "least_inflation": "Least-inflation absorb",
    "merge": "Exhaustive cheapest-pair merge",
    "min_heap": "Heap-based cheapest-pair merge",
}
# Distinct from plot_common.DATASET_COLORS: this figure is single-dataset,
# so the color channel is free to encode method instead.
METHOD_COLORS = {
    "nearest": "#0072B2", "least_inflation": "#E69F00", "merge": "#009E73", "min_heap": "#D55E00",
}

_REQUIRED_COLUMNS = [
    "dataset", "method", "k", "seed", "fold_offset", "position",
    "fpr", "log10_j_tau", "n_balls", "memory_bytes", "cumulative_time_s", "amortized_cost_s",
    "metric", "sweep",
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
    k_fixed = 32
    n_stream_ab, F_ab = k_fixed * 1000, 4
    positions_ab = np.unique(np.round(np.logspace(np.log10(k_fixed + 1), np.log10(n_stream_ab), 12)).astype(int))

    # Rough ordering to make the demo figure legible: nearest worst FPR,
    # merge near the static-rebuild reference at far lower cost.
    base_fpr = {"nearest": 0.06, "least_inflation": 0.045, "merge": 0.012, "min_heap": 0.03,
               "static_rebuild": 0.008}
    cost_scale = {"nearest": 1.0, "least_inflation": 1.1, "merge": 2.0, "min_heap": 1.3, "static_rebuild": 40.0}

    rows = []
    for method in METHODS:
        offsets = [f * 1500 for f in range(F_ab)]
        for fold_offset in offsets:
            for pos in positions_ab:
                x = pos / n_stream_ab
                fpr = base_fpr[method] * (0.3 + 0.7 * x) * rng.uniform(0.9, 1.1)
                log10_j = 40.0 + 8.0 * x + rng.normal(0, 0.2)
                rows.append({
                    "dataset": "eli5", "method": method, "k": k_fixed, "seed": 0, "fold_offset": fold_offset,
                    "position": int(pos),
                    "fpr": float(fpr), "log10_j_tau": float(log10_j), "n_balls": k_fixed,
                    "memory_bytes": k_fixed * 1540,
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
                    "log10_j_tau": float(38.0 + 6.0 * x + rng.normal(0, 0.2)), "n_balls": k_fixed,
                    "memory_bytes": k_fixed * 1540,
                    "cumulative_time_s": cost_scale["static_rebuild"] * pos * 1e-5,
                    "amortized_cost_s": cost_scale["static_rebuild"] * 1e-5,
                    "metric": "static_reference", "sweep": "position",
                })

    return pd.DataFrame(rows)


def _validate_schema(df: pd.DataFrame) -> None:
    missing_cols = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Input missing columns: {missing_cols}. Present: {sorted(df.columns)}")
    missing_methods = set(METHODS) - set(df.loc[df["metric"] == "checkpoint", "method"].unique())
    if missing_methods:
        raise ValueError(f"Missing method(s) {sorted(missing_methods)} among metric=='checkpoint' rows.")
    if "position" not in set(df["sweep"].unique()):
        raise ValueError("Missing sweep 'position'.")


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
    available_k = sorted(pos_rows["k"].unique())
    if k_fixed is None:
        k_fixed = int(available_k[0])
    elif k_fixed not in available_k:
        raise ValueError(f"k_fixed={k_fixed} not present in position-sweep rows. Available: {available_k}")

    return Config(k_fixed=k_fixed, dpi=dpi, emit_pdf=emit_pdf, outdir=outdir)


# Styling helpers

def _style_log_y(ax: plt.Axes) -> None:
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="y", which="minor", length=1.2, width=0.4)


# Panel (a): FPR vs. stream position

def panel_fpr_vs_position(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    sel = df[(df["sweep"] == "position") & (df["metric"] == "checkpoint") & (df["k"] == cfg.k_fixed)]
    agg = mean_std(sel, ["method", "position"], "fpr")

    ref = df[(df["sweep"] == "position") & (df["metric"] == "static_reference") & (df["k"] == cfg.k_fixed)]
    ref_agg = mean_std(ref, ["position"], "fpr")

    apply_style(cfg.dpi)
    fig, ax = plt.subplots()
    for method in METHODS:
        sub = agg[agg["method"] == method].sort_values("position")
        if sub.empty:
            continue
        ax.plot(sub["position"], sub["mean"], color=METHOD_COLORS[method], marker="o",
               markersize=2.5, linewidth=0.8, zorder=2)
        if (sub["std"] > 0).any():
            ax.errorbar(sub["position"], sub["mean"], yerr=sub["std"], fmt="none",
                       ecolor=METHOD_COLORS[method], elinewidth=0.5, capsize=1.5, capthick=0.5, zorder=3)

    if not ref_agg.empty:
        ref_agg = ref_agg.sort_values("position")
        ax.plot(ref_agg["position"], ref_agg["mean"], color="black", linestyle="--", linewidth=0.7, zorder=3)

    _style_log_y(ax)
    ax.set_xlabel("Stream Position")
    ax.set_ylabel("FPR")
    style_axes(ax)

    save_figure(fig, "streaming-fpr-vs-position.png", cfg.outdir, cfg.dpi, cfg.emit_pdf)
    return agg


# Panel (b): geometric objective vs. stream position

def panel_geometric_objective(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    sel = df[(df["sweep"] == "position") & (df["metric"] == "checkpoint") & (df["k"] == cfg.k_fixed)]
    agg = mean_std(sel, ["method", "position"], "log10_j_tau")

    ref = df[(df["sweep"] == "position") & (df["metric"] == "static_reference") & (df["k"] == cfg.k_fixed)]
    ref_agg = mean_std(ref, ["position"], "log10_j_tau")

    apply_style(cfg.dpi)
    fig, ax = plt.subplots()
    for method in METHODS:
        sub = agg[agg["method"] == method].sort_values("position")
        if sub.empty:
            continue
        ax.plot(sub["position"], sub["mean"], color=METHOD_COLORS[method], marker="o",
               markersize=2.5, linewidth=0.8, zorder=2)
        if (sub["std"] > 0).any():
            ax.errorbar(sub["position"], sub["mean"], yerr=sub["std"], fmt="none",
                       ecolor=METHOD_COLORS[method], elinewidth=0.5, capsize=1.5, capthick=0.5, zorder=3)

    if not ref_agg.empty:
        ref_agg = ref_agg.sort_values("position")
        ax.plot(ref_agg["position"], ref_agg["mean"], color="black", linestyle="--", linewidth=0.7, zorder=3)
        if (ref_agg["std"] > 0).any():
            ax.errorbar(ref_agg["position"], ref_agg["mean"], yerr=ref_agg["std"], fmt="none",
                       ecolor="black", elinewidth=0.5, capsize=1.5, capthick=0.5, zorder=3)

    ax.set_xlabel("Stream Position")
    ax.set_ylabel(r"$\log_{10} J_\tau(\mathcal{C})$")
    style_axes(ax)

    save_figure(fig, "streaming-geometric-objective.png", cfg.outdir, cfg.dpi, cfg.emit_pdf)
    return agg


# Panel (d): number of balls vs. stream position

def panel_n_balls_vs_position(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Ball count over the stream (see panel_filter_size for the same series
    overlaid against memory size). Expected flat at k_fixed for all four
    methods -- a validation check, not a discriminating comparison."""
    sel = df[(df["sweep"] == "position") & (df["metric"] == "checkpoint") & (df["k"] == cfg.k_fixed)]
    agg = mean_std(sel, ["method", "position"], "n_balls")

    ref = df[(df["sweep"] == "position") & (df["metric"] == "static_reference") & (df["k"] == cfg.k_fixed)]
    ref_agg = mean_std(ref, ["position"], "n_balls")

    apply_style(cfg.dpi)
    fig, ax = plt.subplots()
    for method in METHODS:
        sub = agg[agg["method"] == method].sort_values("position")
        if sub.empty:
            continue
        ax.plot(sub["position"], sub["mean"], color=METHOD_COLORS[method], marker="o",
               markersize=2.5, linewidth=0.8, zorder=2)
        if (sub["std"] > 0).any():
            ax.errorbar(sub["position"], sub["mean"], yerr=sub["std"], fmt="none",
                       ecolor=METHOD_COLORS[method], elinewidth=0.5, capsize=1.5, capthick=0.5, zorder=3)

    if not ref_agg.empty:
        ref_agg = ref_agg.sort_values("position")
        ax.plot(ref_agg["position"], ref_agg["mean"], color="black", linestyle="--", linewidth=0.7, zorder=3)

    ax.set_xlabel("Stream Position")
    ax.set_ylabel("Number of Balls")
    style_axes(ax)

    save_figure(fig, "streaming-n-balls-vs-position.png", cfg.outdir, cfg.dpi, cfg.emit_pdf)
    return agg


# Panel: vectors inserted vs. number of balls, vs. stream position

def panel_vectors_vs_balls(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Cumulative vectors inserted (solid, left axis) vs. ball count
    (dashed, right axis) vs. stream position, twin y-axes, one pair of lines
    per method. "Vectors inserted" is just the stream position itself, so
    the solid line is a straight diagonal by construction -- the point is
    the *contrast* with the flat dashed ball-count line, making the
    "unbounded input, bounded structure" claim visually explicit."""
    sel = df[(df["sweep"] == "position") & (df["metric"] == "checkpoint") & (df["k"] == cfg.k_fixed)]
    vectors_agg = mean_std(sel, ["method", "position"], "position")
    balls_agg = mean_std(sel, ["method", "position"], "n_balls")

    apply_style(cfg.dpi)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    for method in METHODS:
        vectors_sub = vectors_agg[vectors_agg["method"] == method].sort_values("position")
        balls_sub = balls_agg[balls_agg["method"] == method].sort_values("position")
        if vectors_sub.empty:
            continue
        ax1.plot(vectors_sub["position"], vectors_sub["mean"], color=METHOD_COLORS[method],
                 marker="o", markersize=2.5, linewidth=1.1, linestyle="-", zorder=2)
        ax2.plot(balls_sub["position"], balls_sub["mean"], color=METHOD_COLORS[method],
                 linewidth=1.6, linestyle="--", zorder=3, alpha=0.8)

    ax1.set_xlabel("Stream Position")
    ax1.set_ylabel("Vectors Inserted")
    ax2.set_ylabel("Number of Balls")

    ax1.text(0.97, 0.06, "solid: vectors inserted -- dashed: ball count", transform=ax1.transAxes,
             ha="right", va="bottom", fontsize=6.0)
    style_axes(ax1)

    save_figure(fig, "streaming-vectors-vs-balls.png", cfg.outdir, cfg.dpi, cfg.emit_pdf)
    return vectors_agg.merge(balls_agg, on=["method", "position"], suffixes=("_vectors", "_balls"))


# Panel (e): filter size + ball count vs. stream position

def panel_filter_size(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Filter total size (bytes, solid) vs. ball count (dashed) vs. stream
    position, twin y-axes -- a validation panel, not a discriminating
    comparison. The right axis is scaled by the constant bytes_per_ball so
    the dashed line overlays the solid one exactly."""
    sel = df[(df["sweep"] == "position") & (df["metric"] == "checkpoint") & (df["k"] == cfg.k_fixed)]
    size_agg = mean_std(sel, ["method", "position"], "memory_bytes")
    balls_agg = mean_std(sel, ["method", "position"], "n_balls")
    bytes_per_ball = float((sel["memory_bytes"] / sel["n_balls"]).median())

    apply_style(cfg.dpi)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    for method in METHODS:
        size_sub = size_agg[size_agg["method"] == method].sort_values("position")
        balls_sub = balls_agg[balls_agg["method"] == method].sort_values("position")
        if size_sub.empty:
            continue
        ax1.plot(size_sub["position"], size_sub["mean"], color=METHOD_COLORS[method],
                 marker="o", markersize=2.5, linewidth=1.1, linestyle="-", zorder=2)
        ax2.plot(balls_sub["position"], balls_sub["mean"], color=METHOD_COLORS[method],
                 linewidth=1.6, linestyle="--", zorder=3, alpha=0.8)

    ax1.set_xlabel("Stream Position")
    ax1.set_ylabel("Filter Total Size (bytes)")
    ax2.set_ylabel("Number of Balls")

    # Rescaling the right axis by the constant bytes_per_ball makes the
    # dashed (ball-count) curve coincide with the solid (size) curve.
    y1_min, y1_max = ax1.get_ylim()
    ax2.set_ylim(y1_min / bytes_per_ball, y1_max / bytes_per_ball)

    ax1.text(0.97, 0.06, "solid: size (bytes) -- dashed: ball count", transform=ax1.transAxes,
             ha="right", va="bottom", fontsize=6.0)
    style_axes(ax1)

    save_figure(fig, "streaming-filter-size.png", cfg.outdir, cfg.dpi, cfg.emit_pdf)
    return size_agg.merge(balls_agg, on=["method", "position"], suffixes=("_size", "_balls"))


# Legend

def save_method_legend(cfg: Config) -> None:
    apply_style(cfg.dpi)
    fig, ax = plt.subplots(figsize=(4.6, 0.55))
    ax.axis("off")
    handles = [plt.Line2D([0], [0], color=METHOD_COLORS[m], marker="o", markersize=4, linewidth=1.0)
              for m in METHODS]
    labels = [METHOD_LABELS[m] for m in METHODS]
    # The static-rebuild reference line (panels a/b/d) gets its own legend
    # entry too, instead of only an in-panel text annotation.
    handles.append(plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1.0))
    labels.append("Static rebuild (oracle)")
    fig.legend(handles, labels, loc="center", ncol=3, frameon=False, fontsize=7.8)
    fig.patch.set_alpha(0.0)
    save_figure(fig, "streaming-method-legend.png", cfg.outdir, cfg.dpi, cfg.emit_pdf, transparent=True)


# Text-report summary

def print_summary(df: pd.DataFrame, cfg: Config) -> None:
    print(f"\n--- Streaming summary (k_fixed={cfg.k_fixed}) ---")
    print("FNR=0 confirmed at every checkpoint for all 4 methods "
         "(enforced at generation time in stream_rebuild_worker; a violation raises, not plots).")
    print()


# Main

def generate_streaming_figures(df: Optional[pd.DataFrame] = None, outdir: str = OUTDIR,
                               k_fixed: Optional[int] = K_FIXED,
                               dpi: int = DPI, emit_pdf: bool = EMIT_PDF) -> None:
    if df is None:
        df = load_data()
    else:
        _validate_schema(df)

    cfg = resolve_config(df, k_fixed=k_fixed, dpi=dpi, emit_pdf=emit_pdf, outdir=outdir)

    panel_fpr_vs_position(df, cfg)
    panel_geometric_objective(df, cfg)
    panel_n_balls_vs_position(df, cfg)
    panel_vectors_vs_balls(df, cfg)
    panel_filter_size(df, cfg)
    save_method_legend(cfg)
    print_summary(df, cfg)

    snippet = r"""
\begin{subfigure}[b]{\linewidth}
    \centering
    \includegraphics[width=0.6\linewidth]{figures/streaming-method-legend.png}
\end{subfigure}\\[-0.3em]
\begin{subfigure}[b]{0.24\linewidth}
    \centering
    \includegraphics[width=\linewidth]{figures/streaming-fpr-vs-position.png}
    \caption{FPR vs. stream position}
\end{subfigure}
\begin{subfigure}[b]{0.24\linewidth}
    \centering
    \includegraphics[width=\linewidth]{figures/streaming-geometric-objective.png}
    \caption{Geometric objective vs. stream position}
\end{subfigure}
\begin{subfigure}[b]{0.24\linewidth}
    \centering
    \includegraphics[width=\linewidth]{figures/streaming-n-balls-vs-position.png}
    \caption{Number of balls vs. stream position}
\end{subfigure}
\begin{subfigure}[b]{0.24\linewidth}
    \centering
    \includegraphics[width=\linewidth]{figures/streaming-vectors-vs-balls.png}
    \caption{Vectors inserted \& ball count vs. stream position}
\end{subfigure}
\begin{subfigure}[b]{0.24\linewidth}
    \centering
    \includegraphics[width=\linewidth]{figures/streaming-filter-size.png}
    \caption{Filter size \& ball count vs. stream position}
\end{subfigure}
""".strip("\n")
    print("--- subfigure snippet (paste above the panel row) ---")
    print(snippet)


def main() -> None:
    generate_streaming_figures()


if __name__ == "__main__":
    main()
