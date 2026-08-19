"""Generate the equal-memory-comparison figures (+ shared legend) for the paper.

Self-contained: reads a long-format results file (one row per filter config /
memory budget), and writes PNG (optionally PDF) panels sized for
`\\includegraphics[width=\\linewidth]` inside `subfigure{0.24\\linewidth}` cells
of a two-column `figure*`, matching plot_ablation.py / plot_query_performance.py's
panel styling.

Edit the CONFIG block below to change parameters -- no CLI flags.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter

from plots.plot_ablation import apply_style

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #

INPUT_PATH = "results/equal_memory.csv"
OUTDIR = "plots/equal_memory/"
DPI = 400
EMIT_PDF = False

# --------------------------------------------------------------------------- #
# Fixed filter styling -- must match every other figure in the paper.
# --------------------------------------------------------------------------- #

CONFIGS = ["BallCoverFilter", "DSBFFilter", "CoveringLSHFallbackFilter", "LSBFFilter", "DSHFilter", "HNSWFilter"]
CONFIG_LABELS = {
    "BallCoverFilter": "Ours",
    "DSBFFilter": "DSBF",
    "CoveringLSHFallbackFilter": "CoveringLSH-Fallback",
    "LSBFFilter": "LSBF",
    "DSHFilter": "DSH",
    "HNSWFilter": "HNSW",
}
CONFIG_COLORS = {
    "BallCoverFilter": "#0072B2",
    "DSBFFilter": "#E69F00",
    "CoveringLSHFallbackFilter": "#009E73",
    "LSBFFilter": "#CC79A7",
    "DSHFilter": "#D55E00",
    "HNSWFilter": "#56B4E9",
}
CONFIG_MARKERS = {
    "BallCoverFilter": "o",
    "DSBFFilter": "s",
    "CoveringLSHFallbackFilter": "^",
    "LSBFFilter": "D",
    "DSHFilter": "v",
    "HNSWFilter": "P",
}


@dataclasses.dataclass
class Config:
    dpi: int
    emit_pdf: bool
    outdir: str


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

_REQUIRED_COLUMNS = ["Config", "Memory Budget", "FPR", "FNR", "Latency (ms)", "Throughput (qps)"]


def _validate_schema(df: pd.DataFrame) -> None:
    missing_cols = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Input is missing required columns: {missing_cols}. "
            f"Present columns: {sorted(df.columns)}"
        )

    filters_present = set(df["Config"].str.split(" + ", regex=False).str[0].unique())
    missing = set(CONFIGS) - filters_present
    if missing:
        raise ValueError(
            f"Missing filter config(s) {sorted(missing)} in data. "
            f"Available: {sorted(filters_present)}"
        )


def load_data() -> pd.DataFrame:
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH!r}. Run generate_equal_memory() "
            "(main.py) first, or point INPUT_PATH at a real results file."
        )
    if INPUT_PATH.endswith(".parquet"):
        df = pd.read_parquet(INPUT_PATH)
    else:
        df = pd.read_csv(INPUT_PATH)

    _validate_schema(df)
    return df


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def aggregate(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Mean + 95% CI of `value_col` over any repeated runs, per (filter, memory
    budget). A single run per (filter, budget) is the common case -- ci95 is
    then just 0 and no error bars are drawn -- but repeated runs (e.g. if the
    caller adds a seed sweep later) are averaged transparently."""
    work = df.copy()
    work["filter"] = work["Config"].str.split(" + ", regex=False).str[0]
    work["Memory Budget (MB)"] = work["Memory Budget"] / (1024 * 1024)

    grouped = work.groupby(["filter", "Memory Budget (MB)"])[value_col]
    agg = grouped.agg(mean="mean", sem="sem", n="count").reset_index()
    agg["ci95"] = np.where(agg["n"] > 1, 1.96 * agg["sem"].fillna(0.0), 0.0)
    return agg


# --------------------------------------------------------------------------- #
# Line panels: x = memory budget, one line per filter config
# --------------------------------------------------------------------------- #

def _line_panel(
    df: pd.DataFrame,
    value_col: str,
    ylabel: str,
    filename: str,
    cfg: Config,
    log_y: bool = False,
    y_bounds: Optional[tuple] = None,
) -> pd.DataFrame:
    agg = aggregate(df, value_col)

    apply_style()
    fig, ax = plt.subplots()

    for name in CONFIGS:
        sub = agg[agg["filter"] == name].sort_values("Memory Budget (MB)")
        if sub.empty:
            continue
        has_ci = (sub["ci95"] > 0).any()
        ax.plot(
            sub["Memory Budget (MB)"], sub["mean"],
            color=CONFIG_COLORS[name],
            marker=CONFIG_MARKERS[name], markersize=2.5,
            linewidth=0.8,
            zorder=2,
        )
        if has_ci:
            ax.errorbar(
                sub["Memory Budget (MB)"], sub["mean"], yerr=sub["ci95"],
                fmt="none", ecolor=CONFIG_COLORS[name],
                elinewidth=0.5, capsize=1.5, capthick=0.5,
                zorder=3,
            )

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(LogLocator(base=2.0))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=2.0))

    if log_y:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10.0))
        ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="y", which="minor", length=1.2, width=0.4)
    elif y_bounds is not None:
        ax.set_ylim(*y_bounds)

    ax.set_xlabel("Memory Budget (MB)")
    ax.set_ylabel(ylabel)
    ax.grid(axis="both", which="major", linewidth=0.35, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.5)

    _save(fig, filename, cfg)
    return agg


def panel_fpr(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    return _line_panel(df, "FPR", "False Positive Rate", "equal-memory-fpr.png", cfg, y_bounds=(-0.05, 1.05))


def panel_fnr(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    return _line_panel(df, "FNR", "False Negative Rate", "equal-memory-fnr.png", cfg, y_bounds=(-0.05, 1.05))


def panel_latency(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    return _line_panel(df, "Latency (ms)", "Latency (ms)", "equal-memory-latency.png", cfg, log_y=True)


def panel_throughput(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    return _line_panel(
        df, "Throughput (qps)", "Throughput (queries/s)", "equal-memory-throughput.png", cfg, log_y=True
    )


# --------------------------------------------------------------------------- #
# Shared legend
# --------------------------------------------------------------------------- #

def save_legend(cfg: Config) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(5.6, 0.28))
    ax.axis("off")

    handles = [
        plt.Line2D(
            [0], [0],
            color=CONFIG_COLORS[name], marker=CONFIG_MARKERS[name],
            markersize=3, linewidth=1.0,
        )
        for name in CONFIGS
    ]
    labels = [CONFIG_LABELS[name] for name in CONFIGS]

    fig.legend(
        handles, labels,
        loc="center", ncol=len(CONFIGS),
        frameon=False, fontsize=7,
    )
    fig.patch.set_alpha(0.0)

    path = os.path.join(cfg.outdir, "equal-memory-legend.png")
    fig.savefig(path, dpi=cfg.dpi, bbox_inches="tight", pad_inches=0.01, transparent=True)
    if cfg.emit_pdf:
        fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight", pad_inches=0.01, transparent=True)
    plt.close(fig)
    print(f"Wrote {path}")


# --------------------------------------------------------------------------- #
# Save helper
# --------------------------------------------------------------------------- #

def _save(fig: plt.Figure, filename: str, cfg: Config) -> None:
    os.makedirs(cfg.outdir, exist_ok=True)
    path = os.path.join(cfg.outdir, filename)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.01, dpi=cfg.dpi)
    if cfg.emit_pdf:
        fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"Wrote {path}")


# --------------------------------------------------------------------------- #
# Summary table
# --------------------------------------------------------------------------- #

def print_summary(df: pd.DataFrame) -> None:
    print("\n--- Equal-memory comparison summary ---")
    header = f"{'filter':<24} {'budget (MB)':>12} {'FPR':>8} {'FNR':>8} {'lat (ms)':>10} {'QPS':>12}"
    print(header)
    print("-" * len(header))

    fpr_agg = aggregate(df, "FPR")
    fnr_agg = aggregate(df, "FNR")
    lat_agg = aggregate(df, "Latency (ms)")
    qps_agg = aggregate(df, "Throughput (qps)")

    for name in CONFIGS:
        budgets = sorted(fpr_agg[fpr_agg["filter"] == name]["Memory Budget (MB)"].unique())
        for budget in budgets:
            def _mean(agg: pd.DataFrame) -> float:
                row = agg[(agg["filter"] == name) & (agg["Memory Budget (MB)"] == budget)]
                return float(row["mean"].iloc[0]) if len(row) else float("nan")

            print(
                f"{CONFIG_LABELS[name]:<24} {budget:>12.3f} "
                f"{_mean(fpr_agg):>8.3f} {_mean(fnr_agg):>8.3f} "
                f"{_mean(lat_agg):>10.4f} {_mean(qps_agg):>12.1f}"
            )
    print()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def generate_equal_memory_figures(
    df: Optional[pd.DataFrame] = None,
    outdir: str = OUTDIR,
    dpi: int = DPI,
    emit_pdf: bool = EMIT_PDF,
) -> None:
    """
    Library entry point: render the four equal-memory-comparison panels
    (+ legend) from an in-memory long-format DataFrame, the same way
    plot_ablation.generate_ablation_figures is called directly on an
    in-memory df instead of round-tripping through a file. If `df` is
    omitted, falls back to load_data() (CONFIG-driven, reads INPUT_PATH) so
    the script still works standalone.
    """
    if df is None:
        df = load_data()
    else:
        _validate_schema(df)

    cfg = Config(dpi=dpi, emit_pdf=emit_pdf, outdir=outdir)

    panel_fpr(df, cfg)
    panel_fnr(df, cfg)
    panel_latency(df, cfg)
    panel_throughput(df, cfg)
    save_legend(cfg)

    print_summary(df)

    snippet = r"""
\begin{figure*}[]
  \centering

  \begin{subfigure}{0.6\linewidth}
    \includegraphics[width=\linewidth]{figures/equal-memory-legend.png}
  \end{subfigure}\hfill

  \vspace{0.1em}

  % Row of 4 graphs
  \begin{subfigure}{0.24\linewidth}
    \includegraphics[width=\linewidth]{figures/equal-memory-fpr.png}
    \caption{False positive rate}
    \label{fig:equal-memory-fpr}
  \end{subfigure}\hfill
  \begin{subfigure}{0.24\linewidth}
    \includegraphics[width=\linewidth]{figures/equal-memory-fnr.png}
    \caption{False negative rate}
    \label{fig:equal-memory-fnr}
  \end{subfigure}\hfill
  \begin{subfigure}{0.24\linewidth}
    \includegraphics[width=\linewidth]{figures/equal-memory-latency.png}
    \caption{Query latency}
    \label{fig:equal-memory-latency}
  \end{subfigure}\hfill
  \begin{subfigure}{0.24\linewidth}
    \includegraphics[width=\linewidth]{figures/equal-memory-throughput.png}
    \caption{Query throughput}
    \label{fig:equal-memory-throughput}
  \end{subfigure}\hfill

  \vspace{0.5em}
  \caption{Equal-memory comparison: every method is given the same memory budget $M$, swept across a range of budgets, with the best valid configuration of each chosen on a held-out calibration workload.}
  \label{fig:equal-memory}
\end{figure*}
""".strip("\n")
    print("--- figure snippet ---")
    print(snippet)


def main() -> None:
    generate_equal_memory_figures()


if __name__ == "__main__":
    main()
