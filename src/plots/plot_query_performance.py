"""Generate the query-performance figures (+ shared legend) for the PVLDB paper.

Self-contained: reads a long-format results file (one row per
dataset/k/seed/mode), aggregates over seeds, and writes PNG (optionally PDF)
panels sized for `\\includegraphics[width=\\linewidth]` inside
`subfigure{0.24\\linewidth}` cells of a two-column `figure*`, matching
plot_quantization.py's panel styling.

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

from plots.plot_common import DATASETS, DATASET_LABELS, DATASET_COLORS, apply_style as _apply_style, save_figure, save_dataset_legend, mean_ci95

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #

INPUT_PATH = "results/query_performance.csv"
OUTDIR = "plots/query_performance/"
DPI = 400
EMIT_PDF = False
DEMO = False

MODES = ["single", "batched"]


@dataclasses.dataclass
class Config:
    dpi: int
    emit_pdf: bool
    outdir: str


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

_REQUIRED_COLUMNS = ["dataset", "k", "seed", "mode", "n_queries", "latency_ms", "throughput_qps"]


def _synthesize_demo_data() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    ks = [16, 32, 64, 128, 256, 512, 1024]
    seeds = [0, 1, 2]

    base_latency_ms = {"eli5": 0.010, "mmlu": 0.009, "wildchat": 0.011}

    rows = []
    for dataset in DATASETS:
        d_scale = base_latency_ms[dataset]
        for k in ks:
            for seed in seeds:
                for mode in MODES:
                    noise = rng.uniform(0.9, 1.1)
                    per_query_ms = d_scale * k * noise * (1.0 if mode == "single" else 0.2)
                    n_queries = 2000 if mode == "single" else 80000
                    rows.append({
                        "dataset": dataset, "k": k, "seed": seed, "mode": mode,
                        "n_queries": n_queries,
                        "latency_ms": per_query_ms,
                        "throughput_qps": 1000.0 / per_query_ms,
                    })
    return pd.DataFrame(rows)


def _validate_schema(df: pd.DataFrame) -> None:
    missing_cols = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Input is missing required columns: {missing_cols}. "
            f"Present columns: {sorted(df.columns)}"
        )

    missing_ds = set(DATASETS) - set(df["dataset"].unique())
    if missing_ds:
        raise ValueError(
            f"Missing dataset(s) {sorted(missing_ds)} in data. "
            f"Available: {sorted(df['dataset'].unique())}"
        )

    missing_modes = set(MODES) - set(df["mode"].unique())
    if missing_modes:
        raise ValueError(
            f"Missing mode(s) {sorted(missing_modes)} in data. "
            f"Available: {sorted(df['mode'].unique())}"
        )


def load_data() -> pd.DataFrame:
    if DEMO:
        df = _synthesize_demo_data()
    else:
        if not os.path.exists(INPUT_PATH):
            raise FileNotFoundError(
                f"Input file not found: {INPUT_PATH!r}. Set DEMO = True to render "
                "with synthetic data, or point INPUT_PATH at a real results file."
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

def apply_style() -> None:
    _apply_style(DPI)


def aggregate(df: pd.DataFrame, mode: str, value_col: str) -> pd.DataFrame:
    """Mean + 95% CI of `value_col` over seed, per (dataset, k), at the given mode."""
    return mean_ci95(df[df["mode"] == mode], ["dataset", "k"], value_col)


# --------------------------------------------------------------------------- #
# Line panels: x = filter size k, one line per dataset
# --------------------------------------------------------------------------- #

def _line_panel(
    df: pd.DataFrame,
    mode: str,
    value_col: str,
    ylabel: str,
    filename: str,
    cfg: Config,
) -> pd.DataFrame:
    agg = aggregate(df, mode, value_col)

    apply_style()
    fig, ax = plt.subplots()

    for dataset in DATASETS:
        sub = agg[agg["dataset"] == dataset].sort_values("k")
        if sub.empty:
            continue
        has_ci = (sub["ci95"] > 0).any()
        ax.plot(
            sub["k"], sub["mean"],
            color=DATASET_COLORS[dataset],
            marker="o", markersize=2.5,
            linewidth=0.8,
            zorder=2,
        )
        if has_ci:
            ax.errorbar(
                sub["k"], sub["mean"], yerr=sub["ci95"],
                fmt="none", ecolor=DATASET_COLORS[dataset],
                elinewidth=0.5, capsize=1.5, capthick=0.5,
                zorder=3,
            )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=2.0))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=2.0))
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="y", which="minor", length=1.2, width=0.4)

    ax.set_xlabel("Filter Size (k)")
    ax.set_ylabel(ylabel)
    ax.grid(axis="both", which="major", linewidth=0.35, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.5)

    _save(fig, filename, cfg)
    return agg


def panel_single_latency(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    return _line_panel(df, "single", "latency_ms", "Latency (ms)", "single-query-latency.png", cfg)


def panel_batched_latency(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    return _line_panel(df, "batched", "latency_ms", "Latency (ms)", "batched-query-latency.png", cfg)


def panel_single_throughput(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    return _line_panel(
        df, "single", "throughput_qps", "Throughput (queries/s)", "single-query-throughput.png", cfg
    )


def panel_batched_throughput(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    return _line_panel(
        df, "batched", "throughput_qps", "Throughput (queries/s)", "batched-query-throughput.png", cfg
    )


# --------------------------------------------------------------------------- #
# Shared legend
# --------------------------------------------------------------------------- #

def save_legend(cfg: Config) -> None:
    save_dataset_legend(cfg.outdir, cfg.dpi, cfg.emit_pdf)


# --------------------------------------------------------------------------- #
# Save helper
# --------------------------------------------------------------------------- #

def _save(fig: plt.Figure, filename: str, cfg: Config) -> None:
    save_figure(fig, filename, cfg.outdir, cfg.dpi, cfg.emit_pdf)


# --------------------------------------------------------------------------- #
# Summary table
# --------------------------------------------------------------------------- #

def print_summary(df: pd.DataFrame) -> None:
    print("\n--- Query performance summary ---")
    header = (
        f"{'dataset':<10} {'k':>6} "
        f"{'single lat (ms)':>16} {'batched lat (ms)':>17} "
        f"{'single QPS':>12} {'batched QPS':>12}"
    )
    print(header)
    print("-" * len(header))

    lat_single = aggregate(df, "single", "latency_ms")
    lat_batched = aggregate(df, "batched", "latency_ms")
    qps_single = aggregate(df, "single", "throughput_qps")
    qps_batched = aggregate(df, "batched", "throughput_qps")

    for dataset in DATASETS:
        ks = sorted(df[df["dataset"] == dataset]["k"].unique())
        for k in ks:
            def _mean(agg: pd.DataFrame) -> float:
                row = agg[(agg["dataset"] == dataset) & (agg["k"] == k)]
                return float(row["mean"].iloc[0]) if len(row) else float("nan")

            print(
                f"{DATASET_LABELS[dataset]:<10} {k:>6} "
                f"{_mean(lat_single):>16.4f} {_mean(lat_batched):>17.4f} "
                f"{_mean(qps_single):>12.1f} {_mean(qps_batched):>12.1f}"
            )
    print()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def generate_query_performance_figures(
    df: Optional[pd.DataFrame] = None,
    outdir: str = OUTDIR,
    dpi: int = DPI,
    emit_pdf: bool = EMIT_PDF,
) -> None:
    """
    Library entry point: render the four query-performance panels (+ legend)
    from an in-memory long-format DataFrame, the same way
    plot_quantization.generate_quantization_figures is called directly on an
    in-memory df instead of round-tripping through a file. If `df` is
    omitted, falls back to load_data() (CONFIG-driven, honors DEMO/INPUT_PATH)
    so the script still works standalone.
    """
    if df is None:
        df = load_data()
    else:
        _validate_schema(df)

    cfg = Config(dpi=dpi, emit_pdf=emit_pdf, outdir=outdir)

    panel_single_latency(df, cfg)
    panel_batched_latency(df, cfg)
    panel_single_throughput(df, cfg)
    panel_batched_throughput(df, cfg)
    save_legend(cfg)

    print_summary(df)

    snippet = r"""
\begin{subfigure}{0.6\linewidth}
    \includegraphics[width=\linewidth]{figures/dataset-legend.png}
\end{subfigure}\hfill
\begin{subfigure}{0.24\linewidth}
    \includegraphics[width=\linewidth]{figures/single-query-latency.png}
    \caption{Single query latency}
\end{subfigure}\hfill
\begin{subfigure}{0.24\linewidth}
    \includegraphics[width=\linewidth]{figures/batched-query-latency.png}
    \caption{Batched query latency}
\end{subfigure}\hfill
\begin{subfigure}{0.24\linewidth}
    \includegraphics[width=\linewidth]{figures/single-query-throughput.png}
    \caption{Single query throughput}
\end{subfigure}\hfill
\begin{subfigure}{0.24\linewidth}
    \includegraphics[width=\linewidth]{figures/batched-query-throughput.png}
    \caption{Batched query throughput}
\end{subfigure}
""".strip("\n")
    print("--- subfigure snippet (paste above the panel row) ---")
    print(snippet)


def main() -> None:
    generate_query_performance_figures()


if __name__ == "__main__":
    main()
