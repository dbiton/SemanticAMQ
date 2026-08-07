"""Generate the scaling-evaluation figures (+ shared legend) for the PVLDB paper.

Fixes the filter size k and sweeps the number of indexed embeddings n, to
isolate SAMQ's representation size (memory, query cost) from its
construction cost: query-time memory and query latency are expected to stay
flat across n (k is fixed, so the filter's own buffer is k spheres
regardless of how many points built it), while construction time -- and
possibly FPR, as the indexed geometry grows more complex -- are expected to
grow with n. See main.evaluate_scaling()'s docstring for the full
experimental rationale.

Self-contained: reads a long-format results file (one row per
dataset/n/seed), aggregates over seeds, and writes PNG (optionally PDF)
panels sized for `\\includegraphics[width=\\linewidth]` inside
`subfigure{0.24\\linewidth}` cells of a two-column `figure*`, matching
plot_ablation.py's / plot_query_performance.py's panel styling.

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

from plot_ablation import apply_style

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #

INPUT_PATH = "results/scaling.csv"
OUTDIR = "figures/"
K_FIXED: Optional[int] = None  # None => the single k present in the data
DPI = 300
EMIT_PDF = False
DEMO = False

# --------------------------------------------------------------------------- #
# Fixed dataset styling -- must match every other figure in the paper.
# --------------------------------------------------------------------------- #

DATASETS = ["eli5", "mmlu", "wildchat"]
DATASET_LABELS = {"eli5": "ELI5", "mmlu": "MMLU", "wildchat": "WildChat"}
DATASET_COLORS = {"eli5": "#0072B2", "mmlu": "#E69F00", "wildchat": "#009E73"}
DATASET_HATCHES = {"eli5": "", "mmlu": "//", "wildchat": ".."}


@dataclasses.dataclass
class Config:
    k_fixed: int
    dpi: int
    emit_pdf: bool
    outdir: str


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

_REQUIRED_COLUMNS = ["dataset", "n", "k", "seed", "construct_time_s", "query_latency_us", "fpr", "memory_bytes"]


def _synthesize_demo_data() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    ns = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
    seeds = [0, 1, 2, 3, 4]
    k_fixed = 256

    # Memory and query latency are ~constant in n (k is fixed); construction
    # time grows ~linearly with n; FPR creeps up as the indexed geometry
    # becomes more complex -- the qualitative shape evaluate_scaling() expects.
    base_latency_us = {"eli5": 42.0, "mmlu": 47.0, "wildchat": 38.0}
    base_construct_s = {"eli5": 0.02, "mmlu": 0.026, "wildchat": 0.017}
    base_fpr = {"eli5": 0.01, "mmlu": 0.014, "wildchat": 0.008}
    bytes_per_sphere = {"eli5": 1544.0, "mmlu": 1544.0, "wildchat": 1544.0}

    rows = []
    for dataset in DATASETS:
        for n in ns:
            for seed in seeds:
                noise = rng.uniform(0.95, 1.05)
                rows.append({
                    "dataset": dataset, "n": n, "k": k_fixed, "seed": seed,
                    "construct_time_s": base_construct_s[dataset] * n * noise,
                    "query_latency_us": base_latency_us[dataset] * rng.uniform(0.97, 1.03),
                    "fpr": base_fpr[dataset] * (1.0 + np.log2(n / ns[0])) * rng.uniform(0.9, 1.1),
                    "memory_bytes": bytes_per_sphere[dataset] * k_fixed * rng.uniform(0.99, 1.01),
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

    if df["k"].nunique() != 1:
        raise ValueError(
            "evaluate_scaling fixes k across the whole sweep, but the input has "
            f"multiple k values: {sorted(df['k'].unique())}"
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


def resolve_config(
    df: pd.DataFrame,
    k_fixed: Optional[int] = K_FIXED,
    dpi: int = DPI,
    emit_pdf: bool = EMIT_PDF,
    outdir: str = OUTDIR,
) -> Config:
    available_k = sorted(df["k"].unique())
    if k_fixed is None:
        k_fixed = int(available_k[0])
    elif k_fixed not in available_k:
        raise ValueError(f"k_fixed={k_fixed} not present in data. Available k values: {available_k}")

    return Config(k_fixed=k_fixed, dpi=dpi, emit_pdf=emit_pdf, outdir=outdir)


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def aggregate(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Mean + 95% CI of `value_col` over seed, per (dataset, n)."""
    grouped = df.groupby(["dataset", "n"])[value_col]
    agg = grouped.agg(mean="mean", sem="sem", n_seeds="count").reset_index()
    agg["ci95"] = np.where(agg["n_seeds"] > 1, 1.96 * agg["sem"].fillna(0.0), 0.0)
    return agg


# --------------------------------------------------------------------------- #
# Line panels: x = indexed set size n, one line per dataset, k held fixed
# --------------------------------------------------------------------------- #

def _line_panel(
    df: pd.DataFrame,
    value_col: str,
    ylabel: str,
    filename: str,
    cfg: Config,
    log_y: bool = True,
) -> pd.DataFrame:
    agg = aggregate(df, value_col)

    apply_style()
    fig, ax = plt.subplots()

    for dataset in DATASETS:
        sub = agg[agg["dataset"] == dataset].sort_values("n")
        if sub.empty:
            continue
        has_ci = (sub["ci95"] > 0).any()
        ax.plot(
            sub["n"], sub["mean"],
            color=DATASET_COLORS[dataset],
            marker="o", markersize=2.5,
            linewidth=0.8,
            zorder=2,
        )
        if has_ci:
            ax.errorbar(
                sub["n"], sub["mean"], yerr=sub["ci95"],
                fmt="none", ecolor=DATASET_COLORS[dataset],
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
    else:
        # Flat-in-n quantities (memory, latency): give the y-axis some slack
        # around the data so a genuinely flat line doesn't hug the frame.
        ymin, ymax = agg["mean"].min(), agg["mean"].max()
        pad = 0.15 * max(ymax - ymin, ymax * 0.05, 1e-12)
        ax.set_ylim(ymin - pad, ymax + pad)

    ax.set_xlabel("Indexed Set Size (n)")
    ax.set_ylabel(ylabel)
    ax.text(
        0.03, 0.95, rf"$k = {cfg.k_fixed}$",
        transform=ax.transAxes, ha="left", va="top", fontsize=5.5,
    )
    ax.grid(axis="both", which="major", linewidth=0.35, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.5)

    _save(fig, filename, cfg)
    return agg


def panel_query_latency(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Single-query latency (us) vs n, at fixed k -- expected flat."""
    return _line_panel(df, "query_latency_us", "Query Latency (µs)", "scaling-query-latency.png", cfg, log_y=False)


def panel_construction_time(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Construction time (sec) vs n, at fixed k -- expected to grow with n."""
    return _line_panel(df, "construct_time_s", "Construction Time (s)", "scaling-construction-time.png", cfg, log_y=True)


def panel_fpr(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """FPR vs n, at fixed k, log scale -- may grow as indexed geometry gets more complex."""
    return _line_panel(df, "fpr", "FPR", "scaling-fpr.png", cfg, log_y=True)


def panel_memory(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Query-time memory (bytes) vs n, at fixed k -- expected flat."""
    return _line_panel(df, "memory_bytes", "Query-Time Memory (bytes)", "scaling-query-time-memory.png", cfg, log_y=False)


# --------------------------------------------------------------------------- #
# Shared legend
# --------------------------------------------------------------------------- #

def save_legend(cfg: Config) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(4.2, 0.25))
    ax.axis("off")

    handles = [
        plt.Rectangle(
            (0, 0), 1, 1,
            facecolor=DATASET_COLORS[d],
            hatch=DATASET_HATCHES[d],
            edgecolor="black",
            linewidth=0.4,
        )
        for d in DATASETS
    ]
    labels = [DATASET_LABELS[d] for d in DATASETS]

    fig.legend(
        handles, labels,
        loc="center", ncol=3,
        frameon=False, fontsize=7,
    )
    fig.patch.set_alpha(0.0)

    path = os.path.join(cfg.outdir, "dataset-legend.png")
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

def print_summary(df: pd.DataFrame, cfg: Config) -> None:
    print(f"\n--- Scaling summary (k={cfg.k_fixed}) ---")
    header = (
        f"{'dataset':<10} {'n':>8} "
        f"{'latency (us)':>13} {'construct (s)':>14} {'fpr':>10} {'memory (B)':>14}"
    )
    print(header)
    print("-" * len(header))

    lat = aggregate(df, "query_latency_us")
    con = aggregate(df, "construct_time_s")
    fpr = aggregate(df, "fpr")
    mem = aggregate(df, "memory_bytes")

    for dataset in DATASETS:
        ns = sorted(df[df["dataset"] == dataset]["n"].unique())
        for n in ns:
            def _mean(agg: pd.DataFrame) -> float:
                row = agg[(agg["dataset"] == dataset) & (agg["n"] == n)]
                return float(row["mean"].iloc[0]) if len(row) else float("nan")

            print(
                f"{DATASET_LABELS[dataset]:<10} {n:>8} "
                f"{_mean(lat):>13.3f} {_mean(con):>14.4f} {_mean(fpr):>10.4f} {_mean(mem):>14.1f}"
            )
    print()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def generate_scaling_figures(
    df: Optional[pd.DataFrame] = None,
    outdir: str = OUTDIR,
    k_fixed: Optional[int] = K_FIXED,
    dpi: int = DPI,
    emit_pdf: bool = EMIT_PDF,
) -> None:
    """
    Library entry point: render the four scaling panels (+ legend) from an
    in-memory long-format DataFrame (one row per dataset/n/seed, k fixed
    across the whole sweep), the same way
    plot_query_performance.generate_query_performance_figures is called
    directly on an in-memory df instead of round-tripping through a file. If
    `df` is omitted, falls back to load_data() (CONFIG-driven, honors
    DEMO/INPUT_PATH) so the script still works standalone.
    """
    if df is None:
        df = load_data()
    else:
        _validate_schema(df)

    cfg = resolve_config(df, k_fixed=k_fixed, dpi=dpi, emit_pdf=emit_pdf, outdir=outdir)

    panel_query_latency(df, cfg)
    panel_construction_time(df, cfg)
    panel_fpr(df, cfg)
    panel_memory(df, cfg)
    save_legend(cfg)

    print_summary(df, cfg)

    snippet = r"""
\begin{subfigure}{0.6\linewidth}
    \includegraphics[width=\linewidth]{figures/dataset-legend.png}
\end{subfigure}\hfill
\begin{subfigure}{0.24\linewidth}
    \includegraphics[width=\linewidth]{figures/scaling-query-latency.png}
    \caption{Single-query latency}
\end{subfigure}\hfill
\begin{subfigure}{0.24\linewidth}
    \includegraphics[width=\linewidth]{figures/scaling-construction-time.png}
    \caption{Construction time}
\end{subfigure}\hfill
\begin{subfigure}{0.24\linewidth}
    \includegraphics[width=\linewidth]{figures/scaling-fpr.png}
    \caption{False positive rate}
\end{subfigure}\hfill
\begin{subfigure}{0.24\linewidth}
    \includegraphics[width=\linewidth]{figures/scaling-query-time-memory.png}
    \caption{Query-time memory}
\end{subfigure}
""".strip("\n")
    print("--- subfigure snippet (paste above the panel row) ---")
    print(snippet)


def main() -> None:
    generate_scaling_figures()


if __name__ == "__main__":
    main()
