"""Scaling-evaluation figures: BallCoverFilter throughput vs batch size / stream size.
Rows come from every blocklist corpus/fold (see main.py's generate_scaling(), which
reuses blocklist_workloads()) and are pooled per k/n value, since throughput is
dataset-invariant here. Query, insertion, and construction throughput each get one
line per filter size k.
"""

from __future__ import annotations

import dataclasses
import os
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter

from plots.plot_common import apply_style, save_figure

QUERY_INPUT_PATH = "results/scaling_query.csv"
INSERT_INPUT_PATH = "results/scaling_insert.csv"
CONSTRUCTION_INPUT_PATH = "results/scaling_construction.csv"
OUTDIR = "plots/scaling/"
DPI = 300
EMIT_PDF = False

# Okabe-Ito colorblind-safe palette, extended to 6 distinct lines
LINE_COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
# Distinct markers for each line
LINE_MARKERS = ["o", "s", "D", "v", "^", "p"]

_QUERY_COLUMNS = ["dataset", "k", "batch_size", "seed", "n_queries", "throughput_qps"]
_INSERT_COLUMNS = ["dataset", "k", "batch_size", "seed", "n_inserted", "throughput_ips"]
_CONSTRUCTION_COLUMNS = ["dataset", "k", "n", "seed", "construct_time_s", "throughput_vps"]


@dataclasses.dataclass
class Config:
    dpi: int
    emit_pdf: bool
    outdir: str


def _validate_schema(df: pd.DataFrame, required: List[str], name: str) -> None:
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{name} input is missing columns: {missing_cols}. Present: {sorted(df.columns)}")


def _load(path: str, required: List[str], name: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Input file not found: {path!r}. Run generate_scaling() (main.py) first, "
            "or point the *_INPUT_PATH constant at a real results file."
        )
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    _validate_schema(df, required, name)
    return df


def load_query_data(path: str = QUERY_INPUT_PATH) -> pd.DataFrame:
    return _load(path, _QUERY_COLUMNS, "query")


def load_insert_data(path: str = INSERT_INPUT_PATH) -> pd.DataFrame:
    return _load(path, _INSERT_COLUMNS, "insert")


def load_construction_data(path: str = CONSTRUCTION_INPUT_PATH) -> pd.DataFrame:
    return _load(path, _CONSTRUCTION_COLUMNS, "construction")


def resolve_config(dpi: int = DPI, emit_pdf: bool = EMIT_PDF, outdir: str = OUTDIR) -> Config:
    return Config(dpi=dpi, emit_pdf=emit_pdf, outdir=outdir)


def _aggregate(df: pd.DataFrame, group_cols: List[str], value_col: str) -> pd.DataFrame:
    """Mean + 95% CI of value_col over seed, grouped by group_cols."""
    agg = df.groupby(group_cols)[value_col].agg(mean="mean", sem="sem", count="count").reset_index()
    agg["ci95"] = np.where(agg["count"] > 1, 1.96 * agg["sem"].fillna(0.0), 0.0)
    return agg


def _throughput_panel(
    df: pd.DataFrame,
    x_col: str,
    value_col: str,
    ylabel: str,
    xlabel: str,
    filename: str,
    cfg: Config,
) -> pd.DataFrame:
    """Shared renderer for the three metric-on-x-axis, one-line-per-
    filter-size (k) throughput panels -- only the x-axis column (batch_size or n)
    and the value column differ between query/insertion/construction."""
    agg = _aggregate(df, ["k", x_col], value_col)

    line_values = sorted(agg["k"].unique())
    if len(line_values) > len(LINE_COLORS):
        raise ValueError(
            f"Filter Size (k) sweep has {len(line_values)} distinct values; only "
            f"{len(LINE_COLORS)} line colors are defined in LINE_COLORS."
        )

    apply_style(cfg.dpi)
    fig, ax = plt.subplots()

    for color, marker, lv in zip(LINE_COLORS, LINE_MARKERS, line_values):
        sub = agg[agg["k"] == lv].sort_values(x_col)
        ax.plot(
            sub[x_col], sub["mean"], color=color, marker=marker, markersize=3.5,
            linewidth=0.8, zorder=2, label=str(int(lv)),
        )
        if (sub["ci95"] > 0).any():
            ax.errorbar(
                sub[x_col], sub["mean"], yerr=sub["ci95"], fmt="none", ecolor=color,
                elinewidth=0.5, capsize=1.5, capthick=0.5, zorder=3,
            )

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(LogLocator(base=2.0))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=2.0))
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="y", which="minor", length=1.2, width=0.4)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="both", which="major", linewidth=0.35, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.5)

    save_figure(fig, filename, cfg.outdir, cfg.dpi, cfg.emit_pdf)
    plt.close(fig)
    return agg


def generate_legend_figure(line_values: List[float], cfg: Config) -> None:
    """Generates a standalone horizontal legend for the filter sizes."""
    apply_style(cfg.dpi)
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.axis("off")  # Hide the axes to only render the legend

    handles = []
    labels = []
    for color, marker, lv in zip(LINE_COLORS, LINE_MARKERS, line_values):
        # Create dummy lines for the legend handles
        handle, = ax.plot([], [], color=color, marker=marker, markersize=4.0,
                          linewidth=1.2, label=str(int(lv)))
        handles.append(handle)
        labels.append(str(int(lv)))

    ax.legend(handles, labels, loc="center", ncol=len(line_values),
              title="Filter Size (k)", frameon=False, fontsize=9.6, title_fontsize=10.8)

    save_figure(fig, "scaling-legend.png", cfg.outdir, cfg.dpi, cfg.emit_pdf)
    plt.close(fig)


def panel_query_throughput(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    return _throughput_panel(
        df, "batch_size", "throughput_qps", "Query OPS",
        "Batch size", "scaling-query-throughput.png", cfg,
    )


def panel_insert_throughput(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    return _throughput_panel(
        df, "batch_size", "throughput_ips", "Insertion OPS",
        "Batch size", "scaling-insertion-throughput.png", cfg,
    )


def panel_construction_throughput(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    return _throughput_panel(
        df, "n", "throughput_vps", "Construction OPS",
        r"Initialization size ($|X|$)", "scaling-construction-throughput.png", cfg,
    )


def print_summary(df_query: pd.DataFrame, df_insert: pd.DataFrame, df_construction: pd.DataFrame) -> None:
    print("\n--- Scaling summary ---")
    for label, df, line_col, value_col in (
        ("query", df_query, "batch_size", "throughput_qps"),
        ("insert", df_insert, "batch_size", "throughput_ips"),
        ("construction", df_construction, "n", "throughput_vps"),
    ):
        agg = _aggregate(df, ["k", line_col], value_col)
        print(f"\n{label} throughput ({value_col}):")
        for _, row in agg.sort_values(["k", line_col]).iterrows():
            print(f"  k={int(row['k']):>6}  {line_col}={int(row[line_col]):>7}  mean={row['mean']:>12.1f}")
    print()


def generate_scaling_figures(
    df_query: Optional[pd.DataFrame] = None,
    df_insert: Optional[pd.DataFrame] = None,
    df_construction: Optional[pd.DataFrame] = None,
    outdir: str = OUTDIR,
    dpi: int = DPI,
    emit_pdf: bool = EMIT_PDF,
) -> None:
    if df_query is None:
        df_query = load_query_data()
    else:
        _validate_schema(df_query, _QUERY_COLUMNS, "query")
    if df_insert is None:
        df_insert = load_insert_data()
    else:
        _validate_schema(df_insert, _INSERT_COLUMNS, "insert")
    if df_construction is None:
        df_construction = load_construction_data()
    else:
        _validate_schema(df_construction, _CONSTRUCTION_COLUMNS, "construction")

    cfg = resolve_config(dpi=dpi, emit_pdf=emit_pdf, outdir=outdir)
    
    # Render individual panels without internal legends
    panel_query_throughput(df_query, cfg)
    panel_insert_throughput(df_insert, cfg)
    panel_construction_throughput(df_construction, cfg)

    # Extract unique filter sizes and render the shared standalone legend
    unique_k = sorted(df_query["k"].unique())
    generate_legend_figure(unique_k, cfg)

    print_summary(df_query, df_insert, df_construction)


def main() -> None:
    generate_scaling_figures()


if __name__ == "__main__":
    main()