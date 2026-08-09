"""Scaling-evaluation figures: fixed k, sweep indexed-set size n. Memory and
query latency should stay flat; construction time and FPR may grow with n.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter

from plots.plot_common import DATASETS, DATASET_COLORS, apply_style, save_figure, save_dataset_legend, mean_ci95

INPUT_PATH = "results/scaling.csv"
OUTDIR = "plots/scaling/"
K_FIXED: Optional[int] = None  # None => the single k present in the data
DPI = 300
EMIT_PDF = False
DEMO = False

_REQUIRED_COLUMNS = ["dataset", "n", "k", "seed", "construct_time_s", "query_latency_us", "fpr", "memory_bytes"]


@dataclasses.dataclass
class Config:
    k_fixed: int
    dpi: int
    emit_pdf: bool
    outdir: str


def _synthesize_demo_data() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    ns = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
    seeds = [0, 1, 2, 3, 4]
    k_fixed = 256

    base_latency_us = {"eli5": 42.0, "mmlu": 47.0, "wildchat": 38.0}
    base_construct_s = {"eli5": 0.02, "mmlu": 0.026, "wildchat": 0.017}
    base_fpr = {"eli5": 0.01, "mmlu": 0.014, "wildchat": 0.008}
    bytes_per_sphere = 1544.0

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
                    "memory_bytes": bytes_per_sphere * k_fixed * rng.uniform(0.99, 1.01),
                })
    return pd.DataFrame(rows)


def _validate_schema(df: pd.DataFrame) -> None:
    missing_cols = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Input missing columns: {missing_cols}. Present: {sorted(df.columns)}")
    missing_ds = set(DATASETS) - set(df["dataset"].unique())
    if missing_ds:
        raise ValueError(f"Missing dataset(s) {sorted(missing_ds)}. Available: {sorted(df['dataset'].unique())}")
    if df["k"].nunique() != 1:
        raise ValueError(f"Expected a single fixed k, found: {sorted(df['k'].unique())}")


def load_data() -> pd.DataFrame:
    if DEMO:
        df = _synthesize_demo_data()
    else:
        if not os.path.exists(INPUT_PATH):
            raise FileNotFoundError(f"Input file not found: {INPUT_PATH!r}. Set DEMO=True or fix INPUT_PATH.")
        df = pd.read_parquet(INPUT_PATH) if INPUT_PATH.endswith(".parquet") else pd.read_csv(INPUT_PATH)
    _validate_schema(df)
    return df


def resolve_config(df: pd.DataFrame, k_fixed: Optional[int] = K_FIXED, dpi: int = DPI,
                   emit_pdf: bool = EMIT_PDF, outdir: str = OUTDIR) -> Config:
    available_k = sorted(df["k"].unique())
    if k_fixed is None:
        k_fixed = int(available_k[0])
    elif k_fixed not in available_k:
        raise ValueError(f"k_fixed={k_fixed} not present. Available: {available_k}")
    return Config(k_fixed=k_fixed, dpi=dpi, emit_pdf=emit_pdf, outdir=outdir)


def _line_panel(df: pd.DataFrame, value_col: str, ylabel: str, filename: str, cfg: Config,
                log_y: bool = True) -> pd.DataFrame:
    agg = mean_ci95(df, ["dataset", "n"], value_col)

    apply_style(cfg.dpi)
    fig, ax = plt.subplots()
    for dataset in DATASETS:
        sub = agg[agg["dataset"] == dataset].sort_values("n")
        if sub.empty:
            continue
        ax.plot(sub["n"], sub["mean"], color=DATASET_COLORS[dataset], marker="o", markersize=2.5,
               linewidth=0.8, zorder=2)
        if (sub["ci95"] > 0).any():
            ax.errorbar(sub["n"], sub["mean"], yerr=sub["ci95"], fmt="none", ecolor=DATASET_COLORS[dataset],
                       elinewidth=0.5, capsize=1.5, capthick=0.5, zorder=3)

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
        # linear y makes a flat line read as flat, not a log-scale artifact
        ymin, ymax = agg["mean"].min(), agg["mean"].max()
        pad = 0.15 * max(ymax - ymin, ymax * 0.05, 1e-12)
        ax.set_ylim(ymin - pad, ymax + pad)

    ax.set_xlabel("Indexed Set Size (n)")
    ax.set_ylabel(ylabel)
    ax.text(0.03, 0.95, rf"$k = {cfg.k_fixed}$", transform=ax.transAxes, ha="left", va="top", fontsize=5.5)
    ax.grid(axis="both", which="major", linewidth=0.35, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.5)

    save_figure(fig, filename, cfg.outdir, cfg.dpi, cfg.emit_pdf)
    return agg


def panel_query_latency(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    return _line_panel(df, "query_latency_us", "Query Latency (µs)", "scaling-query-latency.png", cfg, log_y=False)


def panel_construction_time(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    return _line_panel(df, "construct_time_s", "Construction Time (s)", "scaling-construction-time.png", cfg)


def panel_fpr(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    return _line_panel(df, "fpr", "FPR", "scaling-fpr.png", cfg)


def panel_memory(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    return _line_panel(df, "memory_bytes", "Query-Time Memory (bytes)", "scaling-query-time-memory.png", cfg, log_y=False)


def print_summary(df: pd.DataFrame, cfg: Config) -> None:
    print(f"\n--- Scaling summary (k={cfg.k_fixed}) ---")
    lat, con, fpr, mem = (mean_ci95(df, ["dataset", "n"], c)
                          for c in ("query_latency_us", "construct_time_s", "fpr", "memory_bytes"))
    for dataset in DATASETS:
        for n in sorted(df[df["dataset"] == dataset]["n"].unique()):
            def _mean(agg):
                row = agg[(agg["dataset"] == dataset) & (agg["n"] == n)]
                return float(row["mean"].iloc[0]) if len(row) else float("nan")
            print(f"{dataset:<10} n={n:>6} latency={_mean(lat):.2f}us construct={_mean(con):.3f}s "
                 f"fpr={_mean(fpr):.4f} memory={_mean(mem):.0f}B")
    print()


def generate_scaling_figures(df: Optional[pd.DataFrame] = None, outdir: str = OUTDIR,
                             k_fixed: Optional[int] = K_FIXED, dpi: int = DPI, emit_pdf: bool = EMIT_PDF) -> None:
    if df is None:
        df = load_data()
    else:
        _validate_schema(df)

    cfg = resolve_config(df, k_fixed=k_fixed, dpi=dpi, emit_pdf=emit_pdf, outdir=outdir)
    panel_query_latency(df, cfg)
    panel_construction_time(df, cfg)
    panel_fpr(df, cfg)
    panel_memory(df, cfg)
    save_dataset_legend(cfg.outdir, cfg.dpi, cfg.emit_pdf)
    print_summary(df, cfg)


def main() -> None:
    generate_scaling_figures()


if __name__ == "__main__":
    main()
