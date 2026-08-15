"""System-evaluation figures for the *fixed*-tau sweep (see
main.generate_system_tau): fixed ball budget k, tau itself is the swept
independent variable -- a single, dataset-independent set of absolute L2
distances applied identically to all 3 datasets, unlike plot_system.py's
sweep where tau is reverse-engineered per dataset to hit a target
redundancy rate. redundancy_rate is therefore a reported *outcome* of a
given tau here, not the quantity tau is solved for.

Five panels, one line per dataset:
  (a) Throughput Improvement (%) of the filtered pipeline over the
      unfiltered flat-index baseline.
  (b) Mean Full Index Accesses (queries that bypass the filter and reach
      the backend index), averaged over the seed sweep.
  (c) Empirical FPR = FP / (FP + TN).
  (d) Threshold-expanded geometric objective log10 J_tau(C) = log10 sum_i
      (r_i + tau)^d.
  (e) Measured redundancy rate -- how much of the query stream a given tau
      actually counts as a ground-truth duplicate.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plots.plot_common import DATASETS, DATASET_COLORS, apply_style, style_axes, save_figure, save_dataset_legend, mean_ci95, wrap_label

INPUT_PATH = "results/system_tau.csv"
OUTDIR = "plots/system_tau/"
K_FIXED: Optional[int] = None  # None => the single k present in the data
DPI = 300
EMIT_PDF = False
DEMO = False

_REQUIRED_COLUMNS = [
    "dataset", "k", "tau", "seed", "redundancy_rate",
    "throughput_improvement_pct", "mean_index_accesses",
    "fpr", "fnr", "j_tau", "log10_j_tau", "n_balls",
]


@dataclasses.dataclass
class Config:
    k_fixed: int
    dpi: int
    emit_pdf: bool
    outdir: str


def _synthesize_demo_data() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    taus = np.linspace(0.1, 1.0, 10)
    seeds = [0, 1, 2]
    k_fixed = 256

    base_redundancy = {"eli5": 1.6, "mmlu": 1.3, "wildchat": 1.9}  # tau -> redundancy_rate curvature
    base_speedup = {"eli5": 180.0, "mmlu": 150.0, "wildchat": 210.0}
    base_fpr = {"eli5": 0.02, "mmlu": 0.03, "wildchat": 0.015}
    base_log10_j = {"eli5": 140.0, "mmlu": 128.0, "wildchat": 115.0}
    n_queries = 9000

    rows = []
    for dataset in DATASETS:
        for tau in taus:
            redundancy_rate = float(np.clip(1.0 - np.exp(-base_redundancy[dataset] * tau), 0.0, 0.98))
            for seed in seeds:
                noise = rng.uniform(0.95, 1.05)
                throughput_improvement_pct = base_speedup[dataset] * (1.0 - redundancy_rate) * noise
                mean_index_accesses = n_queries * redundancy_rate * rng.uniform(0.97, 1.03)
                fpr = base_fpr[dataset] * (1.0 + redundancy_rate) * rng.uniform(0.9, 1.1)
                log10_j_tau = base_log10_j[dataset] + k_fixed / 64.0 * np.log10(1.0 + tau) * 40 * noise
                rows.append({
                    "dataset": dataset, "k": k_fixed, "tau": float(tau), "seed": seed,
                    "redundancy_rate": redundancy_rate,
                    "throughput_improvement_pct": throughput_improvement_pct,
                    "mean_index_accesses": mean_index_accesses,
                    "fpr": fpr, "fnr": 0.0,
                    "j_tau": float("inf"), "log10_j_tau": log10_j_tau,
                    "n_balls": k_fixed,
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
                bounded_unit: bool = False) -> pd.DataFrame:
    """One line per dataset, plotted against tau itself -- tau is the fixed,
    dataset-independent independent variable in this sweep (contrast
    plot_system.py, where the shared x-axis is the measured redundancy rate
    tau was solved for). Aggregated over the seed sweep by (dataset, tau)."""
    agg = mean_ci95(df, ["dataset", "tau"], value_col)

    apply_style(cfg.dpi)
    fig, ax = plt.subplots()
    for dataset in DATASETS:
        sub = agg[agg["dataset"] == dataset].sort_values("tau")
        if sub.empty:
            continue
        ax.plot(sub["tau"], sub["mean"], color=DATASET_COLORS[dataset], marker="o",
               markersize=2.5, linewidth=0.8, zorder=2)
        if (sub["ci95"] > 0).any():
            ax.errorbar(sub["tau"], sub["mean"], yerr=sub["ci95"], fmt="none",
                       ecolor=DATASET_COLORS[dataset], elinewidth=0.5, capsize=1.5, capthick=0.5, zorder=3)

    if bounded_unit:
        ax.set_ylim(-0.05, 1.05)

    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(wrap_label(ylabel))
    ax.text(0.03, 0.95, rf"$k = {cfg.k_fixed}$", transform=ax.transAxes, ha="left", va="top", fontsize=5.5)
    style_axes(ax)

    save_figure(fig, filename, cfg.outdir, cfg.dpi, cfg.emit_pdf)
    return agg


def panel_throughput_improvement(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    return _line_panel(df, "throughput_improvement_pct", "Throughput Improvement (%)",
                       "system-tau-throughput-improvement.png", cfg)


def panel_index_accesses(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    return _line_panel(df, "mean_index_accesses", "Mean Full Index Accesses",
                       "system-tau-index-accesses.png", cfg)


def panel_fpr(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    return _line_panel(df, "fpr", "FPR", "system-tau-fpr.png", cfg, bounded_unit=True)


def panel_geometric_objective(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    # Hand-wrapped into two self-contained "$...$" spans -- wrap_label passes
    # LaTeX through unchanged since generic word-wrapping would otherwise
    # split a single "$...$" expression across lines and break mathtext.
    ylabel = "$\\log_{10}$\n$J_\\tau(\\mathcal{C})$"
    return _line_panel(df, "log10_j_tau", ylabel, "system-tau-geometric-objective.png", cfg)


def panel_redundancy_rate(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Redundancy rate as a function of tau -- the calibration curve this
    fixed-tau sweep lets a reader look up directly, since (unlike
    plot_system.py's sweep) tau here isn't chosen to hit a target rate."""
    return _line_panel(df, "redundancy_rate", "Redundancy Rate", "system-tau-redundancy-rate.png",
                       cfg, bounded_unit=True)


def print_summary(df: pd.DataFrame, cfg: Config) -> None:
    print(f"\n--- System (fixed tau-sweep) summary (k={cfg.k_fixed}) ---")
    cols = ("throughput_improvement_pct", "mean_index_accesses", "fpr", "log10_j_tau", "redundancy_rate")
    aggs = {c: mean_ci95(df, ["dataset", "tau"], c) for c in cols}
    for dataset in DATASETS:
        for tau in sorted(df[df["dataset"] == dataset]["tau"].unique()):
            def _mean(c):
                row = aggs[c][(aggs[c]["dataset"] == dataset) & (aggs[c]["tau"] == tau)]
                return float(row["mean"].iloc[0]) if len(row) else float("nan")
            print(f"{dataset:<10} tau={tau:.3f} redundancy={_mean('redundancy_rate'):.3f} "
                 f"throughput_gain={_mean('throughput_improvement_pct'):+.1f}% "
                 f"index_accesses={_mean('mean_index_accesses'):.0f} "
                 f"fpr={_mean('fpr'):.4f} log10_J_tau={_mean('log10_j_tau'):.2f}")
    print()


def generate_system_tau_figures(df: Optional[pd.DataFrame] = None, outdir: str = OUTDIR,
                                k_fixed: Optional[int] = K_FIXED, dpi: int = DPI, emit_pdf: bool = EMIT_PDF) -> None:
    if df is None:
        df = load_data()
    else:
        _validate_schema(df)

    cfg = resolve_config(df, k_fixed=k_fixed, dpi=dpi, emit_pdf=emit_pdf, outdir=outdir)
    panel_throughput_improvement(df, cfg)
    panel_index_accesses(df, cfg)
    panel_fpr(df, cfg)
    panel_geometric_objective(df, cfg)
    panel_redundancy_rate(df, cfg)
    save_dataset_legend(cfg.outdir, cfg.dpi, cfg.emit_pdf)
    print_summary(df, cfg)


def main() -> None:
    generate_system_tau_figures()


if __name__ == "__main__":
    main()
