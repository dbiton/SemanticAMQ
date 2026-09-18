"""System-evaluation figures for the tau sweep (see main.generate_system_tau):
tau is the swept independent variable within a given k -- a single,
dataset-independent set of absolute L2 distances applied identically to all
datasets. redundancy_rate is therefore a reported *outcome* of a given tau
here, not a quantity tau is solved for. Ball budget k is itself now swept in
the underlying data (main.generate_system_tau's k_fractions), so this module
renders one full panel set at a single representative k at a time (default:
the median available k, same "pick one representative value" pattern
plot_quantization.py uses for tau) -- pass k_fixed explicitly to render a
different one.

Seven panels, one line per dataset:
  (a) Throughput Improvement (%) of the filtered pipeline over the
      unfiltered flat-index baseline.
  (b) Mean Full Index Accesses (queries that bypass the filter and reach
      the backend index), averaged over the seed sweep.
  (c) Empirical FPR = FP / (FP + TN).
  (d) Threshold-expanded geometric objective log10 J_tau(C) = log10 sum_i
      (r_i + tau)^d.
  (e) Measured redundancy rate -- how much of the query stream a given tau
      actually counts as a ground-truth duplicate.
  (f) Predicted positive rate (%) -- mean_index_accesses / n_queries, i.e.
      how much of the stream the *filter* flags as a possible duplicate,
      as opposed to (e)'s ground-truth rate; the gap between the two is
      the filter's false-positive margin at that tau.
  (g) TP rate (%) -- percentage of query vectors that are an actual
      duplicate (closer than tau to a construct vector); the same quantity
      as (e)'s redundancy_rate, just as a percentage under an explicit
      "TP" label rather than a [0,1] fraction.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plots.plot_common import DATASETS, DATASET_COLORS, DATASET_MARKERS, apply_style, style_axes, save_figure, save_dataset_legend, mean_ci95, mean_std, wrap_label

INPUT_PATH = "results/system_tau.csv"
OUTDIR = "plots/system_tau/"
K_FIXED: Optional[int] = None  # None => the median k present in the data
DPI = 300
EMIT_PDF = False
DEMO = False

_REQUIRED_COLUMNS = [
    "dataset", "k", "tau", "seed", "redundancy_rate", "tp_rate_pct",
    "throughput_improvement_pct", "mean_index_accesses", "n_queries", "positive_rate_pct",
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

    # tau -> redundancy_rate curvature
    base_redundancy = {"eli5": 1.6, "mmlu": 1.3, "wildchat": 1.9, "msmarco": 1.4, "hotpotqa": 1.7}
    base_speedup = {"eli5": 180.0, "mmlu": 150.0, "wildchat": 210.0, "msmarco": 165.0, "hotpotqa": 195.0}
    base_fpr = {"eli5": 0.02, "mmlu": 0.03, "wildchat": 0.015, "msmarco": 0.025, "hotpotqa": 0.018}
    base_log10_j = {"eli5": 140.0, "mmlu": 128.0, "wildchat": 115.0, "msmarco": 132.0, "hotpotqa": 121.0}
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
                    "tp_rate_pct": redundancy_rate * 100.0,
                    "throughput_improvement_pct": throughput_improvement_pct,
                    "mean_index_accesses": mean_index_accesses,
                    "n_queries": n_queries,
                    "positive_rate_pct": mean_index_accesses / n_queries * 100.0,
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
        # Median available k, nearest actual value present (k varies per
        # dataset/fold since it's a fraction of each fold's construct-set
        # size, so the exact median float may not itself be a present k).
        k_fixed = int(min(available_k, key=lambda k: abs(k - np.median(available_k))))
    elif k_fixed not in available_k:
        raise ValueError(f"k_fixed={k_fixed} not present. Available: {available_k}")
    return Config(k_fixed=k_fixed, dpi=dpi, emit_pdf=emit_pdf, outdir=outdir)


def _line_panel(df: pd.DataFrame, value_col: str, ylabel: str, filename: str, cfg: Config,
                bounded_unit: bool = False) -> pd.DataFrame:
    """One line per dataset, plotted against tau itself -- tau is the fixed,
    dataset-independent independent variable in this sweep. Aggregated over
    the seed sweep (one seed per K-fold split) by (dataset, tau); error bars
    show +/- 1 standard deviation across folds, not standard error, so they
    read as "how much do folds actually disagree" rather than shrinking
    just because more folds were run. Each dataset also gets its own marker
    shape (DATASET_MARKERS), not just its own color, so lines stay
    distinguishable without relying on color alone. Filtered to cfg.k_fixed
    first -- k now varies per row (see module docstring), so this must
    select one k's rows before aggregating by (dataset, tau) or it would
    silently blend results from different ball budgets together."""
    sel = df[df["k"] == cfg.k_fixed]
    agg = mean_std(sel, ["dataset", "tau"], value_col)

    apply_style(cfg.dpi)
    fig, ax = plt.subplots()
    for dataset in DATASETS:
        sub = agg[agg["dataset"] == dataset].sort_values("tau")
        if sub.empty:
            continue
        ax.plot(sub["tau"], sub["mean"], color=DATASET_COLORS[dataset], marker=DATASET_MARKERS[dataset],
               markersize=2.5, linewidth=0.8, zorder=2)
        if (sub["std"] > 0).any():
            ax.errorbar(sub["tau"], sub["mean"], yerr=sub["std"], fmt="none",
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
    fixed-tau sweep lets a reader look up directly, since tau here isn't
    chosen to hit a target rate."""
    return _line_panel(df, "redundancy_rate", "Redundancy Rate", "system-tau-redundancy-rate.png",
                       cfg, bounded_unit=True)


def panel_tp_rate(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Percentage of query vectors that are an actual duplicate (closer
    than tau to a construct vector) -- the same quantity as
    panel_redundancy_rate, as an explicit "TP" percentage rather than a
    [0,1] fraction."""
    return _line_panel(df, "tp_rate_pct", "TP Rate (%)", "system-tau-tp-rate.png", cfg)


def panel_positive_rate(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Percentage of the query stream the *filter* predicts positive
    (mean_index_accesses / n_queries), as opposed to panel_redundancy_rate's
    ground-truth rate -- the gap between the two at a given tau is the
    filter's false-positive margin."""
    return _line_panel(df, "positive_rate_pct", "Predicted Positive Rate (%)",
                       "system-tau-positive-rate.png", cfg)


def print_summary(df: pd.DataFrame, cfg: Config) -> None:
    print(f"\n--- System (tau-sweep) summary (k={cfg.k_fixed}) ---")
    sel = df[df["k"] == cfg.k_fixed]
    cols = ("throughput_improvement_pct", "mean_index_accesses", "fpr", "log10_j_tau", "redundancy_rate",
            "positive_rate_pct")
    aggs = {c: mean_ci95(sel, ["dataset", "tau"], c) for c in cols}
    for dataset in DATASETS:
        for tau in sorted(sel[sel["dataset"] == dataset]["tau"].unique()):
            def _mean(c):
                row = aggs[c][(aggs[c]["dataset"] == dataset) & (aggs[c]["tau"] == tau)]
                return float(row["mean"].iloc[0]) if len(row) else float("nan")
            print(f"{dataset:<10} tau={tau:.3f} redundancy={_mean('redundancy_rate'):.3f} "
                 f"predicted_positive={_mean('positive_rate_pct'):.1f}% "
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
    panel_tp_rate(df, cfg)
    panel_positive_rate(df, cfg)
    save_dataset_legend(cfg.outdir, cfg.dpi, cfg.emit_pdf)
    print_summary(df, cfg)


def main() -> None:
    generate_system_tau_figures()


if __name__ == "__main__":
    main()
