"""CapCoverFilter figures for the tau sweep (see main.generate_cap_cover_system_tau):
BallCoverFilter's plot_system_tau.py, with the ball-specific geometric
objective/volume-bounds panels (built from radii/centers, which CapCoverFilter
has no analogue for) replaced by a single n_caps panel. tau is the swept
independent variable within a given k, exactly as in plot_system_tau.

Panels, one line (or band) per dataset:
  (a) Throughput Improvement (%) of the filtered pipeline over the
      unfiltered flat-index baseline.
  (b) Mean Full Index Accesses (queries that bypass the filter and reach
      the backend index), averaged over the seed sweep.
  (c) Empirical FPR = FP / (FP + TN).
  (d) Measured redundancy rate -- how much of the query stream a given tau
      actually counts as a ground-truth duplicate.
  (e) Predicted positive rate (%) -- mean_index_accesses / n_queries.
  (f) Cap count of the finished cover (constant across tau within a k, so
      shown mainly to confirm the budget landed where expected).
"""

from __future__ import annotations

import dataclasses
import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plots.plot_common import DATASETS, DATASET_COLORS, DATASET_MARKERS, TAU_LABEL, cosine_distance, apply_style, style_axes, save_figure, save_dataset_legend, mean_std, wrap_label

INPUT_PATH = "results/cap_cover_system_tau.csv"
OUTDIR = "plots/cap_cover_system_tau/"
K_FIXED: Optional[int] = None  # None => the median k present in the data
DPI = 300
EMIT_PDF = False

_REQUIRED_COLUMNS = [
    "dataset", "k", "tau", "seed", "redundancy_rate", "tp_rate_pct",
    "throughput_improvement_pct", "mean_index_accesses", "n_queries", "positive_rate_pct",
    "fpr", "fnr", "n_caps",
]


@dataclasses.dataclass
class Config:
    k_fixed: int
    dpi: int
    emit_pdf: bool
    outdir: str


def _validate_schema(df: pd.DataFrame) -> None:
    missing_cols = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Input missing columns: {missing_cols}. Present: {sorted(df.columns)}")
    missing_ds = set(DATASETS) - set(df["dataset"].unique())
    if missing_ds:
        raise ValueError(f"Missing dataset(s) {sorted(missing_ds)}. Available: {sorted(df['dataset'].unique())}")


def load_data() -> pd.DataFrame:
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH!r}. Run main.generate_cap_cover_system_tau first.")
    df = pd.read_parquet(INPUT_PATH) if INPUT_PATH.endswith(".parquet") else pd.read_csv(INPUT_PATH)
    _validate_schema(df)
    return df


def resolve_config(df: pd.DataFrame, k_fixed: Optional[int] = K_FIXED, dpi: int = DPI,
                   emit_pdf: bool = EMIT_PDF, outdir: str = OUTDIR) -> Config:
    available_k = sorted(df["k"].unique())
    if k_fixed is None:
        k_fixed = int(min(available_k, key=lambda k: abs(k - np.median(available_k))))
    elif k_fixed not in available_k:
        raise ValueError(f"k_fixed={k_fixed} not present. Available: {available_k}")
    return Config(k_fixed=k_fixed, dpi=dpi, emit_pdf=emit_pdf, outdir=outdir)


def _line_panel(df: pd.DataFrame, value_col: str, ylabel: str, filename: str, cfg: Config,
                bounded_unit: bool = False) -> pd.DataFrame:
    sel = df[df["k"] == cfg.k_fixed]
    agg = mean_std(sel, ["dataset", "tau"], value_col)
    agg["tau_cos"] = cosine_distance(agg["tau"])

    apply_style(cfg.dpi)
    fig, ax = plt.subplots()
    for dataset in DATASETS:
        sub = agg[agg["dataset"] == dataset].sort_values("tau")
        if sub.empty:
            continue
        ax.plot(sub["tau_cos"], sub["mean"], color=DATASET_COLORS[dataset], marker=DATASET_MARKERS[dataset],
               markersize=2.5, linewidth=0.8, zorder=2)
        if (sub["std"] > 0).any():
            ax.errorbar(sub["tau_cos"], sub["mean"], yerr=sub["std"], fmt="none",
                       ecolor=DATASET_COLORS[dataset], elinewidth=0.5, capsize=1.5, capthick=0.5, zorder=3)

    if bounded_unit:
        ax.set_ylim(-0.05, 1.05)

    ax.set_xlabel(TAU_LABEL)
    ax.set_ylabel(wrap_label(ylabel))
    style_axes(ax)

    fig.subplots_adjust(left=0.26, right=0.97, top=0.95, bottom=0.24)
    save_figure(fig, filename, cfg.outdir, cfg.dpi, cfg.emit_pdf, tight=False)
    return agg


def panel_throughput_improvement(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    return _line_panel(df, "throughput_improvement_pct", "Throughput Gain (%)",
                       "cap-cover-system-tau-throughput-improvement.png", cfg)


def panel_index_accesses(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    return _line_panel(df, "mean_index_accesses", "Index Accesses",
                       "cap-cover-system-tau-index-accesses.png", cfg)


def panel_fpr(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    return _line_panel(df, "fpr", "FPR", "cap-cover-system-tau-fpr.png", cfg, bounded_unit=True)


def panel_redundancy_rate(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    return _line_panel(df, "redundancy_rate", "Redundancy Rate", "cap-cover-system-tau-redundancy-rate.png",
                       cfg, bounded_unit=True)


def panel_positive_rate(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    return _line_panel(df, "positive_rate_pct", "Predicted Rate (%)",
                       "cap-cover-system-tau-positive-rate.png", cfg)


def panel_n_caps(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    return _line_panel(df, "n_caps", "Caps in Cover", "cap-cover-system-tau-n-caps.png", cfg)


def print_summary(df: pd.DataFrame, cfg: Config) -> None:
    print(f"\n--- CapCover (tau-sweep) summary (k={cfg.k_fixed}) ---")
    sel = df[df["k"] == cfg.k_fixed]
    agg = sel.groupby(["dataset", "tau"]).agg(
        redundancy_rate=("redundancy_rate", "mean"),
        positive_rate_pct=("positive_rate_pct", "mean"),
        throughput_improvement_pct=("throughput_improvement_pct", "mean"),
        mean_index_accesses=("mean_index_accesses", "mean"),
        fpr=("fpr", "mean"),
        fnr=("fnr", "mean"),
        n_caps=("n_caps", "mean"),
    ).reset_index()
    for dataset in DATASETS:
        for _, row in agg[agg["dataset"] == dataset].sort_values("tau").iterrows():
            print(f"{dataset:<10} tau={row['tau']:.3f} redundancy={row['redundancy_rate']:.3f} "
                 f"predicted_positive={row['positive_rate_pct']:.1f}% "
                 f"throughput_gain={row['throughput_improvement_pct']:+.1f}% "
                 f"index_accesses={row['mean_index_accesses']:.0f} "
                 f"fpr={row['fpr']:.4f} fnr={row['fnr']:.4f} n_caps={row['n_caps']:.0f}")
    print()


def generate_cap_cover_system_tau_figures(df: Optional[pd.DataFrame] = None, outdir: str = OUTDIR,
                                          k_fixed: Optional[int] = K_FIXED, dpi: int = DPI,
                                          emit_pdf: bool = EMIT_PDF) -> None:
    if df is None:
        df = load_data()
    else:
        _validate_schema(df)

    cfg = resolve_config(df, k_fixed=k_fixed, dpi=dpi, emit_pdf=emit_pdf, outdir=outdir)
    panel_throughput_improvement(df, cfg)
    panel_index_accesses(df, cfg)
    panel_fpr(df, cfg)
    panel_redundancy_rate(df, cfg)
    panel_positive_rate(df, cfg)
    panel_n_caps(df, cfg)
    save_dataset_legend(cfg.outdir, cfg.dpi, cfg.emit_pdf)
    print_summary(df, cfg)


def main() -> None:
    generate_cap_cover_system_tau_figures()


if __name__ == "__main__":
    main()
