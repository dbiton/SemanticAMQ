"""Streaming-evaluation figures: (a) absorption rate vs. stream position, (b)
FPR by rebuild policy vs. stream position, (c) final FPR vs. k per merge
criterion, (d) rebuild-policy cost/quality Pareto. Reads one long-format CSV
with a `metric` column (absorption | rebuild | merge_criterion) selecting
which panel(s) a row feeds.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter

from plots.plot_common import DATASETS, DATASET_LABELS, DATASET_COLORS, apply_style, save_figure, save_dataset_legend, mean_ci95

INPUT_PATH = "results/streaming.csv"
OUTDIR = "plots/streaming/"
K_FIXED: Optional[int] = None
T_REPRESENTATIVE: Optional[int] = None
DPI = 300
EMIT_PDF = False
DEMO = False

POLICIES = ["streaming", "periodic", "oracle"]
POLICY_LABELS = {"streaming": "Streaming (no rebuild)", "periodic": "Periodic rebuild", "oracle": "Oracle rebuild"}
POLICY_COLORS = {"streaming": "#D55E00", "periodic": "#CC79A7", "oracle": "#000000"}
POLICY_MARKERS = {"streaming": "o", "periodic": "s", "oracle": "^"}

MERGE_CRITERIA = ["log_space", "r_pow_d", "sum_radii", "min_radius_increase"]
MERGE_CRITERION_LABELS = {
    "log_space": r"$\log$-space $r^d$", "r_pow_d": r"raw $r^d$",
    "sum_radii": r"$\Sigma r$", "min_radius_increase": "min radius increase",
}
MERGE_CRITERION_COLORS = {
    "log_space": "#0072B2", "r_pow_d": "#D55E00", "sum_radii": "#E69F00", "min_radius_increase": "#009E73",
}

_REQUIRED_COLUMNS = ["metric", "dataset", "k", "seed", "position", "absorption_rate", "policy", "T", "fpr",
                     "amortized_cost_s", "merge_criterion"]


@dataclasses.dataclass
class Config:
    k_fixed: int
    t_representative: int
    dpi: int
    emit_pdf: bool
    outdir: str


def _synthesize_demo_data() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    positions = np.unique(np.round(np.logspace(np.log10(50), np.log10(8000), 10)).astype(int))
    seeds = [0, 1, 2]
    k_fixed = 128
    T_values = [200, 400, 800, 1600, 4000]
    rows = []

    plateau = {"eli5": 0.94, "mmlu": 0.90, "wildchat": 0.97}
    rate = {"eli5": 1.6, "mmlu": 1.3, "wildchat": 2.0}
    for dataset in DATASETS:
        for seed in seeds:
            for pos in positions:
                x = pos / positions[-1]
                mean = plateau[dataset] * (1.0 - np.exp(-rate[dataset] * x * 6))
                rows.append({"metric": "absorption", "dataset": dataset, "k": k_fixed, "seed": seed,
                            "position": int(pos), "absorption_rate": float(np.clip(mean + rng.normal(0, 0.02), 0, 1))})

    for seed in seeds:
        for policy, T in [("streaming", None), ("oracle", None)] + [("periodic", t) for t in T_values]:
            frac = 0.0 if policy == "streaming" else (1.0 if policy == "oracle" else
                  1.0 - (T - min(T_values)) / (max(T_values) - min(T_values) + 1e-9))
            cum_time = 0.0
            for pos in positions:
                step = {"streaming": 0.00012, "oracle": 0.02, "periodic": 0.00012 + frac * 0.006}[policy]
                cum_time += step * pos / len(positions)
                fpr = (0.03 - 0.022 * frac) * np.exp(-pos / 2000.0) + 0.005 + 0.005 * (1 - frac)
                rows.append({"metric": "rebuild", "dataset": "eli5", "policy": policy, "T": T, "k": k_fixed,
                            "seed": seed, "position": int(pos), "fpr": float(fpr * rng.uniform(0.9, 1.1)),
                            "cumulative_time_s": cum_time, "amortized_cost_s": cum_time / pos})

    base = {"log_space": 0.02, "r_pow_d": 0.021, "sum_radii": 0.05, "min_radius_increase": 0.045}
    degrade = {"log_space": 1.0, "r_pow_d": 1.9, "sum_radii": 1.3, "min_radius_increase": 1.2}
    for k in [16, 32, 64, 128, 256, 512]:
        kf = k / 512.0
        for crit in MERGE_CRITERIA:
            for seed in seeds:
                fpr = base[crit] / (1.0 + 4.0 * (1 - kf)) * (1.0 + (degrade[crit] - 1.0) * kf)
                rows.append({"metric": "merge_criterion", "dataset": "eli5", "k": k, "seed": seed,
                            "merge_criterion": crit, "fpr": float(fpr * rng.uniform(0.9, 1.1))})
    return pd.DataFrame(rows)


def _validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Input missing columns: {missing}. Present: {sorted(df.columns)}")
    missing_metrics = {"absorption", "rebuild", "merge_criterion"} - set(df["metric"].unique())
    if missing_metrics:
        raise ValueError(f"Missing metric(s) {sorted(missing_metrics)}")
    missing_ds = set(DATASETS) - set(df.loc[df["metric"] == "absorption", "dataset"].unique())
    if missing_ds:
        raise ValueError(f"absorption rows missing dataset(s) {sorted(missing_ds)}")
    missing_pol = set(POLICIES) - set(df.loc[df["metric"] == "rebuild", "policy"].dropna().unique())
    if missing_pol:
        raise ValueError(f"rebuild rows missing policy(ies) {sorted(missing_pol)}")


def load_data() -> pd.DataFrame:
    if DEMO:
        df = _synthesize_demo_data()
    else:
        if not os.path.exists(INPUT_PATH):
            raise FileNotFoundError(f"Input file not found: {INPUT_PATH!r}. Set DEMO=True or fix INPUT_PATH.")
        df = pd.read_parquet(INPUT_PATH) if INPUT_PATH.endswith(".parquet") else pd.read_csv(INPUT_PATH)
    _validate_schema(df)
    return df


def resolve_config(df: pd.DataFrame, k_fixed: Optional[int] = K_FIXED, t_representative: Optional[int] = T_REPRESENTATIVE,
                   dpi: int = DPI, emit_pdf: bool = EMIT_PDF, outdir: str = OUTDIR) -> Config:
    available_k = sorted(df.loc[df["metric"] == "absorption", "k"].unique())
    if k_fixed is None:
        k_fixed = int(available_k[0])
    elif k_fixed not in available_k:
        raise ValueError(f"k_fixed={k_fixed} not present. Available: {available_k}")

    periodic_T = sorted(df.loc[(df["metric"] == "rebuild") & (df["policy"] == "periodic"), "T"].dropna().unique())
    if not periodic_T:
        raise ValueError("No 'periodic' policy rows to pick a representative T from.")
    if t_representative is None:
        t_representative = int(periodic_T[len(periodic_T) // 2])
    elif t_representative not in periodic_T:
        raise ValueError(f"T_representative={t_representative} not present. Available: {periodic_T}")

    return Config(k_fixed=k_fixed, t_representative=t_representative, dpi=dpi, emit_pdf=emit_pdf, outdir=outdir)


def _style_log_y(ax: plt.Axes) -> None:
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="y", which="minor", length=1.2, width=0.4)


def _style_axes(ax: plt.Axes) -> None:
    ax.grid(axis="both", which="major", linewidth=0.35, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.5)


def panel_absorption(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    sel = df[(df["metric"] == "absorption") & (df["k"] == cfg.k_fixed)]
    agg = mean_ci95(sel, ["dataset", "position"], "absorption_rate")

    apply_style(cfg.dpi)
    fig, ax = plt.subplots()
    for dataset in DATASETS:
        sub = agg[agg["dataset"] == dataset].sort_values("position")
        if sub.empty:
            continue
        ax.plot(sub["position"], sub["mean"], color=DATASET_COLORS[dataset], marker="o", markersize=2.5,
               linewidth=0.8, zorder=2)
        if (sub["ci95"] > 0).any():
            ax.errorbar(sub["position"], sub["mean"], yerr=sub["ci95"], fmt="none", ecolor=DATASET_COLORS[dataset],
                       elinewidth=0.5, capsize=1.5, capthick=0.5, zorder=3)

    ax.axhline(1.0, color="black", linewidth=0.4, linestyle=":", zorder=1)
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Stream Position")
    ax.set_ylabel("Absorption Rate")
    ax.text(0.03, 0.06, rf"$k = {cfg.k_fixed}$", transform=ax.transAxes, ha="left", va="bottom", fontsize=5.5)
    _style_axes(ax)

    save_figure(fig, "streaming-absorption.png", cfg.outdir, cfg.dpi, cfg.emit_pdf)
    return agg


def panel_rebuild_fpr(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    sel = df[(df["metric"] == "rebuild") & ((df["policy"] != "periodic") | (df["T"] == cfg.t_representative))]
    agg = mean_ci95(sel, ["policy", "position"], "fpr")

    apply_style(cfg.dpi)
    fig, ax = plt.subplots()
    for policy in POLICIES:
        sub = agg[agg["policy"] == policy].sort_values("position")
        if sub.empty:
            continue
        ax.plot(sub["position"], sub["mean"], color=POLICY_COLORS[policy], marker=POLICY_MARKERS[policy],
               markersize=2.5, linewidth=0.8, zorder=2)
        if (sub["ci95"] > 0).any():
            ax.errorbar(sub["position"], sub["mean"], yerr=sub["ci95"], fmt="none", ecolor=POLICY_COLORS[policy],
                       elinewidth=0.5, capsize=1.5, capthick=0.5, zorder=3)

    ax.set_xscale("log")
    _style_log_y(ax)
    ax.set_xlabel("Stream Position")
    ax.set_ylabel("FPR")
    ax.text(0.03, 0.06, rf"$k = {cfg.k_fixed},\ T = {cfg.t_representative}$",
           transform=ax.transAxes, ha="left", va="bottom", fontsize=5.5)
    _style_axes(ax)

    save_figure(fig, "streaming-fpr.png", cfg.outdir, cfg.dpi, cfg.emit_pdf)
    return agg


def panel_merge_criteria(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    sel = df[df["metric"] == "merge_criterion"]
    agg = mean_ci95(sel, ["merge_criterion", "k"], "fpr")

    apply_style(cfg.dpi)
    fig, ax = plt.subplots()
    for crit in MERGE_CRITERIA:
        sub = agg[agg["merge_criterion"] == crit].sort_values("k")
        if sub.empty:
            continue
        ax.plot(sub["k"], sub["mean"], color=MERGE_CRITERION_COLORS[crit], marker="o", markersize=2.5,
               linewidth=0.8, zorder=2)
        if (sub["ci95"] > 0).any():
            ax.errorbar(sub["k"], sub["mean"], yerr=sub["ci95"], fmt="none", ecolor=MERGE_CRITERION_COLORS[crit],
                       elinewidth=0.5, capsize=1.5, capthick=0.5, zorder=3)

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(LogLocator(base=2.0))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=2.0))
    _style_log_y(ax)
    ax.set_xlabel("Filter Size (k)")
    ax.set_ylabel("Final FPR")
    _style_axes(ax)

    save_figure(fig, "streaming-merge-criteria.png", cfg.outdir, cfg.dpi, cfg.emit_pdf)
    return agg


def panel_rebuild_pareto(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """One point per policy: x = amortized cost at the final checkpoint (mean
    over seeds), y = mean FPR across every checkpoint/seed for that policy."""
    sel = df[df["metric"] == "rebuild"]
    last_pos = sel.groupby(["policy", "T", "seed"], dropna=False)["position"].transform("max")
    final_rows = sel[sel["position"] == last_pos]

    cost = final_rows.groupby(["policy", "T"], dropna=False)["amortized_cost_s"].mean().reset_index(name="cost")
    quality = sel.groupby(["policy", "T"], dropna=False)["fpr"].mean().reset_index(name="mean_fpr")
    points = cost.merge(quality, on=["policy", "T"])

    apply_style(cfg.dpi)
    fig, ax = plt.subplots()

    periodic = points[points["policy"] == "periodic"].sort_values("T")
    if not periodic.empty:
        ax.plot(periodic["cost"], periodic["mean_fpr"], color=POLICY_COLORS["periodic"], linewidth=0.6,
               linestyle="--", zorder=1)
        ax.scatter(periodic["cost"], periodic["mean_fpr"], color=POLICY_COLORS["periodic"],
                  marker=POLICY_MARKERS["periodic"], s=10, zorder=2)
        for _, row in periodic.iterrows():
            ax.annotate(f"T={int(row['T'])}", (row["cost"], row["mean_fpr"]), fontsize=4.5,
                       xytext=(2, 2), textcoords="offset points")

    for policy in ("streaming", "oracle"):
        sub = points[points["policy"] == policy]
        if not sub.empty:
            ax.scatter(sub["cost"], sub["mean_fpr"], color=POLICY_COLORS[policy], marker=POLICY_MARKERS[policy],
                      s=14, zorder=3)

    ax.set_xscale("log")
    _style_log_y(ax)
    ax.set_xlabel("Amortized Cost / Insertion (s)")
    ax.set_ylabel("Mean FPR")
    _style_axes(ax)

    save_figure(fig, "streaming-rebuild-pareto.png", cfg.outdir, cfg.dpi, cfg.emit_pdf)
    return points


def save_policy_legend(cfg: Config) -> None:
    apply_style(cfg.dpi)
    fig, ax = plt.subplots(figsize=(4.6, 0.25))
    ax.axis("off")
    handles = [plt.Line2D([0], [0], color=POLICY_COLORS[p], marker=POLICY_MARKERS[p], markersize=4, linewidth=1.0)
              for p in POLICIES]
    fig.legend(handles, [POLICY_LABELS[p] for p in POLICIES], loc="center", ncol=3, frameon=False, fontsize=7)
    fig.patch.set_alpha(0.0)
    save_figure(fig, "streaming-policy-legend.png", cfg.outdir, cfg.dpi, cfg.emit_pdf, transparent=True)


def save_merge_criterion_legend(cfg: Config) -> None:
    apply_style(cfg.dpi)
    fig, ax = plt.subplots(figsize=(4.6, 0.25))
    ax.axis("off")
    handles = [plt.Line2D([0], [0], color=MERGE_CRITERION_COLORS[c], marker="o", markersize=4, linewidth=1.0)
              for c in MERGE_CRITERIA]
    fig.legend(handles, [MERGE_CRITERION_LABELS[c] for c in MERGE_CRITERIA], loc="center", ncol=2,
              frameon=False, fontsize=7)
    fig.patch.set_alpha(0.0)
    save_figure(fig, "streaming-merge-criterion-legend.png", cfg.outdir, cfg.dpi, cfg.emit_pdf, transparent=True)


def print_summary(df: pd.DataFrame, cfg: Config) -> None:
    print(f"\n--- Streaming summary (k={cfg.k_fixed}, T={cfg.t_representative}) ---")
    abs_final = df[(df["metric"] == "absorption") & (df["k"] == cfg.k_fixed)]
    for dataset in DATASETS:
        sub = abs_final[abs_final["dataset"] == dataset]
        if sub.empty:
            continue
        last = sub[sub["position"] == sub["position"].max()]
        print(f"{DATASET_LABELS[dataset]:<10} final absorption rate: {last['absorption_rate'].mean():.3f}")
    print()


def generate_streaming_figures(df: Optional[pd.DataFrame] = None, outdir: str = OUTDIR,
                               k_fixed: Optional[int] = K_FIXED, T_representative: Optional[int] = T_REPRESENTATIVE,
                               dpi: int = DPI, emit_pdf: bool = EMIT_PDF) -> None:
    if df is None:
        df = load_data()
    else:
        _validate_schema(df)

    cfg = resolve_config(df, k_fixed=k_fixed, t_representative=T_representative, dpi=dpi, emit_pdf=emit_pdf, outdir=outdir)
    panel_absorption(df, cfg)
    panel_rebuild_fpr(df, cfg)
    panel_merge_criteria(df, cfg)
    panel_rebuild_pareto(df, cfg)
    save_dataset_legend(cfg.outdir, cfg.dpi, cfg.emit_pdf)
    save_policy_legend(cfg)
    save_merge_criterion_legend(cfg)
    print_summary(df, cfg)


def main() -> None:
    generate_streaming_figures()


if __name__ == "__main__":
    main()
