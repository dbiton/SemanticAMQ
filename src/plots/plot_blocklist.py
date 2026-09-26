"""Blocklist-workload figures (see main.generate_blocklist).

Ball budget k is fixed (one representative value, see generate_blocklist's
docstring) -- varying the budget is generate_budget's job, at its own fixed
tau. This module's whole figure set is the complementary view: tau swept at
that one fixed budget.

Panels:
  (a) FPR vs tau, pooled over both negative classes.
  (b) FPR on HARD negatives vs tau -- members of duplicate clusters that were
      NOT blocklisted. Same local geometry as B (tight, high-multiplicity),
      so only a filter that actually localises B rejects them. This is the
      headline number.
  (c) FPR on EASY negatives vs tau -- singletons, i.e. background traffic.
  (d) Geometric objective log10 J_tau(C) = log10 sum_i (r_i + tau)^d vs tau --
      the overlap-blind sum of ball volumes once every ball is expanded by tau.
  (e) Index accesses saved vs tau.

All five share one legend, written separately as blocklist-legend.png so
the panels can be placed side by side without repeating it.

Absolute tau is not comparable between corpora -- the distance at which a
duplicate sits is a property of each embedding set. Read each panel as one
corpus's response curve; the dotted vertical marks that corpus's own delta
(d_pos_med, the median true-duplicate distance).
"""

from __future__ import annotations

import dataclasses
import os
from typing import Callable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plots.plot_common import apply_style, style_axes, save_figure, mean_std, wrap_label

INPUT_PATH = "results/blocklist.csv"
OUTDIR = "plots/blocklist/"
DPI = 300
EMIT_PDF = False

# Self-contained styling: these corpora are deliberately not in
# plot_common.DATASETS (that list is the 5-dataset sweep subset, and
# every other plot module validates against it).
BLOCKLIST_DATASETS = ["wildchat", "ccnews", "wikianswers", "quora", "chatgpt_paraphrases"]
BLOCKLIST_LABELS = {
    "wildchat": "WildChat", "ccnews": "CC-News", "wikianswers": "WikiAnswers",
    "quora": "Quora", "chatgpt_paraphrases": "ChatGPT-Paraphrases",
}
# Seaborn "deep" palette, extended by two more of its own entries so the two
# new corpora stay in the same family rather than introducing a new one.
BLOCKLIST_COLORS = {
    "wildchat": "#4C72B0", "ccnews": "#DD8452", "wikianswers": "#55A868",
    "quora": "#C44E52", "chatgpt_paraphrases": "#8172B2",
}
BLOCKLIST_MARKERS = {
    "wildchat": "o", "ccnews": "s", "wikianswers": "^",
    "quora": "D", "chatgpt_paraphrases": "v",
}

_REQUIRED_COLUMNS = [
    "dataset", "seed", "sweep", "k", "k_fraction", "n_reference", "compression", "r_med",
    "tau", "recall", "fpr", "fpr_hard", "fpr_easy", "accesses_saved_pct",
    "log10_j_tau", "d_hard_med", "d_pos_med",
]

# runner.blocklist_tau_worker (mode 'blocklist') only emits sweep="tau" rows:
# the same cover swept over a fixed tau grid at the one fixed k
# generate_blocklist submits, which every panel here reads. SWEEP_TAU is kept
# as a named constant (rather than every reader hardcoding "tau") since it's
# shared with plot_budget, whose rows come from runner.budget_worker instead.
SWEEP_TAU = "tau"


@dataclasses.dataclass
class Config:
    dpi: int
    emit_pdf: bool
    outdir: str


def validate_schema(df: pd.DataFrame) -> None:
    """Shared with plot_budget: both render sweep="tau" rows from the same
    runner._blocklist_cover_rows core (blocklist_tau_worker/budget_worker
    respectively), so both require the same columns."""
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Input missing columns: {missing}. Present: {sorted(df.columns)}")


def load_data() -> pd.DataFrame:
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH!r}")
    df = pd.read_csv(INPUT_PATH)
    validate_schema(df)
    return df


def save_blocklist_legend(outdir: str, dpi: int, emit_pdf: bool = False,
                          filename: str = "blocklist-legend.png") -> None:
    """One shared legend as its own figure, mirroring
    plot_common.save_dataset_legend -- which cannot be reused directly because
    these three corpora are deliberately outside plot_common.DATASETS.

    Kept separate from the panels for the same reason that module does it: the
    panels are placed side by side in the paper, an in-axes legend would be
    repeated three times and would cover data in whichever panel is densest,
    and each panel's axes would end up a different size to accommodate it."""
    apply_style(dpi)
    fig, ax = plt.subplots(figsize=(6.4, 0.25))
    ax.axis("off")
    handles = [
        plt.Line2D([0], [0], color=BLOCKLIST_COLORS[d], marker=BLOCKLIST_MARKERS[d],
                   markersize=4, linewidth=0.8)
        for d in BLOCKLIST_DATASETS
    ]
    labels = [BLOCKLIST_LABELS[d] for d in BLOCKLIST_DATASETS]
    fig.legend(handles, labels, loc="center", ncol=len(BLOCKLIST_DATASETS),
               frameon=False, fontsize=8.4)
    fig.patch.set_alpha(0.0)
    save_figure(fig, filename, outdir, dpi, emit_pdf, transparent=True)


def line_panel(df: pd.DataFrame, value_col: str, ylabel: str, filename: str, cfg: Config,
               bounded_unit: bool = False, logx: bool = True,
               sweep: str = SWEEP_TAU,
               x_from: Callable[[pd.Series], pd.Series] = lambda kf: 1.0 / kf,
               xlabel: str = r"Compression $|X|/k$",
               annotation: Optional[str] = None) -> pd.DataFrame:
    """One line per corpus against a function of the ball budget, averaged over
    folds with +/- 1 std error bars (same convention as plot_system_tau).

    Always groups on k_fraction -- the nominal knob, identical across datasets
    and folds -- and only then maps it onto the x-axis via `x_from`, so a
    caller can plot against compression (|X|/k, the default) or against the
    budget itself (see plot_budget) without grouping on a realised quantity
    that differs per dataset and per fold."""
    agg = mean_std(df[df["sweep"] == sweep], ["dataset", "k_fraction"], value_col)
    agg["x"] = x_from(agg["k_fraction"])

    apply_style(cfg.dpi)
    fig, ax = plt.subplots()
    for dataset in BLOCKLIST_DATASETS:
        sub = agg[agg["dataset"] == dataset].sort_values("x")
        if sub.empty:
            continue
        ax.plot(sub["x"], sub["mean"], color=BLOCKLIST_COLORS[dataset],
                marker=BLOCKLIST_MARKERS[dataset], markersize=2.5, linewidth=0.8, zorder=2)
        if (sub["std"] > 0).any():
            ax.errorbar(sub["x"], sub["mean"], yerr=sub["std"], fmt="none",
                        ecolor=BLOCKLIST_COLORS[dataset], elinewidth=0.5, capsize=1.5,
                        capthick=0.5, zorder=3)

    if bounded_unit:
        ax.set_ylim(-0.05, 1.05)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(wrap_label(ylabel))
    if annotation:
        ax.text(0.97, 0.06, annotation, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=6.0)
    style_axes(ax)
    fig.subplots_adjust(left=0.26, right=0.97, top=0.95, bottom=0.24)
    save_figure(fig, filename, cfg.outdir, cfg.dpi, cfg.emit_pdf, tight=False)
    return agg


def resolve_tau_kfraction(df: pd.DataFrame, k_fraction: Optional[float] = None) -> float:
    """Which ball budget every panel renders, as the nominal k_fraction.
    generate_blocklist now submits one fixed k_fraction, so this trivially
    resolves to that single value; the median-of-available fallback (the
    same "pick one representative value" pattern plot_system_tau uses for k)
    only matters for older multi-k data or an explicit override.

    Selects on k_fraction rather than the realised compression because the
    latter differs per dataset and per fold, so filtering on a single
    compression value would match one corpus and silently drop the others.
    """
    available = sorted(df[df["sweep"] == SWEEP_TAU]["k_fraction"].unique())
    if not available:
        raise ValueError("No tau-sweep rows present; rerun generate_blocklist.")
    if k_fraction is None:
        return float(min(available, key=lambda c: abs(c - np.median(available))))
    if k_fraction not in available:
        raise ValueError(f"k_fraction={k_fraction} not present. Available: {available}")
    return float(k_fraction)


def _tau_panel(df: pd.DataFrame, value_col: str, ylabel: str, filename: str, cfg: Config,
               k_fraction: float, bounded_unit: bool = False,
               horizontal_grid: bool = True) -> pd.DataFrame:
    """One line per corpus against tau, at a single compression ratio.

    Absolute tau is not comparable across corpora -- the distance at which a
    duplicate sits differs per embedding set, which is exactly why the
    headline workload solves tau for a target instead of fixing it. These
    panels are for reading one corpus's response curve; the vertical dashed
    line marks that corpus's own duplicate distance delta, i.e. the smallest
    tau at which its true positives start being caught.
    """
    sel = df[(df["sweep"] == SWEEP_TAU) & (np.isclose(df["k_fraction"], k_fraction))]
    agg = mean_std(sel, ["dataset", "tau"], value_col)

    apply_style(cfg.dpi)
    fig, ax = plt.subplots()
    for dataset in BLOCKLIST_DATASETS:
        sub = agg[agg["dataset"] == dataset].sort_values("tau")
        if sub.empty:
            continue
        ax.plot(sub["tau"], sub["mean"], color=BLOCKLIST_COLORS[dataset],
                marker=BLOCKLIST_MARKERS[dataset], markersize=2.5, linewidth=0.8, zorder=2)
        if (sub["std"] > 0).any():
            ax.errorbar(sub["tau"], sub["mean"], yerr=sub["std"], fmt="none",
                        ecolor=BLOCKLIST_COLORS[dataset], elinewidth=0.5, capsize=1.5,
                        capthick=0.5, zorder=3)
        d_pos = sel[sel["dataset"] == dataset]["d_pos_med"]
        if len(d_pos):
            ax.axvline(float(d_pos.mean()), color=BLOCKLIST_COLORS[dataset],
                       linestyle=":", linewidth=0.5, alpha=0.8, zorder=1)

    if bounded_unit:
        ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(wrap_label(ylabel))
    ax.text(0.97, 0.06, r"dotted: $\delta$",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.0)
    style_axes(ax)
    if not horizontal_grid:
        # style_axes grids both axes for the whole project; drop just the
        # y-axis (horizontal) lines here. Overridden after the shared call
        # rather than by editing style_axes, which every other figure uses.
        ax.grid(visible=False, axis="y")
    fig.subplots_adjust(left=0.26, right=0.97, top=0.95, bottom=0.24)
    save_figure(fig, filename, cfg.outdir, cfg.dpi, cfg.emit_pdf, tight=False)
    return agg


def panel_fpr_vs_tau(df, cfg, k_fraction):
    """Pooled FPR over both negative classes as tau grows."""
    return _tau_panel(df, "fpr", "FPR", "blocklist-fpr-vs-tau.png", cfg,
                      k_fraction, bounded_unit=True)


def panel_fpr_hard_vs_tau(df, cfg, k_fraction):
    return _tau_panel(df, "fpr_hard", "FPR (hard negatives)",
                      "blocklist-fpr-hard-vs-tau.png", cfg, k_fraction, bounded_unit=True)


def panel_fpr_easy_vs_tau(df, cfg, k_fraction):
    return _tau_panel(df, "fpr_easy", "FPR (easy negatives)",
                      "blocklist-fpr-easy-vs-tau.png", cfg, k_fraction, bounded_unit=True)


def panel_geometric_objective_vs_tau(df, cfg, k_fraction):
    """Threshold-expanded geometric objective log10 J_tau(C) = log10 sum_i
    (r_i + tau)^d -- the overlap-blind sum of the cover's ball volumes once
    every ball is expanded by tau."""
    return _tau_panel(df, "log10_j_tau", r"$\log_{10} J_\tau(\mathcal{C})$",
                      "blocklist-geometric-objective-vs-tau.png", cfg, k_fraction)


def panel_saved_vs_tau(df, cfg, k_fraction):
    """Percentage of queries that never reach the backing index, as tau grows.
    Falls monotonically: a larger tau flags more of the stream."""
    return _tau_panel(df, "accesses_saved_pct", "Accesses Saved (%)",
                      "blocklist-accesses-saved-vs-tau.png", cfg, k_fraction,
                      horizontal_grid=False)


def print_summary(df: pd.DataFrame, k_fraction: float) -> None:
    sel = df[(df["sweep"] == SWEEP_TAU) & (np.isclose(df["k_fraction"], k_fraction))]
    print(f"\n--- Blocklist workload summary (tau sweep at k={k_fraction:g}) ---")
    cols = ("fpr", "fpr_hard", "fpr_easy", "recall", "accesses_saved_pct", "log10_j_tau")
    aggs = {c: mean_std(sel, ["dataset", "tau"], c) for c in cols}
    for dataset in BLOCKLIST_DATASETS:
        sub = sel[sel["dataset"] == dataset]
        if sub.empty:
            continue
        print(f"\n{BLOCKLIST_LABELS[dataset]} (delta={sub['d_pos_med'].mean():.3f}, "
              f"d_hard={sub['d_hard_med'].mean():.3f})")
        for tau in sorted(sub["tau"].unique()):
            def _m(c):
                row = aggs[c][(aggs[c]["dataset"] == dataset) & (aggs[c]["tau"] == tau)]
                return float(row["mean"].iloc[0]) if len(row) else float("nan")
            print(f"  tau={tau:.2f}  FPR={_m('fpr'):.3f} FPR_hard={_m('fpr_hard'):.3f} "
                  f"FPR_easy={_m('fpr_easy'):.3f} recall={_m('recall'):.3f} "
                  f"saved={_m('accesses_saved_pct'):5.1f}% log10_J={_m('log10_j_tau'):.2f}")
    print()


def generate_blocklist_figures(df: Optional[pd.DataFrame] = None, outdir: str = OUTDIR,
                               dpi: int = DPI, emit_pdf: bool = EMIT_PDF,
                               k_fraction: Optional[float] = None) -> None:
    if df is None:
        df = load_data()
    else:
        validate_schema(df)
    cfg = Config(dpi=dpi, emit_pdf=emit_pdf, outdir=outdir)

    kf = resolve_tau_kfraction(df, k_fraction)
    panel_fpr_vs_tau(df, cfg, kf)
    panel_fpr_hard_vs_tau(df, cfg, kf)
    panel_fpr_easy_vs_tau(df, cfg, kf)
    panel_geometric_objective_vs_tau(df, cfg, kf)
    panel_saved_vs_tau(df, cfg, kf)

    save_blocklist_legend(cfg.outdir, cfg.dpi, cfg.emit_pdf)

    print_summary(df, kf)


def main() -> None:
    generate_blocklist_figures()


if __name__ == "__main__":
    main()
