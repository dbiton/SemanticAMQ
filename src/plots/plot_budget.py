"""Ball-budget figures (see main.generate_budget).

The same asymmetric blocklist workload as plot_blocklist, read along the other
axis. There, tau is solved per row for a target recall and the ball budget k
moves; here tau is pinned at one value for every row and k moves, so recall is
an outcome rather than a held constant. That makes this the "what does a fixed
operating threshold buy as you pay for more balls" view: recall and FPR both
move, and the question is where the budget stops buying either.

The x-axis is k as a percentage of the reference set |B| -- the budget itself,
not plot_blocklist's derived compression |B|/k, since the sweep is a narrow
linear band (5%-20%) where a reciprocal log axis would compress the whole
range into one corner.

Panels:
  (a) Recall -- held fixed by construction in plot_blocklist, measured here.
  (b) FPR on HARD negatives: members of duplicate clusters that were NOT
      blocklisted, i.e. B's own local geometry. The headline number.
  (c) FPR on EASY negatives: singletons, i.e. background traffic.
  (d) Index accesses saved over the unfiltered baseline.
  (e) Geometric objective log10 J_tau(C) = log10 sum_i (r_i + tau)^d -- the
      overlap-blind sum of ball volumes once every ball is expanded by tau.

All five share one legend, written separately as budget-legend.png so the
panels can be placed side by side without repeating it.

Absolute tau is not comparable between corpora (the distance at which a
duplicate sits is a property of each embedding set), so these panels rank each
corpus's own response to the budget, not the corpora against each other.
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd

from plots.plot_blocklist import (
    BLOCKLIST_DATASETS,
    BLOCKLIST_LABELS,
    SWEEP_TAU,
    Config,
    line_panel,
    save_blocklist_legend,
    validate_schema,
)
from plots.plot_common import mean_std

INPUT_PATH = "results/budget.csv"
OUTDIR = "plots/budget/"
DPI = 300
EMIT_PDF = False

# k as a percentage of |B| rather than plot_blocklist's compression |B|/k.
_XLABEL = r"Ball budget $k$ (% of $|B|$)"


def load_data() -> pd.DataFrame:
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH!r}. Run generate_budget() (main.py) first."
        )
    df = pd.read_csv(INPUT_PATH)
    validate_schema(df)
    return df


def fixed_tau(df: pd.DataFrame) -> float:
    """The single threshold every panel is read at.

    generate_budget submits one tau, so the fixed-tau rows carry exactly one
    value; more than one means the file mixes runs at different thresholds,
    which would silently average two response curves into one line."""
    taus = sorted(df[df["sweep"] == SWEEP_TAU]["tau"].unique())
    if not taus:
        raise ValueError("No fixed-tau rows present; rerun generate_budget.")
    if len(taus) > 1:
        raise ValueError(
            f"Expected a single tau, found {taus}. This evaluation holds tau constant "
            "(see main.generate_budget); use plot_blocklist for the tau-swept view."
        )
    return float(taus[0])


def _budget_panel(df: pd.DataFrame, value_col: str, ylabel: str, filename: str,
                  cfg: Config, tau: float, bounded_unit: bool = False) -> pd.DataFrame:
    """plot_blocklist's shared renderer, pointed at the fixed-tau rows and at a
    budget x-axis: linear, in percent of |B|."""
    return line_panel(
        df, value_col, ylabel, filename, cfg,
        bounded_unit=bounded_unit, logx=False,
        sweep=SWEEP_TAU, x_from=lambda kf: 100.0 * kf, xlabel=_XLABEL,
        annotation=rf"at $\tau$ = {tau:g}",
    )


def panel_recall(df, cfg, tau):
    return _budget_panel(df, "recall", "Recall", "budget-recall.png", cfg, tau,
                         bounded_unit=True)


def panel_fpr_hard(df, cfg, tau):
    return _budget_panel(df, "fpr_hard", "FPR (hard negatives)", "budget-fpr-hard.png",
                         cfg, tau, bounded_unit=True)


def panel_fpr_easy(df, cfg, tau):
    return _budget_panel(df, "fpr_easy", "FPR (easy negatives)", "budget-fpr-easy.png",
                         cfg, tau, bounded_unit=True)


def panel_saved(df, cfg, tau):
    return _budget_panel(df, "accesses_saved_pct", "Index Accesses Saved (%)",
                         "budget-accesses-saved.png", cfg, tau)


def panel_geometric_objective(df, cfg, tau):
    return _budget_panel(df, "log10_j_tau", r"$\log_{10} J_\tau(\mathcal{C})$",
                         "budget-geometric-objective.png", cfg, tau)


def print_summary(df: pd.DataFrame, tau: float) -> None:
    sel = df[df["sweep"] == SWEEP_TAU]
    print(f"\n--- Ball-budget sweep (tau fixed at {tau:g}) ---")
    cols = ("recall", "fpr_hard", "fpr_easy", "accesses_saved_pct", "r_med", "log10_j_tau")
    aggs = {c: mean_std(sel, ["dataset", "k_fraction"], c) for c in cols}
    for dataset in BLOCKLIST_DATASETS:
        sub = sel[sel["dataset"] == dataset]
        if sub.empty:
            continue
        n_ref = int(sub["n_reference"].iloc[0])
        print(f"\n{BLOCKLIST_LABELS[dataset]} (|B|={n_ref}, delta={sub['d_pos_med'].mean():.3f}, "
              f"d_hard={sub['d_hard_med'].mean():.3f})")
        for kf in sorted(sub["k_fraction"].unique()):
            def _m(c):
                row = aggs[c][(aggs[c]["dataset"] == dataset) & (aggs[c]["k_fraction"] == kf)]
                return float(row["mean"].iloc[0]) if len(row) else float("nan")
            print(f"  k={100.0*kf:>5.1f}%  r_med={_m('r_med'):.3f} recall={_m('recall'):.3f} "
                  f"FPR_hard={_m('fpr_hard'):.3f} FPR_easy={_m('fpr_easy'):.3f} "
                  f"saved={_m('accesses_saved_pct'):5.1f}% log10_J={_m('log10_j_tau'):.2f}")
    print()


def generate_budget_figures(df: Optional[pd.DataFrame] = None, outdir: str = OUTDIR,
                            dpi: int = DPI, emit_pdf: bool = EMIT_PDF) -> None:
    if df is None:
        df = load_data()
    else:
        validate_schema(df)
    cfg = Config(dpi=dpi, emit_pdf=emit_pdf, outdir=outdir)
    tau = fixed_tau(df)

    panel_recall(df, cfg, tau)
    panel_fpr_hard(df, cfg, tau)
    panel_fpr_easy(df, cfg, tau)
    panel_saved(df, cfg, tau)
    panel_geometric_objective(df, cfg, tau)

    # Same three corpora and the same palette as the blocklist figures, so the
    # two evaluations' panels read as one set.
    save_blocklist_legend(cfg.outdir, cfg.dpi, cfg.emit_pdf, filename="budget-legend.png")

    print_summary(df, tau)


def main() -> None:
    generate_budget_figures()


if __name__ == "__main__":
    main()
