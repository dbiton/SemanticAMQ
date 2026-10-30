"""Ball-budget figures (see main.generate_budget).

The same asymmetric blocklist workload as plot_blocklist, read along the other
axis: there tau moves at a fixed ball budget; here tau is pinned at one value
for every row and k moves. Queries are labeled by distance at that tau, so the
labels are fixed along the whole sweep and recall is 1 by construction (the
worker raises on a false negative). That makes this the "what does a fixed
operating threshold buy as you pay for more balls" view: FPR and accesses
saved move, and the question is where the budget stops buying either.

The x-axis is k as a percentage of the reference set |X| -- the budget itself,
not plot_blocklist's derived compression |X|/k, since the sweep is a narrow
linear band (5%-20%) where a reciprocal log axis would compress the whole
range into one corner.

Panels:
  (a) FPR on closely related negatives: queries with cosine distance <= 0.2
      to their nearest indexed vector but farther than tau -- where most
      false positives live.
  (b) FPR on related negatives (cosine distance in (0.2, 0.4]).
  (c) FPR on unrelated negatives (cosine distance > 0.4): background traffic.
  (d) Index accesses saved over the unfiltered baseline, with each corpus's
      ceiling (the share of negative queries) as a thin dashed line.
  (e) Geometric objective log10 J_tau(C) = log10 sum_i (r_i + tau)^d -- the
      overlap-blind sum of ball volumes once every ball is expanded by tau.

All of them share one legend, written separately as budget-legend.png so the
panels can be placed side by side without repeating it.

Absolute tau is not comparable between corpora (the distance at which a
duplicate sits is a property of each embedding set), so these panels rank each
corpus's own response to the budget, not the corpora against each other.
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd

import matplotlib.pyplot as plt

from plots.plot_blocklist import (
    BLOCKLIST_COLORS,
    BLOCKLIST_DATASETS,
    BLOCKLIST_LABELS,
    BLOCKLIST_MARKERS,
    SWEEP_TAU,
    Config,
    line_panel,
    save_blocklist_legend,
    validate_schema,
)
from plots.plot_common import apply_style, mean_std, save_figure, style_axes, wrap_label

INPUT_PATH = "results/budget.csv"
OUTDIR = "plots/budget/"
DPI = 300
EMIT_PDF = False

# k as a percentage of |X| rather than plot_blocklist's compression |X|/k.
_XLABEL = r"Ball budget $k$ (% of $|X|$)"

# Embedding dimensionality: fixed at 384 across every blocklist-workload
# evaluation (see generate_budget/generate_blocklist's own dim=384), needed
# here only to convert bytes into "as a fraction of storing every reference
# vector raw" -- the naive float32 baseline the ball cover is compressing
# against.
_EMBEDDING_DIM = 384
_XLABEL_RAW_PCT = "Memory Usage (% of Raw)"

# On top of the columns plot_blocklist requires: generate_budget is the one
# blocklist-workload sweep that can afford these (see its docstring), so unlike
# results/blocklist.csv they are always present here.
_VOLUME_COLUMNS = ["log10_vol_low", "log10_vol_high"]

# memory_bytes/n_reference come from the same shared core (runner's
# _blocklist_cover_rows) that supplies every other required column, so in
# practice they're always present -- validated separately anyway, since the
# raw-storage panels are the only thing here that reads them.
_RAW_STORAGE_COLUMNS = ["memory_bytes", "n_reference"]


def _validate_volume_schema(df: pd.DataFrame) -> None:
    missing = [c for c in _VOLUME_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input missing columns: {missing}. These come from runner.budget_worker, "
            "which always requests volume bounds -- a file without them predates that, "
            "so rerun generate_budget."
        )


def _validate_raw_storage_schema(df: pd.DataFrame) -> None:
    missing = [c for c in _RAW_STORAGE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input missing columns: {missing}, needed for the raw-storage-percentage "
            "panels. Rerun generate_budget with a current runner.py."
        )


def load_data() -> pd.DataFrame:
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH!r}. Run generate_budget() (main.py) first."
        )
    df = pd.read_csv(INPUT_PATH)
    validate_schema(df)
    _validate_volume_schema(df)
    _validate_raw_storage_schema(df)
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
                  cfg: Config, bounded_unit: bool = False,
                  ceiling_col: Optional[str] = None) -> pd.DataFrame:
    """plot_blocklist's shared renderer, pointed at the fixed-tau rows and at a
    budget x-axis: linear, in percent of |X|."""
    return line_panel(
        df, value_col, ylabel, filename, cfg,
        bounded_unit=bounded_unit, logx=False,
        sweep=SWEEP_TAU, x_from=lambda kf: 100.0 * kf, xlabel=_XLABEL,
        ceiling_col=ceiling_col,
    )


def _raw_storage_pct_x(df: pd.DataFrame, dim: int = _EMBEDDING_DIM) -> pd.DataFrame:
    """(dataset, k_fraction, x) frame for line_panel's `x_override`: x is the
    realised memory usage as a percentage of storing every reference vector
    raw (n_reference * dim float32 values), at each (dataset, k_fraction).

    memory_bytes and n_reference are both averaged per group first (the same
    "aggregate, then combine" pattern plot_quantization.aggregate() uses for
    its memory axis) rather than computing the ratio per row and averaging
    that: n_reference is already constant within a (dataset, k_fraction)
    group, so this only matters for memory_bytes, but doing it this way once
    means a future column that does vary per fold is handled correctly too."""
    sel = df[df["sweep"] == SWEEP_TAU]
    grouped = sel.groupby(["dataset", "k_fraction"])
    mem = grouped["memory_bytes"].mean()
    n_ref = grouped["n_reference"].mean()
    x = 100.0 * mem / (n_ref * dim * 4)
    return x.reset_index(name="x")


def _budget_panel_raw_pct(df: pd.DataFrame, value_col: str, ylabel: str, filename: str,
                          cfg: Config, bounded_unit: bool = False,
                          ceiling_col: Optional[str] = None) -> pd.DataFrame:
    """Same panel as _budget_panel, but against realised memory usage as a
    percentage of raw (uncompressed float32) vector storage instead of ball
    budget k -- the two axes agree in shape only up to how compact this
    filter's per-ball encoding is relative to a raw vector."""
    return line_panel(
        df, value_col, ylabel, filename, cfg,
        bounded_unit=bounded_unit, logx=False,
        sweep=SWEEP_TAU, xlabel=_XLABEL_RAW_PCT,
        x_override=_raw_storage_pct_x(df), ceiling_col=ceiling_col,
    )


# One FPR panel per cosine-distance band: (column, label, filename stem).
_FPR_PANELS = (
    ("fpr_close", "FPR (closely related negatives)", "budget-fpr-close"),
    ("fpr_related", "FPR (related negatives)", "budget-fpr-related"),
    ("fpr_unrelated", "FPR (unrelated negatives)", "budget-fpr-unrelated"),
)


def panels_fpr(df, cfg):
    for col, label, stem in _FPR_PANELS:
        _budget_panel(df, col, label, f"{stem}.png", cfg, bounded_unit=True)


def panels_fpr_raw_pct(df, cfg):
    for col, label, stem in _FPR_PANELS:
        _budget_panel_raw_pct(df, col, label, f"{stem}-raw-pct.png", cfg, bounded_unit=True)


def panel_saved(df, cfg):
    return _budget_panel(df, "accesses_saved_pct", "Accesses Saved (%)",
                         "budget-accesses-saved.png", cfg, ceiling_col="neg_share_pct")


def panel_saved_raw_pct(df, cfg):
    return _budget_panel_raw_pct(df, "accesses_saved_pct", "Accesses Saved (%)",
                                 "budget-accesses-saved-raw-pct.png", cfg,
                                 ceiling_col="neg_share_pct")


def panel_geometric_objective(df, cfg):
    return _budget_panel(df, "log10_j_tau", r"$\log_{10} J_\tau(\mathcal{C})$",
                         "budget-geometric-objective.png", cfg)


def panel_geometric_objective_raw_pct(df, cfg):
    return _budget_panel_raw_pct(df, "log10_j_tau", r"$\log_{10} J_\tau(\mathcal{C})$",
                                 "budget-geometric-objective-raw-pct.png", cfg)


def _volume_bounds_panel(df: pd.DataFrame, cfg: Config, filename: str, xlabel: str,
                         x_override: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Lower/upper bounds (one shaded band per corpus) on the volume actually
    covered by the tau-expanded cover, against the budget that bought it --
    see runner._tau_volume_bounds_log10's overlap-aware union bound. Tighter
    than panel_geometric_objective's log10_j_tau, which sums each ball's
    volume independently and so double-counts every overlap; the gap between
    the two panels is what that double-counting costs.

    Rendered as a band rather than through line_panel because there are two
    series per corpus, and the interesting quantity is the region between
    them, not either bound alone. `x_override` mirrors line_panel's own
    parameter of the same name (see there) -- a (dataset, k_fraction, x)
    frame merged in for the raw-storage-percentage variant instead of
    computing x from k_fraction directly."""
    sel = df[df["sweep"] == SWEEP_TAU]
    agg_low = mean_std(sel, ["dataset", "k_fraction"], "log10_vol_low")[
        ["dataset", "k_fraction", "mean"]].rename(columns={"mean": "low"})
    agg_high = mean_std(sel, ["dataset", "k_fraction"], "log10_vol_high")[
        ["dataset", "k_fraction", "mean"]].rename(columns={"mean": "high"})
    agg = agg_low.merge(agg_high, on=["dataset", "k_fraction"])
    if x_override is not None:
        agg = agg.merge(x_override, on=["dataset", "k_fraction"], how="left")
    else:
        agg["x"] = 100.0 * agg["k_fraction"]

    apply_style(cfg.dpi)
    fig, ax = plt.subplots()
    for dataset in BLOCKLIST_DATASETS:
        sub = agg[agg["dataset"] == dataset].sort_values("x")
        if sub.empty:
            continue
        ax.plot(sub["x"], sub["low"], color=BLOCKLIST_COLORS[dataset],
                marker=BLOCKLIST_MARKERS[dataset], markersize=2.0, linewidth=0.6,
                linestyle="--", zorder=2)
        ax.plot(sub["x"], sub["high"], color=BLOCKLIST_COLORS[dataset],
                marker=BLOCKLIST_MARKERS[dataset], markersize=2.0, linewidth=0.6,
                linestyle="-", zorder=2)
        ax.fill_between(sub["x"], sub["low"], sub["high"], color=BLOCKLIST_COLORS[dataset],
                        alpha=0.15, linewidth=0, zorder=1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(wrap_label(r"$\log_{10}$ Volume Bounds"))
    style_axes(ax)
    ax.grid(visible=False, axis="y")  # no horizontal gridlines
    fig.subplots_adjust(left=0.26, right=0.97, top=0.95, bottom=0.24)
    save_figure(fig, filename, cfg.outdir, cfg.dpi, cfg.emit_pdf, tight=False)
    return agg


def panel_volume_bounds(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    return _volume_bounds_panel(df, cfg, "budget-volume-bounds.png", _XLABEL)


def panel_volume_bounds_raw_pct(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    return _volume_bounds_panel(df, cfg, "budget-volume-bounds-raw-pct.png",
                                _XLABEL_RAW_PCT, x_override=_raw_storage_pct_x(df))


def print_summary(df: pd.DataFrame, tau: float) -> None:
    sel = df[df["sweep"] == SWEEP_TAU]
    print(f"\n--- Ball-budget sweep (tau fixed at {tau:g}) ---")
    cols = ("fpr", "fpr_close", "fpr_related", "fpr_unrelated", "accesses_saved_pct",
            "neg_share_pct", "psi_med", "log10_j_tau", "log10_vol_low", "log10_vol_high")
    aggs = {c: mean_std(sel, ["dataset", "k_fraction"], c) for c in cols}
    for dataset in BLOCKLIST_DATASETS:
        sub = sel[sel["dataset"] == dataset]
        if sub.empty:
            continue
        n_ref = int(sub["n_reference"].iloc[0])
        print(f"\n{BLOCKLIST_LABELS[dataset]} (|X|={n_ref}, "
              f"median nn cosine distance={1 - sub['nn_cos_med'].mean():.3f}, "
              f"negatives={sub['neg_share_pct'].mean():.1f}%)")
        for kf in sorted(sub["k_fraction"].unique()):
            def _m(c):
                row = aggs[c][(aggs[c]["dataset"] == dataset) & (aggs[c]["k_fraction"] == kf)]
                return float(row["mean"].iloc[0]) if len(row) else float("nan")
            print(f"  k={100.0*kf:>5.1f}%  psi_med={_m('psi_med'):.3f} FPR={_m('fpr'):.3f} "
                  f"FPR_close={_m('fpr_close'):.3f} FPR_related={_m('fpr_related'):.3f} "
                  f"FPR_unrelated={_m('fpr_unrelated'):.3f} "
                  f"saved={_m('accesses_saved_pct'):5.1f}% log10_J={_m('log10_j_tau'):.2f} "
                  f"vol=[{_m('log10_vol_low'):.2f}, {_m('log10_vol_high'):.2f}]")
    print()


def generate_budget_figures(df: Optional[pd.DataFrame] = None, outdir: str = OUTDIR,
                            dpi: int = DPI, emit_pdf: bool = EMIT_PDF) -> None:
    if df is None:
        df = load_data()
    else:
        validate_schema(df)
        _validate_volume_schema(df)
        _validate_raw_storage_schema(df)
    cfg = Config(dpi=dpi, emit_pdf=emit_pdf, outdir=outdir)
    tau = fixed_tau(df)

    panels_fpr(df, cfg)
    panel_saved(df, cfg)
    panel_geometric_objective(df, cfg)
    panel_volume_bounds(df, cfg)

    # Same panels again, against realised memory usage as a percentage
    # of raw (uncompressed float32) vector storage instead of ball budget k
    # -- see _raw_storage_pct_x.
    panels_fpr_raw_pct(df, cfg)
    panel_saved_raw_pct(df, cfg)
    panel_geometric_objective_raw_pct(df, cfg)
    panel_volume_bounds_raw_pct(df, cfg)

    # Same three corpora and the same palette as the blocklist figures, so the
    # two evaluations' panels read as one set.
    save_blocklist_legend(cfg.outdir, cfg.dpi, cfg.emit_pdf, filename="budget-legend.png")

    print_summary(df, tau)


def main() -> None:
    generate_budget_figures()


if __name__ == "__main__":
    main()
