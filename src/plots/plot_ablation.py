"""Generate the five ablation-figure panels (+ shared legend) for the PVLDB paper.

Self-contained: reads a long-format results file, aggregates over seeds, and
writes PNG (optionally PDF) panels sized for `\\includegraphics[width=\\linewidth]`
inside `subfigure{0.24\\linewidth}` cells of a two-column `figure*`.

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

from filters.ball_cover_filter import Quantize
from plots.plot_common import DATASETS, DATASET_LABELS, DATASET_COLORS, DATASET_HATCHES, apply_style as _apply_style, save_figure, save_dataset_legend

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #

INPUT_PATH = "results/ablation.csv"
OUTDIR = "plots/ablation/"
K = 256
TAU: Optional[float] = None  # None => median tau present in the data
DPI = 400
EMIT_PDF = False

# runner.ABLATION_QUANTIZE_SWEEP, mirrored here (rather than imported) so
# this module has no dependency on runner.py's multiprocessing/harness
# machinery -- just the Quantize enum both sides already agree on. NONE is
# the raw/unquantized cover (tagged stage="ball" by ablation_worker); every
# other entry is a real quantizer sharing that same cover's memory budget
# (tagged stage="quantizing"), which is what makes the n-balls panel
# meaningful: a more compact code packs more balls into the same budget.
QUANTIZE_SWEEP = (
    Quantize.NONE, Quantize.SQ8, Quantize.SQ6, Quantize.SQ4,
    Quantize.FP16, Quantize.BF16, Quantize.SQ4T, Quantize.SQ8T, Quantize.PQ,
)
STAGES = [q.value for q in QUANTIZE_SWEEP]
STAGE_LABELS = {
    Quantize.NONE.value: "None",
    Quantize.SQ8.value: "SQ8",
    Quantize.SQ6.value: "SQ6",
    Quantize.SQ4.value: "SQ4",
    Quantize.FP16.value: "FP16",
    Quantize.BF16.value: "BF16",
    Quantize.SQ4T.value: "SQ4-T",
    Quantize.SQ8T.value: "SQ8-T",
    Quantize.PQ.value: "PQ",
}
# Reference stage for the two relative-objective panels: the raw/unquantized
# cover, matching the "reduction relative to the ball cover" reading the
# panels had before per-quantize ablation existed.
BASELINE_STAGE = Quantize.NONE.value

BAR_WIDTH = 0.26
D_DIM = 384  # embedding dimensionality; objectives are stored pre-logged


@dataclasses.dataclass
class Config:
    k: int
    tau: float
    dpi: int
    emit_pdf: bool
    outdir: str


# --------------------------------------------------------------------------- #
# Style
# --------------------------------------------------------------------------- #

def apply_style() -> None:
    _apply_style(DPI)


def _style_axes(ax: plt.Axes) -> None:
    ax.grid(axis="y", linewidth=0.35, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.5)


def _set_quantize_xaxis(ax: plt.Axes, stages: list) -> None:
    x = np.arange(len(stages))
    ax.set_xticks(x)
    ax.set_xticklabels(
        [STAGE_LABELS[s] for s in stages],
        rotation=30, ha="right", rotation_mode="anchor",
    )
    ax.set_xlim(-0.5, len(stages) - 0.5)


def _dataset_offsets() -> dict:
    # three bars per stage, centered on the stage tick
    offs = np.array([-1, 0, 1]) * BAR_WIDTH
    return dict(zip(DATASETS, offs))


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

_REQUIRED_COLUMNS = [
    "dataset",
    "k",
    "quantize",
    "stage",
    "tau",
    "log10_J",
    "log10_J_tau",
    "fpr",
    "fnr",
    "stage_time_s",
    "cum_time_s",
    "n_balls",
    "seed",
]


def _validate_schema(df: pd.DataFrame) -> None:
    missing_cols = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Input is missing required columns: {missing_cols}. "
            f"Present columns: {sorted(df.columns)}"
        )

    bad = df[df["fnr"] != 0]
    if len(bad):
        raise ValueError(
            "Zero-false-negative invariant violated: found "
            f"{len(bad)} row(s) with fnr != 0, e.g.\n{bad.head()}"
        )

    missing_ds = set(DATASETS) - set(df["dataset"].unique())
    if missing_ds:
        raise ValueError(
            f"Missing dataset(s) {sorted(missing_ds)} in data. "
            f"Available: {sorted(df['dataset'].unique())}"
        )

    present = set(df["quantize"].unique())
    missing_combos = [s for s in STAGES if s not in present]
    if missing_combos:
        raise ValueError(
            f"Missing quantize value(s) {missing_combos} in data. "
            f"Available: {sorted(present)}"
        )


def load_data() -> pd.DataFrame:
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH!r}. Run generate_ablation() (main.py) "
            "first, or point INPUT_PATH at a real results file."
        )
    if INPUT_PATH.endswith(".parquet"):
        df = pd.read_parquet(INPUT_PATH)
    else:
        # keep_default_na=False: Quantize.NONE.value == "None", which pandas'
        # default na_values list otherwise silently reads back as NaN,
        # corrupting the quantize column into a mixed float/str dtype.
        df = pd.read_csv(INPUT_PATH, keep_default_na=False)

    _validate_schema(df)
    return df


def resolve_config(
    df: pd.DataFrame,
    k: int = K,
    tau: Optional[float] = TAU,
    dpi: int = DPI,
    emit_pdf: bool = EMIT_PDF,
    outdir: str = OUTDIR,
) -> Config:
    available_k = sorted(df["k"].unique())
    if k not in available_k:
        raise ValueError(f"k={k} not present in data. Available k values: {available_k}")

    available_tau = sorted(df["tau"].unique())
    if tau is None:
        tau = float(df["tau"].median())
        if tau not in available_tau:
            tau = min(available_tau, key=lambda t: abs(t - tau))
    elif tau not in available_tau:
        raise ValueError(f"tau={tau} not present in data. Available tau values: {available_tau}")

    return Config(k=k, tau=tau, dpi=dpi, emit_pdf=emit_pdf, outdir=outdir)


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def aggregate(df: pd.DataFrame, value_col: str, cfg: Config, use_tau: bool = False,
             stages: list = STAGES) -> pd.DataFrame:
    """Mean + 95% CI of `value_col` over seed, per (dataset, quantize),
    at cfg.k (and cfg.tau if use_tau), restricted to the given `stages`
    (quantize values)."""
    sel = df[df["k"] == cfg.k]
    if use_tau:
        sel = sel[sel["tau"] == cfg.tau]
    sel = sel[sel["quantize"].isin(stages)]

    bad_raw = sel[~np.isfinite(sel[value_col])]
    if len(bad_raw):
        cols = [c for c in ("dataset", "k", "quantize", "tau", "stage", "seed", value_col) if c in bad_raw.columns]
        raise ValueError(
            f"Non-finite {value_col!r} values at k={cfg.k}"
            + (f", tau={cfg.tau}" if use_tau else "") + f":\n{bad_raw[cols].to_string(index=False)}\n"
            "This usually means k >= the construct-set size for that dataset (every ball "
            "becomes a zero-radius singleton, so log10_J is -inf) -- lower k or shrink the "
            "construct-set-relative k sweep in the data-generation step."
        )

    grouped = sel.groupby(["dataset", "quantize"])[value_col]
    agg = grouped.agg(mean="mean", sem="sem", n="count").reset_index()
    agg["ci95"] = np.where(agg["n"] > 1, 1.96 * agg["sem"].fillna(0.0), 0.0)

    for dataset in DATASETS:
        for stage in stages:
            if not ((agg["dataset"] == dataset) & (agg["quantize"] == stage)).any():
                raise ValueError(
                    f"No rows for dataset={dataset!r}, quantize={stage!r} at "
                    f"k={cfg.k}" + (f", tau={cfg.tau}" if use_tau else "") + "."
                )
    return agg


def _lookup(agg: pd.DataFrame, dataset: str, quantize: str) -> tuple:
    row = agg[(agg["dataset"] == dataset) & (agg["quantize"] == quantize)].iloc[0]
    return float(row["mean"]), float(row["ci95"])


# --------------------------------------------------------------------------- #
# Panel (a): geometric objective reduction
# --------------------------------------------------------------------------- #

def panel_geometric_objective(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    agg = aggregate(df, "log10_J", cfg, use_tau=False)

    rel = agg.copy()
    rel["rel_mean"] = np.nan
    rel["rel_ci95"] = np.nan
    for dataset in DATASETS:
        base_mean, _ = _lookup(agg, dataset, BASELINE_STAGE)
        mask = rel["dataset"] == dataset
        rel.loc[mask, "rel_mean"] = rel.loc[mask, "mean"] - base_mean
        rel.loc[mask, "rel_ci95"] = rel.loc[mask, "ci95"]

    _bar_panel(
        rel,
        value_col="rel_mean",
        ci_col="rel_ci95",
        ylabel=r"$\log_{10} J(\mathcal{C})/J_{\mathrm{ball}}$",
        filename="ablation-geometric-objective.png",
        cfg=cfg,
        zero_line=True,
    )
    return rel


# --------------------------------------------------------------------------- #
# Panel (b): tau-expanded objective reduction
# --------------------------------------------------------------------------- #

def panel_threshold_objective(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    agg = aggregate(df, "log10_J_tau", cfg, use_tau=True)

    rel = agg.copy()
    rel["rel_mean"] = np.nan
    rel["rel_ci95"] = np.nan
    for dataset in DATASETS:
        base_mean, _ = _lookup(agg, dataset, BASELINE_STAGE)
        mask = rel["dataset"] == dataset
        rel.loc[mask, "rel_mean"] = rel.loc[mask, "mean"] - base_mean
        rel.loc[mask, "rel_ci95"] = rel.loc[mask, "ci95"]

    def annotate(ax: plt.Axes) -> None:
        ax.text(
            0.97,
            0.95,
            rf"$\tau = {cfg.tau:g}$",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=5.5,
        )

    _bar_panel(
        rel,
        value_col="rel_mean",
        ci_col="rel_ci95",
        ylabel=r"$\log_{10} J_\tau(\mathcal{C})/J_{\tau,\mathrm{ball}}$",
        filename="ablation-threshold-expanded-objective.png",
        cfg=cfg,
        zero_line=True,
        extra=annotate,
    )
    return rel


def _bar_panel(
    agg: pd.DataFrame,
    value_col: str,
    ci_col: str,
    ylabel: str,
    filename: str,
    cfg: Config,
    zero_line: bool = False,
    extra=None,
    stages: list = STAGES,
) -> None:
    apply_style()
    fig, ax = plt.subplots()
    x = np.arange(len(stages))
    offsets = _dataset_offsets()

    for dataset in DATASETS:
        means, cis = [], []
        for stage in stages:
            row = agg[(agg["dataset"] == dataset) & (agg["quantize"] == stage)].iloc[0]
            means.append(row[value_col])
            cis.append(row[ci_col])
        means = np.array(means)
        cis = np.array(cis)
        has_ci = np.any(cis > 0)

        ax.bar(
            x + offsets[dataset],
            means,
            width=BAR_WIDTH,
            color=DATASET_COLORS[dataset],
            hatch=DATASET_HATCHES[dataset],
            edgecolor="black",
            linewidth=0.4,
            zorder=2,
        )
        if has_ci:
            ax.errorbar(
                x + offsets[dataset],
                means,
                yerr=cis,
                fmt="none",
                ecolor="black",
                elinewidth=0.5,
                capsize=1.5,
                capthick=0.5,
                zorder=3,
            )

    if zero_line:
        ax.axhline(0.0, color="black", linewidth=0.5, zorder=1)
        ymin = agg[value_col].min()
        ax.set_ylim(1.1 * min(ymin, 0.0), 0.0)

    ax.set_ylabel(ylabel)
    _set_quantize_xaxis(ax, stages)
    _style_axes(ax)
    if extra is not None:
        extra(ax)

    _save(fig, filename, cfg)


# --------------------------------------------------------------------------- #
# Panel (c): false positive rate
# --------------------------------------------------------------------------- #

def panel_fpr(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    agg = aggregate(df, "fpr", cfg, use_tau=True)

    positive = agg.loc[agg["mean"] > 0, "mean"]
    if len(positive) == 0:
        raise ValueError("All FPR values are zero at the selected k/tau; cannot set a log floor.")
    floor = 10 ** (np.floor(np.log10(positive.min())) - 1)

    apply_style()
    fig, ax = plt.subplots()
    x = np.arange(len(STAGES))
    offsets = _dataset_offsets()

    for dataset in DATASETS:
        means, cis, is_zero = [], [], []
        for stage in STAGES:
            row = agg[(agg["dataset"] == dataset) & (agg["quantize"] == stage)].iloc[0]
            m = row["mean"]
            is_zero.append(m == 0)
            means.append(floor if m == 0 else m)
            cis.append(row["ci95"])
        means = np.array(means)
        cis = np.array(cis)
        is_zero = np.array(is_zero)
        has_ci = np.any(cis > 0)

        for i, xi in enumerate(x + offsets[dataset]):
            hatch = "xxxx" if is_zero[i] else DATASET_HATCHES[dataset]
            facecolor = "none" if is_zero[i] else DATASET_COLORS[dataset]
            ax.bar(
                xi,
                means[i],
                width=BAR_WIDTH,
                color=facecolor,
                hatch=hatch,
                edgecolor=DATASET_COLORS[dataset] if is_zero[i] else "black",
                linewidth=0.4,
                zorder=2,
            )
            if is_zero[i]:
                ax.text(xi, means[i] * 1.3, "0", ha="center", va="bottom", fontsize=5.5)

        if has_ci:
            ax.errorbar(
                (x + offsets[dataset])[~is_zero],
                means[~is_zero],
                yerr=cis[~is_zero],
                fmt="none",
                ecolor="black",
                elinewidth=0.5,
                capsize=1.5,
                capthick=0.5,
                zorder=3,
            )

    ax.set_yscale("log")
    ax.set_ylim(floor / 3.0, None)
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="y", which="minor", length=1.2, width=0.4)

    ax.set_ylabel("FPR")
    _set_quantize_xaxis(ax, STAGES)
    _style_axes(ax)

    _save(fig, "ablation-false-positive-rate.png", cfg)
    return agg


# --------------------------------------------------------------------------- #
# Panel (d): construction time
# --------------------------------------------------------------------------- #

def panel_construction_time(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    agg_stage = aggregate(df, "stage_time_s", cfg, use_tau=False)
    agg_cum = aggregate(df, "cum_time_s", cfg, use_tau=False)

    apply_style()
    fig, ax = plt.subplots()
    x = np.arange(len(STAGES))
    offsets = _dataset_offsets()

    for dataset in DATASETS:
        means, cis = [], []
        for stage in STAGES:
            row = agg_stage[(agg_stage["dataset"] == dataset) & (agg_stage["quantize"] == stage)].iloc[0]
            means.append(row["mean"])
            cis.append(row["ci95"])
        means = np.array(means)
        cis = np.array(cis)
        has_ci = np.any(cis > 0)

        ax.bar(
            x + offsets[dataset],
            means,
            width=BAR_WIDTH,
            color=DATASET_COLORS[dataset],
            hatch=DATASET_HATCHES[dataset],
            edgecolor="black",
            linewidth=0.4,
            zorder=2,
        )
        if has_ci:
            ax.errorbar(
                x + offsets[dataset],
                means,
                yerr=cis,
                fmt="none",
                ecolor="black",
                elinewidth=0.5,
                capsize=1.5,
                capthick=0.5,
                zorder=3,
            )

        cum_means = [
            agg_cum[(agg_cum["dataset"] == dataset) & (agg_cum["quantize"] == stage)].iloc[0]["mean"]
            for stage in STAGES
        ]
        ax.plot(
            x,
            cum_means,
            color=DATASET_COLORS[dataset],
            linestyle="--",
            linewidth=0.7,
            marker="o",
            markersize=2.5,
            zorder=4,
        )

    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="y", which="minor", length=1.2, width=0.4)

    ax.set_ylabel("Time (s)")
    _set_quantize_xaxis(ax, STAGES)
    _style_axes(ax)

    _save(fig, "ablation-construction-time.png", cfg)
    return agg_stage


# --------------------------------------------------------------------------- #
# Panel (e): ball count per quantize variant
# --------------------------------------------------------------------------- #

def panel_n_balls(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Realized ball count per quantize variant, at the one memory budget
    shared by the whole sweep (derived from cfg.k against the unquantized
    cover's byte-per-ball size). Unlike a per-level sweep -- where n_balls
    would always equal k regardless of construction strategy -- this is
    genuinely informative here: a quantizer with a smaller per-center code
    packs more balls into the same budget, so bars are expected to rise as
    the code gets more compact (PQ/SQ4 highest, None/FP16/BF16 lowest, at
    k itself)."""
    agg = aggregate(df, "n_balls", cfg, use_tau=False)

    apply_style()
    fig, ax = plt.subplots()
    x = np.arange(len(STAGES))
    offsets = _dataset_offsets()

    for dataset in DATASETS:
        means, cis = [], []
        for stage in STAGES:
            row = agg[(agg["dataset"] == dataset) & (agg["quantize"] == stage)].iloc[0]
            means.append(row["mean"])
            cis.append(row["ci95"])
        means = np.array(means)
        cis = np.array(cis)
        has_ci = np.any(cis > 0)

        ax.bar(
            x + offsets[dataset],
            means,
            width=BAR_WIDTH,
            color=DATASET_COLORS[dataset],
            hatch=DATASET_HATCHES[dataset],
            edgecolor="black",
            linewidth=0.4,
            zorder=2,
        )
        if has_ci:
            ax.errorbar(
                x + offsets[dataset],
                means,
                yerr=cis,
                fmt="none",
                ecolor="black",
                elinewidth=0.5,
                capsize=1.5,
                capthick=0.5,
                zorder=3,
            )

    ax.axhline(cfg.k, color="black", linewidth=0.6, linestyle=":", zorder=1)
    ax.text(len(STAGES) - 0.5, cfg.k, f"k={cfg.k}", fontsize=5, ha="right", va="bottom")

    ax.set_ylabel("Number of balls")
    _set_quantize_xaxis(ax, STAGES)
    _style_axes(ax)

    _save(fig, "ablation-n-balls.png", cfg)
    return agg


# --------------------------------------------------------------------------- #
# Shared legend
# --------------------------------------------------------------------------- #

def save_legend(cfg: Config) -> None:
    save_dataset_legend(cfg.outdir, cfg.dpi, cfg.emit_pdf, filename="ablation-legend.png")


# --------------------------------------------------------------------------- #
# Save helper
# --------------------------------------------------------------------------- #

def _save(fig: plt.Figure, filename: str, cfg: Config) -> None:
    save_figure(fig, filename, cfg.outdir, cfg.dpi, cfg.emit_pdf)


# --------------------------------------------------------------------------- #
# Summary table
# --------------------------------------------------------------------------- #

def print_summary(df: pd.DataFrame, cfg: Config) -> None:
    print("\n--- Ablation summary (k={}, tau={:g}) ---".format(cfg.k, cfg.tau))
    header = (f"{'dataset':<10} {'quantize':>9} {'d(log10 J) vs None':>20} "
              f"{'fpr':>12} {'time (s)':>10} {'n_balls':>8}")
    print(header)
    print("-" * len(header))

    j_agg = aggregate(df, "log10_J", cfg, use_tau=False)
    fpr_agg = aggregate(df, "fpr", cfg, use_tau=True)
    time_agg = aggregate(df, "cum_time_s", cfg, use_tau=False)
    balls_agg = aggregate(df, "n_balls", cfg, use_tau=False)

    for dataset in DATASETS:
        base_j, _ = _lookup(j_agg, dataset, BASELINE_STAGE)
        for stage in STAGES:
            this_j, _ = _lookup(j_agg, dataset, stage)
            this_fpr, _ = _lookup(fpr_agg, dataset, stage)
            this_time, _ = _lookup(time_agg, dataset, stage)
            this_balls, _ = _lookup(balls_agg, dataset, stage)
            print(
                f"{DATASET_LABELS[dataset]:<10} {STAGE_LABELS[stage]:>9} {this_j - base_j:>20.3f} "
                f"{this_fpr:>12.2e} {this_time:>10.3f} {this_balls:>8.1f}"
            )
    print()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def generate_ablation_figures(
    df: Optional[pd.DataFrame] = None,
    outdir: str = OUTDIR,
    k: int = K,
    tau: Optional[float] = TAU,
    dpi: int = DPI,
    emit_pdf: bool = EMIT_PDF,
) -> None:
    """
    Library entry point: render the five ablation panels (+ legend) from an
    in-memory long-format DataFrame instead of round-tripping through a file.
    If `df` is omitted, falls back to `load_data()` (CONFIG-driven, reads
    INPUT_PATH) so the script still works standalone.
    """
    if df is None:
        df = load_data()
    else:
        _validate_schema(df)

    cfg = resolve_config(df, k=k, tau=tau, dpi=dpi, emit_pdf=emit_pdf, outdir=outdir)

    panel_geometric_objective(df, cfg)
    panel_threshold_objective(df, cfg)
    panel_fpr(df, cfg)
    panel_construction_time(df, cfg)
    panel_n_balls(df, cfg)
    save_legend(cfg)

    print_summary(df, cfg)

    snippet = r"""
\begin{subfigure}[b]{\linewidth}
    \centering
    \includegraphics[width=0.6\linewidth]{figures/ablation-legend.png}
\end{subfigure}\\[-0.3em]
\begin{subfigure}[b]{0.24\linewidth}
    \centering
    \includegraphics[width=\linewidth]{figures/ablation-geometric-objective.png}
    \caption{Geometric objective}
\end{subfigure}
\begin{subfigure}[b]{0.24\linewidth}
    \centering
    \includegraphics[width=\linewidth]{figures/ablation-threshold-expanded-objective.png}
    \caption{$\tau$-expanded objective}
\end{subfigure}
\begin{subfigure}[b]{0.24\linewidth}
    \centering
    \includegraphics[width=\linewidth]{figures/ablation-false-positive-rate.png}
    \caption{False positive rate}
\end{subfigure}
\begin{subfigure}[b]{0.24\linewidth}
    \centering
    \includegraphics[width=\linewidth]{figures/ablation-construction-time.png}
    \caption{Construction time}
\end{subfigure}
\begin{subfigure}[b]{0.24\linewidth}
    \centering
    \includegraphics[width=\linewidth]{figures/ablation-n-balls.png}
    \caption{Ball count}
\end{subfigure}
""".strip("\n")
    print("--- subfigure snippet (paste above the panel row) ---")
    print(snippet)


def main() -> None:
    generate_ablation_figures()


if __name__ == "__main__":
    main()
