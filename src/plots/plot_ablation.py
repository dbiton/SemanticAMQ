"""Generate the five ablation-figure panels (+ shared legend) for the PVLDB paper.

Self-contained: reads a long-format results file (one row per (quantize, k,
seed) on a single dataset -- see main.py::generate_ablation), aggregates over
seeds, and writes PNG (optionally PDF) panels sized for
`\\includegraphics[width=\\linewidth]` inside `subfigure{0.24\\linewidth}` cells
of a two-column `figure*`.

Each panel is a line plot: x-axis = memory usage (MB, log scale, driven by
the k sweep -- see ablation_worker's docstring for how closely memory usage
tracks k for a given quantize choice, which isn't exact for the most compact
quantizers), one line per quantize method (color + marker), for the single
dataset the input was generated on. This replaced an earlier per-dataset
grouped-bar-at-fixed-k design; multi-dataset figures now belong to the other
evaluations (plot_query_performance.py etc.), not this one.

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
from plots.plot_common import apply_style as _apply_style, save_figure

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #

INPUT_PATH = "results/ablation.csv"
OUTDIR = "plots/ablation/"
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

# tab10 has 10 qualitative colors; 9 quantize variants fit with one to spare.
_STAGE_PALETTE = plt.get_cmap("tab10").colors
STAGE_COLORS = dict(zip(STAGES, _STAGE_PALETTE))
_STAGE_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h"]
STAGE_MARKERS = dict(zip(STAGES, _STAGE_MARKERS))

D_DIM = 384  # embedding dimensionality; objectives are stored pre-logged


@dataclasses.dataclass
class Config:
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
    ax.grid(axis="both", which="major", linewidth=0.35, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.5)


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
    "memory_bytes",
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

    n_datasets = df["dataset"].nunique()
    if n_datasets != 1:
        raise ValueError(
            f"Expected a single-dataset input (see generate_ablation()), found "
            f"{n_datasets}: {sorted(df['dataset'].unique())}"
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
    tau: Optional[float] = TAU,
    dpi: int = DPI,
    emit_pdf: bool = EMIT_PDF,
    outdir: str = OUTDIR,
) -> Config:
    available_tau = sorted(df["tau"].unique())
    if tau is None:
        tau = float(df["tau"].median())
        if tau not in available_tau:
            tau = min(available_tau, key=lambda t: abs(t - tau))
    elif tau not in available_tau:
        raise ValueError(f"tau={tau} not present in data. Available tau values: {available_tau}")

    return Config(tau=tau, dpi=dpi, emit_pdf=emit_pdf, outdir=outdir)


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def aggregate(df: pd.DataFrame, value_col: str, cfg: Config, use_tau: bool = False,
             stages: list = STAGES) -> pd.DataFrame:
    """Mean + 95% CI of `value_col` over seed, per (quantize, k) -- k is the
    real sweep/grouping key (memory_bytes is deterministic given (k,
    quantize), so grouping by k and separately averaging memory_bytes gives
    the same result more robustly than grouping by a float column directly),
    restricted to the given `stages` (quantize values) and, if `use_tau`, to
    cfg.tau."""
    sel = df[df["quantize"].isin(stages)]
    if use_tau:
        sel = sel[sel["tau"] == cfg.tau]

    bad_raw = sel[~np.isfinite(sel[value_col])]
    if len(bad_raw):
        cols = [c for c in ("k", "quantize", "tau", "stage", "seed", value_col) if c in bad_raw.columns]
        raise ValueError(
            f"Non-finite {value_col!r} values"
            + (f" at tau={cfg.tau}" if use_tau else "") + f":\n{bad_raw[cols].to_string(index=False)}\n"
            "This usually means k >= the construct-set size (every ball becomes a "
            "zero-radius singleton, so log10_J is -inf) -- lower k or shrink the "
            "construct-set-relative k sweep in the data-generation step."
        )

    grouped = sel.groupby(["quantize", "k"])
    agg = grouped[value_col].agg(mean="mean", sem="sem", n="count").reset_index()
    agg["ci95"] = np.where(agg["n"] > 1, 1.96 * agg["sem"].fillna(0.0), 0.0)
    mem = grouped["memory_bytes"].mean().reset_index(name="memory_bytes")
    agg = agg.merge(mem, on=["quantize", "k"])
    agg["Memory (MB)"] = agg["memory_bytes"] / (1024 * 1024)

    for stage in stages:
        if not (agg["quantize"] == stage).any():
            raise ValueError(
                f"No rows for quantize={stage!r}" + (f" at tau={cfg.tau}" if use_tau else "") + "."
            )
    return agg


def _relative_to_baseline(agg: pd.DataFrame) -> pd.DataFrame:
    """Subtracts BASELINE_STAGE's mean (at the same k) from every stage's
    mean, so each line reads as "how much does this quantizer cost/save
    relative to the unquantized cover at the same memory budget" -- the
    generalization of the old fixed-k bar chart's per-dataset baseline
    subtraction to a k-swept line chart."""
    base = agg[agg["quantize"] == BASELINE_STAGE][["k", "mean"]].rename(columns={"mean": "base_mean"})
    rel = agg.merge(base, on="k", how="left")
    rel["mean"] = rel["mean"] - rel["base_mean"]
    return rel.drop(columns="base_mean")


# --------------------------------------------------------------------------- #
# Generic line panel: x = memory usage, one line per quantize method
# --------------------------------------------------------------------------- #

def _line_panel(
    df: pd.DataFrame,
    value_col: str,
    ylabel: str,
    filename: str,
    cfg: Config,
    use_tau: bool = False,
    log_y: bool = False,
    y_bounds: Optional[tuple] = None,
    relative_to_baseline: bool = False,
    extra=None,
) -> pd.DataFrame:
    agg = aggregate(df, value_col, cfg, use_tau=use_tau)
    if relative_to_baseline:
        agg = _relative_to_baseline(agg)

    apply_style()
    fig, ax = plt.subplots()

    for stage in STAGES:
        sub = agg[agg["quantize"] == stage].sort_values("Memory (MB)")
        if sub.empty:
            continue
        ax.plot(
            sub["Memory (MB)"], sub["mean"],
            color=STAGE_COLORS[stage], marker=STAGE_MARKERS[stage], markersize=2.5,
            linewidth=0.8, zorder=2,
        )
        if (sub["ci95"] > 0).any():
            ax.errorbar(
                sub["Memory (MB)"], sub["mean"], yerr=sub["ci95"],
                fmt="none", ecolor=STAGE_COLORS[stage],
                elinewidth=0.5, capsize=1.5, capthick=0.5, zorder=3,
            )

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(LogLocator(base=2.0))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=2.0))

    if relative_to_baseline:
        ax.axhline(0.0, color="black", linewidth=0.5, zorder=1)
    if log_y:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10.0))
        ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="y", which="minor", length=1.2, width=0.4)
    elif y_bounds is not None:
        ax.set_ylim(*y_bounds)

    ax.set_xlabel("Memory Usage (MB)")
    ax.set_ylabel(ylabel)
    _style_axes(ax)
    if extra is not None:
        extra(ax)

    _save(fig, filename, cfg)
    return agg


# --------------------------------------------------------------------------- #
# Panel (a): geometric objective reduction
# --------------------------------------------------------------------------- #

def panel_geometric_objective(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    return _line_panel(
        df, "log10_J", r"$\log_{10} J(\mathcal{C})/J_{\mathrm{ball}}$",
        "ablation-geometric-objective.png", cfg, use_tau=False, relative_to_baseline=True,
    )


# --------------------------------------------------------------------------- #
# Panel (b): tau-expanded objective reduction
# --------------------------------------------------------------------------- #

def panel_threshold_objective(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    def annotate(ax: plt.Axes) -> None:
        ax.text(
            0.97, 0.95, rf"$\tau = {cfg.tau:g}$",
            transform=ax.transAxes, ha="right", va="top", fontsize=5.5,
        )

    return _line_panel(
        df, "log10_J_tau", r"$\log_{10} J_\tau(\mathcal{C})/J_{\tau,\mathrm{ball}}$",
        "ablation-threshold-expanded-objective.png", cfg, use_tau=True,
        relative_to_baseline=True, extra=annotate,
    )


# --------------------------------------------------------------------------- #
# Panel (c): false positive rate
# --------------------------------------------------------------------------- #

def panel_fpr(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    agg = aggregate(df, "fpr", cfg, use_tau=True)

    positive = agg.loc[agg["mean"] > 0, "mean"]
    if len(positive) == 0:
        raise ValueError("All FPR values are zero at the selected tau; cannot set a log floor.")
    floor = 10 ** (np.floor(np.log10(positive.min())) - 1)
    # Zero FPR can't be drawn on a log axis; clip to the floor so the line
    # still traces through those points instead of vanishing off-chart.
    agg = agg.copy()
    agg["mean"] = agg["mean"].where(agg["mean"] > 0, floor)

    apply_style()
    fig, ax = plt.subplots()
    for stage in STAGES:
        sub = agg[agg["quantize"] == stage].sort_values("Memory (MB)")
        if sub.empty:
            continue
        ax.plot(
            sub["Memory (MB)"], sub["mean"],
            color=STAGE_COLORS[stage], marker=STAGE_MARKERS[stage], markersize=2.5,
            linewidth=0.8, zorder=2,
        )
        if (sub["ci95"] > 0).any():
            ax.errorbar(
                sub["Memory (MB)"], sub["mean"], yerr=sub["ci95"],
                fmt="none", ecolor=STAGE_COLORS[stage],
                elinewidth=0.5, capsize=1.5, capthick=0.5, zorder=3,
            )

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(LogLocator(base=2.0))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=2.0))

    ax.set_yscale("log")
    ax.set_ylim(floor / 3.0, None)
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="y", which="minor", length=1.2, width=0.4)

    ax.set_xlabel("Memory Usage (MB)")
    ax.set_ylabel("FPR")
    _style_axes(ax)

    _save(fig, "ablation-false-positive-rate.png", cfg)
    return agg


# --------------------------------------------------------------------------- #
# Panel (d): construction time
# --------------------------------------------------------------------------- #

def panel_construction_time(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    # stage_time_s == cum_time_s in ablation_worker (each row is a single
    # from-scratch construction, not a multi-stage pipeline), so there's no
    # separate "cumulative" series to overlay here the way the old bar panel
    # did -- one line per quantize method is the whole picture.
    return _line_panel(
        df, "cum_time_s", "Time (s)", "ablation-construction-time.png", cfg,
        use_tau=False, log_y=True,
    )


# --------------------------------------------------------------------------- #
# Panel (e): ball count per quantize variant
# --------------------------------------------------------------------------- #

def panel_n_balls(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Realized ball count per quantize variant across the memory-usage
    sweep. Unlike a per-level sweep -- where n_balls would always equal k
    regardless of construction strategy -- this is genuinely informative
    here: a quantizer with a smaller per-center code packs more balls into
    the same budget, so at a given x (memory usage) the lines are expected to
    separate, with PQ/SQ4 highest and None/FP16/BF16 lowest."""
    return _line_panel(
        df, "n_balls", "Number of Balls", "ablation-n-balls.png", cfg, use_tau=False,
    )


# --------------------------------------------------------------------------- #
# Shared legend
# --------------------------------------------------------------------------- #

def save_legend(cfg: Config) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(5.6, 0.32))
    ax.axis("off")

    handles = [
        plt.Line2D(
            [0], [0],
            color=STAGE_COLORS[stage], marker=STAGE_MARKERS[stage],
            markersize=3, linewidth=1.0,
        )
        for stage in STAGES
    ]
    labels = [STAGE_LABELS[stage] for stage in STAGES]

    fig.legend(
        handles, labels,
        loc="center", ncol=min(5, len(STAGES)),
        frameon=False, fontsize=7,
    )
    fig.patch.set_alpha(0.0)

    path = os.path.join(cfg.outdir, "ablation-legend.png")
    os.makedirs(cfg.outdir, exist_ok=True)
    fig.savefig(path, dpi=cfg.dpi, bbox_inches="tight", pad_inches=0.01, transparent=True)
    if cfg.emit_pdf:
        fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight", pad_inches=0.01, transparent=True)
    plt.close(fig)
    print(f"Wrote {path}")


# --------------------------------------------------------------------------- #
# Save helper
# --------------------------------------------------------------------------- #

def _save(fig: plt.Figure, filename: str, cfg: Config) -> None:
    save_figure(fig, filename, cfg.outdir, cfg.dpi, cfg.emit_pdf)


# --------------------------------------------------------------------------- #
# Summary table
# --------------------------------------------------------------------------- #

def print_summary(df: pd.DataFrame, cfg: Config) -> None:
    print(f"\n--- Ablation summary (tau={cfg.tau:g}) ---")
    header = (f"{'quantize':>9} {'memory (MB)':>12} {'d(log10 J) vs None':>20} "
              f"{'fpr':>12} {'time (s)':>10} {'n_balls':>8}")
    print(header)
    print("-" * len(header))

    j_agg = _relative_to_baseline(aggregate(df, "log10_J", cfg, use_tau=False))
    fpr_agg = aggregate(df, "fpr", cfg, use_tau=True)
    time_agg = aggregate(df, "cum_time_s", cfg, use_tau=False)
    balls_agg = aggregate(df, "n_balls", cfg, use_tau=False)

    for stage in STAGES:
        for _, row in j_agg[j_agg["quantize"] == stage].sort_values("k").iterrows():
            k = row["k"]
            this_fpr = fpr_agg[(fpr_agg["quantize"] == stage) & (fpr_agg["k"] == k)]["mean"].iloc[0]
            this_time = time_agg[(time_agg["quantize"] == stage) & (time_agg["k"] == k)]["mean"].iloc[0]
            this_balls = balls_agg[(balls_agg["quantize"] == stage) & (balls_agg["k"] == k)]["mean"].iloc[0]
            print(
                f"{STAGE_LABELS[stage]:>9} {row['Memory (MB)']:>12.3f} {row['mean']:>20.3f} "
                f"{this_fpr:>12.2e} {this_time:>10.3f} {this_balls:>8.1f}"
            )
    print()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def generate_ablation_figures(
    df: Optional[pd.DataFrame] = None,
    outdir: str = OUTDIR,
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

    cfg = resolve_config(df, tau=tau, dpi=dpi, emit_pdf=emit_pdf, outdir=outdir)

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
