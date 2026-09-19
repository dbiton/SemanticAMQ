"""Generates the six quantization-figure panels (+ shared legend), plus one
diagnostic FNR sanity-check panel not part of the paper figure set, from a
long-format results file (one row per (quantize, k, seed) on a single
dataset -- see main.py::generate_quantization). Each panel: x-axis = memory
usage (MB, log scale), one line per quantize method, single dataset.

Edit the CONFIG block below to change parameters -- no CLI flags."""

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

from filters.ball_cover.ball_cover_filter import Quantize
from plots.plot_common import apply_style as _apply_style, save_figure

# CONFIG

INPUT_PATH = "results/quantization.csv"
OUTDIR = "plots/quantization/"
TAU: Optional[float] = None  # None => median tau present in the data
DPI = 400
EMIT_PDF = False

# Mirrors runner.QUANTIZATION_SWEEP (not imported, to avoid a dependency on
# runner.py's multiprocessing machinery). NONE is the raw/unquantized cover
# (stage="ball"); every other entry shares that cover's memory budget
# (stage="quantizing"), so a more compact code packs more balls into it.
QUANTIZE_SWEEP = (
    Quantize.NONE, Quantize.SQ8, Quantize.SQ4,
    Quantize.FP16, Quantize.BF16, Quantize.SQ4T, Quantize.SQ8T, Quantize.PQ,
)
STAGES = [q.value for q in QUANTIZE_SWEEP]
STAGE_LABELS = {
    Quantize.NONE.value: "None",
    Quantize.SQ8.value: "SQ8",
    Quantize.SQ4.value: "SQ4",
    Quantize.FP16.value: "FP16",
    Quantize.BF16.value: "BF16",
    Quantize.SQ4T.value: "SQ4-T",
    Quantize.SQ8T.value: "SQ8-T",
    Quantize.PQ.value: "PQ",
}
# Reference stage for the two relative-objective panels: the raw/unquantized
# cover, matching the "reduction relative to the ball cover" reading the
# panels had before per-quantize breakdown existed.
BASELINE_STAGE = Quantize.NONE.value

# tab10 has 10 qualitative colors; 10 quantize variants use exactly all of them.
_STAGE_PALETTE = plt.get_cmap("tab10").colors
STAGE_COLORS = dict(zip(STAGES, _STAGE_PALETTE))
_STAGE_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "p"]
STAGE_MARKERS = dict(zip(STAGES, _STAGE_MARKERS))

D_DIM = 384  # embedding dimensionality; objectives are stored pre-logged


@dataclasses.dataclass
class Config:
    tau: float
    dpi: int
    emit_pdf: bool
    outdir: str


# Style

def apply_style() -> None:
    _apply_style(DPI)


def _style_axes(ax: plt.Axes) -> None:
    ax.grid(axis="both", which="major", linewidth=0.35, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.5)


# Data loading

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
    "query_throughput_qps",
    "n_balls",
    "memory_bytes",
    "seed",
    "radius_inflation_mean",
    "radius_inflation_std",
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
            f"Expected a single-dataset input (see generate_quantization()), found "
            f"{n_datasets}: {sorted(df['dataset'].unique())}"
        )

    present = set(df["quantize"].unique())
    missing_combos = [s for s in STAGES if s not in present]
    if missing_combos:
        # Not fatal: quantization_worker skips a candidate whose construction
        # raises (e.g. PQ needing >= 2 ball centers); every panel already
        # tolerates an absent stage by not drawing/counting it.
        print(f"Note: quantize value(s) {missing_combos} have no rows (skipped during "
              f"generation, e.g. InsufficientMemoryError) and won't be plotted. "
              f"Available: {sorted(present)}")


def load_data() -> pd.DataFrame:
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH!r}. Run generate_quantization() (main.py) "
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


# Aggregation

def aggregate(df: pd.DataFrame, value_col: str, cfg: Config, use_tau: bool = False,
             stages: list = STAGES) -> pd.DataFrame:
    """Mean + 95% CI of `value_col` over seed, per (quantize, k) -- k is the
    real grouping key (memory_bytes is deterministic given (k, quantize), so
    grouping by k and separately averaging memory_bytes is more robust than
    grouping by a float column). Restricted to `stages` and, if `use_tau`, cfg.tau."""
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
    return agg


def _relative_to_baseline(agg: pd.DataFrame) -> pd.DataFrame:
    """Subtracts BASELINE_STAGE's mean (at the same k) from every stage's
    mean, so each line reads as "how much does this quantizer cost/save
    relative to the unquantized cover at the same memory budget"."""
    base = agg[agg["quantize"] == BASELINE_STAGE][["k", "mean"]].rename(columns={"mean": "base_mean"})
    rel = agg.merge(base, on="k", how="left")
    rel["mean"] = rel["mean"] - rel["base_mean"]
    return rel.drop(columns="base_mean")


# Generic line panel: x = memory usage, one line per quantize method

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

    _fix_margins(fig)
    _save(fig, filename, cfg)
    return agg


# Panel (a): geometric objective reduction

def panel_geometric_objective(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    return _line_panel(
        df, "log10_J", r"$\log_{10} J(\mathcal{C})/J_{\mathrm{ball}}$",
        "quantization-geometric-objective.png", cfg, use_tau=False, relative_to_baseline=True,
    )


# Panel (b): tau-expanded objective reduction

def panel_threshold_objective(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    def annotate(ax: plt.Axes) -> None:
        ax.text(
            0.97, 0.95, rf"$\tau = {cfg.tau:g}$",
            transform=ax.transAxes, ha="right", va="top", fontsize=5.5,
        )

    return _line_panel(
        df, "log10_J_tau", r"$\log_{10} J_\tau(\mathcal{C})/J_{\tau,\mathrm{ball}}$",
        "quantization-threshold-expanded-objective.png", cfg, use_tau=True,
        relative_to_baseline=True, extra=annotate,
    )


# Panel (c): false positive rate

def panel_fpr(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    # Linear, not log: FPR is legitimately 0 for many points (a log axis
    # can't represent 0). y-min pinned to 0, top auto-scales.
    return _line_panel(
        df, "fpr", "FPR", "quantization-false-positive-rate.png", cfg,
        use_tau=True, y_bounds=(0, None),
    )


# Panel (d): construction time

def panel_construction_time(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    # stage_time_s == cum_time_s in quantization_worker (single from-scratch
    # construction per row), so one line per quantize method is the whole picture.
    return _line_panel(
        df, "cum_time_s", "Time (s)", "quantization-construction-time.png", cfg,
        use_tau=False, log_y=True,
    )


# Panel (e): query throughput

def panel_query_throughput(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Batched-query throughput of filt.query(), per quantize variant. Query
    cost doesn't depend on tau (every ball is checked regardless of
    threshold), so this averages over the tau sweep -- log-scaled since a
    quantizer's decode overhead (e.g. PQ) can separate lines by an order of
    magnitude from the raw/NONE cover."""
    return _line_panel(
        df, "query_throughput_qps", "Throughput (q/s)",
        "quantization-query-throughput.png", cfg, use_tau=False, log_y=True,
    )


# Panel (f): ball count per quantize variant

def panel_n_balls(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Realized ball count per quantize variant across the memory-usage
    sweep: a quantizer with a smaller per-center code packs more balls into
    the same budget, so at a given x the lines are expected to separate,
    with PQ/SQ4 highest and None/FP16/BF16 lowest."""
    return _line_panel(
        df, "n_balls", "Number of Balls", "quantization-n-balls.png", cfg, use_tau=False,
    )


# Panel (g): radius inflation due to quantization

def panel_radius_inflation(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Mean +/- stddev of each ball's radius inflation (the per-center
    quantization reconstruction error BallCoverFilter._build_quantizer adds
    to preserve the no-false-negative guarantee -- see all_radius_inflation).
    Unlike other panels' seed-CI error bars, mean/std here are collapsed
    *across balls* within one job, so the error bars show ball-to-ball
    spread, not seed variance. Quantize.NONE has zero inflation by
    construction, drawing as a flat reference line at 0."""
    mean_agg = aggregate(df, "radius_inflation_mean", cfg, use_tau=False)
    std_agg = aggregate(df, "radius_inflation_std", cfg, use_tau=False)
    std_agg = std_agg[["quantize", "k", "mean"]].rename(columns={"mean": "std"})
    agg = mean_agg.merge(std_agg, on=["quantize", "k"])

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
        ax.errorbar(
            sub["Memory (MB)"], sub["mean"], yerr=sub["std"],
            fmt="none", ecolor=STAGE_COLORS[stage],
            elinewidth=0.5, capsize=1.5, capthick=0.5, zorder=3,
        )

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(LogLocator(base=2.0))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=2.0))

    # Log y: inflation spans orders of magnitude across quantize variants.
    # Quantize.NONE's identically-0 inflation has no representation on a log
    # axis, so its line silently drops off rather than plotting at y=0.
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="y", which="minor", length=1.2, width=0.4)

    ax.set_xlabel("Memory Usage (MB)")
    ax.set_ylabel("Radius Inflation")
    _style_axes(ax)

    _fix_margins(fig)
    _save(fig, "quantization-radius-inflation.png", cfg)
    return agg


# Diagnostic panel (not part of the paper figure set): false negative rate

def panel_fnr(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Sanity-check panel, not for the paper: FNR should be exactly 0 for
    every row (already hard-asserted in quantization_worker and re-checked
    by _validate_schema), so this renders as a flat line at 0 under normal
    operation -- fixed [0, 1] y-range so a violation would be dramatically
    visible instead of auto-scaling into a misleading wobble."""
    return _line_panel(
        df, "fnr", "FNR", "quantization-false-negative-rate.png", cfg,
        use_tau=True, y_bounds=(-0.02, 1.02),
    )


# Shared legend

def save_legend(df: pd.DataFrame, cfg: Config) -> None:
    # Only legend a stage that actually has data.
    present = set(df["quantize"].unique())
    stages = [s for s in STAGES if s in present]

    apply_style()
    fig, ax = plt.subplots(figsize=(5.6, 0.32))
    ax.axis("off")

    handles = [
        plt.Line2D(
            [0], [0],
            color=STAGE_COLORS[stage], marker=STAGE_MARKERS[stage],
            markersize=3, linewidth=1.0,
        )
        for stage in stages
    ]
    labels = [STAGE_LABELS[stage] for stage in stages]

    fig.legend(
        handles, labels,
        loc="center", ncol=min(5, len(stages)),
        frameon=False, fontsize=7,
    )
    fig.patch.set_alpha(0.0)

    path = os.path.join(cfg.outdir, "quantization-legend.png")
    os.makedirs(cfg.outdir, exist_ok=True)
    fig.savefig(path, dpi=cfg.dpi, bbox_inches="tight", pad_inches=0.01, transparent=True)
    if cfg.emit_pdf:
        fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight", pad_inches=0.01, transparent=True)
    plt.close(fig)
    print(f"Wrote {path}")


# Save helper

def _fix_margins(fig: plt.Figure) -> None:
    """Fixed margins (not bbox_inches="tight") so every panel in this figure
    set shares the same canvas size and axes position -- needed since they
    get placed side by side, and "tight" would crop each one differently
    depending on its own y-label/tick width."""
    fig.subplots_adjust(left=0.30, right=0.97, top=0.92, bottom=0.24)


def _save(fig: plt.Figure, filename: str, cfg: Config) -> None:
    save_figure(fig, filename, cfg.outdir, cfg.dpi, cfg.emit_pdf, tight=False)


# Summary table

def print_summary(df: pd.DataFrame, cfg: Config) -> None:
    print(f"\n--- Quantization summary (tau={cfg.tau:g}) ---")
    header = (f"{'quantize':>9} {'memory (MB)':>12} {'d(log10 J) vs None':>20} "
              f"{'fpr':>12} {'time (s)':>10} {'qps':>10} {'n_balls':>8}")
    print(header)
    print("-" * len(header))

    j_agg = _relative_to_baseline(aggregate(df, "log10_J", cfg, use_tau=False))
    fpr_agg = aggregate(df, "fpr", cfg, use_tau=True)
    time_agg = aggregate(df, "cum_time_s", cfg, use_tau=False)
    qps_agg = aggregate(df, "query_throughput_qps", cfg, use_tau=False)
    balls_agg = aggregate(df, "n_balls", cfg, use_tau=False)

    for stage in STAGES:
        for _, row in j_agg[j_agg["quantize"] == stage].sort_values("k").iterrows():
            k = row["k"]
            this_fpr = fpr_agg[(fpr_agg["quantize"] == stage) & (fpr_agg["k"] == k)]["mean"].iloc[0]
            this_time = time_agg[(time_agg["quantize"] == stage) & (time_agg["k"] == k)]["mean"].iloc[0]
            this_qps = qps_agg[(qps_agg["quantize"] == stage) & (qps_agg["k"] == k)]["mean"].iloc[0]
            this_balls = balls_agg[(balls_agg["quantize"] == stage) & (balls_agg["k"] == k)]["mean"].iloc[0]
            print(
                f"{STAGE_LABELS[stage]:>9} {row['Memory (MB)']:>12.3f} {row['mean']:>20.3f} "
                f"{this_fpr:>12.2e} {this_time:>10.3f} {this_qps:>10.1f} {this_balls:>8.1f}"
            )
    print()


# Main

def generate_quantization_figures(
    df: Optional[pd.DataFrame] = None,
    outdir: str = OUTDIR,
    tau: Optional[float] = TAU,
    dpi: int = DPI,
    emit_pdf: bool = EMIT_PDF,
) -> None:
    """Library entry point: renders the six paper panels + one diagnostic
    FNR panel (+ legend) from an in-memory DataFrame, or from `load_data()`
    (reads INPUT_PATH) if `df` is omitted. panel_fnr is excluded from the
    LaTeX subfigure snippet printed below, since it's a sanity check, not
    part of the paper figure set."""
    if df is None:
        df = load_data()
    else:
        _validate_schema(df)

    cfg = resolve_config(df, tau=tau, dpi=dpi, emit_pdf=emit_pdf, outdir=outdir)

    panel_geometric_objective(df, cfg)
    panel_threshold_objective(df, cfg)
    panel_fpr(df, cfg)
    panel_construction_time(df, cfg)
    panel_query_throughput(df, cfg)
    panel_n_balls(df, cfg)
    panel_radius_inflation(df, cfg)
    panel_fnr(df, cfg)
    save_legend(df, cfg)

    print_summary(df, cfg)

    snippet = r"""
\begin{subfigure}[b]{\linewidth}
    \centering
    \includegraphics[width=0.6\linewidth]{figures/quantization-legend.png}
\end{subfigure}\\[-0.3em]
\begin{subfigure}[b]{0.24\linewidth}
    \centering
    \includegraphics[width=\linewidth]{figures/quantization-geometric-objective.png}
    \caption{Geometric objective}
\end{subfigure}
\begin{subfigure}[b]{0.24\linewidth}
    \centering
    \includegraphics[width=\linewidth]{figures/quantization-threshold-expanded-objective.png}
    \caption{$\tau$-expanded objective}
\end{subfigure}
\begin{subfigure}[b]{0.24\linewidth}
    \centering
    \includegraphics[width=\linewidth]{figures/quantization-false-positive-rate.png}
    \caption{False positive rate}
\end{subfigure}
\begin{subfigure}[b]{0.24\linewidth}
    \centering
    \includegraphics[width=\linewidth]{figures/quantization-construction-time.png}
    \caption{Construction time}
\end{subfigure}
\begin{subfigure}[b]{0.24\linewidth}
    \centering
    \includegraphics[width=\linewidth]{figures/quantization-query-throughput.png}
    \caption{Query throughput}
\end{subfigure}
\begin{subfigure}[b]{0.24\linewidth}
    \centering
    \includegraphics[width=\linewidth]{figures/quantization-n-balls.png}
    \caption{Ball count}
\end{subfigure}
\begin{subfigure}[b]{0.24\linewidth}
    \centering
    \includegraphics[width=\linewidth]{figures/quantization-radius-inflation.png}
    \caption{Radius inflation}
\end{subfigure}
""".strip("\n")
    print("--- subfigure snippet (paste above the panel row) ---")
    print(snippet)


def main() -> None:
    generate_quantization_figures()


if __name__ == "__main__":
    main()
