"""Generates the seven quantization-figure panels (+ shared legend), plus one
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
# The quantization sweep is single-tau by design (see
# main.py::generate_quantization): every row shares one threshold, so there
# is nothing to select. Left as a config knob only to pin an expected value
# -- None accepts whatever single tau the data carries.
TAU: Optional[float] = None
DPI = 400
EMIT_PDF = False

# Mirrors runner.QUANTIZATION_SWEEP (not imported, to avoid a dependency on
# runner.py's multiprocessing machinery). NONE is the raw/unquantized cover
# (stage="ball"); every other entry shares that cover's memory budget
# (stage="quantizing"), so a more compact code packs more balls into it.
QUANTIZE_SWEEP = (
    Quantize.FP32, Quantize.FP16, Quantize.SQ8, Quantize.SQ6, Quantize.SQ4,
    Quantize.TQ3, Quantize.PQ4_R2, Quantize.PQ8_R2,
    # Down to 1 b/dim via the same two-subquantizer-width pairing the 2 b/dim
    # rung already has -- see runner.QUANTIZATION_SWEEP. Quantize.RABITQ is
    # deliberately excluded (its zero-false-negative property is an
    # empirical bound, not a proof, unlike every other rung). A results file
    # from before PQ4_R1/PQ8_R1 were swept simply has no rows for them;
    # every panel already tolerates an absent stage by not drawing/counting
    # it -- and one that still carries old RABITQ rows plots without
    # complaint too, since STAGES no longer includes it.
    Quantize.PQ4_R1, Quantize.PQ8_R1,
)
STAGES = [q.value for q in QUANTIZE_SWEEP]
STAGE_LABELS = {
    # Labelled by rate so the ladder reads in bits/dim order on an axis.
    Quantize.FP32.value: "FP32 (32b)",
    Quantize.FP16.value: "FP16 (16b)",
    Quantize.SQ8.value: "SQ8 (8b)",
    Quantize.SQ6.value: "SQ6 (6b)",
    Quantize.SQ4.value: "SQ4 (4b)",
    Quantize.TQ3.value: "TQ3 (3b)",
    Quantize.PQ4_R2.value: "PQ4 (2b)",
    Quantize.PQ8_R2.value: "PQ8 (2b)",
    Quantize.PQ4_R1.value: "PQ4 (1b)",
    Quantize.PQ8_R1.value: "PQ8 (1b)",
    Quantize.RABITQ.value: "RaBitQ (1b)",
}
# Reference stage for the two relative-objective panels: the raw/unquantized
# cover, matching the "reduction relative to the ball cover" reading the
# panels had before per-quantize breakdown existed.
BASELINE_STAGE = Quantize.FP32.value

# tab10 has 10 qualitative colors, enough for the swept variants with room to
# spare; zip() stops at the shorter of the two, so re-adding a rung is safe.
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
    "fpr_hard",
    "fpr_easy",
    "fnr",
    "stage_time_s",
    "cum_time_s",
    "query_throughput_qps",
    "n_queries",
    "n_index_accesses",
    "index_accesses_saved_pct",
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
        # keep_default_na=False: stage/config strings are read verbatim; pandas'
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
    if len(available_tau) != 1:
        raise ValueError(
            f"Expected exactly one tau in the quantization data, found {len(available_tau)}: "
            f"{available_tau}. This sweep varies the quantizer at a fixed threshold; a "
            "multi-tau sweep belongs in plot_system_tau."
        )
    data_tau = float(available_tau[0])
    if tau is not None and tau != data_tau:
        raise ValueError(f"tau={tau} not present in data. The data's tau is {data_tau}.")

    return Config(tau=data_tau, dpi=dpi, emit_pdf=emit_pdf, outdir=outdir)


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

    # n_construct predates some existing results files (see HAS_RAW_STORAGE):
    # only add the raw-storage-percentage column when it's actually there, so
    # aggregate() keeps working against older data for every other column.
    if "n_construct" in sel.columns:
        n_construct = grouped["n_construct"].mean().reset_index(name="n_construct")
        agg = agg.merge(n_construct, on=["quantize", "k"])
        agg["Memory (% of Raw)"] = 100.0 * agg["memory_bytes"] / (agg["n_construct"] * D_DIM * 4)
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
    x_col: str = "Memory (MB)",
    xlabel: str = "Memory Usage (MB)",
) -> pd.DataFrame:
    agg = aggregate(df, value_col, cfg, use_tau=use_tau)
    if relative_to_baseline:
        agg = _relative_to_baseline(agg)

    apply_style()
    fig, ax = plt.subplots()

    for stage in STAGES:
        sub = agg[agg["quantize"] == stage].sort_values(x_col)
        if sub.empty:
            continue
        ax.plot(
            sub[x_col], sub["mean"],
            color=STAGE_COLORS[stage], marker=STAGE_MARKERS[stage], markersize=2.5,
            linewidth=0.8, zorder=2,
        )
        if (sub["ci95"] > 0).any():
            ax.errorbar(
                sub[x_col], sub["mean"], yerr=sub["ci95"],
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

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _style_axes(ax)
    if extra is not None:
        extra(ax)

    _fix_margins(fig)
    _save(fig, filename, cfg)
    return agg


# quantization.csv predates the n_construct column (see quantization_worker);
# a file without it can still render every other panel, just not the
# raw-storage-percentage variants -- checked once per figure-generation run
# rather than once per panel.
_RAW_PCT_X = dict(x_col="Memory (% of Raw)", xlabel="Memory (% of Raw)")


def _raw_pct_filename(filename: str) -> str:
    root, ext = os.path.splitext(filename)
    return f"{root}-raw-pct{ext}"


# Panel (a): geometric objective reduction

def panel_geometric_objective(df: pd.DataFrame, cfg: Config, raw_pct: bool = False) -> pd.DataFrame:
    filename = "quantization-geometric-objective.png"
    return _line_panel(
        df, "log10_J", r"$\log_{10} J(\mathcal{C})/J_{\mathrm{ball}}$",
        _raw_pct_filename(filename) if raw_pct else filename, cfg, use_tau=False,
        relative_to_baseline=True, **(_RAW_PCT_X if raw_pct else {}),
    )


# Panel (b): tau-expanded objective reduction

def panel_threshold_objective(df: pd.DataFrame, cfg: Config, raw_pct: bool = False) -> pd.DataFrame:
    def annotate(ax: plt.Axes) -> None:
        ax.text(
            0.97, 0.95, rf"$\tau = {cfg.tau:g}$",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.6,
        )

    filename = "quantization-threshold-expanded-objective.png"
    return _line_panel(
        df, "log10_J_tau", r"$\log_{10} J_\tau(\mathcal{C})/J_{\tau,\mathrm{ball}}$",
        _raw_pct_filename(filename) if raw_pct else filename, cfg, use_tau=True,
        relative_to_baseline=True, extra=annotate, **(_RAW_PCT_X if raw_pct else {}),
    )


# Panel (c): false positive rate

def panel_fpr(df: pd.DataFrame, cfg: Config, raw_pct: bool = False) -> pd.DataFrame:
    # Linear, not log: FPR is legitimately 0 for many points (a log axis
    # can't represent 0). y-min pinned to 0, top auto-scales.
    filename = "quantization-false-positive-rate.png"
    return _line_panel(
        df, "fpr", "FPR", _raw_pct_filename(filename) if raw_pct else filename, cfg,
        use_tau=True, y_bounds=(0, None), **(_RAW_PCT_X if raw_pct else {}),
    )


# Panel (c-hard)/(c-easy): the same pooled FPR split by query class, the same
# distinction plot_blocklist reports separately for the reason its own
# comment gives -- a pooled rate is dominated by whichever class the corpus
# happens to have more of. fpr_hard is held-out members of duplicate clusters
# NOT in the reference set (same tight local geometry as B, so this is what
# actually tests whether the cover localises B rather than just density);
# fpr_easy is singletons, i.e. undiscriminating background traffic.

def panel_fpr_hard(df: pd.DataFrame, cfg: Config, raw_pct: bool = False) -> pd.DataFrame:
    filename = "quantization-false-positive-rate-hard.png"
    return _line_panel(
        df, "fpr_hard", "FPR (hard negatives)", _raw_pct_filename(filename) if raw_pct else filename,
        cfg, use_tau=True, y_bounds=(0, None), **(_RAW_PCT_X if raw_pct else {}),
    )


def panel_fpr_easy(df: pd.DataFrame, cfg: Config, raw_pct: bool = False) -> pd.DataFrame:
    filename = "quantization-false-positive-rate-easy.png"
    return _line_panel(
        df, "fpr_easy", "FPR (easy negatives)", _raw_pct_filename(filename) if raw_pct else filename,
        cfg, use_tau=True, y_bounds=(0, None), **(_RAW_PCT_X if raw_pct else {}),
    )


# Panel (d): construction time

def panel_construction_time(df: pd.DataFrame, cfg: Config, raw_pct: bool = False) -> pd.DataFrame:
    # stage_time_s == cum_time_s in quantization_worker (single from-scratch
    # construction per row), so one line per quantize method is the whole picture.
    filename = "quantization-construction-time.png"
    return _line_panel(
        df, "cum_time_s", "Time (s)", _raw_pct_filename(filename) if raw_pct else filename, cfg,
        use_tau=False, log_y=True, **(_RAW_PCT_X if raw_pct else {}),
    )


# Panel (e): query throughput

def panel_query_throughput(df: pd.DataFrame, cfg: Config, raw_pct: bool = False) -> pd.DataFrame:
    """Batched-query throughput of filt.query(), per quantize variant.
    Measured at the sweep's single tau, which costs no generality: query cost
    doesn't depend on the threshold (every ball is checked regardless).
    Log-scaled since a quantizer's decode overhead (e.g. PQ) can separate
    lines by an order of magnitude from the raw/NONE cover."""
    filename = "quantization-query-throughput.png"
    return _line_panel(
        df, "query_throughput_qps", "Throughput (q/s)",
        _raw_pct_filename(filename) if raw_pct else filename, cfg, use_tau=False, log_y=True,
        **(_RAW_PCT_X if raw_pct else {}),
    )


# Panel (f): main-index accesses saved

def panel_index_accesses_saved(df: pd.DataFrame, cfg: Config, raw_pct: bool = False) -> pd.DataFrame:
    """Share of queries the filter answers on its own, so they never reach
    the main index. A rejected query is a saved access; a positive (true or
    false) falls through, so saved% == 100 - positive rate. This is the
    system-level reading of panel (c): FPR is the filter's error rate, this
    is what that error rate costs downstream once the workload's real
    duplicate rate is folded in.

    The ceiling is the non-duplicate share of the workload (95% at the
    calibrated tau -- a duplicate *should* reach the index), so a perfect
    filter sits there, not at 100%. Linear axis pinned to [0, 100]: the
    interesting range is the top of it, and a degenerate cover that accepts
    everything sits at 0."""
    def annotate(ax: plt.Axes) -> None:
        ax.text(
            # Top-left, not top-right like panel (b): this curve rises with
            # memory, so the right half is where the lines are.
            0.03, 0.95, rf"$\tau = {cfg.tau:g}$",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.6,
        )

    filename = "quantization-index-accesses-saved.png"
    return _line_panel(
        df, "index_accesses_saved_pct", "Index Accesses Saved (%)",
        _raw_pct_filename(filename) if raw_pct else filename, cfg, use_tau=True,
        y_bounds=(0, 100), extra=annotate, **(_RAW_PCT_X if raw_pct else {}),
    )


# Panel (g): ball count per quantize variant

def panel_n_balls(df: pd.DataFrame, cfg: Config, raw_pct: bool = False) -> pd.DataFrame:
    """Realized ball count per quantize variant across the memory-usage
    sweep: a quantizer with a smaller per-center code packs more balls into
    the same budget, so at a given x the lines are expected to separate,
    with PQ/SQ4 highest and None/FP16/BF16 lowest."""
    filename = "quantization-n-balls.png"
    return _line_panel(
        df, "n_balls", "Number of Balls", _raw_pct_filename(filename) if raw_pct else filename,
        cfg, use_tau=False, **(_RAW_PCT_X if raw_pct else {}),
    )


# Panel (h): radius inflation due to quantization

def panel_radius_inflation(df: pd.DataFrame, cfg: Config, raw_pct: bool = False) -> pd.DataFrame:
    """Mean +/- stddev of what encoding cost each ball's radius: the stored
    radius minus the one the unquantized cover measured, after
    BallCoverFilter._tight_radii re-measures against the decoded centers
    (see all_radius_inflation). Unlike other panels' seed-CI error bars,
    mean/std here are collapsed *across balls* within one job, so the error
    bars show ball-to-ball spread, not seed variance.

    Values go *negative*, including for Quantize.FP32, and the zero line is
    drawn to make that readable: _tight_radii's coverage pass drops any ball
    whose members another ball already holds down to a smaller radius than
    the construction measured, and on a lossless encoding there is no
    reconstruction error to spend those savings on. A rung sitting below
    zero is covering the same points in less volume than the raw cover, so
    read this as encoding cost net of that saving, not as error alone."""
    x_col, xlabel = (_RAW_PCT_X["x_col"], _RAW_PCT_X["xlabel"]) if raw_pct else ("Memory (MB)", "Memory Usage (MB)")
    mean_agg = aggregate(df, "radius_inflation_mean", cfg, use_tau=False)
    std_agg = aggregate(df, "radius_inflation_std", cfg, use_tau=False)
    std_agg = std_agg[["quantize", "k", "mean"]].rename(columns={"mean": "std"})
    agg = mean_agg.merge(std_agg, on=["quantize", "k"])

    apply_style()
    fig, ax = plt.subplots()

    for stage in STAGES:
        sub = agg[agg["quantize"] == stage].sort_values(x_col)
        if sub.empty:
            continue
        ax.plot(
            sub[x_col], sub["mean"],
            color=STAGE_COLORS[stage], marker=STAGE_MARKERS[stage], markersize=2.5,
            linewidth=0.8, zorder=2,
        )
        ax.errorbar(
            sub[x_col], sub["mean"], yerr=sub["std"],
            fmt="none", ecolor=STAGE_COLORS[stage],
            elinewidth=0.5, capsize=1.5, capthick=0.5, zorder=3,
        )

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(LogLocator(base=2.0))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=2.0))

    # Linear y, not log: since _tight_radii measures radii against the
    # decoded centers instead of bounding them, this quantity is signed, and
    # a log axis drops every non-positive point without saying so. The span
    # is a small multiple either way now that no ball is charged a
    # worst-case bound, so log buys nothing to offset that.
    ax.axhline(0.0, color="black", linewidth=0.5, zorder=1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Radius Inflation")
    _style_axes(ax)

    _fix_margins(fig)
    filename = "quantization-radius-inflation.png"
    _save(fig, _raw_pct_filename(filename) if raw_pct else filename, cfg)
    return agg


# Diagnostic panel (not part of the paper figure set): false negative rate

def panel_fnr(df: pd.DataFrame, cfg: Config, raw_pct: bool = False) -> pd.DataFrame:
    """Sanity-check panel, not for the paper: FNR should be exactly 0 for
    every row (already hard-asserted in quantization_worker and re-checked
    by _validate_schema), so this renders as a flat line at 0 under normal
    operation -- fixed [0, 1] y-range so a violation would be dramatically
    visible instead of auto-scaling into a misleading wobble."""
    filename = "quantization-false-negative-rate.png"
    return _line_panel(
        df, "fnr", "FNR", _raw_pct_filename(filename) if raw_pct else filename, cfg,
        use_tau=True, y_bounds=(-0.02, 1.02), **(_RAW_PCT_X if raw_pct else {}),
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
        frameon=False, fontsize=8.4,
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
              f"{'fpr':>12} {'saved (%)':>10} {'time (s)':>10} {'qps':>10} {'n_balls':>8}")
    print(header)
    print("-" * len(header))

    j_agg = _relative_to_baseline(aggregate(df, "log10_J", cfg, use_tau=False))
    fpr_agg = aggregate(df, "fpr", cfg, use_tau=True)
    saved_agg = aggregate(df, "index_accesses_saved_pct", cfg, use_tau=True)
    time_agg = aggregate(df, "cum_time_s", cfg, use_tau=False)
    qps_agg = aggregate(df, "query_throughput_qps", cfg, use_tau=False)
    balls_agg = aggregate(df, "n_balls", cfg, use_tau=False)

    for stage in STAGES:
        for _, row in j_agg[j_agg["quantize"] == stage].sort_values("k").iterrows():
            k = row["k"]
            this_fpr = fpr_agg[(fpr_agg["quantize"] == stage) & (fpr_agg["k"] == k)]["mean"].iloc[0]
            this_saved = saved_agg[(saved_agg["quantize"] == stage) & (saved_agg["k"] == k)]["mean"].iloc[0]
            this_time = time_agg[(time_agg["quantize"] == stage) & (time_agg["k"] == k)]["mean"].iloc[0]
            this_qps = qps_agg[(qps_agg["quantize"] == stage) & (qps_agg["k"] == k)]["mean"].iloc[0]
            this_balls = balls_agg[(balls_agg["quantize"] == stage) & (balls_agg["k"] == k)]["mean"].iloc[0]
            print(
                f"{STAGE_LABELS[stage]:>9} {row['Memory (MB)']:>12.3f} {row['mean']:>20.3f} "
                f"{this_fpr:>12.2e} {this_saved:>10.2f} {this_time:>10.3f} {this_qps:>10.1f} "
                f"{this_balls:>8.1f}"
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
    """Library entry point: renders the ten paper panels + one diagnostic
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
    panel_fpr_hard(df, cfg)
    panel_fpr_easy(df, cfg)
    panel_construction_time(df, cfg)
    panel_query_throughput(df, cfg)
    panel_index_accesses_saved(df, cfg)
    panel_n_balls(df, cfg)
    panel_radius_inflation(df, cfg)
    panel_fnr(df, cfg)

    # Same panels again, against memory usage as a percentage of raw
    # (uncompressed float32) vector storage instead of absolute MB -- needs
    # the n_construct column (see quantization_worker), absent from results
    # files predating it.
    if "n_construct" in df.columns:
        panel_geometric_objective(df, cfg, raw_pct=True)
        panel_threshold_objective(df, cfg, raw_pct=True)
        panel_fpr(df, cfg, raw_pct=True)
        panel_fpr_hard(df, cfg, raw_pct=True)
        panel_fpr_easy(df, cfg, raw_pct=True)
        panel_construction_time(df, cfg, raw_pct=True)
        panel_query_throughput(df, cfg, raw_pct=True)
        panel_index_accesses_saved(df, cfg, raw_pct=True)
        panel_n_balls(df, cfg, raw_pct=True)
        panel_radius_inflation(df, cfg, raw_pct=True)
        panel_fnr(df, cfg, raw_pct=True)
    else:
        print(
            "Note: input has no n_construct column, so the raw-storage-percentage panels "
            "were skipped. Rerun generate_quantization() (main.py) with a current runner.py "
            "to get them."
        )

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
    \includegraphics[width=\linewidth]{figures/quantization-false-positive-rate-hard.png}
    \caption{FPR, hard negatives}
\end{subfigure}
\begin{subfigure}[b]{0.24\linewidth}
    \centering
    \includegraphics[width=\linewidth]{figures/quantization-false-positive-rate-easy.png}
    \caption{FPR, easy negatives}
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
    \includegraphics[width=\linewidth]{figures/quantization-index-accesses-saved.png}
    \caption{Index accesses saved}
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
