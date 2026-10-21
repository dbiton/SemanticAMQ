"""CapCoverFilter's analogue of plot_quantization.py (see
main.py::generate_cap_quantization): same long-format-CSV-to-panels shape
(x-axis = memory usage, one line per quantize method, single dataset), swept
over CAP_QUANTIZATION_SWEEP (AUTO_CANDIDATES) instead of
BallCoverFilter's QUANTIZATION_SWEEP.

Two structural differences from plot_quantization.py, both because the cap
sweep has no FP32/unquantized member to single out:
- panel_geometric_objective/panel_threshold_objective plot log10_J/
  log10_J_tau directly (absolute), not relative to a baseline stage.
- No radius-inflation panel (no cap analogue -- inflation is a radius
  concept) and n_caps replaces n_balls.

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

from filters.cap_cover.cap_cover_filter import AUTO_CANDIDATES
from plots.plot_common import apply_style as _apply_style, collapse_duplicate_x, save_figure

# CONFIG

INPUT_PATH = "results/cap_quantization.csv"
OUTDIR = "plots/cap_quantization/"
TAU: Optional[float] = None
DPI = 400
EMIT_PDF = False

# Mirrors runner.CAP_QUANTIZATION_SWEEP (not imported, to avoid a dependency
# on runner.py's multiprocessing machinery, same reason plot_quantization.py
# hand-keeps its own QUANTIZE_SWEEP instead of importing runner's).
STAGES = [q.value for q in AUTO_CANDIDATES]
# Labelled by name only (unlike Ball-Cover's rate-annotated labels): the cap
# sweep mixes seeded-PQ families at several bit widths with no single
# "bits/dim" reading as clean as Ball-Cover's ladder -- see Quantize's own
# docstring in cap_cover_filter.py for each member's rate.
STAGE_LABELS = {s: s for s in STAGES}

_STAGE_PALETTE = plt.get_cmap("tab20").colors
STAGE_COLORS = dict(zip(STAGES, _STAGE_PALETTE))
_STAGE_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "p", "<", ">", "8", "H", "d"]
STAGE_MARKERS = dict(zip(STAGES, _STAGE_MARKERS))

D_DIM = 384


@dataclasses.dataclass
class Config:
    tau: float
    dpi: int
    emit_pdf: bool
    outdir: str


def apply_style() -> None:
    _apply_style(DPI)


def _style_axes(ax: plt.Axes) -> None:
    ax.grid(axis="both", which="major", linewidth=0.35, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.5)


_REQUIRED_COLUMNS = [
    "dataset", "k", "quantize", "stage", "tau", "log10_J", "log10_J_tau",
    "fpr", "fpr_withheld", "fpr_singleton", "fnr", "stage_time_s", "cum_time_s",
    "query_throughput_qps", "n_queries", "n_index_accesses", "index_accesses_saved_pct",
    "n_caps", "memory_bytes", "seed",
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
            f"Expected a single-dataset input (see generate_cap_quantization()), found "
            f"{n_datasets}: {sorted(df['dataset'].unique())}"
        )

    present = set(df["quantize"].unique())
    missing_combos = [s for s in STAGES if s not in present]
    if missing_combos:
        print(f"Note: quantize value(s) {missing_combos} have no rows (skipped during "
              f"generation, e.g. a dim%sub mismatch or InsufficientMemoryError) and won't "
              f"be plotted. Available: {sorted(present)}")


def load_data() -> pd.DataFrame:
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH!r}. Run generate_cap_quantization() (main.py) "
            "first, or point INPUT_PATH at a real results file."
        )
    df = pd.read_parquet(INPUT_PATH) if INPUT_PATH.endswith(".parquet") else pd.read_csv(INPUT_PATH, keep_default_na=False)
    _validate_schema(df)
    return df


def resolve_config(df: pd.DataFrame, tau: Optional[float] = TAU, dpi: int = DPI,
                   emit_pdf: bool = EMIT_PDF, outdir: str = OUTDIR) -> Config:
    available_tau = sorted(df["tau"].unique())
    if len(available_tau) != 1:
        raise ValueError(
            f"Expected exactly one tau in the quantization data, found {len(available_tau)}: "
            f"{available_tau}."
        )
    data_tau = float(available_tau[0])
    if tau is not None and tau != data_tau:
        raise ValueError(f"tau={tau} not present in data. The data's tau is {data_tau}.")
    return Config(tau=data_tau, dpi=dpi, emit_pdf=emit_pdf, outdir=outdir)


def aggregate(df: pd.DataFrame, value_col: str, cfg: Config, use_tau: bool = False,
             stages: list = STAGES) -> pd.DataFrame:
    sel = df[df["quantize"].isin(stages)]
    if use_tau:
        sel = sel[sel["tau"] == cfg.tau]

    bad_raw = sel[~np.isfinite(sel[value_col])]
    if len(bad_raw):
        cols = [c for c in ("k", "quantize", "tau", "stage", "seed", value_col) if c in bad_raw.columns]
        raise ValueError(
            f"Non-finite {value_col!r} values"
            + (f" at tau={cfg.tau}" if use_tau else "") + f":\n{bad_raw[cols].to_string(index=False)}\n"
            "This usually means k >= the construct-set size, or every cap degenerated -- "
            "lower k or shrink the construct-set-relative k sweep in the data-generation step."
        )

    grouped = sel.groupby(["quantize", "k"])
    agg = grouped[value_col].agg(mean="mean", sem="sem", n="count").reset_index()
    agg["ci95"] = np.where(agg["n"] > 1, 1.96 * agg["sem"].fillna(0.0), 0.0)
    mem = grouped["memory_bytes"].mean().reset_index(name="memory_bytes")
    agg = agg.merge(mem, on=["quantize", "k"])
    agg["Memory (MB)"] = agg["memory_bytes"] / (1024 * 1024)

    if "n_construct" in sel.columns:
        n_construct = grouped["n_construct"].mean().reset_index(name="n_construct")
        agg = agg.merge(n_construct, on=["quantize", "k"])
        agg["Memory (% of Raw)"] = 100.0 * agg["memory_bytes"] / (agg["n_construct"] * D_DIM * 4)
    return agg


def _line_panel(df: pd.DataFrame, value_col: str, ylabel: str, filename: str, cfg: Config,
                use_tau: bool = False, log_y: bool = False, y_bounds: Optional[tuple] = None,
                extra=None, x_col: str = "Memory (MB)", xlabel: str = "Memory Usage (MB)") -> pd.DataFrame:
    agg = aggregate(df, value_col, cfg, use_tau=use_tau)

    apply_style()
    fig, ax = plt.subplots()

    for stage in STAGES:
        sub = agg[agg["quantize"] == stage].sort_values(x_col)
        sub = collapse_duplicate_x(sub, x_col)
        if sub.empty:
            continue
        ax.plot(sub[x_col], sub["mean"], color=STAGE_COLORS[stage], marker=STAGE_MARKERS[stage],
               markersize=2.5, linewidth=0.8, zorder=2)
        if (sub["ci95"] > 0).any():
            ax.errorbar(sub[x_col], sub["mean"], yerr=sub["ci95"], fmt="none",
                       ecolor=STAGE_COLORS[stage], elinewidth=0.5, capsize=1.5, capthick=0.5, zorder=3)

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(LogLocator(base=2.0))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=2.0))

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

    fig.subplots_adjust(left=0.30, right=0.97, top=0.92, bottom=0.24)
    save_figure(fig, filename, cfg.outdir, cfg.dpi, cfg.emit_pdf, tight=False)
    return agg


_RAW_PCT_X = dict(x_col="Memory (% of Raw)", xlabel="Memory (% of Raw)")


def _raw_pct_filename(filename: str) -> str:
    root, ext = os.path.splitext(filename)
    return f"{root}-raw-pct{ext}"


def panel_geometric_objective(df: pd.DataFrame, cfg: Config, raw_pct: bool = False) -> pd.DataFrame:
    filename = "cap-quantization-geometric-objective.png"
    return _line_panel(df, "log10_J", r"$\log_{10} J(\mathcal{C})$",
                       _raw_pct_filename(filename) if raw_pct else filename, cfg, use_tau=False,
                       **(_RAW_PCT_X if raw_pct else {}))


def panel_threshold_objective(df: pd.DataFrame, cfg: Config, raw_pct: bool = False) -> pd.DataFrame:
    def annotate(ax: plt.Axes) -> None:
        ax.text(0.97, 0.95, rf"$\tau = {cfg.tau:g}$", transform=ax.transAxes, ha="right", va="top", fontsize=6.6)

    filename = "cap-quantization-threshold-expanded-objective.png"
    return _line_panel(df, "log10_J_tau", r"$\log_{10} J_\tau(\mathcal{C})$",
                       _raw_pct_filename(filename) if raw_pct else filename, cfg, use_tau=True,
                       extra=annotate, **(_RAW_PCT_X if raw_pct else {}))


def panel_fpr(df: pd.DataFrame, cfg: Config, raw_pct: bool = False) -> pd.DataFrame:
    filename = "cap-quantization-false-positive-rate.png"
    return _line_panel(df, "fpr", "FPR", _raw_pct_filename(filename) if raw_pct else filename, cfg,
                       use_tau=True, y_bounds=(0, None), **(_RAW_PCT_X if raw_pct else {}))


_FPR_SOURCE_PANELS = (
    ("fpr_heldout", "FPR (held-out negatives)", "cap-quantization-false-positive-rate-heldout.png"),
    ("fpr_withheld", "FPR (withheld negatives)", "cap-quantization-false-positive-rate-withheld.png"),
    ("fpr_singleton", "FPR (singleton negatives)", "cap-quantization-false-positive-rate-singleton.png"),
)


def panels_fpr_by_source(df: pd.DataFrame, cfg: Config, raw_pct: bool = False) -> None:
    for col, label, filename in _FPR_SOURCE_PANELS:
        if col not in df.columns:
            continue
        _line_panel(df, col, label, _raw_pct_filename(filename) if raw_pct else filename,
                   cfg, use_tau=True, y_bounds=(0, None), **(_RAW_PCT_X if raw_pct else {}))


def panel_construction_time(df: pd.DataFrame, cfg: Config, raw_pct: bool = False) -> pd.DataFrame:
    filename = "cap-quantization-construction-time.png"
    return _line_panel(df, "cum_time_s", "Time (s)", _raw_pct_filename(filename) if raw_pct else filename, cfg,
                       use_tau=False, log_y=True, **(_RAW_PCT_X if raw_pct else {}))


def panel_query_throughput(df: pd.DataFrame, cfg: Config, raw_pct: bool = False) -> pd.DataFrame:
    filename = "cap-quantization-query-throughput.png"
    return _line_panel(df, "query_throughput_qps", "Throughput (q/s)",
                       _raw_pct_filename(filename) if raw_pct else filename, cfg, use_tau=False, log_y=True,
                       **(_RAW_PCT_X if raw_pct else {}))


def panel_index_accesses_saved(df: pd.DataFrame, cfg: Config, raw_pct: bool = False) -> pd.DataFrame:
    ceiling = (float(df.loc[df["tau"] == cfg.tau, "neg_share_pct"].mean())
               if "neg_share_pct" in df.columns else None)

    def annotate(ax: plt.Axes) -> None:
        if ceiling is not None:
            ax.axhline(ceiling, color="0.4", linestyle="--", linewidth=0.5, zorder=1)
        ax.text(0.03, 0.95, rf"$\tau = {cfg.tau:g}$", transform=ax.transAxes, ha="left", va="top", fontsize=6.6)

    filename = "cap-quantization-index-accesses-saved.png"
    return _line_panel(df, "index_accesses_saved_pct", "Index Accesses Saved (%)",
                       _raw_pct_filename(filename) if raw_pct else filename, cfg, use_tau=True,
                       y_bounds=(0, 100), extra=annotate, **(_RAW_PCT_X if raw_pct else {}))


def panel_n_caps(df: pd.DataFrame, cfg: Config, raw_pct: bool = False) -> pd.DataFrame:
    filename = "cap-quantization-n-caps.png"
    return _line_panel(df, "n_caps", "Number of Caps", _raw_pct_filename(filename) if raw_pct else filename,
                       cfg, use_tau=False, **(_RAW_PCT_X if raw_pct else {}))


def panel_fnr(df: pd.DataFrame, cfg: Config, raw_pct: bool = False) -> pd.DataFrame:
    filename = "cap-quantization-false-negative-rate.png"
    return _line_panel(df, "fnr", "FNR", _raw_pct_filename(filename) if raw_pct else filename, cfg,
                       use_tau=True, y_bounds=(-0.02, 1.02), **(_RAW_PCT_X if raw_pct else {}))


def save_legend(df: pd.DataFrame, cfg: Config) -> None:
    present = set(df["quantize"].unique())
    stages = [s for s in STAGES if s in present]

    apply_style()
    fig, ax = plt.subplots(figsize=(5.6, 0.32))
    ax.axis("off")

    handles = [plt.Line2D([0], [0], color=STAGE_COLORS[s], marker=STAGE_MARKERS[s], markersize=3, linewidth=1.0)
              for s in stages]
    labels = [STAGE_LABELS[s] for s in stages]

    fig.legend(handles, labels, loc="center", ncol=min(5, len(stages)), frameon=False, fontsize=8.4)
    fig.patch.set_alpha(0.0)

    path = os.path.join(cfg.outdir, "cap-quantization-legend.png")
    os.makedirs(cfg.outdir, exist_ok=True)
    fig.savefig(path, dpi=cfg.dpi, bbox_inches="tight", pad_inches=0.01, transparent=True)
    if cfg.emit_pdf:
        fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight", pad_inches=0.01, transparent=True)
    plt.close(fig)
    print(f"Wrote {path}")


def print_summary(df: pd.DataFrame, cfg: Config) -> None:
    print(f"\n--- Cap quantization summary (tau={cfg.tau:g}) ---")
    header = (f"{'quantize':>9} {'memory (MB)':>12} {'log10 J':>12} "
              f"{'fpr':>12} {'saved (%)':>10} {'time (s)':>10} {'qps':>10} {'n_caps':>8}")
    print(header)
    print("-" * len(header))

    j_agg = aggregate(df, "log10_J", cfg, use_tau=False)
    fpr_agg = aggregate(df, "fpr", cfg, use_tau=True)
    saved_agg = aggregate(df, "index_accesses_saved_pct", cfg, use_tau=True)
    time_agg = aggregate(df, "cum_time_s", cfg, use_tau=False)
    qps_agg = aggregate(df, "query_throughput_qps", cfg, use_tau=False)
    caps_agg = aggregate(df, "n_caps", cfg, use_tau=False)

    for stage in STAGES:
        for _, row in j_agg[j_agg["quantize"] == stage].sort_values("k").iterrows():
            k = row["k"]
            this_fpr = fpr_agg[(fpr_agg["quantize"] == stage) & (fpr_agg["k"] == k)]["mean"].iloc[0]
            this_saved = saved_agg[(saved_agg["quantize"] == stage) & (saved_agg["k"] == k)]["mean"].iloc[0]
            this_time = time_agg[(time_agg["quantize"] == stage) & (time_agg["k"] == k)]["mean"].iloc[0]
            this_qps = qps_agg[(qps_agg["quantize"] == stage) & (qps_agg["k"] == k)]["mean"].iloc[0]
            this_caps = caps_agg[(caps_agg["quantize"] == stage) & (caps_agg["k"] == k)]["mean"].iloc[0]
            print(f"{STAGE_LABELS[stage]:>9} {row['Memory (MB)']:>12.3f} {row['mean']:>12.3f} "
                 f"{this_fpr:>12.2e} {this_saved:>10.2f} {this_time:>10.3f} {this_qps:>10.1f} "
                 f"{this_caps:>8.1f}")
    print()


def generate_quantization_figures(df: Optional[pd.DataFrame] = None, outdir: str = OUTDIR,
                                  tau: Optional[float] = TAU, dpi: int = DPI, emit_pdf: bool = EMIT_PDF) -> None:
    if df is None:
        df = load_data()
    else:
        _validate_schema(df)

    cfg = resolve_config(df, tau=tau, dpi=dpi, emit_pdf=emit_pdf, outdir=outdir)

    panel_geometric_objective(df, cfg)
    panel_threshold_objective(df, cfg)
    panel_fpr(df, cfg)
    panels_fpr_by_source(df, cfg)
    panel_construction_time(df, cfg)
    panel_query_throughput(df, cfg)
    panel_index_accesses_saved(df, cfg)
    panel_n_caps(df, cfg)
    panel_fnr(df, cfg)

    if "n_construct" in df.columns:
        panel_geometric_objective(df, cfg, raw_pct=True)
        panel_threshold_objective(df, cfg, raw_pct=True)
        panel_fpr(df, cfg, raw_pct=True)
        panels_fpr_by_source(df, cfg, raw_pct=True)
        panel_construction_time(df, cfg, raw_pct=True)
        panel_query_throughput(df, cfg, raw_pct=True)
        panel_index_accesses_saved(df, cfg, raw_pct=True)
        panel_n_caps(df, cfg, raw_pct=True)
        panel_fnr(df, cfg, raw_pct=True)

    save_legend(df, cfg)
    print_summary(df, cfg)


def main() -> None:
    generate_quantization_figures()


if __name__ == "__main__":
    main()
