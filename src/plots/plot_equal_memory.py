"""Generates the equal-memory-comparison figures (+ shared legend). Reads a
long-format results file (one row per filter config / memory budget), and
writes PNG (optionally PDF) panels matching plot_quantization.py's styling.

Edit the CONFIG block below to change parameters -- no CLI flags."""

from __future__ import annotations

import dataclasses
import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter

from plots.plot_common import apply_style as _apply_style
from plots.plot_common import collapse_duplicate_x

# CONFIG

INPUT_PATH = "results/equal_memory.csv"
OUTDIR = "plots/equal_memory/"
DPI = 400
EMIT_PDF = False

def apply_style() -> None:
    _apply_style(DPI)


# Embedding dimensionality: fixed at 384 across every blocklist-workload
# evaluation (see main.generate_equal_memory's own dim=384).
_EMBEDDING_DIM = 384

# Display overrides for recognized filter names; anything else falls back to
# name[:-len("Filter")] in _label_for.
_KNOWN_LABELS = {
    "HNSWFilter": "HNSW",
    "IVFFilter": "IVF",
    "DSHFilter": "DSH",
    "LSBFFilter": "LSBF",
    "DSBFFilter": "DSBF",
    "CoveringLSHFallbackFilter": "Covering LSH",
    "PacukLSHFilter": "Pacuk LSH",
    "SankowskiWygockiFilter": "Sankowski-Wygocki",
    "VAPlusFilter": "VA+-file",
    "IVFRadiiFilter": "IVF radii",
}
_KNOWN_MARKERS = {
    "HNSWFilter": "P",
    "IVFFilter": "s",
}
# CapCoverFilter (Config strings like "CapCoverFilter" or, if a quantize
# variant is submitted, "CapCoverFilter (TQ4)" -- see
# runner.equal_memory_worker) draws from the cold palette; every other method draws from the hot one. Two colorblind-
# distinguishable-within-family sets, so a method's family reads from color
# alone and its identity within that family still reads from hue. Each list
# has more entries than there are current consumers (1 cover
# method, 7 other methods) for headroom -- see _derive_style_maps for what
# happens if a family ever does run out (loops back rather than crashing).
_COLD_PALETTE_COLORS = ["#023E8A", "#0077B6", "#00B4D8", "#48CAE4", "#3A0CA3", "#7209B7", "#4CC9F0", "#5A189A"]
_HOT_PALETTE_COLORS = ["#9D0208", "#E85D04", "#FAA307", "#DC2F02", "#6A040F", "#F48C06", "#FFBA08", "#D00000"]
_PALETTE_MARKERS = ["o", "s", "^", "D", "v", "P", "*", "X", "h", "p", "<", ">", "8", "H"]


def _label_for(name: str) -> str:
    """`name` is a full Config string, e.g. "CapCoverFilter (TQ4)" for one
    of a filter's quantize variants (see runner.equal_memory_worker) --
    translate the base filter name and reattach the variant suffix verbatim,
    rather than falling through to the raw, un-shortened string."""
    base, sep, suffix = name.partition(" (")
    label = _KNOWN_LABELS.get(base, base[:-len("Filter")] if base.endswith("Filter") else base)
    return f"{label}{sep}{suffix}" if sep else label


def _is_cover_filter(name: str) -> bool:
    return name.partition(" (")[0] == "CapCoverFilter"


def _derive_style_maps(configs: list) -> tuple:
    labels, colors, markers = {}, {}, {}
    cold = iter(_COLD_PALETTE_COLORS)
    hot = iter(_HOT_PALETTE_COLORS)
    palette_markers = iter(m for m in _PALETTE_MARKERS if m not in set(_KNOWN_MARKERS.values()))

    for name in configs:
        labels[name] = _label_for(name)
        if _is_cover_filter(name):
            colors[name] = next(cold, _COLD_PALETTE_COLORS[0])
        else:
            colors[name] = next(hot, _HOT_PALETTE_COLORS[0])
        markers[name] = _KNOWN_MARKERS.get(name) or next(palette_markers, _PALETTE_MARKERS[0])

    return labels, colors, markers


def _derive_configs(df: pd.DataFrame) -> list:
    """Ordered, de-duplicated filter base names present in df["Config"]."""
    base_names = df["Config"].str.split(" + ", regex=False).str[0]
    return list(dict.fromkeys(base_names))


@dataclasses.dataclass
class Config:
    dpi: int
    emit_pdf: bool
    outdir: str
    configs: list
    labels: dict
    colors: dict
    markers: dict


# Data loading

_REQUIRED_COLUMNS = ["Config", "Memory Budget", "FPR", "FNR", "Latency (ms)", "Throughput (qps)"]


def _validate_schema(df: pd.DataFrame) -> None:
    missing_cols = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Input is missing required columns: {missing_cols}. "
            f"Present columns: {sorted(df.columns)}"
        )

    if df["Config"].isna().all() or (df["Config"].str.len() == 0).all():
        raise ValueError("Input has a 'Config' column but no filter configs in it.")


def check_budget_compliance(df: pd.DataFrame) -> None:
    """Every plotted point sits at the budget it was *given*, so a filter
    that overran its budget would be drawn as if it had been cheap. Filters
    now refuse those budgets at construct() time (runner.py records them as
    infeasible instead), so a surviving overrun means a filter's
    get_bytesize() disagrees with its own budget check -- worth shouting
    about rather than silently plotting."""
    if "Memory Usage" not in df.columns:
        return  # results predating the Memory Usage column
    usage = pd.to_numeric(df["Memory Usage"], errors="coerce")
    budget = pd.to_numeric(df["Memory Budget"], errors="coerce")
    over = df[usage > budget]
    if not over.empty:
        worst = (usage / budget).max()
        print(
            f"WARNING: {len(over)} row(s) report Memory Usage above their Memory Budget "
            f"(worst {worst:.2f}x); these are plotted at a budget they did not respect. "
            f"Affected filters: {sorted(over['Config'].unique())}"
        )


def drop_over_budget(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows the runner flagged 'Over Budget' (index exceeded its byte
    budget, so the row is unsound). Tolerates a missing column (old CSVs) and
    'True'/'False' strings from CSV."""
    if "Over Budget" not in df.columns:
        return df
    col = df["Over Budget"]
    over = col.map(lambda v: str(v).strip().lower() in ("true", "1", "1.0"))
    if over.any():
        logging.getLogger(__name__).warning(
            "Dropping %d over-budget row(s) from the equal-memory plots.", int(over.sum())
        )
    return df[~over].copy()


def drop_infeasible(df: pd.DataFrame) -> pd.DataFrame:
    """Budgets a filter could not build at all carry NaN metrics. Dropping
    them leaves a gap in that filter's line, which is the honest rendering:
    it has no FPR at that budget, as opposed to a bad one."""
    if "Infeasible" not in df.columns:
        return df
    infeasible = df["Infeasible"].fillna(False).astype(bool)
    if infeasible.any():
        for name, count in df[infeasible]["Config"].value_counts().items():
            print(f"Note: {name} was infeasible at {count} budget(s); those points are omitted.")
    return df[~infeasible].copy()


def load_data() -> pd.DataFrame:
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH!r}. Run generate_equal_memory() "
            "(main.py) first, or point INPUT_PATH at a real results file."
        )
    if INPUT_PATH.endswith(".parquet"):
        df = pd.read_parquet(INPUT_PATH)
    else:
        df = pd.read_csv(INPUT_PATH)

    _validate_schema(df)
    return df


# Aggregation

def aggregate(df: pd.DataFrame, value_col: str, dim: int = _EMBEDDING_DIM) -> pd.DataFrame:
    """Mean + 95% CI of `value_col` over any repeated runs, per (filter,
    memory budget); a single run per (filter, budget) gives ci95=0.

    Also derives "Memory Usage (% of Raw)" when the input has both a
    "Construct Size" and a "Memory Usage" column: the percentage a filter
    actually costs against the alternative of storing its construct set as a
    plain list of raw float32 vectors (Construct Size * dim * 4 bytes) --
    not the budget it was merely *given*. A filter like IVFPQ can saturate
    well under its assigned budget (see print_summary's "used/budget"
    column), so budget-as-%-of-raw would put every filter at the same x
    position at a given sweep step regardless of what it actually spent,
    collapsing the very comparison this panel exists to show.

    Construct Size is averaged within each (filter, budget) group -- the
    same "aggregate, then combine" pattern plot_quantization.aggregate() and
    plot_budget._raw_storage_pct_x() use for their own memory columns --
    purely so a pooled group's raw-storage denominator is computed from
    real per-row data rather than an arbitrary single row's value; in this
    sweep every corpus is in fact trimmed to the same reference-set size
    (see main._fold_selections), so the average is exact, not approximate."""
    work = df.copy()
    work["filter"] = work["Config"].str.split(" + ", regex=False).str[0]
    work["Memory Budget (MB)"] = work["Memory Budget"] / (1024 * 1024)

    grouped = work.groupby(["filter", "Memory Budget (MB)"])
    agg = grouped[value_col].agg(mean="mean", sem="sem", n="count").reset_index()
    agg["ci95"] = np.where(agg["n"] > 1, 1.96 * agg["sem"].fillna(0.0), 0.0)

    if "Construct Size" in work.columns and "Memory Usage" in work.columns:
        construct_size = grouped["Construct Size"].mean().reset_index(name="Construct Size")
        usage = grouped["Memory Usage"].mean().reset_index(name="Memory Usage (mean)")
        agg = agg.merge(construct_size, on=["filter", "Memory Budget (MB)"])
        agg = agg.merge(usage, on=["filter", "Memory Budget (MB)"])
        raw_bytes = agg["Construct Size"] * dim * 4
        agg["Memory Usage (% of Raw)"] = 100.0 * agg["Memory Usage (mean)"] / raw_bytes
    return agg


# Line panels: x = memory budget (or, in the "_raw_pct" panels, that budget
# as a percentage of raw vector storage), one line per filter config

# Raw-pct panels' x-axis view: a filter that spends more than raw storage
# would cost is a real, plottable outcome (see aggregate()'s docstring on
# DSHFilter), and the panel should show it rather than crop it off -- only
# the left edge (below 1% of raw is never informative) is fixed. The right
# edge is left to matplotlib's autoscale, so it always reaches whatever the
# worst filter in this particular run actually spent, rather than a
# hardcoded ceiling that could either clip real data or leave dead space.
_RAW_PCT_X_MIN = 1.0


def _line_panel(
    df: pd.DataFrame,
    value_col: str,
    ylabel: str,
    filename: str,
    cfg: Config,
    log_y: bool = False,
    y_bounds: Optional[tuple] = None,
    x_min: Optional[float] = None,
    x_col: str = "Memory Budget (MB)",
    xlabel: str = "Memory Budget (MB)",
) -> pd.DataFrame:
    agg = aggregate(df, value_col)

    apply_style()
    fig, ax = plt.subplots()

    for name in cfg.configs:
        sub = agg[agg["filter"] == name].sort_values(x_col)
        sub = collapse_duplicate_x(sub, x_col)
        if sub.empty:
            continue
        has_ci = (sub["ci95"] > 0).any()
        ax.plot(
            sub[x_col], sub["mean"],
            color=cfg.colors[name],
            marker=cfg.markers[name], markersize=2.5,
            linewidth=0.8,
            zorder=2,
        )
        if has_ci:
            ax.errorbar(
                sub[x_col], sub["mean"], yerr=sub["ci95"],
                fmt="none", ecolor=cfg.colors[name],
                elinewidth=0.5, capsize=1.5, capthick=0.5,
                zorder=3,
            )

    if x_min is not None:
        ax.set_xlim(left=x_min)

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
    ax.grid(axis="both", which="major", linewidth=0.35, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.5)

    _save(fig, filename, cfg)
    return agg


def panel_fpr(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    return _line_panel(df, "FPR", "False Positive Rate", "equal-memory-fpr.png", cfg, y_bounds=(-0.05, 1.05))


def panel_fpr_raw_pct(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    return _line_panel(df, "FPR", "False Positive Rate", "equal-memory-fpr-raw-pct.png", cfg,
                       y_bounds=(-0.05, 1.05), x_min=_RAW_PCT_X_MIN,
                       x_col="Memory Usage (% of Raw)", xlabel="Memory Usage (% of Raw)")


def panel_fnr(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    return _line_panel(df, "FNR", "False Negative Rate", "equal-memory-fnr.png", cfg, y_bounds=(-0.05, 1.05))


def panel_fnr_raw_pct(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    return _line_panel(df, "FNR", "False Negative Rate", "equal-memory-fnr-raw-pct.png", cfg,
                       y_bounds=(-0.05, 1.05), x_min=_RAW_PCT_X_MIN,
                       x_col="Memory Usage (% of Raw)", xlabel="Memory Usage (% of Raw)")


def panel_latency(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """"Latency (ms)" is construction latency here (the one-time cost of
    building each filter at a given memory budget), not per-query serving
    latency -- see runner.py::equal_memory_worker."""
    return _line_panel(df, "Latency (ms)", "Construction Latency (ms)", "equal-memory-latency.png", cfg, log_y=True)


def panel_latency_raw_pct(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    return _line_panel(df, "Latency (ms)", "Construction Latency (ms)",
                       "equal-memory-latency-raw-pct.png", cfg, log_y=True,
                       x_min=_RAW_PCT_X_MIN,
                       x_col="Memory Usage (% of Raw)", xlabel="Memory Usage (% of Raw)")


def panel_throughput(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    return _line_panel(
        df, "Throughput (qps)", "Throughput (queries/s)", "equal-memory-throughput.png", cfg, log_y=True
    )


def panel_throughput_raw_pct(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    return _line_panel(
        df, "Throughput (qps)", "Throughput (queries/s)", "equal-memory-throughput-raw-pct.png", cfg,
        log_y=True, x_min=_RAW_PCT_X_MIN,
        x_col="Memory Usage (% of Raw)", xlabel="Memory Usage (% of Raw)",
    )


# Shared legend

def save_legend(cfg: Config) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(5.6, 0.28))
    ax.axis("off")

    handles = [
        plt.Line2D(
            [0], [0],
            color=cfg.colors[name], marker=cfg.markers[name],
            markersize=3, linewidth=1.0,
        )
        for name in cfg.configs
    ]
    labels = [cfg.labels[name] for name in cfg.configs]

    fig.legend(
        handles, labels,
        loc="center", ncol=len(cfg.configs),
        frameon=False, fontsize=8.4,
    )
    fig.patch.set_alpha(0.0)

    path = os.path.join(cfg.outdir, "equal-memory-legend.png")
    fig.savefig(path, dpi=cfg.dpi, bbox_inches="tight", pad_inches=0.01, transparent=True)
    if cfg.emit_pdf:
        fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight", pad_inches=0.01, transparent=True)
    plt.close(fig)
    print(f"Wrote {path}")


# Save helper

def _save(fig: plt.Figure, filename: str, cfg: Config) -> None:
    os.makedirs(cfg.outdir, exist_ok=True)
    path = os.path.join(cfg.outdir, filename)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.01, dpi=cfg.dpi)
    if cfg.emit_pdf:
        fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"Wrote {path}")


# Summary table

def print_summary(df: pd.DataFrame, cfg: Config) -> None:
    print("\n--- Equal-memory comparison summary ---")
    # "used" and "used/budget" are here because the x-axis is the budget a
    # filter was *given*, not what it spent: IVFPQ saturates at dim bytes
    # per vector and then ignores every further MB, so a flat tail in its
    # curve is a filter that stopped growing, not one that stopped
    # improving. Without this column that distinction is invisible.
    header = (
        f"{'filter':<24} {'budget (MB)':>12} {'used (MB)':>10} {'used/budget':>12} "
        f"{'FPR':>8} {'FNR':>8} {'construct (ms)':>15} {'QPS':>12}"
    )
    print(header)
    print("-" * len(header))

    fpr_agg = aggregate(df, "FPR")
    fnr_agg = aggregate(df, "FNR")
    lat_agg = aggregate(df, "Latency (ms)")
    qps_agg = aggregate(df, "Throughput (qps)")
    use_agg = aggregate(df, "Memory Usage") if "Memory Usage" in df.columns else None

    for name in cfg.configs:
        budgets = sorted(fpr_agg[fpr_agg["filter"] == name]["Memory Budget (MB)"].unique())
        for budget in budgets:
            def _mean(agg: Optional[pd.DataFrame]) -> float:
                if agg is None:
                    return float("nan")
                row = agg[(agg["filter"] == name) & (agg["Memory Budget (MB)"] == budget)]
                return float(row["mean"].iloc[0]) if len(row) else float("nan")

            used_mb = _mean(use_agg) / (1024 * 1024)
            print(
                f"{cfg.labels[name]:<24} {budget:>12.3f} {used_mb:>10.3f} "
                f"{used_mb / budget if budget else float('nan'):>12.3f} "
                f"{_mean(fpr_agg):>8.3f} {_mean(fnr_agg):>8.3f} "
                f"{_mean(lat_agg):>15.4f} {_mean(qps_agg):>12.1f}"
            )
    print()


# Main

def generate_equal_memory_figures(
    df: Optional[pd.DataFrame] = None,
    outdir: str = OUTDIR,
    dpi: int = DPI,
    emit_pdf: bool = EMIT_PDF,
) -> None:
    """Library entry point: renders the four equal-memory-comparison panels
    (+ legend) from an in-memory DataFrame, or from load_data() (reads
    INPUT_PATH) if `df` is omitted."""
    if df is None:
        df = load_data()
    else:
        _validate_schema(df)

    check_budget_compliance(df)
    df = drop_over_budget(df)
    df = drop_infeasible(df)
    if df.empty:
        raise ValueError("No feasible rows to plot: every result row is marked infeasible.")

    configs = _derive_configs(df)
    labels, colors, markers = _derive_style_maps(configs)
    cfg = Config(dpi=dpi, emit_pdf=emit_pdf, outdir=outdir, configs=configs, labels=labels, colors=colors, markers=markers)

    panel_fpr(df, cfg)
    panel_fnr(df, cfg)
    panel_latency(df, cfg)
    panel_throughput(df, cfg)

    # Same four panels again, against actual memory usage as a percentage of
    # raw (uncompressed float32) vector storage instead of absolute MB --
    # needs the "Construct Size" and "Memory Usage" columns (see
    # equal_memory_worker), absent from results files predating them.
    if "Construct Size" in df.columns and "Memory Usage" in df.columns:
        panel_fpr_raw_pct(df, cfg)
        panel_fnr_raw_pct(df, cfg)
        panel_latency_raw_pct(df, cfg)
        panel_throughput_raw_pct(df, cfg)
    else:
        print(
            "Note: input is missing 'Construct Size' and/or 'Memory Usage', so the "
            "raw-storage-percentage panels were skipped. Rerun generate_equal_memory() "
            "(main.py) with a current runner.py to get them."
        )

    save_legend(cfg)

    print_summary(df, cfg)

    snippet = r"""
\begin{figure*}[]
  \centering

  \begin{subfigure}{0.6\linewidth}
    \includegraphics[width=\linewidth]{figures/equal-memory-legend.png}
  \end{subfigure}\hfill

  \vspace{0.1em}

  % Row of 4 graphs
  \begin{subfigure}{0.24\linewidth}
    \includegraphics[width=\linewidth]{figures/equal-memory-fpr.png}
    \caption{False positive rate}
    \label{fig:equal-memory-fpr}
  \end{subfigure}\hfill
  \begin{subfigure}{0.24\linewidth}
    \includegraphics[width=\linewidth]{figures/equal-memory-fnr.png}
    \caption{False negative rate}
    \label{fig:equal-memory-fnr}
  \end{subfigure}\hfill
  \begin{subfigure}{0.24\linewidth}
    \includegraphics[width=\linewidth]{figures/equal-memory-latency.png}
    \caption{Construction latency}
    \label{fig:equal-memory-latency}
  \end{subfigure}\hfill
  \begin{subfigure}{0.24\linewidth}
    \includegraphics[width=\linewidth]{figures/equal-memory-throughput.png}
    \caption{Query throughput}
    \label{fig:equal-memory-throughput}
  \end{subfigure}\hfill

  \vspace{0.5em}
  \caption{Equal-memory comparison: every method is given the same memory budget $M$, swept across a range of budgets, with the best valid configuration of each chosen on a held-out calibration workload.}
  \label{fig:equal-memory}
\end{figure*}
""".strip("\n")
    print("--- figure snippet ---")
    print(snippet)


def main() -> None:
    generate_equal_memory_figures()


if __name__ == "__main__":
    main()
