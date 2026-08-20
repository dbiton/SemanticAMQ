"""Streaming-evaluation figures: 5 insertion methods (M1-M5, see
filters.streaming_insertion) that differ in decision rule and cost class,
compared against two references (exhaustive Algorithm 2 and from-scratch
static rebuild) that appear only as dashed lines, never as legend entries.

Reads results/streaming.csv (one row per method/checkpoint, tagged by
`sweep` in {"position","k_sweep"} and `metric` in {"checkpoint",
"static_reference","exhaustive_reference"}) and results/streaming_checks.json
(the M4-exactness / M3-gap / Delta-distribution text-report data).

Four panels, one line per method (consistent color across all four):
  (a) FPR vs. stream position at fixed k_fixed, median + IQR band over the
      K-fold stream-offset sweep, + a dashed static-rebuild reference.
  (b) Threshold-expanded objective inflation (log10_j_tau(method) -
      log10_j_tau(static_reference)) vs. the same stream-position x-axis --
      checks whether the geometric objective actually tracks FPR.
  (c) Final FPR vs. ball budget k.
  (d) Pareto: mean FPR (2nd half of stream) vs. amortized cost/insertion,
      one connected curve per method per k (M5: one curve per b), + the
      exhaustive-Algorithm-2 reference as a 6th dashed curve.
"""

from __future__ import annotations

import dataclasses
import json
import os
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter

from plots.plot_common import apply_style, style_axes, save_figure, save_dataset_legend

INPUT_PATH = "results/streaming.csv"
CHECKS_PATH = "results/streaming_checks.json"
OUTDIR = "plots/streaming/"
K_FIXED: Optional[int] = None
B_REPRESENTATIVE: Optional[int] = None
DPI = 300
EMIT_PDF = False
DEMO = False

METHODS = ["M1", "M2", "M3", "M4", "M5"]
METHOD_LABELS = {
    "M1": "M1: Nearest-center absorb", "M2": "M2: Min radius increase",
    "M3": "M3: Restricted merge", "M4": "M4: Cached-heap merge", "M5": "M5: Periodic rebuild",
}
# Okabe-Ito-derived, distinct from plot_common.DATASET_COLORS (this figure is
# single-dataset -- see main.py's _STREAMING_REPRESENTATIVE_DATASET -- so the
# color channel is free to encode method instead).
METHOD_COLORS = {
    "M1": "#0072B2", "M2": "#E69F00", "M3": "#009E73", "M4": "#D55E00", "M5": "#CC79A7",
}
_M5_B_LINESTYLES = ["-", "--", ":"]

_REQUIRED_COLUMNS = [
    "dataset", "method", "k", "seed", "fold_offset", "b", "position",
    "fpr", "log10_j_tau", "n_balls", "cumulative_time_s", "amortized_cost_s",
    "metric", "sweep",
]


@dataclasses.dataclass
class Config:
    k_fixed: int
    b_representative: int
    dpi: int
    emit_pdf: bool
    outdir: str


# --------------------------------------------------------------------------- #
# Demo data / schema / config
# --------------------------------------------------------------------------- #

def _synthesize_demo_data() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    k_fixed = 32
    k_values = [16, 32, 64, 128, 256, 512, 1024]
    b_values = [1_500, 6_000, 15_000]
    b_representative = 6_000
    n_stream_ab, F_ab = k_fixed * 1000, 4
    n_stream_c, F_cd = 30_000, 4
    positions_ab = np.unique(np.round(np.logspace(np.log10(k_fixed + 1), np.log10(n_stream_ab), 12)).astype(int))
    positions_cd = np.unique(np.round(np.linspace(n_stream_c // 20, n_stream_c, 20)).astype(int))

    # Rough relative quality/cost ordering used only to make the demo figure
    # legible: M1 worst FPR, M4 near the exhaustive reference at far lower
    # cost, M5 bridges cost/quality.
    base_fpr = {"M1": 0.06, "M2": 0.045, "M3": 0.03, "M4": 0.012, "M5": 0.02, "static_reference": 0.008,
               "ref_exhaustive": 0.012}
    cost_scale = {"M1": 1.0, "M2": 1.1, "M3": 1.3, "M4": 2.0, "M5": 4.0, "ref_exhaustive": 40.0}

    rows = []
    for method in METHODS:
        offsets = [f * 1500 for f in range(F_ab)]
        for fold_offset in offsets:
            for pos in positions_ab:
                x = pos / n_stream_ab
                fpr = base_fpr[method] * (0.3 + 0.7 * x) * rng.uniform(0.9, 1.1)
                log10_j = 40.0 + 8.0 * x + rng.normal(0, 0.2)
                rows.append({
                    "dataset": "eli5", "method": method, "k": k_fixed, "seed": 0, "fold_offset": fold_offset,
                    "b": b_representative if method == "M5" else None, "position": int(pos),
                    "fpr": float(fpr), "log10_j_tau": float(log10_j), "n_balls": k_fixed,
                    "cumulative_time_s": cost_scale[method] * pos * 1e-5,
                    "amortized_cost_s": cost_scale[method] * 1e-5,
                    "metric": "checkpoint", "sweep": "position",
                })
        for fold_offset in offsets:
            for pos in positions_ab:
                x = pos / n_stream_ab
                rows.append({
                    "dataset": "eli5", "method": "static_reference", "k": k_fixed, "seed": 0,
                    "fold_offset": fold_offset, "b": None, "position": int(pos),
                    "fpr": float(base_fpr["static_reference"] * (0.3 + 0.7 * x) * rng.uniform(0.9, 1.1)),
                    "log10_j_tau": float(38.0 + 6.0 * x + rng.normal(0, 0.2)), "n_balls": k_fixed,
                    "cumulative_time_s": cost_scale["ref_exhaustive"] * pos * 1e-5,
                    "amortized_cost_s": cost_scale["ref_exhaustive"] * 1e-5,
                    "metric": "static_reference", "sweep": "position",
                })

    offsets_cd = [36500 + f * 1000 for f in range(F_cd)]
    for method in ["M1", "M2", "M3", "M4"]:
        for k in k_values:
            kf = k / 1024.0
            for fold_offset in offsets_cd:
                for pos in positions_cd:
                    fpr = base_fpr[method] / (1.0 + 3.0 * kf) * rng.uniform(0.9, 1.1)
                    rows.append({
                        "dataset": "eli5", "method": method, "k": k, "seed": 0, "fold_offset": fold_offset,
                        "b": None, "position": int(pos), "fpr": float(fpr),
                        "log10_j_tau": float(30.0 + 10.0 * kf), "n_balls": k,
                        "cumulative_time_s": cost_scale[method] * k * pos * 1e-6,
                        "amortized_cost_s": cost_scale[method] * k * 1e-6,
                        "metric": "checkpoint", "sweep": "k_sweep",
                    })
    for k in k_values:
        kf = k / 1024.0
        for b in b_values:
            bf = b / 15000.0
            for fold_offset in offsets_cd:
                for pos in positions_cd:
                    fpr = base_fpr["M5"] / (1.0 + 3.0 * kf) * (1.3 - 0.3 * bf) * rng.uniform(0.9, 1.1)
                    rows.append({
                        "dataset": "eli5", "method": "M5", "k": k, "seed": 0, "fold_offset": fold_offset,
                        "b": b, "position": int(pos), "fpr": float(fpr),
                        "log10_j_tau": float(30.0 + 10.0 * kf), "n_balls": k,
                        "cumulative_time_s": cost_scale["M5"] * k * pos * 1e-6 / bf,
                        "amortized_cost_s": cost_scale["M5"] * k * 1e-6 / bf,
                        "metric": "checkpoint", "sweep": "k_sweep",
                    })
    for k in k_values:
        kf = k / 1024.0
        for fold_offset in offsets_cd:
            for pos in positions_cd:
                rows.append({
                    "dataset": "eli5", "method": "ref_exhaustive", "k": k, "seed": 0, "fold_offset": fold_offset,
                    "b": None, "position": int(pos),
                    "fpr": float(base_fpr["ref_exhaustive"] / (1.0 + 3.0 * kf) * rng.uniform(0.9, 1.1)),
                    "log10_j_tau": float(29.0 + 10.0 * kf), "n_balls": k,
                    "cumulative_time_s": cost_scale["ref_exhaustive"] * k * pos * 1e-6,
                    "amortized_cost_s": cost_scale["ref_exhaustive"] * k * 1e-6,
                    "metric": "exhaustive_reference", "sweep": "k_sweep",
                })

    return pd.DataFrame(rows)


def _synthesize_demo_checks() -> dict:
    return {
        "m4_exactness": {
            "32": {"k": 32, "n_checked": 31968, "n_mismatches": 0, "first_mismatch_position": None},
            "64": {"k": 64, "n_checked": 31936, "n_mismatches": 0, "first_mismatch_position": None},
        },
        "m3_gap": {"k": 32, "n_checked": 31968, "fraction_excluded": 0.083},
        "delta_distribution": {
            "M3": {"n": 31968, "mean": 1.2e-3, "median": 3.1e-4, "p10": -2.0e-4, "p90": 4.5e-3,
                  "min": -1.1e-2, "max": 8.9e-2, "frac_beneficial": 0.12},
            "M4": {"n": 31968, "mean": 4.5e-4, "median": 8.2e-5, "p10": -6.1e-4, "p90": 1.9e-3,
                  "min": -3.4e-2, "max": 5.1e-2, "frac_beneficial": 0.34},
            "ref_exhaustive": {"n": 31968, "mean": 4.5e-4, "median": 8.2e-5, "p10": -6.1e-4, "p90": 1.9e-3,
                              "min": -3.4e-2, "max": 5.1e-2, "frac_beneficial": 0.34},
        },
    }


def _validate_schema(df: pd.DataFrame) -> None:
    missing_cols = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Input missing columns: {missing_cols}. Present: {sorted(df.columns)}")
    missing_methods = set(METHODS) - set(df.loc[df["metric"] == "checkpoint", "method"].unique())
    if missing_methods:
        raise ValueError(f"Missing method(s) {sorted(missing_methods)} among metric=='checkpoint' rows.")
    missing_sweeps = {"position", "k_sweep"} - set(df["sweep"].unique())
    if missing_sweeps:
        raise ValueError(f"Missing sweep(s) {sorted(missing_sweeps)}.")


def load_data() -> pd.DataFrame:
    if DEMO:
        df = _synthesize_demo_data()
    else:
        if not os.path.exists(INPUT_PATH):
            raise FileNotFoundError(f"Input file not found: {INPUT_PATH!r}. Set DEMO=True or fix INPUT_PATH.")
        df = pd.read_parquet(INPUT_PATH) if INPUT_PATH.endswith(".parquet") else pd.read_csv(INPUT_PATH)
    _validate_schema(df)
    return df


def load_checks() -> Optional[dict]:
    if DEMO:
        return _synthesize_demo_checks()
    if not os.path.exists(CHECKS_PATH):
        return None
    with open(CHECKS_PATH, "r") as f:
        return json.load(f)


def resolve_config(df: pd.DataFrame, k_fixed: Optional[int] = K_FIXED,
                   b_representative: Optional[int] = B_REPRESENTATIVE,
                   dpi: int = DPI, emit_pdf: bool = EMIT_PDF, outdir: str = OUTDIR) -> Config:
    pos_rows = df[df["sweep"] == "position"]
    available_k = sorted(pos_rows["k"].unique())
    if k_fixed is None:
        k_fixed = int(available_k[0])
    elif k_fixed not in available_k:
        raise ValueError(f"k_fixed={k_fixed} not present in position-sweep rows. Available: {available_k}")

    m5_b = sorted(pos_rows.loc[pos_rows["method"] == "M5", "b"].dropna().unique())
    if not m5_b:
        raise ValueError("No M5 rows with a 'b' value in the position sweep.")
    if b_representative is None:
        b_representative = int(m5_b[0])
    elif b_representative not in m5_b:
        raise ValueError(f"b_representative={b_representative} not present. Available: {m5_b}")

    return Config(k_fixed=k_fixed, b_representative=b_representative, dpi=dpi, emit_pdf=emit_pdf, outdir=outdir)


# --------------------------------------------------------------------------- #
# Styling helpers
# --------------------------------------------------------------------------- #

def _style_log_y(ax: plt.Axes) -> None:
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="y", which="minor", length=1.2, width=0.4)


def _median_iqr(df: pd.DataFrame, group_cols: List[str], value_col: str) -> pd.DataFrame:
    """Median + IQR (25th/75th percentile) of value_col over the K-fold
    stream-offset sweep, grouped by group_cols -- used instead of mean+-CI95
    since a handful of folds makes single traces cross meaninglessly at
    these FPR scales."""
    agg = df.groupby(group_cols)[value_col].agg(
        median="median", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75), n="count",
    ).reset_index()
    return agg


# --------------------------------------------------------------------------- #
# Panel (a): FPR vs. stream position
# --------------------------------------------------------------------------- #

def panel_fpr_vs_position(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    sel = df[(df["sweep"] == "position") & (df["metric"] == "checkpoint") & (df["k"] == cfg.k_fixed)
            & ((df["method"] != "M5") | (df["b"] == cfg.b_representative))]
    agg = _median_iqr(sel, ["method", "position"], "fpr")

    ref = df[(df["sweep"] == "position") & (df["metric"] == "static_reference") & (df["k"] == cfg.k_fixed)]
    ref_agg = _median_iqr(ref, ["position"], "fpr")

    apply_style(cfg.dpi)
    fig, ax = plt.subplots()
    for method in METHODS:
        sub = agg[agg["method"] == method].sort_values("position")
        if sub.empty:
            continue
        ax.plot(sub["position"], sub["median"], color=METHOD_COLORS[method], marker="o",
               markersize=2.5, linewidth=0.8, zorder=2)
        ax.fill_between(sub["position"], sub["q25"], sub["q75"], color=METHOD_COLORS[method],
                        alpha=0.15, linewidth=0, zorder=1)

    if not ref_agg.empty:
        ref_agg = ref_agg.sort_values("position")
        ax.plot(ref_agg["position"], ref_agg["median"], color="black", linestyle="--", linewidth=0.7, zorder=3)
        ax.text(0.97, 0.95, "dashed: static rebuild", transform=ax.transAxes, ha="right", va="top", fontsize=5)

    ax.set_xscale("log")
    _style_log_y(ax)
    ax.set_xlabel("Stream Position")
    ax.set_ylabel("FPR")
    ax.text(0.03, 0.06, rf"$k = {cfg.k_fixed}$", transform=ax.transAxes, ha="left", va="bottom", fontsize=5.5)
    style_axes(ax)

    save_figure(fig, "streaming-fpr-vs-position.png", cfg.outdir, cfg.dpi, cfg.emit_pdf)
    return agg


# --------------------------------------------------------------------------- #
# Panel (b): objective inflation vs. stream position
# --------------------------------------------------------------------------- #

def panel_objective_inflation(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    sel = df[(df["sweep"] == "position") & (df["metric"] == "checkpoint") & (df["k"] == cfg.k_fixed)
            & ((df["method"] != "M5") | (df["b"] == cfg.b_representative))]
    agg = _median_iqr(sel, ["method", "position"], "log10_j_tau")

    ref = df[(df["sweep"] == "position") & (df["metric"] == "static_reference") & (df["k"] == cfg.k_fixed)]
    ref_agg = _median_iqr(ref, ["position"], "log10_j_tau")[["position", "median"]].rename(
        columns={"median": "ref_median"})

    agg = agg.merge(ref_agg, on="position", how="left")
    agg["inflation"] = agg["median"] - agg["ref_median"]

    apply_style(cfg.dpi)
    fig, ax = plt.subplots()
    for method in METHODS:
        sub = agg[agg["method"] == method].sort_values("position")
        if sub.empty:
            continue
        ax.plot(sub["position"], sub["inflation"], color=METHOD_COLORS[method], marker="o",
               markersize=2.5, linewidth=0.8, zorder=2)

    ax.axhline(0.0, color="black", linewidth=0.4, linestyle=":", zorder=1)
    ax.set_xscale("log")
    ax.set_xlabel("Stream Position")
    ax.set_ylabel(r"$\log_{10}(J_\tau^{\mathrm{method}}/J_\tau^{\mathrm{static}})$")
    ax.text(0.03, 0.95, rf"$k = {cfg.k_fixed}$", transform=ax.transAxes, ha="left", va="top", fontsize=5.5)
    style_axes(ax)

    save_figure(fig, "streaming-objective-inflation.png", cfg.outdir, cfg.dpi, cfg.emit_pdf)
    return agg


# --------------------------------------------------------------------------- #
# Panel (c): final FPR vs. k
# --------------------------------------------------------------------------- #

def panel_fpr_vs_k(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    sel = df[(df["sweep"] == "k_sweep") & (df["metric"] == "checkpoint")
            & ((df["method"] != "M5") | (df["b"] == cfg.b_representative))]
    last_pos = sel.groupby(["method", "k", "fold_offset"], dropna=False)["position"].transform("max")
    final_rows = sel[sel["position"] == last_pos]
    agg = _median_iqr(final_rows, ["method", "k"], "fpr")

    apply_style(cfg.dpi)
    fig, ax = plt.subplots()
    for method in METHODS:
        sub = agg[agg["method"] == method].sort_values("k")
        if sub.empty:
            continue
        ax.plot(sub["k"], sub["median"], color=METHOD_COLORS[method], marker="o",
               markersize=2.5, linewidth=0.8, zorder=2)
        ax.fill_between(sub["k"], sub["q25"], sub["q75"], color=METHOD_COLORS[method],
                        alpha=0.15, linewidth=0, zorder=1)

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(LogLocator(base=2.0))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=2.0))
    _style_log_y(ax)
    ax.set_xlabel("Ball Budget (k)")
    ax.set_ylabel("Final FPR")
    ax.text(0.03, 0.06, rf"$b_{{M5}} = {cfg.b_representative}$", transform=ax.transAxes,
           ha="left", va="bottom", fontsize=5.5)
    style_axes(ax)

    save_figure(fig, "streaming-fpr-vs-k.png", cfg.outdir, cfg.dpi, cfg.emit_pdf)
    return agg


# --------------------------------------------------------------------------- #
# Panel (d): Pareto frontier
# --------------------------------------------------------------------------- #

def panel_pareto(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    sel = df[(df["sweep"] == "k_sweep") & (df["metric"].isin(["checkpoint", "exhaustive_reference"]))]
    last_pos = sel.groupby(["method", "k", "b", "fold_offset"], dropna=False)["position"].transform("max")
    cost_rows = sel[sel["position"] == last_pos]
    cost_agg = cost_rows.groupby(["method", "k", "b"], dropna=False)["amortized_cost_s"].mean().reset_index(
        name="cost")

    half = sel["position"].max() / 2.0
    quality_rows = sel[sel["position"] > half]
    quality_agg = quality_rows.groupby(["method", "k", "b"], dropna=False)["fpr"].mean().reset_index(
        name="mean_fpr")

    points = cost_agg.merge(quality_agg, on=["method", "k", "b"], how="inner")

    apply_style(cfg.dpi)
    fig, ax = plt.subplots()
    for method in METHODS:
        if method == "M5":
            b_vals = sorted(points.loc[points["method"] == "M5", "b"].dropna().unique())
            for ls, b in zip(_M5_B_LINESTYLES, b_vals):
                sub = points[(points["method"] == "M5") & (points["b"] == b)].sort_values("k")
                if sub.empty:
                    continue
                ax.plot(sub["cost"], sub["mean_fpr"], color=METHOD_COLORS["M5"], linestyle=ls,
                       marker="o", markersize=2.2, linewidth=0.8, zorder=2)
        else:
            sub = points[points["method"] == method].sort_values("k")
            if sub.empty:
                continue
            ax.plot(sub["cost"], sub["mean_fpr"], color=METHOD_COLORS[method], marker="o",
                   markersize=2.5, linewidth=0.8, zorder=2)

    ref = points[points["method"] == "ref_exhaustive"].sort_values("k")
    if not ref.empty:
        ax.plot(ref["cost"], ref["mean_fpr"], color="black", linestyle="--", linewidth=0.7, zorder=3)
        ax.text(0.97, 0.95, "dashed: exhaustive Algorithm 2", transform=ax.transAxes,
               ha="right", va="top", fontsize=5)

    ax.set_xscale("log")
    _style_log_y(ax)
    ax.set_xlabel("Amortized Cost / Insertion (s)")
    ax.set_ylabel("Mean FPR (2nd half of stream)")
    style_axes(ax)

    save_figure(fig, "streaming-pareto.png", cfg.outdir, cfg.dpi, cfg.emit_pdf)
    return points


# --------------------------------------------------------------------------- #
# Legend
# --------------------------------------------------------------------------- #

def save_method_legend(cfg: Config) -> None:
    apply_style(cfg.dpi)
    fig, ax = plt.subplots(figsize=(4.6, 0.4))
    ax.axis("off")
    handles = [plt.Line2D([0], [0], color=METHOD_COLORS[m], marker="o", markersize=4, linewidth=1.0)
              for m in METHODS]
    fig.legend(handles, [METHOD_LABELS[m] for m in METHODS], loc="center", ncol=3, frameon=False, fontsize=6.5)
    fig.patch.set_alpha(0.0)
    save_figure(fig, "streaming-method-legend.png", cfg.outdir, cfg.dpi, cfg.emit_pdf, transparent=True)


# --------------------------------------------------------------------------- #
# Text-report checks
# --------------------------------------------------------------------------- #

def print_summary(df: pd.DataFrame, checks: Optional[dict], cfg: Config) -> None:
    print(f"\n--- Streaming summary (k_fixed={cfg.k_fixed}, b_representative={cfg.b_representative}) ---")
    print("FNR=0 confirmed at every checkpoint for all 5 methods "
         "(enforced at generation time in streaming_method_worker; a violation raises, not plots).")

    if checks is None:
        print(f"No {CHECKS_PATH} found -- skipping M4-exactness/M3-gap/Delta-distribution report.\n")
        return

    m4 = checks.get("m4_exactness", {})
    for k_str, result in sorted(m4.items()):
        if result is None:
            continue
        print(f"M4 exactness (k={result['k']}): {result['n_checked']} insertions checked, "
             f"{result['n_mismatches']} mismatches vs. the exhaustive O(k^2 d) reference"
             + (f" (first at position {result['first_mismatch_position']})" if result["n_mismatches"] else "") + ".")

    gap = checks.get("m3_gap")
    if gap:
        print(f"M3 gap: unrestricted argmin excluded the singleton in "
             f"{gap['fraction_excluded']:.1%} of insertions (k={gap['k']}, n={gap['n_checked']}).")

    for method, stats in checks.get("delta_distribution", {}).items():
        print(f"Delta distribution ({method}): mean={stats['mean']:.3g}, median={stats['median']:.3g}, "
             f"[p10,p90]=[{stats['p10']:.3g},{stats['p90']:.3g}], {stats['frac_beneficial']:.1%} beneficial "
             f"(n={stats['n']}).")
    print()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def generate_streaming_figures(df: Optional[pd.DataFrame] = None, outdir: str = OUTDIR,
                               k_fixed: Optional[int] = K_FIXED, b_representative: Optional[int] = B_REPRESENTATIVE,
                               dpi: int = DPI, emit_pdf: bool = EMIT_PDF) -> None:
    if df is None:
        df = load_data()
    else:
        _validate_schema(df)
    checks = load_checks()

    cfg = resolve_config(df, k_fixed=k_fixed, b_representative=b_representative, dpi=dpi, emit_pdf=emit_pdf,
                         outdir=outdir)

    panel_fpr_vs_position(df, cfg)
    panel_objective_inflation(df, cfg)
    panel_fpr_vs_k(df, cfg)
    panel_pareto(df, cfg)
    save_method_legend(cfg)
    print_summary(df, checks, cfg)

    snippet = r"""
\begin{subfigure}[b]{\linewidth}
    \centering
    \includegraphics[width=0.6\linewidth]{figures/streaming-method-legend.png}
\end{subfigure}\\[-0.3em]
\begin{subfigure}[b]{0.24\linewidth}
    \centering
    \includegraphics[width=\linewidth]{figures/streaming-fpr-vs-position.png}
    \caption{FPR vs. stream position}
\end{subfigure}
\begin{subfigure}[b]{0.24\linewidth}
    \centering
    \includegraphics[width=\linewidth]{figures/streaming-objective-inflation.png}
    \caption{Objective inflation}
\end{subfigure}
\begin{subfigure}[b]{0.24\linewidth}
    \centering
    \includegraphics[width=\linewidth]{figures/streaming-fpr-vs-k.png}
    \caption{Final FPR vs. $k$}
\end{subfigure}
\begin{subfigure}[b]{0.24\linewidth}
    \centering
    \includegraphics[width=\linewidth]{figures/streaming-pareto.png}
    \caption{Cost/quality Pareto}
\end{subfigure}
""".strip("\n")
    print("--- subfigure snippet (paste above the panel row) ---")
    print(snippet)


def main() -> None:
    generate_streaming_figures()


if __name__ == "__main__":
    main()
