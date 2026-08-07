"""Generate the streaming-evaluation figures (+ legends) for the PVLDB paper.

Four panels, each answering a distinct question about the online (streaming)
merge policy Algorithm 2 implements:

  (a) Absorption rate vs. stream position, 3 datasets, log x. The paper's
      central empirical claim: if this plateaus near 1, "representation size
      is independent of n" stops being asymptotic and becomes measured.
  (b) FPR vs. stream position for 3 rebuild policies (streaming / periodic at
      one representative T / oracle), one representative dataset. Since no
      theorem bounds streaming quality, this is the empirical evidence for it.
  (c) Final FPR vs. k, one line per pairwise-merge criterion, one
      representative dataset -- compares GonzalezMEBFilter.MERGE_CRITERIA.
  (d) Rebuild-policy Pareto: amortized cost per insertion (incl. rebuilds) on
      x, mean FPR on y, one point per policy (streaming / several periodic T /
      oracle) -- answers "when do you rebuild" and the cost story together.

Self-contained: reads the three long-format results files produced by
main.evaluate_streaming(), aggregates over seeds, and writes PNG (optionally
PDF) panels sized for `\\includegraphics[width=\\linewidth]` inside
`subfigure{0.24\\linewidth}` cells of a two-column `figure*`, matching
plot_ablation.py's / plot_scaling.py's panel styling.

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

from plot_ablation import apply_style

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #

INPUT_PATH_ABSORPTION = "results/streaming_absorption.csv"
INPUT_PATH_REBUILD = "results/streaming_rebuild.csv"
INPUT_PATH_MERGE_CRITERIA = "results/streaming_merge_criteria.csv"
OUTDIR = "figures/"
K_FIXED: Optional[int] = None  # None => the single k present in the absorption/rebuild data
T_REPRESENTATIVE: Optional[int] = None  # None => median T present in the periodic-policy data
DPI = 300
EMIT_PDF = False
DEMO = False

# --------------------------------------------------------------------------- #
# Fixed dataset styling -- must match every other figure in the paper.
# --------------------------------------------------------------------------- #

DATASETS = ["eli5", "mmlu", "wildchat"]
DATASET_LABELS = {"eli5": "ELI5", "mmlu": "MMLU", "wildchat": "WildChat"}
DATASET_COLORS = {"eli5": "#0072B2", "mmlu": "#E69F00", "wildchat": "#009E73"}
DATASET_HATCHES = {"eli5": "", "mmlu": "//", "wildchat": ".."}

POLICIES = ["streaming", "periodic", "oracle"]
POLICY_LABELS = {"streaming": "Streaming (no rebuild)", "periodic": "Periodic rebuild", "oracle": "Oracle rebuild"}
POLICY_COLORS = {"streaming": "#D55E00", "periodic": "#CC79A7", "oracle": "#000000"}
POLICY_MARKERS = {"streaming": "o", "periodic": "s", "oracle": "^"}

MERGE_CRITERIA = ["log_space", "r_pow_d", "sum_radii", "min_radius_increase"]
MERGE_CRITERION_LABELS = {
    "log_space": r"$\log$-space $r^d$", "r_pow_d": r"raw $r^d$",
    "sum_radii": r"$\Sigma r$", "min_radius_increase": "min radius increase",
}
MERGE_CRITERION_COLORS = {
    "log_space": "#0072B2", "r_pow_d": "#D55E00", "sum_radii": "#E69F00", "min_radius_increase": "#009E73",
}


@dataclasses.dataclass
class Config:
    k_fixed: int
    t_representative: int
    dpi: int
    emit_pdf: bool
    outdir: str


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

_ABSORPTION_COLUMNS = ["dataset", "k", "seed", "position", "window_start", "absorption_rate", "n_balls"]
_REBUILD_COLUMNS = ["dataset", "policy", "T", "k", "seed", "position", "fpr", "cumulative_time_s", "amortized_cost_s"]
_MERGE_CRITERIA_COLUMNS = ["dataset", "k", "seed", "merge_criterion", "fpr"]


def _synthesize_demo_absorption() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    positions = np.unique(np.round(np.logspace(np.log10(50), np.log10(8000), 10)).astype(int))
    seeds = [0, 1, 2]
    k_fixed = 128
    plateau = {"eli5": 0.94, "mmlu": 0.90, "wildchat": 0.97}
    rate = {"eli5": 1.6, "mmlu": 1.3, "wildchat": 2.0}

    rows = []
    for dataset in DATASETS:
        for seed in seeds:
            prev = 0
            for pos in positions:
                x = pos / positions[-1]
                mean = plateau[dataset] * (1.0 - np.exp(-rate[dataset] * x * 6))
                noise = rng.normal(0, 0.02)
                rows.append({
                    "dataset": dataset, "k": k_fixed, "seed": seed,
                    "position": int(pos), "window_start": prev + 1,
                    "absorption_rate": float(np.clip(mean + noise, 0.0, 1.0)),
                    "n_balls": k_fixed,
                })
                prev = pos
    return pd.DataFrame(rows)


def _synthesize_demo_rebuild() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    positions = np.unique(np.round(np.logspace(np.log10(50), np.log10(8000), 10)).astype(int))
    seeds = [0, 1, 2]
    k_fixed = 128
    T_values = [200, 400, 800, 1600, 4000]

    rows = []
    for seed in seeds:
        # streaming: highest FPR floor, cheapest.
        cum_time = 0.0
        for pos in positions:
            cum_time += 0.00012 * (pos - (positions[positions < pos][-1] if (positions < pos).any() else 0))
            fpr = 0.03 + 0.05 * np.exp(-pos / 2000.0) * rng.uniform(0.9, 1.1)
            rows.append({"dataset": "eli5", "policy": "streaming", "T": None, "k": k_fixed, "seed": seed,
                        "position": int(pos), "fpr": float(fpr), "cumulative_time_s": cum_time,
                        "amortized_cost_s": cum_time / pos})

        # oracle: lowest FPR, most expensive (cost grows with position, amortized roughly flat-ish).
        cum_time = 0.0
        for pos in positions:
            step_cost = 0.02 * pos
            cum_time += step_cost
            fpr = 0.008 + 0.01 * np.exp(-pos / 1500.0) * rng.uniform(0.9, 1.1)
            rows.append({"dataset": "eli5", "policy": "oracle", "T": None, "k": k_fixed, "seed": seed,
                        "position": int(pos), "fpr": float(fpr), "cumulative_time_s": cum_time,
                        "amortized_cost_s": cum_time / pos})

        # periodic(T): interpolate cost/quality between streaming and oracle as T shrinks.
        for T in T_values:
            frac = 1.0 - (T - min(T_values)) / (max(T_values) - min(T_values) + 1e-9)  # 1 near oracle, 0 near streaming
            cum_time = 0.0
            for pos in positions:
                prev_pos = positions[positions < pos][-1] if (positions < pos).any() else 0
                n_rebuilds = pos // T - prev_pos // T
                cum_time += 0.00012 * (pos - prev_pos) + n_rebuilds * 0.02 * pos * 0.3
                fpr = (0.008 * frac + 0.03 * (1 - frac)) + 0.02 * np.exp(-pos / 1800.0) * rng.uniform(0.9, 1.1)
                rows.append({"dataset": "eli5", "policy": "periodic", "T": T, "k": k_fixed, "seed": seed,
                            "position": int(pos), "fpr": float(fpr), "cumulative_time_s": cum_time,
                            "amortized_cost_s": cum_time / pos})
    return pd.DataFrame(rows)


def _synthesize_demo_merge_criteria() -> pd.DataFrame:
    rng = np.random.default_rng(2)
    ks = [16, 32, 64, 128, 256, 512]
    seeds = [0, 1, 2]
    base = {"log_space": 0.02, "r_pow_d": 0.021, "sum_radii": 0.05, "min_radius_increase": 0.045}
    degrade_at_high_k = {"log_space": 1.0, "r_pow_d": 1.9, "sum_radii": 1.3, "min_radius_increase": 1.2}

    rows = []
    for k in ks:
        k_factor = k / 512.0
        for crit in MERGE_CRITERIA:
            for seed in seeds:
                fpr = base[crit] / (1.0 + 4.0 * (1 - k_factor)) * (1.0 + (degrade_at_high_k[crit] - 1.0) * k_factor)
                fpr *= rng.uniform(0.9, 1.1)
                rows.append({"dataset": "eli5", "k": k, "seed": seed, "merge_criterion": crit, "fpr": float(fpr)})
    return pd.DataFrame(rows)


def _validate_absorption(df: pd.DataFrame) -> None:
    missing = [c for c in _ABSORPTION_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"streaming_absorption input missing columns: {missing}. Present: {sorted(df.columns)}")
    missing_ds = set(DATASETS) - set(df["dataset"].unique())
    if missing_ds:
        raise ValueError(f"streaming_absorption missing dataset(s) {sorted(missing_ds)}")


def _validate_rebuild(df: pd.DataFrame) -> None:
    missing = [c for c in _REBUILD_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"streaming_rebuild input missing columns: {missing}. Present: {sorted(df.columns)}")
    missing_pol = set(POLICIES) - set(df["policy"].unique())
    if missing_pol:
        raise ValueError(f"streaming_rebuild missing policy(ies) {sorted(missing_pol)}")


def _validate_merge_criteria(df: pd.DataFrame) -> None:
    missing = [c for c in _MERGE_CRITERIA_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"streaming_merge_criteria input missing columns: {missing}. Present: {sorted(df.columns)}")
    missing_mc = set(MERGE_CRITERIA) - set(df["merge_criterion"].unique())
    if missing_mc:
        raise ValueError(f"streaming_merge_criteria missing merge_criterion(s) {sorted(missing_mc)}")


def _load(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Input file not found: {path!r}. Set DEMO = True to render with synthetic data, "
            "or point the INPUT_PATH_* constants at real results files."
        )
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)


def load_data() -> tuple:
    if DEMO:
        df_absorption = _synthesize_demo_absorption()
        df_rebuild = _synthesize_demo_rebuild()
        df_merge_criteria = _synthesize_demo_merge_criteria()
    else:
        df_absorption = _load(INPUT_PATH_ABSORPTION)
        df_rebuild = _load(INPUT_PATH_REBUILD)
        df_merge_criteria = _load(INPUT_PATH_MERGE_CRITERIA)

    _validate_absorption(df_absorption)
    _validate_rebuild(df_rebuild)
    _validate_merge_criteria(df_merge_criteria)
    return df_absorption, df_rebuild, df_merge_criteria


def resolve_config(
    df_absorption: pd.DataFrame,
    df_rebuild: pd.DataFrame,
    k_fixed: Optional[int] = K_FIXED,
    t_representative: Optional[int] = T_REPRESENTATIVE,
    dpi: int = DPI,
    emit_pdf: bool = EMIT_PDF,
    outdir: str = OUTDIR,
) -> Config:
    available_k = sorted(df_absorption["k"].unique())
    if k_fixed is None:
        k_fixed = int(available_k[0])
    elif k_fixed not in available_k:
        raise ValueError(f"k_fixed={k_fixed} not present in absorption data. Available: {available_k}")

    periodic_T = sorted(df_rebuild.loc[df_rebuild["policy"] == "periodic", "T"].dropna().unique())
    if not periodic_T:
        raise ValueError("streaming_rebuild data has no 'periodic' policy rows to pick a representative T from.")
    if t_representative is None:
        t_representative = int(periodic_T[len(periodic_T) // 2])
    elif t_representative not in periodic_T:
        raise ValueError(f"T_representative={t_representative} not present. Available periodic T: {periodic_T}")

    return Config(k_fixed=k_fixed, t_representative=t_representative, dpi=dpi, emit_pdf=emit_pdf, outdir=outdir)


# --------------------------------------------------------------------------- #
# Panel (a): absorption rate vs. stream position
# --------------------------------------------------------------------------- #

def panel_absorption(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    sel = df[df["k"] == cfg.k_fixed]
    grouped = sel.groupby(["dataset", "position"])["absorption_rate"]
    agg = grouped.agg(mean="mean", sem="sem", n="count").reset_index()
    agg["ci95"] = np.where(agg["n"] > 1, 1.96 * agg["sem"].fillna(0.0), 0.0)

    apply_style()
    fig, ax = plt.subplots()
    for dataset in DATASETS:
        sub = agg[agg["dataset"] == dataset].sort_values("position")
        if sub.empty:
            continue
        ax.plot(sub["position"], sub["mean"], color=DATASET_COLORS[dataset],
               marker="o", markersize=2.5, linewidth=0.8, zorder=2)
        if (sub["ci95"] > 0).any():
            ax.errorbar(sub["position"], sub["mean"], yerr=sub["ci95"], fmt="none",
                       ecolor=DATASET_COLORS[dataset], elinewidth=0.5, capsize=1.5, capthick=0.5, zorder=3)

    ax.axhline(1.0, color="black", linewidth=0.4, linestyle=":", zorder=1)
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Stream Position")
    ax.set_ylabel("Absorption Rate")
    ax.text(0.03, 0.06, rf"$k = {cfg.k_fixed}$", transform=ax.transAxes, ha="left", va="bottom", fontsize=5.5)
    ax.grid(axis="both", which="major", linewidth=0.35, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.5)

    _save(fig, "streaming-absorption.png", cfg)
    return agg


# --------------------------------------------------------------------------- #
# Panel (b): FPR vs. stream position, one line per rebuild policy
# --------------------------------------------------------------------------- #

def panel_rebuild_fpr(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    sel = df[(df["policy"] != "periodic") | (df["T"] == cfg.t_representative)]
    grouped = sel.groupby(["policy", "position"])["fpr"]
    agg = grouped.agg(mean="mean", sem="sem", n="count").reset_index()
    agg["ci95"] = np.where(agg["n"] > 1, 1.96 * agg["sem"].fillna(0.0), 0.0)

    apply_style()
    fig, ax = plt.subplots()
    for policy in POLICIES:
        sub = agg[agg["policy"] == policy].sort_values("position")
        if sub.empty:
            continue
        ax.plot(sub["position"], sub["mean"], color=POLICY_COLORS[policy],
               marker=POLICY_MARKERS[policy], markersize=2.5, linewidth=0.8, zorder=2)
        if (sub["ci95"] > 0).any():
            ax.errorbar(sub["position"], sub["mean"], yerr=sub["ci95"], fmt="none",
                       ecolor=POLICY_COLORS[policy], elinewidth=0.5, capsize=1.5, capthick=0.5, zorder=3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="y", which="minor", length=1.2, width=0.4)

    ax.set_xlabel("Stream Position")
    ax.set_ylabel("FPR")
    ax.text(0.03, 0.06, rf"$k = {cfg.k_fixed},\ T = {cfg.t_representative}$",
           transform=ax.transAxes, ha="left", va="bottom", fontsize=5.5)
    ax.grid(axis="both", which="major", linewidth=0.35, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.5)

    _save(fig, "streaming-fpr.png", cfg)
    return agg


# --------------------------------------------------------------------------- #
# Panel (c): final FPR vs. k, one line per merge criterion
# --------------------------------------------------------------------------- #

def panel_merge_criteria(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    grouped = df.groupby(["merge_criterion", "k"])["fpr"]
    agg = grouped.agg(mean="mean", sem="sem", n="count").reset_index()
    agg["ci95"] = np.where(agg["n"] > 1, 1.96 * agg["sem"].fillna(0.0), 0.0)

    apply_style()
    fig, ax = plt.subplots()
    for crit in MERGE_CRITERIA:
        sub = agg[agg["merge_criterion"] == crit].sort_values("k")
        if sub.empty:
            continue
        ax.plot(sub["k"], sub["mean"], color=MERGE_CRITERION_COLORS[crit],
               marker="o", markersize=2.5, linewidth=0.8, zorder=2)
        if (sub["ci95"] > 0).any():
            ax.errorbar(sub["k"], sub["mean"], yerr=sub["ci95"], fmt="none",
                       ecolor=MERGE_CRITERION_COLORS[crit], elinewidth=0.5, capsize=1.5, capthick=0.5, zorder=3)

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(LogLocator(base=2.0))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=2.0))
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="y", which="minor", length=1.2, width=0.4)

    ax.set_xlabel("Filter Size (k)")
    ax.set_ylabel("Final FPR")
    ax.grid(axis="both", which="major", linewidth=0.35, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.5)

    _save(fig, "streaming-merge-criteria.png", cfg)
    return agg


# --------------------------------------------------------------------------- #
# Panel (d): rebuild-policy Pareto (amortized cost vs. mean FPR)
# --------------------------------------------------------------------------- #

def panel_rebuild_pareto(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    One point per policy (streaming; oracle; periodic at every T present in
    the data). x = amortized cost per insertion at the final checkpoint,
    averaged over seeds; y = mean FPR across every checkpoint and seed for
    that policy -- overall quality across the policy's whole lifetime, not
    just its endpoint.
    """
    # dropna=False: "streaming"/"oracle" rows carry T=NaN, and pandas groupby
    # drops NaN-keyed groups by default -- without this those two policies
    # silently vanish from the Pareto plot instead of appearing as endpoints.
    last_pos = df.groupby(["policy", "T", "seed"], dropna=False)["position"].transform("max")
    final_rows = df[df["position"] == last_pos]

    cost = final_rows.groupby(["policy", "T"], dropna=False)["amortized_cost_s"].mean().reset_index(name="cost")
    quality = df.groupby(["policy", "T"], dropna=False)["fpr"].mean().reset_index(name="mean_fpr")
    points = cost.merge(quality, on=["policy", "T"])

    apply_style()
    fig, ax = plt.subplots()

    periodic = points[points["policy"] == "periodic"].sort_values("T")
    if not periodic.empty:
        ax.plot(periodic["cost"], periodic["mean_fpr"], color=POLICY_COLORS["periodic"],
               linewidth=0.6, linestyle="--", zorder=1)
        ax.scatter(periodic["cost"], periodic["mean_fpr"], color=POLICY_COLORS["periodic"],
                  marker=POLICY_MARKERS["periodic"], s=10, zorder=2)
        for _, row in periodic.iterrows():
            ax.annotate(f"T={int(row['T'])}", (row["cost"], row["mean_fpr"]),
                       fontsize=4.5, xytext=(2, 2), textcoords="offset points")

    for policy in ("streaming", "oracle"):
        sub = points[points["policy"] == policy]
        if sub.empty:
            continue
        ax.scatter(sub["cost"], sub["mean_fpr"], color=POLICY_COLORS[policy],
                  marker=POLICY_MARKERS[policy], s=14, zorder=3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="y", which="minor", length=1.2, width=0.4)

    ax.set_xlabel("Amortized Cost / Insertion (s)")
    ax.set_ylabel("Mean FPR")
    ax.grid(axis="both", which="major", linewidth=0.35, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.5)

    _save(fig, "streaming-rebuild-pareto.png", cfg)
    return points


# --------------------------------------------------------------------------- #
# Legends
# --------------------------------------------------------------------------- #

def save_dataset_legend(cfg: Config) -> None:
    """Shared with plot_scaling.py's dataset-legend.png (same trio + colors)."""
    apply_style()
    fig, ax = plt.subplots(figsize=(4.2, 0.25))
    ax.axis("off")
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=DATASET_COLORS[d], hatch=DATASET_HATCHES[d],
                     edgecolor="black", linewidth=0.4)
        for d in DATASETS
    ]
    labels = [DATASET_LABELS[d] for d in DATASETS]
    fig.legend(handles, labels, loc="center", ncol=3, frameon=False, fontsize=7)
    fig.patch.set_alpha(0.0)
    path = os.path.join(cfg.outdir, "dataset-legend.png")
    fig.savefig(path, dpi=cfg.dpi, bbox_inches="tight", pad_inches=0.01, transparent=True)
    if cfg.emit_pdf:
        fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight", pad_inches=0.01, transparent=True)
    plt.close(fig)
    print(f"Wrote {path}")


def save_policy_legend(cfg: Config) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(4.6, 0.25))
    ax.axis("off")
    handles = [
        plt.Line2D([0], [0], color=POLICY_COLORS[p], marker=POLICY_MARKERS[p],
                  markersize=4, linewidth=1.0)
        for p in POLICIES
    ]
    labels = [POLICY_LABELS[p] for p in POLICIES]
    fig.legend(handles, labels, loc="center", ncol=3, frameon=False, fontsize=7)
    fig.patch.set_alpha(0.0)
    path = os.path.join(cfg.outdir, "streaming-policy-legend.png")
    fig.savefig(path, dpi=cfg.dpi, bbox_inches="tight", pad_inches=0.01, transparent=True)
    if cfg.emit_pdf:
        fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight", pad_inches=0.01, transparent=True)
    plt.close(fig)
    print(f"Wrote {path}")


def save_merge_criterion_legend(cfg: Config) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(4.6, 0.25))
    ax.axis("off")
    handles = [
        plt.Line2D([0], [0], color=MERGE_CRITERION_COLORS[c], marker="o", markersize=4, linewidth=1.0)
        for c in MERGE_CRITERIA
    ]
    labels = [MERGE_CRITERION_LABELS[c] for c in MERGE_CRITERIA]
    fig.legend(handles, labels, loc="center", ncol=2, frameon=False, fontsize=7)
    fig.patch.set_alpha(0.0)
    path = os.path.join(cfg.outdir, "streaming-merge-criterion-legend.png")
    fig.savefig(path, dpi=cfg.dpi, bbox_inches="tight", pad_inches=0.01, transparent=True)
    if cfg.emit_pdf:
        fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight", pad_inches=0.01, transparent=True)
    plt.close(fig)
    print(f"Wrote {path}")


# --------------------------------------------------------------------------- #
# Save helper
# --------------------------------------------------------------------------- #

def _save(fig: plt.Figure, filename: str, cfg: Config) -> None:
    os.makedirs(cfg.outdir, exist_ok=True)
    path = os.path.join(cfg.outdir, filename)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.01, dpi=cfg.dpi)
    if cfg.emit_pdf:
        fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"Wrote {path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def generate_streaming_figures(
    df_absorption: Optional[pd.DataFrame] = None,
    df_rebuild: Optional[pd.DataFrame] = None,
    df_merge_criteria: Optional[pd.DataFrame] = None,
    outdir: str = OUTDIR,
    k_fixed: Optional[int] = K_FIXED,
    T_representative: Optional[int] = T_REPRESENTATIVE,
    dpi: int = DPI,
    emit_pdf: bool = EMIT_PDF,
) -> None:
    """
    Library entry point: render the four streaming panels (+ legends) from
    in-memory long-format DataFrames, the same way
    plot_scaling.generate_scaling_figures is called directly on an in-memory
    df instead of round-tripping through a file. If any df is omitted, falls
    back to load_data() (CONFIG-driven, honors DEMO/INPUT_PATH_*) so the
    script still works standalone.
    """
    if df_absorption is None or df_rebuild is None or df_merge_criteria is None:
        df_absorption, df_rebuild, df_merge_criteria = load_data()
    else:
        _validate_absorption(df_absorption)
        _validate_rebuild(df_rebuild)
        _validate_merge_criteria(df_merge_criteria)

    cfg = resolve_config(df_absorption, df_rebuild, k_fixed=k_fixed, t_representative=T_representative,
                         dpi=dpi, emit_pdf=emit_pdf, outdir=outdir)

    panel_absorption(df_absorption, cfg)
    panel_rebuild_fpr(df_rebuild, cfg)
    panel_merge_criteria(df_merge_criteria, cfg)
    panel_rebuild_pareto(df_rebuild, cfg)
    save_dataset_legend(cfg)
    save_policy_legend(cfg)
    save_merge_criterion_legend(cfg)

    snippet = r"""
\begin{subfigure}{0.24\linewidth}
    \includegraphics[width=\linewidth]{figures/streaming-absorption.png}
    \caption{Absorption rate}
\end{subfigure}\hfill
\begin{subfigure}{0.24\linewidth}
    \includegraphics[width=\linewidth]{figures/streaming-fpr.png}
    \caption{FPR by rebuild policy}
\end{subfigure}\hfill
\begin{subfigure}{0.24\linewidth}
    \includegraphics[width=\linewidth]{figures/streaming-merge-criteria.png}
    \caption{FPR by merge criterion}
\end{subfigure}\hfill
\begin{subfigure}{0.24\linewidth}
    \includegraphics[width=\linewidth]{figures/streaming-rebuild-pareto.png}
    \caption{Rebuild-policy Pareto}
\end{subfigure}
""".strip("\n")
    print("--- subfigure snippet (paste above the panel row) ---")
    print(snippet)


def main() -> None:
    generate_streaming_figures()


if __name__ == "__main__":
    main()
