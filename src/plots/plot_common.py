"""Shared infrastructure for the per-evaluation figure scripts (plot_quantization.py,
plot_query_performance.py, plot_scaling.py, plot_streaming.py): the dataset
trio + styling every figure in the paper shares, the matplotlib rcParams
preset, and small save/aggregate helpers."""

from __future__ import annotations

import os
import textwrap
from typing import List

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATASETS = ["eli5", "mmlu", "wildchat", "msmarco", "hotpotqa"]
DATASET_LABELS = {
    "eli5": "ELI5", "mmlu": "MMLU", "wildchat": "WildChat",
    "msmarco": "MsMarco", "hotpotqa": "HotPotQA",
}
DATASET_COLORS = {
    "eli5": "#0072B2", "mmlu": "#E69F00", "wildchat": "#009E73",
    "msmarco": "#D55E00", "hotpotqa": "#CC79A7",
}
# Distinct matplotlib marker per dataset, so lines (and the shared legend)
# stay distinguishable by shape alone (grayscale printouts, colorblind
# readers) and not just color.
DATASET_MARKERS = {"eli5": "o", "mmlu": "s", "wildchat": "^", "msmarco": "D", "hotpotqa": "v"}


def apply_style(dpi: int = 300) -> None:
    plt.rcParams.update({
        # Scaled by the same 1.2x as the font sizes below, so the ratio of
        # text to canvas is unchanged -- keeps every fixed, non-tight
        # subplots_adjust() margin elsewhere (tuned for the old ratio, and
        # relied on to align panels placed side by side) still fitting
        # without touching each one individually.
        "figure.figsize": (1.70 * 1.2, 1.50 * 1.2),
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "font.family": "serif",
        "font.serif": ["Linux Libertine O", "Libertine", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.labelsize": 8.4,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "axes.titlesize": 8.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 2,
        "ytick.major.size": 2,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.size": 0,
        "ytick.minor.size": 0,
        "legend.frameon": False,
        "legend.fontsize": 8.4,
        "legend.title_fontsize": 8.4,
    })


def wrap_label(text: str, width: int = 20) -> str:
    """Word-wraps an axis label to `width` chars per line: bbox_inches="tight"
    clips a long single-line label instead of expanding to fit it. Passed
    through unchanged if it contains a "$" -- textwrap would insert a
    newline mid-mathtext-span, breaking the expression; callers with a long
    LaTeX label need to hand-wrap it into separate "$...$" segments instead."""
    if "$" in text:
        return text
    return textwrap.fill(text, width=width)


def style_axes(ax: plt.Axes) -> None:
    ax.grid(axis="both", which="major", linewidth=0.35, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.5)


def save_figure(fig: plt.Figure, filename: str, outdir: str, dpi: int,
                emit_pdf: bool = False, transparent: bool = False, tight: bool = True) -> None:
    """tight=False saves the full figure canvas at its exact configured size
    instead of cropping to content -- needed when a set of figures must align
    when placed side by side, since bbox_inches="tight" crops each one to its
    own label extent, shifting the axes by a different amount per figure."""
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, filename)
    bbox_kwargs = dict(bbox_inches="tight", pad_inches=0.01) if tight else {}
    fig.savefig(path, dpi=dpi, transparent=transparent, **bbox_kwargs)
    if emit_pdf:
        fig.savefig(path.replace(".png", ".pdf"), transparent=transparent, **bbox_kwargs)
    plt.close(fig)
    print(f"Wrote {path}")


def mean_ci95(df: pd.DataFrame, group_cols: List[str], value_col: str) -> pd.DataFrame:
    """Mean + 95% CI of value_col over seed, grouped by group_cols. Stat columns
    are prefixed "_" so they never collide with a group_cols name (e.g. "n")."""
    agg = df.groupby(group_cols)[value_col].agg(_mean="mean", _sem="sem", _count="count").reset_index()
    agg["mean"] = agg["_mean"]
    agg["ci95"] = np.where(agg["_count"] > 1, 1.96 * agg["_sem"].fillna(0.0), 0.0)
    return agg.drop(columns=["_mean", "_sem", "_count"])


def mean_std(df: pd.DataFrame, group_cols: List[str], value_col: str) -> pd.DataFrame:
    """Mean +/- standard deviation of value_col over seed, grouped by
    group_cols -- unlike mean_ci95's standard error, this doesn't shrink as
    more seeds/folds are added, so it stays a direct read of how much
    individual runs (e.g. K-fold splits) actually vary. Stat columns are
    prefixed "_" so they never collide with a group_cols name (e.g. "n")."""
    agg = df.groupby(group_cols)[value_col].agg(_mean="mean", _std="std", _count="count").reset_index()
    agg["mean"] = agg["_mean"]
    agg["std"] = np.where(agg["_count"] > 1, agg["_std"].fillna(0.0), 0.0)
    return agg.drop(columns=["_mean", "_std", "_count"])


def save_dataset_legend(outdir: str, dpi: int, emit_pdf: bool = False, filename: str = "dataset-legend.png") -> None:
    apply_style(dpi)
    fig, ax = plt.subplots(figsize=(6.4, 0.25))
    ax.axis("off")
    handles = [
        plt.Line2D([0], [0], color=DATASET_COLORS[d], marker=DATASET_MARKERS[d],
                  markersize=4, linewidth=0.8)
        for d in DATASETS
    ]
    labels = [DATASET_LABELS[d] for d in DATASETS]
    fig.legend(handles, labels, loc="center", ncol=len(DATASETS), frameon=False, fontsize=8.4)
    fig.patch.set_alpha(0.0)
    save_figure(fig, filename, outdir, dpi, emit_pdf, transparent=True)
