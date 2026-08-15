"""Shared infrastructure for the per-evaluation figure scripts (plot_ablation.py,
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

DATASETS = ["eli5", "mmlu", "wildchat"]
DATASET_LABELS = {"eli5": "ELI5", "mmlu": "MMLU", "wildchat": "WildChat"}
DATASET_COLORS = {"eli5": "#0072B2", "mmlu": "#E69F00", "wildchat": "#009E73"}
DATASET_HATCHES = {"eli5": "", "mmlu": "//", "wildchat": ".."}


def apply_style(dpi: int = 300) -> None:
    plt.rcParams.update({
        "figure.figsize": (1.70, 1.50),
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "font.family": "serif",
        "font.serif": ["Linux Libertine O", "Libertine", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "axes.titlesize": 7,
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
        "legend.fontsize": 7,
        "legend.title_fontsize": 7,
    })


def wrap_label(text: str, width: int = 20) -> str:
    """Word-wraps an axis label/title to `width` chars per line. The figure
    size these scripts use (~1.7in wide) is narrower than many verbose
    labels; bbox_inches="tight" silently clips the trailing text of a long
    single-line label instead of expanding to fit it, so anything wordy
    needs to be wrapped onto multiple lines before it's set on an axis.

    Passed through unchanged if it contains a "$": textwrap breaks purely on
    whitespace, so it can (and will) insert a newline in the middle of a
    mathtext "$...$" span, leaving each half an unparseable expression that
    mathtext falls back to rendering as literal text. Callers with a long
    LaTeX label need to hand-wrap it into separate self-contained "$...$"
    segments joined by "\\n" instead."""
    if "$" in text:
        return text
    return textwrap.fill(text, width=width)


def style_axes(ax: plt.Axes) -> None:
    ax.grid(axis="both", which="major", linewidth=0.35, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.5)


def save_figure(fig: plt.Figure, filename: str, outdir: str, dpi: int,
                emit_pdf: bool = False, transparent: bool = False) -> None:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, filename)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.01, transparent=transparent)
    if emit_pdf:
        fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight", pad_inches=0.01, transparent=transparent)
    plt.close(fig)
    print(f"Wrote {path}")


def mean_ci95(df: pd.DataFrame, group_cols: List[str], value_col: str) -> pd.DataFrame:
    """Mean + 95% CI of value_col over seed, grouped by group_cols. Stat columns
    are prefixed "_" so they never collide with a group_cols name (e.g. "n")."""
    agg = df.groupby(group_cols)[value_col].agg(_mean="mean", _sem="sem", _count="count").reset_index()
    agg["mean"] = agg["_mean"]
    agg["ci95"] = np.where(agg["_count"] > 1, 1.96 * agg["_sem"].fillna(0.0), 0.0)
    return agg.drop(columns=["_mean", "_sem", "_count"])


def save_dataset_legend(outdir: str, dpi: int, emit_pdf: bool = False, filename: str = "dataset-legend.png") -> None:
    apply_style(dpi)
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
    save_figure(fig, filename, outdir, dpi, emit_pdf, transparent=True)
