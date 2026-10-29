"""Construction-method figures (see main.generate_construct_methods).

One figure per corpus, three panels against the ball budget k (% of |B|, log
axis), one line per construction method, mean +/- 1 std over folds:

  (a) log10 volume of the tau-expanded cover. The line is the overlap-aware
      upper bound (runner._tau_volume_bounds_log10), and the band beneath
      it reaches down to the lower bound (usually too thin to see).
  (b) FPR against distance ground truth at tau.
  (c) construct_filter time, single core.

Gonzalez is the default method and the baseline every other line is read
against, so it is drawn as a dashed neutral line rather than taking a
categorical hue. The four hues are the dataviz reference palette's first
three slots plus its violet slot, validated as a set (all pairs, light
surface). Violet rather than the palette's own fourth slot because that one
is yellow, which fails the all-pairs normal-vision floor next to slot 2's
orange (dE 13.7, floor 15); violet clears every gate against all three
(worst pair dE 16.3 normal, CVD 9.2 -- re-run 2026-09-29 before adding
series). Light surface only: these are paper figures and no dark variant is
rendered, which is the one gate violet would miss (dE 1.9 against the dark
blue step). Aqua sits below 3:1 against the surface; every series carries a
marker as well as a hue, and the per-corpus CSVs in
results/construct_methods are the table view that contrast warning requires.

No fifth hue validates all-pairs, so series beyond four are either faceted
out or drawn as a variant of an existing hue:

- minimax_sphere is minimax with a post-pass, so it wears minimax's aqua with
  a dotted line and a thin-diamond marker -- same family, visibly a variant.
- sphere_merge takes the orange slot.

Panel (a) is Euclidean cover volume, which the two sphere methods make
*worse* while their FPR improves: moving a center onto the data's sphere
grows its radius but shrinks the cap it cuts from the sphere, where every
query lies (exact_cover.sphere_centers). Read (a) as a geometry diagnostic,
not a proxy for (b).
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from plots.plot_blocklist import BLOCKLIST_LABELS
from plots.plot_common import apply_style, mean_std, save_figure, style_axes

DPI = 300
EMIT_PDF = False

METHOD_ORDER = ["gonzalez", "merge", "minimax", "minimax_sphere", "sphere_merge", "split"]
METHOD_LABELS = {
    "gonzalez": "Gonzalez (default)",
    "merge": "Bottom-up merge",
    "minimax": "Gonzalez + minimax search",
    "minimax_sphere": "Minimax, centers on sphere",
    "sphere_merge": "Weighted merge, on sphere",
    "split": "Top-down split",
}
METHOD_COLORS = {
    "gonzalez": "#52514e",
    "merge": "#2a78d6",
    "minimax": "#1baf7a",
    "minimax_sphere": "#1baf7a",
    "sphere_merge": "#eb6834",
    "split": "#4a3aa7",
}
# Shape as well as hue, so identity survives grayscale and CVD, and the
# low-contrast aqua line still reads.
METHOD_MARKERS = {"gonzalez": "o", "merge": "s",
                  "minimax": "D", "minimax_sphere": "d", "sphere_merge": "P", "split": "v"}
METHOD_LINESTYLES = {"gonzalez": (0, (3, 1.5)), "merge": "-",
                     "minimax": "-", "minimax_sphere": (0, (1, 1)), "sphere_merge": "-", "split": "-"}

_XLABEL = r"Ball budget $k$ (% of $|B|$)"


def load_results(results_dir: str) -> pd.DataFrame:
    paths = sorted(p for p in glob.glob(os.path.join(results_dir, "*.csv"))
                   if not os.path.basename(p).startswith("scale_"))
    if not paths:
        raise FileNotFoundError(
            f"No results in {results_dir!r}. Run main.generate_construct_methods first.")
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def _line(ax: plt.Axes, df: pd.DataFrame, value_col: str) -> None:
    agg = mean_std(df, ["method", "k_fraction"], value_col)
    for method in METHOD_ORDER:
        sub = agg[agg["method"] == method].sort_values("k_fraction")
        if sub.empty:
            continue
        x = 100.0 * sub["k_fraction"]
        ax.plot(x, sub["mean"], color=METHOD_COLORS[method], marker=METHOD_MARKERS[method],
                linestyle=METHOD_LINESTYLES[method], markersize=2.5, linewidth=0.8,
                label=METHOD_LABELS[method], zorder=2)
        if (sub["std"] > 0).any():
            ax.errorbar(x, sub["mean"], yerr=sub["std"], fmt="none", ecolor=METHOD_COLORS[method],
                        elinewidth=0.5, capsize=1.5, capthick=0.5, zorder=3)


def _volume_band(ax: plt.Axes, df: pd.DataFrame) -> None:
    low = df.groupby(["method", "k_fraction"])["log10_vol_low"].mean()
    high = df.groupby(["method", "k_fraction"])["log10_vol_high"].mean()
    for method in METHOD_ORDER:
        if method not in low.index.get_level_values(0):
            continue
        lo, hi = low.loc[method].sort_index(), high.loc[method].sort_index()
        ax.fill_between(100.0 * lo.index.values, lo.values, hi.values,
                        color=METHOD_COLORS[method], alpha=0.15, linewidth=0, zorder=1)


def plot_dataset(df: pd.DataFrame, dataset: str, outdir: str, dpi: int = DPI) -> None:
    apply_style(dpi)
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.0))
    ax_vol, ax_fpr, ax_time = axes

    _volume_band(ax_vol, df)
    _line(ax_vol, df, "log10_vol_high")
    ax_vol.set_ylabel(r"$\log_{10}$ cover volume")

    _line(ax_fpr, df, "fpr")
    ax_fpr.set_ylim(-0.05, 1.05)
    tau = float(df["tau"].iloc[0])
    ax_fpr.set_ylabel(rf"FPR at $\tau={tau:g}$")

    _line(ax_time, df, "method_s")
    ax_time.set_yscale("log")
    ax_time.set_ylabel("Construction time (s)")

    # 1-2-5 ticks only: on a log axis 10, 15 and 20 sit too close to label.
    ticks = [t for t in sorted(100.0 * df["k_fraction"].unique())
             if round(t / 10 ** np.floor(np.log10(t)), 6) in (1.0, 2.0, 5.0)]
    for ax, tag in zip(axes, "abc"):
        ax.set_xscale("log")
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:g}" for t in ticks], rotation=45)
        ax.minorticks_off()
        ax.set_xlabel(_XLABEL)
        ax.set_title(f"({tag})", loc="left")
        style_axes(ax)
        ax.grid(visible=False, axis="y")

    handles, labels = ax_vol.get_legend_handles_labels()
    n_ref = int(df["n_reference"].iloc[0])
    fig.suptitle(f"{BLOCKLIST_LABELS.get(dataset, dataset)}  (|B| = {n_ref:,}, "
                 f"{df['fold'].nunique()} folds)", fontsize=8.4, y=1.27)
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 3), bbox_to_anchor=(0.5, 1.19),
               frameon=False, fontsize=7.2)
    fig.subplots_adjust(wspace=0.55)
    save_figure(fig, f"construct-methods-{dataset}.png", outdir, dpi, EMIT_PDF)


def generate_construct_methods_figures(df: pd.DataFrame, outdir: str) -> None:
    if "quantize" in df:                      # one line per method: the unquantized covers
        df = df[df["quantize"] == "FP32"]
    for dataset, sub in df.groupby("dataset"):
        plot_dataset(sub, dataset, outdir)
