"""The paper's figures, built only from existing results/ CSVs
(src/paper.py generates exactly the data they need).

Nothing here runs a filter or a sweep. Outputs (PDF for the paper, PNG preview)
go to suggestions/figures/ under the names paper.tex includes:

    sweeps-grid.*                      fig:end-to-end   budget.csv + blocklist.csv
    quantization-saved-vs-memory.*     fig:quantization cap_quantization/<corpus>/*.csv
    quantization-speedup-vs-memory.*  PDF Figure 4    cap_quantization/<corpus>/*.csv
    quantization-throughput-vs-memory.* (supplement) cap_quantization/<corpus>/*.csv
    construction-fpr-vs-k.*            fig:construction construct_methods/<corpus>.csv
    streaming-fpr-vs-position.*        fig:streaming    streaming/<corpus>/streaming.csv
    equal-memory-zero-fn-fpr-vs-budget.*      fig:equal-memory          equal_memory/<corpus>/equal_memory.csv
    equal-memory-nonzero-fn-quora.*     PDF Figure 5    (same)
    throughput-improvement.*           fig:throughput   end_to_end.csv
(The tables of plots/tables_paper.py and plots/table_quantization.py are not drawn here.)

(figures/scheme.pdf is a hand-drawn diagram and is not produced here.)

Run from the repo root with the global Python 3.12:
    PYTHONPATH=src python src/plots/plot_paper.py [results-folder] [figure ...] [--out directory]
Use --paper-only to render data Figures 2–7 in the current paper.pdf.
Each run writes plot-manifest.json with input row counts, SHA-256 hashes,
output hashes, and the mapping from paper figure numbers to filenames.
results-folder defaults to results/. Figure names are the keys of FIGURES
(sweeps, quantization, construction, streaming, equal-memory-zero-fn,
equal-memory-nonzero-fn, throughput, quantization-throughput, quantization-speedup); default is all.

Thresholds are displayed as cosine distance; the stored CSV threshold values
are converted only for presentation.

The sweeps figure's FPR panels group the negatives by how related a query is
to the indexed set -- the cosine distance to its nearest construct vector:
closely related (<= 0.2), related (0.2-0.4, excluding 0.2), unrelated (> 0.4)
(runner.SIMILARITY_BANDS).

Runs may be in progress, so every figure tolerates missing corpora and cells:
a gap is drawn as a gap, never filled from a different subset.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plots.plot_common import TAU_LABEL, cosine_distance

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "results"
OUT_FIG = ROOT / "suggestions" / "figures"

DIM = 384
FP32_BYTES = 4

# Filter allocation relative to the raw FP32 embedding storage.
MEMORY_RATIO = r"M_{\mathrm{filter}}/M_{\mathrm{raw}}"
MEMORY_AXIS = f"${MEMORY_RATIO}$ (%)"
MEMORY_AXIS_LOG = MEMORY_AXIS + "\nlog scale"
MEMORY_FMT = matplotlib.ticker.FuncFormatter(lambda value, _: f"{100 * value:g}")

CORPORA = ["WildChat", "Quora", "WikiAnswers", "ChatGPTParaphrases", "CCNews"]
CORPUS_NAME = {
    "wildchat": "WildChat", "quora": "Quora", "wikianswers": "WikiAnswers",
    "chatgpt_paraphrases": "ChatGPTParaphrases", "chatgptparaphrases": "ChatGPTParaphrases",
    "ccnews": "CCNews",
}
CORPUS_LABEL = {
    "WildChat": "WildChat", "Quora": "Quora", "WikiAnswers": "WikiAnswers",
    "ChatGPTParaphrases": "ChatGPT-Para.", "CCNews": "CC-News",
}

# Categorical palette; colors follow the entity within a figure, never its rank.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
)
PALETTE = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<"]
INK, INK_2, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#8a8983", "#e4e3de"

CORPUS_STYLE = {
    "WildChat": (BLUE, "o"), "Quora": (ORANGE, "s"), "WikiAnswers": (AQUA, "^"),
    "ChatGPTParaphrases": (VIOLET, "D"), "CCNews": (MAGENTA, "v"),
}

# Keep automatic selection distinct from explicitly fixed encodings.
CAP_COVER = "CapCover Filter"
CAP_COVER_TQ2 = "CapCover Filter (TQ2)"
CAP_COVER_TQ4 = "CapCover Filter (TQ4)"
METHOD_STYLE = {
    CAP_COVER: (BLUE, "o", "-"),
    CAP_COVER_TQ2: (BLUE, "o", "-"),
    CAP_COVER_TQ4: (BLUE, "s", "--"),
    "QuantizedVector": (ORANGE, "s", "-"),
    "IVF-PQ": (AQUA, "^", "-"),
    "IVF-Radii": (INK_2, "h", "-"),
    "HNSW": (YELLOW, "D", "-"),
    "DSH": (VIOLET, "v", "--"),
    "MLP": (RED, "x", ":"),
    "Zero-FN LSH": (GREEN, "P", "--"),
    "VA+-file": (MAGENTA, "*", "-"),
}
CONFIG_TO_METHOD = {
    "CapCoverFilter": CAP_COVER, "CapCoverFilter (TQ2)": CAP_COVER_TQ2, "CapCoverFilter (TQ4)": CAP_COVER_TQ4,
    "QuantizedVectorFilter": "QuantizedVector",
    "IVFFilter": "IVF-PQ", "IVFRadiiFilter": "IVF-Radii", "HNSWFilter": "HNSW",
    "DSHFilter": "DSH", "MLPFilter": "MLP",
    "PacukLSHFilter": "Zero-FN LSH", "SankowskiWygockiFilter": "Zero-FN LSH",
    "VAPlusFilter": "VA+-file",
}
# Never drawn, so src/paper.py does not run them.
EQUAL_MEMORY_HIDDEN = {"IVF-Radii"}

def use_corpora(names):
    """Draw these corpora (display names) instead of the paper's five: for a
    run on whatever corpora a machine has (src/paper.py --datasets)."""
    CORPORA[:] = names
    for i, name in enumerate(names):
        CORPUS_NAME[name.lower()] = name
        CORPUS_LABEL.setdefault(name, name)
        CORPUS_STYLE[name] = (PALETTE[i % len(PALETTE)], MARKERS[i % len(MARKERS)])   # distinct within this run


QUANT_TOP_N = 5

# End-to-end throughput (src/paper.py generates exactly these cells): the filter
# in front of a flat index vs the index alone. The tau panel fixes the budget,
# the memory panel fixes tau; the budget sweep is shared by the sweeps figure.
PAPER_TAU_COS = 0.05                                          # cosine distance
PAPER_TAUS_COS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2)    # the tau sweeps (8 points), same unit
TAU_TICKS = [0.02, 0.05, 0.1, 0.2]                            # cosine distance ticks of the tau axes


def l2_threshold(cosine_dist):
    """The L2 threshold of unit vectors at this cosine distance (the filters and CSVs use L2)."""
    return float(np.sqrt(2.0 * cosine_dist))


E2E_TAU = l2_threshold(PAPER_TAU_COS)
# Memory budgets are B/M: the filter's bytes over the raw index (float32 vectors). The values
# must equal main.FIXED_MEMORY_FRACTION and main.BUDGET_MEMORY_FRACTIONS (src/paper.py asserts it).
E2E_MEMORY_FRACTION = 0.02
E2E_TAUS = tuple(l2_threshold(t) for t in PAPER_TAUS_COS)
E2E_MEMORY_FRACTIONS = (0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05)

# Construction methods shown with full-precision directions (PDF Figure 6).
CONSTRUCT_SHOWN = {
    "gonzalez": ("Gonzalez", INK_MUTED, "o"),
    "minimax": ("minimax", ORANGE, "s"),
    "soft_anneal": ("soft anneal", GREEN, "v"),
}
# Insertion strategies shown. The initial cover is built with STREAM_BOOTSTRAP.
STREAM_BOOTSTRAP = "sphere_merge"
INSERT_SHOWN = {
    "nearest": ("nearest", ORANGE, "s", "-"),
    "cap_merge": ("cap merge", BLUE, "D", "-"),
    "static_rebuild": ("static rebuild (oracle)", INK, None, "--"),
}
# FPR panel per cosine-distance band (the stored band columns are unchanged).
BAND_TITLE = {
    "close": "Nearest cosine\ndistance to $X$\n$\\leq0.2$",
    "related": "Nearest cosine\ndistance to $X$\n$(0.2,0.4]$",
    "unrelated": "Nearest cosine\ndistance to $X$\n$>0.4$",
}

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": INK_MUTED, "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.5, "lines.markersize": 4,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    "savefig.dpi": 200, "pdf.fonttype": 42,
})

# Layout, in inches, shared by every figure. Panels are square; gaps depend only
# on whether neighbours share an axis. Axis labels and legends fall outside the
# grid and are kept by the tight bounding box.
PANEL = 1.2
GAP_SHARED = 0.22
GAP_LABELLED = 0.85
MARGIN_TOP = 0.48

PCT_FMT = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")


def save(fig, name):
    _readable_axes(fig)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG / f"{name}.pdf")
    fig.savefig(OUT_FIG / f"{name}.png")
    plt.close(fig)
    print(f"  {OUT_FIG / name}.pdf")


def _readable_axes(fig):
    """Keep tick text clear of neighbouring ticks, axes, and axis titles."""
    for ax in fig.axes:
        ax.tick_params(axis="both", which="major", pad=4)
        ax.xaxis.labelpad = 7
        ax.yaxis.labelpad = 7
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    # Prune crowded x ticks using rendered extents, preserving the end ticks
    # and their original formatting (percentages, ms, etc.).
    for ax in fig.axes:
        for _ in range(12):
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            lo, hi = sorted(ax.get_xlim())
            ticks = [(loc, label) for loc, label in zip(ax.get_xticks(), ax.get_xticklabels())
                     if lo <= loc <= hi and label.get_visible() and label.get_text()]
            collision = next((i for i in range(len(ticks) - 1)
                              if ticks[i][1].get_window_extent(renderer).x1 + 5
                              > ticks[i + 1][1].get_window_extent(renderer).x0), None)
            if collision is None or len(ticks) <= 2:
                break
            remove = collision + 1 if collision + 1 < len(ticks) - 1 else collision
            ticks.pop(remove)
            ax.set_xticks([loc for loc, _ in ticks])
            ax.set_xticklabels([label.get_text() for _, label in ticks])


def fig_legend(fig, handles, labels, ncol):
    """Legend above the panels, filled row by row (matplotlib fills columns)."""
    order = sorted(range(len(labels)), key=lambda i: (i % ncol, i // ncol))
    handles, labels = [handles[i] for i in order], [labels[i] for i in order]
    fig.legend(handles, labels, loc="lower center", ncol=ncol,
               bbox_to_anchor=(0.5, 1.0), borderaxespad=0, handlelength=2.2,
               columnspacing=1.2)


def mark_empty(ax):
    ax.text(0.5, 0.5, "no data yet", transform=ax.transAxes, ha="center",
            va="center", color=INK_MUTED, fontsize=7)


def corpus_panels(nrows=1, sharex=False, sharey=False):
    """One square panel per corpus (columns), `nrows` rows."""
    ncols = len(CORPORA)
    wgap = GAP_SHARED if sharey else GAP_LABELLED
    hgap = GAP_SHARED if sharex else GAP_LABELLED
    width = ncols * PANEL + (ncols - 1) * wgap
    height = nrows * PANEL + (nrows - 1) * hgap + MARGIN_TOP
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(width, height), squeeze=False, sharex=sharex, sharey=sharey,
        gridspec_kw=dict(left=0, right=1, bottom=0, top=1 - MARGIN_TOP / height,
                         wspace=wgap / PANEL, hspace=hgap / PANEL))
    for ax in axes.flat:
        ax.set_box_aspect(1)
    for j, corpus in enumerate(CORPORA):
        axes[0, j].set_title(CORPUS_LABEL[corpus], color=INK)
    return fig, axes


def shared_xlabel(axes, text):
    """One x label for a corpus grid, under the middle bottom panel."""
    axes[-1, axes.shape[1] // 2].set_xlabel(text)


def to_corpus(series):
    return series.map(lambda x: CORPUS_NAME.get(str(x).lower(), x))


def read_all(pattern):
    paths = sorted(RESULTS.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no files match {RESULTS / pattern}")
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


# ---------------------------------------------------------------------------
# fig:end-to-end  sweeps-grid
# ---------------------------------------------------------------------------

def sweeps_grid():
    """Rows: ball-budget sweep (tau=0.1, budget.csv) and tau sweep (fixed k,
    blocklist.csv). Columns: FPR per query source, then accesses saved with
    each corpus's ceiling (its share of negative queries) as the only dashed
    line. Error bars: +-1 std over folds. thresholds are shown as cosine distance."""
    frames = []
    for name in ("budget", "blocklist"):
        d = pd.read_csv(RESULTS / f"{name}.csv")
        d["corpus"] = to_corpus(d.dataset)
        frames.append(d)
    budget, blocklist = frames
    blocklist["tau_cos"] = cosine_distance(blocklist.tau)
    tau_mem = float(blocklist.memory_fraction.iloc[0])
    budget_tau = float(cosine_distance(budget.tau.iloc[0]))
    sweeps = [(budget, "memory_fraction", MEMORY_AXIS, f"FPR\n($\\tau={budget_tau:g}$)", MEMORY_FMT),
              (blocklist, "tau_cos", TAU_LABEL, f"FPR\n(${MEMORY_RATIO}={100 * tau_mem:g}\\%$)", None)]

    n_fpr, nrows = len(BAND_TITLE), len(sweeps)
    spacer = GAP_LABELLED - 2 * GAP_SHARED  # blank column before the accesses-saved panel
    width = (n_fpr + 1) * PANEL + spacer + (n_fpr + 1) * GAP_SHARED
    title_margin = MARGIN_TOP + 0.14  # one extra line for explicit cosine-distance units
    height = nrows * PANEL + (nrows - 1) * GAP_LABELLED + title_margin
    fig, grid = plt.subplots(
        nrows, n_fpr + 2, figsize=(width, height), squeeze=False,
        gridspec_kw=dict(left=0, right=1, bottom=0, top=1 - title_margin / height,
                         width_ratios=[1] * n_fpr + [spacer / PANEL, 1],
                         wspace=GAP_SHARED / PANEL, hspace=GAP_LABELLED / PANEL))
    for ax in grid[:, n_fpr]:
        ax.remove()
    axes = np.concatenate([grid[:, :n_fpr], grid[:, n_fpr + 1:]], axis=1)
    for i in range(nrows):
        for j, ax in enumerate(axes[i]):
            ax.set_box_aspect(1)
            if 0 < j < n_fpr:
                ax.sharey(axes[i, 0])
                ax.tick_params(labelleft=False)

    errkw = dict(ms=3, capsize=1.5, elinewidth=0.8)
    for i, (df, x, xlabel, row_label, xfmt) in enumerate(sweeps):
        agg = (df.groupby(["corpus", x])
               .agg(**{f"fpr_{s}": (f"fpr_{s}", "mean") for s in BAND_TITLE},
                    **{f"fpr_{s}_sd": (f"fpr_{s}", "std") for s in BAND_TITLE},
                    saved=("accesses_saved_pct", "mean"), saved_sd=("accesses_saved_pct", "std"),
                    ceiling=("neg_share_pct", "mean"))
               .reset_index())
        for corpus in CORPORA:
            g = agg[agg.corpus == corpus].sort_values(x)
            if g.empty:
                continue
            color, marker = CORPUS_STYLE[corpus]
            for j, s in enumerate(BAND_TITLE):
                axes[i, j].errorbar(g[x], g[f"fpr_{s}"], yerr=g[f"fpr_{s}_sd"].fillna(0),
                                    color=color, marker=marker, **errkw)
            axes[i, -1].errorbar(g[x], g.saved, yerr=g.saved_sd.fillna(0), color=color,
                                 marker=marker, **errkw)
            axes[i, -1].plot(g[x], g.ceiling, color=color, ls="--", lw=0.7, alpha=0.6, zorder=1)
        for j, ax in enumerate(axes[i]):
            if j in (n_fpr // 2, n_fpr):  # x label under the middle FPR panel and accesses saved
                ax.set_xlabel(xlabel)
            if xfmt:
                ax.xaxis.set_major_formatter(xfmt)
            if x == "tau_cos":
                ax.set_xlim(-0.005, 0.205)
                ax.set_xticks(TAU_TICKS)
                ax.xaxis.set_major_formatter(PCT_FMT)
            if j < n_fpr:
                ax.set_ylim(-0.03, 1.05)
            else:
                ax.set_ylim(-2, 102)
                ax.set_ylabel("index accesses saved (%)")
        axes[i, 0].set_ylabel(row_label)
    for j, title in enumerate(list(BAND_TITLE.values()) + ["accesses saved"]):
        axes[0, j].set_title(title, color=INK)

    present = [c for c in CORPORA if c in set(budget.corpus) | set(blocklist.corpus)]
    handles = [plt.Line2D([], [], color=CORPUS_STYLE[c][0], marker=CORPUS_STYLE[c][1])
               for c in present]
    handles.append(plt.Line2D([], [], color=INK_2, ls="--", lw=0.7))
    labels = [CORPUS_LABEL[c] for c in present] + ["ceiling (share of negative queries)"]
    fig_legend(fig, handles, labels, ncol=len(labels))
    save(fig, "sweeps-grid")


# ---------------------------------------------------------------------------
# fig:quantization  quantization-saved-vs-memory
# ---------------------------------------------------------------------------

def quantization(mode="saved"):
    """Per corpus, the QUANT_TOP_N encodings (ranked by accesses saved) against the budget:
    accesses saved (%); mode="throughput": the median query throughput (qps, log); mode="speedup":
    the simulated end-to-end improvement (%) of the filter in front of a flat index, by the cost
    model of throughput(): L / (t_f + p L) - 1, with t_f = 1 / query throughput, p the share of
    queries reaching the index, and L the flat-index lookup cost measured in end_to_end.csv."""
    throughput, speedup = mode == "throughput", mode == "speedup"
    q = read_all("cap_quantization/*/*.csv")
    q["corpus"] = to_corpus(q.dataset)
    # The budget is B/M; compact encodings fit more caps in it, up to one per vector.
    q["budget_pct"] = q.memory_fraction
    q["singleton"] = q.n_caps >= q.n_construct
    if speedup:
        e2e = pd.read_csv(RESULTS / "end_to_end.csv")
        lookup = float((1.0 / e2e.throughput_without_qps).mean())
        p = np.maximum(q.n_index_accesses / q.n_queries, 0.5 / q.n_queries)
        q["improvement"] = 100.0 * (lookup / (1.0 / q.query_throughput_qps + p * lookup) - 1.0)
        print(f"  simulated speedup at a flat-index lookup of {lookup * 1e3:.1f} ms")
    agg = (q.groupby(["corpus", "quantize", "budget_pct"])
           .agg(saved=("index_accesses_saved_pct", "mean"), qps=("query_throughput_qps", "median"), gain=("improvement" if speedup else "index_accesses_saved_pct", "mean"),
                singleton=("singleton", "mean"))
           .reset_index())
    ceilings = q.groupby("corpus").neg_share_pct.median()

    # Only the QUANT_TOP_N best encodings: lowest mean rank by accesses saved
    # within each corpus x budget cell (a missing cell does not count against
    # an encoding), ties broken by mean accesses saved.
    # auto (the paper's configuration) is always shown and does not take part in the ranking.
    cells = agg[agg.quantize != "auto"].pivot_table(index=["corpus", "budget_pct"], columns="quantize",
                                                   values="saved")
    score = pd.DataFrame({"rank": cells.rank(axis=1, ascending=False).mean(), "saved": cells.mean()})
    top = list(score.sort_values(["rank", "saved"], ascending=[True, False]).index[:QUANT_TOP_N])
    if (agg.quantize == "auto").any():
        top.append("auto")
        score.loc["auto", "rank"] = float("nan")
    top_style = {enc: (PALETTE[i], MARKERS[i], "-") for i, enc in enumerate(top)}
    print("  quantization shows:", ", ".join(f"{e} (rank {score.loc[e, 'rank']:.2f})" for e in top))

    fig, axes = corpus_panels(1, sharex=True, sharey=True)
    for ax, corpus in zip(axes[0], CORPORA):
        d = agg[agg.corpus == corpus]
        if d.empty:
            mark_empty(ax)
        for enc, (color, marker, ls) in top_style.items():
            g = d[d.quantize == enc].sort_values("budget_pct")
            if g.empty:
                continue
            y = g.qps if throughput else g.gain if speedup else g.saved
            ax.plot(g.budget_pct, y, color=color, ls=ls, lw=1.2, zorder=2)
            full = g.singleton >= 0.5
            ax.scatter(g.budget_pct[~full], y[~full], color=color, marker=marker,
                       s=12, edgecolor="white", lw=0.5, zorder=3)
            ax.scatter(g.budget_pct[full], y[full], facecolor="white",
                       edgecolor=color, marker=marker, s=12, lw=0.9, zorder=3)
        if corpus in ceilings and mode == "saved":
            ax.axhline(ceilings[corpus], color=INK_MUTED, ls="--", lw=0.8)
            ax.text(0.97, 0.04, f"ceiling {ceilings[corpus]:.0f}%", color=INK_2, fontsize=6.5,
                    ha="right", transform=ax.transAxes)
        ax.set_xscale("log", base=2)
        all_ticks = sorted(q.budget_pct.unique())
        ticks = [all_ticks[i] for i in sorted({0, len(all_ticks) // 2, len(all_ticks) - 1})]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{100 * t:.2g}" for t in ticks], fontsize=6)
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    if throughput:
        axes[0, 0].set_yscale("log")
        axes[0, 0].set_ylabel("query throughput (qps, log)")
    elif speedup:
        axes[0, 0].set_yscale("symlog", linthresh=Y_LINTHRESH)
        axes[0, 0].axhline(0, color=INK_MUTED, lw=0.8)
        axes[0, 0].set_ylabel("modeled throughput\nimprovement (%)")
    else:
        axes[0, 0].set_ylabel("index accesses saved (%)")
        axes[0, 0].set_ylim(-2, 100)
    handles = [plt.Line2D([], [], color=c, marker=mk, ls=ls) for c, mk, ls in top_style.values()]
    handles.append(plt.Line2D([], [], color=INK_2, marker="o", mfc="white", ls=""))
    spq_labels = {"SPQ12": r"SPQ $48\times12$", "SPQ16": r"SPQ $24\times8$",
                  "SPQ24": r"SPQ $24\times12$", "SPQ36": r"SPQ $16\times12$",
                  "SPQ48": r"SPQ $12\times12$", "SPQ64": r"SPQ $6\times8$"}
    labels = [spq_labels.get(enc, enc) for enc in top] + ["one cap per vector"]
    fig_legend(fig, handles, labels, ncol=len(labels))
    shared_xlabel(axes, MEMORY_AXIS_LOG)
    save(fig, {"saved": "quantization-saved-vs-memory", "throughput": "quantization-throughput-vs-memory",
               "speedup": "quantization-speedup-vs-memory"}[mode])


def quantization_throughput():
    quantization("throughput")


def quantization_speedup():
    quantization("speedup")


# ---------------------------------------------------------------------------
# fig:construction  construction-fpr-vs-k
# ---------------------------------------------------------------------------

def construction():
    """PDF Figure 6: the three constructions with full-precision directions."""
    c = read_all("construct_methods/*.csv")
    c = c[~c.dataset.astype(str).str.startswith("scale_")].copy()
    c["corpus"] = to_corpus(c.dataset)
    if "quantize" not in c:                  # runs from before the encodings were swept were unquantized
        c["quantize"] = "FP32"
    c = c[c["quantize"] == "FP32"]
    if c.empty:
        raise ValueError("no full-precision construction rows for the paper's construction figure")
    agg = c.groupby(["corpus", "method", "quantize", "memory_fraction"]).agg(
        mean=("fpr", "mean"), std=("fpr", "std")).reset_index()
    fig, axes = corpus_panels(sharey=True)
    for ax, corpus in zip(axes[0], CORPORA):
        d = agg[agg.corpus == corpus]
        if d.empty:
            mark_empty(ax)
        for method, (_, color, marker) in CONSTRUCT_SHOWN.items():
            g = d[d.method == method].sort_values("memory_fraction")
            ax.errorbar(g.memory_fraction, g["mean"], yerr=g["std"].fillna(0), color=color,
                        marker=marker, ls="-", ms=3, capsize=1.2, elinewidth=0.7)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(MEMORY_FMT)
    axes[0, 0].set_ylabel("FPR")
    axes[0, 0].set_ylim(-0.03, 1.05)
    handles, labels = [], []
    for _, (name, color, marker) in CONSTRUCT_SHOWN.items():
        handles.append(plt.Line2D([], [], color=color, marker=marker, ls="-", ms=3))
        labels.append(name)
    fig_legend(fig, handles, labels, ncol=len(labels))
    axes[0, len(CORPORA) // 2].set_title(
        CORPUS_LABEL[CORPORA[len(CORPORA) // 2]]
        + f"\n$\\tau={float(cosine_distance(c.tau.iloc[0])):g}$ (unquantized)", color=INK)
    shared_xlabel(axes, MEMORY_AXIS_LOG)
    save(fig, "construction-fpr-vs-k")


# ---------------------------------------------------------------------------
# fig:streaming  streaming-fpr-vs-position
# ---------------------------------------------------------------------------

def streaming():
    s = read_all("streaming/*/streaming.csv")
    s["corpus"] = to_corpus(s.dataset)
    pos = s[(s.sweep == "position") & (s.construct_method == STREAM_BOOTSTRAP)]
    if pos.empty:
        raise ValueError(f"no position-sweep rows bootstrapped with {STREAM_BOOTSTRAP}")
    # The position sweep inserts the final fifth of the indexed set.
    n_final = int(pos.position.max()) * 5
    budget_fraction = float(pos.budget_bytes.iloc[0]) / (n_final * DIM * FP32_BYTES)
    shown = [m for m in INSERT_SHOWN if m in set(pos.method)]
    agg = pos.groupby(["corpus", "method", "position"]).fpr.agg(["mean", "std"]).reset_index()
    fig, axes = corpus_panels(sharex=True, sharey=True)
    for ax, corpus in zip(axes[0], CORPORA):
        d = agg[agg.corpus == corpus]
        if d.empty:
            mark_empty(ax)
        for method in shown:
            _, color, marker, ls = INSERT_SHOWN[method]
            g = d[d.method == method].sort_values("position")
            if g.empty:
                continue
            ax.plot(g.position / 1000, g["mean"], color=color, marker=marker, ls=ls, ms=2.5,
                    lw=1.2 if method == "static_rebuild" else 1.5)
            ax.fill_between(g.position / 1000, g["mean"] - g["std"].fillna(0),
                            g["mean"] + g["std"].fillna(0), color=color, alpha=0.12, lw=0)
    axes[0, 0].set_ylabel(f"FPR\n({STREAM_BOOTSTRAP.replace('_', ' ')} boot.)")
    axes[0, 0].set_ylim(-0.03, 1.05)
    handles = [plt.Line2D([], [], color=INSERT_SHOWN[m][1], marker=INSERT_SHOWN[m][2],
                          ls=INSERT_SHOWN[m][3]) for m in shown]
    fig_legend(fig, handles, [INSERT_SHOWN[m][0] for m in shown], ncol=len(shown))
    shared_xlabel(axes, f"inserts after bootstrap (thousands), $B\\approx{budget_fraction:.3g}M$")
    save(fig, "streaming-fpr-vs-position")


# ---------------------------------------------------------------------------
# fig:equal-memory  equal-memory-zero-fn-fpr-vs-budget
# fig:equal-memory-nonzero-fn  equal-memory-nonzero-fn-quora
# ---------------------------------------------------------------------------

# The methods each figure draws, in legend order. A method the run did not produce is skipped.
ZERO_FN_METHODS = [CAP_COVER, CAP_COVER_TQ2, CAP_COVER_TQ4, "VA+-file", "Zero-FN LSH", "QuantizedVector"]
NONZERO_FN_METHODS = [CAP_COVER, CAP_COVER_TQ2, CAP_COVER_TQ4, "DSH", "MLP", "IVF-PQ", "HNSW"]
NONZERO_FN_CORPUS = "Quora"      # the corpus with the most positive queries in the reduced run (49 at delta 0.02)


def _equal_memory_frame():
    e = read_all("equal_memory/*/equal_memory.csv")
    e = e[~e.Infeasible.astype(bool)].copy()
    e["Dataset"] = to_corpus(e.Dataset)
    e["pct_raw"] = e["Memory Budget"] / (e["Construct Size"] * DIM * FP32_BYTES)
    e["method"] = e.Config.map(CONFIG_TO_METHOD)
    # Average the repeated folds before selecting between the two LSH schemes.
    # The CSV has no fold identifier; dropping duplicates here selected a best fold.
    e = e.groupby(["Config", "method", "Dataset", "Memory Budget", "pct_raw"], as_index=False).agg(
        FPR=("FPR", "mean"), FNR=("FNR", "mean"))
    return e.sort_values("FPR").drop_duplicates(["method", "Dataset", "Memory Budget"], keep="first")


def _method_handles(e, methods):
    styles = {m: METHOD_STYLE[m] for m in methods if m in set(e.method)}
    return styles, [plt.Line2D([], [], color=c, marker=mk, ls=ls) for c, mk, ls in styles.values()]


def equal_memory_zero_fn():
    """FPR against memory of the filters that never miss a member: one column per corpus."""
    e = _equal_memory_frame()
    styles, handles = _method_handles(e, ZERO_FN_METHODS)
    fig, axes = corpus_panels(1, sharex=True, sharey=True)
    for ax, corpus in zip(axes[0], CORPORA):
        d = e[e.Dataset == corpus]
        if d.empty:
            mark_empty(ax)
        for method, (color, marker, ls) in styles.items():
            g = d[d.method == method].sort_values("pct_raw")
            if not g.empty:
                ax.plot(g.pct_raw, g.FPR, color=color, marker=marker, ls=ls, ms=3)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(MEMORY_FMT)
    axes[0, 0].set_ylabel("FPR")
    axes[0, 0].set_ylim(-0.03, 1.05)
    fig_legend(fig, handles, list(styles), ncol=min(4, len(styles)))
    shared_xlabel(axes, MEMORY_AXIS_LOG)
    save(fig, "equal-memory-zero-fn-fpr-vs-budget")


def equal_memory_nonzero_fn():
    """FPR (left) and FNR (right) of the filters that miss members, on one corpus, with the
    Cap-Cover as the zero-FN reference."""
    e = _equal_memory_frame()
    corpus = NONZERO_FN_CORPUS if NONZERO_FN_CORPUS in CORPORA else CORPORA[0]
    d = e[e.Dataset == corpus]
    styles, handles = _method_handles(e, NONZERO_FN_METHODS)
    size = PANEL  # this two-panel figure occupies one paper column
    fig, axes = plt.subplots(1, 2, figsize=(2 * size + GAP_LABELLED, size + MARGIN_TOP), squeeze=False,
                             gridspec_kw=dict(left=0, right=1, bottom=0, wspace=GAP_LABELLED / size))
    for ax, column, label in zip(axes[0], ("FPR", "FNR"), ("FPR", "FNR")):
        ax.set_box_aspect(1)
        if d.empty:
            mark_empty(ax)
        for method, (color, marker, ls) in styles.items():
            g = d[d.method == method].sort_values("pct_raw")
            if not g.empty:
                ax.plot(g.pct_raw, g[column], color=color, marker=marker, ls=ls, ms=3)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(MEMORY_FMT)
        ax.set_ylabel(label)
        ax.set_ylim(-0.03, 1.05)
        ax.set_xlabel(MEMORY_AXIS_LOG)
    fig_legend(fig, handles, list(styles), ncol=min(3, len(styles)))
    save(fig, f"equal-memory-nonzero-fn-{corpus.lower()}")


# ---------------------------------------------------------------------------
# fig:throughput  throughput-improvement
# ---------------------------------------------------------------------------

Y_LINTHRESH = 10     # improvement (%) below which the throughput panels' y axis is linear
Y_TOP = 1e4
BACKEND_COSTS = np.logspace(-4, 0, 60)      # seconds per lookup: 0.1 ms ... 1 s
BACKEND_TICKS = {1e-4: "100 µs", 1e-3: "1 ms", 1e-2: "10 ms", 1e-1: "100 ms", 1.0: "1 s"}


def _backend_cost_panel(ax, d):
    """Improvement (%) against the cost L of one backend lookup, from each row's
    measured t_f and p (see throughput()); mean over folds, one marker per
    corpus at its measured flat-index cost. A fold that accepts nothing has p
    floored at half a query, since the improvement is unbounded as p -> 0."""
    p = np.maximum(d.positive_rate_pct / 100.0, 0.5 / d.n_queries)
    l0 = 1.0 / d.throughput_without_qps
    t_f = (1.0 / d.throughput_with_qps - p * l0).clip(lower=0.0)
    d = d.assign(p=p, l0=l0, t_f=t_f)
    for corpus in CORPORA:
        g = d[d.corpus == corpus]
        if g.empty:
            continue
        color, marker = CORPUS_STYLE[corpus]
        curves = np.array([100.0 * (BACKEND_COSTS / (r.t_f + r.p * BACKEND_COSTS) - 1.0) for r in g.itertuples()])
        ax.plot(BACKEND_COSTS, curves.mean(axis=0), color=color, lw=1.2)
        ax.plot(g.l0.mean(), g.throughput_improvement_pct.mean(), color=color, marker=marker, ms=5,
                ls="", markeredgecolor="white", markeredgewidth=0.5, zorder=3)
    ax.axhline(0, color=INK_MUTED, lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=Y_LINTHRESH)
    ax.set_xticks(list(BACKEND_TICKS))
    ax.set_xticklabels(list(BACKEND_TICKS.values()), fontsize=6, rotation=30, ha="right")
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def throughput():
    """End-to-end throughput improvement (%) of the filter in front of a flat
    index over the index alone, one line per corpus: against cosine
    distance (budget fixed), against the memory the filter uses (tau fixed),
    and against the cost of one backend lookup (both fixed).

    The last panel is the cost model behind the other two. Per query the index
    alone costs L and the filter plus index costs t_f + p L, with p the share
    of queries the filter accepts, so the improvement is L / (t_f + p L) - 1.
    L0 (the measured flat-index lookup) and t_f come from the same run
    (L0 = 1 / qps_without, t_f = 1 / qps_with - p L0); the curve then moves L to
    costlier backends and the marker is the measurement. Error bars: +-1 std
    over folds (not drawn on the model curve)."""
    d = pd.read_csv(RESULTS / "end_to_end.csv")
    d["corpus"] = to_corpus(d.dataset)
    d["tau_cos"] = cosine_distance(d.tau)
    # Older runs sweep cap count, newer ones a byte budget. Keep the actual
    # stored footprint separate from the sweep key so folds average correctly.
    legacy = "memory_fraction" not in d
    d["budget"] = d.k_fraction if legacy else d.memory_fraction
    if "memory_bytes" in d:
        d["memory_ratio"] = d.memory_bytes / (d.n_reference * DIM * FP32_BYTES)
    else:
        d["memory_ratio"] = d.memory_fraction
    fixed_budget = 0.1 if legacy else E2E_MEMORY_FRACTION
    at_budget = d[np.isclose(d.budget, fixed_budget)]
    if at_budget.empty:
        raise ValueError(f"no throughput rows at fixed budget {fixed_budget:g}")
    # Keep both fixed-threshold panels at the requested threshold, even when
    # the available data contains only one measured budget per corpus.
    sweep_tau = E2E_TAU
    at_tau = d[np.isclose(d.tau, sweep_tau)]
    fixed_tau = f"fixed $\\tau={float(cosine_distance(E2E_TAU)):g}$"
    # A fixed cap budget can have different stored footprints across corpora.
    # Show the measured memory range rather than relabelling a cap fraction.
    memory_low = f"{100 * at_budget.memory_ratio.min():.2f}"
    memory_high = f"{100 * at_budget.memory_ratio.max():.2f}"
    memory_value = memory_low if memory_low == memory_high else f"{memory_low}–{memory_high}"
    fixed_memory = f"${MEMORY_RATIO}$\n${memory_value}\\%$"
    panels = [
        (at_budget, "tau_cos", TAU_LABEL, fixed_memory, False),
        (at_tau, "memory_ratio", MEMORY_AXIS_LOG,
         f"fixed $\\tau={float(cosine_distance(sweep_tau)):g}$", True),
        (None, "backend", "backend lookup cost $L$", fixed_tau + "\n" + fixed_memory, True),
    ]
    width = len(panels) * PANEL + (len(panels) - 1) * GAP_LABELLED
    height = PANEL + MARGIN_TOP
    fig, axes = plt.subplots(1, len(panels), figsize=(width, height), squeeze=False,
                             gridspec_kw=dict(left=0, right=1, bottom=0, top=1 - MARGIN_TOP / height,
                                              wspace=GAP_LABELLED / PANEL))
    for ax, (df, x, xlabel, title, logx) in zip(axes[0], panels):
        ax.set_box_aspect(1)
        if x == "backend":
            _backend_cost_panel(ax, at_budget[np.isclose(at_budget.tau, E2E_TAU)])
            ax.set_xlabel(xlabel)
            ax.set_title(title, color=INK)
            continue
        agg = (df.groupby(["corpus", "budget", "tau"])
               .agg(x=(x, "mean"), y=("throughput_improvement_pct", "mean"),
                    sd=("throughput_improvement_pct", "std"))
               .reset_index())
        if agg.empty:
            mark_empty(ax)
        for corpus in CORPORA:
            g = agg[agg.corpus == corpus].sort_values("x")
            if g.empty:
                continue
            color, marker = CORPUS_STYLE[corpus]
            ax.errorbar(g.x, g.y, yerr=g.sd.fillna(0), color=color, marker=marker, ms=3,
                        capsize=1.5, elinewidth=0.8)
        ax.axhline(0, color=INK_MUTED, lw=0.8)
        ax.set_yscale("symlog", linthresh=Y_LINTHRESH)
        if logx:
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(MEMORY_FMT)
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        if x == "tau_cos":
            ax.set_xlim(agg.x.min() - 0.005, agg.x.max() + 0.005)
            ax.set_xticks(TAU_TICKS)
            ax.xaxis.set_major_formatter(PCT_FMT)
        if x == "memory_ratio":
            ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1, 2, 5)))
        ax.set_xlabel(xlabel)
        ax.set_title(title, color=INK)
    # One y range for all panels, so the zero line sits at the same height.
    for ax in axes[0]:
        ax.set_ylim(-Y_LINTHRESH, Y_TOP)
    axes[0, 0].set_ylabel("throughput improvement (%)")
    present = [c for c in CORPORA if c in set(to_corpus(d.dataset))]
    handles = [plt.Line2D([], [], color=CORPUS_STYLE[c][0], marker=CORPUS_STYLE[c][1]) for c in present]
    fig_legend(fig, handles, [CORPUS_LABEL[c] for c in present], ncol=len(present))
    save(fig, "throughput-improvement")


def quantization_table():
    """tab:quantization, as LaTeX (plots/table_quantization.py)."""
    from plots.table_quantization import write
    write(RESULTS, OUT_FIG / "quantization-table.tex")


def paper_tables():
    """tab:ablation, tab:sensitivity, tab:latency, tab:timing and the J_tau correlation
    (plots/tables_paper.py)."""
    from plots.tables_paper import write_all
    write_all(RESULTS, OUT_FIG)


FIGURES = {
    "sweeps": sweeps_grid,
    "quantization": quantization,
    "quantization-throughput": quantization_throughput,
    "quantization-speedup": quantization_speedup,
    "construction": construction,
    "streaming": streaming,
    "equal-memory-zero-fn": equal_memory_zero_fn,
    "equal-memory-nonzero-fn": equal_memory_nonzero_fn,
    "throughput": throughput,
}

# Data figures in the current paper.pdf; Figure 1 is the hand-drawn scheme.
PAPER_FIGURES = ("throughput", "sweeps", "quantization-speedup", "equal-memory-nonzero-fn",
                 "construction", "equal-memory-zero-fn")

FIGURE_FILES = {
    "throughput": "throughput-improvement",
    "sweeps": "sweeps-grid",
    "quantization-speedup": "quantization-speedup-vs-memory",
    "quantization-throughput": "quantization-throughput-vs-memory",
    "quantization": "quantization-saved-vs-memory",
    "equal-memory-nonzero-fn": "equal-memory-nonzero-fn-quora",
    "equal-memory-zero-fn": "equal-memory-zero-fn-fpr-vs-budget",
    "construction": "construction-fpr-vs-k",
    "streaming": "streaming-fpr-vs-position",
}

FIGURE_INPUTS = {
    "throughput": ("end_to_end.csv",),
    "sweeps": ("budget.csv", "blocklist.csv"),
    "quantization-speedup": ("cap_quantization/*/*.csv", "end_to_end.csv"),
    "quantization-throughput": ("cap_quantization/*/*.csv",),
    "quantization": ("cap_quantization/*/*.csv",),
    "equal-memory-nonzero-fn": ("equal_memory/*/equal_memory.csv",),
    "equal-memory-zero-fn": ("equal_memory/*/equal_memory.csv",),
    "construction": ("construct_methods/*.csv",),
    "streaming": ("streaming/*/streaming.csv",),
}


def _fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plot_figures(requested=None, results=None, out=None):
    """Draws the named figures (default: all) from `results`/ into `out`/."""
    global RESULTS, OUT_FIG
    if results is not None:
        RESULTS = Path(results)
    if out is not None:
        OUT_FIG = Path(out)
    requested = list(requested or [])
    unknown = [r for r in requested if r not in FIGURES]
    if unknown:
        raise SystemExit(f"unknown figure(s) {unknown}; choose from {list(FIGURES)}")
    print("reading", RESULTS, "-> writing", OUT_FIG)
    selected = requested or list(FIGURES)
    inputs = {name: sorted({p for pattern in FIGURE_INPUTS[name] for p in RESULTS.glob(pattern)})
              for name in selected}
    hashes = {p: _fingerprint(p) for paths in inputs.values() for p in paths}
    failed = []
    for name in selected:
        try:
            FIGURES[name]()
        except Exception as e:      # one figure's missing data must not stop the others
            failed.append(name)
            print(f"  {name}: not drawn ({type(e).__name__}: {e})")
    if failed:
        raise SystemExit(f"figures not drawn: {', '.join(failed)}")
    if any(_fingerprint(p) != digest for p, digest in hashes.items()):
        raise RuntimeError("Source CSVs changed while plotting; rerun to use one consistent data snapshot")
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(RESULTS.resolve()),
        "plot_script_sha256": _fingerprint(Path(__file__)),
        "threshold_units": "stored L2 thresholds are displayed as cosine distance = L2 squared / 2",
        "paper_caption_checks": [
            "Figure 4 reports modeled speedup, not a direct end-to-end timing measurement.",
            "Figure 6 uses only FP32 rows; its threshold is taken from the source CSV, not the PDF caption.",
            "Fixed TQ2 and TQ4 baselines retain their encoding labels and are not named automatic selection.",
            "Both fixed-threshold throughput panels use the configured cosine distance; a single available budget is plotted as a point.",
        ],
        "figures": {},
    }
    for name in selected:
        stem = FIGURE_FILES[name]
        if name == "equal-memory-nonzero-fn":
            corpus = NONZERO_FN_CORPUS if NONZERO_FN_CORPUS in CORPORA else CORPORA[0]
            stem = f"equal-memory-nonzero-fn-{corpus.lower()}"
        manifest["figures"][name] = {
            "paper_figure": PAPER_FIGURES.index(name) + 2 if name in PAPER_FIGURES else None,
            "inputs": [{"path": str(p.relative_to(RESULTS)), "sha256": hashes[p],
                        "rows": len(pd.read_csv(p))} for p in inputs[name]],
            "outputs": {f"{stem}.{ext}": _fingerprint(OUT_FIG / f"{stem}.{ext}") for ext in ("pdf", "png")},
        }
    (OUT_FIG / "plot-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="*", help="optional results directory, then figure names")
    parser.add_argument("--out", type=Path, default=OUT_FIG)
    parser.add_argument("--paper-only", action="store_true", help="draw data Figures 2–7 in paper.pdf")
    args = parser.parse_args()
    inputs = list(args.inputs)
    results = Path(inputs.pop(0)) if inputs and inputs[0] not in FIGURES else RESULTS
    if args.paper_only and inputs:
        parser.error("use --paper-only or individual figure names")
    plot_figures(PAPER_FIGURES if args.paper_only else inputs, results, args.out)


if __name__ == "__main__":
    main()
