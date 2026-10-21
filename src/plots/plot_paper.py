"""The figures paper.tex uses, built only from existing results/ CSVs
(src/paper.py generates exactly the data they need).

Nothing here runs a filter or a sweep. Outputs (PDF for the paper, PNG preview)
go to suggestions/figures/ under the names paper.tex includes:

    sweeps-grid.*                      fig:end-to-end   budget.csv + blocklist.csv
    quantization-saved-vs-memory.*     fig:quantization cap_quantization/<corpus>/*.csv
    construction-fpr-vs-k.*            fig:construction construct_methods/<corpus>.csv
    streaming-fpr-vs-position.*        fig:streaming    streaming/<corpus>/streaming.csv
    equal-memory-fpr-fnr-vs-budget.*   fig:equal-memory equal_memory/<corpus>/equal_memory.csv
    throughput-improvement.*           fig:throughput   end_to_end.csv

(figures/scheme.pdf is a hand-drawn diagram and is not produced here.)

Run from the repo root with the global Python 3.12:
    PYTHONPATH=src python src/plots/plot_paper.py [results-folder] [figure ...]
results-folder defaults to results/. Figure names are the keys of FIGURES
(sweeps, quantization, construction, streaming, equal-memory, throughput); default is all.

The sweeps figure's FPR panels group the negatives by how related a query is
to the indexed set -- the cosine similarity to its nearest construct vector:
closely related (>= 0.8), related (0.6-0.8), unrelated (< 0.6)
(runner.SIMILARITY_BANDS).

Runs may be in progress, so every figure tolerates missing corpora and cells:
a gap is drawn as a gap, never filled from a different subset.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "results"
OUT_FIG = ROOT / "suggestions" / "figures"

DIM = 384
FP32_BYTES = 4

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

# Equal-memory methods: Cap-Cover is always blue.
CAP_COVER = "Cap-Cover"
METHOD_STYLE = {
    CAP_COVER: (BLUE, "o", "-"),
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
    "CapCoverFilter": CAP_COVER,
    "QuantizedVectorFilter": "QuantizedVector",
    "IVFFilter": "IVF-PQ", "IVFRadiiFilter": "IVF-Radii", "HNSWFilter": "HNSW",
    "DSHFilter": "DSH", "MLPFilter": "MLP",
    "PacukLSHFilter": "Zero-FN LSH", "SankowskiWygockiFilter": "Zero-FN LSH",
    "VAPlusFilter": "VA+-file",
}
# Left out of the equal-memory figure.
EQUAL_MEMORY_HIDDEN = {"QuantizedVector", "MLP", "HNSW", "Zero-FN LSH"}

def use_corpora(names):
    """Draw these corpora (display names) instead of the paper's five: for a
    run on whatever corpora a machine has (src/paper.py --datasets)."""
    CORPORA[:] = names
    for i, name in enumerate(names):
        CORPUS_NAME[name.lower()] = name
        CORPUS_LABEL.setdefault(name, name)
        CORPUS_STYLE.setdefault(name, (PALETTE[i % len(PALETTE)], MARKERS[i % len(MARKERS)]))


QUANT_TOP_N = 5

# End-to-end throughput (src/paper.py generates exactly these cells): the filter
# in front of a flat index vs the index alone. The tau panel fixes the budget,
# the memory and K panels fix tau; the budget sweep is shared by those two.
E2E_TAU = 0.1
E2E_K_FRACTION = 0.1
E2E_TAUS = (0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4)
E2E_K_FRACTIONS = (0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2)

# Construction methods shown (CSV key -> label, color, marker, linestyle).
CONSTRUCT_SHOWN = {
    "gonzalez": ("Gonzalez", INK_MUTED, "o", ":"),
    "minimax": ("minimax", ORANGE, "s", "-"),
    "sphere_merge": ("sphere merge", BLUE, "^", "-"),
}
# Insertion strategies shown. The initial cover is built with STREAM_BOOTSTRAP.
STREAM_BOOTSTRAP = "sphere_merge"
INSERT_SHOWN = {
    "nearest": ("nearest", ORANGE, "s", "-"),
    "cap_merge": ("cap merge", BLUE, "D", "-"),
    "static_rebuild": ("static rebuild (oracle)", INK, None, "--"),
}
# FPR panel per similarity band on the sweeps (column suffix -> title).
BAND_TITLE = {"close": "closely related\n($\\cos\\geq0.8$)", "related": "related\n($0.6\\leq\\cos<0.8$)",
                "unrelated": "unrelated\n($\\cos<0.6$)"}

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
GAP_SHARED = 0.12
GAP_LABELLED = 0.62
MARGIN_TOP = 0.32

PCT_FMT = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")


def save(fig, name):
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG / f"{name}.pdf")
    fig.savefig(OUT_FIG / f"{name}.png")
    plt.close(fig)
    print(f"  figures/{name}.pdf")


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
    line. Error bars: +-1 std over folds."""
    frames = []
    for name in ("budget", "blocklist"):
        d = pd.read_csv(RESULTS / f"{name}.csv")
        d["corpus"] = to_corpus(d.dataset)
        d["pct_raw"] = 100 * d.memory_bytes / (d.n_reference * DIM * FP32_BYTES)
        d["k_pct"] = 100 * d.k_fraction  # memory varies by fold; k does not
        frames.append(d)
    budget, blocklist = frames
    tau_mem = blocklist.pct_raw.mean()
    sweeps = [(budget, "k_pct", "$k$ (% of $|X|$)","FPR\n($\\tau=0.1$)", PCT_FMT),
              (blocklist, "tau", "$\\tau$", f"FPR\n(memory {tau_mem:.0f}% of raw)", None)]

    n_fpr, nrows = len(BAND_TITLE), len(sweeps)
    spacer = GAP_LABELLED - 2 * GAP_SHARED  # blank column before the accesses-saved panel
    width = (n_fpr + 1) * PANEL + spacer + (n_fpr + 1) * GAP_SHARED
    height = nrows * PANEL + (nrows - 1) * GAP_LABELLED + MARGIN_TOP
    fig, grid = plt.subplots(
        nrows, n_fpr + 2, figsize=(width, height), squeeze=False,
        gridspec_kw=dict(left=0, right=1, bottom=0, top=1 - MARGIN_TOP / height,
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

def quantization():
    q = read_all("cap_quantization/*/*.csv")
    q["corpus"] = to_corpus(q.dataset)
    # The budget is k FP32 caps; compact encodings fit more caps in it, up to one per vector.
    q["budget_pct"] = 100 * q.k / q.n_construct
    q["singleton"] = q.n_caps >= q.n_construct
    agg = (q.groupby(["corpus", "quantize", "budget_pct"])
           .agg(saved=("index_accesses_saved_pct", "mean"), singleton=("singleton", "mean"))
           .reset_index())
    ceilings = q.groupby("corpus").neg_share_pct.median()

    # Only the QUANT_TOP_N best encodings: lowest mean rank by accesses saved
    # within each corpus x budget cell (a missing cell does not count against
    # an encoding), ties broken by mean accesses saved.
    cells = agg.pivot_table(index=["corpus", "budget_pct"], columns="quantize", values="saved")
    score = pd.DataFrame({"rank": cells.rank(axis=1, ascending=False).mean(), "saved": cells.mean()})
    top = list(score.sort_values(["rank", "saved"], ascending=[True, False]).index[:QUANT_TOP_N])
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
            ax.plot(g.budget_pct, g.saved, color=color, ls=ls, lw=1.2, zorder=2)
            full = g.singleton >= 0.5
            ax.scatter(g.budget_pct[~full], g.saved[~full], color=color, marker=marker,
                       s=12, edgecolor="white", lw=0.5, zorder=3)
            ax.scatter(g.budget_pct[full], g.saved[full], facecolor="white",
                       edgecolor=color, marker=marker, s=12, lw=0.9, zorder=3)
        if corpus in ceilings:
            ax.axhline(ceilings[corpus], color=INK_MUTED, ls="--", lw=0.8)
            ax.text(0.97, 0.04, f"ceiling {ceilings[corpus]:.0f}%", color=INK_2, fontsize=6.5,
                    ha="right", transform=ax.transAxes)
        ax.set_xscale("log", base=2)
        ticks = sorted(q.budget_pct.unique())
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:.2g}" for t in ticks], fontsize=6)
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axes[0, 0].set_ylabel("index accesses saved (%)")
    axes[0, 0].set_ylim(-2, 100)
    handles = [plt.Line2D([], [], color=c, marker=mk, ls=ls) for c, mk, ls in top_style.values()]
    handles.append(plt.Line2D([], [], color=INK_2, marker="o", mfc="white", ls=""))
    labels = top + ["one cap per vector"]
    fig_legend(fig, handles, labels, ncol=len(labels))
    shared_xlabel(axes, "memory budget (% of raw fp32 index = $k$ FP32 caps, log)")
    save(fig, "quantization-saved-vs-memory")


# ---------------------------------------------------------------------------
# fig:construction  construction-fpr-vs-k
# ---------------------------------------------------------------------------

def construction():
    c = read_all("construct_methods/*.csv")
    c = c[~c.dataset.astype(str).str.startswith("scale_")].copy()
    c["corpus"] = to_corpus(c.dataset)
    n_ref = int(c.n_reference.iloc[0])
    agg = c.groupby(["corpus", "method", "k_fraction"]).fpr.agg(["mean", "std"]).reset_index()
    fig, axes = corpus_panels(sharey=True)
    for ax, corpus in zip(axes[0], CORPORA):
        d = agg[agg.corpus == corpus]
        if d.empty:
            mark_empty(ax)
        for method, (_, color, marker, ls) in CONSTRUCT_SHOWN.items():
            g = d[d.method == method].sort_values("k_fraction")
            ax.errorbar(100 * g.k_fraction, g["mean"], yerr=g["std"].fillna(0), color=color,
                        marker=marker, ls=ls, ms=3, capsize=1.2, elinewidth=0.7)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(PCT_FMT)
    axes[0, 0].set_ylabel("FPR")
    axes[0, 0].set_ylim(-0.03, 1.05)
    handles = [plt.Line2D([], [], color=c_, marker=mk, ls=ls)
               for _, c_, mk, ls in CONSTRUCT_SHOWN.values()]
    fig_legend(fig, handles, [v[0] for v in CONSTRUCT_SHOWN.values()], ncol=len(CONSTRUCT_SHOWN))
    shared_xlabel(axes, f"cap budget $k$ (% of $|B|={n_ref:,}$, log)")
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
    k_pos = int(pos.k.iloc[0])
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
    shared_xlabel(axes, f"inserts after bootstrap (thousands), $k={k_pos:,}$")
    save(fig, "streaming-fpr-vs-position")


# ---------------------------------------------------------------------------
# fig:equal-memory  equal-memory-fpr-fnr-vs-budget
# ---------------------------------------------------------------------------

def equal_memory():
    e = read_all("equal_memory/*/equal_memory.csv")
    e = e[~e.Infeasible.astype(bool)].copy()
    e["Dataset"] = to_corpus(e.Dataset)
    e["pct_raw"] = (100 * e["Memory Budget"] / (e["Construct Size"] * DIM * FP32_BYTES)).round(3)
    e["method"] = e.Config.map(CONFIG_TO_METHOD)
    # Two configs feed "Zero-FN LSH": keep the lower-FPR one at each point.
    e = e.sort_values("FPR").drop_duplicates(["method", "Dataset", "Memory Budget"], keep="first")
    styles = {m: st for m, st in METHOD_STYLE.items()
              if m in set(e.method) and m not in EQUAL_MEMORY_HIDDEN}

    fig, axes = corpus_panels(2, sharex=True, sharey="row")
    for j, corpus in enumerate(CORPORA):
        d = e[e.Dataset == corpus]
        if d.empty:
            mark_empty(axes[0, j])
            mark_empty(axes[1, j])
        for method, (color, marker, ls) in styles.items():
            g = d[d.method == method].sort_values("pct_raw")
            if g.empty:
                continue
            axes[0, j].plot(g.pct_raw, g.FPR, color=color, marker=marker, ls=ls, ms=3)
            axes[1, j].plot(g.pct_raw, g.FNR, color=color, marker=marker, ls=ls, ms=3)
        axes[1, j].set_xscale("log")
        axes[1, j].xaxis.set_major_formatter(PCT_FMT)
    axes[0, 0].set_ylabel("FPR")
    axes[1, 0].set_ylabel("FNR")
    axes[0, 0].set_ylim(-0.03, 1.05)
    axes[1, 0].set_ylim(-0.03, 1.0)
    handles = [plt.Line2D([], [], color=c, marker=mk, ls=ls) for c, mk, ls in styles.values()]
    fig_legend(fig, handles, list(styles), ncol=min(len(styles), 8))
    shared_xlabel(axes, "memory (% of raw fp32 index, log)")
    save(fig, "equal-memory-fpr-fnr-vs-budget")


# ---------------------------------------------------------------------------
# fig:throughput  throughput-improvement
# ---------------------------------------------------------------------------

def throughput():
    """End-to-end throughput improvement (%) of the filter in front of a flat
    index over the index alone, one line per corpus, against tau (budget
    fixed), against the memory the filter actually uses, and against its
    number of caps K (tau fixed). Memory and K both come from the budget
    sweep, but differ: a compact encoding packs more caps into the same
    bytes. Error bars: +-1 std over folds."""
    d = pd.read_csv(RESULTS / "end_to_end.csv")
    d["corpus"] = to_corpus(d.dataset)
    at_k = d[np.isclose(d.k_fraction, E2E_K_FRACTION)]
    at_tau = d[np.isclose(d.tau, E2E_TAU)]
    panels = [
        (at_k, "tau", "$\\tau$", f"fixed $k={100 * E2E_K_FRACTION:g}\\%$ of $|X|$", False),
        (at_tau, "memory_pct_raw", "memory (% of raw fp32 index, log)", f"fixed $\\tau={E2E_TAU:g}$", True),
        (at_tau, "n_caps", "caps $K$ (log)", f"fixed $\\tau={E2E_TAU:g}$", True),
    ]
    width = len(panels) * PANEL + (len(panels) - 1) * GAP_LABELLED
    height = PANEL + MARGIN_TOP
    fig, axes = plt.subplots(1, len(panels), figsize=(width, height), squeeze=False,
                             gridspec_kw=dict(left=0, right=1, bottom=0, top=1 - MARGIN_TOP / height,
                                              wspace=GAP_LABELLED / PANEL))
    for ax, (df, x, xlabel, title, logx) in zip(axes[0], panels):
        ax.set_box_aspect(1)
        agg = (df.groupby(["corpus", "k_fraction", "tau"])
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
        if logx:
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(PCT_FMT)
        if x == "tau":
            ax.set_xticks([0.05, 0.1, 0.2, 0.3, 0.4])
        ax.set_xlabel(xlabel)
        ax.set_title(title, color=INK)
    axes[0, 0].set_ylabel("throughput improvement (%)")
    present = [c for c in CORPORA if c in set(to_corpus(d.dataset))]
    handles = [plt.Line2D([], [], color=CORPUS_STYLE[c][0], marker=CORPUS_STYLE[c][1]) for c in present]
    fig_legend(fig, handles, [CORPUS_LABEL[c] for c in present], ncol=len(present))
    save(fig, "throughput-improvement")


FIGURES = {
    "sweeps": sweeps_grid,
    "quantization": quantization,
    "construction": construction,
    "streaming": streaming,
    "equal-memory": equal_memory,
    "throughput": throughput,
}


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
    failed = []
    for name in requested or FIGURES:
        try:
            FIGURES[name]()
        except Exception as e:      # one figure's missing data must not stop the others
            failed.append(name)
            print(f"  {name}: not drawn ({type(e).__name__}: {e})")
    if failed:
        raise SystemExit(f"figures not drawn: {', '.join(failed)}")


def main():
    args = sys.argv[1:]
    results = ROOT / args.pop(0) if args and (ROOT / args[0]).is_dir() else None
    plot_figures(args, results)


if __name__ == "__main__":
    main()
