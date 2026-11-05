"""The tables paper.tex adds beyond table_quantization.py, as LaTeX (written to
figures/), plus the numbers its text quotes from them. Built only from existing
results/ CSVs (src/paper.py generates them):

    ablation-table.tex       tab:ablation       ablation.csv   FPR of each variant at delta = 0.02
    sensitivity-table.tex    tab:sensitivity    ablation.csv   change in FPR when the cover is built for another
                                                               threshold / sphere-merge parameter
    latency-table.tex        tab:latency        latency.csv    the filter in front of the flat index
    timing-table.tex         tab:timing         timing.csv     construction and per-arrival insertion time
    jtau-correlation.csv     (text)             ablation.csv   rank correlation of J_tau with FPR across variants
    space_bound.csv          (remark)           budget.csv     the space lower bound at the measured accept-region measures

    PYTHONPATH=src python src/plots/tables_paper.py [results-folder] [out-folder]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BS = chr(92)     # LaTeX backslash


def _memory_label(memory_fraction: float) -> str:
    """A budget B/M (filter bytes over the raw FP32 embedding bytes) as a column header."""
    return f"$B={memory_fraction:g}M$"

DELTA = 0.02
# variant -> (component, label); first of each component is the paper's configuration
ABLATION_ROWS = [
    ("full", "Rule", "cap (paper)"),
    ("ball_sphere", "", "ball at the cap direction"),
    ("ball_meb", "", "ball, minimum enclosing"),
    ("full", "Build", "sphere merge (paper)"),
    ("minimax", "", "minimax"),
    ("gonzalez", "", "Gonzalez"),
    ("soft_anneal", "", "sphere merge + soft anneal"),
    ("full", "Enc.", "auto (paper)"),
    ("tq_only", "", "TurboQuant rungs only"),
    ("spq_only", "", "seeded-PQ rungs only"),
    ("tq4_fixed", "", "TQ4 only"),
    ("fp32_fixed", "", "FP32 only"),
    ("no_mixing", "", "one encoding for all caps"),
    ("full", "Codes", "fitted (paper)"),
    ("nearest_codes", "", "nearest to the direction"),
]
SENSITIVITY_ROWS = [
    ("t0.05", "offset $t=0.05$"), ("t0.2", "offset $t=0.2$"), ("t0.4", "offset $t=0.4$"),
    ("p4", "exponent $p=4$"), ("p16", "exponent $p=16$"),
    ("select_delta0.02", "encoding chosen at $\\delta=0.02$"), ("select_delta0.1", "encoding chosen at $\\delta=0.1$"),
]


def _paired(d: pd.DataFrame, arm: str, kf: float, delta: float):
    """(mean FPR over cells, mean paired difference to "full", number of cells worse, cells)."""
    g = d[np.isclose(d.memory_fraction, kf) & np.isclose(d.delta, delta)]
    piv = g.pivot_table(index="arm", columns=["dataset", "fold"], values="fpr")
    if arm not in piv.index or "full" not in piv.index:
        return None
    diff = (piv.loc[arm] - piv.loc["full"]).dropna()
    return float(piv.loc[arm].mean()), float(diff.mean()), int((diff > 1e-9).sum()), len(diff)


def _num(x: float, signed: bool = False) -> str:
    if signed and abs(x) < 0.0005:
        return ".000"
    s = f"{x:+.3f}" if signed else f"{x:.3f}"
    return s.replace("0.", ".", 1) if abs(x) < 1 else s


def ablation_table(d: pd.DataFrame) -> str:
    kfs = sorted(d.memory_fraction.unique())[:2]
    out = [f"{BS}begin{{tabular}}{{@{{}}llrrrr@{{}}}}", f"{BS}toprule",
           f"& & {BS}multicolumn{{2}}{{c}}{{{_memory_label(kfs[0])}}} & {BS}multicolumn{{2}}{{c}}{{{_memory_label(kfs[1])}}} {BS*2}",
           f"{BS}cmidrule(lr){{3-4}}{BS}cmidrule(lr){{5-6}}",
           f"Component & Variant & FPR & {BS}multicolumn{{1}}{{c}}{{$\\Delta$}} & FPR & {BS}multicolumn{{1}}{{c}}{{$\\Delta$}} {BS*2}",
           f"{BS}midrule"]
    for i, (arm, comp, label) in enumerate(ABLATION_ROWS):
        if arm not in set(d.arm):          # a variant the CSV predates
            continue
        if comp and i:
            out.append(f"{BS}addlinespace[2pt]")
        cells = []
        for kf in kfs:
            r = _paired(d, arm, kf, DELTA)
            if r is None:
                cells += ["--", "--"]
                continue
            mean, diff, worse, n = r
            cells.append(_num(mean))
            cells.append("" if arm == "full" else f"{_num(diff, True)}$^{{{worse}}}$")
        out.append(f"{comp} & {label} & " + " & ".join(cells) + f" {BS*2}")
    out += [f"{BS}bottomrule", f"{BS}end{{tabular}}"]
    return "\n".join(out) + "\n"


def sensitivity_table(d: pd.DataFrame) -> str:
    """Change in FPR (variant minus the paper's configuration, mean over cells) of a cover built
    with another sphere-merge offset t / exponent p or another selection threshold, scored at
    several query thresholds delta."""
    deltas = [0.005, 0.02, 0.1]
    kfs = sorted(d.memory_fraction.unique())[:2]
    n = len(deltas)
    out = [f"{BS}begin{{tabular}}{{@{{}}l" + "r" * (n * len(kfs)) + "@{}}", f"{BS}toprule",
           "& " + " & ".join(f"{BS}multicolumn{{{n}}}{{c}}{{{_memory_label(kf)}}}" for kf in kfs) + f" {BS*2}",
           "".join(f"{BS}cmidrule(lr){{{2 + i * n}-{1 + (i + 1) * n}}}" for i in range(len(kfs))),
           "Variant & " + " & ".join(f"{dl:g}" for _ in kfs for dl in deltas) + f" {BS*2}",
           f"{BS}midrule"]
    for arm, label in SENSITIVITY_ROWS:
        cells = []
        for kf in kfs:
            for dl in deltas:
                r = _paired(d, arm, kf, dl)
                cells.append("--" if r is None else _num(r[1], True))
        out.append(f"{label} & " + " & ".join(cells) + f" {BS*2}")
    out += [f"{BS}bottomrule", f"{BS}end{{tabular}}"]
    return "\n".join(out) + "\n"


def jtau_correlation(d: pd.DataFrame) -> pd.DataFrame:
    """Spearman rank correlation between log10 J_tau and FPR across the variants of one cell
    (dataset, fold, budget) at DELTA, for all cap variants and for the competitive ones (those
    within 0.1 FPR of the paper's configuration, where the volume has to rank near-equal covers)."""
    from scipy.stats import spearmanr
    g = d[np.isclose(d.delta, DELTA) & ~d.arm.isin(["ball_sphere", "ball_meb"])]
    rows = []
    for (ds, fold, kf), c in g.groupby(["dataset", "fold", "memory_fraction"]):
        full = float(c.loc[c.arm == "full", "fpr"].iloc[0])
        for scope, cc in (("all", c), ("competitive", c[c.fpr <= full + 0.1])):
            if cc.fpr.nunique() > 2 and cc.log10_j_tau.nunique() > 2:
                rows.append({"dataset": ds, "fold": fold, "memory_fraction": kf, "scope": scope, "n_variants": len(cc),
                             "spearman": float(spearmanr(cc.log10_j_tau, cc.fpr)[0])})
    return pd.DataFrame(rows)


DATASET_ABBR = {"wildchat": "WildChat", "quora": "Quora", "wikianswers": "WikiAnswers",
                "chatgpt_paraphrases": "ChatGPT-Para.", "ccnews": "CC-News"}


def latency_table(b: pd.DataFrame, delta: float = DELTA) -> str:
    """One row per corpus: the filter's cost per query (mean, p99) and acceptance, the flat
    index's mean lookup cost, the break-even lookup cost, and the throughput gain of the
    filtered system. `gain` is negative where the filter costs more than it saves."""
    g = b[np.isclose(b.delta, delta)]
    out = [f"{BS}begin{{tabular}}{{@{{}}lrrrrrr@{{}}}}", f"{BS}toprule",
           f"& {BS}multicolumn{{3}}{{c}}{{Filter}} & {BS}multicolumn{{3}}{{c}}{{Flat index}} {BS*2}",
           f"{BS}cmidrule(lr){{2-4}}{BS}cmidrule(lr){{5-7}}",
           f"Corpus & ms & p99 & accepts {BS}% & lookup ms & break-even ms & gain {BS}% {BS*2}", f"{BS}midrule"]
    for ds in [x for x in DATASET_ABBR if x in set(g.dataset)]:
        r = g[g.dataset == ds].mean(numeric_only=True)      # over the folds
        fmt = ",.1f" if abs(r.throughput_improvement_pct) < 10 else ",.0f"
        gain = f"{r.throughput_improvement_pct:{fmt}}".replace(",", "{,}").replace("-", BS + "textminus ")
        be = "--" if not np.isfinite(r.breakeven_backend_ms) else f"{r.breakeven_backend_ms:.2f}"
        out.append(f"{DATASET_ABBR[ds]} ({r.n_reference / 1000:g}K) & {r.filter_mean_ms:.2f} & {r.filter_p99_ms:.2f} & "
                   f"{r.accept_rate_all_pct:.0f} & {r.backend_mean_ms:.2f} & {be} & {gain} {BS*2}")
    out += [f"{BS}bottomrule", f"{BS}end{{tabular}}"]
    return "\n".join(out) + "\n"


def timing_table(t: pd.DataFrame) -> str:
    c = t[t.kind == "construct"]
    ks = sorted(c.memory_fraction.unique())
    out = [f"{BS}begin{{tabular}}{{@{{}}l" + "r" * len(ks) + "@{}}", f"{BS}toprule",
           f"Construction (s) & " + " & ".join(_memory_label(k) for k in ks) + f" {BS*2}", f"{BS}midrule"]
    for m in ("gonzalez", "minimax", "sphere_merge", "soft_anneal"):
        r = c[c.method == m].groupby("memory_fraction").seconds.mean()
        out.append(f"{m.replace('_', ' ')} & " + " & ".join("--" if k not in r.index else f"{r[k]:.1f}" for k in ks) + f" {BS*2}")
    ins = t[t.kind == "insert"]
    out += [f"{BS}midrule", f"Insertion (ms) & mean & p50 & p99 {BS*2}".replace("& mean & p50 & p99", "& mean & p99 &"),
            f"{BS}midrule"]
    for m in ("nearest", "cap_merge"):
        r = ins[ins.method == m]
        if len(r):
            out.append(f"{m.replace('_', ' ')} & {r.mean_ms.iloc[0]:.2f} & {r.p99_ms.iloc[0]:.2f} & {BS*2}".replace(" & &", " &"))
    out += [f"{BS}bottomrule", f"{BS}end{{tabular}}"]
    return "\n".join(out) + "\n"


def write_all(results: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if (results / "ablation.csv").exists():
        d = pd.read_csv(results / "ablation.csv")
        (out / "ablation-table.tex").write_text(ablation_table(d), encoding="utf-8")
        (out / "sensitivity-table.tex").write_text(sensitivity_table(d), encoding="utf-8")
        jtau_correlation(d).to_csv(out / "jtau-correlation.csv", index=False)
    if (results / "budget.csv").exists():
        from util.space_bound import measured_points
        measured_points(str(results / "budget.csv")).to_csv(out / "space_bound.csv", index=False)
    if (results / "latency.csv").exists():
        (out / "latency-table.tex").write_text(latency_table(pd.read_csv(results / "latency.csv")), encoding="utf-8")
    if (results / "timing.csv").exists():
        (out / "timing-table.tex").write_text(timing_table(pd.read_csv(results / "timing.csv")), encoding="utf-8")


if __name__ == "__main__":
    res = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "results")
    write_all(res, ROOT / (sys.argv[2] if len(sys.argv) > 2 else "figures"))
