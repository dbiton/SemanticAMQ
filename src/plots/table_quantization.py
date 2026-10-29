"""tab:quantization: the best direction encoding at each rate (bits per dimension), as
LaTeX, from results/cap_quantization/<corpus>/*.csv (src/paper.py's generate_quantization).

Per budget (B/M, % of the raw index) a cell is the index accesses saved,
averaged over corpora and folds. Per rate, the encoding with the best mean over the cells
is listed, then every encoding auto() chooses among, then auto() itself. Bold: best in column. Dagger: the budget fits one cap per indexed vector.
Last column (when `qps` is given, name -> queries/s): the query throughput of one batch call, measured by
runner.pinned_timing_worker (one pinned core, results/timing.csv "scan" rows) rather than by the parallel
quantization run, whose timings are not pinned (CLAUDE.md).

    PYTHONPATH=src python src/plots/table_quantization.py [results-folder] [out.tex]
"""
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from filters.cap_cover.cap_cover_filter import AUTO_CANDIDATES
from filters.cap_cover.cap_quantizers import Quantize, _ENCODINGS

AUTO = "auto"       # runner.CAP_QUANTIZE_AUTO: CapCoverFilter.auto() choosing among AUTO_CANDIDATES
ROOT = Path(__file__).resolve().parents[2]
BS = chr(92)      # LaTeX backslash


def _label(name: str) -> str:
    return {"SPQ12": "SPQ 48x12", "SPQ16": "SPQ 24x8", "SPQ24": "SPQ 24x12", "SPQ36": "SPQ 16x12",
            "SPQ48": "SPQ 12x12", "SPQ64": "SPQ 6x8"}.get(name, name)


def _tex(label: str) -> str:
    return label.replace("SPQ ", "SPQ $").replace("x", f"{{{BS}times}}", 1) + "$" if label.startswith("SPQ") else label


def _rate(r: float) -> str:
    return f"{r:.3g}"


def build(q: pd.DataFrame) -> pd.DataFrame:
    """One row per encoding: rate, saved/singleton per budget, qps; then the best per rate."""
    known = {m.value for m in Quantize} | {AUTO}
    q = q[q.quantize.isin(known)].copy()      # a CSV from an older encoding ladder may name rungs that no longer exist
    q["budget_pct"] = 100 * q.memory_fraction
    q["singleton"] = q.n_caps >= q.n_construct
    q["rate"] = q.quantize.map(lambda n: np.nan if n == AUTO else _ENCODINGS[Quantize(n)].rate)
    return q


# The budgets (% of raw) the paper's table shows; the run covers more (main.BUDGET_MEMORY_FRACTIONS).
TABLE_BUDGETS = (0.5, 2, 5)


def table(q: pd.DataFrame, qps: Optional[Dict[str, float]] = None, qps_label: str = "q/s",
          budgets: Optional[tuple] = TABLE_BUDGETS) -> str:
    """`budgets` (% of raw; None: every one in `q`) selects the columns, and the best encoding per rate
    is chosen over those columns only."""
    q = build(q)
    if budgets:
        q = q[np.isclose(q.budget_pct.to_numpy()[:, None], np.asarray(budgets, dtype=float)[None, :], rtol=0.02).any(axis=1)]
    budgets = sorted(q.budget_pct.unique())
    saved = q.pivot_table(index="quantize", columns="budget_pct", values="index_accesses_saved_pct", aggfunc="mean")
    single = q.pivot_table(index="quantize", columns="budget_pct", values="singleton", aggfunc="mean")
    rate = q.groupby("quantize").rate.first()

    fixed = saved.drop(index=AUTO, errors="ignore")
    best = fixed.mean(axis=1).groupby(rate.drop(index=AUTO, errors="ignore")).idxmax()     # per rate: the encoding with the best mean
    # The best at each rate, plus every encoding auto() chooses among, by rate; then auto() itself.
    shown = set(best.values) | {c.value for c in AUTO_CANDIDATES if c.value in fixed.index}
    rows = sorted(((rate[n], n) for n in shown), key=lambda kv: (-kv[0], kv[1]))
    if AUTO in saved.index:
        rows.append((np.nan, AUTO))
    names = [n for _, n in rows]
    colmax = saved.loc[names].max()

    extra = "r" if qps else ""
    qps = qps or {}
    out = [f"{BS}begin{{tabular}}{{@{{}}lr" + "r" * len(budgets) + extra + "@{}}", f"{BS}toprule",
           f"& & {BS}multicolumn{{{len(budgets)}}}{{c}}{{Memory budget $B/M$}}" + (" &" if extra else "") + f" {BS*2}",
           f"{BS}cmidrule(lr){{3-{2 + len(budgets)}}}",
           "Encoding & b/dim & " + " & ".join(f"{b / 100:.3g}" for b in budgets) + (f" & {qps_label}" if extra else "") + f" {BS*2}", f"{BS}midrule"]
    for r, name in rows:
        if name == AUTO:
            out.append(f"{BS}midrule")
        cells = []
        for b in budgets:
            v = saved.loc[name, b]
            if pd.isna(v):
                cells.append("--")
                continue
            c = f"{v:.1f}"
            if v >= colmax[b] - 1e-9:
                c = f"{BS}textbf{{{c}}}"
            cells.append(c + (f"$^{{{BS}dagger}}$" if single.loc[name, b] >= 0.5 else ""))
        q_s = f"{qps[name]:,.0f}".replace(",", "{,}") if name in qps and pd.notna(qps[name]) else "--"
        out.append(f"{_tex(_label(name))} & {'--' if name == AUTO else _rate(r)} & " + " & ".join(cells) + (f" & {q_s}" if extra else "") + f" {BS*2}")
    out += [f"{BS}bottomrule", f"{BS}end{{tabular}}"]
    return "\n".join(out) + "\n"


def write(results: Path, out: Path) -> None:
    paths = sorted(Path(results).glob("cap_quantization/*/*.csv"))
    if not paths:
        raise FileNotFoundError(f"no files match {results / 'cap_quantization/*/*.csv'}")
    qps, label = None, "q/s"
    timing = Path(results) / "timing.csv"
    if timing.exists():
        t = pd.read_csv(timing)
        t = t[t.kind == "scan"] if "kind" in t else t.iloc[0:0]
        if len(t):
            qps = t.groupby("method").qps.median().to_dict()      # measured at pct % of raw: the caption says which
    tex = table(pd.concat([pd.read_csv(p) for p in paths], ignore_index=True), qps, label)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex, encoding="utf-8")
    print(f"  {out}")


if __name__ == "__main__":
    args = sys.argv[1:]
    write(ROOT / (args[0] if args else "results"), ROOT / (args[1] if len(args) > 1 else "suggestions/figures/quantization-table.tex"))
