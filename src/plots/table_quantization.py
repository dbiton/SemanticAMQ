"""tab:quantization: the best direction encoding at each rate (bits per dimension), as
LaTeX, from results/cap_quantization/<corpus>/*.csv (src/paper.py's generate_quantization).

Per budget (k FP32 caps as a % of the construct set) a cell is the index accesses saved,
averaged over corpora and folds. Per rate, the encoding with the best mean over the cells
is listed. Bold: best in column. Dagger: the budget fits one cap per indexed vector.
Last column: median queries/s at the smallest budget.

    PYTHONPATH=src python src/plots/table_quantization.py [results-folder] [out.tex]
"""
import sys
from pathlib import Path

import pandas as pd

from filters.cap_cover.cap_quantizers import Quantize, _ENCODINGS

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
    known = {m.value for m in Quantize}
    q = q[q.quantize.isin(known)].copy()      # a CSV from an older encoding ladder may name rungs that no longer exist
    q["budget_pct"] = 100 * q.k / q.n_construct
    q["singleton"] = q.n_caps >= q.n_construct
    q["rate"] = q.quantize.map(lambda n: _ENCODINGS[Quantize(n)].rate)
    return q


def table(q: pd.DataFrame) -> str:
    q = build(q)
    budgets = sorted(q.budget_pct.unique())
    saved = q.pivot_table(index="quantize", columns="budget_pct", values="index_accesses_saved_pct", aggfunc="mean")
    single = q.pivot_table(index="quantize", columns="budget_pct", values="singleton", aggfunc="mean")
    qps = q[q.budget_pct == budgets[0]].groupby("quantize").query_throughput_qps.median()
    rate = q.groupby("quantize").rate.first()

    best = saved.mean(axis=1).groupby(rate).idxmax()     # per rate: the encoding with the best mean
    rows = sorted(best.items(), key=lambda kv: -kv[0])
    names = [n for _, n in rows]
    colmax = saved.loc[names].max()

    out = [f"{BS}begin{{tabular}}{{@{{}}lr" + "r" * len(budgets) + "r@{}}", f"{BS}toprule",
           f"& & {BS}multicolumn{{{len(budgets)}}}{{c}}{{Memory budget ({BS}% of raw)}} & {BS*2}",
           f"{BS}cmidrule(lr){{3-{2 + len(budgets)}}}",
           "Encoding & b/dim & " + " & ".join(f"{b:.1f}" for b in budgets) + f" & q/s {BS*2}", f"{BS}midrule"]
    for r, name in rows:
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
        out.append(f"{_tex(_label(name))} & {_rate(r)} & " + " & ".join(cells) + f" & {q_s} {BS*2}")
    out += [f"{BS}bottomrule", f"{BS}end{{tabular}}"]
    return "\n".join(out) + "\n"


def write(results: Path, out: Path) -> None:
    paths = sorted(Path(results).glob("cap_quantization/*/*.csv"))
    if not paths:
        raise FileNotFoundError(f"no files match {results / 'cap_quantization/*/*.csv'}")
    tex = table(pd.concat([pd.read_csv(p) for p in paths], ignore_index=True))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex, encoding="utf-8")
    print(f"  {out}")


if __name__ == "__main__":
    args = sys.argv[1:]
    write(ROOT / (args[0] if args else "results"), ROOT / (args[1] if len(args) > 1 else "suggestions/figures/quantization-table.tex"))
