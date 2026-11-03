"""Summarise the controlled old-vs-new run (bench_construct/run_final.sh) as markdown on stdout.
Per method: median paired speedup (cell medians over reps), ratio of aggregate runtimes, slowest/regressing cells,
plus quality checks (caps, FNR, bytes<=budget, FPR delta in percentage points)."""
import sys, pandas as pd
out = sys.argv[1] if len(sys.argv) > 1 else "bench_construct/out"
K = ["method", "quantize", "budget"]
frames = []
for tag in ("gm", "sa"):
    try:
        o = pd.read_csv(f"{out}/final_{tag}_old.csv"); n = pd.read_csv(f"{out}/final_{tag}_new.csv")
    except FileNotFoundError:
        continue
    agg = lambda d: d.groupby(K).agg(construct_s=("construct_s", "median"), method_s=("method_s", "median"),
                                      n_caps=("n_caps", "first"), fpr=("fpr", "first"), fn=("fn", "max"),
                                      bytes=("bytes", "max"), max_memory=("max_memory", "first"),
                                      reps=("rep", "count")).reset_index()
    frames.append(agg(o).merge(agg(n), on=K, suffixes=("_old", "_new")))
m = pd.concat(frames, ignore_index=True)
m["speedup_total"] = m.construct_s_old / m.construct_s_new
m["speedup_method"] = m.method_s_old / m.method_s_new
m["dfpr_pp"] = (m.fpr_new - m.fpr_old) * 100
print("| method | cells | median total speedup | aggregate ratio (total) | median method speedup | min / max total | mean dFPR (pp) | max dFPR (pp) | caps equal | FN | bytes ok |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for meth, g in m.groupby("method"):
    print(f"| {meth} | {len(g)} | {g.speedup_total.median():.2f}x | {g.construct_s_old.sum()/g.construct_s_new.sum():.2f}x | "
          f"{g.speedup_method.median():.2f}x | {g.speedup_total.min():.2f}x / {g.speedup_total.max():.2f}x | "
          f"{g.dfpr_pp.mean():+.3f} | {g.dfpr_pp.abs().max():.2f} | {(g.n_caps_old==g.n_caps_new).all()} | "
          f"{int(g.fn_new.max())} | {(g.bytes_new<=g.max_memory_new).all()} |")
print("\n### Per encoding (median total speedup)\n")
print(m.pivot_table(index="method", columns="quantize", values="speedup_total", aggfunc="median").round(2).to_markdown())
print("\n### Slowest-to-improve cells (lowest total speedup)\n")
print(m.sort_values("speedup_total").head(8)[K + ["construct_s_old", "construct_s_new", "speedup_total", "dfpr_pp"]].round(3).to_markdown(index=False))
print("\n### Cells with n_caps differing (if any)\n")
d = m[m.n_caps_old != m.n_caps_new]
print(d[K + ["n_caps_old", "n_caps_new"]].to_markdown(index=False) if len(d) else "none")
