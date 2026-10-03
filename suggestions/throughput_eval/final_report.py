"""Merge final/*.json into throughput_eval.csv, insert_eval.csv and a markdown summary."""
import os, glob, json
import numpy as np, pandas as pd
S = os.path.dirname(os.path.abspath(__file__)); F = os.path.join(S, "final")
key = ["ds", "k", "qn"]
q = pd.DataFrame([json.load(open(p)) for p in glob.glob(os.path.join(F, "*_rep*.json")) if not os.path.basename(p).startswith("ins_")])
metrics = ["single", "b32", "whole", "single_head", "b32_head"]
best = q.groupby(key + ["tag"])[metrics].max().unstack("tag")
best.columns = [f"{m}_{t}" for m, t in best.columns]
first = q[q.rep == 0].set_index(key + ["tag"])[["fpr", "fnr", "n_balls", "batch32_vs_whole_flips"]].unstack("tag")
first.columns = [f"{m}_{t}" for m, t in first.columns]
df = best.join(first)
for m in metrics: df[m + "_x"] = df[m + "_new"] / df[m + "_old"]
flips = []
for (ds, k, qn) in df.index:
    o = np.load(os.path.join(F, f"old_{ds}_{k}_{qn}_pred.npy")); n = np.load(os.path.join(F, f"new_{ds}_{k}_{qn}_pred.npy"))
    n32 = np.load(os.path.join(F, f"new_{ds}_{k}_{qn}_pred32.npy"))
    flips.append(dict(ds=ds, k=k, qn=qn, flips_old_vs_new_whole=int((o != n).sum()), flips_old_whole_vs_new_b32=int((o != n32).sum())))
df = df.join(pd.DataFrame(flips).set_index(key)).reset_index()
df.to_csv(os.path.join(S, "throughput_eval.csv"), index=False)
ins = [json.load(open(p)) for p in glob.glob(os.path.join(F, "ins_*.json"))]
if ins:
    i = pd.DataFrame(ins).set_index(key + ["tag"])[["insert_vps", "cover_sha1", "fnr", "inserted_covered_tau0", "n_balls", "accept_rate"]].unstack("tag")
    i.columns = [f"{m}_{t}" for m, t in i.columns]
    i["insert_x"] = i["insert_vps_new"] / i["insert_vps_old"]
    i["same_cover"] = i["cover_sha1_old"] == i["cover_sha1_new"]
    i.reset_index().to_csv(os.path.join(S, "insert_eval.csv"), index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
cols = ["ds", "k", "qn", "n_balls_new"] + [f"{m}_{s}" for m in ("single", "b32", "whole") for s in ("old", "new", "x")] + ["fpr_new", "fnr_new", "flips_old_vs_new_whole", "flips_old_whole_vs_new_b32"]
print(df[cols].round(2).to_string(index=False))
g = df.groupby(["k", "qn"])[[m + "_x" for m in metrics]].agg(lambda s: float(np.exp(np.log(s).mean())))
print("\ngeometric-mean speedup by (k, encoding):\n", g.round(2).to_string())
if ins: print("\n", i.reset_index()[key + ["insert_vps_old", "insert_vps_new", "insert_x", "same_cover", "fnr_new", "inserted_covered_tau0_new"]].round(2).to_string(index=False))
