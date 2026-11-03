"""Paired summary of bench_construct/variants.py CSVs. Usage: python bench_construct/variants_summary.py f0.csv f123.csv
Pairs each variant with `exact` on (fold, quantize, budget); FPR/objective from rep 0 (deterministic),
construct_s speedup = median over reps of exact/variant within the same rep."""
import sys
import pandas as pd

df = pd.concat([pd.read_csv(p) for p in sys.argv[1:]], ignore_index=True)
key = ["fold", "quantize", "budget"]
ex = df[df.variant == "exact"]
out = []
for v in sorted(set(df.variant) - {"exact"}):
    m = df[df.variant == v].merge(ex, on=key + ["rep"], suffixes=("", "_e"))
    m["d_fpr_pp"] = 100 * (m.fpr - m.fpr_e)
    m["d_theta8"] = m.sum_theta8 / m.sum_theta8_e - 1
    m["d_psi"] = m.psi_mean - m.psi_mean_e
    m["d_caps"] = m.n_caps - m.n_caps_e
    m["speed"] = m.construct_s_e / m.construct_s
    m["mspeed"] = m.method_s_e / m.method_s
    m["variant"] = v
    out.append(m)
m = pd.concat(out)
cell = m.groupby(["variant"] + key).agg(d_fpr_pp=("d_fpr_pp", "first"), d_theta8=("d_theta8", "first"),
                                          d_psi=("d_psi", "first"), d_caps=("d_caps", "first"),
                                          speed=("speed", "median"), mspeed=("mspeed", "median"),
                                          fpr_e=("fpr_e", "first")).reset_index()
cell.to_csv("bench_construct/out/variants_cells.csv", index=False)
pd.set_option("display.width", 200)
sig = cell[cell.quantize.isin(["TQ2", "TQ4"])]
for name, sub in (("ALL cells", cell), ("SIGNAL TQ2+TQ4", sig)):
    print("==", name)
    print(sub.groupby("variant").agg(mean_dfpr=("d_fpr_pp", "mean"), worst_dfpr=("d_fpr_pp", "max"),
                                      mean_dtheta8=("d_theta8", "mean"), mean_dpsi=("d_psi", "mean"),
                                      caps_changed=("d_caps", lambda x: int((x != 0).sum())),
                                      speed=("speed", "median"), n=("speed", "size")))
print("== per encoding")
print(cell.groupby(["variant", "quantize"]).agg(mean_dfpr=("d_fpr_pp", "mean"), worst_dfpr=("d_fpr_pp", "max"),
                                                 mean_dtheta8=("d_theta8", "mean"), speed=("speed", "median")))
print("== per encoding x budget (mean over folds)")
print(cell.groupby(["variant", "quantize", "budget"]).agg(dfpr=("d_fpr_pp", "mean"), worst=("d_fpr_pp", "max"),
                                                           dtheta8=("d_theta8", "mean"), dpsi=("d_psi", "mean"),
                                                           dcaps=("d_caps", "mean"), fpr_exact=("fpr_e", "mean"),
                                                           speed=("speed", "median")).round(4).to_string())
