"""Paired summary of fpr_spend CSVs: per (ladder, budget) mean/worst delta FPR (spend - reference), time ratio.
  python bench_construct/fpr_spend_summary.py out/fpr_spend_f0.csv out/fpr_spend_f123.csv > out/fpr_spend_summary.csv"""
import sys, csv
import numpy as np
rows = [r for p in sys.argv[1:] for r in csv.DictReader(open(p))]
key = lambda r: (r["fold"], r["method"], r["ladder"], r["budget"])
ref = {key(r): r for r in rows if r["spend"] == "0"}
new = {key(r): r for r in rows if r["spend"] == "1"}
out = []
for grp, sel in (("fold0", lambda k: k[0] == "0"), ("folds1-3", lambda k: k[0] != "0")):
    for lad in ("TQ", "SPQ", "all"):
        for bud in ("0.03", "0.1", "all"):
            ks = [k for k in ref if sel(k) and (lad == "all" or k[2] == lad) and (bud == "all" or k[3] == bud)]
            if not ks:
                continue
            d = np.array([float(new[k]["fpr"]) - float(ref[k]["fpr"]) for k in ks]) * 100
            t = np.array([float(new[k]["construct_s"]) / float(ref[k]["construct_s"]) for k in ks])
            out.append(dict(group=grp, ladder=lad, budget=bud, cells=len(ks), mean_dfpr_pp=round(d.mean(), 3),
                            worst_dfpr_pp=round(d.max(), 3), best_dfpr_pp=round(d.min(), 3),
                            mean_time_ratio=round(t.mean(), 3), max_time_ratio=round(t.max(), 3),
                            fn_total=sum(int(new[k]["fn"]) + int(ref[k]["fn"]) for k in ks),
                            over_budget=sum(1 - int(new[k]["ok"]) for k in ks)))
w = csv.DictWriter(sys.stdout, fieldnames=list(out[0]))
w.writeheader(); w.writerows(out)
