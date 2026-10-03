"""A/B: saved static covers in construction order vs _store_largest_first, interleaved per cell."""
import os, sys, time, json
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="1"
import numpy as np, psutil, bc
psutil.Process().cpu_affinity([int(sys.argv[1])])
def rate(f, Q, bs, m):
    m = (m // bs) * bs; t0 = time.perf_counter()
    for s in range(0, m, bs): f.query(Q[s:s+bs], 0.1)
    return m / (time.perf_counter() - t0)
rows = []
for ds in bc.DATASETS:
  Xc, Q, act = bc.workload(ds)
  P = Q[np.random.default_rng(0).permutation(len(Q))]
  for k, qn in ((16384, "TQ4"), (2048, "TQ4"), (16384, "FP32"), (2048, "FP32")):
    a = bc.load(ds, k, qn); b = bc.load(ds, k, qn); b._store_largest_first()
    flips = int((np.asarray(a.query(Q, 0.1)) != np.asarray(b.query(Q, 0.1))).sum())
    r = dict(ds=ds, k=k, qn=qn, flips=flips)
    for name, bs, m in (("single", 1, 400), ("b32", 32, 2048), ("whole", 20000, 20000)):
        ra, rb = [], []
        for rep in range(2):
            ra.append(rate(a, P, bs, m)); rb.append(rate(b, P, bs, m))
        r[name + "_plain"], r[name + "_sorted"] = max(ra), max(rb)
    rows.append(r)
    print(ds, k, qn, "flips", flips, " ".join(f"{n}: {r[n+'_plain']:.0f}->{r[n+'_sorted']:.0f} ({r[n+'_sorted']/r[n+'_plain']:.2f}x)" for n in ("single", "b32", "whole")), flush=True)
json.dump(rows, open(os.path.join(bc.S, "reorder_eval.json"), "w"))
