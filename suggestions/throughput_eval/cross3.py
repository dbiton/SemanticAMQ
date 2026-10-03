import os, sys, time, json
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="1"
import numpy as np, psutil, bc
psutil.Process().cpu_affinity([0])
rows = []
def rate(fn, Q, bs, m):
    m = (m // bs) * bs; t0 = time.perf_counter()
    for s in range(0, m, bs): fn(Q[s:s+bs])
    return m / (time.perf_counter() - t0)
for ds in bc.DATASETS:
  Xc, Q, act = bc.workload(ds)
  P = Q[np.random.default_rng(0).permutation(len(Q))]
  for k in (16384, 2048):
    f = bc.load(ds, k, "TQ4"); f._store_largest_first()
    kern = lambda b: f._kernel_accepts(b, 0.1); scan = lambda b: f._scan(b, 0.1)
    kern(P[:2]); out = []
    for bs, m in ((64, 2048), (256, 4096), (1024, 8192), (20000, 20000)):
        a = max(rate(kern, P, bs, m) for _ in range(2)); g = max(rate(scan, P, bs, m) for _ in range(2))
        rows.append(dict(ds=ds, k=k, bs=bs, kern=a, scan=g)); out.append(f"{bs}: {a:.0f}/{g:.0f}={a/g:.2f}")
    print(ds, k, "kernel/_scan", " | ".join(out), flush=True)
json.dump(rows, open(os.path.join(bc.S, "cross3.json"), "w"))
