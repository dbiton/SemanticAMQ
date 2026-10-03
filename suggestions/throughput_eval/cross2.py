import os, sys, time
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="1"
import numpy as np, psutil, bc, json
psutil.Process().cpu_affinity([int(sys.argv[1])])
rows = []
def rate(fn, Q, bs, budget=0.6):
    n = 0; t0 = time.perf_counter()
    while True:
        s = (n * bs) % (len(Q) - bs); fn(Q[s:s+bs]); n += 1
        dt = time.perf_counter() - t0
        if dt > budget and n >= 3: return n * bs / dt
for ds in sys.argv[2].split(","):
  Xc, Q, act = bc.workload(ds)
  P = Q[np.random.default_rng(0).permutation(len(Q))]
  for k, qn in ((16384, "TQ4"), (2048, "TQ4"), (16384, "FP32"), (2048, "FP32")):
    f = bc.load(ds, k, qn)
    kern = lambda b: f._kernel_accepts(b, 0.1)
    other = lambda b: (f._range_accepts(b, 0.1) if len(b) < f._SCAN_MIN_BATCH else f._scan(b, 0.1))
    kern(P[:2])
    out = []
    for bs in (1, 2, 4, 8, 16, 32, 64, 128, 256, 1024):
        a, g = rate(kern, P, bs), rate(other, P, bs)
        rows.append(dict(ds=ds, k=k, qn=qn, bs=bs, kern=a, other=g)); out.append(f"{bs}:{a/g:.2f}")
    print(ds, k, qn, "kern/other", " ".join(out), flush=True)
json.dump(rows, open(os.path.join(bc.S, "cross2_%s.json" % sys.argv[2].replace(",", "_")), "w"))
