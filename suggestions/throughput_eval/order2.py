import os, sys, time
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="1"
import numpy as np, psutil, bc
from filters.ball_cover import scan_kernels as sk
psutil.Process().cpu_affinity([0])
for ds in bc.DATASETS:
  Xc, Q, act = bc.workload(ds)
  P = Q[np.random.default_rng(0).permutation(len(Q))][:2048]
  for k in (16384, 2048):
    f = bc.load(ds, k, "TQ4")
    ids, codes = f._stored_codes(); r = f._all_radiuses[ids]
    T = f._nibble_table(2 * codes.shape[1]); Qr = f._rotation().apply(P)
    res = {}
    for name, order in (("storage", np.arange(len(ids))), ("radius_desc", np.argsort(-r, kind="stable")), ("radius_asc", np.argsort(r, kind="stable"))):
        c = np.ascontiguousarray(codes[order]); rr = np.ascontiguousarray(r[order])
        sk.nibble_accepts(c, Qr[:2], T, rr, 0.1, 1e-3)
        best = 0
        for rep in range(3):
            t0 = time.perf_counter()
            for s in range(0, len(Qr), 32): sk.nibble_accepts(c, Qr[s:s+32], T, rr, 0.1, 1e-3)
            best = max(best, len(Qr) / (time.perf_counter() - t0))
        res[name] = best
    print(ds, k, " ".join(f"{n}={v:.0f}" for n, v in res.items()), "gain %.2fx" % (res["radius_desc"] / res["storage"]), flush=True)
