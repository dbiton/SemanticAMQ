import os, sys, time
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="1"
import numpy as np, psutil, bc
psutil.Process().cpu_affinity([int(sys.argv[1])])
for ds in bc.DATASETS:
  Xc, Q, act = bc.workload(ds)
  for k, qn in ((16384, "TQ4"), (2048, "TQ4"), (16384, "FP32"), (2048, "FP32")):
    f = bc.load(ds, k, qn)
    old = np.load(os.path.join(bc.OUT, f"new_static_{ds}_{k}_{qn}_pred.npy"))
    t0 = time.perf_counter()
    got = np.concatenate([f._kernel_accepts(Q[s:s+32], 0.1) for s in range(0, len(Q), 32)])
    dt = time.perf_counter() - t0
    via_query = np.concatenate([np.asarray(f.query(Q[s:s+32], 0.1)) for s in range(0, 2016, 32)])
    fp = (got & ~act).sum(); tn = (~got & ~act).sum(); fn = (~got & act).sum(); tp = (got & act).sum()
    print(f"{ds} {k} {qn}: flips vs saved {int((got != old).sum())}, query() flips {int((via_query != old[:2016]).sum())}, "
          f"FPR {fp/(fp+tn):.4f} FNR {fn/max(fn+tp,1):.4f}, {len(Q)/dt:.0f} q/s", flush=True)
    assert fn == 0
