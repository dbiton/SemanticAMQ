import os, sys, time, cProfile, pstats
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="1"
import numpy as np, psutil, bc, boot
psutil.Process().cpu_affinity([0])
ds, qn, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
Xc, Q, act = bc.workload(ds)
f = boot.load(ds, qn)
X = Xc[20000:20000 + n]
pr = cProfile.Profile()
t0 = time.perf_counter(); pr.enable()
for s in range(0, n, 1000): f.insert(X[s:s+1000])
pr.disable(); dt = time.perf_counter() - t0
print(ds, qn, "%.0f vectors/s" % (n / dt), "balls", f._num_spheres)
pstats.Stats(pr).sort_stats("tottime").print_stats(18)
