import os, sys, time
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="1"
import numpy as np, psutil, bc
from filters.ball_cover import scan_kernels as sk
psutil.Process().cpu_affinity([int(sys.argv[1])])
for ds in ("WildChat", "CCNews"):
  Xc, Q, act = bc.workload(ds); P = Q[np.random.default_rng(0).permutation(len(Q))][:1500]
  for k in (2048, 16384):
    f = bc.load(ds, k, "TQ4"); f._store_largest_first()
    def tm(fn, n=1500):
        fn(0); t0 = time.perf_counter()
        for i in range(n): fn(i)
        return (time.perf_counter() - t0) / n * 1e6
    ids, codes = f._stored_codes(); radii = f._all_radiuses[ids]; rot = f._rotation()
    T = f._nibble_table(2 * codes.shape[1]); Qr = rot.apply(P)
    parts = dict(total=tm(lambda i: f.query(P[i:i+1], 0.1)),
                 stored_codes=tm(lambda i: f._stored_codes()), radii_gather=tm(lambda i: f._all_radiuses[ids]),
                 rotation=tm(lambda i: rot.apply(P[i:i+1])), decode_table=tm(lambda i: f._nibble_table(2 * codes.shape[1])),
                 kernel=tm(lambda i: sk.nibble_accepts(codes, Qr[i:i+1], T, radii, 0.1, 1e-3)),
                 kernel_tableonly=tm(lambda i: sk.nibble_accepts(codes[:0], Qr[i:i+1], T, radii[:0], 0.1, 1e-3)))
    print(ds, k, " ".join(f"{n}={v:.1f}us" for n, v in parts.items()), flush=True)
