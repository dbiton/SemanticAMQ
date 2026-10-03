"""Multi-thread scaling of a whole-batch query: kernel chunks across Python threads (nogil),
_scan with T BLAS threads. Pinned to T physical P-cores (one logical CPU each)."""
import os, sys, time, json
import numpy as np, psutil
from concurrent.futures import ThreadPoolExecutor
from threadpoolctl import threadpool_limits
import bc
from filters.ball_cover import scan_kernels as sk
CORES = [int(c) for c in sys.argv[1].split(",")]
rows = []
for ds in ("WildChat", "CCNews"):
  Xc, Q, act = bc.workload(ds)
  P = Q[np.random.default_rng(0).permutation(len(Q))][:8192]
  for k, qn in ((16384, "TQ4"), (16384, "FP32"), (2048, "TQ4")):
    f = bc.load(ds, k, qn); f._store_largest_first()
    ids, codes = f._stored_codes(); radii = f._all_radiuses[ids]
    rot = f._rotation(); Qr = np.ascontiguousarray(rot.apply(P) if rot is not None else P, dtype=np.float32)
    if qn == "FP32":
        C = codes.view(np.float32); kern = lambda b: sk.flat_accepts(C, b, radii, 0.1, 1e-3)
    else:
        T = f._nibble_table(2 * codes.shape[1]); kern = lambda b: sk.nibble_accepts(codes, b, T, radii, 0.1, 1e-3)
    kern(Qr[:2]); ref = kern(Qr)
    for t in (1, 2, 4, 8):
        if t > len(CORES): continue
        psutil.Process().cpu_affinity(CORES[:t])
        chunks = [Qr[i::t] for i in range(t)]   # interleaved, so accept rates balance
        with ThreadPoolExecutor(t) as ex:
            best_k = 0
            for rep in range(3):
                t0 = time.perf_counter(); outs = list(ex.map(kern, chunks)); dt = time.perf_counter() - t0
                best_k = max(best_k, len(Qr) / dt)
        got = np.empty(len(Qr), bool)
        for i in range(t): got[i::t] = outs[i]
        assert (got == ref).all()
        import faiss; faiss.omp_set_num_threads(t)
        with threadpool_limits(limits=t, user_api="blas"):
            best_g = 0
            for rep in range(2):
                t0 = time.perf_counter(); f._scan(P, 0.1); best_g = max(best_g, len(P) / (time.perf_counter() - t0))
        rows.append(dict(ds=ds, k=k, qn=qn, threads=t, kernel_qps=best_k, scan_qps=best_g))
        print(ds, k, qn, "threads", t, "kernel %.0f q/s, _scan(BLAS) %.0f q/s" % (best_k, best_g), flush=True)
json.dump(rows, open(os.path.join(bc.S, "threads.json"), "w"))
