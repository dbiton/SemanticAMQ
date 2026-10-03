"""One timing pass of query() on a saved static cover, under whichever src/ is on PYTHONPATH.
argv: tag ds k qn core rep. The new tree stores the cover largest-first, as its construct() does."""
import os, sys, time, json
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="1"
import numpy as np, psutil, bc
tag, ds, k, qn, core, rep = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5]), int(sys.argv[6])
psutil.Process().cpu_affinity([core])
Xc, Q, act = bc.workload(ds)
P = Q[np.random.default_rng(0).permutation(len(Q))]
f = bc.load(ds, k, qn)
if hasattr(f, "_store_largest_first"): f._store_largest_first()
f.query(Q[:40], 0.1); f.query(Q[:1], 0.1)        # warm-up (numba cache load in the new tree)
def rate(X, bs, m):
    m = (m // bs) * bs; t0 = time.perf_counter()
    for s in range(0, m, bs): f.query(X[s:s+bs], 0.1)
    return m / (time.perf_counter() - t0)
row = dict(tag=tag, ds=ds, k=k, qn=qn, rep=rep, n_balls=int(f._num_spheres))
row["single"] = rate(P, 1, 400); row["b32"] = rate(P, 32, 2048)
row["single_head"] = rate(Q, 1, 300); row["b32_head"] = rate(Q, 32, 4096)
t0 = time.perf_counter(); pred = np.asarray(f.query(Q, 0.1)); row["whole"] = len(Q) / (time.perf_counter() - t0)
fp = int((pred & ~act).sum()); tn = int((~pred & ~act).sum()); fn = int((~pred & act).sum()); tp = int((pred & act).sum())
row.update(fpr=fp / (fp + tn), fnr=fn / max(fn + tp, 1))
# decisions in batches of 32 too, which is the kernel path in the new tree
pred32 = np.concatenate([np.asarray(f.query(Q[s:s+32], 0.1)) for s in range(0, len(Q), 32)])
row["batch32_vs_whole_flips"] = int((pred32 != pred).sum())
name = f"{tag}_{ds}_{k}_{qn}"
if rep == 0:
    np.save(os.path.join(bc.S, "final", name + "_pred.npy"), pred); np.save(os.path.join(bc.S, "final", name + "_pred32.npy"), pred32)
json.dump(row, open(os.path.join(bc.S, "final", f"{name}_rep{rep}.json"), "w"))
print(json.dumps(row), flush=True)
