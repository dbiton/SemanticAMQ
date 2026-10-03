"""One evaluation job against whichever src/ is on PYTHONPATH.
argv: tag kind dataset k quantize core
kind = static | stream"""
import os, sys
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="1"
import time, json, pickle, numpy as np, faiss, psutil
tag, kind, dataset, k, qname, core = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5], int(sys.argv[6])
psutil.Process().cpu_affinity([core])
import runner
from filters.ball_cover.ball_cover_filter import BallCoverFilter, Quantize
from filters.ball_cover.cap_merge_insert import CapMergeInsert
S = os.environ["S"]; out = os.path.join(S, "eval", "out"); os.makedirs(out, exist_ok=True)
name = f"{tag}_{kind}_{dataset}_{k}_{qname}"
d = np.load(os.path.join(S, "eval", f"{dataset}.npz")); Xc, Q, actual = d["Xc"], d["Q"], d["actual"]
qz = Quantize(qname); TAU = 0.1
def rates(pred):
    pred = np.asarray(pred, bool); fp = (pred & ~actual).sum(); tn = (~pred & ~actual).sum(); fn = (~pred & actual).sum(); tp = (pred & actual).sum()
    return float(fp/(fp+tn)), float(fn/max(fn+tp,1))
row = dict(tag=tag, kind=kind, dataset=dataset, k=k, quantize=qname)
mm = runner._ball_cover_max_memory(384, k, qz)
cm = runner.CONSTRUCT_METHODS[runner.BALL_COVER_CONSTRUCT]()
if kind == "static":
    f = BallCoverFilter(384, max_memory=mm, quantize=qz, construct_method=cm, insert_method=None)
    t0 = time.perf_counter(); f.construct(Xc); row["construct_s"] = time.perf_counter() - t0
else:
    f = BallCoverFilter(384, max_memory=mm, quantize=qz, construct_method=cm, insert_method=CapMergeInsert())
    n0 = 20000
    t0 = time.perf_counter(); f.construct(Xc[:n0]); row["construct_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    for s in range(n0, len(Xc), 1000): f.insert(Xc[s:s+1000])
    row["insert_s"] = time.perf_counter() - t0; row["insert_vps"] = (len(Xc)-n0) / row["insert_s"]
    row["fnr_construct_tau0"] = 1.0 - float(np.mean(f.query(Xc, 0.0)))
row["n_balls"] = int(f._num_spheres); row["bytes"] = int(f.get_bytesize())
t0 = time.perf_counter(); pred = np.asarray(f.query(Q, TAU)); row["qps_batched"] = len(Q)/(time.perf_counter()-t0)
row["fpr"], row["fnr"] = rates(pred); row["accept_rate"] = float(pred.mean())
m = 4000; t0 = time.perf_counter()
for s in range(0, m, 32): f.query(Q[s:s+32], TAU)
row["qps_batch32"] = m/(time.perf_counter()-t0)
m = 300; t0 = time.perf_counter()
for s in range(m): f.query(Q[s:s+1], TAU)
row["qps_single"] = m/(time.perf_counter()-t0)
t0 = time.perf_counter(); surf = f.surface_distance(Q); row["qps_surface"] = len(Q)/(time.perf_counter()-t0)
row["surface_vs_query_mismatch"] = int(((surf <= TAU + f._COVERAGE_EPSILON) != pred).sum())
np.save(os.path.join(out, name + "_pred.npy"), pred); np.save(os.path.join(out, name + "_surf.npy"), surf)
pickle.dump(dict(index=faiss.serialize_index(f.all_centers_index), radii=f._all_radiuses.copy(),
                 spheres=f.all_spheres.copy(), n=f._num_spheres), open(os.path.join(out, name + "_cover.pkl"), "wb"))
json.dump(row, open(os.path.join(out, name + ".json"), "w"))
print(json.dumps(row), flush=True)
