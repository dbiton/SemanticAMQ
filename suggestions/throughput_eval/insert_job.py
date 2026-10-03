"""Time cap_merge inserts from a pickled bootstrap under whichever src/ is on PYTHONPATH. argv: tag ds k qn core rep n"""
import os, sys, time, json, pickle, hashlib
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="1"
import numpy as np, faiss, psutil, bc
from filters.ball_cover.ball_cover_filter import BallCoverFilter
tag, ds, k, qn, core, rep, n = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
psutil.Process().cpu_affinity([core])
Xc, Q, act = bc.workload(ds)
d = pickle.load(open(os.path.join(bc.S, "eval", f"boot_{ds}_{k}_{qn}.pkl"), "rb"))
f = BallCoverFilter.__new__(BallCoverFilter); f.__dict__.update(d["state"]); f.construct_method = None
f.all_centers_index = faiss.deserialize_index(d["index"])
X = Xc[20000:20000 + n]
t0 = time.perf_counter()
for s in range(0, n, 1000): f.insert(X[s:s+1000])
dt = time.perf_counter() - t0
ids, codes = f._stored_codes()
h = hashlib.sha1(); [h.update(np.ascontiguousarray(a).tobytes()) for a in (ids, codes, f._all_radiuses[ids], f.all_spheres)]
pred = np.asarray(f.query(Q, 0.1))
ix = faiss.IndexFlatL2(384); ix.add(np.ascontiguousarray(Xc[:20000 + n])); d2, _ = ix.search(Q, 1)
member = np.sqrt(np.maximum(d2[:, 0], 0)) <= 0.1 - 1e-4      # clearly within tau of an inserted vector
row = dict(tag=tag, ds=ds, k=k, qn=qn, rep=rep, insert_vps=n / dt, n_balls=int(f._num_spheres), cover_sha1=h.hexdigest(),
           inserted_covered_tau0=bool(np.all(f.query(X, 0.0))), accept_rate=float(pred.mean()),
           fnr=float((~pred & member).sum() / max(member.sum(), 1)))
json.dump(row, open(os.path.join(bc.S, "final", f"ins_{tag}_{ds}_{k}_{qn}_rep{rep}.json"), "w"))
print(json.dumps(row), flush=True)
