"""Build one static paper-config cover (minimax_sphere + encoding) on the full 80K construct set and save it like eval_job.py."""
import os, sys, time, pickle, json
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="1"
import numpy as np, faiss, psutil, bc
ds, k, qn, core = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
psutil.Process().cpu_affinity([core])
import runner
from filters.ball_cover.ball_cover_filter import BallCoverFilter, Quantize
Xc, Q, act = bc.workload(ds); qz = Quantize(qn)
f = BallCoverFilter(384, max_memory=runner._ball_cover_max_memory(384, k, qz), quantize=qz,
                    construct_method=runner.CONSTRUCT_METHODS[runner.BALL_COVER_CONSTRUCT](), insert_method=None)
t0 = time.perf_counter(); f.construct(Xc); dt = time.perf_counter() - t0
pickle.dump(dict(index=faiss.serialize_index(f.all_centers_index), radii=f._all_radiuses.copy(), spheres=f.all_spheres.copy(), n=f._num_spheres),
            open(os.path.join(bc.OUT, f"new_static_{ds}_{k}_{qn}_cover.pkl"), "wb"))
pred = np.asarray(f.query(Q, 0.1)); np.save(os.path.join(bc.OUT, f"new_static_{ds}_{k}_{qn}_pred.npy"), pred)
print(ds, k, qn, "construct %.0fs" % dt, "balls", f._num_spheres, "FNR", float((~pred & act).sum() / max(act.sum(), 1)), flush=True)
