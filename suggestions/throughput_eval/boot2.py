"""Build a cap_merge streaming bootstrap (minimax_sphere on the first 20K construct vectors) and pickle its state.
argv: ds k qn core. Construction is identical in both trees for a filter with an insert strategy."""
import os, sys, time, pickle
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="1"
import numpy as np, faiss, psutil, bc
ds, k, qn, core = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
psutil.Process().cpu_affinity([core])
import runner
from filters.ball_cover.ball_cover_filter import BallCoverFilter, Quantize
from filters.ball_cover.cap_merge_insert import CapMergeInsert
Xc, Q, act = bc.workload(ds); qz = Quantize(qn)
f = BallCoverFilter(384, max_memory=runner._ball_cover_max_memory(384, k, qz), quantize=qz,
                    construct_method=runner.CONSTRUCT_METHODS[runner.BALL_COVER_CONSTRUCT](), insert_method=CapMergeInsert())
t0 = time.perf_counter(); f.construct(Xc[:20000]); dt = time.perf_counter() - t0
st = {a: v for a, v in f.__dict__.items() if a not in ("all_centers_index", "construct_method")}
out = os.path.join(bc.S, "eval", f"boot_{ds}_{k}_{qn}.pkl")
pickle.dump(dict(state=st, index=faiss.serialize_index(f.all_centers_index)), open(out, "wb"))
print(ds, k, qn, "bootstrap %.0fs" % dt, "balls", f._num_spheres, flush=True)
