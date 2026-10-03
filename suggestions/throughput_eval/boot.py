"""Build (or load) the cap_merge streaming bootstrap: minimax_sphere on the first
20K construct vectors at k=8192, as eval_job.py's stream mode does."""
import os, sys, time, pickle
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="1"
import numpy as np, faiss, bc
import runner
from filters.ball_cover.ball_cover_filter import BallCoverFilter, Quantize
from filters.ball_cover.cap_merge_insert import CapMergeInsert
N0 = 20000
def path(ds, qn): return os.path.join(bc.S, "eval", f"boot_{ds}_8192_{qn}.pkl")
def build(ds, qn):
    Xc, Q, act = bc.workload(ds); qz = Quantize(qn)
    f = BallCoverFilter(384, max_memory=runner._ball_cover_max_memory(384, 8192, qz), quantize=qz,
                        construct_method=runner.CONSTRUCT_METHODS[runner.BALL_COVER_CONSTRUCT](), insert_method=CapMergeInsert())
    t0 = time.perf_counter(); f.construct(Xc[:N0]); print(ds, qn, "construct %.1fs" % (time.perf_counter() - t0), flush=True)
    st = {k: v for k, v in f.__dict__.items() if k not in ("all_centers_index", "construct_method")}
    pickle.dump(dict(state=st, index=faiss.serialize_index(f.all_centers_index)), open(path(ds, qn), "wb"))
def load(ds, qn):
    d = pickle.load(open(path(ds, qn), "rb"))
    f = BallCoverFilter.__new__(BallCoverFilter); f.__dict__.update(d["state"])
    f.construct_method = None; f.all_centers_index = faiss.deserialize_index(d["index"])
    return f
if __name__ == "__main__":
    import psutil; psutil.Process().cpu_affinity([int(sys.argv[1])])
    for ds in ("WildChat", "CCNews"):
        for qn in ("TQ4", "FP32"):
            if not os.path.exists(path(ds, qn)): build(ds, qn)
