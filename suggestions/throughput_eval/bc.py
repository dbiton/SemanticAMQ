"""Shared harness: load a saved cover + workload into a BallCoverFilter."""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ.setdefault(v, "1")
import pickle, numpy as np, faiss
from filters.ball_cover.ball_cover_filter import BallCoverFilter, Quantize
S = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(S, "eval", "out")
DATASETS = ["WildChat", "CCNews", "WikiAnswers", "Quora", "ChatGPTParaphrases"]
def workload(ds):
    d = np.load(os.path.join(S, "eval", f"{ds}.npz")); return d["Xc"], d["Q"], d["actual"]
def load(ds, k, qn, kind="static", tag="new"):
    c = pickle.load(open(os.path.join(OUT, f"{tag}_{kind}_{ds}_{k}_{qn}_cover.pkl"), "rb"))
    f = BallCoverFilter(384, quantize=Quantize(qn), max_spheres=len(c["radii"]), insert_method=None)
    f.all_centers_index = faiss.deserialize_index(c["index"]); f._num_spheres = c["n"]
    f._all_radiuses[:] = c["radii"]; f._all_spheres[:c["n"]] = c["spheres"]; f._constructed = True
    return f
