"""Is the encoding the volume criterion picks the one with the lowest query FPR? (fold 0, ELI5 8k, diagnostic)
  PYTHONPATH=src python bench_construct/fpr_select.py [method]"""
import os
for v in ("OMP", "OPENBLAS", "MKL", "NUMEXPR"):
    os.environ[f"{v}_NUM_THREADS"] = "1"
import sys, pickle, time
import numpy as np, psutil, faiss
faiss.omp_set_num_threads(1)
from runner import CAP_CONSTRUCT_METHODS, _cap_cover
from filters.cap_cover.cap_cover_filter import CapCoverFilter
from filters.cap_cover.cap_quantizers import Quantize
TAU = 10 ** -0.5
PKL = "/tmp/claude-0/-home-user/faeeddb3-925b-5717-b1cb-caa0100e8a36/scratchpad/eli5/eli5_10k.pkl"
psutil.Process().cpu_affinity([1])
V = pickle.load(open(PKL, "rb")).astype(np.float32)
C, Q = V[:8000], V[8000:]
d = C.shape[1]
idx = faiss.IndexFlatL2(d); idx.add(C)
actual = np.sqrt(np.maximum(idx.search(Q, 1)[0][:, 0], 0)) <= TAU
method = sys.argv[1] if len(sys.argv) > 1 else "minimax"
for b in (0.03, 0.10):
    mm = int(b * 8000 * d * 4)
    for qn in ("auto", "automix"):
        kw = dict(construct_method=CAP_CONSTRUCT_METHODS[method](), max_memory=mm, insert_method=None)
        t0 = time.perf_counter()
        f = CapCoverFilter.auto(d, mixed=(qn == "automix"), **kw)
        f.construct(C)
        s = time.perf_counter() - t0
        p = np.asarray(f.query(Q, TAU), bool)
        print(method, b, qn, "->", f.encoding_label, "caps", len(f.all_angles), "bytes", f.get_bytesize(), "/", mm,
              "fpr %.4f" % ((p & ~actual).sum() / (~actual).sum()), "fn", int((~p & actual).sum()), "t %.1fs" % s,
              "scores", getattr(f, "selection_scores", None), flush=True)
