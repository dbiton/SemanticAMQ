"""Red-team cell runner. PYTHONPATH=<src of version under test> python fpr_cell.py OUT.csv CORE
Per (method, quantize, budget): FNR vs exact faiss ground truth, bytes<=max_memory, every construct
vector accepted at tau=0, n_caps, psi_mean, FPR. Same code runs against old and new trees."""
import os, sys
for v in ("OMP", "OPENBLAS", "MKL", "NUMEXPR"):
    os.environ[f"{v}_NUM_THREADS"] = "1"
import csv, pickle, time
import numpy as np, psutil, faiss
faiss.omp_set_num_threads(1)
out, core = sys.argv[1], int(sys.argv[2])
psutil.Process().cpu_affinity([core])
from runner import CAP_CONSTRUCT_METHODS
from filters.cap_cover.cap_cover_filter import CapCoverFilter
from filters.cap_cover.cap_quantizers import Quantize
PKL = os.environ.get("ELI5_PKL", "/tmp/claude-0/-home-user/faeeddb3-925b-5717-b1cb-caa0100e8a36/scratchpad/eli5/eli5_10k.pkl")
TAU = 10 ** -0.5
V = pickle.load(open(PKL, "rb")).astype(np.float32)
C, Q = V[:8000], V[8000:]
n, d = C.shape
idx = faiss.IndexFlatL2(d); idx.add(C)
actual = np.sqrt(np.maximum(idx.search(Q, 1)[0][:, 0], 0)) <= TAU
rows = []
for method in ("gonzalez", "minimax"):
    for qn in ("FP32", "TQ2", "TQ4"):
        for b in (0.01, 0.03, 0.10):
            mm = int(b * n * d * 4)
            f = CapCoverFilter(d, construct_method=CAP_CONSTRUCT_METHODS[method](), max_memory=mm,
                               quantize=Quantize[qn], insert_method=None)
            t = time.perf_counter(); f.construct(C); ct = time.perf_counter() - t
            pred = np.asarray(f.query(Q, TAU), dtype=bool)
            mem0 = np.asarray(f.query(C, 0.0), dtype=bool)
            ang = f.all_angles
            r = dict(method=method, quantize=qn, budget=b, n_caps=len(ang), psi_mean=float(ang.mean()),
                     fpr=float((pred & ~actual).sum() / max(1, (~actual).sum())), fn=int((~pred & actual).sum()),
                     bytes=f.get_bytesize(), max_memory=mm, bytes_ok=f.get_bytesize() <= mm,
                     members_missed_tau0=int((~mem0).sum()), construct_s=ct)
            print(r, flush=True); rows.append(r)
            with open(out, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
