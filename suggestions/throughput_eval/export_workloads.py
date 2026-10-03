"""Export the five blocklist workloads (80K construct, 20K queries, ground truth at tau=0.1) to
<this folder>/eval/<corpus>.npz, which every other script here reads (via bc.py).

Run it with the working directory in a THROWAWAY worktree of the repo (PYTHONPATH=src): loading the
blocklist workloads can create cache files under results/blocklist_clusters/, and results/ has no backup."""
import os, sys
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="1"
import numpy as np, main
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval"); os.makedirs(out, exist_ok=True)
for w in main.blocklist_workloads(main.WORKLOAD_NUM_VECTORS, n_folds=1):
    sel = w.selections[0]
    Xc = w.rows(sel["reference"]); Q = w.query_vectors(sel)
    actual = np.asarray(main.compute_static_actual_results(Xc, Q, 0.1), dtype=bool)
    np.savez(os.path.join(out, f"{w.name}.npz"), Xc=Xc, Q=Q, actual=actual)
    print(w.name, Xc.shape, Q.shape, "positives", actual.mean(), flush=True)
