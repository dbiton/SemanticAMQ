"""Paired construction benchmark (single pinned core). Run from repo root:
  PYTHONPATH=src python bench_construct/bench.py --out OUT.csv [--methods gonzalez,minimax,soft_anneal]
      [--budgets 0.01,0.03,0.10] [--quant FP32,TQ2,TQ4] [--reps 1] [--vectors eli5_10k.pkl]
Records construct_s / method_s (same boundaries as runner.construct_method_worker), cap count, bytes,
FPR/FNR vs exact distance at tau, hard objective. Outputs stay out of results/."""
import os
for v in ("OMP", "OPENBLAS", "MKL", "NUMEXPR"):
    os.environ[f"{v}_NUM_THREADS"] = "1"
import argparse, pickle, time, csv
import numpy as np, psutil
import faiss
faiss.omp_set_num_threads(1)
from runner import (CAP_CONSTRUCT_METHODS, _TimedCapConstruct, _k_for_memory_fraction)
from filters.cap_cover.cap_cover_filter import CapCoverFilter
from filters.cap_cover.cap_quantizers import Quantize as CapQuantize

TAU = 10 ** -0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--vectors", default=os.environ.get("ELI5_PKL"))
    ap.add_argument("--methods", default="gonzalez,minimax,soft_anneal")
    ap.add_argument("--budgets", default="0.01,0.03,0.10")
    ap.add_argument("--quant", default="FP32")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--core", type=int, default=0)
    a = ap.parse_args()
    try:
        psutil.Process().cpu_affinity([a.core])
    except Exception as e:
        print("affinity not set:", e)
    V = pickle.load(open(a.vectors, "rb")).astype(np.float32)
    n = int(0.8 * len(V)); C, Q = V[:n], V[n:]
    d = C.shape[1]
    index = faiss.IndexFlatL2(d); index.add(C)
    dist = np.sqrt(np.maximum(index.search(Q, 1)[0][:, 0], 0))
    actual = dist <= TAU
    rows = []
    for method in a.methods.split(","):
        for q in a.quant.split(","):
            for b in map(float, a.budgets.split(",")):
                for rep in range(a.reps):
                    timed = _TimedCapConstruct(CAP_CONSTRUCT_METHODS[method]())
                    filt = CapCoverFilter(d, construct_method=timed, max_memory=int(b * n * d * 4),
                                          quantize=CapQuantize(q), insert_method=None)
                    cpu0 = time.process_time(); t0 = time.perf_counter()
                    filt.construct(C)
                    construct_s = time.perf_counter() - t0; cpu_s = time.process_time() - cpu0
                    pred = np.asarray(filt.query(Q, distance_threshold=TAU), dtype=bool)
                    fp = int((pred & ~actual).sum()); fn = int((~pred & actual).sum())
                    ang = filt.all_angles
                    r = dict(method=method, quantize=q, budget=b, rep=rep, method_s=timed.seconds,
                             construct_s=construct_s, cpu_s=cpu_s, n_caps=len(ang), bytes=filt.get_bytesize(),
                             max_memory=int(b * n * d * 4), fpr=fp / max(1, int((~actual).sum())), fn=fn,
                             psi_mean=float(ang.mean()), psi_max=float(ang.max()),
                             peak_rss_mb=psutil.Process().memory_info().rss / 2**20)
                    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
                    rows.append(r)
                    with open(a.out, "w", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

if __name__ == "__main__":
    main()
