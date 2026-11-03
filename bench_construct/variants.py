"""Paired evaluation of opt-in MinimaxConstruct variants vs the exact defaults. Run from repo root:
  PYTHONPATH=src python bench_construct/variants.py --out bench_construct/out/variants.csv \
      --vectors eli5_10k.pkl [--folds 0,1,2,3] [--variants exact,adaptive,approx] [--reps 3]
Fold 0 = original order (first 8000 construct, last 2000 queries); fold s>0 = permutation of the 10000 rows
with seed s, same 8000/2000 split. Variants run back to back (rotating order per rep) in one process
so timing is paired. Ground truth: exact nearest-construct L2 distance (IndexFlatL2) <= tau.
Hard objective = sum theta^8 and mean psi over the final exact cover (construct-set criteria);
FPR/FNR on the held-out queries is validation only."""
import os
for v in ("OMP", "OPENBLAS", "MKL", "NUMEXPR"):
    os.environ[f"{v}_NUM_THREADS"] = "1"
import argparse, pickle, time, csv
import numpy as np, psutil
import faiss
faiss.omp_set_num_threads(1)
from runner import _TimedCapConstruct
from filters.cap_cover.cap_cover_filter import CapCoverFilter
from filters.cap_cover.cap_quantizers import Quantize as CapQuantize
from filters.cap_cover.minimax_construct import MinimaxConstruct

TAU = 10 ** -0.5
VARIANTS = {
    "exact": dict(),
    "adaptive": dict(adaptive=True),
    "approx": dict(approx_assign=True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--vectors", default=os.environ.get("ELI5_PKL"))
    ap.add_argument("--variants", default="exact,adaptive,approx")
    ap.add_argument("--folds", default="0,1,2,3")
    ap.add_argument("--budgets", default="0.01,0.03,0.05,0.07,0.10")
    ap.add_argument("--quant", default="FP32,TQ2,TQ4")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--core", type=int, default=3)
    a = ap.parse_args()
    try:
        psutil.Process().cpu_affinity([a.core])
    except Exception as e:
        print("affinity not set:", e)
    V0 = pickle.load(open(a.vectors, "rb")).astype(np.float32)
    names = a.variants.split(",")
    rows = []
    for fold in map(int, a.folds.split(",")):
        V = V0 if fold == 0 else V0[np.random.default_rng(fold).permutation(len(V0))]
        n = int(0.8 * len(V)); C, Q = V[:n], V[n:]
        d = C.shape[1]
        index = faiss.IndexFlatL2(d); index.add(C)
        dist = np.sqrt(np.maximum(index.search(Q, 1)[0][:, 0], 0))
        actual = dist <= TAU
        for q in a.quant.split(","):
            for b in map(float, a.budgets.split(",")):
                for rep in range(a.reps):
                    s = rep % len(names)
                    for v in names[s:] + names[:s]:  # rotate order each rep
                        timed = _TimedCapConstruct(MinimaxConstruct(**VARIANTS[v]))
                        filt = CapCoverFilter(d, construct_method=timed, max_memory=int(b * n * d * 4),
                                              quantize=CapQuantize(q), insert_method=None)
                        t0 = time.perf_counter()
                        filt.construct(C)
                        construct_s = time.perf_counter() - t0
                        pred = np.asarray(filt.query(Q, distance_threshold=TAU), dtype=bool)
                        fp = int((pred & ~actual).sum()); fn = int((~pred & actual).sum())
                        ang = np.asarray(filt.all_angles, dtype=np.float64)
                        r = dict(fold=fold, variant=v, quantize=q, budget=b, rep=rep, method_s=timed.seconds,
                                 construct_s=construct_s, n_caps=len(ang), bytes=filt.get_bytesize(),
                                 fpr=fp / max(1, int((~actual).sum())), fn=fn,
                                 sum_theta8=float(np.sum(ang ** 8)), psi_mean=float(ang.mean()),
                                 psi_max=float(ang.max()))
                        print({k: (round(x, 5) if isinstance(x, float) else x) for k, x in r.items()}, flush=True)
                        rows.append(r)
                with open(a.out, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
