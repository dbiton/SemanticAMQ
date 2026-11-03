"""Paired FPR/time test of CapCoverFilter(spend_leftover=True) vs the reference path (flag off).
  PYTHONPATH=src python bench_construct/fpr_spend.py --folds 0 --out bench_construct/out/fpr_spend_f0.csv
Ladders: TQ (TQ2,TQ3,TQ4: the plan's signal encodings) and SPQ (the default AUTO_CANDIDATES). Pinned to one core.
Fold 0 = original order; fold s>0 = permutation with seed s (as variants.py). Reports FPR/FNR/bytes/construct time."""
import os
for v in ("OMP", "OPENBLAS", "MKL", "NUMEXPR"):
    os.environ[f"{v}_NUM_THREADS"] = "1"
import argparse, pickle, time, csv
import numpy as np, psutil, faiss
faiss.omp_set_num_threads(1)
from runner import CAP_CONSTRUCT_METHODS
from filters.cap_cover.cap_cover_filter import CapCoverFilter
from filters.cap_cover.cap_quantizers import Quantize
TAU = 10 ** -0.5
PKL = "/tmp/claude-0/-home-user/faeeddb3-925b-5717-b1cb-caa0100e8a36/scratchpad/eli5/eli5_10k.pkl"
LADDERS = {"TQ": (Quantize.TQ2, Quantize.TQ3, Quantize.TQ4), "SPQ": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0"); ap.add_argument("--out", required=True)
    ap.add_argument("--budgets", default="0.03,0.10"); ap.add_argument("--methods", default="gonzalez,minimax")
    ap.add_argument("--ladders", default="TQ,SPQ"); ap.add_argument("--core", type=int, default=1)
    a = ap.parse_args()
    psutil.Process().cpu_affinity([a.core])
    V0 = pickle.load(open(PKL, "rb")).astype(np.float32)
    rows = []
    for fold in map(int, a.folds.split(",")):
        V = V0 if fold == 0 else V0[np.random.default_rng(fold).permutation(len(V0))]
        C, Q = V[:8000], V[8000:]
        d = C.shape[1]
        idx = faiss.IndexFlatL2(d); idx.add(C)
        actual = np.sqrt(np.maximum(idx.search(Q, 1)[0][:, 0], 0)) <= TAU
        for method in a.methods.split(","):
            for ladder in a.ladders.split(","):
                for b in map(float, a.budgets.split(",")):
                    mm = int(b * 8000 * d * 4)
                    for flag in (False, True):
                        kw = dict(construct_method=CAP_CONSTRUCT_METHODS[method](), insert_method=None, spend_leftover=flag)
                        if LADDERS[ladder]:
                            kw["candidates"] = LADDERS[ladder]
                        f = CapCoverFilter.auto(d, mm, **kw)
                        t0 = time.perf_counter()
                        f.construct(C)
                        cs = time.perf_counter() - t0
                        p = np.asarray(f.query(Q, TAU), bool)
                        lv = np.unique(f._cap_level[f.all_caps])
                        r = dict(fold=fold, method=method, ladder=ladder, budget=b, spend=int(flag), construct_s=cs,
                                 fpr=float((p & ~actual).sum() / (~actual).sum()), fn=int((~p & actual).sum()),
                                 bytes=f.get_bytesize(), max_memory=mm, n_caps=len(f.all_angles),
                                 levels="+".join(f.levels[int(l)].value for l in lv), ok=int(f.get_bytesize() <= mm))
                        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
                        rows.append(r)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
