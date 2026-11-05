"""Why does the flat index's range_search cost ~19 ms per query in end_to_end.csv and ~2.9 ms in
latency.csv (same 80K-vector index, same call)? Times that one call the way each worker does, so
the two can be compared in one process on one machine.

    PYTHONPATH=src python suggestions/diag_backend_cost.py                      # real corpus (needs datasets/)
    PYTHONPATH=src python suggestions/diag_backend_cost.py --synthetic          # random unit vectors
    PYTHONPATH=src python suggestions/diag_backend_cost.py --procs 8            # 8 copies, pinned to cores 0..7

Prints ms per query for:
  latency-style   per-call timing of 1,000 randomly sampled queries (filter_latency_worker)
  e2e-style       total time of a loop over the first 1,000 queries in file order, best of 3
                  (end_to_end_worker), divided by 1,000
  random order    the e2e loop over the sampled queries
If the first two differ the cause is the measurement pattern; if they agree and --procs 8 raises
them, it is contention; if neither, it is the machine's state during the run (frequency, other
jobs, memory placement)."""
import os
import sys
import time
import multiprocessing as mp

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import faiss

faiss.omp_set_num_threads(1)
TAU = 0.2            # cosine distance 0.02 as the L2 threshold of the paper's runs
N_TIMED = 1000


def _vectors(synthetic, corpus, n_total):
    if synthetic:
        rng = np.random.default_rng(0)
        x = rng.normal(size=(n_total, 384)).astype("float32")
        return x / np.linalg.norm(x, axis=1, keepdims=True)
    import experiments
    wl = next(w for w in experiments.blocklist_workloads(n_total, n_folds=1) if w.name.lower() == corpus.lower())
    sel = wl.selections[0]
    return np.concatenate([np.asarray(wl.rows(sel["reference"])), np.asarray(wl.query_vectors(sel))]).astype("float32")


def _run(core, synthetic, corpus, n_total):
    if core is not None and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {core})
    x = np.ascontiguousarray(_vectors(synthetic, corpus, n_total))
    n_ref = int(0.8 * n_total)
    B, Q = x[:n_ref], np.ascontiguousarray(x[n_ref:])
    index = faiss.IndexFlatL2(384)
    index.add(B)
    index.search(Q, 1)
    index.range_search(Q[:1], 0.04)
    timed = np.sort(np.random.default_rng(0).choice(len(Q), size=min(N_TIMED, len(Q)), replace=False))

    def per_call(queries):
        out = []
        for i in range(len(queries)):
            v = queries[i:i + 1]
            t = time.perf_counter_ns()
            index.range_search(v, TAU * TAU)
            out.append((time.perf_counter_ns() - t) / 1e6)
        return float(np.mean(out))

    def loop_best(queries, repeats=3):
        best = float("inf")
        for _ in range(repeats):
            t = time.perf_counter()
            for i in range(len(queries)):
                index.range_search(queries[i:i + 1], TAU * TAU)
            best = min(best, time.perf_counter() - t)
        return 1000.0 * best / len(queries)

    return {"core": core, "latency-style": per_call(Q[timed]), "e2e-style": loop_best(Q[:N_TIMED]),
            "random order": loop_best(Q[timed])}


def main():
    args = sys.argv[1:]
    synthetic = "--synthetic" in args
    procs = int(args[args.index("--procs") + 1]) if "--procs" in args else 1
    corpus = args[args.index("--corpus") + 1] if "--corpus" in args else "WikiAnswers"
    n_total = int(args[args.index("--n") + 1]) if "--n" in args else 100_000
    cores = range(procs) if procs > 1 else [0 if hasattr(os, "sched_setaffinity") else None]
    with mp.get_context("spawn").Pool(procs) as pool:
        for r in pool.starmap(_run, [(c, synthetic, corpus, n_total) for c in cores]):
            print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
