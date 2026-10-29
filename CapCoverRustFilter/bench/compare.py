"""Old (numba) vs Rust: agreement and qps per filter and batch size.
Usage (repo root): PYTHONPATH=src:CapCoverRustFilter:CapCoverRustFilter/bench python CapCoverRustFilter/bench/compare.py [names] [tag]"""
import json, sys, time
import numpy as np
from fixtures import load_vectors
import fixtures
from filters.cap_cover.cap_cover_rust_filter import CapCoverRustFilter

NAMES = sys.argv[1].split(",") if len(sys.argv) > 1 else ["TQ1_3000", "TQ2_3000", "TQ4_3000", "SPQ16_3000", "SPQ48_3000", "mixed"]
TAG = sys.argv[2] if len(sys.argv) > 2 else "rust"
TAU = 0.1
import os, capcover_rs
if os.environ.get('FS') == '0':
    capcover_rs.set_fastscan(False)     # A/B: scalar table kernels for TQ


def make(name):
    """Same construction as fixtures.build, in the Rust subclass."""
    from filters.cap_cover.cap_cover_filter import CapCoverFilter, Quantize
    X = load_vectors(20000)
    if name == "mixed":
        f = CapCoverRustFilter.auto(384, max_memory=3000 * 48 * 2)
    else:
        enc, k = name.split("_"); f = CapCoverRustFilter(384, quantize=Quantize[enc], max_caps=int(k))
    f.construct(X)
    return f


def qps(fn, Q, bs, min_time=1.0):
    n = (len(Q) // bs) * bs
    fn(Q[:bs])
    t0 = time.perf_counter(); done = 0
    while time.perf_counter() - t0 < min_time:
        for i in range(0, n, bs):
            fn(Q[i:i + bs]); done += bs
            if time.perf_counter() - t0 >= min_time:
                break
    return done / (time.perf_counter() - t0)


if __name__ == "__main__":
    Q = load_vectors(4096, seed=7)
    rows = []
    for name in NAMES:
        f = make(name)
        new = np.array(f.query(Q, TAU)); old = np.array(f.query_original(Q, TAU)); f.freeze()
        r = {"filter": name, "caps": int(f._num_caps), "disagree": int((new != old).sum()), "positive": float(new.mean())}
        Qneg = Q[~new]
        for bs in (1, 32, 1024):
            r[f"new_b{bs}"] = qps(lambda x: f.query(x, TAU), Q, bs)
            r[f"new_neg_b{bs}"] = qps(lambda x: f.query(x, TAU), Qneg, min(bs, len(Qneg)))
        print(json.dumps(r), flush=True); rows.append(r)
    json.dump(rows, open(f"CapCoverRustFilter/bench/{TAG}.json", "w"), indent=1)
