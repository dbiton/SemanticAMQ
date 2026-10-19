"""Profile of the ORIGINAL numba query path: qps by batch size, and the share
of time spent outside the compiled kernel (rotation, beta, copies, tolist).
Usage (repo root): PYTHONPATH=src:CapCoverRustFilter/bench python CapCoverRustFilter/bench/baseline.py"""
import json, sys, time
import numpy as np
from fixtures import build, load_vectors

NAMES = sys.argv[1].split(",") if len(sys.argv) > 1 else ["TQ1_3000", "TQ2_3000", "TQ4_3000", "SPQ16_3000", "SPQ48_3000", "mixed"]
TAU = 0.1


def qps(f, Q, bs, tau, min_time=1.0):
    n = (len(Q) // bs) * bs
    f.query(Q[:bs], tau)
    t0 = time.perf_counter(); done = 0
    while time.perf_counter() - t0 < min_time:
        for i in range(0, n, bs):
            f.query(Q[i:i + bs], tau); done += bs
            if time.perf_counter() - t0 >= min_time:
                break
    return done / (time.perf_counter() - t0)


def overhead(f, Q, tau, reps=20):
    """seconds per call spent in the Python glue of _scan for batch Q."""
    ids = f.all_caps; lv = f._cap_level[ids]
    t0 = time.perf_counter()
    for _ in range(reps):
        Qw = f.to_working(Q); nq = np.sqrt(np.einsum("ij,ij->i", Qw, Qw))
        beta, feas = f._beta(nq, tau); Qn = Qw / np.maximum(nq, 1e-300)[:, None]; Qn32 = Qn.astype(np.float32)
    pre = (time.perf_counter() - t0) / reps
    t0 = time.perf_counter()
    for _ in range(reps):
        for level in np.unique(lv):
            sel = ids[lv == level]
            np.ascontiguousarray(f._codes[sel][:, :f.code_size(int(level))]); f._angle_value(f._angle_codes[sel])
    cp = (time.perf_counter() - t0) / reps
    out = np.zeros(len(Q), bool)
    t0 = time.perf_counter()
    for _ in range(reps):
        out.tolist()
    return pre, cp, (time.perf_counter() - t0) / reps


if __name__ == "__main__":
    Q = load_vectors(4096, seed=7)
    rows = []
    for name in NAMES:
        f = build(name)
        acc = np.array(f.query(Q, TAU)); pos = float(acc.mean()); Qneg = Q[~acc]
        r = {"filter": name, "caps": int(f._num_caps), "positive_rate": pos}
        for bs in (1, 32, 1024):
            r[f"qps_b{bs}"] = qps(f, Q, bs, TAU)
            pre, cp, tl = overhead(f, Q[:bs], TAU)
            total = bs / r[f"qps_b{bs}"]
            r[f"py_overhead_frac_b{bs}"] = (pre + cp + tl) / total
        for bs in (1, 32, 1024):
            r[f"qps_neg_b{bs}"] = qps(f, Qneg, min(bs, len(Qneg)), TAU)    # rejected queries only: full scans
        print(json.dumps(r), flush=True)
        rows.append(r)
    json.dump(rows, open("CapCoverRustFilter/bench/baseline_numba.json", "w"), indent=1)
