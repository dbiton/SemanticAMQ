"""Thread scaling of the Rust query (b=1024 and 32, positive-mix and rejected-only streams)."""
import json, sys
import numpy as np
from fixtures import load_vectors
from compare import make, qps, TAU
names = sys.argv[1].split(",")
Q = load_vectors(4096, seed=7)
rows = []
for name in names:
    f = make(name); f.query(Q[:2], TAU)
    neg = Q[~np.array(f.query(Q, TAU))]
    for t in (1, 2, 4, 5, 8, 10):
        f._rust.set_threads(t)
        r = {"filter": name, "threads": t}
        for bs in (32, 1024):
            r[f"qps_b{bs}"] = qps(lambda x: f.query(x, TAU), Q, bs, 0.7)
            r[f"qps_neg_b{bs}"] = qps(lambda x: f.query(x, TAU), neg, bs, 0.7)
        print(json.dumps(r), flush=True); rows.append(r)
json.dump(rows, open("CapCoverRustFilter/bench/threads.json", "w"), indent=1)
