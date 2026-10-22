"""ns per cap per query of CapCoverFilter.query() (the Rust scan), by encoding.

Run on a quiet machine, pinned to one core, from the repo root:
    PYTHONPATH=src python CapCoverRustFilter/bench/ns_per_cap.py [n_caps]

Queries are random unit vectors, so (almost) every one is rejected and scans
every cap: the worst case, and the cost a negative query pays. One query at a
time, as the end-to-end evaluation issues them. Prints the CPU features the
extension can use, so a build with and without
RUSTFLAGS="-C target-cpu=native" can be compared on the same machine."""
import os
import sys
import time

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"

import numpy as np

try:
    import psutil
    psutil.Process().cpu_affinity([0])
    print("pinned to cpu 0")
except Exception as e:          # macOS has no cpu_affinity
    print(f"not pinned ({type(e).__name__})")

import capcover_rs
from filters.cap_cover.cap_cover_filter import CapCoverFilter, Quantize

DIM = 384
N_CAPS = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
N_QUERIES = 300
rng = np.random.default_rng(0)


def unit(x):
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


centers = unit(rng.standard_normal((N_CAPS, DIM)))
X = unit(np.repeat(centers, 3, axis=0) + 0.15 * rng.standard_normal((3 * N_CAPS, DIM)) / np.sqrt(DIM))
Q = unit(rng.standard_normal((N_QUERIES, DIM)))
print(f"{N_CAPS} caps requested from {len(X)} vectors, d={DIM}; {N_QUERIES} single queries each")
print(f"{'encoding':8s} {'caps':>7s} {'ms/query':>9s} {'ns/cap':>8s}")
for q in (Quantize.TQ1, Quantize.TQ2, Quantize.TQ4, Quantize.TQ8, Quantize.SPQ16, Quantize.SPQ24, Quantize.SPQ48,
          Quantize.SPQ64):
    f = CapCoverFilter(DIM, quantize=q, max_caps=N_CAPS)
    f.construct(X)
    f.query(Q[:1], 0.1)                                  # freeze: hands the caps to Rust
    t = time.perf_counter()
    for i in range(N_QUERIES):
        f.query(Q[i:i + 1], 0.1)
    s = (time.perf_counter() - t) / N_QUERIES
    n = f._num_caps
    print(f"{q.value:8s} {n:7d} {1000 * s:9.3f} {1e9 * s / n:8.1f}", flush=True)
