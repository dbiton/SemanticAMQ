"""Peak RSS of one construct() at n=80k synthetic (clustered, d=384, FP32, 1% budget).
usage: PYTHONPATH=<src> python peak_rss.py METHOD [N]   (one process per measurement)"""
import os, sys
for v in ("OMP", "OPENBLAS", "MKL", "NUMEXPR"):
    os.environ[v + "_NUM_THREADS"] = "1"
import resource, time
import numpy as np
method = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 80000
from runner import CAP_CONSTRUCT_METHODS
from filters.cap_cover.cap_cover_filter import CapCoverFilter
from filters.cap_cover.cap_quantizers import Quantize
r = np.random.default_rng(0); d = 384
c = r.standard_normal((200, d)); c /= np.linalg.norm(c, axis=1, keepdims=True)
X = np.empty((n, d), np.float32)
for s in range(0, n, 10000):
    m = min(10000, n - s)
    x = c[r.integers(0, 200, m)] + 0.5 * r.standard_normal((m, d)) / np.sqrt(d)
    X[s:s + m] = x / np.linalg.norm(x, axis=1, keepdims=True)
base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
f = CapCoverFilter(d, construct_method=CAP_CONSTRUCT_METHODS[method](), max_memory=int(0.01 * n * d * 4),
                   quantize=Quantize.FP32, insert_method=None)
t = time.perf_counter(); f.construct(X); dt = time.perf_counter() - t
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
print(f"{method} n={n} base_rss={base:.0f}MB peak_rss={peak:.0f}MB delta={peak-base:.0f}MB caps={len(f.all_angles)} "
      f"psi_mean={f.all_angles.mean():.5f} t={dt:.1f}s")
