"""Silent-quality probe: Gonzalez covering radius (max over rows of float64 angle to nearest chosen row)
and mean cap angle of lazy / int8 kernels vs the NumPy reference, plus label optimality
(angle to labelled cap vs angle to the true nearest chosen direction). Run: PYTHONPATH=src python quality_radius.py"""
import os
for v in ("OMP", "OPENBLAS", "MKL"):
    os.environ[v + "_NUM_THREADS"] = "1"
import pickle
import numpy as np
from filters.cap_cover import gonzales_construct as gc
from filters.cap_cover.cap_geometry import angle_between, labelled_cap_cover

PKL = "/tmp/claude-0/-home-user/faeeddb3-925b-5717-b1cb-caa0100e8a36/scratchpad/eli5/eli5_10k.pkl"


def unit(a):
    a = np.asarray(a, np.float64)
    return (a / np.linalg.norm(a, axis=1, keepdims=True)).astype(np.float32)


def radius(X, ch):
    X64 = X.astype(np.float64)
    ip = (X64 @ X64[ch].T).max(1)
    return float(np.max(angle_between(X64, X64[np.argmax(X64 @ X64[ch].T, 1)])))


def run(name, X, ks):
    for k in ks:
        res = {}
        for kern in ("numpy", "lazy", "int8"):
            if kern == "numpy":
                ch = gc.gonzalez_chosen_numpy(X, k); lab = None
            elif kern == "lazy":
                ch, lab = gc._rust_lazy_labels(X, k)
            else:
                if gc._rust_gonzalez_labels is None:
                    continue
                ch, lab = gc._rust_gonzalez_labels(X, k)
            ch = np.asarray(ch, np.int64)
            r = radius(X, ch)
            if lab is not None:
                X64 = X.astype(np.float64)
                _, ang, _, _ = labelled_cap_cover(X64, X64[ch], np.asarray(lab, np.int64))
                _, ang_x, _, _ = labelled_cap_cover(X64, X64[ch])
                res[kern] = (r, ang.mean(), ang_x.mean(), ang.max(), ang_x.max())
            else:
                _, ang_x, _, _ = labelled_cap_cover(X.astype(np.float64), X.astype(np.float64)[ch])
                res[kern] = (r, ang_x.mean(), ang_x.mean(), ang_x.max(), ang_x.max())
        base = res["numpy"]
        for kern, v in res.items():
            print(f"{name:10s} k={k:5d} {kern:6s} radius={v[0]:.6f} (d={v[0]-base[0]:+.2e}) mean_theta={v[1]:.6f} "
                  f"(exact-label {v[2]:.6f}, label loss {v[1]-v[2]:+.2e}) max_theta_loss={v[3]-v[4]:+.2e}")


V = pickle.load(open(PKL, "rb")).astype(np.float32)[:8000]
run("eli5", V, (80, 800, 3000))
r = np.random.default_rng(0)
c = unit(r.standard_normal((40, 384)))
tight = unit(c[r.integers(0, 40, 6000)].astype(np.float64) + 0.02 * r.standard_normal((6000, 384)) / np.sqrt(384))
run("tight", tight, (50, 500, 3000))
run("dupheavy", np.vstack([np.repeat(V[:50], 60, 0), V[50:2050]]), (60, 500, 1500))
