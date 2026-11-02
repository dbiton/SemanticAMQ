"""Probe raw kernel entry points with hostile arguments; prints outcome per call (exception type or ok)."""
import os
for v in ("OMP", "OPENBLAS", "MKL"):
    os.environ[v + "_NUM_THREADS"] = "1"
import numpy as np
import capcover_rs as R

f32 = lambda *s: np.random.default_rng(0).standard_normal(s).astype(np.float32)


def t(name, fn):
    try:
        r = fn()
        print(f"{name}: ok -> {type(r).__name__} {getattr(r, 'shape', '')}")
    except BaseException as e:
        print(f"{name}: {type(e).__name__}: {str(e)[:90]}")


X = f32(50, 16); X /= np.linalg.norm(X, axis=1, keepdims=True)
for fn in ("gonzalez_chosen", "gonzalez_chosen_pruned", "gonzalez_chosen_labels", "gonzalez_chosen_pruned_labels"):
    g = getattr(R, fn)
    t(f"{fn} k=0", lambda: g(X, 0))
    t(f"{fn} k=n", lambda: g(X, 50))
    t(f"{fn} k>n", lambda: g(X, 500))
    t(f"{fn} n=0", lambda: g(np.zeros((0, 16), np.float32), 3))
    t(f"{fn} d=0", lambda: g(np.zeros((5, 0), np.float32), 3))
    t(f"{fn} n=1", lambda: g(X[:1], 1))
    t(f"{fn} float64", lambda: g(X.astype(np.float64), 3))
    t(f"{fn} noncontig", lambda: g(np.asfortranarray(X), 3))
    t(f"{fn} nan", lambda: g(np.full((10, 16), np.nan, np.float32), 3))
    t(f"{fn} inf", lambda: g(np.full((10, 16), np.inf, np.float32), 3))
    t(f"{fn} zeros", lambda: g(np.zeros((10, 16), np.float32), 3))
    t(f"{fn} d=1", lambda: g(np.ones((10, 1), np.float32), 3))
U = np.ascontiguousarray(X[:5]).astype(np.float64)
X64 = X.astype(np.float64)
t("row_angles label=-1", lambda: R.row_angles(X64, U, np.full(50, -1, np.int64)))
t("row_angles label=big", lambda: R.row_angles(X64, U, np.full(50, 99, np.int64)))
t("row_angles len mismatch", lambda: R.row_angles(X64, U, np.zeros(10, np.int64)))
t("row_angles dim mismatch", lambda: R.row_angles(X64, U[:, :8].copy(), np.zeros(50, np.int64)))
t("row_angles empty", lambda: R.row_angles(X64[:0], U, np.zeros(0, np.int64)))
t("row_angles nan", lambda: R.row_angles(np.full((3, 16), np.nan), U, np.zeros(3, np.int64)))
t("pair_angles idx big", lambda: R.pair_angles(X64, U, np.full((50, 2), 99, np.int64)))
t("pair_angles idx mismatch rows", lambda: R.pair_angles(X64, U, np.zeros((10, 2), np.int64)))
t("mm_assign k=0", lambda: R.mm_assign(X, X[:0]))
t("mm_assign dim mismatch", lambda: R.mm_assign(X, X[:3, :8].copy()))
t("mm_assign n=0", lambda: R.mm_assign(X[:0], X[:3]))
t("mm_assign nan", lambda: R.mm_assign(np.full((4, 16), np.nan, np.float32), X[:3]))
