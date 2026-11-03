"""FPR slack profile for Cap-Cover signal cells (fold 0, ELI5 8k). Decomposes the stored cover's accepted volume.
  PYTHONPATH=src python bench_construct/fpr_profile.py [--out out/fpr_profile.csv]
Variants (all float64, over decoded caps; unit queries, accept iff min_c angle(q,u_c)-theta_c <= beta):
 stored      the shipped angles (grid-rounded)       unrounded  pre-grid tight angles (rounding slack)
 shrunk      coordinate-descent re-certified angles  beta0      beta=0 (share due to tau widening)"""
import os
for v in ("OMP", "OPENBLAS", "MKL", "NUMEXPR"):
    os.environ[f"{v}_NUM_THREADS"] = "1"
import pickle, csv, time, argparse
import numpy as np, psutil, faiss
faiss.omp_set_num_threads(1)
from runner import CAP_CONSTRUCT_METHODS
from filters.cap_cover.cap_cover_filter import CapCoverFilter
from filters.cap_cover.cap_quantizers import Quantize
from filters.cap_cover.cap_geometry import unit_rows, log_cap_measure
TAU = 10 ** -0.5
PKL = "/tmp/claude-0/-home-user/faeeddb3-925b-5717-b1cb-caa0100e8a36/scratchpad/eli5/eli5_10k.pkl"


def angle_matrix(A, B):
    return np.arccos(np.clip(A @ B.T, -1, 1))


def shrink(Am, theta, rounds=3):
    """Coordinate descent: theta_c -> max angle of members no other cap holds. Never uncovers a member."""
    theta = theta.copy()
    held = Am <= theta[None, :]
    cnt = held.sum(1)
    for _ in range(rounds):
        changed = 0
        for c in np.argsort(theta):
            mine = held[:, c]
            only = mine & (cnt == 1)
            new = Am[only, c].max() if only.any() else 0.0
            if new < theta[c]:
                changed += 1
                newheld = Am[:, c] <= new
                cnt += newheld.astype(int) - mine.astype(int)
                held[:, c] = newheld
                theta[c] = new
        if not changed:
            break
    return theta


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="bench_construct/out/fpr_profile.csv")
    ap.add_argument("--core", type=int, default=1); a = ap.parse_args()
    psutil.Process().cpu_affinity([a.core])
    V = pickle.load(open(PKL, "rb")).astype(np.float32)
    C, Q = V[:8000], V[8000:]
    d = C.shape[1]
    idx = faiss.IndexFlatL2(d); idx.add(C)
    actual = np.sqrt(np.maximum(idx.search(Q, 1)[0][:, 0], 0)) <= TAU
    rows = []
    for method in ("gonzalez", "minimax"):
        for qn in ("TQ2", "TQ4"):
            for b in (0.03, 0.10):
                f = CapCoverFilter(d, construct_method=CAP_CONSTRUCT_METHODS[method](), max_memory=int(b * 8000 * d * 4),
                                   quantize=Quantize(qn), insert_method=None)
                cap = {}
                orig = f._set_angles
                f._set_angles = lambda ids, ang, orig=orig: (cap.__setitem__("raw", np.array(ang)), orig(ids, ang))[1]
                f.construct(C)
                pred = np.asarray(f.query(Q, TAU), bool)
                ids = f.all_caps
                U = f.decode_caps(ids)
                th = f._angle_value(f._angle_codes[ids])
                raw = cap["raw"]
                Xn = unit_rows(f._prepare_members(C))
                Qn = unit_rows(f.to_working(Q))
                AQ = angle_matrix(Qn, U)
                beta = f._beta(np.ones(1), TAU)[0][0]

                def fpr(t, bt=beta):
                    p = ((AQ - t[None, :]) <= bt).any(1)
                    return float((p & ~actual).sum() / (~actual).sum())
                AM = angle_matrix(Xn, U)
                t0 = time.perf_counter()
                sh = shrink(AM, th)
                saved = f._angle_lo, f._angle_hi
                f._angle_lo, f._angle_hi = 0.0, float(np.pi)
                if f._narrow_angles():
                    f._fit_angle_grid(sh)
                sh_r = f._angle_value(f._angle_code(sh))
                f._angle_lo, f._angle_hi = saved
                assert ((AM <= sh_r[None, :] + 1e-6).any(1)).all(), "coverage"; assert ((AM <= th[None, :] + 1e-6).any(1)).all(), "stored coverage"
                vol = lambda t: float(np.logaddexp.reduce(log_cap_measure(np.minimum(t + beta, np.pi), d)))
                r = dict(method=method, quantize=qn, budget=b, n_caps=len(ids), fpr_rust=float((pred & ~actual).sum() / (~actual).sum()),
                         fpr_stored=fpr(th), fpr_unrounded=fpr(raw), fpr_shrunk=fpr(sh_r), fpr_beta0=fpr(th, 0.0),
                         theta_mean=th.mean(), round_slack=float((th - raw).mean()), shrink_gain=float((th - sh_r).mean()),
                         beta=beta, bytes=f.get_bytesize(), max_memory=f.max_memory, shrink_s=time.perf_counter() - t0,
                         lnvol_stored=vol(th), lnvol_shrunk=vol(sh_r))
                print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
                rows.append(r)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
