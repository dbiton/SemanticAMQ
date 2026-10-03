import os, json, glob, pickle, numpy as np, pandas as pd, faiss
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ[v]="1"
from filters.ball_cover.ball_cover_filter import BallCoverFilter, Quantize
S = os.environ["S"]; out = os.path.join(S, "eval", "out")
rows = [json.load(open(p)) for p in glob.glob(os.path.join(out, "*.json"))]
df = pd.DataFrame(rows)
key = ["kind", "dataset", "k", "quantize"]
old = df[df.tag == "old"].set_index(key); new = df[df.tag == "new"].set_index(key)
both = old.join(new, lsuffix="_old", rsuffix="_new", how="inner")
# cross-check: new query code on the cover the OLD code built
cross = []
for idx in both.index:
    kind, ds, k, qn = idx
    name = f"old_{kind}_{ds}_{k}_{qn}"
    c = pickle.load(open(os.path.join(out, name + "_cover.pkl"), "rb"))
    d = np.load(os.path.join(S, "eval", f"{ds}.npz")); Q = d["Q"]
    f = BallCoverFilter(384, quantize=Quantize(qn), max_spheres=len(c["radii"]), insert_method=None)
    f.all_centers_index = faiss.deserialize_index(c["index"]); f._num_spheres = c["n"]
    f._all_radiuses[:] = c["radii"]; f._all_spheres[:c["n"]] = c["spheres"]; f._constructed = True
    old_pred = np.load(os.path.join(out, name + "_pred.npy")); old_surf = np.load(os.path.join(out, name + "_surf.npy"))
    new_pred = np.asarray(f.query(Q, 0.1)); new_surf = f.surface_distance(Q)
    new_built = np.load(os.path.join(out, f"new_{kind}_{ds}_{k}_{qn}_pred.npy"))
    cross.append(dict(zip(key, idx), same_cover_flips=int((old_pred != new_pred).sum()),
                      same_cover_max_surface_diff=float(np.abs(old_surf - new_surf).max()),
                      rebuilt_cover_flips=int((old_pred != new_built).sum())))
cross = pd.DataFrame(cross).set_index(key)
both = both.join(cross)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
for col in ("qps_batched", "qps_batch32", "qps_single", "qps_surface"):
    both[col + "_x"] = both[col + "_new"] / both[col + "_old"]
both["construct_x"] = both["construct_s_old"] / both["construct_s_new"]
st = both.xs("static", level="kind")
print("=== static covers: throughput (q/s, single pinned P-core each, 8 jobs in parallel) ===")
print(st[["n_balls_new", "qps_batched_old", "qps_batched_new", "qps_batched_x", "qps_batch32_old", "qps_batch32_new", "qps_batch32_x",
          "qps_single_old", "qps_single_new", "qps_single_x", "qps_surface_x", "construct_s_old", "construct_s_new"]].round(1).to_string())
print("\n=== correctness ===")
print(both[["fpr_old", "fpr_new", "fnr_old", "fnr_new", "same_cover_flips", "same_cover_max_surface_diff", "rebuilt_cover_flips",
            "surface_vs_query_mismatch_new", "bytes_old", "bytes_new"]].to_string())
sm = both.xs("stream", level="kind")
print("\n=== streaming (cap_merge, k=8192, construct 20K then insert 60K in batches of 1000) ===")
print(sm[["insert_vps_old", "insert_vps_new", "fnr_construct_tau0_old", "fnr_construct_tau0_new", "qps_batched_old", "qps_batched_new"]].round(4).to_string())
print("\n=== geometric-mean speedups by (k, quantize), static ===")
g = st.groupby(level=["k", "quantize"])[["qps_batched_x", "qps_batch32_x", "qps_single_x", "qps_surface_x", "construct_x"]].agg(lambda s: float(np.exp(np.log(s).mean())))
print(g.round(2).to_string())
both.to_csv(os.path.join(S, "eval", "speedup_eval.csv"))
