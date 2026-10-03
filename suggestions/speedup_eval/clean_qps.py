import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="1"
import time, glob, pickle, numpy as np, pandas as pd, faiss, psutil
psutil.Process().cpu_affinity([0])
from filters.ball_cover.ball_cover_filter import BallCoverFilter, Quantize
S=os.environ["S"]; out=os.path.join(S,"eval","out"); rows=[]
def rate(fn, Q, bs, m):
    t0=time.perf_counter()
    for s in range(0,m,bs): fn(Q[s:s+bs])
    return m/(time.perf_counter()-t0)
for p in sorted(glob.glob(os.path.join(out,"old_static_*_cover.pkl"))):
    _,_,ds,k,qn,_ = os.path.basename(p).split("_")
    c=pickle.load(open(p,"rb")); Q=np.load(os.path.join(S,"eval",f"{ds}.npz"))["Q"]
    f=BallCoverFilter(384,quantize=Quantize(qn),max_spheres=len(c["radii"]),insert_method=None)
    f.all_centers_index=faiss.deserialize_index(c["index"]); f._num_spheres=c["n"]; f._all_radiuses[:]=c["radii"]; f._all_spheres[:c["n"]]=c["spheres"]; f._constructed=True
    old=lambda b: (f._estimator_surface_distance(b) <= 0.1+f._COVERAGE_EPSILON)
    new=lambda b: f.query(b, 0.1)
    r=dict(dataset=ds,k=int(k),quantize=qn,n_balls=c["n"])
    for name,bs,m in (("single",1,300),("batch32",32,2048),("batched",4000,4000)):
        r[f"{name}_old"]=rate(old,Q,bs,m); r[f"{name}_new"]=rate(new,Q,bs,m)
    r["same_decisions_4000"]=bool((np.asarray(f.query(Q[:4000],0.1))==old(Q[:4000])).all())
    rows.append(r); print(r, flush=True)
pd.DataFrame(rows).to_csv(os.path.join(S,"eval","clean_qps.csv"), index=False)
print("DONE")
