import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="1"
import numpy as np, bc, faiss
for ds in bc.DATASETS:
  Xc, Q, act = bc.workload(ds)
  P = Q[np.random.default_rng(0).permutation(len(Q))][:3000]
  for k in (16384, 2048):
    f = bc.load(ds, k, "TQ4")
    ids, codes = f._stored_codes()
    C = f._decode_rotated(codes); r = f._all_radiuses[ids].astype(np.float64)
    Qr = f._rotation().apply(P)
    d2 = (Qr ** 2).sum(1)[:, None] + (C ** 2).sum(1)[None] - 2 * Qr @ C.T
    acc = d2 <= ((r + 0.1 + 1e-3) ** 2)[None]
    hit = acc.any(1); n = len(ids)
    # members per ball from the construct set (nearest decoded center)
    idx = faiss.IndexFlatL2(384); idx.add(C); _, own = idx.search(f._rotation().apply(Xc), 1)
    members = np.bincount(own.ravel(), minlength=n)
    def visits(order):
        a = acc[:, order]
        first = np.where(hit, a.argmax(1), n)
        return (first + (hit * 1)).sum() / (len(P) * n)   # fraction of full scans
    rng = np.random.default_rng(1)
    res = dict(storage=visits(np.arange(n)), random=visits(rng.permutation(n)),
               radius_desc=visits(np.argsort(-r, kind="stable")), members_desc=visits(np.argsort(-members, kind="stable")))
    print(ds, k, "accept rate %.3f" % hit.mean(), " ".join(f"{k_}={v:.3f}" for k_, v in res.items()), flush=True)
