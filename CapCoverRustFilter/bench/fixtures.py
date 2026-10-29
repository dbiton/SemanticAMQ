"""Builds (construction takes seconds; faiss quantizers do not pickle) the filters the benchmarks and differential
tests run on. Construction uses the ORIGINAL CapCoverFilter. Run from the repo
root with PYTHONPATH=src."""
import os, pickle, time, sys
import numpy as np
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(os.path.dirname(HERE), "fixtures")
DS = "./datasets/embeds_wildchat.pkl"


def load_vectors(n, seed=0, path=DS):
    with h5py.File(path, "r") as f:
        ds = f["normalized_embeds"]
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(ds.shape[0], n, replace=False))
        return np.asarray(ds[idx], dtype=np.float32)


def build(name, n_construct=20000, **kw):
    """name -> filter. Single-encoding filters: 'TQ1'... with max_caps=K;
    'mixed' = auto() at a byte budget."""
    from filters.cap_cover.cap_cover_filter import CapCoverFilter, Quantize
    X = load_vectors(n_construct)
    t = time.time()
    if name == "mixed":
        filt = CapCoverFilter.auto(384, max_memory=kw.get("max_memory", 3000 * 48 * 2))
    else:
        enc, k = name.split("_")
        filt = CapCoverFilter(384, quantize=Quantize[enc], max_caps=int(k))
    filt.construct(X)
    print(f"built {name}: {filt._num_caps} caps, {filt.get_bytesize()} B, {time.time()-t:.0f}s", file=sys.stderr)
    return filt
