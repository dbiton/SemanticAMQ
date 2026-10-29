"""Memory check: charged bytes vs what the Rust filter owns, RSS after freeze, peak RSS during a big batch.
Usage (repo root): PYTHONPATH=src:CapCoverRustFilter:CapCoverRustFilter/bench python CapCoverRustFilter/bench/memory.py"""
import gc, json, os, threading, time
import numpy as np, psutil
from fixtures import load_vectors
from filters.cap_cover.cap_cover_rust_filter import CapCoverRustFilter
from filters.cap_cover.cap_cover_filter import Quantize

proc = psutil.Process()
rss = lambda: proc.memory_info().rss / 1e6
out = {}
X = load_vectors(60000)
for name, mk in (("TQ4 x 12000 caps", lambda: CapCoverRustFilter(384, quantize=Quantize.TQ4, max_caps=12000)),
                 ("mixed auto 2.3 MB", lambda: CapCoverRustFilter.auto(384, max_memory=2_300_000))):
    f = mk(); f.construct(X)
    charged = f.get_bytesize(); caps = f._num_caps
    gc.collect(); r0 = rss()
    f.query(X[:8], 0.1)                     # freeze
    gc.collect(); r1 = rss()
    code_b, ang_b, const_b = f._rust.owned_bytes()
    Q = load_vectors(60000, seed=3)
    peak = [r1]; stop = threading.Event()
    def sample():
        while not stop.is_set():
            peak.append(rss()); time.sleep(0.002)
    th = threading.Thread(target=sample); th.start()
    f.query(Q, 0.1)
    stop.set(); th.join()
    res = dict(caps=caps, charged_bytes=charged, rust_code_bytes=code_b, rust_angle_bytes=ang_b, rust_const_bytes=const_b,
               py_codes_after_freeze=int(f._codes.nbytes), rss_before_freeze_MB=r0, rss_after_freeze_MB=r1,
               rss_delta_freeze_MB=r1 - r0, peak_rss_growth_big_batch_MB=max(peak) - r1, batch=len(Q))
    print(name, json.dumps(res, indent=1)); out[name] = res
json.dump(out, open("CapCoverRustFilter/bench/memory.json", "w"), indent=1)
