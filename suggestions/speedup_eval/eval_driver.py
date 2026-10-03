import os, sys, subprocess, itertools, queue, threading
S = os.environ["S"]; ROOT = "C:/Projects/SemanticAMQ"
TREES = {"old": os.path.join(S, "old_tree", "src"), "new": os.path.join(ROOT, "src")}
DATASETS = ["WildChat", "CCNews", "WikiAnswers", "Quora", "ChatGPTParaphrases"]
jobs = []
for ds in DATASETS:
    for k, qz in [(16384, "TQ4"), (16384, "FP32"), (2048, "TQ4"), (2048, "FP32"), (16384, "SQ8"), (16384, "PQ8@1"), (16384, "TQ1")]:
        for tag in ("old", "new"): jobs.append((tag, "static", ds, k, qz))
for ds in ["WildChat", "CCNews"]:
    for qz in ("FP32", "TQ4"):
        for tag in ("old", "new"): jobs.append((tag, "stream", ds, 8192, qz))
# slow ones first
jobs.sort(key=lambda j: (j[1] != "stream", -j[3]))
done = {f for f in os.listdir(os.path.join(S, "eval", "out"))} if os.path.isdir(os.path.join(S, "eval", "out")) else set()
cores = queue.Queue(); [cores.put(c) for c in (0, 2, 4, 6, 8, 10, 12, 14)]
def run(job):
    tag, kind, ds, k, qz = job
    if f"{tag}_{kind}_{ds}_{k}_{qz}.json" in done: return
    c = cores.get()
    try:
        env = dict(os.environ, PYTHONPATH=TREES[tag], S=S)
        r = subprocess.run([sys.executable, os.path.join(S, "eval_job.py"), tag, kind, ds, str(k), qz, str(c)],
                           env=env, cwd=ROOT, capture_output=True, text=True)
        print(("OK " if r.returncode == 0 else "FAIL ") + " ".join(map(str, job)) + ("" if r.returncode == 0 else "\n" + r.stderr[-1500:]), flush=True)
    finally: cores.put(c)
ts = []
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(8) as ex: list(ex.map(run, jobs))
print("ALL DONE", flush=True)
