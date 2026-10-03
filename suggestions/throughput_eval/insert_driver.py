import os, sys, subprocess
S = os.path.dirname(os.path.abspath(__file__)); ROOT = "C:/Projects/SemanticAMQ"
TREES = {"old": os.path.join(S, "old_tree", "src"), "new": os.path.join(ROOT, "src")}
core, n = sys.argv[1], sys.argv[2]
for k in (2048, 16384):
    for ds in ["WildChat", "CCNews", "WikiAnswers", "Quora", "ChatGPTParaphrases"]:
        for qn in ("TQ4", "FP32"):
            for tag in ("old", "new"):
                out = os.path.join(S, "final", f"ins_{tag}_{ds}_{k}_{qn}_rep0.json")
                if os.path.exists(out): continue
                env = dict(os.environ, PYTHONPATH=TREES[tag] + ";" + S)
                r = subprocess.run([sys.executable, os.path.join(S, "insert_job.py"), tag, ds, str(k), qn, core, "0", n],
                                   env=env, cwd=ROOT, capture_output=True, text=True)
                print(("OK " if r.returncode == 0 else "FAIL ") + f"{tag} {ds} {k} {qn}" + ("" if r.returncode == 0 else "\n" + r.stderr[-1500:]), flush=True)
print("ALL DONE", flush=True)
