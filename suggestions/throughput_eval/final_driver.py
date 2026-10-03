import os, sys, subprocess
S = os.path.dirname(os.path.abspath(__file__)); ROOT = "C:/Projects/SemanticAMQ"
TREES = {"old": os.path.join(S, "old_tree", "src"), "new": os.path.join(ROOT, "src")}
core = sys.argv[1]
for ds in ["WildChat", "CCNews", "WikiAnswers", "Quora", "ChatGPTParaphrases"]:
    for k, qn in ((16384, "TQ4"), (2048, "TQ4"), (16384, "FP32"), (2048, "FP32")):
        for rep in (0, 1):
            for tag in ("old", "new"):
                out = os.path.join(S, "final", f"{tag}_{ds}_{k}_{qn}_rep{rep}.json")
                if os.path.exists(out): continue
                env = dict(os.environ, PYTHONPATH=TREES[tag] + ";" + S)
                r = subprocess.run([sys.executable, os.path.join(S, "final_query_job.py"), tag, ds, str(k), qn, core, str(rep)],
                                   env=env, cwd=ROOT, capture_output=True, text=True)
                print(("OK " if r.returncode == 0 else "FAIL ") + f"{tag} {ds} {k} {qn} rep{rep}" + ("" if r.returncode == 0 else "\n" + r.stderr[-1500:]), flush=True)
print("ALL DONE", flush=True)
