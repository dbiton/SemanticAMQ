"""CapCoverFilter against Ball-Cover on the blocklist workload, at equal bytes.

    python src/compare_cap_cover.py prepare [--num-vectors 10000]
    python src/compare_cap_cover.py run TAG CONFIG[,CONFIG...] [--kf ...] [--folds ...]
                                    [--corpora ...] [--cpus 16-27]
    python src/compare_cap_cover.py tau TAG CONFIG[,CONFIG...] [same options as run]
    python src/compare_cap_cover.py stream TAG CONFIG[,CONFIG...] [same options as run]
    python src/compare_cap_cover.py summary TAG[,TAG...] [--base CONFIG]

`prepare` cuts each corpus's blocklist folds exactly as main.generate_budget
does (main._fold_selections on the cached cluster labels, ground truth by
distance at tau = 0.1) and stores each fold's reference set, query stream and
labels under results/cap_cover/folds_<n>/. It never calls a main.generate_*
function, so no evaluation CSV is touched. `run` builds every CONFIG on every
(corpus, fold, k_fraction) in worker processes pinned one per CPU, and writes
one row per filter to results/cap_cover/<TAG>.csv. `tau` builds each filter
once and queries it at several thresholds (tau_job), one row per tau;
`stream` bootstraps on half of B and inserts the rest (stream_job), one row
per checkpoint.

Every filter gets the same byte budget: what Ball-Cover at TQ4 (192 B center
+ 2 B radius, 128 B codebook) needs for round(k_fraction * |B|) balls -- the
budget generate_budget's k_fraction stands for. A config whose filter exceeds
it, or returns a false negative, fails its job.

Run from the repo root. Workers pin themselves to the CPUs given (default:
16-27, the i7-14700's E-cores, leaving the P-cores free)."""
import os
import sys
import time
import types
import argparse

for _env_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env_var, "1")

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

OUT_DIR = "results/cap_cover"
TAU = 0.1
CORPORA = ("wildchat", "ccnews", "wikianswers", "quora", "chatgpt_paraphrases")
K_FRACTIONS = (0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2)
DIM = 384


def ball_budget(k: int) -> int:
    """Bytes of Ball-Cover TQ4 holding k balls (see the module docstring)."""
    return k * (192 + 2) + 124 + 4


def _configs():
    from filters.ball_cover.ball_cover_filter import BallCoverFilter, Quantize as BQ
    from filters.ball_cover.sphere_merge_construct import SphereMergeConstruct as BallSphereMerge
    from filters.ball_cover.minimax_construct import MinimaxConstruct as BallMinimax
    from filters.cap_cover.cap_cover_filter import CapCoverFilter, Quantize as CQ
    from filters.cap_cover.sphere_merge_construct import SphereMergeConstruct as CapSphereMerge

    def ball(construct, q):
        return lambda budget: BallCoverFilter(DIM, construct_method=construct(), max_memory=budget,
                                              insert_method=None, quantize=q)

    def cap_auto(**merge):
        return lambda budget: CapCoverFilter.auto(DIM, budget, construct_method=CapSphereMerge(**merge))

    configs = {
        # The paper's Ball-Cover (runner._ball_cover).
        "ball_paper": ball(lambda: BallMinimax(sphere=True), BQ.TQ4),
        # auto() selects with the held-out veto by default since 2026-10-03;
        # cap_auto rows written before that (final_kf*, v10k_*, final10k_*,
        # c80_*) used the reach proxy alone, which cap_auto_proxy keeps.
        "cap_auto": lambda budget: CapCoverFilter.auto(DIM, budget),
        "cap_auto_proxy": lambda budget: CapCoverFilter.auto(DIM, budget, selection="proxy"),
        # Reach-proxy selection over an unweighted (mass_weight 0) merge.
        "cap_auto0_proxy": lambda budget: CapCoverFilter.auto(DIM, budget, selection="proxy",
                                                              construct_method=CapSphereMerge(mass_weight=0.0)),
        # Merge cost in Ball-Cover's units (sin of the cap angle).
        "cap_auto_sine": cap_auto(cost_space="sine"),
        # Encoding chosen for another tau than the one queried (tau study).
        "cap_auto_sel05": lambda budget: CapCoverFilter.auto(DIM, budget, selection_tau=0.05),
        "cap_auto_sel20": lambda budget: CapCoverFilter.auto(DIM, budget, selection_tau=0.2),
        # Encoding chosen on held-out members instead of the reach proxy.
        "cap_auto_hold25": lambda budget: CapCoverFilter.auto(DIM, budget, selection="holdout"),
        "cap_auto_hold50": lambda budget: CapCoverFilter.auto(DIM, budget, selection="holdout",
                                                              holdout_fraction=0.5),
        "cap_auto_hold25z0": lambda budget: CapCoverFilter.auto(DIM, budget, selection="holdout",
                                                                holdout_z=0.0),
        "cap_auto_hold25z5": lambda budget: CapCoverFilter.auto(DIM, budget, selection="holdout",
                                                                holdout_z=5.0),
        "cap_auto_hold25z8": lambda budget: CapCoverFilter.auto(DIM, budget, selection="holdout",
                                                                holdout_z=8.0),
    }
    # SphereMergeConstruct's large-n paths forced at |B| = 8000, with the
    # defaults they were measured under (min_batch 0, 12 probes; v10k_*,
    # split_*, knn20*_ CSVs): cell-probed neighbour lists (knn_exact_max),
    # member searches restricted to them (exact_pairs), and both.
    old = dict(min_batch=0.0, knn_probes=12)
    for name, kw in (("local", dict(knn_exact_max=0, exact_pairs=0)), ("knn", dict(knn_exact_max=0)),
                     ("cand", dict(exact_pairs=0)), ("mb1", dict(min_batch=0.01)),
                     ("local_mb1", dict(knn_exact_max=0, exact_pairs=0, min_batch=0.01)),
                     # At |B| = 8000 the share of the merge that 80K's
                     # knn_exact_max = 16384 runs approximately: above 20%.
                     ("knn20", dict(knn_exact_max=1600, exact_pairs=0)),
                     ("knn20p24", dict(knn_exact_max=1600, exact_pairs=0, knn_probes=24))):
        configs[f"cap_auto_{name}"] = cap_auto(**{**old, **kw})
    for q in ("TQ4", "TQ3", "TQ2", "TQ1"):
        configs[f"ball_sm_{q}"] = ball(BallSphereMerge, BQ[q])
    # Streaming (stream_job): bootstrap with construct(), then insert().
    from filters.ball_cover.cap_merge_insert import CapMergeInsert as BallCapMerge
    from filters.cap_cover.cap_merge_insert import CapMergeInsert
    for q in ("FP32", "FP16", "TQ4"):
        configs[f"sball_cm_{q}"] = (lambda q: lambda budget: BallCoverFilter(
            DIM, construct_method=BallSphereMerge(), max_memory=budget, insert_method=BallCapMerge(),
            quantize=BQ[q]))(q)
    configs["scap_auto"] = lambda budget: CapCoverFilter.auto(DIM, budget,
                                                              insert_method=CapMergeInsert(fit_codes=False))
    for q in ("TQ2", "TQ3", "TQ4", "TQ8"):
        configs[f"scap_{q}"] = (lambda q: lambda budget: CapCoverFilter(
            DIM, quantize=CQ[q], max_memory=budget, insert_method=CapMergeInsert(fit_codes=False)))(q)
    # Merged caps' codes fitted to their two parts (CapMergeInsert.fit_codes,
    # the default since measured).
    for q in ("TQ2", "TQ3", "TQ4"):
        configs[f"scap_fit_{q}"] = (lambda q: lambda budget: CapCoverFilter(
            DIM, quantize=CQ[q], max_memory=budget, insert_method=CapMergeInsert(fit_codes=True)))(q)
    configs["scap_fit_auto"] = lambda budget: CapCoverFilter.auto(DIM, budget,
                                                                  insert_method=CapMergeInsert(fit_codes=True))
    # main.generate_streaming's budget: the unquantized (FP32) static filter
    # at k balls, 8x this module's.
    fp32 = lambda budget: (budget - 128) // 194 * (1536 + 2) + 128
    for name in ("sball_cm_FP32", "sball_cm_FP16", "sball_cm_TQ4", "scap_fit_auto", "scap_fit_TQ4"):
        configs[f"{name}_b32"] = (lambda make: lambda budget: make(fp32(budget)))(configs[name])
    for q in ("SPQ6", "SPQ9", "SPQ12", "SPQ16", "SPQ24", "SPQ36", "SPQ48", "SPQ64", "TQ2", "TQ3", "TQ4", "TQ8"):
        configs[f"cap_{q}"] = (lambda q: lambda budget: CapCoverFilter(DIM, quantize=CQ[q], max_memory=budget))(q)
    # Query-agnostic covers (no member-count weights): Gonzalez + minimax
    # Lloyd rounds, and sphere-merge with mass_weight 0.
    from filters.cap_cover.minimax_construct import MinimaxConstruct as CapMinimax
    for q in ("TQ2", "TQ4", "TQ8"):
        configs[f"cap_mm_{q}"] = (lambda q: lambda budget: CapCoverFilter(
            DIM, quantize=CQ[q], max_memory=budget, construct_method=CapMinimax()))(q)
    for q in ("SPQ12", "SPQ24", "SPQ48", "SPQ64", "TQ2", "TQ3", "TQ4", "TQ8"):
        configs[f"cap_sm0_{q}"] = (lambda q: lambda budget: CapCoverFilter(
            DIM, quantize=CQ[q], max_memory=budget, construct_method=CapSphereMerge(mass_weight=0.0)))(q)
    return configs


# ------------------------------------------------------------------ prepare
def prepare(num_vectors: int, corpora=None, cpus=None) -> None:
    """Folds for `corpora` (plot keys; default all). Ground truth is one
    exact faiss search per fold, threaded over `cpus` (the process is pinned
    to them): at |B| = 80K that is ~1.2 TFLOP a fold."""
    # main's configure_logging truncates the shared log.txt: stub it.
    stub = types.ModuleType("logging_config")
    stub.configure_logging = lambda *a, **k: None
    sys.modules.setdefault("logging_config", stub)
    import main
    from vector_source import VectorSource
    if cpus:
        import faiss
        import psutil
        psutil.Process().cpu_affinity(list(cpus))
        faiss.omp_set_num_threads(len(cpus))

    out = os.path.join(OUT_DIR, f"folds_{num_vectors}")
    os.makedirs(out, exist_ok=True)
    for name, (plot_key, _) in main.BLOCKLIST_DATASETS.items():
        if corpora and plot_key not in corpora:
            continue
        pool_n = len(VectorSource(name))
        xgroup = name == "CCNews"
        labels = main._cluster_labels(name, None, main.BLOCKLIST_EPS,
                                      exclude_same=np.zeros(1) if xgroup else None, pool_n=pool_n)
        sizes = np.bincount(labels)
        dup = np.where(sizes >= main.BLOCKLIST_MIN_CLUSTER)[0]
        singletons = np.where(np.isin(labels, np.where(sizes == 1)[0]))[0]
        stream = int(round(main.BLOCKLIST_QUERY_FRACTION * num_vectors))
        n_h, n_w = int(0.2 * stream), int(0.4 * stream)
        selections = main._fold_selections(
            name, labels, dup, singletons, main._group_by_label(labels, dup),
            n_folds=main.BLOCKLIST_FOLDS, reference_target=num_vectors - stream,
            n_heldout=n_h, n_withheld=n_w, n_singleton=stream - n_h - n_w, pool_n=pool_n,
            eps=main.BLOCKLIST_EPS, stream_size=stream, query_fraction=main.BLOCKLIST_QUERY_FRACTION,
            xgroup=xgroup, cache_dir=out)
        for fold, sel in enumerate(selections):
            B = main._read_rows(name, sel["reference"])
            Q = np.concatenate([main._read_rows(name, sel[k]) for k in main.QUERY_SOURCES if len(sel[k])])
            actual = np.asarray(main.compute_static_actual_results(B, Q, TAU), dtype=bool)
            np.savez(os.path.join(out, f"{plot_key}_{fold}.npz"), B=B, Q=Q, actual=actual,
                     n_heldout=len(sel["heldout"]), n_withheld=len(sel["withheld"]),
                     n_singleton=len(sel["singleton"]))
            print(f"{name} fold {fold}: |B|={len(B)} queries={len(Q)} positives={int(actual.sum())}", flush=True)


# ---------------------------------------------------------------------- run
def _pin(cpu_queue) -> None:
    import psutil
    psutil.Process().cpu_affinity([cpu_queue.get()])


def accept_angles(f, tau: float) -> np.ndarray:
    """Per stored region, the angle of the cap of unit queries it accepts at
    tau (pi: the whole sphere); regions no unit query can reach are left
    out. A cap: its stored angle plus beta at |q| = 1. A ball (c, r): the cap
    {q : |q - c| <= r + tau + eps} cuts from the sphere."""
    if hasattr(f, "all_angles"):
        beta, _ = f._beta(np.ones(1), tau)
        return np.minimum(f.all_angles + beta[0], np.pi)
    C = np.asarray(f.all_centers, dtype=np.float64)
    n = np.linalg.norm(C, axis=1)
    R = f.all_radiuses.astype(np.float64) + tau + f._COVERAGE_EPSILON
    cos_psi = (1.0 + n * n - R * R) / (2.0 * np.maximum(n, 1e-300))
    return np.arccos(np.clip(cos_psi[cos_psi <= 1.0], -1.0, 1.0))


def accept_volume(f, tau: float) -> dict:
    """How much of the unit sphere (where every query lies) the filter
    accepts at tau, from the stored regions alone: log10 of the summed cap
    measures (an upper bound on their union, the cap form of the paper's
    J_tau) and of the largest (a lower bound), plus the accept angles' max
    and median."""
    from filters.cap_cover.cap_geometry import log10_cap_measure
    psi = accept_angles(f, tau)
    if len(psi) == 0:
        return dict(log10_vol=-np.inf, log10_vol_max=-np.inf, psi_max=np.nan, psi_med=np.nan)
    lv = log10_cap_measure(psi, DIM)
    top = float(lv.max())
    return dict(log10_vol=top + float(np.log10(np.sum(10.0 ** (lv - top)))), log10_vol_max=top,
                psi_max=float(psi.max()), psi_med=float(np.median(psi)))


def run_job(folds_dir: str, config: str, corpus: str, fold: int, kf: float) -> dict:
    z = np.load(os.path.join(folds_dir, f"{corpus}_{fold}.npz"))
    B, Q, actual = z["B"], z["Q"], z["actual"]
    k = max(1, round(kf * len(B)))
    budget = ball_budget(k)
    f = _configs()[config](budget)
    start = time.perf_counter()
    f.construct(B)
    construct_s = time.perf_counter() - start
    start = time.perf_counter()
    predicted = np.asarray(f.query(Q, TAU), dtype=bool)
    query_s = time.perf_counter() - start
    fn = int((~predicted & actual).sum())
    if fn:
        raise ValueError(f"{config} {corpus} fold {fold} kf {kf}: {fn} false negatives")
    if f.get_bytesize() > budget:
        raise ValueError(f"{config} {corpus} fold {fold} kf {kf}: {f.get_bytesize()} B > budget {budget}")
    neg = ~actual
    proxies = {}
    if hasattr(f, "encoding_error") and len(getattr(f, "candidates", getattr(f, "levels", ()))) == 1:
        # The reach proxy auto-selection scores encodings by, in several
        # forms, on the finished filter -- for checking which form ranks
        # encodings like their measured FPR does.
        from filters.cap_cover.cap_geometry import nearest_directions
        alpha = 2.0 * np.arcsin(TAU / 2.0)
        theta = f.all_angles
        phi = np.sqrt(np.maximum((theta + alpha) ** 2 - f.encoding_error ** 2, 1e-300))
        _, home = nearest_directions(f._prepare_members(B), f.all_directions)
        mass = np.bincount(home, minlength=len(phi)).astype(float)
        for p in (4, 8, 12):
            proxies[f"proxy_u{p}"] = float(np.logaddexp.reduce(p * np.log(phi)))
            proxies[f"proxy_w{p}"] = float(np.logaddexp.reduce(p * np.log(phi) + np.log(np.maximum(mass, 1e-300))))
    row = dict(config=config, corpus=corpus, fold=fold, kf=kf, k_ball=k, budget=budget,
               bytes=int(f.get_bytesize()),
               n_regions=int(getattr(f, "_num_caps", getattr(f, "_num_spheres", -1))),
               encoding=getattr(getattr(f, "quantize", None), "value", ""),
               override=str(getattr(f, "selection_override", None) or ""),
               fpr=float(predicted[neg].mean()), construct_s=construct_s, query_s=query_s,
               **accept_volume(f, TAU), **proxies)
    start = 0
    for source, n in (("heldout", int(z["n_heldout"])), ("withheld", int(z["n_withheld"])),
                      ("singleton", int(z["n_singleton"]))):
        m = neg[start:start + n]
        row[f"fpr_{source}"] = float(predicted[start:start + n][m].mean()) if m.any() else np.nan
        start += n
    return row


def tau_job(folds_dir: str, config: str, corpus: str, fold: int, kf: float, taus=(0.05, 0.075, 0.1, 0.15, 0.2)):
    """One filter built at the usual budget, queried at several taus; ground
    truth per tau from the exact nearest-reference distance, the same float32
    test main.compute_static_actual_results applies (|q - x|^2 <= tau^2)."""
    import faiss
    z = np.load(os.path.join(folds_dir, f"{corpus}_{fold}.npz"))
    B, Q = z["B"], z["Q"]
    index = faiss.IndexFlatL2(B.shape[1])
    index.add(np.ascontiguousarray(B, dtype=np.float32))
    d2 = index.search(np.ascontiguousarray(Q, dtype=np.float32), 1)[0][:, 0]
    if not np.array_equal(d2 <= TAU ** 2, z["actual"]):
        raise ValueError(f"{corpus} fold {fold}: ground truth at tau={TAU} does not reproduce the fold's")
    k = max(1, round(kf * len(B)))
    budget = ball_budget(k)
    f = _configs()[config](budget)
    f.construct(B)
    if f.get_bytesize() > budget:
        raise ValueError(f"{config} {corpus} fold {fold} kf {kf}: {f.get_bytesize()} B > budget {budget}")
    rows = []
    for tau in taus:
        actual = d2 <= np.float32(tau) ** 2
        predicted = np.asarray(f.query(Q, tau), dtype=bool)
        fn = int((~predicted & actual).sum())
        if fn:
            raise ValueError(f"{config} {corpus} fold {fold} kf {kf} tau {tau}: {fn} false negatives")
        neg = ~actual
        row = dict(config=config, corpus=corpus, fold=fold, kf=kf, tau=tau, k_ball=k, budget=budget,
                   encoding=getattr(getattr(f, "quantize", None), "value", ""),
                   positives=int(actual.sum()), fpr=float(predicted[neg].mean()))
        start = 0
        for source, n in (("heldout", int(z["n_heldout"])), ("withheld", int(z["n_withheld"])),
                          ("singleton", int(z["n_singleton"]))):
            m = neg[start:start + n]
            row[f"fpr_{source}"] = float(predicted[start:start + n][m].mean()) if m.any() else np.nan
            start += n
        rows.append(row)
    return rows


def stream_job(folds_dir: str, config: str, corpus: str, fold: int, kf: float, bootstrap: float = 0.5,
               checkpoints: int = 4, batch: int = 200):
    """main.generate_streaming's protocol at this module's budget: B in its
    fold order, the first `bootstrap` of it built by construct(), the rest
    inserted (insert() takes one vector at a time inside), FPR at
    `checkpoints` log-spaced positions against the ground truth of the
    prefix that has arrived. max_memory bounds the filter after every
    insert() call, not just at the end."""
    import faiss
    z = np.load(os.path.join(folds_dir, f"{corpus}_{fold}.npz"))
    B, Q = z["B"], z["Q"]
    n = len(B)
    boot = max(1, round(bootstrap * n))
    k = max(1, round(kf * n))
    budget = ball_budget(k)
    stops = sorted(set(int(round(x)) for x in np.logspace(np.log10(boot + 1), np.log10(n), checkpoints)))
    f = _configs()[config](budget)
    budget = int(f.max_memory)   # the *_b32 configs scale it
    index = faiss.IndexFlatL2(B.shape[1])
    Q32 = np.ascontiguousarray(Q, dtype=np.float32)

    def nearest_sq(lo, hi):
        index.reset()
        index.add(np.ascontiguousarray(B[lo:hi], dtype=np.float32))
        return index.search(Q32, 1)[0][:, 0]

    start = time.perf_counter()
    f.construct(B[:boot])
    d2 = nearest_sq(0, boot)
    insert_s, peak, pos, rows = 0.0, f.get_bytesize(), boot, []
    for stop in stops:
        seen = pos
        while pos < stop:
            t = time.perf_counter()
            f.insert(B[pos:min(pos + batch, stop)])
            insert_s += time.perf_counter() - t
            peak = max(peak, f.get_bytesize())
            if peak > budget:
                raise ValueError(f"{config} {corpus} fold {fold}: {peak} B > budget {budget} at {pos}")
            pos = min(pos + batch, stop)
        if stop > seen:
            d2 = np.minimum(d2, nearest_sq(seen, stop))
        actual = d2 <= TAU ** 2
        predicted = np.asarray(f.query(Q, TAU), dtype=bool)
        fn = int((~predicted & actual).sum())
        if fn:
            raise ValueError(f"{config} {corpus} fold {fold} at {stop}: {fn} false negatives")
        neg = ~actual
        rows.append(dict(config=config, corpus=corpus, fold=fold, kf=kf, k_ball=k, budget=budget, position=stop,
                         n_regions=int(getattr(f, "_num_caps", getattr(f, "_num_spheres", -1))),
                         encoding=getattr(getattr(f, "quantize", None), "value", ""),
                         peak_bytes=int(peak), fpr=float(predicted[neg].mean()),
                         insert_s=insert_s, total_s=time.perf_counter() - start))
    return rows


def _parse_cpus(spec: str):
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return out


def run(tag: str, configs, kfs, folds, corpora, cpus, folds_dir: str, job=run_job) -> pd.DataFrame:
    """`job` returns one row (run_job) or a list of rows (tau_job)."""
    import multiprocessing as mp
    unknown = set(configs) - set(_configs())
    if unknown:
        raise SystemExit(f"unknown configs: {sorted(unknown)}")
    jobs = [(folds_dir, c, corpus, fold, kf) for kf in sorted(kfs, reverse=True)
            for corpus in corpora for fold in folds for c in configs]
    queue = mp.Manager().Queue()
    for cpu in cpus:
        queue.put(cpu)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{tag}.csv")
    rows, done, start = [], 0, time.perf_counter()
    # Rows are written as jobs finish (rows are self-describing, so their
    # order does not matter): an interrupted run keeps what it finished.
    with ProcessPoolExecutor(len(cpus), initializer=_pin, initargs=(queue,)) as ex:
        futures = [ex.submit(job, *args) for args in jobs]
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as e:
                print("JOB FAILED:", e, flush=True)
                continue
            rows.extend(result if isinstance(result, list) else [result])
            done += 1
            pd.DataFrame(rows).to_csv(path, index=False)
    df = pd.DataFrame(rows)
    print(f"{done}/{len(jobs)} jobs in {time.perf_counter() - start:.0f}s -> {path}")
    return df


# ------------------------------------------------------------------ summary
def summary(df: pd.DataFrame, base: str = "cap_auto") -> None:
    pd.set_option("display.width", 220)
    print("mean FPR by k_fraction\n", df.pivot_table(index="config", columns="kf", values="fpr").round(4))
    balls = df[df.config.str.startswith("ball")]
    key = ["corpus", "fold", "kf"]
    if base in set(df.config) and len(balls):
        best_ball = balls.groupby(key).fpr.min()
        best_ball_cfg = balls.loc[balls.groupby(key).fpr.idxmin()].set_index(key).config
        mine = df[df.config == base].set_index(key).fpr
        diff = (mine - best_ball).dropna()
        print(f"\n{base} - best Ball-Cover config per cell (min over {sorted(balls.config.unique())}):")
        for kf, d in diff.groupby(level="kf"):
            print(f"  kf={kf:<6} {d.mean():+.4f}  ({int((d < 0).sum())} wins / {int((d > 0).sum())} losses "
                  f"of {len(d)})  best ball mean {best_ball.loc[d.index].mean():.4f}")
        print("  best Ball config counts:", best_ball_cfg.value_counts().to_dict())


def main_cli() -> None:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--num-vectors", type=int, default=10_000)
    p.add_argument("--corpora", default=",".join(CORPORA))
    p.add_argument("--cpus", default="16-27")
    for name in ("run", "tau", "stream"):
        r = sub.add_parser(name)
        r.add_argument("tag")
        r.add_argument("configs")
        r.add_argument("--kf", default=",".join(map(str, K_FRACTIONS)))
        r.add_argument("--folds", default="0,1,2,3,4")
        r.add_argument("--corpora", default=",".join(CORPORA))
        r.add_argument("--cpus", default="16-27")
        r.add_argument("--num-vectors", type=int, default=10_000)
    s = sub.add_parser("summary")
    s.add_argument("tags")
    s.add_argument("--base", default="cap_auto")
    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare(a.num_vectors, a.corpora.split(","), _parse_cpus(a.cpus))
    elif a.cmd in ("run", "tau", "stream"):
        job = {"run": run_job, "tau": tau_job, "stream": stream_job}[a.cmd]
        df = run(a.tag, a.configs.split(","), [float(x) for x in a.kf.split(",")],
                 [int(x) for x in a.folds.split(",")], a.corpora.split(","), _parse_cpus(a.cpus),
                 os.path.join(OUT_DIR, f"folds_{a.num_vectors}"), job=job)
        if a.cmd == "run":
            summary(df)
        elif a.cmd == "tau":
            print(df.pivot_table(index="config", columns="tau", values="fpr").round(4))
        else:
            print(df.pivot_table(index="config", columns="position", values="fpr").round(4))
    else:
        df = pd.concat([pd.read_csv(os.path.join(OUT_DIR, f"{t}.csv")) for t in a.tags.split(",")])
        summary(df, a.base)


if __name__ == "__main__":
    main_cli()
