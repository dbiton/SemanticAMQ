"""BallCoverFilter construction methods compared on the blocklist workload:
GonzalesConstruct (the default) against MergeConstruct, KMeansConstruct,
MinimaxConstruct (plain and with sphere-projected centers), SplitConstruct
and SphereMergeConstruct.

Same construction and query sets as generate_budget / generate_quantization
(main.blocklist_workloads), with ground truth defined as distance to the
reference set at tau (main.compute_static_actual_results). Only the
construction method varies: every filter is BallCoverFilter's default
quantization with k = max_spheres and no insert method.

Run from the repo root (dataset paths are relative to it):

    python src/compare_construct_methods.py                 # sweep, CSVs, figures
    python src/compare_construct_methods.py --plot-only     # figures from existing CSVs
    python src/compare_construct_methods.py --scale 100000  # construction time at n=100K

A nonzero FNR in any row is a bug in that construction method, not a data
point: the CSV is still written so the row can be inspected, and then the run
fails.
"""

import os

# Before numpy/faiss, as in main.py and runner.py: each worker process stays
# single-threaded so MAX_WORKERS processes don't oversubscribe the machine.
# It also means every construction time here is a single-core time.
for _env_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env_var, "1")

import argparse
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Sequence, Tuple

import faiss
import numpy as np
import pandas as pd

faiss.omp_set_num_threads(1)

from logging_config import configure_logging
from runner import _log10_J, _tau_volume_bounds_log10
from filters.ball_cover.ball_cover_filter import BallCoverFilter
from filters.ball_cover.ball_interfaces import ConstructInterface
from filters.ball_cover.gonzales_construct import GonzalesConstruct
from filters.ball_cover.merge_construct import MergeConstruct
from filters.ball_cover.kmeans_construct import KMeansConstruct
from filters.ball_cover.minimax_construct import MinimaxConstruct
from filters.ball_cover.split_construct import SplitConstruct
from filters.ball_cover.sphere_merge_construct import SphereMergeConstruct
from util.dataset_paths import dataset_filenames
from vector_source import VectorSource

configure_logging()
logger = logging.getLogger(__name__)

DIM = 384
TAU = 0.1
# k as a fraction of |B|, log-spaced from 0.5% to 20%. The top of the range is
# generate_budget's; the bottom extends it into the budgets where every ball
# is wide, which is where the construction method matters most.
K_FRACTIONS = (0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2)
METHODS = {
    "gonzalez": GonzalesConstruct,
    "merge": MergeConstruct,
    # seed_multiplier=8: the default that best preserves merge's own quality
    # (log10_J close to plain merge's, unlike 2x/4x) while still giving a
    # meaningful construction-time win at scale -- see the profiling note in
    # merge_construct.py's docstring and this session's report.
    "merge_seeded": lambda: MergeConstruct(seed_multiplier=8),
    "kmeans": KMeansConstruct,
    "minimax": MinimaxConstruct,
    # minimax's own centers, moved onto the data's sphere before the radii are
    # measured (exact_cover.sphere_centers) -- isolates what the sphere
    # projection alone is worth, at minimax's cost.
    "minimax_sphere": lambda: MinimaxConstruct(sphere=True),
    "split": SplitConstruct,
    "sphere_merge": SphereMergeConstruct,
}
# main.BLOCKLIST_DATASETS' registry names, repeated here so importing this
# module (which every worker does) doesn't import main and its ML stack.
DATASETS = ("WildChat", "CCNews", "WikiAnswers", "Quora", "ChatGPTParaphrases")
RESULTS_DIR = "results/construct_methods"
PLOT_DIR = "plots/construct_methods"
MAX_WORKERS = 16


class _Timed(ConstructInterface):
    """Times construct_filter alone, apart from the rest of construct()
    (_tight_radii, index writes), which every method shares."""

    def __init__(self, inner: ConstructInterface) -> None:
        self.inner = inner
        self.seconds = float("nan")

    def construct_filter(self, filter, vectors):
        start = time.perf_counter()
        out = self.inner.construct_filter(filter, vectors)
        self.seconds = time.perf_counter() - start
        return out

    def get_bytesize(self) -> int:
        return self.inner.get_bytesize()


def evaluate_method(dataset: str, fold: int, method: str, k_fractions: Sequence[float],
                    tau: float, construct: np.ndarray, queries: np.ndarray,
                    actual: np.ndarray, n_positive: int, n_hard: int) -> List[Dict[str, Any]]:
    """One row per k: cover volume, radii, FPR/FNR against distance ground
    truth, and construction time. Queries are main's positives, then hard
    negatives, then easy negatives (BlocklistWorkload.query_vectors), which
    is what the membership-based columns slice on -- those are
    budget_worker's definitions, kept only so the Gonzalez rows can be
    checked against results/budget.csv."""
    n = len(construct)
    pos = slice(0, n_positive)
    hard = slice(n_positive, n_positive + n_hard)
    easy = slice(n_positive + n_hard, len(queries))

    rows = []
    for kf in k_fractions:
        k = min(max(1, round(kf * n)), n)
        timed = _Timed(METHODS[method]())
        filt = BallCoverFilter(DIM, construct_method=timed, max_spheres=k, insert_method=None)
        start = time.perf_counter()
        filt.construct(construct)
        construct_s = time.perf_counter() - start

        centers = np.ascontiguousarray(filt.all_centers, dtype=np.float32)
        radii = filt.all_radiuses
        start = time.perf_counter()
        vol_low, vol_high = _tau_volume_bounds_log10(centers, radii, tau)
        volume_s = time.perf_counter() - start

        predicted = np.asarray(filt.query(queries, distance_threshold=tau), dtype=bool)
        tp = int(np.sum(predicted & actual))
        fp = int(np.sum(predicted & ~actual))
        fn = int(np.sum(~predicted & actual))
        tn = int(np.sum(~predicted & ~actual))

        rows.append({
            "dataset": dataset, "fold": fold, "method": method,
            "k_fraction": float(kf), "k": k, "n_balls": len(radii), "n_reference": n,
            "tau": tau,
            "log10_vol_low": vol_low, "log10_vol_high": vol_high,
            "log10_j_tau": _log10_J(radii, DIM, tau=tau),
            "r_mean": float(radii.mean()), "r_med": float(np.median(radii)),
            "r_max": float(radii.max()),
            "fpr": fp / (fp + tn) if fp + tn else 0.0,
            "fnr": fn / (fn + tp) if fn + tp else 0.0,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "recall_membership": float(predicted[pos].mean()) if n_positive else float("nan"),
            "fpr_hard_membership": float(predicted[hard].mean()) if n_hard else float("nan"),
            "fpr_easy_membership": float(predicted[easy].mean()) if easy.stop > easy.start else float("nan"),
            "method_s": timed.seconds, "construct_s": construct_s, "volume_s": volume_s,
        })
    return rows


def run_sweep(datasets: Sequence[str], n_folds: int, num_vectors: int,
              k_fractions: Sequence[float], methods: Sequence[str], tau: float,
              workers: int, results_dir: str) -> pd.DataFrame:
    import main  # parent only: main imports the ML stack for MLPFilter

    missing = [d for d in datasets if not os.path.exists(dataset_filenames[d])]
    if missing:
        raise SystemExit(
            f"Dataset file(s) not on disk: {', '.join(f'{d} ({dataset_filenames[d]})' for d in missing)}. "
            "Generate them with src/util/fetch_datasets.py. Not substituting synthetic data: i.i.d. "
            "random vectors cover nothing like real embeddings do.")

    futures = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for name in datasets:
            workloads = main.blocklist_workloads(num_vectors, n_folds=n_folds, only=name)
            if not workloads:
                raise SystemExit(f"{name}: no usable blocklist workload at num_vectors={num_vectors}; "
                                 "see the log above for why.")
            workload = workloads[0]
            for fold, sel in enumerate(workload.selections):
                construct = workload.rows(sel["reference"])
                queries = workload.query_vectors(sel)
                actual = np.asarray(main.compute_static_actual_results(construct, queries, tau), dtype=bool)
                logger.info(f"{name} fold {fold}: |B|={len(construct)}, {len(queries)} queries, "
                            f"{int(actual.sum())} within tau={tau} of B")
                for method in methods:
                    futures.append(pool.submit(
                        evaluate_method, workload.plot_key, fold, method, k_fractions, tau,
                        construct, queries, actual, len(sel["positive"]), len(sel["hard"])))

        rows = []
        for i, future in enumerate(as_completed(futures), 1):
            job_rows = future.result()
            rows.extend(job_rows)
            first = job_rows[0]
            logger.info(f"[{i}/{len(futures)}] {first['dataset']} fold {first['fold']} {first['method']}")

    # Rows arrive in completion order; each one names its own job.
    df = pd.DataFrame(rows).sort_values(["dataset", "method", "k_fraction", "fold"]).reset_index(drop=True)
    os.makedirs(results_dir, exist_ok=True)
    for dataset, sub in df.groupby("dataset"):
        path = os.path.join(results_dir, f"{dataset}.csv")
        sub.to_csv(path, index=False)
        logger.info(f"Wrote {path}")
    return df


def assert_zero_fnr(df: pd.DataFrame) -> None:
    bad = df[df["fn"] > 0]
    if len(bad):
        raise AssertionError(
            "Zero-false-negative violation -- a bug in the construction method:\n"
            + bad[["dataset", "fold", "method", "k", "fn", "tp", "fnr"]].to_string(index=False))
    logger.info(f"FNR == 0 in all {len(df)} rows "
                f"({int(df['tp'].sum())} true positives across them)")


def time_at_scale(dataset: str, n: int, method: str, k: int) -> Dict[str, Any]:
    """construct_filter alone on the first n rows of a corpus, plus a
    coverage check of its output."""
    X = np.ascontiguousarray(VectorSource(dataset, 0, n).to_array(), dtype=np.float32)
    filt = BallCoverFilter(DIM, max_spheres=k, insert_method=None)
    start = time.perf_counter()
    centers, radii = METHODS[method]().construct_filter(filt, X)
    seconds = time.perf_counter() - start

    # Signed distance to the 16 nearest balls, which include the one each
    # vector was assigned to -- the same test test_ball_cover_construction
    # applies, at the filter's own coverage epsilon.
    index = faiss.IndexFlatL2(DIM)
    index.add(np.ascontiguousarray(centers, dtype=np.float32))
    d2, labels = index.search(X, min(16, len(centers)))
    worst = float((np.sqrt(np.maximum(d2, 0.0)) - radii[labels]).min(axis=1).max())
    return {"dataset": dataset, "n": len(X), "method": method, "k": k, "n_balls": len(radii),
            "method_s": seconds, "r_max": float(radii.max()), "r_mean": float(radii.mean()),
            "worst_uncovered": worst, "covered": worst <= BallCoverFilter._COVERAGE_EPSILON}


def run_scale(dataset: str, n: int, ks: Sequence[int], methods: Sequence[str],
              workers: int, results_dir: str) -> pd.DataFrame:
    if not os.path.exists(dataset_filenames[dataset]):
        raise SystemExit(f"{dataset}: {dataset_filenames[dataset]} not on disk.")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(time_at_scale, dataset, n, m, k) for m in methods for k in ks]
        rows = []
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            logger.info(f"{row['method']} k={row['k']}: {row['method_s']:.1f}s, covered={row['covered']}")
    df = pd.DataFrame(rows).sort_values(["k", "method"]).reset_index(drop=True)
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"scale_{dataset}_{n}.csv")
    df.to_csv(path, index=False)
    logger.info(f"Wrote {path}")
    if not df["covered"].all():
        raise AssertionError("Coverage violation at scale:\n" + df[~df["covered"]].to_string(index=False))
    return df


def main_cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--num-vectors", type=int, default=None,
                        help="workload size (default: main.WORKLOAD_NUM_VECTORS)")
    parser.add_argument("--k-fractions", type=float, nargs="+", default=list(K_FRACTIONS))
    parser.add_argument("--tau", type=float, default=TAU)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--plot-dir", default=PLOT_DIR)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--scale", type=int, default=None, metavar="N",
                        help="time construct_filter on the first N rows instead of sweeping")
    parser.add_argument("--scale-dataset", default="WikiAnswers", choices=DATASETS)
    parser.add_argument("--scale-k", type=int, nargs="+", default=[1000, 5000])
    args = parser.parse_args()

    if args.scale is not None:
        run_scale(args.scale_dataset, args.scale, args.scale_k, args.methods, args.workers, args.results_dir)
        return

    from plots.plot_construct_methods import generate_construct_methods_figures, load_results

    if args.plot_only:
        df = load_results(args.results_dir)
    else:
        num_vectors = args.num_vectors
        if num_vectors is None:
            import main
            num_vectors = main.WORKLOAD_NUM_VECTORS
        df = run_sweep(args.datasets, args.folds, num_vectors, args.k_fractions, args.methods,
                       args.tau, args.workers, args.results_dir)
    generate_construct_methods_figures(df, args.plot_dir)
    assert_zero_fnr(df)


if __name__ == "__main__":
    main_cli()
