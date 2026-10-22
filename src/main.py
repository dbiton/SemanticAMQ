import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

# Must happen before numpy/faiss are imported: otherwise each of Runner's
# MAX_WORKERS processes also grabs every CPU core for its own internal
# threads, and process x thread parallelism multiply into severe
# oversubscription (observed: whole-system unresponsiveness).
for _env_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env_var, "1")

import faiss
import h5py
import numpy as np
import pandas as pd

faiss.omp_set_num_threads(1)  # belt-and-suspenders alongside the env vars above

from logging_config import configure_logging
from runner import (Runner,
                    CAP_QUANTIZATION_SWEEP, CAP_CONSTRUCT_METHODS, CAP_COVER_CONSTRUCT,
                    system_tau_ground_truth_worker)
from plots.plot_query_performance import generate_query_performance_figures
from plots.plot_equal_memory import generate_equal_memory_figures
from plots.plot_scaling import generate_scaling_figures
from plots.plot_streaming import generate_streaming_figures
from plots.plot_blocklist import generate_blocklist_figures
from plots.plot_budget import generate_budget_figures
from plots.plot_construct_methods import generate_construct_methods_figures, load_results as load_construct_methods
from plots.plot_common import DATASETS, DATASET_LABELS
from typing import Any, Dict, List, Optional, Tuple
from filters.cap_cover.cap_cover_filter import CapCoverFilter
from filters.dsh_filter import DSHFilter
from filters.hnsw_filter import HNSWFilter
from filters.ivf_filter import IVFFilter
from filters.ivf_radii_filter import IVFRadiiFilter
from filters.quantized_vector_filter import QuantizedVectorFilter
from filters.mlp_filter import MLPFilter
from filters.pacuk_lsh_filter import PacukLSHFilter
from filters.sankowski_wygocki_filter import SankowskiWygockiFilter
from filters.va_plus_filter import VAPlusFilter
from util.dataset_paths import dataset_filenames, HDF5_DATASET_KEY
from util.fetch_datasets import load_embeds
from vector_source import VectorSource

configure_logging()
logger = logging.getLogger(__name__)

# The evaluation machine: AMD Ryzen Threadripper 3970X (32 cores / 64 threads), Ubuntu
# 24.04, 896 GB RAM (paper.tex). Every job is single-threaded and holds a few GB at
# WORKLOAD_NUM_VECTORS = 100K. One worker per physical core, less two for the
# coordinating process (which pickles each job's vectors and writes the CSVs) and
# the OS: 60 workers on 32 cores oversubscribed the SMT siblings and the memory
# bandwidth, and inflated every timing (a quiet pinned core scans a cap in 1-2x
# the time of a laptop core; a loaded run measured 5-10x). src/paper.py times its
# throughput evaluation separately, on a few pinned cores.
MAX_WORKERS = 30
DATASET = 'ELI5'

# Vectors/dataset, split 80% construction / 20% queries. One knob for every
# evaluation: the ground-truth FPR/FNR ones (cap_cover_system_tau, cap_quantization,
# equal_memory, streaming, blocklist) and the throughput-only sweeps (scaling,
# query_performance), which used to read a separate STREAM_SIZE.
#
# The paper's size: |B| = 80,000 construct and 20,000 query vectors per fold. The
# blocklist workload cuts BLOCKLIST_FOLDS disjoint windows of this many rows from
# each corpus file (build_blocklist_workload), so a corpus needs
# BLOCKLIST_FOLDS * WORKLOAD_NUM_VECTORS rows (500K) or it is skipped with an
# error, and cap_cover_system_tau's non-wrapping folds need 1.8N rows.
WORKLOAD_NUM_VECTORS = 100_000

# The distance threshold (L2, unit vectors: tau^2 = 2 * cosine distance) at which every
# fixed-threshold evaluation runs, and the grid of the blocklist's tau sweep.
# src/paper.py overrides both for the paper's figures.
FIXED_TAU = 0.1
BLOCKLIST_TAUS = [float(t) for t in np.linspace(0.0, 0.4, 8)]

BLOCKLIST_FOLDS = 5
# The blocklist's query/construct split of WORKLOAD_NUM_VECTORS: one total
# size, cut into a q query part and a (1-q) construct part, with
# k = k_fraction * |B|. Position-based (build_blocklist_workload): construct
# rows come from the start of the file, queries from its end.
#
#     stream (queries) = q * N          |B| (reference) = (1 - q) * N
BLOCKLIST_QUERY_FRACTION = 0.2

# Registry name -> canonical dataset key, one source of truth via plot_common.
DATASET_KEYS = {DATASET_LABELS[d]: d for d in DATASETS}

# generate_cap_quantization's analogue, parallel to generate_quantization
# rather than replacing it in place -- the two sweep structurally different
# encodings (see CAP_QUANTIZATION_SWEEP), so keeping both lets either be
# re-run/compared independently.
CAP_QUANTIZATION_DIR = "results/cap_quantization"
CAP_QUANTIZATION_PLOT_DIR = "plots/cap_quantization"


def cap_quantization_csv(dataset: str) -> str:
    return os.path.join(CAP_QUANTIZATION_DIR, dataset, "cap_quantization.csv")


def cap_quantization_plot_dir(dataset: str) -> str:
    return os.path.join(CAP_QUANTIZATION_PLOT_DIR, dataset)

# generate_scaling runs once per blocklist corpus and keeps each run in its
# own folder, mirroring cap_quantization_csv/CAP_QUANTIZATION_DIR above: throughput
# is pooled across corpora for plotting, but keeping each corpus's raw rows
# apart on disk makes a single corpus's run inspectable/rerunnable on its own.
SCALING_DIR = "results/scaling"


def scaling_query_csv(dataset: str) -> str:
    return os.path.join(SCALING_DIR, dataset, "scaling_query.csv")


def scaling_insert_csv(dataset: str) -> str:
    return os.path.join(SCALING_DIR, dataset, "scaling_insert.csv")


def scaling_construction_csv(dataset: str) -> str:
    return os.path.join(SCALING_DIR, dataset, "scaling_construction.csv")


# generate_equal_memory runs once per blocklist corpus and keeps each run in
# its own folder, mirroring cap_quantization_csv/CAP_QUANTIZATION_DIR above. Plotted
# the same way too (see plot_equal_memory): one figure set per corpus, in its
# own folder, rather than pooled into a single cross-corpus line per filter.
EQUAL_MEMORY_DIR = "results/equal_memory"
EQUAL_MEMORY_PLOT_DIR = "plots/equal_memory"


def equal_memory_csv(dataset: str) -> str:
    return os.path.join(EQUAL_MEMORY_DIR, dataset, "equal_memory.csv")


def equal_memory_plot_dir(dataset: str) -> str:
    return os.path.join(EQUAL_MEMORY_PLOT_DIR, dataset)


# generate_streaming runs once per blocklist corpus and keeps each run in its
# own folder, mirroring cap_quantization_csv/CAP_QUANTIZATION_DIR above; plot_streaming
# pools every dataset's rows the same way plot_scaling does (see its
# docstring: "this figure is single-dataset" refers to color channel reuse
# for method, not to a restriction on how many datasets feed it).
STREAMING_DIR = "results/streaming"


def streaming_csv(dataset: str) -> str:
    return os.path.join(STREAMING_DIR, dataset, "streaming.csv")

# generate_equal_memory()'s own comparison set: excludes DSBFFilter/
# CoveringLSHFallbackFilter/LSBFFilter, which degenerate to ~100% duplicate
# predictions at realistic construct-set sizes (still used elsewhere).
# CapCoverFilter is submitted separately (see generate_equal_memory), not
# listed here.
equal_memory_filters = [
    HNSWFilter, IVFFilter, DSHFilter, QuantizedVectorFilter, MLPFilter,
    PacukLSHFilter, SankowskiWygockiFilter, VAPlusFilter, IVFRadiiFilter,
]

def advancing_fold_starts(pool_len: int, k_folds: int, fold_stride: int, window_len: int) -> List[int]:
    """Non-wrapping K-fold-like sweep: fold f's window starts f*fold_stride
    into the pool and is window_len vectors wide, never wrapping back to the
    start. Raises if the pool isn't long enough for all k_folds windows,
    rather than silently reusing or wrapping a shorter one. Returns the
    fold-start offsets (0, fold_stride, 2*fold_stride, ...)."""
    max_start = (k_folds - 1) * fold_stride
    required = max_start + window_len
    if pool_len < required:
        raise ValueError(
            f"advancing_fold_starts: pool of {pool_len} vectors is too short for a "
            f"non-wrapping {k_folds}-fold sweep (window_len={window_len}, "
            f"fold_stride={fold_stride}); needs >= {required} = "
            "(k_folds-1)*fold_stride + window_len. Load/point at a longer stream, "
            "or lower window_len/k_folds/fold_stride."
        )
    return [f * fold_stride for f in range(k_folds)]


def rows_for_mode(job_results: List[Tuple[str, Any]], mode: str) -> list:
    """Pulls one Runner mode's rows out of a shared Runner.run()'s combined
    (mode, result) list -- result is either one dict per job or a list of
    dicts, flattened uniformly here. Used by save_* functions to recover
    just their own evaluation's rows from a Runner several share."""
    rows = []
    for job_mode, result in job_results:
        if job_mode != mode:
            continue
        rows.extend(result if isinstance(result, list) else [result])
    return rows


def compute_static_actual_results(
    construct_vectors: np.ndarray,
    query_vectors: np.ndarray,
    distance_threshold: float,
) -> list[bool]:
    """Static ground-truth: a query vector is a duplicate iff within
    `distance_threshold` of any vector in the fixed construct set. Uses one
    batched faiss nearest-neighbor call rather than a per-query Python loop;
    faiss returns squared L2, hence the distance_threshold**2 comparison."""
    construct = np.ascontiguousarray(construct_vectors, dtype=np.float32)
    queries = np.ascontiguousarray(query_vectors, dtype=np.float32)

    if len(construct) == 0:
        return [False] * len(queries)

    index = faiss.IndexFlatL2(construct.shape[1])
    index.add(construct)
    nn_sq_dist, _ = index.search(queries, 1)
    return (nn_sq_dist[:, 0] <= distance_threshold ** 2).tolist()


def stratified_calibration_split(
    stream: np.ndarray,
    actual: List[bool],
    calibration_fraction: float,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, List[bool], List[bool]]:
    """Cuts a query stream into a calibration slice and a disjoint measurement
    slice that carry the SAME duplicate rate, and returns
    (calibration_vectors, measurement_vectors, calibration_actual,
    measurement_actual).

    Stratified on `actual`, and shuffled within each stratum, because a
    prefix slice of a file-ordered stream is not a sample of it (earlier, the
    stream was a concatenation of query sources and the prefix was exactly
    one of them), so:

      - calibration saw a set that was ~91% duplicates on Quora, where F1 is
        maximised by answering "duplicate" unconditionally -- which is the
        config _build_calibrated_filter then picked, and the reason filters
        reported FPR near 1.0 while their FNR was 0.
      - measurement saw the negatives only: 2 of 16,000 Quora queries were
        within tau of the construct set, 0 of 16,000 on WikiAnswers and
        ChatGPTParaphrases. Every reported FNR was therefore either a
        2-sample estimate or the fn/(fn+tp)==0/0 fallback of 0.0, not a
        measurement -- IVFFilter's "FNR 1.000" was it missing both of Quora's
        two positives.

    Stratifying keeps both slices representative of the whole stream, so the
    dial chosen on one is the dial that should win on the other."""
    stream = np.asarray(stream)
    labels = np.asarray(actual, dtype=bool)
    if len(stream) != len(labels):
        raise ValueError(
            f"stratified_calibration_split: {len(stream)} vectors but {len(labels)} labels.")

    rng = np.random.default_rng(seed)
    calibration_idx, measurement_idx = [], []
    for value in (True, False):
        group = np.flatnonzero(labels == value)
        if len(group) == 0:
            continue
        rng.shuffle(group)
        # round(), not floor, so a small positive stratum still lands in
        # calibration; len(group)-1 so a stratum is never consumed whole,
        # leaving the measurement slice without it.
        n_calibration = 0
        if len(group) > 1:
            n_calibration = min(len(group) - 1,
                                max(1, round(len(group) * calibration_fraction)))
        calibration_idx.append(group[:n_calibration])
        measurement_idx.append(group[n_calibration:])

    calibration_idx = rng.permutation(np.concatenate(calibration_idx)) if calibration_idx else np.empty(0, dtype=np.int64)
    measurement_idx = rng.permutation(np.concatenate(measurement_idx)) if measurement_idx else np.empty(0, dtype=np.int64)

    if labels.any() and not labels[calibration_idx].any():
        # Nothing downstream can recover from this: F1 is 0 for every config
        # dial candidate, so _build_calibrated_filter's `f1 > best_f1` picks
        # whichever candidate it happened to try first.
        logger.warning(
            "stratified_calibration_split: calibration slice holds no duplicates, so every "
            "config dial candidate will score F1=0 and the first one tried wins by default.")
    if not labels.any():
        logger.warning(
            f"stratified_calibration_split: none of the {len(stream)} queries is within tau of "
            "the construct set, so FNR is undefined (reported as 0.0) for every filter. tau is "
            "too tight for this corpus's duplicate geometry.")

    return (stream[calibration_idx], stream[measurement_idx],
            labels[calibration_idx].tolist(), labels[measurement_idx].tolist())


def generate_equal_memory(runner: Runner, filters: Optional[List[Any]] = None) -> None:
    """Equal-memory (SOTA) comparison across equal_memory_filters. Sweeps
    count_memory_steps budgets from 10% to 100% of raw storage (storing the
    construct set as a plain list of float32 vectors, dim*4 bytes each) in
    equal-sized steps -- see raw_storage_bytes below. For each budget, every
    filter's best config is chosen on a held-out calibration set (see
    runner.py::_build_calibrated_filter), then FPR/FNR/latency/throughput are
    measured on a separate held-out query set.

    Reuses blocklist_workloads() -- the same corpora, position-based fold
    splits and reference set B as generate_blocklist/generate_quantization/
    generate_scaling -- rather than its own single-dataset (MMLU) split, so
    the SOTA baselines are compared on the same data the rest of the suite
    evaluates. tau is one fixed absolute value, the same one generate_budget
    and generate_quantization use, so every filter here is compared at the
    same operating point quantization_worker uses. Each fold's query
    stream (see BlocklistWorkload.query_vectors) is split into a calibration
    slice and a disjoint measurement slice -- stratified on the ground truth,
    NOT a prefix cut, see stratified_calibration_split -- replacing the old
    flat percentage_calibration split of one dataset.

    filters narrows the compared set (default equal_memory_filters; CapCoverFilter
    is always submitted).

    Runs over every blocklist corpus and fold; plot_equal_memory pools all of
    them into one line per filter at each memory budget, so -- like
    generate_scaling -- the budget grid is bounded by the *smallest* corpus's
    reference set, keeping the same absolute budgets comparable, and thus
    poolable, across corpora.

    Only submits 'equal_memory' jobs -- does not run `runner`. Each corpus
    writes its own results/equal_memory/<dataset>/equal_memory.csv (see
    equal_memory_csv()), mirroring cap_quantization_csv/CAP_QUANTIZATION_DIR."""
    num_vectors            = WORKLOAD_NUM_VECTORS
    dim                    = 384
    n_folds                = 5
    tau                    = FIXED_TAU
    # Log-spaced, not linear: the baselines' feasibility floors span more
    # than a decade below this range's top (IVF-PQ ~0.6%, quantized-point-set
    # RaBitQ ~3.6%, of raw storage at n=100K/dim=384 -- HNSW's ~30% floor
    # sits above max_memory_fraction entirely, so it is expected to stay
    # infeasible throughout), so a linear sweep would waste points at the top
    # and barely resolve the first two floors. 17 steps over this range's
    # 2 decades is ~8 points/decade, matching the resolution the wider
    # sweep used.
    count_memory_steps     = 16
    # Sweep bounds, as a fraction of raw_storage_bytes (see below).
    min_memory_fraction    = 0.001
    max_memory_fraction    = 0.10
    # Fraction of each fold's blocklist query stream spent calibrating each
    # filter's config dial; the rest measures FPR/FNR/latency/throughput.
    calibration_fraction   = 0.2
    workloads = blocklist_workloads(num_vectors, n_folds=n_folds)
    if not workloads:
        raise RuntimeError(
            "generate_equal_memory: no usable workload for any corpus (see the errors "
            "above -- most likely WORKLOAD_NUM_VECTORS is too large for their "
            "duplicate clusters)")

    # Memory budgets are shared across corpora (plot_equal_memory pools every
    # dataset's rows into one line per filter/budget), so -- like
    # generate_scaling's k -- they are scaled off the smallest corpus's
    # reference set rather than computed per corpus.
    reference_len = min(len(w.selections[0]["reference"]) for w in workloads)
    # 100% = storing the construct set raw, as a plain list of float32
    # vectors (dim*4 bytes each) -- the same denominator
    # plot_equal_memory's raw-storage-% panels divide by. Swept from
    # min_memory_fraction to max_memory_fraction of that, log-spaced (see
    # count_memory_steps above).
    raw_storage_bytes = int(reference_len * dim * 4)
    memory_budgets = np.geomspace(
        raw_storage_bytes * min_memory_fraction, raw_storage_bytes * max_memory_fraction,
        count_memory_steps, dtype=int)

    for workload in workloads:
        output_csv = equal_memory_csv(workload.name)
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        for fold, sel in enumerate(workload.selections):
            vectors_construct = workload.rows(sel["reference"])
            stream = workload.query_vectors(sel)
            # Ground truth over the whole stream first, then a STRATIFIED cut
            # into the two slices -- a prefix cut of the file-ordered stream
            # can put every positive in calibration and none in measurement.
            # See stratified_calibration_split.
            actual_stream = compute_static_actual_results(vectors_construct, stream, tau)
            vectors_calibration, vectors_query, actual_calibration, actual_query = (
                stratified_calibration_split(stream, actual_stream, calibration_fraction, seed=fold))
            logger.info(
                f"{workload.name} fold {fold}: calibration {len(vectors_calibration)} vectors "
                f"({sum(actual_calibration)} duplicates), measurement {len(vectors_query)} "
                f"({sum(actual_query)} duplicates)")

            for memory_budget in memory_budgets:
                logger.info(
                    f"{workload.name} fold {fold}: submitting jobs for "
                    f"memory_budget={memory_budget} bytes...")
                for filter_cls in (equal_memory_filters if filters is None else filters):
                    runner.submit(
                        'equal_memory',
                        output_csv=output_csv,
                        filter_cls=filter_cls,
                        dim=dim, distance_threshold=tau,
                        memory_budget=int(memory_budget),
                        vectors_construct=vectors_construct, vectors_calibration=vectors_calibration,
                        vectors_query=vectors_query,
                        actual_calibration=actual_calibration, actual_query=actual_query,
                        dataset=workload.name,
                    )
                # CapCoverFilter.auto() already chooses its own best encoding
                # for a budget internally, so there is no second axis to
                # sweep here -- one submission per budget, labelled plainly "CapCoverFilter" by
                # equal_memory_worker's Config naming (no quantize in
                # extra_kwargs).
                runner.submit(
                    'equal_memory',
                    output_csv=output_csv,
                    filter_cls=CapCoverFilter,
                    dim=dim, distance_threshold=tau,
                    memory_budget=int(memory_budget),
                    vectors_construct=vectors_construct, vectors_calibration=vectors_calibration,
                    vectors_query=vectors_query,
                    actual_calibration=actual_calibration, actual_query=actual_query,
                    dataset=workload.name,
                )


def save_equal_memory(job_results: List[Tuple[str, Any]]) -> list:
    """Extracts this evaluation's rows from a shared Runner's combined
    run() output (see generate_equal_memory()). The Runner itself already
    wrote one results/equal_memory/<dataset>/equal_memory.csv per corpus
    incrementally, one row-batch per completed job (see
    generate_equal_memory()'s output_csv=), so this no longer writes the
    file -- it only recovers this evaluation's rows for plot_equal_memory()."""
    return rows_for_mode(job_results, 'equal_memory')


def plot_equal_memory(results: list) -> None:
    """One figure set per corpus, each in its own folder -- mirrors
    plot_quantization's split, for the same reason: a filter's line pooled
    across corpora hides which specific corpus it's winning or losing on,
    and generate_equal_memory already keeps each corpus's raw rows in its
    own results/equal_memory/<dataset>/ file (see equal_memory_csv)."""
    df = pd.DataFrame(results)
    for dataset in sorted(df["Dataset"].unique()):
        outdir = equal_memory_plot_dir(str(dataset))
        logger.info(f"Plotting equal-memory comparison for {dataset} -> {outdir}")
        generate_equal_memory_figures(df=df[df["Dataset"] == dataset], outdir=outdir)


# Output of generate_cap_cover_system_tau.
CAP_COVER_SYSTEM_TAU_CSV = "results/cap_cover_system_tau.csv"


def generate_cap_cover_system_tau(runner: Runner, num_vectors: int = WORKLOAD_NUM_VECTORS,
                                  k_folds: int = 5, k_fraction: float = 0.1,
                                  tau_values: Optional[List[float]] = None) -> None:
    """CapCoverFilter's end-to-end filtering benefit: k_folds non-wrapping
    80/20 folds per dataset, one CapCoverFilter.auto() cover
    (runner._cap_cover) per (dataset, fold) in front of a flat index,
    queried one at a time at every tau in tau_values without ever
    rebuilding. Cap budget k is a fixed fraction of each fold's
    construct-set size, converted to bytes by runner._cover_max_memory.

    Ground truth (nearest-construct-set distance per query) is keyed only by
    (dataset, num_vectors, k_folds, fold), not by which filter queries it, so
    it is cached in results/system_tau_ground_truth and reused across runs.

    Only submits 'cap_cover_system_tau' jobs (output_csv=CAP_COVER_SYSTEM_TAU_CSV)
    -- does not run `runner`. num_vectors/k_folds/k_fraction/tau_values take explicit
    arguments so a smaller local validation run doesn't have to
    monkeypatch module globals -- see CLAUDE.md on sizing local runs."""
    dim              = 384
    tau_values       = [float(t) for t in np.logspace(-2, -0.5, num=10)] if tau_values is None else tau_values
    ground_truth_dir = "results/system_tau_ground_truth"

    datasets = [d for d in DATASET_KEYS if os.path.exists(dataset_filenames[d])]
    missing = [d for d in DATASET_KEYS if d not in datasets]
    if missing:
        logger.warning(f"Skipping dataset(s) not found on disk: {', '.join(missing)}")
    if set(datasets) != set(DATASET_KEYS):
        raise RuntimeError(
            f"generate_cap_cover_system_tau needs all {len(DATASET_KEYS)} datasets "
            f"({', '.join(DATASET_KEYS)}) on disk; only found: {', '.join(datasets) or 'none'}"
        )

    logger.info(f"Cap budget fixed as a fraction of the construct set: {k_fraction}")
    logger.info(f"Fixed tau sweep (shared across datasets): {tau_values}")
    logger.info(f"K-fold cross-validation: {k_folds} folds, non-wrapping contiguous 80/20 spans")

    query_len = num_vectors // k_folds
    construct_len = num_vectors - query_len

    splits = []
    for dataset in datasets:
        total_rows = len(VectorSource(dataset))
        fold_starts = advancing_fold_starts(total_rows, k_folds, query_len, num_vectors)
        for fold, fold_start in enumerate(fold_starts):
            query_start, query_end = fold_start, fold_start + query_len
            construct_start, construct_end = query_end, query_end + construct_len
            construct_source = VectorSource(dataset, construct_start, construct_end)
            query_source = VectorSource(dataset, query_start, query_end)
            splits.append((dataset, fold, construct_source, query_source))

    os.makedirs(ground_truth_dir, exist_ok=True)
    nn_dist_by_split = {}
    to_compute = []
    for dataset, fold, construct_source, query_source in splits:
        cache_path = os.path.join(ground_truth_dir, f"{dataset}_{num_vectors}_{k_folds}fold_{fold}.npy")
        if os.path.exists(cache_path):
            nn_dist_by_split[(dataset, fold)] = np.load(cache_path)
        else:
            to_compute.append((dataset, fold, cache_path, construct_source, query_source))

    if to_compute:
        logger.info(f"Computing ground truth for {len(to_compute)} (dataset, fold) pair(s) not yet cached...")
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(system_tau_ground_truth_worker, dim, cs, qs): (dataset, fold, cache_path)
                for dataset, fold, cache_path, cs, qs in to_compute
            }
            for future in as_completed(futures):
                dataset, fold, cache_path = futures[future]
                nn_dist = future.result()
                np.save(cache_path, nn_dist)
                nn_dist_by_split[(dataset, fold)] = nn_dist
    else:
        logger.info("All ground truth already cached on disk; skipping recomputation.")

    for dataset, fold, construct_source, query_source in splits:
        nn_dist = nn_dist_by_split[(dataset, fold)]
        actual_results_by_tau = {t: (nn_dist <= t).tolist() for t in tau_values}

        k = min(max(1, round(k_fraction * construct_len)), construct_len - 1)

        runner.submit(
            'cap_cover_system_tau',
            output_csv=CAP_COVER_SYSTEM_TAU_CSV,
            dim=dim, dataset=DATASET_KEYS[dataset],
            k=k, tau_values=tau_values, seed=fold,
            vectors_construct=construct_source,
            vectors_query=query_source,
            actual_results_by_tau=actual_results_by_tau,
            )


def save_cap_cover_system_tau(job_results: List[Tuple[str, Any]]) -> pd.DataFrame:
    """Extracts this evaluation's rows from a shared Runner's combined
    run() output (see generate_cap_cover_system_tau()); the Runner itself
    already wrote CAP_COVER_SYSTEM_TAU_CSV incrementally."""
    return pd.DataFrame(rows_for_mode(job_results, 'cap_cover_system_tau'))


def plot_cap_cover_system_tau(df: pd.DataFrame) -> None:
    from plots.plot_cap_cover_system_tau import generate_cap_cover_system_tau_figures
    generate_cap_cover_system_tau_figures(df)


# generate_blocklist()'s corpora: deliberately NOT DATASET_KEYS. That list is
# the 5-dataset symmetric sweep, which this workload replaces rather than
# extends -- and these five were chosen for their duplicate-cluster geometry
# (see BLOCKLIST_DATASETS' per-entry comments), verified by running eps=0.4
# seed-ball discovery on each before adding it, not assumed from the label.
BLOCKLIST_CSV = "results/blocklist.csv"
# generate_budget's rows: the same workload, swept over the ball budget at a
# fixed tau via runner.budget_worker rather than runner.blocklist_tau_worker.
# Its own file rather than reusing BLOCKLIST_CSV, since the two evaluations
# sweep different variables and mixing them would need a second grouping
# column on top of `sweep`.
BUDGET_CSV = "results/budget.csv"
BLOCKLIST_DATASETS = {
    # registry name -> plot key
    "WildChat": "wildchat",
    "CCNews": "ccnews",
    "WikiAnswers": "wikianswers",
    "Quora": "quora",
    "ChatGPTParaphrases": "chatgpt_paraphrases",
}


def _selection_is_usable(selections: List[Dict[str, np.ndarray]]) -> bool:
    """Every fold needs a reference set and a query stream."""
    return bool(selections) and all(
        len(sel["reference"]) and len(sel["queries"]) for sel in selections)


_READ_WINDOW = 65_536  # rows per contiguous HDF5 read in _read_rows (~100 MB at d=384)


def _read_rows(dataset: str, idx: np.ndarray) -> np.ndarray:
    """Just the requested rows of a dataset's embeddings, straight from HDF5.

    Read as contiguous windows of _READ_WINDOW rows, keeping the requested rows
    of each, rather than by h5py fancy indexing: that builds one selection per
    point and scales superlinearly -- 582s for a fold's 80K scattered rows
    against 1.6s for all 808K Quora rows read contiguously. The request is
    sorted for the read and then mapped back to the caller's order.
    """
    order = np.argsort(idx, kind="stable")
    sorted_idx = idx[order]
    uniq, inverse = np.unique(sorted_idx, return_inverse=True)
    with h5py.File(dataset_filenames[dataset], "r") as f:
        dset = f[HDF5_DATASET_KEY]
        block = np.empty((len(uniq), dset.shape[1]), dtype=np.float32)
        a = 0
        while a < len(uniq):
            lo = int(uniq[a])
            b = int(np.searchsorted(uniq, lo + _READ_WINDOW))
            block[a:b] = dset[lo:int(uniq[b - 1]) + 1][uniq[a:b] - lo]
            a = b
    out = np.empty((len(idx), block.shape[1]), dtype=np.float32)
    out[order] = block[inverse]
    return np.ascontiguousarray(out)


class BlocklistWorkload:
    """One corpus's blocklist workload: cluster labels, the per-fold
    reference/query index selection, and a reader for the rows it names.

    Shared by generate_blocklist and generate_quantization so both evaluate on
    the same construction and query sets. The two then differ only in what
    they sweep -- ball budget versus quantizer -- which is the point: a
    difference between their results is a difference in the swept variable,
    not in the data underneath.
    """

    def __init__(self, name: str, plot_key: str, selections: List[Dict[str, np.ndarray]],
                 pool: Optional[np.ndarray], stream_size: int, reference_target: int) -> None:
        self.name = name
        self.plot_key = plot_key
        self.selections = selections
        self._pool = pool
        self.stream_size = stream_size
        self.reference_target = reference_target

    def rows(self, idx: np.ndarray) -> np.ndarray:
        idx = np.asarray(idx, dtype=np.int64)
        return self._pool[idx] if self._pool is not None else _read_rows(self.name, idx)

    def query_vectors(self, sel: Dict[str, np.ndarray]) -> np.ndarray:
        """A fold's queries as one array. Ground truth is always distance to
        the reference set at tau (compute_static_actual_results); how related
        a query is to the set (runner.SIMILARITY_BANDS) only groups FPR."""
        return self.rows(sel["queries"])


def build_blocklist_workload(name: str, plot_key: str,
                             num_vectors: int, n_folds: int) -> Optional[BlocklistWorkload]:
    """Cut a corpus into per-fold construct/query splits by file position, or
    return None (logging why) if the file is too short.

    No clustering and no sampling: fold f constructs from the f-th window of
    the first rows of the file and queries from the f-th window of its last
    rows, so the two regions never overlap and folds are disjoint. All
    queries are carried as one stream; strata are not defined by file
    position, and ground truth is distance at tau regardless."""
    if not os.path.exists(dataset_filenames[name]):
        logger.warning(f"Skipping {name}: not found on disk")
        return None

    total = len(VectorSource(name))
    stream_size = int(round(BLOCKLIST_QUERY_FRACTION * num_vectors))
    reference_target = num_vectors - stream_size
    if n_folds * (reference_target + stream_size) > total:
        logger.error(
            f"Skipping {name}: {n_folds} folds of {reference_target} construct + {stream_size} "
            f"query rows need {n_folds * num_vectors} rows but the file has {total}.")
        return None

    selections = [{
        "reference": np.arange(f * reference_target, (f + 1) * reference_target, dtype=np.int64),
        "queries": np.arange(total - (f + 1) * stream_size, total - f * stream_size, dtype=np.int64),
    } for f in range(n_folds)]

    logger.info(f"{name}: |B|={reference_target} from the start of the file, "
                f"stream={stream_size} from the end ({total} rows)")
    return BlocklistWorkload(name, plot_key, selections, None, stream_size, reference_target)


def blocklist_workloads(num_vectors: int, n_folds: int = BLOCKLIST_FOLDS,
                        only: Optional[str] = None) -> List[BlocklistWorkload]:
    """Every corpus's workload, or just the one `only` names."""
    out = []
    for name, plot_key in BLOCKLIST_DATASETS.items():
        if only is not None and name != only:
            continue
        w = build_blocklist_workload(name, plot_key, num_vectors, n_folds)
        if w is not None:
            out.append(w)
    return out


def generate_blocklist(runner: Runner) -> None:
    """Fixed-budget tau sweep. The reference set B is the first rows of each
    corpus's file and the query stream is the last rows of it
    (build_blocklist_workload): no clustering, no stratified sampling. Every
    query is labeled by its distance to B at each tau
    (runner._blocklist_cover_rows), so the positive share is whatever the
    file order yields and bounds the accesses saved. File order can carry
    structure (e.g. paraphrase groups stored together), which this split
    does not correct.

    A single fixed ball budget (k_fraction, not a sweep) -- varying the
    budget is generate_budget's job now, at its own fixed tau.
    This evaluation is the complementary view: fix k, sweep tau, and read
    how FPR/accesses-saved respond at that one representative budget.
    k_fraction=0.1 sits in the middle of generate_budget's own 5%-20% range,
    so the two evaluations are anchored at a comparable operating point.

    Only submits 'blocklist' jobs (output_csv=BLOCKLIST_CSV) -- does not run
    `runner`. Cluster discovery runs eagerly here, since it needs the whole
    pool in memory.
    """
    dim              = 384
    num_vectors      = WORKLOAD_NUM_VECTORS
    k_fraction       = 0.1  # fixed ball budget; see docstring
    # Shared absolute grid for the tau-swept panels. Stops at 0.4: measured on
    # a 0..1.2 grid, every corpus was already pinned at FPR 1.000 and 0% saved
    # by tau=0.3, so the bulk of that range carried no information. All the
    # structure is in [0, 0.4], and 8 points resolve it while leaving just
    # enough flat tail to show the saturation.
    tau_values       = list(BLOCKLIST_TAUS)
    n_folds          = BLOCKLIST_FOLDS

    for workload in blocklist_workloads(num_vectors, n_folds=n_folds):
        for fold, sel in enumerate(workload.selections):
            runner.submit(
                'blocklist',
                output_csv=BLOCKLIST_CSV,
                dim=dim, dataset=workload.plot_key, seed=fold,
                k_fraction=k_fraction, tau_values=tau_values,
                vectors_reference=workload.rows(sel["reference"]),
                vectors_query=workload.query_vectors(sel),
            )


def save_blocklist(job_results: List[Tuple[str, Any]]) -> pd.DataFrame:
    """Extracts this evaluation's rows from a shared Runner's combined run()
    output (see generate_blocklist())."""
    return pd.DataFrame(rows_for_mode(job_results, 'blocklist'))


def plot_blocklist(df: pd.DataFrame) -> None:
    generate_blocklist_figures(df)


def generate_budget(runner: Runner) -> None:
    """The blocklist workload read along its other axis: the ball budget k is
    the swept variable and tau is held at one fixed value, where
    generate_blocklist instead sweeps tau at one fixed budget.

    Holding tau fixed holds the labels fixed, so FPR and accesses saved move
    with the budget alone and the sweep shows where paying for more balls
    stops buying either. The band is deliberately
    narrow (5%-20% of |B|): below it the cover is too coarse to localise B at
    a fixed threshold, and above it the budget approaches saturation, where
    each ball stands in for so few reference vectors that there is nothing
    left to compress.

    tau is one absolute value across every corpus, which is NOT comparable
    between them -- the distance a duplicate sits at is a property of each
    embedding set. Read these panels as one corpus's response to its budget,
    not as a ranking of corpora.

    Reuses blocklist_workloads() and runner.budget_worker (mode 'budget'):
    identical construction and query sets to generate_blocklist (which uses
    runner.blocklist_tau_worker, mode 'blocklist'), so a difference between
    the two evaluations is a difference in the swept variable alone -- both
    share the cover-and-margin core in runner._blocklist_cover_rows.

    Always requests runner._tau_volume_bounds_log10's overlap-aware volume
    bounds: they cost ~O(k^2) per row, which a tau grid multiplies out of
    reach, but a single fixed tau pays it once per budget. That makes the
    covered volume readable against the budget here, rather than only through
    log10_j_tau's overlap-blind sum.

    Only submits 'budget' jobs (output_csv=BUDGET_CSV) -- does not run
    `runner`."""
    dim             = 384
    num_vectors     = WORKLOAD_NUM_VECTORS
    n_folds         = BLOCKLIST_FOLDS
    k_percentages   = [5, 7.5, 10, 12.5, 15, 17.5, 20]  # ball budget k, as a percentage of |B|
    tau             = FIXED_TAU

    k_fractions = [p / 100.0 for p in k_percentages]

    for workload in blocklist_workloads(num_vectors, n_folds=n_folds):
        for fold, sel in enumerate(workload.selections):
            runner.submit(
                'budget',
                output_csv=BUDGET_CSV,
                dim=dim, dataset=workload.plot_key, seed=fold,
                k_fractions=k_fractions, tau=tau,
                vectors_reference=workload.rows(sel["reference"]),
                vectors_query=workload.query_vectors(sel),
            )


def save_budget(job_results: List[Tuple[str, Any]]) -> pd.DataFrame:
    """Extracts this evaluation's rows from a shared Runner's combined run()
    output (see generate_budget()). Tagged mode 'budget' (runner.budget_worker)
    rather than 'blocklist' (runner.blocklist_tau_worker) so the two
    evaluations' rows split cleanly."""
    return pd.DataFrame(rows_for_mode(job_results, 'budget'))


def plot_budget(df: pd.DataFrame) -> None:
    generate_budget_figures(df)


def generate_cap_quantization(runner: Runner, k_values_nominal: Optional[List[int]] = None) -> None:
    """CapCoverFilter.auto() per
    (dataset, k, seed, quantize) over CAP_QUANTIZATION_SWEEP (AUTO_CANDIDATES). Same k ladder, same
    fixed tau, same blocklist workloads/source-size split.

    Only submits 'cap_quantization' jobs -- does not run `runner`."""
    num_vectors      = WORKLOAD_NUM_VECTORS
    dim              = 384
    n_folds          = BLOCKLIST_FOLDS
    k_values_nominal = k_values_nominal or [2 ** i for i in range(11, 17)]
    tau              = FIXED_TAU

    workloads = blocklist_workloads(num_vectors, n_folds=n_folds)
    if not workloads:
        raise RuntimeError(
            "generate_cap_quantization: no usable workload for any corpus (see the errors "
            "above -- most likely WORKLOAD_NUM_VECTORS is too large for their "
            "duplicate clusters)")

    for workload in workloads:
        reference_len = len(workload.selections[0]["reference"])
        k_values = [k for k in k_values_nominal if k < reference_len]
        if not k_values:
            raise RuntimeError(
                f"No k in {k_values_nominal} is below the reference set of "
                f"{workload.name} ({reference_len} vectors). Raise "
                "WORKLOAD_NUM_VECTORS or lower k_values_nominal.")

        logger.info(f"{workload.name}: k_values={k_values} (|B|={reference_len}), tau={tau:.4f}")

        output_csv = cap_quantization_csv(workload.name)
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        for fold, sel in enumerate(workload.selections):
            vectors_construct = workload.rows(sel["reference"])
            vectors_query = workload.query_vectors(sel)
            actual_results = compute_static_actual_results(
                vectors_construct, vectors_query, tau)
            for k in k_values:
                for quantize in CAP_QUANTIZATION_SWEEP:
                    runner.submit(
                        'cap_quantization',
                        output_csv=output_csv,
                        dim=dim, dataset=workload.name,
                        k=k, quantize=quantize, tau=tau, seed=fold,
                        vectors_construct=vectors_construct, vectors_query=vectors_query,
                        actual_results=actual_results,
                    )


def save_cap_quantization(job_results: List[Tuple[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows_for_mode(job_results, 'cap_quantization'))


def plot_cap_quantization(df: pd.DataFrame) -> None:
    from plots.plot_cap_quantization import generate_quantization_figures as generate_cap_quantization_figures
    for dataset in sorted(df["dataset"].unique()):
        outdir = cap_quantization_plot_dir(str(dataset))
        logger.info(f"Plotting cap_quantization for {dataset} -> {outdir}")
        generate_cap_quantization_figures(df[df["dataset"] == dataset], outdir=outdir)


def generate_query_performance() -> pd.DataFrame:
    """Measures single-query vs. batched-query latency/throughput of a
    CapCoverFilter cover as a function of filter size k, across every
    dataset in DATASET_KEYS. One job per (dataset, k, seed), written to
    results/query_performance.csv."""
    num_vectors           = WORKLOAD_NUM_VECTORS
    dim                   = 384
    percentage_construct  = 20
    k_values              = [16, 32, 64, 128, 256, 512, 1024]
    query_tau             = 0.1
    seeds                 = [0, 1, 2]
    single_query_limit    = 2000
    output_csv            = "results/query_performance.csv"

    datasets = [d for d in DATASET_KEYS if os.path.exists(dataset_filenames[d])]
    missing = [d for d in DATASET_KEYS if d not in datasets]
    if missing:
        logger.warning(f"Skipping dataset(s) not found on disk: {', '.join(missing)}")
    if set(datasets) != set(DATASET_KEYS):
        raise RuntimeError(
            f"generate_query_performance needs all {len(DATASET_KEYS)} datasets "
            f"({', '.join(DATASET_KEYS)}) on disk; only found: {', '.join(datasets) or 'none'}"
        )

    # k must stay below a dataset's construct-set size, so clamp the sweep.
    construct_sizes = {d: len(load_embeds(d, num_vectors)[0]) * percentage_construct // 100 for d in datasets}
    min_construct = min(construct_sizes.values())
    usable_k_values = [k for k in k_values if k < min_construct]
    dropped_k = sorted(set(k_values) - set(usable_k_values))
    if dropped_k:
        logger.warning(
            f"Dropping k values >= smallest construct set ({min_construct} vectors, "
            f"from {min(construct_sizes, key=construct_sizes.get)}): {dropped_k}"
        )
    if not usable_k_values:
        raise RuntimeError(
            f"No k value in {k_values} is smaller than the smallest construct set "
            f"({min_construct} vectors). Raise num_vectors/percentage_construct or lower k_values."
        )
    k_values = usable_k_values

    runner = Runner(num_workers=MAX_WORKERS)

    for dataset in datasets:
        vectors = load_embeds(dataset, num_vectors)[0]
        n = len(vectors)  # a dataset may hold fewer than num_vectors
        num_construct = n * percentage_construct // 100

        construct_source = VectorSource(dataset, 0, num_construct)
        query_source = VectorSource(dataset, num_construct, n)

        for k in k_values:
            for seed in seeds:
                runner.submit(
                    'query_performance',
                    dim=dim, dataset=DATASET_KEYS[dataset],
                    k=k, seed=seed, query_tau=query_tau,
                    vectors_construct=construct_source, vectors_query=query_source,
                    single_query_limit=single_query_limit,
                )

    job_results = runner.run()
    logger.info("Query performance evaluation completed.")

    df = pd.DataFrame([row for job_rows in job_results for row in job_rows])

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info(f"Wrote {output_csv}")

    return df


def plot_query_performance(df: pd.DataFrame) -> None:
    generate_query_performance_figures(df)


def generate_scaling(runner: Runner) -> None:
    """Measures three CapCoverFilter throughput surfaces against filter
    size k: query (k x batch_size grid), insertion (same grid, timing add()
    instead), and construction (k x n, where indexed-set size itself
    varies).

    Reuses blocklist_workloads() -- the same corpora, position-based fold
    splits, reference set B and query stream as generate_blocklist and
    generate_quantization -- rather than cutting its own contiguous window
    over a separate dataset, so throughput is measured on the same data the
    correctness sweeps evaluate. Runs over every blocklist corpus and fold;
    plot_scaling pools all of them into one line per k (or n) value, which is
    valid because throughput is dataset-invariant here (Figure 4: "different
    datasets do not really matter") -- so k and the construction n's are
    bounded by the *smallest* corpus's reference set, keeping the same
    absolute values comparable, and thus poolable, across corpora. k itself
    is a fixed absolute ladder (2^11..2^16, one power of two per line color)
    rather than a fraction of the reference set, since "Filter Size (k)" is
    plotted and compared as an absolute count. Both sweeps stay at <=6 values
    because plot_scaling draws one line per value and defines 6 line colors.

    Only submits 'scaling_query'/'scaling_insert'/'scaling_construction'
    jobs -- does not run `runner`. Each corpus writes its own
    results/scaling/<dataset>/scaling_{query,insert,construction}.csv (see
    scaling_query_csv() et al.), mirroring cap_quantization_csv/CAP_QUANTIZATION_DIR,
    so the Runner writes all of them incrementally."""
    num_vectors      = WORKLOAD_NUM_VECTORS
    dim              = 384
    n_folds          = BLOCKLIST_FOLDS
    n_steps          = 6
    # Ball budget k, as absolute filter sizes (plot_scaling's "Filter Size (k)"
    # legend) rather than a fraction of a corpus's reference set: 2^11..2^16,
    # one power of two per line color (LINE_COLORS has exactly 6 entries).
    k_values_nominal = [2 ** i for i in range(11, 17)]
    # Powers of 4 from a single query up to 1024: spans three orders of
    # magnitude in 6 lines.
    batch_sizes      = [1, 4, 16, 64, 256, 1024]
    # Throughput is threshold-independent here (query() scores every ball
    # before comparing), so this only has to be a valid tau, not a tuned one;
    # 0.1 anyway, the operating point of every other evaluation.
    tau_mid          = 0.1
    # Fractions of each fold's blocklist query stream (see
    # BlocklistWorkload.query_vectors) consumed by the
    # query and insert sweeps. Both draw from the same stream since neither
    # cares about overlap: only timing is measured, not correctness.
    # insert_vector_fraction stays smaller: insert() mutates the cover, so
    # its cost per vector grows with the ball count and a full stream would
    # dominate the sweep's runtime for no extra resolution.
    query_vector_fraction  = 1.0
    insert_vector_fraction = 0.5

    workloads = blocklist_workloads(num_vectors, n_folds=n_folds)
    if not workloads:
        raise RuntimeError(
            "generate_scaling: no usable workload for any corpus (see the errors "
            "above -- most likely WORKLOAD_NUM_VECTORS is too large for their "
            "duplicate clusters)")

    # k and the construction n's are shared across corpora (plot_scaling
    # pools every dataset's rows into one line per value), so both are
    # bounded by the smallest corpus's reference set rather than computed
    # per corpus.
    reference_len = min(len(w.selections[0]["reference"]) for w in workloads)
    k_values = [k for k in k_values_nominal if k < reference_len]
    if not k_values:
        raise RuntimeError(
            f"No k in {k_values_nominal} is below the smallest corpus's reference "
            f"set ({reference_len} vectors). Raise WORKLOAD_NUM_VECTORS or lower "
            "k_values_nominal.")
    stream_sizes = sorted({max(1, reference_len // (2 ** i)) for i in range(n_steps)})

    # workload.stream_size is a function of num_vectors alone (see
    # build_blocklist_workload), so it is the same for every corpus.
    stream_len = workloads[0].stream_size
    n_query_vectors = max(1, round(stream_len * query_vector_fraction))
    n_insert_vectors = max(1, round(stream_len * insert_vector_fraction))

    # Drop batch sizes that no longer fit, rather than raising, so resizing
    # the sweep never requires hand-editing the lists too.
    usable_batch_sizes = sorted(b for b in batch_sizes if b < min(n_query_vectors, n_insert_vectors))
    dropped = sorted(set(batch_sizes) - set(usable_batch_sizes))
    if dropped:
        logger.warning(f"Dropping batch_size value(s) >= min(n_query_vectors, n_insert_vectors) "
                        f"({min(n_query_vectors, n_insert_vectors)}): {dropped}")
    if not usable_batch_sizes:
        raise RuntimeError(
            f"No batch_size in {batch_sizes} is below min(n_query_vectors={n_query_vectors}, "
            f"n_insert_vectors={n_insert_vectors}). Raise WORKLOAD_NUM_VECTORS or lower "
            "batch_sizes.")

    for workload in workloads:
        for fold, sel in enumerate(workload.selections):
            vectors_construct = workload.rows(sel["reference"])
            vectors_query = workload.query_vectors(sel)
            if len(vectors_query) < max(n_query_vectors, n_insert_vectors):
                raise RuntimeError(
                    f"{workload.name} fold {fold}: query stream holds {len(vectors_query)} "
                    f"vectors but needs {max(n_query_vectors, n_insert_vectors)}. Lower "
                    "query_vector_fraction/insert_vector_fraction or BLOCKLIST_QUERY_FRACTION.")

            for k in k_values:
                for batch_size in usable_batch_sizes:
                    runner.submit(
                        'scaling_query',
                        output_csv=scaling_query_csv(workload.name),
                        dim=dim, dataset=workload.name, k=k, batch_size=batch_size, seed=fold,
                        tau_mid=tau_mid,
                        vectors_construct=vectors_construct, vectors_query=vectors_query,
                        n_query_vectors=n_query_vectors,
                    )
                    runner.submit(
                        'scaling_insert',
                        output_csv=scaling_insert_csv(workload.name),
                        dim=dim, dataset=workload.name, k=k, batch_size=batch_size, seed=fold,
                        vectors_construct=vectors_construct, vectors_insert=vectors_query,
                        n_insert_vectors=n_insert_vectors,
                    )

            for k in k_values:
                for n in stream_sizes:
                    if k >= n or n > len(vectors_construct):
                        continue
                    runner.submit(
                        'scaling_construction',
                        output_csv=scaling_construction_csv(workload.name),
                        dim=dim, dataset=workload.name, k=k, n=n, seed=fold,
                        vectors_construct=vectors_construct,
                    )


def save_scaling(job_results: List[Tuple[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Extracts this evaluation's rows from a shared Runner's combined
    run() output (see generate_scaling()). Each of the three query/insert/
    construction job types was submitted under its own mode name AND its own
    output_csv, so the Runner already wrote all three results/scaling_*.csv
    files incrementally as their jobs completed -- this only rebuilds the
    three DataFrames for plot_scaling()."""
    df_query = pd.DataFrame(rows_for_mode(job_results, 'scaling_query'))
    df_insert = pd.DataFrame(rows_for_mode(job_results, 'scaling_insert'))
    df_construction = pd.DataFrame(rows_for_mode(job_results, 'scaling_construction'))
    return df_query, df_insert, df_construction


def plot_scaling(df_query: pd.DataFrame, df_insert: pd.DataFrame, df_construction: pd.DataFrame) -> None:
    generate_scaling_figures(df_query, df_insert, df_construction)


def generate_streaming(runner: Runner, construct_methods: Optional[List[str]] = None,
                       k_sweep: bool = True) -> None:
    """The paper's "dynamic evaluation" (Section 5.1/5.4, Figure 4): compares
    CapCoverFilter's insertion strategies (CAP_INSERT_STRATEGIES --
    cap_merge/nearest) via
    stream_rebuild_worker's policy="streaming" against a static rebuild at
    every checkpoint (policy="oracle"). Produces the streaming figure's
    panels: (a) FPR vs. position + the static-rebuild references, (b)
    threshold-expanded objective inflation vs. position -- at one fixed cap
    budget k, where only the stream position varies; (c) insertion
    throughput vs. filter size k.

    Strategies are compared at equal bytes, not equal k: max_memory is the
    static filter's footprint at k caps, and each strategy keeps as many
    caps as fit with its own bookkeeping at its peak over the stream (see
    InsertInterface.max_caps_for_budget; peak_memory_bytes is recorded).
    The position sweep runs once per construct method in construct_methods,
    which builds both the bootstrap and that run's oracle.

    Reuses blocklist_workloads() -- the same corpora, position-based fold
    splits, reference set B, query stream and fixed tau as generate_blocklist/
    generate_budget/generate_quantization/generate_equal_memory -- so the
    dynamic results are read on exactly the data the static ones are. Each
    fold's B is the stream: its first bootstrap_fraction is built with
    construct(), the way the static evaluations build all of it, and the rest
    arrives through insert() one vector at a time. B is kept in the fold's own
    order rather than reshuffled, so at the last checkpoint the oracle is the
    very filter generate_blocklist builds at the same k_fraction.

    Queries are the fold's query_vectors(),
    scored at every checkpoint against the distance-based ground truth
    compute_static_actual_results gives for the part of B that has arrived so
    far -- the definition every evaluation uses, and the one under which zero
    false negatives is a guarantee rather than a hope.
    The truth is computed here once per fold and shared by every job on it.

    Bootstrapping on half of B rather than on k vectors matters: construct()
    on exactly k points gives GonzalesConstruct one point per center and
    hence zero-radius balls at position 0 -- a degenerate starting point that
    saturates FPR within a few dozen insertions regardless of method.

    k is the blocklist workload's own budget, round(k_fraction * |B|), and
    |B| is trimmed to the same size on every corpus (see _fold_selections), so
    k and the checkpoint positions are identical across corpora, which is
    what makes plot_streaming's pooling meaningful.

    FNR=0 is asserted per checkpoint inside stream_rebuild_worker (raises,
    not recorded) -- a correctness requirement, not a plotted quantity.

    construct_methods narrows the position sweep to those constructors and
    k_sweep=False skips panel (c)'s k sweep (src/paper.py needs neither).

    Only submits 'stream_rebuild' jobs -- does not run `runner`. Each corpus
    writes its own results/streaming/<dataset>/streaming.csv (see
    streaming_csv()), mirroring cap_quantization_csv/CAP_QUANTIZATION_DIR."""
    num_vectors        = WORKLOAD_NUM_VECTORS
    dim                = 384
    n_folds            = BLOCKLIST_FOLDS
    methods            = ["nearest", "cap_merge"]
    # The position sweep runs once per constructor: each builds both the
    # streamed filter's bootstrap and its static-rebuild oracle, so every
    # method's gap can be read against either reference. Gonzalez is the
    # filter's own built-in fallback; runner.CAP_COVER_CONSTRUCT
    # (sphere_merge) is CapCoverFilter's actual default. The k sweep uses
    # CAP_COVER_CONSTRUCT only, unquantized like the position sweep.
    construct_methods  = construct_methods or ["gonzalez", CAP_COVER_CONSTRUCT]
    # The fixed absolute tau generate_budget/generate_quantization/
    # generate_equal_memory use, so every blocklist-workload evaluation sits
    # at the same operating point.
    tau                = FIXED_TAU
    k_fraction         = 0.1  # generate_blocklist's fixed ball budget, as a fraction of |B|
    bootstrap_fraction = 0.5  # of B built by construct() before insert() takes over
    n_checkpoints      = 12

    workloads = blocklist_workloads(num_vectors, n_folds=n_folds)
    if not workloads:
        raise RuntimeError(
            "generate_streaming: no usable workload for any corpus (see the errors "
            "above -- most likely WORKLOAD_NUM_VECTORS is too large for their "
            "duplicate clusters)")

    for workload in workloads:
        output_csv = streaming_csv(workload.name)
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        for fold, sel in enumerate(workload.selections):
            vectors_stream = workload.rows(sel["reference"])
            vectors_query = workload.query_vectors(sel)
            n = len(vectors_stream)
            k = min(max(1, round(k_fraction * n)), n)  # as runner._blocklist_cover_rows sizes it
            bootstrap_n = max(1, round(bootstrap_fraction * n))

            # construct() covers positions [0, bootstrap_n); the position
            # sweep only checkpoints the insert-only span after it.
            checkpoints = sorted(set(
                int(round(x)) for x in np.logspace(np.log10(bootstrap_n + 1), np.log10(n), num=n_checkpoints)
            ))
            actual_at_checkpoint = {
                c: compute_static_actual_results(vectors_stream[:c], vectors_query, tau)
                for c in checkpoints
            }
            logger.info(
                f"{workload.name} fold {fold}: |B|={n}, k={k}, bootstrap={bootstrap_n}, "
                f"duplicates among {len(vectors_query)} queries: "
                f"{sum(actual_at_checkpoint[checkpoints[0]])} at the first checkpoint, "
                f"{sum(actual_at_checkpoint[checkpoints[-1]])} at the last")

            # Each row is self-describing (dataset/method/sweep/metric/
            # fold_offset), so no positional reconstruction is needed despite
            # Runner's completion-order results.
            common = dict(
                output_csv=output_csv, dim=dim, dataset=workload.name, seed=fold, tau=tau, T=None,
                vectors_stream=vectors_stream, vectors_query=vectors_query,
                actual_at_checkpoint=actual_at_checkpoint,
                checkpoints=checkpoints, bootstrap_n=bootstrap_n, fold_offset=fold,
            )

            # --- panels (a)/(b): position sweep at the fixed budget k ---
            for construct_method in construct_methods:
                for method in methods:
                    runner.submit('stream_rebuild', k=k, policy="streaming", insertion_level=method,
                                  sweep="position", construct_method=construct_method, **common)
                runner.submit('stream_rebuild', k=k, policy="oracle", sweep="position",
                              construct_method=construct_method, **common)

            # --- panel (c): filter size (k) sweep for insertion throughput ---
            max_p = max(4, int(round(np.log2(k))))
            for method in methods if k_sweep else []:
                for k_val in [2 ** p for p in range(4, max_p + 1)]:
                    runner.submit('stream_rebuild', k=k_val, policy="streaming", insertion_level=method,
                                  sweep="k", construct_method=CAP_COVER_CONSTRUCT, **common)


def save_streaming(job_results: List[Tuple[str, Any]], default_k: int = 16) -> Tuple[pd.DataFrame, int]:
    """Rebuilds this evaluation's DataFrame from a shared Runner's run()
    output, plus k_fixed (the smallest k among 'position'-sweep rows,
    matching plot_all()'s own derivation)."""
    df = pd.DataFrame(rows_for_mode(job_results, "stream_rebuild"))
    if df.empty or "k" not in df.columns:
        return df, default_k

    if "sweep" in df.columns:
        pos_k = df.loc[df["sweep"] == "position", "k"].dropna()
        if not pos_k.empty:
            return df, int(pos_k.min())

    all_k = df["k"].dropna()
    k_fixed = int(all_k.min()) if not all_k.empty else default_k
    return df, k_fixed


def plot_streaming(df: pd.DataFrame, k_fixed: int) -> None:
    generate_streaming_figures(df, k_fixed=k_fixed)


# Construction-method comparison: one CSV per corpus (plot_construct_methods
# draws one figure per corpus), plus one per --scale run of
# generate_construct_methods_scale, whose "scale_" prefix keeps it out of
# the sweep's figures.
CONSTRUCT_METHODS_DIR = "results/construct_methods"
CONSTRUCT_METHODS_PLOT_DIR = "plots/construct_methods"


def construct_methods_csv(dataset: str) -> str:
    return os.path.join(CONSTRUCT_METHODS_DIR, f"{dataset}.csv")


def construct_methods_scale_csv(dataset: str, n: int) -> str:
    return os.path.join(CONSTRUCT_METHODS_DIR, f"scale_{dataset}_{n}.csv")


def generate_construct_methods(runner: Runner) -> None:
    """CapCoverFilter's construction methods (runner.CAP_CONSTRUCT_METHODS --
    only the 3 its package implements: gonzalez/minimax/sphere_merge, no
    split/merge) compared on the blocklist workload: the same construction
    and query sets as generate_budget/generate_quantization, with ground
    truth defined as distance to the reference set at tau. Only the
    construction method varies (see runner.construct_method_worker).

    One job per (corpus, fold, method, k), so rows reach
    construct_methods_csv(corpus) as each cover finishes, and the slow
    large-k covers don't serialize one method's whole sweep onto a single
    worker.

    Only submits 'construct_method' jobs -- does not run `runner`."""
    dim         = 384
    num_vectors = WORKLOAD_NUM_VECTORS
    n_folds     = BLOCKLIST_FOLDS
    tau         = FIXED_TAU
    # k as a fraction of |B|, log-spaced from 0.5% to 20%. The top of the
    # range is generate_budget's; the bottom extends it into the budgets where
    # every ball is wide, which is where the construction method matters most.
    k_fractions = [0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
    methods     = list(CAP_CONSTRUCT_METHODS)

    for workload in blocklist_workloads(num_vectors, n_folds=n_folds):
        output_csv = construct_methods_csv(workload.plot_key)
        for fold, sel in enumerate(workload.selections):
            vectors_construct = workload.rows(sel["reference"])
            vectors_query = workload.query_vectors(sel)
            actual_results = np.asarray(
                compute_static_actual_results(vectors_construct, vectors_query, tau), dtype=bool)
            n = len(vectors_construct)
            logger.info(f"{workload.name} fold {fold}: |B|={n}, {len(vectors_query)} queries, "
                        f"{int(actual_results.sum())} within tau={tau} of B")
            for method in methods:
                for kf in k_fractions:
                    runner.submit(
                        'construct_method',
                        output_csv=output_csv,
                        dim=dim, dataset=workload.plot_key, method=method,
                        k=min(max(1, round(kf * n)), n), k_fraction=kf, tau=tau, seed=fold,
                        vectors_construct=vectors_construct, vectors_query=vectors_query,
                        actual_results=actual_results,
                    )


def generate_construct_methods_scale(runner: Runner, dataset: str = "WikiAnswers", n: int = 100_000,
                                     k_values: Tuple[int, ...] = (1000, 5000)) -> None:
    """construct_filter time for every construction method on the first n
    rows of one corpus, past the blocklist workload's size (see
    runner.construct_method_scale_worker). Not part of the default sweep.

    Only submits 'construct_method_scale' jobs -- does not run `runner`."""
    if not os.path.exists(dataset_filenames[dataset]):
        raise FileNotFoundError(f"{dataset}: {dataset_filenames[dataset]} not on disk.")
    output_csv = construct_methods_scale_csv(dataset, n)
    for method in CAP_CONSTRUCT_METHODS:
        for k in k_values:
            runner.submit('construct_method_scale', output_csv=output_csv,
                          dim=384, dataset=dataset, n=n, method=method, k=k)


def save_construct_methods(job_results: List[Tuple[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows_for_mode(job_results, 'construct_method'))


def plot_construct_methods(df: pd.DataFrame) -> None:
    generate_construct_methods_figures(df, CONSTRUCT_METHODS_PLOT_DIR)


def generate_and_plot_all() -> None:
    """Submit-all, run-all, then plot-all: every generate_* function below
    only submits jobs to one shared Runner instead of running its own, so
    the combined job list executes as a single parallel batch and
    runner.shuffle() can interleave fast and slow jobs across evaluations
    rather than one worker draining a whole slow evaluation alone.

    Each generate_ function tags its jobs with its own output_csv, so the
    Runner writes each results/*.csv file incrementally as that
    evaluation's jobs complete, not after the whole batch finishes -- a
    crash mid-run only loses whichever jobs hadn't completed yet. There's
    no separate save-all step: save_* functions still exist for a caller
    that wants rows back as a DataFrame directly, but aren't needed here.

    Each generate_ call is independently try/except-guarded so one
    evaluation failing to submit doesn't take down the others. A failure
    inside runner.run() itself aborts the whole batch, but every job that
    did complete beforehand is still on disk."""
    runner = Runner(num_workers=MAX_WORKERS)

    # generate_construct_methods_scale is disabled.
    for generate in (
        generate_blocklist,
        generate_budget,
        generate_cap_quantization,
        generate_streaming,
        generate_construct_methods,
        generate_scaling,
        generate_equal_memory,
        generate_cap_cover_system_tau,
    ):
        try:
            generate(runner)
        except Exception:
            logger.exception(f"{generate.__name__} failed to submit")
    runner.shuffle()
    runner.run()
    logger.info("All submitted jobs completed.")

    plot_all()


def plot_all() -> None:
    """Re-renders every figure from data already persisted to disk, without
    repeating the evaluation sweeps -- handy for iterating on plot styling
    alone. Re-derives whatever generate-time metadata (k, k_fixed) each
    plot_ needs from the loaded data. Requires at least one prior
    generate_and_plot_all() (or generate_X + runner.run()) to have produced
    those files."""
    try:
        df = pd.read_csv(CAP_COVER_SYSTEM_TAU_CSV)
        plot_cap_cover_system_tau(df)
    except Exception:
        logger.exception("An error occurred")

    try:
        df = pd.read_csv(BLOCKLIST_CSV)
        plot_blocklist(df)
    except Exception:
        logger.exception("An error occurred")

    try:
        df = pd.read_csv(BUDGET_CSV)
        plot_budget(df)
    except Exception:
        logger.exception("An error occurred")

    try:
        plot_construct_methods(load_construct_methods(CONSTRUCT_METHODS_DIR))
    except Exception:
        logger.exception("An error occurred")

    try:
        # keep_default_na=False: stage/config strings are read verbatim; pandas
        # would otherwise read back as NaN (see plot_quantization.py::load_data()).
        frames = []
        for name in BLOCKLIST_DATASETS:
            path = cap_quantization_csv(name)
            if os.path.exists(path):
                # keep_default_na=False: stage/config strings are read verbatim;
                # pandas would otherwise turn them into NaN.
                frames.append(pd.read_csv(path, keep_default_na=False))
        if frames:
            plot_cap_quantization(pd.concat(frames, ignore_index=True))
    except Exception:
        logger.exception("An error occurred")

    try:
        frames = []
        for name in BLOCKLIST_DATASETS:
            path = equal_memory_csv(name)
            if os.path.exists(path):
                frames.append(pd.read_csv(path))
        if frames:
            plot_equal_memory(pd.concat(frames, ignore_index=True).to_dict("records"))
    except Exception:
        logger.exception("An error occurred")

    try:
        df = pd.read_csv("results/query_performance.csv")
        plot_query_performance(df)
    except Exception:
        logger.exception("An error occurred")

    try:
        def _concat_scaling(csv_fn):
            frames = [pd.read_csv(csv_fn(name)) for name in BLOCKLIST_DATASETS if os.path.exists(csv_fn(name))]
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        df_query = _concat_scaling(scaling_query_csv)
        df_insert = _concat_scaling(scaling_insert_csv)
        df_construction = _concat_scaling(scaling_construction_csv)
        if not (df_query.empty or df_insert.empty or df_construction.empty):
            plot_scaling(df_query, df_insert, df_construction)
    except Exception:
        logger.exception("An error occurred")

    try:
        frames = [pd.read_csv(streaming_csv(name)) for name in BLOCKLIST_DATASETS
                 if os.path.exists(streaming_csv(name))]
        if frames:
            df = pd.concat(frames, ignore_index=True)
            k_fixed = int(df.loc[df["sweep"] == "position", "k"].iloc[0])
            plot_streaming(df, k_fixed=k_fixed)
    except Exception:
        logger.exception("An error occurred")


if __name__ == "__main__":
    # generate_and_plot_all()
    plot_all()  # alternative: re-render figures from an existing results/*.csv run