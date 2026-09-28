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
import time
import inspect
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

faiss.omp_set_num_threads(1)  # belt-and-suspenders alongside the env vars above

from logging_config import configure_logging
from runner import Runner, QUANTIZATION_SWEEP, system_tau_ground_truth_worker
from plots.plot_quantization import generate_quantization_figures
from plots.plot_query_performance import generate_query_performance_figures
from plots.plot_equal_memory import generate_equal_memory_figures
from plots.plot_scaling import generate_scaling_figures
from plots.plot_streaming import generate_streaming_figures
from plots.plot_system_tau import generate_system_tau_figures
from plots.plot_blocklist import generate_blocklist_figures
from plots.plot_budget import generate_budget_figures
from plots.plot_common import DATASETS, DATASET_LABELS
from typing import Any, Dict, List, Optional, Tuple
from filters.ball_cover.ball_cover_filter import BallCoverFilter, Quantize
from filters.dsh_filter import DSHFilter
from filters.hnsw_filter import HNSWFilter
from filters.ivf_filter import IVFFilter
from filters.quantized_vector_filter import QuantizedVectorFilter
from filters.mlp_filter import MLPFilter
from util.dataset_paths import dataset_filenames, HDF5_DATASET_KEY
from util.fetch_datasets import load_embeds, load_meta
from vector_source import VectorSource

configure_logging()
logger = logging.getLogger(__name__)

MAX_WORKERS = 20
DATASET = 'ELI5'

# Vectors/dataset, split 80% construction / 20% queries. One knob for every
# evaluation: the ground-truth FPR/FNR ones (system_tau, quantization,
# equal_memory, streaming, blocklist) and the throughput-only sweeps (scaling,
# query_performance), which used to read a separate STREAM_SIZE.
#
# Capped well below the paper's nominal 100K by the blocklist workload, whose
# reference set must come from duplicate clusters: CC-News holds only 31,696
# usable vectors in clusters of >=3 at eps=0.4, so anything above ~36,018
# drops that corpus and silently reduces the comparison to two. 2^15 leaves
# ~10% headroom there; system_tau's non-wrapping folds need 1.8N = 58,982 rows
# per dataset, which its shortest (HotpotQA, 97,852) clears comfortably.
WORKLOAD_NUM_VECTORS = 10_000

# generate_blocklist() knobs. Cluster discovery scans the whole file on disk.
# Multiplicity is a property of the pool, not of the corpus, so a partial pool
# undercounts it: a cluster of true size m yields only ~m * pool/corpus
# members, which is why a 50k pool over 250k CC-News rows reported a median
# cluster of 3 for a corpus whose syndication clusters are much larger.
BLOCKLIST_POOL_SIZE = None  # None = use every row on disk (cluster DISCOVERY only)
BLOCKLIST_FOLDS = 5
# The blocklist's query/reference split of WORKLOAD_NUM_VECTORS, mirroring
# generate_system_tau: one total size, cut into a q query part and a (1-q)
# reference part, with k = k_fraction * |B|. The blocklist deliberately has no
# size knob of its own -- WORKLOAD_NUM_VECTORS is already the single number
# that sizes every evaluation, so the two knobs that matter (total size,
# k_fraction) mean the same thing here as in system_tau:
#
#     stream (queries) = q * N          |B| (reference) = (1 - q) * N
#
# This is deliberately NOT the pool size. The pool is every row on disk because
# cluster multiplicity is a property of the corpus and a partial pool
# undercounts it; |B| is a property of the WORKLOAD and has to be chosen. Left
# coupled, a whole-file pool gave WildChat |B|=131,776 -- 13% of its corpus,
# which is the symmetric regime this workload exists to avoid (measured: FPR
# 0.016 at |B|=500, 0.051 at 2,000, 0.326 at 10,000) -- and made |B| differ 8x
# between corpora purely because one file is bigger.
BLOCKLIST_QUERY_FRACTION = 0.2

# 0.4 L2 on unit-normalised vectors = cosine 0.92, the conventional
# paraphrase threshold. Inspected rather than assumed: same-cluster
# WikiAnswers pairs at d~0.4 are true rewordings ("How many calories on 10
# strawberries?" / "How many calories in some strawberries?"), while pairs
# at d~0.75 are merely on-topic ("Why was the schlefan plan important to
# britain?" / "Why was the battle of britain important?"). Agreement with
# WikiAnswers' editor labels actually peaks near 0.75, but those labels are
# loose -- they group topically related questions -- so fitting them would
# pull eps into the non-paraphrase range.
BLOCKLIST_EPS = 0.4
BLOCKLIST_MIN_CLUSTER = 3


# Cluster labels are a pure function of (dataset, pool size, eps, and the
# config), so they are computed once and cached here rather than recomputed
# every run -- the same pattern generate_system_tau uses for its ground truth.
BLOCKLIST_CLUSTER_DIR = "results/blocklist_clusters"

# Registry name -> canonical dataset key, one source of truth via plot_common.
DATASET_KEYS = {DATASET_LABELS[d]: d for d in DATASETS}

# Output CSV path per evaluation, shared by each generate_*'s
# runner.submit(output_csv=...), each save_*'s df, and plot_all()'s reload.
SYSTEM_TAU_CSV = "results/system_tau.csv"
# generate_quantization runs once per blocklist corpus and keeps each run in
# its own folder. Separate rather than one combined file because
# plot_quantization asserts a single dataset: its panels already spend both
# dimensions on memory and quantizer, leaving nowhere to put a dataset.
QUANTIZATION_DIR = "results/quantization"
QUANTIZATION_PLOT_DIR = "plots/quantization"


def quantization_csv(dataset: str) -> str:
    return os.path.join(QUANTIZATION_DIR, dataset, "quantization.csv")


def quantization_plot_dir(dataset: str) -> str:
    return os.path.join(QUANTIZATION_PLOT_DIR, dataset)

# generate_scaling runs once per blocklist corpus and keeps each run in its
# own folder, mirroring quantization_csv/QUANTIZATION_DIR above: throughput
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
# its own folder, mirroring quantization_csv/QUANTIZATION_DIR above. Plotted
# the same way too (see plot_equal_memory): one figure set per corpus, in its
# own folder, rather than pooled into a single cross-corpus line per filter.
EQUAL_MEMORY_DIR = "results/equal_memory"
EQUAL_MEMORY_PLOT_DIR = "plots/equal_memory"


def equal_memory_csv(dataset: str) -> str:
    return os.path.join(EQUAL_MEMORY_DIR, dataset, "equal_memory.csv")


def equal_memory_plot_dir(dataset: str) -> str:
    return os.path.join(EQUAL_MEMORY_PLOT_DIR, dataset)


# generate_streaming runs once per blocklist corpus and keeps each run in its
# own folder, mirroring quantization_csv/QUANTIZATION_DIR above; plot_streaming
# pools every dataset's rows the same way plot_scaling does (see its
# docstring: "this figure is single-dataset" refers to color channel reuse
# for method, not to a restriction on how many datasets feed it).
STREAMING_DIR = "results/streaming"


def streaming_csv(dataset: str) -> str:
    return os.path.join(STREAMING_DIR, dataset, "streaming.csv")

# generate_equal_memory()'s own comparison set: excludes DSBFFilter/
# CoveringLSHFallbackFilter/LSBFFilter, which degenerate to ~100% duplicate
# predictions at realistic construct-set sizes (still used elsewhere).
# BallCoverFilter is submitted separately, once per quantize variant (see
# generate_equal_memory's ball_cover_quantizations), rather than listed here
# as a single entry -- unlike every other filter here, it has a `quantize`
# dial worth comparing multiple settings of side by side.
equal_memory_filters = [HNSWFilter, IVFFilter, DSHFilter, QuantizedVectorFilter, MLPFilter]

def calculate_nn_distance_threshold(
    construct_vectors: np.ndarray,
    percentage: float,
    sample_size: int = 5_000,
    seed: int = 42,
) -> float:
    """The `percentage`-th percentile of nearest-*other*-neighbor distance
    within a sample of construct_vectors -- ties the threshold to the actual
    test compute_static_actual_results performs, unlike a random-pairwise
    percentile: at large construct-set sizes a point's nearest neighbor is
    much closer than a random pair, so a random-pairwise percentile silently
    pushes the duplicate rate toward 100%, collapsing FPR/FNR to degenerate
    values."""
    n = len(construct_vectors)
    if n < 2:
        return 0.0

    vecs = np.ascontiguousarray(construct_vectors, dtype=np.float32)
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(n, size=sample_size, replace=False) if sample_size < n else np.arange(n)

    index = faiss.IndexFlatL2(vecs.shape[1])
    index.add(vecs)
    # k=2: rank 0 is the point itself (distance 0); rank 1 is the *other* nearest neighbor.
    sq_dist, _ = index.search(vecs[sample_idx], 2)
    nn_dist = np.sqrt(np.clip(sq_dist[:, 1], 0, None))
    return float(np.percentile(nn_dist, percentage))


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

    Stratified on `actual`, and shuffled within each stratum, because
    BlocklistWorkload.query_vectors hands back its three query classes
    *concatenated in order* -- positives, then hard negatives, then easy
    negatives. A prefix slice of that is not a sample of it. At the sweep's
    own settings (BLOCKLIST_QUERY_FRACTION's frac_positive=0.2 and
    calibration_fraction=0.2) the prefix was exactly the positive block, so:

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
            "too tight for this corpus's duplicate geometry -- see "
            "BlocklistWorkload.cluster_membership.")

    return (stream[calibration_idx], stream[measurement_idx],
            labels[calibration_idx].tolist(), labels[measurement_idx].tolist())


def generate_equal_memory(runner: Runner) -> None:
    """Equal-memory (SOTA) comparison across equal_memory_filters. Sweeps
    count_memory_steps budgets from 10% to 100% of raw storage (storing the
    construct set as a plain list of float32 vectors, dim*4 bytes each) in
    equal-sized steps -- see raw_storage_bytes below. For each budget, every
    filter's best config is chosen on a held-out calibration set (see
    runner.py::_build_calibrated_filter), then FPR/FNR/latency/throughput are
    measured on a separate held-out query set.

    Reuses blocklist_workloads() -- the same corpora, cluster-based fold
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

    Runs over every blocklist corpus and fold; plot_equal_memory pools all of
    them into one line per filter at each memory budget, so -- like
    generate_scaling -- the budget grid is bounded by the *smallest* corpus's
    reference set, keeping the same absolute budgets comparable, and thus
    poolable, across corpora.

    Only submits 'equal_memory' jobs -- does not run `runner`. Each corpus
    writes its own results/equal_memory/<dataset>/equal_memory.csv (see
    equal_memory_csv()), mirroring quantization_csv/QUANTIZATION_DIR."""
    num_vectors            = WORKLOAD_NUM_VECTORS
    dim                    = 384
    n_folds                = 1
    tau                    = 0.1
    # Log-spaced, not linear: the baselines' feasibility floors span more
    # than a decade below this range's top (IVF-PQ ~0.6%, quantized-point-set
    # RaBitQ ~3.6%, of raw storage at n=100K/dim=384 -- HNSW's ~30% floor
    # sits above max_memory_fraction entirely, so it is expected to stay
    # infeasible throughout), so a linear sweep would waste points at the top
    # and barely resolve the first two floors. 17 steps over this range's
    # 2 decades is ~8 points/decade, matching the resolution the wider
    # sweep used.
    count_memory_steps     = 17
    # Sweep bounds, as a fraction of raw_storage_bytes (see below).
    min_memory_fraction    = 0.001
    max_memory_fraction    = 0.10
    # Fraction of each fold's blocklist query stream spent calibrating each
    # filter's config dial; the rest measures FPR/FNR/latency/throughput.
    calibration_fraction   = 0.2
    # BallCoverFilter is the one entry with a `quantize` dial worth comparing
    # multiple settings of (see filters.ball_cover.ball_cover_filter.Quantize)
    # -- submitted once per variant below, each becoming its own labelled
    # line ("BallCoverFilter (TQ4)" etc. -- see runner.equal_memory_worker's
    # Config naming) rather than picking one variant to stand in for it.
    ball_cover_quantizations = [Quantize.TQ3, Quantize.TQ4, Quantize.TQ5, Quantize.TQ8, Quantize.FP16]

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
            # into the two slices -- a prefix cut of query_vectors' ordered
            # (positive, hard, easy) concatenation put every positive in
            # calibration and none in measurement. See
            # stratified_calibration_split for what that did to the numbers.
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
                for filter_cls in equal_memory_filters:
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
                for quantize in ball_cover_quantizations:
                    runner.submit(
                        'equal_memory',
                        output_csv=output_csv,
                        filter_cls=BallCoverFilter,
                        dim=dim, distance_threshold=tau,
                        memory_budget=int(memory_budget),
                        vectors_construct=vectors_construct, vectors_calibration=vectors_calibration,
                        vectors_query=vectors_query,
                        actual_calibration=actual_calibration, actual_query=actual_query,
                        dataset=workload.name,
                        extra_kwargs={"quantize": quantize},
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


def generate_system_tau(runner: Runner) -> None:
    """End-to-end filtering benefit (paper Section 5.2, Figure 2): one
    BallCoverFilter cover per (dataset, fold), queried at every tau in a
    fixed sweep without ever rebuilding. Ball budget k is a fixed fraction
    (k_fraction, 5%) of each fold's construct-set size rather than the
    paper's fixed k=400.

    Implements the paper's Workloads protocol (Section 5.1): 5 datasets x
    k_folds=5 non-wrapping folds via advancing_fold_starts (80%
    construct/20% query per fold); each dataset needs enough vectors on
    disk for that or it raises. plot_system_tau reports the mean +/- std
    over the 5 folds, matching the paper's "5 runs" error bars.

    Ground truth (nearest-construct-set distance per query) doesn't depend
    on k, so it's computed once per (dataset, fold) in parallel and cached
    to results/system_tau_ground_truth/, not redone per (dataset, fold, k).

    Only submits 'system_tau' jobs (output_csv=SYSTEM_TAU_CSV) -- does not
    run `runner`. The ground-truth precomputation above still runs eagerly,
    in its own dedicated ProcessPoolExecutor."""
    num_vectors      = WORKLOAD_NUM_VECTORS
    dim              = 384
    k_folds          = 5
    k_fraction       = 0.1
    tau_values       = [float(t) for t in np.logspace(-2, -0.5, num=10)]
    ground_truth_dir = "results/system_tau_ground_truth"

    datasets = [d for d in DATASET_KEYS if os.path.exists(dataset_filenames[d])]
    missing = [d for d in DATASET_KEYS if d not in datasets]
    if missing:
        logger.warning(f"Skipping dataset(s) not found on disk: {', '.join(missing)}")
    if set(datasets) != set(DATASET_KEYS):
        raise RuntimeError(
            f"generate_system_tau needs all {len(DATASET_KEYS)} datasets "
            f"({', '.join(DATASET_KEYS)}) on disk; only found: {', '.join(datasets) or 'none'}"
        )

    logger.info(f"Ball budget fixed as a fraction of the construct set: {k_fraction}")
    logger.info(f"Fixed tau sweep (shared across datasets): {tau_values}")
    logger.info(f"K-fold cross-validation: {k_folds} folds, non-wrapping contiguous 80/20 spans")

    query_len = num_vectors // k_folds
    construct_len = num_vectors - query_len

    # (dataset, fold, construct_source, query_source), reused by both passes
    # below -- lazy VectorSource pointers, no array materialized here.
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

    # Ground truth (nearest-construct-set distance per query vector): load
    # whatever's already cached on disk, compute the rest in parallel.
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
            'system_tau',
            output_csv=SYSTEM_TAU_CSV,
            dim=dim, dataset=DATASET_KEYS[dataset],
            k=k, tau_values=tau_values, seed=fold,
            vectors_construct=construct_source,
            vectors_query=query_source,
            actual_results_by_tau=actual_results_by_tau,
            )


def save_system_tau(job_results: List[Tuple[str, Any]]) -> pd.DataFrame:
    """Extracts this evaluation's rows from a shared Runner's combined
    run() output (see generate_system_tau()). The Runner itself already
    wrote results/system_tau.csv incrementally (see generate_system_tau()'s
    output_csv=), so this only rebuilds the DataFrame for plot_system_tau()."""
    return pd.DataFrame(rows_for_mode(job_results, 'system_tau'))


def plot_system_tau(df: pd.DataFrame) -> None:
    generate_system_tau_figures(df)


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
    # registry name -> (plot key, label column in the HDF5 meta group or None)
    # WildChat: clusters of repeated prompts, multiplicity up to 500+, but
    # duplicates are mostly exact reposts. No labels -- clusters are discovered
    # geometrically.
    "WildChat": ("wildchat", None),
    # CC-News: cross-outlet wire syndication. Same-domain repeats are templated
    # boilerplate, not syndication, so they are dropped during discovery.
    "CCNews": ("ccnews", None),
    # WikiAnswers: paraphrase questions. Clustered geometrically like the other
    # two rather than by its set_id column, so all three corpora share one
    # cluster definition and a difference between them is a difference in
    # geometry, not in how their clusters were defined.
    #
    # The labels are still the better ground truth in one respect -- they do
    # not come from the same metric the filter thresholds -- but they are loose
    # enough that using them would not buy that: editors group "Why was the
    # schlefan plan important to britain?" with "Why was the battle of britain
    # important?", pairs at d~0.75 that are on-topic rather than duplicates.
    # set_id stays in the metadata and is what eps=0.4 was calibrated against.
    "WikiAnswers": ("wikianswers", None),
    # Quora: is_duplicate question pairs. Its set_id column exists (see
    # build_quora) but the on-disk file predates _write_meta, so load_meta
    # returns {} for it -- geometric discovery is the only option available
    # without a full re-embed, and it doesn't need the label anyway: eps=0.4
    # alone found 48,111 clusters>=3 here (median 4, ~259k usable vectors),
    # the same shared cluster definition every other entry here uses.
    "Quora": ("quora", None),
    # ChatGPT-paraphrases: a source text plus 5 LLM-generated paraphrases,
    # one cluster of 6 by construction (see build_chatgpt_paraphrases).
    # Considered and rejected first: TaPaCo and StackExchange's official
    # duplicate-question pairs, both community "same topic" labels that
    # turned out geometrically sparse once actually clustered (~10-26k
    # usable vectors from a comparable sample) -- "duplicate" there does not
    # imply nearby embeddings the way a paraphrase does. eps=0.4 on this
    # corpus instead recovered a median cluster size of 5 out of the native
    # 6, ~166k usable vectors from a 360k-row sample alone.
    "ChatGPTParaphrases": ("chatgpt_paraphrases", None),
}


def _group_by_label(labels: np.ndarray, wanted: np.ndarray) -> Dict[int, np.ndarray]:
    """{label: row indices} for the wanted labels, in one O(n log n) pass."""
    order = np.argsort(labels, kind="stable")
    sorted_labels = labels[order]
    starts = np.searchsorted(sorted_labels, wanted, side="left")
    stops = np.searchsorted(sorted_labels, wanted, side="right")
    return {int(c): order[a:b] for c, a, b in zip(wanted, starts, stops)}


def _selection_is_usable(selections: List[Dict[str, np.ndarray]]) -> bool:
    """Every fold needs a reference set, something to find, and hard negatives
    to be wrong about. Easy negatives may legitimately be absent (a corpus can
    have no singletons), so they are not required."""
    return bool(selections) and all(
        len(sel["reference"]) and len(sel["positive"]) and len(sel["hard"])
        for sel in selections
    )


def _fold_selections(dataset: str, labels: np.ndarray, dup_clusters: np.ndarray,
                     singletons: np.ndarray, members_by_cluster: Dict[int, np.ndarray],
                     n_folds: int, reference_target: int, n_pos: int, n_hard: int,
                     n_easy: int, pool_n: int, eps: float, stream_size: int,
                     query_fraction: float, xgroup: bool,
                     cache_dir: str = BLOCKLIST_CLUSTER_DIR) -> List[Dict[str, np.ndarray]]:
    """Per-fold reference/positive/hard/easy ROW INDICES, cached to disk.

    Indices rather than vectors: they are the reproducible part, they are three
    orders of magnitude smaller than the embeddings they select, and they let a
    fold read only its own rows (see _read_rows) instead of holding the pool in
    memory. Stored per (dataset, pool size, eps, stream size, query fraction,
    folds), all of which are in the filename, so changing a knob misses the
    cache rather than reusing a stale split.

    Clusters are taken whole until the reference set reaches its target, so the
    multiplicity that bounds compression survives the sizing; the clusters not
    taken are withheld and supply the hard negatives.
    """
    os.makedirs(cache_dir, exist_ok=True)
    tag = "_xgroup" if xgroup else ""
    path = os.path.join(
        cache_dir,
        f"{dataset}_{pool_n}_eps{eps:g}_stream{stream_size}_q{query_fraction:g}"
        f"_folds{n_folds}{tag}_selection.npz")
    if os.path.exists(path):
        with np.load(path) as z:
            cached = [{key: z[f"{f}_{key}"] for key in ("reference", "positive", "hard", "easy")}
                      for f in range(n_folds)]
        if _selection_is_usable(cached):
            logger.info(f"{dataset}: loaded cached fold selection from {path}")
            return cached
        # A run whose sizing was infeasible used to write its empty splits here
        # and every later run replayed them, skipping every fold while
        # reporting nothing worse than "loaded cached fold selection". Drop it
        # and rebuild rather than trusting the file's existence.
        logger.warning(f"{dataset}: cached fold selection at {path} has empty splits; recomputing")
        os.remove(path)

    def take(rng, pool_idx, n):
        pool_idx = np.asarray(pool_idx, dtype=np.int64)
        if len(pool_idx) == 0 or n <= 0:
            return pool_idx[:0]
        return rng.choice(pool_idx, size=min(n, len(pool_idx)), replace=False)

    selections = []
    for fold in range(n_folds):
        rng = np.random.default_rng(fold)
        shuffled = rng.permutation(dup_clusters)

        reference, positives, n_taken = [], [], 0
        for c in shuffled:
            if len(reference) >= reference_target:
                break
            members = members_by_cluster[int(c)].copy()
            rng.shuffle(members)
            cut = max(1, int(0.8 * len(members)))
            reference.extend(members[:cut].tolist())
            positives.extend(members[cut:].tolist())
            n_taken += 1
        # Trimmed to exactly the target so |B| matches across corpora and the
        # three stay comparable at a given k_fraction.
        reference = np.asarray(reference[:reference_target], dtype=np.int64)
        hard_pool = np.concatenate(
            [members_by_cluster[int(c)] for c in shuffled[n_taken:]]
        ) if n_taken < len(shuffled) else np.empty(0, dtype=np.int64)

        selections.append({
            "reference": reference,
            "positive": take(rng, positives, n_pos),
            "hard": take(rng, hard_pool, n_hard),
            "easy": take(rng, singletons, n_easy),
        })

    if not _selection_is_usable(selections):
        # Not cached on purpose: an unusable split is a sizing failure, and
        # persisting it would make the failure sticky across runs.
        logger.error(
            f"{dataset}: fold selection is empty (reference/positive/hard); not caching. "
            f"Reduce WORKLOAD_NUM_VECTORS or loosen eps/min_cluster."
        )
        return selections

    np.savez_compressed(path, **{f"{f}_{k}": v for f, sel in enumerate(selections)
                                 for k, v in sel.items()})
    logger.info(f"{dataset}: wrote fold selection to {path}")
    return selections


def _cluster_cache_path(dataset: str, n: int, eps: float, xgroup: bool,
                        cache_dir: str = BLOCKLIST_CLUSTER_DIR) -> str:
    tag = "_xgroup" if xgroup else ""
    return os.path.join(cache_dir, f"{dataset}_{n}_eps{eps:g}_seedball{tag}.npy")


def _cluster_cache_exists(dataset: str, n: int, eps: float, xgroup: bool) -> bool:
    return os.path.exists(_cluster_cache_path(dataset, n, eps, xgroup))


def _read_rows(dataset: str, idx: np.ndarray) -> np.ndarray:
    """Just the requested rows of a dataset's embeddings, straight from HDF5.

    h5py fancy indexing needs a sorted, unique selection, so the request is
    sorted for the read and then mapped back to the caller's order.
    """
    order = np.argsort(idx, kind="stable")
    sorted_idx = idx[order]
    uniq, inverse = np.unique(sorted_idx, return_inverse=True)
    with h5py.File(dataset_filenames[dataset], "r") as f:
        block = f[HDF5_DATASET_KEY][uniq]
    out = np.empty((len(idx), block.shape[1]), dtype=np.float32)
    out[order] = block[inverse]
    return np.ascontiguousarray(out)


def _ivf_params(n: int) -> Tuple[int, int]:
    """(nlist, nprobe) for a pool of n vectors.

    nlist = 4*sqrt(n) is the usual faiss starting point, clamped so each cell
    still holds enough points to train on. nprobe=8 was chosen by measurement:
    on a 250k WildChat pool, edge recall at eps=0.25 was 0.9906 at nprobe 8, 16
    and 32 alike, so probing harder buys nothing -- near duplicates are nearly
    identical vectors and land in the same Voronoi cell. HNSW was measured too
    and lost on both axes (0.865 recall at efSearch=64, 0.888 at 128, no
    faster). The approximation is free in practice: on a 60k pool the resulting
    partition was *identical* to one built from exact search (adjusted Rand
    index 1.000000).
    """
    nlist = int(4 * np.sqrt(max(n, 1)))
    nlist = max(1, min(nlist, max(1, n // 39), 65_536))
    return nlist, min(8, nlist)


def _eps_neighbors(vectors: np.ndarray, eps: float,
                   exclude_same: Optional[np.ndarray] = None,
                   chunk: int = 50_000) -> csr_matrix:
    """Adjacency of every pair within eps, as a CSR matrix.

    Uses range_search, not a top-k search: a fixed k silently caps how large a
    cluster can be, which is not a tunable but a measurement error. At knn=64
    every WildChat cluster came back as exactly 65 members, and at knn=256 as
    exactly 257, at every eps -- the retrieval cap, not the corpus. WildChat's
    99th-percentile neighbourhood at eps=0.4 is 405 and its largest is over
    1400, so no fixed k is safe.

    Chunked over queries because range_search output is unbounded; only the
    surviving pairs are retained.

    exclude_same: optional per-vector group key (CC-News' publisher domain).
    Edges inside a group are dropped, so a site reposting its own boilerplate
    does not form a cluster no filter should get credit for finding.
    """
    n = len(vectors)
    nlist, nprobe = _ivf_params(n)
    quantizer = faiss.IndexFlatL2(vectors.shape[1])
    index = faiss.IndexIVFFlat(quantizer, vectors.shape[1], nlist, faiss.METRIC_L2)
    rng = np.random.default_rng(0)
    train_n = min(n, 200_000)
    index.train(np.ascontiguousarray(vectors if train_n == n else vectors[rng.choice(n, train_n, replace=False)]))
    index.add(vectors)
    index.nprobe = nprobe
    logger.info(f"  IVF nlist={nlist} nprobe={nprobe}, range search at eps={eps} over {n} vectors...")

    rows_out, cols_out = [], []
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        # faiss METRIC_L2 works in squared distance.
        lims, _, nbr = index.range_search(vectors[start:stop], float(eps) ** 2)
        # faiss returns lims as uint64; np.repeat refuses unsigned counts.
        counts = np.diff(lims).astype(np.int64)
        rows = np.repeat(np.arange(start, stop, dtype=np.int64), counts)
        cols = nbr.astype(np.int64)
        keep = cols != rows
        if exclude_same is not None:
            keep &= exclude_same[rows] != exclude_same[cols]
        rows_out.append(rows[keep])
        cols_out.append(cols[keep])

    rows = np.concatenate(rows_out) if rows_out else np.empty(0, dtype=np.int64)
    cols = np.concatenate(cols_out) if cols_out else np.empty(0, dtype=np.int64)
    logger.info(f"  {len(rows)} edges within eps={eps}")
    return csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n, n))


def _discover_clusters(vectors: np.ndarray, eps: float,
                       exclude_same: Optional[np.ndarray] = None) -> np.ndarray:
    """Cluster id per vector, by greedy seed-ball covering: the highest-degree
    unassigned point becomes a seed and claims every unassigned point within
    eps of it.

    Every cluster is therefore contained in one ball of radius eps -- which is
    both what bounds it and what makes it the right unit here, since the filter
    stores exactly one ball per cluster. Connected components was tried first
    and is the wrong rule: it has no diameter bound, so chains of near
    duplicates fuse unrelated content. On WildChat at 1M it produced a single
    "cluster" of 57,749 vectors (5.8% of the corpus), and on WikiAnswers its
    agreement with editor labels collapsed from 0.75 to 0.004 adjusted Rand
    index over two steps of eps while its largest cluster went 185 -> 24,457.
    Seed-ball degrades gracefully across the same range.

    The seed relation is what carries the semantics: every member is within eps
    of the seed, and the seed is what enters the blocklist, so a held-out
    member is a genuine near-duplicate of a blocklisted item. Two members can
    be up to 2*eps apart, but that pair is never the relation under test.
    """
    n = len(vectors)
    adj = _eps_neighbors(vectors, eps, exclude_same=exclude_same)
    indptr, indices = adj.indptr, adj.indices
    degree = np.diff(indptr)

    labels = np.full(n, -1, dtype=np.int64)
    # Isolated points are their own cluster; assigning them in bulk keeps the
    # Python loop below to the minority of points that actually have a
    # neighbour (over half of WildChat has none at eps=0.4).
    isolated = np.flatnonzero(degree == 0)
    labels[isolated] = np.arange(len(isolated), dtype=np.int64)
    next_label = len(isolated)

    for seed in np.argsort(-degree, kind="stable"):
        if degree[seed] == 0:
            break  # sorted descending: everything from here on is isolated
        if labels[seed] != -1:
            continue
        labels[seed] = next_label
        nbrs = indices[indptr[seed]:indptr[seed + 1]]
        labels[nbrs[labels[nbrs] == -1]] = next_label
        next_label += 1
    return labels


def _cluster_labels(dataset: str, vectors: Optional[np.ndarray], eps: float,
                    exclude_same: Optional[np.ndarray] = None,
                    cache_dir: str = BLOCKLIST_CLUSTER_DIR,
                    pool_n: Optional[int] = None) -> np.ndarray:
    """Cluster labels for a pool, cached to disk by ordering index.

    The saved array holds one label per row of the pool in dataset order, so
    entry i names the cluster of the i-th vector on disk and any caller reading
    the same prefix of the same file can reuse it. Labels depend only on
    (dataset, pool size, eps, whether same-group edges were dropped), all of
    which are in the filename, so changing a knob misses the cache rather than
    silently reusing stale labels.
    """
    os.makedirs(cache_dir, exist_ok=True)
    n = pool_n if vectors is None else len(vectors)
    path = _cluster_cache_path(dataset, n, eps, exclude_same is not None, cache_dir)
    if os.path.exists(path):
        logger.info(f"{dataset}: loaded cached cluster labels from {path}")
        return np.load(path)

    # Discovery runs in the main process, before any worker starts, so it can
    # use every core. The module-level pinning to one thread exists to stop
    # MAX_WORKERS processes each grabbing all cores; it does not apply here,
    # and leaving it would cost roughly an order of magnitude.
    prev_threads = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(os.cpu_count() or prev_threads)
    try:
        if vectors is None:
            raise ValueError(f"{dataset}: no cached labels and no vectors supplied to compute them")
        t0 = time.perf_counter()
        labels = _discover_clusters(vectors, eps, exclude_same=exclude_same)
        logger.info(f"{dataset}: clustered {len(vectors)} vectors in {time.perf_counter() - t0:.1f}s")
    finally:
        faiss.omp_set_num_threads(prev_threads)

    np.save(path, labels)
    logger.info(f"{dataset}: wrote cluster labels to {path}")
    return labels


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
        """A fold's queries as one array: positives, then hard negatives, then
        easy negatives, in that order.

        Ground truth is deliberately NOT returned with them. The two callers
        need different definitions and the difference is load-bearing:
        blocklist_tau_worker/budget_worker judge against cluster membership, while
        generate_quantization must judge against distance to the reference set
        (see cluster_membership and its warning)."""
        parts = [self.rows(sel[key]) for key in ("positive", "hard", "easy") if len(sel[key])]
        return np.concatenate(parts) if parts else np.empty((0, 0), dtype=np.float32)

    @staticmethod
    def cluster_membership(sel: Dict[str, np.ndarray]) -> List[bool]:
        """Label ground truth for query_vectors: positives are held-out members
        of blocklisted clusters, everything else is not.

        NOT usable with quantization_worker, which asserts a zero
        false-negative invariant. That invariant holds only when "duplicate"
        means "within tau of the reference set" -- the cover contains every
        reference vector, so anything within tau of one is necessarily flagged.
        A cluster member can sit FARTHER than tau from the reference set and be
        missed legitimately, so a false negative rate above zero here means
        tau is too tight for this corpus's own duplicate geometry, not a bug
        in the cover itself."""
        actual: List[bool] = []
        for key in ("positive", "hard", "easy"):
            actual.extend([key == "positive"] * len(sel[key]))
        return actual


def build_blocklist_workload(name: str, plot_key: str, label_col: Optional[str],
                             num_vectors: int, n_folds: int) -> Optional[BlocklistWorkload]:
    """Cluster a corpus and cut it into per-fold blocklist splits, or return
    None (logging why) if it cannot supply the requested sizes."""
    eps, min_cluster = BLOCKLIST_EPS, BLOCKLIST_MIN_CLUSTER
    query_fraction, frac_positive, frac_hard = BLOCKLIST_QUERY_FRACTION, 0.2, 0.4

    if not os.path.exists(dataset_filenames[name]):
        logger.warning(f"Skipping {name}: not found on disk")
        return None

    total = len(VectorSource(name))
    pool_n = total if BLOCKLIST_POOL_SIZE is None else min(BLOCKLIST_POOL_SIZE, total)

    # The pool is materialised only to DISCOVER clusters: not needed when a
    # corpus carries its own labels, nor once discovery has cached them.
    needs_pool = label_col is None and not _cluster_cache_exists(
        name, pool_n, eps, name == "CCNews")
    pool = None
    if needs_pool:
        pool = np.ascontiguousarray(VectorSource(name, 0, pool_n).to_array(), dtype=np.float32)

    if label_col is not None:
        meta = load_meta(name, pool_n)
        if label_col not in meta:
            logger.warning(f"{name}: meta column {label_col!r} missing; using geometric discovery")
            labels = _cluster_labels(name, pool, eps, pool_n=pool_n)
        else:
            labels = np.asarray(meta[label_col], dtype=np.int64)
    else:
        exclude = None
        if name == "CCNews":
            dom = load_meta(name, pool_n).get("domain")
            exclude = np.asarray(dom) if dom is not None else None
        labels = _cluster_labels(name, pool, eps, exclude_same=exclude, pool_n=pool_n)

    sizes = np.bincount(labels)
    dup_clusters = np.where(sizes >= min_cluster)[0]
    singletons = np.where(np.isin(labels, np.where(sizes == 1)[0]))[0]
    logger.info(
        f"{name}: pool={pool_n} clusters>={min_cluster}: {len(dup_clusters)} "
        f"(largest={sizes.max()}, median={np.median(sizes[dup_clusters]) if len(dup_clusters) else 0:.0f}), "
        f"singletons={len(singletons)}")
    if len(dup_clusters) < 2:
        logger.warning(f"Skipping {name}: too few duplicate clusters at eps={eps}")
        return None

    stream_size = int(round(query_fraction * num_vectors))
    reference_target = num_vectors - stream_size
    n_pos = int(frac_positive * stream_size)
    n_hard = int(frac_hard * stream_size)
    n_easy = stream_size - n_pos - n_hard

    # Feasibility up front, not fold by fold: a corpus that cannot fill |B|
    # consumes every cluster trying, leaves none withheld, loses its hard
    # negatives, and disappears from the results with only a per-fold warning.
    usable = int(0.8 * sizes[dup_clusters].sum())
    if usable < reference_target + n_hard:
        max_n = int(usable / ((1.0 - query_fraction) + frac_hard * query_fraction))
        logger.error(
            f"Skipping {name}: needs {reference_target} reference + {n_hard} hard-negative "
            f"vectors but only {usable} are available in clusters of >={min_cluster} at "
            f"eps={eps}. Lower WORKLOAD_NUM_VECTORS to <= {max_n} for this corpus, or "
            f"loosen eps/min_cluster.")
        return None

    members_by_cluster = _group_by_label(labels, dup_clusters)
    selections = _fold_selections(
        name, labels, dup_clusters, singletons, members_by_cluster,
        n_folds=n_folds, reference_target=reference_target,
        n_pos=n_pos, n_hard=n_hard, n_easy=n_easy,
        pool_n=pool_n, eps=eps, stream_size=stream_size,
        query_fraction=query_fraction, xgroup=(name == "CCNews"))
    if not _selection_is_usable(selections):
        logger.error(f"Skipping {name}: fold selection is empty")
        return None

    logger.info(
        f"{name}: |B|={len(selections[0]['reference'])} (target {reference_target}), "
        f"stream={stream_size} ({n_pos} pos / {n_hard} hard / {n_easy} easy)")
    return BlocklistWorkload(name, plot_key, selections, pool, stream_size, reference_target)


def blocklist_workloads(num_vectors: int, n_folds: int = BLOCKLIST_FOLDS,
                        only: Optional[str] = None) -> List[BlocklistWorkload]:
    """Every corpus's workload, or just the one `only` names."""
    out = []
    for name, (plot_key, label_col) in BLOCKLIST_DATASETS.items():
        if only is not None and name != only:
            continue
        w = build_blocklist_workload(name, plot_key, label_col, num_vectors, n_folds)
        if w is not None:
            out.append(w)
    return out


def generate_blocklist(runner: Runner) -> None:
    """Asymmetric blocklist workload: a tight reference set B built from
    duplicate clusters, queried by a stream of held-out cluster members
    (positives), members of non-blocklisted clusters (hard negatives) and
    singletons (easy negatives).

    This is the workload the symmetric sweep (generate_system_tau) cannot
    measure. There, B is 80% of the same corpus the queries come from, so B is
    as diffuse as the query stream, d(query -> B) is just the ambient distance
    (~1.38 for these unit-normalised 384-d embeddings) and r + tau exceeds it
    for any ball holding more than one point -- FPR pins at 1 regardless of
    memory. Shrinking B to a set of tight clusters is what opens the budget.

    Clusters are partitioned by cluster, not by vector: half are blocklisted
    (80% of each into B, 20% held out as positives) and half are withheld
    entirely to serve as hard negatives. Splitting by vector instead would
    scatter every cluster across both roles and leave no hard negatives at all.

    A single fixed ball budget (k_fraction, not a sweep) -- varying the
    budget is generate_budget's job now, at its own fixed tau.
    This evaluation is the complementary view: fix k, sweep tau, and read
    how FPR/recall/accesses-saved respond at that one representative budget.
    k_fraction=0.1 sits in the middle of generate_budget's own 5%-20% range,
    so the two evaluations are anchored at a comparable operating point.

    Only submits 'blocklist' jobs (output_csv=BLOCKLIST_CSV) -- does not run
    `runner`. Cluster discovery runs eagerly here, since it needs the whole
    pool in memory.
    """
    dim              = 384
    pool_size        = BLOCKLIST_POOL_SIZE
    num_vectors      = WORKLOAD_NUM_VECTORS
    k_fraction       = 0.1  # fixed ball budget; see docstring
    # Shared absolute grid for the tau-swept panels. Stops at 0.4: measured on
    # a 0..1.2 grid, every corpus was already pinned at FPR 1.000 and 0% saved
    # by tau=0.3, so the bulk of that range carried no information. All the
    # structure is in [0, 0.4], and 8 points resolve it while leaving just
    # enough flat tail to show the saturation.
    tau_values       = [float(t) for t in np.linspace(0.0, 0.4, 8)]
    n_folds          = BLOCKLIST_FOLDS

    for workload in blocklist_workloads(num_vectors, n_folds=n_folds):
        for fold, sel in enumerate(workload.selections):
            runner.submit(
                'blocklist',
                output_csv=BLOCKLIST_CSV,
                dim=dim, dataset=workload.plot_key, seed=fold,
                k_fraction=k_fraction, tau_values=tau_values,
                vectors_reference=workload.rows(sel["reference"]),
                vectors_positive=workload.rows(sel["positive"]),
                vectors_hard_negative=workload.rows(sel["hard"]),
                vectors_easy_negative=workload.rows(sel["easy"]),
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

    Holding tau fixed is what makes recall a measurement here rather than a
    constant, so both recall and FPR move with the budget and the sweep shows
    where paying for more balls stops buying either. The band is deliberately
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
    tau             = 0.1

    k_fractions = [p / 100.0 for p in k_percentages]

    for workload in blocklist_workloads(num_vectors, n_folds=n_folds):
        for fold, sel in enumerate(workload.selections):
            runner.submit(
                'budget',
                output_csv=BUDGET_CSV,
                dim=dim, dataset=workload.plot_key, seed=fold,
                k_fractions=k_fractions, tau=tau,
                vectors_reference=workload.rows(sel["reference"]),
                vectors_positive=workload.rows(sel["positive"]),
                vectors_hard_negative=workload.rows(sel["hard"]),
                vectors_easy_negative=workload.rows(sel["easy"]),
            )


def save_budget(job_results: List[Tuple[str, Any]]) -> pd.DataFrame:
    """Extracts this evaluation's rows from a shared Runner's combined run()
    output (see generate_budget()). Tagged mode 'budget' (runner.budget_worker)
    rather than 'blocklist' (runner.blocklist_tau_worker) so the two
    evaluations' rows split cleanly."""
    return pd.DataFrame(rows_for_mode(job_results, 'budget'))


def plot_budget(df: pd.DataFrame) -> None:
    generate_budget_figures(df)


def generate_quantization(runner: Runner) -> None:
    """Per-quantizer construction data for BallCoverFilter
    (QUANTIZATION_SWEEP), on the same construction and query sets as
    generate_blocklist.

    Every candidate shares one memory budget per k (derived from the
    unquantized cover's byte-per-ball size), so a more compact quantizer packs
    more balls into the same budget and plot_quantization.py can use recorded
    memory usage as its x-axis.

    Reuses build_blocklist_workload rather than cutting its own contiguous
    spans, so the only difference between this sweep and the blocklist one is
    the variable being swept: the quantizer here, the ball budget there. tau is
    a single fixed absolute value, the same one generate_budget and
    generate_equal_memory use, so both evaluations sit at a comparable
    operating point. Ground truth here is distance to the reference set, not the
    blocklist workload's cluster membership: quantization_worker asserts a
    zero false-negative invariant that only holds under the distance
    definition (see BlocklistWorkload.cluster_membership).

    Also passes n_hard/n_easy so quantization_worker can split fpr into
    fpr_hard/fpr_easy the same way blocklist_tau_worker/budget_worker do,
    from the same [positives, hard, easy] query-stream layout (see
    BlocklistWorkload.query_vectors) -- the pooled fpr column is unchanged,
    this only adds the breakdown alongside it.

    Runs once per blocklist corpus, each into its own
    results/quantization/<dataset>/ folder and its own figure folder.
    Kept apart rather than pooled because plot_quantization asserts a single
    dataset -- its panels put memory on the x-axis and one line per quantizer,
    leaving no room for a dataset dimension.

    Only submits 'quantization' jobs -- does not run `runner`."""
    num_vectors      = WORKLOAD_NUM_VECTORS
    dim              = 384
    n_folds          = BLOCKLIST_FOLDS
    # Absolute filter sizes, same 2^11..2^16 ladder generate_scaling uses --
    # no cross-corpus pooling constraint forces these to be shared (each
    # corpus gets its own figure set here), but the same fixed ladder keeps
    # "k" meaning the same absolute ball count across every evaluation that
    # reports it.
    k_values_nominal = [2 ** i for i in range(11, 17)]
    tau              = 0.1

    workloads = blocklist_workloads(num_vectors, n_folds=n_folds)
    if not workloads:
        raise RuntimeError(
            "generate_quantization: no usable workload for any corpus (see the errors "
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

        output_csv = quantization_csv(workload.name)
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        for fold, sel in enumerate(workload.selections):
            vectors_construct = workload.rows(sel["reference"])
            vectors_query = workload.query_vectors(sel)
            # Distance-based truth, not cluster membership: quantization_worker
            # asserts zero false negatives, which only holds under this
            # definition (see BlocklistWorkload.cluster_membership).
            actual_results = compute_static_actual_results(
                vectors_construct, vectors_query, tau)
            for k in k_values:
                for quantize in QUANTIZATION_SWEEP:
                    runner.submit(
                        'quantization',
                        output_csv=output_csv,
                        dim=dim, dataset=workload.name,
                        k=k, quantize=quantize, tau=tau, seed=fold,
                        vectors_construct=vectors_construct, vectors_query=vectors_query,
                        actual_results=actual_results,
                        n_hard=len(sel["hard"]), n_easy=len(sel["easy"]),
                    )


def save_quantization(job_results: List[Tuple[str, Any]]) -> pd.DataFrame:
    """Extracts this evaluation rows from a shared Runner combined run()
    output (see generate_quantization()). The Runner already wrote one
    results/quantization/<dataset>/quantization.csv per corpus, so this only
    rebuilds the combined DataFrame; plot_quantization splits it again."""
    return pd.DataFrame(rows_for_mode(job_results, 'quantization'))


def plot_quantization(df: pd.DataFrame) -> None:
    """One figure set per corpus, each in its own folder.

    Split here rather than handed over whole because
    generate_quantization_figures asserts a single dataset and a single tau,
    and tau is solved per corpus (see generate_quantization)."""
    for dataset in sorted(df["dataset"].unique()):
        outdir = quantization_plot_dir(str(dataset))
        logger.info(f"Plotting quantization for {dataset} -> {outdir}")
        generate_quantization_figures(df[df["dataset"] == dataset], outdir=outdir)


def generate_query_performance() -> pd.DataFrame:
    """Measures single-query vs. batched-query latency/throughput of a
    BallCoverFilter cover as a function of filter size k, across every
    dataset in DATASET_KEYS. One job per (dataset, k, seed), written to
    results/query_performance.csv."""
    num_vectors           = WORKLOAD_NUM_VECTORS
    dim                   = 384
    percentage_construct  = 20
    k_values              = [16, 32, 64, 128, 256, 512, 1024]
    query_tau             = 0.35
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

    # k >= a dataset's construct-set size makes BallCoverFilter fall back to
    # one zero-radius ball per point, so clamp the sweep like generate_quantization() does.
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
    """Measures three BallCoverFilter throughput surfaces against filter
    size k: query (k x batch_size grid), insertion (same grid, timing add()
    instead), and construction (k x n, where indexed-set size itself
    varies).

    Reuses blocklist_workloads() -- the same corpora, cluster-based fold
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
    scaling_query_csv() et al.), mirroring quantization_csv/QUANTIZATION_DIR,
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
    # before comparing), so this only has to be a valid tau, not a tuned one.
    tau_mid          = 0.35
    # Fractions of each fold's blocklist query stream (positives + hard +
    # easy negatives -- see BlocklistWorkload.query_vectors) consumed by the
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


def _build_streaming_negative_set(
    dataset: str, stream_vectors: np.ndarray, candidate_pool: np.ndarray, tau_mid: float, n_negative: int,
) -> np.ndarray:
    """A negative that's true against the FULL stream stays true against any
    prefix of it. `stream_vectors` must cover every fold's own stream
    window (not just fold 0's), since folds see genuinely different data."""
    is_duplicate = compute_static_actual_results(stream_vectors, candidate_pool, tau_mid)
    negatives = candidate_pool[~np.asarray(is_duplicate, dtype=bool)]
    if len(negatives) < n_negative:
        raise RuntimeError(
            f"{dataset}: only found {len(negatives)} true negatives against the full stream "
            f"(needed {n_negative}); shrink n_negative, raise tau_mid, or grow the candidate pool."
        )
    return negatives[:n_negative]


def generate_streaming(runner: Runner) -> None:
    """The paper's "dynamic evaluation" (Section 5.1/5.4, Figure 4): compares
    BallCoverFilter's 4 insertion strategies via stream_rebuild_worker's
    policy="streaming". Produces the streaming figure's panels: (a) FPR vs.
    position + a static-rebuild reference, (b) threshold-expanded objective
    inflation vs. position -- no k sweep: the ball budget k stays fixed for
    the whole run, and only the stream position varies.

    The ball budget k, the bootstrap (construct) window, and the subsequent
    insert-only window are each sized as their own fraction of num_vectors:
    k_percent_stream sets k itself, init_percent_stream controls how many
    vectors seed construct() before any insert() call, and
    insert_percent_stream controls how many are then streamed in one at a
    time. Decoupling the bootstrap window from k matters: the old scheme
    bootstrapped on exactly k vectors (bootstrap_n = min(k, n)), which for
    k <= n gives GonzalesConstruct one point per center and hence
    zero-radius balls at position 0 -- a degenerate starting point that
    saturates FPR to ~1 within a few dozen insertions regardless of method.
    Bootstrapping on a window an order of magnitude larger than k gives the
    filter genuinely non-trivial radii before streaming even starts.

    Implements the paper's Workloads protocol (Section 5.1, "80% construction
    / 20% queries"): init_percent_stream + insert_percent_stream is the
    construction share (default 0.4 + 0.4 = 0.8) and the remaining
    1 - init_percent_stream - insert_percent_stream of num_vectors, past
    every fold's window, is held out as the source of verified-negative
    queries. Each of k_folds=5 folds gets its own stream window, starting
    fold_stride further into the dataset than fold 0's -- non-wrapping via
    advancing_fold_starts, so folds see genuinely different data rather than
    the same pool reordered. A dataset needs enough vectors on disk for both
    the folds' windows and the query span, or this raises.

    Runs once per blocklist corpus (WildChat, CCNews, WikiAnswers), each to
    its own results/streaming/<dataset>/streaming.csv (see streaming_csv()),
    mirroring quantization_csv/QUANTIZATION_DIR -- rather than one
    hardcoded representative dataset outside that trio, so this evaluation
    draws from the same corpora as generate_blocklist/generate_quantization/
    generate_scaling/generate_equal_memory/generate_budget. Unlike those,
    it does not reuse blocklist_workloads(): streaming needs an ordered,
    contiguous span of one corpus to simulate positions in time, not a
    cluster-based reference/query split, so it draws directly from
    VectorSource the same way it always did, just once per corpus instead of
    once for a single hardcoded one. k_fixed and the checkpoint positions are
    pure functions of num_vectors, not of any corpus's geometry, so unlike
    generate_scaling's k or generate_equal_memory's memory_budgets neither
    needs bounding by the smallest corpus -- they are identical across
    corpora already, which is what makes plot_streaming's pooling (see its
    module docstring) meaningful here.

    FNR=0 is asserted per checkpoint inside stream_rebuild_worker (raises,
    not recorded) -- a correctness requirement, not a plotted quantity.

    Only submits 'stream_rebuild' jobs -- does not run `runner`."""
    num_vectors = WORKLOAD_NUM_VECTORS
    dim = 384
    methods = ["nearest", "least_inflation", "merge", "min_heap"]
    n_negative = 500
    # Calibrates tau_mid; see calculate_nn_distance_threshold. Raised from the
    # 10.0 the old single-dataset (MMLU) run used: WildChat's stream window is
    # so duplicate-dense that ~12% of vectors have an exact-duplicate nearest
    # neighbor (distance 0), so a 10th-percentile calibration lands tau_mid on
    # exactly 0.0 and stream_rebuild_worker's zero-FNR invariant then fails on
    # ordinary floating-point round-off at the tau=0 boundary. 20.0 clears
    # that zero-distance mass on every blocklist corpus with real headroom
    # (measured >= ~0.11 on all three) while still calibrating each corpus's
    # own duplicate rate rather than fixing one absolute threshold.
    duplicate_percentage = 20.0
    k_folds = 5  # "we do this 5 times" (Section 5.1)
    seed = 0  # fixed; fold_offset supplies the stream-order variance instead
    k_percent_stream = 0.1  # ball budget k, as a fraction of num_vectors
    init_percent_stream = 0.4  # bootstrap/construct window, as a fraction of num_vectors
    insert_percent_stream = 0.4  # vectors streamed via insert() after bootstrap, as a fraction of num_vectors

    k_fixed = max(16, round(k_percent_stream * num_vectors))

    n_init = max(1, round(init_percent_stream * num_vectors))
    n_insert = max(1, round(insert_percent_stream * num_vectors))
    n_stream_window = n_init + n_insert
    if n_stream_window > num_vectors:
        raise RuntimeError(
            f"init_percent_stream={init_percent_stream} + insert_percent_stream="
            f"{insert_percent_stream} needs {n_stream_window} vectors per fold, but "
            f"num_vectors is only {num_vectors}. Lower either percentage or raise num_vectors."
        )

    # construct()+insert() consume n_stream_window vectors per fold; the
    # remainder of num_vectors, past every fold's window, is held out as the
    # source of verified-negative queries.
    fold_stride = n_stream_window // k_folds
    offsets = [f * fold_stride for f in range(k_folds)]
    negative_pool_size = num_vectors - n_stream_window

    # Placed past every fold's window (all contained in [0, offsets[-1] +
    # n_stream_window)) so it can never overlap any fold's stream.
    negative_pool_start = offsets[-1] + n_stream_window
    negative_pool_end = negative_pool_start + negative_pool_size

    # Bootstrap (construct()) covers positions [0, n_init); the position
    # sweep only checkpoints the subsequent insert-only span, since nothing
    # streams in before n_init. Dataset-independent, computed once.
    checkpoints = sorted(set(
        int(round(x)) for x in np.logspace(np.log10(n_init + 1), np.log10(n_stream_window), num=12)
    ))

    any_usable = False
    for name in BLOCKLIST_DATASETS:
        if not os.path.exists(dataset_filenames[name]):
            logger.warning(f"Skipping {name}: not found on disk")
            continue
        any_usable = True

        total_rows = len(VectorSource(name))
        if total_rows < negative_pool_end:
            raise RuntimeError(
                f"{name}: has only {total_rows} vectors on disk; generate_streaming's "
                f"non-wrapping {k_folds}-fold sweep at num_vectors={num_vectors} needs at least "
                f"{negative_pool_end} (room for {k_folds} stream windows of {n_stream_window} "
                f"vectors each, advancing by {fold_stride}, plus a {negative_pool_size}-vector "
                "negative-candidate pool past the last one). Raise the dataset's available "
                "vectors, or lower num_vectors."
            )

        # The driver needs real arrays only for calibrating tau_mid and
        # building the negative set (one-time per corpus, not per job) --
        # everything else stays lazy. `full_span` covers every fold's
        # window, not just fold 0's.
        full_span = VectorSource(name, 0, negative_pool_start).to_array()
        negative_candidate_pool = VectorSource(name, negative_pool_start, negative_pool_end).to_array()
        if len(negative_candidate_pool) < n_negative:
            raise RuntimeError(
                f"{name}: held-out negative-candidate pool has only "
                f"{len(negative_candidate_pool)} vectors (needed >= {n_negative}). "
                "Raise num_vectors or lower n_negative."
            )

        # Calibrated from fold 0's window (full_span's first n_stream_window vectors).
        logger.info(f"{name}: calibrating tau_mid against the stream window...")
        tau_mid = calculate_nn_distance_threshold(full_span[:n_stream_window], duplicate_percentage)
        logger.info(f"{name}: tau_mid={tau_mid:.4f}")

        logger.info(f"{name}: building fixed streaming negative set from the held-out span...")
        negative_vectors = _build_streaming_negative_set(
            name, full_span, negative_candidate_pool, tau_mid, n_negative,
        )

        def _fold_source(fold_offset: int, length: int, name: str = name) -> VectorSource:
            return VectorSource(name, fold_offset, fold_offset + length)

        output_csv = streaming_csv(name)

        # Each row is self-describing (dataset/method/sweep/metric/fold_offset),
        # so no positional reconstruction is needed despite Runner's
        # completion-order results.

        # --- panels (a)/(b): position sweep at fixed k_fixed, decoupled bootstrap/insert windows ---
        for method in methods:
            for fold_offset in offsets:
                runner.submit(
                    'stream_rebuild',
                    output_csv=output_csv,
                    dim=dim, dataset=name, k=k_fixed, seed=seed, tau_mid=tau_mid, policy="streaming", T=None,
                    insertion_level=method, vectors_stream=_fold_source(fold_offset, n_stream_window),
                    vectors_negative=negative_vectors, bootstrap_n=n_init,
                    checkpoints=checkpoints, sweep="position", fold_offset=fold_offset,
                )
        for fold_offset in offsets:
            runner.submit(
                'stream_rebuild',
                output_csv=output_csv,
                dim=dim, dataset=name, k=k_fixed, seed=seed, tau_mid=tau_mid, policy="oracle", T=None,
                vectors_stream=_fold_source(fold_offset, n_stream_window), vectors_negative=negative_vectors,
                checkpoints=checkpoints, sweep="position", fold_offset=fold_offset,
            )

    if not any_usable:
        raise RuntimeError(
            "generate_streaming: none of the blocklist corpora "
            f"({', '.join(BLOCKLIST_DATASETS)}) were found on disk."
        )


def save_streaming(job_results: List[Tuple[str, Any]]) -> Tuple[pd.DataFrame, int]:
    """Rebuilds this evaluation's DataFrame from a shared Runner's run()
    output, plus k_fixed (the smallest k among 'position'-sweep rows,
    matching plot_all()'s own derivation)."""
    df = pd.DataFrame(rows_for_mode(job_results, 'stream_rebuild'))
    k_fixed = int(df.loc[df["sweep"] == "position", "k"].iloc[0])
    return df, k_fixed


def plot_streaming(df: pd.DataFrame, k_fixed: int) -> None:
    generate_streaming_figures(df, k_fixed=k_fixed)


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

    try:
        generate_equal_memory(runner)
    except Exception:
        logger.exception("An error occurred")
    '''
    try:
        generate_scaling(runner)
    except Exception:
        logger.exception("An error occurred")
    try:
        generate_streaming(runner)
    except Exception:
        logger.exception("An error occurred")
    try:
        generate_system_tau(runner)
    except Exception:
        logger.exception("An error occurred")
    '''
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
        df = pd.read_csv(SYSTEM_TAU_CSV)
        plot_system_tau(df)
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
        # keep_default_na=False: stage/config strings are read verbatim; pandas
        # would otherwise read back as NaN (see plot_quantization.py::load_data()).
        frames = []
        for name in BLOCKLIST_DATASETS:
            path = quantization_csv(name)
            if os.path.exists(path):
                # keep_default_na=False: stage/config strings are read verbatim;
                # pandas would otherwise turn them into NaN.
                frames.append(pd.read_csv(path, keep_default_na=False))
        if frames:
            plot_quantization(pd.concat(frames, ignore_index=True))
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