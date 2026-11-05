import logging
import os

# Must happen before numpy/faiss are imported: otherwise each of Runner's
# MAX_WORKERS processes also grabs every CPU core for its own internal
# threads, and process x thread parallelism multiply into severe
# oversubscription (observed: whole-system unresponsiveness).
for _env_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env_var, "1")

import faiss
import h5py
import numpy as np

faiss.omp_set_num_threads(1)  # belt-and-suspenders alongside the env vars above

from logging_config import configure_logging
from runner import (Runner,
                    CAP_QUANTIZATION_SWEEP, CAP_COVER_CONSTRUCT, _k_for_memory_fraction)
from plots.plot_common import DATASETS, DATASET_LABELS
from typing import Any, Dict, List, Optional, Sequence, Tuple
from filters.cap_cover.cap_cover_filter import CapCoverFilter, Quantize
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

# Memory budgets are chosen as B/M, the filter's bytes over the raw index (the |B| construct
# vectors as float32, dim*4 bytes each), never as a cap count k: the cover sizes itself from
# the bytes (runner._k_for_memory_fraction). 8 points per sweep.
#   BUDGET_MEMORY_FRACTIONS     generate_budget and the cap_quantization sweep
#   FIXED_MEMORY_FRACTION       the one budget of the tau sweep, the streaming run, the throughput
#                               tau panel, the latency timing and the insertion timing
#   CONSTRUCT_MEMORY_FRACTIONS  the construction-method comparison (every encoding; a TQ2 cover has
#                               a cap per vector from B/M = 6.3%, a TQ4 one from 12.6%)
BUDGET_MEMORY_FRACTIONS = (0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05)
FIXED_MEMORY_FRACTION = 0.02
CONSTRUCT_MEMORY_FRACTIONS = (0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1)
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


def cap_quantization_csv(dataset: str) -> str:
    return os.path.join(CAP_QUANTIZATION_DIR, dataset, "cap_quantization.csv")



# generate_scaling runs once per blocklist corpus and keeps each run in its
# own folder, mirroring cap_quantization_csv/CAP_QUANTIZATION_DIR above: throughput
# is pooled across corpora for plotting, but keeping each corpus's raw rows
# apart on disk makes a single corpus's run inspectable/rerunnable on its own.








# generate_equal_memory runs once per blocklist corpus and keeps each run in
# its own folder, mirroring cap_quantization_csv/CAP_QUANTIZATION_DIR above. Plotted
# the same way too (see plot_equal_memory): one figure set per corpus, in its
# own folder, rather than pooled into a single cross-corpus line per filter.
EQUAL_MEMORY_DIR = "results/equal_memory"


def equal_memory_csv(dataset: str) -> str:
    return os.path.join(EQUAL_MEMORY_DIR, dataset, "equal_memory.csv")




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


def generate_equal_memory(runner: Runner, filters: Optional[List[Any]] = None, n_folds: int = BLOCKLIST_FOLDS,
                          count_memory_steps: int = 8, tau: Optional[float] = None,
                          cap_quantizes: Optional[Sequence[Quantize]] = None) -> None:
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
    is always submitted). tau (L2; default FIXED_TAU) is the one threshold of the
    comparison; cap_quantizes fixes CapCoverFilter's encoding, one line each (default: auto()'s choice). n_folds and count_memory_steps (log-spaced budgets) scale the
    sweep down for a smaller run.

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
    tau                    = FIXED_TAU if tau is None else tau
    cap_variants           = [{}] if not cap_quantizes else [{"extra_kwargs": {"quantize": q}} for q in cap_quantizes]
    # Log-spaced, not linear: the baselines' feasibility floors span more
    # than a decade below this range's top (IVF-PQ ~0.6%, quantized-point-set
    # RaBitQ ~3.6%, of raw storage at n=100K/dim=384 -- HNSW's ~30% floor
    # sits above max_memory_fraction entirely, so it is expected to stay
    # infeasible throughout), so a linear sweep would waste points at the top
    # and barely resolve the first two floors. 8 steps over this range's
    # 2 decades is 4 points/decade.
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
                # CapCoverFilter: one submission per encoding in cap_quantizes, labelled
                # "CapCoverFilter (TQ2)" etc. by equal_memory_worker's Config naming; with none,
                # auto() chooses its encoding and the label is plainly "CapCoverFilter".
                for cap_kwargs in cap_variants:
                    runner.submit(
                        'equal_memory',
                        output_csv=output_csv,
                        filter_cls=CapCoverFilter,
                        dim=dim, distance_threshold=tau,
                        memory_budget=int(memory_budget),
                        vectors_construct=vectors_construct, vectors_calibration=vectors_calibration,
                        vectors_query=vectors_query,
                        actual_calibration=actual_calibration, actual_query=actual_query,
                        dataset=workload.name, **cap_kwargs,
                    )






# Output of generate_cap_cover_system_tau.








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


def generate_blocklist(runner: Runner, cap_quantize: Optional[Quantize] = None) -> None:
    """Fixed-budget tau sweep. The reference set B is the first rows of each
    corpus's file and the query stream is the last rows of it
    (build_blocklist_workload): no clustering, no stratified sampling. Every
    query is labeled by its distance to B at each tau
    (runner._blocklist_cover_rows), so the positive share is whatever the
    file order yields and bounds the accesses saved. File order can carry
    structure (e.g. paraphrase groups stored together), which this split
    does not correct.

    A single fixed memory budget (FIXED_MEMORY_FRACTION of the raw index, not a
    sweep) -- varying the budget is generate_budget's job now, at its own fixed
    tau. This evaluation is the complementary view: fix the budget, sweep tau,
    and read how FPR/accesses-saved respond at that one representative budget.
    FIXED_MEMORY_FRACTION sits inside generate_budget's own range, so the two
    evaluations are anchored at a comparable operating point.

    Only submits 'blocklist' jobs (output_csv=BLOCKLIST_CSV) -- does not run
    `runner`. Cluster discovery runs eagerly here, since it needs the whole
    pool in memory.
    """
    dim              = 384
    num_vectors      = WORKLOAD_NUM_VECTORS
    memory_fraction  = FIXED_MEMORY_FRACTION  # fixed budget B/M; see docstring
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
                memory_fraction=memory_fraction, tau_values=tau_values, quantize=cap_quantize,
                vectors_reference=workload.rows(sel["reference"]),
                vectors_query=workload.query_vectors(sel),
            )






def generate_budget(runner: Runner, cap_quantize: Optional[Quantize] = None) -> None:
    """The blocklist workload read along its other axis: the memory budget B/M
    (BUDGET_MEMORY_FRACTIONS) is the swept variable and tau is held at one
    fixed value, where generate_blocklist instead sweeps tau at one fixed
    budget.

    Holding tau fixed holds the labels fixed, so FPR and accesses saved move
    with the budget alone and the sweep shows where paying for more balls
    stops buying either. The band is deliberately
    narrow (0.5%-5% of the raw index): below it the cover is too coarse to
    localise B at a fixed threshold, and above it the budget approaches
    saturation, where each cap stands in for so few reference vectors that
    there is nothing left to compress.

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
    memory_fractions = list(BUDGET_MEMORY_FRACTIONS)
    tau             = FIXED_TAU

    for workload in blocklist_workloads(num_vectors, n_folds=n_folds):
        for fold, sel in enumerate(workload.selections):
            runner.submit(
                'budget',
                output_csv=BUDGET_CSV,
                dim=dim, dataset=workload.plot_key, seed=fold,
                memory_fractions=memory_fractions, tau=tau, quantize=cap_quantize,
                vectors_reference=workload.rows(sel["reference"]),
                vectors_query=workload.query_vectors(sel),
            )






def generate_cap_quantization(runner: Runner, memory_fractions: Optional[Sequence[float]] = None,
                              quantizes: Optional[Sequence[Any]] = None) -> None:
    """One CapCoverFilter per (dataset, memory budget, seed, quantize) over
    CAP_QUANTIZATION_SWEEP (AUTO_CANDIDATES), at the same budgets as
    generate_budget (BUDGET_MEMORY_FRACTIONS), the same fixed tau and the same
    blocklist workloads/source-size split.

    `quantizes` overrides CAP_QUANTIZATION_SWEEP (paper.py adds the FP/SQ/TQ rungs
    and CAP_QUANTIZE_AUTO).

    Only submits 'cap_quantization' jobs -- does not run `runner`."""
    num_vectors      = WORKLOAD_NUM_VECTORS
    dim              = 384
    n_folds          = BLOCKLIST_FOLDS
    quantizes        = tuple(quantizes or CAP_QUANTIZATION_SWEEP)
    memory_fractions = tuple(memory_fractions or BUDGET_MEMORY_FRACTIONS)
    tau              = FIXED_TAU

    workloads = blocklist_workloads(num_vectors, n_folds=n_folds)
    if not workloads:
        raise RuntimeError(
            "generate_cap_quantization: no usable workload for any corpus (see the errors "
            "above -- most likely WORKLOAD_NUM_VECTORS is too large for their "
            "duplicate clusters)")

    for workload in workloads:
        reference_len = len(workload.selections[0]["reference"])
        logger.info(f"{workload.name}: memory_fractions={list(memory_fractions)} (|B|={reference_len}), tau={tau:.4f}")

        output_csv = cap_quantization_csv(workload.name)
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        for fold, sel in enumerate(workload.selections):
            vectors_construct = workload.rows(sel["reference"])
            vectors_query = workload.query_vectors(sel)
            actual_results = compute_static_actual_results(
                vectors_construct, vectors_query, tau)
            for memory_fraction in memory_fractions:
                for quantize in quantizes:
                    runner.submit(
                        'cap_quantization',
                        output_csv=output_csv,
                        dim=dim, dataset=workload.name,
                        memory_fraction=memory_fraction, quantize=quantize, tau=tau, seed=fold,
                        vectors_construct=vectors_construct, vectors_query=vectors_query,
                        actual_results=actual_results,
                    )
















def generate_streaming(runner: Runner, construct_methods: Optional[List[str]] = None,
                       k_sweep: bool = True, bootstrap_fraction: float = 0.5) -> None:
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
    very filter generate_blocklist builds at the same memory budget.

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

    k is the blocklist workload's own budget, the caps that fit FIXED_MEMORY_FRACTION of the raw index, and
    |B| is trimmed to the same size on every corpus (see _fold_selections), so
    k and the checkpoint positions are identical across corpora, which is
    what makes plot_streaming's pooling meaningful.

    FNR=0 is asserted per checkpoint inside stream_rebuild_worker (raises,
    not recorded) -- a correctness requirement, not a plotted quantity.

    construct_methods narrows the position sweep to those constructors and
    k_sweep=False skips panel (c)'s k sweep (src/paper.py needs neither).

    bootstrap_fraction is the share of B built by construct(); the rest is the insert
    stream, so it sets how far the set outgrows the fixed budget k.

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
    memory_fraction    = FIXED_MEMORY_FRACTION  # generate_blocklist's fixed budget B/M
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
            k = _k_for_memory_fraction(dim, n, memory_fraction)  # as runner._blocklist_cover_rows sizes it
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






# Construction-method comparison: one CSV per corpus (plot_construct_methods
# draws one figure per corpus), plus one per --scale run of
# generate_construct_methods_scale, whose "scale_" prefix keeps it out of
# the sweep's figures.
CONSTRUCT_METHODS_DIR = "results/construct_methods"
# The construction evaluation compares these, each unquantized (FP32), at TQ2 and at TQ4; sphere_merge
# (the paper's default) is not evaluated there, and soft_anneal starts from minimax's cover.
CONSTRUCT_EVALUATED = ("gonzalez", "minimax", "soft_anneal")
CONSTRUCT_QUANTIZES = (Quantize.FP32, Quantize.TQ2, Quantize.TQ4)


def construct_methods_csv(dataset: str) -> str:
    return os.path.join(CONSTRUCT_METHODS_DIR, f"{dataset}.csv")




def generate_construct_methods(runner: Runner) -> None:
    """CapCoverFilter's construction methods (CONSTRUCT_EVALUATED: gonzalez,
    minimax, soft_anneal; sphere_merge is not evaluated here), each at
    CONSTRUCT_QUANTIZES (unquantized, TQ2 and TQ4), compared on the blocklist workload: the same construction
    and query sets as generate_budget/generate_quantization, with ground
    truth defined as distance to the reference set at tau. Only the
    construction method varies (see runner.construct_method_worker).

    One job per (corpus, fold, method, encoding, memory budget), so rows reach
    construct_methods_csv(corpus) as each cover finishes, and the slow
    large-k covers don't serialize one method's whole sweep onto a single
    worker.

    Only submits 'construct_method' jobs -- does not run `runner`."""
    dim         = 384
    num_vectors = WORKLOAD_NUM_VECTORS
    n_folds     = BLOCKLIST_FOLDS
    tau         = FIXED_TAU
    # B/M from 1% to 10% (8 points): the budgets where the covers are wide
    # enough for the construction method to matter and not yet saturated.
    memory_fractions = list(CONSTRUCT_MEMORY_FRACTIONS)
    methods     = list(CONSTRUCT_EVALUATED)

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
                for quantize in CONSTRUCT_QUANTIZES:
                    for mf in memory_fractions:
                        runner.submit(
                            'construct_method',
                            output_csv=output_csv,
                            dim=dim, dataset=workload.plot_key, method=method, quantize=quantize,
                            memory_fraction=mf, tau=tau, seed=fold,
                            vectors_construct=vectors_construct, vectors_query=vectors_query,
                            actual_results=actual_results,
                        )











