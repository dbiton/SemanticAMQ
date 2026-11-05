import logging
import os

# Must happen before numpy/faiss are imported: otherwise each of Runner's
# MAX_WORKERS processes also grabs every CPU core for its own internal
# threads, and process x thread parallelism multiply into severe
# oversubscription (observed: whole-system unresponsiveness).
for _env_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env_var, "1")

import faiss
import numpy as np

faiss.omp_set_num_threads(1)  # belt-and-suspenders alongside the env vars above

from logging_config import configure_logging
from runner import (Runner,
                    CAP_QUANTIZATION_SWEEP)
from typing import Any, List, Optional, Sequence
from filters.cap_cover.cap_cover_filter import CapCoverFilter, Quantize
from filters.dsh_filter import DSHFilter
from filters.hnsw_filter import HNSWFilter
from filters.ivf_filter import IVFFilter
from filters.quantized_vector_filter import QuantizedVectorFilter
from filters.mlp_filter import MLPFilter
from filters.pacuk_lsh_filter import PacukLSHFilter
from filters.sankowski_wygocki_filter import SankowskiWygockiFilter
from filters.va_plus_filter import VAPlusFilter
from workloads import (BlocklistWorkload, build_blocklist_workload, fold_ground_truth,
                       stratified_calibration_split)

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

# Where every evaluation writes its CSVs: paper.py's --results sets this before submitting.
RESULTS_DIR = "results"


def results_path(*parts: str) -> str:
    """A path under RESULTS_DIR, read at call time so one process can target another folder."""
    return os.path.join(RESULTS_DIR, *parts)

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
# (the paper's values; plots/plot_paper.py states the same ones and paper.py asserts they agree).
def l2_threshold(cosine_distance: float) -> float:
    """L2 distance of unit vectors at this cosine distance (tau^2 = 2 * delta)."""
    return float(np.sqrt(2.0 * cosine_distance))


FIXED_DELTA = 0.05                                                   # cosine distance, the paper's default
BLOCKLIST_DELTAS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2)   # the tau sweeps (8 points)
FIXED_TAU = l2_threshold(FIXED_DELTA)
BLOCKLIST_TAUS = [l2_threshold(d) for d in BLOCKLIST_DELTAS]

BLOCKLIST_FOLDS = 5

# Memory budgets are chosen as M_filter/M_raw, the filter's bytes over the raw index (the |B| construct
# vectors as float32, dim*4 bytes each), never as a cap count k: the cover sizes itself from
# the bytes (runner._k_for_memory_fraction). 8 points per sweep.
#   BUDGET_MEMORY_FRACTIONS     generate_budget and the cap_quantization sweep
#   FIXED_MEMORY_FRACTION       the one budget of the tau sweep and of the throughput tau panel
#   CONSTRUCT_MEMORY_FRACTIONS  the construction-method comparison (every encoding; a TQ2 cover has
#                               a cap per vector from M_filter/M_raw = 6.3%, a TQ4 one from 12.6%)
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

# generate_cap_quantization's analogue, parallel to generate_quantization
# rather than replacing it in place -- the two sweep structurally different
# encodings (see CAP_QUANTIZATION_SWEEP), so keeping both lets either be
# re-run/compared independently.
CAP_QUANTIZATION_DIR = "cap_quantization"   # under RESULTS_DIR (see results_path)


def cap_quantization_csv(dataset: str) -> str:
    return results_path(CAP_QUANTIZATION_DIR, dataset, "cap_quantization.csv")



# generate_scaling runs once per blocklist corpus and keeps each run in its
# own folder, mirroring cap_quantization_csv/CAP_QUANTIZATION_DIR above: throughput
# is pooled across corpora for plotting, but keeping each corpus's raw rows
# apart on disk makes a single corpus's run inspectable/rerunnable on its own.








# generate_equal_memory runs once per blocklist corpus and keeps each run in
# its own folder, mirroring cap_quantization_csv/CAP_QUANTIZATION_DIR above. Plotted
# the same way too (see plot_equal_memory): one figure set per corpus, in its
# own folder, rather than pooled into a single cross-corpus line per filter.
EQUAL_MEMORY_DIR = "equal_memory"   # under RESULTS_DIR (see results_path)


def equal_memory_csv(dataset: str) -> str:
    return results_path(EQUAL_MEMORY_DIR, dataset, "equal_memory.csv")




# generate_streaming runs once per blocklist corpus and keeps each run in its
# own folder, mirroring cap_quantization_csv/CAP_QUANTIZATION_DIR above; plot_streaming
# pools every dataset's rows the same way plot_scaling does (see its
# docstring: "this figure is single-dataset" refers to color channel reuse
# for method, not to a restriction on how many datasets feed it).



# generate_equal_memory()'s own comparison set: excludes DSBFFilter/
# CoveringLSHFallbackFilter/LSBFFilter, which degenerate to ~100% duplicate
# predictions at realistic construct-set sizes (still used elsewhere).
# CapCoverFilter is submitted separately (see generate_equal_memory), not
# listed here.
equal_memory_filters = [
    HNSWFilter, IVFFilter, DSHFilter, QuantizedVectorFilter, MLPFilter,
    PacukLSHFilter, SankowskiWygockiFilter, VAPlusFilter,
]






def generate_equal_memory(runner: Runner, filters: Optional[List[Any]] = None, num_vectors: Optional[int] = None, n_folds: Optional[int] = None,
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
    stream (the fold's query slice) is split into a calibration
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
    num_vectors            = num_vectors or WORKLOAD_NUM_VECTORS
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
    reference_len = min(len(w.folds[0].reference) for w in workloads)
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
        for fold, sel in enumerate(workload.folds):
            vectors_construct, vectors_stream = sel.reference, sel.queries
            # Ground truth over the whole stream first, then a STRATIFIED cut
            # into the two slices -- a prefix cut of the file-ordered stream
            # can put every positive in calibration and none in measurement.
            # See stratified_calibration_split. Jobs get the stream handle and the two index sets.
            actual_stream = fold_ground_truth(sel, tau)
            calibration_idx, query_idx = stratified_calibration_split(actual_stream, calibration_fraction, seed=fold)
            actual_calibration = actual_stream[calibration_idx].tolist()
            actual_query = actual_stream[query_idx].tolist()
            logger.info(
                f"{workload.name} fold {fold}: calibration {len(calibration_idx)} vectors "
                f"({sum(actual_calibration)} duplicates), measurement {len(query_idx)} "
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
                        vectors_construct=vectors_construct, vectors_stream=vectors_stream,
                        calibration_idx=calibration_idx, query_idx=query_idx,
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
                        vectors_construct=vectors_construct, vectors_stream=vectors_stream,
                        calibration_idx=calibration_idx, query_idx=query_idx,
                        actual_calibration=actual_calibration, actual_query=actual_query,
                        dataset=workload.name, **cap_kwargs,
                    )






# Output of generate_cap_cover_system_tau.








# generate_blocklist()'s corpora: deliberately NOT DATASET_KEYS. That list is
# the 5-dataset symmetric sweep, which this workload replaces rather than
# extends -- and these five were chosen for their duplicate-cluster geometry
# (see BLOCKLIST_DATASETS' per-entry comments), verified by running eps=0.4
# seed-ball discovery on each before adding it, not assumed from the label.
BLOCKLIST_CSV = "blocklist.csv"   # under RESULTS_DIR (see results_path)
# generate_budget's rows: the same workload, swept over the ball budget at a
# fixed tau via runner.budget_worker rather than runner.blocklist_tau_worker.
# Its own file rather than reusing BLOCKLIST_CSV, since the two evaluations
# sweep different variables and mixing them would need a second grouping
# column on top of `sweep`.
BUDGET_CSV = "budget.csv"   # under RESULTS_DIR (see results_path)
BLOCKLIST_DATASETS = {
    # registry name -> plot key
    "WildChat": "wildchat",
    "CCNews": "ccnews",
    "WikiAnswers": "wikianswers",
    "Quora": "quora",
    "ChatGPTParaphrases": "chatgpt_paraphrases",
}

def blocklist_workloads(num_vectors: int, n_folds: int = BLOCKLIST_FOLDS,
                        only: Optional[str] = None) -> List[BlocklistWorkload]:
    """Every corpus's workload, or just the one `only` names."""
    out = []
    for name, plot_key in BLOCKLIST_DATASETS.items():
        if only is not None and name != only:
            continue
        w = build_blocklist_workload(name, plot_key, num_vectors, n_folds, BLOCKLIST_QUERY_FRACTION)
        if w is not None:
            out.append(w)
    return out


def generate_blocklist(runner: Runner, cap_quantize: Optional[Quantize] = None,
                       num_vectors: Optional[int] = None, n_folds: Optional[int] = None) -> None:
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

    Only submits 'blocklist' jobs (output_csv=results_path(BLOCKLIST_CSV)) -- does not run
    `runner`. Cluster discovery runs eagerly here, since it needs the whole
    pool in memory.
    """
    dim              = 384
    num_vectors      = num_vectors or WORKLOAD_NUM_VECTORS
    memory_fraction  = FIXED_MEMORY_FRACTION  # fixed budget M_filter/M_raw; see docstring
    # Shared absolute grid for the tau-swept panels. Stops at 0.4: measured on
    # a 0..1.2 grid, every corpus was already pinned at FPR 1.000 and 0% saved
    # by tau=0.3, so the bulk of that range carried no information. All the
    # structure is in [0, 0.4], and 8 points resolve it while leaving just
    # enough flat tail to show the saturation.
    tau_values       = list(BLOCKLIST_TAUS)
    n_folds          = n_folds or BLOCKLIST_FOLDS

    for workload in blocklist_workloads(num_vectors, n_folds=n_folds):
        for fold, sel in enumerate(workload.folds):
            runner.submit(
                'blocklist',
                output_csv=results_path(BLOCKLIST_CSV),
                dim=dim, dataset=workload.plot_key, seed=fold,
                memory_fraction=memory_fraction, tau_values=tau_values, quantize=cap_quantize,
                vectors_reference=sel.reference, vectors_query=sel.queries,
            )






def generate_budget(runner: Runner, cap_quantize: Optional[Quantize] = None,
                    num_vectors: Optional[int] = None, n_folds: Optional[int] = None) -> None:
    """The blocklist workload read along its other axis: the memory budget M_filter/M_raw
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

    Only submits 'budget' jobs (output_csv=results_path(BUDGET_CSV)) -- does not run
    `runner`."""
    dim             = 384
    num_vectors     = num_vectors or WORKLOAD_NUM_VECTORS
    n_folds         = n_folds or BLOCKLIST_FOLDS
    memory_fractions = list(BUDGET_MEMORY_FRACTIONS)
    tau             = FIXED_TAU

    for workload in blocklist_workloads(num_vectors, n_folds=n_folds):
        for fold, sel in enumerate(workload.folds):
            runner.submit(
                'budget',
                output_csv=results_path(BUDGET_CSV),
                dim=dim, dataset=workload.plot_key, seed=fold,
                memory_fractions=memory_fractions, tau=tau, quantize=cap_quantize,
                vectors_reference=sel.reference, vectors_query=sel.queries,
            )






def generate_cap_quantization(runner: Runner, memory_fractions: Optional[Sequence[float]] = None,
                              quantizes: Optional[Sequence[Any]] = None,
                              num_vectors: Optional[int] = None, n_folds: Optional[int] = None) -> None:
    """One CapCoverFilter per (dataset, memory budget, seed, quantize) over
    CAP_QUANTIZATION_SWEEP (AUTO_CANDIDATES), at the same budgets as
    generate_budget (BUDGET_MEMORY_FRACTIONS), the same fixed tau and the same
    blocklist workloads/source-size split.

    `quantizes` overrides CAP_QUANTIZATION_SWEEP (paper.py adds the FP/SQ/TQ rungs
    and CAP_QUANTIZE_AUTO).

    Only submits 'cap_quantization' jobs -- does not run `runner`."""
    num_vectors      = num_vectors or WORKLOAD_NUM_VECTORS
    dim              = 384
    n_folds          = n_folds or BLOCKLIST_FOLDS
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
        reference_len = len(workload.folds[0].reference)
        logger.info(f"{workload.name}: memory_fractions={list(memory_fractions)} (|B|={reference_len}), tau={tau:.4f}")

        output_csv = cap_quantization_csv(workload.name)
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        for fold, sel in enumerate(workload.folds):
            vectors_construct, vectors_query = sel.reference, sel.queries
            actual_results = fold_ground_truth(sel, tau)
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






















# Construction-method comparison: one CSV per corpus (plot_construct_methods
# draws one figure per corpus), plus one per --scale run of
# generate_construct_methods_scale, whose "scale_" prefix keeps it out of
# the sweep's figures.
CONSTRUCT_METHODS_DIR = "construct_methods"   # under RESULTS_DIR (see results_path)
# The construction evaluation compares these, each unquantized (FP32), at TQ2 and at TQ4; sphere_merge
# (the paper's default) is not evaluated there, and soft_anneal starts from minimax's cover.
CONSTRUCT_EVALUATED = ("gonzalez", "minimax", "soft_anneal")
CONSTRUCT_QUANTIZES = (Quantize.FP32,)      # the paper's construction comparison is unquantized


def construct_methods_csv(dataset: str) -> str:
    return results_path(CONSTRUCT_METHODS_DIR, f"{dataset}.csv")




def generate_construct_methods(runner: Runner, num_vectors: Optional[int] = None, n_folds: Optional[int] = None) -> None:
    """CapCoverFilter's construction methods (CONSTRUCT_EVALUATED: gonzalez,
    minimax, sphere_merge, soft_anneal), each at
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
    num_vectors = num_vectors or WORKLOAD_NUM_VECTORS
    n_folds     = n_folds or BLOCKLIST_FOLDS
    tau         = FIXED_TAU
    # M_filter/M_raw from 1% to 10% (8 points): the budgets where the covers are wide
    # enough for the construction method to matter and not yet saturated.
    memory_fractions = list(CONSTRUCT_MEMORY_FRACTIONS)
    methods     = list(CONSTRUCT_EVALUATED)

    for workload in blocklist_workloads(num_vectors, n_folds=n_folds):
        output_csv = construct_methods_csv(workload.plot_key)
        for fold, sel in enumerate(workload.folds):
            vectors_construct, vectors_query = sel.reference, sel.queries
            actual_results = fold_ground_truth(sel, tau)
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
