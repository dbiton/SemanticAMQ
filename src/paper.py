"""Generates the data behind paper.tex's figures, and the figures: main.py's
counterpart for the paper alone. It submits only the evaluations the paper
plots, and within them only the cells the figures read, then draws
plots/plot_paper.py's figures (written to suggestions/figures/).

    python src/paper.py              # generate_and_plot_all(): full run
    python src/paper.py plot         # re-draw the figures from existing results/

    options (both modes where they apply):
      --num-vectors N    vectors per corpus (main.WORKLOAD_NUM_VECTORS, default 100,000)
      --datasets A,B,..  registry names (util.dataset_paths) to run and draw instead of
                         the paper's five, e.g. those a machine has
      --workers N        parallel jobs (main.MAX_WORKERS)
      --out DIR          figure folder (default suggestions/figures)
      --results DIR      (plot) read the CSVs from DIR instead of results/
      --only A,B,..      generate just these evaluations (names: see GENERATORS); default all
      --ablation-vectors N / --ablation-folds N     size of the ablation workloads (default 20,000 / 3)
      --backend-vectors N / --backend-datasets A,B  size and corpora of the backend timing (default
                                                    --num-vectors / all five)
    A smaller or different run needs a copy of the repo without results/ (generators
    truncate their CSVs -- see CLAUDE.md), e.g. a git worktree with datasets/ linked in.

Each figure and what feeds it (the CSVs are main.py's, so a paper run and a
main.py run write the same files -- and truncate them, see CLAUDE.md):

    sweeps        budget.csv, blocklist.csv                generate_budget, generate_blocklist
    quantization  cap_quantization/<corpus>/*.csv          generate_cap_quantization (figure, and
                                                           tab:quantization via plots/table_quantization.py)
    construction  construct_methods/<corpus>.csv           generate_construct_methods
    streaming     streaming/<corpus>/streaming.csv         generate_streaming (position sweep,
                                                           sphere_merge bootstrap only)
    equal-memory  equal_memory/<corpus>/equal_memory.csv   generate_equal_memory (the baselines
                                                           the two figures show: zero-FN FPR on
                                                           every corpus, FPR/FNR of the rest on one)
    throughput    end_to_end.csv                           generate_end_to_end (defined here)
    ablation      ablation.csv                             generate_ablation: cap vs ball accept rule, construction,
                                                           encoding, codes, sphere-merge t and p, selection delta
    backends      backends.csv                             generate_backends: filter in front of flat/HNSW/IVF,
                                                           per-query latency (pinned, sequential)
    timing        timing.csv                               generate_timing: construction and per-arrival insert
                                                           time (pinned, sequential)

Not generated: scaling, cap_cover_system_tau, query_performance and the
construct-method scale run, which no paper figure reads.
"""
import os
import sys

# Must happen before numpy/faiss are imported (see main.py).
for _env_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env_var, "1")
# auto() drops the encodings that cost too much to scan at a large budget (see
# CapCoverFilter.auto); read in the worker processes too.
os.environ.setdefault("CAPCOVER_SCAN_LIMIT_MS", "1.0")

from functools import partial

import main
from runner import CAP_QUANTIZATION_SWEEP
from filters.cap_cover.cap_quantizers import Quantize
from runner import ABLATION_DELTAS
from main import (Runner, CAP_COVER_CONSTRUCT, BLOCKLIST_FOLDS, blocklist_workloads, equal_memory_filters, logger,
                  generate_blocklist, generate_budget, generate_cap_quantization,
                  generate_construct_methods, generate_streaming, generate_equal_memory)
from plots.plot_paper import (plot_figures, use_corpora, CONFIG_TO_METHOD, EQUAL_MEMORY_HIDDEN, STREAM_BOOTSTRAP,
                              E2E_TAU, E2E_K_FRACTION, E2E_TAUS, E2E_K_FRACTIONS, PAPER_TAUS_COS)

assert STREAM_BOOTSTRAP == CAP_COVER_CONSTRUCT, "the streaming figure bootstraps with the paper's construct method"

# The baselines the two equal-memory figures draw; the rest of
# main.equal_memory_filters would be run and discarded.
PAPER_EQUAL_MEMORY_FILTERS = [cls for cls in equal_memory_filters
                              if CONFIG_TO_METHOD.get(cls.__name__) not in EQUAL_MEMORY_HIDDEN]

END_TO_END_CSV = "results/end_to_end.csv"


def generate_end_to_end(runner: Runner) -> None:
    """The throughput figure's data: per corpus and fold, a CapCoverFilter in
    front of a flat index against the index alone (runner.end_to_end_worker),
    on the blocklist workload. Only the cells the three panels read: tau swept
    at the fixed budget E2E_K_FRACTION, and the budget swept at E2E_TAU.

    Only submits 'end_to_end' jobs -- does not run `runner`."""
    dim = 384
    cells = sorted({(E2E_K_FRACTION, t) for t in E2E_TAUS} | {(kf, E2E_TAU) for kf in E2E_K_FRACTIONS})
    for workload in blocklist_workloads(main.WORKLOAD_NUM_VECTORS, n_folds=BLOCKLIST_FOLDS):
        for fold, sel in enumerate(workload.selections):
            runner.submit(
                'end_to_end', output_csv=END_TO_END_CSV,
                dim=dim, dataset=workload.plot_key, seed=fold, cells=cells,
                vectors_reference=workload.rows(sel["reference"]),
                vectors_query=workload.query_vectors(sel),
            )


ABLATION_CSV = "results/ablation.csv"
BACKENDS_CSV = "results/backends.csv"
TIMING_CSV = "results/timing.csv"

# The ablation builds ~16 covers per (corpus, fold, budget), so it runs on smaller workloads than
# the rest: ABLATION_NUM_VECTORS vectors per corpus (80% reference, 20% queries) and ABLATION_FOLDS folds.
ABLATION_NUM_VECTORS = 20_000
ABLATION_FOLDS = 3
ABLATION_K_FRACTIONS = (0.02, 0.1)
# The backend timing and the pinned construction/insertion timing: None = main.WORKLOAD_NUM_VECTORS.
BACKEND_NUM_VECTORS = None
BACKEND_DATASETS = None            # None = all of main.BLOCKLIST_DATASETS
BACKEND_K_FRACTION = 0.1
TIMING_NUM_VECTORS = 10_000
TIMING_DATASET = "WikiAnswers"


def generate_ablation(runner: Runner) -> None:
    """The ablation table's data (runner.ablation_worker): per corpus, fold and budget, the
    paper's configuration, its one-choice-at-a-time variants and the ball accept rules over its
    own cover, at every cosine distance in ABLATION_DELTAS. Only submits 'ablation' jobs."""
    for workload in blocklist_workloads(ABLATION_NUM_VECTORS, n_folds=ABLATION_FOLDS):
        for fold, sel in enumerate(workload.selections):
            reference, queries = workload.rows(sel["reference"]), workload.query_vectors(sel)
            for kf in ABLATION_K_FRACTIONS:
                runner.submit('ablation', output_csv=ABLATION_CSV, dim=384, dataset=workload.plot_key,
                              seed=fold, k_fraction=kf, vectors_reference=reference, vectors_query=queries,
                              deltas=ABLATION_DELTAS)


def generate_backends(runner: Runner) -> None:
    """The filter in front of flat / HNSW / IVF backends (runner.backend_latency_worker): one
    fold per corpus at k = BACKEND_K_FRACTION |B|. Timed, so it runs alone on one pinned core
    (generate_and_plot_all). Only submits 'backend_latency' jobs."""
    n = BACKEND_NUM_VECTORS or main.WORKLOAD_NUM_VECTORS
    for workload in blocklist_workloads(n, n_folds=1):
        if BACKEND_DATASETS and workload.name not in BACKEND_DATASETS:
            continue
        sel = workload.selections[0]
        runner.submit('backend_latency', output_csv=BACKENDS_CSV, dim=384, dataset=workload.plot_key,
                      seed=0, k_fraction=BACKEND_K_FRACTION, deltas=PAPER_TAUS_COS,
                      vectors_reference=workload.rows(sel["reference"]),
                      vectors_query=workload.query_vectors(sel))


def generate_timing(runner: Runner) -> None:
    """Construction time per method and per-arrival insertion time per strategy
    (runner.pinned_timing_worker) on TIMING_DATASET. Timed: one pinned core, alone."""
    for workload in blocklist_workloads(TIMING_NUM_VECTORS, n_folds=1, only=TIMING_DATASET):
        runner.submit('pinned_timing', output_csv=TIMING_CSV, dim=384, dataset=workload.plot_key,
                      vectors_reference=workload.rows(workload.selections[0]["reference"]))


# Budgets of the quantization comparison, as % of the raw index (k FP32 caps / |B|): the
# range the equal-memory comparison covers. Above ~10% every encoding has converged.
PAPER_BUDGET_PCT = (0.1, 0.2, 0.5, 1, 2, 5, 10)
PAPER_K_LADDER = [round(pct / 100 * 80_000) for pct in PAPER_BUDGET_PCT]    # k at |B| = 80,000


# The encodings the quantization table compares: the full-precision, scalar and
# TurboQuant rungs down to 1 bit/dim, plus the seeded-PQ rungs auto() chooses among.
PAPER_QUANTIZE = tuple(dict.fromkeys(
    (Quantize.FP32, Quantize.FP16, Quantize.SQ8, Quantize.TQ8, Quantize.SQ6, Quantize.SQ4, Quantize.TQ4,
     Quantize.TQ3, Quantize.TQ2, Quantize.TQ1) + tuple(CAP_QUANTIZATION_SWEEP)))


def generate_quantization(runner: Runner) -> None:
    """main.generate_cap_quantization over PAPER_QUANTIZE, with its k ladder scaled
    to the workload size, so a smaller run (--num-vectors) still has k below the
    reference set."""
    scale = main.WORKLOAD_NUM_VECTORS / 100_000
    generate_cap_quantization(runner, k_values_nominal=[max(1, round(k * scale)) for k in PAPER_K_LADDER],
                              quantizes=PAPER_QUANTIZE)


# The stream is the rest of B: 80% bootstrap leaves inserts = 20% of |B| (16,000 at 100K) against a
# budget k = 10% of |B|. At 50% the set doubled, and every method saturated at FPR 1.
STREAM_BOOTSTRAP_FRACTION = 0.8

GENERATORS = {
    "blocklist": generate_blocklist,
    "budget": generate_budget,
    "quantization": generate_quantization,
    "construction": generate_construct_methods,
    "streaming": partial(generate_streaming, construct_methods=[STREAM_BOOTSTRAP], k_sweep=False,
                         bootstrap_fraction=STREAM_BOOTSTRAP_FRACTION),
    "equal-memory": partial(generate_equal_memory, filters=PAPER_EQUAL_MEMORY_FILTERS),
    "ablation": generate_ablation,
    "end-to-end": generate_end_to_end,
    "backends": generate_backends,
    "timing": generate_timing,
}
# Evaluations that time something: run after the rest, alone, one at a time (see generate_and_plot_all).
TIMED = ("end-to-end", "backends", "timing")
PAPER_GENERATORS = tuple(GENERATORS.values())


def check_datasets() -> None:
    """Stops before any job is submitted if a corpus is missing or too short.
    The generators would instead skip it with a warning, and a run of hours
    would end with a figure that quietly lacks a dataset."""
    need = BLOCKLIST_FOLDS * main.WORKLOAD_NUM_VECTORS
    problems = []
    for name in main.BLOCKLIST_DATASETS:
        path = main.dataset_filenames[name]
        if not os.path.exists(path):
            problems.append(f"{name}: {path} not found")
            continue
        rows = len(main.VectorSource(name))
        if rows < need:
            problems.append(f"{name}: {rows:,} rows, need {need:,} ({BLOCKLIST_FOLDS} folds x "
                            f"{main.WORKLOAD_NUM_VECTORS:,})")
    if problems:
        raise SystemExit("cannot run the paper's evaluations:\n  " + "\n  ".join(problems))


# The throughput evaluation times a filter against a flat index, so it runs after the
# rest, alone, on a few cores each pinned to its own (runner._pin_to_own_core): timings
# taken beside dozens of other jobs read several times too slow.
E2E_WORKERS = 8


def _submit_and_run(generators, num_workers) -> None:
    runner = Runner(num_workers=num_workers)
    for generate in generators:
        name = getattr(generate, "__name__", None) or generate.func.__name__
        try:
            generate(runner)
        except Exception:
            logger.exception(f"{name} failed to submit")
    runner.shuffle()
    runner.run()


def generate_and_plot_all(out=None, only=None) -> None:
    """Run the paper's evaluations as one shuffled batch on main.MAX_WORKERS (see
    main.generate_and_plot_all), then the throughput evaluation alone on E2E_WORKERS
    pinned cores, then the backend and construction/insertion timings sequentially on
    one pinned core, then draw the figures. Each generator is guarded so one failing
    to submit doesn't take down the others. `only` names the GENERATORS to run."""
    set_thresholds()
    names = list(only) if only else list(GENERATORS)
    unknown = [n for n in names if n not in GENERATORS]
    if unknown:
        raise SystemExit(f"unknown evaluation(s) {unknown}; choose from {sorted(GENERATORS)}")
    check_datasets()
    batch = [GENERATORS[n] for n in names if n not in TIMED]
    if batch:
        _submit_and_run(batch, main.MAX_WORKERS)
        logger.info("Main batch completed; timing the throughput evaluation.")
    if "end-to-end" in names:
        os.environ["CAPCOVER_PIN_CORES"] = str(E2E_WORKERS)
        _submit_and_run([generate_end_to_end], E2E_WORKERS)
    sequential = [GENERATORS[n] for n in ("backends", "timing") if n in names]
    if sequential:
        os.environ["CAPCOVER_PIN_CORES"] = "1"
        _submit_and_run(sequential, 1)
    logger.info("All submitted jobs completed.")
    if not only:
        plot_all(out=out)


def plot_all(figures=None, out=None, results=None) -> None:
    """Re-draws the paper's figures (default all) from results/."""
    plot_figures(figures, results=results, out=out)


def set_thresholds() -> None:
    """The paper's thresholds (plots.plot_paper.PAPER_TAU_COS and PAPER_TAUS_COS, in
    cosine distance) into main's L2 knobs, before anything is submitted."""
    main.FIXED_TAU = E2E_TAU
    main.BLOCKLIST_TAUS = list(E2E_TAUS)


def configure(num_vectors=None, datasets=None, workers=None, ablation_vectors=None, ablation_folds=None,
              backend_vectors=None, backend_datasets=None) -> None:
    """Overrides main's workload size, corpora and pool size (and the figures'
    corpora to match), and the ablation's and backend timing's own sizes, before
    anything is submitted."""
    global ABLATION_NUM_VECTORS, ABLATION_FOLDS, BACKEND_NUM_VECTORS, BACKEND_DATASETS
    set_thresholds()
    if num_vectors:
        main.WORKLOAD_NUM_VECTORS = num_vectors
    if workers:
        main.MAX_WORKERS = workers
    if ablation_vectors:
        ABLATION_NUM_VECTORS = ablation_vectors
    if ablation_folds:
        ABLATION_FOLDS = ablation_folds
    if backend_vectors:
        BACKEND_NUM_VECTORS = backend_vectors
    if backend_datasets:
        BACKEND_DATASETS = list(backend_datasets)
    if datasets:
        unknown = [d for d in datasets if d not in main.dataset_filenames]
        if unknown:
            raise SystemExit(f"unknown dataset(s) {unknown}; choose from {sorted(main.dataset_filenames)}")
        main.BLOCKLIST_DATASETS.clear()
        main.BLOCKLIST_DATASETS.update({d: d.lower() for d in datasets})
        use_corpora(list(datasets))


def _parse(argv):
    opts = {"--num-vectors": None, "--datasets": None, "--workers": None, "--out": None, "--results": None,
            "--only": None, "--ablation-vectors": None, "--ablation-folds": None, "--backend-vectors": None,
            "--backend-datasets": None}
    rest = []
    it = iter(argv)
    for arg in it:
        if arg in opts:
            opts[arg] = next(it, None)
            if opts[arg] is None:
                raise SystemExit(f"{arg} needs a value")
        else:
            rest.append(arg)
    return opts, rest


def _int(value):
    return int(value) if value else None


if __name__ == "__main__":
    opts, rest = _parse(sys.argv[1:])
    configure(_int(opts["--num-vectors"]), opts["--datasets"].split(",") if opts["--datasets"] else None,
              _int(opts["--workers"]), _int(opts["--ablation-vectors"]), _int(opts["--ablation-folds"]),
              _int(opts["--backend-vectors"]),
              opts["--backend-datasets"].split(",") if opts["--backend-datasets"] else None)
    if rest[:1] == ["plot"]:
        plot_all(rest[1:], opts["--out"], opts["--results"])
    else:
        generate_and_plot_all(opts["--out"], opts["--only"].split(",") if opts["--only"] else None)
