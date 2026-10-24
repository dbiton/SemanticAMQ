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
                                                           the figure shows only)
    throughput    end_to_end.csv                           generate_end_to_end (defined here)

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
from main import (Runner, CAP_COVER_CONSTRUCT, BLOCKLIST_FOLDS, blocklist_workloads, equal_memory_filters, logger,
                  generate_blocklist, generate_budget, generate_cap_quantization,
                  generate_construct_methods, generate_streaming, generate_equal_memory)
from plots.plot_paper import (plot_figures, use_corpora, CONFIG_TO_METHOD, EQUAL_MEMORY_HIDDEN, STREAM_BOOTSTRAP,
                              E2E_TAU, E2E_K_FRACTION, E2E_TAUS, E2E_K_FRACTIONS)

assert STREAM_BOOTSTRAP == CAP_COVER_CONSTRUCT, "the streaming figure bootstraps with the paper's construct method"

# The baselines equal-memory-fpr-fnr-vs-budget draws; the rest of
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

PAPER_GENERATORS = (
    generate_blocklist,
    generate_budget,
    generate_quantization,
    generate_construct_methods,
    partial(generate_streaming, construct_methods=[STREAM_BOOTSTRAP], k_sweep=False,
            bootstrap_fraction=STREAM_BOOTSTRAP_FRACTION),
    partial(generate_equal_memory, filters=PAPER_EQUAL_MEMORY_FILTERS),
    generate_end_to_end,
)


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


def generate_and_plot_all(out=None) -> None:
    """Run the paper's evaluations as one shuffled batch on main.MAX_WORKERS (see
    main.generate_and_plot_all), then the throughput evaluation alone on E2E_WORKERS
    pinned cores, then draw the figures. Each generator is guarded so one failing to
    submit doesn't take down the others."""
    set_thresholds()
    check_datasets()
    _submit_and_run([g for g in PAPER_GENERATORS if g is not generate_end_to_end], main.MAX_WORKERS)
    logger.info("Main batch completed; timing the throughput evaluation.")
    os.environ["CAPCOVER_PIN_CORES"] = str(E2E_WORKERS)
    _submit_and_run([generate_end_to_end], E2E_WORKERS)
    logger.info("All submitted jobs completed.")
    plot_all(out=out)


def plot_all(figures=None, out=None, results=None) -> None:
    """Re-draws the paper's figures (default all) from results/."""
    plot_figures(figures, results=results, out=out)


def set_thresholds() -> None:
    """The paper's thresholds (plots.plot_paper.PAPER_TAU_COS and PAPER_TAUS_COS, in
    cosine distance) into main's L2 knobs, before anything is submitted."""
    main.FIXED_TAU = E2E_TAU
    main.BLOCKLIST_TAUS = list(E2E_TAUS)


def configure(num_vectors=None, datasets=None, workers=None) -> None:
    """Overrides main's workload size, corpora and pool size (and the figures'
    corpora to match) before anything is submitted."""
    set_thresholds()
    if num_vectors:
        main.WORKLOAD_NUM_VECTORS = num_vectors
    if workers:
        main.MAX_WORKERS = workers
    if datasets:
        unknown = [d for d in datasets if d not in main.dataset_filenames]
        if unknown:
            raise SystemExit(f"unknown dataset(s) {unknown}; choose from {sorted(main.dataset_filenames)}")
        main.BLOCKLIST_DATASETS.clear()
        main.BLOCKLIST_DATASETS.update({d: d.lower() for d in datasets})
        use_corpora(list(datasets))


def _parse(argv):
    opts = {"--num-vectors": None, "--datasets": None, "--workers": None, "--out": None, "--results": None}
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


if __name__ == "__main__":
    opts, rest = _parse(sys.argv[1:])
    configure(int(opts["--num-vectors"]) if opts["--num-vectors"] else None,
              opts["--datasets"].split(",") if opts["--datasets"] else None,
              int(opts["--workers"]) if opts["--workers"] else None)
    if rest[:1] == ["plot"]:
        plot_all(rest[1:], opts["--out"], opts["--results"])
    else:
        generate_and_plot_all(opts["--out"])
