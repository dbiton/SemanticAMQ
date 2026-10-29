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
      --missing          generate only the evaluations whose CSVs are not in results/ (see
                         missing_evaluations), then draw the figures; runs nothing if none is missing

Every evaluation runs on the same workload: 80,000 construct and 20,000 query vectors per fold (main.
WORKLOAD_NUM_VECTORS = 100,000 split 80/20) and main.BLOCKLIST_FOLDS = 5 folds. Memory budgets are B/M, the
filter's bytes over the raw float32 index (never a cap count k): sweeps and quantization 0.005-0.05,
construction 0.01-0.1, equal-memory 0.001-0.1, and 0.02 wherever one budget is fixed; 8 points per sweep.
    A smaller or different run needs a copy of the repo without results/ (generators
    truncate their CSVs -- see CLAUDE.md), e.g. a git worktree with datasets/ linked in.

Each figure and what feeds it (the CSVs are main.py's, so a paper run and a
main.py run write the same files -- and truncate them, see CLAUDE.md):

    sweeps        budget.csv, blocklist.csv                generate_budget, generate_blocklist
    quantization  cap_quantization/<corpus>/*.csv          generate_cap_quantization
    construction  construct_methods/<corpus>.csv           generate_construct_methods
    equal-memory  equal_memory/<corpus>/equal_memory.csv   generate_equal_memory (the baselines
                                                           the two figures show: zero-FN FPR on
                                                           every corpus, FPR/FNR of the rest on one)
    throughput    end_to_end.csv                           generate_end_to_end (defined here)
Not generated: the tables and what only they read (ablation, latency, timing), streaming (its figure is not
drawn either), scaling, cap_cover_system_tau, query_performance and the
construct-method scale run, which no paper figure reads.

On the server (100,000 vectors per corpus), `python src/paper.py` runs every evaluation above in order: one shuffled
batch on main.MAX_WORKERS (blocklist, budget, quantization, construction, equal-memory),
then end-to-end on 8 pinned cores, then draws the figures (plots/plot_paper.py: suggestions/figures/). `--only` runs a subset and
`--missing` the ones with no CSVs yet; `python src/paper.py plot` redraws without running. Soft anneal (from minimax) is the construction method of every
cover (runner.CAP_COVER_CONSTRUCT) and the slowest by far; budget the batch accordingly
(tests: tests/test_soft_anneal_construct.py).
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
from pathlib import Path

import main
from runner import CAP_QUANTIZATION_SWEEP, CAP_QUANTIZE_AUTO
from filters.cap_cover.cap_quantizers import Quantize
from main import (Runner, BLOCKLIST_FOLDS, blocklist_workloads, equal_memory_filters, logger,
                  generate_blocklist, generate_budget, generate_cap_quantization,
                  generate_construct_methods, generate_equal_memory)
from plots import plot_paper
from plots.plot_paper import (FIGURES, plot_figures, use_corpora, CONFIG_TO_METHOD, EQUAL_MEMORY_HIDDEN,
                              E2E_TAU, E2E_MEMORY_FRACTION, E2E_TAUS, E2E_MEMORY_FRACTIONS, PAPER_TAUS_COS,
                              l2_threshold)

# The figures read the budgets plot_paper names; the generators use main's. They are one set of values.
assert E2E_MEMORY_FRACTION == main.FIXED_MEMORY_FRACTION
assert tuple(E2E_MEMORY_FRACTIONS) == tuple(main.BUDGET_MEMORY_FRACTIONS)
assert len(main.BUDGET_MEMORY_FRACTIONS) == len(main.CONSTRUCT_MEMORY_FRACTIONS) == len(PAPER_TAUS_COS) == 8

# The baselines the two equal-memory figures draw; the rest of
# main.equal_memory_filters would be run and discarded.
PAPER_EQUAL_MEMORY_FILTERS = [cls for cls in equal_memory_filters
                              if CONFIG_TO_METHOD.get(cls.__name__) not in EQUAL_MEMORY_HIDDEN]

# The sweeps grid (blocklist, budget), the throughput figure and the equal-memory comparison run CapCoverFilter at
# this one encoding (not auto()'s choice); the equal-memory comparison at this cosine distance.
PAPER_CAP_QUANTIZE = Quantize.TQ2
EQUAL_MEMORY_TAU_COS = 0.02
EQUAL_MEMORY_CAP_QUANTIZES = (Quantize.TQ2, Quantize.TQ4)     # one Cap-Cover line each

END_TO_END_CSV = "results/end_to_end.csv"


def generate_end_to_end(runner: Runner) -> None:
    """The throughput figure's data: per corpus and fold, a CapCoverFilter (PAPER_CAP_QUANTIZE) in
    front of a flat index against the index alone (runner.end_to_end_worker),
    on the blocklist workload. Only the cells the three panels read: tau swept
    at the fixed budget E2E_MEMORY_FRACTION, and the budget swept at E2E_TAU.

    Only submits 'end_to_end' jobs -- does not run `runner`."""
    dim = 384
    cells = sorted({(E2E_MEMORY_FRACTION, t) for t in E2E_TAUS} | {(mf, E2E_TAU) for mf in E2E_MEMORY_FRACTIONS})
    for workload in blocklist_workloads(main.WORKLOAD_NUM_VECTORS, n_folds=BLOCKLIST_FOLDS):
        for fold, sel in enumerate(workload.selections):
            runner.submit(
                'end_to_end', output_csv=END_TO_END_CSV,
                dim=dim, dataset=workload.plot_key, seed=fold, cells=cells, quantize=PAPER_CAP_QUANTIZE,
                vectors_reference=workload.rows(sel["reference"]),
                vectors_query=workload.query_vectors(sel),
            )


# The encodings the quantization figure compares: the full-precision, scalar and
# TurboQuant rungs down to 1 bit/dim, plus the seeded-PQ rungs auto() chooses among.
PAPER_QUANTIZE = tuple(dict.fromkeys(
    (Quantize.FP32, Quantize.FP16, Quantize.SQ8, Quantize.TQ8, Quantize.SQ6, Quantize.SQ4, Quantize.TQ4,
     Quantize.TQ3, Quantize.TQ2, Quantize.TQ1) + tuple(CAP_QUANTIZATION_SWEEP) + (CAP_QUANTIZE_AUTO,)))


def generate_quantization(runner: Runner) -> None:
    """main.generate_cap_quantization over PAPER_QUANTIZE at main.BUDGET_MEMORY_FRACTIONS
    (0.005-0.05 of the raw index, like the sweeps figure): every rung, every encoding auto()
    chooses among, and auto() itself."""
    generate_cap_quantization(runner, quantizes=PAPER_QUANTIZE)


GENERATORS = {
    "blocklist": partial(generate_blocklist, cap_quantize=PAPER_CAP_QUANTIZE),
    "budget": partial(generate_budget, cap_quantize=PAPER_CAP_QUANTIZE),
    "quantization": generate_quantization,
    "construction": generate_construct_methods,
    "equal-memory": partial(generate_equal_memory, filters=PAPER_EQUAL_MEMORY_FILTERS, count_memory_steps=8,
                            tau=l2_threshold(EQUAL_MEMORY_TAU_COS), cap_quantizes=EQUAL_MEMORY_CAP_QUANTIZES),
    "end-to-end": generate_end_to_end,
}
# Evaluations that time something: run after the rest, alone, one at a time (see generate_and_plot_all).
TIMED = ("end-to-end",)
PAPER_GENERATORS = tuple(GENERATORS.values())

# What each evaluation writes under results/ (what the figures read), and whether it is
# one file per corpus.
OUTPUTS = {
    "blocklist": ("blocklist.csv", False),
    "budget": ("budget.csv", False),
    "quantization": ("cap_quantization/*/*.csv", True),
    "construction": ("construct_methods/*.csv", True),
    "equal-memory": ("equal_memory/*/equal_memory.csv", True),
    "end-to-end": ("end_to_end.csv", False),
}
assert OUTPUTS.keys() == GENERATORS.keys()


def missing_evaluations(results=None, n_corpora=None) -> list:
    """The evaluations (GENERATORS' names, in order) whose CSVs are not all in `results` (default
    results/): one file for the single-file ones, one per corpus for the rest. An evaluation with
    only some of its corpora counts as missing, and running it rewrites the files it has."""
    results = Path(results) if results else plot_paper.RESULTS
    n_corpora = n_corpora or len(main.BLOCKLIST_DATASETS)
    return [name for name, (pattern, per_corpus) in OUTPUTS.items()
            if len(list(results.glob(pattern))) < (n_corpora if per_corpus else 1)]


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


def generate_and_plot_all(out=None, only=None, plot=None) -> None:
    """Run the paper's evaluations as one shuffled batch on main.MAX_WORKERS (see
    main.generate_and_plot_all), then the throughput evaluation alone on E2E_WORKERS
    pinned cores, then draw the figures. Each generator is guarded so one failing
    to submit doesn't take down the others. `only` names the GENERATORS to run; the figures are drawn
    after a full run, or after a subset when `plot` is True."""
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
    logger.info("All submitted jobs completed.")
    if plot if plot is not None else not only:
        plot_all(out=out)


def plot_all(figures=None, out=None, results=None) -> None:
    """Re-draws the paper's figures (default all but the streaming one, which the paper run does not
    produce) from results/."""
    plot_figures(figures or [name for name in FIGURES if name != "streaming"], results=results, out=out)


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
    opts = {"--num-vectors": None, "--datasets": None, "--workers": None, "--out": None, "--results": None,
            "--only": None, "--missing": False}
    rest = []
    it = iter(argv)
    for arg in it:
        if arg == "--missing":
            opts[arg] = True
        elif arg in opts:
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
              _int(opts["--workers"]))
    if rest[:1] == ["plot"]:
        plot_all(rest[1:], opts["--out"], opts["--results"])
    elif opts["--missing"]:
        names = missing_evaluations()
        logger.info(f"missing evaluations: {', '.join(names) or 'none'}")
        if names:
            generate_and_plot_all(opts["--out"], names, plot=True)
        else:
            plot_all(out=opts["--out"])
    else:
        generate_and_plot_all(opts["--out"], opts["--only"].split(",") if opts["--only"] else None)
