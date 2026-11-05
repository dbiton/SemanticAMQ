"""The one entry point: generates the data behind paper.tex's figures and draws them.

    python src/paper.py                      # build the Rust kernels, run every evaluation, draw
    python src/paper.py --only budget,construction
    python src/paper.py --missing            # only the evaluations with no complete CSVs yet
    python src/paper.py plot                 # re-draw from existing CSVs (needs no datasets, no build)
    python src/paper.py plot sweeps throughput --results DIR --out DIR
    python src/paper.py --list               # evaluations, their scale and outputs

Running builds the Rust kernels first (rust_build.ensure_built: `maturin develop --release`,
incremental), so a run always uses kernels compiled from the checked-out sources; --no-build skips it.

Evaluation definitions live in experiments.py; the figures in plots/plot_paper.py. Every evaluation
runs at the workload paper.tex states: per corpus, five folds of 80,000 indexed vectors (from the start
of the file) and 20,000 queries (from its end). --num-vectors / --folds override that for a reduced
run; --results sets where CSVs are written and read.

Memory budgets are M_filter/M_raw, the filter's bytes over the raw float32 embedding array (never a cap
count k). The query threshold is tau = 0.05 cosine distance (experiments.FIXED_DELTA) unless a sweep
varies it. Cap-Cover is built by soft anneal (runner.CAP_COVER_CONSTRUCT) with automatic encoding
selection, except in the quantization comparison (fixed encodings) and the construction comparison
(Gonzalez, minimax, soft anneal; unquantized).

The end-to-end evaluation times a filter in front of a flat index, so it runs after the parallel batch,
on pinned cores (--timing-workers, default 1: paper.tex times on one core, one job at a time).
Construction times from the parallel batch read several times too slow; they are not timing claims
unless the construction evaluation is run alone (--only construction --workers 1). Running an
evaluation truncates its CSVs (see CLAUDE.md); collecting jobs does not.
"""
import argparse
import os
import sys

# Must happen before numpy/faiss are imported: otherwise every worker process also grabs every CPU core.
for _env_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env_var, "1")
# auto() drops the encodings that cost too much to scan at a large budget (see
# CapCoverFilter.auto); read in the worker processes too.
os.environ.setdefault("CAPCOVER_SCAN_LIMIT_MS", "1.0")

if __name__ == "__main__":
    # Before anything imports capcover_rs. Child processes re-import this file under another
    # name, so only the command line the user typed builds.
    if not ({"--no-build", "--list", "-h", "--help"} & set(sys.argv[1:]) or sys.argv[1:2] == ["plot"]):
        import rust_build
        rust_build.ensure_built()

from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Callable, Optional, Tuple

import experiments
from experiments import (Runner, blocklist_workloads, equal_memory_filters, logger, results_path,
                         generate_blocklist, generate_budget, generate_cap_quantization,
                         generate_construct_methods, generate_equal_memory)
from filters.cap_cover.cap_quantizers import Quantize
from plots import plot_paper
from plots.plot_paper import (FIGURES, plot_figures, use_corpora, CONFIG_TO_METHOD,
                              E2E_TAU, E2E_MEMORY_FRACTION, E2E_TAUS, E2E_MEMORY_FRACTIONS, PAPER_TAUS_COS)
from runner import CAP_QUANTIZATION_SWEEP, CAP_QUANTIZE_AUTO
from util.dataset_paths import dataset_filenames
from vector_source import VectorSource

# The figures read the thresholds and budgets plot_paper names; the generators use experiments'.
# They are one set of values.
assert abs(E2E_TAU - experiments.FIXED_TAU) < 1e-12
assert all(abs(a - b) < 1e-12 for a, b in zip(E2E_TAUS, experiments.BLOCKLIST_TAUS)) and len(E2E_TAUS) == 8
assert tuple(PAPER_TAUS_COS) == tuple(experiments.BLOCKLIST_DELTAS)
assert E2E_MEMORY_FRACTION == experiments.FIXED_MEMORY_FRACTION
assert tuple(E2E_MEMORY_FRACTIONS) == tuple(experiments.BUDGET_MEMORY_FRACTIONS)
assert len(experiments.BUDGET_MEMORY_FRACTIONS) == len(experiments.CONSTRUCT_MEMORY_FRACTIONS) == 8

# The baselines the two equal-memory figures draw.
PAPER_EQUAL_MEMORY_FILTERS = list(equal_memory_filters)
assert all(cls.__name__ in CONFIG_TO_METHOD for cls in PAPER_EQUAL_MEMORY_FILTERS)

# Cap-Cover runs with automatic encoding selection (cap_quantize=None) everywhere except the
# quantization comparison below.

# The encodings the quantization figure compares: the full-precision, scalar and TurboQuant
# rungs down to 1 bit/dim, plus the seeded-PQ rungs auto() chooses among, and auto() itself.
PAPER_QUANTIZE = tuple(dict.fromkeys(
    (Quantize.FP32, Quantize.FP16, Quantize.SQ8, Quantize.TQ8, Quantize.SQ6, Quantize.SQ4, Quantize.TQ4,
     Quantize.TQ3, Quantize.TQ2, Quantize.TQ1) + tuple(CAP_QUANTIZATION_SWEEP) + (CAP_QUANTIZE_AUTO,)))

FULL_SCALE = (100_000, 5)      # vectors per corpus, folds: 80,000 indexed + 20,000 queries per fold

END_TO_END_CSV = "end_to_end.csv"
E2E_WORKERS = 1      # paper.tex: timing runs on one pinned core, one job at a time


def generate_end_to_end(runner: Runner, num_vectors: Optional[int] = None, n_folds: Optional[int] = None) -> None:
    """The throughput figure's data: per corpus and fold, a CapCoverFilter (automatic encoding) in
    front of a flat index against the index alone (runner.end_to_end_worker), on the blocklist
    workload. Only the cells the three panels read: tau swept at the fixed budget
    E2E_MEMORY_FRACTION, and the budget swept at E2E_TAU. Only submits."""
    cells = sorted({(E2E_MEMORY_FRACTION, t) for t in E2E_TAUS} | {(mf, E2E_TAU) for mf in E2E_MEMORY_FRACTIONS})
    num_vectors = num_vectors or experiments.WORKLOAD_NUM_VECTORS
    n_folds = n_folds or experiments.BLOCKLIST_FOLDS
    for workload in blocklist_workloads(num_vectors, n_folds=n_folds):
        for seed, fold in enumerate(workload.folds):
            runner.submit(
                'end_to_end', output_csv=results_path(END_TO_END_CSV),
                dim=384, dataset=workload.plot_key, seed=seed, cells=cells, quantize=None,
                vectors_reference=fold.reference, vectors_query=fold.queries,
            )


@dataclass(frozen=True)
class Evaluation:
    """One evaluation: how to submit its jobs, at what scale, what it writes, how it runs."""
    name: str
    generate: Callable                 # (runner, num_vectors, n_folds) -> None; only submits jobs
    scale: Optional[Tuple[int, int]]   # (vectors per corpus, folds) from paper.tex; None: fixed inside generate
    outputs: str                       # glob under the results folder
    per_corpus: bool                   # one file per corpus (else one file in all)
    workers: Optional[int] = None      # None: the shared parallel batch; N: alone, on N pinned cores


EVALUATIONS = {e.name: e for e in (
    Evaluation("blocklist", generate_blocklist, FULL_SCALE, "blocklist.csv", False),
    Evaluation("budget", generate_budget, FULL_SCALE, "budget.csv", False),
    Evaluation("construction", generate_construct_methods, FULL_SCALE, "construct_methods/*.csv", True),
    Evaluation("quantization", partial(generate_cap_quantization, quantizes=PAPER_QUANTIZE), FULL_SCALE,
               "cap_quantization/*/*.csv", True),
    Evaluation("equal-memory", partial(generate_equal_memory, filters=PAPER_EQUAL_MEMORY_FILTERS,
                                       count_memory_steps=8), FULL_SCALE,
               "equal_memory/*/equal_memory.csv", True),
    Evaluation("end-to-end", generate_end_to_end, FULL_SCALE, END_TO_END_CSV, False, workers=E2E_WORKERS),
)}


def missing_evaluations(results=None, n_corpora=None) -> list:
    """The evaluations (EVALUATIONS' names, in order) whose CSVs are not all in `results` (default
    the results folder): one file for the single-file ones, one per corpus for the rest. An evaluation
    with only some of its corpora counts as missing, and running it rewrites the files it has."""
    results = Path(results) if results else Path(experiments.RESULTS_DIR)
    n_corpora = n_corpora or len(experiments.BLOCKLIST_DATASETS)
    found = lambda pattern: [p for p in results.glob(pattern) if p.stat().st_size > 0]
    return [e.name for e in EVALUATIONS.values()
            if len(found(e.outputs)) < (n_corpora if e.per_corpus else 1)]


def check_datasets(evaluations, num_vectors=None, n_folds=None) -> None:
    """Stops before any job is submitted if a corpus is missing or too short for the largest
    workload an evaluation needs. The generators would instead skip it with a warning, and a
    run of hours would end with a figure that quietly lacks a dataset."""
    need = 0
    for e in evaluations:
        nv, nf = e.scale
        need = max(need, (n_folds or nf) * (num_vectors or nv))
    problems = []
    for name in experiments.BLOCKLIST_DATASETS:
        path = dataset_filenames[name]
        if not os.path.exists(path):
            problems.append(f"{name}: {path} not found")
            continue
        rows = len(VectorSource(name))
        if rows < need:
            problems.append(f"{name}: {rows:,} rows, need {need:,}")
    if problems:
        raise SystemExit("cannot run the paper's evaluations:\n  " + "\n  ".join(problems))


def _submit_and_run(evaluations, num_workers, num_vectors=None, n_folds=None) -> None:
    runner = Runner(num_workers=num_workers)
    for e in evaluations:
        nv, nf = e.scale
        try:
            e.generate(runner, num_vectors=num_vectors or nv, n_folds=n_folds or nf)
        except Exception:
            logger.exception(f"{e.name} failed to submit")
    runner.shuffle()
    runner.run()


def run_evaluations(names, num_vectors=None, n_folds=None) -> None:
    """Runs the named evaluations: the untimed ones as one shuffled batch on experiments.MAX_WORKERS,
    then each timed group alone on its own pinned cores."""
    unknown = [n for n in names if n not in EVALUATIONS]
    if unknown:
        raise SystemExit(f"unknown evaluation(s) {unknown}; choose from {list(EVALUATIONS)}")
    chosen = [EVALUATIONS[n] for n in names]
    check_datasets(chosen, num_vectors, n_folds)
    batch = [e for e in chosen if e.workers is None]
    if batch:
        _submit_and_run(batch, experiments.MAX_WORKERS, num_vectors, n_folds)
        logger.info("Main batch completed.")
    for e in (e for e in chosen if e.workers is not None):
        os.environ["CAPCOVER_PIN_CORES"] = str(e.workers)
        _submit_and_run([e], e.workers, num_vectors, n_folds)
        logger.info(f"{e.name} completed.")
    logger.info("All submitted jobs completed.")


def plot_all(figures=None, out=None, results=None) -> None:
    """Draws the figures (default all), then the tables, from the results folder."""
    results = results or experiments.RESULTS_DIR
    plot_figures(figures or list(FIGURES), results=results, out=out)
    if not figures:
        try:
            plot_paper.construction_times()
        except FileNotFoundError as e:
            logger.warning(f"construction_times: {e}")


def configure(datasets=None, workers=None, results=None) -> None:
    """Overrides the corpora (and the figures' to match), the pool size and the results folder
    before anything is submitted."""
    if workers:
        experiments.MAX_WORKERS = workers
    if results:
        experiments.RESULTS_DIR = str(results)
    if datasets:
        unknown = [d for d in datasets if d not in dataset_filenames]
        if unknown:
            raise SystemExit(f"unknown dataset(s) {unknown}; choose from {sorted(dataset_filenames)}")
        experiments.BLOCKLIST_DATASETS.clear()
        experiments.BLOCKLIST_DATASETS.update({d: d.lower() for d in datasets})
        use_corpora(list(datasets))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", choices=("run", "plot"), default="run",
                        help="run: generate then draw (default); plot: draw from existing CSVs")
    parser.add_argument("figures", nargs="*", help="(plot) figure names; default all (see plots.plot_paper.FIGURES)")
    parser.add_argument("--only", help="comma-separated evaluations to run (default: all)")
    parser.add_argument("--missing", action="store_true",
                        help="run only the evaluations whose CSVs are not all present, then draw")
    parser.add_argument("--no-build", action="store_true", help="do not build the Rust kernels first")
    parser.add_argument("--timing-workers", type=int, help=f"pinned cores for the timed evaluations (default {E2E_WORKERS})")
    parser.add_argument("--no-plot", action="store_true", help="skip drawing after a run")
    parser.add_argument("--list", action="store_true", help="list the evaluations and exit")
    parser.add_argument("--num-vectors", type=int, help="vectors per corpus, for every evaluation")
    parser.add_argument("--folds", type=int, help="folds, for every evaluation")
    parser.add_argument("--datasets", help="comma-separated registry names to run and draw instead of the five")
    parser.add_argument("--workers", type=int, help=f"parallel jobs (default {experiments.MAX_WORKERS})")
    parser.add_argument("--results", type=Path, help="folder the CSVs are written to and read from (default results)")
    parser.add_argument("--out", type=Path, help="figure folder (default figures/)")
    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list:
        for e in EVALUATIONS.values():
            scale = f"{e.scale[0]:,} vectors x {e.scale[1]} fold(s)"
            print(f"{e.name:14s} {scale:34s} {e.outputs}" + (f"  [alone, {e.workers} pinned core(s)]" if e.workers else ""))
        return
    configure(args.datasets.split(",") if args.datasets else None, args.workers, args.results)
    if args.timing_workers:
        EVALUATIONS["end-to-end"] = replace(EVALUATIONS["end-to-end"], workers=args.timing_workers)
    if args.command == "plot":
        plot_all(args.figures, args.out, args.results)
        return
    if args.figures:
        parser.error("figure names only apply to the plot command")
    if args.missing:
        names = missing_evaluations()
        logger.info(f"missing evaluations: {', '.join(names) or 'none'}")
    else:
        names = args.only.split(",") if args.only else list(EVALUATIONS)
    if names:
        run_evaluations(names, args.num_vectors, args.folds)
    if not args.no_plot and (args.missing or not args.only):
        plot_all(out=args.out, results=args.results)


if __name__ == "__main__":
    main(sys.argv[1:])
