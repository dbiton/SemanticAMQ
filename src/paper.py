"""The one entry point: generates the data behind paper.tex's figures and tables, and draws them.

    python src/paper.py                      # every evaluation, then the figures and tables
    python src/paper.py --only budget,streaming
    python src/paper.py --missing            # only the evaluations with no complete CSVs yet
    python src/paper.py plot                 # re-draw from existing CSVs (needs no datasets)
    python src/paper.py plot sweeps throughput --results DIR --out DIR
    python src/paper.py --list               # evaluations, their scale and outputs

Evaluation definitions live in experiments.py; figures in plots/plot_paper.py, tables in
plots/tables_paper.py and plots/table_quantization.py. Each evaluation below runs at the
scale paper.tex states for it (EVALUATIONS): the sweeps at 100,000 vectors per corpus (80,000
indexed, 20,000 queries) on five folds, the quantization table, insertion, equal-memory and
ablation runs at 10,000 vectors and one fold, the latency run on WildChat at 20,000 and
WikiAnswers at 8,000 indexed vectors. --num-vectors / --folds override every evaluation (a
reduced run); --results sets where CSVs are written and read.

Memory budgets are B/M, the filter's bytes over the raw float32 index (never a cap count k).
Thresholds are cosine distance delta = 0.02 (experiments.FIXED_DELTA) unless a sweep varies them.
Cap-Cover is built by runner.CAP_COVER_CONSTRUCT (sphere merge); soft anneal is its own arm in
the construction and ablation evaluations.

Evaluations that time something (end-to-end, latency, timing) run after the parallel batch,
alone, on pinned cores: timings taken beside dozens of other jobs read several times too
slow. Running an evaluation truncates its CSVs (see CLAUDE.md); collecting jobs does not.
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

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Optional, Tuple

import experiments
from experiments import (Runner, blocklist_workloads, equal_memory_filters, logger, results_path,
                         generate_blocklist, generate_budget, generate_cap_quantization,
                         generate_construct_methods, generate_equal_memory, generate_streaming,
                         generate_ablation, generate_latency, generate_timing)
from filters.cap_cover.cap_quantizers import Quantize
from plots import plot_paper
from plots.plot_paper import (FIGURES, plot_figures, use_corpora, CONFIG_TO_METHOD, EQUAL_MEMORY_HIDDEN,
                              E2E_TAU, E2E_MEMORY_FRACTION, E2E_TAUS, E2E_MEMORY_FRACTIONS, PAPER_TAUS_COS,
                              l2_threshold)
from runner import CAP_QUANTIZATION_SWEEP, CAP_QUANTIZE_AUTO

# The figures read the thresholds and budgets plot_paper names; the generators use experiments'.
# They are one set of values.
assert abs(E2E_TAU - experiments.FIXED_TAU) < 1e-12
assert all(abs(a - b) < 1e-12 for a, b in zip(E2E_TAUS, experiments.BLOCKLIST_TAUS)) and len(E2E_TAUS) == 8
assert tuple(PAPER_TAUS_COS) == tuple(experiments.BLOCKLIST_DELTAS)
assert E2E_MEMORY_FRACTION == experiments.FIXED_MEMORY_FRACTION
assert tuple(E2E_MEMORY_FRACTIONS) == tuple(experiments.BUDGET_MEMORY_FRACTIONS)
assert len(experiments.BUDGET_MEMORY_FRACTIONS) == len(experiments.CONSTRUCT_MEMORY_FRACTIONS) == 8

# The baselines the two equal-memory figures draw; the rest would be run and discarded.
PAPER_EQUAL_MEMORY_FILTERS = [cls for cls in equal_memory_filters
                              if CONFIG_TO_METHOD.get(cls.__name__) not in EQUAL_MEMORY_HIDDEN]

# The sweeps grid (blocklist, budget) and the throughput figure run CapCoverFilter at this one
# encoding (not auto()'s choice); the equal-memory comparison draws one Cap-Cover line per
# encoding of EQUAL_MEMORY_CAP_QUANTIZES.
PAPER_CAP_QUANTIZE = Quantize.TQ2
EQUAL_MEMORY_CAP_QUANTIZES = (Quantize.TQ2, Quantize.TQ4)

# The encodings the quantization figure compares: the full-precision, scalar and TurboQuant
# rungs down to 1 bit/dim, plus the seeded-PQ rungs auto() chooses among, and auto() itself.
PAPER_QUANTIZE = tuple(dict.fromkeys(
    (Quantize.FP32, Quantize.FP16, Quantize.SQ8, Quantize.TQ8, Quantize.SQ6, Quantize.SQ4, Quantize.TQ4,
     Quantize.TQ3, Quantize.TQ2, Quantize.TQ1) + tuple(CAP_QUANTIZATION_SWEEP) + (CAP_QUANTIZE_AUTO,)))

# The encodings whose one-batch query throughput the quantization table reports (pinned timing).
SCAN_QUANTIZES = tuple(q for q in PAPER_QUANTIZE if q != CAP_QUANTIZE_AUTO)

FULL_SCALE = (100_000, 5)      # vectors per corpus, folds
SMALL_SCALE = (10_000, 1)

END_TO_END_CSV = "end_to_end.csv"
E2E_WORKERS = 8


def generate_end_to_end(runner: Runner, num_vectors: Optional[int] = None, n_folds: Optional[int] = None) -> None:
    """The throughput figure's data: per corpus and fold, a CapCoverFilter (PAPER_CAP_QUANTIZE) in
    front of a flat index against the index alone (runner.end_to_end_worker), on the blocklist
    workload. Only the cells the three panels read: tau swept at the fixed budget
    E2E_MEMORY_FRACTION, and the budget swept at E2E_TAU. Only submits."""
    cells = sorted({(E2E_MEMORY_FRACTION, t) for t in E2E_TAUS} | {(mf, E2E_TAU) for mf in E2E_MEMORY_FRACTIONS})
    num_vectors = num_vectors or experiments.WORKLOAD_NUM_VECTORS
    n_folds = n_folds or experiments.BLOCKLIST_FOLDS
    for workload in blocklist_workloads(num_vectors, n_folds=n_folds):
        for fold, sel in enumerate(workload.selections):
            runner.submit(
                'end_to_end', output_csv=results_path(END_TO_END_CSV),
                dim=384, dataset=workload.plot_key, seed=fold, cells=cells, quantize=PAPER_CAP_QUANTIZE,
                vectors_reference=workload.rows(sel["reference"]),
                vectors_query=workload.query_vectors(sel),
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
    Evaluation("blocklist", partial(generate_blocklist, cap_quantize=PAPER_CAP_QUANTIZE), FULL_SCALE,
               "blocklist.csv", False),
    Evaluation("budget", partial(generate_budget, cap_quantize=PAPER_CAP_QUANTIZE), FULL_SCALE,
               "budget.csv", False),
    Evaluation("construction", generate_construct_methods, FULL_SCALE, "construct_methods/*.csv", True),
    Evaluation("quantization", partial(generate_cap_quantization, quantizes=PAPER_QUANTIZE), SMALL_SCALE,
               "cap_quantization/*/*.csv", True),
    Evaluation("streaming", partial(generate_streaming, construct_methods=[plot_paper.STREAM_BOOTSTRAP],
                                    k_sweep=False), SMALL_SCALE, "streaming/*/streaming.csv", True),
    Evaluation("equal-memory", partial(generate_equal_memory, filters=PAPER_EQUAL_MEMORY_FILTERS,
                                       count_memory_steps=8, tau=l2_threshold(experiments.FIXED_DELTA),
                                       cap_quantizes=EQUAL_MEMORY_CAP_QUANTIZES),
               SMALL_SCALE, "equal_memory/*/equal_memory.csv", True),
    Evaluation("ablation", generate_ablation, SMALL_SCALE, "ablation.csv", False),
    Evaluation("end-to-end", generate_end_to_end, FULL_SCALE, END_TO_END_CSV, False, workers=E2E_WORKERS),
    Evaluation("latency", lambda runner, num_vectors=None, n_folds=None: generate_latency(runner), None,
               "latency.csv", False, workers=1),
    Evaluation("timing", lambda runner, num_vectors=None, n_folds=None: generate_timing(
        runner, **({"num_vectors": num_vectors} if num_vectors else {}), scan_quantizes=SCAN_QUANTIZES), None,
               "timing.csv", False, workers=1),
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
        nv, nf = e.scale or (SMALL_SCALE[0], 1)
        need = max(need, (n_folds or nf) * (num_vectors or nv))
    if "latency" in {e.name for e in evaluations}:
        need = max(need, 25_000)
    problems = []
    for name in experiments.BLOCKLIST_DATASETS:
        path = experiments.dataset_filenames[name]
        if not os.path.exists(path):
            problems.append(f"{name}: {path} not found")
            continue
        rows = len(experiments.VectorSource(name))
        if rows < need:
            problems.append(f"{name}: {rows:,} rows, need {need:,}")
    if problems:
        raise SystemExit("cannot run the paper's evaluations:\n  " + "\n  ".join(problems))


def _submit_and_run(evaluations, num_workers, num_vectors=None, n_folds=None) -> None:
    runner = Runner(num_workers=num_workers)
    for e in evaluations:
        nv, nf = e.scale or (None, None)
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
        for table in (plot_paper.quantization_table, plot_paper.paper_tables):
            try:
                table()
            except FileNotFoundError as e:
                logger.warning(f"{table.__name__}: {e}")


def configure(datasets=None, workers=None, results=None) -> None:
    """Overrides the corpora (and the figures' to match), the pool size and the results folder
    before anything is submitted."""
    if workers:
        experiments.MAX_WORKERS = workers
    if results:
        experiments.RESULTS_DIR = str(results)
    if datasets:
        unknown = [d for d in datasets if d not in experiments.dataset_filenames]
        if unknown:
            raise SystemExit(f"unknown dataset(s) {unknown}; choose from {sorted(experiments.dataset_filenames)}")
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
            scale = f"{e.scale[0]:,} vectors x {e.scale[1]} fold(s)" if e.scale else "fixed in its generator"
            print(f"{e.name:14s} {scale:34s} {e.outputs}" + (f"  [alone, {e.workers} pinned core(s)]" if e.workers else ""))
        return
    configure(args.datasets.split(",") if args.datasets else None, args.workers, args.results)
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
