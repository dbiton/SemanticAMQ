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
    quantization  cap_quantization/<corpus>/*.csv          generate_cap_quantization
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

from functools import partial

import main
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


PAPER_K_LADDER = [2 ** i for i in range(11, 17)]    # main.generate_cap_quantization's: 2.6-82% of |B| = 80,000


def generate_quantization(runner: Runner) -> None:
    """main.generate_cap_quantization with its k ladder scaled to the workload
    size, so a smaller run (--num-vectors) still has k below the reference set."""
    scale = main.WORKLOAD_NUM_VECTORS / 100_000
    generate_cap_quantization(runner, k_values_nominal=[max(1, round(k * scale)) for k in PAPER_K_LADDER])


PAPER_GENERATORS = (
    generate_blocklist,
    generate_budget,
    generate_quantization,
    generate_construct_methods,
    partial(generate_streaming, construct_methods=[STREAM_BOOTSTRAP], k_sweep=False),
    partial(generate_equal_memory, filters=PAPER_EQUAL_MEMORY_FILTERS),
    generate_end_to_end,
)


def generate_and_plot_all(out=None) -> None:
    """Submit the paper's evaluations to one shared Runner, run them as one
    shuffled batch (see main.generate_and_plot_all), then draw the figures.
    Each generator is guarded so one failing to submit doesn't take down the
    others."""
    runner = Runner(num_workers=main.MAX_WORKERS)
    for generate in PAPER_GENERATORS:
        name = getattr(generate, "__name__", None) or generate.func.__name__
        try:
            generate(runner)
        except Exception:
            logger.exception(f"{name} failed to submit")
    runner.shuffle()
    runner.run()
    logger.info("All submitted jobs completed.")
    plot_all(out=out)


def plot_all(figures=None, out=None, results=None) -> None:
    """Re-draws the paper's figures (default all) from results/."""
    plot_figures(figures, results=results, out=out)


def configure(num_vectors=None, datasets=None, workers=None) -> None:
    """Overrides main's workload size, corpora and pool size (and the figures'
    corpora to match) before anything is submitted."""
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
