"""Generates the data behind paper.tex's figures, and the figures: main.py's
counterpart for the paper alone. It submits only the evaluations the paper
plots, and within them only the cells the figures read, then draws
plots/plot_paper.py's figures (written to suggestions/figures/).

    python src/paper.py              # generate_and_plot_all(): full run
    python src/paper.py plot         # re-draw the figures from existing results/

Each figure and what feeds it (the CSVs are main.py's, so a paper run and a
main.py run write the same files -- and truncate them, see CLAUDE.md):

    sweeps        budget.csv, blocklist.csv                generate_budget, generate_blocklist
    quantization  cap_quantization/<corpus>/*.csv          generate_cap_quantization
    construction  construct_methods/<corpus>.csv           generate_construct_methods
    streaming     streaming/<corpus>/streaming.csv         generate_streaming (position sweep,
                                                           sphere_merge bootstrap only)
    equal-memory  equal_memory/<corpus>/equal_memory.csv   generate_equal_memory (the baselines
                                                           the figure shows only)

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
from main import (Runner, MAX_WORKERS, CAP_COVER_CONSTRUCT, equal_memory_filters, logger,
                  generate_blocklist, generate_budget, generate_cap_quantization,
                  generate_construct_methods, generate_streaming, generate_equal_memory)
from plots.plot_paper import plot_figures, CONFIG_TO_METHOD, EQUAL_MEMORY_HIDDEN, STREAM_BOOTSTRAP

assert STREAM_BOOTSTRAP == CAP_COVER_CONSTRUCT, "the streaming figure bootstraps with the paper's construct method"

# The baselines equal-memory-fpr-fnr-vs-budget draws; the rest of
# main.equal_memory_filters would be run and discarded.
PAPER_EQUAL_MEMORY_FILTERS = [cls for cls in equal_memory_filters
                              if CONFIG_TO_METHOD.get(cls.__name__) not in EQUAL_MEMORY_HIDDEN]

PAPER_GENERATORS = (
    generate_blocklist,
    generate_budget,
    generate_cap_quantization,
    generate_construct_methods,
    partial(generate_streaming, construct_methods=[STREAM_BOOTSTRAP], k_sweep=False),
    partial(generate_equal_memory, filters=PAPER_EQUAL_MEMORY_FILTERS),
)


def generate_and_plot_all() -> None:
    """Submit the paper's evaluations to one shared Runner, run them as one
    shuffled batch (see main.generate_and_plot_all), then draw the figures.
    Each generator is guarded so one failing to submit doesn't take down the
    others."""
    runner = Runner(num_workers=MAX_WORKERS)
    for generate in PAPER_GENERATORS:
        name = getattr(generate, "__name__", None) or generate.func.__name__
        try:
            generate(runner)
        except Exception:
            logger.exception(f"{name} failed to submit")
    runner.shuffle()
    runner.run()
    logger.info("All submitted jobs completed.")
    plot_all()


def plot_all(figures=None) -> None:
    """Re-draws the paper's figures (default all) from results/."""
    plot_figures(figures)


if __name__ == "__main__":
    if sys.argv[1:2] == ["plot"]:
        plot_all(sys.argv[2:])
    else:
        generate_and_plot_all()
