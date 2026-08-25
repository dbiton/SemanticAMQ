# SemanticAMQ (SAMQ)

An approximate-membership-query (AMQ) filter for semantic near-duplicate
detection over dense embedding vectors. SAMQ (`BallCoverFilter`,
[src/filters/ball_cover/](src/filters/ball_cover/)) sits in front of an
exact backend index (e.g. a flat FAISS `IndexFlatL2`) and answers "is
anything within distance tau of this query?" cheaply, so the expensive
backend is only hit when the filter can't rule the query out — with a
guaranteed **no false negatives** invariant, unlike ANN indexes.

## How it works

SAMQ builds a Gonzalez-style greedy cover of enclosing balls over the
construct set (radius per ball, quantizable to shrink memory — see
`Quantize` in [ball_cover_filter.py](src/filters/ball_cover/ball_cover_filter.py)),
then answers a query by checking whether it falls within any ball's
radius (expanded by the query threshold tau). It's compared against a set
of AMQ/ANN baselines implemented in [src/filters/](src/filters/):
DSH, DSBF, LSBF, a CoveringLSH fallback, and FAISS's HNSW and IVF indexes.

## Repository layout

```
src/
  main.py            entry point: defines and runs each evaluation
  runner.py           parallel job harness (ProcessPoolExecutor) + per-job workers
  filters/            SAMQ (ball_cover/) and baseline filter implementations
  plots/               one module per evaluation's figure set
  util/                dataset path registry + embedding loaders
  vector_source.py     lazy/array-backed vector handles passed to worker processes
tests/                 unittest-style tests, one file per filter/module
datasets/              embedding pickles (not checked in, see below)
results/               generated evaluation CSVs
plots/<eval>/           generated figure PNGs
```

## Setup

Dependencies are managed with [uv](https://github.com/astral-sh/uv):

```
uv venv
uv pip install -r requirements.txt
```

Evaluations expect pre-computed embedding pickles under `datasets/`
(see `dataset_filenames` in [src/util/dataset_paths.py](src/util/dataset_paths.py)),
one per dataset (ELI5, MMLU, WildChat, MS MARCO, HotPotQA, and a few
others not currently swept). These aren't checked into the repo.

## Running an evaluation

```
uv run python src/main.py
```

`main.py`'s `generate_and_plot_all()` runs the currently-enabled
evaluation(s) end-to-end (generate results, then render figures); see the
`generate_*`/`plot_*` function pairs in [src/main.py](src/main.py) for the
full set (system_tau, ablation, query_performance, scaling, streaming,
equal_memory, static, dynamic). `plot_all()` re-renders figures from a
prior run's `results/*.csv` without re-running the (expensive) sweeps.

## Running tests

```
uv run pytest
```

## Outputs

Each evaluation writes a long-format CSV to `results/` and PNG panels to
`plots/<evaluation>/`, one line/series per dataset or method per figure.
