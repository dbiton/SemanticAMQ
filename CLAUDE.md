# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code for the paper *Geometric Filters for Semantic Approximate
Membership Queries* ([paper.tex](paper.tex), VLDB format). The filter answers
"is anything already indexed within distance tau of this query vector?" over
dense, L2-normalized sentence embeddings (d=384), in front of an exact backend
(flat FAISS index). The invariant is **zero false negatives**; FPR under a byte
budget is the quality metric.

The method is **Cap-Cover** (`CapCoverFilter`, [src/filters/cap_cover/](src/filters/cap_cover/)):
cover the construct set with k spherical caps on the unit sphere, keep
compressed cap directions and per-cap angles, and accept a query if it lies
within a cap's angle (widened by tau). The earlier ball-based `BallCoverFilter`
has been removed; every evaluation uses `CapCoverFilter`.
Everything else in [src/filters/](src/filters/) is a competitor baseline.

[README.md](README.md) is partly out of date (it lists DSBF/LSBF/CoveringLSH,
which no longer exist, and says to use uv). Trust the code and this file. Dependencies are in `requirements.txt` (no
lockfile or linter config).

## Environment and commands

Use the **global Python 3.12** (`python` on PATH, faiss 1.15.1), not `.venv`
(Python 3.10, faiss 1.14.3). The faiss versions differ in capability (e.g.
`HadamardRotation.reverse_transform` only works in 1.15.1).

Run everything from the **repo root**: dataset paths are relative
(`./datasets/...`) and CSVs go to `results/` (`--results DIR` to change) and figures to `figures/`.

```bash
python -m pytest tests                          # all tests
python -m pytest tests/test_cap_cover_filter.py -k name   # one test
python src/paper.py                             # full run: evaluations, then figures
python src/paper.py plot                        # re-draw figures from existing results/
PYTHONPATH=src python some_script.py            # ad-hoc scripts importing project modules
```

`tests/conftest.py` puts `src/` on `sys.path`; tests are `unittest`-style
classes run through pytest. Imports inside `src/` are top-level
(`from filters.filter import Filter`, `from runner import Runner`), never
`src.`-prefixed.

Full evaluation runs happen on a separate machine (Threadripper 3970X, 32 cores / 64 threads, 896 GB RAM,
`MAX_WORKERS=30` (one per physical core, less two), `WORKLOAD_NUM_VECTORS=100_000`; paper.py times its throughput
evaluation afterwards on 8 pinned cores). The local box (64 GB, 28 threads, i7-14700 with P/E cores)
is for tests and single-cell validation. Size per-job memory against
~10 GB/job, not local RAM.

## Layout

```
src/experiments.py     evaluation definitions: generate_* per evaluation (submit jobs only), workload
                       construction (blocklist_workloads), output paths
src/paper.py           the sole entry point (replaced main.py): EVALUATIONS (scale per paper.tex, outputs, timed or
                       not), argparse CLI (`--list`, `--only`, `--missing`, `plot`, `--results`, `--num-vectors`),
                       then the figures. Builds the Rust kernels first (rust_build.py). Reduced run: `--num-vectors 10000 --folds 1 --results DIR`
src/runner.py          Runner (ProcessPoolExecutor job harness), one *_worker per mode,
                       filter registries, calibration/tuning of baseline dials
src/filters/filter.py  Filter base: construct(), query(), insert() (optional), get_bytesize();
                       InsufficientMemoryError
src/filters/cap_cover/
  cap_cover_filter.py  CapCoverFilter + Quantize enum (direction encodings), auto() encoding choice
  cap_interfaces.py    ConstructInterface / InsertInterface strategy ABCs
  cap_geometry.py      cap measure / angle helpers
  *_construct.py       construction strategies (gonzalez [file: gonzales_construct.py],
                       minimax, sphere_merge, soft_anneal = sphere merge + annealed refinement, ~5-13x the cost)
  *_insert.py          dynamic-insertion strategies (nearest, cap_merge)
  (CAP_COVER_EXPERIMENTS.md at the repo root: what was tried on CapCoverFilter and why variants were removed)
  cap_scan.py          query() path: accept-rule constants (_beta) and the bridge to the
                       Rust scan (freeze/thaw of the codes); float64 fallback for FP32/FP16/SQ*
  scan_kernels.py      numba helper for construction planning (nearest_codeword)
CapCoverRustFilter/    Rust crate capcover_rs (PyO3/maturin) behind CapCoverFilter.query();
                       rust-improve.md = design/benchmark report. Build into the Python in use:
                       `pip install maturin && cd CapCoverRustFilter && maturin develop --release`
src/filters/           baselines: dsh, hnsw, ivf, quantized_vector, va_plus, mlp,
                       pacuk_lsh / sankowski_wygocki (zero-FN LSH, on _floor_lsh.FloorLSHFilter)
src/plots/             plot_paper.py (the paper's figures)
src/util/dataset_paths.py  dataset name -> HDF5 file registry (deliberately import-light)
src/util/fetch_datasets.py builds datasets/ with sentence_transformers (heavy imports)
src/vector_source.py   VectorSource: picklable lazy handle to an HDF5 slice for workers
tests/                 one file per filter/module
suggestions/           untracked scratch analyses and plans (not part of the pipeline)
```

## Architecture

**Evaluation flow.** Each evaluation in `experiments.py` has a `generate_X(runner, num_vectors, n_folds)`
that only *submits* jobs (`runner.submit(mode, output_csv=..., **kwargs)`); `paper.EVALUATIONS`
lists them: blocklist, budget, construction, quantization, equal-memory and end-to-end, all on paper.tex's
workload (100K vectors x 5 folds: 80K indexed, 20K queries) at tau = 0.05 cosine distance, Cap-Cover built by
soft anneal with auto encoding. The untimed ones run as one shuffled batch on `experiments.MAX_WORKERS`;
end-to-end runs afterwards on pinned cores (paper.tex: one core, one job at a time). `python src/paper.py plot` re-renders from existing CSVs.

**Runner.** Modes map to workers in `runner._WORKER_FNS`. Each job's rows are
written incrementally to its `output_csv` by the coordinating process as jobs
finish, so a crash only loses unfinished jobs. A failing job is logged and
skipped. `run()` returns `(mode, result)` pairs **in completion order**:
workers must return self-describing rows, and callers must never zip results
positionally against submission order (each row names its dataset, config and fold).

**Registries** (check these for current names instead of hardcoding lists):
- `runner.CAP_CONSTRUCT_METHODS`, `CAP_COVER_CONSTRUCT = "soft_anneal"`: construction strategies by name.
- `runner.CAP_INSERT_STRATEGIES`: insertion strategies by name.
- `runner._cap_cover(dim, k)`: the paper's configuration (`CapCoverFilter.auto()`).
- `runner.CAP_QUANTIZATION_SWEEP`, `_POST_CONSTRUCTION_CONFIG_DIALS`,
  `_CONSTRUCTION_TIME_CONFIG_DIALS`: per-baseline tuning dials.
- `util.dataset_paths.dataset_filenames`, `experiments.BLOCKLIST_DATASETS`,
  the figures' `plots.plot_paper.CORPORA`: which corpora exist vs. which each evaluation sweeps.

**Filter construction** funnels through `runner._instantiate_filter` and
`runner._build_calibrated_filter`. A filter that cannot fit its construct set
in `max_memory` must raise `InsufficientMemoryError`; the equal-memory worker
records it as an infeasible (NaN) row. Never ship an over-budget index.

**Scan cost.** At a byte budget a query scan costs budget x ns/byte of the encoding
(`cap_cover_filter.SCAN_NS_PER_BYTE`; measure with `CapCoverRustFilter/bench/ns_per_cap.py`):
TurboQuant 0.1-0.3, seeded PQ 0.4-1.9. `CapCoverFilter.auto(scan_limit_ms=...)` (default: env
`CAPCOVER_SCAN_LIMIT_MS`, set to 1.0 by paper.py) drops candidates above the limit.

**Query bands.** FPR is also split by how related a query is to the indexed set: the
cosine similarity to its nearest construct vector, in `runner.SIMILARITY_BANDS`
(close >= 0.8, related 0.6-0.8, unrelated < 0.6; columns `fpr_<band>`, `n_neg_<band>`).
Bands only group results; labels are always distance at tau.

**Workload sizing.** `experiments.WORKLOAD_NUM_VECTORS` is the single size knob for
every evaluation (80% construct / 20% queries). Ground truth is labelled by
true distance at tau, never by which sampling stratum a query came from.

## Rules and gotchas

- **Running deletes outputs.** `Runner.submit()` only collects jobs; `Runner.run()` truncates each
  distinct `output_csv` before executing (`Runner(truncate=False)` appends). So running an evaluation
  rewrites its CSVs in `--results` (default `results/`, gitignored, a copy of the run machine's output, no
  backup; `results-newer/`, `results-midway/`, `results-stale/` are *different* runs). Use `--results` to
  point a test run elsewhere. Calling `generate_*` with a recording runner is safe.
- **Thread guard before numpy/faiss.** `paper.py`, `experiments.py` and `runner.py` set
  `OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=1` before importing numpy/faiss. Any new
  entry point that uses `Runner` must do the same first, or BLAS threads x
  worker processes oversubscribe the machine.
- **Jobs carry lazy handles, never arrays.** `workloads.Fold` holds two `VectorSource` slices (picklable
  pointers into the HDF5 file); workers read their own slice, and `runner._to_construct_array` rejects
  anything else. Ground-truth labels and the equal-memory calibration split are computed once per fold in
  the generator and passed as small bool/index arrays. Tests build handles with `tests.helpers.lazy`.
- **Keep worker imports light.** `dataset_paths.py` and `vector_source.py` must
  not import torch/sentence_transformers/datasets (~400 MB RSS per worker).
  Pass `VectorSource` handles to workers rather than large arrays. Importing
  `cap_cover_filter` loads numba and its compiled kernels (~0.6 s, ~80 MB
  RSS from numba's cache; ~1.9 s the first time on a machine).
- **Dataset files are HDF5 despite `.pkl`.** Open with `h5py`; vectors are
  under `HDF5_DATASET_KEY = "normalized_embeds"`. `datasets/` is not checked in.
- **`max_memory` is bytes, not caps.** For the nominal k-cover budget use
  `runner._cover_max_memory(dim, k)`. Stateful insert strategies
  charge their bookkeeping via `InsertInterface.max_caps_for_budget`, so they
  get fewer caps for the same budget; `max_memory` bounds `get_bytesize()` over
  a whole stream, not just after `construct()`.
- **FNR > 0 is a bug, not a data point.** Several workers raise `ValueError` if
  the no-false-negative invariant is violated. Treat such a failure as breakage.
- **Baselines are implemented as published.** Add only what a byte-budgeted
  membership filter forces (accept rule, honest memory accounting, shared
  coverage epsilon, the harness's usual per-filter dial). Never add adaptations
  that make a competitor stronger. The flip side: run Cap-Cover in its best
  configuration (`_cap_cover`, soft_anneal), not the Gonzalez default.
- **Timings need pinning.** Construction timings measured under many workers
  read 3-14x slow; unpinned runs vary 1.5-2x from E-core scheduling. For timing
  claims run sequentially with `psutil.Process().cpu_affinity([0])`.
- **Never `git worktree remove` a worktree whose `datasets/` is a junction or symlink to the repo's.** On Windows it deleted
  the repo's own `datasets/` contents once (they are not recoverable here: only 100K-row copies of 9 corpora exist elsewhere).
  Copy the one or two files a throwaway worktree needs instead, or `rmdir` the link first.
- **Outputs are gitignored** (`/results/`, `/plots/**` except `.gitkeep`,
  `*.csv`, `*.png`, `*.npz`, `*.npy`). When reporting a result, say which file
  it lives in; scratchpad files are temporary.
