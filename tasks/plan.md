# Publication cleanup plan

## Objective and scope

Make `src/paper.py` the sole evaluation entry point and remove `src/main.py`.
Keep a small, reusable semantic membership filter framework, with only the
algorithms, evaluations, and presentation code needed to reproduce the paper.
This document plans the work; it does not change implementations or results.

Treat the current `paper.tex` as the intended publication scope, including its
tables and quantitative claims in prose. Neither the current `paper.py` subset
nor the older `paper.pdf` figure list is a complete specification. Resolve
disagreements against the recorded experiment inputs before changing defaults
or claiming that existing results are reproducible.

Assumed use: a researcher running batch experiments locally or on the paper's
server, with separate, isolated timing runs. No service architecture, plugin
system, general experiment DSL, or new framework is needed.

## Findings that determine the plan

1. **`paper.py` is still a wrapper around `main.py`.** It imports generators,
   workload helpers, filters, and constants, then mutates `main` globals in
   `configure()` and `set_thresholds()`. Move the retained definitions out before
   deleting `main.py`; do not rename the entire legacy script to `paper.py`.
2. **The paper entry point is incomplete.** Its six generators cover blocklist,
   budget, quantization, construction, equal-memory, and end-to-end throughput.
   `paper.tex` also reports streaming, ablation, sensitivity, latency, construction
   and insertion timing, and quantization query throughput. Workers and table
   writers already exist for much of this, but the entry point does not wire
   them together. `suggestions/run_reduced_batch.py` even requests a nonexistent
   `GENERATORS["streaming"]`.
3. **Publication settings disagree.** `runner.CAP_COVER_CONSTRUCT` is
   `soft_anneal`; the manuscript says the paper uses sphere merge and reports
   soft anneal as a separate experiment. `main.CONSTRUCT_EVALUATED` excludes
   sphere merge. `plot_paper.PAPER_TAU_COS` and `tables_paper.DELTA` are 0.05,
   while the manuscript generally specifies 0.02. The entry point fixes TQ2 in
   places where the prose describes automatic encoding selection. Preserve the
   original configurations as evidence; decide the publication configuration
   once, then regenerate affected results when needed.
4. **Workloads differ by evaluation.** The manuscript uses 100K vectors/five
   folds for some experiments, 10K/one fold for others, and special latency
   workloads. One global workload size and imported default arguments cannot
   represent that faithfully. Changing a module constant also does not update
   defaults already bound in function definitions.
5. **Submission can erase existing results.** `Runner.submit()` initializes and
   deletes existing output CSVs before execution. `--missing` counts matching
   files rather than validating named corpora or completed experiment cells;
   empty files can satisfy it. It is not reliable resumption.
6. **There are two plotting systems.** Nine old plot modules total 3,073 lines;
   the retained paper renderer has its own figure implementations. Keep only
   helpers still used by paper figures/tables, and remove the legacy modules
   after tracing imports and tests. The old dataset palette also names different
   corpora from the paper.
7. **Two baselines are outside the reported comparison.** RACE is referenced by
   its own tests but not the paper runner; IVF-Radii is explicitly hidden from
   the paper comparison. Both are deletion candidates with their tests and
   registry/label entries. Keep IVF-PQ, which is a different baseline.
8. **Loading evaluation definitions imports dataset preparation.** `main.py`
   imports `load_embeds` from `fetch_datasets.py`, bringing in Hugging Face and
   sentence-transformers. Its shown callers are in legacy query-performance
   code. Retained experiments should use `VectorSource`/HDF5 directly and keep
   dataset preparation separate.

These are the highest-value ponytail-audit findings for publication cleanup.
They are supported by source inspection, not execution of expensive sweeps.

## Retained evaluation contract

Task 1 must fill in exact grids, seeds, folds, encoding choices, constructor,
source data identity, metric columns, and output names for each row below.

| Evaluation | Retain because | Existing implementation/output |
| --- | --- | --- |
| Threshold and memory sweeps | Filtering benefit, FPR bands, saved accesses | `generate_blocklist`, `generate_budget`; `blocklist.csv`, `budget.csv` |
| Quantization | Encoding comparison and rate table | `generate_cap_quantization`; `cap_quantization/<corpus>/*.csv` |
| Construction | Gonzalez, minimax, sphere merge, reported soft anneal comparison | `generate_construct_methods`; `construct_methods/<corpus>.csv` |
| Equal memory | Zero-FN and approximate baseline comparisons | `generate_equal_memory`; `equal_memory/<corpus>/equal_memory.csv` |
| End-to-end throughput | Filter plus flat index versus flat index | `generate_end_to_end`, `end_to_end_worker`; `end_to_end.csv` |
| Streaming insertions | Nearest and cap merge versus static rebuild | `generate_streaming`, `stream_rebuild_worker`; `streaming/<corpus>/streaming.csv` |
| Ablation and sensitivity | Component removal, selection parameters, J-tau correlation | `ablation_worker`; `ablation.csv`, LaTeX tables, correlation CSV |
| Latency | Per-query filter cost and backend break-even | `filter_latency_worker`; `latency.csv`, latency table |
| Pinned timing | Construction, insertion, and encoding query throughput claims | `pinned_timing_worker`; `timing.csv`, timing and quantization tables |

The manuscript also quotes a separate soft-anneal comparison on WildChat and
Quora. Trace its source and retain a minimal reproducible invocation; its current
implementation may differ from the sphere-merge-start variant described in the
text. Do not assume that keeping a class with the same name reproduces the claim.
Hand-drawn explanatory diagrams are retained assets, not evaluation outputs.

## Target structure and boundaries

```text
src/
  paper.py              CLI, paper presets, selection, run/plot orchestration
  experiments.py        retained generate_* functions and explicit settings
  workloads.py          positional splits, ground truth, calibration split
  runner.py             job execution, retained workers, baseline calibration
  vector_source.py      lazy HDF5 and array-backed vector access
  filters/              generic filter interface and reported baselines
    cap_cover/          geometry, construction, encoding, scans, insertion
  plots/
    plot_paper.py       publication figures and input/output manifest
    tables_paper.py     ablation, sensitivity, latency, timing tables
    table_quantization.py
    plot_common.py     only shared helpers still used
  util/
    dataset_paths.py   lightweight dataset registry
    fetch_datasets.py  preparation for the paper's corpora
    space_bound.py     retain only if required by the final paper/artifact
CapCoverRustFilter/     required kernels, reference/parity tests, build metadata
tests/                 tests for retained behavior and publication workflow
```

Dependency direction: `paper -> experiments -> workloads/runner -> filters`.
Rendering reads results and settings; it must not decide what algorithms run.
`runner`, filters, and workload readers do not import `paper` or plotting code.
Keep this separation with ordinary functions and one small explicit settings
record; do not introduce repositories, factories, or a generic configuration engine.

### What stays generic

- Preserve `Filter.construct/query/insert/get_bytesize`, honest byte accounting,
  and `InsufficientMemoryError`. Preserve multiple construction and insertion
  implementations: they are actually compared in the paper.
- Keep dataset identity, vector count, folds, thresholds, budgets, seeds, worker
  count, and output locations as explicit inputs. Dimension comes from data or
  validated settings; 384 is a paper preset, not a requirement for every filter.
- Keep the lightweight dataset registry and ability to supply another HDF5
  corpus without editing algorithm code. Ship preparation recipes for the five
  paper corpora; remove unrelated builders after checking consumers.
- Keep `VectorSource`, thread limits before numerical imports, exact distance
  labels, stratified calibration, atomic CSV replacement, invariant failures,
  and timing isolation. These solve real evaluation requirements.
- Retain Rust kernels reachable from retained methods and the numerical
  reference/fallback paths needed for portability and soundness checks.
  `construct_tail`, pruned Gonzalez, and AVX kernels have live Python callers;
  they are not dead just because they originated as optimization experiments.

### What goes

- `src/main.py`, after extracting retained code and migrating every caller.
  No permanent compatibility shim or duplicate CLI.
- Generators and worker registrations for system-tau, legacy query performance,
  scaling query/insert/construction, and construct-method-scale, once the
  evaluation contract confirms no retained output or claim requires them.
- Old `save_*` DataFrame wrappers, legacy `plot_*` wrappers, and their helpers
  when no retained caller remains; CSV persistence already belongs to `Runner`.
- `plot_blocklist.py`, `plot_budget.py`, `plot_cap_cover_system_tau.py`,
  `plot_cap_quantization.py`, `plot_construct_methods.py`, `plot_equal_memory.py`,
  `plot_query_performance.py`, `plot_scaling.py`, and `plot_streaming.py`, after
  migrating any genuinely shared behavior and associated useful tests.
- RACE and IVF-Radii implementations and dedicated tests, unused labels,
  registry entries, tuning dials, stale dataset lists, and old compatibility
  branches shown to serve only retired results.
- Scratch drivers, hierarchy prototypes, exploration benchmarks, stale plans,
  logs, caches, and local skill copies from the publication distribution.
  Promote any experiment still needed to reproduce a claim before excluding
  its scratch source. Preserve a small Rust build/parity check and any timing
  benchmark actually used to calibrate retained behavior.

“Goes” means removal from the maintained/publication tree after dependency
verification. Existing datasets, results, manuscript files, and untracked work
must not be erased as part of cleanup. Historical artifacts can remain outside
the release; do not build an in-repo archive of deprecated executable code.

## Implementation order

Tasks and their acceptance criteria are in `tasks/todo.md`.

1. Freeze the paper-to-experiment inventory and provenance.
2. Make job collection safe and publication completeness verifiable.
3. Extract shared workloads without changing their numerical behavior.
4. Extract retained generators with explicit settings.
5. Wire missing figure/table experiments into `paper.py`.
6. Finish the CLI migration and delete `main.py`.
7. Consolidate paper rendering and remove legacy plot modules.
8. Remove unused workers, baselines, and legacy-only tests/helpers.
9. Prune construction/encoding options only after tracing retained variants.
10. Separate preparation dependencies and trim dataset builders.
11. Prepare the publication tree and update documentation.
12. Validate the complete release from a clean environment.

Keep extraction and scientific configuration changes separate. First prove
equivalence for unchanged experiment cells; then apply the reconciled paper
settings and identify exactly which outputs require regeneration. Timing claims
must be remeasured if cleanup changes the timed execution path.

## Required behavior after cleanup

- `python src/paper.py` selects all publication experiments using their correct
  per-evaluation presets. `--only` selects known experiments and rejects typos.
- `python src/paper.py plot` renders all required figures and tables from a
  specified result root, without loading datasets or executing workers.
- Use stdlib `argparse` instead of `_parse`/`_int`. Keep existing useful options,
  add explicit folds/output control where needed, and document per-evaluation
  defaults versus user overrides. No mutation of another module's globals.
- Use one declared mapping from evaluation names to generators and expected
  outputs. Validate exact corpus/cell identities and successful completion;
  file count alone is insufficient. Keep the existing plot hash manifest and
  record effective experiment settings and failures in a small run manifest.
- Job collection has no filesystem side effects. Validate selected inputs and
  output policy before writing. Use an explicit result root for generation as
  well as plotting, so small test runs never overwrite publication results.
- `--missing` selects incomplete evaluations, reruns them in full, and clearly
  states that behavior. Do not add cell-level resume/deduplication machinery.
- Required job failures or missing cells make a publication run fail, while
  preserving successful rows. Keep best-effort partial plots available through
  an explicit option if still useful; they cannot count as a complete release.
- Timed work runs separately, sequentially on one pinned core for the paper's
  measurements. Report unsupported affinity honestly; never label such a run
  as pinned. Parallel quality sweeps keep bounded worker/thread counts.

## Verification and release criteria

During implementation, use temporary directories and synthetic/small HDF5
fixtures. Never invoke the current real generators as a dry run against the
repository's `results/` directory.

1. Baseline the existing focused tests before edits and record pre-existing
   failures. Preserve meaningful zero-FN, coverage, byte-budget, calibration,
   streaming, atomic-write, Rust parity, and paper presentation tests.
2. For moved code, compare job arguments, fold selections, schemas and
   deterministic non-timing metrics against the frozen baseline. Add only
   focused tests for missing guarantees: explicit settings propagation, safe
   collection, completeness validation, and one small end-to-end publication run.
3. Migrate `tests/test_capcover_wiring.py` away from parsing `main.py` and
   asserting retired outputs. Validate retained experiment submissions through
   a recording runner and real small fixture instead.
4. Run `python -m pytest tests`, build/test the Rust crate, and exercise the CLI
   from a clean Python 3.12 environment with the extension built for that Python.
   Resolve FAISS capabilities and dependency versions from the working setup;
   verify the documented install in a fresh environment.
5. Render every required figure and table from an isolated, fixed result
   snapshot. Check expected corpora/cells, units, mean/std aggregation, labels,
   file names, and manuscript references, including inlined table values.
6. Perform full scientific reruns on the appropriate machine where settings or
   provenance changed. Small fixtures establish wiring, not paper-scale numbers.
   Re-run pinned timings separately and record hardware/build identity.
7. Release only when `main.py` has no remaining callers, every reported claim
   has a source and reproduction command, and no excluded path is required by
   the documented workflow. Keep the scientific settings and source hashes
   beside the released results.

## Expected simplification and limits

The nine legacy plot modules alone offer roughly 3,073 lines for removal before
allowing for shared helpers. Removing legacy evaluation paths, two baselines,
their dedicated tests, and scratch tooling should reduce the tree further.
Do not count moving retained definitions out of `main.py` as deleted complexity.
No dependency reduction is promised: the reported baselines still need SciPy,
scikit-learn, FAISS, and `pybloom_live`; embedding dependencies can be optional
for evaluation-only installs. CPU affinity also needs an explicit dependency
decision because `runner` imports `psutil` on some platforms but requirements
does not list it.

Source inspection covered entry points, manuscript evaluation sections,
generator/worker registries, output handling, plotting/table paths, dependency
declarations, and representative tests and kernel callers. It was not a complete
numerical or performance audit of every Python/Rust implementation. No tests,
dataset fetches, or evaluation sweeps were run while preparing this plan.
