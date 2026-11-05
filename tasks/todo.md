# Publication cleanup tasks

Implemented (see git log): tasks 2-8, 10, 11 and the Python side of 12. Not done: task 9 beyond removing
functions with no caller (option trimming needs the paper-scale ablation arms), the Rust build/tests and
the paper-scale rerun, which need the evaluation machine.
All generator verification uses a recording runner or temporary output root.

## 1. Freeze the evaluation contract

- [ ] Map every figure, table, and quantitative prose claim in `paper.tex` to
  its generator/worker, source CSV, exact workload, constructor, encoding,
  threshold, budget, folds, seeds, and hardware requirements.
- [ ] Record discrepancies and source hashes; resolve sphere merge versus
  soft anneal, fixed versus auto encoding, 0.02 versus 0.05, byte budgets versus
  FP32-equivalent cap percentages, and the separate soft-anneal experiment.
- [ ] Establish required artifacts and distinguish historical results from
  outputs needing regeneration; retain current files intact.

**Verify:** Cross-check each mapping against source and recorded result metadata;
unknown provenance stays marked unresolved, never inferred from file names.
**Dependencies:** None. **Scope:** `tasks/plan.md`, one compact evaluation inventory.

## 2. Make evaluation collection safe

- [ ] Move output initialization out of `Runner.submit()`; collecting jobs
  cannot modify or delete a CSV. Preflight selected inputs/outputs before runs.
- [ ] Add generation result-root plumbing, a small settings/completion manifest,
  and exact expected-cell checks; make required job failures visible to the CLI.
- [ ] Preserve atomic CSV updates, successful rows on failure, and invariant
  propagation; keep `--missing` as whole-evaluation reruns, not resumable cells.

**Verify:** Focused tests for side-effect-free collection, empty/wrong-corpus and
partial outputs, ordinary/invariant failures, and existing atomic-write behavior.
**Dependencies:** 1. **Scope:** `runner.py`, `paper.py`, runner/publication tests.

## Checkpoint A

- [ ] No scientific defaults changed; result files remain intact.
- [ ] Collection/output tests pass; incomplete runs cannot look complete.

## 3. Extract workload handling

- [ ] Move positional fold selection, HDF5 reads, exact ground truth, and
  stratified calibration into `src/workloads.py` with explicit inputs.
- [ ] Keep array-backed small fixtures and lazy `VectorSource` support; remove
  obsolete cluster terminology and helpers only after checking callers.
- [ ] Migrate retained generators to the shared functions without altering
  folds, row order, distance labels, or calibration selection.

**Verify:** Compare old/new selections on small HDF5 data, check disjoint folds
and short-file failures, and run ground-truth and calibration tests.
**Dependencies:** 1–2. **Scope:** `workloads.py`, `main.py`, workload tests.

## 4. Extract retained experiment definitions

- [ ] Move blocklist, budget, quantization, construction, equal-memory, and
  streaming generators to `src/experiments.py`.
- [ ] Pass settings explicitly; remove dependence on plot constants, mutated
  globals, and import-bound defaults. Keep a simple settings record and registry.
- [ ] Preserve submitted cells/schemas before applying paper-setting changes
  separately, with an explicit list of affected outputs.

**Verify:** A recording runner matches baseline submission arguments for fixed
cells; two settings objects can be used independently in one process.
**Dependencies:** 3. **Scope:** `experiments.py`, `main.py`, `paper.py`, focused tests.

## 5. Complete the publication experiments

- [ ] Wire streaming, ablation/sensitivity, latency, pinned construction/insertion
  and query timing to existing workers using the inventory's per-evaluation sizes.
- [ ] Retain a reproducible path for the reported soft-anneal comparison; resolve
  its starting cover and parameters against the manuscript and source evidence.
- [ ] Declare expected outputs for all evaluations and isolate timed work on
  one supported pinned core, separately from parallel quality sweeps.

**Verify:** Generate one small cell per retained mode into a temporary root;
check schemas, zero-FN where guaranteed, byte budgets, and timing selection.
**Dependencies:** 4. **Scope:** `experiments.py`, `paper.py`, `runner.py`, paper tests.

## Checkpoint B

- [ ] Every retained publication claim has a generator or documented static asset.
- [ ] Unchanged cells match the baseline; changed scientific settings are explicit.

## 6. Make paper.py the sole CLI

- [ ] Replace handwritten argument parsing with `argparse`; retain useful
  selection/plot options and expose explicit folds and result locations.
- [ ] Remove `import main` and mutation-based configuration; plotting requires
  no datasets. Update consumers and wiring tests to use retained definitions.
- [ ] Delete `src/main.py` once all retained definitions and callers have moved.

**Verify:** CLI help/errors, subset selection, settings overrides, plot-only mode,
and repository search for remaining executable `main` imports/path references.
**Dependencies:** 4–5. **Scope:** `paper.py`, `main.py`, wiring/publication tests.

## 7. Consolidate rendering

- [ ] Render all required paper figures and tables through the publication CLI;
  reconcile output filenames, default thresholds, corpus names, and figure mapping.
- [ ] Reduce `plot_common.py` to retained helpers, then remove the nine legacy
  plot modules in two small batches, checking imports between batches.
- [ ] Preserve input/output hashes and correct metric aggregation; move useful
  legacy plot tests to the retained renderer instead of dropping their guarantees.

**Verify:** Existing presentation/table tests and an isolated complete render;
check manuscript references and table numbers against generated artifacts.
**Dependencies:** 6. **Scope:** plot modules in batches of at most five files;
then CLI/table and test updates as a separate slice.

## 8. Prune obsolete evaluation code and baselines

- [ ] Remove unneeded system-tau, query-performance, scaling, and
  construct-method-scale workers/registrations plus legacy-only helpers.
- [ ] Remove RACE and IVF-Radii with dedicated tests and all configuration,
  calibration, and presentation entries; preserve IVF-PQ and reported baselines.
- [ ] Remove unused save/plot wrappers and stale metadata/schema compatibility
  only after searching code, tests, scripts, and string references.

**Verify:** Full retained Python suite and small equal-memory comparison;
confirm every remaining worker registration has a retained caller.
**Dependencies:** 6–7. **Scope:** worker pruning and each baseline removal are
separate slices of at most five files.

## Checkpoint C

- [ ] One CLI and one publication plotting path remain.
- [ ] Reported baselines and scientific guarantees still have meaningful tests.

## 9. Trim unused algorithm options conservatively

- [ ] Trace constructor, encoding, insertion, and Rust exports from retained
  experiments and their ablation arms before proposing deletions.
- [ ] Delete only variants/toggles proven irrelevant; preserve numerical
  reference paths, portability fallbacks, and live optimized kernels.
- [ ] Retain generic filter/strategy interfaces used by compared methods;
  simplify redundant delegation without inventing replacement abstractions.

**Verify:** Cap coverage, quantization soundness, adversarial construction,
insertion memory, and Python/Rust parity tests for every retained path.
**Dependencies:** 8. **Scope:** Work method by method in slices of at most five files.

## 10. Separate preparation from evaluation dependencies

- [ ] Remove heavy preparation imports from evaluation/worker paths; keep
  registry and HDF5 loading import-light.
- [ ] Retain builders for the five paper corpora and generic HDF5 input support;
  remove unrelated dataset recipes and stale defaults after caller checks.
- [ ] Separate runtime, dataset-preparation, and test dependencies without
  removing baseline requirements; document/build the Rust extension and resolve
  FAISS and CPU-affinity requirements in a tested environment.

**Verify:** Clean evaluation-only install imports/plots without loading the ML
preparation stack; preparation recipes preserve required HDF5 schema and metadata.
**Dependencies:** 8–9. **Scope:** requirements, dataset utilities, import smoke check.

## 11. Prepare the publication tree and documentation

- [ ] Promote any necessary scratch reproduction code, then exclude obsolete
  benchmarks, hierarchy prototypes, local skills, logs, caches, and stale plans
  from the release without deleting local untracked work or scientific evidence.
- [ ] Update README and repository guidance for the actual layout, setup,
  per-evaluation workloads, output policy, commands, and hardware requirements.
- [ ] Keep manuscript/static assets and result provenance in the release;
  check ignore rules do not exclude required figures, tables, or manifests.

**Verify:** Build a clean release staging tree and follow its README; no command
depends on `main.py`, scratch paths, or untracked local artifacts.
**Dependencies:** 7–10. **Scope:** README, repository guidance, ignore/release rules;
scratch exclusions reviewed in small groups.

## 12. Validate the publication release

- [ ] Run `python -m pytest tests`, Rust tests/build, and a small complete run
  from a clean environment into an isolated output root.
- [ ] Render the full artifact inventory and compare non-timing outputs against
  unchanged-cell baselines; regenerate scientifically changed cells at paper scale.
- [ ] Remeasure affected timings under the documented pinned conditions and
  verify every reported figure/table/prose value against its source manifest.

**Verify:** Release inventory has no missing cells/assets, all checks pass, and
remaining provenance limitations are resolved or explicitly documented before release.
**Dependencies:** 11. **Scope:** Validation artifacts and necessary focused fixes.

## Final checkpoint

- [ ] `paper.py` fully replaces `main.py`; reusable filter/workload/runner APIs remain.
- [ ] Only paper-relevant implementations and support code ship.
- [ ] Complete outputs, environment, settings, source hashes, and reproduction
  commands support the publication's scientific claims.
