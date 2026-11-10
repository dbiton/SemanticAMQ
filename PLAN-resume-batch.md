# Plan: finish the untimed batch that was stopped on 2026-11-09

Stopped run: `paper.py --pin --workers 51 --timing-mode concurrent` (started 2026-11-07 19:07 UTC, stopped 2026-11-09
~23:10 UTC) into `results-epyc-concurrent-20261107/`. Snapshot at stop: `results-epyc-partial-20261109/`.
End-to-end was run separately afterwards (complete, 375 rows) and is NOT part of this plan.
Gap counts below come from `suggestions/resume_gaps.py` (read-only; run with SEMANTICAMQ_LOG_CLEARED=1 so it does
not truncate log.txt).

## What is missing (job = one submitted runner job; rows are written only when a job finishes, so no partial jobs)

| evaluation | done / total jobs | missing | how a job is identified in the CSV |
|---|---|---|---|
| blocklist (25 = 5 corpora x 5 folds) | 17 / 25 | 8 | `dataset`, `seed` |
| budget | 20 / 25 | 5 | `dataset`, `seed` |
| construction (3 methods x 8 budgets x 5 folds x 5 corpora) | 450 / 600 | 150 | `fold`, `method`, `memory_fraction` (per-corpus file) |
| quantization (17 encodings x 8 budgets x 5 folds x 5 corpora) | 2446 / 3400 | 954 | `seed`, `quantize`, `memory_fraction` (per-corpus file) |
| equal-memory (9 filters x 8 budgets x 5 folds x 5 corpora) | 1354 / 1800 | 446 | NOT identifiable: no fold column (see below) |
| **total** | **4287 / 5850 (73%)** | **1563** | |

Missing jobs are scattered (the batch was shuffled), not whole corpora or folds.
Source unchanged since the run started (no src/tests .py newer than 2026-11-07 19:07), so resumed jobs use the same code.

## Rate and time
The stopped run completed ~4287 jobs in ~52 h (~82 jobs/h on 51 workers). 1563 missing jobs => roughly 19 h on the
same settings, if the mix of job sizes is similar (unverified: blocklist/budget jobs are large and may leave a long tail).

## Steps
1. Add `--resume` to src/paper.py, untimed batch only (not end-to-end):
   - `Runner(truncate=False)` (already supported: runner.py `truncate=False` keeps existing rows);
   - wrap `runner.submit` so a job whose key (table above) is already in its `output_csv` is skipped;
   - bypass `refuse_existing_results` for this mode; keep `--results`, `--pin`, `--workers`.
   - Generators still compute ground truth per fold up front (a few minutes), then only missing jobs run.
2. Unit tests (tests/): key extraction per evaluation, skipping, append-not-truncate, key rounding of memory_fraction
   and encoding labels (`FP32`, `auto`, ...).
3. Dry run first: `--resume` with a "list only" switch that prints jobs-to-run per evaluation. It must reproduce the
   counts above (8 / 5 / 150 / 954 / equal-memory per decision below). Do not start the real run if it does not.
4. Back up before running: copy the results dir and log.txt to a new dated folder (nothing is deleted; log.txt is
   reset by any new paper.py process).
5. Run (same settings as the original, inside nohup/tmux, myvenv, repo root):
   `myvenv/bin/python src/paper.py --resume --only blocklist,budget,construction,quantization,equal-memory --pin
   --workers 51 --no-build --results results-epyc-concurrent-20261107 > resume-run.out 2>&1 &`
6. Verify: re-run resume_gaps.py (expect 0 missing, no duplicate keys, no FNR>0 failures in the log), then redraw
   figures from the combined CSVs. Do not run `python src/paper.py` without `--resume` or `--overwrite-results`
   on this folder: it would truncate these CSVs.

## Decision needed: equal-memory (no fold column)
Rows carry Config / Memory Budget / Threshold but not the fold, so which folds are missing in a cell is unknowable.
Cells (filter x budget) with 5 rows are complete: 88 of 360. The other 272 have 1-4 rows.
- A. Rerun every incomplete cell for all 5 folds into a NEW csv (equal_memory_resume/<corpus>/...), add a `fold` column
  to the new rows, and merge: complete old cells + rerun cells (old partial rows of rerun cells are left in place but
  excluded by the merge). ~1360 jobs instead of 446 (+~914, roughly +11 h). Cleanest, no deletion.
- B. Accept ragged folds (3-5 per cell): 0 extra jobs. The missing jobs are random, so no systematic bias, but the
  pooled lines use unequal n per point; the paper would need to say so.
- C. Like A but add the `fold` column first so any further interruption is resumable per job.
Recommendation: A with the `fold` column (it is C), unless the ~11 h is not available before the deadline, then B.
The 1117 other missing jobs (blocklist, budget, construction, quantization) are unaffected by this choice.

## Risks / unknowns
- Timings in construction/equal-memory rows (method_s, construct_s, latency) are from the loaded 51-worker batch, as
  before; resumed rows have the same condition, so they stay comparable to the old rows.
- Ordering is not preserved; consumers must not rely on row order (CLAUDE.md already says so).
- Estimates assume the remaining job mix matches the finished one.
