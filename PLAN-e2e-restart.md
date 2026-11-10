# Plan: stop the epyc batch, run end-to-end now (2026-11-09)

Run being stopped: `myvenv/bin/python src/paper.py --pin --workers 51 --timing-mode concurrent --no-build
--results results-epyc-concurrent-20261107` (main PID 3910911, 51 worker children), started 2026-11-07 19:07 UTC.
Results dir: `results-epyc-concurrent-20261107/` (blocklist, budget, construct_methods, cap_quantization, equal_memory).

0. Read-only checks: PIDs; end-to-end concurrency = min(MAX_WORKERS, 5 corpora x 5 folds) = 25 timed jobs
   (no --timing-workers, same as the original command); end_to_end.csv absent; 31 GB disk free.
1. Snapshot (nothing deleted): copy the results dir, log.txt and run_manifest.json to
   `results-epyc-partial-20261109/`.
2. Stop: SIGTERM main PID, then any surviving workers. Completed jobs stay in the CSVs (rows are written as
   jobs finish); in-flight jobs (<= 51) are lost. Verify no paper.py process remains and CPUs are idle before step 3
   (an orphaned worker would contaminate the timings).
3. Start end-to-end only, same settings, same results dir (it only creates end_to_end.csv; other CSVs untouched):
   `nohup myvenv/bin/python src/paper.py --only end-to-end --pin --timing-mode concurrent --no-build
   --results results-epyc-concurrent-20261107 > e2e-run.out 2>&1 &`
   The new process resets log.txt (accepted). A timestamped run_manifest.*.json is added; the first is kept.
4. (Not done by Claude in this session) Monitor e2e-run.out for Tracebacks / FNR > 0; check timing_condition and
   pinned_cpu in end_to_end.csv; draw throughput and quantization-speedup figures.

Filling the missing data from the stopped batch is handled later. Note: `--missing` / `--only` operate per
evaluation and truncate that evaluation's CSVs (refused without --overwrite-results); there is no per-job resume.
The completion log before 23:02 UTC 2026-11-09 was lost (truncated by an ad-hoc script), so the set of unfinished
jobs is not recorded; the snapshot's CSVs show which (config, fold) rows exist.
