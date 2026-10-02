# Task: improve Ball-Cover throughput further

You are working in the SemanticAMQ repo (read `CLAUDE.md` first; it is
authoritative). Work on a new branch off `speedup`, commit each
self-contained improvement separately, and report the results at the end.
Do not push.

## Background: what has already been done

A mock review of the paper (weakness W1) attacked the filter's query cost.
Branch `speedup` (commits `fa22ad2`, `e0f562f`, `9cc7243`) already did the
following in `src/filters/ball_cover/ball_cover_filter.py`:

- **Bulk decoding.** `_stored_codes()` gives a zero-copy view of the leaf
  index's codes. `_decode()` / `_decode_rotated()` decode them in one pass,
  which replaced faiss `reconstruct_batch` (one rotation per id; 850 ms for
  16K TQ4 centers).
- **`_scan()`** replaced `search(Q, k=num_spheres)` for `query()` and
  `surface_distance()`. It works like this:
  - decode codes in blocks of 4096 in the rotated basis, after rotating the
    queries once;
  - one float32 GEMM per (query chunk, block);
  - test `d² <= (r+τ+ε)²`;
  - re-measure pairs within the norm-scaled float32 error bound in float64;
  - stop scanning a query as soon as one ball accepts it.
- **`_range_accepts()`** answers batches of fewer than 4 queries
  (`_SCAN_MIN_BATCH`) with faiss `range_search` at the largest reach, then
  checks each hit against its own ball.
- **RaBitQ** deliberately still uses faiss search
  (`_estimator_surface_distance`). Keep it that way.

Measured on one pinned P-core, all five blocklist corpora (80K indexed
vectors, 20K queries, τ=0.1), with decisions identical to the old code. The
paper's configuration is minimax_sphere + TQ4. Throughput in q/s:

| k, encoding | single | batch 32 | whole batch |
|---|---|---|---|
| 16384 TQ4 | 188 | 1,514 | 4,208 |
| 16384 FP32 | 386 | 2,142 | 4,128 |
| 16384 PQ8@1 (8.5K balls) | 894 | 2,761 | 8,665 |
| 2048 TQ4 | 1,433 (**5–12% slower than old**) | 9,006 | 24,633 |

Other measurements:

- Streaming insert (cap_merge, k=8192) improved only 1.0–1.3×: 56–460
  vectors/s.
- Construction is unchanged at about 400 s for k=16K. minimax_sphere
  dominates it.
- Raw data is in `suggestions/speedup_eval/` (`speedup_eval.csv`,
  `clean_qps.csv`, plus the scripts that produced them).

What is known about the remaining cost:

- **Single and small-batch queries are decode-bound.** Scoring one query
  against 16K TQ4 codes costs about 5 ms, whether through faiss's scalar-
  quantizer scan or through `sa_decode`. That is roughly 0.8 ns per code
  element. The local faiss wheel runs at **SIMD level NONE**
  (`faiss.SIMDConfig` level 0, `get_compile_options()` empty), so faiss's
  own scans are not vectorized here.
- **Large batches** are bounded by the GEMM plus the per-call decode
  (about 6 ms per 16K TQ4 codes).
- **A two-level ball hierarchy does not prune at d=384.** k-means parents
  whose balls enclose their children still scan 86% / 77% / 52% of balls for
  G = 64 / 256 / 1024, because parent radii are about 1.0. Do not re-try this
  unless you have a genuinely different pruning bound.

## Goal

Raise throughput where it is still weak, in this order:

1. Single-query and small-batch (1–32) query latency for TQ4 at k=2K–16K.
   This is the reviewer's "5.6 ms per query" point.
2. Streaming insert throughput: profile cap_merge first. Earlier work showed
   decoding is not its bottleneck.
3. Large-batch throughput, and construction time, only where the profile
   shows clear headroom.

Ideas worth testing (measure before committing to any of them):

- **A fused lookup-table scan over the 4-bit TQ codes.** Use numba, which is
  available if installed; check with `pip show numba`. The goal is exact
  `||q - decode(c)||²` with no decoded intermediate. Per query, build a
  per-dimension 16-entry table of `(q_j - v)²` in the rotated basis, then sum
  lookups over the codes. Read faiss's `QT_4bit_tqmse` decode to get the
  exact reconstruction values, and verify against `sa_decode`.
- **faiss fast-scan** (`IndexPQFastScan` or similar) with its lookup-table
  quantization error bounded deterministically. Subtract the bound before the
  accept test, then re-check the survivors exactly. Only acceptable with a
  proven, not empirical, error bound.
- **Ordering balls for early exit:** for example, by member count, or so that
  the balls most likely to accept come first. This only helps positives;
  measure on the real query mix.
- **A cheap safe pre-reject** using a contractive projection:
  `||P(q - c)|| <= ||q - c||` for orthogonal P. Any persistent extra per-ball
  state must be charged in `get_bytesize()` and reduce the ball budget. If it
  is not worth its bytes, report that rather than hiding it.
- **Multi-threaded batches.** The paper says throughput scales with cores.
  Measure 1/2/4/8 threads (BLAS threads, or chunking queries across threads)
  and report it separately from the per-core figures.
- **Whether a SIMD faiss build changes the small-batch picture.** For example,
  a conda faiss-cpu with AVX2. Report the finding; do not change the
  project's environment.

## Hard constraints (non-negotiable)

- **Zero false negatives.** Every accept test must be exact distance to the
  decoded center, or a provably conservative bound followed by an exact check.
  An empirical "it never underestimated" is not enough. FNR > 0 anywhere is
  a bug.
- **Honest memory.** Only the codes and fp16 radii (plus charged
  bookkeeping) persist. Transient scratch is fine; a persistent decoded cache
  or any extra per-ball array must be charged in `get_bytesize()` /
  `bytes_per_ball()`.
- **Identical semantics.** `query()` must agree with a float64 exact
  reference except for queries within about 1e-5 of the boundary.
  `surface_distance()` must stay within 1e-5 of the exact minimum. Keep the
  existing tests in `tests/test_ball_cover_filter.py` (`TestBallCoverScan`)
  passing, and add equivalent tests for every new path, including τ=0 exact
  duplicates and un-normalized vectors.
- **Leave baselines alone.** Do not touch competitor filters, and do not
  make them stronger or weaker.
- **Never call `main.generate_*`** from the repo root. It truncates the CSVs
  under `results/`, which has no backup. Do not write into `results/` at all:
  loading the blocklist workloads can create cache files in
  `results/blocklist_clusters/`. Export the workloads once to your scratchpad
  instead. The previous run left three such files; leave them.
- **Thread guard:** set `OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=1` before
  importing numpy or faiss in any new entry point.
- **Use the global Python 3.12** (faiss 1.15.1), not `.venv`.

## How to measure

- Pin timing runs to one P-core with `psutil.Process().cpu_affinity([0])`,
  run sequentially, and use real covers, not random data. Parallel runs read
  3–14× slow.
- Compare against `speedup` HEAD (`9cc7243`) using a throwaway
  `git worktree` for the old code. Reuse `suggestions/speedup_eval/*.py`,
  adjusting their scratchpad paths. `eval_job.py` builds a cover per
  (corpus, k, encoding); `clean_qps.py` re-times queries on saved covers.
- Report single, batch 32 and whole-batch q/s, plus insert vectors/s, for
  TQ4 and FP32 at k=2048 and 16384 on all five corpora.
- Report FPR and FNR per cell, and the number of flipped decisions on the
  *same* cover (this must be 0 except for boundary pairs).
- Run the full suite (`python -m pytest tests`) at every commit in a clean
  worktree. Four `test_blocklist_ground_truth` tests already fail on
  `speedup` with `InsufficientMemoryError`; any other failure is yours.

## Deliverables

1. Commits on your branch, each with a message stating the measured effect
   and why the change preserves zero-FN.
2. A short report:
   - a before/after table;
   - which ideas you tried and dropped, with the numbers that made you drop
     them;
   - any remaining bottleneck, with profile evidence;
   - where the result files live (copy them under `suggestions/`, because
     the scratchpad is temporary).
3. A list of paper sentences or figures your changes make stale.
