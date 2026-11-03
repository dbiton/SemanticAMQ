# Construction speed-up: result report

Branch `improve-construction-claude` (base: `master` 47e3bd4). Goal: ~10x lower construction runtime for Gonzalez,
minimax and soft anneal at similar quality. **Not met.** Measured speedups are 2.7x-5.6x on the controlled 8k run.

## Controlled run (old code 47e3bd4 vs this branch)
- Data: first 8,000 ELI5 vectors construct / last 2,000 queries (d=384), tau = 10^-0.5. **Not** the 80k workload: no 80k
  corpus was available here, so nothing below is an 80k measurement.
- Machine: 4-vCPU Intel Xeon @ 2.1 GHz VM (AVX-512 VNNI), idle during the run; single pinned core (1), BLAS/OMP/faiss
  threads = 1, old and new run back to back, same settings. Gonzalez/minimax: 3 reps (cell medians); soft anneal: 1 rep.
- Commands: `bench_construct/run_final.sh /path/to/worktree-of-47e3bd4 python`, then `bench_construct/summarize_final.py`.
  Raw rows: `bench_construct/out/final_*.csv`.

| method | cells | median paired speedup (total `construct()`) | ratio of aggregate runtimes | median speedup (method only) | range (total) |
|---|---|---|---|---|---|
| Gonzalez | 9 (FP32/TQ2/TQ4 x 1/3/10%) | 4.93x | 4.81x | 4.63x | 3.76x - 6.72x |
| Minimax | 9 | 5.61x | 6.01x | 5.57x | 4.88x - 6.81x |
| Soft anneal | 4 (FP32, TQ4 x 1/3%) | 2.72x | 2.74x | 2.70x | 2.66x - 2.90x |

Slowest-to-improve cells: soft anneal TQ4 3% (232 s -> 87 s, 2.66x), soft anneal FP32 1% (59 s -> 22 s, 2.66x),
Gonzalez FP32 3% (0.78 s -> 0.21 s, 3.76x).

## Quality gates (same run)
- FNR = 0 in every cell; actual bytes <= max_memory in every cell; cap counts identical to the old code in every cell.
- FPR vs old code (paired): Gonzalez and minimax +0.000 pp (worst 0.00); soft anneal mean -0.063 pp, worst |0.15| pp.
  Gate: mean regression <= 0.5 pp, no cell > 2 pp: pass. FPR at FP32 and low budgets is ~1.0 and carries no signal;
  TQ2/TQ4 at 3-10% do.
- Independent red team (`tests/test_adversarial_construct.py`, `bench_construct/redteam/`): no confirmed bugs on duplicates,
  antipodal/near-coincident/denormal/non-unit inputs, budget edges, kernel-divergence fuzzing (320 cases/property);
  18-cell ELI5 matrix FPR delta 0.00 pp. A NaN-row panic introduced by the merged kNN kernel was found while integrating and fixed
  (regression test in `tests/test_construct_tail.py`).
- Not covered: soft anneal TQ2 and budget 10% in the controlled run (cost); real 80k data; Zen 2 hardware.

## What was changed
Fused float64 angle kernels (`construct_common.rs`); Gonzalez lazy/CELF kernel with owner labels, plus an int8-screened
AVX-512 variant (opt-in, `CAPCOVER_GONZALEZ_KERNEL`); minimax fused nearest-direction and cluster-local Badoiu-Clarkson kernels
with mover/stable assignment reuse (`CAPCOVER_ASSIGN_REUSE`); the shared tail (kNN, settle loop, TQ code fit) in Rust with
AMX > VNNI > AVX2 > faiss dispatch (`CAPCOVER_KNN`, `CAPCOVER_TAIL=python` restores the old path); soft anneal fused kernels with a
certified cap-search cache. Reference NumPy/faiss paths remain behind `fast=False` / env switches.

## Agent-reported figures NOT re-measured by the controlled run (treat as indicative)
- 80k synthetic (real rows x8 + noise): Gonzalez selection 5-6x; minimax-excluding-init exact path 2.0-2.7x, with reuse up to 3.6x
  at k=8000; construct tail 7-11x at 80k (AVX-512 kNN); soft anneal refine 3.9x (FP32) and 1.6x (TQ2). Single repeats on a shared box.
- Opt-in approximate variants: adaptive minimax ~1.07x (noise), approx assignment slower and slightly worse (`bench_construct/VARIANTS.md`);
  both kept off.

## Unmet and remaining bottlenecks
- 10x not reached for any method. Dominant remaining costs: n x k nearest-cap searches (GEMM-bound; bound-based pruning does not prune
  in d=384 because caps are ~1.2 rad wide), Gonzalez dot-product arithmetic/bandwidth, and soft anneal's per-refresh full searches
  early in the schedule and at every quantized checkpoint.
- The AVX-512/AMX paths do not exist on the Threadripper 3970X (Zen 2); there the AVX2 fallbacks apply (kNN 1.9-2.9x over faiss, lazy Gonzalez).
  Expect smaller gains than reported here; not measured.
