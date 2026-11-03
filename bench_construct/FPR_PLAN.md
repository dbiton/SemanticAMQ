# FPR quality (and efficiency) improvement plan

## Objective

Lower the false-positive rate of Cap-Cover at a fixed byte budget, with zero false negatives, and (secondary) keep
or reduce construction/query cost. FPR is driven by the volume the stored caps accept at tau: per-cap angle theta,
the accept rule's widening (beta, tau), and how many caps the budget buys at the chosen encoding. Target: a
**paired mean FPR reduction of at least 1 percentage point on the signal cells** (TQ2/TQ4, budgets 3-10%) with no cell
worse than +0.5 pp, or a clearly documented negative result. Do not tune on held-out query FPR.

## Data and provisional baselines

- Data: `/tmp/claude-0/-home-user/faeeddb3-925b-5717-b1cb-caa0100e8a36/scratchpad/eli5/eli5_10k.pkl` (float32 (10000,384)).
  Fold 0: first 8000 construct / last 2000 queries; folds 1-3: seeded permutations (see `bench_construct/variants.py`).
  tau = 10**-0.5 as L2 distance; ground truth by exact nearest-construct distance (`bench_construct/bench.py`).
- Reference numbers: `bench_construct/out/final_gm_new.csv`, `final_sa_new.csv` (current branch) and
  `bench_construct/VARIANTS.md`. FP32 FPR is ~1.0 at low budgets and carries no signal; the signal cells are
  TQ2 3%/10% and TQ4 10% (FPR roughly 0.04-0.8). The sweep is small and single-corpus, so say so in any claim.
- Construction-set criteria to tune on (never query FPR): the encoded cover volume `CapCoverFilter._encoded_volume`,
  sum theta^8, psi_mean. Query FPR on folds is validation only.

## Repository entry points

- `src/filters/cap_cover/cap_cover_filter.py`: `_choose` (encoding by volume), `_encode_cover`, `_tight_angles`,
  angle code/rounding (`_angle_code`, `_angle_value`), `_encoded_volume`, `_beta`, auto() budget accounting.
- `src/filters/cap_cover/soft_anneal_construct.py`: annealed objective (volume-only, tau-free) and `refine_encoded`.
- `src/filters/cap_cover/minimax_construct.py`, `cap_geometry.py`, `cap_quantizers.py`.
- **Read `CAP_COVER_EXPERIMENTS.md` first**: it records what was already tried and why variants were removed. Do not
  re-run a removed idea unless you can say what is different.
- Existing tests: `tests/test_cap_cover_filter.py`, `tests/test_quantization_soundness.py`, `tests/test_soft_anneal_*.py`.

## Agent organization (budget-limited: ONE implementation agent, two subjects, no sub-spawning)

1. **Profile before building** (cheap): decompose the FPR of the signal cells. For the stored cover, how much accepted
   volume comes from (a) cap angles that are loose vs the members they must hold, (b) angle code rounding and the
   conservative epsilon, (c) the beta/tau widening, (d) too few caps for the budget (angle-code bytes, per-cap overhead)?
   Report numbers; choose the subject(s) with the largest measured slack.
2. **Subject A, tighter certified angles**: shrink slack that is not needed for coverage: finer or non-uniform angle
   codes inside the same byte budget, per-cap rounding that stays conservative, a better member-to-cap assignment (a
   member joins the cap where it widens the volume least, not the nearest direction), and exact float64 re-certification.
3. **Subject B, objective alignment**: make the choice of construction/encoding follow the volume the query accepts
   (tau-aware widening of the objective, or ranking candidates by the stored-angle volume at selection_tau rather than a
   proxy) without using query vectors. The soft-anneal module documents a deliberate tau-free design: justify any change
   against `CAP_COVER_EXPERIMENTS.md`.
4. Efficiency is a tie-breaker: any change must not raise construction time more than 1.5x at equal quality unless it
   buys at least 2 pp FPR; query throughput must not regress (check `CapCoverRustFilter/bench`).

## Execution phases (keep each phase short; stop early if a phase shows no slack)

1. Setup and baseline: branch `fpr-opt` from `improve-construction-claude`; copy the venv; build; reproduce the baseline
   FPR on fold 0 for gonzalez/minimax x TQ2/TQ4 x budgets {3,10}% (about 8 cells). 1 rep for FPR (deterministic).
2. Profile the slack (point 1 above), pick the subject(s).
3. Implement behind a flag (default off until gated), with the old path kept as the reference.
4. Develop on fold 0 using construction-set criteria; freeze choices; validate FPR on folds 1-3 only once.
5. Gate and report.

## Acceptance gates

- Zero false negatives on every evaluation workload; every construct vector inside its cap in float64 with conservative
  rounding intact; `get_bytesize() <= max_memory`; no hidden budget change (no fewer bytes charged, no more caps than the
  budget buys).
- Report paired FPR deltas per encoding and budget, plus mean and worst cell, on folds 1-3 (validation) and fold 0.
- Adversarial cases (duplicates, ties, empty/singleton caps, near-coincident vectors, tiny inputs, k near n) pass for any
  changed kernel; run `tests/test_cap_cover_filter.py`, `tests/test_quantization_soundness.py`,
  `tests/test_soft_anneal_construct.py`, `tests/test_adversarial_construct.py`.
- Construction time reported (paired, pinned core) so quality gains are not bought with silent slowdowns.

## Deliverables

1. Committed implementation on branch `fpr-opt` (flagged, old path kept), no push.
2. A script and raw paired results under `bench_construct/out/` (CSV is gitignored; add with `git add -f` only the summary
   CSV), and a short `bench_construct/FPR_RESULTS.md`: slack decomposition, what was tried, per-cell deltas, gate pass/fail,
   limitations, and a recommendation (enable / keep opt-in / reject).
3. If no change clears the gates, say so plainly with the measured slack; a documented negative result is acceptable.

## Run economy (hard limits; total budget is small)

At most ~60 tool calls. No single run longer than ~10 minutes. Use ELI5 8k only, TQ2/TQ4 signal cells, gonzalez and
minimax (skip soft anneal unless the subject needs it, then only 1-2 cells with reduced iters). Pin to core 1,
OMP/OPENBLAS/MKL threads = 1 before importing numpy. Report only numbers you measured.
