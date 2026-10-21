# CapCoverFilter: experiments and removal decisions

What was tried on `CapCoverFilter` and its construct strategy, what it measured,
and why each variant was kept or removed. The measurements are copied from the
docstrings that held them before the cleanup; the CSVs they cite
(`results/cap_cover/*.csv`) are gitignored and have no backup.

**Recovering removed code.** Everything removed below is in commit `5886375`
(the last commit before the cleanup), e.g.
`git show 5886375:src/filters/cap_cover/cap_cover_filter.py`.

Conventions: FPR deltas are absolute and negative is better, given at
k = 0.5 / 1 / 2 / 5 / 10 / 15 / 20 % of |B| unless stated. "Fold-paired" means
the same corpus, fold and k for both arms. Setup unless stated: blocklist
workload, 5 corpora x 5 folds, |B| = 8000, tau = 0.1, equal bytes
(`src/compare_cap_cover.py`).

## Decision rule

Removed: (a) anything not reachable from the shipped configuration
(`runner._cap_cover` = `CapCoverFilter.auto()` + `sphere_merge`), and (b) the
three shipped refinements below (R1-R3), each worth under 5 % relative FPR and
each adding construction/query machinery. Kept: the coarse-code, fitted-code,
8-bit-angle, volume-selection, per-cap mixing and compiled-scan pieces.

## Removed refinements that were shipped (R1-R3)

### R1. Top band (`mix_finer=1`) + dense 10-bit ladder (`DENSE10_LADDER`)

- **Idea.** At d = 384 the accept region is set by the 10-300 caps within
  ~0.02 rad of the widest. `mix_finer` raises the "coarsen" ceiling by that
  many rungs, so a plan gives a finer code only to a cap that would otherwise
  exceed the widest angle it keeps. `DENSE10_LADDER` (SPQ20/30/40/60/80,
  10-bit codebooks) adds smaller rungs between `AUTO_CANDIDATES`.
- **Measured, with set storage (R2) on, 175 cells, fold-paired against the
  previous defaults:** FPR -0.020 / -0.013 / -0.004 / -0.002 / -0.002 /
  -0.005 / -0.002, log10 volume -0.8 / -1.4 / -3.0 / -4.7 / -11.5 / -9.6 /
  -10.2 (smaller in 18-25 of 25 cells per k). Together with R2, against FPR
  .619 / .414 / .292 / .180 / .122 / .102 / .087, this is a 0.8-4.9 % relative
  FPR reduction.
- **Measured on the default ladder, without set storage:** `mix_finer=1`
  traded -1.1 .. -9.9 orders of volume for **+0.006 .. +0.017 FPR** at k <= 5 %.
- **Cost.** 10-bit codebooks cost `dim * 2^bits` multiply-adds per encoded cap
  and are "slow" levels in `_choose_mixed`; they also pushed construction
  towards the 5-minute job limit at 80K. R1 + R2 together: batches of 32
  answered at 1.15x the previous time, single queries at 1.28x (median;
  0.76-1.36x and 0.86-3.5x), slower where the cover holds more caps.
- **Sub-experiments (no room).** Planning on fitted codes and tight angles:
  the plan's widest angle is within 0.001-0.012 rad of the stored one.
  Bisecting the widest caps paid for by merging narrow ones: the cheapest
  merges of non-band caps land above the band floor in 62-100 % of cases at
  k <= 2 %, and the k grid already undoes the last, widest merges.
- **Decision: removed.** Under 5 % relative FPR, volume gains that mostly do
  not reach FPR, and slower queries.

### R2. Set storage (`set_codes`, Elias-Fano groups)

- **Idea.** A static cover is a set, so storing caps in a fixed order spends
  log2(n!) bits on the order. Each encoding group was sorted by the first
  8/16/24 bits of its codes (`_key_width`) and those prefixes stored as an
  Elias-Fano sequence (`_ef_layout`, `_ef_encode`, `_pack`, `_Group`,
  `scan_kernels.ef_decode`, a `keys`/`w` argument on every scan kernel):
  about log2(n) - 2 bits saved per cap.
- **Measured.** 1.7-4.6 % of the budget at k <= 5 % of |B| = 8000 (0.7-3.5 %
  above). 175 cells, fold-paired against fixed-order groups: FPR -0.016 /
  -0.011 / -0.004 / -0.004 / -0.001 / 0.000 / 0.000, log10 volume +0.2 ..
  -1.0. That is 0-2.7 % relative against the previous defaults' FPR (the
  absolute baseline was not recorded, so this uses v5's).
- **Cost.** The scan decoded the sequence once per call; `_Group` cached the
  decoded keys and angles because a static filter never mutates them (that
  cache removed most of a single query's own time apart from the scan
  kernel). Every kernel carried the prefix-accumulator path.
- **Decision: removed.** Under 3 % relative FPR for a second code path in all
  eight scan kernels and a charged-bytes assertion.
- **Known side effect.** Without `_Group`'s cache, `_scan_level` re-gathers
  each level's codes and angles from the per-id arrays on every `query()`
  call. That per-call cost was not re-measured against the old cache; if
  single-query latency matters, cache `(codes, theta)` per level after
  `construct()`.

### R3. Merge by accept volume without member counts (`cost="volume"`, `mass_weight=0`)

- **Idea.** Cost a merge by the increase of `sum_j sigma_p(theta_j + t)`, the
  accept region's measure on the sphere, p = dim - 1, with no member-count
  weights, so the cover depends on B and the caps only, not on where the
  members are dense. Commit `89ba229`.
- **Measured against the earlier weighted power cost
  `sum_i m_i (theta_i + 0.1)^8`:** dropping the weights alone moved FPR
  -0.009 .. +0.000 at every k; the volume cost on top -0.010 .. +0.001, with
  log10 accept volume 0.1-5.4 lower. Streaming (kf = 0.1, 25 cells): -0.012 FPR
  at TQ4 but **+0.016 at TQ2**, all of it ChatGPT-paraphrases (+0.10), which
  loses the member-count weights. Volume on S^8 (`exponent=8`) with
  least-volume selection: FPR -0.035 / -0.020 / -0.010 / -0.032 against
  p = 383 at k = 0.5 / 1 / 2 / 5 %, at log10 volume 2.5 / 2.4 / 1.4 / 0.9
  higher; p = 32 and 128 merged exactly as 383 at |B| = 8000.
- **Runtime.** No timing of the cost itself was recorded; both costs rank the
  same candidate edges. The reason to revert is the <1 % FPR effect and that
  the unweighted cost hurts streaming on duplicate-heavy corpora.
- **Decision: reverted.** `SphereMergeConstruct` defaults back to
  `exponent=8.0, mass_weight=1.0` with the power cost; `cost="volume"`,
  `cost_space` and the S^p measure path are removed from it.
  `CapMergeInsert` (streaming) keeps its own cost options and was not changed.

## Removed because unused

### Trained-basis encodings: `PSPQ*`, `DPQ*`, `KSQ*`, `BPQ*`

(`_BasisPQ`, `_DualPQ`, `_KLTSQ`, `_BootPQ`, `basis_fit`, `boot_anchors`,
`boot_equal`, `_plan_boot`, `_train_boot`, `BOOT_CANDIDATES`.)

- **Idea.** Embedding directions are far from isotropic, so a basis fitted to
  the caps codes most of them more finely at equal bytes: seeded PQ in a
  rank-r principal basis (PSPQ), the better of that and whole-space PQ per
  cap with the choice in the code's top bit (DPQ), VA+'s KLT code with
  reverse water-filling and Lloyd-Max levels (KSQ), and a basis the decoder
  rebuilds from the cover's own whole-space caps so it costs no bytes (BPQ).
  Each charged its basis (r TQ4 rows, 9-44 KB at d = 384).
- **Measured.** Median accept angles dropped 0.1-0.3 rad, but (a) at
  |B| = 8000 the basis takes most of a budget at k <= 2 %; (b) a few
  directions with energy outside the basis code worse, so the widest caps,
  which set the volume at d = 384, often widen; (c) the coding error now lies
  inside the data's subspace, where queries are, so a cap reaches further
  along the data than its angle suggests. Fold 0, Quora, |B| = 80K, against
  whole-space seeded PQ at the same code bytes: k = 5 %: KSQ48 FPR .084 vs
  .096 (log10 volume -43 vs -55), DPQ48 .098 (-59), KSQ24 .143 vs .167
  (-20 vs -28), DPQ24 .152 (-30); k = 1 %: KSQ24 .417 vs .378, DPQ12 .514 vs
  .414 (-9.7 vs -8.0). Mixing "never picked them at k <= 1 %" at |B| = 8000,
  and at 80K they doubled construction time.
- **Decision: removed.** Never in `AUTO_CANDIDATES`; better FPR only at
  k = 5 % on one corpus, worse at small k, 2x build time.

### `mixed="free"`

- Lets the widest caps also take finer codes than the least-volume single
  encoding's. Against that encoding: FPR -0.017 / -0.007 / -0.002 / +0.004 /
  +0.020 / +0.001 / +0.001 for 1.5 / 2.6 / 4.3 / 9.0 / 12.0 / 10.0 / 10.7
  orders less volume (CC-News at 10 %: +0.10, its widest caps took TQ4/TQ3
  and the cover kept 2.3K caps instead of 4.2K).
- **Decision: removed.** Volume wins that do not reach FPR; `mixed` is now a
  bool (`True` = the former `"coarsen"`).

### Exact minimum enclosing caps (`exact_caps`, `cap_geometry.min_enclosing_cap`)

- Re-centred the widest share of multi-member caps on their exact smallest
  enclosing cap (NNLS on a core set). The Badoiu-Clarkson caps are 0.004-0.018
  rad looser on average on the 40 widest (up to 0.14 on CC-News at k = 5 %),
  but on CC-News fold 0 at k = 1 / 5 % FPR moved -0.003 .. -0.007 and log10
  volume +-0.08: the stored angle around the decoded direction is set by the
  encoding error. **Decision: removed** (flat).

### Ball-Cover units in the merge cost (`cost_space="sine"`)

- Cost in the caps' Euclidean MEB radii sin(theta) instead of angles:
  +0.018 FPR at k = 0.5 % in `CapCoverFilter.auto()` (24 of 25 cells worse),
  flat above. **Decision: removed.**

### Second code-fitting round (`fit_rounds`)

- Re-fit the codes to the members the first round's caps actually hold.
  Within 0.002 FPR of one round. **Decision: removed**; one round is the
  behaviour.

### Unused seeded-PQ rungs

- `SPQ6`, `SPQ9`, `SPQ18`, `SPQ72`, `SPQ96`, `SPQ24B8`, `SPQ32B8`, `SPQ48B8`
  (and the `DENSE10_LADDER` rungs above) were never in a candidate set. The
  8-bit rungs "score by table but blur ~0.03 rad more at equal bytes". No FPR
  measurement was recorded. **Decision: removed**; the `Quantize` members that
  remain are the Ball-Cover ladder plus `SPQ12/16/24/36/48/64`.

### Costly-codebook mixing (`_CHEAP_CODEBOOK`)

- Encodings whose codebook exceeded 1 KB joined the mix one at a time; only
  the trained bases did. With them gone every candidate is cheap and the mix
  is a single plan over all levels.

## Merge-method experiments that stay documented in the code

These concern `SphereMergeConstruct` internals that are still in use, and are
described in its docstring: neighbour-list searches (+0.0000 / -0.0004 FPR at
k = 0.5 / 1 %; cell-probed lists above 20 % of |B| +0.0020 / -0.0005 at 12
probes, +0.0006 / -0.0003 at 24), `min_batch` 0.01 (<= 0.0001), merging with
a cost that sees encoded angles (within +-0.008, so the unencoded angles
are used), `refresh = 1` (no better).

## Kept, for the record

| Piece | Measured benefit | Why kept |
|---|---|---|
| Coarse directions (SPQ12-64, TQ2-4) | not isolated; the premise of the filter | core design |
| Fitted codes | -0.02 .. -0.08 FPR at k <= 5 % | free in bytes, large |
| 8-bit angles on [min, max] | -0.005 .. -0.011 FPR at k <= 5 %; +4-8 % caps | trivial cost |
| Selection by accept volume | not isolated | needs no queries |
| Per-cap mixing ("coarsen") | -0.058 / -0.056 / -0.035 / -0.026 / -0.002 / +0.001 / -0.001 vs best single encoding; 1.6-2.7x build | largest single refinement |
| Compiled scan | 2.5x at 80K, 1.9x at 8K, same decisions | pure speed |
| Table / LUT scans | 3.2x on SPQ16 batches; 1.6-1.75x single queries | pure speed |

## Removal check (spot check, not a result)

Run once after the cleanup: `runner._cap_cover(384, k)` before (`5886375`) vs
after, 8000 construct vectors / 2000 queries from the first 10 000 vectors of
each corpus (one shuffle, seed 0), tau = 0.1, FPR over queries with true
nearest distance > tau, batches of 32, one process with BLAS threads = 1 and
**no core pinning** (macOS), so timings are indicative only. Zero false
negatives in all runs; bytes within 0.1 % of each other.

| corpus | k | FPR before | FPR after | build s before -> after | q/s before -> after |
|---|---|---|---|---|---|
| wildchat | 1 % | .802 | .841 | 4.2 -> 3.2 | 1809 -> 1893 |
| wildchat | 5 % | .292 | .335 | 13.6 -> 9.3 | 47442 -> 56829 |
| wildchat | 20 % | .125 | .128 | 14.1 -> 11.2 | 20601 -> 19836 |
| stackoverflow | 1 % | .887 | .941 | 4.0 -> 2.9 | 126890 -> 124380 |
| stackoverflow | 5 % | .226 | .172 | 13.9 -> 9.6 | 59079 -> 44390 |
| stackoverflow | 20 % | .004 | .004 | 12.4 -> 8.4 | 19878 -> 18925 |

Construction is 20-35 % faster. FPR is worse in 4 of 6 cells (+0.003 to
+0.054), better in 1 (stackoverflow 5 %: -0.054), tied in 1; with one fold
and ~1000-1900 negatives per cell, differences of a few points are within
noise (the recorded fold-paired effect of R1-R3 combined is 1-5 % relative).
Query throughput moves both ways by up to ~25 %. Nothing here replaces the
fold-paired runs on the 700 GB machine, which are needed before the paper's
numbers are reused.

## Ablation of the remaining pieces (current code)

Setup: `CapCoverFilter.auto()` at `runner._cover_max_memory(384, k)`, |B| = 5000
construct vectors and 5000 queries per corpus (one shuffle, seed 0), tau = 0.1,
wildchat / stackoverflow / triviaqa, k = 1 / 5 / 15 % of |B|, one process per
arm with BLAS threads = 1, every arm under 55 s wall. Zero false negatives
throughout. Single fold, so cell-level differences below ~0.02 are noise.
dFPR = arm - current (positive = the removal hurts).

| Arm | dFPR per cell (1 / 5 / 15 %; wild, stack, trivia) | Verdict |
|---|---|---|
| `fit_codes=False` | +.097 +.036 +.002 / +.052 +.077 +.003 / +.042 +.076 +.007 (worse 9/9, mean +.043 = 19 % rel; build time unchanged) | keep |
| `angle_bits=16` (no narrow 8-bit grid) | +.033 -.022 +.001 / +.015 +.005 +.001 / +.002 +.001 +.001 (worse 8/9, mean +.004 = 1.3 % rel; same bytes) | keep (about ten lines) |
| `mixed=False` (volume-selected single encoding, `_choose`) | -.023 +.034 +.006 / -.027 +.079 +.001 / -.009 +.056 .000 | mixing helps at 5 % (12 / 49 / 24 % rel) but **hurts at 1 %** (all 3 corpora) and is neutral at 15 %; build 2-3x slower with it (2.2-7.9 s vs 0.95-5.5 s) |
| `single_tables=False`, sequential, best of 3, wildchat | single queries 19 616 vs 19 522 q/s (1 %), 6 467 vs 6 499 (5 %), 3 337 vs 3 577 (15 %) | tables buy 0-7 %, far below the 1.6x once recorded; candidate for removal |

Fixed encodings against the current `auto()` (`CapCoverFilter(quantize=q,
max_memory=...)`, same cells):

| corpus | k | auto() | SPQ12 | SPQ24 | SPQ36 |
|---|---|---|---|---|---|
| wildchat | 1 % | .867 | .879 | **.846** | .861 |
| wildchat | 5 % | .292 | .251 | **.249** | .302 |
| wildchat | 15 % | **.128** | .215 | .148 | .134 |
| stackoverflow | 1 % | .934 | .945 | **.907** | .910 |
| stackoverflow | 5 % | .162 | **.119** | .142 | .186 |
| stackoverflow | 15 % | .008 | .023 | **.004** | .009 |
| triviaqa | 1 % | .939 | .978 | **.930** | .932 |
| triviaqa | 5 % | .233 | **.189** | .207 | .245 |
| triviaqa | 15 % | .045 | .085 | **.036** | .047 |

A fixed SPQ24 beats `auto()` in 8 of 9 cells (loses only wildchat 15 %), and
the volume criterion picked SPQ48 on wildchat at 5 % where SPQ24 / SPQ12 are
.08 better. Limits: |B| = 5000 rather than the 8000 / 80K the original
numbers use, one fold, no k below 1 % or above 15 %, and the best fixed
encoding depends on k, so a fixed choice is not yet a replacement.

Build-time profile (wildchat, `cProfile`): `_plan` is 0.26 s of 2.64 s (1 %) and
0.24 s of 6.83 s (5 %); `_level_psi` is 0.99 s and 4.82 s (70 % at 5 %). The
per-ceiling plans in `_choose_mixed` are not worth trimming.

Static checks: no shipped evaluation passes `mixed=` or reaches `_choose`
(streaming uses the single-level FP32 / TQ4 filter; the static covers use
`auto()` which mixes). `FP16`, `SQ8`, `SQ6`, `SQ4`, `TQ8` and `TQ1` are
referenced only by tests and `_is_turboquant`.
