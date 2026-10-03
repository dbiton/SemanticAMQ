# Ball-Cover throughput, round 2 (branch `speedup`, 2026-10-02)

Follow-up to `suggestions/throughput_agent_prompt.md`. All timings: one pinned
P-core (i7-14700, faiss 1.15.1 pip wheel at SIMD level NONE), sequential, real
covers (minimax_sphere, 80K indexed vectors, 20K queries, tau=0.1). Other
agents were running pinned jobs on CPU 14 and the E-cores throughout, so old
and new code were timed alternately per cell (best of 2 each) to cancel drift.

## Commits

| commit | change | effect |
|---|---|---|
| `bd51bee` | `scan_kernels.py`: early-abandon numba scan for small TQ4/SQ4/FP32 batches | 16K TQ4 single 14-23x, batch 32 1.7-4.3x |
| `5cfa942` | static covers stored largest ball first | kernel 1.16-1.48x more on TQ4 |
| `550e103` | kernels compiled at import, not on the first timed query | removes a 0.56-1.7 s first-query stall |
| `a28fb4e` | `scaling_query_worker` docstring: tau now affects throughput | comment only |
| `59c717e` | kernel for every batch size on 4-bit covers of >= 4096 balls | whole-batch 1.0-2.9x (0.88x on CC-News) |

Zero-FN argument (scan_kernels.py header): every term is a square of a
difference of float32 values formed in float64, so nothing cancels and a
computed (partial) sum P of the exact D obeys |P - D| <= 2^-23 D + 1e-36. A
ball is rejected only if P > reach^2 (1 + 2^-22) + 1e-36 (implies D > reach^2,
also for partial sums since the remaining terms are >= 0), accepted outright
only if P <= reach^2 (1 - 2^-22) - 1e-36, and anything between is re-measured
in float64. Decisions are those of the float64 distance to the decoded center.

## Before / after

Measurement setup:

- **Old:** `27cab92` (= `9cc7243` for `src/`). **New:** `59c717e`.
- **Covers:** the same saved covers in both trees. The new tree stores each
  cover largest first, as its `construct()` does. A fresh new-code build of
  WildChat 2K TQ4 and FP32 gave bit-identical ids, codes and radii to the
  saved cover after reordering.
- **Query mix:** a fixed random permutation of the 20K queries. The query
  files are stratified (the first ~20% are the near-duplicate stratum), so
  the earlier first-N protocol over-weighted positives. It is reported
  further down for comparability.
- **Rates:** single = 400 queries one at a time; batch 32 = 2,048 queries;
  whole = all 20K in one `query()`. Best of 2 alternated runs per tree.
- **Flips:** old-code vs new-code decisions on the same cover, over all 20K
  queries, both whole-batch and in batches of 32.

| encoding, balls | corpus | single old -> new | batch 32 old -> new | whole old -> new | FPR | FNR | flips |
|---|---|---|---|---|---|---|---|
| TQ4 16384 | WildChat | 198 -> 5,016 (25.3x) | 1,700 -> 10,333 (6.1x) | 3,604 -> 10,993 (3.05x) | 0.1586 | 0 | 0 |
| TQ4 16384 | CC-News | 198 -> 3,552 (18.0x) | 1,887 -> 5,040 (2.7x) | 4,824 -> 5,161 (1.07x) | 0.5485 | 0 | 0 |
| TQ4 16384 | WikiAns. | 197 -> 4,093 (20.7x) | 1,626 -> 6,546 (4.0x) | 3,558 -> 6,811 (1.91x) | 0.1630 | 0 | 0 |
| TQ4 16384 | Quora | 197 -> 5,264 (26.7x) | 1,672 -> 9,705 (5.8x) | 3,507 -> 10,439 (2.98x) | 0.0169 | 0 | 0 |
| TQ4 16384 | ChatGPT-P. | 195 -> 3,305 (17.0x) | 1,637 -> 5,136 (3.1x) | 3,561 -> 5,184 (1.46x) | 0.2096 | 0 | 0 |
| TQ4 2048 | WildChat | 1,515 -> 11,067 (7.3x) | 13,191 -> 22,587 (1.7x) | 27,562 -> 27,313 (0.99x) | 0.5267 | 0 | 0 |
| TQ4 2048 | CC-News | 1,488 -> 14,898 (10.0x) | 13,023 -> 39,640 (3.0x) | 29,439 -> 29,475 (1.00x) | 0.9687 | 0 | 0 |
| TQ4 2048 | WikiAns. | 1,480 -> 11,728 (7.9x) | 13,462 -> 21,735 (1.6x) | 28,084 -> 28,348 (1.01x) | 0.7177 | 0 | 0 |
| TQ4 2048 | Quora | 1,491 -> 10,911 (7.3x) | 13,610 -> 21,344 (1.6x) | 28,062 -> 27,771 (0.99x) | 0.5811 | 0 | 0 |
| TQ4 2048 | ChatGPT-P. | 1,477 -> 10,409 (7.0x) | 13,738 -> 22,479 (1.6x) | 28,854 -> 27,928 (0.97x) | 0.8508 | 0 | 0 |
| FP32 16384 | WildChat | 953 -> 4,224 (4.4x) | 2,594 -> 2,870 (1.1x) | 3,633 -> 3,935 (1.08x) | 0.1489 | 0 | 0 |
| FP32 16384 | CC-News | 921 -> 1,595 (1.7x) | 3,150 -> 3,619 (1.1x) | 4,812 -> 5,804 (1.21x) | 0.5237 | 0 | 0 |
| FP32 16384 | WikiAns. | 1,052 -> 2,386 (2.3x) | 2,550 -> 2,720 (1.1x) | 3,582 -> 3,686 (1.03x) | 0.1531 | 0 | 0 |
| FP32 16384 | Quora | 1,097 -> 3,767 (3.4x) | 2,594 -> 2,685 (1.0x) | 3,507 -> 3,626 (1.03x) | 0.0136 | 0 | 0 |
| FP32 16384 | ChatGPT-P. | 960 -> 1,624 (1.7x) | 2,509 -> 2,758 (1.1x) | 3,610 -> 3,729 (1.03x) | 0.2022 | 0 | 0 |
| FP32 2048 | WildChat | 8,942 -> 8,739 (1.0x) | 23,865 -> 24,027 (1.0x) | 29,311 -> 29,306 (1.00x) | 0.5129 | 0 | 0 |
| FP32 2048 | CC-News | 8,711 -> 8,650 (1.0x) | 24,711 -> 24,039 (1.0x) | 31,606 -> 31,625 (1.00x) | 0.9481 | 0 | 0 |
| FP32 2048 | WikiAns. | 8,880 -> 9,320 (1.0x) | 24,412 -> 23,679 (1.0x) | 30,363 -> 30,126 (0.99x) | 0.6987 | 0 | 0 |
| FP32 2048 | Quora | 8,970 -> 9,034 (1.0x) | 24,043 -> 24,159 (1.0x) | 29,537 -> 29,076 (0.98x) | 0.5592 | 0 | 0 |
| FP32 2048 | ChatGPT-P. | 8,795 -> 9,367 (1.1x) | 23,478 -> 24,196 (1.0x) | 30,772 -> 30,941 (1.01x) | 0.8058 | 0 | 0 |

**FPR, FNR and flips:**

- FPR is identical in both trees in every cell.
- FNR is 0 everywhere.
- 0 decisions flipped in any cell, whole-batch or batch-32.

FP32 at 2K goes through unchanged code (`range_search` / `_scan`), and its
±6% spread is the noise floor under the other agents' load.

**Geometric-mean speedups:**

| encoding, balls | single | batch 32 | whole batch |
|---|---|---|---|
| TQ4 16K | 21.2x | 4.1x | 1.9x |
| TQ4 2K | 7.9x | 1.9x | 1.0x |
| FP32 16K | 2.5x | 1.1x | 1.1x |
| FP32 2K | 1.0x | 1.0x | 1.0x |

On the old first-N protocol (first 300 / first 4,096 queries, the earlier
table's), TQ4 16K goes from 196 to 3,650-6,720 q/s single (18-34x) and from
1,950 to 5,050-15,390 at batch 32 (2.6-7.9x). The positive-heavy head exits
earlier.

**TQ4 vs FP32:**

- **Same ball count** (16K, batch 32): TQ4 now answers 1.4-3.6x faster than
  FP32. Its codes are 8x smaller and abandon just as early.
- **Equal memory** (2^11 FP32 vs 2^14 TQ4 balls, batch 32): TQ4 was
  13-15x slower than FP32 and is now 2.3-4.8x slower.

**Streaming insert** (cap_merge; bootstrap = minimax_sphere on the first 20K
construct vectors at a k-ball budget; then 3,000 inserts in batches of 1,000;
the same bootstrap pickle replayed under both trees). Vectors/s, old -> new:

| k | encoding | WildChat | CC-News | WikiAns. | Quora | ChatGPT-P. |
|---|---|---|---|---|---|---|
| 2048 | TQ4 | 2,684 -> 2,699 | 3,230 -> 3,119 | 2,242 -> 2,247 | 1,896 -> 1,882 | 2,188 -> 2,204 |
| 2048 | FP32 | 2,202 -> 2,207 | 571 -> 580 | 1,167 -> 1,143 | 1,679 -> 1,708 | 786 -> 789 |
| 16384 | TQ4 | 561 -> 551 | 126 -> 129 | 160 -> 162 | 1,022 -> 1,010 | 151 -> 153 |
| 16384 | FP32 | 613 -> 606 | 116 -> 120 | 126 -> 129 | 1,115 -> 1,099 | 127 -> 130 |

The cover after the inserts is bit-identical between trees in all 20 cells
(SHA-1 over ids, codes, radii). FNR is 0 against the inserted vectors, and
every inserted vector is accepted at tau=0. Insert speed is unchanged
(0.97-1.03x): the insert path was not touched, as explained under
"Streaming insert" below.

Construction is unchanged too. Reordering storage adds one argsort and one
copy of the codes, and the minimax_sphere build is bit-identical.

## Ideas tried

Kept:

- **Fused 4-bit LUT scan** (numba). TQ4 decodes every dimension through one
  16-entry table (dim 2b = low nibble of byte b). Per query the kernel builds
  a 192 x 256 float32 table of `(q_2b - t_lo)^2 + (q_2b+1 - t_hi)^2`, reads
  codes 8 bytes at a time with four accumulators, and checks the abandon
  bound once per 8-byte word. The table comes from `sa_decode` itself, and a
  test checks it bit for bit.
- **Early abandon = the contractive-projection pre-reject, for free.** The
  partial sum over the first m rotated coordinates is ||P(q - c)||^2 for a
  coordinate projection P. Behind the rotation energy is spread evenly, and
  real covers abandon a ball after a mean of 15-17 (WildChat, Quora) to 91
  (CC-News) of 384 dims. No stored per-ball state is needed, so no extra
  sketch was worth its bytes.
- **Ball order** (radius descending, static covers). Ball visits as a
  fraction of a full scan, storage -> radius order: 16K 0.61-0.92 ->
  0.46-0.88, 2K 0.14-0.59 -> 0.10-0.46. Member-count order was within 0.03
  of radius order but needs counts at query time; radius is free. Measured
  kernel gain 1.16-1.48x. Sorted order also helps branch prediction:
  ascending radius beat construction order at 16K too.
- **Kernel for whole batches on >= 4K-ball 4-bit covers** (numbers in the
  `59c717e` message and in `_use_kernel`'s comment).

Dropped:

| idea | numbers | why dropped |
|---|---|---|
| nibble table (24 KB, L1-resident, 2 loads/byte) | 16K: 5.7K vs 7.0K q/s WildChat, 1.4K vs 2.4K CC-News | byte table wins once built in-kernel |
| table built in numpy | 280 us/query of table build | moved into the kernel (~12 us) |
| two-phase branch-free first word | 16K: WildChat/Quora +17-19%, CC-News +5%; 2K: -2% | u64 loads gave +28-38% for less code |
| float32 SIMD FP32 kernel (fastmath reassoc) | 3.1K vs 4.5K q/s WildChat 16K; a full 16K x 384 pass is 4.3 ms with or without vectorization | FP32 is memory-bound (25 MB per 16K cover); kept scalar float64 for <8 queries only |
| caching the decode table / ids across calls | would save ~30 us/call | a persistent decoded table and an 8 B/ball id copy would have to be charged |
| faiss fast-scan with a bounded LUT error | not built | it needs its own interleaved code layout (a second copy of the codes, charged) or a different encoding. The exact kernel's remaining cost is per-ball loads and branches, not arithmetic. |
| re-sorting storage after `insert()` | O(k log k) per call | adds ~1.5 ms per arrival to per-insert timings, and changes cap_merge's covers (see `5cfa942`) |
| FP32 kernel on small covers / big batches | 2K single 0.82-1.67x of range_search; 16K batch 8 0.73-2.9x | cutoffs `_FLAT_KERNEL_MAX_BATCH = 8`, `_FLAT_KERNEL_MIN_BALLS = 4096` |

**Streaming insert (cap_merge, k=8192)**, profiled on saved bootstraps. Of
6.3 s for 2,000 FP32 inserts on WildChat:

- `_dist` 2.3 s: 0.73 ms/call, one 8192 x 384 float32 matvec at ~22 GB/s,
  L3-bandwidth bound;
- `_covering_row` 1.3 s: another matvec;
- `_pair_costs` 0.76 s: O(k) log/exp;
- faiss `IndexIDMap2.remove_ids` 0.74 s: compacts the code array and
  rebuilds the reverse map.

TQ4 has the same profile. An uncovered vector costs ~3 matvecs, a covered
one 1. There is no decision-preserving change worth making:

- Batching the coverage screen into one GEMM changes float32 rounding in a
  test that already cancels. An exact duplicate of a zero-radius center can
  screen ~6e-3 away. That changes which balls get created.
- The merge costs are argmin-sensitive to float rounding.

So cap_merge is untouched, and insert throughput is the same code in both
trees (see table).

**Multi-threaded batches** (whole batch of 8192 queries; kernel chunks
queries over Python threads, since it releases the GIL; `_scan` uses BLAS
and faiss OpenMP threads; 7 free physical P-cores + 1 HT sibling for 8):

| cover | path | 1 thread | 2 | 4 | 8 |
|---|---|---|---|---|---|
| WildChat 16K TQ4 | kernel | 10.9K | 22.4K | 45.1K | 67.4K |
| | `_scan` | 3.8K | 5.3K | 6.8K | 6.6K |
| CC-News 16K TQ4 | kernel | 5.3K | 10.6K | 20.9K | 33.7K |
| | `_scan` | 5.8K | 8.0K | 10.1K | 9.4K |
| WildChat 16K FP32 | kernel | 5.2K | 10.1K | 20.7K | 32.9K |
| | `_scan` | 3.9K | 5.8K | 7.9K | 8.3K |
| CC-News 16K FP32 | kernel | 1.7K | 3.2K | 6.5K | 11.8K |
| | `_scan` | 5.6K | 8.9K | 12.9K | 12.4K |
| WildChat 2K TQ4 | kernel | 25.5K | 47.1K | 97.0K | 146.8K |
| | `_scan` | 27.4K | 37.4K | 48.1K | 44.1K |
| CC-News 2K TQ4 | kernel | 50.9K | 95.8K | 171.5K | 271.4K |
| | `_scan` | 29.8K | 39.9K | 60.2K | 48.6K |

The kernel scales 2.0x / 4.0x / 5.3-6.8x. `_scan` scales 1.3-1.6x /
1.7-2.3x / 1.6-2.2x, because only its GEMM is threaded. `query()` itself
stays single-threaded, as every evaluation measures it per core. Chunking a
batch over threads is a ~10-line addition if a deployment wants it.

**SIMD faiss.**

- Here, both the pip wheel and conda-forge's faiss-cpu 1.14.3 for Windows
  (tried in a throwaway micromamba env) are SIMD level NONE. There is no WSL
  distribution.
- The **Linux** pip wheel the run box uses (`faiss_cpu-1.15.1-cp310-abi3-manylinux_2_28`)
  dispatches at runtime: libfaiss.so instantiates its templates at
  SIMDLevel 0/1/2 (NONE/AVX2/AVX512).
- So on the Threadripper, faiss's own scans and `sa_decode` (the old
  paths, `_range_accepts` and `_scan`'s decode) are SIMD-accelerated, and
  the kernel's margin there will be smaller than measured here.
- The numba kernel compiles for the host CPU either way.
- **Re-check the cutoffs on the run box** with `cross2.py` / `cross3.py`
  before quoting run-box throughput.

## Remaining bottlenecks (profile, single TQ4 query, sorted cover)

| cover | total | kernel | of which table build | `sa_decode` table probe | `_stored_codes` | rotation | radii gather |
|---|---|---|---|---|---|---|---|
| WildChat 2K | 91 us | 42 | 12 | 21 | 9.5 | 5.8 | 1.3 |
| CC-News 2K | 67 us | 23 | 12 | 20 | 9.5 | 5.7 | 1.2 |
| WildChat 16K | 181 us | 113 | 12 | 20 | 13 | 5.8 | 8.5 |
| CC-News 16K | 278 us | 203 | 12 | 19 | 13 | 5.8 | 8.5 |

- **At 2K, fixed per-call overhead is roughly half the time.** The decode
  table is probed through faiss's scalar `sa_decode`; it would be faster on
  an AVX2 build. `id_map` is copied out of faiss on every call. Removing
  either means persisting derived state.
- **At 16K, the kernel itself** costs ~7-12 ns per ball. That is one or two
  8-byte words of table lookups into a 192 KB (L2) table plus a branch.
- **CC-News** has the largest radii (mean 0.64 vs 0.25), so its balls
  abandon late (~46 code bytes) and it is the one corpus where `_scan` still
  wins large batches (by 8-12%).
- **FP32** is memory-bound.
- **Inserts** are bandwidth-bound matvecs (above).

## Paper sentences/figures this makes stale (`paper` branch `paper.tex`)

1. Setup (~l.1319-1323): "A query is answered by a flat FAISS scan over the
   stored centers, which FAISS parallelizes across the queries of a batch, so
   query throughput scales with additional cores." Small batches, and every
   batch on a TQ4 cover of 4K+ balls, now go through a single-threaded
   compiled scan. It scales with cores only if the caller chunks queries
   across threads (2.0/4.0/5.3-6.8x at 2/4/8 threads, measured above).
2. `tab:quantization`, last column (median q/s at the smallest budget, from
   `src/plots/table_quantization.py` over `results/quantization`, which
   times one whole-batch `query()`): TQ4 180 and FP32 4,377 are stale. A
   whole batch on 16K TQ4 balls (the smallest budget's TQ4 cover) is now
   5.2-11.0K q/s per core. It was already 3.5-4.8K after the first speedup
   round, so the column predates both rounds.
3. Quantization text (~l.1422-1423): "each TQ4 center is also slower to score
   than an FP32 one". This is now reversed: at 16K balls and batch 32, TQ4
   answers faster than FP32 on every corpus (table above).
4. Comparison (~l.1566-1569) and `fig:equal-memory-throughput`: "its
   throughput falls with the ball count. It is below the VA+-file, and one to
   two orders of magnitude below DSH." The equal-memory worker times batches
   of 32, which are now 1.6-3.0x faster for TQ4 at 2K balls and 2.7-6.1x at
   16K, so whether Ball-Cover is still below the VA+-file needs a re-run.
5. The scaling figures (`scaling_query_worker`, throughput vs k and batch
   size) are stale. Small batches gained the most, so the curve's shape
   changes.
6. Conclusion (~l.1587): "trades query throughput for memory, since a query
   scans every stored center". Still true as the O(kd) worst case, but a
   query now reads 15-90 of each center's 384 coordinates on these covers.
7. Older numbers still in `paper copy.tex` are all stale:
   - "answers 112-415 queries per second on one core";
   - "from 5,500 queries/s at 0.1% of raw to 112 at 4%";
   - "TQ4 answers a median of 177 queries/s against 4,542 for FP32, 26 times
     fewer ... about 3 times slower to score".
8. Unchanged: streaming per-arrival insert costs (insert path untouched),
   construction times, every FPR/FNR (decisions identical), and the O(k_E d)
   query-time bound.

## Files

- `suggestions/throughput_eval/` holds every script used:
  - `bc.py` is the shared loader;
  - `export_workloads.py` writes `eval/*.npz` (run it from a throwaway
    worktree);
  - `final_driver.py` + `final_query_job.py`: query table;
  - `insert_driver.py` + `insert_job.py` (with `boot2.py`): insert table;
  - `final_report.py` merges the results;
  - `cross2.py` / `cross3.py`: crossovers;
  - `threads.py`: thread scaling;
  - `order*.py`: ball order;
  - `prof_insert.py`: cap_merge profile.
  
  Scripts locate data relative to their own folder. The drivers expect an
  old-code worktree at `<folder>/old_tree` and hard-code the repo root.
  Also: `correct.py` checks flips and FPR/FNR on all 20K queries,
  `reorder_eval.py` A/Bs the ball order, `over2.py` gives the per-call
  profile, and `build_cover.py` builds a static cover. The query scripts
  load covers saved as `eval/out/new_static_<corpus>_<k>_<enc>_cover.pkl`,
  which `suggestions/speedup_eval/eval_job.py` or `build_cover.py` write.
- `suggestions/throughput_eval/results/` (local only, not committed):
  - `throughput_eval.csv`: the query table, plus first-N columns and FPR
    in both trees;
  - `insert_eval.csv`: the insert table, plus cover hashes;
  - `final/*.json`: per-run rows;
  - `cross2/3/4.json`: kernel vs old path by batch size;
  - `reorder_eval.json`: largest-first A/B;
  - `threads.json`: thread scaling.
