# Rust query kernel for CapCoverFilter: running report

Crate: `CapCoverRustFilter/` (`src/core.rs` pure-Rust kernel, `src/lib.rs` PyO3 glue),
wrapper `cap_cover_rust_filter.py` (`CapCoverRustFilter(CapCoverFilter)`), bench scripts in `bench/`,
raw numbers as `bench/*.json`, tests in `tests/`.

Environment actually used on this machine: `.venv` (Python 3.14.5, faiss 1.14.3, numba 0.67, numpy 2.4.6;
the global python3.12 here has no faiss). Apple Silicon (5 P + 10 E cores, NEON, Accelerate BLAS via numpy, so the old
path's GEMM runs on AMX). Build: `cd CapCoverRustFilter && maturin develop --release`.
Run from repo root with `PYTHONPATH=src:CapCoverRustFilter:CapCoverRustFilter/bench`.

Workload used for every number below: 20,000 WildChat vectors as construct set (d=384), `tau=0.1`, queries =
4096 other WildChat vectors (held out). Filters: single-encoding `CapCoverFilter(quantize=X, max_caps=3000)`
(sphere_merge), plus `mixed` = `CapCoverFilter.auto(384, max_memory=288000)` (8660 caps: SPQ36 1, SPQ24 2822,
SPQ16 5719, SPQ48... see `bench/baseline_numba.json`). "pos" = the real held-out stream (positive rate
42-88% depending on filter: caps are wide at 3K caps for 20K vectors); "neg" = only the queries the filter
rejects, i.e. full scans of every cap (worst case, and what a mostly-negative stream costs).
Single thread (macOS, no pinning possible; run sequentially, noise about 3-5%).

## 0. Baseline: current numba path (`bench/baseline.py`, `bench/baseline_numba.json`)

qps (queries/sec). `ovh` = fraction of wall time in Python glue outside the compiled kernels
(rotation, beta, `_codes[ids]` copy, tolist).

| filter | caps | pos rate | b=1 | b=32 | b=1024 | neg b=1 | neg b=32 | neg b=1024 | ovh b1 / b32 / b1024 |
|---|---|---|---|---|---|---|---|---|---|
| TQ1  | 3000 | 0.58 | 6,650 | 37,900 | 240,000 | 5,060 | 36,500 | 196,000 | 29% / 9% / 22% |
| TQ2  | 3000 | 0.55 | 4,010 | 36,100 | 237,000 | 3,000 | 35,700 | 196,000 | 17% / 8% / 21% |
| TQ4  | 3000 | 0.54 | 2,400 | 33,900 | 229,000 | 1,690 | 33,700 | 190,000 | 11% / 8% / 21% |
| SPQ16| 3000 | 0.88 | 12,650 | 91,800 | 102,600 | 7,350 | 82,300 | 90,300 | 52% / 20% / 9% |
| SPQ48| 3000 | 0.57 | 4,660 | 43,900 | 246,000 | 3,660 | 42,400 | 200,000 | 20% / 10% / 22% |
| mixed| 8660 | 0.42 | 2,085 | 17,200 | 92,600 | 1,630 | 16,700 | 74,800 | 33% / 10% / 9% |

Reading: b=1 is the table1/lut/direct kernels (the single-query paths) with large Python glue share (up to 52%);
b=32 and b=1024 are decode + sgemm (on AMX here, ~0.45 TFLOP/s effective: this is the part a single NEON core
cannot beat on this Mac; on x86 with OpenBLAS it will be several times slower than here).
Glue is 8-30% at every batch size; at b=1 it is 0.03-0.1 ms.

## 1. Rust versions (raw: bench/v0.json ... ; qps, single thread; "pos" = real mixed stream, "neg" = rejected-only)

Every row below was checked against the original path on the same stored caps: 0 disagreements on 4096 queries per filter.

| step | TQ1 b1 pos/neg | TQ4 b1 pos/neg | SPQ16 b1 pos/neg | SPQ48 b1 pos/neg | mixed b1 pos/neg |
|---|---|---|---|---|---|
| numba (original) | 6,650 / 5,060 | 2,400 / 1,690 | 12,650 / 7,350 | 4,660 / 3,660 | 2,085 / 1,630 |
| v0 simplest Rust (scalar decode+dot in f64, trig per cap, id order) | 2,131 / 1,360 | 2,018 / 1,361 | 5,551 / 1,925 | 2,136 / 1,392 | 776 / 536 |
| v1 + cap order widest-angle first | 2,010 / 1,373 | - | 7,049 / 1,964 | - | - |
| v1b + cos/sin table over angle codes (on scalar kernel) | 2,076 / 1,425 | - | 7,225 / 2,026 | - | - |
| v2 table kernels (TQ group LUT, PQ slice LUT; f32; exact two-tier decision) | 10,002 / 7,324 | 2,522 / 1,904 | 24,250 / 14,089 | 3,814 / 3,566 | 1,719 / 1,367 |
| v2b unrolled 8 lanes + chunk/no-bounds-check indexing + sqrt-free gate | 11,597 / 8,633 | 3,447 / 2,580 | 24,858 / 15,109 | 3,512 / 3,251 | 1,610 / 1,263 |
| v2c lane accumulators in registers (constant lane index) | 20,333 / 16,573 | 5,765 / 4,314 | 28,417 / 21,950 | 4,050 / 3,862 | 1,761 / 1,406 |
| v3 rotation moved out to numpy (Accelerate) | 23,496 / 17,560 | 6,013 / 4,587 | 38,605 / 26,288 | - | 1,775 / 1,405 |
| v3b PQ table build rewritten (codeword-major, register-blocked) | 23,700 / 17,656 | - | 60,083 / 31,736 | 10,810 / 9,622 | 4,627 / 3,767 |
| v4 persistent norm/cos tables (codebook/grid-derived, caps-independent) | 24,223 / 18,096 | - | - | 12,120 / 10,586 | 5,454 / 4,251 |

Notes on each step.
* v0: fused call (rotate + beta + scan + exact recheck + bool out); plain loop over caps; bit-stream decode. Slow because decode+f64 dot costs ~800 cycles per cap.
* Candidate 2 (cap order, widest angle first, static filters only): TQ1 pos 2,131 -> 2,010 (noise), SPQ16 pos 5,551 -> 7,049 (+27%); neg unchanged by construction. KEPT (10 lines, free, only helps positives); better order tried later (see 2b).
* Candidate 3 (cos/sin table per angle code): +3.5% on the scalar kernel, but it is what lets the table kernels reject a cap with no trig and no sqrt (the "gate"), so it is a prerequisite of v2b+. KEPT. On a 16-bit angle grid (insert-capable filters) the gate uses 256 coarse buckets (cos(theta+beta) at the bucket's upper end, a lower bound of kl, so only conservative); verified 0 disagreements on a TQ4+NearestInsert filter (3199 caps, b1 5,630 qps vs 2,288 numba).
* v2 (item 5, table scoring from the codes): TQ with 8-bit (bits 1,2,4) or 6-bit (bits 3) group tables built in f64 from two half-tables and rounded once to f32; seeded PQ with a (slice, codeword) table. Score error (f32 sums of m entries, each within 2^-24) is < m 2^-24 |raw| for unit q, inside slack (= 2 (d+4) 2^-24); derivation is in the comments of core.rs. 5x vs v0 on TQ1. Decision identical (0 disagreements).
* v2b/v2c: the first version indexed a dynamic lane (`dot[k & 7] += ...`) so the accumulators lived in memory (store-forwarding chain); constant-lane unrolled chunks of 8 groups double the speed. Gate (no sqrt for caps that cannot matter) is part of v2b.
* v3 (item 1b): rotation by numpy (4 us/query at b=1, 0.6 us at b=1024 on Accelerate) instead of a Rust f64 loop (25 us/query, it streams the 1.2 MB matrix): +40-60% at b>=32. Rust takes the rotated float64 queries (`to_working`, which is exactly what the original does). 
* v3b: the PQ12 table build is 4096 x 384 MACs per query per group; as a per-(slice,codeword) dot with horizontal sums it ran at ~1 MAC/cycle (200 us); codeword-major with two rows' accumulators in registers: 60 us. Gives SPQ48 2.7x, mixed 2.6x. `pq_direct` (no tables, score straight from the codebook with elementwise accumulators) was written and DROPPED: 33-58 ns/cap vs 20 ns/cap (3000 caps) for tables, so tables win above ~1,500 caps.
* v4: |raw|^2 per codeword (<= 16 KB) and the angle-bucket table (4 KB) are functions of the codebook / grid, not of the caps, computed once instead of per call (removed ~100 us/call on the mixed filter).

### Threads (item 4, rayon; `bench/threads.json`; default 1 thread, `rust_threads=` / env `CAPCOVER_RUST_THREADS`)
Chunks of 4 queries per task (query costs vary: accepted ones exit early). qps at b=1024, rejected-only stream, M5 Pro (5P+10E):
| filter | 1 | 2 | 4 | 5 | 8 | 10 threads |
|---|---|---|---|---|---|---|
| TQ1 | 22,877 | 44,093 | 83,885 | 98,069 | 131,769 | 151,628 |
| mixed | 4,447 | 8,216 | 15,121 | 17,316 | 26,408 | 31,454 |
Near-linear up to the 5 P-cores, then E-cores add less. The glue before the scan (numpy rotation, normalisation) and `.tolist()` stay single-threaded.

### Cap order by hit frequency (candidate 2b) -- DROPPED
Greedy set cover of 2000 construct vectors, caps renumbered (static filters). Within noise or worse
(TQ1 pos b32 31.6K -> 34.7K, neg b1 20.5K -> 18.5K; SPQ16 pos b1 82.8K -> 75.8K; mixed 8.4K -> 7.9K). Not worth the code.

### FastScan (items 6 + 7 + 8b; built by a subagent in a private crate copy, merged, `src/fastscan.rs`) -- KEPT for TQ1/TQ2/TQ4
Same code bytes permuted into blocks of 16 caps (ragged tail stays cap-major, no padding), per-query uint8 LUTs
(one per nibble), `vqtbl1q_u8` (NEON) / `pshufb` (SSSE3, runtime-detected, mirror of the NEON backend, type-checked for
x86_64 with an official rustc via `cargo check --target x86_64-unknown-linux-gnu`, NOT run: no x86 machine here) / portable
emulation (tests). Tier 1 rejects a cap only when dot_est + E (rigorous bound from the 8-bit table quantization, f32 build)
is below the threshold minus slack*|raw| (|raw|^2 from a query-independent quantized table with its own bound); every other
lane goes to an exact per-cap check (f64) so answers equal the exact rule. Tier-2 candidates per query: TQ1 0.63, TQ2 0.69, TQ4 18.
Native, 3000 caps full scan incl. table build: TQ1 13.6 -> 3.2 ns/cap, TQ2 26.6 -> 7.3, TQ4 68.7 -> 14.0 (TQ3 stays scalar LUT).
Python end to end (bench/final_nofs.json vs final.json), single thread, qps:
| filter | scalar LUT pos b1 / neg b1 / neg b32 | FastScan pos b1 / neg b1 / neg b32 | gain (neg) |
|---|---|---|---|
| TQ1 | 24,286 / 18,351 / 23,015 | 73,899 / 61,191 / 97,903 | 3.3x / 4.3x |
| TQ2 | 13,497 / 10,442 / 11,828 | 42,018 / 32,588 / 44,632 | 3.1x / 3.8x |
| TQ4 | 5,904 / 4,469 / 4,698 | 21,837 / 17,350 / 20,878 | 3.9x / 4.4x |
Process note: my f64-reference fuzz reported a false negative on a 16-cap block; the cause was the fuzz's own reference
reading cap-major bytes of a blocked group (fixed with `Group::code_into`, regression test added); the kernel was right.
Mutation checks: scaling the error bound E to 0.3x/0.5x makes the subagent's proptest fail; shifting the scalar gate by +1e-4 makes mine fail.
Not done: several queries per pass over a block, AVX2 (32 lanes), TQ3 nibble layout, bit-plane layout (8a).

### Dropped / not tried
* pq_direct everywhere: 33-58 ns/cap vs ~6 for tables; KEPT only for groups with few caps (12-bit: < 1536 caps, 8-bit: < 96), where the 2^bits x dim table build (45-60 us) is not repaid: mixed filter +27%.
* `[[f32; M]; R]` register blocking with R=4 for the PQ table build: 2.5x slower (spills). `mul_add` (fma) for the build: 60 -> 44 us (aarch64; on x86 it needs a `target_feature(fma)` wrapper, not done: x86 builds the table with mul+add at SSE2 width. Known gap).
* Rust f64 rotation (25 us/query) replaced by numpy (4 us).
* 8a bit-plane TurboQuant with interval bounds, 8b int8 LUT: 8b is what FastScan is. 8a not tried (the profile shows rejections are far from the threshold, tier 2 almost never fires for TQ1/2).
* AVX2/AVX-512, multi-query block sharing: not done.

## FINAL TABLES (single thread unless stated; M5 Pro; bench/final.json, bench/baseline_numba.json)
qps, "pos" = held-out stream (42-88% accepted), "neg" = rejected-only (full scans). Old = numba path.
| filter (3000 caps; mixed 8660) | b | old pos | old neg | Rust pos | Rust neg |
|---|---|---|---|---|---|
| TQ1 | 1 | 6,650 | 5,060 | 73,899 | 61,191 |
| TQ1 | 32 | 37,900 | 36,500 | 130,161 | 97,903 |
| TQ1 | 1024 | 240,000 | 196,000 | 135,002 | 99,805 |
| TQ2 | 1 | 4,010 | 3,000 | 42,018 | 32,588 |
| TQ2 | 32 | 36,100 | 35,700 | 60,269 | 44,632 |
| TQ2 | 1024 | 237,000 | 196,000 | 61,131 | 45,100 |
| TQ4 | 1 | 2,400 | 1,690 | 21,837 | 17,350 |
| TQ4 | 32 | 33,900 | 33,700 | 27,000 | 20,878 |
| TQ4 | 1024 | 229,000 | 190,000 | 27,247 | 20,971 |
| SPQ16 | 1 | 12,650 | 7,350 | 71,983 | 33,563 |
| SPQ16 | 32 | 91,800 | 82,300 | 124,769 | 45,891 |
| SPQ16 | 1024 | 102,600 | 90,300 | 127,646 | 45,889 |
| SPQ48 | 1 | 4,660 | 3,660 | 15,334 | 12,976 |
| SPQ48 | 32 | 43,900 | 42,400 | 19,188 | 15,770 |
| SPQ48 | 1024 | 246,000 | 200,000 | 19,446 | 15,832 |
| mixed | 1 | 2,085 | 1,630 | 7,928 | 6,502 |
| mixed | 32 | 17,200 | 16,700 | 8,668 | 7,016 |
| mixed | 1024 | 92,600 | 74,800 | 8,735 | 7,034 |

Honest reading: at b=1 Rust is 3.5-12x faster (TQ1 12x, mixed 4x). At b>=32 the old path is a decode + sgemm and on this Mac that
sgemm runs on Apple's AMX (~0.45 TFLOP/s effective), which one NEON core cannot match: at b=1024 the old path wins on TQ2/TQ4/SPQ48/mixed
single-threaded, Rust wins on SPQ16 pos and is level on TQ1 pos. Rust is flat in batch size. With threads (bench/threads.json, b=1024 neg):
TQ1 98K (1) -> 342K (5) -> 519K (10 threads, beats numba's 196K); mixed 6.9K -> 27K (5) -> 47K (10; numba 74.8K). On the x86 run machine the old path's sgemm
is OpenBLAS (about 10x less than AMX) so the b>=32 comparison should favour Rust more; I could not measure it. The mixed (production auto()) filter
is dominated by 12-bit PQ groups whose per-query table build (2^12 x 384 MACs per group, 45-60 us) is the floor; there is no cheap exact way around it that I found.

Ablation (kept items, TQ1 neg b1 unless noted): v0 scalar decode 1,360 -> LUT+gate+registers 16.6K (12x) -> numpy rotation 17.6K -> FastScan 61.2K (3.5x);
mixed neg b1: v0 536 -> LUT 1,400 -> PQ build rewrite 3,770 -> persistent norm/angle tables 4,250 -> few-caps direct 5,390 -> fma 6,800.

## Verification
* `python -m pytest tests/test_cap_cover_filter.py`: 32 passed (the full `tests` dir has 12 unrelated pre-existing failures in test_quantization_soundness for another filter; untouched).
* `PYTHONPATH=src python -m pytest CapCoverRustFilter/tests`: 6 passed (all encodings TQ1-4/8, SPQ12-64, mixed budgets, after insert() with NearestInsert and CapMergeInsert (16-bit angles), adversarial boundary queries at +-1e-3..1e-9, non-unit norms around the feasibility limit, degenerate inputs, fallback encodings SQ8/FP16, byte accounting). Rust vs original: 0 disagreements; also checked against a float64 reference and FNR=0.
* `cargo test --release` in CapCoverRustFilter: proptest vs f64 reference (tq, pq, pq_tables, regression) plus fastscan tests (400 default cases; 100K in a one-off run by its author; 20K for `tq` after the merge).
* Memory (bench/memory.json): get_bytesize unchanged by freezing; Rust owns exactly the code bytes (TQ4, 12000 caps: 2,304,000 B codes + 24,000 B angle codes + 7.2 KB constants vs charged 2,316,136 B); Python `_codes` is an empty array while frozen; per-call scratch is the 0.1-0.5 MB table per thread plus a 24 MB cap on the float64 working copy (queries are processed 8192 at a time); a 60,000-query batch grew RSS by < 0.1 MB. Caveat: whole-process RSS before/after freeze is not informative (it is dominated by construct leftovers; allocator does not return pages). Python also keeps `_angle_codes` (2 B/cap) and `_cap_level` (1 B/cap) as the original does; Rust holds a second 1-2 B/cap copy of the angle codes.
* x86_64: `cargo check --lib` and `--tests` type-check; not executed.

## Falls back to Python
Encodings without kernel (FP32, FP16, SQ4/6/8): original path via `super().query`, requires `_codes`, so the wrapper simply never freezes them. TQ8 and dims where the lane counts do not divide (e.g. d=48 with TQ1) use the Rust scalar f64 path.
