# Opt-in MinimaxConstruct variants: quality/speed evaluation

Script: `bench_construct/variants.py` (paired, in-process, core 3, 1 BLAS thread; variant order rotated per rep),
summary: `bench_construct/variants_summary.py`. CSVs in `bench_construct/out/` (gitignored: `f0_TQ2.csv`, `f0_TQ4.csv`
(also holds FP32), `f1..f3.csv`, `variants_cells.csv`).

Setup: ELI5 10k pickle, 8000 construct / 2000 queries, tau = 10^-0.5 (L2), truth = exact nearest-construct
distance (faiss IndexFlatL2). Folds: 0 = original split; 1, 2, 3 = `default_rng(seed).permutation` of the 10000 rows,
same split. Encodings FP32/TQ2/TQ4 x budgets {1,3,5,7,10}% = 15 cells/fold, 60 paired cells per variant (40 in the
signal set TQ2+TQ4). Timing: fold 0 has 3 reps (median of per-rep exact/variant ratios); folds 1-3 have 1 rep
(FPR and construct objective are deterministic, timing is noisier). FN = 0 in every run.
Variants: `adaptive` = `MinimaxConstruct(adaptive=True)`; `approx` = `MinimaxConstruct(approx_assign=True)`.

## Gates (not loosened): mean dFPR <= 0.5 pp; no cell > 2 pp

| variant  | cells | mean dFPR pp | worst cell dFPR pp | gate | n_caps changed | mean d sum theta^8 (rel) | mean d psi_mean | median construct speedup |
|----------|-------|-------------:|-------------------:|------|---------------:|-------------------------:|----------------:|-------------------------:|
| adaptive | all 60 | +0.003 | +0.15 | pass | 0 | +0.003% | ~0 | 1.05x |
| adaptive | TQ2+TQ4 (40) | +0.005 | +0.15 | pass | 0 | +0.005% | +2e-6 | 1.07x (IQR 1.00-1.10) |
| approx   | all 60 | +0.078 | +0.90 | pass | 0 | -0.20% | | 1.00x |
| approx   | TQ2+TQ4 (40) | +0.120 | +0.90 | pass | 0 | +0.10% | -7.8e-4 | 0.97x (IQR 0.91-1.03) |

Per encoding (mean / worst dFPR pp): adaptive TQ2 0.000 / +0.05, TQ4 +0.010 / +0.15, FP32 0 / 0;
approx TQ2 +0.043 / +0.60, TQ4 +0.198 / +0.90, FP32 -0.008 / +0.15.

Per encoding x budget (mean over the 4 folds; exact-path FPR in %; speed = exact/variant construct_s):

| enc | budget | exact FPR | adaptive dFPR pp | adaptive speed | approx dFPR pp | approx d theta8 | approx speed |
|-----|-------:|----------:|-----------------:|---------------:|---------------:|----------------:|-------------:|
| TQ2 | 1%  | 99.1 | 0.000 | 1.00 | +0.075 | -0.10% | 1.02 |
| TQ2 | 3%  | 63.9 | +0.013 | 1.03 | +0.125 | +0.16% | 0.99 |
| TQ2 | 5%  | 19.0 | -0.013 | 1.11 | +0.013 | +0.20% | 0.91 |
| TQ2 | 7%  | 3.5  | 0.000 | 0.94 | 0.000 | 0 | 0.89 |
| TQ2 | 10% | 3.5  | 0.000 | 0.90 | 0.000 | 0 | 0.93 |
| TQ4 | 1%  | 99.9 | 0.000 | 1.13 | -0.038 | -0.58% | 1.24 |
| TQ4 | 3%  | 91.0 | +0.013 | 1.05 | +0.163 | +0.11% | 1.01 |
| TQ4 | 5%  | 65.2 | +0.038 | 1.12 | +0.576 | +0.38% | 1.02 |
| TQ4 | 7%  | 39.5 | 0.000 | 1.09 | +0.225 | +0.43% | 0.94 |
| TQ4 | 10% | 12.3 | 0.000 | 1.07 | +0.063 | +0.44% | 0.87 |

n_caps is identical to the exact path in every cell (same budget fits the same count); d psi_mean is |<1e-3| rad
everywhere. Exact-path 8k construct times at the signal cells are 3-10 s (TQ2 5%: 10.0 s, TQ4 10%: 9.8 s).
FP32 and 1% cells have FPR ~1.0 and carry no signal.

## Reading
* adaptive: result is practically identical to exact (objective +0.005%, 37/40 signal cells with dFPR = 0, max
  +0.15 pp), but the gain is only ~1.05x median (range 0.82-1.21x, within noise at 1 rep); at larger k the exact
  path's assignment reuse already makes later rounds cheap, so stopping early saves little.
* approx: passes the gates but is slower than exact on the signal cells (median 0.97x, 0.87-0.94x at the high
  budgets) and slightly worse: signal-cell dFPR > 0 in 20 of 40 cells and < 0 in 9, TQ4 mean +0.20 pp, construction
  objective +0.10% (TQ4 up to +0.44%). The shortlist rescoring costs as much as the reuse-based exact search saves.
* Gate definitions are the proposed operational ones; FPR is validation only (thresholds judged on construction
  objective: neither variant moved sum theta^8 or n_caps materially). The soft-anneal path was not run.
