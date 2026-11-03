# FPR optimisation results (branch fpr-opt)

Single corpus (ELI5 8k construct / 2k queries), tau = 10^-0.5, one pinned core, 1 rep (FPR is deterministic).
Scripts: `fpr_profile.py`, `fpr_select.py`, `fpr_spend.py`, `fpr_spend_summary.py`. Raw: `out/fpr_profile.csv`,
`out/fpr_spend_f0.csv`, `out/fpr_spend_f123.csv`; summary `out/fpr_spend_summary.csv` (force-added).

## 1. Slack decomposition (fold 0; gonzalez/minimax x TQ2/TQ4 x 3%/10%, pinned encoding; float64 FPR equals the Rust FPR in all 8 cells)

| Source | Measured |
|---|---|
| (b) angle-grid rounding | stored vs unrounded angles: mean 0.0002-0.002 rad per cap; FPR 0 to 0.55 pp (max: minimax TQ4 3%), about 0.2 pp mean |
| (a) loose assignment | coordinate-descent re-certified angles (every member stays held, float64): FPR change 0 to -0.25 pp (gonzalez TQ2 3%), 0 in 6 of 8 cells |
| (c) beta/tau widening | FPR with beta=0 is 0 to 25 pp vs 3.7 to 99.8 pp stored: widening is nearly all of the FPR, and it is the price of tau (needed for zero FN), not slack |
| (d) unspent budget | pinned TQ2 at 10%: 776,040 of 1,228,800 bytes used (k = n caps already); auto ladders: SPQ36 at 3% uses 296,016 of 368,640 (20% unspent), TQ3 at 10% uses 1,160,072 of 1,228,800 (5.6%) |

Subject A (tighter certified angles / reassignment) is capped at roughly 0.2 pp mean: not pursued. Subject B
(objective alignment): the volume criterion picks a sound encoding in the cells checked (auto equals the best pinned
encoding at minimax 10%: TQ3, 0.0145); at 3% pinned TQ1 (0.222, not in AUTO_CANDIDATES) beats the chosen SPQ36 (0.270);
not pursued (outside budget, TQ1 was removed from the candidates earlier). The only recoverable slack is (d).

## 2. Change: `CapCoverFilter(spend_leftover=True)` (default off; reference path untouched)

After a single encoding is chosen (non-mixed auto), `_spend_leftover` gives the next finer code to the widest caps
(by construction angle), as many as `get_bytesize()` still keeps within `max_memory` (same charging: finest codebook,
other group's codebook plus 4-byte count; binary search on the exact total). Then the usual per-level fit and
re-certification run (`_fit_codes_by_level`, `_tight_angles`), so coverage is certified in float64 as before.
Chosen on a construction-set argument (bytes unspent, widest caps set the volume), then validated once.

## 3. Paired results (spend minus reference; FPR in pp; ladders TQ = TQ2/3/4, SPQ = default AUTO_CANDIDATES; gonzalez and minimax)

| Group | Ladder | Budget | Cells | mean dFPR | worst dFPR | construct time ratio mean / max |
|---|---|---|---|---|---|---|
| fold 0 | TQ | 3% | 2 | 0.00 | 0.00 | 0.84 / 1.06 |
| fold 0 | TQ | 10% | 2 | -0.15 | -0.15 | 0.99 / 1.01 |
| fold 0 | SPQ | 3% | 2 | -8.76 | -8.76 | 1.23 / 1.24 |
| fold 0 | SPQ | 10% | 2 | -0.15 | -0.15 | 1.02 / 1.07 |
| folds 1-3 | TQ | 3% | 6 | 0.00 | 0.00 | 0.95 / 1.06 |
| folds 1-3 | TQ | 10% | 6 | -0.15 | -0.10 | 0.98 / 1.03 |
| folds 1-3 | SPQ | 3% | 6 | -8.67 | -7.91 | 1.26 / 1.31 |
| folds 1-3 | SPQ | 10% | 6 | -0.15 | -0.10 | 0.97 / 1.07 |

Overall folds 1-3 (24 cells): mean -2.24 pp, worst 0.00 (no cell worse); fold 0 (8 cells): mean -2.27 pp. Per encoding:
TQ (TQ2 3% has no finer-code room: identical; TQ3 to TQ4 at 10%) -0.075 pp mean; SPQ36 to SPQ48 at 3% -8.7 pp. FN = 0 in
all 64 builds; every build within `max_memory` (`ok` = 1). The 8.7 pp is in the SPQ ladder at 3%, where the discrete
ladder leaves 20% of the budget idle; in the plan's pinned TQ2/TQ4 signal cells the effect is 0 to -0.2 pp, so the plan's
"mean >= 1 pp on TQ2/TQ4" target is not met on those cells (no leftover to spend: TQ2 3% and TQ4 are budget-saturated).

## 4. Gates

- Zero FN and budget: pass on all runs above. Tests: `tests/test_spend_leftover.py` (new: budget, FN, tiny/duplicate/
  k near n inputs, flag-off equals reference), `test_cap_cover_filter.py`, `test_quantization_soundness.py`,
  `test_soft_anneal_construct.py`, `test_adversarial_construct.py`: 73 passed, 1 skipped.
- Construction time: SPQ 3% +23-31% (under 1.5x, and buys 8.7 pp); other cells within +-7% (single rep, noise is of that order).
- Query throughput: not measured (`CapCoverRustFilter/bench` not run). Caps at the finer level move to a second scan
  group at 12-48 more bytes per cap; this is a risk for the scan-cost cap (`scan_limit_ms`) and must be checked.
- Soft-anneal path not exercised with the flag (not run).

## 5. Limitations and recommendation

One corpus, 8k/2k, one rep, two construct methods; the tested budgets are 3% and 10% only. The flag was developed on
fold 0 only through the single design, folds 1-3 ran once. Recommendation: keep opt-in (`spend_leftover=False` default);
enable for the paper's auto configuration only after a query-throughput check and a run at other budgets/corpora.
Rounding and reassignment slack are too small (about 0.2 pp) to justify further work on angle codes.
