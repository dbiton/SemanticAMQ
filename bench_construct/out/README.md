# Raw benchmark outputs

- `baseline.csv` / `baseline.log`: reference timings of the *original* code (master 47e3bd4, see
  `baseline_commit.txt`), 8,000 ELI5 construct vectors / 2,000 queries, 3 methods x {FP32,TQ2,TQ4} x
  budgets {1%,3%,10%}, one rep, one pinned core (core 0).
- **Caveat:** these were recorded while up to four other agent jobs shared the 4 cores of the machine
  (memory-bandwidth and cache contention), so the absolute times overstate the old code's cost. Ratios against
  them are indicative, not controlled. Use paired in-process comparisons (bench_construct/variants.py,
  redteam/ scripts) for speedup claims.
- `current_g_m.csv`: Gonzalez/minimax FP32+TQ4 after the first kernel merge (intermediate, same caveat).
