#!/bin/bash
# Controlled old-vs-new comparison. Usage: bench_construct/run_final.sh OLD_TREE_DIR PYTHON
# Runs sequentially on one pinned core of an otherwise idle machine; outputs under bench_construct/out/.
OLD=$1; PY=$2; NEW=$(pwd); OUT=$NEW/bench_construct/out
export ELI5_PKL=${ELI5_PKL:?set ELI5_PKL}
for part in "gm|gonzalez,minimax|FP32,TQ2,TQ4|0.01,0.03,0.10|3" "sa|soft_anneal|FP32,TQ4|0.01,0.03|1"; do
  IFS='|' read tag methods quant budgets reps <<< "$part"
  for side in old new; do
    src=$NEW/src; [ $side = old ] && src=$OLD/src
    PYTHONPATH=$src $PY $NEW/bench_construct/bench.py --out $OUT/final_${tag}_${side}.csv --methods $methods \
      --quant $quant --budgets $budgets --reps $reps --core 1 > $OUT/final_${tag}_${side}.log 2>&1
  done
done
echo done > $OUT/final_done.txt
