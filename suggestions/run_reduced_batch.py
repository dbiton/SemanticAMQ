"""The reduced-scale batch behind the paper's streaming, quantization and equal-memory results
when a run on the 100K-vector machine is not available (run from a throwaway worktree with
datasets/ linked in -- see CLAUDE.md -- because the generators truncate their CSVs):

    python suggestions/run_reduced_batch.py [num_vectors] [workers]

10,000 vectors per corpus (8,000 reference, 2,000 queries), one fold, three quantization
budgets, ten encodings and four equal-memory budgets. Same generators as `python src/paper.py`.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import paper
import experiments
from filters.cap_cover.cap_quantizers import Quantize

n = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
workers = int(sys.argv[2]) if len(sys.argv) > 2 else 18
paper.configure(num_vectors=n, workers=workers)
experiments.BLOCKLIST_FOLDS = 1

paper.PAPER_BUDGET_PCT = (0.5, 2, 10)
paper.PAPER_K_LADDER = [round(pct / 100 * 80_000) for pct in paper.PAPER_BUDGET_PCT]
paper.PAPER_QUANTIZE = (Quantize.FP32, Quantize.SQ8, Quantize.TQ4, Quantize.TQ3, Quantize.TQ2, Quantize.TQ1,
                        Quantize.SPQ12, Quantize.SPQ24, Quantize.SPQ36, Quantize.SPQ64)

if __name__ == "__main__":
    from functools import partial
    generators = [
        paper.GENERATORS["streaming"],
        paper.generate_quantization,
        partial(experiments.generate_equal_memory, filters=paper.PAPER_EQUAL_MEMORY_FILTERS, n_folds=1,
                count_memory_steps=4),
    ]
    paper.set_thresholds()
    paper._submit_and_run(generators, workers)
