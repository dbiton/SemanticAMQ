"""Renames the query-source columns in existing result CSVs, in place.

Query sources were once named for the labels they used to imply
(positive/hard/easy); they are now heldout/withheld/singleton, and every query
is labeled by distance at tau. The quantization, streaming and construction
evaluations already labeled that way, so their CSVs need only the rename --
no re-run. The blocklist and budget sweeps did not, and must be re-run.

    fpr_pos -> fpr_heldout, fpr_hard -> fpr_withheld, fpr_easy -> fpr_singleton

Also drops construction's *_membership columns (source-based labels, kept only
to cross-check the old budget.csv), and adds neg_share_pct where the file
holds what it takes (streaming's n_duplicates / n_queries). Columns a rename
cannot recover -- quantization's fpr_heldout and neg_share_pct, construction's
per-source FPR -- stay absent until the evaluation is re-run.

Usage: python src/util/rename_source_columns.py [CSV ...]
(defaults to every quantization, streaming and construction CSV under results/)
"""

import glob
import sys

import pandas as pd

RENAMES = {"fpr_pos": "fpr_heldout", "fpr_hard": "fpr_withheld", "fpr_easy": "fpr_singleton"}
DROPPED = ["recall_membership", "fpr_hard_membership", "fpr_easy_membership"]
DEFAULT_GLOBS = [
    "results/quantization.csv", "results/quantization/*/quantization.csv",
    "results/streaming.csv", "results/streaming/*/streaming.csv",
    "results/construct_methods/*.csv",
]


def migrate(path: str) -> bool:
    """Rewrites path if it has anything to migrate; returns whether it did."""
    df = pd.read_csv(path)
    if "recall" in df.columns and "fpr_hard" in df.columns:
        raise ValueError(f"{path} is a blocklist/budget sweep, whose labels changed: re-run it")
    out = df.rename(columns=RENAMES).drop(columns=[c for c in DROPPED if c in df.columns])
    if "neg_share_pct" not in out.columns and {"n_duplicates", "n_queries"} <= set(out.columns):
        out["neg_share_pct"] = 100.0 * (1.0 - out["n_duplicates"] / out["n_queries"])
    if list(out.columns) == list(df.columns):
        return False
    out.to_csv(path, index=False)
    return True


def main(paths) -> None:
    paths = paths or sorted(p for g in DEFAULT_GLOBS for p in glob.glob(g))
    for path in paths:
        print(f"{'migrated' if migrate(path) else 'unchanged'}  {path}")


if __name__ == "__main__":
    main(sys.argv[1:])
