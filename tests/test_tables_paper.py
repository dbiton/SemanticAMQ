"""plots.tables_paper and plots.table_quantization: the LaTeX they write has one table row per line
(no stray line breaks inside a cell) and the new columns and rows are present."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from plots import table_quantization, tables_paper
from plots import plot_paper


def _latency_rows():
    base = dict(delta=0.05, n_reference=20000, filter_mean_ms=3.9, filter_p99_ms=5.4, accept_rate_all_pct=11.5,
                backend_mean_ms=1.2, throughput_improvement_pct=-70.0, breakeven_backend_ms=4.4)
    return pd.DataFrame([dict(base, dataset="wildchat"),
                         dict(base, dataset="quora", n_reference=8000, throughput_improvement_pct=250.0,
                              breakeven_backend_ms=0.3)])


class TestLatencyTable(unittest.TestCase):

    def test_rows_and_columns(self):
        tex = tables_paper.latency_table(_latency_rows())
        lines = tex.strip().split("\n")
        rows = [ln for ln in lines if ln.startswith(("WildChat", "Quora"))]
        self.assertEqual(len(rows), 2)
        for ln in rows:
            self.assertTrue(ln.endswith(chr(92) * 2), ln)               # a cell never breaks the row
            self.assertEqual(ln.count("&"), 6)
        self.assertIn("textminus 70", rows[0])
        self.assertIn("(20K)", rows[0])

    def test_no_break_even_when_nothing_is_rejected(self):
        d = _latency_rows()
        d["breakeven_backend_ms"] = np.inf
        self.assertIn("& -- &", tables_paper.latency_table(d))


class TestAblationTable(unittest.TestCase):

    def test_skips_variants_the_csv_predates(self):
        rows = []
        for arm, fpr in (("full", 0.5), ("minimax", 0.6)):
            for ds in ("a", "b"):
                for kf in (0.01, 0.02):
                    rows.append(dict(arm=arm, dataset=ds, fold=0, memory_fraction=kf, delta=0.05, fpr=fpr))
        tex = tables_paper.ablation_table(pd.DataFrame(rows))
        self.assertIn("minimax", tex)
        self.assertNotIn("soft anneal", tex)


class TestQuantizationTable(unittest.TestCase):

    def _frame(self):
        rows = []
        for q, saved in (("TQ4", 40.0), ("TQ2", 60.0)):
            for mf, s in ((0.005, saved), (0.02, saved + 20)):
                rows.append(dict(quantize=q, memory_fraction=mf, n_construct=8000, n_caps=int(mf * 8000) * 5,
                                 index_accesses_saved_pct=s,
                                 query_throughput_qps=1.0))
        return pd.DataFrame(rows)

    def test_qps_column_comes_from_the_given_map(self):
        tex = table_quantization.table(self._frame(), {"TQ4": 12906.0, "TQ2": 9135.0})
        self.assertIn("q/s", tex)
        self.assertIn("12{,}906", tex)
        self.assertNotIn("q/s", table_quantization.table(self._frame()))

    def test_lists_every_auto_candidate_and_the_auto_row(self):
        rows = []
        for q in ("TQ4", "TQ3", "TQ2", "SPQ64", "SPQ12", "auto"):
            rows.append(dict(quantize=q, memory_fraction=0.005, n_construct=8000, n_caps=200,
                             index_accesses_saved_pct=50.0, query_throughput_qps=1.0))
        lines = table_quantization.table(pd.DataFrame(rows)).splitlines()
        body = lines[lines.index(r"\midrule") + 1:]
        self.assertTrue(body[-3].startswith("auto") and "-- &" in body[-3])      # then \bottomrule, \end{tabular}
        self.assertEqual(body[-4], r"\midrule")
        for enc in ("TQ4", "TQ3", "TQ2", "SPQ $6{\\times}8$", "SPQ $48{\\times}12$"):
            self.assertTrue(any(ln.startswith(enc) for ln in body), enc)

    def test_only_the_papers_budgets_are_columns(self):
        f = self._frame()
        extra = f.assign(memory_fraction=0.001, index_accesses_saved_pct=1.0)       # 0.1% of the raw index
        tex = table_quantization.table(pd.concat([f, extra], ignore_index=True), budgets=(0.5, 2))
        header = [ln for ln in tex.splitlines() if ln.startswith("Encoding")][0]
        self.assertEqual(header.count("&"), 3)          # b/dim and the two budgets
        self.assertNotIn("0.1 &", header)


class TestPaperFoldAggregation(unittest.TestCase):

    def test_averages_folds_before_combining_lsh_schemes(self):
        rows = []
        for config, fprs in (("CapCoverFilter", (0.2, 0.8)),
                             ("PacukLSHFilter", (0.1, 0.9)),
                             ("SankowskiWygockiFilter", (0.35, 0.45))):
            for fpr in fprs:
                rows.append({"Config": config, "Dataset": "quora", "Memory Budget": 1536,
                             "Construct Size": 100, "FPR": fpr, "FNR": 0.0, "Infeasible": False})
        with patch.object(plot_paper, "read_all", return_value=pd.DataFrame(rows)):
            result = plot_paper._equal_memory_frame().set_index("method")
        self.assertAlmostEqual(result.loc["Cap-Cover", "FPR"], 0.5)
        self.assertAlmostEqual(result.loc["Zero-FN LSH", "FPR"], 0.4)
        self.assertAlmostEqual(result.loc["Cap-Cover", "pct_raw"], 0.01)


if __name__ == "__main__":
    unittest.main()
