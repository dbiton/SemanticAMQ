import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plots import plot_paper as plots


class TestThroughputProjection(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def frame(self, **changes):
        row = dict(corpus="Quora", n_queries=100, positive_rate_pct=50,
                   throughput_without_qps=100, throughput_with_qps=1000,
                   throughput_improvement_pct=900, filter_query_s=0.2)
        row.update(changes)
        return pd.DataFrame([row])

    def test_uses_direct_filter_cost_even_when_residual_would_be_negative(self):
        _, ax = plt.subplots()
        plots._backend_cost_panel(ax, self.frame())
        expected = 100 * (plots.BACKEND_COSTS / (0.002 + 0.5 * plots.BACKEND_COSTS) - 1)
        np.testing.assert_allclose(ax.lines[0].get_ydata(), expected)
        self.assertLess(expected[0], 0)

    def test_zero_acceptances_are_not_floored(self):
        _, ax = plt.subplots()
        plots._backend_cost_panel(ax, self.frame(positive_rate_pct=0))
        np.testing.assert_allclose(ax.lines[0].get_ydata(),
                                   100 * (plots.BACKEND_COSTS / 0.002 - 1))

    def test_legacy_rows_do_not_invent_filter_timings(self):
        _, ax = plt.subplots()
        plots._backend_cost_panel(ax, self.frame().drop(columns="filter_query_s"))
        self.assertEqual(len(ax.lines), 0)
        self.assertTrue(any("timing unavailable" in t.get_text() for t in ax.texts))

    def test_negative_filter_timing_is_rejected_not_clipped(self):
        _, ax = plt.subplots()
        with self.assertRaisesRegex(ValueError, "filter_query_s"):
            plots._backend_cost_panel(ax, self.frame(filter_query_s=-0.1))
