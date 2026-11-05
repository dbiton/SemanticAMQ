import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plots import plot_paper as plots


class TestPaperPlotPresentation(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_paper_construction_uses_only_unquantized_results(self):
        data = pd.DataFrame([
            {"dataset": "quora", "method": "soft_anneal", "quantize": encoding,
             "memory_fraction": budget, "fpr": fpr, "tau": 0.2}
            for encoding, fpr in (("FP32", 0.7), ("TQ2", 0.2), ("TQ4", 0.1))
            for budget in (0.01, 0.02)
        ])
        with patch.object(plots, "read_all", return_value=data), patch.object(plots, "save") as save:
            plots.construction()
        fig = save.call_args.args[0]
        quora = fig.axes[plots.CORPORA.index("Quora")]
        curves = [line for line in quora.lines if len(line.get_xdata())]
        self.assertTrue(curves)
        for line in curves:
            np.testing.assert_allclose(np.asarray(line.get_ydata(), dtype=float), 0.7)
        self.assertIn("soft anneal", [t.get_text() for t in fig.legends[0].texts])

    def test_legacy_throughput_keeps_requested_threshold_with_one_budget(self):
        data = pd.DataFrame([
            {"dataset": "quora", "seed": 0, "k_fraction": kf, "n_reference": 100,
             "memory_bytes": memory, "tau": tau, "n_queries": 100,
             "positive_rate_pct": 20, "throughput_without_qps": 100,
             "throughput_with_qps": 200, "throughput_improvement_pct": 100}
            for kf, memory, tau in ((0.05, 1000, 0.3), (0.1, 2000, 0.3),
                                    (0.1, 2000, plots.E2E_TAU))
        ])
        original = data.copy(deep=True)
        with patch.object(plots.pd, "read_csv", return_value=data.copy()), patch.object(plots, "save") as save:
            plots.throughput()
        fig = save.call_args.args[0]
        self.assertIn("0.05", fig.axes[1].get_title())
        self.assertIn("0.05", fig.axes[2].get_title())
        for ax in (fig.axes[0], fig.axes[2]):
            self.assertIn(plots.MEMORY_RATIO, ax.get_title())
            self.assertIn("1.30", ax.get_title())
            self.assertNotIn("k/|X|", ax.get_title())
        np.testing.assert_allclose(np.asarray(fig.axes[1].lines[0].get_xdata(), dtype=float),
                                   np.array([2000]) / (100 * 384 * 4))
        pd.testing.assert_frame_equal(data, original)

    def test_crowded_ticks_are_pruned_without_changing_the_curve(self):
        fig, ax = plt.subplots(figsize=(1.2, 1.2))
        x = np.array([0.02, 0.05, 0.1, 0.2])
        ax.plot(x, [1, 2, 3, 4])
        ax.set_xlim(0, 0.2)
        ax.set_xticks(x)
        ax.set_xticklabels(["0.02", "0.05", "0.1", "0.2"])
        plots._readable_axes(fig)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        boxes = [label.get_window_extent(renderer) for label in ax.get_xticklabels()]
        self.assertTrue(all(a.x1 < b.x0 for a, b in zip(boxes, boxes[1:])))
        np.testing.assert_array_equal(ax.lines[0].get_xdata(), x)
        np.testing.assert_array_equal(ax.lines[0].get_ydata(), [1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
