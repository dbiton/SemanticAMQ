import logging
import unittest

import pandas as pd

from plots.plot_paper import drop_over_budget


class DropOverBudgetTest(unittest.TestCase):
    def test_drops_bool_and_string_flags(self):
        for flags in ([False, True, False], ["False", "True", "False"]):
            df = pd.DataFrame({"Config": ["a", "b", "c"], "Over Budget": flags})
            with self.assertLogs("plots.plot_paper", level=logging.WARNING) as cm:
                out = drop_over_budget(df)
            self.assertEqual(list(out["Config"]), ["a", "c"])
            self.assertIn("1 over-budget", cm.output[0])

    def test_missing_column_is_noop(self):
        df = pd.DataFrame({"Config": ["a", "b"]})
        self.assertEqual(len(drop_over_budget(df)), 2)

    def test_none_over_budget_keeps_all(self):
        df = pd.DataFrame({"Config": ["a"], "Over Budget": [False]})
        self.assertEqual(len(drop_over_budget(df)), 1)


if __name__ == "__main__":
    unittest.main()
