import os
import unittest
import tempfile

import pandas as pd

from runner import Runner


class TestRecordResultAtomic(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.csv = os.path.join(self._dir.name, "out.csv")
        self.runner = Runner(num_workers=1)
        self.runner._init_csv(self.csv)

    def test_two_results_accumulate(self):
        self.runner._record_result(self.csv, {"a": 1})
        self.runner._record_result(self.csv, [{"a": 2}])
        self.assertEqual(pd.read_csv(self.csv)["a"].tolist(), [1, 2])
        self.assertEqual(os.listdir(self._dir.name), ["out.csv"])

    def test_failure_keeps_previous_file(self):
        self.runner._record_result(self.csv, {"a": 1})
        with open(self.csv) as f:
            before = f.read()
        real = pd.DataFrame.to_csv

        def partial_then_raise(df, path, *args, **kwargs):
            with open(path, "w") as f:
                f.write("a\n1\n")
            raise OSError("disk full")

        pd.DataFrame.to_csv = partial_then_raise
        try:
            with self.assertRaises(OSError):
                self.runner._record_result(self.csv, {"a": 2})
        finally:
            pd.DataFrame.to_csv = real
        with open(self.csv) as f:
            self.assertEqual(f.read(), before)
        self.assertEqual(os.listdir(self._dir.name), ["out.csv"])


if __name__ == "__main__":
    unittest.main()
