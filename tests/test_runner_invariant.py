import os
import tempfile
import unittest
from unittest import mock

import pandas as pd

import runner
from runner import InvariantViolation, Runner


def _fake_worker(tag: str, bad: bool = False):
    if bad:
        raise InvariantViolation(f"fnr>0 in {tag}")
    return [{"tag": tag}]


class TestRunnerInvariant(unittest.TestCase):
    def test_violation_fails_run_after_other_jobs_recorded(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict(runner._WORKER_FNS, {"fake": _fake_worker}):
            out = os.path.join(tmp, "out.csv")
            r = Runner(num_workers=1)
            r.submit("fake", output_csv=out, tag="a")
            r.submit("fake", output_csv=out, tag="bad", bad=True)
            r.submit("fake", output_csv=out, tag="b")
            with self.assertRaises(InvariantViolation) as cm:
                r.run()
            self.assertIn("bad", str(cm.exception))
            self.assertEqual(sorted(pd.read_csv(out)["tag"]), ["a", "b"])

    def test_plain_failure_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict(runner._WORKER_FNS, {"fake": _fake_worker}):
            out = os.path.join(tmp, "out.csv")
            r = Runner(num_workers=1)
            r.submit("fake", output_csv=out, tag="a")
            r.submit("fake", output_csv=out)  # missing arg -> TypeError
            self.assertEqual(len(r.run()), 1)

    def test_pickles(self):
        import pickle
        e = pickle.loads(pickle.dumps(InvariantViolation("x")))
        self.assertIsInstance(e, InvariantViolation)
