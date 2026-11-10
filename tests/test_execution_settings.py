"""Worker pinning, timing labels and the results-folder guard (paper.py / runner.py / util.cpu_topology)."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import experiments
import runner
from util import cpu_topology


class PinningTests(unittest.TestCase):
    def test_pin_list_beats_core_count(self):
        with patch.dict(os.environ, {"CAPCOVER_PIN_CPUS": "5, 9,12", "CAPCOVER_PIN_CORES": "2"}):
            self.assertEqual(runner.pin_cpus_from_env(), [5, 9, 12])
        with patch.dict(os.environ, {"CAPCOVER_PIN_CORES": "3"}, clear=False):
            os.environ.pop("CAPCOVER_PIN_CPUS", None)
            self.assertEqual(runner.pin_cpus_from_env(), [0, 1, 2])

    def test_unset_means_no_pinning(self):
        with patch.dict(os.environ, {}, clear=False):
            for k in ("CAPCOVER_PIN_CPUS", "CAPCOVER_PIN_CORES"):
                os.environ.pop(k, None)
            self.assertEqual(runner.pin_cpus_from_env(), [])
            before = os.sched_getaffinity(0)
            runner._pin_to_own_core()
            self.assertEqual(os.sched_getaffinity(0), before)

    @unittest.skipUnless(hasattr(os, "sched_setaffinity"), "Linux only")
    def test_worker_pins_to_listed_cpu_and_labels_timings(self):
        original = os.sched_getaffinity(0)
        cpu = sorted(original)[-1]
        try:
            with patch.dict(os.environ, {"CAPCOVER_PIN_CPUS": str(cpu), "CAPCOVER_TIMING_CONDITION": "concurrent",
                                         "CAPCOVER_TIMING_WORKERS": "25"}):
                runner._pin_to_own_core()
                ctx = runner._timing_context()
            self.assertEqual(os.sched_getaffinity(0), {cpu})
            self.assertEqual((ctx["timing_condition"], ctx["timing_concurrent_jobs"], ctx["pinned_cpu"]),
                             ("concurrent", 25, cpu))
        finally:
            os.sched_setaffinity(0, original)

    @unittest.skipUnless(os.path.isdir("/sys/devices/system/cpu"), "Linux only")
    def test_spread_cpus_are_distinct_physical_cores(self):
        n = len(cpu_topology.physical_cores())
        cpus = cpu_topology.spread_cpus(n)
        self.assertEqual(len(set(cpus)), n)
        with self.assertRaises(ValueError):
            cpu_topology.spread_cpus(n + 1)


class ResultsGuardTests(unittest.TestCase):
    def test_refuses_nonempty_results_but_not_empty_folder(self):
        with patch.dict(os.environ):      # importing paper sets CAPCOVER_SCAN_LIMIT_MS; do not leak it
            import paper
        old = experiments.RESULTS_DIR
        try:
            with tempfile.TemporaryDirectory() as d:
                experiments.RESULTS_DIR = d
                paper.refuse_existing_results(["budget", "end-to-end"])          # nothing there: fine
                Path(d, "budget.csv").write_text("a,b\n1,2\n")
                with self.assertRaises(SystemExit):
                    paper.refuse_existing_results(["budget"])
                paper.refuse_existing_results(["end-to-end"])                    # other evaluation: fine
        finally:
            experiments.RESULTS_DIR = old


if __name__ == "__main__":
    unittest.main()
