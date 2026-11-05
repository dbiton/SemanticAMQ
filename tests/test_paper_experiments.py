"""paper.py: evaluation registry, missing-evaluation check, command line, side-effect-free collection."""
import unittest

class TestMissingEvaluations(unittest.TestCase):

    def test_names_the_evaluations_without_all_their_csvs(self):
        import tempfile
        from pathlib import Path
        import paper
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp)
            for name in ("budget.csv", "blocklist.csv", "end_to_end.csv"):
                (results / name).write_text("x\n1\n")
            for corpus in ("A", "B"):
                (results / "construct_methods").mkdir(exist_ok=True)
                (results / "construct_methods" / f"{corpus}.csv").write_text("x\n1\n")
            self.assertEqual(paper.missing_evaluations(results, n_corpora=2),
                             ["quantization", "equal-memory"])
            (results / "cap_quantization" / "A").mkdir(parents=True)
            (results / "cap_quantization" / "A" / "cap_quantization.csv").touch()       # an empty file is not a result
            (results / "equal_memory" / "A").mkdir(parents=True)
            (results / "equal_memory" / "A" / "equal_memory.csv").write_text("x\n1\n")      # one corpus of two: still missing
            self.assertIn("equal-memory", paper.missing_evaluations(results, n_corpora=2))
            self.assertEqual(paper.missing_evaluations(results, n_corpora=1), ["quantization"])


class TestCommandLine(unittest.TestCase):

    def test_parses_subset_and_plot(self):
        import paper
        args = paper.build_parser().parse_args(["--only", "budget,streaming", "--num-vectors", "1000"])
        self.assertEqual((args.command, args.only, args.num_vectors), ("run", "budget,streaming", 1000))
        args = paper.build_parser().parse_args(["plot", "sweeps", "--results", "r"])
        self.assertEqual((args.command, args.figures), ("plot", ["sweeps"]))

    def test_unknown_evaluation_is_rejected_before_anything_runs(self):
        import paper
        with self.assertRaises(SystemExit):
            paper.run_evaluations(["nope"])

    def test_every_evaluation_names_a_registered_mode_or_generator(self):
        import paper
        for e in paper.EVALUATIONS.values():
            self.assertTrue(callable(e.generate))
            self.assertTrue(e.outputs)


class TestSubmitHasNoSideEffects(unittest.TestCase):

    def test_submit_keeps_existing_output_until_run(self):
        import os, tempfile
        from runner import Runner
        with tempfile.TemporaryDirectory() as tmp:
            csv = os.path.join(tmp, "out.csv")
            with open(csv, "w") as f:
                f.write("a\n1\n")
            r = Runner(num_workers=1)
            r.submit("blocklist", output_csv=csv, dim=2)
            with open(csv) as f:
                self.assertEqual(f.read(), "a\n1\n")
