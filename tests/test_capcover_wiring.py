"""Every evaluation paper.py runs (paper.EVALUATIONS) must use CapCoverFilter.

Static check, no generator is called. The AST of experiments.py gives, for each generator paper.EVALUATIONS wires in, the runner modes each one
submits, and the worker each mode maps to; those workers' source must not build
a BallCoverFilter.
"""
import ast
import inspect
import os
import textwrap
import unittest

import runner

MAIN_PATH = os.path.join(os.path.dirname(__file__), "..", "src", "experiments.py")
BALL_COVER_NAMES = ("BallCoverFilter", "_ball_cover")


def _code_names(func) -> set:
    """Identifiers a function's code references, docstring excluded (prose may
    name BallCoverFilter as a comparison without building one)."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    fn = tree.body[0]
    if fn.body and isinstance(fn.body[0], ast.Expr) and isinstance(getattr(fn.body[0], "value", None), ast.Constant):
        fn.body = fn.body[1:]
    return {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}


def _main_tree() -> ast.Module:
    with open(MAIN_PATH, encoding="utf-8") as f:
        return ast.parse(f.read(), filename=MAIN_PATH)


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"experiments.py has no function {name!r}")


def _called_generators(tree: ast.Module) -> list:
    """Names of the experiments.py generate_* functions paper.EVALUATIONS runs."""
    import paper    # lazily: importing it sets CAPCOVER_SCAN_LIMIT_MS for the whole session
    names = []
    for evaluation in paper.EVALUATIONS.values():
        gen = evaluation.generate
        name = getattr(gen, "__name__", None) or gen.func.__name__
        if any(isinstance(n, ast.FunctionDef) and n.name == name for n in tree.body):
            names.append(name)
    return names


def _submitted_modes(generator: ast.FunctionDef) -> tuple:
    """(mode literals, names referenced in submit kwargs) for every runner.submit."""
    modes, kwarg_names = [], set()
    for node in ast.walk(generator):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "submit"):
            if node.args and isinstance(node.args[0], ast.Constant):
                modes.append(node.args[0].value)
            for kw in node.keywords:
                kwarg_names.update(n.id for n in ast.walk(kw.value) if isinstance(n, ast.Name))
    return modes, kwarg_names


class TestCapCoverWiring(unittest.TestCase):

    def setUp(self):
        self.tree = _main_tree()
        self.generators = _called_generators(self.tree)

    def test_paper_runs_generators(self):
        self.assertTrue(self.generators, "paper.EVALUATIONS wires no experiments.py generator")

    def test_no_enabled_generator_submits_ball_cover_filter(self):
        for name in self.generators:
            modes, kwarg_names = _submitted_modes(_function(self.tree, name))
            self.assertNotIn("BallCoverFilter", kwarg_names,
                             f"{name} passes BallCoverFilter as a submit kwarg")
            self.assertNotIn("_ball_cover", kwarg_names, f"{name} submits a ball cover")
            for mode in modes:
                self.assertIn(mode, runner._WORKER_FNS, f"{name} submits unknown mode {mode!r}")

    def test_no_enabled_worker_builds_ball_cover(self):
        for name in self.generators:
            modes, _ = _submitted_modes(_function(self.tree, name))
            for mode in modes:
                used = _code_names(runner._WORKER_FNS[mode]) & set(BALL_COVER_NAMES)
                self.assertFalse(used, f"{name} -> mode {mode!r} worker code uses {sorted(used)}")


if __name__ == "__main__":
    unittest.main()
