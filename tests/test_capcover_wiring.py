"""Every evaluation main.generate_and_plot_all() runs must use CapCoverFilter.

Static check: no generator is called, since generators truncate their output
CSVs (see CLAUDE.md, "Generators delete outputs"). Instead the AST of main.py
gives the generators generate_and_plot_all() invokes, the runner modes each one
submits, and the worker each mode maps to; those workers' source must not build
a BallCoverFilter.
"""
import ast
import inspect
import os
import textwrap
import unittest

import runner

MAIN_PATH = os.path.join(os.path.dirname(__file__), "..", "src", "main.py")
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
    raise AssertionError(f"main.py has no function {name!r}")


def _called_generators(tree: ast.Module) -> list:
    """Names of generate_* functions called anywhere inside generate_and_plot_all."""
    body = _function(tree, "generate_and_plot_all")
    names = []
    for node in ast.walk(body):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id.startswith("generate_")):
            names.append(node.func.id)
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

    def test_generate_and_plot_all_invokes_generators(self):
        self.assertTrue(self.generators, "generate_and_plot_all calls no generators")

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

    def test_plot_all_reads_cap_cover_outputs(self):
        plot_all = _function(self.tree, "plot_all")
        called = {n.func.id for n in ast.walk(plot_all)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        self.assertIn("plot_cap_quantization", called)
        self.assertIn("plot_cap_cover_system_tau", called)
        self.assertNotIn("plot_quantization", called)


if __name__ == "__main__":
    unittest.main()
