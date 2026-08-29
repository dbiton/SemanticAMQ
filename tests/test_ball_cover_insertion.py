import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'src'))
import inspect
import unittest

import numpy as np

from src.filters.ball_cover.ball_cover_insertion import (
    INSERTION_STRATEGIES,
    insert_nearest,
    insert_least_inflation,
    insert_merge,
)

DIM = 8


def _rand(n: int, dim: int = DIM, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, dim)).astype(np.float32)


def _covering_ok(vecs: np.ndarray, centers: np.ndarray, radii: np.ndarray, tol: float = 1e-3) -> bool:
    """Same min-over-*every*-ball check BallCoverFilter.distance_to_balls_
    surfaces actually performs -- not just "nearest ball by raw distance",
    since a farther ball with a larger radius can still be the one that
    covers a point."""
    if len(centers) == 0:
        return len(vecs) == 0
    dist = np.linalg.norm(vecs[:, None, :] - centers[None, :, :], axis=2)
    return bool(np.all((dist - radii[None, :]).min(axis=1) <= tol))


def _initial_cover(n: int = 30, dim: int = DIM, seed: int = 0):
    """A small, non-trivial starting cover: every point its own zero-radius
    ball, which insert_* can then be exercised against."""
    return _rand(n, dim, seed).copy(), np.zeros(n, dtype=np.float32)


# ---------------------------------------------------------------------------
# Interface contract: identical signature, statelessness.
# ---------------------------------------------------------------------------

class TestUniformInterface(unittest.TestCase):

    def test_all_strategies_share_the_exact_same_signature(self):
        signatures = {name: inspect.signature(fn) for name, fn in INSERTION_STRATEGIES.items()}
        first_name, first_sig = next(iter(signatures.items()))
        for name, sig in signatures.items():
            self.assertEqual(
                list(sig.parameters), list(first_sig.parameters),
                f"{name} parameter names differ from {first_name}",
            )

    def test_all_strategies_are_interchangeable_through_the_dispatch_table(self):
        """Every strategy must accept being called with the exact same
        positional arguments -- the whole point of a uniform signature is
        that no caller needs to special-case any one of them."""
        centers, radii = _initial_cover()
        new_vectors = _rand(5, seed=99)
        for name, fn in INSERTION_STRATEGIES.items():
            c, r = fn(centers, radii, new_vectors, 40)
            self.assertIsInstance(c, np.ndarray, name)
            self.assertIsInstance(r, np.ndarray, name)

    def test_stateless_same_inputs_give_same_outputs(self):
        """No hidden state: calling a strategy twice with identical inputs,
        with an unrelated call to a *different* strategy (and different
        data) in between, must not change the result."""
        centers, radii = _initial_cover(seed=1)
        new_vectors = _rand(10, seed=2)

        for name, fn in INSERTION_STRATEGIES.items():
            c1, r1 = fn(centers, radii, new_vectors, 40)

            # Perturb global state a different strategy might (incorrectly)
            # depend on: call every other strategy with unrelated data.
            for other_name, other_fn in INSERTION_STRATEGIES.items():
                if other_name != name:
                    other_fn(*_initial_cover(seed=7), _rand(20, seed=8), 25)

            c2, r2 = fn(centers, radii, new_vectors, 40)
            np.testing.assert_array_equal(c1, c2, err_msg=name)
            np.testing.assert_array_equal(r1, r2, err_msg=name)

    def test_inputs_are_not_mutated_in_place(self):
        """A pure function must not mutate its arguments -- callers (e.g.
        BallCoverFilter.insert) keep using `centers`/`radii` after the call."""
        centers, radii = _initial_cover(seed=3)
        centers_copy, radii_copy = centers.copy(), radii.copy()
        new_vectors = _rand(8, seed=4)
        for name, fn in INSERTION_STRATEGIES.items():
            fn(centers, radii, new_vectors, 40)
            np.testing.assert_array_equal(centers, centers_copy, err_msg=name)
            np.testing.assert_array_equal(radii, radii_copy, err_msg=name)


# ---------------------------------------------------------------------------
# Shared correctness contract: the covering invariant.
# ---------------------------------------------------------------------------

class TestCoveringInvariant(unittest.TestCase):

    def test_covering_invariant_holds_after_insertion(self):
        """Every strategy must leave both the original points and the newly
        inserted ones covered -- BallCoverFilter's FNR=0 guarantee is built
        on this holding after every insert(), not just after construct()."""
        centers, radii = _initial_cover(n=25, seed=5)
        new_vectors = _rand(15, seed=6)
        for name, fn in INSERTION_STRATEGIES.items():
            new_centers, new_radii = fn(centers, radii, new_vectors, 40)
            all_points = np.vstack([centers, new_vectors])
            self.assertTrue(_covering_ok(all_points, new_centers, new_radii), name)

    def test_covering_invariant_holds_when_merge_strategy_hits_capacity(self):
        """insert_merge is the one strategy that can be forced to evict
        (merge) balls -- covering must still hold once capacity forces
        merges, not just in the no-eviction case above."""
        centers, radii = _initial_cover(n=10, seed=10)
        new_vectors = _rand(30, seed=11)  # far more than the budget below
        new_centers, new_radii = insert_merge(centers, radii, new_vectors, max_count_balls=12)
        self.assertLessEqual(len(new_centers), 12)
        all_points = np.vstack([centers, new_vectors])
        self.assertTrue(_covering_ok(all_points, new_centers, new_radii))

    def test_absorbing_an_already_covered_point_does_not_shrink_or_grow_radii(self):
        """A point strictly inside an existing ball needs no change at all
        for insert_nearest/insert_least_inflation."""
        centers = _rand(5, seed=12)
        radii = np.full(5, 10.0, dtype=np.float32)  # huge -- covers everything
        probe = centers[[0]] + 0.01  # just inside ball 0
        for fn in (insert_nearest, insert_least_inflation):
            new_centers, new_radii = fn(centers, radii, probe, 40)
            np.testing.assert_array_equal(new_radii, radii, err_msg=fn.__name__)


# ---------------------------------------------------------------------------
# insert_nearest / insert_least_inflation -- never relocate a ball
# ---------------------------------------------------------------------------

class TestNoRelocationStrategies(unittest.TestCase):

    def test_returns_the_identical_centers_object(self):
        """Documented contract (module docstring): these two strategies
        never add or move a ball, so callers can skip re-quantizing/
        re-indexing centers by checking `new_centers is centers`."""
        centers, radii = _initial_cover(seed=13)
        new_vectors = _rand(6, seed=14)
        for fn in (insert_nearest, insert_least_inflation):
            new_centers, _ = fn(centers, radii, new_vectors, 40)
            self.assertIs(new_centers, centers, fn.__name__)

    def test_ball_count_never_changes(self):
        centers, radii = _initial_cover(n=20, seed=15)
        new_vectors = _rand(50, seed=16)  # far more points than balls
        for fn in (insert_nearest, insert_least_inflation):
            new_centers, new_radii = fn(centers, radii, new_vectors, 40)
            self.assertEqual(len(new_centers), 20, fn.__name__)
            self.assertEqual(len(new_radii), 20, fn.__name__)

    def test_radii_only_grow(self):
        centers, radii = _initial_cover(n=20, seed=17)
        new_vectors = _rand(40, seed=18)
        for fn in (insert_nearest, insert_least_inflation):
            _, new_radii = fn(centers, radii, new_vectors, 40)
            self.assertTrue(np.all(new_radii >= radii - 1e-6), fn.__name__)

    def test_least_inflation_never_grows_more_than_nearest_would(self):
        """insert_least_inflation optimizes the actual r**d objective
        instead of raw distance -- on the same input, growing the
        *least-inflating* ball can never cost more total volume than
        insert_nearest's raw-distance choice."""
        centers, radii = _initial_cover(n=15, seed=19)
        radii = np.random.default_rng(20).uniform(0.1, 2.0, size=15).astype(np.float32)
        new_vectors = _rand(25, seed=21)

        _, r_nearest = insert_nearest(centers, radii, new_vectors, 40)
        _, r_least = insert_least_inflation(centers, radii, new_vectors, 40)

        vol_nearest = float((r_nearest.astype(np.float64) ** DIM).sum())
        vol_least = float((r_least.astype(np.float64) ** DIM).sum())
        self.assertLessEqual(vol_least, vol_nearest * (1 + 1e-9))


# ---------------------------------------------------------------------------
# insert_merge -- the one strategy that adds/evicts balls
# ---------------------------------------------------------------------------

class TestMergeStrategy(unittest.TestCase):

    def test_new_balls_are_created_while_under_capacity(self):
        centers, radii = _initial_cover(n=5, seed=22)
        new_vectors = _rand(3, seed=23)
        new_centers, new_radii = insert_merge(centers, radii, new_vectors, max_count_balls=20)
        self.assertEqual(len(new_centers), 8)  # 5 existing + 3 new, no eviction needed
        self.assertEqual(len(new_radii), 8)

    def test_never_exceeds_max_count_balls(self):
        centers, radii = _initial_cover(n=5, seed=24)
        new_vectors = _rand(50, seed=25)
        new_centers, new_radii = insert_merge(centers, radii, new_vectors, max_count_balls=10)
        self.assertLessEqual(len(new_centers), 10)
        self.assertEqual(len(new_centers), len(new_radii))

    def test_output_dtype_is_float32(self):
        centers, radii = _initial_cover(seed=26)
        new_vectors = _rand(5, seed=27)
        new_centers, new_radii = insert_merge(centers, radii, new_vectors, 40)
        self.assertEqual(new_centers.dtype, np.float32)
        self.assertEqual(new_radii.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
