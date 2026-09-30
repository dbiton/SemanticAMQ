import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'src'))
import inspect
import unittest

import numpy as np

from src.filters.ball_cover.ball_cover_filter import BallCoverFilter, Quantize
from src.filters.ball_cover.ball_interfaces import InsertInterface
from src.filters.ball_cover.min_heap_insert import MinHeapInsert
from src.filters.ball_cover.nearest_insert import NearestInsert
from src.filters.ball_cover.least_inflation_insert import LeastInflationInsert
from src.filters.ball_cover.exhaustive_merge_insert import ExhaustiveMergeInsert
from src.filters.ball_cover.doubling_insert import DoublingInsert
from src.filters.ball_cover.pooled_insert import PooledInsert
from src.filters.ball_cover.cap_merge_insert import CapMergeInsert

DIM = 8

def get_insertion_strategies():
    return {
        'min_heap': MinHeapInsert(),
        'nearest': NearestInsert(),
        'least_inflation': LeastInflationInsert(),
        'exhaustive_merge': ExhaustiveMergeInsert(),
        'doubling': DoublingInsert(),
        'pooled': PooledInsert(),
        'cap_merge': CapMergeInsert(),
    }


def _seed(f: BallCoverFilter, centers: np.ndarray, radii: np.ndarray) -> None:
    """Puts an explicit ball set into `f` the way construct() would -- train the
    quantizer on the centers, then store them via _add_spheres().
    Mirrors BallCoverFilter.construct() minus the construct_method, so these tests 
    exercise the real invariants."""
    centers = np.ascontiguousarray(centers, dtype=np.float32)
    radii = np.ascontiguousarray(radii, dtype=np.float32)
    f._train_quantizer(centers)
    f._add_spheres(centers, radii)
    f._constructed = True


def run_insert(insert_method: InsertInterface, centers: np.ndarray, radii: np.ndarray, new_vectors: np.ndarray, max_count_balls: int):
    """Helper to bridge the old functional tests to the new BallCoverFilter OOP design."""
    safe_budget = max(0, max_count_balls)
    f = BallCoverFilter(dim=DIM, insert_method=insert_method, max_spheres=safe_budget, quantize=Quantize.NONE)
    _seed(f, centers, radii)
    f.insert(new_vectors)
    return f.all_centers, f.all_radiuses


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
        signatures = {name: inspect.signature(strategy.insert) for name, strategy in get_insertion_strategies().items()}
        first_name, first_sig = next(iter(signatures.items()))
        for name, sig in signatures.items():
            self.assertEqual(
                list(sig.parameters), list(first_sig.parameters),
                f"{name} parameter names differ from {first_name}",
            )

    def test_all_strategies_are_interchangeable_through_the_dispatch_table(self):
        """Every strategy must accept being injected into the exact same
        BallCoverFilter setup without needing special parameters."""
        centers, radii = _initial_cover()
        new_vectors = _rand(5, seed=99)
        for name, strategy in get_insertion_strategies().items():
            c, r = run_insert(strategy, centers, radii, new_vectors, 40)
            self.assertIsInstance(c, np.ndarray, name)
            self.assertIsInstance(r, np.ndarray, name)

    def test_stateless_same_inputs_give_same_outputs(self):
        """No hidden state: calling a strategy twice with identical inputs,
        with an unrelated call to a *different* strategy (and different
        data) in between, must not change the result."""
        centers, radii = _initial_cover(seed=1)
        new_vectors = _rand(10, seed=2)

        for name, strategy in get_insertion_strategies().items():
            c1, r1 = run_insert(strategy, centers, radii, new_vectors, 40)

            # Perturb global state a different strategy might (incorrectly)
            # depend on: call every other strategy with unrelated data.
            for other_name, other_strategy in get_insertion_strategies().items():
                if other_name != name:
                    run_insert(other_strategy, *_initial_cover(seed=7), _rand(20, seed=8), 25)

            c2, r2 = run_insert(strategy, centers, radii, new_vectors, 40)
            np.testing.assert_array_equal(c1, c2, err_msg=name)
            np.testing.assert_array_equal(r1, r2, err_msg=name)

    def test_inputs_are_not_mutated_in_place(self):
        """A pure function must not mutate its arguments -- callers (e.g.
        BallCoverFilter.insert) keep using `vectors` after the call."""
        centers, radii = _initial_cover(seed=3)
        new_vectors = _rand(8, seed=4)
        vectors_copy = new_vectors.copy()
        
        for name, strategy in get_insertion_strategies().items():
            run_insert(strategy, centers, radii, new_vectors, 40)
            np.testing.assert_array_equal(new_vectors, vectors_copy, err_msg=name)


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
        for name, strategy in get_insertion_strategies().items():
            new_centers, new_radii = run_insert(strategy, centers, radii, new_vectors, 40)
            all_points = np.vstack([centers, new_vectors])
            self.assertTrue(_covering_ok(all_points, new_centers, new_radii), name)

    def test_covering_invariant_holds_when_merge_strategy_hits_capacity(self):
        """ExhaustiveMergeInsert is the one strategy that can be forced to evict
        (merge) balls -- covering must still hold once capacity forces
        merges, not just in the no-eviction case above."""
        centers, radii = _initial_cover(n=10, seed=10)
        new_vectors = _rand(30, seed=11)  # far more than the budget below
        new_centers, new_radii = run_insert(ExhaustiveMergeInsert(), centers, radii, new_vectors, 12)
        self.assertLessEqual(len(new_centers), 12)
        all_points = np.vstack([centers, new_vectors])
        self.assertTrue(_covering_ok(all_points, new_centers, new_radii))

    def test_absorbing_an_already_covered_point_does_not_shrink_or_grow_radii(self):
        """A point strictly inside an existing ball needs no change at all
        for NearestInsert / LeastInflationInsert."""
        centers = _rand(5, seed=12)
        radii = np.full(5, 10.0, dtype=np.float32)  # huge -- covers everything
        probe = centers[[0]] + 0.01  # just inside ball 0
        for name, strategy in [('nearest', NearestInsert()), ('least_inflation', LeastInflationInsert())]:
            new_centers, new_radii = run_insert(strategy, centers, radii, probe, 40)
            np.testing.assert_array_equal(new_radii, radii, err_msg=name)


# ---------------------------------------------------------------------------
# NearestInsert / LeastInflationInsert -- never relocate a ball
# ---------------------------------------------------------------------------

class TestNoRelocationStrategies(unittest.TestCase):

    def test_returns_the_identical_centers_object(self):
        """Documented contract: these two strategies never add or move a ball, 
        so BallCoverFilter skips re-quantizing/re-indexing centers by checking 
        array equality. Tested here to ensure the underlying values don't move."""
        centers, radii = _initial_cover(seed=13)
        new_vectors = _rand(6, seed=14)
        for name, strategy in [('nearest', NearestInsert()), ('least_inflation', LeastInflationInsert())]:
            f = BallCoverFilter(dim=DIM, insert_method=strategy, max_spheres=40, quantize=Quantize.NONE)
            _seed(f, centers, radii)
            
            old_centers = f.all_centers
            f.insert(new_vectors)
            new_centers = f.all_centers
            
            np.testing.assert_array_equal(new_centers, old_centers, err_msg=name)

    def test_ball_count_never_changes(self):
        centers, radii = _initial_cover(n=20, seed=15)
        new_vectors = _rand(50, seed=16)  # far more points than balls
        for name, strategy in [('nearest', NearestInsert()), ('least_inflation', LeastInflationInsert())]:
            new_centers, new_radii = run_insert(strategy, centers, radii, new_vectors, 40)
            self.assertEqual(len(new_centers), 20, name)
            self.assertEqual(len(new_radii), 20, name)

    def test_radii_only_grow(self):
        centers, radii = _initial_cover(n=20, seed=17)
        new_vectors = _rand(40, seed=18)
        for name, strategy in [('nearest', NearestInsert()), ('least_inflation', LeastInflationInsert())]:
            _, new_radii = run_insert(strategy, centers, radii, new_vectors, 40)
            self.assertTrue(np.all(new_radii >= radii - 1e-6), name)

    def test_least_inflation_never_grows_more_than_nearest_would(self):
        """LeastInflationInsert optimizes the actual r**d objective
        instead of raw distance -- on the same input, growing the
        *least-inflating* ball can never cost more total volume than
        NearestInsert's raw-distance choice."""
        centers, radii = _initial_cover(n=15, seed=19)
        radii = np.random.default_rng(20).uniform(0.1, 2.0, size=15).astype(np.float32)
        new_vectors = _rand(25, seed=21)

        _, r_nearest = run_insert(NearestInsert(), centers, radii, new_vectors, 40)
        _, r_least = run_insert(LeastInflationInsert(), centers, radii, new_vectors, 40)

        vol_nearest = float((r_nearest.astype(np.float64) ** DIM).sum())
        vol_least = float((r_least.astype(np.float64) ** DIM).sum())
        self.assertLessEqual(vol_least, vol_nearest * (1 + 1e-9))


# ---------------------------------------------------------------------------
# ExhaustiveMergeInsert -- the one strategy that adds/evicts balls
# ---------------------------------------------------------------------------

class TestMergeStrategy(unittest.TestCase):

    def test_new_balls_are_created_while_under_capacity(self):
        centers, radii = _initial_cover(n=5, seed=22)
        new_vectors = _rand(3, seed=23)
        new_centers, new_radii = run_insert(ExhaustiveMergeInsert(), centers, radii, new_vectors, 20)
        self.assertEqual(len(new_centers), 8)  # 5 existing + 3 new, no eviction needed
        self.assertEqual(len(new_radii), 8)

    def test_never_exceeds_max_count_balls(self):
        centers, radii = _initial_cover(n=5, seed=24)
        new_vectors = _rand(50, seed=25)
        new_centers, new_radii = run_insert(ExhaustiveMergeInsert(), centers, radii, new_vectors, 10)
        self.assertLessEqual(len(new_centers), 10)
        self.assertEqual(len(new_centers), len(new_radii))

    def test_output_dtype_is_float32(self):
        centers, radii = _initial_cover(seed=26)
        new_vectors = _rand(5, seed=27)
        new_centers, new_radii = run_insert(ExhaustiveMergeInsert(), centers, radii, new_vectors, 40)
        self.assertEqual(new_centers.dtype, np.float32)
        self.assertEqual(new_radii.dtype, np.float32)


# ---------------------------------------------------------------------------
# DoublingInsert -- amortized rebuild: grows past max_spheres between
# periodic Gonzalez re-clustering passes, instead of a hard per-insert cap.
# ---------------------------------------------------------------------------

class TestDoublingStrategy(unittest.TestCase):

    def test_rejects_non_positive_growth_budget_fraction(self):
        for bad in (0.0, -0.5):
            with self.assertRaises(ValueError):
                DoublingInsert(growth_budget_fraction=bad)

    def test_new_balls_are_added_as_singletons_under_the_rebuild_threshold(self):
        """Below the growth threshold, every not-already-covered vector
        becomes its own zero-radius ball -- no merge/relocation, unlike
        ExhaustiveMergeInsert/MinHeapInsert."""
        centers, radii = _initial_cover(n=5, seed=30)
        new_vectors = _rand(3, seed=31)
        new_centers, new_radii = run_insert(DoublingInsert(), centers, radii, new_vectors, 40)
        self.assertEqual(len(new_centers), 8)  # 5 existing + 3 new singletons
        np.testing.assert_array_equal(new_radii[5:], np.zeros(3, dtype=np.float32))

    def test_rebuild_collapses_back_to_at_most_max_spheres(self):
        """Once the active count reaches max_spheres * (1 + growth_budget_fraction),
        a rebuild must fire and bring the count back down to <= max_spheres."""
        max_spheres = 10
        strategy = DoublingInsert(growth_budget_fraction=1.0)
        f = BallCoverFilter(dim=DIM, insert_method=strategy, max_spheres=max_spheres, quantize=Quantize.NONE)
        f.construct(_rand(max_spheres, seed=32))
        self.assertEqual(len(f.all_spheres), max_spheres)

        # Insert one not-already-covered vector at a time -- each becomes a
        # singleton, so this exercises the exact growth -> rebuild -> shrink
        # cycle instead of jumping straight to the threshold in one batch.
        rng = np.random.default_rng(33)
        max_seen = len(f.all_spheres)
        for _ in range(max_spheres + 2):
            f.insert(rng.standard_normal((1, DIM)).astype(np.float32))
            max_seen = max(max_seen, len(f.all_spheres))

        self.assertLessEqual(len(f.all_spheres), max_spheres)
        self.assertGreater(max_seen, max_spheres)  # it did actually grow past budget first
        self.assertLessEqual(max_seen, max_spheres * 2)  # never exceeds the configured cap

    def test_covering_invariant_holds_across_a_long_insertion_stream(self):
        """Regression test for the bug where a not-yet-rebuilt vector was
        invisible to query(): every point ever inserted -- old and new,
        pre- and post-rebuild -- must stay coverable at *every* checkpoint,
        not just immediately after a rebuild."""
        dim = 12
        max_spheres = 12
        f = BallCoverFilter(
            dim=dim, insert_method=DoublingInsert(growth_budget_fraction=1.0),
            max_spheres=max_spheres, quantize=Quantize.NONE,
        )
        rng = np.random.default_rng(34)
        bootstrap = rng.standard_normal((max_spheres, dim)).astype(np.float32)
        f.construct(bootstrap)

        all_vectors = [bootstrap[i] for i in range(max_spheres)]
        for _ in range(400):
            vec = rng.standard_normal((1, dim)).astype(np.float32)
            f.insert(vec)
            all_vectors.append(vec[0])

            all_arr = np.ascontiguousarray(np.array(all_vectors), dtype=np.float32)
            found = f.query(all_arr, 1e-4)
            missing = np.where(~np.array(found))[0]
            self.assertEqual(len(missing), 0, f"false negative(s) at indices {missing.tolist()}")

        self.assertLessEqual(len(f.all_spheres), max_spheres)


# ---------------------------------------------------------------------------
# Every strategy, through construct() and a stream, on unit-norm data -- the
# regime the streaming evaluation runs in (and the one CapMergeInsert's cap
# geometry depends on).
# ---------------------------------------------------------------------------

def _unit_clusters(n: int, dim: int = 16, n_clusters: int = 12, spread: float = 0.35, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    anchors = rng.standard_normal((n_clusters, dim))
    anchors /= np.linalg.norm(anchors, axis=1, keepdims=True)
    x = anchors[rng.integers(0, n_clusters, n)] + spread / np.sqrt(dim) * rng.standard_normal((n, dim))
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def _streaming_filter(name: str, dim: int = 16, max_spheres: int = None, max_memory: int = None) -> BallCoverFilter:
    return BallCoverFilter(dim=dim, insert_method=get_insertion_strategies()[name], max_spheres=max_spheres,
                           max_memory=max_memory, insert_memory_percent=20.0 if name == 'pooled' else 0.0,
                           quantize=Quantize.NONE)


def _recalled(f: BallCoverFilter, vecs: np.ndarray) -> np.ndarray:
    return np.asarray(f.query(np.ascontiguousarray(vecs, dtype=np.float32), 0.0), dtype=bool)


class TestEveryStrategyOnAStream(unittest.TestCase):

    def test_long_stream_keeps_every_vector_covered(self):
        X = _unit_clusters(900, seed=40)
        for name in get_insertion_strategies():
            with self.subTest(strategy=name):
                f = _streaming_filter(name, max_spheres=24)
                f.construct(X[:300])
                for i in range(300, len(X)):
                    f.insert(X[i:i + 1])
                    if i % 150 == 0 or i == len(X) - 1:
                        missing = np.flatnonzero(~_recalled(f, X[:i + 1]))
                        self.assertEqual(len(missing), 0, f"{name}: {len(missing)} vectors lost at {i}")

    def test_peak_bytes_never_exceed_max_memory(self):
        """The byte rule the streaming evaluation ranks strategies under: a
        strategy's balls *and* its own bookkeeping fit max_memory after every
        insert(), not just after construct() (see
        InsertInterface.max_spheres_for_budget)."""
        X = _unit_clusters(700, seed=41)
        budget = 40 * BallCoverFilter(dim=16, max_memory=1).bytes_per_ball()
        for name in get_insertion_strategies():
            with self.subTest(strategy=name):
                f = _streaming_filter(name, max_memory=budget)
                f.construct(X[:200])
                peak = f.get_bytesize()
                for i in range(200, len(X), 3):
                    f.insert(X[i:i + 3])
                    peak = max(peak, f.get_bytesize())
                self.assertLessEqual(peak, budget, name)
                self.assertTrue(_recalled(f, X).all(), name)

    def test_capacity_holds_between_inserts(self):
        X = _unit_clusters(600, seed=42)
        for name in get_insertion_strategies():
            with self.subTest(strategy=name):
                f = _streaming_filter(name, max_spheres=20)
                f.construct(X[:100])
                for i in range(100, len(X), 7):
                    f.insert(X[i:i + 7])
                    # DoublingInsert overshoots max_spheres by design, up to
                    # its rebuild threshold; everything else stays within it.
                    cap = DoublingInsert()._rebuild_threshold(f) - 1 if name == 'doubling' else f.max_spheres
                    self.assertLessEqual(len(f.all_spheres), cap, name)

    def test_empty_batch_changes_nothing(self):
        X = _unit_clusters(300, seed=43)
        for name in get_insertion_strategies():
            with self.subTest(strategy=name):
                f = _streaming_filter(name, max_spheres=20)
                f.construct(X[:150])
                f.insert(X[150:200])
                before = (f.all_centers.copy(), f.all_radiuses.copy())
                f.insert(np.empty((0, 16), dtype=np.float32))
                np.testing.assert_array_equal(f.all_centers, before[0], err_msg=name)
                np.testing.assert_array_equal(f.all_radiuses, before[1], err_msg=name)

    def test_exact_duplicates(self):
        """One vector inserted many times -- a batch of copies, then again
        between other vectors -- stays covered. Strategies that decide one
        vector at a time spend at most one ball on the batch; DoublingInsert
        and PooledInsert test coverage once per batch, so every copy in it
        becomes a singleton until the next rebuild (wasteful, not unsound)."""
        X = _unit_clusters(400, seed=44)
        dup = X[350]
        for name in get_insertion_strategies():
            with self.subTest(strategy=name):
                f = _streaming_filter(name, max_spheres=24)
                f.construct(X[:200])
                n_before = len(f.all_spheres)
                f.insert(np.repeat(dup[None, :], 25, axis=0))
                cap = DoublingInsert()._rebuild_threshold(f) - 1 if name == 'doubling' else f.max_spheres
                self.assertLessEqual(len(f.all_spheres), cap, name)
                if name not in ('doubling', 'pooled'):
                    self.assertLessEqual(len(f.all_spheres), min(n_before + 1, cap), name)
                for i in range(200, 260):
                    f.insert(X[i:i + 1])
                    f.insert(dup[None, :])
                self.assertTrue(_recalled(f, np.vstack([X[:260], dup[None, :]])).all(), name)

    def test_batch_larger_than_capacity(self):
        X = _unit_clusters(500, seed=45)
        for name in get_insertion_strategies():
            with self.subTest(strategy=name):
                f = _streaming_filter(name, max_spheres=16)
                f.construct(X[:40])
                f.insert(X[40:])  # 460 vectors into a 16-ball filter, one call
                self.assertTrue(_recalled(f, X).all(), name)
                cap = DoublingInsert()._rebuild_threshold(f) - 1 if name == 'doubling' else f.max_spheres
                self.assertLessEqual(len(f.all_spheres), cap, name)

    def test_float64_and_non_contiguous_input(self):
        X = _unit_clusters(600, seed=46)
        X64 = np.asfortranarray(X.astype(np.float64))[::2]  # float64, strided, non-C-contiguous
        self.assertFalse(X64.flags['C_CONTIGUOUS'])
        for name in get_insertion_strategies():
            with self.subTest(strategy=name):
                f = _streaming_filter(name, max_spheres=20)
                f.construct(X64[:100])
                for i in range(100, len(X64), 25):
                    f.insert(X64[i:i + 25])
                self.assertTrue(_recalled(f, X64).all(), name)


class TestCapMergeInsert(unittest.TestCase):

    def test_off_sphere_vector_falls_back_to_euclidean_balls(self):
        """The cap geometry assumes every member lies on construct()'s sphere;
        one inserted vector off it must switch merging back to enclosing
        balls without losing anything already covered."""
        X = _unit_clusters(500, seed=47)
        f = _streaming_filter('cap_merge', max_spheres=20)
        f.construct(X[:200])
        self.assertGreater(f.insert_method._rho, 0.0)
        f.insert(X[200:300])
        far = 3.0 * X[300:301]
        f.insert(far)
        self.assertEqual(f.insert_method._rho, 0.0)
        f.insert(X[301:])
        self.assertTrue(_recalled(f, np.vstack([X, far])).all())
        self.assertLessEqual(len(f.all_spheres), 20)

    def test_member_norms_within_the_sphere_tolerance_stay_covered(self):
        """Caps are computed for member norms anywhere in [rho - eps, rho + eps];
        norms spread across the tolerance (not just float32 rounding) must
        not open a gap at a cap's rim."""
        X = _unit_clusters(900, seed=50)
        scale = np.random.default_rng(51).uniform(1 - 4e-4, 1 + 4e-4, size=(len(X), 1))
        X = (X * scale).astype(np.float32)
        f = _streaming_filter('cap_merge', max_spheres=16)
        f.construct(X[:200])
        self.assertGreater(f.insert_method._rho, 0.0)
        for i in range(200, len(X), 5):
            f.insert(X[i:i + 5])
        self.assertGreater(f.insert_method._rho, 0.0)   # still in cap mode
        self.assertGreater(f.insert_method._eps, 1e-4)
        # Exact margins, without query()'s coverage epsilon to hide a rim gap.
        C, r = f.all_centers.astype(np.float64), f.all_radiuses.astype(np.float64)
        margin = (np.linalg.norm(X.astype(np.float64)[:, None, :] - C[None], axis=2) - r[None]).min(axis=1)
        self.assertLessEqual(float(margin.max()), 1e-6)

    def test_covered_insert_is_a_geometric_no_op(self):
        X = _unit_clusters(400, seed=48)
        f = _streaming_filter('cap_merge', max_spheres=20)
        f.construct(X[:300])
        f.insert(X[300:350])
        centers, radii = f.all_centers.copy(), f.all_radiuses.copy()
        f.insert(np.repeat(X[:10], 3, axis=0))  # every one already covered
        np.testing.assert_array_equal(f.all_centers, centers)
        np.testing.assert_array_equal(f.all_radiuses, radii)

    def test_merged_centers_sit_outside_the_sphere(self):
        """Merged balls go to norm 1/cos(theta), not inside the sphere where
        an enclosing ball's center sits."""
        X = _unit_clusters(800, seed=49)
        f = _streaming_filter('cap_merge', max_spheres=16)
        f.construct(X[:100])
        f.insert(X[100:])
        norms = np.linalg.norm(f.all_centers, axis=1)
        self.assertTrue(np.all(norms >= 1.0 - 1e-5), norms.min())


if __name__ == "__main__":
    unittest.main()