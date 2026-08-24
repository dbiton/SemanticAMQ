import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'src'))
import unittest

import faiss
import numpy as np

from src.filters.ball_cover.ball_cover_construction import (
    gonzales_construct_enclosing_balls,
    rptree_construct_enclosing_balls,
    ivf_construct_enclosing_balls,
)

DIM = 8

_CONSTRUCTORS = (
    gonzales_construct_enclosing_balls,
    rptree_construct_enclosing_balls,
    ivf_construct_enclosing_balls,
)


def _rand(n: int, dim: int = DIM, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, dim)).astype(np.float32)


def _clustered(n_clusters: int, per_cluster: int, dim: int = DIM,
               spread: float = 0.05, sep: float = 15.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, dim)) * sep
    parts = [rng.standard_normal((per_cluster, dim)) * spread + c for c in centers]
    return np.vstack(parts).astype(np.float32)


def _covering_ok(vecs: np.ndarray, centers: np.ndarray, radii: np.ndarray, tol: float = 1e-3) -> bool:
    """Every vector must be within (radius + tol) of at least one ball --
    the zero-false-negative invariant every construction strategy must hold.
    Checked the way BallCoverFilter.distance_to_balls_surfaces actually
    queries: signed distance to *every* ball, take the minimum, not just
    the geometrically nearest-by-raw-distance ball (a ball can be farther
    away but still cover the point if its own radius is large enough)."""
    if len(centers) == 0:
        return len(vecs) == 0
    # A faiss exhaustive search rather than a raw (n, k, d) broadcast: the
    # latter materializes a full dense tensor (e.g. 20000 x 5000 x 384
    # floats is ~150GB) at the scale some of the tests below use.
    index = faiss.IndexFlatL2(centers.shape[1])
    index.add(np.ascontiguousarray(centers, dtype=np.float32))
    d2, labels = index.search(np.ascontiguousarray(vecs, dtype=np.float32), len(centers))
    dist = np.sqrt(np.maximum(d2, 0.0))
    return bool(np.all((dist - radii[labels]).min(axis=1) <= tol))


# ---------------------------------------------------------------------------
# Shared contract across all three constructors. Ball count is checked as
# "at most k", not "exactly k": unlike Gonzalez (always returns exactly k),
# rptree rounds down to a power of 2 and IVF can drop empty inverted lists,
# so neither guarantees hitting the requested budget exactly -- see their
# own test classes below for the specifics of by how much and when.
# ---------------------------------------------------------------------------

class TestSharedContract(unittest.TestCase):

    def test_empty_input_returns_empty(self):
        for construct in _CONSTRUCTORS:
            centers, radii = construct(10, np.empty((0, DIM), dtype=np.float32))
            self.assertEqual(len(centers), 0, construct.__name__)
            self.assertEqual(len(radii), 0, construct.__name__)
            self.assertEqual(centers.shape[1], DIM, construct.__name__)

    def test_non_positive_budget_returns_empty(self):
        vecs = _rand(20)
        for construct in _CONSTRUCTORS:
            for budget in (0, -5):
                centers, radii = construct(budget, vecs)
                self.assertEqual(len(centers), 0, f"{construct.__name__} budget={budget}")
                self.assertEqual(len(radii), 0, f"{construct.__name__} budget={budget}")

    def test_fewer_vectors_than_budget_returns_one_ball_per_vector(self):
        vecs = _rand(15)
        for construct in _CONSTRUCTORS:
            centers, radii = construct(50, vecs)
            self.assertEqual(len(centers), 15, construct.__name__)
            np.testing.assert_array_equal(radii, np.zeros(15, dtype=np.float32))
            self.assertTrue(_covering_ok(vecs, centers, radii, tol=1e-6), construct.__name__)

    def test_n_equals_budget_returns_all_as_zero_radius_balls(self):
        vecs = _rand(12)
        for construct in _CONSTRUCTORS:
            centers, radii = construct(12, vecs)
            self.assertEqual(len(centers), 12, construct.__name__)
            np.testing.assert_array_equal(radii, np.zeros(12, dtype=np.float32))

    def test_never_exceeds_requested_budget(self):
        vecs = _rand(200, seed=3)
        for construct in _CONSTRUCTORS:
            centers, radii = construct(20, vecs)
            self.assertLessEqual(len(centers), 20, construct.__name__)
            self.assertEqual(len(centers), len(radii), construct.__name__)
            self.assertEqual(centers.shape[1], DIM, construct.__name__)

    def test_covering_invariant_nontrivial_case(self):
        """The property BallCoverFilter's zero-false-negative guarantee is
        built on: no matter how centers are chosen, every input vector must
        end up within its assigned ball."""
        vecs = _rand(300, seed=4)
        for construct in _CONSTRUCTORS:
            centers, radii = construct(25, vecs)
            self.assertTrue(_covering_ok(vecs, centers, radii), construct.__name__)

    def test_covering_invariant_on_clustered_data(self):
        vecs = _clustered(n_clusters=5, per_cluster=40, seed=7)
        for construct in _CONSTRUCTORS:
            centers, radii = construct(15, vecs)
            self.assertTrue(_covering_ok(vecs, centers, radii), construct.__name__)

    def test_radii_are_non_negative(self):
        vecs = _rand(150, seed=9)
        for construct in _CONSTRUCTORS:
            _, radii = construct(10, vecs)
            self.assertTrue(np.all(radii >= 0.0), construct.__name__)

    def test_duplicate_points_do_not_crash(self):
        vecs = np.repeat(_rand(1, seed=11), 50, axis=0)
        for construct in _CONSTRUCTORS:
            centers, radii = construct(5, vecs)
            self.assertTrue(_covering_ok(vecs, centers, radii), construct.__name__)

    def test_output_dtype_is_float32(self):
        vecs = _rand(100, seed=13)
        for construct in _CONSTRUCTORS:
            centers, radii = construct(10, vecs)
            self.assertEqual(centers.dtype, np.float32, construct.__name__)
            self.assertEqual(radii.dtype, np.float32, construct.__name__)

    def test_accepts_non_contiguous_and_float64_input(self):
        """Callers (BallCoverFilter) don't guarantee contiguous float32
        arrays; every constructor must coerce rather than assume."""
        vecs = _rand(80, seed=19).astype(np.float64)[::2]  # non-contiguous, wrong dtype
        self.assertFalse(vecs.flags["C_CONTIGUOUS"])
        for construct in _CONSTRUCTORS:
            centers, radii = construct(8, vecs)
            self.assertTrue(_covering_ok(vecs.astype(np.float32), centers, radii), construct.__name__)

    def test_deterministic_given_same_input(self):
        """Gonzalez and rptree are both pure, seeded numpy pipelines with no
        threaded reduction order to introduce nondeterminism. IVF is
        excluded here -- see TestIVFConstruct.test_ball_count_and_output_
        can_vary_run_to_run_at_scale, which documents that it genuinely
        isn't reproducible at larger k."""
        vecs = _rand(120, seed=17)
        for construct in (gonzales_construct_enclosing_balls, rptree_construct_enclosing_balls):
            c1, r1 = construct(12, vecs)
            c2, r2 = construct(12, vecs)
            np.testing.assert_array_equal(c1, c2, err_msg=construct.__name__)
            np.testing.assert_array_equal(r1, r2, err_msg=construct.__name__)


# ---------------------------------------------------------------------------
# gonzales_construct_enclosing_balls -- farthest-first specific properties
# ---------------------------------------------------------------------------

class TestGonzalesConstruct(unittest.TestCase):

    def test_returns_exactly_k_balls(self):
        """Unlike rptree/IVF, Gonzalez always fills the requested budget
        exactly in the non-trivial case."""
        vecs = _rand(200, seed=3)
        centers, radii = gonzales_construct_enclosing_balls(20, vecs)
        self.assertEqual(len(centers), 20)
        self.assertEqual(len(radii), 20)

    def test_centers_are_original_data_points(self):
        """Defining property of farthest-first traversal: every chosen
        center is literally one of the input vectors, unlike a
        centroid-based method (rptree, IVF)."""
        vecs = _rand(100, seed=2)
        centers, _ = gonzales_construct_enclosing_balls(10, vecs)
        for c in centers:
            self.assertTrue(np.any(np.all(vecs == c, axis=1)), "center not found among input vectors")

    def test_centers_are_far_apart_relative_to_random_pairs(self):
        """Farthest-first is a 2-approximation to the max-radius k-center
        objective -- chosen centers should be markedly more spread out than
        a same-size random sample."""
        vecs = _clustered(n_clusters=4, per_cluster=50, sep=20.0, seed=5)
        centers, _ = gonzales_construct_enclosing_balls(8, vecs)

        def min_pairwise_dist(pts: np.ndarray) -> float:
            d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
            np.fill_diagonal(d, np.inf)
            return float(d.min())

        rng = np.random.default_rng(0)
        random_sample = vecs[rng.choice(len(vecs), size=8, replace=False)]
        self.assertGreater(min_pairwise_dist(centers), min_pairwise_dist(random_sample))

    def test_matches_reference_brute_force_implementation(self):
        """The dot-product-expansion optimization must select exactly the
        same centers (up to floating-point rounding in the max-distance
        tie-break) as the straightforward ||x - c||^2 computation it
        replaced."""
        vecs = _rand(150, seed=23)
        k = 12

        def reference(k: int, X: np.ndarray) -> list:
            chosen = [0]
            min_d2 = ((X - X[0]) ** 2).sum(1)
            for _ in range(k - 1):
                chosen.append(int(min_d2.argmax()))
                min_d2 = np.minimum(min_d2, ((X - X[chosen[-1]]) ** 2).sum(1))
            return chosen

        expected_idx = reference(k, vecs)
        centers, _ = gonzales_construct_enclosing_balls(k, vecs)
        np.testing.assert_allclose(centers, vecs[expected_idx], atol=1e-3)


# ---------------------------------------------------------------------------
# rptree_construct_enclosing_balls -- random-projection-tree specific
# properties
# ---------------------------------------------------------------------------

class TestRPTreeConstruct(unittest.TestCase):

    def test_leaf_count_is_the_largest_power_of_two_at_most_k(self):
        """depth = floor(log2(k)) is baked into the algorithm: the tree
        always has exactly 2**depth leaves, which is k itself only when k is
        already a power of 2 -- otherwise it's a real, deterministic
        undershoot (e.g. k=99 -> 64 balls, not 99)."""
        vecs = _rand(300, seed=1)
        for k in (1, 2, 5, 20, 99, 200):
            centers, radii = rptree_construct_enclosing_balls(k, vecs, seed=0)
            expected = 2 ** int(np.floor(np.log2(k)))
            self.assertEqual(len(centers), expected, f"k={k}")
            self.assertEqual(len(radii), expected, f"k={k}")

    def test_centers_are_leaf_centroids_not_original_points(self):
        """Unlike Gonzalez, centers are means of each leaf's members, so
        they generally aren't literal input vectors."""
        vecs = _rand(100, seed=2)
        centers, _ = rptree_construct_enclosing_balls(10, vecs, seed=0)
        for c in centers:
            self.assertFalse(np.any(np.all(vecs == c, axis=1)))

    def test_separates_clusters_when_k_matches_leaf_count(self):
        """When k is already a power of 2 matching the true cluster count,
        every leaf should land inside one cluster, not span the gap between
        two -- radii should stay well under half the cluster separation."""
        sep, spread = 25.0, 0.5
        vecs = _clustered(n_clusters=4, per_cluster=60, spread=spread, sep=sep, seed=6)
        _, radii = rptree_construct_enclosing_balls(4, vecs, seed=0)
        self.assertTrue(np.all(radii < sep / 2), f"radii {radii} span across clusters")

    def test_non_power_of_two_k_can_merge_distinct_clusters(self):
        """Documents a real, known trade-off rather than asserting rptree is
        strictly better or worse than Gonzalez in general: with 6 true
        clusters but k=6 rounding down to 4 leaves, at least one leaf must
        absorb two separate clusters, inflating that leaf's radius far past
        the within-cluster scale. If this ever stops reproducing (e.g. the
        depth formula changes to round up, or to pick depth from a size
        target instead of a power of 2), that's a real behavior change this
        test is meant to catch."""
        spread, sep = 0.5, 25.0
        vecs = _clustered(n_clusters=6, per_cluster=40, spread=spread, sep=sep, seed=8)
        _, radii = rptree_construct_enclosing_balls(6, vecs, seed=0)
        self.assertEqual(len(radii), 4)  # 2**floor(log2(6))
        self.assertTrue(np.any(radii > sep / 2), "expected at least one leaf to span two clusters")

    def test_deterministic_given_seed_varies_with_seed(self):
        vecs = _rand(200, seed=14)
        c1, r1 = rptree_construct_enclosing_balls(16, vecs, seed=0)
        c2, r2 = rptree_construct_enclosing_balls(16, vecs, seed=0)
        np.testing.assert_array_equal(c1, c2)
        np.testing.assert_array_equal(r1, r2)
        c3, _ = rptree_construct_enclosing_balls(16, vecs, seed=1)
        self.assertFalse(np.array_equal(c1, c3), "different seeds should (almost always) pick different splits")


# ---------------------------------------------------------------------------
# ivf_construct_enclosing_balls -- IVF-based specific properties
# ---------------------------------------------------------------------------

class TestIVFConstruct(unittest.TestCase):

    def test_k_equals_one_special_case(self):
        vecs = _rand(50, seed=15)
        centers, radii = ivf_construct_enclosing_balls(1, vecs)
        self.assertEqual(len(centers), 1)
        self.assertEqual(len(radii), 1)
        np.testing.assert_allclose(centers[0], vecs.mean(axis=0), atol=1e-4)
        self.assertTrue(_covering_ok(vecs, centers, radii))

    def test_returns_exactly_k_balls_below_hnsw_threshold(self):
        """Below HNSW_QUANTIZER_MIN_K (2048) the coarse quantizer is an
        exact IndexFlatL2 and k-means training reliably fills every list at
        this n/k ratio -- verified empirically to hit the budget exactly
        and reproducibly here, unlike the large-k/HNSW-quantizer regime
        exercised in test_ball_count_can_undershoot_at_scale below."""
        vecs = _rand(300, seed=16)
        centers, radii = ivf_construct_enclosing_balls(50, vecs)
        self.assertEqual(len(centers), 50)
        self.assertEqual(len(radii), 50)
        self.assertTrue(_covering_ok(vecs, centers, radii))

    def test_crosses_hnsw_quantizer_boundary_safely(self):
        """k just at/above HNSW_QUANTIZER_MIN_K switches the coarse
        quantizer from IndexFlatL2 to IndexHNSWFlat -- must still produce a
        valid, fully-covering set of balls, not just avoid crashing."""
        vecs = _rand(3000, seed=18)
        for k in (2047, 2048, 2100):
            centers, radii = ivf_construct_enclosing_balls(k, vecs)
            self.assertLessEqual(len(centers), k, f"k={k}")
            self.assertGreater(len(centers), 0, f"k={k}")
            self.assertTrue(_covering_ok(vecs, centers, radii), f"k={k}")

    def test_ball_count_can_undershoot_at_scale(self):
        """At the project's real scale (n=20000, dim=384, k=5000 -- well
        past the HNSW threshold), a handful of inverted lists routinely end
        up empty after only 8 k-means iterations and get dropped, so the
        returned ball count is noticeably below the k requested, and isn't
        even the same from run to run (multi-threaded k-means/HNSW
        construction isn't bit-reproducible here despite the fixed seed).
        This is the caveat from the earlier IVF-vs-Gonzalez comparison,
        pinned down as a test: BallCoverFilter's memory budget is still
        respected (fewer balls than budgeted, never more) and the covering
        invariant still holds -- just not an exact or reproducible ball
        count at this scale."""
        dim = 384
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20_000, dim)).astype(np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True)

        centers, radii = ivf_construct_enclosing_balls(5000, X)
        self.assertLess(len(centers), 5000, "expected a real undershoot at this scale")
        self.assertGreater(len(centers), 0)
        self.assertTrue(_covering_ok(X, centers, radii))


if __name__ == "__main__":
    unittest.main()
