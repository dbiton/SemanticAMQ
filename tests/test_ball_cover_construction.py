import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'src'))
import unittest

import faiss
import numpy as np

from src.filters.ball_cover.ball_cover_filter import BallCoverFilter
from src.filters.ball_cover.ball_interfaces import ConstructInterface
from src.filters.ball_cover.gonzales_construct import GonzalesConstruct
from src.filters.ball_cover.merge_construct import MergeConstruct
from src.filters.ball_cover.minimax_construct import MinimaxConstruct
from src.filters.ball_cover.split_construct import SplitConstruct
from src.filters.ball_cover.sphere_merge_construct import SphereMergeConstruct
from src.filters.ball_cover.exact_cover import exact_cover, sphere_centers, sphere_radius

DIM = 8

def get_constructors():
    """Every construct strategy; GonzalesConstruct is the live default."""
    return [GonzalesConstruct(), MergeConstruct(), MinimaxConstruct(), MinimaxConstruct(sphere=True),
            SplitConstruct(), SphereMergeConstruct()]


def run_construct(construct_method: ConstructInterface, budget: int, vecs: np.ndarray):
    """Helper to bridge the old functional tests to the new BallCoverFilter OOP design."""
    safe_budget = max(0, budget)
    f = BallCoverFilter(dim=DIM, construct_method=construct_method, max_spheres=safe_budget, insert_memory_percent=0.0)
    return construct_method.construct_filter(f, vecs)


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
    the geometrically nearest-by-raw-distance ball."""
    if len(centers) == 0:
        return len(vecs) == 0
    
    index = faiss.IndexFlatL2(centers.shape[1])
    index.add(np.ascontiguousarray(centers, dtype=np.float32))
    d2, labels = index.search(np.ascontiguousarray(vecs, dtype=np.float32), len(centers))
    dist = np.sqrt(np.maximum(d2, 0.0))
    return bool(np.all((dist - radii[labels]).min(axis=1) <= tol))


# ---------------------------------------------------------------------------
# Shared contract across all three constructors. Ball count is checked as
# "at most k", not "exactly k"
# ---------------------------------------------------------------------------

class TestSharedContract(unittest.TestCase):

    def test_empty_input_returns_empty(self):
        for construct in get_constructors():
            centers, radii = run_construct(construct, 10, np.empty((0, DIM), dtype=np.float32))
            name = construct.__class__.__name__
            self.assertEqual(len(centers), 0, name)
            self.assertEqual(len(radii), 0, name)
            self.assertEqual(centers.shape[1], DIM, name)

    def test_non_positive_budget_returns_empty(self):
        vecs = _rand(20)
        for construct in get_constructors():
            name = construct.__class__.__name__
            for budget in (0, -5):
                centers, radii = run_construct(construct, budget, vecs)
                self.assertEqual(len(centers), 0, f"{name} budget={budget}")
                self.assertEqual(len(radii), 0, f"{name} budget={budget}")

    def test_fewer_vectors_than_budget_returns_one_ball_per_vector(self):
        vecs = _rand(15)
        for construct in get_constructors():
            name = construct.__class__.__name__
            centers, radii = run_construct(construct, 50, vecs)
            self.assertEqual(len(centers), 15, name)
            np.testing.assert_array_equal(radii, np.zeros(15, dtype=np.float32))
            self.assertTrue(_covering_ok(vecs, centers, radii, tol=1e-6), name)

    def test_n_equals_budget_returns_all_as_zero_radius_balls(self):
        vecs = _rand(12)
        for construct in get_constructors():
            name = construct.__class__.__name__
            centers, radii = run_construct(construct, 12, vecs)
            self.assertEqual(len(centers), 12, name)
            np.testing.assert_array_equal(radii, np.zeros(12, dtype=np.float32))

    def test_never_exceeds_requested_budget(self):
        vecs = _rand(200, seed=3)
        for construct in get_constructors():
            name = construct.__class__.__name__
            centers, radii = run_construct(construct, 20, vecs)
            self.assertLessEqual(len(centers), 20, name)
            self.assertEqual(len(centers), len(radii), name)
            self.assertEqual(centers.shape[1], DIM, name)

    def test_covering_invariant_nontrivial_case(self):
        """The property BallCoverFilter's zero-false-negative guarantee is
        built on: no matter how centers are chosen, every input vector must
        end up within its assigned ball."""
        vecs = _rand(300, seed=4)
        for construct in get_constructors():
            centers, radii = run_construct(construct, 25, vecs)
            self.assertTrue(_covering_ok(vecs, centers, radii), construct.__class__.__name__)

    def test_covering_invariant_on_clustered_data(self):
        vecs = _clustered(n_clusters=5, per_cluster=40, seed=7)
        for construct in get_constructors():
            centers, radii = run_construct(construct, 15, vecs)
            self.assertTrue(_covering_ok(vecs, centers, radii), construct.__class__.__name__)

    def test_radii_are_non_negative(self):
        vecs = _rand(150, seed=9)
        for construct in get_constructors():
            _, radii = run_construct(construct, 10, vecs)
            self.assertTrue(np.all(radii >= 0.0), construct.__class__.__name__)

    def test_duplicate_points_do_not_crash(self):
        vecs = np.repeat(_rand(1, seed=11), 50, axis=0)
        for construct in get_constructors():
            centers, radii = run_construct(construct, 5, vecs)
            self.assertTrue(_covering_ok(vecs, centers, radii), construct.__class__.__name__)

    def test_output_dtype_is_float32(self):
        vecs = _rand(100, seed=13)
        for construct in get_constructors():
            centers, radii = run_construct(construct, 10, vecs)
            name = construct.__class__.__name__
            self.assertEqual(centers.dtype, np.float32, name)
            self.assertEqual(radii.dtype, np.float32, name)

    def test_accepts_non_contiguous_and_float64_input(self):
        """Callers (BallCoverFilter) don't guarantee contiguous float32
        arrays; every constructor must coerce rather than assume."""
        vecs = _rand(80, seed=19).astype(np.float64)[::2]  # non-contiguous, wrong dtype
        self.assertFalse(vecs.flags["C_CONTIGUOUS"])
        for construct in get_constructors():
            centers, radii = run_construct(construct, 8, vecs)
            self.assertTrue(_covering_ok(vecs.astype(np.float32), centers, radii), construct.__class__.__name__)

    def test_deterministic_given_same_input(self):
        """Gonzalez is a pure, seeded numpy pipeline with no threaded
        reduction order to introduce nondeterminism."""
        vecs = _rand(120, seed=17)
        construct = GonzalesConstruct()
        c1, r1 = run_construct(construct, 12, vecs)
        c2, r2 = run_construct(construct, 12, vecs)
        np.testing.assert_array_equal(c1, c2)
        np.testing.assert_array_equal(r1, r2)


def _unit(x: np.ndarray) -> np.ndarray:
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


# ---------------------------------------------------------------------------
# exact_cover.sphere_radius / sphere_centers -- the shared sphere projection
# ---------------------------------------------------------------------------

class TestSphereProjection(unittest.TestCase):

    def test_sphere_radius_detects_equal_norm_data(self):
        self.assertAlmostEqual(sphere_radius(_unit(_rand(50, seed=41))), 1.0, places=5)
        self.assertAlmostEqual(sphere_radius(3.0 * _unit(_rand(50, seed=42))), 3.0, places=4)
        self.assertEqual(sphere_radius(_rand(50, seed=43)), 0.0)
        self.assertEqual(sphere_radius(np.zeros((5, DIM), dtype=np.float32)), 0.0)
        self.assertEqual(sphere_radius(np.empty((0, DIM), dtype=np.float32)), 0.0)

    def test_sphere_centers_share_one_norm_and_still_cover(self):
        """Only the centers move; exact_cover must still hold every vector,
        including when some projected centers end up with no members."""
        vecs = _unit(_clustered(6, 30, spread=0.3, seed=44))
        centroids = np.vstack([vecs[i * 30:(i + 1) * 30].mean(0) for i in range(6)] +
                              [_rand(3, seed=45)])  # 3 centers far from any data
        moved = sphere_centers(vecs, centroids, 1.0)
        norms = np.linalg.norm(moved, axis=1)
        self.assertLess(norms.max() - norms.min(), 1e-5)
        self.assertGreaterEqual(norms.min(), 1.0 - 1e-6)
        # directions are preserved
        cos = np.einsum("ij,ij->i", moved / norms[:, None],
                        centroids / np.linalg.norm(centroids, axis=1, keepdims=True))
        self.assertTrue(np.all(cos > 1.0 - 1e-5))
        centers, radii = exact_cover(vecs, moved)
        self.assertTrue(_covering_ok(vecs, centers, radii))

    def test_sphere_centers_norm_is_clipped_for_wide_caps(self):
        e = np.eye(DIM, dtype=np.float32)
        vecs = np.vstack([e, -e])
        moved = sphere_centers(vecs, np.full((1, DIM), 1e-3, dtype=np.float32), 1.0)
        self.assertTrue(np.all(np.isfinite(moved)))
        self.assertLessEqual(float(np.linalg.norm(moved)), 5.0 + 1e-5)


# ---------------------------------------------------------------------------
# SphereMergeConstruct -- the sphere path and its duplicate/empty handling
# ---------------------------------------------------------------------------

class TestSphereMergeConstruct(unittest.TestCase):

    def test_covers_unit_normalized_data(self):
        """Equal-norm input takes the sphere path (centers moved off the
        MEB centers), so the shared contract's Gaussian data never reaches it."""
        for vecs in (_unit(_rand(400, seed=31)), _unit(_clustered(8, 40, spread=0.2, seed=32))):
            for budget in (5, 30, 120):
                centers, radii = run_construct(SphereMergeConstruct(), budget, vecs)
                self.assertLessEqual(len(centers), budget)
                self.assertTrue(_covering_ok(vecs, centers, radii), f"budget={budget}")

    def test_centers_share_one_norm_on_sphere_data(self):
        """A common norm is what keeps nearest-center assignment angular;
        unequal norms break exact_cover's and _tight_radii's assumption."""
        centers, _ = run_construct(SphereMergeConstruct(), 20, _unit(_rand(300, seed=33)))
        norms = np.linalg.norm(centers, axis=1)
        self.assertLess(norms.max() - norms.min(), 1e-5 * norms.mean())
        self.assertGreaterEqual(norms.min(), 1.0 - 1e-6)

    def test_scaled_sphere_is_detected(self):
        vecs = 10.0 * _unit(_rand(300, seed=34))
        centers, radii = run_construct(SphereMergeConstruct(), 20, vecs)
        norms = np.linalg.norm(centers, axis=1)
        self.assertLess(norms.max() - norms.min(), 1e-5 * norms.mean())
        self.assertGreaterEqual(norms.min(), 10.0 * (1.0 - 1e-6))
        self.assertTrue(_covering_ok(vecs, centers, radii, tol=1e-2))

    def test_data_off_a_sphere_keeps_meb_centers(self):
        centers, radii = run_construct(SphereMergeConstruct(), 20, _rand(300, seed=35))
        norms = np.linalg.norm(centers, axis=1)
        self.assertGreater(norms.max() - norms.min(), 0.1)

    def test_heavy_exact_duplicates_on_sphere(self):
        """Duplicates collapse to one zero-radius ball before merging; the
        refinement round must not then lose a vector or leave a ball over
        budget when several centers compete for the same repeated point."""
        rng = np.random.default_rng(36)
        distinct = _unit(_rand(6, seed=37))
        vecs = np.vstack([np.repeat(distinct, 40, axis=0), _unit(_rand(20, seed=38))])
        vecs = vecs[rng.permutation(len(vecs))]
        for budget in (3, 8, 25):
            centers, radii = run_construct(SphereMergeConstruct(), budget, vecs)
            self.assertLessEqual(len(centers), budget)
            self.assertTrue(np.all(radii >= 0.0))
            self.assertTrue(_covering_ok(vecs, centers, radii), f"budget={budget}")

    def test_fewer_distinct_points_than_budget(self):
        """n > k but only 3 distinct vectors: no merge runs, and centers the
        refinement round leaves empty are dropped rather than returned as
        zero-radius balls."""
        vecs = np.repeat(_unit(_rand(3, seed=39)), 20, axis=0)
        centers, radii = run_construct(SphereMergeConstruct(), 10, vecs)
        self.assertLessEqual(len(centers), 3)
        self.assertTrue(_covering_ok(vecs, centers, radii))

    def test_antipodal_points_single_ball(self):
        """A cap wider than a hemisphere: the MEB center is ~0 and the
        norm that minimizes the cap is unbounded. Must still cover, finitely."""
        e = np.eye(DIM, dtype=np.float32)
        vecs = np.vstack([e, -e])
        centers, radii = run_construct(SphereMergeConstruct(), 1, vecs)
        self.assertEqual(len(centers), 1)
        self.assertTrue(np.all(np.isfinite(centers)) and np.all(np.isfinite(radii)))
        self.assertTrue(_covering_ok(vecs, centers, radii))


# ---------------------------------------------------------------------------
# GonzalesConstruct -- farthest-first specific properties
# ---------------------------------------------------------------------------

class TestGonzalesConstruct(unittest.TestCase):

    def test_returns_exactly_k_balls(self):
        """Gonzalez always fills the requested budget exactly in the
        non-trivial case."""
        vecs = _rand(200, seed=3)
        centers, radii = run_construct(GonzalesConstruct(), 20, vecs)
        self.assertEqual(len(centers), 20)
        self.assertEqual(len(radii), 20)

    def test_centers_are_original_data_points(self):
        """Defining property of farthest-first traversal: every chosen
        center is literally one of the input vectors, unlike a
        centroid-based method (rptree, IVF)."""
        vecs = _rand(100, seed=2)
        centers, _ = run_construct(GonzalesConstruct(), 10, vecs)
        for c in centers:
            self.assertTrue(np.any(np.all(vecs == c, axis=1)), "center not found among input vectors")

    def test_centers_are_far_apart_relative_to_random_pairs(self):
        """Farthest-first is a 2-approximation to the max-radius k-center
        objective -- chosen centers should be markedly more spread out than
        a same-size random sample."""
        vecs = _clustered(n_clusters=4, per_cluster=50, sep=20.0, seed=5)
        centers, _ = run_construct(GonzalesConstruct(), 8, vecs)

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
        centers, _ = run_construct(GonzalesConstruct(), k, vecs)
        np.testing.assert_allclose(centers, vecs[expected_idx], atol=1e-3)


if __name__ == "__main__":
    unittest.main()