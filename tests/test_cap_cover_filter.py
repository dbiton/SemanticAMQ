import unittest
import numpy as np
from filters.cap_cover.cap_cover_filter import CapCoverFilter, Quantize, AUTO_CANDIDATES
from filters.cap_cover.cap_geometry import angle_between, unit_rows, enclosing_cap, exact_cap_cover
from filters.cap_cover.gonzales_construct import GonzalesConstruct
from filters.cap_cover.minimax_construct import MinimaxConstruct
from filters.cap_cover.sphere_merge_construct import SphereMergeConstruct
from filters.cap_cover.nearest_insert import NearestInsert
from filters.cap_cover.cap_merge_insert import CapMergeInsert

DIM = 48
TAU = 0.2


def _unit(X):
    return (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.float32)


def _clustered(n_clusters=30, per=12, dim=DIM, spread=0.15, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, dim))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    parts = [c + spread * rng.standard_normal((per, dim)) / np.sqrt(dim) for c in centers]
    return _unit(np.vstack(parts))


def _near(X, tau, n=4, seed=1):
    """Unit vectors strictly within Euclidean distance tau of rows of X."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        noise = rng.standard_normal(X.shape)
        noise /= np.linalg.norm(noise, axis=1, keepdims=True)
        q = X + noise * rng.uniform(0, tau, (len(X), 1))
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
        keep = np.linalg.norm(q - X, axis=1) <= 0.98 * tau
        out.append(q[keep])
    return np.vstack(out).astype(np.float32)


def _assert_no_fn(test, f, X, tau=TAU):
    Q = np.vstack([X, _near(X, tau)])
    test.assertTrue(all(f.query(Q, tau)), "a vector within tau of a member was rejected")


class TestCapGeometry(unittest.TestCase):

    def test_enclosing_cap_holds_both(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            ua, ub = unit_rows(rng.standard_normal((2, DIM)))
            ta, tb = rng.uniform(0, 1.2, 2)
            u, t = enclosing_cap(ua, ta, ub, tb)
            self.assertLessEqual(angle_between(u[None], ua[None])[0] + ta, t + 1e-9)
            self.assertLessEqual(angle_between(u[None], ub[None])[0] + tb, t + 1e-9)

    def test_exact_cap_cover_covers(self):
        X = unit_rows(_clustered())
        U, theta = exact_cap_cover(X, X[::7])
        held = angle_between(X[:, None, :], U[None, :, :]) <= theta[None, :] + 1e-12
        self.assertTrue(held.any(axis=1).all())


class TestCapCoverSoundness(unittest.TestCase):

    def test_no_false_negatives_across_encodings(self):
        X = _clustered()
        for q in (Quantize.FP32, Quantize.TQ4, Quantize.TQ2, Quantize.TQ1):
            for construct in (GonzalesConstruct(), MinimaxConstruct(rounds=2), SphereMergeConstruct()):
                with self.subTest(q=q, construct=type(construct).__name__):
                    f = CapCoverFilter(DIM, construct_method=construct, max_caps=40, quantize=q)
                    f.construct(X)
                    _assert_no_fn(self, f, X)

    def test_no_false_negatives_seeded_pq(self):
        X = _clustered()
        f = CapCoverFilter(48, max_caps=40, quantize=Quantize.SPQ12)   # dim 48: one 48-dim slice per code
        f.construct(X)
        _assert_no_fn(self, f, X)

    def test_no_false_negatives_with_and_without_code_fitting(self):
        X = _clustered(dim=96)
        for fit in (True, False):
            with self.subTest(fit=fit):
                f = CapCoverFilter(96, max_caps=50, quantize=Quantize.SPQ12, fit_codes=fit)
                f.construct(X)
                _assert_no_fn(self, f, X)

    def test_members_inside_their_caps(self):
        X = _clustered()
        f = CapCoverFilter(DIM, max_caps=30, quantize=Quantize.TQ2)
        f.construct(X)
        self.assertTrue((f.surface_angle(X) <= 1e-9).all())

    def test_auto_no_false_negatives_and_on_budget(self):
        X = _clustered(n_clusters=40, per=10, dim=96)
        for budget in (1500, 4000, 12000):
            with self.subTest(budget=budget):
                f = CapCoverFilter.auto(96, budget)
                f.construct(X)
                self.assertLessEqual(f.get_bytesize(), budget)
                self.assertIn(f.quantize, AUTO_CANDIDATES)
                _assert_no_fn(self, f, X)

    def test_query_rule_for_non_unit_queries(self):
        X = _clustered()
        f = CapCoverFilter(DIM, max_caps=40, quantize=Quantize.TQ4)
        f.construct(X)
        # A scaled member within tau of the member must be accepted; one
        # whose norm alone puts it farther than tau from the sphere cannot be.
        self.assertTrue(all(f.query(X * 1.15, TAU)))
        self.assertTrue(all(f.query(X * 0.85, TAU)))
        self.assertFalse(any(f.query(X * 1.5, TAU)))
        self.assertFalse(any(f.query(np.zeros((3, DIM), dtype=np.float32), TAU)))

    def test_far_queries_rejected(self):
        X = _clustered()
        f = CapCoverFilter(DIM, max_caps=60, quantize=Quantize.FP32)
        f.construct(X)
        self.assertFalse(any(f.query(-X, 0.05)))

    def test_rejects_non_unit_members(self):
        f = CapCoverFilter(DIM, max_caps=10)
        with self.assertRaises(ValueError):
            f.construct(np.random.default_rng(0).standard_normal((20, DIM)).astype(np.float32))


class TestCapCoverBudget(unittest.TestCase):

    def test_bytesize_within_max_memory(self):
        X = _clustered(dim=96)
        for q in (Quantize.TQ4, Quantize.TQ1, Quantize.SPQ24, Quantize.SPQ48):
            with self.subTest(q=q):
                f = CapCoverFilter(96, max_memory=3000, quantize=q)
                f.construct(X)
                self.assertLessEqual(f.get_bytesize(), 3000)
                self.assertGreater(f._num_caps, 0)

    def test_cap_costs_match_ball_at_same_encoding(self):
        f = CapCoverFilter(384, quantize=Quantize.TQ4, insert_method=NearestInsert())
        self.assertEqual(f.bytes_per_cap(), 192 + 2)

    def test_angle_codes_round_up(self):
        for bits, insert in ((16, NearestInsert()), (8, None)):
            f = CapCoverFilter(DIM, max_caps=10, insert_method=insert, angle_bits=bits)
            f._angle_lo, f._angle_hi = 0.3, 1.7
            theta = np.random.default_rng(0).uniform(0.3, 1.7, 1000)
            self.assertTrue((f._angle_value(f._angle_code(theta)) >= theta).all())


class TestCapCoverInsert(unittest.TestCase):

    def _stream(self, insert):
        X = _clustered(n_clusters=40, per=10)
        rng = np.random.default_rng(3)
        X = X[rng.permutation(len(X))]
        f = CapCoverFilter(DIM, max_memory=40 * 230, quantize=Quantize.TQ4, insert_method=insert)
        f.construct(X[:100])
        for s in range(100, len(X), 50):
            f.insert(X[s:s + 50])
            self.assertLessEqual(f.get_bytesize(), 40 * 230)
        return f, X

    def test_nearest_insert_no_false_negatives(self):
        f, X = self._stream(NearestInsert())
        _assert_no_fn(self, f, X)

    def test_cap_merge_insert_no_false_negatives(self):
        f, X = self._stream(CapMergeInsert())
        self.assertLessEqual(f._num_caps, f.max_caps)
        _assert_no_fn(self, f, X)


if __name__ == "__main__":
    unittest.main()
