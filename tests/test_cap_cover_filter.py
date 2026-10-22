import os
import unittest
import numpy as np
from filters.cap_cover.cap_cover_filter import CapCoverFilter, Quantize, AUTO_CANDIDATES, SCAN_NS_PER_BYTE
from filters.cap_cover.cap_geometry import (angle_between, unit_rows, enclosing_cap, exact_cap_cover,
                                            knn_directions, log10_cap_measure, log_cap_measure)
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

    def test_log10_cap_measure(self):
        from scipy.special import betainc
        psi = np.array([1e-3, 0.1, 0.5, 1.0, 1.4, np.pi / 2, 2.0, 3.0])
        # On S^2 a cap's share of the sphere is (1 - cos psi) / 2 = sin^2(psi / 2) exactly.
        np.testing.assert_allclose(log10_cap_measure(psi, 3), np.log10(np.sin(psi / 2) ** 2), rtol=1e-12)
        mid = psi[(psi > 0.3) & (psi <= np.pi / 2)]
        np.testing.assert_allclose(log10_cap_measure(mid, 48),
                                   np.log10(betainc(23.5, 0.5, np.sin(mid) ** 2) / 2), rtol=1e-10)
        # Far below float64's range at d = 384, still finite and ordered.
        tiny = log10_cap_measure(np.array([0.01, 0.05, 0.1]), 384)
        self.assertTrue(np.all(np.isfinite(tiny)) and np.all(np.diff(tiny) > 0) and tiny[0] < -700)

    def test_log_cap_measure_matches_series(self):
        rng = np.random.default_rng(0)
        for dim in (9, 48, 384, 32.5):
            psi = np.concatenate([rng.uniform(1e-6, 0.05, 500), rng.uniform(0.05, np.pi, 2000)])
            np.testing.assert_allclose(log_cap_measure(psi, dim), log10_cap_measure(psi, dim) * np.log(10.0),
                                       rtol=0, atol=1e-4)

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

    def test_auto_selects_least_volume(self):
        X = _clustered(n_clusters=60, per=10, dim=96, spread=0.4)
        f = CapCoverFilter.auto(96, 4000, mixed=False)
        f.construct(X)
        self.assertEqual(f.quantize.value, min(f.selection_scores, key=f.selection_scores.get))
        _assert_no_fn(self, f, X)

    def test_auto_mixed_encodings(self):
        X = _clustered(n_clusters=60, per=10, dim=96, spread=0.4)
        rng = np.random.default_rng(7)
        Q = np.vstack([_near(X, 0.6, n=2), _unit(rng.standard_normal((300, 96)))]).astype(np.float32)
        mixed_seen = False
        for budget in (1500, 4000, 12000):
            with self.subTest(budget=budget):
                f = CapCoverFilter.auto(96, budget)
                f.construct(X)
                levels = f._cap_level[f.all_caps]
                mixed_seen |= len(np.unique(levels)) > 1
                self.assertLessEqual(f.get_bytesize(), budget)
                # Bytes are each cap's own code and angle bytes.
                groups = int((f._code_sizes[levels] + 1).sum())
                expect = groups + f._codebook_bytes() + sum(
                    f._quantizer_bytes(int(l)) + f._GROUP_BYTES for l in np.unique(levels) if l != f._uniform)
                self.assertEqual(f.get_bytesize(), expect)
                _assert_no_fn(self, f, X)
                self.assertTrue((f.surface_angle(X) <= 1e-9).all())
                for tau in (0.0, TAU, 0.6):
                    got = np.asarray(f.query(Q, tau))
                    self.assertTrue(np.array_equal(got, TestCapCoverScan._reference(f, Q, tau)))
                    single = np.concatenate([f.query(Q[s:s + 1], tau) for s in range(0, 120)])
                    self.assertTrue(np.array_equal(single, got[:120]))
        self.assertTrue(mixed_seen, "no budget mixed encodings")

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

    def test_auto_defaults(self):
        f = CapCoverFilter.auto(DIM, 20000)
        self.assertEqual(set(f.levels), set(AUTO_CANDIDATES))
        self.assertTrue(f.mixed)

    def test_fast_nearest_codeword_matches_faiss(self):
        """compute_codes(fast=True) (scan_kernels.nearest_codeword, used to
        plan) picks faiss's codeword but for near-ties."""
        from filters.cap_cover.cap_cover_filter import _SeededPQ
        X = unit_rows(np.random.default_rng(4).standard_normal((300, 96))).astype(np.float32)
        for sub, bits in ((24, 12), (8, 10), (12, 8)):
            pq = _SeededPQ(96, sub, bits)
            same = (pq.compute_codes(X, fast=True) == pq.compute_codes(X)).all(axis=1).mean()
            self.assertGreaterEqual(same, 0.99)

    def test_cap_costs_match_ball_at_same_encoding(self):
        f = CapCoverFilter(384, quantize=Quantize.TQ4, insert_method=NearestInsert())
        self.assertEqual(f.bytes_per_cap(), 192 + 2)

    def test_angle_codes_round_up(self):
        for bits, insert in ((16, NearestInsert()), (8, None)):
            f = CapCoverFilter(DIM, max_caps=10, insert_method=insert, angle_bits=bits)
            f._angle_lo, f._angle_hi = 0.3, 1.7
            theta = np.random.default_rng(0).uniform(0.3, 1.7, 1000)
            self.assertTrue((f._angle_value(f._angle_code(theta)) >= theta).all())


class TestCapCoverScan(unittest.TestCase):
    """query()'s Rust scan against the float64 rule it implements."""

    ENCODINGS = (Quantize.TQ1, Quantize.TQ2, Quantize.TQ3, Quantize.TQ4, Quantize.TQ8, Quantize.SPQ12,
                 Quantize.SPQ16, Quantize.SPQ24, Quantize.SPQ36, Quantize.SPQ48, Quantize.SPQ64,
                 Quantize.FP32, Quantize.FP16)

    @staticmethod
    def _reference(f, Q, tau):
        """The accept rule in float64, straight from the stored caps."""
        Qw = f.to_working(Q)
        n = np.linalg.norm(Qw, axis=1)
        beta, feasible = f._beta(n, tau)
        D = f.decode_caps(f.all_caps)
        theta = f.all_angles
        ang = angle_between(unit_rows(Qw)[:, None, :], D[None, :, :])
        hit = (ang <= theta[None, :] + beta[:, None]) | (theta[None, :] + beta[:, None] >= np.pi)
        return hit.any(axis=1) & feasible

    def test_scan_matches_float64_rule(self):
        X = _clustered()
        rng = np.random.default_rng(5)
        Q = np.vstack([_near(X, 0.6, n=2), _unit(rng.standard_normal((200, DIM))), X[:50] * 1.07,
                       X[:50] * 0.9]).astype(np.float32)
        for q in self.ENCODINGS:
            for tau in (0.0, TAU, 0.6, 1.5):
                with self.subTest(q=q, tau=tau):
                    f = CapCoverFilter(DIM, max_caps=30, quantize=q)
                    f.construct(X)
                    got = np.asarray(f.query(Q, tau))
                    self.assertTrue(np.array_equal(got, self._reference(f, Q, tau)))
                    # Batches of any size decide the same.
                    for size in (1, 3, 7):
                        small = np.concatenate([f.query(Q[s:s + size], tau) for s in range(0, len(Q), size)])
                        self.assertTrue(np.array_equal(small, got))


class TestCapCoverScalableConstruction(unittest.TestCase):
    """SphereMergeConstruct's large-n paths, forced at test size."""

    def test_knn_directions_cell_search(self):
        U = unit_rows(_clustered(n_clusters=60, per=30, dim=DIM, spread=0.4))
        exact, exact_dots = knn_directions(U, 8, exact_max=10 ** 9)
        approx, dots = knn_directions(U, 8, exact_max=0, probes=8)
        self.assertEqual(approx.shape, exact.shape)
        self.assertFalse((approx == np.arange(len(U))[:, None]).any())
        valid = approx >= 0
        np.testing.assert_allclose(dots[valid], np.einsum("ij,ij->i", U[np.nonzero(valid)[0]], U[approx[valid]]),
                                   rtol=0, atol=1e-12)
        recall = np.mean([len(np.intersect1d(a, e)) / len(e) for a, e in zip(approx, exact)])
        self.assertGreater(recall, 0.9)

    def test_local_paths_no_false_negatives(self):
        X = _clustered(n_clusters=40, per=10, dim=96)
        local = dict(knn_exact_max=0, exact_pairs=0)
        for budget in (1500, 4000, 12000):
            with self.subTest(budget=budget):
                f = CapCoverFilter.auto(96, budget, construct_method=SphereMergeConstruct(**local))
                f.construct(X)
                self.assertLessEqual(f.get_bytesize(), budget)
                _assert_no_fn(self, f, X)
        for q in (Quantize.TQ2, Quantize.SPQ24):
            with self.subTest(q=q):
                f = CapCoverFilter(96, max_caps=50, quantize=q, construct_method=SphereMergeConstruct(**local))
                f.construct(X)
                _assert_no_fn(self, f, X)

    def test_finish_cover_is_labelled(self):
        X = unit_rows(_clustered(n_clusters=30, per=12))
        sm = SphereMergeConstruct(exact_pairs=0, knn_exact_max=0)
        f = CapCoverFilter(DIM, max_caps=40, quantize=Quantize.FP32, construct_method=sm)
        for _, cover in sm.iter_candidates(f, X, [40]):
            cover = sm.finish(f, X, cover)
        U, theta, labels, nbr = cover.directions, cover.angles, cover.labels, cover.neighbors
        self.assertTrue((angle_between(X, U[labels]) <= theta[labels] + 1e-12).all())
        self.assertEqual(len(nbr), len(U))
        self.assertTrue(((nbr >= -1) & (nbr < len(U))).all())

    def test_duplicates_count_once_as_caps(self):
        X = _clustered(n_clusters=20, per=6)
        Xd = np.vstack([X, X[:40], X[:40]])
        f = CapCoverFilter(DIM, max_caps=10 ** 6, quantize=Quantize.TQ4)
        f.construct(Xd)
        self.assertEqual(f._num_caps, len(X))
        _assert_no_fn(self, f, X)


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

    def test_cap_merge_insert_state_bytes(self):
        self.assertEqual(CapMergeInsert()._state_bytes(1000), 2 + 4)
        self.assertEqual(CapMergeInsert(mass_weight=1.0, cost="power", exponent=8.0)._state_bytes(1000), 2 + 2 + 4)
        f, X = self._stream(CapMergeInsert(mass_weight=1.0, cost="power", exponent=8.0))
        _assert_no_fn(self, f, X)

    def test_cap_merge_insert_fitted_codes_no_false_negatives(self):
        f, X = self._stream(CapMergeInsert(fit_codes=True))
        self.assertLessEqual(f._num_caps, f.max_caps)
        _assert_no_fn(self, f, X)

    def test_fit_cap_code_never_worse_than_own_code(self):
        rng = np.random.default_rng(4)
        X = _clustered(dim=96)
        for q in (Quantize.TQ2, Quantize.TQ4, Quantize.SPQ24, Quantize.SPQ64, Quantize.FP16):
            with self.subTest(q=q):
                f = CapCoverFilter(96, max_caps=40, quantize=q)
                f.construct(X)
                for _ in range(10):
                    P = unit_rows(rng.standard_normal((2, 96)) * 0.3 + f.to_working(X[:1]))
                    t = rng.uniform(0.0, 0.4, 2)
                    u, _ = enclosing_cap(P[0], t[0], P[1], t[1])
                    reach = lambda codes: (angle_between(f.decode(codes), P) + t).max()
                    self.assertLessEqual(reach(f.fit_cap_code(u, P, t)), reach(f.encode(u[None])) + 1e-12)


class TestAutoScanLimit(unittest.TestCase):
    """auto() drops encodings whose estimated scan of the whole budget is too slow."""

    def test_no_limit_keeps_every_candidate(self):
        self.assertEqual(len(CapCoverFilter.auto(384, 1_550_000).levels), len(AUTO_CANDIDATES))

    def test_limit_drops_slow_encodings_only_at_large_budgets(self):
        small = {q for q in CapCoverFilter.auto(384, 123_000, scan_limit_ms=1.0).levels}
        large = {q for q in CapCoverFilter.auto(384, 6_000_000, scan_limit_ms=1.0).levels}
        self.assertEqual(small, set(AUTO_CANDIDATES))
        self.assertEqual(large, {Quantize.TQ2, Quantize.TQ4})
        for q in large:
            self.assertLessEqual(6_000_000 * SCAN_NS_PER_BYTE[q] / 1e6, 1.0)

    def test_nothing_fits_keeps_the_cheapest(self):
        self.assertEqual(CapCoverFilter.auto(384, 10 ** 9, scan_limit_ms=1.0).levels, (Quantize.TQ4,))

    def test_environment_default(self):
        old = os.environ.get("CAPCOVER_SCAN_LIMIT_MS")
        os.environ["CAPCOVER_SCAN_LIMIT_MS"] = "1.0"
        try:
            self.assertEqual(len(CapCoverFilter.auto(384, 6_000_000).levels), 2)
            self.assertEqual(len(CapCoverFilter.auto(384, 6_000_000, scan_limit_ms=1e9).levels), len(AUTO_CANDIDATES))
        finally:
            if old is None:
                os.environ.pop("CAPCOVER_SCAN_LIMIT_MS")
            else:
                os.environ["CAPCOVER_SCAN_LIMIT_MS"] = old


if __name__ == "__main__":
    unittest.main()
