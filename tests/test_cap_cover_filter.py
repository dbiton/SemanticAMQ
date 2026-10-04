import unittest
import numpy as np
from filters.cap_cover.cap_cover_filter import CapCoverFilter, Quantize, AUTO_CANDIDATES
from filters.cap_cover.cap_geometry import (angle_between, unit_rows, enclosing_cap, exact_cap_cover,
                                            knn_directions, log10_cap_measure, log_cap_measure,
                                            min_enclosing_cap)
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

    def test_min_enclosing_cap(self):
        rng = np.random.default_rng(1)
        # Points at angle 0.4 from a pole along +-10 orthonormal tangent
        # directions (their hull holds the pole's axis), plus interior
        # points: the exact cap is the pole at 0.4.
        basis, _ = np.linalg.qr(rng.standard_normal((DIM, 11)))
        pole, tangent = basis[:, 0], basis[:, 1:].T
        ring = np.vstack([tangent, -tangent])
        P = np.cos(0.4) * pole[None, :] + np.sin(0.4) * ring
        inner = unit_rows(np.cos(0.2) * pole[None, :] + np.sin(0.2) * ring[:10])
        P = np.vstack([P, inner])
        start = unit_rows((pole + 0.3 * ring[0])[None])[0]
        u = min_enclosing_cap(P, start)
        self.assertAlmostEqual(float(angle_between(P, np.broadcast_to(u, P.shape)).max()), 0.4, places=6)
        # On clustered members it is never looser than their mean direction.
        X = unit_rows(_clustered(n_clusters=1, per=200, spread=0.6))
        mean = unit_rows(X.mean(axis=0)[None])[0]
        u = min_enclosing_cap(X, mean)
        self.assertLessEqual(angle_between(X, np.broadcast_to(u, X.shape)).max(),
                             angle_between(X, np.broadcast_to(mean, X.shape)).max() + 1e-12)

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
        self.assertEqual(f.selection, "volume")
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
                # Bytes are what each encoding group costs in set storage
                # (at most its caps' own code and angle bytes).
                groups = sum(f._group_bytes(int(l), int(n)) for l, n in zip(*np.unique(levels, return_counts=True)))
                self.assertLessEqual(groups, (f._code_sizes[levels] + 1).sum())
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

    def test_auto_with_basis_candidates(self):
        X = _clustered(n_clusters=60, per=10, dim=96, spread=0.4)
        basis = (Quantize.PSPQ6, Quantize.PSPQ9, Quantize.PSPQ12, Quantize.DPQ6, Quantize.DPQ9, Quantize.DPQ12)
        for budget in (6000, 20000):
            with self.subTest(budget=budget):
                f = CapCoverFilter.auto(96, budget, candidates=AUTO_CANDIDATES + basis)
                f.construct(X)
                self.assertLessEqual(f.get_bytesize(), budget)
                _assert_no_fn(self, f, X)

    def test_boot_basis_encodings(self):
        """Boot levels (_BootPQ): chosen on low-rank data, no false
        negatives, within budget, decisions as the float64 rule, and the
        basis rebuilt from the stored whole-space codes decodes bit for bit
        as stored (what a decoder of the stored filter would do)."""
        from filters.cap_cover.cap_cover_filter import BOOT_CANDIDATES, _BootPQ
        rng = np.random.default_rng(11)
        basis = np.linalg.qr(rng.standard_normal((96, 12)))[0].T
        X = _unit(_clustered(n_clusters=80, per=8, dim=12, spread=0.3) @ basis + 0.05 * rng.standard_normal((640, 96)))
        Q = np.vstack([_near(X, 0.6, n=1), _unit(rng.standard_normal((200, 96)))]).astype(np.float32)
        chosen = False
        for budget in (6000, 12000):
            with self.subTest(budget=budget):
                # Earlier defaults and cheaper boot codes allowed, so the
                # plan takes boot levels on this data.
                f = CapCoverFilter.auto(96, budget, candidates=AUTO_CANDIDATES + BOOT_CANDIDATES, mix_finer=0,
                                        boot_equal=False)
                f.construct(X)
                self.assertLessEqual(f.get_bytesize(), budget)
                _assert_no_fn(self, f, X)
                for tau in (TAU, 0.6):
                    self.assertTrue(np.array_equal(np.asarray(f.query(Q, tau)), TestCapCoverScan._reference(f, Q, tau)))
                levels = f._cap_level[f.all_caps]
                boot = [int(l) for l in np.unique(levels) if f._is_boot(int(l))]
                chosen |= bool(boot)
                if boot:
                    whole = f.all_caps[~np.isin(levels, boot)]
                    anchors = f._decode_levels(f._stored_codes(whole), f._cap_level[whole])
                    for l in boot:
                        sq = _BootPQ(96, f._quantizers[l].rank, f._quantizers[l].sub, f._quantizers[l].bits)
                        sq.train(anchors)
                        codes = f._stored_codes(f.all_caps[levels == l])[:, :f.code_size(l)]
                        self.assertTrue(np.array_equal(sq.decode(codes), f._quantizers[l].decode(codes)))
        self.assertTrue(chosen, "no boot level chosen on low-rank data")

    def test_auto_holdout_selection(self):
        X = _clustered(n_clusters=60, per=10, dim=96, spread=0.4)
        for budget in (1500, 4000):
            with self.subTest(budget=budget):
                proxy = CapCoverFilter.auto(96, budget, selection="proxy")
                proxy.construct(X)
                f = CapCoverFilter.auto(96, budget, selection="holdout")
                f.construct(X)
                self.assertLessEqual(f.get_bytesize(), budget)
                _assert_no_fn(self, f, X)
                if f.selection_override is None:
                    # No override: exactly the proxy's filter.
                    self.assertEqual(f.quantize, proxy.quantize)
                    self.assertTrue(np.array_equal(f._stored_codes(f.all_caps), proxy._stored_codes(proxy.all_caps)))
                    self.assertTrue(np.array_equal(f.all_angles, proxy.all_angles))

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

    def test_basis_encodings_charge_their_basis(self):
        X = _clustered(dim=96)
        for q, rank in ((Quantize.PSPQ12, 64), (Quantize.DPQ12, 64), (Quantize.DPQ9, 48)):
            with self.subTest(q=q):
                f = CapCoverFilter(96, max_memory=12000, quantize=q)
                f.construct(X)
                self.assertLessEqual(f.get_bytesize(), 12000)
                # rank TQ4 rows of 48 B at d = 96, plus the TQ4 table and seeds.
                self.assertGreaterEqual(f._codebook_bytes(), rank * 48)
                _assert_no_fn(self, f, X)

    def test_dual_code_flag_picks_tighter_family(self):
        X = unit_rows(_clustered(dim=96, spread=0.4))
        f = CapCoverFilter(96, max_caps=10 ** 6, quantize=Quantize.DPQ12)
        f.construct(X)
        sq = f._quantizers[f._uniform]
        D = f.to_working(X)
        cb, cf = sq.basis.compute_codes(D), sq.full.compute_codes(D)
        both = np.minimum(angle_between(D, unit_rows(sq.basis.decode(cb))),
                          angle_between(D, unit_rows(sq.full.decode(cf))))
        got = angle_between(D, unit_rows(sq.decode(sq.compute_codes(D))))
        np.testing.assert_allclose(got, both, rtol=0, atol=1e-12)

    def test_auto_defaults(self):
        from filters.cap_cover.cap_cover_filter import DENSE10_LADDER
        f = CapCoverFilter.auto(DIM, 20000)
        self.assertEqual(f.mix_finer, 1)
        self.assertTrue(set(DENSE10_LADDER + AUTO_CANDIDATES) <= set(f.levels))
        self.assertTrue(f._set_storage())

    def test_fast_nearest_codeword_matches_faiss(self):
        """compute_codes(fast=True) (scan_kernels.nearest_codeword, used to
        plan) picks faiss's codeword but for near-ties."""
        from filters.cap_cover.cap_cover_filter import _SeededPQ
        X = unit_rows(np.random.default_rng(4).standard_normal((300, 96))).astype(np.float32)
        for sub, bits in ((24, 12), (8, 10), (12, 8)):
            pq = _SeededPQ(96, sub, bits)
            same = (pq.compute_codes(X, fast=True) == pq.compute_codes(X)).all(axis=1).mean()
            self.assertGreaterEqual(same, 0.99)

    def test_elias_fano_round_trip(self):
        from filters.cap_cover.cap_cover_filter import _ef_encode, _ef_layout
        from filters.cap_cover import scan_kernels
        rng = np.random.default_rng(3)
        for n, w in ((1, 8), (5, 8), (300, 8), (300, 16), (2000, 16), (5000, 8), (70000, 16), (900, 24)):
            with self.subTest(n=n, w=w):
                keys = np.sort(rng.integers(0, 1 << w, n))
                l, low, high = _ef_encode(keys, w)
                self.assertEqual((l, len(low), len(high)), _ef_layout(n, w))
                out = np.empty(n, dtype=np.int64)
                scan_kernels.ef_decode(low, high, n, l, out)
                self.assertTrue(np.array_equal(out, keys))

    def test_set_storage_keeps_codes_and_decisions(self):
        """Set storage only reorders the caps and moves each code's first
        bits into an Elias-Fano sequence: the same caps, the same decisions,
        fewer bytes, and get_bytesize() charges what the groups hold."""
        X = _clustered(n_clusters=60, per=10, dim=96, spread=0.4)
        rng = np.random.default_rng(8)
        Q = np.vstack([X, _unit(rng.standard_normal((300, 96)))]).astype(np.float32)
        for q in (Quantize.SPQ12, Quantize.SPQ16, Quantize.SPQ48, Quantize.TQ2, Quantize.FP16):
            with self.subTest(q=q):
                plain = CapCoverFilter(96, max_caps=400, quantize=q, set_codes=False)
                plain.construct(X)
                f = CapCoverFilter(96, max_caps=400, quantize=q)
                f.construct(X)
                self.assertIsNotNone(f._packed)
                key = lambda g: sorted(map(bytes, np.concatenate(
                    [g._stored_codes(g.all_caps), g._angle_codes[g.all_caps].astype(np.uint8)[:, None]], axis=1)))
                self.assertEqual(key(f), key(plain))
                for tau in (TAU, 0.6):
                    for size in (1, 32, len(Q)):
                        got = np.concatenate([f.query(Q[s:s + size], tau) for s in range(0, len(Q), size)])
                        self.assertTrue(np.array_equal(got, np.asarray(plain.query(Q, tau))))
                self.assertLessEqual(f.get_bytesize(), plain.get_bytesize())
        # Mixed encodings under a budget: within it, no false negatives.
        for budget in (1500, 6000):
            f = CapCoverFilter.auto(96, budget)
            f.construct(X)
            self.assertLessEqual(f.get_bytesize(), budget)
            _assert_no_fn(self, f, X)

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
    """query()'s compiled scan against the float64 rule it implements."""

    ENCODINGS = (Quantize.TQ1, Quantize.TQ2, Quantize.TQ3, Quantize.TQ4, Quantize.TQ8, Quantize.SPQ12,
                 Quantize.SPQ16, Quantize.SPQ24, Quantize.SPQ36, Quantize.SPQ48, Quantize.SPQ64,
                 Quantize.FP32, Quantize.FP16, Quantize.PSPQ6, Quantize.PSPQ9, Quantize.DPQ6, Quantize.DPQ9)

    def test_block_decoder_matches_quantizer(self):
        X = _clustered()
        for q in self.ENCODINGS:
            with self.subTest(q=q):
                f = CapCoverFilter(DIM, max_caps=40, quantize=q)
                f.construct(X)
                codes, keys, w, _ = f._level_codes(f._uniform, f.all_caps)
                raw = np.empty((len(codes), DIM), dtype=np.float32)
                norms = np.empty(len(codes))
                f._block_decoder()(codes, keys, w, raw, norms)
                ref = f._decode_raw(f._stored_codes(f.all_caps), f._uniform)
                self.assertTrue(np.array_equal(raw, ref))
                np.testing.assert_allclose(norms, np.linalg.norm(ref.astype(np.float64), axis=1), rtol=1e-12)

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
                    # Batches of any size decide the same (1-3 queries:
                    # accept_direct, for 12-bit seeded PQ).
                    for size in (1, 3, 7):
                        small = np.concatenate([f.query(Q[s:s + size], tau) for s in range(0, len(Q), size)])
                        self.assertTrue(np.array_equal(small, got))

    def test_single_query_tables_match_rule(self):
        """Batches of 1-3 queries through per-query tables (accept_table1
        for seeded PQ, forced for 12-bit codebooks too; accept_lut_levels
        for TurboQuant) decide as the float64 rule and as without them."""
        X = _clustered()
        rng = np.random.default_rng(9)
        Q = np.vstack([_near(X, 0.6, n=2), _unit(rng.standard_normal((150, DIM))), X[:40] * 1.07]).astype(np.float32)
        for q in (Quantize.SPQ12, Quantize.SPQ16, Quantize.SPQ24, Quantize.SPQ64, Quantize.SPQ20, Quantize.TQ1,
                  Quantize.TQ2, Quantize.TQ3, Quantize.TQ4):
            for tau in (0.0, TAU, 0.6):
                with self.subTest(q=q, tau=tau):
                    f = CapCoverFilter(DIM, max_caps=30, quantize=q)
                    f.construct(X)
                    f._TABLE1_MIN_CAPS = 0
                    ref = self._reference(f, Q, tau)
                    for size in (1, 3):
                        got = np.concatenate([f.query(Q[s:s + size], tau) for s in range(0, len(Q), size)])
                        self.assertTrue(np.array_equal(got, ref))
                    f.single_tables = False
                    plain = np.concatenate([f.query(Q[s:s + 1], tau) for s in range(len(Q))])
                    self.assertTrue(np.array_equal(plain, ref))

    def test_table_scan_matches_float64_rule(self):
        """The codeword-table scan, forced for every seeded-PQ encoding
        (query() takes it only where it pays, see _tables_pay)."""
        X = _clustered()
        rng = np.random.default_rng(6)
        Q = np.vstack([_near(X, 0.6, n=2), _unit(rng.standard_normal((200, DIM))), X[:50] * 1.07]).astype(np.float32)
        for q in (Quantize.SPQ12, Quantize.SPQ16, Quantize.SPQ24, Quantize.SPQ36, Quantize.SPQ48, Quantize.SPQ64):
            for tau in (0.0, TAU, 0.6, 1.5):
                with self.subTest(q=q, tau=tau):
                    f = CapCoverFilter(DIM, max_caps=30, quantize=q)
                    f.construct(X)
                    Qw = f.to_working(Q)
                    n = np.linalg.norm(Qw, axis=1)
                    beta, feasible = f._beta(n, tau)
                    Qn = Qw / n[:, None]
                    codes, keys, w, theta = f._level_codes(f._uniform, f.all_caps)
                    got = f._scan_tables(f._quantizers[f._uniform], codes, keys, w, theta, Qn, Qn.astype(np.float32),
                                         beta, feasible, f._slack(), np.zeros(len(Q), dtype=bool))
                    self.assertTrue(np.array_equal(got, self._reference(f, Q, tau)))


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

    def test_cap_merge_insert_dual_codes_no_false_negatives(self):
        X = _clustered(n_clusters=40, per=10, dim=96)
        X = X[np.random.default_rng(3).permutation(len(X))]
        budget = 40 * 60 + 4000
        f = CapCoverFilter(96, max_memory=budget, quantize=Quantize.DPQ12, insert_method=CapMergeInsert())
        f.construct(X[:150])
        for s in range(150, len(X), 50):
            f.insert(X[s:s + 50])
            self.assertLessEqual(f.get_bytesize(), budget)
        _assert_no_fn(self, f, X)

    def test_cap_merge_insert_fitted_codes_no_false_negatives(self):
        f, X = self._stream(CapMergeInsert(fit_codes=True))
        self.assertLessEqual(f._num_caps, f.max_caps)
        _assert_no_fn(self, f, X)

    def test_fit_cap_code_never_worse_than_own_code(self):
        rng = np.random.default_rng(4)
        X = _clustered(dim=96)
        for q in (Quantize.TQ2, Quantize.TQ4, Quantize.SPQ24, Quantize.SPQ64, Quantize.FP16, Quantize.PSPQ12,
                  Quantize.DPQ12):
            with self.subTest(q=q):
                f = CapCoverFilter(96, max_caps=40, quantize=q)
                f.construct(X)
                for _ in range(10):
                    P = unit_rows(rng.standard_normal((2, 96)) * 0.3 + f.to_working(X[:1]))
                    t = rng.uniform(0.0, 0.4, 2)
                    u, _ = enclosing_cap(P[0], t[0], P[1], t[1])
                    reach = lambda codes: (angle_between(f.decode(codes), P) + t).max()
                    self.assertLessEqual(reach(f.fit_cap_code(u, P, t)), reach(f.encode(u[None])) + 1e-12)


if __name__ == "__main__":
    unittest.main()
