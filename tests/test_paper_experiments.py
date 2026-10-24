"""The paper's added experiments: the ball accept rules of the cap-vs-ball ablation
(runner.ball_accept_thresholds), runner.ablation_worker, filter_latency_worker and
pinned_timing_worker."""
import unittest

import numpy as np

import runner


def _unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


class TestBallAcceptRules(unittest.TestCase):

    def test_ball_rules_contain_the_cap_rule(self):
        # Theorem 3: both balls accept a superset of the cap's tau-neighbourhood.
        theta = np.linspace(0.01, 1.2, 40)
        for tau in (0.05, 0.2, 0.45):
            alpha = 2 * np.arcsin(tau / 2)
            cap = np.cos(np.minimum(theta + alpha, np.pi))
            for rule in ("ball_sphere", "ball_meb"):
                thr = runner.ball_accept_thresholds(theta, tau, rule)
                self.assertTrue(np.all(thr <= cap + 1e-12), (rule, tau))
                self.assertTrue(np.any(thr < cap - 1e-6), (rule, tau))     # strictly more somewhere

    def test_meb_matches_the_closed_form(self):
        theta, tau = np.array([0.3, 0.7, 1.1]), 0.2
        want = np.cos(theta) - tau * np.tan(theta) - tau ** 2 / (2 * np.cos(theta))
        np.testing.assert_allclose(runner.ball_accept_thresholds(theta, tau, "ball_meb"), want, atol=1e-12)

    def test_wide_cap_meb_accepts_everything(self):
        thr = runner.ball_accept_thresholds(np.array([np.pi / 2, 2.0]), 0.1, "ball_meb")
        self.assertTrue(np.all(np.isneginf(thr)))

    def test_ball_rules_have_no_false_negatives(self):
        rng = np.random.default_rng(1)
        d, theta, tau = 24, 0.5, 0.25
        u = _unit(rng.normal(size=d))
        members = []
        while len(members) < 200:                      # members of the cap C(u, theta)
            x = _unit(u + 0.35 * rng.normal(size=d))
            if np.arccos(np.clip(x @ u, -1, 1)) <= theta:
                members.append(x)
        members = np.array(members)
        queries = _unit(members[rng.integers(0, 200, 400)] + 0.1 * rng.normal(size=(400, d)))
        near = (np.linalg.norm(queries[:, None] - members[None], axis=2).min(axis=1) <= tau)
        for rule in ("cap", "ball_sphere", "ball_meb"):
            acc = runner._accepted_by_rule(queries, u[None, :], np.array([theta]), [tau], rule)[0]
            self.assertTrue(np.all(acc[near]), rule)
            self.assertTrue(near.any())


class TestAblationWorker(unittest.TestCase):
    DIM = 384

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        centers = _unit(rng.normal(size=(15, cls.DIM)))
        cls.B = _unit(centers[rng.integers(0, 15, 600)] + 0.04 * rng.normal(size=(600, cls.DIM))).astype(np.float32)
        cls.Q = _unit(np.vstack([centers[rng.integers(0, 15, 80)] + 0.05 * rng.normal(size=(80, cls.DIM)),
                                 rng.normal(size=(40, cls.DIM))])).astype(np.float32)

    def test_rows_and_cap_vs_ball_ordering(self):
        rows = runner.ablation_worker(self.DIM, "synthetic", 2, 0.2, self.B, self.Q,
                                      deltas=(0.01, 0.05), arms=["full", "tq4_fixed"])
        self.assertEqual({r["arm"] for r in rows}, {"full", "ball_sphere", "ball_meb", "tq4_fixed"})
        self.assertEqual(len(rows), 4 * 2)
        by = {(r["arm"], r["delta"]): r for r in rows}
        for r in rows:
            self.assertEqual(r["fnr"], 0.0)
            self.assertEqual((r["dataset"], r["fold"], r["n_queries"]), ("synthetic", 2, 120))
        for delta in (0.01, 0.05):
            cap = by[("full", delta)]
            for ball in ("ball_sphere", "ball_meb"):
                self.assertGreaterEqual(by[(ball, delta)]["fpr"], cap["fpr"] - 1e-12)
                self.assertGreaterEqual(by[(ball, delta)]["log10_j_tau"], cap["log10_j_tau"])
                self.assertEqual(by[(ball, delta)]["memory_bytes"], cap["memory_bytes"])    # same bytes
        # the cap arm's rule is the filter's own: its decisions match query()
        filt = runner._cap_cover(self.DIM, 120)
        filt.construct(self.B)
        want = np.mean(filt.query(self.Q, distance_threshold=runner.tau_of_cosine_distance(0.05)))
        self.assertAlmostEqual(1.0 - by[("full", 0.05)]["accesses_saved_pct"] / 100.0, want, places=9)


class TestFilterLatencyWorker(unittest.TestCase):

    def test_rows_and_composition(self):
        dim = 384
        rng = np.random.default_rng(3)
        centers = _unit(rng.normal(size=(12, dim)))
        B = _unit(centers[rng.integers(0, 12, 900)] + 0.04 * rng.normal(size=(900, dim))).astype(np.float32)
        Q = _unit(np.vstack([centers[rng.integers(0, 12, 120)] + 0.05 * rng.normal(size=(120, dim)),
                             rng.normal(size=(80, dim))])).astype(np.float32)
        rows = runner.filter_latency_worker(dim, "synthetic", 0, 0.2, (0.02, 0.1), B, Q, n_timed=60)
        self.assertEqual(len(rows), 2)
        for r in rows:
            self.assertEqual(r["n_timed"], 60)
            self.assertGreaterEqual(r["accept_rate_all_pct"], r["positive_rate_pct"] - 1e-9)   # zero FN
            self.assertAlmostEqual(r["qps_with"], 1000.0 / r["with_mean_ms"], places=6)
            self.assertAlmostEqual(r["qps_without"], 1000.0 / r["backend_mean_ms"], places=6)
            self.assertGreater(r["filter_bytes"], 0)
            # the filter pays off exactly when the backend costs more than the break-even cost
            if np.isfinite(r["breakeven_backend_ms"]):
                self.assertEqual(r["throughput_improvement_pct"] > 0, r["backend_mean_ms"] > r["breakeven_backend_ms"])


class TestPinnedTimingWorker(unittest.TestCase):

    def test_rows(self):
        dim = 384
        rng = np.random.default_rng(4)
        centers = _unit(rng.normal(size=(8, dim)))
        B = _unit(centers[rng.integers(0, 8, 400)] + 0.05 * rng.normal(size=(400, dim))).astype(np.float32)
        rows = runner.pinned_timing_worker(dim, "synthetic", B, k_fractions=(0.05,), max_inserts=20)
        kinds = {(r["kind"], r["method"]) for r in rows}
        self.assertEqual(kinds, {("construct", m) for m in runner.CAP_CONSTRUCT_METHODS}
                         | {("insert", m) for m in runner.CAP_INSERT_STRATEGIES})
        for r in rows:
            if r["kind"] == "construct":
                self.assertGreater(r["seconds"], 0)
            else:
                self.assertEqual(r["n_inserts"], 20)
                self.assertLessEqual(r["p50_ms"], r["max_ms"])


if __name__ == "__main__":
    unittest.main()
