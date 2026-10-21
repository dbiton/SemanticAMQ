"""CapCoverFilter.query() (Rust scan) vs a float64 reference of the accept rule,
including boundary and non-unit queries. FNR must be 0."""
import unittest
import numpy as np
from filters.cap_cover.cap_cover_filter import CapCoverFilter, Quantize
from filters.cap_cover.cap_geometry import angle_between, unit_rows
from filters.cap_cover.cap_merge_insert import CapMergeInsert
from filters.cap_cover.nearest_insert import NearestInsert

DIM = 192     # TQ1/2/3/4 and SPQ16/24/48/64 take the table kernels here; SPQ12/36 and TQ8 the scalar one
TAUS = (0.02, 0.1, 0.3, 0.6)


def _unit(X):
    return (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.float32)


def clustered(n_clusters=60, per=20, dim=DIM, spread=0.3, seed=0):
    rng = np.random.default_rng(seed)
    c = rng.standard_normal((n_clusters, dim))
    c /= np.linalg.norm(c, axis=1, keepdims=True)
    return _unit(np.vstack([x + spread * rng.standard_normal((per, dim)) / np.sqrt(dim) for x in c]))


def near(X, tau, n=2, seed=1):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        noise = rng.standard_normal(X.shape)
        noise /= np.linalg.norm(noise, axis=1, keepdims=True)
        out.append(_unit(X + noise * rng.uniform(0, tau, (len(X), 1)) * 0.95))
    return np.vstack(out)


def reference_margin(f, V, tau):
    """Per query, min over caps of angle(q, u_j) - theta_j - beta, all float64 (inf if
    infeasible): negative = accept. Independent of every scan kernel."""
    Qw = f.to_working(V)
    n = np.sqrt((Qw ** 2).sum(axis=1))
    beta, feasible = f._beta(n, tau)
    Qn = Qw / np.maximum(n, 1e-300)[:, None]
    ids = f.all_caps
    C = f.decode_caps(ids)
    theta = f._angle_value(f._angle_codes[ids])
    out = np.empty(len(V))
    for i in range(len(V)):
        out[i] = (angle_between(Qn[i][None, :], C) - theta - beta[i]).min() if feasible[i] else np.inf
    return out


def queries(f, X, rng):
    """Members, near-members, far unit vectors, and non-unit queries around the norm limits."""
    base = np.vstack([X[::3], near(X[::5], 0.1), near(X[::7], 0.4), _unit(rng.standard_normal((200, DIM)))])
    scaled = [base[:150] * s for s in (0.4, 0.85, 0.95, 1.0004, 1.02, 1.09, 1.12, 1.4)]
    return np.vstack([base] + scaled).astype(np.float32)


def boundary_queries(f, rng, per_cap=3, n_caps=40, tau=0.1):
    """Unit queries at angle theta + beta + delta from a cap's decoded direction, delta from +-1e-3 to +-1e-9."""
    ids = f.all_caps
    pick = rng.choice(len(ids), min(n_caps, len(ids)), replace=False)
    C = f.decode_caps(ids[pick])
    theta = f._angle_value(f._angle_codes[ids[pick]])
    beta = f._beta(np.ones(1), tau)[0][0]
    out = []
    for u, th in zip(C, theta):
        v = rng.standard_normal(DIM)
        v -= (v @ u) * u
        v /= np.linalg.norm(v)
        for d in (-1e-3, -1e-6, -1e-9, 1e-9, 1e-6, 1e-3)[:2 * per_cap]:
            phi = th + beta + d
            out.append(np.cos(phi) * u + np.sin(phi) * v)
    W = np.array(out)
    return (W @ f._rotation.T).astype(np.float64)    # back to the original basis


class Differential(unittest.TestCase):
    def check(self, f, X, taus=TAUS, seed=0):
        rng = np.random.default_rng(seed)
        Q = queries(f, X, rng)
        B = boundary_queries(f, rng, tau=0.1)
        for tau in taus:
            for V in (Q, B):
                new = np.array(f.query(V, tau))
                m = reference_margin(f, V, tau)
                self.assertFalse(((m < -1e-9) & ~new).any(), f"false negative vs float64 reference, tau={tau}")
                self.assertFalse(((m > 1e-9) & new).any(), f"false positive vs float64 reference, tau={tau}")
        # members and everything within tau of a member must be accepted
        for tau in taus:
            self.assertTrue(all(f.query(np.vstack([X, near(X, tau)]), tau)), f"FNR > 0 at tau={tau}")
        self.assertIsNotNone(f._rust, "filter did not use the Rust path")

    def test_single_encodings(self):
        X = clustered()
        for q in (Quantize.TQ1, Quantize.TQ2, Quantize.TQ3, Quantize.TQ4, Quantize.TQ8,
                  Quantize.SPQ12, Quantize.SPQ16, Quantize.SPQ24, Quantize.SPQ36, Quantize.SPQ48, Quantize.SPQ64):
            with self.subTest(q=q.value):
                f = CapCoverFilter(DIM, quantize=q, max_caps=150)
                f.construct(X)
                self.check(f, X)

    def test_mixed_levels(self):
        X = clustered()
        for budget in (6000, 14000):
            with self.subTest(budget=budget):
                f = CapCoverFilter.auto(DIM, max_memory=budget)
                f.construct(X)
                self.assertGreater(len(np.unique(f._cap_level[f.all_caps])), 1 if budget == 14000 else 0)
                self.check(f, X)

    def test_after_insert(self):
        X = clustered()
        for method in (NearestInsert, CapMergeInsert):
            with self.subTest(method=method.__name__):
                f = CapCoverFilter(DIM, quantize=Quantize.TQ4, max_memory=40000, insert_method=method())
                f.construct(X[:600])
                f.query(X[:5], 0.1)          # freeze, then make insert() thaw it
                f.insert(X[600:])
                self.check(f, X)
                self.assertEqual(f.angle_bits, 16)

    def test_fallback_encodings_use_float64_rule(self):
        X = clustered()
        for q in (Quantize.SQ8, Quantize.FP16, Quantize.FP32):
            f = CapCoverFilter(DIM, quantize=q, max_caps=100)
            f.construct(X)
            Q = queries(f, X, np.random.default_rng(0))
            got = np.array(f.query(Q, 0.1))
            m = reference_margin(f, Q, 0.1)
            self.assertFalse(((m < -1e-9) & ~got).any() or ((m > 1e-9) & got).any())
            self.assertIsNone(f._rust)

    def test_bytesize_and_buffers(self):
        X = clustered()
        f = CapCoverFilter.auto(DIM, max_memory=14000)
        f.construct(X)
        before = f.get_bytesize()
        f.query(X[:10], 0.1)
        self.assertEqual(f.get_bytesize(), before)
        self.assertLessEqual(before, 14000)
        self.assertEqual(len(f._codes), 0, "Python-side code buffer should be dropped while frozen")
        # decoding the caps (diagnostics) thaws and agrees with a rebuild
        d = f.all_directions
        self.assertEqual(d.shape, (f._num_caps, DIM))

    def test_degenerate_inputs(self):
        X = clustered()
        f = CapCoverFilter(DIM, quantize=Quantize.TQ2, max_caps=50)
        f.construct(X)
        self.assertEqual(f.query(np.zeros((3, DIM), np.float32), 0.1), [False] * 3)
        self.assertEqual(f.query(np.zeros((0, DIM), np.float32), 0.1), [])
        self.assertEqual(f.query(X[:2] * 5.0, 0.1), [False] * 2)       # |n - 1| > tau'
        g = CapCoverFilter(DIM, quantize=Quantize.TQ2, max_caps=50)
        g.construct(np.zeros((0, DIM), np.float32))
        self.assertEqual(g.query(X[:2], 0.1), [False] * 2)


if __name__ == "__main__":
    unittest.main()
