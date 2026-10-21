"""The blocklist sweeps label every query by distance at tau, never by the
similarity band it falls in (runner._blocklist_cover_rows)."""
import unittest

import numpy as np

import runner
from filters.cap_cover.cap_cover_filter import CapCoverFilter

DIM = 16
TAUS = [0.0, 0.05, 0.1, 0.2, 0.4]
K_FRACTIONS = [0.1, 0.3]
BANDS = [name for name, _ in runner.SIMILARITY_BANDS]


def _unit(x: np.ndarray) -> np.ndarray:
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def _workload(seed: int = 0):
    """Reference set B from 80% of 8 tight clusters; queries are the rest of
    those clusters (close to B), 8 other clusters (further), and unit noise.

    The first query is moved 0.3 from its cluster, so it is a negative at
    tau < 0.3 although it is still close to B."""
    rng = np.random.default_rng(seed)
    centers = _unit(rng.normal(size=(16, DIM)))
    clusters = [_unit(c + 0.05 * rng.normal(size=(20, DIM))) for c in centers]
    B = np.concatenate([c[:16] for c in clusters[:8]])
    Q = np.concatenate([np.concatenate([c[16:] for c in clusters[:8]]),
                        np.concatenate(clusters[8:]),
                        _unit(rng.normal(size=(60, DIM)))])
    Q[0] = Q[0] + 0.3 * _unit(rng.normal(size=(1, DIM)))[0]
    return B, Q


def _nn(B, Q):
    return np.sqrt(((Q[:, None, :] - B[None, :, :]) ** 2).sum(-1)).min(axis=1)


class TestBlocklistGroundTruth(unittest.TestCase):

    def setUp(self):
        self.B, self.Q = _workload()
        self.rows = runner._blocklist_cover_rows(
            DIM, "synthetic", 0, K_FRACTIONS, TAUS, self.B, self.Q, volume_bounds=False)
        self.nn = _nn(self.B, self.Q)

    def test_labels_and_bands_come_from_distance(self):
        cos = 1.0 - self.nn ** 2 / 2.0
        edges = {"close": (0.8, np.inf), "related": (0.6, 0.8), "unrelated": (-np.inf, 0.6)}
        for row in self.rows:
            tau = row["tau"]
            self.assertEqual(row["n_pos"], int(np.sum(self.nn <= tau)))
            for band, (lo, hi) in edges.items():
                in_band = (cos >= lo) & (cos < hi)
                self.assertEqual(row[f"n_neg_{band}"], int(np.sum(in_band & (self.nn > tau))), band)
        # The planted query is a negative at 0.1 yet lands in a populated band.
        self.assertGreater(self.nn[0], 0.1)
        self.assertGreaterEqual(sum(r[f"n_neg_{b}"] for r in self.rows if r["tau"] == 0.1
                                    for b in BANDS) // len(K_FRACTIONS), 1)

    def test_zero_fnr_and_saved_within_ceiling(self):
        for row in self.rows:
            self.assertEqual(row["fnr"], 0.0)
            self.assertLessEqual(row["accesses_saved_pct"], row["neg_share_pct"] + 1e-9)

    def test_predictions_match_the_filters_own_query(self):
        # _blocklist_cover_rows builds its cover via runner._cap_cover, i.e.
        # CapCoverFilter.auto() at the same equal-bytes budget -- this rebuilds
        # exactly that cover to cross-check row predictions against.
        #
        # Tolerance, not exact equality: surface_angle() (float64) and query()
        # (the Rust scan) decide the same rule, but a handful of queries sitting
        # at a cap's boundary may differ at small dimensions (DIM=16 here;
        # not seen at DIM=384). The zero-FN invariant is checked against
        # ground truth independently by _blocklist_cover_rows.
        #
        # The rows apply the angle rule alone (surface_angle ignores a query's
        # norm), while query() also rejects a query whose norm is more than
        # tau' from 1; the planted query is off the sphere, so compare on
        # directions.
        Qdir = _unit(self.Q)
        n_q = len(self.Q)
        for kf in K_FRACTIONS:
            k = min(max(1, round(kf * len(self.B))), len(self.B))
            filt = runner._cap_cover(DIM, k)
            filt.construct(self.B)
            for row in (r for r in self.rows if r["k_fraction"] == kf):
                predicted = np.asarray(filt.query(Qdir, distance_threshold=row["tau"]), dtype=bool)
                self.assertAlmostEqual(row["accesses_saved_pct"], 100.0 * (1.0 - predicted.mean()),
                                       delta=100.0 / n_q + 1e-9)
                expected = runner._per_band_rates(predicted, self.nn <= row["tau"],
                                                  1.0 - self.nn ** 2 / 2.0)
                for band in BANDS:
                    np.testing.assert_allclose(row[f"fpr_{band}"], expected[f"fpr_{band}"],
                                               atol=0.05, equal_nan=True)


class TestPerBandRates(unittest.TestCase):

    def test_restricts_fpr_to_each_bands_negatives(self):
        predicted = [True, False, True, True, False, False]
        actual = [True, False, False, True, False, False]
        nn_cos = [0.95, 0.9, 0.7, 0.65, 0.3, 0.1]       # close, close, related, related, unrelated x2
        out = runner._per_band_rates(predicted, actual, nn_cos)
        self.assertEqual(out["fpr_close"], 0.0)         # its one negative is rejected
        self.assertEqual(out["fpr_related"], 1.0)       # its one negative is accepted
        self.assertEqual(out["fpr_unrelated"], 0.0)
        self.assertEqual((out["n_neg_close"], out["n_neg_related"], out["n_neg_unrelated"]), (1, 1, 2))
        self.assertEqual(out["n_pos"], 2)
        self.assertAlmostEqual(out["neg_share_pct"], 100.0 * 4 / 6)

    def test_band_edges_are_lower_inclusive(self):
        out = runner._per_band_rates([True, True], [False, False], [0.8, 0.6])
        self.assertEqual((out["n_neg_close"], out["n_neg_related"], out["n_neg_unrelated"]), (1, 1, 0))

    def test_band_without_negatives_is_nan(self):
        out = runner._per_band_rates([True, True], [True, True], [0.9, 0.1])
        self.assertTrue(np.isnan(out["fpr_close"]))
        self.assertTrue(np.isnan(out["fpr_unrelated"]))

    def test_lengths_must_match(self):
        with self.assertRaises(ValueError):
            runner._per_band_rates([True, False], [True, False], [0.9])

    def test_nn_cosine(self):
        B = np.eye(3, dtype=np.float32)
        Q = _unit(np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]))
        np.testing.assert_allclose(runner._nn_cosine(B, Q), [np.sqrt(0.5), 1.0], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
