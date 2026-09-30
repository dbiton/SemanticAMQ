"""The blocklist sweeps label every query by distance at tau, never by the
source it was drawn from (runner._blocklist_cover_rows)."""
import unittest

import numpy as np

import runner
from filters.ball_cover.ball_cover_filter import BallCoverFilter

DIM = 16
TAUS = [0.0, 0.05, 0.1, 0.2, 0.4]
K_FRACTIONS = [0.1, 0.3]
# Columns that depend on the labels and predictions but not on which source a
# query came from.
POOLED = ("fpr", "fnr", "accesses_saved_pct", "neg_share_pct", "n_pos", "n_queries")


def _unit(x: np.ndarray) -> np.ndarray:
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def _workload(seed: int = 0):
    """Reference set B from 80% of 8 tight clusters, held-out queries from the
    rest of them, withheld queries from 8 other clusters, and singletons.

    Planted so labels and sources disagree in both directions: the first
    held-out query is moved 0.3 from its cluster (a negative at tau < 0.3),
    and the first withheld query is a copy of B[0] nudged by 0.01 (a positive
    at every tau >= 0.01)."""
    rng = np.random.default_rng(seed)
    centers = _unit(rng.normal(size=(16, DIM)))
    clusters = [_unit(c + 0.05 * rng.normal(size=(20, DIM))) for c in centers]
    B = np.concatenate([c[:16] for c in clusters[:8]])
    heldout = np.concatenate([c[16:] for c in clusters[:8]])
    withheld = np.concatenate(clusters[8:])
    singleton = _unit(rng.normal(size=(60, DIM)))

    direction = _unit(rng.normal(size=(1, DIM)))[0]
    heldout[0] = heldout[0] + 0.3 * direction
    withheld[0] = B[0] + 0.01 * direction
    return B, heldout, withheld, singleton


def _rows(B, heldout, withheld, singleton):
    return runner._blocklist_cover_rows(
        DIM, "synthetic", 0, K_FRACTIONS, TAUS, B, heldout, withheld, singleton,
        volume_bounds=False)


def _nn(B, Q):
    return np.sqrt(((Q[:, None, :] - B[None, :, :]) ** 2).sum(-1)).min(axis=1)


class TestBlocklistGroundTruth(unittest.TestCase):

    def setUp(self):
        self.B, self.heldout, self.withheld, self.singleton = _workload()
        self.rows = _rows(self.B, self.heldout, self.withheld, self.singleton)

    def test_moving_a_query_between_sources_changes_only_per_source_columns(self):
        # The last withheld query becomes a singleton: same queries, same
        # labels, different strata.
        moved = _rows(self.B, self.heldout, self.withheld[:-1],
                      np.concatenate([self.withheld[-1:], self.singleton]))
        self.assertEqual(len(moved), len(self.rows))
        changed = set()
        for a, b in zip(self.rows, moved):
            for col in POOLED:
                self.assertAlmostEqual(a[col], b[col], places=12, msg=col)
            changed |= {c for c in ("n_withheld", "n_singleton", "n_neg_withheld", "n_neg_singleton")
                        if a[c] != b[c]}
        self.assertTrue({"n_withheld", "n_singleton"} <= changed)

    def test_labels_come_from_distance_not_source(self):
        nn_heldout = _nn(self.B, self.heldout)
        nn_withheld = _nn(self.B, self.withheld)
        nn_all = _nn(self.B, np.concatenate([self.heldout, self.withheld, self.singleton]))
        for row in self.rows:
            tau = row["tau"]
            self.assertEqual(row["n_neg_heldout"], int(np.sum(nn_heldout > tau)))
            self.assertEqual(row["n_neg_withheld"], int(np.sum(nn_withheld > tau)))
            self.assertEqual(row["n_pos"], int(np.sum(nn_all <= tau)))
        at = {r["tau"]: r for r in self.rows if r["k_fraction"] == K_FRACTIONS[0]}
        # The planted held-out query is a negative at 0.1; the planted
        # withheld query a positive.
        self.assertGreater(nn_heldout[0], 0.1)
        self.assertLessEqual(nn_withheld[0], 0.1)
        self.assertGreaterEqual(at[0.1]["n_neg_heldout"], 1)
        self.assertLess(at[0.1]["n_neg_withheld"], len(self.withheld))

    def test_zero_fnr_and_saved_within_ceiling(self):
        for row in self.rows:
            self.assertEqual(row["fnr"], 0.0)
            self.assertLessEqual(row["accesses_saved_pct"], row["neg_share_pct"] + 1e-9)

    def test_predictions_match_the_filters_own_query(self):
        Q = np.concatenate([self.heldout, self.withheld, self.singleton])
        nn = _nn(self.B, Q)
        sizes = (len(self.heldout), len(self.withheld), len(self.singleton))
        for kf in K_FRACTIONS:
            k = min(max(1, round(kf * len(self.B))), len(self.B))
            filt = BallCoverFilter(dim=DIM, max_memory=runner._ball_cover_max_memory(DIM, k))
            filt.construct(self.B)
            for row in (r for r in self.rows if r["k_fraction"] == kf):
                predicted = np.asarray(filt.query(Q, distance_threshold=row["tau"]), dtype=bool)
                self.assertAlmostEqual(row["accesses_saved_pct"], 100.0 * (1.0 - predicted.mean()))
                expected = runner._per_source_rates(predicted, nn <= row["tau"], *sizes)
                for source in runner.QUERY_SOURCES:
                    np.testing.assert_equal(row[f"fpr_{source}"], expected[f"fpr_{source}"])


class TestPerSourceRates(unittest.TestCase):

    def test_restricts_fpr_to_each_sources_negatives(self):
        predicted = [True, False, True, True, False, False]
        actual = [True, False, False, True, False, False]
        out = runner._per_source_rates(predicted, actual, 2, 2, 2)
        self.assertEqual(out["fpr_heldout"], 0.0)       # its one negative is rejected
        self.assertEqual(out["fpr_withheld"], 1.0)      # its one negative is accepted
        self.assertEqual(out["fpr_singleton"], 0.0)
        self.assertEqual((out["n_neg_heldout"], out["n_neg_withheld"], out["n_neg_singleton"]),
                         (1, 1, 2))
        self.assertEqual(out["n_pos"], 2)
        self.assertAlmostEqual(out["neg_share_pct"], 100.0 * 4 / 6)

    def test_source_without_negatives_is_nan(self):
        out = runner._per_source_rates([True, True], [True, True], 2, 0, 0)
        self.assertTrue(np.isnan(out["fpr_heldout"]))
        self.assertTrue(np.isnan(out["fpr_singleton"]))

    def test_sizes_must_partition_the_stream(self):
        with self.assertRaises(ValueError):
            runner._per_source_rates([True, False], [True, False], 1, 0, 0)


if __name__ == "__main__":
    unittest.main()
