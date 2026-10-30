"""SoftAnnealMinimax: minimax's cover, annealed; the cover stays valid and the construct-set
objective never gets worse than minimax's."""
import unittest
from unittest.mock import patch

import numpy as np

from filters.cap_cover.cap_cover_filter import CapCoverFilter, Quantize
from filters.cap_cover.cap_geometry import log_cap_measure
from filters.cap_cover.soft_anneal_construct import SoftAnneal, SoftAnnealMinimax, hard_eval
from filters.cap_cover.minimax_construct import MinimaxConstruct
import runner


def _unit(a):
    return a / np.linalg.norm(a, axis=1, keepdims=True)


class TestSoftAnneal(unittest.TestCase):
    DIM = 384

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        centers = _unit(rng.normal(size=(25, cls.DIM)))
        cls.X = _unit(centers[rng.integers(0, 25, 700)] + 0.05 * rng.normal(size=(700, cls.DIM))).astype(np.float32)
        near = cls.X[rng.integers(0, 700, 100)] + 0.004 * rng.normal(size=(100, cls.DIM))   # within tau of an indexed vector
        cls.Q = _unit(np.vstack([near, rng.normal(size=(60, cls.DIM))])).astype(np.float32)

    def test_registered(self):
        self.assertIsInstance(runner.CAP_CONSTRUCT_METHODS["soft_anneal"](), SoftAnnealMinimax)

    def test_valid_cover_and_objective_not_worse(self):
        k = 60
        X = self.X.astype(np.float64)
        filt = CapCoverFilter(self.DIM, max_caps=k)
        base_U, _ = MinimaxConstruct().construct_filter(filt, X)
        from filters.cap_cover.cap_geometry import labelled_cap_cover
        U, theta, _, _ = labelled_cap_cover(X, SoftAnneal(iters=40).refine(X, base_U))
        self.assertLessEqual(len(U), k)
        # every vector lies in some cap, within its angle
        ang = np.arccos(np.clip(X @ U.T, -1, 1))
        self.assertTrue(np.all((ang <= theta[None, :] + 1e-6).any(axis=1)))
        j_base = hard_eval(X, base_U, self.DIM)[0]
        j_new = hard_eval(X, U, self.DIM)[0]
        self.assertLessEqual(j_new, j_base + 1e-9)

    def test_filter_has_no_false_negatives(self):
        filt = CapCoverFilter.auto(self.DIM, max_memory=runner._cover_max_memory(self.DIM, 70),
                                   construct_method=SoftAnnealMinimax(anneal=SoftAnneal(iters=30)))
        filt.construct(self.X)
        for cosine_distance in (0.005, 0.02, 0.05):
            with self.subTest(cosine_distance=cosine_distance):
                tau = runner.tau_of_cosine_distance(cosine_distance)
                actual = (2.0 - 2.0 * (self.Q @ self.X.T).max(axis=1)) <= tau * tau
                accepted = np.asarray(filt.query(self.Q, distance_threshold=tau), dtype=bool)
                self.assertTrue(actual.any())
                self.assertTrue(np.all(accepted[actual]))

    def test_refinement_does_not_use_query_expansion(self):
        X = self.X.astype(np.float64)
        filt = CapCoverFilter(self.DIM, max_caps=60)
        def unexpected_expansion(*args, **kwargs):
            self.fail("soft anneal must not consult the query expansion")
        filt._beta = unexpected_expansion
        filt.construct_method = SoftAnnealMinimax(anneal=SoftAnneal(iters=10))
        filt.construct(X)
        self.assertTrue(np.isfinite(filt.all_angles).all())

    def test_annealing_waits_until_encoding_and_runs_through_timing_wrapper(self):
        X = self.X.astype(np.float64)
        filt = CapCoverFilter(self.DIM, max_caps=60)
        method = SoftAnnealMinimax(anneal=SoftAnneal(iters=30))
        (k, cover), = list(method.iter_candidates(filt, X, [60]))
        base_U, _ = MinimaxConstruct().construct_filter(filt, X)
        self.assertEqual(len(cover.directions), len(base_U))
        with patch.object(method.anneal, 'refine', wraps=method.anneal.refine) as refine:
            method.finish(filt, X, cover)
            refine.assert_not_called()
            filt.construct_method = runner._TimedCapConstruct(method)
            filt.construct(X)
            refine.assert_called_once()

    def test_stored_volume_not_worse_after_quantization_and_reassignment(self):
        # Compare complete stored filters, including the narrowed 8-bit angle grid.
        X = self.X[:240]
        for encoding in (Quantize.FP32, Quantize.TQ2, Quantize.TQ4, Quantize.SPQ12):
            with self.subTest(encoding=encoding):
                base = CapCoverFilter(self.DIM, max_caps=24, quantize=encoding,
                                      angle_bits=8, construct_method=MinimaxConstruct())
                refined = CapCoverFilter(self.DIM, max_caps=24, quantize=encoding,
                                         angle_bits=8, construct_method=SoftAnnealMinimax(
                                             anneal=SoftAnneal(iters=10, eval_every=5)))
                base.construct(X)
                refined.construct(X)
                score = lambda f: float(np.logaddexp.reduce(log_cap_measure(f.all_angles, self.DIM)))
                self.assertLessEqual(score(refined), score(base) + 1e-9)
                self.assertEqual(base.get_bytesize(), refined.get_bytesize())
                self.assertTrue(all(refined.query(X, distance_threshold=0.0)))

    def test_mixed_encoding_refinement_keeps_budget_and_coverage(self):
        budget = runner._cover_max_memory(self.DIM, 24)
        base = CapCoverFilter.auto(self.DIM, max_memory=budget,
                                  candidates=[Quantize.TQ2, Quantize.TQ4], mixed=True,
                                  construct_method=MinimaxConstruct())
        filt = CapCoverFilter.auto(self.DIM, max_memory=budget,
                                  candidates=[Quantize.TQ2, Quantize.TQ4], mixed=True,
                                  construct_method=SoftAnnealMinimax(anneal=SoftAnneal(iters=10)))
        base.construct(self.X)
        filt.construct(self.X)
        self.assertLessEqual(filt.get_bytesize(), budget)
        self.assertTrue(all(filt.query(self.X, distance_threshold=0.0)))
        self.assertLessEqual(
            np.logaddexp.reduce(log_cap_measure(filt.all_angles, self.DIM)),
            np.logaddexp.reduce(log_cap_measure(base.all_angles, self.DIM)) + 1e-9)

    def test_reassignment_is_used_for_next_code_fit_and_worse_candidates_rejected(self):
        class ProposeBadDirections:
            def refine(self, X, U, objective, project):
                self.decoded = project(-U)
                objective(-U)
                return -U

        proposal = ProposeBadDirections()
        method = SoftAnnealMinimax(anneal=proposal)
        filt = CapCoverFilter(self.DIM, max_caps=24, quantize=Quantize.TQ2)
        X = filt._prepare_members(self.X[:240])
        U, theta = MinimaxConstruct().construct_filter(filt, X)
        _, _, home, _ = hard_eval(X, U, self.DIM)
        levels = np.zeros(len(U), dtype=np.int64)
        filt._train_quantizers(U)
        initial = filt._encode_cover(X, U, theta, home, levels)
        inputs, outputs = [], []
        original = filt._encode_cover
        def record(X, U, theta, owner, levels):
            inputs.append(owner.copy())
            result = original(X, U, theta, owner, levels)
            outputs.append(result)
            return result
        with patch.object(filt, '_encode_cover', side_effect=record):
            result = method.refine_encoded(filt, X, initial, levels)
        self.assertEqual(len(outputs), 2)
        np.testing.assert_array_equal(inputs[1], outputs[0].owner)
        self.assertEqual(proposal.decoded.shape, U.shape)
        # Opposite directions enclose almost the whole sphere: retain the baseline.
        self.assertIs(result, initial)


if __name__ == "__main__":
    unittest.main()
