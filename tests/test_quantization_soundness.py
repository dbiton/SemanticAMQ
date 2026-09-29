import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'src'))
import unittest

import numpy as np

from src.filters.ball_cover.ball_cover_filter import BallCoverFilter, Quantize
from src.filters.ball_cover.min_heap_insert import MinHeapInsert
from src.filters.ball_cover.nearest_insert import NearestInsert
from src.filters.ball_cover.least_inflation_insert import LeastInflationInsert
from src.filters.ball_cover.exhaustive_merge_insert import ExhaustiveMergeInsert
from src.filters.ball_cover.doubling_insert import DoublingInsert
from src.filters.ball_cover.pooled_insert import PooledInsert
from src.filters.quantized_vector_filter import (
    QuantizedVectorFilter, ErrorBudgetMode, QuantizationLevel, _footprint_bytes,
)

DIM = 64
EPS = BallCoverFilter._COVERAGE_EPSILON

# Every encoding whose zero-false-negative property is a proof, i.e. whose
# query() must score exactly ||q - decode(c)||. RaBitQ scores through its own
# estimator instead (see the CAVEAT on Quantize.RABITQ), so it is left out.
EXACT_ENCODINGS = [q for q in Quantize if q not in (Quantize.NONE, Quantize.RABITQ)]
# Lossy encodings the streaming tests run on, one per codec family, plus TQ5.
STREAMING_ENCODINGS = [Quantize.FP32, Quantize.SQ4, Quantize.TQ3, Quantize.TQ5, Quantize.PQ4_R1]


def _strategies():
    return {
        'nearest': NearestInsert, 'least_inflation': LeastInflationInsert,
        'merge': ExhaustiveMergeInsert, 'min_heap': MinHeapInsert,
        'doubling': DoublingInsert, 'pooled': PooledInsert,
    }


def _clustered(n: int, dim: int = DIM, n_clusters: int = 40, spread: float = 0.25, seed: int = 0) -> np.ndarray:
    """Unit-norm vectors around n_clusters anchors -- tight enough that balls
    hold several points, the regime every radius argument here is about."""
    rng = np.random.default_rng(seed)
    anchors = rng.standard_normal((n_clusters, dim))
    anchors /= np.linalg.norm(anchors, axis=1, keepdims=True)
    x = anchors[rng.integers(0, n_clusters, n)] + spread / np.sqrt(dim) * rng.standard_normal((n, dim))
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def _exact_margin(f: BallCoverFilter, queries: np.ndarray) -> np.ndarray:
    """min over balls of ||q - decode(c)|| - r, from the decoded centers
    rather than through the index's search()."""
    centers, radii = f.all_centers, f.all_radiuses
    d = np.linalg.norm(queries[:, None, :] - centers[None, :, :], axis=2)
    return (d - radii[None, :]).min(axis=1)


class TestQueryIsExactADC(unittest.TestCase):
    """query() must accept exactly the queries the decoded geometry says it
    should. faiss's plain TurboQuant codec (TQ5's QT_5bit_tq) folds each
    center's stored norm into search(), which over-estimated distances and
    dropped true duplicates until query() subtracted that offset back out."""

    def test_query_matches_decoded_geometry(self):
        X = _clustered(3000)
        Q = np.vstack([_clustered(300, seed=1), X[:300]])
        for quantize in EXACT_ENCODINGS:
            with self.subTest(quantize=quantize.value):
                f = BallCoverFilter(dim=DIM, max_spheres=400, quantize=quantize)
                f.construct(X)
                margin = _exact_margin(f, Q)
                for tau in (0.0, 0.1, 0.3):
                    got = np.asarray(f.query(Q, distance_threshold=tau))
                    must_accept = margin <= tau
                    must_reject = margin > tau + 2 * EPS
                    self.assertTrue(got[must_accept].all(),
                                    f"{quantize.value} tau={tau}: rejected {np.sum(~got[must_accept])} covered queries")
                    self.assertFalse(got[must_reject].any(),
                                     f"{quantize.value} tau={tau}: accepted {np.sum(got[must_reject])} uncovered queries")

    def test_tq5_search_needs_the_offset_removed(self):
        """Guards the premise of the fix: if a faiss upgrade ever makes
        QT_5bit_tq exact, _Encoding.norm_offset should come off TQ5."""
        f = BallCoverFilter(dim=DIM, max_spheres=400, quantize=Quantize.TQ5)
        f.construct(_clustered(3000))
        offsets = f._search_offsets()[f.all_spheres]
        self.assertGreater(np.abs(offsets).max(), 1e-3)


class TestStreamingInsertSoundness(unittest.TestCase):
    """Zero false negatives, fp16-settled radii and no radius creep on lossy
    encodings, for every insertion strategy."""

    def _filter(self, quantize: Quantize, name: str) -> BallCoverFilter:
        return BallCoverFilter(dim=DIM, max_spheres=128, quantize=quantize,
                               insert_method=_strategies()[name](),
                               insert_memory_percent=20.0 if name == 'pooled' else 0.0)

    def test_inserted_vectors_are_recalled(self):
        X = _clustered(2400)
        half = len(X) // 2
        for quantize in STREAMING_ENCODINGS:
            for name in _strategies():
                for batch in (len(X) - half, 50):  # one insert() call, then many small ones
                    with self.subTest(quantize=quantize.value, strategy=name, batch=batch):
                        f = self._filter(quantize, name)
                        f.construct(X[:half])
                        for i in range(half, len(X), batch):
                            f.insert(X[i:i + batch])
                        recalled = np.asarray(f.query(X, distance_threshold=0.0))
                        self.assertTrue(recalled.all(),
                                        f"{np.sum(~recalled)} of {len(X)} inserted vectors not recalled")
                        radii = f.all_radiuses
                        np.testing.assert_array_equal(
                            BallCoverFilter._fp16_ceil(radii), radii,
                            err_msg="radii are not on the fp16 grid bytes_per_ball() charges for")

    def test_covered_insert_does_not_grow_radii(self):
        X = _clustered(2000)
        for quantize in STREAMING_ENCODINGS:
            for name in _strategies():
                with self.subTest(quantize=quantize.value, strategy=name):
                    f = self._filter(quantize, name)
                    f.construct(X)
                    before = dict(zip(f.all_spheres.tolist(), f.all_radiuses.tolist()))
                    # A decoded center sits at distance 0 from its own ball,
                    # so every strategy's own coverage test sees it as
                    # covered -- the insert is a no-op by construction.
                    covered = f.all_centers[:1].copy()
                    for _ in range(30):
                        f.insert(covered)
                    after = dict(zip(f.all_spheres.tolist(), f.all_radiuses.tolist()))
                    self.assertEqual(sorted(before.values()), sorted(after.values()))


class TestQuantizedVectorFilterTQ5(unittest.TestCase):
    """QuantizedVectorFilter shares QT_5bit_tq with BallCoverFilter's TQ5 and
    relied on the same (false) exact-ADC assumption."""

    def test_tq5_level_has_no_false_negatives(self):
        dim = 128  # TQ5 (80 B + 8 B norm) undercuts SQ6 (96 B) only from here up
        X = _clustered(1500, dim=dim)
        mode = ErrorBudgetMode.PER_VECTOR
        budget = _footprint_bytes(QuantizationLevel.TQ5, len(X), dim, mode, 8)
        f = QuantizedVectorFilter(dim=dim, distance_threshold=0.1, max_memory=budget, error_budget_mode=mode)
        f.construct(X)
        self.assertEqual(f.level, QuantizationLevel.TQ5)
        # Near-duplicates strictly inside tau: QVF has no coverage epsilon, so
        # a query exactly at tau + eps_i is decided by float noise on every
        # level -- a separate, level-independent issue this test stays clear of.
        tau = 0.1
        rng = np.random.default_rng(3)
        step = rng.standard_normal(X.shape)
        step *= rng.uniform(0.2, 0.9, (len(X), 1)) * tau / np.linalg.norm(step, axis=1, keepdims=True)
        near = (X + step).astype(np.float32)
        got = np.asarray(f.query(near, tau=tau))
        self.assertTrue(got.all(), f"{np.sum(~got)} of {len(near)} near-duplicates rejected")


if __name__ == '__main__':
    unittest.main()
