import numpy as np
from typing import List, Tuple
from filters.filter import Filter
from filters.ball_cover.ball_interfaces import ConstructInterface
from filters.ball_cover.exact_cover import exact_cover

# Rows per distance block, as in MergeConstruct._merge: the root ball holds
# all of X, and a plain X[rows] - center there would materialize a second
# copy of the dataset.
_BLOCK = 16_384


class SplitConstruct(ConstructInterface):
    """Top-down bisection of the widest ball (LBG's split step, on the cover
    objective). Starts from a single ball over the whole set and splits,
    k-1 times, whichever ball currently has the largest radius. A split
    seeds two centers at that ball's farthest member from its own center and
    the member farthest from *that* -- the usual two-point diameter
    approximation -- then runs a few 2-means rounds over the ball's own
    members to pull both centers off the surface and in among their halves.

    Greedy on the max radius, which is nearly all of what the cover volume
    is: at dim=384 a ball 10% wider than another contributes 1.1^384 ~ 1e16
    times the volume, so sum_i r_i^dim is set by the widest few balls and
    almost nothing else. The other methods reach that objective only
    indirectly -- Gonzalez peels off one far point at a time, k-means
    minimizes squared error and so never targets the widest ball, and
    bottom-up merging can undo an unlucky merge only by making some ball
    wider still. Splitting worst-first is what keeps the centroid step from
    inheriting k-means' outlier problem: a ball holding an outlier is wide,
    so it is exactly what gets split next, until the outlier is alone.

    The 2-means rounds keep whichever round gave the pair the smallest
    max(r_a, r_b), with the raw seed split as round 0, so a split is never
    worse on the pair's own radius than the diameter seeds alone -- the same
    keep-the-best-round guard MinimaxConstruct uses.

    Each split touches only the members of the one ball being split, so the
    whole build costs what a k-d tree build costs -- O(n*dim) per level of
    the split tree -- with no per-round pass over all of X.

    Splitting alone decides the centers; the radii come from exact_cover's
    re-search, as in every other method here."""

    def __init__(self, lloyd_rounds: int = 3) -> None:
        self.lloyd_rounds = lloyd_rounds


    def construct_filter(self, filter: Filter, vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = np.ascontiguousarray(vectors, dtype=np.float32)
        n = X.shape[0]
        dim = X.shape[1]
        k = min(filter.max_construct_spheres, n)
        if k <= 0:
            return np.empty((0, dim), dtype=np.float32), np.empty(0, dtype=np.float32)
        if n <= k:
            return X.copy(), np.zeros(n, dtype=np.float32)

        # The root's center is the mean only so that the first seed is the
        # point farthest from the bulk.
        members: List[np.ndarray] = [np.arange(n, dtype=np.int64)]
        centers: List[np.ndarray] = [X.mean(axis=0, dtype=np.float64).astype(np.float32)]
        radii = np.zeros(k, dtype=np.float32)
        # Each ball's farthest member, kept from the split that created it,
        # so reaching for a split's first seed costs no pass of its own.
        farthest = np.zeros(k, dtype=np.int64)
        d = self._distances(X, members[0], centers[0])
        radii[0], farthest[0] = d.max(), members[0][d.argmax()]

        m = 1
        while m < k:
            b = int(radii[:m].argmax())
            if radii[b] <= 0.0:
                break  # every ball is a single distinct point

            rows = members[b]
            seed = farthest[b]
            d = self._distances(X, rows, X[seed])
            if d.max() <= 0.0:
                # Every member is the same point -- a ball of exact
                # duplicates, which no split can improve. It has a radius at
                # all only because a centroid center sits off its members,
                # so put the center on them and take the ball out of
                # contention at radius 0.
                centers[b], radii[b] = X[seed], 0.0
                continue
            c_a, c_b = X[seed], X[rows[d.argmax()]]

            best = None
            for round_ in range(self.lloyd_rounds + 1):
                d_a = self._distances(X, rows, c_a)
                d_b = self._distances(X, rows, c_b)
                take_a = d_a <= d_b
                n_a = int(take_a.sum())
                # Round 0's seeds each keep themselves (own distance 0) and
                # lose the other, so it always lands here and best is always
                # set. A later centroid pair can starve one side instead,
                # and then the rounds before it are what stands.
                if not 0 < n_a < len(rows):
                    break
                pair_r = max(d_a[take_a].max(), d_b[~take_a].max())
                if best is None or pair_r < best[0]:
                    best = (pair_r, take_a, d_a, d_b, c_a, c_b)
                if round_ == self.lloyd_rounds:
                    break
                c_a = self._centroid(X, rows[take_a])
                c_b = self._centroid(X, rows[~take_a])

            _, take_a, d_a, d_b, c_a, c_b = best
            rows_a, rows_b = rows[take_a], rows[~take_a]
            d_a, d_b = d_a[take_a], d_b[~take_a]

            members[b], centers[b] = rows_a, c_a
            radii[b], farthest[b] = d_a.max(), rows_a[d_a.argmax()]
            members.append(rows_b)
            centers.append(c_b)
            radii[m], farthest[m] = d_b.max(), rows_b[d_b.argmax()]
            m += 1

        # Radii from the exact re-search, never from the ones tracked above.
        # The re-search also reassigns every vector globally, which can only
        # move it to a nearer center, so no radius grows against what the
        # split tree tracked.
        return exact_cover(X, np.vstack(centers))


    @staticmethod
    def _centroid(X: np.ndarray, rows: np.ndarray) -> np.ndarray:
        """Mean of X[rows], accumulated in float64 and blocked for the same
        reason _distances is."""
        acc = np.zeros(X.shape[1], dtype=np.float64)
        for s in range(0, len(rows), _BLOCK):
            acc += X[rows[s:s + _BLOCK]].sum(axis=0, dtype=np.float64)
        return (acc / len(rows)).astype(np.float32)


    @staticmethod
    def _distances(X: np.ndarray, rows: np.ndarray, center: np.ndarray) -> np.ndarray:
        """||X[rows] - center||, blocked. float32 throughout: these distances
        only choose seeds and rank balls by width, never a returned radius."""
        out = np.empty(len(rows), dtype=np.float32)
        for s in range(0, len(rows), _BLOCK):
            block = X[rows[s:s + _BLOCK]] - center
            out[s:s + _BLOCK] = np.einsum("ij,ij->i", block, block)
        np.maximum(out, 0.0, out=out)
        return np.sqrt(out, out=out)


    def get_bytesize(self) -> int:
        return 0
