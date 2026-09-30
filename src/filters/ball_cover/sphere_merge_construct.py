import math
import numpy as np
from typing import Optional, Tuple
from filters.filter import Filter
from filters.ball_cover.merge_construct import MergeConstruct
from filters.ball_cover.minimax_construct import MinimaxConstruct
from filters.ball_cover.exact_cover import (exact_cover, nearest_centers, segments, segment_max,
                                            sphere_centers, sphere_radius)


class SphereMergeConstruct(MergeConstruct):
    """Bottom-up merging on an FPR-shaped cost, one global minimax round, and
    centers placed on the data's own sphere.

    Three changes to MergeConstruct, each measured on the blocklist workload
    (main.generate_construct_methods, 5 corpora, tau=0.1) against
    MinimaxConstruct's false-positive rate:

    1. The merge cost is the increase of sum_i m_i (r_i + offset)^exponent,
       m_i = member count, rather than the enclosing ball's volume. That is a
       sum over *points* -- every reference vector pays for the width of the
       ball it lives in -- because queries arrive where the reference set is
       dense: near-duplicates of its members, and hard negatives from the same
       corpus regions. The volume cost weights *balls*, and at dim=384 so
       steeply (r^384) that only the widest one registers, which pushes every
       ball toward the same radius. Where queries crowd one dense region
       (WildChat) that spends the budget exactly backwards: tight balls belong
       there, and wide ones can go where no query ever lands. The exponent
       (8) is far below dim because what a ball captures grows with the
       radius like the data's intrinsic dimension, not the ambient one:
       measured, a ball's hard-negative captures rise ~10x per +0.1 of radius.
       It and the mass weight were swept jointly (p 2-384, beta 0-3); the
       optimum is flat around mass_weight/exponent = 0.1-0.2, and
       exponent-only changes (mass_weight 0) do nothing at all.

    2. At most `merge_fraction` of the merges still needed happen per round.
       MergeConstruct's rounds take every mutual-cheapest pair whenever
       those don't outnumber the merges still needed -- true in early rounds
       -- so two isolated points that are each other's nearest neighbour at
       d~1 get merged in round one, although sequential cheapest-first
       merging would never reach them before k. At k = 20% of |B| that left
       47 balls wider than 0.5 on WikiAnswers; capped, none.

    3. On data that lies on a sphere (unit-normalized embeddings), the final
       centers are moved onto one common sphere by exact_cover.sphere_centers
       (see there for the geometry). A ball only matters through the cap it
       cuts from the data's sphere, where every query also lies, and an MEB
       center inside the sphere widens its tau-expanded cap by ~tau*tan(theta)
       against ~tau*sin(theta) for the best norm. Euclidean cover volume gets
       *worse* under this move while FPR improves.

    Between merging and projection, one round of MinimaxConstruct's
    alternation (nearest-center reassignment of every vector, then
    Badoiu-Clarkson steps toward each cluster's minimum enclosing ball)
    re-centers the merged partition; for unit data the MEB center's direction
    is exactly the minimum enclosing cap's. Further rounds measured to add
    ~0.002 FPR. Merged balls are also re-centered with `recenter_steps`
    Badoiu-Clarkson steps as they form, so merge costs are measured around
    near-MEB centers instead of the enclosing-ball construction's.

    Measured 2026-09-29 by main.generate_construct_methods (5 corpora x 5
    folds), fold-paired against MinimaxConstruct: FPR -0.24 / -0.15 / -0.12
    / -0.10 at k = 5/10/15/20% of |B|, lower in 94 of those 100 rows (worst
    loss +0.01), and lower in all 75 rows at k = 0.5-2%. At 15-20% on
    WikiAnswers, Quora and ChatGPT-paraphrases the two tie: nearly every
    false positive left there is a held-out cluster-mate inside its own
    cluster's ball, a floor no cover of k < #clusters balls gets under.

    Cost, single core: about minimax's at this scale (4.5 s vs 3.9-4.1 s at
    n=10K, k=1000; 0.74-1.0x at n=8000, k=5-20%). It grows faster with n,
    though: the merge phase's candidate search is all-pairs over the live
    balls, O(n^2 * dim) in its first rounds, against minimax's O(n * k * dim).
    Seeding the merge from m >> k Gonzalez cells instead of single points
    (a MergeConstruct option since removed) measured to lose most of the FPR
    gain -- the first, exact nearest-pair merges are what the method needs.

    Radii come from exact_cover's re-search, as in every method here. On
    data that is not equal-norm, step 3 is skipped and centers stay at the
    MEB centers."""

    def __init__(self, exponent: float = 8.0, mass_weight: float = 1.0, merge_fraction: float = 0.25,
                 recenter_steps: int = 8, bc_steps: int = 64, cost_offset: float = 0.1,
                 candidates: int = 16) -> None:
        super().__init__(candidates=candidates)
        self.exponent = exponent
        self.mass_weight = mass_weight
        self.merge_fraction = merge_fraction
        self.recenter_steps = recenter_steps
        self.bc_steps = bc_steps
        self.cost_offset = cost_offset
        self._offset = cost_offset


    def construct_filter(self, filter: Filter, vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = np.ascontiguousarray(vectors, dtype=np.float32)
        n = X.shape[0]
        dim = X.shape[1]
        k = min(filter.max_construct_spheres, n)
        if k <= 0:
            return np.empty((0, dim), dtype=np.float32), np.empty(0, dtype=np.float32)
        if n <= k:
            return X.copy(), np.zeros(n, dtype=np.float32)

        rho = sphere_radius(X)
        # The cost's additive offset is relative to the data's scale, so the
        # merge order doesn't depend on units: rho on a sphere (1 for
        # unit-normalized embeddings, where the offset is the project's own
        # tau -- any value in 0.02-0.3 measured within 0.006 FPR of it), else
        # the RMS distance to the mean.
        scale = rho if rho else float(np.sqrt(np.mean(np.sum((X - X.mean(axis=0)) ** 2, axis=1))))
        self._offset = self.cost_offset * scale

        # Exact duplicates start as one ball, as in MergeConstruct.
        rows = X.view(np.dtype((np.void, X.dtype.itemsize * dim))).ravel()
        _, first, owner = np.unique(rows, return_index=True, return_inverse=True)
        centers = X[first].astype(np.float64)
        radii = np.zeros(len(first), dtype=np.float64)
        owner = owner.ravel().astype(np.int64)

        while len(centers) > k:
            excess = len(centers) - k
            mass = np.bincount(owner, minlength=len(centers)).astype(np.float64)
            max_pairs = min(excess, math.ceil(self.merge_fraction * excess))
            p, q = self._mutual_cheapest_pairs(centers, radii, dim, max_pairs, mass=mass)
            # Where each p lands once _merge compacts the retired q's away.
            alive = np.ones(len(centers), dtype=bool)
            alive[q] = False
            merged = (np.cumsum(alive) - 1)[p]
            centers, radii, owner = self._merge(X, centers, radii, owner, p, q)
            if self.recenter_steps > 0:
                self._recenter(X, centers, radii, owner, merged)

        C = np.ascontiguousarray(centers, dtype=np.float32)
        if rho:
            # Onto the sphere before reassigning, so nearest is nearest by angle.
            C = (rho * C / np.maximum(np.linalg.norm(C, axis=1), 1e-12)[:, None]).astype(np.float32)
        mm = MinimaxConstruct(rounds=1, bc_steps=self.bc_steps)
        d2, labels = nearest_centers(X, C)
        order, starts, counts = segments(labels, len(C))
        C = mm._badoiu_clarkson(X[order], labels[order], C, starts, counts)
        C = mm._reseed_empty(X, C, counts, d2)
        if rho:
            C = sphere_centers(X, C, rho)

        # Radii from the exact re-search, never from the ones tracked above.
        return exact_cover(X, C)


    def _edge_costs(self, lo: np.ndarray, hi: np.ndarray, dist: np.ndarray, radii: np.ndarray,
                    dim: int, mass: Optional[np.ndarray] = None) -> np.ndarray:
        """log of the increase in sum_i m_i^w (r_i + offset)^p that merging
        each edge's two balls would cause, with the merged radius taken from
        the enclosing ball (MergeConstruct._merge's geometry)."""
        merged_r = 0.5 * (dist + radii[lo] + radii[hi])
        merged_r = np.maximum(merged_r, np.maximum(radii[lo], radii[hi]))
        w, p, off = self.mass_weight, self.exponent, self._offset
        log_a = w * np.log(mass[lo]) + p * np.log(radii[lo] + off)
        log_b = w * np.log(mass[hi]) + p * np.log(radii[hi] + off)
        log_c = w * np.log(mass[lo] + mass[hi]) + p * np.log(merged_r + off)
        # c > a + b whenever the merged ball holds any new mass, so the log of
        # the difference is finite; the floor only guards float rounding.
        rest = np.exp(log_a - log_c) + np.exp(log_b - log_c)
        return log_c + np.log(np.maximum(1.0 - rest, 1e-12))


    def _recenter(self, X: np.ndarray, centers: np.ndarray, radii: np.ndarray, owner: np.ndarray,
                  ids: np.ndarray) -> None:
        """Badoiu-Clarkson steps on each just-merged ball's own members, in
        place. MinimaxConstruct._badoiu_clarkson keeps each ball's best center
        seen, so no radius grows."""
        remap = np.full(len(centers), -1, dtype=np.int64)
        remap[ids] = np.arange(len(ids))
        ball = remap[owner]
        rows = np.flatnonzero(ball >= 0)
        rows = rows[np.argsort(ball[rows], kind="stable")]
        label = ball[rows]
        _, starts, counts = segments(label, len(ids))
        mm = MinimaxConstruct(rounds=1, bc_steps=self.recenter_steps)
        new = mm._badoiu_clarkson(X[rows], label, centers[ids].astype(np.float32), starts, counts)
        delta = X[rows].astype(np.float64) - new.astype(np.float64)[label]
        r = segment_max(np.sqrt(np.einsum("ij,ij->i", delta, delta)), starts, counts)
        better = r < radii[ids]
        centers[ids[better]] = new[better]
        radii[ids[better]] = r[better]


    def get_bytesize(self) -> int:
        return 0
