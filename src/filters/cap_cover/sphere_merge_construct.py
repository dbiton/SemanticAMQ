import math
import numpy as np
from typing import Iterator, List, Optional, Tuple
from filters.filter import Filter
from filters.cap_cover.cap_interfaces import ConstructInterface, Cover
from filters.cap_cover.minimax_construct import badoiu_clarkson, reseed_empty
from filters.cap_cover.cap_geometry import (KNN_EXACT_MAX, angle_between, knn_directions, labelled_cap_cover,
                                            log_cap_measure, min_enclosing_cap, nearest_directions,
                                            nearest_in_candidates,
                                            pair_dots, remap_neighbors, segments, segment_max, unit_rows)


class SphereMergeConstruct(ConstructInterface):
    """Ball-Cover's SphereMergeConstruct on caps: bottom-up merging of caps
    by the accept region they add, then global minimax refinement.

    Every distinct vector starts as a zero-angle cap. In rounds, every cap
    is scored against its `candidates` nearest directions, and the pairs that
    are each other's cheapest merge are merged -- at most `merge_fraction` of
    the merges still needed per round. A merge costs the increase of
    sum_j sigma_p(theta_j + t): sigma_p(psi) the share of S^p a cap of angle
    psi covers (cost="volume"; p = `exponent`, default dim - 1, the sphere
    queries live on; t = cost_offset ~ the accept rule's expansion at
    tau = 0.1), so each merge adds the least accept region it can, whatever
    the queries. At p = dim - 1 the widest caps dominate the sum and the
    cost ranks pairs by their merged angle; p = 32 and 128 merged exactly as
    383 at |B| = 8000. Optional member-count weights m_j^w (`mass_weight`)
    and the small-angle form (theta + t)^p (cost="power") give the earlier
    cost, sum_i m_i (theta_i + 0.1)^8, calibrated on how Ball-Cover's balls
    capture queries; merge_fraction 0.05-0.25 and 1-3 refine rounds
    measured flat under it.

    Measured (src/compare_cap_cover.py, |B| = 8000, 5 corpora x 5 folds,
    CapCoverFilter.auto, results/cap_cover/v1_*.csv): dropping the weights
    (cap_auto0_proxy) moves FPR by -0.009..+0.000 at every k = 0.5-20% of
    |B|; the volume cost on top (cap_auto_vproxy) by -0.010..+0.001, with
    log10 accept volume -0.1..-5.4 lower against the weighted cost. The
    volume on S^8 (exponent=8) with least-volume selection: FPR -0.035 /
    -0.020 / -0.010 / -0.032 against p = 383 at k = 0.5/1/2/5%, at log10
    volume 2.5 / 2.4 / 1.4 / 0.9 higher; within +-0.0003 FPR and 0.2-0.5
    lower volume at 10-20%. The default stays the sphere's own measure:
    choosing p by FPR would tune the objective on queries.

    Exact minimum enclosing caps (`exact_caps`: the widest share of the
    caps with two or more members re-centered by
    cap_geometry.min_enclosing_cap in finish()) measured flat: the
    Badoiu-Clarkson caps are 0.004-0.018 rad looser on average on the 40
    widest (up to 0.14 on CC-News at k = 5%), but on CC-News fold 0 at
    k = 1/5% FPR moved -0.003..-0.007 and log10 volume +-0.08: the stored
    angle around the decoded direction is set by the encoding error.

    A merged cap starts as the smallest cap enclosing its two parts and is
    then re-centered on its own members with `recenter_steps`
    Badoiu-Clarkson steps; its angle is always re-measured from the members.
    After merging, `refine_rounds` rounds of nearest-direction reassignment
    and `bc_steps` Badoiu-Clarkson steps re-center the whole partition
    (finish()).

    Candidate neighbours are searched for only when the cap count has
    fallen below `refresh` of what it was at the last search; in between,
    each cap keeps its list, and a merged cap takes the nearest `candidates`
    of its two parts' lists. Merges are local, so the lists stay good, and
    the search runs a handful of times instead of once per round. It is
    exact up to `knn_exact_max` caps and probes `knn_probes` k-means cells
    per cap above (knn_directions; at 80K distinct vectors, 24 probes find
    the nearest neighbour 99.99% of the time and 97% of the 16 nearest, at
    a third of an exact search's cost); the inner product of every listed
    pair is cached and refreshed only where a merge moved one of its caps,
    so a round costs O(caps * m), not O(caps * m * d). A round merges at
    least `min_batch` of the caps (while that many merges are still needed),
    which keeps the last rounds before each candidate k from rescoring
    every edge to merge a handful of pairs.

    Above `exact_pairs` members x caps, finish() reassigns each member among
    its cap's own neighbour list instead of all caps, and hands the lists on
    (Cover.neighbors) so the filter's own searches stay local too: an
    exact search over 80K members and 60K caps is ~1.8e12 multiply-adds.

    Measured at |B| = 8000 (src/compare_cap_cover.py, 5 corpora x 5 folds,
    CapCoverFilter.auto): restricting the searches to neighbour lists moves
    FPR by +0.0000/-0.0004 at k = 0.5/1% of |B|; cell-probed lists above
    20% of |B| (80K's share of approximate refreshes) by +0.0020/-0.0005 at
    12 probes and +0.0006/-0.0003 at 24; min_batch 0.01 by <= 0.0001 at
    every k = 0.5-20%.

    Merging measured no better (within +-0.008 FPR) with a cost that sees
    caps as the encoder will widen them -- theta' = sqrt(theta^2 + delta^2),
    or the in-data reach CapCoverFilter selects encodings by, both with
    delta the encoding's angular error -- so the cost uses the unencoded
    angles. Ball-Cover's units (the caps' Euclidean MEB radii sin(theta),
    cost_space="sine") measured +0.018 FPR at k = 0.5% of |B| in
    CapCoverFilter.auto() (24 of 25 cells worse), flat above; re-searching
    the neighbour lists every round (refresh = 1) no better (0.092 vs 0.088
    on WildChat fold 1's exact cover at 1063 caps). Ball-Cover's own
    sphere_merge cover is about as good overall (fold-paired, exact
    directions, 600 and 1600 caps: 12 wins / 13 losses), better on
    WildChat (0.19 vs 0.21 at 600)."""

    def __init__(self, exponent: Optional[float] = None, mass_weight: float = 0.0, merge_fraction: float = 0.25,
                 recenter_steps: int = 8, bc_steps: int = 64, cost_offset: float = 0.1,
                 candidates: int = 16, refine_rounds: int = 1, refresh: float = 0.7,
                 min_batch: float = 0.01, knn_exact_max: int = KNN_EXACT_MAX, knn_probes: int = 24,
                 exact_pairs: int = 1 << 27, cost_space: str = "angle", cost: str = "volume",
                 exact_caps: float = 0.0) -> None:
        if cost not in ("volume", "power"):
            raise ValueError(f"cost must be 'volume' or 'power', got {cost!r}")
        self.cost = cost
        self.exact_caps = exact_caps
        self.exponent = exponent
        self.mass_weight = mass_weight
        self.merge_fraction = merge_fraction
        self.recenter_steps = recenter_steps
        self.bc_steps = bc_steps
        self.cost_offset = cost_offset
        self.candidates = candidates
        self.refine_rounds = refine_rounds
        self.refresh = refresh
        self.min_batch = min_batch
        self.knn_exact_max = knn_exact_max
        self.knn_probes = knn_probes
        self.exact_pairs = exact_pairs
        self.cost_space = cost_space

    def construct_filter(self, filter: Filter, vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = np.ascontiguousarray(vectors, dtype=np.float64)
        for _, cover in self.iter_candidates(filter, X, [filter.max_construct_caps]):
            cover = self.finish(filter, X, cover)
            return cover.directions, cover.angles

    def iter_candidates(self, filter: Filter, vectors: np.ndarray, ks: List[int],
                        weights: Optional[np.ndarray] = None) -> Iterator[Tuple[int, Cover]]:
        """Unrefined covers at each k of `ks`, largest first, from one merge:
        merging toward each k continues from the cover at the previous one.
        Each target sets its own merge_fraction cap, so the cover at the
        largest k is exactly construct_filter's before finish()."""
        X = np.ascontiguousarray(vectors, dtype=np.float64)
        n, dim = X.shape
        self._dim = dim
        if weights is None:
            rows = X.view(np.dtype((np.void, X.dtype.itemsize * dim))).ravel()
            _, first, owner = np.unique(rows, return_index=True, return_inverse=True)
            owner = owner.ravel().astype(np.int64)
            U = X[first].copy()
            mass = np.bincount(owner, minlength=len(first)).astype(np.float64)
        else:
            U = X.copy()
            owner = np.arange(n, dtype=np.int64)
            mass = np.asarray(weights, dtype=np.float64).copy()
        theta = np.zeros(len(U))
        nbr, dots, searched_at = None, None, 0

        for k in sorted(set(int(k) for k in ks), reverse=True):
            if k <= 0:
                yield k, Cover(np.empty((0, dim)), np.empty(0), np.zeros(n, dtype=np.int64))
                continue
            while len(U) > k:
                if nbr is None or len(U) < self.refresh * searched_at:
                    nbr, dots = self._knn(U)
                    searched_at = len(U)
                excess = len(U) - k
                max_pairs = min(excess, max(math.ceil(self.merge_fraction * excess),
                                            math.ceil(self.min_batch * len(U))))
                p, q = self._mutual_cheapest_pairs(theta, mass, nbr, dots, max_pairs)
                U, theta, mass, owner, nbr, dots, merged = self._merge(X, U, theta, mass, owner, nbr, dots, p, q)
                if self.recenter_steps > 0:
                    self._recenter(X, U, theta, owner, merged)
                self._refresh_dots(U, nbr, dots, merged)
            yield k, Cover(U.copy(), theta.copy(), owner.copy(), None if nbr is None else nbr.copy())

    def finish(self, filter: Filter, vectors: np.ndarray, cover: Cover) -> Cover:
        """refine_rounds of global minimax refinement, then exact angles."""
        X = np.ascontiguousarray(vectors, dtype=np.float64)
        U, labels = cover.directions, cover.labels
        if len(U) == 0:
            return cover
        if len(X) * len(U) <= self.exact_pairs:
            for _ in range(self.refine_rounds):
                _, labels = nearest_directions(X, U)
                order, starts, counts = segments(labels, len(U))
                U = badoiu_clarkson(X[order], labels[order], U, starts, counts, self.bc_steps)
                U = reseed_empty(X, U, counts, labels)
            U, angles, labels, kept = labelled_cap_cover(X, U)
            self._exact_recenter(X, U, angles, labels)
            return Cover(U, angles, labels, kept=kept)

        nbr = cover.neighbors
        if nbr is None:
            nbr, _ = self._knn(U)
        cand = np.concatenate([np.arange(len(U))[:, None], nbr], axis=1)
        # Caps that each hold only copies of one vector are already exact.
        if np.any(cover.angles > 0):
            for _ in range(self.refine_rounds):
                labels = nearest_in_candidates(X, U, cand, labels)
                order, starts, counts = segments(labels, len(U))
                U = badoiu_clarkson(X[order], labels[order], U, starts, counts, self.bc_steps)
                U, far = reseed_empty(X, U, counts, labels, return_rows=True)
                if len(far):
                    cand = _adopt(cand, np.flatnonzero(counts == 0), labels[far])
            labels = nearest_in_candidates(X, U, cand, labels)
        U, angles, labels, kept = labelled_cap_cover(X, U, labels)
        self._exact_recenter(X, U, angles, labels)
        return Cover(U, angles, labels, remap_neighbors(cand[:, 1:], kept, len(cand)), kept)

    def _exact_recenter(self, X, U, angles, labels) -> None:
        """Moves the widest `exact_caps` share of the caps with two or more
        members onto their exact minimum enclosing caps
        (cap_geometry.min_enclosing_cap), in place, where that is tighter."""
        if self.exact_caps <= 0:
            return
        order, starts, counts = segments(labels, len(U))
        multi = np.flatnonzero(counts >= 2)
        multi = multi[np.argsort(-angles[multi], kind="stable")][:int(np.ceil(self.exact_caps * len(multi)))]
        for j in multi:
            P = X[order[starts[j]:starts[j] + counts[j]]]
            u = min_enclosing_cap(P, U[j])
            t = float(angle_between(P, np.broadcast_to(u, P.shape)).max())
            if t < angles[j]:
                U[j], angles[j] = u, t

    # ------------------------------------------------------------- merging
    def _knn(self, U):
        return knn_directions(U, self.candidates, exact_max=self.knn_exact_max, probes=self.knn_probes)

    def _size(self, psi: np.ndarray) -> np.ndarray:
        """log of what a cap of accept angle psi costs: its measure on S^p
        (cost="volume", p = exponent, default dim - 1: the share of the
        sphere it accepts), or p ln psi (cost="power"), the small-angle
        form of the same."""
        if self.cost == "power":
            return (8.0 if self.exponent is None else self.exponent) * np.log(psi)
        p = self._dim - 1.0 if self.exponent is None else self.exponent
        return log_cap_measure(np.minimum(psi, np.pi), p + 1.0)

    def _edge_costs(self, lo, hi, phi, theta, mass) -> np.ndarray:
        """log of the increase of sum_i m_i^w size(theta_i + t) a merge
        causes, the merged angle taken from the enclosing cap."""
        ta, tb = theta[lo], theta[hi]
        merged = np.minimum(np.maximum(np.maximum(ta, tb), 0.5 * (phi + ta + tb)), np.pi)
        if self.cost_space == "sine":
            # Ball-Cover's units: the Euclidean MEB radius of a cap is sin(theta).
            ta, tb, merged = np.sin(np.minimum(ta, np.pi / 2)), np.sin(np.minimum(tb, np.pi / 2)),                 np.sin(np.minimum(merged, np.pi / 2))
        w, t, size = self.mass_weight, self.cost_offset, self._size
        log_a = size(ta + t)
        log_b = size(tb + t)
        log_c = size(merged + t)
        if w:
            log_a = log_a + w * np.log(mass[lo])
            log_b = log_b + w * np.log(mass[hi])
            log_c = log_c + w * np.log(mass[lo] + mass[hi])
        rest = np.exp(log_a - log_c) + np.exp(log_b - log_c)
        return log_c + np.log(np.maximum(1.0 - rest, 1e-12))

    def _mutual_cheapest_pairs(self, theta, mass, nbr, dots, max_pairs):
        a = len(theta)
        src = np.repeat(np.arange(a, dtype=np.int64), nbr.shape[1])
        dst = nbr.ravel()
        ok = (dst >= 0) & (dst != src)
        lo, hi = np.minimum(src[ok], dst[ok]), np.maximum(src[ok], dst[ok])
        edge, first = np.unique(lo * a + hi, return_index=True)
        phi = np.arccos(np.clip(dots.ravel()[ok][first], -1.0, 1.0))
        lo, hi = edge // a, edge % a
        cost = self._edge_costs(lo, hi, phi, theta, mass)
        rank = np.empty(len(edge), dtype=np.int64)
        rank[np.lexsort((edge, cost))] = np.arange(len(edge))
        best = np.full(a, len(edge), dtype=np.int64)
        np.minimum.at(best, lo, rank)
        np.minimum.at(best, hi, rank)
        mutual = np.flatnonzero((best[lo] == rank) & (best[hi] == rank))
        mutual = mutual[np.argsort(rank[mutual])][:max_pairs]
        return lo[mutual], hi[mutual]

    @staticmethod
    def _enclosing(ua, ta, ub, tb) -> np.ndarray:
        """Directions of the smallest caps enclosing each pair (vectorized
        cap_geometry.enclosing_cap); angles are re-measured by the caller."""
        phi = angle_between(ua, ub)
        th = 0.5 * (phi + ta + tb)
        ok = (phi > 1e-9) & (phi < np.pi - 1e-6)
        f = np.where(ok, (th - ta) / np.where(ok, phi, 1.0), 0.0)
        s = np.where(ok, np.sin(phi), 1.0)
        u = (np.sin((1.0 - f) * phi)[:, None] * ua + np.sin(f * phi)[:, None] * ub) / s[:, None]
        u = np.where(ok[:, None], u, ua)
        b_in_a = phi + tb <= ta
        a_in_b = ~b_in_a & (phi + ta <= tb)
        u[b_in_a] = ua[b_in_a]
        u[a_in_b] = ub[a_in_b]
        return unit_rows(u)

    def _merge(self, X, U, theta, mass, owner, nbr, dots, p, q):
        U[p] = self._enclosing(U[p], theta[p], U[q], theta[q])
        mass[p] += mass[q]
        a, m = nbr.shape
        alive = np.ones(a, dtype=bool)
        alive[q] = False
        target = np.arange(a, dtype=np.int64)
        target[q] = p
        new_id = np.cumsum(alive) - 1
        owner = new_id[target[owner]]

        # Neighbour lists follow the merge: ids remapped, and a merged cap
        # keeps the nearest m of its two parts' lists.
        both = np.concatenate([nbr[p], nbr[q]], axis=1)
        remap = lambda ids: np.where(ids >= 0, new_id[target[np.maximum(ids, 0)]], -1)
        nbr = remap(nbr)
        both = remap(both)
        both = np.sort(np.where(both == new_id[p][:, None], -1, both), axis=1)
        dup = np.zeros(both.shape, dtype=bool)
        dup[:, 1:] = both[:, 1:] == both[:, :-1]
        both[dup] = -1
        U = U[alive]
        theta = theta[alive]
        mass = mass[alive]
        nbr = nbr[alive]
        dots = dots[alive]
        merged_ids = new_id[p]
        ip = np.einsum("rd,rcd->rc", U[merged_ids], U[np.maximum(both, 0)])
        ip = np.where(both >= 0, ip, -np.inf)
        top = np.argsort(-ip, axis=1, kind="stable")[:, :m]
        nbr[merged_ids] = np.where(np.take_along_axis(ip, top, axis=1) > -np.inf,
                                   np.take_along_axis(both, top, axis=1), -1)

        merged = np.zeros(len(U), dtype=bool)
        merged[merged_ids] = True
        members = np.flatnonzero(merged[owner])
        ang = angle_between(X[members], U[owner[members]])
        order, starts, counts = segments(owner[members], len(U))
        theta[merged] = segment_max(ang[order], starts, counts)[merged]
        return U, theta, mass, owner, nbr, dots, merged_ids

    @staticmethod
    def _refresh_dots(U, nbr, dots, changed) -> None:
        """Re-measures the cached inner products of every listed pair with a
        cap in `changed` (merged or moved this round)."""
        flag = np.zeros(len(U), dtype=bool)
        flag[changed] = True
        stale = (nbr >= 0) & (flag[np.maximum(nbr, 0)] | flag[:, None])
        r, c = np.nonzero(stale)
        if len(r):
            dots[r, c] = pair_dots(U, r, nbr[r, c])

    def _recenter(self, X, U, theta, owner, ids) -> None:
        remap = np.full(len(U), -1, dtype=np.int64)
        remap[ids] = np.arange(len(ids))
        ball = remap[owner]
        rows = np.flatnonzero(ball >= 0)
        rows = rows[np.argsort(ball[rows], kind="stable")]
        label = ball[rows]
        _, starts, counts = segments(label, len(ids))
        new = badoiu_clarkson(X[rows], label, U[ids], starts, counts, self.recenter_steps)
        t = segment_max(angle_between(X[rows], new[label]), starts, counts)
        better = t < theta[ids]
        U[ids[better]] = new[better]
        theta[ids[better]] = t[better]

    def get_bytesize(self) -> int:
        return 0


def _adopt(cand: np.ndarray, empty: np.ndarray, host: np.ndarray) -> np.ndarray:
    """Candidate rows after reseed_empty moved caps `empty` onto members of
    caps `host`: each reseeded cap takes its host's row (and the host), and
    joins its host's row, so the host's members may move to it."""
    order = np.argsort(host, kind="stable")
    h = host[order]
    slot = np.arange(len(h)) - np.searchsorted(h, h)
    width = int(slot.max()) + 1
    extra = np.full((len(cand), width + 1), -1, dtype=np.int64)
    cand = cand.copy()
    cand[empty] = cand[host]
    cand[empty, 0] = empty
    extra[empty, width] = host
    extra[h, slot] = empty[order]
    return np.concatenate([cand, extra], axis=1)
