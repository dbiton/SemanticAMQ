import math
import numpy as np
import faiss
from typing import Iterator, List, Tuple
from filters.filter import Filter
from filters.cap_cover.cap_interfaces import ConstructInterface
from filters.cap_cover.minimax_construct import badoiu_clarkson, reseed_empty
from filters.cap_cover.cap_geometry import (angle_between, exact_cap_cover, nearest_directions, segments,
                                            segment_max, unit_rows)


class SphereMergeConstruct(ConstructInterface):
    """Ball-Cover's SphereMergeConstruct on caps: bottom-up merging of caps
    by an FPR-shaped cost, then global minimax refinement.

    Every distinct vector starts as a zero-angle cap. In rounds, every cap
    is scored against its `candidates` nearest directions, and the pairs that
    are each other's cheapest merge are merged -- at most `merge_fraction` of
    the merges still needed per round. A merge costs the increase of
    sum_i m_i^w (theta_i + t)^p (m = member count): a sum over points, so
    every member pays for the width of the cap it lives in, and tight caps go
    where the members (and so the queries) are dense. Ball-Cover measured the
    optimum flat around p = 8, w = 1, t = tau; on caps, merge_fraction
    0.05-0.25 and 1-3 refine rounds also measured flat.

    A merged cap starts as the smallest cap enclosing its two parts and is
    then re-centered on its own members with `recenter_steps`
    Badoiu-Clarkson steps; its angle is always re-measured from the members.
    After merging, `refine_rounds` rounds of nearest-direction reassignment
    and `bc_steps` Badoiu-Clarkson steps re-center the whole partition
    (finish()).

    Candidate neighbours are found by an exact search only when the cap
    count has fallen below `refresh` of what it was at the last search; in
    between, each cap keeps its list, and a merged cap takes the nearest
    `candidates` of its two parts' lists. Merges are local, so the lists
    stay good, and the all-pairs search -- the bulk of the cost -- runs a
    handful of times instead of once per round.

    Merging measured no better (within +-0.008 FPR) with a cost that sees
    caps as the encoder will widen them -- theta' = sqrt(theta^2 + delta^2),
    or the in-data reach CapCoverFilter selects encodings by, both with
    delta the encoding's angular error -- so the cost uses the unencoded
    angles."""

    def __init__(self, exponent: float = 8.0, mass_weight: float = 1.0, merge_fraction: float = 0.25,
                 recenter_steps: int = 8, bc_steps: int = 64, cost_offset: float = 0.1,
                 candidates: int = 16, refine_rounds: int = 1, refresh: float = 0.7) -> None:
        self.exponent = exponent
        self.mass_weight = mass_weight
        self.merge_fraction = merge_fraction
        self.recenter_steps = recenter_steps
        self.bc_steps = bc_steps
        self.cost_offset = cost_offset
        self.candidates = candidates
        self.refine_rounds = refine_rounds
        self.refresh = refresh

    def construct_filter(self, filter: Filter, vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = np.ascontiguousarray(vectors, dtype=np.float64)
        for _, U, theta, _ in self.iter_candidates(filter, X, [filter.max_construct_caps]):
            return self.finish(filter, X, U, theta)

    def iter_candidates(self, filter: Filter, vectors: np.ndarray,
                        ks: List[int]) -> Iterator[Tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
        """Unrefined covers at each k of `ks`, largest first, from one merge:
        merging toward each k continues from the cover at the previous one.
        Each target sets its own merge_fraction cap, so the cover at the
        largest k is exactly construct_filter's before finish()."""
        X = np.ascontiguousarray(vectors, dtype=np.float64)
        n, dim = X.shape
        rows = X.view(np.dtype((np.void, X.dtype.itemsize * dim))).ravel()
        _, first, owner = np.unique(rows, return_index=True, return_inverse=True)
        U = X[first].copy()
        theta = np.zeros(len(first))
        owner = owner.ravel().astype(np.int64)
        nbr, searched_at = None, 0

        for k in sorted(set(int(k) for k in ks), reverse=True):
            if k <= 0:
                yield k, np.empty((0, dim)), np.empty(0), np.zeros(n, dtype=np.int64)
                continue
            while len(U) > k:
                if nbr is None or len(U) < self.refresh * searched_at:
                    nbr, searched_at = self._search_neighbors(U), len(U)
                excess = len(U) - k
                mass = np.bincount(owner, minlength=len(U)).astype(np.float64)
                max_pairs = min(excess, math.ceil(self.merge_fraction * excess))
                p, q = self._mutual_cheapest_pairs(U, theta, mass, nbr, max_pairs)
                U, theta, owner, nbr, merged = self._merge(X, U, theta, owner, nbr, p, q)
                if self.recenter_steps > 0:
                    self._recenter(X, U, theta, owner, merged)
            yield k, U.copy(), theta.copy(), owner.copy()

    def finish(self, filter: Filter, vectors: np.ndarray, U: np.ndarray,
               theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """refine_rounds of global minimax refinement, then exact angles."""
        X = np.ascontiguousarray(vectors, dtype=np.float64)
        if len(U) == 0:
            return U, theta
        for _ in range(self.refine_rounds):
            _, labels = nearest_directions(X, U)
            order, starts, counts = segments(labels, len(U))
            U = badoiu_clarkson(X[order], labels[order], U, starts, counts, self.bc_steps)
            U = reseed_empty(X, U, counts, labels)
        return exact_cap_cover(X, U)

    # ------------------------------------------------------------- merging
    def _edge_costs(self, lo, hi, phi, theta, mass) -> np.ndarray:
        """log of the increase of sum_i m_i^w (theta_i + t)^p a merge causes,
        the merged angle taken from the enclosing cap."""
        ta, tb = theta[lo], theta[hi]
        merged = np.minimum(np.maximum(np.maximum(ta, tb), 0.5 * (phi + ta + tb)), np.pi)
        w, p, t = self.mass_weight, self.exponent, self.cost_offset
        log_a = w * np.log(mass[lo]) + p * np.log(ta + t)
        log_b = w * np.log(mass[hi]) + p * np.log(tb + t)
        log_c = w * np.log(mass[lo] + mass[hi]) + p * np.log(merged + t)
        rest = np.exp(log_a - log_c) + np.exp(log_b - log_c)
        return log_c + np.log(np.maximum(1.0 - rest, 1e-12))

    def _search_neighbors(self, U: np.ndarray) -> np.ndarray:
        """(caps x candidates) nearest other directions, -1 padded."""
        a = len(U)
        m = min(self.candidates, a - 1)
        index = faiss.IndexFlatIP(U.shape[1])
        U32 = np.ascontiguousarray(U, dtype=np.float32)
        index.add(U32)
        _, nbr = index.search(U32, m + 1)
        nbr = np.where(nbr == np.arange(a)[:, None], -1, nbr)
        # Drop each row's self-match (not always column 0 under ties).
        keep = np.argsort(nbr < 0, axis=1, kind="stable")[:, :m]
        return np.take_along_axis(nbr, keep, axis=1)

    def _mutual_cheapest_pairs(self, U, theta, mass, nbr, max_pairs):
        a = len(U)
        src = np.repeat(np.arange(a, dtype=np.int64), nbr.shape[1])
        dst = nbr.ravel()
        ok = (dst >= 0) & (dst != src)
        lo, hi = np.minimum(src[ok], dst[ok]), np.maximum(src[ok], dst[ok])
        edge = np.unique(lo * a + hi)
        lo, hi = edge // a, edge % a
        # Exact angles for the edges: lists are reused across rounds, so the
        # distances they were found at are stale.
        phi = angle_between(U[lo], U[hi])
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

    def _merge(self, X, U, theta, owner, nbr, p, q):
        U[p] = self._enclosing(U[p], theta[p], U[q], theta[q])
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
        nbr = nbr[alive]
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
        return U, theta, owner, nbr, merged_ids

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
