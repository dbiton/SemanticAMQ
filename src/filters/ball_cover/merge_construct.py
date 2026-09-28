import numpy as np
import faiss
from types import SimpleNamespace
from typing import Optional, Tuple
from filters.filter import Filter
from filters.ball_cover.ball_interfaces import ConstructInterface
from filters.ball_cover.gonzales_construct import GonzalesConstruct
from filters.ball_cover.min_heap_insert import MinHeapInsert
from filters.ball_cover.exact_cover import exact_cover, nearest_centers, segments, segment_max


class MergeConstruct(ConstructInterface):
    """Bottom-up cheapest-pair merging: the streaming merge (MinHeapInsert,
    ExhaustiveMergeInsert) run as a bulk build. Every distinct vector starts
    as a zero-radius ball, and pairs are merged into their enclosing ball,
    cheapest first by MinHeapInsert._compute_merged_radius's cost, until k
    balls remain.

    Merges happen in rounds rather than one heap pop at a time, which is
    what keeps this vectorized. Each round scores every ball against its
    `candidates` nearest centers and merges every pair that is each other's
    cheapest option (reciprocal-nearest-neighbour agglomeration). Such a
    pair is close to a merge sequential greedy would also make: the
    enclosing ball of a merged ball and a third one contains one of the
    originals and that third one, so a merge rarely makes a later merge
    cheaper. It is not exact, because candidates are limited and radii are
    re-measured. Once mutual pairs outnumber the merges still needed, only
    the cheapest are taken. Rounds shrink the ball count geometrically, so
    the cost is a few exact center-to-center searches instead of one Python
    iteration per merge.

    A merged ball's radius is re-measured against its member vectors, which
    streaming can't do. The enclosing ball of two balls is only an upper
    bound on its members' spread, and letting that bound compound across
    rounds would bias every later cost toward whichever balls merged early.

    With `seed_multiplier` set, the merge-down loop starts not from one ball
    per distinct point but from an oversampled GonzalesConstruct run of
    m = seed_multiplier * k centers (capped at n), each with its exact
    per-cluster radius. Early rounds on the unseeded path pay for all-pairs
    candidate search (IndexFlatL2 over the current ball set) at a working
    set close to n several times before it shrinks geometrically; starting
    from m << n balls skips most of that, at the cost of merging from a
    coarser, already-committed partition instead of every original point."""

    def __init__(self, candidates: int = 16, seed_multiplier: Optional[float] = None) -> None:
        self.candidates = candidates
        self.seed_multiplier = seed_multiplier


    def construct_filter(self, filter: Filter, vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = np.ascontiguousarray(vectors, dtype=np.float32)
        n = X.shape[0]
        dim = X.shape[1]
        k = min(filter.max_construct_spheres, n)
        if k <= 0:
            return np.empty((0, dim), dtype=np.float32), np.empty(0, dtype=np.float32)
        if n <= k:
            return X.copy(), np.zeros(n, dtype=np.float32)

        if self.seed_multiplier is None:
            # Exact duplicates start as one ball. The row bytes are the key,
            # so this is O(n log n) with no distance computation at all.
            rows = X.view(np.dtype((np.void, X.dtype.itemsize * dim))).ravel()
            _, first, owner = np.unique(rows, return_index=True, return_inverse=True)
            centers = X[first].astype(np.float64)
            radii = np.zeros(len(first), dtype=np.float64)
            owner = owner.ravel().astype(np.int64)
        else:
            m = min(round(self.seed_multiplier * k), n)
            seeds, _ = GonzalesConstruct().construct_filter(SimpleNamespace(max_construct_spheres=m), X)
            if len(seeds) <= k:
                return exact_cover(X, seeds)
            d2, owner = nearest_centers(X, seeds)
            order, starts, counts = segments(owner, len(seeds))
            r2 = segment_max(d2[order], starts, counts)
            centers = seeds.astype(np.float64)
            radii = np.sqrt(r2).astype(np.float64)
            owner = owner.astype(np.int64)

        while len(centers) > k:
            p, q = self._mutual_cheapest_pairs(centers, radii, dim, len(centers) - k)
            centers, radii, owner = self._merge(X, centers, radii, owner, p, q)

        # Radii from the exact re-search, never from the ones tracked above.
        return exact_cover(X, centers)


    def _mutual_cheapest_pairs(self, centers: np.ndarray, radii: np.ndarray, dim: int,
                               max_pairs: int) -> Tuple[np.ndarray, np.ndarray]:
        """Disjoint (p, q) ball pairs that are each other's cheapest merge
        among their candidate edges, cheapest first, at most max_pairs."""
        a = len(centers)
        m = min(self.candidates, a - 1)
        C = np.ascontiguousarray(centers, dtype=np.float32)
        index = faiss.IndexFlatL2(dim)
        index.add(C)
        d2, nbr = index.search(C, m + 1)

        # Drop each row's self-match. Not always column 0: a center that
        # ties with another can come back second.
        keep = np.argsort(nbr == np.arange(a)[:, None], axis=1, kind="stable")[:, :m]
        nbr = np.take_along_axis(nbr, keep, axis=1).ravel()
        d2 = np.take_along_axis(d2, keep, axis=1).ravel()
        src = np.repeat(np.arange(a, dtype=np.int64), m)

        # One undirected edge per pair. Its approximate float32 distance only
        # ranks candidates -- _merge measures the pairs it acts on exactly.
        lo, hi = np.minimum(src, nbr), np.maximum(src, nbr)
        edge, idx = np.unique(lo * a + hi, return_index=True)
        lo, hi = edge // a, edge % a
        dist = np.sqrt(np.maximum(d2[idx], 0.0))
        cost = MinHeapInsert._compute_merged_radius(dim, radii[lo], radii[hi], dist)[1]

        # Total order on edges, ties broken by edge id, so the globally
        # cheapest edge is always mutual and every round makes progress --
        # duplicate-heavy corpora produce many exactly tied costs.
        rank = np.empty(len(edge), dtype=np.int64)
        rank[np.lexsort((edge, cost))] = np.arange(len(edge))
        best = np.full(a, len(edge), dtype=np.int64)
        np.minimum.at(best, lo, rank)
        np.minimum.at(best, hi, rank)

        mutual = np.flatnonzero((best[lo] == rank) & (best[hi] == rank))
        mutual = mutual[np.argsort(rank[mutual])][:max_pairs]
        return lo[mutual], hi[mutual]


    @staticmethod
    def _merge(X: np.ndarray, centers: np.ndarray, radii: np.ndarray, owner: np.ndarray,
               p: np.ndarray, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Replaces each ball p with the enclosing ball of p and q (the same
        geometry as ExhaustiveMergeInsert._merge_two_balls), retires q, and
        re-measures each merged radius from its members."""
        diff = centers[q] - centers[p]
        dist = np.sqrt(np.einsum("ij,ij->i", diff, diff))
        rp, rq = radii[p], radii[q]
        new_r = 0.5 * (dist + rp + rq)
        step = np.divide(new_r - rp, dist, out=np.zeros_like(dist), where=dist > 0.0)
        new_c = centers[p] + step[:, None] * diff

        q_inside_p = dist + rq <= rp
        p_inside_q = ~q_inside_p & (dist + rp <= rq)
        new_c[q_inside_p] = centers[p[q_inside_p]]
        new_c[p_inside_q] = centers[q[p_inside_q]]
        centers[p] = new_c

        a = len(centers)
        alive = np.ones(a, dtype=bool)
        alive[q] = False
        target = np.arange(a, dtype=np.int64)
        target[q] = p
        new_id = np.cumsum(alive) - 1
        owner = new_id[target[owner]]
        centers = centers[alive]
        radii = radii[alive]

        # Exact member radius around each new center: never larger than the
        # enclosing-ball bound, since every member was inside p or q.
        merged = np.zeros(len(centers), dtype=bool)
        merged[new_id[p]] = True
        members = np.flatnonzero(merged[owner])
        member_d = np.empty(len(members), dtype=np.float64)
        for s in range(0, len(members), 16_384):
            rows = members[s:s + 16_384]
            delta = X[rows] - centers[owner[rows]]
            member_d[s:s + 16_384] = np.sqrt(np.einsum("ij,ij->i", delta, delta))
        order, starts, counts = segments(owner[members], len(centers))
        radii[merged] = segment_max(member_d[order], starts, counts)[merged]
        return centers, radii, owner


    def get_bytesize(self) -> int:
        return 0
