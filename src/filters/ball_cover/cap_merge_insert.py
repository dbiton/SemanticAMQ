import numpy as np
from typing import List, Optional
from filters.filter import Filter
from filters.ball_cover.ball_interfaces import InsertInterface
from filters.ball_cover.exact_cover import nearest_centers, sphere_radius


def _cap_cos(s: np.ndarray, r: np.ndarray, e: float) -> np.ndarray:
    """cos of the angular radius of the cap a ball with center norm s and
    radius r cuts from the unit sphere, for members whose norms lie in
    [1 - e, 1 + e] (all in units of the data's sphere radius): the least
    cos(angle(x, c)) = (n^2 + s^2 - |x - c|^2) / (2 s n) any such member
    inside the ball can have."""
    s = np.maximum(np.asarray(s, dtype=np.float64), 1e-12)
    r = np.asarray(r, dtype=np.float64)
    lo, hi = 1.0 - e, 1.0 + e

    def h(n):
        return (n * n + s * s - r * r) / (2.0 * s * n)

    out = np.minimum(h(lo), h(hi))
    crit = np.sqrt(np.maximum(s * s - r * r, 0.0))  # where h is least, if inside
    inside = (crit > lo) & (crit < hi)
    if np.any(inside):
        out = np.where(inside, np.minimum(out, h(np.clip(crit, lo, hi))), out)
    return np.clip(out, -1.0, 1.0)


def _cap_radius(s: float, cos_t: float, e: float) -> float:
    """Radius a ball centered at norm s needs to hold every point of norm in
    [1 - e, 1 + e] within the cap of cosine cos_t around its direction. The
    squared distance is convex in the point's norm, so the ends decide."""
    r2 = max((1 - e) ** 2 + s * s - 2 * s * (1 - e) * cos_t,
             (1 + e) ** 2 + s * s - 2 * s * (1 + e) * cos_t)
    return float(np.sqrt(max(r2, 0.0)))


def _angle(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Angle between unit vectors, well conditioned at small angles."""
    return 2.0 * np.arcsin(np.minimum(1.0, 0.5 * np.linalg.norm(u - v, axis=-1)))


class CapMergeInsert(InsertInterface):
    """Cheapest-pair merging, like ExhaustiveMergeInsert, with three changes:
    a merge cost shaped like the false-positive rate, merges done on the
    caps the balls cut from the data's sphere, and the global cheapest pair
    found from a cached best partner per ball instead of a k x k matrix.

    A not-already-covered vector becomes a zero-radius ball; past capacity
    the pair of balls whose merge costs least is replaced by its merged ball,
    and any ball the merged one wholly contains is dropped (its members are
    covered), so a merge can free more than one slot.

    1. Cost. Each ball carries a member count m (uint16): its share of the
       construct vectors, plus every vector inserted into it since. A merge
       costs the increase of sum_i m_i^w (theta_i + t)^p, theta = the ball's
       cap angle, with SphereMergeConstruct's exponent 8, mass weight 1 and
       offset 0.1 -- a sum over points, so a merge in a dense region (where
       the queries land) is dear and one between sparse balls cheap. The
       volume cost the other merge strategies use, log(R^dim - ...), is
       ~dim * log R_merged: it merges whichever pair makes the smallest
       ball, which equalizes radii regardless of how many points pay for
       them.

    2. Caps. Unit-norm members make a ball matter only through the cap it
       cuts from the sphere: ball (c, r) holds exactly the members within
       theta of c/|c|, cos(theta) = (1 + |c|^2 - r^2) / (2|c|). Two caps
       merge into their smallest enclosing cap, (phi + theta_a + theta_b)/2
       around the point on the great circle between them, and the center
       can then sit anywhere on that direction's ray with radius
       sqrt(1 + s^2 - 2 s cos(theta)) holding the very same unit vectors.
       Moving a center along its own ray therefore costs no radius, unlike
       the displacement a stream must otherwise add for moving a center
       without its members. Each merged center goes to its own norm
       1 / cos(theta), where the tau-expansion of the cap is least (~tau,
       against ~tau / cos(theta) for the Euclidean enclosing ball's center
       inside the sphere). Per-ball norms cost nothing here -- the norm is
       the center's own -- and, unlike in construction, nothing reassigns
       vectors to their nearest center afterwards.

       This rests on every member lying on the data's sphere. construct()'s
       vectors fix the sphere (exact_cover.sphere_radius) and a norm
       tolerance eps, and every inserted vector is checked against them:
       the caps are computed for member norms anywhere in [rho - eps,
       rho + eps], and one vector off the sphere switches the strategy to
       Euclidean enclosing balls for good. So does a filter whose balls
       were not built by construct() (no vectors to check).

    3. Search. Each ball caches its cheapest partner and that pair's cost
       (8 B per ball with the count, at uint16 ids -- charged by
       get_bytesize() and paid for in balls by max_spheres_for_budget()).
       A cached cost only ever turns into a lower bound -- its partner is
       merged away, or a partner gains members, and every cost grows with
       mass for mass_weight >= 1 -- so such balls are only marked stale and
       re-scanned, together, once one reaches the top: the pair picked is
       the one a full k x k scan would pick (checked bit-for-bit), at
       O(k * d) per insert instead of O(k^2 * d).

    Measured 2026-09-29 by main.generate_streaming (5 corpora x 5 folds,
    k = 800 at equal peak bytes -- 795 balls here -- 4000 inserts after a
    4000-vector bootstrap), fold-paired against ExhaustiveMergeInsert: FPR
    -0.089 with a Gonzalez bootstrap and -0.144 with a sphere_merge one,
    lower in 25 of 25 folds each. With the stream's order shuffled the
    sphere_merge-bootstrapped gain is -0.069 (25/25), but on a Gonzalez
    bootstrap the two tie: a coarse bootstrap cover is permanent, since a
    stream cannot split a ball. The cap geometry alone is worth ~-0.06 and
    the mass cost alone ~-0.05; moving merged centers onto the sphere with
    the radius grown by the displacement instead is +0.15. Pinned to one
    core it costs 2.3-2.7 ms per insert (worst single insert 9-35 ms)
    against ExhaustiveMergeInsert's 13-23 ms (56-74 ms).

    remove() is not supported."""

    # Relative spread of member norms beyond which the data is not treated as
    # lying on a sphere (exact_cover.sphere_radius's own tolerance).
    _SPHERE_RTOL = 1e-3

    def __init__(self, exponent: float = 8.0, mass_weight: float = 1.0, cost_offset: float = 0.1,
                 max_norm: float = 5.0) -> None:
        self.exponent = exponent
        self.mass_weight = mass_weight
        self.cost_offset = cost_offset
        self.max_norm = max_norm
        self._rho = 0.0     # data sphere radius; 0 -> Euclidean enclosing balls
        self._eps = 0.0     # max | |x| - rho | over every member seen
        self._mass = np.zeros(0, dtype=np.uint16)
        self._partner = np.zeros(0, dtype=np.int64)    # sphere id, -1 = stale
        self._cost = np.zeros(0, dtype=np.float32)     # partner's cost, or a lower bound if stale

    # ----------------------------------------------------------------- state
    @staticmethod
    def _id_bytes(max_spheres: int) -> int:
        # Ids come from the filter's id pool, at most ~2 * (max_spheres + 1).
        return 2 if 2 * (max_spheres + 1) < np.iinfo(np.uint16).max else 4

    def _state_bytes(self, max_spheres: int) -> int:
        return np.dtype(np.uint16).itemsize + self._id_bytes(max_spheres) + np.dtype(np.float32).itemsize

    def max_spheres_for_budget(self, usable_bytes: int, bytes_per_ball: int) -> int:
        k = usable_bytes // (bytes_per_ball + self._state_bytes(1))
        while k > 1 and k * (bytes_per_ball + self._state_bytes(k)) > usable_bytes:
            k -= 1
        return k

    def get_bytesize(self, filter: Filter) -> int:
        return filter._num_spheres * self._state_bytes(filter.max_spheres)

    def ensure_capacity(self, required_cap: int) -> None:
        cur = len(self._mass)
        if required_cap <= cur:
            return
        grow = required_cap - cur
        self._mass = np.concatenate([self._mass, np.ones(grow, dtype=np.uint16)])
        self._partner = np.concatenate([self._partner, np.full(grow, -1, dtype=np.int64)])
        self._cost = np.concatenate([self._cost, np.full(grow, -np.inf, dtype=np.float32)])

    def reset_buffers(self, capacity: int) -> None:
        self.ensure_capacity(capacity)
        self._mass[:] = 1
        self._partner[:] = -1
        self._cost[:] = -np.inf
        self._rho, self._eps = 0.0, 0.0

    def remove(self, filter: Filter, vectors: np.ndarray) -> List[bool]:
        raise NotImplementedError("CapMergeInsert does not support remove operation.")

    # -------------------------------------------------------------- geometry
    def _caps(self) -> bool:
        return self._rho > 0.0

    def _size(self, s: np.ndarray, r: np.ndarray) -> np.ndarray:
        """Per-ball size the cost and merges use: the cap angle, or the
        Euclidean radius off the sphere."""
        if self._caps():
            return np.arccos(_cap_cos(s / self._rho, r / self._rho, self._eps / self._rho))
        return np.asarray(r, dtype=np.float64)

    def _merged_size(self, sa, sb, d):
        sab = np.maximum(np.maximum(sa, sb), 0.5 * (d + sa + sb))
        return np.minimum(sab, np.pi) if self._caps() else sab

    def _pair_costs(self, sa, sb, sab, ma, mb):
        """log of the increase of sum m^w (size + t)^p a merge causes."""
        w, p = self.mass_weight, self.exponent
        t = self.cost_offset * (1.0 if self._caps() else self._scale)
        log_a = w * np.log(ma) + p * np.log(sa + t)
        log_b = w * np.log(mb) + p * np.log(sb + t)
        log_c = w * np.log(ma + mb) + p * np.log(sab + t)
        rest = np.exp(log_a - log_c) + np.exp(log_b - log_c)
        return log_c + np.log(np.maximum(1.0 - rest, 1e-12))

    def _merge_caps(self, ua, ta, ub, tb):
        phi = float(_angle(ua, ub))
        if phi + tb <= ta:
            return ua.copy(), ta
        if phi + ta <= tb:
            return ub.copy(), tb
        theta = 0.5 * (phi + ta + tb)
        if theta >= np.pi:
            return ua.copy(), np.pi
        if phi < 1e-12:
            u = ua.copy()
        else:
            f = (theta - ta) / phi
            u = (np.sin((1.0 - f) * phi) * ua + np.sin(f * phi) * ub) / np.sin(phi)
            u /= np.linalg.norm(u)
        # Re-measured from the u actually computed, so rounding only widens it.
        theta = max(theta, float(_angle(u, ua)) + ta, float(_angle(u, ub)) + tb)
        return u, min(theta, np.pi)

    def _place(self, u: np.ndarray, theta: float):
        """Center and radius holding the cap (u, theta), at norm 1/cos(theta)."""
        cos_t = float(np.cos(theta))
        s = 1.0 / min(1.0, max(cos_t, 1.0 / self.max_norm))
        r = _cap_radius(s, cos_t, self._eps / self._rho)
        return self._rho * s * u, self._rho * r

    @staticmethod
    def _merge_balls(c1, r1, c2, r2):
        """Smallest enclosing ball of two balls (ExhaustiveMergeInsert's)."""
        diff = c2 - c1
        dist = float(np.sqrt(max(float(np.dot(diff, diff)), 0.0)))
        if dist + min(r1, r2) <= max(r1, r2):
            return (c1.copy(), r1) if r1 >= r2 else (c2.copy(), r2)
        new_r = 0.5 * (dist + r1 + r2)
        new_c = c1 if dist < 1e-12 else c1 + ((new_r - r1) / dist) * diff
        return new_c, new_r

    # ------------------------------------------------------------- bootstrap
    def initialize_buffers(self, filter: Filter, spheres_indices: np.ndarray, centers: np.ndarray, radii: np.ndarray,
                           vectors: Optional[np.ndarray] = None) -> None:
        self.reset_buffers(len(filter._all_radiuses))
        ids = np.asarray(spheres_indices, dtype=np.int64)
        if vectors is not None and len(vectors) and len(ids):
            X = np.ascontiguousarray(vectors, dtype=np.float32)
            self._rho = sphere_radius(X, rtol=self._SPHERE_RTOL)
            if self._rho:
                norms = np.sqrt(np.einsum("ij,ij->i", X, X, dtype=np.float64))
                self._eps = float(np.abs(norms - self._rho).max())
            _, owner = nearest_centers(X, centers)
            counts = np.bincount(owner, minlength=len(ids))
            self._mass[ids] = np.clip(counts, 1, np.iinfo(np.uint16).max)
        self._rescan_all(filter)

    def _rescan_all(self, filter: Filter) -> None:
        """Every ball's cheapest partner -- at construct time, so the first
        insert doesn't pay for it."""
        self._load(filter, extra=0)
        if self._L >= 2:
            self._rescan_rows(np.arange(self._L))
        self._save()

    # ------------------------------------------------------------------ view
    # Each insert() works on a dense view of the live balls, rebuilt from the
    # filter's decoded centers (float32 directions, float64 scalars) and
    # dropped at the end, like BallCoverFilter's own all_centers cache: only
    # the per-ball count, partner and cost persist between calls. Rows are
    # stable within a call; balls merged away are tombstoned.
    def _load(self, filter: Filter, extra: int) -> None:
        ids = filter.all_spheres.astype(np.int64)
        L = len(ids)
        n = L + extra + 8
        C = filter._all_centers[ids] if filter._all_centers is not None else \
            filter.all_centers_index.reconstruct_batch(ids)
        self._ids = np.full(n, -1, dtype=np.int64)
        self._U = np.empty((n, filter.dim), dtype=np.float32)
        self._s = np.zeros(n)
        self._r = np.zeros(n)
        self._sig = np.zeros(n)
        self._m = np.ones(n)
        self._best = np.full(n, -1, dtype=np.int64)
        self._bcost = np.full(n, np.inf)
        self._stale = np.zeros(n, dtype=bool)
        self._alive = np.zeros(n, dtype=bool)
        self._L = self._n_alive = L
        if L == 0:
            self._scale = 1.0
            return
        s = np.sqrt(np.einsum("ij,ij->i", C, C, dtype=np.float64))
        r = filter._all_radiuses[ids].astype(np.float64)
        self._ids[:L] = ids
        self._U[:L] = C / np.maximum(s, 1e-30)[:, None].astype(np.float32) if self._caps() else C
        self._s[:L] = s
        self._r[:L] = r
        self._sig[:L] = self._size(s, r)
        self._m[:L] = self._mass[ids]
        self._alive[:L] = True
        # Off the sphere, the cost offset scales with the data like
        # SphereMergeConstruct's: RMS distance of the centers to their mean.
        self._scale = 1.0 if self._caps() else max(
            float(np.sqrt(np.mean(np.sum((C - C.mean(axis=0)) ** 2, axis=1)))), 1e-12)
        row = np.full(len(self._mass), -1, dtype=np.int64)
        row[ids] = np.arange(L)
        partner = self._partner[ids]
        self._best[:L] = np.where(partner >= 0, row[np.maximum(partner, 0)], -1)
        self._bcost[:L] = self._cost[ids]
        self._stale[:L] = self._best[:L] < 0

    def _save(self) -> None:
        live = np.flatnonzero(self._alive[:self._L])
        ids = self._ids[live]
        self._mass[ids] = np.clip(self._m[live], 1, np.iinfo(np.uint16).max).astype(np.uint16)
        best = self._best[live]
        fresh = ~self._stale[live] & (best >= 0)
        self._partner[ids] = np.where(fresh, self._ids[np.maximum(best, 0)], -1)
        self._cost[ids] = self._bcost[live]
        self._U = None

    def _grow(self) -> None:
        extra = max(len(self._ids), 16)
        for name, fill in (("_ids", -1), ("_s", 0.0), ("_r", 0.0), ("_sig", 0.0), ("_m", 1.0), ("_best", -1),
                           ("_bcost", np.inf), ("_stale", False), ("_alive", False)):
            a = getattr(self, name)
            setattr(self, name, np.concatenate([a, np.full(extra, fill, dtype=a.dtype)]))
        self._U = np.concatenate([self._U, np.empty((extra, self._U.shape[1]), dtype=np.float32)])

    def _dist(self, U: np.ndarray, rows: np.ndarray) -> np.ndarray:
        """Approximate (float32) angle, or Euclidean distance off the sphere,
        from each of `rows` to every row -- ranks candidates only; every
        decision that coverage rests on is re-made exactly."""
        L = self._L
        dots = (U @ self._U[:L].T).astype(np.float64)
        if self._caps():
            return 2.0 * np.arcsin(np.minimum(1.0, 0.5 * np.sqrt(np.maximum(2.0 - 2.0 * dots, 0.0))))
        sq = self._s[:L] ** 2
        return np.sqrt(np.maximum(sq[None, :] + (self._s[rows] ** 2)[:, None] - 2.0 * dots, 0.0))

    def _costs_from(self, y: int, d: np.ndarray) -> np.ndarray:
        L = self._L
        c = self._pair_costs(self._sig[y], self._sig[:L], self._merged_size(self._sig[y], self._sig[:L], d),
                             self._m[y], self._m[:L])
        c = np.where(self._alive[:L], c, np.inf)
        c[y] = np.inf
        return c

    def _rescan(self, y: int) -> None:
        c = self._costs_from(y, self._dist(self._U[y][None, :], np.array([y]))[0])
        j = int(np.argmin(c))
        self._best[y] = j if np.isfinite(c[j]) else -1
        self._bcost[y] = c[j]
        self._stale[y] = False

    def _rescan_rows(self, rows: np.ndarray, block: int = 256) -> None:
        """_rescan for many rows at once, as blocked matrix products rather
        than one matrix-vector product per row."""
        L = self._L
        for i in range(0, len(rows), block):
            r = rows[i:i + block]
            d = self._dist(self._U[r], r)
            sa, sb = self._sig[r, None], self._sig[None, :L]
            c = self._pair_costs(sa, sb, self._merged_size(sa, sb, d), self._m[r, None], self._m[None, :L])
            c = np.where(self._alive[None, :L], c, np.inf)
            c[np.arange(len(r)), r] = np.inf
            j = np.argmin(c, axis=1)
            cheapest = c[np.arange(len(r)), j]
            self._best[r] = np.where(np.isfinite(cheapest), j, -1)
            self._bcost[r] = cheapest
            self._stale[r] = False

    def _center(self, filter: Filter, y: int) -> np.ndarray:
        return filter._all_centers[self._ids[y]].astype(np.float64)

    # ---------------------------------------------------------------- insert
    def insert(self, filter: Filter, vectors: np.ndarray) -> None:
        X = np.ascontiguousarray(vectors, dtype=np.float32)
        if len(X) == 0:
            return
        if self._caps():
            norms = np.sqrt(np.einsum("ij,ij->i", X, X, dtype=np.float64))
            dev = float(np.abs(norms - self._rho).max())
            if dev > self._SPHERE_RTOL * self._rho:
                self._rho = 0.0     # a member off the sphere: Euclidean from here on
            else:
                self._eps = max(self._eps, dev)
        self.ensure_capacity(len(filter._all_radiuses))
        self._load(filter, extra=2 * len(X))
        # One vector at a time: genuinely sequential, since every merge
        # changes what the next vector finds.
        for x in X:
            self._insert_one(filter, x)
        self._save()

    def _covering_row(self, filter: Filter, x: np.ndarray) -> int:
        """Nearest-center live ball that covers x, or -1. Screened in float32,
        decided in float64 from the decoded centers query() scores against."""
        L = self._L
        if self._n_alive == 0:
            return -1
        x64 = x.astype(np.float64)
        dots = (self._U[:L] @ x).astype(np.float64)
        xx = float(x64 @ x64)
        if self._caps():
            d2 = xx + self._s[:L] ** 2 - 2.0 * self._s[:L] * dots
        else:
            d2 = xx + self._s[:L] ** 2 - 2.0 * dots
        near = np.flatnonzero(self._alive[:L] & (np.sqrt(np.maximum(d2, 0.0)) <= self._r[:L] + 1e-3))
        if len(near) == 0:
            return -1
        diff = filter._all_centers[self._ids[near]].astype(np.float64) - x64
        dist = np.sqrt(np.einsum("ij,ij->i", diff, diff))
        ok = dist <= self._r[near]
        return int(near[ok][np.argmin(dist[ok])]) if ok.any() else -1

    def _insert_one(self, filter: Filter, x: np.ndarray) -> None:
        y = self._covering_row(filter, x)
        if y >= 0:
            self._add_mass(y, 1.0)
            return
        sid = int(filter._add_spheres(x[None, :], np.zeros(1, dtype=np.float32))[0])
        self._append(filter, sid, 1.0)
        while self._n_alive > filter.max_spheres:
            self._merge_cheapest(filter)

    def _add_mass(self, y: int, dm: float) -> None:
        """Every pair with y costs more now, so y's cached cost and those of
        the balls whose partner is y become lower bounds (mass_weight >= 1);
        below 1 a cost can fall with mass, so they are re-scanned outright."""
        self._m[y] += dm
        L = self._L
        dependents = self._alive[:L] & (self._best[:L] == y)
        if self.mass_weight >= 1.0:
            self._stale[y] = True
            self._stale[:L] |= dependents
        else:
            for z in [y, *np.flatnonzero(dependents)]:
                self._rescan(int(z))

    def _kill(self, filter: Filter, rows: np.ndarray) -> None:
        filter._remove_spheres(self._ids[rows])
        self._alive[rows] = False
        self._bcost[rows] = np.inf
        self._n_alive -= len(rows)
        L = self._L
        self._stale[:L] |= self._alive[:L] & np.isin(self._best[:L], rows)

    def _append(self, filter: Filter, sid: int, m: float) -> None:
        if self._L >= len(self._ids):
            self._grow()
        y = self._L
        self._L += 1
        self._n_alive += 1
        c = filter._all_centers[sid].astype(np.float64)
        s = float(np.sqrt(c @ c))
        r = float(filter._all_radiuses[sid])
        self._ids[y] = sid
        self._U[y] = c / max(s, 1e-300) if self._caps() else c
        self._s[y], self._r[y], self._m[y] = s, r, m
        self._sig[y] = float(self._size(np.array([s]), np.array([r]))[0])
        self._alive[y] = True
        self._stale[y] = False
        L = self._L
        d = self._dist(self._U[y][None, :], np.array([y]))[0]
        # Balls wholly inside the new one hold nothing it doesn't: drop them
        # and keep their members' count. Screened in float32, decided exactly.
        near = np.flatnonzero(self._alive[:L] & (d + self._sig[:L] <= self._sig[y] + 1e-3))
        near = near[near != y]
        if len(near):
            other = filter._all_centers[self._ids[near]].astype(np.float64)
            if self._caps():
                exact = _angle(other / np.linalg.norm(other, axis=1)[:, None], c / s)
            else:
                exact = np.linalg.norm(other - c, axis=1)
            inside = near[exact + self._sig[near] <= self._sig[y]]
            if len(inside):
                self._m[y] += self._m[inside].sum()
                self._kill(filter, inside)
        costs = self._costs_from(y, d)
        cheaper = costs < self._bcost[:L]
        self._bcost[:L][cheaper] = costs[cheaper]
        self._best[:L][cheaper] = y
        self._stale[:L][cheaper] = False    # a real cost below a lower bound is the min
        j = int(np.argmin(costs))
        self._best[y] = j if np.isfinite(costs[j]) else -1
        self._bcost[y] = costs[j]

    def _merge_cheapest(self, filter: Filter) -> None:
        L = self._L
        live = self._alive[:L]
        a = int(np.argmin(np.where(live, self._bcost[:L], np.inf)))
        b = int(self._best[a])
        if self._stale[a] or b < 0 or not self._alive[b]:
            # Any stale ball whose lower bound is at or under the cheapest
            # known cost may hold the true minimum; none above it can. Re-scan
            # them together: merging away a popular partner can stale
            # hundreds at once, and one matrix product beats hundreds of
            # matrix-vector ones (the spike was ~0.1 ms per stale ball).
            fresh = live & ~self._stale[:L]
            bound = np.min(np.where(fresh, self._bcost[:L], np.inf))
            rows = np.flatnonzero(live & self._stale[:L] & (self._bcost[:L] <= bound))
            self._rescan_rows(np.union1d(rows, [a]))
            return
        ca, cb = self._center(filter, a), self._center(filter, b)
        if self._caps():
            u, theta = self._merge_caps(ca / self._s[a], self._sig[a], cb / self._s[b], self._sig[b])
            center, radius = self._place(u, theta)
        else:
            center, radius = self._merge_balls(ca, self._r[a], cb, self._r[b])
        m = self._m[a] + self._m[b]
        self._kill(filter, np.array([a, b]))
        sid = int(filter._add_spheres(np.asarray(center, dtype=np.float32)[None, :],
                                      np.array([radius], dtype=np.float32))[0])
        self._append(filter, sid, m)
