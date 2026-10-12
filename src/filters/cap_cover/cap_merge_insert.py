import numpy as np
from typing import List, Optional
from filters.filter import Filter
from filters.cap_cover.cap_interfaces import InsertInterface
from filters.cap_cover.cap_geometry import angle_between, enclosing_cap, log_cap_measure, nearest_directions


class CapMergeInsert(InsertInterface):
    """Ball-Cover's CapMergeInsert on stored caps: a vector no cap holds
    becomes a new cap; past capacity the pair whose merge costs least is
    replaced by its enclosing cap, and caps the merged one wholly contains
    are dropped (their members are held), so a merge can free several
    slots.

    Cost: the increase of the caps' summed accept-region measure,
    sum_j sigma_p(theta_j + t), as SphereMergeConstruct's (p = `exponent`,
    default dim - 1; t = cost_offset). mass_weight w > 0 weights each cap
    by m^w, m its member count (construct's share plus every vector
    inserted into it since), and cost="power" takes (theta + t)^p: w = 1,
    p = 8 is the earlier cost. Measured (compare_cap_cover.py stream,
    |B| = 8000, kf = 0.1, bootstrap on half of B in fold order, 25 cells,
    streamv_*.csv against stream_cap_*.csv; the bootstrap merge unweighted
    and by volume too), FPR at the end of the stream: -0.012 at TQ4
    (20 of 25 cells better), +0.016 at TQ2 and +0.024 with the encoding
    chosen at bootstrap -- all of it ChatGPT-paraphrases (+0.10 at TQ2,
    where paraphrase clusters lose the member-count weights: the
    unweighted (theta + 0.1)^8 does the same, .865 against .766 weighted),
    -0.03..+0.015 on the other corpora. The measure on S^8 (exponent=8)
    saturates near pi, so merging two wide caps into one around the whole
    sphere costs little: FPR 1.0 on every ChatGPT-paraphrases fold at TQ2.

    Geometry: Ball-Cover's version had to recover each ball's cap from a
    center and radius; here the caps are stored as such. Two caps merge into
    their smallest enclosing cap (u, theta), which is encoded like any
    direction; the stored angle is measured around the *decoded* direction
    from the two parts' own (stored) caps, max(angle(dec, u_a) + theta_a,
    angle(dec, u_b) + theta_b), so it holds every unit vector either part
    held -- the triangle inequality on the sphere, never an estimate. A new
    cap for vector x stores angle(x, decode(x)).

    Codes: a merged cap's code is fitted to hold its two parts within the
    least stored angle (fit_codes, CapCoverFilter.fit_cap_code) rather than
    taken nearest the enclosing cap's direction: every merge re-encodes, so
    the saving compounds over a stream. Measured (compare_cap_cover.py
    stream, |B| = 8000, kf = 0.1, bootstrap on half of B in fold order, 25
    cells), FPR at the end of the stream: -0.071 at TQ2 (25/25 cells),
    -0.039 with the encoding auto-chosen at bootstrap (19 of 20 that
    differ), -0.008 at TQ4.

    Search: each cap caches its cheapest partner and that pair's cost (6 B
    per cap at uint16 ids, 8 B with the member count a weighted cost needs
    -- charged by get_bytesize() and paid for in caps by
    max_caps_for_budget()). A cached cost only ever becomes a lower bound
    -- its partner is merged away, or gains mass and every cost grows with
    mass for mass_weight >= 1 -- so such caps are marked stale and
    re-scanned together once one reaches the top: the pair picked is the
    one a full k x k scan would pick, at O(k * d) per insert.

    remove() is not supported."""

    def __init__(self, exponent: Optional[float] = None, mass_weight: float = 0.0, cost_offset: float = 0.1,
                 fit_codes: bool = True, cost: str = "volume") -> None:
        if cost not in ("volume", "power"):
            raise ValueError(f"cost must be 'volume' or 'power', got {cost!r}")
        self.cost = cost
        self.exponent = exponent
        self.mass_weight = mass_weight
        self.cost_offset = cost_offset
        self.fit_codes = fit_codes
        self._mass = np.zeros(0, dtype=np.uint16)
        self._partner = np.zeros(0, dtype=np.int64)    # cap id, -1 = stale
        self._cost = np.zeros(0, dtype=np.float32)     # partner's cost, or a lower bound if stale

    # ----------------------------------------------------------------- state
    @staticmethod
    def _id_bytes(max_caps: int) -> int:
        # Ids come from the filter's id pool, at most ~2 * (max_caps + 1).
        return 2 if 2 * (max_caps + 1) < np.iinfo(np.uint16).max else 4

    def _state_bytes(self, max_caps: int) -> int:
        # The member count only matters to a mass-weighted cost.
        count = np.dtype(np.uint16).itemsize if self.mass_weight else 0
        return count + self._id_bytes(max_caps) + np.dtype(np.float32).itemsize

    def max_caps_for_budget(self, usable_bytes: int, bytes_per_cap: int) -> int:
        k = usable_bytes // (bytes_per_cap + self._state_bytes(1))
        while k > 1 and k * (bytes_per_cap + self._state_bytes(k)) > usable_bytes:
            k -= 1
        return k

    def get_bytesize(self, filter: Filter) -> int:
        return filter._num_caps * self._state_bytes(filter.max_caps)

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

    def remove(self, filter: Filter, vectors: np.ndarray) -> List[bool]:
        raise NotImplementedError("CapMergeInsert does not support remove operation.")

    # ------------------------------------------------------------- bootstrap
    def initialize_buffers(self, filter: Filter, caps_indices: np.ndarray, directions: np.ndarray,
                           angles: np.ndarray, vectors: Optional[np.ndarray] = None) -> None:
        self.reset_buffers(len(filter._angle_codes))
        ids = np.asarray(caps_indices, dtype=np.int64)
        if self.mass_weight and vectors is not None and len(vectors) and len(ids):
            _, owner = nearest_directions(vectors, directions)
            counts = np.bincount(owner, minlength=len(ids))
            self._mass[ids] = np.clip(counts, 1, np.iinfo(np.uint16).max)
        self._load(filter, extra=0)
        if self._L >= 2:
            self._rescan_rows(np.arange(self._L))
        self._save()

    # ------------------------------------------------------------------ view
    # Each insert() works on a dense view of the live caps (decoded unit
    # directions, angles as stored), dropped at the end: only the per-cap
    # count, partner and cost persist between calls. Rows are stable within
    # a call; caps merged away are tombstoned.
    def _load(self, filter: Filter, extra: int) -> None:
        ids = filter.all_caps.astype(np.int64)
        L = len(ids)
        n = L + extra + 8
        self._ids = np.full(n, -1, dtype=np.int64)
        self._U = np.zeros((n, filter.dim), dtype=np.float64)
        self._t = np.zeros(n)
        self._m = np.ones(n)
        self._best = np.full(n, -1, dtype=np.int64)
        self._bcost = np.full(n, np.inf)
        self._stale = np.zeros(n, dtype=bool)
        self._alive = np.zeros(n, dtype=bool)
        self._L = self._n_alive = L
        if L == 0:
            return
        self._ids[:L] = ids
        self._U[:L] = filter.all_directions
        self._t[:L] = filter._angle_value(filter._angle_codes[ids])
        self._m[:L] = self._mass[ids]
        self._alive[:L] = True
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
        for name, fill in (("_ids", -1), ("_t", 0.0), ("_m", 1.0), ("_best", -1),
                           ("_bcost", np.inf), ("_stale", False), ("_alive", False)):
            a = getattr(self, name)
            setattr(self, name, np.concatenate([a, np.full(extra, fill, dtype=a.dtype)]))
        self._U = np.concatenate([self._U, np.zeros((extra, self._U.shape[1]))])

    def _dist(self, U: np.ndarray) -> np.ndarray:
        """Approximate angles (float32 inner products) from rows of U to every
        row -- ranks candidates only; every decision coverage rests on is
        re-made exactly."""
        dots = (U.astype(np.float32) @ self._U[:self._L].astype(np.float32).T).astype(np.float64)
        return 2.0 * np.arcsin(np.minimum(1.0, 0.5 * np.sqrt(np.maximum(2.0 - 2.0 * dots, 0.0))))

    def _size(self, psi):
        """log of a cap's cost at accept angle psi, as
        SphereMergeConstruct._size: its measure on S^p (default p = dim - 1)
        or p ln psi."""
        if self.cost == "power":
            return (8.0 if self.exponent is None else self.exponent) * np.log(psi)
        p = self._U.shape[1] - 1.0 if self.exponent is None else self.exponent
        return log_cap_measure(np.minimum(psi, np.pi), p + 1.0)

    def _pair_costs(self, ta, tb, phi, ma, mb):
        """log of the increase of sum m^w size(theta + t) a merge causes."""
        tab = np.minimum(np.maximum(np.maximum(ta, tb), 0.5 * (phi + ta + tb)), np.pi)
        w, t = self.mass_weight, self.cost_offset
        ta, tb, tab = np.broadcast_arrays(ta, tb, tab)
        log_a = self._size(ta + t)
        log_b = self._size(tb + t)
        log_c = self._size(tab + t)
        if w:
            log_a = log_a + w * np.log(ma)
            log_b = log_b + w * np.log(mb)
            log_c = log_c + w * np.log(ma + mb)
        rest = np.exp(log_a - log_c) + np.exp(log_b - log_c)
        return log_c + np.log(np.maximum(1.0 - rest, 1e-12))

    def _costs_from(self, y: int, d: np.ndarray) -> np.ndarray:
        L = self._L
        c = self._pair_costs(self._t[y], self._t[:L], d, self._m[y], self._m[:L])
        c = np.where(self._alive[:L], c, np.inf)
        c[y] = np.inf
        return c

    def _rescan_rows(self, rows: np.ndarray, block: int = 256) -> None:
        L = self._L
        for i in range(0, len(rows), block):
            r = rows[i:i + block]
            d = self._dist(self._U[r])
            c = self._pair_costs(self._t[r, None], self._t[None, :L], d, self._m[r, None], self._m[None, :L])
            c = np.where(self._alive[None, :L], c, np.inf)
            c[np.arange(len(r)), r] = np.inf
            j = np.argmin(c, axis=1)
            cheapest = c[np.arange(len(r)), j]
            self._best[r] = np.where(np.isfinite(cheapest), j, -1)
            self._bcost[r] = cheapest
            self._stale[r] = False

    # ---------------------------------------------------------------- insert
    def insert(self, filter: Filter, vectors: np.ndarray) -> None:
        X = np.asarray(vectors, dtype=np.float64)
        if len(X) == 0:
            return
        self.ensure_capacity(len(filter._angle_codes))
        self._load(filter, extra=2 * len(X))
        # One vector at a time: every merge changes what the next one finds.
        for x in X:
            self._insert_one(filter, x)
        self._save()

    def _covering_row(self, x: np.ndarray) -> int:
        """Live cap holding x (nearest edge first), or -1. Screened in
        float32, decided in float64."""
        L = self._L
        if self._n_alive == 0:
            return -1
        d = self._dist(x[None, :])[0]
        near = np.flatnonzero(self._alive[:L] & (d <= self._t[:L] + 1e-3))
        if len(near) == 0:
            return -1
        exact = angle_between(np.broadcast_to(x, (len(near), len(x))), self._U[near])
        ok = exact <= self._t[near]
        return int(near[ok][np.argmin((exact - self._t[near])[ok])]) if ok.any() else -1

    def _insert_one(self, filter: Filter, x: np.ndarray) -> None:
        y = self._covering_row(x)
        if y >= 0:
            self._add_mass(y, 1.0)
            return
        sid, dec = filter._add_caps(x[None, :], np.zeros(1))
        self._append(filter, int(sid[0]), dec[0], 1.0)
        while self._n_alive > filter.max_caps:
            self._merge_cheapest(filter)

    def _add_mass(self, y: int, dm: float) -> None:
        """Every pair with y costs more now, so y's cached cost and those of
        the caps whose partner is y become lower bounds (mass_weight >= 1);
        below 1 a cost can fall with mass, so they are re-scanned outright."""
        self._m[y] += dm
        if not self.mass_weight:
            return    # no cost depends on the count
        L = self._L
        dependents = self._alive[:L] & (self._best[:L] == y)
        if self.mass_weight >= 1.0:
            self._stale[y] = True
            self._stale[:L] |= dependents
        else:
            self._rescan_rows(np.array([y, *np.flatnonzero(dependents)]))

    def _kill(self, filter: Filter, rows: np.ndarray) -> None:
        filter._remove_caps(self._ids[rows])
        self._alive[rows] = False
        self._bcost[rows] = np.inf
        self._n_alive -= len(rows)
        L = self._L
        self._stale[:L] |= self._alive[:L] & np.isin(self._best[:L], rows)

    def _append(self, filter: Filter, sid: int, direction: np.ndarray, m: float) -> None:
        if self._L >= len(self._ids):
            self._grow()
        y = self._L
        self._L += 1
        self._n_alive += 1
        self._ids[y] = sid
        self._U[y] = direction
        self._t[y] = float(filter._angle_value(filter._angle_codes[sid]))
        self._m[y] = m
        self._alive[y] = True
        self._stale[y] = False
        L = self._L
        d = self._dist(self._U[y][None, :])[0]
        # Caps wholly inside the new one hold nothing it doesn't: drop them
        # and keep their members' count. Screened in float32, decided exactly.
        near = np.flatnonzero(self._alive[:L] & (d + self._t[:L] <= self._t[y] + 1e-3))
        near = near[near != y]
        if len(near):
            exact = angle_between(self._U[near], np.broadcast_to(self._U[y], (len(near), self._U.shape[1])))
            inside = near[exact + self._t[near] <= self._t[y]]
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
            # Any stale cap whose lower bound is at or under the cheapest
            # known cost may hold the true minimum; none above it can.
            fresh = live & ~self._stale[:L]
            bound = np.min(np.where(fresh, self._bcost[:L], np.inf))
            rows = np.flatnonzero(live & self._stale[:L] & (self._bcost[:L] <= bound))
            self._rescan_rows(np.union1d(rows, [a]))
            return
        ua, ta, ub, tb = self._U[a], self._t[a], self._U[b], self._t[b]
        u, theta = enclosing_cap(ua, ta, ub, tb)
        m = self._m[a] + self._m[b]
        self._kill(filter, np.array([a, b]))
        # The merged cap's code, fitted to hold the two parts within the
        # least stored angle (CapCoverFilter.fit_cap_code), or u's own.
        codes = filter.fit_cap_code(u, np.stack([ua, ub]), np.array([ta, tb])) if self.fit_codes else None
        sid, dec = filter._add_caps(u[None, :], np.array([theta]), codes=codes)
        sid, dec = int(sid[0]), dec[0]
        # Around the decoded direction, from the parts' own caps: tighter
        # than _add_caps' theta + angle(u, dec), and as sound.
        held = max(float(angle_between(dec[None, :], ua[None, :])[0]) + ta,
                   float(angle_between(dec[None, :], ub[None, :])[0]) + tb)
        filter._set_angles(np.array([sid]), np.array([min(held, np.pi)]))
        self._append(filter, sid, dec, m)
