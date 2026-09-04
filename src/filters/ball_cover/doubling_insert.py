import numpy as np
import faiss
from typing import List, Tuple
from filters.filter import Filter
from filters.ball_cover.ball_interfaces import InsertInterface


class DoublingInsert(InsertInterface):
    """Amortized-rebuild insertion: a newly-arrived, not-already-covered raw
    vector is added immediately as its own zero-radius ball (cheap -- an
    array append, no merge work), letting the filter's *active* ball count
    grow past `filter.max_spheres` between rebuilds instead of being kept at
    a hard cap on every insertion the way NearestInsert/LeastInflationInsert/
    MinHeapInsert/ExhaustiveMergeInsert all do. Once the active count reaches
    `max_spheres * (1 + growth_budget_fraction)`, a single Gonzalez-style
    re-clustering pass folds every currently-active ball -- old, already-
    summarized ones and the raw zero-radius singletons added since the last
    rebuild alike -- back down to at most `max_spheres` balls. Like a dynamic
    array's doubling reallocation, this makes amortized per-insertion cost
    O(1) (a coverage check + an append) at the price of an infrequent
    O((max_spheres * (1 + growth_budget_fraction))^2 * d) rebuild spike,
    trading the other strategies' steady per-point merge cost for occasional
    latency spikes -- the point being to see whether a periodic *global*
    re-optimization reaches better steady-state coverage quality than any
    sequence of local, greedy per-point decisions can.

    Every newly-added ball is immediately part of the filter's own queryable
    structure (filter._all_centers/_all_radiuses/_all_spheres), so the zero-
    false-negative guarantee holds at every single checkpoint, not just right
    after a rebuild -- unlike keeping raw vectors in separate, query()-
    invisible storage until the next rebuild fires.

    growth_budget_fraction is set at construction (__init__) and is exactly
    the "how much memory is reserved for this" knob: since insertion
    strategy objects are instantiated independently of (and normally before)
    the BallCoverFilter that will use them, this can't be an absolute byte
    budget -- filter.max_memory/max_spheres aren't known yet. It's instead a
    multiplier on filter.max_spheres, resolved lazily against whatever
    max_spheres the filter that ends up owning this strategy turns out to
    have. The default, 1.0, is what makes this "doubling": the active count
    may grow to 2x max_spheres (max_spheres old balls plus up to
    max_spheres freshly-added singletons) before a rebuild is forced. A
    smaller fraction triggers more frequent, cheaper rebuilds; a larger one
    defers longer, amortizing each rebuild's cost over more insertions at
    the price of a bigger latency spike and a bigger transient memory
    footprint (reflected automatically in filter.get_bytesize(), since the
    extra balls live in the filter's own arrays -- this strategy needs no
    separate byte accounting of its own).

    Rebuild construction, and why it isn't just GonzalesConstruct: a rebuild
    must preserve coverage of every point absorbed by the *current* balls,
    not just the freshly-added raw singletons -- but those original points
    were never retained (that's the whole point of a covering-ball summary).
    Each currently-active ball's center is fed into the same greedy
    farthest-first selection, carrying its own existing radius as an
    "uncertainty" allowance: the true points it once covered could be
    anywhere within that radius, so a rebuilt ball absorbing that center
    must have room to cover that whole region, not just the single center
    point. Freshly-added singletons carry radius (and therefore uncertainty)
    0, since they're genuine, exact points -- so this is a single uniform
    rule ("use each active ball's own current radius as its uncertainty"),
    not a special case. This is the same reconstruction-error "inflate the
    radius by however much positional uncertainty exists" technique
    BallCoverFilter._build_quantizer already uses for quantized centers, just
    applied to old, already-summarized ball centers. GonzalesConstruct's own
    construct_filter has no parameter for this per-point uncertainty (and
    ConstructInterface is shared with IVFConstruct/RPTreeConstruct, so
    changing its signature would ripple into unrelated construction
    strategies), so the greedy selection + exact-radius algorithm is mirrored
    here directly rather than reused, with the one addition (per-point
    uncertainty added into the final radius) that correctness requires.
    """

    def __init__(self, growth_budget_fraction: float = 1.0):
        if growth_budget_fraction <= 0.0:
            raise ValueError(
                f"growth_budget_fraction={growth_budget_fraction} must be > 0 "
                "(it sizes the rebuild-trigger threshold as a multiple of max_spheres)."
            )
        self.growth_budget_fraction = growth_budget_fraction

    # ------------------------------------------------------------------ InsertInterface plumbing

    def ensure_capacity(self, required_cap: int) -> None:
        # No strategy-local state beyond the filter's own sphere arrays,
        # which already grow themselves via filter._get_new_index().
        pass

    def reset_buffers(self, capacity: int) -> None:
        pass

    def add_sphere(self, filter: Filter, center: np.ndarray, radius: float) -> None:
        idx = filter._get_new_index()
        filter._all_centers[idx] = center
        filter._all_radiuses[idx] = radius
        filter._num_spheres += 1

    # ------------------------------------------------------------------ Gonzalez re-clustering

    @staticmethod
    def _gonzalez_select_with_uncertainty(
        points: np.ndarray, uncertainty: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Gonzalez farthest-first center selection over `points`, each
        carrying its own pre-existing uncertainty radius (see class
        docstring). Mirrors GonzalesConstruct.construct_filter's own
        greedy-selection-then-exact-radii algorithm; the only change is
        `reach = distance-to-center + that point's own uncertainty` instead
        of plain distance when computing each cluster's final radius."""
        n = points.shape[0]
        dim = points.shape[1]
        k = min(k, n)
        if k <= 0:
            return np.empty((0, dim), dtype=np.float32), np.empty(0, dtype=np.float32)
        if n <= k:
            return points.copy(), uncertainty.copy()

        sq = np.einsum("ij,ij->i", points, points)

        chosen = np.empty(k, dtype=np.int64)
        min_d2 = np.empty(n, dtype=np.float32)
        d2 = np.empty(n, dtype=np.float32)

        chosen[0] = 0
        np.dot(points, points[0], out=min_d2)
        min_d2 *= -2.0
        min_d2 += sq
        min_d2 += sq[0]

        for j in range(1, k):
            c = int(min_d2.argmax())
            if min_d2[c] <= 0.0:
                chosen = chosen[:j]
                k = j
                break
            chosen[j] = c

            np.dot(points, points[c], out=d2)
            d2 *= -2.0
            d2 += sq
            d2 += sq[c]
            np.minimum(min_d2, d2, out=min_d2)

        centers = np.ascontiguousarray(points[chosen])

        index = faiss.IndexFlatL2(dim)
        index.add(centers)
        d2_exact, labels = index.search(points, 1)
        d_exact = np.sqrt(np.maximum(d2_exact.ravel(), 0.0))
        labels = labels.ravel()

        # Each assigned point's contribution to its cluster's radius must
        # cover the *region* it stands for, not just itself: distance to
        # the new center, plus that point's own prior uncertainty radius.
        reach = (d_exact + uncertainty).astype(np.float32)

        order = np.argsort(labels, kind="stable")
        counts = np.bincount(labels, minlength=k)
        starts = np.zeros(k, dtype=np.int64)
        np.cumsum(counts[:-1], out=starts[1:])
        nonempty = counts > 0

        radii = np.zeros(k, dtype=np.float32)
        radii[nonempty] = np.maximum.reduceat(reach[order], starts[nonempty])
        return centers, radii

    def _rebuild_threshold(self, filter: Filter) -> int:
        return max(
            filter.max_spheres + 1,
            int(round(filter.max_spheres * (1.0 + self.growth_budget_fraction))),
        )

    def _rebuild(self, filter: Filter) -> None:
        points = np.ascontiguousarray(filter.all_centers, dtype=np.float32)
        uncertainty = np.asarray(filter.all_radiuses, dtype=np.float32)

        new_centers, new_radii = self._gonzalez_select_with_uncertainty(
            points, uncertainty, filter.max_spheres
        )

        # Retire every currently-active sphere (fold the whole active set
        # back to the front-of-array free list) before re-adding the freshly
        # computed set via the same add_sphere() every other strategy uses
        # for a from-scratch build -- mirrors how MinHeapInsert._merge_spheres
        # retires exactly the merged pair before re-adding the merge result,
        # just for the entire active set at once here.
        filter._num_spheres = 0
        for center, radius in zip(new_centers, new_radii):
            self.add_sphere(filter, center, radius)
        filter._rebuild_from_centers(filter.all_spheres, filter.all_centers, filter.all_radiuses)

    # ------------------------------------------------------------------ InsertInterface: insert/remove

    @staticmethod
    def _exhaustive_search(centers: np.ndarray, queries: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        dim = centers.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(centers)
        d2, labels = index.search(queries, len(centers))
        return np.sqrt(np.maximum(d2, 0.0)), labels

    def insert(self, filter: Filter, vectors: np.ndarray) -> None:
        X = np.ascontiguousarray(vectors, dtype=np.float32)

        centers = filter.all_centers
        radii = filter.all_radiuses
        # _exhaustive_search (FAISS index.search) returns dist/labels sorted
        # by distance, with labels giving each match's original position in
        # `centers` -- NOT the same order as `centers`/`radii` themselves.
        # Comparing dist directly against radii[None, :] would pair each
        # distance with the wrong ball's radius (whichever one happens to
        # sit at that sorted rank), silently marking points "covered" by
        # radii that have nothing to do with the actual nearest ball.
        dist, labels = self._exhaustive_search(centers, X)
        is_already_covered = np.any(dist <= radii[labels], axis=1)
        X = X[~is_already_covered]

        for vec in X:
            self.add_sphere(filter, vec, 0.0)

        if filter._num_spheres >= self._rebuild_threshold(filter):
            self._rebuild(filter)

    def remove(self, filter: Filter, vectors: np.ndarray) -> List[bool]:
        raise NotImplementedError("DoublingInsert does not support remove operation.")

    def get_bytesize(self) -> int:
        return 0
