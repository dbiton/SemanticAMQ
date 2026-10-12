"""VAPlusFilter -- the VA+-file (Ferhatosmanoglu, Tuncel, Agrawal, El Abbadi,
CIKM 2000, doi:10.1145/354756.354820) as a zero-false-negative membership
filter: the database literature's per-vector approximation, run as published.

VA+ as published is filter-and-refine for exact k-NN:
  1. KLT/PCA rotation, which concentrates variance in the leading dims and
     preserves every L2 distance;
  2. a total of B bits per vector handed out greedily, one bit at a time, to
     the dimension with the largest remaining quantization error
     var_j / 4^b_j (low-variance dims get 0 bits);
  3. non-uniform cells per dimension from 1-D Lloyd-Max, boundaries at the
     midpoints between centroids, outer boundaries at the data min/max;
  4. each vector stored as one cell index per dimension -- a box in PCA space;
  5. a query lower-bounds its distance to every box and refines survivors.

As an AMQ there is no refine step and no stored vector: accept iff
min_i LB(q, box_i) <= tau. Zero-FN because every x lies in its own box, so
LB(q, box(x)) <= d(q, x) -- the triangle-inequality argument BallCoverFilter
makes for balls, with boxes instead.

Kept to the published method on purpose: it is a baseline, so nothing here
goes beyond the paper (no truncated KLT, no alternative cell schemes, no
per-budget tuning). The only choices the paper leaves open are the ones a
membership filter under a byte budget forces:
  - B is the largest bit count whose whole footprint fits max_memory, and
    the allocation for B is the paper's greedy (the first B picks of one
    greedy sequence, so it does not depend on the budget otherwise);
  - everything read at query time is charged: packed codes (n * B bits),
    cell boundaries (sum_j (2^b_j + 1) fp32), and the full dim x dim fp32
    KLT matrix plus mean. The KLT is data-dependent, so unlike
    BallCoverFilter's and QuantizedVectorFilter's random rotations it cannot
    be regenerated from a seed. At dim=384 the matrix alone is 590 KB, so
    this filter is infeasible below roughly that budget. No vector ids --
    the same fairness rule as QuantizedVectorFilter.

Numerics. The stored fp32 rows are re-orthonormalized in float64 (QR) on
both the build and query paths; boxes are assigned from the same float64
coordinates the query path computes, against the same stored fp32
boundaries (outer ones rounded outward), so each x is inside its own box
exactly. What is left is the float noise between this filter's distances and
the ground truth's, covered by the same _COVERAGE_EPSILON margin
BallCoverFilter and QuantizedVectorFilter use."""

from typing import List, Optional, Tuple

import numpy as np

from filters.filter import Filter, InsufficientMemoryError

_BYTES_FP32 = 4

# 1-D Lloyd-Max: iterations, and the per-dimension sample it trains on.
# Boundaries only shape the cells; every point is still assigned to its own
# cell afterwards, so training on a sample cannot cost a false negative.
_LLOYD_ITERS = 25
_LLOYD_SAMPLE = 20_000

# Query layout: dims are packed into byte-sized groups so one table lookup
# covers up to 8 bits (a 1-bit dim costs 1/8 of a gather, not one). Survivors
# are pruned every _PRUNE_EVERY groups -- a partial LB^2 sum is already a
# lower bound, so a point past the threshold can be dropped early. Neither
# changes any answer, only the speed.
_GROUP_BITS = 8
_PRUNE_EVERY = 4
_QUERY_BLOCK = 64


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


# Layout accounting: one formula used both to size B against the budget and,
# after construction, to report get_bytesize(), so they can't drift.

def _footprint_bytes(n: int, dim: int, bits: np.ndarray) -> int:
    matrix = (dim * dim + dim) * _BYTES_FP32  # KLT + mean
    boundaries = int(np.sum(np.left_shift(1, bits) + 1)) * _BYTES_FP32
    codes = _ceil_div(n * int(np.sum(bits)), 8)
    return matrix + boundaries + codes


def _greedy_sequence(variances: np.ndarray, max_bits_per_dim: int) -> np.ndarray:
    """The dim each successive bit goes to under VA+'s greedy rule (largest
    var_j / 4^b_j; ties to the lower index, i.e. the higher-variance dim),
    until every dim holds max_bits_per_dim. The allocation for B bits is the
    first B entries."""
    var = np.maximum(np.asarray(variances, dtype=np.float64), 0.0)
    bits = np.zeros(len(var), dtype=np.int64)
    seq = np.empty(len(var) * max_bits_per_dim, dtype=np.int64)
    for i in range(len(seq)):
        j = int(np.argmax(np.where(bits < max_bits_per_dim, var / 4.0 ** bits, -1.0)))
        seq[i] = j
        bits[j] += 1
    return seq


def _allocate_bits(variances: np.ndarray, n: int, dim: int, max_bits_per_dim: int,
                   max_memory: Optional[int]) -> np.ndarray:
    """VA+'s allocation at the largest B whose footprint fits max_memory
    (every B when max_memory is None). The footprint only grows with B, so
    the first B that does not fit ends the scan."""
    seq = _greedy_sequence(variances, max_bits_per_dim)
    bits = np.zeros(len(variances), dtype=np.int64)
    for j in seq:
        bits[j] += 1
        if max_memory is not None and _footprint_bytes(n, dim, bits) > max_memory:
            bits[j] -= 1
            break
    return bits


def _round_out(lo: float, hi: float) -> Tuple[np.float32, np.float32]:
    """fp32 bounds that still contain [lo, hi] when compared in float64."""
    lo32, hi32 = np.float32(lo), np.float32(hi)
    if float(lo32) > lo:
        lo32 = np.nextafter(lo32, np.float32(-np.inf))
    if float(hi32) < hi:
        hi32 = np.nextafter(hi32, np.float32(np.inf))
    return lo32, hi32


def _cell_boundaries(y: np.ndarray, num_cells: int, rng: np.random.Generator) -> np.ndarray:
    """num_cells + 1 non-decreasing fp32 boundaries covering all of y: 1-D
    Lloyd-Max, inner boundaries at the midpoints between centroids."""
    lo, hi = _round_out(float(y.min()), float(y.max()))
    if num_cells == 1:
        return np.array([lo, hi], dtype=np.float32)
    sample = y if len(y) <= _LLOYD_SAMPLE else rng.choice(y, _LLOYD_SAMPLE, replace=False)
    s = np.sort(sample)
    c = np.quantile(s, (np.arange(num_cells) + 0.5) / num_cells)
    for _ in range(_LLOYD_ITERS):
        assign = np.searchsorted((c[:-1] + c[1:]) / 2, s, side="right")
        count = np.bincount(assign, minlength=num_cells)
        total = np.bincount(assign, weights=s, minlength=num_cells)
        # An empty cell keeps its centroid rather than collapsing.
        new_c = np.sort(np.where(count > 0, total / np.maximum(count, 1), c))
        if np.array_equal(new_c, c):
            break
        c = new_c
    inner = np.maximum.accumulate(np.clip(((c[:-1] + c[1:]) / 2).astype(np.float32), lo, hi))
    return np.concatenate([[lo], inner, [hi]]).astype(np.float32)


def _assign_cells(y: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Cell c with bounds[c] <= y <= bounds[c + 1], compared in float64."""
    return np.searchsorted(bounds[1:-1].astype(np.float64), y, side="right")


def _interval_dist_sq(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Squared distance from each x (rows) to each [lo, hi] (columns)."""
    x = x[:, None]
    gap = np.maximum(np.maximum(lo[None, :] - x, x - hi[None, :]), 0.0)
    return gap * gap


def _orthonormal_rows(components: np.ndarray) -> np.ndarray:
    """float64 orthonormal basis of the stored fp32 rows' span, signs matched
    to the rows. Derived only from what is stored, so build and query agree."""
    q, r = np.linalg.qr(components.astype(np.float64).T)
    return (q * np.where(np.diag(r) < 0, -1.0, 1.0)).T


class VAPlusFilter(Filter):
    """Published VA+-file as a membership filter (see module docstring).

    tau is a pure query-time knob, as for QuantizedVectorFilter: the boxes
    depend only on the data and max_memory.
    """

    # See BallCoverFilter._COVERAGE_EPSILON / QuantizedVectorFilter's copy.
    _COVERAGE_EPSILON = 1e-3

    def __init__(
        self,
        dim: int,
        distance_threshold: float,
        max_memory: Optional[int] = None,
        max_bits_per_dim: int = 8,
        seed: int = 42,
    ):
        super().__init__(dim, max_memory=max_memory)
        self.distance_threshold = float(distance_threshold)
        if not 1 <= max_bits_per_dim <= _GROUP_BITS:
            raise ValueError(f"max_bits_per_dim must be in [1, {_GROUP_BITS}], got {max_bits_per_dim}")
        self.max_bits_per_dim = max_bits_per_dim
        self.seed = seed

        self._n = 0
        self.bits: Optional[np.ndarray] = None        # (dim,) bits per PCA dim
        self.mean: Optional[np.ndarray] = None        # (dim,) fp32, stored
        self.components: Optional[np.ndarray] = None  # (dim, dim) fp32 KLT rows, stored
        self._basis: Optional[np.ndarray] = None      # float64 orthonormal rows of components
        self.boundaries: List[np.ndarray] = []        # per dim, 2^b_j + 1 fp32
        self._groups: List[np.ndarray] = []           # dim indices packed into each code byte
        self._group_codes: List[np.ndarray] = []      # (n,) uint8 per group
        self._constructed = False


    def _check_floor(self, n: int) -> None:
        """Raise before any PCA work if not even one code bit fits."""
        if self.max_memory is None:
            return
        one_bit = np.zeros(self.dim, dtype=np.int64)
        one_bit[0] = 1
        floor = _footprint_bytes(n, self.dim, one_bit)
        if floor > self.max_memory:
            raise InsufficientMemoryError(
                f"max_memory={self.max_memory} bytes is too small for VAPlusFilter with n={n} vectors of "
                f"dim={self.dim}: one code bit per vector needs {floor} bytes (the {self.dim}x{self.dim} "
                f"fp32 KLT matrix alone is {self.dim * self.dim * _BYTES_FP32})."
            )

    def _project(self, vectors: np.ndarray) -> np.ndarray:
        """KLT coordinates (n, dim), float64."""
        centered = np.asarray(vectors, dtype=np.float32).astype(np.float64) - self.mean.astype(np.float64)
        return centered @ self._basis.T


    def construct(self, vectors: np.ndarray) -> None:
        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        n = len(vecs)
        self._n = n
        self._constructed = True
        if n == 0:
            self.bits = None
            return
        self._check_floor(n)

        x = vecs.astype(np.float64)
        mean = x.mean(axis=0)
        centered = x - mean
        evals, evecs = np.linalg.eigh(centered.T @ centered / n)
        order = np.argsort(evals)[::-1]

        self.bits = _allocate_bits(evals[order], n, self.dim, self.max_bits_per_dim, self.max_memory)
        self.mean = mean.astype(np.float32)
        self.components = np.ascontiguousarray(evecs[:, order].T, dtype=np.float32)
        self._basis = _orthonormal_rows(self.components)

        coords = self._project(vecs)
        rng = np.random.default_rng(self.seed)
        self.boundaries = [_cell_boundaries(coords[:, j], 1 << int(b), rng) for j, b in enumerate(self.bits)]
        cells = [_assign_cells(coords[:, j], self.boundaries[j]) for j in range(self.dim)]

        # Pack coded dims, in variance order, into byte groups, big-endian so
        # a group's code indexes its C-order outer-sum table (see _query_tables).
        self._groups, self._group_codes = [], []
        group, group_bits = [], 0
        for j in [int(j) for j in np.flatnonzero(self.bits)] + [None]:
            if j is None or group_bits + int(self.bits[j]) > _GROUP_BITS:
                if group:
                    code = np.zeros(n, dtype=np.int64)
                    for d in group:
                        code = (code << int(self.bits[d])) | cells[d]
                    self._groups.append(np.array(group))
                    self._group_codes.append(code.astype(np.uint8))
                group, group_bits = [], 0
            if j is not None:
                group.append(j)
                group_bits += int(self.bits[j])

    def cell_ids(self) -> np.ndarray:
        """(n, dim) cell index of every stored vector, decoded from the packed
        groups (0 for 0-bit dims)."""
        cells = np.zeros((self._n, self.dim), dtype=np.int64)
        for dims, code in zip(self._groups, self._group_codes):
            code = code.astype(np.int64)
            for d in dims[::-1]:
                b = int(self.bits[d])
                cells[:, d] = code & ((1 << b) - 1)
                code = code >> b
        return cells


    def _query_tables(self, q: np.ndarray) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
        """For a query block: the LB^2 every stored box shares (0-bit dims,
        one cell each), shape (Qb,), and one (codes, table) pair per byte
        group. table has shape (Qb, 2^bits) and table[i, code] is that code's
        LB^2 contribution for query i."""
        coords = self._project(q)
        per_dim = {}
        base = np.zeros(len(q), dtype=np.float64)
        for j, bounds in enumerate(self.boundaries):
            b64 = bounds.astype(np.float64)
            t = _interval_dist_sq(coords[:, j], b64[:-1], b64[1:])
            if self.bits[j] == 0:
                base += t[:, 0]
            else:
                per_dim[j] = t

        lookups = []
        for dims, codes in zip(self._groups, self._group_codes):
            table = per_dim[int(dims[0])]
            for d in dims[1:]:
                table = (table[:, :, None] + per_dim[int(d)][:, None, :]).reshape(len(q), -1)
            lookups.append((codes, table))
        return base, lookups

    def min_lower_bound_sq(self, vectors: np.ndarray) -> np.ndarray:
        """min over stored boxes of LB^2, per query, with no pruning -- the
        quantity query() thresholds. For tests and diagnostics."""
        out = np.empty(len(vectors), dtype=np.float64)
        for start in range(0, len(vectors), _QUERY_BLOCK):
            base, lookups = self._query_tables(vectors[start:start + _QUERY_BLOCK])
            for i in range(len(base)):
                acc = np.full(self._n, base[i])
                for codes, table in lookups:
                    acc += table[i][codes]
                out[start + i] = acc.min()
        return out

    def query(self, vectors: np.ndarray, tau: Optional[float] = None) -> List[bool]:
        if not self._constructed:
            raise ValueError("VAPlusFilter has not been constructed.")
        if tau is None:
            tau = self.distance_threshold
        if self._n == 0 or len(vectors) == 0:
            return [False] * len(vectors)

        q = np.ascontiguousarray(vectors, dtype=np.float32)
        thresh_sq = (float(tau) + self._COVERAGE_EPSILON) ** 2
        hits = np.zeros(len(q), dtype=bool)
        for start in range(0, len(q), _QUERY_BLOCK):
            base, lookups = self._query_tables(q[start:start + _QUERY_BLOCK])
            for i in range(len(base)):
                hits[start + i] = self._any_within(base[i], lookups, i, thresh_sq)
        return hits.tolist()

    def _any_within(self, base: float, lookups, i: int, thresh_sq: float) -> bool:
        if base > thresh_sq:
            return False
        survivors = None  # None = every stored vector
        acc = np.full(self._n, base)
        for g, (codes, table) in enumerate(lookups, start=1):
            acc += table[i][codes if survivors is None else codes[survivors]]
            if g % _PRUNE_EVERY == 0 or g == len(lookups):
                keep = acc <= thresh_sq
                if not keep.any():
                    return False
                survivors = np.flatnonzero(keep) if survivors is None else survivors[keep]
                acc = acc[keep]
        return True


    def get_bytesize(self) -> int:
        if not self._constructed or self.bits is None:
            return 0
        return _footprint_bytes(self._n, self.dim, self.bits)
