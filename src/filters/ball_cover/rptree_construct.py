import numpy as np
from typing import Tuple
from filters.filter import Filter
from filters.ball_cover.ball_interfaces import ConstructInterface


class RPTreeConstruct(ConstructInterface):
    """Random Projection Tree strategy for enclosing ball construction."""

    def __init__(self, seed: int = 0):
        self.seed = seed


    def construct_filter(self, filter: Filter, vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = np.ascontiguousarray(vectors, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"expected a 2-D array, got shape {X.shape}")
        n, dim = X.shape

        k = min(filter.max_construct_spheres, n)
        if k <= 0:
            return np.empty((0, dim), dtype=np.float32), np.empty(0, dtype=np.float32)
        if n <= k:
            return X.copy(), np.zeros(n, dtype=np.float32)

        depth = int(np.floor(np.log2(k)))
        rng = np.random.default_rng(self.seed)

        # One GEMM covers every split in the tree. Per-node direction diversity comes
        # from each node drawing an index into the bank rather than a fresh vector.
        _BANK = 32
        bank = rng.standard_normal((dim, _BANK)).astype(np.float32)
        bank /= np.linalg.norm(bank, axis=0, keepdims=True)
        proj_bank = X @ bank

        rows = np.arange(n)
        seg = np.zeros(n, dtype=np.int64)
        n_seg = 1

        for _ in range(depth):
            direction = rng.integers(0, _BANK, size=n_seg)
            proj = proj_bank[rows, direction[seg]]

            # Group by segment, ordered by projection within each segment.
            order = np.lexsort((proj, seg))
            seg_sorted = seg[order]
            counts = np.bincount(seg_sorted, minlength=n_seg)
            starts = np.zeros(n_seg, dtype=np.int64)
            np.cumsum(counts[:-1], out=starts[1:])

            rank = rows - starts[seg_sorted]  # position within segment
            goes_right = rank >= (counts // 2)[seg_sorted]
            seg[order] = 2 * seg_sorted + goes_right
            n_seg *= 2

        order = np.argsort(seg, kind="stable")
        counts = np.bincount(seg, minlength=n_seg)
        starts = np.zeros(n_seg, dtype=np.int64)
        np.cumsum(counts[:-1], out=starts[1:])
        nonempty = counts > 0
        offsets, sizes = starts[nonempty], counts[nonempty]

        Xs = X[order]  # grouped by leaf, so leaves are contiguous spans
        centers = np.add.reduceat(Xs, offsets, axis=0) / sizes[:, None]
        centers = np.ascontiguousarray(centers, dtype=np.float32)

        leaf_of = np.repeat(np.arange(sizes.size), sizes)
        diff = Xs - centers[leaf_of]
        d2 = np.einsum("ij,ij->i", diff, diff)

        radii = np.maximum.reduceat(d2, offsets)
        np.sqrt(radii, out=radii)
        return centers, radii.astype(np.float32, copy=False)


    def get_bytesize(self) -> int:
        return 0