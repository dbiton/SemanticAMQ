"""IVFRadiiFilter -- semantic AMQ baseline built from an IVF index's coarse
quantizer alone: the nlist k-means centroids plus one covering radius per
inverted list, with the lists' contents thrown away.

List i keeps r_i = max ||x - c_i|| over the construct vectors IVF assigned
to it. By the triangle inequality, any x in list i satisfies
||q - x|| >= ||q - c_i|| - r_i, so a list whose ||q - c_i|| - r_i exceeds tau
provably holds nothing within tau of q. query() therefore answers
"duplicate" iff some list survives that test:

    min_i (||q - c_i|| - r_i) <= tau

Unlike IVFFilter, which probes only the nprobe nearest cells and so can miss
a near neighbor that landed in a farther one, this probes every list's
bound, so it has **zero false negatives** by construction. There is no
nprobe dial: any probe limit would reintroduce misses.

insert() follows IVF's own add(): assign the vector to its nearest centroid
and grow that list's radius to cover it. Centroids never move, so the
footprint is fixed after construct() and zero-FN holds over a whole stream.

Memory: nlist float32 centroids (the serialized IndexFlatL2 coarse
quantizer) plus nlist float32 radii. The budget sets nlist directly, at most
n. Lists k-means left empty cover nothing and are dropped rather than
charged. If not even one list fits, construct() raises
InsufficientMemoryError.
"""

from typing import List, Optional

import faiss
import numpy as np

from filters.filter import Filter, InsufficientMemoryError

_BYTES_PER_RADIUS = 4

# faiss's k-means default; matches what an IndexIVF's own train() runs.
_KMEANS_NITER = 25

# Query batch size cap, in (query, list) pairs: an all-lists search returns
# an nq x nlist distance matrix, so unbatched it grows to GBs per job.
_SEARCH_CELLS = 1 << 22


class IVFRadiiFilter(Filter):
    """IVF coarse quantizer + per-list covering radii as a zero-FN semantic
    AMQ filter. tau is a pure query-time knob: the structure depends only on
    dim/n/max_memory."""

    # Same margin and reason as BallCoverFilter._COVERAGE_EPSILON: faiss
    # distances are not bit-reproducible across query-batch shapes, and the
    # sqrt of a near-zero squared distance amplifies that error.
    _COVERAGE_EPSILON = 1e-3

    def __init__(
        self,
        dim: int,
        distance_threshold: float,
        max_memory: Optional[int] = None,
        nlist: Optional[int] = None,
        seed: int = 42,
    ):
        super().__init__(dim, max_memory=max_memory)
        self.distance_threshold = float(distance_threshold)
        self._nlist_override = nlist
        self.seed = seed

        self.nlist: Optional[int] = None
        self.quantizer: Optional[faiss.IndexFlatL2] = None
        self.radii: Optional[np.ndarray] = None  # float32, one per centroid in quantizer
        self._constructed = False

    def _header_bytes(self) -> int:
        """Serialized size of an empty IndexFlatL2 of this dim: everything
        the quantizer costs beyond its centroids. Measured, not assumed."""
        return len(faiss.serialize_index(faiss.IndexFlatL2(self.dim)))

    def bytes_per_list(self) -> int:
        return self.dim * 4 + _BYTES_PER_RADIUS

    def _budget_nlist(self) -> int:
        """Most lists max_memory pays for (may be < 1)."""
        return (self.max_memory - self._header_bytes()) // self.bytes_per_list()

    def _choose_nlist(self, n: int) -> int:
        if self._nlist_override is not None:
            nlist = self._nlist_override
        elif self.max_memory is not None:
            nlist = self._budget_nlist()
        else:
            # IVF's usual sqrt(n) rule when nothing else constrains it.
            nlist = round(n ** 0.5)
        return max(1, min(int(nlist), n))

    def construct(self, vectors: np.ndarray) -> None:
        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        n = len(vecs)
        self._constructed = True
        self.quantizer, self.radii, self.nlist = None, None, 0
        if n == 0:
            return

        if self.max_memory is not None and self._budget_nlist() < 1:
            raise InsufficientMemoryError(
                f"max_memory={self.max_memory} bytes is too small for IVFRadiiFilter: one "
                f"list costs {self.bytes_per_list()} bytes plus a {self._header_bytes()}-byte "
                f"quantizer header."
            )

        nlist = self._choose_nlist(n)
        if nlist == n:
            # Every vector its own list: k-means would only reproduce them.
            centroids = vecs.copy()
        else:
            kmeans = faiss.Kmeans(
                self.dim, nlist, niter=_KMEANS_NITER, seed=self.seed,
                min_points_per_centroid=1,
            )
            kmeans.train(vecs)
            centroids = kmeans.centroids

        quantizer = faiss.IndexFlatL2(self.dim)
        quantizer.add(centroids)
        d2, assign = quantizer.search(vecs, 1)
        sq_radii = np.full(nlist, -1.0, dtype=np.float64)
        np.maximum.at(sq_radii, assign[:, 0], np.maximum(d2[:, 0].astype(np.float64), 0.0))

        keep = sq_radii >= 0.0
        self.quantizer = faiss.IndexFlatL2(self.dim)
        self.quantizer.add(np.ascontiguousarray(centroids[keep]))
        self.radii = self._round_up_f32(np.sqrt(sq_radii[keep]))
        self.nlist = int(keep.sum())

        if self.max_memory is not None and self.get_bytesize() > self.max_memory:
            actual = self.get_bytesize()
            self.quantizer, self.radii, self.nlist = None, None, 0
            raise InsufficientMemoryError(
                f"IVFRadiiFilter built {actual} bytes against max_memory={self.max_memory}."
            )

    @staticmethod
    def _round_up_f32(values: np.ndarray) -> np.ndarray:
        """float64 -> float32 never rounding a radius down, which would
        shrink a list's ball below a point it must cover."""
        out = values.astype(np.float32)
        low = out.astype(np.float64) < values
        out[low] = np.nextafter(out[low], np.float32(np.inf))
        return out

    def surface_distance(self, vectors: np.ndarray) -> np.ndarray:
        """min_i (||q - c_i|| - r_i) per query: a lower bound on the distance
        to the nearest construct vector, np.inf with no lists."""
        if not self._constructed:
            raise ValueError("IVFRadiiFilter has not been constructed.")
        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        if len(vecs) == 0 or not self.nlist:
            return np.full(len(vecs), np.inf)
        out = np.empty(len(vecs))
        batch = max(1, _SEARCH_CELLS // self.nlist)
        for start in range(0, len(vecs), batch):
            d2, ids = self.quantizer.search(vecs[start:start + batch], self.nlist)
            dist = np.sqrt(np.maximum(d2.astype(np.float64), 0.0))
            surface = np.where(ids >= 0, dist - self.radii[np.maximum(ids, 0)], np.inf)
            out[start:start + batch] = surface.min(axis=1)
        return out

    def query(self, vectors: np.ndarray, tau: Optional[float] = None) -> List[bool]:
        if tau is None:
            tau = self.distance_threshold
        return (self.surface_distance(vectors) <= float(tau) + self._COVERAGE_EPSILON).tolist()

    def insert(self, vectors: np.ndarray) -> None:
        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        if len(vecs) == 0:
            return
        if not self.nlist:
            self.construct(vecs)
            return
        d2, assign = self.quantizer.search(vecs, 1)
        grown = self._round_up_f32(np.sqrt(np.maximum(d2[:, 0].astype(np.float64), 0.0)))
        np.maximum.at(self.radii, assign[:, 0], grown)

    def get_bytesize(self) -> int:
        if self.quantizer is None:
            return 0
        return len(faiss.serialize_index(self.quantizer)) + self.radii.nbytes
