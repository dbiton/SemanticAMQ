"""
QuantizedVectorFilter -- per-point quantization baseline: the "compress the
vectors" counterpart to BallCoverFilter's "compress the geometry". Keeps
every construct vector, quantized to whichever scheme on a fixed fidelity
ladder fits the memory budget, finest first: float16, 8-bit scalar, 4-bit
scalar, then RaBitQ (~1 bit/dim, no codebook). Exists to find the dataset
size at which BallCoverFilter's geometric summary wins over storing
(compressed) points directly.

Zero-false-negative by the same triangle-inequality argument
BallCoverFilter._build_quantizer uses: d(q,x) <= tau implies
d(q,x_hat) <= tau + ||x - x_hat||, so accepting whenever
d(q, x_hat_i) <= tau + eps_i, with eps_i the *measured* reconstruction error
(never a probabilistic bound), can never produce a false negative -- a
coarse level costs false-positive margin instead.

If not even RaBitQ fits max_memory for n vectors, this raises
InsufficientMemoryError rather than silently dropping vectors or
subsampling: that's the regime where only a geometric summary can answer the
query at all (see runner.py::equal_memory_worker's handling of it).
"""

from enum import Enum
from typing import List, Optional, Tuple

import faiss
import numpy as np

from filters.filter import Filter


class QuantizationLevel(Enum):
    FP16 = "fp16"
    SQ8 = "sq8"
    SQ4 = "sq4"
    RABITQ = "rabitq"


class ErrorBudgetMode(Enum):
    GLOBAL = "global"
    PER_CLUSTER = "per_cluster"
    PER_VECTOR = "per_vector"


class InsufficientMemoryError(ValueError):
    """Raised when max_memory is too small to store every construct vector
    at even the coarsest quantization level (RaBitQ) -- see module
    docstring."""


# Finest -> coarsest. Walked in this order by _choose_level.
_LADDER: Tuple[QuantizationLevel, ...] = (
    QuantizationLevel.FP16, QuantizationLevel.SQ8, QuantizationLevel.SQ4, QuantizationLevel.RABITQ,
)

_SQ_QTYPES = {
    QuantizationLevel.FP16: faiss.ScalarQuantizer.QT_fp16,
    QuantizationLevel.SQ8: faiss.ScalarQuantizer.QT_8bit,
    QuantizationLevel.SQ4: faiss.ScalarQuantizer.QT_4bit,
}

_BYTES_FP32 = 4
# Plain sign-bit RaBitQ (nb_bits=1) -- exactly "1 bit per dimension", and
# also the only nb_bits value that doesn't crash: faiss's extra-bits RaBitQ
# variant (nb_bits > 1) segfaults on this platform/faiss build (verified
# empirically), so this isn't a capability we're giving up, just one we
# can't safely reach.
_RABITQ_NB_BITS = 1


# --------------------------------------------------------------------------- #
# Layout accounting -- one formula, used both to pick a quantization level
# against a byte budget (before any real data exists) and, after
# construction, to report get_bytesize(), so the two can never drift apart.
# --------------------------------------------------------------------------- #

def _code_size(level: QuantizationLevel, dim: int) -> int:
    if level in _SQ_QTYPES:
        return faiss.ScalarQuantizer(dim, _SQ_QTYPES[level]).code_size
    return faiss.RaBitQuantizer(dim, faiss.METRIC_L2).compute_code_size(dim, _RABITQ_NB_BITS)


def _fixed_overhead_bytes(level: QuantizationLevel, dim: int) -> int:
    """Per-structure cost independent of n: a scalar quantizer's trained
    per-dimension min/diff table, computed via a throwaway probe trained on
    stand-in data (mirrors BallCoverFilter._codebook_bytes' own probe-train
    trick, so this is knowable before real vectors exist). RaBitQ has no
    such trained table."""
    if level not in _SQ_QTYPES:
        return 0
    probe = faiss.ScalarQuantizer(dim, _SQ_QTYPES[level])
    probe.train(np.zeros((2, dim), dtype=np.float32))
    return int(faiss.vector_to_array(probe.trained).nbytes)


def _cluster_count(n: int) -> int:
    """sqrt(n) coarse-cluster count, mirroring IVFFilter's own nlist default
    (see IVFFilter._choose_params)."""
    return max(1, min(n, round(n ** 0.5))) if n > 0 else 1


def _id_bytes(count: int) -> int:
    if count <= 256:
        return 1
    if count <= 65536:
        return 2
    return 4


def _id_dtype(count: int):
    return {1: np.uint8, 2: np.uint16, 4: np.uint32}[_id_bytes(count)]


def _error_budget_bytes(mode: ErrorBudgetMode, n: int, eps_bits: int) -> int:
    """Bytes needed to store the error budget at the given granularity (see
    module docstring / QuantizedVectorFilter's error_budget_mode):
    global -- one float32; per_cluster -- one float32 per coarse-quantizer
    cluster plus one small cluster-id per vector; per_vector -- eps_i rounded
    up into eps_bits each."""
    if mode == ErrorBudgetMode.GLOBAL:
        return _BYTES_FP32
    if mode == ErrorBudgetMode.PER_CLUSTER:
        k = _cluster_count(n)
        return k * _BYTES_FP32 + n * _id_bytes(k)
    if mode == ErrorBudgetMode.PER_VECTOR:
        return -(-(n * eps_bits) // 8)  # ceil(n * eps_bits / 8)
    raise ValueError(f"Unknown error_budget_mode {mode!r}")


def _footprint_bytes(level: QuantizationLevel, n: int, dim: int, mode: ErrorBudgetMode, eps_bits: int) -> int:
    """Total stored-structure footprint for `level` at `n` vectors. Does not
    count vector identifiers -- this filter answers yes/no and never reports
    which vector matched, a deliberate fairness decision matching what a
    membership-only baseline is allowed to skip storing."""
    return n * _code_size(level, dim) + _fixed_overhead_bytes(level, dim) + _error_budget_bytes(mode, n, eps_bits)


class QuantizedVectorFilter(Filter):
    """Per-point quantization baseline (see module docstring).

    tau is a pure query-time knob, like HNSWFilter/IVFFilter: nothing about
    the quantizer depends on the target distance, only on dim/n/max_memory.
    """

    def __init__(
        self,
        dim: int,
        distance_threshold: float,
        max_memory: Optional[int] = None,
        error_budget_mode: ErrorBudgetMode = ErrorBudgetMode.PER_CLUSTER,
        per_vector_eps_bits: int = 8,
        max_spheres: Optional[int] = None,
        seed: int = 42,
    ):
        super().__init__(dim, max_memory=max_memory)
        self.distance_threshold = float(distance_threshold)
        self.error_budget_mode = error_budget_mode
        if per_vector_eps_bits not in (1, 2, 4, 8):
            raise ValueError(f"per_vector_eps_bits must be one of (1, 2, 4, 8), got {per_vector_eps_bits}")
        self.per_vector_eps_bits = per_vector_eps_bits
        self._max_spheres = max_spheres  # unused; accepted for _instantiate_filter parity
        self.seed = seed

        self.level: Optional[QuantizationLevel] = None
        self._n = 0
        self._sq_index: Optional[faiss.IndexScalarQuantizer] = None  # fp16 / sq8 / sq4 only
        self._rq: Optional[faiss.RaBitQuantizer] = None               # rabitq only
        self._codes: Optional[np.ndarray] = None                      # rabitq only

        self._eps_global: float = 0.0
        self._eps_table: Optional[np.ndarray] = None       # per_cluster
        self._cluster_ids: Optional[np.ndarray] = None     # per_cluster
        self._eps_bucket_width: float = 0.0                # per_vector
        self._eps_bucket_ids: Optional[np.ndarray] = None  # per_vector

        self._constructed = False

    # ------------------------------------------------------------------ level selection

    def _choose_level(self, n: int) -> QuantizationLevel:
        for level in _LADDER:
            if self.max_memory is None or _footprint_bytes(
                level, n, self.dim, self.error_budget_mode, self.per_vector_eps_bits
            ) <= self.max_memory:
                return level
        smallest = _footprint_bytes(_LADDER[-1], n, self.dim, self.error_budget_mode, self.per_vector_eps_bits)
        raise InsufficientMemoryError(
            f"max_memory={self.max_memory} bytes is too small to store n={n} vectors of dim={self.dim} "
            f"at even the coarsest quantization level ({_LADDER[-1].value}) under "
            f"error_budget_mode={self.error_budget_mode.value}: smallest achievable footprint is "
            f"{smallest} bytes. Only a geometric summary (e.g. BallCoverFilter) can operate under this budget."
        )

    # ------------------------------------------------------------------ construct

    def construct(self, vectors: np.ndarray) -> None:
        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        n = len(vecs)
        self._n = n
        self._constructed = True
        if n == 0:
            self.level = None
            return

        self.level = self._choose_level(n)

        if self.level in _SQ_QTYPES:
            index = faiss.IndexScalarQuantizer(self.dim, _SQ_QTYPES[self.level], faiss.METRIC_L2)
            index.train(vecs)
            index.add(vecs)
            decoded = index.reconstruct_n(0, n)
            self._sq_index, self._rq, self._codes = index, None, None
        else:
            rq = faiss.RaBitQuantizer(self.dim, faiss.METRIC_L2)
            rq.train(vecs)
            codes = rq.compute_codes(vecs)
            decoded = rq.decode(codes)
            self._rq, self._codes, self._sq_index = rq, codes, None

        # Measured directly against the quantizer's own decode() output --
        # never a probabilistic bound, and computed before any further
        # processing, so this is exactly the error the acceptance rule needs
        # (see module docstring).
        eps = np.linalg.norm(vecs - decoded, axis=1).astype(np.float32)
        self._build_error_budget(vecs, eps)

    def _build_error_budget(self, vecs: np.ndarray, eps: np.ndarray) -> None:
        n = len(eps)
        mode = self.error_budget_mode

        if mode == ErrorBudgetMode.GLOBAL:
            self._eps_global = float(eps.max()) if n else 0.0
            return

        if mode == ErrorBudgetMode.PER_CLUSTER:
            k = _cluster_count(n)
            km = faiss.Kmeans(self.dim, k, niter=10, seed=self.seed, verbose=False)
            km.train(vecs)
            _, assign = km.index.search(vecs, 1)
            assign = assign[:, 0].astype(np.int64)
            table = np.zeros(k, dtype=np.float32)
            # Per-cluster max, not mean -- the reconstructed eps for every
            # member of a cluster must be >= its own true eps_i (round up).
            np.maximum.at(table, assign, eps)
            self._eps_table = table
            self._cluster_ids = assign.astype(_id_dtype(k))
            return

        if mode == ErrorBudgetMode.PER_VECTOR:
            bits = self.per_vector_eps_bits
            levels = (1 << bits) - 1
            eps_max = float(eps.max()) if n else 0.0
            if eps_max <= 0.0:
                self._eps_bucket_width = 0.0
                self._eps_bucket_ids = np.zeros(n, dtype=np.uint8)
                return
            width = eps_max / levels
            # Ceiling bucket assignment: bucket_id * width must never be
            # less than the true eps_i, or the zero-false-negative guarantee
            # breaks (see module docstring).
            bucket_ids = np.minimum(np.ceil(eps / width).astype(np.int64), levels)
            self._eps_bucket_width = width
            self._eps_bucket_ids = bucket_ids.astype(np.uint8)
            return

        raise ValueError(f"Unknown error_budget_mode {mode!r}")

    def _eps_per_vector(self) -> np.ndarray:
        mode = self.error_budget_mode
        if mode == ErrorBudgetMode.GLOBAL:
            return np.full(self._n, self._eps_global, dtype=np.float64)
        if mode == ErrorBudgetMode.PER_CLUSTER:
            return self._eps_table[self._cluster_ids].astype(np.float64)
        return self._eps_bucket_ids.astype(np.float64) * self._eps_bucket_width

    # ------------------------------------------------------------------ query

    def query(self, vectors: np.ndarray, tau: Optional[float] = None) -> List[bool]:
        if not self._constructed:
            raise ValueError("QuantizedVectorFilter has not been constructed.")
        if tau is None:
            tau = self.distance_threshold
        if self._n == 0 or len(vectors) == 0:
            return [False] * len(vectors)

        q = np.ascontiguousarray(vectors, dtype=np.float32)
        thresh_sq = (float(tau) + self._eps_per_vector()) ** 2

        if self._sq_index is not None:
            # Exact SIMD asymmetric distance search: faiss's ScalarQuantizer
            # ADC is exact w.r.t. decode() (no further approximation beyond
            # the quantization already priced into eps_i), so this is safe
            # for the zero-false-negative acceptance rule -- see module
            # docstring.
            d2, ids = self._sq_index.search(q, self._n)
            hit = (d2 <= thresh_sq[ids]).any(axis=1)
        else:
            # IndexRaBitQ's own search is the paper's fast *probabilistic*
            # estimator, not the exact distance to decode() eps_i was
            # measured against -- using it would reintroduce the
            # probabilistic-bound failure mode this filter avoids. Falls
            # back to an explicit decode + batched (BLAS) distance instead.
            decoded = self._rq.decode(self._codes)
            d2 = np.asarray(faiss.pairwise_distances(q, decoded))
            np.clip(d2, 0, None, out=d2)
            hit = (d2 <= thresh_sq[None, :]).any(axis=1)

        return hit.tolist()

    # ------------------------------------------------------------------ memory accounting

    def get_bytesize(self) -> int:
        if not self._constructed or self.level is None:
            return 0
        return _footprint_bytes(self.level, self._n, self.dim, self.error_budget_mode, self.per_vector_eps_bits)
