"""QuantizedVectorFilter -- per-point quantization baseline: the "compress
the vectors" counterpart to BallCoverFilter's "compress the geometry". Keeps
every construct vector, quantized to whichever scheme on a fixed fidelity
ladder fits the memory budget, finest first: float16, then the same
TurboQuant/scalar-quantizer rungs BallCoverFilter's own Quantize ladder sweeps
(8, 6, 5, 4, 3, 2 bits/dim -- see filters.ball_cover.ball_cover_filter.Quantize
for the full bits-per-dim table this mirrors), then RaBitQ (~1 bit/dim, no
codebook) as the coarsest fallback. The finer, evenly-spaced steps (vs. the
old FP16/8-bit/4-bit/RaBitQ ladder's big jumps) give a smoother
accuracy-vs-memory curve to compare BallCoverFilter's own quantized variants
against at comparable bit rates. Exists to find the dataset size at which
BallCoverFilter's geometric summary wins over storing (compressed) points
directly.

Zero-false-negative by the same triangle-inequality argument
BallCoverFilter._add_spheres uses: d(q,x) <= tau implies
d(q,x_hat) <= tau + ||x - x_hat||, so accepting whenever
d(q, x_hat_i) <= tau + eps_i, with eps_i the *measured* reconstruction error
(never a probabilistic bound), can never produce a false negative. The
TurboQuant rungs sit behind an orthonormal rotation (like BallCoverFilter's
own TQ* rungs -- see that module's _make_rotation), which preserves every L2
distance exactly, so this argument and eps_i's measurement both carry over
unchanged; RaBitQ is unrotated here, unchanged from before.

If not even RaBitQ fits max_memory for n vectors, this raises
InsufficientMemoryError rather than silently dropping vectors or
subsampling (see runner.py::equal_memory_worker's handling of it)."""

from enum import Enum
from typing import List, Optional, Tuple

import faiss
import numpy as np

from filters.filter import Filter, InsufficientMemoryError


class QuantizationLevel(Enum):
    FP16 = "fp16"
    TQ8 = "tq8"
    SQ6 = "sq6"
    TQ5 = "tq5"
    TQ4 = "tq4"
    TQ3 = "tq3"
    TQ2 = "tq2"
    RABITQ = "rabitq"


class ErrorBudgetMode(Enum):
    GLOBAL = "global"
    PER_CLUSTER = "per_cluster"
    PER_VECTOR = "per_vector"


# Finest -> coarsest. Walked in this order by _choose_level.
_LADDER: Tuple[QuantizationLevel, ...] = (
    QuantizationLevel.FP16, QuantizationLevel.TQ8, QuantizationLevel.SQ6,
    QuantizationLevel.TQ5, QuantizationLevel.TQ4, QuantizationLevel.TQ3,
    QuantizationLevel.TQ2, QuantizationLevel.RABITQ,
)

_SQ_QTYPES = {
    QuantizationLevel.FP16: faiss.ScalarQuantizer.QT_fp16,
    QuantizationLevel.TQ8: faiss.ScalarQuantizer.QT_8bit_tqmse,
    QuantizationLevel.SQ6: faiss.ScalarQuantizer.QT_6bit,
    QuantizationLevel.TQ5: faiss.ScalarQuantizer.QT_5bit_tq,
    QuantizationLevel.TQ4: faiss.ScalarQuantizer.QT_4bit_tqmse,
    QuantizationLevel.TQ3: faiss.ScalarQuantizer.QT_3bit_tqmse,
    QuantizationLevel.TQ2: faiss.ScalarQuantizer.QT_2bit_tqmse,
}

# Rungs that need BallCoverFilter-style rotation before quantizing (every
# TurboQuant rung; SQ6 is a plain min/max scalar quantizer like the old
# ladder's SQ8/SQ4 were, so it doesn't). See _make_rotation.
_ROTATED_LEVELS = {
    QuantizationLevel.TQ8, QuantizationLevel.TQ5, QuantizationLevel.TQ4,
    QuantizationLevel.TQ3, QuantizationLevel.TQ2,
}

# Rungs whose faiss codec folds each vector's stored exact norm into
# search(): QT_5bit_tq scores ||q - x_hat||^2 + (||x||^2 - ||x_hat||^2), not
# ||q - x_hat||^2, so it is not exact w.r.t. decode() and the eps_i argument
# fails unless query() subtracts that per-vector constant back out (see
# _search_offsets and BallCoverFilter's _Encoding.norm_offset).
_NORM_OFFSET_LEVELS = {QuantizationLevel.TQ5}

_BYTES_FP32 = 4
# Plain sign-bit RaBitQ (nb_bits=1) -- exactly "1 bit per dimension", and
# also the only nb_bits value that doesn't crash: faiss's extra-bits RaBitQ
# variant (nb_bits > 1) segfaults on this platform/faiss build (verified
# empirically), so this isn't a capability we're giving up, just one we
# can't safely reach.
_RABITQ_NB_BITS = 1

# Rotation seed for the TurboQuant rungs, mirroring BallCoverFilter's own
# _ROTATION_SEED: seeded explicitly so it's already is_trained (no separate
# training step) and reproducible, and charged as a fixed 4-byte cost rather
# than the dim*dim matrix it expands to (see _make_rotation).
_ROTATION_SEED = 0x5EED
_ROTATION_SEED_BYTES = 4

# Throwaway-probe training data for _fixed_overhead_bytes: random, not zeros
# -- the MSE-trained TurboQuant rungs divide through by the data's spread
# while training, so all-zero data (spread=0) would corrupt the trained
# table's size. A dedicated local stream, never np.random.*, so probing a
# level's overhead can't shift any globally-seeded sequence downstream.
_PROBE_RNG = np.random.default_rng(_ROTATION_SEED)


def _make_rotation(dim: int) -> faiss.VectorTransform:
    """Same choice BallCoverFilter._make_rotation makes: Hadamard at
    power-of-two dimensions, RandomRotationMatrix elsewhere. This project's
    dim=384 always takes the RandomRotationMatrix branch (384 isn't a power
    of two, so Hadamard's forced d_out=512 mismatches immediately)."""
    rotation = faiss.HadamardRotation(dim)
    if rotation.d_out != dim:
        rotation = faiss.RandomRotationMatrix(dim, dim)
    rotation.init(_ROTATION_SEED)
    return rotation


def _make_index(level: QuantizationLevel, dim: int) -> faiss.Index:
    """The scalar-quantizer index this level queries/reconstructs through --
    wrapped in an orthonormal rotation for the TurboQuant rungs (see
    _ROTATED_LEVELS), which train()/add()/search()/reconstruct_n() all pass
    through transparently, exactly like BallCoverFilter's own quantized
    centers do."""
    base = faiss.IndexScalarQuantizer(dim, _SQ_QTYPES[level], faiss.METRIC_L2)
    if level in _ROTATED_LEVELS:
        return faiss.IndexPreTransform(_make_rotation(dim), base)
    return base


# Layout accounting: one formula used both to pick a level against a byte
# budget and, after construction, to report get_bytesize(), so they can't drift apart.

def _code_size(level: QuantizationLevel, dim: int) -> int:
    if level in _SQ_QTYPES:
        return faiss.ScalarQuantizer(dim, _SQ_QTYPES[level]).code_size
    return faiss.RaBitQuantizer(dim, faiss.METRIC_L2).compute_code_size(dim, _RABITQ_NB_BITS)


def _fixed_overhead_bytes(level: QuantizationLevel, dim: int) -> int:
    """Per-structure cost independent of n: a scalar quantizer's trained
    per-dimension min/diff (or MSE) table, computed via a throwaway probe
    trained on stand-in data so this is knowable before real vectors exist,
    plus the rotation seed's 4 bytes for a rotated rung. RaBitQ has no
    trained table and no rotation here."""
    if level not in _SQ_QTYPES:
        return 0
    overhead = _ROTATION_SEED_BYTES if level in _ROTATED_LEVELS else 0
    probe = faiss.ScalarQuantizer(dim, _SQ_QTYPES[level])
    probe.train(_PROBE_RNG.random((2, dim)).astype(np.float32))
    return overhead + int(faiss.vector_to_array(probe.trained).nbytes)


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
            index = _make_index(self.level, self.dim)
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

        # Measured directly against decode() output -- never a probabilistic
        # bound (see module docstring).
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
            # Max, not mean: a cluster's stored eps must be >= every member's true eps_i.
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
            # Ceiling: bucket_id * width must never be less than the true eps_i.
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
            # faiss's ScalarQuantizer ADC is exact w.r.t. decode() -- for the
            # norm-offset rungs only once _search_offsets() is removed -- so
            # this is safe for the zero-false-negative acceptance rule.
            d2, ids = self._sq_index.search(q, self._n)
            if self.level in _NORM_OFFSET_LEVELS:
                d2 = d2 - self._search_offsets()[np.maximum(ids, 0)]
            hit = (d2 <= thresh_sq[ids]).any(axis=1)
        else:
            # RaBitQ's own search is a fast *probabilistic* estimator, not
            # the exact distance eps_i was measured against -- would
            # reintroduce the failure mode this filter avoids. Decode + batched distance instead.
            decoded = self._rq.decode(self._codes)
            d2 = np.asarray(faiss.pairwise_distances(q, decoded))
            np.clip(d2, 0, None, out=d2)
            hit = (d2 <= thresh_sq[None, :]).any(axis=1)

        return hit.tolist()


    def _search_offsets(self) -> np.ndarray:
        """Per-vector constant search() adds on top of ||q - decode(x)||^2
        (see _NORM_OFFSET_LEVELS), measured on every call rather than stored:
        searching from the origin returns the codec's stored ||x||^2, and
        decode gives ||decode(x)||^2, so it costs no bytes beyond the codes."""
        origin = np.zeros((1, self.dim), dtype=np.float32)
        d2, ids = self._sq_index.search(origin, self._n)
        decoded = self._sq_index.reconstruct_n(0, self._n)
        offsets = np.zeros(self._n, dtype=np.float32)
        valid = ids[0] >= 0
        offsets[ids[0][valid]] = d2[0][valid] - np.einsum("ij,ij->i", decoded, decoded)[ids[0][valid]]
        return offsets


    def get_bytesize(self) -> int:
        if not self._constructed or self.level is None:
            return 0
        return _footprint_bytes(self.level, self._n, self.dim, self.error_budget_mode, self.per_vector_eps_bits)
