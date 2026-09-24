import numpy as np
import faiss
import sys
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
from filters.filter import Filter, InsufficientMemoryError
from filters.ball_cover.ball_interfaces import ConstructInterface, InsertInterface
from filters.ball_cover.gonzales_construct import GonzalesConstruct
from filters.ball_cover.min_heap_insert import MinHeapInsert
from filters.ball_cover.nearest_insert import NearestInsert
from filters.ball_cover.least_inflation_insert import LeastInflationInsert
from filters.ball_cover.exhaustive_merge_insert import ExhaustiveMergeInsert
from filters.ball_cover.doubling_insert import DoublingInsert
from filters.ball_cover.pooled_insert import PooledInsert


class Quantize(Enum):
    """Center encodings, one per rung of a bits-per-dimension ladder. Byte
    figures are for dim=384 (this project's sweep dimension):

      b/dim  member   encoding                     center B   codebook
        32   FP32     raw float32                      1536   0
        16   FP16     scalar fp16                       768   0
         8   SQ8      scalar 8-bit                      384   3 KB
         8   TQ8      rotation + TurboQuant 8-bit        384   seed only
         6   SQ6      scalar 6-bit                      288   3 KB
         5   TQ5      rotation + TurboQuant 5-bit        240   seed only
         4   SQ4      scalar 4-bit                      192   3 KB
         4   TQ4      rotation + TurboQuant 4-bit        192   seed only
         3   TQ3      rotation + TurboQuant 3-bit       144   60 B + seed
         2   TQ2      rotation + TurboQuant 2-bit         96   seed only
         2   PQ4_R2   PQ, 4-bit subquantizers (m=192)    96   24 KB
         2   PQ8_R2   PQ, 8-bit subquantizers (m=96)     96   384 KB
         1   TQ1      rotation + TurboQuant 1-bit         48   seed only
         1   PQ4_R1   PQ, 4-bit subquantizers (m=96)     48   24 KB
         1   PQ8_R1   PQ, 8-bit subquantizers (m=48)     48   384 KB
         1   RABITQ   rotation + RaBitQ                  56   seed only

    TQ1/TQ2/TQ4/TQ5/TQ8 fill out the rest of faiss's TurboQuant family
    alongside TQ3 (see _ENCODINGS): evaluation-only for now (added to
    runner.QUANTIZATION_SWEEP but deliberately not to
    plot_quantization.QUANTIZE_SWEEP, so they generate rows without yet
    appearing in any figure). Unlike TQ3, none of these five have had
    rotation's benefit individually re-measured against real embeddings --
    they inherit rotate=True from TQ3 for consistency within the TurboQuant
    family, not because it has been shown to pay off at every one of these
    bit rates (a synthetic probe suggested it might not always help, e.g.
    at 8-bit -- see the results this sweep produces before trusting that).
    TQ5 has no MSE-trained ("tqmse") faiss variant, unlike the others --
    QT_5bit_tq is the plain (quantile-based) TurboQuant encoder.

    Every ball also stores its radius as fp16, so a ball costs center + 2
    bytes -- see bytes_per_ball().

    CAVEAT on RABITQ: the other ten rows are exact ADC, so search() returns
    ||q - decode(c)|| and the one-sided inflation in _store_spheres is a
    proof. RaBitQ instead scores through its own estimator, which is not tied
    to sa_decode's output: measured against exact distances over 3.6M
    (query, center) pairs on real embeddings it under-estimated every single
    time, by 0.13 to 0.33, which is safe (an under-estimate only over-accepts)
    but is an empirical observation, not a bound. At qb=1 it over-estimates
    3.2% of the time by up to +0.37, so the one-sidedness is a property of
    the configuration, not of the scheme. RABITQ's zero-false-negative
    guarantee is therefore weaker in kind than the rest of the ladder's.

    Member names are dimension-independent: PQ4_R2 reads "4-bit
    subquantizers at a 2 bit/dim rate", and the subquantizer count m is
    derived from dim by _pq_m_for(), so the rates hold at any dimension that
    admits them. The m values above are what dim=384 resolves to.
    """
    FP32 = "FP32"
    NONE = "FP32"     # back-compat alias: FP32 *is* the unquantized baseline
    FP16 = "FP16"
    SQ8 = "SQ8"
    TQ8 = "TQ8"
    SQ6 = "SQ6"
    TQ5 = "TQ5"
    SQ4 = "SQ4"
    TQ4 = "TQ4"
    TQ3 = "TQ3"
    TQ2 = "TQ2"
    PQ4_R2 = "PQ4@2"
    PQ8_R2 = "PQ8@2"
    TQ1 = "TQ1"
    PQ4_R1 = "PQ4@1"
    PQ8_R1 = "PQ8@1"
    RABITQ = "RaBitQ"


@dataclass(frozen=True)
class _Encoding:
    """How one Quantize member maps onto faiss."""
    rate: int                      # target bits per dimension
    kind: str                      # "flat" | "sq" | "pq" | "rabitq"
    qtype: Optional[int] = None    # ScalarQuantizer.QT_* -- kind == "sq"
    nbits: int = 0                 # bits per subquantizer -- kind == "pq"
    rotate: bool = False           # see _ROTATION_SEED


# An orthonormal rotation in front of the quantizer. Orthonormal is the
# load-bearing word: it preserves every L2 distance, so radii, the measured
# encoding error and query()'s triangle-inequality argument all carry over
# unchanged -- the rotation is invisible to the geometry and only changes how
# energy is spread across coordinates before it is quantized.
#
# Applied only where it measurably pays. Measured on this project's real
# embeddings (msmarco/nq/quora_qp, dim=384, unit norm), reconstruction error
# over 1024 ball centers:
#
#                    mean               p99              max
#     TurboQuant3   0.18808 -> 0.18508  0.21796 -> 0.21404  0.25265 -> 0.23092
#     RaBitQ        0.76208 -> 0.74719  0.82933 -> 0.81938  0.86208 -> 0.85356
#     PQ, plain SQ  no change (they already adapt per subspace or per
#                   dimension, so there is nothing for a rotation to even out)
#
# Modest on the mean (~2%) but worth it on the tail (TurboQuant3's worst ball
# improves 8.6%), and it is free at query time: the rotation is one batched
# GEMM amortised over the ADC scan, and measured query throughput did not
# drop. Note that an earlier synthetic benchmark put the gain at 23%; that
# used deliberately anisotropic data and overstated it badly. Unit-norm
# embeddings are far closer to isotropic, so there is much less for a
# rotation to fix -- always re-measure this on real data.
_ROTATION_SEED = 0x5EED
_ROTATION_SEED_BYTES = 4

# Private stream for _codebook_bytes()'s throwaway probe -- see there.
_PROBE_RNG = np.random.default_rng(0x5EED)


_ENCODINGS = {
    Quantize.FP32: _Encoding(rate=32, kind="flat"),
    Quantize.FP16: _Encoding(rate=16, kind="sq", qtype=faiss.ScalarQuantizer.QT_fp16),
    Quantize.SQ8: _Encoding(rate=8, kind="sq", qtype=faiss.ScalarQuantizer.QT_8bit),
    # The rest of faiss's TurboQuant family, alongside TQ3 -- see the
    # Quantize docstring for why these are evaluation-only (swept but not
    # plotted) for now. All MSE-trained ("tqmse") except TQ5, which faiss
    # only offers as the plain quantile-based QT_5bit_tq.
    Quantize.TQ8: _Encoding(rate=8, kind="sq", qtype=faiss.ScalarQuantizer.QT_8bit_tqmse, rotate=True),
    Quantize.SQ6: _Encoding(rate=6, kind="sq", qtype=faiss.ScalarQuantizer.QT_6bit),
    Quantize.TQ5: _Encoding(rate=5, kind="sq", qtype=faiss.ScalarQuantizer.QT_5bit_tq, rotate=True),
    Quantize.SQ4: _Encoding(rate=4, kind="sq", qtype=faiss.ScalarQuantizer.QT_4bit),
    Quantize.TQ4: _Encoding(rate=4, kind="sq", qtype=faiss.ScalarQuantizer.QT_4bit_tqmse, rotate=True),
    # QT_3bit_tqmse is the MSE-trained TurboQuant variant; there is no plain
    # (min-max) 3-bit scalar quantizer in faiss to compare it against.
    Quantize.TQ3: _Encoding(rate=3, kind="sq", qtype=faiss.ScalarQuantizer.QT_3bit_tqmse, rotate=True),
    Quantize.TQ2: _Encoding(rate=2, kind="sq", qtype=faiss.ScalarQuantizer.QT_2bit_tqmse, rotate=True),
    Quantize.PQ4_R2: _Encoding(rate=2, kind="pq", nbits=4),
    Quantize.PQ8_R2: _Encoding(rate=2, kind="pq", nbits=8),
    Quantize.TQ1: _Encoding(rate=1, kind="sq", qtype=faiss.ScalarQuantizer.QT_1bit_tqmse, rotate=True),
    Quantize.PQ4_R1: _Encoding(rate=1, kind="pq", nbits=4),
    Quantize.PQ8_R1: _Encoding(rate=1, kind="pq", nbits=8),
    Quantize.RABITQ: _Encoding(rate=1, kind="rabitq", rotate=True),
}


class BallCoverFilter(Filter):
    _BYTES_PER_FLOAT16 = 2
    _BYTES_PER_FLOAT32 = 4
    _BYTES_PER_FLOAT64 = 8

    # Coverage-comparison safety margin: FAISS's IndexFlatL2.search() isn't
    # bit-reproducible across different query-batch shapes, so a ball whose
    # radius was grown to exactly match one measured distance can see a
    # slightly different distance later when re-queried in a bigger batch
    # (e.g. a no-false-negative sweep). sqrt() of a near-zero squared
    # distance (near-duplicate points, ~0 radius) amplifies that error --
    # measured up to ~9.1e-4 on real data at dim=384. 1e-3 covers that with
    # margin, matches this project's own tol=1e-3 precedent
    # (test_ball_cover_construction.py's _covering_ok), and stays far below
    # any real distance_threshold used in this project's sweeps.
    _COVERAGE_EPSILON = 1e-3

    # How many of a construct vector's nearest balls _tight_radii tests when
    # looking for one that already covers it. The pass only ever skips
    # growing a ball when it positively finds a cover, so a vector held by
    # some ball outside this shortlist is a missed saving, never a coverage
    # hole -- which is what lets this stay a small constant instead of an
    # all-pairs scan.
    _COVERAGE_NEIGHBORS = 16

    def __init__(
        self,
        dim: int,
        construct_method: Optional[ConstructInterface] = GonzalesConstruct(),
        insert_method: Optional[InsertInterface] = MinHeapInsert(),
        quantize: Quantize = Quantize.NONE,
        max_memory: Optional[int] = None,
        max_spheres: Optional[int] = None,
        insert_memory_percent: Optional[float] = 0.0
    ) -> None:
        super().__init__(dim, max_memory=max_memory)
        self.quantize = quantize
        self.construct_method = construct_method
        self.insert_method = insert_method
        self.insert_memory_percent = insert_memory_percent
        self.all_centers_index = self._make_faiss_index(self.quantize)
        self.max_spheres = (max_spheres if max_spheres is not None 
                    else self._calculate_max_spheres(self.max_memory))
        self.max_construct_spheres = self._calculate_max_construct_spheres(self.insert_memory_percent)
        self._constructed = False

        self._num_spheres = 0
        self._all_spheres = np.arange(self.max_spheres, dtype=np.int32)
        self._all_centers = None # temporary decoded array used only during insertions, not persistent allocation
        self._all_radiuses = np.empty((self.max_spheres,), dtype=np.float32)

        if self.insert_method:
            self.insert_method.ensure_capacity(self.max_spheres)

    @property
    def all_spheres(self) -> np.ndarray:
        """A NumPy view of active sphere indices currently allocated in the filter."""
        return self._all_spheres[:self._num_spheres]

    @property
    def all_centers(self) -> np.ndarray:
        """A NumPy view of active sphere centers decoded directly from the FAISS index."""
        if self._all_centers is None:
            self._all_centers = np.empty((len(self._all_radiuses), self.dim), dtype=np.float32)
            if self._num_spheres > 0:
                sphere_indices = np.ascontiguousarray(self.all_spheres, dtype=np.int64)
                self._all_centers[self.all_spheres] = self.all_centers_index.reconstruct_batch(sphere_indices)
        return self._all_centers[self.all_spheres]

    @property
    def all_radiuses(self) -> np.ndarray:
        """A NumPy view of active sphere radii corresponding to the active spheres."""
        return self._all_radiuses[self.all_spheres]


    def _ensure_capacity(self, required_cap: int) -> None:
        current_cap = len(self._all_radiuses)
        if required_cap <= current_cap:
            return

        new_cap = max(current_cap * 2, required_cap)
        new_spheres = np.empty((new_cap,), dtype=np.int32)
        new_spheres[:current_cap] = self._all_spheres
        new_spheres[current_cap:] = np.arange(current_cap, new_cap, dtype=np.int32)
        self._all_spheres = new_spheres

        if self._all_centers is not None:
            new_centers = np.empty((new_cap, self.dim), dtype=np.float32)
            new_centers[:current_cap] = self._all_centers
            self._all_centers = new_centers

        new_radiuses = np.empty((new_cap,), dtype=np.float32)
        new_radiuses[:current_cap] = self._all_radiuses
        self._all_radiuses = new_radiuses

        if self.insert_method:
            self.insert_method.ensure_capacity(new_cap)


    def _reset_buffers(self, capacity: int) -> None:
        self._ensure_capacity(capacity)
        self._num_spheres = 0
        self.all_centers_index = self._make_faiss_index(self.quantize)
        self._all_radiuses[:] = 0.0
        self._all_centers = None
        self._constructed = False

        if self.insert_method:
            self.insert_method.reset_buffers(capacity)


    def _add_spheres(self, centers: np.ndarray, radii: np.ndarray) -> np.ndarray:
        if len(centers) == 0:
            return np.empty(0, dtype=np.int32)

        centers = np.ascontiguousarray(centers, dtype=np.float32)
        radii = np.asarray(radii, dtype=np.float32)
        num_new_spheres = len(centers)

        self._ensure_capacity(self._num_spheres + num_new_spheres)
        new_sphere_indices = self._all_spheres[self._num_spheres : self._num_spheres + num_new_spheres]
        self.all_centers_index.add_with_ids(centers, new_sphere_indices.astype(np.int64))
        self._all_radiuses[new_sphere_indices] = self._fp16_ceil(radii + self._encoding_error(centers))
        self._num_spheres += num_new_spheres

        if self._all_centers is not None:
            self._all_centers[new_sphere_indices] = centers

        return new_sphere_indices


    def _remove_spheres(self, sphere_indices: np.ndarray) -> None:
        if len(sphere_indices) == 0:
            return

        sphere_indices = np.ascontiguousarray(sphere_indices, dtype=np.int64)
        removed_sphere_mask = np.isin(self.all_spheres, sphere_indices)
        active_spheres = self.all_spheres[~removed_sphere_mask]
        removed_spheres = self.all_spheres[removed_sphere_mask]
        num_active_spheres = len(active_spheres)

        self._all_spheres[:num_active_spheres] = active_spheres
        self._all_spheres[num_active_spheres : self._num_spheres] = removed_spheres
        self.all_centers_index.remove_ids(faiss.IDSelectorBatch(sphere_indices))
        self._all_radiuses[sphere_indices] = 0.0
        self._num_spheres = num_active_spheres

        if self._all_centers is not None:
            self._all_centers[sphere_indices] = 0.0


    def _calculate_max_spheres(self, max_memory: Optional[int]) -> int:
        if max_memory is None:
            return sys.maxsize
        usable = max(0, max_memory - self._codebook_bytes())
        return max(1, usable // self.bytes_per_ball())


    def _calculate_max_construct_spheres(self, insertion_memory_percent: float) -> int:
        if insertion_memory_percent <= 0.0:
            return self.max_spheres

        if self.max_memory is not None:
            construct_memory = max(0, int(self.max_memory * (1.0 - insertion_memory_percent / 100.0)))
            return self._calculate_max_spheres(construct_memory)

        return max(1, int(self.max_spheres * (1.0 - insertion_memory_percent / 100.0)))


    @staticmethod
    def _pq_m_for(dim: int, rate: int, nbits: int) -> int:
        """Subquantizer count hitting `rate` bits/dim with `nbits`-bit codes:
        m = dim * rate / nbits, snapped *down* to a divisor of dim (faiss
        requires dim % m == 0, and rounding down can only come in under the
        byte budget, never over). At dim=384 this gives the table's m: 192
        for PQ4_R2, 96 for PQ8_R2 and PQ4_R1."""
        target = min((dim * rate) // nbits, dim)
        for m in range(target, 0, -1):
            if dim % m == 0:
                return m
        raise InsufficientMemoryError(
            f"No valid PQ subquantizer count for dim={dim} at {rate} bit(s)/dim "
            f"with {nbits}-bit codes.")


    @staticmethod
    def _fp16_ceil(radii: np.ndarray) -> np.ndarray:
        """Rounds each radius up to the nearest fp16 value, so the 2 bytes
        bytes_per_ball() charges for it are honest.

        Up, never to-nearest: a radius that shrinks even by one ulp can drop
        a point the ball was covering, which is a false negative. Rounding up
        only ever over-covers. Radii past fp16's 65504 become +inf, which
        over-covers absolutely and is still sound. Idempotent, so re-applying
        it on every insert costs nothing."""
        with np.errstate(over="ignore"):
            half = radii.astype(np.float16)
        up = np.nextafter(half, np.float16(np.inf))
        return np.where(half.astype(np.float32) < radii, up, half).astype(np.float32)


    def _make_faiss_index(self, quantize: Quantize) -> faiss.IndexIDMap2:
        spec = _ENCODINGS[quantize]

        if spec.kind == "flat":
            return faiss.IndexIDMap2(faiss.IndexFlatL2(self.dim))

        if spec.kind == "sq":
            index = faiss.IndexScalarQuantizer(self.dim, spec.qtype, faiss.METRIC_L2)
        elif spec.kind == "pq":
            m = self._pq_m_for(self.dim, spec.rate, spec.nbits)
            index = faiss.IndexPQ(self.dim, m, spec.nbits, faiss.METRIC_L2)
            assert index.search_type == faiss.IndexPQ.ST_PQ, (
                f"BallCoverFilter requires ADC (ST_PQ); got search_type={index.search_type}")
        elif spec.kind == "rabitq":
            index = faiss.IndexRaBitQ(self.dim, faiss.METRIC_L2)
            index.qb = 0
        else:
            raise ValueError(f"Unknown encoding kind {spec.kind!r} for {quantize}")

        if spec.rotate:
            index = faiss.IndexPreTransform(self._make_rotation(self.dim), index)

        return faiss.IndexIDMap2(index)


    @staticmethod
    def _make_rotation(dim: int) -> faiss.VectorTransform:
        """faiss.HadamardRotation is the better transform wherever it applies:
        same fidelity as a random rotation (measured identical to 3 decimals
        on real embeddings at dim=512), ~25% cheaper to apply, and it stores
        three sign vectors -- 6 KB at dim=512 -- rather than a dense d*d
        matrix, 1 MB at the same dimension.

        But it rounds d_out up to a power of two, and faiss refuses to invert
        it when d_out != d_in ("HadamardRotation inverse requires equal
        input/output dimensions"). That breaks sa_decode and reconstruct_batch,
        which this filter needs for _encoding_error and all_centers. At
        dim=384 it would also pad to 512 and inflate TQ3's code from 144 to
        192 bytes, breaking the 3 b/dim rate outright -- and padding then
        costs *more* accuracy than the rotation recovers (err 0.18590 at
        192 B padded, against 0.18808 at 144 B unpadded).

        So: Hadamard at power-of-two dimensions, RandomRotationMatrix
        elsewhere -- which is what dim=384 gets.

        Both are seeded explicitly rather than left to train(): a seeded
        rotation is already is_trained, so IndexPreTransform.train() leaves it
        alone. That makes the filter reproducible and lets _codebook_bytes()
        charge 4 bytes for the seed rather than the matrix it expands to.
        """
        rotation = faiss.HadamardRotation(dim)
        if rotation.d_out != dim:
            rotation = faiss.RandomRotationMatrix(dim, dim)
        rotation.init(_ROTATION_SEED)
        return rotation


    def _coder(self, index: Optional[faiss.IndexIDMap2] = None) -> faiss.Index:
        """The level that owns sa_encode/sa_decode: the pre-transform wrapper
        when there is one, so encoding applies the rotation and decoding
        undoes it. Never the leaf -- encoding at the leaf would skip the
        rotation and decode into the wrong basis.

        `index` overrides the committed one, so construct() can price an
        index it has built but not yet installed."""
        if index is None:
            index = self.all_centers_index
        return faiss.downcast_index(index.index)


    def _leaf_index(self, index: Optional[faiss.IndexIDMap2] = None) -> faiss.Index:
        """The quantizer itself, past any pre-transform -- for reading .pq /
        .sq / RaBitQ model parameters."""
        leaf = self._coder(index)
        if isinstance(leaf, faiss.IndexPreTransform):
            return faiss.downcast_index(leaf.index)
        return leaf


    def bytes_per_ball(self, index: Optional[faiss.IndexIDMap2] = None) -> int:
        """Encoded center + fp16 radius. sa_code_size() is well defined
        before training and through a pre-transform, so this is callable at
        __init__ time, when the memory budget is turned into a ball count."""
        return self._coder(index).sa_code_size() + self._BYTES_PER_FLOAT16


    def _codebook_bytes(self, index: Optional[faiss.IndexIDMap2] = None) -> int:
        """Fixed, per-filter cost that does not scale with the ball count."""
        total = _ROTATION_SEED_BYTES if _ENCODINGS[self.quantize].rotate else 0
        index = self._leaf_index(index)

        if isinstance(index, faiss.IndexPQ):
            pq = index.pq
            total += int(pq.M) * (1 << int(pq.nbits)) * int(pq.dsub) * self._BYTES_PER_FLOAT32
        elif isinstance(index, faiss.IndexScalarQuantizer):
            trained = faiss.vector_to_array(index.sq.trained)
            if len(trained) == 0:
                probe = faiss.IndexScalarQuantizer(self.dim, index.sq.qtype, faiss.METRIC_L2)
                probe.train(_PROBE_RNG.random((2, self.dim)).astype(np.float32))
                trained = faiss.vector_to_array(probe.sq.trained)
            total += int(trained.nbytes)
        elif isinstance(index, faiss.IndexRaBitQ) and index.centered:
            total += int(faiss.vector_to_array(index.center).nbytes)

        return total


    def _encoding_error(self, centers: np.ndarray) -> np.ndarray:
        """Per-center ||c - decode(encode(c))||, measured against the trained
        codebook -- never a probabilistic bound. sa_encode/sa_decode are used
        as the uniform binding fallback across index types, and at the _coder()
        level so a rotation is applied on encode and undone on decode."""
        coder = self._coder()
        if _ENCODINGS[self.quantize].kind == "flat" or not coder.is_trained:
            return np.zeros(len(centers), dtype=np.float32)
        decoded = coder.sa_decode(coder.sa_encode(centers))
        return np.linalg.norm(centers - decoded, axis=1).astype(np.float32)


    def _train_quantizer(self, centers: np.ndarray) -> faiss.IndexIDMap2:
        """Fits the one codebook this filter will ever use, on the ball
        centers themselves, and *returns* the index. Called exactly once, by
        construct().

        Returns rather than assigns so that every way this can fail happens
        before construct() has touched any filter state -- see the commit
        ordering there.

        Trained on `centers`, not on the construct set, because the centers
        are the *only* vectors this index ever encodes: queries go in raw
        (see query()) and the construct vectors are never stored. Fitting to
        a wider sample is therefore a pure loss -- an SQ's per-dimension
        min/max widens and its buckets coarsen, and PQ's k-means centroids
        drift off the centers toward the bulk distribution. Measured at
        dim=32 over 32 centers, training on the full construct set instead
        cost ~1.22x the reconstruction error for SQ4/SQ8, nothing for the
        TurboQuant variants, and turned PQ from exact (dsub=1, one codeword
        per center) into ~1.8 mean error. All of that lands on every radius
        as inflation, and so straight on the false-positive rate.

        The encoding's rate fixes the code size up front (see _ENCODINGS),
        so unlike the codebook's *contents*, nothing about the filter's byte
        footprint depends on what this sees -- only its fidelity does.
        """
        spec = _ENCODINGS[self.quantize]

        if spec.kind == "flat" or len(centers) == 0:
            return self._make_faiss_index(self.quantize)

        min_centers = (1 << spec.nbits) if spec.kind == "pq" else 2
        if len(centers) < min_centers:
            raise InsufficientMemoryError(
                f"Not enough memory to fit a {self.quantize.value} codebook: {len(centers)} ball "
                f"center(s), but this encoding needs at least {min_centers} to train on. "
                "Requires a larger max_memory or a coarser encoding.")

        index = self._make_faiss_index(self.quantize)
        leaf = faiss.downcast_index(index.index)
        if isinstance(leaf, faiss.IndexPreTransform):
            leaf = faiss.downcast_index(leaf.index)
        if isinstance(leaf, faiss.IndexPQ):
            leaf.pq.cp.min_points_per_centroid = 1

        index.train(centers)

        codebook, per_ball = self._codebook_bytes(index), self.bytes_per_ball(index)
        if self.max_memory is not None and codebook + per_ball > self.max_memory:
            raise InsufficientMemoryError(
                f"max_memory={self.max_memory} bytes cannot hold a {self.quantize.value} filter: its "
                f"fixed codebook alone is {codebook} bytes, leaving no room for even "
                f"one {per_ball}-byte ball.")

        return index


    def _tight_radii(
        self,
        vectors: np.ndarray,
        centers: np.ndarray,
        decoded: np.ndarray,
        baseline_radii: np.ndarray,
    ) -> np.ndarray:
        """The smallest radii around the *decoded* centers that still cover
        every construct vector -- the exact measurement the `radii + error`
        bound in _store_spheres only approximates.

        Two things make this tighter than that bound. First, the distance is
        measured from where the ball actually sits after encoding, so a ball
        pays for its center's error only in the direction of its own farthest
        member, instead of paying the full ||c - decode(c)|| in every
        direction. Second, a vector that some *other* ball already covers
        doesn't force its own ball to grow at all -- with fixed centers,
        covering every vector is all the no-false-negative guarantee needs,
        and which ball does the covering is free.

        Vectors are processed farthest-from-their-center first, so the balls
        a vector might hide inside are grown before the vectors that could
        hide there are tested. Radii only ever grow inside the loop, so a
        vector found covered at one point stays covered at the end, which is
        what makes the result a genuine cover rather than an ordering
        artifact.

        No ball is ever pushed past `baseline_radii + ||c - decode(c)||`, the
        radius the bound would have given it. That cap is load-bearing, not a
        formality: re-measuring lets a vector settle on a nearer *decoded*
        ball than the one construction assigned it, and without the cap that
        ball can be dragged past its own old radius -- covering points it has
        no reason to, and trading a tighter cover globally for a worse one
        right there. Every vector keeps its construction ball as a fallback,
        which by the triangle inequality always sits inside that ball's cap,
        so a legal home always exists and the cap can never deadlock.
        """
        n_centers = len(decoded)
        radii = np.zeros(n_centers, dtype=np.float32)
        if n_centers == 0 or len(vectors) == 0:
            return radii

        X = np.ascontiguousarray(vectors, dtype=np.float32)
        decoded = np.ascontiguousarray(decoded, dtype=np.float32)
        # Measured off `decoded` itself rather than _encoding_error(), so the
        # cap is consistent with the very array the radii are measured from.
        caps = np.asarray(baseline_radii, dtype=np.float32) + np.linalg.norm(
            np.ascontiguousarray(centers, dtype=np.float32) - decoded, axis=1
        ).astype(np.float32)

        probe = faiss.IndexFlatL2(self.dim)
        probe.add(decoded)
        k = min(self._COVERAGE_NEIGHBORS, n_centers)
        d2, labels = probe.search(X, k)
        dist = np.sqrt(np.maximum(d2, 0.0))

        # Each vector's construction ball, and how far it sits from that
        # ball once encoded -- the fallback the cap argument rests on.
        exact = faiss.IndexFlatL2(self.dim)
        exact.add(np.ascontiguousarray(centers, dtype=np.float32))
        _, home = exact.search(X, 1)
        home = home.ravel()
        home_dist = np.linalg.norm(X - decoded[home], axis=1).astype(np.float32)

        nearest_d = dist[:, 0]
        for i in np.argsort(-nearest_d, kind="stable"):
            row_l = labels[i]
            row_d = dist[i]
            # Scalar fast path: the nearest ball, already grown by a farther
            # member, covers the overwhelming majority.
            if nearest_d[i] <= radii[row_l[0]]:
                continue
            if (row_d <= radii[row_l]).any() or home_dist[i] <= radii[home[i]]:
                continue
            # Nearest ball that can take it without breaching its own cap;
            # the construction ball always can, so this always resolves.
            for j, d in zip(row_l, row_d):
                if d <= caps[j]:
                    radii[j] = d
                    break
            else:
                radii[home[i]] = home_dist[i]
        return radii


    def construct(self, vectors: np.ndarray) -> None:
        """Builds the filter in bulk from construct_method's output via _add_spheres().

        If no insert_method is provided, the index acts as a static filter. Otherwise,
        construct() delegates state setup to insert_method.initialize_buffers(...)
        to seed bookkeeping (e.g., MinHeapInsert) in a vectorized pass for streaming.

        This is also the only place the quantizer is ever trained (see
        _train_quantizer), and it is re-runnable: calling it again on a
        filter that has already been constructed and inserted into is what
        runner's "periodic" streaming policy does, so it fully resets the id
        pool, the per-ball inflation, and the insert strategy's bookkeeping.

        Returns nothing. This used to return estimate_hypersphere_volume's
        overlap-aware bounds on the cover's volume, which made every caller
        pay for them whether or not it read them -- and almost none did. That
        measurement now belongs to whoever wants it: runner's
        _tau_volume_bounds_log10 computes the same bounds from all_centers and
        all_radiuses, at any tau, and is called only by the worker whose
        figure needs them."""
        if len(vectors) == 0:
            self._constructed = True
            return

        centers, radii = self.construct_method.construct_filter(self, vectors)
        centers = np.ascontiguousarray(centers, dtype=np.float32)
        radii = np.asarray(radii, dtype=np.float32)
        self._reset_buffers(len(vectors))

        self.all_centers_index = self._train_quantizer(centers)
        sphere_indices = self._add_spheres(centers, radii)
        self._all_radiuses[sphere_indices] = self._fp16_ceil(self._tight_radii(vectors, centers, self.all_centers, radii))

        if self.insert_method:
            _ = self.all_centers # intialize centers view for insertion
            self.insert_method.initialize_buffers(self, sphere_indices, self.all_centers, self.all_radiuses)

        self._constructed = True
        self._all_centers = None # clear temporary centers view


    def query(self, vectors: np.ndarray, distance_threshold: float) -> list[bool]:
        if not self._constructed:
            raise ValueError("BallCoverFilter has not been constructed.")

        if self._num_spheres == 0 or len(vectors) == 0:
            return [False] * len(vectors)

        Q = np.ascontiguousarray(vectors, dtype=np.float32)
        d2, labels = self.all_centers_index.search(Q, self._num_spheres)
        surface = np.where(labels >= 0, np.sqrt(np.maximum(d2, 0.0)) - self._all_radiuses[np.maximum(labels, 0)], np.inf)
        return (surface.min(axis=1) <= distance_threshold + self._COVERAGE_EPSILON).tolist()


    def insert(self, vectors: np.ndarray) -> None:
        if not self._constructed:
            raise ValueError(
                "BallCoverFilter.insert() requires a constructed filter -- call construct() first. "
                "insert() reuses the codebook and PQ nbits that construct() fixed, and has no "
                "centers to fit them from on its own.")

        if not self._coder().is_trained:
            raise ValueError(
                "BallCoverFilter.insert() has no codebook to encode with: construct() produced "
                "zero balls, so the quantizer was never fitted. Call construct() on a non-empty "
                "vector set first.")

        if not self.insert_method:
            raise ValueError(
                "BallCoverFilter.insert() requires an insertion strategy: insert_method is None. "
                "Provide an InsertInterface implementation (e.g., NearestInsert, MinHeapInsert) "
                "when instantiating BallCoverFilter to enable streaming insertions.")

        if len(vectors) == 0:
            return

        _ = self.all_centers # intialize centers view for insertion
        self.insert_method.insert(self, vectors)
        self._all_centers = None # clear temporary centers view


    def add(self, vectors: np.ndarray) -> None:
        self.insert(vectors)


    def remove(self, vectors: np.ndarray) -> List[bool]:
        if not self._constructed:
            raise ValueError(
                "BallCoverFilter.remove() requires a constructed filter -- call construct() first.")

        if not self.insert_method:
            raise ValueError(
                "BallCoverFilter.remove() requires an insertion strategy: insert_method is None. "
                "Provide an InsertInterface implementation (e.g., NearestInsert, MinHeapInsert) "
                "when instantiating BallCoverFilter to enable streaming insertions.")

        if len(vectors) == 0:
            return []

        _ = self.all_centers # intialize centers view for insertion
        result = self.insert_method.remove(self, vectors)
        self._all_centers = None # clear temporary centers view

        return result


    def get_bytesize(self) -> int:
        if not self._constructed:
            return 0
        filter_bytes = self._num_spheres * self.bytes_per_ball() + self._codebook_bytes()
        if self.insert_method:
            filter_bytes += self.insert_method.get_bytesize(self)
        return filter_bytes


    def get_construction_bytesize(self) -> int:
        if not self.construct_method:
            return 0
        return self.construct_method.get_bytesize()