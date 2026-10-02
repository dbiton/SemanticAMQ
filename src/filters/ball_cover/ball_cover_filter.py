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
from filters.ball_cover import scan_kernels


class Quantize(Enum):
    """Center encodings, one per rung of a bits-per-dimension ladder. Byte
    figures are for dim=384 (this project's sweep dimension):

      b/dim  member   encoding                     center B   codebook
        32   FP32     raw float32                      1536   0
        16   FP16     scalar fp16                       768   0
         8   SQ8      scalar 8-bit                      384   3 KB
         8   TQ8      rotation + TurboQuant 8-bit        384   seed only
         6   SQ6      scalar 6-bit                      288   3 KB
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

    TQ1/TQ2/TQ4/TQ8 fill out the rest of faiss's TurboQuant family
    alongside TQ3 (see _ENCODINGS): evaluation-only for now (added to
    runner.QUANTIZATION_SWEEP but deliberately not to
    plot_quantization.QUANTIZE_SWEEP, so they generate rows without yet
    appearing in any figure). Unlike TQ3, none of these five have had
    rotation's benefit individually re-measured against real embeddings --
    they inherit rotate=True from TQ3 for consistency within the TurboQuant
    family, not because it has been shown to pay off at every one of these
    bit rates (a synthetic probe suggested it might not always help, e.g.
    at 8-bit -- see the results this sweep produces before trusting that).
    There is deliberately no 5-bit rung: faiss offers 5-bit TurboQuant only
    as the plain QT_5bit_tq, which folds each vector's stored exact norm into
    search() -- scoring ||q - decode(c)||^2 plus a per-center constant, not
    a distance to any point -- and it produced false negatives in the
    equal-memory sweep, so it was removed.

    Every ball also stores its radius as fp16, so a ball costs center + 2
    bytes -- see bytes_per_ball().

    CAVEAT on RABITQ: every other row is exact ADC -- search() returns
    ||q - decode(c)|| -- so the
    one-sided inflation in _add_spheres is a proof. RaBitQ instead scores through its own estimator, which is not tied
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
    # plotted) for now. All MSE-trained ("tqmse"); see the Quantize
    # docstring for why the plain 5-bit QT_5bit_tq is not among them.
    Quantize.TQ8: _Encoding(rate=8, kind="sq", qtype=faiss.ScalarQuantizer.QT_8bit_tqmse, rotate=True),
    Quantize.SQ6: _Encoding(rate=6, kind="sq", qtype=faiss.ScalarQuantizer.QT_6bit),
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
        insert_method: Optional[InsertInterface] = NearestInsert(),
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
        self._radius_inflation = np.zeros((self.max_spheres,), dtype=np.float32)

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
                ids, codes = self._stored_codes()
                self._all_centers[ids] = self._decode(codes)
        return self._all_centers[self.all_spheres]

    @property
    def all_radiuses(self) -> np.ndarray:
        """A NumPy view of active sphere radii corresponding to the active spheres."""
        return self._all_radiuses[self.all_spheres]

    @property
    def all_radius_inflation(self) -> np.ndarray:
        """Per-ball radius cost of encoding (0 for Quantize.NONE). Set by
        construct() to the stored radius minus construct_method's pre-encoding
        radius -- negative where the coverage pass more than paid for the
        center's error -- and by _add_spheres() to the center's own encoding
        error for every ball an insert strategy allocates."""
        return self._radius_inflation[self.all_spheres]


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

        new_inflation = np.zeros((new_cap,), dtype=np.float32)
        new_inflation[:current_cap] = self._radius_inflation
        self._radius_inflation = new_inflation

        if self.insert_method:
            self.insert_method.ensure_capacity(new_cap)


    def _reset_buffers(self, capacity: int) -> None:
        self._ensure_capacity(capacity)
        self._num_spheres = 0
        self.all_centers_index = self._make_faiss_index(self.quantize)
        self._all_radiuses[:] = 0.0
        self._radius_inflation[:] = 0.0
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
        encoding_error = self._encoding_error(centers)
        self._all_radiuses[new_sphere_indices] = self._fp16_ceil(radii + encoding_error)
        self._radius_inflation[new_sphere_indices] = encoding_error
        self._num_spheres += num_new_spheres

        if self._all_centers is not None:
            # Decoded, not `centers`: insert strategies read this cache to
            # decide what a ball already covers, and the stored radius is
            # measured around where the ball actually sits. Caching the raw
            # center let MinHeapInsert treat a vector within r + err of it as
            # covered while query() scored it against a center up to err
            # farther away -- 27.7% false negatives at TQ1 on WildChat.
            # add_with_ids appends, so the new balls are the last rows.
            _, codes = self._stored_codes()
            self._all_centers[new_sphere_indices] = self._decode(codes[-num_new_spheres:])

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
        self._radius_inflation[sphere_indices] = 0.0
        self._num_spheres = num_active_spheres

        if self._all_centers is not None:
            self._all_centers[sphere_indices] = 0.0


    def _calculate_max_spheres(self, max_memory: Optional[int]) -> int:
        if max_memory is None:
            return sys.maxsize
        usable = max(0, max_memory - self._codebook_bytes())
        if self.insert_method is None:
            return max(1, usable // self.bytes_per_ball())
        # The insert strategy's own state counts against max_memory too, at
        # its peak over a stream -- see InsertInterface.max_spheres_for_budget.
        return max(1, self.insert_method.max_spheres_for_budget(usable, self.bytes_per_ball()))


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


    def _rotation(self) -> Optional[faiss.VectorTransform]:
        """The orthonormal rotation in front of the quantizer, or None."""
        coder = self._coder()
        if isinstance(coder, faiss.IndexPreTransform):
            return faiss.downcast_VectorTransform(coder.chain.at(0))
        return None


    def _stored_codes(self) -> tuple:
        """(ids, codes) for every stored ball, in the index's storage order:
        row i of `codes` is ball ids[i]. `codes` is a zero-copy view of the
        leaf's code array, so it is only valid until the index next changes.

        Reading the codes directly is what makes decoding cheap.
        IndexIDMap2.reconstruct_batch decodes one id at a time, and through a
        rotation that means one 384x384 matrix-vector product per ball: 850 ms
        for 16K TQ4 balls, against ~6 ms to decode the same codes in bulk."""
        leaf = self._leaf_index()
        n, size = int(leaf.ntotal), int(leaf.code_size)
        ids = faiss.vector_to_array(self.all_centers_index.id_map).astype(np.int64)
        if n == 0:
            return ids, np.empty((0, size), dtype=np.uint8)
        codes = np.asarray(faiss.rev_swig_ptr(leaf.codes.data(), n * size)).reshape(n, size)
        return ids, codes


    def _decode_rotated(self, codes: np.ndarray) -> np.ndarray:
        """Decoded centers in the quantizer's own basis -- past the rotation,
        where there is one. The rotation is orthonormal, so distances
        measured here between rotated queries and these centers equal
        distances between raw queries and the decoded centers."""
        if _ENCODINGS[self.quantize].kind == "flat":
            return codes.view(np.float32)
        return self._leaf_index().sa_decode(np.ascontiguousarray(codes))


    def _decode(self, codes: np.ndarray) -> np.ndarray:
        """Decoded centers in the input basis: decode(c), as reconstruct
        returns it, but in one batched pass."""
        decoded = self._decode_rotated(codes)
        rotation = self._rotation()
        if rotation is not None and len(decoded):
            decoded = rotation.reverse_transform(decoded)
        return np.ascontiguousarray(decoded, dtype=np.float32)


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
        bound in _add_spheres only approximates.

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
            self._reset_buffers(0)
            self._constructed = True
            return

        centers, radii = self.construct_method.construct_filter(self, vectors)
        centers = np.ascontiguousarray(centers, dtype=np.float32)
        radii = np.asarray(radii, dtype=np.float32)
        self._reset_buffers(len(centers))

        self.all_centers_index = self._train_quantizer(centers)
        sphere_indices = self._add_spheres(centers, radii)
        stored = self._fp16_ceil(self._tight_radii(vectors, centers, self.all_centers, radii))
        self._all_radiuses[sphere_indices] = stored
        self._radius_inflation[sphere_indices] = stored - radii

        if self.insert_method:
            _ = self.all_centers # intialize centers view for insertion
            self.insert_method.initialize_buffers(self, sphere_indices, self.all_centers, self.all_radiuses,
                                                  vectors=vectors)

        self._store_largest_first()
        self._constructed = True
        self._all_centers = None # clear temporary centers view


    def _store_largest_first(self) -> None:
        """Reorders the index's storage -- codes and id_map together -- so
        the largest balls are scanned first. Ids, radii and every per-id
        array are untouched, so no query decision changes, only how soon
        it is reached: a query stops at the first ball that accepts it, and
        large balls accept most often. Measured with scan_kernels on the five
        blocklist covers (TQ4, batch 32, random query mix): 1.20-1.48x at
        16K balls and 1.16-1.37x at 2K over construction order. Neighbouring
        balls then also tend to abandon at the same word, which the branch
        predictor rewards.

        Static filters only. all_centers un-rotates the decoded codes with
        one batched GEMM, whose float32 rounding depends on a row's position
        in the batch (up to 1.2e-7 per coordinate at dim=384), and insert
        strategies decide from all_centers: reordering a streaming filter
        changed the cover cap_merge built after 1500 TQ4 inserts, though
        not a single query decision. With no insert strategy nothing
        decides from all_centers after construct(), so the cover stays
        bit-identical."""
        if self.insert_method is not None:
            return
        ids, codes = self._stored_codes()
        order = np.argsort(-self._all_radiuses[ids], kind="stable")
        if len(order) < 2 or (order == np.arange(len(order))).all():
            return
        codes[:] = codes[order]
        id_map = self.all_centers_index.id_map
        faiss.rev_swig_ptr(id_map.data(), id_map.size())[:] = ids[order]
        self.all_centers_index.construct_rev_map()


    def surface_distance(self, vectors: np.ndarray) -> np.ndarray:
        """Per query, the smallest (distance to a ball's center - its radius)
        over every ball; inf with no balls. query() accepts the rows where
        this is <= distance_threshold + _COVERAGE_EPSILON (up to float32
        rounding far from any center -- see _scan), so a caller sweeping many
        thresholds over one cover can scan once."""
        if not self._constructed:
            raise ValueError("BallCoverFilter has not been constructed.")

        if self._num_spheres == 0 or len(vectors) == 0:
            return np.full(len(vectors), np.inf)

        Q = np.ascontiguousarray(vectors, dtype=np.float32)
        if _ENCODINGS[self.quantize].kind == "rabitq":
            return self._estimator_surface_distance(Q)
        return self._scan(Q, distance_threshold=None)


    def query(self, vectors: np.ndarray, distance_threshold: float) -> list[bool]:
        if not self._constructed:
            raise ValueError("BallCoverFilter has not been constructed.")

        if self._num_spheres == 0 or len(vectors) == 0:
            return [False] * len(vectors)

        if _ENCODINGS[self.quantize].kind == "rabitq":
            return (self.surface_distance(vectors) <= distance_threshold + self._COVERAGE_EPSILON).tolist()
        Q = np.ascontiguousarray(vectors, dtype=np.float32)
        if self._use_kernel(len(Q)):
            return self._kernel_accepts(Q, distance_threshold).tolist()
        if len(Q) < self._SCAN_MIN_BATCH:
            return self._range_accepts(Q, distance_threshold).tolist()
        return self._scan(Q, distance_threshold=distance_threshold).tolist()


    # 4-bit scalar quantizers whose decode is a per-dimension table lookup on
    # each nibble, which is what scan_kernels.nibble_accepts reads.
    _NIBBLE_QTYPES = (faiss.ScalarQuantizer.QT_4bit, faiss.ScalarQuantizer.QT_4bit_tqmse)

    # Batches below these sizes go to scan_kernels instead of _scan /
    # _range_accepts. Measured on the five blocklist covers (random query
    # mix, one pinned P-core), kernel q/s over the old path's:
    #   TQ4, 16K balls: 14-23x at 1 query, 1.7-4.3x at 32, 1.2-3.4x at 64
    #   TQ4,  2K balls:  6-9x  at 1 query, 1.3-2.5x at 32, 0.96-1.8x at 64
    #   FP32, 16K balls: 1.1-4.5x at 1-4 queries, 0.73-2.9x at 8
    #   FP32,  2K balls: 0.8-1.7x at 1 query -- faiss's vectorized flat scan
    #                    wins on a cache-sized cover, hence the ball floor.
    # FP32 gains less because a float32 center is 8x the bytes of a TQ4
    # code, so abandoning early saves memory traffic rather than arithmetic.
    _NIBBLE_KERNEL_MAX_BATCH = 64
    _FLAT_KERNEL_MAX_BATCH = 8
    _FLAT_KERNEL_MIN_BALLS = 4096


    def _kernel_kind(self) -> Optional[str]:
        spec = _ENCODINGS[self.quantize]
        if spec.kind == "flat":
            return "flat"
        if spec.kind == "sq" and spec.qtype in self._NIBBLE_QTYPES:
            return "nibble"
        return None


    def _use_kernel(self, num_queries: int) -> bool:
        kind = self._kernel_kind()
        if kind == "nibble":
            return num_queries < self._NIBBLE_KERNEL_MAX_BATCH
        if kind == "flat":
            return (num_queries < self._FLAT_KERNEL_MAX_BATCH
                    and self._num_spheres >= self._FLAT_KERNEL_MIN_BALLS)
        return False


    def _nibble_table(self, rows: int) -> np.ndarray:
        """(rows, 16) float64: row j, column v is what the quantizer decodes
        nibble v of dimension j to, read off sa_decode itself (one probe code
        per nibble value) so it is decode(c) by definition, not a
        re-derivation of faiss's formula. Rows past dim stay zero. Rebuilt
        per call from the trained codebook get_bytesize() already charges,
        rather than cached, so nothing decoded outlives the call."""
        leaf = self._leaf_index()
        probe = np.repeat((np.arange(16, dtype=np.uint8) * 17)[:, None], int(leaf.code_size), axis=1)
        table = np.zeros((rows, 16), dtype=np.float64)
        table[:self.dim] = leaf.sa_decode(probe).T
        return table


    def _kernel_accepts(self, Q: np.ndarray, distance_threshold: float) -> np.ndarray:
        """query() through scan_kernels: exact distance to every decoded
        center, abandoning a ball once a partial sum is provably past its
        reach -- see scan_kernels for the bound that keeps this exact."""
        ids, codes = self._stored_codes()
        radii = self._all_radiuses[ids]
        tau, eps = float(distance_threshold), float(self._COVERAGE_EPSILON)
        rotation = self._rotation()
        Qr = np.ascontiguousarray(rotation.apply(Q) if rotation is not None else Q, dtype=np.float32)
        if self._kernel_kind() == "flat":
            return scan_kernels.flat_accepts(codes.view(np.float32), Qr, radii, tau, eps)
        if codes.shape[1] % 8:
            # The kernel reads codes a 64-bit word at a time; padding bytes
            # meet zero table rows, which add exactly nothing.
            codes = np.pad(codes, ((0, 0), (0, -codes.shape[1] % 8)))
        return scan_kernels.nibble_accepts(codes, Qr, self._nibble_table(2 * codes.shape[1]), radii, tau, eps)


    # Below this many queries, _scan's per-call decode of every code is not
    # amortized and _range_accepts is faster. Measured on 16K-ball WildChat
    # covers, one pinned core: a lone query runs at 197 vs 165 q/s (TQ4), 864
    # vs 390 (PQ8@1), 1,187 vs 658 (FP32); from 4 queries up _scan wins or
    # ties for every encoding.
    _SCAN_MIN_BATCH = 4


    def _range_accepts(self, Q: np.ndarray, distance_threshold: float) -> np.ndarray:
        """query() for a handful of queries: faiss range_search at the
        largest reach any ball has, then each hit checked against its own
        ball's reach -- the same d^2 <= (r + tau + eps)^2 test _scan applies.
        Exact for the same reason _scan is: faiss scores ||q - decode(c)||^2
        directly here (SQ/PQ ADC, and IndexFlat below 20 queries), with no
        GEMM cancellation, and needs no k-wide heap."""
        reach = self._all_radiuses[self.all_spheres] + float(distance_threshold) + self._COVERAGE_EPSILON
        max_reach = float(reach.max())
        if not np.isfinite(max_reach):
            return np.ones(len(Q), dtype=bool)
        if max_reach < 0.0:
            return np.zeros(len(Q), dtype=bool)
        # faiss keeps d^2 strictly below the radius; one ulp up keeps "<=".
        radius = float(np.nextafter(np.float32(max_reach * max_reach), np.float32(np.inf)))
        lims, d2, labels = self.all_centers_index.range_search(Q, radius)
        ball_reach = self._all_radiuses[labels].astype(np.float64) + float(distance_threshold) + self._COVERAGE_EPSILON
        hit = (ball_reach >= 0.0) & (d2 <= ball_reach * ball_reach)
        accepted = np.zeros(len(Q), dtype=bool)
        owner = np.repeat(np.arange(len(Q)), np.diff(lims).astype(np.int64))
        accepted[owner[hit]] = True
        return accepted


    # Balls decoded per block of _scan, and queries per GEMM: bounds the
    # scratch at (block x dim) decoded floats plus (chunk x block) distances,
    # ~22 MB in all, whatever the cover size. That scratch is transient --
    # only the codes persist, which is what get_bytesize() charges.
    _SCAN_BLOCK = 4096
    _SCAN_QUERY_CHUNK = 1024

    # _scan computes ||q - c||^2 as ||q||^2 + ||c||^2 - 2<q, c> in float32,
    # which cancels for near-identical vectors. Its rounding error is at most
    # about (2d + 4) * 2^-24 * (||q||^2 + ||c||^2) -- 9e-5 for unit vectors at
    # d=384 -- so every (query, ball) pair whose float32 value lands within
    # that bound of the decision boundary is re-measured directly in float64.
    # Accept decisions are therefore exact up to float64 rounding, not up to
    # float32 cancellation -- the failure the 1e-3 _COVERAGE_EPSILON was
    # otherwise the only guard against at tau = 0. The bound scales with the
    # norms, so it holds for un-normalized vectors too (where refinement
    # just fires more often).
    #
    # surface_distance has no boundary to test against, so it instead
    # re-measures every pair with d^2 < _SCAN_NEAR_D2 * s^2 / 4, where s =
    # ||q||^2 + max ||c||^2 (so 1e-2 for unit vectors). Past that, sqrt()
    # damps the same error to under 1e-3 even at the worst-case bound
    # (typically ~5e-5), and that floor scales with the data, not against it.
    _SCAN_NEAR_D2 = 1e-2


    def _scan(self, Q: np.ndarray, distance_threshold: Optional[float]) -> np.ndarray:
        """Scores every query against every ball by exact L2 distance to the
        decoded center, in blocks: decode a block of codes, one GEMM against
        the queries, compare. Returns the accept mask when distance_threshold
        is given, otherwise surface_distance's per-query minimum.

        Replaces faiss's search(Q, k=num_spheres), which kept a k-wide result
        heap per query only for the caller to take one minimum, and scored
        TQ codes through a per-element scalar path: 180 q/s per core at 16K
        TQ4 balls. The distances are the same ones query() always used -- to
        decode(c), not an estimate -- so the triangle-inequality argument is
        unchanged, and the boundary refinement above makes them more precise
        than faiss's float32 GEMM path was.

        Works in the quantizer's rotated basis: rotating the queries once
        costs one GEMM per batch, where un-rotating the centers would cost
        one per block. The rotation is orthonormal, so distances agree.

        With a threshold, the test is d^2 <= (r + tau + eps)^2, which needs no
        sqrt, and a query drops out of the remaining blocks as soon as any
        ball accepts it."""
        ids, codes = self._stored_codes()
        radii = self._all_radiuses[ids]
        rotation = self._rotation()
        Qr = np.ascontiguousarray(rotation.apply(Q) if rotation is not None else Q, dtype=np.float32)
        q_norms = np.einsum("ij,ij->i", Qr, Qr)
        rel_error = (2 * self.dim + 4) * 2.0 ** -24

        if distance_threshold is None:
            out = np.full(len(Q), np.inf, dtype=np.float32)
        else:
            out = np.zeros(len(Q), dtype=bool)
            reach = radii.astype(np.float64) + float(distance_threshold) + self._COVERAGE_EPSILON
            # A negative reach admits nothing, which squaring would lose.
            reach_sq = np.where(reach >= 0.0, reach * reach, -1.0)

        for start in range(0, len(ids), self._SCAN_BLOCK):
            stop = min(start + self._SCAN_BLOCK, len(ids))
            C = self._decode_rotated(codes[start:stop])
            c_norms = np.einsum("ij,ij->i", C, C)
            pending = np.arange(len(Q)) if distance_threshold is None else np.flatnonzero(~out)
            for q0 in range(0, len(pending), self._SCAN_QUERY_CHUNK):
                rows = pending[q0:q0 + self._SCAN_QUERY_CHUNK]
                scale = q_norms[rows] + c_norms.max()
                d2 = Qr[rows] @ C.T
                d2 *= -2.0
                d2 += q_norms[rows, None]
                d2 += c_norms[None, :]
                if distance_threshold is None:
                    near = self._SCAN_NEAR_D2 * scale * scale / 4.0
                    out[rows] = np.minimum(out[rows], self._block_surface(
                        d2, Qr[rows], C, radii[start:stop], near))
                else:
                    out[rows] |= self._block_accepts(
                        d2, Qr[rows], C, reach_sq[start:stop], rel_error * scale)
            if distance_threshold is not None and out.all():
                break
        return out


    @staticmethod
    def _exact_d2(Qb: np.ndarray, C: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
        diff = Qb[i].astype(np.float64) - C[j].astype(np.float64)
        return np.einsum("ij,ij->i", diff, diff)


    def _block_accepts(self, d2: np.ndarray, Qb: np.ndarray, C: np.ndarray,
                       reach_sq: np.ndarray, slack: np.ndarray) -> np.ndarray:
        """Rows of d2 with some ball in reach. A pair more than its row's
        `slack` inside the boundary is a hit outright; one within `slack`
        of it either way is re-measured in float64. Overwrites d2."""
        reach32 = reach_sq.astype(np.float32)
        slack = slack.astype(np.float32)[:, None]
        d2 += slack
        hit = (d2 <= reach32[None, :]).any(axis=1)
        undecided = np.flatnonzero(~hit)
        if len(undecided):
            i, j = np.nonzero(d2[undecided] - 2.0 * slack[undecided] < reach32[None, :])
            if len(i):
                i = undecided[i]
                ok = self._exact_d2(Qb, C, i, j) <= reach_sq[j]
                hit[i[ok]] = True
        return hit


    def _block_surface(self, d2: np.ndarray, Qb: np.ndarray, C: np.ndarray,
                       radii: np.ndarray, near: np.ndarray) -> np.ndarray:
        dist = np.sqrt(np.maximum(d2, 0.0))
        i, j = np.nonzero(d2 < near.astype(np.float32)[:, None])
        if len(i):
            dist[i, j] = np.sqrt(self._exact_d2(Qb, C, i, j))
        return (dist - radii[None, :]).min(axis=1)


    def _estimator_surface_distance(self, Q: np.ndarray) -> np.ndarray:
        """RABITQ only: faiss's own search, which scores through RaBitQ's
        distance estimator rather than ||q - decode(c)|| (see the Quantize
        docstring). Kept as it was so RABITQ rows keep measuring the
        estimator the paper describes; _scan would quietly turn them into
        exact-decode rows."""
        d2, labels = self.all_centers_index.search(Q, self._num_spheres)
        surface = np.where(labels >= 0, np.sqrt(np.maximum(d2, 0.0)) - self._all_radiuses[np.maximum(labels, 0)], np.inf)
        return surface.min(axis=1)


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
        # Strategies that grow a ball in place (NearestInsert,
        # LeastInflationInsert) write raw float32 radii straight into
        # _all_radiuses, so re-settle every active ball onto the fp16 grid
        # bytes_per_ball() charges for. Idempotent, and only ever rounds up.
        active = self.all_spheres
        self._all_radiuses[active] = self._fp16_ceil(self._all_radiuses[active])
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