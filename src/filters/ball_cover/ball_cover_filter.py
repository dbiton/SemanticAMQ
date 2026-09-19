import numpy as np
import faiss
import sys
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional
from hyperspheres_volume import estimate_hypersphere_volume
from filters.filter import Filter
from filters.ball_cover.ball_interfaces import ConstructInterface, InsertInterface
from filters.ball_cover.gonzales_construct import GonzalesConstruct
from filters.ball_cover.nearest_insert import NearestInsert
from filters.ball_cover.least_inflation_insert import LeastInflationInsert
from filters.ball_cover.exhaustive_merge_insert import ExhaustiveMergeInsert


class Quantize(Enum):
    """Center encodings, one per rung of a bits-per-dimension ladder. Byte
    figures are for dim=384 (this project's sweep dimension):

      b/dim  member   encoding                     center B   codebook
        32   FP32     raw float32                      1536   0
        16   FP16     scalar fp16                       768   0
         8   SQ8      scalar 8-bit                      384   3 KB
         6   SQ6      scalar 6-bit                      288   3 KB
         4   SQ4      scalar 4-bit                      192   3 KB
         3   TQ3      rotation + TurboQuant 3-bit       144   60 B + seed
         2   PQ4_R2   PQ, 4-bit subquantizers (m=192)    96   24 KB
         2   PQ8_R2   PQ, 8-bit subquantizers (m=96)     96   384 KB
         1   PQ4_R1   PQ, 4-bit subquantizers (m=96)     48   24 KB
         1   RABITQ   rotation + RaBitQ                  56   seed only

    Every ball also stores its radius as fp16, so a ball costs center + 2
    bytes -- see bytes_per_ball().

    Member names are dimension-independent: PQ4_R2 reads "4-bit
    subquantizers at a 2 bit/dim rate", and the subquantizer count m is
    derived from dim by _pq_m_for(), so the rates hold at any dimension that
    admits them. The m values above are what dim=384 resolves to.
    """
    FP32 = "fp32"
    NONE = "fp32"     # back-compat alias: FP32 *is* the unquantized baseline
    FP16 = "fp16"
    SQ8 = "SQ8"
    SQ6 = "SQ6"
    SQ4 = "SQ4"
    TQ3 = "TQ3"
    PQ4_R2 = "PQ4@2"
    PQ8_R2 = "PQ8@2"
    PQ4_R1 = "PQ4@1"
    RABITQ = "RaBitQ"


class InsufficientMemoryError(ValueError):
    """Raised when max_memory cannot support the requested encoding at all --
    either its fixed codebook alone exhausts the budget (PQ8_R2's 384 KB is
    the one that bites), or there are too few ball centers to fit a codebook
    to."""


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


_ENCODINGS = {
    Quantize.FP32: _Encoding(rate=32, kind="flat"),
    Quantize.FP16: _Encoding(rate=16, kind="sq", qtype=faiss.ScalarQuantizer.QT_fp16),
    Quantize.SQ8: _Encoding(rate=8, kind="sq", qtype=faiss.ScalarQuantizer.QT_8bit),
    Quantize.SQ6: _Encoding(rate=6, kind="sq", qtype=faiss.ScalarQuantizer.QT_6bit),
    Quantize.SQ4: _Encoding(rate=4, kind="sq", qtype=faiss.ScalarQuantizer.QT_4bit),
    # QT_3bit_tqmse is the MSE-trained TurboQuant variant; there is no plain
    # (min-max) 3-bit scalar quantizer in faiss to compare it against.
    Quantize.TQ3: _Encoding(rate=3, kind="sq", qtype=faiss.ScalarQuantizer.QT_3bit_tqmse, rotate=True),
    Quantize.PQ4_R2: _Encoding(rate=2, kind="pq", nbits=4),
    Quantize.PQ8_R2: _Encoding(rate=2, kind="pq", nbits=8),
    Quantize.PQ4_R1: _Encoding(rate=1, kind="pq", nbits=4),
    Quantize.RABITQ: _Encoding(rate=1, kind="rabitq", rotate=True),
}


class BallCoverFilter(Filter):
    _BYTES_PER_FLOAT32 = 4
    _BYTES_PER_FLOAT64 = 8
    _BYTES_PER_FLOAT16 = 2

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

    def __init__(
        self,
        dim: int,
        construct_method: Optional[ConstructInterface] = None,
        insert_method: Optional[InsertInterface] = None,
        quantize: Quantize = Quantize.NONE,
        max_memory: Optional[int] = None,
        max_spheres: Optional[int] = None,
        insert_memory_percent: Optional[float] = 0.0
    ) -> None:
        super().__init__(dim, max_memory=max_memory)
        self.quantize = quantize
        # Constructed fresh here, not as parameter defaults: a mutable default
        # argument is evaluated once at def-time and shared by every call that
        # doesn't override it. Both defaults are stateless (harmless to
        # share), but strategies with real per-filter state (e.g.
        # MinHeapInsert's _sphere_versions/_minimal_pairs) must not be shared
        # this way if passed explicitly across filters in the same process.
        self.construct_method = construct_method if construct_method is not None else GonzalesConstruct()
        self.insert_method = insert_method if insert_method is not None else NearestInsert()
        self.insert_memory_percent = insert_memory_percent
        # The codebook is fitted once, by construct(), and never retrained:
        # insert() encodes new centers with whatever construct() fit.
        # Retraining would re-encode every stored center under a new codebook
        # while their radii still carry the old codebook's inflation -- and
        # the unquantized centers needed to recompute that inflation no
        # longer exist.
        self.all_centers_index = self._make_faiss_index(self.quantize)
        self.max_spheres = (max_spheres if max_spheres is not None 
                    else self._calculate_max_spheres(self.max_memory))
        self.max_construct_spheres = self._calculate_max_construct_spheres(self.insert_memory_percent)
        self._constructed = False

        self._num_spheres = 0
        self._all_spheres = np.arange(self.max_spheres, dtype=np.int32)
        self._all_centers = None  # lazily rebuilt cache, not persistent state
        self._all_radiuses = np.empty((self.max_spheres,), dtype=np.float32)
        # Per-ball reconstruction error (||center - decode(encode(center))||),
        # 0 for Quantize.NONE: added onto each radius by _store_spheres, once,
        # to preserve the no-false-negative guarantee under lossy encoding.
        self._radius_inflation = np.zeros((self.max_spheres,), dtype=np.float32)

        self.insert_method.ensure_capacity(self.max_spheres)

    @property
    def all_spheres(self) -> np.ndarray:
        return self._all_spheres[:self._num_spheres]

    @property
    def all_centers(self) -> np.ndarray:
        if self._all_centers is None:
            self._all_centers = np.empty((len(self._all_radiuses), self.dim), dtype=np.float32)
            if self._num_spheres > 0:
                sphere_indices = np.ascontiguousarray(self.all_spheres, dtype=np.int64)
                self._all_centers[self.all_spheres] = self.all_centers_index.reconstruct_batch(sphere_indices)
        return self._all_centers[self.all_spheres]

    @property
    def all_radiuses(self) -> np.ndarray:
        return self._all_radiuses[self.all_spheres]

    @property
    def all_radius_inflation(self) -> np.ndarray:
        """Per-ball radius inflation (0 for Quantize.NONE) -- see
        _radius_inflation's docstring in __init__."""
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

        self.insert_method.ensure_capacity(new_cap)

    def _get_new_index(self) -> int:
        if self._num_spheres == len(self._all_radiuses):
            self._ensure_capacity(len(self._all_radiuses) * 2)
        return self._all_spheres[self._num_spheres]

    def _reset_buffers(self, capacity: int) -> None:
        self._ensure_capacity(capacity)
        self._num_spheres = 0
        self._all_centers = None
        # Back to an untrained index: construct() has to run again to refit
        # the codebook before any sphere can be stored.
        self.all_centers_index = self._make_faiss_index(self.quantize)
        self._constructed = False
        self.insert_method.reset_buffers(capacity)

    def _calculate_max_spheres(self, max_memory: Optional[int]) -> int:
        if max_memory is None:
            # When unconstrained, fallback to standard integer bounds to avoid crash
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
            f"with {nbits}-bit codes."
        )

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
            # ST_PQ is asymmetric distance computation (ADC): the raw float32
            # query is scored against the *decoded* stored codes, so the query
            # is never encoded. That is what makes the one-sided radius
            # inflation in _store_spheres correct and tight -- see query().
            # ST_SDC/ST_polysemous encode the query too and would silently
            # break the zero-false-negative guarantee, so pin it.
            assert index.search_type == faiss.IndexPQ.ST_PQ, (
                f"BallCoverFilter requires ADC (ST_PQ); got search_type={index.search_type}"
            )
        elif spec.kind == "rabitq":
            index = faiss.IndexRaBitQ(self.dim, faiss.METRIC_L2)
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

    def _coder(self) -> faiss.Index:
        """The level that owns sa_encode/sa_decode: the pre-transform wrapper
        when there is one, so encoding applies the rotation and decoding
        undoes it. Never the leaf -- encoding at the leaf would skip the
        rotation and decode into the wrong basis."""
        return faiss.downcast_index(self.all_centers_index.index)

    def _leaf_index(self) -> faiss.Index:
        """The quantizer itself, past any pre-transform -- for reading .pq /
        .sq / RaBitQ model parameters."""
        index = self._coder()
        if isinstance(index, faiss.IndexPreTransform):
            return faiss.downcast_index(index.index)
        return index

    def bytes_per_ball(self) -> int:
        """Encoded center + fp16 radius. sa_code_size() is well defined
        before training and through a pre-transform, so this is callable at
        __init__ time, when the memory budget is turned into a ball count."""
        return self._coder().sa_code_size() + self._BYTES_PER_FLOAT16

    def _codebook_bytes(self) -> int:
        """Fixed, per-filter cost that does not scale with the ball count."""
        total = _ROTATION_SEED_BYTES if _ENCODINGS[self.quantize].rotate else 0
        index = self._leaf_index()

        if isinstance(index, faiss.IndexPQ):
            # Computed from the layout, not from pq.centroids.nbytes, which
            # reads 0 until the index is trained -- and __init__ sizes the
            # ball budget against an untrained one.
            pq = index.pq
            total += int(pq.M) * (1 << int(pq.nbits)) * int(pq.dsub) * self._BYTES_PER_FLOAT32
        elif isinstance(index, faiss.IndexScalarQuantizer):
            trained = faiss.vector_to_array(index.sq.trained)
            if len(trained) == 0:
                probe = faiss.IndexScalarQuantizer(self.dim, index.sq.qtype, faiss.METRIC_L2)
                # Random probes, not zeros: the MSE-trained (TurboQuant)
                # variants divide through by the spread while training.
                probe.train(np.random.rand(2, self.dim).astype(np.float32))
                trained = faiss.vector_to_array(probe.sq.trained)
            total += int(trained.nbytes)
        elif isinstance(index, faiss.IndexRaBitQ) and index.centered:
            total += int(faiss.vector_to_array(index.center).nbytes)

        return total

    def _train_quantizer(self, centers: np.ndarray) -> None:
        """Fits the one codebook this filter will ever use, on the ball
        centers themselves. Called exactly once, by
        construct().

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
        if _ENCODINGS[self.quantize].kind == "flat":
            self.all_centers_index = self._make_faiss_index(self.quantize)
            return

        if len(centers) < 2:
            raise InsufficientMemoryError(
                f"Not enough memory to fit a {self.quantize.value} codebook: {len(centers)} ball "
                "center(s) is too few to train on (need >= 2, which requires a larger "
                "max_memory or a smaller ball budget)."
            )

        index = self._make_faiss_index(self.quantize)
        leaf = faiss.downcast_index(index.index)
        if isinstance(leaf, faiss.IndexPreTransform):
            leaf = faiss.downcast_index(leaf.index)
        if isinstance(leaf, faiss.IndexPQ):
            # k-means over ball centers, of which there are usually far fewer
            # than the 39 * ksub faiss would like. The codebook is allowed to
            # come out under-filled rather than refusing to build.
            leaf.pq.cp.min_points_per_centroid = 1

        index.train(centers)
        self.all_centers_index = index

        if self.max_memory is not None and self._codebook_bytes() + self.bytes_per_ball() > self.max_memory:
            raise InsufficientMemoryError(
                f"max_memory={self.max_memory} bytes cannot hold a {self.quantize.value} filter: its "
                f"fixed codebook alone is {self._codebook_bytes()} bytes, leaving no room for even "
                f"one {self.bytes_per_ball()}-byte ball."
            )

    def _encoding_error(self, centers: np.ndarray) -> np.ndarray:
        """Per-center ||c - decode(encode(c))||, measured against the trained
        codebook -- never a probabilistic bound. sa_encode/sa_decode are used
        as the uniform binding fallback across index types, and at the _coder()
        level so a rotation is applied on encode and undone on decode."""
        if _ENCODINGS[self.quantize].kind == "flat":
            return np.zeros(len(centers), dtype=np.float32)
        coder = self._coder()
        decoded = coder.sa_decode(coder.sa_encode(centers))
        return np.linalg.norm(centers - decoded, axis=1).astype(np.float32)

    def _store_spheres(self, sphere_indices: np.ndarray, centers: np.ndarray, radii: np.ndarray) -> None:
        """Writes (center, radius) pairs into the already-trained index,
        inflating each radius by that center's own encoding error.

        The inflation is applied exactly once, here, at the moment a center
        is written -- which is the whole reason nothing rebuilds. A radius
        only needs inflating when it was measured against a center that is
        about to be replaced by its lossy encoding (construct()'s radii, and
        any center an insert strategy relocates). Once stored, `all_centers`
        hands back the *decoded* center, so every radius an insert strategy
        derives from it is already correct with respect to what query()
        scores against, and must not be inflated again. Re-inflating on each
        touch is what made the old rebuild path grow radii without bound.
        """
        sphere_indices = np.ascontiguousarray(sphere_indices, dtype=np.int64)
        if len(sphere_indices) == 0:
            return
        centers = np.ascontiguousarray(centers, dtype=np.float32)
        radii = np.asarray(radii, dtype=np.float32)

        error = self._encoding_error(centers)
        # add_with_ids appends rather than overwrites, so an id being
        # rewritten (a relocated/merged ball) has to be dropped first.
        if self.all_centers_index.ntotal > 0:
            self.all_centers_index.remove_ids(faiss.IDSelectorBatch(sphere_indices))
        self.all_centers_index.add_with_ids(centers, sphere_indices)
        self._all_radiuses[sphere_indices] = self._fp16_ceil(radii + error)
        self._radius_inflation[sphere_indices] = error

    def _drop_spheres(self, sphere_indices: np.ndarray) -> None:
        """Evicts ids an insert strategy retired (merged away, absorbed) from
        the index, so ntotal keeps tracking _num_spheres."""
        sphere_indices = np.ascontiguousarray(sphere_indices, dtype=np.int64)
        if len(sphere_indices) == 0 or self.all_centers_index.ntotal == 0:
            return
        self.all_centers_index.remove_ids(faiss.IDSelectorBatch(sphere_indices))
        self._radius_inflation[sphere_indices] = 0.0

    def construct(self, vectors: np.ndarray) -> Tuple[float, float]:
        """Builds directly from construct_method's output, independent of
        insert_method: routing through insert_method.add_sphere() here was
        redundant (its result was immediately overwritten below) and, for
        strategies with per-ball bookkeeping (e.g. MinHeapInsert's
        merge-candidate heap), unintentionally seeded that state from
        construction -- a cost only the streaming policy needs, paid by every
        one-shot construct-then-query use of this filter.

        This is also the only place the quantizer is ever trained, and the
        only place PQ's nbits is chosen (see _train_quantizer)."""
        centers, radii = self.construct_method.construct_filter(self, vectors)
        centers = np.ascontiguousarray(centers, dtype=np.float32)
        radii = np.asarray(radii, dtype=np.float32)
        n = len(centers)

        self._ensure_capacity(max(n, 1))
        self._num_spheres = 0
        self._all_centers = None
        self._radius_inflation[:] = 0.0

        if n == 0:
            self.all_centers_index = self._make_faiss_index(Quantize.FP32)
            self._constructed = True
            return self._log_volume_bounds()

        self._train_quantizer(centers)

        sphere_indices = np.arange(n, dtype=np.int64)
        self._all_spheres[:n] = sphere_indices.astype(np.int32)
        self._num_spheres = n
        # construct_method measures radii against the true fp32 centers, so
        # this is where those radii pick up their encoding error.
        self._store_spheres(sphere_indices, centers, radii)
        self._constructed = True

        return self._log_volume_bounds()

    def query(self, vectors: np.ndarray, distance_threshold: float) -> list[bool]:
        if not self._constructed:
            raise ValueError("BallCoverFilter has not been constructed.")

        if self._num_spheres == 0 or len(vectors) == 0:
            return [False] * len(vectors)

        # Q goes in raw and is never encoded: both IndexPQ (search_type
        # ST_PQ, pinned in _make_faiss_index) and IndexScalarQuantizer score
        # with asymmetric distance computation, comparing the float32 query
        # against the *decoded* stored codes. Verified to match an exact
        # float64 ||q - decode(c)|| to 2e-6 at dim=32 and 2e-5 at dim=384 --
        # float32 accumulation noise, well inside _COVERAGE_EPSILON.
        #
        # That is what makes the single-sided inflation in _store_spheres
        # both correct and tight:
        #     d(q, c) <= d(q, decode(c)) + ||c - decode(c)||
        # so d(q, c) <= tau implies d(q, decode(c)) <= tau + error_c, and
        # only the center side of the triangle inequality is lossy. Adding a
        # query-side term on top would be over-inflation, pure false
        # positives for no gain. (For reference, if the query *were* encoded,
        # SQ4 at dim=384 would need another ~2.7 on every radius.)
        Q = np.ascontiguousarray(vectors, dtype=np.float32)
        # The raw ID-mapped index, not the unwrapped inner one -- the latter
        # returns labels keyed by insertion position, which stops matching
        # sphere ID after a merge reassigns some, silently indexing
        # self._all_radiuses[labels] by the wrong sphere.
        d2, labels = self.all_centers_index.search(Q, self._num_spheres)
        surface = np.sqrt(np.maximum(d2, 0.0)) - self._all_radiuses[labels]
        return (surface.min(axis=1) <= distance_threshold + self._COVERAGE_EPSILON).tolist()

    def insert(self, vectors: np.ndarray) -> None:
        """Streaming insertion into an already-constructed filter. Nothing is
        retrained and nothing is rebuilt: whatever centers the strategy
        leaves behind are encoded once, with construct()'s codebook, and
        written straight into the index.

        insert_method strategies mutate _all_centers/_all_radiuses/
        _all_spheres in place, so the sync below is by sphere id: ids whose
        center the strategy actually changed (relocated, merged, or newly
        allocated) get re-encoded and re-inflated; ids it only grew the
        radius of are left alone, because a radius measured against an
        already-decoded center needs no inflation; ids it retired are dropped
        from the index."""
        if not self._constructed:
            raise ValueError(
                "BallCoverFilter.insert() requires a constructed filter -- call construct() first. "
                "insert() reuses the codebook and PQ nbits that construct() fixed, and has no "
                "centers to fit them from on its own."
            )

        if len(vectors) == 0:
            return

        prev_ids = self.all_spheres.copy()
        # Materializes the _all_centers cache the strategies write through.
        prev_centers = self.all_centers.copy()

        self.insert_method.insert(self, vectors)

        new_ids = self.all_spheres.copy()
        new_centers = self._all_centers[new_ids]

        # Position of each still-active id in the pre-insert snapshot, -1 for
        # ids the strategy newly allocated.
        prev_pos = np.full(len(self._all_radiuses), -1, dtype=np.int64)
        prev_pos[prev_ids] = np.arange(len(prev_ids), dtype=np.int64)
        pos = prev_pos[new_ids]

        unchanged = pos >= 0
        if unchanged.any():
            unchanged[unchanged] = np.all(
                new_centers[unchanged] == prev_centers[pos[unchanged]], axis=1
            )
        changed = ~unchanged

        alive = np.zeros(len(self._all_radiuses), dtype=bool)
        alive[new_ids] = True
        self._drop_spheres(prev_ids[~alive[prev_ids]])

        if changed.any():
            changed_ids = new_ids[changed]
            self._store_spheres(changed_ids, new_centers[changed], self._all_radiuses[changed_ids])

        # Strategies write raw float32 radii straight into _all_radiuses, so
        # re-settle every active ball onto the fp16 grid bytes_per_ball()
        # charges for. Idempotent, and only ever rounds up.
        self._all_radiuses[new_ids] = self._fp16_ceil(self._all_radiuses[new_ids])

        # Drop the cache so the next read comes back decoded, i.e. equal to
        # what query() actually scores against.
        self._all_centers = None

    def add(self, vectors: np.ndarray) -> None:
        self.insert(vectors)

    def remove(self, vectors: np.ndarray) -> List[bool]:
        return self.insert_method.remove(self, vectors)

    def _log_volume_bounds(self) -> Tuple[float, float]:
        if self._num_spheres == 0:
            return -np.inf, -np.inf
        current_centers = self.all_centers
        self._all_centers = None
        return estimate_hypersphere_volume(current_centers, self.all_radiuses, return_log=True)

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