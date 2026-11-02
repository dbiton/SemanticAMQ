import numpy as np
import faiss
import os
import sys
from typing import List, Optional, Sequence, Tuple, Union
from filters.filter import Filter, InsufficientMemoryError
from filters.cap_cover.cap_quantizers import Quantize, _ENCODINGS, _SeededPQ, _fit_scalar_codes, _PROBE_RNG  # noqa: F401
from filters.cap_cover.cap_scan import _CapScanMixin
from filters.cap_cover.cap_interfaces import ConstructInterface, Cover, EncodedCover, InsertInterface
from filters.cap_cover.cap_geometry import (angle_between, log_cap_measure, nearest_directions, segments,
                                            segment_max, unit_rows, row_angles, pair_angles, tail_rust, _rs)


_ROTATION_SEED = 0x5EED
_ROTATION_SEED_BYTES = 4
_ANGLE_GRID_BYTES = 8      # a static 8-bit filter's angle grid: two float32


def _fits(dim: int, q: Quantize) -> bool:
    """Whether encoding q can code directions of `dim` dimensions."""
    return dim % max(_ENCODINGS[q].sub, 1) == 0


# Candidate encodings CapCoverFilter.auto() chooses among: the seeded-PQ rungs
# below 1 bit/dim plus TurboQuant above it, so small budgets can buy many
# coarse caps and large ones fewer, finer caps.
AUTO_CANDIDATES = (Quantize.SPQ12, Quantize.SPQ16, Quantize.SPQ24, Quantize.SPQ36, Quantize.SPQ48,
                   Quantize.SPQ64, Quantize.TQ2, Quantize.TQ3, Quantize.TQ4)

# What scanning a cap costs, per stored byte, in nanoseconds: one query against
# the Rust scan, one core of an Apple M-series laptop at d = 384 (a Threadripper
# 3970X is 1.0-2x slower; CapCoverRustFilter/bench/ns_per_cap.py measures it).
# A scan costs caps x ns/cap and caps = budget / bytes per cap, so at a byte
# budget the cost is budget x this number. The TurboQuant rungs run through the
# SIMD table scan; the seeded-PQ ones through codebook tables.
SCAN_NS_PER_BYTE = {
    Quantize.TQ4: 0.09, Quantize.TQ2: 0.10, Quantize.TQ3: 0.32, Quantize.SPQ64: 0.39, Quantize.SPQ48: 0.53,
    Quantize.SPQ16: 0.58, Quantize.SPQ36: 0.63, Quantize.SPQ24: 0.94, Quantize.SPQ12: 1.89,
}


def _scan_limit_ms() -> Optional[float]:
    """auto()'s default scan_limit_ms: env CAPCOVER_SCAN_LIMIT_MS (read in
    worker processes too), else none."""
    value = os.environ.get("CAPCOVER_SCAN_LIMIT_MS")
    return float(value) if value else None


class CapCoverFilter(_CapScanMixin, Filter):
    """Ball-Cover for unit-norm vectors, with spherical caps in place of
    balls: each cap stores an encoded unit direction u and an angle theta,
    and a query is accepted iff it lies within distance tau of some unit
    vector of some cap. Zero false negatives by construction: every member
    lies in a cap, measured exactly around the direction as stored.

    Accept rule. For a unit query, angle(q, u) <= theta + alpha with
    alpha = 2 arcsin(tau / 2): the exact tau-neighbourhood of the cap on the
    sphere, so no rule that knows only the cap can be tighter. For a query
    of norm n the rule generalizes exactly: angle(q, u) - theta <= beta(n),
    cos beta = (n^2 + 1 - tau^2) / (2n), and nothing is accepted when
    |n - 1| > tau. Members must be unit vectors (within _NORM_TOLERANCE,
    which the rule adds to tau).

    Coarse directions. A cap needs only where its direction points: a
    code's norm error costs nothing and each cap's angle is measured around
    its decoded direction. A direction encoded delta off its cap's center
    holds the members within theta' ~ sqrt(theta^2 + delta^2), but the
    error is nearly orthogonal to the data, where queries are, so along the
    data the cap reaches only phi = sqrt((theta' + alpha)^2 - delta^2)
    ~ sqrt(theta^2 + 2 alpha theta' + alpha^2). That makes many coarse caps
    beat fewer precise ones at equal bytes, down to 12-byte directions
    (0.25 bit/dim, Quantize's seeded PQ rungs) at small budgets.

    Fitted codes. A cap's code need not be the one nearest its center, only
    one whose decoded direction holds its members tightly: construct()
    searches codes for the least members' angle (_SeededPQ.fit_codes,
    _fit_scalar_codes), free in bytes.

    Choosing the encoding. How coarse is best depends on the budget per
    member, so `quantize` may be a sequence of candidates (auto() uses
    AUTO_CANDIDATES), and construct() builds covers at the candidates' cap
    counts from one merge (ConstructInterface.iter_candidates) and keeps the
    one with the least accept region on the sphere (_volume): ln of the sum
    over caps of sigma(theta' + beta) at selection_tau (theta' the members'
    widest angle around the decoded direction) -- from B and the stored caps
    only, so it holds whatever the queries. With `mixed` (static filters)
    each cap takes its own encoding (_choose_mixed): caps narrower than the
    widest take coarser codes than the least-volume single encoding where
    that keeps them within the widest angle, buying more caps with the bytes
    freed. mixed=False keeps one encoding (_choose).

    Angles are stored in 8 bits over the filter's own [min, max] when no
    insert strategy can grow one past max (else 16 bits over [0, pi]).

    Working basis. Rotating encodings quantize in a seeded orthonormal
    basis; members and queries are rotated in float64 and every cap angle is
    measured in float64 in that basis around the decoded direction query()
    scores against. Strategies only ever see working-basis unit vectors;
    angles are rotation-invariant, so none of them needs to know.

    query() runs in Rust (capcover_rs, see cap_scan.py): per cap, the unit
    query's float32 score against the decoded direction (table lookups on the
    stored codes, nothing decoded persists between calls) is compared with
    |raw| cos(theta + beta); pairs within float32's error bound of that
    boundary are re-decided exactly in float64, so decisions match a
    float64 evaluation of the rule. Encodings Rust has no kernel for (FP32,
    FP16, SQ*) are answered by that float64 rule directly.

    Measurements behind each of these, and the variants that were tried and
    removed, are in CAP_COVER_EXPERIMENTS.md."""

    # Added to tau, as BallCoverFilter's own _COVERAGE_EPSILON: ground truth
    # is computed by faiss in float32 (|q - x|^2 <= tau^2), whose cancellation
    # error at d=384 can put a pair a few 1e-5 on the wrong side of tau.
    _COVERAGE_EPSILON = 1e-3
    # Largest | |x| - 1 | accepted for a member, also added to tau: a query
    # within tau of member x is within tau + | |x| - 1 | of x / |x|.
    _NORM_TOLERANCE = 1e-4

    # How many of a member's nearest caps _tight_angles tests when looking for
    # one that already covers it (BallCoverFilter._COVERAGE_NEIGHBORS).
    _COVERAGE_NEIGHBORS = 16

    def __init__(
        self,
        dim: int,
        construct_method: Optional[ConstructInterface] = None,
        insert_method: Optional[InsertInterface] = None,
        quantize: Union[Quantize, Sequence[Quantize]] = Quantize.FP32,
        max_memory: Optional[int] = None,
        max_caps: Optional[int] = None,
        insert_memory_percent: Optional[float] = 0.0,
        angle_bits: Optional[int] = None,
        fit_codes: bool = True,
        fit_candidates: int = 8,
        fit_passes: int = 2,
        selection_tau: float = 0.1,
        mixed: bool = True,
        rust_threads: Optional[int] = None,
    ) -> None:
        super().__init__(dim, max_memory=max_memory)
        # Threads of query()'s Rust scan: the evaluation runs one process per
        # core, so 1 (or env CAPCOVER_RUST_THREADS) is the default.
        self.rust_threads = int(os.environ.get("CAPCOVER_RUST_THREADS", 1)) if rust_threads is None else rust_threads
        self._rust = None            # capcover_rs.CapIndex while frozen (cap_scan.py)
        self._groups = []            # encodings in Rust's group order
        if mixed not in (False, True):
            raise ValueError(f"mixed must be True or False, got {mixed!r}")
        self.mixed = mixed
        if construct_method is None:
            from filters.cap_cover.sphere_merge_construct import SphereMergeConstruct
            construct_method = SphereMergeConstruct()
        levels = (quantize,) if isinstance(quantize, Quantize) else tuple(dict.fromkeys(quantize))
        if not levels:
            raise ValueError("quantize needs at least one encoding")
        self.levels = tuple(sorted(levels, key=lambda q: self._code_size_of(dim, q)))
        if len(self.levels) > 1 and max_caps is not None:
            raise ValueError("Choosing among encodings needs a byte budget (max_memory), not max_caps.")
        if angle_bits is None:
            # 8 bits on the static filter's own [min, max] costs <= 0.001 rad
            # per cap and buys 4-8% more caps (-0.005..-0.011 FPR at k <= 5%
            # of |B|); a filter that inserts needs all of [0, pi].
            angle_bits = 8 if insert_method is None else 16
        if angle_bits not in (8, 16):
            raise ValueError(f"angle_bits must be 8 or 16, got {angle_bits}")
        self.construct_method = construct_method
        self.insert_method = insert_method
        self.insert_memory_percent = insert_memory_percent
        self.angle_bits = angle_bits
        self.fit_codes = fit_codes
        self.fit_candidates = fit_candidates
        self.fit_passes = fit_passes
        self._angle_levels = (1 << angle_bits) - 1
        # Angle grid [lo, hi]: all of [0, pi] by default; a static 8-bit
        # filter narrows it to its own angles at construct() (see there).
        self._angle_lo, self._angle_hi = 0.0, float(np.pi)
        self.selection_tau = selection_tau
        self.selection_scores = None    # encoding -> score, set by construct() when choosing
        # Choosing compares encodings in one working basis, so every
        # candidate gets the rotation then (harmless where not needed).
        rotate = any(_ENCODINGS[q].rotate for q in self.levels) or len(self.levels) > 1
        self._rotation = self._make_rotation(dim) if rotate else None
        self._quantizers = [self._make_quantizer(q) for q in self.levels]
        self._code_sizes = np.array([self._code_size_of(dim, q) for q in self.levels], dtype=np.int64)
        self._max_caps_override = max_caps
        self._constructed = False
        self._num_caps = 0
        self._codes = np.zeros((0, int(self._code_sizes.max())), dtype=np.uint8)
        self._angle_codes = np.zeros(0, dtype=np.uint16)
        self._all_caps = np.zeros(0, dtype=np.int32)
        self._cap_level = np.zeros(0, dtype=np.int8)   # each cap's encoding (index into levels)
        self._dirs = None   # decoded unit directions by id: transient, like BallCoverFilter's all_centers
        self.selection_k = None
        self._set_uniform(0)

    def _set_uniform(self, level: int) -> None:
        """Every cap at one encoding: fixes the cap budget insert strategies
        and construct methods read (max_caps, max_construct_caps)."""
        self._uniform = level
        self.quantize = self.levels[level]
        self.max_caps = (self._max_caps_override if self._max_caps_override is not None
                         else self._calculate_max_caps(self.max_memory))
        self.max_construct_caps = self._calculate_max_construct_caps(self.insert_memory_percent)
        if self.max_caps < sys.maxsize:
            self._ensure_capacity(self.max_caps)
            if self.insert_method:
                self.insert_method.ensure_capacity(self.max_caps)

    @classmethod
    def auto(cls, dim: int, max_memory: int, **kwargs) -> "CapCoverFilter":
        """A filter that picks one encoding for the budget at construct(): the
        AUTO_CANDIDATES' covers are each built at that encoding's cap count and
        the one with the least accept-region volume wins (_choose). Every cap
        takes that encoding; pass mixed=True for per-cap encodings
        (_choose_mixed).

        scan_limit_ms (default: env CAPCOVER_SCAN_LIMIT_MS, else none) drops the
        candidates whose estimated query scan of a full budget, max_memory x
        SCAN_NS_PER_BYTE, exceeds it: a coarse code stores more caps per byte
        but can cost several times more to scan, which only matters where the
        budget is large. If none is left the cheapest candidate is kept."""
        kwargs.setdefault("mixed", False)
        candidates = [q for q in kwargs.pop("candidates", AUTO_CANDIDATES) if _fits(dim, q)]
        limit = kwargs.pop("scan_limit_ms", None)
        limit = _scan_limit_ms() if limit is None else limit
        if limit is not None:
            cheap = [q for q in candidates if max_memory * SCAN_NS_PER_BYTE[q] / 1e6 <= limit]
            candidates = cheap or [min(candidates, key=SCAN_NS_PER_BYTE.__getitem__)]
        return cls(dim, quantize=candidates, max_memory=max_memory, **kwargs)

    # ------------------------------------------------------------ properties
    @property
    def all_caps(self) -> np.ndarray:
        return self._all_caps[:self._num_caps]

    @property
    def all_directions(self) -> np.ndarray:
        """Decoded unit directions of the active caps, in the working basis,
        float64. Cached until the filter clears it."""
        if self._dirs is None:
            self._dirs = np.zeros((len(self._angle_codes), self.dim), dtype=np.float64)
            ids = self.all_caps
            if len(ids):
                self._dirs[ids] = self.decode_caps(ids)
        return self._dirs[self.all_caps]

    @property
    def all_angles(self) -> np.ndarray:
        return self._angle_value(self._angle_codes[self.all_caps])

    # --------------------------------------------------------------- budget
    @staticmethod
    def _code_size_of(dim: int, q: Quantize) -> int:
        spec = _ENCODINGS[q]
        if spec.sub:
            return (dim // spec.sub * spec.bits + 7) // 8
        if spec.qtype is None:
            return dim * 4
        return int(faiss.ScalarQuantizer(dim, spec.qtype).code_size)

    def code_size(self, level: Optional[int] = None) -> int:
        return int(self._code_sizes[self._uniform if level is None else level])

    def bytes_per_cap(self, level: Optional[int] = None) -> int:
        return self.code_size(level) + self.angle_bits // 8

    def _quantizer_bytes(self, level: int) -> int:
        sq = self._quantizers[level]
        if sq is None:
            return 0
        if isinstance(sq, _SeededPQ):
            return sq.codebook_bytes()
        trained = faiss.vector_to_array(sq.trained)
        if len(trained) == 0:
            probe = faiss.ScalarQuantizer(self.dim, _ENCODINGS[self.levels[level]].qtype)
            probe.train(_PROBE_RNG.random((2, self.dim)).astype(np.float32))
            trained = faiss.vector_to_array(probe.trained)
        return int(trained.nbytes)

    def _codebook_bytes(self, level: Optional[int] = None) -> int:
        """Fixed cost of the encoding in use (or `level`): the rotation's
        seed and the encoding's codebook."""
        level = self._uniform if level is None else level
        total = _ROTATION_SEED_BYTES if self._rotation is not None else 0
        if self._narrow_angles():
            total += _ANGLE_GRID_BYTES
        return total + self._quantizer_bytes(level)

    def _calculate_max_caps(self, max_memory: Optional[int]) -> int:
        if max_memory is None:
            return sys.maxsize
        usable = max(0, max_memory - self._codebook_bytes())
        if self.insert_method is None:
            return max(1, usable // self.bytes_per_cap())
        return max(1, self.insert_method.max_caps_for_budget(usable, self.bytes_per_cap()))

    def _calculate_max_construct_caps(self, insertion_memory_percent: float) -> int:
        if insertion_memory_percent is None or insertion_memory_percent <= 0.0:
            return self.max_caps
        if self.max_memory is not None:
            construct_memory = max(0, int(self.max_memory * (1.0 - insertion_memory_percent / 100.0)))
            return self._calculate_max_caps(construct_memory)
        return max(1, int(self.max_caps * (1.0 - insertion_memory_percent / 100.0)))

    # ------------------------------------------------------- basis, encoding
    @staticmethod
    def _make_rotation(dim: int) -> np.ndarray:
        """Seeded Haar-random orthonormal matrix, float64. Charged as its
        4-byte seed, as BallCoverFilter charges its faiss rotation."""
        rng = np.random.default_rng(_ROTATION_SEED)
        q, r = np.linalg.qr(rng.standard_normal((dim, dim)))
        return q * np.sign(np.diag(r))[None, :]

    def _make_quantizer(self, q: Quantize):
        spec = _ENCODINGS[q]
        if spec.sub:
            return _SeededPQ(self.dim, spec.sub, spec.bits)
        if spec.qtype is None:
            return None
        return faiss.ScalarQuantizer(self.dim, spec.qtype)

    def _train_quantizers(self, directions: np.ndarray, level: Optional[int] = None) -> None:
        """A fresh quantizer for the encoding in use (or `level`), trained on
        the cap directions."""
        level = self._uniform if level is None else level
        sq = self._quantizers[level] = self._make_quantizer(self.levels[level])
        if sq is None or len(directions) == 0:
            return
        sq.train(np.ascontiguousarray(directions, dtype=np.float32))

    def to_working(self, vectors: np.ndarray) -> np.ndarray:
        """Rows in the working basis, float64 (norms preserved)."""
        X = np.asarray(vectors, dtype=np.float64)
        return X @ self._rotation if self._rotation is not None else X.copy()

    def _prepare_members(self, vectors: np.ndarray) -> np.ndarray:
        """Members as unit rows in the working basis; raises if any is off
        the unit sphere by more than _NORM_TOLERANCE."""
        X = self.to_working(vectors)
        if len(X) == 0:
            return X.reshape(0, self.dim)
        norms = np.sqrt(np.einsum("ij,ij->i", X, X))
        if float(np.abs(norms - 1.0).max()) > self._NORM_TOLERANCE:
            raise ValueError(
                f"CapCoverFilter needs unit-norm vectors; got norms in "
                f"[{norms.min():.6g}, {norms.max():.6g}]. Normalize the data first.")
        return X / norms[:, None]

    def encode(self, directions: np.ndarray, level: Optional[int] = None, fast: bool = False) -> np.ndarray:
        """Codes (n x code_size(level)) for unit working-basis directions;
        `fast` (seeded PQ, for planning) searches codewords with
        scan_kernels.nearest_codeword."""
        level = self._uniform if level is None else level
        D = np.ascontiguousarray(directions, dtype=np.float32)
        sq = self._quantizers[level]
        if sq is None:
            return D.view(np.uint8).reshape(len(D), -1).copy()
        if len(D) == 0:
            return np.empty((0, self.code_size(level)), dtype=np.uint8)
        if fast and isinstance(sq, _SeededPQ):
            return np.ascontiguousarray(sq.compute_codes(D, fast=True), dtype=np.uint8)
        return np.ascontiguousarray(sq.compute_codes(D), dtype=np.uint8)

    def decode(self, codes: np.ndarray, level: Optional[int] = None) -> np.ndarray:
        """Unit directions (float64, working basis) that `level`'s codes
        decode to."""
        level = self._uniform if level is None else level
        codes = np.ascontiguousarray(codes[:, :self.code_size(level)], dtype=np.uint8)
        if len(codes) == 0:
            return np.empty((0, self.dim))
        sq = self._quantizers[level]
        if sq is None:
            return unit_rows(codes.view(np.float32).reshape(len(codes), self.dim))
        return unit_rows(sq.decode(codes))

    def _decode_raw(self, codes: np.ndarray, level: int) -> np.ndarray:
        """The float32 vectors `level`'s codes decode to, before
        normalization: decode() is exactly unit_rows() of these, so a scan
        can screen in float32 and re-decide the few boundary pairs from the
        same values."""
        codes = np.ascontiguousarray(codes[:, :self.code_size(level)], dtype=np.uint8)
        sq = self._quantizers[level]
        if sq is None:
            return codes.view(np.float32).reshape(len(codes), self.dim)
        return np.ascontiguousarray(sq.decode(codes), dtype=np.float32)

    def fit_cap_code(self, u: np.ndarray, parts: np.ndarray, part_angles: np.ndarray,
                     candidates: int = 8, passes: int = 2, max_moves: int = 64) -> np.ndarray:
        """A code (1 x code_size) for a cap on direction u that must hold the
        caps (parts[i], part_angles[i]): chosen to minimize the angle it
        would store, max_i angle(decode, parts[i]) + part_angles[i], rather
        than lying nearest u -- what CapMergeInsert stores for a merge.
        Seeded PQ: a coordinate search over each slice's `candidates`
        codewords nearest u's (_SeededPQ.fit_cap_code); TurboQuant: greedy
        one-level moves of single coordinates. Other encodings: u's own
        code. Never worse than u's own code, which is where both searches
        start."""
        u = np.asarray(u, dtype=np.float64)
        P = np.asarray(parts, dtype=np.float64)
        t = np.asarray(part_angles, dtype=np.float64)
        sq = self._quantizers[self._uniform]
        own = self.encode(u[None, :])
        if hasattr(sq, "fit_cap_code"):
            return sq.fit_cap_code(u, P, t, candidates, passes)

        def stored(cos):
            return (np.arccos(np.clip(cos, -1.0, 1.0)) + t[:, None]).max(axis=0)

        if self._is_turboquant():
            bits = int(round(_ENCODINGS[self.quantize].rate))
            levels = np.sort(faiss.vector_to_array(sq.trained)[:1 << bits].astype(np.float64))
            C = sq.decode(own).astype(np.float64)[0]
            idx = np.clip(np.searchsorted(levels, C - 1e-12), 0, len(levels) - 1)
            dots, sq_norm = P @ C, float(C @ C)
            current = float(stored((dots / np.sqrt(sq_norm))[:, None])[0])
            for _ in range(max_moves):
                best = (current, None, None)
                for step in (-1, 1):
                    j = np.clip(idx + step, 0, len(levels) - 1)
                    delta = levels[j] - C
                    trial_n = sq_norm + delta * (2.0 * C + delta)
                    score = stored((dots[:, None] + P * delta[None, :]) / np.sqrt(np.maximum(trial_n, 1e-300))[None, :])
                    score[delta == 0] = np.inf
                    i = int(np.argmin(score))
                    if score[i] < best[0]:
                        best = (float(score[i]), i, j[i])
                if best[1] is None:
                    break
                current, i, ji = best
                delta = levels[ji] - C[i]
                dots += P[:, i] * delta
                sq_norm += delta * (2.0 * C[i] + delta)
                C[i], idx[i] = levels[ji], ji
            return np.ascontiguousarray(sq.compute_codes(C[None, :].astype(np.float32)), dtype=np.uint8)
        return own

    def _decode_levels(self, codes: np.ndarray, levels: np.ndarray) -> np.ndarray:
        """decode() of each row at its own entry of `levels`."""
        levels = np.asarray(levels)
        if len(levels) and (levels == levels[0]).all():
            return self.decode(codes, int(levels[0]))
        out = np.empty((len(codes), self.dim))
        for level in np.unique(levels):
            rows = np.flatnonzero(levels == level)
            out[rows] = self.decode(codes[rows], int(level))
        return out

    def decode_caps(self, ids: np.ndarray) -> np.ndarray:
        """Decoded unit directions (float64, working basis) of caps `ids`,
        each at its own encoding."""
        return self._decode_levels(self._stored_codes(ids), self._cap_level[ids])

    def _stored_codes(self, ids: np.ndarray) -> np.ndarray:
        """Codes of caps `ids` (rows of the widest code size)."""
        self._thaw()
        return self._codes[np.asarray(ids, dtype=np.int64)]

    def decode_directions(self, directions: np.ndarray, level: Optional[int] = None) -> np.ndarray:
        """Where each unit direction would point once stored at `level`."""
        return self.decode(self.encode(directions, level), level)

    # ---------------------------------------------------------------- angles
    def _angle_code(self, theta: np.ndarray) -> np.ndarray:
        """Smallest grid angle >= theta, on a uniform grid of 2^angle_bits
        levels over [_angle_lo, _angle_hi]. Up, never to nearest: a stored
        angle below the measured one would drop a member. Over all of
        [0, pi], 8 bits (step 0.012 rad) measured +0.01 FPR against 16 at
        TQ1; a static filter's own [min, max] makes 8 bits ~as fine as 16
        (see construct())."""
        lo, hi, levels = self._angle_lo, self._angle_hi, self._angle_levels
        step = max(hi - lo, 0.0) / levels
        theta = np.clip(np.asarray(theta, dtype=np.float64), 0.0, np.pi)
        if step == 0.0:
            return np.zeros(theta.shape, dtype=np.uint16)
        code = np.ceil((theta - lo) / step).astype(np.int64)
        code = np.maximum(code, 0)
        code = np.where(self._angle_value(code) < theta, code + 1, code)
        return np.minimum(code, levels).astype(np.uint16)

    def _angle_value(self, code: np.ndarray) -> np.ndarray:
        lo, hi = self._angle_lo, self._angle_hi
        return lo + np.asarray(code, dtype=np.float64) * (max(hi - lo, 0.0) / self._angle_levels)

    def _fit_angle_grid(self, angles: np.ndarray) -> None:
        """Narrows the angle grid to [min, max] of `angles`, each stored as a
        float32 rounded outward (charged in _codebook_bytes). Only for a
        static filter: an insert could need an angle past max."""
        lo = np.float32(np.min(angles))
        hi = np.float32(np.max(angles))
        lo = float(np.nextafter(lo, np.float32(-np.inf))) if lo > 0 else 0.0
        hi = float(np.nextafter(hi, np.float32(np.inf)))
        self._angle_lo, self._angle_hi = max(lo, 0.0), min(hi, float(np.pi))
        if self._angle_hi < float(np.max(angles)):
            self._angle_hi = float(np.pi)

    def _is_turboquant(self, level: Optional[int] = None) -> bool:
        q = self.quantize if level is None else self.levels[level]
        return q in (Quantize.TQ1, Quantize.TQ2, Quantize.TQ3, Quantize.TQ4, Quantize.TQ8)

    @property
    def encoding_label(self) -> str:
        """The encodings in use, finest first ("SPQ64+SPQ36" for a filter
        whose caps mix them)."""
        used = np.unique(self._cap_level[self.all_caps]) if self._num_caps else [self._uniform]
        return "+".join(self.levels[int(l)].value for l in sorted(used, key=lambda l: -self._code_sizes[l]))

    def _narrow_angles(self) -> bool:
        return self.angle_bits == 8 and self.insert_method is None

    # ----------------------------------------------------------- cap storage
    def _ensure_capacity(self, required_cap: int) -> None:
        current = len(self._angle_codes)
        if required_cap <= current:
            return
        new_cap = max(current * 2, required_cap)
        ids = np.empty(new_cap, dtype=np.int32)
        ids[:current] = self._all_caps
        ids[current:] = np.arange(current, new_cap, dtype=np.int32)
        self._all_caps = ids
        codes = np.zeros((new_cap, int(self._code_sizes.max())), dtype=np.uint8)
        codes[:current] = self._codes
        self._codes = codes
        angles = np.zeros(new_cap, dtype=np.uint16)
        angles[:current] = self._angle_codes
        self._angle_codes = angles
        levels = np.full(new_cap, self._uniform, dtype=np.int8)
        levels[:current] = self._cap_level
        self._cap_level = levels
        if self._dirs is not None:
            dirs = np.zeros((new_cap, self.dim), dtype=np.float64)
            dirs[:current] = self._dirs
            self._dirs = dirs
        if self.insert_method:
            self.insert_method.ensure_capacity(new_cap)

    def _reset_buffers(self, capacity: int) -> None:
        self._ensure_capacity(capacity)
        self._num_caps = 0
        self._all_caps[:] = np.arange(len(self._all_caps), dtype=np.int32)
        self._angle_codes[:] = 0
        self._cap_level[:] = self._uniform
        self._dirs = None
        self._constructed = False
        if self.insert_method:
            self.insert_method.reset_buffers(capacity)

    def _add_caps(self, directions: np.ndarray, angles: np.ndarray, codes: Optional[np.ndarray] = None,
                  levels: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Stores caps (u, theta), each widened by its direction's encoding
        error so it still holds every unit vector the unencoded cap held
        (triangle inequality on the sphere). Returns (ids, decoded unit
        directions). `codes`, if given, are stored instead of U's own
        (fitted codes, see construct()), each at its entry of `levels`
        (default: the encoding in use)."""
        U = unit_rows(directions)
        n = len(U)
        if n == 0:
            return np.empty(0, dtype=np.int32), np.empty((0, self.dim))
        self._ensure_capacity(self._num_caps + n)
        ids = self._all_caps[self._num_caps:self._num_caps + n].copy()
        if levels is None:
            levels = np.full(n, self._uniform, dtype=np.int64)
            if codes is None:
                codes = self.encode(U)
        self._codes[ids, :codes.shape[1]] = codes
        self._cap_level[ids] = levels
        decoded = self._decode_levels(codes, levels)
        self._angle_codes[ids] = self._angle_code(np.asarray(angles, dtype=np.float64)
                                                  + angle_between(U, decoded))
        self._num_caps += n
        if self._dirs is not None:
            self._dirs[ids] = decoded
        return ids, decoded

    def _remove_caps(self, cap_ids: np.ndarray) -> None:
        if len(cap_ids) == 0:
            return
        removed = np.isin(self.all_caps, cap_ids)
        active = self.all_caps[~removed]
        gone = self.all_caps[removed]
        self._all_caps[:len(active)] = active
        self._all_caps[len(active):self._num_caps] = gone
        self._angle_codes[gone] = 0
        self._num_caps = len(active)
        if self._dirs is not None:
            self._dirs[gone] = 0.0

    def _set_angles(self, cap_ids: np.ndarray, angles: np.ndarray) -> None:
        """Sets measured angles, rounded up onto the angle grid."""
        self._angle_codes[cap_ids] = self._angle_code(angles)

    def _grow_angles(self, cap_ids: np.ndarray, angles: np.ndarray) -> None:
        """Raises each cap's stored angle to at least `angles`."""
        cur = self._angle_value(self._angle_codes[cap_ids])
        self._angle_codes[cap_ids] = self._angle_code(np.maximum(cur, angles))

    # ------------------------------------------------------------- construct
    def _tight_angles(self, X: np.ndarray, U: np.ndarray, decoded: np.ndarray,
                      baseline: np.ndarray, home: np.ndarray,
                      neighbors: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """The smallest angles around the *decoded* directions that still
        cover every member: BallCoverFilter._tight_radii on the sphere.

        A member already held by some cap does not grow its own; members are
        settled farthest-from-their-nearest-cap first, so the caps a member
        might hide in grow before it is tested. No cap is pushed past its
        bound baseline + angle(u, decode(u)), the angle _add_caps would have
        stored; each member's construction cap home[i] (whose cap (U, baseline)
        holds it) is its always-legal fallback, by the triangle inequality.
        A member looks among its _COVERAGE_NEIGHBORS nearest decoded
        directions, or, given the cover's `neighbors`, among its home cap and
        that cap's neighbours.

        Returns (angles, owner): owner[i] is a cap holding member i."""
        n_caps = len(decoded)
        angles = np.zeros(n_caps)
        owner = np.zeros(len(X), dtype=np.int64)
        if n_caps == 0 or len(X) == 0:
            return angles, owner
        bound = np.asarray(baseline, dtype=np.float64) + angle_between(U, decoded)
        home = np.asarray(home, dtype=np.int64)

        if neighbors is None:
            k = min(self._COVERAGE_NEIGHBORS, n_caps)
            _, labels = nearest_directions(X, decoded, k)
            labels = labels.reshape(len(X), k)
        else:
            labels = np.concatenate([home[:, None], neighbors[home]], axis=1)
        dist = pair_angles(X, decoded, labels)

        home_d = row_angles(X, decoded, home)
        # A construct method whose cap does not hold its own nearest member
        # is repaired here rather than trusted: the fallback must be legal.
        np.maximum.at(bound, home, home_d)

        if tail_rust():
            # Same settle loop in Rust (construct_tail.rs); CAPCOVER_TAIL=python keeps the loop below.
            return _rs.settle(np.ascontiguousarray(labels, dtype=np.int64), np.ascontiguousarray(dist),
                              home, np.ascontiguousarray(home_d), bound)
        nearest_d = dist.min(axis=1)
        for i in np.argsort(-nearest_d, kind="stable"):
            row_l, row_d = labels[i], dist[i]
            held = np.flatnonzero(row_d <= angles[row_l])
            if len(held):
                owner[i] = row_l[held[0]]
                continue
            if home_d[i] <= angles[home[i]]:
                owner[i] = home[i]
                continue
            for j in np.argsort(row_d, kind="stable"):
                if row_d[j] <= bound[row_l[j]]:
                    angles[row_l[j]] = row_d[j]
                    owner[i] = row_l[j]
                    break
            else:
                angles[home[i]] = home_d[i]
                owner[i] = home[i]
        return angles, owner

    def _volume(self, X: np.ndarray, U: np.ndarray, labels: np.ndarray) -> float:
        """ln of sum over caps of sigma(theta' + beta): the share of the unit
        sphere the cover on directions U, members X[labels == j], would
        accept at selection_tau once encoded (an upper bound on its union's
        measure), theta' the members' widest angle around the decoded
        direction and beta the accept rule's own expansion at |q| = 1.
        Trains the quantizer on U."""
        self._train_quantizers(U)
        order, starts, counts = segments(labels, len(U))
        decoded = self.decode_directions(U)
        theta = segment_max(angle_between(X, decoded[labels])[order], starts, counts)
        beta, _ = self._beta(np.ones(1), self.selection_tau)
        psi = np.minimum(theta[counts > 0] + beta[0], np.pi)
        return float(np.logaddexp.reduce(log_cap_measure(psi, self.dim))) if len(psi) else -np.inf

    def _choose(self, X: np.ndarray, weights: np.ndarray) -> Cover:
        """Scores a cover per candidate encoding at that encoding's cap count
        (one merge, see ConstructInterface.iter_candidates) by its accept
        region's volume (_volume) and keeps the best, fixes the filter's encoding to it, and returns its finished
        cover. Candidates come largest cap count first, and the scan stops
        once the score has risen twice in a row: it is unimodal along the
        ladder (more, coarser caps against fewer, finer ones).

        Levels whose budget buys a cap per distinct member or more all get
        that same cover, on which a finer code always accepts less: only
        the finest of them is scored."""
        ks = []
        for level in range(len(self.levels)):
            self._set_uniform(level)
            ks.append(min(self.max_construct_caps, len(X)))
        levels_at = {}
        for level, k in enumerate(ks):
            levels_at.setdefault(k, []).append(level)
        if len(X) in levels_at:
            levels_at[len(X)] = [max(levels_at[len(X)], key=lambda lv: self._code_sizes[lv])]
        scores, best, worse = {}, None, 0
        for k, cover in self.construct_method.iter_candidates(self, X, ks, weights):
            cover = cover._replace(directions=unit_rows(cover.directions))
            improved = False
            for level in levels_at[k]:
                self._set_uniform(level)
                score = self._volume(X, cover.directions, cover.labels)
                scores[self.levels[level].value] = score
                if best is None or score < best[0]:
                    best, improved = (score, level, cover), True
            worse = 0 if improved else worse + 1
            if worse >= 2:
                break
        self.selection_scores = scores
        _, level, cover = best
        self._set_uniform(level)
        return self.construct_method.finish(self, X, cover)

    # Mixed encodings (_choose_mixed): cap counts tried step down by this
    # factor; the scan stops after this many counts without a better cover.
    _MIX_STEP = 1.1
    _MIX_PATIENCE = 4

    def _mixing(self) -> bool:
        return (bool(self.mixed) and self.insert_method is None
                and len(self.levels) > 1 and self.max_memory is not None)

    def _construct_budget(self) -> int:
        if self.insert_memory_percent:
            return int(self.max_memory * (1.0 - self.insert_memory_percent / 100.0))
        return int(self.max_memory)

    def _quick_level(self, level: int) -> bool:
        """Whether `level` encodes fast: TurboQuant / scalar / raw floats,
        or seeded PQ with 8-bit codebooks."""
        spec = _ENCODINGS[self.levels[level]]
        return spec.qtype is not None or not spec.sub or spec.bits <= 8

    def _data_free(self, level: int) -> bool:
        """Whether `level`'s quantizer is the same whatever it is trained on
        (seeded PQ, TurboQuant, raw floats)."""
        spec = _ENCODINGS[self.levels[level]]
        return spec.qtype is None or spec.rotate

    def _level_psi(self, X: np.ndarray, U: np.ndarray, labels: np.ndarray,
                   cache: Optional[dict] = None, levels: Optional[List[int]] = None,
                   psi: Optional[np.ndarray] = None) -> np.ndarray:
        """(caps x levels): each cap's accept angle at selection_tau were its
        direction stored at each candidate encoding -- the members' widest
        angle around the nearest-codeword decoded direction (its own encoding
        error for a cap with no member), plus beta. `cache` (level -> {direction bytes: decoded
        direction}) carries data-free levels' decodings across the covers of
        one merge, where most caps are unchanged from one cover to the next:
        a merge only moves the caps it merges, so a cap whose direction is
        unchanged holds the same members and keeps its angles. Only
        `levels` (default all) are filled in, into `psi` if given; the rest
        stay inf."""
        order, starts, counts = segments(labels, len(U))
        beta = float(self._beta(np.ones(1), self.selection_tau)[0][0])
        if psi is None:
            psi = np.full((len(U), len(self.levels)), np.inf)
        keys = [u.tobytes() for u in U] if cache is not None else None
        for level in (range(len(self.levels)) if levels is None else levels):
            if cache is None or not self._data_free(level):
                self._train_quantizers(U, level=level)
                D = self.decode(self.encode(U, level, fast=True), level)
                theta = segment_max(angle_between(X, D[labels])[order], starts, counts)
                theta = np.where(counts > 0, theta, angle_between(U, D))
                psi[:, level] = np.minimum(theta + beta, np.pi)
                continue
            known = cache.setdefault(level, {})
            if not known:
                self._train_quantizers(U, level=level)
            new = np.array([i for i, key in enumerate(keys) if key not in known], dtype=np.int64)
            if len(new):
                D = self.decode(self.encode(U[new], level, fast=True), level)
                local = np.full(len(U), -1, dtype=np.int64)
                local[new] = np.arange(len(new))
                rows = np.flatnonzero(local[labels] >= 0)
                lab = local[labels[rows]]
                o, st, ct = segments(lab, len(new))
                theta = segment_max(angle_between(X[rows], D[lab])[o], st, ct)
                theta = np.where(ct > 0, theta, angle_between(U[new], D))
                for i, t in zip(new, np.minimum(theta + beta, np.pi)):
                    known[keys[i]] = t
            psi[:, level] = [known[key] for key in keys]
        return psi

    def _plan(self, psi: np.ndarray, allowed: List[int], budget: int):
        """The least widest accept angle T a cover's caps can share within
        `budget` bytes, each cap at the cheapest of the `allowed` levels
        that keeps it within T (ties: the narrower). Returns (ln of the
        summed cap measures, each cap's level, bytes, T) or None. Bisects over
        the candidate angles: bytes only fall as T grows, up to the
        codebooks of the levels in use."""
        allowed = np.asarray(allowed, dtype=np.int64)
        P = psi[:, allowed]
        cost = (self._code_sizes[allowed] + self.angle_bits // 8).astype(np.float64)
        fixed = np.array([self._quantizer_bytes(int(l)) for l in allowed], dtype=np.float64)
        base = (_ROTATION_SEED_BYTES if self._rotation is not None else 0) + \
            (_ANGLE_GRID_BYTES if self._narrow_angles() else 0)
        rows = np.arange(len(P))

        def at(T):
            ok = (P <= T) & np.isfinite(P)
            if not ok.any(axis=1).all():
                return None
            choice = np.argmin(np.where(ok, cost[None, :] + 1e-3 * P, np.inf), axis=1)
            counts = np.bincount(choice, minlength=len(allowed))
            used = np.flatnonzero(counts)
            total = (base + sum(int(counts[i]) * int(cost[i]) for i in used)
                     + fixed[used].sum() + self._GROUP_BYTES * (len(used) - 1))
            return total, choice

        T = np.unique(P[np.isfinite(P)])
        if len(T) == 0:
            return None
        top = at(T[-1])
        if top is None or top[0] > budget:
            return None
        lo, hi = 0, len(T) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            r = at(T[mid])
            if r is not None and r[0] <= budget:
                hi = mid
            else:
                lo = mid + 1
        total, choice = at(T[lo])
        score = float(np.logaddexp.reduce(log_cap_measure(P[rows, choice], self.dim)))
        return score, allowed[choice], int(total), float(T[lo])

    def _choose_mixed(self, X: np.ndarray, weights: np.ndarray) -> Tuple[Cover, np.ndarray]:
        """_choose with each cap at its own encoding. At d = 384 a cover's
        accept region is set by its widest caps, so a cap narrower than the
        widest can take a coarser code without widening the region, and the
        bytes it frees buy more caps. Covers at cap counts stepping down by
        _MIX_STEP from the coarsest encoding's (one merge) are each planned
        by _plan -- the cheap encodings together, plus at most one with a
        costly codebook (a trained basis) -- and the least summed measure
        wins, and the finished cover is planned again. No cap takes a code
        finer than the least-volume single encoding's, so caps only coarsen
        and the cover keeps at least that encoding's cap count. Returns
        (cover, each cap's level); the filter's own encoding (whose codebook
        _codebook_bytes charges) becomes the finest one in use."""
        grid_levels = list(range(len(self.levels)))
        ks = []
        for level in grid_levels:
            self._set_uniform(level)
            ks.append(max(1, min(self.max_construct_caps, len(X))))
        budget = self._construct_budget()
        grid, k = set(ks), max(ks)
        while k > min(ks):
            grid.add(k)
            k = int(k / self._MIX_STEP)
        # Scoring a level costs an encoding of every new cap: cheap for
        # TurboQuant and 8-bit seeded PQ, not for 12-bit codebooks. Those
        # are scored only when their encoding error on a sample of the
        # members (its 1st percentile, plus beta) is under the widest angle
        # the cheap levels already reach: a cap at such a level could not
        # narrow the cover's widest cap.
        beta = float(self._beta(np.ones(1), self.selection_tau)[0][0])
        quick = [l for l in grid_levels if self._quick_level(l)]
        slow = [l for l in grid_levels if l not in quick]
        sample = X[np.random.default_rng(0x5EED).choice(len(X), min(len(X), 512), replace=False)]
        floor = {}
        for l in slow:
            self._train_quantizers(sample, level=l)
            D = self.decode(self.encode(sample, l, fast=True), l)
            floor[l] = float(np.quantile(angle_between(sample, D), 0.01)) + beta
        # The best plan under each ceiling on code size (None: no ceiling),
        # and each level's own best uniform score.
        ceilings = [None] + sorted(set(int(c) for c in self._code_sizes))
        best, uniform, worse, scores, cache = {}, {}, 0, {}, {}
        for k, cover in self.construct_method.iter_candidates(self, X, sorted(grid, reverse=True), weights):
            U = unit_rows(cover.directions)
            psi = self._level_psi(X, U, cover.labels, cache, levels=quick)
            here = self._plan(psi, quick, budget)
            reach = max(([here[3]] if here is not None else []) + [b[4] for b in best.values()] + [-np.inf])
            reach = np.inf if reach == -np.inf else reach
            psi = self._level_psi(X, U, cover.labels, cache, levels=[l for l in slow if floor[l] < reach], psi=psi)
            improved = False
            for c in ceilings:
                capped = [l for l in grid_levels if c is None or self._code_sizes[l] <= c]
                plan = self._plan(psi, capped, budget) if capped else None
                if plan is None:
                    continue
                if c is None:
                    scores[k] = min(scores.get(k, np.inf), plan[0])
                if c not in best or plan[0] < best[c][0]:
                    best[c] = (plan[0], cover._replace(directions=U), capped, plan[1], plan[3])
                    improved = True
            for l in grid_levels:
                plan = self._plan(psi, [l], budget) if np.isfinite(psi[:, l]).all() else None
                if plan is not None and plan[0] < uniform.get(l, np.inf):
                    uniform[l] = plan[0]
            worse = 0 if improved else worse + 1
            if best and worse >= self._MIX_PATIENCE:
                break
        self.selection_scores = scores
        key = None
        if uniform:
            # No code finer than the best single encoding's: caps only
            # coarsen where that keeps them within the widest angle, and the
            # bytes freed buy more caps.
            key = int(self._code_sizes[min(uniform, key=uniform.get)])
            key = key if key in best else None
        _, cover, allowed, levels, reach = best[key]
        cover = self.construct_method.finish(self, X, cover)
        # finish() re-centers caps and moves members between them, so a cap
        # planned coarse can become the widest: plan the finished cover
        # again (over the levels that could matter), falling back on the
        # carried plan (fits: finish() never adds caps).
        U = unit_rows(cover.directions)
        use = [l for l in allowed if l in quick or floor[l] < reach + 0.05]
        psi = self._level_psi(X, U, cover.labels, levels=use)
        plan = self._plan(psi, use, budget)
        if plan is not None:
            levels = plan[1]
        elif cover.kept is not None:
            levels = levels[cover.kept]
        used = np.unique(levels)
        self._set_uniform(int(used[np.argmax(self._code_sizes[used])]))
        self.selection_k = len(U)
        return cover, levels

    @staticmethod
    def _distinct_rows(vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(rows, inverse, counts): the distinct rows of `vectors` in order of
        first appearance, vectors == rows[inverse], counts[j] copies of row
        j. A cover is a function of the distinct members; only a construct
        method that weighs members by their copies (SphereMergeConstruct's
        mass) needs the counts."""
        V = np.ascontiguousarray(vectors, dtype=np.float32)
        keys = V.view(np.dtype((np.void, V.dtype.itemsize * V.shape[1]))).ravel()
        _, first, inverse, counts = np.unique(keys, return_index=True, return_inverse=True,
                                              return_counts=True)
        order = np.argsort(first, kind="stable")
        rank = np.empty(len(order), dtype=np.int64)
        rank[order] = np.arange(len(order))
        return V[first[order]], rank[inverse.ravel()], counts[order]

    def _fit_codes_by_level(self, U: np.ndarray, X: np.ndarray, owner: np.ndarray,
                            levels: np.ndarray) -> np.ndarray:
        """Codes for caps U, each at its entry of `levels`: fitted to the
        members `owner` puts in it where the encoding can be (fit_codes),
        else nearest U's direction."""
        codes = np.zeros((len(U), int(self._code_sizes.max())), dtype=np.uint8)
        for level in np.unique(levels):
            level = int(level)
            caps = np.flatnonzero(levels == level)
            local = np.full(len(U), -1, dtype=np.int64)
            local[caps] = np.arange(len(caps))
            rows = np.flatnonzero(local[owner] >= 0)
            sq = self._quantizers[level]
            if self.fit_codes and hasattr(sq, "fit_codes"):
                c = sq.fit_codes(U[caps], X[rows], local[owner[rows]], self.fit_candidates, self.fit_passes)
            elif self.fit_codes and self._is_turboquant(level):
                c = _fit_scalar_codes(sq, U[caps], X[rows], local[owner[rows]], self.fit_passes)
            else:
                c = self.encode(U[caps], level)
            codes[caps, :c.shape[1]] = c
        return codes

    def construct(self, vectors: np.ndarray) -> None:
        self._invalidate()
        if len(vectors) == 0:
            self._reset_buffers(0)
            self._constructed = True
            return
        distinct, inverse, copies = self._distinct_rows(vectors)
        X = self._prepare_members(distinct)
        levels = None
        if len(self.levels) > 1:
            if self._mixing():
                cover, levels = self._choose_mixed(X, copies)
            else:
                cover = self._choose(X, copies)
        else:
            for _, cover in self.construct_method.iter_candidates(self, X, [self.max_construct_caps], copies):
                cover = self.construct_method.finish(self, X, cover)
                break
        U = unit_rows(cover.directions)
        theta = np.asarray(cover.angles, dtype=np.float64)
        home, neighbors = np.asarray(cover.labels, dtype=np.int64), cover.neighbors
        if levels is None:
            levels = np.full(len(U), self._uniform, dtype=np.int64)
        self._reset_buffers(len(U))
        for level in np.unique(levels):
            self._train_quantizers(U, level=int(level))
        codebook = self._codebook_bytes()
        if self.max_memory is not None and codebook + self.bytes_per_cap() > self.max_memory:
            raise InsufficientMemoryError(
                f"max_memory={self.max_memory} bytes cannot hold one {self.quantize.value} cap "
                f"plus its {codebook}-byte codebook.")
        # Fit codes to the members each cap holds, then settle the members
        # around the fitted directions.
        encoded = self._encode_cover(X, U, theta, home, levels, neighbors)
        encoded = self.construct_method.refine_encoded(self, X, encoded, levels)
        U, codes, angles, _ = encoded
        ids, decoded = self._add_caps(U, theta, codes=codes, levels=levels)
        self._angle_lo, self._angle_hi = 0.0, float(np.pi)
        if self._narrow_angles() and len(angles):
            self._fit_angle_grid(angles)
        self._set_angles(ids, angles)
        # Diagnostic only (not part of the stored filter): each cap's angle
        # between its intended and its decoded direction.
        self.encoding_error = angle_between(U, decoded)
        if self.insert_method:
            self.insert_method.initialize_buffers(self, ids, self.all_directions, self.all_angles,
                                                  vectors=X[inverse])
        self._constructed = True
        self._dirs = None

    def _encode_cover(self, X, U, theta, home, levels, neighbors=None) -> EncodedCover:
        """Fit codes and reassign members using the already trained codebooks."""
        codes = self._fit_codes_by_level(U, X, home, levels)
        angles, owner = self._tight_angles(
            X, U, self._decode_levels(codes, levels), theta, home, neighbors)
        return EncodedCover(U, codes, angles, owner)

    def _encoded_volume(self, angles: np.ndarray) -> float:
        """Unexpanded volume using the exact angle rounding used by storage.

        Candidate evaluation must not change the eventual filter's angle grid.
        """
        if not len(angles):
            return -np.inf
        saved = self._angle_lo, self._angle_hi
        try:
            self._angle_lo, self._angle_hi = 0.0, float(np.pi)
            if self._narrow_angles():
                self._fit_angle_grid(angles)
            stored = self._angle_value(self._angle_code(angles))
            return float(np.logaddexp.reduce(log_cap_measure(stored, self.dim)))
        finally:
            self._angle_lo, self._angle_hi = saved

    # ----------------------------------------------------------------- query
    def query(self, vectors: np.ndarray, distance_threshold: float) -> List[bool]:
        if not self._constructed:
            raise ValueError("CapCoverFilter has not been constructed.")
        if self._num_caps == 0 or len(vectors) == 0:
            return [False] * len(vectors)
        return self._scan(vectors, float(distance_threshold))

    def surface_angle(self, vectors: np.ndarray) -> np.ndarray:
        """Per query, min over caps of angle(q, u) - theta, in float64 (an
        O(n * caps * d) diagnostic, not the query path): a unit query is
        accepted at tau iff this is <= 2 arcsin(tau' / 2)."""
        Qn = unit_rows(self.to_working(vectors))
        ids = self.all_caps
        if len(ids) == 0:
            return np.full(len(Qn), np.inf)
        C = self.decode_caps(ids)
        theta = self._angle_value(self._angle_codes[ids])
        out = np.full(len(Qn), np.inf)
        for s in range(0, len(Qn), 256):
            approx = np.arccos(np.clip(Qn[s:s + 256] @ C.T, -1.0, 1.0)) - theta[None, :]
            best = approx.min(axis=1)
            # arccos loses precision near 0: re-measure the leading candidates.
            i, j = np.nonzero(approx <= best[:, None] + 1e-3)
            res = np.full(len(approx), np.inf)
            np.minimum.at(res, i, angle_between(Qn[s + i], C[j]) - theta[j])
            out[s:s + 256] = res
        return out

    # ---------------------------------------------------------------- insert
    def insert(self, vectors: np.ndarray) -> None:
        if not self._constructed:
            raise ValueError("CapCoverFilter.insert() requires a constructed filter -- call construct() first.")
        if not self.insert_method:
            raise ValueError("CapCoverFilter.insert() requires an insertion strategy: insert_method is None.")
        if len(vectors) == 0:
            return
        self._invalidate()
        if self._num_caps == 0:
            raise ValueError("CapCoverFilter.insert() needs a non-empty construct() first (no codebook).")
        X = self._prepare_members(vectors)
        _ = self.all_directions
        self.insert_method.insert(self, X)
        self._dirs = None

    def add(self, vectors: np.ndarray) -> None:
        self.insert(vectors)

    def remove(self, vectors: np.ndarray) -> List[bool]:
        if not self.insert_method:
            raise ValueError("CapCoverFilter.remove() requires an insertion strategy.")
        self._invalidate()
        return self.insert_method.remove(self, self._prepare_members(vectors))

    # A static filter with caps at several encodings stores them grouped by
    # encoding, each group after the first behind its 4-byte count.
    _GROUP_BYTES = 4

    def get_bytesize(self) -> int:
        if not self._constructed:
            return 0
        levels = self._cap_level[self.all_caps]
        total = self._codebook_bytes()
        for level, n in zip(*np.unique(levels, return_counts=True)):
            total += int(n) * self.bytes_per_cap(int(level))
            if level != self._uniform:
                total += self._quantizer_bytes(int(level)) + self._GROUP_BYTES
        if self.insert_method:
            total += self.insert_method.get_bytesize(self)
        return total
