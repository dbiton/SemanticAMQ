import numpy as np
import faiss
import sys
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence, Tuple, Union
from filters.filter import Filter, InsufficientMemoryError
from filters.cap_cover.cap_interfaces import ConstructInterface, Cover, InsertInterface
from filters.cap_cover.cap_geometry import (angle_between, nearest_directions, segments, segment_max,
                                            unit_rows)


class Quantize(Enum):
    """Direction encodings: BallCoverFilter's ladder minus PQ and RaBitQ.

      b/dim  member   encoding                     code B   codebook
        32   FP32     raw float32                    1536   0
        16   FP16     scalar fp16                     768   0
         8   SQ8      scalar 8-bit                    384   3 KB
         8   TQ8      rotation + TurboQuant 8-bit     384   seed only
         6   SQ6      scalar 6-bit                    288   3 KB
         4   SQ4      scalar 4-bit                    192   3 KB
         4   TQ4      rotation + TurboQuant 4-bit     192   seed only
         3   TQ3      rotation + TurboQuant 3-bit     144   60 B + seed
         2   TQ2      rotation + TurboQuant 2-bit      96   seed only
         1   TQ1      rotation + TurboQuant 1-bit      48   seed only

    A cap needs only a direction, so a code's norm error is free: the decoded
    vector is renormalized and the cap's angle measured around where it
    actually points. Angular error of a unit vector at d=384, measured:
    TQ8 0.008, TQ4 0.097, TQ3 0.186, TQ2 0.35, TQ1 ~0.64 rad. PQ is left out
    because its codebooks (24-384 KB) dwarf the caps at this filter's sizes;
    RaBitQ because its estimator is not tied to a decoded vector, which the
    angle measurement needs.

    Every cap also stores its angle in `angle_bits` (see _angle_code): 16
    bits over [0, pi] when an insert strategy may grow it, else 8 bits over
    the filter's own range of angles."""
    FP32 = "FP32"
    FP16 = "FP16"
    SQ8 = "SQ8"
    TQ8 = "TQ8"
    SQ6 = "SQ6"
    SQ4 = "SQ4"
    TQ4 = "TQ4"
    TQ3 = "TQ3"
    TQ2 = "TQ2"
    TQ1 = "TQ1"
    # Seeded product quantizers, named by code bytes at d=384 (see
    # _SeededPQ): sub-vectors of `sub` dims, each the nearest of 2^bits
    # Gaussian codewords drawn from a fixed seed, so the codebook costs
    # nothing to store. Below 1 bit/dim, where TurboQuant has no rung.
    SPQ6 = "SPQ6"       # sub 96, 12 bits: ~1.2 rad (not in AUTO_CANDIDATES)
    SPQ9 = "SPQ9"       # sub 64, 12 bits: ~1.12 rad (not in AUTO_CANDIDATES)
    SPQ12 = "SPQ12"     # sub 48, 12 bits: ~1.06 rad
    SPQ16 = "SPQ16"     # sub 24,  8 bits: ~1.00 rad
    SPQ24 = "SPQ24"     # sub 24, 12 bits: ~0.85 rad
    SPQ36 = "SPQ36"     # sub 16, 12 bits: ~0.71 rad
    SPQ48 = "SPQ48"     # sub 12, 12 bits: ~0.60 rad (TQ1: 0.65 at 48 B)
    SPQ64 = "SPQ64"     # sub  6,  8 bits: ~0.53 rad


@dataclass(frozen=True)
class _Encoding:
    rate: float                    # bits per dimension
    qtype: Optional[int] = None    # ScalarQuantizer.QT_*; None = raw float32 (or seeded PQ)
    rotate: bool = False           # seeded orthonormal rotation in front
    sub: int = 0                   # seeded PQ: sub-vector dims
    bits: int = 0                  # seeded PQ: bits per sub-vector code


_ENCODINGS = {
    Quantize.FP32: _Encoding(rate=32),
    Quantize.FP16: _Encoding(rate=16, qtype=faiss.ScalarQuantizer.QT_fp16),
    Quantize.SQ8: _Encoding(rate=8, qtype=faiss.ScalarQuantizer.QT_8bit),
    Quantize.TQ8: _Encoding(rate=8, qtype=faiss.ScalarQuantizer.QT_8bit_tqmse, rotate=True),
    Quantize.SQ6: _Encoding(rate=6, qtype=faiss.ScalarQuantizer.QT_6bit),
    Quantize.SQ4: _Encoding(rate=4, qtype=faiss.ScalarQuantizer.QT_4bit),
    Quantize.TQ4: _Encoding(rate=4, qtype=faiss.ScalarQuantizer.QT_4bit_tqmse, rotate=True),
    Quantize.TQ3: _Encoding(rate=3, qtype=faiss.ScalarQuantizer.QT_3bit_tqmse, rotate=True),
    Quantize.TQ2: _Encoding(rate=2, qtype=faiss.ScalarQuantizer.QT_2bit_tqmse, rotate=True),
    Quantize.TQ1: _Encoding(rate=1, qtype=faiss.ScalarQuantizer.QT_1bit_tqmse, rotate=True),
    Quantize.SPQ6: _Encoding(rate=0.125, rotate=True, sub=96, bits=12),
    Quantize.SPQ9: _Encoding(rate=0.1875, rotate=True, sub=64, bits=12),
    Quantize.SPQ12: _Encoding(rate=0.25, rotate=True, sub=48, bits=12),
    Quantize.SPQ16: _Encoding(rate=1 / 3, rotate=True, sub=24, bits=8),
    Quantize.SPQ24: _Encoding(rate=0.5, rotate=True, sub=24, bits=12),
    Quantize.SPQ36: _Encoding(rate=0.75, rotate=True, sub=16, bits=12),
    Quantize.SPQ48: _Encoding(rate=1, rotate=True, sub=12, bits=12),
    Quantize.SPQ64: _Encoding(rate=4 / 3, rotate=True, sub=6, bits=8),
}


class _SeededPQ:
    """Product quantizer with a seeded random codebook: each `sub`-dim slice
    of a (rotated) direction is coded as the codeword of largest inner
    product among 2^bits unit Gaussian codewords, and decodes to the
    concatenated codewords, renormalized. After a random rotation every
    slice of a unit vector carries ~sub/dim of its energy, so giving each
    decoded slice equal weight loses little, and nothing per vector (no
    norms, no scales) needs storing. The codebook is regenerated from its
    seed, so it costs only the seed -- what makes rates below 1 bit/dim
    usable at budgets of a few KB, where any trained codebook would not
    fit. Same compute_codes/decode/train interface as faiss.ScalarQuantizer."""

    _SEED_BYTES = 4

    def __init__(self, dim: int, sub: int, bits: int, seed: int = 0x5EED) -> None:
        if dim % sub:
            raise ValueError(f"seeded PQ needs dim % sub == 0; got dim={dim}, sub={sub}")
        self.dim, self.sub, self.bits = dim, sub, bits
        self.m = dim // sub
        self.code_size = (self.m * bits + 7) // 8
        cb = np.random.default_rng(seed + 1000 * sub + bits).standard_normal((1 << bits, sub))
        self.codebook = (cb / np.linalg.norm(cb, axis=1, keepdims=True)).astype(np.float32)
        self._index = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_index"] = None    # rebuilt from the codebook on demand
        return state

    def train(self, X: np.ndarray) -> None:
        pass

    def codebook_bytes(self) -> int:
        return self._SEED_BYTES

    def _nearest_codewords(self, S: np.ndarray, top: int) -> np.ndarray:
        """(n x top) indices of the codewords of largest inner product with
        each row of S (one slice per row), best first: faiss's blocked GEMM
        with a running top-k, rather than an (n x 2^bits) score matrix."""
        if self._index is None:
            self._index = faiss.IndexFlatIP(self.sub)
            self._index.add(self.codebook)
        _, idx = self._index.search(np.ascontiguousarray(S, dtype=np.float32), top)
        return idx.astype(np.int64)

    def compute_codes(self, X: np.ndarray) -> np.ndarray:
        X = np.ascontiguousarray(X, dtype=np.float32).reshape(len(X), self.m, self.sub)
        idx = np.empty((len(X), self.m), dtype=np.int64)
        for j in range(self.m):
            idx[:, j] = self._nearest_codewords(X[:, j, :], 1)[:, 0]
        return self._pack(idx)

    def _pack(self, idx: np.ndarray) -> np.ndarray:
        bits = ((idx[:, :, None] >> np.arange(self.bits)[None, None, :]) & 1).astype(np.uint8)
        return np.packbits(bits.reshape(len(idx), -1), axis=1, bitorder="little")

    def fit_codes(self, U: np.ndarray, X: np.ndarray, labels: np.ndarray, candidates: int = 8,
                  passes: int = 2) -> np.ndarray:
        """Codes for cap directions U chosen so each decoded direction holds
        its cap's members X[labels == j] within the least angle, rather than
        lying nearest U[j]. Every decoded vector has the same norm (unit
        codewords), so cos(x, decode) is a sum over slices of <x_j, cw_j>,
        and a coordinate search -- each slice in turn, over its
        `candidates` codewords nearest U[j]'s slice, keeping the one that
        maximizes the members' least cosine -- never makes a cap wider than
        nearest-codeword encoding does. Caps with no member keep that
        encoding."""
        k, m, s, T = len(U), self.m, self.sub, min(candidates, 1 << self.bits)
        Us = np.ascontiguousarray(U, dtype=np.float32).reshape(k, m, s)
        cand = np.empty((k, m, T), dtype=np.int64)
        for j in range(m):
            cand[:, j, :] = self._nearest_codewords(Us[:, j, :], T)
        choice = np.zeros((k, m), dtype=np.int64)
        labels = np.asarray(labels, dtype=np.int64)
        n = len(X)
        if n:
            Xs = np.ascontiguousarray(X, dtype=np.float32).reshape(n, m, s)
            G = np.empty((n, m, T), dtype=np.float32)
            for j in range(m):
                G[:, j, :] = np.einsum("ns,nts->nt", Xs[:, j, :], self.codebook[cand[labels, j, :]])
            rows = np.arange(n)
            cur = G[rows[:, None], np.arange(m)[None, :], choice[labels]].sum(axis=1)
            order = np.argsort(labels, kind="stable")
            caps, starts = np.unique(labels[order], return_index=True)
            for _ in range(passes):
                for j in range(m):
                    base = cur - G[rows, j, choice[labels, j]]
                    trial = base[:, None] + G[:, j, :]
                    worst = np.minimum.reduceat(trial[order], starts, axis=0)
                    choice[caps, j] = np.argmax(worst, axis=1)
                    cur = base + G[rows, j, choice[labels, j]]
        idx = np.take_along_axis(cand, choice[:, :, None], axis=2)[:, :, 0]
        return self._pack(idx)

    def _indices(self, codes: np.ndarray) -> np.ndarray:
        codes = np.ascontiguousarray(codes, dtype=np.uint8)
        if self.bits == 8:
            return codes[:, :self.m].astype(np.int64)
        if self.bits == 12 and self.m % 2 == 0:
            # Two 12-bit codes per 3 bytes, little-endian (as _pack writes them).
            b = codes[:, :3 * self.m // 2].reshape(len(codes), self.m // 2, 3).astype(np.int64)
            idx = np.empty((len(codes), self.m // 2, 2), dtype=np.int64)
            idx[:, :, 0] = b[:, :, 0] | ((b[:, :, 1] & 0x0F) << 8)
            idx[:, :, 1] = (b[:, :, 1] >> 4) | (b[:, :, 2] << 4)
            return idx.reshape(len(codes), self.m)
        bits = np.unpackbits(codes, axis=1, count=self.m * self.bits, bitorder="little")
        bits = bits.reshape(len(codes), self.m, self.bits).astype(np.int64)
        return (bits << np.arange(self.bits)[None, None, :]).sum(axis=2)

    def decode(self, codes: np.ndarray) -> np.ndarray:
        return self.codebook[self._indices(codes)].reshape(len(codes), self.dim)

def _fit_scalar_codes(sq: faiss.ScalarQuantizer, U: np.ndarray, X: np.ndarray, labels: np.ndarray,
                      passes: int = 2) -> np.ndarray:
    """_SeededPQ.fit_codes for a TurboQuant scalar quantizer: every
    coordinate decodes to one of the 2^b levels at the head of sq.trained,
    so starting from U's own code, each coordinate in turn moves one level
    up or down when that raises the least cosine between the cap's members
    and its decoded direction (the norm now varies, so it is tracked). The
    fitted vector of levels re-encodes to exactly those levels."""
    D = np.ascontiguousarray(U, dtype=np.float32)
    C = sq.decode(sq.compute_codes(D)).astype(np.float64)
    n = len(X)
    if n == 0 or len(C) == 0:
        return sq.compute_codes(np.ascontiguousarray(C, dtype=np.float32))
    n_levels = 1 << int(round(np.log2(len(faiss.vector_to_array(sq.trained)) + 1) - 1))
    levels = np.sort(faiss.vector_to_array(sq.trained)[:n_levels].astype(np.float64))
    idx = np.searchsorted(levels, C - 1e-12)
    idx = np.clip(idx, 0, n_levels - 1)
    labels = np.asarray(labels, dtype=np.int64)
    order = np.argsort(labels, kind="stable")
    caps, starts = np.unique(labels[order], return_index=True)
    Xo, lo = X[order], labels[order]
    dot = np.einsum("nd,nd->n", Xo, C[lo])
    norm2 = np.einsum("kd,kd->k", C, C)
    for _ in range(passes):
        for i in range(C.shape[1]):
            cur = np.minimum.reduceat(dot / np.sqrt(norm2[lo]), starts)
            best, best_step = cur, np.zeros(len(caps), dtype=np.int64)
            for step in (-1, 1):
                j = np.clip(idx[caps, i] + step, 0, n_levels - 1)
                delta = levels[j] - C[caps, i]
                full = np.zeros(len(C))
                full[caps] = delta
                trial_dot = dot + Xo[:, i] * full[lo]
                trial_n2 = norm2 + full * (2.0 * C[:, i] + full)
                val = np.minimum.reduceat(trial_dot / np.sqrt(trial_n2[lo]), starts)
                better = (val > best) & (delta != 0)
                best = np.where(better, val, best)
                best_step = np.where(better, step, best_step)
            moved = caps[best_step != 0]
            if len(moved):
                j = np.clip(idx[moved, i] + best_step[best_step != 0], 0, n_levels - 1)
                delta = np.zeros(len(C))
                delta[moved] = levels[j] - C[moved, i]
                dot += Xo[:, i] * delta[lo]
                norm2 += delta * (2.0 * C[:, i] + delta)
                C[moved, i] = levels[j]
                idx[moved, i] = j
    return sq.compute_codes(np.ascontiguousarray(C, dtype=np.float32))


_ROTATION_SEED = 0x5EED
_ROTATION_SEED_BYTES = 4
_ANGLE_GRID_BYTES = 8      # a static 8-bit filter's angle grid: two float32
_PROBE_RNG = np.random.default_rng(0x5EED)

# Candidate encodings CapCoverFilter.auto() chooses among: the seeded-PQ rungs
# below 1 bit/dim plus TurboQuant above it, so small budgets can buy many
# coarse caps and large ones fewer, finer caps. Coarser rungs (6 and 9 B,
# ~1.2 and 1.12 rad) measured worse: the reach proxy over-rates them once
# the encoding error rivals the caps' own angles (+0.16 FPR at k = 0.5%).
AUTO_CANDIDATES = (Quantize.SPQ12, Quantize.SPQ16, Quantize.SPQ24, Quantize.SPQ36, Quantize.SPQ48,
                   Quantize.SPQ64, Quantize.TQ2, Quantize.TQ3, Quantize.TQ4)

class CapCoverFilter(Filter):
    """Ball-Cover for unit-norm vectors, with spherical caps in place of
    balls: each cap stores an encoded unit direction u and an angle theta,
    and a query is accepted iff it lies within distance tau of some unit
    vector of some cap. Zero false negatives by construction: every member
    lies in a cap, measured exactly around the direction as stored.

    Accept rule. For a unit query, angle(q, u) <= theta + alpha with
    alpha = 2 arcsin(tau / 2): the exact tau-neighbourhood of the cap on the
    sphere, so no rule that knows only the cap can be tighter. (A ball's
    |q - c| <= r + tau holds the same cap but over-expands it unless its
    center sits at norm 1 / cos(theta); BallCoverFilter's common norm costs
    it a little -- 0.1-0.5% FPR on the same cover.) For a query of norm n
    the rule generalizes exactly: angle(q, u) - theta <= beta(n),
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
    _fit_scalar_codes) -- -0.02..-0.08 FPR at k <= 5% of |B|, free in bytes.

    Choosing the encoding. How coarse is best depends on the budget per
    member, so `quantize` may be a sequence of candidates (auto() uses
    AUTO_CANDIDATES). construct() then builds covers at each candidate's cap
    count from one merge (ConstructInterface.iter_candidates), scores each
    by sum over caps of phi^p (p = selection_exponent, phi at
    selection_tau, exact per encoding) and keeps the best (_choose). A
    proxy over member angles (what the merge cost uses) did 5x worse;
    giving each cap its own encoding scored no better at 3-4x the
    construction time. The proxy errs coarse at small budgets: against
    the best fixed encoding per cell (|B| = 8000, folds 0-2) it is +0.015 /
    +0.014 / +0.008 FPR at k = 1/2/5% of |B| (CC-News: SPQ12 where SPQ24
    or SPQ16 is 0.04-0.08 better), within +-0.003 elsewhere. The default
    selection="holdout" (_choose_holdout) keeps the proxy's pick unless
    held-out members show a finer encoding significantly better: -0.003 /
    -0.008 / -0.015 / -0.006 FPR against the proxy alone at k = 0.5/1/2/5%
    (25 cells each, no cell worse by more than 0.0012), unchanged at
    10-20%, within 0.001 of the hindsight-best encoding at 2-5%, for ~1.7x
    the construction time. Letting it override toward coarser encodings
    measured +0.003..+0.011 at 10-15% (ChatGPT-paraphrases); choosing on
    held-out members alone, without the proxy, +0.024 on paraphrase
    corpora at 1% (held-out cluster-mates are all accepted there, and
    their differences are noise). Sub-12-byte rungs (SPQ6, SPQ9) win only
    3 of 30 cells at k <= 1% even in hindsight.

    Angles are stored in 8 bits over the filter's own [min, max] when no
    insert strategy can grow one past max (else 16 bits over [0, pi]).

    Measured (src/compare_cap_cover.py, blocklist workload, 5 corpora x 5
    folds, tau = 0.1, equal bytes = Ball-Cover TQ4 at k = k_fraction * |B|
    balls). |B| = 8000: auto() against the best of five Ball-Cover
    configurations *per cell* (sphere_merge at TQ1-TQ4, minimax_sphere at
    TQ4) has mean FPR .609/.394/.289/.166/.121/.098/.087 against
    .927/.705/.445/.272/.147/.111/.097 at k = 0.5/1/2/5/10/15/20%, winning
    25/25/25/25/20/22/16 of the 25 cells (losses <= 0.028, WildChat and
    ChatGPT-paraphrases at k >= 10%: WildChat's withheld-cluster queries
    crowd a region where Ball's sphere_merge cover is tighter -- 0.076 vs
    0.088 with exact directions on fold 1 -- and ChatGPT's want TQ3 where
    the proxy picks TQ2). The advantage holds for query tau 0.05-0.2 at
    k <= 5% (every cell). |B| = 80K (proxy selection): .772/.614/.441/.249/
    .171/.136/.114; Ball-Cover's paper configuration .963/.897/.780 at
    k = 0.5/1/2% (fold-paired -0.19/-0.28/-0.34; its larger k and every
    sphere_merge configuration exceed the benchmark's 5-minute job limit
    on this machine). One P-core, batches of 32: 10K vectors construct in
    2.8-15.9 s and answer 4.3K-42K queries/s; 100K (|B| = 80K) in 23-88 s
    (proxy; 166-190 s with the held-out veto) at 1.0K-5.6K queries/s.

    Working basis. Rotating encodings quantize in a seeded orthonormal
    basis; members and queries are rotated in float64 and every cap angle is
    measured in float64 in that basis around the decoded direction query()
    scores against. Strategies only ever see working-basis unit vectors;
    angles are rotation-invariant, so none of them needs to know.

    query() scans the caps in blocks: decode a block (compiled for seeded-PQ
    and TurboQuant codes, see scan_kernels), one float32 GEMM of the unit
    queries against the decoded vectors, compared in one compiled pass with
    |raw| cos(theta + beta); pairs within float32's error bound of that
    boundary are re-decided exactly in float64, so decisions match a
    float64 evaluation of the rule. Nothing decoded persists between calls.
    One P-core, batches of 32: 2.5x the throughput of the numpy scan at
    |B| = 80K (701 -> 1720 q/s on 21K SPQ36 caps), 1.9x at 8K (3.1K caps).
    Seeded-PQ codes with small codebooks skip decoding: per chunk of
    queries, a table of each slice against each codeword, summed per cap
    (_scan_tables, where _tables_pay; 3.2x on SPQ16). Exact pruning (cell
    bounds, partial dot products) cannot help much here: a coarse cap's
    decoded direction sits up to ~1 rad off its members, so caps that
    neighbour in the data do not neighbour in decoded space."""

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

    # Caps decoded per block of the scan (1.5 MB of float32 at d = 384, so
    # the GEMM reads them from L2), and queries per GEMM.
    _SCAN_BLOCK = 1024
    _SCAN_QUERY_CHUNK = 1024

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
        fit_rounds: int = 1,
        selection_tau: float = 0.1,
        selection_exponent: float = 8.0,
        selection: str = "proxy",
        holdout_fraction: float = 0.25,
        holdout_queries: int = 2000,
        holdout_z: float = 3.0,
    ) -> None:
        super().__init__(dim, max_memory=max_memory)
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
        self.fit_rounds = fit_rounds
        self._angle_levels = (1 << angle_bits) - 1
        # Angle grid [lo, hi]: all of [0, pi] by default; a static 8-bit
        # filter narrows it to its own angles at construct() (see there).
        self._angle_lo, self._angle_hi = 0.0, float(np.pi)
        self.selection_tau = selection_tau
        self.selection_exponent = selection_exponent
        if selection not in ("proxy", "holdout"):
            raise ValueError(f"selection must be 'proxy' or 'holdout', got {selection!r}")
        self.selection = selection
        self.holdout_fraction = holdout_fraction
        self.holdout_queries = holdout_queries
        self.holdout_z = holdout_z
        self.selection_scores = None    # encoding -> score, set by construct() when choosing
        self.holdout_scores = None      # encoding -> held-out acceptance (selection="holdout")
        self.selection_override = None  # (proxy's pick, override, b, c) when the held-out test overrode
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
        self._dirs = None   # decoded unit directions by id: transient, like BallCoverFilter's all_centers
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
        """A filter that picks its encoding(s) from AUTO_CANDIDATES for the
        budget at construct(): by the reach proxy with the held-out veto
        (selection="holdout", see _choose_holdout) unless told otherwise."""
        candidates = [q for q in AUTO_CANDIDATES if dim % max(_ENCODINGS[q].sub, 1) == 0]
        kwargs.setdefault("selection", "holdout")
        return cls(dim, quantize=candidates, max_memory=max_memory, **kwargs)

    # ------------------------------------------------------------ properties
    # BallCoverFilter's names too, so code written against max_spheres reads
    # either filter.
    @property
    def max_spheres(self) -> int:
        return self.max_caps

    @property
    def max_construct_spheres(self) -> int:
        return self.max_construct_caps

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
                self._dirs[ids] = self.decode(self._codes[ids])
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

    def bytes_per_ball(self) -> int:
        return self.bytes_per_cap()

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

    def _train_quantizers(self, directions: np.ndarray) -> None:
        self._quantizers = [self._make_quantizer(q) for q in self.levels]
        D = np.ascontiguousarray(directions, dtype=np.float32)
        for sq in self._quantizers:
            if sq is not None and len(D):
                sq.train(D)

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

    def encode(self, directions: np.ndarray, level: Optional[int] = None) -> np.ndarray:
        """Codes (n x code_size(level)) for unit working-basis directions."""
        level = self._uniform if level is None else level
        D = np.ascontiguousarray(directions, dtype=np.float32)
        sq = self._quantizers[level]
        if sq is None:
            return D.view(np.uint8).reshape(len(D), -1).copy()
        if len(D) == 0:
            return np.empty((0, self.code_size(level)), dtype=np.uint8)
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
        codewords nearest u's; TurboQuant: greedy one-level moves of single
        coordinates. Other encodings: u's own code. Never worse than u's own
        code, which is where both searches start."""
        u = np.asarray(u, dtype=np.float64)
        P = np.asarray(parts, dtype=np.float64)
        t = np.asarray(part_angles, dtype=np.float64)
        sq = self._quantizers[self._uniform]
        own = self.encode(u[None, :])

        def stored(cos):
            return (np.arccos(np.clip(cos, -1.0, 1.0)) + t[:, None]).max(axis=0)

        if isinstance(sq, _SeededPQ):
            m, s, T = sq.m, sq.sub, min(candidates, 1 << sq.bits)
            cand = sq._nearest_codewords(u.reshape(m, s), T)                    # (m, T)
            cw = sq.codebook[cand].astype(np.float64)                           # (m, T, s)
            norm = np.sqrt(np.einsum("mts,mts->mt", cw, cw))
            G = np.einsum("mts,pms->pmt", cw, P.reshape(len(P), m, s))          # (p, m, T)
            n2 = norm[:, 0] ** 2
            choice = np.zeros(m, dtype=np.int64)
            dots, sq_norm = G[:, np.arange(m), choice].sum(axis=1), n2.sum()
            for _ in range(passes):
                for j in range(m):
                    base = dots - G[:, j, choice[j]]
                    base_n = sq_norm - norm[j, choice[j]] ** 2
                    trial_n = base_n + norm[j] ** 2                              # (T,)
                    score = stored((base[:, None] + G[:, j, :]) / np.sqrt(trial_n)[None, :])
                    choice[j] = int(np.argmin(score))
                    dots, sq_norm = base + G[:, j, choice[j]], base_n + norm[j, choice[j]] ** 2
            return sq._pack(cand[np.arange(m), choice][None, :])
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

    def _is_turboquant(self) -> bool:
        return self.quantize in (Quantize.TQ1, Quantize.TQ2, Quantize.TQ3, Quantize.TQ4, Quantize.TQ8)

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
        codes = np.zeros((new_cap, self._codes.shape[1]), dtype=np.uint8)
        codes[:current] = self._codes
        self._codes = codes
        angles = np.zeros(new_cap, dtype=np.uint16)
        angles[:current] = self._angle_codes
        self._angle_codes = angles
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
        self._dirs = None
        self._constructed = False
        if self.insert_method:
            self.insert_method.reset_buffers(capacity)

    def _add_caps(self, directions: np.ndarray, angles: np.ndarray,
                  codes: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Stores caps (u, theta), each widened by its direction's encoding
        error so it still holds every unit vector the unencoded cap held
        (triangle inequality on the sphere). Returns (ids, decoded unit
        directions). `codes`, if given, are stored instead of U's own
        (fitted codes, see construct())."""
        U = unit_rows(directions)
        n = len(U)
        if n == 0:
            return np.empty(0, dtype=np.int32), np.empty((0, self.dim))
        self._ensure_capacity(self._num_caps + n)
        ids = self._all_caps[self._num_caps:self._num_caps + n].copy()
        if codes is None:
            codes = self.encode(U)
        self._codes[ids, :codes.shape[1]] = codes
        decoded = self.decode(codes)
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
        dist = np.full(labels.shape, np.inf)
        for s in range(0, len(X), 2048):
            rows = slice(s, s + 2048)
            lab = labels[rows]
            d = angle_between(X[rows, None, :], decoded[np.maximum(lab, 0)])
            dist[rows] = np.where(lab >= 0, d, np.inf)

        home_d = angle_between(X, decoded[home])
        # A construct method whose cap does not hold its own nearest member
        # is repaired here rather than trusted: the fallback must be legal.
        np.maximum.at(bound, home, home_d)

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

    def _reach(self, X: np.ndarray, U: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Per cap, phi: how far along the data the cap on direction U[j]
        holding members X[labels == j] reaches once encoded. phi =
        sqrt((theta' + alpha)^2 - delta^2), theta' the members' widest angle
        around the decoded direction, delta the direction's encoding error
        -- the in-data reach when the error is orthogonal to the data (see
        the class docstring). Trains the quantizer on U."""
        self._train_quantizers(U)
        order, starts, counts = segments(labels, len(U))
        alpha = 2.0 * np.arcsin(min(1.0, 0.5 * self.selection_tau))
        # Scored on nearest-codeword encodings: scoring on fitted codes
        # (what construct() stores) measured to pick the same encoding.
        decoded = self.decode_directions(U)
        delta = angle_between(U, decoded)
        theta = segment_max(angle_between(X, decoded[labels])[order], starts, counts)
        return np.sqrt(np.maximum((theta + alpha) ** 2 - delta ** 2, 1e-300))[counts > 0]

    def _choose(self, X: np.ndarray, weights: np.ndarray) -> Cover:
        """Scores a cover per candidate encoding at that encoding's cap count
        (one merge, see ConstructInterface.iter_candidates) by how far its
        caps reach along the data -- sum over caps of phi^p -- keeps the
        best, fixes the filter's encoding to it, and returns its finished
        cover. Candidates come largest cap count first, and the scan stops
        once the score has risen twice in a row: it is unimodal along the
        ladder (more, coarser caps against fewer, finer ones).

        Levels whose budget buys a cap per distinct member or more all get
        that same cover, on which a finer code always reaches less (phi
        grows with delta at fixed theta): only the finest of them is
        scored."""
        ks = []
        for level in range(len(self.levels)):
            self._set_uniform(level)
            ks.append(min(self.max_construct_caps, len(X)))
        levels_at = {}
        for level, k in enumerate(ks):
            levels_at.setdefault(k, []).append(level)
        if len(X) in levels_at:
            levels_at[len(X)] = [max(levels_at[len(X)], key=lambda lv: self._code_sizes[lv])]
        p = self.selection_exponent
        scores, best, worse = {}, None, 0
        for k, cover in self.construct_method.iter_candidates(self, X, ks, weights):
            cover = cover._replace(directions=unit_rows(cover.directions))
            improved = False
            for level in levels_at[k]:
                self._set_uniform(level)
                phi = self._reach(X, cover.directions, cover.labels)
                score = float(np.logaddexp.reduce(p * np.log(phi))) if len(phi) else -np.inf
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

    def _choose_holdout(self, X: np.ndarray, weights: np.ndarray) -> Cover:
        """_choose with a held-out veto. The reach proxy's cover is built
        as _choose builds it; then candidate covers are built per encoding
        on all but `holdout_fraction` of the distinct members, at that share
        of each encoding's cap count, with codes fitted as construct() fits
        them and angles from the training members, and each encoding gets
        the set of held-out members farther than selection_tau from every
        training member that its caps would accept at selection_tau. The
        finer encoding accepting the fewest replaces the proxy's only if it
        beats it in a paired test on those members (b - c >= holdout_z sqrt(b + c),
        b and c the members only one of the two accepts) -- held-out members
        are cluster-mates, often all accepted at small k, where differences
        are noise -- and its cover is then built on all members. Without
        an override the result is exactly _choose's."""
        cover = self._choose(X, weights)
        chosen = self._uniform
        rng = np.random.default_rng(0x5EED)
        hold = rng.random(len(X)) < self.holdout_fraction
        if hold.sum() < 50 or (~hold).sum() < 50:
            return cover
        Xt, wt, V = X[~hold], weights[~hold], X[hold]
        alpha = 2.0 * np.arcsin(min(1.0, 0.5 * self.selection_tau))
        ip, _ = nearest_directions(V, Xt)
        far = V[np.flatnonzero(2.0 - 2.0 * ip.astype(np.float64) > self.selection_tau ** 2)]
        if len(far) > self.holdout_queries:
            far = far[rng.choice(len(far), self.holdout_queries, replace=False)]
        if len(far) < 50:
            self._set_uniform(chosen)
            return cover
        far32 = np.ascontiguousarray(far, dtype=np.float32)
        ks_full, ks = [], []
        for level in range(len(self.levels)):
            self._set_uniform(level)
            ks_full.append(min(self.max_construct_caps, len(X)))
            ks.append(max(1, min(round(self.max_construct_caps * len(Xt) / len(X)), len(Xt))))
        levels_at = {}
        for level, k in enumerate(ks):
            levels_at.setdefault(k, []).append(level)
        if len(Xt) in levels_at:
            levels_at[len(Xt)] = [max(levels_at[len(Xt)], key=lambda lv: self._code_sizes[lv])]
        hits = {}
        for k, cand in self.construct_method.iter_candidates(self, Xt, ks, wt):
            U, labels = unit_rows(cand.directions), cand.labels
            for level in levels_at[k]:
                self._set_uniform(level)
                self._train_quantizers(U)
                sq = self._quantizers[level]
                if hasattr(sq, "fit_codes"):
                    codes = sq.fit_codes(U, Xt, labels, self.fit_candidates, self.fit_passes)
                elif self._is_turboquant():
                    codes = _fit_scalar_codes(sq, U, Xt, labels, self.fit_passes)
                else:
                    codes = self.encode(U)
                D = self.decode(codes)
                order, starts, counts = segments(labels, len(U))
                theta = segment_max(angle_between(Xt, D[labels])[order], starts, counts)
                thr = np.cos(np.minimum(theta + alpha, np.pi)).astype(np.float32)
                D32 = D.astype(np.float32)
                hit = np.zeros(len(far32), dtype=bool)
                for s in range(0, len(D32), 4096):
                    hit |= (far32 @ D32[s:s + 4096].T >= thr[None, s:s + 4096]).any(axis=1)
                hits[level] = hit
        self.holdout_scores = {self.levels[lv].value: float(h.mean()) for lv, h in hits.items()}
        # The proxy's pick may have no held-out row (its count rounded onto
        # a coarser level's): then the comparison is to the nearest scored.
        base = chosen if chosen in hits else min(hits, key=lambda lv: abs(self._code_sizes[lv]
                                                                       - self._code_sizes[chosen]))
        # Only finer encodings may override: the proxy's known bias is toward
        # coarse codes, and held-out cluster-mates favour more caps than the
        # query mix does (coarser overrides measured +0.003..+0.011 at
        # kf = 10-15% on ChatGPT-paraphrases).
        finer = [lv for lv in hits if self._code_sizes[lv] >= self._code_sizes[base]]
        rival = min(finer, key=lambda lv: (hits[lv].sum(), abs(lv - base)))
        b = int((hits[base] & ~hits[rival]).sum())
        c = int((hits[rival] & ~hits[base]).sum())
        if rival == base or b - c < self.holdout_z * np.sqrt(max(b + c, 1)):
            self._set_uniform(chosen)
            return cover
        self._set_uniform(rival)
        self.selection_override = (self.levels[chosen].value, self.levels[rival].value, b, c)
        for _, cand in self.construct_method.iter_candidates(self, X, [ks_full[rival]], weights):
            return self.construct_method.finish(self, X, cand)

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

    def construct(self, vectors: np.ndarray) -> None:
        if len(vectors) == 0:
            self._reset_buffers(0)
            self._constructed = True
            return
        distinct, inverse, copies = self._distinct_rows(vectors)
        X = self._prepare_members(distinct)
        if len(self.levels) > 1:
            cover = self._choose_holdout(X, copies) if self.selection == "holdout" else self._choose(X, copies)
        else:
            for _, cover in self.construct_method.iter_candidates(self, X, [self.max_construct_caps], copies):
                cover = self.construct_method.finish(self, X, cover)
                break
        U = unit_rows(cover.directions)
        theta = np.asarray(cover.angles, dtype=np.float64)
        home, neighbors = np.asarray(cover.labels, dtype=np.int64), cover.neighbors
        self._reset_buffers(len(U))
        self._train_quantizers(U)
        codebook = self._codebook_bytes()
        if self.max_memory is not None and codebook + self.bytes_per_cap() > self.max_memory:
            raise InsufficientMemoryError(
                f"max_memory={self.max_memory} bytes cannot hold one {self.quantize.value} cap "
                f"plus its {codebook}-byte codebook.")
        sq = self._quantizers[self._uniform]
        if self.fit_codes and (hasattr(sq, "fit_codes") or self._is_turboquant()):
            # Fit codes to the members each cap holds, re-settle the members
            # around the new directions, and repeat: each round's members
            # are the ones the last round's caps actually hold. A second
            # round measured within 0.002 FPR of one.
            owner = home
            for _ in range(max(1, self.fit_rounds)):
                if hasattr(sq, "fit_codes"):
                    codes = sq.fit_codes(U, X, owner, self.fit_candidates, self.fit_passes)
                else:
                    codes = _fit_scalar_codes(sq, U, X, owner, self.fit_passes)
                angles, owner = self._tight_angles(X, U, self.decode(codes), theta, home, neighbors)
        else:
            codes = self.encode(U)
            angles, _ = self._tight_angles(X, U, self.decode(codes), theta, home, neighbors)
        ids, decoded = self._add_caps(U, theta, codes=codes)
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

    # ----------------------------------------------------------------- query
    def _beta(self, norms: np.ndarray, tau: float) -> Tuple[np.ndarray, np.ndarray]:
        """(beta, feasible): how far past a cap's edge (in angle) a query of
        each norm may be and still be within tau' of a unit vector in the
        cap, and whether any unit vector is within tau' of it at all."""
        tp = float(tau) + self._COVERAGE_EPSILON + self._NORM_TOLERANCE
        n = np.asarray(norms, dtype=np.float64)
        feasible = (np.abs(n - 1.0) <= tp) & (n > 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            cos_b = (n * n + 1.0 - tp * tp) / (2.0 * n)
        beta = np.arccos(np.clip(np.where(feasible, cos_b, 1.0), -1.0, 1.0))
        return beta, feasible

    def _slack(self) -> float:
        """Bound on a float32 inner product's error for unit vectors,
        (d + 4) * 2^-24, doubled: pairs this close to the boundary are
        re-decided in float64."""
        return 2.0 * (self.dim + 4) * 2.0 ** -24

    def query(self, vectors: np.ndarray, distance_threshold: float) -> List[bool]:
        if not self._constructed:
            raise ValueError("CapCoverFilter has not been constructed.")
        if self._num_caps == 0 or len(vectors) == 0:
            return [False] * len(vectors)
        return self._scan(vectors, float(distance_threshold)).tolist()

    def _block_decoder(self):
        """decode(codes, out, norms): the float32 vectors `codes` decode to
        (exactly _decode_raw's) into `out`, with their float64 norms."""
        from filters.cap_cover import scan_kernels
        sq, cs = self._quantizers[self._uniform], self.code_size()
        if isinstance(sq, _SeededPQ):
            return lambda codes, out, norms: scan_kernels.decode_codebook(
                np.ascontiguousarray(codes[:, :cs]), sq.codebook, sq.bits, sq.m, out, norms)
        if self._is_turboquant():
            bits = int(round(_ENCODINGS[self.quantize].rate))
            levels = faiss.vector_to_array(sq.trained)[:1 << bits].astype(np.float32)
            return lambda codes, out, norms: scan_kernels.decode_levels(
                np.ascontiguousarray(codes[:, :cs]), levels, bits, self.dim, out, norms)

        def generic(codes, out, norms):
            out[:] = self._decode_raw(codes, self._uniform)
            norms[:] = np.sqrt(np.einsum("ij,ij->i", out.astype(np.float64), out.astype(np.float64)))
        return generic

    def _scan(self, vectors: np.ndarray, tau: float) -> np.ndarray:
        from filters.cap_cover import scan_kernels
        Qw = self.to_working(vectors)
        nq = np.sqrt(np.einsum("ij,ij->i", Qw, Qw))
        beta, feasible = self._beta(nq, tau)
        Qn = Qw / np.maximum(nq, 1e-300)[:, None]
        Qn32 = Qn.astype(np.float32)
        out = np.zeros(len(Qw), dtype=bool)
        slack = self._slack()
        ids = self.all_caps
        sq = self._quantizers[self._uniform]
        if isinstance(sq, _SeededPQ) and self._tables_pay(sq, len(ids)):
            return self._scan_tables(sq, ids, Qn, Qn32, beta, feasible, slack, out)
        decode = self._block_decoder()
        raw = np.empty((self._SCAN_BLOCK, self.dim), dtype=np.float32)
        norms = np.empty(self._SCAN_BLOCK)

        for start in range(0, len(ids), self._SCAN_BLOCK):
            pending = np.flatnonzero(~out & feasible)
            if len(pending) == 0:
                break
            block = ids[start:start + self._SCAN_BLOCK]
            nb = len(block)
            decode(self._codes[block], raw[:nb], norms[:nb])
            theta = self._angle_value(self._angle_codes[block])
            for q0 in range(0, len(pending), self._SCAN_QUERY_CHUNK):
                rows = pending[q0:q0 + self._SCAN_QUERY_CHUNK]
                S = Qn32[rows] @ raw[:nb].T
                scan_kernels.accept_block(S, rows, raw[:nb], norms[:nb], theta, beta, Qn, slack, out)
        return out

    # Queries per set of per-query codeword tables (scan_kernels.accept_tables).
    _TABLE_QUERY_CHUNK = 32

    @staticmethod
    def _tables_pay(sq: "_SeededPQ", n_caps: int) -> bool:
        """Whether scoring seeded-PQ codes through per-query codeword tables
        beats decoding them, measured on one P-core in batches of 32:
        8-bit codebooks always (tables fit L2; 3.2x on 3.1K SPQ16 caps,
        1.2x on 2.4K SPQ64 caps); 12-bit ones only with few slices and many
        caps (1.16x on 12K SPQ12 caps at |B| = 80K; a tie on 21K SPQ36 caps,
        0.6x on 3.1K SPQ24 caps), where a batch's tables outgrow L2."""
        if sq.bits <= 8:
            return True
        return sq.m <= 8 and n_caps >= 8192

    def _scan_tables(self, sq, ids, Qn, Qn32, beta, feasible, slack, out) -> np.ndarray:
        """_scan for seeded-PQ codes without decoding: per chunk of queries,
        a table of every slice against every codeword, summed per cap by
        scan_kernels.accept_tables, which decides exactly as accept_block."""
        from filters.cap_cover import scan_kernels
        codes = np.ascontiguousarray(self._codes[ids][:, :self.code_size()])
        theta = self._angle_value(self._angle_codes[ids])
        sq_norms = np.einsum("ij,ij->i", sq.codebook.astype(np.float64), sq.codebook.astype(np.float64))
        pending = np.flatnonzero(feasible)
        for q0 in range(0, len(pending), self._TABLE_QUERY_CHUNK):
            rows = pending[q0:q0 + self._TABLE_QUERY_CHUNK]
            Qs = Qn32[rows].reshape(len(rows), sq.m, sq.sub)
            T = np.ascontiguousarray(np.matmul(sq.codebook[None], Qs.transpose(1, 2, 0)))
            scan_kernels.accept_tables(codes, sq.bits, sq.m, T, sq_norms, sq.codebook, theta, rows, beta, Qn,
                                       slack, out)
        return out

    def surface_angle(self, vectors: np.ndarray) -> np.ndarray:
        """Per query, min over caps of angle(q, u) - theta, in float64 (an
        O(n * caps * d) diagnostic, not the query path): a unit query is
        accepted at tau iff this is <= 2 arcsin(tau' / 2)."""
        Qn = unit_rows(self.to_working(vectors))
        ids = self.all_caps
        if len(ids) == 0:
            return np.full(len(Qn), np.inf)
        C = self.decode(self._codes[ids])
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
        return self.insert_method.remove(self, self._prepare_members(vectors))

    def get_bytesize(self) -> int:
        if not self._constructed:
            return 0
        total = self._num_caps * self.bytes_per_cap() + self._codebook_bytes()
        if self.insert_method:
            total += self.insert_method.get_bytesize(self)
        return total

    def get_construction_bytesize(self) -> int:
        return self.construct_method.get_bytesize() if self.construct_method else 0
