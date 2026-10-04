import numpy as np
import faiss
import sys
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence, Tuple, Union
from filters.filter import Filter, InsufficientMemoryError
from filters.cap_cover.cap_interfaces import ConstructInterface, Cover, InsertInterface
from filters.cap_cover.cap_geometry import (angle_between, log_cap_measure, nearest_directions, segments,
                                            segment_max, unit_rows)


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
    # A denser ladder (not in AUTO_CANDIDATES; see AUTO_LADDER). 10- and
    # 12-bit rungs between the ones above, and 8-bit rungs at their byte
    # sizes, which score by table but blur ~0.03 rad more at equal bytes.
    SPQ18 = "SPQ18"     # sub 32, 12 bits: ~0.95 rad
    SPQ20 = "SPQ20"     # sub 24, 10 bits: ~0.93 rad
    SPQ30 = "SPQ30"     # sub 16, 10 bits: ~0.79 rad
    SPQ40 = "SPQ40"     # sub 12, 10 bits: ~0.69 rad
    SPQ60 = "SPQ60"     # sub  8, 10 bits: ~0.53 rad
    SPQ72 = "SPQ72"     # sub  8, 12 bits: ~0.46 rad
    SPQ96 = "SPQ96"     # sub  6, 12 bits: ~0.38 rad
    SPQ80 = "SPQ80"     # sub  6, 10 bits: ~0.44 rad
    SPQ24B8 = "SPQ24B8"  # sub 16, 8 bits: ~0.88 rad
    SPQ32B8 = "SPQ32B8"  # sub 12, 8 bits: ~0.79 rad
    SPQ48B8 = "SPQ48B8"  # sub  8, 8 bits: ~0.63 rad
    # Seeded PQ in a rank-r basis trained on the caps (see _BasisPQ), named
    # by code bytes; the basis costs r TQ4 rows (r * 192 B at d = 384).
    # Angles: mean over members of fold 0 at |B| = 8000, 5 corpora.
    PSPQ6 = "PSPQ6"     # r  48, sub 12, 12 bits: 0.81-0.95 rad
    PSPQ9 = "PSPQ9"     # r  48, sub  8, 12 bits: 0.75-0.88 rad
    PSPQ12 = "PSPQ12"   # r  64, sub  8, 12 bits: 0.70-0.82 rad
    PSPQ16 = "PSPQ16"   # r  96, sub  6,  8 bits: 0.68-0.76 rad
    PSPQ24 = "PSPQ24"   # r 128, sub  8, 12 bits: 0.58-0.64 rad
    PSPQ36 = "PSPQ36"   # r 144, sub  6, 12 bits: 0.50-0.56 rad
    PSPQ48 = "PSPQ48"   # r 192, sub  6, 12 bits: 0.45-0.48 rad
    # Per cap, PSPQ<b>'s code or SPQ<b>'s, whichever holds it tighter, the
    # choice in the code's top bit (see _DualPQ).
    DPQ6 = "DPQ6"
    DPQ9 = "DPQ9"
    DPQ12 = "DPQ12"
    DPQ16 = "DPQ16"
    DPQ24 = "DPQ24"
    DPQ36 = "DPQ36"
    DPQ48 = "DPQ48"
    # VA+-style KLT codes (see _KLTSQ), named by code bytes; the basis costs
    # `rank` TQ4 rows (rank * 196 B at d = 384) plus 768 B of mean.
    KSQ6 = "KSQ6"       # rank  48
    KSQ12 = "KSQ12"     # rank  80
    KSQ24 = "KSQ24"     # rank 144
    KSQ48 = "KSQ48"     # rank 224
    # Seeded PQ in a basis the decoder rebuilds from the cover's own whole-
    # space caps (see _BootPQ): no basis bytes, only for mixed static
    # covers (CapCoverFilter's `boot`). Rank, slice and bits as PSPQ*.
    BPQ12 = "BPQ12"
    BPQ16 = "BPQ16"
    BPQ24 = "BPQ24"
    BPQ36 = "BPQ36"
    BPQ48 = "BPQ48"


@dataclass(frozen=True)
class _Encoding:
    rate: float                    # bits per dimension
    qtype: Optional[int] = None    # ScalarQuantizer.QT_*; None = raw float32 (or seeded PQ)
    rotate: bool = False           # seeded orthonormal rotation in front
    sub: int = 0                   # seeded PQ: sub-vector dims
    bits: int = 0                  # seeded PQ: bits per sub-vector code
    rank: int = 0                  # seeded PQ in a trained basis of this rank (0: the whole space)
    dual: bool = False             # per cap, the basis code or a whole-space one of the same layout
    klt_bits: int = 0              # VA+-style KLT code of this many bits (rank: its KLT rows)
    boot: bool = False             # basis rebuilt from the cover's whole-space caps (_BootPQ)


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
    Quantize.SPQ18: _Encoding(rate=0.375, rotate=True, sub=32, bits=12),
    Quantize.SPQ20: _Encoding(rate=5 / 12, rotate=True, sub=24, bits=10),
    Quantize.SPQ30: _Encoding(rate=0.625, rotate=True, sub=16, bits=10),
    Quantize.SPQ40: _Encoding(rate=5 / 6, rotate=True, sub=12, bits=10),
    Quantize.SPQ60: _Encoding(rate=1.25, rotate=True, sub=8, bits=10),
    Quantize.SPQ72: _Encoding(rate=1.5, rotate=True, sub=8, bits=12),
    Quantize.SPQ96: _Encoding(rate=2, rotate=True, sub=6, bits=12),
    Quantize.SPQ80: _Encoding(rate=5 / 3, rotate=True, sub=6, bits=10),
    Quantize.SPQ24B8: _Encoding(rate=0.5, rotate=True, sub=16, bits=8),
    Quantize.SPQ32B8: _Encoding(rate=2 / 3, rotate=True, sub=12, bits=8),
    Quantize.SPQ48B8: _Encoding(rate=1, rotate=True, sub=8, bits=8),
    Quantize.PSPQ6: _Encoding(rate=0.125, rotate=True, sub=12, bits=12, rank=48),
    Quantize.PSPQ9: _Encoding(rate=0.1875, rotate=True, sub=8, bits=12, rank=48),
    Quantize.PSPQ12: _Encoding(rate=0.25, rotate=True, sub=8, bits=12, rank=64),
    Quantize.PSPQ16: _Encoding(rate=1 / 3, rotate=True, sub=6, bits=8, rank=96),
    Quantize.PSPQ24: _Encoding(rate=0.5, rotate=True, sub=8, bits=12, rank=128),
    Quantize.PSPQ36: _Encoding(rate=0.75, rotate=True, sub=6, bits=12, rank=144),
    Quantize.PSPQ48: _Encoding(rate=1, rotate=True, sub=6, bits=12, rank=192),
    Quantize.DPQ6: _Encoding(rate=0.125, rotate=True, sub=12, bits=12, rank=48, dual=True),
    Quantize.DPQ9: _Encoding(rate=0.1875, rotate=True, sub=8, bits=12, rank=48, dual=True),
    Quantize.DPQ12: _Encoding(rate=0.25, rotate=True, sub=8, bits=12, rank=64, dual=True),
    Quantize.DPQ16: _Encoding(rate=1 / 3, rotate=True, sub=6, bits=8, rank=96, dual=True),
    Quantize.DPQ24: _Encoding(rate=0.5, rotate=True, sub=8, bits=12, rank=128, dual=True),
    Quantize.DPQ36: _Encoding(rate=0.75, rotate=True, sub=6, bits=12, rank=144, dual=True),
    Quantize.DPQ48: _Encoding(rate=1, rotate=True, sub=6, bits=12, rank=192, dual=True),
    Quantize.KSQ6: _Encoding(rate=0.125, rotate=True, rank=48, klt_bits=48),
    Quantize.KSQ12: _Encoding(rate=0.25, rotate=True, rank=80, klt_bits=96),
    Quantize.KSQ24: _Encoding(rate=0.5, rotate=True, rank=144, klt_bits=192),
    Quantize.KSQ48: _Encoding(rate=1, rotate=True, rank=224, klt_bits=384),
    Quantize.BPQ12: _Encoding(rate=0.25, rotate=True, sub=8, bits=12, rank=64, boot=True),
    Quantize.BPQ16: _Encoding(rate=1 / 3, rotate=True, sub=6, bits=8, rank=96, boot=True),
    Quantize.BPQ24: _Encoding(rate=0.5, rotate=True, sub=8, bits=12, rank=128, boot=True),
    Quantize.BPQ36: _Encoding(rate=0.75, rotate=True, sub=6, bits=12, rank=144, boot=True),
    Quantize.BPQ48: _Encoding(rate=1, rotate=True, sub=6, bits=12, rank=192, boot=True),
}
BOOT_CANDIDATES = (Quantize.BPQ12, Quantize.BPQ16, Quantize.BPQ24, Quantize.BPQ36, Quantize.BPQ48)


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

    def __init__(self, dim: int, sub: int, bits: int, seed: int = 0x5EED, last_bits: Optional[int] = None) -> None:
        """`last_bits` < bits: the last slice draws only from the first
        2^last_bits codewords, leaving the top bits of its code free (for
        _DualPQ's family flag)."""
        if dim % sub:
            raise ValueError(f"seeded PQ needs dim % sub == 0; got dim={dim}, sub={sub}")
        self.dim, self.sub, self.bits = dim, sub, bits
        self.m = dim // sub
        self.last_bits = bits if last_bits is None else last_bits
        self.code_size = (self.m * bits + 7) // 8
        cb = np.random.default_rng(seed + 1000 * sub + bits).standard_normal((1 << bits, sub))
        self.codebook = (cb / np.linalg.norm(cb, axis=1, keepdims=True)).astype(np.float32)
        self._index = None
        self._last_index = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_index"] = state["_last_index"] = None    # rebuilt from the codebook on demand
        state["_cbt"] = None
        return state

    def train(self, X: np.ndarray) -> None:
        pass

    def codebook_bytes(self) -> int:
        return self._SEED_BYTES

    def squared_norms(self) -> np.ndarray:
        """Each codeword's squared norm in float64 (from the seed, like the
        codebook; cached)."""
        if getattr(self, "_sq_norms", None) is None:
            cb = self.codebook.astype(np.float64)
            self._sq_norms = np.einsum("ij,ij->i", cb, cb)
        return self._sq_norms

    def _nearest_codewords(self, S: np.ndarray, top: int, slice_: int = 0) -> np.ndarray:
        """(n x top) indices of the codewords of largest inner product with
        each row of S (one slice per row, slice number `slice_`), best
        first: faiss's blocked GEMM with a running top-k, rather than an
        (n x 2^bits) score matrix."""
        if slice_ == self.m - 1 and self.last_bits < self.bits:
            if self._last_index is None:
                self._last_index = faiss.IndexFlatIP(self.sub)
                self._last_index.add(self.codebook[:1 << self.last_bits])
            index = self._last_index
            top = min(top, 1 << self.last_bits)
        else:
            if self._index is None:
                self._index = faiss.IndexFlatIP(self.sub)
                self._index.add(self.codebook)
            index = self._index
        _, idx = index.search(np.ascontiguousarray(S, dtype=np.float32), top)
        return idx.astype(np.int64)

    def compute_codes(self, X: np.ndarray, fast: bool = False) -> np.ndarray:
        """Nearest-codeword codes; `fast` searches with
        scan_kernels.nearest_codeword (for planning: near-ties may resolve
        differently from faiss)."""
        X = np.ascontiguousarray(X, dtype=np.float32).reshape(len(X), self.m, self.sub)
        idx = np.empty((len(X), self.m), dtype=np.int64)
        if fast:
            from filters.cap_cover import scan_kernels
            if getattr(self, "_cbt", None) is None:
                self._cbt = np.ascontiguousarray(self.codebook.T)
            full = self.m - (1 if self.last_bits < self.bits else 0)
            out = np.empty(len(X) * full, dtype=np.int64)
            scan_kernels.nearest_codeword(np.ascontiguousarray(X[:, :full, :]).reshape(-1, self.sub), self._cbt, out)
            idx[:, :full] = out.reshape(len(X), full)
            for j in range(full, self.m):
                idx[:, j] = self._nearest_codewords(X[:, j, :], 1, j)[:, 0]
            return self._pack(idx)
        for j in range(self.m):
            idx[:, j] = self._nearest_codewords(X[:, j, :], 1, j)[:, 0]
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
        k, m, s, T = len(U), self.m, self.sub, min(candidates, 1 << self.last_bits)
        Us = np.ascontiguousarray(U, dtype=np.float32).reshape(k, m, s)
        cand = np.empty((k, m, T), dtype=np.int64)
        for j in range(m):
            cand[:, j, :] = self._nearest_codewords(Us[:, j, :], T, j)
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

    def fit_cap_code(self, u: np.ndarray, P: np.ndarray, t: np.ndarray, candidates: int = 8,
                     passes: int = 2) -> np.ndarray:
        """CapCoverFilter.fit_cap_code's search: the code (1 x code_size)
        whose decoded direction holds caps (P[i], t[i]) within the least
        angle, max_i angle(decode, P[i]) + t[i], by coordinate search over
        each slice's `candidates` codewords nearest u's, starting from u's
        own code."""
        def stored(cos):
            return (np.arccos(np.clip(cos, -1.0, 1.0)) + t[:, None]).max(axis=0)

        m, s, T = self.m, self.sub, min(candidates, 1 << self.last_bits)
        cand = np.concatenate([self._nearest_codewords(u.reshape(m, s)[j:j + 1], T, j) for j in range(m)])
        cw = self.codebook[cand].astype(np.float64)                         # (m, T, s)
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
        return self._pack(cand[np.arange(m), choice][None, :])


class _BasisPQ:
    """_SeededPQ on a direction's coordinates in a rank-r basis trained on
    the data. Embedding directions are far from isotropic (the top 64 of
    384 principal directions hold 58-72% of the energy on the five
    blocklist corpora), so spending the code's bits on r < dim coordinates
    buys a finer direction than spreading them over all dim: at 12 / 24 /
    48 bytes, 0.70-0.82 / 0.58-0.64 / 0.45-0.48 rad against seeded PQ's
    1.06 / 0.85 / 0.60 (mean angle to the decoded direction over fold 0's
    members at |B| = 8000, five corpora).

    train(): the top `rank` eigenvectors of the training rows' second
    moment (uncentered), each coded as a TQ4 row (`basis_qtype`) -- the
    basis is stored, and charged in codebook_bytes() -- and re-orthonormalized
    (QR of the decoded rows: what a decoder rebuilds from the stored codes;
    coding the rows costs <= 0.015 rad of blur against an exact basis, TQ3
    rows ~0.01 more, TQ2 ~0.05), then turned by a seeded rank x rank
    rotation so every slice carries a similar share of the energy (equal-
    weight slices of very unequal energy lose most of the gain). M (rank x
    dim, orthonormal rows) is the result: a direction's coefficients are
    D @ M.T, coded by seeded PQ, and decode to codewords @ M. Correctness
    never rests on the basis: every cap angle is measured around the
    decoded direction.

    Rows are fitted on the caps' directions (fit_on="caps", what the filter
    encodes) or on the members (fit_on="members")."""

    _SEED_BYTES = 4

    def __init__(self, dim: int, rank: int, sub: int, bits: int, seed: int = 0x5EED,
                 basis_qtype: int = faiss.ScalarQuantizer.QT_4bit_tqmse, fit_on: str = "caps",
                 last_bits: Optional[int] = None) -> None:
        if rank > dim or rank % sub:
            raise ValueError(f"basis PQ needs rank <= dim and rank % sub == 0; got dim={dim}, rank={rank}, sub={sub}")
        self.dim, self.rank, self.sub, self.bits = dim, rank, sub, bits
        self.pq = _SeededPQ(rank, sub, bits, seed, last_bits=last_bits)
        self.m, self.code_size = self.pq.m, self.pq.code_size
        self.seed = seed
        self.basis_qtype = basis_qtype
        self.fit_on = fit_on
        self.M = None

    def _basis_quantizer(self) -> faiss.ScalarQuantizer:
        sq = faiss.ScalarQuantizer(self.dim, self.basis_qtype)
        sq.train(_PROBE_RNG.random((2, self.dim)).astype(np.float32))
        return sq

    def codebook_bytes(self) -> int:
        """The basis's codes and its quantizer's table, plus two seeds (the
        PQ codebook's and the in-basis rotation's)."""
        sq = self._basis_quantizer()
        return (self.rank * int(sq.code_size) + int(faiss.vector_to_array(sq.trained).nbytes)
                + 2 * self._SEED_BYTES)

    def train(self, X: np.ndarray, members: Optional[np.ndarray] = None) -> None:
        rows = members if self.fit_on == "members" and members is not None else X
        R = np.asarray(rows, dtype=np.float64)
        _, vecs = np.linalg.eigh(R.T @ R)
        W = np.ascontiguousarray(vecs[:, ::-1][:, :self.rank].T, dtype=np.float32)
        sq = self._basis_quantizer()
        Wd = sq.decode(sq.compute_codes(W)).astype(np.float64)
        Q, _ = np.linalg.qr(Wd.T)
        rng = np.random.default_rng(self.seed + self.rank)
        q, r = np.linalg.qr(rng.standard_normal((self.rank, self.rank)))
        rot = q * np.sign(np.diag(r))[None, :]
        self.M = rot.T @ Q.T

    def _coefficients(self, D: np.ndarray) -> np.ndarray:
        return np.asarray(D, dtype=np.float64) @ self.M.T

    def compute_codes(self, X: np.ndarray, fast: bool = False) -> np.ndarray:
        return self.pq.compute_codes(self._coefficients(X), fast=fast)

    def fit_codes(self, U: np.ndarray, X: np.ndarray, labels: np.ndarray, candidates: int = 8,
                  passes: int = 2) -> np.ndarray:
        """_SeededPQ.fit_codes in the basis: M's rows are orthonormal, so
        <x, codewords @ M> = <x @ M.T, codewords> and every decoded vector
        still has the same norm."""
        return self.pq.fit_codes(self._coefficients(U), self._coefficients(X), labels, candidates, passes)

    def decode(self, codes: np.ndarray) -> np.ndarray:
        return (self.pq.decode(codes).astype(np.float64) @ self.M).astype(np.float32)

    def fit_cap_code(self, u: np.ndarray, P: np.ndarray, t: np.ndarray, candidates: int = 8,
                     passes: int = 2) -> np.ndarray:
        """_SeededPQ.fit_cap_code on the coefficients (decoded vectors keep
        a common norm, and <p, c @ M> = <p @ M.T, c>)."""
        return self.pq.fit_cap_code(self._coefficients(u[None, :])[0], self._coefficients(P), t, candidates, passes)


class _BootPQ(_BasisPQ):
    """_BasisPQ whose basis is not stored: the decoder rebuilds it from the
    decoded directions of the cover's caps at whole-space encodings (the
    anchors, stored first), so it costs only seeds. Basis: the top `rank`
    principal subspace of the anchors (uncentered second moment), spanned
    by the projection of a seeded Gaussian dim x rank matrix onto it, the
    projection rounded to fp16 and orthonormalized (QR). It depends on the
    subspace only, not on which eigenvectors an eigensolver returns inside
    it, and the fp16 rounding absorbs solver round-off, so a rebuild on the
    stored codes gives the same basis; a residual difference of an fp16 ulp
    moves decoded directions by ~1e-4 rad, inside the accept rule's
    _COVERAGE_EPSILON.

    Isotropic coding noise in the anchors leaves the principal subspace in
    place, but a subspace estimated from n anchors is only as good as n:
    measured on finished covers (blur of the cap directions, 48 B), 500
    whole-space anchors gave no gain over seeded PQ and 2,000 gained
    0.07 rad (suggestions/cap_cover_volume_scripts/boot_basis.py).

    Measured (suggestions/cap_cover_volume_report.md, idea 1): with every
    cap as an anchor (no decoder has that) the free plan's volume at
    |B| = 8000, k = 5% drops 13-20 orders; with honest anchors -- the 25%
    of caps with most slack, fixed at whole-space codes -- 1-6 orders at
    8K and 5-16 at 80K under the free plan, 0-3.5 under coarsen's ceiling,
    on a fixed cover. In the filter (boot levels at the finished cover's
    plan, results/cap_cover/boot8_*, beq8_*.csv), |B| = 8000: FPR +0.036 /
    +0.044 at k = 1 / 0.5% at equal volume (equal-bytes boot codes) and
    +0.076 / +0.087 when cheaper boot codes are allowed: a boot code's
    error lies inside the data's subspace, where queries are. Not in any
    default; BOOT_CANDIDATES adds it."""

    def codebook_bytes(self) -> int:
        """Two seeds (the PQ codebook's and the projected Gaussian's)."""
        return 2 * self._SEED_BYTES

    def train(self, anchors: np.ndarray, members: Optional[np.ndarray] = None) -> None:
        A = np.asarray(anchors, dtype=np.float64)
        _, vecs = np.linalg.eigh(A.T @ A)
        V = vecs[:, ::-1][:, :self.rank]
        G = np.random.default_rng(self.seed + self.rank).standard_normal((self.dim, self.rank))
        B = (V @ (V.T @ G)).astype(np.float16).astype(np.float64)
        Q, R = np.linalg.qr(B)
        self.M = (Q * np.sign(np.diag(R))[None, :]).T


class _DualPQ:
    """Per cap, the better of two codes with one layout: _BasisPQ's (a
    basis trained on the caps) and a whole-space _SeededPQ's (dim / m dims
    per slice), both m slices of `bits` bits. Each family's last slice draws from the first half of
    its codebook, so the code's top bit says which family it is: no byte
    is added.

    The trained basis codes most directions far more finely than a
    whole-space code of the same size, but a few directions with much of
    their energy outside it more coarsely; an encoding whose accept region
    is summed over caps at d = 384 is set by its widest caps, so neither
    family alone keeps both the typical and the worst cap tight. Median /
    99th-percentile / max angle to the decoded direction over fold 0's
    distinct members at |B| = 8000 (Quora, WildChat, CC-News), 48 B: basis
    .45-.48 / .57-.60 / .67-.79, whole-space .60 / .63 / .64-.65, the
    better of the two .45-.48 / .57-.59 / .62-.63 (the basis is the worse
    for 0.4-1.2% of directions); 24 B: the better of the two .60-.63 /
    .77-.80 / .87-.88 against whole-space .85-.86 / .88-.89 / .90-.91;
    12 B: .74-.79 / 1.00-1.04 / 1.07-1.08 against 1.06 / 1.09 / 1.10-1.12.

    compute_codes() keeps the code whose decoded direction is nearer;
    fit_codes() and fit_cap_code() fit both families and keep the one that
    holds the cap's members (or parts) within the smaller angle."""

    def __init__(self, dim: int, rank: int, sub: int, bits: int, fit_on: str = "caps") -> None:
        self.basis = _BasisPQ(dim, rank, sub, bits, fit_on=fit_on, last_bits=bits - 1)
        self.full = _SeededPQ(dim, dim // self.basis.m, bits, last_bits=bits - 1)
        if self.basis.m != self.full.m or (self.full.m * bits) % 8:
            raise ValueError("dual PQ needs both families on the same whole-byte layout")
        self.dim, self.bits, self.m = dim, bits, self.full.m
        self.code_size = self.full.code_size

    def codebook_bytes(self) -> int:
        return self.basis.codebook_bytes() + self.full.codebook_bytes()

    def train(self, X: np.ndarray, members: Optional[np.ndarray] = None) -> None:
        self.basis.train(X, members)

    def _split(self, codes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(codes without the flag, whether each is whole-space)."""
        codes = np.array(codes[:, :self.code_size], dtype=np.uint8)
        full = (codes[:, -1] >> 7).astype(bool)
        codes[:, -1] &= 0x7F
        return codes, full

    def _join(self, basis_codes: np.ndarray, full_codes: np.ndarray, use_full: np.ndarray) -> np.ndarray:
        codes = np.where(use_full[:, None], full_codes, basis_codes).astype(np.uint8)
        codes[use_full, -1] |= 0x80
        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        codes, full = self._split(codes)
        out = np.empty((len(codes), self.dim), dtype=np.float32)
        if (~full).any():
            out[~full] = self.basis.decode(codes[~full])
        if full.any():
            out[full] = self.full.decode(codes[full])
        return out

    def compute_codes(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        cb, cf = self.basis.compute_codes(X), self.full.compute_codes(X)
        eb = angle_between(X, unit_rows(self.basis.decode(cb)))
        ef = angle_between(X, unit_rows(self.full.decode(cf)))
        return self._join(cb, cf, ef < eb)

    def fit_codes(self, U: np.ndarray, X: np.ndarray, labels: np.ndarray, candidates: int = 8,
                  passes: int = 2) -> np.ndarray:
        cb = self.basis.fit_codes(U, X, labels, candidates, passes)
        cf = self.full.fit_codes(U, X, labels, candidates, passes)
        labels = np.asarray(labels, dtype=np.int64)
        order, starts, counts = segments(labels, len(U))
        widest = []
        for codes, q in ((cb, self.basis), (cf, self.full)):
            D = unit_rows(q.decode(codes))
            held = segment_max(angle_between(X, D[labels])[order], starts, counts)
            # A cap with no member: its own direction.
            widest.append(np.where(counts > 0, held, angle_between(U, D)))
        return self._join(cb, cf, widest[1] < widest[0])

    def fit_cap_code(self, u: np.ndarray, P: np.ndarray, t: np.ndarray, candidates: int = 8,
                     passes: int = 2) -> np.ndarray:
        cb = self.basis.fit_cap_code(u, P, t, candidates, passes)
        cf = self.full.fit_cap_code(u, P, t, candidates, passes)
        reach = [float((angle_between(unit_rows(q.decode(c)), P) + t).max())
                 for c, q in ((cb, self.basis), (cf, self.full))]
        return self._join(cb, cf, np.array([reach[1] < reach[0]]))


_LLOYD_MAX = {}


def _gaussian_lloyd_max(bits: int) -> np.ndarray:
    """The 2^bits Lloyd-Max levels of N(0, 1), ascending: data-independent
    constants (computed by integrating the density on a fine grid), so a
    filter stores only each coordinate's mean and spread."""
    if bits not in _LLOYD_MAX:
        g = np.linspace(-9.0, 9.0, 72001)
        w = np.exp(-0.5 * g * g)
        k = 1 << bits
        cdf = np.cumsum(w) / w.sum()
        lv = np.interp((np.arange(k) + 0.5) / k, cdf, g)
        for _ in range(500):
            idx = np.searchsorted(0.5 * (lv[1:] + lv[:-1]), g)
            lv = np.bincount(idx, w * g, k) / np.bincount(idx, w, k)
        _LLOYD_MAX[bits] = lv
    return _LLOYD_MAX[bits]


class _KLTSQ:
    """VA+-style direction code (Ferhatosmanoglu et al.'s VA+-file): the
    KLT of the training rows (centered PCA), the code's bits spread over
    the top `rank` KLT coordinates by greedy reverse water-filling on their
    variances (each next bit to the coordinate whose Lloyd-Max error falls
    most; most coordinates get none -- the dimension reduction), and each
    coded coordinate scalar-quantized to Gaussian Lloyd-Max levels scaled
    by its own mean and spread. Decode = mean + sum of level x KLT row.

    Stored, and charged in codebook_bytes(): the mean (fp16), the `rank`
    KLT rows (TQ4, re-orthonormalized by QR on decode as _BasisPQ's), each
    row's coefficient mean and spread (fp16) and bit count (4 bits). Mean
    angle to the decoded direction over fold 0's distinct members at
    |B| = 8000 (Quora, WildChat, CC-News; exact KLT, uncapped rank): 12 B
    .69-.77, 24 B .55-.60, 48 B .38-.41 rad, against seeded PQ's
    1.06 / .85 / .60, _BasisPQ's .72-.79 / .60-.63 / .46-.48, and plain
    truncation (top-r KLT coordinates as int8) .87-1.03 / .76-.90 /
    .62-.74. The bits went to 73-77 / 132-137 / 221-223 coordinates, at
    most 3-5 bits each."""

    _SEED_BYTES = 0

    def __init__(self, dim: int, total_bits: int, rank: int, max_bits: int = 8,
                 basis_qtype: int = faiss.ScalarQuantizer.QT_4bit_tqmse, fit_on: str = "caps") -> None:
        if total_bits % 8:
            raise ValueError("KLT code needs a whole number of bytes")
        self.dim, self.total_bits, self.rank, self.max_bits = dim, total_bits, min(rank, dim), max_bits
        self.code_size = total_bits // 8
        self.basis_qtype = basis_qtype
        self.fit_on = fit_on
        self.M = None

    def _basis_quantizer(self) -> faiss.ScalarQuantizer:
        sq = faiss.ScalarQuantizer(self.dim, self.basis_qtype)
        sq.train(_PROBE_RNG.random((2, self.dim)).astype(np.float32))
        return sq

    def codebook_bytes(self) -> int:
        sq = self._basis_quantizer()
        return (2 * self.dim + self.rank * (int(sq.code_size) + 4) + (self.rank + 1) // 2
                + int(faiss.vector_to_array(sq.trained).nbytes))

    def train(self, X: np.ndarray, members: Optional[np.ndarray] = None) -> None:
        rows = members if self.fit_on == "members" and members is not None else X
        R = np.asarray(rows, dtype=np.float64)
        self.mu = R.mean(axis=0).astype(np.float16).astype(np.float64)
        C = R - self.mu
        _, vecs = np.linalg.eigh(C.T @ C)
        W = np.ascontiguousarray(vecs[:, ::-1][:, :self.rank].T, dtype=np.float32)
        sq = self._basis_quantizer()
        Q, _ = np.linalg.qr(sq.decode(sq.compute_codes(W)).astype(np.float64).T)
        self.M = Q.T
        coef = C @ self.M.T
        self.mean = coef.mean(axis=0).astype(np.float16).astype(np.float64)
        self.spread = np.maximum(coef.std(axis=0), 1e-6).astype(np.float16).astype(np.float64)
        # Greedy reverse water-filling: each bit to the coordinate whose
        # Lloyd-Max error drops most (the error of b bits on N(0, s^2) is
        # s^2 d(b), d computed once from the levels).
        dist = [1.0] + [self._lm_error(b) for b in range(1, self.max_bits + 1)] + [0.0]
        bits = np.zeros(self.rank, dtype=np.int64)
        gain = self.spread ** 2 * (dist[0] - dist[1])
        for _ in range(self.total_bits):
            i = int(np.argmax(gain))
            bits[i] += 1
            gain[i] = self.spread[i] ** 2 * (dist[bits[i]] - dist[bits[i] + 1]) if bits[i] < self.max_bits else -np.inf
        self.bits = bits
        self.coded = np.flatnonzero(bits)
        self.levels = [self.mean[i] + self.spread[i] * _gaussian_lloyd_max(int(bits[i])) for i in self.coded]

    @staticmethod
    def _lm_error(bits: int) -> float:
        """Mean squared error of the Lloyd-Max levels on N(0, 1)."""
        if ("error", bits) in _LLOYD_MAX:
            return _LLOYD_MAX["error", bits]
        g = np.linspace(-9.0, 9.0, 72001)
        w = np.exp(-0.5 * g * g)
        lv = _gaussian_lloyd_max(bits)
        q = lv[np.searchsorted(0.5 * (lv[1:] + lv[:-1]), g)]
        _LLOYD_MAX["error", bits] = float(np.sum(w * (g - q) ** 2) / w.sum())
        return _LLOYD_MAX["error", bits]

    def _indices(self, D: np.ndarray) -> np.ndarray:
        c = (np.asarray(D, dtype=np.float64) - self.mu) @ self.M.T
        idx = np.empty((len(c), len(self.coded)), dtype=np.int64)
        for t, i in enumerate(self.coded):
            lv = self.levels[t]
            idx[:, t] = np.searchsorted(0.5 * (lv[1:] + lv[:-1]), c[:, i])
        return idx

    def _pack(self, idx: np.ndarray) -> np.ndarray:
        b = self.bits[self.coded]
        cols = [((idx[:, t:t + 1] >> np.arange(b[t])[None, :]) & 1) for t in range(len(b))]
        allbits = np.concatenate(cols, axis=1).astype(np.uint8)
        return np.packbits(allbits, axis=1, bitorder="little")[:, :self.code_size]

    def _unpack(self, codes: np.ndarray) -> np.ndarray:
        b = self.bits[self.coded]
        allbits = np.unpackbits(np.ascontiguousarray(codes[:, :self.code_size], dtype=np.uint8), axis=1,
                                count=int(b.sum()), bitorder="little").astype(np.int64)
        idx = np.empty((len(codes), len(b)), dtype=np.int64)
        pos = 0
        for t in range(len(b)):
            idx[:, t] = (allbits[:, pos:pos + b[t]] << np.arange(b[t])[None, :]).sum(axis=1)
            pos += b[t]
        return idx

    def _values(self, idx: np.ndarray) -> np.ndarray:
        return np.stack([self.levels[t][idx[:, t]] for t in range(len(self.coded))], axis=1)

    def compute_codes(self, X: np.ndarray) -> np.ndarray:
        return self._pack(self._indices(X))

    def decode(self, codes: np.ndarray) -> np.ndarray:
        vals = self._values(self._unpack(codes))
        coef = np.tile(self.mean, (len(codes), 1))
        coef[:, self.coded] = vals
        return (self.mu + coef @ self.M).astype(np.float32)

    def fit_codes(self, U: np.ndarray, X: np.ndarray, labels: np.ndarray, candidates: int = 8,
                  passes: int = 2) -> np.ndarray:
        """_fit_scalar_codes for the KLT code: starting from U's own levels,
        each coded coordinate in turn moves one level up or down where that
        raises the least cosine between a cap's members and its decoded
        direction (the KLT rows are orthonormal, so each move changes the
        decoded vector along one row)."""
        idx = self._indices(U)
        n = len(X)
        if n == 0 or len(U) == 0:
            return self._pack(idx)
        k, c = idx.shape
        coef = np.tile(self.mean, (k, 1))
        coef[:, self.coded] = self._values(idx)
        D = self.mu + coef @ self.M
        labels = np.asarray(labels, dtype=np.int64)
        order = np.argsort(labels, kind="stable")
        caps, starts = np.unique(labels[order], return_index=True)
        Xo, lo = np.asarray(X, dtype=np.float64)[order], labels[order]
        P = Xo @ self.M[self.coded].T                     # members' coordinates on the coded rows
        dot = np.einsum("nd,nd->n", Xo, D[lo])
        norm2 = np.einsum("kd,kd->k", D, D)
        along = D @ self.M[self.coded].T                  # each decoded vector's coordinate on each row
        for _ in range(passes):
            for t in range(c):
                lv = self.levels[t]
                cur = np.minimum.reduceat(dot / np.sqrt(norm2[lo]), starts)
                best, best_step = cur, np.zeros(len(caps), dtype=np.int64)
                for step in (-1, 1):
                    j = np.clip(idx[caps, t] + step, 0, len(lv) - 1)
                    delta = lv[j] - lv[idx[caps, t]]
                    full = np.zeros(k)
                    full[caps] = delta
                    trial = (dot + P[:, t] * full[lo]) / np.sqrt((norm2 + full * (2.0 * along[:, t] + full))[lo])
                    val = np.minimum.reduceat(trial, starts)
                    better = (val > best) & (delta != 0)
                    best = np.where(better, val, best)
                    best_step = np.where(better, step, best_step)
                moved = caps[best_step != 0]
                if len(moved):
                    j = np.clip(idx[moved, t] + best_step[best_step != 0], 0, len(lv) - 1)
                    full = np.zeros(k)
                    full[moved] = lv[j] - lv[idx[moved, t]]
                    dot += P[:, t] * full[lo]
                    norm2 += full * (2.0 * along[:, t] + full)
                    along[:, t] += full
                    idx[moved, t] = j
        return self._pack(idx)


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


# Set storage (CapCoverFilter's set_codes). A static cover is a set of caps,
# so storing a group of n caps in a fixed order spends log2(n!) bits on the
# order alone. Each group is instead sorted by the first w bits of its codes
# (w = 8, 16 or 24, whichever saves most), and those prefixes are stored as
# an Elias-Fano sequence: n (l + 1) + (2^w >> l) + 1 bits, l = floor(log2(2^w
# / n)), about log2(e 2^w / n) + 0.6 per cap instead of w -- log2 n - 2 bits
# saved per cap. The rest of each code is stored as it was, byte-aligned.

def _ef_layout(n: int, w: int) -> Tuple[int, int, int]:
    """(l, low bytes, high bytes) of an Elias-Fano sequence of n values
    below 2^w (scan_kernels.ef_decode's layout)."""
    universe = 1 << w
    l = 0
    while n << (l + 1) <= universe:
        l += 1
    return l, (n * l + 7) // 8, (n + ((universe - 1) >> l) + 1 + 7) // 8


def _key_width(n: int, code_size: int) -> int:
    """Bits of each code (0, 8, 16 or 24) a group of n codes of code_size
    bytes keeps as an Elias-Fano set: the width that saves most bytes, 0 if
    none saves any."""
    best, best_w = 0, 0
    for w in (8, 16, 24):
        if w > 8 * code_size:
            break
        _, low, high = _ef_layout(n, w)
        save = n * (w // 8) - low - high
        if save > best:
            best, best_w = save, w
    return best_w


def _ef_encode(keys: np.ndarray, w: int) -> Tuple[int, np.ndarray, np.ndarray]:
    """(l, low, high) for non-decreasing keys below 2^w: little-endian bit
    streams, low = each key's low l bits, high = a set bit at
    (key >> l) + i for the i-th key."""
    keys = np.asarray(keys, dtype=np.int64)
    n = len(keys)
    l, low_bytes, high_bytes = _ef_layout(n, w)
    low = np.zeros(low_bytes, dtype=np.uint8)
    if l:
        bits = ((keys[:, None] >> np.arange(l)[None, :]) & 1).astype(np.uint8).ravel()
        low[:] = np.packbits(np.concatenate([bits, np.zeros(8 * low_bytes - len(bits), np.uint8)]),
                             bitorder="little")
    high_bits = np.zeros(8 * high_bytes, dtype=np.uint8)
    high_bits[(keys >> l) + np.arange(n)] = 1
    return l, low, np.packbits(high_bits, bitorder="little")


@dataclass
class _Group:
    """One static encoding group in set storage: caps `ids` (sorted by their
    codes' first w bits), those prefixes as an Elias-Fano sequence (l, low,
    high), the rest of each code (rem) and each cap's angle code."""
    ids: np.ndarray
    w: int
    l: int
    low: np.ndarray
    high: np.ndarray
    rem: np.ndarray
    ang: np.ndarray

    def keys(self) -> np.ndarray:
        from filters.cap_cover import scan_kernels
        out = np.empty(len(self.ids), dtype=np.int64)
        if self.w:
            scan_kernels.ef_decode(self.low, self.high, len(self.ids), self.l, out)
        return out

    def codes(self) -> np.ndarray:
        """The full codes, in group order."""
        k = self.keys()
        prefix = ((k[:, None] >> (8 * np.arange(self.w // 8))[None, :]) & 0xFF).astype(np.uint8)
        return np.ascontiguousarray(np.concatenate([prefix, self.rem], axis=1))

def _fits(dim: int, q: Quantize) -> bool:
    """Whether encoding q can code directions of `dim` dimensions."""
    spec = _ENCODINGS[q]
    if spec.klt_bits:
        return spec.rank <= dim
    if spec.rank:
        return spec.rank <= dim and spec.rank % spec.sub == 0 and not (spec.dual and dim % (spec.rank // spec.sub))
    return dim % max(spec.sub, 1) == 0


# Candidate encodings CapCoverFilter.auto() chooses among: the seeded-PQ rungs
# below 1 bit/dim plus TurboQuant above it, so small budgets can buy many
# coarse caps and large ones fewer, finer caps. Coarser rungs (6 and 9 B,
# ~1.2 and 1.12 rad) measured worse: the reach proxy over-rates them once
# the encoding error rivals the caps' own angles (+0.16 FPR at k = 0.5%).
# Encodings with a trained basis (PSPQ*, DPQ*, KSQ*) are left out: at
# |B| = 8000 their basis takes most of a small budget and mixing never
# picked them at k <= 1% (v2_*.csv), and at 80K they double construction
# time; KSQ_CANDIDATES adds the VA+-style KLT codes for budgets that hold
# a basis (see CapCoverFilter's Encodings trained on B).
AUTO_CANDIDATES = (Quantize.SPQ12, Quantize.SPQ16, Quantize.SPQ24, Quantize.SPQ36, Quantize.SPQ48,
                   Quantize.SPQ64, Quantize.TQ2, Quantize.TQ3, Quantize.TQ4)
KSQ_CANDIDATES = AUTO_CANDIDATES + (Quantize.KSQ6, Quantize.KSQ12, Quantize.KSQ24, Quantize.KSQ48)
# 10- and 12-bit seeded-PQ rungs between AUTO_CANDIDATES' (2026-10-04): with
# them a cap can coarsen, or the top band refine (mix_finer), by a smaller
# step. Every new rung is used. |B| = 8000, 175 cells, fold-paired against
# AUTO_CANDIDATES (no set storage; lad_*.csv): FPR -0.014 / -0.006 / -0.006
# at k = 0.5/1/2% of |B| (+-0.002 above), log10 volume -0.3 / -0.6 / -1.2
# (+0.9 at 20%), for 1.5-2x the construction time (encoding 10/12-bit
# codes). With mix_finer=1 and set storage see CapCoverFilter's Choosing the
# encoding. 8-bit rungs at the same byte sizes (SPQ24B8, SPQ32B8, SPQ48B8;
# 0.03 rad more blur than 12-bit at equal bytes, but scored by table)
# measured volume- and FPR-neutral in the mix (+-0.003 FPR, +-0.6 orders).
DENSE_LADDER = (Quantize.SPQ18, Quantize.SPQ20, Quantize.SPQ30, Quantize.SPQ40, Quantize.SPQ60, Quantize.SPQ72,
                Quantize.SPQ96)
# The dense rungs at 10 bits only (auto()'s default since 2026-10-04, with
# mix_finer=1): a 2^bits codebook costs dim * 2^bits multiply-adds per
# encoded cap, so these five cost about one 12-bit rung's. Against
# DENSE_LADDER (both with mix_finer=1 and set storage, d10f1_*, cdf1_*.csv):
# within +-0.002 FPR and +-0.4 orders at k >= 2% of |B| = 8000, +0.005 /
# +0.002 FPR at k = 0.5 / 1%; at |B| = 80K every job fits the benchmark's
# 5-minute limit (DENSE_LADDER: 9 of 35 do not).
DENSE10_LADDER = (Quantize.SPQ20, Quantize.SPQ30, Quantize.SPQ40, Quantize.SPQ60, Quantize.SPQ80)

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
    AUTO_CANDIDATES + DENSE10_LADDER), and construct() builds covers at the candidates' cap
    counts from one merge (ConstructInterface.iter_candidates) and keeps
    the best by `selection`:

    - "volume" (default): the least accept region on the sphere, ln of the
      sum over caps of sigma(theta' + beta) at selection_tau (theta' the
      members' widest angle around the decoded direction) -- from B and
      the stored caps only, so it holds whatever the queries (_volume). At
      d = 384 the widest caps dominate the sum. With `mixed` (static
      filters) each cap takes its own encoding (_choose_mixed): "coarsen"
      (default) lets caps narrower than the widest take coarser codes than
      the least-volume single encoding where that keeps them within the
      widest angle, buying more caps with the bytes freed; "free" also
      lets the widest caps take finer codes, which shrinks the region most
      but spends bytes on few caps; False keeps one encoding (_choose).
    - "proxy": sum over caps of phi^p, the in-data reach (p =
      selection_exponent): assumes queries lie near the data.
    - "holdout": the proxy with a veto by held-out members used as pseudo-
      queries (_choose_holdout); fits the choice to query-like data, so it
      is kept only to reproduce earlier rows.

    Measured (src/compare_cap_cover.py, |B| = 8000, 5 corpora x 5 folds,
    fold-paired; results/cap_cover/v1_*.csv, v5_*.csv): against the reach
    proxy, the least-volume single encoding (mixed=False) costs +0.064 /
    +0.068 / +0.024 / +0.034 FPR at k = 0.5/1/2/5% of |B| (within +0.003
    at 10-20%) for 3.6 / 6.6 / 6.5 / 6.3 orders less log10 volume (4.6-5.2
    at 10-20%). Against that single encoding, mixed="coarsen" moves FPR by
    -0.058 / -0.056 / -0.035 / -0.026 / -0.002 / +0.001 / -0.001 at
    k = 0.5/1/2/5/10/15/20% at about its volume (+0.8 / +0.5 / -0.5 / -0.6
    / +0.4 / +1.0 / -1.2 orders), for 1.6-2.7x its construction time;
    mixed="free" by -0.017 / -0.007 / -0.002 / +0.004 / +0.020 / +0.001 /
    +0.001 for 1.5 / 2.6 / 4.3 / 9.0 / 12.0 / 10.0 / 10.7 orders less
    volume (CC-News at 10%: +0.10, its widest caps take TQ4/TQ3 codes and
    the cover keeps 2.3K caps instead of 4.2K). Mixing must plan the
    finished cover again: carrying the plan over finish(), which moves
    members between caps, left the coarsened covers 0.4-1.2 orders wider.

    The top band. At d = 384 the region is set by the caps within ~0.02
    rad of the widest (often 10-300 of them), so `mix_finer` raises
    coarsen's ceiling by that many rungs: a plan then gives a finer code
    only to a cap that would otherwise exceed the widest angle it keeps.
    With DENSE_LADDER's smaller steps and set storage (cdf1_*.csv, 175
    cells, fold-paired against the v5 default): FPR -0.020 / -0.013 /
    -0.004 / -0.002 / -0.002 / -0.005 / -0.002 and log10 volume -0.8 /
    -1.4 / -3.0 / -4.7 / -11.5 / -9.6 / -10.2 at k = 0.5/1/2/5/10/15/20%
    of |B| (smaller in 18-25 of 25 cells per k) -- at 10-20% the volume of
    mixed="free" (-0.8 .. +0.7 orders from it) at better FPR than coarsen.
    On the default ladder (band_*.csv, no set storage) mix_finer=1 trades:
    -1.1 .. -9.9 orders for +0.006 .. +0.017 FPR at k <= 5%. Two other
    top-band moves measured no room (suggestions/cap_cover_volume_report.md):
    planning on fitted codes and tight angles (the plan's widest angle is
    within 0.001-0.012 rad of the stored one), and bisecting the widest
    caps paid for by merging narrow ones (the cheapest merges of non-band
    caps land above the band floor in 62-100% of cases at k <= 2%; the k
    grid already undoes the last, widest merges).

    The reach proxy errs coarse at small budgets: against the best fixed
    encoding per cell (|B| = 8000, folds 0-2) it is +0.015 / +0.014 /
    +0.008 FPR at k = 1/2/5% of |B|, within +-0.003 elsewhere; the
    held-out veto recovered -0.003 / -0.008 / -0.015 / -0.006 at
    k = 0.5/1/2/5% for ~1.7x the construction time (veto2_*.csv). A proxy
    over member angles did 5x worse than the reach proxy; sub-12-byte rungs
    (SPQ6, SPQ9) win only 3 of 30 cells at k <= 1% even in hindsight.

    Angles are stored in 8 bits over the filter's own [min, max] when no
    insert strategy can grow one past max (else 16 bits over [0, pi]).

    Set storage (set_codes, static filters; see _ef_layout and _pack). A
    static cover is a set, so each encoding group is stored sorted by its
    codes' first 8-24 bits, those as an Elias-Fano sequence: log2(n) - 2
    bits saved per cap of a group of n, 1.7-4.6% of the budget at k <= 5%
    of |B| = 8000 (0.7-3.5% above; fold 0's v5 covers). The scan decodes
    the sequence once per call and the kernels take each code's first bits
    from it, so what is charged is what is read. Every byte-level step
    (plan, cap counts, get_bytesize) charges the exact Elias-Fano size.
    |B| = 8000, 175 cells, fold-paired against fixed-order groups (v5_*,
    set_*.csv): FPR -0.016 / -0.011 / -0.004 / -0.004 / -0.001 / 0.000 /
    0.000 at k = 0.5/1/2/5/10/15/20% (the bytes buy caps), log10 volume
    +0.2 .. -1.0; set_codes=False reproduces the earlier rows exactly.

    Measured (src/compare_cap_cover.py, blocklist workload, 5 corpora x 5
    folds, tau = 0.1, equal bytes = Ball-Cover TQ4 at k = k_fraction * |B|
    balls; suggestions/cap_cover_volume_report.md). auto()'s defaults since
    2026-10-04 (DENSE10_LADDER, mix_finer=1, set storage; d10f1_*.csv),
    |B| = 8000: mean FPR .603/.404/.289/.178/.121/.097/.085 at k = 0.5/1/2/
    5/10/15/20%, log10 volume -6.8/-13.4/-25.1/-50.2/-89.7/-117.4/-137.1;
    fold-paired against the previous defaults (v5_*.csv) FPR -0.016 /
    -0.011 / -0.003 / -0.002 / -0.001 / -0.005 / -0.002 and volume -0.8 /
    -1.3 / -3.0 / -4.7 / -11.2 / -9.6 / -10.2 orders. Against the best of
    five Ball-Cover configurations *per cell* (sphere_merge at TQ1-TQ4,
    which weights its merges by member counts, and the paper's
    minimax_sphere at TQ4; ballvol_*.csv) .927/.705/.445/.272/.147/.111/.097
    it wins 25/25/25/25/21/21/20 of 25 cells (losses <= 0.030), against the
    paper's configuration (.970/.893/.740/.469/.312/.237/.188) all 175, and
    its volume is below every Ball configuration's (best: -2.4/-5.6/-11.5/
    -27.1/-50.3/-82.9/-94.0) in all 175. |B| = 80K (fold 0, d1080_*.csv):
    FPR .725/.561/.419/.243/.166/.142/.114, volume -12.1/-18.7/-32.0/-55.8/
    -95.1/-121.0/-136.5; against the previous defaults (m80_*, m80c_*.csv)
    FPR -0.011/-0.018/+0.004/-0.009/-0.003/+0.006/-0.003 and volume -1.7 /
    -1.3 / -4.7 / -6.2 / -11.2 / -10.3 / -10.3 (smaller in 33 of 35);
    Ball-Cover's paper configuration .963/.896/.782 at -5.2/-8.8/-13.5 for
    k <= 2% (ball80vol_*.csv; its larger k and every sphere_merge
    configuration exceed the 5-minute job limit on this machine).
    Construction: 3.9-24.7 s at |B| = 8000 (median 10.7; the cells of
    timing_v5_10k.csv, one P-core, timing_s4_10k.csv), 75-103 s at 80K
    alone, up to ~290 s with five 80K jobs at once.

    The previous defaults (AUTO_CANDIDATES, mix_finer=0, fixed-order groups;
    v5_*.csv, m80*_*.csv) had FPR .619/.414/.292/.180/.122/.102/.087 at
    log10 volume -6.0/-12.0/-22.0/-45.5/-78.6/-107.9/-126.9 (|B| = 8000)
    and .736/.578/.415/.252/.169/.136/.117 at -10.3/-17.4/-27.3/-49.6/
    -83.9/-110.8/-126.3 (80K); the reach proxy, then its held-out veto,
    before that: .612/.402/.303/.172/.121/.098/.087 and .609/.394/.289/
    .166/.121/.098/.087 at 3-7 orders more volume. With the KLT codes as
    candidates (KSQ_CANDIDATES) FPR was .657 at 80K, k = 1% (worse: the
    basis takes budget).

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
    (_scan_tables, where _tables_pay; 3.2x on SPQ16). Batches of 1-3
    queries on 12-bit seeded PQ skip both: each cap's slices are scored
    straight from the codebook and a cap is given up once the rest of the
    code cannot reach its threshold (scan_kernels.accept_direct; single
    queries 2.5-3.5x: 2.0K -> 7.1K q/s on 635 SPQ48 caps, 0.3K -> 0.9K on
    7.9K SPQ36, |B| = 8000, one P-core); TurboQuant codes likewise sum
    q . raw and |raw|^2 in one pass over the codes (accept_direct_levels;
    single queries 1.7-2.4x on 1.6K-3.2K TQ2/TQ3 caps, with other jobs
    running). Since 2026-10-04 batches of 1-3 queries go through per-query
    tables instead (single_tables): seeded PQ through a table of each slice
    against each codeword, m reads per cap with an exact early abandon on
    the remaining slices' best entries (scan_kernels.accept_table1; always
    for 8-bit codebooks, for 10/12-bit ones from _TABLE1_MIN_CAPS caps --
    below that the table's 2^bits * dim costs more than it saves, 3.7x
    slower on 1K 12-bit caps), TurboQuant through tables of 2-8 dimensions
    per entry (scan_kernels.accept_lut_levels). One P-core, the same filter
    (kernel_timing.py): single queries 1.6x on 7.9K SPQ36 caps (1.0K ->
    1.6K q/s), 1.75x on 1.7K TQ2+SPQ64 caps (1.5K -> 2.6K), unchanged on
    covers below the 12-bit threshold. Against the previous defaults' rows
    (timing_v5_*.csv vs timing_s4_*.csv, the same cells and core), auto()
    now answers batches of 32 at 1.15x (0.76-1.36x; slower where it holds
    more caps) and single queries at 1.28x (0.86-3.5x), median.
    Exact pruning (cell bounds,
    partial dot products) cannot help much here: a coarse cap's decoded
    direction sits up to ~1 rad off its members, so caps that neighbour in
    the data do not neighbour in decoded space; storing caps widest first
    (BallCoverFilter's 1.16-1.48x) measured flat, as the scan decides a
    block of 1024 caps per GEMM.

    Encodings trained on B (not in AUTO_CANDIDATES; pass `candidates` to
    auto()). Embedding directions are far from isotropic, so a basis
    fitted to the caps codes most of them more finely at equal bytes:
    seeded PQ in a rank-r principal basis (PSPQ*, _BasisPQ), the better of
    that and whole-space seeded PQ per cap (DPQ*, _DualPQ), or VA+'s KLT
    code -- bits by reverse water-filling over the KLT coordinates, scalar
    Lloyd-Max levels on each (KSQ*, _KLTSQ). Each charges its basis (r TQ4
    rows, 9-44 KB at d = 384). Median accept angles drop 0.1-0.3 rad, but
    (a) at |B| = 8000 the basis takes most of a budget at k <= 2% of |B|,
    (b) a few directions with energy outside the basis code worse, so the
    widest caps -- which set the accept volume at d = 384 -- often widen,
    and (c) the coding error now lies inside the data's subspace, where
    queries are, so a cap reaches further along the data than its angle
    suggests. Fold 0, Quora, |B| = 80K, against whole-space seeded PQ at
    the same code bytes: k = 5%, KSQ48 FPR .084 vs .096 (log10 volume -43
    vs -55), DPQ48 .098 (-59), KSQ24 .143 vs .167 (-20 vs -28), DPQ24 .152
    (-30); k = 1%, KSQ24 .417 vs .378, DPQ12 .514 vs .414 (-9.7 vs -8.0).
    Per-cap mixing (see Choosing the encoding) can take a KLT code for
    the caps it suits and seeded PQ for the rest."""

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
        selection: str = "volume",
        holdout_fraction: float = 0.25,
        holdout_queries: int = 2000,
        holdout_z: float = 3.0,
        basis_fit: str = "caps",
        mixed: Union[bool, str] = "coarsen",
        mix_finer: int = 0,
        set_codes: bool = True,
        single_tables: bool = True,
        boot_anchors: float = 0.25,
        boot_equal: bool = True,
    ) -> None:
        super().__init__(dim, max_memory=max_memory)
        self.boot_anchors = boot_anchors
        self.boot_equal = boot_equal
        self.basis_fit = basis_fit
        self.mix_finer = mix_finer
        self.set_codes = set_codes
        self.single_tables = single_tables
        self._packed = None    # level -> _Group, a static filter's set storage (see _pack)
        if mixed not in (False, True, "coarsen", "free"):
            raise ValueError(f"mixed must be False, 'coarsen' or 'free', got {mixed!r}")
        self.mixed = "coarsen" if mixed is True else mixed
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
        if selection not in ("volume", "proxy", "holdout"):
            raise ValueError(f"selection must be 'volume', 'proxy' or 'holdout', got {selection!r}")
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
        """A filter that picks its encodings for the budget at construct()
        (by `selection`, see _choose_mixed) among AUTO_CANDIDATES plus
        DENSE10_LADDER, with the top band one rung finer (mix_finer=1): the
        defaults since 2026-10-04 (see Choosing the encoding and The top
        band). candidates=AUTO_CANDIDATES, mix_finer=0 gives the earlier
        ones."""
        candidates = [q for q in kwargs.pop("candidates", AUTO_CANDIDATES + DENSE10_LADDER) if _fits(dim, q)]
        kwargs.setdefault("mix_finer", 1)
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
                self._dirs[ids] = self.decode_caps(ids)
        return self._dirs[self.all_caps]

    @property
    def all_angles(self) -> np.ndarray:
        return self._angle_value(self._angle_codes[self.all_caps])

    # --------------------------------------------------------------- budget
    @staticmethod
    def _code_size_of(dim: int, q: Quantize) -> int:
        spec = _ENCODINGS[q]
        if spec.klt_bits:
            return spec.klt_bits // 8
        if spec.sub:
            return ((spec.rank or dim) // spec.sub * spec.bits + 7) // 8
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
        if isinstance(sq, (_SeededPQ, _BasisPQ, _DualPQ, _KLTSQ)):
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

    def _set_storage(self) -> bool:
        """Whether groups are stored as sets (static filters; see _ef_layout)."""
        return self.set_codes and self.insert_method is None

    def _group_bytes(self, level: int, n: int) -> int:
        """Bytes of n caps stored at `level`: codes and angles, in set
        storage with each code's first w bits as an Elias-Fano sequence."""
        per = self.bytes_per_cap(level)
        if n <= 0:
            return 0
        w = _key_width(n, self.code_size(level)) if self._set_storage() else 0
        if w == 0:
            return n * per
        _, low, high = _ef_layout(n, w)
        return low + high + n * (per - w // 8)

    def _calculate_max_caps(self, max_memory: Optional[int]) -> int:
        if max_memory is None:
            return sys.maxsize
        usable = max(0, max_memory - self._codebook_bytes())
        if self.insert_method is None:
            n = max(1, usable // self.bytes_per_cap())
            if self._set_storage():
                # Most caps whose set-stored group fits (bytes grow with n).
                lo, hi = n, 2 * n + 16
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if self._group_bytes(self._uniform, mid) <= usable:
                        lo = mid
                    else:
                        hi = mid - 1
                n = max(1, lo)
            return n
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
        if spec.klt_bits:
            return _KLTSQ(self.dim, spec.klt_bits, spec.rank, fit_on=self.basis_fit)
        if spec.dual:
            return _DualPQ(self.dim, spec.rank, spec.sub, spec.bits, fit_on=self.basis_fit)
        if spec.boot:
            return _BootPQ(self.dim, spec.rank, spec.sub, spec.bits)
        if spec.rank:
            return _BasisPQ(self.dim, spec.rank, spec.sub, spec.bits, fit_on=self.basis_fit)
        if spec.sub:
            return _SeededPQ(self.dim, spec.sub, spec.bits)
        if spec.qtype is None:
            return None
        return faiss.ScalarQuantizer(self.dim, spec.qtype)

    def _train_quantizers(self, directions: np.ndarray, members: Optional[np.ndarray] = None,
                          level: Optional[int] = None) -> None:
        """A fresh quantizer for the encoding in use (or `level`), trained on
        the cap directions (and, for a basis fitted on them, the members)."""
        level = self._uniform if level is None else level
        sq = self._quantizers[level] = self._make_quantizer(self.levels[level])
        if sq is None or len(directions) == 0:
            return
        if isinstance(sq, (_BasisPQ, _DualPQ, _KLTSQ)):
            sq.train(directions, members)
        else:
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
        if fast and isinstance(sq, (_SeededPQ, _BasisPQ)) and not isinstance(sq, _DualPQ):
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
        """Codes of caps `ids` (rows of the widest code size), from the
        per-id array or, in set storage, rebuilt from the groups."""
        ids = np.asarray(ids, dtype=np.int64)
        if self._packed is None:
            return self._codes[ids]
        out = np.zeros((len(ids), int(self._code_sizes.max())), dtype=np.uint8)
        levels = self._cap_level[ids]
        for level in np.unique(levels):
            g = self._packed[int(level)]
            slot = np.full(int(g.ids.max()) + 1, -1, dtype=np.int64)
            slot[g.ids] = np.arange(len(g.ids))
            rows = np.flatnonzero(levels == level)
            codes = g.codes()
            out[rows, :codes.shape[1]] = codes[slot[ids[rows]]]
        return out

    def _pack(self) -> None:
        """Moves a static filter's codes into set storage: per encoding
        group, the caps sorted by their codes' first w bits (_key_width),
        those prefixes Elias-Fano coded and the rest of each code kept as it
        was. The scan reads the groups (_level_codes); ids are reordered so
        each group's caps come in its stored order."""
        ids = self.all_caps
        levels = self._cap_level[ids]
        packed, order = {}, []
        for level in np.unique(levels):
            level = int(level)
            lid = ids[levels == level]
            cs = self.code_size(level)
            codes = self._codes[lid][:, :cs]
            w = _key_width(len(lid), cs)
            if w:
                keys = (codes[:, :w // 8].astype(np.int64) << (8 * np.arange(w // 8))[None, :]).sum(axis=1)
                srt = np.argsort(keys, kind="stable")
                lid, codes = lid[srt], codes[srt]
                l, low, high = _ef_encode(keys[srt], w)
            else:
                l, low, high = 0, np.zeros(0, dtype=np.uint8), np.zeros(0, dtype=np.uint8)
            packed[level] = _Group(ids=lid, w=w, l=l, low=low, high=high,
                                   rem=np.ascontiguousarray(codes[:, w // 8:]), ang=self._angle_codes[lid].copy())
            order.append(lid)
        if order:
            self._all_caps[:len(ids)] = np.concatenate(order)
        self._packed = packed
        self._codes = None

    _NO_KEYS = np.zeros(1, dtype=np.int64)

    def _level_codes(self, level: int, ids: np.ndarray):
        """(codes, keys, w, theta) the scan reads for caps `ids`, all at
        `level`: the codes without their first w bits, those bits per cap
        (decoded from the group's Elias-Fano sequence; w = 0 outside set
        storage) and the stored angles."""
        if self._packed is not None:
            g = self._packed[int(level)]
            return g.rem, (g.keys() if g.w else self._NO_KEYS), g.w, self._angle_value(g.ang)
        codes = np.ascontiguousarray(self._codes[ids][:, :self.code_size(level)])
        return codes, self._NO_KEYS, 0, self._angle_value(self._angle_codes[ids])

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
        if self._codes is not None:
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
        if self._codes is None:
            # Set storage held the codes (see _pack): back to one row per id.
            self._codes = np.zeros((len(self._angle_codes), int(self._code_sizes.max())), dtype=np.uint8)
        self._packed = None
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
        self._train_quantizers(U, X)
        order, starts, counts = segments(labels, len(U))
        alpha = 2.0 * np.arcsin(min(1.0, 0.5 * self.selection_tau))
        # Scored on nearest-codeword encodings: scoring on fitted codes
        # (what construct() stores) measured to pick the same encoding.
        decoded = self.decode_directions(U)
        delta = angle_between(U, decoded)
        theta = segment_max(angle_between(X, decoded[labels])[order], starts, counts)
        return np.sqrt(np.maximum((theta + alpha) ** 2 - delta ** 2, 1e-300))[counts > 0]

    def _volume(self, X: np.ndarray, U: np.ndarray, labels: np.ndarray) -> float:
        """ln of sum over caps of sigma(theta' + beta): the share of the unit
        sphere the cover on directions U, members X[labels == j], would
        accept at selection_tau once encoded (an upper bound on its union's
        measure), theta' the members' widest angle around the decoded
        direction and beta the accept rule's own expansion at |q| = 1.
        Trains the quantizer on U."""
        self._train_quantizers(U, X)
        order, starts, counts = segments(labels, len(U))
        decoded = self.decode_directions(U)
        theta = segment_max(angle_between(X, decoded[labels])[order], starts, counts)
        beta, _ = self._beta(np.ones(1), self.selection_tau)
        psi = np.minimum(theta[counts > 0] + beta[0], np.pi)
        return float(np.logaddexp.reduce(log_cap_measure(psi, self.dim))) if len(psi) else -np.inf

    def _score(self, X: np.ndarray, U: np.ndarray, labels: np.ndarray) -> float:
        """What _choose minimizes for the encoding in use (see there)."""
        if self.selection == "volume":
            return self._volume(X, U, labels)
        phi = self._reach(X, U, labels)
        return float(np.logaddexp.reduce(self.selection_exponent * np.log(phi))) if len(phi) else -np.inf

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
        scores, best, worse = {}, None, 0
        for k, cover in self.construct_method.iter_candidates(self, X, ks, weights):
            cover = cover._replace(directions=unit_rows(cover.directions))
            improved = False
            for level in levels_at[k]:
                self._set_uniform(level)
                score = self._score(X, cover.directions, cover.labels)
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
    # factor; the scan stops after this many counts without a better cover;
    # encodings whose codebook costs more than this many bytes join the
    # mix one at a time (their codebook paid only where used).
    _MIX_STEP = 1.1
    _MIX_PATIENCE = 4
    _CHEAP_CODEBOOK = 1024

    def _mixing(self) -> bool:
        return (bool(self.mixed) and self.selection == "volume" and self.insert_method is None
                and len(self.levels) > 1 and self.max_memory is not None)

    def _construct_budget(self) -> int:
        if self.insert_memory_percent:
            return int(self.max_memory * (1.0 - self.insert_memory_percent / 100.0))
        return int(self.max_memory)

    def _is_boot(self, level: int) -> bool:
        """Whether `level`'s basis is rebuilt from the cover's own caps (_BootPQ)."""
        return _ENCODINGS[self.levels[level]].boot

    def _quick_level(self, level: int) -> bool:
        """Whether `level` encodes fast: TurboQuant / scalar / raw floats,
        or seeded PQ with 8-bit codebooks."""
        spec = _ENCODINGS[self.levels[level]]
        if spec.rank or spec.klt_bits:
            return False
        return spec.qtype is not None or not spec.sub or spec.bits <= 8

    def _data_free(self, level: int) -> bool:
        """Whether `level`'s quantizer is the same whatever it is trained on
        (seeded PQ, TurboQuant, raw floats)."""
        q = self.levels[level]
        spec = _ENCODINGS[q]
        return not (spec.rank or spec.klt_bits) and (spec.qtype is None or spec.rotate)

    def _level_psi(self, X: np.ndarray, U: np.ndarray, labels: np.ndarray,
                   cache: Optional[dict] = None, levels: Optional[List[int]] = None,
                   psi: Optional[np.ndarray] = None) -> np.ndarray:
        """(caps x levels): each cap's accept angle at selection_tau were its
        direction stored at each candidate encoding -- the members' widest
        angle around the nearest-codeword decoded direction (its own encoding
        error for a cap with no member), plus beta. Trains the data-dependent
        levels' quantizers on U. `cache` (level -> {direction bytes: decoded
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
                if not self._is_boot(level):
                    self._train_quantizers(U, X, level=level)
                D = self.decode(self.encode(U, level, fast=True), level)
                theta = segment_max(angle_between(X, D[labels])[order], starts, counts)
                theta = np.where(counts > 0, theta, angle_between(U, D))
                psi[:, level] = np.minimum(theta + beta, np.pi)
                continue
            known = cache.setdefault(level, {})
            if not known:
                self._train_quantizers(U, X, level=level)
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
            total = (base + sum(self._group_bytes(int(allowed[i]), int(counts[i])) for i in used)
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
        wins, and the finished cover is planned again. With mixed="coarsen", no cap
        takes a code finer than the least-volume single encoding's, so caps
        only coarsen and the cover keeps at least that encoding's cap
        count. Returns (cover, each cap's level); the filter's own encoding
        (whose codebook _codebook_bytes charges) becomes the finest one in
        use.

        Measured: see the class docstring (Choosing the encoding)."""
        L = len(self.levels)
        # Boot levels (_BootPQ) need anchors, which only a finished cover
        # has: they join at its final plan (_plan_boot), not the grid.
        grid_levels = [l for l in range(L) if not self._is_boot(l)]
        ks = []
        for level in grid_levels:
            self._set_uniform(level)
            ks.append(max(1, min(self.max_construct_caps, len(X))))
        budget = self._construct_budget()
        cheap = [l for l in grid_levels if self._quantizer_bytes(l) <= self._CHEAP_CODEBOOK]
        costly = [l for l in grid_levels if l not in cheap]
        mixes = ([cheap] if cheap else []) + [cheap + [l] for l in costly]
        grid, k = set(ks), max(ks)
        while k > min(ks):
            grid.add(k)
            k = int(k / self._MIX_STEP)
        # Scoring a level costs an encoding of every new cap: cheap for
        # TurboQuant and 8-bit seeded PQ, not for 12-bit codebooks or a
        # trained basis. Those are scored only when their encoding error on
        # a sample of the members (its 1st percentile, plus beta) is under
        # the widest angle the cheap levels already reach: a cap at such a
        # level could not narrow the cover's widest cap.
        beta = float(self._beta(np.ones(1), self.selection_tau)[0][0])
        quick = [l for l in grid_levels if self._quick_level(l)]
        slow = [l for l in grid_levels if l not in quick]
        sample = X[np.random.default_rng(0x5EED).choice(len(X), min(len(X), 512), replace=False)]
        floor = {}
        for l in slow:
            self._train_quantizers(sample, sample, level=l)
            D = self.decode(self.encode(sample, l, fast=True), l)
            floor[l] = float(np.quantile(angle_between(sample, D), 0.01)) + beta
        # The best plan under each ceiling on code size (None: no ceiling),
        # and each level's own best uniform score.
        ceilings = [None] + sorted(set(int(c) for c in self._code_sizes))
        best, uniform, worse, scores, cache = {}, {}, 0, {}, {}
        for k, cover in self.construct_method.iter_candidates(self, X, sorted(grid, reverse=True), weights):
            U = unit_rows(cover.directions)
            psi = self._level_psi(X, U, cover.labels, cache, levels=quick)
            here = [self._plan(psi, [l for l in m if l in quick], budget) for m in mixes]
            reach = max([r[3] for r in here if r is not None] + [b[4] for b in best.values()] + [-np.inf])
            reach = np.inf if reach == -np.inf else reach
            psi = self._level_psi(X, U, cover.labels, cache, levels=[l for l in slow if floor[l] < reach], psi=psi)
            improved = False
            for allowed in mixes:
                for c in ceilings:
                    capped = [l for l in allowed if c is None or self._code_sizes[l] <= c]
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
        if self.mixed == "coarsen" and uniform:
            # No code finer than the best single encoding's: caps only
            # coarsen where that keeps them within the widest angle, and the
            # bytes freed buy more caps. mix_finer > 0 raises that ceiling
            # by as many rungs (a plan takes a finer code only for a cap
            # that would otherwise exceed its widest angle: the top band).
            key = int(self._code_sizes[min(uniform, key=uniform.get)])
            sizes = ceilings[1:]
            key = sizes[min(sizes.index(key) + self.mix_finer, len(sizes) - 1)]
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
        boot = [l for l in range(L) if self._is_boot(l) and (key is None or self._code_sizes[l] <= key)]
        if plan is not None:
            levels = plan[1]
            if boot:
                levels = self._plan_boot(X, U, cover.labels, psi, use, boot, plan, budget)
        elif cover.kept is not None:
            levels = levels[cover.kept]
        used = np.unique(levels)
        self._set_uniform(int(used[np.argmax(self._code_sizes[used])]))
        self.selection_k = len(U)
        return cover, levels

    def _plan_boot(self, X: np.ndarray, U: np.ndarray, labels: np.ndarray, psi: np.ndarray, use: List[int],
                   boot: List[int], plan, budget: int) -> np.ndarray:
        """The finished cover's plan with boot levels (_BootPQ) allowed: the
        boot_anchors share of caps with most room under the plan's widest
        angle T stay at their planned whole-space codes as anchors, the
        basis is rebuilt from their codes as construct() fits them, and the
        other caps are planned again over `use` plus `boot`. Returns each
        cap's level, the whole-space plan's if the boot plan is no smaller.
        construct() rebuilds the basis from all whole-space caps it stores,
        so stored angles may differ a little from these planned ones."""
        levels, T = plan[1], plan[3]
        rows = np.arange(len(U))
        A = np.argsort(-(T - psi[rows, levels]), kind="stable")[:int(np.ceil(self.boot_anchors * len(U)))]
        local = np.full(len(U), -1, dtype=np.int64)
        local[A] = np.arange(len(A))
        held = np.flatnonzero(local[labels] >= 0)
        codes = self._fit_codes_by_level(U[A], X[held], local[labels[held]], levels[A])
        self._train_boot(self._decode_levels(codes, levels[A]), boot)
        P = np.full((len(U), len(self.levels)), np.inf)
        P[:, use] = psi[:, use]
        P = self._level_psi(X, U, labels, levels=boot, psi=P)
        if self.boot_equal:
            # A cap may take a boot code only of its planned code's size:
            # at the finished cover's cap count, bytes a cheaper code frees
            # buy nothing, and the cap would only widen toward T.
            for l in boot:
                P[self._code_sizes[levels] != self._code_sizes[l], l] = np.inf
        keep = P[A, levels[A]].copy()
        P[A, :] = np.inf
        P[A, levels[A]] = keep
        new = self._plan(P, list(use) + list(boot), budget)
        self.boot_plan = (len(A), None if new is None else float(new[0]), float(plan[0]))
        if new is None or new[0] >= plan[0]:
            return levels
        return new[1]

    def _train_boot(self, anchors: np.ndarray, boot: List[int]) -> None:
        """Rebuilds boot levels' bases from decoded anchor directions."""
        for l in boot:
            self._quantizers[l] = self._make_quantizer(self.levels[l])
            self._quantizers[l].train(anchors)

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
                self._train_quantizers(U, Xt)
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

    def _fit_codes_by_level(self, U: np.ndarray, X: np.ndarray, owner: np.ndarray,
                            levels: np.ndarray) -> np.ndarray:
        """Codes for caps U, each at its entry of `levels`: fitted to the
        members `owner` puts in it where the encoding can be (fit_codes),
        else nearest U's direction."""
        codes = np.zeros((len(U), int(self._code_sizes.max())), dtype=np.uint8)
        boot = [int(l) for l in np.unique(levels) if self._is_boot(int(l))]
        for level in sorted(np.unique(levels), key=lambda l: self._is_boot(int(l))):
            level = int(level)
            if level == (boot[0] if boot else None):
                # The anchors -- every cap at a whole-space level, fitted
                # above -- decoded as a decoder would, rebuild the basis.
                anchors = np.flatnonzero(~np.isin(levels, boot))
                self._train_boot(self._decode_levels(codes[anchors], levels[anchors]), boot)
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
        if len(vectors) == 0:
            self._reset_buffers(0)
            self._constructed = True
            return
        distinct, inverse, copies = self._distinct_rows(vectors)
        X = self._prepare_members(distinct)
        levels = None
        if len(self.levels) > 1:
            if self.selection == "holdout":
                cover = self._choose_holdout(X, copies)
            elif self._mixing():
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
            if not self._is_boot(int(level)):
                self._train_quantizers(U, X, level=int(level))
        codebook = self._codebook_bytes()
        if self.max_memory is not None and codebook + self.bytes_per_cap() > self.max_memory:
            raise InsufficientMemoryError(
                f"max_memory={self.max_memory} bytes cannot hold one {self.quantize.value} cap "
                f"plus its {codebook}-byte codebook.")
        # Fit codes to the members each cap holds, re-settle the members
        # around the new directions, and repeat: each round's members are
        # the ones the last round's caps actually hold. A second round
        # measured within 0.002 FPR of one.
        owner = home
        for _ in range(max(1, self.fit_rounds) if self.fit_codes else 1):
            codes = self._fit_codes_by_level(U, X, owner, levels)
            angles, owner = self._tight_angles(X, U, self._decode_levels(codes, levels), theta, home, neighbors)
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
        if self._set_storage():
            self._pack()

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

    def _block_decoder(self, level: Optional[int] = None):
        """decode(codes, keys, w, out, norms): the float32 vectors `codes`
        (at the encoding in use, or `level`; their first w bits in keys, see
        _level_codes) decode to (exactly _decode_raw's) into `out`, with
        their float64 norms."""
        from filters.cap_cover import scan_kernels
        level = self._uniform if level is None else level
        sq = self._quantizers[level]
        if isinstance(sq, _SeededPQ):
            return lambda codes, keys, w, out, norms: scan_kernels.decode_codebook(
                np.ascontiguousarray(codes), keys, w, sq.codebook, sq.bits, sq.m, out, norms)
        if self._is_turboquant(level):
            bits = int(round(_ENCODINGS[self.levels[level]].rate))
            levels = faiss.vector_to_array(sq.trained)[:1 << bits].astype(np.float32)
            return lambda codes, keys, w, out, norms: scan_kernels.decode_levels(
                np.ascontiguousarray(codes), keys, w, levels, bits, self.dim, out, norms)

        def generic(codes, keys, w, out, norms):
            if w:
                prefix = ((keys[:, None] >> (8 * np.arange(w // 8))[None, :]) & 0xFF).astype(np.uint8)
                codes = np.concatenate([prefix, codes], axis=1)
            out[:] = self._decode_raw(codes, level)
            norms[:] = np.sqrt(np.einsum("ij,ij->i", out.astype(np.float64), out.astype(np.float64)))
        return generic

    def _scan(self, vectors: np.ndarray, tau: float) -> np.ndarray:
        Qw = self.to_working(vectors)
        nq = np.sqrt(np.einsum("ij,ij->i", Qw, Qw))
        beta, feasible = self._beta(nq, tau)
        Qn = Qw / np.maximum(nq, 1e-300)[:, None]
        Qn32 = Qn.astype(np.float32)
        out = np.zeros(len(Qw), dtype=bool)
        slack = self._slack()
        ids = self.all_caps
        levels = self._cap_level[ids]
        if (levels == self._uniform).all():
            return self._scan_level(self._uniform, ids, Qn, Qn32, beta, feasible, slack, out)
        for level in np.unique(levels):
            if not (feasible & ~out).any():
                break
            self._scan_level(int(level), ids[levels == level], Qn, Qn32, beta, feasible, slack, out)
        return out

    def _scan_level(self, level, ids, Qn, Qn32, beta, feasible, slack, out) -> np.ndarray:
        """_scan over caps `ids`, all at encoding `level`."""
        from filters.cap_cover import scan_kernels
        sq = self._quantizers[level]
        codes, keys, w, theta = self._level_codes(level, ids)
        live = feasible & ~out
        n_live = int(live.sum())
        if n_live <= self._DIRECT_MAX_QUERIES and self.single_tables:
            if isinstance(sq, _SeededPQ) and (sq.bits <= 8 or len(codes) >= self._TABLE1_MIN_CAPS):
                rows = np.flatnonzero(live)
                Qs = Qn[rows].reshape(len(rows), sq.m, sq.sub)
                T = np.ascontiguousarray(np.matmul(Qs, sq.codebook.T.astype(np.float64)))
                rest = np.zeros((len(rows), sq.m + 1))
                rest[:, :-1] = np.cumsum(T.max(axis=2)[:, ::-1], axis=1)[:, ::-1]
                scan_kernels.accept_table1(codes, keys, w, sq.bits, sq.m, T, rest, sq.squared_norms(), sq.codebook,
                                           theta, rows, beta, np.ascontiguousarray(Qn), slack, out)
                return out
            gbits = self._lut_bits(level)
            if gbits:
                bits = int(round(_ENCODINGS[self.levels[level]].rate))
                levels = faiss.vector_to_array(sq.trained)[:1 << bits].astype(np.float32)
                g = gbits // bits
                V = levels.astype(np.float64)[(np.arange(1 << gbits)[:, None] >> (bits * np.arange(g))[None, :])
                                              & ((1 << bits) - 1)]
                rows = np.flatnonzero(live)
                T = np.ascontiguousarray(np.matmul(Qn[rows].reshape(len(rows), self.dim // g, g), V.T))
                scan_kernels.accept_lut_levels(codes, keys, w, gbits, self.dim // g, T, (V * V).sum(axis=1), levels,
                                               bits, self.dim, theta, rows, beta, np.ascontiguousarray(Qn), slack,
                                               out)
                return out
        if isinstance(sq, _SeededPQ) and self._direct_pays(sq, n_live):
            scan_kernels.accept_direct(codes, keys, w, sq.bits, sq.m, sq.codebook, sq.squared_norms(), theta,
                                       np.flatnonzero(live), beta, np.ascontiguousarray(Qn), slack, out)
            return out
        if self._is_turboquant(level) and int(live.sum()) <= self._DIRECT_MAX_QUERIES:
            bits = int(round(_ENCODINGS[self.levels[level]].rate))
            levels = faiss.vector_to_array(sq.trained)[:1 << bits].astype(np.float32)
            scan_kernels.accept_direct_levels(codes, keys, w, levels, bits, self.dim, theta, np.flatnonzero(live),
                                              beta, np.ascontiguousarray(Qn), slack, out)
            return out
        if isinstance(sq, _SeededPQ) and self._tables_pay(sq, len(ids)):
            return self._scan_tables(sq, codes, keys, w, theta, Qn, Qn32, beta, live, slack, out)
        decode = self._block_decoder(level)
        raw = np.empty((self._SCAN_BLOCK, self.dim), dtype=np.float32)
        norms = np.empty(self._SCAN_BLOCK)

        for start in range(0, len(codes), self._SCAN_BLOCK):
            pending = np.flatnonzero(~out & feasible)
            if len(pending) == 0:
                break
            stop = min(start + self._SCAN_BLOCK, len(codes))
            nb = stop - start
            decode(codes[start:stop], keys[start:stop] if w else keys, w, raw[:nb], norms[:nb])
            th = theta[start:stop]
            for q0 in range(0, len(pending), self._SCAN_QUERY_CHUNK):
                rows = pending[q0:q0 + self._SCAN_QUERY_CHUNK]
                S = Qn32[rows] @ raw[:nb].T
                scan_kernels.accept_block(S, rows, raw[:nb], norms[:nb], th, beta, Qn, slack, out)
        return out

    # Queries per set of per-query codeword tables (scan_kernels.accept_tables).
    _TABLE_QUERY_CHUNK = 32
    # Largest batch scored by scan_kernels.accept_direct.
    _DIRECT_MAX_QUERIES = 3
    # Fewest caps of a 10/12-bit seeded-PQ group a batch of 1-3 queries
    # scores through per-query tables (accept_table1) rather than straight
    # from the codebook (accept_direct): a table costs 2^bits * dim
    # multiply-adds per query.
    _TABLE1_MIN_CAPS = 2048

    def _lut_bits(self, level: int) -> int:
        """Bits per table index for TurboQuant `level`'s group tables
        (accept_lut_levels): whole bytes of 1/2/4-bit codes, 6 bits (two
        dimensions) of 3-bit ones; 0 = no tables (8-bit codes)."""
        if not self._is_turboquant(level):
            return 0
        bits = int(round(_ENCODINGS[self.levels[level]].rate))
        return {1: 8, 2: 8, 3: 6, 4: 8}.get(bits, 0)

    @classmethod
    def _direct_pays(cls, sq: "_SeededPQ", n_queries: int) -> bool:
        """Whether a batch is small enough to score seeded-PQ codes one query
        at a time straight from the codebook (scan_kernels.accept_direct)."""
        return sq.bits > 8 and n_queries <= cls._DIRECT_MAX_QUERIES

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

    def _scan_tables(self, sq, codes, keys, w, theta, Qn, Qn32, beta, feasible, slack, out) -> np.ndarray:
        """_scan for seeded-PQ codes (as _level_codes gives them) without
        decoding: per chunk of queries, a table of every slice against every
        codeword, summed per cap by scan_kernels.accept_tables, which decides
        exactly as accept_block."""
        from filters.cap_cover import scan_kernels
        sq_norms = sq.squared_norms()
        pending = np.flatnonzero(feasible)
        for q0 in range(0, len(pending), self._TABLE_QUERY_CHUNK):
            rows = pending[q0:q0 + self._TABLE_QUERY_CHUNK]
            Qs = Qn32[rows].reshape(len(rows), sq.m, sq.sub)
            T = np.ascontiguousarray(np.matmul(sq.codebook[None], Qs.transpose(1, 2, 0)))
            scan_kernels.accept_tables(codes, keys, w, sq.bits, sq.m, T, sq_norms, sq.codebook, theta, rows, beta,
                                       Qn, slack, out)
        return out

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

    # A static filter with caps at several encodings stores them grouped by
    # encoding, each group after the first behind its 4-byte count.
    _GROUP_BYTES = 4

    def get_bytesize(self) -> int:
        if not self._constructed:
            return 0
        levels = self._cap_level[self.all_caps]
        total = self._codebook_bytes()
        for level, n in zip(*np.unique(levels, return_counts=True)):
            total += self._group_bytes(int(level), int(n))
            if level != self._uniform:
                total += self._quantizer_bytes(int(level)) + self._GROUP_BYTES
        if self._packed is not None:
            # What the groups actually hold (equal to the charge above).
            held = sum(len(g.low) + len(g.high) + g.rem.nbytes + len(g.ids) * (self.angle_bits // 8)
                       for g in self._packed.values())
            charged = sum(self._group_bytes(int(l), int(n)) for l, n in zip(*np.unique(levels, return_counts=True)))
            if held != charged:
                raise AssertionError(f"set storage holds {held} B but charges {charged} B")
        if self.insert_method:
            total += self.insert_method.get_bytesize(self)
        return total

    def get_construction_bytesize(self) -> int:
        return self.construct_method.get_bytesize() if self.construct_method else 0
