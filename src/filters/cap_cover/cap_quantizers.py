"""Direction quantizers behind CapCoverFilter's encodings (moved verbatim out of
cap_cover_filter.py)."""
import numpy as np
import faiss
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple
from filters.cap_cover.cap_geometry import angle_between, segments, segment_max, unit_rows

_PROBE_RNG = np.random.default_rng(0x5EED)


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
