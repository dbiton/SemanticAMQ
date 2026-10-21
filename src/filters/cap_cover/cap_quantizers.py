"""Direction quantizers behind CapCoverFilter's encodings."""
import numpy as np
import faiss
from dataclasses import dataclass
from enum import Enum
from typing import Optional

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

    def _nearest_codewords(self, S: np.ndarray, top: int) -> np.ndarray:
        """(n x top) indices of the codewords of largest inner product with
        each row of S (one slice per row), best
        first: faiss's blocked GEMM with a running top-k, rather than an
        (n x 2^bits) score matrix."""
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
            out = np.empty(len(X) * self.m, dtype=np.int64)
            scan_kernels.nearest_codeword(np.ascontiguousarray(X).reshape(-1, self.sub), self._cbt, out)
            return self._pack(out.reshape(len(X), self.m))
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

    def fit_cap_code(self, u: np.ndarray, P: np.ndarray, t: np.ndarray, candidates: int = 8,
                     passes: int = 2) -> np.ndarray:
        """CapCoverFilter.fit_cap_code's search: the code (1 x code_size)
        whose decoded direction holds caps (P[i], t[i]) within the least
        angle, max_i angle(decode, P[i]) + t[i], by coordinate search over
        each slice's `candidates` codewords nearest u's, starting from u's
        own code."""
        def stored(cos):
            return (np.arccos(np.clip(cos, -1.0, 1.0)) + t[:, None]).max(axis=0)

        m, s, T = self.m, self.sub, min(candidates, 1 << self.bits)
        cand = np.concatenate([self._nearest_codewords(u.reshape(m, s)[j:j + 1], T) for j in range(m)])
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
