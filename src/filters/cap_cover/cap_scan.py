"""Query-side scan dispatch of CapCoverFilter (moved verbatim out of
cap_cover_filter.py), as a mixin: it reads the filter's attributes and class
constants through self."""
import numpy as np
import faiss
from typing import Optional, Tuple
from filters.cap_cover.cap_quantizers import _ENCODINGS, _SeededPQ


class _CapScanMixin:
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
        if self._packed is not None:
            # A static cover's groups are the whole story: _pack() fixes
            # which levels exist and which caps each holds, so redoing
            # self.all_caps/self._cap_level/np.unique bookkeeping here on
            # every query is pure waste -- _level_codes already reads
            # straight from the group and ignores `ids` in this case (see
            # the len(codes) fix in _scan_level's _tables_pay call below).
            levels_present = sorted(self._packed)
            if len(levels_present) == 1 and levels_present[0] == self._uniform:
                return self._scan_level(self._uniform, self._NO_IDS, Qn, Qn32, beta, feasible, slack, out)
            for level in levels_present:
                if not (feasible & ~out).any():
                    break
                self._scan_level(level, self._NO_IDS, Qn, Qn32, beta, feasible, slack, out)
            return out
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
        if isinstance(sq, _SeededPQ) and self._tables_pay(sq, len(codes)):
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
