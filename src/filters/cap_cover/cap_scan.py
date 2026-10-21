"""Query path of CapCoverFilter, as a mixin: the accept rule's constants and the
bridge to the Rust scan (capcover_rs, built from CapCoverRustFilter/ with maturin).

The first query() after a construct()/insert()/remove() "freezes" the filter:
each encoding's caps are handed to Rust, which owns the one copy of the code
bytes; the Python-side code buffer is replaced by an empty array until
something needs it again (insert, remove, decoding the caps), when it is
rebuilt from Rust ("thawed"). Nothing derived is kept between calls.

Encodings Rust has no kernel for (FP32, FP16, SQ*) are answered by the float64
rule over the decoded caps: exact but O(queries * caps * d), a reference."""
import os
import numpy as np
import faiss
from typing import Tuple
import capcover_rs
from filters.cap_cover.cap_quantizers import _ENCODINGS, _SeededPQ
from filters.cap_cover.cap_geometry import angle_between, unit_rows


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

    # ------------------------------------------------------------ freeze/thaw
    def _table(self, level: int):
        """(kind, bits, table) of encoding `level`, or None if Rust has no kernel for it."""
        sq = self._quantizers[level]
        if isinstance(sq, _SeededPQ):
            return "pq", sq.bits, sq.codebook
        if self._is_turboquant(level):
            bits = int(round(_ENCODINGS[self.levels[level]].rate))
            levels = faiss.vector_to_array(sq.trained)[:1 << bits].astype(np.float32)
            return "tq", bits, levels[:, None]
        return None

    def _supported(self) -> bool:
        used = np.unique(self._cap_level[self.all_caps])
        return all(self._table(int(l)) is not None for l in used)

    def _group_ids(self, level: int) -> np.ndarray:
        """Ids of the caps at `level`, in the order Rust scans them: widest
        angle first for a static filter (a query inside a wide cap exits
        early; caps are a set, so order changes no answer), else id order
        (insert strategies keep per-id bookkeeping)."""
        ids = self.all_caps
        gids = ids[self._cap_level[ids] == level]
        if self.insert_method is None:
            gids = gids[np.argsort(-self._angle_codes[gids].astype(np.int64), kind="stable")]
        return gids

    def _freeze(self) -> None:
        ids = self.all_caps
        rust = capcover_rs.CapIndex(self.dim, (self._angle_lo, self._angle_hi, float(self._angle_levels)))
        levels = [int(l) for l in np.unique(self._cap_level[ids])]
        for level in levels:
            gids = self._group_ids(level)
            kind, bits, table = self._table(level)
            codes = np.ascontiguousarray(self._codes[gids][:, :self.code_size(level)])
            rust.add_group(kind, bits, np.ascontiguousarray(table), codes, np.ascontiguousarray(self._angle_codes[gids]))
        rust.set_threads(self.rust_threads)
        self._rust, self._groups = rust, levels
        self._codes = np.zeros((0, self._codes.shape[1]), dtype=np.uint8)

    def _thaw(self) -> None:
        if self._rust is None or len(self._codes):
            return
        codes = np.zeros((len(self._angle_codes), int(self._code_sizes.max())), dtype=np.uint8)
        for g, level in enumerate(self._groups):
            codes[self._group_ids(level), :self.code_size(level)] = self._rust.export_codes(g)
        self._codes = codes

    def _invalidate(self) -> None:
        """Called before anything that changes the caps."""
        self._thaw()
        self._rust, self._groups = None, []

    def freeze(self) -> None:
        """Drops the Python-side code buffer again after something thawed it."""
        if self._rust is not None:
            self._codes = np.zeros((0, self._codes.shape[1]), dtype=np.uint8)

    # ----------------------------------------------------------------- scan
    # Queries per Rust call: the float64 rotated copy is 3 KB per query, so a
    # 100K-query batch would otherwise allocate 300 MB.
    _QUERY_BLOCK = 8192
    # Queries per pass of the float64 fallback (a chunk x caps x d tensor).
    _DENSE_CHUNK = 64

    def _scan(self, vectors: np.ndarray, tau: float) -> list:
        if self._rust is None:
            if not self._supported():
                return self._scan_dense(vectors, tau).tolist()
            self._freeze()
        tp = float(tau) + self._COVERAGE_EPSILON + self._NORM_TOLERANCE
        out = []
        for i in range(0, len(vectors), self._QUERY_BLOCK):     # bounds the float64 working copy
            out.extend(self._rust.query(np.ascontiguousarray(self.to_working(vectors[i:i + self._QUERY_BLOCK])), tp).tolist())
        return out

    def _scan_dense(self, vectors: np.ndarray, tau: float) -> np.ndarray:
        """The accept rule in float64 over the decoded caps: angle(q, u) <=
        theta + beta(|q|), or theta + beta >= pi."""
        Qw = self.to_working(vectors)
        n = np.sqrt(np.einsum("ij,ij->i", Qw, Qw))
        beta, feasible = self._beta(n, tau)
        Qn = unit_rows(Qw)
        ids = self.all_caps
        C = self.decode_caps(ids)
        theta = self._angle_value(self._angle_codes[ids])
        out = np.zeros(len(Qw), dtype=bool)
        for s in range(0, len(Qw), self._DENSE_CHUNK):
            reach = theta[None, :] + beta[s:s + self._DENSE_CHUNK, None]
            ang = angle_between(Qn[s:s + self._DENSE_CHUNK, None, :], C[None, :, :])
            out[s:s + self._DENSE_CHUNK] = ((ang <= reach) | (reach >= np.pi)).any(axis=1)
        return out & feasible
