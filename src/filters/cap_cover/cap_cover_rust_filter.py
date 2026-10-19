"""CapCoverFilter whose query() runs in Rust (capcover_rs, PyO3).

Construction, insertion and removal are the original code. The first query()
after a construct()/insert()/remove() "freezes" the filter: each encoding's
caps are handed to Rust, which owns the one copy of the code bytes; the
Python-side code buffer is dropped (replaced by an empty array) until
something needs it again (insert, remove, decoding the caps), when it is
rebuilt from Rust ("thawed"). Nothing derived is kept between calls.

Needs the capcover_rs extension (built from CapCoverRustFilter/ with maturin)."""
import os
import numpy as np
import faiss
import capcover_rs
from filters.cap_cover.cap_cover_filter import CapCoverFilter
from filters.cap_cover.cap_quantizers import _ENCODINGS, _SeededPQ


class CapCoverRustFilter(CapCoverFilter):
    def __init__(self, *args, rust_threads: int = None, **kwargs) -> None:
        """rust_threads: worker threads of query() (default: env CAPCOVER_RUST_THREADS, else 1;
        the evaluation runs one process per core, so 1 is the right default there)."""
        super().__init__(*args, **kwargs)
        self.rust_threads = int(os.environ.get("CAPCOVER_RUST_THREADS", 1)) if rust_threads is None else rust_threads
        self._rust = None            # capcover_rs.CapIndex while frozen
        self._groups = []            # encodings in Rust's group order

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
        self._thaw()
        self._rust, self._groups = None, []

    # ----------------------------------------------------- hooks on the base
    def _stored_codes(self, ids):
        self._thaw()
        return super()._stored_codes(ids)

    def construct(self, vectors) -> None:
        self._invalidate()
        super().construct(vectors)

    def insert(self, vectors) -> None:
        self._invalidate()
        super().insert(vectors)

    def remove(self, vectors):
        self._invalidate()
        return super().remove(vectors)

    # ----------------------------------------------------------------- query
    # Queries per Rust call: the float64 rotated copy is 3 KB per query, so a
    # 100K-query batch would otherwise allocate 300 MB.
    _QUERY_BLOCK = 8192

    def query_original(self, vectors: np.ndarray, distance_threshold: float):
        """The unmodified numba path, on the same stored caps (for comparison:
        thaws the code buffer, so the filter holds two copies until the next
        freeze())."""
        self._thaw()
        return CapCoverFilter.query(self, vectors, distance_threshold)

    def freeze(self) -> None:
        """Drops the Python-side code buffer again (after query_original)."""
        if self._rust is not None:
            self._codes = np.zeros((0, self._codes.shape[1]), dtype=np.uint8)

    def query(self, vectors: np.ndarray, distance_threshold: float):
        if not self._constructed:
            raise ValueError("CapCoverFilter has not been constructed.")
        if self._num_caps == 0 or len(vectors) == 0:
            return [False] * len(vectors)
        if self._rust is None:
            if not self._supported():
                return super().query(vectors, distance_threshold)
            self._freeze()
        tp = float(distance_threshold) + self._COVERAGE_EPSILON + self._NORM_TOLERANCE
        out = []
        for i in range(0, len(vectors), self._QUERY_BLOCK):     # bounds the float64 working copy
            out.extend(self._rust.query(np.ascontiguousarray(self.to_working(vectors[i:i + self._QUERY_BLOCK])), tp).tolist())
        return out
