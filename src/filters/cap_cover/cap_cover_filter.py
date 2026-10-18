import numpy as np
import faiss
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence, Tuple, Union
from filters.filter import Filter, InsufficientMemoryError
from filters.cap_cover.cap_quantizers import (Quantize, _Encoding, _ENCODINGS, BOOT_CANDIDATES, _SeededPQ, _BasisPQ, _BootPQ, _DualPQ, _KLTSQ,  # noqa: F401
                                                _gaussian_lloyd_max, _LLOYD_MAX, _fit_scalar_codes, _PROBE_RNG)
from filters.cap_cover.cap_scan import _CapScanMixin
from filters.cap_cover.cap_interfaces import ConstructInterface, Cover, InsertInterface
from filters.cap_cover.cap_geometry import (angle_between, log_cap_measure, nearest_directions, segments,
                                            segment_max, unit_rows)




_ROTATION_SEED = 0x5EED
_ROTATION_SEED_BYTES = 4
_ANGLE_GRID_BYTES = 8      # a static 8-bit filter's angle grid: two float32


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
    high), the rest of each code (rem) and each cap's angle code.

    `keys()` and the decoded angle array (`_level_codes`'s theta) used to be
    recomputed on every query() call -- pure waste for a static filter
    (set_codes's only user, see _set_storage), whose groups are rebuilt
    wholesale by _pack() and never mutated in place afterward: caching them
    here is exactly as safe as not calling _pack() twice. On a cover with a
    few thousand caps this removed most of a single query's own time, behind
    only the scan kernel itself (measured: cap_cover_system_tau_worker's
    single-query throughput, suggestions/cap_cover_query_cache_report.md)."""
    ids: np.ndarray
    w: int
    l: int
    low: np.ndarray
    high: np.ndarray
    rem: np.ndarray
    ang: np.ndarray
    _keys_cache: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    _theta_cache: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    def keys(self) -> np.ndarray:
        if self._keys_cache is None:
            from filters.cap_cover import scan_kernels
            out = np.empty(len(self.ids), dtype=np.int64)
            if self.w:
                scan_kernels.ef_decode(self.low, self.high, len(self.ids), self.l, out)
            self._keys_cache = out
        return self._keys_cache

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
# coarse caps and large ones fewer, finer caps. Encodings with a trained
# basis (PSPQ*, DPQ*, KSQ*) are left out: at |B| = 8000 their basis takes
# most of a small budget and mixing never picked them at k <= 1%, and at 80K
# they double construction time (see CapCoverFilter's Encodings trained on B).
AUTO_CANDIDATES = (Quantize.SPQ12, Quantize.SPQ16, Quantize.SPQ24, Quantize.SPQ36, Quantize.SPQ48,
                   Quantize.SPQ64, Quantize.TQ2, Quantize.TQ3, Quantize.TQ4)
# Seeded-PQ rungs between AUTO_CANDIDATES' at 10 bits (auto()'s default, with
# mix_finer=1): a cap can coarsen, or the top band refine, by a smaller step,
# and a 2^bits codebook costs dim * 2^bits multiply-adds per encoded cap, so
# 10-bit rungs keep construction inside the benchmark's 5-minute limit at 80K.
DENSE10_LADDER = (Quantize.SPQ20, Quantize.SPQ30, Quantize.SPQ40, Quantize.SPQ60, Quantize.SPQ80)

class CapCoverFilter(_CapScanMixin, Filter):
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
    the one with the least accept region on the sphere (_volume): ln of the
    sum over caps of sigma(theta' + beta) at selection_tau (theta' the
    members' widest angle around the decoded direction) -- from B and
    the stored caps only, so it holds whatever the queries. At
    d = 384 the widest caps dominate the sum. With `mixed` (static
    filters) each cap takes its own encoding (_choose_mixed): "coarsen"
    (default) lets caps narrower than the widest take coarser codes than
    the least-volume single encoding where that keeps them within the
    widest angle, buying more caps with the bytes freed; "free" also
    lets the widest caps take finer codes, which shrinks the region most
    but spends bytes on few caps; False keeps one encoding (_choose).

    Measured (src/compare_cap_cover.py, |B| = 8000, 5 corpora x 5 folds,
    fold-paired; results/cap_cover/v1_*.csv, v5_*.csv): against the
    least-volume single encoding (mixed=False), mixed="coarsen" moves FPR by
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
    With DENSE10_LADDER's smaller steps and set storage (cdf1_*.csv, 175
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
    -83.9/-110.8/-126.3 (80K).

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
        """A filter that picks its encodings for the budget at construct()
        (by accept-region volume, see _choose_mixed) among AUTO_CANDIDATES plus
        DENSE10_LADDER, with the top band one rung finer (mix_finer=1): the
        defaults since 2026-10-04 (see Choosing the encoding and The top
        band). candidates=AUTO_CANDIDATES, mix_finer=0 gives the earlier
        ones."""
        candidates = [q for q in kwargs.pop("candidates", AUTO_CANDIDATES + DENSE10_LADDER) if _fits(dim, q)]
        kwargs.setdefault("mix_finer", 1)
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
    # _scan's placeholder `ids` when self._packed is not None: _level_codes
    # ignores it entirely in that case (it reads the group's own g.ids), and
    # _scan_level's one other use of `ids` (the _tables_pay call) reads
    # len(codes) instead -- see both below.
    _NO_IDS = np.zeros(0, dtype=np.int64)

    def _level_codes(self, level: int, ids: np.ndarray):
        """(codes, keys, w, theta) the scan reads for caps `ids`, all at
        `level`: the codes without their first w bits, those bits per cap
        (decoded from the group's Elias-Fano sequence; w = 0 outside set
        storage) and the stored angles."""
        if self._packed is not None:
            g = self._packed[int(level)]
            if g._theta_cache is None:
                g._theta_cache = self._angle_value(g.ang)
            return g.rem, (g.keys() if g.w else self._NO_KEYS), g.w, g._theta_cache
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
    # factor; the scan stops after this many counts without a better cover;
    # encodings whose codebook costs more than this many bytes join the
    # mix one at a time (their codebook paid only where used).
    _MIX_STEP = 1.1
    _MIX_PATIENCE = 4
    _CHEAP_CODEBOOK = 1024

    def _mixing(self) -> bool:
        return (bool(self.mixed) and self.insert_method is None
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
    def query(self, vectors: np.ndarray, distance_threshold: float) -> List[bool]:
        if not self._constructed:
            raise ValueError("CapCoverFilter has not been constructed.")
        if self._num_caps == 0 or len(vectors) == 0:
            return [False] * len(vectors)
        return self._scan(vectors, float(distance_threshold)).tolist()

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
