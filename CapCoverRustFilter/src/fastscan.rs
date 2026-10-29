//! FastScan-style SIMD kernel for TurboQuant groups with 1, 2 or 4 bits per
//! dimension (Quick-ADC / FAISS FastScan: 16 table lookups per instruction).
//!
//! Layout. Groups of this kind store their code bytes transposed in blocks of
//! 16 caps: block b, byte position B, cap lane c lives at
//! `codes[b*16*cs + B*16 + c]`, so one 16-byte load fetches byte B of 16 caps.
//! This is a permutation of exactly the bytes the array-of-structures layout
//! holds; the ragged tail (n % 16 caps) stays array-of-structures after the
//! blocks, so nothing is padded. A byte holds two 4-bit nibbles = 4/b
//! dimensions each, so a nibble indexes a 16-entry table.
//!
//! Per query: one u8 table per nibble position p (entry v = the nibble's
//! dimensions of q times the levels v decodes to, minus a per-table offset,
//! divided by one common scale `delta`, rounded). `pshufb`/`tbl` looks up 16
//! caps at once and the u8 results are summed in u16 lanes (flushed to u32
//! every 128 bytes so 255 * 256 lookups cannot overflow).
//!
//!   dot_est = sum_p off_p + delta * acc,   |dot - dot_est| <= E.
//!
//! Error bound E (all constants generous). Each table entry is within
//! delta (1/2 + 2e-4) of the exact entry (rounding to the nearest integer;
//! 2e-4 covers f32 evaluation of the quotient, which is below 255 by
//! construction since delta >= (max table range) / 254). The f32 construction
//! of the exact entries (q cast to f32, <= 4 products and sums, the offset
//! subtraction) adds <= 16 u Lmax |q|_1 over all tables (u = 2^-24, Lmax =
//! max |level|, Cauchy-free: sum of |q_j| |level|), and evaluating
//! so + delta acc in f32 <= 1e-5 (|so| + 255 delta tables). Hence
//!   E = tables delta (1/2 + 2e-4) + 16 u Lmax |q|_1 + 1e-5 (|so| + 255 delta tables).
//!
//! |raw|^2 does not depend on the query: one 16-entry table (sum of squared
//! levels per nibble value), quantized to 14 bits (Q = 128 hi + lo, two 7-bit
//! tables so a nibble pair adds in u8 before widening; scale dn, offset
//! min) with explicit error
//!   En = nibbles dn (1/2 + 1e-4) + 1e-5 (|no| + 255 dn nibbles).
//! (An 8-bit norm table was tried first: its bound was ~55% of |raw|^2 at TQ4
//! and sent 11% of caps to tier 2; 14 bits sends 0.8%.) When all levels have
//! equal squares (TQ1: +-a) the table is constant and the kernel skips the
//! norm lookups entirely.
//!
//! Decision, per cap, in tiers. With kl = cos(theta + beta) - slack (the
//! scalar gate's constant), a cap is *skipped* iff kl > 0 and
//!   max(dot_est + E, 0)^2 < (n2_est - En) kl^2.
//! Since dot <= dot_est + E and |raw|^2 >= n2_est - En this gives
//! dot < |raw| kl < |raw| cos(theta + beta): the exact rule rejects the cap,
//! with slack |raw| (4.6e-5 at d = 384) to spare for the f32 rounding of this
//! test itself. Every other cap is a *candidate* and is decided exactly by
//! `exact_cap` (float64 from the decoded code, same rule as the scalar
//! path's recheck), so the group answer equals the scalar path's. A cap
//! that clearly passes is also just a candidate: it costs one exact check
//! (~0.3 us) and then ends the scan.

use crate::core::*;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};

pub const BLOCK: usize = 16;
/// Bytes per u16 accumulator flush: 128 bytes = 256 lookups of <= 255.
const CHUNK: usize = 128;
const U: f64 = 1.0 / 16777216.0;

// ------------------------------------------------------------------ layout

/// A/B switch for benchmarks: groups added while this is false use the scalar kernel.
pub static ENABLED: AtomicBool = AtomicBool::new(true);

/// Whether a group can use the layout: TQ with 1, 2 or 4 bits and whole nibbles.
pub fn eligible(codec: &Codec, dim: usize, code_size: usize) -> bool {
    if !ENABLED.load(Relaxed) { return false; }
    match codec {
        Codec::Tq { bits, .. } => {
            matches!(bits, 1 | 2 | 4) && dim % (4 / bits) == 0 && code_size * 8 == dim * bits && code_size > 0
        }
        _ => false,
    }
}

/// Array-of-structures codes -> blocks of 16 (tail unchanged).
pub fn to_soa(aos: &[u8], cs: usize) -> Vec<u8> {
    let n = aos.len() / cs;
    let nb = n / BLOCK;
    let mut out = aos.to_vec();
    for b in 0..nb {
        for c in 0..BLOCK {
            for j in 0..cs { out[(b * cs + j) * BLOCK + c] = aos[(b * BLOCK + c) * cs + j]; }
        }
    }
    out
}

/// Cap k's code bytes (inverse of `to_soa` for one cap); `n` = caps in the group.
pub fn gather(codes: &[u8], cs: usize, n: usize, k: usize, out: &mut [u8]) {
    if k < n / BLOCK * BLOCK {
        let base = (k / BLOCK) * BLOCK * cs + k % BLOCK;
        for j in 0..cs { out[j] = codes[base + j * BLOCK]; }
    } else {
        out[..cs].copy_from_slice(&codes[k * cs..(k + 1) * cs]);
    }
}

// ---------------------------------------------------------------- backends

/// The few 128-bit operations the kernel needs. Lookup indices are always
/// < 16, so tbl (out-of-range -> 0) and pshufb (high bit -> 0) agree.
pub trait Ops {
    type V: Copy;
    /// 16 u16 lanes (cap lanes) of running sums.
    type A: Copy;
    fn load(p: &[u8; 16]) -> Self::V;
    fn lo(v: Self::V) -> Self::V;
    fn hi(v: Self::V) -> Self::V;
    fn look(table: Self::V, idx: Self::V) -> Self::V;
    /// Lane-wise u8 sum (callers keep it below 256).
    fn sum8(a: Self::V, b: Self::V) -> Self::V;
    fn zero() -> Self::A;
    fn add(a: Self::A, v: Self::V) -> Self::A;
    /// out[i] += lane i.
    fn flush(a: Self::A, out: &mut [u32; 16]);
}

/// Portable emulation: fallback and the reference for the tests.
pub struct Emu;
impl Ops for Emu {
    type V = [u8; 16];
    type A = [u16; 16];
    #[inline(always)] fn load(p: &[u8; 16]) -> [u8; 16] { *p }
    #[inline(always)] fn lo(v: [u8; 16]) -> [u8; 16] { v.map(|x| x & 15) }
    #[inline(always)] fn hi(v: [u8; 16]) -> [u8; 16] { v.map(|x| x >> 4) }
    #[inline(always)] fn look(t: [u8; 16], i: [u8; 16]) -> [u8; 16] { i.map(|x| t[(x & 15) as usize]) }
    #[inline(always)] fn sum8(a: [u8; 16], b: [u8; 16]) -> [u8; 16] { std::array::from_fn(|i| a[i] + b[i]) }
    #[inline(always)] fn zero() -> [u16; 16] { [0; 16] }
    #[inline(always)] fn add(mut a: [u16; 16], v: [u8; 16]) -> [u16; 16] { for i in 0..16 { a[i] += v[i] as u16; } a }
    #[inline(always)] fn flush(a: [u16; 16], out: &mut [u32; 16]) { for i in 0..16 { out[i] += a[i] as u32; } }
}

#[cfg(target_arch = "aarch64")]
pub use neon::Neon;
#[cfg(target_arch = "aarch64")]
mod neon {
    use super::Ops;
    use std::arch::aarch64::*;
    /// NEON is part of the aarch64 baseline: the intrinsics are always available.
    /// Pointer loads/stores are of exactly 16 u8 / 8 u16 from references of that size.
    pub struct Neon;
    #[allow(unused_unsafe)]
    impl Ops for Neon {
        type V = uint8x16_t;
        type A = (uint16x8_t, uint16x8_t);
        #[inline(always)] fn load(p: &[u8; 16]) -> uint8x16_t { unsafe { vld1q_u8(p.as_ptr()) } }
        #[inline(always)] fn lo(v: uint8x16_t) -> uint8x16_t { unsafe { vandq_u8(v, vdupq_n_u8(15)) } }
        #[inline(always)] fn hi(v: uint8x16_t) -> uint8x16_t { unsafe { vshrq_n_u8::<4>(v) } }
        #[inline(always)] fn look(t: uint8x16_t, i: uint8x16_t) -> uint8x16_t { unsafe { vqtbl1q_u8(t, i) } }
        #[inline(always)] fn sum8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t { unsafe { vaddq_u8(a, b) } }
        #[inline(always)] fn zero() -> Self::A { unsafe { (vdupq_n_u16(0), vdupq_n_u16(0)) } }
        #[inline(always)] fn add(a: Self::A, v: uint8x16_t) -> Self::A {
            unsafe { (vaddw_u8(a.0, vget_low_u8(v)), vaddw_high_u8(a.1, v)) }
        }
        #[inline(always)] fn flush(a: Self::A, out: &mut [u32; 16]) {
            let mut t = [0u16; 16];
            unsafe { vst1q_u16(t.as_mut_ptr(), a.0); vst1q_u16(t.as_mut_ptr().add(8), a.1); }
            for i in 0..16 { out[i] += t[i] as u32; }
        }
    }
}

#[cfg(target_arch = "x86_64")]
use ssse3::Ssse3;
#[cfg(target_arch = "x86_64")]
mod ssse3 {
    use super::Ops;
    use std::arch::x86_64::*;
    /// Only used through `run_ssse3`, which is entered after
    /// `is_x86_feature_detected!("ssse3")` and is compiled with that feature
    /// so these inline into it. Mirror of the NEON backend, line by line.
    pub struct Ssse3;
    impl Ops for Ssse3 {
        type V = __m128i;
        type A = (__m128i, __m128i);
        #[inline(always)] fn load(p: &[u8; 16]) -> __m128i { unsafe { _mm_loadu_si128(p.as_ptr() as *const __m128i) } }
        #[inline(always)] fn lo(v: __m128i) -> __m128i { unsafe { _mm_and_si128(v, _mm_set1_epi8(15)) } }
        #[inline(always)] fn hi(v: __m128i) -> __m128i { unsafe { _mm_and_si128(_mm_srli_epi16::<4>(v), _mm_set1_epi8(15)) } }
        #[inline(always)] fn look(t: __m128i, i: __m128i) -> __m128i { unsafe { _mm_shuffle_epi8(t, i) } }
        #[inline(always)] fn sum8(a: __m128i, b: __m128i) -> __m128i { unsafe { _mm_add_epi8(a, b) } }
        #[inline(always)] fn zero() -> Self::A { unsafe { (_mm_setzero_si128(), _mm_setzero_si128()) } }
        #[inline(always)] fn add(a: Self::A, v: __m128i) -> Self::A {
            unsafe {
                let z = _mm_setzero_si128();
                (_mm_add_epi16(a.0, _mm_unpacklo_epi8(v, z)), _mm_add_epi16(a.1, _mm_unpackhi_epi8(v, z)))
            }
        }
        #[inline(always)] fn flush(a: Self::A, out: &mut [u32; 16]) {
            let mut t = [0u16; 16];
            unsafe {
                _mm_storeu_si128(t.as_mut_ptr() as *mut __m128i, a.0);
                _mm_storeu_si128(t.as_mut_ptr().add(8) as *mut __m128i, a.1);
            }
            for i in 0..16 { out[i] += t[i] as u32; }
        }
    }
}

// ------------------------------------------------------------ query tables

/// Query-independent per-group data (16-entry nibble tables of the levels).
struct GroupFs {
    per: usize,             // dimensions per nibble (4 / bits)
    nibs: usize,            // nibble positions = 2 * code_size
    nl: [[f32; 16]; 4],     // nl[j][v]: level of nibble value v's j-th dimension
    nv: [f64; 16],          // squared norm of nibble value v
    lmin: f32,
    lmax: f32,
    lmax_abs: f64,
    norm: NormLut,
}

/// Quantized sum-of-squares table: n2_est = no + dn * acc, |n2 - n2_est| <= en.
struct NormLut {
    /// Q = 128 hi + lo (7 bits each), Q <= 16383: n2_est = no + dn (128 sum(hi) + sum(lo)).
    hi: [u8; 16],
    lo: [u8; 16],
    no: f32,
    dn: f32,
    en: f32,
    /// False when every nibble value has the same squared norm.
    varies: bool,
}

impl GroupFs {
    fn new(levels: &[f32], bits: usize, code_size: usize) -> GroupFs {
        let per = 4 / bits;
        let mask = (1usize << bits) - 1;
        let mut nl = [[0f32; 16]; 4];
        let mut nv = [0f64; 16];
        for v in 0..16 {
            for j in 0..per {
                let l = levels[(v >> (bits * j)) & mask];
                nl[j][v] = l;
                nv[v] += l as f64 * l as f64;
            }
        }
        let nibs = 2 * code_size;
        let lmin = levels.iter().copied().fold(f32::INFINITY, f32::min);
        let lmax = levels.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let lmax_abs = levels.iter().map(|l| l.abs() as f64).fold(0.0, f64::max);
        let (nmin, nmax) = nv.iter().fold((f64::INFINITY, 0f64), |(a, b), &x| (a.min(x), b.max(x)));
        let varies = nmax > nmin;
        let dn = if varies { (nmax - nmin) / 16383.0 * (1.0 + 1e-5) } else { 0.0 };
        let (mut hi, mut lo) = ([0u8; 16], [0u8; 16]);
        if varies {
            for v in 0..16 {
                let q = ((nv[v] - nmin) / dn + 0.5) as u32;
                (hi[v], lo[v]) = ((q >> 7) as u8, (q & 127) as u8);
            }
        }
        let no = nibs as f64 * nmin;
        let en = if varies { nibs as f64 * dn * (0.5 + 1e-4) } else { 0.0 } + 1e-5 * (no.abs() + 255.0 * dn * nibs as f64) + 1e-6 * no;
        let norm = NormLut { hi, lo, no: no as f32, dn: dn as f32, en: en as f32, varies };
        GroupFs { per, nibs, nl, nv, lmin, lmax, lmax_abs, norm }
    }
}

/// dot_est = so + delta * acc, error at most `e`.
struct DotLut { so: f32, delta: f32, e: f32 }

/// Fills `out` (16 bytes per nibble position) for the unit query `qn`.
fn build_dot_lut(g: &GroupFs, qn: &[f64], out: &mut Vec<u8>) -> DotLut {
    let (per, nibs) = (g.per, g.nibs);
    out.resize(nibs * 16, 0);
    let span = (g.lmax - g.lmin) as f64;
    let (mut max_a, mut l1) = (0f64, 0f64);
    for qs in qn.chunks_exact(per) {
        let a: f64 = qs.iter().map(|x| x.abs()).sum();
        max_a = max_a.max(a);
        l1 += a;
    }
    // Table p's entries span exactly span * sum|q_j| (dimensions are independent).
    let delta = (span * max_a / 254.0 * (1.0 + 1e-5)) as f32;
    let inv = if delta > 0.0 { 1.0 / delta } else { 0.0 };
    let mut so = 0f64;
    for (qs, o) in qn.chunks_exact(per).zip(out.chunks_exact_mut(16)) {
        let (mut t, mut off) = ([0f32; 16], 0f32);
        for (j, &qj) in qs.iter().enumerate() {
            let q = qj as f32;
            off += (q * g.lmin).min(q * g.lmax);
            for v in 0..16 { t[v] += q * g.nl[j][v]; }
        }
        for v in 0..16 { o[v] = ((t[v] - off) * inv + 0.5) as u8; }
        so += off as f64;
    }
    let d = delta as f64;
    let e = nibs as f64 * d * (0.5 + 2e-4) + 16.0 * U * g.lmax_abs * l1 + 1e-5 * (so.abs() + 255.0 * d * nibs as f64);
    DotLut { so: so as f32, delta, e: e as f32 }
}

// ------------------------------------------------------------------ kernel

/// Sums of the dot and norm tables for the 16 caps of one block. The norm
/// halves are 7-bit, so a nibble pair adds in u8 before widening (<= 254).
#[inline(always)]
fn block_sums<O: Ops, const NORM: bool>(blk: &[u8], lut: &[u8], nm: &NormLut, d: &mut [u32; 16], n: &mut [u32; 16]) {
    let cs = blk.len() / BLOCK;
    let (nh, nlo) = (O::load(&nm.hi), O::load(&nm.lo));
    let mut nacc = [0u32; 16];
    for c0 in (0..cs).step_by(CHUNK) {
        let (mut ad, mut ah, mut al) = (O::zero(), O::zero(), O::zero());
        for byte in c0..(c0 + CHUNK).min(cs) {
            let c = O::load(blk[byte * 16..byte * 16 + 16].try_into().unwrap());
            let (l, h) = (O::lo(c), O::hi(c));
            let t = &lut[byte * 32..byte * 32 + 32];
            ad = O::add(ad, O::look(O::load(t[..16].try_into().unwrap()), l));
            ad = O::add(ad, O::look(O::load(t[16..].try_into().unwrap()), h));
            if NORM {
                ah = O::add(ah, O::sum8(O::look(nh, l), O::look(nh, h)));
                al = O::add(al, O::sum8(O::look(nlo, l), O::look(nlo, h)));
            }
        }
        O::flush(ad, d);
        if NORM {
            let (mut th, mut tl) = ([0u32; 16], [0u32; 16]);
            O::flush(ah, &mut th);
            O::flush(al, &mut tl);
            for i in 0..16 { nacc[i] += 128 * th[i] + tl[i]; }
        }
    }
    *n = nacc;
}

/// Bit i set iff lane i is not provably rejected (see the module comment).
#[inline(always)]
fn candidates(d: &[u32; 16], n: &[u32; 16], g2: &[f32; 16], q: &DotLut, nm: &NormLut) -> u16 {
    let mut mask = 0u16;
    for i in 0..16 {
        let dot_hi = q.so + q.delta * d[i] as f32 + q.e;
        let n2_lo = nm.no + nm.dn * n[i] as f32 - nm.en;
        let dd = dot_hi.max(0.0);
        // g2 = 0 for "no skipping"; a NaN/negative n2_lo also fails the compare.
        mask |= (!(dd * dd < n2_lo * g2[i]) as u16) << i;
    }
    mask
}

/// Cap k decided in float64 from its code (the scalar path's recheck rule,
/// without the f32 prefilter): dot and |raw|^2 are summed per nibble from
/// the f32 levels, so they equal the decoded vector's sums up to f64
/// rounding (~1e-15 relative); anything within 1e-9 |raw| of the threshold
/// is settled by decoding and `exact_angle`, exactly like the scalar path.
fn exact_cap(idx: &Index, g: &Group, gf: &GroupFs, k: usize, qn: &[f64], b: f64, code: &mut [u8], raw: &mut [f32]) -> bool {
    let theta = idx.grid.theta(g.acodes[k]);
    if theta + b >= PI { return true; }
    let cs = g.code_size;
    gather(&g.codes, cs, g.acodes.len(), k, &mut code[..cs]);
    let (mut dot, mut n2) = (0f64, 0f64);
    for (p, qs) in qn.chunks_exact(gf.per).enumerate() {
        let v = ((code[p / 2] >> (4 * (p & 1))) & 15) as usize;
        for (j, &x) in qs.iter().enumerate() { dot += x * gf.nl[j][v] as f64; }
        n2 += gf.nv[v];
    }
    let n = n2.sqrt();
    let thr = n * (b.cos() * theta.cos() - b.sin() * theta.sin());
    let tol = 1e-9 * n;
    if dot >= thr + tol { return true; }
    if dot <= thr - tol { return false; }
    g.codec.decode(&code[..cs], idx.dim, raw);
    let n = raw.iter().map(|&v| v as f64 * v as f64).sum::<f64>().sqrt();
    exact_angle(qn, raw, n) <= theta + b
}

/// Counters (summed into the globals once per `Index::query`).
#[derive(Default, Clone, Copy)]
pub struct Stats { pub lanes: u64, pub candidates: u64, pub queries: u64 }
static LANES: AtomicU64 = AtomicU64::new(0);
static CANDS: AtomicU64 = AtomicU64::new(0);
static QUERIES: AtomicU64 = AtomicU64::new(0);

/// (cap lanes tested by tier 1, tier-2 candidates, group scans) since process start.
pub fn global_stats() -> (u64, u64, u64) { (LANES.load(Relaxed), CANDS.load(Relaxed), QUERIES.load(Relaxed)) }

/// Per-call scratch of the kernel: sizes depend on dim and the codebooks only.
pub struct Fs {
    groups: Vec<Option<GroupFs>>,
    lut: Vec<u8>,
    gate: Vec<f32>,
    blk: Vec<u8>,
    code: Vec<u8>,
    pub stats: Stats,
}

impl Fs {
    pub fn new(idx: &Index) -> Fs {
        let groups: Vec<Option<GroupFs>> = idx.groups.iter().map(|g| match &g.codec {
            Codec::Tq { bits, levels } if g.soa => Some(GroupFs::new(levels, *bits, g.code_size)),
            _ => None,
        }).collect();
        let max_cs = idx.groups.iter().map(|g| g.code_size).max().unwrap_or(0);
        Fs { groups, lut: vec![], gate: vec![], blk: vec![0; BLOCK * max_cs], code: vec![0; max_cs], stats: Stats::default() }
    }

    /// Adds this scratch's counters to the process-wide ones.
    pub fn flush_stats(&mut self) {
        LANES.fetch_add(self.stats.lanes, Relaxed);
        CANDS.fetch_add(self.stats.candidates, Relaxed);
        QUERIES.fetch_add(self.stats.queries, Relaxed);
        self.stats = Stats::default();
    }
}

/// Whether some cap of the SoA group `gi` accepts (unit `qn`, expansion `b`).
#[inline(always)]
pub fn accepts_with<O: Ops>(idx: &Index, gi: usize, g: &Group, qn: &[f64], b: f64, fs: &mut Fs,
                            cos_sin: &[(f64, f64)], raw: &mut [f32]) -> bool {
    let Fs { groups, lut, gate, blk: tail, code, stats } = fs;
    let gf = groups[gi].as_ref().expect("not a FastScan group");
    let (cb, sb, slack) = (b.cos(), b.sin(), slack(idx.dim));
    let gate_of = |ac: u16| -> f32 {
        let (ct, st) = if cos_sin.is_empty() { let t = idx.grid.theta(ac); (t.cos(), t.sin()) } else { cos_sin[ac as usize] };
        let kl = cb * ct - sb * st - slack;
        // theta + beta >= pi accepts outright (cos can be positive beyond 3 pi / 2): never skip.
        if kl > 0.0 && idx.grid.theta(ac) + b < PI { (kl * kl) as f32 } else { 0.0 }
    };
    if !cos_sin.is_empty() {
        gate.clear();
        gate.extend((0..cos_sin.len()).map(|a| gate_of(a as u16)));
    }
    let (cs, n) = (g.code_size, g.acodes.len());
    let q = build_dot_lut(gf, qn, lut);
    let nm = &gf.norm;
    stats.queries += 1;
    let g2_of = |ac: u16| if cos_sin.is_empty() { gate_of(ac) } else { gate[ac as usize] };

    let mut run = |blk: &[u8], k0: usize, lanes: usize| -> bool {
        let (mut d, mut nn) = ([0u32; 16], [0u32; 16]);
        if nm.varies { block_sums::<O, true>(blk, lut, nm, &mut d, &mut nn); }
        else { block_sums::<O, false>(blk, lut, nm, &mut d, &mut nn); }
        let mut g2 = [0f32; 16];
        for i in 0..lanes { g2[i] = g2_of(g.acodes[k0 + i]); }
        let mut mask = candidates(&d, &nn, &g2, &q, nm) & (((1u32 << lanes) - 1) as u16);
        stats.lanes += lanes as u64;
        while mask != 0 {
            let i = mask.trailing_zeros() as usize;
            mask &= mask - 1;
            stats.candidates += 1;
            if exact_cap(idx, g, gf, k0 + i, qn, b, code, raw) { return true; }
        }
        false
    };

    let nb = n / BLOCK;
    for bl in 0..nb {
        if run(&g.codes[bl * BLOCK * cs..(bl + 1) * BLOCK * cs], bl * BLOCK, BLOCK) { return true; }
    }
    let m = n - nb * BLOCK;
    if m > 0 {
        // Ragged tail: transposed into per-call scratch (zero lanes are ignored), never stored.
        let t = &mut tail[..BLOCK * cs];
        t.fill(0);
        let src = &g.codes[nb * BLOCK * cs..];
        for c in 0..m { for j in 0..cs { t[j * BLOCK + c] = src[c * cs + j]; } }
        if run(t, nb * BLOCK, m) { return true; }
    }
    false
}

/// Runtime-selected backend.
#[inline]
pub fn accepts(idx: &Index, gi: usize, g: &Group, qn: &[f64], b: f64, fs: &mut Fs, cos_sin: &[(f64, f64)], raw: &mut [f32]) -> bool {
    #[cfg(target_arch = "aarch64")]
    { return accepts_with::<Neon>(idx, gi, g, qn, b, fs, cos_sin, raw); }
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("ssse3") {
        // SAFETY: the CPU supports SSSE3 (checked just above).
        return unsafe { accepts_ssse3(idx, gi, g, qn, b, fs, cos_sin, raw) };
    }
    #[allow(unreachable_code)]
    accepts_with::<Emu>(idx, gi, g, qn, b, fs, cos_sin, raw)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
unsafe fn accepts_ssse3(idx: &Index, gi: usize, g: &Group, qn: &[f64], b: f64, fs: &mut Fs, cos_sin: &[(f64, f64)], raw: &mut [f32]) -> bool {
    accepts_with::<Ssse3>(idx, gi, g, qn, b, fs, cos_sin, raw)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
        fn f(&mut self) -> f64 { (self.next() >> 11) as f64 / (1u64 << 53) as f64 }
        fn below(&mut self, n: usize) -> usize { (self.next() % n as u64) as usize }
        fn gauss(&mut self) -> f64 { (-2.0 * self.f().max(1e-300).ln()).sqrt() * (2.0 * PI * self.f()).cos() }
    }

    fn unit(v: &mut [f64]) { let n = v.iter().map(|x| x * x).sum::<f64>().sqrt(); v.iter_mut().for_each(|x| *x /= n); }

    fn random_levels(rng: &mut Rng, bits: usize) -> Vec<f32> {
        let scale = 10f64.powf(rng.f() * 8.0 - 4.0);
        let sym = rng.below(2) == 0;
        let mut l: Vec<f32> = (0..1usize << bits).map(|i| {
            let m = ((i as f64 + 0.5) / (1usize << bits) as f64 * 2.0 - 1.0) * (0.5 + rng.f());
            (if sym { m } else { m + rng.f() * 0.4 } * scale) as f32
        }).collect();
        if bits == 1 && rng.below(2) == 0 { l = vec![-l[1].abs(), l[1].abs()]; } // TQ1-like: +-a
        for x in l.iter_mut() { if *x == 0.0 { *x = 1e-3; } }
        l
    }

    struct Case { soa: Index, aos: Index, grid: Grid }

    fn make_case(rng: &mut Rng) -> Case {
        let dim = [8usize, 16, 24, 64, 96, 384][rng.below(6)];
        let n_groups = 1 + rng.below(2);
        let lo = rng.f() * 0.5;
        let hi = if rng.below(4) == 0 { PI + 0.3 } else { 0.5 + rng.f() * 2.5 };
        let levels = if rng.f() < 0.5 { 255.0 } else { 65535.0 };
        let (mut gs_soa, mut gs_aos) = (vec![], vec![]);
        for _ in 0..n_groups {
            let bits = [1usize, 2, 4][rng.below(3)];
            let levs = random_levels(rng, bits);
            let cs = dim * bits / 8;
            let n = [0usize, 1, 5, 15, 16, 17, 31, 47][rng.below(8)] + rng.below(20);
            let codes: Vec<u8> = (0..n * cs).map(|_| rng.next() as u8).collect();
            let lim = levels as usize + 1;
            let acodes: Vec<u16> = (0..n).map(|_| rng.below(lim) as u16).collect();
            let mk = |soa: bool| {
                let c = Codec::Tq { bits, levels: levs.clone() };
                if soa { Group::new(c, dim, cs, codes.clone(), acodes.clone()) } else { Group::aos(c, dim, cs, codes.clone(), acodes.clone()) }
            };
            gs_soa.push(mk(true));
            gs_aos.push(mk(false));
        }
        let g = Grid { lo, hi, levels };
        Case {
            soa: { let mut i = Index::new(dim, Grid { lo, hi, levels }); i.groups = gs_soa; i },
            aos: { let mut i = Index::new(dim, Grid { lo, hi, levels }); i.groups = gs_aos; i },
            grid: g,
        }
    }

    /// Float64 reference of the rule; None if some cap is within 1e-12 of a tie.
    fn reference(idx: &Index, q: &[f64], b: f64) -> Option<bool> {
        let mut hit = false;
        let mut raw = vec![0f32; idx.dim];
        for g in &idx.groups {
            let mut code = vec![0u8; g.code_size];
            for k in 0..g.acodes.len() {
                let theta = idx.grid.theta(g.acodes[k]);
                if theta + b >= PI { hit = true; continue; }
                g.code_into(k, &mut code);
                g.codec.decode(&code, idx.dim, &mut raw);
                let n = raw.iter().map(|&v| v as f64 * v as f64).sum::<f64>().sqrt();
                let ang = exact_angle(q, &raw, n);
                if (ang - (theta + b)).abs() < 1e-12 { return None; }
                // The chord formula (the rule's own definition) is only good to ~1e-8 near pi.
                if theta + b > PI - 1e-3 && (ang - (theta + b)).abs() < 1e-6 { return None; }
                hit |= ang <= theta + b;
            }
        }
        Some(hit)
    }

    fn fast_answer<O: Ops>(idx: &Index, q: &[f64], b: f64, fs: &mut Fs) -> bool {
        let cos_sin: Vec<(f64, f64)> = if idx.grid.levels <= 255.0 {
            (0..=idx.grid.levels as usize).map(|a| { let t = idx.grid.theta(a as u16); (t.cos(), t.sin()) }).collect()
        } else { vec![] };
        let mut raw = vec![0f32; idx.dim];
        idx.groups.iter().enumerate().any(|(gi, g)| accepts_with::<O>(idx, gi, g, q, b, fs, &cos_sin, &mut raw))
    }

    /// Queries: random, and aimed at a cap's boundary angle +- delta (inside the tier-1 error band).
    fn queries(rng: &mut Rng, c: &Case, tp: f64) -> Vec<Vec<f64>> {
        let d = c.soa.dim;
        let mut out: Vec<Vec<f64>> = (0..4).map(|_| { let mut v: Vec<f64> = (0..d).map(|_| rng.gauss()).collect(); unit(&mut v); v }).collect();
        let b = beta(1.0, tp).unwrap();
        let mut raw = vec![0f32; d];
        for g in c.soa.groups.iter().filter(|g| !g.acodes.is_empty()) {
            for _ in 0..12 {
                let k = rng.below(g.acodes.len());
                let mut code = vec![0u8; g.code_size];
                g.code_into(k, &mut code);
                g.codec.decode(&code, d, &mut raw);
                let mut u: Vec<f64> = raw.iter().map(|&x| x as f64).collect();
                unit(&mut u);
                let mut w: Vec<f64> = (0..d).map(|_| rng.gauss()).collect();
                let p: f64 = w.iter().zip(&u).map(|(a, b)| a * b).sum();
                w.iter_mut().zip(&u).for_each(|(a, b)| *a -= p * b);
                unit(&mut w);
                let theta = c.soa.grid.theta(g.acodes[k]);
                let delta = [0.0, 1e-12, -1e-12, 1e-9, -1e-9, 1e-6, -1e-6, 1e-3, -1e-3, 1e-2, -1e-2, 5e-2, -5e-2][rng.below(13)];
                let phi = (theta + b + delta).clamp(0.0, PI);
                let mut q: Vec<f64> = u.iter().zip(&w).map(|(a, b)| phi.cos() * a + phi.sin() * b).collect();
                unit(&mut q);
                out.push(q);
            }
        }
        out
    }

    fn debug_dump(idx: &Index, q: &[f64], b: f64, tp: f64) {
        let mut raw = vec![0f32; idx.dim];
        eprintln!("dim {} b {b} tp {tp} grid {} {} {}", idx.dim, idx.grid.lo, idx.grid.hi, idx.grid.levels);
        for g in &idx.groups { let mut code = vec![0u8; g.code_size];
            for k in 0..g.acodes.len() {
                let theta = idx.grid.theta(g.acodes[k]);
                g.code_into(k, &mut code); g.codec.decode(&code, idx.dim, &mut raw);
                let n = raw.iter().map(|&v| v as f64 * v as f64).sum::<f64>().sqrt();
                let dot: f64 = raw.iter().zip(q).map(|(&r, &x)| r as f64 * x).sum();
                let ang = exact_angle(q, &raw, n);
                if ang <= theta + b + 1e-6 || theta + b >= PI { eprintln!("k {k} theta+b {} ang {ang} dot/n {} cos {}", theta + b, dot / n, (theta + b).cos()); }
            } }
    }

    fn check_case(seed: u64) {
        let mut rng = Rng(seed | 1);
        for _ in 0..3 { rng.next(); }
        let c = make_case(&mut rng);
        let tp = [0.01, 0.1, 0.4, 1.0, 1.9][rng.below(5)];
        let b = beta(1.0, tp).unwrap();
        let mut fs = Fs::new(&c.soa);
        for q in queries(&mut rng, &c, tp) {
            let Some(want) = reference(&c.soa, &q, b) else { continue };
            // The existing scalar kernel reads codes 8 table groups at a time, so it is only
            // right when dim / (8 / bits) is a multiple of 8 (true at d = 384).
            let scalar_ok = c.aos.groups.iter().all(|g| match &g.codec {
                Codec::Tq { bits, .. } => (c.aos.dim / (8 / bits)) % 8 == 0, _ => true })
                // The scalar gate also skips caps with theta + beta >= pi when cos(theta + beta) > 0.
                && c.aos.grid.hi + b < PI;
            let scalar = if scalar_ok { c.aos.query(&q, tp, None)[0] } else { want };
            let emu = fast_answer::<Emu>(&c.soa, &q, b, &mut fs);
            if emu != want { debug_dump(&c.soa, &q, b, tp); }
            assert_eq!(emu, want, "emu vs f64 reference (seed {seed})");
            if scalar != want { debug_dump(&c.aos, &q, b, tp); }
            assert_eq!(scalar, want, "scalar vs f64 reference (seed {seed})");
            assert_eq!(c.soa.query(&q, tp, None)[0], want, "Index::query vs reference (seed {seed})");
            #[cfg(target_arch = "aarch64")]
            assert_eq!(fast_answer::<Neon>(&c.soa, &q, b, &mut fs), want, "neon vs reference (seed {seed})");
        }
        let _ = &c.grid;
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(400))]
        #[test]
        fn fastscan_matches_scalar_and_reference(seed in any::<u64>()) { check_case(seed); }
    }

    #[test]
    #[ignore]
    fn one_seed() { check_case(std::env::var("SEED").unwrap().parse().unwrap()); }

    /// The error bounds hold for every cap: |dot - dot_est| <= e and |n2 - n2_est| <= en,
    /// also for spiky queries and extreme level scales.
    #[test]
    fn estimates_within_stated_bounds() {
        let mut rng = Rng(0x9E3779B97F4A7C15);
        let mut worst = 0f64;
        for round in 0..300 {
            let bits = [1usize, 2, 4][round % 3];
            let dim = [16usize, 96, 384][rng.below(3)];
            let cs = dim * bits / 8;
            let levs = random_levels(&mut rng, bits);
            let gf = GroupFs::new(&levs, bits, cs);
            let mut q: Vec<f64> = (0..dim).map(|_| rng.gauss()).collect();
            match round % 4 { // spiky / sparse / dense
                0 => q.iter_mut().enumerate().for_each(|(i, x)| if i != 3 { *x *= 1e-3 }),
                1 => q.iter_mut().for_each(|x| *x = x.signum()),
                _ => {}
            }
            unit(&mut q);
            let mut lut = vec![];
            let dl = build_dot_lut(&gf, &q, &mut lut);
            let codes: Vec<u8> = (0..16 * cs).map(|_| rng.next() as u8).collect();
            let blk = to_soa(&codes, cs);
            let (mut d, mut n) = ([0u32; 16], [0u32; 16]);
            block_sums::<Emu, true>(&blk, &lut, &gf.norm, &mut d, &mut n);
            let mut raw = vec![0f32; dim];
            for c in 0..16 {
                Codec::Tq { bits, levels: levs.clone() }.decode(&codes[c * cs..(c + 1) * cs], dim, &mut raw);
                let dot: f64 = raw.iter().zip(&q).map(|(&r, &x)| r as f64 * x).sum();
                let n2: f64 = raw.iter().map(|&r| r as f64 * r as f64).sum();
                let est = dl.so as f64 + dl.delta as f64 * d[c] as f64;
                let nest = gf.norm.no as f64 + gf.norm.dn as f64 * n[c] as f64;
                assert!((dot - est).abs() <= dl.e as f64, "dot err {} > {}", (dot - est).abs(), dl.e);
                assert!((n2 - nest).abs() <= gf.norm.en as f64, "n2 err {} > {}", (n2 - nest).abs(), gf.norm.en);
                worst = worst.max((dot - est).abs() / dl.e as f64);
            }
        }
        assert!(worst > 0.0);
        eprintln!("largest actual/stated dot error ratio: {worst:.3}");
    }

    #[test]
    fn soa_roundtrip() {
        let mut rng = Rng(7);
        for n in 0..50usize {
            let cs = 12;
            let aos: Vec<u8> = (0..n * cs).map(|_| rng.next() as u8).collect();
            let soa = to_soa(&aos, cs);
            assert_eq!(soa.len(), aos.len());
            let mut out = vec![0u8; cs];
            for k in 0..n { gather(&soa, cs, n, k, &mut out); assert_eq!(&out[..], &aos[k * cs..(k + 1) * cs]); }
        }
    }

    /// Neon and the emulation agree bit for bit on the sums.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_matches_emulation() {
        let mut rng = Rng(11);
        for _ in 0..50 {
            let cs = 48 * (1 + rng.below(4));
            let blk: Vec<u8> = (0..16 * cs).map(|_| rng.next() as u8).collect();
            let lut: Vec<u8> = (0..32 * cs).map(|_| rng.next() as u8).collect();
            let nm = NormLut { hi: std::array::from_fn(|_| rng.next() as u8 & 127), lo: std::array::from_fn(|_| rng.next() as u8 & 127),
                               no: 0.0, dn: 0.0, en: 0.0, varies: true };
            let (mut a, mut b, mut c, mut d) = ([0u32; 16], [0u32; 16], [0u32; 16], [0u32; 16]);
            block_sums::<Emu, true>(&blk, &lut, &nm, &mut a, &mut b);
            block_sums::<Neon, true>(&blk, &lut, &nm, &mut c, &mut d);
            assert_eq!((a, b), (c, d));
        }
    }
}
