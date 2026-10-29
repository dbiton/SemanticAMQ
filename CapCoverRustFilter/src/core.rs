//! Query path of CapCoverFilter (src/filters/cap_cover/cap_scan.py,
//! scan_kernels.py), without Python. Same stored bytes, same accept rule.
//!
//! A unit query q (working basis) is accepted iff some cap (code u, angle
//! theta) has angle(q, u) <= theta + beta, where u is the code's decoded
//! vector `raw` normalized. In terms of the float32-decodable `raw`:
//!     q . raw >= |raw| cos(theta + beta).
//! The score q . raw is computed with a small float error; a pair is decided
//! outright only when the score clears the threshold by more than
//! slack * |raw| (slack bounds the score's error for unit q); pairs inside
//! that band are re-decided in float64 from the decoded `raw`, exactly as
//! scan_kernels._exact_angle does.

pub const PI: f64 = std::f64::consts::PI;

pub enum Codec {
    /// TurboQuant: one `bits`-bit code per dimension, decoded via `levels`.
    Tq { bits: usize, levels: Vec<f32> },
    /// Seeded PQ: one `bits`-bit code per `sub`-dim slice, decoded to that codeword.
    Pq { bits: usize, m: usize, sub: usize, codebook: Vec<f32> },
}

pub struct Group {
    pub codec: Codec,
    pub code_size: usize,
    pub codes: Vec<u8>,   // n * code_size, little-endian bit streams
    pub acodes: Vec<u16>, // angle codes
    plan: Plan,
    /// FastScan layout (fastscan.rs): `codes` holds n/16 transposed blocks, then the AoS tail.
    pub soa: bool,
    /// |raw|^2 contribution of each table index: a function of the codebook
    /// only (<= 16 KB), not of the caps.
    norms: Vec<f32>,
}

impl Group {
    /// Cap k's code bytes, whatever the layout (a gather for blocked caps).
    pub fn code_into(&self, k: usize, out: &mut [u8]) {
        if self.soa { crate::fastscan::gather(&self.codes, self.code_size, self.acodes.len(), k, out) }
        else { out[..self.code_size].copy_from_slice(&self.codes[k * self.code_size..(k + 1) * self.code_size]) }
    }

    /// Group in the FastScan layout when its encoding has a kernel, else plain.
    pub fn new(codec: Codec, dim: usize, code_size: usize, codes: Vec<u8>, acodes: Vec<u16>) -> Group {
        Group::build(codec, dim, code_size, codes, acodes, true)
    }

    /// A group in plain cap-major layout (scalar kernels only).
    pub fn aos(codec: Codec, dim: usize, code_size: usize, codes: Vec<u8>, acodes: Vec<u16>) -> Group {
        Group::build(codec, dim, code_size, codes, acodes, false)
    }

    fn build(codec: Codec, dim: usize, code_size: usize, codes: Vec<u8>, acodes: Vec<u16>, blocked: bool) -> Group {
        let soa = blocked && crate::fastscan::eligible(&codec, dim, code_size);
        let codes = if soa { crate::fastscan::to_soa(&codes, code_size) } else { codes };
        let mut g = Group { codec, code_size, codes, acodes, plan: Plan::Scalar, soa, norms: vec![] };
        g.plan = plan(&g, dim);
        g.norms = match (&g.codec, &g.plan) {
            (Codec::Tq { bits, levels }, Plan::TqLut { gbits, gdims }) => tq_norm_table(levels, *bits, *gbits, *gdims),
            (Codec::Pq { codebook, sub, .. }, Plan::PqLut | Plan::PqDirect) => codebook.chunks(*sub)
                .map(|c| c.iter().map(|&v| v as f64 * v as f64).sum::<f64>() as f32).collect(),
            _ => vec![],
        };
        g
    }
}

pub struct Grid {
    pub lo: f64,
    pub hi: f64,
    pub levels: f64,
}

impl Grid {
    /// cap_cover_filter._angle_value, op for op.
    #[inline]
    pub fn theta(&self, code: u16) -> f64 {
        self.lo + code as f64 * ((self.hi - self.lo).max(0.0) / self.levels)
    }
}

pub struct Index {
    pub dim: usize,
    pub groups: Vec<Group>,
    pub grid: Grid,
    /// The gate's angle buckets (angle code >> shift: 256 of them), as
    /// (theta at the bucket's upper end, its cos and sin): a function of the
    /// grid only (4 KB). Exact per code on an 8-bit grid.
    buckets: Vec<(f64, f64, f64)>,
    shift: u32,
    /// Exact cos/sin per angle code on an 8-bit grid (4 KB; empty otherwise), for fastscan.
    pub(crate) cos_sin: Vec<(f64, f64)>,
}

/// j-th `bits`-bit code of a little-endian bit stream.
#[inline]
fn get_code(code: &[u8], j: usize, bits: usize) -> usize {
    let bit = j * bits;
    let byte = bit >> 3;
    let mut w = code[byte] as u32;
    if byte + 1 < code.len() { w |= (code[byte + 1] as u32) << 8; }
    ((w >> (bit & 7)) & ((1 << bits) - 1)) as usize
}

impl Codec {
    /// (q . raw, |raw|^2) accumulated in float64 straight from the decoded
    /// values: the reference the tables approximate, used where no table pays.
    fn score_f64(&self, code: &[u8], qn: &[f64]) -> (f32, f32) {
        let (mut dot, mut n2) = (0f64, 0f64);
        match self {
            Codec::Tq { bits, levels } => {
                for (i, &q) in qn.iter().enumerate() {
                    let v = levels[get_code(code, i, *bits)] as f64;
                    dot += v * q;
                    n2 += v * v;
                }
            }
            Codec::Pq { bits, m, sub, codebook } => {
                for j in 0..*m {
                    let c = get_code(code, j, *bits);
                    for t in 0..*sub {
                        let v = codebook[c * sub + t] as f64;
                        dot += v * qn[j * sub + t];
                        n2 += v * v;
                    }
                }
            }
        }
        (dot as f32, n2 as f32)
    }

    /// Decodes a cap's `raw` vector (float32, as the Python decoders do).
    pub(crate) fn decode(&self, code: &[u8], dim: usize, out: &mut [f32]) {
        match self {
            Codec::Tq { bits, levels } => {
                for i in 0..dim { out[i] = levels[get_code(code, i, *bits)]; }
            }
            Codec::Pq { bits, m, sub, codebook } => {
                for j in 0..*m {
                    let c = get_code(code, j, *bits);
                    out[j * sub..(j + 1) * sub].copy_from_slice(&codebook[c * sub..(c + 1) * sub]);
                }
            }
        }
    }
}

/// angle(q, raw / norm) in float64 by the chord formula (scan_kernels._exact_angle).
pub(crate) fn exact_angle(q: &[f64], raw: &[f32], norm: f64) -> f64 {
    let mut s = 0.0;
    for i in 0..q.len() {
        let d = q[i] - raw[i] as f64 / norm;
        s += d * d;
    }
    2.0 * (0.5 * s.sqrt()).min(1.0).asin()
}

/// cap_scan._beta for one norm: Some(beta) iff a unit vector can be within tp of it.
pub fn beta(n: f64, tp: f64) -> Option<f64> {
    if !((n - 1.0).abs() <= tp && n > 0.0) { return None; }
    let c = (n * n + 1.0 - tp * tp) / (2.0 * n);
    Some(c.clamp(-1.0, 1.0).acos())
}

pub fn slack(dim: usize) -> f64 {
    2.0 * (dim as f64 + 4.0) * 2f64.powi(-24)
}

/// Per-thread, per-call scratch: the tables filled for one query at a time.
/// Sizes depend on dim and the codebook, never on the number of caps.
struct Scratch {
    /// Per query: kl^2 for the scan's gate, by angle code.
    gate2: Vec<f32>,
    table: Vec<f32>,
    q32: Vec<f32>,
    qt: Vec<f32>,
    raw: Vec<f32>,
    fs: crate::fastscan::Fs,
}

/// How a group's scores are computed.
enum Plan {
    /// Group tables: codes read `gbits` bits at a time, `gdims` dimensions per
    /// index, T[k][v] = those dimensions of q times the levels v decodes to.
    TqLut { gbits: usize, gdims: usize },
    /// One table per slice and codeword: T[s][c] = q's slice s . codeword c.
    PqLut,
    /// Seeded PQ scored from the codebook (pq_direct), no tables: a group of
    /// few caps does not repay building 2^bits x dim table entries per query.
    PqDirect,
    /// No tables: decode and accumulate in float64 (TQ8, dims the lanes do not divide).
    Scalar,
}

fn plan(g: &Group, dim: usize) -> Plan {
    match &g.codec {
        Codec::Tq { bits, .. } => {
            let gbits = match bits { 1 | 2 | 4 => 8, 3 => 6, _ => 0 };
            if gbits == 0 { return Plan::Scalar; }
            let gdims = gbits / bits;
            if dim % (gdims * 8) != 0 { Plan::Scalar } else { Plan::TqLut { gbits, gdims } }
        }
        Codec::Pq { bits, m, sub, .. } => {
            if m % 8 != 0 || !matches!(bits, 8 | 12) { Plan::Scalar }
            // measured (3000 caps, d = 384): a cap costs ~40 ns direct, a table ~6 ns, and
            // building the tables 2^bits x dim multiply-adds ~ 0.25 ns per codeword per 3 slices
            else if g.acodes.len() * 8 < (1usize << bits) * 3 && crate::pq_direct::sets(*sub, *m).is_some() { Plan::PqDirect }
            else { Plan::PqLut }
        }
    }
}

/// T[k][v] for TurboQuant group tables, in float64 then rounded once: a group's
/// index splits into two halves of gbits/2 bits, so an entry is the sum of a
/// low-half and a high-half partial (2 * 2^(gbits/2) products per group
/// instead of 2^gbits * gdims).
pub fn build_tq_tables(levels: &[f32], bits: usize, gbits: usize, gdims: usize, qn: &[f64], out: &mut Vec<f32>) {
    let groups = qn.len() / gdims;
    let half = gbits / 2;
    let hdims = gdims / 2;
    let (hn, mask) = (1usize << half, (1usize << bits) - 1);
    out.resize(groups << gbits, 0.0);
    let (mut lo, mut hi) = (vec![0f64; hn], vec![0f64; hn]);
    for k in 0..groups {
        let q = &qn[k * gdims..(k + 1) * gdims];
        for v in 0..hn {
            let (mut a, mut b) = (0.0, 0.0);
            for j in 0..hdims {
                a += levels[(v >> (bits * j)) & mask] as f64 * q[j];
                b += levels[(v >> (bits * j)) & mask] as f64 * q[hdims + j];
            }
            lo[v] = a;
            hi[v] = b;
        }
        let t = &mut out[k << gbits..(k + 1) << gbits];
        for h in 0..hn {
            for l in 0..hn { t[h * hn + l] = (lo[l] + hi[h]) as f32; }
        }
    }
}

/// Sum over a group's dimensions of levels^2 for every table index.
fn tq_norm_table(levels: &[f32], bits: usize, gbits: usize, gdims: usize) -> Vec<f32> {
    let mask = (1usize << bits) - 1;
    (0..1usize << gbits).map(|v| {
        (0..gdims).map(|j| { let l = levels[(v >> (bits * j)) & mask] as f64; l * l }).sum::<f64>() as f32
    }).collect()
}

/// T[c][s] = q's slice s . codeword c (codeword-major: for each codeword one
/// pass over all slices, as an axpy over s with the query transposed to
/// qt[t][s], so the 2^bits x sub x m multiply-adds vectorize). Entries are
/// float32 sums of `sub` float32 products: within sub * 2^-24 of the exact
/// slice dot product (|q_s| |c|), so a cap's sum of m entries is within
/// (sub + m) 2^-24 |raw| for unit q, inside slack.
pub fn build_pq_tables(codebook: &[f32], sub: usize, q32: &[f32], qt: &mut Vec<f32>, out: &mut Vec<f32>) {
    let words = codebook.len() / sub;
    let m = q32.len() / sub;
    qt.resize(sub * m, 0.0);
    for s in 0..m { for t in 0..sub { qt[t * m + s] = q32[s * sub + t]; } }
    out.resize(words * m, 0.0);
    match m {
        8 => pq_rows::<8>(codebook, sub, qt, out),
        16 => pq_rows::<16>(codebook, sub, qt, out),
        24 => pq_rows::<24>(codebook, sub, qt, out),
        32 => pq_rows::<32>(codebook, sub, qt, out),
        64 => pq_rows::<64>(codebook, sub, qt, out),
        _ => {
            for (w, row) in codebook.chunks_exact(sub).zip(out.chunks_exact_mut(m)) {
                row.fill(0.0);
                for (t, &wt) in w.iter().enumerate() {
                    for (o, &qv) in row.iter_mut().zip(&qt[t * m..(t + 1) * m]) { *o += wt * qv; }
                }
            }
        }
    }
}

/// build_pq_tables for M slices: two codeword rows at a time, their M
/// accumulators each held in registers across the `sub` steps. (Measured:
/// the same loop over `[[f32; M]; R]` with R = 4 spills and is 2.5x slower.)
fn pq_rows<const M: usize>(codebook: &[f32], sub: usize, qt: &[f32], out: &mut [f32]) {
    for (w, o) in codebook.chunks_exact(2 * sub).zip(out.chunks_exact_mut(2 * M)) {
        let (mut a, mut b) = ([0f32; M], [0f32; M]);
        for t in 0..sub {
            let (wa, wb) = (w[t], w[sub + t]);
            let qr: &[f32; M] = qt[t * M..(t + 1) * M].try_into().unwrap();
            for j in 0..M { a[j] = wa.mul_add(qr[j], a[j]); b[j] = wb.mul_add(qr[j], b[j]); }
        }
        o[..M].copy_from_slice(&a);
        o[M..].copy_from_slice(&b);
    }
}

impl Index {
    pub fn new(dim: usize, grid: Grid) -> Index {
        let top = grid.levels as usize;
        let shift = (usize::BITS - top.leading_zeros()).saturating_sub(8);
        let buckets = (0..=top >> shift).map(|b| {
            let t = grid.theta((((b + 1) << shift) - 1).min(top) as u16);
            (t, t.cos(), t.sin())
        }).collect();
        let cos_sin = if grid.levels <= 255.0 {
            (0..=top).map(|a| { let t = grid.theta(a as u16); (t.cos(), t.sin()) }).collect()
        } else { vec![] };
        Index { dim, groups: vec![], grid, buckets, shift, cos_sin }
    }

    /// (code bytes, angle-code bytes, codebook/table bytes).
    pub fn owned_bytes(&self) -> (usize, usize, usize) {
        let mut o = (0, 0, self.buckets.len() * 24);
        for g in &self.groups {
            o.0 += g.codes.len();
            o.1 += g.acodes.len() * 2;
            o.2 += g.norms.len() * 4 + match &g.codec { Codec::Tq { levels, .. } => levels.len() * 4, Codec::Pq { codebook, .. } => codebook.len() * 4 };
        }
        o
    }

    pub fn num_caps(&self) -> usize { self.groups.iter().map(|g| g.acodes.len()).sum() }

    /// Normalizes one working-basis query in place; returns its norm. (The
    /// rotation into the working basis is done by the caller in float64 BLAS:
    /// 25 us per query as a plain loop here, 4 us there.)
    fn normalize(q: &mut [f64]) -> f64 {
        let n = q.iter().map(|v| v * v).sum::<f64>().sqrt();
        let inv = n.max(1e-300);
        q.iter_mut().for_each(|u| *u /= inv);
        n
    }

    /// Re-decides one pair in float64 from the decoded vector.
    fn recheck(&self, g: &Group, k: usize, qn: &[f64], theta: f64, b: f64, raw: &mut [f32]) -> bool {
        let code = &g.codes[k * g.code_size..(k + 1) * g.code_size];
        g.codec.decode(code, self.dim, raw);
        let n = raw.iter().map(|&v| v as f64 * v as f64).sum::<f64>().sqrt();
        exact_angle(qn, raw, n) <= theta + b
    }

    /// Whether some cap of `g` accepts the query (unit `qn`, expansion `b`).
    /// `score(code)` returns (q . raw, |raw|^2) in float32.
    ///
    /// Gate: the exact test is dot >= |raw| cos(theta + beta) up to +-margin,
    /// margin = slack * |raw|. With kl = cos(theta + beta) - slack > 0 a cap
    /// can only matter if dot > |raw| kl, i.e. dot > 0 and dot^2 > |raw|^2 kl^2:
    /// no square root, no trig for the (common) caps that fail it; kl^2 is
    /// tabulated per angle bucket (`gate2`). cos(theta + beta) falls with theta,
    /// so a bucket's value at its upper end bounds all its codes' kl from
    /// below: a coarse bucket (16-bit grids) only lets more caps through to
    /// the exact test, never fewer; -inf where kl <= 0 or theta + beta might
    /// reach pi (the exact test handles those). Rounding in the gate (a few
    /// 2^-24) is far inside the slack not spent by the score's error
    /// (< 1.5 d 2^-24 of slack's 2 (d + 4) 2^-24).
    fn scan<F: Fn(&[u8]) -> (f32, f32)>(&self, g: &Group, qn: &[f64], b: f64, sc: &mut Scratch, score: F) -> bool {
        let (cb, sb) = (b.cos(), b.sin());
        let slack = slack(self.dim);
        let cs = g.code_size;
        for (g2, &(t, ct, st)) in sc.gate2.iter_mut().zip(&self.buckets) {
            let kl = cb * ct - sb * st - slack;
            *g2 = if kl > 0.0 && t + b < PI { (kl * kl) as f32 } else { f32::NEG_INFINITY };
        }
        for (k, (&ac, code)) in g.acodes.iter().zip(g.codes.chunks_exact(cs)).enumerate() {
            let (dot, n2) = score(code);
            let d = dot.max(0.0);
            if d * d <= n2 * sc.gate2[(ac >> self.shift) as usize] { continue; }
            let theta = self.grid.theta(ac);
            if theta + b >= PI { return true; }
            let kappa = cb * theta.cos() - sb * theta.sin();
            let n = (n2 as f64).sqrt();
            let thr = n * kappa;
            let margin = slack * n;
            let dot = dot as f64;
            if dot >= thr + margin { return true; }
            if dot > thr - margin && self.recheck(g, k, qn, theta, b, &mut sc.raw) { return true; }
        }
        false
    }

    /// Whether some cap accepts the (already rotated, normalized) query.
    fn accepts(&self, qn: &[f64], b: f64, sc: &mut Scratch) -> bool {
        for (o, &v) in sc.q32.iter_mut().zip(qn) { *o = v as f32; }
        for (gi, g) in self.groups.iter().enumerate() {
            let hit = match (&g.codec, &g.plan) {
                (Codec::Tq { .. }, _) if g.soa => crate::fastscan::accepts(self, gi, g, qn, b, &mut sc.fs, &self.cos_sin, &mut sc.raw),
                (Codec::Tq { bits, levels }, Plan::TqLut { gbits, gdims }) => {
                    build_tq_tables(levels, *bits, *gbits, *gdims, qn, &mut sc.table);
                    let t = std::mem::take(&mut sc.table);
                    let r = self.scan(g, qn, b, sc, |code| lut_score(code, *gbits, &t, &g.norms));
                    sc.table = t;
                    r
                }
                (Codec::Pq { bits, m, sub, codebook }, Plan::PqLut) => {
                    build_pq_tables(codebook, *sub, &sc.q32, &mut sc.qt, &mut sc.table);
                    let t = std::mem::take(&mut sc.table);
                    let r = self.scan(g, qn, b, sc, |code| pq_score(code, *bits, *m, &t, &g.norms));
                    sc.table = t;
                    r
                }
                (Codec::Pq { bits, m, sub, codebook }, Plan::PqDirect) => {
                    let (bits, m, nt) = (*bits, *m, &g.norms);
                    use crate::pq_direct::score as ps;
                    let q32 = std::mem::take(&mut sc.q32);
                    let r = match *sub {
                        6 => self.scan(g, qn, b, sc, |code| ps::<6, 4>(code, bits, m, codebook, nt, &q32)),
                        12 => self.scan(g, qn, b, sc, |code| ps::<12, 4>(code, bits, m, codebook, nt, &q32)),
                        16 => self.scan(g, qn, b, sc, |code| ps::<16, 2>(code, bits, m, codebook, nt, &q32)),
                        24 => self.scan(g, qn, b, sc, |code| ps::<24, 2>(code, bits, m, codebook, nt, &q32)),
                        _ => self.scan(g, qn, b, sc, |code| ps::<48, 1>(code, bits, m, codebook, nt, &q32)),
                    };
                    sc.q32 = q32;
                    r
                }
                (codec, _) => self.scan(g, qn, b, sc, |code| codec.score_f64(code, qn)),
            };
            if hit { return true; }
        }
        false
    }

    fn scratch(&self) -> Scratch {
        Scratch { gate2: vec![0f32; self.buckets.len()], table: vec![], q32: vec![0f32; self.dim], qt: vec![], raw: vec![0f32; self.dim], fs: crate::fastscan::Fs::new(self) }
    }

    /// Answers for working-basis queries `x` (rows of dim float64) into `out`.
    fn answer(&self, x: &[f64], tp: f64, out: &mut [bool]) {
        let d = self.dim;
        let mut unit = vec![0.0; d];
        let mut sc = self.scratch();
        for (row, o) in x.chunks_exact(d).zip(out.iter_mut()) {
            unit.copy_from_slice(row);
            let n = Self::normalize(&mut unit);
            *o = match beta(n, tp) { Some(b) => self.accepts(&unit, b, &mut sc), None => false };
        }
        sc.fs.flush_stats();
    }

    /// Answers for a batch of working-basis queries (nq x dim, float64), threshold tp = tau'.
    /// With a pool, chunks of `QUERY_CHUNK` queries are spread over its threads
    /// (query costs differ a lot: an accepted query stops early).
    pub fn query(&self, x: &[f64], tp: f64, pool: Option<&rayon::ThreadPool>) -> Vec<bool> {
        let d = self.dim;
        let mut out = vec![false; x.len() / d];
        match pool {
            None => self.answer(x, tp, &mut out),
            Some(pool) => pool.install(|| {
                use rayon::prelude::*;
                x.par_chunks(d * QUERY_CHUNK).zip(out.par_chunks_mut(QUERY_CHUNK))
                    .for_each(|(xc, oc)| self.answer(xc, tp, oc));
            }),
        }
        out
    }
}

/// Queries per parallel task: small enough to balance, large enough that a
/// task's scratch allocation is amortized.
const QUERY_CHUNK: usize = 4;

/// Eight running (dot, |raw|^2) sums, one per lane, so the adds of a cap's
/// table reads do not wait on each other. Lanes stay in registers because
/// every call indexes them with a constant.
struct Lanes { dot: [f32; 8], n2: [f32; 8] }

impl Lanes {
    #[inline(always)]
    fn new() -> Lanes { Lanes { dot: [0.0; 8], n2: [0.0; 8] } }
    #[inline(always)]
    fn sum(&self) -> (f32, f32) { (self.dot.iter().sum(), self.n2.iter().sum()) }
}

/// (q . raw, |raw|^2) of a TurboQuant cap from its group tables. float32
/// sums of `groups` table entries, each within 2^-24 of the exact group
/// partial: total error below groups * 2^-24 * |raw| for unit q (Cauchy-
/// Schwarz over groups), inside slack. Group counts are multiples of 8 at
/// d = 384; a ragged tail is added on lane 0.
#[inline]
fn lut_score(code: &[u8], gbits: usize, t: &[f32], nt: &[f32]) -> (f32, f32) {
    let mut l = Lanes::new();
    if gbits == 8 {
        let (tabs, _) = t.as_chunks::<256>();
        let nt: &[f32; 256] = nt.try_into().unwrap();
        let (cc, ct) = (code.chunks_exact(8), tabs.chunks_exact(8));
        for (c, tb) in cc.zip(ct) {
            for j in 0..8 {
                l.dot[j] += tb[j][c[j] as usize];
                l.n2[j] += nt[c[j] as usize];
            }
        }
    } else {
        // 6-bit indices: 8 per 6 bytes
        let (tabs, _) = t.as_chunks::<64>();
        let nt: &[f32; 64] = nt.try_into().unwrap();
        for (c, tb) in code.chunks_exact(6).zip(tabs.chunks_exact(8)) {
            let w = u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], 0, 0]);
            for j in 0..8 {
                let v = ((w >> (6 * j)) & 63) as usize;
                l.dot[j] += tb[j][v];
                l.n2[j] += nt[v];
            }
        }
    }
    l.sum()
}

/// (q . raw, |raw|^2) of a seeded-PQ cap from its tables T[c][s] (m a multiple of 8).
#[inline]
fn pq_score(code: &[u8], bits: usize, m: usize, t: &[f32], nt: &[f32]) -> (f32, f32) {
    let mut l = Lanes::new();
    if bits == 8 {
        for (c, cc) in code.chunks_exact(8).enumerate() {
            for j in 0..8 {
                l.dot[j] += t[cc[j] as usize * m + c * 8 + j];
                l.n2[j] += nt[cc[j] as usize];
            }
        }
    } else {
        // 12-bit: 8 codes per 12 bytes
        for (c, cc) in code.chunks_exact(12).enumerate() {
            let lo = u64::from_le_bytes([cc[0], cc[1], cc[2], cc[3], cc[4], cc[5], cc[6], cc[7]]);
            let hi = u32::from_le_bytes([cc[8], cc[9], cc[10], cc[11]]) as u64;
            let w = lo as u128 | (hi as u128) << 64;
            for j in 0..8 {
                let v = ((w >> (12 * j)) & 4095) as usize;
                l.dot[j] += t[v * m + c * 8 + j];
                l.n2[j] += nt[v];
            }
        }
    }
    l.sum()
}
