//! Fuzz of the Rust decision against a plain float64 reference of the accept rule
//! (decode every cap, angle by the chord formula, compare with theta + beta), on random
//! caps/queries including queries placed within 1e-9..1e-3 of a cap's boundary,
//! non-unit norms, and norms beyond the feasibility limit. Pairs within 1e-9 of the
//! boundary are don't-care (the reference's own float64 error is larger than that).
use capcover_rs::core::*;
use proptest::prelude::*;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn f(&mut self) -> f64 { (self.next() >> 11) as f64 / (1u64 << 53) as f64 }          // [0, 1)
    fn g(&mut self) -> f64 { (self.f() + self.f() + self.f() + self.f() - 2.0) * 1.7 }      // ~N(0,1)
}

fn unit(v: &mut [f64]) { let n = v.iter().map(|x| x * x).sum::<f64>().sqrt(); v.iter_mut().for_each(|x| *x /= n); }

/// Decoded raw vector of cap k of group g (reference: reads the bit stream bit by bit).
fn raw(g: &Group, k: usize, dim: usize) -> Vec<f32> {
    let mut code = vec![0u8; g.code_size];
    g.code_into(k, &mut code);
    let bit = |i: usize| (code[i >> 3] >> (i & 7)) & 1;
    let field = |j: usize, bits: usize| (0..bits).map(|b| (bit(j * bits + b) as usize) << b).sum::<usize>();
    match &g.codec {
        Codec::Tq { bits, levels } => (0..dim).map(|i| levels[field(i, *bits)]).collect(),
        Codec::Pq { bits, m, sub, codebook } => (0..*m).flat_map(|j| {
            let c = field(j, *bits);
            codebook[c * sub..(c + 1) * sub].to_vec()
        }).collect(),
    }
}

fn chord_angle(q: &[f64], u: &[f64]) -> f64 {
    let s: f64 = q.iter().zip(u).map(|(a, b)| (a - b) * (a - b)).sum();
    2.0 * (0.5 * s.sqrt()).min(1.0).asin()
}

/// min over caps of angle(q, u_j) - theta_j - beta (inf if infeasible): < 0 accepts.
fn margin(idx: &Index, dim: usize, x: &[f64], tp: f64) -> f64 {
    let n = x.iter().map(|v| v * v).sum::<f64>().sqrt();
    let b = match beta(n, tp) { Some(b) => b, None => return f64::INFINITY };
    let q: Vec<f64> = x.iter().map(|v| v / n).collect();
    let mut best = f64::INFINITY;
    for g in &idx.groups {
        for (k, &ac) in g.acodes.iter().enumerate() {
            let r = raw(g, k, dim);
            let nr = r.iter().map(|&v| v as f64 * v as f64).sum::<f64>().sqrt();
            let u: Vec<f64> = r.iter().map(|&v| v as f64 / nr).collect();
            best = best.min(chord_angle(&q, &u) - idx.grid.theta(ac) - b);
        }
    }
    best
}

#[derive(Clone, Debug)]
enum Kind { Tq(usize), Pq(usize, usize) }   // bits | (bits, sub)

fn build(kind: &Kind, dim: usize, caps: usize, grid_bits: u32, rng: &mut Rng) -> Index {
    let (lo, hi) = (0.2 + 0.3 * rng.f(), 0.8 + 0.5 * rng.f());
    let levels = ((1u32 << grid_bits) - 1) as f64;
    let mut idx = Index::new(dim, Grid { lo, hi, levels });
    let (codec, code_size) = match *kind {
        Kind::Tq(bits) => {
            let mut lv: Vec<f32> = (0..1 << bits).map(|_| rng.g() as f32 * 0.3).collect();
            lv.sort_by(|a, b| a.partial_cmp(b).unwrap());
            (Codec::Tq { bits, levels: lv }, (dim * bits + 7) / 8)
        }
        Kind::Pq(bits, sub) => {
            let mut cb: Vec<f32> = (0..(1usize << bits) * sub).map(|_| rng.g() as f32).collect();
            for c in cb.chunks_mut(sub) { let n = c.iter().map(|v| v * v).sum::<f32>().sqrt(); c.iter_mut().for_each(|v| *v /= n); }
            let m = dim / sub;
            (Codec::Pq { bits, m, sub, codebook: cb }, (m * bits + 7) / 8)
        }
    };
    let codes = (0..caps * code_size).map(|_| rng.next() as u8).collect();
    let acodes = (0..caps).map(|_| (rng.next() % (levels as u64 + 1)) as u16).collect();
    idx.groups.push(Group::new(codec, dim, code_size, codes, acodes));
    idx
}

fn queries(idx: &Index, dim: usize, tp: f64, rng: &mut Rng) -> Vec<f64> {
    let mut out = vec![];
    let g = &idx.groups[0];
    for i in 0..60 {
        let k = (rng.next() as usize) % g.acodes.len();
        let r = raw(g, k, dim);
        let mut u: Vec<f64> = r.iter().map(|&v| v as f64).collect();
        unit(&mut u);
        let mut v: Vec<f64> = (0..dim).map(|_| rng.g()).collect();
        let d: f64 = v.iter().zip(&u).map(|(a, b)| a * b).sum();
        v.iter_mut().zip(&u).for_each(|(a, b)| *a -= d * b);
        unit(&mut v);
        // norm: mostly 1, sometimes off by up to a bit past the feasibility limit
        let norm = match i % 4 { 0 => 1.0, 1 => 1.0 + (rng.f() - 0.5) * 1.8 * tp, 2 => 1.0 + (rng.f() - 0.5) * 3.0 * tp, _ => 0.5 + rng.f() };
        let b = beta(norm, tp);
        let delta = [-1e-3, -1e-5, -1e-7, -1e-9, 1e-9, 1e-7, 1e-5, 1e-3][i % 8];
        let phi = b.map(|b| idx.grid.theta(g.acodes[k]) + b + delta).unwrap_or(1.0);
        for j in 0..dim { out.push(norm * (phi.cos() * u[j] + phi.sin() * v[j])); }
    }
    for _ in 0..20 {
        let mut x: Vec<f64> = (0..dim).map(|_| rng.g()).collect();
        unit(&mut x);
        out.extend(x);
    }
    out
}

fn check(kind: Kind, dim: usize, caps: usize, grid_bits: u32, seed: u64) -> Result<(), TestCaseError> {
    if std::env::var("FS_OFF").is_ok() { capcover_rs::fastscan::ENABLED.store(false, std::sync::atomic::Ordering::Relaxed); }
    let mut rng = Rng(seed | 1);
    let idx = build(&kind, dim, caps, grid_bits, &mut rng);
    let tp = [0.05, 0.1, 0.3, 0.7][(rng.next() % 4) as usize] + 1.1e-3;
    let q = queries(&idx, dim, tp, &mut rng);
    let got = idx.query(&q, tp, None);
    for (i, row) in q.chunks(dim).enumerate() {
        let m = margin(&idx, dim, row, tp);
        prop_assert!(!(m < -1e-9 && !got[i]), "false negative: margin {m}, query {i}, {kind:?}");
        prop_assert!(!(m > 1e-9 && got[i]), "false positive: margin {m}, query {i}, {kind:?}");
    }
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(60))]
    #[test]
    fn tq(bits in prop::sample::select(vec![1usize, 2, 3, 4, 8]), caps in 1usize..120, grid16 in any::<bool>(), seed in any::<u64>()) {
        check(Kind::Tq(bits), 64, caps, if grid16 { 16 } else { 8 }, seed)?;
    }
    #[test]
    fn pq(cfg in prop::sample::select(vec![(8usize, 8usize), (8, 6), (8, 12), (12, 12), (12, 16), (12, 8), (12, 24)]),
          caps in 1usize..120, grid16 in any::<bool>(), seed in any::<u64>()) {
        let sub = cfg.1;
        check(Kind::Pq(cfg.0, sub), 96, caps, if grid16 { 16 } else { 8 }, seed)?;
    }
    /// Enough caps that the few-caps threshold sends 12-bit (>= 1536) and 8-bit (>= 96) groups to their tables.
    #[test]
    fn pq_tables(seed in any::<u64>()) {
        check(Kind::Pq(12, 12), 96, 1700, 8, seed)?;
        check(Kind::Pq(8, 12), 96, 150, 16, seed)?;
    }
}

#[test]
#[ignore]
fn dbg_seed() {
    let seed: u64 = std::env::var("SEED").unwrap().parse().unwrap();
    let (dim, caps) = (64, 16);
    let mut rng = Rng(seed | 1);
    let idx = build(&Kind::Tq(2), dim, caps, 8, &mut rng);
    let tp = [0.05, 0.1, 0.3, 0.7][(rng.next() % 4) as usize] + 1.1e-3;
    let q = queries(&idx, dim, tp, &mut rng);
    let got = idx.query(&q, tp, None);
    for (i, row) in q.chunks(dim).enumerate() {
        let m = margin(&idx, dim, row, tp);
        if (m < -1e-9 && !got[i]) { 
            let g = &idx.groups[0];
            eprintln!("query {i} margin {m} tp {tp} levels {:?}", match &g.codec { Codec::Tq{levels,..} => levels.clone(), _ => vec![] });
            let n = row.iter().map(|v| v * v).sum::<f64>().sqrt();
            eprintln!("norm {n} beta {:?} grid {} {} {}", beta(n, tp), idx.grid.lo, idx.grid.hi, idx.grid.levels);
            for k in 0..caps { let r = raw(g, k, dim); let nr = r.iter().map(|&v| v as f64 * v as f64).sum::<f64>().sqrt();
                let u: Vec<f64> = r.iter().map(|&v| v as f64 / nr).collect(); let q: Vec<f64> = row.iter().map(|v| v / n).collect();
                eprintln!("cap {k} nr {nr:.4} ac {} theta {:.4} ang {:.6}", g.acodes[k], idx.grid.theta(g.acodes[k]), chord_angle(&q, &u)); }
            break;
        }
    }
}

#[test]
fn full_block_reference_reads_blocked_codes() { check(Kind::Tq(2), 64, 16, 8, 37550337053380924).unwrap(); }
