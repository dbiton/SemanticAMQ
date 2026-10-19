// Native timing of full scans on synthetic caps (all rejected: random codes, narrow caps).
// cargo run --release --bin microbench
use capcover_rs::core::*;
use std::time::Instant;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn f(&mut self) -> f64 { (self.next() >> 11) as f64 / (1u64 << 53) as f64 - 0.5 }
}

fn index(kind: &str, bits: usize, sub: usize, caps: usize, rng: &mut Rng) -> Index {
    let dim = 384;
    let codec = if kind == "tq" {
        Codec::Tq { bits, levels: (0..1 << bits).map(|i| (i as f32 + 0.5 - (1 << bits) as f32 / 2.0) * 0.1 / (1 << bits) as f32 * 4.0).collect() }
    } else {
        let mut cb: Vec<f32> = (0..(1usize << bits) * sub).map(|_| rng.f() as f32).collect();
        for c in cb.chunks_mut(sub) { let n = c.iter().map(|v| v * v).sum::<f32>().sqrt(); c.iter_mut().for_each(|v| *v /= n); }
        Codec::Pq { bits, m: dim / sub, sub, codebook: cb }
    };
    let code_size = match &codec { Codec::Tq { bits, .. } => dim * bits / 8, Codec::Pq { bits, m, .. } => m * bits / 8 };
    let codes = (0..caps * code_size).map(|_| rng.next() as u8).collect();
    let acodes = (0..caps).map(|_| (rng.next() % 40) as u16 + 100).collect();
    let mut idx = Index::new(dim, Grid { lo: 0.0, hi: 1.0, levels: 255.0 });
    idx.groups.push(Group::new(codec, dim, code_size, codes, acodes));
    idx
}

fn main() {
    time_build();
    let mut rng = Rng(88172645463325252);
    let caps = 3000;
    for &(kind, bits, sub) in &[("tq", 1usize, 0usize), ("tq", 2, 0), ("tq", 3, 0), ("tq", 4, 0), ("pq", 8, 24), ("pq", 8, 6), ("pq", 12, 12), ("pq", 12, 16), ("pq", 12, 24), ("pq", 12, 48)] {
        let idx = index(kind, bits, sub, caps, &mut rng);
        let nq = 300;
        let mut x: Vec<f64> = (0..nq * 384).map(|_| rng.f()).collect();
        for r in x.chunks_mut(384) { let n = r.iter().map(|v| v * v).sum::<f64>().sqrt(); r.iter_mut().for_each(|v| *v /= n); }
        idx.query(&x[..384], 0.01, None);
        let t0 = Instant::now();
        let hits = idx.query(&x, 0.01, None).iter().filter(|&&b| b).count();
        let dt = t0.elapsed().as_secs_f64();
        println!("{kind}{bits} sub{sub}: {:.1} us/query, {:.1} ns/cap ({hits} accepted of {nq})", dt / nq as f64 * 1e6, dt / (nq * caps) as f64 * 1e9);
    }
}

#[allow(dead_code)]
pub fn time_build() {
    let mut rng = Rng(12345);
    for &(bits, sub) in &[(8usize, 24usize), (12, 12), (12, 16), (12, 24), (12, 48)] {
        let cb: Vec<f32> = (0..(1usize << bits) * sub).map(|_| rng.f() as f32).collect();
        let q: Vec<f32> = (0..384).map(|_| rng.f() as f32).collect();
        let (mut qt, mut out) = (vec![], vec![]);
        let t0 = Instant::now();
        for _ in 0..200 { build_pq_tables(&cb, sub, &q, &mut qt, &mut out); }
        println!("build pq{bits} sub{sub}: {:.1} us", t0.elapsed().as_secs_f64() / 200.0 * 1e6);
    }
}
