//! Seeded-PQ scoring straight from the codebook, no tables: per cap, each
//! slice's codeword row is multiplied into the query's slice and summed.
//! The accumulators are elementwise (one per slice position, reduced once per
//! cap) so everything vectorizes; SETS independent sets hide the fma latency.
//! Error: float32 sums of d products of float32-rounded operands, the same
//! bound as the original GEMM (< (d + 4) 2^-24 |q| |raw|, inside slack).

/// Slice code s of a little-endian bit stream of 8- or 12-bit codes.
#[inline(always)]
fn slice_code(code: &[u8], s: usize, bits: usize) -> usize {
    if bits == 8 {
        code[s] as usize
    } else {
        let p = 3 * (s >> 1);
        if s & 1 == 0 { code[p] as usize | (code[p + 1] as usize & 15) << 8 }
        else { (code[p + 1] as usize >> 4) | (code[p + 2] as usize) << 4 }
    }
}

/// (q . raw, |raw|^2); `q` is the float32 query, `sqn` each codeword's squared norm.
#[inline]
pub fn score<const SUB: usize, const SETS: usize>(code: &[u8], bits: usize, m: usize, cb: &[f32], sqn: &[f32], q: &[f32]) -> (f32, f32) {
    let mut acc = [[0f32; SUB]; SETS];
    let mut n2 = [0f32; SETS];
    for s0 in (0..m).step_by(SETS) {
        for j in 0..SETS {
            let s = s0 + j;
            let c = slice_code(code, s, bits);
            let w = &cb[c * SUB..(c + 1) * SUB];
            let qs = &q[s * SUB..(s + 1) * SUB];
            for t in 0..SUB { acc[j][t] = w[t].mul_add(qs[t], acc[j][t]); }
            n2[j] += sqn[c];
        }
    }
    let mut dot = 0f32;
    for t in 0..SUB { for j in 0..SETS { dot += acc[j][t]; } }
    (dot, n2.iter().sum())
}

/// Number of independent accumulator sets for slices of `sub` dims, or None
/// if `sub`/`m` has no instantiation.
pub fn sets(sub: usize, m: usize) -> Option<usize> {
    let s = match sub { 6 | 12 => 4, 16 | 24 => 2, 48 => 1, _ => return None };
    if m % s == 0 { Some(s) } else { None }
}
