//! AVX2+FMA exact top-k inner-product search (the `knn_ip` path for CPUs without AVX-512).
//!
//! A register-tiled float32 GEMM (6 queries x 16 directions, 12 ymm accumulators; the micro-kernel
//! of `construct_minimax::assign_group_avx2`) fused with a per-query running top-k: after each
//! tile a single vector compare against the query's current k-th best score decides whether any of
//! its 16 scores can enter the top-k; only then does a scalar insertion run. The n x m score matrix
//! is never materialised. Directions are packed into 16-wide panels; queries are processed in
//! chunks of CHUNK rows so the chunk stays in L2 while panels stream past.
//!
//! Every kernel here is compiled with `#[target_feature(enable = "avx2,fma")]` only, so the
//! compiler cannot emit an AVX-512 instruction. Ties: a score only enters the top-k when it is
//! strictly greater than the current k-th best, so among equal scores the lowest index wins.
use rayon::prelude::*;
use std::arch::x86_64::*;

const MR: usize = 6;
const NR: usize = 16;
/// Queries per chunk (multiple of MR): chunk rows (CHUNK*d*4 bytes) stay resident in L2.
const CHUNK: usize = 96;

pub fn available() -> bool {
    std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
}

fn pack_panels(u: &[f32], m: usize, d: usize) -> Vec<f32> {
    let np = m.div_ceil(NR);
    let mut out = vec![0f32; np * d * NR];
    out.par_chunks_mut(d * NR).enumerate().for_each(|(p, panel)| {
        for j in 0..NR.min(m - p * NR) {
            let row = &u[(p * NR + j) * d..(p * NR + j + 1) * d];
            for kk in 0..d {
                panel[kk * NR + j] = row[kk];
            }
        }
    });
    out
}

/// Top-k of `nq` query rows `x` (stride d) against all panels; results (descending) into ip/id.
#[target_feature(enable = "avx2,fma")]
unsafe fn search_chunk(x: *const f32, nq: usize, d: usize, panels: &[f32], m: usize, k: usize,
                       ip: &mut [f32], id: &mut [i64]) {
    let np = m.div_ceil(NR);
    let neg_inf = f32::NEG_INFINITY;
    let mut tv = vec![neg_inf; nq * k];
    let mut ti = vec![-1i64; nq * k];
    let mut thr = vec![neg_inf; nq];
    let ntile = nq.div_ceil(MR);
    for p in 0..np {
        let panel = panels.as_ptr().add(p * d * NR);
        let base = p * NR;
        for t in 0..ntile {
            let r0 = t * MR;
            let mut rows = [std::ptr::null::<f32>(); MR];
            for i in 0..MR {
                rows[i] = x.add((r0 + i).min(nq - 1) * d);
            }
            let mut c = [_mm256_setzero_ps(); 2 * MR];
            for kk in 0..d {
                let b0 = _mm256_loadu_ps(panel.add(kk * NR));
                let b1 = _mm256_loadu_ps(panel.add(kk * NR + 8));
                for i in 0..MR {
                    let a = _mm256_broadcast_ss(&*rows[i].add(kk));
                    c[2 * i] = _mm256_fmadd_ps(a, b0, c[2 * i]);
                    c[2 * i + 1] = _mm256_fmadd_ps(a, b1, c[2 * i + 1]);
                }
            }
            for i in 0..MR.min(nq - r0) {
                let r = r0 + i;
                let tvv = _mm256_set1_ps(thr[r]);
                let g0 = _mm256_movemask_ps(_mm256_cmp_ps(c[2 * i], tvv, _CMP_GT_OQ)) as u32;
                let g1 = _mm256_movemask_ps(_mm256_cmp_ps(c[2 * i + 1], tvv, _CMP_GT_OQ)) as u32;
                let mut mask = g0 | (g1 << 8);
                if mask == 0 {
                    continue;
                }
                let mut v = [0f32; NR];
                _mm256_storeu_ps(v.as_mut_ptr(), c[2 * i]);
                _mm256_storeu_ps(v.as_mut_ptr().add(8), c[2 * i + 1]);
                let (sv, si) = (&mut tv[r * k..(r + 1) * k], &mut ti[r * k..(r + 1) * k]);
                let mut th = thr[r];
                while mask != 0 {
                    let l = mask.trailing_zeros() as usize;
                    mask &= mask - 1;
                    let j = base + l;
                    if j >= m {
                        break; // zero padding of the last panel
                    }
                    if v[l] > th {
                        // replace the current minimum (last one among equals), refresh the threshold
                        let mut w = 0;
                        for s in 1..k {
                            if sv[s] <= sv[w] {
                                w = s;
                            }
                        }
                        sv[w] = v[l];
                        si[w] = j as i64;
                        th = sv[0];
                        for s in 1..k {
                            if sv[s] < th {
                                th = sv[s];
                            }
                        }
                    }
                }
                thr[r] = th;
            }
        }
    }
    for r in 0..nq {
        let mut ord: Vec<usize> = (0..k).collect();
        let (sv, si) = (&tv[r * k..(r + 1) * k], &ti[r * k..(r + 1) * k]);
        ord.sort_unstable_by(|&a, &b| sv[b].partial_cmp(&sv[a]).unwrap_or(std::cmp::Ordering::Equal).then(si[a].cmp(&si[b])));
        for (t, &o) in ord.iter().enumerate() {
            ip[r * k + t] = sv[o];
            id[r * k + t] = si[o];
        }
    }
}

/// (ip, idx) n x k, descending. Requires all inputs finite, 1 <= k <= m, d >= 1.
pub fn knn(xs: &[f32], n: usize, ds: &[f32], m: usize, d: usize, k: usize, threads: usize) -> (Vec<f32>, Vec<i64>) {
    let panels = pack_panels(ds, m, d);
    let mut ip = vec![0f32; n * k];
    let mut id = vec![0i64; n * k];
    let run = |b: usize, ipc: &mut [f32], idc: &mut [i64]| unsafe {
        let r0 = b * CHUNK;
        let nq = CHUNK.min(n - r0);
        search_chunk(xs.as_ptr().add(r0 * d), nq, d, &panels, m, k, ipc, idc);
    };
    if threads <= 1 {
        for (b, (ipc, idc)) in ip.chunks_mut(CHUNK * k).zip(id.chunks_mut(CHUNK * k)).enumerate() {
            run(b, ipc, idc);
        }
    } else if let Ok(pool) = rayon::ThreadPoolBuilder::new().num_threads(threads).build() {
        pool.install(|| {
            ip.par_chunks_mut(CHUNK * k).zip(id.par_chunks_mut(CHUNK * k)).enumerate()
                .for_each(|(b, (ipc, idc))| run(b, ipc, idc));
        });
    }
    (ip, id)
}
