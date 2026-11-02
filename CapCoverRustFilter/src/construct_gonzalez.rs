//! Fused Gonzalez farthest-first kernel.
//!
//! Reference (gonzales_construct.py): best = X @ x0; repeat: c = first argmin(best);
//! stop if best[c] >= 1; best = max(best, X @ x_c). Here one pass over X per
//! iteration does the dot product, the max-update and the running (first-)argmin of
//! the *updated* best, which is the next iteration's pick. The pass is
//! memory-bandwidth-bound for large n (X is n*d*4 bytes), so the only savings over
//! BLAS+NumPy are the removed passes over the length-n vectors and the temporaries.
//!
//! Divergence from the reference: dot products use a different float32 summation
//! order than BLAS, so a value may differ in the last bit; the chosen sequence can
//! then differ only where two candidates' best-ip values are within a few ulp of
//! each other (exact ties between *identical* rows stay ties, broken to the
//! lowest index like argmin). Coverage is unaffected (angles come from float64).

use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Dot products of rows [r0, r1) with `c`, fused with the max-update of `best` and the
/// running argmin of the updated `best`. `first`: best is set to the dot (no max).
/// Returns (min value, its row), first minimum on ties.
type PassFn = unsafe fn(*const f32, usize, usize, usize, *const f32, *mut f32, bool, *mut u32, u32) -> (f32, usize);

#[inline(always)]
fn upd(best: &mut f32, ip: f32, first: bool, row: usize, mn: &mut f32, mi: &mut usize,
       lab: *mut u32, cur: u32) {
    let v = if first || ip > *best { ip } else { *best };
    if first || ip > *best {
        unsafe { *lab.add(row) = cur };
    }
    *best = v;
    if v < *mn {
        *mn = v;
        *mi = row;
    }
}

unsafe fn pass_scalar(x: *const f32, d: usize, r0: usize, r1: usize, c: *const f32,
                      best: *mut f32, first: bool, lab: *mut u32, cur: u32) -> (f32, usize) {
    let (mut mn, mut mi) = (f32::INFINITY, r0);
    for r in r0..r1 {
        let row = x.add(r * d);
        let mut s = [0f32; 8];
        let mut j = 0;
        while j + 8 <= d {
            for l in 0..8 {
                s[l] += *row.add(j + l) * *c.add(j + l);
            }
            j += 8;
        }
        let mut t = 0f32;
        while j < d {
            t += *row.add(j) * *c.add(j);
            j += 1;
        }
        let ip = ((s[0] + s[4]) + (s[1] + s[5])) + ((s[2] + s[6]) + (s[3] + s[7])) + t;
        upd(&mut *best.add(r), ip, first, r, &mut mn, &mut mi, lab, cur);
    }
    (mn, mi)
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::upd;
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx512f")]
    pub unsafe fn pass_avx512(x: *const f32, d: usize, r0: usize, r1: usize, c: *const f32,
                              best: *mut f32, first: bool, lab: *mut u32, cur: u32) -> (f32, usize) {
        let (mut mn, mut mi) = (f32::INFINITY, r0);
        let full = d / 16 * 16;
        let rem = d - full;
        let mask: __mmask16 = if rem == 0 { 0 } else { ((1u32 << rem) - 1) as u16 };
        let mut r = r0;
        while r + 4 <= r1 {
            let p0 = x.add(r * d);
            let p1 = p0.add(d);
            let p2 = p1.add(d);
            let p3 = p2.add(d);
            let (mut a0, mut a1, mut a2, mut a3) =
                (_mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps());
            let mut j = 0;
            while j < full {
                let cv = _mm512_loadu_ps(c.add(j));
                a0 = _mm512_fmadd_ps(_mm512_loadu_ps(p0.add(j)), cv, a0);
                a1 = _mm512_fmadd_ps(_mm512_loadu_ps(p1.add(j)), cv, a1);
                a2 = _mm512_fmadd_ps(_mm512_loadu_ps(p2.add(j)), cv, a2);
                a3 = _mm512_fmadd_ps(_mm512_loadu_ps(p3.add(j)), cv, a3);
                j += 16;
            }
            if rem != 0 {
                let cv = _mm512_maskz_loadu_ps(mask, c.add(full));
                a0 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(mask, p0.add(full)), cv, a0);
                a1 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(mask, p1.add(full)), cv, a1);
                a2 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(mask, p2.add(full)), cv, a2);
                a3 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(mask, p3.add(full)), cv, a3);
            }
            upd(&mut *best.add(r), _mm512_reduce_add_ps(a0), first, r, &mut mn, &mut mi, lab, cur);
            upd(&mut *best.add(r + 1), _mm512_reduce_add_ps(a1), first, r + 1, &mut mn, &mut mi, lab, cur);
            upd(&mut *best.add(r + 2), _mm512_reduce_add_ps(a2), first, r + 2, &mut mn, &mut mi, lab, cur);
            upd(&mut *best.add(r + 3), _mm512_reduce_add_ps(a3), first, r + 3, &mut mn, &mut mi, lab, cur);
            r += 4;
        }
        while r < r1 {
            let p = x.add(r * d);
            let mut a = _mm512_setzero_ps();
            let mut j = 0;
            while j < full {
                a = _mm512_fmadd_ps(_mm512_loadu_ps(p.add(j)), _mm512_loadu_ps(c.add(j)), a);
                j += 16;
            }
            if rem != 0 {
                a = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(mask, p.add(full)),
                                    _mm512_maskz_loadu_ps(mask, c.add(full)), a);
            }
            upd(&mut *best.add(r), _mm512_reduce_add_ps(a), first, r, &mut mn, &mut mi, lab, cur);
            r += 1;
        }
        (mn, mi)
    }

    #[inline(always)]
    unsafe fn hsum(v: __m256) -> f32 {
        let lo = _mm256_castps256_ps128(v);
        let hi = _mm256_extractf128_ps(v, 1);
        let s = _mm_add_ps(lo, hi);
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
        _mm_cvtss_f32(s)
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn pass_avx2(x: *const f32, d: usize, r0: usize, r1: usize, c: *const f32,
                            best: *mut f32, first: bool, lab: *mut u32, cur: u32) -> (f32, usize) {
        let (mut mn, mut mi) = (f32::INFINITY, r0);
        let full = d / 8 * 8;
        let mut r = r0;
        while r + 4 <= r1 {
            let p0 = x.add(r * d);
            let p1 = p0.add(d);
            let p2 = p1.add(d);
            let p3 = p2.add(d);
            let (mut a0, mut a1, mut a2, mut a3) =
                (_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps());
            let mut j = 0;
            while j < full {
                let cv = _mm256_loadu_ps(c.add(j));
                a0 = _mm256_fmadd_ps(_mm256_loadu_ps(p0.add(j)), cv, a0);
                a1 = _mm256_fmadd_ps(_mm256_loadu_ps(p1.add(j)), cv, a1);
                a2 = _mm256_fmadd_ps(_mm256_loadu_ps(p2.add(j)), cv, a2);
                a3 = _mm256_fmadd_ps(_mm256_loadu_ps(p3.add(j)), cv, a3);
                j += 8;
            }
            let mut t = [0f32; 4];
            for j in full..d {
                let cv = *c.add(j);
                t[0] += *p0.add(j) * cv;
                t[1] += *p1.add(j) * cv;
                t[2] += *p2.add(j) * cv;
                t[3] += *p3.add(j) * cv;
            }
            upd(&mut *best.add(r), hsum(a0) + t[0], first, r, &mut mn, &mut mi, lab, cur);
            upd(&mut *best.add(r + 1), hsum(a1) + t[1], first, r + 1, &mut mn, &mut mi, lab, cur);
            upd(&mut *best.add(r + 2), hsum(a2) + t[2], first, r + 2, &mut mn, &mut mi, lab, cur);
            upd(&mut *best.add(r + 3), hsum(a3) + t[3], first, r + 3, &mut mn, &mut mi, lab, cur);
            r += 4;
        }
        while r < r1 {
            let p = x.add(r * d);
            let mut a = _mm256_setzero_ps();
            let mut j = 0;
            while j < full {
                a = _mm256_fmadd_ps(_mm256_loadu_ps(p.add(j)), _mm256_loadu_ps(c.add(j)), a);
                j += 8;
            }
            let mut t = 0f32;
            for j in full..d {
                t += *p.add(j) * *c.add(j);
            }
            upd(&mut *best.add(r), hsum(a) + t, first, r, &mut mn, &mut mi, lab, cur);
            r += 1;
        }
        (mn, mi)
    }

    /// One row's exact float32 dot (same lane order as pass_avx512's single-row tail).
    #[target_feature(enable = "avx512f")]
    unsafe fn row_dot(p: *const f32, c: *const f32, full: usize, rem: usize, mask: __mmask16) -> f32 {
        let mut a = _mm512_setzero_ps();
        let mut j = 0;
        while j < full {
            a = _mm512_fmadd_ps(_mm512_loadu_ps(p.add(j)), _mm512_loadu_ps(c.add(j)), a);
            j += 16;
        }
        if rem != 0 {
            a = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(mask, p.add(full)), _mm512_maskz_loadu_ps(mask, c.add(full)), a);
        }
        _mm512_reduce_add_ps(a)
    }

    /// Screened pass. xq: u8 rows (value + 128), stride dpad (multiple of 64), zero-padded;
    /// cq: i8 center (same padding); corr = 128 * sum(cq). A row is skipped (best unchanged)
    /// when sx*sc*dot_q + eps[i] <= best[i], eps[i] = rigorous bound on |x.c - xq.cq|; otherwise
    /// the exact float32 dot decides. Returns (min, argmin, number of exact rows).
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
    pub unsafe fn pass_q(x: *const f32, d: usize, xq: *const u8, dpad: usize, sx: *const f32, ex: *const f32,
                         n: usize, c: *const f32, cq: *const i8, sc: f32, ec: f32, corr: i32,
                         best: *mut f32, lab: *mut u32, cur: u32) -> (f32, usize, usize) {
        let mut cnt = 0usize;
        let full = d / 16 * 16;
        let rem = d - full;
        let mask: __mmask16 = if rem == 0 { 0 } else { ((1u32 << rem) - 1) as u16 };
        let nv = dpad / 64;
        let slack = 2e-5f32 + ec;
        let (scv, ecv, slv, corrv) = (_mm_set1_ps(sc), _mm_set1_ps(ec), _mm_set1_ps(slack), _mm_set1_epi32(corr));
        let mut mnv = _mm_set1_ps(f32::INFINITY);
        let mut idxv = _mm_setzero_si128();
        let mut curv = _mm_setr_epi32(0, 1, 2, 3);
        let four = _mm_set1_epi32(4);
        let mut r = 0usize;
        #[inline(always)]
        unsafe fn fold(a: __m512i) -> __m128i {
            let h = _mm256_add_epi32(_mm512_castsi512_si256(a), _mm512_extracti64x4_epi64(a, 1));
            _mm_add_epi32(_mm256_castsi256_si128(h), _mm256_extracti128_si256(h, 1))
        }
        while r + 4 <= n {
            let r0 = xq.add(r * dpad);
            let r1 = r0.add(dpad);
            let r2 = r1.add(dpad);
            let r3 = r2.add(dpad);
            let (mut a0, mut a1, mut a2, mut a3) =
                (_mm512_setzero_si512(), _mm512_setzero_si512(), _mm512_setzero_si512(), _mm512_setzero_si512());
            for v in 0..nv {
                let o = v * 64;
                let cv = _mm512_loadu_si512(cq.add(o) as *const _);
                a0 = _mm512_dpbusd_epi32(a0, _mm512_loadu_si512(r0.add(o) as *const _), cv);
                a1 = _mm512_dpbusd_epi32(a1, _mm512_loadu_si512(r1.add(o) as *const _), cv);
                a2 = _mm512_dpbusd_epi32(a2, _mm512_loadu_si512(r2.add(o) as *const _), cv);
                a3 = _mm512_dpbusd_epi32(a3, _mm512_loadu_si512(r3.add(o) as *const _), cv);
            }
            let sums = _mm_hadd_epi32(_mm_hadd_epi32(fold(a0), fold(a1)), _mm_hadd_epi32(fold(a2), fold(a3)));
            let dots = _mm_cvtepi32_ps(_mm_sub_epi32(sums, corrv));
            let exv = _mm_loadu_ps(ex.add(r));
            let approx = _mm_mul_ps(_mm_mul_ps(dots, _mm_loadu_ps(sx.add(r))), scv);
            let eps = _mm_add_ps(_mm_fmadd_ps(exv, ecv, exv), slv);
            let mut bv = _mm_loadu_ps(best.add(r));
            let cand = _mm_cmp_ps_mask::<_CMP_GT_OQ>(_mm_add_ps(approx, eps), bv);
            if cand != 0 {
                for l in 0..4 {
                    if cand & (1 << l) != 0 {
                        cnt += 1;
                        let ip = row_dot(x.add((r + l) * d), c, full, rem, mask);
                        let b = &mut *best.add(r + l);
                        if ip > *b { *b = ip; *lab.add(r + l) = cur; }
                    }
                }
                bv = _mm_loadu_ps(best.add(r));
            }
            let lt = _mm_cmp_ps_mask::<_CMP_LT_OQ>(bv, mnv);
            mnv = _mm_mask_blend_ps(lt, mnv, bv);
            idxv = _mm_mask_blend_epi32(lt, idxv, curv);
            curv = _mm_add_epi32(curv, four);
            r += 4;
        }
        let (mut mn, mut mi) = (f32::INFINITY, 0usize);
        let mut mv = [0f32; 4];
        let mut iv = [0i32; 4];
        _mm_storeu_ps(mv.as_mut_ptr(), mnv);
        _mm_storeu_si128(iv.as_mut_ptr() as *mut _, idxv);
        for l in 0..4 {
            if mv[l] < mn || (mv[l] == mn && (iv[l] as usize) < mi) { mn = mv[l]; mi = iv[l] as usize; }
        }
        while r < n {
            let row = xq.add(r * dpad);
            let mut a = _mm512_setzero_si512();
            for v in 0..nv {
                a = _mm512_dpbusd_epi32(a, _mm512_loadu_si512(row.add(v * 64) as *const _),
                                        _mm512_loadu_si512(cq.add(v * 64) as *const _));
            }
            let dot = _mm512_reduce_add_epi32(a) - corr;
            let e = *ex.add(r);
            let bp = &mut *best.add(r);
            let approx = (dot as f32) * *sx.add(r) * sc;
            if approx + (e + e * ec + slack) > *bp {
                cnt += 1;
                let ip = row_dot(x.add(r * d), c, full, rem, mask);
                if ip > *bp { *bp = ip; *lab.add(r) = cur; }
            }
            if *bp < mn { mn = *bp; mi = r; }
            r += 1;
        }
        (mn, mi, cnt)
    }
}

fn pick_pass() -> PassFn {
    #[cfg(target_arch = "x86_64")]
    {
        if std::env::var_os("CAPCOVER_GONZALEZ_SCALAR").is_some() {
            return pass_scalar;
        }
        if std::env::var_os("CAPCOVER_GONZALEZ_AVX2").is_none() && is_x86_feature_detected!("avx512f") {
            return x86::pass_avx512;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return x86::pass_avx2;
        }
    }
    pass_scalar
}

/// Gonzalez farthest-first on the rows of x (row-major n x d, float32). Returns the chosen
/// row indices (at most k; fewer on early stop when every point is already a chosen direction).
/// int8 screening copy of X: rows scaled by max|x|/127, stored as value+128 (u8).
struct Quant {
    dpad: usize,
    xq: Vec<u8>,
    sx: Vec<f32>,
    ex: Vec<f32>, // ||x - dequant(x)||, measured in f64
}

fn quantize_row(src: &[f32], dst: &mut [u8]) -> (f32, f32, Vec<i8>) {
    let m = src.iter().fold(0f32, |a, &v| a.max(v.abs()));
    let s = if m > 0.0 { m / 127.0 } else { 1.0 };
    let mut err = 0f64;
    let mut q = vec![0i8; dst.len()];
    for (j, &v) in src.iter().enumerate() {
        let qi = (v / s).round().clamp(-127.0, 127.0) as i8;
        q[j] = qi;
        let e = v as f64 - qi as f64 * s as f64;
        err += e * e;
    }
    for (j, &qi) in q.iter().enumerate() {
        dst[j] = (qi as i16 + 128) as u8;
    }
    // padding entries stay q = 0 -> byte 128
    for j in src.len()..dst.len() {
        dst[j] = 128;
    }
    (s, (err.sqrt() * (1.0 + 1e-6)) as f32 + 1e-9, q)
}

/// Gonzalez farthest-first on the rows of x (row-major n x d, float32). Returns the chosen
/// row indices (at most k; fewer on early stop when every point is already a chosen direction).
pub fn gonzalez(x: &[f32], n: usize, d: usize, k: usize) -> (Vec<i64>, Vec<u32>) {
    let mut chosen: Vec<i64> = Vec::with_capacity(k.min(n));
    let mut lab = vec![0u32; n];
    if n == 0 || k == 0 {
        return (chosen, lab);
    }
    chosen.push(0);
    let pass = pick_pass();
    let mut best = vec![0f32; n];
    let mut cbuf = vec![0f32; d];
    let xp = x.as_ptr();
    let (mut mn, mut mi) = unsafe { pass(xp, d, 0, n, xp, best.as_mut_ptr(), true, lab.as_mut_ptr(), 0) };

    #[cfg(target_arch = "x86_64")]
    let quant: Option<Quant> = if k > 8 && use_quant() {
        let dpad = (d + 63) / 64 * 64;
        let mut q = Quant { dpad, xq: vec![128u8; n * dpad], sx: vec![0f32; n], ex: vec![0f32; n] };
        for r in 0..n {
            let (s, e, _) = quantize_row(&x[r * d..(r + 1) * d], &mut q.xq[r * dpad..(r + 1) * dpad]);
            q.sx[r] = s;
            q.ex[r] = e;
        }
        Some(q)
    } else {
        None
    };
    #[cfg(not(target_arch = "x86_64"))]
    let quant: Option<Quant> = None;

    while chosen.len() < k {
        if mn >= 1.0 {
            break;
        }
        chosen.push(mi as i64);
        cbuf.copy_from_slice(&x[mi * d..(mi + 1) * d]);
        match &quant {
            #[cfg(target_arch = "x86_64")]
            Some(q) => {
                let mut cq = vec![0i8; q.dpad];
                let mut tmp = vec![0u8; q.dpad];
                let (sc, ec, qc) = quantize_row(&cbuf, &mut tmp);
                cq[..d].copy_from_slice(&qc[..d]);
                let corr: i32 = 128 * cq.iter().map(|&v| v as i32).sum::<i32>();
                let r = unsafe {
                    x86::pass_q(xp, d, q.xq.as_ptr(), q.dpad, q.sx.as_ptr(), q.ex.as_ptr(), n, cbuf.as_ptr(),
                                cq.as_ptr(), sc, ec, corr, best.as_mut_ptr(), lab.as_mut_ptr(), (chosen.len() - 1) as u32)
                };
                mn = r.0;
                mi = r.1;
            }
            _ => {
                let r = unsafe { pass(xp, d, 0, n, cbuf.as_ptr(), best.as_mut_ptr(), false, lab.as_mut_ptr(), (chosen.len() - 1) as u32) };
                mn = r.0;
                mi = r.1;
            }
        }
    }
    (chosen, lab)
}

#[cfg(target_arch = "x86_64")]
fn use_quant() -> bool {
    std::env::var_os("CAPCOVER_GONZALEZ_NOQ").is_none()
        && std::env::var_os("CAPCOVER_GONZALEZ_SCALAR").is_none()
        && is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512vnni")
        && is_x86_feature_detected!("avx512vl")
}

#[pyfunction]
fn gonzalez_chosen(py: Python<'_>, x: PyReadonlyArray2<'_, f32>, k: usize) -> PyResult<Vec<i64>> {
    let shape = x.shape();
    let (n, d) = (shape[0], shape[1]);
    let sl = x.as_slice().map_err(|_| PyValueError::new_err("x must be C-contiguous float32"))?;
    Ok(py.detach(|| gonzalez(sl, n, d, k)).0)
}

/// (chosen, labels): labels[i] = position in `chosen` of the direction row i has the largest
/// float32 inner product with (earliest position on ties). A valid assignment for the exact cap
/// cover (it only decides which cap a row joins; angles are measured in float64 afterwards).
#[pyfunction]
fn gonzalez_chosen_labels(py: Python<'_>, x: PyReadonlyArray2<'_, f32>, k: usize) -> PyResult<(Vec<i64>, Vec<u32>)> {
    let shape = x.shape();
    let (n, d) = (shape[0], shape[1]);
    let sl = x.as_slice().map_err(|_| PyValueError::new_err("x must be C-contiguous float32"))?;
    Ok(py.detach(|| gonzalez(sl, n, d, k)))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(gonzalez_chosen, m)?)?;
    m.add_function(wrap_pyfunction!(gonzalez_chosen_labels, m)?)?;
    Ok(())
}
