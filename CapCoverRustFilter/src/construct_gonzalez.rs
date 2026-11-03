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

/// True when gonzalez_chosen(_labels) would take the int8 screening path (AVX512 VNNI+VL+BW
/// present and CAPCOVER_GONZALEZ_NOQ / _SCALAR unset).
#[pyfunction]
fn gonzalez_screening_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    { use_quant() }
    #[cfg(not(target_arch = "x86_64"))]
    { false }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(gonzalez_screening_available, m)?)?;
    m.add_function(wrap_pyfunction!(gonzalez_chosen, m)?)?;
    m.add_function(wrap_pyfunction!(gonzalez_chosen_labels, m)?)?;
    lazy::register(m)?;
    Ok(())
}

// ======================================================================
// Lazy ("CELF"-style) exact Gonzalez, AVX2/FMA tiles with scalar fallback (no AVX512 needed).
//
// `lb[x]` is a *stale* best_ip: it was exact when `x` had seen `at[x]` centers, and best_ip only
// grows, so `lb[x] <= best_ip[x]` always. The argmin of the true best_ip is found by popping the
// smallest `lb`, bringing that point up to date against the centers it missed (a batch of dots,
// point rows in L1, centers streamed) and accepting it once it is current and still the smallest:
// every other point's true value is >= its stale lb >= the accepted value. No dot product that can
// change the answer is skipped, so the sequence equals the reference's up to float32 summation order.
//
// Labels: `own[x]` is the position (in `chosen`) of the argmax of the centers x has seen, so the
// bound structure carries the owner next to the value (strict `>` keeps the earliest on ties, as
// argmax / the int8 kernel do). At the end every non-chosen row is brought up to date against the
// centers it never saw (the completion pass), which makes `own` the exact float32 nearest
// direction, the same quantity gonzalez_chosen_labels returns. Triangle-inequality pruning was
// measured too weak at d=384 and is not used. Counters report the dot products actually evaluated.
pub mod lazy {
    use numpy::{PyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static DOTS: AtomicU64 = AtomicU64::new(0);
    static REFRESHES: AtomicU64 = AtomicU64::new(0);
    static ROUNDS: AtomicU64 = AtomicU64::new(0);

    const BLK: usize = 128;
    const BATCH: usize = 33;
    const CHUNK: usize = 128;

    #[cfg(target_arch = "x86_64")]
    mod simd {
        use std::arch::x86_64::*;

        #[inline]
        pub fn have_avx2() -> bool { is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") }

        #[inline(always)]
        unsafe fn hsum(v: __m256) -> f32 {
            let lo = _mm256_castps256_ps128(v);
            let hi = _mm256_extractf128_ps(v, 1);
            let s = _mm_add_ps(lo, hi);
            let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
            let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
            _mm_cvtss_f32(s)
        }

        /// N dots of `x` against N rows (pointers) of length d.
        #[inline(always)]
        unsafe fn dotn<const N: usize>(x: *const f32, c: [*const f32; N], d: usize, out: &mut [f32; N]) {
            let mut acc = [_mm256_setzero_ps(); N];
            let mut i = 0;
            while i + 8 <= d {
                let xv = _mm256_loadu_ps(x.add(i));
                for j in 0..N {
                    acc[j] = _mm256_fmadd_ps(xv, _mm256_loadu_ps(c[j].add(i)), acc[j]);
                }
                i += 8;
            }
            for j in 0..N {
                let mut s = hsum(acc[j]);
                let mut t = i;
                while t < d { s += *x.add(t) * *c[j].add(t); t += 1; }
                out[j] = s;
            }
        }

        /// (max, argmax row) over rows cen[lo..hi] of <x, row>, kept only if strictly greater than
        /// (mx, mi); earliest row on ties.
        #[target_feature(enable = "avx2,fma")]
        pub unsafe fn max_dot_avx2(x: &[f32], cen: &[f32], d: usize, lo: usize, hi: usize,
                                   mut mx: f32, mut mi: usize) -> (f32, usize) {
            let xp = x.as_ptr();
            let base = cen.as_ptr();
            let mut r = lo;
            while r + 8 <= hi {
                let mut c = [std::ptr::null(); 8];
                for j in 0..8 { c[j] = base.add((r + j) * d); }
                let mut o = [0f32; 8];
                dotn::<8>(xp, c, d, &mut o);
                for j in 0..8 { if o[j] > mx { mx = o[j]; mi = r + j; } }
                r += 8;
            }
            while r + 4 <= hi {
                let mut c = [std::ptr::null(); 4];
                for j in 0..4 { c[j] = base.add((r + j) * d); }
                let mut o = [0f32; 4];
                dotn::<4>(xp, c, d, &mut o);
                for j in 0..4 { if o[j] > mx { mx = o[j]; mi = r + j; } }
                r += 4;
            }
            while r < hi {
                let mut o = [0f32; 1];
                dotn::<1>(xp, [base.add(r * d)], d, &mut o);
                if o[0] > mx { mx = o[0]; mi = r; }
                r += 1;
            }
            (mx, mi)
        }

        /// P points x N centers register tile: each center chunk is loaded once for P points.
        #[inline(always)]
        unsafe fn dotpn<const P: usize, const N: usize>(x: [*const f32; P], c: [*const f32; N], d: usize,
                                                         r: usize, mx: &mut [f32; P], mi: &mut [usize; P]) {
            let mut acc = [[_mm256_setzero_ps(); N]; P];
            let mut i = 0;
            while i + 8 <= d {
                let mut xv = [_mm256_setzero_ps(); P];
                for p in 0..P { xv[p] = _mm256_loadu_ps(x[p].add(i)); }
                for j in 0..N {
                    let cv = _mm256_loadu_ps(c[j].add(i));
                    for p in 0..P { acc[p][j] = _mm256_fmadd_ps(xv[p], cv, acc[p][j]); }
                }
                i += 8;
            }
            for j in 0..N {
                for p in 0..P {
                    let mut s = hsum(acc[p][j]);
                    let mut t = i;
                    while t < d { s += *x[p].add(t) * *c[j].add(t); t += 1; }
                    if s > mx[p] { mx[p] = s; mi[p] = r + j; }
                }
            }
        }

        /// For 3 points: (mx[p], mi[p]) updated with rows lo..hi (strictly greater, earliest first).
        #[target_feature(enable = "avx2,fma")]
        pub unsafe fn max_dot_p3_avx2(x: [*const f32; 3], cen: &[f32], d: usize, lo: usize, hi: usize,
                                      mx: &mut [f32; 3], mi: &mut [usize; 3]) {
            let base = cen.as_ptr();
            let mut r = lo;
            while r + 4 <= hi {
                let mut c = [std::ptr::null(); 4];
                for j in 0..4 { c[j] = base.add((r + j) * d); }
                dotpn::<3, 4>(x, c, d, r, mx, mi);
                r += 4;
            }
            while r < hi {
                dotpn::<3, 1>(x, [base.add(r * d)], d, r, mx, mi);
                r += 1;
            }
        }

        /// out[i] = <x, rows[i]> for all n rows (initial pass).
        #[target_feature(enable = "avx2,fma")]
        pub unsafe fn all_dots_avx2(x: &[f32], rows: &[f32], d: usize, out: &mut [f32]) {
            let xp = x.as_ptr();
            let base = rows.as_ptr();
            let n = out.len();
            let mut r = 0;
            while r + 8 <= n {
                let mut c = [std::ptr::null(); 8];
                for j in 0..8 { c[j] = base.add((r + j) * d); }
                let mut o = [0f32; 8];
                dotn::<8>(xp, c, d, &mut o);
                out[r..r + 8].copy_from_slice(&o);
                r += 8;
            }
            while r < n {
                let mut o = [0f32; 1];
                dotn::<1>(xp, [base.add(r * d)], d, &mut o);
                out[r] = o[0];
                r += 1;
            }
        }
    }

    #[inline]
    fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
        let mut acc = [0f32; 8];
        let ca = a.chunks_exact(8);
        let cb = b.chunks_exact(8);
        let (ra, rb) = (ca.remainder(), cb.remainder());
        for (x, y) in ca.zip(cb) { for l in 0..8 { acc[l] += x[l] * y[l]; } }
        let mut s: f32 = acc.iter().sum();
        for (x, y) in ra.iter().zip(rb) { s += x * y; }
        s
    }

    /// (max, argmax) of <x, cen[r]> for r in lo..hi, kept only if strictly above (mx, mi).
    fn max_dot(avx: bool, x: &[f32], cen: &[f32], d: usize, lo: usize, hi: usize, mx: f32, mi: usize) -> (f32, usize) {
        #[cfg(target_arch = "x86_64")]
        if avx { return unsafe { simd::max_dot_avx2(x, cen, d, lo, hi, mx, mi) }; }
        let _ = avx;
        let (mut mx, mut mi) = (mx, mi);
        for r in lo..hi { let v = dot_scalar(x, &cen[r * d..(r + 1) * d]); if v > mx { mx = v; mi = r; } }
        (mx, mi)
    }

    fn all_dots(avx: bool, x: &[f32], rows: &[f32], d: usize, out: &mut [f32]) {
        #[cfg(target_arch = "x86_64")]
        if avx { return unsafe { simd::all_dots_avx2(x, rows, d, out) }; }
        let _ = avx;
        for (i, o) in out.iter_mut().enumerate() { *o = dot_scalar(x, &rows[i * d..(i + 1) * d]); }
    }

    struct BlockMin { v: Vec<f32> }

    impl BlockMin {
        fn new(lb: &[f32]) -> Self {
            let nb = lb.len().div_ceil(BLK);
            let mut s = BlockMin { v: vec![f32::INFINITY; nb] };
            for b in 0..nb { s.refresh(lb, b); }
            s
        }
        #[inline]
        fn refresh(&mut self, lb: &[f32], b: usize) {
            let mut m = f32::INFINITY;
            for &v in &lb[b * BLK..((b + 1) * BLK).min(lb.len())] { if v < m { m = v; } }
            self.v[b] = m;
        }
        /// (index, value) of the first minimum of lb.
        #[inline]
        fn argmin(&self, lb: &[f32]) -> (usize, f32) {
            let mut bi = 0;
            let mut bv = self.v[0];
            for (b, &v) in self.v.iter().enumerate() { if v < bv { bv = v; bi = b; } }
            let lo = bi * BLK;
            let hi = (lo + BLK).min(lb.len());
            for i in lo..hi { if lb[i] == bv { return (i, bv); } }
            (lo, bv) // NaN guard
        }
    }

    struct State<'a> {
        x: &'a [f32],
        d: usize,
        avx: bool,
        cen: Vec<f32>,   // chosen rows, position-major
        lb: Vec<f32>,
        own: Vec<u32>,   // position of the argmax behind lb
        at: Vec<u32>,    // number of centers lb has seen
        dots: u64,
    }

    impl State<'_> {
        /// Bring every point of `batch` up to date against centers [at, j).
        fn refresh_batch(&mut self, batch: &[usize], j: usize) {
            let (x, d, avx) = (self.x, self.d, self.avx);
            let amin = batch.iter().map(|&c| self.at[c] as usize).min().unwrap();
            let mut r0 = amin;
            while r0 < j {
                let r1 = (r0 + CHUNK).min(j);
                let mut t = 0;
                while t < batch.len() {
                    #[cfg(target_arch = "x86_64")]
                    if avx && t + 3 <= batch.len() {
                        let cs = [batch[t], batch[t + 1], batch[t + 2]];
                        let ls = [0, 1, 2].map(|q| (self.at[cs[q]] as usize).max(r0));
                        let m = ls[0].max(ls[1]).max(ls[2]);
                        let mut mxs = [self.lb[cs[0]], self.lb[cs[1]], self.lb[cs[2]]];
                        let mut mis = [self.own[cs[0]] as usize, self.own[cs[1]] as usize, self.own[cs[2]] as usize];
                        // the part before the common start first, so ties still resolve to the earliest center
                        for q in 0..3 {
                            let hi = m.min(r1);
                            if ls[q] < hi {
                                let (v, i) = max_dot(avx, &x[cs[q] * d..(cs[q] + 1) * d], &self.cen, d, ls[q], hi, mxs[q], mis[q]);
                                mxs[q] = v;
                                mis[q] = i;
                                self.dots += (hi - ls[q]) as u64;
                            }
                        }
                        if m < r1 {
                            let xp = [x[cs[0] * d..].as_ptr(), x[cs[1] * d..].as_ptr(), x[cs[2] * d..].as_ptr()];
                            unsafe { simd::max_dot_p3_avx2(xp, &self.cen, d, m, r1, &mut mxs, &mut mis) };
                            self.dots += 3 * (r1 - m) as u64;
                        }
                        for q in 0..3 { self.lb[cs[q]] = mxs[q]; self.own[cs[q]] = mis[q] as u32; }
                        t += 3;
                        continue;
                    }
                    let c = batch[t];
                    let lo = (self.at[c] as usize).max(r0);
                    if lo < r1 {
                        let (v, i) = max_dot(avx, &x[c * d..(c + 1) * d], &self.cen, d, lo, r1, self.lb[c], self.own[c] as usize);
                        self.dots += (r1 - lo) as u64;
                        self.lb[c] = v;
                        self.own[c] = i as u32;
                    }
                    t += 1;
                }
                r0 = r1;
            }
            for &c in batch { self.at[c] = j as u32; }
        }
    }

    /// Lazy farthest-first from index 0: (chosen indices, at most k, stopping once the least
    /// best_ip reaches 1.0; labels). With `labels`, labels[i] = position in `chosen` of the center
    /// row i has the largest float32 inner product with (earliest on ties; a chosen row labels
    /// itself); otherwise the vector is empty and the completion pass is skipped.
    pub fn gonzalez_lazy(x: &[f32], n: usize, d: usize, k: usize, labels: bool) -> (Vec<i64>, Vec<u32>) {
        let mut chosen: Vec<i64> = Vec::with_capacity(k.max(1));
        if n == 0 || k == 0 { return (chosen, Vec::new()); }
        let avx = {
            #[cfg(target_arch = "x86_64")] { simd::have_avx2() && std::env::var_os("CAPCOVER_GONZALEZ_SCALAR").is_none() }
            #[cfg(not(target_arch = "x86_64"))] { false }
        };
        let mut s = State { x, d, avx, cen: Vec::with_capacity(k.min(n) * d), lb: vec![0f32; n],
                            own: vec![0u32; n], at: vec![1u32; n], dots: n as u64 };
        all_dots(avx, &x[0..d], x, d, &mut s.lb);
        chosen.push(0);
        s.cen.extend_from_slice(&x[0..d]);
        s.lb[0] = f32::INFINITY;
        let mut bm = BlockMin::new(&s.lb);
        let mut refreshes = 0u64;

        let mut batch: Vec<usize> = Vec::with_capacity(BATCH);
        let mut stale: Vec<f32> = Vec::with_capacity(BATCH);
        let mut rounds = 1usize;
        'outer: for j in 1..k {
            let c = loop {
                // Collect up to BATCH stale points in lb order (temporarily hidden from the argmin),
                // then bring them up to date together so the recent centers are read once per batch.
                batch.clear();
                stale.clear();
                let mut stop = None;
                while batch.len() < BATCH {
                    let (c, v) = bm.argmin(&s.lb);
                    if v >= 1.0 || s.at[c] as usize == j { stop = Some((c, v)); break; }
                    batch.push(c);
                    stale.push(s.lb[c]);
                    s.lb[c] = f32::INFINITY;
                    bm.refresh(&s.lb, c / BLK);
                }
                if batch.is_empty() {
                    let (c, v) = stop.unwrap();
                    if v >= 1.0 { break 'outer; } // includes INFINITY: nothing left to choose
                    break c;
                }
                for (t, &c) in batch.iter().enumerate() { s.lb[c] = stale[t]; }
                s.refresh_batch(&batch, j);
                for &c in &batch { bm.refresh(&s.lb, c / BLK); }
                refreshes += batch.len() as u64;
            };
            chosen.push(c as i64);
            s.cen.extend_from_slice(&x[c * d..(c + 1) * d]);
            s.lb[c] = f32::INFINITY;
            s.own[c] = j as u32;
            bm.refresh(&s.lb, c / BLK);
            rounds = j + 1;
        }
        let m = chosen.len();
        let mut lab = Vec::new();
        if labels {
            // Completion: every non-chosen row against the centers it has not seen, grouped by
            // `at` so a batch shares its start.
            let mut is_c = vec![false; n];
            for &c in &chosen { is_c[c as usize] = true; }
            let mut todo: Vec<usize> = (0..n).filter(|&i| !is_c[i] && (s.at[i] as usize) < m).collect();
            todo.sort_by_key(|&i| s.at[i]);
            for b in todo.chunks(BATCH) {
                s.refresh_batch(b, m);
                refreshes += b.len() as u64;
            }
            lab = std::mem::take(&mut s.own);
        }
        DOTS.fetch_add(s.dots, Ordering::Relaxed);
        REFRESHES.fetch_add(refreshes, Ordering::Relaxed);
        ROUNDS.fetch_add(rounds as u64, Ordering::Relaxed);
        (chosen, lab)
    }

    fn view<'a>(x: &'a PyReadonlyArray2<'_, f32>) -> PyResult<(&'a [f32], usize, usize)> {
        let shape = x.shape();
        let xs = x.as_slice().map_err(|_| PyValueError::new_err("x must be C-contiguous float32"))?;
        Ok((xs, shape[0], shape[1]))
    }

    /// Gonzalez farthest-first indices (int64) for float32 C-contiguous unit rows `x`.
    #[pyfunction]
    fn gonzalez_chosen_pruned<'py>(py: Python<'py>, x: PyReadonlyArray2<f32>, k: usize) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let (xs, n, d) = view(&x)?;
        let out = py.detach(|| gonzalez_lazy(xs, n, d, k, false)).0;
        Ok(PyArray1::from_vec(py, out))
    }

    /// (chosen, labels): as gonzalez_chosen_labels, from the lazy kernel (labels = exact float32
    /// nearest chosen direction, earliest position on ties).
    #[pyfunction]
    fn gonzalez_chosen_pruned_labels(py: Python<'_>, x: PyReadonlyArray2<'_, f32>, k: usize) -> PyResult<(Vec<i64>, Vec<u32>)> {
        let (xs, n, d) = view(&x)?;
        Ok(py.detach(|| gonzalez_lazy(xs, n, d, k, true)))
    }

    /// (dot products evaluated, point refreshes, rounds) accumulated since the last reset.
    #[pyfunction]
    fn gonzalez_pruned_stats(reset: bool) -> (u64, u64, u64) {
        if reset {
            (DOTS.swap(0, Ordering::Relaxed), REFRESHES.swap(0, Ordering::Relaxed), ROUNDS.swap(0, Ordering::Relaxed))
        } else {
            (DOTS.load(Ordering::Relaxed), REFRESHES.load(Ordering::Relaxed), ROUNDS.load(Ordering::Relaxed))
        }
    }

    pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(gonzalez_chosen_pruned, m)?)?;
        m.add_function(wrap_pyfunction!(gonzalez_chosen_pruned_labels, m)?)?;
        m.add_function(wrap_pyfunction!(gonzalez_pruned_stats, m)?)?;
        Ok(())
    }
}
