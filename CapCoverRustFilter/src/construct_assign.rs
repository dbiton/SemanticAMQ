//! Nearest-direction search that also returns the runner-up (`as_top2`), the primitive behind the
//! bound-based reuse in `AssignState` (minimax_construct.py).
//!
//! Same register-blocked FMA GEMM and ranking as `mm_assign` (float32 inner product, lowest index on
//! ties), but every lane keeps its best and second-best score, so one pass yields for each row
//! (label, best ip, second-best ip). The second-best is the lower bound on the angle to every other
//! direction that the caller then carries between rounds.
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::construct_minimax::{dot_scalar, has_avx2, pack_panels, MR, NR};

/// Folds one group's lane results (16 bests + 16 runner-ups) into the running (gv, gi, gv2) of a row.
#[inline]
fn fold_lanes(v: &[f32; NR], ix: &[i32; NR], v2: &[f32; NR], gv: &mut f32, gi: &mut i32, gv2: &mut f32) {
    let mut best = v[0];
    let mut bix = ix[0];
    let mut bl = 0;
    for j in 1..NR {
        if v[j] > best || (v[j] == best && ix[j] < bix) {
            best = v[j];
            bix = ix[j];
            bl = j;
        }
    }
    let mut sec = v2[bl];
    for j in 0..NR {
        if j != bl && v[j] > sec {
            sec = v[j];
        }
        if v2[j] > sec {
            sec = v2[j];
        }
    }
    if best > *gv {
        *gv2 = if *gv > sec { *gv } else { sec };
        *gv = best;
        *gi = bix;
    } else {
        *gv2 = if best > *gv2 { best } else { *gv2 };
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn top2_group_avx2(x: &[f32], n: usize, d: usize, panels: &[f32], p0: usize, p1: usize, k: usize,
                          gv: &mut [f32], gi: &mut [i32], gv2: &mut [f32]) {
    let lane = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    let neg_inf = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut r0 = 0;
    while r0 < n {
        let mut rows = [std::ptr::null::<f32>(); MR];
        for i in 0..MR {
            rows[i] = x.as_ptr().add((r0 + i).min(n - 1) * d);
        }
        let mut bv = [[neg_inf; 2]; MR];
        let mut b2 = [[neg_inf; 2]; MR];
        let mut bi = [[_mm256_setzero_si256(); 2]; MR];
        for p in p0..p1 {
            let panel = panels.as_ptr().add(p * d * NR);
            let z = _mm256_setzero_ps();
            let mut c = [z; 2 * MR];
            for kk in 0..d {
                let b0 = _mm256_loadu_ps(panel.add(kk * NR));
                let b1 = _mm256_loadu_ps(panel.add(kk * NR + 8));
                for i in 0..MR {
                    let a = _mm256_broadcast_ss(&*rows[i].add(kk));
                    c[2 * i] = _mm256_fmadd_ps(a, b0, c[2 * i]);
                    c[2 * i + 1] = _mm256_fmadd_ps(a, b1, c[2 * i + 1]);
                }
            }
            let base = (p * NR) as i32;
            let valid = (k - p * NR).min(NR);
            let idx0 = _mm256_add_epi32(_mm256_set1_epi32(base), lane);
            let idx1 = _mm256_add_epi32(idx0, _mm256_set1_epi32(8));
            let (mut m0, mut m1) = (_mm256_castsi256_ps(_mm256_set1_epi32(-1)), _mm256_castsi256_ps(_mm256_set1_epi32(-1)));
            if valid < NR {
                let lim = _mm256_set1_epi32(valid as i32);
                m0 = _mm256_castsi256_ps(_mm256_cmpgt_epi32(lim, lane));
                m1 = _mm256_castsi256_ps(_mm256_cmpgt_epi32(lim, _mm256_add_epi32(lane, _mm256_set1_epi32(8))));
            }
            for i in 0..MR {
                let v0 = _mm256_blendv_ps(neg_inf, c[2 * i], m0);
                let v1 = _mm256_blendv_ps(neg_inf, c[2 * i + 1], m1);
                let g0 = _mm256_cmp_ps(v0, bv[i][0], _CMP_GT_OQ);
                let g1 = _mm256_cmp_ps(v1, bv[i][1], _CMP_GT_OQ);
                // runner-up: the old best if the new value wins, else the larger of old runner-up and value
                b2[i][0] = _mm256_blendv_ps(_mm256_max_ps(b2[i][0], v0), bv[i][0], g0);
                b2[i][1] = _mm256_blendv_ps(_mm256_max_ps(b2[i][1], v1), bv[i][1], g1);
                bv[i][0] = _mm256_blendv_ps(bv[i][0], v0, g0);
                bv[i][1] = _mm256_blendv_ps(bv[i][1], v1, g1);
                bi[i][0] = _mm256_blendv_epi8(bi[i][0], idx0, _mm256_castps_si256(g0));
                bi[i][1] = _mm256_blendv_epi8(bi[i][1], idx1, _mm256_castps_si256(g1));
            }
        }
        for i in 0..MR.min(n - r0) {
            let mut v = [0f32; NR];
            let mut v2 = [0f32; NR];
            let mut ix = [0i32; NR];
            _mm256_storeu_ps(v.as_mut_ptr(), bv[i][0]);
            _mm256_storeu_ps(v.as_mut_ptr().add(8), bv[i][1]);
            _mm256_storeu_ps(v2.as_mut_ptr(), b2[i][0]);
            _mm256_storeu_ps(v2.as_mut_ptr().add(8), b2[i][1]);
            _mm256_storeu_si256(ix.as_mut_ptr() as *mut __m256i, bi[i][0]);
            _mm256_storeu_si256(ix.as_mut_ptr().add(8) as *mut __m256i, bi[i][1]);
            let r = r0 + i;
            fold_lanes(&v, &ix, &v2, &mut gv[r], &mut gi[r], &mut gv2[r]);
        }
        r0 += MR;
    }
}

/// (labels, best ip, second-best ip) of each row of x against the k rows of u.
pub(crate) fn top2_inner(x: &[f32], n: usize, u: &[f32], k: usize, d: usize) -> (Vec<i32>, Vec<f32>, Vec<f32>) {
    let mut gv = vec![f32::NEG_INFINITY; n];
    let mut gv2 = vec![f32::NEG_INFINITY; n];
    let mut gi = vec![0i32; n];
    if n == 0 || k == 0 {
        return (gi, gv, gv2);
    }
    #[cfg(target_arch = "x86_64")]
    if has_avx2() && d > 0 {
        let panels = pack_panels(u, k, d);
        let np = (k + NR - 1) / NR;
        let per = (262144 / (d * NR * 4)).max(1);
        let mut p0 = 0;
        while p0 < np {
            let p1 = (p0 + per).min(np);
            unsafe { top2_group_avx2(x, n, d, &panels, p0, p1, k, &mut gv, &mut gi, &mut gv2) };
            p0 = p1;
        }
        return (gi, gv, gv2);
    }
    for r in 0..n {
        let xr = &x[r * d..(r + 1) * d];
        for c in 0..k {
            let s = dot_scalar(xr, &u[c * d..(c + 1) * d]);
            if s > gv[r] {
                gv2[r] = gv[r];
                gv[r] = s;
                gi[r] = c as i32;
            } else if s > gv2[r] {
                gv2[r] = s;
            }
        }
    }
    (gi, gv, gv2)
}

/// (labels int64, best float32 ip, second-best float32 ip) of each row of x (n x d) against u (k x d):
/// exactly mm_assign's labels, plus the runner-up score (-inf when k == 1).
#[pyfunction]
fn as_top2<'py>(py: Python<'py>, x: PyReadonlyArray2<f32>, u: PyReadonlyArray2<f32>)
                -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>)> {
    let (n, d) = (x.shape()[0], x.shape()[1]);
    let k = u.shape()[0];
    if u.shape()[1] != d {
        return Err(PyValueError::new_err("dimension mismatch"));
    }
    let xs = x.as_slice().map_err(|_| PyValueError::new_err("x must be C-contiguous"))?;
    let us = u.as_slice().map_err(|_| PyValueError::new_err("u must be C-contiguous"))?;
    let (gi, g1, g2) = py.detach(|| top2_inner(xs, n, us, k, d));
    Ok((PyArray1::from_vec(py, gi.into_iter().map(|v| v as i64).collect()), PyArray1::from_vec(py, g1),
        PyArray1::from_vec(py, g2)))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(as_top2, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top2_matches_naive() {
        let (n, k, d) = (29, 70, 19);
        let mut s = 99u64;
        let mut r = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / 2147483648.0) - 0.5
        };
        let x: Vec<f32> = (0..n * d).map(|_| r()).collect();
        let u: Vec<f32> = (0..k * d).map(|_| r()).collect();
        let (gi, g1, g2) = top2_inner(&x, n, &u, k, d);
        for i in 0..n {
            let mut v: Vec<f32> = (0..k).map(|c| (0..d).map(|j| x[i * d + j] * u[c * d + j]).sum()).collect();
            let at = v[gi[i] as usize];
            assert!((at - g1[i]).abs() < 1e-4);
            v.sort_by(|a, b| b.partial_cmp(a).unwrap());
            assert!((v[0] - g1[i]).abs() < 1e-4 && (v[1] - g2[i]).abs() < 1e-4);
        }
    }
}
