//! Construction kernels for the minimax method (MinimaxConstruct, minimax_construct.py).
//!
//! * `mm_assign`: exact nearest-direction search (largest float32 inner product, lowest index
//!   on ties), a register-blocked FMA GEMM fused with the arg-max so the n x k score matrix is
//!   never materialised.
//! * `mm_assign_cand`: the same ranking restricted to a candidate list per point (opt-in
//!   approximate reassignment; the caller measures every angle it keeps).
//! * `mm_bc`: all Badoiu-Clarkson steps of every cluster, cluster-locally (members gathered
//!   once, 64 x {dot, first arg-min, float64 angle of the farthest member, update}).
//! * `mm_stats`: float64 per-point angles to the assigned direction, per-cluster max and counts.
//!
//! Everything runs without the GIL and on the calling thread. x86_64 uses AVX2+FMA when the CPU
//! has it (runtime detection), everything else a portable scalar path with the same tie rules.
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

fn has_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        return std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma");
    }
    #[allow(unreachable_code)]
    false
}

// ---------------------------------------------------------------------------------------------
// dot products
// ---------------------------------------------------------------------------------------------

fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut s = [0f32; 8];
    let n = a.len() / 8 * 8;
    for i in (0..n).step_by(8) {
        for j in 0..8 {
            s[j] += a[i + j] * b[i + j];
        }
    }
    let mut t = ((s[0] + s[4]) + (s[2] + s[6])) + ((s[1] + s[5]) + (s[3] + s[7]));
    for i in n..a.len() {
        t += a[i] * b[i];
    }
    t
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn hsum(v: __m256) -> f32 {
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps(v, 1);
    let s = _mm_add_ps(lo, hi);
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    _mm_cvtss_f32(s)
}

/// out[i] = <rows[i], v> for m rows of length d.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dots_avx2(rows: &[f32], m: usize, d: usize, v: &[f32], out: &mut [f32]) {
    let main = d / 16 * 16;
    let vp = v.as_ptr();
    let mut i = 0;
    while i + 4 <= m {
        let r0 = rows.as_ptr().add(i * d);
        let r1 = r0.add(d);
        let r2 = r1.add(d);
        let r3 = r2.add(d);
        let z = _mm256_setzero_ps();
        let (mut a0, mut a1, mut a2, mut a3) = (z, z, z, z);
        let (mut b0, mut b1, mut b2, mut b3) = (z, z, z, z);
        let mut k = 0;
        while k < main {
            let v0 = _mm256_loadu_ps(vp.add(k));
            let v1 = _mm256_loadu_ps(vp.add(k + 8));
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(r0.add(k)), v0, a0);
            b0 = _mm256_fmadd_ps(_mm256_loadu_ps(r0.add(k + 8)), v1, b0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(r1.add(k)), v0, a1);
            b1 = _mm256_fmadd_ps(_mm256_loadu_ps(r1.add(k + 8)), v1, b1);
            a2 = _mm256_fmadd_ps(_mm256_loadu_ps(r2.add(k)), v0, a2);
            b2 = _mm256_fmadd_ps(_mm256_loadu_ps(r2.add(k + 8)), v1, b2);
            a3 = _mm256_fmadd_ps(_mm256_loadu_ps(r3.add(k)), v0, a3);
            b3 = _mm256_fmadd_ps(_mm256_loadu_ps(r3.add(k + 8)), v1, b3);
            k += 16;
        }
        let mut s = [hsum(_mm256_add_ps(a0, b0)), hsum(_mm256_add_ps(a1, b1)), hsum(_mm256_add_ps(a2, b2)),
                     hsum(_mm256_add_ps(a3, b3))];
        let rs = [r0, r1, r2, r3];
        for j in 0..4 {
            for kk in main..d {
                s[j] += *rs[j].add(kk) * *vp.add(kk);
            }
            out[i + j] = s[j];
        }
        i += 4;
    }
    while i < m {
        let r = rows.as_ptr().add(i * d);
        let z = _mm256_setzero_ps();
        let (mut a, mut b) = (z, z);
        let mut k = 0;
        while k < main {
            a = _mm256_fmadd_ps(_mm256_loadu_ps(r.add(k)), _mm256_loadu_ps(vp.add(k)), a);
            b = _mm256_fmadd_ps(_mm256_loadu_ps(r.add(k + 8)), _mm256_loadu_ps(vp.add(k + 8)), b);
            k += 16;
        }
        let mut s = hsum(_mm256_add_ps(a, b));
        for kk in main..d {
            s += *r.add(kk) * *vp.add(kk);
        }
        out[i] = s;
        i += 1;
    }
}

fn dots(avx: bool, rows: &[f32], m: usize, d: usize, v: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if avx {
        unsafe { dots_avx2(rows, m, d, v, out) };
        return;
    }
    let _ = avx;
    for i in 0..m {
        out[i] = dot_scalar(&rows[i * d..(i + 1) * d], v);
    }
}

// ---------------------------------------------------------------------------------------------
// float64 helpers (same formulas as cap_geometry.unit_rows / angle_between)
// ---------------------------------------------------------------------------------------------

#[inline(always)]
fn sumsq(src: &[f64]) -> f64 {
    let mut a = [0f64; 4];
    let n4 = src.len() / 4 * 4;
    for i in (0..n4).step_by(4) {
        for j in 0..4 {
            a[j] += src[i + j] * src[i + j];
        }
    }
    let mut s = (a[0] + a[2]) + (a[1] + a[3]);
    for i in n4..src.len() {
        s += src[i] * src[i];
    }
    s
}

#[inline(always)]
fn unit_into(src: &[f64], dst: &mut [f64]) {
    let mut a = [0f64; 4];
    let n4 = src.len() / 4 * 4;
    for i in (0..n4).step_by(4) {
        for j in 0..4 {
            a[j] += src[i + j] * src[i + j];
        }
    }
    let mut s = (a[0] + a[2]) + (a[1] + a[3]);
    for i in n4..src.len() {
        s += src[i] * src[i];
    }
    // x / norm, as unit_rows does; the divisions are what the vector unit is slow at, so scale by
    // the reciprocal (differs from the quotient by one rounding at most).
    let inv = 1.0 / s.sqrt().max(1e-300);
    for (o, &x) in dst.iter_mut().zip(src) {
        *o = x * inv;
    }
}

#[inline(always)]
fn angle64(a: &[f64], b: &[f64]) -> f64 {
    let mut s = [0f64; 4];
    let n = a.len() / 4 * 4;
    for i in (0..n).step_by(4) {
        for j in 0..4 {
            let t = a[i + j] - b[i + j];
            s[j] += t * t;
        }
    }
    let mut t = (s[0] + s[2]) + (s[1] + s[3]);
    for i in n..a.len() {
        let u = a[i] - b[i];
        t += u * u;
    }
    2.0 * (0.5 * t.sqrt()).min(1.0).asin()
}

// ---------------------------------------------------------------------------------------------
// Badoiu-Clarkson, cluster-local
// ---------------------------------------------------------------------------------------------

/// Cluster c's members are rows order[starts[c] .. starts[c] + counts[c]] of x32/x64. Writes each
/// cluster's best unit direction (by float64 cap angle) into `out` (k x d).
fn bc_inner(x32: &[f32], x64: &[f64], order: &[i64], starts: &[i64], counts: &[i64], u: &[f64], d: usize,
            steps: usize, out: &mut [f64]) {
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        unsafe { bc_avx2(x32, x64, order, starts, counts, u, d, steps, out) };
        return;
    }
    bc_body(x32, x64, order, starts, counts, u, d, steps, out)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn bc_avx2(x32: &[f32], x64: &[f64], order: &[i64], starts: &[i64], counts: &[i64], u: &[f64], d: usize,
                  steps: usize, out: &mut [f64]) {
    bc_body(x32, x64, order, starts, counts, u, d, steps, out)
}

#[inline(always)]
fn bc_body(x32: &[f32], x64: &[f64], order: &[i64], starts: &[i64], counts: &[i64], u: &[f64], d: usize,
            steps: usize, out: &mut [f64]) {
    let avx = has_avx2();
    let k = counts.len();
    let mut buf: Vec<f32> = Vec::new();
    let mut ip: Vec<f32> = Vec::new();
    let mut c = vec![0f64; d];
    let mut un = vec![0f64; d];
    let mut un32 = vec![0f32; d];
    for cl in 0..k {
        let m = counts[cl] as usize;
        let s = starts[cl] as usize;
        let o = &mut out[cl * d..(cl + 1) * d];
        let urow = &u[cl * d..(cl + 1) * d];
        if m == 0 {
            unit_into(urow, o);
            continue;
        }
        if m == 1 {
            let r = order[s] as usize;
            unit_into(&x64[r * d..(r + 1) * d], o);
            continue;
        }
        buf.resize(m * d, 0.0);
        ip.resize(m, 0.0);
        for i in 0..m {
            let r = order[s + i] as usize;
            buf[i * d..(i + 1) * d].copy_from_slice(&x32[r * d..(r + 1) * d]);
        }
        c.copy_from_slice(urow);
        unit_into(&c, o); // the answer if steps == 0
        let mut best_t = f64::INFINITY;
        let mut ss = sumsq(&c);
        for t in 1..=steps {
            let inv = 1.0 / ss.sqrt().max(1e-300);
            for j in 0..d {
                let v = c[j] * inv;
                un[j] = v;
                un32[j] = v as f32;
            }
            dots(avx, &buf, m, d, &un32, &mut ip);
            let mut far = 0;
            let mut low = ip[0];
            for i in 1..m {
                if ip[i] < low {
                    low = ip[i];
                    far = i;
                }
            }
            let r = order[s + far] as usize;
            let xf = &x64[r * d..(r + 1) * d];
            let theta = angle64(xf, &un);
            if theta < best_t {
                best_t = theta;
                o.copy_from_slice(&un);
            }
            let w = 1.0 / (t as f64 + 1.0);
            let mut acc = [0f64; 4];
            let d4 = d / 4 * 4;
            for j in (0..d4).step_by(4) {
                for l in 0..4 {
                    let v = c[j + l] + (xf[j + l] - c[j + l]) * w;
                    c[j + l] = v;
                    acc[l] += v * v;
                }
            }
            ss = (acc[0] + acc[2]) + (acc[1] + acc[3]);
            for j in d4..d {
                let v = c[j] + (xf[j] - c[j]) * w;
                c[j] = v;
                ss += v * v;
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Exact nearest direction: fused GEMM + arg-max
// ---------------------------------------------------------------------------------------------

const NR: usize = 16;
const MR: usize = 6;

/// U (k x d) packed into panels of NR directions: panel p, element [kk * NR + j] = U[p*NR + j][kk].
fn pack_panels(u: &[f32], k: usize, d: usize) -> Vec<f32> {
    let np = (k + NR - 1) / NR;
    let mut out = vec![0f32; np * d * NR];
    for p in 0..np {
        for j in 0..NR.min(k - p * NR) {
            let row = &u[(p * NR + j) * d..(p * NR + j + 1) * d];
            let base = p * d * NR;
            for kk in 0..d {
                out[base + kk * NR + j] = row[kk];
            }
        }
    }
    out
}

/// Lane-wise running best of one MR x NR tile over a group of panels.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn assign_group_avx2(x: &[f32], n: usize, d: usize, panels: &[f32], p0: usize, p1: usize, k: usize,
                            gv: &mut [f32], gi: &mut [i32], lanes: bool) {
    let lane = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    let neg_inf = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut r0 = 0;
    while r0 < n {
        let mut rows = [std::ptr::null::<f32>(); MR];
        for i in 0..MR {
            rows[i] = x.as_ptr().add((r0 + i).min(n - 1) * d);
        }
        let mut bv = [[neg_inf; 2]; MR];
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
                bv[i][0] = _mm256_blendv_ps(bv[i][0], v0, g0);
                bv[i][1] = _mm256_blendv_ps(bv[i][1], v1, g1);
                bi[i][0] = _mm256_blendv_epi8(bi[i][0], idx0, _mm256_castps_si256(g0));
                bi[i][1] = _mm256_blendv_epi8(bi[i][1], idx1, _mm256_castps_si256(g1));
            }
        }
        for i in 0..MR.min(n - r0) {
            let mut v = [0f32; NR];
            let mut ix = [0i32; NR];
            _mm256_storeu_ps(v.as_mut_ptr(), bv[i][0]);
            _mm256_storeu_ps(v.as_mut_ptr().add(8), bv[i][1]);
            _mm256_storeu_si256(ix.as_mut_ptr() as *mut __m256i, bi[i][0]);
            _mm256_storeu_si256(ix.as_mut_ptr().add(8) as *mut __m256i, bi[i][1]);
            if lanes {
                // keep each lane's winner (the best of the directions congruent mod NR)
                let r = r0 + i;
                for j in 0..NR {
                    if v[j] > gv[r * NR + j] {
                        gv[r * NR + j] = v[j];
                        gi[r * NR + j] = ix[j];
                    }
                }
                continue;
            }
            let mut best = v[0];
            let mut bix = ix[0];
            for j in 1..NR {
                if v[j] > best || (v[j] == best && ix[j] < bix) {
                    best = v[j];
                    bix = ix[j];
                }
            }
            let r = r0 + i;
            if best > gv[r] {
                gv[r] = best;
                gi[r] = bix;
            }
        }
        r0 += MR;
    }
}

fn assign_group_scalar(x: &[f32], n: usize, d: usize, u: &[f32], k: usize, c0: usize, c1: usize,
                       gv: &mut [f32], gi: &mut [i32]) {
    for r in 0..n {
        let xr = &x[r * d..(r + 1) * d];
        for c in c0..c1.min(k) {
            let s = dot_scalar(xr, &u[c * d..(c + 1) * d]);
            if s > gv[r] {
                gv[r] = s;
                gi[r] = c as i32;
            }
        }
    }
}

/// (labels, best inner products): each row of x's direction of largest float32 inner product,
/// the lowest index among equals.
fn assign_inner(x: &[f32], n: usize, u: &[f32], k: usize, d: usize) -> (Vec<i32>, Vec<f32>) {
    assign_run(x, n, u, k, d, false)
}

/// With `lanes`, instead of the arg-max returns n x NR lane winners: for lane j the best of the
/// directions with index = j (mod NR) (-1 where there is none).
fn assign_run(x: &[f32], n: usize, u: &[f32], k: usize, d: usize, lanes: bool) -> (Vec<i32>, Vec<f32>) {
    let w = if lanes { NR } else { 1 };
    let mut gv = vec![f32::NEG_INFINITY; n * w];
    let mut gi = vec![if lanes { -1 } else { 0 }; n * w];
    if n == 0 || k == 0 {
        return (gi, gv);
    }
    if has_avx2() && d > 0 {
        #[cfg(target_arch = "x86_64")]
        {
            let panels = pack_panels(u, k, d);
            let np = (k + NR - 1) / NR;
            // Keep a group of panels (~256 KB) resident in L2 while all rows stream past it.
            let per = (262144 / (d * NR * 4)).max(1);
            let mut p0 = 0;
            while p0 < np {
                let p1 = (p0 + per).min(np);
                unsafe { assign_group_avx2(x, n, d, &panels, p0, p1, k, &mut gv, &mut gi, lanes) };
                p0 = p1;
            }
            return (gi, gv);
        }
    }
    if lanes {
        for r in 0..n {
            for c in 0..k {
                let sc = dot_scalar(&x[r * d..(r + 1) * d], &u[c * d..(c + 1) * d]);
                if sc > gv[r * NR + c % NR] {
                    gv[r * NR + c % NR] = sc;
                    gi[r * NR + c % NR] = c as i32;
                }
            }
        }
        return (gi, gv);
    }
    let step = 64;
    let mut c0 = 0;
    while c0 < k {
        assign_group_scalar(x, n, d, u, k, c0, c0 + step, &mut gv, &mut gi);
        c0 += step;
    }
    (gi, gv)
}

/// Rank only candidates[i*w .. (i+1)*w] (-1 padded) for row i.
fn assign_cand_inner(x: &[f32], n: usize, u: &[f32], d: usize, cand: &[i32], w: usize) -> (Vec<i32>, Vec<f32>) {
    let avx = has_avx2();
    let mut gi = vec![0i32; n];
    let mut gv = vec![f32::NEG_INFINITY; n];
    let mut ip = vec![0f32; w];
    let mut ids = vec![0i32; w];
    for r in 0..n {
        let mut m = 0;
        for &c in &cand[r * w..(r + 1) * w] {
            if c >= 0 {
                ids[m] = c;
                let row = &u[c as usize * d..(c as usize + 1) * d];
                dots(avx, &x[r * d..(r + 1) * d], 1, d, row, &mut ip[m..m + 1]);
                m += 1;
            }
        }
        if m == 0 {
            continue;
        }
        let mut b = 0;
        for i in 1..m {
            if ip[i] > ip[b] || (ip[i] == ip[b] && ids[i] < ids[b]) {
                b = i;
            }
        }
        gi[r] = ids[b];
        gv[r] = ip[b];
    }
    (gi, gv)
}

// ---------------------------------------------------------------------------------------------
// Python glue
// ---------------------------------------------------------------------------------------------

fn sl<'a, T: numpy::Element>(a: &'a PyReadonlyArray2<'a, T>, name: &str) -> PyResult<&'a [T]> {
    a.as_slice().map_err(|_| PyValueError::new_err(format!("{name} must be C-contiguous")))
}

fn sl1<'a, T: numpy::Element>(a: &'a PyReadonlyArray1<'a, T>, name: &str) -> PyResult<&'a [T]> {
    a.as_slice().map_err(|_| PyValueError::new_err(format!("{name} must be C-contiguous")))
}

/// (labels int64, best float32 inner product) of each row of x (float32 n x d) against the rows
/// of u (float32 k x d): exact nearest direction by inner product, lowest index on ties.
#[pyfunction]
fn mm_assign<'py>(py: Python<'py>, x: PyReadonlyArray2<f32>, u: PyReadonlyArray2<f32>)
                  -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f32>>)> {
    let (n, d) = (x.shape()[0], x.shape()[1]);
    let k = u.shape()[0];
    if u.shape()[1] != d {
        return Err(PyValueError::new_err("dimension mismatch"));
    }
    let (xs, us) = (sl(&x, "x")?, sl(&u, "u")?);
    let (gi, gv) = py.detach(|| assign_inner(xs, n, us, k, d));
    Ok((PyArray1::from_vec(py, gi.into_iter().map(|v| v as i64).collect()), PyArray1::from_vec(py, gv)))
}

/// Lane winners (int32 n x 16, -1 padded) of x (n x r) against u (k x r) in the dot-product
/// ranking: for each of 16 classes of directions (index mod 16) the best one. Run on projected
/// rows as a cheap shortlist for mm_assign_cand.
#[pyfunction]
fn mm_lanes<'py>(py: Python<'py>, x: PyReadonlyArray2<f32>, u: PyReadonlyArray2<f32>)
                 -> PyResult<Bound<'py, PyArray2<i32>>> {
    let (n, d) = (x.shape()[0], x.shape()[1]);
    let k = u.shape()[0];
    if u.shape()[1] != d {
        return Err(PyValueError::new_err("dimension mismatch"));
    }
    let (xs, us) = (sl(&x, "x")?, sl(&u, "u")?);
    let (gi, _) = py.detach(|| assign_run(xs, n, us, k, d, true));
    PyArray1::from_vec(py, gi).reshape([n, NR])
}

/// mm_assign restricted to cand (int32 n x w, -1 padded): the best candidate per row.
#[pyfunction]
fn mm_assign_cand<'py>(py: Python<'py>, x: PyReadonlyArray2<f32>, u: PyReadonlyArray2<f32>,
                       cand: PyReadonlyArray2<i32>) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f32>>)> {
    let (n, d) = (x.shape()[0], x.shape()[1]);
    let k = u.shape()[0];
    let w = cand.shape()[1];
    if u.shape()[1] != d || cand.shape()[0] != n {
        return Err(PyValueError::new_err("shape mismatch"));
    }
    let (xs, us, cs) = (sl(&x, "x")?, sl(&u, "u")?, sl(&cand, "cand")?);
    if cs.iter().any(|&c| c >= k as i32) {
        return Err(PyValueError::new_err("candidate out of range"));
    }
    let (gi, gv) = py.detach(|| assign_cand_inner(xs, n, us, d, cs, w));
    Ok((PyArray1::from_vec(py, gi.into_iter().map(|v| v as i64).collect()), PyArray1::from_vec(py, gv)))
}

/// Badoiu-Clarkson on every cluster. Cluster c's members are rows order[starts[c]:+counts[c]] of
/// x32 / x64 (float32 / float64 copies of the same n x d rows); u is the k x d float64 start.
/// Returns each cluster's best unit direction, k x d float64.
#[pyfunction]
fn mm_bc<'py>(py: Python<'py>, x32: PyReadonlyArray2<f32>, x64: PyReadonlyArray2<f64>,
              order: PyReadonlyArray1<i64>, starts: PyReadonlyArray1<i64>, counts: PyReadonlyArray1<i64>,
              u: PyReadonlyArray2<f64>, steps: usize) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (n, d) = (x32.shape()[0], x32.shape()[1]);
    let k = u.shape()[0];
    if x64.shape()[0] != n || x64.shape()[1] != d || u.shape()[1] != d || starts.len() != k || counts.len() != k {
        return Err(PyValueError::new_err("shape mismatch"));
    }
    let (a, b, o, s, c, uu) = (sl(&x32, "x32")?, sl(&x64, "x64")?, sl1(&order, "order")?, sl1(&starts, "starts")?,
                               sl1(&counts, "counts")?, sl(&u, "u")?);
    for cl in 0..k {
        let (st, ct) = (s[cl], c[cl]);
        if st < 0 || ct < 0 || (st + ct) as usize > o.len() {
            return Err(PyValueError::new_err("segment out of range"));
        }
    }
    if o.iter().any(|&r| r < 0 || r as usize >= n) {
        return Err(PyValueError::new_err("order out of range"));
    }
    let mut out = vec![0f64; k * d];
    py.detach(|| bc_inner(a, b, o, s, c, uu, d, steps, &mut out));
    PyArray1::from_vec(py, out).reshape([k, d])
}

/// (per-point float64 angle to its assigned direction, per-cluster max angle (0 if empty), counts).
#[pyfunction]
fn mm_stats<'py>(py: Python<'py>, x64: PyReadonlyArray2<f64>, u: PyReadonlyArray2<f64>, labels: PyReadonlyArray1<i64>)
                 -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<i64>>)> {
    let (n, d) = (x64.shape()[0], x64.shape()[1]);
    let k = u.shape()[0];
    if u.shape()[1] != d || labels.len() != n {
        return Err(PyValueError::new_err("shape mismatch"));
    }
    let (xs, us, ls) = (sl(&x64, "x")?, sl(&u, "u")?, sl1(&labels, "labels")?);
    if ls.iter().any(|&l| l < 0 || l as usize >= k) {
        return Err(PyValueError::new_err("label out of range"));
    }
    let (th, tc, cnt) = py.detach(|| {
        let mut th = vec![0f64; n];
        let mut tc = vec![0f64; k];
        let mut cnt = vec![0i64; k];
        for i in 0..n {
            let l = ls[i] as usize;
            let t = angle64(&xs[i * d..(i + 1) * d], &us[l * d..(l + 1) * d]);
            th[i] = t;
            if t > tc[l] {
                tc[l] = t;
            }
            cnt[l] += 1;
        }
        (th, tc, cnt)
    });
    Ok((PyArray1::from_vec(py, th), PyArray1::from_vec(py, tc), PyArray1::from_vec(py, cnt)))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(mm_assign, m)?)?;
    m.add_function(wrap_pyfunction!(mm_assign_cand, m)?)?;
    m.add_function(wrap_pyfunction!(mm_lanes, m)?)?;
    m.add_function(wrap_pyfunction!(mm_bc, m)?)?;
    m.add_function(wrap_pyfunction!(mm_stats, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assign_matches_naive() {
        let (n, k, d) = (37, 53, 21);
        let mut s = 12345u64;
        let mut r = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / 2147483648.0) - 0.5
        };
        let x: Vec<f32> = (0..n * d).map(|_| r()).collect();
        let u: Vec<f32> = (0..k * d).map(|_| r()).collect();
        let (gi, _) = assign_inner(&x, n, &u, k, d);
        for i in 0..n {
            let mut best = 0;
            let mut bv = f32::NEG_INFINITY;
            for c in 0..k {
                let v: f32 = (0..d).map(|j| x[i * d + j] * u[c * d + j]).sum();
                if v > bv + 1e-5 {
                    bv = v;
                    best = c;
                }
            }
            let v: f32 = (0..d).map(|j| x[i * d + j] * u[gi[i] as usize * d + j]).sum();
            assert!(v >= bv - 1e-4, "row {i}: got {} want {}", gi[i], best);
        }
    }
}
