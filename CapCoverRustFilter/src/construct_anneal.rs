//! Kernels for soft-anneal construction (src/filters/cap_cover/soft_anneal_construct.py).
//!
//! The relaxation lives on the n x m lists of each vector's top-m caps. Python keeps the
//! (k-sized) cap-level algebra; these kernels fuse the O(n m d) and O(n m) parts:
//!   anneal_stage1  angles, row softmax, per-cap online log-sum-exp accumulators
//!   anneal_stage2  gradient coefficients and the scatter G[c] += coef * x
//!   assigned_theta per-cap max angle of hard-assigned members (float64, like angle_between)
//!   cache_topm     top-m of a vector's cached candidate caps plus an explicit cap set,
//!                  with a certificate that no uncached cap can have entered the top-m
//! Reductions that Python does in float64 are float64 here; inner products are float32 (they only
//! rank, exactly as faiss does), and the gradient accumulator is float32 (as scipy's csr @ float32).

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

const FMA_NATIVE: bool = cfg!(target_arch = "aarch64");

#[inline(always)]
fn fmadd<const FMA: bool>(a: f32, b: f32, c: f32) -> f32 {
    if FMA { a.mul_add(b, c) } else { a * b + c }
}

#[inline(always)]
fn dot<const FMA: bool>(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 16];
    let mut ca = a.chunks_exact(16);
    let mut cb = b.chunks_exact(16);
    for (x, y) in (&mut ca).zip(&mut cb) {
        for l in 0..16 {
            acc[l] = fmadd::<FMA>(x[l], y[l], acc[l]);
        }
    }
    let mut tail = 0f32;
    for (x, y) in ca.remainder().iter().zip(cb.remainder()) {
        tail = fmadd::<FMA>(*x, *y, tail);
    }
    let mut s = 0f32;
    for l in 0..8 {
        s += acc[l] + acc[l + 8];
    }
    s + tail
}

#[inline(always)]
fn axpy<const FMA: bool>(g: &mut [f32], c: f32, x: &[f32]) {
    for (gi, xi) in g.iter_mut().zip(x) {
        *gi = fmadd::<FMA>(c, *xi, *gi);
    }
}

#[inline(always)]
fn lse_push(mx: &mut f64, s: &mut f64, v: f64) {
    if v <= *mx {
        *s += (v - *mx).exp();
    } else {
        *s = *s * (*mx - v).exp() + 1.0;
        *mx = v;
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn stage1_impl<const FMA: bool>(x: &[f32], u: &[f32], cand: &[i64], order: &[i64], n: usize, d: usize, m: usize, ta: f64, tm: f64,
                                dd: &mut [f64], lnr: &mut [f64], mxl: &mut [f64], sl: &mut [f64],
                                mxr: &mut [f64], sr: &mut [f64]) {
    let mut a = vec![0f64; m];
    for t in 0..n {
        let i = if order.is_empty() { t } else { order[t] as usize };
        let xr = &x[i * d..(i + 1) * d];
        let mut mxa = f64::NEG_INFINITY;
        for j in 0..m {
            let c = cand[i * m + j] as usize;
            let ip = dot::<FMA>(xr, &u[c * d..(c + 1) * d]) as f64;
            let dj = ip.clamp(-1.0, 1.0).acos();
            dd[i * m + j] = dj;
            a[j] = -dj / ta;
            if a[j] > mxa {
                mxa = a[j];
            }
        }
        let mut z = 0f64;
        for j in 0..m {
            z += (a[j] - mxa).exp();
        }
        let ln_z = z.ln() + mxa;
        for j in 0..m {
            let c = cand[i * m + j] as usize;
            let lr = a[j] - ln_z;
            lnr[i * m + j] = lr;
            let l = lr + dd[i * m + j] / tm;
            lse_push(&mut mxl[c], &mut sl[c], l);
            lse_push(&mut mxr[c], &mut sr[c], lr);
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn stage2_impl<const FMA: bool>(x: &[f32], cand: &[i64], order: &[i64], dd: &[f64], lnr: &[f64], mxl: &[f64], sl: &[f64],
                                mxr: &[f64], sr: &[f64], gth: &[f64], n: usize, d: usize, m: usize, ta: f64,
                                tm: f64, g: &mut [f32]) {
    let mut a = vec![0f64; m];
    let mut e = vec![0f64; m];
    let mut r = vec![0f64; m];
    for t in 0..n {
        let i = if order.is_empty() { t } else { order[t] as usize };
        let xr = &x[i * d..(i + 1) * d];
        let mut esum = 0f64;
        for j in 0..m {
            let c = cand[i * m + j] as usize;
            let lr = lnr[i * m + j];
            let l = lr + dd[i * m + j] / tm;
            let aj = (l - mxl[c]).exp() / sl[c].max(1e-300);
            let bj = (lr - mxr[c]).exp() / sr[c].max(1e-300);
            let ej = gth[c] * tm * (aj - bj);
            a[j] = aj;
            e[j] = ej;
            r[j] = lr.exp();
            esum += ej;
        }
        for j in 0..m {
            let c = cand[i * m + j] as usize;
            let dd_ = gth[c] * a[j] - e[j] / ta + r[j] / ta * esum;
            let sin_d = dd[i * m + j].sin().max(0.02);
            let coef = (-dd_ / sin_d) as f32;
            axpy::<FMA>(&mut g[c * d..(c + 1) * d], coef, xr);
        }
    }
}

macro_rules! dispatch {
    ($name:ident, $imp:ident, $avx:ident, ($($arg:ident : $ty:ty),*)) => {
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2,fma")]
        unsafe fn $avx($($arg: $ty),*) { $imp::<true>($($arg),*) }
        #[allow(clippy::too_many_arguments)]
        fn $name($($arg: $ty),*) {
            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return unsafe { $avx($($arg),*) };
            }
            $imp::<FMA_NATIVE>($($arg),*)
        }
    };
}

dispatch!(stage1, stage1_impl, stage1_avx, (x: &[f32], u: &[f32], cand: &[i64], order: &[i64], n: usize, d: usize, m: usize,
    ta: f64, tm: f64, dd: &mut [f64], lnr: &mut [f64], mxl: &mut [f64], sl: &mut [f64], mxr: &mut [f64],
    sr: &mut [f64]));
dispatch!(stage2, stage2_impl, stage2_avx, (x: &[f32], cand: &[i64], order: &[i64], dd: &[f64], lnr: &[f64], mxl: &[f64],
    sl: &[f64], mxr: &[f64], sr: &[f64], gth: &[f64], n: usize, d: usize, m: usize, ta: f64, tm: f64,
    g: &mut [f32]));

fn check_cand(cand: &[i64], k: usize) -> PyResult<()> {
    if cand.iter().any(|&c| c < 0 || c as usize >= k) {
        return Err(PyValueError::new_err("candidate cap id out of range"));
    }
    Ok(())
}

fn check_order<'a>(o: &'a PyReadonlyArray1<'_, i64>, n: usize) -> PyResult<&'a [i64]> {
    let s = o.as_slice().map_err(|_| PyValueError::new_err("order must be contiguous"))?;
    if !s.is_empty() && (s.len() != n || s.iter().any(|&v| v < 0 || v as usize >= n)) {
        return Err(PyValueError::new_err("order must be empty or a permutation of rows"));
    }
    Ok(s)
}

fn slice2<'a, T: numpy::Element>(a: &'a PyReadonlyArray2<'_, T>, what: &str) -> PyResult<&'a [T]> {
    a.as_slice().map_err(|_| PyValueError::new_err(format!("{what} must be C-contiguous")))
}

/// (D, lnR, (4, k): per-cap [max, sum] of L = lnR + D/Tm and of lnR, as online log-sum-exp).
#[pyfunction]
#[pyo3(name = "anneal_stage1")]
#[allow(clippy::type_complexity)]
fn anneal_stage1_py<'py>(py: Python<'py>, x32: PyReadonlyArray2<f32>, u32: PyReadonlyArray2<f32>,
                         cand: PyReadonlyArray2<i64>, order: PyReadonlyArray1<i64>, ta: f64, tm: f64)
                         -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    let (n, d) = (x32.shape()[0], x32.shape()[1]);
    let (k, m) = (u32.shape()[0], cand.shape()[1]);
    if u32.shape()[1] != d || cand.shape()[0] != n {
        return Err(PyValueError::new_err("shape mismatch"));
    }
    let (xs, us, cs) = (slice2(&x32, "x")?, slice2(&u32, "u")?, slice2(&cand, "cand")?);
    check_cand(cs, k)?;
    let os = check_order(&order, n)?;
    let (mut dd, mut lnr) = (vec![0f64; n * m], vec![0f64; n * m]);
    let mut cap = vec![0f64; 4 * k];
    py.detach(|| {
        let (a, b) = cap.split_at_mut(2 * k);
        let (mxl, sl) = a.split_at_mut(k);
        let (mxr, sr) = b.split_at_mut(k);
        mxl.fill(f64::NEG_INFINITY);
        mxr.fill(f64::NEG_INFINITY);
        stage1(xs, us, cs, os, n, d, m, ta, tm, &mut dd, &mut lnr, mxl, sl, mxr, sr);
    });
    let dd = PyArray1::from_vec(py, dd).reshape([n, m])?;
    let lnr = PyArray1::from_vec(py, lnr).reshape([n, m])?;
    let cap = PyArray1::from_vec(py, cap).reshape([4, k])?;
    Ok((dd, lnr, cap))
}

/// G (k x d, float64) = sum over (x, slot) with cap c of coef * x; coefficients as in the reference.
#[pyfunction]
#[pyo3(name = "anneal_stage2")]
#[allow(clippy::too_many_arguments)]
fn anneal_stage2_py<'py>(py: Python<'py>, x32: PyReadonlyArray2<f32>, cand: PyReadonlyArray2<i64>, order: PyReadonlyArray1<i64>,
                         dd: PyReadonlyArray2<f64>, lnr: PyReadonlyArray2<f64>, cap: PyReadonlyArray2<f64>,
                         gth: PyReadonlyArray1<f64>, ta: f64, tm: f64) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (n, d) = (x32.shape()[0], x32.shape()[1]);
    let (k, m) = (gth.len(), cand.shape()[1]);
    if cap.shape()[0] != 4 || cap.shape()[1] != k {
        return Err(PyValueError::new_err("cap stats must be (4, k)"));
    }
    let (xs, cs) = (slice2(&x32, "x")?, slice2(&cand, "cand")?);
    check_cand(cs, k)?;
    let os = check_order(&order, n)?;
    let (ds, ls, caps) = (slice2(&dd, "D")?, slice2(&lnr, "lnR")?, slice2(&cap, "cap")?);
    let gs = gth.as_slice().map_err(|_| PyValueError::new_err("gth must be contiguous"))?;
    let mut g = vec![0f32; k * d];
    py.detach(|| {
        stage2(xs, cs, os, ds, ls, &caps[..k], &caps[k..2 * k], &caps[2 * k..3 * k], &caps[3 * k..], gs, n, d, m, ta,
               tm, &mut g);
    });
    let out: Vec<f64> = g.iter().map(|&v| v as f64).collect();
    Ok(PyArray1::from_vec(py, out).reshape([k, d])?)
}

/// Per-cap max over members of the angle 2 asin(|x - u_label| / 2), float64 (0 for an empty cap).
#[pyfunction]
fn assigned_theta<'py>(py: Python<'py>, x: PyReadonlyArray2<f64>, u: PyReadonlyArray2<f64>,
                       labels: PyReadonlyArray1<i64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (n, d) = (x.shape()[0], x.shape()[1]);
    let k = u.shape()[0];
    if u.shape()[1] != d || labels.len() != n {
        return Err(PyValueError::new_err("shape mismatch"));
    }
    let (xs, us) = (slice2(&x, "x")?, slice2(&u, "u")?);
    let ls = labels.as_slice().map_err(|_| PyValueError::new_err("labels must be contiguous"))?;
    check_cand(ls, k)?;
    let mut theta = vec![0f64; k];
    py.detach(|| {
        let mut chord = vec![0f64; k];
        for i in 0..n {
            let c = ls[i] as usize;
            let (xr, ur) = (&xs[i * d..(i + 1) * d], &us[c * d..(c + 1) * d]);
            let mut s = 0f64;
            for t in 0..d {
                let df = xr[t] - ur[t];
                s += df * df;
            }
            if s > chord[c] {
                chord[c] = s;
            }
        }
        for c in 0..k {
            theta[c] = 2.0 * (0.5 * chord[c].sqrt()).min(1.0).asin();
        }
    });
    Ok(PyArray1::from_vec(py, theta))
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn cache_impl<const FMA: bool>(x: &[f32], u: &[f32], idx: &[i64], bids: &[i64], ipb: &[f32], in_b: &[u8],
                               n: usize, d: usize, mc: usize, m: usize, bound: &[f64], margin: f64,
                               cand: &mut [i64], ipo: &mut [f32], ok: &mut [u8]) {
    let nb = bids.len();
    let mut buf: Vec<(f32, i64)> = Vec::with_capacity(mc + nb);
    for i in 0..n {
        buf.clear();
        let xr = &x[i * d..(i + 1) * d];
        for j in 0..mc {
            let c = idx[i * mc + j];
            if in_b[c as usize] == 0 {
                buf.push((dot::<FMA>(xr, &u[c as usize * d..(c as usize + 1) * d]), c));
            }
        }
        for (bi, &c) in bids.iter().enumerate() {
            buf.push((ipb[i * nb + bi], c));
        }
        let take = m.min(buf.len());
        let cmp = |a: &(f32, i64), b: &(f32, i64)| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal).then(a.1.cmp(&b.1));
        if take < buf.len() {
            buf.select_nth_unstable_by(take - 1, cmp);
        }
        buf[..take].sort_unstable_by(cmp);
        for j in 0..take {
            cand[i * m + j] = buf[j].1;
            ipo[i * m + j] = buf[j].0;
        }
        // The m-th best angle must beat every uncached cap's lower bound.
        let dm = (buf[take - 1].0 as f64).clamp(-1.0, 1.0).acos();
        ok[i] = (take == m && dm + margin <= bound[i]) as u8;
    }
}

dispatch!(cache_run, cache_impl, cache_avx, (x: &[f32], u: &[f32], idx: &[i64], bids: &[i64], ipb: &[f32],
    in_b: &[u8], n: usize, d: usize, mc: usize, m: usize, bound: &[f64], margin: f64, cand: &mut [i64],
    ipo: &mut [f32], ok: &mut [u8]));

/// For each row, the top-m caps (by float32 inner product, descending) among its cached candidates
/// `idx` (n x mc, ids not in `bids`' set are re-scored here) and the explicit caps `bids` (whose
/// inner products `ipb`, n x |B|, the caller computed). `in_b[c]` flags membership in `bids`.
/// ok[i] is 1 iff the m-th best angle + margin <= bound[i], i.e. the caller's lower bound on the
/// angle of any cap outside both sets, so that the row's top-m is exact.
#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn cache_topm<'py>(py: Python<'py>, x32: PyReadonlyArray2<f32>, u32: PyReadonlyArray2<f32>,
                   idx: PyReadonlyArray2<i64>, bids: PyReadonlyArray1<i64>, ipb: PyReadonlyArray2<f32>,
                   in_b: PyReadonlyArray1<u8>, m: usize, bound: PyReadonlyArray1<f64>, margin: f64)
                   -> PyResult<(Bound<'py, PyArray2<i64>>, Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<u8>>)> {
    let (n, d) = (x32.shape()[0], x32.shape()[1]);
    let (k, mc) = (u32.shape()[0], idx.shape()[1]);
    if idx.shape()[0] != n || bound.len() != n || in_b.len() != k || m == 0 || m > mc + bids.len() {
        return Err(PyValueError::new_err("shape mismatch"));
    }
    let nb = bids.len();
    if nb > 0 && (ipb.shape()[0] != n || ipb.shape()[1] != nb) {
        return Err(PyValueError::new_err("ipb must be n x |B|"));
    }
    let (xs, us, is) = (slice2(&x32, "x")?, slice2(&u32, "u")?, slice2(&idx, "idx")?);
    check_cand(is, k)?;
    let bs = bids.as_slice().map_err(|_| PyValueError::new_err("bids must be contiguous"))?;
    check_cand(bs, k)?;
    let ib = if nb > 0 { slice2(&ipb, "ipb")? } else { &[][..] };
    let inb = in_b.as_slice().map_err(|_| PyValueError::new_err("in_b must be contiguous"))?;
    let bd = bound.as_slice().map_err(|_| PyValueError::new_err("bound must be contiguous"))?;
    let (mut cand, mut ipo, mut ok) = (vec![0i64; n * m], vec![0f32; n * m], vec![0u8; n]);
    py.detach(|| cache_run(xs, us, is, bs, ib, inb, n, d, mc, m, bd, margin, &mut cand, &mut ipo, &mut ok));
    Ok((PyArray1::from_vec(py, cand).reshape([n, m])?, PyArray1::from_vec(py, ipo).reshape([n, m])?,
        PyArray1::from_vec(py, ok)))
}

/// The settling loop of CapCoverFilter._tight_angles for a cover without `neighbors`, exactly:
/// members are settled farthest-from-their-nearest-cap first (stable on ties); a member already
/// held by one of its candidate caps joins the first such cap, else its home cap if that holds it,
/// else the nearest candidate whose bound allows it grows to the member, else its home grows.
/// `labels` is n x K candidate caps per member (decoded directions `dec`), `bound` the per-cap
/// limits (raised here to the home members' own angles, as np.maximum.at does).
/// Returns (angles, owner).
#[pyfunction]
fn tight_angles<'py>(py: Python<'py>, x: PyReadonlyArray2<f64>, dec: PyReadonlyArray2<f64>,
                     labels: PyReadonlyArray2<i64>, home: PyReadonlyArray1<i64>, bound: PyReadonlyArray1<f64>)
                     -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<i64>>)> {
    let (n, d) = (x.shape()[0], x.shape()[1]);
    let (k, kk) = (dec.shape()[0], labels.shape()[1]);
    if dec.shape()[1] != d || labels.shape()[0] != n || home.len() != n || bound.len() != k {
        return Err(PyValueError::new_err("shape mismatch"));
    }
    let (xs, ds, ls) = (slice2(&x, "x")?, slice2(&dec, "dec")?, slice2(&labels, "labels")?);
    check_cand(ls, k)?;
    let hs = home.as_slice().map_err(|_| PyValueError::new_err("home must be contiguous"))?;
    check_cand(hs, k)?;
    let mut bnd = bound.as_slice().map_err(|_| PyValueError::new_err("bound must be contiguous"))?.to_vec();
    let mut angles = vec![0f64; k];
    let mut owner = vec![0i64; n];
    py.detach(|| {
        let ang = |i: usize, c: usize| -> f64 {
            let (xr, ur) = (&xs[i * d..(i + 1) * d], &ds[c * d..(c + 1) * d]);
            let mut s = 0f64;
            for t in 0..d {
                let df = xr[t] - ur[t];
                s += df * df;
            }
            2.0 * (0.5 * s.sqrt()).min(1.0).asin()
        };
        let mut dist = vec![0f64; n * kk];
        let mut home_d = vec![0f64; n];
        let mut nearest = vec![0f64; n];
        for i in 0..n {
            let mut mn = f64::INFINITY;
            for j in 0..kk {
                let v = ang(i, ls[i * kk + j] as usize);
                dist[i * kk + j] = v;
                if v < mn {
                    mn = v;
                }
            }
            nearest[i] = mn;
            let h = hs[i] as usize;
            home_d[i] = ang(i, h);
            if home_d[i] > bnd[h] {
                bnd[h] = home_d[i];
            }
        }
        let mut ord: Vec<usize> = (0..n).collect();
        ord.sort_by(|&a, &b| nearest[b].partial_cmp(&nearest[a]).unwrap_or(std::cmp::Ordering::Equal));
        let mut idx: Vec<usize> = Vec::with_capacity(kk);
        for &i in &ord {
            let (rl, rd) = (&ls[i * kk..(i + 1) * kk], &dist[i * kk..(i + 1) * kk]);
            if let Some(j) = (0..kk).find(|&j| rd[j] <= angles[rl[j] as usize]) {
                owner[i] = rl[j];
                continue;
            }
            let h = hs[i] as usize;
            if home_d[i] <= angles[h] {
                owner[i] = h as i64;
                continue;
            }
            idx.clear();
            idx.extend(0..kk);
            idx.sort_by(|&a, &b| rd[a].partial_cmp(&rd[b]).unwrap_or(std::cmp::Ordering::Equal));
            let mut placed = false;
            for &j in &idx {
                if rd[j] <= bnd[rl[j] as usize] {
                    angles[rl[j] as usize] = rd[j];
                    owner[i] = rl[j];
                    placed = true;
                    break;
                }
            }
            if !placed {
                angles[h] = home_d[i];
                owner[i] = h as i64;
            }
        }
    });
    Ok((PyArray1::from_vec(py, angles), PyArray1::from_vec(py, owner)))
}

/// The coordinate search of cap_quantizers._fit_scalar_codes, cap by cap (caps are independent):
/// for each pass and coordinate, a cap moves that coordinate one level down or up when that
/// raises the least member cosine to its decoded direction. Same arithmetic, same order, so the
/// result matches the NumPy version bit for bit.
///   xo      members sorted by cap (rows), starts/caps: segment s covers rows starts[s]..starts[s+1]
///           and belongs to cap caps[s]; c (k x d) decoded directions, idx (k x d) level indices,
///           dot (rows) member . decoded, norm2 (k) squared norms. Updated in place; c returned.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn fit_scalar_levels<'py>(py: Python<'py>, xo: PyReadonlyArray2<f64>, starts: PyReadonlyArray1<i64>,
                          caps: PyReadonlyArray1<i64>, c: PyReadonlyArray2<f64>, idx: PyReadonlyArray2<i64>,
                          dot: PyReadonlyArray1<f64>, norm2: PyReadonlyArray1<f64>, levels: PyReadonlyArray1<f64>,
                          passes: usize) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (rows, d) = (xo.shape()[0], xo.shape()[1]);
    let k = c.shape()[0];
    if c.shape()[1] != d || idx.shape()[0] != k || idx.shape()[1] != d || dot.len() != rows || norm2.len() != k
        || starts.len() != caps.len() {
        return Err(PyValueError::new_err("shape mismatch"));
    }
    let xs = slice2(&xo, "xo")?;
    let st = starts.as_slice().map_err(|_| PyValueError::new_err("starts must be contiguous"))?;
    let cp = caps.as_slice().map_err(|_| PyValueError::new_err("caps must be contiguous"))?;
    check_cand(cp, k)?;
    let mut cc = slice2(&c, "c")?.to_vec();
    let mut ix = slice2(&idx, "idx")?.to_vec();
    let lv = levels.as_slice().map_err(|_| PyValueError::new_err("levels must be contiguous"))?;
    let nl = lv.len() as i64;
    if ix.iter().any(|&v| v < 0 || v >= nl) {
        return Err(PyValueError::new_err("level index out of range"));
    }
    let mut dt = dot.as_slice().map_err(|_| PyValueError::new_err("dot must be contiguous"))?.to_vec();
    let mut n2 = norm2.as_slice().map_err(|_| PyValueError::new_err("norm2 must be contiguous"))?.to_vec();
    py.detach(|| {
        for s in 0..cp.len() {
            let cap = cp[s] as usize;
            let (a, b) = (st[s] as usize, if s + 1 < st.len() { st[s + 1] as usize } else { rows });
            for _ in 0..passes {
                for i in 0..d {
                    let sq = n2[cap].sqrt();
                    let mut cur = f64::INFINITY;
                    for r in a..b {
                        let v = dt[r] / sq;
                        if v < cur {
                            cur = v;
                        }
                    }
                    let (mut best, mut best_step) = (cur, 0i64);
                    for step in [-1i64, 1] {
                        let j = (ix[cap * d + i] + step).clamp(0, nl - 1);
                        let delta = lv[j as usize] - cc[cap * d + i];
                        let trial_n2 = n2[cap] + delta * (2.0 * cc[cap * d + i] + delta);
                        let tsq = trial_n2.sqrt();
                        let mut val = f64::INFINITY;
                        for r in a..b {
                            let v = (dt[r] + xs[r * d + i] * delta) / tsq;
                            if v < val {
                                val = v;
                            }
                        }
                        if val > best && delta != 0.0 {
                            best = val;
                            best_step = step;
                        }
                    }
                    if best_step != 0 {
                        let j = (ix[cap * d + i] + best_step).clamp(0, nl - 1);
                        let delta = lv[j as usize] - cc[cap * d + i];
                        for r in a..b {
                            dt[r] += xs[r * d + i] * delta;
                        }
                        n2[cap] += delta * (2.0 * cc[cap * d + i] + delta);
                        cc[cap * d + i] = lv[j as usize];
                        ix[cap * d + i] = j;
                    }
                }
            }
        }
    });
    Ok(PyArray1::from_vec(py, cc).reshape([k, d])?)
}

/// One Riemannian Adam step with a per-cap scalar second moment, rows independent, float64 like the
/// NumPy loop it replaces: tangent gradient, moments (`m1`, `v` updated in place), step in radians,
/// renormalised direction, momentum re-projected onto the new tangent. Returns the new directions.
/// `bc1`, `bc2` are the bias corrections 1 - beta^(t+1).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn adam_step<'py>(py: Python<'py>, g: PyReadonlyArray2<f64>, u: PyReadonlyArray2<f64>,
                  mut m1: numpy::PyReadwriteArray2<f64>, mut v: numpy::PyReadwriteArray1<f64>, lr: f64, beta1: f64,
                  beta2: f64, bc1: f64, bc2: f64) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (k, d) = (u.shape()[0], u.shape()[1]);
    if g.shape() != u.shape() || m1.shape() != u.shape() || v.len() != k {
        return Err(PyValueError::new_err("shape mismatch"));
    }
    let (gs, us) = (slice2(&g, "g")?, slice2(&u, "u")?);
    let ms = m1.as_slice_mut().map_err(|_| PyValueError::new_err("m1 must be contiguous"))?;
    let vs = v.as_slice_mut().map_err(|_| PyValueError::new_err("v must be contiguous"))?;
    let mut out = vec![0f64; k * d];
    py.detach(|| {
        let mut gt = vec![0f64; d];
        for r in 0..k {
            let (ur, gr) = (&us[r * d..(r + 1) * d], &gs[r * d..(r + 1) * d]);
            let mr = &mut ms[r * d..(r + 1) * d];
            let mut gu = 0f64;
            for t in 0..d {
                gu += gr[t] * ur[t];
            }
            let mut gn2 = 0f64;
            for t in 0..d {
                gt[t] = gr[t] - gu * ur[t];
                gn2 += gt[t] * gt[t];
            }
            vs[r] = beta2 * vs[r] + (1.0 - beta2) * gn2;
            let denom = (vs[r] / bc2).sqrt() + 1e-30;
            let mut nn = 0f64;
            let o = &mut out[r * d..(r + 1) * d];
            for t in 0..d {
                mr[t] = beta1 * mr[t] + (1.0 - beta1) * gt[t];
                o[t] = ur[t] - lr * ((mr[t] / bc1) / denom);
                nn += o[t] * o[t];
            }
            let inv = 1.0 / nn.sqrt().max(1e-300);
            let mut mu = 0f64;
            for t in 0..d {
                o[t] *= inv;
                mu += mr[t] * o[t];
            }
            for t in 0..d {
                mr[t] -= mu * o[t];
            }
        }
    });
    Ok(PyArray1::from_vec(py, out).reshape([k, d])?)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(adam_step, m)?)?;
    m.add_function(wrap_pyfunction!(fit_scalar_levels, m)?)?;
    m.add_function(wrap_pyfunction!(tight_angles, m)?)?;
    m.add_function(wrap_pyfunction!(anneal_stage1_py, m)?)?;
    m.add_function(wrap_pyfunction!(anneal_stage2_py, m)?)?;
    m.add_function(wrap_pyfunction!(assigned_theta, m)?)?;
    m.add_function(wrap_pyfunction!(cache_topm, m)?)?;
    Ok(())
}
