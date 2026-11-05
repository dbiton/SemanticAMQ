//! Kernels for the shared tail of CapCoverFilter.construct() (everything after
//! the construction method): an exact top-k inner-product search, the member
//! settle loop of `_tight_angles`, and the scalar-code coordinate search of
//! `_fit_scalar_codes`. All three reproduce the NumPy/faiss reference.
//!
//! knn_ip: exact, not approximate. An int8 VNNI GEMM screens every (query,
//! database row) pair; a pair is only dropped when a rigorous bound
//! (Cauchy-Schwarz on the quantization residuals) shows it cannot be among the
//! query's k largest inner products. The survivors are re-scored with a plain
//! float32 dot and sorted. Result: the same index sets as an exhaustive float32
//! search (the float32 scores differ from BLAS's only by summation order,
//! ~1e-7, which can reorder pairs tied within that noise).
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

// ------------------------------------------------------------------ knn_ip

const PANEL: usize = 32; // database columns per packed panel
const RB: usize = 16; // queries per block (two 8-row VNNI tiles, or one 16-row AMX tile)
const WARM_PANELS: usize = 16; // panels scored unconditionally to seed the threshold

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum KnnPath { Amx, Vnni, Avx2, Faiss }

fn vnni_ok() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512vl")
            && std::is_x86_feature_detected!("avx512vnni")
            && std::env::var_os("CAPCOVER_TAIL_NOKNN").is_none()
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

fn avx2_ok() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        crate::construct_knn_avx2::available()
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Some Rust knn kernel (VNNI or AVX2) exists on this CPU, regardless of CAPCOVER_KNN.
pub fn knn_available() -> bool {
    vnni_ok() || avx2_ok()
}

/// The path knn_ip takes: CAPCOVER_KNN=amx|vnni|avx2|faiss forces one (an unavailable forced path
/// resolves to faiss); otherwise AMX > AVX-512 VNNI > AVX2+FMA > faiss.
fn knn_path() -> KnnPath {
    let amx = vnni_ok() && amx_ok();
    match std::env::var("CAPCOVER_KNN").ok().as_deref() {
        Some("amx") => if amx { KnnPath::Amx } else { KnnPath::Faiss },
        Some("vnni") => if vnni_ok() { KnnPath::Vnni } else { KnnPath::Faiss },
        Some("avx2") => if avx2_ok() { KnnPath::Avx2 } else { KnnPath::Faiss },
        Some("faiss") => KnnPath::Faiss,
        _ => if amx { KnnPath::Amx } else if vnni_ok() { KnnPath::Vnni } else if avx2_ok() { KnnPath::Avx2 } else { KnnPath::Faiss },
    }
}

#[pyfunction]
fn knn_path_py() -> &'static str {
    match knn_path() { KnnPath::Amx => "amx", KnnPath::Vnni => "vnni", KnnPath::Avx2 => "avx2", KnnPath::Faiss => "faiss" }
}

/// AMX int8 tiles usable (CPU flags, kernel permission granted, not disabled by CAPCOVER_TAIL_NOAMX).
fn amx_ok() -> bool {
    static OK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OK.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            if std::env::var_os("CAPCOVER_TAIL_NOAMX").is_some() {
                return false;
            }
            let info = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
            if !(info.contains(" amx_int8") && info.contains(" amx_tile")) {
                return false;
            }
            // arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)
            let r: i64;
            unsafe {
                std::arch::asm!("syscall", inlateout("rax") 158i64 => r, in("rdi") 0x1023i64, in("rsi") 18i64,
                                lateout("rcx") _, lateout("r11") _, options(nostack));
            }
            r == 0
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    })
}

struct Rows {
    n: usize,
    dpad: usize,
    q: Vec<i8>,       // n x dpad, symmetric int8
    scale: Vec<f32>,  // per row
    err: Vec<f32>,    // per row: ||x - scale*q||_2 (rigorously rounded up)
    norm: Vec<f32>,   // per row: ||x||_2 (rounded up)
}

fn quantize(x: &[f32], n: usize, d: usize, dpad: usize) -> Rows {
    let mut q = vec![0i8; n * dpad];
    let mut scale = vec![0f32; n];
    let mut err = vec![0f32; n];
    let mut norm = vec![0f32; n];
    for i in 0..n {
        let row = &x[i * d..(i + 1) * d];
        let mx = row.iter().fold(0f32, |a, &v| a.max(v.abs()));
        let s = if mx > 0.0 { mx / 127.0 } else { 1.0 };
        // float32 throughout (vectorizes); the 1e-3 relative margins below dwarf its ~1e-5 accumulation error.
        let (mut e2, mut n2) = ([0f32; 8], [0f32; 8]);
        let inv = 1.0 / s;
        let qrow = &mut q[i * dpad..i * dpad + d];
        for (cv, cq) in row.chunks(8).zip(qrow.chunks_mut(8)) {
            for l in 0..cv.len() {
                let v = cv[l];
                let qi = (v * inv).round().clamp(-127.0, 127.0);
                cq[l] = qi as i8;
                let r = v - s * qi;
                e2[l] += r * r;
                n2[l] += v * v;
            }
        }
        scale[i] = s;
        err[i] = ((e2.iter().sum::<f32>() as f64).sqrt() * (1.0 + 1e-3) + 1e-7) as f32;
        norm[i] = ((n2.iter().sum::<f32>() as f64).sqrt() * (1.0 + 1e-3) + 1e-7) as f32;
    }
    Rows { n, dpad, q, scale, err, norm }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx512f")]
    pub unsafe fn dot_f32(a: *const f32, b: *const f32, d: usize) -> f32 {
        let full = d / 16 * 16;
        let mut acc = _mm512_setzero_ps();
        let mut i = 0;
        while i < full {
            acc = _mm512_fmadd_ps(_mm512_loadu_ps(a.add(i)), _mm512_loadu_ps(b.add(i)), acc);
            i += 16;
        }
        let rem = d - full;
        if rem != 0 {
            let m: __mmask16 = ((1u32 << rem) - 1) as u16;
            acc = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(m, a.add(full)), _mm512_maskz_loadu_ps(m, b.add(full)), acc);
        }
        _mm512_reduce_add_ps(acc)
    }

    /// AMX: palette 1, tmm0..7 all 16 rows x 64 bytes. Per thread/block; pair with amx_release.
    pub unsafe fn amx_config() {
        let mut cfg = [0u8; 64];
        cfg[0] = 1;
        for i in 0..8 {
            cfg[16 + 2 * i] = 64;
            cfg[48 + i] = 16;
        }
        std::arch::asm!("ldtilecfg [{c}]", c = in(reg) cfg.as_ptr(), options(nostack));
    }

    pub unsafe fn amx_release() {
        std::arch::asm!("tilerelease", options(nostack));
    }

    /// 16 queries (u8 rows, stride dpad) x one 32-column panel via tdpbusd:
    /// acc[r*32 + col] (int32). tmm0/tmm1 = C halves, tmm2 = A, tmm3/tmm4 = B halves.
    pub unsafe fn amx_panel(xq: *const u8, dpad: usize, panel: *const i8, out: *mut i32) {
        let (sa, sb, sc) = (dpad, 128usize, 128usize);
        std::arch::asm!("tilezero tmm0", "tilezero tmm1", options(nostack));
        for c in 0..dpad / 64 {
            let a = xq.add(c * 64);
            let b0 = panel.add(c * 16 * 128);
            let b1 = b0.add(64);
            std::arch::asm!(
                "tileloadd tmm2, [{a} + {sa}]",
                "tileloadd tmm3, [{b0} + {sb}]",
                "tileloadd tmm4, [{b1} + {sb}]",
                "tdpbusd tmm0, tmm2, tmm3",
                "tdpbusd tmm1, tmm2, tmm4",
                a = in(reg) a, b0 = in(reg) b0, b1 = in(reg) b1, sa = in(reg) sa, sb = in(reg) sb,
                options(nostack, readonly));
        }
        std::arch::asm!(
            "tilestored [{o0} + {sc}], tmm0",
            "tilestored [{o1} + {sc}], tmm1",
            o0 = in(reg) out, o1 = in(reg) out.add(16), sc = in(reg) sc,
            options(nostack));
    }

    /// Bitmask (32 lanes) of the panel's columns whose approx score for one query
    /// is >= thr: approx = (acc - corr) * sx * scale.
    #[target_feature(enable = "avx512f,avx512bw")]
    pub unsafe fn hits(acc: *const i32, corr: *const i32, scale: *const f32, sx: f32, thr: f32, valid: u32) -> u32 {
        let (sxv, tv) = (_mm512_set1_ps(sx), _mm512_set1_ps(thr));
        let mut out = 0u32;
        for h in 0..2 {
            let raw = _mm512_sub_epi32(_mm512_loadu_si512(acc.add(h * 16) as *const _), _mm512_loadu_si512(corr.add(h * 16) as *const _));
            let a = _mm512_mul_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(raw), sxv), _mm512_loadu_ps(scale.add(h * 16)));
            out |= (_mm512_cmp_ps_mask::<_CMP_GE_OQ>(a, tv) as u32) << (16 * h);
        }
        out & valid
    }

    /// 8 queries (u8 rows, value + 128, stride dpad) x one 32-column panel.
    /// Returns the 16 raw int32 accumulator vectors (row-major, 2 per row).
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
    pub unsafe fn tile(xq: *const u8, dpad: usize, panel: *const i8, out: *mut i32) {
        let mut a = [_mm512_setzero_si512(); 16];
        let steps = dpad / 4;
        for s in 0..steps {
            let b0 = _mm512_loadu_si512(panel.add(s * 128) as *const _);
            let b1 = _mm512_loadu_si512(panel.add(s * 128 + 64) as *const _);
            for r in 0..8 {
                let x = _mm512_set1_epi32((xq.add(r * dpad + s * 4) as *const i32).read_unaligned());
                a[2 * r] = _mm512_dpbusd_epi32(a[2 * r], x, b0);
                a[2 * r + 1] = _mm512_dpbusd_epi32(a[2 * r + 1], x, b1);
            }
        }
        for (i, v) in a.iter().enumerate() {
            _mm512_storeu_si512(out.add(i * 16) as *mut _, *v);
        }
    }
}

/// Smallest threshold keeping the k largest scores: returns the k-th largest
/// approx score of `cand` (requires len >= k >= 1).
fn kth_largest(cand: &[(f32, u32)], k: usize) -> f32 {
    let mut v: Vec<f32> = cand.iter().map(|c| c.0).collect();
    let (_, kth, _) = v.select_nth_unstable_by(k - 1, |a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    *kth
}

struct Db {
    m: usize,
    dpad: usize,
    panels: Vec<i8>,   // ceil(m/32) panels, each (dpad/4) steps x 32 cols x 4 bytes
    corr: Vec<i32>,    // per padded column: 128 * sum(q)
    scale: Vec<f32>,   // per padded column (0 for padding)
    err_max: f32,
    norm_max: f32,
}

fn pack_db(r: &Rows) -> Db {
    let np = (r.n + PANEL - 1) / PANEL;
    let steps = r.dpad / 4;
    let mut panels = vec![0i8; np * steps * PANEL * 4];
    let mut corr = vec![0i32; np * PANEL];
    let mut scale = vec![0f32; np * PANEL];
    for j in 0..r.n {
        let (p, c) = (j / PANEL, j % PANEL);
        let row = &r.q[j * r.dpad..(j + 1) * r.dpad];
        let mut s = 0i32;
        for (k, &v) in row.iter().enumerate() {
            let (st, b) = (k / 4, k % 4);
            panels[p * steps * PANEL * 4 + st * PANEL * 4 + c * 4 + b] = v;
            s += v as i32;
        }
        corr[j] = 128 * s;
        scale[j] = r.scale[j];
    }
    let err_max = r.err.iter().fold(0f32, |a, &v| a.max(v));
    let norm_max = r.norm.iter().fold(0f32, |a, &v| a.max(v));
    Db { m: r.n, dpad: r.dpad, panels, corr, scale, err_max, norm_max }
}

/// Top-k (descending inner product) of queries[i0..i0+nq) against the database.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
unsafe fn search_block(amx: bool, qx: &Rows, xu8: &[u8], x: &[f32], db: &Db, dbf: &[f32], d: usize, k: usize,
                       i0: usize, ip_out: &mut [f32], id_out: &mut [i64]) {
    let nq = RB.min(qx.n - i0);
    let dpad = qx.dpad;
    let steps = dpad / 4;
    let np = db.m.div_ceil(PANEL);
    // 8 query rows (pad with zero rows = value 128).
    let mut xb = vec![128u8; RB * dpad];
    xb[..nq * dpad].copy_from_slice(&xu8[i0 * dpad..(i0 + nq) * dpad]);
    let mut eps = [0f32; RB];
    for r in 0..nq {
        let i = i0 + r;
        let ex = qx.err[i];
        // |x.d - xq.dq| <= (|x|+ex)*ed + ex*(|d|+ed) + ex*ed ; + slack for the float32 rescale.
        eps[r] = (qx.norm[i] + ex) * db.err_max + ex * (db.norm_max + db.err_max) + ex * db.err_max + 3e-5;
    }
    let mut thr = [f32::NEG_INFINITY; RB];
    let mut cand: Vec<Vec<(f32, u32)>> = (0..RB).map(|_| Vec::new()).collect();
    let mut acc = [0i32; RB * PANEL];
    if amx { x86::amx_config(); }
    let warm = WARM_PANELS.min(np);
    for p in 0..np {
        let panel = db.panels.as_ptr().add(p * steps * PANEL * 4);
        if amx {
            x86::amx_panel(xb.as_ptr(), dpad, panel, acc.as_mut_ptr());
        } else {
            x86::tile(xb.as_ptr(), dpad, panel, acc.as_mut_ptr());
            x86::tile(xb.as_ptr().add(8 * dpad), dpad, panel, acc.as_mut_ptr().add(8 * PANEL));
        }
        let valid = (db.m - p * PANEL).min(PANEL);
        let vmask: u32 = if valid >= 32 { u32::MAX } else { (1u32 << valid) - 1 };
        for r in 0..nq {
            let sx = qx.scale[i0 + r];
            let c = &mut cand[r];
            let mut mask = x86::hits(acc.as_ptr().add(r * PANEL), db.corr.as_ptr().add(p * PANEL),
                                     db.scale.as_ptr().add(p * PANEL), sx, thr[r], vmask);
            while mask != 0 {
                let col = mask.trailing_zeros() as usize;
                mask &= mask - 1;
                let j = p * PANEL + col;
                let raw = acc[r * PANEL + col] - db.corr[j];
                c.push(((raw as f32) * sx * db.scale[j], j as u32));
            }
            if p + 1 == warm || (p >= warm && (c.len() > 512 || (p + 1) % 16 == 0)) {
                if c.len() >= k {
                    let t = kth_largest(c, k) - 2.0 * eps[r];
                    if t > thr[r] {
                        thr[r] = t;
                    }
                    c.retain(|e| e.0 >= thr[r]);
                }
            }
        }
    }
    if amx {
        x86::amx_release();
    }
    for r in 0..nq {
        let i = i0 + r;
        let c = &mut cand[r];
        if c.len() >= k {
            let t = kth_largest(c, k) - 2.0 * eps[r];
            c.retain(|e| e.0 >= t);
        }
        let xr = x.as_ptr().add(i * d);
        let mut sc: Vec<(f32, u32)> = c.iter().map(|&(_, j)| (x86::dot_f32(xr, dbf.as_ptr().add(j as usize * d), d), j)).collect();
        sc.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal).then(a.1.cmp(&b.1)));
        for t in 0..k {
            // Fewer than k survivors only happens for non-finite rows; pad rather than index out of bounds.
            let (v, j) = sc.get(t).map_or((f32::NEG_INFINITY, -1i64), |e| (e.0, e.1 as i64));
            ip_out[r * k + t] = v;
            id_out[r * k + t] = j;
        }
    }
}

#[pyfunction]
fn knn_available_py() -> bool {
    knn_available()
}

/// (ip, idx) of each row of `x` (n x d float32) against `db` (m x d float32):
/// the k = min(k, m) largest inner products, descending.
#[pyfunction]
#[pyo3(signature = (x, db, k, threads=1))]
fn knn_ip<'py>(py: Python<'py>, x: PyReadonlyArray2<f32>, db: PyReadonlyArray2<f32>, k: usize, threads: usize)
               -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<i64>>)> {
    let path = knn_path();
    if path == KnnPath::Faiss {
        return Err(PyValueError::new_err("knn_ip: no Rust kernel selected (CAPCOVER_KNN=faiss or unsupported CPU)"));
    }
    let xs = x.as_slice().map_err(|_| PyValueError::new_err("x must be C-contiguous float32"))?;
    let ds = db.as_slice().map_err(|_| PyValueError::new_err("db must be C-contiguous float32"))?;
    let (n, d) = (x.shape()[0], x.shape()[1]);
    let m = db.shape()[0];
    if db.shape()[1] != d {
        return Err(PyValueError::new_err("dimension mismatch"));
    }
    let k = k.min(m);
    if n == 0 || k == 0 {
        return Ok((PyArray2::zeros(py, [n, k], false), PyArray2::zeros(py, [n, k], false)));
    }
    #[cfg(target_arch = "x86_64")]
    if path == KnnPath::Avx2 {
        if d == 0 || !xs.iter().all(|v| v.is_finite()) || !ds.iter().all(|v| v.is_finite()) {
            return Err(PyValueError::new_err("knn_ip: empty or non-finite input"));
        }
        let (ip, id) = py.detach(|| crate::construct_knn_avx2::knn(xs, n, ds, m, d, k, threads));
        let ipa = PyArray1::from_vec(py, ip).reshape([n, k])?;
        let ida = PyArray1::from_vec(py, id).reshape([n, k])?;
        return Ok((ipa, ida));
    }
    #[cfg(target_arch = "x86_64")]
    {
        let amx = path == KnnPath::Amx;
        let dpad = (d + 63) / 64 * 64;
        let mut ip = vec![0f32; n * k];
        let mut id = vec![0i64; n * k];
        py.detach(|| {
            let qx = quantize(xs, n, d, dpad);
            let qd = quantize(ds, m, d, dpad);
            let db_packed = pack_db(&qd);
            let xu8: Vec<u8> = qx.q.iter().map(|&v| (v as i16 + 128) as u8).collect();
            let nblocks = n.div_ceil(RB);
            let run = |b: usize, ipc: &mut [f32], idc: &mut [i64]| unsafe {
                search_block(amx, &qx, &xu8, xs, &db_packed, ds, d, k, b * RB, ipc, idc);
            };
            if threads <= 1 {
                for (b, (ipc, idc)) in ip.chunks_mut(RB * k).zip(id.chunks_mut(RB * k)).enumerate() {
                    run(b, ipc, idc);
                }
            } else if let Ok(pool) = rayon::ThreadPoolBuilder::new().num_threads(threads).build() {
                pool.install(|| {
                    ip.par_chunks_mut(RB * k).zip(id.par_chunks_mut(RB * k)).enumerate().for_each(|(b, (ipc, idc))| run(b, ipc, idc));
                });
            }
            let _ = nblocks;
        });
        let ipa = PyArray1::from_vec(py, ip).reshape([n, k])?;
        let ida = PyArray1::from_vec(py, id).reshape([n, k])?;
        return Ok((ipa, ida));
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        return Err(PyValueError::new_err("knn_ip: no Rust kernel on this architecture"));
    }
}

// ------------------------------------------------------------------ settle

/// The member settle loop of CapCoverFilter._tight_angles. labels/dist are the
/// (n x L) candidate caps and member-to-cap angles (label < 0: no candidate);
/// home/home_d the construction cap and its angle; bound[j] the most cap j may
/// grow to. Members are visited by decreasing nearest distance (stable).
/// Returns (angles, owner).
#[pyfunction]
fn settle<'py>(py: Python<'py>, labels: PyReadonlyArray2<i64>, dist: PyReadonlyArray2<f64>, home: PyReadonlyArray1<i64>,
               home_d: PyReadonlyArray1<f64>, bound: PyReadonlyArray1<f64>)
               -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<i64>>)> {
    let ls = labels.as_slice().map_err(|_| PyValueError::new_err("labels must be C-contiguous int64"))?;
    let ds = dist.as_slice().map_err(|_| PyValueError::new_err("dist must be C-contiguous float64"))?;
    let (n, l) = (labels.shape()[0], labels.shape()[1]);
    let hm = home.as_slice().map_err(|_| PyValueError::new_err("home"))?;
    let hd = home_d.as_slice().map_err(|_| PyValueError::new_err("home_d"))?;
    let bd = bound.as_slice().map_err(|_| PyValueError::new_err("bound"))?;
    let nc = bd.len();
    if dist.shape() != labels.shape() || hm.len() != n || hd.len() != n {
        return Err(PyValueError::new_err("shape mismatch"));
    }
    if hm.iter().any(|&h| h < 0 || h as usize >= nc) || ls.iter().any(|&v| v >= nc as i64) {
        return Err(PyValueError::new_err("label out of range"));
    }
    let (angles, owner) = py.detach(|| {
        let mut angles = vec![0f64; nc];
        let mut owner = vec![0i64; n];
        let nearest: Vec<f64> = (0..n).map(|i| ds[i * l..(i + 1) * l].iter().cloned().fold(f64::INFINITY, f64::min)).collect();
        let mut order: Vec<u32> = (0..n as u32).collect();
        // argsort(-nearest, stable): decreasing nearest, ties by index.
        order.sort_by(|&a, &b| (-nearest[a as usize]).partial_cmp(&(-nearest[b as usize])).unwrap_or(std::cmp::Ordering::Equal));
        let mut idx: Vec<usize> = Vec::with_capacity(l);
        for &iu in &order {
            let i = iu as usize;
            let (rl, rd) = (&ls[i * l..(i + 1) * l], &ds[i * l..(i + 1) * l]);
            if let Some(j) = (0..l).find(|&j| rl[j] >= 0 && rd[j] <= angles[rl[j] as usize]) {
                owner[i] = rl[j];
                continue;
            }
            let h = hm[i] as usize;
            if hd[i] <= angles[h] {
                owner[i] = h as i64;
                continue;
            }
            idx.clear();
            idx.extend(0..l);
            idx.sort_by(|&a, &b| rd[a].partial_cmp(&rd[b]).unwrap_or(std::cmp::Ordering::Equal));
            let mut done = false;
            for &j in &idx {
                if rl[j] >= 0 && rd[j] <= bd[rl[j] as usize] {
                    angles[rl[j] as usize] = rd[j];
                    owner[i] = rl[j];
                    done = true;
                    break;
                }
            }
            if !done {
                angles[h] = hd[i];
                owner[i] = h as i64;
            }
        }
        (angles, owner)
    });
    Ok((PyArray1::from_vec(py, angles), PyArray1::from_vec(py, owner)))
}

// -------------------------------------------------------- fit_scalar_codes

#[inline]
fn nan_min(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() { f64::NAN } else if b < a { b } else { a }
}

/// _fit_scalar_codes' coordinate search, one cap at a time (caps do not
/// interact): c0 = decoded start (ncaps x d, float64), x = member rows,
/// labels[i] the cap of row i, levels ascending. Returns the fitted levels
/// (ncaps x d, float64).
#[pyfunction]
fn fit_scalar<'py>(py: Python<'py>, c0: PyReadonlyArray2<f64>, x: PyReadonlyArray2<f64>, labels: PyReadonlyArray1<i64>,
                   levels: PyReadonlyArray1<f64>, passes: usize) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let cs = c0.as_slice().map_err(|_| PyValueError::new_err("c0 must be C-contiguous float64"))?;
    let xs = x.as_slice().map_err(|_| PyValueError::new_err("x must be C-contiguous float64"))?;
    let lv = levels.as_slice().map_err(|_| PyValueError::new_err("levels"))?;
    let lab = labels.as_slice().map_err(|_| PyValueError::new_err("labels"))?;
    let (nc, d) = (c0.shape()[0], c0.shape()[1]);
    let n = x.shape()[0];
    if x.shape()[1] != d || lab.len() != n || lv.is_empty() || lab.iter().any(|&v| v < 0 || v as usize >= nc) {
        return Err(PyValueError::new_err("shape mismatch"));
    }
    let nl = lv.len();
    let out = py.detach(|| {
        let mut c = cs.to_vec();
        // members per cap, ascending row order
        let mut start = vec![0usize; nc + 1];
        for &v in lab { start[v as usize + 1] += 1; }
        for i in 0..nc { start[i + 1] += start[i]; }
        let mut fill = start.clone();
        let mut mem = vec![0u32; n];
        for (i, &v) in lab.iter().enumerate() { mem[fill[v as usize]] = i as u32; fill[v as usize] += 1; }
        let mut chunks: Vec<&mut [f64]> = c.chunks_mut(d).collect();
        chunks.par_iter_mut().enumerate().for_each(|(cap, crow)| {
            let ms = &mem[start[cap]..start[cap + 1]];
            if ms.is_empty() { return; }
            let mut idx: Vec<usize> = crow.iter().map(|&v| {
                let t = v - 1e-12;
                lv.partition_point(|&l| l < t).min(nl - 1)
            }).collect();
            let mut dot: Vec<f64> = ms.iter().map(|&m| {
                let row = &xs[m as usize * d..(m as usize + 1) * d];
                row.iter().zip(crow.iter()).map(|(a, b)| a * b).sum::<f64>()
            }).collect();
            let mut norm2: f64 = crow.iter().map(|v| v * v).sum();
            for _ in 0..passes {
                for i in 0..d {
                    let mut best = f64::INFINITY;
                    let s = norm2.sqrt();
                    for &dt in &dot { best = nan_min(best, dt / s); }
                    let mut best_step = 0i64;
                    for step in [-1i64, 1] {
                        let j = (idx[i] as i64 + step).clamp(0, nl as i64 - 1) as usize;
                        let delta = lv[j] - crow[i];
                        let tn = norm2 + delta * (2.0 * crow[i] + delta);
                        let sq = tn.sqrt();
                        let mut val = f64::INFINITY;
                        for (t, &m) in ms.iter().enumerate() {
                            val = nan_min(val, (dot[t] + xs[m as usize * d + i] * delta) / sq);
                        }
                        if val > best && delta != 0.0 { best = val; best_step = step; }
                    }
                    if best_step != 0 {
                        let j = (idx[i] as i64 + best_step).clamp(0, nl as i64 - 1) as usize;
                        let delta = lv[j] - crow[i];
                        for (t, &m) in ms.iter().enumerate() { dot[t] += xs[m as usize * d + i] * delta; }
                        norm2 += delta * (2.0 * crow[i] + delta);
                        crow[i] = lv[j];
                        idx[i] = j;
                    }
                }
            }
        });
        c
    });
    Ok(PyArray1::from_vec(py, out).reshape([nc, d])?)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(knn_ip, m)?)?;
    m.add_function(wrap_pyfunction!(knn_available_py, m)?)?;
    m.add_function(wrap_pyfunction!(knn_path_py, m)?)?;
    m.add_function(wrap_pyfunction!(settle, m)?)?;
    m.add_function(wrap_pyfunction!(fit_scalar, m)?)?;
    Ok(())
}
