//! Shared construction helpers (owner: coordinator). Fused float64 angle
//! measurement between rows of X and selected rows of U: no gathered copy of
//! U and no difference temporaries. angle = 2*asin(min(1, |a-b|/2)), the same
//! formula as cap_geometry.angle_between; the squared norm is accumulated in
//! four independent lanes (summation order differs from numpy's einsum at the
//! 1e-16 relative level, which the filter's coverage epsilon absorbs).
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[inline]
fn chord_angle(a: &[f64], b: &[f64]) -> f64 {
    let mut s = [0.0f64; 4];
    let (ca, cb) = (a.chunks_exact(4), b.chunks_exact(4));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for l in 0..4 {
            let d = x[l] - y[l];
            s[l] += d * d;
        }
    }
    let mut tail = 0.0;
    for (x, y) in ra.iter().zip(rb) {
        let d = x - y;
        tail += d * d;
    }
    let chord = ((s[0] + s[1]) + (s[2] + s[3]) + tail).sqrt();
    2.0 * (0.5 * chord).min(1.0).asin()
}

fn slice2<'a>(a: &'a PyReadonlyArray2<'_, f64>, what: &str) -> PyResult<(&'a [f64], usize, usize)> {
    let s = a.as_slice().map_err(|_| PyValueError::new_err(format!("{what} must be C-contiguous float64")))?;
    Ok((s, a.shape()[0], a.shape()[1]))
}

/// out[i] = angle(x[i], u[labels[i]]).
#[pyfunction]
fn row_angles<'py>(py: Python<'py>, x: PyReadonlyArray2<f64>, u: PyReadonlyArray2<f64>,
                   labels: PyReadonlyArray1<i64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (xs, n, d) = slice2(&x, "x")?;
    let (us, k, du) = slice2(&u, "u")?;
    let lab = labels.as_slice().map_err(|_| PyValueError::new_err("labels must be contiguous int64"))?;
    if du != d || lab.len() != n {
        return Err(PyValueError::new_err("shape mismatch"));
    }
    if lab.iter().any(|&l| l < 0 || l as usize >= k) {
        return Err(PyValueError::new_err("label out of range"));
    }
    let out: Vec<f64> = py.detach(|| {
        (0..n).map(|i| {
            let l = lab[i] as usize;
            chord_angle(&xs[i * d..(i + 1) * d], &us[l * d..(l + 1) * d])
        }).collect()
    });
    Ok(PyArray1::from_vec(py, out))
}

/// out[i, j] = angle(x[i], u[idx[i, j]]), +inf where idx[i, j] < 0.
#[pyfunction]
fn pair_angles<'py>(py: Python<'py>, x: PyReadonlyArray2<f64>, u: PyReadonlyArray2<f64>,
                    idx: PyReadonlyArray2<i64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (xs, n, d) = slice2(&x, "x")?;
    let (us, k, du) = slice2(&u, "u")?;
    let ids = idx.as_slice().map_err(|_| PyValueError::new_err("idx must be contiguous int64"))?;
    let m = idx.shape()[1];
    if du != d || idx.shape()[0] != n {
        return Err(PyValueError::new_err("shape mismatch"));
    }
    if ids.iter().any(|&l| l >= k as i64) {
        return Err(PyValueError::new_err("index out of range"));
    }
    let out: Vec<f64> = py.detach(|| {
        let mut out = vec![f64::INFINITY; n * m];
        for i in 0..n {
            let xi = &xs[i * d..(i + 1) * d];
            for j in 0..m {
                let l = ids[i * m + j];
                if l >= 0 {
                    let l = l as usize;
                    out[i * m + j] = chord_angle(xi, &us[l * d..(l + 1) * d]);
                }
            }
        }
        out
    });
    PyArray1::from_vec(py, out).reshape([n, m])
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(row_angles, m)?)?;
    m.add_function(wrap_pyfunction!(pair_angles, m)?)?;
    Ok(())
}
