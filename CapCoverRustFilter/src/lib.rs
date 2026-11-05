pub mod core;
pub mod fastscan;
pub mod pq_direct;
pub mod construct_common;
pub mod construct_gonzalez;
pub mod construct_minimax;
pub mod construct_anneal;
pub mod construct_assign;
pub mod construct_tail;
#[cfg(target_arch = "x86_64")]
pub mod construct_knn_avx2;

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

use crate::core::{Codec, Grid, Group, Index};

#[pyclass]
struct CapIndex {
    inner: Index,
    pool: Option<rayon::ThreadPool>,
}

#[pymethods]
impl CapIndex {
    /// grid = (lo, hi, levels) of the angle grid.
    #[new]
    fn new(dim: usize, grid: (f64, f64, f64)) -> Self {
        CapIndex { inner: Index::new(dim, Grid { lo: grid.0, hi: grid.1, levels: grid.2 }), pool: None }
    }

    /// Adds a group of caps at one encoding. kind "tq": table = levels, bits;
    /// kind "pq": table = codebook (2^bits x sub), bits.
    fn add_group(&mut self, kind: &str, bits: usize, table: PyReadonlyArray2<f32>, codes: PyReadonlyArray2<u8>,
                 acodes: PyReadonlyArray1<u16>) -> PyResult<()> {
        let dim = self.inner.dim;
        let t = table.as_array();
        let codec = match kind {
            "tq" => Codec::Tq { bits, levels: t.iter().copied().collect() },
            "pq" => {
                let sub = t.shape()[1];
                Codec::Pq { bits, m: dim / sub, sub, codebook: t.iter().copied().collect() }
            }
            _ => return Err(PyValueError::new_err("unknown codec")),
        };
        let c = codes.as_array();
        if c.shape()[0] != acodes.len() { return Err(PyValueError::new_err("codes/angles length mismatch")); }
        self.inner.groups.push(Group::new(codec, dim, c.shape()[1], c.iter().copied().collect(),
                                          acodes.as_array().iter().copied().collect()));
        Ok(())
    }

    /// Bytes this index owns: (code bytes, angle-code bytes, codebooks and query-independent
    /// tables, which do not grow with the number of caps).
    fn owned_bytes(&self) -> (usize, usize, usize) {
        self.inner.owned_bytes()
    }

    /// Worker threads for query(): 1 (default) scans on the calling thread.
    fn set_threads(&mut self, n: usize) -> PyResult<()> {
        self.pool = if n <= 1 { None } else {
            Some(rayon::ThreadPoolBuilder::new().num_threads(n).build().map_err(|e| PyValueError::new_err(e.to_string()))?)
        };
        Ok(())
    }

    /// Group g's code bytes (n x code_size), for thawing the Python-side buffer.
    fn export_codes<'py>(&self, py: Python<'py>, g: usize) -> PyResult<Bound<'py, PyArray2<u8>>> {
        let grp = self.inner.groups.get(g).ok_or_else(|| PyValueError::new_err("no such group"))?;
        let rows: Vec<Vec<u8>> = (0..grp.acodes.len()).map(|k| {
            let mut c = vec![0u8; grp.code_size];
            grp.code_into(k, &mut c);
            c
        }).collect();
        PyArray2::from_vec2(py, &rows).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Accept/reject per row of `x` (float64, C-contiguous, in the working
    /// basis, i.e. already rotated: CapCoverFilter.to_working), threshold tp = tau'.
    fn query<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<f64>, tp: f64) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let xs = x.as_slice().map_err(|_| PyValueError::new_err("queries must be C-contiguous"))?;
        let out = py.detach(|| self.inner.query(xs, tp, self.pool.as_ref()));
        Ok(PyArray1::from_vec(py, out))
    }
}

/// (cap lanes tested by the FastScan kernel, tier-2 candidates, group scans) since import.
#[pyfunction]
fn fs_stats() -> (u64, u64, u64) { crate::fastscan::global_stats() }

/// Benchmark switch: groups added from now on use the FastScan kernel iff `on`.
#[pyfunction]
fn set_fastscan(on: bool) { crate::fastscan::ENABLED.store(on, std::sync::atomic::Ordering::Relaxed) }

#[pymodule]
fn capcover_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CapIndex>()?;
    m.add_function(wrap_pyfunction!(fs_stats, m)?)?;
    m.add_function(wrap_pyfunction!(set_fastscan, m)?)?;
    crate::construct_common::register(m)?;
    crate::construct_gonzalez::register(m)?;
    crate::construct_minimax::register(m)?;
    crate::construct_anneal::register(m)?;
    crate::construct_assign::register(m)?;
    crate::construct_tail::register(m)?;
    Ok(())
}
