//! Construction kernels for the minimax method (owner: its implementation agent).
//! Register #[pyfunction]s in `register`; lib.rs already calls it. Keep all
//! computation GIL-free (py.allow_threads) and single-threaded by default.
use pyo3::prelude::*;

pub fn register(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
