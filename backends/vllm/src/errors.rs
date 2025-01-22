use pyo3::PyErr;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum VllmBackendError {
    #[error("{0}")]
    Python(PyErr),
}

impl From<PyErr> for VllmBackendError {
    fn from(value: PyErr) -> Self {
        Self::Python(value)
    }
}
