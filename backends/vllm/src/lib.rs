mod backend;
mod engine;
mod errors;

pub use backend::VllmBackend;
pub use engine::{EngineArgs, LlmEngine};
pub use errors::VllmBackendError;
use pyo3::prelude::PyAnyMethods;
use pyo3::sync::GILOnceCell;
use pyo3::types::PyModule;
use pyo3::{Py, PyAny, PyErr, PyObject, Python};
use tokio::time::Instant;

pub(crate) const STARTUP_INSTANT: Instant = Instant::now();

static PY_TOKENS_PROMPT_CLASS: GILOnceCell<Py<PyAny>> = GILOnceCell::new();
static PY_SAMPLING_PARAMS_CLASS: GILOnceCell<Py<PyAny>> = GILOnceCell::new();

#[inline]
pub(crate) fn tokens_prompt(py: Python) -> &Py<PyAny> {
    PY_TOKENS_PROMPT_CLASS.get_or_init(py, || {
        PyModule::import_bound(py, "vllm.inputs")
            .expect("Failed to import vllm.inputs")
            .getattr("TokensPrompt")
            .expect("Failed to import vllm.inputs.TokensPrompt")
            .unbind()
    })
}

#[inline]
pub(crate) fn sampling_params(py: Python) -> &Py<PyAny> {
    PY_SAMPLING_PARAMS_CLASS.get_or_init(py, || {
        PyModule::import_bound(py, "vllm")
            .expect("Failed to import vllm")
            .getattr("SamplingParams")
            .expect("Failed to import vllm.SamplingParams")
            .unbind()
    })
}

pub(crate) trait TryToPyObject {
    fn try_to_object(&self, py: Python<'_>) -> Result<PyObject, PyErr>;
}
