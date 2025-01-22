use crate::errors::VllmBackendError;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict, PyList};

pub struct EngineArgs {
    pub model: String,
    pub pipeline_parallel_size: u32,
    pub tensor_parallel_size: u32,
}

impl IntoPyDict for EngineArgs {
    fn into_py_dict_bound(self, py: Python<'_>) -> Bound<'_, PyDict> {
        PyDict::from_sequence_bound(
            PyList::new_bound(
                py,
                [
                    ("model", self.model.into_py(py)),
                    (
                        "pipeline_parallel_size",
                        self.pipeline_parallel_size.into_py(py),
                    ),
                    (
                        "tensor_parallel_size",
                        self.tensor_parallel_size.into_py(py),
                    ),
                ],
            )
            .as_any(),
        )
        .expect("Failed to create Python Dict from EngineArgs")
    }
}

// impl IntoPy<PyObject> for EngineArgs {
//     fn into_py(self, py: Python<'_>) -> PyObject {
//         PyDict::from_sequence_bound(
//             PyList::new_bound(
//                 py,
//                 [
//                     ("model", self.model.into_py(py)),
//                     (
//                         "pipeline_parallel_size",
//                         self.pipeline_parallel_size.into_py(py),
//                     ),
//                     (
//                         "tensor_parallel_size",
//                         self.tensor_parallel_size.into_py(py),
//                     ),
//                 ],
//             )
//             .as_any(),
//         )
//         .expect("Failed to create Python Dict from EngineArgs")
//     }
// }

pub struct LlmEngine {
    engine: PyObject,
}

impl LlmEngine {
    fn py_from_engine_args(args: EngineArgs) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            // Create the EngineArgs from Rust
            // from vllm.engine.arg_util import EngineArgs
            // engine_args = EngineArgs(**args)
            let py_engine_args_mod = PyModule::import_bound(py, "vllm.engine.arg_utils")?;
            let py_engine_args_class = py_engine_args_mod.getattr("EngineArgs")?;
            let py_engine_args =
                py_engine_args_class.call((), Some(&args.into_py_dict_bound(py)))?;

            // Next create the LLMEngine from the EngineArgs
            // from vllm.engine.llm_engine import LLMEngine
            // engine = LLMEngine.from_engine_args(engine_args)
            let py_engine_llm_mod = PyModule::import_bound(py, "vllm.engine.llm_engine")?;
            let py_engine_llm_class = py_engine_llm_mod.getattr("LLMEngine")?;
            py_engine_llm_class
                .call_method("from_engine_args", (py_engine_args,), None)?
                .extract()
        })
    }

    pub fn from_engine_args(args: EngineArgs) -> Result<LlmEngine, VllmBackendError> {
        let engine = Self::py_from_engine_args(args)?;

        Ok(Self { engine })
    }

    pub fn step(&mut self) {}
}
