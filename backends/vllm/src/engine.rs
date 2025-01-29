use crate::errors::VllmBackendError;
use crate::{sampling_params, tokens_prompt, TryToPyObject};
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict, PyList, PyString};
use text_generation_router::validation::{ValidParameters, ValidStoppingParameters};
use tracing::info;
use uuid::Uuid;

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

pub struct SamplingParams<'a> {
    sampling_params: &'a ValidParameters,
    stopping_params: &'a ValidStoppingParameters,
}

impl TryToPyObject for SamplingParams<'_> {
    fn try_to_object(&self, py: Python<'_>) -> Result<PyObject, PyErr> {
        let py_sampling_params_class = sampling_params(py);

        let kwargs = PyDict::from_sequence_bound(&PyList::new_bound(
            py,
            [
                ("seed", self.sampling_params.seed.into_py(py)),
                ("n", 1.into_py(py)),
                ("top_k", self.sampling_params.top_k.into_py(py)),
                ("top_p", self.sampling_params.top_p.into_py(py)),
                ("temperature", self.sampling_params.temperature.into_py(py)),
                (
                    "frequency_penalty",
                    self.sampling_params.frequency_penalty.into_py(py),
                ),
                (
                    "repetition_penalty",
                    self.sampling_params.repetition_penalty.into_py(py),
                ),
                (
                    "ignore_eos",
                    self.stopping_params.ignore_eos_token.into_py(py),
                ),
                (
                    "max_tokens",
                    self.stopping_params.max_new_tokens.into_py(py),
                ),
                (
                    "stop",
                    PyList::new_bound(py, self.stopping_params.stop_sequences.iter()).into(),
                ),
            ],
        ));

        Ok(py_sampling_params_class
            .call_method_bound(py, "from_optional", (), Some(&kwargs?))?
            .to_object(py))
    }
}

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
            let py_engine_llm_mod = PyModule::import_bound(py, "vllm.v1.engine.llm_engine")?;
            let py_engine_llm_class = py_engine_llm_mod.getattr("LLMEngine")?;
            py_engine_llm_class
                .call_method("from_engine_args", (py_engine_args,), None)?
                .extract()
        })
    }

    fn py_add_request(
        &self,
        request_id: &str,
        prompt: &[u32],
        sampling_params: SamplingParams,
    ) -> Result<(), VllmBackendError> {
        Python::with_gil(|py| {
            // Create vllm.Tokens
            let kwargs = [("prompt_token_ids", prompt)].into_py_dict_bound(py);
            let py_tokens_prompt_class = tokens_prompt(py);
            let py_tokens_prompt = py_tokens_prompt_class.call_bound(py, (), Some(&kwargs))?;
            let py_sampling_params = sampling_params.try_to_object(py)?;

            self.engine.call_method1(
                py,
                "add_request",
                (
                    PyString::new_bound(py, request_id),
                    py_tokens_prompt,
                    py_sampling_params,
                ),
            )?;

            self.engine.call_method0(py, "step")
        })?;

        Ok(())
    }

    pub fn from_engine_args(args: EngineArgs) -> Result<LlmEngine, VllmBackendError> {
        let engine = Self::py_from_engine_args(args)?;

        Ok(Self { engine })
    }

    pub fn add_request(
        &self,
        prompt: &[u32],
        sampling_params: &ValidParameters,
        stopping_params: &ValidStoppingParameters,
    ) -> Result<Uuid, VllmBackendError> {
        let request_id = Uuid::new_v4();
        let sampling_params = SamplingParams {
            sampling_params,
            stopping_params,
        };
        self.py_add_request(&request_id.to_string(), prompt, sampling_params)?;

        info!("Submitted new request: {request_id}");
        Ok(request_id)
    }

    pub fn step(&mut self) {}
}
