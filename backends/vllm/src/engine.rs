use crate::errors::VllmBackendError;
use crate::{sampling_params, tokens_prompt, TryToPyObject};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::sync::GILOnceCell;
use pyo3::types::{IntoPyDict, PyDict, PyList, PyString};
use text_generation_router::validation::{ValidParameters, ValidStoppingParameters};
use tracing::{info, instrument};
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

static FINAL_OUTPUT_ONLY: GILOnceCell<PyObject> = GILOnceCell::new();

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
                (intern!(py, "output_kind"), 2.into_py(py)),
                (intern!(py, "logprobs"), 1.into_py(py)),
                (intern!(py, "n"), 1.into_py(py)),
                (intern!(py, "seed"), self.sampling_params.seed.into_py(py)),
                (intern!(py, "top_k"), self.sampling_params.top_k.into_py(py)),
                (intern!(py, "top_p"), self.sampling_params.top_p.into_py(py)),
                (
                    intern!(py, "temperature"),
                    self.sampling_params.temperature.into_py(py),
                ),
                (
                    intern!(py, "frequency_penalty"),
                    self.sampling_params.frequency_penalty.into_py(py),
                ),
                (
                    intern!(py, "repetition_penalty"),
                    self.sampling_params.repetition_penalty.into_py(py),
                ),
                (
                    intern!(py, "ignore_eos"),
                    self.stopping_params.ignore_eos_token.into_py(py),
                ),
                (
                    intern!(py, "max_tokens"),
                    self.stopping_params.max_new_tokens.into_py(py),
                ),
                (
                    intern!(py, "stop"),
                    PyList::new_bound(py, self.stopping_params.stop_sequences.iter()).into(),
                ),
            ],
        ));

        Ok(py_sampling_params_class
            .call_method_bound(py, "from_optional", (), Some(&kwargs?))?
            .to_object(py))
    }
}

#[derive(Debug)]
pub(crate) struct CompletionOutput {
    pub token_ids: Vec<u32>, // TODO: TinyVec?
    pub text: String,        // TODO: SmallString?
    // pub logprobs: Vec<f32>,            // TODO: TinyVec?
    pub finish_reason: Option<String>, // lora_request: LATER
    pub index: usize,
}

#[derive(Debug, Copy, Clone)]
pub(crate) struct RequestMetrics {
    pub arrival_time: f32,
    pub first_scheduled_time: f32,
    pub first_token_time: f32,
    pub time_in_queue: f32,
}

impl<'py> FromPyObject<'py> for RequestMetrics {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let py = ob.py();
        Ok(Self {
            arrival_time: ob.getattr(intern!(py, "arrival_time"))?.extract()?,
            first_scheduled_time: ob.getattr(intern!(py, "first_scheduled_time"))?.extract()?,
            first_token_time: ob.getattr(intern!(py, "first_token_time"))?.extract()?,
            time_in_queue: ob.getattr(intern!(py, "time_in_queue"))?.extract()?,
        })
    }
}

#[derive(Debug)]
pub(crate) struct RequestOutput {
    pub outputs: Vec<CompletionOutput>,
    // pub metrics: Vec<RequestMetrics>,
    pub request_id: String,
    pub finished: bool,
}

impl<'py> FromPyObject<'py> for CompletionOutput {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let py = ob.py();
        Ok(Self {
            index: ob.getattr(intern!(py, "index"))?.extract()?,
            text: ob.getattr(intern!(py, "text"))?.extract()?,
            token_ids: ob.getattr(intern!(py, "token_ids"))?.extract()?,
            // logprobs: ob.getattr(intern!(py, "logprobs"))?.extract()?,
            finish_reason: ob.getattr(intern!(py, "finish_reason"))?.extract()?,
        })
    }
}

impl<'py> FromPyObject<'py> for RequestOutput {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let py = ob.py();
        Ok(Self {
            request_id: ob.getattr(intern!(py, "request_id"))?.extract()?,
            outputs: ob.getattr(intern!(py, "outputs"))?.extract()?,
            finished: ob.getattr(intern!(py, "finished"))?.extract()?,
            // metrics: ob.getattr(intern!(py, "metrics"))?.extract()?,
        })
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
            let kwargs = [(intern!(py, "prompt_token_ids"), prompt)].into_py_dict_bound(py);
            let py_tokens_prompt_class = tokens_prompt(py);
            let py_tokens_prompt = py_tokens_prompt_class.call_bound(py, (), Some(&kwargs))?;
            let py_sampling_params = sampling_params.try_to_object(py)?;

            self.engine.call_method1(
                py,
                intern!(py, "add_request"),
                (
                    PyString::new_bound(py, request_id),
                    py_tokens_prompt,
                    py_sampling_params,
                ),
            )?;

            self.engine.call_method0(py, intern!(py, "step"))
        })?;

        Ok(())
    }

    fn py_step(&self) -> Result<Vec<RequestOutput>, VllmBackendError> {
        Ok(Python::with_gil(|py| {
            self.engine
                .call_method0(py, intern!(py, "step"))?
                .extract::<Vec<RequestOutput>>(py)
        })?)
    }

    pub fn from_engine_args(args: EngineArgs) -> Result<LlmEngine, VllmBackendError> {
        let engine = Self::py_from_engine_args(args)?;

        Ok(Self { engine })
    }

    #[instrument(skip_all)]
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

    #[instrument(skip_all)]
    pub fn abort_request(&self, _request_id: &str) {}

    #[instrument(skip_all)]
    pub fn step(&mut self) -> Result<Vec<RequestOutput>, VllmBackendError> {
        self.py_step()
    }
}
