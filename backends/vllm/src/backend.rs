use crate::errors::VllmBackendError;
use crate::{EngineArgs, LlmEngine};
use async_trait::async_trait;
use crossbeam_channel::internal::SelectHandle;
use crossbeam_channel::{unbounded, Receiver, RecvTimeoutError, Sender};
use std::collections::{HashMap, HashSet};
use std::hint::spin_loop;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::thread::{spawn, JoinHandle};
use std::time::Duration;
use text_generation_router::infer::{Backend, InferError, InferStreamResponse};
use text_generation_router::validation::{
    ValidGenerateRequest, ValidParameters, ValidStoppingParameters,
};
use text_generation_router::Token;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{debug, error, info, warn};

type InferResult = Result<InferStreamResponse, InferError>;

struct VllmRequestContext {
    tokens: Arc<Vec<u32>>,
    params: ValidParameters,
    stopping_params: ValidStoppingParameters,
    stream: UnboundedSender<InferResult>,
}

pub struct VllmBackend {
    waiting_requests: Sender<VllmRequestContext>,
}

impl VllmBackend {
    pub fn from_engine_args(args: EngineArgs) -> Result<VllmBackend, VllmBackendError> {
        let engine = LlmEngine::from_engine_args(args)?;
        let (sender, receiver) = unbounded();
        let looper = spawn(|| engine_background_loop(engine, receiver));
        Ok(Self {
            waiting_requests: sender,
        })
    }
}

#[async_trait]
impl Backend for VllmBackend {
    fn schedule(
        &self,
        request: ValidGenerateRequest,
    ) -> Result<UnboundedReceiverStream<Result<InferStreamResponse, InferError>>, InferError> {
        let (sender, receiver) = unbounded_channel();

        // Send the query to the vLLM Engine
        if let Some(input_ids) = request.input_ids {
            debug!("Queuing new request");
            if let Err(err) = self.waiting_requests.send(VllmRequestContext {
                tokens: Arc::clone(&input_ids),
                params: request.parameters,
                stopping_params: request.stopping_parameters,
                stream: sender,
            }) {
                warn!("Waiting Requests queue has been closed: {err}")
            }
        };

        Ok(UnboundedReceiverStream::new(receiver))
    }

    async fn health(&self, _current_health: bool) -> bool {
        true
    }
}

fn engine_background_loop(
    mut engine: LlmEngine,
    mut waiting_requests: Receiver<VllmRequestContext>,
) {
    info!("Starting vLLM engine background loop");
    static DURATION_100_MS: Duration = Duration::from_millis(100);
    let mut in_flight_requests = HashMap::with_capacity(256);
    'outer: loop {
        if !waiting_requests.is_empty() {
            match waiting_requests.recv_timeout(DURATION_100_MS) {
                Ok(context) => match engine.add_request(
                    &context.tokens,
                    &context.params,
                    &context.stopping_params,
                ) {
                    Ok(request_id) => {
                        debug!("Successfully scheduled request {request_id}");
                        in_flight_requests.insert(request_id.to_string(), context);
                    }
                    Err(err) => {
                        warn!("Failed to schedule new request: {err}");
                    }
                },
                Err(err) => match err {
                    RecvTimeoutError::Disconnected => break 'outer,
                    _ => {} // timeout all fine
                },
            }
        }

        if !in_flight_requests.is_empty() {
            match engine.step() {
                Ok(outputs) => outputs.iter().for_each(|output| {
                    let ctx = &in_flight_requests[&output.request_id];

                    // We only need to check on Err meaning the channel is not open anymore, so abort the request
                    if let Err(_) = ctx.stream.send(InferResult {}) {
                        debug!("Request {}'s channel dropped, aborting", &output.request_id);
                        in_flight_requests.remove(&output.request_id);
                        engine.abort_request(&output.request_id);
                    }
                }),
                Err(err) => {
                    error!("LLMEngine::step got an error: {err}");
                }
            }
        }

        spin_loop();
    }

    info!("Shutting down vLLM engine background loop");
}
