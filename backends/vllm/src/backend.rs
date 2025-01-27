use crate::errors::VllmBackendError;
use crate::{EngineArgs, LlmEngine};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use std::thread::{spawn, JoinHandle};
use text_generation_router::infer::{Backend, InferError, InferStreamResponse};
use text_generation_router::validation::{
    ValidGenerateRequest, ValidParameters, ValidStoppingParameters,
};
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{debug, info, warn};

type InferResult = Result<InferStreamResponse, InferError>;

struct Request {
    tokens: Arc<Vec<u32>>,
    params: ValidParameters,
    stopping_params: ValidStoppingParameters,
    streamer: UnboundedSender<InferResult>,
}

pub struct VllmBackend {
    looper: JoinHandle<()>,
    waiting_requests: UnboundedSender<Request>,
}

impl VllmBackend {
    pub fn from_engine_args(args: EngineArgs) -> Result<VllmBackend, VllmBackendError> {
        let engine = LlmEngine::from_engine_args(args)?;
        let (sender, receiver) = unbounded_channel();
        let looper = spawn(|| engine_background_loop(engine, receiver));
        Ok(Self {
            looper,
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
            debug!("Attempt to queue new request");
            if let Err(err) = self.waiting_requests.send(Request {
                tokens: Arc::clone(&input_ids),
                params: request.parameters,
                stopping_params: request.stopping_parameters,
                streamer: sender,
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

fn engine_background_loop(mut engine: LlmEngine, mut waiting_requests: UnboundedReceiver<Request>) {
    info!("Starting vLLM engine background loop");

    let mut in_flight_requests = HashMap::with_capacity(256);
    loop {
        if !waiting_requests.is_empty() {
            let num_waiting_requests = waiting_requests.len();
            debug!(
                "Adding {} requests to the vLLM engine",
                num_waiting_requests
            );

            let mut requests = Vec::with_capacity(num_waiting_requests);
            waiting_requests.blocking_recv_many(&mut requests, num_waiting_requests);

            for request in requests {
                match engine.add_request(&request.tokens, &request.params, &request.stopping_params)
                {
                    Ok(request_id) => {
                        debug!("Successfully scheduled request {request_id}");
                        in_flight_requests.insert(request_id.to_string(), request);
                    }
                    Err(err) => {
                        warn!("Failed to schedule new request: {err}");
                    }
                }
            }
        }
        engine.step();
    }

    info!("Shutting down vLLM engine background loop");
}
