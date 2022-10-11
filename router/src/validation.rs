use crate::server::GenerateRequest;
use tokenizers::tokenizer::Tokenizer;
use tokio::sync::{mpsc, oneshot};

#[derive(Debug)]
pub struct ValidationError {}

type ValidationRequest = (
    GenerateRequest,
    oneshot::Sender<Result<(usize, GenerateRequest), ValidationError>>,
);

#[derive(Debug, Clone)]
pub(crate) struct Validation {
    sender: mpsc::Sender<ValidationRequest>,
}

impl Validation {
    pub(crate) fn new(tokenizer: Tokenizer) -> Self {
        let (validation_sender, validation_receiver) = mpsc::channel(128);

        tokio::spawn(validation_task(tokenizer, validation_receiver));

        Self {
            sender: validation_sender,
        }
    }

    pub(crate) async fn validate(
        &self,
        request: GenerateRequest,
    ) -> Result<(usize, GenerateRequest), ValidationError> {
        let (sender, receiver) = oneshot::channel();
        self.sender.send((request, sender)).await.unwrap();
        receiver.await.unwrap()
    }
}

async fn validation_task(tokenizer: Tokenizer, mut receiver: mpsc::Receiver<ValidationRequest>) {
    while let Some((request, response_tx)) = receiver.recv().await {
        if request.parameters.temperature < 0.0 {
            response_tx.send(Err(ValidationError {})).unwrap_or(());
            continue;
        }
        if request.parameters.top_p <= 0.0 || request.parameters.top_p > 1.0 {
            response_tx.send(Err(ValidationError {})).unwrap_or(());
            continue;
        }
        if request.parameters.max_new_tokens > 512 {
            response_tx.send(Err(ValidationError {})).unwrap_or(());
            continue;
        }

        let inputs = tokenizer.encode(request.inputs.clone(), false).unwrap();
        let input_length = inputs.len();

        if input_length > 512 {
            response_tx.send(Err(ValidationError {})).unwrap_or(());
            continue;
        }

        response_tx.send(Ok((input_length, request))).unwrap_or(());
    }
    println!("drop here");
}
