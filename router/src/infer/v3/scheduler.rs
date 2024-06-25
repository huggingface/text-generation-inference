/// Batching and inference logic
use crate::infer::v3::queue::{Entry, Queue};
use crate::infer::{
    GenerateStreamResponse, GeneratedText, InferError, InferStreamResponse, Scheduler,
};
use crate::validation::ValidGenerateRequest;
use crate::{FinishReason, PrefillToken, Token};
use nohash_hasher::IntMap;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use text_generation_client::v3::{Batch, CachedBatch, Generation, ShardedClient};
use text_generation_client::ClientError;
use tokio::sync::mpsc::error::SendError;
use tokio::sync::{mpsc, Notify, OwnedSemaphorePermit};
use tokio::time::Instant;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{info_span, instrument, Instrument, Span};

pub(crate) struct SchedulerV3 {
    /// Request queue
    queue: Queue,
    /// Notify batcher on queue appends
    batching_task_notifier: Arc<Notify>,
}

impl SchedulerV3 {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        client: ShardedClient,
        waiting_served_ratio: f32,
        max_batch_prefill_tokens: u32,
        max_batch_total_tokens: u32,
        max_waiting_tokens: usize,
        max_batch_size: Option<usize>,
        requires_padding: bool,
        window_size: Option<u32>,
        speculate: u32,
        generation_health: Arc<AtomicBool>,
    ) -> Self {
        let flashdecoding = if let Ok(flashdecoding) = std::env::var("FLASH_DECODING") {
            matches!(flashdecoding.to_lowercase().as_str(), "1" | "true")
        } else {
            false
        };
        let block_size = if flashdecoding { 256 } else { 16 };
        let queue = Queue::new(
            requires_padding,
            block_size,
            window_size,
            speculate,
            max_batch_total_tokens,
        );
        let batching_task_notifier = Arc::new(Notify::new());

        // Spawn batching background task that contains all the inference logic
        tokio::spawn(batching_task(
            client,
            waiting_served_ratio,
            max_batch_prefill_tokens,
            max_batch_total_tokens,
            max_waiting_tokens,
            max_batch_size,
            queue.clone(),
            batching_task_notifier.clone(),
            generation_health,
        ));

        Self {
            queue,
            batching_task_notifier,
        }
    }
}

impl Scheduler for SchedulerV3 {
    #[instrument(skip_all)]
    fn schedule(
        &self,
        request: ValidGenerateRequest,
        permit: OwnedSemaphorePermit,
    ) -> Result<GenerateStreamResponse, InferError> {
        // MPSC channel to communicate with the background batching task
        let (response_tx, response_rx) = mpsc::unbounded_channel();
        let input_length = request.input_length;

        // Append the request to the queue
        self.queue.append(Entry {
            request,
            response_tx,
            span: Span::current(),
            temp_span: None,
            queue_time: Instant::now(),
            batch_time: None,
            block_allocation: None,
        });

        // Notify the background task that we have a new entry in the queue that needs
        // to be batched
        self.batching_task_notifier.notify_one();

        // Return stream
        Ok((
            permit,
            input_length,
            UnboundedReceiverStream::new(response_rx),
        ))
    }
}

/// Batching logic
/// Will be launched in a background Tokio task
///
/// Batches requests and sends them to the inference server
#[allow(clippy::too_many_arguments)]
pub(crate) async fn batching_task(
    mut client: ShardedClient,
    waiting_served_ratio: f32,
    max_batch_prefill_tokens: u32,
    max_batch_total_tokens: u32,
    max_waiting_tokens: usize,
    max_batch_size: Option<usize>,
    queue: Queue,
    notifier: Arc<Notify>,
    generation_health: Arc<AtomicBool>,
) {
    // Infinite loop
    loop {
        // Wait for a notification from the Infer struct
        notifier.notified().await;

        // Get the next batch from the queue
        // This batch might be smaller than the maximum batch size if there are not enough requests
        // waiting in the queue
        while let Some((mut entries, batch, span)) = queue
            .next_batch(
                None,
                max_batch_size,
                max_batch_prefill_tokens,
                max_batch_total_tokens,
            )
            .await
        {
            let mut cached_batch = prefill(&mut client, batch, &mut entries, &generation_health)
                .instrument(span)
                .await;
            let mut waiting_tokens = 1;

            // We loop until we do not receive any cached batch from the inference server (== until
            // all requests have met their stopping criteria)
            while let Some(batch) = cached_batch {
                // Get current batch info
                let batch_size = batch.size;
                let batch_max_tokens = batch.max_tokens;
                let mut batches = vec![batch];
                metrics::gauge!("tgi_batch_current_size", batch_size as f64);
                metrics::gauge!("tgi_batch_current_max_tokens", batch_max_tokens as f64);

                let min_size = if waiting_tokens >= max_waiting_tokens {
                    // If we didn't onboard any new requests since >= max_waiting_tokens, we try
                    // to add a new batch even though its size might be small
                    None
                } else {
                    // Minimum batch size
                    Some((batch_size as f32 * waiting_served_ratio).floor() as usize)
                };

                let token_budget = max_batch_total_tokens.saturating_sub(batch_max_tokens);
                let max_size = max_batch_size.map(|max_size| max_size - batch_size as usize);

                // Try to get a new batch
                if let Some((mut new_entries, new_batch, span)) = queue
                    .next_batch(min_size, max_size, max_batch_prefill_tokens, token_budget)
                    .await
                {
                    // Tracking metrics
                    if min_size.is_some() {
                        metrics::increment_counter!("tgi_batch_concat", "reason" => "backpressure");
                    } else {
                        metrics::increment_counter!("tgi_batch_concat", "reason" => "wait_exceeded");
                    }

                    entries.iter_mut().for_each(|(_, entry)| {
                        // Create a new span to add the info that this entry is waiting
                        // because a new batch is being computed
                        let entry_waiting_span = info_span!(parent: &entry.span, "waiting");
                        // Add relationships
                        span.follows_from(&entry_waiting_span);
                        entry_waiting_span.follows_from(&span);
                        // Update entry
                        entry.temp_span = Some(entry_waiting_span);
                    });

                    // Generate one token for this new batch to have the attention past in cache
                    let new_cached_batch =
                        prefill(&mut client, new_batch, &mut new_entries, &generation_health)
                            .instrument(span)
                            .await;
                    // Reset waiting counter
                    waiting_tokens = 1;
                    // Extend current batch with the new batch
                    if let Some(new_cached_batch) = new_cached_batch {
                        entries.extend(new_entries);
                        batches.push(new_cached_batch);
                    }
                }

                // Create span for this batch to add context to inference calls
                let next_batch_size = entries.len();
                let next_batch_span =
                    info_span!(parent: None, "batch", batch_size = next_batch_size);
                entries.iter_mut().for_each(|(_, entry)| {
                    // Create a new span to link the batch back to this entry
                    let entry_batch_span = info_span!(parent: &entry.span, "infer");
                    // Add relationships
                    next_batch_span.follows_from(&entry_batch_span);
                    entry_batch_span.follows_from(&next_batch_span);
                    // Update entry
                    entry.temp_span = Some(entry_batch_span);
                });

                cached_batch = decode(&mut client, batches, &mut entries, &generation_health)
                    .instrument(next_batch_span)
                    .await;
                waiting_tokens += 1;
            }
            metrics::gauge!("tgi_batch_current_size", 0.0);
            metrics::gauge!("tgi_batch_current_max_tokens", 0.0);
        }
    }
}

#[instrument(skip_all)]
async fn prefill(
    client: &mut ShardedClient,
    batch: Batch,
    entries: &mut IntMap<u64, Entry>,
    generation_health: &Arc<AtomicBool>,
) -> Option<CachedBatch> {
    let start_time = Instant::now();
    let batch_id = batch.id;
    metrics::increment_counter!("tgi_batch_inference_count", "method" => "prefill");

    match client.prefill(batch).await {
        Ok((generations, next_batch, timings)) => {
            // Update health
            generation_health.store(true, Ordering::SeqCst);

            let start_filtering_time = Instant::now();
            // Send generated tokens and filter stopped entries
            filter_send_generations(generations, entries);

            // Filter next batch and remove requests that were stopped
            let next_batch = filter_batch(client, next_batch, entries).await;

            metrics::histogram!("tgi_batch_forward_duration", timings.forward.as_secs_f64(), "method" => "prefill");
            metrics::histogram!("tgi_batch_decode_duration", timings.decode.as_secs_f64(), "method" => "prefill");
            metrics::histogram!("tgi_batch_filter_duration", start_filtering_time.elapsed().as_secs_f64(), "method" => "prefill");
            metrics::histogram!("tgi_batch_inference_duration", start_time.elapsed().as_secs_f64(), "method" => "prefill");
            metrics::increment_counter!("tgi_batch_inference_success", "method" => "prefill");
            next_batch
        }
        // If we have an error, we discard the whole batch
        Err(err) => {
            // Update health
            generation_health.store(false, Ordering::SeqCst);
            let _ = client.clear_cache(Some(batch_id)).await;
            send_errors(err, entries);
            metrics::increment_counter!("tgi_batch_inference_failure", "method" => "prefill");
            None
        }
    }
}

#[instrument(skip_all)]
async fn decode(
    client: &mut ShardedClient,
    batches: Vec<CachedBatch>,
    entries: &mut IntMap<u64, Entry>,
    generation_health: &Arc<AtomicBool>,
) -> Option<CachedBatch> {
    let start_time = Instant::now();
    let batch_ids: Vec<u64> = batches.iter().map(|b| b.id).collect();
    metrics::increment_counter!("tgi_batch_inference_count", "method" => "decode");

    match client.decode(batches).await {
        Ok((generations, next_batch, timings)) => {
            // Update health
            generation_health.store(true, Ordering::SeqCst);

            let start_filtering_time = Instant::now();
            // Send generated tokens and filter stopped entries
            filter_send_generations(generations, entries);

            // Filter next batch and remove requests that were stopped
            let next_batch = filter_batch(client, next_batch, entries).await;

            if let Some(concat_duration) = timings.concat {
                metrics::histogram!("tgi_batch_concat_duration", concat_duration.as_secs_f64(), "method" => "decode");
            }
            metrics::histogram!("tgi_batch_forward_duration", timings.forward.as_secs_f64(), "method" => "decode");
            metrics::histogram!("tgi_batch_decode_duration", timings.decode.as_secs_f64(), "method" => "decode");
            metrics::histogram!("tgi_batch_filter_duration", start_filtering_time.elapsed().as_secs_f64(), "method" => "decode");
            metrics::histogram!("tgi_batch_inference_duration", start_time.elapsed().as_secs_f64(), "method" => "decode");
            metrics::increment_counter!("tgi_batch_inference_success", "method" => "decode");
            next_batch
        }
        // If we have an error, we discard the whole batch
        Err(err) => {
            generation_health.store(false, Ordering::SeqCst);
            for id in batch_ids {
                let _ = client.clear_cache(Some(id)).await;
            }
            send_errors(err, entries);
            metrics::increment_counter!("tgi_batch_inference_failure", "method" => "decode");
            None
        }
    }
}

/// Filter a `batch` and remove all requests not present in `entries`
#[instrument(skip_all)]
async fn filter_batch(
    client: &mut ShardedClient,
    next_batch: Option<CachedBatch>,
    entries: &IntMap<u64, Entry>,
) -> Option<CachedBatch> {
    let mut batch = next_batch?;

    // No need to filter
    if batch.size as usize == entries.len() {
        return Some(batch);
    }

    let id = batch.id;

    // Retain only requests that are still in entries
    batch.request_ids.retain(|id| entries.contains_key(id));

    if batch.request_ids.is_empty() {
        // All requests have been filtered out
        // Next batch is now empty
        // Clear it from the Python shards cache
        // We unwrap here as we need to panic since we cannot recover if this method fails
        client.clear_cache(Some(id)).await.unwrap();
        None
    } else {
        // Filter Python shard cache
        // We unwrap here as we need to panic since we cannot recover if this method fails
        client.filter_batch(id, batch.request_ids).await.unwrap()
    }
}

/// Send one or multiple `InferStreamResponse` to Infer for all `entries`
/// and filter entries
#[instrument(skip_all)]
fn filter_send_generations(generations: Vec<Generation>, entries: &mut IntMap<u64, Entry>) {
    generations.into_iter().for_each(|generation| {
        let id = generation.request_id;
        // Get entry
        // We can `expect` here as the request id should always be in the entries
        let entry = entries
            .get(&id)
            .expect("ID not found in entries. This is a bug.");

        // Create and enter a span to link this function back to the entry
        let _span = info_span!(parent: entry.temp_span.as_ref().expect("batch_span is None. This is a bug."), "send_generation", generation = ?generation).entered();
        // Send generation responses back to the infer task
        // If the receive an error from the Flume channel, it means that the client dropped the
        // request and we need to stop generating hence why we unwrap_or(true)
        let stopped = send_responses(generation, entry).map_err(|err| {
            tracing::error!("Entry response channel error.");
            metrics::increment_counter!("tgi_request_failure", "err" => "dropped");
            err
        }).unwrap_or(true);
        if stopped {
            entries.remove(&id).expect("ID not found in entries. This is a bug.");
        }
    });
}

/// Send responses through the `entry` response channel
fn send_responses(
    generation: Generation,
    entry: &Entry,
) -> Result<bool, Box<SendError<Result<InferStreamResponse, InferError>>>> {
    // Return directly if the channel is disconnected
    if entry.response_tx.is_closed() {
        metrics::increment_counter!("tgi_request_failure", "err" => "dropped");
        return Ok(true);
    }

    let mut stopped = false;

    if let Some(prefill_tokens) = generation.prefill_tokens {
        // Create Token objects
        // We do that here instead of in the Python code as Rust for loops are faster
        let prefill_tokens = prefill_tokens
            .ids
            .into_iter()
            .zip(prefill_tokens.logprobs)
            .zip(prefill_tokens.texts)
            .map(|((id, logprob), text)| PrefillToken { id, text, logprob })
            .collect();

        // Send message
        entry
            .response_tx
            .send(Ok(InferStreamResponse::Prefill(prefill_tokens)))?;
    }

    // Create last Token
    let tokens_ = generation.tokens.expect("Non empty tokens in generation");
    let n = tokens_.ids.len();
    metrics::histogram!("tgi_request_skipped_tokens", (n - 1) as f64);
    let mut iterator = tokens_
        .ids
        .into_iter()
        .zip(tokens_.logprobs)
        .zip(tokens_.texts)
        .zip(tokens_.is_special)
        .enumerate()
        .peekable();
    while let Some((i, (((id, logprob), text), special))) = iterator.next() {
        let token = Token {
            id,
            text,
            logprob,
            special,
        };
        let top_tokens = if let Some(top_tokens_) = generation.top_tokens.get(i) {
            top_tokens_
                .ids
                .iter()
                .zip(top_tokens_.logprobs.iter())
                .zip(top_tokens_.texts.iter())
                .zip(top_tokens_.is_special.iter())
                .map(|(((&id, &logprob), text), &special)| Token {
                    id,
                    text: text.to_string(),
                    logprob,
                    special,
                })
                .collect()
        } else {
            vec![]
        };
        match (&generation.generated_text, iterator.peek()) {
            (Some(generated_text), None) => {
                // Generation has ended
                stopped = true;
                // Send message
                entry.response_tx.send(Ok(InferStreamResponse::End {
                    token,
                    top_tokens,
                    generated_text: GeneratedText::from(generated_text.clone()),
                    queued: entry.queue_time,
                    start: entry.batch_time.unwrap(),
                }))?;
            }
            _ => {
                // Send message
                entry
                    .response_tx
                    .send(Ok(InferStreamResponse::Intermediate { token, top_tokens }))?;
            }
        }
    }

    Ok(stopped)
}

/// Send errors to Infer for all `entries`
#[instrument(skip_all)]
fn send_errors(error: ClientError, entries: &mut IntMap<u64, Entry>) {
    entries.drain().for_each(|(_, entry)| {
        // Create and enter a span to link this function back to the entry
        let _send_error_span = info_span!(parent: entry.temp_span.as_ref().expect("batch_span is None. This is a bug."), "send_error").entered();
        let err = InferError::GenerationError(error.to_string());
        metrics::increment_counter!("tgi_request_failure", "err" => "generation");
        tracing::error!("{err}");

        // unwrap_or is valid here as we don't care if the receiver is gone.
        entry
            .response_tx
            .send(Err(err))
            .unwrap_or(());
    });
}

impl From<text_generation_client::v3::GeneratedText> for GeneratedText {
    fn from(value: text_generation_client::v3::GeneratedText) -> Self {
        let v3_finish_reason =
            text_generation_client::v3::FinishReason::try_from(value.finish_reason).unwrap();
        let finish_reason = match v3_finish_reason {
            text_generation_client::v3::FinishReason::Length => FinishReason::Length,
            text_generation_client::v3::FinishReason::EosToken => FinishReason::EndOfSequenceToken,
            text_generation_client::v3::FinishReason::StopSequence => FinishReason::StopSequence,
        };

        Self {
            text: value.text,
            generated_tokens: value.generated_tokens,
            finish_reason,
            seed: value.seed,
        }
    }
}

// tests
#[cfg(test)]
mod tests {
    use crate::infer::raise_exception;
    use crate::{ChatTemplateInputs, TextMessage};
    use minijinja::Environment;

    #[test]
    fn test_chat_template() {
        let env = Environment::new();

        let source = r#"
        {% for message in messages %}
            {% if message['role'] == 'system' %}
                {% if message['content']%}
                    {{'### System:\n' + message['content']+'\n\n'}}
                {% endif %}
            {% elif message['role'] == 'user' %}
                {{'### User:\n' + message['content']+'\n\n'}}
            {% elif message['role'] == 'assistant' %}
                {{'### Assistant:\n'  + message['content']}}
            {% endif %}
            {% if loop.last and add_generation_prompt %}
                {{ '### Assistant:\n' }}
            {% endif %}
        {% endfor %}"#;

        // trim all the whitespace
        let source = source
            .lines()
            .map(|line| line.trim())
            .collect::<Vec<&str>>()
            .join("");

        let tmpl = env.template_from_str(&source);

        let chat_template_inputs = ChatTemplateInputs {
            messages: vec![
                TextMessage {
                    role: "user".to_string(),
                    content: "Hi!".to_string(),
                },
                TextMessage {
                    role: "assistant".to_string(),
                    content: "Hello how can I help?".to_string(),
                },
                TextMessage {
                    role: "user".to_string(),
                    content: "What is Deep Learning?".to_string(),
                },
                TextMessage {
                    role: "assistant".to_string(),
                    content: "magic!".to_string(),
                },
            ],
            bos_token: Some("[BOS]"),
            eos_token: Some("[EOS]"),
            add_generation_prompt: true,
            ..Default::default()
        };

        let result = tmpl.unwrap().render(chat_template_inputs).unwrap();

        assert_eq!(
            result,
            "### User:\nHi!\n\n### Assistant:\nHello how can I help?### User:\nWhat is Deep Learning?\n\n### Assistant:\nmagic!### Assistant:\n"
        );
    }

    #[test]
    fn test_chat_template_invalid_with_raise() {
        let mut env = Environment::new();
        env.add_function("raise_exception", raise_exception);

        let source = r#"
        {{ bos_token }}
        {% for message in messages %}
        {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
        {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
        {% endif %}
        {% if message['role'] == 'user' %}
        {{ '[INST] ' + message['content'] + ' [/INST]' }}
        {% elif message['role'] == 'assistant' %}
        {{ message['content'] + eos_token}}
        {% else %}
        {{ raise_exception('Only user and assistant roles are supported!') }}
        {% endif %}
        {% endfor %}"#;

        // trim all the whitespace
        let source = source
            .lines()
            .map(|line| line.trim())
            .collect::<Vec<&str>>()
            .join("");

        let tmpl = env.template_from_str(&source);

        let chat_template_inputs = ChatTemplateInputs {
            messages: vec![
                TextMessage {
                    role: "user".to_string(),
                    content: "Hi!".to_string(),
                },
                TextMessage {
                    role: "user".to_string(),
                    content: "Hi again!".to_string(),
                },
                TextMessage {
                    role: "assistant".to_string(),
                    content: "Hello how can I help?".to_string(),
                },
                TextMessage {
                    role: "user".to_string(),
                    content: "What is Deep Learning?".to_string(),
                },
                TextMessage {
                    role: "assistant".to_string(),
                    content: "magic!".to_string(),
                },
            ],
            bos_token: Some("[BOS]"),
            eos_token: Some("[EOS]"),
            add_generation_prompt: true,
            ..Default::default()
        };

        let result = tmpl.unwrap().render(chat_template_inputs); //.err().unwrap();

        match result {
            Ok(_) => panic!("Should have failed"),
            Err(e) => {
                assert_eq!(
                    e.detail().unwrap(),
                    "Conversation roles must alternate user/assistant/user/assistant/..."
                );
            }
        }
    }

    #[test]
    fn test_chat_template_valid_with_raise() {
        let mut env = Environment::new();
        env.add_function("raise_exception", raise_exception);

        let source = r#"
        {{ bos_token }}
        {% for message in messages %}
        {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
        {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
        {% endif %}
        {% if message['role'] == 'user' %}
        {{ '[INST] ' + message['content'] + ' [/INST]' }}
        {% elif message['role'] == 'assistant' %}
        {{ message['content'] + eos_token}}
        {% else %}
        {{ raise_exception('Only user and assistant roles are supported!') }}
        {% endif %}
        {% endfor %}"#;

        // trim all the whitespace
        let source = source
            .lines()
            .map(|line| line.trim())
            .collect::<Vec<&str>>()
            .join("");

        let tmpl = env.template_from_str(&source);

        let chat_template_inputs = ChatTemplateInputs {
            messages: vec![
                TextMessage {
                    role: "user".to_string(),
                    content: "Hi!".to_string(),
                },
                TextMessage {
                    role: "assistant".to_string(),
                    content: "Hello how can I help?".to_string(),
                },
                TextMessage {
                    role: "user".to_string(),
                    content: "What is Deep Learning?".to_string(),
                },
                TextMessage {
                    role: "assistant".to_string(),
                    content: "magic!".to_string(),
                },
            ],
            bos_token: Some("[BOS]"),
            eos_token: Some("[EOS]"),
            add_generation_prompt: true,
            ..Default::default()
        };

        let result = tmpl.unwrap().render(chat_template_inputs).unwrap();
        assert_eq!(result, "[BOS][INST] Hi! [/INST]Hello how can I help?[EOS][INST] What is Deep Learning? [/INST]magic![EOS]");
    }

    #[test]
    fn test_chat_template_valid_with_add_generation_prompt() {
        let mut env = Environment::new();
        env.add_function("raise_exception", raise_exception);

        let source = r#"
        {% for message in messages %}
        {{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}
        {% endfor %}
        {% if add_generation_prompt %}
            {{ '<|im_start|>assistant\n' }}
        {% endif %}"#;

        // trim all the whitespace
        let source = source
            .lines()
            .map(|line| line.trim())
            .collect::<Vec<&str>>()
            .join("");

        let tmpl = env.template_from_str(&source);

        let chat_template_inputs = ChatTemplateInputs {
            messages: vec![
                TextMessage {
                    role: "user".to_string(),
                    content: "Hi!".to_string(),
                },
                TextMessage {
                    role: "assistant".to_string(),
                    content: "Hello how can I help?".to_string(),
                },
                TextMessage {
                    role: "user".to_string(),
                    content: "What is Deep Learning?".to_string(),
                },
                TextMessage {
                    role: "assistant".to_string(),
                    content: "magic!".to_string(),
                },
            ],
            bos_token: Some("[BOS]"),
            eos_token: Some("[EOS]"),
            add_generation_prompt: true,
            ..Default::default()
        };

        let result = tmpl.unwrap().render(chat_template_inputs).unwrap();
        assert_eq!(result, "<|im_start|>user\nHi!<|im_end|>\n<|im_start|>assistant\nHello how can I help?<|im_end|>\n<|im_start|>user\nWhat is Deep Learning?<|im_end|>\n<|im_start|>assistant\nmagic!<|im_end|>\n<|im_start|>assistant\n");
    }

    struct ChatTemplateTestItem {
        name: &'static str,
        chat_template: &'static str,
        input: ChatTemplateInputs<'static>,
        target: &'static str,
    }

    #[test]
    fn test_many_chat_templates() {
        let example_chat = vec![
            TextMessage {
                role: "user".to_string(),
                content: "Hello, how are you?".to_string(),
            },
            TextMessage {
                role: "assistant".to_string(),
                content: "I'm doing great. How can I help you today?".to_string(),
            },
            TextMessage {
                role: "user".to_string(),
                content: "I'd like to show off how chat templating works!".to_string(),
            },
        ];

        let example_chat_with_system = [TextMessage {
            role: "system".to_string(),
            content: "You are a friendly chatbot who always responds in the style of a pirate"
                .to_string(),
        }]
        .iter()
        .chain(&example_chat)
        .cloned()
        .collect::<Vec<_>>();

        let test_default_templates = vec![
            ChatTemplateTestItem {
                name: "_base",
                chat_template: "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}",
                input: ChatTemplateInputs {
                    messages: example_chat.clone(),
                    add_generation_prompt: false,
                    bos_token: Some(""),
                    eos_token: Some(""),
                    ..Default::default()
                },
                target: "<|im_start|>user\nHello, how are you?<|im_end|>\n<|im_start|>assistant\nI'm doing great. How can I help you today?<|im_end|>\n<|im_start|>user\nI'd like to show off how chat templating works!<|im_end|>\n",
            },
            ChatTemplateTestItem {
                name: "blenderbot",
                chat_template: "{% for message in messages %}{% if message['role'] == 'user' %}{{ ' ' }}{% endif %}{{ message['content'] }}{% if not loop.last %}{{ '  ' }}{% endif %}{% endfor %}{{ eos_token }}",
                input: ChatTemplateInputs {
                    messages: example_chat.clone(),
                    add_generation_prompt: false,
                    bos_token: Some(""),
                    eos_token: Some("</s>"),
                    ..Default::default()
                },
                target: " Hello, how are you?  I'm doing great. How can I help you today?   I'd like to show off how chat templating works!</s>",
            },
            ChatTemplateTestItem {
                name: "blenderbot_small",
                chat_template: "{% for message in messages %}{% if message['role'] == 'user' %}{{ ' ' }}{% endif %}{{ message['content'] }}{% if not loop.last %}{{ '  ' }}{% endif %}{% endfor %}{{ eos_token }}",
                input: ChatTemplateInputs {
                    messages: example_chat.clone(),
                    add_generation_prompt: false,
                    bos_token: Some(""),
                    eos_token: Some("</s>"),
                    ..Default::default()
                },
                target: " Hello, how are you?  I'm doing great. How can I help you today?   I'd like to show off how chat templating works!</s>",
            },
            ChatTemplateTestItem {
                name: "bloom",
                chat_template: "{% for message in messages %}{{ message.content }}{{ eos_token }}{% endfor %}",
                input: ChatTemplateInputs {
                    messages: example_chat.clone(),
                    add_generation_prompt: false,
                    bos_token: Some(""),
                    eos_token: Some("</s>"),
                    ..Default::default()
                },
                target: "Hello, how are you?</s>I'm doing great. How can I help you today?</s>I'd like to show off how chat templating works!</s>",
            },
            ChatTemplateTestItem {
                name: "gpt_neox",
                chat_template: "{% for message in messages %}{{ message.content }}{{ eos_token }}{% endfor %}",
                input: ChatTemplateInputs {
                    messages: example_chat.clone(),
                    add_generation_prompt: false,
                    bos_token: Some(""),
                    eos_token: Some("<|endoftext|>"),
                    ..Default::default()
                },
                target: "Hello, how are you?<|endoftext|>I'm doing great. How can I help you today?<|endoftext|>I'd like to show off how chat templating works!<|endoftext|>",
            },
            ChatTemplateTestItem {
                name: "gpt2",
                chat_template: "{% for message in messages %}{{ message.content }}{{ eos_token }}{% endfor %}",
                input: ChatTemplateInputs {
                    messages: example_chat.clone(),
                    add_generation_prompt: false,
                    bos_token: Some(""),
                    eos_token: Some("<|endoftext|>"),
                    ..Default::default()
                },
                target: "Hello, how are you?<|endoftext|>I'm doing great. How can I help you today?<|endoftext|>I'd like to show off how chat templating works!<|endoftext|>",
            },
            ChatTemplateTestItem {
                name: "llama",
                // NOTE: the `.strip()` has been replaced with `| trim` in the following template
                chat_template: "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% elif USE_DEFAULT_PROMPT == true and not '<<SYS>>' in messages[0]['content'] %}{% set loop_messages = messages %}{% set system_message = 'DEFAULT_SYSTEM_MESSAGE' %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token +'[INST] ' + content | trim + ' [/INST]' }}{% elif message['role'] == 'system' %}{{ '<<SYS>>\\n' + content | trim + '\\n<</SYS>>\\n\\n' }}{% elif message['role'] == 'assistant' %}{{ ' ' + content | trim + ' ' + eos_token }}{% endif %}{% endfor %}",
                input: ChatTemplateInputs {
                    messages: example_chat_with_system.clone(),
                    add_generation_prompt: true,
                    bos_token: Some("<s>"),
                    eos_token: Some("</s>"),
                    ..Default::default()
                },
                target: "<s>[INST] <<SYS>>\nYou are a friendly chatbot who always responds in the style of a pirate\n<</SYS>>\n\nHello, how are you? [/INST] I'm doing great. How can I help you today? </s><s>[INST] I'd like to show off how chat templating works! [/INST]",
            },
            ChatTemplateTestItem {
                name: "whisper",
                chat_template: "{% for message in messages %}{{ message.content }}{{ eos_token }}{% endfor %}",
                input: ChatTemplateInputs {
                    messages: example_chat.clone(),
                    add_generation_prompt: true,
                    bos_token: Some(""),
                    eos_token: Some("<|endoftext|>"),
                    ..Default::default()
                },
                target: "Hello, how are you?<|endoftext|>I'm doing great. How can I help you today?<|endoftext|>I'd like to show off how chat templating works!<|endoftext|>",
            },
        ];

        #[allow(unused_variables)] // name is unused
        for ChatTemplateTestItem {
            name,
            chat_template,
            input,
            target,
        } in test_default_templates
        {
            let mut env = Environment::new();
            env.add_function("raise_exception", raise_exception);
            let tmpl = env.template_from_str(chat_template);
            let result = tmpl.unwrap().render(input).unwrap();
            assert_eq!(result, target);
        }

        let test_custom_templates = vec![
            ChatTemplateTestItem {
                name: "HuggingFaceH4/zephyr-7b-beta (add_generation_prompt=false)",
                chat_template: "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}",
                input: ChatTemplateInputs {
                    messages: example_chat_with_system.clone(),
                    add_generation_prompt: false,
                    bos_token: Some(""),
                    eos_token: Some("</s>"),
                    ..Default::default()
                },
                target: "<|system|>\nYou are a friendly chatbot who always responds in the style of a pirate</s><|user|>\nHello, how are you?</s><|assistant|>\nI'm doing great. How can I help you today?</s><|user|>\nI'd like to show off how chat templating works!</s>",
            },
            ChatTemplateTestItem {
                name: "HuggingFaceH4/zephyr-7b-beta (add_generation_prompt=true)",
                chat_template: "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}",
                input: ChatTemplateInputs {
                    messages: vec![
                        TextMessage {
                            role: "system".to_string(),
                            content: "You are a friendly chatbot who always responds in the style of a pirate".to_string(),
                        },
                        TextMessage {
                            role: "user".to_string(),
                            content: "How many helicopters can a human eat in one sitting?".to_string(),
                        },
                    ],
                    add_generation_prompt: true,
                    bos_token: Some(""),
                    eos_token: Some("</s>"),
                    ..Default::default()
                },
                target: "<|system|>\nYou are a friendly chatbot who always responds in the style of a pirate</s><|user|>\nHow many helicopters can a human eat in one sitting?</s><|assistant|>",
            },
            ChatTemplateTestItem {
                name: "HuggingFaceH4/zephyr-7b-gemma-v0.1",
                chat_template: "{% if messages[0]['role'] == 'user' or messages[0]['role'] == 'system' %}{{ bos_token }}{% endif %}{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% elif messages[-1]['role'] == 'assistant' %}{{ eos_token }}{% endif %}",
                input: ChatTemplateInputs {
                    messages: example_chat.clone(),
                    add_generation_prompt: false,
                    bos_token: Some("<bos>"),
                    eos_token: Some("<eos>"),
                    ..Default::default()
                },
                target: "<bos><|im_start|>user\nHello, how are you?<|im_end|>\n<|im_start|>assistant\nI'm doing great. How can I help you today?<|im_end|>\n<|im_start|>user\nI'd like to show off how chat templating works!<|im_end|>\n",
            },
            ChatTemplateTestItem {
                name: "mistralai/Mistral-7B-Instruct-v0.1",
                chat_template: "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token + ' ' }}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}",
                input: ChatTemplateInputs {
                    messages: example_chat.clone(),
                    add_generation_prompt: false,
                    bos_token: Some("<s>"),
                    eos_token: Some("</s>"),
                    ..Default::default()
                },
                target: "<s>[INST] Hello, how are you? [/INST]I'm doing great. How can I help you today?</s> [INST] I'd like to show off how chat templating works! [/INST]",
            },
            ChatTemplateTestItem {
                name: "mistralai/Mixtral-8x7B-Instruct-v0.1",
                chat_template: "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}",
                input: ChatTemplateInputs {
                    messages: example_chat.clone(),
                    add_generation_prompt: false,
                    bos_token: Some("<s>"),
                    eos_token: Some("</s>"),
                    ..Default::default()
                },
                target: "<s>[INST] Hello, how are you? [/INST]I'm doing great. How can I help you today?</s>[INST] I'd like to show off how chat templating works! [/INST]",
            },
            ChatTemplateTestItem {
                name: "cognitivecomputations/dolphin-2.5-mixtral-8x7b",
                chat_template: "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}",
                input: ChatTemplateInputs {
                    messages: example_chat.clone(),
                    add_generation_prompt: false,
                    bos_token: Some("<s>"),
                    eos_token: Some("</s>"),
                    ..Default::default()
                },
                target: "<|im_start|>user\nHello, how are you?<|im_end|>\n<|im_start|>assistant\nI'm doing great. How can I help you today?<|im_end|>\n<|im_start|>user\nI'd like to show off how chat templating works!<|im_end|>\n",
            },
            ChatTemplateTestItem {
                name: "openchat/openchat-3.5-0106",
                // `.title()` has been replaced with `| upper` in the following template
                chat_template: "{{ bos_token }}{% for message in messages %}{{ 'GPT4 Correct ' + (message['role'] | title) + ': ' + message['content'] + '<|end_of_turn|>'}}{% endfor %}{% if add_generation_prompt %}{{ 'GPT4 Correct Assistant:' }}{% endif %}",
                input: ChatTemplateInputs {
                    messages: example_chat.clone(),
                    add_generation_prompt: false,
                    bos_token: Some("<s>"),
                    eos_token: Some("</s>"),
                    ..Default::default()
                },
                target: "<s>GPT4 Correct User: Hello, how are you?<|end_of_turn|>GPT4 Correct Assistant: I'm doing great. How can I help you today?<|end_of_turn|>GPT4 Correct User: I'd like to show off how chat templating works!<|end_of_turn|>",
            },
            ChatTemplateTestItem {
                name: "upstage/SOLAR-10.7B-Instruct-v1.0",
                chat_template: "{% for message in messages %}{{ message.content }}{{ eos_token }}{% endfor %}",
                input: ChatTemplateInputs {
                    messages: example_chat.clone(),
                    add_generation_prompt: false,
                    bos_token: Some("<s>"),
                    eos_token: Some("</s>"),
                    ..Default::default()
                },
                target: "Hello, how are you?</s>I'm doing great. How can I help you today?</s>I'd like to show off how chat templating works!</s>",
            },
            ChatTemplateTestItem {
                name: "codellama/CodeLlama-70b-Instruct-hf",
                // NOTE: `.strip()` has been replaced with `| trim` in the following template
                chat_template: "{% if messages[0]['role'] == 'system' %}{% set user_index = 1 %}{% else %}{% set user_index = 0 %}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != ((loop.index0 + user_index) % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 %}{{ '<s>' }}{% endif %}{% set content = 'Source: ' + message['role'] + '\\n\\n ' + message['content'] | trim %}{{ content + ' <step> ' }}{% endfor %}{{'Source: assistant\\nDestination: user\\n\\n '}}",
                input: ChatTemplateInputs {
                    messages: example_chat.clone(),
                    add_generation_prompt: false,
                    bos_token: Some("<s>"),
                    eos_token: Some("</s>"),
                    ..Default::default()
                },
                target: "<s>Source: user\n\n Hello, how are you? <step> Source: assistant\n\n I'm doing great. How can I help you today? <step> Source: user\n\n I'd like to show off how chat templating works! <step> Source: assistant\nDestination: user\n\n ",
            },
            ChatTemplateTestItem {
                name: "Deci/DeciLM-7B-instruct",
                chat_template: "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '### User:\\n' + message['content'] }}\n{% elif message['role'] == 'system' %}\n{{ '### System:\\n' + message['content'] }}\n{% elif message['role'] == 'assistant' %}\n{{ '### Assistant:\\n'  + message['content'] }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '### Assistant:' }}\n{% endif %}\n{% endfor %}",
                input: ChatTemplateInputs {
                    messages: example_chat.clone(),
                    add_generation_prompt: false,
                    bos_token: Some("<s>"),
                    eos_token: Some("</s>"),
                    ..Default::default()
                },
                target: "### User:\nHello, how are you?### Assistant:\nI'm doing great. How can I help you today?### User:\nI'd like to show off how chat templating works!",
            },
            ChatTemplateTestItem {
                name: "Qwen/Qwen1.5-72B-Chat",
                chat_template: "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\\nYou are a helpful assistant<|im_end|>\\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\\n' + message['content']}}{% if (loop.last and add_generation_prompt) or not loop.last %}{{ '<|im_end|>' + '\\n'}}{% endif %}{% endfor %}{% if add_generation_prompt and messages[-1]['role'] != 'assistant' %}{{ '<|im_start|>assistant\\n' }}{% endif %}",
                input: ChatTemplateInputs {
                    messages: example_chat.clone(),
                    add_generation_prompt: false,
                    bos_token: Some("<s>"),
                    eos_token: Some("</s>"),
                    ..Default::default()
                },
                target: "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\nHello, how are you?<|im_end|>\n<|im_start|>assistant\nI'm doing great. How can I help you today?<|im_end|>\n<|im_start|>user\nI'd like to show off how chat templating works!",
            },
            ChatTemplateTestItem {
                name: "deepseek-ai/deepseek-llm-7b-chat",
                chat_template: "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}{{ 'User: ' + message['content'] + '\\n\\n' }}{% elif message['role'] == 'assistant' %}{{ 'Assistant: ' + message['content'] + eos_token }}{% elif message['role'] == 'system' %}{{ message['content'] + '\\n\\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}",
                input: ChatTemplateInputs {
                    messages: example_chat.clone(),
                    add_generation_prompt: false,
                    bos_token: Some("<｜begin▁of▁sentence｜>"),
                    eos_token: Some("<｜end▁of▁sentence｜>"),
                    ..Default::default()
                },
                target: "<｜begin▁of▁sentence｜>User: Hello, how are you?\n\nAssistant: I'm doing great. How can I help you today?<｜end▁of▁sentence｜>User: I'd like to show off how chat templating works!\n\n",
            },
            ChatTemplateTestItem {
                name: "h2oai/h2o-danube-1.8b-chat",
                chat_template: "{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|prompt|>' + message['content'] + eos_token }}{% elif message['role'] == 'system' %}{{ '<|system|>' + message['content'] + eos_token }}{% elif message['role'] == 'assistant' %}{{ '<|answer|>'  + message['content'] + eos_token }}{% endif %}{% if loop.last and add_generation_prompt %}{{ '<|answer|>' }}{% endif %}{% endfor %}",
                input: ChatTemplateInputs {
                    messages: example_chat.clone(),
                    add_generation_prompt: false,
                    bos_token: Some("<s>"),
                    eos_token: Some("</s>"),
                    ..Default::default()
                },
                target: "<|prompt|>Hello, how are you?</s><|answer|>I'm doing great. How can I help you today?</s><|prompt|>I'd like to show off how chat templating works!</s>",
            },
            ChatTemplateTestItem {
                name: "internlm/internlm2-chat-7b",
                chat_template: "{% if messages[0]['role'] == 'user' or messages[0]['role'] == 'system' %}{{ bos_token }}{% endif %}{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% elif messages[-1]['role'] == 'assistant' %}{{ eos_token }}{% endif %}",
                input: ChatTemplateInputs {
                    messages: example_chat.clone(),
                    add_generation_prompt: false,
                    bos_token: Some("<s>"),
                    eos_token: Some("</s>"),
                    ..Default::default()
                },
                target: "<s><|im_start|>user\nHello, how are you?<|im_end|>\n<|im_start|>assistant\nI'm doing great. How can I help you today?<|im_end|>\n<|im_start|>user\nI'd like to show off how chat templating works!<|im_end|>\n",
            },
            ChatTemplateTestItem {
                name: "TheBloke/deepseek-coder-33B-instruct-AWQ",
                chat_template: "{%- set found_item = false -%}\n{%- for message in messages -%}\n    {%- if message['role'] == 'system' -%}\n        {%- set found_item = true -%}\n    {%- endif -%}\n{%- endfor -%}\n{%- if not found_item -%}\n{{'You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.\\n'}}\n{%- endif %}\n{%- for message in messages %}\n    {%- if message['role'] == 'system' %}\n{{ message['content'] }}\n    {%- else %}\n        {%- if message['role'] == 'user' %}\n{{'### Instruction:\\n' + message['content'] + '\\n'}}\n        {%- else %}\n{{'### Response:\\n' + message['content'] + '\\n<|EOT|>\\n'}}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{{'### Response:\\n'}}\n",
                input: ChatTemplateInputs {
                    messages: example_chat.clone(),
                    add_generation_prompt: false,
                    bos_token: Some("<｜begin▁of▁sentence｜>"),
                    eos_token: Some("<|EOT|>"),
                    ..Default::default()
                },
                target: "You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.\n### Instruction:\nHello, how are you?\n### Response:\nI'm doing great. How can I help you today?\n<|EOT|>\n### Instruction:\nI'd like to show off how chat templating works!\n### Response:\n",
            },
            ChatTemplateTestItem {
                name: "ericzzz/falcon-rw-1b-chat",
                // `.strip()` has been replaced with `| trim` in the following template
                chat_template: "{% for message in messages %}{% if loop.index > 1 and loop.previtem['role'] != 'assistant' %}{{ ' ' }}{% endif %}{% if message['role'] == 'system' %}{{ '[SYS] ' + message['content'] | trim }}{% elif message['role'] == 'user' %}{{ '[INST] ' + message['content'] | trim }}{% elif message['role'] == 'assistant' %}{{ '[RESP] '  + message['content'] + eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ ' [RESP] ' }}{% endif %}",
                input: ChatTemplateInputs {
                    messages: example_chat.clone(),
                    add_generation_prompt: false,
                    bos_token: Some("<|endoftext|>"),
                    eos_token: Some("<|endoftext|>"),
                    ..Default::default()
                },
                target: "[INST] Hello, how are you? [RESP] I'm doing great. How can I help you today?<|endoftext|>[INST] I'd like to show off how chat templating works!",
            },
            ChatTemplateTestItem {
                name: "abacusai/Smaug-34B-v0.1",
                chat_template: "{%- for idx in range(0, messages|length) -%}\n{%- if messages[idx]['role'] == 'user' -%}\n{%- if idx > 1 -%}\n{{- bos_token + '[INST] ' + messages[idx]['content'] + ' [/INST]' -}}\n{%- else -%}\n{{- messages[idx]['content'] + ' [/INST]' -}}\n{%- endif -%}\n{% elif messages[idx]['role'] == 'system' %}\n{{- '[INST] <<SYS>>\\n' + messages[idx]['content'] + '\\n<</SYS>>\\n\\n' -}}\n{%- elif messages[idx]['role'] == 'assistant' -%}\n{{- ' '  + messages[idx]['content'] + ' ' + eos_token -}}\n{% endif %}\n{% endfor %}",
                input: ChatTemplateInputs {
                    messages: example_chat.clone(),
                    add_generation_prompt: false,
                    bos_token: Some("<s>"),
                    eos_token: Some("</s>"),
                    ..Default::default()
                },
                target: "Hello, how are you? [/INST] I'm doing great. How can I help you today? </s><s>[INST] I'd like to show off how chat templating works! [/INST]",
            },
            ChatTemplateTestItem {
                name: "maywell/Synatra-Mixtral-8x7B",
                chat_template: "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n{% for message in messages %}{% if message['role'] == 'user' %}### Instruction:\n{{ message['content']|trim -}}{% if not loop.last %}{% endif %}\n{% elif message['role'] == 'assistant' %}### Response:\n{{ message['content']|trim -}}{% if not loop.last %}{% endif %}\n{% elif message['role'] == 'system' %}{{ message['content']|trim -}}{% if not loop.last %}{% endif %}\n{% endif %}\n{% endfor %}\n{% if add_generation_prompt and messages[-1]['role'] != 'assistant' %}\n### Response:\n{% endif %}",
                input: ChatTemplateInputs {
                    messages: example_chat.clone(),
                    add_generation_prompt: false,
                    bos_token: Some("<s>"),
                    eos_token: Some("</s>"),
                    ..Default::default()
                },
                target: "Below is an instruction that describes a task. Write a response that appropriately completes the request.### Instruction:Hello, how are you?### Response:I'm doing great. How can I help you today?### Instruction:I'd like to show off how chat templating works!",
            },
            ChatTemplateTestItem {
                name: "deepseek-ai/deepseek-coder-33b-instruct",
                chat_template: "{% if not add_generation_prompt is defined %}\n{% set add_generation_prompt = false %}\n{% endif %}\n{%- set ns = namespace(found=false) -%}\n{%- for message in messages -%}\n    {%- if message['role'] == 'system' -%}\n        {%- set ns.found = true -%}\n    {%- endif -%}\n{%- endfor -%}\n{{bos_token}}{%- if not ns.found -%}\n{{'You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer\\n'}}\n{%- endif %}\n{%- for message in messages %}\n    {%- if message['role'] == 'system' %}\n{{ message['content'] }}\n    {%- else %}\n        {%- if message['role'] == 'user' %}\n{{'### Instruction:\\n' + message['content'] + '\\n'}}\n        {%- else %}\n{{'### Response:\\n' + message['content'] + '\\n<|EOT|>\\n'}}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{% if add_generation_prompt %}\n{{'### Response:'}}\n{% endif %}",
                input: ChatTemplateInputs {
                    messages: example_chat.clone(),
                    add_generation_prompt: false,
                    bos_token: Some("<｜begin▁of▁sentence｜>"),
                    eos_token: Some("</EOT>"),
                    ..Default::default()
                },
                target: "<｜begin▁of▁sentence｜>You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer\n### Instruction:\nHello, how are you?\n### Response:\nI'm doing great. How can I help you today?\n<|EOT|>\n### Instruction:\nI'd like to show off how chat templating works!\n",
            },
            // NOT INCLUDED
            // - meetkai/functionary-medium-v3.2
            // - fireworks-ai/firefunction-v1
            // https://github
            ChatTemplateTestItem {
                name: "maywell/PiVoT-MoE",
                chat_template: "{{ (messages|selectattr('role', 'equalto', 'system')|list|last).content|trim if (messages|selectattr('role', 'equalto', 'system')|list) else '' }}{% for message in messages %}{% if message['role'] == 'system' %}{{ message['content']|trim }}{% elif message['role'] == 'user' %}### Instruction: {{ message['content']|trim }}{% elif message['role'] == 'assistant' %}### Response: {{ message['content']|trim }}{% elif message['role'] == 'user_context' %}### Input: {{ message['content']|trim }}{% endif %}{% if not loop.last %}\n{% endif %}{% endfor %}{% if add_generation_prompt and messages[-1]['role'] != 'assistant' %}### Response:{% endif %}",
                input: ChatTemplateInputs {
                    messages: example_chat_with_system.clone(),
                    add_generation_prompt: false,
                    bos_token: Some("<s>"),
                    eos_token: Some("</s>"),
                    ..Default::default()
                },
                target: "You are a friendly chatbot who always responds in the style of a pirateYou are a friendly chatbot who always responds in the style of a pirate### Instruction: Hello, how are you?### Response: I'm doing great. How can I help you today?### Instruction: I'd like to show off how chat templating works!",
            },
        ];

        #[allow(unused_variables)] // name is unused
        for ChatTemplateTestItem {
            name,
            chat_template,
            input,
            target,
        } in test_custom_templates
        {
            let mut env = Environment::new();
            env.add_function("raise_exception", raise_exception);
            // trim all the whitespace
            let chat_template = chat_template
                .lines()
                .map(|line| line.trim())
                .collect::<Vec<&str>>()
                .join("");

            let tmpl = env.template_from_str(&chat_template);
            let result = tmpl.unwrap().render(input).unwrap();
            assert_eq!(result, target);
        }
    }
}
