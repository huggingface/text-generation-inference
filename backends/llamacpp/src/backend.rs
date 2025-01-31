mod bindings {
    #![allow(non_upper_case_globals)]
    #![allow(non_camel_case_types)]
    #![allow(non_snake_case)]
    #![allow(dead_code)]
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}
use async_trait::async_trait;
use std::ffi::CString;
use std::sync::{mpsc, Once};
use text_generation_router::infer::{Backend, GeneratedText, InferError, InferStreamResponse};
use text_generation_router::validation::{ValidGenerateRequest};
use text_generation_router::{FinishReason, Token};
use thiserror::Error;
use tokenizers::Tokenizer;
use tokio::sync::mpsc::{unbounded_channel, UnboundedSender};
use tokio::sync::{watch, oneshot};
use tokio::task::{spawn, spawn_blocking};
use tokio::time::{Duration, Instant, timeout};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{debug, info, warn, error, trace};
use tracing::{instrument};
use std::str::FromStr;

#[derive(Debug, Clone, Copy)]
pub enum LlamacppSplitMode {
    GPU(usize),
    Layer,
    Row,
}

impl FromStr for LlamacppSplitMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "layer" => Ok(LlamacppSplitMode::Layer),
            "row"   => Ok(LlamacppSplitMode::Row),
            _ => match s.parse::<usize>() {
                Ok(n)  => Ok(LlamacppSplitMode::GPU(n)),
                Err(_) => Err(format!("Choose a GPU number or `layer` or `row`")),
            }
        }
    }
}

pub struct LlamacppConfig {
    pub model_gguf: String,
    pub n_ctx: usize,
    pub max_batch_total_tokens: usize,
    pub max_batch_size: usize,
    pub batch_timeout: Duration,
    pub n_threads: usize,
    pub n_gpu_layers: usize,
    pub split_mode: LlamacppSplitMode,
    pub use_mmap: bool,
    pub use_mlock: bool,
    pub flash_attention: bool,
}

#[derive(Debug)]
struct LlamacppRequest {
    input_ids: Vec<i32>,
    top_k: i32,
    top_p: f32,
    typical_p: f32,
    min_keep: usize,
    temp: f32,
    seed: u32,
    penalty_last_n: i32,
    penalty_repeat: f32,
    penalty_freq: f32,
    penalty_present: f32,
    max_new_tokens: usize,
    tx: UnboundedSender<Result<InferStreamResponse, InferError>>,
    time: Instant,
}

pub struct LlamacppBackend {
    tx: UnboundedSender<LlamacppRequest>,
    status: watch::Receiver<bool>,
}

impl LlamacppRequest {
    fn new(
        from: &ValidGenerateRequest,
        tx: UnboundedSender<Result<InferStreamResponse, InferError>>,
    ) -> Option<Self>{
        if let Some(input_ids) = from.input_ids.as_ref() {
            Some(LlamacppRequest {
                input_ids:       input_ids.iter().map(|&x| x as i32).collect(),
                top_k:           from.parameters.top_k as _,
                top_p:           from.parameters.top_p as _,
                typical_p:       from.parameters.typical_p as _,
                min_keep:        0, // disabled
                temp:            from.parameters.temperature as _,
                seed:            from.parameters.seed as _,
                penalty_last_n:  -1, // 0 = disabled, -1 = context size
                penalty_repeat:  from.parameters.repetition_penalty as _,
                penalty_freq:    from.parameters.frequency_penalty as _,
                penalty_present: 0.0, // disabled
                max_new_tokens:  from.stopping_parameters.max_new_tokens as _,
                tx:              tx,
                time:            Instant::now(),
            })
        } else {
            None
        }
    }
}

struct Llamacpp {
    model: *mut bindings::llama_model,
    ctx: *mut bindings::llama_context,
    vocab: *const bindings::llama_vocab,
    logprobs: Vec<bindings::llama_token_data>,
    batch: bindings::llama_batch,
    n_ctx: u32,
}

extern "C" fn llamacpp_log_callback(
    level: bindings::ggml_log_level,
    msg: *const std::os::raw::c_char,
    _user_data: *mut std::os::raw::c_void,
) {
    let cmsg = unsafe { std::ffi::CStr::from_ptr(msg) };
    let rmsg = cmsg.to_string_lossy().trim_end_matches('\n').to_string();

    match level {
        bindings::GGML_LOG_LEVEL_DEBUG => debug!(target: "llamacpp", "{}", rmsg),
        bindings::GGML_LOG_LEVEL_INFO  =>  info!(target: "llamacpp", "{}", rmsg),
        bindings::GGML_LOG_LEVEL_WARN  =>  warn!(target: "llamacpp", "{}", rmsg),
        bindings::GGML_LOG_LEVEL_ERROR => error!(target: "llamacpp", "{}", rmsg),
        _                              => trace!(target: "llamacpp", "{}", rmsg),
    }
}

impl Llamacpp {
    fn new(conf: LlamacppConfig) -> Result<Self, BackendError> {
        let gguf = CString::new(conf.model_gguf)?;

        let model = unsafe {
            let mut params = bindings::llama_model_default_params();
            params.n_gpu_layers = conf.n_gpu_layers as _;
            params.split_mode = match conf.split_mode {
                LlamacppSplitMode::GPU(_) => bindings::LLAMA_SPLIT_MODE_NONE,
                LlamacppSplitMode::Layer  => bindings::LLAMA_SPLIT_MODE_LAYER,
                LlamacppSplitMode::Row    => bindings::LLAMA_SPLIT_MODE_ROW,
            };
            params.main_gpu = match conf.split_mode {
                LlamacppSplitMode::GPU(n) => n as _,
                _ => 0,
            };
            info!(?params.split_mode);
            info!(?params.main_gpu);
            params.use_mmap  = conf.use_mmap;
            params.use_mlock = conf.use_mlock;
            bindings::llama_model_load_from_file(gguf.as_ptr(), params)
        };
        if model.is_null() {
            return Err(BackendError::Llamacpp("Failed to load model".to_string()))
        }
        let ctx = unsafe {
            let mut params = bindings::llama_context_default_params();
            params.n_ctx           = conf.n_ctx as _;
            params.n_batch         = conf.max_batch_total_tokens as _;
            params.n_ubatch        = conf.max_batch_total_tokens as _; // TODO ?
            params.n_seq_max       = conf.max_batch_size as _;
            params.n_threads       = conf.n_threads as _;
            params.n_threads_batch = conf.n_threads as _; // TODO ?
            params.flash_attn      = conf.flash_attention;
            params.no_perf         = true;
            bindings::llama_init_from_model(model, params)
        };
        if ctx.is_null() {
            return Err(BackendError::Llamacpp("Failed to init context".to_string()))
        }
        let n_ctx = unsafe { bindings::llama_n_ctx(ctx) };

        let vocab = unsafe {
            bindings::llama_model_get_vocab(model)
        };
        if vocab.is_null() {
            return Err(BackendError::Llamacpp("Failed to get vocab".to_string()));
        }
        let n_tokens = unsafe {
            bindings::llama_vocab_n_tokens(vocab)
        };
        let mut logprobs = Vec::with_capacity(n_tokens as usize);

        for token in 0..n_tokens {
            logprobs.push(bindings::llama_token_data {
                id: token,
                logit: 0.0,
                p: 0.0,
            });
        }
        let batch = unsafe {
            bindings::llama_batch_init(conf.max_batch_total_tokens as _, 0, 1)
        };
        Ok(Llamacpp{model, ctx, vocab, logprobs, n_ctx, batch})
    }

    fn batch_clear_logits(&mut self) {
        for n in 0..self.batch.n_tokens as usize{
            unsafe {
                *self.batch.logits.add(n) = 0 as i8;
            }
        }
    }

    fn batch_push(
        &mut self,
        token: bindings::llama_token,
        pos: bindings::llama_pos,
        seq_ids: &[bindings::llama_seq_id],
        logits: bool,
    ) {
        debug!("push {token} {pos} {logits}");
        // TODO check evertyhing..
        let n = self.batch.n_tokens as usize;

        unsafe {
            *self.batch.token.add(n) = token;
            *self.batch.pos.add(n) = pos;
            *self.batch.n_seq_id.add(n) = seq_ids.len() as i32;
        }
        for (i, &seq_id) in seq_ids.iter().enumerate() {
            unsafe {
                *(*self.batch.seq_id.add(n)).add(i) = seq_id;
            }
        }
        unsafe {
            *self.batch.logits.add(n) = logits as i8;
        }
        self.batch.n_tokens += 1;
    }

    // useless ?
    fn warmup(&self) {
        let mut buf: Vec<bindings::llama_token> = Vec::new();

        let bos = unsafe {
            bindings::llama_vocab_bos(self.vocab)
        };
        if bos != bindings::LLAMA_TOKEN_NULL {
            buf.push(bos);
        }
        let eos = unsafe {
            bindings::llama_vocab_eos(self.vocab)
        };
        if eos != bindings::LLAMA_TOKEN_NULL {
            buf.push(eos);
        }
        if buf.is_empty() {
            warn!("Warmup failed: no bos/eos...");
            return;
        }
        let batch = unsafe {
            bindings::llama_batch_get_one(buf.as_ptr() as _, buf.len() as _)
        };
        if unsafe { bindings::llama_decode(self.ctx, batch) } != 0 {
            error!("Warmup failed: llama_decode() returned an error");
        }
        unsafe {
            bindings::llama_kv_cache_clear(self.ctx);
            bindings::llama_synchronize(self.ctx);
        }
    }
}

impl Drop for Llamacpp {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            unsafe { bindings::llama_free(self.ctx) };
        }
        if !self.model.is_null() {
            unsafe { bindings::llama_model_free(self.model) };
        }
        unsafe { bindings::llama_batch_free(self.batch) };
    }
}

struct LlamacppSampler {
    chain: *mut bindings::llama_sampler,
}

impl LlamacppSampler {
    fn new(req: &LlamacppRequest) -> Option<Self> {
        let chain = unsafe {
            let params = bindings::llama_sampler_chain_default_params();
            bindings::llama_sampler_chain_init(params)
        };
        if chain.is_null() {
            error!("Failed to init sampler");
            return None;
        }
        let top_k = unsafe {
            bindings::llama_sampler_init_top_k(req.top_k)
        };
        let top_p = unsafe {
            bindings::llama_sampler_init_top_p(req.top_p, req.min_keep)
        };
        let typical_p = unsafe {
            bindings::llama_sampler_init_typical(req.typical_p, req.min_keep)
        };
        let temp = unsafe {
            bindings::llama_sampler_init_temp(req.temp)
        };
        let penalties = unsafe {
            bindings::llama_sampler_init_penalties(
                req.penalty_last_n,
                req.penalty_repeat,
                req.penalty_freq,
                req.penalty_present,
            )
        };
        let dist = unsafe {
            bindings::llama_sampler_init_dist(req.seed)
        };
        let mut failed = false;

        for (k, v) in &[(    "top_k", top_k    ),
                        (    "top_p", top_p    ),
                        ("typical_p", typical_p),
                        (     "temp", temp     ),
                        ("penalties", penalties),
                        (     "dist", dist     )] {
            if v.is_null() {
                error!("Failed to init {k} sampler");
                failed = true;
            } else {
                unsafe { bindings::llama_sampler_chain_add(chain, *v) };
            }
        }
        if failed {
            None
        } else {
            Some(LlamacppSampler{chain})
        }
    }

    fn sample(&self, llamacpp: &mut Llamacpp, idx: usize) -> (bindings::llama_token, f32) {
        let logits = unsafe {
            bindings::llama_get_logits_ith(llamacpp.ctx, idx as _)
        };
        for (token, logprob) in llamacpp.logprobs.iter_mut().enumerate() {
            *logprob = bindings::llama_token_data {
                id: token as _,
                logit: unsafe { *logits.offset(token as _) },
                p: 0.0,
            };
        }
        let mut view = bindings::llama_token_data_array {
            data: llamacpp.logprobs.as_mut_ptr(),
            size: llamacpp.logprobs.len(),
            selected: -1,
            sorted: false,
        };
        unsafe {
            bindings::llama_sampler_apply(self.chain, &mut view);
            let logprob = *view.data.offset(view.selected as _);
            bindings::llama_sampler_accept(self.chain, logprob.id);
            (logprob.id, logprob.logit) // maybe p.ln() ?
        }
    }
}

impl Drop for LlamacppSampler {
    fn drop(&mut self) {
        if !self.chain.is_null() {
            unsafe { bindings::llama_sampler_free(self.chain) };
        }
    }
}

static INIT: Once = Once::new();

impl LlamacppBackend {
    pub fn new(
        conf: LlamacppConfig,
        tokenizer: Tokenizer,
    ) -> (Self, oneshot::Receiver<Result<(),BackendError>>) {

        // Setup llama & export logs, once and for all
        INIT.call_once(|| unsafe {
            bindings::llama_log_set(Some(llamacpp_log_callback), std::ptr::null_mut());
            bindings::llama_backend_init();
            bindings::llama_numa_init(bindings::GGML_NUMA_STRATEGY_NUMACTL); // TODO add option & test
        });

        let (status_tx, status_rx) = watch::channel(false);
        let (ok_tx, ok_rx) = oneshot::channel();
        let (tx, mut rx) = unbounded_channel::<LlamacppRequest>();
        let (sync_tx, sync_rx) = mpsc::channel();

        spawn(async move {
            let mut n_tokens = 0;
            let mut requests = Vec::new();

            loop {
                match timeout(conf.batch_timeout, rx.recv()).await {
                    Ok(None) => break, // closed
                    Ok(Some(request)) => {
                        if requests.len() + 1 == conf.max_batch_size {
                            requests.push(request);
                            let _ = sync_tx.send(requests);
                            n_tokens = 0;
                            requests = Vec::new();
                            continue;
                        }
                        if n_tokens + request.input_ids.len() > conf.max_batch_total_tokens as usize {
                            let _ = sync_tx.send(requests);
                            n_tokens = request.input_ids.len();
                            requests = vec![request];
                            continue;
                        }
                        requests.push(request);
                    },
                    Err(_) => {
                        if !requests.is_empty() {
                            let _ = sync_tx.send(requests);
                            n_tokens = 0;
                            requests = Vec::new();
                        }
                    }
                }
            }
        });

        spawn_blocking(move || {
            let mut llamacpp = match Llamacpp::new(conf) {
                Ok(v)  => { let _ = ok_tx.send(Ok(())); v       },
                Err(e) => { let _ = ok_tx.send(Err(e)); return; },
            };
            llamacpp.warmup();

            let vocab = tokenizer.get_added_vocabulary();

            // health() returns true
            let _ = status_tx.send(true);

            while let Ok(requests) = sync_rx.recv() {

                // TODO: do a real batch
                for (_seq_id, request) in requests.iter().enumerate() {

                    debug!("Request: {:?}", request);
                    let start_time = Instant::now();
                    llamacpp.batch.n_tokens = 0;

                    for (pos, &token_id) in request.input_ids.iter().enumerate() {
                        llamacpp.batch_push(
                            token_id as bindings::llama_token,
                            pos      as bindings::llama_pos,
                            &[/* seq_id */ 0 as bindings::llama_seq_id],
                            true,
                        );
                    }
                    let mut pos = request.input_ids.len();

                // TODO: close this loop :)

                // TODO: move up for perf ?
                let sampler = match LlamacppSampler::new(&request) {
                    Some(sampler) => sampler,
                    _ => {
                        let _ = request.tx.send(Err(InferError::IncompleteGeneration));
                        continue;
                    },
                };
                let mut text = String::with_capacity(1024);
                let mut n_tokens: usize = 0;
                let mut n_new_tokens: usize = 0;

                loop {
                    match unsafe { bindings::llama_decode(llamacpp.ctx, llamacpp.batch) } {
                        0 => { },
                        1 => {
                            unsafe {
                                // TODO: seq_rm & seq_add if model is compatible
                                bindings::llama_kv_cache_clear(llamacpp.ctx);
                            }
                            let _ = request.tx.send(Err(InferError::IncompleteGeneration));
                            break;
                        },
                        _ => {
                            debug!("decode return <0");
                            let _ = request.tx.send(Err(InferError::IncompleteGeneration));
                            break;
                        },
                    };
                    let idx = llamacpp.batch.n_tokens as usize - 1;
                    let (next, logprob) = sampler.sample(&mut llamacpp, idx);
                    n_new_tokens += 1;

                    debug!("tokens: {n_tokens} new: {n_new_tokens}");

                    let kv_cache_used_cells = unsafe {
                        bindings::llama_get_kv_cache_used_cells(llamacpp.ctx)
                    };
                    let piece = match tokenizer.decode(&[next as u32], false) {
                        Ok(piece) => piece,
                        Err(e) => {
                            error!("Failed to decode token: {e}");
                            let _ = request.tx.send(Err(InferError::IncompleteGeneration));
                            break;
                        },
                    };
                    let special = vocab.is_special_token(&piece);

                    if !special {
                        text.push_str(&piece);
                    }
                    let token = Token {
                        id: next as _,
                        text: piece,
                        logprob: logprob,
                        special: special,
                    };
                    let finish: Option<FinishReason> = {
                        if unsafe { bindings::llama_vocab_is_eog(llamacpp.vocab, next) } {
                            Some(FinishReason::EndOfSequenceToken)
                        } else if n_new_tokens == request.max_new_tokens {
                            Some(FinishReason::Length)
                        } else if kv_cache_used_cells == llamacpp.n_ctx as i32 {
                            Some(FinishReason::Length) // TODO: check
                        } else {
                            None
                        }
                    };
                    if let Some(reason) = finish {
                        let _ = request.tx.send(Ok(InferStreamResponse::End {
                            token: token,
                            top_tokens: vec![],
                            generated_text: GeneratedText {
                                text: text,
                                generated_tokens: n_new_tokens as _,
                                finish_reason: reason,
                                seed: Some(request.seed as _),
                            },
                            start: start_time,
                            queued: request.time,
                        }));
                        break;
                    }
                    let _ = request.tx.send(Ok(InferStreamResponse::Intermediate {
                        token: token,
                        top_tokens: vec![],
                    }));
                    llamacpp.batch.n_tokens = 0;
                 // llamacpp.batch_clear_logits();
                    llamacpp.batch_push(next, pos as _, &[0], true);
                    pos += 1;
                }
            }
            } // TODO remove this
        });
        (Self{tx, status: status_rx}, ok_rx)
    }
}

#[async_trait]
impl Backend for LlamacppBackend {
    #[instrument(skip_all)]
    fn schedule(
        &self,
        request: ValidGenerateRequest,
    ) -> Result<UnboundedReceiverStream<Result<InferStreamResponse, InferError>>, InferError> {
        debug!(?request);
        let (tx, rx) = unbounded_channel::<Result<InferStreamResponse, InferError>>();
        match LlamacppRequest::new(&request, tx) {
            Some(v) => match self.tx.send(v) {
                Err(e) => Err(InferError::GenerationError(e.to_string())),
                _ => Ok(UnboundedReceiverStream::new(rx)),
            },
            _ => Err(InferError::GenerationError("Bad request".to_string())),
        }
    }

    async fn health(&self, _: bool) -> bool {
        *self.status.borrow()
    }

    fn name(&self) -> &'static str {
        "llamacpp"
    }
}

#[derive(Debug, Error)]
pub enum BackendError {
    #[error("CString error: {0}")]
    CStringError(#[from] std::ffi::NulError),
    #[error("Llamacpp error: {0}")]
    Llamacpp(String),
}
