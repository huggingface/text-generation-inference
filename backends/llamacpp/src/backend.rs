use crate::llamacpp;

use async_trait::async_trait;
use std::ffi::CString;
use std::mem::replace;
use std::str::FromStr;
use std::sync::{mpsc, Once};
use text_generation_router::infer::{Backend, GeneratedText, InferError, InferStreamResponse};
use text_generation_router::validation::ValidGenerateRequest;
use text_generation_router::{FinishReason, Token};
use thiserror::Error;
use tokenizers::Tokenizer;
use tokio::sync::mpsc::{unbounded_channel, UnboundedSender};
use tokio::sync::{oneshot, watch};
use tokio::task::{spawn, spawn_blocking};
use tokio::time::{timeout, Duration, Instant};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::instrument;
use tracing::{debug, error, info, trace, warn};

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
            "row" => Ok(LlamacppSplitMode::Row),
            _ => match s.parse::<usize>() {
                Ok(n) => Ok(LlamacppSplitMode::GPU(n)),
                Err(_) => Err("Choose a GPU number or `layer` or `row`".to_string()),
            },
        }
    }
}

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum LlamacppNuma {
    Disabled,
    Distribute,
    Isolate,
    Numactl,
    Mirror,
}

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum LlamacppGGMLType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_K,
    IQ2_XXS,
    IQ2_XS,
    IQ3_XXS,
    IQ1_S,
    IQ4_NL,
    IQ3_S,
    IQ2_S,
    IQ4_XS,
    I8,
    I16,
    I32,
    I64,
    F64,
    IQ1_M,
    BF16,
    TQ1_0,
    TQ2_0,
}

// TODO: macro
impl LlamacppGGMLType {
    fn to_ggml_type(self) -> llamacpp::ggml_type {
        match self {
            LlamacppGGMLType::F32 => llamacpp::GGML_TYPE_F32,
            LlamacppGGMLType::F16 => llamacpp::GGML_TYPE_F16,
            LlamacppGGMLType::Q4_0 => llamacpp::GGML_TYPE_Q4_0,
            LlamacppGGMLType::Q4_1 => llamacpp::GGML_TYPE_Q4_1,
            LlamacppGGMLType::Q5_0 => llamacpp::GGML_TYPE_Q5_0,
            LlamacppGGMLType::Q5_1 => llamacpp::GGML_TYPE_Q5_1,
            LlamacppGGMLType::Q8_0 => llamacpp::GGML_TYPE_Q8_0,
            LlamacppGGMLType::Q8_1 => llamacpp::GGML_TYPE_Q8_1,
            LlamacppGGMLType::Q2_K => llamacpp::GGML_TYPE_Q2_K,
            LlamacppGGMLType::Q3_K => llamacpp::GGML_TYPE_Q3_K,
            LlamacppGGMLType::Q4_K => llamacpp::GGML_TYPE_Q4_K,
            LlamacppGGMLType::Q5_K => llamacpp::GGML_TYPE_Q5_K,
            LlamacppGGMLType::Q6_K => llamacpp::GGML_TYPE_Q6_K,
            LlamacppGGMLType::Q8_K => llamacpp::GGML_TYPE_Q8_K,
            LlamacppGGMLType::IQ2_XXS => llamacpp::GGML_TYPE_IQ2_XXS,
            LlamacppGGMLType::IQ2_XS => llamacpp::GGML_TYPE_IQ2_XS,
            LlamacppGGMLType::IQ3_XXS => llamacpp::GGML_TYPE_IQ3_XXS,
            LlamacppGGMLType::IQ1_S => llamacpp::GGML_TYPE_IQ1_S,
            LlamacppGGMLType::IQ4_NL => llamacpp::GGML_TYPE_IQ4_NL,
            LlamacppGGMLType::IQ3_S => llamacpp::GGML_TYPE_IQ3_S,
            LlamacppGGMLType::IQ2_S => llamacpp::GGML_TYPE_IQ2_S,
            LlamacppGGMLType::IQ4_XS => llamacpp::GGML_TYPE_IQ4_XS,
            LlamacppGGMLType::I8 => llamacpp::GGML_TYPE_I8,
            LlamacppGGMLType::I16 => llamacpp::GGML_TYPE_I16,
            LlamacppGGMLType::I32 => llamacpp::GGML_TYPE_I32,
            LlamacppGGMLType::I64 => llamacpp::GGML_TYPE_I64,
            LlamacppGGMLType::F64 => llamacpp::GGML_TYPE_F64,
            LlamacppGGMLType::IQ1_M => llamacpp::GGML_TYPE_IQ1_M,
            LlamacppGGMLType::BF16 => llamacpp::GGML_TYPE_BF16,
            LlamacppGGMLType::TQ1_0 => llamacpp::GGML_TYPE_TQ1_0,
            LlamacppGGMLType::TQ2_0 => llamacpp::GGML_TYPE_TQ2_0,
        }
    }
}

pub struct LlamacppConfig {
    pub model_gguf: String,
    pub max_batch_total_tokens: usize,
    pub max_physical_batch_total_tokens: usize,
    pub max_batch_size: usize,
    pub batch_timeout: Duration,
    pub n_threads: usize,
    pub n_threads_batch: usize,
    pub n_gpu_layers: usize,
    pub split_mode: LlamacppSplitMode,
    pub numa: LlamacppNuma,
    pub defrag_threshold: f32,
    pub use_mmap: bool,
    pub use_mlock: bool,
    pub offload_kqv: bool,
    pub flash_attention: bool,
    pub type_k: LlamacppGGMLType,
    pub type_v: LlamacppGGMLType,
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
    ) -> Option<Self> {
        from.input_ids.as_ref().map(|input_ids| LlamacppRequest {
            input_ids: input_ids.iter().map(|&x| x as i32).collect(),
            top_k: from.parameters.top_k as _,
            top_p: from.parameters.top_p as _,
            typical_p: from.parameters.typical_p as _,
            min_keep: 0, // disabled
            temp: from.parameters.temperature as _,
            seed: from.parameters.seed as _,
            penalty_last_n: 64, // 0 = disabled, -1 = context size
            penalty_repeat: from.parameters.repetition_penalty as _,
            penalty_freq: from.parameters.frequency_penalty as _,
            penalty_present: 0.0, // disabled
            max_new_tokens: from.stopping_parameters.max_new_tokens as _,
            tx,
            time: Instant::now(),
        })
    }
}

struct Llamacpp {
    model: *mut llamacpp::llama_model,
    ctx: *mut llamacpp::llama_context,
    vocab: *const llamacpp::llama_vocab,
    logprobs: Vec<llamacpp::llama_token_data>,
    batch: llamacpp::llama_batch,
}

extern "C" fn llamacpp_log_callback(
    level: llamacpp::ggml_log_level,
    msg: *const std::os::raw::c_char,
    _user_data: *mut std::os::raw::c_void,
) {
    let cmsg = unsafe { std::ffi::CStr::from_ptr(msg) };
    let rmsg = cmsg.to_string_lossy().trim_end_matches('\n').to_string();

    match level {
        llamacpp::GGML_LOG_LEVEL_DEBUG => debug!(target: "llamacpp", "{}", rmsg),
        llamacpp::GGML_LOG_LEVEL_INFO => info!(target: "llamacpp", "{}", rmsg),
        llamacpp::GGML_LOG_LEVEL_WARN => warn!(target: "llamacpp", "{}", rmsg),
        llamacpp::GGML_LOG_LEVEL_ERROR => error!(target: "llamacpp", "{}", rmsg),
        _ => trace!(target: "llamacpp", "{}", rmsg),
    }
}

impl Llamacpp {
    fn new(conf: LlamacppConfig) -> Result<Self, BackendError> {
        let gguf = CString::new(conf.model_gguf)?;

        let model = unsafe {
            let mut params = llamacpp::model_default_params();
            params.n_gpu_layers = conf.n_gpu_layers as _;
            params.split_mode = match conf.split_mode {
                LlamacppSplitMode::GPU(_) => llamacpp::LLAMA_SPLIT_MODE_NONE,
                LlamacppSplitMode::Layer => llamacpp::LLAMA_SPLIT_MODE_LAYER,
                LlamacppSplitMode::Row => llamacpp::LLAMA_SPLIT_MODE_ROW,
            };
            params.main_gpu = match conf.split_mode {
                LlamacppSplitMode::GPU(n) => n as _,
                _ => 0,
            };
            params.use_mmap = conf.use_mmap;
            params.use_mlock = conf.use_mlock;
            llamacpp::model_load_from_file(gguf.as_ptr(), params)
        };
        if model.is_null() {
            return Err(BackendError::Llamacpp("Failed to load model".to_string()));
        }
        let ctx = unsafe {
            let mut params = llamacpp::context_default_params();
            params.n_ctx = conf.max_batch_total_tokens as _;
            params.n_batch = conf.max_batch_total_tokens as _;
            params.n_ubatch = conf.max_physical_batch_total_tokens as _;
            params.n_seq_max = conf.max_batch_size as _;
            params.n_threads = conf.n_threads as _;
            params.n_threads_batch = conf.n_threads_batch as _;
            params.defrag_thold = conf.defrag_threshold;
            params.offload_kqv = conf.offload_kqv;
            params.flash_attn = conf.flash_attention;
            params.type_k = conf.type_k.to_ggml_type();
            params.type_v = conf.type_v.to_ggml_type();
            params.no_perf = true;
            llamacpp::init_from_model(model, params)
        };
        if ctx.is_null() {
            return Err(BackendError::Llamacpp("Failed to init context".to_string()));
        }
        let vocab = unsafe { llamacpp::model_get_vocab(model) };
        if vocab.is_null() {
            return Err(BackendError::Llamacpp("Failed to get vocab".to_string()));
        }
        let n_tokens = unsafe { llamacpp::vocab_n_tokens(vocab) };
        let mut logprobs = Vec::with_capacity(n_tokens as usize);

        for token in 0..n_tokens {
            logprobs.push(llamacpp::llama_token_data {
                id: token,
                logit: 0.0,
                p: 0.0,
            });
        }
        let batch = unsafe { llamacpp::batch_init(conf.max_batch_total_tokens as _, 0, 1) };
        Ok(Llamacpp {
            model,
            ctx,
            vocab,
            logprobs,
            batch,
        })
    }

    fn decode(&mut self) -> i32 {
        unsafe { llamacpp::decode(self.ctx, self.batch) }
    }

    fn clear_kv_cache(&mut self, seq_id: llamacpp::llama_seq_id) {
        unsafe {
            llamacpp::kv_cache_seq_rm(self.ctx, seq_id, -1, -1);
        }
    }

    fn batch_push(
        &mut self,
        token: llamacpp::llama_token,
        pos: llamacpp::llama_pos,
        seq_id: llamacpp::llama_seq_id,
        logits: bool,
    ) -> usize {
        let n = self.batch.n_tokens as usize;
        unsafe {
            *self.batch.token.add(n) = token;
            *self.batch.pos.add(n) = pos;
            *self.batch.n_seq_id.add(n) = 1;
            *(*self.batch.seq_id.add(n)).add(0) = seq_id;
            *self.batch.logits.add(n) = logits as i8;
        }
        self.batch.n_tokens += 1;
        n
    }
}

impl Drop for Llamacpp {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            unsafe { llamacpp::free(self.ctx) };
        }
        if !self.model.is_null() {
            unsafe { llamacpp::model_free(self.model) };
        }
        unsafe { llamacpp::batch_free(self.batch) };
    }
}

struct LlamacppSampler {
    chain: *mut llamacpp::llama_sampler,
}

impl LlamacppSampler {
    fn new(req: &LlamacppRequest) -> Option<Self> {
        let chain = unsafe {
            let params = llamacpp::sampler_chain_default_params();
            llamacpp::sampler_chain_init(params)
        };
        if chain.is_null() {
            error!("Failed to init sampler");
            return None;
        }
        let (top_k, top_p, typical_p, temp, penalties, dist) = unsafe {
            (
                llamacpp::sampler_init_top_k(req.top_k),
                llamacpp::sampler_init_top_p(req.top_p, req.min_keep),
                llamacpp::sampler_init_typical(req.typical_p, req.min_keep),
                llamacpp::sampler_init_temp(req.temp),
                llamacpp::sampler_init_penalties(
                    req.penalty_last_n,
                    req.penalty_repeat,
                    req.penalty_freq,
                    req.penalty_present,
                ),
                llamacpp::sampler_init_dist(req.seed),
            )
        };
        let all = &[
            ("top_k", top_k),
            ("top_p", top_p),
            ("typical_p", typical_p),
            ("temp", temp),
            ("penalties", penalties),
            ("dist", dist),
        ];
        let mut failed = false;

        for (k, v) in all {
            if v.is_null() {
                error!("Failed to init {k} sampler");
                failed = true;
            } else {
                unsafe { llamacpp::sampler_chain_add(chain, *v) };
            }
        }
        if failed {
            unsafe { llamacpp::sampler_free(chain) };
            None
        } else {
            Some(LlamacppSampler { chain })
        }
    }

    fn sample(&self, llamacpp: &mut Llamacpp, idx: usize) -> (llamacpp::llama_token, f32) {
        let logits = unsafe { llamacpp::get_logits_ith(llamacpp.ctx, idx as _) };
        for (token, logprob) in llamacpp.logprobs.iter_mut().enumerate() {
            *logprob = llamacpp::llama_token_data {
                id: token as _,
                logit: unsafe { *logits.add(token) },
                p: 0.0,
            };
        }
        let mut view = llamacpp::llama_token_data_array {
            data: llamacpp.logprobs.as_mut_ptr(),
            size: llamacpp.logprobs.len(),
            selected: -1,
            sorted: false,
        };
        unsafe {
            llamacpp::sampler_apply(self.chain, &mut view);
            let logprob = *view.data.offset(view.selected as _);
            llamacpp::sampler_accept(self.chain, logprob.id);
            (logprob.id, logprob.p.ln())
        }
    }
}

impl Drop for LlamacppSampler {
    fn drop(&mut self) {
        if !self.chain.is_null() {
            unsafe { llamacpp::sampler_free(self.chain) };
        }
    }
}

struct LlamacppSeq {
    id: usize,
    batch_pos: usize,
    token: llamacpp::llama_token,
    pos: llamacpp::llama_pos,
    sampler: LlamacppSampler,
    text: String,
    n_new_tokens: usize,
    running: bool,
}

static INIT: Once = Once::new();

impl LlamacppBackend {
    pub fn new(
        conf: LlamacppConfig,
        tokenizer: Tokenizer,
    ) -> (
        Self,
        oneshot::Receiver<Result<(), BackendError>>,
        watch::Sender<bool>,
    ) {
        // Setup llama & export logs, once and for all
        INIT.call_once(|| unsafe {
            llamacpp::log_set(Some(llamacpp_log_callback), std::ptr::null_mut());
            llamacpp::backend_init();
            llamacpp::numa_init(match conf.numa {
                LlamacppNuma::Disabled => llamacpp::GGML_NUMA_STRATEGY_DISABLED,
                LlamacppNuma::Distribute => llamacpp::GGML_NUMA_STRATEGY_DISTRIBUTE,
                LlamacppNuma::Isolate => llamacpp::GGML_NUMA_STRATEGY_ISOLATE,
                LlamacppNuma::Numactl => llamacpp::GGML_NUMA_STRATEGY_NUMACTL,
                LlamacppNuma::Mirror => llamacpp::GGML_NUMA_STRATEGY_MIRROR,
            });
        });

        let (status_tx, status_rx) = watch::channel(false);
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let (ok_tx, ok_rx) = oneshot::channel();
        let (tx, mut rx) = unbounded_channel::<LlamacppRequest>();
        let (sync_tx, sync_rx) = mpsc::channel();

        spawn(async move {
            let mut n_tokens = 0;
            let mut requests = Vec::with_capacity(conf.max_batch_size);

            let flush = |requests: &mut Vec<_>, n_tokens: &mut usize| {
                if !requests.is_empty() {
                    let _ =
                        sync_tx.send(replace(requests, Vec::with_capacity(conf.max_batch_size)));
                    *n_tokens = 0;
                }
            };
            loop {
                match timeout(conf.batch_timeout, rx.recv()).await {
                    Ok(Some(request)) => {
                        let n_tokens_to_add = request.input_ids.len();

                        if n_tokens + n_tokens_to_add > conf.max_batch_total_tokens {
                            flush(&mut requests, &mut n_tokens);
                        }
                        n_tokens += n_tokens_to_add;
                        requests.push(request);

                        if requests.len() == conf.max_batch_size {
                            flush(&mut requests, &mut n_tokens);
                        }
                    }
                    Ok(None) => break,                             // closed
                    Err(_) => flush(&mut requests, &mut n_tokens), // timeout
                }
            }
        });

        spawn_blocking(move || {
            let mut llamacpp = match Llamacpp::new(conf) {
                Ok(v) => {
                    let _ = ok_tx.send(Ok(()));
                    v
                }
                Err(e) => {
                    let _ = ok_tx.send(Err(e));
                    return;
                }
            };
            let vocab = tokenizer.get_added_vocabulary();

            // health() returns true
            let _ = status_tx.send(true);

            while let Ok(requests) = sync_rx.recv() {
                if *shutdown_rx.borrow() {
                    break;
                }
                let start_time = Instant::now();
                let mut seqs: Vec<LlamacppSeq> = Vec::with_capacity(requests.len());
                llamacpp.batch.n_tokens = 0;

                for (seq_id, request) in requests.iter().enumerate() {
                    debug!("Request: {:?}", request);
                    // TODO remove this
                    let sampler = match LlamacppSampler::new(request) {
                        Some(sampler) => sampler,
                        _ => {
                            let _ = request.tx.send(Err(InferError::IncompleteGeneration));
                            continue;
                        }
                    };
                    let last_pos = request.input_ids.len() - 1;

                    for (pos, &token_id) in request.input_ids.iter().enumerate() {
                        llamacpp.batch_push(
                            token_id as llamacpp::llama_token,
                            pos as llamacpp::llama_pos,
                            seq_id as llamacpp::llama_seq_id,
                            pos == last_pos, // check samplers
                        );
                    }
                    seqs.push(LlamacppSeq {
                        id: seq_id,
                        batch_pos: llamacpp.batch.n_tokens as usize - 1,
                        token: llamacpp::LLAMA_TOKEN_NULL,
                        pos: last_pos as llamacpp::llama_pos + 1,
                        sampler,
                        text: String::with_capacity(1024),
                        n_new_tokens: 0,
                        running: true,
                    });
                }
                while llamacpp.batch.n_tokens > 0 {
                    if llamacpp.decode() != 0 {
                        warn!("llama_decode failed, clearing kv cache");
                        llamacpp.clear_kv_cache(-1);
                        for seq in seqs.iter_mut() {
                            let _ = requests[seq.id]
                                .tx
                                .send(Err(InferError::IncompleteGeneration));
                            seq.running = false;
                        }
                        break;
                    }
                    for seq in seqs.iter_mut() {
                        if !seq.running {
                            continue;
                        }
                        let (next, logprob) = seq.sampler.sample(&mut llamacpp, seq.batch_pos);
                        seq.n_new_tokens += 1;
                        seq.token = next;

                        let piece = match tokenizer.decode(&[next as u32], false) {
                            Ok(piece) => piece,
                            Err(e) => {
                                error!("Failed to decode token: {e}");
                                let _ = requests[seq.id]
                                    .tx
                                    .send(Err(InferError::IncompleteGeneration));
                                seq.running = false;
                                continue;
                            }
                        };
                        let special = vocab.is_special_token(&piece);

                        if !special {
                            seq.text.push_str(&piece);
                        }
                        let token = Token {
                            id: next as _,
                            text: piece,
                            logprob,
                            special,
                        };
                        let finish: Option<FinishReason> = {
                            if unsafe { llamacpp::vocab_is_eog(llamacpp.vocab, next) } {
                                Some(FinishReason::EndOfSequenceToken)
                            } else if seq.n_new_tokens == requests[seq.id].max_new_tokens {
                                Some(FinishReason::Length)
                            } else {
                                None
                            }
                        };
                        if let Some(reason) = finish {
                            let _ = requests[seq.id].tx.send(Ok(InferStreamResponse::End {
                                token,
                                top_tokens: vec![],
                                generated_text: GeneratedText {
                                    text: seq.text.clone(),
                                    generated_tokens: seq.n_new_tokens as _,
                                    finish_reason: reason,
                                    seed: Some(requests[seq.id].seed as _),
                                },
                                start: start_time,
                                queued: requests[seq.id].time,
                            }));
                            seq.running = false;
                            continue;
                        }
                        let _ = requests[seq.id]
                            .tx
                            .send(Ok(InferStreamResponse::Intermediate {
                                token,
                                top_tokens: vec![],
                            }));
                    }
                    // generate a new batch
                    llamacpp.batch.n_tokens = 0;

                    for seq in seqs.iter_mut() {
                        if seq.running {
                            seq.batch_pos =
                                llamacpp.batch_push(seq.token, seq.pos, seq.id as _, true);
                            seq.pos += 1;
                        } else {
                            llamacpp.clear_kv_cache(seq.id as _);
                        }
                    }
                }
            }
        });
        (
            Self {
                tx,
                status: status_rx,
            },
            ok_rx,
            shutdown_tx,
        )
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
