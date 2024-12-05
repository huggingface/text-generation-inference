use clap::{Parser, ValueEnum};
use hf_hub::{
    api::sync::{Api, ApiBuilder},
    Repo, RepoType,
};
use nix::sys::signal::{self, Signal};
use nix::unistd::Pid;
use regex::Regex;
use serde::Deserialize;
use std::env;
use std::ffi::OsString;
use std::io::{BufRead, BufReader};
use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::path::Path;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::TryRecvError;
use std::sync::{mpsc, Arc};
use std::thread;
use std::thread::sleep;
use std::time::{Duration, Instant};
use std::{
    fs, io,
    io::{Read, Write},
};
use thiserror::Error;
use tracing_subscriber::{filter::LevelFilter, EnvFilter};

mod env_runtime;
mod gpu;

fn compute_optimal(config: Option<&Config>, compute: Option<&ComputeType>) -> Option<usize> {
    if let (Some(config), Some(compute)) = (config, compute) {
        if let (Some(f16_max_compute), Some(model_compute)) = (compute.f16_flop(), config.flop()) {
            tracing::debug!("MAx compute {f16_max_compute} model compute {model_compute}");
            let optimal_size = (f16_max_compute / model_compute) as usize;
            if optimal_size > 100 {
                // Ignore calculations that's too low
                // Most likely an error
                Some(optimal_size)
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    }
}

fn get_config(
    model_id: &str,
    revision: &Option<String>,
) -> Result<Config, Box<dyn std::error::Error>> {
    let mut path = std::path::Path::new(model_id).to_path_buf();
    let model_id = model_id.to_string();
    let filename = if !path.exists() {
        // Assume it's a hub id

        let api = if let Ok(token) = std::env::var("HF_TOKEN") {
            // env variable has precedence over on file token.
            ApiBuilder::new().with_token(Some(token)).build()?
        } else {
            Api::new()?
        };
        let repo = if let Some(ref revision) = revision {
            api.repo(Repo::with_revision(
                model_id,
                RepoType::Model,
                revision.to_string(),
            ))
        } else {
            api.model(model_id)
        };
        repo.get("config.json")?
    } else {
        path.push("config.json");
        path
    };

    let content = std::fs::read_to_string(filename)?;
    let config: RawConfig = serde_json::from_str(&content)?;

    let config: Config = config.into();
    Ok(config)
}

fn resolve_attention(config: &Option<Config>, lora_adapters: &Option<String>) -> (String, String) {
    let compute_capability = gpu::get_cuda_capability();
    let mut prefix_caching: Option<String> = std::env::var("PREFIX_CACHING").ok();
    let mut attention: Option<String> = std::env::var("ATTENTION").ok();
    if let Some(config) = config {
        if prefix_caching.is_none() {
            if config.vision_config.is_some() {
                tracing::info!("Disabling prefix caching because of VLM model");
                prefix_caching = Some("0".to_string());
            } else if config.is_encoder_decoder {
                tracing::info!("Disabling prefix caching because of seq2seq model");
                prefix_caching = Some("0".to_string());
            }
        }

        let fallback_attention = if matches!(compute_capability, Some((major, _)) if major < 8) {
            "paged"
        } else {
            "flashdecoding"
        };

        match config.head_dim {
            Some(h) if h == 64 || h == 128 || h == 256 => {
                if lora_adapters.is_some() && prefix_caching.is_none() {
                    tracing::info!("Disabling prefix caching because of lora adapters");
                    prefix_caching = Some("0".to_string());
                }
                match config.model_type.as_deref() {
                    Some("falcon") | Some("deepseek_v2") => {
                        // Required because gemma2 needs bfloat16 which is not supported by
                        // flashinfer ?
                        if attention.is_none() {
                            tracing::info!(
                                "Forcing attention to '{fallback_attention}' because model {} requires it",
                                config.model_type.as_ref().unwrap()
                            );
                            attention = Some(fallback_attention.to_string());
                        }
                        if fallback_attention == "paged" && prefix_caching.is_none() {
                            tracing::info!("Disabling prefix caching because it is not supported with 'paged' attention");
                            prefix_caching = Some("0".to_string());
                        }
                    }
                    Some("t5") => {}
                    _ => {}
                }
            }
            _ => {
                if attention.is_none() {
                    tracing::info!("Forcing attention to '{fallback_attention}' because head dim is not supported by flashinfer, also disabling prefix caching");
                    attention = Some(fallback_attention.to_string());
                }
                if prefix_caching.is_none() {
                    prefix_caching = Some("0".to_string());
                }
            }
        }
    }
    if attention == Some("paged".to_string()) && prefix_caching.is_none() {
        tracing::info!("Disabling prefix caching on paged attention");
        prefix_caching = Some("0".to_string());
    }

    let attention = attention.unwrap_or("flashinfer".to_string());
    let prefix_caching = prefix_caching.unwrap_or("true".to_string());

    (prefix_caching, attention)
}

#[derive(Deserialize)]
struct RawConfig {
    max_position_embeddings: Option<usize>,
    n_positions: Option<usize>,
    model_type: Option<String>,
    max_seq_len: Option<usize>,
    quantization_config: Option<QuantizationConfig>,
    n_embd: Option<usize>,
    hidden_size: Option<usize>,
    intermediate_size: Option<usize>,
    num_attention_heads: Option<usize>,
    num_key_value_heads: Option<usize>,
    num_hidden_layers: Option<usize>,
    head_dim: Option<usize>,
    vision_config: Option<VisionConfig>,
    is_encoder_decoder: Option<bool>,
    #[serde(rename = "num_experts_per_tok")]
    experts: Option<usize>,
}

#[derive(Deserialize)]
struct QuantizationConfig {
    quant_method: Option<Quantization>,
}

#[derive(Debug, Deserialize)]
struct VisionConfig {}

#[derive(Debug, Deserialize)]
struct Config {
    max_position_embeddings: Option<usize>,
    quantize: Option<Quantization>,
    head_dim: Option<usize>,
    num_heads: Option<usize>,
    num_kv_heads: Option<usize>,
    num_layers: Option<usize>,
    intermediate_size: Option<usize>,
    hidden_size: Option<usize>,
    model_type: Option<String>,
    vision_config: Option<VisionConfig>,
    is_encoder_decoder: bool,
    experts: Option<usize>,
}

impl Config {
    fn flop(&self) -> Option<u64> {
        if self.vision_config.is_some() {
            // VLM are much harder to predict and VRAM requirements
            // Are more complex.
            return None;
        }
        let num_heads = self.num_heads? as u64;
        let num_kv_heads = self.num_kv_heads? as u64;
        let head_dim = self.head_dim? as u64;
        let hidden_size = self.hidden_size? as u64;
        let intermediate_size = if let Some(experts) = self.experts {
            (self.intermediate_size? * experts) as u64
        } else {
            self.intermediate_size? as u64
        };
        let num_layers = self.num_layers? as u64;

        let q_flops = 2 * num_heads * head_dim * hidden_size;
        let k_flops = 2 * num_kv_heads * head_dim * hidden_size;
        let v_flops = 2 * num_kv_heads * head_dim * hidden_size;
        let attn_flops = 2 * num_heads * head_dim * hidden_size;
        let o_flops = 2 * num_heads * head_dim * hidden_size;
        let attn_layer_flops = q_flops + k_flops + v_flops + attn_flops + o_flops;

        let gate_up_down_flops = 2 * 3 * hidden_size * intermediate_size;

        let layer_flops = attn_layer_flops + gate_up_down_flops;
        let total = layer_flops * num_layers;
        Some(total)
    }
}

impl From<RawConfig> for Config {
    fn from(other: RawConfig) -> Self {
        let max_position_embeddings = other
            .max_position_embeddings
            .or(other.max_seq_len)
            .or(other.n_positions);
        let quantize = other.quantization_config.and_then(|q| q.quant_method);
        let hidden_size = other.hidden_size.or(other.n_embd);
        let head_dim = other
            .head_dim
            .or_else(|| match (hidden_size, other.num_attention_heads) {
                (Some(hidden_size), Some(num_attention_heads))
                    if hidden_size % num_attention_heads == 0 =>
                {
                    Some(hidden_size / num_attention_heads)
                }
                _ => None,
            });
        let num_heads = other.num_attention_heads;
        let num_layers = other.num_hidden_layers;
        let num_kv_heads = other.num_key_value_heads.or(other.num_attention_heads);
        let intermediate_size = other.intermediate_size;
        let model_type = other.model_type;
        let vision_config = other.vision_config;
        let is_encoder_decoder = other.is_encoder_decoder.unwrap_or(false);
        let experts = other.experts;
        Config {
            max_position_embeddings,
            quantize,
            head_dim,
            model_type,
            vision_config,
            is_encoder_decoder,
            hidden_size,
            num_heads,
            num_kv_heads,
            intermediate_size,
            num_layers,
            experts,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum Quantization {
    /// 4 bit quantization. Requires a specific AWQ quantized model:
    ///   <https://hf.co/models?search=awq>.
    /// Should replace GPTQ models wherever possible because of the better latency
    Awq,
    /// Compressed tensors, which can be a mixture of different quantization methods.
    CompressedTensors,
    /// 8 bit quantization, doesn't require specific model.
    /// Should be a drop-in replacement to bitsandbytes with much better performance.
    /// Kernels are from <https://github.com/NetEase-FuXi/EETQ.git>
    Eetq,
    /// Variable bit quantization. Requires a specific EXL2 quantized model:
    /// <https://hf.co/models?search=exl2>. Requires exllama2 kernels and does
    /// not support tensor parallelism (num_shard > 1).
    Exl2,
    /// 4 bit quantization. Requires a specific GTPQ quantized model: <https://hf.co/models?search=gptq>.
    /// text-generation-inference will use exllama (faster) kernels wherever possible, and use
    /// triton kernel (wider support) when it's not.
    /// AWQ has faster kernels.
    Gptq,
    /// 4 bit quantization. Requires a specific Marlin quantized model: <https://hf.co/models?search=marlin>.
    Marlin,
    /// Bitsandbytes 8bit. Can be applied on any model, will cut the memory requirement in half,
    /// but it is known that the model will be much slower to run than the native f16.
    // #[deprecated(
    //     since = "1.1.0",
    //     note = "Use `eetq` instead, which provides better latencies overall and is drop-in in most cases"
    // )]
    Bitsandbytes,
    /// Bitsandbytes 4bit. Can be applied on any model, will cut the memory requirement by 4x,
    /// but it is known that the model will be much slower to run than the native f16.
    BitsandbytesNf4,
    /// Bitsandbytes 4bit. nf4 should be preferred in most cases but maybe this one has better
    /// perplexity performance for you model
    BitsandbytesFp4,
    /// [FP8](https://developer.nvidia.com/blog/nvidia-arm-and-intel-publish-fp8-specification-for-standardization-as-an-interchange-format-for-ai/) (e4m3) works on H100 and above
    /// This dtype has native ops should be the fastest if available.
    /// This is currently not the fastest because of local unpacking + padding to satisfy matrix
    /// multiplication limitations.
    Fp8,
}

impl std::fmt::Display for Quantization {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // To keep in track with `server`.
        match self {
            #[allow(deprecated)]
            // Use `eetq` instead, which provides better latencies overall and is drop-in in most cases
            Quantization::Bitsandbytes => {
                write!(f, "bitsandbytes")
            }
            Quantization::BitsandbytesNf4 => {
                write!(f, "bitsandbytes-nf4")
            }
            Quantization::BitsandbytesFp4 => {
                write!(f, "bitsandbytes-fp4")
            }
            Quantization::Exl2 => {
                write!(f, "exl2")
            }
            Quantization::Gptq => {
                write!(f, "gptq")
            }
            Quantization::Marlin => {
                write!(f, "marlin")
            }
            Quantization::Awq => {
                write!(f, "awq")
            }
            Quantization::CompressedTensors => {
                write!(f, "compressed-tensors")
            }
            Quantization::Eetq => {
                write!(f, "eetq")
            }
            Quantization::Fp8 => {
                write!(f, "fp8")
            }
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Dtype {
    Float16,
    #[clap(name = "bfloat16")]
    BFloat16,
}

impl std::fmt::Display for Dtype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // To keep in track with `server`.
        match self {
            Dtype::Float16 => {
                write!(f, "float16")
            }
            Dtype::BFloat16 => {
                write!(f, "bfloat16")
            }
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum KVCacheDtype {
    #[clap(name = "fp8_e4m3fn")]
    Fp8e4m3fn,

    #[clap(name = "fp8_e5m2")]
    Fp8e5m2,
}

impl std::fmt::Display for KVCacheDtype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KVCacheDtype::Fp8e4m3fn => {
                write!(f, "fp8_e4m3fn")
            }
            KVCacheDtype::Fp8e5m2 => {
                write!(f, "fp8_e5m2")
            }
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum RopeScaling {
    Linear,
    Dynamic,
}

impl std::fmt::Display for RopeScaling {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // To keep in track with `server`.
        match self {
            RopeScaling::Linear => {
                write!(f, "linear")
            }
            RopeScaling::Dynamic => {
                write!(f, "dynamic")
            }
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum UsageStatsLevel {
    /// Default option, usage statistics are collected anonymously
    On,
    /// Disables all collection of usage statistics
    Off,
    /// Doesn't send the error stack trace or error type, but allows sending a crash event
    NoStack,
}

impl std::fmt::Display for UsageStatsLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // To keep in track with `server`.
        match self {
            UsageStatsLevel::On => {
                write!(f, "on")
            }
            UsageStatsLevel::Off => {
                write!(f, "off")
            }
            UsageStatsLevel::NoStack => {
                write!(f, "no-stack")
            }
        }
    }
}

/// App Configuration
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// The name of the model to load.
    /// Can be a MODEL_ID as listed on <https://hf.co/models> like
    /// `gpt2` or `OpenAssistant/oasst-sft-1-pythia-12b`.
    /// Or it can be a local directory containing the necessary files
    /// as saved by `save_pretrained(...)` methods of transformers
    #[clap(default_value = "bigscience/bloom-560m", long, env)]
    model_id: String,

    /// The actual revision of the model if you're referring to a model
    /// on the hub. You can use a specific commit id or a branch like `refs/pr/2`.
    #[clap(long, env)]
    revision: Option<String>,

    /// The number of tokenizer workers used for payload validation and truncation inside the
    /// router.
    #[clap(default_value = "2", long, env)]
    validation_workers: usize,

    /// Whether to shard the model across multiple GPUs
    /// By default text-generation-inference will use all available GPUs to run
    /// the model. Setting it to `false` deactivates `num_shard`.
    #[clap(long, env)]
    sharded: Option<bool>,

    /// The number of shards to use if you don't want to use all GPUs on a given machine.
    /// You can use `CUDA_VISIBLE_DEVICES=0,1 text-generation-launcher... --num_shard 2`
    /// and `CUDA_VISIBLE_DEVICES=2,3 text-generation-launcher... --num_shard 2` to
    /// launch 2 copies with 2 shard each on a given machine with 4 GPUs for instance.
    #[clap(long, env)]
    num_shard: Option<usize>,

    /// Quantization method to use for the model. It is not necessary to specify this option
    /// for pre-quantized models, since the quantization method is read from the model
    /// configuration.
    ///
    /// Marlin kernels will be used automatically for GPTQ/AWQ models.
    #[clap(long, env, value_enum)]
    quantize: Option<Quantization>,

    /// The number of input_ids to speculate on
    /// If using a medusa model, the heads will be picked up automatically
    /// Other wise, it will use n-gram speculation which is relatively free
    /// in terms of compute, but the speedup heavily depends on the task.
    #[clap(long, env)]
    speculate: Option<usize>,

    /// The dtype to be forced upon the model. This option cannot be used with `--quantize`.
    #[clap(long, env, value_enum)]
    dtype: Option<Dtype>,

    /// Specify the dtype for the key-value cache. When this option is not provided,
    /// the dtype of the model is used (typically `float16` or `bfloat16`). Currently
    /// the only supported value are `fp8_e4m3fn` and `fp8_e5m2` on CUDA.
    #[clap(long, env, value_enum)]
    kv_cache_dtype: Option<KVCacheDtype>,

    /// Whether you want to execute hub modelling code. Explicitly passing a `revision` is
    /// encouraged when loading a model with custom code to ensure no malicious code has been
    /// contributed in a newer revision.
    #[clap(long, env, value_enum)]
    trust_remote_code: bool,

    /// The maximum amount of concurrent requests for this particular deployment.
    /// Having a low limit will refuse clients requests instead of having them
    /// wait for too long and is usually good to handle backpressure correctly.
    #[clap(default_value = "128", long, env)]
    max_concurrent_requests: usize,

    /// This is the maximum allowed value for clients to set `best_of`.
    /// Best of makes `n` generations at the same time, and return the best
    /// in terms of overall log probability over the entire generated sequence
    #[clap(default_value = "2", long, env)]
    max_best_of: usize,

    /// This is the maximum allowed value for clients to set `stop_sequences`.
    /// Stop sequences are used to allow the model to stop on more than just
    /// the EOS token, and enable more complex "prompting" where users can preprompt
    /// the model in a specific way and define their "own" stop token aligned with
    /// their prompt.
    #[clap(default_value = "4", long, env)]
    max_stop_sequences: usize,

    /// This is the maximum allowed value for clients to set `top_n_tokens`.
    /// `top_n_tokens` is used to return information about the the `n` most likely
    /// tokens at each generation step, instead of just the sampled token. This
    /// information can be used for downstream tasks like for classification or
    /// ranking.
    #[clap(default_value = "5", long, env)]
    max_top_n_tokens: u32,

    /// This is the maximum allowed input length (expressed in number of tokens)
    /// for users. The larger this value, the longer prompt users can send which
    /// can impact the overall memory required to handle the load.
    /// Please note that some models have a finite range of sequence they can handle.
    /// Default to min(max_allocatable, max_position_embeddings) - 1
    #[clap(long, env)]
    max_input_tokens: Option<usize>,

    /// Legacy version of [`Args::max_input_tokens`].
    #[clap(long, env)]
    max_input_length: Option<usize>,

    /// This is the most important value to set as it defines the "memory budget"
    /// of running clients requests.
    /// Clients will send input sequences and ask to generate `max_new_tokens`
    /// on top. with a value of `1512` users can send either a prompt of
    /// `1000` and ask for `512` new tokens, or send a prompt of `1` and ask for
    /// `1511` max_new_tokens.
    /// The larger this value, the larger amount each request will be in your RAM
    /// and the less effective batching can be.
    /// Default to min(max_allocatable, max_position_embeddings)
    #[clap(long, env)]
    max_total_tokens: Option<usize>,

    /// This represents the ratio of waiting queries vs running queries where
    /// you want to start considering pausing the running queries to include the waiting
    /// ones into the same batch.
    /// `waiting_served_ratio=1.2` Means when 12 queries are waiting and there's
    /// only 10 queries left in the current batch we check if we can fit those 12
    /// waiting queries into the batching strategy, and if yes, then batching happens
    /// delaying the 10 running queries by a `prefill` run.
    ///
    /// This setting is only applied if there is room in the batch
    /// as defined by `max_batch_total_tokens`.
    #[clap(default_value = "0.3", long, env)]
    waiting_served_ratio: f32,

    /// Limits the number of tokens for the prefill operation.
    /// Since this operation take the most memory and is compute bound, it is interesting
    /// to limit the number of requests that can be sent.
    /// Default to `max_input_tokens + 50` to give a bit of room.
    #[clap(long, env)]
    max_batch_prefill_tokens: Option<u32>,

    /// **IMPORTANT** This is one critical control to allow maximum usage
    /// of the available hardware.
    ///
    /// This represents the total amount of potential tokens within a batch.
    /// When using padding (not recommended) this would be equivalent of
    /// `batch_size` * `max_total_tokens`.
    ///
    /// However in the non-padded (flash attention) version this can be much finer.
    ///
    /// For `max_batch_total_tokens=1000`, you could fit `10` queries of `total_tokens=100`
    /// or a single query of `1000` tokens.
    ///
    /// Overall this number should be the largest possible amount that fits the
    /// remaining memory (after the model is loaded). Since the actual memory overhead
    /// depends on other parameters like if you're using quantization, flash attention
    /// or the model implementation, text-generation-inference cannot infer this number
    /// automatically.
    #[clap(long, env)]
    max_batch_total_tokens: Option<u32>,

    /// This setting defines how many tokens can be passed before forcing the waiting
    /// queries to be put on the batch (if the size of the batch allows for it).
    /// New queries require 1 `prefill` forward, which is different from `decode`
    /// and therefore you need to pause the running batch in order to run `prefill`
    /// to create the correct values for the waiting queries to be able to join the batch.
    ///
    /// With a value too small, queries will always "steal" the compute to run `prefill`
    /// and running queries will be delayed by a lot.
    ///
    /// With a value too big, waiting queries could wait for a very long time
    /// before being allowed a slot in the running batch. If your server is busy
    /// that means that requests that could run in ~2s on an empty server could
    /// end up running in ~20s because the query had to wait for 18s.
    ///
    /// This number is expressed in number of tokens to make it a bit more
    /// "model" agnostic, but what should really matter is the overall latency
    /// for end users.
    #[clap(default_value = "20", long, env)]
    max_waiting_tokens: usize,

    /// Enforce a maximum number of requests per batch
    /// Specific flag for hardware targets that do not support unpadded inference
    #[clap(long, env)]
    max_batch_size: Option<usize>,

    /// Specify the batch sizes to compute cuda graphs for.
    /// Use "0" to disable.
    /// Default = "1,2,4,8,16,32"
    #[clap(long, env, value_delimiter = ',')]
    cuda_graphs: Option<Vec<usize>>,

    /// The IP address to listen on
    #[clap(default_value = "0.0.0.0", long, env)]
    hostname: String,

    /// The port to listen on.
    #[clap(default_value = "3000", long, short, env)]
    port: u16,

    /// The name of the socket for gRPC communication between the webserver
    /// and the shards.
    #[clap(default_value = "/tmp/text-generation-server", long, env)]
    shard_uds_path: String,

    /// The address the master shard will listen on. (setting used by torch distributed)
    #[clap(default_value = "localhost", long, env)]
    master_addr: String,

    /// The address the master port will listen on. (setting used by torch distributed)
    #[clap(default_value = "29500", long, env)]
    master_port: usize,

    /// The location of the huggingface hub cache.
    /// Used to override the location if you want to provide a mounted disk for instance
    #[clap(long, env)]
    huggingface_hub_cache: Option<String>,

    /// The location of the huggingface hub cache.
    /// Used to override the location if you want to provide a mounted disk for instance
    #[clap(long, env)]
    weights_cache_override: Option<String>,

    /// For some models (like bloom), text-generation-inference implemented custom
    /// cuda kernels to speed up inference. Those kernels were only tested on A100.
    /// Use this flag to disable them if you're running on different hardware and
    /// encounter issues.
    #[clap(long, env)]
    disable_custom_kernels: bool,

    /// Limit the CUDA available memory.
    /// The allowed value equals the total visible memory multiplied by cuda-memory-fraction.
    #[clap(default_value = "1.0", long, env)]
    cuda_memory_fraction: f32,

    /// Rope scaling will only be used for RoPE models
    /// and allow rescaling the position rotary to accomodate for
    /// larger prompts.
    ///
    /// Goes together with `rope_factor`.
    ///
    /// `--rope-factor 2.0` gives linear scaling with a factor of 2.0
    /// `--rope-scaling dynamic` gives dynamic scaling with a factor of 1.0
    /// `--rope-scaling linear` gives linear scaling with a factor of 1.0 (Nothing will be changed
    /// basically)
    ///
    /// `--rope-scaling linear --rope-factor` fully describes the scaling you want
    #[clap(long, env)]
    rope_scaling: Option<RopeScaling>,

    /// Rope scaling will only be used for RoPE models
    /// See `rope_scaling`
    #[clap(long, env)]
    rope_factor: Option<f32>,

    /// Outputs the logs in JSON format (useful for telemetry)
    #[clap(long, env)]
    json_output: bool,

    #[clap(long, env)]
    otlp_endpoint: Option<String>,

    #[clap(default_value = "text-generation-inference.router", long, env)]
    otlp_service_name: String,

    #[clap(long, env)]
    cors_allow_origin: Vec<String>,

    #[clap(long, env)]
    api_key: Option<String>,

    #[clap(long, env)]
    watermark_gamma: Option<f32>,
    #[clap(long, env)]
    watermark_delta: Option<f32>,

    /// Enable ngrok tunneling
    #[clap(long, env)]
    ngrok: bool,

    /// ngrok authentication token
    #[clap(long, env)]
    ngrok_authtoken: Option<String>,

    /// ngrok edge
    #[clap(long, env)]
    ngrok_edge: Option<String>,

    /// The path to the tokenizer config file. This path is used to load the tokenizer configuration which may
    /// include a `chat_template`. If not provided, the default config will be used from the model hub.
    #[clap(long, env)]
    tokenizer_config_path: Option<String>,

    /// Disable outlines grammar constrained generation.
    /// This is a feature that allows you to generate text that follows a specific grammar.
    #[clap(long, env)]
    disable_grammar_support: bool,

    /// Display a lot of information about your runtime environment
    #[clap(long, short, action)]
    env: bool,

    /// Control the maximum number of inputs that a client can send in a single request
    #[clap(default_value = "4", long, env)]
    max_client_batch_size: usize,

    /// Lora Adapters a list of adapter ids i.e. `repo/adapter1,repo/adapter2` to load during
    /// startup that will be available to callers via the `adapter_id` field in a request.
    #[clap(long, env)]
    lora_adapters: Option<String>,

    /// Control if anonymous usage stats are collected.
    /// Options are "on", "off" and "no-stack"
    /// Defaul is on.
    #[clap(default_value = "on", long, env)]
    usage_stats: UsageStatsLevel,

    /// Payload size limit in bytes
    ///
    /// Default is 2MB
    #[clap(default_value = "2000000", long, env)]
    payload_limit: usize,

    /// Enables prefill logprobs
    ///
    /// Logprobs in the prompt are deactivated by default because they consume
    /// a large amount of VRAM (especially for long prompts).
    /// Using this flag reallows users to ask for them.
    #[clap(long, env)]
    enable_prefill_logprobs: bool,
}

#[derive(Debug)]
enum ShardStatus {
    Ready,
    Failed(usize),
}

#[allow(clippy::too_many_arguments)]
fn shard_manager(
    model_id: String,
    revision: Option<String>,
    quantize: Option<Quantization>,
    speculate: Option<usize>,
    dtype: Option<Dtype>,
    kv_cache_dtype: Option<KVCacheDtype>,
    trust_remote_code: bool,
    uds_path: String,
    rank: usize,
    world_size: usize,
    master_addr: String,
    master_port: usize,
    huggingface_hub_cache: Option<String>,
    weights_cache_override: Option<String>,
    disable_custom_kernels: bool,
    watermark_gamma: Option<f32>,
    watermark_delta: Option<f32>,
    cuda_graphs: Vec<usize>,
    cuda_memory_fraction: f32,
    rope_scaling: Option<RopeScaling>,
    rope_factor: Option<f32>,
    max_total_tokens: Option<usize>,
    max_batch_size: Option<usize>,
    max_input_tokens: Option<usize>,
    lora_adapters: Option<String>,
    enable_prefill_logprobs: bool,
    otlp_endpoint: Option<String>,
    otlp_service_name: String,
    log_level: LevelFilter,
    status_sender: mpsc::Sender<ShardStatus>,
    shutdown: Arc<AtomicBool>,
    _shutdown_sender: mpsc::Sender<()>,
) {
    // Enter shard-manager tracing span
    let _span = tracing::span!(tracing::Level::INFO, "shard-manager", rank = rank).entered();

    // Get UDS path
    let uds_string = format!("{uds_path}-{rank}");
    let uds = Path::new(&uds_string);
    // Clean previous runs
    if uds.exists() {
        fs::remove_file(uds).unwrap();
    }

    // Process args
    let mut shard_args = vec![
        "serve".to_string(),
        model_id,
        "--uds-path".to_string(),
        uds_path,
        "--logger-level".to_string(),
        log_level.to_string().to_uppercase(),
        "--json-output".to_string(),
    ];

    // Activate trust remote code
    if trust_remote_code {
        shard_args.push("--trust-remote-code".to_string());
    }

    // Activate tensor parallelism
    if world_size > 1 {
        shard_args.push("--sharded".to_string());
    }

    if let Some(quantize) = quantize {
        shard_args.push("--quantize".to_string());
        shard_args.push(quantize.to_string())
    }

    if let Some(speculate) = speculate {
        shard_args.push("--speculate".to_string());
        shard_args.push(speculate.to_string())
    }

    if let Some(dtype) = dtype {
        shard_args.push("--dtype".to_string());
        shard_args.push(dtype.to_string())
    }

    if let Some(kv_cache_dtype) = kv_cache_dtype {
        shard_args.push("--kv-cache-dtype".to_string());
        shard_args.push(kv_cache_dtype.to_string())
    }

    // Model optional revision
    if let Some(revision) = revision {
        shard_args.push("--revision".to_string());
        shard_args.push(revision)
    }

    let rope = match (rope_scaling, rope_factor) {
        (None, None) => None,
        (Some(scaling), None) => Some((scaling, 1.0)),
        (Some(scaling), Some(factor)) => Some((scaling, factor)),
        (None, Some(factor)) => Some((RopeScaling::Linear, factor)),
    };

    // OpenTelemetry Endpoint
    if let Some(otlp_endpoint) = otlp_endpoint {
        shard_args.push("--otlp-endpoint".to_string());
        shard_args.push(otlp_endpoint);
    }

    // OpenTelemetry Service Name
    shard_args.push("--otlp-service-name".to_string());
    shard_args.push(otlp_service_name);

    // In case we use sliding window, we may ignore the sliding in flash for some backends depending on the parameter.
    if let Some(max_input_tokens) = max_input_tokens {
        shard_args.push("--max-input-tokens".to_string());
        shard_args.push(max_input_tokens.to_string());
    }

    // Copy current process env
    let mut envs: Vec<(OsString, OsString)> = env::vars_os().collect();

    // Remove LOG_LEVEL if present
    envs.retain(|(name, _)| name != "LOG_LEVEL");

    // Torch Distributed Env vars
    envs.push(("RANK".into(), rank.to_string().into()));
    envs.push(("WORLD_SIZE".into(), world_size.to_string().into()));
    envs.push(("MASTER_ADDR".into(), master_addr.into()));
    envs.push(("MASTER_PORT".into(), master_port.to_string().into()));
    envs.push(("TORCH_NCCL_AVOID_RECORD_STREAMS".into(), "1".into()));

    // CUDA memory fraction
    envs.push((
        "CUDA_MEMORY_FRACTION".into(),
        cuda_memory_fraction.to_string().into(),
    ));

    // Safetensors load fast
    envs.push(("SAFETENSORS_FAST_GPU".into(), "1".into()));

    // Disable progress bar
    envs.push(("HF_HUB_DISABLE_PROGRESS_BARS".into(), "1".into()));

    // Enable hf transfer for insane download speeds
    let enable_hf_transfer = env::var("HF_HUB_ENABLE_HF_TRANSFER").unwrap_or("1".to_string());
    envs.push((
        "HF_HUB_ENABLE_HF_TRANSFER".into(),
        enable_hf_transfer.into(),
    ));

    // Parse Inference API token
    if let Ok(api_token) = env::var("HF_API_TOKEN") {
        envs.push(("HF_TOKEN".into(), api_token.into()))
    };

    // Detect rope scaling
    // Sending as env instead of CLI args to not bloat everything
    // those only can be used by RoPE models, so passing information around
    // for all models will complexify code unnecessarily
    if let Some((scaling, factor)) = rope {
        envs.push(("ROPE_SCALING".into(), scaling.to_string().into()));
        envs.push(("ROPE_FACTOR".into(), factor.to_string().into()));
    }

    if let Some(max_total_tokens) = max_total_tokens {
        envs.push((
            "MAX_TOTAL_TOKENS".into(),
            max_total_tokens.to_string().into(),
        ));
    }
    if let Some(max_batch_size) = max_batch_size {
        envs.push(("MAX_BATCH_SIZE".into(), max_batch_size.to_string().into()));
    }

    // Lora Adapters
    if let Some(lora_adapters) = lora_adapters {
        envs.push(("LORA_ADAPTERS".into(), lora_adapters.into()));
    }

    // Logprobs
    if enable_prefill_logprobs {
        envs.push(("REQUEST_LOGPROBS".into(), "1".into()));
    }

    // If huggingface_hub_cache is some, pass it to the shard
    // Useful when running inside a docker container
    if let Some(huggingface_hub_cache) = huggingface_hub_cache {
        envs.push(("HUGGINGFACE_HUB_CACHE".into(), huggingface_hub_cache.into()));
    };

    // If weights_cache_override is some, pass it to the shard
    // Useful when running inside a HuggingFace Inference Endpoint
    if let Some(weights_cache_override) = weights_cache_override {
        envs.push((
            "WEIGHTS_CACHE_OVERRIDE".into(),
            weights_cache_override.into(),
        ));
    };

    // Enable experimental support for cuda graphs
    if !cuda_graphs.is_empty() {
        envs.push((
            "CUDA_GRAPHS".into(),
            cuda_graphs
                .into_iter()
                .map(|c| c.to_string())
                .collect::<Vec<_>>()
                .join(",")
                .into(),
        ));
    }

    // If disable_custom_kernels is true, pass it to the shard as an env var
    if disable_custom_kernels {
        envs.push(("DISABLE_CUSTOM_KERNELS".into(), "True".into()))
    }

    // Watermark Gamma
    if let Some(watermark_gamma) = watermark_gamma {
        envs.push(("WATERMARK_GAMMA".into(), watermark_gamma.to_string().into()))
    }

    // Watermark Delta
    if let Some(watermark_delta) = watermark_delta {
        envs.push(("WATERMARK_DELTA".into(), watermark_delta.to_string().into()))
    }

    // Start process
    tracing::info!("Starting shard");
    let mut p = match Command::new("text-generation-server")
        .args(shard_args)
        .env_clear()
        .envs(envs)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0)
        .spawn()
    {
        Ok(p) => p,
        Err(err) => {
            if err.kind() == io::ErrorKind::NotFound {
                tracing::error!("text-generation-server not found in PATH");
                tracing::error!("Please install it with `make install-server`")
            }
            {
                tracing::error!("{}", err);
            }

            status_sender.send(ShardStatus::Failed(rank)).unwrap();
            return;
        }
    };

    // Redirect STDOUT to the console
    let mut pstdin = p.stdin.take().unwrap();
    let shard_stdout_reader = BufReader::new(p.stdout.take().unwrap());
    let shard_stderr_reader = BufReader::new(p.stderr.take().unwrap());

    //stdout tracing thread
    thread::spawn(move || {
        log_lines(shard_stdout_reader);
    });
    // We read stderr in another thread as it seems that lines() can block in some cases
    let (err_sender, err_receiver) = mpsc::channel();
    thread::spawn(move || {
        for line in shard_stderr_reader.lines().map_while(Result::ok) {
            err_sender.send(line).unwrap_or(());
        }
    });
    // We read stdin in another thread as it seems that lines() can block in some cases
    if LevelFilter::current() >= tracing::Level::DEBUG {
        thread::spawn(move || {
            let mut stdin = io::stdin(); // We get `Stdin` here.
            loop {
                let mut buffer = vec![0; 4096];
                if let Ok(n) = stdin.read(&mut buffer) {
                    if n > 0 {
                        let _ = pstdin.write_all(&buffer[..n]);
                    }
                }
            }
        });
    }

    let mut ready = false;
    let start_time = Instant::now();
    let mut wait_time = Instant::now();
    loop {
        // Process exited
        if let Some(exit_status) = p.try_wait().unwrap() {
            let mut err = String::new();
            while let Ok(line) = err_receiver.recv_timeout(Duration::from_millis(10)) {
                err = err + "\n" + &line;
            }

            tracing::error!("Shard complete standard error output:\n{err}");

            if let Some(signal) = exit_status.signal() {
                tracing::error!("Shard process was signaled to shutdown with signal {signal}");
            }

            status_sender.send(ShardStatus::Failed(rank)).unwrap();
            return;
        }

        // We received a shutdown signal
        if shutdown.load(Ordering::SeqCst) {
            terminate("shard", p, Duration::from_secs(90)).unwrap();
            return;
        }

        // Shard is ready
        if uds.exists() && !ready {
            tracing::info!("Shard ready in {:?}", start_time.elapsed());
            status_sender.send(ShardStatus::Ready).unwrap();
            ready = true;
        } else if !ready && wait_time.elapsed() > Duration::from_secs(10) {
            tracing::info!("Waiting for shard to be ready...");
            wait_time = Instant::now();
        }
        sleep(Duration::from_millis(100));
    }
}

fn shutdown_shards(shutdown: Arc<AtomicBool>, shutdown_receiver: &mpsc::Receiver<()>) {
    tracing::info!("Shutting down shards");
    // Update shutdown value to true
    // This will be picked up by the shard manager
    shutdown.store(true, Ordering::SeqCst);

    // Wait for shards to shutdown
    // This will block till all shutdown_sender are dropped
    let _ = shutdown_receiver.recv();
}

fn num_cuda_devices() -> Option<usize> {
    let devices = match env::var("CUDA_VISIBLE_DEVICES") {
        Ok(devices) => devices,
        Err(_) => match env::var("NVIDIA_VISIBLE_DEVICES") {
            Ok(devices) => devices,
            Err(_) => env::var("ZE_AFFINITY_MASK").ok()?,
        },
    };
    let n_devices = devices.split(',').count();
    Some(n_devices)
}

#[derive(Deserialize)]
#[serde(rename_all = "UPPERCASE")]
enum PythonLogLevelEnum {
    Trace,
    Debug,
    Info,
    Success,
    Warning,
    Error,
    Critical,
}

#[derive(Deserialize)]
struct PythonLogLevel {
    name: PythonLogLevelEnum,
}

#[derive(Deserialize)]
struct PythonLogRecord {
    level: PythonLogLevel,
}

#[derive(Deserialize)]
struct PythonLogMessage {
    text: String,
    record: PythonLogRecord,
}

impl PythonLogMessage {
    fn trace(&self) {
        match self.record.level.name {
            PythonLogLevelEnum::Trace => tracing::trace!("{}", self.text.trim_end()),
            PythonLogLevelEnum::Debug => tracing::debug!("{}", self.text.trim_end()),
            PythonLogLevelEnum::Info => tracing::info!("{}", self.text.trim_end()),
            PythonLogLevelEnum::Success => tracing::info!("{}", self.text.trim_end()),
            PythonLogLevelEnum::Warning => tracing::warn!("{}", self.text.trim_end()),
            PythonLogLevelEnum::Error => tracing::error!("{}", self.text.trim_end()),
            PythonLogLevelEnum::Critical => tracing::error!("{}", self.text.trim_end()),
        }
    }
}

impl TryFrom<&[u8]> for PythonLogMessage {
    type Error = serde_json::Error;

    fn try_from(value: &[u8]) -> Result<Self, Self::Error> {
        serde_json::from_slice::<Self>(value)
    }
}

fn log_lines<R: Sized + Read>(mut bufread: BufReader<R>) {
    let mut buffer = vec![0u8; 8 * 4096];
    let mut stdout = std::io::stdout();
    loop {
        let n = bufread.read(&mut buffer);
        if let Ok(n) = n {
            if n > 0 {
                let mut lines = buffer[..n].split(|i| *i == b'\n').peekable();
                while let Some(line) = lines.next() {
                    match PythonLogMessage::try_from(line) {
                        Ok(log) => log.trace(),
                        // For interactive debugging ?
                        Err(_) => {
                            if LevelFilter::current() >= tracing::Level::DEBUG {
                                stdout.write_all(line).unwrap();
                                if lines.peek().is_some() {
                                    stdout.write_all(b"\n").unwrap();
                                }
                                stdout.flush().unwrap();
                            }
                        }
                    }
                }
            } else {
                break;
            }
        }
    }
}

fn find_num_shards(
    sharded: Option<bool>,
    num_shard: Option<usize>,
) -> Result<usize, LauncherError> {
    // get the number of shards given `sharded` and `num_shard`
    let num_shard = match (sharded, num_shard) {
        (Some(true), None) => {
            // try to default to the number of available GPUs
            tracing::info!("Parsing num_shard from CUDA_VISIBLE_DEVICES/NVIDIA_VISIBLE_DEVICES/ZE_AFFINITY_MASK");
            let n_devices = num_cuda_devices()
                .expect("--num-shard and CUDA_VISIBLE_DEVICES/NVIDIA_VISIBLE_DEVICES/ZE_AFFINITY_MASK are not set");
            if n_devices <= 1 {
                return Err(LauncherError::NotEnoughCUDADevices(format!(
                    "`sharded` is true but only found {n_devices} CUDA devices"
                )));
            }
            n_devices
        }
        (Some(true), Some(num_shard)) => {
            // we can't have only one shard while sharded
            if num_shard <= 1 {
                return Err(LauncherError::ArgumentValidation(
                    "`sharded` is true but `num_shard` <= 1".to_string(),
                ));
            }
            num_shard
        }
        (Some(false), Some(num_shard)) => num_shard,
        (Some(false), None) => 1,
        (None, None) => num_cuda_devices().unwrap_or(1),
        (None, Some(num_shard)) => num_shard,
    };
    if num_shard < 1 {
        return Err(LauncherError::ArgumentValidation(
            "`num_shard` cannot be < 1".to_string(),
        ));
    }
    Ok(num_shard)
}

#[derive(Debug, Error)]
enum LauncherError {
    #[error("Invalid argument: {0}")]
    ArgumentValidation(String),
    #[error("not enough cuda devices: {0}")]
    NotEnoughCUDADevices(String),
    #[error("Download error")]
    DownloadError,
    #[error("Shard cannot start")]
    ShardCannotStart,
    #[error("Shard disconnected")]
    ShardDisconnected,
    #[error("Shard failed")]
    ShardFailed,
    #[error("Webserver failed")]
    WebserverFailed,
    #[error("Webserver cannot start")]
    WebserverCannotStart,
}

fn download_convert_model(
    model_id: &str,
    revision: Option<&str>,
    trust_remote_code: bool,
    huggingface_hub_cache: Option<&str>,
    weights_cache_override: Option<&str>,
    running: Arc<AtomicBool>,
    merge_lora: bool,
) -> Result<(), LauncherError> {
    // Enter download tracing span
    let _span = tracing::span!(tracing::Level::INFO, "download").entered();

    let mut download_args = vec![
        "download-weights".to_string(),
        model_id.to_string(),
        "--extension".to_string(),
        ".safetensors".to_string(),
        "--logger-level".to_string(),
        "INFO".to_string(),
        "--json-output".to_string(),
    ];

    if merge_lora {
        download_args.push("--merge-lora".to_string());
    }

    // Model optional revision
    if let Some(revision) = &revision {
        download_args.push("--revision".to_string());
        download_args.push(revision.to_string())
    }

    // Trust remote code for automatic peft fusion
    if trust_remote_code {
        download_args.push("--trust-remote-code".to_string());
    }

    // Copy current process env
    let mut envs: Vec<(OsString, OsString)> = env::vars_os().collect();

    // Remove LOG_LEVEL if present
    envs.retain(|(name, _)| name != "LOG_LEVEL");

    // Disable progress bar
    envs.push(("HF_HUB_DISABLE_PROGRESS_BARS".into(), "1".into()));

    // If huggingface_hub_cache is set, pass it to the download process
    // Useful when running inside a docker container
    if let Some(ref huggingface_hub_cache) = huggingface_hub_cache {
        envs.push(("HUGGINGFACE_HUB_CACHE".into(), huggingface_hub_cache.into()));
    };

    // Enable hf transfer for insane download speeds
    let enable_hf_transfer = env::var("HF_HUB_ENABLE_HF_TRANSFER").unwrap_or("1".to_string());
    envs.push((
        "HF_HUB_ENABLE_HF_TRANSFER".into(),
        enable_hf_transfer.into(),
    ));

    // Parse Inference API token
    if let Ok(api_token) = env::var("HF_API_TOKEN") {
        envs.push(("HF_TOKEN".into(), api_token.into()))
    };

    // If args.weights_cache_override is some, pass it to the download process
    // Useful when running inside a HuggingFace Inference Endpoint
    if let Some(weights_cache_override) = &weights_cache_override {
        envs.push((
            "WEIGHTS_CACHE_OVERRIDE".into(),
            weights_cache_override.into(),
        ));
    };

    // Start process
    tracing::info!("Starting check and download process for {model_id}");
    let mut download_process = match Command::new("text-generation-server")
        .args(download_args)
        .env_clear()
        .envs(envs)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0)
        .spawn()
    {
        Ok(p) => p,
        Err(err) => {
            if err.kind() == io::ErrorKind::NotFound {
                tracing::error!("text-generation-server not found in PATH");
                tracing::error!("Please install it with `make install-server`")
            } else {
                tracing::error!("{}", err);
            }

            return Err(LauncherError::DownloadError);
        }
    };

    let download_stdout = BufReader::new(download_process.stdout.take().unwrap());

    thread::spawn(move || {
        log_lines(download_stdout);
    });

    let download_stderr = BufReader::new(download_process.stderr.take().unwrap());

    // We read stderr in another thread as it seems that lines() can block in some cases
    let (err_sender, err_receiver) = mpsc::channel();
    thread::spawn(move || {
        for line in download_stderr.lines().map_while(Result::ok) {
            err_sender.send(line).unwrap_or(());
        }
    });

    loop {
        if let Some(status) = download_process.try_wait().unwrap() {
            if status.success() {
                tracing::info!("Successfully downloaded weights for {model_id}");
                break;
            }

            let mut err = String::new();
            while let Ok(line) = err_receiver.recv_timeout(Duration::from_millis(10)) {
                err = err + "\n" + &line;
            }

            if let Some(signal) = status.signal() {
                tracing::error!(
                    "Download process was signaled to shutdown with signal {signal}: {err}"
                );
            } else {
                tracing::error!("Download encountered an error: {err}");
            }

            return Err(LauncherError::DownloadError);
        }
        if !running.load(Ordering::SeqCst) {
            terminate("download", download_process, Duration::from_secs(10)).unwrap();
            return Ok(());
        }
        sleep(Duration::from_millis(100));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn spawn_shards(
    num_shard: usize,
    args: &Args,
    cuda_graphs: Vec<usize>,
    max_total_tokens: Option<usize>,
    max_input_tokens: Option<usize>,
    quantize: Option<Quantization>,
    max_log_level: LevelFilter,
    shutdown: Arc<AtomicBool>,
    shutdown_receiver: &mpsc::Receiver<()>,
    shutdown_sender: mpsc::Sender<()>,
    status_receiver: &mpsc::Receiver<ShardStatus>,
    status_sender: mpsc::Sender<ShardStatus>,
    running: Arc<AtomicBool>,
) -> Result<(), LauncherError> {
    // Start shard processes
    for rank in 0..num_shard {
        let model_id = args.model_id.clone();
        let revision = args.revision.clone();
        let uds_path = args.shard_uds_path.clone();
        let master_addr = args.master_addr.clone();
        let huggingface_hub_cache = args.huggingface_hub_cache.clone();
        let weights_cache_override = args.weights_cache_override.clone();
        let status_sender = status_sender.clone();
        let shutdown = shutdown.clone();
        let shutdown_sender = shutdown_sender.clone();
        let otlp_endpoint = args.otlp_endpoint.clone();
        let otlp_service_name = args.otlp_service_name.clone();
        let speculate = args.speculate;
        let dtype = args.dtype;
        let kv_cache_dtype = args.kv_cache_dtype;
        let trust_remote_code = args.trust_remote_code;
        let master_port = args.master_port;
        let disable_custom_kernels = args.disable_custom_kernels;
        let watermark_gamma = args.watermark_gamma;
        let watermark_delta = args.watermark_delta;
        let cuda_graphs_clone = cuda_graphs.clone();
        let cuda_memory_fraction = args.cuda_memory_fraction;
        let rope_scaling = args.rope_scaling;
        let rope_factor = args.rope_factor;
        let max_batch_size = args.max_batch_size;
        let lora_adapters = args.lora_adapters.clone();
        let enable_prefill_logprobs = args.enable_prefill_logprobs;
        thread::spawn(move || {
            shard_manager(
                model_id,
                revision,
                quantize,
                speculate,
                dtype,
                kv_cache_dtype,
                trust_remote_code,
                uds_path,
                rank,
                num_shard,
                master_addr,
                master_port,
                huggingface_hub_cache,
                weights_cache_override,
                disable_custom_kernels,
                watermark_gamma,
                watermark_delta,
                cuda_graphs_clone,
                cuda_memory_fraction,
                rope_scaling,
                rope_factor,
                max_total_tokens,
                max_batch_size,
                max_input_tokens,
                lora_adapters,
                enable_prefill_logprobs,
                otlp_endpoint,
                otlp_service_name,
                max_log_level,
                status_sender,
                shutdown,
                shutdown_sender,
            )
        });
    }
    drop(shutdown_sender);

    // Wait for shard to start
    let mut shard_ready = 0;
    while running.load(Ordering::SeqCst) {
        match status_receiver.try_recv() {
            Ok(ShardStatus::Ready) => {
                shard_ready += 1;
                if shard_ready == num_shard {
                    break;
                }
            }
            Err(TryRecvError::Empty) => {
                sleep(Duration::from_millis(100));
            }
            Ok(ShardStatus::Failed(rank)) => {
                tracing::error!("Shard {rank} failed to start");
                shutdown_shards(shutdown, shutdown_receiver);
                return Err(LauncherError::ShardCannotStart);
            }
            Err(TryRecvError::Disconnected) => {
                tracing::error!("Shard status channel disconnected");
                shutdown_shards(shutdown, shutdown_receiver);
                return Err(LauncherError::ShardDisconnected);
            }
        }
    }
    Ok(())
}

#[derive(Debug)]
struct ComputeType {
    count: usize,
    card: String,
}

impl ComputeType {
    fn f16_flop(&self) -> Option<u64> {
        let card_flop = match &self.card[..] {
            // https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/
            // Specs are unclear https://www.itcreations.com/nvidia-gpu/nvidia-geforce-rtx-4090-gpu
            "nvidia-4090" => Some(82 * 10u64.pow(12)),
            // https://www.nvidia.com/en-us/data-center/tesla-t4/
            "nvidia-t4" => Some(65 * 10u64.pow(12)),
            // https://www.nvidia.com/en-us/data-center/l4/
            "nvidia-l4" => Some(121 * 10u64.pow(12)),
            // https://www.nvidia.com/en-us/data-center/products/a10-gpu/
            "nvidia-a10g" => Some(125 * 10u64.pow(12)),
            // https://www.nvidia.com/en-us/data-center/h100/
            // https://www.techpowerup.com/gpu-specs/docs/nvidia-gh100-architecture.pdf
            "nvidia-h100-80gb-hbm3" => Some(900 * 10u64.pow(12)),
            // https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf
            "nvidia-a100" => Some(312 * 10u64.pow(12)),
            card => {
                tracing::warn!("Unkown compute for card {card}");
                None
            }
        };
        card_flop.map(|f| f * self.count as u64)
    }
}

impl From<ComputeType> for OsString {
    fn from(value: ComputeType) -> Self {
        format!("{}-{}", value.count, value.card).into()
    }
}

fn compute_type(num_shard: usize) -> Option<ComputeType> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=gpu_name", "--format=csv"])
        .output()
        .ok()?;
    let output = String::from_utf8(output.stdout).ok()?;
    let fullname = output.split('\n').nth(1)?;
    let cardname = fullname.replace(' ', "-").to_lowercase();
    Some(ComputeType {
        count: num_shard,
        card: cardname,
    })
}

fn spawn_webserver(
    num_shard: usize,
    args: Args,
    max_input_tokens: Option<usize>,
    max_total_tokens: Option<usize>,
    max_batch_prefill_tokens: u32,
    shutdown: Arc<AtomicBool>,
    shutdown_receiver: &mpsc::Receiver<()>,
) -> Result<Child, LauncherError> {
    // All shard started
    // Start webserver
    tracing::info!("Starting Webserver");
    let mut router_args = vec![
        "--max-client-batch-size".to_string(),
        args.max_client_batch_size.to_string(),
        "--max-concurrent-requests".to_string(),
        args.max_concurrent_requests.to_string(),
        "--max-best-of".to_string(),
        args.max_best_of.to_string(),
        "--max-stop-sequences".to_string(),
        args.max_stop_sequences.to_string(),
        "--max-top-n-tokens".to_string(),
        args.max_top_n_tokens.to_string(),
        "--max-batch-prefill-tokens".to_string(),
        max_batch_prefill_tokens.to_string(),
        "--waiting-served-ratio".to_string(),
        args.waiting_served_ratio.to_string(),
        "--max-waiting-tokens".to_string(),
        args.max_waiting_tokens.to_string(),
        "--validation-workers".to_string(),
        args.validation_workers.to_string(),
        "--hostname".to_string(),
        args.hostname.to_string(),
        "--port".to_string(),
        args.port.to_string(),
        "--master-shard-uds-path".to_string(),
        format!("{}-0", args.shard_uds_path),
        "--tokenizer-name".to_string(),
        args.model_id,
        "--payload-limit".to_string(),
        args.payload_limit.to_string(),
    ];
    if let Some(max_input_tokens) = max_input_tokens {
        router_args.extend_from_slice(&[
            "--max-input-tokens".to_string(),
            max_input_tokens.to_string(),
        ]);
    }
    if let Some(max_total_tokens) = max_total_tokens {
        router_args.extend_from_slice(&[
            "--max-total-tokens".to_string(),
            max_total_tokens.to_string(),
        ]);
    }

    // Pass usage stats flags to router
    router_args.push("--usage-stats".to_string());
    router_args.push(args.usage_stats.to_string());

    // Grammar support
    if args.disable_grammar_support {
        router_args.push("--disable-grammar-support".to_string());
    }

    // Tokenizer config path
    if let Some(ref tokenizer_config_path) = args.tokenizer_config_path {
        router_args.push("--tokenizer-config-path".to_string());
        router_args.push(tokenizer_config_path.to_string());
    }

    // Model optional max batch total tokens
    if let Some(max_batch_total_tokens) = args.max_batch_total_tokens {
        router_args.push("--max-batch-total-tokens".to_string());
        router_args.push(max_batch_total_tokens.to_string());
    }

    // Router optional max batch size
    if let Some(max_batch_size) = args.max_batch_size {
        router_args.push("--max-batch-size".to_string());
        router_args.push(max_batch_size.to_string());
    }

    // Model optional revision
    if let Some(ref revision) = args.revision {
        router_args.push("--revision".to_string());
        router_args.push(revision.to_string())
    }

    if args.trust_remote_code {
        router_args.push("--trust-remote-code".to_string());
    }

    if args.json_output {
        router_args.push("--json-output".to_string());
    }

    // OpenTelemetry
    if let Some(otlp_endpoint) = args.otlp_endpoint {
        router_args.push("--otlp-endpoint".to_string());
        router_args.push(otlp_endpoint);
    }

    // OpenTelemetry
    let otlp_service_name = args.otlp_service_name;
    router_args.push("--otlp-service-name".to_string());
    router_args.push(otlp_service_name);

    // CORS origins
    for origin in args.cors_allow_origin.into_iter() {
        router_args.push("--cors-allow-origin".to_string());
        router_args.push(origin);
    }

    // API Key
    if let Some(api_key) = args.api_key {
        router_args.push("--api-key".to_string());
        router_args.push(api_key);
    }
    // Ngrok
    if args.ngrok {
        router_args.push("--ngrok".to_string());
        router_args.push("--ngrok-authtoken".to_string());
        router_args.push(args.ngrok_authtoken.unwrap());
        router_args.push("--ngrok-edge".to_string());
        router_args.push(args.ngrok_edge.unwrap());
    }

    // Copy current process env
    let mut envs: Vec<(OsString, OsString)> = env::vars_os().collect();

    // Parse Inference API token
    if let Ok(api_token) = env::var("HF_API_TOKEN") {
        envs.push(("HF_TOKEN".into(), api_token.into()))
    };

    // Parse Compute type
    if let Ok(compute_type) = env::var("COMPUTE_TYPE") {
        envs.push(("COMPUTE_TYPE".into(), compute_type.into()))
    } else if let Some(compute_type) = compute_type(num_shard) {
        envs.push(("COMPUTE_TYPE".into(), compute_type.into()))
    }

    let mut webserver = match Command::new("text-generation-router")
        .args(router_args)
        .envs(envs)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0)
        .spawn()
    {
        Ok(p) => p,
        Err(err) => {
            tracing::error!("Failed to start webserver: {}", err);
            if err.kind() == io::ErrorKind::NotFound {
                tracing::error!("text-generation-router not found in PATH");
                tracing::error!("Please install it with `make install-router`")
            } else {
                tracing::error!("{}", err);
            }

            shutdown_shards(shutdown, shutdown_receiver);
            return Err(LauncherError::WebserverCannotStart);
        }
    };

    // Redirect STDOUT and STDERR to the console
    let webserver_stdout = webserver.stdout.take().unwrap();
    let webserver_stderr = webserver.stderr.take().unwrap();

    thread::spawn(move || {
        let stdout = BufReader::new(webserver_stdout);
        let stderr = BufReader::new(webserver_stderr);
        for line in stdout.lines() {
            println!("{}", line.unwrap());
        }
        for line in stderr.lines() {
            println!("{}", line.unwrap());
        }
    });
    Ok(webserver)
}

fn terminate(process_name: &str, mut process: Child, timeout: Duration) -> io::Result<ExitStatus> {
    tracing::info!("Terminating {process_name}");

    let terminate_time = Instant::now();
    signal::kill(Pid::from_raw(process.id() as i32), Signal::SIGTERM).unwrap();

    tracing::info!("Waiting for {process_name} to gracefully shutdown");
    while terminate_time.elapsed() < timeout {
        if let Some(status) = process.try_wait()? {
            tracing::info!("{process_name} terminated");
            return Ok(status);
        }
        sleep(Duration::from_millis(100));
    }
    tracing::info!("Killing {process_name}");

    process.kill()?;
    let exit_status = process.wait()?;

    tracing::info!("{process_name} killed");
    Ok(exit_status)
}

fn main() -> Result<(), LauncherError> {
    // Pattern match configuration
    let args: Args = Args::parse();

    // Filter events with LOG_LEVEL
    let varname = "LOG_LEVEL";
    let env_filter = if let Ok(log_level) = std::env::var(varname) {
        // Override to avoid simple logs to be spammed with tokio level informations
        let log_level = match &log_level[..] {
            "warn" => "text_generation_launcher=warn,text_generation_router=warn",
            "info" => "text_generation_launcher=info,text_generation_router=info",
            "debug" => "text_generation_launcher=debug,text_generation_router=debug",
            log_level => log_level,
        };
        EnvFilter::builder()
            .with_default_directive(LevelFilter::INFO.into())
            .parse_lossy(log_level)
    } else {
        EnvFilter::new("info")
    };
    let max_log_level = env_filter.max_level_hint().unwrap_or(LevelFilter::INFO);

    if args.json_output {
        tracing_subscriber::fmt()
            .with_env_filter(env_filter)
            .json()
            .init();
    } else {
        tracing_subscriber::fmt()
            .with_env_filter(env_filter)
            .compact()
            .init();
    }

    if args.env {
        let env_runtime = env_runtime::Env::new();
        tracing::info!("{}", env_runtime);
    }

    tracing::info!("{:#?}", args);

    let config: Option<Config> = get_config(&args.model_id, &args.revision).ok();
    let quantize = config.as_ref().and_then(|c| c.quantize);
    // Quantization usually means you're even more RAM constrained.

    let (prefix_caching, attention) = resolve_attention(&config, &args.lora_adapters);
    tracing::info!("Using attention {attention} - Prefix caching {prefix_caching}");
    std::env::set_var("PREFIX_CACHING", prefix_caching);
    std::env::set_var("ATTENTION", attention);

    let num_shard = find_num_shards(args.sharded, args.num_shard)?;
    if num_shard > 1 {
        if matches!(args.quantize, Some(Quantization::Exl2)) {
            return Err(LauncherError::ArgumentValidation(
                "Sharding is currently not supported with `exl2` quantization".into(),
            ));
        }
        tracing::info!("Sharding model on {num_shard} processes");
    }

    let max_input_tokens = {
        match (args.max_input_tokens, args.max_input_length) {
            (Some(max_input_tokens), Some(max_input_length)) => {
                return Err(LauncherError::ArgumentValidation(
                    format!("Both `max_input_tokens` ({max_input_tokens}) and `max_input_length` ({max_input_length}) are set. Please define only `max_input_tokens` as `max_input_length is deprecated for naming consistency.",
                )));
            }
            (Some(max_input_tokens), None) | (None, Some(max_input_tokens)) => {
                Some(max_input_tokens)
            }
            (None, None) => None,
        }
    };
    let max_total_tokens = args.max_total_tokens;
    let max_batch_prefill_tokens = {
        match args.max_batch_prefill_tokens {
            Some(max_batch_prefill_tokens) => max_batch_prefill_tokens,
            None => {
                // TODO figure out hardware optimal value
                let compute_type = compute_type(num_shard);
                let compute_optimal = compute_optimal(config.as_ref(), compute_type.as_ref());
                let default = compute_optimal.unwrap_or(4096);
                let max_position_embeddings = config.and_then(|c| c.max_position_embeddings);
                let value = if let Some(max_position_embeddings) = max_position_embeddings {
                    default.min(max_position_embeddings)
                } else {
                    default
                };
                tracing::info!("Default `max_batch_prefill_tokens` to {value}");
                value as u32
            }
        }
    };

    // Validate args
    if let (Some(max_input_tokens), Some(max_total_tokens)) = (max_input_tokens, max_total_tokens) {
        if max_input_tokens >= max_total_tokens {
            return Err(LauncherError::ArgumentValidation(
                    format!("`max_input_tokens`({max_input_tokens}) must be < `max_total_tokens`({max_total_tokens})"),
                ));
        }
    }

    if matches!(args.quantize, Some(Quantization::Bitsandbytes)) {
        tracing::warn!("Bitsandbytes is deprecated, use `eetq` instead, which provides better latencies overall and is drop-in in most cases.");
    }
    let quantize = args.quantize.or(quantize);
    let cuda_graphs = match (&args.cuda_graphs, &quantize) {
        (Some(cuda_graphs), _) => cuda_graphs.iter().cloned().filter(|&c| c > 0).collect(),
        #[allow(deprecated)]
        (
            None,
            Some(
                Quantization::Bitsandbytes
                | Quantization::BitsandbytesNf4
                | Quantization::BitsandbytesFp4,
            ),
        ) => {
            tracing::warn!("Bitsandbytes doesn't work with cuda graphs, deactivating them");
            vec![]
        }
        (None, Some(Quantization::Exl2)) => {
            tracing::warn!("Exl2 doesn't work with cuda graphs, deactivating them");
            vec![]
        }
        _ => {
            let cuda_graphs = vec![1, 2, 4, 8, 16, 32];
            tracing::info!("Using default cuda graphs {cuda_graphs:?}");
            cuda_graphs
        }
    };

    if args.validation_workers == 0 {
        return Err(LauncherError::ArgumentValidation(
            "`validation_workers` must be > 0".to_string(),
        ));
    }
    if args.trust_remote_code {
        tracing::warn!(
            "`trust_remote_code` is set. Trusting that model `{}` do not contain malicious code.",
            args.model_id
        );
    }

    if let Some(ref max_batch_total_tokens) = args.max_batch_total_tokens {
        if let Some(max_total_tokens) = max_total_tokens {
            if max_total_tokens as u32 > *max_batch_total_tokens {
                return Err(LauncherError::ArgumentValidation(format!(
                    "`max_total_tokens` must be <= `max_batch_total_tokens`. Given: {} and {}",
                    max_total_tokens, max_batch_total_tokens
                )));
            }
        }
    }

    if args.ngrok {
        if args.ngrok_authtoken.is_none() {
            return Err(LauncherError::ArgumentValidation(
                "`ngrok-authtoken` must be set when using ngrok tunneling".to_string(),
            ));
        }

        if args.ngrok_edge.is_none() {
            return Err(LauncherError::ArgumentValidation(
                "`ngrok-edge` must be set when using ngrok tunneling".to_string(),
            ));
        }
    }

    // Signal handler
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
    })
    .expect("Error setting Ctrl-C handler");

    // Download and convert model weights
    download_convert_model(
        &args.model_id,
        args.revision.as_deref(),
        args.trust_remote_code,
        args.huggingface_hub_cache.as_deref(),
        args.weights_cache_override.as_deref(),
        running.clone(),
        true, // if its only a lora model - we should merge the lora adapters
    )?;

    // Download and convert lora adapters if any
    if let Some(lora_adapters) = &args.lora_adapters {
        for adapter in lora_adapters.split(',') {
            // skip download if a path is provided
            if adapter.contains('=') {
                continue;
            }

            let adapter = adapter.trim();

            // check if adapter has more than 1 '@'
            if adapter.matches('@').count() > 1 {
                return Err(LauncherError::ArgumentValidation(format!(
                    "Invalid LoRA adapter format: {}",
                    adapter
                )));
            }

            // capture adapter_id, path, revision in format of adapter_id=path@revision
            let re = Regex::new(r"^([^=@]+)(?:=([^@]+))?(?:@(.+))?$").unwrap();
            if let Some(caps) = re.captures(adapter) {
                let adapter_id = caps.get(1).map_or("", |m| m.as_str());
                let revision = caps.get(3).map(|m| m.as_str());

                download_convert_model(
                    adapter_id,
                    revision,
                    args.trust_remote_code,
                    args.huggingface_hub_cache.as_deref(),
                    args.weights_cache_override.as_deref(),
                    running.clone(),
                    false, // avoid merging lora adapters if using multi-lora
                )?;
            } else {
                return Err(LauncherError::ArgumentValidation(format!(
                    "Invalid LoRA adapter format: {}",
                    adapter
                )));
            }
        }
    }

    if !running.load(Ordering::SeqCst) {
        // Launcher was asked to stop
        return Ok(());
    }

    // Shared shutdown bool
    let shutdown = Arc::new(AtomicBool::new(false));
    // Shared shutdown channel
    // When shutting down, the main thread will wait for all senders to be dropped
    let (shutdown_sender, shutdown_receiver) = mpsc::channel();

    // Shared channel to track shard status
    let (status_sender, status_receiver) = mpsc::channel();

    spawn_shards(
        num_shard,
        &args,
        cuda_graphs,
        max_total_tokens,
        max_input_tokens,
        quantize,
        max_log_level,
        shutdown.clone(),
        &shutdown_receiver,
        shutdown_sender,
        &status_receiver,
        status_sender,
        running.clone(),
    )?;

    // We might have received a termination signal
    if !running.load(Ordering::SeqCst) {
        shutdown_shards(shutdown, &shutdown_receiver);
        return Ok(());
    }

    let mut webserver = spawn_webserver(
        num_shard,
        args,
        max_input_tokens,
        max_total_tokens,
        max_batch_prefill_tokens,
        shutdown.clone(),
        &shutdown_receiver,
    )
    .inspect_err(|_| {
        shutdown_shards(shutdown.clone(), &shutdown_receiver);
    })?;

    // Default exit code
    let mut exit_code = Ok(());

    while running.load(Ordering::SeqCst) {
        if let Ok(ShardStatus::Failed(rank)) = status_receiver.try_recv() {
            tracing::error!("Shard {rank} crashed");
            exit_code = Err(LauncherError::ShardFailed);
            break;
        };

        match webserver.try_wait().unwrap() {
            Some(_) => {
                tracing::error!("Webserver Crashed");
                shutdown_shards(shutdown, &shutdown_receiver);
                return Err(LauncherError::WebserverFailed);
            }
            None => {
                sleep(Duration::from_millis(100));
            }
        };
    }

    // Graceful termination
    terminate("webserver", webserver, Duration::from_secs(90)).unwrap();
    shutdown_shards(shutdown, &shutdown_receiver);

    exit_code
}
