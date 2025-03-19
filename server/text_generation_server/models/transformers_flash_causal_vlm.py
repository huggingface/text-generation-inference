import math
from typing import List, Optional, Dict, Tuple, Any, ContextManager
from collections import namedtuple

import torch
from opentelemetry import trace
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers.modeling_utils
from contextlib import nullcontext

from text_generation_server.models.flash_causal_lm import FlashCausalLM
from text_generation_server.models.vlm_causal_lm import VlmCausalLM, VlmCausalLMBatch
from text_generation_server.utils import initialize_torch_distributed

from text_generation_server.layers.attention import paged_attention, attention, Seqlen
from text_generation_server.layers.attention.kv_cache import KVScales, KVCache
from text_generation_server.models.globals import ATTENTION
from text_generation_server.models.metadata_kernels import block_tables_to_ragged


tracer = trace.get_tracer(__name__)


def tgi_flash_attention_forward(
    module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],  # This is a positional arg in Transformers
    kv_cache: List[KVCache],
    kv_head_mapping: torch.Tensor,
    slots: torch.Tensor,
    cu_seqlen_prefill: Optional[torch.Tensor],
    seqlen: Seqlen,
    block_tables: torch.Tensor,
    max_s: int,
    kv_scales: KVScales,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    **kwargs,  # This is needed to "absorb" other args passed by Transformers modeling
):
    kv_cache = kv_cache[module.layer_idx]
    query_states = query_states.transpose(1, 2).squeeze(dim=0)
    key_states = key_states.transpose(1, 2).squeeze(dim=0)
    value_states = value_states.transpose(1, 2).squeeze(dim=0)

    # Take care of updating the cache in-place
    kv_cache.store(key=key_states, value=value_states, slots=slots, kv_scales=kv_scales)

    _, num_heads, head_dim = query_states.shape
    softmax_scale = 1 / math.sqrt(head_dim) if softmax_scale is None else softmax_scale
    sliding_window = -1 if sliding_window is None else sliding_window

    if cu_seqlen_prefill is not None:
        attn_output = attention(
            query=query_states,
            key=key_states,
            value=value_states,
            kv_cache=kv_cache,
            kv_scales=kv_scales,
            seqlen=seqlen,
            block_tables=block_tables,
            softmax_scale=softmax_scale,
            window_size_left=sliding_window,
            softcap=softcap,
        )
    else:
        attn_output = paged_attention(
            query_states,
            kv_cache,
            kv_head_mapping,
            softmax_scale,
            block_tables,
            seqlen,
            max_s,
            kv_scales=kv_scales,
            softcap=softcap,
        )

    attn_output = attn_output.view(-1, num_heads * head_dim)

    return attn_output, None


transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS["tgi"] = tgi_flash_attention_forward

# The base TP plan of these models has replicated q/k/v. This means that each process will see the full states,
# hence we should not divide the number of heads by the world size. This is a known waste of VRAM (the cache
# will be fully replicated on each process) and GPU communication (additional all-gather operations), however due
# to internal constraints it was not (yet?) possible to circumvent
REPLICATED_ATTENTION_MODELS = [
    "olmo2",
    "phi3",
]


class TransformersFlashCausalVLM(VlmCausalLM):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        quantize: Optional[str] = None,
        speculator: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        default_dtype=torch.float16,
        trust_remote_code: bool = False,
        tokenizer_class=AutoTokenizer,
        kv_cache_dtype: Optional[torch.dtype] = None,
        processor_class=None,
        processor_kwargs=None,
        model_class=AutoModelForCausalLM,
    ):
        self.quantize = quantize
        self.process_group, rank, world_size = initialize_torch_distributed()

        if speculator:
            raise RuntimeError("Speculator decoding is not enabled for AutoModel")

        if torch.cuda.is_available():
            device = torch.device(f"cuda:{rank}")
            dtype = default_dtype if dtype is None else dtype
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            device = torch.device("xpu")
            dtype = default_dtype if dtype is None else dtype
        else:
            raise ValueError(
                "Flash `Transformers` modeling backend is not available on cpu."
            )

        # Initialize processor
        if processor_kwargs is None:
            processor_kwargs = {}
        self.processor = processor_class.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=trust_remote_code,
            **processor_kwargs,
        )
        self.batch_class = VlmCausalLMBatch

        tokenizer = tokenizer_class.from_pretrained(
            model_id,
            revision=revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=trust_remote_code,
        )

        model = model_class.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=dtype,
            load_in_8bit=quantize == "bitsandbytes",
            trust_remote_code=trust_remote_code,
            # attn_implementation="tgi",
            # TODO: prefer custom implementation
            attn_implementation="sdpa",
            device_map=device if world_size == 1 else None,
            tp_plan="auto" if world_size > 1 else None,
        )

        torch.distributed.barrier(group=self.process_group)

        if tokenizer.pad_token_id is None:
            if model.config.pad_token_id is not None:
                tokenizer.pad_token_id = model.config.pad_token_id
            elif model.config.eos_token_id is not None and isinstance(
                model.config.eos_token_id, int
            ):
                tokenizer.pad_token_id = model.config.eos_token_id
            elif tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        # VLM models define the config we care about in their text_config
        text_config = getattr(model.config, "text_config", None)
        if text_config is not None:
            config = text_config
        else:
            config = model.config

        self.num_layers = config.num_hidden_layers
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = getattr(config, "num_key_value_heads", self.num_heads)
        self.head_size = config.hidden_size // config.num_attention_heads

        # Skip it for models in the exception list
        if model.config.model_type not in REPLICATED_ATTENTION_MODELS:
            self.num_heads = self.num_heads // self.process_group.size()
            self.num_kv_heads = (
                self.num_kv_heads // self.process_group.size()
                if self.num_kv_heads > 1
                else self.num_kv_heads
            )

        self.cuda_graphs = {}
        self.kv_cache = []
        self.kv_cache_dtype = dtype if kv_cache_dtype is None else kv_cache_dtype

        if ATTENTION == "flashinfer":
            from text_generation_server.layers.attention.flashinfer import (
                create_prefill_state,
                create_decode_state,
                create_prefill_with_paged_kv_state,
            )

            self.prefill_state = create_prefill_state(device=device)
            self.prefill_with_paged_kv_state = create_prefill_with_paged_kv_state(
                device=device
            )

            self.decode_state = create_decode_state(
                device=device,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
            )

        self.num_groups = self.num_heads // self.num_kv_heads

        # Those will never change and will be used in the forwards
        self.kv_head_mapping = torch.arange(
            0, self.num_kv_heads, dtype=torch.int32, device=device
        ).repeat_interleave(self.num_groups)
        # This means no scale
        self.kv_scales = KVScales(
            torch.tensor(1.0, device=device),
            torch.tensor(1.0, device=device),
        )

        # Skip FlashCausalLM init.
        super(FlashCausalLM, self).__init__(
            model_id=model_id,
            model=model,
            tokenizer=tokenizer,
            requires_padding=False,
            dtype=dtype,
            device=device,
            rank=rank,
            world_size=world_size,
            sliding_window=getattr(config, "sliding_window", None),
            support_chunking=False,
        )

        # Monkey patch of `self.model.forward` to match `FlashCausalLM`. It avoids duplicating a lot of code
        # We first copy the original model.forward because we still need it in the monkey patch
        self.model.original_forward = self.model.forward
        self.model.forward = self._model_forward

        self.model.get_position_ids = self._get_position_ids

        text_model = namedtuple("TextModel", ["max_past"])

        # model has text_model.max_past
        self.model.text_model = text_model(max_past=1024)

        torch.distributed.barrier(group=self.process_group)

    @classmethod
    def fallback(
        cls,
        model_id: str,
        revision: Optional[str] = None,
        quantize: Optional[str] = None,
        speculator: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
        processor_class=None,
        processor_kwargs=None,
        model_class=AutoModelForCausalLM,
    ):
        return cls(
            model_id=model_id,
            revision=revision,
            quantize=quantize,
            speculator=speculator,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            processor_class=processor_class,
            processor_kwargs=processor_kwargs,
            model_class=model_class,
        )

    def warmup(self, batch, max_input_tokens, max_total_tokens):
        return super().warmup(batch, max_input_tokens, max_total_tokens)
        # max_supported_total_tokens, max_input_tokens, max_total_tokens = (
        #     5000, 5000, 5001
        # )
        # return max_supported_total_tokens, max_input_tokens, max_total_tokens

    # TODO: may need to be updated
    def _get_position_ids(self, input_ids: torch.Tensor, image_grid_thw: torch.Tensor):
        # return a 4D tensor of shape (batch_size, num_heads, num_kv_heads, head_size)
        # return torch.arange(input_ids.shape[-1], device=input_ids.device).view(
        #     1, 1, 1, -1
        # )
        return torch.arange(input_ids.shape[-1], device=input_ids.device)

    def _model_forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlen_prefill: Optional[torch.Tensor],
        kv_cache: List[KVCache],
        block_tables: torch.Tensor,
        slots: torch.Tensor,
        seqlen: Seqlen,
        max_s: int,
        lm_head_indices: Optional[torch.Tensor],
        pixel_values=None,
        pixel_attention_mask=None,
        image_sizes=None,
        image_grid_thw=None,
        prefill_cache_indices=None,  # not used, but passed to match original signature
        adapter_data=None,  # not supported, but passed to match original signature
    ):
        print("\n----")
        # TODO: adjust the values to the correct ones
        # initalized to None as they are expected by the original forward
        attention_mask = None
        past_key_values = None
        inputs_embeds = None
        labels = None
        use_cache = None
        output_attentions = None
        output_hidden_states = None
        return_dict = None
        pixel_values_videos = None
        video_grid_thw = None
        rope_deltas = None
        cache_position = None

        if pixel_values is not None:
            print("pixel_values shape", pixel_values.shape)

        logits = self.model.original_forward(
            input_ids=input_ids.unsqueeze(0),  # expand dim to fit Transformers
            attention_mask=attention_mask,
            position_ids=position_ids.unsqueeze(0),  # expand dim to fit Transformers
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            rope_deltas=rope_deltas,
            cache_position=cache_position,
        ).logits.squeeze(dim=0)

        print("logits shape", logits.shape)

        # dummy logic to avoid errors (must be correct size (batch_size, seq_len, vocab_size))
        # vocab_size = (logits.shape[0], 151936)
        vocab_size = (1, 151936)
        dummy_logits = torch.zeros(vocab_size, device=input_ids.device)

        # add the logits to the dummy_logits to the first index only
        dummy_logits[0, :logits.shape[1]] = logits[0]

        print("dummy_logits shape", dummy_logits.shape)

        return dummy_logits, None
