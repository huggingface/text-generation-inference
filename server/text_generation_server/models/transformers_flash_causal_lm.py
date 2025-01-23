import math
from typing import List, Optional

import torch
from opentelemetry import trace
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers.modeling_utils

from text_generation_server.models.flash_causal_lm import FlashCausalLM
from text_generation_server.utils import initialize_torch_distributed

from text_generation_server.layers.attention import paged_attention, attention, Seqlen
from text_generation_server.layers.attention.kv_cache import KVScales, KVCache
from text_generation_server.models.globals import ATTENTION


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


class TransformersFlashCausalLM(FlashCausalLM):
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

        tokenizer = tokenizer_class.from_pretrained(
            model_id,
            revision=revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=trust_remote_code,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=dtype,
            load_in_8bit=quantize == "bitsandbytes",
            trust_remote_code=trust_remote_code,
            attn_implementation="tgi",
            device_map=device if world_size == 1 else None,
            tp_plan="auto" if world_size > 1 else None,
        )

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

        self.num_layers = model.config.num_hidden_layers
        self.num_heads = model.config.num_attention_heads // self.process_group.size()
        self.num_kv_heads = model.config.num_key_value_heads
        self.num_kv_heads = (
            self.num_kv_heads // self.process_group.size()
            if self.num_kv_heads > 1
            else self.num_kv_heads
        )
        self.head_size = model.config.hidden_size // model.config.num_attention_heads

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

        torch.distributed.barrier(group=self.process_group)
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
        )

        # Monkey patch of `self.model.forward` to match `FlashCausalLM`. It avoids duplicating a lot of code
        # We first copy the original model.forward because we still need it in the monkey patch
        self.model.original_forward = self.model.forward
        self.model.forward = self._model_forward

    @classmethod
    def fallback(
        cls,
        model_id: str,
        revision: Optional[str] = None,
        quantize: Optional[str] = None,
        speculator: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
    ):
        return cls(
            model_id=model_id,
            revision=revision,
            quantize=quantize,
            speculator=speculator,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )

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
        prefill_cache_indices=None,  # not used, but passed to match original signature
        adapter_data=None,  # not supported, but passed to match original signature
    ):
        hidden_states = self.model.model.forward(
            input_ids=input_ids.unsqueeze(0),  # expand dim to fit Transformers
            position_ids=position_ids.unsqueeze(0),  # expand dim to fit Transformers
            past_key_values=None,  # we use self.kv_cache instead of transformers cache object
            use_cache=False,  # we use self.kv_cache instead of transformers cache object
            return_dict=True,
            cu_seqlen_prefill=cu_seqlen_prefill,
            kv_cache=kv_cache,
            block_tables=block_tables,
            slots=slots,
            seqlen=seqlen,
            max_s=max_s,
            kv_head_mapping=self.kv_head_mapping,
            kv_scales=self.kv_scales,
        )[0].squeeze(dim=0)

        # And compute logits from the lm_head, slicing correctly the indices
        # NOTE: some logits post-processing (e.g. in gemma2) may be absent here with the split of the modules
        # To update with full Transformers support asap
        if lm_head_indices is not None:
            hidden_states = hidden_states[lm_head_indices]
        logits = self.model.lm_head(hidden_states)

        # For Granite while next transformers version is released and we can use `lm_head_indices` natively
        if hasattr(self.model.config, "logits_scaling"):
            logits = logits / self.model.config.logits_scaling
        # For Cohere for similar reasons
        elif hasattr(self.model, "logit_scale"):
            logits = logits * self.model.logit_scale

        return logits, None
