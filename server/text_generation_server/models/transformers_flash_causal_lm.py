import torch
import time
import sys
from dataclasses import dataclass
from opentelemetry import trace
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerBase
from typing import Optional, Tuple, List, Type, Dict, Any
from text_generation_server.utils.import_utils import SYSTEM
from text_generation_server.models import Model
from text_generation_server.utils.chunks import concat_text_chunks
from text_generation_server.utils.tokens import batch_top_tokens
from text_generation_server.models.types import (
    Batch,
    Tokens,
    Generation,
    GeneratedText,
)
from text_generation_server.pb import generate_pb2
from text_generation_server.utils import NextTokenChooser, StoppingCriteria, Sampling
from text_generation_server.models.flash_causal_lm import (
    FlashCausalLMBatch,
    FlashCausalLM,
)

from text_generation_server.utils.import_utils import (
    empty_cache,
    synchronize,
    get_free_memory,
)
from text_generation_server.utils.speculate import get_speculate
from text_generation_server.utils.dist import MEMORY_FRACTION

tracer = trace.get_tracer(__name__)

from text_generation_server.adapters import AdapterBatchData
from text_generation_server.layers.attention import reshape_and_cache
from transformers.cache_utils import Cache
from transformers.flash_attention_utils import _flash_supports_window_size
from flash_attn import flash_attn_varlen_func
from text_generation_server.layers.attention import paged_attention

from loguru import logger

# Why define it here?
BLOCK_SIZE: int = 16


def patch_everywhere(
    attribute_name: str, patch: Any, module_name_prefix: Optional[str] = None
):
    """
    Finds all occurences of `attribute_name` in the loaded modules and patches them with `patch`.

    Args:
        attribute_name (`str`):
            The name of attribute to patch.
        patch (`Any`):
            The patch for the attribute.
        module_name_prefix (`Optional[str]`, defaults to `None`):
            If set, only module names starting with this prefix will be considered for patching.
    """
    # sys.modules may be updated while being iterated over, hence the list copy.
    for name in list(sys.modules):
        module = sys.modules[name]
        if module_name_prefix is not None and not name.startswith(module_name_prefix):
            continue
        if hasattr(module, attribute_name):
            setattr(module, attribute_name, patch)


def _flash_attention_forward_patched(
    query_states,
    key_states,
    value_states,
    attention_mask,
    query_length,
    layer_idx: int,
    dropout=0.0,
    softmax_scale=None,
    is_causal=False,
    _flash_attn_uses_top_left_mask=False,
    sliding_window=None,
    cache_position=0,
    **kwargs,  #: Unpack[ExtraKwargs],
):
    _flash_attn_uses_top_left_mask = True  # TODO felix: fix rocm

    if not _flash_attn_uses_top_left_mask:
        causal = is_causal
    else:
        # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
        causal = is_causal and query_length != 1

    print(f"causal: {causal}")

    use_sliding_windows = (
        _flash_supports_window_size
        and sliding_window is not None
        and cache_position > sliding_window
    )
    flash_kwargs = (
        {"window_size": (sliding_window, sliding_window)} if use_sliding_windows else {}
    )

    print(f"kwargs {kwargs.keys()}")

    cu_seqlen_prefill = kwargs.get("cu_seqlen_prefill")
    max_seq_lens = kwargs.get("max_seq_lens")

    if cu_seqlen_prefill is not None:
        attn_output = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlen_prefill,
            cu_seqlens_k=cu_seqlen_prefill,
            max_seqlen_q=kwargs["max_s"],
            max_seqlen_k=kwargs["max_s"],
            dropout_p=dropout,
            softmax_scale=softmax_scale,
            causal=causal,
            # **kwargs,
            **flash_kwargs,
        )
    else:
        attn_output = torch.empty_like(query_states)

        paged_attention(
            attn_output,
            query_states,
            kwargs["kv_cache"][layer_idx][0],
            kwargs["kv_cache"][layer_idx][1],
            kwargs["kv_head_mapping"],
            softmax_scale,
            kwargs["block_tables"],
            kwargs["input_lengths"],
            kwargs["max_s"],
        )

    attn_output = attn_output.view(attn_output.shape[0], -1)

    return attn_output


class PagedCache(Cache):
    def __init__(self) -> None:
        pass

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        kv_cache = cache_kwargs["kv_cache"]
        reshape_and_cache(
            key_states,
            value_states,
            kv_cache[layer_idx][0],
            kv_cache[layer_idx][1],
            cache_kwargs["slots"],
        )

        if cache_kwargs["cu_seqlen_prefill"] is not None:
            return key_states, value_states
        else:
            return None, None

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        raise ValueError(
            "PagedCache.get_seq_length should never be called, please open an issue."
        )

    def get_max_length(self) -> Optional[int]:
        raise ValueError(
            "PagedCache.get_max_length should never be called, please open an issue."
        )


class TransformersFlashCausalLM(FlashCausalLM):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        quantize: Optional[str] = None,
        speculator: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
    ):
        if speculator:
            raise RuntimeError("Speculator decoding is not enabled for AutoModel")

        if torch.cuda.is_available():
            device = torch.device("cuda:0")  # TODO felix: fix support for accelerate
            dtype = torch.float16 if dtype is None else dtype
        else:
            if quantize:
                raise ValueError("quantization is not available on CPU")

            device = torch.device("cpu")
            dtype = torch.float32 if dtype is None else dtype

        tokenizer = AutoTokenizer.from_pretrained(
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
            device_map=None,
            load_in_8bit=quantize == "bitsandbytes",
            trust_remote_code=trust_remote_code,
            attn_implementation="flash_attention_2",
        )
        if (
            torch.cuda.is_available()
            and torch.cuda.device_count() == 1
            and quantize != "bitsandbytes"
        ):
            model = model.cuda()

        self.kv_cache = []

        # TODO felix: make this more general.
        self.num_layers = len(model.model.layers)
        self.num_kv_heads = model.config.num_key_value_heads
        self.head_size = model.config.hidden_size // model.config.num_attention_heads

        if tokenizer.pad_token_id is None:
            if model.config.pad_token_id is not None:
                tokenizer.pad_token_id = model.config.pad_token_id
            elif model.config.eos_token_id is not None:
                tokenizer.pad_token_id = model.config.eos_token_id
            elif tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        # Skip FlashCausalLM init.
        super(FlashCausalLM, self).__init__(
            model_id=model_id,
            model=model,
            tokenizer=tokenizer,
            requires_padding=False,
            dtype=dtype,
            device=device,
        )

    def warmup(self, batch: FlashCausalLMBatch):
        # The warmup batch is the biggest batch we could ever receive
        empty_cache()

        patch_everywhere("_flash_attention_forward", _flash_attention_forward_patched)

        try:
            self.init_kv_cache(
                batch.num_blocks,
                self.num_layers,
                self.num_kv_heads,
                self.head_size,
                self.dtype,
                self.device,
            )
            max_bt = batch.max_blocks
            max_s = max_bt * BLOCK_SIZE

            _, batch, _ = self.generate_token(batch)
        except torch.cuda.OutOfMemoryError as e:
            raise RuntimeError(
                f"Not enough memory to handle {len(batch.input_ids)} prefill tokens. "
                f"You need to decrease `--max-batch-prefill-tokens`"
            ) from e

        synchronize(self.device)

        # Inspired by the original implementation in [vllm](https://github.com/vllm-project/vllm)
        # Calculate the number of blocks that can be allocated with the free memory
        dtype_size = torch.tensor([], dtype=self.dtype).element_size()
        cache_block_size = BLOCK_SIZE * self.num_kv_heads * self.head_size
        total_cache_size = self.num_layers * cache_block_size * 2 * dtype_size

        free_memory = get_free_memory(self.device, MEMORY_FRACTION)
        batch_num_blocks = batch.num_blocks if batch is not None else 0

        num_blocks = (
            # Leave 5% for some wiggle room
            int((free_memory * 0.95) // total_cache_size)
            # Add batch.num_blocks as we allocated it above, so it is included in the peak memory.
            + batch_num_blocks
        )

        del batch

        self.init_kv_cache(
            num_blocks,
            self.num_layers,
            self.num_kv_heads,
            self.head_size,
            self.dtype,
            self.device,
        )

        return int(num_blocks * BLOCK_SIZE)

    def forward(
        self, batch: FlashCausalLMBatch, adapter_data: AdapterBatchData
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # NOTE: adapter_data: not supported

        input_ids = batch.input_ids
        position_ids = batch.position_ids
        cu_seqlen_prefill = batch.cu_seqlen_prefill
        kv_cache = self.kv_cache
        block_tables = batch.block_tables_tensor
        slots = batch.slots[batch.slot_indices]
        input_lengths = batch.input_lengths_tensor
        max_s = batch.max_seqlen
        lm_head_indices = batch.prefill_head_indices

        # TODO felix: support window attention
        # if cu_seqlen_prefill is None and self.max_past() is not None:
        #     # In decode, not prefill, we're actually overwriting the KV-cache
        #     # in a circular buffer mode.
        #     # This makes sure the max_s for the decode pass is correct.
        #     max_s = min(self.max_past(), max_s)

        bs = input_ids.shape[0]

        logits = self.model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=PagedCache(),
            cu_seqlen_prefill=cu_seqlen_prefill,
            kv_cache=kv_cache,
            block_tables=block_tables,
            slots=slots,
            input_lengths=input_lengths,
            max_s=max_s,
            prefill_cache_indices=batch.prefill_cache_indices,
            lm_head_indices=lm_head_indices,
            cache_position=False,
            return_dict=False,
        )[0]

        if lm_head_indices is not None:
            logits = logits[lm_head_indices]

        if batch.prefill_cache_indices is not None:
            batch.prefill_cache_indices = None

        speculative_logits = None

        return logits, speculative_logits
