import math
import sys
from typing import Optional, Tuple, Dict, Any

import torch
from opentelemetry import trace
from loguru import logger
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from text_generation_server.models.flash_causal_lm import (
    FlashCausalLMBatch,
    FlashCausalLM,
)
from text_generation_server.utils import (
    initialize_torch_distributed,
    weight_files,
    Weights,
)
from text_generation_server.utils.import_utils import (
    empty_cache,
    synchronize,
    get_free_memory,
)
from text_generation_server.adapters import AdapterBatchData
from text_generation_server.layers.attention import paged_attention, attention, Seqlen
from text_generation_server.layers.attention.kv_cache import KVScales
from text_generation_server.models.globals import ATTENTION
from text_generation_server.models.metadata_kernels import block_tables_to_ragged


tracer = trace.get_tracer(__name__)


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
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
    query_length: int,
    is_causal: bool,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    **kwargs,
):

    kv_cache = kwargs["kv_cache"][kwargs["layer_idx"]]
    # This means no scale
    kv_scales=KVScales(torch.tensor(1., device=key_states.device), torch.tensor(1., device=key_states.device))

    # Correctly reshape the states
    _, _, num_heads, head_dim = query_states.size()
    _, _, num_kv_heads, _ = key_states.size()
    # query_states = query_states.view(-1, num_heads, head_dim)
    # key_states = key_states.view(-1, num_kv_heads, head_dim)
    # value_states = value_states.view(-1, num_kv_heads, head_dim)
    query_states = query_states.squeeze(dim=0)
    key_states = key_states.squeeze(dim=0)
    value_states = value_states.squeeze(dim=0)

    # Take care of updating the cache in-place
    kv_cache.store(
        key=key_states,
        value=value_states,
        slots=kwargs["slots"],
        kv_scales=kv_scales
    )

    softmax_scale = 1 / math.sqrt(head_dim) if softmax_scale is None else softmax_scale
    sliding_window = -1 if sliding_window is None else sliding_window

    if kwargs["cu_seqlen_prefill"] is not None:
        attn_output = attention(
            query=query_states,
            key=key_states,
            value=value_states,
            kv_cache=kv_cache,
            kv_scales=kv_scales,
            seqlen=kwargs["seqlen"],
            block_tables=kwargs["block_tables"],
            softmax_scale=softmax_scale,
            window_size_left=sliding_window,
            softcap=softcap,
        )
    else:
        attn_output = paged_attention(
            query_states,
            kv_cache,
            kwargs["kv_head_mapping"],
            softmax_scale,
            kwargs["block_tables"],
            kwargs["seqlen"],
            kwargs["max_s"],
            kv_scales=kv_scales,
            softcap=softcap,
        )

    # attn_output = attn_output.view(attn_output.shape[0], -1)
    attn_output = attn_output.view(-1, num_heads * head_dim)

    return attn_output


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
        config_class=AutoConfig,
        kv_cache_dtype: Optional[torch.dtype] = None,
    ):
        self.quantize = quantize
        self.process_group, rank, world_size = initialize_torch_distributed()

        if speculator:
            raise RuntimeError("Speculator decoding is not enabled for AutoModel")

        device_count = 0
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            device_count = torch.cuda.device_count()
            dtype = torch.float16 if dtype is None else dtype
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            device = torch.device("xpu")
            device_count = torch.xpu.device_count()
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
            device_map=("auto" if device_count > 1 else None),
            load_in_8bit=quantize == "bitsandbytes",
            trust_remote_code=trust_remote_code,
            attn_implementation="flash_attention_2"
        )
        if device_count == 1 and quantize != "bitsandbytes":
            model = model.to(device)

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
        self.kv_head_mapping = torch.arange(
            0, self.num_kv_heads, dtype=torch.int32, device=device
        ).repeat_interleave(self.num_groups)

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

    def warmup(self, batch: FlashCausalLMBatch, max_input_tokens: Optional[int], max_total_tokens: Optional[int],):
        patch_everywhere("_flash_attention_forward", _flash_attention_forward_patched)
        return super().warmup(batch, max_input_tokens, max_total_tokens)

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
        cache_lengths_tensor = batch.cache_lengths_tensor
        max_s = batch.max_current_length
        lm_head_indices = batch.prefill_head_indices

        if cu_seqlen_prefill is None and self.max_past() is not None:
            # In decode, not prefill, we're actually overwriting the KV-cache
            # in a circular buffer mode.
            # This makes sure the max_s for the decode pass is correct.
            max_s = min(self.max_past(), max_s)

        bs = input_ids.shape[0]
        sorted_padded_bs = sorted([k for k in self.cuda_graphs.keys() if k >= bs])
        if sorted_padded_bs:
            # Get associated cuda graph
            cuda_graph = self.cuda_graphs[sorted_padded_bs[0]]
        else:
            cuda_graph = None

        if cu_seqlen_prefill is not None or cuda_graph is None:
            if ATTENTION == "flashinfer":
                block_tables = block_tables_to_ragged(
                    block_tables=block_tables,
                    input_lengths=batch.input_lengths,
                    cache_lengths=batch.cache_lengths,
                    input_lengths_tensor=batch.input_lengths_tensor,
                    cache_lengths_tensor=batch.cache_lengths_tensor,
                    max_current_length=batch.max_current_length,
                )
            with self._forward_context(
                block_tables=block_tables,
                cu_seqlen_prefill=cu_seqlen_prefill,
                input_lengths_tensor=input_lengths,
                cache_lengths_tensor=cache_lengths_tensor,
            ):
                seqlen = Seqlen(
                    input_lengths=input_lengths,
                    cache_lengths=cache_lengths_tensor,
                    cu_seqlen_q=cu_seqlen_prefill,
                    max_q=batch.max_input_length,
                    max_k=batch.max_current_length,
                )
                logits = self.model.forward(
                    input_ids=input_ids[None, ...],
                    position_ids=position_ids[None, ...],
                    past_key_values=None,
                    use_cache=False,  # we use self.kv_cache instead of transformers cache object
                    cu_seqlen_prefill=cu_seqlen_prefill,
                    kv_cache=kv_cache,
                    block_tables=block_tables,
                    slots=slots,
                    seqlen=seqlen,
                    max_s=max_s,
                    prefill_cache_indices=batch.prefill_cache_indices,
                    lm_head_indices=lm_head_indices,
                    kv_head_mapping=self.kv_head_mapping,
                ).logits[0, ...]
                print("SUCCESSFUL FORWARD")
                if batch.prefill_cache_indices is not None:
                    batch.prefill_cache_indices = None
                return logits, None

        # Copy inputs to the static inputs of the cuda graph
        # Static inputs are potentially padded
        cuda_graph["input_ids"][: input_ids.shape[0]] = input_ids
        cuda_graph["position_ids"][: position_ids.shape[-1]] = position_ids
        if ATTENTION == "flashinfer":
            block_tables = block_tables_to_ragged(
                block_tables=block_tables,
                input_lengths=batch.input_lengths,
                cache_lengths=batch.cache_lengths,
                input_lengths_tensor=batch.input_lengths_tensor,
                cache_lengths_tensor=batch.cache_lengths_tensor,
                max_current_length=batch.max_current_length,
            )
            # assert block_tables.shape[0] >= slots.shape[0]
            cuda_graph["block_tables"][: block_tables.shape[0]] = block_tables
        else:
            cuda_graph["block_tables"][
                : block_tables.shape[0], : block_tables.shape[1]
            ] = block_tables

        # XXX: This is working only because block 0 is reserved for the healthcheck
        # so it doesn't matter if we override it with bogus values.
        cuda_graph["slots"].fill_(0)
        cuda_graph["slots"][: slots.shape[0]] = slots
        cuda_graph["input_lengths"].zero_()
        cuda_graph["input_lengths"][: input_lengths.shape[0]] = input_lengths
        cuda_graph["cache_lengths"].zero_()
        cuda_graph["cache_lengths"][
            : cache_lengths_tensor.shape[0]
        ] = cache_lengths_tensor

        with self._forward_context(
            block_tables=cuda_graph["block_tables"],
            cu_seqlen_prefill=None,
            input_lengths_tensor=cuda_graph["input_lengths"],
            cache_lengths_tensor=cuda_graph["cache_lengths"],
            state=cuda_graph["state"],
        ):
            # Replay the graph
            cuda_graph["graph"].replay()

        # Slice output to the correct shape
        logits = cuda_graph["logits"][:bs]
        return logits, None
