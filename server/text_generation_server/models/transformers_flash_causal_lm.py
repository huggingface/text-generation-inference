import math
from typing import List, Optional

import torch
from opentelemetry import trace
from transformers import AutoTokenizer, AutoProcessor
import transformers.modeling_utils

from text_generation_server.models.flash_causal_lm import FlashCausalLM
from text_generation_server.models.vlm_causal_lm import VlmCausalLM, VlmCausalLMBatch
from text_generation_server.utils import initialize_torch_distributed

from text_generation_server.layers.attention import paged_attention, attention, Seqlen
from text_generation_server.layers.attention.kv_cache import KVScales, KVCache
from text_generation_server.models.globals import ATTENTION, BLOCK_SIZE
import torch.nn.functional as F
import numpy as np

tracer = trace.get_tracer(__name__)

# The base TP plan of these models has replicated q/k/v. This means that each process will see the full states,
# hence we should not divide the number of heads by the world size. This is a known waste of VRAM (the cache
# will be fully replicated on each process) and GPU communication (additional all-gather operations), however due
# to internal constraints it was not (yet?) possible to circumvent
REPLICATED_ATTENTION_MODELS = [
    "olmo2",
    "phi3",
]


def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return -(a // -b)


# Adapted from: https://github.com/vllm-project/vllm/blob/e1a2c699dda82199e88e433c144eae66f3b31878/vllm/v1/attention/backends/flash_attn.py
def make_local_attention_virtual_batches(
    attn_chunk_size: int,
    query_start_loc_np: np.ndarray,
    seq_lens_np: np.ndarray,
    block_table: torch.Tensor,
    page_size: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, torch.Tensor]:
    q_seqlens = query_start_loc_np[1:] - query_start_loc_np[:-1]
    actual_batch_size = seq_lens_np.shape[0]
    # Handle if we are starting in the middle of a local attention block,
    #  we assume q_seqlens > 0 (for all elements), for each batch idx we compute
    #  the number of tokens that are not in the first local attention block and
    #  then we can simply use a cdiv for the rest.
    # For example if we have:
    #   attn_chunk_size = 4
    #   q_seqlens = [4, 10, 5]
    #   k_seqlens = [6, 17, 9]
    # Then we would get:
    #   new_tokens_in_first_block = [2, 1, 4]
    #   local_blocks = [2, 4, 2]
    q_tokens_in_first_block = np.minimum(
        attn_chunk_size - ((seq_lens_np - q_seqlens) % attn_chunk_size), q_seqlens
    ).astype(np.int32)
    tokens_in_last_block = attn_chunk_size + (seq_lens_np % -attn_chunk_size)
    local_blocks = 1 + cdiv(q_seqlens - q_tokens_in_first_block, attn_chunk_size)

    # Once we know the number of local blocks we can compute the request spans
    #  for each batch idx, we can figure out the number of "virtual" requests we
    #  have to make,
    # For the above example we would get:
    #   seqlens_q_local = [2, 2, 1, 4, 4, 1, 4, 1]
    #
    # First Get batched arange. (E.g., [2, 4, 2] -> [0, 1, 0, 1, 2, 3, 0, 1])
    #   (TODO: max a utility to share this code with _prepare_inputs)
    # arange step 1. [2, 4, 2] -> [2, 6, 8]
    cu_num_blocks = np.cumsum(local_blocks)
    virtual_batches = cu_num_blocks[-1]
    # arange step 2. [2, 6, 8] -> [0, 0, 2, 2, 2, 2, 6, 6]
    block_offsets = np.repeat(cu_num_blocks - local_blocks, local_blocks)
    # arange step 3. [0, 1, 0, 1, 2, 3, 0, 1]
    arange = np.arange(virtual_batches, dtype=np.int32) - block_offsets
    # also compute reverse arange (i.e. [1, 0, 3, 2, 1, 0, 1, 0])
    rarange = np.repeat(local_blocks, local_blocks) - arange - 1
    # Then we can compute the seqlens_q_local, handling the fact that the
    #  first and last blocks could be partial
    seqlens_q_local = np.repeat(q_seqlens - q_tokens_in_first_block, local_blocks)
    # set the first block since this may be a partial block
    seqlens_q_local[arange == 0] = q_tokens_in_first_block
    # set the remaining blocks
    seqlens_q_local[arange > 0] = np.minimum(
        seqlens_q_local - attn_chunk_size * (arange - 1), attn_chunk_size
    )[arange > 0]

    # convert from q_seqlens to cu_seqlens_q
    cu_seqlens_q_local = np.pad(np.cumsum(seqlens_q_local), (1, 0)).astype(np.int32)

    # compute the seqlens_k_local,
    #  basically a full local attention block for all but the last block in each
    #  batch
    # For our example this will be:
    #   seqlens_k_local = [4, 2, 4, 4, 4, 1, 4, 1]
    seqlens_k_local = np.full(cu_num_blocks[-1], attn_chunk_size, dtype=np.int32)
    seqlens_k_local[cu_num_blocks - 1] = tokens_in_last_block

    k_seqstarts_absolute = np.repeat(seq_lens_np, local_blocks) - (
        rarange * attn_chunk_size + np.repeat(tokens_in_last_block, local_blocks)
    )

    # For the example the local attention blocks start at:
    #                           _b0_  _____b1_____  _b2_
    #   k_seqstarts_absolute = [0, 4, 4, 8, 12, 16, 4, 8]
    block_starts = k_seqstarts_absolute // page_size
    assert attn_chunk_size % page_size == 0, (
        f"attn_chunk_size {attn_chunk_size} is not "
        f"divisible by page_size {page_size}"
    )
    pages_per_local_batch = attn_chunk_size // page_size

    # Create a block_table for the local attention blocks
    # For out example if we have a block-table like (assuming page_size=2):
    #   block_table = [
    #     [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],  < batch 0
    #     [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],  < batch 1
    #     [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],  < batch 2
    #   ]
    # Then for the local batches we would want a block-table like
    #   block_table_local = [
    #     [  0,  1 ], < local-batch 0, (batch 0, starting from k[0])
    #     [  2,  3 ], < local-batch 1, (batch 0, starting from k[4])
    #     [ 12, 13 ], < local-batch 2, (batch 1, starting from k[4])
    #     [ 14, 15 ], < local-batch 3, (batch 1, starting from k[8])
    #     [ 16, 17 ], < local-batch 4, (batch 1, starting from k[12])
    #     [ 18, 19 ], < local-batch 5, (batch 1, starting from k[16])
    #     [ 22, 23 ], < local-batch 6, (batch 2, starting from k[4])
    #     [ 24, 25 ], < local-batch 7, (batch 2, starting from k[8])
    #   ]
    block_indices = np.broadcast_to(
        np.arange(pages_per_local_batch, dtype=np.int32),
        (virtual_batches, pages_per_local_batch),
    ) + np.expand_dims(block_starts, axis=1)
    block_indices = block_indices.flatten().clip(max=block_table.shape[1] - 1)
    batch_indices = np.repeat(
        np.arange(actual_batch_size, dtype=np.int32),
        local_blocks * pages_per_local_batch,
    )
    block_table_local = block_table[batch_indices, block_indices].view(
        virtual_batches, -1
    )

    return seqlens_q_local, cu_seqlens_q_local, seqlens_k_local, block_table_local


# # Qwen2VL
# transformers.models.qwen2_vl.modeling_qwen2_vl.QWEN2_VL_VISION_ATTENTION_CLASSES[
#     "tgi"
# ] = transformers.models.qwen2_vl.modeling_qwen2_vl.QWEN2_VL_VISION_ATTENTION_CLASSES[
#     "eager"
# ]
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
    use_sdpa: Optional[bool] = False,
    local_seqlen: Optional[Seqlen] = None,
    local_block_tables: Optional[torch.Tensor] = None,
    **kwargs,  # This is needed to "absorb" other args passed by Transformers modeling
):
    if module.use_rope:
        seqlen = local_seqlen
        block_tables = local_block_tables
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
        if not use_sdpa:
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
            lengths = cu_seqlen_prefill[1:] - cu_seqlen_prefill[:-1]
            max_length = max(lengths)
            attention_mask = attention_mask[:, :, :, :max_length]
            enable_gqa = query_states.shape[1] != key_states.shape[1]
            # Split tensors using vectorized split
            query_list = torch.split(query_states, lengths.tolist(), dim=0)
            key_list = torch.split(key_states, lengths.tolist(), dim=0)
            value_list = torch.split(value_states, lengths.tolist(), dim=0)

            padded_query = torch.nn.utils.rnn.pad_sequence(query_list, batch_first=True)
            padded_key = torch.nn.utils.rnn.pad_sequence(key_list, batch_first=True)
            padded_value = torch.nn.utils.rnn.pad_sequence(value_list, batch_first=True)

            padded_query = padded_query.transpose(1, 2).contiguous()
            padded_key = padded_key.transpose(1, 2).contiguous()
            padded_value = padded_value.transpose(1, 2).contiguous()

            # Compute attention
            attn_output = F.scaled_dot_product_attention(
                padded_query,
                padded_key,
                padded_value,
                attn_mask=attention_mask,
                scale=softmax_scale,
                enable_gqa=enable_gqa,
            )

            attn_output = attn_output.transpose(
                1, 2
            )  # [batch_size, seq_len, num_heads, head_dim]
            max_seq_len = padded_query.size(2)
            seq_range = torch.arange(max_seq_len, device=padded_query.device).unsqueeze(
                0
            )
            lengths_tensor = torch.tensor(
                lengths, device=padded_query.device
            ).unsqueeze(1)
            mask = seq_range < lengths_tensor  # [batch, max_seq_len]
            attn_output = attn_output[mask]  # [total_seq_len, num_heads, head_dim]

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
            window_size_left=sliding_window,
        )

    attn_output = attn_output.view(-1, num_heads * head_dim)

    return attn_output, None


transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS["tgi"] = tgi_flash_attention_forward


# TODO: implement
# tgi_cross_attention_forward


class TransformersFlashVlmCausalLM(VlmCausalLM):
    def __init__(
        self,
        model_id: str,
        model_class,
        revision: Optional[str] = None,
        quantize: Optional[str] = None,
        speculator: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        default_dtype=torch.float16,
        trust_remote_code: bool = False,
        tokenizer_class=AutoTokenizer,
        processor_class=AutoProcessor,
        processor_kwargs=None,
        kv_cache_dtype: Optional[torch.dtype] = None,
        batch_class=VlmCausalLMBatch,
    ):
        self.batch_class = batch_class
        self.quantize = quantize
        self.process_group, rank, world_size = initialize_torch_distributed()
        self.dtype = dtype

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

        if processor_kwargs is None:
            processor_kwargs = {}

        self.processor = processor_class.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=trust_remote_code,
            **processor_kwargs,
        )

        attn_implementation = {
            "text_config": "tgi",
            "vision_config": "sdpa",
        }

        model = model_class.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=dtype,
            load_in_8bit=quantize == "bitsandbytes",
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
            device_map=device if world_size == 1 else None,
            tp_plan="auto" if world_size > 1 else None,
        )

        torch.distributed.barrier(group=self.process_group)
        self.config = model.config
        config = model.config

        # VLM models define the config we care about in their text_config
        text_config = getattr(model.config, "text_config", None)
        if text_config is not None:
            config = text_config

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

        self.num_layers = config.num_hidden_layers
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        # Some models use GQA and different sizes for o_proj
        # and q_proj, that allows for that.
        if hasattr(config, "head_dim"):
            self.head_size = config.head_dim
        else:
            self.head_size = config.hidden_size // config.num_attention_heads

        # Skip it for models in the exception list
        if config.model_type not in REPLICATED_ATTENTION_MODELS:
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
        )

        # Monkey patch of `self.model.forward` to match `FlashCausalLM`. It avoids duplicating a lot of code
        # We first copy the original model.forward because we still need it in the monkey patch
        self.model.original_forward = self.model.forward
        self.model.forward = self._model_forward
        self.model.get_position_ids = self.get_position_ids

        torch.distributed.barrier(group=self.process_group)

    def get_position_ids(self, input_ids, image_grid_thw, position_ids):
        return position_ids

    def pre_process_inputs(self, **kwargs):
        input_ids = kwargs["input_ids"]
        position_ids = kwargs["position_ids"]
        return {
            "input_ids": input_ids.unsqueeze(0),
            "position_ids": position_ids.unsqueeze(0),
        }

    def post_process_outputs(self, logits, lm_head_indices):
        return logits.squeeze(dim=0)

    @classmethod
    def fallback(
        cls,
        model_id: str,
        model_class,
        revision: Optional[str] = None,
        quantize: Optional[str] = None,
        speculator: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
        batch_class: Optional[type] = VlmCausalLMBatch,
        processor_kwargs: Optional[dict] = None,
    ):
        return cls(
            model_id=model_id,
            model_class=model_class,
            revision=revision,
            quantize=quantize,
            speculator=speculator,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            batch_class=batch_class,
            processor_kwargs=processor_kwargs,
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
        pixel_values: torch.FloatTensor = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        pixel_attention_mask=None,
        image_sizes: Optional[torch.LongTensor] = None,
    ):
        # A value of `None` (i.e. no logit slicing) translates to `0` in Transformers
        logits_to_keep = lm_head_indices if lm_head_indices is not None else 0

        inputs = self.pre_process_inputs(
            input_ids=input_ids,
            position_ids=position_ids,
            cu_seqlen_prefill=cu_seqlen_prefill,
            seqlen=seqlen,
            block_tables=block_tables,
        )
        # This is equivalent to `self.model.forward`, see the monkey patch in __init__
        logits = self.model.original_forward(
            input_ids=inputs["input_ids"],
            position_ids=inputs["position_ids"],
            past_key_values=None,  # we use self.kv_cache instead of transformers cache object
            use_cache=False,  # we use self.kv_cache instead of transformers cache object
            logits_to_keep=logits_to_keep,
            return_dict=True,
            cu_seqlen_prefill=cu_seqlen_prefill,
            kv_cache=kv_cache,
            block_tables=block_tables,
            slots=slots,
            seqlen=seqlen,
            max_s=max_s,
            kv_head_mapping=self.kv_head_mapping,
            kv_scales=self.kv_scales,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            image_sizes=image_sizes,
            image_grid_thw=image_grid_thw,
            attention_mask=inputs.get("attention_mask", None),
            use_sdpa=inputs.get("use_sdpa", False),
            cache_position=inputs.get("cache_position", None),
            local_seqlen=inputs.get("local_seqlen", None),
            local_block_tables=inputs.get("local_block_tables", None),
        ).logits

        logits = self.post_process_outputs(logits, lm_head_indices)

        return logits, None


class TransformersQwen2VlmCausalLM(TransformersFlashVlmCausalLM):
    def get_position_ids(self, input_ids: torch.Tensor, image_grid_thw: torch.Tensor):
        if image_grid_thw is None:
            return (
                torch.arange(input_ids.shape[0], device=input_ids.device)
                .unsqueeze(1)
                .repeat(1, 3)
            )

        spatial_merge_size = self.config.vision_config.spatial_merge_size
        vision_start_token_id = self.config.vision_start_token_id
        vision_end_token_id = self.config.vision_end_token_id
        device = input_ids.device
        dtype = input_ids.dtype
        input_ids_len = input_ids.shape[0]

        vision_starts = torch.where(input_ids == vision_start_token_id)[0]
        vision_ends = torch.where(input_ids == vision_end_token_id)[0]
        vision_segments = torch.stack((vision_starts, vision_ends), dim=1)
        prev_vision_end = torch.cat(
            [torch.zeros(1, device=vision_ends.device, dtype=dtype), vision_ends[:-1]]
        )
        text_lengths_between_vision = vision_segments[:, 0] - prev_vision_end + 1
        vision_widths_max = torch.cat(
            [
                torch.zeros(1, device=image_grid_thw.device, dtype=dtype),
                image_grid_thw[:-1, 2] // spatial_merge_size,
            ]
        )
        vision_segment_lengths = vision_widths_max + text_lengths_between_vision
        vision_segment_lengths = vision_segment_lengths.cumsum(dim=0)
        text_segment_lengths = vision_segment_lengths - text_lengths_between_vision

        # create position ids for each vision segment based on the image grid
        llm_pos_ids_list = []
        for i, _ in enumerate(vision_segments):
            t, h, w = (
                image_grid_thw[i][0],
                image_grid_thw[i][1] // spatial_merge_size,
                image_grid_thw[i][2] // spatial_merge_size,
            )
            t_indices = torch.arange(t, device=device).repeat_interleave(h * w)
            h_indices = torch.arange(h, device=device).repeat_interleave(w).repeat(t)
            w_indices = torch.arange(w, device=device).repeat(t * h)
            image_position_ids = torch.stack([t_indices, h_indices, w_indices], dim=0)

            # offset by the position of the last vision segment
            im = image_position_ids + vision_segment_lengths[i]
            llm_pos_ids_list.append(im)

        # create position ids for each text segment
        text_ranges = [
            torch.arange(seq_len, device=device).view(1, -1).expand(3, -1)
            + text_segment_lengths[i]
            for i, seq_len in enumerate(text_lengths_between_vision)
        ]

        full_llm_pos_ids_list = [
            item for sublist in zip(text_ranges, llm_pos_ids_list) for item in sublist
        ]
        # import ipdb

        # ipdb.set_trace()
        max_s = full_llm_pos_ids_list[-1].max() + 1
        final_text_len = input_ids_len - vision_ends[-1]
        if final_text_len > 0:
            m = torch.arange(final_text_len, device=device).view(1, -1).expand(3, -1)
            full_llm_pos_ids_list.append(m + max_s)

        position_ids = (
            torch.cat(full_llm_pos_ids_list, dim=1).reshape(3, -1).transpose(0, 1)
        )
        return position_ids

    def post_process_outputs(self, logits, lm_head_indices):
        return logits.squeeze(dim=0)[lm_head_indices].unsqueeze(0)

    def pre_process_inputs(self, **kwargs):
        input_ids = kwargs["input_ids"]
        position_ids = kwargs["position_ids"]

        input_ids = input_ids.unsqueeze(0)
        position_ids = position_ids.transpose(0, 1).unsqueeze(1)
        return {"input_ids": input_ids, "position_ids": position_ids}


class TransformersGemma3VlmCausalLM(TransformersFlashVlmCausalLM):
    def get_attention_mask(self, input_ids, cu_seqlen_prefill):
        device = input_ids.device
        dtype = self.dtype
        min_dtype = torch.finfo(dtype).min

        lengths = (cu_seqlen_prefill[1:] - cu_seqlen_prefill[:-1]).tolist()
        batch_size = len(lengths)

        sequence_length = max(lengths)
        target_length = sequence_length
        # Create the padding mask from the computed lengths.
        # pad_mask: [batch, sequence_length] where True indicates valid tokens.
        seq_range = torch.arange(sequence_length, device=device).unsqueeze(0)
        lengths_tensor = torch.tensor(lengths, device=device).unsqueeze(1)
        pad_mask = seq_range < lengths_tensor  # shape: [batch, sequence_length]

        # Build the base causal mask (for non-image tokens):
        causal_mask = torch.tril(
            torch.ones(
                (sequence_length, sequence_length), dtype=torch.bool, device=device
            )
        )
        base_mask = pad_mask.unsqueeze(2) & pad_mask.unsqueeze(
            1
        )  # [batch, sequence_length, sequence_length]
        base_mask = base_mask & causal_mask.unsqueeze(0)  # apply causal constraint

        image_token_mask = (input_ids == self.config.image_token_index).to(
            input_ids.device
        )

        image_token_mask = torch.nn.utils.rnn.pad_sequence(
            torch.split(image_token_mask, lengths), batch_first=True, padding_value=0
        )
        bidirectional_mask = image_token_mask.unsqueeze(2) & image_token_mask.unsqueeze(
            1
        )

        # Combine the causal base mask and the bidirectional mask.
        combined_mask = torch.logical_or(
            base_mask.unsqueeze(1), bidirectional_mask.unsqueeze(1)
        ).to(device)
        # combined_mask now has shape [batch, 1, sequence_length, sequence_length]

        full_attention_mask = torch.zeros(
            (batch_size, 1, sequence_length, target_length),
            device=device,
            dtype=torch.bool,
        )
        full_attention_mask[:, :, :, :sequence_length] = combined_mask

        final_attention_mask = torch.where(full_attention_mask, 0, min_dtype).to(device)

        return final_attention_mask

    def pre_process_inputs(self, **kwargs):
        input_ids = kwargs["input_ids"]
        position_ids = kwargs["position_ids"]
        cu_seqlen_prefill = kwargs["cu_seqlen_prefill"]

        inputs = {
            "input_ids": input_ids.unsqueeze(0),
            "position_ids": position_ids.unsqueeze(0),
        }

        if cu_seqlen_prefill is not None:
            attention_mask = self.get_attention_mask(
                input_ids.squeeze(0), cu_seqlen_prefill
            )
            inputs["attention_mask"] = attention_mask
            inputs["use_sdpa"] = True

        return inputs


class TransformersLlama4VlmCausalLM(TransformersFlashVlmCausalLM):
    def pre_process_inputs(self, **kwargs):
        input_ids = kwargs["input_ids"]
        position_ids = kwargs["position_ids"]
        seqlen = kwargs["seqlen"]
        block_tables = kwargs["block_tables"]

        inputs = super().pre_process_inputs(**kwargs)
        inputs["cache_position"] = position_ids
        inputs["attention_mask"] = torch.zeros((1, 1, 1, 1), device=input_ids.device)
        from loguru import logger

        logger.info(f"input_ids: {input_ids.shape}, position_ids: {position_ids.shape}")
        cu_seqlen_k = seqlen.cu_seqlen_k
        cu_seqlen_q = seqlen.cu_seqlen_q
        seq_lens_np = cu_seqlen_k[1:] - cu_seqlen_k[:-1]
        (
            seqlens_q_local_np,
            virt_q_cu_seqlens_np,
            virt_k_seqlens_np,
            virt_block_table,
        ) = make_local_attention_virtual_batches(
            self.model.config.text_config.attention_chunk_size,
            cu_seqlen_q.cpu().numpy(),
            seq_lens_np.cpu().numpy(),
            block_tables,
            BLOCK_SIZE,
        )
        local_seqlen = Seqlen(
            input_lengths=torch.from_numpy(virt_k_seqlens_np).to(
                input_ids.device, non_blocking=True
            ),
            cache_lengths=torch.zeros(virt_k_seqlens_np.shape).to(
                input_ids.device, non_blocking=True
            ),
            cu_seqlen_q=torch.from_numpy(virt_q_cu_seqlens_np).to(
                input_ids.device, non_blocking=True
            ),
            max_q=int(seqlens_q_local_np.max()),
            max_k=int(virt_k_seqlens_np.max()),
        )

        inputs["local_seqlen"] = local_seqlen
        inputs["local_block_tables"] = virt_block_table

        return inputs
