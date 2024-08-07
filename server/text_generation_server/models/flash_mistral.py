import math
import torch
import torch.distributed

import numpy as np

from dataclasses import dataclass
from opentelemetry import trace
from transformers import PreTrainedTokenizerBase, AutoTokenizer, AutoConfig
from typing import Optional, Tuple, Type

from text_generation_server.pb import generate_pb2
from text_generation_server.models.flash_causal_lm import FlashCausalLM
from text_generation_server.models.flash_causal_lm import FlashCausalLMBatch, BLOCK_SIZE
from text_generation_server.models.cache_manager import (
    get_cache_manager,
)
from text_generation_server.models.custom_modeling.flash_mistral_modeling import (
    FlashMistralForCausalLM,
    MistralConfig,
)
from text_generation_server.utils.speculate import get_speculate
from text_generation_server.utils import (
    initialize_torch_distributed,
    weight_files,
    Weights,
    HeterogeneousNextTokenChooser,
    StoppingCriteria,
)

tracer = trace.get_tracer(__name__)

# Will be set in init
SLIDING_WINDOW: Optional[int] = None
SLIDING_WINDOW_BLOCKS: Optional[int] = None
from text_generation_server.utils.import_utils import IS_XPU_SYSTEM

MEM_POOL = torch.cuda.graph_pool_handle() if torch.cuda.is_available() else None


def set_sliding_window(sliding_window: int, sliding_window_blocks: int):
    global SLIDING_WINDOW
    global SLIDING_WINDOW_BLOCKS
    SLIDING_WINDOW = sliding_window
    SLIDING_WINDOW_BLOCKS = sliding_window_blocks


def get_sliding_windows() -> Tuple[int, int]:
    global SLIDING_WINDOW
    global SLIDING_WINDOW_BLOCKS
    return SLIDING_WINDOW, SLIDING_WINDOW_BLOCKS


# Adds windowing logic to FlashCausalLMBatch
@dataclass
class FlashMistralBatch(FlashCausalLMBatch):
    # Prefill cache indices is used to slice into the kv tensor before caching it into the paged attention buffers
    # as we only keep SLIDING_WINDOW values instead of the whole tensor
    prefill_cache_indices: Optional[torch.Tensor] = None

    @classmethod
    def from_pb(
        cls,
        pb: generate_pb2.Batch,
        tokenizer: PreTrainedTokenizerBase,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "FlashCausalLMBatch":
        batch_tokenized_inputs = cls.batch_tokenized_inputs(pb.requests, tokenizer)
        return cls.from_tokenized(pb, tokenizer, batch_tokenized_inputs, dtype, device)

    @classmethod
    def from_tokenized(
        cls,
        pb: generate_pb2.Batch,
        tokenizer: PreTrainedTokenizerBase,
        batch_tokenized_inputs,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "FlashCausalLMBatch":
        sliding_window, sliding_window_blocks = get_sliding_windows()

        position_ids = []
        cu_seqlen_prefill = [0]
        needed_blocks_slots = []
        start_slots = []
        slot_indices = []
        prefill_cache_indices = []

        input_lengths = []
        prefix_offsets = []
        read_offsets = []
        all_input_ids = []
        requests_idx_mapping = {}

        all_prefill_logprobs = True
        no_prefill_logprobs = True
        prefill_head_indices = []
        prefill_next_token_indices = []
        prefill_cu_outlens = [0]

        next_token_chooser_parameters = []
        stopping_criterias = []
        top_n_tokens = []

        # Cumulative length
        cumulative_length = 0
        cumulative_max_length = 0
        prefill_out_cumulative_length = 0

        blocks = 0
        max_seqlen = 0
        max_length = 0
        max_blocks = 0

        # Parse batch
        for i, (r, tokenized_input) in enumerate(
            zip(pb.requests, batch_tokenized_inputs)
        ):
            # request id -> idx in list mapping
            requests_idx_mapping[r.id] = i

            tokenized_input = tokenized_input[-r.truncate :]

            input_length = len(tokenized_input)
            input_lengths.append(input_length)

            prefix_offsets.append(input_length - 5)
            read_offsets.append(input_length)

            all_input_ids.append(tokenized_input)

            # Position ids
            request_position_ids = torch.arange(0, input_length, dtype=torch.int32)
            position_ids.append(request_position_ids)

            # Add cumulative lengths of all previous inputs
            cu_seqlen_prefill.append(cumulative_length + input_length)

            next_token_chooser_parameters.append(r.parameters)

            stopping_criteria = StoppingCriteria.from_pb(
                r.stopping_parameters, tokenizer
            )
            max_new_tokens = stopping_criteria.max_new_tokens
            stopping_criterias.append(stopping_criteria)
            top_n_tokens.append(r.top_n_tokens)

            # Paged attention
            # Remove one as the first token des not have a past
            speculative_length = get_speculate()
            total_tokens = input_length + max_new_tokens - 1 + speculative_length

            # Needed blocks can not go over SLIDING_WINDOW_BLOCKS
            needed_blocks = math.ceil(total_tokens / BLOCK_SIZE)
            if sliding_window_blocks is not None:
                needed_blocks = min(needed_blocks, sliding_window_blocks)
            blocks += needed_blocks

            needed_blocks_slots.append((needed_blocks, total_tokens))
            start_slots.append(cumulative_max_length)

            request_slot_indices = torch.arange(
                cumulative_max_length,
                cumulative_max_length + input_length,
                dtype=torch.int64,
            )
            slot_indices.append(request_slot_indices)

            # Create tensor to slice into the kv tensor in prefill
            if sliding_window is not None:
                request_prefill_cache_indices = torch.arange(
                    cumulative_length + max(0, input_length - sliding_window),
                    cumulative_length + input_length,
                    dtype=torch.int64,
                )
                prefill_cache_indices.append(request_prefill_cache_indices)

            all_prefill_logprobs = all_prefill_logprobs and r.prefill_logprobs
            no_prefill_logprobs = no_prefill_logprobs and not r.prefill_logprobs

            if r.prefill_logprobs:
                prefill_head_indices.append(request_position_ids + cumulative_length)
                prefill_next_token_indices.append(
                    prefill_out_cumulative_length + input_length - 1
                )
                prefill_cu_outlens.append(prefill_out_cumulative_length + input_length)
                prefill_out_cumulative_length += input_length
            else:
                prefill_head_indices.append(
                    torch.tensor(
                        [cumulative_length + input_length - 1], dtype=torch.int32
                    )
                )
                prefill_next_token_indices.append(prefill_out_cumulative_length)
                prefill_cu_outlens.append(prefill_out_cumulative_length + 1)
                prefill_out_cumulative_length += 1

            # Update
            cumulative_length += input_length
            cumulative_max_length += total_tokens
            max_seqlen = max(max_seqlen, input_length)
            max_blocks = max(max_blocks, needed_blocks)
            max_length = max(
                max_length, input_length + max_new_tokens + speculative_length
            )

        next_token_chooser = HeterogeneousNextTokenChooser.from_pb(
            next_token_chooser_parameters, dtype, device, tokenizer
        )
        start_slots = torch.tensor(start_slots, dtype=torch.int64)

        # Padded all_input_ids_tensor
        all_input_ids_tensor = np.zeros(
            (len(all_input_ids), max_length), dtype=np.int64
        )
        for i, input_ids in enumerate(all_input_ids):
            all_input_ids_tensor[i, : len(input_ids)] = input_ids

        # Create tensors on device
        all_input_ids_tensor = torch.tensor(
            all_input_ids_tensor, dtype=torch.int64, device=device
        )

        if len(pb.requests) > 1:
            input_ids = np.concatenate(all_input_ids, dtype=np.int64)
            position_ids = torch.cat(position_ids)
            slot_indices = torch.cat(slot_indices)
            if sliding_window is not None:
                prefill_cache_indices = torch.cat(prefill_cache_indices)
        else:
            input_ids = all_input_ids[0]
            position_ids = position_ids[0]
            slot_indices = slot_indices[0]
            if sliding_window is not None:
                prefill_cache_indices = prefill_cache_indices[0]

        cu_seqlen_prefill = torch.tensor(
            cu_seqlen_prefill, device=device, dtype=torch.int32
        )

        position_ids = position_ids.to(device)
        slot_indices = slot_indices.to(device)
        prefill_cache_indices = (
            prefill_cache_indices.to(device) if sliding_window is not None else None
        )
        input_ids = torch.tensor(input_ids, dtype=torch.int64, device=device)
        input_lengths_tensor = torch.tensor(
            input_lengths, dtype=torch.int32, device=device
        )

        if all_prefill_logprobs:
            prefill_head_indices = None
            prefill_next_token_indices = cu_seqlen_prefill[1:] - 1
        elif no_prefill_logprobs:
            prefill_head_indices = cu_seqlen_prefill[1:] - 1
            prefill_next_token_indices = None
        else:
            prefill_head_indices = torch.tensor(
                torch.cat(prefill_head_indices), dtype=torch.int64, device=device
            )
            prefill_next_token_indices = torch.tensor(
                prefill_next_token_indices, dtype=torch.int64, device=device
            )
        top_n_tokens_tensor = torch.tensor(
            top_n_tokens, device=device, dtype=torch.int64
        )

        return cls(
            batch_id=pb.id,
            requests=pb.requests,
            requests_idx_mapping=requests_idx_mapping,
            input_ids=input_ids,
            position_ids=position_ids,
            cu_seqlen_prefill=cu_seqlen_prefill,
            start_slots=start_slots,
            slot_indices=slot_indices,
            needed_blocks_slots=needed_blocks_slots,
            block_tables=None,
            block_tables_tensor=None,
            slots=None,
            max_seqlen=max_seqlen,
            prefill_head_indices=prefill_head_indices,
            prefill_next_token_indices=prefill_next_token_indices,
            prefill_cu_outlens=prefill_cu_outlens,
            input_lengths=input_lengths,
            input_lengths_tensor=input_lengths_tensor,
            prefix_offsets=prefix_offsets,
            read_offsets=read_offsets,
            all_input_ids=all_input_ids,
            all_input_ids_tensor=all_input_ids_tensor,
            next_token_chooser=next_token_chooser,
            stopping_criterias=stopping_criterias,
            top_n_tokens=top_n_tokens,
            top_n_tokens_tensor=top_n_tokens_tensor,
            blocks=blocks,
            max_blocks=max_blocks,
            prefill_cache_indices=prefill_cache_indices,
            speculative_ids=None,
        )


class BaseFlashMistral(FlashCausalLM):
    def __init__(
        self,
        model_cls,
        model_id: str,
        config_cls=AutoConfig,
        revision: Optional[str] = None,
        quantize: Optional[str] = None,
        use_medusa: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
        tokenizer_class=AutoTokenizer,
    ):
        self.process_group, rank, world_size = initialize_torch_distributed()
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{rank}")
            dtype = torch.float16 if dtype is None else dtype
        elif IS_XPU_SYSTEM:
            device = torch.device(f"xpu:{rank}")
            dtype = torch.float16 if dtype is None else dtype
        else:
            raise NotImplementedError("FlashMistral is only available on GPU")

        tokenizer = tokenizer_class.from_pretrained(
            model_id,
            revision=revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=trust_remote_code,
        )

        config = config_cls.from_pretrained(
            model_id, revision=revision, trust_remote_code=trust_remote_code
        )
        config.quantize = quantize
        config.use_medusa = use_medusa

        # Set context windows
        if getattr(config, "sliding_window", None) is not None:
            set_sliding_window(
                config.sliding_window, math.ceil(config.sliding_window / BLOCK_SIZE)
            )
        else:
            config.sliding_window = None

        torch.distributed.barrier(group=self.process_group)

        filenames = weight_files(model_id, revision=revision, extension=".safetensors")
        weights = Weights(filenames, device, dtype, process_group=self.process_group)
        if config.quantize in ["gptq", "awq"]:
            weights._set_gptq_params(model_id, revision)

        prefix = ""
        model = model_cls(prefix, config, weights)

        self.cuda_graphs = {}

        torch.distributed.barrier(group=self.process_group)
        num_layers, num_kv_heads, head_size = self.get_layer_config(model)
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype=dtype,
            device=device,
            rank=rank,
            world_size=world_size,
            sliding_window=config.sliding_window,
        )

    def get_layer_config(self, model) -> Tuple[int, int, int]:
        return (
            len(model.model.layers),
            model.model.num_key_value_heads,
            model.model.head_size,
        )

    def max_past(self) -> int:
        return self.model.max_past

    @property
    def batch_type(self) -> Type[FlashMistralBatch]:
        return FlashMistralBatch

    def cuda_graph_warmup(self, bs: int, max_s: int, max_bt: int):
        input_ids = torch.zeros(bs, dtype=torch.int64, device=self.device)
        position_ids = torch.zeros(bs, dtype=torch.int32, device=self.device)
        slots = torch.arange(bs, dtype=torch.int64, device=self.device)
        input_lengths = torch.ones(bs, dtype=torch.int32, device=self.device) * max_s
        block_tables = (
            torch.arange(max_bt, dtype=torch.int32, device=self.device)
            .repeat(bs)
            .reshape((bs, max_bt))
        )
        kv_cache = get_cache_manager().kv_cache

        self.cuda_graphs[bs] = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "kv_cache": kv_cache,
            "block_tables": block_tables,
            "slots": slots,
            "input_lengths": input_lengths,
        }
        graph = torch.cuda.CUDAGraph()
        self.cuda_graphs[bs]["graph"] = graph

        torch.cuda.synchronize()
        # Run once outside to warmup
        self.model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            cu_seqlen_prefill=None,
            kv_cache=kv_cache,
            block_tables=block_tables,
            slots=slots,
            input_lengths=input_lengths,
            max_s=max_s,
            prefill_cache_indices=None,
            lm_head_indices=None,
        )
        torch.cuda.synchronize()

        with torch.cuda.graph(graph, pool=MEM_POOL):
            logits, speculative_logits = self.model.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                cu_seqlen_prefill=None,
                kv_cache=kv_cache,
                block_tables=block_tables,
                slots=slots,
                input_lengths=input_lengths,
                max_s=max_s,
                prefill_cache_indices=None,
                lm_head_indices=None,
            )
            self.cuda_graphs[bs]["logits"] = logits
            self.cuda_graphs[bs]["speculative_logits"] = speculative_logits
        torch.cuda.synchronize()

    def forward(
        self, batch: FlashMistralBatch
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Model Forward
        if batch.speculative_ids is not None:
            input_ids = batch.input_ids
            position_ids = batch.position_ids
            cu_seqlen_prefill = batch.cu_seqlen_prefill
            kv_cache = get_cache_manager().kv_cache
            block_tables = batch.block_tables_tensor
            slots = batch.slots[batch.slot_indices]
            input_lengths = batch.input_lengths_tensor
            max_s = batch.max_seqlen
            lm_head_indices = batch.prefill_head_indices

            speculative_ids = batch.speculative_ids

            B, speculative_length = speculative_ids.shape
            new_length = speculative_length + 1
            new_input_ids = torch.cat(
                [input_ids.unsqueeze(-1), speculative_ids], dim=1
            ).reshape(-1)
            arange = torch.arange(new_length, device=position_ids.device).unsqueeze(0)
            arange_int = arange.to(dtype=torch.int32)
            new_position_ids = (
                position_ids.unsqueeze(-1).expand(B, new_length) + arange
            ).view(-1)
            slots = (slots.unsqueeze(-1).expand(B, new_length) + arange_int).view(-1)
            input_lengths = (
                input_lengths.unsqueeze(-1).expand(B, new_length) + arange_int
            ).view(-1)

            # Add Copy the block tables for all members
            block_tables = (
                block_tables.unsqueeze(1)
                .expand(B, new_length, -1)
                .reshape(B * new_length, -1)
                .contiguous()
            )
            max_s = max_s + speculative_length

            input_ids = new_input_ids
            position_ids = new_position_ids
        else:
            input_ids = batch.input_ids
            position_ids = batch.position_ids
            cu_seqlen_prefill = batch.cu_seqlen_prefill
            kv_cache = get_cache_manager().kv_cache
            block_tables = batch.block_tables_tensor
            slots = batch.slots[batch.slot_indices]
            input_lengths = batch.input_lengths_tensor
            max_s = batch.max_seqlen
            lm_head_indices = batch.prefill_head_indices

        if cu_seqlen_prefill is None and self.max_past() is not None:
            # In decode, not prefill, we're actually overwriting the KV-cache
            # in a circular buffer mode.
            # This makes sure the max_s for the decode pass is correct.
            max_s = min(self.max_past(), max_s)

        bs = input_ids.shape[0]
        padded_bs = bs
        if bs == 3:
            padded_bs = 4
        elif 3 < bs <= 8:
            padded_bs = 8
        elif bs > 8:
            padded_bs = (bs + 7) // 8 * 8

        # Try to find an associated cuda graph
        cuda_graph = self.cuda_graphs.get(padded_bs, None)

        if cu_seqlen_prefill is not None or cuda_graph is None:
            logits, speculative_logits = self.model.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                cu_seqlen_prefill=cu_seqlen_prefill,
                kv_cache=kv_cache,
                block_tables=block_tables,
                slots=slots,
                input_lengths=input_lengths,
                max_s=max_s,
                prefill_cache_indices=batch.prefill_cache_indices,
                lm_head_indices=lm_head_indices,
            )
            if batch.prefill_cache_indices is not None:
                batch.prefill_cache_indices = None
            return logits, speculative_logits

        # Copy inputs to the static inputs of the cuda graph
        # Static inputs are potentially padded
        cuda_graph["input_ids"][: input_ids.shape[0]] = input_ids
        cuda_graph["position_ids"][: position_ids.shape[0]] = position_ids
        cuda_graph["block_tables"][
            : block_tables.shape[0], : block_tables.shape[1]
        ] = block_tables
        cuda_graph["slots"].fill_(-1)
        cuda_graph["slots"][: slots.shape[0]] = slots
        cuda_graph["input_lengths"].zero_()
        cuda_graph["input_lengths"][: input_lengths.shape[0]] = input_lengths

        # Replay the graph
        cuda_graph["graph"].replay()

        # Slice output to the correct shape
        speculative_logits = (
            cuda_graph["speculative_logits"][:bs]
            if cuda_graph["speculative_logits"] is not None
            else None
        )
        logits = cuda_graph["logits"][:bs]
        return logits, speculative_logits


class FlashMistral(BaseFlashMistral):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        quantize: Optional[str] = None,
        use_medusa: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
    ):
        super(FlashMistral, self).__init__(
            config_cls=MistralConfig,
            model_cls=FlashMistralForCausalLM,
            model_id=model_id,
            revision=revision,
            quantize=quantize,
            use_medusa=use_medusa,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
