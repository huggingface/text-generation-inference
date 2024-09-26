# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.

import bisect
from dataclasses import dataclass
from functools import wraps
import itertools
import math
import os
import tempfile
import time
import copy
from typing import Dict, List, Optional, Tuple, Type

import torch
import torch._dynamo
from loguru import logger
from opentelemetry import trace

import text_generation_server.habana_quantization_env as hq_env
import habana_frameworks.torch as htorch
from optimum.habana.utils import HabanaProfile
from optimum.habana.transformers.generation import MODELS_OPTIMIZED_WITH_STATIC_SHAPES
from text_generation_server.utils.chunks import concat_text_chunks
from optimum.habana.checkpoint_utils import (
    get_repo_root,
    model_on_meta,
    write_checkpoints_json,
)
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizerBase,
    AutoConfig,
)

from text_generation_server.utils.tokens import batch_top_tokens
from text_generation_server.models import Model
from text_generation_server.models.types import (
    Batch,
    Tokens,
    Generation,
    GeneratedText,
)
from text_generation_server.pb import generate_pb2
from text_generation_server.utils import (
    HeterogeneousNextTokenChooser,
    StoppingCriteria,
    make_tokenizer_optional,
    is_tokenizer_transparent,
    pad_next_token_chooser_parameters,
)
from optimum.habana.utils import get_hpu_memory_stats
from text_generation_server.utils.debug import dbg_trace
from text_generation_server.utils.speculate import get_speculate

tracer = trace.get_tracer(__name__)

# MAX_TOTAL_TOKENS = int(os.environ.get('MAX_TOTAL_TOKENS', 2048))
# BATCH_BUCKET_SIZE = int(os.environ.get('BATCH_BUCKET_SIZE', 8))
# PAD_SEQUENCE_TO_MULTIPLE_OF = int(os.environ.get('PAD_SEQUENCE_TO_MULTIPLE_OF', 128))
# PREFILL_BATCH_BUCKET_SIZE = int(os.environ.get('PREFILL_BATCH_BUCKET_SIZE', 4))
# CHUNK_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
# LAZY_MODE = int(os.environ.get('PT_HPU_LAZY_MODE', 1))
MAX_TOTAL_TOKENS = int(os.environ.get('MAX_TOTAL_TOKENS', 8192))
MAX_BATCH_TOTAL_TOKENS = int(os.environ.get('MAX_BATCH_TOTAL_TOKENS', 65536))
PAD_SEQUENCE_TO_MULTIPLE_OF = int(os.environ.get('PAD_SEQUENCE_TO_MULTIPLE_OF', 256))
CHUNK_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
LAZY_MODE = int(os.environ.get('PT_HPU_LAZY_MODE', 1))


PREFILL_WARMUP_BATCH_SIZE_LIST = []
PREFILL_WARMUP_SEQLEN_LIST = []
DECODE_WARMUP_BATCH_SIZE_LIST = []


def torch_compile_for_eager(func):
    if LAZY_MODE == 1:
        return func
    return torch.compile(func, backend="hpu_backend", options={"keep_input_mutations": True})


def round_up(warmup_list:list, num) :
    i = 0
    for i in warmup_list:
        if num <= i :
            break
    return i


def to_tensor_indices(indices, device):
    return torch.tensor(indices, dtype=torch.long, device=device)


def calculate_chunks(offset):
    result = []
    while offset != 0:
        sign = 1 if offset > 0 else -1
        best_chunk = min((abs(offset - sign * c), sign * c) for c in CHUNK_SIZES)[1]
        result.append(best_chunk)
        offset = offset - best_chunk
    return result


def biggest_single_chunk(offset):
    if offset != 0:
        idx = bisect.bisect(CHUNK_SIZES, abs(offset))
        return int(math.copysign(CHUNK_SIZES[idx - 1], offset))
    else:
        return 0


@torch_compile_for_eager
def grouped_pad(tensor_groups, dims, values):
    grouped_result = []
    for tensors, dim, value in zip(tensor_groups, dims, values):
        padding = MAX_TOTAL_TOKENS - tensors[0].size(dim) if dim is not None else 0
        if padding > 0:
            assert dim in [-1, -2], f'Only dims -1 and -2 are supported! {dim}'
            pad_shape = (0, 0, 0, padding) if dim == -2 else (0, padding)
            result = [torch.nn.functional.pad(t, pad_shape, value=value) for t in tensors]
        else:
            result = [t for t in tensors]
        grouped_result.append(result)
        htorch.core.mark_step()
    return grouped_result


@torch_compile_for_eager
def roll(tensor, chunk, dim, merge_graphs):
    if dim is None:
        return tensor
    tensor = torch.roll(tensor, chunk, dim)
    if not merge_graphs:
        htorch.core.mark_step()
    return tensor


def grouped_roll(tensor_groups, chunk, dims, merge_graphs):
    tensor_groups = [[roll(t, chunk, dim, merge_graphs) for t in tensors] for tensors, dim in zip(tensor_groups, dims)]
    if merge_graphs:
        htorch.core.mark_step()
    return tensor_groups


@torch_compile_for_eager
def grouped_shift(tensor_groups, dims, offset, merge_graphs):
    chunks = calculate_chunks(offset)
    for c in chunks:
        tensor_groups = grouped_roll(tensor_groups, c, dims, merge_graphs)
    return tensor_groups


def move(dst_tensors, dst_indices, src_tensors):
    bs_dim = 0
    num_indices = dst_indices.size(0)
    for i, (dst_t, src_t) in enumerate(zip(dst_tensors, src_tensors)):
        if src_t.size(bs_dim) != num_indices:
            src_t = torch.narrow(src_t, bs_dim, 0, num_indices)
        dst_t.index_copy_(bs_dim, dst_indices, src_t)
    htorch.core.mark_step()


def grouped_move(dst_tensor_groups, dst_indices, src_tensor_groups):
    for dst_tensors, src_tensors in zip(dst_tensor_groups, src_tensor_groups):
        move(dst_tensors, dst_indices, src_tensors)


@torch_compile_for_eager
def extend_tensor(tensor, padding, dim):
    result = torch.cat([tensor, padding], dim=dim)
    htorch.core.mark_step()
    return result


@torch_compile_for_eager
def extend_batch(tensors, target_bs, dim):
    diff = target_bs - tensors[0].size(dim)
    # TODO: add support for shrinking bs
    if diff <= 0:
        return tensors
    shape = list(tensors[0].shape)
    shape[dim] = diff
    padding = torch.empty(shape, device=tensors[0].device, dtype=tensors[0].dtype)
    tensors = [extend_tensor(t, padding, dim) for t in tensors]
    return tensors


def grouped_extend_batch(tensor_groups, target_bs, bs_dims):
    tensor_groups = [extend_batch(tensors, target_bs, dim) for tensors, dim in zip(tensor_groups, bs_dims)]
    return tensor_groups


@torch_compile_for_eager
def merge(tensor_group):
    tensor_group = [torch.stack(tensor_group)]
    htorch.core.mark_step()
    return tensor_group


@torch_compile_for_eager
def split(tensor_group, clone_data):
    tensor_group = [t.squeeze(0) for t in torch.split(tensor_group[0], 1)]
    if clone_data:
        tensor_group = [t.clone() for t in tensor_group]
    htorch.core.mark_step()
    return tensor_group


def remove_kv_cache_from_output(module):
    orig_fwd = module.forward

    @wraps(orig_fwd)
    def forward(*args, **kwargs):
        if kwargs["past_key_values"] is not None:
            kwargs["return_dict"] = False
            output = orig_fwd(*args, **kwargs)
            first_value, second_value, *_ = output
            if first_value.nelement() < 2:
                return second_value
            else:
                return first_value
        else:
            kwargs["return_dict"] = True
            return orig_fwd(*args, **kwargs)

    module.forward = forward
    return module


@dataclass
class CausalLMRequest:
    idx: int
    data: generate_pb2.Request
    input_length: int
    prefix_offset: int
    read_offset: int
    stopping_criteria: StoppingCriteria

    all_input_ids: torch.Tensor

    @classmethod
    def from_pb(cls, idx: int, data: generate_pb2.Request, tokenizer: PreTrainedTokenizerBase):
        return cls(
            idx=idx,
            data=data,
            input_length=None,
            prefix_offset=None,
            read_offset=None,
            stopping_criteria=StoppingCriteria.from_pb(data.stopping_parameters, tokenizer),
            all_input_ids=None,)

    def update_idx(self, new_idx):
        prev = self.idx
        self.idx = new_idx
        return (new_idx, prev)


@dataclass
class CausalLMBatch(Batch):
    batch_id: int
    requests: List[CausalLMRequest]

    # Decoder values
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    past_key_values: Optional[List[Tuple]]
    merged_kv_cache: bool

    # Lengths of all generations present in the batch
    input_length: int

    # Generation helpers
    next_token_chooser: HeterogeneousNextTokenChooser
    top_n_tokens: List[int]
    top_n_tokens_tensor: torch.Tensor

    input_length: int

    # Past metadata
    logits = None
    past = None

    keys_head_dim_last: bool = True

    def to_pb(self) -> generate_pb2.CachedBatch:
        return generate_pb2.CachedBatch(
            id=self.batch_id,
            request_ids=[r.data.id for r in self.requests],
            size=len(self),
            max_tokens=self.max_tokens,
        )

    def detach_kv_cache(self):
        past_keys = [past[0] for past in self.past_key_values]
        past_values = [past[1] for past in self.past_key_values]
        del self.past_key_values
        return past_keys, past_values

    def attach_kv_cache(self, past_keys, past_values):
        # TODO: Add support for models that don't store kv_cache in a list
        self.past_key_values = list(zip(past_keys, past_values))

    def merge_kv_cache_if_needed(self, target_bs, offset):
        pad_needed = self.seq_length < MAX_TOTAL_TOKENS
        shift_needed = offset != 0
        expand_needed = target_bs > self.batch_size
        # Very simple heuristic to determine whether we should merge tensors
        # this needs tuning for other models/scenarios
        small_bs = len(self.past_key_values) > self.batch_size
        if not self.merged_kv_cache and small_bs and (pad_needed or shift_needed or expand_needed):
            past_keys, past_values = self.detach_kv_cache()
            past_keys = merge(past_keys)
            past_values = merge(past_values)
            self.attach_kv_cache(past_keys, past_values)
            self.merged_kv_cache = True

    def split_kv_cache_if_needed(self, clone_data):
        if self.merged_kv_cache:
            past_keys, past_values = self.detach_kv_cache()
            past_keys = split(past_keys, clone_data)
            past_values = split(past_values, clone_data)
            self.attach_kv_cache(past_keys, past_values)
            self.merged_kv_cache = False

    def get_tensor_groups(self):
        past_keys, past_values = self.detach_kv_cache()
        seq_dim = -1
        key_dim = -2 if self.keys_head_dim_last else -1
        value_dim = -2
        tensors = [[self.input_ids], [self.attention_mask], [self.position_ids], past_keys, past_values]
        # We don't need to align position_ids
        seq_dims = [seq_dim, seq_dim, None, key_dim, value_dim]
        bs_dims = [0, 0, 0] + ([1, 1] if self.merged_kv_cache else [0, 0])
        return tensors, seq_dims, bs_dims

    def set_tensor_groups(self, tensors):
        self.input_ids = tensors.pop(0)[0]
        self.attention_mask = tensors.pop(0)[0]
        self.position_ids = tensors.pop(0)[0]
        past_keys = tensors.pop(0)
        past_values = tensors.pop(0)
        self.attach_kv_cache(past_keys, past_values)

    def realign(self, target_bs, offset, pad_token_id):
        tensors, seq_dims, _ = self.get_tensor_groups()
        tensors = grouped_pad(tensors, seq_dims, [pad_token_id, 0, 0, 0, 0])
        tensors = grouped_shift(tensors, seq_dims, offset, self.merged_kv_cache)
        self.set_tensor_groups(tensors)

    def expand_bs(self, target_bs):
        tensors, _, bs_dims = self.get_tensor_groups()
        tensors = grouped_extend_batch(tensors, target_bs, bs_dims)
        self.set_tensor_groups(tensors)

    def used_indices(self):
        return [req.idx for req in self.requests]

    def update_indices(self, new_indices):
        for req, new_idx in zip(self.requests, new_indices):
            req.idx = new_idx
        return self.used_indices()

    def free_indices_generator(self):
        used = set(req.idx for req in self.requests)
        return (i for i in range(self.batch_size) if i not in used)

    def move_data(self, src_batches):
        dst_tensors, _, dst_dims = self.get_tensor_groups()
        free_indices_gen = self.free_indices_generator()
        for src_b in src_batches:
            dst_indices = to_tensor_indices(src_b.update_indices(free_indices_gen), self.input_ids.device)
            src_tensors, _, src_dims = src_b.get_tensor_groups()
            grouped_move(dst_tensors, dst_indices, src_tensors)
        self.set_tensor_groups(dst_tensors)

    @classmethod
    def recombine(cls, batches: List["CausalLMBatch"], pad_token_id: int, is_warmup: bool =False) -> "CausalLMBatch":
        if not all(b.past_key_values is not None for b in batches):
            raise ValueError("KV cache not allocated! Cannot recombine before prefill!")

        total_requests = sum(len(b) for b in batches)
        new_bs = total_requests
        if is_warmup is False :
            new_bs = round_up(DECODE_WARMUP_BATCH_SIZE_LIST, total_requests)

        batch_id = batches[0].batch_id
        device = batches[0].input_ids.device

        input_lengths = [b.input_length for b in batches]
        max_input_length = max(input_lengths)
        offsets = [max_input_length - b.input_length for b in batches]

        cur_padding = [b.right_padding for b in batches]
        # For prefill there is a space allocated only for first token
        # Need to add padding to the max total tokens before first decode

        moves_needed = [total_requests - len(b) if b.batch_size == new_bs else total_requests for b in batches]
        dst_batch_idx = min(enumerate(moves_needed), key=lambda idx_val: idx_val[1])[0]
        reshape = (batches[dst_batch_idx].batch_size < new_bs)

        # TODO: Add support for changing max seq len, i.e. due to output length bucketing
        # FIXME: max_seq_len for non optimized code
        if len(batches) > 1:
            scenario = 'CONCAT'
        elif reshape:
            scenario = 'RESHAPE'
        elif cur_padding[dst_batch_idx] <= 0:
            scenario = 'SHIFT'
            offsets = [biggest_single_chunk(b.max_input_length - max_input_length) for b in batches]
            max_input_length = max_input_length + offsets[dst_batch_idx]
        else:
            # Nothing to do
            return batches[0]

        dbg_trace(
            scenario, f'bs:{[b.batch_size for b in batches]}->{new_bs}'
                      f' reqs:{[len(b) for b in batches]}'
                      f' offsets:{offsets}'
                      f' input_lengths:{input_lengths}'
                      f' cur_padding:{cur_padding}'
                      f' dst_batch:{dst_batch_idx}')

        grouped_requests = [[req for req in batch.requests] for batch in batches]
        flat_requests = list(itertools.chain(*grouped_requests))

        for i in range(len(batches)):
            target_bs = new_bs if i == dst_batch_idx else batches[i].batch_size
            batches[i].merge_kv_cache_if_needed(target_bs, offsets[i])
            batches[i].realign(target_bs, offsets[i], pad_token_id)
            batches[i].split_kv_cache_if_needed(i == dst_batch_idx)
        batches[dst_batch_idx].expand_bs(new_bs)
        batches[dst_batch_idx].move_data([batches[i] for i in range(len(batches)) if i != dst_batch_idx])

        top_n_tokens = [r.data.top_n_tokens for r in flat_requests]
        top_n_tokens_tensor = torch.tensor(top_n_tokens, device=device, dtype=torch.int64)

        parameters = [r.data.parameters for r in flat_requests]
        # append the dummy parameters for dummy requests
        batch_size = batches[dst_batch_idx].batch_size
        parameters = pad_next_token_chooser_parameters(parameters, batch_size)

        # update past grammar states
        fsm_grammar_states = [0] * batch_size
        for batch in batches:
            for i, req in enumerate(batch.requests):
                fsm_grammar_states[req.idx] = batch.next_token_chooser.fsm_grammar_states[i]

        next_token_chooser = HeterogeneousNextTokenChooser.from_pb(
            parameters,
            batches[dst_batch_idx].next_token_chooser.dtype,
            batches[dst_batch_idx].next_token_chooser.device,
            batches[dst_batch_idx].next_token_chooser.tokenizer,
            fsm_grammar_states,
            quantization_enabled=hq_env.is_quantization_enabled,
        )

        input_ids = batches[dst_batch_idx].input_ids
        attention_mask = batches[dst_batch_idx].attention_mask
        position_ids = batches[dst_batch_idx].position_ids
        past_key_values = batches[dst_batch_idx].past_key_values
        input_length = max_input_length

        htorch.core.mark_step()

        return cls(
            batch_id=batch_id,
            requests=flat_requests,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            merged_kv_cache=False,
            next_token_chooser=next_token_chooser,
            top_n_tokens=top_n_tokens,
            top_n_tokens_tensor=top_n_tokens_tensor,
            input_length=input_length,
        )

    @classmethod
    def from_pb(
        cls,
        pb: generate_pb2.Batch,
        tokenizer: PreTrainedTokenizerBase,
        dtype: torch.dtype,
        device: torch.device,
        is_warmup: bool = False,
    ) -> "CausalLMBatch":
        dbg_trace('FROM_PB', f'num_reqs:{len(pb.requests)}')
        requests = [CausalLMRequest.from_pb(idx, req, tokenizer) for idx, req in enumerate(pb.requests)]
        inputs = []
        top_n_tokens = []

        # Parse batch
        max_truncation = 0
        for i, r in enumerate(pb.requests):
            inputs.append(concat_text_chunks(r.input_chunks.chunks))
            top_n_tokens.append(r.top_n_tokens)
            max_truncation = max(max_truncation, r.truncate)

        max_new_tokens = max(r.stopping_criteria.max_new_tokens for r in requests)
        max_input_length = max_truncation
        # TODO: by tokenizing all inputs at once we loose information on actual input lengths
        # this means that we cannot shift inputs to the left after a long input sequence
        # was filtered out
        new_bs = round_up(PREFILL_WARMUP_BATCH_SIZE_LIST, len(requests))
        missing_inputs = new_bs - len(inputs)
        dummy_inputs = ["?"] * missing_inputs
        parameters = [r.parameters for r in pb.requests]
        # append the dummy parameters for dummy request
        parameters = pad_next_token_chooser_parameters(parameters, new_bs)

        next_token_chooser = HeterogeneousNextTokenChooser.from_pb(
            pb=parameters,
            dtype=dtype,
            device=device,
            tokenizer=tokenizer,
            quantization_enabled=hq_env.is_quantization_enabled,
        )

        tokenized_inputs = tokenizer(
            inputs+dummy_inputs,
            return_tensors="pt",
            padding="longest",
            return_token_type_ids=False,
            truncation=True,
            max_length=max_truncation,
        ).to(device)

        input_len = tokenized_inputs["input_ids"].shape[1]

        # Round up sequence length
        bucket_size = max_input_length
        left_padding = max_input_length - input_len
        if is_warmup is False:
            if input_len < max_input_length and PAD_SEQUENCE_TO_MULTIPLE_OF != 0:
                assert PAD_SEQUENCE_TO_MULTIPLE_OF <= max_input_length, "PAD_SEQUENCE_TO_MULTIPLE_OF cannot be higher than max_input_length"
                rounded_seq_len = round_up(PREFILL_WARMUP_SEQLEN_LIST, input_len + 1)
                if rounded_seq_len <= max_input_length:
                    bucket_size = rounded_seq_len - 1
                else:
                    bucket_size = max_input_length - 1
                left_padding = bucket_size - input_len

        input_ids = tokenized_inputs["input_ids"]
        attention_mask = tokenized_inputs["attention_mask"]

        # Allocate space for first token
        input_ids = torch.nn.functional.pad(
            input_ids, (left_padding, 1), value=tokenizer.pad_token_id
        )
        attention_mask = torch.nn.functional.pad(
            attention_mask, (left_padding, 1), value=0
        )
        all_input_ids = torch.nn.functional.pad(
            input_ids, (0, max_new_tokens), value=tokenizer.pad_token_id
        ).T.split(1, dim=1)[0:len(pb.requests)]
        input_len = bucket_size
        for r in requests:
            r.input_length = input_len
            r.prefix_offset = input_len - 5
            r.read_offset = input_len
            r.all_input_ids = all_input_ids[r.idx]

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        htorch.core.mark_step()

        top_n_tokens_tensor = torch.tensor(
            top_n_tokens, device=device, dtype=torch.int64
        )
        htorch.core.mark_step()
        return cls(
            batch_id=pb.id,
            requests=requests,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            merged_kv_cache=False,
            next_token_chooser=next_token_chooser,
            top_n_tokens=top_n_tokens,
            top_n_tokens_tensor=top_n_tokens_tensor,
            input_length=input_len,
        )

    @tracer.start_as_current_span("filter")
    def filter(self, request_ids: List[int]) -> Optional["CausalLMBatch"]:
        dbg_trace('FILTER', f'num_reqs:{len(self.requests)} -> {len(request_ids)}')
        request_ids = set(request_ids)
        self.requests = [req for req in self.requests if req.data.id in request_ids]
        return self

    @classmethod
    @tracer.start_as_current_span("concatenate")
    def concatenate(cls, batches: List["CausalLMBatch"], pad_token_id: int = 0, is_warmup: bool = False) -> "CausalLMBatch":
        return cls.recombine(batches, pad_token_id, is_warmup)

    def __len__(self):
        return len(self.requests)

    @property
    def max_input_length(self):
        return max(req.input_length for req in self.requests)

    @property
    def batch_size(self):
        return self.attention_mask.size(0)

    @property
    def seq_length(self):
        return self.attention_mask.size(1)

    @property
    def right_padding(self):
        return self.seq_length - self.input_length

    # Maximum number of tokens this batch will grow to
    @property
    def max_tokens(self):
        max_total_tokens = self.attention_mask.size(1)
        return len(self.requests) * max_total_tokens


class CausalLM(Model):
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
        config_class=AutoConfig,
        batch_class=CausalLMBatch,

    ):

        if speculator:
            raise RuntimeError("Speculator decoding is not enabled for AutoModel")

        self.prev_bs = 0
        self.quantize = quantize

        # Create tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=trust_remote_code,
        )
        make_tokenizer_optional(tokenizer)

        # Create model
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        rank = int(os.getenv("RANK", "0"))
        dtype = torch.bfloat16 if dtype is None else dtype
        device = torch.device("hpu")

        if hq_env.is_quantization_enabled:
            htorch.core.hpu_set_env()

        if world_size > 1:
            model = self.get_deepspeed_model(
                model_id, dtype, revision
            )
            model = self.prepare_model_for_quantization(model)
        else:
            get_repo_root(model_id)

            # Check support for rope scaling
            model_kwargs = {}
            config = AutoConfig.from_pretrained(
                model_id
            )
            if hasattr(config, "rope_scaling"):
                model_kwargs["rope_scaling"] = self.get_rope_scaling()

            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                revision=revision,
                torch_dtype=dtype,
                trust_remote_code=trust_remote_code,
                **model_kwargs
            )
            model = self.prepare_model_for_quantization(model)
            model = model.eval().to(device)

        self.enable_hpu_graph = os.getenv("ENABLE_HPU_GRAPH", "true").lower() == "true" and LAZY_MODE == 1
        self.limit_hpu_graph = os.getenv("LIMIT_HPU_GRAPH", "false").lower() == "true"
        model = remove_kv_cache_from_output(model)
        if self.enable_hpu_graph:
            from habana_frameworks.torch.hpu import wrap_in_hpu_graph
            model = wrap_in_hpu_graph(model, disable_tensor_cache=True)
        else:
            if LAZY_MODE == 0:
                # It is said that "keep_input_mutations" is safe for inference to be done
                dbg_trace(
                    "TORCH COMPILE", f'Torch compiling of model')
                model.model = torch.compile(model.model, backend="hpu_backend", options={"keep_input_mutations": True})

        model = self.setup_quantization(model)

        if model.config.model_type not in MODELS_OPTIMIZED_WITH_STATIC_SHAPES:
            raise ValueError(f"Model type {model.config.model_type} is not supported!")

        if tokenizer.pad_token_id is None:
            if model.config.pad_token_id is not None:
                tokenizer.pad_token_id = model.config.pad_token_id
            elif model.config.eos_token_id is not None:
                if isinstance(model.config.eos_token_id, int):
                    tokenizer.pad_token_id = model.config.eos_token_id
                elif isinstance(model.config.eos_token_id, list):
                    tokenizer.pad_token_id = model.config.eos_token_id[0]
                else:
                    raise ValueError(
                        f"{type(model.config.eos_token_id)} type of eos_token_id in the model's config is not supported for tokenizer.pad_token_id"
                    )
            elif tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        self.kwargs = {
            "use_cache": True,
            "return_dict": True,
        }

        if model.config.model_type in ["llama", "mistral", "starcoder2", "qwen2"]:
            
            if model.config.model_type in ["llama", "mistral", "qwen2"]:
                self.kwargs["attn_softmax_bf16"] = True
                self.kwargs["trim_logits"] = True

            if os.getenv("USE_FLASH_ATTENTION", "false").lower() == "true":
                self.kwargs["use_flash_attention"] = True
            if os.getenv("FLASH_ATTENTION_RECOMPUTE", "false").lower() == "true":
                self.kwargs["flash_attention_recompute"] = True

        self.speculate = get_speculate()

        super(CausalLM, self).__init__(
            model_id=model_id,
            model=model,
            tokenizer=tokenizer,
            requires_padding=True,
            dtype=dtype,
            device=device,
            rank=rank,
        )

        # Create profiler
        ranks_to_profile = [int(val) for val in os.getenv("PROF_RANKS", "0").split(',')]
        record_shapes = os.getenv("PROF_RECORD_SHAPES", "false").lower() == "true"
        output_dir = os.getenv("PROF_PATH", "/tmp/hpu_profile")
        self.profiling_warmup_steps = int(os.getenv("PROF_WARMUPSTEP", "0")) if rank in ranks_to_profile else 0
        self.profiling_steps = int(os.getenv("PROF_STEP", "0")) if rank in ranks_to_profile else 0
        self.profiling_wait_steps = int(os.getenv("PROF_WAITSTEP", "0"))
        if self.profiling_steps > 0:
            self.hb_profiler = HabanaProfile(
                wait=self.profiling_wait_steps,
                warmup=self.profiling_warmup_steps,
                active=self.profiling_steps,
                output_dir=output_dir,
                record_shapes=record_shapes
            )
            self.hb_profiler.start()
        else:
            self.hb_profiler = None
        self.step = 0

    def get_deepspeed_model(
        self,
        model_id: str,
        dtype: torch.dtype,
        revision: Optional[str] = None
    ) -> torch.nn.Module:
        import deepspeed
        from habana_frameworks.torch.distributed.hccl import initialize_distributed_hpu

        world_size, rank, local_rank = initialize_distributed_hpu()
        model_kwargs = {
            "revision": revision
        }

        # Initialize process(es) for DeepSpeed
        deepspeed.init_distributed(dist_backend="hccl")
        logger.info(
            "DeepSpeed is enabled. world_size {} rank {} local_rank {}".format(world_size, rank, local_rank)
        )
        config = AutoConfig.from_pretrained(model_id, **model_kwargs)
        load_to_meta = model_on_meta(config)

        # Check support for rope scaling
        if hasattr(config, "rope_scaling"):
            config.rope_scaling = self.get_rope_scaling()
            model_kwargs["rope_scaling"] = self.get_rope_scaling()

        if load_to_meta:
            # Construct model with fake meta tensors, later will be replaced on devices during ds-inference ckpt load
            with deepspeed.OnDevice(dtype=dtype, device="meta"):
                model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype)
        else:
            get_repo_root(model_id, local_rank=os.getenv("LOCAL_RANK"))
            # TODO: revisit placement on CPU when auto-injection is possible
            with deepspeed.OnDevice(dtype=dtype, device="cpu"):
                model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, **model_kwargs)
        model = model.eval()

        # Initialize the model
        ds_inference_kwargs = {"dtype": dtype}
        ds_inference_kwargs["tensor_parallel"] = {"tp_size": world_size}
        ds_inference_kwargs["enable_cuda_graph"] = False

        if load_to_meta:
            # model loaded to meta is managed differently
            checkpoints_json = tempfile.NamedTemporaryFile(suffix=".json", mode="+w")
            write_checkpoints_json(model_id, local_rank, checkpoints_json)
            ds_inference_kwargs["checkpoint"] = checkpoints_json.name
        model = deepspeed.init_inference(model, **ds_inference_kwargs)

        return model.module

    def get_rope_scaling(self) -> Optional[Dict]:
        rope_scaling = os.getenv("ROPE_SCALING", None)
        if rope_scaling is None:
            return None

        rope_factor = float(os.getenv("ROPE_FACTOR", 1.0))
        return {
            'type': rope_scaling, 'factor': float(rope_factor)
        }

    def setup_quantization(self, model):
        if hq_env.is_quantization_enabled:
            htorch.core.quantization._mark_params_as_const(model)
            htorch.core.quantization._check_params_as_const(model)
            htorch.core.hpu_initialize(model)
        return model

    def prepare_model_for_quantization(self, model):
        if hq_env.is_quantization_enabled:
            if model.config.model_type == "llama":
                self.patch_scoped_linear_all_reduce(model)
            model = hq_env.prepare_model_for_quantization(model)
        return model

    def patch_scoped_linear_all_reduce(self, model):
        from deepspeed.module_inject.layers import LinearAllreduce
        from optimum.habana.transformers.models.modeling_all_models import ScopedLinearAllReduce
        for name, module in model.named_children():
            if type(module) is LinearAllreduce:
                SL = ScopedLinearAllReduce(mod=module)
                setattr(model, name, SL)
            self.patch_scoped_linear_all_reduce(module)

    @property
    def batch_type(self) -> Type[CausalLMBatch]:
        return CausalLMBatch

    def decode(self, generated_ids: List[int]) -> str:
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def decode_token(
        self,
        all_input_ids: List[int],
        prefix_offset: int = 0,
        read_offset: int = 0,
    ) -> Tuple[str, int, int]:
        if is_tokenizer_transparent(self.tokenizer):
            new_text = self.tokenizer.decode(all_input_ids[read_offset:], skip_special_tokens=False)
            return new_text, read_offset, len(all_input_ids)
        else:
            return super().decode_token(all_input_ids, prefix_offset, read_offset)

    def forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        token_idx,
        past_key_values: Optional[List[Tuple]] = None,
        bypass_hpu_graph: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        # Model Forward
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "token_idx": token_idx,
        }

        # Optimum Habana got "lazy_mode" key-val only supported for llama type of models
        if self.model.config.model_type == "llama" :
            kwargs["lazy_mode"] = LAZY_MODE == 1

        if self.has_position_ids:
            kwargs["position_ids"] = position_ids

        if bypass_hpu_graph != None:
            kwargs["bypass_hpu_graphs"] = bypass_hpu_graph

        kwargs.update(self.kwargs)
        if past_key_values is not None:
            return self.model.forward(**kwargs)
        else:
            outputs = self.model.forward(**kwargs)
            return outputs.logits, outputs.past_key_values

    @tracer.start_as_current_span("generate_token")
    def generate_token(
        self, batches: List[CausalLMBatch], is_warmup: bool = False
    ) -> Tuple[List[Generation], Optional[CausalLMBatch], Tuple[int, int]]:
        start = time.time_ns()
        # Results
        generations: List[Generation] = []
        prev_batches = []
        requests_to_generate = []
        # In order to pipeline any actions on CPU we perform the operation in 3 main stages:
        # Stage 1. Collect next token ids of any previously started generations
        for batch_id, batch in enumerate(batches):
            if batch.logits is not None:
                logits = batch.logits
                past = batch.past
                prefill = batch.past_key_values is None
                if prefill:
                    # no right padding for prefill
                    token_idx_scalar = batch.attention_mask.shape[-1] - 1
                    token_idx = torch.tensor(token_idx_scalar).to(self.device)
                else:
                    token_idx_scalar = batch.attention_mask.shape[-1] - batch.right_padding
                    token_idx = torch.tensor(token_idx_scalar).to(self.device)

                # Select next token
                input_length = batch.input_length
                if logits.shape[-2] > 1:
                    next_token_ids, next_token_logprobs, logprobs, _, _ = batch.next_token_chooser(
                        batch.input_ids, logits[:, input_length - 1: input_length, :].squeeze(-2), self.speculate
                    )
                else:
                    next_token_ids, next_token_logprobs, logprobs, _, _ = batch.next_token_chooser(
                        batch.input_ids, logits.squeeze(-2), self.speculate
                    )
                # Speculation is not active for causal
                accepted_ids = torch.ones_like(batch.input_ids)[:, 0]
                batch_top_token_ids, batch_top_token_logprobs = batch_top_tokens(
                    batch.top_n_tokens,
                    batch.top_n_tokens_tensor,
                    logprobs,
                    accepted_ids,
                )

                prev_batches.append({
                    'next_token_ids': next_token_ids,
                    'next_token_logprobs': next_token_logprobs,
                })

                for req_idx, req in enumerate(batch.requests):
                    requests_to_generate.append({
                        'req': req,
                        'prev_req_idx': req.idx,
                        'batch_id': batch_id,
                        'seed': batch.next_token_chooser.seeds[req_idx],
                        'do_sample': batch.next_token_chooser.do_sample[req_idx],
                        'top_n_tokens': batch.top_n_tokens[req_idx],
                        'top_token_ids': batch_top_token_ids[req_idx],
                        'top_token_logprobs': batch_top_token_logprobs[req_idx],
                        'grammar_state': batch.next_token_chooser.fsm_grammar_states[req_idx],

                    })

                htorch.core.mark_step()

                # Add new token into input_ids
                batch.input_ids.index_copy_(1, token_idx, next_token_ids.unsqueeze(1))

                # Update attention_mask as we added a new token to input_ids
                batch.attention_mask.index_fill_(1, token_idx, 1)

                # Adjust lengths
                batch.input_length += 1

                # Update position_ids
                if prefill:
                    batch.position_ids = torch.index_select(batch.position_ids, 1, token_idx - 1) + 1
                else:
                    batch.position_ids += 1
                # Update past key values
                if prefill:
                    batch.past_key_values = past

        htorch.core.mark_step()

        # Stage 2. Prepare new batch for speculative scheduling
        if len(batches) > 1:
            batch = self.batch_type.concatenate(batches, self.tokenizer.pad_token_id, is_warmup)
        else:
            batch = batches[0]

        prefill = batch.past_key_values is None

        # Check if we need to do any bookkeeping first
        if not prefill:
            batch = batch.__class__.recombine([batch], self.tokenizer.pad_token_id, is_warmup)

        scenario = 'PREFILL' if prefill else 'GENERATE'
        if self.enable_hpu_graph and self.limit_hpu_graph and round_up(DECODE_WARMUP_BATCH_SIZE_LIST, batch.batch_size) != self.prev_bs:
            self.model.clear_cache()
            self.prev_bs = round_up(DECODE_WARMUP_BATCH_SIZE_LIST, batch.batch_size)
        dbg_trace(
            scenario, f'bs:{batch.batch_size} num_reqs:{len(batch.requests)} seq_len:{batch.seq_length} padding:{batch.right_padding}')
        assert batch.right_padding > 0, 'No more room for next token!'

        # Execute batch
        if prefill:
            # no right padding for prefill
            token_idx = torch.tensor(batch.attention_mask.shape[-1] - 1).to(self.device)
            batch.logits, batch.past = self.forward(
                batch.input_ids,
                batch.attention_mask,
                batch.position_ids,
                token_idx,
                batch.past_key_values,
                bypass_hpu_graph=prefill and self.limit_hpu_graph if self.enable_hpu_graph else None,
            )
        elif all([req.stopping_criteria.max_new_tokens == 1 for req in batch.requests]):
            # Don't schedule next forward if max_new_tokens for all requests equals 1
            # - we've already generated the first and only needed token in the prefill phase
            pass
        else:
            token_idx = torch.tensor(batch.attention_mask.shape[-1] - batch.right_padding).to(self.device)
            input_ids = torch.index_select(batch.input_ids, 1, token_idx - 1)
            batch.logits = self.forward(
                input_ids,
                batch.attention_mask,
                batch.position_ids,
                token_idx,
                batch.past_key_values,
                bypass_hpu_graph=prefill and self.limit_hpu_graph if self.enable_hpu_graph else None,
            )

        htorch.core.mark_step()

        start_decode = time.time_ns()

        # Stage 3. Finish and return previous generations
        stopped = len(requests_to_generate) > 0
        for prev_batch in prev_batches:
            prev_batch['next_token_logprobs'] = prev_batch['next_token_logprobs'].tolist()
            prev_batch['next_token_ids_cpu'] = prev_batch['next_token_ids'].cpu()
        htorch.core.mark_step()

        for req_data in requests_to_generate:
            req = req_data['req']
            i = req_data['prev_req_idx']
            prev_batch_id = req_data['batch_id']
            assert len(prev_batches) > prev_batch_id
            next_token_ids_cpu = prev_batches[prev_batch_id]['next_token_ids_cpu']
            next_token_logprobs = prev_batches[prev_batch_id]['next_token_logprobs']

            request = req.data
            input_length = req.input_length
            prefix_offset = req.prefix_offset
            read_offset = req.read_offset
            do_sample = req_data['do_sample']
            seed = req_data['seed']
            stopping_criteria = req.stopping_criteria
            all_input_ids = req.all_input_ids
            next_token_id = next_token_ids_cpu[i]
            next_token_logprob = next_token_logprobs[i]
            top_n_tokens = req_data['top_n_tokens']
            top_token_ids = req_data['top_token_ids']
            top_token_logprobs = req_data['top_token_logprobs']
            grammar_state = req_data['grammar_state']

            # Append next token to all tokens
            all_input_ids[input_length] = next_token_id
            new_input_length = input_length + 1

            # Generated token
            if is_tokenizer_transparent(self.tokenizer) and len(stopping_criteria.stop_sequence_criterias) == 0:
                next_token_text = ''
            else:
                next_token_text, prefix_offset, read_offset = self.decode_token(
                    all_input_ids[0:new_input_length, 0], prefix_offset, read_offset
                )

            # Evaluate stopping criteria
            stop, reason = stopping_criteria(
                next_token_id,
                next_token_text,
            )

            if not stop:
                stopped = False

            # Shard generations
            # All generations will be appended in the rust sharded client
            if i % self.world_size == self.rank:
                if stop:
                    # Decode generated tokens
                    if is_tokenizer_transparent(self.tokenizer):
                        output_text = None
                    else:
                        output_text = self.decode(
                            all_input_ids[new_input_length - stopping_criteria.current_tokens: new_input_length, 0]
                        )
                    generated_text = GeneratedText(
                        output_text,
                        stopping_criteria.current_tokens,
                        reason,
                        seed if do_sample else None,
                    )
                else:
                    generated_text = None

                # Prefill
                if stopping_criteria.current_tokens == 1 and request.prefill_logprobs:
                    # Remove generated token to only have prefill and add nan for first prompt token
                    prefill_logprobs = [float("nan")] + next_token_logprobs
                    prefill_token_ids = all_input_ids[0: new_input_length - 1]
                    prefill_texts = self.tokenizer.batch_decode(
                        prefill_token_ids,
                        clean_up_tokenization_spaces=False,
                        skip_special_tokens=False,
                    )
                    prefill_tokens = Tokens(
                        prefill_token_ids,
                        prefill_logprobs,
                        prefill_texts,
                        is_special=[],
                    )
                else:
                    prefill_tokens = None

                if top_n_tokens > 0:
                    all_top_tokens = []
                    for top_token_ids, top_token_logprobs in zip(
                        top_token_ids, top_token_logprobs
                    ):
                        toptoken_texts = self.tokenizer.batch_decode(
                            top_token_ids,
                            clean_up_tokenization_spaces=False,
                            skip_special_tokens=False,
                        )
                        special_toptokens = [
                            token_id in self.all_special_ids
                            for token_id in top_token_ids
                        ]
                        top_tokens = Tokens(
                            top_token_ids,
                            top_token_logprobs,
                            toptoken_texts,
                            special_toptokens,
                        )
                        all_top_tokens.append(top_tokens)
                    top_tokens = all_top_tokens
                else:
                    top_tokens = None

                generation = Generation(
                    request.id,
                    prefill_tokens,
                    Tokens(
                        [next_token_id],
                        [next_token_logprob],
                        [next_token_text],
                        [next_token_id in self.all_special_ids],
                    ),
                    generated_text,
                    top_tokens,
                )

                generations.append(generation)

            batch.next_token_chooser = (
                batch.next_token_chooser.advance_grammar_single_with_past_state(
                    req.idx, next_token_id, grammar_state
                )
            )

            req.all_input_ids = all_input_ids
            req.input_length = new_input_length
            req.prefix_offset = prefix_offset
            req.read_offset = read_offset

        htorch.core.mark_step()
        self.step = self.step + 1
        if self.hb_profiler is not None:
            if self.step > self.profiling_wait_steps + self.profiling_warmup_steps + self.profiling_steps:
                self.hb_profiler.stop()
            else:
                self.hb_profiler.step()

        forward_ns = start_decode - start
        decode_ns = time.time_ns() - start_decode
        return generations, batch if not stopped else None, (forward_ns, decode_ns)

    def generate_warmup_batch(self, request, seq_len, batch_size, is_warmup):
        batch = copy.deepcopy(request.batch)
        for req in batch.requests:
            req.truncate = seq_len

        for i in range(len(batch.requests) - batch_size):
            batch.requests.pop()

        return CausalLMBatch.from_pb(batch, self.tokenizer, self.dtype, self.device, is_warmup)


    def warmup(self, request) -> None:
        is_warmup = True
        batch = CausalLMBatch.from_pb(request.batch, self.tokenizer, self.dtype, self.device, is_warmup = is_warmup)
        try:
            # max prefill batch size warmup
            _, prefill_batch, _ = self.generate_token([batch], is_warmup)
        except:
            raise RuntimeError(
                f"Not enough memory to handle {len(batch.input_ids)} prefill tokens. "
                f"You need to decrease `--max-batch-prefill-tokens`"
            )

        global MAX_TOTAL_TOKENS, MAX_BATCH_TOTAL_TOKENS, PREFILL_WARMUP_BATCH_SIZE_LIST, PREFILL_WARMUP_SEQLEN_LIST, DECODE_WARMUP_BATCH_SIZE_LIST
        max_input_length =  batch.input_ids.shape[1]
        max_prefill_batch_size = batch.input_ids.shape[0]
        PREFILL_WARMUP_BATCH_SIZE_LIST = []
        batch_size = 1
        while batch_size <= max_prefill_batch_size:
            PREFILL_WARMUP_BATCH_SIZE_LIST.append(batch_size)
            batch_size = batch_size * 2
        if PREFILL_WARMUP_BATCH_SIZE_LIST[-1] < max_prefill_batch_size :
            PREFILL_WARMUP_BATCH_SIZE_LIST.append(max_prefill_batch_size)

        seq_len = PAD_SEQUENCE_TO_MULTIPLE_OF
        PREFILL_WARMUP_SEQLEN_LIST = []
        i = 0
        while seq_len <= max_input_length:
            PREFILL_WARMUP_SEQLEN_LIST.append(seq_len)
            seq_len += PAD_SEQUENCE_TO_MULTIPLE_OF*(2**i)
            i += 1
        if PREFILL_WARMUP_SEQLEN_LIST[-1] < max_input_length:
            PREFILL_WARMUP_SEQLEN_LIST.append(max_input_length)

        #Prefill and decode warmup
        DECODE_WARMUP_BATCH_SIZE_LIST = []
        prefill_batch = None
        decode_batch = None
        try:
            for batch_size in PREFILL_WARMUP_BATCH_SIZE_LIST :
                for seq_len in PREFILL_WARMUP_SEQLEN_LIST :
                    batch = self.generate_warmup_batch(request, seq_len, batch_size, is_warmup)
                    _, prefill_batch, _ = self.generate_token([batch], is_warmup)
                    _, decode_batch, _ = self.generate_token([prefill_batch], is_warmup)

                DECODE_WARMUP_BATCH_SIZE_LIST.append(batch_size)

        except:
            raise RuntimeError(
                f"Not enough memory to handle following prefill and decode warmup."
                f"Prefill batch size list:{PREFILL_WARMUP_BATCH_SIZE_LIST}"
                f"Prefill sequence length list:{PREFILL_WARMUP_SEQLEN_LIST}"
                f"Decode batch size list:{DECODE_WARMUP_BATCH_SIZE_LIST}"
                f"You need to decrease `--max-batch-prefill-tokens`"
            )

        mem_stats = get_hpu_memory_stats(self.device)
        logger.info(
                f"\nFollowing prefill and decode warmup successfully.\n"
                f"Prefill batch size list:{PREFILL_WARMUP_BATCH_SIZE_LIST}\n"
                f"Prefill sequence length list:{PREFILL_WARMUP_SEQLEN_LIST}\n"
                f"Decode batch size list:{DECODE_WARMUP_BATCH_SIZE_LIST}\n"
                f"Memory stats: {mem_stats} "
            )

        max_decode_batch_size = math.floor(MAX_BATCH_TOTAL_TOKENS / MAX_TOTAL_TOKENS)
        batch_size = max_prefill_batch_size * 2
        # Decode warmup with bigger batch_size
        try:
            if DECODE_WARMUP_BATCH_SIZE_LIST[-1] < max_decode_batch_size and batch_size <= max_decode_batch_size:
                batches = []
                for i in range(int(batch_size/max_prefill_batch_size)) :
                    batch = self.generate_warmup_batch(request, PREFILL_WARMUP_SEQLEN_LIST[0], DECODE_WARMUP_BATCH_SIZE_LIST[-1], is_warmup)
                    _, prefill_batch, _ = self.generate_token([batch], is_warmup)
                    batches.append(prefill_batch)
                while batch_size <= max_decode_batch_size:
                    _, decode_batch, _ = self.generate_token(batches, is_warmup)
                    DECODE_WARMUP_BATCH_SIZE_LIST.append(batch_size)
                    batch_size = batch_size * 2
                    batches.clear()

                    for i in range(int(batch_size/max_prefill_batch_size)) :
                        batch = self.generate_warmup_batch(request, PREFILL_WARMUP_SEQLEN_LIST[0], DECODE_WARMUP_BATCH_SIZE_LIST[-1], is_warmup)
                        _, prefill_batch, _ = self.generate_token([batch], is_warmup)
                        batches.append(prefill_batch)

                batches.clear()
                if DECODE_WARMUP_BATCH_SIZE_LIST[-1] < max_decode_batch_size:
                    max_decode_batch_size = math.floor( max_decode_batch_size / 2) * 2
                    batch_size = max_decode_batch_size
                    for i in range(int(max_decode_batch_size / 2)) :
                        batch = self.generate_warmup_batch(request, PREFILL_WARMUP_SEQLEN_LIST[0], 2, is_warmup)
                        _, prefill_batch, _ = self.generate_token([batch], is_warmup)
                        batches.append(prefill_batch)
                    _, decode_batch, _ = self.generate_token(batches, is_warmup)
                    DECODE_WARMUP_BATCH_SIZE_LIST.append(max_decode_batch_size)
                max_batch_total_tokens = max_decode_batch_size * MAX_TOTAL_TOKENS
                MAX_BATCH_TOTAL_TOKENS = max_batch_total_tokens
        except :
            raise RuntimeError(
                f"Not enough memory to handle batch_size({batch_size}) decode warmup."
                f"Decode batch size list:{DECODE_WARMUP_BATCH_SIZE_LIST}"
                f"max_decode_batch_size is {max_decode_batch_size}"
                f"You need to decrease env `MAX_BATCH_TOTAL_TOKENS` or '--max_batch_total_tokens'"
            )

        mem_stats = get_hpu_memory_stats(self.device)
        logger.info(
                f"\nFollowing decode warmup successfully.\n"
                f"Decode batch size list:{DECODE_WARMUP_BATCH_SIZE_LIST}\n"
                f"Memory stats: {mem_stats}"
            )

        return MAX_BATCH_TOTAL_TOKENS