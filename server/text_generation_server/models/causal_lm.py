import os
import tempfile
import itertools
import bisect
import math

import torch

from dataclasses import dataclass
from opentelemetry import trace
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerBase, AutoConfig
from typing import Optional, Tuple, List, Type, Dict

import text_generation_server.habana_quantization_env as hq_env
import habana_frameworks.torch as htorch
from habana_frameworks.torch.hpu import wrap_in_hpu_graph
from contextlib import nullcontext
from optimum.habana.utils import HabanaProfile

from optimum.habana.transformers.generation import MODELS_OPTIMIZED_WITH_STATIC_SHAPES
from optimum.habana.checkpoint_utils import (
    get_repo_root,
    model_on_meta,
    write_checkpoints_json,
)

from text_generation_server.utils.tokens import batch_top_tokens
from text_generation_server.models import Model
from text_generation_server.models.types import (
    Batch,
    PrefillTokens,
    Generation,
    GeneratedText,
    TopTokens,
)
from text_generation_server.pb import generate_pb2
from text_generation_server.utils import HeterogeneousNextTokenChooser, StoppingCriteria, Sampling, make_tokenizer_optional, is_tokenizer_transparent
from text_generation_server.utils.debug import dbg_trace
from loguru import logger
from functools import wraps


tracer = trace.get_tracer(__name__)

MAX_TOTAL_TOKENS = int(os.environ.get('MAX_TOTAL_TOKENS', 2048))
BATCH_BUCKET_SIZE = int(os.environ.get('BATCH_BUCKET_SIZE', 8))
PAD_SEQUENCE_TO_MULTIPLE_OF = int(os.environ.get('PAD_SEQUENCE_TO_MULTIPLE_OF', 128))
PREFILL_BATCH_BUCKET_SIZE = int(os.environ.get('PREFILL_BATCH_BUCKET_SIZE', 4))
CHUNK_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]


def round_up(number, k):
    return (number + k - 1) // k * k


def to_tensor_indices(indices, device):
    return torch.tensor(indices, dtype=torch.int32, device=device)


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


def extend_tensor(tensor, padding, dim):
    result = torch.cat([tensor, padding], dim=dim)
    htorch.core.mark_step()
    return result


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


def merge(tensor_group):
    tensor_group = [torch.stack(tensor_group)]
    htorch.core.mark_step()
    return tensor_group


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

    # Generation helpers
    next_token_chooser: HeterogeneousNextTokenChooser
    top_n_tokens: List[int]
    top_n_tokens_tensor: torch.Tensor

    input_length: int

    logits = None
    past = None

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
        key_dim = -2  # TODO: Add case for Bloom and other models
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
    def recombine(cls, batches: List["CausalLMBatch"], pad_token_id: int) -> "CausalLMBatch":
        total_requests = sum(len(b) for b in batches)
        new_bs = round_up(total_requests, BATCH_BUCKET_SIZE)
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
        reshape = (batches[dst_batch_idx].batch_size != new_bs)

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
        if len(flat_requests) < new_bs:
            for i in range(new_bs-len(flat_requests)) :
                #append the dummy parameters for dummy request
                parameters.append(parameters[0])

        next_token_chooser = HeterogeneousNextTokenChooser.from_pb(
            parameters,
            batches[dst_batch_idx].next_token_chooser.dtype,
            batches[dst_batch_idx].next_token_chooser.device
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
        is_optimized_for_gaudi: bool = False,
    ) -> "CausalLMBatch":
        dbg_trace('FROM_PB', f'num_reqs:{len(pb.requests)}')
        requests = [CausalLMRequest.from_pb(idx, req, tokenizer) for idx, req in enumerate(pb.requests)]

        max_input_length = max(r.data.truncate for r in requests)
        max_new_tokens = max(r.stopping_criteria.max_new_tokens for r in requests)

        # TODO: Add support for sparse batches
        top_n_tokens = [r.top_n_tokens for r in pb.requests]
        top_n_tokens_tensor = torch.tensor(top_n_tokens, device=device, dtype=torch.int64)

        # TODO: by tokenizing all inputs at once we loose information on actual input lengths
        # this means that we cannot shift inputs to the left after a long input sequence
        # was filtered out
        new_bs = round_up(len(requests), PREFILL_BATCH_BUCKET_SIZE)
        dummy_inputs = ["?"] * (new_bs - len(requests))
        parameters = [r.parameters for r in pb.requests]
        if len(pb.requests) < new_bs:
            for i in range(new_bs-len(pb.requests)) :
                #append the dummy parameters for dummy request
                parameters.append(parameters[0])

        next_token_chooser = HeterogeneousNextTokenChooser.from_pb(parameters, dtype, device)
        tokenized_inputs = tokenizer(
            [r.data.inputs for r in requests] + dummy_inputs,
            return_tensors="pt",
            padding="longest",
            return_token_type_ids=False,
            truncation=True,
            max_length=max_input_length,
        )

        input_len = tokenized_inputs["input_ids"].shape[1]

        bucket_size = max_input_length
        left_padding = max_input_length - input_len
        if input_len < max_input_length and PAD_SEQUENCE_TO_MULTIPLE_OF != 0:
            assert PAD_SEQUENCE_TO_MULTIPLE_OF <= max_input_length, "PAD_SEQUENCE_TO_MULTIPLE_OF cannot be higher than max_input_length"
            rounded_seq_len = round_up(input_len + 1, PAD_SEQUENCE_TO_MULTIPLE_OF)
            if rounded_seq_len <= max_input_length:    
                bucket_size = rounded_seq_len - 1
            else:
                bucket_size = max_input_length - 1
            left_padding = bucket_size - input_len

        input_ids = tokenized_inputs["input_ids"]
        attention_mask = tokenized_inputs["attention_mask"]

        if is_optimized_for_gaudi:
            # Allocate space for first token
            input_ids = torch.nn.functional.pad(
                input_ids, (left_padding, 1), value=tokenizer.pad_token_id
            )
            attention_mask = torch.nn.functional.pad(
                attention_mask, (left_padding, 1), value=0
            )
            all_input_ids = torch.nn.functional.pad(
                input_ids, (0, max_new_tokens), value=tokenizer.pad_token_id
            ).T.split(1, dim=1)
        else:
            all_input_ids = input_ids.clone().T.split(1, dim=1)

        # New input length after left padding
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
    def concatenate(cls, batches: List["CausalLMBatch"], pad_token_id: int = 0) -> "CausalLMBatch":
        return cls.recombine(batches, pad_token_id)

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
        revision: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = torch.device("hpu")
        if hq_env.is_quantization_enabled:
            htorch.core.hpu_set_env()

        dtype = torch.bfloat16 if dtype is None else dtype

        from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi

        adapt_transformers_to_gaudi()

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            padding_side="left",
            truncation_side="left",
        )
        make_tokenizer_optional(tokenizer)

        model_kwargs = {
            "revision": revision,
        }

        world_size = int(os.getenv("WORLD_SIZE", "1"))
        rank = int(os.getenv("RANK", "0"))
        self.enable_hpu_graph = os.getenv("ENABLE_HPU_GRAPH", "true").lower() == "true"
        self.limit_hpu_graph = os.getenv("LIMIT_HPU_GRAPH", "false").lower() == "true"

        if world_size > 1:
            import habana_frameworks.torch.hpu as torch_hpu

            # Get world size, rank and local rank
            from habana_frameworks.torch.distributed.hccl import initialize_distributed_hpu

            world_size, rank, local_rank = initialize_distributed_hpu()
            import deepspeed

            # Initialize process(es) for DeepSpeed
            deepspeed.init_distributed(dist_backend="hccl")
            logger.info(
                "DeepSpeed is enabled. world_size {} rank {} local_rank {}".format(world_size, rank, local_rank)
            )
            config = AutoConfig.from_pretrained(model_id, **model_kwargs)
            load_to_meta = model_on_meta(config)

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
            model = model.module
            model = self.prepare_model_for_quantization(model)
            model = remove_kv_cache_from_output(model)
            if self.enable_hpu_graph:
                model = wrap_in_hpu_graph(model, disable_tensor_cache=True)

        else:
            get_repo_root(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                revision=revision,
                torch_dtype=dtype,
            )
            model = self.prepare_model_for_quantization(model)
            model = model.eval().to(device)
            # wrap in hpu_graph only if self.enable_hpu_graph is set
            model = remove_kv_cache_from_output(model)
            if self.enable_hpu_graph:
                model = wrap_in_hpu_graph(model, disable_tensor_cache=True)
        model = self.setup_quantization(model)

        if model.config.model_type in MODELS_OPTIMIZED_WITH_STATIC_SHAPES:
            self.is_optimized_for_gaudi = True
        else:
            self.is_optimized_for_gaudi = False

        if tokenizer.pad_token_id is None:
            if model.config.pad_token_id is not None:
                tokenizer.pad_token_id = model.config.pad_token_id
            elif model.config.eos_token_id is not None:
                tokenizer.pad_token_id = model.config.eos_token_id
            elif tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        kwargs = {
            "use_cache": True,
            "return_dict": True,
        }

        if model.config.model_type == "llama":
            kwargs["attn_softmax_bf16"] = True
            kwargs["trim_logits"] = True

        super(CausalLM, self).__init__(
            model=model,
            tokenizer=tokenizer,
            requires_padding=True,
            dtype=dtype,
            device=device,
            rank=rank,
            kwargs=kwargs,
        )
        prof_ranks = [int(val) for val in os.getenv("PROF_RANKS", "0").split(',')]
        self.profiling_warmup_steps = int(os.getenv("PROF_WARMUPSTEP", "0")) if rank in prof_ranks else 0
        self.profiling_steps = int(os.getenv("PROF_STEP", "0")) if rank in prof_ranks else 0
        self.profiling_wait_steps = int(os.getenv("PROF_WAITSTEP", "0"))
        record_shapes = os.getenv("PROF_RECORD_SHAPES", "false").lower() == "true"
        output_dir = os.getenv("PROF_PATH", "/tmp/hpu_profile")
        if self.profiling_steps > 0:
            self.hb_profiler = HabanaProfile(
                wait=self.profiling_wait_steps,
                warmup=self.profiling_warmup_steps,
                active=self.profiling_steps,
                output_dir=output_dir, record_shapes=record_shapes
            )
            self.hb_profiler.start()
        else:
            self.hb_profiler = None
        self.step = 0

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
            import habana_quantization_toolkit
            habana_quantization_toolkit.prep_model(model)
        return model

    def finish_quantization_measurements(self, model):
        if hq_env.is_quantization_enabled:
            import habana_quantization_toolkit
            habana_quantization_toolkit.finish_measurements(self.model)
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
        token_idx: Optional = None,
        past_key_values: Optional = None,
        bypass_hpu_graph: Optional = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        # Model Forward
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
        }

        if self.is_optimized_for_gaudi:
            kwargs["token_idx"] = token_idx

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
    def generate_token(self, batches: List[CausalLMBatch]) -> Tuple[List[Generation], Optional[CausalLMBatch]]:
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
                if self.is_optimized_for_gaudi:
                    if prefill:
                        # no right padding for prefill
                        token_idx_scalar = batch.attention_mask.shape[-1] - 1
                        token_idx = torch.tensor(token_idx_scalar).to(self.device)
                    else:
                        token_idx_scalar = batch.attention_mask.shape[-1] - batch.right_padding
                        token_idx = torch.tensor(token_idx_scalar).to(self.device)
                else:
                    token_idx = None

                # Select next token
                input_length = batch.input_length
                if self.is_optimized_for_gaudi and logits.shape[-2] > 1:
                    next_token_ids, next_token_logprobs, logprobs = batch.next_token_chooser(
                        batch.input_ids, logits[:, input_length - 1: input_length, :].squeeze(-2)
                    )
                else:
                    next_token_ids, next_token_logprobs, logprobs = batch.next_token_chooser(
                        batch.input_ids, logits.squeeze(-2)
                    )
                batch_top_token_ids, batch_top_token_logprobs = batch_top_tokens(
                    batch.top_n_tokens,
                    batch.top_n_tokens_tensor,
                    logprobs,
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
                    })

                htorch.core.mark_step()

                if token_idx is None:
                    batch.input_ids[:, 0] = next_token_ids[:, 0]
                else:
                    batch.input_ids.index_copy_(1, token_idx, next_token_ids.unsqueeze(1))

                # Slice unused values from prefill, use it to store next token
                if token_idx is None:
                    batch.input_ids = batch.input_ids[:, :1]

                # Update attention_mask as we added a new token to input_ids
                if self.is_optimized_for_gaudi:
                    batch.attention_mask.index_fill_(1, token_idx, 1)
                else:
                    batch.attention_mask[:, -batch.padding_right_offset] = 1

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
            batch = self.batch_type.concatenate(batches, self.tokenizer.pad_token_id)
        else:
            batch = batches[0]

        prefill = batch.past_key_values is None

        # Check if we need to do any bookkeeping first
        if not prefill:
            batch = batch.__class__.recombine([batch], self.tokenizer.pad_token_id)

        scenario = 'PREFILL' if prefill else 'GENERATE'
        dbg_trace(
            scenario, f'bs:{batch.batch_size} num_reqs:{len(batch.requests)} seq_len:{batch.seq_length} padding:{batch.right_padding}')
        assert batch.right_padding > 0, 'No more room for next token!'

        if self.is_optimized_for_gaudi:
            if prefill:
                # no right padding for prefill
                token_idx = torch.tensor(batch.attention_mask.shape[-1] - 1).to(self.device)
            else:
                token_idx = torch.tensor(batch.attention_mask.shape[-1] - batch.right_padding).to(self.device)
            attention_mask = batch.attention_mask
        else:
            token_idx = None
            # slice the attention mask to the correct shape
            # TODO fix me!
            attention_mask = batch.attention_mask[:, : -batch.padding_right_offset]

        if not prefill and token_idx is not None:
            input_ids = torch.index_select(batch.input_ids, 1, token_idx - 1)
        else:
            input_ids = batch.input_ids

        if prefill:
            batch.logits, batch.past = self.forward(
                input_ids,
                attention_mask,
                batch.position_ids,
                token_idx,
                batch.past_key_values,
                bypass_hpu_graph=prefill and self.limit_hpu_graph if self.enable_hpu_graph else None
            )
        else:
            batch.logits = self.forward(
                input_ids,
                attention_mask,
                batch.position_ids,
                token_idx,
                batch.past_key_values,
                bypass_hpu_graph=prefill and self.limit_hpu_graph if self.enable_hpu_graph else None
            )

        htorch.core.mark_step()

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

            # Append next token to all tokens
            if self.is_optimized_for_gaudi:
                all_input_ids[input_length] = next_token_id
            else:
                all_input_ids = torch.cat([all_input_ids, next_token_id])
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
                    prefill_tokens = PrefillTokens(prefill_token_ids, prefill_logprobs, prefill_texts)
                else:
                    prefill_tokens = None

                if top_n_tokens > 0:
                    toptoken_texts = self.tokenizer.batch_decode(
                        top_token_ids,
                        clean_up_tokenization_spaces=False,
                        skip_special_tokens=False,
                    )
                    special_toptokens = [token_id in self.all_special_ids for token_id in top_token_ids]
                    top_tokens = TopTokens(
                        top_token_ids,
                        top_token_logprobs,
                        toptoken_texts,
                        special_toptokens,
                    )
                else:
                    top_tokens = None

                generation = Generation(
                    request.id,
                    prefill_tokens,
                    next_token_id,
                    next_token_logprob,
                    next_token_text,
                    next_token_id in self.all_special_ids,
                    generated_text,
                    top_tokens,
                )

                generations.append(generation)

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
        return generations, batch if not stopped else None

    def warmup(self, batches: List[CausalLMBatch]) -> None:
        # prefill
        _, prefill_batch = self.generate_token([batches.pop(0)])
        # decode
        _, decode_batch = self.generate_token([prefill_batch])
        # shifts
        self.shifting_warmup(decode_batch)

        # if decode bs is 1 warmup ends here
        if len(batches) == 0:
            return
        
        # prefill
        _, prefill_batch = self.generate_token([batches.pop(0)])
        # concatenate and decode
        _, decode_batch = self.generate_token([decode_batch, prefill_batch])
        # decodes
        while decode_batch is not None:
            _, decode_batch = self.generate_token([decode_batch])

    def shifting_warmup(self, batch: CausalLMBatch) -> None:
        chunk_sizes = CHUNK_SIZES.copy()
        chunk_sizes.extend([-chunk for chunk in chunk_sizes])

        for chunk in chunk_sizes:
            batch.merge_kv_cache_if_needed(batch.batch_size, chunk)
            batch.realign(batch.batch_size, chunk, 0)
            batch.split_kv_cache_if_needed(True)
