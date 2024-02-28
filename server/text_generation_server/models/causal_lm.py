import os
import tempfile
import itertools
import time
import glob

import torch

from dataclasses import dataclass
from opentelemetry import trace
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerBase, AutoConfig
from typing import Optional, Tuple, List, Type, Dict

import text_generation_server.habana_quantization_env as hq_env
import habana_frameworks.torch as htorch
from habana_frameworks.torch.hpu import wrap_in_hpu_graph
from contextlib import nullcontext
from optimum.habana.utils import HabanaProfile, to_gb_rounded

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
from loguru import logger
from functools import wraps


tracer = trace.get_tracer(__name__)

if 'GRAPH_VISUALIZATION' in os.environ:
    for f in glob.glob('.graph_dumps/*'):
        os.remove(f)

MAX_TOTAL_TOKENS = int(os.getenv("MAX_TOTAL_TOKENS", "0"))
BATCH_BUCKET_SIZE = int(os.environ.get('BATCH_BUCKET_SIZE', 8))
PAD_SEQUENCE_TO_MULTIPLE_OF = int(os.environ.get('PAD_SEQUENCE_TO_MULTIPLE_OF', 128))
PREFILL_BATCH_BUCKET_SIZE = int(os.environ.get('PREFILL_BATCH_BUCKET_SIZE', 4))
DBG_TRACE_FILENAME = os.environ.get('DBG_TRACE_FILENAME')
START_TS = None


def count_hpu_graphs():
    return len(glob.glob('.graph_dumps/*PreGraph*'))


def dbg_trace(tag, txt):
    global START_TS
    if DBG_TRACE_FILENAME is not None and int(os.getenv("RANK", 0)) == 0:
        if START_TS is None:
            START_TS = time.perf_counter()
        time_offset = time.perf_counter() - START_TS
        mem_stats = htorch.hpu.memory.memory_stats()
        mem_used = to_gb_rounded(mem_stats['InUse'])
        max_mem_used = to_gb_rounded(mem_stats['MaxInUse'])
        print(f'ts:{time_offset:.3f}s g:{count_hpu_graphs()} mu:{mem_used:.1f}GB '
              f'mmu:{max_mem_used:.1f}GB | {tag} | {txt}', flush=True, file=open(DBG_TRACE_FILENAME, 'a'))


def round_up(number, k):
    return (number + k - 1) // k * k


def prepare_memory(new_bs, tensor, inplace):
    if inplace:
        return tensor
    else:
        return tensor.new_empty((new_bs,) + tensor.shape[1:])


def move_data(dst_tensor, chunk_size, indices, src_tensors):
    batch_dim = 0
    bs = dst_tensor.size(batch_dim)
    assert bs % chunk_size == 0, 'Batch dim must be divisible by chunk size!'
    result = dst_tensor
    if chunk_size > 1:
        dst_tensor = dst_tensor.view(bs // chunk_size, chunk_size, *dst_tensor.shape[1:])
    htorch.core.mark_step()
    for ind, src_t in zip(indices, src_tensors):
        if chunk_size > 1:
            src_t = src_t.view(bs // chunk_size, chunk_size, *src_t.shape[1:])
        for dst_idx, src_idx in ind:
            src_data = torch.index_select(src_t, batch_dim, src_idx)
            dst_tensor.index_copy_(batch_dim, dst_idx, src_data)
            htorch.core.mark_step()
    return result


def generate_shift_chunks(offset):
    chunk_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    result = []
    while offset != 0:
        sign = 1 if offset > 0 else -1
        best_chunk = min((abs(offset - sign * c), sign * c) for c in chunk_sizes)[1]
        result.append(best_chunk)
        offset = offset - best_chunk
    return result


def roll(tensor, dim, chunks):
    dbg_trace('ROLL', f'shape:{list(tensor.shape)} dim:{dim} chunks:{chunks}')
    for c in chunks:
        tensor = torch.roll(tensor, c, dim)
        htorch.core.mark_step()
    return tensor


def shift(tensor, dim, offset):
    assert dim < 0, 'Only negative dims are supported'
    if offset == 0:
        return tensor
    chunks = generate_shift_chunks(offset)
    tensor = roll(tensor, dim, chunks)
    return tensor


def shift_all(srcs, dim, offsets):
    return [shift(src, dim, offset) for src, offset in zip(srcs, offsets)]


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


def pad_tensors(tensors, paddings, dim, value):
    for i, (tensor, padding) in enumerate(zip(tensors, paddings)):
        if padding > 0:
            pad_shape = (0, 0, 0, padding) if dim == -2 else (0, padding)
            tensors[i] = torch.nn.functional.pad(tensor, pad_shape, value=value)
            htorch.core.mark_step()
    return tensors


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

    @classmethod
    def recombine(cls, batches: List["CausalLMBatch"], pad_token_id: int) -> "CausalLMBatch":
        total_requests = sum(len(b) for b in batches)
        new_bs = round_up(total_requests, BATCH_BUCKET_SIZE)
        batch_id = batches[0].batch_id
        device = batches[0].input_ids.device

        input_lengths = [b.input_length for b in batches]
        max_input_length = max(input_lengths)
        offsets = [max_input_length - b.input_length for b in batches]
        padding = [b.right_padding for b in batches]
        # For prefill there is a space allocated only for first token
        # Need to add padding to the max total tokens before first decode
        extra_padding = [MAX_TOTAL_TOKENS - b.seq_length for b in batches]

        moves_needed = [total_requests - len(b) if b.batch_size == new_bs else total_requests for b in batches]
        target_batch_idx = min(enumerate(moves_needed), key=lambda idx_val: idx_val[1])[0]

        # TODO: Add support for changing max seq len, i.e. due to output length bucketing
        # FIXME: max_seq_len for non optimized code
        if len(batches) > 1:
            scenario = 'CONCAT'
        elif batches[target_batch_idx].batch_size != new_bs:
            scenario = 'RESHAPE'
        elif padding[target_batch_idx] <= 0:
            scenario = 'SHIFT'
            offsets = [b.max_input_length - max_input_length for b in batches]
            max_input_length = max(b.max_input_length for b in batches)
        else:
            # Nothing to do
            return batches[0]

        inplace = (batches[target_batch_idx].batch_size == new_bs)

        dbg_trace(
            scenario, f'bs:{[b.batch_size for b in batches]}->{new_bs}'
                      f' reqs:{[len(b) for b in batches]}'
                      f' offsets:{offsets}'
                      f' input_lengths:{input_lengths}'
                      f' cur_padding:{padding}'
                      f' inplace:{inplace}')

        grouped_requests = [[req for req in batch.requests] for batch in batches]
        flat_requests = list(itertools.chain(*grouped_requests))
        if inplace:
            # The data is already present in the batch. No need to move it
            grouped_requests[target_batch_idx] = []
            free_indices = batches[target_batch_idx].free_indices()
        else:
            free_indices = itertools.count(0)

        def to_tensors(ind): return (torch.tensor(ind[0], device=device), torch.tensor(ind[1], device=device))
        indices = [[to_tensors(req.update_idx(next(free_indices))) for req in batch_reqs]
                   for batch_reqs in grouped_requests]

        chunk_size = batches[0].past_key_values[0][0].size(0) // batches[0].batch_size
        num_layers = len(batches[0].past_key_values)
        past_key_values_type = type(batches[0].past_key_values)

        seq_dim = -1
        if batches[0].past_key_values[0][0].size(-1) != batches[0].past_key_values[0][1].size(-1):
            # Case for Bloom
            key_dim = -1
        else:
            key_dim = -2
        value_dim = -2

        for b in batches:
            b.past_key_values = list(b.past_key_values)

        src = [b.input_ids for b in batches]
        for b in batches:
            del b.input_ids
        src = pad_tensors(src, extra_padding, seq_dim, pad_token_id)
        src = shift_all(src, seq_dim, offsets)
        input_ids = prepare_memory(new_bs, src[target_batch_idx], inplace)
        input_ids = move_data(input_ids, 1, indices, src)

        src = [b.attention_mask for b in batches]
        for b in batches:
            del b.attention_mask
        src = pad_tensors(src, extra_padding, seq_dim, 0)
        src = shift_all(src, seq_dim, offsets)
        attention_mask = prepare_memory(new_bs, src[target_batch_idx], inplace)
        attention_mask = move_data(attention_mask, 1, indices, src)

        src = [b.position_ids for b in batches]
        for b in batches:
            del b.position_ids
        position_ids = prepare_memory(new_bs, src[target_batch_idx], inplace)
        position_ids = move_data(position_ids, 1, indices, src)

        src = None
        src_keys = [[b.past_key_values[layer_num][0] for layer_num in range(num_layers)] for b in batches]
        src_values = [[b.past_key_values[layer_num][1] for layer_num in range(num_layers)] for b in batches]
        for b in batches:
            del b.past_key_values

        src_keys = [torch.stack(src) for src in src_keys]
        htorch.core.mark_step()
        src_keys = pad_tensors(src_keys, extra_padding, key_dim, 0)
        src_keys = shift_all(src_keys, key_dim, offsets)
        src_keys = [[t.squeeze(0).clone() for t in torch.split(src, 1)] for src in src_keys]
        htorch.core.mark_step()

        dst_keys = [prepare_memory(new_bs * chunk_size, prev, inplace) for prev in src_keys[target_batch_idx]]
        dst_keys = [move_data(dst_keys[layer_num], chunk_size, indices, [src[layer_num]
                              for src in src_keys]) for layer_num in range(num_layers)]

        src_values = [torch.stack(src) for src in src_values]
        htorch.core.mark_step()
        src_values = pad_tensors(src_values, extra_padding, value_dim, 0)
        src_values = shift_all(src_values, value_dim, offsets)
        src_values = [[t.squeeze(0).clone() for t in torch.split(src, 1)] for src in src_values]
        htorch.core.mark_step()

        dst_values = [prepare_memory(new_bs * chunk_size, prev, inplace) for prev in src_values[target_batch_idx]]
        dst_values = [move_data(dst_values[layer_num], chunk_size, indices, [src[layer_num]
                                for src in src_values]) for layer_num in range(num_layers)]

        past_key_values = past_key_values_type(zip(dst_keys, dst_values))

        top_n_tokens = [r.data.top_n_tokens for r in flat_requests]
        top_n_tokens_tensor = torch.tensor(top_n_tokens, device=device, dtype=torch.int64)
        next_token_chooser = HeterogeneousNextTokenChooser.from_pb(
            [r.data.parameters for r in flat_requests],
            batches[0].next_token_chooser.dtype,
            batches[0].next_token_chooser.device
        )

        max_seq_len = attention_mask.size(1)
        input_length = max_input_length

        htorch.core.mark_step()

        return cls(
            batch_id=batch_id,
            requests=flat_requests,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
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
        next_token_chooser = HeterogeneousNextTokenChooser.from_pb([r.parameters for r in pb.requests], dtype, device)

        # TODO: by tokenizing all inputs at once we loose information on actual input lengths
        # this means that we cannot shift inputs to the left after a long input sequence
        # was filtered out
        new_bs = round_up(len(requests), PREFILL_BATCH_BUCKET_SIZE)
        dummy_inputs = ["?"] * (new_bs - len(requests))
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
            bucket_size = round_up(input_len + 1, PAD_SEQUENCE_TO_MULTIPLE_OF) - 1
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

    def free_indices(self):
        used = set(req.idx for req in self.requests)
        for i in range(self.batch_size):
            if i in used:
                continue
            yield i


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
                        token_idx = torch.tensor(batch.attention_mask.shape[-1] - 1).to(self.device)
                    else:
                        token_idx = torch.tensor(batch.attention_mask.shape[-1] - batch.right_padding).to(self.device)
                else:
                    token_idx = None

                # Select next token
                input_length = batch.input_length
                if self.is_optimized_for_gaudi and logits.shape[-2] > 1:
                    next_token_ids, next_token_logprobs, logprobs = batch.next_token_chooser(
                        batch.input_ids[:, :token_idx], logits[:, input_length - 1: input_length, :].squeeze(-2)
                    )
                else:
                    next_token_ids, next_token_logprobs, logprobs = batch.next_token_chooser(
                        batch.input_ids[:, :token_idx], logits.squeeze(-2)
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
                    batch.input_ids.index_copy_(1, token_idx.cpu(), next_token_ids.unsqueeze(1))

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
                    batch.position_ids = batch.position_ids[:, token_idx - 1: token_idx] + 1
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
