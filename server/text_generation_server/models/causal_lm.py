import os
import tempfile

from text_generation_server.utils.tokens import batch_top_tokens
import torch

from dataclasses import dataclass
from opentelemetry import trace
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerBase, AutoConfig
from typing import Optional, Tuple, List, Type, Dict
from habana_frameworks.torch.hpu import wrap_in_hpu_graph
import habana_frameworks.torch as htorch
from contextlib import nullcontext
from optimum.habana.utils import HabanaProfile

from optimum.habana.transformers.generation import MODELS_OPTIMIZED_WITH_STATIC_SHAPES
from optimum.habana.checkpoint_utils import (
    get_repo_root,
    model_on_meta,
    write_checkpoints_json,
)

from text_generation_server.models import Model
from text_generation_server.models.types import (
    Batch,
    PrefillTokens,
    Generation,
    GeneratedText,
    TopTokens,
)
from text_generation_server.pb import generate_pb2
from text_generation_server.utils import HeterogeneousNextTokenChooser, StoppingCriteria, Sampling
from loguru import logger

tracer = trace.get_tracer(__name__)

@dataclass
class CausalLMBatch(Batch):
    batch_id: int
    requests: List[generate_pb2.Request]
    requests_idx_mapping: Dict[int, int]

    # Decoder values
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    past_key_values: Optional[List[Tuple]]

    # All tokens
    all_input_ids: List[torch.Tensor]

    # Lengths of all generations present in the batch
    input_lengths: List[int]
    prefix_offsets: List[int]
    read_offsets: List[int]

    # Generation helpers
    next_token_chooser: HeterogeneousNextTokenChooser
    stopping_criterias: List[StoppingCriteria]
    top_n_tokens: List[int]
    top_n_tokens_tensor: torch.Tensor

    # Metadata used for padding
    max_input_length: int
    padding_right_offset: int

    # Maximum number of tokens this batch will grow to
    max_tokens: int

    # Past metadata
    keys_head_dim_last: bool = True

    def to_pb(self) -> generate_pb2.CachedBatch:
        return generate_pb2.CachedBatch(
            id=self.batch_id,
            request_ids=[r.id for r in self.requests],
            size=len(self),
            max_tokens=self.max_tokens,
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
        inputs = []
        next_token_chooser_parameters = []
        stopping_criterias = []
        top_n_tokens = []
        prefix_offsets = []
        read_offsets = []
        requests_idx_mapping = {}
        input_lengths = []

        # Parse batch
        max_truncation = 0
        padding_right_offset = 0
        max_decode_tokens = 0

        # TODO: this should be set to rust side `max_total_tokens`,
        # (see https://github.com/huggingface/text-generation-inference/blob/main/launcher/src/main.rs#L177)
        # but TGI does not offer an API to expose this variable to python, as this variable
        # is handled by the client but it appears the model is initialized by the server.
        # An alternative could be to initialize the buffers during warmup.
        # Dummy
        max_total_tokens = int(os.getenv("MAX_TOTAL_TOKENS", "0"))
        logger.info("MAX_TOTAL_TOKENS = {}".format(max_total_tokens))

        for i, r in enumerate(pb.requests):
            requests_idx_mapping[r.id] = i
            inputs.append(r.inputs)
            next_token_chooser_parameters.append(r.parameters)
            stopping_criteria = StoppingCriteria.from_pb(r.stopping_parameters, tokenizer)
            stopping_criterias.append(stopping_criteria)
            top_n_tokens.append(r.top_n_tokens)
            max_truncation = max(max_truncation, r.truncate)
            max_decode_tokens += stopping_criteria.max_new_tokens
            padding_right_offset = max(padding_right_offset, stopping_criteria.max_new_tokens)

        next_token_chooser = HeterogeneousNextTokenChooser.from_pb(
            next_token_chooser_parameters, dtype, device
        )

        tokenized_inputs = tokenizer(
            inputs,
            return_tensors="pt",
            padding="max_length",
            return_token_type_ids=False,
            truncation=True,
            max_length=max_truncation,
        )

        for _ in pb.requests:
            input_len = tokenized_inputs["input_ids"].shape[1]
            input_lengths.append(input_len)
            prefix_offsets.append(input_len - 5)
            read_offsets.append(input_len)

        max_input_length = max(input_lengths)
        if max_total_tokens == 0:
            max_total_tokens = max_input_length
        max_tokens = len(inputs) * max_input_length + max_decode_tokens
        if is_optimized_for_gaudi and max_total_tokens > max_input_length:
            # pad to max_total_tokens in case max_new_token changes per request and triggers new hpu graph generation
            padding_right_offset = max_total_tokens - max_input_length

        input_ids = tokenized_inputs["input_ids"]
        attention_mask = tokenized_inputs["attention_mask"]
        # only move model inputs to device
        attention_mask = attention_mask.to(device)

        if is_optimized_for_gaudi:
            input_ids_cpu = torch.nn.functional.pad(
                input_ids, (0, padding_right_offset), value=tokenizer.pad_token_id
            )
            input_ids = input_ids_cpu.to(device)
            attention_mask = torch.nn.functional.pad(attention_mask, (0, padding_right_offset), value=0)
            all_input_ids = input_ids_cpu.T.split(1, dim=1)
        else:
            all_input_ids = input_ids.clone().T.split(1, dim=1)
            input_ids = input_ids.to(device)

        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        htorch.core.mark_step()

        top_n_tokens_tensor = torch.tensor(top_n_tokens, device=device, dtype=torch.int64)

        return cls(
            batch_id=pb.id,
            requests=pb.requests,
            requests_idx_mapping=requests_idx_mapping,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            all_input_ids=list(all_input_ids),
            input_lengths=input_lengths,
            prefix_offsets=prefix_offsets,
            read_offsets=read_offsets,
            next_token_chooser=next_token_chooser,
            stopping_criterias=stopping_criterias,
            top_n_tokens=top_n_tokens,
            top_n_tokens_tensor=top_n_tokens_tensor,
            max_input_length=max_input_length,
            padding_right_offset=padding_right_offset,
            max_tokens=max_tokens,
        )

    @tracer.start_as_current_span("filter")
    def filter(self, request_ids: List[int], is_optimized_for_gaudi: bool = False) -> Optional["CausalLMBatch"]:
        if len(request_ids) == 0:
            raise ValueError("Batch must have at least one request")
        if len(request_ids) == len(self):
            return self

        keep_indices = []

        # New values after filtering
        requests_idx_mapping = {}
        requests = []
        input_lengths = []
        prefix_offsets = []
        read_offsets = []
        all_input_ids = []
        max_input_length = 0

        stopping_criterias = []
        top_n_tokens = []

        total_remaining_decode_tokens = 0
        new_padding_right_offset = 0

        for i, request_id in enumerate(request_ids):
            idx = self.requests_idx_mapping[request_id]
            requests_idx_mapping[request_id] = i
            keep_indices.append(idx)

            requests.append(self.requests[idx])
            prefix_offsets.append(self.prefix_offsets[idx])
            read_offsets.append(self.read_offsets[idx])
            all_input_ids.append(self.all_input_ids[idx])

            request_input_length = self.input_lengths[idx]
            input_lengths.append(request_input_length)
            max_input_length = max(max_input_length, request_input_length)

            stopping_criteria = self.stopping_criterias[idx]
            stopping_criterias.append(stopping_criteria)
            top_n_tokens.append(self.top_n_tokens[idx])
            remaining_decode_tokens = stopping_criteria.max_new_tokens - stopping_criteria.current_tokens
            total_remaining_decode_tokens += remaining_decode_tokens
            new_padding_right_offset = max(new_padding_right_offset, remaining_decode_tokens)

        # Apply indices to input_ids, attention mask, past key values and other items that need to be cached
        input_ids = self.input_ids[keep_indices]
        position_ids = self.position_ids[keep_indices]
        next_token_chooser = self.next_token_chooser.filter(keep_indices)
        if is_optimized_for_gaudi:
            self.attention_mask = self.attention_mask[keep_indices]
        else:
            self.attention_mask = self.attention_mask[
                keep_indices,
                -(self.padding_right_offset + max_input_length) : (
                    self.attention_mask.shape[1] - self.padding_right_offset
                )
                + new_padding_right_offset,
            ]

        # Ensure that past_key_values tensors can be updated in-place
        kv_tuple = False
        if type(self.past_key_values[0]) == tuple:
            self.past_key_values = [list(layer) for layer in self.past_key_values]
            kv_tuple = True

        # Update tensors in-place to allow incremental garbage collection
        past_kv_length = max_input_length - 1
        for layer in self.past_key_values:
            past_keys, past_values = layer
            past_keys_dims = len(past_keys.shape)
            if past_keys_dims == 3:
                # Force past to be of dim [self_size, num_heads, ...] for easy indexing
                past_keys = past_keys.view(len(self), -1, *past_keys.shape[-2:])
                past_values = past_values.view(len(self), -1, *past_values.shape[-2:])
            if is_optimized_for_gaudi:
                layer[0] = past_keys[keep_indices]
                del past_keys
                layer[1] = past_values[keep_indices]
                del past_values
            else:
                if self.keys_head_dim_last:
                    layer[0] = past_keys[keep_indices, :, -past_kv_length:, :]
                else:
                    layer[0] = past_keys[keep_indices, :, :, -past_kv_length:]
                del past_keys
                layer[1] = past_values[keep_indices, :, -past_kv_length:, :]
                del past_values
            if past_keys_dims == 3:
                layer[0] = layer[0].view(layer[0].shape[0] * layer[0].shape[1], *layer[0].shape[-2:])
                layer[1] = layer[1].view(layer[1].shape[0] * layer[1].shape[1], *layer[1].shape[-2:])

        top_n_tokens_tensor = self.top_n_tokens_tensor[keep_indices]
        max_tokens = len(request_ids) * max_input_length + total_remaining_decode_tokens

        if kv_tuple:
            self.past_key_values = tuple([tuple(layer) for layer in self.past_key_values])

        self.requests = requests
        self.requests_idx_mapping = requests_idx_mapping
        self.input_ids = input_ids
        self.position_ids = position_ids
        self.all_input_ids = all_input_ids
        self.input_lengths = input_lengths
        self.prefix_offsets = prefix_offsets
        self.read_offsets = read_offsets
        self.next_token_chooser = next_token_chooser
        self.stopping_criterias = stopping_criterias
        self.top_n_tokens = top_n_tokens
        self.top_n_tokens_tensor = top_n_tokens_tensor
        self.max_input_length = max_input_length
        self.padding_right_offset = new_padding_right_offset
        self.max_tokens = max_tokens

        return self

    @classmethod
    @tracer.start_as_current_span("concatenate")
    def concatenate(cls, batches: List["CausalLMBatch"], is_optimized_for_gaudi: bool = False) -> "CausalLMBatch":
        # Used for padding
        total_batch_size = 0
        max_input_length = 0
        padding_right_offset = 0
        max_total_tokens = 0
        for batch in batches:
            total_batch_size += len(batch)
            max_input_length = max(max_input_length, batch.max_input_length)
            padding_right_offset = max(padding_right_offset, batch.padding_right_offset)
            max_total_tokens = max(max_total_tokens, batch.max_input_length + batch.padding_right_offset)

        if is_optimized_for_gaudi and max_total_tokens > max_input_length:
            padding_right_offset = max_total_tokens - max_input_length

        # Batch attributes
        requests = []
        requests_idx_mapping = {}
        input_lengths = []
        prefix_offsets = []
        read_offsets = []
        all_input_ids = []
        next_token_chooser_parameters = []
        stopping_criterias = []
        top_n_tokens = []
        max_tokens = 0

        # Batch tensors
        input_ids = None
        attention_mask = None
        position_ids = None
        past_key_values = []
        top_n_tokens_tensor = None

        # Used for slicing correctly inside the tensors
        # Equivalent to a cumsum on batch sizes
        start_index = 0
        for i, batch in enumerate(batches):
            requests.extend(batch.requests)
            input_lengths.extend(batch.input_lengths)
            prefix_offsets.extend(batch.prefix_offsets)
            read_offsets.extend(batch.read_offsets)
            all_input_ids.extend(batch.all_input_ids)
            next_token_chooser_parameters.extend([r.parameters for r in batch.requests])
            stopping_criterias.extend(batch.stopping_criterias)
            top_n_tokens.extend(batch.top_n_tokens)

            if i == 0:
                requests_idx_mapping = batch.requests_idx_mapping
            else:
                # We need to offset the mapping for each batch by the cumulative batch size
                for k, v in batch.requests_idx_mapping.items():
                    requests_idx_mapping[k] = v + start_index

            # Slicing end index for this batch
            end_index = start_index + len(batch)

            # We only concatenate batches that did at least one step
            if batch.past_key_values is None:
                raise ValueError("only concatenate prefilled batches")

            # Create empty tensor
            # input_ids is always of shape [batch_size, 1]
            # We do not need to pad it
            if input_ids is None:
                input_ids = batch.input_ids.new_empty((total_batch_size, max_total_tokens))
            # Copy to correct indices
            input_ids[start_index:end_index] = batch.input_ids

            # Create padded tensor
            if attention_mask is None:
                attention_mask = batch.attention_mask.new_zeros(
                    (total_batch_size, max_input_length + padding_right_offset),
                )

            if top_n_tokens_tensor is None:
                top_n_tokens_tensor = batches[0].top_n_tokens_tensor.new_zeros(
                    total_batch_size,
                )
            top_n_tokens_tensor[start_index:end_index] = batch.top_n_tokens_tensor

            # We need to slice the attention mask to remove padding from previous steps
            # and to remove unused allocated space
            left_offset = max_input_length - batch.max_input_length
            batch_left_offset = batch.attention_mask.shape[1] - batch.max_input_length - batch.padding_right_offset
            attention_mask[start_index:end_index, left_offset:-padding_right_offset] = batch.attention_mask[
                :,
                batch_left_offset : -batch.padding_right_offset,
            ]

            # Create empty tensor
            # position_ids is always of shape [batch_size, 1]
            if position_ids is None:
                position_ids = batch.position_ids.new_empty((total_batch_size, 1))
            position_ids[start_index:end_index] = batch.position_ids

            # Shenanigans to get dimensions because BLOOM outputs a past with a different shape
            # BLOOM Keys:   [batch_size * num_heads, head_dim, seq_length]
            # BLOOM Values: [batch_size * num_heads, seq_length, head_dim]
            # And ensure that we can update tensors in-place
            kv_tuple = False
            past_key_values_dims = len(batch.past_key_values[0][0].shape)
            if type(batch.past_key_values[0]) == tuple:
                batch.past_key_values = [
                    [t.view(len(batch), -1, *t.shape[-2:]) for t in layer] for layer in batch.past_key_values
                ]
                kv_tuple = True
            elif past_key_values_dims == 3:
                for layer in batch.past_key_values:
                    for k, t in enumerate(layer):
                        layer[k] = t.view(len(batch), -1, *t.shape[-2:])

            # Add eventual padding tokens that were added while concatenating
            max_tokens += batch.max_tokens + (max_input_length - batch.max_input_length) * len(batch)

            start_index = end_index

        next_token_chooser = HeterogeneousNextTokenChooser.from_pb(
            next_token_chooser_parameters,
            dtype=batches[0].next_token_chooser.dtype,
            device=batches[0].next_token_chooser.device,
        )

        first_past_kvs = batches[0].past_key_values
        _, num_heads, _, head_dim = first_past_kvs[0][1].shape
        padded_sequence_length = (
            max_input_length + padding_right_offset if is_optimized_for_gaudi else max_input_length - 1
        )
        padded_past_values_shape = (
            total_batch_size,
            num_heads,
            padded_sequence_length,
            head_dim,
        )

        if batches[0].keys_head_dim_last:
            padded_past_keys_shape = padded_past_values_shape
        else:
            # seq_length is last for BLOOM
            padded_past_keys_shape = (
                total_batch_size,
                num_heads,
                head_dim,
                padded_sequence_length,
            )

        # Iterate over attention layers
        # Concatenate past key values layer by layer to allow incremental garbage collection
        for j in range(len(first_past_kvs)):
            padded_past_keys = first_past_kvs[j][0].new_zeros(padded_past_keys_shape)
            start_index = 0
            for batch in batches:
                past_keys = batch.past_key_values[j][0]
                # Clear reference to the original tensor
                batch.past_key_values[j][0] = None

                # Slicing end index for this batch
                end_index = start_index + len(batch)
                # We slice the keys to remove the padding from previous batches
                past_seq_len = batch.max_input_length - 1
                # recaculate the offset
                left_offset = max_input_length - batch.max_input_length
                batch_left_offset = batch.attention_mask.shape[1] - batch.max_input_length - batch.padding_right_offset

                if batch.keys_head_dim_last:
                    padded_past_keys[
                        start_index:end_index, :, left_offset : left_offset + past_seq_len, :
                    ] = past_keys[:, :, batch_left_offset : batch_left_offset + past_seq_len, :]
                else:
                    # BLOOM case
                    padded_past_keys[
                        start_index:end_index, :, :, left_offset : left_offset + past_seq_len
                    ] = past_keys[:, :, :, batch_left_offset : batch_left_offset + past_seq_len]
                del past_keys

                start_index = end_index

            padded_past_values = first_past_kvs[j][1].new_zeros(padded_past_values_shape)
            start_index = 0
            for batch in batches:
                past_values = batch.past_key_values[j][1]
                # Clear reference to the original tensor
                batch.past_key_values[j][1] = None

                # Slicing end index for this batch
                end_index = start_index + len(batch)
                # We slice the past values to remove the padding from previous batches
                past_seq_len = batch.max_input_length - 1
                # recaculate the offset
                left_offset = max_input_length - batch.max_input_length
                batch_left_offset = batch.attention_mask.shape[1] - batch.max_input_length - batch.padding_right_offset

                padded_past_values[
                    start_index:end_index, :, left_offset : left_offset + past_seq_len, :
                ] = past_values[:, :, batch_left_offset : batch_left_offset + past_seq_len, :]
                del past_values

                # Update values
                start_index = end_index

            if past_key_values_dims == 3:
                padded_past_keys = padded_past_keys.view(
                    padded_past_keys.shape[0] * padded_past_keys.shape[1], *padded_past_keys.shape[-2:]
                )
                padded_past_values = padded_past_values.view(
                    padded_past_values.shape[0] * padded_past_values.shape[1], *padded_past_values.shape[-2:]
                )

            if kv_tuple:
                past_key_values.append((padded_past_keys, padded_past_values))
            else:
                past_key_values.append([padded_past_keys, padded_past_values])

        if kv_tuple:
            past_key_values = tuple(past_key_values)

        return cls(
            batch_id=batches[0].batch_id,
            requests=requests,
            requests_idx_mapping=requests_idx_mapping,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            all_input_ids=all_input_ids,
            input_lengths=input_lengths,
            prefix_offsets=prefix_offsets,
            read_offsets=read_offsets,
            next_token_chooser=next_token_chooser,
            stopping_criterias=stopping_criterias,
            top_n_tokens=top_n_tokens,
            top_n_tokens_tensor=top_n_tokens_tensor,
            max_input_length=max_input_length,
            padding_right_offset=padding_right_offset,
            keys_head_dim_last=batches[0].keys_head_dim_last,
            max_tokens=max_tokens,
        )

    def __len__(self):
        return len(self.requests)


class CausalLM(Model):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = torch.device("hpu")

        dtype = torch.bfloat16 if dtype is None else dtype

        from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi

        adapt_transformers_to_gaudi()

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            padding_side="left",
            truncation_side="left",
        )

        model_kwargs = {
            "revision": revision,
        }

        world_size = int(os.getenv("WORLD_SIZE", "1"))
        rank = int(os.getenv("RANK"), 0)
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
            ds_inference_kwargs["enable_cuda_graph"] = self.enable_hpu_graph

            if load_to_meta:
                # model loaded to meta is managed differently
                checkpoints_json = tempfile.NamedTemporaryFile(suffix=".json", mode="+w")
                write_checkpoints_json(model_id, local_rank, checkpoints_json)
                ds_inference_kwargs["checkpoint"] = checkpoints_json.name
            model = deepspeed.init_inference(model, **ds_inference_kwargs)
            model = model.module
        else:
            get_repo_root(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                revision=revision,
                torch_dtype=dtype,
            )
            model = model.eval().to(device)
            #wrap in hpu_graph only if self.enable_hpu_graph is set
            if self.enable_hpu_graph:
                model = wrap_in_hpu_graph(model)

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
        self.profiling_warmup_steps = int(os.getenv("PROF_WARMUPSTEP", "0"))
        self.profiling_steps = int(os.getenv("PROF_STEP", "5"))
        output_dir = os.getenv("PROF_PATH", "/tmp/hpu_profile")
        self.hb_profer = HabanaProfile(
            warmup=self.profiling_warmup_steps, active=self.profiling_steps, output_dir=output_dir
        )
        if self.profiling_warmup_steps > 0:
            self.hb_profer_started = True
            self.hb_profer.start()
        else:
            self.hb_profer = None
            self.hb_profer_started = False
        self.step = 0

    @property
    def batch_type(self) -> Type[CausalLMBatch]:
        return CausalLMBatch

    def decode(self, generated_ids: List[int]) -> str:
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

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
        outputs = self.model.forward(**kwargs)
        return outputs.logits, outputs.past_key_values

    @tracer.start_as_current_span("generate_token")
    def generate_token(self, batch: CausalLMBatch) -> Tuple[List[Generation], Optional[CausalLMBatch]]:
        self.step = self.step + 1
        if self.hb_profer_started == True and self.step > self.profiling_warmup_steps + self.profiling_steps:
            self.hb_profer.stop()
            self.hb_profer_started = False

        if self.is_optimized_for_gaudi:
            token_idx = torch.tensor(batch.attention_mask.shape[-1] - batch.padding_right_offset).to(self.device)
            attention_mask = batch.attention_mask

        else:
            token_idx = None
            # slice the attention mask to the correct shape
            attention_mask = batch.attention_mask[:, : -batch.padding_right_offset]
        prefill = batch.past_key_values is None
        if batch.past_key_values:
            if token_idx is not None:
                input_ids = torch.index_select(batch.input_ids, 1, token_idx - 1)
        else:
            input_ids = batch.input_ids

        logits, past = self.forward(
            input_ids,
            attention_mask,
            batch.position_ids,
            token_idx,
            batch.past_key_values,
            bypass_hpu_graph = prefill and self.limit_hpu_graph if self.enable_hpu_graph else None
        )

        # Results
        generations: List[Generation] = []
        stopped = True

        # Select next token
        input_length = batch.input_lengths[0]
        if self.is_optimized_for_gaudi and logits.shape[-2] > 1:
            next_input_ids, next_token_logprobs, logprobs = batch.next_token_chooser(
                batch.input_ids[:, :token_idx], logits[:, input_length - 1 : input_length, :].squeeze(-2)
            )
        else:
            next_input_ids, next_token_logprobs, logprobs = batch.next_token_chooser(
                batch.input_ids[:, :token_idx], logits.squeeze(-2)
            )

        batch_top_token_ids, batch_top_token_logprobs = batch_top_tokens(
            batch.top_n_tokens,
            batch.top_n_tokens_tensor,
            logprobs,
        )

        htorch.core.mark_step()
        logits = logits.to("cpu")

        next_token_logprobs = next_token_logprobs.tolist()
        next_token_ids = next_input_ids

        # Zipped iterator
        iterator = zip(
            batch.requests,
            batch.input_lengths,
            batch.prefix_offsets,
            batch.read_offsets,
            logits,
            batch.next_token_chooser.do_sample,
            batch.next_token_chooser.seeds,
            batch.stopping_criterias,
            batch.all_input_ids,
            batch.top_n_tokens,
            next_token_ids,
            next_token_logprobs,
            batch_top_token_ids,
            batch_top_token_logprobs,
        )
        # For each member of the batch
        for i, (
            request,
            input_length,
            prefix_offset,
            read_offset,
            logits,
            do_sample,
            seed,
            stopping_criteria,
            all_input_ids,
            top_n_tokens,
            next_token_id,
            next_token_logprob,
            top_token_ids,
            top_token_logprobs,
        ) in enumerate(iterator):
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
                        all_input_ids[new_input_length - stopping_criteria.current_tokens : new_input_length, 0]
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
                    prefill_token_ids = all_input_ids[0 : new_input_length - 1]
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

            batch.all_input_ids[i] = all_input_ids
            batch.input_lengths[i] = new_input_length
            batch.prefix_offsets[i] = prefix_offset
            batch.read_offsets[i] = read_offset
            batch.max_input_length = max(batch.max_input_length, new_input_length)

        if token_idx is None:
            batch.input_ids[:, 0] = next_token_ids[:, 0]
        else:
            batch.input_ids[:, token_idx] = next_token_ids
        # We finished all generations in the batch; there is no next batch
        if stopped:
            if self.hb_profer_started == True:
                self.hb_profer.step()
            return generations, None

        # Slice unused values from prefill, use it to store next token
        if token_idx is None:
            batch.input_ids = batch.input_ids[:, :1]

        # Update attention_mask as we added a new token to input_ids
        if self.is_optimized_for_gaudi:
            batch.attention_mask.index_fill_(1, token_idx, 1)
        else:
            batch.attention_mask[:, -batch.padding_right_offset] = 1
        # Decrease right offset
        batch.padding_right_offset -= 1

        # Update position_ids
        if prefill:
            batch.position_ids = batch.position_ids[:, token_idx - 1 : token_idx] + 1
        else:
            batch.position_ids += 1
        # Update past key values
        batch.past_key_values = past
        if self.hb_profer_started == True:
            self.hb_profer.step()

        return generations, batch
