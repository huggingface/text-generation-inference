import torch
import os
import math

from dataclasses import dataclass

from opentelemetry import trace
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerBase
from typing import Optional, Tuple, List, Type, Dict, Union


from text_generation_server.models import Model
from text_generation_server.models.types import (
    Batch,
    PrefillTokens,
    Generation,
    GeneratedText,
)
from text_generation_server.pb import generate_pb2
from text_generation_server.utils import StoppingCriteria
from text_generation_server.utils.tokens_heterogeneous import (
    HeterogeneousNextTokenChooser,
)

tracer = trace.get_tracer(__name__)


@dataclass
class VectorizedCausalLMBatch(Batch):
    batch_id: int
    requests: List[generate_pb2.Request]
    requests_idx_mapping: Dict[int, int]

    # Decoder values
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    past_key_values: Optional[
        List[Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]]]
    ]

    # All tokens
    input_ids: torch.Tensor

    # Lengths of all generations present in the batch
    input_lengths: List[int]
    offsets: List[Optional[int]]
    token_offsets: List[Optional[int]]

    # Generation helpers
    next_token_chooser: "HeterogeneousNextTokenChooser"
    stopping_criterias: List[StoppingCriteria]

    # Metadata used for padding
    max_input_length: int

    # Maximum number of tokens this batch will grow to
    max_tokens: int

    kv_cache_seq_dim: int = 2

    # Prefill the attention mask for the generated tokens
    attention_mask_fill_value=True

    # TODO: Get from requests (should these be lists?)
    details: bool = os.environ.get("RETURN_DETAILS") is not None
    generate_stream: bool = os.environ.get("GENERATE_STREAM") is not None

    def to_pb(self) -> generate_pb2.Batch:
        return generate_pb2.Batch(
            id=self.batch_id,
            requests=self.requests,
            size=len(self),
            max_tokens=self.max_tokens,
        )

    @classmethod
    def from_pb(
        cls,
        pb: generate_pb2.Batch,
        tokenizer: PreTrainedTokenizerBase,
        device: torch.device,
    ) -> "VectorizedCausalLMBatch":
        inputs = [r.inputs for r in pb.requests]
        offsets = [None] * len(inputs)
        token_offsets = [None] * len(inputs)
        requests_idx_mapping = {r.id: i for i, r in enumerate(pb.requests)}

        # Parse batch
        stopping_criterias = [
            StoppingCriteria.from_pb(r.stopping_parameters, tokenizer)
            for r in pb.requests
        ]
        max_new_tokens = (
            stopping_criteria.max_new_tokens for stopping_criteria in stopping_criterias
        )

        next_token_chooser = HeterogeneousNextTokenChooser.from_pb(
            [r.parameters for r in pb.requests], device
        )

        tokenized_inputs = tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
            truncation=True,
            max_length=max(r.truncate for r in pb.requests),
        ).to(device)

        input_lengths = tokenized_inputs["attention_mask"].sum(1)
        max_input_length = input_lengths.max().item()

        input_shape = (pb.size, max_input_length + max(max_new_tokens))

        # Allocate maximum attention_mask
        attention_mask = torch.empty(input_shape, dtype=torch.bool, device=device)
        # Copy tokenizer attention_mask into fully allocated attention_mask
        attention_mask[:, :max_input_length].copy_(tokenized_inputs["attention_mask"])
        attention_mask[:, max_input_length:].fill_(cls.attention_mask_fill_value)

        position_ids = attention_mask.cumsum(-1).sub_(1)
        position_ids[:, :max_input_length].relu_()

        input_ids = torch.empty(input_shape, dtype=torch.int64, device=device)
        input_ids[:, :max_input_length].copy_(tokenized_inputs["input_ids"])

        max_tokens = len(inputs) * max_input_length + sum(max_new_tokens)

        generate_stream = cls.generate_stream or any(
            stopping_criteria.stop_sequence_criterias
            for stopping_criteria in stopping_criterias
        )

        return cls(
            batch_id=pb.id,
            requests=pb.requests,
            requests_idx_mapping=requests_idx_mapping,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            input_ids=input_ids,
            input_lengths=input_lengths.tolist(),
            offsets=offsets,
            token_offsets=token_offsets,
            next_token_chooser=next_token_chooser,
            stopping_criterias=stopping_criterias,
            max_input_length=max_input_length,
            max_tokens=max_tokens,
            generate_stream=generate_stream,
        )

    @tracer.start_as_current_span("filter")
    def filter(
        self, requests: List[generate_pb2.Request]
    ) -> Optional["VectorizedCausalLMBatch"]:
        if len(requests) == 0:
            raise ValueError("Batch must have at least one request")
        if len(requests) == len(self):
            return self

        self.requests = requests
        keep_indices = [self.requests_idx_mapping[r.id] for r in self.requests]

        # New values after filtering
        self.requests_idx_mapping = {r.id: i for i, r in enumerate(self.requests)}
        self.input_lengths = [self.input_lengths[i] for i in keep_indices]
        self.offsets = [self.offsets[i] for i in keep_indices]
        self.token_offsets = [self.token_offsets[i] for i in keep_indices]

        self.next_token_chooser = HeterogeneousNextTokenChooser.from_pb(
            [r.parameters for r in self.requests], self.input_ids.device
        )

        self.stopping_criterias = [self.stopping_criterias[i] for i in keep_indices]
        remaining_decode_tokens = [
            stopping_criteria.max_new_tokens - stopping_criteria.current_tokens
            for stopping_criteria in self.stopping_criterias
        ]

        # Select the remaining indices and remove unnecessary padding
        max_input_length = max(self.input_lengths)
        sequence_slice = slice(
            self.max_input_length - max_input_length,
            self.max_input_length + max(remaining_decode_tokens),
        )
        self.max_input_length = max_input_length
        self.max_tokens = len(self.requests) * self.max_input_length + sum(
            remaining_decode_tokens
        )

        self.input_ids = self.input_ids[keep_indices, sequence_slice]
        self.position_ids = self.position_ids[keep_indices, sequence_slice]
        self.attention_mask = self.attention_mask[keep_indices, sequence_slice]

        self._filter_kv_caches(keep_indices, sequence_slice)

        return self

    def _filter_kv_caches(self, keep_indices, sequence_slice):
        tensors_to_update = []
        if self.past_key_values is not None:
            if not isinstance(self.past_key_values, (list, tuple)):
                raise NotImplementedError(
                    f"Unsupported kv cache type: {type(self.past_key_values)}"
                )
            for layer_kv in self.past_key_values:
                if isinstance(layer_kv, torch.Tensor):
                    tensors_to_update.append(layer_kv)
                elif isinstance(layer_kv, (list, tuple)):
                    tensors_to_update.extend(layer_kv)
                else:
                    raise NotImplementedError(
                        f"Unsupported layer  kv cache type: {type(layer_kv)}"
                    )

        kv_cache_slice = [
            keep_indices,
            *(slice(None) for _ in range(1, self.kv_cache_seq_dim)),
            sequence_slice,
        ]
        for tensor in tensors_to_update:
            # Update tensors in-place to allow incremental garbage collection
            tensor.data = tensor[kv_cache_slice]

    @classmethod
    @tracer.start_as_current_span("concatenate")
    def concatenate(
        cls, batches: List["VectorizedCausalLMBatch"]
    ) -> "VectorizedCausalLMBatch":
        if len(batches) == 0:
            raise ValueError("Cannot concatenate empty list.")
        requests = [request for batch in batches for request in batch.requests]
        batch_sizes = [len(batch.requests) for batch in batches]
        batch_size = sum(batch_sizes)

        end_indices = torch.tensor(batch_sizes).cumsum(0).tolist()
        start_indices = [0] + end_indices[:-1]

        input_lengths = [length for batch in batches for length in batch.input_lengths]
        offsets = [offset for batch in batches for offset in batch.offsets]
        token_offsets = [
            token_offset for batch in batches for token_offset in batch.token_offsets
        ]
        next_token_chooser = HeterogeneousNextTokenChooser.from_pb(
            [r.parameters for r in requests], batches[0].input_ids.device
        )
        stopping_criterias = [
            stopping_criteria
            for batch in batches
            for stopping_criteria in batch.stopping_criterias
        ]

        requests_idx_mapping = {
            k: v + start_index
            for batch, start_index in zip(batches, start_indices)
            for k, v in batch.requests_idx_mapping.items()
        }

        max_input_length = max(input_lengths)
        left_indices = [max_input_length - batch.max_input_length for batch in batches]

        input_shape = (
            batch_size,
            max_input_length
            + max(
                batch.input_ids.size(1) - batch.max_input_length for batch in batches
            ),
        )
        device = batches[0].input_ids.device

        # Allocate maximum attention_mask
        attention_mask = torch.empty(input_shape, dtype=torch.bool, device=device)
        attention_mask[:, :max_input_length].fill_(0)
        attention_mask[:, max_input_length:].fill_(cls.attention_mask_fill_value)

        input_ids = torch.empty(input_shape, dtype=torch.int64, device=device)
        # TODO : only needed for prefill
        input_ids[:, :max_input_length].fill_(0)

        for batch, start_index, end_index, left_index in zip(
            batches, start_indices, end_indices, left_indices
        ):
            attention_mask[start_index:end_index, left_index:max_input_length].copy_(
                batch.attention_mask[:, : batch.max_input_length]
            )
            input_ids[start_index:end_index, left_index:max_input_length].copy_(
                batch.input_ids[:, : batch.max_input_length]
            )

        position_ids = attention_mask.cumsum(-1).sub_(1)
        position_ids[:, :max_input_length].relu_()

        max_tokens = sum(
            batch.max_tokens + (max_input_length - batch.max_input_length) * len(batch)
            for batch in batches
        )

        kv_cache_seq_dim = batches[0].kv_cache_seq_dim
        past_key_values=cls._concatenate_key_values(batches, start_indices, end_indices, left_indices, max_input_length)

        return cls(
            batch_id=batches[0].batch_id,
            requests=requests,
            requests_idx_mapping=requests_idx_mapping,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            input_lengths=input_lengths,
            offsets=offsets,
            token_offsets=token_offsets,
            next_token_chooser=next_token_chooser,
            stopping_criterias=stopping_criterias,
            max_input_length=max_input_length,
            kv_cache_seq_dim=kv_cache_seq_dim,
            max_tokens=max_tokens,
        )

    @classmethod
    def _concatenate_key_values(cls, batches, start_indices, end_indices, left_indices, max_input_length):
        device = batches[0].input_ids.device
        batch_size = sum([len(batch.requests) for batch in batches])

        kv_formats = None
        for batch in batches:
            if batch.past_key_values is None:
                raise ValueError("Only concatenate prefilled batches")
            if not isinstance(batch.past_key_values, (list, tuple)):
                raise NotImplementedError(
                    f"Unsupported kv cache type: {type(batch.past_key_values)}"
                )
            if kv_formats is None:
                num_layers = len(batch.past_key_values)
                if num_layers == 0:
                    raise ValueError("Empty KV cache")
                kv_formats = [0] * num_layers
            elif len(batch.past_key_values) != len(kv_formats):
                raise ValueError("Num layers is not constant")
            for i, layer_kv in enumerate(batch.past_key_values):
                if isinstance(layer_kv, (list, tuple)):
                    kv_format = len(layer_kv)
                else:
                    kv_format = None
                if kv_formats[i] == 0:
                    if kv_format == 0:
                        raise ValueError("Empty KV cache")
                    kv_formats[i] = kv_format
                elif kv_formats[i] != kv_format:
                    raise ValueError("Incompatible KV cache format.")

        kv_cache_seq_dim = batches[0].kv_cache_seq_dim
        past_key_values = []
        for i, kv_format in enumerate(kv_formats):
            for j in range(1 if kv_format is None else kv_format):
                tensors_to_merge = [
                    batch.past_key_values[i]
                    if kv_format is None
                    else batch.past_key_values[i][j]
                    for batch in batches
                ]
                combined_shape = [batch_size] + list(tensors_to_merge[0].shape[1:])
                combined_shape[kv_cache_seq_dim] = max_input_length
                # Set to zero to avoid propagating nans in padded values.
                kv_cache = torch.zeros(
                    combined_shape, dtype=tensors_to_merge[0].dtype, device=device
                )
                for tensor, start_index, end_index, left_index in zip(
                    tensors_to_merge,
                    start_indices,
                    end_indices,
                    left_indices,
                ):
                    kv_cache[
                        [
                            slice(start_index, end_index),
                            *(slice(None) for _ in range(1, kv_cache_seq_dim)),
                            slice(left_index, max_input_length),
                        ]
                    ].copy_(tensor)
                if kv_format is None:
                    past_key_values.append(kv_cache)
                elif j == 0:
                    past_key_values.append([kv_cache])
                else:
                    past_key_values[-1].append(kv_cache)

        return


    def __len__(self):
        return len(self.requests)

class VectorizedCausalLM(Model):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        quantize: Optional[str] = None,
    ):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            dtype = torch.float16
        else:
            if quantize:
                raise ValueError("quantization is not available on CPU")

            device = torch.device("cpu")
            dtype = torch.float32

        tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, padding_side="left", truncation_side="left"
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            load_in_8bit=quantize == "bitsandbytes",
            trust_remote_code=True,
        )
        tokenizer.pad_token_id = (
            model.config.pad_token_id
            if model.config.pad_token_id is not None
            else model.config.eos_token_id
        )

        super().__init__(
            model=model,
            tokenizer=tokenizer,
            requires_padding=True,
            dtype=dtype,
            device=device,
        )

    @property
    def batch_type(self) -> Type[VectorizedCausalLMBatch]:
        return VectorizedCausalLMBatch

    def decode(self, generated_ids: List[int]) -> str:
        return self.tokenizer.decode(
            generated_ids, skip_special_tokens=True, cleanup_tokenization_spaces=False
        )

    def forward(self, batch:VectorizedCausalLMBatch):
        key_length = batch.max_input_length
        query_length = key_length if batch.past_key_values is None else 1
        input_ids = batch.input_ids[:, key_length - query_length : key_length]
        # Model Forward
        logits, batch.past_key_values, *_ = self.model.forward(
            input_ids=input_ids,
            attention_mask=batch.attention_mask[:, :key_length],
            position_ids=batch.position_ids[:, key_length - query_length : key_length],
            past_key_values=batch.past_key_values,
            return_dict=False,
            use_cache=True,
        )
        next_token_ids, logprobs = batch.next_token_chooser(
            input_ids, logits, batch.details
        )
        # Update batch
        # TODO: Why do we need all input ids?
        batch.input_ids[:, key_length].copy_(next_token_ids)
        batch.input_lengths = [length + 1 for length in batch.input_lengths]
        batch.max_input_length += 1

        return next_token_ids, logprobs

    @tracer.start_as_current_span("generate_token")
    def generate_token(
        self, batch: VectorizedCausalLMBatch
    ) -> Tuple[List[Generation], Optional[VectorizedCausalLMBatch]]:
        key_length = batch.max_input_length
        if key_length > batch.input_ids.size(1):
            raise RuntimeError("Cannot generate more than `max_tokens`.")
        is_prefill = batch.past_key_values is None

        next_token_ids, logprobs = self.forward(batch)

        if batch.generate_stream:
            # TODO: self.decode_token, offsets?
            next_token_texts = self.tokenizer.batch_decode(next_token_ids.tolist())

        if batch.details:
            token_logprobs = (
                logprobs[:, -1, :]
                .gather(1, next_token_ids.unsqueeze(1))
                .squeeze(1)
                .tolist()
            )
            if is_prefill:
                prefill_token_ids = batch.input_ids[:, :key_length].tolist()
                prefill_logprobs = (
                    logprobs.gather(2, batch.input_ids[:, 1:key_length, None])
                    .squeeze(2)
                    .tolist()
                )
                prefill_tokens = []
                for prefill_token_ids_, prefill_logprobs_, input_length in zip(
                    prefill_token_ids, prefill_logprobs, batch.input_lengths
                ):
                    # Input length has already been incremented so we subtract 1.
                    prefill_token_ids_ = prefill_token_ids_[-(input_length-1):]
                    prefill_texts = self.tokenizer.batch_decode(
                        prefill_token_ids_,
                        clean_up_tokenization_spaces=False,
                        skip_special_tokens=False,
                    )
                    prefill_tokens.append(
                        PrefillTokens(
                            prefill_token_ids_,
                            [math.nan, *prefill_logprobs_],
                            prefill_texts,
                        )
                    )

        # TODO: Vectorize some of this?

        generations: List[Generation] = []
        next_batch = None

        for i, next_token_id in enumerate(next_token_ids):
            next_token_text = next_token_texts[i] if batch.generate_stream else ""
            stopping_criterias = batch.stopping_criterias[i]
            stop, reason = stopping_criterias(
                next_token_id,
                next_token_text,
            )
            if stop:
                # Decode generated tokens
                # TODO: Same as stopping_criteria.current_output?
                output_text = self.decode(
                    batch.input_ids[
                        i,
                        batch.max_input_length
                        - stopping_criterias.current_tokens : batch.max_input_length,
                    ]
                )
                # TODO: Seed
                generated_text = GeneratedText(
                    output_text, stopping_criterias.current_tokens, reason, seed=None
                )
            else:
                # Keep request in the batch
                generated_text = None
                next_batch = batch

            generation = Generation(
                batch.requests[i].id,
                prefill_tokens[i] if batch.details and is_prefill else None,
                next_token_id,
                token_logprobs[i] if batch.details else 0.0,
                next_token_text,
                next_token_id in self.all_special_ids,
                generated_text,
            )

            generations.append(generation)

        return generations, next_batch

    def mock_kv_cache(self, batch: VectorizedCausalLMBatch, dtype:Optional[torch.dtype]):
        from transformers.models.gpt_bigcode.modeling_gpt_bigcode import GPTBigCodeForCausalLM
        if not isinstance(self.model, GPTBigCodeForCausalLM):
            raise NotImplementedError()
        return [torch.empty(
            [len(batch), batch.max_input_length-1, 2 * self.model.config.n_embd // self.model.config.n_head],
            dtype=dtype,
            device=batch.input_ids.device,
        ) for _ in range(self.model.config.n_layer)]

    def fast_forward(self, batch: VectorizedCausalLMBatch, max_input_length: int, cache_dtype:Optional[torch.dtype]):
        diff=max_input_length-batch.max_input_length
        batch.input_ids[:, batch.max_input_length:max_input_length].fill_(self.tokenizer.pad_token_id)
        batch.input_lengths = [length + diff for length in batch.input_lengths]
        batch.max_input_length += diff
        for stopping_criteria in batch.stopping_criterias:
            stopping_criteria.current_tokens+=diff
        batch.past_key_values = None if cache_dtype is None else self.mock_kv_cache(batch, cache_dtype)


