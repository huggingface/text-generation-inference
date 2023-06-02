import torch
import torch.distributed

import numpy as np

from torch.nn import functional as F

from dataclasses import dataclass
from opentelemetry import trace
from transformers import AutoTokenizer, PreTrainedTokenizerBase, PreTrainedModel
from typing import Optional, Tuple, List, Type, Union, Dict

from text_generation_server.models import Model
from text_generation_server.models.types import (
    Batch,
    PrefillTokens,
    Generation,
    GeneratedText,
)
from text_generation_server.pb import generate_pb2
from text_generation_server.utils import StoppingCriteria, HeterogeneousNextTokenChooser

tracer = trace.get_tracer(__name__)


@dataclass
class FlashCausalLMBatch(Batch):
    batch_id: int
    requests: List[generate_pb2.Request]
    # request id -> idx in list mapping
    requests_idx_mapping: Dict[int, int]

    # Decoder values
    input_ids: torch.Tensor
    position_ids: torch.Tensor

    # cumulative sequence lengths
    cu_seqlens: torch.Tensor
    # cumulative query sequence lengths, only used in decode
    cu_seqlens_q: Optional[torch.Tensor]
    # past key values, only used in decode
    past_key_values: Optional[torch.Tensor]
    max_seqlen: int

    # Prefill metadata tensors to efficiently compute logprobs
    prefill_head_indices: Optional[torch.Tensor]
    prefill_next_token_indices: Optional[torch.tensor]
    prefill_cu_outlens: Optional[List[int]]

    # All tokens
    all_input_ids: List[List[int]]
    all_input_ids_tensor: torch.Tensor

    # Lengths of all generations present in the batch
    input_lengths: List[int]
    prefix_offsets: List[Optional[int]]
    read_offsets: List[Optional[int]]

    # Generation helpers
    next_token_chooser: HeterogeneousNextTokenChooser
    stopping_criterias: List[StoppingCriteria]

    # Maximum number of tokens this batch will grow to
    max_tokens: int

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
    ) -> "FlashCausalLMBatch":
        position_ids = []
        cu_seqlens = [0]
        max_seqlen = 0

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

        # Cumulative length
        cumulative_length = 0
        prefill_out_cumulative_length = 0

        max_tokens = 0
        max_length = 0

        # Parse batch
        for i, r in enumerate(pb.requests):
            # request id -> idx in list mapping
            requests_idx_mapping[r.id] = i

            tokenized_input = tokenizer(
                r.inputs, truncation=True, max_length=r.truncate
            )["input_ids"]

            input_length = len(tokenized_input)
            max_seqlen = max(max_seqlen, input_length)
            input_lengths.append(input_length)

            prefix_offsets.append(input_length - 5)
            read_offsets.append(input_length)

            all_input_ids.append(tokenized_input)

            # Position ids
            request_position_ids = torch.arange(0, input_length, dtype=torch.int32)
            position_ids.append(request_position_ids)

            # Add cumulative lengths of all previous inputs
            cu_seqlens.append(cumulative_length + input_length)

            next_token_chooser_parameters.append(r.parameters)

            stopping_criteria = StoppingCriteria.from_pb(
                r.stopping_parameters, tokenizer
            )
            max_new_tokens = stopping_criteria.max_new_tokens
            stopping_criterias.append(stopping_criteria)

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
            max_tokens += input_length + max_new_tokens
            max_length = max(max_length, input_length + max_new_tokens)

        next_token_chooser = HeterogeneousNextTokenChooser.from_pb(
            next_token_chooser_parameters, dtype, device
        )

        # Padded all_input_ids_tensor
        all_input_ids_tensor = np.zeros(
            (len(all_input_ids), max_length), dtype=np.int64
        )
        for i, input_ids in enumerate(all_input_ids):
            all_input_ids_tensor[i, : len(input_ids)] = input_ids

        if len(pb.requests) > 1:
            input_ids = np.concatenate(all_input_ids, dtype=np.int64)
            position_ids = torch.cat(position_ids)
        else:
            input_ids = all_input_ids[0]
            position_ids = position_ids[0]

        # Create tensors on device
        input_ids = torch.tensor(input_ids, dtype=torch.int64, device=device)
        all_input_ids_tensor = torch.tensor(
            all_input_ids_tensor, dtype=torch.int64, device=device
        )
        position_ids = torch.tensor(position_ids, dtype=torch.int32, device=device)
        cu_seqlens = torch.tensor(cu_seqlens, device=device, dtype=torch.int32)

        if all_prefill_logprobs:
            prefill_head_indices = None
            prefill_next_token_indices = cu_seqlens[1:] - 1
        elif no_prefill_logprobs:
            prefill_head_indices = cu_seqlens[1:] - 1
            prefill_next_token_indices = None
        else:
            prefill_head_indices = torch.tensor(
                torch.cat(prefill_head_indices), dtype=torch.int64, device=device
            )
            prefill_next_token_indices = torch.tensor(
                prefill_next_token_indices, dtype=torch.int64, device=device
            )

        return cls(
            batch_id=pb.id,
            requests=pb.requests,
            requests_idx_mapping=requests_idx_mapping,
            input_ids=input_ids,
            position_ids=position_ids,
            cu_seqlens=cu_seqlens,
            cu_seqlens_q=None,
            max_seqlen=max_seqlen,
            prefill_head_indices=prefill_head_indices,
            prefill_next_token_indices=prefill_next_token_indices,
            prefill_cu_outlens=prefill_cu_outlens,
            past_key_values=None,
            input_lengths=input_lengths,
            prefix_offsets=prefix_offsets,
            read_offsets=read_offsets,
            all_input_ids=all_input_ids,
            all_input_ids_tensor=all_input_ids_tensor,
            next_token_chooser=next_token_chooser,
            stopping_criterias=stopping_criterias,
            max_tokens=max_tokens,
        )

    @tracer.start_as_current_span("filter")
    def filter(self, request_ids: List[int]) -> "FlashCausalLMBatch":
        if len(request_ids) == 0:
            raise ValueError("Batch must have at least one request")
        # We assume that if len(requests) == len(self) then the requests are the same
        if len(request_ids) == len(self):
            return self

        single_request = len(request_ids) == 1

        # Cumulative length
        cumulative_length = 0

        # New values after filtering
        requests_idx_mapping = {}

        # Used to index into tensors
        indices = []

        # Create on CPU to only move to GPU once instead of at every copy
        cu_seqlens = torch.zeros(len(request_ids) + 1, dtype=torch.int32)
        cu_seqlens_q = self.cu_seqlens_q[: len(request_ids) + 1]
        max_seqlen = 0
        past_key_values = []

        requests = []
        all_input_ids = []

        input_lengths = []
        prefix_offsets = []
        read_offsets = []

        stopping_criterias = []

        max_tokens = 0

        for i, request_id in enumerate(request_ids):
            idx = self.requests_idx_mapping[request_id]
            indices.append(idx)
            requests_idx_mapping[request_id] = i

            requests.append(self.requests[idx])

            # Get length
            request_input_length = self.input_lengths[idx]

            # Copy to tensor (CPU)
            cu_seqlens[i + 1] = cumulative_length + request_input_length
            max_seqlen = max(max_seqlen, request_input_length)

            # Slice from past
            past_key_values.append(
                self.past_key_values[:, self.cu_seqlens[idx] : self.cu_seqlens[idx + 1]]
            )

            all_input_ids.append(self.all_input_ids[idx])

            input_lengths.append(request_input_length)
            prefix_offsets.append(self.prefix_offsets[idx])
            read_offsets.append(self.read_offsets[idx])

            stopping_criteria = self.stopping_criterias[idx]
            stopping_criterias.append(stopping_criteria)

            cumulative_length += request_input_length
            max_tokens += request_input_length + (
                stopping_criteria.max_new_tokens - stopping_criteria.current_tokens
            )

        if single_request:
            # Preallocate tensor for bs = 1 case
            past_key_values = F.pad(
                past_key_values[0],
                (
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    stopping_criterias[0].max_new_tokens
                    - stopping_criterias[0].current_tokens,
                ),
            )
        else:
            # Cat all past
            past_key_values = torch.cat(past_key_values, dim=1)

        # Index into tensors
        input_ids = self.input_ids[indices]
        position_ids = self.position_ids[indices]
        all_input_ids_tensor = self.all_input_ids_tensor[indices]
        next_token_chooser = self.next_token_chooser.filter(indices)

        # Move to GPU now that we have the whole tensor
        cu_seqlens = cu_seqlens.to(self.cu_seqlens.device)

        return FlashCausalLMBatch(
            batch_id=self.batch_id,
            requests=requests,
            requests_idx_mapping=requests_idx_mapping,
            input_ids=input_ids,
            position_ids=position_ids,
            cu_seqlens=cu_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen=max_seqlen,
            prefill_head_indices=None,
            prefill_next_token_indices=None,
            prefill_cu_outlens=None,
            past_key_values=past_key_values,
            input_lengths=input_lengths,
            prefix_offsets=prefix_offsets,
            read_offsets=read_offsets,
            all_input_ids=all_input_ids,
            all_input_ids_tensor=all_input_ids_tensor,
            next_token_chooser=next_token_chooser,
            stopping_criterias=stopping_criterias,
            max_tokens=max_tokens,
        )

    @classmethod
    @tracer.start_as_current_span("concatenate")
    def concatenate(cls, batches: List["FlashCausalLMBatch"]) -> "FlashCausalLMBatch":
        # Batch attributes
        requests = []
        requests_idx_mapping = {}

        total_batch_size = sum([len(b) for b in batches])

        dtype = batches[0].past_key_values.dtype
        device = batches[0].input_ids.device

        input_ids = batches[0].input_ids.new_empty(total_batch_size)
        position_ids = batches[0].position_ids.new_empty(total_batch_size)
        cu_seqlens = [0]
        cu_seqlens_q = torch.arange(
            0, total_batch_size + 1, device=device, dtype=torch.int32
        )
        max_seqlen = 0
        past_key_values = []

        all_input_ids = []

        input_lengths = []
        prefix_offsets = []
        read_offsets = []

        next_token_chooser_parameters = []
        stopping_criterias = []

        # Cumulative length
        cumulative_batch_size = 0
        cumulative_length = 0
        max_tokens = 0
        max_length = 0

        for i, batch in enumerate(batches):
            requests.extend(batch.requests)

            if i == 0:
                requests_idx_mapping = batch.requests_idx_mapping
            else:
                # We need to offset the mapping for each batch by the cumulative batch size
                for k, v in batch.requests_idx_mapping.items():
                    requests_idx_mapping[k] = v + cumulative_batch_size

            start_index = cumulative_batch_size
            end_index = cumulative_batch_size + len(batch)

            # Copy tensors (GPU)
            input_ids[start_index:end_index] = batch.input_ids
            position_ids[start_index:end_index] = batch.position_ids

            # Add cumulative lengths of all previous inputs
            cu_seqlens.extend([l + cumulative_length for l in batch.cu_seqlens[1:]])
            max_seqlen = max(max_seqlen, batch.max_seqlen)

            if len(batch) != 1:
                past_key_values.append(batch.past_key_values)
            else:
                # past was pre-allocated for this batch
                # We need to slice to remove the padding
                past_key_values.append(
                    batch.past_key_values[:, : batch.input_lengths[0]]
                )

            all_input_ids.extend(batch.all_input_ids)

            input_lengths.extend(batch.input_lengths)
            prefix_offsets.extend(batch.prefix_offsets)
            read_offsets.extend(batch.read_offsets)

            next_token_chooser_parameters.extend([r.parameters for r in batch.requests])
            stopping_criterias.extend(batch.stopping_criterias)

            # Update
            cumulative_length += batch.cu_seqlens[-1]
            cumulative_batch_size += len(batch)
            max_tokens += batch.max_tokens
            max_length = max(
                max_length,
                max(
                    input_length
                    + stopping_criteria.max_new_tokens
                    - stopping_criteria.current_tokens
                    for input_length, stopping_criteria in zip(
                        batch.input_lengths, batch.stopping_criterias
                    )
                ),
            )

        all_input_ids_tensor = torch.zeros(
            (total_batch_size, max_length), dtype=torch.int64, device=device
        )

        cumulative_batch_size = 0
        for i, batch in enumerate(batches):
            start_index = cumulative_batch_size
            end_index = cumulative_batch_size + len(batch)

            all_input_ids_tensor[
                start_index:end_index, : batch.all_input_ids_tensor.shape[1]
            ] = batch.all_input_ids_tensor[:, :max_length]

            cumulative_batch_size += len(batch)

        # Cat past
        past_key_values = torch.cat(past_key_values, dim=1)
        # Create final tensor on GPU
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

        next_token_chooser = HeterogeneousNextTokenChooser.from_pb(
            next_token_chooser_parameters, dtype=dtype, device=device
        )

        return FlashCausalLMBatch(
            batch_id=batches[0].batch_id,
            requests=requests,
            requests_idx_mapping=requests_idx_mapping,
            input_ids=input_ids,
            position_ids=position_ids,
            cu_seqlens=cu_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen=max_seqlen,
            prefill_head_indices=None,
            prefill_next_token_indices=None,
            prefill_cu_outlens=None,
            past_key_values=past_key_values,
            input_lengths=input_lengths,
            prefix_offsets=prefix_offsets,
            read_offsets=read_offsets,
            all_input_ids=all_input_ids,
            all_input_ids_tensor=all_input_ids_tensor,
            next_token_chooser=next_token_chooser,
            stopping_criterias=stopping_criterias,
            max_tokens=max_tokens,
        )

    def __len__(self):
        return len(self.requests)


class FlashCausalLM(Model):
    def __init__(
        self,
        model_cls: Type[PreTrainedModel],
        model_id: str,
        revision: Optional[str] = None,
        quantize: Optional[str] = None,
        trust_remote_code: bool = False,
    ):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            dtype = torch.float16
        else:
            raise NotImplementedError("FlashCausalLM is only available on GPU")

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=trust_remote_code,
        )
        model = model_cls.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=dtype,
            load_in_8bit=quantize == "bitsandbytes",
            trust_remote_code=trust_remote_code,
        ).to(device)

        super(FlashCausalLM, self).__init__(
            model=model,
            tokenizer=tokenizer,
            requires_padding=False,
            dtype=dtype,
            device=device,
        )

    @property
    def batch_type(self) -> Type[FlashCausalLMBatch]:
        return FlashCausalLMBatch

    def decode(self, generated_ids: Union[torch.Tensor, List[int]]) -> str:
        return self.tokenizer.decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor],
        max_s: int,
        past_key_values: Optional = None,
        pre_allocate_past_size: Optional[int] = None,
        lm_head_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Model Forward
        return self.model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            cu_seqlens=cu_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            max_s=max_s,
            past_key_values=past_key_values,
            pre_allocate_past_size=pre_allocate_past_size,
            lm_head_indices=lm_head_indices,
        )

    @tracer.start_as_current_span("generate_token")
    def generate_token(
        self, batch: FlashCausalLMBatch
    ) -> Tuple[List[Generation], Optional[FlashCausalLMBatch]]:
        prefill = batch.past_key_values is None
        prefill_logprobs = batch.prefill_next_token_indices is not None
        single_request = len(batch) == 1

        if prefill and single_request:
            # Ask to pre-allocate kv to its max size
            # == number of tokens + max_new_tokens
            pre_allocate_past_size = (
                batch.input_lengths[0] + batch.stopping_criterias[0].max_new_tokens
            )
        else:
            pre_allocate_past_size = None

        out, present = self.forward(
            batch.input_ids,
            batch.position_ids,
            batch.cu_seqlens,
            batch.cu_seqlens_q,
            batch.max_seqlen,
            batch.past_key_values,
            pre_allocate_past_size,
            batch.prefill_head_indices,
        )

        if prefill:
            next_token_logits = (
                out[batch.prefill_next_token_indices] if prefill_logprobs else out
            )
        else:
            next_token_logits = out

        next_input_ids, next_token_logprobs = batch.next_token_chooser(
            batch.all_input_ids_tensor[:, : batch.max_seqlen], next_token_logits
        )

        if prefill:
            if len(batch) > 1 and prefill_logprobs:
                # We create the prefill_tokens_indices tensor that will be used to gather prefill logprobs
                # When batch == 1, we will just use the batch.input_ids values directly
                prefill_tokens_indices = batch.input_ids.new_zeros(len(out))

            # Create batch.cu_seqlens_q for decode
            batch.cu_seqlens_q = torch.arange(
                0, len(batch) + 1, device=self.device, dtype=torch.int32
            )
            next_position_ids = batch.position_ids.new_empty(len(batch))
        else:
            prefill_logprobs = None
            next_position_ids = batch.position_ids

        # Prepare past for next decode
        if len(batch) > 1:
            # Used to slice next batch past
            past_indices = torch.empty(
                present.shape[1], dtype=torch.int64, device=self.device
            )
            batch.past_key_values = present.new_empty(
                (
                    present.shape[0],
                    present.shape[1] + len(batch.requests),
                    *present.shape[2:],
                )
            )

            # It is actually faster to do a whole other for loop here as the copy from present to past is fairly slow
            # and will run asynchronously while we do the next for loop
            cumulative_length = 0
            for i, input_length in enumerate(batch.input_lengths):
                # Indexing metadata
                start_index = cumulative_length
                end_index = cumulative_length + input_length

                # Indices to copy present at the correct place in past_key_values
                torch.arange(
                    start_index + i,
                    end_index + i,
                    dtype=torch.int64,
                    device=self.device,
                    out=past_indices[start_index:end_index],
                )
                cumulative_length += input_length

            # Copy from present to past_key_values
            batch.past_key_values[:, past_indices] = present

        # Initialize past_key_values in prefill for len(batch) == 1
        elif prefill:
            # present is already pre-padded
            batch.past_key_values = present

        # Cumulative length
        cumulative_length = 0

        # Results
        generations: List[Generation] = []
        stopped = True

        # Zipped iterator
        iterator = zip(
            batch.input_lengths,
            batch.all_input_ids,
        )

        # We do two for loops as the first one can run completely asynchronously from the GPU while for the second
        # one, we need to first do a GPU <-> CPU sync
        # It is faster if we delay this sync for the maximum amount of time

        # For each member of the batch
        for i, (
            input_length,
            all_input_ids,
        ) in enumerate(iterator):
            start_index = cumulative_length
            end_index = cumulative_length + input_length

            if prefill:
                # Indexing metadata
                out_start_index = batch.prefill_cu_outlens[i]
                out_end_index = batch.prefill_cu_outlens[i + 1]
                out_length = out_end_index - out_start_index

                # Initialize position_ids
                # In decode, we do not need this as we can just increment position ids
                next_position_ids[i] = batch.position_ids[end_index - 1]

                # Used to gather prefill logprobs
                # Copy batch.input_ids to prefill_token_indices
                if prefill_logprobs:
                    if len(batch) > 1:
                        prefill_tokens_indices[
                            out_start_index : out_end_index - 1
                        ] = batch.input_ids[start_index + 1 : start_index + out_length]
                    else:
                        # Set prefill_tokens_indices to the correct slice
                        prefill_tokens_indices = batch.input_ids[
                            start_index + 1 : start_index + out_length
                        ]

            batch.all_input_ids_tensor[i, input_length] = next_input_ids[i]

            cumulative_length += input_length

        # Set values in batch
        batch.input_ids = next_input_ids
        batch.position_ids = next_position_ids + 1
        batch.cu_seqlens = batch.cu_seqlens + batch.cu_seqlens_q

        if prefill and prefill_logprobs:
            # Get prefill logprobs
            prefill_logprobs_tensor = torch.log_softmax(out, -1)
            prefill_logprobs = torch.gather(
                prefill_logprobs_tensor, 1, prefill_tokens_indices.view(-1, 1)
            )
            # GPU <-> CPU sync
            prefill_logprobs = prefill_logprobs.view(-1).tolist()

        # GPU <-> CPU sync
        next_token_logprobs = next_token_logprobs.tolist()
        next_token_ids = batch.input_ids.tolist()

        # Zipped iterator
        iterator = zip(
            batch.requests,
            batch.input_lengths,
            batch.prefix_offsets,
            batch.read_offsets,
            batch.stopping_criterias,
            batch.all_input_ids,
            batch.all_input_ids_tensor,
            batch.next_token_chooser.do_sample,
            batch.next_token_chooser.seeds,
            next_token_ids,
            next_token_logprobs,
        )

        # For each member of the batch
        for i, (
            request,
            input_length,
            prefix_offset,
            read_offset,
            stopping_criteria,
            all_input_ids,
            all_input_ids_tensor,
            do_sample,
            seed,
            next_token_id,
            next_token_logprob,
        ) in enumerate(iterator):
            # Append next token to all tokens
            all_input_ids.append(next_token_id)

            # Generated token
            next_token_text, prefix_offset, read_offset = self.decode_token(
                all_input_ids,
                prefix_offset,
                read_offset,
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
                        all_input_ids[-stopping_criteria.current_tokens :]
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
                if prefill and request.prefill_logprobs:
                    out_start_index = batch.prefill_cu_outlens[i]
                    out_end_index = batch.prefill_cu_outlens[i + 1]

                    # Remove generated token to only have prefill and add nan for first prompt token
                    request_prefill_logprobs = [float("nan")] + prefill_logprobs[
                        out_start_index : out_end_index - 1
                    ]
                    prefill_token_ids = all_input_ids[:-1]
                    prefill_texts = self.tokenizer.batch_decode(
                        prefill_token_ids,
                        clean_up_tokenization_spaces=False,
                        skip_special_tokens=False,
                    )
                    prefill_tokens = PrefillTokens(
                        prefill_token_ids, request_prefill_logprobs, prefill_texts
                    )
                else:
                    prefill_tokens = None

                generation = Generation(
                    request.id,
                    prefill_tokens,
                    next_token_id,
                    next_token_logprob,
                    next_token_text,
                    next_token_id in self.all_special_ids,
                    generated_text,
                )

                generations.append(generation)

            new_input_length = input_length + 1

            # Update values
            batch.input_lengths[i] = new_input_length
            batch.prefix_offsets[i] = prefix_offset
            batch.read_offsets[i] = read_offset
            batch.all_input_ids[i] = all_input_ids

        batch.prefill_cu_outlens = None
        batch.prefill_head_indices = None
        batch.prefill_next_token_indices = None
        batch.max_seqlen = batch.max_seqlen + 1

        # No need to return a batch if we know that all requests stopped
        return generations, batch if not stopped else None
