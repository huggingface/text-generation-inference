import torch
import torch.distributed

import numpy as np

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

    # Indices to copy present to the correct indices is the pre-allocated past key values
    past_present_indices: torch.Tensor

    # tensor of length b holding starting offset of each sequence
    start_seq: torch.Tensor
    # tensor of length b holding ending offset of each sequence
    end_seq: torch.Tensor
    # tensor of length b holding starting offset of each sequence, only used in prefill
    start_seq_prefill: Optional[torch.Tensor]
    # tensor of length b holding ending offset of each sequence, only used in prefill
    end_seq_prefill: Optional[torch.Tensor]
    # tensor of length b holding starting offset of each query sequence, only used in decode
    start_seq_q: Optional[torch.Tensor]
    # tensor of length b holding ending offset of each query sequence, only used in decode
    end_seq_q: Optional[torch.Tensor]
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
        batch_inputs = []
        max_truncation = 0
        for r in pb.requests:
            batch_inputs.append(r.inputs)
            max_truncation = max(max_truncation, r.truncate)

        batch_tokenized_inputs = tokenizer(
            batch_inputs, truncation=True, max_length=max_truncation
        )["input_ids"]

        position_ids = []
        past_present_indices = []
        start_seq = []
        end_seq = []
        start_seq_prefill = []
        end_seq_prefill = []
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
        cumulative_max_length = 0
        prefill_out_cumulative_length = 0

        max_length = 0

        # Parse batch
        for i, (r, tokenized_input) in enumerate(
            zip(pb.requests, batch_tokenized_inputs)
        ):
            # request id -> idx in list mapping
            requests_idx_mapping[r.id] = i

            tokenized_input = tokenized_input[-r.truncate :]

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
            start_seq_prefill.append(cumulative_length)
            end_seq_prefill.append(cumulative_length + input_length)
            start_seq.append(cumulative_max_length)
            end_seq.append(cumulative_max_length + input_length)

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

            request_past_present_indices = torch.arange(
                cumulative_max_length,
                cumulative_max_length + input_length,
                dtype=torch.int64,
            )
            past_present_indices.append(request_past_present_indices)

            # Update
            # Remove one as the first token des not have a past
            cumulative_length += input_length
            cumulative_max_length += input_length + max_new_tokens - 1
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

        # Create tensors on device
        all_input_ids_tensor = torch.tensor(
            all_input_ids_tensor, dtype=torch.int64, device=device
        )
        start_seq = torch.tensor(start_seq, device=device, dtype=torch.int32)
        end_seq = torch.tensor(end_seq, device=device, dtype=torch.int32)

        if len(pb.requests) > 1:
            input_ids = np.concatenate(all_input_ids, dtype=np.int64)
            position_ids = torch.cat(position_ids)

            past_present_indices = np.concatenate(past_present_indices, dtype=np.int64)

            start_seq_prefill = torch.tensor(
                start_seq_prefill, device=device, dtype=torch.int32
            )
            end_seq_prefill = torch.tensor(
                end_seq_prefill, device=device, dtype=torch.int32
            )
        else:
            input_ids = all_input_ids[0]
            position_ids = position_ids[0]

            past_present_indices = past_present_indices[0]

            start_seq_prefill = start_seq
            end_seq_prefill = end_seq

        input_ids = torch.tensor(input_ids, dtype=torch.int64, device=device)
        position_ids = torch.tensor(position_ids, dtype=torch.int32, device=device)
        past_present_indices = torch.tensor(
            past_present_indices, device=device, dtype=torch.int64
        )

        if all_prefill_logprobs:
            prefill_head_indices = None
            prefill_next_token_indices = end_seq_prefill - 1
        elif no_prefill_logprobs:
            prefill_head_indices = end_seq_prefill - 1
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
            past_present_indices=past_present_indices,
            start_seq=start_seq,
            end_seq=end_seq,
            start_seq_prefill=start_seq_prefill,
            end_seq_prefill=end_seq_prefill,
            start_seq_q=None,
            end_seq_q=None,
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
            max_tokens=cumulative_max_length,
        )

    @tracer.start_as_current_span("filter")
    def filter(self, request_ids: List[int]) -> "FlashCausalLMBatch":
        if len(request_ids) == 0:
            raise ValueError("Batch must have at least one request")
        # We assume that if len(requests) == len(self) then the requests are the same
        if len(request_ids) == len(self):
            return self

        device = self.input_ids.device

        # Cumulative length
        cumulative_max_length = 0

        # New values after filtering
        requests_idx_mapping = {}

        # Used to index into tensors
        indices = []

        # past indices to keep
        past_indices = torch.zeros(
            self.past_key_values.shape[0], dtype=torch.bool, device=device
        )

        # Create on CPU to only move to GPU once instead of at every copy
        start_seq = torch.empty(len(request_ids), dtype=torch.int32)
        end_seq = torch.empty(len(request_ids), dtype=torch.int32)
        start_seq_q = self.start_seq_q[: len(request_ids)]
        end_seq_q = self.end_seq_q[: len(request_ids)]
        max_seqlen = 0

        requests = []
        all_input_ids = []

        input_lengths = []
        prefix_offsets = []
        read_offsets = []

        stopping_criterias = []

        for i, request_id in enumerate(request_ids):
            idx = self.requests_idx_mapping[request_id]
            indices.append(idx)
            requests_idx_mapping[request_id] = i

            requests.append(self.requests[idx])

            # Get length
            request_input_length = self.input_lengths[idx]
            max_seqlen = max(max_seqlen, request_input_length)

            all_input_ids.append(self.all_input_ids[idx])

            input_lengths.append(request_input_length)
            prefix_offsets.append(self.prefix_offsets[idx])
            read_offsets.append(self.read_offsets[idx])

            stopping_criteria = self.stopping_criterias[idx]
            stopping_criterias.append(stopping_criteria)

            remaining_tokens = (
                stopping_criteria.max_new_tokens - stopping_criteria.current_tokens
            )

            # Copy to tensor (CPU)
            start_seq[i] = cumulative_max_length
            end_seq[i] = cumulative_max_length + request_input_length

            # Set slice
            past_indices[
                self.start_seq[idx] : self.end_seq[idx] + remaining_tokens - 1
            ] = True

            cumulative_max_length += request_input_length + remaining_tokens - 1

        # Index into tensors
        input_ids = self.input_ids[indices]
        position_ids = self.position_ids[indices]
        all_input_ids_tensor = self.all_input_ids_tensor[indices]
        next_token_chooser = self.next_token_chooser.filter(indices)
        past_key_values = self.past_key_values[past_indices]

        # Move to GPU now that we have the whole tensor
        start_seq = start_seq.to(device)
        end_seq = end_seq.to(device)
        past_present_indices = end_seq - 1

        return FlashCausalLMBatch(
            batch_id=self.batch_id,
            requests=requests,
            requests_idx_mapping=requests_idx_mapping,
            input_ids=input_ids,
            position_ids=position_ids,
            past_present_indices=past_present_indices,
            start_seq=start_seq,
            end_seq=end_seq,
            start_seq_prefill=None,
            end_seq_prefill=None,
            start_seq_q=start_seq_q,
            end_seq_q=end_seq_q,
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
            max_tokens=cumulative_max_length,
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
        start_seq = batches[0].start_seq.new_empty(total_batch_size)
        end_seq = batches[0].end_seq.new_empty(total_batch_size)
        start_seq_q = torch.arange(
            0, total_batch_size, device=device, dtype=torch.int32
        )
        end_seq_q = start_seq_q + 1
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

            start_seq[start_index:end_index] = batch.start_seq + max_tokens
            end_seq[start_index:end_index] = batch.end_seq + max_tokens

            max_seqlen = max(max_seqlen, batch.max_seqlen)

            all_input_ids.extend(batch.all_input_ids)

            input_lengths.extend(batch.input_lengths)
            prefix_offsets.extend(batch.prefix_offsets)
            read_offsets.extend(batch.read_offsets)

            next_token_chooser_parameters.extend([r.parameters for r in batch.requests])
            stopping_criterias.extend(batch.stopping_criterias)
            past_key_values.append(batch.past_key_values)

            # Update
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

        past_key_values = torch.cat(past_key_values, dim=0)
        past_present_indices = end_seq - 1

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

        next_token_chooser = HeterogeneousNextTokenChooser.from_pb(
            next_token_chooser_parameters, dtype=dtype, device=device
        )

        return FlashCausalLMBatch(
            batch_id=batches[0].batch_id,
            requests=requests,
            requests_idx_mapping=requests_idx_mapping,
            input_ids=input_ids,
            position_ids=position_ids,
            past_present_indices=past_present_indices,
            start_seq=start_seq,
            end_seq=end_seq,
            start_seq_prefill=None,
            end_seq_prefill=None,
            start_seq_q=start_seq_q,
            end_seq_q=end_seq_q,
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
        start_seq: torch.Tensor,
        end_seq: torch.Tensor,
        start_seq_q: Optional[torch.Tensor],
        end_seq_q: Optional[torch.Tensor],
        max_s: int,
        past_present_indices: torch.Tensor,
        past_key_values: Optional = None,
        pre_allocate_past_size: Optional[int] = None,
        lm_head_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Model Forward
        return self.model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            start_seq=start_seq,
            end_seq=end_seq,
            start_seq_q=start_seq_q,
            end_seq_q=end_seq_q,
            max_s=max_s,
            past_present_indices=past_present_indices,
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

        if prefill:
            # Ask to pre-allocate kv to its max size
            # == Sum over batch size (number of tokens + max_new_tokens) - batch size
            pre_allocate_past_size = batch.max_tokens
            start_seq = batch.start_seq_prefill
            end_seq = batch.end_seq_prefill
        else:
            pre_allocate_past_size = None
            start_seq = batch.start_seq
            end_seq = batch.end_seq

        out, present = self.forward(
            batch.input_ids,
            batch.position_ids,
            start_seq,
            end_seq,
            batch.start_seq_q,
            batch.end_seq_q,
            batch.max_seqlen,
            batch.past_present_indices,
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

            # Create batch.start_seq_q and batch.end_seq_q for decode
            batch.start_seq_q = torch.arange(
                0, len(batch), device=self.device, dtype=torch.int32
            )
            batch.end_seq_q = batch.start_seq_q + 1
            next_position_ids = batch.position_ids.new_empty(len(batch))
            # We do not need start_seq_prefill and end_seq_prefill anymore
            batch.start_seq_prefill = None
            batch.end_seq_prefill = None
        else:
            prefill_logprobs = None
            next_position_ids = batch.position_ids

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
            # Indexing metadata
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
        batch.past_present_indices = batch.end_seq
        batch.end_seq = batch.end_seq + 1

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
        batch.past_key_values = present

        # No need to return a batch if we know that all requests stopped
        return generations, batch if not stopped else None
