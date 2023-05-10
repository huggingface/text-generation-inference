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
from text_generation_server.utils import (
    NextTokenChooser,
    StoppingCriteria,
    Sampling,
)

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

    # All tokens
    all_input_ids: List[List[int]]
    all_input_ids_tensor: List[torch.Tensor]

    # Lengths of all generations present in the batch
    input_lengths: List[int]
    offsets: List[Optional[int]]
    token_offsets: List[Optional[int]]

    # Generation helpers
    next_token_choosers: List[NextTokenChooser]
    stopping_criterias: List[StoppingCriteria]

    # Maximum number of tokens this batch will grow to
    max_tokens: int

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
    ) -> "FlashCausalLMBatch":
        position_ids = []
        cu_seqlens = [0]
        max_seqlen = 0

        input_lengths = []
        offsets = []
        token_offsets = []
        all_input_ids = []
        requests_idx_mapping = {}

        next_token_choosers = []
        stopping_criterias = []

        # Cumulative length
        cumulative_length = 0

        max_tokens = 0

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

            offsets.append(None)
            token_offsets.append(None)

            all_input_ids.append(tokenized_input)

            # Position ids
            position_ids.append(np.arange(0, input_length))

            # Add cumulative lengths of all previous inputs
            cu_seqlens.append(cumulative_length + input_length)

            next_token_choosers.append(NextTokenChooser.from_pb(r.parameters, device))

            stopping_criteria = StoppingCriteria.from_pb(
                r.stopping_parameters, tokenizer
            )
            max_new_tokens = stopping_criteria.max_new_tokens
            stopping_criterias.append(stopping_criteria)

            # Update
            cumulative_length += input_length
            max_tokens += input_length + max_new_tokens

        # Create tensors on device
        input_ids = torch.tensor(
            np.concatenate(all_input_ids), dtype=torch.int64, device=device
        )
        position_ids = torch.tensor(
            np.concatenate(position_ids), dtype=torch.int32, device=device
        )
        cu_seqlens = torch.tensor(cu_seqlens, device=device, dtype=torch.int32)

        return cls(
            batch_id=pb.id,
            requests=pb.requests,
            requests_idx_mapping=requests_idx_mapping,
            input_ids=input_ids,
            position_ids=position_ids,
            cu_seqlens=cu_seqlens,
            cu_seqlens_q=None,
            max_seqlen=max_seqlen,
            past_key_values=None,
            input_lengths=input_lengths,
            offsets=offsets,
            token_offsets=token_offsets,
            all_input_ids=all_input_ids,
            all_input_ids_tensor=[],
            next_token_choosers=next_token_choosers,
            stopping_criterias=stopping_criterias,
            max_tokens=max_tokens,
        )

    @tracer.start_as_current_span("filter")
    def filter(self, requests: List[generate_pb2.Request]) -> "FlashCausalLMBatch":
        if len(requests) == 0:
            raise ValueError("Batch must have at least one request")
        # We assume that if len(requests) == len(self) then the requests are the same
        if len(requests) == len(self):
            return self

        single_request = len(requests) == 1

        # Cumulative length
        cumulative_length = 0

        # New values after filtering
        requests_idx_mapping = {}

        input_ids = self.input_ids.new_empty(len(requests))
        position_ids = self.position_ids.new_empty(len(requests))
        # Create on CPU to only move to GPU once instead of at every copy
        cu_seqlens = torch.zeros(len(requests) + 1, dtype=torch.int32)
        cu_seqlens_q = torch.arange(
            0, len(requests) + 1, device=self.cu_seqlens_q.device, dtype=torch.int32
        )
        max_seqlen = 0
        past_key_values = []

        all_input_ids = []
        all_input_ids_tensor = []

        input_lengths = []
        offsets = []
        token_offsets = []

        next_token_choosers = []
        stopping_criterias = []

        max_tokens = 0

        for i, r in enumerate(requests):
            idx = self.requests_idx_mapping[r.id]
            requests_idx_mapping[r.id] = i

            # Get length
            request_input_length = self.input_lengths[idx]

            # Copy tensors (GPU)
            input_ids[i] = self.input_ids[idx]
            position_ids[i] = self.position_ids[idx]

            # Copy to tensor (CPU)
            cu_seqlens[i + 1] = cumulative_length + request_input_length
            max_seqlen = max(max_seqlen, request_input_length)

            # Slice from past
            past_key_values.append(
                self.past_key_values[:, self.cu_seqlens[idx] : self.cu_seqlens[idx + 1]]
            )

            all_input_ids.append(self.all_input_ids[idx])
            all_input_ids_tensor.append(self.all_input_ids_tensor[idx])

            input_lengths.append(request_input_length)
            offsets.append(self.offsets[idx])
            token_offsets.append(self.token_offsets[idx])

            next_token_choosers.append(self.next_token_choosers[idx])

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
            past_key_values=past_key_values,
            input_lengths=input_lengths,
            offsets=offsets,
            token_offsets=token_offsets,
            all_input_ids=all_input_ids,
            all_input_ids_tensor=all_input_ids_tensor,
            next_token_choosers=next_token_choosers,
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
        all_input_ids_tensor = []

        input_lengths = []
        offsets = []
        token_offsets = []

        next_token_choosers = []
        stopping_criterias = []

        # Cumulative length
        cumulative_batch_size = 0
        cumulative_length = 0
        max_tokens = 0

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
            all_input_ids_tensor.extend(batch.all_input_ids_tensor)

            input_lengths.extend(batch.input_lengths)
            offsets.extend(batch.offsets)
            token_offsets.extend(batch.token_offsets)

            next_token_choosers.extend(batch.next_token_choosers)
            stopping_criterias.extend(batch.stopping_criterias)

            # Update
            cumulative_length += batch.cu_seqlens[-1]
            cumulative_batch_size += len(batch)
            max_tokens += batch.max_tokens

        # Cat past
        past_key_values = torch.cat(past_key_values, dim=1)
        # Create final tensor on GPU
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

        return FlashCausalLMBatch(
            batch_id=batches[0].batch_id,
            requests=requests,
            requests_idx_mapping=requests_idx_mapping,
            input_ids=input_ids,
            position_ids=position_ids,
            cu_seqlens=cu_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen=max_seqlen,
            past_key_values=past_key_values,
            input_lengths=input_lengths,
            offsets=offsets,
            token_offsets=token_offsets,
            all_input_ids=all_input_ids,
            all_input_ids_tensor=all_input_ids_tensor,
            next_token_choosers=next_token_choosers,
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
        quantize: bool = False,
        decode_buffer: int = 3,
    ):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            dtype = torch.float16
        else:
            raise NotImplementedError("FlashCausalLM is only available on GPU")

        tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, padding_side="left", truncation_side="left"
        )
        self.model = (
            model_cls.from_pretrained(
                model_id,
                revision=revision,
                torch_dtype=dtype,
                load_in_8bit=quantize,
            )
            .eval()
            .to(device)
        )

        super(FlashCausalLM, self).__init__(
            tokenizer=tokenizer,
            requires_padding=False,
            dtype=dtype,
            device=device,
            decode_buffer=decode_buffer,
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
        )

    @tracer.start_as_current_span("generate_token")
    def generate_token(
        self, batch: FlashCausalLMBatch
    ) -> Tuple[List[Generation], Optional[FlashCausalLMBatch]]:
        prefill = batch.past_key_values is None

        if prefill and len(batch) == 1:
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
        )

        if prefill:
            if len(batch) > 1:
                # We create the prefill_tokens_indices tensor that will be used to gather prefill logprobs
                # When batch == 1, we will just use the batch.input_ids values directly
                prefill_tokens_indices = batch.input_ids.new_zeros(len(batch.input_ids))

            # Create batch.cu_seqlens_q for decode
            batch.cu_seqlens_q = torch.arange(
                0, len(batch) + 1, device=self.device, dtype=torch.int32
            )
            next_input_ids = batch.input_ids.new_empty(len(batch))
            next_position_ids = batch.position_ids.new_empty(len(batch))
        else:
            prefill_logprobs = None
            next_input_ids = batch.input_ids
            next_position_ids = batch.position_ids

        next_token_logprobs = out.new_empty(len(batch))

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
            batch.next_token_choosers,
            batch.stopping_criterias,
            batch.all_input_ids,
        )

        # We do two for loops as the first one can run completely asynchronously from the GPU while for the second
        # one, we need to first do a GPU <-> CPU sync
        # It is faster if we delay this sync for the maximum amount of time

        # For each member of the batch
        for i, (
            input_length,
            next_token_chooser,
            stopping_criteria,
            all_input_ids,
        ) in enumerate(iterator):
            # Indexing metadata
            start_index = cumulative_length
            end_index = cumulative_length + input_length

            if prefill:
                # Prefill mode
                # out is of shape [cumulative_sequence_lengths, vocab_size]
                # only take last token logit
                logits = out[end_index - 1 : end_index]

                # Create all_input_ids_tensor that will be used by token warpers (for example, RepetitionPenalty)
                all_input_ids_tensor = batch.input_ids.new_empty(
                    input_length + stopping_criteria.max_new_tokens
                )
                # Copy from batch.input_ids to all_input_ids_tensor
                all_input_ids_tensor[:input_length] = batch.input_ids[
                    start_index:end_index
                ]
                batch.all_input_ids_tensor.append(all_input_ids_tensor)

                # Initialize position_ids
                # In decode, we do not need this as we can just increment position ids
                next_position_ids[i] = batch.position_ids[end_index - 1]

                # Used to gather prefill logprobs
                # Copy batch.input_ids to prefill_token_indices
                if len(batch) > 1:
                    prefill_tokens_indices[
                        start_index : end_index - 1
                    ] = batch.input_ids[start_index + 1 : end_index]
                else:
                    # Set prefill_tokens_indices to the correct slice
                    prefill_tokens_indices = batch.input_ids[
                        start_index + 1 : end_index
                    ]
            else:
                # Decode mode
                # out is of shape [batch_size, vocab_size]
                logits = out[i].view(1, -1)

            all_input_ids_tensor = batch.all_input_ids_tensor[i]

            # Select next token
            next_token_id, logprob = next_token_chooser(
                all_input_ids_tensor[None, :input_length], logits
            )

            # Add to all_input_ids_tensor
            next_token_id_squeezed = next_token_id.view(1)
            all_input_ids_tensor[input_length] = next_token_id_squeezed

            # Set values
            next_input_ids[i] = next_token_id_squeezed
            next_token_logprobs[i] = logprob[-1, next_token_id].view(1)

            cumulative_length += input_length

        # Set values in batch
        batch.input_ids = next_input_ids
        batch.position_ids = next_position_ids + 1
        batch.cu_seqlens = batch.cu_seqlens + batch.cu_seqlens_q

        if prefill:
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

        cumulative_length = 0

        # Zipped iterator
        iterator = zip(
            batch.requests,
            batch.input_lengths,
            batch.offsets,
            batch.token_offsets,
            batch.next_token_choosers,
            batch.stopping_criterias,
            batch.all_input_ids,
            batch.all_input_ids_tensor,
            next_token_ids,
            next_token_logprobs,
        )

        # For each member of the batch
        for i, (
            request,
            input_length,
            offset,
            token_offset,
            next_token_chooser,
            stopping_criteria,
            all_input_ids,
            all_input_ids_tensor,
            next_token_id,
            next_token_logprob,
        ) in enumerate(iterator):
            start_index = cumulative_length
            end_index = cumulative_length + input_length

            # Append next token to all tokens
            all_input_ids.append(next_token_id)

            # Generated token
            next_token_text, offset, token_offset = self.decode_token(
                all_input_ids,
                offset,
                token_offset,
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
                    # Get seed
                    if isinstance(next_token_chooser.choice, Sampling):
                        seed = next_token_chooser.choice.seed
                    else:
                        seed = None

                    generated_text = GeneratedText(
                        output_text, stopping_criteria.current_tokens, reason, seed
                    )
                else:
                    generated_text = None

                # Prefill
                if prefill:
                    # Remove generated token to only have prefill and add nan for first prompt token
                    request_prefill_logprobs = [float("nan")] + prefill_logprobs[
                        start_index : end_index - 1
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
            batch.offsets[i] = offset
            batch.token_offsets[i] = token_offset
            batch.all_input_ids[i] = all_input_ids
            batch.max_seqlen = batch.max_seqlen + 1
            cumulative_length += input_length

        # No need to return a batch if we know that all requests stopped
        return generations, batch if not stopped else None
