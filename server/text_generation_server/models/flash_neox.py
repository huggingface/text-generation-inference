import torch
import torch.distributed

from accelerate import init_empty_weights
from dataclasses import dataclass
from opentelemetry import trace
from safetensors import safe_open
from transformers import AutoTokenizer, PreTrainedTokenizerBase, AutoConfig
from typing import Optional, Tuple, List, Type, Union

from text_generation_server.models import Model
from text_generation_server.models.flash_neox_modeling import (
    FlashGPTNeoXForCausalLM,
    TensorParallelEmbedding,
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
)
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
    initialize_torch_distributed,
    weight_files,
)

tracer = trace.get_tracer(__name__)


@dataclass
class FlashNeoXBatch(Batch):
    batch_id: int
    requests: List[generate_pb2.Request]

    # Decoder values
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    # cumulative sequence lengths
    cu_seqlens: torch.Tensor
    max_seqlen: int
    past_key_values: Optional[torch.Tensor]

    # All tokens
    all_input_ids: List[torch.Tensor]

    # Generation helpers
    next_token_choosers: List[NextTokenChooser]
    stopping_criterias: List[StoppingCriteria]

    def to_pb(self) -> generate_pb2.Batch:
        return generate_pb2.Batch(
            id=self.batch_id, requests=self.requests, size=len(self)
        )

    @classmethod
    def from_pb(
        cls,
        pb: generate_pb2.Batch,
        tokenizer: PreTrainedTokenizerBase,
        device: torch.device,
    ) -> "CausalLMBatch":
        input_ids = []
        position_ids = []
        cu_seqlens = [0]
        max_seqlen = 0

        next_token_choosers = []
        stopping_criterias = []

        # Parse batch
        for r in pb.requests:
            tokenized_input = (
                tokenizer(r.inputs, return_tensors="pt")["input_ids"]
                .to(device)
                .squeeze(0)
            )
            input_ids.append(tokenized_input)
            position_ids.append(
                torch.arange(0, len(tokenized_input), dtype=torch.int32, device=device)
            )
            cu_seqlens.append(len(tokenized_input))
            max_seqlen = max(max_seqlen, len(tokenized_input))

            next_token_choosers.append(NextTokenChooser.from_pb(r.parameters, device))
            stopping_criterias.append(
                StoppingCriteria.from_pb(r.stopping_parameters, tokenizer)
            )

        all_input_ids = input_ids
        input_ids = torch.concat(input_ids).unsqueeze(1)
        position_ids = torch.concat(position_ids)
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

        return cls(
            batch_id=pb.id,
            requests=pb.requests,
            input_ids=input_ids,
            position_ids=position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            past_key_values=None,
            all_input_ids=all_input_ids,
            next_token_choosers=next_token_choosers,
            stopping_criterias=stopping_criterias,
        )

    @classmethod
    @tracer.start_as_current_span("concatenate")
    def concatenate(cls, batches: List["CausalLMBatch"]) -> "CausalLMBatch":
        raise NotImplementedError

    def __len__(self):
        return len(self.requests)


class FlashNeoX(Model):
    def __init__(self, model_id: str, revision: Optional[str] = None, quantize=False):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            raise NotImplementedError("FlashNeoX is only available on GPU")

        if quantize:
            raise NotImplementedError("FlashNeoX does not support quantization")

        tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, padding_side="left"
        )
        self.model = (
            FlashGPTNeoXForCausalLM.from_pretrained(
                model_id,
                revision=revision,
                torch_dtype=dtype,
            )
            .eval()
            .cuda()
        )
        tokenizer.pad_token_id = (
            self.model.config.pad_token_id
            if self.model.config.pad_token_id is not None
            else self.model.config.eos_token_id
        )

        super(FlashNeoX, self).__init__(
            tokenizer=tokenizer,
            device=device,
        )

    @property
    def batch_type(self) -> Type[FlashNeoXBatch]:
        return FlashNeoXBatch

    def decode(self, generated_ids: Union[torch.Tensor, List[int]]) -> str:
        return self.tokenizer.decode(
            generated_ids, skip_special_tokens=True, cleanup_tokenization_spaces=False
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_s: int,
        past_key_values: Optional = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Model Forward
        return self.model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            cu_seqlens=cu_seqlens,
            max_s=max_s,
            past_key_values=past_key_values,
        )

    @tracer.start_as_current_span("generate_token")
    def generate_token(
        self, batch: FlashNeoXBatch
    ) -> Tuple[List[Generation], Optional[FlashNeoXBatch]]:
        out, present = self.forward(
            batch.input_ids.squeeze(1),
            batch.position_ids,
            batch.cu_seqlens,
            batch.max_seqlen,
            batch.past_key_values,
        )

        # List of indices to cache
        next_batch_keep_indices = []

        # New values for next forward
        next_batch_input_ids = []
        next_batch_position_ids = []
        next_batch_cu_seqlens = [0]
        next_batch_max_seqlen = 0
        next_batch_past_key_values = []
        next_batch_all_input_ids = []

        # Results
        generations: List[Generation] = []

        # Zipped iterator
        iterator = zip(
            batch.requests,
            batch.next_token_choosers,
            batch.stopping_criterias,
            batch.all_input_ids,
        )

        # For each member of the batch
        for i, (
            request,
            next_token_chooser,
            stopping_criteria,
            all_input_ids,
        ) in enumerate(iterator):
            # Indexing metadata
            start_index = batch.cu_seqlens[i]
            end_index = batch.cu_seqlens[i + 1]
            seq_length = (end_index - start_index).item()

            if batch.past_key_values is None:
                # Prefill mode
                # out is of shape [cumulative_sequence_lengths, vocab_size]
                logits = out[start_index:end_index]
            else:
                # Decode mode
                # out is of shape [batch_size, vocab_size]
                logits = out[i].unsqueeze(0)

            # Select next token
            next_token_id, logprobs = next_token_chooser(
                all_input_ids.view(1, -1), logits
            )

            # Append next token to all tokens
            all_input_ids = torch.cat([all_input_ids, next_token_id.squeeze(1)])
            new_input_length = seq_length + 1

            # Generated token
            next_token_logprob = logprobs[-1, next_token_id]
            next_token_id_squeezed = next_token_id.squeeze()
            next_token_text = self.decode_token(
                next_token_id_squeezed,
            )

            # Evaluate stopping criteria
            stop, reason = stopping_criteria(
                next_token_id_squeezed,
                next_token_text,
            )

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
                # Keep request in the batch
                seq_present = present[:, start_index:end_index]
                past = torch.nn.functional.pad(seq_present, (0, 0, 0, 0, 0, 0, 0, 1))
                next_batch_past_key_values.append(past)

                generated_text = None
                next_batch_keep_indices.append(i)
                next_batch_input_ids.append(next_token_id)
                next_batch_position_ids.append(seq_length)
                next_batch_cu_seqlens.append(
                    next_batch_cu_seqlens[i] + new_input_length
                )
                next_batch_all_input_ids.append(all_input_ids)
                next_batch_max_seqlen = max(next_batch_max_seqlen, new_input_length)

            # Prefill
            if stopping_criteria.current_tokens == 1:
                # Remove generated token to only have prefill and add nan for first prompt token
                prefill_logprobs = [float("nan")] + logprobs.gather(
                    1, all_input_ids[1:].unsqueeze(1)
                ).squeeze(1)[:-1].tolist()
                prefill_token_ids = all_input_ids[:-1]
                prefill_texts = self.tokenizer.batch_decode(
                    prefill_token_ids,
                    clean_up_tokenization_spaces=False,
                    skip_special_tokens=False,
                )
                prefill_tokens = PrefillTokens(
                    prefill_token_ids, prefill_logprobs, prefill_texts
                )
            else:
                prefill_tokens = None

            generation = Generation(
                request.id,
                prefill_tokens,
                next_token_id_squeezed,
                next_token_logprob,
                next_token_text,
                next_token_id_squeezed.item() in self.all_special_ids,
                generated_text,
            )

            generations.append(generation)

        # We finished all generations in the batch; there is no next batch
        if not next_batch_keep_indices:
            return generations, None

        # If we finished at least one generation, we need to evict the indices of the generations that finished
        # from the values of the next batch
        if len(next_batch_keep_indices) != len(batch):
            # Apply indices to requests, token_choosers and stopping_criterias that need to be cached
            next_batch_requests = [batch.requests[i] for i in next_batch_keep_indices]
            next_batch_next_token_choosers = [
                batch.next_token_choosers[i] for i in next_batch_keep_indices
            ]
            next_batch_stopping_criterias = [
                batch.stopping_criterias[i] for i in next_batch_keep_indices
            ]
        else:
            next_batch_requests = batch.requests
            next_batch_next_token_choosers = batch.next_token_choosers
            next_batch_stopping_criterias = batch.stopping_criterias

        # Create final next batch tensors
        device = out.device
        next_batch_input_ids = torch.concat(next_batch_input_ids, dim=0)
        next_batch_position_ids = torch.tensor(
            next_batch_position_ids, dtype=torch.int32, device=device
        )
        next_batch_cu_seqlens = torch.tensor(
            next_batch_cu_seqlens, dtype=torch.int32, device=device
        )
        if len(next_batch_keep_indices) > 1:
            next_batch_past_key_values = torch.concat(next_batch_past_key_values)
        else:
            next_batch_past_key_values = next_batch_past_key_values[0]

        next_batch = FlashNeoXBatch(
            batch_id=batch.batch_id,
            requests=next_batch_requests,
            input_ids=next_batch_input_ids,
            position_ids=next_batch_position_ids,
            cu_seqlens=next_batch_cu_seqlens,
            max_seqlen=next_batch_max_seqlen,
            past_key_values=next_batch_past_key_values,
            all_input_ids=next_batch_all_input_ids,
            next_token_choosers=next_batch_next_token_choosers,
            stopping_criterias=next_batch_stopping_criterias,
        )
        return generations, next_batch


class FlashNeoXSharded(FlashNeoX):
    def __init__(
        self, model_id: str, revision: Optional[str] = None, quantize: bool = False
    ):
        self.process_group, self.rank, self.world_size = initialize_torch_distributed()
        self.master = self.rank == 0
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{self.rank}")
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            raise NotImplementedError("FlashNeoX is only available on GPU")

        if quantize:
            raise NotImplementedError("FlashNeoX does not support quantization")

        tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, padding_side="left"
        )

        config = AutoConfig.from_pretrained(
            model_id, revision=revision, tp_parallel=True
        )

        torch.distributed.barrier(group=self.process_group)
        filenames = weight_files(model_id, revision=revision, extension=".safetensors")

        with init_empty_weights():
            model = FlashGPTNeoXForCausalLM(config)

        torch.distributed.barrier(group=self.process_group)
        self.load_weights(
            model,
            filenames,
            quantize=quantize,
            device=device,
            rank=self.rank,
            world_size=self.world_size,
        )
        self.model = model.eval().to(dtype)
        torch.distributed.barrier(group=self.process_group)
        super(FlashNeoX, self).__init__(
            tokenizer=tokenizer,
            device=device,
        )

    @staticmethod
    def load_weights(
        model,
        filenames: List[str],
        quantize: bool,
        device: torch.device,
        rank: int,
        world_size: int,
    ):
        parameters = dict(model.named_parameters())
        for file in filenames:
            with safe_open(
                file, framework="pt", device=str(device) if not quantize else "cpu"
            ) as f:
                for name in f.keys():
                    module_name, param_name = name.rsplit(".", 1)
                    module = model.get_submodule(module_name)

                    current_parameter_tensor = parameters.get(name, None)

                    slice_ = f.get_slice(name)

                    if isinstance(module, TensorParallelColumnLinear):
                        size = slice_.get_shape()[0]
                        block_size = size // world_size
                        start = rank * block_size
                        stop = (rank + 1) * block_size
                        tensor = slice_[start:stop]
                    elif isinstance(module, TensorParallelRowLinear):
                        if param_name == "weight":
                            size = slice_.get_shape()[1]
                            block_size = size // world_size
                            start = rank * block_size
                            stop = (rank + 1) * block_size
                            tensor = slice_[:, start:stop]
                        else:
                            tensor = slice_[:]
                            # XXX: Hack for Rowlinear to add the bias only once.
                            if rank != 0:
                                tensor = torch.zeros_like(tensor)
                    elif isinstance(module, TensorParallelEmbedding):
                        size = slice_.get_shape()[0]
                        block_size = size // world_size
                        start = rank * block_size
                        stop = (rank + 1) * block_size
                        tensor = slice_[start:stop]
                    elif name == "embed_out.weight" and model.gpt_neox.tp_embeddings:
                        size = slice_.get_shape()[0]
                        block_size = size // world_size
                        start = rank * block_size
                        stop = (rank + 1) * block_size
                        tensor = slice_[start:stop]
                    else:
                        try:
                            tensor = slice_[:]
                        except:
                            tensor = f.get_tensor(name)

                    if (
                        current_parameter_tensor is not None
                        and current_parameter_tensor.shape != tensor.shape
                    ):
                        raise ValueError(
                            f"Name {name} -- Current {current_parameter_tensor.shape} and got {tensor.shape}"
                        )

                    tensor = tensor.contiguous()

                    if current_parameter_tensor is not None:
                        module._parameters[param_name] = tensor
                    else:
                        module._buffers[param_name] = tensor

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_s: int,
        past_key_values: Optional = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.model.gpt_neox.tp_embeddings:
            logits, present = self.model.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                cu_seqlens=cu_seqlens,
                max_s=max_s,
                past_key_values=past_key_values,
            )

            # Logits are sharded, so we need to gather them
            world_logits = [torch.empty_like(logits) for _ in range(self.world_size)]
            torch.distributed.all_gather(world_logits, logits, group=self.process_group)
            world_logits = torch.cat(world_logits, dim=1)

            return world_logits, present
        # While the model itself is sharded, the embeddings might not as they might not be dividable by num-shard
        else:
            return super(FlashNeoXSharded, self).forward(
                input_ids, position_ids, cu_seqlens, max_s, past_key_values
            )
