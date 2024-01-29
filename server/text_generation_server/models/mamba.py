import torch
import torch.distributed

from transformers import AutoTokenizer, PreTrainedTokenizerBase
from typing import Optional

from text_generation_server.models import CausalLM
from text_generation_server.models.causal_lm import CausalLMBatch
from text_generation_server.models.custom_modeling.mamba_modeling import (
    MambaConfig,
)
from text_generation_server.pb import generate_pb2
from text_generation_server.utils import (
    initialize_torch_distributed,
    weight_files,
    Weights,
)

import time
from text_generation_server.models.custom_modeling.mamba_modeling import MambaModel
from text_generation_server.models import Model
from typing import Any, List, Optional, Tuple, Type
from text_generation_server.models.types import (
    Batch,
    Tokens,
    Generation,
    GeneratedText,
)
from text_generation_server.utils.tokens import batch_top_tokens, Sampling


class MambaCausalLMBatch(CausalLMBatch):
    past_transformed_states: Optional[List[torch.Tensor]]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.past_input_ids = None
        self.past_transformed_states = None

    @classmethod
    def from_pb(
        cls,
        pb: generate_pb2.Batch,
        tokenizer: PreTrainedTokenizerBase,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "CausalLMBatch":
        batch = super().from_pb(pb=pb, tokenizer=tokenizer, dtype=dtype, device=device)
        batch.keys_head_dim_last = False
        return batch


class Mamba(Model):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        quantize: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
    ):
        self.process_group, _rank, _world_size = initialize_torch_distributed()
        if torch.cuda.is_available():
            device = torch.device("cuda")
            dtype = torch.float16 if dtype is None else dtype
        else:
            if quantize:
                raise ValueError("quantization is not available on CPU")

            device = torch.device("cpu")
            dtype = torch.float32 if dtype is None else dtype

        tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/gpt-neox-20b",
            revision=revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=trust_remote_code,
        )
        config = MambaConfig.from_pretrained(
            model_id, revision=revision, trust_remote_code=trust_remote_code
        )

        tokenizer.bos_token_id = config.bos_token_id
        tokenizer.eos_token_id = config.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token

        config.quantize = quantize
        torch.distributed.barrier(group=self.process_group)
        filenames = weight_files(model_id, revision=revision, extension=".safetensors")
        weights = Weights(filenames, device, dtype, process_group=self.process_group)
        model = MambaModel(config, weights)
        torch.distributed.barrier(group=self.process_group)
        super(Mamba, self).__init__(
            model=model,
            tokenizer=tokenizer,
            requires_padding=True,
            dtype=dtype,
            device=device,
        )

    @property
    def batch_type(self) -> Type[CausalLMBatch]:
        return MambaCausalLMBatch

    def forward(
        self,
        input_ids: torch.Tensor,
        past: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model(
            input_ids,
            past=past,
        )

    def generate_token(self, batch) -> Tuple[List[Any], Optional[Any], Tuple[int, int]]:
        start = time.time_ns()

        input_ids = batch.input_ids
        past_input_ids = batch.past_input_ids
        past_transformed_states = batch.past_transformed_states

        model_output = self.model(
            input_ids,
            past_input_ids,
            past_transformed_states,
        )

        logits = model_output[0]
        past_input_ids = model_output[1]
        past_transformed_states = model_output[2]

        # Results
        generations: List[Generation] = []
        stopped = True

        batch_top_token_ids, batch_top_token_logprobs = batch_top_tokens(
            batch.top_n_tokens,
            batch.top_n_tokens_tensor,
            torch.log_softmax(logits[:, -1], -1),
        )

        start_decode = time.time_ns()

        # Zipped iterator
        iterator = zip(
            batch.requests,
            batch.input_lengths,
            batch.prefix_offsets,
            batch.read_offsets,
            logits,
            batch.next_token_choosers,
            batch.stopping_criterias,
            batch.all_input_ids,
            batch.top_n_tokens,
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
            next_token_chooser,
            stopping_criteria,
            all_input_ids,
            top_n_tokens,
            top_token_ids,
            top_token_logprobs,
        ) in enumerate(iterator):
            # Select next token
            next_token_id, logprobs = next_token_chooser(
                all_input_ids.view(1, -1), logits[-1:, :]
            )

            # add next token to past_input_ids
            past_input_ids = torch.cat([past_input_ids, next_token_id], dim=1)

            # Append next token to all tokens
            all_input_ids = torch.cat([all_input_ids, next_token_id])
            new_input_length = input_length + 1

            # Generated token
            next_token_logprob = logprobs[-1, next_token_id]
            next_token_id_squeezed = next_token_id.squeeze()
            next_token_text, prefix_offset, read_offset = self.decode_token(
                all_input_ids[:, 0], prefix_offset, read_offset
            )

            # Evaluate stopping criteria
            stop, reason = stopping_criteria(
                next_token_id_squeezed,
                next_token_text,
            )

            if not stop:
                stopped = False

            if stop:
                # Decode generated tokens
                output_text, _, _ = self.decode_token(
                    all_input_ids[:, 0],
                    prefix_offset=len(all_input_ids)
                    - stopping_criteria.current_tokens
                    - 1,
                    read_offset=len(all_input_ids) - stopping_criteria.current_tokens,
                    skip_special_tokens=True,
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
            if stopping_criteria.current_tokens == 1 and request.prefill_logprobs:
                # Remove generated token to only have prefill and add nan for first prompt token
                prefill_logprobs = [float("nan")] + torch.log_softmax(
                    logits, -1
                ).gather(1, all_input_ids[1:]).squeeze(1)[-new_input_length:-1].tolist()
                prefill_token_ids = all_input_ids[-new_input_length:-1]
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

            generation = Generation(
                batch.batch_id,
                None,
                Tokens(
                    [next_token_id_squeezed],
                    [next_token_logprob],
                    [next_token_text],
                    [next_token_id_squeezed.item() in self.all_special_ids],
                ),
                generated_text,
                None,
            )

            generations.append(generation)
            next_token_tensor = next_token_id_squeezed.view(1, 1)
            # Update values
            batch.input_ids = torch.cat(
                [batch.input_ids, next_token_tensor], dim=1
            )
            batch.all_input_ids[i] = all_input_ids
            batch.input_lengths[i] = new_input_length
            batch.prefix_offsets[i] = prefix_offset
            batch.read_offsets[i] = read_offset
            batch.max_input_length = max(batch.max_input_length, new_input_length)

        # We finished all generations in the batch; there is no next batch
        if stopped:
            forward_ns = start_decode - start
            decode_ns = time.time_ns() - start_decode
            return generations, None, (forward_ns, decode_ns)

        # Slice unused values from prefill
        batch.input_ids = batch.input_ids[:, :1]
        batch.past_input_ids = past_input_ids
        batch.past_transformed_states = past_transformed_states

        forward_ns = start_decode - start
        decode_ns = time.time_ns() - start_decode
        return generations, batch, (forward_ns, decode_ns)
