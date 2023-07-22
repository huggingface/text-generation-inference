import torch
import inspect
import numpy as np
import os

from dataclasses import dataclass
from opentelemetry import trace
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizerBase,
    AutoConfig,
)
from typing import Optional, Tuple, List, Type, Dict

from text_generation_server.models import Model
from text_generation_server.models.types import (
    Batch,
    PrefillTokens,
    Generation,
    GeneratedText,
)
from text_generation_server.pb import generate_pb2
from text_generation_server.utils import NextTokenChooser, StoppingCriteria, Sampling
from text_generation_server.models.causal_lm import CausalLMBatch

tracer = trace.get_tracer(__name__)

try:
    import ctranslate2
    from ctranslate2.converters import TransformersConverter
except ImportError:
    ctranslate2 = None


class CT2CausalLM(Model):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        quantize: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
    ):
        if ctranslate2 is None:
            raise ValueError(
                "for your configuration, pip install ctranslate2>=3.16.0 is required.",
            )

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=trust_remote_code,
        )

        # Start CT2
        if torch.cuda.is_available():
            self.ct2_device = "cuda"
        else:
            self.ct2_device = "cpu"

        if dtype == torch.float16:
            ct2_compute_type = "float16"
        elif dtype == torch.float16:
            ct2_compute_type = "bfloat16"
        else:
            # default, int8 quantization.
            if "cuda" in self.ct2_device:
                ct2_compute_type = "int8_float16"
            else:
                ct2_compute_type = "int8"
                # raise ValueError("cpu is currently experimental due to"
                #                  " sampling based / non-greedy next_token"
                #                  " of code only working in float16.")
        # Start CT2 - conversion
        out_dir = f"./ct2-{model_id.replace('/','_')}-{ct2_compute_type}"
        if not os.path.exists(os.path.join(out_dir, "model.bin")):
            ex = ""
            try:
                converter = ctranslate2.converters.TransformersConverter(
                    model_id,
                    activation_scales=None,
                    load_as_float16=True,
                    revision=None,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                )
                converter.convert(
                    output_dir=out_dir,
                    vmap=None,
                    quantization=ct2_compute_type,
                    force=True,
                )
            except Exception as ex:
                pass
        if not os.path.exists(os.path.join(out_dir, "model.bin")) or ex:
            raise ValueError(
                f"conversion for {model_id} failed with ctranslate2: Error {ex}"
            )

        # Start CT2
        self.ct2_model = ctranslate2.Generator(
            out_dir, device=self.ct2_device, compute_type=ct2_compute_type
        )

        class DummyModel(torch.nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.config = AutoConfig.from_pretrained(model_id, revision=revision)

        model = DummyModel()
        self.vocab_size = model.config.vocab_size

        if tokenizer.pad_token_id is None:
            if model.config.pad_token_id is not None:
                tokenizer.pad_token_id = model.config.pad_token_id
            elif model.config.eos_token_id is not None:
                tokenizer.pad_token_id = model.config.eos_token_id
            elif tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        super(CT2CausalLM, self).__init__(
            model=model,
            tokenizer=tokenizer,
            requires_padding=True,
            dtype=torch.int32,
            device=torch.device("cuda"),
        )

    @property
    def batch_type(self) -> Type[CausalLMBatch]:
        return CausalLMBatch

    def decode(self, generated_ids: List[int]) -> str:
        return self.tokenizer.decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    # def forward_slowtokenize_ct2(
    #     self, all_input_ids,
    # ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
    #     # Model Forward, by copy between cpu and cuda
    #     tokens_in  = [self.tokenizer.convert_ids_to_tokens(i) for i in all_input_ids]
    #     logits = self.ct2_model.forward_batch(
    #         tokens_in
    #     )
    #     logits = torch.as_tensor(logits, device="cuda")
    #     logits =  logits.to("cuda").to(torch.float16)
    #     return logits, None

    # def forward_greedy_logits(
    #     self, all_input_ids: List[List[int]],
    # ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
    #     # fallback just to
    #     tokens_in  = [self.tokenizer.convert_ids_to_tokens(i) for i in all_input_ids]
    #     ids = self.ct2_model.generate_batch(
    #         tokens_in,
    #         min_length=1,
    #         max_length=1,
    #         include_prompt_in_result=False,
    #         sampling_temperature=0,
    #     )
    #     # create fake logits from greedy token
    #     logits = torch.full((len(tokens_in), 1, self.vocab_size), -10, dtype=torch.float16, device="cuda")
    #     for i, seq in enumerate(ids):
    #         token = seq.sequences_ids[0]
    #         logits[i, 0, token] = 10
    #     return logits, None

    def forward_ct2(
        self,
        all_input_ids,
        input_lengths,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        # CT2 forward requires a list of list of input tokens ids and lengths
        ids_input = (
            torch.nested.to_padded_tensor(
                torch.nested.nested_tensor(all_input_ids), 1234
            )
            .flatten(1)
            .to(torch.int32)
        )
        # lengths of the padded ids_input, i.e. how often 1234 is used.
        lengths = torch.from_numpy(np.array(input_lengths, dtype=np.int32)).to(
            ids_input.device
        )

        if self.ct2_device == "cpu":
            ids_input, lengths = ids_input.numpy(), lengths.numpy()
        ids_input = ctranslate2.StorageView.from_array(ids_input)
        lengths = ctranslate2.StorageView.from_array(lengths)
        # now, forward through the network
        logits = self.ct2_model.forward_batch(ids_input, lengths)
        logits = torch.as_tensor(logits, device=self.ct2_device)
        # continue with logits as torch tensor, move it to dtype
        if self.ct2_device == "cpu":
            logits = logits.to(self.ct2_device).to(torch.float32)
        else:
            logits = logits.to("cuda").to(torch.float16)
        return logits, None

    @tracer.start_as_current_span("generate_token")
    def generate_token(
        self, batch: CausalLMBatch
    ) -> Tuple[List[Generation], Optional[CausalLMBatch]]:
        # slice the attention mask to the correct shape
        # attention_mask = batch.attention_mask[:, : -batch.padding_right_offset]

        logits, past = self.forward_ct2(batch.all_input_ids, batch.input_lengths)

        # do some verification, see if other forward methods produce same result.
        # logits2, past2 = self.forward_slowtokenize_ct2(
        #     batch.all_input_ids
        # )

        # if sum := torch.isnan(logits).sum():
        #     sum2 = torch.isnan(logits2).sum()
        #     raise ValueError(f"logits {sum}, {sum2}")
        # if sum2 := torch.isnan(logits2).sum():
        #     raise ValueError(f"logits2 {sum}")
        # torch.testing.assert_close(logits, logits2)
        # raise ValueError(f"all_input_ids={len(batch.all_input_ids)},{batch.all_input_ids[0].shape}, logits={logits.shape}, tokens_in={len(tokens_in)},{len(tokens_in[0])}")

        # Results
        generations: List[Generation] = []
        stopped = True

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
        ) in enumerate(iterator):
            # Select next token
            next_token_id, logprobs = next_token_chooser(
                all_input_ids.view(1, -1), logits[-1:, :]
            )

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

            # Shard generations
            # All generations will be appended in the rust sharded client
            if i % self.world_size == self.rank:
                if stop:
                    # Decode generated tokens
                    output_text = self.decode(
                        all_input_ids[-stopping_criteria.current_tokens :, 0]
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
                    ).gather(1, all_input_ids[1:]).squeeze(1)[
                        -new_input_length:-1
                    ].tolist()
                    prefill_token_ids = all_input_ids[-new_input_length:-1]
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

            # Update values
            batch.input_ids[i, 0] = next_token_id
            batch.all_input_ids[i] = all_input_ids
            batch.input_lengths[i] = new_input_length
            batch.prefix_offsets[i] = prefix_offset
            batch.read_offsets[i] = read_offset
            batch.max_input_length = max(batch.max_input_length, new_input_length)

        # We finished all generations in the batch; there is no next batch
        if stopped:
            return generations, None

        # Slice unused values from prefill
        batch.input_ids = batch.input_ids[:, :1]

        # Update attention_mask as we added a new token to input_ids
        batch.attention_mask[:, -batch.padding_right_offset] = 1
        # Decrease right offset
        batch.padding_right_offset -= 1

        # Update position_ids
        batch.position_ids = batch.position_ids[:, -1:] + 1

        # Update past key values
        batch.past_key_values = past

        return generations, batch
