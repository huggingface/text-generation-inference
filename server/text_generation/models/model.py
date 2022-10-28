import torch
import torch.distributed

from typing import List, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from text_generation.models.types import Batch, GeneratedText


class Model:
    def __init__(self, model_name: str):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            dtype = torch.float16
        else:
            self.device = torch.device("cpu")
            dtype = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map="auto"
        ).eval()

        self.num_heads = self.model.config.num_attention_heads

    def forward(
        self, input_ids, attention_mask, past_key_values: Optional = None
    ) -> CausalLMOutputWithPast:
        # Model Forward
        return self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )

    def generate_token(
        self, batch: Batch
    ) -> Tuple[List[GeneratedText], Optional[Batch]]:
        # For some reason, inference_mode does not work well with GLOO which we use on CPU
        context_manager = (
            torch.no_grad if self.device.type == "cpu" else torch.inference_mode
        )
        with context_manager():
            outputs = self.forward(**batch.input_ids)

        # List of indices to cache
        next_batch_keep_indices = []
        next_batch_past_keep_indices = []

        # New input_ids for next forward
        next_batch_input_ids = []
        next_batch_all_input_ids = []
        next_all_input_lengths = []

        next_batch_size = 0
        next_batch_max_sequence_length = 0

        # Finished requests
        generated_texts: List[GeneratedText] = []

        # Zipped iterator
        iterator = zip(
            batch.requests,
            batch.all_input_lengths,
            outputs.logits,
            batch.next_token_choosers,
            batch.stopping_criterias,
            batch.all_input_ids,
        )

        # For each member of the batch
        for i, (
            request,
            input_length,
            logits,
            next_token_chooser,
            stopping_criteria,
            all_tokens,
        ) in enumerate(iterator):
            # Select next token
            next_token = next_token_chooser(all_tokens, logits.unsqueeze(0)[:, -1])

            # Append next token to all tokens
            all_tokens = torch.cat([all_tokens, next_token])

            # Evaluate stopping criteria
            if stopping_criteria(all_tokens):
                # Decode all tokens
                output = self.tokenizer.decode(
                    all_tokens.squeeze(-1), skip_special_tokens=True
                )
                # Add to the list of finished generations with the original request
                generated_texts.append(GeneratedText(request, output))
            # add to the next batch
            else:
                next_batch_keep_indices.append(i)
                # past_key_values is of shape [batch_size * num_heads, ...]
                # so we need to take into account the `num_heads` stride here
                next_batch_past_keep_indices.extend(
                    [j for j in range(i * self.num_heads, (i + 1) * self.num_heads)]
                )
                next_batch_input_ids.append(next_token)
                next_batch_all_input_ids.append(all_tokens)
                next_batch_size += 1
                new_input_length = input_length + 1
                next_all_input_lengths.append(new_input_length)
                next_batch_max_sequence_length = max(
                    next_batch_max_sequence_length, new_input_length
                )

        # We finished all generations in the batch; there is no next batch
        if not next_batch_keep_indices:
            return generated_texts, None

        # If we finished at least one generation
        next_batch_input_ids = {"input_ids": torch.cat(next_batch_input_ids, dim=0)}
        if generated_texts:
            # Apply indices to attention mask, past key values and other items that need to be cached
            next_batch_input_ids["attention_mask"] = batch.input_ids["attention_mask"][
                next_batch_keep_indices
            ]
            next_batch_input_ids["past_key_values"] = [
                (
                    keys[next_batch_past_keep_indices],
                    values[next_batch_past_keep_indices],
                )
                for keys, values in outputs["past_key_values"]
            ]
            next_batch_requests = [batch.requests[i] for i in next_batch_keep_indices]
            next_batch_next_token_choosers = [
                batch.next_token_choosers[i] for i in next_batch_keep_indices
            ]
            next_batch_stopping_criterias = [
                batch.stopping_criterias[i] for i in next_batch_keep_indices
            ]
        else:
            next_batch_input_ids["attention_mask"] = batch.input_ids["attention_mask"]
            next_batch_input_ids["past_key_values"] = outputs["past_key_values"]
            next_batch_requests = batch.requests
            next_batch_next_token_choosers = batch.next_token_choosers
            next_batch_stopping_criterias = batch.stopping_criterias

        # Update attention_mask with padding as we added a new token to input_ids
        next_batch_input_ids["attention_mask"] = torch.cat(
            [
                next_batch_input_ids["attention_mask"],
                torch.ones((next_batch_size, 1)).to(self.device),
            ],
            dim=1,
        )

        next_batch = Batch(
            batch_id=batch.batch_id,
            requests=next_batch_requests,
            all_input_lengths=next_all_input_lengths,
            input_ids=next_batch_input_ids,
            all_input_ids=next_batch_all_input_ids,
            next_token_choosers=next_batch_next_token_choosers,
            stopping_criterias=next_batch_stopping_criterias,
            size=next_batch_size,
            max_sequence_length=next_batch_max_sequence_length,
        )
        return generated_texts, next_batch
