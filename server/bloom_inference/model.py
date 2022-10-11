import torch
import torch.distributed

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.modeling_utils import no_init_weights

from bloom_inference.pb import generate_pb2
from bloom_inference.shard_model import shard_model, match_suffix
from bloom_inference.utils import (
    StoppingCriteria,
    NextTokenChooser,
    initialize_torch_distributed,
    set_default_dtype,
)

torch.manual_seed(0)


@dataclass
class Batch:
    batch_id: int
    requests: List[generate_pb2.Request]
    input_ids: Dict[str, torch.Tensor]
    all_input_ids: List[torch.Tensor]
    next_token_choosers: List[NextTokenChooser]
    stopping_criterias: List[StoppingCriteria]
    size: int
    max_sequence_length: int

    def to_pb(self):
        return generate_pb2.Batch(
            id=self.batch_id,
            requests=self.requests,
            size=self.size,
            max_sequence_length=self.max_sequence_length,
        )

    @classmethod
    def from_pb(
        cls, pb: generate_pb2.Batch, tokenizer: AutoTokenizer, device: torch.device
    ) -> "Batch":
        inputs = []
        next_token_choosers = []
        stopping_criterias = []
        input_lengths = []

        # Parse batch
        for r in pb.requests:
            inputs.append(r.inputs)
            input_lengths.append(r.input_length)
            next_token_choosers.append(
                NextTokenChooser(
                    temperature=r.parameters.temperature,
                    top_k=r.parameters.top_k,
                    top_p=r.parameters.top_p,
                    do_sample=r.parameters.do_sample,
                )
            )
            stopping_criterias.append(StoppingCriteria(max_new_tokens=r.max_new_tokens))

        input_ids = tokenizer(inputs, return_tensors="pt", padding=True).to(device)
        # Remove padding from all_input_ids
        all_input_ids = [
            input_ids.squeeze(0)[-length:].unsqueeze(-1)
            for length, input_ids in zip(
                input_lengths, input_ids["input_ids"].split(1, dim=0)
            )
        ]

        return cls(
            batch_id=pb.id,
            requests=pb.requests,
            input_ids=input_ids,
            all_input_ids=all_input_ids,
            next_token_choosers=next_token_choosers,
            stopping_criterias=stopping_criterias,
            size=pb.size,
            max_sequence_length=pb.max_sequence_length,
        )

    @classmethod
    def concatenate(cls, batches: List["Batch"]) -> "Batch":
        # Used for padding
        total_batch_size = sum(batch.size for batch in batches)
        max_sequence_length = max(batch.max_sequence_length for batch in batches)

        # Batch attributes
        input_ids = {"input_ids": None, "attention_mask": None, "past_key_values": []}
        requests = []
        all_input_ids = []
        next_token_choosers = []
        stopping_criterias = []

        # Used for slicing correctly inside the tensors
        # Equivalent to a cumsum on batch sizes
        start_index = 0
        for i, batch in enumerate(batches):
            requests.extend(batch.requests)
            all_input_ids.extend(batch.all_input_ids)
            next_token_choosers.extend(batch.next_token_choosers)
            stopping_criterias.extend(batch.stopping_criterias)

            # Slicing end index for this batch
            end_index = start_index + batch.size

            # We only concatenate batches that did at least one step
            if batch.input_ids["input_ids"].shape[1] > 1:
                raise ValueError("Batch input_ids should be of shape (batch_size, 1)")

            # Initialize tensors
            if i == 0:
                input_ids["input_ids"] = torch.empty(
                    (total_batch_size, 1),
                    dtype=batch.input_ids["input_ids"].dtype,
                    device=batch.input_ids["input_ids"].device,
                )
                input_ids["attention_mask"] = torch.zeros(
                    (total_batch_size, max_sequence_length),
                    dtype=batch.input_ids["attention_mask"].dtype,
                    device=batch.input_ids["attention_mask"].device,
                )

            # input_ids["input_ids"] is always of shape [batch_size, 1]
            # We do not need to pad it
            input_ids["input_ids"][start_index:end_index] = batch.input_ids["input_ids"]

            # We need to slice the attention mask to remove padding from previous steps
            input_ids["attention_mask"][
                start_index:end_index, -batch.max_sequence_length :
            ] = batch.input_ids["attention_mask"][:, -batch.max_sequence_length :]

            for j, past in enumerate(batch.input_ids["past_key_values"]):
                past_keys = past[0]
                past_values = past[1]

                _, head_dim, padded_sequence_length = past_keys.shape

                # Reshape the tensors to make slicing easier
                past_keys = past_keys.view(
                    batch.size, -1, head_dim, padded_sequence_length
                )
                past_values = past_values.view(
                    batch.size, -1, padded_sequence_length, head_dim
                )
                num_heads = past_keys.shape[1]

                # Initialize tensors
                # This will run only once per layer
                if j == len(input_ids["past_key_values"]):
                    padded_past_keys = torch.zeros(
                        (
                            total_batch_size,
                            num_heads,
                            head_dim,
                            max_sequence_length - 1,
                        ),
                        dtype=past_keys.dtype,
                        device=past_keys.device,
                    )
                    padded_past_values = torch.zeros(
                        (
                            total_batch_size,
                            num_heads,
                            max_sequence_length - 1,
                            head_dim,
                        ),
                        dtype=past_values.dtype,
                        device=past_values.device,
                    )
                    input_ids["past_key_values"].append(
                        [padded_past_keys, padded_past_values]
                    )

                # We slice the past keys and values to remove the padding from previous batches
                input_ids["past_key_values"][j][0][
                    start_index:end_index, :, :, -(batch.max_sequence_length - 1) :
                ] = past_keys[:, :, :, -(batch.max_sequence_length - 1) :]

                input_ids["past_key_values"][j][1][
                    start_index:end_index, :, -(batch.max_sequence_length - 1) :, :
                ] = past_values[:, :, -(batch.max_sequence_length - 1) :, :]

                # If we are on the last batch, we need to reshape the tensors
                if (i + 1) == len(batches):
                    input_ids["past_key_values"][j][0] = input_ids["past_key_values"][
                        j
                    ][0].view(total_batch_size * num_heads, head_dim, -1)
                    input_ids["past_key_values"][j][1] = input_ids["past_key_values"][
                        j
                    ][1].view(total_batch_size * num_heads, -1, head_dim)

            start_index += batch.size

        return cls(
            batch_id=batches[0].batch_id,
            requests=requests,
            input_ids=input_ids,
            all_input_ids=all_input_ids,
            next_token_choosers=next_token_choosers,
            stopping_criterias=stopping_criterias,
            size=total_batch_size,
            max_sequence_length=max_sequence_length,
        )


@dataclass
class GeneratedText:
    request: generate_pb2.Request
    output: str

    def to_pb(self) -> generate_pb2.GeneratedText:
        return generate_pb2.GeneratedText(request=self.request, output=self.output)


class BLOOM:
    def __init__(self, model_name: str):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = (
            AutoModelForCausalLM.from_pretrained(model_name).eval().to(self.device)
        )
        self.num_heads = self.model.base_model.num_heads

    def forward(self, input_ids, attention_mask, past_key_values: Optional = None):
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
        with torch.no_grad():
            outputs = self.forward(**batch.input_ids)

        # List of indices to cache
        next_batch_keep_indices = []
        next_batch_past_keep_indices = []

        # New input_ids for next forward
        next_batch_input_ids = []
        next_batch_all_input_ids = []

        next_batch_size = 0
        next_batch_max_sequence_length = 0

        # Finished requests
        generated_texts: List[GeneratedText] = []

        # Zipped iterator
        iterator = zip(
            batch.requests,
            outputs.logits,
            batch.next_token_choosers,
            batch.stopping_criterias,
            batch.all_input_ids,
        )

        # For each member of the batch
        for i, (
            request,
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
                next_batch_max_sequence_length = max(
                    next_batch_max_sequence_length, len(all_tokens)
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
            input_ids=next_batch_input_ids,
            all_input_ids=next_batch_all_input_ids,
            next_token_choosers=next_batch_next_token_choosers,
            stopping_criterias=next_batch_stopping_criterias,
            size=next_batch_size,
            max_sequence_length=next_batch_max_sequence_length,
        )
        return generated_texts, next_batch


class BLOOMSharded(BLOOM):
    def __init__(self, model_name: str, shard_directory: Path):
        super(BLOOM, self).__init__()
        self.process_group, self.rank, self.world_size = initialize_torch_distributed()
        self.master = self.rank == 0
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.rank}")
            dtype = torch.bfloat16
        else:
            self.device = torch.device("cpu")
            dtype = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

        # shard state_dict
        if self.master:
            # TODO @thomasw21 do some caching
            shard_state_dict_paths = shard_model(
                model_name, shard_directory, tp_world_size=self.world_size, dtype=dtype
            )
            shard_state_dict_paths = [
                str(path.absolute()) for path in shard_state_dict_paths
            ]
        else:
            shard_state_dict_paths = [None] * self.world_size

        torch.distributed.broadcast_object_list(
            shard_state_dict_paths, src=0, group=self.process_group
        )
        shard_state_dict_path = shard_state_dict_paths[self.rank]

        config = AutoConfig.from_pretrained(
            model_name, slow_but_exact=False, tp_parallel=True
        )
        config.pad_token_id = 3

        # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
        # in PyTorch 1.12 and later.
        torch.backends.cuda.matmul.allow_tf32 = True

        # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
        torch.backends.cudnn.allow_tf32 = True

        with set_default_dtype(dtype):
            with no_init_weights():
                # we can probably set the device to `meta` here?
                model = AutoModelForCausalLM.from_config(config).to(dtype)

        torch.distributed.barrier(group=self.process_group)
        # print_rank_0(f"Initialized model")
        state_dict = torch.load(shard_state_dict_path)
        # TODO @thomasw21: HACK in order to transpose all weight prior
        for key in state_dict.keys():
            do_transpose = False
            if not match_suffix(key, "weight"):
                continue

            for potential_suffix in [
                "self_attention.query_key_value.weight",
                "self_attention.dense.weight",
                "dense_h_to_4h.weight",
                "dense_4h_to_h.weight",
            ]:
                if match_suffix(key, potential_suffix):
                    do_transpose = True

            if do_transpose:
                state_dict[key] = state_dict[key].transpose(1, 0).contiguous()

        model.load_state_dict(state_dict)
        self.model = model.to(self.device).eval()
        self.num_heads = config.n_head // self.process_group.size()
        torch.distributed.barrier(group=self.process_group)

    def forward(self, input_ids, attention_mask, past_key_values: Optional = None):
        outputs = self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )

        logits_shard = outputs.logits[:, -1, :].contiguous()

        batch_size, vocab_shard_size = logits_shard.shape
        vocab_size = self.world_size * vocab_shard_size
        logits = [torch.empty_like(logits_shard) for _ in range(self.world_size)]
        torch.distributed.all_gather(logits, logits_shard, group=self.process_group)
        logits = torch.cat(logits, dim=1).view(batch_size, 1, vocab_size)

        outputs.logits = logits
        return outputs
