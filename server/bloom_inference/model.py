import torch
import torch.distributed

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.modeling_utils import no_init_weights

from bloom_inference.cache import CacheEntry
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
    request_ids: List[int]
    input_ids: Dict[str, torch.Tensor]
    all_input_ids: List[torch.Tensor]
    next_token_choosers: List[NextTokenChooser]
    stopping_criterias: List[StoppingCriteria]

    @classmethod
    def from_batch_pb(
            cls, pb: generate_pb2.Batch, tokenizer: AutoTokenizer, device: torch.device
    ) -> "Batch":
        request_ids = []
        inputs = []
        next_token_choosers = []
        stopping_criterias = []

        # Parse batch
        for r in pb.requests:
            request_ids.append(r.id)
            inputs.append(r.inputs)
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
        all_input_ids = input_ids["input_ids"].unsqueeze(-1)

        return cls(
            pb.id,
            request_ids,
            input_ids,
            all_input_ids,
            next_token_choosers,
            stopping_criterias,
        )

    @classmethod
    def from_cache_entry(cls, cache_entry: CacheEntry) -> "Batch":
        return cls(
            cache_entry.batch_id,
            cache_entry.request_ids,
            cache_entry.input_ids,
            cache_entry.all_input_ids,
            cache_entry.next_token_choosers,
            cache_entry.stopping_criterias,
        )

    @classmethod
    def from_batch_cached_pb(cls, pb: generate_pb2.BatchCached, cache) -> "Batch":
        if len(pb.batch_cached_ids) == 1:
            cache_entry = cache.pop(pb.batch_cached_ids[0])
            if cache_entry is None:
                raise ValueError(f"Batch ID {pb.batch_id} not found in cache")
            return cls.from_cache_entry(cache_entry)

        total_batch_size = pb.total_batch_size
        max_sequence_length = pb.max_sequence_length
        input_ids = {"input_ids": None, "attention_mask": None, "past_key_values": []}
        request_ids = []
        all_input_ids = []
        next_token_choosers = []
        stopping_criterias = []
        start_index = 0
        for i, batch_id in enumerate(pb.batch_cached_ids):
            cache_entry = cache.pop(batch_id)
            if cache_entry is None:
                raise ValueError(f"Batch ID {batch_id} not found in cache")
            request_ids.extend(cache_entry.request_ids)
            all_input_ids.extend(cache_entry.all_input_ids)
            next_token_choosers.extend(cache_entry.next_token_choosers)
            stopping_criterias.extend(cache_entry.stopping_criterias)

            batch_size = len(cache_entry.request_ids)
            end_index = start_index + batch_size
            sequence_length = max(len(entry) for entry in cache_entry.all_input_ids)

            if input_ids["input_ids"] is None:
                input_ids["input_ids"] = torch.empty(
                    (total_batch_size, 1),
                    dtype=cache_entry.input_ids["input_ids"].dtype,
                    device=cache_entry.input_ids["input_ids"].device,
                )

            input_ids["input_ids"][start_index:end_index] = cache_entry.input_ids[
                "input_ids"
            ]

            if input_ids["attention_mask"] is None:
                input_ids["attention_mask"] = torch.zeros(
                    (total_batch_size, max_sequence_length),
                    dtype=cache_entry.input_ids["attention_mask"].dtype,
                    device=cache_entry.input_ids["attention_mask"].device,
                )

            input_ids["attention_mask"][
            start_index:end_index, -sequence_length:
            ] = cache_entry.input_ids["attention_mask"][:, -sequence_length:]

            for j, past in enumerate(cache_entry.input_ids["past_key_values"]):
                # TODO: this could be done without the views by using indices
                past_keys = past[0]
                past_values = past[1]

                _, head_dim, padded_sequence_length = past_keys.shape

                past_keys = past_keys.view(
                    batch_size, -1, head_dim, padded_sequence_length
                )
                past_values = past_values.view(
                    batch_size, -1, padded_sequence_length, head_dim
                )
                num_heads = past_keys.shape[1]

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

                input_ids["past_key_values"][j][0][
                start_index:end_index, :, :, -(sequence_length - 1):
                ] = past_keys[:, :, :, -(sequence_length - 1):]

                input_ids["past_key_values"][j][1][
                start_index:end_index, :, -(sequence_length - 1):, :
                ] = past_values[:, :, -(sequence_length - 1):, :]

                if (i + 1) == len(pb.batch_cached_ids):
                    input_ids["past_key_values"][j][0] = input_ids["past_key_values"][
                        j
                    ][0].view(total_batch_size * num_heads, head_dim, -1)
                    input_ids["past_key_values"][j][1] = input_ids["past_key_values"][
                        j
                    ][1].view(total_batch_size * num_heads, -1, head_dim)

            start_index += batch_size

        assert pb.request_ids == request_ids

        return cls(
            pb.id,
            request_ids,
            input_ids,
            all_input_ids,
            next_token_choosers,
            stopping_criterias,
        )


@dataclass
class FinishedGeneration:
    request_id: str
    output: str

    def to_pb(self) -> generate_pb2.FinishedGeneration:
        return generate_pb2.FinishedGeneration(id=self.request_id, output=self.output)


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
    ) -> Tuple[List[FinishedGeneration], Optional[CacheEntry]]:
        with torch.no_grad():
            outputs = self.forward(**batch.input_ids)

        # List of indices to cache
        cache_indices = []
        cache_past_indices = []

        # New input_ids for next forward; keep in cache
        cache_next_input_ids = []
        cache_all_input_ids = []

        # Finished requests
        finished_generations: List[FinishedGeneration] = []

        # Zipped iterator
        iterator = zip(
            batch.request_ids,
            outputs.logits,
            batch.next_token_choosers,
            batch.stopping_criterias,
            batch.all_input_ids,
        )

        # For each member of the batch
        for i, (
                request_id,
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
                # Add to the list of finished generations with the original request id
                finished_generations.append(FinishedGeneration(request_id, output))
            # must be added to the cache
            else:
                cache_indices.append(i)
                cache_past_indices.extend([j for j in range(i * self.num_heads, (i + 1) * self.num_heads)])
                cache_next_input_ids.append(next_token)
                cache_all_input_ids.append(all_tokens)

        # No cache is needed, we finished all generations in the batch
        if not cache_indices:
            return finished_generations, None

        # If we finished at least one generation
        cache_input_ids = {"input_ids": torch.cat(cache_next_input_ids, dim=0)}
        if finished_generations:
            # Apply indices to attention mask, past key values and other items that need to be cached
            cache_input_ids["attention_mask"] = batch.input_ids["attention_mask"][
                cache_indices
            ]
            cache_input_ids["past_key_values"] = [
                (keys[cache_past_indices], values[cache_past_indices])
                for keys, values in outputs["past_key_values"]
            ]
            cache_request_ids = [batch.request_ids[i] for i in cache_indices]
            cache_next_token_choosers = [
                batch.next_token_choosers[i] for i in cache_indices
            ]
            cache_stopping_criterias = [
                batch.stopping_criterias[i] for i in cache_indices
            ]
        else:
            cache_input_ids["attention_mask"] = batch.input_ids["attention_mask"]
            cache_input_ids["past_key_values"] = outputs["past_key_values"]
            cache_request_ids = batch.request_ids
            cache_next_token_choosers = batch.next_token_choosers
            cache_stopping_criterias = batch.stopping_criterias

        # Update attention_mask with padding as we added a new token to input_ids
        cache_input_ids["attention_mask"] = torch.cat(
            [
                cache_input_ids["attention_mask"],
                torch.ones((cache_input_ids["attention_mask"].shape[0], 1)).to(
                    cache_input_ids["attention_mask"].device
                ),
            ],
            dim=1,
        )

        cache_entry = CacheEntry(
            batch.batch_id,
            cache_request_ids,
            cache_input_ids,
            cache_all_input_ids,
            cache_next_token_choosers,
            cache_stopping_criterias,
        )
        return finished_generations, cache_entry


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
