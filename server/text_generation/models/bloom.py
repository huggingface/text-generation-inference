import torch
import torch.distributed

from typing import List, Optional, Tuple, Type

from accelerate import init_empty_weights
from safetensors import safe_open
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.models.bloom.parallel_layers import (
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    TensorParallelRowLinear,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from text_generation.models import Model
from text_generation.models.types import Batch, GeneratedText
from text_generation.utils import (
    initialize_torch_distributed,
    weight_files,
    download_weights,
)

HAS_BITS_AND_BYTES = True
try:
    import bitsandbytes as bnb
    from bitsandbytes.nn import Int8Params
except Exception as e:
    HAS_BITS_AND_BYTES = False

torch.manual_seed(0)


class BloomBatch(Batch):
    @classmethod
    def concatenate(cls, batches: List["Batch"]) -> "BloomBatch":
        # Used for padding
        total_batch_size = sum(batch.size for batch in batches)
        max_sequence_length = max(batch.max_sequence_length for batch in batches)

        # Batch attributes
        input_ids = {"input_ids": None, "attention_mask": None, "past_key_values": []}
        requests = []
        all_input_lengths = []
        all_input_ids = []
        next_token_choosers = []
        stopping_criterias = []

        # Used for slicing correctly inside the tensors
        # Equivalent to a cumsum on batch sizes
        start_index = 0
        for i, batch in enumerate(batches):
            requests.extend(batch.requests)
            all_input_lengths.extend(batch.all_input_lengths)
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
            start_index:end_index, -batch.max_sequence_length:
            ] = batch.input_ids["attention_mask"][:, -batch.max_sequence_length:]

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
                start_index:end_index, :, :, -(batch.max_sequence_length - 1):
                ] = past_keys[:, :, :, -(batch.max_sequence_length - 1):]

                input_ids["past_key_values"][j][1][
                start_index:end_index, :, -(batch.max_sequence_length - 1):, :
                ] = past_values[:, :, -(batch.max_sequence_length - 1):, :]

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
            all_input_lengths=all_input_lengths,
            input_ids=input_ids,
            all_input_ids=all_input_ids,
            next_token_choosers=next_token_choosers,
            stopping_criterias=stopping_criterias,
            size=total_batch_size,
            max_sequence_length=max_sequence_length,
        )


class BLOOM(Model):
    def __init__(self, model_name: str):
        if not model_name.startswith("bigscience/bloom"):
            raise ValueError(f"Model {model_name} is not supported")

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            self.device = torch.device("cpu")
            dtype = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map="auto" if torch.cuda.is_available() else None
        ).eval()

        self.num_heads = self.model.config.num_attention_heads

    @property
    def batch_type(self) -> Type[BloomBatch]:
        return BloomBatch

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
            self, batch: BloomBatch
    ) -> Tuple[List[GeneratedText], Optional[BloomBatch]]:
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

        next_batch = BloomBatch(
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


class BLOOMSharded(BLOOM):
    def __init__(self, model_name: str, quantize: bool = False):
        super(Model, self).__init__()
        if not model_name.startswith("bigscience/bloom"):
            raise ValueError(f"Model {model_name} is not supported")

        self.process_group, self.rank, self.world_size = initialize_torch_distributed()
        self.master = self.rank == 0
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.rank}")
            dtype = torch.float16
        else:
            self.device = torch.device("cpu")
            dtype = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

        config = AutoConfig.from_pretrained(
            model_name, slow_but_exact=False, tp_parallel=True
        )
        config.pad_token_id = 3
        self.num_heads = config.n_head // self.process_group.size()

        # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
        # in PyTorch 1.12 and later.
        torch.backends.cuda.matmul.allow_tf32 = True

        # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
        torch.backends.cudnn.allow_tf32 = True

        # Only download weights for small models
        if self.master and model_name == "bigscience/bloom-560m":
            download_weights(model_name, extension=".safetensors")

        torch.distributed.barrier(group=self.process_group)
        filenames = weight_files(model_name, extension=".safetensors")

        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config)

        torch.distributed.barrier(group=self.process_group)
        self.load_weights(
            model,
            filenames,
            quantize=quantize,
            device=self.device,
            rank=self.rank,
            world_size=self.world_size,
        )
        self.model = model.eval().to(dtype)
        torch.distributed.barrier(group=self.process_group)

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
                    full_name = f"transformer.{name}"

                    module_name, param_name = full_name.rsplit(".", 1)
                    module = model.get_submodule(module_name)
                    current_tensor = parameters[full_name]

                    slice_ = f.get_slice(name)

                    if isinstance(module, TensorParallelColumnLinear):
                        if param_name == "weight":
                            size = slice_.get_shape()[0]
                            block_size = size // world_size
                            start = rank * block_size
                            stop = (rank + 1) * block_size
                            tensor = slice_[start:stop]
                            tensor = tensor.transpose(1, 0)
                        else:
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
                            tensor = tensor.transpose(1, 0)
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
                    else:
                        tensor = slice_[:]

                    if current_tensor.shape != tensor.shape:
                        raise ValueError(
                            f"Name {name} -- Current {current_tensor.shape} and got {tensor.shape}"
                        )

                    tensor = tensor.contiguous()

                    if quantize:
                        if not HAS_BITS_AND_BYTES:
                            raise ImportError(
                                "bitsandbytes is not available on your machine either because it is not installed "
                                "or you don't have a GPU.\n"
                                "You can install it with `pip install bitsandbytes`."
                            )

                        if (
                                type(module)
                                in [TensorParallelRowLinear, TensorParallelColumnLinear]
                                and param_name == "weight"
                        ):
                            tensor = Int8Params(
                                tensor.transpose(1, 0),
                                has_fp16_weights=False,
                                requires_grad=False,
                            ).to(device)
                            state = bnb.MatmulLtState()
                            state.threshold = 6.0
                            state.has_fp16_weights = False
                            state.memory_efficient_backward = False
                            state.use_pool = True
                            state.CB = tensor.CB
                            state.SCB = tensor.SCB
                            tensor.CB = None
                            tensor.SCB = None

                            def replace_linear(state, in_features, out_features):
                                def linear(input, weight, bias):
                                    size_out = input.size()[:-1] + (out_features,)
                                    input = input.view(-1, in_features)
                                    out = torch.empty(
                                        size_out, device=input.device, dtype=input.dtype
                                    )
                                    out = bnb.matmul(
                                        input,
                                        weight,
                                        out=out.view(-1, out_features),
                                        state=state,
                                        threshold=state.threshold,
                                        bias=bias,
                                    )

                                    if state.CB is not None:
                                        # we converted 8-bit row major to turing/ampere format
                                        # in the first inference pass
                                        # we no longer need the row-major weight
                                        del state.CB
                                        weight.data = state.CxB

                                    return out.view(size_out)

                                return linear

                            module.linear = replace_linear(
                                state, module.in_features, module.out_features
                            )

                        else:
                            tensor = tensor.to(device)

                    module._parameters[param_name] = tensor
                    if name == "word_embeddings.weight":
                        model.lm_head._parameters["weight"] = tensor

    def forward(self, input_ids, attention_mask, past_key_values: Optional = None):
        outputs = self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )

        # Logits are sharded, so we need to gather them
        logits_shard = outputs.logits[:, -1, :].contiguous()

        batch_size, vocab_shard_size = logits_shard.shape
        vocab_size = self.world_size * vocab_shard_size
        logits = [torch.empty_like(logits_shard) for _ in range(self.world_size)]
        torch.distributed.all_gather(logits, logits_shard, group=self.process_group)
        logits = torch.cat(logits, dim=1).view(batch_size, 1, vocab_size)

        outputs.logits = logits
        return outputs
