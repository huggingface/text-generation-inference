import torch
import torch.distributed

from typing import List, Optional, Type

from accelerate import init_empty_weights
from safetensors import safe_open
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    PreTrainedTokenizerBase,
)
from transformers.models.bloom.parallel_layers import (
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    TensorParallelRowLinear,
)

from text_generation.models import CausalLM
from text_generation.models.causal_lm import CausalLMBatch
from text_generation.pb import generate_pb2
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


class BloomCausalLMBatch(CausalLMBatch):
    @classmethod
    def from_pb(
        cls,
        pb: generate_pb2.Batch,
        tokenizer: PreTrainedTokenizerBase,
        device: torch.device,
    ) -> "CausalLMBatch":
        batch = super(BloomCausalLMBatch, cls).from_pb(
            pb=pb, tokenizer=tokenizer, device=device
        )
        batch.keys_head_dim_last = False
        return batch


class BLOOM(CausalLM):
    @property
    def batch_type(self) -> Type[CausalLMBatch]:
        return BloomCausalLMBatch


class BLOOMSharded(BLOOM):
    def __init__(self, model_name: str, quantize: bool = False):
        if not model_name.startswith("bigscience/bloom"):
            raise ValueError(f"Model {model_name} is not supported")

        self.process_group, self.rank, self.world_size = initialize_torch_distributed()
        self.master = self.rank == 0
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{self.rank}")
            dtype = torch.bfloat16
        else:
            device = torch.device("cpu")
            dtype = torch.float32

        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

        config = AutoConfig.from_pretrained(
            model_name, slow_but_exact=False, tp_parallel=True
        )
        config.pad_token_id = 3

        # Only download weights for small models
        if self.master and model_name == "bigscience/bloom-560m":
            download_weights(model_name, extension=".safetensors")

        torch.distributed.barrier(group=self.process_group)
        filenames = weight_files(model_name, extension=".safetensors")
        if not filenames:
            raise ValueError("No safetensors weights found")

        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config)

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
        super(CausalLM, self).__init__(
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
                                tensor,
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
                                    out = input.new_empty(size_out)
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

    def forward(self, input_ids, attention_mask, position_ids, past_key_values: Optional = None):
        outputs = self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )

        # Logits are sharded, so we need to gather them
        logits = [torch.empty_like(outputs.logits) for _ in range(self.world_size)]
        torch.distributed.all_gather(logits, outputs.logits, group=self.process_group)
        logits = torch.cat(logits, dim=2)

        return logits, outputs.past_key_values
