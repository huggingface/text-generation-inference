import torch
import torch.distributed

from accelerate import init_empty_weights
from opentelemetry import trace
from safetensors import safe_open
from transformers import AutoTokenizer, AutoConfig
from typing import Optional, List

from text_generation_server.models import FlashCausalLM
from text_generation_server.models.custom_modeling.flash_neox_modeling import (
    FlashGPTNeoXForCausalLM,
    TensorParallelEmbedding,
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
)
from text_generation_server.utils import (
    initialize_torch_distributed,
    weight_files,
)

tracer = trace.get_tracer(__name__)


class FlashNeoX(FlashCausalLM):
    def __init__(self, model_id: str, revision: Optional[str] = None, quantize=False):
        super(FlashNeoX, self).__init__(
            FlashGPTNeoXForCausalLM, model_id, revision, quantize
        )


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
            model_id, revision=revision, padding_side="left", truncation_side="left"
        )

        config = AutoConfig.from_pretrained(
            model_id,
            revision=revision,
        )

        torch.distributed.barrier(group=self.process_group)
        filenames = weight_files(model_id, revision=revision, extension=".safetensors")

        with init_empty_weights():
            model = FlashGPTNeoXForCausalLM(config, self.process_group)

        torch.distributed.barrier(group=self.process_group)
        self.load_weights(
            model,
            filenames,
            device=device,
            dtype=dtype,
            rank=self.rank,
            world_size=self.world_size,
        )
        model.post_load_weights()
        self.model = model.eval()
        torch.distributed.barrier(group=self.process_group)
        super(FlashCausalLM, self).__init__(
            tokenizer=tokenizer,
            device=device,
        )

    @staticmethod
    def load_weights(
        model,
        filenames: List[str],
        device: torch.device,
        dtype: torch.dtype,
        rank: int,
        world_size: int,
    ):
        parameters = dict(model.named_parameters())
        for file in filenames:
            with safe_open(file, framework="pt", device=str(device)) as f:
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

                    tensor = tensor.contiguous().to(dtype)

                    if current_parameter_tensor is not None:
                        module._parameters[param_name] = tensor
                    else:
                        module._buffers[param_name] = tensor
