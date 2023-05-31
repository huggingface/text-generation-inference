import torch
import torch.distributed

from pathlib import Path
from accelerate import init_empty_weights
from opentelemetry import trace
from safetensors import safe_open
from transformers import AutoTokenizer, AutoConfig
from typing import Optional, List

from text_generation_server.models import FlashCausalLM
from text_generation_server.models.custom_modeling.flash_rw_modeling import (
    RWConfig,
    FlashRWForCausalLM,
    TensorParallelEmbedding,
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
)
from text_generation_server.utils import (
    initialize_torch_distributed,
    weight_files,
    download_weights,
    weight_hub_files,
    LocalEntryNotFoundError,
)

tracer = trace.get_tracer(__name__)


class FlashRW(FlashCausalLM):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        quantize: Optional[str] = None,
        trust_remote_code: bool = False,
    ):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            dtype = torch.float16
        else:
            raise NotImplementedError("RW is only available on GPU")

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=trust_remote_code,
        )

        config = RWConfig.from_pretrained(
            model_id,
            revision=revision,
        )

        # We do not use from_pretrained as it is too slow
        try:
            filenames = weight_files(model_id, revision, ".bin")
        # Local files not found
        except LocalEntryNotFoundError:
            hub_files = weight_hub_files(model_id, revision, ".bin")
            filenames = download_weights(hub_files, model_id, revision)

        with init_empty_weights():
            model = FlashRWForCausalLM(config)

        self.load_weights(
            model,
            filenames,
            quantize,
            device,
            dtype,
        )

        super(FlashCausalLM, self).__init__(
            model=model.to(device),
            tokenizer=tokenizer,
            requires_padding=False,
            dtype=dtype,
            device=device,
        )

    @staticmethod
    def load_weights(
        model: FlashRWForCausalLM,
        filenames: List[Path],
        quantize: Optional[str],
        device: torch.device,
        dtype: torch.dtype,
    ):
        for filename in filenames:
            state_dict = torch.load(filename, map_location="cpu")
            for key, value in state_dict.items():
                value = value.to(device if quantize is None else "cpu").to(dtype)

                module_name, param_name = key.rsplit(".", 1)
                module = model.get_submodule(module_name)

                try:
                    current_parameter_tensor = module._parameters[param_name]
                    if current_parameter_tensor.shape != value.shape:
                        raise ValueError(
                            f"Name {key} -- Current {current_parameter_tensor.shape} and got {value.shape}"
                        )
                    module._parameters[param_name] = value
                except KeyError:
                    module._buffers[param_name] = value

                del value

        torch.cuda.empty_cache()
        model.post_load_weights(quantize)


class FlashRWSharded(FlashRW):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        quantize: Optional[str] = None,
        trust_remote_code: bool = False,
    ):
        self.process_group, rank, world_size = initialize_torch_distributed()
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{rank}")
            dtype = torch.float16
        else:
            raise NotImplementedError("FlashRW is only available on GPU")

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=trust_remote_code,
        )

        config = RWConfig.from_pretrained(
            model_id, revision=revision, trust_remote_code=trust_remote_code
        )

        torch.distributed.barrier(group=self.process_group)
        filenames = weight_files(model_id, revision=revision, extension=".safetensors")

        with init_empty_weights():
            model = FlashRWForCausalLM(config, self.process_group)

        torch.distributed.barrier(group=self.process_group)
        self.load_weights(
            model,
            filenames,
            quantize=quantize,
            device=device,
            dtype=dtype,
            rank=rank,
            world_size=world_size,
        )
        torch.distributed.barrier(group=self.process_group)
        super(FlashCausalLM, self).__init__(
            model=model.to(device),
            tokenizer=tokenizer,
            requires_padding=False,
            dtype=dtype,
            device=device,
            rank=rank,
            world_size=world_size,
        )

    @staticmethod
    def load_weights(
        model,
        filenames: List[str],
        quantize: Optional[str],
        device: torch.device,
        dtype: torch.dtype,
        rank: int,
        world_size: int,
    ):
        parameters = dict(model.named_parameters())
        for file in filenames:
            with safe_open(
                file, framework="pt", device=str(device) if quantize is None else "cpu"
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
                    elif name == "lm_head.weight" and model.transformer.tp_embeddings:
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

        model.post_load_weights(quantize)
