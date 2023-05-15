import torch
import torch.distributed

from accelerate import init_empty_weights
from opentelemetry import trace
from pathlib import Path
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.llama import LlamaTokenizer
from typing import Optional, List

from text_generation_server.models import FlashCausalLM
from text_generation_server.models.custom_modeling.flash_llama_modeling import (
    FlashLlamaForCausalLM,
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


class FlashLlama(FlashCausalLM):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        quantize: Optional[str] = None,
    ):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            dtype = torch.float16
        else:
            raise NotImplementedError("FlashLlama is only available on GPU")

        tokenizer = LlamaTokenizer.from_pretrained(
            model_id,
            revision=revision,
            padding_side="left",
            truncation_side="left",
        )

        config = AutoConfig.from_pretrained(
            model_id,
            revision=revision,
        )

        # We do not use from_pretrained as we modified the model internal module layout
        try:
            filenames = weight_files(model_id, revision, ".bin")
        # Local files not found
        except LocalEntryNotFoundError:
            hub_files = weight_hub_files(model_id, revision, ".bin")
            filenames = download_weights(hub_files, model_id, revision)

        with init_empty_weights():
            model = FlashLlamaForCausalLM(config)

        self.load_weights(model, filenames, quantize, device, dtype)
        self.model = model.eval().to(device)

        super(FlashCausalLM, self).__init__(
            tokenizer=tokenizer,
            requires_padding=False,
            dtype=dtype,
            device=device,
        )

    @staticmethod
    def load_weights(
        model,
        filenames: List[Path],
        quantize: Optional[str],
        device: torch.device,
        dtype: torch.dtype,
    ):
        for filename in filenames:
            state_dict = torch.load(filename, map_location="cpu")
            for key, value in state_dict.items():
                value = value.to(device if quantize is None else "cpu").to(dtype)

                layer_name = ".".join(key.split(".")[:4])

                # Fused qkv
                if "q_proj" in key or "k_proj" in key or "v_proj" in key:
                    final_key = layer_name + ".query_key_value.weight"

                # Fused gate and up projs
                elif "gate_proj" in key or "up_proj" in key:
                    final_key = layer_name + ".gate_up_proj.weight"
                else:
                    final_key = key

                module_name, param_name = final_key.rsplit(".", 1)
                module = model.get_submodule(module_name)

                try:
                    current_parameter_tensor = module._parameters[param_name]
                except KeyError:
                    current_parameter_tensor = None

                if current_parameter_tensor is not None:
                    if current_parameter_tensor.device == torch.device("meta"):
                        # Init qkv
                        if "query_key_value" in final_key:
                            module._parameters[param_name] = value.new_empty(
                                (value.shape[0] * 3, value.shape[1])
                            )
                        # Init gate and up proj
                        elif "gate_up_proj" in final_key:
                            module._parameters[param_name] = value.new_empty(
                                (value.shape[0] * 2, value.shape[1])
                            )

                    # Copy to correct slice
                    if "q_proj" in key:
                        module._parameters[param_name][: value.shape[0]] = value
                    elif "k_proj" in key:
                        module._parameters[param_name][
                            value.shape[0] : value.shape[0] * 2
                        ] = value
                    elif "v_proj" in key:
                        module._parameters[param_name][value.shape[0] * 2 :] = value
                    elif "gate_proj" in key:
                        module._parameters[param_name][: value.shape[0]] = value
                    elif "up_proj" in key:
                        module._parameters[param_name][value.shape[0] :] = value
                    else:
                        if current_parameter_tensor.shape != value.shape:
                            raise ValueError(
                                f"Name {final_key} -- Current {current_parameter_tensor.shape} and got {value.shape}"
                            )
                        module._parameters[param_name] = value
                else:
                    module._buffers[param_name] = value

                del value

        torch.cuda.empty_cache()
        model.post_load_weights(quantize)


class FlashLlamaSharded(FlashLlama):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        quantize: Optional[str] = None,
    ):
        self.process_group, rank, world_size = initialize_torch_distributed()
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{rank}")
            dtype = torch.float16
        else:
            raise NotImplementedError("FlashLlama is only available on GPU")

        tokenizer = LlamaTokenizer.from_pretrained(
            model_id,
            revision=revision,
            padding_side="left",
            truncation_side="left",
        )

        config = AutoConfig.from_pretrained(
            model_id,
            revision=revision,
        )

        torch.distributed.barrier(group=self.process_group)
        filenames = weight_files(model_id, revision=revision, extension=".safetensors")

        with init_empty_weights():
            model = FlashLlamaForCausalLM(config, process_group=self.process_group)

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
        self.model = model.eval().to(device)
        torch.distributed.barrier(group=self.process_group)
        super(FlashCausalLM, self).__init__(
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
        for file in filenames:
            with safe_open(
                file, framework="pt", device=str(device) if quantize is None else "cpu"
            ) as f:
                for name in f.keys():
                    slice_ = f.get_slice(name)

                    layer_name = ".".join(name.split(".")[:4])

                    # Fused qkv
                    if "q_proj" in name or "k_proj" in name or "v_proj" in name:
                        final_name = layer_name + ".query_key_value.weight"

                    # Fused gate and up projs
                    elif "gate_proj" in name or "up_proj" in name:
                        final_name = layer_name + ".gate_up_proj.weight"
                    else:
                        final_name = name

                    module_name, param_name = final_name.rsplit(".", 1)
                    module = model.get_submodule(module_name)

                    if isinstance(module, TensorParallelColumnLinear):
                        size = slice_.get_shape()[0]
                        block_size = size // world_size
                        start = rank * block_size
                        stop = (rank + 1) * block_size
                        tensor = slice_[start:stop]
                    elif isinstance(module, TensorParallelRowLinear):
                        size = slice_.get_shape()[1]
                        block_size = size // world_size
                        start = rank * block_size
                        stop = (rank + 1) * block_size
                        tensor = slice_[:, start:stop]
                    elif isinstance(module, TensorParallelEmbedding):
                        size = slice_.get_shape()[0]
                        block_size = size // world_size
                        start = rank * block_size
                        stop = (rank + 1) * block_size
                        tensor = slice_[start:stop]
                    elif name == "lm_head.weight" and model.model.tp_embeddings:
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

                    tensor = tensor.contiguous().to(dtype)

                    try:
                        current_parameter_tensor = module._parameters[param_name]
                    except KeyError:
                        current_parameter_tensor = None

                    if current_parameter_tensor is not None:
                        if current_parameter_tensor.device == torch.device("meta"):
                            # Init qkv
                            if "query_key_value" in final_name:
                                module._parameters[param_name] = tensor.new_empty(
                                    (tensor.shape[0] * 3, tensor.shape[1])
                                )
                            # Init gate and up proj
                            elif "gate_up_proj" in final_name:
                                module._parameters[param_name] = tensor.new_empty(
                                    (tensor.shape[0] * 2, tensor.shape[1])
                                )

                        # Init gate and up proj
                        if "q_proj" in name:
                            module._parameters[param_name][: tensor.shape[0]] = tensor
                        elif "k_proj" in name:
                            module._parameters[param_name][
                                tensor.shape[0] : tensor.shape[0] * 2
                            ] = tensor
                        elif "v_proj" in name:
                            module._parameters[param_name][
                                tensor.shape[0] * 2 :
                            ] = tensor
                        elif "gate_proj" in name:
                            module._parameters[param_name][: tensor.shape[0]] = tensor
                        elif "up_proj" in name:
                            module._parameters[param_name][tensor.shape[0] :] = tensor
                        else:
                            if current_parameter_tensor.shape != tensor.shape:
                                raise ValueError(
                                    f"Name {name} -- Current {current_parameter_tensor.shape} and got {tensor.shape}"
                                )

                            module._parameters[param_name] = tensor

                    else:
                        module._buffers[param_name] = tensor

        torch.cuda.empty_cache()
        model.post_load_weights(quantize)
