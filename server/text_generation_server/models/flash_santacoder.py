import torch
import torch.distributed

from accelerate import init_empty_weights
from opentelemetry import trace
from safetensors import safe_open
from pathlib import Path
from transformers import AutoTokenizer, GPT2Config
from typing import Optional, List

from text_generation_server.models import FlashCausalLM
from text_generation_server.models.custom_modeling.flash_santacoder_modeling import (
    FlashSantacoderForCausalLM,
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
)
from text_generation_server.utils import (
    initialize_torch_distributed,
    weight_files,
    download_weights,
    weight_hub_files,
    LocalEntryNotFoundError,
)

tracer = trace.get_tracer(__name__)


class FlashSantacoder(FlashCausalLM):
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
            raise NotImplementedError("FlashSantacoder is only available on GPU")

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=trust_remote_code,
        )

        config = GPT2Config.from_pretrained(
            model_id,
            revision=revision,
        )

        # We do not use from_pretrained as we modified the model internal module layout
        filenames = weight_files(model_id, revision, ".safetensors")

        with init_empty_weights():
            model = FlashSantacoderForCausalLM(config)

        self.load_weights(
            model,
            filenames,
            quantize,
            device,
            dtype,
            config.architectures[0].startswith("GPT2"),
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
        model: FlashSantacoderForCausalLM,
        filenames: List[Path],
        quantize: Optional[str],
        device: torch.device,
        dtype: torch.dtype,
        transpose: bool,
    ):
        for filename in filenames:
            with safe_open(
                filename,
                framework="pt",
                device=str(device) if quantize is None else "cpu",
            ) as f:
                for key in f.keys():
                    value = f.get_tensor(key)
                    value = value.to(device if quantize is None else "cpu").to(dtype)

                    layer_name = ".".join(key.split(".")[:4])

                    # Fused qkv
                    if "q_attn.weight" in key or "kv_attn.weight" in key:
                        final_key = layer_name + ".c_attn.weight"
                    elif "q_attn.bias" in key or "kv_attn.bias" in key:
                        final_key = layer_name + ".c_attn.bias"

                    else:
                        final_key = key

                    module_name, param_name = final_key.rsplit(".", 1)
                    module = model.get_submodule(module_name)

                    try:
                        current_parameter_tensor = module._parameters[param_name]
                    except KeyError:
                        current_parameter_tensor = None

                    if current_parameter_tensor is not None:
                        if transpose and (
                            "c_fc.weight" in key
                            or "c_proj.weight" in key
                            or "q_attn.weight" in key
                            or "kv_attn.weight" in key
                            or "c_attn.weight" in key
                        ):
                            # Tranpose as we use nn.Linear instead of Conv1D
                            value = value.T

                        if current_parameter_tensor.device == torch.device("meta"):
                            # Init qkv
                            if "c_attn.weight" in final_key:
                                module._parameters[param_name] = value.new_empty(
                                    (
                                        model.transformer.head_size
                                        * (model.transformer.num_heads + 2),
                                        value.shape[1],
                                    )
                                )
                            elif "c_attn.bias" in final_key:
                                module._parameters[param_name] = value.new_empty(
                                    (
                                        model.transformer.head_size
                                        * (model.transformer.num_heads + 2)
                                    )
                                )

                        # Copy to correct slice
                        if "q_attn.weight" in key:
                            module._parameters[param_name][: value.shape[0]] = value
                        elif "q_attn.bias" in key:
                            module._parameters[param_name][: value.shape[0]] = value
                        elif "kv_attn.weight" in key:
                            module._parameters[param_name][
                                model.transformer.head_size
                                * model.transformer.num_heads :
                            ] = value
                        elif "kv_attn.bias" in key:
                            module._parameters[param_name][
                                model.transformer.head_size
                                * model.transformer.num_heads :
                            ] = value
                        else:
                            if current_parameter_tensor.shape != value.shape:
                                raise ValueError(
                                    f"Name {final_key} -- Current {current_parameter_tensor.shape} and got {value.shape}"
                                )
                            module._parameters[param_name] = value
                    else:
                        module._buffers[param_name] = value

                    del value

        if model.lm_head.weight.device == torch.device("meta"):
            model.lm_head.weight = torch.nn.Parameter(model.transformer.wte.weight)

        torch.cuda.empty_cache()
        model.post_load_weights(quantize)

        uninitialized_parameters = []
        for n, p in model.named_parameters():
            if p.data.device == torch.device("meta"):
                uninitialized_parameters.append(n)
        if uninitialized_parameters:
            raise RuntimeError(
                f"found uninitialized parameters in model : {uninitialized_parameters}"
            )

    def decode(self, generated_ids: List[int]) -> str:
        # Do not skip special tokens as they are used for custom parsing rules of the generated text
        return self.tokenizer.decode(
            generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )


class FlashSantacoderSharded(FlashSantacoder):
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
            raise NotImplementedError("FlashSantacoderSharded is only available on GPU")

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=trust_remote_code,
        )

        config = GPT2Config.from_pretrained(
            model_id,
            revision=revision,
        )

        torch.distributed.barrier(group=self.process_group)
        filenames = weight_files(model_id, revision=revision, extension=".safetensors")

        with init_empty_weights():
            model = FlashSantacoderForCausalLM(config, self.process_group)

        torch.distributed.barrier(group=self.process_group)
        self.load_weights(
            model,
            filenames,
            quantize=quantize,
            device=device,
            dtype=dtype,
            rank=rank,
            world_size=world_size,
            transpose=config.architectures[0].startswith("GPT2"),
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
        transpose: bool,
    ):
        for file in filenames:
            with safe_open(
                file, framework="pt", device=str(device) if quantize is None else "cpu"
            ) as f:
                for key in f.keys():
                    slice_ = f.get_slice(key)

                    layer_name = ".".join(key.split(".")[:4])

                    # Fused qkv
                    if "q_attn.weight" in key or "kv_attn.weight" in key:
                        final_key = layer_name + ".c_attn.weight"
                    elif "q_attn.bias" in key or "kv_attn.bias" in key:
                        final_key = layer_name + ".c_attn.bias"
                    else:
                        final_key = key

                    module_name, param_name = final_key.rsplit(".", 1)
                    module = model.get_submodule(module_name)

                    if isinstance(module, TensorParallelColumnLinear):
                        dim = 1 if transpose and "weight" in param_name else 0
                        size = slice_.get_shape()[dim]
                        block_size = size // world_size
                        start = rank * block_size
                        stop = (rank + 1) * block_size
                        tensor = (
                            slice_[start:stop] if dim == 0 else slice_[:, start:stop]
                        )
                    elif isinstance(module, TensorParallelRowLinear):
                        if param_name == "weight":
                            dim = 0 if transpose else 1
                            size = slice_.get_shape()[dim]
                            block_size = size // world_size
                            start = rank * block_size
                            stop = (rank + 1) * block_size
                            tensor = (
                                slice_[start:stop]
                                if dim == 0
                                else slice_[:, start:stop]
                            )
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
                    elif key == "lm_head.weight" and model.transformer.tp_embeddings:
                        size = slice_.get_shape()[0]
                        block_size = size // world_size
                        start = rank * block_size
                        stop = (rank + 1) * block_size
                        tensor = slice_[start:stop]
                    else:
                        try:
                            tensor = slice_[:]
                        except:
                            tensor = f.get_tensor(key)

                    tensor = tensor.contiguous().to(dtype)

                    try:
                        current_parameter_tensor = module._parameters[param_name]
                    except KeyError:
                        current_parameter_tensor = None

                    if current_parameter_tensor is not None:
                        if transpose and (
                            "c_fc.weight" in key
                            or "c_proj.weight" in key
                            or "q_attn.weight" in key
                            or "kv_attn.weight" in key
                            or "c_attn.weight" in key
                        ):
                            # Tranpose as we use nn.Linear instead of Conv1D
                            tensor = tensor.T

                        if current_parameter_tensor.device == torch.device("meta"):
                            # Init qkv
                            if "c_attn.weight" in final_key:
                                module._parameters[param_name] = tensor.new_empty(
                                    (
                                        model.transformer.head_size
                                        * (model.transformer.num_heads + 2),
                                        tensor.shape[1],
                                    )
                                )
                            elif "c_attn.bias" in final_key:
                                module._parameters[param_name] = tensor.new_empty(
                                    (
                                        model.transformer.head_size
                                        * (model.transformer.num_heads + 2)
                                    )
                                )

                        # Copy to correct slice
                        if "q_attn" in key:
                            size = tensor.shape[0]
                            block_size = size // world_size
                            start = rank * block_size
                            stop = (rank + 1) * block_size
                            tensor = tensor[start:stop]
                            module._parameters[param_name][: tensor.shape[0]] = tensor
                        elif "kv_attn.weight" in key:
                            module._parameters[param_name][
                                model.transformer.head_size
                                * model.transformer.num_heads :
                            ] = tensor
                        elif "kv_attn.bias" in key:
                            module._parameters[param_name][
                                model.transformer.head_size
                                * model.transformer.num_heads :
                            ] = tensor
                        elif "c_attn" in key:
                            # Slice q_tensor by shard
                            q_tensor = tensor[: -2 * model.transformer.head_size]
                            block_size = q_tensor.shape[0] // world_size
                            start = rank * block_size
                            stop = (rank + 1) * block_size
                            q_tensor = q_tensor[start:stop]

                            module._parameters[param_name][
                                : q_tensor.shape[0]
                            ] = q_tensor

                            # Kv tensor is copied for every shard
                            kv_tensor = tensor[-2 * model.transformer.head_size :]
                            module._parameters[param_name][
                                q_tensor.shape[0] :
                            ] = kv_tensor
                        else:
                            if current_parameter_tensor.shape != tensor.shape:
                                raise ValueError(
                                    f"Name {key} -- Current {current_parameter_tensor.shape} and got {tensor.shape}"
                                )

                            module._parameters[param_name] = tensor
                    else:
                        module._buffers[param_name] = tensor

        if model.lm_head.weight.device == torch.device("meta"):
            model.lm_head.weight = torch.nn.Parameter(model.transformer.wte.weight)

        torch.cuda.empty_cache()
        model.post_load_weights(quantize)
