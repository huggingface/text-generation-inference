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
    def __init__(self, model_id: str, revision: Optional[str] = None, quantize=False):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            raise NotImplementedError("FlashSantacoder is only available on GPU")

        if quantize:
            raise NotImplementedError("FlashSantacoder does not support quantization")

        tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, padding_side="left", truncation_side="left"
        )

        config = GPT2Config.from_pretrained(
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
            model = FlashSantacoderForCausalLM(config)

        self.load_weights(
            model,
            filenames,
            device,
            dtype,
        )
        self.model = model.eval().to(device).to(dtype)

        super(FlashCausalLM, self).__init__(
            tokenizer=tokenizer,
            device=device,
        )

    @staticmethod
    def load_weights(
        model: FlashSantacoderForCausalLM,
        filenames: List[Path],
        device: torch.device,
        dtype: torch.dtype,
    ):
        for filename in filenames:
            state_dict = torch.load(filename, map_location="cpu")
            for key, value in state_dict.items():
                value = value.to(device).to(dtype)

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
                    if (
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
                            model.transformer.head_size * model.transformer.num_heads :
                        ] = value
                    elif "kv_attn.bias" in key:
                        module._parameters[param_name][
                            model.transformer.head_size * model.transformer.num_heads :
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

        torch.cuda.empty_cache()
        model.post_load_weights()

    def decode(self, generated_ids: List[int]) -> str:
        # Do not skip special tokens as they are used for custom parsing rules of the generated text
        return self.tokenizer.decode(
            generated_ids, skip_special_tokens=False, cleanup_tokenization_spaces=False
        )


class FlashSantacoderSharded(FlashSantacoder):
    def __init__(
        self, model_id: str, revision: Optional[str] = None, quantize: bool = False
    ):
        self.process_group, self.rank, self.world_size = initialize_torch_distributed()
        self.master = self.rank == 0
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{self.rank}")
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            raise NotImplementedError("FlashSantacoder is only available on GPU")

        if quantize:
            raise NotImplementedError("FlashSantacoder does not support quantization")

        tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, padding_side="left"
        )

        config = GPT2Config.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=True,  # Needed as the config is not part of Transformers
        )

        torch.distributed.barrier(group=self.process_group)
        filenames = weight_files(model_id, revision=revision, extension=".safetensors")

        with init_empty_weights():
            model = FlashSantacoderForCausalLM(config, self.process_group)

        torch.distributed.barrier(group=self.process_group)
        self.load_weights(
            model,
            filenames,
            device=device,
            rank=self.rank,
            world_size=self.world_size,
        )
        self.model = model.eval().to(dtype)
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
        rank: int,
        world_size: int,
    ):
        for file in filenames:
            with safe_open(file, framework="pt", device=str(device)) as f:
                for name in f.keys():
                    slice_ = f.get_slice(name)

                    layer_name = ".".join(name.split(".")[:4])

                    # Fused qkv
                    if "q_attn.weight" in name or "kv_attn.weight" in name:
                        final_name = layer_name + ".c_attn.weight"
                    elif "q_attn.bias" in name or "kv_attn.bias" in name:
                        final_name = layer_name + ".c_attn.bias"
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
                    elif "c_attn" in name:
                        size = slice_.get_shape()[0]
                        block_size =
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

                    tensor = tensor.contiguous()

                    try:
                        current_parameter_tensor = module._parameters[param_name]
                    except KeyError:
                        current_parameter_tensor = None

                    if current_parameter_tensor is not None:
                        if current_parameter_tensor.device == torch.device("meta"):
                            # Init qkv
                            if "c_attn.weight" in final_name:
                                module._parameters[param_name] = tensor.new_empty(
                                    (
                                        model.transformer.head_size
                                        * (model.transformer.num_heads + 2),
                                        tensor.shape[1],
                                    )
                                )
                            elif "c_attn.bias" in final_name:
                                module._parameters[param_name] = tensor.new_empty(
                                    (
                                        model.transformer.head_size
                                        * (model.transformer.num_heads + 2)
                                    )
                                )

                        # Copy to correct slice
                        if "q_attn.weight" in name:
                            module._parameters[param_name][: tensor.shape[0]] = tensor
                        elif "q_attn.bias" in name:
                            module._parameters[param_name][: tensor.shape[0]] = tensor
                        elif "kv_attn.weight" in name:
                            module._parameters[param_name][
                                model.transformer.head_size
                                * model.transformer.num_heads :
                            ] = tensor
                        elif "kv_attn.bias" in name:
                            module._parameters[param_name][
                                model.transformer.head_size
                                * model.transformer.num_heads :
                            ] = tensor
                        else:
                            if current_parameter_tensor.shape != tensor.shape:
                                raise ValueError(
                                    f"Name {name} -- Current {current_parameter_tensor.shape} and got {tensor.shape}"
                                )

                            module._parameters[param_name] = tensor

                    else:
                        module._buffers[param_name] = tensor
        torch.cuda.empty_cache()
        model.post_load_weights()
