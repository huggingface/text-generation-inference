import torch
import torch.distributed

from accelerate import init_empty_weights
from opentelemetry import trace
from pathlib import Path
from safetensors import safe_open
from transformers import AutoTokenizer, AutoConfig
from typing import Optional, Tuple, List

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
    def __init__(self, model_id: str, revision: Optional[str] = None, quantize=False):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            raise NotImplementedError("FlashCausalLM is only available on GPU")

        if quantize:
            raise NotImplementedError("FlashCausalLM does not support quantization")

        tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, padding_side="left"
        )

        config = AutoConfig.from_pretrained(
            model_id, revision=revision, tp_parallel=True
        )

        try:
            filenames = weight_files(model_id, revision, ".bin")
        # Local files not found
        except LocalEntryNotFoundError:
            hub_files = weight_hub_files(model_id, revision, ".bin")
            filenames = download_weights(hub_files, model_id, revision)

        with init_empty_weights():
            model = FlashLlamaForCausalLM(config)

        self.load_weights(
            model,
            filenames,
        )
        self.model = model.eval().to(device).to(dtype)

        super(FlashCausalLM, self).__init__(
            tokenizer=tokenizer,
            device=device,
        )

    @staticmethod
    def load_weights(
        model,
        filenames: List[Path],
    ):
        final_state_dict = {}
        for filename in filenames:
            state_dict = torch.load(filename, map_location="cpu")
            for key, value in state_dict.items():
                layer_name = ".".join(key.split(".")[:4])
                if "q_proj" in key:
                    final_key = layer_name + ".query_key_value.weight"
                    if final_key not in final_state_dict:
                        final_state_dict[final_key] = value.new_empty(
                            (value.shape[0] * 3, value.shape[1])
                        )
                    final_state_dict[final_key][: value.shape[0]] = value
                elif "k_proj" in key:
                    final_key = layer_name + ".query_key_value.weight"
                    if final_key not in final_state_dict:
                        final_state_dict[final_key] = value.new_empty(
                            (value.shape[0] * 3, value.shape[1])
                        )
                    final_state_dict[final_key][
                        value.shape[0] : value.shape[0] * 2
                    ] = value
                elif "v_proj" in key:
                    final_key = layer_name + ".query_key_value.weight"
                    if final_key not in final_state_dict:
                        final_state_dict[final_key] = value.new_empty(
                            (value.shape[0] * 3, value.shape[1])
                        )
                    final_state_dict[final_key][value.shape[0] * 2 :] = value
                elif "gate_proj" in key:
                    final_key = layer_name + ".gate_up_proj.weight"
                    if final_key not in final_state_dict:
                        final_state_dict[final_key] = value.new_empty(
                            (value.shape[0] * 2, value.shape[1])
                        )
                    final_state_dict[final_key][: value.shape[0]] = value
                elif "up_proj" in key:
                    final_key = layer_name + ".gate_up_proj.weight"
                    if final_key not in final_state_dict:
                        final_state_dict[final_key] = value.new_empty(
                            (value.shape[0] * 2, value.shape[1])
                        )
                    final_state_dict[final_key][value.shape[0] :] = value
                else:
                    final_state_dict[key] = value
            del state_dict

        parameters = dict(model.named_parameters())
        for key, value in final_state_dict.items():
            current_parameter_tensor = parameters.get(key, None)
            module_name, param_name = key.rsplit(".", 1)
            module = model.get_submodule(module_name)

            if (
                current_parameter_tensor is not None
                and current_parameter_tensor.shape != value.shape
            ):
                raise ValueError(
                    f"Name {key} -- Current {current_parameter_tensor.shape} and got {value.shape}"
                )

            value = value.contiguous()

            if current_parameter_tensor is not None:
                module._parameters[param_name] = value
            else:
                module._buffers[param_name] = value

        model.post_load_weights()


class FlashLlamaSharded(FlashLlama):
    def __init__(
        self, model_id: str, revision: Optional[str] = None, quantize: bool = False
    ):
        self.process_group, self.rank, self.world_size = initialize_torch_distributed()
        self.master = self.rank == 0
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{self.rank}")
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            raise NotImplementedError("FlashLlama is only available on GPU")

        if quantize:
            raise NotImplementedError("FlashLlama does not support quantization")

        tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, padding_side="left"
        )

        config = AutoConfig.from_pretrained(
            model_id, revision=revision, tp_parallel=True
        )

        torch.distributed.barrier(group=self.process_group)
        filenames = weight_files(model_id, revision=revision, extension=".safetensors")

        with init_empty_weights():
            model = FlashGPTNeoXForCausalLM(config)

        torch.distributed.barrier(group=self.process_group)
        self.load_weights(
            model,
            filenames,
            quantize=quantize,
            device=device,
            rank=self.rank,
            world_size=self.world_size,
        )
        model.post_load_weights()
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

                    tensor = tensor.contiguous()

                    if current_parameter_tensor is not None:
                        module._parameters[param_name] = tensor
                    else:
                        module._buffers[param_name] = tensor

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_s: int,
        past_key_values: Optional = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.model.gpt_neox.tp_embeddings:
            logits, present = self.model.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                cu_seqlens=cu_seqlens,
                max_s=max_s,
                past_key_values=past_key_values,
            )

            # Logits are sharded, so we need to gather them
            world_logits = [torch.empty_like(logits) for _ in range(self.world_size)]
            torch.distributed.all_gather(world_logits, logits, group=self.process_group)
            world_logits = torch.cat(world_logits, dim=1)

            return world_logits, present
        # While the model itself is sharded, the embeddings might not as they might not be dividable by num-shard
        else:
            return super(FlashLlamaSharded, self).forward(
                input_ids, position_ids, cu_seqlens, max_s, past_key_values
            )
