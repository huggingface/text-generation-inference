import inspect
import torch

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, TypeVar, Type
from transformers import PreTrainedTokenizerBase, PretrainedConfig

from text_generation_server.models.types import Batch, GeneratedText
from text_generation_server.pb.generate_pb2 import InfoResponse

from text_generation_server.utils.gptq.quant_linear import Ex4bitLinear
from custom_kernels.exllama import prepare_buffers, set_tuning_params

from text_generation_server.utils.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear
)

B = TypeVar("B", bound=Batch)

class Model(ABC):
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        config: PretrainedConfig,
        requires_padding: bool,
        dtype: torch.dtype,
        device: torch.device,
        rank: int = 0,
        world_size: int = 1,
    ):
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(1.0)

        self.model = model.eval()
        self.tokenizer = tokenizer
        self.all_special_ids = set(tokenizer.all_special_ids)
        self.requires_padding = requires_padding
        self.dtype = dtype
        self.device = device
        self.rank = rank
        self.world_size = world_size

        self.has_position_ids = (
            inspect.signature(model.forward).parameters.get("position_ids", None)
            is not None
        )
        self.config = config

        if config.quantize == "gptq":
            # Buffers need to be persistent to avoid any bug.
            self.buffers = {}
            use_exllama_act_order = False
            max_dq_buffer_size = 1
            max_inner_outer_dim = 1
            for name, submodule in model.named_modules():
                if isinstance(submodule, (TensorParallelColumnLinear, TensorParallelRowLinear)) and isinstance(submodule.linear, Ex4bitLinear):

                    max_dq_buffer_size = max(max_dq_buffer_size, submodule.linear.qweight.numel() * 8)

                    if submodule.linear.act_order:
                        max_inner_outer_dim = max(max_inner_outer_dim, submodule.linear.height, submodule.linear.width)

                        use_exllama_act_order = True
                                    
            if use_exllama_act_order:
                # TODO: this should be set to rust side `max_total_tokens`, but TGI
                # does not offer an API to expose this variable to python, as this variable
                # is handled by the client but it appears the model is initialized by the server.
                # An alternative could be to initialize the buffers during warmup.
                max_total_tokens = 2048
            else:
                max_total_tokens = 1

            # This temp_state buffer is required to reorder X in the act-order case.
            self.buffers["temp_state"] = torch.zeros((max_total_tokens, max_inner_outer_dim), dtype=torch.float16, device=device)

            # This temp_dq buffer is required to dequantize weights when using cuBLAS, typically for the prefill.
            self.buffers["temp_dq"] = torch.zeros((1, max_dq_buffer_size), dtype=torch.float16, device=device)
            
            prepare_buffers(device, self.buffers["temp_state"], self.buffers["temp_dq"])

            matmul_recons_thd = 8
            matmul_fused_remap = False
            matmul_no_half2 = False
            set_tuning_params(matmul_recons_thd, matmul_fused_remap, matmul_no_half2)

            torch.cuda.empty_cache()

        self.check_initialized()

    @property
    def info(self) -> InfoResponse:
        return InfoResponse(
            requires_padding=self.requires_padding,
            dtype=str(self.dtype),
            device_type=self.device.type,
        )

    @property
    @abstractmethod
    def batch_type(self) -> Type[B]:
        raise NotImplementedError

    @abstractmethod
    def generate_token(self, batch: B) -> Tuple[List[GeneratedText], Optional[B]]:
        raise NotImplementedError

    def warmup(self, batch: B, max_total_tokens: int):
        self.generate_token(batch)

    def decode_token(
        self,
        all_input_ids: List[int],
        prefix_offset: int = 0,
        read_offset: int = 0,
    ) -> Tuple[str, int, int]:
        """Hack to hopefully support generate_stream for the maximum number of tokenizers"""

        # The prefix text is necessary only to defeat cleanup algorithms in the decode
        # which decide to add a space or not depending on the surrounding ids.
        prefix_text = self.tokenizer.decode(
            all_input_ids[prefix_offset:read_offset], skip_special_tokens=False
        )
        new_text = self.tokenizer.decode(
            all_input_ids[prefix_offset:], skip_special_tokens=False
        )

        if len(new_text) > len(prefix_text) and not new_text.endswith("�"):
            # utf-8 char at the end means it's a potential unfinished byte sequence
            # from byte fallback tokenization.
            # If it's in the middle, it's probably a real invalid id generated
            # by the model
            new_text = new_text[len(prefix_text) :]
            return new_text, read_offset, len(all_input_ids)
        else:
            return "", prefix_offset, read_offset

    def check_initialized(self):
        uninitialized_parameters = []
        for n, p in self.model.named_parameters():
            if p.data.device == torch.device("meta"):
                uninitialized_parameters.append(n)
        if uninitialized_parameters:
            raise RuntimeError(
                f"found uninitialized parameters in model {self.__class__.__name__}: {uninitialized_parameters}"
            )
