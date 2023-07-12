import inspect
import torch

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, TypeVar, Type
from transformers import PreTrainedTokenizerBase

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

        if model.config.quantize == "gptq":
            # Buffers need to be persistent to avoid any bug.
            self.buffers = {}
            max_dq_buffer_size = 0
            for name, submodule in self.model.named_modules():
                if isinstance(submodule, (TensorParallelColumnLinear, TensorParallelRowLinear)) and isinstance(submodule.linear, Ex4bitLinear):
                    max_dq_buffer_size = max(max_dq_buffer_size, submodule.linear.qweight.numel() * 8)
            
            intermediate_size = model.config.n_inner
            max_seq_len = 2048  # TODO: we should be able to set it
            
            self.buffers["temp_state"] = torch.zeros((max_seq_len, intermediate_size), dtype=torch.float16, device=device)
            self.buffers["temp_dq"] = torch.zeros((1, max_dq_buffer_size), dtype=torch.float16, device=device)

            prepare_buffers(device, self.buffers["temp_state"], self.buffers["temp_dq"])

            # TODO: ability to set them
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
