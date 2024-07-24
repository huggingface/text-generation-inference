import inspect
import torch

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, TypeVar, Type, Dict, DefaultDict
from collections import defaultdict
from transformers import PreTrainedTokenizerBase

from text_generation_server.models.types import Batch, Generation
from text_generation_server.utils.speculate import get_speculate
from text_generation_server.pb.generate_pb2 import InfoResponse
from text_generation_server.adapters.weights import LayerAdapterWeights

BASE_MODEL_ADAPTER_ID = "__base_model__"


B = TypeVar("B", bound=Batch)


class Model(ABC):
    def __init__(
        self,
        model_id: str,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        requires_padding: bool,
        dtype: torch.dtype,
        device: torch.device,
        rank: int = 0,
        world_size: int = 1,
        sliding_window: Optional[int] = None,
        speculate: Optional[int] = None,
        adapter_id: str = BASE_MODEL_ADAPTER_ID,
    ):
        self.model_id = model_id
        self.model = model.eval()
        self.tokenizer = tokenizer

        # all_special_ids is not set correctly if the rust tokenizer is unpacked
        # TODO report this to transformers.
        other_special_ids = {
            id for id, token in tokenizer.added_tokens_decoder.items() if token.special
        }
        self.all_special_ids = set(tokenizer.all_special_ids)
        self.all_special_ids.update(other_special_ids)
        self.requires_padding = requires_padding
        self.dtype = dtype
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.sliding_window = sliding_window if sliding_window != -1 else None

        self.layer_to_adapter_weights: Dict[str, LayerAdapterWeights] = defaultdict(
            LayerAdapterWeights
        )
        self.loaded_adapters = set()
        self.static_adapter_id = adapter_id

        if speculate is None:
            speculate = get_speculate()
        self.speculate = speculate

        self.has_position_ids = (
            inspect.signature(model.forward).parameters.get("position_ids", None)
            is not None
        )

        self.check_initialized()

    @property
    def info(self) -> InfoResponse:
        if self.requires_padding and self.sliding_window is not None:
            raise NotImplementedError("sliding_window is not implemented with padding")

        return InfoResponse(
            requires_padding=self.requires_padding,
            dtype=str(self.dtype),
            device_type=self.device.type,
            window_size=self.sliding_window,
            speculate=self.speculate,
        )

    @property
    @abstractmethod
    def batch_type(self) -> Type[B]:
        raise NotImplementedError

    @abstractmethod
    def generate_token(
        self, batch: B
    ) -> Tuple[List[Generation], Optional[B], Tuple[int, int]]:
        raise NotImplementedError

    def warmup(self, batch: B) -> Optional[int]:
        self.generate_token(batch)
        return None

    def decode_token(
        self,
        all_input_ids: List[int],
        prefix_offset: int = 0,
        read_offset: int = 0,
        skip_special_tokens: bool = False,
    ) -> Tuple[str, int, int]:
        """Hack to hopefully support generate_stream for the maximum number of tokenizers"""

        # The prefix text is necessary only to defeat cleanup algorithms in the decode
        # which decide to add a space or not depending on the surrounding ids.
        prefix_text = self.tokenizer.decode(
            all_input_ids[prefix_offset:read_offset],
            skip_special_tokens=skip_special_tokens,
        )
        new_text = self.tokenizer.decode(
            all_input_ids[prefix_offset:], skip_special_tokens=skip_special_tokens
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
