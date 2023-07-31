import inspect
import torch

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, TypeVar, Type
from transformers import PreTrainedTokenizerBase, PretrainedConfig

from text_generation_server.models.types import Batch, Generation, TopToken
from text_generation_server.pb.generate_pb2 import InfoResponse

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
    def generate_token(self, batch: B) -> Tuple[List[Generation], Optional[B]]:
        raise NotImplementedError

    def warmup(self, batch: B) -> Optional[int]:
        self.generate_token(batch)
        return None

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

    def decode_tokens(
        self,
        input_ids: List[int],
        new_input_ids: List[int],
        prefix_offset: int = 0,
        read_offset: int = 0,
    ) -> Tuple[str, int, int]:
        """Version of decode_token that supports multiple new tokens for the same prefix."""

        # The prefix text is necessary only to defeat cleanup algorithms in the decode
        # which decide to add a space or not depending on the surrounding ids.
        prefix_text = self.tokenizer.decode(
            input_ids[prefix_offset:read_offset], skip_special_tokens=False
        )

        new_sequences = [
            input_ids[prefix_offset:] + [new_id] for new_id in new_input_ids
        ]
        new_texts = self.tokenizer.batch_decode(
            new_sequences, skip_special_tokens=False
        )

        prefix_len = len(prefix_text)
        results = []
        for new_text in new_texts:
            if len(new_text) > prefix_len and not new_text.endswith("�"):
                # utf-8 char at the end means it's a potential unfinished byte sequence
                # from byte fallback tokenization.
                # If it's in the middle, it's probably a real invalid id generated
                # by the model
                new_text = new_text[prefix_len:]
                results.append((new_text, read_offset, len(input_ids) + 1))
            else:
                results.append(("", prefix_offset, read_offset))
        return results

    def decode_top_tokens(
        self,
        input_ids,
        top_n_tokens,
        top_token_ids,
        top_token_logprobs,
        prefix_offset,
        read_offset,
    ):
        if top_n_tokens == 0:
            return []

        top_token_texts = self.decode_tokens(
            input_ids=input_ids,
            new_input_ids=top_token_ids,
            prefix_offset=prefix_offset,
            read_offset=read_offset,
        )

        top_tokens = []
        for token_id, (top_token_text, _, _), token_logprob in zip(
            top_token_ids, top_token_texts, top_token_logprobs
        ):
            tok_itm = token_id
            top_tokens.append(
                TopToken(
                    token_id=token_id,
                    token_logprob=token_logprob,
                    token_text=top_token_text,
                    token_is_special=tok_itm in self.all_special_ids,
                )
            )
        return top_tokens

    def check_initialized(self):
        uninitialized_parameters = []
        for n, p in self.model.named_parameters():
            if p.data.device == torch.device("meta"):
                uninitialized_parameters.append(n)
        if uninitialized_parameters:
            raise RuntimeError(
                f"found uninitialized parameters in model {self.__class__.__name__}: {uninitialized_parameters}"
            )
