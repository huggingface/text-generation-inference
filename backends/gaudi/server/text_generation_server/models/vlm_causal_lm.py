import re
import torch
import os
import time
import math
from PIL import Image
from io import BytesIO
from opentelemetry import trace
from loguru import logger
from typing import Iterable, Optional, Tuple, List, Type, Dict
import tempfile
import copy
from text_generation_server.models import Model
from transformers import PreTrainedTokenizerBase
from text_generation_server.utils.tokens import batch_top_tokens
from text_generation_server.pb import generate_pb2
from text_generation_server.models.causal_lm import (
    CausalLMBatch,
    CausalLMRequest,
    remove_kv_cache_from_output,
)

from transformers.models.llava_next.modeling_llava_next import (
    get_anyres_image_grid_shape,
)

from transformers import AutoProcessor
import text_generation_server.habana_quantization_env as hq_env
from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
from text_generation_server.utils import (
    HeterogeneousNextTokenChooser,
    make_tokenizer_optional,
    is_tokenizer_transparent,
    pad_next_token_chooser_parameters,
)
import habana_frameworks.torch as htorch
from optimum.habana.utils import HabanaProfile
from optimum.habana.transformers.generation import MODELS_OPTIMIZED_WITH_STATIC_SHAPES
from optimum.habana.utils import get_hpu_memory_stats
from optimum.habana.checkpoint_utils import get_ds_injection_policy

from transformers import (
    AutoTokenizer,
    AutoConfig,
)
from optimum.habana.checkpoint_utils import (
    get_repo_root,
    model_on_meta,
    write_checkpoints_json,
)

from text_generation_server.utils.speculate import get_speculate
from text_generation_server.models.types import (
    Tokens,
    Generation,
    GeneratedText,
)
from text_generation_server.utils.debug import dbg_trace

tracer = trace.get_tracer(__name__)

IDEFICS2_FAKE_TOKEN = "<fake_token_around_image>"
IDEFICS2_IMAGE_TOKEN = "<image>"


IMAGES = re.compile(r"!\[[^\]]*\]\((.*?)\s*(\"(?:.*[^\"])\")?\s*\)")
BASE_IMAGE_TOKENS = int(os.environ.get("BASE_IMAGE_TOKENS", 2048))
MAX_TOTAL_TOKENS = int(os.environ.get("MAX_TOTAL_TOKENS", 8192))
PAD_SEQUENCE_TO_MULTIPLE_OF = int(os.environ.get("PAD_SEQUENCE_TO_MULTIPLE_OF", 128))
CHUNK_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
LAZY_MODE = int(os.environ.get("PT_HPU_LAZY_MODE", 1))


PREFILL_WARMUP_BATCH_SIZE_LIST = []
PREFILL_WARMUP_SEQLEN_LIST = []
DECODE_WARMUP_BATCH_SIZE_LIST = []
CROSS_ATTENTION_LAYERS = []


def round_up(warmup_list: list, num):
    i = 0
    for i in warmup_list:
        if num <= i:
            break
    return i if i > 0 else num


def split(string) -> List[Dict[str, str]]:
    parts = []
    cursor = 0
    for pattern in IMAGES.finditer(string):
        start = pattern.start()
        if start != cursor:
            parts.append({"type": "text", "content": string[cursor:start]})

        parts.append({"type": "image", "content": pattern.group(1)})
        cursor = pattern.end()

    if cursor != len(string):
        parts.append({"type": "text", "content": string[cursor:]})

    return parts


def image_text_replacement(config) -> str:
    if config.model_type == "idefics2":
        image_seq_len = 64
        image_str = f"{IDEFICS2_FAKE_TOKEN}{IDEFICS2_IMAGE_TOKEN * image_seq_len}{IDEFICS2_FAKE_TOKEN}"
        return image_str
    elif config.model_type == "llava_next":
        return "<image>"
    elif config.model_type == "paligemma":
        return "<image>"
    elif config.model_type == "mllama":
        return "<|image|>"
    else:
        raise RuntimeError(f"Unknown config {config.model_type} for multimodal")


def image_text_replacement_fixup(config, text: str) -> str:
    if config.model_type == "idefics2":
        return text.replace(
            f"{IDEFICS2_FAKE_TOKEN}{IDEFICS2_FAKE_TOKEN}", IDEFICS2_FAKE_TOKEN
        )
    return text


def get_unpadded_features(
    original_height: int,
    original_width: int,
    npatches: int,
    num_patch_height: int,
    num_patch_width: int,
) -> Tuple[int, int]:
    current_height = npatches * num_patch_height
    current_width = npatches * num_patch_width

    aspect_ratio: float = original_width / original_height
    current_aspect_ratio: float = current_width / current_height

    if aspect_ratio > current_aspect_ratio:
        new_height = (original_height * current_width) // original_width
        padding = (current_height - new_height) // 2
        current_height = current_height - (2 * padding)
    else:
        new_width = (original_width * current_height) // original_height
        padding = (current_width - new_width) // 2
        current_width = current_width - (2 * padding)

    unpadded_features = current_height * current_width
    newline_features = current_height
    return (unpadded_features, newline_features)


def get_number_of_features(height: int, width: int, config) -> int:
    # From config
    # Hardcoded for CLIP for now
    # image_grid_pinpoints = [[336, 672], [672, 336], [672, 672], [1008, 336], [336, 1008]]
    image_grid_pinpoints = config.image_grid_pinpoints
    image_size = config.vision_config.image_size
    patch_size = config.vision_config.patch_size

    assert image_size % patch_size == 0

    npatches = image_size // patch_size

    # Dimensions are intentionally swapped to be bug-compatible with
    # upstream: https://github.com/LLaVA-VL/LLaVA-NeXT/issues/59
    num_patch_width, num_patch_height = get_anyres_image_grid_shape(
        [height, width],
        image_grid_pinpoints,
        image_size,
    )

    unpadded_features, newline_features = get_unpadded_features(
        height, width, npatches, num_patch_height, num_patch_width
    )
    # The base patch covers the entire image
    base_features = npatches**2
    return unpadded_features + newline_features + base_features


class VlmCausalLMBatch(CausalLMBatch):
    pixel_values: Optional[List[torch.Tensor]]
    pixel_attention_mask: Optional[List[torch.Tensor]]
    image_sizes: Optional[List[Tuple[int, int]]]
    aspect_ratio_ids: Optional[torch.Tensor] = None
    aspect_ratio_mask: Optional[torch.Tensor] = None
    cross_attention_mask: Optional[torch.Tensor] = None
    prefilling: bool = True
    token_idx: torch.Tensor = None

    def __init__(
        self,
        batch_id,
        requests,
        input_ids,
        attention_mask,
        position_ids,
        past_key_values,
        merged_kv_cache,
        next_token_chooser,
        top_n_tokens,
        top_n_tokens_tensor,
        input_length,
        pixel_values: Optional[List[torch.Tensor]] = None,
        pixel_attention_mask: Optional[List[torch.Tensor]] = None,
        image_sizes: Optional[List[Tuple[int, int]]] = None,
        aspect_ratio_ids: Optional[torch.Tensor] = None,
        aspect_ratio_mask: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        prefilling: Optional[bool] = True,
    ):
        super().__init__(
            batch_id=batch_id,
            requests=requests,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            merged_kv_cache=merged_kv_cache,
            next_token_chooser=next_token_chooser,
            top_n_tokens=top_n_tokens,
            top_n_tokens_tensor=top_n_tokens_tensor,
            input_length=input_length,
        )

        self.pixel_values = pixel_values
        self.pixel_attention_mask = pixel_attention_mask
        self.image_sizes = image_sizes
        self.aspect_ratio_ids = aspect_ratio_ids
        self.aspect_ratio_mask = aspect_ratio_mask
        self.cross_attention_mask = cross_attention_mask
        self.prefilling = prefilling

    @property
    def token_idx(self):
        if self.prefilling:
            # no right padding for prefill
            token_idx_scalar = self.attention_mask.shape[-1] - 1
            return torch.tensor(token_idx_scalar).to(self.attention_mask.device)
        else:
            token_idx_scalar = self.attention_mask.shape[-1] - self.right_padding
            return torch.tensor(token_idx_scalar).to(self.attention_mask.device)

    def padding_process(self, pad_id: int):
        # self.input_ids = torch.index_select(self.input_ids, 1, self.token_idx - 1)
        right_padding = MAX_TOTAL_TOKENS - self.attention_mask.shape[1]
        self.input_ids = torch.nn.functional.pad(
            self.input_ids, (0, right_padding), value=pad_id
        )
        self.attention_mask = torch.nn.functional.pad(
            self.attention_mask, (0, right_padding), value=0
        )
        # if self.position_ids is not None:
        #     self.position_ids = torch.index_select(self.position_ids, 1, self.token_idx - 1) + 1
        if self.cross_attention_mask is not None:
            self.cross_attention_mask = torch.nn.functional.pad(
                self.cross_attention_mask, (0, 0, 0, 0, 0, right_padding), value=0
            )
        if self.past is not None:
            past_key_values_list = list(self.past_key_values)
            for layer_id in range(len(self.past)):
                past_key_value_list = list(self.past_key_values[layer_id])
                if layer_id not in CROSS_ATTENTION_LAYERS:
                    past_key_value_list[0] = torch.nn.functional.pad(
                        self.past_key_values[layer_id][0],
                        (0, 0, 0, right_padding),
                        value=0,
                    )
                    past_key_value_list[1] = torch.nn.functional.pad(
                        self.past_key_values[layer_id][1],
                        (0, 0, 0, right_padding),
                        value=0,
                    )
                past_key_values_list[layer_id] = tuple(past_key_value_list)
            self.past_key_values = tuple(past_key_values_list)

        self.prefilling = False
        self.input_length = self.input_length

    @classmethod
    def from_tokenized(
        cls,
        pb: generate_pb2.Batch,
        tokenizer: PreTrainedTokenizerBase,
        batch_tokenized_inputs,
        dtype: torch.dtype,
        device: torch.device,
        is_warmup: bool = False,
    ) -> "VlmCausalLMBatch":

        dbg_trace("FROM_PB", f"num_reqs:{len(pb.requests)}")
        requests = [
            CausalLMRequest.from_pb(idx, req, tokenizer)
            for idx, req in enumerate(pb.requests)
        ]

        max_input_length = max(r.data.truncate for r in requests)
        max_new_tokens = max(r.stopping_criteria.max_new_tokens for r in requests)
        # TODO: Add support for sparse batches
        top_n_tokens = [r.top_n_tokens for r in pb.requests]
        top_n_tokens_tensor = torch.tensor(
            top_n_tokens, device=device, dtype=torch.int64
        )

        # TODO: by tokenizing all inputs at once we loose information on actual input lengths
        # this means that we cannot shift inputs to the left after a long input sequence
        # was filtered out
        new_bs = round_up(PREFILL_WARMUP_BATCH_SIZE_LIST, len(requests))
        parameters = [r.parameters for r in pb.requests]
        # append the dummy parameters for dummy request
        parameters = pad_next_token_chooser_parameters(parameters, new_bs)

        next_token_chooser = HeterogeneousNextTokenChooser.from_pb(
            pb=parameters,
            dtype=dtype,
            device=device,
            tokenizer=tokenizer,
            quantization_enabled=hq_env.is_quantization_enabled,
        )
        tokenized_inputs = batch_tokenized_inputs
        input_len = tokenized_inputs["input_ids"].shape[1]

        bucket_size = max_input_length
        left_padding = max_input_length - input_len
        if is_warmup is False:
            rounded_seq_len = round_up(PREFILL_WARMUP_SEQLEN_LIST, input_len + 1)
            bucket_size = rounded_seq_len - 1
            left_padding = bucket_size - input_len

        input_ids = tokenized_inputs["input_ids"]
        attention_mask = tokenized_inputs["attention_mask"]
        cross_attention_mask = tokenized_inputs.get("cross_attention_mask", None)
        # Allocate space for first token
        input_ids = torch.nn.functional.pad(
            input_ids, (left_padding, 1), value=tokenizer.pad_token_id
        )
        attention_mask = torch.nn.functional.pad(
            attention_mask, (left_padding, 1), value=0
        )
        if cross_attention_mask is not None:
            cross_attention_mask = torch.nn.functional.pad(
                cross_attention_mask, (0, 0, 0, 0, left_padding, 1), value=0
            )
        all_input_ids = torch.nn.functional.pad(
            input_ids, (0, max_new_tokens), value=tokenizer.pad_token_id
        ).T.split(1, dim=1)

        # New input length after left padding
        input_len = bucket_size
        for r in requests:
            r.input_length = input_len
            r.prefix_offset = input_len - 5
            r.read_offset = input_len
            r.all_input_ids = all_input_ids[r.idx]
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        cross_attention_mask = (
            cross_attention_mask.to(device)
            if cross_attention_mask is not None
            else None
        )
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        htorch.core.mark_step()

        return cls(
            batch_id=pb.id,
            requests=requests,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            merged_kv_cache=False,
            next_token_chooser=next_token_chooser,
            top_n_tokens=top_n_tokens,
            top_n_tokens_tensor=top_n_tokens_tensor,
            input_length=input_len,
            cross_attention_mask=cross_attention_mask,
        )

    @classmethod
    def batch_tokenized_inputs(
        cls,
        requests: Iterable[generate_pb2.Request],
        tokenizer,
        processor,
        config,
        is_warmup,
    ):
        image_inputs = {}
        texts = []
        images = []
        batch_tokenized_inputs = {}

        for i, r in enumerate(requests):
            # Each input is encoded into a list, where each element of this input list is either a string or a URL
            curr_text = ""
            curr_image = None
            for chunk in r.input_chunks.chunks:
                chunk_type = chunk.WhichOneof("chunk")
                if chunk_type == "text":
                    curr_text += chunk.text
                elif chunk_type == "image":
                    image = Image.open(BytesIO(chunk.image.data))
                    # TODO unsure about BOS
                    curr_image = image
                else:
                    raise RuntimeError(f"Invalid chunk type {chunk_type}")

            if image_text_replacement(config) not in curr_text:
                if "<image>" in curr_text:
                    curr_text = curr_text.replace(
                        "<image>", image_text_replacement(config)
                    )
                else:
                    curr_text = image_text_replacement(config) + curr_text

            texts.append(curr_text)
            if curr_image is not None:
                if config.model_type == "mllama":
                    images.append([curr_image])
                else:
                    images.append(curr_image)

        if is_warmup is True:
            images += [images[0]] * (len(texts) - len(images))

        missing_inputs = 0
        dummy_images = None
        if is_warmup is False:
            new_bs = round_up(PREFILL_WARMUP_BATCH_SIZE_LIST, len(requests))
            missing_inputs = new_bs - len(requests)
            if missing_inputs > 0:
                dummy_inputs = []
                if len(texts) > 0:
                    dummy_inputs = [texts[0]] * missing_inputs
                    dummy_images = [images[0]] * missing_inputs
                texts += dummy_inputs
                images += dummy_images

        processor_output = processor(
            images,
            texts,
            truncation=True,
            max_length=r.truncate,
            add_special_tokens=r.add_special_tokens,
            return_tensors="pt",
            padding_side="left",
            padding="longest",
        )
        if "input_ids" in processor_output:
            batch_tokenized_inputs.update({"input_ids": processor_output["input_ids"]})
        if "attention_mask" in processor_output:
            batch_tokenized_inputs.update(
                {"attention_mask": processor_output["attention_mask"]}
            )
        if "cross_attention_mask" in processor_output:
            batch_tokenized_inputs.update(
                {"cross_attention_mask": processor_output["cross_attention_mask"]}
            )
        if "pixel_values" in processor_output:
            image_inputs.update({"pixel_values": processor_output["pixel_values"]})
        if "pixel_attention_mask" in processor_output:
            image_inputs.update(
                {"pixel_attention_mask": processor_output["pixel_attention_mask"]}
            )
        if "aspect_ratio_ids" in processor_output:
            image_inputs.update(
                {"aspect_ratio_ids": processor_output["aspect_ratio_ids"]}
            )
        if "aspect_ratio_mask" in processor_output:
            image_inputs.update(
                {"aspect_ratio_mask": processor_output["aspect_ratio_mask"]}
            )
        if "image_sizes" in processor_output:
            image_inputs.update({"image_sizes": processor_output["image_sizes"]})

        return batch_tokenized_inputs, image_inputs

    @classmethod
    def from_pb_processor(
        cls,
        pb: generate_pb2.Batch,
        tokenizer: PreTrainedTokenizerBase,
        processor,
        config,
        dtype: torch.dtype,
        device: torch.device,
        is_warmup: bool = False,
    ) -> "VlmCausalLMBatch":
        batch_tokenized_inputs, image_inputs = cls.batch_tokenized_inputs(
            pb.requests, tokenizer, processor, config, is_warmup
        )
        batch = cls.from_tokenized(
            pb, tokenizer, batch_tokenized_inputs, dtype, device, is_warmup=is_warmup
        )
        if image_inputs is not None:
            batch.pixel_values = image_inputs["pixel_values"].to(device=device)
            if "pixel_attention_mask" in image_inputs:
                batch.pixel_attention_mask = image_inputs["pixel_attention_mask"].to(
                    device=device
                )
            else:
                batch.pixel_attention_mask = None
            if "image_sizes" in image_inputs:
                batch.image_sizes = image_inputs["image_sizes"].to(device=device)
            else:
                batch.image_sizes = None
            if "aspect_ratio_ids" in image_inputs:
                batch.aspect_ratio_ids = image_inputs["aspect_ratio_ids"].to(
                    device=device
                )
            else:
                batch.aspect_ratio_ids = None
            if "aspect_ratio_mask" in image_inputs:
                batch.aspect_ratio_mask = image_inputs["aspect_ratio_mask"].to(
                    device=device
                )
            else:
                batch.aspect_ratio_mask = None
        else:
            batch.pixel_values = None
            batch.pixel_attention_mask = None
            batch.image_sizes = None
            batch.aspect_ratio_ids = None
            batch.aspect_ratio_mask = None
            batch.cross_attention_mask = None

        return batch

    @classmethod
    @tracer.start_as_current_span("concatenate")
    def concatenate(
        cls,
        batches: List["CausalLMBatch"],
        pad_token_id: int = 0,
        is_warmup: bool = False,
    ) -> "CausalLMBatch":
        return cls.recombine(batches, pad_token_id, is_warmup)

    @classmethod
    def recombine(
        cls,
        batches: List["VlmCausalLMBatch"],
        pad_token_id: int,
        is_warmup: bool = False,
    ) -> "VlmCausalLMBatch":
        if not all(b.past_key_values is not None for b in batches):
            raise ValueError("KV cache not allocated! Cannot recombine before prefill!")
            # Used for padding

        total_requests = sum(len(b) for b in batches)
        new_bs = total_requests
        if not is_warmup:
            new_bs = round_up(DECODE_WARMUP_BATCH_SIZE_LIST, total_requests)

        if len(batches) > 1:
            scenario = "CONCAT"
        elif batches[0].prefilling:
            scenario = "SHIFT"
        else:
            return batches[0]

        dbg_trace(
            scenario,
            f"bs:{[b.batch_size for b in batches]}->{new_bs}"
            f" reqs:{[len(b) for b in batches]}",
        )

        if scenario == "SHIFT":
            batch = batches[0]
            batch.padding_process(pad_token_id)
            return batch

        total_batch_size = 0
        max_input_length = 0
        for i, batch in enumerate(batches):
            total_batch_size += len(batch)
            max_input_length = max(max_input_length, batch.input_length)
        # Batch attributes
        requests = []
        input_lengths = []
        top_n_tokens = []
        parameters = []
        fsm_grammar_states = []

        # Batch tensors
        input_ids = None
        attention_mask = None
        position_ids = None
        past_key_values = []
        top_n_tokens_tensor = None
        cross_attention_mask = None
        # Used for slicing correctly inside the tensors
        # Equivalent to a cumsum on batch sizes
        start_index = 0
        for i, batch in enumerate(batches):
            keep_indices = []
            for req in batch.requests:
                keep_indices.append(req.idx)

            requests.extend(batch.requests)
            parameters.extend([r.data.parameters for r in batch.requests])
            fsm_grammar_states.extend(
                [batch.next_token_chooser.fsm_grammar_states[i] for i in keep_indices]
            )
            input_lengths.extend([batch.input_length])
            top_n_tokens.extend([batch.top_n_tokens[i] for i in keep_indices])

            # Slicing end index for this batch
            end_index = start_index + len(batch)

            # We only concatenate batches that did at least one step
            if batch.past_key_values is None:
                raise ValueError("only concatenate prefilled batches")

            # Create empty tensor
            # input_ids is always of shape [batch_size, 1]
            # We do not need to pad it
            if input_ids is None:
                input_ids = batch.input_ids.new_empty((new_bs, MAX_TOTAL_TOKENS))
            # # Copy to correct indices

            left_offset = max_input_length - batch.input_length
            right_padding = MAX_TOTAL_TOKENS - max_input_length
            input_ids[start_index:end_index, left_offset:-right_padding] = (
                batch.input_ids[keep_indices, : batch.input_length]
            )

            # Create padded tensor
            if top_n_tokens_tensor is None:
                top_n_tokens_tensor = batches[0].top_n_tokens_tensor.new_zeros(
                    new_bs,
                )
            top_n_tokens_tensor[start_index:end_index] = batch.top_n_tokens_tensor[
                keep_indices
            ]

            if attention_mask is None:
                attention_mask = batch.attention_mask.new_zeros(
                    (new_bs, MAX_TOTAL_TOKENS),
                )

            attention_mask[
                start_index:end_index,
                left_offset:-right_padding,
            ] = batch.attention_mask[
                keep_indices,
                : batch.input_length,
            ]

            if batch.cross_attention_mask is not None:
                cross_attention_mask_shape = list(batch.cross_attention_mask.shape)
                cross_attention_mask_shape[1] = MAX_TOTAL_TOKENS
                cross_attention_mask_shape[0] = new_bs
                cross_attention_mask_shape = torch.Size(cross_attention_mask_shape)
                if cross_attention_mask is None:
                    cross_attention_mask = batch.cross_attention_mask.new_zeros(
                        cross_attention_mask_shape,
                    )
                cross_attention_mask[
                    start_index:end_index,
                    left_offset:-right_padding,
                ] = batch.cross_attention_mask[
                    keep_indices,
                    : batch.input_length,
                ]

            # Create empty tensor
            # position_ids is always of shape [batch_size, 1]
            if position_ids is None:
                position_ids = batch.position_ids.new_empty((new_bs, 1))
            position_ids[start_index:end_index] = batch.position_ids[keep_indices, :]

            # Shenanigans to get dimensions because BLOOM outputs a past with a different shape
            # BLOOM Keys:   [batch_size * num_heads, head_dim, seq_length]
            # BLOOM Values: [batch_size * num_heads, seq_length, head_dim]
            # And ensure that we can update tensors in-place
            if isinstance(batch.past_key_values, tuple):
                batch.past_key_values = [
                    [t.view(batch.batch_size, -1, *t.shape[-2:]) for t in layer]
                    for layer in batch.past_key_values
                ]
            elif len(batch.past_key_values[0][0].shape) == 3:
                for layer in batch.past_key_values:
                    for k, t in enumerate(layer):
                        layer[k] = t.view(batch.batch_size, -1, *t.shape[-2:])

            start_index = end_index

        first_past_kvs = batches[0].past_key_values
        _, num_heads, padded_sequence_length, head_dim = first_past_kvs[0][1].shape
        past_key_values = []
        for layer_id in range(len(batches[0].past_key_values)):
            if layer_id in CROSS_ATTENTION_LAYERS:
                padded_past_keys_shape = list(
                    batches[0].past_key_values[layer_id][0].shape
                )
                padded_past_keys_shape[0] = new_bs
                padded_past_keys_shape = torch.Size(padded_past_keys_shape)
            else:
                padded_past_keys_shape = (
                    new_bs,
                    num_heads,
                    MAX_TOTAL_TOKENS,
                    head_dim,
                )

            padded_past_keys = first_past_kvs[layer_id][0].new_zeros(
                padded_past_keys_shape
            )
            padded_past_values = first_past_kvs[layer_id][1].new_zeros(
                padded_past_keys_shape
            )
            start_index = 0
            for batch in batches:
                keep_indices = []
                for req in batch.requests:
                    keep_indices.append(req.idx)

                left_offset = max_input_length - batch.input_length
                right_padding = MAX_TOTAL_TOKENS - max_input_length
                past_keys = batch.past_key_values[layer_id][0]
                past_values = batch.past_key_values[layer_id][1]
                # Clear reference to the original tensor
                batch.past_key_values[layer_id] = None

                # Slicing end index for this batch
                end_index = start_index + len(batch)
                # We slice the keys to remove the padding from previous batches
                if layer_id in CROSS_ATTENTION_LAYERS:
                    padded_past_keys[start_index:end_index, :, :, :] = past_keys[
                        keep_indices, :, :, :
                    ]
                    padded_past_values[start_index:end_index, :, :, :] = past_values[
                        keep_indices, :, :, :
                    ]

                else:
                    padded_past_keys[
                        start_index:end_index, :, left_offset:-right_padding, :
                    ] = past_keys[keep_indices, :, : batch.input_length, :]
                    padded_past_values[
                        start_index:end_index, :, left_offset:-right_padding, :
                    ] = past_values[keep_indices, :, : batch.input_length, :]

                start_index = end_index

            past_key_values.append(tuple([padded_past_keys, padded_past_values]))
        past_key_values = tuple(past_key_values)

        batch_id = batches[0].batch_id
        top_n_tokens.extend([-1] * (new_bs - total_batch_size))
        fsm_grammar_states.extend([-1] * (new_bs - total_batch_size))

        for idx, req in enumerate(requests):
            req.idx = idx

        parameters = pad_next_token_chooser_parameters(parameters, new_bs)
        next_token_chooser = HeterogeneousNextTokenChooser.from_pb(
            parameters,
            batches[0].next_token_chooser.dtype,
            batches[0].next_token_chooser.device,
            batches[0].next_token_chooser.tokenizer,
            fsm_grammar_states,
            quantization_enabled=hq_env.is_quantization_enabled,
        )
        input_length = max_input_length

        htorch.core.mark_step()

        return cls(
            batch_id=batch_id,
            requests=requests,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            merged_kv_cache=False,
            next_token_chooser=next_token_chooser,
            top_n_tokens=top_n_tokens,
            top_n_tokens_tensor=top_n_tokens_tensor,
            input_length=input_length,
            pixel_values=None,
            pixel_attention_mask=None,
            image_sizes=None,
            aspect_ratio_ids=None,
            aspect_ratio_mask=None,
            cross_attention_mask=cross_attention_mask,
            prefilling=False,
        )


class VlmCausalLM(Model):
    def __init__(
        self,
        model_class,
        model_id: str,
        *,
        processor_class=AutoProcessor,
        processor_kwargs=None,
        batch_class=VlmCausalLMBatch,
        revision,
        quantize: Optional[str] = None,
        dtype,
        trust_remote_code: bool,
        **kwargs,
    ):
        adapt_transformers_to_gaudi()
        if processor_kwargs is None:
            processor_kwargs = {}
        self.processor = processor_class.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=trust_remote_code,
            **processor_kwargs,
        )
        self.batch_class = batch_class
        self.prev_bs = 0
        self.quantize = quantize

        # Create tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=trust_remote_code,
        )
        make_tokenizer_optional(tokenizer)

        # Create model
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        rank = int(os.getenv("RANK", "0"))
        dtype = torch.bfloat16 if dtype is None else dtype
        device = torch.device("hpu")

        if hq_env.is_quantization_enabled:
            htorch.core.hpu_set_env()

        if world_size > 1:
            os.environ.setdefault(
                "DEEPSPEED_USE_HABANA_FRAMEWORKS_DETERMINISTIC_API", "1"
            )
            model = self.get_deepspeed_model(model_class, model_id, dtype, revision)
            model = hq_env.prepare_model_for_quantization(model)
        else:
            get_repo_root(model_id)

            # Check support for rope scaling
            model_kwargs = {}
            config = AutoConfig.from_pretrained(model_id)
            if hasattr(config, "rope_scaling"):
                model_kwargs["rope_scaling"] = self.get_rope_scaling()

            model = model_class.from_pretrained(
                model_id,
                revision=revision,
                torch_dtype=dtype,
                trust_remote_code=trust_remote_code,
                **model_kwargs,
            )
            model = hq_env.prepare_model_for_quantization(model)
            model = model.eval().to(device)

        self.enable_hpu_graph = (
            os.getenv("ENABLE_HPU_GRAPH", "true").lower() == "true" and LAZY_MODE == 1
        )
        self.limit_hpu_graph = os.getenv("LIMIT_HPU_GRAPH", "true").lower() == "true"
        model = remove_kv_cache_from_output(model)
        if self.enable_hpu_graph:
            from habana_frameworks.torch.hpu import wrap_in_hpu_graph

            model = wrap_in_hpu_graph(model, disable_tensor_cache=True)
        else:
            if LAZY_MODE == 0:
                # It is said that "keep_input_mutations" is safe for inference to be done
                dbg_trace("TORCH COMPILE", "Torch compiling of model")
                model.model = torch.compile(
                    model.model,
                    backend="hpu_backend",
                    options={"keep_input_mutations": True},
                )

        model = hq_env.setup_quantization(model)

        if model.config.model_type not in MODELS_OPTIMIZED_WITH_STATIC_SHAPES:
            raise ValueError(f"Model type {model.config.model_type} is not supported!")

        if tokenizer.pad_token_id is None:
            if model.config.pad_token_id is not None:
                tokenizer.pad_token_id = model.config.pad_token_id
            elif model.config.eos_token_id is not None:
                if isinstance(model.config.eos_token_id, int):
                    tokenizer.pad_token_id = model.config.eos_token_id
                elif isinstance(model.config.eos_token_id, list):
                    tokenizer.pad_token_id = model.config.eos_token_id[0]
                else:
                    raise ValueError(
                        f"{type(model.config.eos_token_id)} type of eos_token_id in the model's config is not supported for tokenizer.pad_token_id"
                    )
            elif tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        self.kwargs = {
            "use_cache": True,
            "return_dict": True,
        }

        if model.config.model_type in ["llava_next"]:
            self.kwargs["attn_softmax_bf16"] = True
            self.kwargs["trim_logits"] = True

            if os.getenv("USE_FLASH_ATTENTION", "true").lower() == "true":
                self.kwargs["use_flash_attention"] = True
            if os.getenv("FLASH_ATTENTION_RECOMPUTE", "true").lower() == "true":
                self.kwargs["flash_attention_recompute"] = True

        self.speculate = get_speculate()
        if model.config.model_type == "mllama":
            global CROSS_ATTENTION_LAYERS, BASE_IMAGE_TOKENS
            CROSS_ATTENTION_LAYERS = model.config.text_config.cross_attention_layers
            BASE_IMAGE_TOKENS = 0

        super(VlmCausalLM, self).__init__(
            model_id=model_id,
            model=model,
            tokenizer=tokenizer,
            requires_padding=True,
            dtype=dtype,
            device=device,
            rank=rank,
        )

        # Create profiler
        ranks_to_profile = [int(val) for val in os.getenv("PROF_RANKS", "0").split(",")]
        record_shapes = os.getenv("PROF_RECORD_SHAPES", "false").lower() == "true"
        output_dir = os.getenv("PROF_PATH", "/tmp/hpu_profile")
        self.profiling_warmup_steps = (
            int(os.getenv("PROF_WARMUPSTEP", "0")) if rank in ranks_to_profile else 0
        )
        self.profiling_steps = (
            int(os.getenv("PROF_STEP", "0")) if rank in ranks_to_profile else 0
        )
        self.profiling_wait_steps = int(os.getenv("PROF_WAITSTEP", "0"))
        if self.profiling_steps > 0:
            self.hb_profiler = HabanaProfile(
                wait=self.profiling_wait_steps,
                warmup=self.profiling_warmup_steps,
                active=self.profiling_steps,
                output_dir=output_dir,
                record_shapes=record_shapes,
            )
            self.hb_profiler.start()
        else:
            self.hb_profiler = None
        self.step = 0

    @property
    def batch_type(self) -> Type[VlmCausalLMBatch]:
        return self.batch_class

    def max_past(self) -> Optional[int]:
        return getattr(self.model.text_model, "max_past", None)

    def get_deepspeed_model(
        self,
        model_class,
        model_id: str,
        dtype: torch.dtype,
        revision: Optional[str] = None,
    ) -> torch.nn.Module:
        import deepspeed
        from habana_frameworks.torch.distributed.hccl import initialize_distributed_hpu

        world_size, rank, local_rank = initialize_distributed_hpu()
        model_kwargs = {"revision": revision}

        # Initialize process(es) for DeepSpeed
        deepspeed.init_distributed(dist_backend="hccl")
        logger.info(
            "DeepSpeed is enabled. world_size {} rank {} local_rank {}".format(
                world_size, rank, local_rank
            )
        )
        config = AutoConfig.from_pretrained(model_id, **model_kwargs)
        load_to_meta = model_on_meta(config)

        # Check support for rope scaling
        if hasattr(config, "rope_scaling"):
            config.rope_scaling = self.get_rope_scaling()
            model_kwargs["rope_scaling"] = self.get_rope_scaling()

        if load_to_meta:
            # Construct model with fake meta tensors, later will be replaced on devices during ds-inference ckpt load
            with deepspeed.OnDevice(dtype=dtype, device="meta"):
                model = model_class.from_config(config, torch_dtype=dtype)
        else:
            get_repo_root(model_id, local_rank=os.getenv("LOCAL_RANK"))
            # TODO: revisit placement on CPU when auto-injection is possible
            with deepspeed.OnDevice(dtype=dtype, device="cpu"):
                model = model_class.from_pretrained(
                    model_id, torch_dtype=dtype, **model_kwargs
                )
        model = model.eval()

        # Initialize the model
        ds_inference_kwargs = {"dtype": dtype}
        ds_inference_kwargs["tensor_parallel"] = {"tp_size": world_size}
        ds_inference_kwargs["enable_cuda_graph"] = False
        ds_inference_kwargs["injection_policy"] = get_ds_injection_policy(
            model.language_model.config
        )

        if load_to_meta:
            # model loaded to meta is managed differently
            checkpoints_json = tempfile.NamedTemporaryFile(suffix=".json", mode="+w")
            write_checkpoints_json(model_id, local_rank, checkpoints_json)
            ds_inference_kwargs["checkpoint"] = checkpoints_json.name
        model = deepspeed.init_inference(model, **ds_inference_kwargs)

        return model.module

    def get_rope_scaling(self) -> Optional[Dict]:
        rope_scaling = os.getenv("ROPE_SCALING", None)
        if rope_scaling is None:
            return None

        rope_factor = float(os.getenv("ROPE_FACTOR", 1.0))
        return {"type": rope_scaling, "factor": float(rope_factor)}

    def decode(self, generated_ids: List[int]) -> str:
        return self.tokenizer.decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    def decode_token(
        self,
        all_input_ids: List[int],
        prefix_offset: int = 0,
        read_offset: int = 0,
    ) -> Tuple[str, int, int]:
        if is_tokenizer_transparent(self.tokenizer):
            new_text = self.tokenizer.decode(
                all_input_ids[read_offset:], skip_special_tokens=False
            )
            return new_text, read_offset, len(all_input_ids)
        else:
            return super().decode_token(all_input_ids, prefix_offset, read_offset)

    def forward(
        self,
        batch: VlmCausalLMBatch,
        bypass_hpu_graph: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        # Model Forward
        kwargs = {
            "input_ids": batch.input_ids,
            "attention_mask": batch.attention_mask,
            "past_key_values": batch.past_key_values,
            "token_idx": batch.token_idx,
            "pixel_values": batch.pixel_values,
        }

        if self.model.config.model_type == "mllama":
            kwargs["aspect_ratio_ids"] = batch.aspect_ratio_ids
            kwargs["aspect_ratio_mask"] = batch.aspect_ratio_mask
            kwargs["cross_attention_mask"] = batch.cross_attention_mask
        else:
            kwargs["image_sizes"] = batch.image_sizes

        hpu_kwargs = {}
        # Optimum Habana got "lazy_mode" key-val only supported for llama type of models
        if self.model.config.model_type == "llama":
            hpu_kwargs["lazy_mode"] = LAZY_MODE == 1

        if self.has_position_ids:
            kwargs["position_ids"] = batch.position_ids
        if bypass_hpu_graph is not None:
            hpu_kwargs["bypass_hpu_graphs"] = bypass_hpu_graph

        kwargs.update(self.kwargs)
        model_inputs = self.model.prepare_inputs_for_generation(**kwargs)

        if batch.past_key_values is not None:
            return self.model.forward(**model_inputs, **hpu_kwargs)
        else:
            outputs = self.model.forward(**model_inputs, **hpu_kwargs)
            return outputs.logits, outputs.past_key_values

    @tracer.start_as_current_span("generate_token")
    def generate_token(
        self, batches: list[VlmCausalLMBatch], is_warmup: bool = False
    ) -> Tuple[List[Generation], Optional[VlmCausalLMBatch], Tuple[int, int]]:

        start = time.time_ns()
        # Results
        generations: List[Generation] = []
        prev_batches = []
        requests_to_generate = []
        # In order to pipeline any actions on CPU we perform the operation in 3 main stages:
        # Stage 1. Collect next token ids of any previously started generations
        for batch_id, batch in enumerate(batches):
            if batch.logits is not None:
                logits = batch.logits
                past = batch.past
                prefill = batch.past_key_values is None
                if prefill:
                    # no right padding for prefill
                    token_idx_scalar = batch.attention_mask.shape[-1] - 1
                    token_idx = torch.tensor(token_idx_scalar).to(self.device)
                else:
                    token_idx_scalar = (
                        batch.attention_mask.shape[-1] - batch.right_padding
                    )
                    token_idx = torch.tensor(token_idx_scalar).to(self.device)

                # Select next token
                input_length = batch.input_length
                if logits.shape[-2] > 1:
                    next_token_ids, next_token_logprobs, logprobs, _, _ = (
                        batch.next_token_chooser(
                            batch.input_ids,
                            logits[:, input_length - 1 : input_length, :].squeeze(-2),
                            self.speculate,
                        )
                    )
                else:
                    next_token_ids, next_token_logprobs, logprobs, _, _ = (
                        batch.next_token_chooser(
                            batch.input_ids, logits.squeeze(-2), self.speculate
                        )
                    )
                # Speculation is not active for causal
                accepted_ids = torch.ones_like(batch.input_ids)[:, 0]
                batch_top_token_ids, batch_top_token_logprobs = batch_top_tokens(
                    batch.top_n_tokens,
                    batch.top_n_tokens_tensor,
                    logprobs,
                    accepted_ids,
                )

                prev_batches.append(
                    {
                        "next_token_ids": next_token_ids,
                        "next_token_logprobs": next_token_logprobs,
                    }
                )

                for req_idx, req in enumerate(batch.requests):
                    requests_to_generate.append(
                        {
                            "req": req,
                            "prev_req_idx": req.idx,
                            "batch_id": batch_id,
                            "seed": batch.next_token_chooser.seeds[req_idx],
                            "do_sample": batch.next_token_chooser.do_sample[req_idx],
                            "top_n_tokens": batch.top_n_tokens[req_idx],
                            "top_token_ids": batch_top_token_ids[req_idx],
                            "top_token_logprobs": batch_top_token_logprobs[req_idx],
                            "grammar_state": batch.next_token_chooser.fsm_grammar_states[
                                req.idx
                            ],
                        }
                    )

                htorch.core.mark_step()

                # Add new token into input_ids
                batch.input_ids.index_copy_(1, token_idx, next_token_ids.unsqueeze(1))

                # Update attention_mask as we added a new token to input_ids
                batch.attention_mask.index_fill_(1, token_idx, 1)

                # add cross-attn mask for new token
                if batch.cross_attention_mask is not None:
                    cross_attention_mask_prev = batch.cross_attention_mask
                    if token_idx is not None:
                        mask = cross_attention_mask_prev[
                            :, token_idx - 2 : token_idx - 1, ...
                        ]
                        cross_attention_mask_prev.index_copy_(1, token_idx - 1, mask)
                        batch.cross_attention_mask = cross_attention_mask_prev

                # Adjust lengths
                batch.input_length += 1
                # Update position_ids
                if prefill:
                    batch.position_ids = (
                        torch.index_select(batch.position_ids, 1, token_idx - 1) + 1
                    )
                else:
                    batch.position_ids += 1
                # Update past key values
                if prefill:
                    batch.past_key_values = past

        htorch.core.mark_step()

        # Stage 2. Prepare new batch for speculative scheduling
        if len(batches) > 1:
            batch = self.batch_type.concatenate(
                batches, self.tokenizer.pad_token_id, is_warmup
            )
        else:
            batch = batches[0]

        prefill = batch.past_key_values is None

        # Check if we need to do any bookkeeping first
        if not prefill:
            batch = self.batch_type.recombine(
                [batch], self.tokenizer.pad_token_id, is_warmup
            )

        scenario = "PREFILL" if prefill else "GENERATE"
        if (
            self.enable_hpu_graph
            and self.limit_hpu_graph
            and round_up(DECODE_WARMUP_BATCH_SIZE_LIST, batch.batch_size)
            != self.prev_bs
        ):
            self.model.clear_cache()
            self.prev_bs = round_up(DECODE_WARMUP_BATCH_SIZE_LIST, batch.batch_size)
        dbg_trace(
            scenario,
            f"bs:{batch.batch_size} num_reqs:{len(batch.requests)} seq_len:{batch.seq_length} padding:{batch.right_padding}",
        )
        # assert batch.right_padding > 0, 'No more room for next token!'

        # Execute batch
        if prefill:
            # no right padding for prefill
            # token_idx = torch.tensor(batch.attention_mask.shape[-1] - 1).to(self.device)
            batch.logits, batch.past = self.forward(
                batch,
                bypass_hpu_graph=(
                    prefill and self.limit_hpu_graph if self.enable_hpu_graph else None
                ),
            )

        elif all([req.stopping_criteria.max_new_tokens == 1 for req in batch.requests]):
            # Don't schedule next forward if max_new_tokens for all requests equals 1
            # - we've already generated the first and only needed token in the prefill phase
            pass
        else:
            # token_idx = torch.tensor(batch.attention_mask.shape[-1] - batch.right_padding).to(self.device)
            batch.logits = self.forward(
                batch,
                bypass_hpu_graph=(
                    prefill and self.limit_hpu_graph if self.enable_hpu_graph else None
                ),
            )

        if batch.pixel_values is not None:
            batch.pixel_values = None
        if batch.aspect_ratio_ids is not None:
            batch.aspect_ratio_ids = None
        if batch.aspect_ratio_mask is not None:
            batch.aspect_ratio_mask = None

        htorch.core.mark_step()

        start_decode = time.time_ns()

        # Stage 3. Finish and return previous generations
        stopped = len(requests_to_generate) > 0
        for prev_batch in prev_batches:
            prev_batch["next_token_logprobs"] = prev_batch[
                "next_token_logprobs"
            ].tolist()
            prev_batch["next_token_ids_cpu"] = prev_batch["next_token_ids"].cpu()
        htorch.core.mark_step()

        for req_data in requests_to_generate:
            req = req_data["req"]
            i = req_data["prev_req_idx"]
            prev_batch_id = req_data["batch_id"]
            assert len(prev_batches) > prev_batch_id
            next_token_ids_cpu = prev_batches[prev_batch_id]["next_token_ids_cpu"]
            next_token_logprobs = prev_batches[prev_batch_id]["next_token_logprobs"]

            request = req.data
            input_length = req.input_length
            prefix_offset = req.prefix_offset
            read_offset = req.read_offset
            do_sample = req_data["do_sample"]
            seed = req_data["seed"]
            stopping_criteria = req.stopping_criteria
            all_input_ids = req.all_input_ids
            next_token_id = next_token_ids_cpu[i]
            next_token_logprob = next_token_logprobs[i]
            top_n_tokens = req_data["top_n_tokens"]
            top_token_ids = req_data["top_token_ids"]
            top_token_logprobs = req_data["top_token_logprobs"]
            grammar_state = req_data["grammar_state"]

            # Append next token to all tokens
            all_input_ids[input_length] = next_token_id
            new_input_length = input_length + 1

            # Generated token
            if (
                is_tokenizer_transparent(self.tokenizer)
                and len(stopping_criteria.stop_sequence_criterias) == 0
            ):
                next_token_text = ""
            else:
                next_token_text, prefix_offset, read_offset = self.decode_token(
                    all_input_ids[0:new_input_length, 0], prefix_offset, read_offset
                )

            # Evaluate stopping criteria
            stop, reason = stopping_criteria(
                next_token_id,
                next_token_text,
            )

            if not stop:
                stopped = False

            # Shard generations
            # All generations will be appended in the rust sharded client
            if i % self.world_size == self.rank:
                if stop:
                    # Decode generated tokens
                    if is_tokenizer_transparent(self.tokenizer):
                        output_text = None
                    else:
                        output_text = self.decode(
                            all_input_ids[
                                new_input_length
                                - stopping_criteria.current_tokens : new_input_length,
                                0,
                            ]
                        )
                    generated_text = GeneratedText(
                        output_text,
                        stopping_criteria.current_tokens,
                        reason,
                        seed if do_sample else None,
                    )
                else:
                    generated_text = None

                # Prefill
                if stopping_criteria.current_tokens == 1 and request.prefill_logprobs:
                    # Remove generated token to only have prefill and add nan for first prompt token
                    prefill_logprobs = [float("nan")] + next_token_logprobs
                    prefill_token_ids = all_input_ids[0 : new_input_length - 1]
                    prefill_texts = self.tokenizer.batch_decode(
                        prefill_token_ids,
                        clean_up_tokenization_spaces=False,
                        skip_special_tokens=False,
                    )
                    prefill_tokens = Tokens(
                        prefill_token_ids,
                        prefill_logprobs,
                        prefill_texts,
                        is_special=[],
                    )
                else:
                    prefill_tokens = None

                if top_n_tokens > 0:
                    all_top_tokens = []
                    for top_token_ids, top_token_logprobs in zip(
                        top_token_ids, top_token_logprobs
                    ):
                        toptoken_texts = self.tokenizer.batch_decode(
                            top_token_ids,
                            clean_up_tokenization_spaces=False,
                            skip_special_tokens=False,
                        )
                        special_toptokens = [
                            token_id in self.all_special_ids
                            for token_id in top_token_ids
                        ]
                        top_tokens = Tokens(
                            top_token_ids,
                            top_token_logprobs,
                            toptoken_texts,
                            special_toptokens,
                        )
                        all_top_tokens.append(top_tokens)
                    top_tokens = all_top_tokens
                else:
                    top_tokens = None

                generation = Generation(
                    request.id,
                    prefill_tokens,
                    Tokens(
                        [next_token_id],
                        [next_token_logprob],
                        [next_token_text],
                        [next_token_id in self.all_special_ids],
                    ),
                    generated_text,
                    top_tokens,
                )

                generations.append(generation)

            batch.next_token_chooser = (
                batch.next_token_chooser.advance_grammar_single_with_past_state(
                    req.idx, next_token_id, grammar_state
                )
            )

            req.all_input_ids = all_input_ids
            req.input_length = new_input_length
            req.prefix_offset = prefix_offset
            req.read_offset = read_offset

        htorch.core.mark_step()
        self.step = self.step + 1
        if self.hb_profiler is not None:
            if (
                self.step
                > self.profiling_wait_steps
                + self.profiling_warmup_steps
                + self.profiling_steps
            ):
                self.hb_profiler.stop()
            else:
                self.hb_profiler.step()

        forward_ns = start_decode - start
        decode_ns = time.time_ns() - start_decode
        return generations, batch if not stopped else None, (forward_ns, decode_ns)

    def batch_from_pb(self, batch, is_warmup):
        return self.batch_type.from_pb_processor(
            batch,
            self.tokenizer,
            self.processor,
            self.model.config,
            self.dtype,
            self.device,
            is_warmup,
        )

    def generate_warmup_batch(self, request, seq_len, batch_size, is_warmup):
        batch = copy.deepcopy(request.batch)
        for req in batch.requests:
            req.truncate = seq_len

        for i in range(len(batch.requests) - batch_size):
            batch.requests.pop()

        return self.batch_from_pb(batch, is_warmup)

    def warmup(
        self, request: generate_pb2.WarmupRequest
    ) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        global MAX_TOTAL_TOKENS
        MAX_TOTAL_TOKENS = request.max_total_tokens
        batch = self.batch_from_pb(request.batch, is_warmup=True)
        max_input_tokens = request.max_input_tokens
        max_prefill_batch_size = batch.input_ids.shape[0]
        max_batch_size_str = os.environ.get("MAX_BATCH_SIZE")
        if max_batch_size_str is not None:
            MAX_BATCH_SIZE = int(max_batch_size_str)
        else:
            raise ValueError("MAX_BATCH_SIZE is not set")

        try:
            # max prefill batch size warmup
            _, prefill_batch, _ = self.generate_token([batch], is_warmup=True)
        except Exception:
            raise RuntimeError(
                f"Not enough memory to handle {len(batch.input_ids)} prefill tokens. "
                f"You need to decrease `--max-batch-prefill-tokens`"
            )

        global BASE_IMAGE_TOKENS, PREFILL_WARMUP_BATCH_SIZE_LIST, PREFILL_WARMUP_SEQLEN_LIST, DECODE_WARMUP_BATCH_SIZE_LIST
        PREFILL_WARMUP_BATCH_SIZE_LIST = []
        batch_size = 1
        while batch_size <= max_prefill_batch_size:
            PREFILL_WARMUP_BATCH_SIZE_LIST.append(batch_size)
            batch_size = batch_size * 2
        if PREFILL_WARMUP_BATCH_SIZE_LIST[-1] < max_prefill_batch_size:
            PREFILL_WARMUP_BATCH_SIZE_LIST.append(max_prefill_batch_size)

        if self.model.config.model_type == "mllama":
            seq_len = PAD_SEQUENCE_TO_MULTIPLE_OF
        else:
            seq_len = BASE_IMAGE_TOKENS

        PREFILL_WARMUP_SEQLEN_LIST = []
        i = 0
        while seq_len <= max_input_tokens:
            PREFILL_WARMUP_SEQLEN_LIST.append(seq_len)
            seq_len += PAD_SEQUENCE_TO_MULTIPLE_OF * (2**i)
            i += 1
        if PREFILL_WARMUP_SEQLEN_LIST[-1] < max_input_tokens:
            PREFILL_WARMUP_SEQLEN_LIST.append(max_input_tokens)

        # Prefill and decode warmup
        DECODE_WARMUP_BATCH_SIZE_LIST = []
        prefill_batch = None
        decode_batch = None
        try:
            for batch_size in PREFILL_WARMUP_BATCH_SIZE_LIST:
                for seq_len in PREFILL_WARMUP_SEQLEN_LIST:
                    batch = self.generate_warmup_batch(
                        request, seq_len, batch_size, is_warmup=True
                    )
                    _, prefill_batch, _ = self.generate_token([batch], is_warmup=True)
                    assert prefill_batch is not None
                    _, decode_batch, _ = self.generate_token(
                        [prefill_batch], is_warmup=True
                    )

                DECODE_WARMUP_BATCH_SIZE_LIST.append(batch_size)

        except Exception:
            raise RuntimeError(
                f"Not enough memory to handle following prefill and decode warmup."
                f"Prefill batch size list:{PREFILL_WARMUP_BATCH_SIZE_LIST}"
                f"Prefill sequence length list:{PREFILL_WARMUP_SEQLEN_LIST}"
                f"Decode batch size list:{DECODE_WARMUP_BATCH_SIZE_LIST}"
                f"You need to decrease `--max-batch-prefill-tokens`"
            )

        mem_stats = get_hpu_memory_stats(self.device)
        logger.info(
            f"\nFollowing prefill and decode warmup successfully.\n"
            f"Prefill batch size list:{PREFILL_WARMUP_BATCH_SIZE_LIST}\n"
            f"Prefill sequence length list:{PREFILL_WARMUP_SEQLEN_LIST}\n"
            f"Decode batch size list:{DECODE_WARMUP_BATCH_SIZE_LIST}\n"
            f"Memory stats: {mem_stats} "
        )

        max_decode_batch_size = MAX_BATCH_SIZE
        batch_size = max_prefill_batch_size * 2
        # Decode warmup with bigger batch_size
        try:
            if (
                DECODE_WARMUP_BATCH_SIZE_LIST[-1] < max_decode_batch_size
                and batch_size <= max_decode_batch_size
            ):
                batches = []
                while batch_size <= max_decode_batch_size:
                    for i in range(int(batch_size / max_prefill_batch_size)):
                        batch = self.generate_warmup_batch(
                            request,
                            PREFILL_WARMUP_SEQLEN_LIST[0] - 1,
                            max_prefill_batch_size,
                            is_warmup=True,
                        )
                        _, prefill_batch, _ = self.generate_token(
                            [batch], is_warmup=True
                        )
                        batches.append(prefill_batch)

                    _, decode_batch, _ = self.generate_token(batches, is_warmup=True)
                    DECODE_WARMUP_BATCH_SIZE_LIST.append(batch_size)
                    batch_size = batch_size * 2
                    batches.clear()

                if DECODE_WARMUP_BATCH_SIZE_LIST[-1] < max_decode_batch_size:
                    max_decode_batch_size = math.floor(max_decode_batch_size / 2) * 2
                    batch_size = max_decode_batch_size
                    for i in range(int(max_decode_batch_size / 2)):
                        batch = self.generate_warmup_batch(
                            request,
                            PREFILL_WARMUP_SEQLEN_LIST[0] - 1,
                            2,
                            is_warmup=True,
                        )
                        _, prefill_batch, _ = self.generate_token(
                            [batch], is_warmup=True
                        )
                        batches.append(prefill_batch)
                    _, decode_batch, _ = self.generate_token(batches, is_warmup=True)
                    DECODE_WARMUP_BATCH_SIZE_LIST.append(max_decode_batch_size)

        except Exception:
            raise RuntimeError(
                f"Not enough memory to handle batch_size({batch_size}) decode warmup."
                f"Decode batch size list:{DECODE_WARMUP_BATCH_SIZE_LIST}"
                f"max_decode_batch_size is {max_decode_batch_size}"
                f"You need to decrease env `MAX_BATCH_SIZE` or '--max_batch_size'"
            )

        mem_stats = get_hpu_memory_stats(self.device)
        logger.info(
            f"\nFollowing decode warmup successfully.\n"
            f"Decode batch size list:{DECODE_WARMUP_BATCH_SIZE_LIST}\n"
            f"Memory stats: {mem_stats}"
        )

        max_supported_total_tokens = MAX_BATCH_SIZE * MAX_TOTAL_TOKENS
        max_input_tokens = max_input_tokens
        max_total_tokens = MAX_TOTAL_TOKENS

        return max_supported_total_tokens, max_input_tokens, max_total_tokens
