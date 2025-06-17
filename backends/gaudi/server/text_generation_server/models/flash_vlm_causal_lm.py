import torch
from PIL import Image
from io import BytesIO
from dataclasses import dataclass
from opentelemetry import trace
from typing import Iterable, Optional, Tuple, List, Type, Dict

from transformers import PreTrainedTokenizerBase
from transformers.image_processing_utils import select_best_resolution
from text_generation_server.pb import generate_pb2
from text_generation_server.models.flash_causal_lm import (
    FlashCausalLMBatch,
    FlashCausalLM,
    prepare_for_decode,
)
from text_generation_server.models.globals import PREFIX_CACHING, BLOCK_SIZE
from loguru import logger
from text_generation_server.utils.log import log_master
from transformers import AutoProcessor
from text_generation_server.layers.attention import (
    Seqlen,
    trim_seqlen_metadata,
    _async_h2d_tensor_copy,
)
import habana_frameworks.torch as htorch
import time
from text_generation_server.utils.import_utils import (
    synchronize,
)
from vllm_hpu_extension.profiler import HabanaMemoryProfiler, format_bytes

tracer = trace.get_tracer(__name__)

IDEFICS2_FAKE_TOKEN = "<fake_token_around_image>"
IDEFICS2_IMAGE_TOKEN = "<image>"

IDEFICS3_IMAGE_TOKEN = "<image>"
IDEFICS3_FAKE_IMAGE_TOKEN = "<fake_token_around_image>"
IDEFICS3_GLOBAL_IMG_TOKEN = "<global-img>"


def prompt_split_image_llama4(aspect_ratio, num_patches_per_chunk):
    """
    Create a structured string representation of image tokens

    Args:
       num_patches: Number of patches in the image

    Returns:
        String with appropriate image tokens
    """
    img_string = "<|image_start|>"
    ratio_h, ratio_w = aspect_ratio
    if ratio_h * ratio_w > 1:
        for yy in range(ratio_h):
            for xx in range(ratio_w):
                img_string += "<|patch|>" * num_patches_per_chunk
                if xx < ratio_w - 1:
                    img_string += "<|tile_x_separator|>"

            img_string += "<|tile_y_separator|>"
    img_string += "<|image|>"
    img_string += "<|patch|>" * num_patches_per_chunk
    img_string += "<|image_end|>"

    return img_string


# copied from: https://github.com/huggingface/transformers/blob/02ed609285c2448b3b54c31e362f2c389fa952ab/src/transformers/models/idefics3/processing_idefics3.py#L44-L60
def _prompt_split_image(
    *,
    image_seq_len: int,
    image_rows: int,
    image_cols: int,
    fake_token_around_image: str,
    image_token: str,
    global_img_token: str,
):
    """Prompt with expanded image tokens for when the image is split into patches."""
    text_split_images = ""
    for n_h in range(image_rows):
        for n_w in range(image_cols):
            text_split_images += (
                f"{fake_token_around_image}"
                + f"<row_{n_h + 1}_col_{n_w + 1}>"
                + f"{image_token}" * image_seq_len
            )
        text_split_images += "\n"

    text_split_images += (
        f"\n{fake_token_around_image}"
        + f"{global_img_token}"
        + f"{image_token}" * image_seq_len
        + f"{fake_token_around_image}"
    )
    return text_split_images


def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    """
    Calculate the shape of the image patch grid after the preprocessing for images of any resolution.

    Args:
        image_size (`tuple`):
            The size of the input image in the format (height, width).
        grid_pinpoints (`List`):
            A list containing possible resolutions. Each item in the list should be a tuple or list
            of the form `(height, width)`.
        patch_size (`int`):
            The size of each image patch.

    Returns:
        tuple: The shape of the image patch grid in the format (width, height).
    """
    if not isinstance(grid_pinpoints, list):
        raise ValueError("grid_pinpoints should be a list of tuples or lists")

    height, width = select_best_resolution(image_size, grid_pinpoints)
    return height // patch_size, width // patch_size


def image_text_replacement(processor, image_input, config) -> str:
    if config.model_type == "idefics2":
        image_seq_len = 64
        image_str = f"{IDEFICS2_FAKE_TOKEN}{IDEFICS2_IMAGE_TOKEN * image_seq_len}{IDEFICS2_FAKE_TOKEN}"
        if processor.image_processor.do_image_splitting:
            image_str *= 5
        return image_str, IDEFICS2_FAKE_TOKEN
    if config.model_type == "idefics3":
        # TODO: implement this in a more general way
        n_rows = image_input["rows"][0][0]
        n_cols = image_input["cols"][0][0]
        image_seq_len = int(
            ((config.vision_config.image_size // config.vision_config.patch_size) ** 2)
            / (config.scale_factor**2)
        )
        image_str = _prompt_split_image(
            image_seq_len=image_seq_len,
            image_rows=n_rows,
            image_cols=n_cols,
            fake_token_around_image=IDEFICS3_FAKE_IMAGE_TOKEN,
            image_token=IDEFICS3_IMAGE_TOKEN,
            global_img_token=IDEFICS3_GLOBAL_IMG_TOKEN,
        )
        return image_str, IDEFICS3_FAKE_IMAGE_TOKEN
    elif config.model_type == "llava_next":
        height, width = image_input["image_sizes"][0]
        num_features = get_number_of_features(height, width, config)

        log_master(
            logger.info,
            f"Found {num_features} features in image of resolution {height}x{width}",
        )
        return "<image>" * num_features, "<image>"

    elif config.model_type == "paligemma":
        return "<image>" * config.text_config.num_image_tokens, "<image>"
    elif config.model_type == "qwen2_vl":
        grid_t, grid_h, grid_w = image_input["image_grid_thw"][0]
        num_pads = grid_t * grid_h * grid_w // 4
        padding = "<|image_pad|>" * num_pads
        return f"<|vision_start|>{padding}<|vision_end|>", "<|vision_start|>"
    elif config.model_type == "qwen2_5_vl":
        grid_t, grid_h, grid_w = image_input["image_grid_thw"][0]
        num_pads = grid_t * grid_h * grid_w // 4
        padding = "<|image_pad|>" * num_pads
        return f"<|vision_start|>{padding}<|vision_end|>", "<|vision_start|>"
    elif config.model_type == "gemma3":
        # TODO: get correct number of features via reviewing the Gemma3 architecture
        # and calculating the number of image tokens
        num_pads = 256
        padding = "<image_soft_token>" * num_pads
        return f"\n\n<start_of_image>{padding}<end_of_image>\n\n", "<start_of_image>"
    elif config.model_type == "llama4":
        patch_size = config.vision_config.patch_size
        pixel_shuffle_ratio = config.vision_config.pixel_shuffle_ratio
        downsample_ratio = int(round(1.0 / (pixel_shuffle_ratio**2)))
        aspect_ratios = image_input["aspect_ratios"][0]
        image_height, image_width = image_input["pixel_values"][0].shape[-2:]

        num_patches_per_chunk = int(
            (image_height // patch_size)
            * (image_width // patch_size)
            // downsample_ratio
        )
        tokens_for_this_image = prompt_split_image_llama4(
            aspect_ratios, num_patches_per_chunk
        )

        return tokens_for_this_image, "<|image_start|>"
    else:
        raise RuntimeError(f"Unknown config {config.model_type} for multimodal")


def image_text_replacement_fixup(config, text: str) -> str:
    if config.model_type == "idefics2":
        return text.replace(
            f"{IDEFICS2_FAKE_TOKEN}{IDEFICS2_FAKE_TOKEN}", IDEFICS2_FAKE_TOKEN
        )
    return text


def preprocess_text(config, text: str) -> str:
    if config.model_type == "paligemma":
        return "<bos>" + text + "\n"
    return text


def preprocess_image(config, img):
    model_type = config.model_type

    if model_type in {"qwen2_vl", "qwen2_5_vl"} and img.width <= 20:
        img = img.resize((img.width * 2, img.height * 2))
    if model_type == "paligemma":
        img = img.convert("RGB")

    if model_type not in {"llava_next", "gemma3", "llama4"}:
        # TODO: check if this is needed
        img = [img]

    return img


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


def scatter_image_embeds(
    embeds: torch.Tensor, is_embed: Optional[torch.Tensor]
) -> torch.Tensor:
    if is_embed is None:
        return embeds

    placeholders = embeds.new_full(
        (is_embed.shape[0], embeds.shape[-1]),
        fill_value=torch.nan,
    )
    placeholders[is_embed.to(embeds.device)] = embeds
    return placeholders


def gather_image_embeds(
    embeds: torch.Tensor, is_embed: Optional[torch.Tensor]
) -> Optional[torch.Tensor]:
    if is_embed is None:
        return embeds
    sel = embeds[is_embed.to(embeds.device)]
    return sel if sel.numel() else None


@dataclass
class ImagePositions:
    offset: int
    length: int
    id: int
    num_placeholder_tokens: int
    is_embed: Optional[torch.Tensor] = None


class FlashVlmCausalLMBatch(FlashCausalLMBatch):
    image_inputs: Optional[List[List[Dict[str, torch.Tensor]]]]
    image_positions: Optional[List[List[ImagePositions]]]
    encoder_cache: Optional[List[Dict[int, torch.Tensor]]]
    pixel_values: Optional[List[torch.Tensor]]
    pixel_attention_mask: Optional[List[torch.Tensor]]
    image_sizes: Optional[List[Tuple[int, int]]]
    image_grid_thw: Optional[torch.Tensor]
    cache_entries_to_free: List[Tuple[int, int]]
    has_image_inputs: bool = False
    inputs_embeds: Optional[torch.Tensor] = None

    @classmethod
    @tracer.start_as_current_span("concatenate")
    def concatenate(cls, batches, padded_total_bs: int = 0):
        batch = super(FlashVlmCausalLMBatch, cls).concatenate(batches, padded_total_bs)
        batch.image_inputs = []
        batch.image_positions = []
        batch.encoder_cache = []
        for b in batches:
            if b.image_inputs is not None:
                batch.image_inputs.extend(b.image_inputs)
            else:
                batch.image_inputs.append(None)
            if b.image_positions is not None:
                batch.image_positions.extend(b.image_positions)
            else:
                batch.image_positions.append(None)
            if b.encoder_cache is not None:
                batch.encoder_cache.extend(b.encoder_cache)
            else:
                batch.encoder_cache.append(None)

        batch.pixel_values = None
        batch.pixel_attention_mask = None
        batch.image_sizes = None
        batch.image_grid_thw = None
        batch.inputs_embeds = None
        # To be filled in prepare_for_prefill
        batch.has_image_inputs = False
        batch.cache_entries_to_free = []
        return batch

    @tracer.start_as_current_span("filter")
    def filter(self, request_ids: List[int]):
        if len(request_ids) == 0:
            raise ValueError("Batch must have at least one request")

        image_inputs = []
        image_positions = []
        encoder_cache = []

        for request_id in request_ids:
            idx = self.requests_idx_mapping[request_id]
            image_inputs.append(self.image_inputs[idx])
            image_positions.append(self.image_positions[idx])
            encoder_cache.append(self.encoder_cache[idx])

        batch = super().filter(request_ids)
        batch.pixel_values = None
        batch.pixel_attention_mask = None
        batch.image_sizes = None
        batch.image_grid_thw = None
        batch.inputs_embeds = None
        batch.image_inputs = image_inputs
        batch.image_positions = image_positions
        batch.encoder_cache = encoder_cache

        # To be filled in prepare_for_prefill
        batch.has_image_inputs = False
        batch.cache_entries_to_free = []
        return batch

    @classmethod
    def batch_tokenized_inputs(
        cls, requests: Iterable[generate_pb2.Request], tokenizer, processor, config
    ):
        kwargs = {}
        if (
            hasattr(processor, "image_processor_class")
            and processor.image_processor_class == "Idefics3ImageProcessor"
        ):
            kwargs["return_row_col_info"] = True

        max_length = 0
        vocab = tokenizer.get_vocab()

        if not hasattr(config, "image_token_index"):
            config.image_token_index = config.image_token_id

        batch_tokenized_inputs: List[List[int]] = []
        batch_image_inputs: List[Optional[List[dict]]] = []
        batch_image_positions: List[Optional[List[ImagePositions]]] = []

        for r in requests:
            text_parts = []
            image_inputs = []
            image_texts = []

            image_id = 0

            for chunk in r.input_chunks.chunks:
                chunk_type = chunk.WhichOneof("chunk")
                if chunk_type == "text":
                    text = preprocess_text(config, chunk.text)
                    text_parts.append(text)
                elif chunk_type == "image":
                    img = Image.open(BytesIO(chunk.image.data))
                    img = preprocess_image(config, img)

                    image_input = processor.image_processor(
                        [img], return_tensors="pt", **kwargs
                    )
                    image_inputs.append(image_input)

                    img_text, img_start_token_str = image_text_replacement(
                        processor, image_input, config
                    )
                    text_parts.append(img_text)

                    image_texts.append([image_id, img_start_token_str, img_text])
                    image_id += 1
                else:
                    raise RuntimeError(f"Invalid chunk type {chunk_type}")

            full_text = image_text_replacement_fixup(config, "".join(text_parts))
            input_ids = tokenizer(
                full_text,
                truncation=True,
                max_length=r.truncate,
                add_special_tokens=(
                    r.add_special_tokens if config.model_type != "paligemma" else False
                ),
            )["input_ids"]
            max_length = max(max_length, len(input_ids))

            if len(image_inputs) > 0:
                img_start_token = vocab[image_texts[0][1]]
                image_positions = cls.get_image_positions(
                    input_ids, image_texts, img_start_token, config, tokenizer
                )
            else:
                image_inputs = None
                image_positions = None

            batch_tokenized_inputs.append(input_ids)
            batch_image_inputs.append(image_inputs)
            batch_image_positions.append(image_positions)

        return batch_tokenized_inputs, batch_image_inputs, batch_image_positions

    @classmethod
    def get_image_positions(
        cls,
        input_ids: List[int],
        image_texts: List[Tuple[int, str, str]],
        img_start_token: int,
        config,
        tokenizer: PreTrainedTokenizerBase,
    ) -> List[ImagePositions]:
        image_positions = []
        num_images = len(image_texts)

        input_ids_t = torch.as_tensor(input_ids)
        img_start_token_pos = torch.where(input_ids_t.eq(img_start_token))[0]
        num_tokens = input_ids_t.numel()

        last_pos = 0
        for i in range(num_images):
            image_id, img_start_token_str, img_text = image_texts[i]
            img_text = image_text_replacement_fixup(config, img_text)

            if config.model_type == "gemma3":
                img_text = img_text.replace("\n\n", "")

            tokens = tokenizer(img_text, add_special_tokens=False, return_tensors="pt")[
                "input_ids"
            ][0]
            length = tokens.numel()

            assert (
                length <= num_tokens
            ), f"{length} > {num_tokens} Image is truncated, try increasing --max-batch-prefill-tokens"

            pos = torch.searchsorted(img_start_token_pos, last_pos, right=False)
            index = img_start_token_pos[pos]
            assert torch.equal(
                input_ids_t[index : index + length], tokens
            ), "Image tokens not found in input_ids"

            is_embed = tokens == config.image_token_index
            num_placeholder_tokens = int(is_embed.sum())
            if num_placeholder_tokens == length:
                is_embed = None

            pos = ImagePositions(
                offset=index,
                length=length,
                id=image_id,
                num_placeholder_tokens=num_placeholder_tokens,
                is_embed=is_embed,
            )

            image_positions.append(pos)
            last_pos = index + length

            if (
                config.model_type == "idefics2"
                and i + 1 != num_images
                and input_ids[last_pos] == config.image_token_index
            ):
                fake_token = last_pos - 1
                fake_token_index = torch.searchsorted(
                    img_start_token_pos, fake_token, right=False
                )
                img_start_token_pos[fake_token_index] = last_pos
                image_texts[i + 1][2] = image_texts[i + 1][2][
                    len(img_start_token_str) :
                ]

        return image_positions

    @classmethod
    def from_pb_processor(
        cls,
        pb: generate_pb2.Batch,
        tokenizer: PreTrainedTokenizerBase,
        processor,
        config,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "FlashVlmCausalLMBatch":
        batch_tokenized_inputs, image_inputs, image_positions = (
            cls.batch_tokenized_inputs(pb.requests, tokenizer, processor, config)
        )
        batch = cls.from_tokenized(pb, tokenizer, batch_tokenized_inputs, dtype, device)
        batch.image_inputs = image_inputs
        batch.image_positions = image_positions
        batch.encoder_cache = [{} for _ in range(len(pb.requests))]
        if len(image_inputs):
            batch.pixel_values = None
            batch.pixel_attention_mask = None
            batch.image_sizes = None
            batch.image_grid_thw = None
        return batch

    def prepare_for_prefill(
        self, max_padded_input_len, max_padded_bs, max_total_tokens
    ):
        super().prepare_for_prefill(
            max_padded_input_len, max_padded_bs, max_total_tokens
        )

        self.has_image_inputs = False
        self.cache_entries_to_free = []

        self.pixel_values = []

        assert (
            len(self.cache_lengths)
            == len(self.input_lengths)
            == len(self.prefilling_mask)
        ), "Mismatch in lengths of cache_lengths, input_lengths, and prefilling_mask"

        for i, (
            cache_length,
            input_length,
            request_prefilling,
        ) in enumerate(
            zip(
                self.cache_lengths,
                self.input_lengths,
                self.prefilling_mask,
            )
        ):
            if not request_prefilling or self.image_positions[i] is None:
                continue

            for image_position in self.image_positions[i]:
                if image_position is None:
                    continue
                start_pos = image_position.offset
                length = image_position.length

                if start_pos >= cache_length + input_length:
                    # No encoder input required at this step
                    break
                if start_pos + length <= cache_length:
                    # The encode input is already processed
                    continue

                self.has_image_inputs = True

                if image_position.id not in self.encoder_cache[i]:
                    image_inputs = self.image_inputs[i][image_position.id]
                    self.pixel_values.append((i, image_position.id, image_inputs))

                    # Remove the image from the image_inputs
                    self.image_inputs[i][image_position.id] = None

        if not self.has_image_inputs:
            self.pixel_values = None
            self.pixel_attention_mask = None
            self.image_sizes = None
            self.image_grid_thw = None
        else:
            image_grid_thw_list = [
                x[2]["image_grid_thw"]
                for x in self.pixel_values
                if "image_grid_thw" in x[2]
            ]
            if image_grid_thw_list:
                self.image_grid_thw = torch.cat(image_grid_thw_list, dim=0)
            else:
                self.image_grid_thw = None

    def update_encoder_cache(self, encoder_outputs, request_id, img_pos):
        self.encoder_cache[request_id][img_pos.id] = scatter_image_embeds(
            encoder_outputs, img_pos.is_embed
        )

    def gather_vision_embeds(self):
        device = self.input_ids.device
        chunks = []
        for (
            i,
            cache_length,
            input_length,
            request_prefilling,
        ) in zip(
            range(len(self.requests)),
            self.cache_lengths,
            self.input_lengths,
            self.prefilling_mask,
        ):
            if not request_prefilling or self.image_positions[i] is None:
                continue

            for image_position in self.image_positions[i]:
                if image_position is None:
                    continue
                start_pos = image_position.offset
                length = image_position.length

                if start_pos >= cache_length + input_length:
                    # No encoder input required at this step
                    break
                if start_pos + length <= cache_length:
                    # The encode input is already processed
                    continue

                start_idx = max(cache_length - start_pos, 0)
                end_idx = min(cache_length - start_pos + input_length, length)

                assert (
                    image_position.id in self.encoder_cache[i]
                ), f"image_id {image_position.id} not in encoder_cache {self.encoder_cache[i]}"
                encoder_output = self.encoder_cache[i][image_position.id]

                is_embed = image_position.is_embed
                if is_embed is not None:
                    is_embed = is_embed[start_idx:end_idx]

                from loguru import logger

                logger.info(
                    f"image_id {image_position.id} start_idx {start_idx} end_idx {end_idx}, length {length}"
                )

                embeds = gather_image_embeds(
                    encoder_output[start_idx:end_idx],
                    is_embed=is_embed,
                )
                if embeds is not None:
                    chunks.append(embeds)

                if end_idx == length:
                    self.cache_entries_to_free.append((i, image_position.id))
                    self.image_positions[i][image_position.id] = None

        if len(chunks) == 0:
            return None
        return torch.cat(chunks, dim=0).to(device)

    def free_encoder_cache(self):
        for i, image_id in self.cache_entries_to_free:
            self.encoder_cache[i].pop(image_id, None)

        self.cache_entries_to_free = []


class FlashVlmCausalLM(FlashCausalLM):
    def __init__(
        self,
        model_id: str,
        *,
        processor_class=AutoProcessor,
        processor_kwargs=None,
        batch_class=FlashVlmCausalLMBatch,
        revision,
        trust_remote_code: bool,
        support_chunking: bool = False,
        **kwargs,
    ):
        if PREFIX_CACHING:
            raise NotImplementedError("Vlm do not work with prefix caching yet")
        if processor_kwargs is None:
            processor_kwargs = {}
        self.processor = processor_class.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=trust_remote_code,
            **processor_kwargs,
        )
        self.batch_class = batch_class
        super().__init__(
            model_id=model_id,
            revision=revision,
            trust_remote_code=trust_remote_code,
            support_chunking=support_chunking,
            **kwargs,
        )

    @property
    def batch_type(self) -> Type[FlashVlmCausalLMBatch]:
        return self.batch_class

    def max_past(self) -> Optional[int]:
        return getattr(self.model.text_model, "max_past", None)

    def warmup_decode(
        self, batch_size: int, block_num: int, batch: FlashVlmCausalLMBatch
    ):
        input_ids = torch.zeros(batch_size, dtype=batch.input_ids.dtype)
        position_ids = torch.arange(batch_size, dtype=batch.position_ids.dtype)
        if batch.position_ids is not None and batch.position_ids.dim() == 2:
            # qwen2_vl and qwen2_5_vl case
            position_ids = position_ids.unsqueeze(-1).repeat(
                (1, batch.position_ids.shape[-1])
            )
        blocks = [block_num // batch_size for _ in range(batch_size)]
        blocks[0] += block_num % batch_size
        past_len = []
        block_tables = []
        slots = []
        start_idx = 0

        # fetch the last blocked to warmup block num
        for i in range(batch_size):
            block_array = list(range(start_idx, start_idx + blocks[i]))
            slots.append(BLOCK_SIZE * block_array[-1] + BLOCK_SIZE - 1)
            block_tables.append(block_array)
            past_len.append(blocks[i] * BLOCK_SIZE - 1)
            start_idx += blocks[i]
        input_lengths = torch.ones(batch_size, dtype=torch.int32)

        seqlen = Seqlen(
            input_lengths=_async_h2d_tensor_copy(input_lengths),
        )

        hpu_attention_meta = prepare_for_decode(
            self.dtype,
            self.use_contiguous_pa,
            self.device,
            slots,
            block_tables,
            batch_size,
            bucketing_ctx=None,
        )
        slots_tensor = torch.tensor(slots, dtype=batch.slots.dtype)
        inputs_embeds = self.get_inputs_embeds(
            input_ids=input_ids.to(self.device),
        )
        # We pass a `cu_seqlen_prefill` in order not to have to deal with paged attention cache allocation/deallocation.
        self.model.forward(
            inputs_embeds=inputs_embeds,
            position_ids=_async_h2d_tensor_copy(position_ids),
            cu_seqlen_prefill=None,
            kv_cache=self.kv_cache,
            slots=_async_h2d_tensor_copy(slots_tensor),
            seqlen=trim_seqlen_metadata(seqlen),
            hpu_attention_meta=hpu_attention_meta,
            lm_head_indices=None,
            attention_mask=None,
        )

    def warmup_hpu_graph(self, batch: FlashVlmCausalLMBatch):
        free_mem = HabanaMemoryProfiler.current_free_device_memory()
        graph_free_mem = free_mem - self.mem_reserved
        graph_free_mem = self.align_workers(
            graph_free_mem, torch.distributed.ReduceOp.MIN
        )
        decode_available_memory = graph_free_mem
        msg = (
            f"Using {format_bytes(graph_free_mem)}"
            f"/{format_bytes(free_mem)} "
            "of free device memory for HPUGraphs, "
            f"{format_bytes(decode_available_memory)} for decode "
        )
        log_master(logger.info, msg)
        start_time = time.time()
        warmup_shape_count = 0
        warmup_times = 3

        # only warmup decode, for prefill, image pixal size may change, make the warmup useless
        def ordering_function_max_bs(b):
            return (-b[0], b[1])

        self.bucketing_ctx.generate_decode_buckets(self.bucketing_ctx.num_hpu_blocks)
        buckets = list(
            sorted(self.bucketing_ctx.decode_buckets, key=ordering_function_max_bs)
        )
        total_batch_seq = 0.001
        total_mem = 0
        available_mem = decode_available_memory
        log_master(logger.info, f"Decode batch size list:{[bsz[0] for bsz in buckets]}\n")
        for i, (batch_size, block_num) in enumerate(buckets):
            if batch_size > block_num:
                continue
            # Graph memory usage is proportional to seq dimension in a batch
            batch_seq = batch_size
            mem_estimate = batch_seq / total_batch_seq * total_mem
            graphed_bucket = (batch_size, block_num, False)
            if not mem_estimate >= available_mem:
                if graphed_bucket not in self.graphed_buckets:
                    self.graphed_buckets.add(graphed_bucket)
            warmup_shape_count += 1
            self.log_warmup(False, i, len(buckets), batch_size, block_num)
            with HabanaMemoryProfiler() as mem_prof:
                for index in range(warmup_times):
                    self.warmup_decode(batch_size, block_num, batch)
                    synchronize(self.device)
            used_mem = self.align_workers(
                mem_prof.consumed_device_memory, torch.distributed.ReduceOp.MAX
            )
            if graphed_bucket in self.graphed_buckets:

                available_mem -= used_mem
                total_mem += used_mem
                total_batch_seq += batch_seq

        log_master(logger.info, "Decode warmup successful.\n")

        log_master(
            logger.info,
            f"warmup hpu graph time {int(time.time() - start_time)}s warmup shape count {warmup_shape_count}",
        )

    def get_vision_embeds(
        self,
        pixel_values: torch.Tensor,
        pixel_attention_mask: torch.Tensor,
        image_sizes: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ):
        embeds = self.model.get_vision_embeds(
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            image_sizes=image_sizes,
            image_grid_thw=image_grid_thw,
        )
        return embeds

    def get_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        vision_embeds: Optional[torch.Tensor] = None,
    ):
        return self.model.get_inputs_embeds(
            input_ids=input_ids,
            vision_embeds=vision_embeds,
        )

    def encode_images(self, batch):
        if batch.pixel_values is not None:
            device = batch.input_ids.device
            for request_id, image_id, image_input in batch.pixel_values:
                pixel_values = image_input["pixel_values"].to(device)

                if "pixel_attention_mask" in image_input:
                    pixel_attention_mask = image_input["pixel_attention_mask"].to(
                        device
                    )
                else:
                    pixel_attention_mask = None

                if "image_sizes" in image_input:
                    image_sizes = image_input["image_sizes"].to(device)
                else:
                    image_sizes = None

                if "image_grid_thw" in image_input:
                    image_grid_thw = image_input["image_grid_thw"]
                else:
                    image_grid_thw = None

                encoder_outputs = self.get_vision_embeds(
                    pixel_values=pixel_values,
                    pixel_attention_mask=pixel_attention_mask,
                    image_sizes=image_sizes,
                    image_grid_thw=image_grid_thw,
                )
                batch.update_encoder_cache(
                    encoder_outputs,
                    request_id,
                    batch.image_positions[request_id][image_id],
                )

        batch.pixel_values = None
        batch.pixel_attention_mask = None
        batch.image_sizes = None

    def set_inputs_embeds(self, batch):
        if batch.has_image_inputs:
            self.encode_images(batch)
            vision_embeds = batch.gather_vision_embeds()
            batch.has_image_inputs = False
        else:
            vision_embeds = None

        inputs_embeds = self.get_inputs_embeds(
            batch.input_ids, vision_embeds=vision_embeds
        )

        batch.inputs_embeds = inputs_embeds

    def forward(
        self,
        batch: FlashVlmCausalLMBatch,
        adapter_data: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Model Forward
        if batch.speculative_ids is not None:
            input_ids = batch.input_ids
            position_ids = batch.position_ids
            cu_seqlen_prefill = batch.cu_seqlen_prefill
            kv_cache = self.kv_cache
            block_tables = batch.block_tables_tensor
            slots = batch.slots[batch.slot_indices]
            input_lengths = batch.input_lengths_tensor
            max_s = batch.max_current_length
            lm_head_indices = batch.prefill_head_indices

            speculative_ids = batch.speculative_ids

            B, speculative_length = speculative_ids.shape
            new_length = speculative_length + 1
            new_input_ids = torch.cat(
                [input_ids.unsqueeze(-1), speculative_ids], dim=1
            ).reshape(-1)
            arange = torch.arange(new_length, device=position_ids.device).unsqueeze(0)
            arange_int = arange.to(dtype=torch.int32)
            new_position_ids = (
                position_ids.unsqueeze(-1).expand(B, new_length) + arange
            ).view(-1)
            slots = (slots.unsqueeze(-1).expand(B, new_length) + arange_int).view(-1)
            input_lengths = (
                input_lengths.unsqueeze(-1).expand(B, new_length) + arange_int
            ).view(-1)

            # Add Copy the block tables for all members
            block_tables = (
                block_tables.unsqueeze(1)
                .expand(B, new_length, -1)
                .reshape(B * new_length, -1)
                .contiguous()
            )
            max_s = max_s + speculative_length

            input_ids = new_input_ids
            position_ids = new_position_ids
        else:
            input_ids = batch.input_ids
            inputs_embeds = batch.inputs_embeds
            position_ids = batch.position_ids
            cu_seqlen_prefill = batch.cu_seqlen_prefill
            kv_cache = self.kv_cache
            block_tables = batch.block_tables_tensor
            slots = batch.slots[batch.slot_indices]
            input_lengths = batch.input_lengths_tensor
            max_s = batch.max_current_length
            lm_head_indices = batch.prefill_head_indices

        if self.model.config.model_type in {"qwen2_vl", "qwen2_5_vl"}:
            if position_ids.dim() == 1 and batch.prefilling:
                position_ids = self.model.get_position_ids(
                    input_ids.cpu(), batch.image_grid_thw
                )
                batch.position_ids = position_ids

        attention_mask = None
        attention_mask_forward = None
        if self.model.config.model_type == "gemma3" and cu_seqlen_prefill is not None:
            attention_mask = self.model.get_attention_mask(
                input_ids, cu_seqlen_prefill, self.dtype, bool_mask=True
            )
            min_dtype = torch.finfo(self.dtype).min
            attention_mask_forward = torch.where(attention_mask, 0, min_dtype).to(
                input_ids.device
            )
            attention_mask = attention_mask.reshape(-1)
        if self.model.config.model_type == "llama4":
            attention_mask = (input_ids != 0).long()
            attention_mask_forward = attention_mask.view(input_lengths.shape[0], -1)

        if cu_seqlen_prefill is None and self.max_past() is not None:
            # In decode, not prefill, we're actually overwriting the KV-cache
            # in a circular buffer mode.
            # This makes sure the max_s for the decode pass is correct.
            max_s = min(self.max_past(), max_s)

        kwargs = {}
        if htorch.utils.internal.is_lazy():
            batch_size = input_lengths.shape[0]
            seqlen = (
                input_ids.shape[0] // batch_size
                if batch.prefilling
                else batch.hpu_attn_meta.block_list.shape[0]
            )
            kwargs["bypass_hpu_graphs"] = not self.use_graphs(
                batch.prefilling, seqlen, batch_size
            )
        if batch.prefill_cache_indices is not None:
            slots_pad = torch.zeros_like(input_ids, device=slots.device)
            slots_pad[batch.prefill_cache_indices] = slots
            slots = slots_pad
        else:
            slots_pad = torch.zeros_like(input_ids, device=slots.device)
            slots_pad[: slots.shape[0]] = slots
            slots = slots_pad

        seqlen = Seqlen(
            input_lengths=_async_h2d_tensor_copy(input_lengths),
        )
        logits, speculative_logits = self.model.forward(
            inputs_embeds=inputs_embeds,
            position_ids=_async_h2d_tensor_copy(position_ids),
            cu_seqlen_prefill=_async_h2d_tensor_copy(cu_seqlen_prefill),
            kv_cache=kv_cache,
            slots=_async_h2d_tensor_copy(slots),
            seqlen=trim_seqlen_metadata(seqlen),
            hpu_attention_meta=batch.hpu_attn_meta,
            lm_head_indices=_async_h2d_tensor_copy(lm_head_indices),
            attention_mask=attention_mask_forward,
            **kwargs,
        )
        batch.image_grid_thw = None
        batch.free_encoder_cache()
        return logits, speculative_logits
