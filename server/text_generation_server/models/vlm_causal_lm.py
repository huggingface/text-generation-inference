from dataclasses import dataclass
import torch
from PIL import Image
from io import BytesIO

from opentelemetry import trace
from typing import Iterable, Optional, Tuple, List, Type, Dict

from transformers import PreTrainedTokenizerBase
from transformers.image_processing_utils import select_best_resolution
from text_generation_server.pb import generate_pb2
from text_generation_server.models.flash_causal_lm import (
    FlashCausalLMBatch,
    FlashCausalLM,
)
from text_generation_server.models.globals import PREFIX_CACHING, ATTENTION, MEM_POOL
from loguru import logger
from text_generation_server.utils.log import log_master
from transformers import AutoProcessor
from text_generation_server.layers.attention import Seqlen
from text_generation_server.models.metadata_kernels import block_tables_to_ragged

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
    elif model_type == "paligemma":
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
    placeholders[is_embed] = embeds
    return placeholders


def gather_image_embeds(
    embeds: torch.Tensor, is_embed: Optional[torch.Tensor]
) -> Optional[torch.Tensor]:
    if is_embed is None:
        return embeds
    sel = embeds[is_embed]
    return sel if sel.numel() else None


@dataclass
class ImagePositions:
    offset: int
    length: int
    id: int
    num_placeholder_tokens: int
    is_embed: Optional[torch.Tensor] = None


class VlmCausalLMBatch(FlashCausalLMBatch):
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
    def concatenate(cls, batches):
        batch = super(VlmCausalLMBatch, cls).concatenate(batches)

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

        config.image_token_index = getattr(
            config, "image_token_index", config.image_token_id
        )

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
                add_special_tokens=r.add_special_tokens,
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
    ) -> "VlmCausalLMBatch":
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

    def prepare_for_prefill(self):
        super().prepare_for_prefill()

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

        # release any freed GPU memory immediately?


class VlmCausalLM(FlashCausalLM):
    def __init__(
        self,
        model_id: str,
        *,
        processor_class=AutoProcessor,
        processor_kwargs=None,
        batch_class=VlmCausalLMBatch,
        revision,
        trust_remote_code: bool,
        support_chunking: bool = True,
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
    def batch_type(self) -> Type[VlmCausalLMBatch]:
        return self.batch_class

    def cuda_graph_warmup(self, bs: int, max_s: int, max_bt: int):
        max_bs = max(self.cuda_graphs.keys()) if self.cuda_graphs else None
        input_lengths = [max_s] * bs
        cache_lengths = [0] * bs
        if max_bs is None:
            inputs_embeds = torch.zeros(
                (bs, self.model.config.text_config.hidden_size),
                device=self.device,
                dtype=self.dtype,
            )
            position_ids = torch.zeros(bs, dtype=torch.int32, device=self.device)
            config = getattr(self.model, "config", None)
            rope_scaling = getattr(config, "rope_scaling", None) if config else None
            if (  # mrope have position_ids per section, if so repeat n times
                isinstance(rope_scaling, dict) and rope_scaling["rope_type"] == "mrope"
            ):
                n_sections = len(self.model.config.rope_scaling["mrope_section"])
                position_ids = position_ids.unsqueeze(1).repeat(1, n_sections)
            slots = torch.arange(bs, dtype=torch.int64, device=self.device)
            input_lengths_tensor = (
                torch.ones(bs, dtype=torch.int32, device=self.device) * max_s
            )
            cache_lengths_tensor = torch.zeros(
                bs, dtype=torch.int32, device=self.device
            )
            block_tables = torch.arange(
                max_bt, dtype=torch.int32, device=self.device
            ).repeat(bs)
            block_tables = block_tables.reshape((bs, max_bt))
            if ATTENTION == "flashinfer":
                block_tables = block_tables_to_ragged(
                    block_tables=block_tables,
                    input_lengths=input_lengths,
                    cache_lengths=cache_lengths,
                    input_lengths_tensor=input_lengths_tensor,
                    cache_lengths_tensor=cache_lengths_tensor,
                    max_current_length=max_s,
                )
        else:
            if bs > max_bs:
                raise RuntimeError(
                    "Cuda graphs should be generated in decreasing order size to reduce VRAM usage"
                )
            inputs_embeds = self.cuda_graphs[max_bs]["inputs_embeds"][:bs]
            position_ids = self.cuda_graphs[max_bs]["position_ids"][:bs]
            if ATTENTION == "flashinfer":
                block_tables = self.cuda_graphs[max_bs]["block_tables"][: bs * max_bt]
            else:
                block_tables = self.cuda_graphs[max_bs]["block_tables"][:bs]
            slots = self.cuda_graphs[max_bs]["slots"][:bs]
            input_lengths_tensor = self.cuda_graphs[max_bs]["input_lengths"][:bs]
            cache_lengths_tensor = self.cuda_graphs[max_bs]["cache_lengths"][:bs]

        if ATTENTION == "flashinfer":
            from text_generation_server.layers.attention.flashinfer import (
                create_decode_state_cuda_graphs,
            )

            block_tables_ptr = torch.zeros(
                bs + 1, dtype=torch.int32, device=self.device
            )
            last_page_len = torch.ones(bs, dtype=torch.int32, device=self.device)
            state = create_decode_state_cuda_graphs(
                device=inputs_embeds.device,
                block_tables=block_tables,
                block_tables_ptr=block_tables_ptr,
                last_page_len=last_page_len,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
            )
        else:
            state = None

        graph = torch.cuda.CUDAGraph()
        self.cuda_graphs[bs] = {
            "inputs_embeds": inputs_embeds,
            "position_ids": position_ids,
            "kv_cache": self.kv_cache,
            "block_tables": block_tables,
            "slots": slots,
            "input_lengths": input_lengths_tensor,
            "cache_lengths": cache_lengths_tensor,
            "state": state,
            "graph": graph,
        }

        torch.cuda.synchronize()
        # Run once outside to warmup
        with self._forward_context(
            block_tables=block_tables,
            cu_seqlen_prefill=None,
            input_lengths_tensor=input_lengths_tensor,
            state=state,
            cache_lengths_tensor=cache_lengths_tensor,
        ):
            seqlen = Seqlen(
                input_lengths=input_lengths_tensor,
                cache_lengths=cache_lengths_tensor,
                cu_seqlen_q=None,
                max_q=1,
                max_k=max_s,
            )
            self.model.forward(
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                cu_seqlen_prefill=None,
                kv_cache=self.kv_cache,
                block_tables=block_tables,
                slots=slots,
                seqlen=seqlen,
                max_s=max_s,
                prefill_cache_indices=None,
                lm_head_indices=None,
            )
            del seqlen

            torch.cuda.synchronize()

            with torch.cuda.graph(graph, pool=MEM_POOL):
                seqlen = Seqlen(
                    input_lengths=input_lengths_tensor,
                    cache_lengths=cache_lengths_tensor,
                    cu_seqlen_q=None,
                    max_q=1,
                    max_k=max_s,
                )
                logits, speculative_logits = self.model.forward(
                    inputs_embeds=inputs_embeds,
                    position_ids=position_ids,
                    cu_seqlen_prefill=None,
                    kv_cache=self.kv_cache,
                    block_tables=block_tables,
                    slots=slots,
                    seqlen=seqlen,
                    max_s=max_s,
                    prefill_cache_indices=None,
                    lm_head_indices=None,
                )
                self.cuda_graphs[bs]["logits"] = logits
                self.cuda_graphs[bs]["speculative_logits"] = speculative_logits
        torch.cuda.synchronize()

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
                    image_grid_thw = image_input["image_grid_thw"].to(device)
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
        batch: VlmCausalLMBatch,
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
            cache_lengths_tensor = (
                batch.cache_lengths_tensor.unsqueeze(-1).expand(B, new_length)
            ).reshape(-1)

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
            cache_lengths_tensor = batch.cache_lengths_tensor
            max_s = batch.max_current_length
            lm_head_indices = batch.prefill_head_indices

        if self.model.config.model_type in {"qwen2_vl", "qwen2_5_vl"}:
            if position_ids.dim() == 1 and batch.prefilling:
                position_ids = self.model.get_position_ids(
                    input_ids, batch.image_grid_thw
                )
                batch.position_ids = position_ids

        attention_mask = None
        attention_mask_forward = None
        if self.model.config.model_type == "gemma3" and cu_seqlen_prefill is not None:
            # Get the mask, needed for flashinfer.
            has_image = (input_ids == self.model.config.image_token_index).any()

            if has_image:
                attention_mask = self.model.get_attention_mask(
                    input_ids, cu_seqlen_prefill, self.dtype, bool_mask=True
                )
                min_dtype = torch.finfo(self.dtype).min
                attention_mask_forward = torch.where(attention_mask, 0, min_dtype).to(
                    input_ids.device
                )
                attention_mask = attention_mask.reshape(-1)

        # Try to find an associated cuda graph
        bs = input_ids.shape[0]
        sorted_padded_bs = sorted([k for k in self.cuda_graphs.keys() if k >= bs])
        if sorted_padded_bs:
            # Get associated cuda graph
            cuda_graph = self.cuda_graphs[sorted_padded_bs[0]]
        else:
            cuda_graph = None
        if cu_seqlen_prefill is not None or cuda_graph is None:
            if ATTENTION == "flashinfer":
                block_tables = block_tables_to_ragged(
                    block_tables=block_tables,
                    input_lengths=batch.input_lengths,
                    cache_lengths=batch.cache_lengths,
                    input_lengths_tensor=batch.input_lengths_tensor,
                    cache_lengths_tensor=batch.cache_lengths_tensor,
                    max_current_length=batch.max_current_length,
                )
            with self._forward_context(
                block_tables=block_tables,
                cu_seqlen_prefill=cu_seqlen_prefill,
                input_lengths_tensor=input_lengths,
                cache_lengths_tensor=cache_lengths_tensor,
                attention_mask=attention_mask,
            ):
                seqlen = Seqlen(
                    input_lengths=input_lengths,
                    cache_lengths=cache_lengths_tensor,
                    cu_seqlen_q=cu_seqlen_prefill,
                    max_q=batch.max_input_length,
                    max_k=batch.max_current_length,
                )
                logits, speculative_logits = self.model.forward(
                    inputs_embeds=inputs_embeds,
                    position_ids=position_ids,
                    cu_seqlen_prefill=cu_seqlen_prefill,
                    kv_cache=kv_cache,
                    block_tables=block_tables,
                    slots=slots,
                    seqlen=seqlen,
                    max_s=max_s,
                    prefill_cache_indices=batch.prefill_cache_indices,
                    lm_head_indices=lm_head_indices,
                    attention_mask=attention_mask_forward,
                )
                if batch.prefill_cache_indices is not None:
                    batch.prefill_cache_indices = None
                if batch.pixel_values is not None:
                    batch.pixel_values = None
                batch.free_encoder_cache()
                return logits, speculative_logits

        # Copy inputs to the static inputs of the cuda graph
        # Static inputs are potentially padded
        cuda_graph["inputs_embeds"][: inputs_embeds.shape[0]] = inputs_embeds
        cuda_graph["position_ids"][: position_ids.shape[0]] = position_ids
        if ATTENTION == "flashinfer":
            block_tables = block_tables_to_ragged(
                block_tables=block_tables,
                input_lengths=batch.input_lengths,
                cache_lengths=batch.cache_lengths,
                input_lengths_tensor=batch.input_lengths_tensor,
                cache_lengths_tensor=batch.cache_lengths_tensor,
                max_current_length=batch.max_current_length,
            )
            cuda_graph["block_tables"][: block_tables.shape[0]] = block_tables
        else:
            cuda_graph["block_tables"][
                : block_tables.shape[0], : block_tables.shape[1]
            ] = block_tables

        # XXX: This is working only because block 0 is reserved for the healthcheck
        # so it doesn't matter if we override it with bogus values.
        cuda_graph["slots"].fill_(0)
        cuda_graph["slots"][: slots.shape[0]] = slots
        cuda_graph["input_lengths"].zero_()
        cuda_graph["input_lengths"][: input_lengths.shape[0]] = input_lengths
        cuda_graph["cache_lengths"].zero_()
        cuda_graph["cache_lengths"][
            : cache_lengths_tensor.shape[0]
        ] = cache_lengths_tensor

        with self._forward_context(
            block_tables=cuda_graph["block_tables"],
            cu_seqlen_prefill=None,
            input_lengths_tensor=cuda_graph["input_lengths"],
            cache_lengths_tensor=cuda_graph["cache_lengths"],
            state=cuda_graph["state"],
        ):
            # Replay the graph
            cuda_graph["graph"].replay()

        # Slice output to the correct shape
        speculative_logits = (
            cuda_graph["speculative_logits"][:bs]
            if cuda_graph["speculative_logits"] is not None
            else None
        )
        logits = cuda_graph["logits"][:bs]

        batch.free_encoder_cache()
        return logits, speculative_logits
