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
from text_generation_server.models.globals import PREFIX_CACHING, ATTENTION
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


def image_text_replacement(processor, image_input, config, image_id: int) -> str:
    if config.model_type == "idefics2":
        image_seq_len = 64
        image_str = f"{IDEFICS2_FAKE_TOKEN}{IDEFICS2_IMAGE_TOKEN * image_seq_len}{IDEFICS2_FAKE_TOKEN}"
        if processor.image_processor.do_image_splitting:
            image_str *= 5
        return image_str, IDEFICS2_FAKE_TOKEN
    if config.model_type == "idefics3":
        # TODO: implement this in a more general way
        n_rows = image_input["rows"][0][image_id]
        n_cols = image_input["cols"][0][image_id]
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
        height, width = image_input["image_sizes"][image_id]
        num_features = get_number_of_features(height, width, config)

        log_master(
            logger.info,
            f"Found {num_features} features in image of resolution {height}x{width}",
        )
        return "<image>" * num_features, "<image>"

    elif config.model_type == "paligemma":
        return "<image>" * config.text_config.num_image_tokens, "<image>"
    elif config.model_type == "qwen2_vl":
        grid_t, grid_h, grid_w = image_input["image_grid_thw"][image_id]
        num_pads = grid_t * grid_h * grid_w // 4
        padding = "<|image_pad|>" * num_pads
        return f"<|vision_start|>{padding}<|vision_end|>", "<|vision_start|>"
    elif config.model_type == "qwen2_5_vl":
        grid_t, grid_h, grid_w = image_input["image_grid_thw"][image_id]
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
        aspect_ratios = image_input[image_id]["aspect_ratios"][0]
        image_height, image_width = image_input[image_id]["pixel_values"][0].shape[-2:]

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


def scatter_image_embeds(embeds, is_embed):
    if is_embed is None:
        return embeds

    placeholders = embeds.new_full(
        (is_embed.shape[0], embeds.shape[-1]),
        fill_value=torch.nan,
    )
    placeholders[is_embed] = embeds
    return placeholders


def gather_image_embeds(embeds, is_embed):
    if is_embed is None:
        return embeds

    gathered = embeds[is_embed]

    if len(gathered) == 0:
        return None
    return gathered


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

    @classmethod
    @tracer.start_as_current_span("concatenate")
    def concatenate(cls, batches):
        batch = super(VlmCausalLMBatch, cls).concatenate(batches)
        batch.pixel_values = None
        batch.pixel_attention_mask = None
        batch.image_sizes = None
        batch.image_grid_thw = None
        return batch

    @tracer.start_as_current_span("filter")
    def filter(self, request_ids: List[int]):
        batch = super().filter(request_ids)
        batch.pixel_values = None
        batch.pixel_attention_mask = None
        batch.image_sizes = None
        batch.image_grid_thw = None
        return batch

    @classmethod
    def batch_tokenized_inputs(
        cls, requests: Iterable[generate_pb2.Request], tokenizer, processor, config
    ):
        # Process images first. We need all of them so that the processor
        # can make the image splits the same size. And we need the final
        # sizes to insert correct number of image tokens.
        kwargs = {}
        if (
            hasattr(processor, "image_processor_class")
            and processor.image_processor_class == "Idefics3ImageProcessor"
        ):
            kwargs["return_row_col_info"] = True

        batch_image_inputs = []
        for i, r in enumerate(requests):
            image_inputs = []
            batch_image_inputs.append(None)
            for chunk in r.input_chunks.chunks:
                chunk_type = chunk.WhichOneof("chunk")
                if chunk_type == "text":
                    pass
                elif chunk_type == "image":
                    image = Image.open(BytesIO(chunk.image.data))
                    # qwen2_vl expects images to be greater than 20 pixels, this is for warmup since the
                    # default warmup image is 20x20
                    if config.model_type in {"qwen2_vl", "qwen2_5_vl"}:
                        if image.width <= 20:
                            w = image.width * 2
                            h = image.height * 2
                            image = image.resize((w, h))

                    if config.model_type in {"llava_next", "gemma3", "llama4"}:
                        image = image
                    else:
                        image = [image]

                    pixel_values = processor.image_processor(
                        [image], return_tensors="pt", **kwargs
                    )

                    image_inputs.append(pixel_values)
                else:
                    raise RuntimeError(f"Invalid chunk type {chunk_type}")

            if len(image_inputs) > 0:
                batch_image_inputs[i] = image_inputs

        batch_image_positions = []
        batch_tokenized_inputs = []
        max_length = 0
        for i, r in enumerate(requests):
            full_text = ""
            image_tokens = []
            batch_image_positions.append(None)
            image_id = 0
            for chunk in r.input_chunks.chunks:
                chunk_type = chunk.WhichOneof("chunk")
                if chunk_type == "text":
                    full_text += chunk.text
                elif chunk_type == "image":
                    image_text, id_token = image_text_replacement(
                        processor, batch_image_inputs[i], config, image_id
                    )
                    full_text += image_text

                    if config.model_type == "gemma3":
                        image_text = image_text.replace("\n\n", "")

                    image_tokens.append(
                        (
                            image_id,
                            id_token,
                            tokenizer(image_text, add_special_tokens=False)[
                                "input_ids"
                            ],
                        )
                    )
                    image_id += 1

            full_text = image_text_replacement_fixup(config, full_text)
            input_ids = tokenizer(
                full_text,
                truncation=True,
                max_length=r.truncate,
                add_special_tokens=r.add_special_tokens,
            )["input_ids"]
            max_length = max(max_length, len(input_ids))
            batch_tokenized_inputs.append(input_ids)

            prev = 0
            image_positions = []
            for image_id, id_token, tokens in image_tokens:
                id_token = tokenizer.get_vocab()[id_token]

                length = len(tokens)
                index = input_ids[prev:].index(id_token)

                index = index + prev
                assert (
                    input_ids[index : index + length] == tokens
                ), "Image token not found in input_ids"

                if config.model_type in {"llava_next", "paligemma"}:
                    is_embed = None
                    num_placeholder_tokens = length
                else:
                    is_embed = torch.tensor(tokens) == config.image_token_index
                    num_placeholder_tokens = is_embed.sum().item()

                pos = ImagePositions(
                    offset=index,
                    length=length,
                    id=image_id,
                    num_placeholder_tokens=num_placeholder_tokens,
                    is_embed=is_embed,
                )

                image_positions.append(pos)
                prev = index + length

            if len(image_positions) > 0:
                batch_image_positions[i] = image_positions

        return batch_tokenized_inputs, batch_image_inputs, batch_image_positions

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

        self.has_image = False
        self.scheduled_image_input = []
        scheduled_image_pixel_values = []

        device = self.input_ids.device

        for i, (
            r,
            cache_length,
            input_length,
            request_prefilling,
        ) in enumerate(
            zip(
                self.requests,
                self.cache_lengths,
                self.input_lengths,
                self.prefilling_mask,
            )
        ):
            if not request_prefilling or self.image_positions[i] is None:
                continue

            for j, image_position in enumerate(self.image_positions[i]):
                image_id = image_position.id
                pixel_values = self.image_inputs[i][j]

                start_pos = image_position.offset
                length = image_position.length

                if start_pos >= cache_length + input_length:
                    # No encoder input required at this step
                    break
                if start_pos + length <= cache_length:
                    # The encode input is already processed
                    continue

                self.has_image = True

                if image_id not in self.encoder_cache[i]:
                    self.scheduled_image_input.append((i, image_position))
                    scheduled_image_pixel_values.append(pixel_values)

        if self.has_image and len(scheduled_image_pixel_values):
            self.pixel_values = torch.cat(
                [d["pixel_values"] for d in scheduled_image_pixel_values], dim=0
            ).to(device)

            if "pixel_attention_mask" in scheduled_image_pixel_values[0]:
                self.pixel_attention_mask = torch.cat(
                    [d["pixel_attention_mask"] for d in scheduled_image_pixel_values],
                    dim=0,
                ).to(device)

            if "image_sizes" in scheduled_image_pixel_values[0]:
                self.image_sizes = torch.cat(
                    [d["image_sizes"] for d in scheduled_image_pixel_values], dim=0
                ).to(device)

            if "image_grid_thw" in scheduled_image_pixel_values[0]:
                self.image_grid_thw = torch.cat(
                    [d["image_grid_thw"] for d in scheduled_image_pixel_values], dim=0
                ).to(device)
        else:
            self.pixel_values = None
            self.pixel_attention_mask = None
            self.image_sizes = None
            self.image_grid_thw = None

    def update_encoder_cache(self, encoder_outputs):
        prev = 0
        for i, input in self.scheduled_image_input:
            length = input.num_placeholder_tokens
            self.encoder_cache[i][input.id] = scatter_image_embeds(
                encoder_outputs[prev : prev + length], input.is_embed
            )

            prev = prev + length

    def get_mm_embeddings(self):
        device = self.input_ids.device
        mm_embeds = []
        for i, (
            r,
            cache_length,
            input_length,
            prompt_length,
            request_prefilling,
        ) in enumerate(
            zip(
                self.requests,
                self.cache_lengths,
                self.input_lengths,
                self.prompt_lengths,
                self.prefilling_mask,
            )
        ):
            if not request_prefilling or self.image_positions[i] is None:
                continue

            for j, image_position in enumerate(self.image_positions[i]):
                image_id = image_position.id

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
                    image_id in self.encoder_cache[i]
                ), f"image_id {image_id} not in encoder_cache {self.encoder_cache[i]}"
                encoder_output = self.encoder_cache[i][image_id]

                is_embed = image_position.is_embed
                if is_embed is not None:
                    is_embed = is_embed[start_idx:end_idx]

                mm_embeds_item = gather_image_embeds(
                    encoder_output[start_idx:end_idx],
                    is_embed=is_embed,
                )
                mm_embeds.append(mm_embeds_item)

        return torch.cat(mm_embeds, dim=0).to(device)


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
            # FIXME: VLM do not work with context chunking yet
            support_chunking=False,
            **kwargs,
        )

    @property
    def batch_type(self) -> Type[VlmCausalLMBatch]:
        return self.batch_class

    def get_mm_embeddings(self, batch):
        if batch.pixel_values is not None:
            encoder_outputs = self.get_vision_embeds(batch.pixel_values)
            batch.update_encoder_cache(encoder_outputs)
            batch.pixel_values = None
        return batch.get_mm_embeddings()

    def get_input_embeddings(self, batch):
        if batch.has_image:
            vision_embeddings = self.get_mm_embeddings(batch)
            batch.has_image = False
        else:
            vision_embeddings = None

        inputs_embeds = self.get_input_embeds(
            batch.input_ids, vision_embeddings=vision_embeddings
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

        if self.model.config.model_type == "gemma3" and cu_seqlen_prefill is not None:
            # Get the mask, needed for flashinfer.
            attention_mask = self.model.get_attention_mask(
                input_ids, cu_seqlen_prefill, self.dtype, bool_mask=True
            ).reshape(-1)
        else:
            attention_mask = None

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
                    input_ids=input_ids,
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
                    pixel_values=batch.pixel_values,
                    pixel_attention_mask=batch.pixel_attention_mask,
                    image_sizes=batch.image_sizes,
                    image_grid_thw=batch.image_grid_thw,
                )
                if batch.prefill_cache_indices is not None:
                    batch.prefill_cache_indices = None
                if batch.pixel_values is not None:
                    batch.pixel_values = None
                if batch.pixel_attention_mask is not None:
                    batch.pixel_attention_mask = None
                if batch.image_sizes is not None:
                    batch.image_sizes = None
                if batch.image_grid_thw is not None:
                    batch.image_grid_thw = None
                return logits, speculative_logits

        # Copy inputs to the static inputs of the cuda graph
        # Static inputs are potentially padded
        cuda_graph["input_ids"][: input_ids.shape[0]] = input_ids
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
        return logits, speculative_logits
