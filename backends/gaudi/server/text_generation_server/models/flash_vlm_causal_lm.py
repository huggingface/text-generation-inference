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


def image_text_replacement(processor, image_input, config, image_id: int) -> str:
    if config.model_type == "idefics2":
        image_seq_len = 64
        image_str = f"{IDEFICS2_FAKE_TOKEN}{IDEFICS2_IMAGE_TOKEN * image_seq_len}{IDEFICS2_FAKE_TOKEN}"
        if processor.image_processor.do_image_splitting:
            image_str *= 5
        return image_str
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
        return image_str
    elif config.model_type == "llava_next":
        height, width = image_input["image_sizes"][image_id]
        num_features = get_number_of_features(height, width, config)

        log_master(
            logger.info,
            f"Found {num_features} features in image of resolution {height}x{width}",
        )
        return "<image>" * num_features

    elif config.model_type == "paligemma":
        return "<image>" * config.text_config.num_image_tokens
    elif config.model_type == "qwen2_vl":
        grid_t, grid_h, grid_w = image_input["image_grid_thw"][image_id]
        num_pads = grid_t * grid_h * grid_w // 4
        padding = "<|image_pad|>" * num_pads
        return f"<|vision_start|>{padding}<|vision_end|>"
    elif config.model_type == "qwen2_5_vl":
        grid_t, grid_h, grid_w = image_input["image_grid_thw"][image_id]
        num_pads = grid_t * grid_h * grid_w // 4
        padding = "<|image_pad|>" * num_pads
        return f"<|vision_start|>{padding}<|vision_end|>"
    elif config.model_type == "gemma3":
        # TODO: get correct number of features via reviewing the Gemma3 architecture
        # and calculating the number of image tokens
        num_pads = 256
        padding = "<image_soft_token>" * num_pads
        return f"\n\n<start_of_image>{padding}<end_of_image>\n\n"
    elif config.model_type == "llama4":
        patch_size = config.vision_config.patch_size
        pixel_shuffle_ratio = config.vision_config.pixel_shuffle_ratio
        downsample_ratio = int(round(1.0 / (pixel_shuffle_ratio**2)))
        aspect_ratios = image_input["aspect_ratios"][image_id]
        image_height, image_width = image_input["pixel_values"][image_id].shape[-2:]

        num_patches_per_chunk = int(
            (image_height // patch_size)
            * (image_width // patch_size)
            // downsample_ratio
        )
        tokens_for_this_image = prompt_split_image_llama4(
            aspect_ratios, num_patches_per_chunk
        )

        return tokens_for_this_image
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


class FlashVlmCausalLMBatch(FlashCausalLMBatch):
    pixel_values: Optional[List[torch.Tensor]]
    pixel_attention_mask: Optional[List[torch.Tensor]]
    image_sizes: Optional[List[Tuple[int, int]]]
    image_grid_thw: Optional[torch.Tensor]

    @classmethod
    @tracer.start_as_current_span("concatenate")
    def concatenate(cls, batches, padded_total_bs: int = 0):
        batch = super(FlashVlmCausalLMBatch, cls).concatenate(batches, padded_total_bs)
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
        images = []
        for r in requests:
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

                    if config.model_type == "llava_next":
                        images.append(image)
                    elif config.model_type == "gemma3":
                        images.append(image)
                    elif config.model_type == "llama4":
                        images.append(image)
                    else:
                        images.append([image])
                else:
                    raise RuntimeError(f"Invalid chunk type {chunk_type}")

        if images:
            kwargs = {}
            if (
                hasattr(processor, "image_processor_class")
                and processor.image_processor_class == "Idefics3ImageProcessor"
            ):
                kwargs["return_row_col_info"] = True

            image_inputs = processor.image_processor(
                images, return_tensors="pt", **kwargs
            )
        else:
            image_inputs = None

        batch_tokenized_inputs = []
        max_length = 0
        image_id = 0
        for r in requests:
            full_text = ""
            for chunk in r.input_chunks.chunks:
                chunk_type = chunk.WhichOneof("chunk")
                if chunk_type == "text":
                    full_text += chunk.text
                elif chunk_type == "image":
                    full_text += image_text_replacement(
                        processor, image_inputs, config, image_id
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
    ) -> "FlashVlmCausalLMBatch":
        batch_tokenized_inputs, image_inputs = cls.batch_tokenized_inputs(
            pb.requests, tokenizer, processor, config
        )
        batch = cls.from_tokenized(pb, tokenizer, batch_tokenized_inputs, dtype, device)
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
            if "image_grid_thw" in image_inputs:
                batch.image_grid_thw = image_inputs["image_grid_thw"].to(device=device)
            else:
                batch.image_grid_thw = None
        else:
            batch.pixel_values = None
            batch.pixel_attention_mask = None
            batch.image_sizes = None
            batch.image_grid_thw = None
        return batch


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
        # We pass a `cu_seqlen_prefill` in order not to have to deal with paged attention cache allocation/deallocation.
        self.model.forward(
            input_ids=_async_h2d_tensor_copy(input_ids),
            position_ids=_async_h2d_tensor_copy(position_ids),
            cu_seqlen_prefill=None,
            kv_cache=self.kv_cache,
            slots=_async_h2d_tensor_copy(slots_tensor),
            seqlen=trim_seqlen_metadata(seqlen),
            hpu_attention_meta=hpu_attention_meta,
            lm_head_indices=None,
            pixel_values=None,
            pixel_attention_mask=None,
            image_sizes=None,
            image_grid_thw=None,
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

        log_master(
            logger.info,
            f"warmup hpu graph time {int(time.time() - start_time)}s warmup shape count {warmup_shape_count}",
        )

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
                    input_ids, batch.image_grid_thw
                )
                batch.position_ids = position_ids

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
            input_ids=input_ids,
            position_ids=_async_h2d_tensor_copy(position_ids),
            cu_seqlen_prefill=_async_h2d_tensor_copy(cu_seqlen_prefill),
            kv_cache=kv_cache,
            slots=_async_h2d_tensor_copy(slots),
            seqlen=trim_seqlen_metadata(seqlen),
            hpu_attention_meta=batch.hpu_attn_meta,
            lm_head_indices=_async_h2d_tensor_copy(lm_head_indices),
            pixel_values=batch.pixel_values,
            pixel_attention_mask=batch.pixel_attention_mask,
            image_sizes=batch.image_sizes,
            image_grid_thw=batch.image_grid_thw,
            **kwargs,
        )
        if batch.pixel_values is not None:
            batch.pixel_values = None
        if batch.pixel_attention_mask is not None:
            batch.pixel_attention_mask = None
        if batch.image_sizes is not None:
            batch.image_sizes = None
        if batch.image_grid_thw is not None:
            batch.image_grid_thw = None
        return logits, speculative_logits
