# coding=utf-8
# Copyright 2025 the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Qwen2.5 VL model."""

from typing import Optional, Tuple, List

import torch
import torch.utils.checkpoint
from torch import nn

from habana_frameworks.torch.hpex.kernels import FusedSDPA
from vllm_hpu_extension.utils import ModuleFusedSDPA


import numpy as np

from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig

import torch.nn.functional as F

from text_generation_server.layers.layernorm import FastRMSNorm
from text_generation_server.layers import (
    TensorParallelColumnLinear,
    TensorParallelRowLinear,
    TensorParallelEmbedding,
    SpeculativeHead,
)
from text_generation_server.layers.attention import (
    Seqlen,
    HPUPagedAttentionMetadata,
)
from text_generation_server.models.custom_modeling.flash_qwen2_modeling import (
    Qwen2Model,
)

# Copied from: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_5_vl/processing_qwen2_5_vl.py
from typing import Union
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput, VideoInput
from transformers.processing_utils import (
    ProcessingKwargs,
    ProcessorMixin,
    Unpack,
    VideosKwargs,
)
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput


class Qwen2_5_VLVideosProcessorKwargs(VideosKwargs, total=False):
    fps: Union[List[float], float]


class Qwen2_5_VLProcessorKwargs(ProcessingKwargs, total=False):
    videos_kwargs: Qwen2_5_VLVideosProcessorKwargs
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
        "videos_kwargs": {"fps": 2.0},
    }


class Qwen2_5_VLProcessor(ProcessorMixin):
    r"""
    Constructs a Qwen2.5-VL processor which wraps a Qwen2.5-VL image processor and a Qwen2 tokenizer into a single processor.
    [`Qwen2_5_VLProcessor`] offers all the functionalities of [`Qwen2VLImageProcessor`] and [`Qwen2TokenizerFast`]. See the
    [`~Qwen2_5_VLProcessor.__call__`] and [`~Qwen2_5_VLProcessor.decode`] for more information.
    Args:
        image_processor ([`Qwen2VLImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`Qwen2TokenizerFast`], *optional*):
            The tokenizer is a required input.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]

    image_processor_class = "AutoImageProcessor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    def __init__(
        self, image_processor=None, tokenizer=None, chat_template=None, **kwargs
    ):
        self.image_token = (
            "<|image_pad|>"
            if not hasattr(tokenizer, "image_token")
            else tokenizer.image_token
        )
        self.video_token = (
            "<|video_pad|>"
            if not hasattr(tokenizer, "video_token")
            else tokenizer.video_token
        )
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[
            TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]
        ] = None,
        videos: VideoInput = None,
        **kwargs: Unpack[Qwen2_5_VLProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to Qwen2TokenizerFast's [`~Qwen2TokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the vision inputs, this method forwards the `vision_infos` and `kwrags` arguments to
        Qwen2VLImageProcessor's [`~Qwen2VLImageProcessor.__call__`] if `vision_infos` is not `None`.

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            videos (`np.ndarray`, `torch.Tensor`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of videos to be prepared. Each video can be a 4D NumPy array or PyTorch
                tensor, or a nested list of 3D frames. Both channels-first and channels-last formats are supported.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:
                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
            - **pixel_values_videos** -- Pixel values of videos to be fed to a model. Returned when `videos` is not `None`.
            - **image_grid_thw** -- List of image 3D grid in LLM. Returned when `images` is not `None`.
            - **video_grid_thw** -- List of video 3D grid in LLM. Returned when `videos` is not `None`.
            - **second_per_grid_ts** -- List of video seconds per time grid. Returned when `videos` is not `None`.
        """
        output_kwargs = self._merge_kwargs(
            Qwen2_5_VLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        if images is not None:
            image_inputs = self.image_processor(
                images=images, videos=None, **output_kwargs["images_kwargs"]
            )
            image_grid_thw = image_inputs["image_grid_thw"]
        else:
            image_inputs = {}
            image_grid_thw = None

        if videos is not None:
            videos_inputs = self.image_processor(
                images=None, videos=videos, **output_kwargs["images_kwargs"]
            )
            video_grid_thw = videos_inputs["video_grid_thw"]

            fps = output_kwargs["videos_kwargs"].pop("fps", 2.0)
            if isinstance(fps, (int, float)):
                second_per_grid_ts = [
                    self.image_processor.temporal_patch_size / fps
                ] * len(video_grid_thw)
            elif hasattr(fps, "__len__") and len(fps) == len(video_grid_thw):
                second_per_grid_ts = [
                    self.image_processor.temporal_patch_size / tmp for tmp in fps
                ]
            else:
                raise ValueError(
                    f"The length of fps ({len(fps) if hasattr(fps, '__len__') else fps}) must be equal to the length of video_grid_thw ({len(video_grid_thw)}) or fps should be a single number."
                )
            videos_inputs.update({"second_per_grid_ts": second_per_grid_ts})

        else:
            videos_inputs = {}
            video_grid_thw = None

        if not isinstance(text, list):
            text = [text]

        if image_grid_thw is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    text[i] = text[i].replace(
                        self.image_token,
                        "<|placeholder|>"
                        * (image_grid_thw[index].prod() // merge_length),
                        1,
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        if video_grid_thw is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.video_token in text[i]:
                    text[i] = text[i].replace(
                        self.video_token,
                        "<|placeholder|>"
                        * (video_grid_thw[index].prod() // merge_length),
                        1,
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.video_token)

        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        return BatchFeature(data={**text_inputs, **image_inputs, **videos_inputs})

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    def post_process_image_text_to_text(self, generated_outputs):
        """
        Post-process the output of the model to decode the text.

        Args:
            generated_outputs (`torch.Tensor` or `np.ndarray`):
                The output of the model `generate` function. The output is expected to be a tensor of shape `(batch_size, sequence_length)`
                or `(sequence_length,)`.

        Returns:
            `List[str]`: The decoded text.
        """
        return self.tokenizer.batch_decode(
            generated_outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        names_from_processor = list(
            dict.fromkeys(tokenizer_input_names + image_processor_input_names)
        )
        return names_from_processor + ["second_per_grid_ts"]


# Copied from: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_5_vl/configuration_qwen2_5_vl.py
class Qwen2_5_VLVisionConfig(PretrainedConfig):
    model_type = "qwen2_5_vl"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth=32,
        hidden_size=3584,
        hidden_act="silu",
        intermediate_size=3420,
        num_heads=16,
        in_channels=3,
        patch_size=14,
        spatial_merge_size=2,
        spatial_patch_size=14,
        temporal_patch_size=2,
        tokens_per_second=4,
        window_size=112,
        out_hidden_size=3584,
        fullatt_block_indexes=[7, 15, 23, 31],
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_patch_size = spatial_patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.tokens_per_second = tokens_per_second
        self.window_size = window_size
        self.fullatt_block_indexes = fullatt_block_indexes
        self.out_hidden_size = out_hidden_size


class Qwen2_5_VLConfig(PretrainedConfig):

    def __init__(
        self,
        vocab_size=152064,
        hidden_size=8192,
        intermediate_size=29568,
        num_hidden_layers=80,
        num_attention_heads=64,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-05,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=1000000.0,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=80,
        attention_dropout=0.0,
        vision_config=None,
        rope_scaling=None,
        **kwargs,
    ):
        if vision_config is not None:
            self.vision_config = Qwen2_5_VLVisionConfig(**vision_config)

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.rope_scaling = rope_scaling

        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        # and change type from 'mrope' to 'default' because `mrope` does defeault RoPE calculations
        # one can set it to "linear"/"dynamic" etc. to have scaled RoPE
        # TODO: @raushan update config in the hub
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            if self.rope_scaling["type"] == "mrope":
                self.rope_scaling["type"] = "default"
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(
    tensor: torch.Tensor, freqs: torch.Tensor
) -> torch.Tensor:
    orig_dtype = tensor.dtype
    tensor = tensor.float()
    cos = freqs.cos()
    sin = freqs.sin()
    cos = cos.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    sin = sin.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    output = (tensor * cos) + (rotate_half(tensor) * sin)
    output = output.to(orig_dtype)
    return output


class Qwen2_5VLAttention(nn.Module):
    def __init__(self, *, prefix, config, weights):
        super().__init__()
        self.embed_dim = config.hidden_size // weights.process_group.size()
        self.head_dim = config.hidden_size // config.num_heads
        self.num_heads = config.num_heads // weights.process_group.size()

        self.qkv = TensorParallelColumnLinear.load_qkv(
            config,
            prefix=f"{prefix}.qkv",
            weights=weights,
            bias=False,
            num_heads=self.num_heads,
            num_key_value_heads=self.num_heads,
        )
        self.qkv.linear.bias = weights.get_sharded(f"{prefix}.qkv.bias", dim=0)

        self.proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.proj",
            weights=weights,
            bias=True,
        )
        self.softmax_scale = 1.0 / np.sqrt(self.embed_dim // self.num_heads)

    def forward(
        self,
        hidden_state: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        # apply the qkv linear layer to the hidden state
        qkv = self.qkv(hidden_state)
        query, key, value = qkv.split(
            [self.embed_dim, self.embed_dim, self.embed_dim], dim=1
        )

        # reshape the query, key, and value tensors
        _shape = (
            hidden_state.shape[0],
            self.num_heads,
            self.embed_dim // self.num_heads,
        )
        query = query.view(*_shape)
        key = key.view(*_shape)
        value = value.view(*_shape)

        # apply rotary positional embeddings
        query = apply_rotary_pos_emb_vision(query.unsqueeze(0), rotary_pos_emb).squeeze(
            0
        )
        key = apply_rotary_pos_emb_vision(key.unsqueeze(0), rotary_pos_emb).squeeze(0)

        # calc maximum sequence length for any batch
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        causal = False

        # execute sdpa
        query = query.unsqueeze(0).transpose(1, 2)
        key = key.unsqueeze(0).transpose(1, 2)
        value = value.unsqueeze(0).transpose(1, 2)
        fsdpa_op = ModuleFusedSDPA(FusedSDPA)
        attn_output = fsdpa_op(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=causal,
            scale=None,
            softmax_mode="None",
            recompute_mode=None,
            valid_sequence_lengths=None,
        )
        attn_output = attn_output.transpose(1, 2).squeeze(0).contiguous()

        # reshape output to original dimensions
        attn_output = attn_output.reshape(hidden_state.shape[0], -1)
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen2_5VLVisionMLP(nn.Module):
    def __init__(self, *, prefix, config, weights):
        super().__init__()
        self.activation_fn = ACT2FN[config.hidden_act]

        self.intermediate_size = (
            config.intermediate_size // weights.process_group.size()
        )

        self.up = TensorParallelColumnLinear.load(
            prefix=f"{prefix}.up_proj", weights=weights, config=config, bias=True
        )
        self.gate = TensorParallelColumnLinear.load(
            prefix=f"{prefix}.gate_proj", weights=weights, config=config, bias=True
        )
        self.down = TensorParallelRowLinear.load(
            prefix=f"{prefix}.down_proj", weights=weights, config=config, bias=True
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_states = self.gate(hidden_states)
        up_states = self.up(hidden_states)
        activated_states = self.activation_fn(gate_states) * up_states
        down_states = self.down(activated_states)
        return down_states


class Qwen2_5VLVisionBlock(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        self.attn = Qwen2_5VLAttention(
            prefix=f"{prefix}.attn",
            config=config,
            weights=weights,
        )
        self.norm1 = FastRMSNorm.load(
            prefix=f"{prefix}.norm1",
            weights=weights,
            eps=1e-6,
        )
        self.norm2 = FastRMSNorm.load(
            prefix=f"{prefix}.norm2",
            weights=weights,
            eps=1e-6,
        )
        self.mlp = Qwen2_5VLVisionMLP(
            prefix=f"{prefix}.mlp",
            config=config,
            weights=weights,
        )

    def forward(
        self, hidden_states, cu_seqlens, rotary_pos_emb, max_seqlen
    ) -> torch.Tensor:
        norm1_out, _ = self.norm1(hidden_states)
        attn_out = self.attn(norm1_out, cu_seqlens, rotary_pos_emb, max_seqlen)
        hidden_states = hidden_states + attn_out
        norm2_out, _ = self.norm2(hidden_states)
        mlp_out = self.mlp(norm2_out)
        hidden_states = hidden_states + mlp_out
        return hidden_states


class Qwen2_5VLPatchMerger(nn.Module):
    def __init__(self, *, prefix, config, weights):
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.patch_merger_ln_q = FastRMSNorm.load(
            prefix=f"{prefix}.ln_q",
            weights=weights,
            eps=1e-6,
        )
        self.fc1 = TensorParallelColumnLinear.load(
            prefix=f"{prefix}.mlp.0", weights=weights, config=config, bias=True
        )
        self.fc2 = TensorParallelRowLinear.load(
            prefix=f"{prefix}.mlp.2", weights=weights, config=config, bias=True
        )

    def forward(self, hidden_states) -> torch.Tensor:
        hidden_states, _ = self.patch_merger_ln_q(hidden_states)
        hidden_states = hidden_states.view(-1, self.hidden_size)
        hidden_states = self.fc1(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class Qwen2_5VisionModel(nn.Module):
    def __init__(self, *, prefix, config, weights):
        super().__init__()

        self.spatial_merge_size = config.spatial_merge_size
        kernel_size = [config.temporal_patch_size, config.patch_size, config.patch_size]
        self.patch_embedding = nn.Conv3d(
            in_channels=config.in_channels,
            out_channels=config.hidden_size,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=False,
        )
        self.patch_embedding.weight = nn.Parameter(
            weights.get_tensor(f"{prefix}.patch_embed.proj.weight"), requires_grad=False
        )
        head_dim = config.hidden_size // config.num_heads

        theta = 10000.0
        dim = head_dim // 2
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self.blocks = nn.ModuleList(
            [
                Qwen2_5VLVisionBlock(
                    prefix=f"{prefix}.blocks.{i}",
                    config=config,
                    weights=weights,
                )
                for i in range(config.depth)
            ]
        )
        self.merger = Qwen2_5VLPatchMerger(
            prefix=f"{prefix}.merger",
            config=config,
            weights=weights,
        )
        # import ipdb; ipdb.set_trace()
        self.temporal_patch_size = config.temporal_patch_size
        self.spatial_patch_size = config.spatial_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size
        self.window_size = config.window_size
        self.patch_size = config.patch_size
        self.spatial_merge_unit = config.spatial_merge_size * config.spatial_merge_size
        self.fullatt_block_indexes = config.fullatt_block_indexes

    def apply_class_embedding(self, hidden_state: torch.Tensor) -> torch.Tensor:
        batch_size, _, hidden_size = hidden_state.shape
        class_embedding = self.class_embedding.expand(batch_size, 1, hidden_size)
        hidden_state = torch.cat([class_embedding, hidden_state], dim=1)
        return hidden_state

    def get_window_index(self, grid_thw):
        window_index: list = []
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = (
            self.window_size // self.spatial_merge_size // self.patch_size
        )

        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = (
                grid_h // self.spatial_merge_size,
                grid_w // self.spatial_merge_size,
            )
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(
                grid_t, llm_grid_h, llm_grid_w
            )
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = (
                seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            )
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)

        return window_index, cu_window_seqlens

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:

        # reshape the input tensor for processing
        shape = (
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.spatial_patch_size,
            self.spatial_patch_size,
        )
        pixel_values = pixel_values.view(shape).to(self.patch_embedding.weight.dtype)
        hidden_states = self.patch_embedding(pixel_values).view(-1, self.embed_dim)
        # TODO: revisit to see if we can avoid some of these reshapes

        # find the position ids for the input tensor based on the grid_thw
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))

        pos_ids = torch.cat(pos_ids, dim=0)

        max_grid_size = grid_thw[:, 1:].max()

        # apply the positional embeddings to the position ids
        seq = torch.arange(
            max_grid_size, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        rotary_pos_emb_full = torch.outer(seq, self.inv_freq)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        seq_len = hidden_states.shape[0]
        patch_shape = (seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        og_shape = (seq_len, -1)

        hidden_states = hidden_states.view(patch_shape)[window_index, :, :].view(
            og_shape
        )
        rotary_pos_emb = rotary_pos_emb.view(patch_shape)[window_index, :, :].view(
            og_shape
        )

        rotary_pos_emb = rotary_pos_emb.to(device=hidden_states.device)

        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device="cpu",
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens).to(
            hidden_states.device
        )

        # create a cu_seqlens tensor to be used in the attention mask
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        max_seqlen = torch.max(cu_seqlens[1:] - cu_seqlens[:-1])

        # iterately apply the blocks to the hidden states
        for layer_num, block in enumerate(self.blocks):
            # NOTE: qwen2_5_vl.py has a concept of full attention blocks
            # that are applied at specific layers.
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens

            hidden_states = block(
                hidden_states, cu_seqlens_now, rotary_pos_emb, max_seqlen
            )

        # apply the final patch merger to the hidden states
        hidden_states = self.merger(hidden_states)
        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :]
        return hidden_states


class Qwen2_5VLForConditionalGeneration(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        self.config = config
        config.vision_config.quantize = None
        config.vision_config.speculator = config.speculator
        # set rope_scaling.type == "mrope" since AutoConfig.from_pretrained incorrectly
        # returns rope_scaling.type == "default" for Qwen2_5-VL model at the moment
        if (
            hasattr(config, "rope_scaling")
            and config.rope_scaling is not None
            and config.rope_scaling.get("type", None) == "default"
        ):
            config.rope_scaling.update({"rope_type": "mrope"})
        self.hidden_size = config.hidden_size
        self.vision_start_token_id = config.vision_start_token_id
        self.vision_end_token_id = config.vision_end_token_id
        self.image_token_id = config.image_token_id
        self.video_token_id = config.video_token_id
        self.spatial_merge_size = config.vision_config.spatial_merge_size
        self.embed_tokens = TensorParallelEmbedding(
            prefix="model.embed_tokens", weights=weights
        )
        self.visual = Qwen2_5VisionModel(
            prefix="visual", config=config.vision_config, weights=weights
        )
        self.text_model = Qwen2Model(prefix=None, config=config, weights=weights)
        if config.tie_word_embeddings:
            suffix = "model.embed_tokens"
        else:
            suffix = "lm_head"

        self.lm_head = SpeculativeHead.load(
            config,
            prefix=suffix if not prefix else f"{prefix}.{suffix}",
            weights=weights,
        )
        self.device = weights.device

    # based on https://github.com/huggingface/transformers/blob/e284c7e954abe12c34b50461c17f8115a0afe115/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py#L1391
    # modified to first find segments then initialize position ids for each segment
    # Steps:
    #  locate all vision and text segments
    #  calculate `vision_segment_lengths` for each vision segment to be use as offset
    #  calculate `text_segment_lengths` for each text segment to be used as offset
    #  create position ids for each vision segment based on the image grid
    #  create position ids for each text segment
    #  combine all the position ids
    #  the final segment is the difference between the last vision segment and the end of the input
    #  combine all the position ids and reshape to (3, input_ids_len) then swap dimensions to (input_ids_len, 3)
    def get_position_ids(
        self,
        input_ids: torch.Tensor,
        image_grid_thw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if image_grid_thw is None:
            return (
                torch.arange(input_ids.shape[0], device=input_ids.device)
                .unsqueeze(1)
                .repeat(1, 3)
            )

        spatial_merge_size = self.spatial_merge_size
        vision_start_token_id = self.vision_start_token_id
        vision_end_token_id = self.vision_end_token_id
        device = input_ids.device
        dtype = input_ids.dtype
        input_ids_len = input_ids.shape[0]

        vision_starts = torch.where(input_ids == vision_start_token_id)[0]
        vision_ends = torch.where(input_ids == vision_end_token_id)[0]
        vision_segments = torch.stack((vision_starts, vision_ends), dim=1)
        prev_vision_end = torch.cat(
            [torch.zeros(1, device=vision_ends.device, dtype=dtype), vision_ends[:-1]]
        )
        text_lengths_between_vision = vision_segments[:, 0] - prev_vision_end + 1
        vision_widths_max = torch.cat(
            [
                torch.zeros(1, device=image_grid_thw.device, dtype=dtype),
                image_grid_thw[:-1, 2] // spatial_merge_size,
            ]
        )
        vision_segment_lengths = vision_widths_max + text_lengths_between_vision
        vision_segment_lengths = vision_segment_lengths.cumsum(dim=0)
        text_segment_lengths = vision_segment_lengths - text_lengths_between_vision

        # create position ids for each vision segment based on the image grid
        llm_pos_ids_list = []
        for i, _ in enumerate(vision_segments):
            t, h, w = (
                image_grid_thw[i][0],
                image_grid_thw[i][1] // spatial_merge_size,
                image_grid_thw[i][2] // spatial_merge_size,
            )
            t_indices = torch.arange(t, device=device).repeat_interleave(h * w)
            h_indices = torch.arange(h, device=device).repeat_interleave(w).repeat(t)
            w_indices = torch.arange(w, device=device).repeat(t * h)
            image_position_ids = torch.stack([t_indices, h_indices, w_indices], dim=0)

            # offset by the position of the last vision segment
            im = image_position_ids + vision_segment_lengths[i]
            llm_pos_ids_list.append(im)

        # create position ids for each text segment
        text_ranges = [
            torch.arange(seq_len, device=device).view(1, -1).expand(3, -1)
            + text_segment_lengths[i]
            for i, seq_len in enumerate(text_lengths_between_vision)
        ]

        full_llm_pos_ids_list = [
            item for sublist in zip(text_ranges, llm_pos_ids_list) for item in sublist
        ]
        # import ipdb

        # ipdb.set_trace()
        max_s = full_llm_pos_ids_list[-1].max() + 1
        final_text_len = input_ids_len - vision_ends[-1]
        if final_text_len > 0:
            m = torch.arange(final_text_len, device=device).view(1, -1).expand(3, -1)
            full_llm_pos_ids_list.append(m + max_s)

        position_ids = (
            torch.cat(full_llm_pos_ids_list, dim=1).reshape(3, -1).transpose(0, 1)
        )
        return position_ids

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlen_prefill: Optional[torch.Tensor],
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        slots: torch.Tensor,
        seqlen: Seqlen,
        hpu_attention_meta: Optional[HPUPagedAttentionMetadata],
        lm_head_indices: Optional[torch.Tensor],
        pixel_values: torch.FloatTensor = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        # Unused in this model
        video_grid_thw: Optional[torch.LongTensor] = None,
        pixel_attention_mask=None,
        image_sizes: Optional[torch.LongTensor] = None,
        adapter_data: Optional[torch.Tensor] = None,
        cross_attention_states: Optional[torch.Tensor] = None,
        image_indices=None,
    ):
        inputs_embeds = self.embed_tokens(input_ids)

        # apply the visual model to the pixel values if they are provided
        if pixel_values is not None and len(pixel_values) > 0:
            if pixel_values is not None:
                image_embeds = self.visual(
                    pixel_values, grid_thw=image_grid_thw
                ).squeeze(0)
                mask = torch.where(input_ids == self.image_token_id)
                inputs_embeds[mask] = image_embeds

        hidden_states = self.text_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            cu_seqlen_prefill=cu_seqlen_prefill,
            kv_cache=kv_cache,
            slots=slots,
            seqlen=seqlen,
            hpu_attention_meta=hpu_attention_meta,
        )
        if lm_head_indices is not None:
            hidden_states = hidden_states[lm_head_indices]
        logits, speculative_logits = self.lm_head(hidden_states)
        return logits, speculative_logits
