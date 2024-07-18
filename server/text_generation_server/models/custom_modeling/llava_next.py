# coding=utf-8
# Copyright 2024 the HuggingFace Inc. team. All rights reserved.
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
""" PyTorch Llava-NeXT model."""

from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.image_processing_utils import select_best_resolution

from text_generation_server.models.custom_modeling.vlm import (
    load_text_model,
    load_vision_model,
)
from text_generation_server.layers import (
    TensorParallelColumnLinear,
    TensorParallelRowLinear,
)


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
        tuple: The shape of the image patch grid in the format (height, width).
    """
    if not isinstance(grid_pinpoints, list):
        raise ValueError("grid_pinpoints should be a list of tuples or lists")

    height, width = select_best_resolution(image_size, grid_pinpoints)
    return height // patch_size, width // patch_size


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
        tensor (`torch.Tensor`):
            The image tensor, assumed to be of shape (num_channels, height, width).
        original_size (`tuple`):
            The original size of the image (height, width).

    Returns:
        `torch.Tensor`: The unpadded image tensor.
    """
    original_height, original_width = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor


# Copied from transformers.models.llava.modeling_llava.LlavaMultiModalProjector with Llava->LlavaNext
class LlavaNextMultiModalProjector(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()

        self.linear_1 = TensorParallelColumnLinear.load(
            prefix=f"{prefix}.linear_1", config=config, weights=weights, bias=True
        )
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = TensorParallelRowLinear.load(
            prefix=f"{prefix}.linear_2", config=config, weights=weights, bias=True
        )

    def forward(self, image_features):
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class LlavaNextForConditionalGeneration(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        config.vision_config.quantize = config.quantize
        vision_config = config.vision_config
        # Instead of selecting in hidden_states[-2].
        # Instead compute only the n -2 + 1 layers and don't pool
        if config.vision_feature_layer < 0:
            vision_config.num_hidden_layers += config.vision_feature_layer + 1
        else:
            vision_config.num_hidden_layers = config.vision_feature_layer + 1
        self.vision_tower = load_vision_model(
            prefix="vision_tower" if not prefix else f"{prefix}.vision_tower",
            config=config.vision_config,
            weights=weights,
        )

        self.multi_modal_projector = LlavaNextMultiModalProjector(
            prefix="multi_modal_projector", config=config, weights=weights
        )

        self.image_newline = weights.get_tensor("image_newline")

        self.vocab_size = config.text_config.vocab_size
        self.config = config
        config.text_config.quantize = config.quantize
        config.text_config.speculator = config.speculator
        self.text_model = load_text_model(
            prefix="language_model" if not prefix else f"{prefix}.language_model",
            config=config.text_config,
            weights=weights,
        )
        self.pad_token_id = (
            config.pad_token_id if config.pad_token_id is not None else -1
        )

    def _merge_input_ids_with_image_features(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        image_features: torch.Tensor,
    ):
        """In place merges in vision_embeddings with inputs_embeds."""
        mask = input_ids == self.config.image_token_index
        # Let's pray we have enabled enough slots !
        try:
            inputs_embeds[mask] = image_features.view(-1, image_features.shape[-1])
        except Exception as e:
            raise RuntimeError(
                f"Cannot fill images right now. If error happens at warmup, make sure you have enough `--max-input-tokens`  to handle images. If error happens at regular runtime, please fill in an issue: {e}"
            )
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlen_prefill: Optional[torch.Tensor],
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        block_tables: torch.Tensor,
        slots: torch.Tensor,
        input_lengths: torch.Tensor,
        max_s: int,
        prefill_cache_indices: Optional[torch.Tensor],
        lm_head_indices: Optional[torch.Tensor] = None,
        pixel_values: torch.FloatTensor = None,
        # Unused for this model
        pixel_attention_mask=None,
        image_sizes: Optional[torch.LongTensor] = None,
        adapter_data: Optional[torch.Tensor] = None,
    ):
        inputs_embeds = self.text_model.embed_tokens(input_ids)
        if pixel_values is not None and len(pixel_values) > 0:
            # num_special_image_tokens = (input_ids == self.config.image_token_index).sum()
            # assert num_special_image_tokens == len(pixel_values), f"Received {num_special_image_tokens} for {len(pixel_values)} images, this is invalid"
            # 1. Extract the input embeddings

            # 2. Merge text and images
            num_images, num_patches, channels, height, width = pixel_values.shape
            pixel_values = pixel_values.view(
                num_images * num_patches, channels, height, width
            )
            image_features = self.vision_tower(pixel_values)

            # selected_image_feature = image_features.hidden_states[self.config.vision_feature_layer]
            # Already done within the clip model
            selected_image_feature = image_features.last_hidden_state

            if self.config.vision_feature_select_strategy == "default":
                selected_image_feature = selected_image_feature[:, 1:]
            elif self.config.vision_feature_select_strategy == "full":
                selected_image_feature = selected_image_feature
            else:
                raise RuntimeError(
                    f"Strategy `{self.config.vision_feature_select_strategy}` is not supported/valid."
                )

            image_features = self.multi_modal_projector(selected_image_feature)

            # split up image_features for each of the individual images
            # hence we get a list of image_features, each of shape (5, num_patches, hidden_size)
            # if we assume each image has 5 image features (base image + 4 patches)
            split_sizes = [num_patches] * num_images
            image_features = torch.split(image_features, split_sizes, dim=0)

            # NOTE we only support multimodal_patch_merge_type == "spatial_unpad"
            height = width = (
                self.config.vision_config.image_size
                // self.config.vision_config.patch_size
            )

            new_image_features = []
            for image_idx, image_feature in enumerate(image_features):
                if image_feature.shape[0] > 1:
                    base_image_feature = image_feature[0]
                    image_feature = image_feature[1:]

                    if height * width != base_image_feature.shape[0]:
                        raise ValueError(
                            "The number of patches is not consistent with the image size."
                        )

                    # Dimensions are intentionally swapped to be bug-compatible with
                    # upstream: https://github.com/LLaVA-VL/LLaVA-NeXT/issues/59
                    num_patch_width, num_patch_height = get_anyres_image_grid_shape(
                        image_sizes[image_idx],
                        self.config.image_grid_pinpoints,
                        self.config.vision_config.image_size,
                    )
                    image_feature = image_feature.view(
                        num_patch_height, num_patch_width, height, width, -1
                    )
                    image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                    image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                    image_feature = unpad_image(image_feature, image_sizes[image_idx])
                    image_feature = torch.cat(
                        (
                            image_feature,
                            self.image_newline[:, None, None].expand(
                                *image_feature.shape[:-1], 1
                            ),
                        ),
                        dim=-1,
                    )
                    image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                    image_feature = torch.cat(
                        (base_image_feature, image_feature), dim=0
                    )
                else:
                    image_feature = image_feature[0]
                    image_feature = torch.cat(
                        (image_feature, self.image_newline[None]), dim=0
                    )
                new_image_features.append(image_feature)
            image_features = torch.stack(new_image_features, dim=0)

            inputs_embeds = self._merge_input_ids_with_image_features(
                input_ids, inputs_embeds, image_features
            )

        hidden_states = self.text_model.model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            cu_seqlen_prefill=cu_seqlen_prefill,
            kv_cache=kv_cache,
            block_tables=block_tables,
            slots=slots,
            input_lengths=input_lengths,
            max_s=max_s,
            true_max_s=max_s,
            prefill_cache_indices=None,
            adapter_data=adapter_data,
        )
        if lm_head_indices is not None:
            hidden_states = hidden_states[lm_head_indices]
        logits, speculative_logits = self.text_model.lm_head(hidden_states)
        return logits, speculative_logits
