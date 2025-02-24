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
from transformers.models.llava_next.modeling_llava_next import (
    unpad_image,
)
from optimum.habana.transformers.models import GaudiLlavaNextForConditionalGeneration 
from transformers.image_processing_utils import select_best_resolution

def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    """
    Calculate the shape of the image patch grid after the preprocessing for images of any resolution.

    Args:
        image_size (`tuple`):
            The size of the input image in the format (width, height).
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


class LlavaNextForConditionalGeneration(GaudiLlavaNextForConditionalGeneration):
       
    def _merge_input_ids_with_image_features(
        self,
        inputs_embeds: torch.Tensor,
        image_features: torch.Tensor,
        input_ids: torch.Tensor,
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
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        image_sizes: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[int] = None,
        vision_feature_select_strategy: Optional[str] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        token_idx: Optional[torch.Tensor] = None,
        use_flash_attention: Optional[bool] = False,
        flash_attention_recompute: Optional[bool] = False,
    ):

        if token_idx is not None:
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict
            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings()(input_ids)

            outputs = self.language_model(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                token_idx=token_idx,
                use_flash_attention=use_flash_attention,
                flash_attention_recompute=flash_attention_recompute,
            )

            logits = outputs[0]

            if not return_dict:
                output = (logits,) + outputs[1:]
                return output

        return outputs

    def prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=None,
            inputs_embeds=None,
            pixel_values=None,
            image_sizes=None,
            attention_mask=None,
            **kwargs,
        ):
            """
            Inherits from LlavaForConditionalGeneration: https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/llava_next/modeling_llava_next.py#L635
            The only differences are:
            - add new args token_idx
            - add the process of merging images into inputs_embeds
            """
            token_idx = kwargs.get("token_idx", None)
            if token_idx is None:
                return super().prepare_inputs_for_generation(
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    pixel_values=pixel_values,
                    image_sizes=image_sizes,
                    attention_mask=attention_mask,
                    **kwargs,
                )
            else:
                use_flash_attention = kwargs.get("use_flash_attention", False)
                flash_attention_recompute = kwargs.get("flash_attention_recompute", False)
                
                position_ids = kwargs.get("position_ids", None)
                labels = kwargs.get("labels", None)
                if past_key_values is None and pixel_values is not None and input_ids.shape[1] != 1:
                    vision_feature_select_strategy = kwargs.get("vision_feature_select_strategy", None)
                    vision_feature_layer = kwargs.get("vision_feature_layer", None)
                    vision_feature_select_strategy = (
                        vision_feature_select_strategy
                        if vision_feature_select_strategy is not None
                        else self.config.vision_feature_select_strategy
                    )
                    vision_feature_layer = (
                        vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
                    )

                    # 1. Extract the input embeddings
                    inputs_embeds = self.get_input_embeddings()(input_ids)
                    # 2. Merge text and images
                    batch_size, num_patches, num_channels, height, width = pixel_values.shape
                    reshaped_pixel_values = pixel_values.view(batch_size * num_patches, num_channels, height, width)
                    image_features = self.vision_tower(
                        reshaped_pixel_values,
                        output_hidden_states=True,
                        use_flash_attention=use_flash_attention,
                        flash_attention_recompute=flash_attention_recompute,
                    )

                    selected_image_feature = image_features.hidden_states[vision_feature_layer]

                    if vision_feature_select_strategy == "default":
                        selected_image_feature = selected_image_feature[:, 1:]
                    elif vision_feature_select_strategy == "full":
                        selected_image_feature = selected_image_feature

                    image_features = self.multi_modal_projector(selected_image_feature)

                    # split up image_features for each of the individual images
                    # hence we get a list of image_features, each of shape (5, num_patches, hidden_size)
                    # if we assume each image has 5 image features (base image + 4 patches)
                    split_sizes = [image.shape[0] for image in pixel_values]
                    image_features = torch.split(image_features, split_sizes, dim=0)

                    # NOTE we only support multimodal_patch_merge_type == "spatial_unpad"
                    height = width = self.config.vision_config.image_size // self.config.vision_config.patch_size

                    new_image_features = []
                    for image_idx, image_feature in enumerate(image_features):
                        if image_feature.shape[0] > 1:
                            base_image_feature = image_feature[0]
                            image_feature = image_feature[1:]

                            if height * width != base_image_feature.shape[0]:
                                raise ValueError("The number of patches is not consistent with the image size.")

                            num_patch_height, num_patch_width = get_anyres_image_grid_shape(
                                image_sizes[image_idx].tolist(),
                                self.config.image_grid_pinpoints,
                                self.config.vision_config.image_size,
                            )
                            
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat(
                                (
                                    image_feature,
                                    self.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1),
                                ),
                                dim=-1,
                            )
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                            image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                        else:
                            image_feature = image_feature[0]
                            image_feature = torch.cat((image_feature, self.image_newline[None]), dim=0)
                        new_image_features.append(image_feature)
                    image_features = torch.stack(new_image_features, dim=0)
                    inputs_embeds = self._merge_input_ids_with_image_features(inputs_embeds, image_features, input_ids)
                    self.image_offset = image_features.shape[1] - 1  # image_token has occupied 1 token position.
                # In case input_ids.shape[1] == 1 & pixel_values==None & past_key_values != None, we are in the case of
                # generation with cache
                elif past_key_values is not None:
                    seq_len = input_ids.shape[1]
                    pad_len = seq_len - token_idx.item()
                    input_ids = torch.index_select(input_ids, 1, token_idx - 1)
                    # Retrieve the first layer to inspect the logits and mask out the hidden states
                    # that are set to 0
                    first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

                    # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
                    batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)

                    # Get the target length
                    past_length = first_layer_past_key_value.shape[-1]
                    extended_attention_mask = torch.ones(
                        (attention_mask.shape[0], past_length),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    # Filter out only the tokens that can be un-attended, this can happen
                    # if one uses Llava + Fused modules where the cache on the
                    # first iteration is already big enough, or if one passes custom cache
                    valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
                    new_batch_index = batch_index[valid_indices]
                    new_non_attended_tokens = non_attended_tokens[valid_indices]

                    # Zero-out the places where we don't need to attend
                    extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

                    attention_mask = extended_attention_mask
                    attention_mask[:, -pad_len:] = 0

                if attention_mask is not None and position_ids is None:
                    # create position_ids on the fly for batch generation
                    position_ids = attention_mask.long().cumsum(-1) - 1
                    position_ids.masked_fill_(attention_mask == 0, 1)
                    if past_key_values:
                        if token_idx is not None:
                            position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
                        else:
                            position_ids = position_ids[:, -input_ids.shape[1] :]

                # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
                if inputs_embeds is not None and past_key_values is None:
                    model_inputs = {"inputs_embeds": inputs_embeds}
                else:
                    model_inputs = {"input_ids": input_ids}

                model_inputs.update(
                    {
                        "position_ids": position_ids,
                        "past_key_values": past_key_values,
                        "use_cache": kwargs.get("use_cache"),
                        "attention_mask": attention_mask,
                        "token_idx": token_idx,
                        "labels": labels,
                        "use_flash_attention": use_flash_attention,
                        "flash_attention_recompute": flash_attention_recompute,
                    }
                )

                return model_inputs
