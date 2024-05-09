# coding=utf-8
# Copyright 2024 HuggingFace Inc. team. All rights reserved.
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

import torch
import torch.distributed
from torch import nn
from transformers.configuration_utils import PretrainedConfig
from typing import Optional, List, Tuple

from text_generation_server.utils.layers import TensorParallelColumnLinear
from text_generation_server.models.custom_modeling.vlm import (
    load_text_model,
    load_vision_model,
)
from text_generation_server.models.custom_modeling.flash_gemma_modeling import (
    GemmaConfig,
)


class PaliGemmaConfig(PretrainedConfig):
    model_type = "paligemma"

    def from_pretrained(pretrained_model_name_or_path, **kwargs):
        vision_config = VisionConfig(
            hidden_size=1152,
            intermediate_size=4304,
            model_type="siglip_vision_model",
            num_attention_heads=16,
            num_hidden_layers=27,
            num_image_tokens=256,
            patch_size=14,
            projection_dim=2048,
            projector_hidden_act="gelu_fast",
            vision_use_head=False,
            vocab_size=257152,
        )

        return GemmaConfig.from_pretrained(
            pretrained_model_name_or_path, vision_config=vision_config, **kwargs
        )


class VisionConfig(PretrainedConfig):
    def __init__(
        self,
        hidden_size: int = 1152,
        intermediate_size: int = 4304,
        model_type: str = "siglip_vision_model",
        num_attention_heads: int = 16,
        num_hidden_layers: int = 27,
        num_image_tokens: int = 256,
        patch_size: int = 14,
        projection_dim: int = 2048,
        projector_hidden_act: str = "gelu_fast",
        vision_use_head: bool = False,
        vocab_size: int = 257152,
        quantize: Optional[str] = None,
        image_size: int = 224,
        layer_norm_eps: float = 1e-06,
        attention_dropout: float = 0.0,
        hidden_act: str = "gelu_pytorch_tanh",
        num_channels: int = 3,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.model_type = model_type
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_image_tokens = num_image_tokens
        self.patch_size = patch_size
        self.projection_dim = projection_dim
        self.projector_hidden_act = projector_hidden_act
        self.vision_use_head = vision_use_head
        self.vocab_size = vocab_size
        self.quantize = quantize
        self.image_size = image_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.hidden_act = hidden_act
        self.num_channels = num_channels

        super().__init__(**kwargs)


class FlashPaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        config.vision_config.quantize = config.quantize
        self.vision_tower = load_vision_model(
            prefix="vision_tower" if not prefix else f"{prefix}.vision_tower",
            config=config.vision_config,
            weights=weights,
        ).to(weights.device, weights.dtype)

        self.multi_modal_projector = TensorParallelColumnLinear.load(
            config,
            prefix="multi_modal_projector.linear",
            weights=weights,
            bias=True,
        ).to(weights.device, weights.dtype)

        self.vocab_size = config.vocab_size
        self.config = config

        self.language_model = load_text_model(
            prefix=prefix,
            config=config,
            weights=weights,
        ).to(weights.device, weights.dtype)
        self.pad_token_id = (
            config.pad_token_id if config.pad_token_id is not None else -1
        )

    def _merge_input_ids_with_image_features(
        self, image_features, inputs_embeds, input_ids
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
        prefill_cache_indices: Optional[torch.Tensor] = None,
        lm_head_indices: Optional[torch.Tensor] = None,
        pixel_values: torch.FloatTensor = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        pixel_attention_mask=None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        if pixel_values is not None:
            pixel_values = pixel_values.to(inputs_embeds.device, inputs_embeds.dtype)

        # merge text and images
        if pixel_values is not None and len(pixel_values) > 0:
            image_outputs = self.vision_tower(pixel_values)
            selected_image_feature = image_outputs.last_hidden_state
            image_features = self.multi_modal_projector(selected_image_feature)
            image_features = image_features / (self.config.hidden_size**0.5)
            inputs_embeds = self._merge_input_ids_with_image_features(
                image_features, inputs_embeds, input_ids
            )

        if input_ids.size(0) != 3000:
            import ipdb

            ipdb.set_trace()

            ## TODO: remove this
            ## load in values from reference
            # tensor = torch.load("../../new-model-addition-palma/inputs_embeds.npz")
            # inputs_embeds = torch.tensor(
            #     tensor, device=inputs_embeds.device, dtype=inputs_embeds.dtype
            # ).squeeze()

        hidden_states = self.language_model.model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            cu_seqlen_prefill=cu_seqlen_prefill,
            kv_cache=kv_cache,
            block_tables=block_tables,
            slots=slots,
            input_lengths=input_lengths,
            max_s=max_s,
        )

        if lm_head_indices is not None:
            hidden_states = hidden_states[lm_head_indices]
        logits, speculative_logits = self.language_model.lm_head(hidden_states)

        return logits, speculative_logits
