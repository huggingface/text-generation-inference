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


# TODO: prefer using the following config classes
# * instead of the hack inside of the gemma modeling file
class VisionConfig(PretrainedConfig):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        model_type: str,
        num_attention_heads: int,
        num_hidden_layers: int,
        num_image_tokens: int,
        patch_size: int,
        projection_dim: int,
        projector_hidden_act: str,
        vision_use_head: bool,
        vocab_size: int,
        quantize: Optional[str] = None,
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


class PaliTextConfig(PretrainedConfig):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        model_type: str,
        num_attention_heads: int,
        num_hidden_layers: int,
        num_image_tokens: int,
        num_key_value_heads: int,
        torch_dtype: str,
        vocab_size: int,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.model_type = model_type
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_image_tokens = num_image_tokens
        self.num_key_value_heads = num_key_value_heads
        self.torch_dtype = torch_dtype
        self.vocab_size = vocab_size


class PaliGemmaConfig(PretrainedConfig):
    def __init__(
        self,
        vocab_size=257216,
        hidden_size=2048,
        intermediate_size=24576,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=16,
        head_dim=256,
        hidden_act="gelu_pytorch_tanh",
        max_position_embeddings=8192,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=1,
        tie_word_embeddings=True,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        text_config=None,
        vision_config=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        self.text_config = GemmaConfig(
            hidden_size=2048,
            intermediate_size=16384,
            model_type="gemma",
            num_attention_heads=8,
            num_hidden_layers=18,
            num_image_tokens=256,
            num_key_value_heads=1,
            torch_dtype="float32",
            vocab_size=257216,
        )

        self.vision_config = VisionConfig(
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

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


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
            # TODO: make sure to handle the specialized attention mask correctly
            inputs_embeds = self._merge_input_ids_with_image_features(
                image_features, inputs_embeds, input_ids
            )

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
