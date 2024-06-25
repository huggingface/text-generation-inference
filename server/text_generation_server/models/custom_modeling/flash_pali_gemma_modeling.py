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

from text_generation_server.layers.tensor_parallel import TensorParallelColumnLinear
from text_generation_server.models.custom_modeling.vlm import (
    load_text_model,
    load_vision_model,
)


class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        config.vision_config.quantize = config.quantize
        self.vision_tower = load_vision_model(
            prefix="vision_tower" if not prefix else f"{prefix}.vision_tower",
            config=config.vision_config,
            weights=weights,
        )

        self.multi_modal_projector = TensorParallelColumnLinear.load(
            config,
            prefix="multi_modal_projector.linear",
            weights=weights,
            bias=True,
        )

        self.vocab_size = config.vocab_size
        self.config = config

        text_config = config.text_config
        text_config.speculator = config.speculator
        text_config.quantize = config.quantize
        self.text_model = load_text_model(
            prefix="language_model" if not prefix else f"{prefix}.language_model",
            config=config.text_config,
            weights=weights,
        )
        self.pad_token_id = (
            config.pad_token_id if config.pad_token_id is not None else -1
        )

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
        # Unused here
        pixel_attention_mask: Optional[torch.BoolTensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        adapter_data: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        inputs_embeds = self.text_model.embed_tokens(input_ids)
        # TODO This is odd but apparently pali gemma position ids start at 1.
        if cu_seqlen_prefill is not None:
            max_s += 1
            position_ids += 1

        if pixel_values is not None:
            pixel_values = pixel_values.to(dtype=inputs_embeds.dtype)
            image_outputs = self.vision_tower(pixel_values)
            image_features = self.multi_modal_projector(image_outputs.last_hidden_state)

            # mask where image or padding tokens
            mask = input_ids == self.config.image_token_index

            # insert image features into input embeddings
            inputs_embeds[mask] = image_features.view(-1, image_features.shape[-1])

        hidden_states = self.text_model.model(
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
        logits, speculative_logits = self.text_model.lm_head(hidden_states)

        return logits, speculative_logits
