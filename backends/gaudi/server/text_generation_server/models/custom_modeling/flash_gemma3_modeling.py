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
from typing import Optional, List, Tuple
import copy

from text_generation_server.layers import (
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    TensorParallelRowLinear,
    get_linear,
    #
    SpeculativeHead,
    TensorParallelMultiAdapterLinear,
    TensorParallelAdapterRowLinear,
)

import torch


from text_generation_server.models.custom_modeling.vlm import (
    load_text_model,
    load_vision_model,
)


from text_generation_server.layers.attention.kv_cache import get_kv_scales
from text_generation_server.layers.rotary import PositionRotaryEmbedding
from text_generation_server.layers.layernorm import (
    FastRMSNorm,
)
from text_generation_server.utils.weights import UnquantizedWeight
from transformers.activations import ACT2FN
from text_generation_server.layers.attention import (
    paged_attention,
    attention,
    Seqlen,
    set_block_mapping,
    HPUPagedAttentionMetadata,
)
import habana_frameworks.torch as htorch

ATTENTION_TYPE_GLOBAL = "global"
ATTENTION_TYPE_LOCAL = "local_sliding"


class Gemma3FastRMSNorm(FastRMSNorm):
    @classmethod
    def load(cls, prefix: str, weights, eps=1e-6):
        dtype = weights.dtype
        weights.dtype = torch.float32
        weight = weights.get_tensor(f"{prefix}.weight") + 1
        weights.dtype = dtype
        new = cls(weight, eps)
        new.dtype = dtype
        return new

    # perform the multiplication in full precision and downcast after
    def forward(self, hidden_states, residual=None):
        if residual is not None:
            hidden_states += residual
        residual = hidden_states
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = hidden_states * self.weight
        return hidden_states.to(self.dtype), residual


def load_attention(config, prefix: str, weights):
    if config.num_attention_heads != config.num_key_value_heads:
        return _load_gqa(config, prefix, weights)
    else:
        return TensorParallelColumnLinear.load_multi(
            config,
            prefixes=[f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"],
            dim=0,
            weights=weights,
            bias=False,
        )


def _load_gqa(config, prefix: str, weights):
    assert config.num_attention_heads % weights.process_group.size() == 0

    weight = weights.get_multi_weights_col(
        prefixes=[f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"],
        dim=0,
    )

    if isinstance(weight, UnquantizedWeight):
        weight.weight = weight.weight.to(dtype=weights.dtype).to(device=weights.device)

        head_size = config.head_dim
        num_heads = config.num_attention_heads // weights.process_group.size()
        num_key_value_heads = config.num_key_value_heads // weights.process_group.size()
        assert list(weight.weight.shape) == [
            (num_heads + 2 * num_key_value_heads) * head_size,
            config.hidden_size,
        ], f"{list(weight.weight.shape)} != {[(num_heads + 2 * config.num_key_value_heads) * head_size, config.hidden_size]}"

    return TensorParallelColumnLinear(get_linear(weight, bias=None))


class FlashGemma3Attention(torch.nn.Module):
    def __init__(
        self,
        prefix: str,
        config,
        weights,
        layer_id,
        causal: bool,
        is_sliding: bool,
        local_rotary_emb,
        global_rotary_emb,
    ):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_size = config.head_dim
        self.causal = causal
        if is_sliding:
            self.window_size = config.sliding_window
            # TODO: remove this hack to support local sliding window
            config = copy.deepcopy(config)
            config.rope_scaling = dict(rope_type="default")
            self.rotary_emb = local_rotary_emb
        else:
            self.window_size = -1
            self.rotary_emb = global_rotary_emb

        self.softmax_scale = (
            config.query_pre_attn_scalar**-0.5
            if config.query_pre_attn_scalar is not None
            else None
        )
        if self.num_heads % weights.process_group.size() != 0:
            raise ValueError(
                f"`num_heads` must be divisible by `num_shards` (got `num_heads`: {self.num_heads} "
                f"and `num_shards`: {weights.process_group.size()}"
            )
        self.num_heads = self.num_heads // weights.process_group.size()
        self.num_key_value_heads = (
            config.num_key_value_heads // weights.process_group.size()
        )
        self.softcap = None  # config.attn_logit_softcapping

        query_key_value = load_attention(config, prefix, weights)
        self.query_key_value = TensorParallelMultiAdapterLinear.load(
            query_key_value,
            layer_id,
            ["q_proj", "k_proj", "v_proj"],
            sizes=[
                self.head_size * config.num_attention_heads,
                self.head_size * config.num_key_value_heads,
                self.head_size * config.num_key_value_heads,
            ],
            process_group=weights.process_group,
        )
        self.kv_scales = get_kv_scales(weights, f"{prefix}")

        o_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.o_proj",
            weights=weights,
            bias=False,
        )
        self.o_proj = TensorParallelAdapterRowLinear.load(
            o_proj,
            layer_id,
            "o_proj",
            process_group=weights.process_group,
        )

        self.num_groups = self.num_heads // self.num_key_value_heads
        self.kv_head_mapping = torch.arange(
            0, self.num_key_value_heads, dtype=torch.int32, device=weights.device
        ).repeat_interleave(self.num_groups)
        self.q_norm = Gemma3FastRMSNorm.load(
            prefix=f"{prefix}.q_norm", weights=weights, eps=config.rms_norm_eps
        )
        self.k_norm = Gemma3FastRMSNorm.load(
            prefix=f"{prefix}.k_norm", weights=weights, eps=config.rms_norm_eps
        )
        self.enable_gqa = self.num_heads != self.num_key_value_heads

    def forward(
        self,
        hidden_states,
        cos,
        sin,
        cu_seqlen_prefill,
        kv_cache,
        slots,
        seqlen,
        adapter_data,
        hpu_attention_meta,
    ):

        qkv = self.query_key_value(hidden_states, adapter_data)
        query, kv = qkv.split(
            [
                self.head_size * self.num_heads,
                2 * self.head_size * self.num_key_value_heads,
            ],
            dim=1,
        )

        kv = kv.view(-1, 2, self.num_key_value_heads * self.head_size)
        key = kv[:, 0]
        value = kv[:, 1]

        query = query.reshape(-1, self.head_size)
        key = key.reshape(-1, self.head_size)

        query, _ = self.q_norm(query.contiguous())
        key, _ = self.k_norm(key.contiguous())

        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_key_value_heads, self.head_size)
        value = value.view(-1, self.num_key_value_heads, self.head_size)

        self.rotary_emb(query, key, cos, sin)

        kv_cache.store(
            key=key,
            value=value,
            slots=slots,
            kv_scales=self.kv_scales,
        )
        # Prefill
        if cu_seqlen_prefill is not None:
            # sdpa
            attn_output = attention(
                query=query,
                key=key,
                value=value,
                kv_cache=kv_cache,
                kv_scales=self.kv_scales,
                seqlen=seqlen,
                softmax_scale=self.softmax_scale,
                window_size_left=self.window_size,
                softcap=self.softcap,
            )
        # Decode
        else:
            attn_output = paged_attention(
                query,
                kv_cache,
                self.kv_head_mapping,
                self.softmax_scale,
                seqlen,
                softcap=self.softcap,
                kv_scales=self.kv_scales,
                hpu_attention_meta=hpu_attention_meta,
            )

        return self.o_proj(
            attn_output.view(-1, self.num_heads * self.head_size), adapter_data
        )


class Gemma3MLP(nn.Module):
    def __init__(self, prefix, config, weights, layer_id):
        super().__init__()
        act = config.hidden_activation
        self.act = (
            ACT2FN[act]
            if "gelu" not in act
            else lambda x: torch.nn.functional.gelu(
                x,
                approximate=(
                    "tanh" if act in ["gelu_fast", "gelu_pytorch_tanh"] else "none"
                ),
            )
        )
        # Fuse gate and up proj
        gate_up_proj = TensorParallelColumnLinear.load_multi(
            config,
            prefixes=[f"{prefix}.gate_proj", f"{prefix}.up_proj"],
            weights=weights,
            dim=0,
            bias=False,
        )
        self.gate_up_proj = TensorParallelMultiAdapterLinear.load(
            gate_up_proj,
            layer_id,
            ["gate_proj", "up_proj"],
            sizes=[
                config.intermediate_size,
                config.intermediate_size,
            ],
            process_group=weights.process_group,
        )

        down_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.down_proj",
            weights=weights,
            bias=False,
        )
        self.down_proj = TensorParallelAdapterRowLinear.load(
            down_proj,
            layer_id,
            "down_proj",
            process_group=weights.process_group,
        )

        self.intermediate_size = (
            config.intermediate_size // weights.process_group.size()
        )

    def forward(self, hidden_states, adapter_data):
        gate_up_states = self.gate_up_proj(hidden_states, adapter_data)
        gate_up_states = gate_up_states.view(-1, 2, self.intermediate_size)
        return self.down_proj(
            self.act(gate_up_states[:, 0]) * gate_up_states[:, 1], adapter_data
        )


class FlashGemma3Layer(nn.Module):
    def __init__(
        self,
        prefix: str,
        config,
        weights,
        layer_id,
        causal: bool,
        is_sliding: bool,
        local_rotary_emb,
        global_rotary_emb,
    ):
        super().__init__()
        self.self_attn = FlashGemma3Attention(
            prefix=f"{prefix}.self_attn",
            config=config,
            weights=weights,
            layer_id=layer_id,
            causal=causal,
            is_sliding=is_sliding,
            local_rotary_emb=local_rotary_emb,
            global_rotary_emb=global_rotary_emb,
        )
        self.mlp = Gemma3MLP(
            prefix=f"{prefix}.mlp", config=config, weights=weights, layer_id=layer_id
        )

        self.input_layernorm = Gemma3FastRMSNorm.load(
            prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Gemma3FastRMSNorm.load(
            prefix=f"{prefix}.post_attention_layernorm",
            weights=weights,
            eps=config.rms_norm_eps,
        )
        self.pre_feedforward_layernorm = Gemma3FastRMSNorm.load(
            prefix=f"{prefix}.pre_feedforward_layernorm",
            weights=weights,
            eps=config.rms_norm_eps,
        )
        self.post_feedforward_layernorm = Gemma3FastRMSNorm.load(
            prefix=f"{prefix}.post_feedforward_layernorm",
            weights=weights,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        hidden_states,
        residual,
        cos,
        sin,
        cu_seqlen_prefill,
        kv_cache,
        slots,
        seqlen,
        adapter_data,
        hpu_attention_meta,
    ):
        normed_hidden_states, res = self.input_layernorm(hidden_states, residual)

        # Self Attention
        attn_output = self.self_attn(
            normed_hidden_states,
            cos,
            sin,
            cu_seqlen_prefill,
            kv_cache,
            slots,
            seqlen,
            adapter_data,
            hpu_attention_meta,
        )

        # faster post attention rms norm
        normed_attn_res_output, _ = self.post_attention_layernorm(attn_output)
        normed_attn_res_output = normed_attn_res_output + res
        res = normed_attn_res_output

        pre_normed, _ = self.pre_feedforward_layernorm(normed_attn_res_output)
        mlp_output = self.mlp(pre_normed, adapter_data)
        post_hidden_states, _ = self.post_feedforward_layernorm(mlp_output)

        return post_hidden_states, normed_attn_res_output


class FlashGemma3Model(torch.nn.Module):
    def __init__(self, prefix: str, config, weights, causal: bool):
        super().__init__()

        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        local_rotary_emb = PositionRotaryEmbedding.static(
            config=config,
            dim=config.head_dim,
            base=config.rope_local_base_freq,
            device=weights.device,
        )
        global_rotary_emb = PositionRotaryEmbedding.static(
            config=config,
            dim=config.head_dim,
            base=config.rope_theta,
            device=weights.device,
        )

        self.layers = nn.ModuleList(
            [
                FlashGemma3Layer(
                    prefix=f"{prefix}.layers.{layer_id}",
                    config=config,
                    weights=weights,
                    layer_id=layer_id,
                    causal=causal,
                    is_sliding=bool((layer_id + 1) % config.sliding_window_pattern),
                    local_rotary_emb=local_rotary_emb,
                    global_rotary_emb=global_rotary_emb,
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = Gemma3FastRMSNorm.load(
            prefix=f"{prefix}.norm", weights=weights, eps=config.rms_norm_eps
        )

        self.head_size = self.layers[0].self_attn.head_size
        self.num_heads = self.layers[0].self_attn.num_heads
        self.num_key_value_heads = self.layers[0].self_attn.num_key_value_heads

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlen_prefill: Optional[torch.Tensor],
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        slots: torch.Tensor,
        seqlen: Seqlen,
        adapter_data: Optional[torch.Tensor],
        hpu_attention_meta: Optional[HPUPagedAttentionMetadata],
    ) -> torch.Tensor:
        if hpu_attention_meta is not None:
            hpu_attention_meta = set_block_mapping(
                hpu_attention_meta, inputs_embeds.shape[0]
            )
        hidden_states = inputs_embeds

        residual = None
        lazy_mode = htorch.utils.internal.is_lazy()
        if lazy_mode:
            htorch.core.mark_step()

        # Get rotary cos and sin for this forward
        # Avoid to index in each layer

        residual = None
        for i, layer in enumerate(self.layers):
            # Get rotary cos and sin for this forward
            # Avoid to index in each layer
            cos, sin = layer.self_attn.rotary_emb.get_cos_sin(position_ids)
            hidden_states, residual = layer(
                hidden_states,
                residual,
                cos,
                sin,
                cu_seqlen_prefill,
                kv_cache[i],
                slots,
                seqlen,
                adapter_data,
                hpu_attention_meta,
            )
            if lazy_mode:
                htorch.core.mark_step()

        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states


class FlashGemma3ForCausalLM(torch.nn.Module):
    def __init__(self, prefix: str, config, weights, *, causal: bool = True):
        super().__init__()

        embed_norm = config.hidden_size**0.5
        if not prefix:
            prefix = "model"
        else:
            prefix = f"{prefix}.model"

        self.embed_tokens = TensorParallelEmbedding(
            prefix=f"{prefix}.embed_tokens", weights=weights
        )
        self.embed_tokens.weight *= embed_norm

        self.model = FlashGemma3Model(
            prefix=prefix, config=config, weights=weights, causal=causal
        )
        self.lm_head = SpeculativeHead.load(
            prefix=(
                f"{prefix}.embed_tokens"
                if config.tie_word_embeddings
                else f"{prefix}.lm_head"
            ),
            config=config,
            weights=weights,
        )
        # self.softcap = config.attn_logit_softcapping
        # assert isinstance(self.softcap, float)
        self.softcap = None

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlen_prefill: Optional[torch.Tensor],
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        slots: torch.Tensor,
        seqlen: Seqlen,
        hpu_attention_meta: Optional[HPUPagedAttentionMetadata],
        lm_head_indices: Optional[torch.Tensor] = None,
        adapter_data: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_embeds = self.embed_tokens(input_ids)

        hidden_states = self.model(
            input_embeds,
            position_ids,
            cu_seqlen_prefill,
            kv_cache,
            slots,
            seqlen,
            adapter_data,
            hpu_attention_meta,
        )
        if lm_head_indices is not None:
            hidden_states = hidden_states[lm_head_indices]
        logits, speculative_logits = self.lm_head(hidden_states)

        return logits, speculative_logits


class Gemma3MultimodalInputProjection(torch.nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()

        self.mm_input_projection_weight = weights.get_tensor(
            "multi_modal_projector.mm_input_projection_weight"
        )

        self.mm_soft_emb_norm = Gemma3FastRMSNorm.load(
            prefix=f"{prefix}.mm_soft_emb_norm",
            weights=weights,
            eps=config.vision_config.layer_norm_eps,
        )

        self.patches_per_image = int(
            config.vision_config.image_size // config.vision_config.patch_size
        )
        self.tokens_per_side = int(config.mm_tokens_per_image**0.5)
        self.kernel_size = self.patches_per_image // self.tokens_per_side
        self.avg_pool = nn.AvgPool2d(
            kernel_size=self.kernel_size, stride=self.kernel_size
        )

    def forward(self, vision_outputs: torch.Tensor):
        batch_size, _, seq_length = vision_outputs.shape

        reshaped_vision_outputs = vision_outputs.transpose(1, 2)
        reshaped_vision_outputs = reshaped_vision_outputs.reshape(
            batch_size, seq_length, self.patches_per_image, self.patches_per_image
        )
        reshaped_vision_outputs = reshaped_vision_outputs.contiguous()

        pooled_vision_outputs = self.avg_pool(reshaped_vision_outputs)
        pooled_vision_outputs = pooled_vision_outputs.flatten(2)
        pooled_vision_outputs = pooled_vision_outputs.transpose(1, 2)

        normed_vision_outputs, _ = self.mm_soft_emb_norm(pooled_vision_outputs)

        projected_vision_outputs = torch.matmul(
            normed_vision_outputs, self.mm_input_projection_weight
        )
        return projected_vision_outputs.type_as(vision_outputs)


class Gemma3ForConditionalGeneration(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()

        self.config = config

        if config.vision_config is not None:

            config.vision_config.quantize = config.quantize

            self.post_vision_model_layernorm = nn.LayerNorm.load(
                prefix="vision_tower.vision_model.post_layernorm",
                weights=weights,
                eps=config.vision_config.layer_norm_eps,
            )

            self.multimodal_projector = Gemma3MultimodalInputProjection(
                prefix="multi_modal_projector",
                config=config,
                weights=weights,
            )

            text_config = config.text_config
            text_config.speculator = config.speculator
            text_config.quantize = config.quantize

            self.vision_model = load_vision_model(
                prefix="vision_tower" if not prefix else f"{prefix}.vision_tower",
                config=config.vision_config,
                weights=weights,
            )

            self.text_model = load_text_model(
                prefix="language_model" if not prefix else f"{prefix}.language_model",
                config=config.text_config,
                weights=weights,
            )
        else:
            config.text_config.quantize = config.quantize
            config.text_config.speculator = config.speculator
            self.text_model = load_text_model(
                prefix=prefix,
                config=config.text_config,
                weights=weights,
            )

        self.pad_token_id = (
            config.pad_token_id if config.pad_token_id is not None else -1
        )
        self.dtype = weights.dtype

    def get_vision_embeds(
        self,
        pixel_values: torch.FloatTensor,
        pixel_attention_mask: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
    ):
        pixel_values = pixel_values.to(dtype=self.dtype)
        image_outputs = self.vision_model(pixel_values)
        vision_outputs = self.post_vision_model_layernorm(
            image_outputs.last_hidden_state
        )
        image_features = self.multimodal_projector(vision_outputs)
        image_features = image_features.view(-1, image_features.shape[-1])
        return image_features

    def get_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        vision_embeds: torch.Tensor = None,
    ):
        inputs_embeds = self.text_model.embed_tokens(input_ids)

        if vision_embeds is not None:
            # Replace the image token embeddings with the vision features
            image_token_mask = (input_ids == self.config.image_token_index).to(
                input_ids.device
            )
            inputs_embeds[image_token_mask] = vision_embeds.view(
                -1, vision_embeds.shape[-1]
            )
        return inputs_embeds

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlen_prefill: Optional[torch.Tensor],
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        slots: torch.Tensor,
        seqlen: Seqlen,
        hpu_attention_meta: Optional[HPUPagedAttentionMetadata],
        lm_head_indices: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        adapter_data: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if cu_seqlen_prefill is not None:
            position_ids += 1

        if attention_mask is not None:
            min_dtype = torch.finfo(inputs_embeds.dtype).min
            # prefill may be larger than sliding window
            effective_seq_len = max(
                position_ids.shape[0], self.config.text_config.sliding_window
            )
            sliding_window_mask = torch.tril(
                torch.ones_like(attention_mask, dtype=torch.bool),
                diagonal=-self.config.text_config.sliding_window,
            )
            attention_mask_local = torch.where(
                sliding_window_mask, min_dtype, attention_mask
            )
            offset = max(0, position_ids.shape[0] - effective_seq_len)
            attention_mask_local = attention_mask_local[
                :, :, :, offset : offset + effective_seq_len
            ]
        else:
            attention_mask_local = None

        hidden_states = self.text_model.model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            cu_seqlen_prefill=cu_seqlen_prefill,
            kv_cache=kv_cache,
            slots=slots,
            seqlen=seqlen,
            hpu_attention_meta=hpu_attention_meta,
            adapter_data=adapter_data,
        )

        if lm_head_indices is not None:
            hidden_states = hidden_states[lm_head_indices]
        logits, speculative_logits = self.text_model.lm_head(hidden_states)

        return logits, speculative_logits
