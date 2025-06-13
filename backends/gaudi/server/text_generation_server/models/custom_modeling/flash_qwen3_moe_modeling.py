# coding=utf-8
# Copyright 5 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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

from typing import List, Optional, Tuple, Type

import torch
from torch import nn
import torch.nn.functional as F
from text_generation_server.layers.attention import (
    attention,
    paged_attention,
    Seqlen,
    HPUPagedAttentionMetadata,
)
from text_generation_server.layers.attention.kv_cache import get_kv_scales
from text_generation_server.layers.moe import DenseMoELayer, MoELayer, SparseMoELayer
from text_generation_server.layers import (
    TensorParallelEmbedding,
    TensorParallelColumnLinear,
    TensorParallelRowLinear,
    SpeculativeHead,
    FastLinear,
)

from text_generation_server.layers.layernorm import (
    FastRMSNorm,
)
from .flash_qwen2_modeling import Qwen2MLP
from .flash_qwen3_modeling import Qwen3Attention
from transformers.activations import ACT2FN
from text_generation_server.layers.rotary import PositionRotaryEmbedding


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3MoeAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, prefix, weights, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = FastLinear.load(
            config, f"{prefix}.q_proj", weights, bias=config.attention_bias
        )

        self.k_proj = FastLinear.load(
            config, f"{prefix}.k_proj", weights, bias=config.attention_bias
        )
        self.v_proj = FastLinear.load(
            config, f"{prefix}.v_proj", weights, bias=config.attention_bias
        )
        self.o_proj = FastLinear.load(
            config, f"{prefix}.o_proj", weights, bias=config.attention_bias
        )

        self.rotary_emb = PositionRotaryEmbedding.static(
            config=config,
            dim=self.head_dim,
            base=config.rope_theta,
            device=weights.device,
        )

        self.q_norm = FastRMSNorm.load(
            prefix=f"{prefix}.q_norm",
            weights=weights,
            eps=config.rms_norm_eps,
        )

        self.k_norm = FastRMSNorm.load(
            prefix=f"{prefix}.k_norm",
            weights=weights,
            eps=config.rms_norm_eps,
        )

        self.max_past = (
            config.sliding_window if config.sliding_window is not None else -1
        )

        self.kv_scales = get_kv_scales(weights, f"{prefix}")
        self.kv_head_mapping = torch.arange(
            0, self.num_key_value_heads, dtype=torch.int32, device=weights.device
        ).repeat_interleave(self.num_key_value_groups)

        self.sliding_window = config.sliding_window
        if not (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            self.sliding_window = None

    def forward(
        self,
        hidden_states,
        cos,
        sin,
        cu_seqlen_prefill,
        kv_cache,
        slots,
        seqlen,
        hpu_attention_meta,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, _ = self.q_norm(self.q_proj(hidden_states).view(hidden_shape))
        key_states, _ = self.k_norm(self.k_proj(hidden_states).view(hidden_shape))
        value_states = self.v_proj(hidden_states).view(hidden_shape)

        self.rotary_emb(query_states, key_states, cos, sin)
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        kv_cache.store(
            key=key_states,
            value=value_states,
            slots=slots,
            kv_scales=self.kv_scales,
        )

        # Prefill
        if cu_seqlen_prefill is not None:
            # sdpa
            attn_output = attention(
                query=query_states,
                key=key_states,
                value=value_states,
                kv_cache=kv_cache,
                kv_scales=self.kv_scales,
                seqlen=seqlen,
                softmax_scale=self.scaling,
                window_size_left=self.max_past,
            )
        # Decode
        else:
            attn_output = paged_attention(
                query_states,
                kv_cache,
                self.kv_head_mapping,
                self.scaling,
                seqlen,
                kv_scales=self.kv_scales,
                hpu_attention_meta=hpu_attention_meta,
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output


class Qwen3MoE(nn.Module):
    def __init__(self, prefix, config, moe_layer_cls: Type[MoELayer], weights):
        super().__init__()

        # gating
        self.gate = FastLinear.load(config, f"{prefix}.gate", weights, bias=False)

        self.moe = moe_layer_cls(
            n_expert_group=None,
            n_experts=config.num_experts,
            prefix=f"{prefix}.experts",
            renormalize=True,
            topk=config.num_experts_per_tok,
            topk_group=None,
            weights=weights,
        )
        # gate_proj_name="w1",
        # up_proj_name="w3",
        # down_proj_name="w2",

        assert isinstance(self.moe, MoELayer)

        self.process_group = weights.process_group

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        router_logits = self.gate(x)
        out = self.moe(x, gating_output=router_logits)

        # Reduce sum
        if self.process_group.size() > 1:
            torch.distributed.all_reduce(out, group=self.process_group)

        return out.view(*x.shape)


class Qwen3MoeMLP(nn.Module):
    def __init__(self, prefix, config, weights, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = (
            intermediate_size
            if intermediate_size is not None
            else config.intermediate_size
        )
        # Fuse gate and up proj
        self.gate_up_proj = TensorParallelColumnLinear.load_multi(
            config,
            prefixes=[f"{prefix}.gate_proj", f"{prefix}.up_proj"],
            weights=weights,
            dim=0,
            bias=False,
        )
        self.down_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.down_proj",
            weights=weights,
            bias=False,
        )
        self.intermediate_size = (
            config.intermediate_size // weights.process_group.size()
        )

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        gate_up_states = self.gate_up_proj(x)
        gate_up_states = gate_up_states.view(-1, 2, self.intermediate_size)
        return self.down_proj(self.act(gate_up_states[:, 0]) * gate_up_states[:, 1])


class Qwen3MoeSparseMoeBlock(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        # gating
        # self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.gate = FastLinear.load(config, f"{prefix}.gate", weights, bias=False)
        self.experts = nn.ModuleList(
            [
                Qwen3MoeMLP(
                    prefix=f"{prefix}.experts.{i}",
                    config=config,
                    weights=weights,
                    intermediate_size=config.moe_intermediate_size,
                )
                for i in range(self.num_experts)
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        input_shape = hidden_states.shape
        _, hidden_dim = hidden_states.shape
        # hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=hidden_states.dtype)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (input_shape), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts
        ).permute(2, 1, 0)
        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = (
                expert_layer(current_state) * routing_weights[top_x, idx, None]
            )

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states.dtype)
            )
        final_hidden_states = final_hidden_states.reshape(input_shape)
        return final_hidden_states


class Qwen3MoeDecoderLayer(nn.Module):
    def __init__(self, config, prefix, weights, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        if config.num_key_value_heads // weights.process_group.size() > 0:
            self.self_attn = Qwen3Attention(
                config,
                prefix=f"{prefix}.self_attn",
                weights=weights,
                layer_idx=layer_idx,
            )
        else:
            self.self_attn = Qwen3MoeAttention(
                config,
                prefix=f"{prefix}.self_attn",
                weights=weights,
                layer_idx=layer_idx,
            )

        moe_layer_cls = (
            SparseMoELayer if SparseMoELayer.is_supported(weights) else DenseMoELayer
        )

        if (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = Qwen3MoE(f"{prefix}.mlp", config, moe_layer_cls, weights)
            # self.mlp = Qwen3MoeSparseMoeBlock(f"{prefix}.mlp", config, weights)

        else:
            self.mlp = Qwen2MLP(config=config, prefix=f"{prefix}.mlp", weights=weights)

        self.input_layernorm = FastRMSNorm.load(
            prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = FastRMSNorm.load(
            prefix=f"{prefix}.post_attention_layernorm",
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
        hpu_attention_meta,
    ) -> torch.Tensor:
        if residual is None:
            residual = hidden_states

        hidden_states, _ = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states,
            cos,
            sin,
            cu_seqlen_prefill,
            kv_cache,
            slots,
            seqlen,
            hpu_attention_meta,
        )

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states, _ = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states
        return hidden_states


class Qwen3MoeModel(nn.Module):
    def __init__(self, config, prefix: str, weights):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.layers = nn.ModuleList(
            [
                Qwen3MoeDecoderLayer(
                    config=config,
                    prefix=f"{prefix}.layers.{layer_idx}",
                    weights=weights,
                    layer_idx=layer_idx,
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = FastRMSNorm.load(
            prefix=f"{prefix}.norm", weights=weights, eps=config.rms_norm_eps
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlen_prefill: Optional[torch.Tensor],
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        slots: torch.Tensor,
        seqlen: Seqlen,
        hpu_attention_meta: Optional[HPUPagedAttentionMetadata],
    ) -> torch.Tensor:

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        cos, sin = self.layers[0].self_attn.rotary_emb.get_cos_sin(
            position_ids,
        )

        residual = None
        for i, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                residual,
                cos,
                sin,
                cu_seqlen_prefill,
                kv_cache[i],
                slots,
                seqlen,
                hpu_attention_meta,
            )

        hidden_states, _ = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        return hidden_states


class Qwen3MoeForCausalLM(nn.Module):

    def __init__(self, prefix: str, config, weights):
        super().__init__()
        self.model = Qwen3MoeModel(config=config, prefix="model", weights=weights)
        self.vocab_size = config.vocab_size
        if config.tie_word_embeddings:
            suffix = "model.embed_tokens"
        else:
            suffix = "lm_head"

        self.lm_head = SpeculativeHead.load(
            config,
            prefix=f"{prefix}.{suffix}" if prefix else suffix,
            weights=weights,
        )

        self.embed_tokens = TensorParallelEmbedding(
            prefix=f"{prefix}.embed_tokens" if prefix else "model.embed_tokens",
            weights=weights,
        )

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
    ) -> torch.Tensor:

        inputs_embeds = self.embed_tokens(input_ids)
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        hidden_states = self.model(
            inputs_embeds,
            position_ids,
            cu_seqlen_prefill,
            kv_cache,
            slots,
            seqlen,
            hpu_attention_meta,
        )

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        if lm_head_indices is not None:
            hidden_states = hidden_states[lm_head_indices]
        logits = self.lm_head(hidden_states)

        return logits
