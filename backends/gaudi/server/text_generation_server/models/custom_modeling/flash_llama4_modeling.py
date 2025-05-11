# coding=utf-8
# Copyright 2025 The LLAMA4 and HuggingFace Inc. team. All rights reserved.
#
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

from typing import List, Optional, Tuple, Union

import torch
import math
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F

from transformers import Llama4TextConfig
from transformers.cache_utils import Cache
from transformers.activations import ACT2FN
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_outputs import (
    BaseModelOutput,
)

from transformers.modeling_attn_mask_utils import AttentionMaskConverter

from text_generation_server.layers import (
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    TensorParallelRowLinear,
    SpeculativeHead,
    FastLinear,
)
from text_generation_server.layers.layernorm import FastRMSNorm
from text_generation_server.layers.attention import (
    KVCache,
    paged_attention,
    Seqlen,
    HPUPagedAttentionMetadata,
)
from text_generation_server.models.custom_modeling.flash_llama_modeling import (
    FlashLlamaAttention,
    LlamaMLP,
)


def reshape_for_broadcast(freqs: torch.Tensor, target):
    ndim = len(target)
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(target)]
    return freqs.view(*shape)


def apply_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    freqs_ci: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    query_shape = query.shape
    key_shape = key.shape
    cos_emb, sin_emb = freqs_ci.split(1, dim=-1)

    if len(query.shape) == 3:
        query = query.unsqueeze(0)
        key = key.unsqueeze(0)

    query_reshaped = query.float().reshape(*query.shape[:-1], -1, 2)
    key_reshaped = key.float().reshape(*key.shape[:-1], -1, 2)
    q_shape = query_reshaped.shape[:-1]
    cos_emb = reshape_for_broadcast(cos_emb, q_shape)
    sin_emb = reshape_for_broadcast(sin_emb, q_shape)
    x_q, y_q = query_reshaped.unbind(-1)
    x_k, y_k = key_reshaped.unbind(-1)

    x_q_rot = x_q * cos_emb - y_q * sin_emb
    y_q_rot = x_q * sin_emb + y_q * cos_emb
    x_k_rot = x_k * cos_emb - y_k * sin_emb
    y_k_rot = x_k * sin_emb + y_k * cos_emb

    query_out = torch.stack([x_q_rot, y_q_rot], dim=-1).flatten(-2)
    key_out = torch.stack([x_k_rot, y_k_rot], dim=-1).flatten(-2)
    query_out = query_out.view(*query_shape)
    key_out = key_out.view(*key_shape)
    return query_out.type_as(query), key_out.type_as(key)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Llama4TextExperts(nn.Module):
    def __init__(self, prefix, config: Llama4TextConfig, weights):
        super().__init__()
        self.process_group = weights.process_group
        self.num_experts = config.num_local_experts
        self.intermediate_size = (
            config.intermediate_size // weights.process_group.size()
        )
        self.hidden_size = config.hidden_size
        self.expert_dim = self.intermediate_size
        self.gate_up_proj = nn.Parameter(
            weights.get_packed_sharded(f"{prefix}.gate_up_proj", dim=-1, block_sizes=2),
            requires_grad=False,
        )
        self.down_proj = nn.Parameter(
            weights.get_sharded(f"{prefix}.down_proj", dim=1), requires_grad=False
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        This should really not be run on a single machine, as we are reaching compute bound:
        - the inputs are expected to be "sorted" per expert already.
        - the weights are viewed with another dim, to match num_expert, 1, shape * num_tokens, shape

        Args:
            hidden_states (torch.Tensor): (batch_size * token_num, hidden_size)
            selected_experts (torch.Tensor): (batch_size * token_num, top_k)
            routing_weights (torch.Tensor): (batch_size * token_num, top_k)
        Returns:
            torch.Tensor
        """
        gate_up_proj = self.gate_up_proj.view(self.num_experts, -1, 2 * self.expert_dim)
        down_proj = self.down_proj.view(self.num_experts, self.expert_dim, -1)
        hidden_states = hidden_states.view(self.num_experts, -1, self.hidden_size)
        gate_up = torch.bmm(hidden_states, gate_up_proj)
        gate, up = gate_up.chunk(2, dim=-1)  # not supported for DTensors
        next_states = torch.bmm((up * self.act_fn(gate)), down_proj)
        next_states = next_states.view(-1, self.hidden_size)

        # Reduce sum
        if self.process_group.size() > 1:
            torch.distributed.all_reduce(next_states, group=self.process_group)

        return next_states


# Phi3MLP
class Llama4TextMLP(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = (
            config.intermediate_size // weights.process_group.size()
        )
        self.gate_up_proj = TensorParallelColumnLinear.load_multi(
            config=config,
            prefixes=[f"{prefix}.gate_proj", f"{prefix}.up_proj"],
            weights=weights,
            dim=0,
            bias=False,
        )
        self.down_proj = TensorParallelRowLinear.load(
            config=config,
            prefix=f"{prefix}.down_proj",
            weights=weights,
            bias=False,
        )

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        gate_up_states = self.gate_up_proj(x)
        gate_up_states = gate_up_states.view(-1, 2, self.intermediate_size)
        return self.down_proj(self.act_fn(gate_up_states[:, 0]) * gate_up_states[:, 1])


class Llama4TextL2Norm(torch.nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x)

    def extra_repr(self):
        return f"eps={self.eps}"


class Llama4TextMoe(nn.Module):
    def __init__(
        self,
        prefix,
        config,
        weights,
    ):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_local_experts
        self.experts = Llama4TextExperts(
            config=config, prefix=f"{prefix}.experts", weights=weights
        )
        self.router = FastLinear.load(
            config=config, prefix=f"{prefix}.router", weights=weights, bias=False
        )
        self.shared_expert = Llama4TextMLP(
            config=config, prefix=f"{prefix}.shared_expert", weights=weights
        )
        self.process_group = weights.process_group

    def forward(self, hidden_states, adapter_data):
        seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_dim)
        tokens_per_expert = hidden_states.shape[0]
        router_logits = self.router(hidden_states)

        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=1)
        router_scores = (
            torch.full_like(router_logits, float("-inf"))
            .scatter_(1, router_indices, router_top_value)
            .transpose(0, 1)
        )
        # We do this to make sure we have -inf for non topK tokens before going through the !
        # Here we are just creating a tensor to index each and every single one of the hidden states. Let s maybe register a buffer for this!
        router_indices = (
            torch.arange(tokens_per_expert, device=hidden_states.device)
            .view(1, -1)
            .expand(router_scores.size(0), -1)
        )
        router_scores = torch.sigmoid(router_scores.float()).to(hidden_states.dtype)

        router_indices = router_indices.reshape(-1, 1).expand(-1, self.hidden_dim)
        routed_in = torch.gather(
            input=hidden_states,
            dim=0,
            index=router_indices,
        ).to(hidden_states.device)

        # we gather inputs corresponding to each expert based on the router indices
        routed_in = routed_in * router_scores.reshape(-1, 1)
        routed_out = self.experts(routed_in)
        out = self.shared_expert(hidden_states)

        # now that we finished expert computation -> we scatter add because we gathered previously
        # we have to do this because we used all experts on all tokens. This is faster than the for loop, tho you are compute bound
        # this scales a lot better if you do EP!
        out.scatter_add_(
            dim=0, index=router_indices, src=routed_out.view(-1, self.hidden_dim)
        )
        return out


class Llama4TextRotaryEmbedding(nn.Module):
    def __init__(self, config: Llama4TextConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        self.rope_type = "llama3" if config.rope_scaling is not None else "default"

        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def forward(self, x, position_ids):
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        inv_freq_expanded = inv_freq_expanded.to(device_type)
        position_ids_expanded = position_ids_expanded.to(device_type)
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
            cos = torch.cos(freqs) * self.attention_scaling
            sin = torch.sin(freqs) * self.attention_scaling
            cos = cos.reshape(-1, 1, cos.shape[-1])
            sin = sin.reshape(-1, 1, sin.shape[-1])
            freqs_cis = torch.cat([cos, sin], dim=-1) * self.attention_scaling
            freqs_cis = torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1)
        return freqs_cis


class Llama4TextAttention(FlashLlamaAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, prefix, config, weights, layer_idx):
        super().__init__(layer_idx, prefix, config, weights)
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attn_scale = config.attn_scale
        self.floor_scale = config.floor_scale
        self.attn_temperature_tuning = config.attn_temperature_tuning
        self.attention_dropout = config.attention_dropout
        self.use_rope = int((layer_idx + 1) % 4 != 0)  # rope unused for dense layers

        if self.config.use_qk_norm and self.use_rope:
            self.qk_norm = Llama4TextL2Norm(config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_ci,
        cu_seqlen_prefill,
        kv_cache: KVCache,
        slots,
        seqlen,
        adapter_data,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        hpu_attention_meta: Optional[HPUPagedAttentionMetadata] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bs = seqlen.input_lengths.shape[0]
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        qkv = self.query_key_value(hidden_states, adapter_data)
        query_states, key_states, value_states = qkv.split(
            [
                self.head_dim * self.num_heads,
                self.head_dim * self.num_key_value_heads,
                self.head_dim * self.num_key_value_heads,
            ],
            dim=-1,
        )

        query_states = query_states.view(hidden_shape)
        key_states = key_states.view(hidden_shape)
        value_states = value_states.view(hidden_shape)

        if self.use_rope:  # the 16E model skips rope for long context on certain layers
            query_states, key_states = apply_rotary_emb(
                query_states, key_states, freqs_ci
            )

        if hasattr(self, "qk_norm"):  # the 128E model does not use qk_norm
            query_states = self.qk_norm(query_states)
            key_states = self.qk_norm(key_states)

        kv_cache.store(
            key=key_states,
            value=value_states,
            slots=slots,
            kv_scales=self.kv_scales,
        )

        # Use temperature tuning from https://arxiv.org/abs/2501.19399) to NoROPE layers
        if self.attn_temperature_tuning and not self.use_rope:
            attn_scales = (
                torch.log(
                    torch.floor((position_ids.float() + 1.0) / self.floor_scale) + 1.0
                )
                * self.attn_scale
                + 1.0
            )
            attn_scales = attn_scales.view(*input_shape, 1, 1)
            query_states = (query_states * attn_scales).to(query_states.dtype)

        # Prefill
        if cu_seqlen_prefill is not None:
            # sdpa
            query = query_states.view(bs, -1, self.num_heads, self.head_dim).transpose(
                1, 2
            )
            key = key_states.view(
                bs, -1, self.num_key_value_heads, self.head_dim
            ).transpose(1, 2)
            value = value_states.view(
                bs, -1, self.num_key_value_heads, self.head_dim
            ).transpose(1, 2)
            key = repeat_kv(key, self.num_key_value_groups)
            value = repeat_kv(value, self.num_key_value_groups)

            causal_mask = attention_mask
            if attention_mask is not None and causal_mask.ndim == 4:
                causal_mask = causal_mask[:, :, :, : key.shape[-2]]
            is_causal = query.shape[2] > 1 and causal_mask is None
            # SDPA with memory-efficient backend is bugged with non-contiguous inputs and custom attn_mask for some torch versions
            # Reference: https://github.com/pytorch/pytorch/issues/112577.
            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()

            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=causal_mask,
                dropout_p=0,
                scale=self.scaling,
                is_causal=is_causal,
            )
            attn_output = attn_output.transpose(1, 2).contiguous()
        # Decode
        else:
            attn_output = paged_attention(
                query_states,
                kv_cache,
                self.kv_head_mapping,
                self.softmax_scale,
                seqlen,
                kv_scales=self.kv_scales,
                hpu_attention_meta=hpu_attention_meta,
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output, adapter_data)
        return attn_output


class Llama4TextDecoderLayer(nn.Module):
    def __init__(self, prefix, config, weights, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Llama4TextAttention(
            f"{prefix}.self_attn", config, weights, layer_idx
        )
        self.use_chunked_attention = int((layer_idx + 1) % 4 != 0)  # <=> use rope
        self.is_moe_layer = layer_idx in config.moe_layers
        if self.is_moe_layer:  # the 128E model interleaves dense / sparse
            self.feed_forward = Llama4TextMoe(f"{prefix}.feed_forward", config, weights)
        else:
            self.feed_forward = LlamaMLP(f"{prefix}.feed_forward", config, weights)

        self.input_layernorm = FastRMSNorm.load(
            prefix=f"{prefix}.input_layernorm",
            weights=weights,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = FastRMSNorm.load(
            prefix=f"{prefix}.post_attention_layernorm",
            weights=weights,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        hidden_states,
        freqs_ci,
        cu_seqlen_prefill,
        kv_cache,
        slots,
        seqlen,
        adapter_data,
        attention_mask: Optional[torch.Tensor] = None,
        chunk_causal_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        hpu_attention_meta: Optional[HPUPagedAttentionMetadata] = None,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        residual = hidden_states
        hidden_states, _ = self.input_layernorm(hidden_states)

        # use local attention mask for ROPE layers
        if self.use_chunked_attention and chunk_causal_mask is not None:
            attention_mask = chunk_causal_mask

        attention_states = self.self_attn(
            hidden_states,
            freqs_ci,
            cu_seqlen_prefill,
            kv_cache,
            slots,
            seqlen,
            adapter_data,
            attention_mask=attention_mask,
            position_ids=position_ids,
            hpu_attention_meta=hpu_attention_meta,
        )

        hidden_states = residual + attention_states

        # Fully Connected
        residual = hidden_states

        hidden_states, _ = self.post_attention_layernorm(hidden_states)
        hidden_states = self.feed_forward(hidden_states, adapter_data)
        hidden_states = residual + hidden_states.view(residual.shape)
        return hidden_states


class Llama4TextModel(nn.Module):

    def __init__(self, prefix, config, weights):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = TensorParallelEmbedding(
            prefix=f"{prefix}.embed_tokens", weights=weights
        )
        self.layers = nn.ModuleList(
            [
                Llama4TextDecoderLayer(
                    prefix=f"{prefix}.layers.{layer_idx}",
                    config=config,
                    weights=weights,
                    layer_idx=layer_idx,
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        # self.norm = Llama4TextRMSNorm(prefix=f"{prefix}.norm", config=config, weights=weights)
        self.norm = FastRMSNorm.load(
            prefix=f"{prefix}.norm",
            weights=weights,
            eps=config.rms_norm_eps,
        )

        self.rotary_emb = Llama4TextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlen_prefill: Optional[torch.Tensor],
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        slots: torch.Tensor,
        seqlen: Seqlen,
        adapter_data,
        hpu_attention_meta: Optional[HPUPagedAttentionMetadata],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        hidden_states = inputs_embeds
        bs = seqlen.input_lengths.shape[0]
        seq_len = inputs_embeds.shape[0] / bs
        cache_position = torch.arange(0, seq_len, device=inputs_embeds.device)

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask, chunk_causal_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds.view(bs, int(seq_len), -1),
            cache_position,
            None,
            output_attentions=False,
            use_cache=False,
        )

        freqs_ci = self.rotary_emb(hidden_states, position_ids.view(bs, -1))

        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                freqs_ci,
                cu_seqlen_prefill,
                kv_cache[i],
                slots,
                seqlen,
                adapter_data,
                attention_mask=causal_mask,
                chunk_causal_mask=chunk_causal_mask,
                position_ids=position_ids,
                hpu_attention_meta=hpu_attention_meta,
            )

        hidden_states, _ = self.norm(hidden_states)

        return hidden_states

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
        chunked_attention_mask=None,
        use_cache=True,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return (
                    attention_mask,
                    attention_mask,
                )  # flash does not support chunked attn TODO support flash
            return None, None

        if self.config._attn_implementation not in ["sdpa", "flex_attention", "eager"]:
            return None, None

        sequence_length = input_tensor.shape[1]
        attention_chunk_size = self.config.attention_chunk_size

        first_cache_position = cache_position[0]

        if past_key_values is not None:
            full_cache_length = past_key_values.get_max_cache_shape() or sequence_length
        else:
            full_cache_length = (
                attention_mask.shape[-1]
                if attention_mask is not None
                else sequence_length
            )

        cond1 = first_cache_position >= attention_chunk_size
        cond2 = (first_cache_position < attention_chunk_size) & (
            first_cache_position + sequence_length > attention_chunk_size
        )
        key_length = (
            torch.where(
                cond1,
                attention_chunk_size + sequence_length - 1,
                torch.where(
                    cond2, first_cache_position + sequence_length, attention_chunk_size
                ),
            )
            if use_cache
            else full_cache_length
        )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        dtype, device = input_tensor.dtype, input_tensor.device
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=max(full_cache_length, attention_chunk_size),
            dtype=dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
            device=device,
        )
        if full_cache_length > self.config.attention_chunk_size:
            start_idx = max(first_cache_position - attention_chunk_size + 1, 0)
            end_idx = start_idx + key_length
            chunked_attention_mask = self.create_chunked_attention_mask(
                self.config.attention_chunk_size,
                start=start_idx,  # same offset as with flex
                end=end_idx,
                device=device,
            )

            local_attention_mask = attention_mask[
                :, start_idx:end_idx
            ]  # offset here as well
            # It may be smaller than attention_chunk_size -> pad it
            requires_padding = local_attention_mask.shape[-1] < attention_chunk_size
            if requires_padding:
                local_attention_mask = nn.functional.pad(
                    local_attention_mask,
                    (0, attention_chunk_size - local_attention_mask.shape[-1]),
                )
            # Depending on the padding, take the query tokens from the end or the cache_position
            if not requires_padding:
                chunked_attention_mask = chunked_attention_mask[
                    None, None, -sequence_length:, :
                ]
            else:
                chunked_attention_mask = chunked_attention_mask[
                    None, None, cache_position, :
                ]

            chunked_attention_mask = chunked_attention_mask.expand(
                input_tensor.shape[0], -1, -1, -1
            )
            chunked_attention_mask = (
                chunked_attention_mask * local_attention_mask[:, None, None, :]
            )
            if self.config._attn_implementation == "eager":
                min_dtype = torch.finfo(dtype).min
                chunked_attention_mask = torch.where(
                    chunked_attention_mask == 0, min_dtype, 0.0
                ).to(dtype)

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu", "npu"]
            and attention_mask.ndim == 4
            and not output_attentions  # Only unmask for 4d masks
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(
                causal_mask, min_dtype
            )

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if (
            self.config._attn_implementation == "sdpa"
            and chunked_attention_mask is not None
        ):
            chunked_attention_mask = chunked_attention_mask.bool()
            causal_mask = causal_mask.bool()
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=first_cache_position,
                is_training=self.training,
            ):
                causal_mask = None
        return causal_mask, chunked_attention_mask

    def create_chunked_attention_mask(
        self, attention_chunk_size: int, start: int, end: int, device: torch.device
    ) -> torch.Tensor:
        """
        Generate the following:

        'What'      :  0 ■ ⬚ ⬚ ⬚ ⬚ ⬚    |
        '▁is'       :  1 ■ ■ ⬚ ⬚ ⬚ ⬚     |
        '▁ch'       :  2 ■ ■ ■ ⬚ ⬚ ⬚     |
        'unked'     :  3 ⬚ ⬚ ⬚ ■ ⬚ ⬚    |
        '▁attention':  4 ⬚ ⬚ ⬚ ■ ■ ⬚    |
        '?'         :  5 ⬚ ⬚ ⬚ ■ ■ ■     |

        If the chunk size is 3.
        This can just be applied over the already created attention mask
        """
        arange_vector = torch.arange(start, end, device=device)
        block_pos = torch.abs(
            arange_vector.unsqueeze(0) // attention_chunk_size
            - arange_vector.unsqueeze(1) // attention_chunk_size
        )
        token_pos = arange_vector.unsqueeze(0) - arange_vector.unsqueeze(1)
        mask = (block_pos == 0) & (token_pos <= 0)
        return mask.to(device)

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to place the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length),
                fill_value=min_dtype,
                dtype=dtype,
                device=device,
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(
                target_length, device=device
            ) > cache_position.to(device).reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = (
                    causal_mask.clone()
                )  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[
                    :, None, None, :
                ].to(device)
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[
                    :, :, :, :mask_length
                ].masked_fill(padding_mask, min_dtype)

        return causal_mask


class Llama4ForCausalLM(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        self.model = Llama4TextModel(
            prefix=f"{prefix}.model", config=config, weights=weights
        )
        self.vocab_size = config.vocab_size
        self.lm_head = SpeculativeHead.load(
            config,
            f"{prefix}.lm_head",
            weights,
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
        adapter_data: Optional[torch.Tensor] = None,
        lm_head_indices: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        hidden_states = self.model(
            inputs_embeds,
            position_ids,
            cu_seqlen_prefill,
            kv_cache,
            slots,
            seqlen,
            adapter_data=adapter_data,
            hpu_attention_meta=hpu_attention_meta,
            attention_mask=attention_mask,
        )

        if lm_head_indices is not None:
            hidden_states = hidden_states[lm_head_indices]

        logits, speculative_logits = self.lm_head(hidden_states)
        return logits, speculative_logits


class Llama4VisionMLP2(torch.nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.fc1 = TensorParallelColumnLinear.load(
            config=config, prefix=f"{prefix}.fc1", weights=weights, bias=False
        )
        self.fc2 = TensorParallelRowLinear.load(
            config=config, prefix=f"{prefix}.fc2", weights=weights, bias=False
        )
        self.activation_fn = nn.GELU()  # ACT2FN[config.hidden_act]
        self.dropout = config.projector_dropout

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        return self.activation_fn(
            hidden_states
        )  # TODO: check if we need to apply activation again


class Llama4MultiModalProjector(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        self.linear_1 = FastLinear.load(
            config=config, prefix=f"{prefix}.linear_1", weights=weights, bias=False
        )

    def forward(self, image_features):
        hidden_states = self.linear_1(image_features)
        return hidden_states


def pixel_shuffle(input_tensor, shuffle_ratio):
    # input_tensor: [batch_size, num_patches, channels]
    batch_size, num_patches, channels = input_tensor.shape
    patch_size = int(math.sqrt(num_patches))

    input_tensor = input_tensor.view(batch_size, patch_size, patch_size, -1)
    batch_size, height, width, channels = input_tensor.size()
    reshaped_tensor = input_tensor.view(
        batch_size, height, int(width * shuffle_ratio), int(channels / shuffle_ratio)
    )
    reshaped_tensor = reshaped_tensor.permute(0, 2, 1, 3).contiguous()
    reshaped_tensor = reshaped_tensor.view(
        batch_size,
        int(height * shuffle_ratio),
        int(width * shuffle_ratio),
        int(channels / (shuffle_ratio**2)),
    )
    reshaped_tensor = reshaped_tensor.permute(0, 2, 1, 3).contiguous()

    output_tensor = reshaped_tensor.view(batch_size, -1, reshaped_tensor.shape[-1])
    return output_tensor


class Llama4VisionPixelShuffleMLP(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        self.pixel_shuffle_ratio = config.pixel_shuffle_ratio
        self.inner_dim = int(
            config.projector_input_dim // (self.pixel_shuffle_ratio**2)
        )
        self.output_dim = config.projector_output_dim
        self.mlp = Llama4VisionMLP2(
            prefix=f"{prefix}.mlp", config=config, weights=weights
        )

    def forward(self, encoded_patches: torch.Tensor) -> torch.Tensor:
        encoded_patches = pixel_shuffle(encoded_patches, self.pixel_shuffle_ratio)
        return self.mlp(encoded_patches)


# TODO there is a different RoPE for vision encoder, defined as below
def vision_reshape_for_broadcast(freqs_ci: torch.Tensor, query: torch.Tensor):
    ndim = query.ndim
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(query.shape)]
    return freqs_ci.view(*shape)


class Llama4VisionAttention(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads // weights.process_group.size()
        self.progress_group = weights.process_group

        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_groups = 1
        self.attention_dropout = config.attention_dropout
        self.qkv_proj = TensorParallelColumnLinear.load_multi(
            config,
            prefixes=[f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"],
            dim=0,
            weights=weights,
            bias=True,
        )
        self.o_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.o_proj",
            weights=weights,
            bias=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_ci: torch.Tensor,  # Now takes (cos_theta, sin_theta) instead of complex
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        qkv = self.qkv_proj(hidden_states)

        query_states, key_states, value_states = qkv.split(
            [
                self.head_dim * self.num_heads,
                self.head_dim * self.num_heads,
                self.head_dim * self.num_heads,
            ],
            dim=2,
        )
        query_states = query_states.view(hidden_shape)
        key_states = key_states.view(hidden_shape)
        value_states = value_states.view(hidden_shape)

        query_states, key_states = apply_rotary_emb(
            query_states, key_states, freqs_ci=freqs_ci
        )

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            is_causal=False,
            dropout_p=0,
        )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output


class Llama4VisionMLP(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        self.config = config
        self.activation_fn = nn.GELU()  # ACT2FN[config.hidden_act]
        self.fc1 = TensorParallelColumnLinear.load(
            prefix=f"{prefix}.fc1", weights=weights, config=config, bias=True
        )
        self.fc2 = TensorParallelRowLinear.load(
            prefix=f"{prefix}.fc2", weights=weights, config=config, bias=True
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class Llama4VisionEncoderLayer(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Llama4VisionAttention(
            prefix=f"{prefix}.self_attn", config=config, weights=weights
        )
        self.mlp = Llama4VisionMLP(
            prefix=f"{prefix}.mlp", config=config, weights=weights
        )

        self.input_layernorm = nn.LayerNorm.load(
            prefix=f"{prefix}.input_layernorm", weights=weights, eps=1e-05
        )
        self.post_attention_layernorm = nn.LayerNorm.load(
            prefix=f"{prefix}.post_attention_layernorm", weights=weights, eps=1e-05
        )

    def forward(
        self,
        hidden_state: torch.Tensor,
        freqs_ci: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        # Self Attention
        residual = hidden_state

        hidden_state = self.input_layernorm(hidden_state)

        hidden_state = self.self_attn(
            hidden_state,
            freqs_ci=freqs_ci,
            attention_mask=attention_mask,
        )

        hidden_state = residual + hidden_state

        # Feed forward
        residual = hidden_state
        hidden_state = self.post_attention_layernorm(hidden_state)
        hidden_state = self.mlp(hidden_state)
        hidden_state = residual + hidden_state
        outputs = (hidden_state,)
        return outputs


class Llama4VisionEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`Llama4VisionEncoderLayer`].

    Args:
        config: Llama4VisionConfig
    """

    def __init__(self, prefix, config, weights):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [
                Llama4VisionEncoderLayer(
                    prefix=f"{prefix}.layers.{layer_id}", config=config, weights=weights
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.gradient_checkpointing = False
        self.config = config

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_ci: torch.Tensor,  # TODO move this to an attribute instead of keeping it around
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutput]:

        for encoder_layer in self.layers:
            layer_outputs = encoder_layer(
                hidden_state=hidden_states,
                attention_mask=attention_mask,
                freqs_ci=freqs_ci,
            )

            hidden_states = layer_outputs[0]

        return hidden_states


class Llama4UnfoldConvolution(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        kernel_size = config.patch_size
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.unfold = torch.nn.Unfold(kernel_size=kernel_size, stride=config.patch_size)
        self.linear = FastLinear.load(
            config=config, prefix=f"{prefix}.linear", weights=weights, bias=False
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.unfold(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = self.linear(hidden_states)
        return hidden_states


class Llama4VisionRotaryEmbedding(nn.Module):
    def __init__(self, config, weights):
        super().__init__()
        # Calculate image grid indices
        idx = config.image_size // config.patch_size
        img_idx = torch.arange(
            idx**2, dtype=torch.int32, device=weights.device
        ).reshape(idx**2, 1)
        img_idx = torch.cat([img_idx, img_idx[:1]], dim=0)

        img_idx[-1, -1] = -2  # ID_CLS_TOKEN
        # Calculate x and y coordinates
        frequencies_x = img_idx % idx  # x coordinates
        frequencies_y = torch.div(img_idx, idx, rounding_mode="floor")  # y coordinates
        # Calculate frequency components
        freq_dim = config.hidden_size // config.num_attention_heads // 2
        rope_freq = 1.0 / (
            config.rope_theta
            ** (
                torch.arange(0, freq_dim, 2, device=weights.device)[
                    : (freq_dim // 2)
                ].float()
                / freq_dim
            )
        )

        # Compute frequencies for x and y directions
        freqs_x = (frequencies_x + 1)[..., None] * rope_freq[None, None, :]
        freqs_x = freqs_x.repeat_interleave(2, dim=-1)
        freqs_y = (frequencies_y + 1)[..., None] * rope_freq[None, None, :]
        freqs_y = freqs_y.repeat_interleave(2, dim=-1)

        # Combine frequencies and mask special tokens
        freqs = torch.cat([freqs_x, freqs_y], dim=-1).float().contiguous()[..., ::2]
        freqs = freqs.masked_fill(img_idx.reshape(-1, 1, 1) < 0, 0)

        freq_cis = torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1)
        self.freqs_ci = freq_cis  # idx**2, idx**2, idx * 2

    def forward(self, hidden_states):
        """
        Returns the rotary embedding components (cosθ, sinθ) for the given hidden states
        """
        return self.freqs_ci.to(dtype=hidden_states.dtype, device=hidden_states.device)


class Llama4VisionModel(nn.Module):

    def __init__(self, prefix, config, weights):
        super().__init__()
        self.config = config
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.hidden_size = config.hidden_size
        self.num_channels = config.num_channels

        self.num_patches = (self.image_size // self.patch_size) ** 2 + 1
        self.scale = config.hidden_size**-0.5

        self.patch_embedding = Llama4UnfoldConvolution(
            prefix=f"{prefix}.patch_embedding", config=config, weights=weights
        )

        self.class_embedding = nn.Parameter(
            weights.get_tensor(f"{prefix}.class_embedding"), requires_grad=False
        )

        self.positional_embedding_vlm = nn.Parameter(
            weights.get_tensor(f"{prefix}.positional_embedding_vlm"),
            requires_grad=False,
        )

        self.rotary_embedding = Llama4VisionRotaryEmbedding(config, weights)

        # layer norms
        self.layernorm_pre = nn.LayerNorm.load(
            prefix=f"{prefix}.layernorm_pre", weights=weights, eps=config.norm_eps
        )
        self.layernorm_post = nn.LayerNorm.load(
            prefix=f"{prefix}.layernorm_post", weights=weights, eps=config.norm_eps
        )

        # encoders
        self.model = Llama4VisionEncoder(
            prefix=f"{prefix}.model", config=config, weights=weights
        )
        self.vision_adapter = Llama4VisionPixelShuffleMLP(
            prefix=f"{prefix}.vision_adapter", config=config, weights=weights
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ):
        # num_concurrent_media and num_chunks are both currently 1
        batch_size_times_num_tiles, num_channels, height, width = pixel_values.shape
        num_concurrent_media = 1
        num_chunks = 1
        hidden_state = self.patch_embedding(pixel_values)
        _, num_patches, hidden_dim = hidden_state.shape

        # Add cls token
        hidden_state = hidden_state.reshape(
            batch_size_times_num_tiles * num_concurrent_media * num_chunks,
            num_patches,
            hidden_dim,
        )
        class_embedding = self.class_embedding.expand(
            hidden_state.shape[0], 1, hidden_state.shape[-1]
        )
        hidden_state = torch.cat([hidden_state, class_embedding], dim=1)
        num_patches += 1

        # Position embeddings
        hidden_state = hidden_state.reshape(
            batch_size_times_num_tiles * num_concurrent_media,
            num_chunks,
            num_patches,
            hidden_dim,
        )
        positional_embedding = self.positional_embedding_vlm.to(
            dtype=hidden_state.dtype, device=hidden_state.device
        )
        hidden_state = hidden_state + positional_embedding
        hidden_state = self.layernorm_pre(hidden_state)
        hidden_state = hidden_state.view(batch_size_times_num_tiles, -1, hidden_dim)
        freqs_ci = self.rotary_embedding(pixel_values)

        hidden_state = self.model(
            hidden_state,
            attention_mask=None,
            freqs_ci=freqs_ci,
        )

        hidden_state = self.layernorm_post(hidden_state)

        hidden_state = hidden_state[:, :-1, :]

        # now, we use Llama4VisionPixelShuffle + mlp to project embeddings
        hidden_state = self.vision_adapter(hidden_state)
        return hidden_state


class Llama4ForConditionalGeneration(nn.Module):

    def __init__(self, prefix: str, config, weights):
        super().__init__()
        self.config = config
        config.vision_config.quantize = None
        config.vision_config.speculator = config.speculator
        config.text_config.quantize = config.quantize
        config.text_config.speculator = config.speculator
        config.text_config._attn_implementation = None

        self.vision_model = Llama4VisionModel(
            prefix="vision_model", config=config.vision_config, weights=weights
        )

        self.multi_modal_projector = Llama4MultiModalProjector(
            prefix="multi_modal_projector", config=config, weights=weights
        )

        self.text_model = Llama4ForCausalLM(
            prefix="language_model", config=config.text_config, weights=weights
        )
        self.vocab_size = config.text_config.vocab_size
        self.pad_token_id = (
            self.config.pad_token_id if self.config.pad_token_id is not None else -1
        )
        self.config = config
        self.dtype = weights.dtype
        self.device = weights.device

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        vision_feature_layer: Union[int, List[int]],
        vision_feature_select_strategy: str,
        **kwargs,
    ):
        """
        Obtains image last hidden states from the vision tower and apply al projection.

        Args:
            pixel_values (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`)
               The tensors corresponding to the input images.
            vision_feature_layer (`Union[int, List[int]]`):
                The index of the layer to select the vision feature. If multiple indices are provided,
                the vision feature of the corresponding indices will be concatenated to form the
                vision features.
            vision_feature_select_strategy (`str`):
                The feature selection strategy used to select the vision feature from the vision backbone.
                Can be one of `"default"` or `"full"`
        Returns:
            image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
        """
        if vision_feature_select_strategy not in ["default", "full"]:
            raise ValueError(
                f"Unexpected select feature strategy: {self.vision_feature_select_strategy}"
            )
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        hidden_state = self.vision_model(pixel_values)
        return hidden_state

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        pixel_attention_mask=None,
        position_ids: Optional[torch.LongTensor] = None,
        cu_seqlen_prefill: Optional[torch.Tensor] = None,
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]] = None,
        slots: torch.Tensor = None,
        seqlen: Seqlen = None,
        hpu_attention_meta: Optional[HPUPagedAttentionMetadata] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[Union[int, List[int]]] = None,
        vision_feature_select_strategy: Optional[str] = None,
        image_sizes: torch.Tensor = None,
        lm_head_indices: Optional[torch.Tensor] = None,
        adapter_data: Optional[torch.Tensor] = None,
        **lm_kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        def _get_padding_mask(input_ids, pad_token_id=0):
            return (input_ids != pad_token_id).long()

        attention_mask = _get_padding_mask(input_ids)
        attention_mask = attention_mask.view(seqlen.input_lengths.shape[0], -1)
        inputs_embeds = self.text_model.model.embed_tokens(input_ids)
        vision_feature_layer = (
            vision_feature_layer
            if vision_feature_layer is not None
            else self.config.vision_config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_config.vision_feature_select_strategy
        )

        if pixel_values is not None:
            image_features = self.get_image_features(
                pixel_values=pixel_values,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
                image_sizes=image_sizes,
            )
            original_inputs_embeds_shape = inputs_embeds.shape

            vision_flat = image_features.view(-1, image_features.size(-1))
            projected_vision_flat = self.multi_modal_projector(vision_flat)

            special_image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1)
            final_mask = special_image_mask.to(inputs_embeds.device)
            inputs_embeds = inputs_embeds.view(-1, inputs_embeds.size(-1))

            final_mask_1d = final_mask[..., 0].reshape(-1)
            num_tokens_to_fill = final_mask_1d.sum()

            if num_tokens_to_fill != projected_vision_flat.size(0):
                raise ValueError(
                    f"Mismatch: final_mask wants {num_tokens_to_fill} embeddings, "
                    f"but multi_modal_projector returned {projected_vision_flat.size(0)}"
                )

            expanded_mask = final_mask_1d.unsqueeze(-1).expand(
                -1, inputs_embeds.size(-1)
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                expanded_mask, projected_vision_flat
            )
            inputs_embeds = inputs_embeds.view(original_inputs_embeds_shape)

        logits, speculative_logits = self.text_model(
            inputs_embeds,
            position_ids,
            cu_seqlen_prefill,
            kv_cache,
            slots,
            seqlen,
            hpu_attention_meta,
            adapter_data,
            lm_head_indices,
            attention_mask,
        )

        return logits, speculative_logits
