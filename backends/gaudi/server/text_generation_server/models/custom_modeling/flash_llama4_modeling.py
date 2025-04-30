# coding=utf-8
# Copyright 2023, 2024 DeepSeek-AI and The HuggingFace Inc. team. All rights reserved.
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

# import math
# from dataclasses import dataclass
# from typing import Callable, List, Optional, Tuple, Union

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.utils.checkpoint

# from transformers.models.llama4.configuration_llama4 import Llama4VisionConfig

# from ...activations import ACT2FN
# from ...cache_utils import Cache, HybridChunkedCache
# from ...generation import GenerationMixin
# from ...integrations.hub_kernels import use_kernel_forward_from_hub
# from ...modeling_attn_mask_utils import AttentionMaskConverter
# from ...modeling_flash_attention_utils import FlashAttentionKwargs
# from ...modeling_outputs import (
#     BaseModelOutput,
#     BaseModelOutputWithPast,
#     CausalLMOutputWithPast,
#     ModelOutput,
# )
# from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
# from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
# from ...processing_utils import Unpack
# from ...utils import (
#     add_start_docstrings,
#     add_start_docstrings_to_model_forward,
#     is_torch_flex_attn_available,
#     logging,
#     replace_return_docstrings,
# )
# from .configuration_llama4 import Llama4Config, Llama4TextConfig


# if is_torch_flex_attn_available():
#     from torch.nn.attention.flex_attention import BlockMask

#     from ...integrations.flex_attention import make_flex_block_causal_mask

from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F

from transformers import Llama4TextConfig
from transformers.cache_utils import Cache, HybridChunkedCache
from transformers.activations import ACT2FN
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPast,
    ModelOutput,
)
from transformers.processing_utils import Unpack
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs

from text_generation_server.layers import (
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    TensorParallelRowLinear,
    SpeculativeHead,
    FastLinear,
    TensorParallelAdapterRowLinear
)
from text_generation_server.layers.layernorm import FastRMSNorm
from text_generation_server.layers.attention import (
    KVCache,
    get_kv_scales,
    paged_attention,
    attention,
    Seqlen,
    HPUPagedAttentionMetadata,
)
from text_generation_server.models.custom_modeling.flash_llama_modeling import (
    load_attention,
)
from habana_frameworks.torch.hpex.kernels import FusedSDPA
from vllm_hpu_extension.utils import ModuleFusedSDPA
from text_generation_server.utils.import_utils import (
    empty_cache,
    synchronize,
    get_free_memory,
)

from loguru import logger
from text_generation_server.utils.log import log_master

_CHECKPOINT_FOR_DOC = "meta-ai/Llama-4-17B"
_CONFIG_FOR_DOC = "Llama4Config"


class Llama4TextExperts(nn.Module):
    def __init__(self, prefix, config: Llama4TextConfig, weights):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.intermediate_size = config.intermediate_size
        self.hidden_size = config.hidden_size
        self.expert_dim = self.intermediate_size
        self.gate_up_proj = nn.Parameter(weights.get_sharded(f"{prefix}.gate_up_proj", dim=0), requires_grad=False)
        synchronize(weights.device)
        real_free_memory = get_free_memory(weights.device, 1)
        log_master(
            logger.debug,
            f"textExperts1 Free memory real: {real_free_memory / 1e9:.2f}GB"
        )


        self.down_proj = nn.Parameter(weights.get_sharded(f"{prefix}.down_proj", dim=0), requires_grad=False)
        synchronize(weights.device)
        real_free_memory = get_free_memory(weights.device, 1)
        log_master(
            logger.debug,
            f"textExperts2 Free memory real: {real_free_memory / 1e9:.2f}GB"
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
        hidden_states = hidden_states.view(self.num_experts, -1, self.hidden_size)
        gate_up = torch.bmm(hidden_states, self.gate_up_proj)
        gate, up = gate_up.chunk(2, dim=-1)  # not supported for DTensors
        next_states = torch.bmm((up * self.act_fn(gate)), self.down_proj)
        next_states = next_states.view(-1, self.hidden_size)
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
        shape = x.shape
        gate_up_states = self.gate_up_proj(x)
        gate_up_states = gate_up_states.view(*shape[:-1], 2, self.intermediate_size)
        result = self.down_proj(
            self.act_fn(gate_up_states[:, 0]) * gate_up_states[:, 1]
        )
        return result



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


class Llama4TextRMSNorm(nn.Module):
    def __init__(self, prefix, config, weights):
        """
        Llama4RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.eps = config.rms_norm_eps
        self.weight = nn.Parameter(weights.get_sharded(f"{prefix}.weight", dim=0), requires_grad=False)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class Llama4TextMoe(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_local_experts
        self.experts = Llama4TextExperts(config=config, prefix=f"{prefix}.experts", weights=weights)
        synchronize(weights.device)
        real_free_memory = get_free_memory(weights.device, 1)
        log_master(
            logger.debug,
            f"TextMode1 Free memory real: {real_free_memory / 1e9:.2f}GB"
        )


        self.router = FastLinear.load(config, f"{prefix}.router", weights, bias=False)
        synchronize(weights.device)
        real_free_memory = get_free_memory(weights.device, 1)
        log_master(
            logger.debug,
            f"TextMode2 Free memory real: {real_free_memory / 1e9:.2f}GB"
        )
        self.shared_expert = Llama4TextMLP(config=config, prefix=f"{prefix}.shared_expert", weights=weights)
        synchronize(weights.device)
        real_free_memory = get_free_memory(weights.device, 1)
        log_master(
            logger.debug,
            f"TextMode3 Free memory real: {real_free_memory / 1e9:.2f}GB"
        )

    def forward(self, hidden_states):
        batch, seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_dim)
        router_logits = self.router(hidden_states)
        tokens_per_expert = batch * seq_len

        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=1)
        router_scores = (
            torch.full_like(router_logits, float("-inf")).scatter_(1, router_indices, router_top_value).transpose(0, 1)
        )
        # We do this to make sure we have -inf for non topK tokens before going through the !
        # Here we are just creating a tensor to index each and every single one of the hidden states. Let s maybe register a buffer for this!
        router_indices = (
            torch.arange(tokens_per_expert, device=hidden_states.device).view(1, -1).expand(router_scores.size(0), -1)
        )
        router_scores = torch.sigmoid(router_scores.float()).to(hidden_states.dtype)

        router_indices = router_indices.reshape(-1, 1).expand(-1, hidden_dim)
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
        out.scatter_add_(dim=0, index=router_indices, src=routed_out.view(-1, hidden_dim))
        return out, router_scores

class Llama4TextRotaryEmbedding(nn.Module):
    def __init__(self, config: 'Llama4TextConfig', device=None):
        super().__init__()
        self.rope_type = "llama3" if config.rope_scaling is not None else "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch, seq_len, heads, dim]
            position_ids: Position indices of shape [batch, seq_len]
        Returns:
            Rotary embeddings as float tensors [batch, seq_len, heads, dim]
        """
        # Expand inv_freq and position_ids for broadcasting
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        # Compute frequencies (replaces complex phase)
        freqs = (inv_freq_expanded.to(x.device) @ position_ids_expanded).transpose(1, 2)  # [batch, seq_len, dim//2]
        
        # Generate cos/sin components directly (replaces torch.polar)
        cos_vals = torch.cos(freqs) * self.attention_scaling
        sin_vals = torch.sin(freqs) * self.attention_scaling
        
        # Interleave cos/sin values to match original complex format
        dim = x.size(-1)
        if dim % 2 != 0:
            raise ValueError(f"Feature dimension {dim} must be even for Rotary Embedding")
        
        # Stack and reshape to [batch, seq_len, dim] format
        freqs_cis = torch.stack([cos_vals, sin_vals], dim=-1)  # [batch, seq_len, dim//2, 2]
        freqs_cis = freqs_cis.reshape(*freqs_cis.shape[:-2], dim)  # [batch, seq_len, dim]
        
        return freqs_cis
    
# class Llama4TextRotaryEmbedding(nn.Module):
#     def __init__(self, config: Llama4TextConfig, device=None):
#         super().__init__()
#         # BC: "rope_type" was originally "type"
#         self.rope_type = "llama3" if config.rope_scaling is not None else "default"

#         self.max_seq_len_cached = config.max_position_embeddings
#         self.original_max_seq_len = config.max_position_embeddings

#         self.config = config
#         self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

#         inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
#         self.register_buffer("inv_freq", inv_freq, persistent=False)
#         self.original_inv_freq = self.inv_freq

#     @torch.no_grad()
#     @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
#     def forward(self, x, position_ids):
#         inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
#         position_ids_expanded = position_ids[:, None, :].float()

#         device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
#         with torch.autocast(device_type=device_type, enabled=False):  # Force float32
#             freqs = (inv_freq_expanded.to(x.device) @ position_ids_expanded).transpose(1, 2)
#             freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # Convert to complex representation
#             freqs_cis = freqs_cis * self.attention_scaling

#         return freqs_cis

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,  # Should be [cosθ, sinθ] instead of complex numbers
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embedding to query and key tensors using floating-point operations only.
    
    Args:
        xq: Query tensor of shape (batch, seq_len, n_heads, head_dim)
        xk: Key tensor of shape (batch, seq_len, n_heads, head_dim)
        freqs_cis: Precomputed rotation frequencies as [cosθ, sinθ] 
                  of shape (batch, seq_len, head_dim//2, 2)
    Returns:
        Rotated query and key tensors with same shape as input
    """
    # Verify head_dim is even
    assert xq.size(-1) % 2 == 0, "Feature dimension must be even for rotary embedding"
    
    # Reshape to separate real and imaginary components (pairs of adjacent elements)
    xq_reshaped = xq.float().reshape(*xq.shape[:-1], -1, 2)  # [..., head_dim//2, 2]
    xk_reshaped = xk.float().reshape(*xk.shape[:-1], -1, 2)  # [..., head_dim//2, 2]
    
    # Extract cosθ and sinθ (assuming freqs_cis is already in [cosθ, sinθ] format)
    cos_theta = freqs_cis[..., 0]  # [batch, seq_len, head_dim//2]
    sin_theta = freqs_cis[..., 1]  # [batch, seq_len, head_dim//2]
    
    # Expand dimensions for broadcasting [batch, seq_len, n_heads, head_dim//2]
    cos_theta = cos_theta.unsqueeze(2)  # Add n_heads dimension
    sin_theta = sin_theta.unsqueeze(2)
    
    # Rotary transformation (mathematically equivalent to complex multiplication)
    # xq_rotated = [xq0*cosθ - xq1*sinθ, xq0*sinθ + xq1*cosθ]
    xq_out = torch.stack([
        xq_reshaped[..., 0] * cos_theta - xq_reshaped[..., 1] * sin_theta,
        xq_reshaped[..., 0] * sin_theta + xq_reshaped[..., 1] * cos_theta
    ], dim=-1)
    
    xk_out = torch.stack([
        xk_reshaped[..., 0] * cos_theta - xk_reshaped[..., 1] * sin_theta,
        xk_reshaped[..., 0] * sin_theta + xk_reshaped[..., 1] * cos_theta
    ], dim=-1)
    
    # Restore original shape
    xq_out = xq_out.flatten(-2)  # [batch, seq_len, n_heads, head_dim]
    xk_out = xk_out.flatten(-2)
    
    # Maintain original dtype
    return xq_out.type_as(xq), xk_out.type_as(xk)

# def apply_rotary_emb(
#     xq: torch.Tensor,
#     xk: torch.Tensor,
#     freqs_cis: torch.Tensor,
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
#     xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
#     xq_out = torch.view_as_real(xq_ * freqs_cis[:, :, None, :]).flatten(3)
#     xk_out = torch.view_as_real(xk_ * freqs_cis[:, :, None, :]).flatten(3)
#     return xq_out.type_as(xq), xk_out.type_as(xk)


# def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
#     """
#     This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
#     num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
#     """
#     batch, num_key_value_heads, slen, head_dim = hidden_states.shape
#     if n_rep == 1:
#         return hidden_states
#     hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
#     return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# def eager_attention_forward(
#     module: nn.Module,
#     query: torch.Tensor,
#     key: torch.Tensor,
#     value: torch.Tensor,
#     attention_mask: Optional[torch.Tensor],
#     scaling: float,
#     dropout: float = 0.0,
#     **kwargs,
# ):
#     key_states = repeat_kv(key, module.num_key_value_groups)
#     value_states = repeat_kv(value, module.num_key_value_groups)
#     attn_weights = torch.matmul(query, key_states.transpose(2, 3)) / math.sqrt(module.head_dim)
#     if attention_mask is not None:
#         causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
#         attn_weights = attn_weights + causal_mask

#     attn_weights = nn.functional.softmax(attn_weights.float(), dim=-1).to(query.dtype)
#     attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
#     attn_output = torch.matmul(attn_weights, value_states)
#     attn_output = attn_output.transpose(1, 2).contiguous()

#     return attn_output, attn_weights


class Llama4TextAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, prefix, config, weights, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attn_scale = config.attn_scale
        self.floor_scale = config.floor_scale
        self.attn_temperature_tuning = config.attn_temperature_tuning
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.use_rope = int((layer_idx + 1) % 4 != 0)  # rope unused for dense layers
        
        # `config.attention_multiplier` is used in Granite
        self.softmax_scale = getattr(
            config, "attention_multiplier", self.head_dim**-0.5
        )

        if self.num_attention_heads % weights.process_group.size() != 0:
            raise ValueError(
                f"`num_attention_heads` must be divisible by `num_shards` (got `num_attention_heads`: {self.num_attention_heads} "
                f"and `num_shards`: {weights.process_group.size()}"
            )
        if config.num_key_value_heads % weights.process_group.size() != 0:
            raise ValueError(
                f"`num_key_value_heads` must be divisible by `num_shards` (got `num_key_value_heads`: {config.num_key_value_heads} "
                f"and `num_shards`: {weights.process_group.size()}"
            )
        self.num_heads = self.num_attention_heads // weights.process_group.size()
        self.num_key_value_heads = (
            config.num_key_value_heads // weights.process_group.size()
        )
        
        self.query_key_value = load_attention(config, prefix, weights, layer_idx)

        self.kv_scales = get_kv_scales(weights, f"{prefix}")

        o_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.o_proj",
            weights=weights,
            bias=getattr(config, "attention_bias", False),
        )

        self.o_proj = TensorParallelAdapterRowLinear.load(
            o_proj,
            layer_idx,
            "o_proj",
            process_group=weights.process_group,
        )

        self.num_groups = self.num_heads // self.num_key_value_heads
        self.kv_head_mapping = torch.arange(
            0, self.num_key_value_heads, dtype=torch.int32, device=weights.device
        ).repeat_interleave(self.num_groups)


        # self.q_proj = nn.Linear(
        #     config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        # )
        # self.k_proj = nn.Linear(
        #     config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        # )
        # self.v_proj = nn.Linear(
        #     config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        # )
        # self.o_proj = nn.Linear(
        #     config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        # )
        if self.config.use_qk_norm and self.use_rope:
            self.qk_norm = Llama4TextL2Norm(config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlen_prefill,
        kv_cache: KVCache,
        slots,
        seqlen,
        adapter_data,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        cache_position: Optional[torch.LongTensor] = None,
        hpu_attention_meta: Optional[HPUPagedAttentionMetadata] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        #hidden_shape = (*input_shape, -1, self.head_dim)
        qkv = self.query_key_value(hidden_states, adapter_data)
        query_states, key_states, value_states = qkv.split(
            [
                self.head_dim * self.num_heads,
                self.head_dim * self.num_key_value_heads,
                self.head_dim * self.num_key_value_heads,
            ],
            dim=-1,
        )
        
        query_states = query_states.view(-1, self.num_heads, self.head_dim)
        key_states = key_states.view(-1, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(-1, self.num_key_value_heads, self.head_dim)


        # query_states = self.q_proj(hidden_states).view(hidden_shape)
        # key_states = self.k_proj(hidden_states).view(*input_shape, -1, self.head_dim)
        # value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if self.use_rope:  # the 16E model skips rope for long context on certain layers
            query_states, key_states = apply_rotary_emb(
                query_states, key_states, position_embeddings.to(query_states.device)
            )

        if hasattr(self, "qk_norm"):  # the 128E model does not use qk_norm
            query_states = self.qk_norm(query_states)
            key_states = self.qk_norm(key_states)

        # Use temperature tuning from https://arxiv.org/abs/2501.19399) to NoROPE layers
        if self.attn_temperature_tuning and not self.use_rope:
            attn_scales = (
                torch.log(torch.floor((cache_position.float() + 1.0) / self.floor_scale) + 1.0) * self.attn_scale + 1.0
            )
            attn_scales = attn_scales.view((1, input_shape[-1], 1, 1)).expand((*input_shape, 1, 1))  # batch size > 1
            query_states = (query_states * attn_scales).to(query_states.dtype)

        # query_states = query_states.transpose(1, 2)
        # key_states = key_states.transpose(1, 2)
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
                kv_scales=self.kv_scales,
                kv_cache=kv_cache,
                seqlen=seqlen,
                softmax_scale=self.softmax_scale,
            )
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

        return self.o_proj(
            attn_output.view(-1, self.num_heads * self.head_size)
        )


        # if past_key_value is not None:
        #     # sin and cos are specific to RoPE models; cache_position needed for the static cache
        #     cache_kwargs = {"cache_position": cache_position}
        #     key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # attention_interface: Callable = eager_attention_forward
        # if self.config._attn_implementation != "eager":
        #     if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
        #         logger.warning_once(
        #             "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
        #             'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
        #         )
        #     else:
        #         attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        # attn_output, attn_weights = attention_interface(
        #     self,
        #     query_states,
        #     key_states,
        #     value_states,
        #     attention_mask,
        #     dropout=0.0 if not self.training else self.attention_dropout,
        #     scaling=self.scaling,
        #     **kwargs,
        # )

        # attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        # attn_output = self.o_proj(attn_output)
        # return attn_output, attn_weights


class Llama4TextDecoderLayer(nn.Module):
    def __init__(self, prefix, config, weights, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Llama4TextAttention(f"{prefix}.self_attn", config, weights, layer_idx)
        synchronize(weights.device)
        real_free_memory = get_free_memory(weights.device, 1)
        log_master(
            logger.debug,
            f"layer_idx: {layer_idx} Free memory real: {real_free_memory / 1e9:.2f}GB"
        )
        


        self.use_chunked_attention = int((layer_idx + 1) % 4 != 0)  # <=> use rope
        self.is_moe_layer = layer_idx in config.moe_layers
        if self.is_moe_layer:  # the 128E model interleaves dense / sparse
            self.feed_forward = Llama4TextMoe(f"{prefix}.feed_forward", config, weights)
        else:
            self.feed_forward = Llama4TextMLP(f"{prefix}.feed_forward", config, weights)

        self.input_layernorm = Llama4TextRMSNorm(prefix=f"{prefix}.input_layernorm", config=config, weights=weights)
        self.post_attention_layernorm = Llama4TextRMSNorm(prefix=f"{prefix}.post_attention_layernorm", config=config, weights=weights)
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        chunk_causal_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # use local attention mask for ROPE layers
        if self.use_chunked_attention and chunk_causal_mask is not None:
            attention_mask = chunk_causal_mask

        # Self Attention
        attention_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + attention_states

        # Fully Connected
        residual = hidden_states

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        if self.is_moe_layer:
            hidden_states, router_logits = hidden_states
        else:
            router_logits = None
        hidden_states = residual + hidden_states.view(residual.shape)
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs


# LLAMA4_START_DOCSTRING = r"""
#     This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
#     library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
#     etc.)

#     This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
#     Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
#     and behavior.

#     Parameters:
#         config ([`Llama4Config`]):
#             Model configuration class with all the parameters of the model. Initializing with a config file does not
#             load the weights associated with the model, only the configuration. Check out the
#             [`~PreTrainedModel.from_pretrained`] method to load the model weights.
# """


# @add_start_docstrings(
#     "The bare Llama4 Model outputting raw hidden-states without any specific head on top.",
#     LLAMA4_START_DOCSTRING,
# )
# class Llama4PreTrainedModel(PreTrainedModel):
#     config_class = Llama4Config
#     supports_gradient_checkpointing = True
#     _skip_keys_device_placement = ["past_key_values"]
#     _supports_flash_attn_2 = False
#     _supports_sdpa = True
#     _supports_flex_attn = True
#     _supports_cache_class = True
#     _supports_quantized_cache = True
#     _supports_static_cache = True
#     _supports_attention_backend = True

#     def _init_weights(self, module):
#         std = (
#             self.config.initializer_range
#             if hasattr(self.config, "initializer_range")
#             else self.config.text_config.initializer_range
#         )
#         if isinstance(module, nn.Linear):
#             module.weight.data.normal_(mean=0.0, std=std)
#             if module.bias is not None:
#                 module.bias.data.zero_()
#         elif isinstance(module, nn.Embedding):
#             module.weight.data.normal_(mean=0.0, std=std)
#             if module.padding_idx is not None:
#                 module.weight.data[module.padding_idx].zero_()
#         elif isinstance(module, nn.LayerNorm):
#             module.weight.data.fill_(1.0)
#             module.bias.data.zero_()
#         elif isinstance(module, Llama4TextRMSNorm):
#             module.weight.data.fill_(1.0)
#         elif isinstance(module, Llama4TextExperts):
#             module.gate_up_proj.data.normal_(mean=0.0, std=std)
#             module.down_proj.data.normal_(mean=0.0, std=std)
#         elif isinstance(module, Llama4VisionModel):
#             module.class_embedding.data.normal_(std=module.scale)
#             module.positional_embedding_vlm.data.normal_(std=module.scale)


# LLAMA4_INPUTS_DOCSTRING = r"""
#     Args:
#         input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
#             Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
#             it.

#             Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
#             [`PreTrainedTokenizer.__call__`] for details.

#             [What are input IDs?](../glossary#input-ids)
#         attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
#             Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

#             - 1 for tokens that are **not masked**,
#             - 0 for tokens that are **masked**.

#             [What are attention masks?](../glossary#attention-mask)

#             Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
#             [`PreTrainedTokenizer.__call__`] for details.

#             If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
#             `past_key_values`).

#             If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
#             and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
#             information on the default strategy.

#             - 1 indicates the head is **not masked**,
#             - 0 indicates the head is **masked**.
#         position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
#             Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
#             config.n_positions - 1]`.

#             [What are position IDs?](../glossary#position-ids)
#         past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
#             Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
#             blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
#             returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

#             Two formats are allowed:
#             - a [`~cache_utils.Cache`] instance, see our
#             [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache);
#             - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
#             shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
#             cache format.

#             The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
#             legacy cache format will be returned.

#             If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
#             have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
#             of shape `(batch_size, sequence_length)`.
#         inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
#             Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
#             is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
#             model's internal embedding lookup matrix.
#         use_cache (`bool`, *optional*):
#             If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
#             `past_key_values`).
#         output_attentions (`bool`, *optional*):
#             Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
#             tensors for more detail.
#         output_hidden_states (`bool`, *optional*):
#             Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
#             more detail.
#         return_dict (`bool`, *optional*):
#             Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
#         cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
#             Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
#             this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
#             the complete sequence length.
# """


# @add_start_docstrings(
#     "The bare Llama4 Model outputting raw hidden-states without any specific head on top.",
#     LLAMA4_START_DOCSTRING,
# )
class Llama4TextModel(nn.Module):
    _no_split_modules = ["Llama4TextDecoderLayer"]
    # base_model_prefix = "model"
    # config_class = Llama4TextConfig

    def __init__(self, prefix, config, weights):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = TensorParallelEmbedding(prefix=f"{prefix}.embed_tokens", weights=weights)
        synchronize(weights.device)
        real_free_memory = get_free_memory(weights.device, 1)
        log_master(
            logger.debug,
            f"textModel Free memory real: {real_free_memory / 1e9:.2f}GB"
        )
        log_master(
            logger.debug,
            f"config.num_hidden_layers: {config.num_hidden_layers} "
        )
        self.layers = nn.ModuleList(
            [Llama4TextDecoderLayer(prefix=f"{prefix}.layers.{layer_idx}", config=config, weights=weights, layer_idx=layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        
        self.norm = Llama4TextRMSNorm(prefix=f"{prefix}.norm", config=config, weights=weights)
        self.rotary_emb = Llama4TextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        #self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            # logger.warning_once(
            #     "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            # )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids.to(self.embed_tokens.weight.device))

        if use_cache and past_key_values is None:
            past_key_values = HybridChunkedCache(self.config, inputs_embeds.shape[0], inputs_embeds.shape[1])

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask, chunk_causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions, use_cache=use_cache
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        freq_cis = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                chunk_causal_mask=chunk_causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=freq_cis,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()

    @torch.compiler.disable(recursive=False)  # the operations in this method are not compilable
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
                return attention_mask, attention_mask  # flash does not support chunked attn TODO support flash
            return None, None

        if self.config._attn_implementation not in ["sdpa", "flex_attention", "eager"]:
            return None, None

        sequence_length = input_tensor.shape[1]
        cache_position = cache_position.to(self.device)
        attention_chunk_size = self.config.attention_chunk_size

        first_cache_position = cache_position[0]

        if past_key_values is not None:
            full_cache_length = past_key_values.get_max_cache_shape() or sequence_length
        else:
            full_cache_length = attention_mask.shape[-1] if attention_mask is not None else sequence_length

        cond1 = first_cache_position >= attention_chunk_size
        cond2 = (first_cache_position < attention_chunk_size) & (
            first_cache_position + sequence_length > attention_chunk_size
        )
        key_length = (
            torch.where(
                cond1,
                attention_chunk_size + sequence_length - 1,
                torch.where(cond2, first_cache_position + sequence_length, attention_chunk_size),
            )
            if use_cache
            else full_cache_length
        )

        if self.config._attn_implementation == "flex_attention":
            if isinstance(attention_mask, torch.Tensor):
                offsets = (first_cache_position, max(first_cache_position - attention_chunk_size + 1, 0))
                chunked_attention_mask = make_flex_block_causal_mask(
                    attention_mask, self.config.attention_chunk_size, sequence_length, key_length, offsets=offsets
                )
                attention_mask = make_flex_block_causal_mask(
                    attention_mask,
                    query_length=sequence_length,
                    key_length=full_cache_length,
                    offsets=(first_cache_position, 0),
                )
                return attention_mask, chunked_attention_mask
            if isinstance(attention_mask, BlockMask):
                return attention_mask, chunked_attention_mask

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        dtype, device = input_tensor.dtype, input_tensor.device
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=max(full_cache_length, attention_chunk_size),
            dtype=dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
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

            local_attention_mask = attention_mask[:, start_idx:end_idx]  # offset here as well
            # It may be smaller than attention_chunk_size -> pad it
            requires_padding = local_attention_mask.shape[-1] < attention_chunk_size
            if requires_padding:
                local_attention_mask = nn.functional.pad(
                    local_attention_mask, (0, attention_chunk_size - local_attention_mask.shape[-1])
                )
            # Depending on the padding, take the query tokens from the end or the cache_position
            if not requires_padding:
                chunked_attention_mask = chunked_attention_mask[None, None, -sequence_length:, :]
            else:
                chunked_attention_mask = chunked_attention_mask[None, None, cache_position, :]

            chunked_attention_mask = chunked_attention_mask.expand(input_tensor.shape[0], -1, -1, -1)
            chunked_attention_mask = chunked_attention_mask * local_attention_mask[:, None, None, :]
            if self.config._attn_implementation == "eager":
                min_dtype = torch.finfo(dtype).min
                chunked_attention_mask = torch.where(chunked_attention_mask == 0, min_dtype, 0.0).to(dtype)

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
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and chunked_attention_mask is not None:
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
            arange_vector.unsqueeze(0) // attention_chunk_size - arange_vector.unsqueeze(1) // attention_chunk_size
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
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.to(device).reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(device)
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask


class Llama4ForCausalLM(nn.Module):
    _no_split_modules = ["Llama4TextDecoderLayer"]
    # base_model_prefix = "language_model"
    # _tied_weights_keys = ["lm_head.weight"]
    # _tp_plan = {"lm_head": "colwise_rep"}
    # config_class = Llama4TextConfig

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


        #nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        #self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Llama4ForCausalLM

        >>> model = Llama4ForCausalLM.from_pretrained("meta-llama4/Llama4-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama4/Llama4-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits, speculative_logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return logits, speculative_logits
        # return CausalLMOutputWithPast(
        #     loss=loss,
        #     logits=logits,
        #     past_key_values=outputs.past_key_values,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )


class Llama4CausalLMOutputWithPast(ModelOutput):
    """
    Base class for Llava causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        image_hidden_states (`torch.FloatTensor`, *optional*):
            A `torch.FloatTensor` of size (batch_size, num_images, sequence_length, hidden_size)`.
            image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[torch.FloatTensor] = None


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
        return self.activation_fn(self.fc2(hidden_states))


class Llama4MultiModalProjector(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        self.linear_1 = TensorParallelColumnLinear.load(
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

    reshaped_tensor = input_tensor.view(batch_size, height, int(width * shuffle_ratio), int(channels / shuffle_ratio))
    reshaped_tensor = reshaped_tensor.permute(0, 2, 1, 3).contiguous()

    reshaped_tensor = reshaped_tensor.view(
        batch_size, int(height * shuffle_ratio), int(width * shuffle_ratio), int(channels / (shuffle_ratio**2))
    )
    reshaped_tensor = reshaped_tensor.permute(0, 2, 1, 3).contiguous()

    output_tensor = reshaped_tensor.view(batch_size, -1, reshaped_tensor.shape[-1])
    return output_tensor


class Llama4VisionPixelShuffleMLP(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        self.pixel_shuffle_ratio = config.pixel_shuffle_ratio
        self.inner_dim = int(config.projector_input_dim // (self.pixel_shuffle_ratio**2))
        self.output_dim = config.projector_output_dim
        self.mlp = Llama4VisionMLP2(prefix=f"{prefix}.mlp", config=config, weights=weights)

    def forward(self, encoded_patches: torch.Tensor) -> torch.Tensor:
        encoded_patches = pixel_shuffle(encoded_patches, self.pixel_shuffle_ratio)
        return self.mlp(encoded_patches)


LLAVA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlavaConfig`] or [`LlavaVisionConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""
def reshape_for_broadcast(freqs: torch.Tensor, target: torch.Tensor):
    """Reshape frequency tensor for broadcasting to target tensor."""
    ndim = target.ndim
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(target.shape)]
    return freqs.view(*shape)

def vision_apply_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    rotary_emb: torch.Tensor,  # Now takes (cos_theta, sin_theta) instead of complex
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embedding to query and key tensors using floating-point operations.
    
    Args:
        query: Query tensor of shape (batch, seq_len, n_heads, head_dim)
        key: Key tensor of shape (batch, seq_len, n_heads, head_dim)
        rotary_emb: Tuple of (cos_theta, sin_theta) tensors from Llama4VisionRotaryEmbedding
    Returns:
        Rotated query and key tensors
    """
    cos_theta, sin_theta = rotary_emb.split(1, dim=-1)  # Unpack cos and sin components
    
    # Reshape query/key to separate real and imaginary components
    query_reshaped = query.float().reshape(*query.shape[:-1], -1, 2)  # [..., head_dim//2, 2]
    key_reshaped = key.float().reshape(*key.shape[:-1], -1, 2)        # [..., head_dim//2, 2]
    
    # Reshape cos/sin for broadcasting
    cos_theta = reshape_for_broadcast(cos_theta, query_reshaped)
    sin_theta = reshape_for_broadcast(sin_theta, query_reshaped)
    
    # Apply rotary transformation (equivalent to complex multiplication)
    # For each pair (x0, x1): [x0*cosθ - x1*sinθ, x0*sinθ + x1*cosθ]
    query_out = torch.stack([
        query_reshaped[..., 0] * cos_theta - query_reshaped[..., 1] * sin_theta,
        query_reshaped[..., 0] * sin_theta + query_reshaped[..., 1] * cos_theta
    ], dim=-1)
    
    key_out = torch.stack([
        key_reshaped[..., 0] * cos_theta - key_reshaped[..., 1] * sin_theta,
        key_reshaped[..., 0] * sin_theta + key_reshaped[..., 1] * cos_theta
    ], dim=-1)
    
    # Restore original shape
    query_out = query_out.flatten(-2)  # [batch, seq_len, n_heads, head_dim]
    key_out = key_out.flatten(-2)
    
    # Maintain original dtype
    return query_out.type_as(query), key_out.type_as(key)

# # TODO there is a different RoPE for vision encoder, defined as below
# def reshape_for_broadcast(freqs_ci: torch.Tensor, query: torch.Tensor):
#     ndim = query.ndim
#     shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(query.shape)]
#     return freqs_ci.view(*shape)


# def vision_apply_rotary_emb(
#     query: torch.Tensor,
#     key: torch.Tensor,
#     freqs_ci: torch.Tensor,
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     query_ = torch.view_as_complex(query.float().reshape(*query.shape[:-1], -1, 2))
#     key_ = torch.view_as_complex(key.float().reshape(*key.shape[:-1], -1, 2))
#     freqs_ci = reshape_for_broadcast(freqs_ci=freqs_ci, query=query_)  # freqs_ci[:,:,None,:]
#     freqs_ci = freqs_ci.to(query_.device)
#     query_out = torch.view_as_real(query_ * freqs_ci).flatten(3)
#     key_out = torch.view_as_real(key_ * freqs_ci).flatten(3)
#     return query_out.type_as(query), key_out.type_as(key)  # but this drops to 8e-3


class Llama4VisionAttention(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_groups = 1
        self.attention_dropout = config.attention_dropout

        self.qkv_proj = TensorParallelColumnLinear.load_multi(
            config=config,
            prefixes=[f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"],
            dim=0,
            weights=weights,
            bias=True,
        )
        self.o_proj = TensorParallelRowLinear.load(
            config=config,
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

        query_states, key_states = vision_apply_rotary_emb(query_states, key_states, freqs_ci=freqs_ci)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states, attn_mask=attention_mask
        )
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
        output_attentions: Optional[bool] = None,
    ):
        # Self Attention
        residual = hidden_state

        hidden_state = self.input_layernorm(hidden_state)

        hidden_state, attn_weights = self.self_attn(
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

        if output_attentions:
            outputs += (attn_weights,)

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
        self.layers = nn.ModuleList([
            Llama4VisionEncoderLayer(prefix=f"{prefix}.layers.{layer_id}", config=config, weights=weights)
            for layer_id in range(config.num_hidden_layers)
        ])
        self.gradient_checkpointing = False
        self.config = config

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_ci: torch.Tensor, # TODO move this to an attribute instead of keeping it around
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            layer_outputs = encoder_layer(
                hidden_state=hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                freqs_ci=freqs_ci,
            )

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

            hidden_states = layer_outputs[0]

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class Llama4UnfoldConvolution(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        kernel_size = config.patch_size
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.unfold = torch.nn.Unfold(kernel_size=kernel_size, stride=config.patch_size)
        self.linear = TensorParallelColumnLinear.load(
            config=config, prefix=f"{prefix}.linear", weights=weights, bias=False
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.unfold(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = self.linear(hidden_states)
        return hidden_states

class Llama4VisionRotaryEmbedding(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        # Calculate image grid indices
        idx = config.image_size // config.patch_size
        print(f"idx: {idx}")
        img_idx = torch.arange(idx**2, dtype=torch.int32, device=device).reshape(idx**2, 1)
        img_idx = torch.cat([img_idx, img_idx[:1]], dim=0)
        img_idx[-1, -1] = -2  # ID_CLS_TOKEN
        
        # Calculate x and y coordinates
        frequencies_x = img_idx % idx  # x coordinates
        frequencies_y = img_idx // idx  # y coordinates
        print(f"frequencies_x device: {frequencies_x.device}, frequencies_y device: {frequencies_y.device}") 
        # Calculate frequency components
        freq_dim = config.hidden_size // config.num_attention_heads // 2
        rope_freq = 1.0 / (config.rope_theta ** (torch.arange(0, freq_dim, 2, device=device)[: (freq_dim // 2)].float() / freq_dim))
        
        # Compute frequencies for x and y directions
        freqs_x = ((frequencies_x + 1)[..., None] * rope_freq[None, None, :]).repeat_interleave(2, dim=-1)
        freqs_y = ((frequencies_y + 1)[..., None] * rope_freq[None, None, :]).repeat_interleave(2, dim=-1)
        
        # Combine frequencies and mask special tokens
        freqs = torch.cat([freqs_x, freqs_y], dim=-1).float().contiguous()[..., ::2]
        freqs = freqs.masked_fill(img_idx.reshape(-1, 1, 1) < 0, 0)
        
        # Store cosθ and sinθ separately instead of complex numbers
        cos_freq = torch.cos(freqs)
        sin_freq = torch.sin(freqs)
        print(f"cos_freq shape: {cos_freq.shape}, sin_freq shape: {sin_freq.shape}")
        self.freq_cis = torch.stack([cos_freq, sin_freq], dim=-1)
        print(f"self.freq_cis.device= {self.freq_cis.device}, dtype: {self.freq_cis.dtype}")
        print(f"self.freq_cis shape: {self.freq_cis.shape}")
        # # Store sequence length for validation
        # self.seq_len = idx**2 + 1  # +1 for CLS token
        # print(f"self.seq_len: {self.seq_len}, freqs shape: {freqs.shape}")

    def forward(self, hidden_states):
        """
        Returns the rotary embedding components (cosθ, sinθ) for the given hidden states
        """
        return self.freq_cis
        # batch_size, seq_len, _, _ = hidden_states.shape
        # if seq_len != self.seq_len:
        #     raise ValueError(f"Input sequence length {seq_len} doesn't match expected length {self.seq_len}")
        
        # Return both components on the correct device
        

# class Llama4VisionRotaryEmbedding(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         idx = config.image_size // config.patch_size
#         img_idx = torch.arange(idx**2, dtype=torch.int32).reshape(idx**2, 1)
#         img_idx = torch.cat([img_idx, img_idx[:1]], dim=0)
#         img_idx[-1, -1] = -2  # ID_CLS_TOKEN
#         frequencies_x = img_idx % idx  # get the coordinates of the 2d matrix along x
#         frequencies_y = img_idx // idx  # get the coordinates of the 2d matrix along y
#         freq_dim = config.hidden_size // config.num_attention_heads // 2
#         rope_freq = 1.0 / (config.rope_theta ** (torch.arange(0, freq_dim, 2)[: (freq_dim // 2)].float() / freq_dim))
#         freqs_x = ((frequencies_x + 1)[..., None] * rope_freq[None, None, :]).repeat_interleave(2, dim=-1)
#         freqs_y = ((frequencies_y + 1)[..., None] * rope_freq[None, None, :]).repeat_interleave(2, dim=-1)
#         freqs = torch.cat([freqs_x, freqs_y], dim=-1).float().contiguous()[..., ::2]
#         freqs = freqs.masked_fill(img_idx.reshape(-1, 1, 1) < 0, 0)
#         freq_cis = torch.view_as_complex(torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1))
#         self.freqs_ci = freq_cis  # idx**2, idx**2, idx * 2

#     def forward(self, hidden_states):
#         return self.freqs_ci.to(hidden_states.device)


class Llama4VisionModel(nn.Module):
    #base_model_prefix = "vision_model"
    _no_split_modules = ["Llama4VisionEncoderLayer"]
    #config_class = Llama4VisionConfig

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
            weights.get_sharded(f"{prefix}.class_embedding", dim=0), requires_grad=False
        )
        print(f"self.class_embedding device: {self.class_embedding.device}")

        self.positional_embedding_vlm = nn.Parameter(
            weights.get_sharded(f"{prefix}.positional_embedding_vlm", dim=1), requires_grad=False
        )
        print(f"self.positional_embedding_vlm device: {self.positional_embedding_vlm.device}")
        print(
            f"positional_embedding_vlm shape: {self.positional_embedding_vlm.shape}, "
            f"num_patches: {self.num_patches}, hidden_size: {self.hidden_size}"
        )
        self.rotary_embedding = Llama4VisionRotaryEmbedding(config, weights.device)

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
        #self.post_init()

    def get_input_embeddings(self):
        """
        This function is used to fetch the first embedding layer to activate grads on inputs.
        """
        return self.patch_embedding

    def forward(
        self,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[BaseModelOutput, Tuple[torch.Tensor, ...]]:
        r"""

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, MllamaVisionModel

        >>> checkpoint = "meta-llama/Llama-3.2-11B-Vision"
        >>> model = MllamaVisionModel.from_pretrained(checkpoint)
        >>> processor = AutoProcessor.from_pretrained(checkpoint)

        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = processor(images=image, return_tensors="pt")

        >>> output = model(**inputs)

        >>> print(output.last_hidden_state.shape)
        torch.Size([1, 1, 4, 1025, 7680])
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # num_concurrent_media and num_chunks are both currently 1
        batch_size_times_num_tiles, num_channels, height, width = pixel_values.shape
        num_concurrent_media = 1
        num_chunks = 1
        hidden_state = self.patch_embedding(pixel_values)
        _, num_patches, hidden_dim = hidden_state.shape

        # Add cls token
        hidden_state = hidden_state.reshape(
            batch_size_times_num_tiles * num_concurrent_media * num_chunks, num_patches, hidden_dim
        )
        class_embedding = self.class_embedding.expand(hidden_state.shape[0], 1, hidden_state.shape[-1])
        hidden_state = torch.cat([hidden_state, class_embedding], dim=1)
        num_patches += 1

        # Position embeddings
        hidden_state = hidden_state.reshape(
            batch_size_times_num_tiles * num_concurrent_media, num_chunks, num_patches, hidden_dim
        )
        positional_embedding = self.positional_embedding_vlm.to(dtype=hidden_state.dtype, device=hidden_state.device)
        print(
            f"positional_embedding_vlm shape: {positional_embedding.shape}, hidden_state shape: {hidden_state.shape}"
        )
        hidden_state = hidden_state + positional_embedding

        hidden_state = self.layernorm_pre(hidden_state)

        hidden_state = hidden_state.view(batch_size_times_num_tiles, -1, hidden_dim)
        print(
            f"hidden_state shape: {hidden_state.shape}, batch_size_times_num_tiles: {batch_size_times_num_tiles}, "
            f"num_patches: {num_patches}, hidden_dim: {hidden_dim}"
        )
        print(f"pixel_values shape: {pixel_values.shape}, hidden_state shape: {hidden_state.shape}")
        freqs_ci = self.rotary_embedding(pixel_values)

        output = self.model(
            hidden_state,
            attention_mask=None,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            freqs_ci=freqs_ci,
        )

        hidden_state = output.last_hidden_state

        hidden_state = self.layernorm_post(hidden_state)

        hidden_state = hidden_state[:, :-1, :]

        # now, we use Llama4VisionPixelShuffle + mlp to project embeddings
        hidden_state = self.vision_adapter(hidden_state)

        hidden_states = output.hidden_states if output_hidden_states else None

        if output_attentions:
            attentions = output[2]
        else:
            attentions = None

        if not return_dict:
            return tuple(v for v in [hidden_state, hidden_states, attentions] if v is not None)

        return BaseModelOutput(
            last_hidden_state=hidden_state,
            hidden_states=hidden_states,
            attentions=attentions,
        )


class Llama4ForConditionalGeneration(nn.Module):
    # _no_split_modules = ["Llama4TextDecoderLayer", "Llama4VisionEncoderLayer"]
    # _tp_plan = {}
    # base_model_prefix = ""
    # config_class = Llama4Config
    # _supports_flex_attn = True

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
        
        synchronize(weights.device)
        real_free_memory = get_free_memory(weights.device, 1)
        log_master(
            logger.debug,
            f"Free memory real: {real_free_memory / 1e9:.2f}GB"
        )
 
        self.multi_modal_projector = Llama4MultiModalProjector(
            prefix="multi_modal_projector", config=config, weights=weights
        )
        
        synchronize(weights.device)
        real_free_memory = get_free_memory(weights.device, 1)
        log_master(
            logger.debug,
            f"Free memory real: {real_free_memory / 1e9:.2f}GB"
        )

        self.language_model = Llama4ForCausalLM(
            prefix="language_model", config=config.text_config, weights=weights
        )
        self.vocab_size = config.text_config.vocab_size
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.config = config
        self.dtype = weights.dtype
        self.device = weights.device
        print(f"self.dtype={self.dtype}, self.device={self.device}")
        #self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

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
            raise ValueError(f"Unexpected select feature strategy: {self.vision_feature_select_strategy}")
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        image_outputs = self.vision_model(pixel_values, output_hidden_states=False, **kwargs)
        hidden_state = image_outputs.last_hidden_state
        return hidden_state

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[Union[int, List[int]]] = None,
        vision_feature_select_strategy: Optional[str] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        image_sizes: torch.Tensor = None,
        **lm_kwargs,
    ) -> Union[Tuple, Llama4CausalLMOutputWithPast]:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).


        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, LlavaForConditionalGeneration

        >>> model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
        >>> processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

        >>> prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, text=prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_new_tokens=15)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "USER:  \nWhat's the content of the image? ASSISTANT: The image features a busy city street with a stop sign prominently displayed"
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
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

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

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

            expanded_mask = final_mask_1d.unsqueeze(-1).expand(-1, inputs_embeds.size(-1))
            inputs_embeds = inputs_embeds.masked_scatter(expanded_mask, projected_vision_flat)
            inputs_embeds = inputs_embeds.view(original_inputs_embeds_shape)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **lm_kwargs,
        )

        return outputs
        # logits = outputs[0]

        # loss = None
        # if labels is not None:
        #     # Shift so that tokens < n predict n
        #     if attention_mask is not None:
        #         # we use the input attention mask to shift the logits and labels, because it is 2D.
        #         # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
        #         shift_attention_mask = attention_mask[:, -(logits.shape[1] - 1) :].to(logits.device)
        #         shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
        #         shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
        #     else:
        #         shift_logits = logits[..., :-1, :].contiguous()
        #         shift_labels = labels[..., 1:].contiguous()
        #     # Flatten the tokens
        #     loss_fct = nn.CrossEntropyLoss()
        #     loss = loss_fct(
        #         shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
        #     )

        # if not return_dict:
        #     output = (logits,) + outputs[1:]
        #     return (loss,) + output if loss is not None else output

        # return Llama4CausalLMOutputWithPast(
        #     loss=loss,
        #     logits=logits,
        #     past_key_values=outputs.past_key_values,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        #     image_hidden_states=image_features if pixel_values is not None else None,
        # )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        attention_mask=None,
        cache_position=None,
        logits_to_keep=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        model_inputs = self.language_model.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        if cache_position[0] == 0:
            # If we're in cached decoding stage, pixel values should be None because input ids do not contain special image token anymore
            # Otherwise we need pixel values to be passed to model
            model_inputs["pixel_values"] = pixel_values

        return model_inputs

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
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
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=cache_position.device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask