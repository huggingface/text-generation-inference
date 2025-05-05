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

from typing import Callable, List, Optional, Tuple, Union

import torch
import math
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
    FlashLlamaAttention,
    FlashLlamaForCausalLM,
    LlamaMLP,
)
from habana_frameworks.torch.hpex.kernels import FusedSDPA
from vllm_hpu_extension.utils import ModuleFusedSDPA
from text_generation_server.utils.import_utils import (
    synchronize,
    get_free_memory,
)

from loguru import logger
from text_generation_server.utils.log import log_master
from text_generation_server.layers.moe import DenseMoELayer, MoELayer, SparseMoELayer

_CHECKPOINT_FOR_DOC = "meta-ai/Llama-4-17B"
_CONFIG_FOR_DOC = "Llama4Config"

def torch_save(tensor, name):
    # Only save on the main process (rank 0) when using distributed training
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        torch.save(tensor, name)


class Llama4TextExperts(nn.Module):
    def __init__(self, prefix, config: Llama4TextConfig, weights):
        super().__init__()
        self.process_group = weights.process_group
        self.num_experts = config.num_local_experts
        self.intermediate_size = config.intermediate_size // weights.process_group.size()
        self.hidden_size = config.hidden_size
        self.expert_dim = self.intermediate_size
        self.gate_up_proj = nn.Parameter(weights.get_sharded(f"{prefix}.gate_up_proj", dim=1), requires_grad=False)
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
        gate_up_proj = self.gate_up_proj.view(self.num_experts, -1, 2*self.expert_dim)
        down_proj = self.down_proj.view(self.num_experts, self.expert_dim, -1)
        hidden_states = hidden_states.view(self.num_experts, -1, self.hidden_size)
        gate_up = torch.bmm(hidden_states, gate_up_proj)
        gate, up = gate_up.chunk(2, dim=-1)  # not supported for DTensors
        next_states = torch.bmm((up * self.act_fn(gate)), down_proj)
        
        # Reduce sum
        if self.process_group.size() > 1:
            torch.distributed.all_reduce(next_states, group=self.process_group)
        
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
        self.weight = nn.Parameter(weights.get_tensor(f"{prefix}.weight"), requires_grad=False)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class Llama4TextMoe(nn.Module):
    def __init__(self, prefix, config, weights, layer_idx):
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


        self.router = FastLinear.load(config=config, prefix=f"{prefix}.router", weights=weights, bias=False)
        synchronize(weights.device)
        real_free_memory = get_free_memory(weights.device, 1)
        log_master(
            logger.debug,
            f"TextMode2 Free memory real: {real_free_memory / 1e9:.2f}GB"
        )
        self.shared_expert = LlamaMLP(config=config, prefix=f"{prefix}.shared_expert", weights=weights, index=layer_idx)
        synchronize(weights.device)
        real_free_memory = get_free_memory(weights.device, 1)
        log_master(
            logger.debug,
            f"TextMode3 Free memory real: {real_free_memory / 1e9:.2f}GB"
        )
        self.process_group = weights.process_group
        
        
    def forward(self, hidden_states, adapter_data):
        #seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_dim)
        tokens_per_expert = hidden_states.shape[0]
        router_logits = self.router(hidden_states)

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

        router_indices = router_indices.reshape(-1, 1).expand(-1, self.hidden_dim)
        routed_in = torch.gather(
            input=hidden_states,
            dim=0,
            index=router_indices,
        ).to(hidden_states.device)
        # we gather inputs corresponding to each expert based on the router indices
        routed_in = routed_in * router_scores.reshape(-1, 1)
        routed_out = self.experts(routed_in)
        out = self.shared_expert(hidden_states, adapter_data)
        # now that we finished expert computation -> we scatter add because we gathered previously
        # we have to do this because we used all experts on all tokens. This is faster than the for loop, tho you are compute bound
        # this scales a lot better if you do EP!
        out.scatter_add_(dim=0, index=router_indices, src=routed_out.view(-1, self.hidden_dim))
        
        return out

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

class Llama4TextAttention(FlashLlamaAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, prefix, config, weights, layer_idx):
        super().__init__(layer_idx, prefix, config, weights)
        self.config = config
        # self.layer_idx = layer_idx
        #self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        # self.num_attention_heads = config.num_attention_heads
        # self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        # self.num_key_value_heads = config.num_key_value_heads
        # self.scaling = self.head_dim**-0.5
        # self.attn_scale = config.attn_scale
        # self.floor_scale = config.floor_scale
        # self.attn_temperature_tuning = config.attn_temperature_tuning
        # self.attention_dropout = config.attention_dropout
        # self.is_causal = True
        self.use_rope = int((layer_idx + 1) % 4 != 0)  # rope unused for dense layers
        
        # # `config.attention_multiplier` is used in Granite
        # self.softmax_scale = getattr(
        #     config, "attention_multiplier", self.head_dim**-0.5
        # )

        # if self.num_attention_heads % weights.process_group.size() != 0:
        #     raise ValueError(
        #         f"`num_attention_heads` must be divisible by `num_shards` (got `num_attention_heads`: {self.num_attention_heads} "
        #         f"and `num_shards`: {weights.process_group.size()}"
        #     )
        # if config.num_key_value_heads % weights.process_group.size() != 0:
        #     raise ValueError(
        #         f"`num_key_value_heads` must be divisible by `num_shards` (got `num_key_value_heads`: {config.num_key_value_heads} "
        #         f"and `num_shards`: {weights.process_group.size()}"
        #     )
        # self.num_heads = self.num_attention_heads // weights.process_group.size()
        # self.num_key_value_heads = (
        #     config.num_key_value_heads // weights.process_group.size()
        # )
        
        # self.query_key_value = load_attention(config, prefix, weights, layer_idx)

        # self.kv_scales = get_kv_scales(weights, f"{prefix}")

        # o_proj = TensorParallelRowLinear.load(
        #     config,
        #     prefix=f"{prefix}.o_proj",
        #     weights=weights,
        #     bias=getattr(config, "attention_bias", False),
        # )

        # self.o_proj = TensorParallelAdapterRowLinear.load(
        #     o_proj,
        #     layer_idx,
        #     "o_proj",
        #     process_group=weights.process_group,
        # )

        # self.num_groups = self.num_heads // self.num_key_value_heads
        # self.kv_head_mapping = torch.arange(
        #     0, self.num_key_value_heads, dtype=torch.int32, device=weights.device
        # ).repeat_interleave(self.num_groups)


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
        cos,
        sin,
        cu_seqlen_prefill,
        kv_cache: KVCache,
        slots,
        seqlen,
        adapter_data,
        run_index,
        hpu_attention_meta: Optional[HPUPagedAttentionMetadata] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        #hidden_shape = (*input_shape, -1, self.head_dim)
        qkv = self.query_key_value(hidden_states, adapter_data)
        # query_states, kv_states = qkv.split(
        #     [
        #         self.head_size * self.num_heads,
        #         2 * self.head_size * self.num_key_value_heads,
        #     ],
        #     dim=-1,
        # )
        query_states, key_states, value_states = qkv.split(
            [
                self.head_size * self.num_heads,
                self.head_size * self.num_key_value_heads,
                self.head_size * self.num_key_value_heads,
            ],
            dim=-1,
        )
        
        query_states = query_states.view(-1, self.num_heads, self.head_size)
        key_states = key_states.view(-1, self.num_key_value_heads, self.head_size)
        value_states = value_states.view(-1, self.num_key_value_heads, self.head_size)

        if run_index != -1:
            torch_save(query_states, f"trans.{run_index}.Llama4TextDecoderLayer.{self.index}.attention.query_states.pt")
            torch_save(key_states, f"trans.{run_index}.Llama4TextDecoderLayer.{self.index}.attention.key_states.pt")
            torch_save(value_states, f"trans.{run_index}.Llama4TextDecoderLayer.{self.index}.attention.value_states.pt")

        # query_states = self.q_proj(hidden_states).view(hidden_shape)
        # key_states = self.k_proj(hidden_states).view(*input_shape, -1, self.head_dim)
        # value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if self.use_rope:  # the 16E model skips rope for long context on certain layers
            #self.rotary_emb(query_states, torch.select(kv_states, dim=1, index=0), cos, sin)
            self.rotary_emb(query_states, key_states, cos, sin)

        if run_index != -1:
            torch_save(query_states, f"trans.{run_index}.Llama4TextDecoderLayer.{self.index}.attention.emb.query_states.pt")
            torch_save(key_states, f"trans.{run_index}.Llama4TextDecoderLayer.{self.index}.attention.emb.key_states.pt")


        if hasattr(self, "qk_norm"):  # the 128E model does not use qk_norm
            query_states = self.qk_norm(query_states)
            key_states = self.qk_norm(key_states)

        if run_index != -1:
            torch_save(query_states, f"trans.{run_index}.Llama4TextDecoderLayer.{self.index}.attention.qk_norm.query_states.pt")
            torch_save(key_states, f"trans.{run_index}.Llama4TextDecoderLayer.{self.index}.attention.qk_norm.key_states.pt")



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
            log_master(
                logger.debug,
                f"Prefill: {cu_seqlen_prefill} {seqlen} {slots} {self.kv_head_mapping}"
            )
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
            log_master(
                logger.debug,
                f"Decode: {cu_seqlen_prefill} {seqlen} {slots} {self.kv_head_mapping}"
            )
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
            attn_output.view(-1, self.num_heads * self.head_size), adapter_data
        )

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
            self.feed_forward = Llama4TextMoe(f"{prefix}.feed_forward", config, weights, layer_idx)
        else:
            self.feed_forward = LlamaMLP(f"{prefix}.feed_forward", config, weights, layer_idx)

        self.input_layernorm = Llama4TextRMSNorm(prefix=f"{prefix}.input_layernorm", config=config, weights=weights)
        self.post_attention_layernorm = Llama4TextRMSNorm(prefix=f"{prefix}.post_attention_layernorm", config=config, weights=weights)
        # self.input_layernorm = FastRMSNorm.load(
        #         prefix=f"{prefix}.input_layernorm",
        #         weights=weights,
        #         eps=config.rms_norm_eps,
        #     )
        # self.post_attention_layernorm = FastRMSNorm.load(
        #         prefix=f"{prefix}.post_attention_layernorm",
        #         weights=weights,
        #         eps=config.rms_norm_eps,
        #     )


        self.layer_idx = layer_idx

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
        hpu_attention_meta: Optional[HPUPagedAttentionMetadata],
        run_index
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        if run_index != -1:
            torch_save(hidden_states, f"trans.{run_index}.Llama4TextDecoderLayer.{self.layer_idx}.input.hidden_states.pt")
        hidden_states = self.input_layernorm(hidden_states)
        if run_index != -1:
            torch_save(hidden_states, f"trans.{run_index}.Llama4TextDecoderLayer.{self.layer_idx}.input_layernorm.hidden_states.pt")

        attention_states = self.self_attn(
            hidden_states,
            cos,
            sin,
            cu_seqlen_prefill,
            kv_cache,
            slots,
            seqlen,
            adapter_data,
            run_index,
            hpu_attention_meta=hpu_attention_meta,
        )
        if run_index != -1:
            torch_save(attention_states, f"trans.{run_index}.Llama4TextDecoderLayer.{self.layer_idx}.attention.attention_states.pt")
        hidden_states = residual + attention_states
        if run_index != -1:
            torch_save(hidden_states, f"trans.{run_index}.Llama4TextDecoderLayer.{self.layer_idx}.attention.hidden_states.pt")

        # Fully Connected
        residual = hidden_states

        hidden_states = self.post_attention_layernorm(hidden_states)
        if run_index != -1:
            torch_save(hidden_states, f"trans.{run_index}.Llama4TextDecoderLayer.{self.layer_idx}.post_attention_layernorm.hidden_states.pt")
        hidden_states = self.feed_forward(hidden_states, adapter_data)
        if run_index != -1:
            torch_save(hidden_states, f"trans.{run_index}.Llama4TextDecoderLayer.{self.layer_idx}.feed_forward.hidden_states.pt")
        hidden_states = residual + hidden_states.view(residual.shape)
        if run_index != -1:
            torch_save(hidden_states, f"trans.{run_index}.Llama4TextDecoderLayer.{self.layer_idx}.output.hidden_states.pt")
        #outputs = (hidden_states,)
        return hidden_states
        # if residual is None:
        #     residual = hidden_states
        #     hidden_states, _ = self.input_layernorm(hidden_states)
        # else:
        #     hidden_states, residual = self.input_layernorm(
        #         hidden_states, residual)
        # hidden_states = self.self_attn(
        #     hidden_states,
        #     cos,
        #     sin,
        #     cu_seqlen_prefill,
        #     kv_cache,
        #     slots,
        #     seqlen,
        #     adapter_data,
        #     hpu_attention_meta=hpu_attention_meta,
        # )

        # # Fully Connected
        # hidden_states, residual = self.post_attention_layernorm(
        #     hidden_states, residual)
        # hidden_states = self.feed_forward(hidden_states, adapter_data)
        # return hidden_states, residual

class Llama4TextModel(nn.Module):

    def __init__(self, prefix, config, weights):
        super().__init__()
        self.config = config
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
        
        #self.norm = Llama4TextRMSNorm(prefix=f"{prefix}.norm", config=config, weights=weights)
        self.norm = FastRMSNorm.load(
            prefix=f"{prefix}.norm",
            weights=weights,
            eps=config.rms_norm_eps,
        )
        self.run_index = -1

        #self.rotary_emb = Llama4TextRotaryEmbedding(config=config)
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
    ) -> torch.Tensor:
     
        hidden_states = inputs_embeds
        if self.run_index != -1:
            torch_save(hidden_states, f"trans.{self.run_index}.Llama4TextModel.input.hidden_states.pt")
        log_master(logger.debug, f"inputs_embeds.shape={inputs_embeds.shape}")
        # Get rotary cos and sin for this forward
        # Avoid to index in each layer
        cos, sin = self.layers[0].self_attn.rotary_emb.get_cos_sin(position_ids)

        residual = None
        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                residual,
                cos,
                sin,
                cu_seqlen_prefill,
                kv_cache[i],
                slots,
                seqlen,
                adapter_data,
                hpu_attention_meta=hpu_attention_meta,
                run_index=self.run_index,
            )

        if self.run_index != -1:
            torch_save(hidden_states, f"trans.{self.run_index}.Llama4TextModel.layers.hidden_states.pt")
        log_master(logger.debug, f"hidden_states.shape={hidden_states.shape}")
        hidden_states, _ = self.norm(hidden_states)
        if self.run_index != -1:
            torch_save(hidden_states, f"trans.{self.run_index}.Llama4TextModel.norm.hidden_states.pt")
        log_master(logger.debug, f"normalized hidden_states.shape={hidden_states.shape}")
        self.run_index += 1
        return hidden_states


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
        )
        print(f"lm_head_indices={lm_head_indices}") 
        if lm_head_indices is not None:
            hidden_states = hidden_states[lm_head_indices]

        logits, speculative_logits = self.lm_head(hidden_states)
        return logits, speculative_logits


class Llama4VisionMLP2(torch.nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.fc1 = FastLinear.load(
            config=config, prefix=f"{prefix}.fc1", weights=weights, bias=False
        )
        self.fc2 = FastLinear.load(
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
# def reshape_for_broadcast(freqs: torch.Tensor, target: torch.Tensor):
#     """Reshape frequency tensor for broadcasting to target tensor."""
#     ndim = target.ndim
#     shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(target.shape)]
#     return freqs.view(*shape)
# def reshape_for_broadcast(freqs: torch.Tensor, target: torch.Tensor):
#     ndim = target.ndim
#     shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(target.shape)]
#     return freqs.view(*shape)

def reshape_for_broadcast(freqs: torch.Tensor, target):
    ndim = len(target)
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(target)]
    return freqs.view(*shape)

def vision_apply_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    freqs_ci: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # 调整cos和sin的维度以匹配广播
    cos_emb,sin_emb = freqs_ci.split(1, dim=-1)
   # 将query和key的最后一维拆分为二维向量
    query_reshaped = query.float().reshape(*query.shape[:-1], -1, 2)
    key_reshaped = key.float().reshape(*key.shape[:-1], -1, 2)
    q_shape = query_reshaped.shape[:-1]
    cos_emb = reshape_for_broadcast(cos_emb, q_shape)
    sin_emb = reshape_for_broadcast(sin_emb, q_shape)
    
    # 分离x和y分量
    x_q, y_q = query_reshaped.unbind(-1)
    x_k, y_k = key_reshaped.unbind(-1)
    # 应用旋转矩阵
    x_q_rot = x_q * cos_emb - y_q * sin_emb
    y_q_rot = x_q * sin_emb + y_q * cos_emb
    x_k_rot = x_k * cos_emb - y_k * sin_emb
    y_k_rot = x_k * sin_emb + y_k * cos_emb
    
    # 合并结果并恢复形状
    query_out = torch.stack([x_q_rot, y_q_rot], dim=-1).flatten(-2)
    key_out = torch.stack([x_k_rot, y_k_rot], dim=-1).flatten(-2)
    return query_out.type_as(query), key_out.type_as(key)


# def vision_apply_rotary_emb(
#     query: torch.Tensor,
#     key: torch.Tensor,
#     rotary_emb: torch.Tensor,  # Now takes (cos_theta, sin_theta) instead of complex
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     Apply rotary position embedding to query and key tensors using floating-point operations.
    
#     Args:
#         query: Query tensor of shape (batch, seq_len, n_heads, head_dim)
#         key: Key tensor of shape (batch, seq_len, n_heads, head_dim)
#         rotary_emb: Tuple of (cos_theta, sin_theta) tensors from Llama4VisionRotaryEmbedding
#     Returns:
#         Rotated query and key tensors
#     """
#     from habana_frameworks.torch.hpex.kernels import (
#         RotaryPosEmbeddingMode,
#         apply_rotary_pos_emb,
#     )
#     cos, sin = rotary_emb.split(1, dim=-1)  # Unpack cos and sin components
#     # # cos_emb = reshape_for_broadcast(cos_theta, query)
#     # # sin_emb = reshape_for_broadcast(sin_theta, query)
    
#     # # 将query和key的最后一维拆分为二维向量
#     # query_reshaped = query.float().reshape(*query.shape[:-1], -1, 2)
#     # key_reshaped = key.float().reshape(*key.shape[:-1], -1, 2)
    
#     # # 分离x和y分量
#     # x_q, y_q = query_reshaped.unbind(-1)
#     # x_k, y_k = key_reshaped.unbind(-1)
    
#     # # 应用旋转矩阵
#     # x_q_rot = x_q * cos_emb - y_q * sin_emb
#     # y_q_rot = x_q * sin_emb + y_q * cos_emb
#     # x_k_rot = x_k * cos_emb - y_k * sin_emb
#     # y_k_rot = x_k * sin_emb + y_k * cos_emb
    
#     # # 合并结果并恢复形状
#     # query_out = torch.stack([x_q_rot, y_q_rot], dim=-1).flatten(-2)
#     # key_out = torch.stack([x_k_rot, y_k_rot], dim=-1).flatten(-2)
    
#     # return query_out.type_as(query), key_out.type_as(key)   
#     num_tokens = query.shape[0]
#     head_size = query.shape[-1]
#     # HPU RoPE kernel requires hidden dimension for cos and sin to be equal
#     # to query hidden dimension, so the original tensors need to be
#     # expanded
#     # GPT-NeoX kernel requires position_ids = None, offset, mode = BLOCKWISE
#     # and expansion of cos/sin tensors via concatenation
#     print(f"query.shape: {query.shape}, key.shape: {key.shape}")
#     print(f"cos.shape: {cos.shape}, sin.shape: {sin.shape}")
#     rope_mode = RotaryPosEmbeddingMode.BLOCKWISE
#     cos = torch.cat((cos, cos), dim=-1)
#     sin = torch.cat((sin, sin), dim=-1)
#     rotary_dim = cos.shape[-1]
#     query_shape = query.shape
#     query = query.reshape(num_tokens, -1, head_size)
#     query_rot = query[..., :rotary_dim]
#     query_pass = query[..., rotary_dim:]
#     query_rot = apply_rotary_pos_emb(query_rot, cos, sin, None, 0, rope_mode)
#     query.copy_(torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape))

#     key_shape = key.shape
#     key = key.reshape(num_tokens, -1, head_size)
#     key_rot = key[..., :rotary_dim]
#     key_pass = key[..., rotary_dim:]
#     key_rot = apply_rotary_pos_emb(key_rot, cos, sin, None, 0, rope_mode)
#     key.copy_(torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape))
#     return query, key
    # # Reshape query/key to separate real and imaginary components
    # query_reshaped = query.float().reshape(*query.shape[:-1], -1, 2)  # [..., head_dim//2, 2]
    # key_reshaped = key.float().reshape(*key.shape[:-1], -1, 2)        # [..., head_dim//2, 2]
    
    # # Reshape cos/sin for broadcasting
    # # cos_theta = reshape_for_broadcast(cos_theta, query_reshaped)
    # # sin_theta = reshape_for_broadcast(sin_theta, query_reshaped)
    
    # # Apply rotary transformation (equivalent to complex multiplication)
    # # For each pair (x0, x1): [x0*cosθ - x1*sinθ, x0*sinθ + x1*cosθ]
    # query_out = torch.stack([
    #     query_reshaped[..., 0] * cos_theta - query_reshaped[..., 1] * sin_theta,
    #     query_reshaped[..., 0] * sin_theta + query_reshaped[..., 1] * cos_theta
    # ], dim=-1)
    
    # key_out = torch.stack([
    #     key_reshaped[..., 0] * cos_theta - key_reshaped[..., 1] * sin_theta,
    #     key_reshaped[..., 0] * sin_theta + key_reshaped[..., 1] * cos_theta
    # ], dim=-1)
    
    # # Restore original shape
    # query_out = query_out.flatten(-2)  # [batch, seq_len, n_heads, head_dim]
    # key_out = key_out.flatten(-2)
    
    # # Maintain original dtype
    # return query_out.type_as(query), key_out.type_as(key)

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
        self.num_heads = config.num_attention_heads #// weights.process_group.size()
        self.progress_group = weights.process_group

        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_groups = 1
        self.attention_dropout = config.attention_dropout
        self.q_proj = FastLinear.load(
            prefix=f"{prefix}.q_proj", weights=weights, config=config, bias=True
        )
        self.k_proj = FastLinear.load(
            prefix=f"{prefix}.k_proj", weights=weights, config=config, bias=True
        )
        self.v_proj = FastLinear.load(
            prefix=f"{prefix}.v_proj", weights=weights, config=config, bias=True
        )
        self.o_proj = FastLinear.load(
            prefix=f"{prefix}.o_proj", weights=weights, config=config, bias=True
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_ci: torch.Tensor,  # Now takes (cos_theta, sin_theta) instead of complex
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape)
        #qkv = self.qkv_proj(hidden_states)
        #print(f"qkv shape: {qkv.shape}")
        
        # if self.process_group.size() > 1:
        #     torch.distributed.all_reduce(qkv, group=self.process_group)
            
        # query_states, key_states, value_states = qkv.split(
        #     [
        #         self.head_dim * self.num_heads,
        #         self.head_dim * self.num_heads,
        #         self.head_dim * self.num_heads,
        #     ],
        #     dim=2,
        # )
        # query_states = query_states.view(hidden_shape)
        # key_states = key_states.view(hidden_shape)
        # value_states = value_states.view(hidden_shape)

        query_states, key_states = vision_apply_rotary_emb(query_states, key_states, freqs_ci=freqs_ci)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        #print(f"attention_mask shape: {attention_mask.shape}")
        #print(f"attention_mask: {attention_mask}")
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
        self.fc1 = FastLinear.load(
            prefix=f"{prefix}.fc1", weights=weights, config=config, bias=True
        )
        self.fc2 = FastLinear.load(
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
        # self.linear = TensorParallelColumnLinear.load(
        #     config=config, prefix=f"{prefix}.linear", weights=weights, bias=False
        # )
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
        img_idx = torch.arange(idx**2, dtype=torch.int32, device=weights.device).reshape(idx**2, 1)
        img_idx = torch.cat([img_idx, img_idx[:1]], dim=0)
        img_idx[-1, -1] = -2  # ID_CLS_TOKEN
        
        # Calculate x and y coordinates
        frequencies_x = img_idx % idx  # x coordinates
        frequencies_y = img_idx // idx  # y coordinates
        # Calculate frequency components
        freq_dim = config.hidden_size // config.num_attention_heads // 2
        rope_freq = 1.0 / (config.rope_theta ** (torch.arange(0, freq_dim, 2, device=weights.device)[: (freq_dim // 2)].float() / freq_dim))
        
        # Compute frequencies for x and y directions
        freqs_x = ((frequencies_x + 1)[..., None] * rope_freq[None, None, :]).repeat_interleave(2, dim=-1)
        freqs_y = ((frequencies_y + 1)[..., None] * rope_freq[None, None, :]).repeat_interleave(2, dim=-1)
        
        # Combine frequencies and mask special tokens
        freqs = torch.cat([freqs_x, freqs_y], dim=-1).float().contiguous()[..., ::2]
        freqs = freqs.masked_fill(img_idx.reshape(-1, 1, 1) < 0, 0)
        
        # Store cosθ and sinθ separately instead of complex numbers
        cos_freq = torch.cos(freqs)
        sin_freq = torch.sin(freqs)
        self.freqs_ci = torch.stack([cos_freq, sin_freq], dim=-1).to(weights.dtype)
        # # Store sequence length for validation
        # self.seq_len = idx**2 + 1  # +1 for CLS token
        # print(f"self.seq_len: {self.seq_len}, freqs shape: {freqs.shape}")

    def forward(self, hidden_states):
        """
        Returns the rotary embedding components (cosθ, sinθ) for the given hidden states
        """
        return self.freqs_ci


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
            weights.get_tensor(f"{prefix}.positional_embedding_vlm"), requires_grad=False
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

        self.text_model = Llama4ForCausalLM(
            prefix="language_model", config=config.text_config, weights=weights
        )
        self.vocab_size = config.text_config.vocab_size
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
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
            raise ValueError(f"Unexpected select feature strategy: {self.vision_feature_select_strategy}")
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
        log_master(
            logger.debug,
            f"input_ids: {input_ids}, shape = {input_ids.shape}, input_ids={input_ids[-20:]}"
        )      
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

        # if (input_ids is None) ^ (inputs_embeds is not None):
        #     raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # if pixel_values is not None and inputs_embeds is not None:
        #     raise ValueError(
        #         "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
        #     )
        if pixel_values is not None:
            print(f"pixel_values!!!!!!!!!!!!!!!!!")
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

        logits, speculative_logits= self.text_model(
            inputs_embeds,
            position_ids,
            cu_seqlen_prefill,
            kv_cache,
            slots,
            seqlen,
            hpu_attention_meta,
            adapter_data,
            lm_head_indices,
        )

        return logits, speculative_logits