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

from typing import Callable, List, Optional, Tuple, Union, Type

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
def print_0(*args, **kwargs):
    """
    Only print on rank 0 in distributed training.
    Works like built-in print() function but only executes on rank 0.
    """
    # 检查是否处于分布式环境
    if torch.distributed.is_initialized():
        # 获取当前rank
        if torch.distributed.get_rank() == 0:
            print(*args, **kwargs)
    else:
        # 如果不是分布式环境，正常打印
        print(*args, **kwargs)
 
def torch_save(tensor, name):
    # Only save on the main process (rank 0) when using distributed training
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        torch.save(tensor, name)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    print_0(f"batch={batch}, num_key_value_heads={num_key_value_heads}, slen={slen}, head_dim={head_dim}")
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Llama4TextExperts(nn.Module):
    def __init__(self, prefix, config: Llama4TextConfig, weights, layer_idx):
        super().__init__()
        self.process_group = weights.process_group
        self.num_experts = config.num_local_experts
        self.intermediate_size = config.intermediate_size // weights.process_group.size()
        self.hidden_size = config.hidden_size
        self.expert_dim = self.intermediate_size
        self.gate_up_proj = nn.Parameter(weights.get_packed_sharded(f"{prefix}.gate_up_proj", dim=-1, block_sizes=2), requires_grad=False)
        synchronize(weights.device)
        real_free_memory = get_free_memory(weights.device, 1)
        log_master(
            logger.debug,
            f"textExperts1 Free memory real: {real_free_memory / 1e9:.2f}GB"
        )


        self.down_proj = nn.Parameter(weights.get_sharded(f"{prefix}.down_proj", dim=1), requires_grad=False)
        synchronize(weights.device)
        real_free_memory = get_free_memory(weights.device, 1)
        log_master(
            logger.debug,
            f"textExperts2 Free memory real: {real_free_memory / 1e9:.2f}GB"
        )

        self.layer_idx = layer_idx
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor, run_index) -> torch.Tensor:
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
        if run_index == 0:
            torch_save(gate_up_proj, f"trans.{run_index}.Llama4TextDecoderLayer.{self.layer_idx}.expert.gate_up_proj.pt")


        down_proj = self.down_proj.view(self.num_experts, self.expert_dim, -1)
        if run_index == 0:
            torch_save(down_proj, f"trans.{run_index}.Llama4TextDecoderLayer.{self.layer_idx}.expert.down_proj.pt")


        hidden_states = hidden_states.view(self.num_experts, -1, self.hidden_size)
        if run_index == 0:
            torch_save(hidden_states, f"trans.{run_index}.Llama4TextDecoderLayer.{self.layer_idx}.expert.hidden_states.pt")


        gate_up = torch.bmm(hidden_states, gate_up_proj)
        gate, up = gate_up.chunk(2, dim=-1)  # not supported for DTensors
        if run_index == 0:
            torch_save(gate, f"trans.{run_index}.Llama4TextDecoderLayer.{self.layer_idx}.expert.gate.pt")
            torch_save(up, f"trans.{run_index}.Llama4TextDecoderLayer.{self.layer_idx}.expert.up.pt")


        next_states = torch.bmm((up * self.act_fn(gate)), down_proj)
       
       
        next_states = next_states.view(-1, self.hidden_size)
        if run_index == 0:
            torch_save(next_states, f"trans.{run_index}.Llama4TextDecoderLayer.{self.layer_idx}.expert.next_states.pt")
  
        # Reduce sum
        if self.process_group.size() > 1:
            torch.distributed.all_reduce(next_states, group=self.process_group)
       
        return next_states


# Phi3MLP
class Llama4TextMLP(nn.Module):
    def __init__(self, prefix, config, weights, layer_idx):
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
        self.layer_idx = layer_idx
        self.act_fn = ACT2FN[config.hidden_act]
        # self.intermediate_size = (
        #     config.intermediate_size // weights.process_group.size()
        # )

        # self.config = config
        # # self.gate_up_proj = TensorParallelColumnLinear.load_multi(
        # #     config,
        # #     prefixes=[f"{prefix}.gate_proj", f"{prefix}.up_proj"],
        # #     weights=weights,
        # #     dim=0,
        # #     bias=False,
        # # )
        # self.gate_proj = TensorParallelColumnLinear.load(config=config, prefix=f"{prefix}.gate_proj", weights=weights, bias=False)
        # self.up_proj = TensorParallelColumnLinear.load(config=config, prefix=f"{prefix}.up_proj", weights=weights, bias=False)
        # self.down_proj = TensorParallelRowLinear.load(config=config, prefix=f"{prefix}.down_proj", weights=weights, bias=False) 
        # self.activation_fn = ACT2FN[config.hidden_act]



    def forward(self, x, run_index, reduce=True):
        shape = x.shape
        # gate_up_states = self.gate_up_proj(x)
        # gate_up_states = gate_up_states.view(*shape[:-1], 2, self.intermediate_size)
        # result = self.down_proj(
        #     self.activation_fn(gate_up_states[:, 0]) * gate_up_states[:, 1]
        # )
        # return result
        # down_proj = self.activation_fn(self.gate_proj(x)) * self.up_proj(x)
        # return self.down_proj(down_proj)
        gate_up_states = self.gate_up_proj(x)
        gate_up_states = gate_up_states.view(-1, 2, self.intermediate_size)
        return self.down_proj(
            self.act_fn(gate_up_states[:, 0]) * gate_up_states[:, 1], reduce=True
        )




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
    def __init__(
        self,
        prefix,
        config,
        weights,
        layer_idx,
        moe_layer_cls: Type[MoELayer],
    ):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_local_experts
        log_master(logger.debug, f"weights.load: {weights.loader}")
        # self.experts = moe_layer_cls(
        #     prefix=f"{prefix}.experts",
        #     n_experts=config.num_local_experts,
        #     n_expert_group=None,
        #     renormalize=True,
        #     topk=config.num_experts_per_tok,
        #     topk_group=None,
        #     weights=weights,
        #     scoring_func="sigmoid",
        # )
        # assert isinstance(self.experts, MoELayer)


        self.experts = Llama4TextExperts(config=config, prefix=f"{prefix}.experts", weights=weights, layer_idx=layer_idx)
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
        self.shared_expert = Llama4TextMLP(config=config, prefix=f"{prefix}.shared_expert", weights=weights, layer_idx=layer_idx)
        synchronize(weights.device)
        real_free_memory = get_free_memory(weights.device, 1)
        log_master(
            logger.debug,
            f"TextMode3 Free memory real: {real_free_memory / 1e9:.2f}GB"
        )
        self.process_group = weights.process_group
        self.layer_idx = layer_idx  
        
    def forward(self, hidden_states, adapter_data, run_index):
        seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_dim)
        tokens_per_expert = hidden_states.shape[0]
        router_logits = self.router(hidden_states)
        #if run_index != -1:
        #    torch_save(router_logits, f"trans.{run_index}.Llama4TextDecoderLayer.{self.layer_idx}.moe.routed_logits.pt")

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
        #if run_index != -1:
        #    torch_save(router_scores, f"trans.{run_index}.Llama4TextDecoderLayer.{self.layer_idx}.moe.router_scores.pt")


        router_indices = router_indices.reshape(-1, 1).expand(-1, self.hidden_dim)
        routed_in = torch.gather(
            input=hidden_states,
            dim=0,
            index=router_indices,
        ).to(hidden_states.device)
        #if run_index != -1:
        #    torch_save(routed_in, f"trans.{run_index}.Llama4TextDecoderLayer.{self.layer_idx}.moe.gather.pt")


        # we gather inputs corresponding to each expert based on the router indices
        routed_in = routed_in * router_scores.reshape(-1, 1)
        #if run_index != -1:
        #    torch_save(routed_in, f"trans.{run_index}.Llama4TextDecoderLayer.{self.layer_idx}.moe.routed_in.pt")
        routed_out = self.experts(routed_in, run_index)
        #if run_index != -1:
       #     torch_save(routed_out, f"trans.{run_index}.Llama4TextDecoderLayer.{self.layer_idx}.moe.routed_out.pt")
        out = self.shared_expert(hidden_states, run_index, reduce=False)
        #if run_index != -1:
        #    torch_save(out, f"trans.{run_index}.Llama4TextDecoderLayer.{self.layer_idx}.moe.out.pt")
        # now that we finished expert computation -> we scatter add because we gathered previously
        # we have to do this because we used all experts on all tokens. This is faster than the for loop, tho you are compute bound
        # this scales a lot better if you do EP!
        out.scatter_add_(dim=0, index=router_indices, src=routed_out.view(-1, self.hidden_dim))
        # if run_index != -1:
        #     torch_save(out, f"trans.{run_index}.Llama4TextDecoderLayer.{self.layer_idx}.moe.add.out.pt")
        #Reduce sum
        # if self.process_group.size() > 1:
        #     torch.distributed.all_reduce(out, group=self.process_group)
        
        #if run_index != -1:
        #    torch_save(out, f"trans.{run_index}.Llama4TextDecoderLayer.{self.layer_idx}.moe.add.out.pt")
 
        return out
        
        # shared_output = self.shared_expert(hidden_states, reduce=False)

        # router_logits = self.router(hidden_states)

        # out = self.experts(hidden_states, gating_output=router_logits)

        # if shared_output is not None:
        #     out = out + shared_output

        # # Reduce sum
        # if self.process_group.size() > 1:
        #     torch.distributed.all_reduce(out, group=self.process_group)

        # return out.view(*hidden_states.shape)

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

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        origin_device = x.device
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" and x.device.type != "hpu" else "cpu"
        inv_freq_expanded = inv_freq_expanded.to(device_type)
        position_ids_expanded = position_ids_expanded.to(device_type)
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # Convert to complex representation
            freqs_cis = freqs_cis * self.attention_scaling
        return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    orig_device= xq.device
    xq = xq.to("cpu")
    xk = xk.to("cpu")
    xq = xq.view(freqs_cis.shape[0], -1, *xq.shape[-2:])
    xk = xk.view(freqs_cis.shape[0], -1, *xk.shape[-2:])
    #log_master(logger.debug, f"xq: {xq.shape}, xk: {xk.shape}")
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    #log_master(logger.debug, f"xq_: {xq_.shape}, xk_: {xk_.shape}")
    #log_master(logger.debug, f"freqs_cis: {freqs_cis.shape}")
    xq_out = torch.view_as_real(xq_ * freqs_cis[:, :, None, :]).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis[:, :, None, :]).flatten(3)
    xq_out = xq_out.view(-1, *xq_out.shape[-2:]).to(orig_device)
    xk_out = xk_out.view(-1, *xk_out.shape[-2:]).to(orig_device)
    xq = xq.to(orig_device)
    xk = xk.to(orig_device)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# class Llama4TextRotaryEmbedding(nn.Module):
#     def __init__(self, config: 'Llama4TextConfig', device=None):
#         super().__init__()
#         self.rope_type = "llama3" if config.rope_scaling is not None else "default"
#         self.max_seq_len_cached = config.max_position_embeddings
#         self.original_max_seq_len = config.max_position_embeddings
#         self.config = config
#         self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

#         inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
#         self.register_buffer("inv_freq", inv_freq, persistent=False)
#         self.original_inv_freq = self.inv_freq

#     @torch.no_grad()
#     @dynamic_rope_update
#     def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             x: Input tensor of shape [batch, seq_len, heads, dim]
#             position_ids: Position indices of shape [batch, seq_len]
#         Returns:
#             Rotary embeddings as float tensors [batch, seq_len, heads, dim]
#         """
#         # Expand inv_freq and position_ids for broadcasting
#         inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
#         position_ids_expanded = position_ids[:, None, :].float()

#         # Compute frequencies (replaces complex phase)
#         freqs = (inv_freq_expanded.to(x.device) @ position_ids_expanded).transpose(1, 2)  # [batch, seq_len, dim//2]
        
#         # Generate cos/sin components directly (replaces torch.polar)
#         cos_vals = torch.cos(freqs) * self.attention_scaling
#         sin_vals = torch.sin(freqs) * self.attention_scaling
        
#         # Interleave cos/sin values to match original complex format
#         dim = x.size(-1)
#         if dim % 2 != 0:
#             raise ValueError(f"Feature dimension {dim} must be even for Rotary Embedding")
        
#         # Stack and reshape to [batch, seq_len, dim] format
#         freqs_cis = torch.stack([cos_vals, sin_vals], dim=-1)  # [batch, seq_len, dim//2, 2]
#         freqs_cis = freqs_cis.reshape(*freqs_cis.shape[:-2], dim)  # [batch, seq_len, dim]
        
#         return freqs_cis
    
# def apply_rotary_emb(
#     xq: torch.Tensor,
#     xk: torch.Tensor,
#     freqs_cis: torch.Tensor,  # Should be [cosθ, sinθ] instead of complex numbers
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     Apply rotary position embedding to query and key tensors using floating-point operations only.
    
#     Args:
#         xq: Query tensor of shape (batch, seq_len, n_heads, head_dim)
#         xk: Key tensor of shape (batch, seq_len, n_heads, head_dim)
#         freqs_cis: Precomputed rotation frequencies as [cosθ, sinθ] 
#                   of shape (batch, seq_len, head_dim//2, 2)
#     Returns:
#         Rotated query and key tensors with same shape as input
#     """
#     # Verify head_dim is even
#     assert xq.size(-1) % 2 == 0, "Feature dimension must be even for rotary embedding"
    
#     # Reshape to separate real and imaginary components (pairs of adjacent elements)
#     xq_reshaped = xq.float().reshape(*xq.shape[:-1], -1, 2)  # [..., head_dim//2, 2]
#     xk_reshaped = xk.float().reshape(*xk.shape[:-1], -1, 2)  # [..., head_dim//2, 2]
    
#     # Extract cosθ and sinθ (assuming freqs_cis is already in [cosθ, sinθ] format)
#     cos_theta = freqs_cis[..., 0]  # [batch, seq_len, head_dim//2]
#     sin_theta = freqs_cis[..., 1]  # [batch, seq_len, head_dim//2]
    
#     # Expand dimensions for broadcasting [batch, seq_len, n_heads, head_dim//2]
#     cos_theta = cos_theta.unsqueeze(2)  # Add n_heads dimension
#     sin_theta = sin_theta.unsqueeze(2)
    
#     # Rotary transformation (mathematically equivalent to complex multiplication)
#     # xq_rotated = [xq0*cosθ - xq1*sinθ, xq0*sinθ + xq1*cosθ]
#     xq_out = torch.stack([
#         xq_reshaped[..., 0] * cos_theta - xq_reshaped[..., 1] * sin_theta,
#         xq_reshaped[..., 0] * sin_theta + xq_reshaped[..., 1] * cos_theta
#     ], dim=-1)
    
#     xk_out = torch.stack([
#         xk_reshaped[..., 0] * cos_theta - xk_reshaped[..., 1] * sin_theta,
#         xk_reshaped[..., 0] * sin_theta + xk_reshaped[..., 1] * cos_theta
#     ], dim=-1)
    
#     # Restore original shape
#     xq_out = xq_out.flatten(-2)  # [batch, seq_len, n_heads, head_dim]
#     xk_out = xk_out.flatten(-2)
    
#     # Maintain original dtype
#     return xq_out.type_as(xq), xk_out.type_as(xk)

class Llama4TextAttention(FlashLlamaAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, prefix, config, weights, layer_idx):
        super().__init__(layer_idx, prefix, config, weights)
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
        
        #self.query_key_value = load_attention(config, prefix, weights, layer_idx)

        self.kv_scales = get_kv_scales(weights, f"{prefix}")
        self.q_proj = TensorParallelColumnLinear.load(
            config=config,
            prefix=f"{prefix}.q_proj",
            weights=weights,
            bias=getattr(config, "attention_bias", False),
        )
        self.k_proj = TensorParallelColumnLinear.load(
            config=config,
            prefix=f"{prefix}.k_proj",
            weights=weights,
            bias=getattr(config, "attention_bias", False),
        )
        self.v_proj = TensorParallelColumnLinear.load(
            config=config,
            prefix=f"{prefix}.v_proj",
            weights=weights,
            bias=getattr(config, "attention_bias", False),
        )

        self.o_proj = TensorParallelRowLinear.load(
            config=config,
            prefix=f"{prefix}.o_proj",
            weights=weights,
            bias=getattr(config, "attention_bias", False),
        )

        # self.o_proj = TensorParallelAdapterRowLinear.load(
        #     o_proj,
        #     layer_idx,
        #     "o_proj",
        #     process_group=weights.process_group,
        # )

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
        freqs_ci,
        cu_seqlen_prefill,
        kv_cache: KVCache,
        slots,
        seqlen,
        adapter_data,
        run_index,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        hpu_attention_meta: Optional[HPUPagedAttentionMetadata] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bs = seqlen.input_lengths.shape[0]
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        #qkv = self.query_key_value(hidden_states, adapter_data)
        # query_states, kv_states = qkv.split(
        #     [
        #         self.head_size * self.num_heads,
        #         2 * self.head_size * self.num_key_value_heads,
        #     ],
        #     dim=-1,
        # )
        # query_states, key_states, value_states = qkv.split(
        #     [
        #         self.head_size * self.num_heads,
        #         self.head_size * self.num_key_value_heads,
        #         self.head_size * self.num_key_value_heads,
        #     ],
        #     dim=-1,
        # )
        query_states = self.q_proj(hidden_states).view(-1, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(-1,  self.num_key_value_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(-1,  self.num_key_value_heads, self.head_dim)
        
        # query_states = query_states.view(-1, self.num_heads, self.head_size)
        # key_states = key_states.view(-1, self.num_key_value_heads, self.head_size)
        # value_states = value_states.view(-1, self.num_key_value_heads, self.head_size)

        #if run_index != -1:
       #     torch_save(query_states, f"trans.{run_index}.Llama4TextDecoderLayer.{self.index}.attention.query_states.pt")
        #    torch_save(key_states, f"trans.{run_index}.Llama4TextDecoderLayer.{self.index}.attention.key_states.pt")
        #    torch_save(value_states, f"trans.{run_index}.Llama4TextDecoderLayer.{self.index}.attention.value_states.pt")

        if self.use_rope:  # the 16E model skips rope for long context on certain layers
            #self.rotary_emb(query_states, torch.select(kv_states, dim=1, index=0), cos, sin)
            #self.rotary_emb(query_states, key_states, cos, sin)
            query_states, key_states = apply_rotary_emb(
                query_states, key_states, freqs_ci
            )

            

        #if run_index != -1:
        #    torch_save(query_states, f"trans.{run_index}.Llama4TextDecoderLayer.{self.index}.attention.emb.query_states.pt")
        #    torch_save(key_states, f"trans.{run_index}.Llama4TextDecoderLayer.{self.index}.attention.emb.key_states.pt")


        if hasattr(self, "qk_norm"):  # the 128E model does not use qk_norm
            query_states = self.qk_norm(query_states)
            key_states = self.qk_norm(key_states)

        #if run_index != -1:
        #    torch_save(query_states, f"trans.{run_index}.Llama4TextDecoderLayer.{self.index}.attention.qk_norm.query_states.pt")
        #    torch_save(key_states, f"trans.{run_index}.Llama4TextDecoderLayer.{self.index}.attention.qk_norm.key_states.pt")


        # query_states = query_states.view(-1, self.num_heads, self.head_size)
        # key_states = key_states.view(-1, self.num_key_value_heads, self.head_size)
        # value_states = value_states.view(-1, self.num_key_value_heads, self.head_size)

        # query_states = query_states.transpose(1, 2)
        # key_states = key_states.transpose(1, 2)
        kv_cache.store(
            key=key_states,
            value=value_states,
            slots=slots,
            kv_scales=self.kv_scales,
        )
        # Use temperature tuning from https://arxiv.org/abs/2501.19399) to NoROPE layers
        if self.attn_temperature_tuning and not self.use_rope:
            #indice = torch.tensor([0]).to(query_states.device)
            #cache_position = position_ids
            #log_master(logger.debug, f"cache_position: {cache_position.shape}")
            
            
            attn_scales = (
                torch.log(torch.floor((position_ids.float() + 1.0) / self.floor_scale) + 1.0) * self.attn_scale + 1.0
            )
            #seq_len = input_shape / bs 
            attn_scales = attn_scales.view(*input_shape, 1, 1)
            query_states = (query_states * attn_scales).to(query_states.dtype)
        #if run_index != -1:
        #    torch_save(query_states, f"trans.{run_index}.Llama4TextDecoderLayer.{self.index}.attention.attn_scales.query_states.pt")
        #    torch_save(attention_mask, f"trans.{run_index}.Llama4TextDecoderLayer.{self.index}.attention.attention_mask.pt")
        

        # Prefill
        if cu_seqlen_prefill is not None:
            # sdpa
            # log_master(logger.debug, f"self.softmax_scale: {self.softmax_scale}")
            # attn_output = attention(
            #     query=query_states,
            #     key=key_states,
            #     value=value_states,
            #     kv_scales=self.kv_scales,
            #     kv_cache=kv_cache,
            #     seqlen=seqlen,
            #     softmax_scale=self.softmax_scale,
            #     causal=True
            # )
            query = query_states.view(bs, -1, self.num_heads, self.head_dim).transpose(1, 2)
            key = key_states.view(bs, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value = value_states.view(bs, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            print_0(f"self.num_key_value_groups={self.num_key_value_groups}")
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
            print_0(f"query.shape={query.shape}, query={query}")
            print_0(f"key.shape={key.shape}, key={key}")
            print_0(f"value.shape={value.shape}, value={value}")
            print_0(f"attention_mask.shape={causal_mask.shape}, attention_mask={causal_mask}")
            print_0(f"scaling={self.scaling}, is_causal={is_causal}")

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
        #if run_index != -1:
        #    torch_save(attn_output, f"trans.{run_index}.Llama4TextDecoderLayer.{self.index}.attention.reshape.attn_output.pt")
        attn_output = self.o_proj(attn_output)
        return attn_output 


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
        log_master(logger.debug, f"self.is_moe_layer: {self.is_moe_layer}, layer_idx:{layer_idx}")
        log_master(logger.debug, f"moe_layers:{config.moe_layers}")
        if self.is_moe_layer:  # the 128E model interleaves dense / sparse
            moe_layer_cls = (
                SparseMoELayer
                if SparseMoELayer.is_supported(weights)
                else DenseMoELayer
            )

            self.feed_forward = Llama4TextMoe(f"{prefix}.feed_forward", config, weights, layer_idx, moe_layer_cls)
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
        run_index: int = 0,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        #if run_index != -1:
        #    torch_save(hidden_states, f"trans.{run_index}.Llama4TextDecoderLayer.{self.layer_idx}.input.hidden_states.pt")
        hidden_states = self.input_layernorm(hidden_states)
        #if run_index != -1:
        #    torch_save(hidden_states, f"trans.{run_index}.Llama4TextDecoderLayer.{self.layer_idx}.input_layernorm.hidden_states.pt")

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
            run_index,
            attention_mask=attention_mask,
            position_ids=position_ids,
            hpu_attention_meta=hpu_attention_meta,
        )
        #if run_index != -1:
        #    torch_save(attention_states, f"trans.{run_index}.Llama4TextDecoderLayer.{self.layer_idx}.attention.attention_states.pt")
        hidden_states = residual + attention_states
        #if run_index != -1:
        #    torch_save(hidden_states, f"trans.{run_index}.Llama4TextDecoderLayer.{self.layer_idx}.attention.hidden_states.pt")

        # Fully Connected
        residual = hidden_states

        hidden_states = self.post_attention_layernorm(hidden_states)
        #if run_index != -1:
        #    torch_save(hidden_states, f"trans.{run_index}.Llama4TextDecoderLayer.{self.layer_idx}.post_attention_layernorm.hidden_states.pt")
        hidden_states = self.feed_forward(hidden_states, adapter_data, run_index)
        #if run_index != -1:
        #    torch_save(hidden_states, f"trans.{run_index}.Llama4TextDecoderLayer.{self.layer_idx}.feed_forward.hidden_states.pt")
        hidden_states = residual + hidden_states.view(residual.shape)
        #if run_index != -1:
        #    torch_save(hidden_states, f"trans.{run_index}.Llama4TextDecoderLayer.{self.layer_idx}.output.hidden_states.pt")
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
        #if self.run_index != -1:
        #    torch_save(hidden_states, f"trans.{self.run_index}.Llama4TextModel.input.hidden_states.pt")
        log_master(logger.debug, f"inputs_embeds.shape={inputs_embeds.shape}")
        # Get rotary cos and sin for this forward
        # Avoid to index in each layer
        #cos, sin = self.layers[0].self_attn.rotary_emb.get_cos_sin(position_ids)
        log_master(logger.debug, f"position_ids.shape={position_ids.shape}, position_ids={position_ids}")
        bs = seqlen.input_lengths.shape[0]
        seq_len = inputs_embeds.shape[0] / bs
        cache_position = torch.arange(0, seq_len, device=inputs_embeds.device)

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        log_master(logger.debug, f"cache_position={cache_position}")
        log_master(logger.debug, f"position_ids={position_ids}")
        causal_mask, chunk_causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds.view(bs, int(seq_len), -1), cache_position, None, output_attentions=False, use_cache=False
        )
        log_master(logger.debug, f"causal_mask={causal_mask}")
        log_master(logger.debug, f"causal_mask={causal_mask.shape}")
        log_master(logger.debug, f"chunk_causal_mask={chunk_causal_mask}")
        
        
        
        
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
                run_index=self.run_index,
            )

        if self.run_index == 0:
            torch_save(hidden_states, f"trans.{self.run_index}.Llama4TextModel.layers.hidden_states.pt")
        log_master(logger.debug, f"hidden_states.shape={hidden_states.shape}")
        hidden_states, _ = self.norm(hidden_states)
        if self.run_index == 0:
            torch_save(hidden_states, f"trans.{self.run_index}.Llama4TextModel.norm.hidden_states.pt")
        log_master(logger.debug, f"normalized hidden_states.shape={hidden_states.shape}")
        self.run_index += 1
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
        print(f"update 11111111111111111")
        print(f"self.config._attn_implementation={self.config._attn_implementation}")
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask, attention_mask  # flash does not support chunked attn TODO support flash
            return None, None

        if self.config._attn_implementation not in ["sdpa", "flex_attention", "eager"]:
            return None, None

        print(f"update 222222222222222222")
        sequence_length = input_tensor.shape[1]
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

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        dtype, device = input_tensor.dtype, input_tensor.device
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=max(full_cache_length, attention_chunk_size),
            dtype=dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
            device=device
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
        
        def _get_padding_mask(input_ids, pad_token_id=0):
            return (input_ids != pad_token_id).long()  # 非填充位置为1，填充位置为0

        # 示例
        attention_mask = _get_padding_mask(input_ids)
        attention_mask = attention_mask.view(seqlen.input_lengths.shape[0], -1)
        log_master(logger.debug,f"attention_mask={attention_mask}")      
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
            attention_mask
        )

        return logits, speculative_logits
