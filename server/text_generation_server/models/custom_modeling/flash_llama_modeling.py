# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
from transformers.activations import ACT2FN
from typing import Optional

# Flash attention imports
import flash_attn_cuda
import dropout_layer_norm

from text_generation_server.utils.layers import (
    FastLinear,
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    PositionRotaryEmbedding,
)


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, residual=None):
        if hidden_states.shape[-1] > 8192:
            if residual is not None:
                hidden_states += residual
            residual = hidden_states

            variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(
                variance + self.variance_epsilon
            )

            # convert into half-precision if necessary
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                hidden_states = hidden_states.to(self.weight.dtype)

            return self.weight * hidden_states, residual
        else:
            # faster post attention rms norm
            normed_hidden_states, res, *rest = dropout_layer_norm.dropout_add_ln_fwd(
                hidden_states,
                residual,
                self.weight,
                None,
                None,
                None,
                None,
                None,
                0.0,
                self.variance_epsilon,
                1.0,
                0,
                None,
                False,
                True,  # Activate RMSNorm
            )
            if res is None:
                res = hidden_states

            return normed_hidden_states, res


class FlashLlamaAttention(torch.nn.Module):
    def __init__(
        self,
        num_heads,
        hidden_size,
        process_group=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_size = hidden_size // num_heads

        self.rotary_emb = PositionRotaryEmbedding(self.head_size, base=10000)
        self.softmax_scale = self.head_size ** (-0.5)

        if process_group is None:
            self.query_key_value = FastLinear(hidden_size, 3 * hidden_size, bias=False)
            self.o_proj = FastLinear(hidden_size, hidden_size, bias=False)
        else:
            self.num_heads = self.num_heads // process_group.size()
            self.query_key_value = TensorParallelColumnLinear(
                hidden_size,
                3 * hidden_size,
                bias=False,
                process_group=process_group,
            )
            self.o_proj = TensorParallelRowLinear(
                hidden_size,
                hidden_size,
                bias=False,
                process_group=process_group,
            )

    def forward(
        self,
        hidden_states,
        cos,
        sin,
        cu_seqlens,
        max_s,
        layer_past,
        layer_past_present_indices,
        cu_seqlens_q,
    ):
        qkv = self.query_key_value(hidden_states)
        qkv = qkv.view(-1, 3, self.num_heads, self.head_size)

        # Inplace rotary
        self.rotary_emb(qkv[:, 0], cos, sin)
        self.rotary_emb(qkv[:, 1], cos, sin)

        # Prefill
        if layer_past_present_indices is None:
            # Copy to layer past
            layer_past[...] = qkv[:, 1:]

            # output
            attn_output = torch.empty_like(qkv[:, 0])
            # flash attention
            flash_attn_cuda.fwd(
                qkv[:, 0],
                qkv[:, 1],
                qkv[:, 2],
                attn_output,
                cu_seqlens,
                cu_seqlens,
                max_s,
                max_s,
                0.0,
                self.softmax_scale,
                False,
                True,
                False,
                0,
                None,
            )
        # Decode
        else:
            query = qkv[:, 0]
            # Add present to the layer_past tensor at the correct indices
            layer_past[layer_past_present_indices] = qkv[:, 1:]

            # output
            attn_output = torch.empty_like(query)
            # flash attention
            flash_attn_cuda.fwd(
                query,
                layer_past[:, 0],
                layer_past[:, 1],
                attn_output,
                cu_seqlens_q,
                cu_seqlens,
                1,
                max_s,
                0.0,
                self.softmax_scale,
                False,
                False,
                False,
                0,
                None,
            )

        return self.o_proj(attn_output.view(-1, self.num_heads * self.head_size))


class LlamaMLP(nn.Module):
    def __init__(self, act, hidden_size, intermediate_size, process_group=None):
        super().__init__()
        self.act = (
            ACT2FN[act]
            if "gelu" not in act
            else lambda x: torch.nn.functional.gelu(
                x,
                approximate="tanh"
                if act in ["gelu_fast", "gelu_pytorch_tanh"]
                else "none",
            )
        )

        if process_group is None:
            # Fuse gate and up proj
            self.gate_up_proj = FastLinear(
                hidden_size, 2 * intermediate_size, bias=False
            )
            self.down_proj = FastLinear(intermediate_size, hidden_size, bias=False)
            self.intermediate_size = intermediate_size
        else:
            # Fuse gate and up proj
            self.gate_up_proj = TensorParallelColumnLinear(
                hidden_size,
                2 * intermediate_size,
                bias=False,
                process_group=process_group,
            )
            self.down_proj = TensorParallelRowLinear(
                intermediate_size,
                hidden_size,
                bias=False,
                process_group=process_group,
                reduce=True,
            )
            self.intermediate_size = self.down_proj.in_features

        self.process_group = process_group

    def forward(self, hidden_states):
        gate_up_states = self.gate_up_proj(hidden_states)
        gate_up_states = gate_up_states.view(-1, 2, self.intermediate_size)
        return self.down_proj(self.act(gate_up_states[:, 0]) * gate_up_states[:, 1])


class FlashLlamaLayer(nn.Module):
    def __init__(
        self,
        num_heads,
        act,
        hidden_size,
        intermediate_size,
        rms_norm_eps,
        process_group=None,
    ):
        super().__init__()

        self.self_attn = FlashLlamaAttention(num_heads, hidden_size, process_group)
        self.mlp = LlamaMLP(act, hidden_size, intermediate_size, process_group)

        self.input_layernorm = LlamaRMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        hidden_states,
        residual,
        cos,
        sin,
        cu_seqlens,
        max_s,
        layer_past,
        layer_past_present_indices,
        cu_seqlens_q,
    ):
        normed_hidden_states, res = self.input_layernorm(hidden_states, residual)

        # Self Attention
        attn_output = self.self_attn(
            normed_hidden_states,
            cos,
            sin,
            cu_seqlens,
            max_s,
            layer_past,
            layer_past_present_indices,
            cu_seqlens_q,
        )

        # faster post attention rms norm
        normed_attn_res_output, attn_res = self.post_attention_layernorm(
            attn_output, res
        )

        mlp_output = self.mlp(normed_attn_res_output)

        return mlp_output, attn_res


class FlashLlamaModel(torch.nn.Module):
    def __init__(self, config, process_group=None):
        super(FlashLlamaModel, self).__init__()
        self.config = config

        self.tp_embeddings = False
        if process_group is not None:
            self.tp_rank = process_group.rank()
            self.tp_world_size = process_group.size()
            if config.vocab_size % self.tp_world_size == 0:
                self.tp_embeddings = True

        if self.tp_embeddings:
            self.embed_tokens = TensorParallelEmbedding(
                config.vocab_size, config.hidden_size, process_group=process_group
            )
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        self.layers = nn.ModuleList(
            [
                FlashLlamaLayer(
                    config.num_attention_heads,
                    config.hidden_act,
                    config.hidden_size,
                    config.intermediate_size,
                    config.rms_norm_eps,
                    process_group,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False

        self.head_size = self.layers[0].self_attn.head_size
        self.num_heads = self.layers[0].self_attn.num_heads

    def post_load_weights(self, quantize: Optional[str] = None):
        if isinstance(self.embed_tokens, TensorParallelEmbedding):
            self.embed_tokens.add_null_idx()
        for layer in self.layers:
            layer: FlashLlamaLayer
            layer.self_attn.query_key_value.prepare_weights(quantize)
            layer.self_attn.o_proj.prepare_weights(quantize)
            layer.mlp.gate_up_proj.prepare_weights(quantize)
            layer.mlp.down_proj.prepare_weights(quantize)

    def forward(
        self,
        input_ids,
        position_ids,
        cu_seqlens,
        cu_seqlens_q,
        max_s,
        past_key_values: Optional[torch.Tensor] = None,
        pre_allocate_past_size: Optional[int] = None,
    ):
        hidden_states = self.embed_tokens(input_ids)

        # Prefill
        if past_key_values is None:
            # Create past tensor
            past_key_values = hidden_states.new_empty(
                (
                    len(self.layers),
                    len(hidden_states)
                    if pre_allocate_past_size is None
                    else pre_allocate_past_size,
                    2,
                    self.num_heads,
                    self.head_size,
                )
            )
            layer_past_present_indices = None
            slice_past_index = len(hidden_states)
        # Decode
        else:
            # Create indices from cumulative sequence lengths
            layer_past_present_indices = cu_seqlens[1:] - 1
            slice_past_index = None

        # Get rotary cos and sin for this forward
        # Avoid to index in each layer
        cos, sin = self.layers[0].self_attn.rotary_emb.get_cos_sin(
            position_ids, max_s, hidden_states.dtype
        )

        residual = None
        for i, layer in enumerate(self.layers):
            # We added padding that we now need to slice
            layer_past_key_values = (
                past_key_values[i]
                if slice_past_index is None
                else past_key_values[i, :slice_past_index]
            )

            hidden_states, residual = layer(
                hidden_states,
                residual,
                cos,
                sin,
                cu_seqlens,
                max_s,
                layer_past_key_values,
                layer_past_present_indices,
                cu_seqlens_q,
            )

        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states, past_key_values


class FlashLlamaForCausalLM(torch.nn.Module):
    def __init__(self, config, process_group=None):
        super().__init__()

        self.process_group = process_group
        if self.process_group is not None:
            self.world_size = self.process_group.size()
        else:
            self.world_size = 1

        self.model = FlashLlamaModel(config, process_group)

        if self.model.tp_embeddings:
            self.lm_head = FastLinear(
                config.hidden_size,
                config.vocab_size // process_group.size(),
                bias=False,
            )
        else:
            self.lm_head = FastLinear(config.hidden_size, config.vocab_size, bias=False)

    def post_load_weights(self, quantize: Optional[str] = None):
        self.model.post_load_weights(quantize)
        self.lm_head.prepare_weights()

    def forward(
        self,
        input_ids,
        position_ids,
        cu_seqlens,
        cu_seqlens_q,
        max_s,
        past_key_values: Optional[torch.Tensor] = None,
        pre_allocate_past_size: Optional[int] = None,
        lm_head_indices: Optional[torch.Tensor] = None,
    ):
        hidden_states, present = self.model(
            input_ids,
            position_ids,
            cu_seqlens,
            cu_seqlens_q,
            max_s,
            past_key_values,
            pre_allocate_past_size,
        )
        if lm_head_indices is not None:
            hidden_states = hidden_states[lm_head_indices]
        logits = self.lm_head(hidden_states)

        if self.model.tp_embeddings:
            # Logits are sharded, so we need to gather them
            world_logits = [torch.empty_like(logits) for _ in range(self.world_size)]
            torch.distributed.all_gather(world_logits, logits, group=self.process_group)
            world_logits = torch.cat(world_logits, dim=1)

            return world_logits, present
        return logits, present
