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
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    PositionRotaryEmbedding,
    TensorParallelHead,
)


class LlamaRMSNorm(nn.Module):
    def __init__(self, prefix, weights, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()

        weight = weights.get_tensor(f"{prefix}.weight")
        self.weight = nn.Parameter(weight)
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
        prefix: str,
        config,
        weights,
    ):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_heads

        self.rotary_emb = PositionRotaryEmbedding.load(
            prefix=f"{prefix}.rotary_emb", weights=weights
        )

        self.softmax_scale = self.head_size ** (-0.5)

        self.num_heads = self.num_heads // weights.process_group.size()
        self.query_key_value = TensorParallelColumnLinear.load_multi(
            config,
            prefixes=[f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"],
            dim=0,
            weights=weights,
            bias=False,
        )
        self.o_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.o_proj",
            weights=weights,
            bias=False,
        )

    def forward(
        self,
        hidden_states,
        cos,
        sin,
        start_seq,
        end_seq,
        start_seq_q,
        end_seq_q,
        max_s,
        layer_past,
        past_present_indices,
        prefill,
    ):
        qkv = self.query_key_value(hidden_states)
        qkv = qkv.view(-1, 3, self.num_heads, self.head_size)

        # Inplace rotary
        self.rotary_emb(qkv[:, 0], cos, sin)
        self.rotary_emb(qkv[:, 1], cos, sin)

        # Prefill
        if prefill:
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
                start_seq,
                end_seq,
                start_seq,
                end_seq,
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
            layer_past[past_present_indices] = qkv[:, 1:]

            # output
            attn_output = torch.empty_like(query)
            # flash attention
            flash_attn_cuda.fwd(
                query,
                layer_past[:, 0],
                layer_past[:, 1],
                attn_output,
                start_seq_q,
                end_seq_q,
                start_seq,
                end_seq,
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
    def __init__(self, prefix, config, weights):
        super().__init__()
        act = config.hidden_act
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

    def forward(self, hidden_states):
        gate_up_states = self.gate_up_proj(hidden_states)
        gate_up_states = gate_up_states.view(-1, 2, self.intermediate_size)
        return self.down_proj(self.act(gate_up_states[:, 0]) * gate_up_states[:, 1])


class FlashLlamaLayer(nn.Module):
    def __init__(self, layer_id, config, weights):
        super().__init__()
        prefix = f"model.layers.{layer_id}"
        self.self_attn = FlashLlamaAttention(
            prefix=f"{prefix}.self_attn", config=config, weights=weights
        )
        self.mlp = LlamaMLP(prefix=f"{prefix}.mlp", config=config, weights=weights)

        self.input_layernorm = LlamaRMSNorm(
            prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = LlamaRMSNorm(
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
        start_seq,
        end_seq,
        start_seq_q,
        end_seq_q,
        max_s,
        layer_past,
        past_present_indices,
        prefill,
    ):
        normed_hidden_states, res = self.input_layernorm(hidden_states, residual)

        # Self Attention
        attn_output = self.self_attn(
            normed_hidden_states,
            cos,
            sin,
            start_seq,
            end_seq,
            start_seq_q,
            end_seq_q,
            max_s,
            layer_past,
            past_present_indices,
            prefill,
        )

        # faster post attention rms norm
        normed_attn_res_output, attn_res = self.post_attention_layernorm(
            attn_output, res
        )

        mlp_output = self.mlp(normed_attn_res_output)

        return mlp_output, attn_res


class FlashLlamaModel(torch.nn.Module):
    def __init__(self, config, weights):
        super().__init__()
        self.config = config

        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.embed_tokens = TensorParallelEmbedding(
            prefix="model.embed_tokens", weights=weights
        )
        self.layers = nn.ModuleList(
            [
                FlashLlamaLayer(
                    layer_id,
                    config,
                    weights,
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = LlamaRMSNorm(
            prefix="model.norm", weights=weights, eps=config.rms_norm_eps
        )

        self.gradient_checkpointing = False

        self.head_size = self.layers[0].self_attn.head_size
        self.num_heads = self.layers[0].self_attn.num_heads

    def forward(
        self,
        input_ids,
        position_ids,
        start_seq,
        end_seq,
        start_seq_q,
        end_seq_q,
        max_s,
        past_present_indices,
        past_key_values=None,
        pre_allocate_past_size: Optional[int] = None,
    ):
        hidden_states = self.embed_tokens(input_ids)

        # Prefill
        if past_key_values is None:
            assert pre_allocate_past_size is not None

            prefill = True

            # Create past tensor
            # We create a tensor of the same size as input_ids as we don't want to slice at every layer
            past_key_values = hidden_states.new_empty(
                (
                    len(input_ids),
                    len(self.layers),
                    2,
                    self.num_heads,
                    self.head_size,
                )
            )
        # Decode
        else:
            prefill = False

        # Get rotary cos and sin for this forward
        # Avoid to index in each layer
        cos, sin = self.layers[0].self_attn.rotary_emb.get_cos_sin(
            position_ids, max_s, hidden_states.dtype
        )

        residual = None
        for i, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                hidden_states,
                residual,
                cos,
                sin,
                start_seq,
                end_seq,
                start_seq_q,
                end_seq_q,
                max_s,
                past_key_values[:, i],
                past_present_indices,
                prefill,
            )

        if prefill:
            present = past_key_values
            # Create padded past tensor
            past_key_values = hidden_states.new_empty(
                (
                    pre_allocate_past_size,
                    len(self.layers),
                    2,
                    self.num_heads,
                    self.head_size,
                )
            )
            # We slice only once instead of at every layer
            past_key_values[past_present_indices] = present

        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states, past_key_values


class FlashLlamaForCausalLM(torch.nn.Module):
    def __init__(self, config, weights):
        super().__init__()

        self.model = FlashLlamaModel(config, weights)
        self.lm_head = TensorParallelHead.load(
            config,
            prefix="lm_head",
            weights=weights,
        )

    def forward(
        self,
        input_ids,
        position_ids,
        start_seq,
        end_seq,
        start_seq_q,
        end_seq_q,
        max_s,
        past_present_indices,
        past_key_values: Optional[torch.Tensor] = None,
        pre_allocate_past_size: Optional[int] = None,
        lm_head_indices: Optional[torch.Tensor] = None,
    ):
        hidden_states, present = self.model(
            input_ids,
            position_ids,
            start_seq,
            end_seq,
            start_seq_q,
            end_seq_q,
            max_s,
            past_present_indices,
            past_key_values,
            pre_allocate_past_size,
        )
        if lm_head_indices is not None:
            hidden_states = hidden_states[lm_head_indices]
        logits = self.lm_head(hidden_states)
        return logits, present
