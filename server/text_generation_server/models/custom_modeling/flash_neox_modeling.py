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
from transformers.modeling_utils import PreTrainedModel
from transformers.models.gpt_neox import GPTNeoXConfig
from typing import Optional

# Flash attention imports
import flash_attn_cuda

from text_generation_server.utils.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    TensorParallelHead,
    FastLayerNorm,
    PositionRotaryEmbedding,
    get_linear,
)


def load_row(config, prefix: str, weights, bias: bool):
    weight = weights.get_sharded(f"{prefix}.weight", dim=1)
    if bias and weights.process_group.rank() == 0:
        # Rank is only on the first rank process
        bias = weights.get_tensor(f"{prefix}.bias")
    else:
        bias = None

    linear = get_linear(weight, bias, config.quantize)
    if config.use_parallel_residual:
        return linear
    else:
        return TensorParallelRowLinear(linear, process_group=weights.process_group)


def load_qkv(config, prefix: str, weights, num_heads, head_size, hidden_size):
    weight = weights.get_sharded(f"{prefix}.weight", dim=0)
    bias = weights.get_sharded(f"{prefix}.bias", dim=0)

    weight = (
        weight.view(
            num_heads,
            3,
            head_size,
            hidden_size,
        )
        .permute(1, 0, 2, 3)
        .reshape(-1, hidden_size)
    )
    bias = bias.view(num_heads, 3, head_size).permute(1, 0, 2).reshape(-1)

    linear = get_linear(weight, bias, config.quantize)
    if config.use_parallel_residual:
        return linear
    else:
        return TensorParallelColumnLinear(linear)


class FlashNeoxAttention(torch.nn.Module):
    def __init__(self, config, prefix, weights):
        super().__init__()
        num_heads = config.num_attention_heads
        hidden_size = config.hidden_size

        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_size = hidden_size // num_heads
        self.num_heads = self.num_heads // weights.process_group.size()

        self.rotary_emb = PositionRotaryEmbedding.load(
            prefix=f"{prefix}.rotary_emb", weights=weights
        )

        self.softmax_scale = self.head_size ** (-0.5)

        self.query_key_value = load_qkv(
            config,
            prefix=f"{prefix}.query_key_value",
            weights=weights,
            num_heads=self.num_heads,
            head_size=self.head_size,
            hidden_size=self.hidden_size,
        )
        self.dense = load_row(
            config, prefix=f"{prefix}.dense", weights=weights, bias=True
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

        return self.dense(attn_output.view(-1, self.num_heads * self.head_size))


class FlashMLP(nn.Module):
    def __init__(self, config, prefix, weights):
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

        self.dense_h_to_4h = TensorParallelColumnLinear.load(
            config, prefix=f"{prefix}.dense_h_to_4h", weights=weights, bias=True
        )
        self.dense_4h_to_h = load_row(
            config, prefix=f"{prefix}.dense_4h_to_h", weights=weights, bias=True
        )

    def forward(self, hidden_states):
        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dense_4h_to_h(hidden_states)
        return hidden_states


class FlashNeoXLayer(nn.Module):
    def __init__(self, layer_id, config, weights):
        super().__init__()

        layer_norm_eps = config.layer_norm_eps

        prefix = f"gpt_neox.layers.{layer_id}"

        self.use_parallel_residual = config.use_parallel_residual
        self.input_layernorm = FastLayerNorm.load(
            prefix=f"{prefix}.input_layernorm", weights=weights, eps=layer_norm_eps
        )
        self.post_attention_layernorm = FastLayerNorm.load(
            prefix=f"{prefix}.post_attention_layernorm",
            weights=weights,
            eps=layer_norm_eps,
        )
        self.attention = FlashNeoxAttention(
            config, prefix=f"{prefix}.attention", weights=weights
        )

        self.mlp = FlashMLP(config, prefix=f"{prefix}.mlp", weights=weights)
        self.process_group = weights.process_group

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
        if self.use_parallel_residual:
            ln1_hidden_states, _ = self.input_layernorm(hidden_states)

            attn_output = self.attention(
                ln1_hidden_states,
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

            ln2_hidden_states, _ = self.post_attention_layernorm(hidden_states)

            mlp_output = self.mlp(ln2_hidden_states)
            intermediate = mlp_output + attn_output

            if self.process_group.size() > 1:
                torch.distributed.all_reduce(intermediate, group=self.process_group)

            return intermediate + hidden_states, None
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

            hidden_states = self.attention(
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
            )

            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )

            mlp_output = self.mlp(hidden_states)

            return mlp_output, residual


class FlashGPTNeoXPreTrainedModel(PreTrainedModel):
    config_class = GPTNeoXConfig
    base_model_prefix = "gpt_neox"
    supports_gradient_checkpointing = False
    _no_split_modules = None


class FlashGPTNeoXModel(FlashGPTNeoXPreTrainedModel):
    def __init__(self, config, weights):
        super().__init__(config)
        self.config = config

        self.embed_in = TensorParallelEmbedding(
            prefix="gpt_neox.embed_in", weights=weights
        )

        self.layers = nn.ModuleList(
            [
                FlashNeoXLayer(layer_id, config, weights)
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.final_layer_norm = FastLayerNorm.load(
            prefix="gpt_neox.final_layer_norm",
            weights=weights,
            eps=config.layer_norm_eps,
        )

        self.gradient_checkpointing = False

        self.head_size = self.layers[0].attention.head_size
        self.num_heads = self.layers[0].attention.num_heads

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
        hidden_states = self.embed_in(input_ids)

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
        cos, sin = self.layers[0].attention.rotary_emb.get_cos_sin(
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

        hidden_states, _ = self.final_layer_norm(hidden_states, residual)

        return hidden_states, past_key_values


class FlashGPTNeoXForCausalLM(FlashGPTNeoXPreTrainedModel):
    def __init__(self, config, weights):
        super().__init__(config)
        self.gpt_neox = FlashGPTNeoXModel(config, weights)

        self.embed_out = TensorParallelHead.load(
            config, prefix="embed_out", weights=weights
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
        hidden_states, present = self.gpt_neox(
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
        logits = self.embed_out(hidden_states)
        return logits, present
