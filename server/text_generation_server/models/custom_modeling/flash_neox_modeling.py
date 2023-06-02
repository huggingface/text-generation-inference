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
    FastLinear,
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    FastLayerNorm,
    PositionRotaryEmbedding,
)


class FlashNeoxAttention(torch.nn.Module):
    def __init__(
        self,
        num_heads,
        hidden_size,
        rotary_pct,
        rotary_emb_base,
        process_group=None,
        reduce=True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_size = hidden_size // num_heads

        rotary_ndims = int(self.head_size * rotary_pct)
        self.rotary_emb = PositionRotaryEmbedding(rotary_ndims, base=rotary_emb_base)
        self.softmax_scale = self.head_size ** (-0.5)

        if process_group is None:
            self.query_key_value = FastLinear(hidden_size, 3 * hidden_size)
            self.dense = FastLinear(hidden_size, hidden_size)
        else:
            self.num_heads = self.num_heads // process_group.size()
            self.query_key_value = TensorParallelColumnLinear(
                hidden_size,
                3 * hidden_size,
                process_group=process_group,
            )
            self.dense = TensorParallelRowLinear(
                hidden_size, hidden_size, process_group=process_group, reduce=reduce
            )

    def shuffle_qkv_dims(self):
        """Swap dims to avoid an additional permute"""
        self.query_key_value.weight = torch.nn.Parameter(
            self.query_key_value.weight.view(
                self.num_heads, 3, self.head_size, self.hidden_size
            )
            .permute(1, 0, 2, 3)
            .reshape(-1, self.hidden_size)
        )
        self.query_key_value.bias = torch.nn.Parameter(
            self.query_key_value.bias.view(self.num_heads, 3, self.head_size)
            .permute(1, 0, 2)
            .reshape(-1)
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

        return self.dense(attn_output.view(-1, self.num_heads * self.head_size))


class FlashMLP(nn.Module):
    def __init__(
        self, act, hidden_size, intermediate_size, process_group=None, reduce=True
    ):
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
            self.dense_h_to_4h = FastLinear(hidden_size, intermediate_size)
            self.dense_4h_to_h = FastLinear(intermediate_size, hidden_size)
        else:
            self.dense_h_to_4h = TensorParallelColumnLinear(
                hidden_size,
                intermediate_size,
                process_group=process_group,
            )
            self.dense_4h_to_h = TensorParallelRowLinear(
                intermediate_size,
                hidden_size,
                process_group=process_group,
                reduce=reduce,
            )
        self.process_group = process_group

    def forward(self, hidden_states):
        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dense_4h_to_h(hidden_states)
        return hidden_states


class FlashNeoXLayer(nn.Module):
    def __init__(
        self,
        num_heads,
        act,
        hidden_size,
        intermediate_size,
        rotary_pct,
        rotary_emb_base,
        layer_norm_eps,
        use_parallel_residual,
        process_group=None,
    ):
        super().__init__()
        self.use_parallel_residual = use_parallel_residual
        self.input_layernorm = FastLayerNorm(hidden_size, eps=layer_norm_eps)
        self.post_attention_layernorm = FastLayerNorm(hidden_size, eps=layer_norm_eps)
        self.attention = FlashNeoxAttention(
            num_heads,
            hidden_size,
            rotary_pct,
            rotary_emb_base,
            process_group,
            reduce=not use_parallel_residual,
        )
        self.mlp = FlashMLP(
            act,
            hidden_size,
            intermediate_size,
            process_group,
            reduce=not use_parallel_residual,
        )
        self.process_group = process_group

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
        if self.use_parallel_residual:
            ln1_hidden_states, _ = self.input_layernorm(hidden_states)

            attn_output = self.attention(
                ln1_hidden_states,
                cos,
                sin,
                cu_seqlens,
                max_s,
                layer_past,
                layer_past_present_indices,
                cu_seqlens_q,
            )

            ln2_hidden_states, _ = self.post_attention_layernorm(hidden_states)

            mlp_output = self.mlp(ln2_hidden_states)
            intermediate = mlp_output + attn_output

            # Only reduce once and after the addition instead of once per layer
            if self.process_group is not None:
                torch.distributed.all_reduce(intermediate, group=self.process_group)

            return intermediate + hidden_states, None
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

            hidden_states = self.attention(
                hidden_states,
                cos,
                sin,
                cu_seqlens,
                max_s,
                layer_past,
                layer_past_present_indices,
                cu_seqlens_q,
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
    def __init__(self, config, process_group=None):
        super().__init__(config)
        self.config = config

        self.tp_embeddings = False
        if process_group is not None:
            self.tp_rank = process_group.rank()
            self.tp_world_size = process_group.size()
            if config.vocab_size % self.tp_world_size == 0:
                self.tp_embeddings = True

        if self.tp_embeddings:
            self.embed_in = TensorParallelEmbedding(
                config.vocab_size, config.hidden_size, process_group=process_group
            )
        else:
            self.embed_in = nn.Embedding(config.vocab_size, config.hidden_size)

        self.layers = nn.ModuleList(
            [
                FlashNeoXLayer(
                    config.num_attention_heads,
                    config.hidden_act,
                    config.hidden_size,
                    config.intermediate_size,
                    config.rotary_pct,
                    config.rotary_emb_base,
                    config.layer_norm_eps,
                    config.use_parallel_residual,
                    process_group,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.final_layer_norm = FastLayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

        self.gradient_checkpointing = False

        self.head_size = self.layers[0].attention.head_size
        self.num_heads = self.layers[0].attention.num_heads

    def post_load_weights(self, quantize: Optional[str] = None):
        if isinstance(self.embed_in, TensorParallelEmbedding):
            self.embed_in.add_null_idx()
        for layer in self.layers:
            layer: FlashNeoXLayer
            layer.attention.shuffle_qkv_dims()
            layer.attention.query_key_value.prepare_weights(quantize)
            layer.attention.dense.prepare_weights(quantize)
            layer.mlp.dense_h_to_4h.prepare_weights(quantize)
            layer.mlp.dense_4h_to_h.prepare_weights(quantize)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # Pop here as we will replace the layer in our own logic and don't want from_pretrained
        # to do it for us
        load_in_8bit = kwargs.pop("load_in_8bit", False)
        model = super(FlashGPTNeoXModel, cls).from_pretrained(
            pretrained_model_name_or_path, load_in_8bit=False, *model_args, **kwargs
        )

        model.post_load_weights("bitsandbytes" if load_in_8bit else None)
        return model

    def forward(
        self,
        input_ids,
        position_ids,
        cu_seqlens,
        cu_seqlens_q,
        max_s,
        past_key_values=None,
        pre_allocate_past_size: Optional[int] = None,
    ):
        hidden_states = self.embed_in(input_ids)

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
        cos, sin = self.layers[0].attention.rotary_emb.get_cos_sin(
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

        hidden_states, _ = self.final_layer_norm(hidden_states, residual)

        return hidden_states, past_key_values


class FlashGPTNeoXForCausalLM(FlashGPTNeoXPreTrainedModel):
    def __init__(self, config, process_group=None):
        super().__init__(config)

        self.process_group = process_group
        if self.process_group is not None:
            self.world_size = self.process_group.size()
        else:
            self.world_size = 1

        self.gpt_neox = FlashGPTNeoXModel(config, process_group)

        if self.gpt_neox.tp_embeddings:
            self.embed_out = FastLinear(
                config.hidden_size,
                config.vocab_size // process_group.size(),
                bias=False,
            )
        else:
            self.embed_out = FastLinear(
                config.hidden_size, config.vocab_size, bias=False
            )

    def post_load_weights(self, quantize: Optional[str] = None):
        self.gpt_neox.post_load_weights(quantize)
        self.embed_out.prepare_weights()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # Pop here as we will replace the layer in our own logic and don't want from_pretrained
        # to do it for us
        load_in_8bit = kwargs.pop("load_in_8bit", False)
        model = super(FlashGPTNeoXForCausalLM, cls).from_pretrained(
            pretrained_model_name_or_path, load_in_8bit=False, *model_args, **kwargs
        )
        model.post_load_weights("bitsandbytes" if load_in_8bit else None)
        return model

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
        hidden_states, present = self.gpt_neox(
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
        logits = self.embed_out(hidden_states)

        if self.gpt_neox.tp_embeddings:
            # Logits are sharded, so we need to gather them
            world_logits = [torch.empty_like(logits) for _ in range(self.world_size)]
            torch.distributed.all_gather(world_logits, logits, group=self.process_group)
            world_logits = torch.cat(world_logits, dim=1)

            return world_logits, present
        return logits, present
