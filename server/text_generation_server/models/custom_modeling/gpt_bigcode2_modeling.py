# coding=utf-8
# Copyright 2023 The Bigcode team and HuggingFace Inc. team.
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
"""PyTorch GPTBigCode model."""
import math
from typing import Optional, Tuple, Any, Union, List
from enum import IntEnum

import torch
import torch.utils.checkpoint
from torch import nn

from dropout_layer_norm import dropout_layer_norm
from flash_attn.bert_padding import pad_input, unpad_input
from flash_attn.flash_attn_interface import flash_attn_unpadded_func

from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.models.gpt_bigcode.configuration_gpt_bigcode import (
    GPTBigCodeConfig,
)

class FastLayerNorm(nn.LayerNorm):
    # TODO: Validate dimension
    def forward(self, hidden_states, residual=None):
        return dropout_layer_norm.dropout_add_ln_fwd(
            hidden_states,
            residual,
            self.weight,
            self.bias,
            None,
            None,
            None,
            None,
            0.0,
            self.eps,
            1.0,
            0,
            None,
            False,
            False,
        )


class FastLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.addmm(self.bias, input, self.weight)

class FastLinearNoBias(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.mm(input, self.weight)

logger = logging.get_logger(__name__)

@torch.jit.script
def upcast_masked_softmax(
    x: torch.Tensor, mask: torch.Tensor, mask_value: torch.Tensor, scale: float
):
    input_dtype = x.dtype
    x = x.to(torch.float32) * scale
    x = torch.where(mask, x, mask_value)
    x = torch.nn.functional.softmax(x, dim=-1).to(input_dtype)
    return x

@torch.jit.script
def masked_softmax(x: torch.Tensor, mask: torch.Tensor, mask_value: torch.Tensor):
    x = torch.where(mask, x, mask_value)
    x = torch.nn.functional.softmax(x, dim=-1)
    return x

class GPTBigCodeAttention(nn.Module):
    def __init__(self, config:GPTBigCodeConfig, layer_idx:int, dtype:torch.dtype):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.layer_idx = layer_idx
        # Note: Does not support module dtype conversion.
        self.register_buffer("mask_value", torch.empty((), dtype=torch.float32, device="meta"))

        self.c_attn = FastLinear(self.embed_dim, self.embed_dim + 2 * self.head_dim, dtype=dtype, device="meta")
        self.c_proj = FastLinear(self.embed_dim, self.embed_dim, dtype=dtype, device="meta")

    def prefill(
        self,
        hidden_states: torch.Tensor,
        sequence_lengths,
        key_length:int,
    ) -> Tuple[torch.Tensor, Any]:
        hidden_shape = hidden_states.shape
        query, key_value = self.c_attn.forward(hidden_states).split((self.embed_dim, 2 * self.head_dim), dim=-1)
        query = query.view(hidden_shape[0], self.num_heads, self.head_dim)
        key, value = key_value.unsqueeze(1).expand(hidden_shape[0], self.num_heads, 2*self.head_dim).split((self.head_dim, self.head_dim), dim=-1)

        # attn_output: (sum_seq_len, num_heads * head_dim)
        attn_output = flash_attn_unpadded_func(
            query,
            key,
            value,
            sequence_lengths,
            sequence_lengths,
            key_length,
            key_length,
            0.0,
            softmax_scale=self.head_dim**-0.5,
            causal=True,
        ).view(hidden_shape)

        attn_output = self.c_proj.forward(attn_output)

        return attn_output, key_value

    def decode(
        self,
        hidden_states: torch.Tensor,
        layer_past: torch.Tensor,
        attention_mask: torch.Tensor,
        batch_size:int,
        key_length:int,
    ) -> Tuple[torch.Tensor, Any]:
        query, key_value = self.c_attn.forward(hidden_states).split((self.embed_dim, 2 * self.head_dim), dim=-1)

        # Calculate dimensions and recover layer_past
        padded_key_length = attention_mask.size(-1)
        allocated_key_length=layer_past.size(-2)

        # TODO: Allow pre-allocation with size > padded_key_length
        if padded_key_length > allocated_key_length:
            # Re-allocate kv cache and copy last value
            allocated_kv_cache = torch.empty(
                [batch_size, padded_key_length, 2 * self.head_dim],
                dtype=key_value.dtype,
                device=key_value.device,
            )
            allocated_kv_cache[:, :key_length-1].copy_(layer_past[:, :key_length-1])
            # Nans in `value` can propagate through the matrix multiplication,
            # so we set the remaining values to zero. (`last_key_length:key_length` is set below.)
            allocated_kv_cache[:, allocated_key_length:, self.head_dim :].zero_()
            layer_past=allocated_kv_cache

        # Copy the new values.
        layer_past[:, key_length-1:key_length].copy_(key_value)

        key, value = layer_past.split((self.head_dim, self.head_dim), dim=-1)

        # TODO: Upcasting needed for bf16?
        upcast = query.dtype != torch.float32
        unscale = self.layer_idx + 1 if upcast else 1
        scale_factor = unscale ** -1 / self.head_dim ** 0.5

        hidden_states = torch.baddbmm(
            torch.empty((batch_size, self.num_heads, padded_key_length), device=query.device, dtype=query.dtype),
            query.view(batch_size, self.num_heads, self.head_dim),
            key.transpose(-1, -2),
            beta=0,
            alpha=scale_factor
        ).unsqueeze_(1)

        if self.mask_value is None or self.mask_value.device != hidden_states.device:
            self.mask_value = torch.full([], torch.finfo(torch.float32).min, dtype=torch.float32, device=hidden_states.device)

        if upcast:
            hidden_states = upcast_masked_softmax(hidden_states, attention_mask, self.mask_value, unscale)
        else:
            hidden_states = masked_softmax(hidden_states, attention_mask, self.mask_value)

        hidden_states = torch.bmm(hidden_states.squeeze_(1), value).view(query.shape)

        hidden_states = self.c_proj.forward(hidden_states)

        return hidden_states, layer_past

class GPTBigCodeMLP(nn.Module):
    # TODO: Merge into GPTBigCodeBlock (needs renaming in state dict)
    def __init__(self, config:GPTBigCodeConfig, dtype:torch.dtype):
        super().__init__()
        embed_dim = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * embed_dim
        self.c_fc = FastLinear(embed_dim, inner_dim, dtype=dtype, device="meta")
        self.c_proj = FastLinear(inner_dim, embed_dim, dtype=dtype, device="meta")

class GPTBigCodeBlock(nn.Module):
    def __init__(self, config:GPTBigCodeConfig, layer_idx:int, dtype:torch.dtype):
        super().__init__()
        self.ln_1 = FastLayerNorm(config.hidden_size, eps=config.layer_norm_epsilon, dtype=dtype, device="meta")
        self.attn = GPTBigCodeAttention(config, layer_idx=layer_idx, dtype=dtype)
        self.ln_2 = FastLayerNorm(config.hidden_size, eps=config.layer_norm_epsilon, dtype=dtype, device="meta")
        self.mlp = GPTBigCodeMLP(config, dtype=dtype)

    def prefill(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        sequence_lengths,
        key_length:int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        hidden_states, residual, *_ = self.ln_1.forward(hidden_states, residual)
        hidden_states, present = self.attn.prefill(
            hidden_states,
            sequence_lengths=sequence_lengths,
            key_length=key_length,
        )
        hidden_states, residual, *_ = self.ln_2.forward(hidden_states, residual)
        hidden_states = self.mlp.c_proj.forward(nn.functional.gelu(self.mlp.c_fc.forward(hidden_states), approximate="tanh"))
        return hidden_states, residual, present

    def decode(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        layer_past: torch.Tensor,
        attention_mask: torch.Tensor,
        batch_size:int,
        key_length:int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        hidden_states, residual, *_ = self.ln_1.forward(hidden_states, residual)
        hidden_states, present = self.attn.decode(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            batch_size=batch_size,
            key_length=key_length,
        )
        hidden_states, residual, *_ = self.ln_2.forward(hidden_states, residual)
        hidden_states = self.mlp.c_proj.forward(nn.functional.gelu(self.mlp.c_fc.forward(hidden_states), approximate="tanh"))
        return hidden_states, residual, present


class GPTBigCodePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GPTBigCodeConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = False
    _no_split_modules = ["GPTBigCodeBlock"]

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, GPTBigCodeModel):
            module.bias.fill_(True).tril_()
        elif isinstance(module, (GPTBigCodeBlock, GPTBigCodeAttention)):
            if isinstance(module, GPTBigCodeAttention):
                module.mask_value.fill_(torch.finfo(torch.float32).min)
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            module.c_proj.weight.data.normal_(
                mean=0.0, std=(self.config.initializer_range / math.sqrt(2 * self.config.n_layer))
            )
            module.c_proj._is_hf_initialized = True
        elif isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class GPTBigCodeModel(GPTBigCodePreTrainedModel):
    # TODO: Merge into GPTBigCodeForCausalLM (needs renaming in state dict)
    def __init__(self, config:GPTBigCodeConfig, dtype:torch.dtype):
        super().__init__(config)

        self.wte = nn.Embedding(config.vocab_size, config.hidden_size, dtype=dtype, device="meta")
        self.wpe = nn.Embedding(config.max_position_embeddings, config.hidden_size, dtype=dtype, device="meta")

        self.h = nn.ModuleList([GPTBigCodeBlock(config, layer_idx=i, dtype=dtype) for i in range(config.num_hidden_layers)])
        self.ln_f = FastLayerNorm(config.hidden_size, dtype=dtype, device="meta", eps=config.layer_norm_epsilon)

        # Causal mask
        self.register_buffer(
            "causal_mask", torch.empty((config.max_position_embeddings, config.max_position_embeddings), dtype=torch.bool, device="meta")
        )

class GPTBigCodeForCausalLM(GPTBigCodePreTrainedModel):
    pad_key_length_to_multiple=8

    def __init__(self, config, dtype:torch.dtype, device:torch.device=torch.device("cuda")):
        super().__init__(config)
        if device.type!="cuda":
            raise NotImplementedError(f"Device {device} not supported")

        self.transformer = GPTBigCodeModel(config, dtype=dtype)
        self.lm_head = FastLinearNoBias(config.n_embd, config.vocab_size, bias=False, dtype=dtype, device="meta")

        self.to_empty(device=device)

        self._apply=self._apply_not_allowed

        # Initialize weights and apply final processing
        self.post_init()

    def _apply_not_allowed(self):
        # Dtype or device conversion would break the model.
        raise NotImplementedError("Device or dtype conversion not supported!")

    def prefill(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        position_ids: torch.Tensor,
        predict_all_tokens: bool=True,
    ) -> Tuple:
        batch_size, query_length = input_ids.shape

        hidden_states = self.transformer.wte.forward(input_ids) + self.transformer.wpe.forward(position_ids)

        # Prefill (flash attn)
        # TODO: Unpad earlier (input ids)?
        hidden_states, padding_index, sequence_lengths, key_length = unpad_input(hidden_states, attention_mask)
        assert key_length==query_length

        residual = None
        past_key_values = []
        block:GPTBigCodeBlock
        for block in self.transformer.h:
            hidden_states, residual, key_value = block.prefill(
                hidden_states,
                residual=residual,
                sequence_lengths=sequence_lengths,
                key_length=query_length,
            )
            past_key_values.append(pad_input(key_value, padding_index, batch_size, query_length))

        hidden_states = self.transformer.ln_f.forward(hidden_states, residual)

        # Next bit is the memory bottleneck with predict_all_tokens so we free as much memory as possible.
        del residual

        if predict_all_tokens:
            hidden_states = self.lm_head.forward(hidden_states)
            hidden_states = pad_input(hidden_states, padding_index, batch_size, query_length)
        else:
            # TODO: Index directly instead
            hidden_states = pad_input(hidden_states, padding_index, batch_size, query_length)[:, -1]
            hidden_states = self.lm_head.forward(hidden_states).unsqueeze_(1)

        return hidden_states, past_key_values

    def decode(
        self,
        *,
        input_ids: torch.Tensor,
        past_key_values: List[torch.Tensor],
        attention_mask: [torch.Tensor],
        position_ids: torch.Tensor,
        key_length:int,
    ) -> Tuple:

        batch_size, query_length = input_ids.shape
        assert query_length == 1

        hidden_states = self.transformer.wte.forward(input_ids) + self.transformer.wpe.forward(position_ids)

        # Standardize shape to (batch_size, hidden_size)
        hidden_states.squeeze_(1)

        # Self-attention mask (padding + causal).
        # TODO: Avoid unsqueeze
        attention_mask = self.transformer.causal_mask[None, key_length - 1: key_length,
                         :key_length] * attention_mask.unsqueeze(1)
        attention_mask.unsqueeze_(2)

        residual = None
        block:GPTBigCodeBlock
        for i, (block, layer_past) in enumerate(zip(self.transformer.h, past_key_values)):
            hidden_states, residual, past_key_values[i] = block.decode(
                hidden_states,
                residual=residual,
                layer_past=layer_past,
                attention_mask=attention_mask,
                batch_size=batch_size,
                key_length=key_length,
            )

        hidden_states = self.transformer.ln_f.forward(hidden_states, residual)
        hidden_states = self.lm_head.forward(hidden_states).unsqueeze_(1)

        return hidden_states, past_key_values
