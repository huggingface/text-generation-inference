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
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.models.gpt_bigcode.configuration_gpt_bigcode import (
    GPTBigCodeConfig,
    InferenceRunnerType,
)


try:
    from flash_attn.bert_padding import pad_input, unpad_input
    from flash_attn.flash_attn_interface import flash_attn_unpadded_func
except ImportError:
    flash_attn_unpadded_func = None
    pad_input = None
    unpad_input = None


logger = logging.get_logger(__name__)

def upcast_masked_softmax(
    x: torch.Tensor, mask: torch.Tensor, mask_value: torch.Tensor, scale: float, softmax_dtype: torch.dtype
):
    input_dtype = x.dtype
    x = x.to(softmax_dtype) * scale
    x = torch.where(mask, x, mask_value)
    x = torch.nn.functional.softmax(x, dim=-1).to(input_dtype)
    return x


@torch.jit.script
def upcast_masked_softmax_fused(
    x: torch.Tensor, mask: torch.Tensor, mask_value: torch.Tensor, scale: float, softmax_dtype: torch.dtype
):
    return upcast_masked_softmax(x, mask, mask_value, scale, softmax_dtype)


def upcast_softmax(x: torch.Tensor, scale: float, softmax_dtype: torch.dtype):
    input_dtype = x.dtype
    x = x.to(softmax_dtype) * scale
    x = torch.nn.functional.softmax(x, dim=-1).to(input_dtype)
    return x


@torch.jit.script
def upcast_softmax_fused(x: torch.Tensor, scale: float, softmax_dtype: torch.dtype):
    return upcast_softmax(x, scale, softmax_dtype)


def masked_softmax(x: torch.Tensor, mask: torch.Tensor, mask_value: torch.Tensor):
    x = torch.where(mask, x, mask_value)
    x = torch.nn.functional.softmax(x, dim=-1)
    return x


@torch.jit.script
def masked_softmax_fused(x: torch.Tensor, mask: torch.Tensor, mask_value: torch.Tensor):
    return masked_softmax(x, mask, mask_value)


def softmax_function(
    x: torch.Tensor,
    mask: torch.Tensor,
    mask_value: torch.Tensor,
    scale: float,
    softmax_dtype: torch.dtype,
    upcast: bool = True,
    fused_softmax: Optional[bool] = None,
):
    """
    This selects the appropriate (fused) (upcast) (masked) softmax method. Because of the way jit works, each case
    needs to be handled through a separate method. The fused kernels remove most of the overhead from masking, casting
    and scaling, but only work well when the key length is a multiple of 8. For other key lengths, it is extremely
    inefficient. TODO: Could have better fused kernels depending on scaling, dropout and head mask.
     Is it doable without writing 32 functions?
    """
    if fused_softmax is None:
        fused_softmax = x.size(-1) % 8 == 0
    if upcast:
        if mask is None:
            return (upcast_softmax_fused if fused_softmax else upcast_softmax)(x, scale, softmax_dtype)
        else:
            return (upcast_masked_softmax_fused if fused_softmax else upcast_masked_softmax)(
                x, mask, mask_value, scale, softmax_dtype
            )
    else:
        if mask is None:
            return torch.nn.functional.softmax(x, dim=-1)
        else:
            return (masked_softmax_fused if fused_softmax else masked_softmax)(x, mask, mask_value)


class GPTBigCodeAttention(nn.Module):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__()
        self.mask_value = None

        self.multi_query = config.multi_query
        self.seq_dim = -2 if self.multi_query else -1
        self.flash_attention = config.flash_attention
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.kv_heads = 1 if self.multi_query else self.num_heads
        self.kv_dim = self.kv_heads * self.head_dim
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention

        self.layer_idx = layer_idx
        self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32
        self.scale_attention_softmax_in_fp32 = (
            config.scale_attention_softmax_in_fp32 and config.attention_softmax_in_fp32
        )
        self.fused_softmax = config.fused_softmax

        # KV caching and padding
        self.pre_allocate_kv_cache = (
            config.n_embd if config.pre_allocate_kv_cache is True else config.pre_allocate_kv_cache
        )
        pad_key_length = config.pre_allocate_kv_cache if config.pad_key_length is None else config.pad_key_length
        self._tuple_cache_format = self.pre_allocate_kv_cache or pad_key_length or self.flash_attention

        if self.is_cross_attention:
            raise NotImplementedError("Cross-attention is not supported for gpt_bigcode.")
        self.c_attn = nn.Linear(self.embed_dim, self.embed_dim + 2 * self.kv_dim)

        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        if self.flash_attention:
            if flash_attn_unpadded_func is None:
                raise RuntimeError(
                    "Flash Attention requires `flash_attn` and `einops`. "
                    "To install, run `pip install flash-attn einops`."
                )
            if not self.multi_query:
                # TODO: Flash Attention is implemented but not tested for MHA
                raise ValueError("Flash Attention is not supported with multi-head attention.")

    def _get_mask_value(self, device, dtype):
        # torch.where expects a tensor. We use a cache to avoid recreating it every time.
        if self.mask_value is None or self.mask_value.dtype != dtype or self.mask_value.device != device:
            self.mask_value = torch.full([], torch.finfo(dtype).min, dtype=dtype, device=device)
        return self.mask_value

    def _attn(self, query, key, value, attention_mask, head_mask=None):
        dtype = query.dtype
        softmax_dtype = torch.float32 if self.attention_softmax_in_fp32 else dtype
        upcast = dtype != softmax_dtype

        unscale = self.layer_idx + 1 if self.scale_attention_softmax_in_fp32 and upcast else 1
        scale_factor = unscale**-1
        if self.scale_attn_weights:
            scale_factor /= self.head_dim**0.5

        # MQA models: (batch_size, query_length, num_heads * head_dim)
        # MHA models: (batch_size, num_heads, query_length, head_dim)
        query_shape = query.shape
        batch_size = query_shape[0]
        key_length = key.size(-2)

        key = key.transpose(-1, -2)
        if self.multi_query:
            # (batch_size, query_length, num_heads, head_dim) x (batch_size, head_dim, key_length)
            # -> (batch_size, query_length, num_heads, key_length)
            query_length = query_shape[1]
            attn_shape = (batch_size, query_length, self.num_heads, key_length)
            attn_view = (batch_size, query_length * self.num_heads, key_length)
            # No copy needed for MQA 2, or when layer_past is provided.
            query = query.reshape(batch_size, query_length * self.num_heads, self.head_dim)
        else:
            # (batch_size, num_heads, query_length, head_dim) x (batch_size, num_heads, head_dim, key_length)
            # -> (batch_size, num_heads, query_length, key_length)
            query_length = query_shape[2]
            attn_shape = (batch_size, self.num_heads, query_length, key_length)
            attn_view = (batch_size * self.num_heads, query_length, key_length)
            # Always copies
            query = query.reshape(batch_size * self.num_heads, query_length, self.head_dim)
            # No copy when layer_past is provided.
            key = key.reshape(batch_size * self.num_heads, self.head_dim, key_length)

        attn_weights = torch.empty(attn_view, device=query.device, dtype=query.dtype)
        if query.device.type == "cpu":
            # This is needed because of a bug in pytorch https://github.com/pytorch/pytorch/issues/80588.
            # The bug was fixed in https://github.com/pytorch/pytorch/pull/96086,
            # but the fix has not been released as of pytorch version 2.0.0.
            attn_weights.zero_()
            beta = 1
        else:
            beta = 0
        attn_weights = torch.baddbmm(attn_weights, query, key, beta=beta, alpha=scale_factor).view(attn_shape)

        attn_weights = softmax_function(
            attn_weights,
            attention_mask,
            None if attention_mask is None else self._get_mask_value(attn_weights.device, softmax_dtype),
            unscale,
            softmax_dtype,
            upcast,
            self.fused_softmax,
        )

        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            if self.multi_query:
                head_mask = head_mask.transpose(1, 2)
            attn_weights = attn_weights * head_mask

        if self.multi_query:
            attn_output = torch.bmm(attn_weights.view(attn_view), value).view(query_shape)
        else:
            attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _attn_flash(self, query, key, value, attention_mask, head_mask=None):
        if head_mask is not None:
            raise NotImplementedError("Head mask is not supported with flash attention.")

        query_shape = query.shape
        attn_shape = query_shape[0], self.num_heads, self.head_dim
        query = query.view(attn_shape)
        if self.multi_query:
            key = key.unsqueeze(1).expand(attn_shape)
            value = value.unsqueeze(1).expand(attn_shape)
        else:
            key = key.view(attn_shape)
            value = value.view(attn_shape)

        sequence_lengths, padding_index, _, max_sequence_length = attention_mask

        # attn_output: (sum_seq_len, num_heads * head_dim)
        attn_output = flash_attn_unpadded_func(
            query,
            key,
            value,
            sequence_lengths,
            sequence_lengths,
            max_sequence_length,
            max_sequence_length,
            self.dropout_p if self.training else 0.0,
            softmax_scale=self.head_dim**-0.5 if self.scale_attn_weights else 1,
            causal=True,
        ).view(query_shape)

        return attn_output, None

    def _re_allocate_kv_cache(self, kv_cache, key_length, padded_key_length, allocate_key_length):
        batch_size = kv_cache.size(-1)
        assert not self.training
        if self.multi_query:
            allocated_kv_cache = torch.empty(
                [batch_size, allocate_key_length, self.head_dim], dtype=kv_cache.dtype, device=kv_cache.device
            )
            allocated_kv_cache[:, :key_length].copy_(kv_cache)
            padded_kv_cache = allocated_kv_cache[:, :padded_key_length]
        else:
            allocated_kv_cache = torch.empty(
                [batch_size, self.num_heads, allocate_key_length, self.head_dim],
                dtype=kv_cache.dtype,
                device=kv_cache.device,
            )
            allocated_kv_cache[:, :, key_length].copy_(kv_cache)
            padded_kv_cache = allocated_kv_cache[:, :, :padded_key_length]
        return allocated_kv_cache, padded_kv_cache

    def _merge_kv_caches(self, key_value, use_cache, layer_past, attention_mask):
        flash_attention = self.flash_attention and layer_past is None

        # Convert to standard KV cache format.
        if flash_attention and use_cache:
            _, padding_index, batch_size, max_sequence_length = attention_mask
            current_kv_cache = pad_input(key_value, padding_index, batch_size, max_sequence_length)
            if not self.multi_query:
                current_kv_cache = current_kv_cache.view(
                    batch_size, max_sequence_length, self.num_heads, 2 * self.head_dim
                ).transpose(1, 2)
        else:
            current_kv_cache = key_value

        # Calculate dimensions and recover layer_past
        batch_size = current_kv_cache.size(0)
        query_length = current_kv_cache.size(self.seq_dim)
        if layer_past is None:
            allocated_kv_cache, last_key_length = None, 0
            last_kv_cache = None
            key_length = query_length
            allocated_key_length = key_length
        else:
            allocated_kv_cache, last_key_length = layer_past
            last_kv_cache = (
                allocated_kv_cache[:, :last_key_length]
                if self.multi_query
                else allocated_kv_cache[:, :, :last_key_length]
            )
            key_length = query_length + last_key_length
            allocated_key_length = allocated_kv_cache.size(self.seq_dim)

        padded_key_length = key_length if flash_attention else attention_mask.size(-1)
        allocate_key_length = padded_key_length if use_cache else max(self.pre_allocate_kv_cache, padded_key_length)

        # Re-allocate kv cache and copy last value
        if allocate_key_length > allocated_key_length:
            if self.multi_query:
                allocated_kv_cache = torch.empty(
                    [batch_size, allocate_key_length, 2 * self.head_dim],
                    dtype=current_kv_cache.dtype,
                    device=current_kv_cache.device,
                )
                if layer_past is not None:
                    allocated_kv_cache[:, :last_key_length].copy_(last_kv_cache)
                if allocate_key_length > key_length:
                    # Nans in `value` can propagate through the matrix multiplication,
                    # so we set the remaining values to zero. (`last_key_length:key_length` is set below.)
                    allocated_kv_cache[:, key_length:, self.head_dim :].zero_()
            else:
                allocated_kv_cache = torch.empty(
                    [batch_size, self.num_heads, allocate_key_length, 2 * self.head_dim],
                    dtype=current_kv_cache.dtype,
                    device=current_kv_cache.device,
                )
                if layer_past is not None:
                    allocated_kv_cache[:, :, :last_key_length].copy_(last_kv_cache)
                if allocate_key_length > key_length:
                    allocated_kv_cache[:, :, key_length:, self.head_dim :].zero_()

        # Copy the new values.
        if allocate_key_length > allocated_key_length or layer_past is not None:
            if self.multi_query:
                allocated_kv_cache[:, last_key_length:key_length].copy_(current_kv_cache)
                padded_kv_cache = allocated_kv_cache[:, :padded_key_length]
            else:
                allocated_kv_cache[:, :, last_key_length:key_length].copy_(current_kv_cache)
                padded_kv_cache = allocated_kv_cache[:, :, :padded_key_length]
            if not flash_attention:
                # Use the merged KV cache.
                # Not needed when layer_past is None but frees some memory.
                key_value = padded_kv_cache

        if use_cache:
            if allocated_kv_cache is None:
                allocated_kv_cache = current_kv_cache
            present = allocated_kv_cache, key_length
        else:
            present = None
        return key_value, present

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_past: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[
        Tuple[torch.Tensor, Optional[torch.Tensor]],
        Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[torch.Tensor, ...]],
    ]:
        flash_attention = self.flash_attention and layer_past is None
        if self.multi_query or flash_attention:
            query, key_value = self.c_attn(hidden_states).split((self.embed_dim, 2 * self.kv_dim), dim=-1)
        else:
            # Note: We split as (self.num_heads, 3, self.head_dim) instead of (3, self.num_heads, self.head_dim),
            # i.e., the memory layout is not the same as GPT2.
            # This makes the concatenation with past_key_value more efficient.
            query, key_value = (
                self.c_attn(hidden_states)
                .view(*hidden_states.shape[:2], self.num_heads, 3 * self.head_dim)
                .transpose(1, 2)
                .split((self.head_dim, 2 * self.head_dim), dim=3)
            )

        if self._tuple_cache_format:
            # present =  (allocated_kv_cache, key_length)
            key_value, present = self._merge_kv_caches(key_value, use_cache, layer_past, attention_mask)
        else:
            # present = key_value
            if layer_past is not None:
                key_value = torch.cat((layer_past, key_value), dim=-2)
            present = key_value if use_cache else None

        key, value = key_value.split((self.head_dim, self.head_dim), dim=-1)

        attn_output, attn_weights = (self._attn_flash if flash_attention else self._attn)(
            query, key, value, attention_mask, head_mask
        )

        if not self.multi_query:
            attn_output = attn_output.transpose(1, 2).reshape(hidden_states.shape)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            if flash_attention:
                raise ValueError("`output_attentions` is not supported with Flash Attention.")
            if self.multi_query:
                # Transpose to return weights in the usual format (batch_size, num_heads, query_length, key_length)
                attn_weights = attn_weights.transpose(1, 2)
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class GPTBigCodeMLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = nn.Linear(embed_dim, intermediate_size)
        self.c_proj = nn.Linear(intermediate_size, embed_dim)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    # Copied from transformers.models.gpt2.modeling_gpt2.GPT2MLP.forward
    def forward(self, hidden_states: Optional[Tuple[torch.Tensor]]) -> torch.Tensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class GPTBigCodeBlock(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        hidden_size = config.hidden_size
        self.inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPTBigCodeAttention(config, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        if config.add_cross_attention:
            raise NotImplementedError("Cross-attention is not supported for gpt_bigcode.")

        self.mlp = GPTBigCodeMLP(self.inner_dim, config)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.Tensor]],
        layer_past: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[
        Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        if encoder_hidden_states is not None or encoder_attention_mask is not None:
            raise NotImplementedError("Cross-attention is not supported for gpt_bigcode.")

        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)


class GPTBigCodePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GPTBigCodeConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GPTBigCodeBlock"]

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (GPTBigCodeMLP, GPTBigCodeAttention)):
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

    # Copied from transformers.models.gpt2.modeling_gpt2.GPT2PreTrainedModel._set_gradient_checkpointing with GPT2->GPTBigCode
    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, GPTBigCodeModel):
            module.gradient_checkpointing = value


class GPTBigCodeModel(GPTBigCodePreTrainedModel):
    _keys_to_ignore_on_load_missing = ["attn.masked_bias"]

    def __init__(self, config):
        super().__init__(config)
        self.multi_query = config.multi_query
        self.embed_dim = config.hidden_size

        if config.add_cross_attention:
            raise NotImplementedError("Cross-attention is not supported for gpt_bigcode.")

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPTBigCodeBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.pad_key_length = config.pre_allocate_kv_cache if config.pad_key_length is None else config.pad_key_length
        self._tuple_cache_format = config.pre_allocate_kv_cache or self.pad_key_length or config.flash_attention
        self.inference_runner_type = InferenceRunnerType(config.inference_runner)

        self.flash_attention = config.flash_attention

        if self.inference_runner_type == InferenceRunnerType.NO_RUNNER:
            self.inference_runner = None
        else:
            from .inference_runner import GPTBigCodeInferenceRunner

            self.inference_runner = GPTBigCodeInferenceRunner(config, self)

        max_positions = config.max_position_embeddings
        # Causal mask
        self.register_buffer(
            "bias", torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)), persistent=False
        )

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def _get_causal_mask(self, padding_mask, query_length, key_length):
        # Self-attention mask.
        attention_mask = self.bias[None, key_length - query_length : key_length, :key_length]

        if padding_mask is not None:
            attention_mask = attention_mask * padding_mask.unsqueeze(1).to(
                dtype=torch.bool, device=attention_mask.device
            )

        # MQA models: (batch_size, query_length, n_heads, key_length)
        # MHA models: (batch_size, n_heads, query_length, key_length)
        return attention_mask.unsqueeze(2 if self.multi_query else 1)

    def _get_position_ids(self, position_ids, padding_mask, query_length, key_length, device):
        if position_ids is not None:
            position_ids = position_ids.to(device)
        elif padding_mask is not None and padding_mask.ndim == 2:
            # create position_ids on the fly for batch generation
            position_ids = padding_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(padding_mask == 0, 1)
            if key_length > query_length:
                position_ids = position_ids[:, key_length - query_length : key_length :]
        else:
            position_ids = torch.arange(key_length - query_length, key_length, dtype=torch.long, device=device)
        return position_ids.view(-1, query_length)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Union[List[torch.Tensor], int]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        if self.inference_runner is not None and past_key_values is not None:
            if self.config.validate_runner_input:
                assert input_ids is not None
                assert past_key_values is not None
                assert attention_mask is not None
                assert token_type_ids is None
                assert position_ids is not None
                assert head_mask is None
                assert inputs_embeds is None
                assert encoder_hidden_states is None
                assert encoder_attention_mask is None
                use_cache = use_cache if use_cache is not None else self.config.use_cache
                assert use_cache is True
                output_attentions = (
                    output_attentions if output_attentions is not None else self.config.output_attentions
                )
                assert output_attentions is False
                output_hidden_states = (
                    output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
                )
                assert output_hidden_states is False
                return_dict = return_dict if return_dict is not None else self.config.use_return_dict
                assert return_dict is True
            return self.inference_runner.forward(input_ids, attention_mask, position_ids, past_key_values)

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = self.config.use_cache if use_cache is None else use_cache
        return_dict = self.config.use_return_dict if return_dict is None else return_dict

        if input_ids is not None:
            if inputs_embeds is not None:
                raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
            input_shape = input_ids.shape
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size, query_length = input_ids.shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
            inputs_embeds = inputs_embeds.view(-1, input_shape[-2:])
            batch_size, query_length = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if batch_size <= 0:
            raise ValueError("batch_size has to be defined and > 0")

        flash_attention = self.flash_attention and past_key_values is None
        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        elif self._tuple_cache_format:
            past_length = past_key_values[0][1]
        else:
            past_length = past_key_values[0].size(-2)
        key_length = past_length + query_length

        position_ids = self._get_position_ids(position_ids, attention_mask, query_length, key_length, input_ids.device)

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, query_length)

        if not flash_attention:
            # Self-attention mask (padding + causal).
            attention_mask = self._get_causal_mask(attention_mask, query_length, key_length)
            if self.pad_key_length:
                pad = -key_length % 8
                if pad > 0:
                    attention_mask = torch.nn.functional.pad(attention_mask, (0, pad), mode="constant", value=False)

        if encoder_hidden_states is not None or encoder_attention_mask is not None:
            raise NotImplementedError("Cross-attention is not supported for gpt_bigcode.")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        # TODO: Unpad earlier (input ids), support unpadded input?
        if flash_attention:
            hidden_states, padding_index, sequence_lengths, max_sequence_length = unpad_input(
                hidden_states, attention_mask
            )
            # Pass the required parameters through the attention_mask argument
            attention_mask = (sequence_lengths, padding_index, batch_size, max_sequence_length)

        presents = [] if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache:
                presents.append(outputs[1])

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

        hidden_states = self.ln_f(hidden_states)

        if flash_attention:
            hidden_states = pad_input(hidden_states, padding_index, batch_size, query_length)

        hidden_states = hidden_states.view(input_shape + (hidden_states.size(-1),))

        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class GPTBigCodeForCausalLM(GPTBigCodePreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPTBigCodeModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.predict_last_token = config.predict_last_token

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values:
            input_ids = input_ids[:, -1:]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": kwargs.get("position_ids", None),
                "attention_mask": kwargs.get("attention_mask", None),
                "token_type_ids": token_type_ids,
            }
        )
        return model_inputs

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        if self.predict_last_token and not self.training:
            # We only care about the last token.
            hidden_states = hidden_states[:, -1:]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    @staticmethod
    def _reorder_cache(
        past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(layer_past.index_select(0, beam_idx.to(layer_past.device)) for layer_past in past_key_values)

