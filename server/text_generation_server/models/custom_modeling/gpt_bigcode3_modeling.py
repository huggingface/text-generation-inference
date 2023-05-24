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

from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.models.gpt_bigcode.configuration_gpt_bigcode import (
    GPTBigCodeConfig,
)

class InferenceRunnerType(IntEnum):
    NO_RUNNER = 0
    # Use the inference runner without cuda graphs.
    BASE_RUNNER = 1
    # Use cuda graphs in the inference runner. Leave out the attention which has a variable shape.
    # This significantly lowers the cpu time and prevent a cpu bottleneck for smaller batches and models.
    PARTIAL_GRAPH = 2
    # Turn the whole model into a cuda graph. One graph for each sequence length.
    # Note: only useful for small batches and models, graphs take some time to generate, flaky.
    # Crashes with jit on A100 but seems to work without jit (PYTORCH_JIT=0) and on V100.
    FULL_GRAPH = 3

try:
    from flash_attn.bert_padding import pad_input, unpad_input
    from flash_attn.flash_attn_interface import flash_attn_unpadded_func
except ImportError:
    flash_attn_unpadded_func = None
    pad_input = None
    unpad_input = None


logger = logging.get_logger(__name__)


@torch.jit.script
def upcast_masked_softmax(
    x: torch.Tensor, mask: torch.Tensor, mask_value: torch.Tensor, scale: float, softmax_dtype: torch.dtype
):
    input_dtype = x.dtype
    x = x.to(softmax_dtype) * scale
    x = torch.where(mask, x, mask_value)
    x = torch.nn.functional.softmax(x, dim=-1).to(input_dtype)
    return x


@torch.jit.script
def upcast_softmax(x: torch.Tensor, scale: float, softmax_dtype: torch.dtype):
    input_dtype = x.dtype
    x = x.to(softmax_dtype) * scale
    x = torch.nn.functional.softmax(x, dim=-1).to(input_dtype)
    return x


@torch.jit.script
def masked_softmax(x: torch.Tensor, mask: torch.Tensor, mask_value: torch.Tensor):
    x = torch.where(mask, x, mask_value)
    x = torch.nn.functional.softmax(x, dim=-1)
    return x


@torch.profiler.record_function("softmax_function")
def softmax_function(
    x: torch.Tensor,
    mask: torch.Tensor,
    mask_value: torch.Tensor,
    scale: float,
    softmax_dtype: torch.dtype,
    upcast: bool = True,
):
    """
    This selects the appropriate (fused) (upcast) (masked) softmax method. Because of the way jit works, each case
    needs to be handled through a separate method. The fused kernels remove most of the overhead from masking, casting
    and scaling, but only work well when the key length is a multiple of 8. For other key lengths, it is extremely
    inefficient.
    """
    #assert x.size(-1) % 8 == 0
    if upcast:
        if mask is None:
            return upcast_softmax(x, scale, softmax_dtype)
        else:
            return upcast_masked_softmax(x, mask, mask_value, scale, softmax_dtype)
    else:
        if mask is None:
            return torch.nn.functional.softmax(x, dim=-1)
        else:
            return masked_softmax(x, mask, mask_value)


class GPTBigCodeAttention(nn.Module):
    def __init__(self, config, layer_idx=None, dtype:torch.dtype=torch.float32, device:torch.device=torch.device("cpu")):
        super().__init__()
        self.mask_value = None
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.layer_idx = layer_idx

        # KV caching and padding

        self.c_attn = nn.Linear(self.embed_dim, self.embed_dim + 2 * self.head_dim, dtype=dtype, device=device)
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim, dtype=dtype, device=device)

    @torch.profiler.record_function("GPTBigCodeAttention._get_mask_value")
    def _get_mask_value(self, device, dtype):
        # torch.where expects a tensor. We use a cache to avoid recreating it every time.
        if self.mask_value is None or self.mask_value.dtype != dtype or self.mask_value.device != device:
            self.mask_value = torch.full([], torch.finfo(dtype).min, dtype=dtype, device=device)
        return self.mask_value

    @torch.profiler.record_function("GPTBigCodeAttention._attn")
    def _attn(self, query, key, value, attention_mask):
        softmax_dtype = torch.float32
        upcast = query.dtype != softmax_dtype

        unscale = self.layer_idx + 1 if upcast else 1
        scale_factor = unscale**-1 / self.head_dim**0.5

        # (batch_size, query_length, num_heads * head_dim)
        query_shape = query.shape
        batch_size = query_shape[0]
        key_length = key.size(-2)

        key = key.transpose(-1, -2)
        # (batch_size, query_length, num_heads, head_dim) x (batch_size, head_dim, key_length)
        # -> (batch_size, query_length, num_heads, key_length)
        query_length = query_shape[1]
        attn_shape = (batch_size, query_length, self.num_heads, key_length)
        attn_view = (batch_size, query_length * self.num_heads, key_length)
        # No copy needed for MQA 2, or when layer_past is provided.
        query = query.reshape(batch_size, query_length * self.num_heads, self.head_dim)

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
        )
        attn_output = torch.bmm(attn_weights.view(attn_view), value).view(query_shape)

        return attn_output

    @torch.profiler.record_function("GPTBigCodeAttention._attn_flash")
    def _attn_flash(self, query, key, value, flash_params):

        query_shape = query.shape
        attn_shape = query_shape[0], self.num_heads, self.head_dim
        query = query.view(attn_shape)
        key = key.unsqueeze(1).expand(attn_shape)
        value = value.unsqueeze(1).expand(attn_shape)

        sequence_lengths, padding_index, _, max_sequence_length = flash_params

        # attn_output: (sum_seq_len, num_heads * head_dim)
        attn_output = flash_attn_unpadded_func(
            query,
            key,
            value,
            sequence_lengths,
            sequence_lengths,
            max_sequence_length,
            max_sequence_length,
            0.0,
            softmax_scale=self.head_dim**-0.5,
            causal=True,
        ).view(query_shape)

        return attn_output

    @torch.profiler.record_function("GPTBigCodeAttention._merge_kv_caches")
    def _merge_kv_caches(self, key_value, layer_past, attention_mask, flash_params):

        # Convert to standard KV cache format.
        if flash_params is not None:
            _, padding_index, batch_size, max_sequence_length = flash_params
            current_kv_cache = pad_input(key_value, padding_index, batch_size, max_sequence_length)
            return key_value, (current_kv_cache, max_sequence_length)

        current_kv_cache = key_value

        # Calculate dimensions and recover layer_past
        batch_size = current_kv_cache.size(0)
        query_length = current_kv_cache.size(-2)
        if layer_past is None:
            allocated_kv_cache, last_key_length = None, 0
            last_kv_cache = None
            key_length = query_length
            allocated_key_length = key_length
        else:
            allocated_kv_cache, last_key_length = layer_past
            last_kv_cache = allocated_kv_cache[:, :last_key_length]
            key_length = query_length + last_key_length
            allocated_key_length = allocated_kv_cache.size(-2)

        padded_key_length = attention_mask.size(-1)

        # Re-allocate kv cache and copy last value
        if padded_key_length > allocated_key_length:
            allocated_kv_cache = torch.empty(
                [batch_size, padded_key_length, 2 * self.head_dim],
                dtype=current_kv_cache.dtype,
                device=current_kv_cache.device,
            )
            if layer_past is not None:
                allocated_kv_cache[:, :last_key_length].copy_(last_kv_cache)
            if padded_key_length > key_length:
                # Nans in `value` can propagate through the matrix multiplication,
                # so we set the remaining values to zero. (`last_key_length:key_length` is set below.)
                allocated_kv_cache[:, key_length:, self.head_dim :].zero_()

        # Copy the new values.
        if padded_key_length > allocated_key_length or layer_past is not None:
            allocated_kv_cache[:, last_key_length:key_length].copy_(current_kv_cache)
            padded_kv_cache = allocated_kv_cache[:, :padded_key_length]
            # Use the merged KV cache.
            # Not needed when layer_past is None but frees some memory.
            key_value = padded_kv_cache

        if allocated_kv_cache is None:
            allocated_kv_cache = current_kv_cache
        present = allocated_kv_cache, key_length
        return key_value, present

    @torch.profiler.record_function("GPTBigCodeAttention.forward")
    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_past: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        flash_params: Optional[Tuple] = None
    ) -> Tuple[torch.Tensor, Any]:
        query, key_value = self.c_attn(hidden_states).split((self.embed_dim, 2 * self.head_dim), dim=-1)

        # present =  (allocated_kv_cache, key_length)
        key_value, present = self._merge_kv_caches(key_value, layer_past, attention_mask, flash_params)

        key, value = key_value.split((self.head_dim, self.head_dim), dim=-1)

        if flash_params is None:
            attn_output=self._attn(query, key, value, attention_mask)
        else:
            attn_output=self._attn_flash(query, key, value, flash_params)

        attn_output = self.c_proj(attn_output)

        return attn_output, present


class GPTBigCodeMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * embed_dim
        self.c_fc = nn.Linear(embed_dim, inner_dim)
        self.c_proj = nn.Linear(inner_dim, embed_dim)

    @torch.profiler.record_function("GPTBigCodeMLP.forward")
    # Copied from transformers.models.gpt2.modeling_gpt2.GPT2MLP.forward
    def forward(self, hidden_states: Optional[Tuple[torch.Tensor]]) -> torch.Tensor:
        return self.c_proj(nn.functional.gelu(self.c_fc(hidden_states), approximate="tanh"))


class GPTBigCodeBlock(nn.Module):
    def __init__(self, config, layer_idx=None, dtype:torch.dtype=torch.float32, device:torch.device=torch.device("cpu")):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon, dtype=dtype, device=device)
        self.attn = GPTBigCodeAttention(config, layer_idx=layer_idx, dtype=dtype, device=device)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon, dtype=dtype, device=device)
        self.mlp = GPTBigCodeMLP(config)

    @torch.profiler.record_function("GPTBigCodeBlock.forward")
    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_past: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        flash_params: Optional[Tuple] = None
    ) -> Tuple[torch.Tensor, Any]:
        with torch.profiler.record_function("GPTBigCodeAttention.ln"):
            ai=self.ln_1(hidden_states)
        attn_output, present = self.attn(
            ai,
            layer_past=layer_past,
            attention_mask=attention_mask,
            flash_params=flash_params,
        )
        with torch.profiler.record_function("GPTBigCodeAttention.residual"):
            hidden_states.add_(attn_output)

        with torch.profiler.record_function("GPTBigCodeAttention.dummy"):
            pass
        with torch.profiler.record_function("GPTBigCodeAttention.ln"):
            ai=self.ln_2(hidden_states)
        ai=self.mlp(ai)
        with torch.profiler.record_function("GPTBigCodeAttention.residual"):
            hidden_states.add_(ai)
        return hidden_states, present



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
    def __init__(self, config, dtype:torch.dtype=torch.float32, device:torch.device=torch.device("cpu")):
        super().__init__(config)

        self.wte = nn.Embedding(config.vocab_size, config.hidden_size, dtype=dtype, device=device)
        self.wpe = nn.Embedding(config.max_position_embeddings, config.hidden_size, dtype=dtype, device=device)

        self.h = nn.ModuleList([GPTBigCodeBlock(config, layer_idx=i, dtype=dtype, device=device) for i in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(config.hidden_size, dtype=dtype, device=device, eps=config.layer_norm_epsilon)

        self.inference_runner_type = InferenceRunnerType.NO_RUNNER #InferenceRunnerType(config.inference_runner)

        self.flash_attention = True #config.flash_attention

        if self.flash_attention:
            if flash_attn_unpadded_func is None:
                raise RuntimeError(
                    "Flash Attention requires `flash_attn` and `einops`. "
                    "To install, run `pip install flash-attn einops`."
                )

        if self.inference_runner_type == InferenceRunnerType.NO_RUNNER:
            self.inference_runner = None
        else:
            from .inference_runner import GPTBigCodeInferenceRunner

            self.inference_runner = GPTBigCodeInferenceRunner(config, self)

        # Causal mask
        self.register_buffer(
            "bias", torch.empty((config.max_position_embeddings, config.max_position_embeddings), dtype=torch.bool, device=device)
        )

    #@torch.profiler.record_function("GPTBigCodeModel._get_causal_mask")
    def _get_causal_mask(self, padding_mask, query_length, key_length):
        # Self-attention mask.
        attention_mask = self.bias[None, key_length - query_length : key_length, :key_length]

        if padding_mask is not None:
            attention_mask = attention_mask * padding_mask.unsqueeze(1).to(
                dtype=torch.bool, device=attention_mask.device
            )
        pad = -key_length % 8
        if pad > 0:
            attention_mask = torch.nn.functional.pad(attention_mask, (0, pad), mode="constant", value=False)

        # (batch_size, query_length, n_heads, key_length)
        return attention_mask.unsqueeze(2)

    #@torch.profiler.record_function("GPTBigCodeModel.forward")
    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        past_key_values: Optional[Union[List[torch.Tensor], int]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: torch.Tensor,
    ) -> Tuple:
        if self.inference_runner is not None and past_key_values is not None:
            if self.config.validate_runner_input:
                assert past_key_values is not None
            return self.inference_runner.forward(input_ids, attention_mask, position_ids, past_key_values)

        batch_size, query_length = input_ids.shape

        flash_attention = self.flash_attention and past_key_values is None
        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][1]


        hidden_states = self.wte(input_ids) + self.wpe(position_ids)

        # TODO: Unpad earlier (input ids), support unpadded input?
        if flash_attention:
            hidden_states, padding_index, sequence_lengths, max_sequence_length = unpad_input(
                hidden_states, attention_mask
            )
            flash_params = (sequence_lengths, padding_index, batch_size, max_sequence_length)
            attention_mask=None
        else:
            key_length = past_length + query_length
            # Self-attention mask (padding + causal).
            attention_mask = self._get_causal_mask(attention_mask, query_length, key_length)
            flash_params=None

        presents = []
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):

            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                flash_params=flash_params
            )

            hidden_states = outputs[0]
            presents.append(outputs[1])

        hidden_states = self.ln_f(hidden_states)

        if flash_attention:
            hidden_states = pad_input(hidden_states, padding_index, batch_size, query_length)

        return hidden_states, presents


class GPTBigCodeForCausalLM(GPTBigCodePreTrainedModel):
    def __init__(self, config, dtype:torch.dtype=torch.float32, device:torch.device=torch.device("cpu")):
        super().__init__(config)
        meta=torch.device("meta")
        self.transformer = GPTBigCodeModel(config, dtype=dtype, device=meta)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False, dtype=dtype, device=meta)

        self.to_empty(device=device)

        # Initialize weights and apply final processing
        self.post_init()

    #@torch.profiler.record_function("GPTBigCodeForCausalLM.forward")
    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        past_key_values: Optional[Union[List[torch.Tensor], int]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: torch.Tensor,
        predict_all_tokens: bool=True,
    ) -> Tuple:

        hidden_states, presents=self.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids
        )
        #with torch.profiler.record_function("GPTBigCodeForCausalLM.head"):
        if not predict_all_tokens:
            # We only care about the last token.
            hidden_states = hidden_states[:, -1:]

        lm_logits = self.lm_head(hidden_states)

        return lm_logits, presents
