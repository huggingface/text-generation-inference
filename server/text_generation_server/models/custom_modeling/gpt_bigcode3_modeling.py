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
from typing import Optional, Tuple, Any, List

from torch import where, addmm, mm, float32, dtype, Tensor, baddbmm, empty, device, bmm, full, ones, finfo, jit
from torch.nn import Linear, Embedding, Module, LayerNorm, ModuleList
from torch.nn.functional import gelu, softmax, embedding

from dropout_layer_norm import dropout_add_ln_fwd
from flash_attn.bert_padding import pad_input, unpad_input
from flash_attn.flash_attn_interface import flash_attn_unpadded_func

from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.models.gpt_bigcode.configuration_gpt_bigcode import (
    GPTBigCodeConfig,
)


logger = logging.get_logger(__name__)


@jit.script
def upcast_masked_softmax(
    x: Tensor, mask: Tensor, mask_value: Tensor, scale: float
):
    input_dtype = x.dtype
    x = x.to(float32) * scale
    x = where(mask, x, mask_value)
    x = softmax(x, dim=-1).to(input_dtype)
    return x


class GPTBigCodeAttention(Module):
    mask_value: Tensor

    def __init__(self, config: GPTBigCodeConfig, layer_idx: int, dtype: dtype):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.layer_idx = layer_idx

        self.c_attn = Linear(
            self.embed_dim,
            self.embed_dim + 2 * self.head_dim,
            dtype=dtype,
            device="meta",
        )
        self.c_proj = Linear(
            self.embed_dim, self.embed_dim, dtype=dtype, device="meta"
        )

class GPTBigCodeMLP(Module):
    # TODO: Merge into GPTBigCodeBlock (needs renaming in state dict)
    def __init__(self, config: GPTBigCodeConfig, dtype: dtype):
        super().__init__()
        embed_dim = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * embed_dim
        self.c_fc = Linear(embed_dim, inner_dim, dtype=dtype, device="meta")
        self.c_proj = Linear(inner_dim, embed_dim, dtype=dtype, device="meta")


class GPTBigCodeBlock(Module):
    def __init__(self, config: GPTBigCodeConfig, layer_idx: int, dtype: dtype):
        super().__init__()
        self.ln_1 = LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
            dtype=dtype,
            device="meta",
        )
        self.attn = GPTBigCodeAttention(config, layer_idx=layer_idx, dtype=dtype)
        self.ln_2 = LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
            dtype=dtype,
            device="meta",
        )
        self.mlp = GPTBigCodeMLP(config, dtype=dtype)

    def post_load_weights(self, mask_value):
        self.attn.mask_value = mask_value

        self.mask_value = mask_value
        self.hd=self.attn.head_dim
        self.split0=(self.attn.embed_dim, 2*self.hd)
        self.split1=(self.hd, self.hd)

        self.aaw=self.attn.c_attn.weight.t_()
        self.aab=self.attn.c_attn.bias
        self.apw=self.attn.c_proj.weight.t_()
        self.apb=self.attn.c_proj.bias

        self.unscale=self.attn.layer_idx + 1
        self.ps=self.hd**-0.5
        self.ds=self.unscale**-1 * self.ps

        self.l1w=self.ln_1.weight
        self.l1b=self.ln_1.bias
        self.e1=self.ln_1.eps
        self.l2w=self.ln_2.weight
        self.l2b=self.ln_2.bias
        self.e2=self.ln_2.eps
        self.mfb=self.mlp.c_fc.bias
        self.mfw=self.mlp.c_fc.weight.t_()
        self.mfb=self.mlp.c_fc.bias
        self.mpw=self.mlp.c_proj.weight.t_()
        self.mpb=self.mlp.c_proj.bias


    def prefill(
        self,
        hidden_states: Tensor,
        residual: Optional[Tensor],
        sequence_lengths,
        key_length: int,
    ) -> Tuple[Tensor, Tensor, Any]:
        if residual is None: # First layer
            residual = hidden_states
            hidden_states, *_ = dropout_add_ln_fwd(
                hidden_states,
                residual,
                self.l1w,
                self.l1b,
                None,
                None,
                None,
                None,
                0.0,
                self.e1,
                1.0,
                0,
                None,
                False,
                False,
            )
        else:
            hidden_states, residual, *_ = dropout_add_ln_fwd(
                hidden_states,
                residual,
                self.l1w,
                self.l1b,
                None,
                None,
                None,
                None,
                0.0,
                self.e1,
                1.0,
                0,
                None,
                False,
                False,
            )
        hidden_shape = hidden_states.shape
        query, key_value = addmm(self.aab, hidden_states, self.aaw).split(self.split0, dim=-1)
        query = query.view(hidden_shape[0], self.num_heads, self.head_dim)
        key, value = (
            key_value.unsqueeze(1)
            .expand(hidden_shape[0], self.num_heads, 2 * self.head_dim)
            .split(self.split1, dim=-1)
        )

        # attn_output: (sum_seq_len, num_heads * head_dim)
        hidden_states = flash_attn_unpadded_func(
            query,
            key,
            value,
            sequence_lengths,
            sequence_lengths,
            key_length,
            key_length,
            0.0,
            softmax_scale=self.ps,
            causal=True,
        ).view(hidden_shape)
        hidden_states = addmm(self.apb, hidden_states, self.apw, out=query)

        hidden_states, residual, *_ = dropout_add_ln_fwd(
            hidden_states,
            residual,
            self.l2w,
            self.l2b,
            None,
            None,
            None,
            None,
            0.0,
            self.e2,
            1.0,
            0,
            None,
            False,
            False,
        )
        # TODO: Find an inplace and/or fused (with addmm) gelu kernel?
        hidden_states = addmm(self.mpb, gelu(addmm(self.mfb, hidden_states, self.mfw), approximate="tanh"), self.mpw, out=hidden_states)
        return hidden_states, residual, key_value

    def decode(
        self,
        hidden_states: Tensor,
        residual: Optional[Tensor],
        layer_past: Tensor,
        attention_mask: Tensor,
        key_length: int,
    ) -> Tuple[Tensor, Tensor, Any]:

        batch_size=hidden_states.size(0)

        if residual is None: # First layer
            residual = hidden_states
            hidden_states, *_ = dropout_add_ln_fwd(
                hidden_states,
                residual,
                self.l1w,
                self.l1b,
                None,
                None,
                None,
                None,
                0.0,
                self.e1,
                1.0,
                0,
                None,
                False,
                False,
            )
        else:
            hidden_states, residual, *_ = dropout_add_ln_fwd(
                hidden_states,
                residual,
                self.l1w,
                self.l1b,
                None,
                None,
                None,
                None,
                0.0,
                self.e1,
                1.0,
                0,
                None,
                False,
                False,
            )

        query, key_value = addmm(self.aab, hidden_states, self.aaw).split(self.split0, dim=-1)
        query_view = query.view(batch_size, self.num_heads, self.head_dim)

        # Calculate dimensions and recover layer_past
        padded_key_length = attention_mask.size(-1)
        allocated_key_length = layer_past.size(-2)

        # TODO: Allow pre-allocation with size > padded_key_length
        if padded_key_length > allocated_key_length:
            # Re-allocate kv cache and copy last value
            allocated_kv_cache = empty(
                [batch_size, padded_key_length, 2*self.hd],
                dtype=key_value.dtype,
                device=key_value.device,
            )
            allocated_kv_cache[:, : key_length - 1].copy_(
                layer_past[:, : key_length - 1]
            )
            # Nans in `value` can propagate through the matrix multiplication,
            # so we set the remaining values to zero. (`last_key_length:key_length` is set below.)
            allocated_kv_cache[:, allocated_key_length:, self.head_dim :].zero_()
            layer_past = allocated_kv_cache

        # Copy the new values.
        layer_past[:, key_length - 1].copy_(key_value)

        key, value = layer_past.split(self.split1, dim=-1)

        # Assume we always upcast (optimized for fp16/bf16)
        # TODO: Upcasting needed for bf16?
        hidden_states = baddbmm(
            empty(
                (batch_size, self.num_heads, padded_key_length),
                device=query.device,
                dtype=query.dtype,
            ),
            query_view,
            key.transpose(-1, -2),
            beta=0,
            alpha=self.ds,
        )
        hidden_states = upcast_masked_softmax(
            hidden_states, attention_mask, self.mask_value, self.unscale
        )

        # TODO: Write attn output directly into query, avoids both allocation and view.
        bmm(hidden_states.squeeze_(1), value, out=query_view)
        # TODO: Reuse attn weight tensor for c_proj output?
        hidden_states = addmm(self.apb, query, self.apw)

        hidden_states, residual, *_ = dropout_add_ln_fwd(
            hidden_states,
            residual,
            self.l2w,
            self.l2b,
            None,
            None,
            None,
            None,
            0.0,
            self.e2,
            1.0,
            0,
            None,
            False,
            False,
        )
        # TODO: Reuse attn weight tensor for c_fc output? (ok if padded_key_length>=4*head_dim, otherwise need to allocate a bigger one).
        # TODO: Find an inplace and/or fused (with addmm) gelu kernel?
        hidden_states = addmm(self.mpb, gelu(addmm(self.mfb, hidden_states, self.mfw), approximate="tanh"), self.mpw, out=hidden_states)
        return hidden_states, residual, layer_past


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
        if isinstance(module, (GPTBigCodeMLP, GPTBigCodeAttention)):
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            module.c_proj.weight.data.normal_(
                mean=0.0,
                std=(
                    self.config.initializer_range / math.sqrt(2 * self.config.n_layer)
                ),
            )
            module.c_proj._is_hf_initialized = True
        elif isinstance(module, Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class GPTBigCodeModel(GPTBigCodePreTrainedModel):
    # TODO: Merge into GPTBigCodeForCausalLM (needs renaming in state dict)
    def __init__(self, config: GPTBigCodeConfig, dtype: dtype):
        super().__init__(config)

        self.wte = Embedding(
            config.vocab_size, config.hidden_size, dtype=dtype, device="meta"
        )
        self.wpe = Embedding(
            config.max_position_embeddings,
            config.hidden_size,
            dtype=dtype,
            device="meta",
        )

        self.h = ModuleList(
            [
                GPTBigCodeBlock(config, layer_idx=i, dtype=dtype)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.ln_f = LayerNorm(
            config.hidden_size,
            dtype=dtype,
            device="meta",
            eps=config.layer_norm_epsilon,
        )


class GPTBigCodeForCausalLM(GPTBigCodePreTrainedModel):
    def __init__(
        self, config, dtype: dtype, device: device = device("cuda")
    ):
        super().__init__(config)
        if device.type != "cuda":
            raise NotImplementedError(f"Device {device} not supported")

        self.transformer = GPTBigCodeModel(config, dtype=dtype)
        self.lm_head = Linear(
            config.n_embd, config.vocab_size, bias=False, dtype=dtype, device="meta"
        )
        self.mask_value = full(
            (), finfo(float32).min, dtype=float32, device=device
        )

        self.to_empty(device=device)

        # Initialize weights and apply final processing
        # TODO: Skip?
        self.post_init()

    def post_load_weights(self):
        layer: GPTBigCodeBlock
        for layer in self.transformer.h:
            layer.post_load_weights(self.mask_value)
        self.tw=self.transformer.wte.weight
        self.pw=self.transformer.wpe.weight
        self.hw=self.lm_head.weight.t_()
        self.lw=self.transformer.ln_f.weight
        self.lb=self.transformer.ln_f.bias
        self.le=self.transformer.ln_f.eps

    def prefill(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor = None,
        position_ids: Tensor,
        predict_all_tokens: bool = True,
    ) -> Tuple:
        batch_size, query_length = input_ids.shape

        hidden_states = embedding(input_ids, self.tw).add_(embedding(position_ids, self.pw))

        # Prefill (flash attn)
        # TODO: Unpad earlier (input ids)?
        hidden_states, padding_index, sequence_lengths, key_length = unpad_input(
            hidden_states, attention_mask
        )
        assert key_length == query_length

        residual = None
        past_key_values = []
        block: GPTBigCodeBlock
        for block in self.transformer.h:
            hidden_states, residual, key_value = block.prefill(
                hidden_states,
                residual,
                sequence_lengths,
                key_length,
            )
            past_key_values.append(
                pad_input(key_value, padding_index, batch_size, query_length)
            )

        hidden_states, *_ = dropout_add_ln_fwd(
            hidden_states,
            residual,
            self.lw,
            self.lb,
            None,
            None,
            None,
            None,
            0.0,
            self.le,
            1.0,
            0,
            None,
            False,
            False,
        )

        # Next bit is the memory bottleneck with predict_all_tokens so we free as much memory as possible.
        del residual

        if predict_all_tokens:
            hidden_states = pad_input(
                mm(hidden_states, self.hw), padding_index, batch_size, query_length
            )
        else:
            # TODO: Index directly using cu_seqlens instead
            hidden_states = mm(pad_input(
                hidden_states, padding_index, batch_size, query_length
            )[:, -1], self.hw).unsqueeze_(1)

        return hidden_states, past_key_values

    def decode(
        self,
        *,
        input_ids: Tensor,
        past_key_values: List[Tensor],
        attention_mask: [Tensor],
        position_ids: Tensor,
        key_length: int,
    ) -> Tuple:
        hidden_states = embedding(input_ids, self.tw).add_(embedding(position_ids, self.pw))

        residual = None
        block: GPTBigCodeBlock
        for i, (block, layer_past) in enumerate(
            zip(self.transformer.h, past_key_values)
        ):
            hidden_states, residual, past_key_values[i] = block.decode(
                hidden_states,
                residual,
                layer_past,
                attention_mask,
                key_length,
            )
        hidden_states, *_ = dropout_add_ln_fwd(
            hidden_states,
            residual,
            self.lw,
            self.lb,
            None,
            None,
            None,
            None,
            0.0,
            self.le,
            1.0,
            0,
            None,
            False,
            False,
        )
        # TODO: Reuse residual?
        return mm(hidden_states, self.hw).unsqueeze_(1), past_key_values
