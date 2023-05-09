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

from torch.nn import functional as F

from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel
from transformers.models.gpt_neox import GPTNeoXConfig
from typing import Optional

# Flash attention imports
import rotary_emb
import flash_attn_cuda
import dropout_layer_norm

from flash_attn.layers.rotary import RotaryEmbedding

HAS_BITS_AND_BYTES = True
try:
    from bitsandbytes.nn import Linear8bitLt
except ImportError as e:
    HAS_BITS_AND_BYTES = False


class FastLayerNorm(nn.LayerNorm):
    def forward(self, hidden_states, residual=None):
        if hidden_states.shape[-1] > 8192:
            if residual is not None:
                hidden_states += residual
            residual = hidden_states

            return super(FastLayerNorm, self).forward(hidden_states), residual
        else:
            (
                normed_hidden_states,
                residual,
                *rest,
            ) = dropout_layer_norm.dropout_add_ln_fwd(
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
            if residual is None:
                residual = hidden_states

            return normed_hidden_states, residual


class FastLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super(FastLinear, self).__init__(in_features, out_features, bias, device, dtype)
        self.quantized = False
        self.bnb_linear = None

    def prepare_weights(self, quantize: bool = False):
        if quantize:
            if not HAS_BITS_AND_BYTES:
                raise ImportError(
                    "bitsandbytes is not available on your machine either because it is not installed "
                    "or you don't have a GPU.\n"
                    "You can install it with `pip install bitsandbytes`."
                )

            self.quantized = True
            self.bnb_linear = Linear8bitLt(
                self.in_features,
                self.out_features,
                has_fp16_weights=False,
                threshold=6.0,
                bias=False,
            )
            # Copy data to bnb_linear
            self.bnb_linear.weight.data = self.weight.data
            if self.bias is not None:
                self.bnb_linear.bias = nn.Parameter(self.bias)

            # Delete reference to data
            self.weight = None
            self.bias = None
        else:
            self.weight = nn.Parameter(self.weight.T)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.quantized:
            return self.bnb_linear(input)
        else:
            if self.bias is not None:
                return torch.addmm(self.bias, input, self.weight)
            return torch.matmul(input, self.weight)


class TensorParallelColumnLinear(FastLinear):
    def __init__(
        self,
        in_features,
        out_features,
        process_group: torch.distributed.ProcessGroup,
        bias=True,
        device=None,
        dtype=None,
    ):
        self.process_group = process_group
        self.tp_world_size = process_group.size()
        assert out_features % self.tp_world_size == 0
        out_features = out_features // self.tp_world_size

        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )


class TensorParallelRowLinear(FastLinear):
    def __init__(
        self,
        in_features,
        out_features,
        process_group: torch.distributed.ProcessGroup,
        reduce=True,
        bias=True,
        device=None,
        dtype=None,
    ):
        self.process_group = process_group
        self.tp_world_size = process_group.size()
        self.reduce = reduce
        assert in_features % self.tp_world_size == 0
        in_features = in_features // self.tp_world_size

        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = super(TensorParallelRowLinear, self).forward(input)
        if self.reduce:
            torch.distributed.all_reduce(out, group=self.process_group)

        return out


class TensorParallelEmbedding(nn.Embedding):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        process_group: torch.distributed.ProcessGroup,
        padding_idx=None,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        sparse=False,
        _weight=None,
        device=None,
        dtype=None,
    ):
        self.process_group = process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()

        self.original_num_embeddings = num_embeddings

        assert num_embeddings % self.tp_world_size == 0
        block_size = num_embeddings // self.tp_world_size
        # inputs in `[min_id, max_id[` are handled by `self` to get embeddings
        self.min_id = self.tp_rank * block_size
        self.max_id = (self.tp_rank + 1) * block_size

        # Additional entry that will map to zero
        # Used for masking
        self.null_idx = block_size

        super().__init__(
            block_size,
            embedding_dim,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
            _weight=_weight,
            device=device,
            dtype=dtype,
        )

    def add_null_idx(self):
        """Additional 0 entry used for masking"""
        self.weight = nn.Parameter(F.pad(self.weight, (0, 0, 0, 1)))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # default all out of bounds values to `self.null_idx` that will then be mapped to 0
        # translate for [0, self.max_id - self.min_id[
        input = torch.where(
            (self.min_id > input) | (input >= self.max_id),
            self.null_idx,
            input - self.min_id,
        )
        out = super().forward(input)
        torch.distributed.all_reduce(out, group=self.process_group)
        return out


class PositionRotaryEmbedding(RotaryEmbedding):
    def _update_cos_sin_cache(self, dtype, device, seqlen):
        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if (
            seqlen > self._seq_len_cached
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
        ):
            self._seq_len_cached = seqlen
            t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
            # Don't do einsum, it converts fp32 to fp16
            # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            freqs = torch.outer(t, self.inv_freq.to(device=t.device))
            self._cos_cached = torch.cos(freqs).to(dtype)
            self._sin_cached = torch.sin(freqs).to(dtype)

    def get_cos_sin(self, position_ids: torch.Tensor, max_s: int, dtype: torch.dtype):
        """
        Return cos and sin for the asked position ids
        """

        self._update_cos_sin_cache(dtype, position_ids.device, max_s)

        cos = torch.index_select(self._cos_cached, 0, position_ids)
        sin = torch.index_select(self._sin_cached, 0, position_ids)
        return cos.unsqueeze(1), sin.unsqueeze(1)

    def forward(self, qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        rotary_dim = cos.shape[-1]
        q1 = qkv[:, 0, :, :rotary_dim]
        q2 = qkv[:, 0, :, rotary_dim : 2 * rotary_dim]
        k1 = qkv[:, 1, :, :rotary_dim]
        k2 = qkv[:, 1, :, rotary_dim : 2 * rotary_dim]

        rotary_emb.apply_rotary(q1, q2, cos, sin, q1, q2, False)
        rotary_emb.apply_rotary(k1, k2, cos, sin, k1, k2, False)
        return qkv


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
        qkv_rot = self.rotary_emb(qkv, cos, sin)

        # Prefill
        if layer_past_present_indices is None:
            # Copy to layer past
            layer_past[...] = qkv_rot[:, 1:]

            # output
            attn_output = torch.empty_like(qkv_rot[:, 0])
            # flash attention
            flash_attn_cuda.fwd(
                qkv_rot[:, 0],
                qkv_rot[:, 1],
                qkv_rot[:, 2],
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
            query = qkv_rot[:, 0]
            # Add present to the layer_past tensor at the correct indices
            layer_past[layer_past_present_indices] = qkv_rot[:, 1:]

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

    def post_load_weights(self, load_in_8bit=False):
        if isinstance(self.embed_in, TensorParallelEmbedding):
            self.embed_in.add_null_idx()
        for layer in self.layers:
            layer: FlashNeoXLayer
            layer.attention.shuffle_qkv_dims()
            layer.attention.query_key_value.prepare_weights(load_in_8bit)
            layer.attention.dense.prepare_weights(load_in_8bit)
            layer.mlp.dense_h_to_4h.prepare_weights(load_in_8bit)
            layer.mlp.dense_4h_to_h.prepare_weights(load_in_8bit)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # Pop here as we will replace the layer in our own logic and don't want from_pretrained
        # to do it for us
        load_in_8bit = kwargs.pop("load_in_8bit", False)
        model = super(FlashGPTNeoXModel, cls).from_pretrained(
            pretrained_model_name_or_path, load_in_8bit=False, *model_args, **kwargs
        )

        model.post_load_weights(load_in_8bit)
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

    def post_load_weights(self, load_in_8bit=False):
        self.gpt_neox.post_load_weights(load_in_8bit)
        self.embed_out.prepare_weights()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # Pop here as we will replace the layer in our own logic and don't want from_pretrained
        # to do it for us
        load_in_8bit = kwargs.pop("load_in_8bit", False)
        model = super(FlashGPTNeoXForCausalLM, cls).from_pretrained(
            pretrained_model_name_or_path, load_in_8bit=False, *model_args, **kwargs
        )
        model.post_load_weights(load_in_8bit)
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
        logits = self.embed_out(hidden_states)

        if self.gpt_neox.tp_embeddings:
            # Logits are sharded, so we need to gather them
            world_logits = [torch.empty_like(logits) for _ in range(self.world_size)]
            torch.distributed.all_gather(world_logits, logits, group=self.process_group)
            world_logits = torch.cat(world_logits, dim=1)

            return world_logits, present
        return logits, present
