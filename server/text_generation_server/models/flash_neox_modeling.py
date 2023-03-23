import torch
import torch.distributed

import torch.nn.functional as F

from torch import nn
from transformers.modeling_utils import PreTrainedModel
from transformers.models.gpt_neox import GPTNeoXConfig

import rotary_emb
import flash_attn_cuda

from flash_attn.flash_attn_interface import (
    flash_attn_unpadded_qkvpacked_func,
    flash_attn_unpadded_kvpacked_func,
)
# from flash_attn.ops.fused_dense import (
#     FusedDense,
#     ColumnParallelLinear,
#     RowParallelLinear,
#     fused_mlp_func,
# )
from flash_attn.layers.rotary import RotaryEmbedding, apply_rotary_emb_qkv_


# from flash_attn.ops.layer_norm import dropout_add_layer_norm


class TensorParallelColumnLinear(nn.Linear):
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

        super().__init__(in_features=in_features,
                         out_features=out_features,
                         bias=bias,
                         device=device,
                         dtype=dtype)

    @staticmethod
    def linear(input, weight, bias):
        return F.linear(input, weight, bias)

    def forward(self, input):
        return self.linear(input, self.weight, self.bias)


class TensorParallelRowLinear(nn.Linear):
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
        assert in_features % self.tp_world_size == 0
        in_features = in_features // self.tp_world_size

        super().__init__(in_features=in_features,
                         out_features=out_features,
                         bias=bias,
                         device=device,
                         dtype=dtype)

    @staticmethod
    def linear(input, weight, bias):
        return F.linear(input, weight, bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.linear(input, self.weight, self.bias)
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
            dtype=None
    ):
        self.process_group = process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()

        self.original_num_embeddings = num_embeddings

        # TODO @thomasw21 fix and remove that constraint
        assert num_embeddings % self.tp_world_size == 0
        block_size = num_embeddings // self.tp_world_size
        # inputs in `[min_id, max_id[` are handled by `self` to get embeddings
        self.min_id = self.tp_rank * block_size
        self.max_id = (self.tp_rank + 1) * block_size

        super().__init__(block_size, embedding_dim, padding_idx=padding_idx, max_norm=max_norm, norm_type=norm_type,
                         scale_grad_by_freq=scale_grad_by_freq, sparse=sparse, _weight=_weight, device=device,
                         dtype=dtype)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Sanity check
        if torch.any(torch.logical_or(0 > input, input >= self.original_num_embeddings)):
            raise IndexError(
                f"Input is required to be in [0, {self.original_num_embeddings}[, got min: {torch.min(input)} and max: {torch.max(input)}")

        # `0` if input is in the correct interval, else `1`
        input_mask = torch.logical_or(self.min_id > input, input >= self.max_id)
        # translate for [0, self.max_id - self.min_id[
        input = input - self.min_id
        # default all out of bounds values to `0`
        input[input_mask] = 0
        out = super().forward(input)
        out[input_mask] = 0.0
        torch.distributed.all_reduce(out, group=self.process_group)
        return out


class PositionRotaryEmbedding(RotaryEmbedding):
    def _update_cos_sin_cache(self, dtype, device, seqlen):
        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if (seqlen > self._seq_len_cached or self._cos_cached.device != device
                or self._cos_cached.dtype != dtype):
            self._seq_len_cached = seqlen
            t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
            # Don't do einsum, it converts fp32 to fp16
            # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            freqs = torch.outer(t, self.inv_freq.to(device=t.device))
            if self.scale is None:
                self._cos_cached = torch.cos(freqs).to(dtype)
                self._sin_cached = torch.sin(freqs).to(dtype)
            else:
                power = ((torch.arange(seqlen, dtype=self.scale.dtype, device=self.scale.device)
                          - seqlen // 2) / self.scale_base)
                scale = self.scale.to(device=power.device) ** power.unsqueeze(1)
                # We want the multiplication by scale to happen in fp32
                self._cos_cached = (torch.cos(freqs) * scale).to(dtype)
                self._sin_cached = (torch.sin(freqs) * scale).to(dtype)
                self._cos_k_cached = (torch.cos(freqs) / scale).to(dtype)
                self._sin_k_cached = (torch.sin(freqs) / scale).to(dtype)

    def forward(self, qkv: torch.Tensor, position_ids: torch.Tensor, max_s: int):
        self._update_cos_sin_cache(qkv.dtype, qkv.device, max_s)

        q1, q2, k1, k2, cos, sin = _prepare_rotary(qkv, self._cos_cached, self._sin_cached, position_ids)
        rotary_emb.apply_rotary(q1, q2, cos, sin, q1, q2, False)
        rotary_emb.apply_rotary(k1, k2, cos, sin, k1, k2, False)
        return qkv


@torch.jit.script
def _prepare_rotary(qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor):
    cos = torch.index_select(cos, 0, position_ids)
    sin = torch.index_select(sin, 0, position_ids)

    rotary_dim = cos.shape[-1]
    q1 = qkv[:, 0, :, :rotary_dim]
    q2 = qkv[:, 0, :, rotary_dim:2*rotary_dim]
    k1 = qkv[:, 1, :, :rotary_dim]
    k2 = qkv[:, 1, :, rotary_dim: 2*rotary_dim]

    return q1, q2, k1, k2, cos.unsqueeze(1), sin.unsqueeze(1)


class FlashNeoxAttention(torch.nn.Module):
    def __init__(
            self, num_heads, hidden_size, rotary_pct, rotary_emb_base, process_group=None
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_size = hidden_size // num_heads

        rotary_ndims = int(self.head_size * rotary_pct)
        self.rotary_emb = PositionRotaryEmbedding(rotary_ndims, base=rotary_emb_base)
        self.softmax_scale = self.head_size ** (-0.5)

        if process_group is None:
            self.query_key_value = nn.Linear(hidden_size, 3 * hidden_size)
            self.dense = nn.Linear(hidden_size, hidden_size)
        else:
            self.num_heads = self.num_heads // process_group.size()
            self.query_key_value = TensorParallelColumnLinear(
                hidden_size,
                3 * hidden_size,
                process_group=process_group,
            )
            self.dense = TensorParallelRowLinear(
                hidden_size,
                hidden_size,
                process_group=process_group,
            )
        self.swap_dims = False

    def _swap_dims(self):
        self.query_key_value.weight = torch.nn.Parameter(
            self.query_key_value.weight.view(self.num_heads, 3, self.head_size, self.hidden_size)
            .permute(1, 0, 2, 3).reshape(-1, self.hidden_size)
        )
        self.query_key_value.bias = torch.nn.Parameter(
            self.query_key_value.bias.view(self.num_heads, 3, self.head_size)
            .permute(1, 0, 2).reshape(-1)
        )
        self.swap_dims = True

    def forward(
            self, hidden_states, position_ids, cu_seqlens, max_s, layer_past, prefill
    ):
        if not self.swap_dims:
            self._swap_dims()

        qkv = self.query_key_value(hidden_states)
        qkv = qkv.view(-1, 3, self.num_heads, self.head_size)
        qkv_rot = self.rotary_emb(qkv, position_ids, max_s)

        if prefill:
            layer_past[...] = qkv_rot[:, 1:]

            attn_output = torch.empty_like(qkv[:, 0])
            flash_attn_cuda.fwd(
                qkv[:, 0], qkv[:, 1], qkv[:, 2], attn_output, cu_seqlens, cu_seqlens, max_s, max_s, 0.0,
                self.softmax_scale,
                False, True, False, 0, None
            )
        else:
            query = qkv_rot[:, 0]
            layer_past[cu_seqlens[1:] - 1] = qkv_rot[:, 1:]

            attn_output = torch.empty_like(query)
            flash_attn_cuda.fwd(
                query, layer_past[:, 0], layer_past[:, 1], attn_output,
                torch.arange(len(cu_seqlens), dtype=torch.int32).to(
                    query.device
                ), cu_seqlens, torch.tensor(1, dtype=torch.int32).to(query.device), max_s, 0.0,
                self.softmax_scale,
                False, False, False, 0, None
            )

        return self.dense(attn_output.view(-1, self.num_heads * self.head_size))


class FlashMLP(nn.Module):
    def __init__(self, act, hidden_size, intermediate_size, process_group=None):
        super().__init__()
        assert "gelu" in act
        # if "gelu" in act:
        #     act = "gelu_approx"
        # assert act in ["gelu_approx", "relu"]
        self.act = lambda x: F.gelu(x, approximate="tanh")

        if process_group is None:
            self.dense_h_to_4h = nn.Linear(hidden_size, intermediate_size)
            self.dense_4h_to_h = nn.Linear(intermediate_size, hidden_size)
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
            )
        self.heuristic = "auto"
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
        self.input_layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.attention = FlashNeoxAttention(
            num_heads, hidden_size, rotary_pct, rotary_emb_base, process_group
        )
        self.mlp = FlashMLP(act, hidden_size, intermediate_size, process_group)

    def forward(
            self,
            hidden_states,
            residual,
            position_ids,
            cu_seqlens,
            max_s,
            layer_past,
            prefill,
    ):
        if self.use_parallel_residual:
            attn_output = self.attention(
                self.input_layernorm(hidden_states), position_ids, cu_seqlens, max_s, layer_past, prefill
            )

            mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
            return mlp_output + attn_output + hidden_states, None

        else:
            raise NotImplementedError
            hidden_states, residual = dropout_add_layer_norm(
                hidden_states,
                residual,
                self.input_layernorm.weight,
                self.input_layernorm.bias,
                0.0,
                self.input_layernorm.eps,
                rowscale=None,
                prenorm=True,
                residual_in_fp32=True,
            )

            hidden_states = self.attention(
                hidden_states, position_ids, cu_seqlens, max_s, layer_past, prefill
            )

            hidden_states, residual = dropout_add_layer_norm(
                hidden_states,
                residual,
                self.post_attention_layernorm.weight,
                self.post_attention_layernorm.bias,
                0.0,
                self.post_attention_layernorm.eps,
                rowscale=None,
                prenorm=True,
                residual_in_fp32=True,
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
        self.final_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

        self.gradient_checkpointing = False

        self.head_size = self.layers[0].attention.head_size
        self.num_heads = self.layers[0].attention.num_heads

    def forward(
            self,
            input_ids,
            position_ids,
            cu_seqlens,
            max_s,
            past_key_values=None,
    ):
        hidden_states = self.embed_in(input_ids)

        prefill = False
        if past_key_values is None:
            past_key_values = hidden_states.new_empty(
                (
                    len(self.layers),
                    len(hidden_states),
                    2,
                    self.num_heads,
                    self.head_size,
                )
            )
            prefill = True

        residual = None
        for i, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                hidden_states,
                residual,
                position_ids,
                cu_seqlens,
                max_s,
                past_key_values[i],
                prefill,
            )

        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states, past_key_values


class FlashGPTNeoXForCausalLM(FlashGPTNeoXPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        if config.tp_parallel:
            process_group = torch.distributed.distributed_c10d._get_default_group()
        else:
            process_group = None

        self.gpt_neox = FlashGPTNeoXModel(config, process_group)

        if self.gpt_neox.tp_embeddings:
            self.embed_out = nn.Linear(
                config.hidden_size,
                config.vocab_size // process_group.size(),
                bias=False,
            )
        else:
            self.embed_out = nn.Linear(
                config.hidden_size, config.vocab_size, bias=False
            )

    def forward(
            self,
            input_ids,
            position_ids,
            cu_seqlens,
            max_s,
            past_key_values=None,
    ):
        hidden_states, present = self.gpt_neox(
            input_ids, position_ids, cu_seqlens, max_s, past_key_values
        )
        return self.embed_out(hidden_states), present
