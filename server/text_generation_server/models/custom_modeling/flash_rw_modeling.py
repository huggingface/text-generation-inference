import os

import torch
import torch.distributed

from torch import nn
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
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


class RWConfig(PretrainedConfig):
    attribute_map = {
        "num_hidden_layers": "n_layer",
        "num_attention_heads": "n_head",
    }

    def __init__(
        self,
        model_type="RefinedWeb",
        vocab_size=250880,
        hidden_size=64,
        n_layer=2,
        n_head=8,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        use_cache=True,
        bos_token_id=1,
        eos_token_id=2,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        n_head_kv=None,
        multi_query=False,
        alibi=False,
        bias=False,
        parallel_attn=False,
        **kwargs,
    ):
        if alibi:
            raise NotImplementedError(
                "alibi is not supported by this version of the model"
            )

        self.model_type = model_type
        self.alibi = False
        self.rotary = True

        self.vocab_size = vocab_size
        # Backward compatibility with n_embed kwarg
        n_embed = kwargs.pop("n_embed", None)
        self.hidden_size = hidden_size if n_embed is None else n_embed
        self.n_layer = n_layer
        self.n_head = n_head
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.bias = bias
        self.parallel_attn = parallel_attn

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        if n_head_kv is not None:
            self.n_head_kv = n_head_kv
        else:
            self.n_head_kv = 1 if multi_query else n_head

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)


class FlashRWAttention(torch.nn.Module):
    def __init__(
        self,
        num_heads,
        num_heads_kv,
        hidden_size,
        bias,
        process_group=None,
        reduce=True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_heads_kv = num_heads_kv
        self.hidden_size = hidden_size
        self.head_size = hidden_size // num_heads

        self.rotary_emb = PositionRotaryEmbedding(self.head_size, base=10000)
        self.softmax_scale = self.head_size ** (-0.5)

        if process_group is None:
            self.query_key_value = FastLinear(
                hidden_size,
                self.head_size * (self.num_heads + 2 * self.num_heads_kv),
                bias=bias,
            )
            self.dense = FastLinear(hidden_size, hidden_size, bias=bias)
        else:
            self.query_key_value = TensorParallelColumnLinear(
                hidden_size,
                self.head_size * (self.num_heads + 2 * self.num_heads_kv),
                bias=bias,
                process_group=process_group,
            )
            self.dense = TensorParallelRowLinear(
                hidden_size,
                hidden_size,
                bias=bias,
                process_group=process_group,
                reduce=reduce,
            )
            self.num_heads = self.num_heads // process_group.size()

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

        # Split query from key_value
        query, kv = qkv.split(
            [self.head_size * self.num_heads, 2 * self.head_size * self.num_heads_kv],
            dim=1,
        )

        # Prepare query and key_value for indexing
        query = query.view(-1, self.num_heads, self.head_size)
        kv = kv.view(-1, 2, self.num_heads_kv, self.head_size)

        # Inplace rotary
        self.rotary_emb(query, cos, sin)
        self.rotary_emb(kv[:, 0], cos, sin)

        # Prefill
        if layer_past_present_indices is None:
            # Copy to layer past
            layer_past[...] = kv
            # Expand to query shape
            kv = kv.expand(-1, 2, self.num_heads, self.head_size)

            # output
            attn_output = torch.empty_like(query)
            # flash attention
            flash_attn_cuda.fwd(
                query,
                kv[:, 0],
                kv[:, 1],
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
            # Add present to the layer_past tensor at the correct indices
            layer_past[layer_past_present_indices] = kv
            # Expand to query shape
            kv = layer_past.expand(-1, 2, self.num_heads, self.head_size)

            # output
            attn_output = torch.empty_like(query)
            # flash attention
            flash_attn_cuda.fwd(
                query,
                kv[:, 0],
                kv[:, 1],
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


class FlashRWLargeAttention(torch.nn.Module):
    def __init__(
        self,
        num_heads,
        num_heads_kv,
        hidden_size,
        bias,
        process_group=None,
        reduce=True,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.head_size = hidden_size // num_heads

        self.rotary_emb = PositionRotaryEmbedding(self.head_size, base=10000)
        self.softmax_scale = self.head_size ** (-0.5)

        self.num_groups = num_heads // (num_heads_kv * 2)
        self.num_heads = num_heads // self.num_groups
        self.num_heads_kv = num_heads_kv // self.num_groups

        if process_group is None:
            self.query_key_value = FastLinear(
                hidden_size,
                self.num_groups
                * self.head_size
                * (self.num_heads + 2 * self.num_heads_kv),
                bias=bias,
            )
            self.dense = FastLinear(hidden_size, hidden_size, bias=bias)
        else:
            if process_group.size() > self.num_groups:
                raise NotImplementedError(
                    f"Tensor Parallelism is not implemented for world_size > n groups"
                )

            self.query_key_value = TensorParallelColumnLinear(
                hidden_size,
                self.num_groups
                * self.head_size
                * (self.num_heads + 2 * self.num_heads_kv),
                bias=bias,
                process_group=process_group,
            )
            self.dense = TensorParallelRowLinear(
                hidden_size,
                hidden_size,
                bias=bias,
                process_group=process_group,
                reduce=reduce,
            )

            self.num_groups = self.num_groups // process_group.size()

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
        qkv = qkv.view(-1, self.num_groups, self.num_heads + 2, self.head_size)

        # Split on group dimension
        query, kv = qkv.split(
            [self.num_heads, 2],
            dim=2,
        )
        # Merge groups and heads
        query = query.reshape(-1, self.num_groups * self.num_heads, self.head_size)

        # Inplace rotary
        self.rotary_emb(query, cos, sin)
        self.rotary_emb(kv[:, :, 0], cos, sin)

        # Prefill
        if layer_past_present_indices is None:
            # Copy to layer past
            layer_past[...] = kv
            # Expand to query shape
            kv = (
                kv.unsqueeze(2)
                .expand(-1, self.num_groups, self.num_heads, 2, self.head_size)
                .reshape(-1, self.num_groups * self.num_heads, 2, self.head_size)
            )

            # output
            attn_output = torch.empty_like(query)
            # flash attention
            flash_attn_cuda.fwd(
                query,
                kv[:, :, 0],
                kv[:, :, 1],
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
            # Add present to the layer_past tensor at the correct indices
            layer_past[layer_past_present_indices] = kv
            # Expand to query shape
            kv = (
                layer_past.unsqueeze(2)
                .expand(-1, self.num_groups, self.num_heads, 2, self.head_size)
                .reshape(-1, self.num_groups * self.num_heads, 2, self.head_size)
            )

            # output
            attn_output = torch.empty_like(query)
            # flash attention
            flash_attn_cuda.fwd(
                query,
                kv[:, :, 0],
                kv[:, :, 1],
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

        return self.dense(
            attn_output.view(-1, self.num_groups * self.num_heads * self.head_size)
        )


class FlashMLP(nn.Module):
    def __init__(self, hidden_size, bias, process_group=None, reduce=True):
        super().__init__()
        self.act = torch.nn.functional.gelu

        if process_group is None:
            self.dense_h_to_4h = FastLinear(hidden_size, 4 * hidden_size, bias=bias)
            self.dense_4h_to_h = FastLinear(4 * hidden_size, hidden_size, bias=bias)
        else:
            self.dense_h_to_4h = TensorParallelColumnLinear(
                hidden_size,
                4 * hidden_size,
                bias=bias,
                process_group=process_group,
            )
            self.dense_4h_to_h = TensorParallelRowLinear(
                4 * hidden_size,
                hidden_size,
                bias=bias,
                process_group=process_group,
                reduce=reduce,
            )
        self.process_group = process_group

    def forward(self, hidden_states):
        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dense_4h_to_h(hidden_states)
        return hidden_states


class FlashRWLayer(nn.Module):
    def __init__(
        self,
        num_heads,
        num_heads_kv,
        hidden_size,
        bias,
        layer_norm_eps,
        parallel_attn,
        process_group=None,
    ):
        super().__init__()

        self.parallel_attn = parallel_attn

        self.input_layernorm = FastLayerNorm(hidden_size, eps=layer_norm_eps)
        self.self_attention = FlashRWAttention(
            num_heads,
            num_heads_kv,
            hidden_size,
            bias,
            process_group=process_group,
            reduce=False,
        )
        self.post_attention_layernorm = (
            FastLayerNorm(hidden_size, eps=layer_norm_eps)
            if not parallel_attn
            else None
        )

        self.mlp = FlashMLP(
            hidden_size, bias, process_group=process_group, reduce=False
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
        if self.parallel_attn:
            ln_hidden_states, residual = self.input_layernorm(hidden_states, residual)

            attn_output = self.self_attention(
                ln_hidden_states,
                cos,
                sin,
                cu_seqlens,
                max_s,
                layer_past,
                layer_past_present_indices,
                cu_seqlens_q,
            )

            mlp_output = self.mlp(ln_hidden_states)
            intermediate = mlp_output + attn_output

            # Only reduce once and after the addition instead of once per layer
            if self.process_group is not None:
                torch.distributed.all_reduce(intermediate, group=self.process_group)

            return intermediate, residual
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

            hidden_states = self.self_attention(
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


class FlashRWLargeLayer(nn.Module):
    def __init__(
        self,
        num_heads,
        num_heads_kv,
        hidden_size,
        bias,
        layer_norm_eps,
        process_group=None,
    ):
        super().__init__()
        self.ln_attn = FastLayerNorm(hidden_size, eps=layer_norm_eps)
        self.ln_mlp = FastLayerNorm(hidden_size, eps=layer_norm_eps)

        self.self_attention = FlashRWLargeAttention(
            num_heads,
            num_heads_kv,
            hidden_size,
            bias,
            process_group=process_group,
            reduce=False,
        )

        self.mlp = FlashMLP(
            hidden_size, bias, process_group=process_group, reduce=False
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
        ln_attn, residual = self.ln_attn(hidden_states, residual)
        ln_mlp, _ = self.ln_mlp(residual)

        # Self attention.
        attn_output = self.self_attention(
            ln_attn,
            cos,
            sin,
            cu_seqlens,
            max_s,
            layer_past,
            layer_past_present_indices,
            cu_seqlens_q,
        )

        # MLP.
        mlp_output = self.mlp(ln_mlp)

        intermediate = attn_output + mlp_output

        # Only reduce once and after the addition instead of once per layer
        if self.process_group is not None:
            torch.distributed.all_reduce(intermediate, group=self.process_group)

        return intermediate, residual


class FlashRWPreTrainedModel(PreTrainedModel):
    config_class = RWConfig


class FlashRWModel(FlashRWPreTrainedModel):
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
            self.word_embeddings = TensorParallelEmbedding(
                config.vocab_size, config.hidden_size, process_group=process_group
            )
        else:
            self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)

        if config.model_type == "RefinedWebModel":
            self.h = nn.ModuleList(
                [
                    FlashRWLayer(
                        config.n_head,
                        config.n_head_kv,
                        config.hidden_size,
                        config.bias,
                        config.layer_norm_epsilon,
                        config.parallel_attn,
                        process_group,
                    )
                    for _ in range(config.num_hidden_layers)
                ]
            )
            self.cache_size = (
                2,
                self.h[0].self_attention.num_heads_kv,
                self.h[0].self_attention.head_size,
            )
        elif config.model_type == "RefinedWeb":
            self.h = nn.ModuleList(
                [
                    FlashRWLargeLayer(
                        config.n_head,
                        config.n_head_kv,
                        config.hidden_size,
                        config.bias,
                        config.layer_norm_epsilon,
                        process_group,
                    )
                    for _ in range(config.num_hidden_layers)
                ]
            )
            self.cache_size = (
                self.h[0].self_attention.num_groups,
                2,
                self.h[0].self_attention.head_size,
            )
        else:
            raise NotImplementedError(
                f"model_type {config.model_type} is not supported."
            )

        self.ln_f = FastLayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        self.head_size = self.h[0].self_attention.head_size

    def post_load_weights(self, quantize: Optional[str] = None):
        if isinstance(self.word_embeddings, TensorParallelEmbedding):
            self.word_embeddings.add_null_idx()
        for layer in self.h:
            layer: FlashRWLayer
            layer.self_attention.query_key_value.prepare_weights(quantize)
            layer.self_attention.dense.prepare_weights(quantize)
            layer.mlp.dense_h_to_4h.prepare_weights(quantize)
            layer.mlp.dense_4h_to_h.prepare_weights(quantize)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # Pop here as we will replace the layer in our own logic and don't want from_pretrained
        # to do it for us
        load_in_8bit = kwargs.pop("load_in_8bit", False)
        model = super(FlashRWModel, cls).from_pretrained(
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
        hidden_states = self.word_embeddings(input_ids)

        # Prefill
        if past_key_values is None:
            # Create past tensor
            past_key_values = hidden_states.new_empty(
                (
                    len(self.h),
                    len(hidden_states)
                    if pre_allocate_past_size is None
                    else pre_allocate_past_size,
                    *self.cache_size,
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
        cos, sin = self.h[0].self_attention.rotary_emb.get_cos_sin(
            position_ids, max_s, hidden_states.dtype
        )

        residual = None
        for i, layer in enumerate(self.h):
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

        hidden_states, _ = self.ln_f(hidden_states, residual)

        return hidden_states, past_key_values


class FlashRWForCausalLM(FlashRWPreTrainedModel):
    def __init__(self, config, process_group=None):
        super().__init__(config)

        self.process_group = process_group
        if self.process_group is not None:
            self.world_size = self.process_group.size()
        else:
            self.world_size = 1

        self.transformer = FlashRWModel(config, process_group)

        if self.transformer.tp_embeddings:
            self.lm_head = FastLinear(
                config.hidden_size,
                config.vocab_size // process_group.size(),
                bias=False,
            )
        else:
            self.lm_head = FastLinear(config.hidden_size, config.vocab_size, bias=False)

    def post_load_weights(self, quantize: Optional[str] = None):
        self.transformer.post_load_weights(quantize)
        self.lm_head.prepare_weights()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # Pop here as we will replace the layer in our own logic and don't want from_pretrained
        # to do it for us
        load_in_8bit = kwargs.pop("load_in_8bit", False)
        model = super(FlashRWForCausalLM, cls).from_pretrained(
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
        hidden_states, present = self.transformer(
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

        if self.transformer.tp_embeddings:
            # Logits are sharded, so we need to gather them
            world_logits = [torch.empty_like(logits) for _ in range(self.world_size)]
            torch.distributed.all_gather(world_logits, logits, group=self.process_group)
            world_logits = torch.cat(world_logits, dim=1)

            return world_logits, present
        return logits, present
