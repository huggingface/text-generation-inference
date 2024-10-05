# coding=utf-8
# Copyright 2024 Starcoder2 AI and the HuggingFace Inc. team. All rights reserved.
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
from transformers.configuration_utils import PretrainedConfig
from typing import Optional, List, Tuple

from text_generation_server.layers.attention import (
    paged_attention,
    attention,
    Seqlen,
)
from text_generation_server.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    SpeculativeHead,
    get_linear,
)
from text_generation_server.layers.attention import PREFILL_IN_KV_CACHE
from text_generation_server.layers.layernorm import (
    FastLayerNorm,
    FastRMSNorm,
)
from text_generation_server.layers.rotary import (
    PositionRotaryEmbedding,
)
from text_generation_server.utils.weights import UnquantizedWeight


class Starcoder2Config(PretrainedConfig):
    model_type = "starcoder2"

    def __init__(
        self,
        vocab_size=49152,
        hidden_size=3072,
        intermediate_size=12288,
        num_hidden_layers=30,
        num_attention_heads=24,
        num_key_value_heads=2,
        mlp_type="default",
        hidden_act="gelu_pytorch_tanh",
        max_position_embeddings=4096,
        initializer_range=0.018042,
        norm_type="layer_norm",
        norm_epsilon=1e-5,
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
        rope_theta=10000.0,
        sliding_window=None,
        attention_dropout=0.0,
        residual_dropout=0.0,
        embedding_dropout=0.0,
        use_bias: bool = True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.sliding_window = sliding_window
        self.use_bias = use_bias

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.mlp_type = mlp_type
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.norm_type = norm_type
        self.norm_epsilon = norm_epsilon
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.residual_dropout = residual_dropout
        self.embedding_dropout = embedding_dropout

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )


def load_attention(config, prefix, weights):
    if config.num_attention_heads != config.num_key_value_heads:
        return _load_gqa(config, prefix, weights)
    else:
        return TensorParallelColumnLinear.load_multi(
            config,
            prefixes=[f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"],
            dim=0,
            weights=weights,
            bias=config.use_bias,
        )


def _load_gqa(config, prefix: str, weights):
    assert config.hidden_size % config.num_attention_heads == 0
    assert config.num_attention_heads % weights.process_group.size() == 0

    weight = weights.get_multi_weights_col(
        prefixes=[f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"],
        dim=0,
    )

    if isinstance(weight, UnquantizedWeight):
        weight.weight = weight.weight.to(dtype=weights.dtype).to(device=weights.device)

        head_size = config.hidden_size // config.num_attention_heads
        num_heads = config.num_attention_heads // weights.process_group.size()
        num_key_value_heads = config.num_key_value_heads // weights.process_group.size()
        assert list(weight.weight.shape) == [
            (num_heads + 2 * num_key_value_heads) * head_size,
            config.hidden_size,
        ], f"{list(weight.weight.shape)} != {[(num_heads + 2 * config.num_key_value_heads) * head_size, config.hidden_size]}"

    if config.use_bias:
        w = [
            weights.get_sharded(f"{p}.bias", dim=0)
            for p in [f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"]
        ]
        bias = torch.cat(w, dim=0).to(dtype=weights.dtype).to(device=weights.device)
    else:
        bias = None

    return TensorParallelColumnLinear(get_linear(weight, bias=bias))


class Starcoder2Attention(torch.nn.Module):
    def __init__(
        self,
        prefix: str,
        config,
        weights,
    ):
        super().__init__()
        self.max_past = (
            config.sliding_window if config.sliding_window is not None else -1
        )
        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_heads

        self.rotary_emb = PositionRotaryEmbedding.static(
            config=config,
            dim=self.head_size,
            base=config.rope_theta,
            device=weights.device,
        )

        self.softmax_scale = self.head_size**-0.5

        if self.num_heads % weights.process_group.size() != 0:
            raise ValueError(
                f"`num_heads` must be divisible by `num_shards` (got `num_heads`: {self.num_heads} "
                f"and `num_shards`: {weights.process_group.size()}"
            )
        self.num_heads = self.num_heads // weights.process_group.size()
        self.num_key_value_heads = (
            config.num_key_value_heads // weights.process_group.size()
        )

        self.query_key_value = load_attention(config, prefix, weights)

        self.o_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.o_proj",
            weights=weights,
            bias=config.use_bias,
        )
        self.num_groups = self.num_heads // self.num_key_value_heads
        self.kv_head_mapping = torch.arange(
            0, self.num_key_value_heads, dtype=torch.int32, device=weights.device
        ).repeat_interleave(self.num_groups)

    def forward(
        self,
        hidden_states,
        cos,
        sin,
        cu_seqlen_prefill,
        kv_cache,
        block_tables,
        slots,
        seqlen,
        max_s,
        prefill_cache_indices,
    ):
        qkv = self.query_key_value(hidden_states)
        query, kv = qkv.split(
            [
                self.head_size * self.num_heads,
                2 * self.head_size * self.num_key_value_heads,
            ],
            dim=1,
        )
        query = query.view(-1, self.num_heads, self.head_size)
        kv = kv.view(-1, 2, self.num_key_value_heads, self.head_size)

        self.rotary_emb(query, torch.select(kv, dim=1, index=0), cos, sin)

        if prefill_cache_indices is not None:
            kv_to_cache = kv[prefill_cache_indices]
        else:
            kv_to_cache = kv

        kv_cache.store(key=kv_to_cache[:, 0], value=kv_to_cache[:, 1], slots=slots)

        # Prefill
        if cu_seqlen_prefill is not None:
            # flash attention
            attn_output = attention(
                query,
                kv_cache.key if PREFILL_IN_KV_CACHE else kv_to_cache[:, 0],
                kv_cache.value if PREFILL_IN_KV_CACHE else kv_to_cache[:, 1],
                seqlen,
                block_tables,
                self.softmax_scale,
                window_size_left=self.max_past,
            )
        # Decode
        else:
            attn_output = paged_attention(
                query,
                kv_cache.key,
                kv_cache.value,
                self.kv_head_mapping,
                self.softmax_scale,
                block_tables,
                seqlen,
                max_s,
            )

        return self.o_proj(attn_output.view(-1, self.num_heads * self.head_size))


class Starcoder2MLP(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        act = config.hidden_act
        self.act = (
            ACT2FN[act]
            if "gelu" not in act
            else lambda x: torch.nn.functional.gelu(
                x,
                approximate=(
                    "tanh" if act in ["gelu_fast", "gelu_pytorch_tanh"] else "none"
                ),
            )
        )
        # Fuse gate and up proj
        self.c_fc = TensorParallelColumnLinear.load(
            config,
            prefix=f"{prefix}.c_fc",
            weights=weights,
            bias=config.use_bias,
        )
        self.c_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.c_proj",
            weights=weights,
            bias=config.use_bias,
        )

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        return self.c_proj(hidden_states)


class Starcoder2GatedMLP(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        act = config.hidden_act
        self.act = (
            ACT2FN[act]
            if "gelu" not in act
            else lambda x: torch.nn.functional.gelu(
                x,
                approximate=(
                    "tanh" if act in ["gelu_fast", "gelu_pytorch_tanh"] else "none"
                ),
            )
        )
        # Fuse gate and up proj
        self.gate_up_proj = TensorParallelColumnLinear.load_multi(
            config,
            prefixes=[f"{prefix}.gate_proj", f"{prefix}.up_proj"],
            weights=weights,
            dim=0,
            bias=config.use_bias,
        )
        self.down_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.down_proj",
            weights=weights,
            bias=config.use_bias,
        )
        self.intermediate_size = (
            config.intermediate_size // weights.process_group.size()
        )

    def forward(self, hidden_states):
        gate_up_states = self.gate_up_proj(hidden_states)
        gate_up_states = gate_up_states.view(-1, 2, self.intermediate_size)
        return self.down_proj(self.act(gate_up_states[:, 0]) * gate_up_states[:, 1])


STARCODER2_NORMALIZATION_CLASSES = {
    "layer_norm": FastLayerNorm,
    "rms_norm": FastRMSNorm,
}

STARCODER2_MLP_CLASSES = {
    "default": Starcoder2MLP,
    "gated": Starcoder2GatedMLP,
}


class Starcoder2Layer(nn.Module):
    def __init__(self, layer_id, config, weights):
        super().__init__()
        prefix = f"model.layers.{layer_id}"
        self.self_attn = Starcoder2Attention(
            prefix=f"{prefix}.self_attn", config=config, weights=weights
        )

        self.mlp = STARCODER2_MLP_CLASSES[config.mlp_type](
            prefix=f"{prefix}.mlp", config=config, weights=weights
        )

        self.input_layernorm = STARCODER2_NORMALIZATION_CLASSES[config.norm_type].load(
            prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.norm_epsilon
        )
        self.post_attention_layernorm = STARCODER2_NORMALIZATION_CLASSES[
            config.norm_type
        ].load(
            prefix=f"{prefix}.post_attention_layernorm",
            weights=weights,
            eps=config.norm_epsilon,
        )

    def forward(
        self,
        hidden_states,
        residual,
        cos,
        sin,
        cu_seqlen_prefill,
        kv_cache,
        block_tables,
        slots,
        seqlen,
        max_s,
        prefill_cache_indices,
    ):
        normed_hidden_states, res = self.input_layernorm(hidden_states, residual)

        # Self Attention
        attn_output = self.self_attn(
            normed_hidden_states,
            cos,
            sin,
            cu_seqlen_prefill,
            kv_cache,
            block_tables,
            slots,
            seqlen,
            max_s,
            prefill_cache_indices,
        )

        # faster post attention rms norm
        normed_attn_res_output, attn_res = self.post_attention_layernorm(
            attn_output, res
        )

        mlp_output = self.mlp(normed_attn_res_output)

        return mlp_output, attn_res


class Starcoder2Model(torch.nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()

        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.embed_tokens = TensorParallelEmbedding(
            prefix=f"{prefix}.embed_tokens", weights=weights
        )
        self.layers = nn.ModuleList(
            [
                Starcoder2Layer(
                    layer_id,
                    config,
                    weights,
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = STARCODER2_NORMALIZATION_CLASSES[config.norm_type].load(
            prefix=f"{prefix}.norm", weights=weights, eps=config.norm_epsilon
        )

        self.gradient_checkpointing = False

        self.head_size = self.layers[0].self_attn.head_size
        self.num_heads = self.layers[0].self_attn.num_heads
        self.num_key_value_heads = self.layers[0].self_attn.num_key_value_heads

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlen_prefill: Optional[torch.Tensor],
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        block_tables: torch.Tensor,
        slots: torch.Tensor,
        seqlen: Seqlen,
        max_s: int,
        true_max_s: int,
        prefill_cache_indices: Optional[torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)

        # Get rotary cos and sin for this forward
        # Avoid to index in each layer
        cos, sin = self.layers[0].self_attn.rotary_emb.get_cos_sin(
            position_ids, true_max_s, hidden_states.dtype
        )

        residual = None
        for i, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                hidden_states,
                residual,
                cos,
                sin,
                cu_seqlen_prefill,
                kv_cache[i],
                block_tables,
                slots,
                seqlen,
                max_s,
                prefill_cache_indices,
            )

        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states


class FlashStarcoder2ForCausalLM(torch.nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()

        if not prefix:
            prefix = "model"
        else:
            prefix = f"{prefix}.model"

        self.model = Starcoder2Model(prefix, config, weights)
        try:
            self.lm_head = SpeculativeHead.load(
                config,
                prefix="lm_head",
                weights=weights,
            )
        except RuntimeError:
            self.lm_head = SpeculativeHead.load(
                config,
                prefix=f"{prefix}.embed_tokens",
                weights=weights,
            )

        self.max_past = config.sliding_window
        self.max_past_tensor = (
            torch.tensor(config.sliding_window, device=weights.device)
            if self.max_past is not None
            else None
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlen_prefill: Optional[torch.Tensor],
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        block_tables: torch.Tensor,
        slots: torch.Tensor,
        seqlen: Seqlen,
        max_s: int,
        prefill_cache_indices: Optional[torch.Tensor],
        lm_head_indices: Optional[torch.Tensor] = None,
        adapter_data: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        true_max_s = max_s
        if prefill_cache_indices is not None:
            # Slots also need to be sliced as it has the same size as the whole kv tensor
            slots = slots[prefill_cache_indices]
        elif self.max_past is not None:
            # Clamp in decode mode as paged attention requires clamped values whereas the flash attention
            # kernel requires the true values
            seqlen = seqlen.clamp(max=self.max_past_tensor)

        hidden_states = self.model(
            input_ids,
            position_ids,
            cu_seqlen_prefill,
            kv_cache,
            block_tables,
            slots,
            seqlen,
            max_s,
            true_max_s,
            prefill_cache_indices,
        )
        if lm_head_indices is not None:
            hidden_states = hidden_states[lm_head_indices]
        logits = self.lm_head(hidden_states)
        return logits
