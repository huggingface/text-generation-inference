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
from transformers.configuration_utils import PretrainedConfig
from typing import Optional, List, Tuple

from text_generation_server.utils.import_utils import SYSTEM
from text_generation_server.layers.attention import (
    paged_attention,
    attention,
    reshape_and_cache,
    Seqlen,
)
from text_generation_server.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    SpeculativeHead,
    TensorParallelMultiAdapterLinear,
    TensorParallelAdapterRowLinear,
)
from text_generation_server.layers.attention import PREFILL_IN_KV_CACHE
from text_generation_server.layers.rotary import PositionRotaryEmbedding
from text_generation_server.layers.layernorm import (
    FastRMSNorm,
)


if SYSTEM == "rocm":
    try:
        from vllm import _custom_C
    except Exception as e:
        raise ImportError(f"Could not load `vllm._custom_C`. Full error: {e}")


class MistralConfig(PretrainedConfig):
    model_type = "mistral"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=4096 * 32,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        sliding_window=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.sliding_window = sliding_window

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class MistralAttention(torch.nn.Module):
    def __init__(self, prefix: str, config, weights, layer_id):
        super().__init__()
        self.max_past = (
            config.sliding_window if config.sliding_window is not None else -1
        )
        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        if hasattr(config, "head_dim"):
            self.head_size = config.head_dim
        else:
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

        query_key_value = TensorParallelColumnLinear.load_multi(
            config,
            prefixes=[f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"],
            dim=0,
            weights=weights,
            bias=False,
        )

        self.query_key_value = TensorParallelMultiAdapterLinear.load(
            query_key_value,
            layer_id,
            ["q_proj", "k_proj", "v_proj"],
            sizes=[
                self.head_size * config.num_attention_heads,
                self.head_size * config.num_key_value_heads,
                self.head_size * config.num_key_value_heads,
            ],
            process_group=weights.process_group,
        )

        o_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.o_proj",
            weights=weights,
            bias=False,
        )
        self.o_proj = TensorParallelAdapterRowLinear.load(
            o_proj,
            layer_id,
            "o_proj",
            process_group=weights.process_group,
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
        adapter_data,
    ):
        qkv = self.query_key_value(hidden_states, adapter_data)
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

        reshape_and_cache(
            kv_to_cache[:, 0], kv_to_cache[:, 1], kv_cache[0], kv_cache[1], slots
        )

        # Prefill
        if cu_seqlen_prefill is not None:
            # flash attention
            attn_output = attention(
                query,
                kv_cache[0] if PREFILL_IN_KV_CACHE else kv_to_cache[:, 0],
                kv_cache[1] if PREFILL_IN_KV_CACHE else kv_to_cache[:, 1],
                seqlen,
                block_tables,
                self.softmax_scale,
                window_size_left=self.max_past,
            )
        # Decode
        else:
            attn_output = paged_attention(
                query,
                kv_cache[0],
                kv_cache[1],
                self.kv_head_mapping,
                self.softmax_scale,
                block_tables,
                seqlen,
                max_s,
            )

        return self.o_proj(
            attn_output.view(-1, self.num_heads * self.head_size), adapter_data
        )


class MistralMLP(nn.Module):
    def __init__(self, prefix: str, config, weights, layer_id):
        super().__init__()
        self.hidden_act = config.hidden_act
        self.act = (
            ACT2FN[self.hidden_act]
            if "gelu" not in self.hidden_act
            else lambda x: torch.nn.functional.gelu(
                x,
                approximate=(
                    "tanh"
                    if self.hidden_act in ["gelu_fast", "gelu_pytorch_tanh"]
                    else "none"
                ),
            )
        )
        # Fuse gate and up proj
        gate_up_proj = TensorParallelColumnLinear.load_multi(
            config,
            prefixes=[f"{prefix}.gate_proj", f"{prefix}.up_proj"],
            weights=weights,
            dim=0,
            bias=False,
        )
        self.gate_up_proj = TensorParallelMultiAdapterLinear.load(
            gate_up_proj,
            layer_id,
            ["gate_proj", "up_proj"],
            sizes=[
                config.intermediate_size,
                config.intermediate_size,
            ],
            process_group=weights.process_group,
        )

        down_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.down_proj",
            weights=weights,
            bias=False,
        )

        self.down_proj = TensorParallelAdapterRowLinear.load(
            down_proj,
            layer_id,
            "down_proj",
            process_group=weights.process_group,
        )
        self.intermediate_size = (
            config.intermediate_size // weights.process_group.size()
        )

        # TODO: This is a hotfix to be removed & properly refactored.
        self.quantize = config.quantize

    def forward(self, hidden_states, adapter_data):
        if (
            SYSTEM == "rocm"
            and self.hidden_act == "silu"
            and hidden_states.shape[0] == 1
            and not self.quantize
        ):
            out = torch.empty(
                hidden_states.shape[0],
                self.intermediate_size,
                dtype=hidden_states.dtype,
                device="cuda",
            )
            _custom_C.LLMM_Silu(
                self.gate_up_proj.base_layer.linear.weight, hidden_states, out, 8
            )
            return self.down_proj(out, adapter_data)
        else:
            gate_up_states = self.gate_up_proj(hidden_states, adapter_data)
            gate_up_states = gate_up_states.view(-1, 2, self.intermediate_size)
            return self.down_proj(
                self.act(gate_up_states[:, 0]) * gate_up_states[:, 1], adapter_data
            )


class MistralLayer(nn.Module):
    def __init__(self, prefix: str, config, weights, layer_id):
        super().__init__()
        self.self_attn = MistralAttention(
            prefix=f"{prefix}.self_attn",
            config=config,
            weights=weights,
            layer_id=layer_id,
        )
        self.mlp = MistralMLP(
            prefix=f"{prefix}.mlp", config=config, weights=weights, layer_id=layer_id
        )

        self.input_layernorm = FastRMSNorm.load(
            prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = FastRMSNorm.load(
            prefix=f"{prefix}.post_attention_layernorm",
            weights=weights,
            eps=config.rms_norm_eps,
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
        adapter_data,
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
            adapter_data,
        )

        # faster post attention rms norm
        normed_attn_res_output, attn_res = self.post_attention_layernorm(
            attn_output, res
        )

        mlp_output = self.mlp(normed_attn_res_output, adapter_data)

        return mlp_output, attn_res


class MistralModel(torch.nn.Module):
    def __init__(self, prefix: str, config, weights):
        super().__init__()

        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.layers = nn.ModuleList(
            [
                MistralLayer(
                    prefix=f"{prefix}.layers.{layer_id}",
                    config=config,
                    weights=weights,
                    layer_id=layer_id,
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = FastRMSNorm.load(
            prefix=f"{prefix}.norm", weights=weights, eps=config.rms_norm_eps
        )

        self.gradient_checkpointing = False

        self.head_size = self.layers[0].self_attn.head_size
        self.num_heads = self.layers[0].self_attn.num_heads
        self.num_key_value_heads = self.layers[0].self_attn.num_key_value_heads

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlen_prefill: Optional[torch.Tensor],
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        block_tables: torch.Tensor,
        slots: torch.Tensor,
        seqlen: Seqlen,
        max_s: int,
        true_max_s: int,
        prefill_cache_indices: Optional[torch.Tensor],
        adapter_data: Optional[torch.Tensor] = None,
    ):
        hidden_states = inputs_embeds
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
                adapter_data,
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class FlashMistralForCausalLM(torch.nn.Module):
    def __init__(self, prefix: str, config, weights, name=None):
        if name is None:
            name = "model"
        super().__init__()
        self.embed_tokens = TensorParallelEmbedding(
            prefix=(
                f"{name}.embed_tokens"
                if not prefix
                else f"{prefix}.{name}.embed_tokens"
            ),
            weights=weights,
        )
        self.model = MistralModel(
            prefix=name if not prefix else f"{prefix}.{name}",
            config=config,
            weights=weights,
        )
        self.lm_head = SpeculativeHead.load(
            config,
            # TODO dirty hack for idefics2.
            prefix=(
                "lm_head" if not prefix or name != "model" else f"{prefix}.lm_head"
            ),
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

        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = self.model(
            inputs_embeds,
            position_ids,
            cu_seqlen_prefill,
            kv_cache,
            block_tables,
            slots,
            seqlen,
            max_s,
            true_max_s,
            prefill_cache_indices,
            adapter_data,
        )
        if lm_head_indices is not None:
            hidden_states = hidden_states[lm_head_indices]
        logits = self.lm_head(hidden_states)
        return logits
