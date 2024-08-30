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
from typing import Optional, List, Tuple

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
    get_linear,
)
from text_generation_server.layers.rotary import (
    PositionRotaryEmbedding,
)
from text_generation_server.layers.layernorm import (
    FastLayerNorm,
)
from text_generation_server.utils.import_utils import SYSTEM


def load_attention(config, prefix: str, weights):
    return TensorParallelColumnLinear.load_multi(
        config,
        prefixes=[f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"],
        dim=0,
        weights=weights,
        bias=False,
    )


def load_row(config, prefix: str, weights, bias: bool):
    weight = weights.get_weights_row(prefix)

    if bias and weights.process_group.rank() == 0:
        # Rank is only on the first rank process
        bias = weights.get_tensor(f"{prefix}.bias")
    else:
        bias = None

    linear = get_linear(weight, bias)
    return TensorParallelRowLinear(linear, process_group=weights.process_group)


class GPTJRotary(PositionRotaryEmbedding):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ):
        # Such controlflows may add some overhead.
        if SYSTEM == "cuda":
            import rotary_emb

            q1 = query[..., ::2]
            q2 = query[..., 1::2]

            rotary_emb.apply_rotary(q1, q2, cos, sin, q1, q2, False)

            k1 = key[..., ::2]
            k2 = key[..., 1::2]

            rotary_emb.apply_rotary(k1, k2, cos, sin, k1, k2, False)
        elif SYSTEM == "rocm":
            from vllm._C import ops

            # NOTE: On RoCm systems, we use a ROPE implementatation adapted from VLLM which launches a single kernel for both query/key, contrary to flash-attn implementation used on NVIDIA systems.
            # Compiling flash-attn rotary on RoCm, it appears hipcc is unable to unroll loops, resulting in an even slower inference compared to eager: https://github.com/pytorch/pytorch/issues/113773

            head_size = query.shape[-1]

            # Inplace operation, updating query and key.
            ops.rotary_embedding(query, key, head_size, cos, sin, False)
        elif SYSTEM == "ipex":
            import intel_extension_for_pytorch as ipex

            ipex.llm.functional.rotary_embedding(
                query, key, sin, cos, query.size(-1), False
            )
        else:
            raise ValueError(
                "Your system seem to be not supported. Please check your install or open an issue at https://github.com/huggingface/text-generation-inference/issues with a clear reproduction."
            )


class FlashGPTJAttention(torch.nn.Module):
    def __init__(
        self,
        prefix: str,
        config,
        weights,
    ):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size

        self.head_size = self.hidden_size // self.num_heads
        self.softmax_scale = self.head_size**-0.5
        self.rotary_dim = config.rotary_dim

        if self.num_heads % weights.process_group.size() != 0:
            raise ValueError(
                f"`num_heads` must be divisible by `num_shards` (got `num_heads`: {self.num_heads} "
                f"and `num_shards`: {weights.process_group.size()}"
            )
        self.num_heads = self.num_heads // weights.process_group.size()

        self.query_key_value = load_attention(
            config,
            prefix=prefix,
            weights=weights,
        )

        self.o_proj = load_row(
            config,
            prefix=f"{prefix}.out_proj",
            weights=weights,
            bias=False,
        )

        self.kv_head_mapping = torch.arange(
            0, self.num_heads, dtype=torch.int32, device=weights.device
        )

        self.rotary_emb = GPTJRotary.static(
            config=config,
            dim=self.rotary_dim,
            base=10000,
            device=weights.device,
        )

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
    ):
        query, key, value = self.query_key_value(hidden_states).split(
            self.head_size * self.num_heads, dim=1
        )
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_heads, self.head_size)
        value = value.view(-1, self.num_heads, self.head_size)

        # Compute rotary embeddings on rotary_ndims
        if self.rotary_dim is not None:
            self.rotary_emb(
                query[..., : self.rotary_dim], key[..., : self.rotary_dim], cos, sin
            )
        else:
            self.rotary_emb(query, key, cos, sin)

        reshape_and_cache(key, value, kv_cache[0], kv_cache[1], slots)

        # Prefill
        if cu_seqlen_prefill is not None:
            # flash attention
            attn_output = attention(
                query,
                kv_cache[0],
                kv_cache[1],
                seqlen,
                block_tables,
                self.softmax_scale,
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

        return self.o_proj(attn_output.view(-1, self.num_heads * self.head_size))


class GPTJMLP(nn.Module):
    def __init__(self, prefix: str, config, weights):
        super().__init__()
        act = config.activation_function
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

        self.fc_in = TensorParallelColumnLinear.load(
            config, prefix=f"{prefix}.fc_in", weights=weights, bias=True
        )

        self.fc_out = load_row(
            config,
            prefix=f"{prefix}.fc_out",
            weights=weights,
            bias=True,
        )

    def forward(self, hidden_states):
        hidden_states = self.fc_in(hidden_states)
        hidden_states = self.act(hidden_states)
        return self.fc_out(hidden_states)


class FlashGPTJLayer(nn.Module):
    def __init__(self, prefix: str, config, weights):
        super().__init__()
        self.self_attn = FlashGPTJAttention(
            prefix=f"{prefix}.attn", config=config, weights=weights
        )
        self.mlp = GPTJMLP(prefix=f"{prefix}.mlp", config=config, weights=weights)

        self.input_layernorm = FastLayerNorm.load(
            prefix=f"{prefix}.ln_1", weights=weights, eps=config.layer_norm_epsilon
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
    ):
        hidden_states, residual = self.input_layernorm(hidden_states, residual)
        # Self Attention
        attn_output = self.self_attn(
            hidden_states,
            cos,
            sin,
            cu_seqlen_prefill,
            kv_cache,
            block_tables,
            slots,
            seqlen,
            max_s,
        )

        feed_forward_hidden_states = self.mlp(hidden_states)

        return attn_output + feed_forward_hidden_states, residual


class FlashGPTJModel(torch.nn.Module):
    def __init__(self, prefix: str, config, weights):
        super().__init__()
        self.config = config

        self.wte = TensorParallelEmbedding(prefix=f"{prefix}.wte", weights=weights)
        self.layers = nn.ModuleList(
            [
                FlashGPTJLayer(
                    prefix=(
                        f"h.{layer_id}" if not prefix else f"{prefix}.h.{layer_id}"
                    ),
                    config=config,
                    weights=weights,
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )

        self.ln_f = FastLayerNorm.load(
            prefix="ln_f" if not prefix else f"{prefix}.ln_f",
            weights=weights,
            eps=config.layer_norm_epsilon,
        )

        self.gradient_checkpointing = False

        self.head_size = self.layers[0].self_attn.head_size
        self.num_heads = self.layers[0].self_attn.num_heads

    def forward(
        self,
        input_ids: Optional[torch.LongTensor],
        position_ids: torch.Tensor,
        cu_seqlen_prefill: Optional[torch.Tensor],
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        block_tables: torch.Tensor,
        slots: torch.Tensor,
        seqlen: Seqlen,
        max_s: int,
        prefill_cache_indices: Optional[torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = self.wte(input_ids)

        # Get rotary cos and sin for this forward
        # Avoid to index in each layer
        cos, sin = self.layers[0].self_attn.rotary_emb.get_cos_sin(
            position_ids, max_s, hidden_states.dtype
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
            )

        hidden_states, _ = self.ln_f(hidden_states, residual)

        return hidden_states


class FlashGPTJForCausalLM(torch.nn.Module):
    def __init__(self, prefix: str, config, weights):
        super().__init__()
        if not prefix:
            prefix = "transformer"
        else:
            prefix = f"{prefix}.transformer"
        self.model = FlashGPTJModel(prefix, config, weights)
        self.lm_head = SpeculativeHead.load(
            config,
            prefix="lm_head",
            weights=weights,
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
        prefill_cache_indices: Optional[torch.Tensor] = None,
        lm_head_indices: Optional[torch.Tensor] = None,
        adapter_data: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        hidden_states = self.model(
            input_ids,
            position_ids,
            cu_seqlen_prefill,
            kv_cache,
            block_tables,
            slots,
            seqlen,
            max_s,
            prefill_cache_indices=prefill_cache_indices,
        )
        if lm_head_indices is not None:
            hidden_states = hidden_states[lm_head_indices]
        logits, speculative_logits = self.lm_head(hidden_states)
        return logits, speculative_logits
