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

from typing import Optional, Tuple, List

import torch
from torch import nn
import habana_frameworks.torch as htorch
from text_generation_server.layers.attention import (
    paged_attention,
    attention,
    Seqlen,
    HPUPagedAttentionMetadata,
)
from text_generation_server.layers.attention.kv_cache import get_kv_scales
from text_generation_server.layers import (
    TensorParallelEmbedding,
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    SpeculativeHead,
)


from text_generation_server.layers.layernorm import (
    FastRMSNorm,
)
from .flash_qwen2_modeling import Qwen2MLP
from text_generation_server.layers.rotary import PositionRotaryEmbedding


class Qwen3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, prefix, weights, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.num_heads = config.num_attention_heads
        self.attention_dropout = config.attention_dropout
        self.softmax_scale = self.head_dim**-0.5
        self.rotary_emb = PositionRotaryEmbedding.static(
            config=config,
            dim=self.head_dim,
            base=config.rope_theta,
            device=weights.device,
        )

        if self.num_heads % weights.process_group.size() != 0:
            raise ValueError(
                f"`num_heads` must be divisible by `num_shards` (got `num_heads`: {self.num_heads} "
                f"and `num_shards`: {weights.process_group.size()}"
            )
        self.num_heads = self.num_heads // weights.process_group.size()
        self.num_key_value_heads = (
            config.num_key_value_heads // weights.process_group.size()
        )
        self.query_key_value = TensorParallelColumnLinear.load_multi(
            config,
            prefixes=[f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"],
            dim=0,
            weights=weights,
            bias=False,
        )

        self.kv_scales = get_kv_scales(weights, f"{prefix}")

        self.o_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.o_proj",
            weights=weights,
            bias=False,
        )


        self.num_groups = self.num_heads // self.num_key_value_heads
        self.kv_head_mapping = torch.arange(
            0, self.num_key_value_heads, dtype=torch.int32, device=weights.device
        ).repeat_interleave(self.num_groups)

        self.max_past = (
            config.sliding_window if config.sliding_window is not None else -1
        )

        self.q_norm = FastRMSNorm.load(
            prefix=f"{prefix}.q_norm",
            weights=weights,
            eps=config.rms_norm_eps,
        )
        self.k_norm = FastRMSNorm.load(
            prefix=f"{prefix}.k_norm",
            weights=weights,
            eps=config.rms_norm_eps,
        )
        self.sliding_window = config.sliding_window
        if not (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            self.sliding_window = None

    def forward(
        self,
        hidden_states,
        cos,
        sin,
        cu_seqlen_prefill,
        kv_cache,
        slots,
        seqlen,
        hpu_attention_meta,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        qkv = self.query_key_value(hidden_states)
        query_states, key_states, value_states = qkv.split(
            [
                self.head_dim * self.num_heads,
                self.head_dim * self.num_key_value_heads,
                self.head_dim * self.num_key_value_heads,
            ],
            dim=1,
        )

        query_states, _ = self.q_norm(query_states.view(hidden_shape))
        key_states, _ = self.k_norm(key_states.view(hidden_shape))
        value_states = value_states.view(hidden_shape)
        self.rotary_emb(query_states, key_states, cos, sin)

        kv_cache.store(
            key=key_states,
            value=value_states,
            slots=slots,
            kv_scales=self.kv_scales,
        )

        # Prefill
        if cu_seqlen_prefill is not None:
            # sdpa
            attn_output = attention(
                query=query_states,
                key=key_states,
                value=value_states,
                kv_cache=kv_cache,
                kv_scales=self.kv_scales,
                seqlen=seqlen,
                softmax_scale=self.softmax_scale,
                window_size_left=self.max_past,
                num_key_value_groups=self.num_key_value_groups,
            )
        # Decode
        else:
            attn_output = paged_attention(
                query_states,
                kv_cache,
                self.kv_head_mapping,
                self.softmax_scale,
                seqlen,
                kv_scales=self.kv_scales,
                hpu_attention_meta=hpu_attention_meta,
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output)


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config, prefix, weights, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3Attention(
            config=config,
            prefix=f"{prefix}.self_attn",
            weights=weights,
            layer_idx=layer_idx,
        )
        self.mlp = Qwen2MLP(config=config, prefix=f"{prefix}.mlp", weights=weights)
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
        slots,
        seqlen,
        hpu_attention_meta,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states, _ = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states,
            cos,
            sin,
            cu_seqlen_prefill,
            kv_cache,
            slots,
            seqlen,
            hpu_attention_meta,
        )

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states, _ = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen3Model(nn.Module):
    def __init__(self, config, prefix: str, weights):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(
                    config=config,
                    prefix=f"{prefix}.layers.{layer_idx}",
                    weights=weights,
                    layer_idx=layer_idx,
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = FastRMSNorm.load(
            prefix=f"{prefix}.norm", weights=weights, eps=config.rms_norm_eps
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlen_prefill: Optional[torch.Tensor],
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        slots: torch.Tensor,
        seqlen: Seqlen,
        hpu_attention_meta: Optional[HPUPagedAttentionMetadata],
    ) -> torch.Tensor:

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        cos, sin = self.layers[0].self_attn.rotary_emb.get_cos_sin(
            position_ids,
        )

        residual = None

        lazy_mode = htorch.utils.internal.is_lazy()
        if lazy_mode:
            htorch.core.mark_step()

        for i, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                residual,
                cos,
                sin,
                cu_seqlen_prefill,
                kv_cache[i],
                slots,
                seqlen,
                hpu_attention_meta,
            )
            if lazy_mode:
                htorch.core.mark_step()

        hidden_states, _ = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        return hidden_states


class Qwen3ForCausalLM(nn.Module):

    def __init__(self, prefix: str, config, weights):
        super().__init__()
        self.model = Qwen3Model(config=config, prefix="model", weights=weights)
        self.vocab_size = config.vocab_size
        if config.tie_word_embeddings:
            suffix = "model.embed_tokens"
        else:
            suffix = "lm_head"

        self.lm_head = SpeculativeHead.load(
            config,
            prefix=f"{prefix}.{suffix}" if prefix else suffix,
            weights=weights,
        )

        self.embed_tokens = TensorParallelEmbedding(
            prefix=f"{prefix}.embed_tokens" if prefix else "model.embed_tokens",
            weights=weights,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlen_prefill: Optional[torch.Tensor],
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        slots: torch.Tensor,
        seqlen: Seqlen,
        hpu_attention_meta: Optional[HPUPagedAttentionMetadata],
        lm_head_indices: Optional[torch.Tensor] = None,
        adapter_data: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        inputs_embeds = self.embed_tokens(input_ids)
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        hidden_states = self.model(
            inputs_embeds,
            position_ids,
            cu_seqlen_prefill,
            kv_cache,
            slots,
            seqlen,
            hpu_attention_meta,
        )

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss

        if lm_head_indices is not None:
            hidden_states = hidden_states[lm_head_indices]
        logits = self.lm_head(hidden_states)

        return logits
