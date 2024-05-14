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

from text_generation_server.utils import paged_attention, flash_attn
from text_generation_server.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    SpeculativeHead,
    get_linear,
)
from text_generation_server.layers.rotary import PositionRotaryEmbedding
from text_generation_server.layers.layernorm import (
    FastRMSNorm,
)

# TODO: used for debugging; to avoid breaking during warmup
count = 0


class GemmaConfig(PretrainedConfig):
    def __init__(
        self,
        vocab_size=256128,
        hidden_size=3072,
        intermediate_size=24576,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=16,
        head_dim=256,
        hidden_act="gelu_pytorch_tanh",
        max_position_embeddings=8192,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=True,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        quantize: Optional[str] = None,
        use_medusa: Optional[str] = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.quantize = quantize
        self.use_medusa = use_medusa

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class GemmaFastRMSNorm(FastRMSNorm):
    @classmethod
    def load(cls, prefix, weights, eps=1e-6):
        weight = weights.get_tensor(f"{prefix}.weight")
        return cls(weight, eps)

    # perform the multiplication in full precision and downcast after
    def forward(self, hidden_states, residual=None):
        if residual is not None:
            hidden_states += residual
        residual = hidden_states
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = hidden_states * (self.weight.float() + 1.0)
        return hidden_states.to(self.weight.dtype), residual


def load_attention(config, prefix, weights):
    if config.num_attention_heads != config.num_key_value_heads:
        return _load_gqa(config, prefix, weights)
    else:
        return TensorParallelColumnLinear.load_multi(
            config,
            prefixes=[f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"],
            dim=0,
            weights=weights,
            bias=False,
        )


def _load_gqa(config, prefix: str, weights):
    assert config.num_attention_heads % weights.process_group.size() == 0

    weight = weights.get_multi_weights_col(
        prefixes=[f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"],
        quantize=config.quantize,
        dim=0,
    )

    if config.quantize not in ["gptq", "awq"]:
        weight = weight.to(dtype=weights.dtype).to(device=weights.device)

        head_size = config.head_dim
        num_heads = config.num_attention_heads // weights.process_group.size()
        num_key_value_heads = config.num_key_value_heads // weights.process_group.size()
        assert list(weight.shape) == [
            (num_heads + 2 * num_key_value_heads) * head_size,
            config.hidden_size,
        ], f"{list(weight.shape)} != {[(num_heads + 2 * config.num_key_value_heads) * head_size, config.hidden_size]}"

    return TensorParallelColumnLinear(
        get_linear(weight, bias=None, quantize=config.quantize)
    )


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class GemmaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.register_buffer("inv_freq", None, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if self.inv_freq is None:
            self.inv_freq = 1.0 / (
                self.base
                ** (
                    torch.arange(
                        0, self.dim, 2, dtype=torch.int64, device=x.device
                    ).float()
                    / self.dim
                )
            )

        position_ids = position_ids.unsqueeze(0)

        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )

        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = (
            device_type
            if isinstance(device_type, str) and device_type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):

            # import ipdb
            # ipdb.set_trace()
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class FlashGemmaAttention(torch.nn.Module):
    def __init__(
        self,
        prefix: str,
        config,
        weights,
    ):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_size = config.head_dim

        # self._rotary_emb = PositionRotaryEmbedding.static(
        #     config=config,
        #     dim=self.head_size,
        #     base=config.rope_theta,
        #     device=weights.device,
        # )

        self.rotary_emb = GemmaRotaryEmbedding(
            dim=self.head_size,
            max_position_embeddings=config.max_position_embeddings,
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

        # TODO: prefer this implementation after debugging
        # self.query_key_value = load_attention(config, prefix, weights)

        self.k_proj = TensorParallelColumnLinear.load(
            config,
            prefix=f"{prefix}.k_proj",
            weights=weights,
            bias=False,
        )
        self.v_proj = TensorParallelColumnLinear.load(
            config, prefix=f"{prefix}.v_proj", weights=weights, bias=False
        )
        self.q_proj = TensorParallelColumnLinear.load(
            config, prefix=f"{prefix}.q_proj", weights=weights, bias=False
        )

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
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

    def forward(
        self,
        hidden_states,
        cos,
        sin,
        cu_seqlen_prefill,
        kv_cache,
        block_tables,
        slots,
        input_lengths,
        max_s,
    ):
        global count

        # TODO: replace with better implementation after debugging
        tgt_len, src_len = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = (
            query_states.unsqueeze(0).view(1, tgt_len, 8, 256).transpose(1, 2)
        )
        key_states = key_states.unsqueeze(0).view(1, tgt_len, 1, 256).transpose(1, 2)
        value_states = (
            value_states.unsqueeze(0).view(1, tgt_len, 1, 256).transpose(1, 2)
        )

        # reshape for flash/paged attention
        kv = torch.cat([key_states, value_states], dim=1).transpose(0, 2)

        # cos2, sin2 = self.rotary_emb(value_states, position_ids, seq_len=None)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, None
        )

        # replace the kv with the rotated kv
        kv[:, 0] = key_states.squeeze(0).transpose(0, 1)
        query = query_states.squeeze(0).transpose(0, 1)

        # TODO: remove after debugging
        # ipdb> ref_query_states = torch.load("/home/ubuntu/Projects/new-model-addition-palma/ref_query_states.pt").to("cuda:0")
        # ipdb> torch.allclose(query,ref_query_states.squeeze(0).transpose(0,1))

        # ipdb> ref_key_states = torch.load("/home/ubuntu/Projects/new-model-addition-palma/ref_key_states.pt").to("cuda:0")
        # ipdb> torch.allclose(kv[:, 0],ref_key_states.squeeze(0).transpose(0,1))

        # ipdb> ref_value_states = torch.load("/home/ubuntu/Projects/new-model-addition-palma/ref_value_states.pt").to("cuda:0")
        # ipdb> torch.allclose(kv[:, 1],ref_value_states.squeeze(0).transpose(0,1))

        if count > 0:
            import ipdb

            # ipdb.set_trace()

        paged_attention.reshape_and_cache(
            kv[:, 0],
            kv[:, 1],
            kv_cache[0],
            kv_cache[1],
            slots,
        )

        # output tensor
        attn_output = torch.empty_like(query)

        # Prefill
        if cu_seqlen_prefill is not None:
            # flash attention
            flash_attn.attention(
                query,
                torch.select(kv, dim=1, index=0),
                torch.select(kv, dim=1, index=1),
                attn_output,
                cu_seqlen_prefill,
                max_s,
                self.softmax_scale,
            )
        # Decode
        else:
            paged_attention.attention(
                attn_output,
                query,
                kv_cache[0],
                kv_cache[1],
                self.kv_head_mapping,
                self.softmax_scale,
                block_tables,
                input_lengths,
                max_s,
            )

        return self.o_proj(attn_output.view(-1, self.num_heads * self.head_size))


class GemmaMLP(nn.Module):
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
            bias=False,
        )
        self.down_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.down_proj",
            weights=weights,
            bias=False,
        )
        self.intermediate_size = (
            config.intermediate_size // weights.process_group.size()
        )

    def forward(self, hidden_states):
        gate_up_states = self.gate_up_proj(hidden_states)
        gate_up_states = gate_up_states.view(-1, 2, self.intermediate_size)
        return self.down_proj(self.act(gate_up_states[:, 0]) * gate_up_states[:, 1])


class FlashGemmaLayer(nn.Module):
    def __init__(self, prefix, layer_id, config, weights):
        super().__init__()
        prefix = f"{prefix or ''}model.layers.{layer_id}"
        self.self_attn = FlashGemmaAttention(
            prefix=f"{prefix}.self_attn", config=config, weights=weights
        )
        self.mlp = GemmaMLP(prefix=f"{prefix}.mlp", config=config, weights=weights)

        self.input_layernorm = GemmaFastRMSNorm.load(
            prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = GemmaFastRMSNorm.load(
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
        input_lengths,
        max_s,
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
            input_lengths,
            max_s,
        )

        # faster post attention rms norm
        normed_attn_res_output, attn_res = self.post_attention_layernorm(
            attn_output, res
        )

        mlp_output = self.mlp(normed_attn_res_output)

        return mlp_output, attn_res


class FlashGemmaModel(torch.nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()

        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        embed_norm = config.hidden_size**0.5
        pvalue = f"{prefix + '.' if prefix else ''}model.embed_tokens"
        self.embed_tokens = TensorParallelEmbedding(
            prefix=pvalue,
            weights=weights,
        )
        self.embed_tokens.weight = torch.nn.Parameter(
            self.embed_tokens.weight[: config.vocab_size, : config.hidden_size]
        )

        # TODO: avoid making a copy of the embedding matrix. added for debugging
        self.unscaled_embed_tokens = torch.nn.Parameter(
            self.embed_tokens.weight.clone()
        )

        self.embed_tokens.weight *= embed_norm

        self.layers = nn.ModuleList(
            [
                FlashGemmaLayer(
                    f"{prefix + '.' if prefix else ''}",
                    layer_id,
                    config,
                    weights,
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = GemmaFastRMSNorm.load(
            prefix=f"{prefix + '.' if prefix else ''}model.norm",
            weights=weights,
            eps=config.rms_norm_eps,
        )

        self.gradient_checkpointing = False

        self.head_size = self.layers[0].self_attn.head_size
        self.num_heads = self.layers[0].self_attn.num_heads
        self.num_key_value_heads = self.layers[0].self_attn.num_key_value_heads

    def forward(
        self,
        # input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlen_prefill: Optional[torch.Tensor],
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        block_tables: torch.Tensor,
        slots: torch.Tensor,
        input_lengths: torch.Tensor,
        max_s: int,
    ) -> torch.Tensor:
        global count
        hidden_states = inputs_embeds

        residual = None
        for i, layer in enumerate(self.layers):
            # TODO: prefer a single rotary embedding implementation after debugging
            cos, sin = self.layers[i].self_attn.rotary_emb(
                hidden_states,
                position_ids,
            )

            hidden_states, residual = layer(
                hidden_states,
                residual,
                cos,
                sin,
                cu_seqlen_prefill,
                kv_cache[i],
                block_tables,
                slots,
                input_lengths,
                max_s,
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        count += 1  # for debugging; to avoid breaking during warmup
        return hidden_states


class FlashGemmaForCausalLM(torch.nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        self.config = config
        self.model = FlashGemmaModel(prefix, config, weights)
        prefix = f"{prefix + '.' if prefix else ''}model.embed_tokens"
        prefix = prefix if config.tie_word_embeddings else "lm_head"
        self.lm_head = SpeculativeHead.load(
            config,
            prefix=prefix,
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
        input_lengths: torch.Tensor,
        max_s: int,
        lm_head_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        inputs_embeds = self.model.embed_tokens(input_ids)
        hidden_states = self.model(
            inputs_embeds,
            position_ids,
            cu_seqlen_prefill,
            kv_cache,
            block_tables,
            slots,
            input_lengths,
            max_s,
        )
        if lm_head_indices is not None:
            hidden_states = hidden_states[lm_head_indices]
        logits, speculative_logits = self.lm_head(hidden_states)
        return logits, speculative_logits
