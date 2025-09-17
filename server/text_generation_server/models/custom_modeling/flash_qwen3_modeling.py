import torch
import torch.distributed

from torch import nn
from transformers.activations import ACT2FN
from typing import Optional, List, Tuple

from text_generation_server.layers.attention import (
    paged_attention,
    attention,
    Seqlen,
)
from text_generation_server.layers import (
    TensorParallelMultiAdapterLinear,
    TensorParallelAdapterRowLinear,
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    SpeculativeHead,
)
from text_generation_server.layers.attention.kv_cache import get_kv_scales
from text_generation_server.layers.rotary import PositionRotaryEmbedding
from text_generation_server.layers.layernorm import (
    FastRMSNorm,
)


def load_attention(config, prefix, weights, layer_id):
    prefixes = [f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"]
    head_size = getattr(
        config, "head_dim", config.hidden_size // config.num_attention_heads
    )
    sizes = [
        head_size * config.num_attention_heads,
        head_size * config.num_key_value_heads,
        head_size * config.num_key_value_heads,
    ]
    if config.num_attention_heads != config.num_key_value_heads:
        base_layer = _load_gqa(config, prefix, weights)
    else:
        base_layer = TensorParallelColumnLinear.load_multi(
            config,
            prefixes=prefixes,
            dim=0,
            weights=weights,
            bias=getattr(config, 'attention_bias', False),  # Use config value like vLLM
        )
    return TensorParallelMultiAdapterLinear.load(
        base_layer=base_layer,
        layer_id=layer_id,
        layer_names=prefixes,
        sizes=sizes,
        process_group=weights.process_group,
    )


def _load_gqa(config, prefix: str, weights):
    assert config.hidden_size % config.num_attention_heads == 0
    assert config.num_attention_heads % weights.process_group.size() == 0

    return TensorParallelColumnLinear.load_multi(
        config,
        prefixes=[f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"],
        dim=0,
        weights=weights,
        bias=getattr(config, 'attention_bias', False),  # Use config value like vLLM
    )


class Qwen3Attention(torch.nn.Module):
    def __init__(
        self,
        index: int,
        prefix: str,
        config,
        weights,
    ):
        super().__init__()
        self.layer_idx = index
        self.config = config
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.num_heads = config.num_attention_heads
        self.attention_dropout = config.attention_dropout
        self.softmax_scale = self.head_dim**-0.5

        self.window_size = (
            config.sliding_window if config.sliding_window is not None else -1
        )
        # Handle sliding window configuration similar to Intel Gaudi version
        self.sliding_window = config.sliding_window
        if hasattr(config, 'use_sliding_window') and hasattr(config, 'max_window_layers'):
            if not (
                config.use_sliding_window
                and getattr(config, "sliding_window", None) is not None
                and self.layer_idx >= config.max_window_layers
            ):
                self.sliding_window = None

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

        self.query_key_value = load_attention(config, prefix, weights, index)

        self.kv_scales = get_kv_scales(weights, f"{prefix}")

        # Q and K normalization layers
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

        o_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.o_proj",
            weights=weights,
            bias=False,
        )
        self.o_proj = TensorParallelAdapterRowLinear.load(
            o_proj,
            index,
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
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        qkv = self.query_key_value(hidden_states, adapter_data)
        query_states, key_states, value_states = qkv.split(
            [
                self.head_dim * self.num_heads,
                self.head_dim * self.num_key_value_heads,
                self.head_dim * self.num_key_value_heads,
            ],
            dim=1,
        )

        # First reshape to head dimensions
        query_states = query_states.reshape(hidden_shape)
        key_states = key_states.reshape(hidden_shape)
        value_states = value_states.reshape(hidden_shape)

        # Apply Q and K normalization on head_dim - following vLLM/SGLang correct pattern
        # This matches the reference implementations and is the correct approach
        q_by_head = query_states.reshape(-1, self.head_dim)
        q_by_head, _ = self.q_norm(q_by_head)
        query_states = q_by_head.view(query_states.shape)
        
        k_by_head = key_states.reshape(-1, self.head_dim)
        k_by_head, _ = self.k_norm(k_by_head)
        key_states = k_by_head.view(key_states.shape)

        self.rotary_emb(query_states, key_states, cos, sin)

        if prefill_cache_indices is not None:
            key_to_cache = key_states[prefill_cache_indices]
            value_to_cache = value_states[prefill_cache_indices]
        else:
            key_to_cache = key_states
            value_to_cache = value_states

        kv_cache.store(
            key=key_to_cache,
            value=value_to_cache,
            slots=slots,
            kv_scales=self.kv_scales,
        )

        # Prefill
        if cu_seqlen_prefill is not None:
            # flash attention
            attn_output = attention(
                query=query_states,
                key=key_to_cache,
                value=value_to_cache,
                kv_cache=kv_cache,
                kv_scales=self.kv_scales,
                seqlen=seqlen,
                block_tables=block_tables,
                softmax_scale=self.softmax_scale,
                window_size_left=self.window_size,
            )
        # Decode
        else:
            attn_output = paged_attention(
                query_states,
                kv_cache,
                self.kv_head_mapping,
                self.softmax_scale,
                block_tables,
                seqlen,
                max_s,
                kv_scales=self.kv_scales,
                window_size_left=self.window_size,
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output, adapter_data)


# Import Qwen2MLP from the existing module to reuse the implementation
from .flash_qwen2_modeling import Qwen2MLP as Qwen3MLP


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, prefix, layer_id, config, weights):
        super().__init__()
        prefix = f"{prefix}.layers.{layer_id}"
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3Attention(
            index=layer_id, prefix=f"{prefix}.self_attn", config=config, weights=weights
        )
        self.mlp = Qwen3MLP(
            prefix=f"{prefix}.mlp", config=config, weights=weights, index=layer_id
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
        residual = hidden_states
        hidden_states, _ = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
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
        )

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states, _ = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, adapter_data)
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen3Model(torch.nn.Module):
    def __init__(self, prefix: str, config, weights):
        super().__init__()

        prefix = f"{prefix}.model" if prefix else "model"
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(
                    prefix,
                    layer_id,
                    config,
                    weights,
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = FastRMSNorm.load(
            prefix=f"{prefix}.norm", weights=weights, eps=config.rms_norm_eps
        )

        self.gradient_checkpointing = False

        self.head_size = self.layers[0].self_attn.head_dim
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
        adapter_data,
    ) -> torch.Tensor:
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        cos, sin = self.layers[0].self_attn.rotary_emb.get_cos_sin(
            position_ids,
            true_max_s,
            hidden_states.dtype,
        )

        residual = None
        for i, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
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

        hidden_states, _ = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        return hidden_states


class Qwen3ForCausalLM(torch.nn.Module):
    def __init__(self, prefix: str, config, weights):
        super().__init__()

        self.model = Qwen3Model(prefix, config, weights)
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

        self.window_size = config.sliding_window
        self.window_size_tensor = (
            torch.tensor(config.sliding_window, device=weights.device)
            if self.window_size is not None
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
        prefill_cache_indices: Optional[torch.Tensor] = None,
        lm_head_indices: Optional[torch.Tensor] = None,
        adapter_data: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        true_max_s = max_s
        if prefill_cache_indices is not None:
            # Slots also need to be sliced as it has the same size as the whole kv tensor
            slots = slots[prefill_cache_indices]
        elif self.window_size is not None:
            # Clamp in decode mode as paged attention requires clamped values whereas the flash attention
            # kernel requires the true values
            seqlen = seqlen.clamp(max=self.window_size_tensor)

        inputs_embeds = self.embed_tokens(input_ids)
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
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

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        if lm_head_indices is not None:
            hidden_states = hidden_states[lm_head_indices]
        logits = self.lm_head(hidden_states)

        return logits