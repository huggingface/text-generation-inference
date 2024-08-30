import torch
import torch.distributed

from torch import nn
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
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
from text_generation_server.layers.attention import PREFILL_IN_KV_CACHE
from text_generation_server.layers.layernorm import (
    FastLayerNorm,
)
from text_generation_server.layers.rotary import (
    PositionRotaryEmbedding,
)

from text_generation_server.layers import (
    FastLinear,
)

from text_generation_server.utils.import_utils import SYSTEM

if SYSTEM != "ipex":
    from vllm.model_executor.layers.fused_moe import fused_moe


class PhiConfig(PretrainedConfig):
    def __init__(
        self,
        vocab_size=51200,
        hidden_size=2560,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="gelu_fast",  # llama uses silu
        layer_norm_eps=1e-05,  # rms in llama,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        resid_pdrop=0.1,  # llama doesn't have this
        partial_rotary_factor=0.5,  # important difference between llama and phi
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.rope_theta = rope_theta
        self.resid_pdrop = resid_pdrop
        self.partial_rotary_factor = partial_rotary_factor

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


# this is the same as llama except for Phi uses bias=True
def load_attention(config, prefix, weights):
    if config._name_or_path == "microsoft/Phi-3.5-MoE-instruct":
        return TensorParallelColumnLinear.load_multi(
            config,
            prefixes=[f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"],
            dim=0,
            weights=weights,
            bias=True,
        )
    if config.num_attention_heads != config.num_key_value_heads \
            and False:
        return _load_gqa(config, prefix, weights)
    else:
        return TensorParallelColumnLinear.load_multi(
            config,
            prefixes=[f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"],
            dim=0,
            weights=weights,
            bias=True,
        )


def _load_gqa(config, prefix: str, weights):
    assert config.hidden_size % config.num_attention_heads == 0
    assert config.num_attention_heads % weights.process_group.size() == 0

    weight = weights.get_multi_weights_col(
        prefixes=[f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"],
        dim=0,
    )

    if config.quantize and (config.quantize not in ["gptq", "awq", "marlin"]):
        weight = weight.to(dtype=weights.dtype).to(device=weights.device)

        head_size = config.hidden_size // config.num_attention_heads
        num_heads = config.num_attention_heads // weights.process_group.size()
        num_key_value_heads = config.num_key_value_heads // weights.process_group.size()
        assert list(weight.shape) == [
            (num_heads + 2 * num_key_value_heads) * head_size,
            config.hidden_size,
        ], f"{list(weight.shape)} != {[(num_heads + 2 * config.num_key_value_heads) * head_size, config.hidden_size]}"

    # this is the same as llama except for Phi uses bias=True
    return TensorParallelColumnLinear(get_linear(weight, bias=True))


def _load_experts(config, prefix: str, mat, weights):
    if config.quantize is not None:
        raise NotImplementedError("Mixtral does not support weight quantization yet.")

    assert mat in ["w1", "w2", "w3"]

    world_size = weights.process_group.size()
    rank = weights.process_group.rank()

    assert (
        config.intermediate_size % world_size == 0
    ), f"The chosen size {config.intermediate_size} is not compatible with sharding on {world_size} shards"

    block_size = config.intermediate_size // world_size
    start = rank * block_size
    stop = (rank + 1) * block_size

    tensor = torch.empty(
        (config.num_local_experts * block_size, config.hidden_size),
        dtype=weights.dtype,
        device=weights.device,
    )

    for i in range(config.num_local_experts):
        slice_ = weights._get_slice(f"{prefix}.{i}.{mat}.weight")

        if mat == "w2":
            expert_slice = slice_[:, start:stop].t().contiguous()
        else:
            expert_slice = slice_[start:stop]
        tensor[i * block_size : (i + 1) * block_size] = expert_slice.to(
            dtype=weights.dtype
        ).to(device=weights.device)
    return tensor


class BlockSparseMoE(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size // weights.process_group.size()
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok

        act = config.hidden_act
        if "gelu" in act:
            self.act = lambda x: torch.nn.functional.gelu(
                x,
                approximate=(
                    "tanh" if act in ["gelu_fast", "gelu_pytorch_tanh"] else "none"
                ),
            )
        elif "silu" in act:
            self.act = torch.nn.functional.silu
        else:
            self.act = ACT2FN[act]

        # gating
        self.gate = FastLinear.load(config, f"{prefix}.gate", weights, bias=False)

        # merged expert weights, all of size  (n_experts * ffn_dim, hidden_dim)
        w1 = _load_experts(config, f"{prefix}.experts", "w1", weights).view(
            self.num_experts, self.ffn_dim, self.hidden_dim
        )
        w3 = _load_experts(config, f"{prefix}.experts", "w3", weights).view(
            self.num_experts, self.ffn_dim, self.hidden_dim
        )
        self.w13 = torch.cat([w1, w3], dim=1)
        self.w2 = (
            _load_experts(config, f"{prefix}.experts", "w2", weights)
            .view(self.num_experts, self.ffn_dim, self.hidden_dim)
            .transpose(1, 2)
            .contiguous()
        )

        self.process_group = weights.process_group

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # router_logits: (num_tokens, n_experts)
        router_logits = self.gate(x)
        out = fused_moe(
            x,
            self.w13,
            self.w2,
            router_logits,
            self.top_k,
            renormalize=True,
            inplace=True,
        )

        # Reduce sum
        if self.process_group.size() > 1:
            torch.distributed.all_reduce(out, group=self.process_group)

        return out.view(*x.shape)


class FlashPhiAttention(torch.nn.Module):
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
        self.use_moe = config._name_or_path == "microsoft/Phi-3.5-MoE-instruct"
        self.head_dim = self.head_size
        if hasattr(config, "partial_rotary_factor"):
            self.rotary_dim = int(config.partial_rotary_factor * self.head_size)
        else:
            self.rotary_dim = self.head_size

        self.rotary_emb = PositionRotaryEmbedding.static(
            config=config,
            dim=self.rotary_dim,
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

        self.query_key_value = load_attention(config, prefix, weights)

        # in llama the dense layer is called "o_proj" and has bias=False
        proj_layer_name = f"{prefix}.o_proj" if self.use_moe else f"{prefix}.dense"
        self.dense = TensorParallelRowLinear.load(
            config,
            prefix=proj_layer_name,
            weights=weights,
            bias=True,
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
    ):
        # Compute query, key, value and split
        qkv = self.query_key_value(hidden_states)
        query, kv = qkv.split(
            [
                self.head_size * self.num_heads,
                2 * self.head_size * self.num_key_value_heads,
            ],
            dim=1,
        )

        # Reshape query and key for rotary embeddings
        query = query.view(-1, self.num_heads, self.head_size)
        kv = kv.view(-1, 2, self.num_key_value_heads, self.head_size)

        # NOTE: this is the main difference between Llama and Phi
        # in llama the rotary embeddings are applied to the whole query and key.
        # Phi uses PARTIAL rotary embeddings, which are applied to the first 32 dimensions
        #
        # Apply partial positional embeddings in place
        if self.use_moe:
            # rotate half rotary_dim
            # half_size = torch.select(kv, dim=1, index=0).size(1) // 2
            self.rotary_emb(query, torch.select(kv, dim=1, index=0), cos, sin)
        else:
            self.rotary_emb(
                query[:, :, : self.rotary_dim], kv[:, 0, :, : self.rotary_dim], cos, sin
            )

        # Reshape key and value and cache
        reshape_and_cache(kv[:, 0], kv[:, 1], kv_cache[0], kv_cache[1], slots)

        # Prefill
        if cu_seqlen_prefill is not None:
            attn_output = attention(
                query,
                kv_cache[0] if PREFILL_IN_KV_CACHE else kv[:, 0],
                kv_cache[1] if PREFILL_IN_KV_CACHE else kv[:, 1],
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

        return self.dense(attn_output.view(-1, self.num_heads * self.head_size))


class PhiMLP(nn.Module):
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

        # llama weights are up_proj and down_proj and bias=False
        self.up_proj = TensorParallelColumnLinear.load(
            config,
            prefix=f"{prefix}.fc1",
            weights=weights,
            bias=True,
        )
        self.down_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.fc2",
            weights=weights,
            bias=True,
        )

    def forward(self, hidden_states):
        # NOTE: Llama requires the gate up states to an intermediate size
        # Phi does not and we can avoid the `view` operation
        return self.down_proj(self.act(self.up_proj(hidden_states)))


class FlashPhiLayer(nn.Module):
    def __init__(self, prefix: str, layer_id, config, weights):
        super().__init__()
        prefix = f"{prefix}.layers.{layer_id}"
        self.self_attn = FlashPhiAttention(
            prefix=f"{prefix}.self_attn", config=config, weights=weights
        )

        self.use_moe = config._name_or_path == "microsoft/Phi-3.5-MoE-instruct"

        if self.use_moe:
            self.moe = BlockSparseMoE(f"{prefix}.block_sparse_moe", config, weights)
        else:
            self.mlp = PhiMLP(prefix=f"{prefix}.mlp", config=config, weights=weights)

        self.input_layernorm = FastLayerNorm.load(
            prefix=f"{prefix}.input_layernorm",
            weights=weights,
            # eps=config.layer_norm_eps,
            eps=1e-5,
        )

        if self.use_moe:
            self.post_attn_layernorm = FastLayerNorm.load(
                prefix=f"{prefix}.post_attention_layernorm",
                weights=weights,
                eps=1e-5,
            )

        # self.resid_dropout = torch.nn.Dropout(config.resid_pdrop)
        self.resid_dropout = torch.nn.Dropout(config.attention_dropout)

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
        hidden_states, res = self.input_layernorm(hidden_states, residual)
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


        if self.use_moe:
            # hidden_states = attn_output
            # if residual is not None:
            #     hidden_states = hidden_states + residual
            # residual = hidden_states
            # hidden_states, router_states = self.post_attn_layernorm(
            #     hidden_states
            # )
            # hidden_states = self.moe(hidden_states)
            # # hidden_states = hidden_states + residual
            # res = residual

            hidden_states = self.resid_dropout(attn_output)
            _hidden_states = self.resid_dropout(self.moe(hidden_states))
            hidden_states = hidden_states.add(_hidden_states)
            

        else:
            hidden_states = self.resid_dropout(attn_output).add(
                self.resid_dropout(self.mlp(hidden_states))
            )

        return hidden_states, res


class FlashPhiModel(torch.nn.Module):
    def __init__(self, prefix: str, config, weights):
        super().__init__()

        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.embed_tokens = TensorParallelEmbedding(
            prefix=f"{prefix}.embed_tokens", weights=weights
        )
        self.layers = nn.ModuleList(
            [
                FlashPhiLayer(
                    prefix,
                    layer_id,
                    config,
                    weights,
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.gradient_checkpointing = False

        self.head_size = self.layers[0].self_attn.head_size
        self.num_heads = self.layers[0].self_attn.num_heads
        self.num_key_value_heads = self.layers[0].self_attn.num_key_value_heads

        self.use_moe = config._name_or_path == "microsoft/Phi-3.5-MoE-instruct"

        if self.use_moe:
            self.norm = FastLayerNorm.load(
                prefix="model.norm",
                weights=weights,
                eps=1e-5,
            )
        else:
            self.norm = FastLayerNorm.load(
                prefix="model.final_layernorm",
                weights=weights,
                eps=config.layer_norm_eps,
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
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)

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

        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states


class FlashPhiForCausalLM(torch.nn.Module):
    def __init__(self, prefix: str, config, weights):
        super().__init__()

        if not prefix:
            prefix = "model"
        else:
            prefix = f"{prefix}.model"

        self.model = FlashPhiModel(prefix, config, weights)
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
        prefill_cache_indices: Optional[torch.Tensor],
        lm_head_indices: Optional[torch.Tensor] = None,
        adapter_data: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids,
            position_ids,
            cu_seqlen_prefill,
            kv_cache,
            block_tables,
            slots,
            seqlen,
            max_s,
        )
        if lm_head_indices is not None:
            hidden_states = hidden_states[lm_head_indices]

        return self.lm_head(hidden_states)
