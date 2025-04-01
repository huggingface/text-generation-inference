# coding=utf-8
# Copyright 2023, 2024 DeepSeek-AI and The HuggingFace Inc. team. All rights reserved.
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

from typing import List, Optional, Tuple, Type

import torch
import torch.distributed
from torch import nn
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig

from text_generation_server.layers import (
    FastLinear,
    SpeculativeHead,
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    TensorParallelRowLinear,
    TensorParallelEmbedding,
    TensorParallelMultiAdapterLinear,
    TensorParallelAdapterRowLinear,
    get_linear,
)
from text_generation_server.layers.attention import (
    Seqlen,
    attention,
    paged_attention,
)
from text_generation_server.layers.attention.kv_cache import KVCache, get_kv_scales
from text_generation_server.layers.layernorm import FastRMSNorm
from text_generation_server.layers.moe import DenseMoELayer, MoELayer, SparseMoELayer
from text_generation_server.layers.rotary import PositionRotaryEmbedding, get_mscale
from text_generation_server.utils.import_utils import SYSTEM
from text_generation_server.utils.weights import Weights

if SYSTEM == "rocm":
    try:
        import vllm._custom_ops as ops
    except Exception as e:
        raise ImportError(f"Could not load `vllm._custom_ops`. Full error: {e}")
    

# class FlashLlama4VisionModel(torch.nn.Module):
#     def __init__(self, prefix: str, config, weights: Weights):
#         super().__init__()
#         self.config = config
#         self.prefix = prefix
#         self.weights = weights
        
#         self.image_size = config.image_size
#         self.patch_size = config.patch_size
#         # self.max_num_tiles = config.max_num_tiles
#         self.hidden_size = config.hidden_size
#         self.num_channels = config.num_channels
#         # self.intermediate_layers_indices = config.intermediate_layers_indices

#         self.num_patches = (self.image_size // self.patch_size) ** 2 + 1
#         self.scale = config.hidden_size**-0.5
        
#         self.patch_embedding = UnfoldConvolution(
#             in_channels=config.num_channels,
#             out_channels=self.hidden_size,
#             kernel_size=self.patch_size,
#             stride=self.patch_size,
#             bias=False,
#         )
        
#         self.class_embedding = nn.Parameter(self.scale * torch.randn(self.hidden_size))
#         self.positional_embedding_vlm = nn.Parameter(
#             self.scale * torch.randn(self.num_patches, self.hidden_size)
#         )
        
#         idx = self.image_size // self.patch_size
#         img_idx = torch.arange((self.image_size // self.patch_size) ** 2 , dtype=torch.int32)
#         img_idx = img_idx.reshape(idx ** 2, 1)
#         img_idx = torch.cat([img_idx, img_idx[:1]], dim=0)
#         img_idx[-1, -1] = PackingIndex.ID_CLS_TOKEN

#         packed_img_idx = torch.empty(
#             img_idx.shape[0],
#             img_idx.shape[1],
#             PackingIndex.NUM_METADATA - 1,
#             dtype=torch.int32,
#         )

#         packed_img_idx[:, :, PackingIndex.Y] = img_idx // idx
#         packed_img_idx[:, :, PackingIndex.X] = img_idx % idx
#         packed_img_idx[:, :, PackingIndex.HEIGHT].fill_(idx)
#         packed_img_idx[:, :, PackingIndex.WIDTH].fill_(idx)
#         packed_img_idx[:, :, PackingIndex.IDX] = img_idx
#         packed_img_idx = packed_img_idx.reshape(1, -1, PackingIndex.NUM_METADATA - 1)

#         rope_freq = self.get_rope_freqs(self.hidden_size // config.attention_heads // 2)
#         self.freqs_ci = self.update_rope_frequencies(packed_img_idx, rope_freq)

#         # layer norms
#         self.layernorm_pre =  LayerNorm(self.hidden_size, eps=1e-5)
#         self.layernorm_post = LayerNorm(self.hidden_size, eps=1e-5)

#         # encoders
#         self.model = Llama4VisionEncoder(config)
#         self.vision_adapter = Llama4VisionPixelShuffleMLP(config)

#     def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
#         inputs_embeds = self.embed_tokens(pixel_values)
#         return inputs_embeds
    

def load_attention(config, prefix: str, weights, layer_id):
    # Only defined in granite.
    bias = getattr(config, "attention_bias", False)
    head_size = config.hidden_size // config.num_attention_heads
    sizes = None
    prefixes = None

    # base_layer = TensorParallelColumnLinear.load_qkv(
    #     config,
    #     prefix=f"{prefix}.qkv_proj",
    #     weights=weights,
    #     bias=bias,
    #     num_heads=config.num_attention_heads,
    #     num_key_value_heads=config.num_key_value_heads,
    # )
    # prefixes = ["qkv_proj"]
    
    prefixes = ["q_proj", "k_proj", "v_proj"]
    sizes = [
            head_size * config.num_attention_heads,
            head_size * config.num_key_value_heads,
            head_size * config.num_key_value_heads,
        ]
    base_layer = TensorParallelColumnLinear.load_multi(
            config,
            prefixes=[f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"],
            dim=0,
            weights=weights,
            bias=bias,
        )

    return TensorParallelMultiAdapterLinear.load(
        base_layer=base_layer,
        layer_id=layer_id,
        layer_names=prefixes,
        sizes=sizes,
        process_group=weights.process_group,
    )
    
class Llama4TextL2Norm(torch.nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = 1e-6

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Reshape to complex: last dim becomes complex numbers
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))  # [12, 40, 64]
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))  # [12, 40, 64]

    # Apply rotary embedding (elementwise complex multiplication)
    xq_out = torch.view_as_real(xq_ * freqs_cis)  # [12, 40, 64, 2]
    xk_out = torch.view_as_real(xk_ * freqs_cis)  # [12, 40, 64, 2]

    # Flatten the last two dims back to real-valued representation
    xq_out = xq_out.reshape(*xq.shape)  # [12, 40, 128]
    xk_out = xk_out.reshape(*xk.shape)  # [12, 40, 128]

    return xq_out.type_as(xq), xk_out.type_as(xk)


class Llama4Attention(torch.nn.Module):
    def __init__(
        self,
        index: int,
        prefix: str,
        config,
        weights,
    ):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_heads

        config.rope_theta = getattr(config, "rope_theta", 10000)
        config.num_key_value_heads = getattr(
            config, "num_key_value_heads", config.num_attention_heads
        )
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
        if config.num_key_value_heads % weights.process_group.size() != 0:
            raise ValueError(
                f"`num_key_value_heads` must be divisible by `num_shards` (got `num_key_value_heads`: {config.num_key_value_heads} "
                f"and `num_shards`: {weights.process_group.size()}"
            )
        self.num_heads = self.num_heads // weights.process_group.size()
        self.num_key_value_heads = (
            config.num_key_value_heads // weights.process_group.size()
        )

        self.query_key_value = load_attention(config, prefix, weights, index)
        self.index = index

        self.kv_scales = get_kv_scales(weights, f"{prefix}")

        o_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.o_proj",
            weights=weights,
            bias=getattr(config, "attention_bias", False),
        )

        self.o_proj = TensorParallelAdapterRowLinear.load(
            o_proj,
            index,
            "o_proj",
            process_group=weights.process_group,
        )
        
        self.qk_norm = Llama4TextL2Norm(config.rms_norm_eps)

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
        kv_cache: KVCache,
        block_tables,
        slots,
        seqlen,
        max_s,
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

        kv = kv.view(-1, 2, self.num_key_value_heads * self.head_size)
        key = kv[:, 0]
        value = kv[:, 1]

        x, y = hidden_states.shape
        query = query.reshape(1, x, 8, -1)
        key = key.reshape(1, x, 8, -1)

        # query = query.reshape(-1,  self.head_size)
        # key = key.reshape(-1, self.head_size)

        query = self.qk_norm(query.contiguous())
        key = self.qk_norm(key.contiguous())

        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_key_value_heads, self.head_size)
        value = value.view(-1, self.num_key_value_heads, self.head_size)
        freqs_cis = torch.complex(cos, sin)
        query, key = apply_rotary_emb(
            query, key, freqs_cis.to(query.device)
        )
        # self.rotary_emb(query, key, cos.to(hidden_states.dtype), sin.to(hidden_states.dtype))
        # from pdb import set_trace; set_trace()
        # query = query.to(hidden_states.dtype)
        # key = key.to(hidden_states.dtype)
        # from pdb import set_trace; set_trace()
        kv_cache.store(
            key=key,
            value=value,
            slots=slots,
            kv_scales=self.kv_scales,
        )

        # Prefill
        if cu_seqlen_prefill is not None:
            # flash attention
            attn_output = attention(
                query=query,
                key=key,
                value=value,
                kv_scales=self.kv_scales,
                kv_cache=kv_cache,
                seqlen=seqlen,
                block_tables=block_tables,
                softmax_scale=self.softmax_scale,
            )
        # Decode
        else:
            attn_output = paged_attention(
                query,
                kv_cache,
                self.kv_head_mapping,
                self.softmax_scale,
                block_tables,
                seqlen,
                max_s,
                kv_scales=self.kv_scales,
            )
        # from pdb import set_trace; set_trace()
        return self.o_proj(
            attn_output.view(-1, self.num_heads * self.head_size), adapter_data
        )


class Llama4MLP(nn.Module):
    def __init__(self, prefix: str, config, weights, intermediate_size: int):
        super().__init__()
        self.hidden_act = config.hidden_act
        if self.hidden_act != "silu":
            # Bail out because MoE only supports silu.
            raise NotImplementedError(
                "Currently only `silu` is supported as an activation for Deepseek V2."
            )
        self.act = ACT2FN[self.hidden_act]

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

        self.intermediate_size = intermediate_size // weights.process_group.size()

        # TODO: This is a hotfix to be removed & properly refactored.
        self.quantize = config.quantize

    def forward(self, hidden_states: torch.Tensor, reduce: bool = True):
        if (
            SYSTEM == "rocm"
            and self.hidden_act == "silu"
            and hidden_states.dtype == torch.float16
            and hidden_states.shape[0] == 1
            and not self.quantize
            and self.hidden_size
            != 16384  # TODO: Temporary workaround for `LLMM_Silu` kernel not working with LLama3.1 405B; needs refactoring once fixed.
        ):
            out = torch.empty(
                hidden_states.shape[0],
                self.intermediate_size,
                dtype=hidden_states.dtype,
                device="cuda",
            )
            ops.LLMM_Silu(self.gate_up_proj.linear.weight, hidden_states, out, 8)
            return self.down_proj(out, reduce=reduce)
        else:
            gate_up_states = self.gate_up_proj(hidden_states)
            gate_up_states = gate_up_states.view(-1, 2, self.intermediate_size)
            return self.down_proj(
                self.act(gate_up_states[:, 0]) * gate_up_states[:, 1], reduce=reduce
            )


class Llama4MoE(nn.Module):
    def __init__(
        self,
        prefix,
        config,
        moe_layer_cls: Type[MoELayer],
        weights,
    ):
        super().__init__()

        self.hidden_dim = config.hidden_size

        # Gating
        self.gate = FastLinear.load(config, f"{prefix}.router", weights, bias=False)

        self.moe_layer = moe_layer_cls(
            prefix=f"{prefix}.experts",
            n_experts=config.num_local_experts,
            n_expert_group=None,
            renormalize=True,
            topk=config.num_experts_per_tok,
            topk_group=None,
            scoring_func="sigmoid",
            weights=weights,
        )
        assert isinstance(self.moe_layer, MoELayer)

        self.shared_experts = Llama4MLP(
            prefix=f"{prefix}.shared_expert",
            config=config,
            weights=weights,
            intermediate_size=config.intermediate_size
        )

        self.process_group = weights.process_group

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from pdb import set_trace; set_trace()
        if self.shared_experts is not None:
            shared_output = self.shared_experts(x, reduce=False)
        else:
            shared_output = None

        router_logits = self.gate(x)
        from pdb import set_trace; set_trace()

        out = self.moe_layer(x, gating_output=router_logits)
        from pdb import set_trace; set_trace()

        if shared_output is not None:
            out = out + shared_output

        # Reduce sum
        if self.process_group.size() > 1:
            torch.distributed.all_reduce(out, group=self.process_group)
        from pdb import set_trace; set_trace()

        return out.view(*x.shape)


class Llama4Layer(nn.Module):
    def __init__(self, prefix, layer_id, config, weights):
        super().__init__()
        prefix = f"{prefix}.layers.{layer_id}"

        self.self_attn = Llama4Attention(
            index=layer_id,
            prefix=f"{prefix}.self_attn",
            config=config,
            weights=weights,
        )

        # if (
        #     config.n_routed_experts is not None
        #     and layer_id >= config.first_k_dense_replace
        #     and layer_id % config.moe_layer_freq == 0
        # ):
        moe_layer_cls = (
            SparseMoELayer
            if SparseMoELayer.is_supported(weights)
            else DenseMoELayer
        )
        self.mlp = Llama4MoE(f"{prefix}.feed_forward", config, moe_layer_cls, weights)
        # else:
        #     self.mlp = Llama4MLP(
        #         prefix=f"{prefix}.mlp",
        #         config=config,
        #         weights=weights,
        #         intermediate_size=config.intermediate_size,
        #     )

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
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlen_prefill: torch.Tensor,
        kv_cache,
        block_tables: torch.Tensor,
        slots: torch.Tensor,
        seqlen: Seqlen,
        max_s: int,
        adapter_data,
    ):
        normed_hidden_states, residual = self.input_layernorm(hidden_states, residual)

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
            adapter_data,
        )
        from pdb import set_trace; set_trace()

        # faster post attention rms norm
        normed_attn_res_output, residual = self.post_attention_layernorm(
            attn_output, residual
        )
        from pdb import set_trace; set_trace()

        output = self.mlp(normed_attn_res_output)

        return output, residual


class Llama4Model(torch.nn.Module):
    def __init__(self, prefix: str, config, weights: Weights):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                Llama4Layer(
                    prefix,
                    layer_id,
                    config,
                    weights,
                )
                for layer_id in range(1)
            ]
        )
        self.norm = FastRMSNorm.load(
            prefix=f"{prefix}.norm", weights=weights, eps=config.rms_norm_eps
        )

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
        adapter_data,
    ) -> torch.Tensor:
        hidden_states = inputs_embeds
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
                adapter_data,
            )

        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states


class FlashLlama4ForCausalLM(torch.nn.Module):
    def __init__(self, prefix: str, config, weights: Weights):
        super().__init__()
        self.embed_tokens = TensorParallelEmbedding(
            prefix=f"{prefix}.model.embed_tokens", weights=weights
        )

        self.model = Llama4Model(
            "model" if not prefix else f"{prefix}.model", config, weights
        )
        self.lm_head = SpeculativeHead.load(
            config,
            prefix="lm_head" if not prefix else f"{prefix}.lm_head",
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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        hidden_states = self.embed_tokens(input_ids)
        
        hidden_states = self.model(
            hidden_states,
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
        logits, speculative_logits = self.lm_head(hidden_states)
        return logits, speculative_logits


class Llama4ForConditionalGeneration(torch.nn.Module):
    def __init__(
        self,
        prefix: str,
        config: PretrainedConfig,
        weights: Weights,
    ):
        super().__init__()
        self.config = config

        config.vision_config.quantize = config.quantize

        text_config = config.text_config
        text_config.speculator = config.speculator
        text_config.quantize = config.quantize
        
        # self.vision_model = FlashLlama4VisionModel(
        #     prefix=f"{prefix}.vision_model",
        #     config=config.vision_config,
        #     weights=weights,
        # )
        
        self.text_model = FlashLlama4ForCausalLM(
            prefix=f"language_model",
            config=text_config,
            weights=weights,
        )
        
        
        self.pad_token_id = (
            config.pad_token_id if config.pad_token_id is not None else -1
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
        pixel_values: torch.FloatTensor = None,
        # Unused here
        attention_mask: Optional[torch.BoolTensor] = None,
        pixel_attention_mask: Optional[torch.BoolTensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        adapter_data: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        inputs_embeds = self.text_model.embed_tokens(input_ids)

        # if pixel_values is not None:
        #     pixel_values = pixel_values.to(dtype=inputs_embeds.dtype)
        #     image_outputs = self.vision_model(pixel_values)
        #     vision_outputs = self.post_vision_model_layernorm(
        #         image_outputs.last_hidden_state
        #     )
        #     image_features = self.multimodal_projector(vision_outputs)

        #     image_token_mask = (input_ids == self.config.image_token_index).to(
        #         input_ids.device
        #     )
        #     inputs_embeds[image_token_mask] = image_features.view(
        #         -1, image_features.shape[-1]
        #     )

        hidden_states = self.text_model.model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            cu_seqlen_prefill=cu_seqlen_prefill,
            kv_cache=kv_cache,
            block_tables=block_tables,
            slots=slots,
            seqlen=seqlen,
            max_s=max_s,
            adapter_data=adapter_data,
        )

        if lm_head_indices is not None:
            hidden_states = hidden_states[lm_head_indices]
        logits, speculative_logits = self.text_model.lm_head(hidden_states)

        return logits, speculative_logits

