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
""" PyTorch Idefics model."""
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    dataclass,
)
from text_generation_server.models.custom_modeling.idefics_config import IdeficsConfig
from text_generation_server.models.custom_modeling.idefics_vision import (
    IdeficsVisionTransformer,
)
from text_generation_server.models.custom_modeling.idefics_perceiver import (
    IdeficsPerceiverResampler,
)
from text_generation_server.layers import (
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    TensorParallelRowLinear,
    SpeculativeHead,
    FastLinear,
)
from text_generation_server.layers.rotary import PositionRotaryEmbedding
from text_generation_server.utils.import_utils import SYSTEM
from loguru import logger

if SYSTEM == "cuda":
    import dropout_layer_norm
elif SYSTEM == "rocm":
    from vllm._C import ops
else:
    dropout_layer_norm = None


@dataclass
class BaseModelOutputWithPastImage(BaseModelOutputWithPast):
    image_hidden_states: Optional[torch.FloatTensor] = None


@dataclass
class CausalLMOutputWithPastImage(CausalLMOutputWithPast):
    image_hidden_states: Optional[torch.FloatTensor] = None


# logger = logging.get_logger(__name__)

# _CONFIG_FOR_DOC = "IdeficsConfig"

# IDEFICS_PRETRAINED_MODEL_ARCHIVE_LIST = [
#     "HuggingFaceM4/idefics-9b",
#     "HuggingFaceM4/idefics-80b",
#     # See all Idefics models at https://huggingface.co/models?filter=idefics
# ]


def expand_inputs_for_generation(
    input_ids,
    expand_size=1,
    is_encoder_decoder=False,
    attention_mask=None,
    encoder_outputs=None,
    **model_kwargs,
):
    expanded_return_idx = (
        torch.arange(input_ids.shape[0])
        .view(-1, 1)
        .repeat(1, expand_size)
        .view(-1)
        .to(input_ids.device)
    )
    input_ids = input_ids.index_select(0, expanded_return_idx)

    if "token_type_ids" in model_kwargs:
        token_type_ids = model_kwargs["token_type_ids"]
        model_kwargs["token_type_ids"] = token_type_ids.index_select(
            0, expanded_return_idx
        )

    if attention_mask is not None:
        model_kwargs["attention_mask"] = attention_mask.index_select(
            0, expanded_return_idx
        )
        model_kwargs["image_attention_mask"] = model_kwargs[
            "image_attention_mask"
        ].index_select(0, expanded_return_idx)
        model_kwargs["pixel_values"] = model_kwargs["pixel_values"].index_select(
            0, expanded_return_idx
        )

    if is_encoder_decoder:
        if encoder_outputs is None:
            raise ValueError(
                "If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined."
            )
        encoder_outputs["last_hidden_state"] = (
            encoder_outputs.last_hidden_state.index_select(
                0, expanded_return_idx.to(encoder_outputs.last_hidden_state.device)
            )
        )
        model_kwargs["encoder_outputs"] = encoder_outputs
    return input_ids, model_kwargs


def update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=False):
    # must have this key set to at least None
    model_kwargs["past_key_values"] = model_kwargs.get("past_key_values", None)

    # update past
    if "past_key_values" in outputs:
        model_kwargs["past"] = outputs.past_key_values
    elif "mems" in outputs:
        model_kwargs["past"] = outputs.mems
    elif "past_buckets_states" in outputs:
        model_kwargs["past"] = outputs.past_buckets_states
    else:
        model_kwargs["past"] = None

    # update token_type_ids with last value
    if "token_type_ids" in model_kwargs:
        token_type_ids = model_kwargs["token_type_ids"]
        model_kwargs["token_type_ids"] = torch.cat(
            [token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1
        )

    # update attention masks
    if not is_encoder_decoder:
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))],
                dim=-1,
            )
        if "image_attention_mask" in model_kwargs:
            image_attention_mask = model_kwargs["image_attention_mask"]
            last_mask = image_attention_mask[:, -1, :].unsqueeze(1)
            model_kwargs["image_attention_mask"] = last_mask

    return model_kwargs


def prepare_inputs_for_generation(input_ids, past=None, **kwargs):
    token_type_ids = kwargs.get("token_type_ids", None)
    # only last token for inputs_ids if past is defined in kwargs
    if past:
        input_ids = input_ids[:, -1].unsqueeze(-1)
        if token_type_ids is not None:
            token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

    attention_mask = kwargs.get("attention_mask", None)
    position_ids = kwargs.get("position_ids", None)

    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past:
            position_ids = position_ids[:, -1].unsqueeze(-1)

    pixel_values = kwargs.get("pixel_values", None)
    image_attention_mask = kwargs.get("image_attention_mask", None)
    # if pixel_values is None or image_attention_mask is None:
    #     raise ValueError("pixel values and image attention mask cannot be None")

    return {
        "input_ids": input_ids,
        "past_key_values": past,
        "use_cache": kwargs.get("use_cache"),
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "pixel_values": pixel_values,
        "image_attention_mask": image_attention_mask,
    }


def freeze_model(model, module_exceptions=[]):
    mapping = {
        "LayerNorm": nn.LayerNorm,
        "Linear": nn.Linear,
        "Embedding": nn.Embedding,
    }
    module_exceptions_mapped = [mapping[m] for m in module_exceptions]
    for module in model.modules():
        if module_exceptions and any(
            [isinstance(module, t) for t in module_exceptions_mapped]
        ):
            module.requires_grad_(
                True
            )  # Explicitely setting it to true to avoid any mistakes
        else:
            module.requires_grad_(False)
    return model


class IdeficsDecoupledPartialTPEmbedding(nn.Module):
    def __init__(
        self,
        config,
        weights,
    ):
        super().__init__()
        self.num_embeddings = config.vocab_size
        self.weight = TensorParallelEmbedding(
            prefix="model.embed_tokens", weights=weights
        )
        self.additional_weight = nn.Parameter(
            weights.get_tensor("model.embed_tokens.additional_embedding.weight")
        )

    def forward(self, input_ids):
        # Clone so that we don't modify the original input_ids later on
        input_ids = input_ids.clone()
        additional_vocab_indices = torch.where(input_ids >= self.num_embeddings)
        input_ids_additional_vocab = input_ids[additional_vocab_indices]
        additional_embeddings = torch.nn.functional.embedding(
            input_ids_additional_vocab - self.num_embeddings, self.additional_weight
        )

        # for successful lookup replace input_ids with 0, the results of these will be discarded anyway
        input_ids[additional_vocab_indices] = 0
        full_vector = self.weight(input_ids)

        # overwrite the records with high indices
        full_vector[additional_vocab_indices] = additional_embeddings

        return full_vector


class IdeficsDecoupledTensorParallelLinear(nn.Module):
    # Derived from https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
    """
    Implements a decoupling of parameters to allow freezing (or not) a subset of the parameters. In practise, the
    regular `weight` can be trained or frozen (i.e. `partially_freeze=True`), and if `out_additional_features` > 0,
    then it will create `out_additional_features * in_features` additional parameters that are always trained. If
    `out_additional_features=0`, then the module defaults back to the regular behavior of `nn.Linear`.
    """

    def __init__(
        self,
        config,
        weights,
    ) -> None:
        super().__init__()
        self.fc = SpeculativeHead.load(config=config, prefix="lm_head", weights=weights)
        self.additional_fc = FastLinear.load(
            config=config,
            prefix="lm_head.additional_fc",
            weights=weights,
            bias=False,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output, speculative_logits = self.fc(input)
        additional_features = self.additional_fc(input)
        output = torch.cat((output, additional_features), -1)

        return output, speculative_logits

    def extra_repr(self) -> str:
        """Overwriting `nn.Linear.extra_repr` to include new parameters."""
        return "in_features={}, out_features={}, out_additional_features={}, bias={}, partially_freeze={}".format(
            self.in_features,
            self.out_features,
            self.out_additional_features,
            self.bias is not None,
            self.partially_freeze,
        )


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat(
            [
                torch.zeros(
                    tgt_len, past_key_values_length, dtype=dtype, device=device
                ),
                mask,
            ],
            dim=-1,
        )
    return mask[None, None, :, :].expand(
        bsz, 1, tgt_len, tgt_len + past_key_values_length
    )


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min
    )


class IdeficsRMSNorm(nn.Module):
    def __init__(self, prefix, weights, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()

        weight = weights.get_tensor(f"{prefix}.weight")
        self.weight = nn.Parameter(weight)
        self.variance_epsilon = eps

    def forward(self, hidden_states, residual=None):
        if hidden_states.shape[-1] > 8192:
            if residual is not None:
                hidden_states += residual
            residual = hidden_states

            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(
                variance + self.variance_epsilon
            )

            # convert into half-precision if necessary
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                hidden_states = hidden_states.to(self.weight.dtype)

            return self.weight * hidden_states
        elif SYSTEM == "cuda":
            # faster post attention rms norm
            unwrap = False
            if len(hidden_states.shape) > 2:
                unwrap = True
                shape = hidden_states.shape
                hidden_states = hidden_states.reshape(-1, shape[-1])

            normed_hidden_states, res, *rest = dropout_layer_norm.dropout_add_ln_fwd(
                hidden_states,
                residual,
                self.weight,
                None,
                None,
                None,
                None,
                None,
                0.0,
                self.variance_epsilon,
                1.0,
                0,
                None,
                False,
                True,  # Activate RMSNorm
            )
            if res is None:
                res = hidden_states

            if unwrap:
                normed_hidden_states = normed_hidden_states.view(*shape)

            return normed_hidden_states
        elif SYSTEM == "rocm":
            # We use VLLM RMSNorm kernel that can be compiled for RoCm, instead of Flash Attention ones that can not.
            if residual is not None:
                hidden_states += residual
            residual = hidden_states

            unwrap = False
            if len(hidden_states.shape) > 2:
                unwrap = True
                shape = hidden_states.shape
                hidden_states = hidden_states.reshape(-1, shape[-1])

            out = torch.empty_like(hidden_states)
            ops.rms_norm(
                out,
                hidden_states,
                self.weight.data,
                self.variance_epsilon,
            )

            if unwrap:
                out = out.view(*shape)

            return out
        else:
            raise ValueError(
                "Your system seem to be not supported. Please check your install or open an issue at https://github.com/huggingface/text-generation-inference/issues with a clear reproduction."
            )


# this was adapted from LlamaMLP
class IdeficsMLP(nn.Module):
    def __init__(
        self,
        config,
        prefix,
        weights,
    ):
        super().__init__()
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
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        gate_up_states = self.gate_up_proj(hidden_states)
        shape = gate_up_states.shape
        gate_up_states = gate_up_states.view(*shape[:-1], 2, shape[-1] // 2)
        return self.down_proj(
            self.act_fn(gate_up_states[:, :, 0]) * gate_up_states[:, :, 1]
        )


# this was adapted from LlamaAttention
class IdeficsAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config,
        prefix,
        weights,
        qk_layer_norms: bool = False,
        is_cross_attention: bool = False,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.dropout = config.dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.is_cross_attention = is_cross_attention

        # if not hasattr(nn.functional, "scaled_dot_product_attention"):
        #     raise ValueError("this model requires pytorch 2.0 or higher")

        if self.num_heads % weights.process_group.size() != 0:
            raise ValueError(
                f"`num_heads` must be divisible by `num_shards` (got `num_heads`: {self.num_heads} "
                f"and `num_shards`: {weights.process_group.size()}"
            )
        self.num_heads //= weights.process_group.size()

        if self.is_cross_attention:
            # kv_input_dim = (
            #     self.hidden_size if not hasattr(config.vision_config, "embed_dim") else config.vision_config.embed_dim
            # )
            self.q_proj = TensorParallelColumnLinear.load(
                config, prefix=f"{prefix}.q_proj", weights=weights, bias=False
            )
            self.k_proj = TensorParallelColumnLinear.load(
                config, prefix=f"{prefix}.k_proj", weights=weights, bias=False
            )
            self.v_proj = TensorParallelColumnLinear.load(
                config, prefix=f"{prefix}.v_proj", weights=weights, bias=False
            )
        else:
            self.qkv = TensorParallelColumnLinear.load_multi(
                config,
                prefixes=[f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"],
                dim=0,
                weights=weights,
                bias=False,
            )
        self.o_proj = TensorParallelRowLinear.load(
            config, prefix=f"{prefix}.o_proj", weights=weights, bias=False
        )
        self.rotary_emb = PositionRotaryEmbedding.static(
            config=config, dim=self.head_dim, base=10000.0, device=weights.device
        )
        self.qk_layer_norms = qk_layer_norms
        if self.qk_layer_norms:
            self.q_layer_norm = IdeficsRMSNorm(
                prefix=f"{prefix}.q_layer_norm",
                weights=weights,
                eps=config.rms_norm_eps,
            )
            self.k_layer_norm = IdeficsRMSNorm(
                prefix=f"{prefix}.q_layer_norm",
                weights=weights,
                eps=config.rms_norm_eps,
            )

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # if key_value_states are provided this layer is used as a cross-attention layer
        is_cross_attention = self.is_cross_attention or key_value_states is not None

        bsz, q_len, _ = hidden_states.size()

        if is_cross_attention:
            query_states = self.q_proj(hidden_states).view(
                bsz, q_len, self.num_heads, self.head_dim
            )  # .transpose(1, 2)
            query_states = query_states.transpose(1, 2)
            (
                _,
                kv_len,
                _,
            ) = (
                key_value_states.size()
            )  # Note that, in this case, `kv_len` == `kv_seq_len`
            key_states = (
                self.k_proj(key_value_states)
                .view(bsz, kv_len, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )
            value_states = (
                self.v_proj(key_value_states)
                .view(bsz, kv_len, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )
        else:
            qkv = self.qkv(hidden_states)
            query_states, key_states, value_states = qkv.split(
                self.num_heads * self.head_dim, dim=2
            )

            query_states = query_states.view(
                bsz, q_len, self.num_heads, self.head_dim
            )  # .transpose(1, 2)
            key_states = key_states.view(
                bsz, q_len, self.num_heads, self.head_dim
            )  # . transpose(1, 2)
            value_states = value_states.view(
                bsz, q_len, self.num_heads, self.head_dim
            )  # .transpose(1, 2)
            kv_seq_len = q_len
            if past_key_value is not None:
                kv_seq_len += past_key_value[0].shape[-2]
            max_s = max(kv_seq_len, q_len)
            cos, sin = self.rotary_emb.get_cos_sin(
                position_ids.view(-1), max_s, hidden_states.dtype
            )

            query_shape = query_states.shape
            key_shape = key_states.shape
            self.rotary_emb(
                query_states.view(-1, *query_shape[2:]),
                key_states.reshape(-1, *key_shape[2:]),
                cos,
                sin,
            )

            query_states = query_states.view(query_shape)
            key_states = key_states.view(key_shape)

            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        if self.qk_layer_norms:
            query_states = self.q_layer_norm(query_states)
            key_states = self.k_layer_norm(key_states)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

        attn_output = nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.dropout,
        )

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        attn_weights = None
        if output_attentions:
            logger.warning_once(
                "attn_weights are not extracted in scaled_dot_product_attention. The model returns None instead"
            )

        return attn_output, attn_weights, past_key_value


# this was adapted from LlamaDecoderLayer
class IdeficsDecoderLayer(nn.Module):
    def __init__(self, layer_id: int, config: IdeficsConfig, weights):
        super().__init__()
        self.process_group = weights.process_group
        self.hidden_size = config.hidden_size
        prefix = f"model.layers.{layer_id}"
        self.self_attn = IdeficsAttention(
            config=config,
            prefix=f"{prefix}.self_attn",
            weights=weights,
            qk_layer_norms=False,
            is_cross_attention=False,
        )
        self.mlp = IdeficsMLP(
            config=config,
            prefix=f"{prefix}.mlp",
            weights=weights,
        )
        self.input_layernorm = IdeficsRMSNorm(
            prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = IdeficsRMSNorm(
            prefix=f"{prefix}.post_attention_layernorm",
            weights=weights,
            eps=config.rms_norm_eps,
        )
        self.dropout = config.dropout

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        # hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        # hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class IdeficsGatedCrossAttentionLayer(nn.Module):
    def __init__(self, layer_id, config: IdeficsConfig, weights):
        super().__init__()
        self.process_group = weights.process_group
        self.hidden_size = config.hidden_size
        prefix = f"model.gated_cross_attn_layers.{layer_id}"
        self.cross_attn = IdeficsAttention(
            config=config,
            prefix=f"{prefix}.cross_attn",
            weights=weights,
            qk_layer_norms=True,
            is_cross_attention=True,
        )
        self.mlp = IdeficsMLP(
            config=config,
            prefix=f"{prefix}.mlp",
            weights=weights,
        )
        self.input_layernorm = IdeficsRMSNorm(
            prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = IdeficsRMSNorm(
            prefix=f"{prefix}.post_attention_layernorm",
            weights=weights,
            eps=config.rms_norm_eps,
        )
        self.config = config.dropout

        self.act_cross_attn = nn.Tanh()
        self.act_dense = nn.Tanh()

        self.alpha_cross_attn = nn.Parameter(
            weights.get_tensor(f"{prefix}.alpha_cross_attn")
        )
        self.alpha_dense = nn.Parameter(weights.get_tensor(f"{prefix}.alpha_dense"))

        if not (hasattr(self, "alpha_cross_attn") and hasattr(self, "alpha_dense")):
            raise ValueError("Alpha parameters not initialized correctly!")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_hidden_states: Optional[torch.Tensor] = None,
        image_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        no_images: Optional[bool] = False,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            no_images (`bool`, *optional*, defaults to `False`): If `True` the vision part is ignored
        """
        if image_hidden_states is None:
            raise ValueError(
                "`image_hidden_states` is required for Idefics cross attention module which are visual features to be"
                " conditioned on."
            )

        if past_key_value is not None:
            raise NotImplementedError(
                "Past key value states are not implemented for Idefics cross attention module."
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.cross_attn(
            hidden_states=hidden_states,
            key_value_states=image_hidden_states,
            attention_mask=image_attention_mask,
            output_attentions=output_attentions,
        )
        # hidden_states = nn.functional.dropout(hidden_states, p=self.config, training=self.training)
        # when there are no images the model is used in pure language mode
        gate = 0 if no_images else 1
        hidden_states = (
            residual + gate * self.act_cross_attn(self.alpha_cross_attn) * hidden_states
        )

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        # hidden_states = nn.functional.dropout(hidden_states, p=self.config, training=self.training)
        hidden_states = residual + self.act_dense(self.alpha_dense) * hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`IdeficsConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


# @add_start_docstrings(
#     "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
#     LLAMA_START_DOCSTRING,
# )
class IdeficsPreTrainedModel(PreTrainedModel):
    config_class = IdeficsConfig
    # base_model_prefix = "model"
    # supports_gradient_checkpointing = True
    # _no_split_modules = ["IdeficsDecoderLayer", "IdeficsGatedCrossAttentionLayer"]

    # def _init_weights(self, module):
    #     # important: this ported version of Idefics isn't meant for training from scratch - only
    #     # inference and fine-tuning - so the proper init weights code has been removed - the m4 code
    #     # base should be used for training from scratch and it contains the correct code.
    #     std = self.config.initializer_range
    #     if isinstance(module, nn.Linear):
    #         module.weight.data.normal_(mean=0.0, std=std)
    #         if module.bias is not None:
    #             module.bias.data.zero_()
    #     elif isinstance(module, nn.Embedding):
    #         module.weight.data.normal_(mean=0.0, std=std)
    #         if module.padding_idx is not None:
    #             module.weight.data[module.padding_idx].zero_()

    # def _set_gradient_checkpointing(self, module, value=False):
    #     if isinstance(module, IdeficsModel):
    #         module.gradient_checkpointing = value


# LLAMA_INPUTS_DOCSTRING = r"""
#     Args:
#         input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
#             Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
#             it.

#             Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
#             [`PreTrainedTokenizer.__call__`] for details.

#             [What are input IDs?](../glossary#input-ids)
#         attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
#             Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

#             - 1 for tokens that are **not masked**,
#             - 0 for tokens that are **masked**.

#             [What are attention masks?](../glossary#attention-mask)

#             Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
#             [`PreTrainedTokenizer.__call__`] for details.

#             If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
#             `past_key_values`).

#             If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
#             and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
#             information on the default strategy.

#             - 1 indicates the head is **not masked**,
#             - 0 indicates the head is **masked**.
#         position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
#             Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
#             config.n_positions - 1]`. [What are position IDs?](../glossary#position-ids)
#         past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
#             Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
#             `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
#             `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

#             Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
#             blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

#             If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
#             don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
#             `decoder_input_ids` of shape `(batch_size, sequence_length)`.
#         inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
#             Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
#             is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
#             model's internal embedding lookup matrix.
#         use_cache (`bool`, *optional*):
#             If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
#             `past_key_values`).
#         output_attentions (`bool`, *optional*):
#             Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
#             tensors for more detail.
#         output_hidden_states (`bool`, *optional*):
#             Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
#             more detail.
#         return_dict (`bool`, *optional*):
#             Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
# """


# @add_start_docstrings(
#     "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
#     LLAMA_START_DOCSTRING,
# )
class IdeficsModel(IdeficsPreTrainedModel):
    # """
    # Transformer decoder consisting of `config.num_hidden_layers` layers. Each layer is a [`IdeficsDecoderLayer`]

    # Args:
    #     config: IdeficsConfig
    # """

    def __init__(self, config: IdeficsConfig, weights):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = IdeficsDecoupledPartialTPEmbedding(
            config=config,
            weights=weights,
        )

        self.image_size = config.vision_config.image_size
        self.vision_config = config.vision_config
        self.vision_model = IdeficsVisionTransformer(
            prefix="model.vision_model",
            config=config.vision_config,
            weights=weights,
        )

        # Perceiver Resampler
        if config.use_resampler:
            perceiver_config = config.perceiver_config
            self.perceiver_resampler = IdeficsPerceiverResampler(
                prefix="model.perceiver_resampler",
                config=config,
                embed_dim=config.vision_config.embed_dim,
                depth=perceiver_config.resampler_depth,
                n_heads=perceiver_config.resampler_n_heads,
                head_dim=perceiver_config.resampler_head_dim,
                n_latents=perceiver_config.resampler_n_latents,
                weights=weights,
            )

        self.layers = nn.ModuleList(
            [
                IdeficsDecoderLayer(layer_id, config, weights)
                for layer_id in range(config.num_hidden_layers)
            ]
        )

        self.cross_layer_interval = config.cross_layer_interval
        num_cross_layers = config.num_hidden_layers // self.cross_layer_interval
        self.gated_cross_attn_layers = nn.ModuleList(
            [
                IdeficsGatedCrossAttentionLayer(layer_id, config, weights)
                for layer_id in range(num_cross_layers)
            ]
        )
        # self.gradient_checkpointing = False

        self.norm = IdeficsRMSNorm(
            prefix="model.norm", weights=weights, eps=config.rms_norm_eps
        )

        # self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        # self.post_init()

        # self.freeze_relevant_params(config)

    # def freeze_relevant_params(self, config=None):
    #     if config is None:
    #         config = self.config

    #     if config.freeze_text_layers:
    #         self.freeze_text_layers(config.freeze_text_module_exceptions)

    #     if config.freeze_vision_layers:
    #         freeze_model(self.vision_model, module_exceptions=config.freeze_vision_module_exceptions)

    # def freeze_text_layers(self, module_exceptions=[]):
    #     for module in [self.layers, self.norm]:
    #         freeze_model(module, module_exceptions=module_exceptions)

    # def freeze_vision_layers(self, module_exceptions=[]):
    #     freeze_model(self.vision_model, module_exceptions=module_exceptions)

    # def get_input_embeddings(self):
    #     return self.embed_tokens

    # def set_input_embeddings(self, value):
    #     self.embed_tokens = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(
        self, attention_mask, input_shape, inputs_embeds, past_key_values_length
    ):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            ).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    # @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_hidden_states: Optional[torch.FloatTensor] = None,
        image_embeddings: Optional[torch.FloatTensor] = None,
        image_attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastImage]:
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
        elif position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        no_images = False

        if image_hidden_states is None:
            if pixel_values is None and image_embeddings is None:
                raise ValueError(
                    "Either pixel_values and image_embeddings have to be not-None."
                )

            elif pixel_values is not None and image_embeddings is not None:
                raise ValueError(
                    "You cannot specify both pixel_values and image_embeddings at the same time"
                )

            elif pixel_values is not None:
                no_images = len(torch.nonzero(pixel_values)) == 0
                pixel_values = pixel_values.to(
                    dtype=self.dtype, device=device
                )  # fp16 compatibility
                batch_size, num_images = pixel_values.shape[:2]
                pixel_values = pixel_values.contiguous().view(
                    batch_size * num_images, *pixel_values.shape[2:]
                )

                # Get sequence from the vision encoder
                image_hidden_states = self.vision_model(
                    pixel_values=pixel_values
                ).last_hidden_state

            elif image_embeddings is not None:
                (
                    batch_size,
                    num_images,
                    image_seq_len,
                    image_hidden_size,
                ) = image_embeddings.size()
                image_hidden_states = image_embeddings.to(
                    dtype=self.dtype, device=input_ids.device
                )
                image_hidden_states = image_hidden_states.view(
                    batch_size * num_images, image_seq_len, image_hidden_size
                )

            if self.config.use_resampler:
                image_hidden_states = self.perceiver_resampler(image_hidden_states)
            image_seq_len, image_hidden_size = image_hidden_states.size(
                1
            ), image_hidden_states.size(2)
            image_hidden_states = image_hidden_states.view(
                batch_size, num_images * image_seq_len, image_hidden_size
            )
        else:
            no_images = False
            num_images = pixel_values.shape[1]
            image_seq_len = image_hidden_states.shape[1] // num_images

        # # Hack to use the model in full language modeling mode
        # image_attention_mask = torch.zeros(batch_size, seq_length, 1, dtype=torch.long, device=image_hidden_states.device)
        # Make image_attention_mask compatible with hidden states
        text_seq_len = image_attention_mask.size(1)
        image_attention_mask = image_attention_mask.unsqueeze(-1)
        image_attention_mask = image_attention_mask.repeat(1, 1, 1, image_seq_len)
        image_attention_mask = image_attention_mask.view(
            batch_size, text_seq_len, num_images * image_seq_len
        )
        image_batch_size, image_sequence_length, _ = image_hidden_states.size()
        image_hidden_shape = (image_batch_size, image_sequence_length)
        if image_attention_mask is None:
            image_attention_mask = torch.ones(image_hidden_shape, device=device)
        image_attention_mask = self.invert_attention_mask(image_attention_mask)

        # if list(image_attention_mask.shape) != [4, 1, 1024, 64]:
        #     raise ValueError(f"Image hidden_states {image_hidden_states.shape} - mask {image_attention_mask.shape} {num_images} {image_seq_len} {text_seq_len}")

        # if image_hidden_states is not None:
        # else:
        #     image_attention_mask = None

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past),
                dtype=torch.bool,
                device=inputs_embeds.device,
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )

        hidden_states = inputs_embeds

        # if self.gradient_checkpointing and self.training:
        #     if use_cache:
        #         logger.warning_once(
        #             "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
        #         )
        #         use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            def vblock(
                main_block,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                image_hidden_states,
                image_attention_mask,
                output_attentions,
                use_cache,
                no_images,
                layer_idx,
                cross_layer_interval,
                gated_cross_attn_layers,
            ):
                # TODO(ls): Add cross attention values to respective lists
                if layer_idx % cross_layer_interval == 0:
                    xblock = gated_cross_attn_layers[layer_idx // cross_layer_interval]
                    outputs = xblock(
                        hidden_states,
                        attention_mask=attention_mask,
                        image_hidden_states=image_hidden_states,
                        image_attention_mask=image_attention_mask,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        past_key_value=None,  # not implemented
                        no_images=no_images,
                    )
                    hidden_states = outputs[0]

                layer_outputs = main_block(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

                return layer_outputs

            # if self.gradient_checkpointing and self.training:
            #     past_key_value = None
            #     if use_cache:
            #         logger.warning_once(
            #             "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            #         )
            #         use_cache = False

            #     layer_outputs = torch.utils.checkpoint.checkpoint(
            #         vblock,
            #         decoder_layer,
            #         hidden_states,
            #         attention_mask,
            #         position_ids,
            #         past_key_value,
            #         image_hidden_states,
            #         image_attention_mask,
            #         output_attentions,
            #         use_cache,
            #         no_images,
            #         idx,
            #         self.cross_layer_interval,
            #         self.gated_cross_attn_layers,
            #     )
            # else:
            layer_outputs = vblock(
                decoder_layer,
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                image_hidden_states=image_hidden_states,
                image_attention_mask=image_attention_mask,
                output_attentions=output_attentions,
                use_cache=use_cache,
                no_images=no_images,
                layer_idx=idx,
                cross_layer_interval=self.cross_layer_interval,
                gated_cross_attn_layers=self.gated_cross_attn_layers,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPastImage(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            image_hidden_states=image_hidden_states,
        )


class IdeficsForVisionText2Text(IdeficsPreTrainedModel):
    def __init__(
        self,
        config,
        weights,
    ):
        super().__init__(config)
        self.model = IdeficsModel(
            config=config,
            weights=weights,
        )

        self.lm_head = IdeficsDecoupledTensorParallelLinear(
            config=config,
            weights=weights,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_embeddings: Optional[torch.FloatTensor] = None,
        image_hidden_states: Optional[torch.FloatTensor] = None,
        image_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPastImage]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you consciours? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you consciours? Can you talk to me?\nI'm not consciours, but I can talk to you."
        ```"""

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            image_embeddings=image_embeddings,
            image_hidden_states=image_hidden_states,
            image_attention_mask=image_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits, speculative_logits = self.lm_head(hidden_states)

        loss = None

        return (
            CausalLMOutputWithPastImage(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                image_hidden_states=outputs.image_hidden_states,
            ),
            speculative_logits,
        )

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        inputs = prepare_inputs_for_generation(input_ids, past=past, **kwargs)
        unwanted_kwargs = ["token_type_ids"]
        for kwarg in unwanted_kwargs:
            inputs.pop(kwarg, None)
        return inputs

    @staticmethod
    def _expand_inputs_for_generation(
        *args,
        **model_kwargs,
    ):
        return expand_inputs_for_generation(*args, **model_kwargs)

    @staticmethod
    def _update_model_kwargs_for_generation(
        outputs, model_kwargs, is_encoder_decoder=False
    ):
        return update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=is_encoder_decoder
        )

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx) for past_state in layer_past
                ),
            )
        return reordered_past
