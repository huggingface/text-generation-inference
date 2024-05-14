from typing import Optional, Tuple, Union

import math
import torch
from torch import nn

from transformers.activations import ACT2FN
from transformers.modeling_attn_mask_utils import (
    _create_4d_causal_attention_mask,
    _prepare_4d_attention_mask,
)
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    ImageClassifierOutput,
)
from transformers import SiglipConfig, SiglipTextConfig, SiglipVisionConfig

from text_generation_server.layers.tensor_parallel import (
    TensorParallelEmbedding,
    TensorParallelColumnLinear,
    TensorParallelRowLinear,
)


class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, prefix, config: SiglipVisionConfig, weights):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )
        self.patch_embedding.weight = nn.Parameter(
            weights.get_tensor(f"{prefix}.patch_embedding.weight"), requires_grad=False
        )
        self.patch_embedding.bias = nn.Parameter(
            weights.get_tensor(f"{prefix}.patch_embedding.bias"), requires_grad=False
        )
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = TensorParallelEmbedding(
            prefix=f"{prefix}.position_embedding", weights=weights
        )
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions, device=weights.device).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        patch_embeds = self.patch_embedding(
            pixel_values
        )  # shape = [*, width, grid, grid]
        embeddings = patch_embeds.flatten(2).transpose(1, 2)

        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class SiglipTextEmbeddings(nn.Module):
    def __init__(self, config: SiglipTextConfig):
        super().__init__()
        embed_dim = config.hidden_size

        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(
            config.max_position_embeddings, embed_dim
        )

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        seq_length = (
            input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]
        )

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings

        return embeddings


class SiglipAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, prefix, config, weights):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.head_size = self.head_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.num_heads = self.num_heads // weights.process_group.size()
        self.embed_dim = self.embed_dim // weights.process_group.size()
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = TensorParallelColumnLinear.load(
            config, prefix=f"{prefix}.k_proj", weights=weights, bias=True
        )
        self.v_proj = TensorParallelColumnLinear.load(
            config, prefix=f"{prefix}.v_proj", weights=weights, bias=True
        )
        self.q_proj = TensorParallelColumnLinear.load(
            config, prefix=f"{prefix}.q_proj", weights=weights, bias=True
        )
        self.out_proj = TensorParallelRowLinear.load(
            config, prefix=f"{prefix}.out_proj", weights=weights, bias=True
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
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        # scale post matmul
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2)) * self.scale

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = (
                attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
                + attention_mask
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(attn_weights.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_size):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_size)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_size)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class SiglipMLP(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = TensorParallelColumnLinear.load(  # config.hidden_size, config.intermediate_size
            prefix=f"{prefix}.fc1", config=config, weights=weights, bias=True
        )
        self.fc2 = TensorParallelRowLinear.load(  # config.intermediate_size, config.hidden_size
            prefix=f"{prefix}.fc2", config=config, weights=weights, bias=True
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class SiglipEncoderLayer(nn.Module):
    def __init__(self, prefix, config: SiglipConfig, weights):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(
            prefix=f"{prefix}.self_attn", config=config, weights=weights
        )
        self.layer_norm1 = nn.LayerNorm.load(
            prefix=f"{prefix}.layer_norm1", weights=weights, eps=config.layer_norm_eps
        )
        self.mlp = SiglipMLP(prefix=f"{prefix}.mlp", config=config, weights=weights)
        self.layer_norm2 = nn.LayerNorm.load(
            prefix=f"{prefix}.layer_norm2", weights=weights, eps=config.layer_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(batch, seq_len, embed_dim)`.
            attention_mask (`torch.FloatTensor`):
                Attention mask of shape `(batch, 1, q_len, k_v_seq_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        if output_attentions:
            return hidden_states, attn_weights
        return hidden_states, None


class SiglipMultiheadAttentionPoolingHead(nn.Module):
    """Multihead Attention Pooling."""

    def __init__(self, prefix, config: SiglipVisionConfig, weights):
        super().__init__()

        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.attention = torch.nn.MultiheadAttention(
            config.hidden_size, config.num_attention_heads, batch_first=True
        )
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(prefix, config, weights)

    def forward(self, hidden_state):
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)

        hidden_state = self.attention(probe, hidden_state, hidden_state)[0]

        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)

        return hidden_state[:, 0]


import warnings


def _trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    # Values are generated by using a truncated uniform distribution and
    # then using the inverse CDF for the normal distribution.
    # Get upper and lower cdf values
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    # Uniformly fill tensor with values from [l, u], then translate to
    # [2l-1, 2u-1].
    tensor.uniform_(2 * l - 1, 2 * u - 1)

    # Use inverse cdf transform for normal distribution to get truncated
    # standard normal
    tensor.erfinv_()

    # Transform to proper mean, std
    tensor.mul_(std * math.sqrt(2.0))
    tensor.add_(mean)

    # Clamp to ensure it's in the proper range
    tensor.clamp_(min=a, max=b)


def trunc_normal_tf_(
    tensor: torch.Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0,
) -> torch.Tensor:
    """Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \\leq \text{mean} \\leq b`.

    NOTE: this 'tf' variant behaves closer to Tensorflow / JAX impl where the
    bounds [a, b] are applied when sampling the normal distribution with mean=0, std=1.0
    and the result is subsquently scaled and shifted by the mean and std args.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    """
    with torch.no_grad():
        _trunc_normal_(tensor, 0, 1.0, a, b)
        tensor.mul_(std).add_(mean)


from torch.nn.init import _calculate_fan_in_and_fan_out


def variance_scaling_(tensor, scale=1.0, mode="fan_in", distribution="normal"):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == "fan_in":
        denom = fan_in
    elif mode == "fan_out":
        denom = fan_out
    elif mode == "fan_avg":
        denom = (fan_in + fan_out) / 2

    variance = scale / denom

    if distribution == "truncated_normal":
        # constant is stddev of standard normal truncated to (-2, 2)
        trunc_normal_tf_(tensor, std=math.sqrt(variance) / 0.87962566103423978)
    elif distribution == "normal":
        with torch.no_grad():
            tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        with torch.no_grad():
            tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode="fan_in", distribution="truncated_normal")


def default_flax_embed_init(tensor):
    variance_scaling_(tensor, mode="fan_in", distribution="normal")


from transformers import PreTrainedModel


class SiglipPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = SiglipConfig
    base_model_prefix = "siglip"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, SiglipVisionEmbeddings):
            width = (
                self.config.vision_config.hidden_size
                if isinstance(self.config, SiglipConfig)
                else self.config.hidden_size
            )
            nn.init.normal_(module.position_embedding.weight, std=1 / np.sqrt(width))
        elif isinstance(module, nn.Embedding):
            default_flax_embed_init(module.weight)
        elif isinstance(module, SiglipAttention):
            nn.init.xavier_uniform_(module.q_proj.weight)
            nn.init.xavier_uniform_(module.k_proj.weight)
            nn.init.xavier_uniform_(module.v_proj.weight)
            nn.init.xavier_uniform_(module.out_proj.weight)
            nn.init.zeros_(module.q_proj.bias)
            nn.init.zeros_(module.k_proj.bias)
            nn.init.zeros_(module.v_proj.bias)
            nn.init.zeros_(module.out_proj.bias)
        elif isinstance(module, SiglipMLP):
            nn.init.xavier_uniform_(module.fc1.weight)
            nn.init.xavier_uniform_(module.fc2.weight)
            nn.init.normal_(module.fc1.bias, std=1e-6)
            nn.init.normal_(module.fc2.bias, std=1e-6)
        elif isinstance(module, SiglipMultiheadAttentionPoolingHead):
            nn.init.xavier_uniform_(module.probe.data)
            nn.init.xavier_uniform_(module.attention.in_proj_weight.data)
            nn.init.zeros_(module.attention.in_proj_bias.data)
        elif isinstance(module, SiglipModel):
            logit_scale_init = torch.log(torch.tensor(1.0))
            module.logit_scale.data.fill_(logit_scale_init)
            module.logit_bias.data.zero_()
        elif isinstance(module, (nn.Linear, nn.Conv2d)):
            lecun_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class SiglipEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`SiglipEncoderLayer`].

    Args:
        config: SiglipConfig
    """

    def __init__(self, prefix, config: SiglipConfig, weights):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [
                SiglipEncoderLayer(
                    prefix=f"{prefix}.layers.{i}", config=config, weights=weights
                )
                for i in range(config.num_hidden_layers)
            ]
        )

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[torch.Tensor] = None,
    ):
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            causal_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Causal mask for the text model. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
        """

        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            hidden_states, _ = encoder_layer(
                hidden_states,
                attention_mask,
                output_attentions=output_attentions,
            )

        return hidden_states


class SiglipVisionTransformer(nn.Module):
    def __init__(self, prefix, config: SiglipVisionConfig, weights):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(
            prefix=f"{prefix}.embeddings", config=config, weights=weights
        )
        self.encoder = SiglipEncoder(
            prefix=f"{prefix}.encoder", config=config, weights=weights
        )
        self.post_layernorm = nn.LayerNorm.load(
            prefix=f"{prefix}.post_layernorm",
            weights=weights,
            eps=config.layer_norm_eps,
        )

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
    ):
        r"""
        Returns:

        """
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.embeddings(pixel_values)

        # NOTE: up until this point, the code logits are exactly
        # the same as the transformers code. The values evaulate
        # slightly differently in our encoder layer.
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
        )
        last_hidden_state = encoder_outputs
        post_last_hidden_state = self.post_layernorm(last_hidden_state)

        return BaseModelOutputWithPooling(
            last_hidden_state=post_last_hidden_state,
            # pooler_output=pooled_output,
            # hidden_states=encoder_outputs,
        )
