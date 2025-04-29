# coding=utf-8
# Copyright 2024 the HuggingFace Inc. team. All rights reserved.
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
""" PyTorch Idefics3 model."""

from typing import List, Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from text_generation_server.models.custom_modeling.vlm import (
    load_text_model,
)
from text_generation_server.layers.attention import Seqlen, HPUPagedAttentionMetadata
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask

from text_generation_server.layers import (
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    TensorParallelRowLinear,
)
from text_generation_server.utils.weights import DefaultWeightsLoader, UnquantizedWeight


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


class Idefics3VisionEmbeddings(nn.Module):
    """
    This is a modified version of `siglip.modelign_siglip.SiglipVisionEmbeddings` to enable images of variable
    resolution.

    The modifications are adapted from [Patch n' Pack: NaViT, a Vision Transformer for any Aspect Ratio and Resolution](https://arxiv.org/abs/2307.06304)
    which allows treating images in their native aspect ratio and without the need to resize them to the same
    fixed size. In particular, we start from the original pre-trained SigLIP model
    (which uses images of fixed-size square images) and adapt it by training on images of variable resolutions.
    """

    def __init__(self, prefix, config, weights):
        super().__init__()
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

        self.num_patches_per_side = self.image_size // self.patch_size
        self.num_patches = self.num_patches_per_side**2
        self.num_positions = self.num_patches
        self.position_embedding = TensorParallelEmbedding(
            prefix=f"{prefix}.position_embedding", weights=weights
        )

    def forward(
        self, pixel_values: torch.FloatTensor, patch_attention_mask: torch.BoolTensor
    ) -> torch.Tensor:
        batch_size, _, max_im_h, max_im_w = pixel_values.shape

        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(2).transpose(1, 2)

        max_nb_patches_h, max_nb_patches_w = (
            max_im_h // self.patch_size,
            max_im_w // self.patch_size,
        )
        boundaries = torch.arange(
            1 / self.num_patches_per_side, 1.0, 1 / self.num_patches_per_side
        )
        position_ids = torch.full(
            size=(batch_size, max_nb_patches_h * max_nb_patches_w), fill_value=0
        )

        for batch_idx, p_attn_mask in enumerate(patch_attention_mask):
            nb_patches_h = p_attn_mask[:, 0].sum()
            nb_patches_w = p_attn_mask[0].sum()

            fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / nb_patches_h)
            fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / nb_patches_w)

            bucket_coords_h = torch.bucketize(
                fractional_coords_h, boundaries, right=True
            )
            bucket_coords_w = torch.bucketize(
                fractional_coords_w, boundaries, right=True
            )

            pos_ids = (
                bucket_coords_h[:, None] * self.num_patches_per_side + bucket_coords_w
            ).flatten()
            position_ids[batch_idx][p_attn_mask.view(-1).cpu()] = pos_ids

        position_ids = position_ids.to(self.position_embedding.weight.device)
        embeddings = embeddings + self.position_embedding(position_ids)
        return embeddings


class Idefics3VisionAttention(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_size = self.embed_dim // self.num_heads
        if self.head_size * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_size**-0.5
        self.dropout = config.attention_dropout

        self.num_heads = self.num_heads // weights.process_group.size()
        self.embed_dim = self.embed_dim // weights.process_group.size()

        self.qkv = TensorParallelColumnLinear.load_multi(
            config,
            prefixes=[f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"],
            dim=0,
            weights=weights,
            bias=True,
        )
        self.out_proj = TensorParallelRowLinear.load(
            config=config, prefix=f"{prefix}.out_proj", weights=weights, bias=True
        )
        self.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, q_len, _ = hidden_states.size()

        qkv = self.qkv(hidden_states)
        query_states, key_states, value_states = qkv.split(
            [
                self.head_size * self.num_heads,
                self.head_size * self.num_heads,
                self.head_size * self.num_heads,
            ],
            dim=2,
        )

        query_states = query_states.view(
            batch_size, q_len, self.num_heads, self.head_size
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size, q_len, self.num_heads, self.head_size
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, q_len, self.num_heads, self.head_size
        ).transpose(1, 2)

        k_v_seq_len = key_states.shape[-2]
        attn_weights = (
            torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale
        )

        if attn_weights.size() != (batch_size, self.num_heads, q_len, k_v_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, q_len, k_v_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (batch_size, 1, q_len, k_v_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(batch_size, 1, q_len, k_v_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, q_len, self.head_size):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, q_len, self.head_size)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output


class Idefics3VisionMLP(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = TensorParallelColumnLinear.load(
            prefix=f"{prefix}.fc1", config=config, weights=weights, bias=True
        )
        self.fc2 = TensorParallelRowLinear.load(
            prefix=f"{prefix}.fc2", config=config, weights=weights, bias=True
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class Idefics3EncoderLayer(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = Idefics3VisionAttention(
            prefix=f"{prefix}.self_attn", config=config, weights=weights
        )
        self.layer_norm1 = nn.LayerNorm.load(
            prefix=f"{prefix}.layer_norm1", eps=config.layer_norm_eps, weights=weights
        )
        self.layer_norm2 = nn.LayerNorm.load(
            prefix=f"{prefix}.layer_norm2", eps=config.layer_norm_eps, weights=weights
        )
        self.mlp = Idefics3VisionMLP(
            prefix=f"{prefix}.mlp", config=config, weights=weights
        )

    # Copied from transformers.models.siglip.modeling_siglip.SiglipEncoderLayer.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Idefics3Encoder(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [
                Idefics3EncoderLayer(
                    prefix=f"{prefix}.layers.{i}", config=config, weights=weights
                )
                for i in range(config.num_hidden_layers)
            ]
        )

    # Ignore copy
    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask,
            )
        return hidden_states


class Idefics3VisionTransformer(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        self.config = config
        self.embeddings = Idefics3VisionEmbeddings(
            prefix=f"{prefix}.embeddings", config=config, weights=weights
        )
        self.encoder = Idefics3Encoder(
            prefix=f"{prefix}.encoder", config=config, weights=weights
        )
        self.post_layernorm = nn.LayerNorm.load(
            prefix=f"{prefix}.post_layernorm",
            weights=weights,
            eps=config.layer_norm_eps,
        )

    def forward(
        self,
        pixel_values,
        patch_attention_mask: Optional[torch.BoolTensor] = None,
    ):
        batch_size = pixel_values.size(0)
        if patch_attention_mask is None:
            patch_size = self.config.patch_size
            patch_attention_mask = torch.ones(
                (
                    batch_size,
                    pixel_values.size(2) // patch_size,
                    pixel_values.size(3) // patch_size,
                )
            )
            patch_attention_mask = patch_attention_mask.to(
                dtype=torch.bool, device=pixel_values.device
            )

        hidden_states = self.embeddings(
            pixel_values=pixel_values, patch_attention_mask=patch_attention_mask
        )

        patch_attention_mask = patch_attention_mask.view(batch_size, -1)
        # The call to `_upad_input` in `_flash_attention_forward` is expensive
        # So when the `patch_attention_mask` is full of 1s (i.e. attending to the whole sequence),
        # avoiding passing the attention_mask, which is equivalent to attending to the full sequence
        if not torch.any(~patch_attention_mask):
            patch_attention_mask = None
        else:
            patch_attention_mask = _prepare_4d_attention_mask(
                patch_attention_mask, hidden_states.dtype
            )

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=patch_attention_mask,
        )

        last_hidden_state = encoder_outputs
        last_hidden_state = self.post_layernorm(last_hidden_state)

        return last_hidden_state


class Idefics3SimpleMLP(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        input_size = config.vision_config.hidden_size * (config.scale_factor**2)
        output_size = config.text_config.hidden_size
        proj = nn.Parameter(
            weights.get_tensor(f"{prefix}.modality_projection.proj.weight"),
            requires_grad=False,
        ).to(weights.dtype)
        self.proj = nn.Linear(input_size, output_size, bias=False)
        self.proj.weight = proj

    def forward(self, x):
        return self.proj(x)


class Idefics3Connector(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        self.modality_projection = Idefics3SimpleMLP(prefix, config, weights)
        self.scale_factor = config.scale_factor

    def pixel_shuffle(self, x, scale_factor=2):
        bsz, seq, embed_dim = x.size()
        height = width = int(seq**0.5)
        x = x.view(bsz, height, width, embed_dim)
        x = x.view(bsz, height, int(width / scale_factor), embed_dim * scale_factor)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(
            bsz,
            int(width / scale_factor),
            int(height / scale_factor),
            embed_dim * (scale_factor**2),
        )
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(bsz, int(seq / (scale_factor**2)), embed_dim * (scale_factor**2))
        return x

    def forward(self, image_hidden_states):
        image_hidden_states = self.pixel_shuffle(image_hidden_states, self.scale_factor)
        image_hidden_states = self.modality_projection(image_hidden_states)
        return image_hidden_states


class Idefics3ForConditionalGeneration(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        config.vision_config.quantize = None
        config.vision_config.speculator = config.speculator
        config.text_config.quantize = config.quantize
        config.text_config.speculator = config.speculator
        # set tie_word_embeddings to True to load `.embed_tokens.weight` instead of `.lm_head.weight`
        # since Idefics3 uses the `embed_tokens` for the final prediction
        # config.text_config.tie_word_embeddings = True

        vision_config = config.vision_config
        self.text_model = load_text_model(
            prefix="model" if not prefix else f"{prefix}.model",
            config=config.text_config,
            weights=weights,
            name="text_model",
        )
        self.dtype = weights.dtype

        # The vision and connector models are not quantized.
        with weights.use_loader(DefaultWeightsLoader(UnquantizedWeight)):
            self.vision_model = Idefics3VisionTransformer(
                prefix=(
                    f"{prefix}.model.vision_model" if prefix else "model.vision_model"
                ),
                config=vision_config,
                weights=weights,
            )

            config.quantize = None
            self.connector = Idefics3Connector(
                prefix=f"{prefix}.model.connector" if prefix else "model.connector",
                config=config,
                weights=weights,
            )

        self.config = config
        self.image_token_id = config.image_token_id
        self.pad_token_id = (
            config.pad_token_id if config.pad_token_id is not None else -1
        )

    def _merge_input_ids_with_image_features(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        image_features: torch.Tensor,
    ):
        """In place merges in vision_embeddings with inputs_embeds."""
        # mask = input_ids == self.config.image_token_index
        #  - replace `==` with torch.where to fix the issue in hpu graph
        mask = torch.where(input_ids == self.config.image_token_id)
        # Let's pray we have enabled enough slots !
        inputs_embeds[mask] = image_features.view(-1, image_features.shape[-1])
        return inputs_embeds

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
        pixel_values: torch.FloatTensor = None,
        pixel_attention_mask: Optional[torch.BoolTensor] = None,
        # Unused here
        image_sizes: Optional[torch.Tensor] = None,
        adapter_data: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cross_attention_states: Optional[torch.Tensor] = None,
        image_indices=None,
    ):
        inputs_embeds = self.text_model.embed_tokens(input_ids)
        if pixel_values is not None:
            batch_size, num_images, num_channels, height, width = pixel_values.shape
            all_states = []
            all_pixel_values = pixel_values
            all_pixel_mask = pixel_attention_mask
            for i in range(batch_size):
                pixel_values = all_pixel_values.to(
                    dtype=self.dtype
                )  # fp16 compatibility
                pixel_values = pixel_values[i : i + 1]
                pixel_values = pixel_values.view(num_images, *pixel_values.shape[2:])

                # Remove padding images - padding images are full 0.
                nb_values_per_image = pixel_values.shape[1:].numel()
                real_images_inds = (pixel_values == 0.0).sum(
                    dim=(-1, -2, -3)
                ) != nb_values_per_image
                pixel_values = pixel_values[real_images_inds].contiguous()
                # Handle the vision attention mask
                if pixel_attention_mask is None:
                    pixel_attention_mask = torch.ones(
                        size=(
                            pixel_values.size(0),
                            pixel_values.size(2),
                            pixel_values.size(3),
                        ),
                        dtype=torch.bool,
                        device=pixel_values.device,
                    )
                else:
                    # Remove padding images from the mask/pP p
                    pixel_attention_mask = all_pixel_mask[i : i + 1]
                    pixel_attention_mask = pixel_attention_mask.view(
                        1 * num_images, *pixel_attention_mask.shape[2:]
                    )
                    pixel_attention_mask = pixel_attention_mask[
                        real_images_inds
                    ].contiguous()

                patch_size = self.config.vision_config.patch_size
                """
                patches_subgrid = pixel_attention_mask.unfold(
                    dimension=1, size=patch_size, step=patch_size
                )
                patches_subgrid = patches_subgrid.unfold(
                    dimension=2, size=patch_size, step=patch_size
                )
                patch_attention_mask = (patches_subgrid.sum(dim=(-1, -2)) > 0).bool()
                """
                # hpu does none support unfold
                conv_kernel = torch.ones(
                    [1, 1, patch_size, patch_size],
                    dtype=pixel_values.dtype,
                    device=pixel_values.device,
                )
                patches_subgrid = torch.nn.functional.conv2d(
                    pixel_attention_mask.unsqueeze(1).to(conv_kernel.dtype),
                    conv_kernel,
                    stride=patch_size,
                ).squeeze(1)
                patch_attention_mask = torch.eq(
                    patches_subgrid, (patch_size * patch_size)
                )

                # Get sequence from the vision encoder
                image_hidden_states = self.vision_model(
                    pixel_values=pixel_values,
                    patch_attention_mask=patch_attention_mask,
                )

                # Modality projection & resampling
                image_hidden_states = self.connector(
                    image_hidden_states,
                )

                all_states.append(image_hidden_states)
            image_hidden_states = torch.stack(all_states, dim=0)

            inputs_embeds = self._merge_input_ids_with_image_features(
                input_ids, inputs_embeds, image_hidden_states
            )

        hidden_states = self.text_model.model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            cu_seqlen_prefill=cu_seqlen_prefill,
            kv_cache=kv_cache,
            slots=slots,
            seqlen=seqlen,
            hpu_attention_meta=hpu_attention_meta,
            adapter_data=adapter_data,
        )
        if lm_head_indices is not None:
            hidden_states = hidden_states[lm_head_indices]
        logits, speculative_logits = self.text_model.lm_head(hidden_states)
        return logits, speculative_logits
