import torch
import torch.distributed

from torch import nn
from transformers.activations import ACT2FN
from typing import Optional

# Flash attention imports
import flash_attn_cuda
from text_generation_server.utils.layers import (
    FastLinear,
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    FastLayerNorm,
)


class FlashMQAttention(torch.nn.Module):
    def __init__(
        self,
        num_heads,
        hidden_size,
        process_group=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_size = hidden_size // num_heads

        self.softmax_scale = self.head_size ** (-0.5)

        if process_group is None:
            self.c_attn = FastLinear(hidden_size, hidden_size + 2 * self.head_size)
            self.c_proj = FastLinear(hidden_size, hidden_size)
        else:
            self.num_heads = self.num_heads // process_group.size()
            self.c_attn = FastLinear(hidden_size, self.head_size * (self.num_heads + 2))
            self.c_proj = TensorParallelRowLinear(
                hidden_size,
                hidden_size,
                process_group=process_group,
            )

    def forward(
        self,
        hidden_states,
        cu_seqlens,
        max_s,
        layer_past,
        layer_past_present_indices,
        cu_seqlens_q,
    ):
        qkv = self.c_attn(hidden_states)

        # Split query from key_value
        query, key_value = qkv.split(
            [self.head_size * self.num_heads, 2 * self.head_size], dim=1
        )

        # Prepare query and key_value for indexing
        query = query.view(-1, self.num_heads, self.head_size)
        key_value = key_value.view(-1, 2, 1, self.head_size)

        # Prefill
        if layer_past_present_indices is None:
            # Copy to layer past
            layer_past[...] = key_value
            # Expand from 1 to num_heads
            key_value = key_value.expand(-1, 2, self.num_heads, self.head_size)

            # output
            attn_output = torch.empty_like(query)
            # flash attention
            flash_attn_cuda.fwd(
                query,
                key_value[:, 0],
                key_value[:, 1],
                attn_output,
                cu_seqlens,
                cu_seqlens,
                max_s,
                max_s,
                0.0,
                self.softmax_scale,
                False,
                True,
                False,
                0,
                None,
            )
        # Decode
        else:
            # Add present to the layer_past tensor at the correct indices
            layer_past[layer_past_present_indices] = key_value
            # Expand from 1 to num_heads
            key_value = layer_past.expand(-1, 2, self.num_heads, self.head_size)

            # output
            attn_output = torch.empty_like(query)
            # flash attention
            flash_attn_cuda.fwd(
                query,
                key_value[:, 0],
                key_value[:, 1],
                attn_output,
                cu_seqlens_q,
                cu_seqlens,
                1,
                max_s,
                0.0,
                self.softmax_scale,
                False,
                False,
                False,
                0,
                None,
            )

        return self.c_proj(attn_output.view(-1, self.num_heads * self.head_size))


class MLP(nn.Module):
    def __init__(self, act, hidden_size, intermediate_size, process_group=None):
        super().__init__()
        self.act = (
            ACT2FN[act]
            if "gelu" not in act
            else lambda x: torch.nn.functional.gelu(
                x,
                approximate="tanh"
                if act in ["gelu_fast", "gelu_pytorch_tanh"]
                else "none",
            )
        )

        if process_group is None:
            self.c_fc = FastLinear(hidden_size, intermediate_size)
            self.c_proj = FastLinear(intermediate_size, hidden_size)
        else:
            self.c_fc = TensorParallelColumnLinear(
                hidden_size,
                intermediate_size,
                process_group=process_group,
            )
            self.c_proj = TensorParallelRowLinear(
                intermediate_size,
                hidden_size,
                process_group=process_group,
            )

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        return hidden_states


class Block(nn.Module):
    def __init__(
        self,
        num_heads,
        act,
        hidden_size,
        intermediate_size,
        layer_norm_eps,
        process_group=None,
    ):
        super().__init__()
        self.ln_1 = FastLayerNorm(hidden_size, eps=layer_norm_eps)
        self.ln_2 = FastLayerNorm(hidden_size, eps=layer_norm_eps)
        self.attn = FlashMQAttention(
            num_heads,
            hidden_size,
            process_group,
        )
        self.mlp = MLP(
            act,
            hidden_size,
            intermediate_size,
            process_group,
        )

    def forward(
        self,
        hidden_states,
        residual,
        cu_seqlens,
        max_s,
        layer_past,
        layer_past_present_indices,
        cu_seqlens_q,
    ):
        hidden_states, residual = self.ln_1(hidden_states, residual)

        hidden_states = self.attn(
            hidden_states,
            cu_seqlens,
            max_s,
            layer_past,
            layer_past_present_indices,
            cu_seqlens_q,
        )

        hidden_states, residual = self.ln_2(hidden_states, residual)

        mlp_output = self.mlp(hidden_states)

        return mlp_output, residual


class FlashSantacoderModel(nn.Module):
    def __init__(self, config, process_group=None):
        super().__init__()
        self.config = config

        self.process_group = process_group
        self.tp_embeddings = False
        if process_group is not None:
            self.tp_rank = process_group.rank()
            self.tp_world_size = process_group.size()
            if config.vocab_size % self.tp_world_size == 0:
                self.tp_embeddings = True

        if self.tp_embeddings:
            self.wte = TensorParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                reduce=False,
                process_group=process_group,
            )
            self.wpe = TensorParallelEmbedding(
                config.max_position_embeddings,
                config.hidden_size,
                reduce=False,
                process_group=process_group,
            )
        else:
            self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
            self.wpe = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.h = nn.ModuleList(
            [
                Block(
                    config.num_attention_heads,
                    config.activation_function,
                    config.hidden_size,
                    config.n_inner
                    if config.n_inner is not None
                    else 4 * config.hidden_size,
                    config.layer_norm_epsilon,
                    process_group,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.ln_f = FastLayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        self.head_size = self.h[0].attn.head_size
        self.num_heads = self.h[0].attn.num_heads

    def post_load_weights(self, quantize: Optional[str] = None):
        if self.tp_embeddings:
            self.wte.add_null_idx()
            self.wpe.add_null_idx()
        for layer in self.h:
            layer: Block
            layer.attn.c_attn.prepare_weights(quantize)
            layer.attn.c_proj.prepare_weights(quantize)
            layer.mlp.c_fc.prepare_weights(quantize)
            layer.mlp.c_proj.prepare_weights(quantize)

    def forward(
        self,
        input_ids,
        position_ids,
        cu_seqlens,
        cu_seqlens_q,
        max_s,
        past_key_values: Optional[torch.Tensor] = None,
        pre_allocate_past_size: Optional[int] = None,
    ):
        hidden_states = self.wte(input_ids) + self.wpe(position_ids)
        if self.tp_embeddings:
            torch.distributed.all_reduce(hidden_states, group=self.process_group)

        # Prefill
        if past_key_values is None:
            # Create past tensor
            past_key_values = hidden_states.new_empty(
                (
                    len(self.h),
                    len(hidden_states)
                    if pre_allocate_past_size is None
                    else pre_allocate_past_size,
                    2,
                    1,
                    self.head_size,
                )
            )
            layer_past_present_indices = None
            slice_past_index = len(hidden_states)
        # Decode
        else:
            # Create indices from cumulative sequence lengths
            layer_past_present_indices = cu_seqlens[1:] - 1
            slice_past_index = None

        residual = None
        for i, layer in enumerate(self.h):
            # We added padding that we now need to slice
            layer_past_key_values = (
                past_key_values[i]
                if slice_past_index is None
                else past_key_values[i, :slice_past_index]
            )

            hidden_states, residual = layer(
                hidden_states,
                residual,
                cu_seqlens,
                max_s,
                layer_past_key_values,
                layer_past_present_indices,
                cu_seqlens_q,
            )

        hidden_states, _ = self.ln_f(hidden_states, residual)

        return hidden_states, past_key_values


class FlashSantacoderForCausalLM(nn.Module):
    def __init__(self, config, process_group=None):
        super().__init__()

        self.transformer = FlashSantacoderModel(config, process_group)

        if self.transformer.tp_embeddings:
            self.lm_head = FastLinear(
                config.hidden_size,
                config.vocab_size // process_group.size(),
                bias=False,
            )
        else:
            self.lm_head = FastLinear(config.hidden_size, config.vocab_size, bias=False)

    def post_load_weights(self, quantize: Optional[str] = None):
        self.transformer.post_load_weights(quantize)
        self.lm_head.prepare_weights()

    def forward(
        self,
        input_ids,
        position_ids,
        cu_seqlens,
        cu_seqlens_q,
        max_s,
        past_key_values: Optional[torch.Tensor] = None,
        pre_allocate_past_size: Optional[int] = None,
        lm_head_indices: Optional[torch.Tensor] = None,
    ):
        hidden_states, present = self.transformer(
            input_ids,
            position_ids,
            cu_seqlens,
            cu_seqlens_q,
            max_s,
            past_key_values,
            pre_allocate_past_size,
        )
        if lm_head_indices is not None:
            hidden_states = hidden_states[lm_head_indices]
        logits = self.lm_head(hidden_states)

        if self.transformer.tp_embeddings:
            # Logits are sharded, so we need to gather them
            if logits.shape[0] == 1:
                # Fast path when batch size is 1
                world_logits = logits.new_empty(
                    (logits.shape[1] * self.transformer.tp_world_size)
                )
                torch.distributed.all_gather_into_tensor(
                    world_logits, logits.view(-1), group=self.transformer.process_group
                )
                world_logits = world_logits.view(1, -1)
            else:
                # We cannot use all_gather_into_tensor as it only support concatenating on the first dim
                world_logits = [
                    torch.empty_like(logits)
                    for _ in range(self.transformer.tp_world_size)
                ]
                torch.distributed.all_gather(
                    world_logits, logits, group=self.transformer.process_group
                )
                world_logits = torch.cat(world_logits, dim=1)

            return world_logits, present

        return logits, present
