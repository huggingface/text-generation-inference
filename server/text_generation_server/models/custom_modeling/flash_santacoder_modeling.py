import torch
import torch.distributed

from torch import nn
from transformers.activations import ACT2FN
from typing import Optional

# Flash attention imports
import flash_attn_cuda

from text_generation_server.utils.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    TensorParallelHead,
    TensorParallelEmbedding,
    FastLayerNorm,
    get_linear,
)


def load_multi_mqa(
    config, prefix: str, weights, bias: bool, head_size, num_heads, hidden_size
):
    if any("c_attn" in k for k in weights.routing.keys()):
        slice_ = weights._get_slice(f"{prefix}.c_attn.weight")
        shape = slice_.get_shape()
        world_size = weights.process_group.size()
        rank = weights.process_group.rank()
        if config.transpose:
            block_size = (shape[1] - 2 * head_size) // world_size
            start = rank * block_size
            stop = (rank + 1) * block_size
            assert (shape[1] - 2 * head_size) % world_size == 0
            q_tensor = slice_[:, start:stop]
            kv_tensor = slice_[:, -2 * head_size :]
            weight = torch.cat([q_tensor, kv_tensor], dim=1).T
        else:
            block_size = (shape[0] - 2 * head_size) // world_size
            start = rank * block_size
            stop = (rank + 1) * block_size
            assert (shape[0] - 2 * head_size) % world_size == 0
            q_tensor = slice_[start:stop]
            kv_tensor = slice_[-2 * head_size :]
            weight = torch.cat([q_tensor, kv_tensor], dim=0)
        if bias:
            slice_ = weights._get_slice(f"{prefix}.c_attn.bias")
            shape = slice_.get_shape()
            block_size = (shape[0] - 2 * head_size) // world_size
            assert (shape[0] - 2 * head_size) % world_size == 0
            q_tensor = slice_[start:stop]
            start = rank * block_size
            stop = (rank + 1) * block_size
            q_tensor = slice_[start:stop]
            kv_tensor = slice_[-2 * head_size :]
            bias = torch.cat([q_tensor, kv_tensor], dim=0)
    else:
        if config.transpose:
            w = [
                weights.get_sharded(f"{prefix}.q_attn.weight", dim=1).T,
                weights.get_tensor(f"{prefix}.kv_attn.weight").T,
            ]
            weight = torch.cat(w, dim=0)
        else:
            w = [
                weights.get_sharded(f"{prefix}.q_attn.weight", dim=0),
                weights.get_tensor(f"{prefix}.kv_attn.weight"),
            ]
            weight = torch.cat(w, dim=1)

        if bias:
            b = [
                weights.get_sharded(f"{prefix}.q_attn.bias", dim=0),
                weights.get_tensor(f"{prefix}.kv_attn.bias"),
            ]
            bias = torch.cat(b, dim=0)
        else:
            bias = None

    weight = weight.to(dtype=weights.dtype).to(device=weights.device)
    assert list(weight.shape) == [
        (num_heads + 2) * head_size,
        hidden_size,
    ], f"{weight.shape} != {[(num_heads + 2) * head_size, hidden_size]}"
    if bias is not None:
        bias = bias.to(dtype=weights.dtype).to(device=weights.device)
        assert list(bias.shape) == [
            (num_heads + 2) * head_size
        ], f"{weight.shape} != {[(num_heads + 2) * head_size]}"
    return TensorParallelColumnLinear(get_linear(weight, bias, config.quantize))


def load_col(config, prefix: str, weights, bias: bool):
    if config.transpose:
        weight = weights.get_sharded(f"{prefix}.weight", dim=1).T
    else:
        weight = weights.get_sharded(f"{prefix}.weight", dim=0)

    if bias:
        bias = weights.get_sharded(f"{prefix}.bias", dim=0)
    else:
        bias = None
    return TensorParallelColumnLinear(get_linear(weight, bias, config.quantize))


def load_row(config, prefix: str, weights, bias: bool):
    if config.transpose:
        weight = weights.get_sharded(f"{prefix}.weight", dim=0).T
    else:
        weight = weights.get_sharded(f"{prefix}.weight", dim=1)

    if bias and weights.process_group.rank() == 0:
        # Rank is only on the first rank process
        bias = weights.get_tensor(f"{prefix}.bias")
    else:
        bias = None
    return TensorParallelRowLinear(
        get_linear(weight, bias, config.quantize), process_group=weights.process_group
    )


class FlashMQAttention(torch.nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        num_heads = config.num_attention_heads
        hidden_size = config.hidden_size

        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_size = hidden_size // num_heads

        assert self.num_heads % weights.process_group.size() == 0
        self.num_heads = self.num_heads // weights.process_group.size()

        self.softmax_scale = self.head_size ** (-0.5)

        self.c_attn = load_multi_mqa(
            config,
            prefix=prefix,
            weights=weights,
            bias=True,
            head_size=self.head_size,
            hidden_size=hidden_size,
            num_heads=self.num_heads,
        )
        self.c_proj = load_row(
            config, prefix=f"{prefix}.c_proj", weights=weights, bias=True
        )

    def forward(
        self,
        hidden_states,
        start_seq,
        end_seq,
        start_seq_q,
        end_seq_q,
        max_s,
        layer_past,
        past_present_indices,
        prefill,
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
        if prefill:
            # Copy to layer past
            layer_past[...] = key_value
            # Expand from 1 to num_heads
            key_value = key_value.expand(-1, 2, self.num_heads, self.head_size)

            # output
            attn_output = torch.empty_like(query)
            # flash attention
            flash_attn_cuda.fwd(
                query,
                torch.select(key_value, dim=1, index=0),
                torch.select(key_value, dim=1, index=1),
                attn_output,
                start_seq,
                end_seq,
                start_seq,
                end_seq,
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
            layer_past[past_present_indices] = key_value
            # Expand from 1 to num_heads
            key_value = layer_past.expand(-1, 2, self.num_heads, self.head_size)

            # output
            attn_output = torch.empty_like(query)
            # flash attention
            flash_attn_cuda.fwd(
                query,
                torch.select(key_value, dim=1, index=0),
                torch.select(key_value, dim=1, index=1),
                attn_output,
                start_seq_q,
                end_seq_q,
                start_seq,
                end_seq,
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
    def __init__(self, prefix, config, weights):
        super().__init__()
        act = config.activation_function
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

        self.c_fc = load_col(
            config, prefix=f"{prefix}.c_fc", weights=weights, bias=True
        )
        self.c_proj = load_row(
            config, prefix=f"{prefix}.c_proj", weights=weights, bias=True
        )

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        return hidden_states


class Block(nn.Module):
    def __init__(self, layer_id, config, weights):
        super().__init__()
        prefix = f"transformer.h.{layer_id}"
        self.ln_1 = FastLayerNorm.load(
            prefix=f"{prefix}.ln_1", weights=weights, eps=config.layer_norm_epsilon
        )
        self.ln_2 = FastLayerNorm.load(
            prefix=f"{prefix}.ln_2", weights=weights, eps=config.layer_norm_epsilon
        )
        self.attn = FlashMQAttention(
            prefix=f"{prefix}.attn",
            config=config,
            weights=weights,
        )
        self.mlp = MLP(
            prefix=f"{prefix}.mlp",
            config=config,
            weights=weights,
        )

    def forward(
        self,
        hidden_states,
        residual,
        start_seq,
        end_seq,
        start_seq_q,
        end_seq_q,
        max_s,
        layer_past,
        past_present_indices,
        prefill,
    ):
        hidden_states, residual = self.ln_1(hidden_states, residual)

        hidden_states = self.attn(
            hidden_states,
            start_seq,
            end_seq,
            start_seq_q,
            end_seq_q,
            max_s,
            layer_past,
            past_present_indices,
            prefill,
        )

        hidden_states, residual = self.ln_2(hidden_states, residual)

        mlp_output = self.mlp(hidden_states)

        return mlp_output, residual


class FlashSantacoderModel(nn.Module):
    def __init__(self, config, weights):
        super().__init__()
        self.config = config

        self.process_group = weights.process_group
        self.wte = TensorParallelEmbedding(
            prefix="transformer.wte",
            weights=weights,
            reduce=False,
        )
        self.wpe = TensorParallelEmbedding(
            prefix="transformer.wpe",
            weights=weights,
            reduce=False,
        )

        self.h = nn.ModuleList(
            [
                Block(
                    layer_id,
                    config,
                    weights,
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.ln_f = FastLayerNorm.load(
            prefix="transformer.ln_f", weights=weights, eps=config.layer_norm_epsilon
        )

        self.head_size = self.h[0].attn.head_size
        self.num_heads = self.h[0].attn.num_heads

    def forward(
        self,
        input_ids,
        position_ids,
        start_seq,
        end_seq,
        start_seq_q,
        end_seq_q,
        max_s,
        past_present_indices,
        past_key_values=None,
        pre_allocate_past_size: Optional[int] = None,
    ):
        hidden_states = self.wte(input_ids) + self.wpe(position_ids)

        if self.process_group.size() > 1:
            torch.distributed.all_reduce(hidden_states, group=self.process_group)

        # Prefill
        if past_key_values is None:
            assert pre_allocate_past_size is not None

            prefill = True

            # Create past tensor
            # We create a tensor of the same size as input_ids as we don't want to slice at every layer
            past_key_values = hidden_states.new_zeros(
                (len(input_ids), len(self.h), 2, 1, self.head_size)
            )
        # Decode
        else:
            prefill = False

        residual = None
        for i, layer in enumerate(self.h):
            hidden_states, residual = layer(
                hidden_states,
                residual,
                start_seq,
                end_seq,
                start_seq_q,
                end_seq_q,
                max_s,
                torch.select(past_key_values, dim=1, index=i),
                past_present_indices,
                prefill,
            )

        if prefill:
            present = past_key_values
            # Create padded past tensor
            past_key_values = hidden_states.new_empty(
                (pre_allocate_past_size, len(self.h), 2, 1, self.head_size)
            )
            # We slice only once instead of at every layer
            past_key_values[past_present_indices] = present

        hidden_states, _ = self.ln_f(hidden_states, residual)

        return hidden_states, past_key_values


class FlashSantacoderForCausalLM(nn.Module):
    def __init__(self, config, weights):
        super().__init__()
        self.transformer = FlashSantacoderModel(config, weights)
        self.lm_head = TensorParallelHead.load(
            config, prefix="transformer.wte", weights=weights
        )

    def forward(
        self,
        input_ids,
        position_ids,
        start_seq,
        end_seq,
        start_seq_q,
        end_seq_q,
        max_s,
        past_present_indices,
        past_key_values: Optional[torch.Tensor] = None,
        pre_allocate_past_size: Optional[int] = None,
        lm_head_indices: Optional[torch.Tensor] = None,
    ):
        hidden_states, present = self.transformer(
            input_ids,
            position_ids,
            start_seq,
            end_seq,
            start_seq_q,
            end_seq_q,
            max_s,
            past_present_indices,
            past_key_values,
            pre_allocate_past_size,
        )
        if lm_head_indices is not None:
            hidden_states = hidden_states[lm_head_indices]
        logits = self.lm_head(hidden_states)
        return logits, present
