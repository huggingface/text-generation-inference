import torch
import torch.distributed

from torch import nn
from transformers.activations import ACT2FN

# Flash attention imports
import flash_attn_cuda
import dropout_layer_norm


class FastLayerNorm(nn.LayerNorm):
    def forward(self, hidden_states, residual=None):
        if hidden_states.shape[-1] > 6144:
            if residual is not None:
                hidden_states += residual
            residual = hidden_states

            return super(FastLayerNorm, self).forward(hidden_states), residual
        else:
            (
                normed_hidden_states,
                residual,
                *rest,
            ) = dropout_layer_norm.dropout_add_ln_fwd(
                hidden_states,
                residual,
                self.weight,
                self.bias,
                None,
                None,
                None,
                None,
                0.0,
                self.eps,
                1.0,
                0,
                None,
                False,
                False,
            )
            if residual is None:
                residual = hidden_states

            return normed_hidden_states, residual


class FastLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super(FastLinear, self).__init__(in_features, out_features, bias, device, dtype)

    def transpose_weight(self):
        self.weight = nn.Parameter(self.weight.T)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.bias is not None:
            return torch.addmm(self.bias, input, self.weight)
        return torch.matmul(input, self.weight)


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
            self.attn = FastLinear(hidden_size, hidden_size + 2 * self.head_size)
            self.c_proj = FastLinear(hidden_size, hidden_size)
        else:
            raise NotImplementedError

    def forward(
        self,
        hidden_states,
        cu_seqlens,
        max_s,
        layer_past,
        layer_past_present_indices,
        cu_seqlens_q,
    ):
        qkv = self.attn(hidden_states)

        # Split query from key_value
        query, key_value = qkv.split([self.hidden_size, 2 * self.head_size], dim=1)

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
                else None,
            )
        )

        if process_group is None:
            self.c_fc = FastLinear(hidden_size, intermediate_size)
            self.c_proj = FastLinear(intermediate_size, hidden_size)
        else:
            raise NotImplementedError

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

        if process_group is not None:
            raise NotImplementedError

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

    def post_load_weights(self):
        for layer in self.h:
            layer: Block
            layer.attn.attn.transpose_weight()
            layer.attn.c_proj.transpose_weight()
            layer.mlp.c_fc.transpose_weight()
            layer.mlp.c_proj.transpose_weight()

    def forward(
        self,
        input_ids,
        position_ids,
        cu_seqlens,
        max_s,
        past_key_values=None,
    ):
        hidden_states = self.wte(input_ids) + self.wpe(position_ids)

        # Prefill
        if past_key_values is None:
            # Create past tensor
            past_key_values = hidden_states.new_empty(
                (
                    len(self.h),
                    len(hidden_states),
                    2,
                    1,
                    self.head_size,
                )
            )
            layer_past_present_indices = None
            cu_seqlens_q = None
        # Decode
        else:
            # Create indices from cumulative sequence lengths
            layer_past_present_indices = cu_seqlens[1:] - 1
            cu_seqlens_q = torch.arange(
                cu_seqlens.shape[0], dtype=torch.int32, device=hidden_states.device
            )

        residual = None
        for i, layer in enumerate(self.h):
            hidden_states, residual = layer(
                hidden_states,
                residual,
                cu_seqlens,
                max_s,
                past_key_values[i],
                layer_past_present_indices,
                cu_seqlens_q,
            )

        hidden_states, _ = self.ln_f(hidden_states, residual)

        return hidden_states, past_key_values


class FlashSantacoderForCausalLM(nn.Module):
    def __init__(self, config, process_group=None):
        super().__init__()

        self.transformer = FlashSantacoderModel(config, process_group)

        self.lm_head = FastLinear(config.hidden_size, config.vocab_size, bias=False)

    def post_load_weights(self):
        self.transformer.post_load_weights()
        self.lm_head.transpose_weight()

    def forward(
        self,
        input_ids,
        position_ids,
        cu_seqlens,
        max_s,
        past_key_values=None,
    ):
        hidden_states, present = self.transformer(
            input_ids, position_ids, cu_seqlens, max_s, past_key_values
        )
        return self.lm_head(hidden_states), present
