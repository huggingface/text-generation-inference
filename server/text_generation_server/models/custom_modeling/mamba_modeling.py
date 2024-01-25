import torch
import torch.distributed

from torch import nn
from typing import Optional, List, Tuple, Any
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch.nn.functional as F

from text_generation_server.utils.layers import (
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
)


class MambaConfig(PretrainedConfig):
    def __init__(
        self,
        vocab_size=50280,
        n_layer=32,
        layer_norm_epsilon=1e-5,
        tie_word_embeddings=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.layer_norm_epsilon = layer_norm_epsilon

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class MambaBlock(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()

        # TODO: use model config to set the dt_rank instead of hardcoding it
        d_inner = 768 * 2
        d_conv = 4
        self.dt_rank = (768 + 15) // 16

        # TODO: improve how we load the conv1d weights
        # explore a transposed conv1d that avoids the need for
        # a transpose during inference
        self.conv1 = nn.Conv1d(
            d_inner,
            d_inner,
            kernel_size=d_conv,
            groups=d_inner,
            padding=d_conv - 1,
        )
        self.conv1.weight = nn.Parameter(weights.get_tensor(f"{prefix}.conv1d.weight"))
        self.conv1.bias = nn.Parameter(weights.get_tensor(f"{prefix}.conv1d.bias"))

        self.dt_proj = TensorParallelColumnLinear.load(
            config=config,
            prefix=f"{prefix}.dt_proj",
            weights=weights,
            bias=True,
        )
        self.in_proj = TensorParallelColumnLinear.load(
            config=config,
            prefix=f"{prefix}.in_proj",
            weights=weights,
            bias=False,
        )
        self.x_proj = TensorParallelColumnLinear.load(
            config=config,
            prefix=f"{prefix}.x_proj",
            weights=weights,
            bias=False,
        )
        self.out_proj = TensorParallelColumnLinear.load(
            config=config,
            prefix=f"{prefix}.out_proj",
            weights=weights,
            bias=False,
        )

        # TODO: improve how we load the weights
        self.A_log = nn.Parameter(weights.get_tensor(f"{prefix}.A_log"))
        self.D = nn.Parameter(weights.get_tensor(f"{prefix}.D"))

    def selective_scan(
        self, input_tensor, delta, a_tensor, b_tensor, c_tensor, d_tensor
    ):
        batch_size, sequence_length, input_dim = input_tensor.shape
        num_cols = a_tensor.shape[1]

        # TODO: revisit this math to avoid the transposes when possible
        # reshape and process delta
        delta = delta.transpose(1, 2).view((batch_size, input_dim, sequence_length, 1))
        exp_delta_a = (delta * a_tensor.view((1, input_dim, 1, num_cols))).exp()

        # calc involving delta, b_tensor, and input_tensor
        delta_b_input = (
            delta
            * b_tensor.view((batch_size, 1, sequence_length, num_cols))
            * input_tensor.transpose(1, 2).view(
                (batch_size, input_dim, sequence_length, 1)
            )
        )

        # init output tensor
        output_tensor = torch.zeros(
            (batch_size, input_dim, num_cols),
            dtype=exp_delta_a.dtype,
            device=exp_delta_a.device,
        )

        # iterate over sequence_length
        output_sequence = []
        for i in range(sequence_length):
            multiplier = exp_delta_a[:, :, i]
            output_tensor = (multiplier * output_tensor) + delta_b_input[:, :, i]
            y = output_tensor.matmul(c_tensor[:, i, :].unsqueeze(2)).squeeze(2)
            output_sequence.append(y)

        stacked_output = torch.stack(output_sequence, 1)
        return stacked_output + input_tensor * d_tensor

    def ssm(self, hidden_states):
        _input_dim, num_cols = self.A_log.shape
        negative_exponential_a = self.A_log.exp().neg()
        d_matrix = self.D
        projected_hidden_states = self.x_proj(hidden_states)

        # narrow operations for delta, b, and c
        delta = projected_hidden_states.narrow(-1, 0, self.dt_rank)
        b_matrix = projected_hidden_states.narrow(-1, self.dt_rank, num_cols)
        c_matrix = projected_hidden_states.narrow(-1, self.dt_rank + num_cols, num_cols)

        # process delta
        delta = self.dt_proj(delta)
        delta = torch.log(torch.exp(delta) + 1)

        # apply selective scan
        selective_scan_output = self.selective_scan(
            hidden_states, delta, negative_exponential_a, b_matrix, c_matrix, d_matrix
        )
        return selective_scan_output

    def forward(self, hidden_states):
        sequence_length = hidden_states.shape[1]
        projected_states = self.in_proj(hidden_states)
        split_states = torch.chunk(projected_states, 2, dim=-1)
        transformed_states, residual_states = split_states

        # TODO: avoid the transpose by using a transposed conv1d
        # apply convolution and narrowing operation
        conv_output = (
            self.conv1(transformed_states.transpose(1, 2))
            .narrow(-1, 0, sequence_length)
            .transpose(1, 2)
        )

        # apply silu (Swish) activation function
        activated_transformed = F.silu(conv_output)
        activated_residual = F.silu(residual_states)

        # Subsequent operations
        output = self.ssm(activated_transformed)
        combined_output = output * activated_residual

        return self.out_proj(combined_output)

# TODO: prefer a more optimized implementation of RMSNorm if possible
class RMSNorm(nn.Module):
    def __init__(self, num_features, eps=1e-8):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(num_features))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x = x / rms
        return self.scale * x


class ResidualBlock(nn.Module):
    def __init__(self, layer_id, config, weights):
        super().__init__()
        self.layer_id = layer_id
        self.mamba_block = MambaBlock(
            prefix=f"{layer_id}.mixer", config=config, weights=weights
        )
        self.layer_norm = RMSNorm(768, eps=config.layer_norm_epsilon)
        self.layer_norm.scale = nn.Parameter(
            weights.get_tensor(f"{layer_id}.norm.weight")
        )

    def forward(
        self,
        hidden_states,
    ):
        residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        attn_outputs = self.mamba_block(hidden_states)
        hidden_states = residual + attn_outputs
        return hidden_states


class MambaModel(nn.Module):
    def __init__(self, config, weights):
        super().__init__()
        self.tp_rank = weights.process_group.rank()
        self.tp_world_size = weights.process_group.size()
        self.embed_tokens = TensorParallelEmbedding(
            prefix="backbone.embedding", weights=weights
        )
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(f"backbone.layers.{layer_id}", config, weights)
                for layer_id in range(config.n_layer)
            ]
        )

        # TODO: avoid hardcoded sizes and improve how we load the weights
        self.norm_f = RMSNorm(768, eps=config.layer_norm_epsilon)
        self.norm_f.scale = nn.Parameter(weights.get_tensor(f"backbone.norm_f.weight"))
        # use the same weights for the embedding and the final layer norm
        self.lm_head = nn.Linear(768, config.vocab_size, bias=False)
        self.lm_head.weight = nn.Parameter(
            self.embed_tokens.weight[: config.vocab_size, :]
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        past_input_ids: Optional[List[Tuple[torch.FloatTensor]]] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        # TODO: dont use past_input_ids for the input_ids
        # find a way to cache previous states/work
        if past_input_ids is not None:
            # append the contents to the input_ids
            input_ids = torch.cat((past_input_ids, input_ids), dim=1)

        hidden_states = self.embed_tokens(input_ids)

        for _, block in enumerate(self.blocks):
            hidden_states = block(hidden_states)

        final_hidden_states = self.norm_f(hidden_states)
        after_lm_head = self.lm_head(final_hidden_states)
        return after_lm_head, input_ids


# TODO: revisit if we want to use CausalLM
class MambaForCausalLM(torch.nn.Module):
    def __init__(self, config, weights):
        super().__init__()
        self.model = MambaModel(config, weights)

    def forward(
        self,
        input_ids: torch.LongTensor,
        # TODO: dont abuse past_key_values for the input_ids
        past_key_values: Optional[List[Tuple[torch.FloatTensor]]] = None,
        # below are unused since this model is attention free
        attention_mask: Optional[torch.ByteTensor] = None,
        return_dict: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        model_output = self.model(
            input_ids,
            past_input_ids=past_key_values,
        )
        logits = model_output[0]
        past_hidden_states = model_output[1]
        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=past_hidden_states,
            hidden_states=None,
            attentions=None,
        )
