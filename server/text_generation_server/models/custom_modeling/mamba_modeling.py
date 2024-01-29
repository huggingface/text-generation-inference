import torch
import torch.distributed

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
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
        d_model=768,
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
        self.d_model = d_model
        self.d_inner = d_model * 2
        self.d_conv = 4

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
        self.dt_rank = (config.d_model + 15) // 16
        self.x_proj_weight = weights.get_tensor(f"{prefix}.x_proj.weight")
        self.dt_proj_weight = weights.get_tensor(f"{prefix}.dt_proj.weight")
        self.dt_proj_bias = weights.get_tensor(f"{prefix}.dt_proj.bias")
        self.out_proj_weight = weights.get_tensor(f"{prefix}.out_proj.weight")
        self.out_proj_bias = None
        # TODO: avoid loading the same weights twice
        self.in_proj_weight = weights.get_tensor(f"{prefix}.in_proj.weight")
        self.in_proj_bias = None
        self.in_proj = TensorParallelColumnLinear.load(
            config=config,
            prefix=f"{prefix}.in_proj",
            weights=weights,
            bias=False,
        )
        self.conv1d = nn.Conv1d(
            config.d_inner,
            config.d_inner,
            kernel_size=config.d_conv,
            groups=config.d_inner,
            padding=config.d_conv - 1,
        )
        self.conv1d.weight = nn.Parameter(weights.get_tensor(f"{prefix}.conv1d.weight"))
        self.conv1d.bias = nn.Parameter(weights.get_tensor(f"{prefix}.conv1d.bias"))
        self.A_log = nn.Parameter(weights.get_tensor(f"{prefix}.A_log"))
        self.D = nn.Parameter(weights.get_tensor(f"{prefix}.D"))

    def forward(self, index, hidden_states, past_transformed_state):
        projected_states = self.in_proj(hidden_states)

        A = -torch.exp(self.A_log.float()) 

        # conv1d, ssm, and selective_scan are all fused into one kernel
        attn_outputs = mamba_inner_fn(
            projected_states.transpose(1,2),
            self.conv1d.weight,
            self.conv1d.bias,
            self.x_proj_weight,
            self.dt_proj_weight,
            self.out_proj_weight,
            self.out_proj_bias,
            A,
            None,
            None,
            self.D.float(),
            delta_bias=self.dt_proj_bias.float(),
            delta_softplus=True,
        )
        
        return attn_outputs, projected_states


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
        self.layer_norm = RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.layer_norm.scale = nn.Parameter(
            weights.get_tensor(f"{layer_id}.norm.weight")
        )

    def forward(
        self,
        index,
        hidden_states,
        past_transformed_state,
    ):
        residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        attn_outputs, transformed_states = self.mamba_block(
            index, hidden_states, past_transformed_state
        )
        hidden_states = residual + attn_outputs
        return hidden_states, transformed_states


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
        self.norm_f = RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.norm_f.scale = nn.Parameter(weights.get_tensor(f"backbone.norm_f.weight"))
        # use the same weights for the embedding and the final layer norm
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = nn.Parameter(
            self.embed_tokens.weight[: config.vocab_size, :]
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        past_input_ids: Optional[List[Tuple[torch.FloatTensor]]] = None,
        past_transformed_states: Optional[List[Tuple[torch.FloatTensor]]] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        # NOTE: we need all input_ids to compute the correct embeddings
        if past_input_ids is not None:
            input_ids = past_input_ids

        hidden_states = self.embed_tokens(input_ids)

        past_transformed_states = (
            [None] * len(self.blocks)
            if past_transformed_states is None
            else past_transformed_states
        )

        for index, block in enumerate(self.blocks):
            hidden_states, transformed_states = block(
                index, hidden_states, past_transformed_states[index]
            )
            past_transformed_states[index] = transformed_states

        final_hidden_states = self.norm_f(hidden_states)
        after_lm_head = self.lm_head(final_hidden_states)
        return after_lm_head, input_ids, past_transformed_states
