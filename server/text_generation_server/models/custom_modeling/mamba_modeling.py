import torch
import torch.distributed

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
from torch import nn
from typing import Optional, List, Tuple, Any
from transformers.configuration_utils import PretrainedConfig
import torch.nn.functional as F

from text_generation_server.utils.layers import (
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    FastRMSNorm,
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
        self.in_proj = TensorParallelColumnLinear.load(
            config=config, prefix=f"{prefix}.in_proj", weights=weights, bias=False
        )
        # helper for loading weights
        self.load_weights(prefix, weights)

    def load_weights(self, prefix, weights):
        weight_names = ["x_proj.weight", "dt_proj.weight", "dt_proj.bias", 
                        "out_proj.weight", "in_proj.weight", 
                        "conv1d.weight", "conv1d.bias", "A_log", "D"]
        for name in weight_names:
            param_name = name.replace('.', '_')
            setattr(self, param_name, nn.Parameter(weights.get_tensor(f"{prefix}.{name}")))
        self.out_proj_bias = None
        self.negA = -torch.exp(self.A_log.float()) 

    def forward(self, hidden_states: torch.Tensor):
        projected_states = self.in_proj(hidden_states).transpose(1,2)
        # conv1d, ssm, and selective_scan are all fused into one kernel
        attn_outputs = mamba_inner_fn(
            projected_states,
            self.conv1d_weight,
            self.conv1d_bias,
            self.x_proj_weight,
            self.dt_proj_weight,
            self.out_proj_weight,
            self.out_proj_bias,
            self.negA,
            None,
            None,
            self.D.float(),
            delta_bias=self.dt_proj_bias.float(),
            delta_softplus=True,
        )
        return attn_outputs

class ResidualBlock(nn.Module):
    def __init__(self, layer_id, config, weights):
        super().__init__()
        self.mamba_block = MambaBlock(prefix=f"{layer_id}.mixer", config=config, weights=weights)
        self.layer_norm = FastRMSNorm.load(prefix=f"{layer_id}.norm", weights=weights, eps=config.layer_norm_epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ):
        residual = hidden_states
        hidden_states, _ = self.layer_norm(hidden_states.squeeze(0))
        hidden_states = residual + self.mamba_block(hidden_states.unsqueeze(0))
        return hidden_states

class MambaModel(nn.Module):
    def __init__(self, config, weights):
        super().__init__()
        self.tp_rank = weights.process_group.rank()
        self.tp_world_size = weights.process_group.size()
        prefix = "backbone"

        self.embed_tokens = TensorParallelEmbedding(f"{prefix}.embedding", weights)
        self.blocks = nn.ModuleList(
            [ResidualBlock(f"{prefix}.layers.{i}", config, weights) for i in range(config.n_layer)]
        )
        self.norm_f = FastRMSNorm.load(f"{prefix}.norm_f", weights, eps=config.layer_norm_epsilon)
        self.lm_head = TensorParallelColumnLinear.load(config, f"{prefix}.embedding", weights, False)

    def forward(self, input_ids: torch.Tensor):
        hidden_states = self.embed_tokens(input_ids)
        for block in self.blocks:
            hidden_states = block(hidden_states)

        final_hidden_states, _ = self.norm_f(hidden_states.squeeze(0))
        return self.lm_head(final_hidden_states.unsqueeze(0)), input_ids
