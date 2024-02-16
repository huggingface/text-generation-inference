import torch
import torch.distributed

from mamba_ssm.ops.triton.selective_state_update import selective_state_update
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from torch import nn
from typing import Optional, Tuple, Any
from transformers.configuration_utils import PretrainedConfig
import torch.nn.functional as F

from text_generation_server.utils.layers import (
    TensorParallelEmbedding,
    FastRMSNorm,
    FastLinear,
)

from einops import rearrange
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
import math
from dataclasses import dataclass


@dataclass
class InferenceParams:
    """Inference parameters that are passed to the main model in order
    to efficienly calculate and store the context during inference."""

    max_seqlen: int
    max_batch_size: int
    conv_states: torch.Tensor
    ssm_states: torch.Tensor
    seqlen_offset: int


class MambaConfig(PretrainedConfig):
    def __init__(
        self,
        vocab_size=50280,
        d_model=768,
        d_state=16,
        n_layer=32,
        layer_norm_epsilon=1e-5,
        tie_word_embeddings=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        expand=2,
        dt_rank="auto",
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.layer_norm_epsilon = layer_norm_epsilon
        self.d_model = d_model
        self.d_inner = d_model * 2
        self.d_conv = 4
        self.d_state = d_state
        self.expand = expand
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class MambaBlock(nn.Module):
    def __init__(self, prefix, config, weights, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.in_proj = FastLinear.load(config, f"{prefix}.in_proj", weights, bias=False)
        self.x_proj = FastLinear.load(config, f"{prefix}.x_proj", weights, bias=False)
        self.dt_proj = FastLinear.load(config, f"{prefix}.dt_proj", weights, bias=True)
        self.dt_proj_no_bias = FastLinear.load(
            config, f"{prefix}.dt_proj", weights, bias=False
        )
        self.out_proj = FastLinear.load(
            config, f"{prefix}.out_proj", weights, bias=False
        )
        self.conv1d = FastLinear.load(config, f"{prefix}.conv1d", weights, bias=True)
        self.negA = -torch.exp(weights.get_tensor(f"{prefix}.A_log").float())
        self.D = weights.get_tensor(f"{prefix}.D")
        self.activation = "silu"
        self.dt_rank = config.dt_rank
        self.d_state = config.d_state
        self.d_conv = config.d_conv
        self.act = nn.SiLU()

    # inference_params
    def forward(self, hidden_states: torch.Tensor, inference_params=None):
        if inference_params.seqlen_offset > 0:
            conv_state = inference_params.conv_states[self.layer_id]
            ssm_state = inference_params.ssm_states[self.layer_id]
            out, conv_state, ssm_state = self.step(hidden_states, conv_state, ssm_state)
            return out, conv_state, ssm_state

        _, seqlen, _ = hidden_states.shape
        projected_states = self.in_proj(hidden_states).transpose(1, 2)
        # assert projected_states.shape == [batch_size, 2 * dstate, seqlen], f"{projected_states.shape} [{batch_size}, {dstate}, {seqlen}]"
        x, z = projected_states.chunk(2, dim=1)
        conv_state = F.pad(x, (self.d_conv - seqlen, 0))
        x = causal_conv1d_fn(
            x=x,
            weight=self.conv1d.weight.squeeze(1),
            bias=self.conv1d.bias,
            activation=self.activation,
        )

        # We're careful here about the layout, to avoid extra transposes.
        # We want dt to have d as the slowest moving dimension
        # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
        dt, B, C = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        y, last_state = selective_scan_fn(
            x,
            dt,
            self.negA,
            B,
            C,
            self.D.float(),
            z=z,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=True,
        )
        y = rearrange(y, "b d l -> b l d")
        attn_outputs = self.out_proj(y)
        return attn_outputs, conv_state, last_state

    def step(self, hidden_states, conv_state, ssm_state):
        xz = self.in_proj(hidden_states.squeeze(1))
        x, z = xz.chunk(2, dim=-1)  # (B D)
        x = causal_conv1d_update(
            x,
            conv_state,
            self.conv1d.weight.squeeze(1),
            self.conv1d.bias,
            self.activation,
        )
        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.linear(dt, self.dt_proj.weight)
        A = self.negA
        y = selective_state_update(
            ssm_state,
            x,
            dt,
            A,
            B,
            C,
            self.D,
            z=z,
            dt_bias=self.dt_proj.bias,
            dt_softplus=True,
        )
        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state.clone(), ssm_state.clone()


class ResidualBlock(nn.Module):
    def __init__(self, prefix, config, weights, layer_id):
        super().__init__()
        self.mamba_block = MambaBlock(
            prefix=f"{prefix}.mixer", config=config, weights=weights, layer_id=layer_id
        )
        self.layer_norm = FastRMSNorm.load(
            prefix=f"{prefix}.norm", weights=weights, eps=config.layer_norm_epsilon
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        inference_params: Optional[Any] = None,
    ):
        residual = (hidden_states + residual) if residual is not None else hidden_states
        shape = residual.shape
        hidden_states, _ = self.layer_norm(residual.view(-1, shape[-1]))
        hidden_states, conv_state, last_ssm_state = self.mamba_block(
            hidden_states.view(*shape), inference_params
        )
        return hidden_states, residual, conv_state, last_ssm_state


class MambaModel(nn.Module):
    def __init__(self, config, weights):
        super().__init__()
        prefix = "backbone"
        self.embed_tokens = TensorParallelEmbedding(f"{prefix}.embedding", weights)
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(f"{prefix}.layers.{i}", config, weights, layer_id=i)
                for i in range(config.n_layer)
            ]
        )
        self.norm_f = FastRMSNorm.load(
            f"{prefix}.norm_f", weights, eps=config.layer_norm_epsilon
        )
        self.lm_head = FastLinear.load(
            config, f"{prefix}.embedding", weights, bias=False
        )
        self.config = config

    def forward(
        self, input_ids: torch.Tensor, inference_params=None, residual=None
    ) -> Tuple[torch.Tensor, torch.Tensor, InferenceParams]:
        hidden_states = self.embed_tokens(input_ids)
        for i, block in enumerate(self.blocks):
            hidden_states, residual, conv_state, ssm_state = block(
                hidden_states, residual, inference_params
            )
            inference_params.conv_states[i].copy_(conv_state)
            inference_params.ssm_states[i].copy_(ssm_state)

        hidden_states = (
            hidden_states + residual if residual is not None else hidden_states
        )
        hidden_states, _ = self.norm_f(hidden_states.view(-1, hidden_states.size(-1)))
        hidden_states = hidden_states.view(residual.shape)
        logits = self.lm_head(hidden_states)

        # update the offset for the next inference using these params
        inference_params.seqlen_offset += input_ids.size(1)
        return logits
