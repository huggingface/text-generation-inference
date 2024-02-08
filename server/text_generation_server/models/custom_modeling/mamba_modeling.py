import torch
import torch.distributed

from mamba_ssm.ops.triton.selective_state_update import selective_state_update
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from mamba_ssm.utils.generation import InferenceParams
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
    def __init__(self, prefix, config, weights):
        super().__init__()
        self.layer_idx = int(prefix.split(".")[2])
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
        _, seqlen, _ = hidden_states.shape
        conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]

        if inference_params.seqlen_offset > 0:
            out, conv_state, ssm_state = self.step(hidden_states, conv_state, ssm_state)
            return out, conv_state, ssm_state

        projected_states = self.in_proj(hidden_states).transpose(1, 2)
        x, z = projected_states.chunk(2, dim=1)
        conv_state = F.pad(x, (self.d_conv - seqlen, 0))
        x = causal_conv1d_fn(
            x=x,
            weight=self.conv1d.weight.view(
                self.conv1d.weight.size(0), self.conv1d.weight.size(2)
            ),
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
        _xz = self.in_proj(hidden_states)
        _x, _z = _xz.chunk(2, dim=-1)  # (B D)
        conv_state_new = torch.cat([conv_state, _x.transpose(1, 2)], dim=-1)
        conv_out = causal_conv1d_fn(
            x=conv_state_new,
            weight=self.conv1d.weight.view(
                self.conv1d.weight.size(0), self.conv1d.weight.size(2)
            ),
            bias=self.conv1d.bias,
            activation=self.activation,
        )
        conv_state = conv_state_new[:, :, 1:]
        bsz, seqlen, dim = hidden_states.shape
        output_tensor = torch.zeros(
            (bsz, seqlen, dim), device=hidden_states.device, dtype=hidden_states.dtype
        )
        for i in range(0, bsz):
            x = conv_out[i : i + 1, :, -1]
            z = _z[i : i + 1, -1, :]
            x_db = self.x_proj(x)
            dt, B, C = torch.split(
                x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1
            )
            dt = F.linear(dt, self.dt_proj.weight)
            y = selective_state_update(
                ssm_state[i : i + 1, :, :],
                x,
                dt,
                self.negA,
                B,
                C,
                self.D,
                z=z,
                dt_bias=self.dt_proj.bias,
                dt_softplus=True,
            )
            out = self.out_proj(y)
            output_tensor[i] = out

        return output_tensor, conv_state, ssm_state


class ResidualBlock(nn.Module):
    def __init__(self, layer_id, config, weights):
        super().__init__()
        self.mamba_block = MambaBlock(
            prefix=f"{layer_id}.mixer", config=config, weights=weights
        )
        self.layer_norm = FastRMSNorm.load(
            prefix=f"{layer_id}.norm", weights=weights, eps=config.layer_norm_epsilon
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
                ResidualBlock(f"{prefix}.layers.{i}", config, weights)
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
        for block in self.blocks:
            hidden_states, residual, conv_state, ssm_state = block(
                hidden_states, residual, inference_params
            )
            inference_params.key_value_memory_dict[block.mamba_block.layer_idx] = (
                conv_state,
                ssm_state,
            )

        hidden_states = (
            hidden_states + residual if residual is not None else hidden_states
        )
        hidden_states, _ = self.norm_f(hidden_states.view(-1, hidden_states.size(-1)))
        hidden_states = hidden_states.view(residual.shape)
        logits = self.lm_head(hidden_states)

        # update the offset for the next inference using these params
        inference_params.seqlen_offset += input_ids.size(1)
        return logits, input_ids, inference_params
