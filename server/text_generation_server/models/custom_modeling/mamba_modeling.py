import torch
import torch.distributed

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
from mamba_ssm.utils.generation import InferenceParams
from torch import nn
from typing import Optional, List, Tuple, Any, Dict
from transformers.configuration_utils import PretrainedConfig
import torch.nn.functional as F

from text_generation_server.utils.layers import (
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    FastRMSNorm,
    FastLinear,
)

from einops import rearrange, repeat
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
        self.out_proj = FastLinear.load(config, f"{prefix}.out_proj", weights, bias=False)
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
        seqlen = hidden_states.shape[1]

        # TODO: use the inference_params to get the previous states when decoding
        conv_state, ssm_state = None, None
        if inference_params is not None:
            if hidden_states.shape[1] == 1:
                print("Decoding")
                conv_state, ssm_state = self._get_states_from_cache(inference_params, hidden_states.shape[0])
                if inference_params.seqlen_offset > 0:
                    # The states are updated inplace
                    out, _conv_state, _ssm_state = self.step(hidden_states, conv_state, ssm_state)
                    # import ipdb; ipdb.set_trace()
                    return out, _conv_state, _ssm_state

        projected_states = self.in_proj(hidden_states).transpose(1,2)

        x, z = projected_states.chunk(2, dim=1)
        # Compute short convolution
        if conv_state is not None:
            # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
            # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
            conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
        if causal_conv1d_fn is None:
            x = self.act(self.conv1d(x)[..., :seqlen])
        else:
            assert self.activation in ["silu", "swish"]
            x = causal_conv1d_fn(
                x=x,
                weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation=self.activation,
            )

        # We're careful here about the layout, to avoid extra transposes.
        # We want dt to have d as the slowest moving dimension
        # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        assert self.activation in ["silu", "swish"]
        y, last_ssm_state = selective_scan_fn(
            x,
            dt,
            self.negA,
            B,
            C,
            self.D.float(),
            z=z,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=True, # ssm_state is not None,
        )
        y = rearrange(y, "b d l -> b l d")
        attn_outputs = self.out_proj(y)

        return attn_outputs, conv_state, last_ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
        return conv_state, ssm_state
    
    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)

        # SSM step
        # Discretize A and B
        dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
        dA = torch.exp(torch.einsum("bd,dn->bdn", dt, self.negA))
        dB = torch.einsum("bd,bn->bdn", dt, B)
        ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
        y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
        y = y + self.D.to(dtype) * x
        y = y * self.act(z)  # (B D)

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

class ResidualBlock(nn.Module):
    def __init__(self, layer_id, config, weights):
        super().__init__()
        self.mamba_block = MambaBlock(prefix=f"{layer_id}.mixer", config=config, weights=weights)
        self.layer_norm = FastRMSNorm.load(prefix=f"{layer_id}.norm", weights=weights, eps=config.layer_norm_epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,
        inference_params: Optional[Any] = None,
    ):  
        residual = hidden_states
        shape = hidden_states.shape
        hidden_states, _ = self.layer_norm(hidden_states.view(-1, shape[-1]))
        hidden_states, _conv_state, last_ssm_state = self.mamba_block(hidden_states.view(*shape), inference_params)
        hidden_states = residual + hidden_states
        return hidden_states, _conv_state, last_ssm_state

class MambaModel(nn.Module):
    def __init__(self, config, weights):
        super().__init__()
        prefix = "backbone"
        self.embed_tokens = TensorParallelEmbedding(f"{prefix}.embedding", weights)
        self.blocks = nn.ModuleList(
            [ResidualBlock(f"{prefix}.layers.{i}", config, weights) for i in range(config.n_layer)]
        )
        self.norm_f = FastRMSNorm.load(f"{prefix}.norm_f", weights, eps=config.layer_norm_epsilon)
        self.lm_head = FastLinear.load(config, f"{prefix}.embedding", weights, bias=False)
        self.config = config

    def forward(self, input_ids: torch.Tensor, inference_params=None):
        hidden_states = self.embed_tokens(input_ids)
        print("Input ids: ", input_ids)
        for block in self.blocks:
            hidden_states, _conv_state, last_ssm_state = block(hidden_states, inference_params)
            # inference_params.key_value_memory_dict[block.mamba_block.layer_idx] = (_conv_state, last_ssm_state)


        shape = hidden_states.shape
        final_hidden_states, _ = self.norm_f(hidden_states.view(-1, shape[-1]))
        return self.lm_head(final_hidden_states.view(*shape)), input_ids, inference_params
