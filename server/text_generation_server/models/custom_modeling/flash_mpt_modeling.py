"""A simple, flexible implementation of a GPT model.

Inspired by https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
"""
# import math
# import warnings
# from typing import List, Optional, Tuple, Union
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
# from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
# from .attention import attn_bias_shape, build_attn_bias
# from .blocks import MPTBlock
# from .custom_embedding import SharedEmbedding
# from .norm import NORM_CLASS_REGISTRY
# from .configuration_mpt import MPTConfig
# from .adapt_tokenizer import AutoTokenizerForMOD, adapt_tokenizer_for_denoising
# from .hf_prefixlm_converter import add_bidirectional_mask_if_missing, convert_hf_causal_lm_to_prefix_lm
# from .meta_init_context import init_empty_weights
# from .param_init_fns import MODEL_INIT_REGISTRY, generic_param_init_fn_
# try:
#     from .flash_attn_triton import flash_attn_func
# except:
#     pass

"""GPT Blocks used for the GPT Model."""
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import math

from text_generation_server.utils.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    PositionRotaryEmbedding,
    TensorParallelHead,
    FastLayerNorm,
)

EPS = 1e-5

def _gen_slopes(n_heads, alibi_bias_max=8, device=None):
    _n_heads = 2 ** math.ceil(math.log2(n_heads))
    m = torch.arange(1, _n_heads + 1, dtype=torch.float32, device=device)
    m = m.mul(alibi_bias_max / _n_heads)
    slopes = 1.0 / torch.pow(2, m)
    if _n_heads != n_heads:
        slopes = torch.concat([slopes[1::2], slopes[::2]])[:n_heads]
    return slopes.view(1, n_heads, 1, 1)

def _build_alibi_bias(n_heads, seq_len, device, dtype, alibi_bias_max):
    alibi_bias = torch.arange(1 - seq_len, 1, dtype=torch.int32, device=device).view(1, 1, 1, seq_len)
    slopes = _gen_slopes(n_heads, alibi_bias_max, device=device)
    alibi_bias = alibi_bias * slopes
    return alibi_bias.to(dtype=dtype)

ALIBI = None

def build_alibi_bias(n_heads, seq_len, device, dtype, alibi_bias_max=8):
    global ALIBI
    if ALIBI is None or seq_len > ALIBI.shape[-1]:
        ALIBI = _build_alibi_bias(n_heads, seq_len, device, dtype, alibi_bias_max=alibi_bias_max)
    return ALIBI[:, :, :, :seq_len]


class MPTAttention(nn.Module):
    def __init__(self, config, prefix, weights):
        super().__init__()

        self.num_heads = config.n_heads
        self.hidden_size = config.d_model
        self.head_size = self.hidden_size // self.num_heads
        self.Wqkv = TensorParallelColumnLinear.load(
            config,
            prefix=f"{prefix}.Wqkv",
            weights=weights,
            bias=False,
        )
        self.out_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.out_proj",
            weights=weights,
            bias=False,
        )

    def forward(self, 
            hidden_states,
            alibi,
            start_seq,
            end_seq,
            start_seq_q,
            end_seq_q,
            max_s,
            past_key_values,
            past_present_indices,
            prefill,
        ):
        qkv = self.Wqkv(hidden_states)
        qkv = qkv.view(-1, 3, self.num_heads, self.head_size)

        # Todo
        raise Exception("Apply alibi ?");

        # Prefill
        if prefill:
            # Copy to layer past
            layer_past[...] = qkv[:, 1:]

            # output
            attn_output = torch.empty_like(qkv[:, 0])
            # flash attention
            flash_attn_cuda.fwd(
                qkv[:, 0],
                qkv[:, 1],
                qkv[:, 2],
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
            query = qkv[:, 0]
            # Add present to the layer_past tensor at the correct indices
            layer_past[past_present_indices] = qkv[:, 1:]

            # output
            attn_output = torch.empty_like(query)
            # flash attention
            flash_attn_cuda.fwd(
                query,
                layer_past[:, 0],
                layer_past[:, 1],
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

        return self.o_proj(attn_output.view(-1, self.num_heads * self.head_size))

class MPTMLP(nn.Module):
    def __init__(self, config, prefix, weights):
        super().__init__()

        self.up_proj = TensorParallelColumnLinear.load(
            config,
            prefix=f"{prefix}.up_proj",
            weights=weights,
            bias=False,
        )
        self.act = nn.GELU(approximate='none')
        self.down_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.down_proj",
            weights=weights,
            bias=False,
        )

    def forward(self, x):
        return self.down_proj(self.act(self.up_proj(x)))

class MPTBlock(nn.Module):
    def __init__(self, config, prefix, weights):
        super().__init__()
        self.norm_1 = FastLayerNorm.load_no_bias(prefix=f"{prefix}.norm_1", weights=weights, eps=EPS)
        self.attn = MPTAttention(config, prefix=f"{prefix}.attn", weights=weights)
        self.norm_2 = FastLayerNorm.load_no_bias(prefix=f"{prefix}.norm_2", weights=weights, eps=EPS)
        self.ffn = MPTMLP(config, prefix=f"{prefix}.ffn", weights=weights)

    def forward(self,
                hidden_states,
                residual,
                alibi,
                start_seq,
                end_seq,
                start_seq_q,
                end_seq_q,
                max_s,
                past_key_values,
                past_present_indices,
                prefill,
            ):
        residual = hidden_states
        hidden_states, _ = self.norm_1(hidden_states)
        # (hidden_states, attn_weights) = self.attn(
        hidden_states = self.attn(
            hidden_states,
            alibi,
            start_seq,
            end_seq,
            start_seq_q,
            end_seq_q,
            max_s,
            past_key_values,
            past_present_indices,
            prefill,
        )
        hidden_states += residual
        residual = hidden_states
        hidden_states, _ = self.norm_2(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states += residual
        return (x, attn_weights)

class MPTModel(nn.Module):
    def __init__(self, config, weights):
        super().__init__()
        self.wte = TensorParallelEmbedding(
            prefix="transformer.wte", weights=weights
        )
        self.num_heads = config.n_heads
        self.hidden_size = config.d_model
        self.head_size = self.hidden_size // self.num_heads
        self.blocks = nn.ModuleList([MPTBlock(config, prefix=f"transformer.blocks.{i}", weights=weights) for i in range(config.n_layers)])
        self.norm_f = FastLayerNorm.load_no_bias(
            prefix="transformer.norm_f", weights=weights, eps=EPS
        )

        # Create a default sizeable global alibi
        build_alibi_bias(n_heads=self.num_heads, seq_len=1024,device=weights.device, dtype = weights.dtype)

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
        hidden_states = self.wte(input_ids)



        # Prefill
        if past_key_values is None:
            assert pre_allocate_past_size is not None

            prefill = True

            # Create past tensor
            # We create a tensor of the same size as input_ids as we don't want to slice at every layer
            past_key_values = hidden_states.new_empty(
                (
                    len(input_ids),
                    len(self.blocks),
                    2,
                    self.num_heads,
                    self.head_size,
                )
            )
        # Decode
        else:
            prefill = False

        alibi = build_alibi_bias(n_heads=self.num_heads, seq_len=max_s,device=hidden_states.device, dtype = hidden_states.dtype)
        # Cast alibi into correct shape
        alibi = alibi[:, :, :, position_ids]

        residual = None
        for i, layer in enumerate(self.blocks):
            hidden_states, residual = layer(
                hidden_states,
                residual,
                alibi,
                start_seq,
                end_seq,
                start_seq_q,
                end_seq_q,
                max_s,
                past_key_values[:, i],
                past_present_indices,
                prefill,
            )

        if prefill:
            present = past_key_values
            # Create padded past tensor
            past_key_values = hidden_states.new_empty(
                (
                    pre_allocate_past_size,
                    len(self.blocks),
                    2,
                    self.num_heads,
                    self.head_size,
                )
            )
            # We slice only once instead of at every layer
            past_key_values[past_present_indices] = present

        hidden_states, _ = self.norm_f(hidden_states, residual)

        return hidden_states, past_key_values

class MPTForCausalLM(nn.Module):
    def __init__(self, config, weights):
        super().__init__()
        self.transformer = MPTModel(config, weights)
        self.lm_head = TensorParallelHead.load(
            config,
            prefix="transformer.wte",
            weights=weights,
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
