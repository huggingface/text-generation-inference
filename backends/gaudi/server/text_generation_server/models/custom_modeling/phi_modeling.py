# imlementation of the PhiModel and PhiForCausalLM classes

import torch
import torch.distributed

import math
from torch import nn
from typing import Optional, List, Tuple
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from text_generation_server.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    SpeculativeHead,
    FastLinear,
)


# PhiConfig is the configuration class for the PhiModel.
class PhiConfig(PretrainedConfig):
    def __init__(
        self,
        vocab_size=51200,
        n_positions=2048,
        n_embd=2560,
        n_layer=32,
        n_inner=None,
        n_head=32,
        rotary_dim=32,
        layer_norm_epsilon=1e-5,
        tie_word_embeddings=False,
        pad_vocab_size_multiple=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        no_bias=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_inner = n_inner
        self.n_head = n_head
        self.rotary_dim = rotary_dim

        self.layer_norm_epsilon = layer_norm_epsilon
        self.tie_word_embeddings = tie_word_embeddings
        self.pad_vocab_size_multiple = pad_vocab_size_multiple
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.no_bias = no_bias

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


# RotaryEmbedding is a class that implements the rotary embedding.
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        inv_freq = [1.0 / 10000.0 ** (i / dim) for i in range(0, dim, 2)]
        inv_freq_len = len(inv_freq)
        inv_freq = torch.tensor(inv_freq).view(1, inv_freq_len)
        t = torch.arange(0, max_seq_len, dtype=torch.float).view(max_seq_len, 1)
        freqs = t.matmul(inv_freq)
        self.sin = freqs.sin()
        self.cos = freqs.cos()

    def apply_rotary_emb_qkv(self, qkv, seqlen_offset):
        b_size, seqlen, three, _, _headdim = qkv.shape
        if three != 3:
            raise Exception("unexpected shape for qkv")
        _, rotary_dim = self.cos.shape
        rotary_dim = rotary_dim * 2
        q_rot = qkv[:, :, 0, :, :rotary_dim]
        q_pass = qkv[:, :, 0, :, rotary_dim:]
        k_rot = qkv[:, :, 1, :, :rotary_dim]
        k_pass = qkv[:, :, 1, :, rotary_dim:]
        q12 = torch.chunk(q_rot, 2, dim=-1)
        k12 = torch.chunk(k_rot, 2, dim=-1)
        q1, q2 = q12[0], q12[1]
        k1, k2 = k12[0], k12[1]
        c = self.cos.narrow(0, seqlen_offset, seqlen).unsqueeze(1)
        s = self.sin.narrow(0, seqlen_offset, seqlen).unsqueeze(1)
        q_rot = torch.cat(
            [
                q1 * c - q2 * s,
                q1 * s + q2 * c,
            ],
            dim=-1,
        )
        k_rot = torch.cat(
            [
                k1 * c - k2 * s,
                k1 * s + k2 * c,
            ],
            dim=-1,
        )
        q = torch.cat([q_rot, q_pass], dim=-1)
        k = torch.cat([k_rot, k_pass], dim=-1)
        v = qkv[:, :, 2]
        return q, k, v


# PhiCausalLMHead is the head of the PhiModel. It is a linear layer with a layer norm.
class PhiCausalLMHead(nn.Module):
    def __init__(self, config, weights):
        super().__init__()
        self.ln = nn.LayerNorm.load(
            prefix="lm_head.ln",
            weights=weights,
            eps=config.layer_norm_epsilon,
        )
        self.linear = SpeculativeHead.load(
            config=config, prefix="lm_head.linear", weights=weights
        )

    def forward(self, hidden_states):
        hidden_states = self.ln(hidden_states)
        hidden_states = self.linear(hidden_states)
        return hidden_states


# PhiMHA is a multi-head attention layer. This layer uses an attention mask to prevent tokens from attending to subsequent tokens.
class PhiMHA(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        self.Wqkv = TensorParallelColumnLinear.load(
            config, prefix=f"{prefix}.Wqkv", weights=weights, bias=not config.no_bias
        )
        self.out_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.out_proj",
            weights=weights,
            bias=not config.no_bias,
        )
        self.op_size = config.n_embd
        self.head_dim = int(config.n_embd / config.n_head)
        self.num_heads = config.n_head
        self.rotary_emb = RotaryEmbedding(
            config.rotary_dim,
            config.n_positions,
        )
        self.softmax_scale = 1.0 / math.sqrt(self.head_dim)

    def forward(
        self,
        hidden_states,
        past_kv_cache,
        attention_mask=None,
    ):
        b_size, seq_len, _n_embd = hidden_states.shape
        qkv = self.Wqkv(hidden_states)
        qkv = qkv.view(b_size, seq_len, 3, self.num_heads, self.head_dim)
        seqlen_offset = 0 if past_kv_cache is None else past_kv_cache[0].shape[1]
        q, k, v = self.rotary_emb.apply_rotary_emb_qkv(qkv, seqlen_offset)

        # if there is a kv_cache, then we need to concatenate
        if past_kv_cache is not None:
            prev_k, prev_v = past_kv_cache
            k = torch.cat([prev_k, k], dim=1)
            v = torch.cat([prev_v, v], dim=1)

        past_kv_cache = [k, v]
        attn_weights = torch.einsum("bthd,bshd->bhts", q, k * self.softmax_scale)

        if attention_mask is not None:
            seqlen_k = k.shape[1]
            seqlen_q = q.shape[1]
            causal_mask = torch.triu(
                torch.full((seqlen_q, seqlen_k), -10000.0, device=attn_weights.device),
                1,
            )
            attn_weights = attn_weights + causal_mask.to(dtype=attn_weights.dtype)

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        attn_output = attn_weights.matmul(v.transpose(1, 2)).squeeze(0)
        attn_output = (
            attn_output.view((b_size, self.num_heads, seq_len, self.head_dim))
            .transpose(1, 2)
            .flatten(-2)
        )
        return self.out_proj(attn_output), past_kv_cache


# PhiMLP is a multi-layer perceptron. It contains two linear layers with a gelu activation function.
class PhiMLP(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()

        self.n_inner = config.n_inner
        self.fc1 = FastLinear.load(
            config=config,
            prefix=f"{prefix}.fc1",
            weights=weights,
            bias=False,
        )
        self.fc2 = FastLinear.load(
            config=config,
            prefix=f"{prefix}.fc2",
            weights=weights,
            bias=False,
        )
        self.activation = torch.nn.functional.gelu

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


# PhiBlock is a single transformer block. It contains a layer norm, a multi-head attention layer and an multi-layer perceptron.
class PhiBlock(nn.Module):
    def __init__(self, layer_id, config, weights):
        super().__init__()
        self.layer_id = layer_id
        self.layer_norm = nn.LayerNorm.load(
            prefix=f"{layer_id}.ln", weights=weights, eps=config.layer_norm_epsilon
        )
        self.mixer = PhiMHA(prefix=f"{layer_id}.mixer", config=config, weights=weights)
        self.mlp = PhiMLP(prefix=f"{layer_id}.mlp", config=config, weights=weights)

    def forward(
        self,
        hidden_states,
        kv_cache,
        attention_mask,
    ):
        residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        attn_outputs, past_kv_cache = self.mixer(
            hidden_states, kv_cache, attention_mask
        )
        feed_forward_hidden_states = self.mlp(hidden_states)
        out = attn_outputs + feed_forward_hidden_states + residual
        return out, past_kv_cache


# PhiModel implements the embedding layer and the transformer blocks.
class PhiModel(nn.Module):
    def __init__(self, prefix: str, config, weights):
        super().__init__()
        self.tp_rank = weights.process_group.rank()
        self.tp_world_size = weights.process_group.size()
        self.embed_tokens = TensorParallelEmbedding(
            prefix=f"{prefix}.embd.wte", weights=weights
        )
        self.blocks = nn.ModuleList(
            [
                PhiBlock(f"{prefix}.h.{layer_id}", config, weights)
                for layer_id in range(config.n_layer)
            ]
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[List[Tuple[torch.FloatTensor]]] = None,
        attention_mask: Optional[torch.ByteTensor] = None,
        return_dict: Optional[bool] = None,
        use_cache: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        hidden_states = self.embed_tokens(input_ids)
        seq_len = hidden_states.shape[1]
        mask = None if seq_len <= 1 else attention_mask

        past_key_values = (
            [None] * len(self.blocks) if past_key_values is None else past_key_values
        )

        for index, block in enumerate(self.blocks):
            hidden_states, new_key_values = block(
                hidden_states, past_key_values[index], mask
            )
            past_key_values[index] = new_key_values

        return hidden_states, past_key_values


# PhiForCausalLM wraps the PhiModel and PhiCausalLMHead together and returns a CausalLMOutputWithPast object.
class PhiForCausalLM(torch.nn.Module):
    def __init__(self, prefix: str, config, weights):
        super().__init__()

        if not prefix:
            prefix = "transformer"
        else:
            prefix = f"{prefix}.transformer"

        self.model = PhiModel(prefix, config, weights)
        self.lm_head = PhiCausalLMHead(config, weights)

    def forward(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[List[Tuple[torch.FloatTensor]]] = None,
        attention_mask: Optional[torch.ByteTensor] = None,
        return_dict: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        model_output = self.model(
            input_ids, past_key_values, attention_mask, return_dict, use_cache
        )
        logits = self.lm_head(model_output[0])

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(
                logits[:, :-1].view(-1, logits.size(-1)), labels[:, 1:].view(-1)
            )

        if not return_dict:
            return (
                ((loss,) + (logits,) + model_output[1:])
                if loss is not None
                else (logits,) + model_output[1:]
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=model_output[1],
            hidden_states=None,
            attentions=None,
        )
