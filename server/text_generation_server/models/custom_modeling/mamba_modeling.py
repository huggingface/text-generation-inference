import torch
import torch.distributed

import math
from torch import nn
from typing import Optional, List, Tuple, Any
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from text_generation_server.utils.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    TensorParallelHead,
    FastLinear,
    FastRMSNorm,
)

class MambaConfig(PretrainedConfig):
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
        rms_norm_eps=1e-8,
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
        self.rms_norm_eps = rms_norm_eps

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
        # TODO: adjust how weights are loaded

        # conv1d 768*2, 768*2, 4
        self.conv1 = nn.Conv1d(768, 768, 4)
        # add weight and bias to conv1
        self.conv1.weight = nn.Parameter(weights.get_tensor(f"{prefix}.conv1d.weight").transpose(0, 1))
        self.conv1.bias = nn.Parameter(weights.get_tensor(f"{prefix}.conv1d.bias"))

        # TODO: load weights in correctly for other operations
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
        self.A_log = nn.Parameter(torch.randn(config.n_head, config.n_head, config.rotary_dim))
        self.D = nn.Parameter(torch.randn(config.n_head, config.rotary_dim))

    def forward(
        self,
        hidden_states,
        past_kv_cache,
        attention_mask=None,
    ):
        hidden_states_in_proj = self.in_proj(hidden_states)
        hidden_states_and_residual = torch.chunk(hidden_states_in_proj, 2, dim=-1)

        hs, res = hidden_states_and_residual[0], hidden_states_and_residual[1]

        import ipdb; ipdb.set_trace()

class ResidualBlock(nn.Module):
    def __init__(self, layer_id, config, weights):
        super().__init__()
        self.layer_id = layer_id
        self.mixer = MambaBlock(prefix=f"{layer_id}.mixer", config=config, weights=weights)
        self.layer_norm = FastLinear.load(
            config=config,
            prefix=f"{layer_id}.norm", 
            weights=weights,
            bias=False,
        )

    def forward(
        self,
        hidden_states,
        kv_cache,
        attention_mask,
    ):  
        residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        attn_outputs, past_kv_cache = self.mixer(hidden_states, kv_cache, attention_mask)
        hidden_states = residual + attn_outputs
        return hidden_states, residual

class MambaModel(nn.Module):
    def __init__(self, config, weights):
        super().__init__()
        self.tp_rank = weights.process_group.rank()
        self.tp_world_size = weights.process_group.size()
        self.embed_tokens = TensorParallelEmbedding(
            prefix="backbone.embedding", weights=weights
        )        
        self.blocks = nn.ModuleList(
            [ResidualBlock(f"backbone.layers.{layer_id}", config, weights) for layer_id in range(config.n_layer)]
        )
        self.norm_f = FastRMSNorm.load(
            prefix="backbone.norm_f", 
            weights=weights, 
            eps=config.rms_norm_eps
        )
        print("🌈 model init done")

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

        past_key_values = [None] * len(self.blocks) if past_key_values is None else past_key_values

        for index, block in enumerate(self.blocks):
            hidden_states, new_key_values = block(hidden_states, past_key_values[index], mask)
            past_key_values[index] = new_key_values

        hidden_states = self.norm_f(hidden_states)
        return hidden_states, past_key_values

class MambaForCausalLM(torch.nn.Module):
    def __init__(self, config, weights):
        super().__init__()
        self.model = MambaModel(config, weights)

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
        print("🌈 model output done")