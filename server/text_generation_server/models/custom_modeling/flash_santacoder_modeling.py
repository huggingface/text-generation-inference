import torch
import torch.distributed

import torch.nn.functional as F

from torch import nn
from transformers.activations import ACT2FN
from typing import Optional

# Flash attention imports
import flash_attn_cuda
import dropout_layer_norm

HAS_BITS_AND_BYTES = True
try:
    from bitsandbytes.nn import Linear8bitLt
except ImportError as e:
    HAS_BITS_AND_BYTES = False


class FastLayerNorm(nn.LayerNorm):
    def forward(self, hidden_states, residual=None):
        if hidden_states.shape[-1] > 8192:
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
        self.quantized = False
        self.bnb_linear = None

    def prepare_weights(self, quantize: bool = False):
        if quantize:
            if not HAS_BITS_AND_BYTES:
                raise ImportError(
                    "bitsandbytes is not available on your machine either because it is not installed "
                    "or you don't have a GPU.\n"
                    "You can install it with `pip install bitsandbytes`."
                )

            self.quantized = True
            self.bnb_linear = Linear8bitLt(
                self.in_features,
                self.out_features,
                has_fp16_weights=False,
                threshold=6.0,
                bias=False,
            )
            # Copy data to bnb_linear
            self.bnb_linear.weight.data = self.weight.data
            if self.bias is not None:
                self.bnb_linear.bias = nn.Parameter(self.bias)

            # Delete reference to data
            self.weight = None
            self.bias = None
        else:
            self.weight = nn.Parameter(self.weight.T)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.quantized:
            return self.bnb_linear(input)
        else:
            if self.bias is not None:
                return torch.addmm(self.bias, input, self.weight)
            return torch.matmul(input, self.weight)


class TensorParallelColumnLinear(FastLinear):
    def __init__(
        self,
        in_features,
        out_features,
        process_group: torch.distributed.ProcessGroup,
        bias=True,
        device=None,
        dtype=None,
    ):
        self.process_group = process_group
        self.tp_world_size = process_group.size()
        assert out_features % self.tp_world_size == 0
        out_features = out_features // self.tp_world_size

        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )


class TensorParallelRowLinear(FastLinear):
    def __init__(
        self,
        in_features,
        out_features,
        process_group: torch.distributed.ProcessGroup,
        reduce=True,
        bias=True,
        device=None,
        dtype=None,
    ):
        self.process_group = process_group
        self.tp_world_size = process_group.size()
        self.reduce = reduce
        assert in_features % self.tp_world_size == 0
        in_features = in_features // self.tp_world_size

        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = super(TensorParallelRowLinear, self).forward(input)
        if self.reduce:
            torch.distributed.all_reduce(out, group=self.process_group)

        return out


class TensorParallelEmbedding(nn.Embedding):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        process_group: torch.distributed.ProcessGroup,
        reduce=True,
        padding_idx=None,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        sparse=False,
        _weight=None,
        device=None,
        dtype=None,
    ):
        self.process_group = process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.reduce = reduce

        self.original_num_embeddings = num_embeddings

        assert num_embeddings % self.tp_world_size == 0
        block_size = num_embeddings // self.tp_world_size
        # inputs in `[min_id, max_id[` are handled by `self` to get embeddings
        self.min_id = self.tp_rank * block_size
        self.max_id = (self.tp_rank + 1) * block_size

        # Additional entry that will map to zero
        # Used for masking
        self.null_idx = block_size

        super().__init__(
            block_size,
            embedding_dim,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
            _weight=_weight,
            device=device,
            dtype=dtype,
        )

    def add_null_idx(self):
        """Additional 0 entry used for masking"""
        self.weight = nn.Parameter(F.pad(self.weight, (0, 0, 0, 1)))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # default all out of bounds values to `self.null_idx` that will then be mapped to 0
        # translate for [0, self.max_id - self.min_id[
        input = torch.where(
            (self.min_id > input) | (input >= self.max_id),
            self.null_idx,
            input - self.min_id,
        )
        out = super().forward(input)
        if self.reduce:
            torch.distributed.all_reduce(out, group=self.process_group)
        return out


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
            self.c_attn = FastLinear(hidden_size, hidden_size + 2 * self.head_size)
            self.c_proj = FastLinear(hidden_size, hidden_size)
        else:
            self.num_heads = self.num_heads // process_group.size()
            self.c_attn = FastLinear(hidden_size, self.head_size * (self.num_heads + 2))
            self.c_proj = TensorParallelRowLinear(
                hidden_size,
                hidden_size,
                process_group=process_group,
            )

    def forward(
        self,
        hidden_states,
        cu_seqlens,
        max_s,
        layer_past,
        layer_past_present_indices,
        cu_seqlens_q,
    ):
        qkv = self.c_attn(hidden_states)

        # Split query from key_value
        query, key_value = qkv.split(
            [self.head_size * self.num_heads, 2 * self.head_size], dim=1
        )

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
                else "none",
            )
        )

        if process_group is None:
            self.c_fc = FastLinear(hidden_size, intermediate_size)
            self.c_proj = FastLinear(intermediate_size, hidden_size)
        else:
            self.c_fc = TensorParallelColumnLinear(
                hidden_size,
                intermediate_size,
                process_group=process_group,
            )
            self.c_proj = TensorParallelRowLinear(
                intermediate_size,
                hidden_size,
                process_group=process_group,
            )

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

        self.process_group = process_group
        self.tp_embeddings = False
        if process_group is not None:
            self.tp_rank = process_group.rank()
            self.tp_world_size = process_group.size()
            if config.vocab_size % self.tp_world_size == 0:
                self.tp_embeddings = True

        if self.tp_embeddings:
            self.wte = TensorParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                reduce=False,
                process_group=process_group,
            )
            self.wpe = TensorParallelEmbedding(
                config.max_position_embeddings,
                config.hidden_size,
                reduce=False,
                process_group=process_group,
            )
        else:
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

    def post_load_weights(self, load_in_8bit: bool = False):
        if self.tp_embeddings:
            self.wte.add_null_idx()
            self.wpe.add_null_idx()
        for layer in self.h:
            layer: Block
            layer.attn.c_attn.prepare_weights(load_in_8bit)
            layer.attn.c_proj.prepare_weights(load_in_8bit)
            layer.mlp.c_fc.prepare_weights(load_in_8bit)
            layer.mlp.c_proj.prepare_weights(load_in_8bit)

    def forward(
        self,
        input_ids,
        position_ids,
        cu_seqlens,
        cu_seqlens_q,
        max_s,
        past_key_values: Optional[torch.Tensor] = None,
        pre_allocate_past_size: Optional[int] = None,
    ):
        hidden_states = self.wte(input_ids) + self.wpe(position_ids)
        if self.tp_embeddings:
            torch.distributed.all_reduce(hidden_states, group=self.process_group)

        # Prefill
        if past_key_values is None:
            # Create past tensor
            past_key_values = hidden_states.new_empty(
                (
                    len(self.h),
                    len(hidden_states)
                    if pre_allocate_past_size is None
                    else pre_allocate_past_size,
                    2,
                    1,
                    self.head_size,
                )
            )
            layer_past_present_indices = None
            slice_past_index = len(hidden_states)
        # Decode
        else:
            # Create indices from cumulative sequence lengths
            layer_past_present_indices = cu_seqlens[1:] - 1
            slice_past_index = None

        residual = None
        for i, layer in enumerate(self.h):
            # We added padding that we now need to slice
            layer_past_key_values = (
                past_key_values[i]
                if slice_past_index is None
                else past_key_values[i, :slice_past_index]
            )

            hidden_states, residual = layer(
                hidden_states,
                residual,
                cu_seqlens,
                max_s,
                layer_past_key_values,
                layer_past_present_indices,
                cu_seqlens_q,
            )

        hidden_states, _ = self.ln_f(hidden_states, residual)

        return hidden_states, past_key_values


class FlashSantacoderForCausalLM(nn.Module):
    def __init__(self, config, process_group=None):
        super().__init__()

        self.transformer = FlashSantacoderModel(config, process_group)

        if self.transformer.tp_embeddings:
            self.lm_head = FastLinear(
                config.hidden_size,
                config.vocab_size // process_group.size(),
                bias=False,
            )
        else:
            self.lm_head = FastLinear(config.hidden_size, config.vocab_size, bias=False)

    def post_load_weights(self, load_in_8bit: bool = False):
        self.transformer.post_load_weights(load_in_8bit)
        self.lm_head.prepare_weights()

    def forward(
        self,
        input_ids,
        position_ids,
        cu_seqlens,
        cu_seqlens_q,
        max_s,
        past_key_values: Optional[torch.Tensor] = None,
        pre_allocate_past_size: Optional[int] = None,
    ):
        hidden_states, present = self.transformer(
            input_ids,
            position_ids,
            cu_seqlens,
            cu_seqlens_q,
            max_s,
            past_key_values,
            pre_allocate_past_size,
        )
        logits = self.lm_head(hidden_states)

        if self.transformer.tp_embeddings:
            # Logits are sharded, so we need to gather them
            world_logits = [
                torch.empty_like(logits) for _ in range(self.transformer.tp_world_size)
            ]
            torch.distributed.all_gather(
                world_logits, logits, group=self.transformer.process_group
            )
            world_logits = torch.cat(world_logits, dim=1)

            return world_logits, present

        return logits, present
