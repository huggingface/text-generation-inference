import torch
import torch.distributed

from torch.nn import functional as F

from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel
from transformers.models.gpt_neox import GPTNeoXConfig

# Flash attention imports
import flash_attn_cuda
import dropout_layer_norm


class FastLayerNorm(nn.LayerNorm):
    def forward(self, hidden_states, residual=None):
        if hidden_states.shape[-1] > 6144:
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

    def transpose_weight(self):
        self.weight = nn.Parameter(self.weight.T)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
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
        torch.distributed.all_reduce(out, group=self.process_group)
        return out



class FlashNeoxAttention(torch.nn.Module):
    def __init__(
        self,
        num_heads,
        hidden_size,
        process_group=None,
        reduce=True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_size = hidden_size // num_heads

        self.softmax_scale = self.head_size ** (-0.5)

        if process_group is None:
            self.query_key_value = FastLinear(hidden_size, 3 * hidden_size)
            self.c_proj = FastLinear(hidden_size, hidden_size)
        else:
            self.num_heads = self.num_heads // process_group.size()
            self.query_key_value = TensorParallelColumnLinear(
                hidden_size,
                3 * hidden_size,
                process_group=process_group,
            )
            self.c_proj = TensorParallelRowLinear(
                hidden_size, hidden_size, process_group=process_group, reduce=reduce
            )

    def shuffle_qkv_dims(self):
        """Swap dims to avoid an additional permute"""
        self.query_key_value.weight = torch.nn.Parameter(
            self.query_key_value.weight.view(
                self.num_heads, 3, self.head_size, self.hidden_size
            )
            .permute(1, 0, 2, 3)
            .reshape(-1, self.hidden_size)
        )
        self.query_key_value.bias = torch.nn.Parameter(
            self.query_key_value.bias.view(self.num_heads, 3, self.head_size)
            .permute(1, 0, 2)
            .reshape(-1)
        )

    def forward(
        self,
        hidden_states,
        cos,
        sin,
        cu_seqlens,
        max_s,
        layer_past,
        layer_past_present_indices,
        cu_seqlens_q,
    ):
        qkv = self.query_key_value(hidden_states)
        qkv = qkv.view(-1, 3, self.num_heads, self.head_size)
        qkv_rot = self.rotary_emb(qkv, cos, sin)

        # Prefill
        if layer_past_present_indices is None:
            # Copy to layer past
            layer_past[...] = qkv_rot[:, 1:]

            # output
            attn_output = torch.empty_like(qkv[:, 0])
            # flash attention
            flash_attn_cuda.fwd(
                qkv[:, 0],
                qkv[:, 1],
                qkv[:, 2],
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
            query = qkv_rot[:, 0]
            # Add present to the layer_past tensor at the correct indices
            layer_past[layer_past_present_indices] = qkv_rot[:, 1:]

            # output
            attn_output = torch.empty_like(query)
            # flash attention
            flash_attn_cuda.fwd(
                query,
                layer_past[:, 0],
                layer_past[:, 1],
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

        return self.dense(attn_output.view(-1, self.num_heads * self.head_size))


class FlashMLP(nn.Module):
    def __init__(
        self, act, hidden_size, intermediate_size, process_group=None, reduce=True
    ):
        super().__init__()
        self.act = (
            ACT2FN[act]
            if "gelu" not in act
            else lambda x: torch.nn.functional.gelu(x, approximate="tanh")
        )

        if process_group is None:
            self.dense_h_to_4h = FastLinear(hidden_size, intermediate_size)
            self.dense_4h_to_h = FastLinear(intermediate_size, hidden_size)
        else:
            self.dense_h_to_4h = TensorParallelColumnLinear(
                hidden_size,
                intermediate_size,
                process_group=process_group,
            )
            self.dense_4h_to_h = TensorParallelRowLinear(
                intermediate_size,
                hidden_size,
                process_group=process_group,
                reduce=reduce,
            )
        self.process_group = process_group

    def forward(self, hidden_states):
        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dense_4h_to_h(hidden_states)
        return hidden_states


class FlashNeoXLayer(nn.Module):
    def __init__(
        self,
        num_heads,
        act,
        hidden_size,
        intermediate_size,
        rotary_pct,
        rotary_emb_base,
        layer_norm_eps,
        use_parallel_residual,
        process_group=None,
    ):
        super().__init__()
        self.use_parallel_residual = use_parallel_residual
        self.input_layernorm = FastLayerNorm(hidden_size, eps=layer_norm_eps)
        self.post_attention_layernorm = FastLayerNorm(hidden_size, eps=layer_norm_eps)
        self.attention = FlashNeoxAttention(
            num_heads,
            hidden_size,
            rotary_pct,
            rotary_emb_base,
            process_group,
            reduce=not use_parallel_residual,
        )
        self.mlp = FlashMLP(
            act,
            hidden_size,
            intermediate_size,
            process_group,
            reduce=not use_parallel_residual,
        )
        self.process_group = process_group

    def forward(
        self,
        hidden_states,
        residual,
        cos,
        sin,
        cu_seqlens,
        max_s,
        layer_past,
        layer_past_present_indices,
        cu_seqlens_q,
    ):
        if self.use_parallel_residual:
            ln1_hidden_states, _ = self.input_layernorm(hidden_states)

            attn_output = self.attention(
                ln1_hidden_states,
                cos,
                sin,
                cu_seqlens,
                max_s,
                layer_past,
                layer_past_present_indices,
                cu_seqlens_q,
            )

            ln2_hidden_states, _ = self.post_attention_layernorm(hidden_states)

            mlp_output = self.mlp(ln2_hidden_states)
            intermediate = mlp_output + attn_output

            # Only reduce once and after the addition instead of once per layer
            if self.process_group is not None:
                torch.distributed.all_reduce(intermediate, group=self.process_group)

            return intermediate + hidden_states, None
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

            hidden_states = self.attention(
                hidden_states,
                cos,
                sin,
                cu_seqlens,
                max_s,
                layer_past,
                layer_past_present_indices,
                cu_seqlens_q,
            )

            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )

            mlp_output = self.mlp(hidden_states)

            return mlp_output, residual


class FlashGPTNeoXPreTrainedModel(PreTrainedModel):
    config_class = GPTNeoXConfig
    base_model_prefix = "gpt_neox"
    supports_gradient_checkpointing = False
    _no_split_modules = None


class FlashGPTNeoXModel(FlashGPTNeoXPreTrainedModel):
    def __init__(self, config, process_group=None):
        super().__init__(config)
        self.config = config

        self.tp_embeddings = False
        if process_group is not None:
            self.tp_rank = process_group.rank()
            self.tp_world_size = process_group.size()
            if config.vocab_size % self.tp_world_size == 0:
                self.tp_embeddings = True

        if self.tp_embeddings:
            self.embed_in = TensorParallelEmbedding(
                config.vocab_size, config.hidden_size, process_group=process_group
            )
        else:
            self.embed_in = nn.Embedding(config.vocab_size, config.hidden_size)

        self.layers = nn.ModuleList(
            [
                FlashNeoXLayer(
                    config.num_attention_heads,
                    config.hidden_act,
                    config.hidden_size,
                    config.intermediate_size,
                    config.rotary_pct,
                    config.rotary_emb_base,
                    config.layer_norm_eps,
                    config.use_parallel_residual,
                    process_group,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.final_layer_norm = FastLayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

        self.gradient_checkpointing = False

        self.head_size = self.layers[0].attention.head_size
        self.num_heads = self.layers[0].attention.num_heads

    def post_load_weights(self):
        if isinstance(self.embed_in, TensorParallelEmbedding):
            self.embed_in.add_null_idx()
        for layer in self.layers:
            layer: FlashNeoXLayer
            layer.attention.shuffle_qkv_dims()
            layer.attention.query_key_value.transpose_weight()
            layer.attention.dense.transpose_weight()
            layer.mlp.dense_h_to_4h.transpose_weight()
            layer.mlp.dense_4h_to_h.transpose_weight()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model = super(FlashGPTNeoXModel, cls).from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )
        model.post_load_weights()
        return model

    def forward(
        self,
        input_ids,
        position_ids,
        cu_seqlens,
        max_s,
        past_key_values=None,
    ):
        hidden_states = self.embed_in(input_ids)

        # Prefill
        if past_key_values is None:
            # Create past tensor
            past_key_values = hidden_states.new_empty(
                (
                    len(self.layers),
                    len(hidden_states),
                    2,
                    self.num_heads,
                    self.head_size,
                )
            )
            layer_past_present_indices = None
            cu_seqlens_q = None
        # Decode
        else:
            # Create indices from cumulative sequence lengths
            layer_past_present_indices = cu_seqlens[1:] - 1
            cu_seqlens_q = torch.arange(
                cu_seqlens.shape[0], dtype=torch.int32, device=hidden_states.device
            )

        # Get rotary cos and sin for this forward
        # Avoid to index in each layer
        cos, sin = self.layers[0].attention.rotary_emb.get_cos_sin(
            position_ids, max_s, hidden_states.dtype
        )

        residual = None
        for i, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                hidden_states,
                residual,
                cos,
                sin,
                cu_seqlens,
                max_s,
                past_key_values[i],
                layer_past_present_indices,
                cu_seqlens_q,
            )

        hidden_states, _ = self.final_layer_norm(hidden_states, residual)

        return hidden_states, past_key_values


class FlashGPTNeoXForCausalLM(FlashGPTNeoXPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        if config.tp_parallel:
            process_group = torch.distributed.distributed_c10d._get_default_group()
        else:
            process_group = None

        self.gpt_neox = FlashGPTNeoXModel(config, process_group)

        if self.gpt_neox.tp_embeddings:
            self.embed_out = FastLinear(
                config.hidden_size,
                config.vocab_size // process_group.size(),
                bias=False,
            )
        else:
            self.embed_out = FastLinear(
                config.hidden_size, config.vocab_size, bias=False
            )

    def post_load_weights(self):
        self.gpt_neox.post_load_weights()
        self.embed_out.transpose_weight()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model = super(FlashGPTNeoXForCausalLM, cls).from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )
        model.post_load_weights()
        return model

    def forward(
        self,
        input_ids,
        position_ids,
        cu_seqlens,
        max_s,
        past_key_values=None,
    ):
        hidden_states, present = self.gpt_neox(
            input_ids, position_ids, cu_seqlens, max_s, past_key_values
        )
        return self.embed_out(hidden_states), present
