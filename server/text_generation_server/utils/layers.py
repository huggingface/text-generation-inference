import torch

from torch import nn
from torch.nn import functional as F
from typing import Optional

HAS_BITS_AND_BYTES = True
try:
    from bitsandbytes.nn import Linear8bitLt
except ImportError as e:
    HAS_BITS_AND_BYTES = False


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

    def prepare_weights(self, quantize: Optional[str] = None):
        if quantize == "bitsandbytes":
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
        elif quantize == "gptq":
            raise NotImplementedError("`gptq` is not implemented for now")
        elif quantize is None:
            self.weight = nn.Parameter(self.weight.T)
        else:
            raise ValueError(f"Unexpected quantize `{quantize}`")

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
        self.reduce = reduce
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
        if self.reduce:
            torch.distributed.all_reduce(out, group=self.process_group)
        return out


try:
    import dropout_layer_norm

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

except ImportError:
    pass


try:
    from flash_attn.layers.rotary import RotaryEmbedding
    import rotary_emb

    class PositionRotaryEmbedding(RotaryEmbedding):
        def _update_cos_sin_cache(self, dtype, device, seqlen):
            # Reset the tables if the sequence length has changed,
            # or if we're on a new device (possibly due to tracing for instance)
            if (
                seqlen > self._seq_len_cached
                or self._cos_cached.device != device
                or self._cos_cached.dtype != dtype
            ):
                self._seq_len_cached = seqlen
                t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
                # Don't do einsum, it converts fp32 to fp16
                # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
                freqs = torch.outer(t, self.inv_freq.to(device=t.device))
                self._cos_cached = torch.cos(freqs).to(dtype)
                self._sin_cached = torch.sin(freqs).to(dtype)

        def get_cos_sin(
            self, position_ids: torch.Tensor, max_s: int, dtype: torch.dtype
        ):
            """
            Return cos and sin for the asked position ids
            """

            self._update_cos_sin_cache(dtype, position_ids.device, max_s)

            cos = torch.index_select(self._cos_cached, 0, position_ids)
            sin = torch.index_select(self._sin_cached, 0, position_ids)
            return cos.unsqueeze(1), sin.unsqueeze(1)

        def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
            rotary_dim = cos.shape[-1]
            x1 = x[..., :rotary_dim]
            x2 = x[..., rotary_dim : 2 * rotary_dim]

            rotary_emb.apply_rotary(x1, x2, cos, sin, x1, x2, False)
            return x

except ImportError:
    pass
