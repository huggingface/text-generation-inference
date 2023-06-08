import torch
import torch.distributed

from torch import nn
from torch.nn import functional as F
from typing import List

HAS_BITS_AND_BYTES = True
try:
    import bitsandbytes as bnb
    from bitsandbytes.nn import Int8Params

except ImportError:
    HAS_BITS_AND_BYTES = False

from accelerate import init_empty_weights


# Monkey patching
@classmethod
def load_layer_norm(cls, prefix, weights, eps):
    weight = weights.get_tensor(f"{prefix}.weight")
    bias = weights.get_tensor(f"{prefix}.bias")
    with init_empty_weights():
        ln = cls(weight.shape, eps=eps)

    ln.weight = nn.Parameter(weight)
    ln.bias = nn.Parameter(bias)
    return ln


torch.nn.LayerNorm.load = load_layer_norm


class FastLinear(nn.Module):
    def __init__(
        self,
        weight,
        bias,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight)
        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

    @classmethod
    def load(cls, config, prefix: str, weights, bias: bool):
        weight = weights.get_tensor(f"{prefix}.weight")
        if bias:
            bias = weights.get_tensor(f"{prefix}.bias")
        else:
            bias = None
        return cls(weight, bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight, self.bias)


class Linear8bitLt(nn.Module):
    def __init__(
        self,
        weight,
        bias,
        has_fp16_weights=True,
        memory_efficient_backward=False,
        threshold=0.0,
        index=None,
    ):
        super().__init__()
        assert (
            not memory_efficient_backward
        ), "memory_efficient_backward is no longer required and the argument is deprecated in 0.37.0 and will be removed in 0.39.0"
        self.state = bnb.MatmulLtState()
        self.index = index

        # Necessary for stacked layers
        self.state.threshold = threshold
        self.state.has_fp16_weights = has_fp16_weights
        self.state.memory_efficient_backward = memory_efficient_backward
        if threshold > 0.0 and not has_fp16_weights:
            self.state.use_pool = True

        self.weight = Int8Params(
            weight.data,
            has_fp16_weights=has_fp16_weights,
            requires_grad=has_fp16_weights,
        )
        self.weight.cuda(weight.device)
        self.bias = bias

    def init_8bit_state(self):
        self.state.CB = self.weight.CB
        self.state.SCB = self.weight.SCB
        self.weight.CB = None
        self.weight.SCB = None

    def forward(self, x: torch.Tensor):
        self.state.is_training = self.training
        if self.weight.CB is not None:
            self.init_8bit_state()

        # weights are cast automatically as Int8Params, but the bias has to be cast manually
        if self.bias is not None and self.bias.dtype != x.dtype:
            self.bias.data = self.bias.data.to(x.dtype)

        out = bnb.matmul(x, self.weight, bias=self.bias, state=self.state)

        if not self.state.has_fp16_weights:
            if self.state.CB is not None and self.state.CxB is not None:
                # we converted 8-bit row major to turing/ampere format in the first inference pass
                # we no longer need the row-major weight
                del self.state.CB
                self.weight.data = self.state.CxB
        return out


def get_linear(weight, bias, quantize):
    if quantize is None:
        linear = FastLinear(weight, bias)
    elif quantize == "bitsandbytes":
        linear = Linear8bitLt(
            weight,
            bias,
            has_fp16_weights=False,
            threshold=6.0,
        )
        if bias is not None:
            linear.bias = nn.Parameter(bias)
    elif quantize == "gptq":
        raise NotImplementedError("Soon")
    else:
        raise NotImplementedError(f"Quantization `{quantize}` is not implemented yet.")
    return linear


class SuperLayer(nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.linear = linear

    def forward(self, x):
        return self.linear.forward(x)


class TensorParallelHead(SuperLayer):
    def __init__(self, linear, process_group):
        super().__init__(linear)
        self.process_group = process_group

    @staticmethod
    def load(config, prefix: str, weights):
        weight = weights.get_sharded(f"{prefix}.weight", dim=0)
        return TensorParallelHead(
            get_linear(weight, bias=None, quantize=config.quantize),
            process_group=weights.process_group,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        world_size = self.process_group.size()
        if world_size == 1:
            return super().forward(input)

        if len(input.shape) == 2 and isinstance(self.linear, FastLinear):
            out_dim = self.linear.weight.shape[0]

            if input.shape[0] == 1:
                world_out = input.new_empty(1, out_dim * world_size)
                local_out = input.new_empty(1, out_dim)
                gather_input = local_out
            else:
                world_out = input.new_empty(out_dim * world_size, input.shape[0])
                gather_input = input.new_empty(out_dim, input.shape[0])
                local_out = gather_input.T

            torch.mm(input, self.linear.weight.T, out=local_out)

            torch.distributed.all_gather_into_tensor(
                world_out, gather_input, group=self.process_group
            )

            if input.shape[0] == 1:
                return world_out
            return world_out.T

        output = super().forward(input)
        world_output = [
            torch.empty_like(output) for _ in range(self.process_group.size())
        ]
        torch.distributed.all_gather(world_output, output, group=self.process_group)
        world_output = torch.cat(world_output, dim=-1)
        return world_output


class TensorParallelColumnLinear(SuperLayer):
    @classmethod
    def load(cls, config, prefix: str, weights, bias: bool):
        weight = weights.get_sharded(f"{prefix}.weight", dim=0)
        if bias:
            bias = weights.get_sharded(f"{prefix}.bias", dim=0)
        else:
            bias = None
        return cls(get_linear(weight, bias, config.quantize))

    @classmethod
    def load_multi(cls, config, prefixes: List[str], weights, bias: bool, dim: int):
        w = [weights.get_sharded(f"{p}.weight", dim=0) for p in prefixes]
        weight = torch.cat(w, dim=dim)

        if bias:
            b = [weights.get_sharded(f"{p}.bias", dim=0) for p in prefixes]
            bias = torch.cat(b, dim=0)
        else:
            bias = None
        return cls(get_linear(weight, bias, config.quantize))


class TensorParallelRowLinear(SuperLayer):
    def __init__(self, linear, process_group):
        super().__init__(linear)
        self.process_group = process_group

    @classmethod
    def load(cls, config, prefix: str, weights, bias: bool):
        weight = weights.get_sharded(f"{prefix}.weight", dim=1)
        if bias and weights.process_group.rank() == 0:
            # Rank is only on the first rank process
            bias = weights.get_tensor(f"{prefix}.bias")
        else:
            bias = None
        return cls(
            get_linear(weight, bias, config.quantize),
            process_group=weights.process_group,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = super().forward(input)
        if self.process_group.size() > 1:
            torch.distributed.all_reduce(out, group=self.process_group)
        return out


class TensorParallelEmbedding(nn.Module):
    def __init__(self, prefix: str, weights, reduce=True):
        super().__init__()
        weight = weights.get_sharded(f"{prefix}.weight", dim=0)
        num_embeddings = weights.get_shape(f"{prefix}.weight")[0]

        process_group = weights.process_group

        world_size = process_group.size()
        rank = process_group.rank()

        block_size = num_embeddings // world_size
        self.min_id = rank * block_size
        self.max_id = min(num_embeddings, (rank + 1) * block_size)
        self.null_idx = block_size
        self.process_group = weights.process_group
        self.reduce = reduce

        """Additional 0 entry used for masking"""
        self.weight = nn.Parameter(F.pad(weight, (0, 0, 0, 1)))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # default all out of bounds values to `self.null_idx` that will then be mapped to 0
        # translate for [0, self.max_id - self.min_id[
        input = torch.where(
            (self.min_id > input) | (input >= self.max_id),
            self.null_idx,
            input - self.min_id,
        )
        out = torch.nn.functional.embedding(input, self.weight)
        if self.reduce and self.process_group.size() > 1:
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

    class PositionRotaryEmbedding(nn.Module):
        def __init__(self, inv_freq):
            super().__init__()

            self.register_buffer("inv_freq", inv_freq)
            self._seq_len_cached = 0
            self._cos_cached = None
            self._sin_cached = None
            self._cos_k_cached = None
            self._sin_k_cached = None

        @classmethod
        def static(cls, dim, base, device):
            inv_freq = 1.0 / (
                base
                ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
            )
            return cls(inv_freq)

        @classmethod
        def load(cls, prefix, weights):
            # XXX: Always load this in float32 !
            dtype = weights.dtype
            weights.dtype = torch.float32
            inv_freq = weights.get_tensor(f"{prefix}.inv_freq")
            weights.dtype = dtype
            return cls(inv_freq)

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
