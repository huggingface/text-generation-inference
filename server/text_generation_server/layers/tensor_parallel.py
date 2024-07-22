import torch
from torch.nn import functional as F
from typing import Iterable, List
from text_generation_server.layers.linear import get_linear, FastLinear
from text_generation_server.layers.exl2 import Exl2Weight
from text_generation_server.utils.import_utils import SYSTEM

if SYSTEM == "ipex":
    import intel_extension_for_pytorch as ipex


class LayerConcat(torch.nn.Module):
    """
    Apply multiple layers to the input and concatenate their
    outputs.
    """

    def __init__(self, layers: Iterable[torch.nn.Module], dim: int = -1):
        """
        `dim` is the dimension along which layer outputs are concatenated.
        """
        super().__init__()
        self.layers = layers
        self.dim = dim

    def forward(self, x: torch.Tensor):
        outputs = [layer(x) for layer in self.layers]
        return torch.cat(outputs, self.dim)


class SuperLayer(torch.nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.linear = linear

    def forward(self, x):
        return self.linear.forward(x)


class TensorParallelHead(SuperLayer):
    def __init__(self, linear, process_group, should_gather: bool):
        super().__init__(linear)
        self.process_group = process_group
        self.should_gather = should_gather

    @staticmethod
    def load(config, prefix: str, weights):
        if config.quantize == "exl2":
            try:
                # If the piece and LM head embeddings are shared, we have
                # non-quantized weights...
                weight = weights.get_tensor(f"{prefix}.weight")
            except:
                # ...otherwise they are quantized.
                weight = weights.get_weights_col(prefix)
            should_gather = weights.process_group.size() > 1
        elif weights.process_group.size() > 1:
            try:
                weight = weights.get_sharded(f"{prefix}.weight", dim=0)
                should_gather = True
            except AssertionError:
                # If the vocab size is not divisible by number of shards
                # just load the entire thing.
                weight = weights.get_tensor(f"{prefix}.weight")
                should_gather = False
        else:
            weight = weights.get_tensor(f"{prefix}.weight")
            should_gather = False

        # GPTQ,AWQ,EETQ don't quantize heads (nor embeddings)
        if config.quantize in ["gptq", "awq", "eetq", "marlin"]:
            quantize = None
        # See above, exl2 LM head can be quantized or not.
        elif config.quantize == "exl2" and not isinstance(weight, Exl2Weight):
            quantize = None
        else:
            quantize = config.quantize

        return TensorParallelHead(
            get_linear(weight, bias=None),
            process_group=weights.process_group,
            should_gather=should_gather,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self.should_gather:
            return super().forward(input)

        world_size = self.process_group.size()
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
            if SYSTEM == "ipex":
                ipex.distributed.all_gather_into_tensor(
                    world_out, gather_input, group=self.process_group
                )
            else:
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
        if SYSTEM == "ipex":
            ipex.distributed.all_gather(world_output, output, group=self.process_group)
        else:
            torch.distributed.all_gather(world_output, output, group=self.process_group)
        world_output = torch.cat(world_output, dim=-1)
        return world_output


class TensorParallelColumnLinear(SuperLayer):
    @classmethod
    def load_gate_up(cls, config, prefix: str, weights, bias: bool):
        """Specific method when the QKV was joined after the fact"""
        weight = weights.get_weights_col_packed_gate_up(prefix)
        if bias:
            raise NotImplementedError("packed_gate_up only implemented without bias")
        else:
            bias = None
        linear = get_linear(weight, bias)
        return cls(linear)

    @classmethod
    def load_qkv(
        cls,
        config,
        prefix: str,
        weights,
        bias: bool,
        num_heads: int,
        num_key_value_heads: int,
    ):
        """Specific method when the QKV was joined after the fact"""
        weight = weights.get_weights_col_packed_qkv(
            prefix,
            num_heads=num_heads,
            num_key_value_heads=num_key_value_heads,
        )
        if bias:
            raise NotImplementedError("packed_qkv only implemented for baichuan")
        else:
            bias = None
        linear = get_linear(weight, bias)
        return cls(linear)

    @classmethod
    def load(cls, config, prefix: str, weights, bias: bool):
        weight = weights.get_weights_col(prefix)
        if bias:
            bias = weights.get_sharded(f"{prefix}.bias", dim=0)
        else:
            bias = None
        linear = get_linear(weight, bias)
        return cls(linear)

    @classmethod
    def load_multi(cls, config, prefixes: List[str], weights, bias: bool, dim: int):
        if config.quantize == "exl2":
            linears = []
            for prefix in prefixes:
                weight = weights.get_weights_col(prefix)
                b = weights.get_tensor(f"{prefix}.bias") if bias else None
                linears.append(get_linear(weight, b))
            linear = LayerConcat(linears)
        else:
            weight = weights.get_multi_weights_col(prefixes, dim=dim)
            if bias:
                b = [weights.get_sharded(f"{p}.bias", dim=0) for p in prefixes]
                bias = torch.cat(b, dim=dim)
            else:
                bias = None
            linear = get_linear(weight, bias)
        return cls(linear)


class TensorParallelRowLinear(SuperLayer):
    def __init__(self, linear, process_group):
        super().__init__(linear)
        self.process_group = process_group

    @classmethod
    def load(cls, config, prefix: str, weights, bias: bool):
        weight = weights.get_weights_row(prefix)

        if bias and weights.process_group.rank() == 0:
            # Rank is only on the first rank process
            bias = weights.get_tensor(f"{prefix}.bias")
        else:
            bias = None
        return cls(
            get_linear(weight, bias),
            process_group=weights.process_group,
        )

    def forward(self, input: torch.Tensor, reduce: bool = True) -> torch.Tensor:
        out = super().forward(input)
        if self.process_group.size() > 1 and reduce:
            if SYSTEM == "ipex":
                ipex.distributed.all_reduce(out, group=self.process_group)
            else:
                torch.distributed.all_reduce(out, group=self.process_group)
        return out


class TensorParallelEmbedding(torch.nn.Module):
    def __init__(self, prefix: str, weights, reduce=True):
        super().__init__()
        weight = weights.get_partial_sharded(f"{prefix}.weight", dim=0)
        num_embeddings = weights.get_shape(f"{prefix}.weight")[0]

        process_group = weights.process_group

        world_size = process_group.size()
        rank = process_group.rank()

        block_size = (num_embeddings + world_size - 1) // world_size
        self.min_id = rank * block_size
        self.max_id = min(num_embeddings, (rank + 1) * block_size)
        self.null_idx = weight.shape[
            0
        ]  # Usually block_size, might be less in non even vocab_size.
        self.process_group = weights.process_group
        self.reduce = reduce

        """Additional 0 entry used for masking"""
        self.weight = torch.nn.Parameter(F.pad(weight, (0, 0, 0, 1)))

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
            if SYSTEM == "ipex":
                ipex.distributed.all_reduce(out, group=self.process_group)
            else:
                torch.distributed.all_reduce(out, group=self.process_group)
        return out
