import torch

from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Union, Type
from safetensors import safe_open
from dataclasses import dataclass

from text_generation_server.utils.import_utils import SYSTEM


class WeightsLoader(ABC):
    """
    Instances of this type implement higher-level weight loading.

    At a low-level, every weight is stored in the Safetensors format.
    The interpretation of weights may be different however, for instance
    could be packed, quantized weights. Loaders are responsible for
    interpreting the raw tensors, sharding tensors in a manner compatible
    with the format, etc.
    """

    @abstractmethod
    def get_weights(self, weights: "Weights", prefix: str):
        """
        Get weights at the given prefix and apply without tensor paralllism.
        """
        ...

    @abstractmethod
    def get_weights_col_packed(
        self,
        weights: "Weights",
        prefix: str,
        block_sizes: Union[int, List[int]],
    ):
        """
        Get the packed weights at the given prefix with column-splitting for
        tensor parallelism. This method should be used when multiple different
        weights are packed into a tensor, for instance, query/key/value
        weights or a gate/up projection.

        The `block_sizes` determines the proportions of the packed tensors.
        The columns are split in equally sized blocks when `block_sizes` is an
        `int`, or in blocks proportional given to the sizes. For instance
        `[2, 1, 1]` will divide an input with dimensionality `1024` in
        `[512, 256, 256]`.
        """
        ...

    def get_weights_col(self, weights: "Weights", prefix: str):
        """
        Get weights at the given prefix and apply column-splitting for tensor
        paralllism.
        """
        return weights.get_multi_weights_col([prefix], 0)

    @abstractmethod
    def get_multi_weights_col(self, weights: "Weights", prefixes: List[str], dim: int):
        """
        Get the weights at the given prefixes, column-split them for tensor
        parallelim, and then concatenate the weights along the given dimension.
        """
        ...

    @abstractmethod
    def get_weights_row(self, weights: "Weights", prefix: str):
        """
        Get the weights at the given prefix and apply row-splitting for tensor
        parallism.
        """
        ...


class Weight(ABC):
    """Instances of this type implement unquantized/quantized/to-be
    quantized weights."""

    @abstractmethod
    def get_linear(self, bias: torch.Tensor):
        """Create a linear layer from this weight."""
        ...


@dataclass
class UnquantizedWeight(Weight):
    weight: torch.Tensor

    def get_linear(self, bias: torch.Tensor):
        from text_generation_server.layers.linear import FastLinear, FastLinearROCm

        if SYSTEM == "rocm":
            return FastLinearROCm(self.weight, bias)
        else:
            return FastLinear(self.weight, bias)


class DefaultWeightsLoader(WeightsLoader):
    """Weight loader that loads (unquantized) Torch tensors."""

    def __init__(self, weight_class: Type[UnquantizedWeight]):
        """Create a loader. Weights will be wrapped using the given `weights_class`,
        normally this will be `UnquantizedWeight`, but a quantizer-specific class
        such as `Fp8Weight` can be used to quantize the weights during loading.
        """
        self.weight_class = weight_class

    """
    Loader that uses tensors as-is with the exception of applying sharding
    and/or concatenation.
    """

    def get_weights(self, weights: "Weights", prefix: str):
        return weights.get_tensor(f"{prefix}.weight")

    def get_weights_col_packed(
        self,
        weights: "Weights",
        prefix: str,
        block_sizes: Union[int, List[int]],
    ):
        return self.weight_class(
            weights.get_packed_sharded(
                f"{prefix}.weight", dim=0, block_sizes=block_sizes
            ),
        )

    def get_multi_weights_col(self, weights: "Weights", prefixes: List[str], dim: int):
        w = [weights.get_sharded(f"{p}.weight", dim=0) for p in prefixes]
        return self.weight_class(torch.cat(w, dim=dim))

    def get_weights_row(self, weights: "Weights", prefix: str):
        return self.weight_class(
            weights.get_sharded(f"{prefix}.weight", dim=1),
        )


class Weights:
    def __init__(
        self,
        filenames: List[Path],
        device,
        dtype,
        process_group,
        weights_loader: WeightsLoader,
        aliases: Optional[Dict[str, List[str]]] = None,
        prefix: Optional[str] = None,
    ):
        routing = {}
        for filename in filenames:
            with safe_open(filename, framework="pytorch") as f:
                for k in f.keys():
                    if k in routing:
                        raise RuntimeError(
                            f"Key {k} was found in multiple files: {filename} and {routing[k]}"
                        )
                    routing[k] = filename
        if aliases is None:
            aliases = {}
        self.aliases = aliases
        self.routing = routing
        self.device = device
        self.dtype = dtype
        self.process_group = process_group
        self.prefix = prefix
        self.weights_loader = weights_loader
        self._handles = {}

    def _get_handle(self, filename):
        if filename not in self._handles:
            f = safe_open(filename, framework="pytorch")
            self._handles[filename] = f

        return self._handles[filename]

    def get_filename(self, tensor_name: str) -> (str, str):
        names = [tensor_name]
        if self.prefix is not None:
            prefixed = f"{self.prefix}.{tensor_name}"
            names.append(prefixed)
        for name in names:
            filename = self.routing.get(name, None)
            if filename is not None:
                return str(filename), name

            aliases = self.aliases.get(name, [])
            for alias in aliases:
                filename = self.routing.get(alias, None)
                if filename is not None:
                    return str(filename), alias
        raise RuntimeError(f"weight {tensor_name} does not exist")

    def _get_slice(self, tensor_name: str):
        filename, tensor_name = self.get_filename(tensor_name)
        f = self._get_handle(filename)
        slice_ = f.get_slice(tensor_name)
        return slice_

    def _has_tensor(self, tensor_name: str):
        try:
            self.get_filename(tensor_name)
        except Exception:
            return False
        return True

    def get_shape(self, tensor_name: str):
        return self._get_slice(tensor_name).get_shape()

    def get_tensor(self, tensor_name: str, to_device=True, to_dtype=True):
        filename, tensor_name = self.get_filename(tensor_name)
        f = self._get_handle(filename)
        tensor = f.get_tensor(tensor_name)
        # Special case for gptq which shouldn't convert
        # u4 which are disguised as int32. Exl2 uses int16
        # as well. FP8 uses torch.float8_e4m3fn
        if (
            tensor.dtype
            not in [
                torch.float8_e4m3fn,
                torch.int16,
                torch.int32,
                torch.int64,
            ]
            and to_dtype
        ):
            tensor = tensor.to(dtype=self.dtype)
        if to_device:
            tensor = tensor.to(device=self.device)
        return tensor

    def get_partial_sharded(
        self, tensor_name: str, dim: int, to_device=True, to_dtype=True
    ):
        filename, tensor_name = self.get_filename(tensor_name)
        f = self._get_handle(filename)
        slice_ = f.get_slice(tensor_name)
        world_size = self.process_group.size()
        rank = self.process_group.rank()

        size = slice_.get_shape()[dim]
        block_size = (size + world_size - 1) // world_size
        start = rank * block_size
        stop = (rank + 1) * block_size

        if dim == 0:
            tensor = slice_[start:stop]
        elif dim == 1:
            tensor = slice_[:, start:stop]
        else:
            raise NotImplementedError("Let's make that generic when needed")
        # Special case for gptq which shouldn't convert
        # u4 which are disguised as int32. exl2 uses int16.
        # FP8 uses torch.float8_e4m3fn.
        if (
            tensor.dtype not in (torch.float8_e4m3fn, torch.int16, torch.int32)
            and to_dtype
        ):
            tensor = tensor.to(dtype=self.dtype)
        if to_device:
            tensor = tensor.to(device=self.device)
        return tensor

    def get_sharded(self, tensor_name: str, dim: int, to_device=True, to_dtype=True):
        filename, tensor_name = self.get_filename(tensor_name)
        f = self._get_handle(filename)
        slice_ = f.get_slice(tensor_name)
        world_size = self.process_group.size()
        size = slice_.get_shape()[dim]
        assert (
            size % world_size == 0
        ), f"The choosen size {size} is not compatible with sharding on {world_size} shards"
        return self.get_partial_sharded(
            tensor_name, dim, to_device=to_device, to_dtype=to_dtype
        )

    def get_packed_sharded(
        self,
        tensor_name: str,
        dim: int,
        block_sizes: Union[int, List[int]],
        to_dtype=True,
    ) -> torch.Tensor:
        """
        Get a shard from a tensor that packs multiple tensors.

        When a tensor packs multiple tensors (such as QKV or an up
        projection + gate projection), sharding with `get_sharded` is not
        safe since it would not split the packed tensors across shards.

        This method shards a tensor, such that the packed tensors are
        split across shards.

        The columns are split in equally sized blocks when blocks is an `int`, or
        in blocks proportional given to the sizes. For instance `[2, 1, 1]` will
        divide an input with dimensionality `1024` in `[512, 256, 256]`. This is
        convenient for e.g. splitting QKV without knowing the storage details of
        quantized weights.
        """
        slice_ = self._get_slice(tensor_name)
        total_size = slice_.get_shape()[dim]
        block_sizes = _blocks_to_block_sizes(total_size=total_size, blocks=block_sizes)

        world_size = self.process_group.size()
        rank = self.process_group.rank()

        tensors = []
        block_offset = 0
        for block_size in block_sizes:
            assert (
                block_size % world_size == 0
            ), f"Prepacked tensor cannot be sharded across {world_size} shards"
            shard_block_size = block_size // world_size
            start = rank * shard_block_size
            stop = (rank + 1) * shard_block_size
            if dim == 0:
                tensor = slice_[block_offset + start : block_offset + stop]
            elif dim == 1:
                tensor = slice_[:, block_offset + start : block_offset + stop]
            else:
                raise NotImplementedError("Currently only dim=0 or dim=1 is supported")
            tensors.append(tensor)
            block_offset += block_size
        tensor = torch.cat(tensors, dim=dim)
        tensor = tensor.to(device=self.device)

        # Avoid casting quantizer dtypes.
        if (
            tensor.dtype
            not in [
                torch.float8_e4m3fn,
                torch.int16,
                torch.int32,
                torch.int64,
            ]
            and to_dtype
        ):
            tensor = tensor.to(dtype=self.dtype)

        return tensor

    def get_weights(self, prefix: str):
        return self.weights_loader.get_weights(self, prefix)

    def get_weights_col_packed_qkv(
        self,
        prefix: str,
        num_heads: int,
        num_key_value_heads: int,
    ):
        return self.get_weights_col_packed(
            prefix, [num_heads, num_key_value_heads, num_key_value_heads]
        )

    def get_weights_col_packed_gate_up(self, prefix: str):
        return self.get_weights_col_packed(prefix, 2)

    def get_weights_col_packed(self, prefix: str, block_sizes: Union[int, List[int]]):
        """
        The columns are split in equally sized blocks when blocks is an `int`, or
        in blocks proportional given to the sizes. For instance `[2, 1, 1]` will
        divide an input with dimensionality `1024` in `[512, 256, 256]`. This is
        convenient for e.g. splitting QKV without knowing the storage details of
        quantized weights.
        """
        return self.weights_loader.get_weights_col_packed(self, prefix, block_sizes)

    def get_weights_col(self, prefix: str):
        return self.weights_loader.get_weights_col(self, prefix)

    def get_multi_weights_col(self, prefixes: List[str], dim: int):
        return self.weights_loader.get_multi_weights_col(self, prefixes, dim)

    def get_tensor_shard(self, var, dim):
        world_size = self.process_group.size()
        rank = self.process_group.rank()
        block_size = var.size()[dim] // world_size
        start = rank * block_size
        stop = (rank + 1) * block_size
        if dim == 0:
            tensor = var[start:stop]
        elif dim == 1:
            tensor = var[:, start:stop]
        else:
            raise NotImplementedError("Let's make that generic when needed")
        tensor = tensor.to(dtype=self.dtype)
        tensor = tensor.to(device=self.device)
        return tensor

    def get_weights_row(self, prefix: str):
        return self.weights_loader.get_weights_row(self, prefix)

    @contextmanager
    def use_loader(self, weights_loader: WeightsLoader):
        """
        This method is a context manager that can be used to use `Weights` with
        a different loader for the duration of the context.
        """

        old_loader = self.weights_loader
        self.weights_loader = weights_loader
        try:
            yield
        finally:
            self.weights_loader = old_loader

    @property
    def loader(self):
        return self.weights_loader


def _blocks_to_block_sizes(total_size: int, blocks: Union[int, List[int]]) -> List[int]:
    """
    Convert block count or proportions to block sizes.

    This function accepts

    - The number of blocks (int), in which case the block size is
      total_size//blocks; or
    - A list of block sizes (List[int]).

    In the latter case, if sum(blocks) < total_size, the ratios between
    the block sizes will be preserved. For instance, if blocks is
    [2, 1, 1] and total_size is 1024, the returned block sizes are
    [512, 256, 256].
    """
    if isinstance(blocks, list):
        total_blocks = sum(blocks)
        assert (
            total_size % total_blocks == 0
        ), f"Cannot split {total_size} in proportional blocks: {blocks}"
        part_size = total_size // total_blocks
        return [part_size * block for block in blocks]
    else:
        assert total_size % blocks == 0, f"Prepacked is not divisible by {blocks}"
        single_size = total_size // blocks
        return [single_size] * blocks
