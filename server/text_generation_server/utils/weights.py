import os
from pathlib import Path
from typing import Dict, List, Optional, Union
from safetensors import safe_open
import torch


class Weights:
    def __init__(
        self,
        filenames: List[Path],
        device,
        dtype,
        process_group,
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
        self._handles = {}

    @staticmethod
    def open(
        filenames: List[Path],
        device: torch.device,
        dtype: torch.dtype,
        quantize: Optional[str],
        process_group,
        aliases: Optional[Dict[str, List[str]]] = None,
        prefix: Optional[str] = None,
    ) -> "Weights":
        if quantize == "marlin":
            from text_generation_server.layers.marlin import MarlinWeights

            return MarlinWeights(
                filenames=filenames,
                device=device,
                dtype=dtype,
                process_group=process_group,
                aliases=aliases,
                prefix=prefix,
            )
        elif quantize in {"awq", "gptq"}:
            from text_generation_server.layers.gptq import GPTQWeights

            return GPTQWeights(
                filenames=filenames,
                device=device,
                dtype=dtype,
                process_group=process_group,
                aliases=aliases,
                prefix=prefix,
            )
        else:
            return Weights(
                filenames=filenames,
                device=device,
                dtype=dtype,
                process_group=process_group,
                aliases=aliases,
                prefix=prefix,
            )

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

    def get_shape(self, tensor_name: str):
        return self._get_slice(tensor_name).get_shape()

    def get_tensor(self, tensor_name: str, to_device=True):
        filename, tensor_name = self.get_filename(tensor_name)
        f = self._get_handle(filename)
        tensor = f.get_tensor(tensor_name)
        # Special case for gptq which shouldn't convert
        # u4 which are disguised as int32. Exl2 uses int16
        # as well.
        if tensor.dtype not in [torch.int16, torch.int32, torch.int64]:
            tensor = tensor.to(dtype=self.dtype)
        if to_device:
            tensor = tensor.to(device=self.device)
        return tensor

    def get_partial_sharded(self, tensor_name: str, dim: int):
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
        if tensor.dtype not in (torch.int16, torch.int32):
            tensor = tensor.to(dtype=self.dtype)
        tensor = tensor.to(device=self.device)
        return tensor

    def get_sharded(self, tensor_name: str, dim: int):
        filename, tensor_name = self.get_filename(tensor_name)
        f = self._get_handle(filename)
        slice_ = f.get_slice(tensor_name)
        world_size = self.process_group.size()
        size = slice_.get_shape()[dim]
        assert (
            size % world_size == 0
        ), f"The choosen size {size} is not compatible with sharding on {world_size} shards"
        return self.get_partial_sharded(tensor_name, dim)

    def get_packed_sharded(
        self, tensor_name: str, dim: int, block_sizes: Union[int, List[int]]
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
        if tensor.dtype not in [torch.int16, torch.int32, torch.int64]:
            tensor = tensor.to(dtype=self.dtype)

        return tensor

    def get_weights_col_packed_qkv(
        self,
        prefix: str,
        quantize: str,
        num_heads: int,
        num_key_value_heads: int,
    ):
        return self.get_weights_col_packed(
            prefix, quantize, [num_heads, num_key_value_heads, num_key_value_heads]
        )

    def get_weights_col_packed_gate_up(self, prefix: str, quantize: str):
        return self.get_weights_col_packed(prefix, quantize, 2)

    def get_weights_col_packed(
        self, prefix: str, quantize: str, block_sizes: Union[int, List[int]]
    ):
        """
        Highly specific when the underlying tensor is a simple cat of Q,K,V instead of being
        already alternating Q,K,V within the main tensor.

        The columns are split in equally sized blocks when blocks is an `int`, or
        in blocks proportional given to the sizes. For instance `[2, 1, 1]` will
        divide an input with dimensionality `1024` in `[512, 256, 256]`. This is
        convenient for e.g. splitting QKV without knowing the storage details of
        quantized weights.
        """
        return self.get_packed_sharded(
            f"{prefix}.weight", dim=0, block_sizes=block_sizes
        )

    def get_weights_col(self, prefix: str, quantize: str):
        if quantize == "exl2":
            from text_generation_server.layers.exl2 import Exl2Weight

            try:
                q_weight = self.get_tensor(f"{prefix}.q_weight")
            except RuntimeError:
                raise RuntimeError(
                    f"Cannot load `exl2`-quantized weight, make sure the model is already quantized."
                )

            q_scale = self.get_tensor(f"{prefix}.q_scale")
            q_invperm = self.get_tensor(f"{prefix}.q_invperm")
            q_scale_max = self.get_tensor(f"{prefix}.q_scale_max")
            q_groups = self.get_tensor(f"{prefix}.q_groups")

            return Exl2Weight(
                q_weight=q_weight,
                q_scale=q_scale,
                q_invperm=q_invperm,
                q_scale_max=q_scale_max,
                q_groups=q_groups,
            )

        return self.get_multi_weights_col([prefix], quantize, 0)

    def get_multi_weights_col(self, prefixes: List[str], quantize: str, dim: int):
        if quantize == "exl2":
            raise ValueError("get_multi_weights_col is not supported for exl2")
        else:
            w = [self.get_sharded(f"{p}.weight", dim=0) for p in prefixes]
            weight = torch.cat(w, dim=dim)

        return weight

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

    def get_multi_weights_row(self, prefix: str, quantize: str):
        if quantize == "exl2":
            from text_generation_server.layers.exl2 import Exl2Weight

            try:
                q_weight = self.get_tensor(f"{prefix}.q_weight")
            except RuntimeError:
                raise RuntimeError(
                    f"Cannot load `exl2`-quantized weight, make sure the model is already quantized."
                )

            q_scale = self.get_tensor(f"{prefix}.q_scale")
            q_invperm = self.get_tensor(f"{prefix}.q_invperm")
            q_scale_max = self.get_tensor(f"{prefix}.q_scale_max")
            q_groups = self.get_tensor(f"{prefix}.q_groups")

            return Exl2Weight(
                q_weight=q_weight,
                q_scale=q_scale,
                q_invperm=q_invperm,
                q_scale_max=q_scale_max,
                q_groups=q_groups,
            )
        elif quantize == "awq":
            from text_generation_server.layers.gptq import GPTQWeight

            gptq_params = self._get_gptq_params()

            try:
                qweight = self.get_sharded(f"{prefix}.qweight", dim=0)
            except RuntimeError:
                raise RuntimeError(
                    "Cannot load `awq` weight, make sure the model is already quantized"
                )

            qzeros = self.get_sharded(f"{prefix}.qzeros", dim=0)
            scales = self.get_sharded(f"{prefix}.scales", dim=0)
            g_idx = None
            use_exllama = False

            weight = GPTQWeight(
                qweight=qweight,
                qzeros=qzeros,
                scales=scales,
                g_idx=g_idx,
                bits=gptq_params.bits,
                groupsize=gptq_params.groupsize,
                use_exllama=use_exllama,
            )
        else:
            weight = self.get_sharded(f"{prefix}.weight", dim=1)
        return weight


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
