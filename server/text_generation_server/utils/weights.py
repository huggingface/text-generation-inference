import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from safetensors import safe_open, SafetensorError
import torch
from loguru import logger
from huggingface_hub import hf_hub_download
import json
from text_generation_server.utils.log import log_once


@dataclass
class _GPTQParams:
    bits: int
    checkpoint_format: Optional[str]
    groupsize: int
    desc_act: bool
    quant_method: str
    sym: bool


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
        if quantize in ["gptq", "awq"]:
            from text_generation_server.layers.gptq import GPTQWeight

            try:
                qweight = self.get_packed_sharded(
                    f"{prefix}.qweight", dim=1, block_sizes=block_sizes
                )
            except RuntimeError:
                raise RuntimeError(
                    f"Cannot load `{quantize}` weight, make sure the model is already quantized."
                )

            gptq_params = self._get_gptq_params()

            qzeros = self.get_packed_sharded(
                f"{prefix}.qzeros", dim=1, block_sizes=block_sizes
            )
            scales = self.get_packed_sharded(
                f"{prefix}.scales", dim=1, block_sizes=block_sizes
            )
            scales = scales.to(dtype=self.dtype)

            if quantize == "gptq" and gptq_params.quant_method == "gptq":
                g_idx = self.get_tensor(f"{prefix}.g_idx")
            elif quantize == "gptq" and gptq_params.quant_method == "awq":
                log_once(
                    logger.info, "Converting AWQ model to Exllama/GPTQ packing format."
                )
                from text_generation_server.layers.awq.conversion_utils import (
                    fast_awq_to_gptq,
                )

                qweight, qzeros = fast_awq_to_gptq(qweight, qzeros)
                g_idx = (
                    torch.arange(
                        qweight.shape[0] * (32 // gptq_params.bits),
                        device=qweight.device,
                    )
                    // gptq_params.groupsize
                ).to(dtype=torch.int32)
            else:
                g_idx = None

            weight = GPTQWeight(
                qweight=qweight,
                qzeros=qzeros,
                scales=scales,
                g_idx=g_idx,
                bits=gptq_params.bits,
                groupsize=gptq_params.groupsize,
                use_exllama=False,
            )
        elif quantize == "marlin":
            from text_generation_server.layers.marlin import (
                GPTQMarlin24Weight,
                MarlinWeight,
                repack_gptq_for_marlin,
            )

            quant_method = getattr(self, "quant_method", "marlin")
            is_marlin_24 = getattr(self, "gptq_checkpoint_format", None) == "marlin_24"
            if is_marlin_24:
                B = self.get_packed_sharded(
                    f"{prefix}.B_24", dim=1, block_sizes=block_sizes
                )
                B_meta = self.get_packed_sharded(
                    f"{prefix}.B_meta", dim=1, block_sizes=block_sizes
                )
                s = self.get_packed_sharded(
                    f"{prefix}.s", dim=1, block_sizes=block_sizes
                )

                gptq_params = self._get_gptq_params()
                weight = GPTQMarlin24Weight(
                    B=B, B_meta=B_meta, s=s, bits=gptq_params.bits
                )
            elif quant_method == "gptq":
                gptq_params = self._get_gptq_params()
                try:
                    qweight = self.get_packed_sharded(
                        f"{prefix}.qweight", dim=1, block_sizes=block_sizes
                    )
                except RuntimeError:
                    raise RuntimeError(
                        f"Cannot load `{quantize}` weight for GPTQ -> Marlin repacking, make sure the model is already quantized"
                    )

                scales = self.get_packed_sharded(
                    f"{prefix}.scales", dim=1, block_sizes=block_sizes
                )
                g_idx = self.get_tensor(f"{prefix}.g_idx")
                weight = repack_gptq_for_marlin(
                    qweight=qweight,
                    scales=scales,
                    g_idx=g_idx,
                    bits=gptq_params.bits,
                    desc_act=gptq_params.desc_act,
                    groupsize=gptq_params.groupsize,
                    sym=gptq_params.sym,
                    sharded_infeatures=False,
                )
            else:
                B = self.get_packed_sharded(
                    f"{prefix}.B", dim=1, block_sizes=block_sizes
                )
                s = self.get_packed_sharded(
                    f"{prefix}.s", dim=1, block_sizes=block_sizes
                )
                weight = MarlinWeight(B=B, s=s)
        else:
            weight = self.get_packed_sharded(
                f"{prefix}.weight", dim=0, block_sizes=block_sizes
            )
        return weight

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
        elif quantize in ["gptq", "awq"]:
            from text_generation_server.layers.gptq import GPTQWeight

            try:
                qweight = torch.cat(
                    [self.get_sharded(f"{p}.qweight", dim=1) for p in prefixes], dim=1
                )
            except RuntimeError:
                raise RuntimeError(
                    f"Cannot load `{quantize}` weight, make sure the model is already quantized"
                )

            qzeros = torch.cat(
                [self.get_sharded(f"{p}.qzeros", dim=1) for p in prefixes], dim=1
            )
            scales = torch.cat(
                [self.get_sharded(f"{p}.scales", dim=1) for p in prefixes], dim=1
            )

            gptq_params = self._get_gptq_params()

            from text_generation_server.layers.gptq import HAS_EXLLAMA

            use_exllama = (
                gptq_params.bits == 4
                and HAS_EXLLAMA
                and quantize == "gptq"
                and not gptq_params.desc_act
            )

            if quantize == "gptq" and gptq_params.quant_method == "gptq":
                w = [self.get_tensor(f"{p}.g_idx") for p in prefixes]
                for w2 in w[1:]:
                    torch.testing.assert_close(w2, w[0])
                g_idx = w[0]
            elif quantize == "gptq" and gptq_params.quant_method == "awq":
                log_once(
                    logger.info, "Converting AWQ model to Exllama/GPTQ packing format."
                )
                from text_generation_server.layers.awq.conversion_utils import (
                    fast_awq_to_gptq,
                )

                qweight, qzeros = fast_awq_to_gptq(qweight, qzeros)
                if use_exllama:
                    g_idx = None
                else:
                    g_idx = (
                        torch.arange(
                            qweight.shape[0] * (32 // gptq_params.bits),
                            device=qweight.device,
                        )
                        // gptq_params.groupsize
                    ).to(dtype=torch.int32)
            else:
                g_idx = None

            weight = GPTQWeight(
                qweight=qweight,
                qzeros=qzeros,
                scales=scales,
                g_idx=g_idx,
                bits=gptq_params.bits,
                groupsize=gptq_params.groupsize,
                use_exllama=use_exllama,
            )
        elif quantize == "marlin":
            from text_generation_server.layers.gptq import GPTQWeight
            from text_generation_server.layers.marlin import (
                GPTQMarlin24Weight,
                MarlinWeight,
                repack_gptq_for_marlin,
            )

            quant_method = getattr(self, "quant_method", "marlin")
            is_marlin_24 = getattr(self, "gptq_checkpoint_format", None) == "marlin_24"
            if is_marlin_24:
                try:
                    B = torch.cat(
                        [self.get_sharded(f"{p}.B_24", dim=1) for p in prefixes], dim=1
                    )
                except RuntimeError:
                    raise RuntimeError(
                        f"Cannot load `{quantize}` weight, make sure the model is already quantized"
                    )

                B_meta = torch.cat(
                    [self.get_sharded(f"{p}.B_meta", dim=1) for p in prefixes], dim=1
                )

                s = torch.cat(
                    [self.get_sharded(f"{p}.s", dim=1) for p in prefixes], dim=1
                )

                gptq_params = self._get_gptq_params()
                weight = GPTQMarlin24Weight(
                    B=B, B_meta=B_meta, s=s, bits=gptq_params.bits
                )
            elif quant_method == "gptq":
                gptq_params = self._get_gptq_params()
                try:
                    qweight = torch.cat(
                        [self.get_sharded(f"{p}.qweight", dim=1) for p in prefixes],
                        dim=1,
                    )
                except RuntimeError:
                    raise RuntimeError(
                        f"Cannot load `{quantize}` weight for GPTQ -> Marlin repacking, make sure the model is already quantized"
                    )

                scales = torch.cat(
                    [self.get_sharded(f"{p}.scales", dim=1) for p in prefixes], dim=1
                )
                w = [self.get_tensor(f"{p}.g_idx") for p in prefixes]
                for w2 in w[1:]:
                    torch.testing.assert_close(w2, w[0])
                g_idx = w[0]

                weight = repack_gptq_for_marlin(
                    qweight=qweight,
                    scales=scales,
                    g_idx=g_idx,
                    bits=gptq_params.bits,
                    desc_act=gptq_params.desc_act,
                    groupsize=gptq_params.groupsize,
                    sym=gptq_params.sym,
                    sharded_infeatures=False,
                )
            else:
                try:
                    B = torch.cat(
                        [self.get_sharded(f"{p}.B", dim=1) for p in prefixes], dim=1
                    )
                except RuntimeError:
                    raise RuntimeError(
                        f"Cannot load `{quantize}` weight, make sure the model is already quantized"
                    )
                s = torch.cat(
                    [self.get_sharded(f"{p}.s", dim=1) for p in prefixes], dim=1
                )

                weight = MarlinWeight(B=B, s=s)

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

        elif quantize == "gptq":
            use_exllama = True
            gptq_params = self._get_gptq_params()

            if gptq_params.bits != 4:
                use_exllama = False

            if gptq_params.desc_act:
                log_once(logger.warning, "Disabling exllama because desc_act=True")
                use_exllama = False

            try:
                qweight = self.get_sharded(f"{prefix}.qweight", dim=0)
            except RuntimeError:
                raise RuntimeError(
                    "Cannot load `gptq` weight, make sure the model is already quantized, or quantize it with `text-generation-server quantize ORIGINAL_MODEL_ID NEW_MODEL_ID`"
                )

            if gptq_params.quant_method == "gptq":
                g_idx = self.get_sharded(f"{prefix}.g_idx", dim=0)
            elif gptq_params.quant_method == "awq":
                g_idx = None

            if self.process_group.size() > 1:
                if g_idx is not None:
                    if (
                        not torch.equal(
                            g_idx.cpu(),
                            torch.tensor(
                                [
                                    i // gptq_params.groupsize
                                    for i in range(g_idx.shape[0])
                                ],
                                dtype=torch.int32,
                            ),
                        )
                        and not (g_idx == 0).all()
                    ):
                        # Exllama implementation does not support row tensor parallelism with act-order, as
                        # it would require to reorder input activations that are split unto several GPUs
                        use_exllama = False

            from text_generation_server.layers.gptq import (
                HAS_EXLLAMA,
                CAN_EXLLAMA,
                GPTQWeight,
            )

            if use_exllama:
                if not HAS_EXLLAMA:
                    if CAN_EXLLAMA:
                        log_once(
                            logger.warning,
                            "Exllama GPTQ cuda kernels (which are faster) could have been used, but are not currently installed, try using BUILD_EXTENSIONS=True",
                        )
                    use_exllama = False
                else:
                    log_once(logger.info, f"Using exllama kernels v{HAS_EXLLAMA}")

            if use_exllama and gptq_params.groupsize != -1:
                qzeros = self.get_sharded(f"{prefix}.qzeros", dim=0)
                scales = self.get_sharded(f"{prefix}.scales", dim=0)
            else:
                qzeros = self.get_tensor(f"{prefix}.qzeros")
                scales = self.get_tensor(f"{prefix}.scales")

            if use_exllama and g_idx is not None:
                g_idx = g_idx - g_idx[0]

            if gptq_params.quant_method == "awq":
                log_once(
                    logger.info, "Converting AWQ model to Exllama/GPTQ packing format."
                )
                from text_generation_server.layers.awq.conversion_utils import (
                    fast_awq_to_gptq,
                )

                qweight, qzeros = fast_awq_to_gptq(qweight, qzeros)
                if use_exllama:
                    g_idx = None
                else:
                    g_idx = (
                        torch.arange(
                            qweight.shape[0] * (32 // gptq_params.bits),
                            device=qweight.device,
                        )
                        // gptq_params.groupsize
                    ).to(dtype=torch.int32)

            weight = GPTQWeight(
                qweight=qweight,
                qzeros=qzeros,
                scales=scales,
                g_idx=g_idx,
                bits=gptq_params.bits,
                groupsize=gptq_params.groupsize,
                use_exllama=use_exllama,
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
        elif quantize == "marlin":
            from text_generation_server.layers.gptq import GPTQWeight
            from text_generation_server.layers.marlin import (
                GPTQMarlin24Weight,
                MarlinWeight,
                repack_gptq_for_marlin,
            )

            quant_method = getattr(self, "quant_method", "marlin")
            is_marlin_24 = getattr(self, "gptq_checkpoint_format", None) == "marlin_24"
            if is_marlin_24:
                try:
                    B = self.get_sharded(f"{prefix}.B_24", dim=0)
                except RuntimeError:
                    raise RuntimeError(
                        "Cannot load `marlin` 2:4 sparsity weight, make sure the model is already quantized."
                    )

                B_meta = self.get_sharded(f"{prefix}.B_meta", dim=0)
                num_groups = self._get_slice(f"{prefix}.s").get_shape()[0]
                if num_groups == 1:
                    # The number of groups is 1 when groupsize == -1. share
                    # scales between all shards in this case.
                    s = self.get_tensor(f"{prefix}.s")
                else:
                    s = self.get_sharded(f"{prefix}.s", dim=0)

                gptq_params = self._get_gptq_params()
                weight = GPTQMarlin24Weight(
                    B=B, B_meta=B_meta, s=s, bits=gptq_params.bits
                )
            elif quant_method == "gptq":
                log_once(logger.info, "Converting GPTQ model to Marlin packing format.")
                gptq_params = self._get_gptq_params()

                try:
                    qweight = self.get_sharded(f"{prefix}.qweight", dim=0)
                except RuntimeError:
                    raise RuntimeError(
                        f"Cannot load `{quantize}` weight for GPTQ -> Marlin repacking, make sure the model is already quantized"
                    )

                g_idx = self.get_sharded(f"{prefix}.g_idx", dim=0)
                if gptq_params.desc_act or gptq_params.groupsize == -1:
                    scales = self.get_tensor(f"{prefix}.scales")
                else:
                    scales = self.get_sharded(f"{prefix}.scales", dim=0)

                sharded_in_features = self.process_group.size() > 1

                weight = repack_gptq_for_marlin(
                    qweight=qweight,
                    scales=scales,
                    g_idx=g_idx,
                    bits=gptq_params.bits,
                    desc_act=gptq_params.desc_act,
                    groupsize=gptq_params.groupsize,
                    sym=gptq_params.sym,
                    sharded_infeatures=sharded_in_features,
                )
            else:
                try:
                    B = self.get_sharded(f"{prefix}.B", dim=0)
                except RuntimeError:
                    raise RuntimeError(
                        "Cannot load `marlin` weight, make sure the model is already quantized."
                    )

                num_groups = self._get_slice(f"{prefix}.s").get_shape()[0]
                if num_groups == 1:
                    # The number of groups is 1 when groupsize == -1. share
                    # scales between all shards in this case.
                    s = self.get_tensor(f"{prefix}.s")
                else:
                    s = self.get_sharded(f"{prefix}.s", dim=0)
                weight = MarlinWeight(B=B, s=s)

        else:
            weight = self.get_sharded(f"{prefix}.weight", dim=1)
        return weight

    def _get_gptq_params(self) -> _GPTQParams:
        try:
            bits = self.get_tensor("gptq_bits").item()
            groupsize = self.get_tensor("gptq_groupsize").item()
            checkpoint_format = getattr(self, "gptq_checkpoint_format", None)
            desc_act = False
            sym = True
            quant_method = "gptq"
        except (SafetensorError, RuntimeError) as e:
            try:
                bits = self.gptq_bits
                groupsize = self.gptq_groupsize
                checkpoint_format = getattr(self, "gptq_checkpoint_format", None)
                desc_act = getattr(self, "gptq_desc_act", False)
                quant_method = getattr(self, "quant_method", "gptq")
                sym = getattr(self, "sym", True)
            except Exception:
                raise e

        return _GPTQParams(
            bits=bits,
            checkpoint_format=checkpoint_format,
            desc_act=desc_act,
            groupsize=groupsize,
            quant_method=quant_method,
            sym=sym,
        )

    def _set_gptq_params(self, model_id, revision):
        filename = "config.json"
        try:
            if os.path.exists(os.path.join(model_id, filename)):
                filename = os.path.join(model_id, filename)
            else:
                filename = hf_hub_download(
                    model_id, filename=filename, revision=revision
                )
            with open(filename, "r") as f:
                data = json.load(f)
            self.gptq_bits = data["quantization_config"]["bits"]
            self.gptq_groupsize = data["quantization_config"]["group_size"]
            # Order is important here, desc_act is missing on some real models
            self.quant_method = data["quantization_config"]["quant_method"]
            self.gptq_checkpoint_format = data["quantization_config"].get(
                "checkpoint_format"
            )
            self.gptq_sym = data["quantization_config"]["sym"]
            self.gptq_desc_act = data["quantization_config"]["desc_act"]
        except Exception:
            filename = "quantize_config.json"
            try:
                if os.path.exists(os.path.join(model_id, filename)):
                    filename = os.path.join(model_id, filename)
                else:
                    filename = hf_hub_download(
                        model_id, filename=filename, revision=revision
                    )
                with open(filename, "r") as f:
                    data = json.load(f)
                self.gptq_bits = data["bits"]
                self.gptq_groupsize = data["group_size"]
                self.gptq_sym = data["sym"]
                self.gptq_desc_act = data["desc_act"]
                if "version" in data and data["version"] == "GEMM":
                    self.quant_method = "awq"
            except Exception:
                filename = "quant_config.json"
                try:
                    if os.path.exists(os.path.join(model_id, filename)):
                        filename = os.path.join(model_id, filename)
                    else:
                        filename = hf_hub_download(
                            model_id, filename=filename, revision=revision
                        )
                    with open(filename, "r") as f:
                        data = json.load(f)
                    self.gptq_bits = data["w_bit"]
                    self.gptq_groupsize = data["q_group_size"]
                    self.gptq_desc_act = data["desc_act"]
                    if "version" in data and data["version"] == "GEMM":
                        self.quant_method = "awq"
                except Exception:
                    pass


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
