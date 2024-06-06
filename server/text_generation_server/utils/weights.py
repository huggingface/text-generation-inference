from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple, Union
from safetensors import safe_open, SafetensorError
import torch
from loguru import logger
from huggingface_hub import hf_hub_download
import json
from text_generation_server.utils.log import log_once


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

    def _get_qweight(self, name: str, blocks: int):
        slice_ = self._get_slice(name)
        total_size = slice_.get_shape()[1]
        assert (
            total_size % blocks == 0
        ), f"Prepacked quantized matrix is not divisible by {blocks}"
        single_size = total_size // blocks
        world_size = self.process_group.size()
        rank = self.process_group.rank()

        assert (
            single_size % world_size == 0
        ), f"Prepacked quantized matrix cannot be sharded across {world_size} shards"
        block_size = single_size // world_size
        start = rank * block_size
        stop = (rank + 1) * block_size

        weights = []
        for block in range(blocks):
            weights.append(
                slice_[:, start + block * single_size : stop + block * single_size]
            )

        weight = torch.cat(weights, dim=1)
        weight = weight.to(device=self.device)
        return weight

    def get_weights_col_packed_qkv(self, prefix: str, quantize: str):
        return self.get_weights_col_packed(prefix, quantize, 3)

    def get_weights_col_packed_gate_up(self, prefix: str, quantize: str):
        return self.get_weights_col_packed(prefix, quantize, 2)

    def get_weights_col_packed(self, prefix: str, quantize: str, blocks: int):
        """
        Highly specific when the underlying tensor is a simple cat of Q,K,V instead of being
        already alternating Q,K,V within the main tensor
        """
        if quantize in ["gptq", "awq"]:
            from text_generation_server.layers.gptq import GPTQWeight

            try:
                qweight = self._get_qweight(f"{prefix}.qweight", blocks)
            except RuntimeError:
                raise RuntimeError(
                    f"Cannot load `{quantize}` weight, make sure the model is already quantized."
                )

            bits, groupsize, _, quant_method = self._get_gptq_params()

            qzeros = self._get_qweight(f"{prefix}.qzeros", blocks)
            scales = self._get_qweight(f"{prefix}.scales", blocks)
            scales = scales.to(dtype=self.dtype)

            if quantize == "gptq" and quant_method == "gptq":
                g_idx = self.get_tensor(f"{prefix}.g_idx")
            elif quantize == "gptq" and quant_method == "awq":
                log_once(
                    logger.info, "Converting AWQ model to Exllama/GPTQ packing format."
                )
                from text_generation_server.layers.awq.conversion_utils import (
                    fast_awq_to_gptq,
                )

                qweight, qzeros = fast_awq_to_gptq(qweight, qzeros)
                g_idx = (
                    torch.arange(qweight.shape[0] * (32 // bits), device=qweight.device)
                    // groupsize
                ).to(dtype=torch.int32)
            else:
                g_idx = None

            weight = GPTQWeight(
                qweight=qweight,
                qzeros=qzeros,
                scales=scales,
                g_idx=g_idx,
                bits=bits,
                groupsize=groupsize,
                use_exllama=False,
            )
        elif quantize == "marlin":
            from text_generation_server.layers.marlin import MarlinWeight

            B = self._get_qweight(f"{prefix}.B", blocks)
            s = self._get_qweight(f"{prefix}.s", blocks)
            weight = MarlinWeight(B=B, s=s)
        else:
            slice_ = self._get_slice(f"{prefix}.weight")
            total_size = slice_.get_shape()[0]
            assert total_size % blocks == 0, f"Prepacked is not divisible by {blocks}"
            single_size = total_size // blocks
            world_size = self.process_group.size()
            rank = self.process_group.rank()

            assert (
                single_size % world_size == 0
            ), f"Prepacked qkv cannot be sharded across {world_size} shards"
            block_size = single_size // world_size
            start = rank * block_size
            stop = (rank + 1) * block_size
            tensors = []
            for i in range(blocks):
                tensor = slice_[start + i * single_size : stop + i * single_size]
                tensors.append(tensor)
            weight = torch.cat(tensors, dim=0)
            weight = weight.to(device=self.device)
            weight = weight.to(dtype=self.dtype)
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

            bits, groupsize, desc_act, quant_method = self._get_gptq_params()

            from text_generation_server.layers.gptq import HAS_EXLLAMA

            use_exllama = (
                bits == 4 and HAS_EXLLAMA and quantize == "gptq" and not desc_act
            )

            if quantize == "gptq" and quant_method == "gptq":
                w = [self.get_tensor(f"{p}.g_idx") for p in prefixes]
                for w2 in w[1:]:
                    torch.testing.assert_close(w2, w[0])
                g_idx = w[0]
            elif quantize == "gptq" and quant_method == "awq":
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
                            qweight.shape[0] * (32 // bits), device=qweight.device
                        )
                        // groupsize
                    ).to(dtype=torch.int32)
            else:
                g_idx = None

            weight = GPTQWeight(
                qweight=qweight,
                qzeros=qzeros,
                scales=scales,
                g_idx=g_idx,
                bits=bits,
                groupsize=groupsize,
                use_exllama=use_exllama,
            )
        elif quantize == "marlin":
            from text_generation_server.layers.marlin import MarlinWeight

            try:
                B = torch.cat(
                    [self.get_sharded(f"{p}.B", dim=1) for p in prefixes], dim=1
                )
            except RuntimeError:
                raise RuntimeError(
                    f"Cannot load `{quantize}` weight, make sure the model is already quantized"
                )
            s = torch.cat([self.get_sharded(f"{p}.s", dim=1) for p in prefixes], dim=1)

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
            bits, groupsize, desc_act, quant_method = self._get_gptq_params()

            if bits != 4:
                use_exllama = False

            if desc_act:
                log_once(logger.warning, "Disabling exllama because desc_act=True")
                use_exllama = False

            try:
                qweight = self.get_sharded(f"{prefix}.qweight", dim=0)
            except RuntimeError:
                raise RuntimeError(
                    "Cannot load `gptq` weight, make sure the model is already quantized, or quantize it with `text-generation-server quantize ORIGINAL_MODEL_ID NEW_MODEL_ID`"
                )

            if quant_method == "gptq":
                g_idx = self.get_sharded(f"{prefix}.g_idx", dim=0)
            elif quant_method == "awq":
                g_idx = None

            if self.process_group.size() > 1:
                if g_idx is not None:
                    if (
                        not torch.equal(
                            g_idx.cpu(),
                            torch.tensor(
                                [i // groupsize for i in range(g_idx.shape[0])],
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

            if use_exllama and groupsize != -1:
                qzeros = self.get_sharded(f"{prefix}.qzeros", dim=0)
                scales = self.get_sharded(f"{prefix}.scales", dim=0)
            else:
                qzeros = self.get_tensor(f"{prefix}.qzeros")
                scales = self.get_tensor(f"{prefix}.scales")

            if use_exllama and g_idx is not None:
                g_idx = g_idx - g_idx[0]

            if quant_method == "awq":
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
                            qweight.shape[0] * (32 // bits), device=qweight.device
                        )
                        // groupsize
                    ).to(dtype=torch.int32)

            weight = GPTQWeight(
                qweight=qweight,
                qzeros=qzeros,
                scales=scales,
                g_idx=g_idx,
                bits=bits,
                groupsize=groupsize,
                use_exllama=use_exllama,
            )
        elif quantize == "awq":
            from text_generation_server.layers.gptq import GPTQWeight

            bits, groupsize, _, _ = self._get_gptq_params()

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
                bits=bits,
                groupsize=groupsize,
                use_exllama=use_exllama,
            )
        elif quantize == "marlin":
            from text_generation_server.layers.marlin import MarlinWeight

            try:
                B = self.get_sharded(f"{prefix}.B", dim=0)
            except RuntimeError:
                raise RuntimeError(
                    "Cannot load `marlin` weight, make sure the model is already quantized, or quantize it with `text-generation-server quantize ORIGINAL_MODEL_ID NEW_MODEL_ID`"
                )

            num_groups = self._get_slice(f"{prefix}.s").get_shape()[0]
            if num_groups == 1:
                # The number of groups is 1 when group_size == -1. share
                # scales between all shards in this case.
                s = self.get_tensor(f"{prefix}.s")
            else:
                s = self.get_sharded(f"{prefix}.s", dim=0)
            weight = MarlinWeight(B=B, s=s)

        else:
            weight = self.get_sharded(f"{prefix}.weight", dim=1)
        return weight

    def _get_gptq_params(self) -> Tuple[int, int, int, str]:
        try:
            bits = self.get_tensor("gptq_bits").item()
            groupsize = self.get_tensor("gptq_groupsize").item()
            desc_act = False
            quant_method = "gptq"
        except (SafetensorError, RuntimeError) as e:
            try:
                bits = self.gptq_bits
                groupsize = self.gptq_groupsize
                desc_act = getattr(self, "gptq_desc_act", False)
                quant_method = getattr(self, "quant_method", "gptq")
            except Exception:
                raise e

        return bits, groupsize, desc_act, quant_method

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
