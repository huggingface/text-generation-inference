from pathlib import Path
from typing import List, Dict, Optional, Tuple
from safetensors import safe_open, SafetensorError
import torch
from loguru import logger
from huggingface_hub import hf_hub_download
import json


class Weights:
    def __init__(
        self,
        filenames: List[Path],
        device,
        dtype,
        process_group,
        aliases: Optional[Dict[str, List[str]]] = None,
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
        self._handles = {}

    def _get_handle(self, filename):
        if filename not in self._handles:
            f = safe_open(filename, framework="pytorch")
            self._handles[filename] = f

        return self._handles[filename]

    def get_filename(self, tensor_name: str) -> (str, str):
        filename = self.routing.get(tensor_name, None)
        if filename is None:
            aliases = self.aliases.get(tensor_name, [])
            for alias in aliases:
                filename = self.routing.get(alias, None)
                if filename is not None:
                    return str(filename), alias
            raise RuntimeError(f"weight {tensor_name} does not exist")
        return str(filename), tensor_name

    def _get_slice(self, tensor_name: str):
        filename, tensor_name = self.get_filename(tensor_name)
        f = self._get_handle(filename)
        slice_ = f.get_slice(tensor_name)
        return slice_

    def get_shape(self, tensor_name: str):
        return self._get_slice(tensor_name).get_shape()

    def get_tensor(self, tensor_name: str):
        filename, tensor_name = self.get_filename(tensor_name)
        f = self._get_handle(filename)
        tensor = f.get_tensor(tensor_name)
        # Special case for gptq which shouldn't convert
        # u4 which are disguised as int32
        if tensor.dtype not in [torch.int32, torch.int64]:
            tensor = tensor.to(dtype=self.dtype)
        tensor = tensor.to(device=self.device)
        return tensor

    def get_partial_sharded(self, tensor_name: str, dim: int):
        filename, tensor_name = self.get_filename(tensor_name)
        world_size = self.process_group.size()
        rank = self.process_group.rank()

        f = self._get_handle(filename)
        slice_ = f.get_slice(tensor_name)
        size = slice_.get_shape()[dim]
        block_size = size // world_size
        start = rank * block_size
        stop = (rank + 1) * block_size

        if dim == 0:
            tensor = slice_[start:stop]
        elif dim == 1:
            tensor = slice_[:, start:stop]
        else:
            raise NotImplementedError("Let's make that generic when needed")
        # Special case for gptq which shouldn't convert
        # u4 which are disguised as int32
        if tensor.dtype != torch.int32:
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

    def get_multi_weights_col(self, prefixes: List[str], quantize: str, dim: int):
        if quantize == "gptq":
            try:
                qweight = torch.cat(
                    [self.get_sharded(f"{p}.qweight", dim=1) for p in prefixes], dim=1
                )
            except RuntimeError:
                raise RuntimeError(
                    "Cannot load `gptq` weight, make sure the model is already quantized, or quantize it with `text-generation-server quantize ORIGINAL_MODEL_ID NEW_MODEL_ID`"
                )

            qzeros = torch.cat(
                [self.get_sharded(f"{p}.qzeros", dim=1) for p in prefixes], dim=1
            )
            scales = torch.cat(
                [self.get_sharded(f"{p}.scales", dim=1) for p in prefixes], dim=1
            )
            w = [self.get_tensor(f"{p}.g_idx") for p in prefixes]
            for w2 in w[1:]:
                torch.testing.assert_close(w2, w[0])
            g_idx = w[0]

            bits, groupsize = self._get_gptq_params()
            weight = (qweight, qzeros, scales, g_idx, bits, groupsize, False)
        else:
            w = [self.get_sharded(f"{p}.weight", dim=0) for p in prefixes]
            weight = torch.cat(w, dim=dim)
        return weight

    def get_multi_weights_row(self, prefix: str, quantize: str):
        if quantize == "gptq":
            use_exllama = True
            bits, groupsize = self._get_gptq_params()

            if bits != 4:
                use_exllama = False

            if self.process_group.size() > 1:
                g_idx = self.get_tensor(f"{prefix}.g_idx")
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

            try:
                qweight = self.get_sharded(f"{prefix}.qweight", dim=0)
            except RuntimeError:
                raise RuntimeError(
                    "Cannot load `gptq` weight, make sure the model is already quantized, or quantize it with `text-generation-server quantize ORIGINAL_MODEL_ID NEW_MODEL_ID`"
                )

            from text_generation_server.utils.layers import HAS_EXLLAMA

            if use_exllama:
                if not HAS_EXLLAMA:
                    logger.warning(
                        "Exllama GPTQ cuda kernels (which are faster) could have been used, but are not currently installed, try using BUILD_EXTENSIONS=True"
                    )
                    use_exllama = False
                else:
                    logger.info("Using exllama kernels")

            if use_exllama:
                if groupsize >= 0:
                    # Exllama reorders the weights in advance and the activations on the fly, thus
                    # the scales and zero-points do not need to be reordered.
                    qzeros = self.get_sharded(f"{prefix}.qzeros", dim=0)
                    scales = self.get_sharded(f"{prefix}.scales", dim=0)
                else:
                    qzeros = self.get_tensor(f"{prefix}.qzeros")
                    scales = self.get_tensor(f"{prefix}.scales")

                # For tp > 1, at this point we know we do not use act-order
                if self.process_group.size() == 1:
                    g_idx = self.get_tensor(f"{prefix}.g_idx")
                else:
                    g_idx = None
            else:
                # The triton kernel reorders the scales/zero points instead of the weight/activation.
                # Thus, each rank needs the full qzeros/scales.
                qzeros = self.get_tensor(f"{prefix}.qzeros")
                scales = self.get_tensor(f"{prefix}.scales")
                g_idx = self.get_sharded(f"{prefix}.g_idx", dim=0)

            weight = (qweight, qzeros, scales, g_idx, bits, groupsize, use_exllama)
        else:
            weight = self.get_sharded(f"{prefix}.weight", dim=1)
        return weight

    def _get_gptq_params(self) -> Tuple[int, int]:
        try:
            bits = self.get_tensor("gptq_bits").item()
            groupsize = self.get_tensor("gptq_groupsize").item()
        except (SafetensorError, RuntimeError) as e:
            try:
                bits = self.gptq_bits
                groupsize = self.gptq_groupsize
            except Exception:
                raise e

        return bits, groupsize

    def _set_gptq_params(self, model_id):
        try:
            filename = hf_hub_download(model_id, filename="quantize_config.json")
            with open(filename, "r") as f:
                data = json.load(f)
            self.gptq_bits = data["bits"]
            self.gptq_groupsize = data["group_size"]
        except Exception:
            pass
