from typing import List, Union

import torch


from compressed_tensors.quantization import QuantizationArgs, QuantizationType
from text_generation_server.layers.marlin.marlin import GPTQMarlin24Weight
from text_generation_server.utils.weights import Weights, WeightsLoader


class WNA16Int24Loader(WeightsLoader):
    """
    Loader for W4A16/W8A16 INT 2:4 sparsity compressed-tensors checkpoints.
    """

    def __init__(self, weight_args: QuantizationArgs):
        super().__init__()

        if weight_args.type != QuantizationType.INT:
            raise ValueError(
                f"{type(self).__name__} only supports wNa8 int checkpoints"
            )

        if weight_args.strategy == "group" and weight_args.group_size is None:
            raise ValueError("`group_size` must be set when `actorder` is `group`")

        self.bits = weight_args.num_bits
        self.group_size = weight_args.group_size

    def __str__(self) -> str:
        quantization_type = f"W{self.bits}A16 2:4 sparsity"

        return f"{self.__class__.__name__} ({quantization_type})"

    def get_weights(self, weights: Weights, prefix: str):
        """
        Get weights at the given prefix and apply without tensor paralllism.
        """
        weight_packed = weights.get_tensor(f"{prefix}.weight_packed")
        meta = weights.get_tensor(f"{prefix}.meta")
        scale_packed = weights.get_tensor(f"{prefix}.scale_packed")
        return GPTQMarlin24Weight(
            weight_packed=weight_packed,
            meta=meta,
            scale_packed=scale_packed,
            bits=self.bits,
        )

    def get_weights_col_packed(
        self,
        weights: Weights,
        prefix: str,
        block_sizes: Union[int, List[int]],
    ):
        weight_packed = weights.get_packed_sharded(
            f"{prefix}.weight_packed", dim=1, block_sizes=block_sizes
        )
        meta = weights.get_packed_sharded(
            f"{prefix}.meta", dim=1, block_sizes=block_sizes
        )
        scale_packed = weights.get_packed_sharded(
            f"{prefix}.scale_packed", dim=1, block_sizes=block_sizes
        )
        return GPTQMarlin24Weight(
            weight_packed=weight_packed,
            meta=meta,
            scale_packed=scale_packed,
            bits=self.bits,
        )

    def get_multi_weights_col(self, weights: Weights, prefixes: List[str], dim: int):
        weight_packed = torch.cat(
            [weights.get_sharded(f"{p}.weight_packed", dim=1) for p in prefixes], dim=1
        )
        meta = torch.cat(
            [weights.get_sharded(f"{p}.meta", dim=1) for p in prefixes], dim=1
        )
        scale_packed = torch.cat(
            [weights.get_sharded(f"{p}.scale_packed", dim=1) for p in prefixes], dim=1
        )
        return GPTQMarlin24Weight(
            weight_packed=weight_packed,
            meta=meta,
            scale_packed=scale_packed,
            bits=self.bits,
        )

    def get_weights_row(self, weights: Weights, prefix: str):
        weight_packed = weights.get_sharded(f"{prefix}.weight_packed", dim=0)
        meta = weights.get_sharded(f"{prefix}.meta", dim=0)
        if self.group_size is None:
            scale_packed = weights.get_tensor(f"{prefix}.scale_packed")
        else:
            scale_packed = weights.get_sharded(f"{prefix}.scale_packed", dim=0)

        return GPTQMarlin24Weight(
            weight_packed=weight_packed,
            meta=meta,
            scale_packed=scale_packed,
            bits=self.bits,
        )
