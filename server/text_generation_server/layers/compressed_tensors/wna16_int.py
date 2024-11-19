from typing import List, Union

import torch
from compressed_tensors.quantization import ActivationOrdering, QuantizationArgs
from loguru import logger

from text_generation_server.layers.marlin.gptq import repack_gptq_for_marlin
from text_generation_server.utils.log import log_once
from text_generation_server.utils.weights import Weights, WeightsLoader


class WNA16IntLoader(WeightsLoader):
    """
    Loader for W4A16/W8A16 INT compressed-tensors parameters.
    """

    def __init__(self, weights: QuantizationArgs):
        self.weights = weights
        self.desc_act = self.weights.actorder == ActivationOrdering.GROUP
        self.groupsize = (
            -1 if self.weights.group_size is None else self.weights.group_size
        )

    def __str__(self) -> str:
        quantization_type = f"W{self.weights.num_bits}A16"

        return f"{self.__class__.__name__} ({quantization_type})"

    def get_weights(self, weights: Weights, prefix: str):
        log_once(logger.info, "Using GPTQ-Marlin kernels")
        try:
            weight_packed = weights.get_tensor(f"{prefix}.weight_packed").t()
        except RuntimeError:
            raise RuntimeError(
                f"Cannot load w{self.weights.num_bits}a16 weight, make sure the model is already quantized"
            )

        zero_point = None
        if not self.weights.symmetric:
            zero_point = weights.get_tensor(f"{prefix}.weight_zero_point").t()

        g_idx = None
        if self.desc_act:
            g_idx = weights.get_tensor(f"{prefix}.weight_g_idx")

        scales = weights.get_tensor(f"{prefix}.weight.scales").t()

        return repack_gptq_for_marlin(
            qweight=weight_packed.contiguous(),
            scales=scales,
            qzeros=zero_point,
            g_idx=g_idx,
            bits=self.weights.num_bits,
            desc_act=self.desc_act,
            groupsize=self.groupsize,
            quant_method="compressed-tensors",
            sym=self.weights.symmetric,
            sharded_infeatures=False,
        )

    def get_weights_col_packed(
        self,
        weights: Weights,
        prefix: str,
        block_sizes: Union[int, List[int]],
    ):
        try:
            weight_packed = weights.get_packed_sharded(
                f"{prefix}.weight_packed", dim=0, block_sizes=block_sizes
            ).t()
        except RuntimeError:
            raise RuntimeError(
                f"Cannot load w{self.weights.num_bits}a16 weight, make sure the model is already quantized"
            )
        scales = weights.get_packed_sharded(
            f"{prefix}.weight_scale", dim=0, block_sizes=block_sizes
        ).t()
        scales = scales.to(dtype=weights.dtype)

        zero_point = None
        if not self.weights.symmetric:
            zero_point = weights.get_packed_sharded(
                f"{prefix}.qzeros", dim=0, block_sizes=block_sizes
            ).t()

        g_idx = None
        if self.desc_act:
            g_idx = weights.get_tensor(f"{prefix}.g_idx")

        return repack_gptq_for_marlin(
            qweight=weight_packed.contiguous(),
            scales=scales,
            qzeros=zero_point,
            g_idx=g_idx,
            bits=self.weights.num_bits,
            desc_act=self.desc_act,
            groupsize=self.groupsize,
            quant_method="compressed-tensors",
            sym=self.weights.symmetric,
            sharded_infeatures=False,
        )

    def get_multi_weights_col(self, weights: Weights, prefixes: List[str], dim: int):
        try:
            weight_packed = torch.cat(
                [
                    weights.get_sharded(f"{p}.weight_packed", dim=0).t()
                    for p in prefixes
                ],
                dim=1,
            )
        except RuntimeError:
            raise RuntimeError(
                f"Cannot load w{self.weights.num_bits}a16 weight, make sure the model is already quantized"
            )

        scales = torch.cat(
            [weights.get_sharded(f"{p}.weight_scale", dim=0).t() for p in prefixes],
            dim=1,
        )

        zero_point = None
        if not self.weights.symmetric:
            zero_point = torch.cat(
                [weights.get_sharded(f"{p}.qzeros", dim=0).t() for p in prefixes], dim=1
            ).t()

        g_idx = None
        if self.desc_act:
            w = [weights.get_tensor(f"{p}.g_idx") for p in prefixes]
            for w2 in w[1:]:
                torch.testing.assert_close(w2, w[0])
            g_idx = w[0]

        return repack_gptq_for_marlin(
            qweight=weight_packed.contiguous(),
            scales=scales,
            qzeros=zero_point,
            g_idx=g_idx,
            bits=self.weights.num_bits,
            desc_act=self.desc_act,
            groupsize=self.groupsize,
            quant_method="compressed-tensors",
            sym=self.weights.symmetric,
            sharded_infeatures=False,
        )

    def get_weights_row(self, weights: Weights, prefix: str):
        log_once(logger.info, "Using GPTQ-Marlin kernels")
        try:
            weight_packed = weights.get_sharded(f"{prefix}.weight_packed", dim=1).t()
        except RuntimeError:
            raise RuntimeError(
                f"Cannot load `{self.quantize}` weight, make sure the model is already quantized."
            )

        zero_point = None
        if not self.weights.symmetric:
            if self.desc_act or self.groupsize == -1:
                zero_point = weights.get_tensor(f"{prefix}.weight_zero_point").t()
            else:
                zero_point = weights.get_sharded(
                    f"{prefix}.weight_zero_point", dim=1
                ).t()

        g_idx = None
        if self.desc_act:
            g_idx = weights.get_sharded(f"{prefix}.g_idx", dim=0)

        if self.desc_act or self.groupsize == -1:
            scales = weights.get_tensor(f"{prefix}.weight_scale").t()
        else:
            scales = weights.get_sharded(f"{prefix}.weight_scale", dim=1).t()

        sharded_in_features = weights.process_group.size() > 1

        return repack_gptq_for_marlin(
            qweight=weight_packed.contiguous(),
            scales=scales,
            qzeros=zero_point,
            g_idx=g_idx,
            bits=self.weights.num_bits,
            desc_act=self.desc_act,
            groupsize=self.groupsize,
            quant_method="compressed-tensors",
            sym=self.weights.symmetric,
            sharded_infeatures=sharded_in_features,
        )
