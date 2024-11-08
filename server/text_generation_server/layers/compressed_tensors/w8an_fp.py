from typing import List, Optional, Union

import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationType

from text_generation_server.layers.fp8 import Fp8Weight, _load_scalar_or_matrix_scale
from text_generation_server.utils.weights import Weights, WeightsLoader


class W8ANFpLoader(WeightsLoader):
    """
    Loader for W8A8/W8A16 FP compressed-tensors parameters.
    """

    def __init__(
        self,
        *,
        input_activations: Optional[QuantizationArgs],
        weights: QuantizationArgs,
    ):
        assert weights.type == QuantizationType.FLOAT and weights.num_bits == 8

        # We ignore the `strategy` option which sets the scales to be
        # per-tensor, per-channel or per-token. What scales are supported
        # is dependent on the kernels used (e.g. cutlass can do tokenwise,
        # Torch cannot, and FP8-Marlin does not quantize inputs at all).
        # So, instead we try to use the best-possible configuration.

        self.load_weight_scale = not weights.dynamic
        self.load_input_scale = (
            input_activations is not None and not input_activations.dynamic
        )
        self.force_w8a16 = (
            input_activations is not None and input_activations.num_bits == 16
        )

    def __str__(self) -> str:
        def scale_to_str(scale):
            return "static" if scale else "dynamic"

        quantization_type = f"W8A{16 if self.force_w8a16 else 8}"

        return f"{self.__class__.__name__} ({quantization_type}, weight: {scale_to_str(self.load_weight_scale)}, input: {scale_to_str(self.load_input_scale)})"

    def get_weights(self, weights: "Weights", prefix: str):
        w = weights.get_tensor(f"{prefix}.weight")

        weight_scale = None
        if self.load_weight_scale:
            weight_scale = (
                weights.get_tensor(f"{prefix}.weight_scale", to_dtype=False)
                .reshape(-1)
                .expand(w.shape[0])
            )

        input_scale = None
        if self.load_input_scale:
            input_scale = weights.get_tensor(
                f"{prefix}.input_scale", to_dtype=False
            ).reshape(-1)

        return Fp8Weight(
            weight=w,
            weight_scale=weight_scale,
            input_scale=input_scale,
            dtype=weights.dtype,
            force_w8a16=self.force_w8a16,
        )

    def get_weights_col_packed(
        self,
        weights: Weights,
        prefix: str,
        block_sizes: Union[int, List[int]],
    ):
        w = weights.get_packed_sharded(
            f"{prefix}.weight", dim=0, block_sizes=block_sizes
        )

        weight_scale = None
        if self.load_weight_scale:
            weight_scale = weights.get_tensor(f"{prefix}.weight_scale", to_dtype=False)
            if weight_scale.numel() > 1:
                weight_scale = weights.get_packed_sharded(
                    f"{prefix}.weight_scale",
                    dim=0,
                    block_sizes=block_sizes,
                    to_dtype=False,
                )
            weight_scale = weight_scale.reshape(-1).expand(w.shape[0])

        input_scale = None
        if self.load_input_scale:
            input_scale = weights.get_tensor(f"{prefix}.input_scale", to_dtype=False)
            if input_scale.numel() > 1:
                input_scale = weights.get_packed_sharded(
                    f"{prefix}.input_scale",
                    dim=0,
                    block_sizes=block_sizes,
                    to_dtype=False,
                )
            input_scale = input_scale.reshape(-1).max()

        return Fp8Weight(
            weight=w,
            weight_scale=weight_scale,
            input_scale=input_scale,
            dtype=weights.dtype,
            force_w8a16=self.force_w8a16,
        )

    def get_multi_weights_col(self, weights: "Weights", prefixes: List[str], dim: int):
        # FIXME: Force to_device to false as fp8 weights do not support torch.cat on device yet
        w = [
            weights.get_sharded(f"{p}.weight", dim=0, to_device=False) for p in prefixes
        ]
        shapes = [x.shape for x in w]

        # Concat then send to the device
        w = torch.cat(w, dim=dim).to(weights.device)

        weight_scale = None
        if self.load_weight_scale:
            weight_scale = [
                _load_scalar_or_matrix_scale(weights, f"{p}.weight_scale", shape)
                for p, shape in zip(prefixes, shapes)
            ]
            weight_scale = torch.cat(weight_scale, dim=0).reshape(-1)

        input_scale = None
        if self.load_input_scale:
            input_scale = [
                _load_scalar_or_matrix_scale(weights, f"{p}.input_scale", shape)
                for p, shape in zip(prefixes, shapes)
                if weights.has_tensor(f"{p}.input_scale")
            ]
            assert len(input_scale) == 0 or len(input_scale) == len(prefixes)
            input_scale = (
                torch.cat(input_scale, dim=0).reshape(-1).max()
                if len(input_scale) != 0
                else None
            )

        return Fp8Weight(
            weight=w,
            weight_scale=weight_scale,
            input_scale=input_scale,
            dtype=weights.dtype,
            force_w8a16=self.force_w8a16,
        )

    def get_weights_row(self, weights: "Weights", prefix: str):
        w = weights.get_sharded(f"{prefix}.weight", dim=1)
        weight_scale = None
        if self.load_weight_scale:
            weight_scale = (
                weights.get_tensor(f"{prefix}.weight_scale", to_dtype=False)
                .reshape(-1)
                .expand(w.shape[0])
            )

        input_scale = None
        if self.load_input_scale:
            input_scale = weights.get_tensor(
                f"{prefix}.input_scale", to_dtype=False
            ).reshape(-1)

        return Fp8Weight(
            weight=w,
            weight_scale=weight_scale,
            input_scale=input_scale,
            dtype=weights.dtype,
            force_w8a16=self.force_w8a16,
        )
