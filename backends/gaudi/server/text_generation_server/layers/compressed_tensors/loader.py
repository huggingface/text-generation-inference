from typing import Any, Dict, List, Union

from compressed_tensors import QuantizationConfig, QuantizationStatus
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import (
    QuantizationScheme,
    QuantizationType,
    find_name_or_class_matches,
)
from loguru import logger
from pydantic import ValidationError
from torch import nn

from text_generation_server.layers.compressed_tensors.w8an_fp import W8ANFpLoader
from text_generation_server.layers.compressed_tensors.w8a8_int import W8A8IntLoader
from text_generation_server.layers.compressed_tensors.wna16_int_24 import (
    WNA16Int24Loader,
)
from text_generation_server.layers.compressed_tensors.wna16_int import WNA16IntLoader
from text_generation_server.utils.log import log_once
from text_generation_server.utils.weights import (
    DefaultWeightsLoader,
    UnquantizedWeight,
    Weights,
    WeightsLoader,
)

# compressed-tensors can match modules as quantization targets. However,
# they need to be objects rather than classes or class names. Since we
# need to match `Linear` targets, make an instance that can be re-used.
_EMPTY_LINEAR: nn.Module = nn.Linear(0, 0)


class CompressedTensorsLoader(WeightsLoader):
    """Loader for checkpoints stored in the compressed-tensors format."""

    def __init__(self, config: Dict[str, Any]):
        quantization_config_raw = config.get("quantization_config")
        if quantization_config_raw is None:
            # `compression_config` was renamed to `quantization_config`; support
            # retained for backward compatibility.
            quantization_config_raw = config.get("compression_config")
        if quantization_config_raw is None:
            raise ValueError(
                "Checkpoint does not have compressed-tensors configuration"
            )

        try:
            quantization_config = QuantizationConfig.model_validate(
                quantization_config_raw
            )
        except ValidationError as e:
            raise ValueError("Cannot parse compressed-tensors configuration") from e

        if quantization_config.quantization_status not in (
            QuantizationStatus.COMPRESSED,
            QuantizationStatus.FROZEN,
        ):
            raise ValueError(
                f"Model quantization was not finished, status was: {quantization_config.quantization_status}"
            )

        self.ignore = (
            quantization_config.ignore if quantization_config.ignore is not None else []
        )
        self.loaders = self._get_target_loaders(quantization_config)

        for target, loader in self.loaders.items():
            log_once(
                logger.info,
                f"Using {loader} for compressed-tensors target '{target}'",
            )

    def get_weights(self, weights: Weights, prefix: str):
        loader = self._lookup_loader(prefix)
        return loader.get_weights(weights, prefix)

    def get_weights_col_packed(
        self,
        weights: "Weights",
        prefix: str,
        block_sizes: Union[int, List[int]],
    ):
        loader = self._lookup_loader(prefix)
        return loader.get_weights_col_packed(weights, prefix, block_sizes)

    def get_multi_weights_col(self, weights: Weights, prefixes: List[str], dim: int):
        loader = self._lookup_loader(prefixes[0])
        return loader.get_multi_weights_col(weights, prefixes, dim)

    def get_weights_row(self, weights: Weights, prefix: str):
        loader = self._lookup_loader(prefix)
        return loader.get_weights_row(weights, prefix)

    def _get_target_loaders(
        self, quantization_config: QuantizationConfig
    ) -> Dict[str, WeightsLoader]:
        """
        A compressed-tensors checkpoint can use different quantizations
        for different targets. This method returns a dictionary with a
        loader per target.
        """

        loaders: Dict[str, WeightsLoader] = {}

        format = quantization_config.format

        for group_name, group in quantization_config.config_groups.items():
            # The group configuration can be a string, but does that ever
            # happen in a serialized quantization config?
            assert isinstance(group, QuantizationScheme)

            loader = self._create_loader_for_group(format, group_name, group)

            # A quantized parameter group can have multiple targets, add the
            # loader for all the targets.
            for target in group.targets:
                if target in loaders:
                    raise ValueError(
                        f"Target '{target} has multiple configured loaders'"
                    )
                loaders[target] = loader

        return loaders

    def _create_loader_for_group(
        self, format: str, group_name: str, group: QuantizationScheme
    ) -> WeightsLoader:
        """
        Find and create a loader for the group with the given quantization
        scheme.
        """
        # NOTE: we ignore group.output_activations because we don't support
        #       output quantization yet.

        input_activations = group.input_activations
        weights = group.weights
        if (
            format
            in {
                CompressionFormat.float_quantized.value,
                CompressionFormat.naive_quantized.value,
            }
            and weights is not None
            and weights.type == QuantizationType.FLOAT
            and weights.num_bits == 8
        ):
            # FP W8A8 or W8A16.
            return W8ANFpLoader(input_activations=input_activations, weights=weights)
        elif (
            format == CompressionFormat.pack_quantized.value
            and weights is not None
            and weights.type == QuantizationType.INT
            and weights.num_bits in (4, 8)
        ):
            # INT W4A16 or W8A16 (GPTQ/AWQ-like).
            return WNA16IntLoader(weights)
        elif (
            format == CompressionFormat.marlin_24.value
            and weights is not None
            and weights.type == QuantizationType.INT
            and weights.num_bits in (4, 8)
        ):
            return WNA16Int24Loader(weights)
        elif (
            format
            in {
                CompressionFormat.int_quantized.value,
                CompressionFormat.naive_quantized.value,
            }
            and weights is not None
            and weights.type == QuantizationType.INT
            and weights.num_bits == 8
        ):
            return W8A8IntLoader(input_args=input_activations, weight_args=weights)
        else:
            raise ValueError(
                f"Group '{group_name}' has unsupported compressed-tensors configurtion"
            )

    def _lookup_loader(self, prefix: str) -> WeightsLoader:
        """
        Look up the loader to use for a given parameter name (prefix).
        """

        if len(find_name_or_class_matches(prefix, _EMPTY_LINEAR, self.ignore)) > 0:
            return DefaultWeightsLoader(UnquantizedWeight)

        # We currently only handle linear layers, so unconditionally pass
        # a `Linear` instance.
        targets = find_name_or_class_matches(prefix, _EMPTY_LINEAR, self.loaders.keys())
        if len(targets) == 0:
            raise ValueError(
                f"Cannot find compressed-tensors target for prefix: {prefix}"
            )
        return self.loaders[targets[0]]
