from dataclasses import dataclass
from typing import List, Optional, Union

import torch
from loguru import logger
from text_generation_server.utils.log import log_once
from text_generation_server.utils.weights import Weight, Weights, WeightsLoader


from .hpu import QuantLinear


@dataclass
class GPTQWeight(Weight):
    qweight: torch.Tensor
    qzeros: torch.Tensor
    scales: torch.Tensor
    g_idx: Optional[torch.Tensor]
    bits: int
    groupsize: int
    use_awq_kernel: bool
    use_exllama: bool

    def __post_init__(self):
        if self.scales.dtype == torch.float:
            self.scales = self.scales.half()

    @property
    def device(self) -> torch.device:
        return self.qweight.device

    def get_linear(self, bias: torch.Tensor):
        if self.use_awq_kernel:
            try:
                from text_generation_server.layers.awq.quantize import WQLinear

                return WQLinear(
                    w_bit=self.bits,
                    group_size=self.groupsize,
                    qweight=self.qweight,
                    qzeros=self.qzeros,
                    scales=self.scales,
                    bias=bias,
                )
            except ImportError:
                raise NotImplementedError(
                    "You do not seem to have awq installed, either install it (cd server &&  make install-awq), or try using GPTQ `---quantize gptq` a conversion AWQ->GPTQ will happen on the fly"
                )
        else:
            return QuantLinear(
                self.qweight,
                self.qzeros,
                self.scales,
                self.g_idx,
                bias,
                self.bits,
                self.groupsize,
            )


class GPTQWeightsLoader(WeightsLoader):
    """
    Loader for GPTQ- and AWQ-quantized weights.
    """

    def __init__(
        self,
        *,
        bits: int,
        desc_act: bool,
        groupsize: int,
        quant_method: str,
        quantize: str,
        sym: bool,
    ):
        self.bits = bits
        self.desc_act = desc_act
        self.groupsize = groupsize
        self.quant_method = quant_method
        self.quantize = quantize
        self.sym = sym

    def get_weights(self, weights: Weights, prefix: str):
        self._get_gptq_params(weights)

        use_exllama = True
        if self.bits != 4:
            use_exllama = False

        if self.desc_act:
            log_once(logger.warning, "Disabling exllama because desc_act=True")
            use_exllama = False

        try:
            qweight = weights.get_tensor(f"{prefix}.qweight")
        except RuntimeError:
            raise RuntimeError(
                "Cannot load `gptq` weight, make sure the model is already quantized, or quantize it with `text-generation-server quantize ORIGINAL_MODEL_ID NEW_MODEL_ID`"
            )

        if self.quantize == "gptq" and self.quant_method == "gptq":
            g_idx = weights.get_tensor(f"{prefix}.g_idx")
        else:
            g_idx = None

        qzeros = weights.get_tensor(f"{prefix}.qzeros")
        scales = weights.get_tensor(f"{prefix}.scales")

        if use_exllama and g_idx is not None:
            g_idx = g_idx - g_idx[0]

        if self.quantize == "gptq" and self.quant_method == "awq":
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
                        qweight.shape[0] * (32 // self.bits),
                        device=qweight.device,
                    )
                    // self.groupsize
                ).to(dtype=torch.int32)

        return GPTQWeight(
            qweight=qweight,
            qzeros=qzeros,
            scales=scales,
            g_idx=g_idx,
            bits=self.bits,
            groupsize=self.groupsize,
            use_exllama=use_exllama,
        )

    def get_weights_col_packed(
        self,
        weights: Weights,
        prefix: str,
        block_sizes: Union[int, List[int]],
    ):
        try:
            qweight = weights.get_packed_sharded(
                f"{prefix}.qweight", dim=1, block_sizes=block_sizes
            )
        except RuntimeError:
            raise RuntimeError(
                f"Cannot load `{self.quantize}` weight, make sure the model is already quantized."
            )
        scales = weights.get_packed_sharded(
            f"{prefix}.scales", dim=1, block_sizes=block_sizes
        )
        scales = scales.to(dtype=weights.dtype)

        self._get_gptq_params(weights)

        qzeros = weights.get_packed_sharded(
            f"{prefix}.qzeros", dim=1, block_sizes=block_sizes
        )
        if self.quantize == "gptq" and self.quant_method == "gptq":
            g_idx = weights.get_tensor(f"{prefix}.g_idx")
        elif self.quantize == "gptq" and self.quant_method == "awq":
            log_once(
                logger.info, "Converting AWQ model to Exllama/GPTQ packing format."
            )
            from text_generation_server.layers.awq.conversion_utils import (
                fast_awq_to_gptq,
            )

            qweight, qzeros = fast_awq_to_gptq(qweight, qzeros)
            g_idx = (
                torch.arange(
                    qweight.shape[0] * (32 // self.bits),
                    device=qweight.device,
                )
                // self.groupsize
            ).to(dtype=torch.int32)
        else:
            g_idx = None

        return GPTQWeight(
            qweight=qweight,
            qzeros=qzeros,
            scales=scales,
            g_idx=g_idx,
            bits=self.bits,
            groupsize=self.groupsize,
            use_awq_kernel=self.quantize == "awq",
            use_exllama=False,
        )

    def get_multi_weights_col(self, weights: Weights, prefixes: List[str], dim: int):
        try:
            qweight = torch.cat(
                [weights.get_sharded(f"{p}.qweight", dim=1) for p in prefixes], dim=1
            )
        except RuntimeError:
            raise RuntimeError(
                f"Cannot load `{self.quantize}` weight, make sure the model is already quantized"
            )

        scales = torch.cat(
            [weights.get_sharded(f"{p}.scales", dim=1) for p in prefixes], dim=1
        )

        self._get_gptq_params(weights)

        qzeros = torch.cat(
            [weights.get_sharded(f"{p}.qzeros", dim=1) for p in prefixes], dim=1
        )

        use_exllama = self.bits == 4 and self.quantize == "gptq" and not self.desc_act

        if self.quantize == "gptq" and self.quant_method == "gptq":
            w = [weights.get_tensor(f"{p}.g_idx") for p in prefixes]
            for w2 in w[1:]:
                torch.testing.assert_close(w2, w[0])
            g_idx = w[0]
        elif self.quantize == "gptq" and self.quant_method == "awq":
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
                        qweight.shape[0] * (32 // self.bits),
                        device=qweight.device,
                    )
                    // self.groupsize
                ).to(dtype=torch.int32)
        else:
            g_idx = None

        return GPTQWeight(
            qweight=qweight,
            qzeros=qzeros,
            scales=scales,
            g_idx=g_idx,
            bits=self.bits,
            groupsize=self.groupsize,
            use_awq_kernel=self.quantize == "awq",
            use_exllama=use_exllama,
        )

    def get_weights_row(self, weights: Weights, prefix: str):
        self._get_gptq_params(weights)

        use_exllama = True
        desc_act = self.desc_act
        if self.bits != 4:
            use_exllama = False

        if self.desc_act:
            log_once(logger.warning, "Disabling exllama because desc_act=True")
            use_exllama = False

        try:
            qweight = weights.get_sharded(f"{prefix}.qweight", dim=0)
        except RuntimeError:
            raise RuntimeError(
                "Cannot load `gptq` weight, make sure the model is already quantized, or quantize it with `text-generation-server quantize ORIGINAL_MODEL_ID NEW_MODEL_ID`"
            )

        if self.quantize == "gptq" and self.quant_method == "gptq":
            g_idx = weights.get_sharded(f"{prefix}.g_idx", dim=0)
        else:
            g_idx = None

        if weights.process_group.size() > 1:
            if g_idx is not None:
                if (
                    not torch.equal(
                        # Remove g_idx[0] to adapt the check with TP>1.
                        (g_idx - g_idx[0]).cpu(),
                        torch.tensor(
                            [i // self.groupsize for i in range(g_idx.shape[0])],
                            dtype=torch.int32,
                        ),
                    )
                    and not (g_idx == 0).all()
                ):
                    # Exllama implementation does not support row tensor parallelism with act-order, as
                    # it would require to reorder input activations that are split unto several GPUs
                    use_exllama = False
                    desc_act = True

        from text_generation_server.layers.gptq import (
            GPTQWeight,
        )

        if not desc_act and self.groupsize != -1:
            qzeros = weights.get_sharded(f"{prefix}.qzeros", dim=0)
            scales = weights.get_sharded(f"{prefix}.scales", dim=0)
            if g_idx is not None:
                # qzeros, scales sharded, and g_idx must be adjusted accordingly
                g_idx = g_idx - g_idx[0]
        else:
            qzeros = weights.get_tensor(f"{prefix}.qzeros")
            scales = weights.get_tensor(f"{prefix}.scales")

        if self.quantize == "gptq" and self.quant_method == "awq":
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
                        qweight.shape[0] * (32 // self.bits),
                        device=qweight.device,
                    )
                    // self.groupsize
                ).to(dtype=torch.int32)

        return GPTQWeight(
            qweight=qweight,
            qzeros=qzeros,
            scales=scales,
            g_idx=g_idx,
            bits=self.bits,
            groupsize=self.groupsize,
            use_awq_kernel=self.quantize == "awq",
            use_exllama=use_exllama,
        )

    def _get_gptq_params(self, weights: Weights):
        if weights.has_tensor("gptq_bits") and weights.has_tensor("gptq_groupsize"):
            self.bits = weights.get_tensor("gptq_bits").item()
            self.groupsize = weights.get_tensor("gptq_groupsize").item()
            self.desc_act = False
            # `server quantize` used asymmetric quantization unconditionally
            # before the `gptq_sym` setting tensor was added.
            self.sym = (
                weights.get_tensor("gptq_sym").item()
                if weights.has_tensor("gptq_sym")
                else False
            )
            self.quant_method = "gptq"
