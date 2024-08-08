from dataclasses import dataclass
from typing import Optional
import torch
from text_generation_server.utils.weights import Weight
from text_generation_server.utils.import_utils import SYSTEM


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
            if SYSTEM == "rocm":
                raise NotImplementedError(
                    "AWQ GEMM kernel can't be used on ROCm systems, please use `--quantize gptq` instead "
                    "to use Exllama/GPTQ kernels for AWQ inference."
                )
            try:
                from text_generation_server.layers.awq.quantize.qmodule import WQLinear

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
        elif self.use_exllama:
            try:
                from text_generation_server.layers.gptq import ExllamaQuantLinear
            except ImportError:
                raise NotImplementedError(
                    "Exllama gptq kernels are not installed. Install them `cd server/exllama_kernels && python setup.py install && cd ../exllamav2_kernels && python setup.py install`"
                )

            return ExllamaQuantLinear(self, bias)
        else:
            from text_generation_server.layers.gptq.quant_linear import QuantLinear

            return QuantLinear(
                self.qweight,
                self.qzeros,
                self.scales,
                self.g_idx,
                bias,
                self.bits,
                self.groupsize,
            )
