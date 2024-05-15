import torch
from torch.nn import functional as F
from text_generation_server.utils.import_utils import SYSTEM

if SYSTEM == "rocm":
    try:
        from vllm import _custom_C
    except Exception as e:
        raise ImportError(f"Could not load `vllm._custom_C`. Full error: {e}")

class FastLinear(torch.nn.Module):
    def __init__(
        self,
        weight,
        bias,
    ) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(weight)
        if bias is not None:
            self.bias = torch.nn.Parameter(bias)
        else:
            self.bias = None

    @classmethod
    def load(cls, config, prefix: str, weights, bias: bool):
        weight = weights.get_tensor(f"{prefix}.weight")
        if bias:
            bias = weights.get_tensor(f"{prefix}.bias")
        else:
            bias = None
        return cls(weight, bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight, self.bias)


class FastLinearROCm(torch.nn.Module):
    def __init__(
        self,
        weight,
        bias,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight)
        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

    @classmethod
    def load(cls, config, prefix: str, weights, bias: bool):
        weight = weights.get_tensor(f"{prefix}.weight")
        if bias:
            bias = weights.get_tensor(f"{prefix}.bias")
        else:
            bias = None
        return cls(weight, bias)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        bias = self.bias

        if SYSTEM == "rocm" and inp.numel() // inp.size(-1) == 1:
            batched = False

            if inp.dim() == 3:
                inp = inp.view(-1, inp.size(-1))
                batched = True

            m, k = weight.shape[0], inp.shape[1]
            out = torch.empty(
                inp.shape[0], weight.shape[0], dtype=inp.dtype, device="cuda"
            )
            if (k == 8192 and (m == 1280 or m == 7168)) or (k == 3584 and m == 8192):
                _custom_C.LLMM1(weight, inp, out, 8)
            elif k <= 8192 and k % 8 == 0 and m % 4 == 0:
                _custom_C.LLMM1(weight, inp, out, 4)
            else:
                out = F.linear(inp, weight)
            if batched:
                out = out.view(inp.shape[0], inp.shape[1], weight.shape[0])
            if bias is not None:
                out = out + bias
            return out
        return F.linear(inp, self.weight, self.bias)


def get_linear(weight, bias, quantize):
    if quantize is None:
        if SYSTEM == "rocm":
            linear = FastLinearROCm(weight, bias)
        else:
            linear = FastLinear(weight, bias)
    elif quantize == "eetq":
        try:
            from text_generation_server.layers.eetq import EETQLinear

            linear = EETQLinear(weight, bias)
        except ImportError:
            raise ImportError(
                "Please install EETQ from https://github.com/NetEase-FuXi/EETQ"
            )
    elif quantize == "fp8":
        from text_generation_server.layers.fp8 import Fp8Linear

        linear = Fp8Linear(weight, bias)
    elif quantize == "bitsandbytes":
        try:
            from text_generation_server.layers.bnb import (
                warn_deprecate_bnb,
                Linear8bitLt,
            )
        except ImportError:
            raise NotImplementedError(
                f"Bitsandbytes is missing install it with `pip install bitsandbytes`."
            )
        warn_deprecate_bnb()
        linear = Linear8bitLt(
            weight,
            bias,
            has_fp16_weights=False,
            threshold=6.0,
        )
        if bias is not None:
            linear.bias = nn.Parameter(bias)
    elif quantize == "bitsandbytes-fp4":
        try:
            from text_generation_server.layers.bnb import Linear4bit
        except ImportError:
            raise NotImplementedError(
                f"Bitsandbytes is missing install it with `pip install bitsandbytes`."
            )
        linear = Linear4bit(
            weight,
            bias,
            quant_type="fp4",
        )
    elif quantize == "bitsandbytes-nf4":
        try:
            from text_generation_server.layers.bnb import Linear4bit
        except ImportError:
            raise NotImplementedError(
                f"Bitsandbytes is missing install it with `pip install bitsandbytes`."
            )
        linear = Linear4bit(
            weight,
            bias,
            quant_type="nf4",
        )
    elif quantize == "gptq":
        try:
            qweight, qzeros, scales, g_idx, bits, groupsize, use_exllama = weight
        except Exception:
            raise NotImplementedError(
                f"The passed weight is not `gptq` compatible, loader needs to be updated."
            )

        if use_exllama:
            try:
                from text_generation_server.layers.gptq import (
                    ExllamaQuantLinear,
                )
            except ImportError:
                raise NotImplementedError(
                    f"Exllama gptq kernels are not installed. Install them `cd server/exllama_kernels && python setup.py install && cd ../exllamav2_kernels && python setup.py install`"
                )

            linear = ExllamaQuantLinear(
                qweight, qzeros, scales, g_idx, bias, bits, groupsize
            )
        else:
            from text_generation_server.layers.gptq.quant_linear import QuantLinear

            linear = QuantLinear(
                qweight,
                qzeros,
                scales,
                g_idx,
                bias,
                bits,
                groupsize,
            )
    elif quantize == "awq":
        try:
            qweight, qzeros, scales, _, bits, groupsize, _ = weight
        except Exception:
            raise NotImplementedError(
                f"The passed weight is not `awq` compatible, loader needs to be updated."
            )
        if SYSTEM == "rocm":
            raise NotImplementedError(
                "AWQ GEMM kernel can't be used on ROCm systems, please use `--quantize gptq` instead "
                "to use Exllama/GPTQ kernels for AWQ inference."
            )
        try:
            from text_generation_server.layers.awq.quantize.qmodule import WQLinear

            linear = WQLinear(
                w_bit=bits,
                group_size=groupsize,
                qweight=qweight,
                qzeros=qzeros,
                scales=scales,
                bias=bias is not None,
            )
        except ImportError:
            raise NotImplementedError(
                "You do not seem to have awq installed, either install it (cd server &&  make install-awq), or try using GPTQ `---quantize gptq` a conversion AWQ->GPTQ will happen on the fly"
            )
    else:
        raise NotImplementedError(f"Quantization `{quantize}` is not implemented yet.")
    return linear
