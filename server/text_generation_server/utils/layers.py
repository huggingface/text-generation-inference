import json
import os
from pathlib import Path

import torch
import torch.distributed

from torch import nn
from torch.nn import functional as F
from typing import List, Tuple, Optional
from loguru import logger
from functools import lru_cache

from text_generation_server.utils.speculate import get_speculate

HAS_BITS_AND_BYTES = True
try:
    import bitsandbytes as bnb
    from bitsandbytes.nn import Int8Params, Params4bit
except ImportError:
    HAS_BITS_AND_BYTES = False

from accelerate import init_empty_weights

from text_generation_server.utils.gptq.quant_linear import QuantLinear
from text_generation_server.utils.import_utils import (
    IS_CUDA_SYSTEM,
    IS_ROCM_SYSTEM,
    IS_XPU_SYSTEM,
)

if IS_XPU_SYSTEM:
    import intel_extension_for_pytorch as ipex

HAS_AWQ = True
try:
    from text_generation_server.utils.awq.quantize.qmodule import WQLinear
except ImportError:
    HAS_AWQ = False

try:
    major, _minor = torch.cuda.get_device_capability()
except Exception:
    major = 1

HAS_EXLLAMA = False
CAN_EXLLAMA = major >= 8 or IS_ROCM_SYSTEM
V2 = os.getenv("EXLLAMA_VERSION", "2") == "2"

if os.getenv("DISABLE_EXLLAMA") == "True":
    HAS_EXLLAMA = False
elif CAN_EXLLAMA:
    try:
        if V2:
            from text_generation_server.utils.gptq.exllamav2 import (
                QuantLinear as ExllamaQuantLinear,
                create_exllama_buffers,
                set_device,
            )

            HAS_EXLLAMA = "2"
        else:
            from text_generation_server.utils.gptq.exllama import (
                Ex4bitLinear as ExllamaQuantLinear,
                create_exllama_buffers,
                set_device,
            )

            HAS_EXLLAMA = "1"

    except ImportError:
        pass

HAS_EETQ = False
try:
    from EETQ import quant_weights, w8_a16_gemm

    HAS_EETQ = True
except ImportError:
    pass


# Monkey patching
@classmethod
def load_layer_norm(cls, prefix, weights, eps):
    weight = weights.get_tensor(f"{prefix}.weight")
    bias = weights.get_tensor(f"{prefix}.bias")
    with init_empty_weights():
        ln = cls(weight.shape, eps=eps)

    ln.weight = nn.Parameter(weight)
    ln.bias = nn.Parameter(bias)
    return ln


@classmethod
def load_layer_norm_no_bias(cls, prefix, weights, eps):
    weight = weights.get_tensor(f"{prefix}.weight")
    with init_empty_weights():
        ln = cls(weight.shape, eps=eps)

    ln.weight = nn.Parameter(weight)
    ln.bias = None
    return ln


@classmethod
def load_conv2d(cls, prefix, weights, in_channels, out_channels, kernel_size, stride):
    weight = weights.get_tensor(f"{prefix}.weight")
    bias = weights.get_tensor(f"{prefix}.bias")
    with init_empty_weights():
        conv2d = cls(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )

    conv2d.weight = nn.Parameter(weight)
    conv2d.bias = nn.Parameter(bias)
    return conv2d


@classmethod
def load_conv2d_no_bias(
    cls, prefix, weights, in_channels, out_channels, kernel_size, stride
):
    weight = weights.get_tensor(f"{prefix}.weight")
    with init_empty_weights():
        conv2d = cls(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )

    conv2d.weight = nn.Parameter(weight)
    conv2d.bias = None
    return conv2d


torch.nn.Conv2d.load = load_conv2d
torch.nn.Conv2d.load_no_bias = load_conv2d_no_bias
torch.nn.LayerNorm.load = load_layer_norm
torch.nn.LayerNorm.load_no_bias = load_layer_norm_no_bias


class FastLinear(nn.Module):
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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight, self.bias)


class EETQLinear(nn.Module):
    def __init__(
        self,
        weight,
        bias,
    ) -> None:
        super().__init__()
        device = weight.device
        if weight.dtype != torch.float16:
            weight = weight.to(dtype=torch.float16)
        weight = torch.t(weight).contiguous().cpu()
        weight, scale = quant_weights(weight, torch.int8, False)

        self.weight = weight.cuda(device)
        self.scale = scale.cuda(device)
        self.bias = bias.cuda(device) if bias is not None else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = w8_a16_gemm(input, self.weight, self.scale)
        output = output + self.bias if self.bias is not None else output
        return output


def fp8_quantize(weight, qdtype=torch.float8_e4m3fn):
    device = weight.device
    # weight, scale = quant_weights(weight, torch.int8, False)
    finfo = torch.finfo(qdtype)
    # Calculate the scale as dtype max divided by absmax
    scale = finfo.max / weight.abs().max().clamp(min=1e-12)
    # scale and clamp the tensor to bring it to
    # the representative range of float8 data type
    # (as default cast is unsaturated)
    qweight = (weight * scale).clamp(min=finfo.min, max=finfo.max)
    # Return both float8 data and the inverse scale (as float),
    # as both required as inputs to torch._scaled_mm
    qweight = qweight.to(qdtype)
    scale = scale.float().reciprocal()
    return qweight, scale


class Fp8Linear(nn.Module):
    def __init__(
        self,
        weight,
        bias,
    ) -> None:
        super().__init__()
        self.dtype = weight.dtype
        self.qweight, self.scale = fp8_quantize(weight)

        self.bias = bias if bias is not None else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        qinput, scale = fp8_quantize(input)
        output, _ = torch._scaled_mm(
            qinput,
            self.qweight.t(),
            out_dtype=self.dtype,
            scale_a=scale,
            scale_b=self.scale,
            bias=self.bias,
        )
        return output


class Linear8bitLt(nn.Module):
    def __init__(
        self,
        weight,
        bias,
        has_fp16_weights=True,
        memory_efficient_backward=False,
        threshold=0.0,
        index=None,
    ):
        super().__init__()
        assert (
            not memory_efficient_backward
        ), "memory_efficient_backward is no longer required and the argument is deprecated in 0.37.0 and will be removed in 0.39.0"
        self.state = bnb.MatmulLtState()
        self.index = index

        # Necessary for stacked layers
        self.state.threshold = threshold
        self.state.has_fp16_weights = has_fp16_weights
        self.state.memory_efficient_backward = memory_efficient_backward
        if threshold > 0.0 and not has_fp16_weights:
            self.state.use_pool = True

        self.weight = Int8Params(
            weight.data,
            has_fp16_weights=has_fp16_weights,
            requires_grad=has_fp16_weights,
        )
        self.weight.cuda(weight.device)
        self.bias = bias

    def init_8bit_state(self):
        self.state.CB = self.weight.CB
        self.state.SCB = self.weight.SCB
        self.weight.CB = None
        self.weight.SCB = None

    def forward(self, x: torch.Tensor):
        self.state.is_training = self.training
        if self.weight.CB is not None:
            self.init_8bit_state()

        # weights are cast automatically as Int8Params, but the bias has to be cast manually
        if self.bias is not None and self.bias.dtype != x.dtype:
            self.bias.data = self.bias.data.to(x.dtype)

        out = bnb.matmul(x, self.weight, bias=self.bias, state=self.state)

        if not self.state.has_fp16_weights:
            if self.state.CB is not None and self.state.CxB is not None:
                # we converted 8-bit row major to turing/ampere format in the first inference pass
                # we no longer need the row-major weight
                del self.state.CB
                self.weight.data = self.state.CxB
        return out


class Linear4bit(nn.Module):
    def __init__(self, weight, bias, quant_type):
        super().__init__()
        self.weight = Params4bit(
            weight.data,
            requires_grad=False,
            compress_statistics=True,
            quant_type=quant_type,
        )
        self.compute_dtype = None
        self.weight.cuda(weight.device)
        self.bias = bias

    def forward(self, x: torch.Tensor):
        # weights are cast automatically as Int8Params, but the bias has to be cast manually
        if self.bias is not None and self.bias.dtype != x.dtype:
            self.bias.data = self.bias.data.to(x.dtype)

        if getattr(self.weight, "quant_state", None) is None:
            print(
                "FP4 quantization state not initialized. Please call .cuda() or .to(device) on the LinearFP4 layer first."
            )
        inp_dtype = x.dtype
        if self.compute_dtype is not None:
            x = x.to(self.compute_dtype)

        bias = None if self.bias is None else self.bias.to(self.compute_dtype)
        out = bnb.matmul_4bit(
            x, self.weight.t(), bias=bias, quant_state=self.weight.quant_state
        )

        out = out.to(inp_dtype)

        return out


@lru_cache(1)
def warn_deprecate_bnb():
    logger.warning(
        "Bitsandbytes 8bit is deprecated, using `eetq` is a drop-in replacement, and has much better performnce"
    )


def get_linear(weight, bias, quantize):
    if quantize is None:
        linear = FastLinear(weight, bias)
    elif quantize == "eetq":
        if HAS_EETQ:
            linear = EETQLinear(weight, bias)
        else:
            raise ImportError(
                "Please install EETQ from https://github.com/NetEase-FuXi/EETQ"
            )
    elif quantize == "fp8":
        linear = Fp8Linear(weight, bias)
    elif quantize == "bitsandbytes":
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
        linear = Linear4bit(
            weight,
            bias,
            quant_type="fp4",
        )
    elif quantize == "bitsandbytes-nf4":
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
            linear = ExllamaQuantLinear(
                qweight, qzeros, scales, g_idx, bias, bits, groupsize
            )
        else:
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
        if IS_ROCM_SYSTEM:
            raise NotImplementedError(
                "AWQ GEMM kernel can't be used on ROCm systems, please use `--quantize gptq` instead "
                "to use Exllama/GPTQ kernels for AWQ inference."
            )
        if not HAS_AWQ:
            raise NotImplementedError(
                "You do not seem to have awq installed, either install it (cd server &&  make install-awq), or try using GPTQ `---quantize gptq` a conversion AWQ->GPTQ will happen on the fly"
            )
        linear = WQLinear(
            w_bit=bits,
            group_size=groupsize,
            qweight=qweight,
            qzeros=qzeros,
            scales=scales,
            bias=bias is not None,
        )
    else:
        raise NotImplementedError(f"Quantization `{quantize}` is not implemented yet.")
    return linear


class SuperLayer(nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.linear = linear

    def forward(self, x):
        return self.linear.forward(x)


class ResBlock(torch.nn.Module):
    def __init__(self, config, prefix, weights):
        super().__init__()
        self.linear = FastLinear.load(
            config, prefix=f"{prefix}.linear", weights=weights, bias=True
        )
        self.act = torch.nn.SiLU()

    def forward(self, x):
        return x + self.act(self.linear(x))

class MLPSpeculatorLayerNorm(nn.Module):
    """
    A L2 normalization implementation
    ...
    Args
    ----
    normalized_shape : int
        Dimensionality of input data (size of final tensor axis)
    elementwise_scale_weight : torch.Tensor
        learned scaling term after normalization?
    elementwise_shift_bias : torch.Tensor
        learned bias term after normalization?
    eps : float
        Safety term to prevent division by zero. Make sure the chosen value fits in the range of your encoding scheme (i.e. fp16 requires eps >= 6e-8).

    """

    def __init__(
        self,
        normalized_shape,
        elementwise_scale_weight: torch.Tensor,
        elementwise_shift_bias: torch.Tensor,
        eps=1e-06,
    ):
        super(MLPSpeculatorLayerNorm, self).__init__()
        self.normalized_shape = normalized_shape
        self.weight = nn.Parameter(elementwise_scale_weight)
        self.bias = nn.Parameter(elementwise_shift_bias)
        self.eps = eps

    def forward(self, x):
        xf = x
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        x = xf.type_as(x)
        x = self.weight * x
        x = x + self.bias
        return x

class MLPSpeculatorModel(torch.nn.Module):
    def __init__(self, config, prefix, weights):
        super().__init__()
        self.config = config
        self.n_predict = config.n_predict
        self.emb_dim = config.emb_dim
        inner_dim = config.inner_dim if config.inner_dim != 0 else config.emb_dim
        self.inner_dim = inner_dim
        self.config = config.vocab_size
        self.emb = nn.ModuleList(
            [TensorParallelEmbedding(f"{prefix}.emb.{i}", weights) for i in range(config.n_predict)]
        )
        self.proj = TensorParallelColumnLinear.load_multi(
            config,
            prefixes=[f"{prefix}.proj.{i}" for i in range(config.n_predict)],
            weights=weights,
            bias=False,
            dim=0
        )
        self.head = nn.ModuleList(
            [TensorParallelRowLinear.load(config, f"{prefix}.head.{i}", weights, bias=False) for i in range(config.n_predict)]
        )
        self.ln = nn.ModuleList(
            [
                MLPSpeculatorLayerNorm(
                    config.inner_dim,
                    elementwise_scale_weight=weights.get_tensor(f"{prefix}.ln.{i}.weight"),
                    elementwise_shift_bias=weights.get_tensor(f"{prefix}.ln.{i}.bias")
                )
                for i in range(config.n_predict)
            ]
        )

        # Weights ensure that state_0 accounts for 50% of state magnitude by final head in expectation
        self.state_weight = 0.5 ** (0.5 / self.n_predict)
        self.emb_weight = math.sqrt(1 - self.state_weight ** 2)
        self.activation = nn.GELU()

    def forward(self, state: torch.Tensor, ind: torch.Tensor, top_k_tokens_per_head: Optional[List[int]], num_candidates: int = 1):
        if top_k_tokens_per_head is None:
            top_k_tokens_per_head = self.config.top_k_tokens_per_head

        # k indicates # of candidates
        # h indicates # of generated tokens
        b = state.size(0)
        out = torch.empty(b, 1, 0, device=state.device).int()  # b k h
        log_probs = torch.zeros(b, 1, device=state.device)  # b k
        all_probs = torch.empty(b, 1, 0, self.vsize, device=state.device)  # b k h v
        assert (
                len(top_k_tokens_per_head) == self.n_predict
        ), f"You must provide a topk number for each head ({self.n_predict} heads, {len(top_k_tokens_per_head)} provided)"
        for i in range(self.n_predict):
            # Project and predict
            z = self.emb[i](ind)
            z = z.mul(self.emb_weight * math.sqrt(self.inner_dim / 2))  # b k d
            state = self.proj[i](state) * self.state_weight + z
            state = self.activation(self.ln[i](state))  # b k d
            _probs = F.log_softmax(self.head[i](state), dim=2)  # b k v
            probs, preds = _probs.topk(top_k_tokens_per_head[i], dim=2)  # b k k'

            # Update candidate set with new predictions
            out = out.unsqueeze(2).expand(-1, -1, top_k_tokens_per_head[i], -1)  # b k k' h
            out = torch.cat([out, preds.unsqueeze(3)], dim=3)  # b k k' h+1
            out = out.view(b, -1, i + 1)  # b kk' h+1

            # Update distribution set with new logits
            all_probs = torch.cat([all_probs, _probs.exp().unsqueeze(2)], dim=2)  # b k h+1 v
            all_probs = all_probs.repeat(1, top_k_tokens_per_head[i], 1, 1)  # b kk' h+1 v

            # Update state, log_probs and ind for new predictions
            state = state.unsqueeze(2).expand(-1, -1, top_k_tokens_per_head[i], -1)  # b k k' d
            state = state.reshape(b, -1, state.size(3))  # b kk' d
            ind = preds.view(b, -1)  # b kk'
            log_probs = log_probs.unsqueeze(2).expand(b, -1, top_k_tokens_per_head[i])  # b k k'
            log_probs = log_probs.add(probs).reshape(b, -1)  # b kk'

        # Take only top n best guesses
        best_guesses = log_probs.topk(num_candidates, dim=1)[1]  # b k
        return all_probs.gather(
            1, best_guesses[:, :, None, None].expand(-1, -1, self.n_predict, self.vsize)
        )  # b n h v


class MLPSpeculatorHead(nn.Module):
    def __init__(self, lm_head, mlp_speculator):
        super().__init__()
        self.lm_head = lm_head
        self.mlp_speculator = mlp_speculator

    def forward(
        self, input: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        logits = self.lm_head(input)
        # If we have too many tokens, we skip speculative logits
        if input.shape[0] > 128:
            return logits, None

        speculative_logits = self.mlp_speculator(input)
        return logits, speculative_logits

    @staticmethod
    def load(speculator_config, prefix: str, weights):
        from pathlib import Path
        from safetensors import safe_open

        speculator_path = speculator_config.use_speculator

        filename = str(Path(speculator_path) / "*.safetensors")

        routing = weights.routing
        with safe_open(filename, framework="pytorch") as f:
            for k in f.keys():
                if k in routing and routing[k] != filename:
                    raise RuntimeError(
                        f"Key {k} was found in multiple files: {filename} and {routing[k]}"
                    )
                routing[k] = filename

        mlp_speculator = MLPSpeculatorModel(speculator_config, "speculator", weights)
        lm_head = TensorParallelHead.load(speculator_config, prefix, weights)
        return MLPSpeculatorHead(lm_head, mlp_speculator)


class MedusaModel(torch.nn.Module):
    def __init__(self, config, medusa_config, weights):
        super().__init__()
        self.heads = torch.nn.ModuleList(
            [
                MedusaHead(config, medusa_config, prefix=f"{i}", weights=weights)
                for i in range(get_speculate())
            ]
        )

    def forward(self, x):
        speculative_logits = torch.stack([head(x) for head in self.heads], dim=1)
        return speculative_logits


class MedusaHead(torch.nn.Module):
    def __init__(self, config, medusa_config, prefix, weights):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [
                ResBlock(config, prefix=f"{prefix}.{i}", weights=weights)
                for i in range(medusa_config["medusa_num_layers"])
            ]
        )
        n = len(self.blocks)
        self.out = FastLinear.load(
            config, prefix=f"{prefix}.{n}", weights=weights, bias=False
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.out(x)
        return x


class MedusaHeadV1(nn.Module):
    def __init__(self, lm_head, medusa):
        super().__init__()
        self.lm_head = lm_head
        self.medusa = medusa

    @staticmethod
    def load(config, prefix: str, weights):
        from pathlib import Path
        from safetensors import safe_open
        import json

        use_medusa = config.use_medusa

        medusa_config = str(Path(use_medusa) / "config.json")
        filename = str(Path(use_medusa) / "medusa_lm_head.safetensors")

        with open(medusa_config, "r") as f:
            medusa_config = json.load(f)
        routing = weights.routing
        with safe_open(filename, framework="pytorch") as f:
            for k in f.keys():
                if k in routing and routing[k] != filename:
                    raise RuntimeError(
                        f"Key {k} was found in multiple files: {filename} and {routing[k]}"
                    )
                routing[k] = filename

        medusa = MedusaModel(config, medusa_config, weights)
        lm_head = TensorParallelHead.load(config, prefix, weights)
        return MedusaHeadV1(lm_head, medusa)

    def forward(
        self, input: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        logits = self.lm_head(input)
        # If we have too many tokens, we skip speculative logits
        if input.shape[0] > 128:
            return logits, None

        speculative_logits = self.medusa(input)
        return logits, speculative_logits


class MedusaHeadV2(nn.Module):
    def __init__(self, config, prefix, weights):
        super().__init__()
        from pathlib import Path
        from safetensors import safe_open
        import json

        use_medusa = config.use_medusa

        medusa_config = str(Path(use_medusa) / "config.json")
        filename = str(Path(use_medusa) / "medusa_lm_head.safetensors")

        with open(medusa_config, "r") as f:
            medusa_config = json.load(f)
        routing = weights.routing
        with safe_open(filename, framework="pytorch") as f:
            for k in f.keys():
                if k in routing and routing[k] != filename:
                    raise RuntimeError(
                        f"Key {k} was found in multiple files: {filename} and {routing[k]}"
                    )
                routing[k] = filename

        self.n_medusa_heads = get_speculate()

        assert medusa_config["medusa_num_layers"] == 1
        self.linear = TensorParallelColumnLinear.load_multi(
            config,
            prefixes=[f"{i}.0.linear" for i in range(self.n_medusa_heads)],
            dim=0,
            weights=weights,
            bias=True,
        )
        self.process_group = weights.process_group
        self.world_size = self.process_group.size()
        self.rank = self.process_group.rank()

        self.act = torch.nn.SiLU()

        self.lm_head = TensorParallelHead.load(config, prefix, weights)

    def forward(self, x):
        # If we have too many tokens, we skip speculative logits
        if x.shape[0] > 128:
            logits = self.lm_head(x)
            return logits, None

        size = x.shape[-1]
        block_size = (size + self.world_size - 1) // self.world_size
        start = self.rank * block_size
        stop = (self.rank + 1) * block_size

        x_block = x[:, start:stop]

        # Compute all medusa heads at the same time, then reshape and move the n_medusa_heads dim to dim 1
        medusa_res = self.act(self.linear(x)).reshape(
            *x_block.shape[:-1], self.n_medusa_heads, x_block.shape[-1]
        )

        # Apply all residual medusa heads
        output = x[:, start:stop].unsqueeze(-2) + medusa_res

        # Gather medusa heads
        world_output = [
            torch.empty_like(output) for _ in range(self.process_group.size())
        ]
        torch.distributed.all_gather(world_output, output, group=self.process_group)
        world_output = torch.cat(world_output, dim=-1)

        # Stack x and medusa residual x
        stacked_x = torch.cat([x.unsqueeze(-2), world_output], dim=-2)

        # Compute lm head on x + medusa residual x
        logits = self.lm_head(stacked_x)

        # Finally, split logits from speculative logits
        logits, speculative_logits = torch.split(
            logits, [1, self.n_medusa_heads], dim=-2
        )
        # Squeeze added dimension
        logits = logits.squeeze(-2)

        return logits, speculative_logits


class SpeculativeHead(nn.Module):
    def __init__(self, lm_head, speculator):
        super().__init__()
        self.head = lm_head
        self.speculator = speculator

    @staticmethod
    def load(config, prefix: str, weights):
        use_speculator = config.use_speculator
        if use_speculator:
            speculator_config = str(Path(use_speculator) / "config.json")

            with open(speculator_config, "r") as f:
                speculator_config = json.load(f)
            lm_head = None

            # currently medusa does not have an architecture specified, so try-except for now
            # this should really be handled in a better way though (maybe the classname can be part of the config)
            try:
                architecture = speculator_config["architectures"][0]

                if architecture == "MLPSpeculatorPreTrainedModel":
                    speculator_config.use_speculator = config.use_speculator
                    speculator = MLPSpeculatorHead.load(speculator_config, prefix, weights)
                else:
                    speculator = None
            except KeyError:
                try:
                    speculator = MedusaHeadV1.load(config, prefix, weights)
                except:
                    speculator = MedusaHeadV2(config, prefix, weights)
        else:
            lm_head = TensorParallelHead.load(config, prefix, weights)
            speculator = None
        return SpeculativeHead(lm_head, speculator)

    def forward(
        self, input: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.medusa is not None:
            return self.medusa(input)

        assert self.head is not None
        logits = self.head(input)
        return logits, None


class TensorParallelHead(SuperLayer):
    def __init__(self, linear, process_group, should_gather: bool):
        super().__init__(linear)
        self.process_group = process_group
        self.should_gather = should_gather

    @staticmethod
    def load(config, prefix: str, weights):
        if weights.process_group.size() > 1:
            try:
                weight = weights.get_sharded(f"{prefix}.weight", dim=0)
                should_gather = True
            except AssertionError:
                # If the vocab size is not divisible by number of shards
                # just load the entire thing.
                weight = weights.get_tensor(f"{prefix}.weight")
                should_gather = False
        else:
            weight = weights.get_tensor(f"{prefix}.weight")
            should_gather = False

        # GPTQ,AWQ,EETQ don't quantize heads (nor embeddings)
        if config.quantize in ["gptq", "awq", "eetq"]:
            quantize = None
        else:
            quantize = config.quantize
        return TensorParallelHead(
            get_linear(weight, bias=None, quantize=quantize),
            process_group=weights.process_group,
            should_gather=should_gather,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self.should_gather:
            return super().forward(input)

        world_size = self.process_group.size()
        if len(input.shape) == 2 and isinstance(self.linear, FastLinear):
            out_dim = self.linear.weight.shape[0]

            if input.shape[0] == 1:
                world_out = input.new_empty(1, out_dim * world_size)
                local_out = input.new_empty(1, out_dim)
                gather_input = local_out
            else:
                world_out = input.new_empty(out_dim * world_size, input.shape[0])
                gather_input = input.new_empty(out_dim, input.shape[0])
                local_out = gather_input.T

            torch.mm(input, self.linear.weight.T, out=local_out)

            torch.distributed.all_gather_into_tensor(
                world_out, gather_input, group=self.process_group
            )

            if input.shape[0] == 1:
                return world_out
            return world_out.T

        output = super().forward(input)
        world_output = [
            torch.empty_like(output) for _ in range(self.process_group.size())
        ]
        torch.distributed.all_gather(world_output, output, group=self.process_group)
        world_output = torch.cat(world_output, dim=-1)
        return world_output


class TensorParallelColumnLinear(SuperLayer):
    @classmethod
    def load_gate_up(cls, config, prefix: str, weights, bias: bool):
        """Specific method when the QKV was joined after the fact"""
        weight = weights.get_weights_col_packed_gate_up(
            prefix, quantize=config.quantize
        )
        if bias:
            raise NotImplementedError("packed_gate_up only implemented without bias")
        else:
            bias = None
        linear = get_linear(weight, bias, config.quantize)
        return cls(linear)

    @classmethod
    def load_qkv(cls, config, prefix: str, weights, bias: bool):
        """Specific method when the QKV was joined after the fact"""
        weight = weights.get_weights_col_packed_qkv(prefix, quantize=config.quantize)
        if bias:
            raise NotImplementedError("packed_qkv only implemented for baichuan")
        else:
            bias = None
        linear = get_linear(weight, bias, config.quantize)
        return cls(linear)

    @classmethod
    def load(cls, config, prefix: str, weights, bias: bool):
        return cls.load_multi(config, [prefix], weights, bias, dim=0)

    @classmethod
    def load_multi(cls, config, prefixes: List[str], weights, bias: bool, dim: int):
        weight = weights.get_multi_weights_col(
            prefixes, quantize=config.quantize, dim=dim
        )

        if bias:
            b = [weights.get_sharded(f"{p}.bias", dim=0) for p in prefixes]
            bias = torch.cat(b, dim=dim)
        else:
            bias = None
        linear = get_linear(weight, bias, config.quantize)
        return cls(linear)


class TensorParallelRowLinear(SuperLayer):
    def __init__(self, linear, process_group):
        super().__init__(linear)
        self.process_group = process_group

    @classmethod
    def load(cls, config, prefix: str, weights, bias: bool):
        weight = weights.get_multi_weights_row(prefix, quantize=config.quantize)

        if bias and weights.process_group.rank() == 0:
            # Rank is only on the first rank process
            bias = weights.get_tensor(f"{prefix}.bias")
        else:
            bias = None
        return cls(
            get_linear(weight, bias, config.quantize),
            process_group=weights.process_group,
        )

    def forward(self, input: torch.Tensor, reduce: bool = True) -> torch.Tensor:
        out = super().forward(input)
        if self.process_group.size() > 1 and reduce:
            torch.distributed.all_reduce(out, group=self.process_group)
        return out


class TensorParallelEmbedding(nn.Module):
    def __init__(self, prefix: str, weights, reduce=True):
        super().__init__()
        weight = weights.get_partial_sharded(f"{prefix}.weight", dim=0)
        num_embeddings = weights.get_shape(f"{prefix}.weight")[0]

        process_group = weights.process_group

        world_size = process_group.size()
        rank = process_group.rank()

        block_size = (num_embeddings + world_size - 1) // world_size
        self.min_id = rank * block_size
        self.max_id = min(num_embeddings, (rank + 1) * block_size)
        self.null_idx = weight.shape[
            0
        ]  # Usually block_size, might be less in non even vocab_size.
        self.process_group = weights.process_group
        self.reduce = reduce

        """Additional 0 entry used for masking"""
        self.weight = nn.Parameter(F.pad(weight, (0, 0, 0, 1)))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # default all out of bounds values to `self.null_idx` that will then be mapped to 0
        # translate for [0, self.max_id - self.min_id[
        input = torch.where(
            (self.min_id > input) | (input >= self.max_id),
            self.null_idx,
            input - self.min_id,
        )
        out = torch.nn.functional.embedding(input, self.weight)
        if self.reduce and self.process_group.size() > 1:
            torch.distributed.all_reduce(out, group=self.process_group)
        return out


try:
    if IS_CUDA_SYSTEM:
        import dropout_layer_norm
    elif IS_ROCM_SYSTEM:
        from vllm import layernorm_ops
    else:
        dropout_layer_norm = None

    class FastLayerNorm(nn.LayerNorm):
        def forward(self, hidden_states, residual=None):
            if IS_XPU_SYSTEM:
                res_out = hidden_states
                out = ipex.llm.functional.add_layer_norm(
                    residual, hidden_states, self.weight, self.bias, self.eps, True
                )
                if residual is not None:
                    res_out = residual
                return out, res_out
            elif hidden_states.shape[-1] > 8192 or IS_ROCM_SYSTEM:
                if residual is not None:
                    hidden_states += residual
                residual = hidden_states

                return super(FastLayerNorm, self).forward(hidden_states), residual
            else:
                (
                    normed_hidden_states,
                    residual,
                    *rest,
                ) = dropout_layer_norm.dropout_add_ln_fwd(
                    hidden_states,
                    residual,
                    self.weight,
                    self.bias,
                    None,
                    None,
                    None,
                    None,
                    0.0,
                    self.eps,
                    1.0,
                    0,
                    None,
                    False,
                    False,
                )
                if residual is None:
                    residual = hidden_states

                return normed_hidden_states, residual

    class FastRMSNorm(nn.Module):
        def __init__(self, weight: torch.Tensor, eps: float):
            super().__init__()

            self.weight = nn.Parameter(weight)
            self.variance_epsilon = eps

        @classmethod
        def load(cls, prefix, weights, eps=1e-6):
            weight = weights.get_tensor(f"{prefix}.weight")
            return cls(weight, eps)

        def forward(self, hidden_states, residual=None):
            if IS_XPU_SYSTEM:
                residual_out = hidden_states
                out = ipex.llm.functional.add_rms_norm(
                    residual,
                    hidden_states,
                    self.weight,
                    None,
                    self.variance_epsilon,
                    True,
                )
                if residual is not None:
                    residual_out = residual
                return out, residual_out
            elif hidden_states.shape[-1] > 8192:
                if residual is not None:
                    hidden_states += residual
                residual = hidden_states

                hidden_states = hidden_states.to(torch.float32)
                variance = hidden_states.pow(2).mean(-1, keepdim=True)
                hidden_states = hidden_states * torch.rsqrt(
                    variance + self.variance_epsilon
                )

                # convert into half-precision if necessary
                if self.weight.dtype in [torch.float16, torch.bfloat16]:
                    hidden_states = hidden_states.to(self.weight.dtype)

                return self.weight * hidden_states, residual
            elif IS_CUDA_SYSTEM:
                # faster post attention rms norm
                (
                    normed_hidden_states,
                    res,
                    *rest,
                ) = dropout_layer_norm.dropout_add_ln_fwd(
                    hidden_states,
                    residual,
                    self.weight,
                    None,
                    None,
                    None,
                    None,
                    None,
                    0.0,
                    self.variance_epsilon,
                    1.0,
                    0,
                    None,
                    False,
                    True,  # Activate RMSNorm
                )
                if res is None:
                    res = hidden_states

                return normed_hidden_states, res
            elif IS_ROCM_SYSTEM:
                # We use VLLM RMSNorm kernel that can be compiled for RoCm, instead of Flash Attention ones that can not.
                if residual is not None:
                    hidden_states += residual
                residual = hidden_states

                out = torch.empty_like(hidden_states)
                layernorm_ops.rms_norm(
                    out,
                    hidden_states,
                    self.weight.data,
                    self.variance_epsilon,
                )
                return out, residual
            else:
                raise ValueError(
                    "Your system seem to be not supported. Please check your install or open an issue at https://github.com/huggingface/text-generation-inference/issues with a clear reproduction."
                )

except ImportError:
    pass

try:
    if IS_CUDA_SYSTEM:
        from flash_attn.layers.rotary import RotaryEmbedding
        import rotary_emb
    elif IS_ROCM_SYSTEM:
        from vllm import pos_encoding_ops

    def _create_inv_freq(dim, base, device):
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
        )
        return inv_freq

    def _get_rope_config(config):
        if os.getenv("ROPE_SCALING", None) is not None:
            rope_scaling = {
                "type": os.environ["ROPE_SCALING"],
                "factor": float(os.environ["ROPE_FACTOR"]),
            }
            return rope_scaling
        return getattr(config, "rope_scaling", None)

    class PositionRotaryEmbedding(nn.Module):
        def __init__(self, inv_freq, scaling_factor):
            super().__init__()
            self.inv_freq = inv_freq
            self._seq_len_cached = 0
            self._cos_cached = None
            self._sin_cached = None
            self._cos_k_cached = None
            self._sin_k_cached = None
            self.scaling_factor = scaling_factor
            self.dynamic_args = None

        def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor,
        ):
            # Such controlflows may add some overhead.
            if IS_CUDA_SYSTEM:
                rotary_dim = cos.shape[-1]
                q1 = query[..., :rotary_dim]
                q2 = query[..., rotary_dim : 2 * rotary_dim]

                rotary_emb.apply_rotary(q1, q2, cos, sin, q1, q2, False)

                k1 = key[..., :rotary_dim]
                k2 = key[..., rotary_dim : 2 * rotary_dim]

                rotary_emb.apply_rotary(k1, k2, cos, sin, k1, k2, False)
            elif IS_ROCM_SYSTEM:
                # NOTE: On RoCm systems, we use a ROPE implementatation adapted from VLLM which launches a single kernel for both query/key, contrary to flash-attn implementation used on NVIDIA systems.
                # Compiling flash-attn rotary on RoCm, it appears hipcc is unable to unroll loops, resulting in an even slower inference compared to eager: https://github.com/pytorch/pytorch/issues/113773

                head_size = query.shape[-1]

                # Inplace operation, updating query and key.
                pos_encoding_ops.rotary_embedding(query, key, head_size, cos, sin, True)
            elif IS_XPU_SYSTEM:
                ipex.llm.functional.rotary_embedding(
                    query, key, sin, cos, query.size(-1), True
                )
            else:
                raise ValueError(
                    "Your system seem to be not supported. Please check your install or open an issue at https://github.com/huggingface/text-generation-inference/issues with a clear reproduction."
                )

        @classmethod
        def static(cls, config, dim, base, device):
            inv_freq = _create_inv_freq(dim, base, device)
            scaling_factor = None
            rope_scaling = _get_rope_config(config)
            if rope_scaling is not None:
                scaling_factor = rope_scaling["factor"]
                if rope_scaling["type"] == "linear":
                    pass
                elif rope_scaling["type"] == "dynamic":
                    return DynamicPositionRotaryEmbedding(
                        dim=dim,
                        max_position_embeddings=config.max_position_embeddings,
                        base=base,
                        device=inv_freq.device,
                        scaling_factor=scaling_factor,
                    )
                elif rope_scaling["type"] == "yarn":
                    return YarnPositionRotaryEmbedding(
                        dim=2 * inv_freq.shape[0],
                        max_position_embeddings=rope_scaling[
                            "original_max_position_embeddings"
                        ],
                        base=10000.0,
                        device=inv_freq.device,
                        scaling_factor=scaling_factor,
                        extrapolation_factor=1,
                        attn_factor=1,
                        beta_fast=32,
                        beta_slow=1,
                    )
                else:
                    raise NotImplementedError(
                        f"rope scaling type {rope_scaling['type']} is not implemented or invalid"
                    )
            return cls(inv_freq, scaling_factor)

        @classmethod
        def load(cls, config, prefix, weights):
            # XXX: Always load this in float32 !
            dtype = weights.dtype
            weights.dtype = torch.float32
            inv_freq = weights.get_tensor(f"{prefix}.inv_freq")
            weights.dtype = dtype

            scaling_factor = None
            rope_scaling = _get_rope_config(config)
            if rope_scaling is not None:
                scaling_factor = rope_scaling["factor"]
                if rope_scaling["type"] == "linear":
                    pass
                elif rope_scaling["type"] == "dynamic":
                    return DynamicPositionRotaryEmbedding(
                        dim=2 * inv_freq.shape[0],
                        max_position_embeddings=config.max_position_embeddings,
                        base=10000.0,
                        device=inv_freq.device,
                        scaling_factor=scaling_factor,
                    )
                elif rope_scaling["type"] == "yarn":
                    return YarnPositionRotaryEmbedding(
                        dim=2 * inv_freq.shape[0],
                        max_position_embeddings=rope_scaling[
                            "original_max_position_embeddings"
                        ],
                        base=10000.0,
                        device=inv_freq.device,
                        scaling_factor=scaling_factor,
                        extrapolation_factor=1,
                        attn_factor=1,
                        beta_fast=32,
                        beta_slow=1,
                    )
                else:
                    raise NotImplementedError(
                        f"rope scaling type {rope_scaling['type']} is not implemented or invalid"
                    )
            return cls(inv_freq, scaling_factor)

        def _update_cos_sin_cache(self, dtype, device, seqlen):
            # Reset the tables if the sequence length has changed,
            # or if we're on a new device (possibly due to tracing for instance)
            if (
                seqlen > self._seq_len_cached
                or self._cos_cached.device != device
                or self._cos_cached.dtype != dtype
            ):
                self._seq_len_cached = seqlen
                t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
                if self.scaling_factor is not None:
                    t /= self.scaling_factor
                # Don't do einsum, it converts fp32 to fp16
                # freqs = torch.einsum("i,j->ij", t, self.inv_freq)

                freqs = torch.outer(t, self.inv_freq.to(device=t.device))
                self._cos_cached = torch.cos(freqs).to(dtype)
                self._sin_cached = torch.sin(freqs).to(dtype)

        def get_cos_sin(
            self, position_ids: torch.Tensor, max_s: int, dtype: torch.dtype
        ):
            """
            Return cos and sin for the asked position ids
            """
            if IS_ROCM_SYSTEM:
                # For RoCm, we always use float cos/sin to avoid a cast.
                # For NVIDIA, for some reason, the flash-attn rotary kernel requires cos/sin and query/key to be of same dtype: https://github.com/Dao-AILab/flash-attention/blob/017716451d446e464dde9aca3a3c1ed2209caaa9/csrc/rotary/rotary.cpp#L26
                # But later on goes and cast cos/sin to float anyway: https://github.com/Dao-AILab/flash-attention/blob/017716451d446e464dde9aca3a3c1ed2209caaa9/csrc/rotary/rotary_cuda.cu#L29, which looks suboptimal.
                dtype = torch.float32

            self._update_cos_sin_cache(dtype, position_ids.device, max_s)

            cos = torch.index_select(self._cos_cached, 0, position_ids)
            sin = torch.index_select(self._sin_cached, 0, position_ids)

            # Note: this unsqueeze is not necessary on RoCm + VLLM ROPE implementation, but we leave it as is to avoid yet an other controlflow.
            return cos.unsqueeze(1), sin.unsqueeze(1)

    class DynamicPositionRotaryEmbedding(PositionRotaryEmbedding):
        def __init__(self, dim, max_position_embeddings, base, device, scaling_factor):
            inv_freq = _create_inv_freq(dim, base, device)
            super().__init__(inv_freq, scaling_factor)
            self.dim = dim
            self.max_position_embeddings = max_position_embeddings
            self.base = base

        def _update_cos_sin_cache(self, dtype, device, seqlen):
            # Reset the tables if the sequence length has changed,
            # or if we're on a new device (possibly due to tracing for instance)
            if (
                seqlen > self._seq_len_cached
                or self._cos_cached.device != device
                or self._cos_cached.dtype != dtype
            ):
                if seqlen > self.max_position_embeddings:
                    newbase = self.base * (
                        (self.scaling_factor * seqlen / self.max_position_embeddings)
                        - (self.scaling_factor - 1)
                    ) ** (self.dim / (self.dim - 2))
                    self.inv_freq = _create_inv_freq(
                        self.dim, newbase, self.inv_freq.device
                    )
                self._seq_len_cached = seqlen
                t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
                # Don't do einsum, it converts fp32 to fp16
                # freqs = torch.einsum("i,j->ij", t, self.inv_freq)

                freqs = torch.outer(t, self.inv_freq.to(device=t.device))
                self._cos_cached = torch.cos(freqs).to(dtype)
                self._sin_cached = torch.sin(freqs).to(dtype)

    # Inverse dim formula to find dim based on number of rotations
    import math

    def find_correction_dim(
        num_rotations, dim, base=10000, max_position_embeddings=2048
    ):
        return (
            dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))
        ) / (2 * math.log(base))

    # Find dim range bounds based on rotations
    def find_correction_range(
        low_rot, high_rot, dim, base=10000, max_position_embeddings=2048
    ):
        low = math.floor(
            find_correction_dim(low_rot, dim, base, max_position_embeddings)
        )
        high = math.ceil(
            find_correction_dim(high_rot, dim, base, max_position_embeddings)
        )
        return max(low, 0), min(high, dim - 1)  # Clamp values just in case

    def linear_ramp_mask(min, max, dim):
        if min == max:
            max += 0.001  # Prevent singularity

        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    def get_mscale(scale=1):
        if scale <= 1:
            return 1.0
        return 0.1 * math.log(scale) + 1.0

    class YarnPositionRotaryEmbedding(PositionRotaryEmbedding):
        def __init__(
            self,
            dim,
            max_position_embeddings,
            base,
            device,
            scaling_factor,
            *,
            extrapolation_factor,
            attn_factor,
            beta_fast,
            beta_slow,
        ):
            inv_freq = _create_inv_freq(dim, base, device)
            super().__init__(inv_freq, scaling_factor)
            self.dim = dim
            self.max_position_embeddings = max_position_embeddings
            self.base = base
            self.extrapolation_factor = extrapolation_factor
            self.attn_factor = attn_factor
            self.beta_fast = beta_fast
            self.beta_slow = beta_slow
            self.mscale = float(
                get_mscale(self.scaling_factor) * self.attn_factor
            )  # Get n-d magnitude scaling corrected for interpolation

        def _update_cos_sin_cache(self, dtype, device, seqlen):
            # Reset the tables if the sequence length has changed,
            # or if we're on a new device (possibly due to tracing for instance)
            if (
                seqlen > self._seq_len_cached
                or self._cos_cached.device != device
                or self._cos_cached.dtype != dtype
            ):
                if seqlen > self.max_position_embeddings:
                    inv_freq_extrapolation = _create_inv_freq(
                        self.dim, self.base, self.inv_freq.device
                    )
                    freqs = 1.0 / inv_freq_extrapolation
                    inv_freq_interpolation = 1.0 / (self.scaling_factor * freqs)
                    low, high = find_correction_range(
                        self.beta_fast,
                        self.beta_slow,
                        self.dim,
                        self.base,
                        self.max_position_embeddings,
                    )
                    inv_freq_mask = (
                        1
                        - linear_ramp_mask(low, high, self.dim // 2).float().to(device)
                    ) * self.extrapolation_factor  # Get n-d rotational scaling corrected for extrapolation
                    inv_freq = (
                        inv_freq_interpolation * (1 - inv_freq_mask)
                        + inv_freq_extrapolation * inv_freq_mask
                    )

                    self.inv_freq = inv_freq
                    self.mscale = float(
                        get_mscale(self.scaling_factor) * self.attn_factor
                    )  # Get n-d magnitude scaling corrected for interpolation

                self._seq_len_cached = seqlen
                t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
                # Don't do einsum, it converts fp32 to fp16
                # freqs = torch.einsum("i,j->ij", t, self.inv_freq)

                freqs = torch.outer(t, self.inv_freq.to(device=t.device))
                self._cos_cached = (torch.cos(freqs) * self.mscale).to(dtype)
                self._sin_cached = (torch.sin(freqs) * self.mscale).to(dtype)

except ImportError:
    pass
