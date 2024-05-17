import torch
from torch import nn
from accelerate import init_empty_weights
from text_generation_server.utils.import_utils import (
    SYSTEM,
)


# Monkey patching
@classmethod
def load_layer_norm(cls, prefix, weights, eps):
    weight = weights.get_tensor(f"{prefix}.weight")
    bias = weights.get_tensor(f"{prefix}.bias")
    with init_empty_weights():
        ln = cls(weight.shape, eps=eps)

    ln.weight = torch.nn.Parameter(weight)
    ln.bias = torch.nn.Parameter(bias)
    return ln


@classmethod
def load_layer_norm_no_bias(cls, prefix, weights, eps):
    weight = weights.get_tensor(f"{prefix}.weight")
    with init_empty_weights():
        ln = cls(weight.shape, eps=eps)

    ln.weight = torch.nn.Parameter(weight)
    ln.bias = None
    return ln


torch.nn.LayerNorm.load = load_layer_norm
torch.nn.LayerNorm.load_no_bias = load_layer_norm_no_bias

if SYSTEM == "cuda":
    import dropout_layer_norm

    class FastLayerNorm(nn.LayerNorm):
        def forward(self, hidden_states, residual=None):
            if hidden_states.shape[-1] > 8192:
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

elif SYSTEM == "rocm":
    from vllm._C import ops

    class FastLayerNorm(nn.LayerNorm):
        def forward(self, hidden_states, residual=None):
            if residual is not None:
                hidden_states += residual
            residual = hidden_states

            return super().forward(hidden_states), residual

elif SYSTEM == "xpu":
    import intel_extension_for_pytorch as ipex

    class FastLayerNorm(nn.LayerNorm):
        def forward(self, hidden_states, residual=None):
            res_out = hidden_states
            out = ipex.llm.functional.add_layer_norm(
                residual, hidden_states, self.weight, self.bias, self.eps, True
            )
            if residual is not None:
                res_out = residual
            return out, res_out


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
        if SYSTEM == "xpu":
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
        elif SYSTEM == "cuda":
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
        elif SYSTEM == "rocm":
            # We use VLLM RMSNorm kernel that can be compiled for RoCm, instead of Flash Attention ones that can not.
            if residual is not None:
                hidden_states += residual
            residual = hidden_states

            out = torch.empty_like(hidden_states)
            ops.rms_norm(
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
