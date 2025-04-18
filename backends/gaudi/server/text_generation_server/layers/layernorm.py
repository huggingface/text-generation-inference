import torch
from torch import nn
from accelerate import init_empty_weights


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


class FastLayerNorm(nn.LayerNorm):
    def forward(self, hidden_states, residual=None):
        if residual is not None:
            hidden_states += residual
        residual = hidden_states

        return super().forward(hidden_states), residual


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
        from vllm_hpu_extension.kernels import rms_norm

        orig_shape = hidden_states.shape
        if residual is not None:
            residual += hidden_states.view(residual.shape)
        else:
            residual = hidden_states
        # Note: HPUFusedRMSNorm requires 3D tensors as inputs
        if len(orig_shape) == 2:
            residual = residual.unsqueeze(0)
        x = rms_norm().apply(residual, self.weight, self.variance_epsilon)
        return x.view(orig_shape), residual.view(orig_shape)
