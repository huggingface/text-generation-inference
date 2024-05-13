from accelerate import init_empty_weights
import torch


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

    conv2d.weight = torch.nn.Parameter(weight)
    conv2d.bias = torch.nn.Parameter(bias)
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

    conv2d.weight = torch.nn.Parameter(weight)
    conv2d.bias = None
    return conv2d


torch.nn.Conv2d.load = load_conv2d
torch.nn.Conv2d.load_no_bias = load_conv2d_no_bias
