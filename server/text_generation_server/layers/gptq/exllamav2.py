# Adapted from turboderp exllama: https://github.com/turboderp/exllamav2

from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn

from loguru import logger

from text_generation_server.layers.exl2 import Exl2Weight
from text_generation_server.layers.gptq import GPTQWeight
from text_generation_server.utils.log import log_master

try:
    from exllamav2.ext import exllamav2_ext

    make_q_matrix = exllamav2_ext.make_q_matrix
    gemm_half_q_half = exllamav2_ext.gemm_half_q_half
except ImportError:
    log_master(logger.warning, "exllamav2_kernels not installed.")
    raise

# Dummy tensor to pass instead of g_idx since there is no way to pass "None" to a C++ extension
none_tensor = torch.empty((1, 1), device="meta")


@dataclass
class _ExtraTensors:
    """Additional generated quantizer tensors."""

    q_group_map: Optional[torch.Tensor] = None
    q_invperm: Optional[torch.Tensor] = None
    q_perm: Optional[torch.Tensor] = None


def ext_gemm_half_q_half(x, q_handle, q4_width, force_cuda):
    """Matrix multiplication, returns x @ q4"""
    output_shape = x.shape[:-1] + (q4_width,)
    x = x.view(-1, x.shape[-1])
    output = torch.empty((x.shape[0], q4_width), dtype=torch.half, device=x.device)
    gemm_half_q_half(x, q_handle, output, force_cuda)
    return output.view(output_shape)


def make_group_map(q_groups: torch.Tensor, num_qrows: int):
    gr = q_groups.tolist()
    group_map = []
    num_groups = len(gr) // 2

    for i in range(num_groups):
        bits = gr[i * 2]
        if i < num_groups - 1:
            qrows = gr[i * 2 + 3] - gr[i * 2 + 1]
        else:
            qrows = num_qrows - gr[i * 2 + 1]
        rows = qrows * 32 // bits
        for j in range(rows):
            group_map += [i]
            group_map += [rows - j]

    return torch.tensor(group_map, dtype=torch.short, device=q_groups.device)


# Create Q matrix


def ext_make_q_matrix(
    w: Exl2Weight | GPTQWeight,
    extra: _ExtraTensors,
    temp_dq,
    key: Optional[str] = None,
):
    """
    Create Q matrix
    """
    # max_dq_size = 512*(1024**2)
    # max_dq_rows = max_dq_size // out_features[0]
    max_dq_rows = 0

    # EXL2
    if isinstance(w, Exl2Weight):
        extra.q_group_map = make_group_map(w.q_groups, w.q_weight.shape[0])
        extra.q_perm = torch.argsort(w.q_invperm).short()

        return make_q_matrix(
            w.q_weight,
            extra.q_perm,
            w.q_invperm,
            w.q_scale,
            w.q_scale_max,
            w.q_groups,
            extra.q_group_map,
            none_tensor,  # zeros
            none_tensor,  # scales
            none_tensor,  # g_idx
            none_tensor,  # bias
            temp_dq,
            max_dq_rows,
        )
    # GPTQ
    elif isinstance(w, GPTQWeight):
        if w.scales.dtype == torch.float:
            w.scales = w.scales.half()

        # GPTQ with g_idx (act_order)
        if w.g_idx is not None and not (w.g_idx == 0).all().item():
            extra.q_perm = torch.empty(
                (w.qweight.shape[0] * 8,),
                dtype=torch.short,
                device=w.qweight.device,
            )
            extra.q_invperm = torch.empty_like(extra.q_perm)
            # make_q4 segfaults if g_idx is not on cpu in the act-order case. In the non act-order case, None needs to be passed for g_idx.
            return make_q_matrix(
                w.qweight,
                extra.q_perm,
                extra.q_invperm,
                none_tensor,  # q_scale
                none_tensor,  # q_scale_max
                none_tensor,  # q_groups
                none_tensor,  # q_group_map
                w.qzeros,
                w.scales,
                w.g_idx.cpu(),
                none_tensor,  # bias
                temp_dq,
                max_dq_rows,
            )
        # GPTQ without g_idx
        else:
            return make_q_matrix(
                w.qweight,
                none_tensor,  # q_perm
                none_tensor,  # q_invperm
                none_tensor,  # q_scale
                none_tensor,  # q_scale_max
                none_tensor,  # q_groups
                none_tensor,  # q_group_map
                w.qzeros,
                w.scales,
                none_tensor,  # g_idx
                none_tensor,  # bias
                temp_dq,
                max_dq_rows,
            )
    else:
        RuntimeError("Cannot create handle")


DEVICE = None
LAYERS = []


def set_device(device):
    global DEVICE
    DEVICE = device


def create_exllama_buffers(max_total_tokens: int):
    global LAYERS, DEVICE

    # No need to initialize scratch space if there are no layers
    # that use ExLLamav2.
    if len(LAYERS) == 0:
        return

    # Find the size of the scratch space.
    scratch_bytes = max(
        layer.scratch_space_fixed(max_input_len=max_total_tokens, max_batch_size=1)
        for layer in LAYERS
    )
    temp_dq = ExLlamaV2DeviceTensors(DEVICE, scratch_bytes)

    for layer in LAYERS:
        layer.post_init(temp_dq)


class QuantLinear(nn.Module):
    QUANT_TYPE = "exllamav2"

    """Linear layer implementation with per-group 4-bit quantization of the weights"""

    def __init__(
        self,
        weight: Exl2Weight | GPTQWeight,
        bias: torch.Tensor,
    ):
        super().__init__()

        self.q_handle = None
        self.q_tensors = weight
        self.extra_tensors = _ExtraTensors()

        if isinstance(weight, Exl2Weight):
            self.infeatures = weight.q_invperm.shape[0]
            self.outfeatures = weight.q_weight.shape[1]
        elif isinstance(weight, GPTQWeight):
            if weight.bits != 4:
                raise ValueError(
                    f"Exllamav2 kernel supports only bits=4, requested bits={weight.bits}. Something is wrong in the model initialization."
                )

            self.infeatures = weight.qweight.shape[0] // weight.bits * 32
            self.outfeatures = weight.qweight.shape[1]

        self.padding = -self.outfeatures % 32
        self.outfeatures = self.outfeatures + self.padding

        self.device = weight.device
        self.bias = bias if bias is not None else None

        global LAYERS
        LAYERS.append(self)

    def post_init(self, temp_dq):
        device = self.q_tensors.device
        assert device.type == "cuda"
        assert device.index is not None
        temp_dq = temp_dq.get_scratch_slice(self.temp_dq_size())

        # We NEED to keep a pointer on Python side, otherwise the garbage collector will mess with us,
        # and `Memory access fault by GPU node-2` will EAT you.
        self.temp_dq = temp_dq
        self.q_handle = ext_make_q_matrix(self.q_tensors, self.extra_tensors, temp_dq)

    def forward(self, x, force_cuda=False):
        output = ext_gemm_half_q_half(x, self.q_handle, self.outfeatures, force_cuda)

        if self.bias is not None:
            output.add_(self.bias)
        return output

    def temp_dq_size(self):
        return self.infeatures * self.outfeatures * 2 + 128

    def temp_fwd_size(self, max_input_len, max_batch_size):
        return self.outfeatures * max_input_len * max_batch_size * 4 + 128

    def scratch_space_fixed(self, max_input_len, max_batch_size):
        return self.temp_dq_size() + self.temp_fwd_size(max_input_len, max_batch_size)


class ExLlamaV2DeviceTensors:

    device_idx: int
    scratch_bytes: int
    scratch_idx: int
    scratch: torch.tensor = None

    def __init__(self, device, scratch_bytes):
        self.device = device
        self.scratch_bytes = scratch_bytes

    def prepare(self):
        self.scratch = torch.empty(
            (self.scratch_bytes // 2,), dtype=torch.half, device=self.device
        )

    def get_scratch_slice(self, size_bytes):

        if self.scratch is None:
            self.prepare()

        size_bytes = ((size_bytes + 127) // 128) * 128
        size_half = size_bytes // 2
        scratch_slice = self.scratch.narrow(0, 0, size_half)
        return scratch_slice
