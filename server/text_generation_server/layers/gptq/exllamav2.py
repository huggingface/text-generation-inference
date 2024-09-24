# Adapted from turboderp exllama: https://github.com/turboderp/exllamav2

import torch
import torch.nn as nn

from loguru import logger

try:
    from exllamav2_kernels import make_q_matrix, gemm_half_q_half
except ImportError:
    logger.error("exllamav2_kernels not installed.")
    raise

# Dummy tensor to pass instead of g_idx since there is no way to pass "None" to a C++ extension
none_tensor = torch.empty((1, 1), device="meta")


def ext_gemm_half_q_half(x, q_handle, q4_width, force_cuda):
    """Matrix multiplication, returns x @ q4"""
    output_shape = x.shape[:-1] + (q4_width,)
    x = x.view(-1, x.shape[-1])
    output = torch.empty((x.shape[0], q4_width), dtype=torch.half, device=x.device)
    gemm_half_q_half(x, q_handle, output, force_cuda)
    return output.view(output_shape)


# Group map needed for irregular group sizes


def make_group_map(q_groups, num_qrows):

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


def ext_make_q_matrix(w: dict, temp_dq, key: str = None):
    """
    Create Q matrix
    """
    # EXL2
    # won't work as the moment because the tensors are not the same.
    if "q_weight" in w:
        w["q_scale_max"] /= 256
        w["q_perm"] = w["q_perm"].short()
        w["q_invperm"] = w["q_invperm"].short()

        if "q_group_map" not in w:
            w["q_group_map"] = make_group_map(w["q_groups"], w["q_weight"].shape[0])

        return make_q_matrix(
            w["q_weight"],
            w["q_perm"],
            w["q_invperm"],
            w["q_scale"],
            w["q_scale_max"],
            w["q_groups"],
            w["q_group_map"],
            none_tensor,
            none_tensor,
            none_tensor,
            temp_dq,
        )
    # GPTQ
    elif "qweight" in w:
        if w["scales"].dtype == torch.float:
            w["scales"] = w["scales"].half()

        # GPTQ with g_idx (act_order)
        if w.get("g_idx", None) is not None and not (w["g_idx"] == 0).all().item():
            w["q_perm"] = torch.empty(
                (w["qweight"].shape[0] * 8,),
                dtype=torch.short,
                device=w["qweight"].device,
            )
            w["q_invperm"] = torch.empty_like(w["q_perm"])
            # make_q4 segfaults if g_idx is not on cpu in the act-order case. In the non act-order case, None needs to be passed for g_idx.
            return make_q_matrix(
                w["qweight"],
                w["q_perm"],
                w["q_invperm"],
                none_tensor,
                none_tensor,
                none_tensor,
                none_tensor,
                w["qzeros"],
                w["scales"],
                w["g_idx"].cpu(),
                temp_dq,
            )
        # GPTQ without g_idx
        else:
            return make_q_matrix(
                w["qweight"],
                none_tensor,
                none_tensor,
                none_tensor,
                none_tensor,
                none_tensor,
                none_tensor,
                w["qzeros"],
                w["scales"],
                none_tensor,
                temp_dq,
            )
    else:
        RuntimeError("Cannot create handle")


DEVICE = None
FIXED_BYTES = 0
LAYERS = []


def set_device(device):
    global DEVICE
    DEVICE = device


def create_exllama_buffers(max_total_tokens: int):
    global FIXED_BYTES, LAYERS, DEVICE
    temp_dq = ExLlamaV2DeviceTensors(DEVICE, FIXED_BYTES)

    for layer in LAYERS:
        layer.post_init(temp_dq)


class QuantLinear(nn.Module):
    QUANT_TYPE = "exllamav2"

    """Linear layer implementation with per-group 4-bit quantization of the weights"""

    # def __init__(self, bits, group_size, infeatures, outfeatures, bias, trainable=False, **kwargs):
    def __init__(self, qweight, qzeros, scales, g_idx, bias, bits, groupsize):
        super().__init__()
        if bits != 4:
            raise ValueError(
                f"Exllamav2 kernel supports only bits=4, requested bits={bits}. Something is wrong in the model initialization."
            )
        self.q_handle = None
        self.q_tensors = None
        self.bits = bits
        self.maxq = 2**self.bits - 1
        self.infeatures = qweight.shape[0] // self.bits * 32
        self.outfeatures = qweight.shape[1]
        self.padding = -self.outfeatures % 32
        self.outfeatures = self.outfeatures + self.padding

        self.device = qweight.device
        self.qweight = qweight
        self.qzeros = qzeros
        self.scales = scales
        self.g_idx = g_idx
        self.bias = bias if bias is not None else None
        self.group_size = groupsize

        global FIXED_BYTES, LAYERS
        FIXED_BYTES = max(FIXED_BYTES, self.scratch_space_fixed())
        LAYERS.append(self)

    def post_init(self, temp_dq):
        assert self.qweight.device.type == "cuda"
        assert self.qweight.device.index is not None
        self.q_tensors = {
            "qweight": self.qweight,
            "qzeros": self.qzeros,
            "scales": self.scales,
            "g_idx": self.g_idx,
        }
        temp_dq = temp_dq.get_scratch_slice(self.temp_dq_size())

        # We NEED to keep a pointer on Python side, otherwise the garbage collector will mess with us,
        # and `Memory access fault by GPU node-2` will EAT you.
        self.temp_dq = temp_dq
        self.q_handle = ext_make_q_matrix(self.q_tensors, temp_dq)

    def forward(self, x, force_cuda=False):
        output = ext_gemm_half_q_half(x, self.q_handle, self.outfeatures, force_cuda)

        if self.bias is not None:
            output.add_(self.bias)
        return output

    def temp_dq_size(self):
        return self.infeatures * self.outfeatures * 2 + 128

    def temp_fwd_size(self, max_input_len, max_batch_size):
        return self.outfeatures * max_input_len * max_batch_size * 4 + 128

    def scratch_space_fixed(self, max_input_len=4096, max_batch_size=16):
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
