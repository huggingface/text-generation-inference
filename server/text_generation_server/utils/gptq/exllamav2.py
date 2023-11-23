# Adapted from turboderp exllama: https://github.com/turboderp/exllamav2

from logging import getLogger

import torch
import torch.nn as nn
import math

logger = getLogger(__name__)

try:
    from exllamav2_kernels import make_q_matrix, gemm_half_q_half
except ImportError:
    logger.error('exllamav2_kernels not installed.')
    raise

# Dummy tensor to pass instead of g_idx since there is no way to pass "None" to a C++ extension
none_tensor = torch.empty((1, 1), device="meta")

def ext_gemm_half_q_half(x, q_handle, q4_width, force_cuda):
    """Matrix multiplication, returns x @ q4"""
    output_shape = x.shape[:-1] + (q4_width,)
    x = x.view(-1, x.shape[-1])
    output = torch.empty((x.shape[0], q4_width), dtype = torch.half, device = x.device)
    gemm_half_q_half(x, q_handle, output, force_cuda)
    return output.view(output_shape)

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
        return make_q_matrix(w["q_weight"],
                                w["q_perm"],
                                w["q_invperm"],
                                w["q_scale"],
                                w["q_scale_max"],
                                w["q_groups"],
                                none_tensor,
                                none_tensor,
                                none_tensor,
                                temp_dq)
    # GPTQ
    elif "qweight" in w:
        if w["scales"].dtype == torch.float:
            w["scales"] = w["scales"].half()

        # GPTQ with g_idx (act_order)
        if "g_idx" in w and not (w["g_idx"] == 0).all().item():
            w["q_perm"] = torch.empty((w["qweight"].shape[0] * 8,), dtype = torch.short, device = w["qweight"].device)
            w["q_invperm"] = torch.empty_like(w["q_perm"])
            # make_q4 segfaults if g_idx is not on cpu in the act-order case. In the non act-order case, None needs to be passed for g_idx.
            return make_q_matrix(w["qweight"],
                                 w["q_perm"],
                                 w["q_invperm"],
                                 none_tensor,
                                 none_tensor,
                                 none_tensor,
                                 w["qzeros"],
                                 w["scales"],
                                 w["g_idx"].cpu(),
                                 temp_dq)
        # GPTQ without g_idx
        else:
            return make_q_matrix(w["qweight"],
                                none_tensor,
                                none_tensor,
                                none_tensor,
                                none_tensor,
                                none_tensor,
                                w["qzeros"],
                                w["scales"],
                                none_tensor,
                                temp_dq)

DEVICE = None
FIXED_BYTES = 0
LAYERS = []


def set_device(device):
    global DEVICE
    DEVICE = device


def create_exllama_buffers():
    global FIXED_BYTES, LAYERS, DEVICE
    temp_dq = ExLlamaV2DeviceTensors(DEVICE, FIXED_BYTES)

    for layer in LAYERS:
        layer.post_init(temp_dq)


    # assert DEVICE is not None, "call set_device first"

    # if ACT_ORDER:
    #     # TODO: this should be set to rust side `max_total_tokens`, but TGI
    #     # does not offer an API to expose this variable to python, as this variable
    #     # is handled by the client but it appears the model is initialized by the server.
    #     # An alternative could be to initialize the buffers during warmup.
    #     # Dummy
    #     max_total_tokens = 2048
    # else:
    #     max_total_tokens = 1

    # # This temp_state buffer is required to reorder X in the act-order case.
    # temp_state = torch.zeros(
    #     (max_total_tokens, MAX_INNER), dtype=torch.float16, device=DEVICE
    # )
    # temp_dq = torch.zeros((1, MAX_DQ), dtype=torch.float16, device=DEVICE)

    # # This temp_dq buffer is required to dequantize weights when using cuBLAS, typically for the prefill.
    # prepare_buffers(DEVICE, temp_state, temp_dq)

    # matmul_recons_thd = 8
    # matmul_fused_remap = False
    # matmul_no_half2 = False
    # set_tuning_params(matmul_recons_thd, matmul_fused_remap, matmul_no_half2)

    # TEMP_STATE, TEMP_DQ = temp_state, temp_dq


class QuantLinear(nn.Module):
    QUANT_TYPE = "exllamav2"

    """Linear layer implementation with per-group 4-bit quantization of the weights"""

    # def __init__(self, bits, group_size, infeatures, outfeatures, bias, trainable=False, **kwargs):
    def __init__(self, qweight, qzeros, scales, g_idx, bias, bits, groupsize):
        super().__init__()
        if bits != 4:
            raise ValueError(
                f"Exllamav2 kernel supports only bits=4, requested bits={bits}. Something is wrong in the model initialization.")
        # if trainable:
        #     raise NotImplementedError("Exllamav2 kernel does not support training.")

        self.q_handle = None
        self.q_tensors = None
        # 
        # self.infeatures = infeatures
        # self.outfeatures = outfeatures + self.padding
        self.bits = bits
        # self.group_size = group_size if group_size != -1 else infeatures
        # self.trainable = trainable
        self.maxq = 2 ** self.bits - 1
        self.infeatures = qweight.shape[0] // self.bits * 32
        self.outfeatures = qweight.shape[1]

        # assert infeatures % 32 == 0
        # assert infeatures % self.group_size == 0
        # assert outfeatures % 32 == 0
        # self.padding = - outfeatures % 32

        # # I need to register the tensors, otherwise, we won't be able to load them easily using transformers ... 
        # self.register_buffer(
        #     'qweight',
        #     torch.zeros((infeatures // 32 * self.bits, outfeatures), dtype=torch.int32)
        # )
        # self.register_buffer(
        #     'qzeros',
        #     torch.zeros((math.ceil(infeatures / self.group_size), outfeatures // 32 * self.bits), dtype=torch.int32)
        # )
        # self.register_buffer(
        #     'scales',
        #     torch.zeros((math.ceil(infeatures / self.group_size), outfeatures), dtype=torch.float16)
        # )
        # self.register_buffer(
        #     'g_idx',
        #     torch.tensor([i // self.group_size for i in range(infeatures)], dtype=torch.int32)
        # )
        self.device = qweight.device
        self.qweight = qweight
        self.qzeros = qzeros
        self.scales = scales
        self.g_idx = g_idx
        self.bias = bias if bias is not None else None

        global FIXED_BYTES, LAYERS
        FIXED_BYTES = max(FIXED_BYTES, self.scratch_space_fixed())
        LAYERS.append(self)

        # if bias:
        #     self.register_buffer('bias', torch.zeros((outfeatures), dtype=torch.float16))
        # else:
        #     self.bias = None

    def post_init(self, temp_dq):
        assert self.qweight.device.type == "cuda"
        assert self.qweight.device.index is not None
        self.q_tensors = {
            "qweight":self.qweight,
            "qzeros":self.qzeros,
            "scales":self.scales,
            "g_idx":self.g_idx
        }
        temp_dq = temp_dq.get_scratch_slice(self.temp_dq_size())
        self.q_handle = ext_make_q_matrix(
            self.q_tensors, temp_dq
        )
    
    def forward(self, x, force_cuda = False):
        output = ext_gemm_half_q_half(x, self.q_handle, self.outfeatures, force_cuda)

        if self.bias is not None:
            output.add_(self.bias)
        return output
    
    def temp_dq_size(self):
        return self.infeatures * self.outfeatures * 2 + 128
    
    def temp_fwd_size(self, max_input_len, max_batch_size):
        return self.outfeatures * max_input_len * max_batch_size * 4 + 128
    
    def scratch_space_fixed(self, max_input_len=2048, max_batch_size=8):
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
        self.scratch = torch.empty((self.scratch_bytes // 2,), dtype = torch.half, device = self.device)

    def get_scratch_slice(self, size_bytes):

        if self.scratch is None: self.prepare()

        size_bytes = ((size_bytes + 127) // 128) * 128
        size_half = size_bytes // 2
        scratch_slice = self.scratch.narrow(0, 0, size_half)
        return scratch_slice
