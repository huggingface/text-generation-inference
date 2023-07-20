import torch
from custom_kernels.exllama import make_q4, q4_matmul, set_tuning_params, prepare_buffers
from loguru import logger

# Dummy tensor to pass instead of g_idx since there is no way to pass "None" to a C++ extension

def ext_q4_matmul(x, q4, q4_width):
    """Matrix multiplication, returns x @ q4"""
    outshape = x.shape[:-1] + (q4_width,)
    x = x.view(-1, x.shape[-1])
    output = torch.empty((x.shape[0], q4_width), dtype = torch.float16, device = x.device)

    q4_matmul(x, q4, output)

    return output.view(outshape)


import os
RANK = os.getenv("RANK", "0")
DEVICE = torch.device(f"cuda:{RANK}")
MAX_TOTAL_TOKENS = 1
MAX_INNER_OUTER_DIM  = 0
MAX_DQ_BUFFER_SIZE = 0


def create_buffers():
    temp_state = torch.zeros((MAX_TOTAL_TOKENS, MAX_INNER_OUTER_DIM), dtype=torch.float16, device=DEVICE)
    temp_dq = torch.zeros((1, MAX_DQ_BUFFER_SIZE), dtype=torch.float16, device=DEVICE)
    logger.info(f"Creating buffers {temp_state.shape} - {temp_dq.shape} - {DEVICE}")

    prepare_buffers(DEVICE, temp_state, temp_dq)

    matmul_recons_thd = 8
    matmul_fused_remap = False
    matmul_no_half2 = False
    set_tuning_params(matmul_recons_thd, matmul_fused_remap, matmul_no_half2)

class Ex4bitLinear:
    """Linear layer implementation with per-group 4-bit quantization of the weights"""
    def __init__(self, qweight, qzeros, scales, bias, bits):
        assert bits == 4, "We cannot run exllama GPTQ kernels if bits != 4"

        global MAX_INNER_OUTER_DIM, MAX_DQ_BUFFER_SIZE
        dq = qweight.numel() * 8 
        if dq > MAX_DQ_BUFFER_SIZE:
            MAX_DQ_BUFFER_SIZE = dq
            
        width = qweight.shape[1]
        if width > MAX_INNER_OUTER_DIM:
            MAX_INNER_OUTER_DIM = width
        height = qweight.shape[0] * 8
        if height > MAX_INNER_OUTER_DIM:
            MAX_INNER_OUTER_DIM = height

        # prepare_buffers(DEVICE, TEMP_STATE, TEMP_DQ)


        self.q4 = make_q4(
            qweight,
            qzeros,
            scales,
            # Never send g_idx, it MUST be like act_order=False, the exllama kernel does not expect it
            torch.zeros((0, 0), device=torch.device("meta")),
            DEVICE.index
        )
        self.bias = bias if bias is not None else None
        self.width = width

        # # Infer groupsize from height of qzeros
        # self.groupsize = None
        # if self.qzeros.shape[0] > 1:
        #     self.groupsize = (self.qweight.shape[0] * 8) // (self.qzeros.shape[0])

        # if self.groupsize is not None:
        #     assert groupsize == self.groupsize

        # # Handle act-order matrix
        # if self.g_idx is not None:
        #     if self.groupsize is None: raise ValueError("Found group index but no groupsize. What do?")
        #     self.act_order = True
        # else:
        #     self.act_order = False
    
    def forward(self, x):
        out = ext_q4_matmul(x, self.q4, self.width)

        if self.bias is not None:
            out.add_(self.bias)
        return out
