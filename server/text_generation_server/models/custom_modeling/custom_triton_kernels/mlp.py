from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
import triton
import triton.language as tl
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream
from torch import device, empty, empty_strided
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
alloc_from_pool = torch.ops.inductor._alloc_from_pool
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
async_compile = AsyncCompile()


triton_poi_fused_mul_silu_0 = async_compile.triton(
    "triton_poi_fused_mul_silu_0",
    """
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384],
    filename=__file__,
    # triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    # inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_silu_0', 'mutated_arg_names': []},
    # min_elem_per_thread=0
    triton_meta={
        'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'},
        'device': 0,
        'device_type': 'cuda',
        'constants': {},
        'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]
    },
    inductor_meta={
        'autotune_hints': set(),
        'kernel_name': 'triton_poi_fused_mul_silu_0',
        'mutated_arg_names': []
    },
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_silu_0(in_ptr0, out_ptr0, xnumel: int, XBLOCK : tl.constexpr):
    # xnumel = 11008
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (xnumel + x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr0 + (x0), tmp6, xmask)
""",
)


triton_tem_fused_addmm_1 = async_compile.triton(
    "triton_tem_fused_addmm_1",
    """
import triton.language as tl
import triton
from torch._inductor.triton_heuristics import template
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers


@template(
    num_stages=5,
    num_warps=4,
    triton_meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'kernel_name': 'triton_tem_fused_addmm_1'},
)
@triton.jit

def triton_tem_fused_addmm_1(in_ptr0, arg_A, arg_B, out_ptr0):
    GROUP_M : tl.constexpr = 8
    EVEN_K : tl.constexpr = True
    ALLOW_TF32 : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    B_PROLOGUE_CAST_TYPE : tl.constexpr = None
    BLOCK_M : tl.constexpr = 16
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 32

    A = arg_A
    B = arg_B

    M = 1
    N = 4096
    K = 11008
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 11008
    stride_ak = 1
    stride_bk = 1
    stride_bn = 11008

    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        if B_PROLOGUE_CAST_TYPE is not None:
            b = b.to(B_PROLOGUE_CAST_TYPE)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + (4096*idx_m)
    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, mask.shape)), mask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = acc + tmp0
    tl.store(out_ptr0 + (tl.broadcast_to(idx_n, mask.shape)), tmp1, mask)
""",
)
import torch._inductor.kernel.mm_common

meta0 = {
    "GROUP_M": 8,
    "EVEN_K": True,
    "ALLOW_TF32": False,
    "ACC_TYPE": "tl.float32",
    "B_PROLOGUE_CAST_TYPE": None,
    "BLOCK_M": 16,
    "BLOCK_N": 64,
    "BLOCK_K": 32,
}


async_compile.wait(globals())
del async_compile


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (22016, 4096), (4096, 1))
    assert_size_stride(primals_2, (22016,), (1,))
    assert_size_stride(primals_3, (4096, 11008), (11008, 1))
    assert_size_stride(primals_4, (4096,), (1,))
    assert_size_stride(primals_5, (1, 4096), (4096, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)  # no-op to ensure context
        buf0 = empty((1, 22016), device="cuda", dtype=torch.float16)
        # Source Nodes: [gate_up_states], Original ATen: [aten.addmm]
        extern_kernels.bias_addmm(
            reinterpret_tensor(primals_2, (1, 22016), (0, 1), 0),
            primals_5,
            reinterpret_tensor(primals_1, (4096, 22016), (1, 4096), 0),
            alpha=1,
            beta=1,
            out=buf0,
        )
        del primals_1
        del primals_2
        buf1 = empty((1, 11008), device="cuda", dtype=torch.float16)
        # Source Nodes: [l__self___act, mul], Original ATen: [aten.mul, aten.silu]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_mul_silu_0.run(
            buf0, buf1, 11008, grid=grid(11008), stream=stream0
        )
        buf2 = empty((1, 4096), device="cuda", dtype=torch.float16)
        # Source Nodes: [l__self___down_proj], Original ATen: [aten.addmm]
        triton_tem_fused_addmm_1.run(
            primals_4,
            buf1,
            primals_3,
            buf2,
            grid=torch._inductor.kernel.mm_common.mm_grid(1, 4096, meta0),
            stream=stream0,
        )
        del primals_4
        return (
            buf2,
            primals_5,
            buf0,
            buf1,
            reinterpret_tensor(primals_3, (11008, 4096), (1, 11008), 0),
        )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance

    primals_1 = rand_strided(
        (22016, 4096), (4096, 1), device="cuda:0", dtype=torch.float16
    )
    primals_2 = rand_strided((22016,), (1,), device="cuda:0", dtype=torch.float16)
    primals_3 = rand_strided(
        (4096, 11008), (11008, 1), device="cuda:0", dtype=torch.float16
    )
    primals_4 = rand_strided((4096,), (1,), device="cuda:0", dtype=torch.float16)
    primals_5 = rand_strided((1, 4096), (4096, 1), device="cuda:0", dtype=torch.float16)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main

    compiled_module_main("None", benchmark_compiled_module)


from torch import nn


class CustomMLP(nn.Module):
    # TODO: replace with load when we have a way to handle dynamic shapes
    # def __init__(self, prefix, config, weights):
    #     super(CustomMLP, self).__init__()

    #     prefixes = [f"{prefix}.gate_proj", f"{prefix}.up_proj"]
    #     dim = 0
    #     self.gate_up_weights = weights.get_multi_weights_col(
    #         prefixes, quantize=config.quantize, dim=dim
    #     )
    #     self.gate_up_bias = torch.zeros(
    #         self.gate_up_weights.size(0), device=self.gate_up_weights[0].device
    #     )

    #     self.down_weights = weights.get_multi_weights_row(
    #         f"{prefix}.down_proj", quantize=config.quantize
    #     )
    #     self.down_bias = torch.zeros(
    #         self.down_weights.size(0), device=self.down_weights.device
    #     )

    def __init__(self, gate_up_weights, down_weights):
        super(CustomMLP, self).__init__()
        self.gate_up_weights = gate_up_weights
        self.gate_up_bias = torch.zeros(
            self.gate_up_weights.size(0),
            device=self.gate_up_weights[0].device,
            dtype=self.gate_up_weights[0].dtype,
        )
        self.down_weights = down_weights
        self.down_bias = torch.zeros(
            self.down_weights.size(0),
            device=self.down_weights.device,
            dtype=self.down_weights.dtype,
        )

    def forward(self, x):
        return call(
            [
                self.gate_up_weights,
                self.gate_up_bias,
                self.down_weights,
                self.down_bias,
                x,
            ]
        )[0]
