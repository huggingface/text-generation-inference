from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align

from torch import device, empty, empty_strided
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
alloc_from_pool = torch.ops.inductor._alloc_from_pool
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
async_compile = AsyncCompile()


# kernel path: /tmp/torchinductor_ubuntu/sb/csbqhwspefuy7jp6pcbyvalomk4wvqyt4mplgbphfugdwu66zbp7.py
# Source Nodes: [add, attn_res, hidden_states_1, hidden_states_2, hidden_states_3, normed_attn_res_output, pow_1, rsqrt, variance], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
# add => add_1
# attn_res => add
# hidden_states_1 => convert_element_type
# hidden_states_2 => mul
# hidden_states_3 => mul_1
# normed_attn_res_output => convert_element_type_1
# pow_1 => pow_1
# rsqrt => rsqrt
# variance => mean
triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_0 = async_compile.triton(
    "triton_",
    """
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_0', 'mutated_arg_names': ['in_ptr0', 'out_ptr3']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    rnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (3072*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r1 + (3072*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp3 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tl.store(out_ptr0 + (r1 + (3072*x0)), tmp2, rmask & xmask)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp8 = tl.load(out_ptr0 + (r1 + (3072*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp16 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp9 = tmp8.to(tl.float32)
        tmp10 = 3072.0
        tmp11 = tmp6 / tmp10
        tmp12 = 1e-06
        tmp13 = tmp11 + tmp12
        tmp14 = tl.math.rsqrt(tmp13)
        tmp15 = tmp9 * tmp14
        tmp17 = tmp16.to(tl.float32)
        tmp18 = tmp15 * tmp17
        tmp19 = tmp18.to(tl.float32)
        tl.store(out_ptr2 + (r1 + (3072*x0)), tmp19, rmask & xmask)
        tl.store(out_ptr3 + (r1 + (3072*x0)), tmp8, rmask & xmask)
""",
)

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_ubuntu/3y/c3ycepp3q64t65lfi2wcqq5skonqvz6t4okgw4dboyrxwdspdaff.py
# Source Nodes: [add, gate_up_states, hidden_states_1, hidden_states_2, hidden_states_3, normed_attn_res_output, pow_1, rsqrt, variance], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mm, aten.mul, aten.pow, aten.rsqrt]
# add => add_1
# gate_up_states => mm
# hidden_states_1 => convert_element_type
# hidden_states_2 => mul
# hidden_states_3 => mul_1
# normed_attn_res_output => convert_element_type_1
# pow_1 => pow_1
# rsqrt => rsqrt
# variance => mean
triton_tem_fused__to_copy_add_mean_mm_mul_pow_rsqrt_1 = async_compile.triton(
    "triton_",
    """
import triton.language as tl
import triton
from torch._inductor.triton_heuristics import template
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers


@template(
    num_stages=2,
    num_warps=4,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'kernel_name': 'triton_tem_fused__to_copy_add_mean_mm_mul_pow_rsqrt_1'},
)
@triton.jit

def triton_(arg_A, arg_B, out_ptr0, ks0):
    GROUP_M : tl.constexpr = 8
    EVEN_K : tl.constexpr = True
    ALLOW_TF32 : tl.constexpr = True
    ACC_TYPE : tl.constexpr = tl.float32
    B_PROLOGUE_CAST_TYPE : tl.constexpr = None
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 32

    A = arg_A
    B = arg_B

    M = ks0
    N = 49152
    K = 3072
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 3072
    stride_ak = 1
    stride_bk = 1
    stride_bn = 3072

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
    xindex = idx_n + (49152*idx_m)
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, mask.shape)), acc, mask)
""",
)
import torch._inductor.kernel.mm_common

meta0 = {
    "GROUP_M": 8,
    "EVEN_K": True,
    "ALLOW_TF32": True,
    "ACC_TYPE": "tl.float32",
    "B_PROLOGUE_CAST_TYPE": None,
    "BLOCK_M": 64,
    "BLOCK_N": 64,
    "BLOCK_K": 32,
}


# kernel path: /tmp/torchinductor_ubuntu/hz/chzpreui7cisscnbe7lrfewue7t4pg7pbnbouuqqq5d2jjbc3wsv.py
# Source Nodes: [gelu, inter], Original ATen: [aten.gelu, aten.mul]
# gelu => add_2, convert_element_type_4, convert_element_type_5, erf, mul_2, mul_3, mul_4
# inter => mul_5
triton_poi_fused_gelu_mul_2 = async_compile.triton(
    "triton_",
    """
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8388608],
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_mul_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 24576
    x1 = (xindex // 24576)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (49152*x1)), None).to(tl.float32)
    tmp11 = tl.load(in_ptr0 + (24576 + x0 + (49152*x1)), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = 0.7071067811865476
    tmp5 = tmp1 * tmp4
    tmp6 = tl.math.erf(tmp5)
    tmp7 = 1.0
    tmp8 = tmp6 + tmp7
    tmp9 = tmp3 * tmp8
    tmp10 = tmp9.to(tl.float32)
    tmp12 = tmp10 * tmp11
    tl.store(out_ptr0 + (x2), tmp12, None)
""",
)


async_compile.wait(globals())
del async_compile


def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = args
    args.clear()
    s0 = arg3_1
    assert_size_stride(arg0_1, (3072,), (1,))
    assert_size_stride(arg1_1, (49152, 3072), (3072, 1))
    assert_size_stride(arg2_1, (3072, 24576), (24576, 1))
    assert_size_stride(arg4_1, (s0, 3072), (3072, 1))
    assert_size_stride(arg5_1, (s0, 3072), (3072, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)  # no-op to ensure context
        buf0 = empty((s0, 3072), device="cuda", dtype=torch.bfloat16)
        buf2 = empty((s0, 3072), device="cuda", dtype=torch.bfloat16)
        # Source Nodes: [add, attn_res, hidden_states_1, hidden_states_2, hidden_states_3, normed_attn_res_output, pow_1, rsqrt, variance], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt]
        stream0 = get_cuda_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_0.run(
            arg5_1,
            arg4_1,
            arg0_1,
            buf0,
            buf2,
            arg5_1,
            s0,
            3072,
            grid=grid(s0),
            stream=stream0,
        )
        del arg0_1
        del arg4_1
        del arg5_1
        buf3 = empty((s0, 49152), device="cuda", dtype=torch.bfloat16)
        # Source Nodes: [add, gate_up_states, hidden_states_1, hidden_states_2, hidden_states_3, normed_attn_res_output, pow_1, rsqrt, variance], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mm, aten.mul, aten.pow, aten.rsqrt]
        triton_tem_fused__to_copy_add_mean_mm_mul_pow_rsqrt_1.run(
            buf2,
            arg1_1,
            buf3,
            s0,
            grid=torch._inductor.kernel.mm_common.mm_grid(s0, 49152, meta0),
            stream=stream0,
        )
        del arg1_1
        buf4 = empty((s0, 24576), device="cuda", dtype=torch.bfloat16)
        # Source Nodes: [gelu, inter], Original ATen: [aten.gelu, aten.mul]
        triton_poi_fused_gelu_mul_2_xnumel = 24576 * s0
        triton_poi_fused_gelu_mul_2.run(
            buf3,
            buf4,
            triton_poi_fused_gelu_mul_2_xnumel,
            grid=grid(triton_poi_fused_gelu_mul_2_xnumel),
            stream=stream0,
        )
        del buf3
        buf5 = buf2
        del buf2  # reuse
        # Source Nodes: [gelu, inter, mlp_output], Original ATen: [aten.gelu, aten.mm, aten.mul]
        extern_kernels.mm(
            buf4, reinterpret_tensor(arg2_1, (24576, 3072), (1, 24576), 0), out=buf5
        )
        del arg2_1
        return (
            buf5,
            buf0,
        )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance

    arg0_1 = rand_strided((3072,), (1,), device="cuda:0", dtype=torch.bfloat16)
    arg1_1 = rand_strided(
        (49152, 3072), (3072, 1), device="cuda:0", dtype=torch.bfloat16
    )
    arg2_1 = rand_strided(
        (3072, 24576), (24576, 1), device="cuda:0", dtype=torch.bfloat16
    )
    arg3_1 = 256
    arg4_1 = rand_strided((256, 3072), (3072, 1), device="cuda:0", dtype=torch.bfloat16)
    arg5_1 = rand_strided((256, 3072), (3072, 1), device="cuda:0", dtype=torch.bfloat16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main

    compiled_module_main("None", benchmark_compiled_module)
