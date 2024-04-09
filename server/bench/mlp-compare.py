import torch
import torch._dynamo.config
import torch._inductor.config
from torch.nn import functional as F

import torch

# prefer using pre-compiled ops

# from text_generation_server.utils.layers import (
#     TensorParallelColumnLinear,
#     TensorParallelRowLinear,
# )


torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(0)
# share input for both cases
x = torch.randn(4096, 4096, device=device)


class DummyLayer(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.gate_up_proj = torch.nn.Linear(4096, 4096).to(device)
        self.down_proj = torch.nn.Linear(4096, 4096).to(device)
        self.act = torch.nn.GELU().to(device)

    def forward(self, x):
        y = self.gate_up_proj(x)
        y = self.act(y)
        y = self.down_proj(y)
        return y


class DummyModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        # add multiple layers to magnify the effect
        N = 10
        self.layer = torch.nn.Sequential(*[DummyLayer() for _ in range(N)]).to(device)

    def forward(self, x):
        return self.layer(x)


model = DummyModule()

print("Model")
print(model)


# run the model via a forward pass
def forward_pass(x):
    return model(x)


# same as above but compiled
forward_pass_compiled = torch.compile(
    forward_pass, mode="reduce-overhead", fullgraph=True
)


# one pass to avoid the compilation overhead
y = forward_pass_compiled(x)

# start profiling
torch.profiler._utils._init_for_cuda_graphs()
prof = torch.profiler.profile()

# run on compiled model
with prof:
    y = forward_pass_compiled(x)
    prof.step()

print("Compiled")
print(prof.key_averages().table(sort_by="self_cuda_time_total"))

# one pass to avoid the compilation overhead (just to align with the compiled case)
y = forward_pass(x)

# remove the profiling data to avoid any contamination
del prof

# start a new profiling session
torch.profiler._utils._init_for_cuda_graphs()
prof = torch.profiler.profile()

# run on non-compiled model
with prof:
    y = forward_pass(x)
    prof.step()

print("")
print("Not Compiled")
print(prof.key_averages().table(sort_by="self_cuda_time_total"))


# Expected optimized code:

# {"XBLOCK": 256, "num_warps": 8, "num_stages": 1, "configs_hash": "3ca5c3e34d35093f3c9ab2829a9faeebad5e61c4ca13d5ed6053d7b71ce60d5a", "found_by_coordesc": true}
# coordesc: https://github.com/pytorch/pytorch/blob/main/torch/_inductor/triton_heuristics.py#L652

# import triton
# import triton.language as tl
# from torch._inductor.ir import ReductionHint
# from torch._inductor.ir import TileHint
# from torch._inductor.triton_heuristics import AutotuneHint, pointwise
# from torch._inductor.utils import instance_descriptor
# from torch._inductor import triton_helpers

# @pointwise(
#     size_hints=[16777216],
#     filename=__file__,
#     triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
#     inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_0', 'mutated_arg_names': []},
#     min_elem_per_thread=0
# )
# @triton.jit
# def triton_poi_fused_gelu_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
#     xnumel = 16777216
#     xoffset = tl.program_id(0) * XBLOCK
#     xindex = xoffset + tl.arange(0, XBLOCK)[:]
#     xmask = xindex < xnumel
#     x0 = xindex
#     tmp0 = tl.load(in_ptr0 + (x0), None)
#     tmp1 = 0.5
#     tmp2 = tmp0 * tmp1
#     tmp3 = 0.7071067811865476
#     tmp4 = tmp0 * tmp3
#     tmp5 = tl.math.erf(tmp4)
#     tmp6 = 1.0
#     tmp7 = tmp5 + tmp6
#     tmp8 = tmp2 * tmp7
#     tl.store(out_ptr0 + (x0), tmp8, None)


# Compiled
# -------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
#                                              Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls
# -------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
#                                       aten::addmm         0.39%     586.000us         0.63%     945.000us      47.250us     144.767ms        98.15%     145.039ms       7.252ms            20
#                           ampere_sgemm_128x128_tn         0.00%       0.000us         0.00%       0.000us       0.000us     144.748ms        98.13%     144.748ms       7.237ms            20
#                                        cudaMalloc         3.20%       4.804ms         3.20%       4.804ms     160.133us       7.329ms         4.97%       7.329ms     244.300us            30
#                                   cudaEventRecord         0.00%       5.000us         0.00%       5.000us       5.000us       6.778ms         4.60%       6.778ms       6.778ms             1
#                           triton_poi_fused_gelu_0         0.18%     271.000us         0.25%     372.000us      37.200us       2.732ms         1.85%       2.732ms     273.200us            10
#                   triton_poi_fused_gelu_0_0d1d2de         0.00%       0.000us         0.00%       0.000us       0.000us       2.732ms         1.85%       2.732ms     273.200us            10
#                             cudaStreamIsCapturing         0.02%      30.000us         0.02%      30.000us       1.000us     272.000us         0.18%     272.000us       9.067us            30
#     cudaOccupancyMaxActiveBlocksPerMultiprocessor         0.01%      22.000us         0.01%      22.000us       1.100us     272.000us         0.18%     272.000us      13.600us            20
#                                   Memset (Device)         0.00%       0.000us         0.00%       0.000us       0.000us      19.000us         0.01%      19.000us       0.950us            20
#                          TorchDynamo Cache Lookup         0.03%      41.000us         0.03%      41.000us      41.000us       0.000us         0.00%       0.000us       0.000us             1
#                             Torch-Compiled Region         0.09%     141.000us        99.97%     150.195ms     150.195ms       0.000us         0.00%     162.150ms     162.150ms             1
#                                      aten::detach         0.00%       4.000us         0.01%      14.000us      14.000us       0.000us         0.00%       0.000us       0.000us             1
#                                            detach         0.01%      10.000us         0.01%      10.000us      10.000us       0.000us         0.00%       0.000us       0.000us             1
#                                  CompiledFunction         2.37%       3.566ms        99.87%     150.040ms     150.040ms       0.000us         0.00%     162.150ms     162.150ms             1
#                             cudaDeviceSynchronize        93.08%     139.840ms        93.08%     139.840ms      46.613ms       0.000us         0.00%       0.000us       0.000us             3
#                               cudaStreamWaitEvent         0.00%       2.000us         0.00%       2.000us       2.000us       0.000us         0.00%       0.000us       0.000us             1
#                                       aten::empty         0.28%     426.000us         3.50%       5.260ms     175.333us       0.000us         0.00%       7.601ms     253.367us            30
#                     inductor::_reinterpret_tensor         0.04%      55.000us         0.04%      55.000us       1.410us       0.000us         0.00%       0.000us       0.000us            39
#                                   cudaMemsetAsync         0.09%     136.000us         0.09%     136.000us       6.800us       0.000us         0.00%       0.000us       0.000us            20
#                                  cudaLaunchKernel         0.13%     201.000us         0.13%     201.000us      10.050us       0.000us         0.00%       0.000us       0.000us            20
#                                    cuLaunchKernel         0.07%     101.000us         0.07%     101.000us      10.100us       0.000us         0.00%       0.000us       0.000us            10
# -------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
# Self CPU time total: 150.241ms
# Self CUDA time total: 147.499ms


# Not Compiled
# -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
#                                                    Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls
# -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
#                                             aten::addmm         0.67%       1.885ms         1.93%       5.415ms     270.750us     145.866ms        98.15%     145.866ms       7.293ms            20
#                                 ampere_sgemm_128x128_tn         0.00%       0.000us         0.00%       0.000us       0.000us     145.847ms        98.14%     145.847ms       7.292ms            20
#                                              aten::gelu         0.09%     245.000us         0.70%       1.955ms     195.500us       2.747ms         1.85%       2.747ms     274.700us            10
# void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       2.747ms         1.85%       2.747ms     274.700us            10
#                                         Memset (Device)         0.00%       0.000us         0.00%       0.000us       0.000us      19.000us         0.01%      19.000us       0.950us            20
#                                            aten::linear         0.03%      82.000us         2.02%       5.671ms     283.550us       0.000us         0.00%     145.866ms       7.293ms            20
#                                                 aten::t         0.04%     110.000us         0.06%     174.000us       8.700us       0.000us         0.00%       0.000us       0.000us            20
#                                         aten::transpose         0.01%      42.000us         0.02%      64.000us       3.200us       0.000us         0.00%       0.000us       0.000us            20
#                                        aten::as_strided         0.01%      22.000us         0.01%      22.000us       1.100us       0.000us         0.00%       0.000us       0.000us            20
#                                   cudaStreamIsCapturing         0.01%      30.000us         0.01%      30.000us       1.000us       0.000us         0.00%       0.000us       0.000us            30
#                                              cudaMalloc         1.70%       4.770ms         1.70%       4.770ms     159.000us       0.000us         0.00%       0.000us       0.000us            30
#                                         cudaMemsetAsync         0.04%     124.000us         0.04%     124.000us       6.200us       0.000us         0.00%       0.000us       0.000us            20
#           cudaOccupancyMaxActiveBlocksPerMultiprocessor         0.01%      20.000us         0.01%      20.000us       1.000us       0.000us         0.00%       0.000us       0.000us            20
#                                        cudaLaunchKernel         0.11%     296.000us         0.11%     296.000us       9.867us       0.000us         0.00%       0.000us       0.000us            30
#                                   cudaDeviceSynchronize        97.28%     272.916ms        97.28%     272.916ms     272.916ms       0.000us         0.00%       0.000us       0.000us             1
# -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
# Self CPU time total: 280.542ms
# Self CUDA time total: 148.613ms
