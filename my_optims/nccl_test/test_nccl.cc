#include "src/fastertransformer/utils/mpi_utils.h"
#include "src/fastertransformer/utils/nccl_utils.h"
#include "src/fastertransformer/utils/nvtx_utils.h"
#include "src/fastertransformer/utils/memory_utils.h"
#include <string>
#include <cuda_profiler_api.h>


using namespace fastertransformer;

template<typename T>
void test_nccl();

int main(int argc, char **argv){
    mpi::initialize(&argc, &argv);

    test_nccl<half>();

    mpi::finalize();

    return 0;
}


template<typename T>
void test_nccl()
{
    // int rank       = 0;
    // int world_size = 1;
    int rank       = mpi::getCommWorldRank();
    int world_size = mpi::getCommWorldSize();
    if (rank == 0) {
        printf("Total ranks: %d.\n", world_size);
    }
    int device, device_count;
    check_cuda_error(cudaGetDeviceCount(&device_count));
    check_cuda_error(cudaSetDevice(rank % device_count));
    check_cuda_error(cudaGetDevice(&device));
    struct cudaDeviceProp prop;
    check_cuda_error(cudaGetDeviceProperties(&prop, device));
    printf("Device %s\n", prop.name);
    printf("P%d is running with GPU #%d.\n", rank, device);
    print_mem_usage();

    // 声明数据指针
    std::vector<T*> weights_ptr        = std::vector<T*>(1);
    size_t shape = 4096 * 2048;
    deviceMalloc(&weights_ptr[0], shape, true); // 从gpu中开辟空间，并随即初始化
    // deviceFill(weights_ptr[0], (size_t)shape, (T)2.0); // 给gpu空间赋值

    // 初始化nccl
    int tensor_para_size   = 2;
    int pipeline_para_size = 1;
    NcclParam tensor_para;
    NcclParam pipeline_para;
    ftNcclInitialize(tensor_para, pipeline_para, tensor_para_size, pipeline_para_size);
    std::cout << "tensor_para info:" << tensor_para.rank_ << std::endl;

    cudaStream_t     stream;
    cudaStreamCreate(&stream);
    
    mpi::barrier();
    cudaProfilerStart();
    ft_nvtx::setScope("run_time");
    PUSH_RANGE("run time")
    for (int i = 0; i < 32; i++) {
        cudaDeviceSynchronize();
        ftNcclAllReduceSum(weights_ptr[0], weights_ptr[0], shape, tensor_para, stream);
        cudaDeviceSynchronize();
        deviceMalloc(&weights_ptr[0], shape, true);
    }

    mpi::barrier();
    POP_RANGE;
    ft_nvtx::resetScope();

    // T* hBuf = new T[shape]; // 开辟CPU空间
    // cudaD2Hcpy(hBuf, weights_ptr[0], shape);
    // { // 从cpu打印，查看数据
    //     for (size_t i = 0; i < shape; i++) {
    //         printf("%f ", (float)hBuf[i]);
    //     }
    //     std::cout << std::endl;
    // }
    // delete[] hBuf;

    return;
}