#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>
#include <string>
#include <torch/extension.h>

struct NcclParam
{
    int rank_{0};
    int world_size_{1};
    ncclUniqueId nccl_uid_;
    ncclComm_t nccl_comm_ = nullptr;
    cudaStream_t stream_;

    NcclParam() : rank_(0), world_size_(1), nccl_comm_(nullptr){};
    NcclParam(int rank, int world_size) : rank_(rank), world_size_(world_size){};
    NcclParam(NcclParam const &param) : rank_(param.rank_), world_size_(param.world_size_), nccl_uid_(param.nccl_uid_), nccl_comm_(param.nccl_comm_), stream_(param.stream_){};
};

#define NCCLCHECK(cmd)                                                                            \
    do                                                                                            \
    {                                                                                             \
        ncclResult_t r = cmd;                                                                     \
        if (r != ncclSuccess)                                                                     \
        {                                                                                         \
            printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(r)); \
            exit(EXIT_FAILURE);                                                                   \
        }                                                                                         \
    } while (0)

std::tuple<NcclParam *, NcclParam *> init_nccl(int tensor_para_size, int pipeline_para_size)
{
    // int    argc = 0;
    // char** argv = nullptr;

    // // 初始化 MPI
    // MPI_Init(&argc, &argv);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);       // 获取当前进程的 rank
    MPI_Comm_size(MPI_COMM_WORLD, &world_size); // 获取进程总数
    // printf("rank:%d, world_size:%d\n", rank, world_size);

    // 设定gpu
    int device, device_count;
    cudaGetDeviceCount(&device_count);
    cudaSetDevice(rank % device_count);
    cudaGetDevice(&device);
    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    int mpi_initialized;
    MPI_Initialized(&mpi_initialized);

    static NcclParam tensor_para;
    static NcclParam pipeline_para;
    // Convert WORLD communicator into 2D grid (k * n) communicator.
    // row = a tensor parallel group, col = a pipeline parallel group.
    MPI_Comm grid_comm, tp_comm, pp_comm;

    int dims[2] = {pipeline_para_size, tensor_para_size};
    int periods[2] = {0, 0};
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &grid_comm);

    // Split 2D communicator into rows and cols.
    int tp_remain_dims[2] = {false, true};
    int pp_remain_dims[2] = {true, false};
    MPI_Cart_sub(grid_comm, tp_remain_dims, &tp_comm);
    MPI_Cart_sub(grid_comm, pp_remain_dims, &pp_comm);

    int tp_rank, pp_rank;
    MPI_Comm_rank(tp_comm, &tp_rank);
    MPI_Comm_rank(pp_comm, &pp_rank);
    printf("tp_rank:%d, pp_rank:%d\n", tp_rank, pp_rank);

    ncclUniqueId tp_uid;
    ncclUniqueId pp_uid;
    // The root of each group creates a nccl uid.
    if (tp_rank == 0)
    {
        NCCLCHECK(ncclGetUniqueId(&tp_uid));
    }
    if (pp_rank == 0)
    {
        NCCLCHECK(ncclGetUniqueId(&pp_uid));
    }
    // Broadcast nccl uid to share the same nccl uid across gpus in the same group.
    MPI_Bcast(&tp_uid, sizeof(tp_uid), MPI_BYTE, 0, tp_comm);
    MPI_Bcast(&pp_uid, sizeof(pp_uid), MPI_BYTE, 0, pp_comm);

    ncclComm_t tp_nccl_comm, pp_nccl_comm;
    NCCLCHECK(ncclCommInitRank(&tp_nccl_comm, tensor_para_size, tp_uid, tp_rank));
    NCCLCHECK(ncclCommInitRank(&pp_nccl_comm, pipeline_para_size, pp_uid, pp_rank));

    tensor_para.world_size_ = tensor_para_size;
    tensor_para.rank_ = tp_rank;
    tensor_para.nccl_uid_ = tp_uid;
    tensor_para.nccl_comm_ = tp_nccl_comm;
    cudaStreamCreate(&tensor_para.stream_);
    pipeline_para.world_size_ = pipeline_para_size;
    pipeline_para.rank_ = pp_rank;
    pipeline_para.nccl_uid_ = pp_uid;
    pipeline_para.nccl_comm_ = pp_nccl_comm;

    NcclParam *tensor_para_ptr = &tensor_para;
    NcclParam *pipeline_para_ptr = &pipeline_para;

    return std::make_tuple(tensor_para_ptr, pipeline_para_ptr);
}

void finalize_nccl(NcclParam *tensor_para_ptr, NcclParam *pipeline_para_ptr)
{
    // 销毁nccl
    NcclParam tensor_para = *tensor_para_ptr;
    NcclParam pipeline_para = *pipeline_para_ptr;
    if (tensor_para.nccl_comm_ != nullptr)
    {
        ncclCommDestroy(tensor_para.nccl_comm_);
    }
    if (pipeline_para.nccl_comm_ != nullptr)
    {
        ncclCommDestroy(pipeline_para.nccl_comm_);
    }
    // MPI_Finalize();
}

ncclDataType_t getNcclDataType(torch::ScalarType torch_type)
{
    ncclDataType_t nccl_type;
    if (torch_type == torch::kFloat16)
    {
        nccl_type = ncclHalf;
    }
    else if (torch_type == torch::kFloat32)
    {
        nccl_type = ncclFloat;
    }
    else if (torch_type == torch::kFloat64)
    {
        nccl_type = ncclDouble;
    }
    else if (torch_type == torch::kInt32)
    {
        nccl_type = ncclInt32;
    }
    else if (torch_type == torch::kInt64)
    {
        nccl_type = ncclInt64;
    }
    else if (torch_type == torch::kInt8)
    {
        nccl_type = ncclInt8;
    }
    else
    {
        printf("[ERROR] NCCL only support float, half, int \n");
        exit(-1);
    }

    return nccl_type;
}
void custom_allreduce(torch::Tensor tensor, NcclParam *nccl_param_ptr)
{
    void *data_ptr = tensor.data_ptr();
    size_t data_size = tensor.numel();
    torch::ScalarType torch_type = tensor.scalar_type();

    NcclParam nccl_param = *nccl_param_ptr;
    // cudaStream_t   stream         = at::cuda::getCurrentCUDAStream();
    ncclDataType_t nccl_data_type = getNcclDataType(torch_type);
    NCCLCHECK(ncclGroupStart());
    NCCLCHECK(ncclAllReduce(
        (const void *)data_ptr, (void *)data_ptr, data_size, nccl_data_type, ncclSum, nccl_param.nccl_comm_, nccl_param.stream_));
    NCCLCHECK(ncclGroupEnd());
}
void custom_allgather_into_tensor(torch::Tensor recv_tensor, torch::Tensor send_tensor, NcclParam *nccl_param_ptr)
{
    void *send_ptr = send_tensor.data_ptr();
    void *recv_ptr = recv_tensor.data_ptr();
    size_t data_size = send_tensor.numel();

    torch::ScalarType torch_type = send_tensor.scalar_type();

    NcclParam nccl_param = *nccl_param_ptr;
    ncclDataType_t nccl_data_type = getNcclDataType(torch_type);
    NCCLCHECK(ncclGroupStart());
    NCCLCHECK(ncclAllGather(
        (const void *)(send_ptr), (void *)recv_ptr, data_size, nccl_data_type, nccl_param.nccl_comm_, nccl_param.stream_));
    NCCLCHECK(ncclGroupEnd());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    py::class_<NcclParam>(m, "NcclParam").def(py::init<>());
    m.def("init_nccl", &init_nccl, py::return_value_policy::reference, "");
    m.def("finalize_nccl", &finalize_nccl, "");
    m.def("custom_allreduce", &custom_allreduce, "");
    m.def("custom_allgather_into_tensor", &custom_allgather_into_tensor, "");
}