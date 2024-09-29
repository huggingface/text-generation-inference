//
// Created by morgan on 26/09/24.
//

#include <vector>
#include <torch/torch.h>
#include "tgiccl.hpp"

int main() {
    auto backend = huggingface::tgi::tgiccl::TgiCclBackend(0, 4);
    auto tensor = torch::zeros({128});
    auto tensors = std::vector<torch::Tensor>();
    tensors.push_back(tensor);
    backend.allreduce(tensors, c10d::AllreduceOptions());
}