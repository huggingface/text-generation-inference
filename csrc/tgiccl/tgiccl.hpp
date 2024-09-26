//
// Created by mfuntowicz on 9/25/24.
//

#ifndef TEXT_GENERATION_INFERENCE_TGICCL_H
#define TEXT_GENERATION_INFERENCE_TGICCL_H

#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

constexpr const char *CLL_BACKEND_NAME = "tgiccl";

namespace huggingface::tgi {
    class TgiCcl {
    private:

    public:

    };
}

#endif //TEXT_GENERATION_INFERENCE_TGICCL_H
