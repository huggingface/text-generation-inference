//
// Created by Morgan Funtowicz on 26/09/24.
//

#ifndef TGICCLPROCESSGROUP_H
#define TGICCLPROCESSGROUP_H

#include <spdlog/spdlog.h>
#include <torch/csrc/distributed/c10d/Backend.hpp>


namespace huggingface::tgi::tgiccl
{
    void InitTgiCcl();

    class TgiCclBackend final : c10d::Backend {
    public:
        TgiCclBackend(const int rank, const int size): Backend(rank, size)
        {
            SPDLOG_INFO(FMT_STRING("Creating TgiCclBackend on rank {:d} over {:d}"), rank, size);
        }

        c10::intrusive_ptr<c10d::Work> allreduce(std::vector<at::Tensor>&, const c10d::AllreduceOptions&) override;
    };
}



#endif //TGICCLPROCESSGROUP_H
