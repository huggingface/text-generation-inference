//
// Created by Morgan Funtowicz on 26/09/24.
//

#ifndef TGICCLPROCESSGROUP_H
#define TGICCLPROCESSGROUP_H

#include <spdlog/spdlog.h>
#include <torch/csrc/distributed/c10d/Backend.hpp>


template <> struct fmt::formatter<c10d::ReduceOp>: formatter<string_view> {
    auto format(c10d::ReduceOp op, format_context& ctx) const -> format_context::iterator;
};



namespace huggingface::tgi::tgiccl
{
#define CCL_BACKEND_NAME "tgiccl";

    void InitTgiCcl();

    class TgiCclBackend;
    class TgiCclBackendWork final: c10d::Work {
        friend TgiCclBackend;


    };


    class TgiCclBackend final : c10d::Backend {
    public:
        TgiCclBackend(int rank, int size);
        const std::string getBackendName() const override;
        c10::intrusive_ptr<c10d::Work> allreduce(std::vector<at::Tensor>&, const c10d::AllreduceOptions&) override;
    };
}



#endif //TGICCLPROCESSGROUP_H
