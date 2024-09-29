//
// Created by Morgan Funtowicz on 26/09/24.
//

#include "TgiCclBackend.hpp"


auto fmt::formatter<c10d::ReduceOp>::format(c10d::ReduceOp op, format_context& ctx) const -> format_context::iterator {
    string_view name = "unknown";
    switch (op) {
        case c10d::ReduceOp::AVG:          name = "ReduceOp::AVG"; break;
        case c10d::ReduceOp::BAND:         name = "ReduceOp::BAND"; break;
        case c10d::ReduceOp::BOR:          name = "ReduceOp::BOR"; break;
        case c10d::ReduceOp::BXOR:         name = "ReduceOp::BXOR"; break;
        case c10d::ReduceOp::MAX:          name = "ReduceOp::MAX"; break;
        case c10d::ReduceOp::MIN:          name = "ReduceOp::MIN"; break;
        case c10d::ReduceOp::PREMUL_SUM:   name = "ReduceOp::PREMUL_SUM"; break;
        case c10d::ReduceOp::PRODUCT:      name = "ReduceOp::PRODUCT"; break;
        case c10d::ReduceOp::SUM:          name = "ReduceOp::SUM"; break;
        case c10d::ReduceOp::UNUSED:       name = "ReduceOp::UNUSED"; break;
    }
    return formatter<string_view>::format(name, ctx);
}



void huggingface::tgi::tgiccl::InitTgiCcl()
{

}

huggingface::tgi::tgiccl::TgiCclBackend::TgiCclBackend(const int rank, const int size) : Backend(rank, size) {
    SPDLOG_INFO(FMT_STRING("Creating {} on rank {:d} (world_size={:d})"), getBackendName(), rank, size);

}

const std::string huggingface::tgi::tgiccl::TgiCclBackend::getBackendName() const {
    return CCL_BACKEND_NAME;
}

c10::intrusive_ptr<c10d::Work>
huggingface::tgi::tgiccl::TgiCclBackend::allreduce(std::vector<at::Tensor> &tensors, const c10d::AllreduceOptions &options) {
    TORCH_CHECK(options.reduceOp == c10d::ReduceOp::SUM, fmt::format(FMT_STRING("tgiccl only supports ReduceOp::SUM, got {}"), options.reduceOp))
    tensors[0] += 1;
    return c10::make_intrusive<c10d::Work>();
}
