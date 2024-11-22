//
// Created by mfuntowicz on 10/23/24.
//

#ifndef TGI_LLAMA_CPP_BACKEND_FFI_HPP
#define TGI_LLAMA_CPP_BACKEND_FFI_HPP

#include <cstdint>
#include <exception>
#include <filesystem>
#include <memory>
#include <ranges>
#include <string_view>
#include <thread>

#include <spdlog/spdlog.h>
#include <spdlog/fmt/ranges.h>
#include <spdlog/fmt/std.h>

#ifdef NUMA_AVAILABLE
#define CURRENT_THREAD 0
#include <algorithm>
#include <unordered_set>
#include <numa.h>
#endif

namespace huggingface::tgi::backends::llamacpp {
    class llama_cpp_worker_frontend_t;
}

#include "backend.hpp"
#include "backends/llamacpp/src/lib.rs.h"
#include "rust/cxx.h"


namespace huggingface::tgi::backends::llamacpp {

    auto llama_model_deleter = [](llama_model *model) { llama_free_model(model); };
    auto make_shared_llama_model = [](llama_model *model) {
        return std::shared_ptr<llama_model>(model, llama_model_deleter);
    };

    class llama_cpp_backend_exception_t : std::exception {};

    /**
     * Llama.cpp frontend over the worker interfacing with Rust FFI layer
     */
    class llama_cpp_worker_frontend_t {
    private:
        std::shared_ptr<llama_model> model_;
        worker_t worker_;

    public:
        explicit llama_cpp_worker_frontend_t(llama_model *model, int32_t num_threads):
            model_{ make_shared_llama_model(model) }, worker_(model_, {.n_ubatch = 1, .n_threads = num_threads, .no_perf = true}) {}

        size_t stream(
                rust::Slice<const uint32_t> input_tokens,
                const generation_params_t generation_params,
                const sampling_params_t &sampling_params,
                InferContext *ctx,
                rust::Fn<bool(InferContext *, uint32_t, float_t, bool, size_t)> callback
        ) {
            // Wrapper around the provided Rust callback to inject the InferContext when returning from the C++ FFI boundaries
            // It captures the context (ctx) using reference and will automatically call the Rust callback forwarding the InferContext
            auto context_forwarding_callback =
                    [=, &ctx](uint32_t new_token_id, float_t logits, bool is_eos, size_t n_generated_tokens) -> bool {
                return callback(ctx, new_token_id, logits, is_eos, n_generated_tokens);
            };

            // Ask the compiler to create view over Rust slice transmuting from uint32_t* to llama_token*
            static auto as_llama_token = [](const uint32_t x){ return static_cast<llama_token>(x); };

#ifdef __cpp_lib_ranges_to_container
            auto input_tokens_v = input_tokens | std::views::transform(as_llama_token) | std::ranges::to<std::vector>();
#else
            auto input_tokens_ = input_tokens | std::views::transform(as_llama_token);
            auto input_tokens_v = std::vector<llama_token>(input_tokens_.begin(), input_tokens_.end());
#endif

            // Defer the generation to the actual worker_t
            const auto generation_context = generation_context_t {generation_params, sampling_params, input_tokens_v};
            if(const auto result = worker_.generate(generation_context, context_forwarding_callback); result.has_value()) [[likely]] {
                return *result;
            } else {
                throw llama_cpp_backend_exception_t {};
            }
        }
    };

    std::unique_ptr<llama_cpp_worker_frontend_t> create_worker_frontend(rust::Str modelPath, uint32_t num_threads) {
#ifdef TGI_LLAMACPP_BACKEND_DEBUG
        spdlog::set_level(spdlog::level::debug);
#endif

        // Initialize the numa context from numactl
        static const bool INITIALIZED_NUMA_CONTEXT_ONCE = [](){
            llama_numa_init(GGML_NUMA_STRATEGY_NUMACTL);
            return true;
        }();

        // Allocate model weights parameters
        auto params = llama_model_default_params();
        params.use_mmap = true;

        // Allocate the model from the Rust provided, string path
        auto *model = (llama_load_model_from_file(static_cast<std::string>(modelPath).c_str(), params));
        return std::make_unique<llama_cpp_worker_frontend_t>(model, static_cast<int32_t>(num_threads));
    }

    struct numa_cpumask_deleter { void operator()(struct bitmask* cpumask){ numa_free_cpumask(cpumask); }};
    typedef std::unique_ptr<struct bitmask, numa_cpumask_deleter> unique_cpumask_ptr;

    void set_numa_core_affinity(rust::Slice<const size_t> affinity) {
//    void set_numactl_core_affinity(std::vector<size_t> affinity) {
#ifdef NUMA_AVAILABLE
        if(numa_available()) {
            SPDLOG_INFO("Setting numactl cores affinity to {} for thread {}", affinity, std::this_thread::get_id());

            auto cpumask = unique_cpumask_ptr(numa_allocate_cpumask());
            std::ranges::for_each(affinity, [&cpumask](size_t cpu) { numa_bitmask_setbit(cpumask.get(), cpu); });
            numa_sched_setaffinity(CURRENT_THREAD, cpumask.get());

            // Retrieve some information about the current setup
            if(const auto numa_num_nodes = numa_num_configured_nodes(); numa_num_nodes > 1) {
                const auto *numa_all_cpus = numa_all_cpus_ptr;
                SPDLOG_INFO(FMT_STRING("All CPUs: {:b} (# Nodes: {:d}"), *numa_all_cpus->maskp, numa_num_nodes);

                // Retrieve the cpumask specific for the current node
                auto cpus_per_node = unique_cpumask_ptr(numa_allocate_cpumask());

                // Allocate a set which keeps track of which nodes is being targeted
                auto numa_spawning_nodes = std::unordered_set<size_t>();
                for(auto node = 0; node < numa_num_nodes; ++node) {
                    // Retrieve the cpumask for the target node
                    numa_node_to_cpus(node, cpus_per_node.get());

                    // intersect which cores on the nodes are targeted, in no one on that specific node
                    // the value of allocated_cpus_on_node will be 0 as the result of the AND operation.
                    const auto allocated_cpus_on_node = *cpus_per_node->maskp & *cpumask->maskp;
                    if(allocated_cpus_on_node > 0) {

                        // If we have some cores on the node, attempt to insert in the set of targeted node
                        if(const auto [_, was_inserted] = numa_spawning_nodes.emplace(node); was_inserted) {
                            SPDLOG_DEBUG("Allocated thread spawning node: {:d}", node);
                        }
                    }

                    // Clear all the bits relative to the current node
                    numa_bitmask_clearall(cpus_per_node.get());
                }

                // Bind the memory if we spawn a single node, otherwise, let's display a warning
                if(numa_spawning_nodes.size() == 1) {
                    SPDLOG_INFO(FMT_STRING("Setting memory affinity to node: {:d}"), *numa_spawning_nodes.begin());
                    numa_set_preferred(*numa_spawning_nodes.begin());
                } else {
                    SPDLOG_WARN(FMT_STRING("Specified thread affinity spawn multiple NUMA nodes: {}"), numa_spawning_nodes);
                }
            }

#ifdef TGI_LLAMACPP_BACKEND_DEBUG
            // Sanity check in the logs...
            auto *cpumask_check = numa_allocate_cpumask();
            numa_sched_getaffinity(CURRENT_THREAD, cpumask_check);
            SPDLOG_DEBUG(
                    FMT_STRING("numa_sched_affinity for thread {} -> {:b}"),
                    std::this_thread::get_id(), *cpumask_check->maskp);
            numa_free_cpumask(cpumask_check);
#endif
        }
#else
        SPDLOG_WARN("TGI's llama.cpp backend was compiled without NUMA support");
#endif
    }

    /**
     * 
     */
    void update_numa_affinity() {
        SPDLOG_INFO("Rebinding NUMA affinity for current worker on thread: {}", std::this_thread::get_id());
        llama_numa_init(ggml_numa_strategy::GGML_NUMA_STRATEGY_NUMACTL);
    }
}


#endif //TGI_LLAMA_CPP_BACKEND_FFI_HPP
