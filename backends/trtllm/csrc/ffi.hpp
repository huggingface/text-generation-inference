
#include <tensorrt_llm/common/tllmException.h>

namespace rust::behavior {
    template<typename Try, typename Fail>
    static void trycatch(Try &&func, Fail &&fail) noexcept try {
        func();
    } catch (tensorrt_llm::common::TllmException &e) {
        fail(e.what());
    }
}

#include <backend.hpp>

namespace huggingface::tgi::backends::trtllm {

    class tensorrt_llm_backend_t {
    private:
        backend_t inner_;

    public:
        tensorrt_llm_backend_t(std::filesystem::path &engine_folder): inner_(engine_folder) {}

        size_t num_tokens_ready() const noexcept {
            return inner_.num_tokens_ready();
        }

        request_id_t submit(
                rust::Slice<const uint32_t> tokens,
                uint32_t max_new_tokens,
                uint32_t top_k,
                float_t top_p,
                float_t temperature,
                float_t repetition_penalty,
                float_t frequency_penalty,
                uint64_t seed
        ) {
            // Submit the request to the executor and get back a potential request_id used to track request status
            const auto maybe_request_id = inner_.submit(
                {tokens_.data(), tokens.size()},
                {max_new_tokens},
                {top_k, top_p, repetition_penalty, frequency_penalty, temperature, seed}
            );

            // If we do have a value, let's return the request_id
            if(maybe_request_id.has_value()) [[likely]] {
                return *maybe_request_id;
            } else {

            }
        }

        void cancel(request_id_t requestId) noexcept {
            SPDLOG
            inner_.cancel(requestId);
        }
    };


}