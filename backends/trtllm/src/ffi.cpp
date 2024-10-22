//
// Created by mfuntowicz on 6/30/24.
//
#pragma once

#include <algorithm>
#include <exception>
#include <filesystem>
#include <functional>
#include <limits>
#include <iterator>
#include <ranges>
#include <vector>

#include <spdlog/spdlog.h>
#include "backends/trtllm/include/ffi.h"


huggingface::tgi::backends::TensorRtLlmBackendImpl::TensorRtLlmBackendImpl(
        const std::string_view &engineFolder,
        const std::string_view &executorWorker
) : TensorRtLlmBackend(engineFolder, executorWorker) {}


uint64_t huggingface::tgi::backends::TensorRtLlmBackendImpl::Submit(
        rust::Slice<const uint32_t> tokens,
        uint32_t maxNewTokens,
        int32_t topK,
        float_t topP,
        float_t temperature,
        float_t repetition_penalty,
        float_t frequency_penalty,
        uint64_t seed) {

    // This will copy all the items from the initial slice
    std::vector<int32_t> tokens_(tokens.begin(), tokens.end());
    return TensorRtLlmBackend::Submit(
            std::move(tokens_), maxNewTokens, topK, topP, temperature, repetition_penalty, frequency_penalty, seed);
}

std::unique_ptr<std::vector<huggingface::tgi::backends::GenerationStep>>
huggingface::tgi::backends::TensorRtLlmBackendImpl::PullTokens() {
    const auto responses = TensorRtLlmBackend::PullNewTokens();

    auto steps = std::make_unique<std::vector<GenerationStep>>();
    steps->reserve(responses.size());

#ifndef NDEBUG
    SPDLOG_DEBUG(FMT_STRING("Pulled out {:d} new tokens"), responses->size());
#endif

    // Transform tle::Response to GenerationStep
    std::ranges::transform(responses.begin(), responses.end(), std::back_inserter(*steps), [](const tle::Response &r) {
        const auto reqId = r.getRequestId();
        if (!r.hasError()) {
            const auto result = r.getResult();
            return GenerationStep{
                    reqId,
                    static_cast<uint32_t>(result.outputTokenIds[0][0]),
                    result.logProbs.value()[0][0],
                    result.isFinal,
                    false,
                    std::string()
            };
        } else {
            return GenerationStep{
                    reqId,
                    0,
                    0.0,
                    true,
                    true,
                    std::move(r.getErrorMsg())
            };
        }
    });

    return steps;
}

std::unique_ptr<huggingface::tgi::backends::TensorRtLlmBackendImpl>
huggingface::tgi::backends::CreateTensorRtLlmBackend(rust::Str engineFolder, rust::Str executorWorker) {
    SPDLOG_INFO("Creating TensorRT-LLM Backend");
    // Unconditionally call this to initialize and discover TRTLLM plugins
    InitializeBackend();

    const auto enginePath = std::string_view(engineFolder.begin(), engineFolder.end());
    const auto executorPath = std::string_view(executorWorker.begin(), executorWorker.end());
    return std::make_unique<TensorRtLlmBackendImpl>(std::move(enginePath), std::move(executorPath));
}
