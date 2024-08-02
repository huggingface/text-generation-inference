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
        rust::Slice<const uint32_t> tokens, uint32_t maxNewTokens,
        int32_t topK, float_t topP, float_t temperature,
        float_t repetition_penalty, float_t frequency_penalty, uint64_t seed) {

    // This will copy all the items from the initial slice
    std::vector<int32_t> tokens_(std::make_move_iterator(tokens.begin()), std::make_move_iterator(tokens.end()));
    return TensorRtLlmBackend::Submit(
            std::move(tokens_), maxNewTokens, topK, topP, temperature, repetition_penalty, frequency_penalty, seed);
}

std::unique_ptr<std::vector<huggingface::tgi::backends::GenerationStep>>
huggingface::tgi::backends::TensorRtLlmBackendImpl::PullTokens() {
    const auto responses = TensorRtLlmBackend::PullNewTokens();
    auto steps = std::make_unique<std::vector<GenerationStep>>(responses.size());
    std::ranges::copy(std::views::transform(responses, ConvertResponseToGenerationStep), std::back_inserter(*steps));
    return steps;
}

huggingface::tgi::backends::GenerationStep
huggingface::tgi::backends::ConvertResponseToGenerationStep(const tle::Response &response) {
    const auto reqId = response.getRequestId();
    if (!response.hasError()) {
        const auto result = response.getResult();
        return std::move(GenerationStep{
                reqId,
                result.outputTokenIds[0][0],
                result.logProbs.value()[0][0],
                result.isFinal,
                false,
                std::string()
        });
    } else {
        return std::move(GenerationStep{
                reqId,
                0,
                0.0,
                true,
                true,
                std::move(response.getErrorMsg())
        });
    }
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
