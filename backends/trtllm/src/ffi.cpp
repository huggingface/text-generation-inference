//
// Created by mfuntowicz on 6/30/24.
//
#pragma once

#include <cmath>
#include <exception>
#include <filesystem>
#include <limits>
#include <iterator>
#include <vector>

#include <spdlog/spdlog.h>
#include "backends/trtllm/include/ffi.h"


huggingface::tgi::backends::TensorRtLlmBackendImpl::TensorRtLlmBackendImpl(
        const std::string_view &engineFolder,
        const std::string_view &executorWorker
) : TensorRtLlmBackend(engineFolder, executorWorker) {}


bool huggingface::tgi::backends::TensorRtLlmBackendImpl::IsReady() const {
    return TensorRtLlmBackend::IsReady();
}

uint64_t huggingface::tgi::backends::TensorRtLlmBackendImpl::Submit(
        rust::Slice<const uint32_t> tokens, int32_t topK, float_t topP, float_t temperature, float_t repetition_penalty,
        float_t frequency_penalty, uint64_t seed) {

    // This will copy all the items from the initial slice
    std::vector<int32_t> tokens_(std::make_move_iterator(tokens.begin()), std::make_move_iterator(tokens.end()));
    return TensorRtLlmBackend::Submit(
            std::move(tokens_), topK, topP, temperature, repetition_penalty, frequency_penalty, seed);
}

size_t huggingface::tgi::backends::TensorRtLlmBackendImpl::StreamTokens(
        const uint64_t requestId,
        huggingface::tgi::backends::GenerationContext *ctx,
        rust::Fn<void(huggingface::tgi::backends::GenerationContext *,
                      huggingface::tgi::backends::GenerationStep)> callback) {

    size_t numTokens = 0;
    for (const auto &item: Poll(requestId)) {
        GenerationStep step;
        if (!item.hasError()) {
            SPDLOG_DEBUG("\tStreamTokens -> Decoding token...");
            const auto decoded = item.getResult();

            const auto token = decoded.outputTokenIds[0][0];
            const auto isFinal = decoded.isFinal;
            const auto logProb = decoded.logProbs.value()[0][0];

            ++numTokens;

            SPDLOG_DEBUG(FMT_STRING("\tStreamTokens -> {:d} {:.2f} (final = {})"), token, logProb, isFinal);
            step = huggingface::tgi::backends::GenerationStep{
                    static_cast<uint32_t>(token), logProb, isFinal, false, std::move(std::string())
            };
            SPDLOG_DEBUG("\tStreamTokens -> Post callback");
        } else {
            // TODO : Return rest::Result with error
            const auto what = item.getErrorMsg();
            SPDLOG_WARN("\tStreamTokens -> Got error while decoding: {}", what);
            step = huggingface::tgi::backends::GenerationStep{
                    std::numeric_limits<uint32_t>::max(), 0.0, true, true, std::move(what)
            };
        }

        callback(std::move(ctx), std::move(step));
    }

    return numTokens;
}

std::unique_ptr<huggingface::tgi::backends::TensorRtLlmBackendImpl>
huggingface::tgi::backends::CreateTensorRtLlmBackend(rust::Str engineFolder, rust::Str executorWorker) {
    // Unconditionally call this to initialize and discover TRTLLM plugins
    InitializeBackend();

    const auto enginePath = std::string_view(engineFolder.begin(), engineFolder.end());
    const auto executorPath = std::string_view(executorWorker.begin(), executorWorker.end());
    return std::make_unique<TensorRtLlmBackendImpl>(std::move(enginePath), std::move(executorPath));
}
