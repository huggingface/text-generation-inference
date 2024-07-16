//
// Created by mfuntowicz on 6/30/24.
//
#pragma once

#include <cmath>
#include <filesystem>
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
        rust::Slice<const uint32_t> tokens, int32_t topK, float_t topP, float_t temperature, uint64_t seed) {

    // This will copy all the items from the initial slice
    std::vector<int32_t> tokens_(tokens.size());
    tokens_.assign(tokens.begin(), tokens.end());

    return TensorRtLlmBackend::Submit(std::move(tokens_), topK, topP, temperature, seed);
}

size_t huggingface::tgi::backends::TensorRtLlmBackendImpl::StreamTokens(const uint64_t requestId,
                                             rust::Box<huggingface::tgi::backends::GenerationContext> ctx,
                                             rust::Fn<void(rust::Box<huggingface::tgi::backends::GenerationContext>, uint32_t, float_t, bool)> callback) {

    SPDLOG_INFO("Entering StreamTokens");
    for (const auto &item: Poll(requestId)) {
        if (!item.hasError()) {
            SPDLOG_INFO("\tStreamTokens -> Decoding token...");
            const auto decoded = item.getResult();
            SPDLOG_INFO("\tStreamTokens -> Successfully read decoded token ({})", decoded.outputTokenIds[0].size());

            const auto token = decoded.outputTokenIds[0][0];
            const auto isFinal = decoded.isFinal;
//            const auto logProb = decoded.logProbs.value()[0][0];
            const auto logProb = 0.0;

            SPDLOG_INFO(FMT_STRING("\tStreamTokens -> {:d} {:.2f} (final = {})"), token, logProb, isFinal);
            callback(std::move(ctx), token, logProb, isFinal);
            SPDLOG_INFO("\tStreamTokens -> Post callback");
        } else {
            // TODO : Return rest::Result with error
            SPDLOG_WARN("\tStreamTokens -> Got error while decoding: {}", item.getErrorMsg());
            callback(std::move(ctx), 0, 0.0, true);
        }
    }

    SPDLOG_INFO("Exiting StreamTokens");
    return 0;
}

std::unique_ptr<huggingface::tgi::backends::TensorRtLlmBackendImpl>
huggingface::tgi::backends::CreateTensorRtLlmBackend(rust::Str engineFolder, rust::Str executorWorker) {
    // Unconditionally call this to initialize and discover TRTLLM plugins
    InitializeBackend();

    const auto enginePath = std::string_view(engineFolder.begin(), engineFolder.end());
    const auto executorPath = std::string_view(executorWorker.begin(), executorWorker.end());
    return std::make_unique<TensorRtLlmBackendImpl>(std::move(enginePath), std::move(executorPath));
}
