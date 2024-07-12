//
// Created by mfuntowicz on 6/30/24.
//
#pragma once

#include <cmath>
#include <filesystem>
#include <vector>

//#include "rust/cxx.h"
//#include "../include/ffi.h"
#include "backends/trtllm/include/ffi.h"


huggingface::tgi::backends::TensorRtLlmBackendImpl::TensorRtLlmBackendImpl(
        const std::string_view &engineFolder,
        const std::string_view &executorWorker
) : TensorRtLlmBackend(engineFolder, executorWorker) {}


bool huggingface::tgi::backends::TensorRtLlmBackendImpl::IsReady() const {
    return TensorRtLlmBackend::IsReady();
}

uint64_t huggingface::tgi::backends::TensorRtLlmBackendImpl::Submit(
        rust::Slice<const uint32_t> tokens,
        int32_t maxNewTokens, int32_t topK, float_t topP,
        float_t temperature, uint64_t seed) {

    // This will copy all the items from the initial slice
    std::vector<int32_t> tokens_(tokens.size());
    tokens_.assign(tokens.begin(), tokens.end());

    return TensorRtLlmBackend::Submit(std::move(tokens_), maxNewTokens, topK, topP, temperature, seed);
}

uint32_t huggingface::tgi::backends::TensorRtLlmBackendImpl::Stream(
        rust::Box<huggingface::tgi::backends::GenerationContext> ctx,
        uint64_t requestId,
        rust::Fn<void(rust::Box<huggingface::tgi::backends::GenerationContext>, uint32_t, uint32_t, bool)> handler) {
    bool isDone = false;
    uint32_t numGeneratedTokens = 0;

    do {
        const auto responses = Poll(requestId);
        for (const auto &response: responses) {
            if (response.hasError()) {
                isDone = true;
                // TODO : bubble up the error to rust
            } else {
                const auto generation = response.getResult();
                const auto token = generation.outputTokenIds[0][0];
                isDone = generation.isFinal;

                // Propagate through the handler
                handler(std::move(ctx), token, numGeneratedTokens, isDone);
            }
        }
    } while (!isDone);

    return numGeneratedTokens;
}

std::unique_ptr<huggingface::tgi::backends::TensorRtLlmBackendImpl>
huggingface::tgi::backends::CreateTensorRtLlmBackend(rust::Str engineFolder, rust::Str executorWorker) {
    // Unconditionally call this to initialize and discover TRTLLM plugins
    InitializeBackend();

    const auto enginePath = std::string_view(engineFolder.begin(), engineFolder.end());
    const auto executorPath = std::string_view(executorWorker.begin(), executorWorker.end());
    return std::make_unique<TensorRtLlmBackendImpl>(std::move(enginePath), std::move(executorPath));
}
