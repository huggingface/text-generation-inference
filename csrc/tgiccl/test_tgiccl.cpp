//
// Created by morgan on 26/09/24.
//

#include "tgiccl.hpp"

int main() {
    auto a = huggingface::tgi::tgiccl::IsNvLinkAvailable(0, 1);
    auto b = huggingface::tgi::tgiccl::IsNvLinkAvailable(0, 2);
    auto d = huggingface::tgi::tgiccl::IsNvLinkAvailable(0, 3);
}