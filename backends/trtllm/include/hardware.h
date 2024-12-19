//
// Created by mfuntowicz on 7/23/24.
//

#ifndef TGI_TRTLLM_BACKEND_HARDWARE_H
#define TGI_TRTLLM_BACKEND_HARDWARE_H

#include <cstdint>
#include <limits>
#include <fmt/base.h>
#include <spdlog/spdlog.h>
#include <nvml.h>

namespace huggingface::hardware::cuda {

#define AMPERE_SM_MAJOR 8
#define HOPPER_SM_MAJOR 8

    /**
     * Store information about the version of the CUDA Compute Capabilities detected on the device
     */
    struct CudaComputeCapabilities {
        int32_t major;
        int32_t minor;

        [[nodiscard]] constexpr bool isPostAmpere() const { return major >= AMPERE_SM_MAJOR; }

        [[nodiscard]] constexpr bool isPostHopper() const { return major >= HOPPER_SM_MAJOR; }
    };

    CudaComputeCapabilities GetCudaComputeCapabilities() {
        // Get the compute capabilities of the current hardware
        nvmlDevice_t device;
        CudaComputeCapabilities capabilities{0, 0};
        if (nvmlDeviceGetHandleByIndex_v2(0, &device) == NVML_SUCCESS) {
            SPDLOG_DEBUG("Successfully acquired nvmlDevice_t = 0");
            if (nvmlDeviceGetCudaComputeCapability(device, &capabilities.major, &capabilities.minor) == NVML_SUCCESS) {
                SPDLOG_INFO("Detected sm_{:d}{:d} compute capabilities", capabilities.major, capabilities.minor);
            }
        }

        return capabilities;
    }

    /**
     * Return the number of GPU detected. If no GPU is detected, return size_t::max()
     * @return
     */
    std::optional<size_t> GetNumDevices() {
        uint32_t numGpus = 0;
        if (nvmlDeviceGetCount_v2(&numGpus) == NVML_SUCCESS) {
            return std::optional(numGpus);
        } else {
            return std::nullopt;
        }
    }
}

#endif //TGI_TRTLLM_BACKEND_HARDWARE_H
