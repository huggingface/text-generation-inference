//
// Created by mfuntowicz on 9/25/24.
//

#ifndef TEXT_GENERATION_INFERENCE_TGICCL_H
#define TEXT_GENERATION_INFERENCE_TGICCL_H

#include <optional>

#include <nvml.h>

#include "TgiCclBackend.hpp"

constexpr auto CLL_BACKEND_NAME = "tgiccl";

namespace huggingface::tgi::tgiccl
{
    static std::once_flag NVML_INIT_FLAG;
#define  ENSURE_NVML_INIT() std::call_once(NVML_INIT_FLAG, nvmlInit_v2);

    inline std::optional<nvmlDevice_t> GetDeviceByIndex(const size_t index)
    {
        ENSURE_NVML_INIT();

        nvmlDevice_t device;
        if(nvmlDeviceGetHandleByIndex_v2(index, &device) == NVML_SUCCESS)
            return std::optional{ device };

        return std::nullopt;
    }

    inline bool IsNvLinkAvailable(const int from, const int to)
    {
        ENSURE_NVML_INIT();

        // Get devices
        const auto devFrom = GetDeviceByIndex(from);
        const auto devTo = GetDeviceByIndex(to);

        if(!devFrom.has_value())
            SPDLOG_ERROR(FMT_STRING("Failed to retrieve device at index {:d}"), from);

        if(!devTo.has_value())
            SPDLOG_ERROR(FMT_STRING("Failed to retrieve device at index {:d}"), to);

        // Query link between both
        nvmlGpuP2PStatus_t status;
        if(nvmlDeviceGetP2PStatus(devFrom.value(), devTo.value(), NVML_P2P_CAPS_INDEX_NVLINK, &status) != NVML_SUCCESS)
        {
            SPDLOG_ERROR(FMT_STRING("Failed to retrieve the p2p status for device {:d} <-> {:d}"), from, to);
            return false;
        }

        return status == NVML_P2P_STATUS_OK;
    }

}

#endif //TEXT_GENERATION_INFERENCE_TGICCL_H
