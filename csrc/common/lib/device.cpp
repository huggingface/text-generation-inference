//
// Created by morgan on 27/09/24.
//
#include <fmt/format.h>
#include <spdlog/spdlog.h>
#include <nvml.h>
#include "device.hpp"

std::optional<nvmlDevice_t> huggingface::tgi::GetDeviceByIndex(device_index_t index)
{
    ENSURE_NVML_INIT();

    nvmlDevice_t device;
    if(nvmlDeviceGetHandleByIndex_v2(index, &device) == NVML_SUCCESS)
        return std::optional{ device };

    return std::nullopt;
}

bool huggingface::tgi::IsP2PAvailable(device_index_t from, device_index_t to)
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

bool huggingface::tgi::IsP2PComplete()
{
    return false;
}
