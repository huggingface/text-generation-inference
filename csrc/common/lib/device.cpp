//
// Created by morgan on 27/09/24.
//

#include "device.hpp"

std::optional<nvmlDevice_t> huggingface::tgi::GetDeviceByIndex(device_index_t device)
{
    return std::nullopt;
}

bool huggingface::tgi::IsP2PComplete()
{
    return false;
}

bool huggingface::tgi::IsP2PAvailable(device_index_t from, device_index_t to)
{
    return false;
}
