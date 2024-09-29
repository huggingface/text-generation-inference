//
// Created by morgan on 27/09/24.
//

#ifndef TGI_DEVICE_HPP
#define TGI_DEVICE_HPP
#include <cstdint>
#include <nvml.h>
#include <optional>
#include <mutex>

namespace huggingface::tgi {
    static std::once_flag NVML_INIT_FLAG;
#define  ENSURE_NVML_INIT() std::call_once(NVML_INIT_FLAG, nvmlInit_v2);

    using device_index_t = uint8_t;

    /**
     * Attempt to retrieve the referred GPU by its index on the system
     * @param device Device index
     * @return
     */
    std::optional<nvmlDevice_t> GetDeviceByIndex(device_index_t device);

    /**
     * Check whether all the GPUs have direct remote memory access to each other
     */
    bool IsP2PComplete();

    /**
     * Check if GPU "from" has remote memory access to GPU "to"
     * @param from Originating GPU memory
     * @param to Destination GPU memory
     * @return True if p2p is available, false otherwise
     */
    bool IsP2PAvailable(device_index_t from, device_index_t to);
}

#endif // TGI_DEVICE_HPP
