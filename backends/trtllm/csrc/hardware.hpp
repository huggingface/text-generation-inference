#ifndef TGI_HARDWARE_CUDA
#define TGI_HARDWARE_CUDA
#include <cstdint>
#include <optional>

#include <nvml.h>

namespace huggingface::tgi::hardware::cuda {
    static constexpr auto VOLTA = std::make_tuple(7u, 0u);
    static constexpr auto TURING = std::make_tuple(7u, 5u);
    static constexpr auto AMPERE = std::make_tuple(8u, 0u);
    static constexpr auto HOPPER = std::make_tuple(9u, 0u);
    static constexpr auto ADA_LOVELACE = std::make_tuple(8u, 9u);

    /**
     * Get the number of GPUs on the local machine
     * @return std::nullopt if no device is available, otherwise >= 1
     */
    inline std::optional<size_t> get_device_count() {
        uint32_t numGpus = 0;
        if (nvmlDeviceGetCount_v2(&numGpus) == NVML_SUCCESS) {
            return numGpus;
        }
        return std::nullopt;
    }

    /**
     * Store information about the version of the CUDA Compute Capabilities detected on the device
     */
    struct compute_capabilities_t {
        int32_t major;
        int32_t minor;

        compute_capabilities_t(): compute_capabilities_t(0) {}
        explicit compute_capabilities_t(size_t device_idx): major(-1), minor(-1) {
            nvmlDevice_t device;
            if (nvmlDeviceGetHandleByIndex_v2(device_idx, &device) == NVML_SUCCESS) {
               nvmlDeviceGetCudaComputeCapability(device, &major, &minor);
            }
        };
        compute_capabilities_t(int32_t major, int32_t minor): major(major), minor(minor) {}

        /**
         * Evaluate if the underlying capabilities is at least greater or equals to the provided 2-tuple (major, minor)
         * @param sm Architecture version (major, minor)
         * @return True if greater or equals to the underlying compute capabilities
         */
        [[nodiscard]] constexpr auto is_at_least(std::tuple<uint32_t, uint32_t> sm) const -> decltype(auto) { return std::tie(major, minor) >= sm; }

        /**
         * Check if the capabilities match at least Volta architecture (sm_70)
         * @return true if at least Volta (>= sm_70), false otherwise
         */
        [[nodiscard]] constexpr bool is_at_least_volta() const { return is_at_least(VOLTA); }

        /**
         * Check if the capabilities match at least Turing architecture (sm_75)
         * @return true if at least Turing (>= sm_75), false otherwise
         */
        [[nodiscard]] constexpr bool is_at_least_turing() const { return is_at_least(TURING); }

        /**
         * Check if the capabilities match at least Ampere architecture (sm_80)
         * @return true if at least Ampere (>= sm_80), false otherwise
         */
        [[nodiscard]] constexpr bool is_at_least_ampere() const { return is_at_least(AMPERE); }

        /**
         * Check if the capabilities match at least Ada Lovelace architecture (sm_89)
         * @return true if at least Ada Lovelace (>= sm_89), false otherwise
         */
        [[nodiscard]] constexpr bool is_at_least_ada_lovelace() const { return is_at_least(ADA_LOVELACE); }

        /**
         * Check if the capabilities match at least Hopper architecture (sm_90)
         * @return true if at least Hopper (>= sm_90), false otherwise
         */
        [[nodiscard]] constexpr bool is_at_least_hopper() const { return is_at_least(HOPPER); }
    };
}
#endif
