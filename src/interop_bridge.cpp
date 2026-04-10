// ─────────────────────────────────────────────────────────────────
// interop_bridge.cpp — Vulkan↔HIP external memory/semaphore bridge
//
// Implements the gold-standard API surface defined in interop_bridge.hpp.
// ─────────────────────────────────────────────────────────────────
#include "ave/interop_bridge.hpp"

#include "ave/vulkan_runtime.hpp"

#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>

#if defined(__linux__)
#  include <unistd.h>
#endif

#ifdef AVE_HAVE_HIP
#  include <hip/hip_runtime.h>
#endif

namespace ave {
namespace {

#ifdef AVE_HAVE_HIP
std::string hipErrorMessage(const hipError_t status) {
    const char* text = hipGetErrorString(status);
    if (text == nullptr || *text == '\0') {
        return "HIP error " + std::to_string(static_cast<int>(status));
    }
    return text;
}
#endif

void closeFdIfValid(const int fd) {
#if defined(__linux__)
    if (fd >= 0) {
        (void)::close(fd);
    }
#else
    (void)fd;
#endif
}

}  // namespace

struct InteropBridge::Impl {
    mutable bool availabilityProbed = false;
    mutable bool available = false;
    mutable std::string availabilityReason;
    InteropSyncState state = InteropSyncState::Idle;

#ifdef AVE_HAVE_HIP
    std::unordered_map<void*, hipExternalMemory_t> mappedMemory;
    std::unordered_map<std::uint64_t, hipExternalSemaphore_t> mappedSemaphores;
#endif
};

InteropBridge::InteropBridge() : impl_(std::make_unique<Impl>()) {}

InteropBridge::~InteropBridge() {
#ifdef AVE_HAVE_HIP
    for (auto it = impl_->mappedSemaphores.begin(); it != impl_->mappedSemaphores.end(); ++it) {
        (void)hipDestroyExternalSemaphore(it->second);
    }
    impl_->mappedSemaphores.clear();

    for (auto it = impl_->mappedMemory.begin(); it != impl_->mappedMemory.end(); ++it) {
        (void)hipFree(it->first);
        (void)hipDestroyExternalMemory(it->second);
    }
    impl_->mappedMemory.clear();
#endif
}

bool InteropBridge::isAvailable(std::string& reason) const {
    if (impl_->availabilityProbed) {
        reason = impl_->availabilityReason;
        return impl_->available;
    }

    impl_->availabilityProbed = true;

#if !defined(AVE_HAVE_HIP)
    impl_->availabilityReason =
        "HIP support not compiled in – Vulkan↔HIP interop is unavailable. "
        "GPU data will cross through host staging.";
    reason = impl_->availabilityReason;
    impl_->available = false;
    return false;
#elif !defined(AVE_HAVE_VULKAN)
    impl_->availabilityReason =
        "Vulkan support not compiled in – Vulkan↔HIP interop is unavailable.";
    reason = impl_->availabilityReason;
    impl_->available = false;
    return false;
#else
    int hipDeviceCount = 0;
    const hipError_t deviceStatus = hipGetDeviceCount(&hipDeviceCount);
    if (deviceStatus != hipSuccess || hipDeviceCount <= 0) {
        impl_->availabilityReason =
            "HIP external-resource path unavailable: no AMD HIP device is visible ("
            + hipErrorMessage(deviceStatus) + ").";
        reason = impl_->availabilityReason;
        impl_->available = false;
        return false;
    }

    VulkanRuntime runtime;
    std::string runtimeError;
    if (!runtime.initialize(runtimeError)) {
        impl_->availabilityReason =
            "Vulkan external-memory path unavailable: " + runtimeError;
        reason = impl_->availabilityReason;
        impl_->available = false;
        return false;
    }

    const auto vkGetMemoryFd =
        reinterpret_cast<PFN_vkGetMemoryFdKHR>(
            vkGetDeviceProcAddr(runtime.getDevice(), "vkGetMemoryFdKHR"));
    const auto vkGetSemaphoreFd =
        reinterpret_cast<PFN_vkGetSemaphoreFdKHR>(
            vkGetDeviceProcAddr(runtime.getDevice(), "vkGetSemaphoreFdKHR"));
    if (vkGetMemoryFd == nullptr || vkGetSemaphoreFd == nullptr) {
        impl_->availabilityReason =
            "Vulkan external-memory/semaphore FD export functions are unavailable on "
            "the active Vulkan device.";
        reason = impl_->availabilityReason;
        impl_->available = false;
        return false;
    }

    impl_->available = true;
    impl_->availabilityReason =
        "Vulkan external memory and HIP external-resource APIs are ready.";
    reason = impl_->availabilityReason;
    return true;
#endif
}

bool InteropBridge::importMemory(const VulkanMemoryExport& vkExport,
                                 void** hipPtrOut,
                                 std::string& error) {
#ifdef AVE_HAVE_HIP
    if (hipPtrOut == nullptr) {
        error = "InteropBridge::importMemory: output pointer is null.";
        closeFdIfValid(vkExport.fd);
        return false;
    }
    if (vkExport.fd < 0 || vkExport.sizeBytes == 0u) {
        error = "InteropBridge::importMemory: invalid exported Vulkan memory handle.";
        closeFdIfValid(vkExport.fd);
        return false;
    }

    if (!isAvailable(error)) {
        closeFdIfValid(vkExport.fd);
        return false;
    }

    hipExternalMemoryHandleDesc desc{};
    desc.type = vkExport.isOpaqueFd
        ? hipExternalMemoryHandleTypeOpaqueFd
        : hipExternalMemoryHandleTypeOpaqueFd;
    desc.handle.fd = vkExport.fd;
    desc.size = vkExport.sizeBytes;

    hipExternalMemory_t extMem{};
    const hipError_t importStatus = hipImportExternalMemory(&extMem, &desc);
    closeFdIfValid(vkExport.fd);
    if (importStatus != hipSuccess) {
        error = "hipImportExternalMemory failed: " + hipErrorMessage(importStatus);
        return false;
    }

    hipExternalMemoryBufferDesc bufDesc{};
    bufDesc.offset = 0;
    bufDesc.size = vkExport.sizeBytes;

    void* mappedPtr = nullptr;
    const hipError_t mapStatus =
        hipExternalMemoryGetMappedBuffer(&mappedPtr, extMem, &bufDesc);
    if (mapStatus != hipSuccess || mappedPtr == nullptr) {
        (void)hipDestroyExternalMemory(extMem);
        error = "hipExternalMemoryGetMappedBuffer failed: "
              + hipErrorMessage(mapStatus);
        return false;
    }

    *hipPtrOut = mappedPtr;
    impl_->mappedMemory[mappedPtr] = extMem;
    return true;
#else
    (void)vkExport;
    (void)hipPtrOut;
    error = "InteropBridge::importMemory: HIP not integrated.";
    return false;
#endif
}

bool InteropBridge::releaseMemory(void* hipPtr, std::string& error) {
#ifdef AVE_HAVE_HIP
    auto it = impl_->mappedMemory.find(hipPtr);
    if (it == impl_->mappedMemory.end()) {
        error = "Memory pointer not found in imported external-memory map.";
        return false;
    }

    std::ostringstream errors;
    const hipError_t freeStatus = hipFree(hipPtr);
    if (freeStatus != hipSuccess) {
        errors << "hipFree failed: " << hipErrorMessage(freeStatus);
    }
    const hipError_t destroyStatus = hipDestroyExternalMemory(it->second);
    if (destroyStatus != hipSuccess) {
        if (!errors.str().empty()) {
            errors << " | ";
        }
        errors << "hipDestroyExternalMemory failed: "
               << hipErrorMessage(destroyStatus);
    }

    impl_->mappedMemory.erase(it);
    if (!errors.str().empty()) {
        error = errors.str();
        return false;
    }
    return true;
#else
    (void)hipPtr;
    error = "InteropBridge::releaseMemory: HIP not integrated.";
    return false;
#endif
}

bool InteropBridge::importSemaphore(const VulkanSemaphoreExport& vkExport,
                                    std::uint64_t* hipSemOut,
                                    std::string& error) {
#ifdef AVE_HAVE_HIP
    if (hipSemOut == nullptr) {
        error = "InteropBridge::importSemaphore: output semaphore handle is null.";
        closeFdIfValid(vkExport.fd);
        return false;
    }
    if (vkExport.fd < 0) {
        error = "InteropBridge::importSemaphore: invalid exported semaphore FD.";
        closeFdIfValid(vkExport.fd);
        return false;
    }
    if (!isAvailable(error)) {
        closeFdIfValid(vkExport.fd);
        return false;
    }

    hipExternalSemaphoreHandleDesc desc{};
    desc.type = hipExternalSemaphoreHandleTypeOpaqueFd;
    desc.handle.fd = vkExport.fd;

    hipExternalSemaphore_t semaphore{};
    const hipError_t importStatus = hipImportExternalSemaphore(&semaphore, &desc);
    closeFdIfValid(vkExport.fd);
    if (importStatus != hipSuccess) {
        error = "hipImportExternalSemaphore failed: " + hipErrorMessage(importStatus);
        return false;
    }

    *hipSemOut = reinterpret_cast<std::uint64_t>(semaphore);
    impl_->mappedSemaphores[*hipSemOut] = semaphore;
    return true;
#else
    (void)vkExport;
    (void)hipSemOut;
    error = "InteropBridge::importSemaphore: HIP not integrated.";
    return false;
#endif
}

bool InteropBridge::signalSemaphore(std::uint64_t hipSem, std::string& error) {
#ifdef AVE_HAVE_HIP
    auto it = impl_->mappedSemaphores.find(hipSem);
    if (it == impl_->mappedSemaphores.end()) {
        error = "Semaphore not found.";
        return false;
    }

    hipExternalSemaphoreSignalParams params{};
    const hipError_t signalStatus =
        hipSignalExternalSemaphoresAsync(&it->second, &params, 1, nullptr);
    if (signalStatus != hipSuccess) {
        error = "hipSignalExternalSemaphoresAsync failed: "
              + hipErrorMessage(signalStatus);
        return false;
    }

    impl_->state = InteropSyncState::InferenceDone;
    return true;
#else
    (void)hipSem;
    error = "InteropBridge::signalSemaphore: HIP not integrated.";
    return false;
#endif
}

bool InteropBridge::waitSemaphore(std::uint64_t hipSem, std::string& error) {
#ifdef AVE_HAVE_HIP
    auto it = impl_->mappedSemaphores.find(hipSem);
    if (it == impl_->mappedSemaphores.end()) {
        error = "Semaphore not found.";
        return false;
    }

    hipExternalSemaphoreWaitParams params{};
    const hipError_t waitStatus =
        hipWaitExternalSemaphoresAsync(&it->second, &params, 1, nullptr);
    if (waitStatus != hipSuccess) {
        error = "hipWaitExternalSemaphoresAsync failed: "
              + hipErrorMessage(waitStatus);
        return false;
    }

    impl_->state = InteropSyncState::BufferReady;
    return true;
#else
    (void)hipSem;
    error = "InteropBridge::waitSemaphore: HIP not integrated.";
    return false;
#endif
}

bool InteropBridge::releaseSemaphore(std::uint64_t hipSem, std::string& error) {
#ifdef AVE_HAVE_HIP
    auto it = impl_->mappedSemaphores.find(hipSem);
    if (it == impl_->mappedSemaphores.end()) {
        error = "Semaphore not found.";
        return false;
    }

    const hipError_t destroyStatus = hipDestroyExternalSemaphore(it->second);
    impl_->mappedSemaphores.erase(it);
    if (destroyStatus != hipSuccess) {
        error = "hipDestroyExternalSemaphore failed: "
              + hipErrorMessage(destroyStatus);
        return false;
    }
    return true;
#else
    (void)hipSem;
    error = "InteropBridge::releaseSemaphore: HIP not integrated.";
    return false;
#endif
}

bool InteropBridge::isUsingGpuInterop() const {
    std::string reason;
    return isAvailable(reason);
}

void InteropBridge::logConfig() const {
    std::string reason;
    const bool available = isAvailable(reason);
    std::cout << "[interop-bridge] status="
              << (available ? "gpu-interop-ready"
                            : "host-staging fallback")
              << "\n"
              << "[interop-bridge] detail: " << reason << "\n"
              << "[interop-bridge] required: VK_KHR_external_memory_fd, "
                 "VK_KHR_external_semaphore_fd, VK_KHR_synchronization2"
              << "\n"
              << "[interop-bridge] handle-type: OPAQUE_FD (same-process preferred)"
              << std::endl;
}

}  // namespace ave
