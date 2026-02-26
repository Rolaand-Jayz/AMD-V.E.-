// ─────────────────────────────────────────────────────────────────
// interop_bridge.cpp — Vulkan↔HIP external memory/semaphore bridge
//
// Implements the gold-standard API surface defined in interop_bridge.hpp.
// ─────────────────────────────────────────────────────────────────
#include "ave/interop_bridge.hpp"
#include "ave/vulkan_runtime.hpp"

#include <iostream>
#include <string>
#include <unordered_map>

#ifdef AVE_HAVE_HIP
#  include <hip/hip_runtime.h>
#endif

namespace ave {

struct InteropBridge::Impl {
    bool available = false;
    InteropSyncState state = InteropSyncState::Idle;

#ifdef AVE_HAVE_HIP
    std::unordered_map<void*, hipExternalMemory_t> mappedMemory;
    std::unordered_map<std::uint64_t, hipExternalSemaphore_t> mappedSemaphores;
#endif
};

InteropBridge::InteropBridge()  : impl_(std::make_unique<Impl>()) {
#ifdef AVE_HAVE_HIP
    impl_->available = true;
#endif
}
InteropBridge::~InteropBridge() = default;

// ── Availability probe ───────────────────────────────────────────
bool InteropBridge::isAvailable(std::string& reason) const {
#ifdef AVE_HAVE_HIP
    if (impl_->available) {
        return true;
    }
    reason = "InteropBridge failed to initialize.";
    return false;
#else
    reason = "HIP support not compiled in – Vulkan↔HIP interop is unavailable. "
             "GPU data will cross through CPU staging (degraded mode, logged).";
    return false;
#endif
}

// ── Memory lifecycle ─────────────────────────────────────────────
bool InteropBridge::importMemory(const VulkanMemoryExport& vkExport,
                                  void**                    hipPtrOut,
                                  std::string&              error) {
#ifdef AVE_HAVE_HIP
    if (!impl_->available) {
        error = "InteropBridge not available.";
        return false;
    }

    hipExternalMemoryHandleDesc desc{};
    desc.type = hipExternalMemoryHandleTypeOpaqueFd;
    desc.handle.fd = vkExport.fd;
    desc.size = vkExport.sizeBytes;

    hipExternalMemory_t extMem{};
    if (hipImportExternalMemory(&extMem, &desc) != hipSuccess) {
        error = "hipImportExternalMemory failed.";
        return false;
    }

    hipExternalMemoryBufferDesc bufDesc{};
    bufDesc.offset = 0;
    bufDesc.size = vkExport.sizeBytes;

    if (hipExternalMemoryGetMappedBuffer(hipPtrOut, extMem, &bufDesc) != hipSuccess) {
        error = "hipExternalMemoryGetMappedBuffer failed.";
        return false;
    }

    impl_->mappedMemory[*hipPtrOut] = extMem;
    return true;
#else
    error = "InteropBridge::importMemory: HIP not integrated.";
    return false;
#endif
}

bool InteropBridge::releaseMemory(void* hipPtr, std::string& error) {
#ifdef AVE_HAVE_HIP
    auto it = impl_->mappedMemory.find(hipPtr);
    if (it == impl_->mappedMemory.end()) {
        error = "Memory pointer not found in mapped memory.";
        return false;
    }

    hipFree(hipPtr);
    hipDestroyExternalMemory(it->second);
    impl_->mappedMemory.erase(it);
    return true;
#else
    error = "InteropBridge::releaseMemory: HIP not integrated.";
    return false;
#endif
}

// ── Semaphore lifecycle ──────────────────────────────────────────
bool InteropBridge::importSemaphore(const VulkanSemaphoreExport& vkExport,
                                     std::uint64_t*               hipSemOut,
                                     std::string&                 error) {
#ifdef AVE_HAVE_HIP
    if (!impl_->available) {
        error = "InteropBridge not available.";
        return false;
    }

    hipExternalSemaphoreHandleDesc desc{};
    desc.type = hipExternalSemaphoreHandleTypeOpaqueFd;
    desc.handle.fd = vkExport.fd;

    hipExternalSemaphore_t sem{};
    if (hipImportExternalSemaphore(&sem, &desc) != hipSuccess) {
        error = "hipImportExternalSemaphore failed.";
        return false;
    }

    *hipSemOut = reinterpret_cast<std::uint64_t>(sem);
    impl_->mappedSemaphores[*hipSemOut] = sem;
    return true;
#else
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
    if (hipSignalExternalSemaphoresAsync(&it->second, &params, 1, nullptr) != hipSuccess) {
        error = "hipSignalExternalSemaphoresAsync failed.";
        return false;
    }
    
    impl_->state = InteropSyncState::InferenceDone;
    return true;
#else
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
    if (hipWaitExternalSemaphoresAsync(&it->second, &params, 1, nullptr) != hipSuccess) {
        error = "hipWaitExternalSemaphoresAsync failed.";
        return false;
    }

    impl_->state = InteropSyncState::BufferReady;
    return true;
#else
    error = "InteropBridge::waitSemaphore: HIP not integrated.";
    return false;
#endif
}

bool InteropBridge::releaseSemaphore(std::uint64_t /*hipSem*/, std::string& error) {
    error = "InteropBridge::releaseSemaphore: VulkanRuntime not integrated.";
    return false;
}

// ── Diagnostics ──────────────────────────────────────────────────
bool InteropBridge::isUsingGpuInterop() const {
    return impl_->available;
}

void InteropBridge::logConfig() const {
    std::cout << "[interop-bridge] status="
              << (impl_->available ? "gpu-interop" : "unavailable (CPU-staging degraded mode)")
              << "\n"
              << "[interop-bridge] required: VK_KHR_external_memory_fd, "
                 "VK_KHR_external_semaphore_fd, VK_KHR_synchronization2"
              << "\n"
              << "[interop-bridge] handle-type: OPAQUE_FD (same-process preferred)"
              << std::endl;
}

}  // namespace ave
