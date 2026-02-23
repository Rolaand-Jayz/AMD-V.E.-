// ─────────────────────────────────────────────────────────────────
// interop_bridge.cpp — Vulkan↔HIP external memory/semaphore bridge
//
// Current state: documented stub.
// The gold-standard API surface is defined in interop_bridge.hpp.
// This implementation returns "unavailable" because VulkanRuntime
// (VkDevice/VkQueue initialisation) has not yet been integrated.
//
// To activate:
//   1. Implement VulkanRuntime with VkInstance, VkDevice, VkQueue
//      and memory allocator (VMA recommended, with external handle
//      types configured for tensor-buffer allocations).
//   2. Replace the stub bodies below with:
//      – hipImportExternalMemory + hipExternalMemoryGetMappedBuffer
//        (using the FD from VkMemoryGetFdInfoKHR with OPAQUE_FD).
//      – hipImportExternalSemaphore
//        (using the FD from VkSemaphoreGetFdInfoKHR with OPAQUE_FD).
//   3. Enforce the semaphore handshake states (InteropSyncState).
//
// References (gold standard):
//   HIP External Resource Interoperability documentation
//   VK_KHR_external_memory_fd, VK_KHR_external_semaphore_fd
//   VK_KHR_synchronization2 (Vulkan 1.3 core)
// ─────────────────────────────────────────────────────────────────
#include "ave/interop_bridge.hpp"

#include <iostream>
#include <string>

#ifdef AVE_HAVE_HIP
#  include <hip/hip_runtime.h>
#endif

namespace ave {

struct InteropBridge::Impl {
    bool available = false;  // Set true once VulkanRuntime is integrated
};

InteropBridge::InteropBridge()  : impl_(std::make_unique<Impl>()) {}
InteropBridge::~InteropBridge() = default;

// ── Availability probe ───────────────────────────────────────────
bool InteropBridge::isAvailable(std::string& reason) const {
    // TODO: Query VkPhysicalDevice for VK_KHR_external_memory_fd and
    // VK_KHR_external_semaphore_fd extension support.  Check HIP external
    // resource API availability.  Requires VulkanRuntime to be initialised.
    reason = "VulkanRuntime not yet integrated – Vulkan↔HIP interop is unavailable. "
             "GPU data will cross through CPU staging (degraded mode, logged).";
    return false;
}

// ── Memory lifecycle ─────────────────────────────────────────────
bool InteropBridge::importMemory(const VulkanMemoryExport& /*vkExport*/,
                                  void**                    /*hipPtrOut*/,
                                  std::string&              error) {
    error = "InteropBridge::importMemory: VulkanRuntime not integrated.";
    return false;
    // Gold standard implementation note:
    //   hipExternalMemoryHandleDesc desc{};
    //   desc.type = hipExternalMemoryHandleTypeOpaqueFd;  // OPAQUE_FD path
    //   desc.handle.fd      = vkExport.fd;  // ownership transferred
    //   desc.size           = vkExport.sizeBytes;
    //   hipExternalMemory_t extMem{};
    //   if (hipImportExternalMemory(&extMem, &desc) != hipSuccess) { ... }
    //   hipExternalMemoryBufferDesc bufDesc{};
    //   bufDesc.offset = 0;
    //   bufDesc.size   = vkExport.sizeBytes;
    //   if (hipExternalMemoryGetMappedBuffer(hipPtrOut, extMem, &bufDesc) != hipSuccess) { ... }
}

bool InteropBridge::releaseMemory(void* /*hipPtr*/, std::string& error) {
    error = "InteropBridge::releaseMemory: VulkanRuntime not integrated.";
    return false;
}

// ── Semaphore lifecycle ──────────────────────────────────────────
bool InteropBridge::importSemaphore(const VulkanSemaphoreExport& /*vkExport*/,
                                     std::uint64_t*               /*hipSemOut*/,
                                     std::string&                 error) {
    error = "InteropBridge::importSemaphore: VulkanRuntime not integrated.";
    return false;
    // Gold standard implementation note:
    //   hipExternalSemaphoreHandleDesc desc{};
    //   desc.type = hipExternalSemaphoreHandleTypeOpaqueFd;
    //   desc.handle.fd = vkExport.fd;  // ownership transferred
    //   hipExternalSemaphore_t sem{};
    //   if (hipImportExternalSemaphore(&sem, &desc) != hipSuccess) { ... }
    //   *hipSemOut = reinterpret_cast<std::uint64_t>(sem);
}

bool InteropBridge::signalSemaphore(std::uint64_t /*hipSem*/, std::string& error) {
    error = "InteropBridge::signalSemaphore: VulkanRuntime not integrated.";
    return false;
}

bool InteropBridge::waitSemaphore(std::uint64_t /*hipSem*/, std::string& error) {
    error = "InteropBridge::waitSemaphore: VulkanRuntime not integrated.";
    return false;
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
