#pragma once

// ─────────────────────────────────────────────────────────────────
// interop_bridge.hpp — Vulkan ↔ HIP external memory/semaphore bridge
//
// Gold-standard-compliant interface for GPU memory sharing and
// synchronisation between Vulkan (preprocessing/postprocessing) and
// HIP/MiGraphX (inference), without implicit copies.
//
// ── Why this exists ──────────────────────────────────────────────
// The gold standard requires explicit, auditable GPU memory sharing.
// Specifically:
//   1. Vulkan preprocesses a frame into a VkBuffer (tensor layout).
//   2. That VkBuffer's VkDeviceMemory is exported as a POSIX FD via
//      VK_KHR_external_memory_fd (OPAQUE_FD handle type preferred for
//      same-process use; matches HIP documentation example).
//   3. HIP imports the FD via hipImportExternalMemory +
//      hipExternalMemoryGetMappedBuffer and passes the device pointer
//      directly as a MiGraphX program argument.
//   4. Semaphore handshake (VK_KHR_external_semaphore_fd +
//      hipImportExternalSemaphore) orders Vulkan writes before HIP
//      reads, and HIP writes before Vulkan postprocess reads.
//
// HIP explicitly documents that shared memory MUST be synchronised
// between APIs (queue syncs or semaphores).
//
// ── Semaphore states ─────────────────────────────────────────────
//   Vulkan signals BufferReady  → HIP waits; runs eval; signals InferenceDone
//   HIP   signals InferenceDone → Vulkan waits; runs postprocessing
//
// ── Handle type note ─────────────────────────────────────────────
// Prefer OPAQUE_FD for same-process HIP↔Vulkan interop.
// dma_buf (VK_EXT_external_memory_dma_buf) for cross-process or
// display compositor integration only.
//
// ── Extension requirements ───────────────────────────────────────
//   Vulkan: VK_KHR_external_memory_fd, VK_KHR_external_semaphore_fd,
//           VK_KHR_synchronization2 (or Vulkan 1.3)
//   HIP:    hipImportExternalMemory, hipExternalMemoryGetMappedBuffer,
//           hipImportExternalSemaphore (ROCm ≥ 5.0)
//
// ── Implementation status ────────────────────────────────────────
// This header defines the gold-standard API contract.
// VulkanRuntime initialises Vulkan context to correctly interact with
// hipImportExternal* implementations in interop_bridge.cpp.
// ─────────────────────────────────────────────────────────────────

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace ave {

// ─────────────────────────────────────────────────────────────────
// Handle types (opaque, no Vulkan headers required here)
// ─────────────────────────────────────────────────────────────────

// Represents a Vulkan VkDeviceMemory exported as a POSIX FD.
// Created by vkGetMemoryFdKHR(VkMemoryGetFdInfoKHR{handleType=OPAQUE_FD}).
struct VulkanMemoryExport {
    int         fd         = -1;    // POSIX file descriptor (-1 = invalid)
    std::size_t sizeBytes  = 0;     // Allocation size in bytes
    bool        isOpaqueFd = true;  // false → dma_buf handle type
};

// Represents a Vulkan VkSemaphore exported as a POSIX FD.
// Created by vkGetSemaphoreFdKHR(VkSemaphoreGetFdInfoKHR{handleType=OPAQUE_FD}).
struct VulkanSemaphoreExport {
    int fd = -1;  // POSIX file descriptor (-1 = invalid)
};

// ─────────────────────────────────────────────────────────────────
// Semaphore handshake states
// ─────────────────────────────────────────────────────────────────
// Must never transition out-of-order:
//   Idle → BufferReady → InferenceDone → Idle
//
// Invariant: no HIP kernel may read from the shared tensor buffer
// unless the bridge is in BufferReady state.  No Vulkan postprocess
// compute may read from the result buffer unless in InferenceDone.
enum class InteropSyncState {
    Idle,           // No frame in flight
    BufferReady,    // Vulkan finished writing; HIP may begin inference
    InferenceDone,  // HIP finished eval; Vulkan may begin postprocessing
};

// ─────────────────────────────────────────────────────────────────
// InteropBridge
// ─────────────────────────────────────────────────────────────────
class InteropBridge {
  public:
    InteropBridge();
    ~InteropBridge();

    // ── Availability probe ───────────────────────────────────────
    // Returns true only when all required Vulkan extensions are
    // supported on the active VkDevice AND HIP external-resource
    // APIs are available in the linked ROCm runtime.
    // Must be called after VulkanRuntime initialisation.
    bool isAvailable(std::string& reason) const;

    // ── Per-frame-slot memory lifecycle ──────────────────────────

    // Import a Vulkan-exported memory FD into HIP.
    // On success, *hipPtrOut is the device pointer to the mapped buffer.
    // HIP holds a reference until releaseMemory() is called.
    //   Vulkan side: allocate VkBuffer bound to exportable VkDeviceMemory
    //                (VkExportMemoryAllocateInfo with OPAQUE_FD handleType).
    //   HIP side:    hipImportExternalMemory + hipExternalMemoryGetMappedBuffer.
    // Closes fd on both success and failure (transfer ownership).
    bool importMemory(const VulkanMemoryExport& vkExport,
                      void**                    hipPtrOut,
                      std::string&              error);

    // Release a previously imported HIP external memory mapping.
    // Must be called before the associated VkDeviceMemory is freed.
    bool releaseMemory(void* hipPtr, std::string& error);

    // ── Per-frame-slot semaphore lifecycle ───────────────────────

    // Import a Vulkan-exported semaphore FD into HIP.
    // On success, *hipSemOut is a HIP external semaphore handle.
    //   HIP side: hipImportExternalSemaphore.
    // Closes fd on both success and failure.
    bool importSemaphore(const VulkanSemaphoreExport& vkExport,
                         std::uint64_t*               hipSemOut,
                         std::string&                 error);

    // HIP signals this semaphore after inference is complete.
    // Vulkan must be waiting on the corresponding semaphore FD
    // in vkQueueSubmit2 (VkSemaphoreSubmitInfo::stageMask=COMPUTE_SHADER).
    bool signalSemaphore(std::uint64_t hipSem, std::string& error);

    // HIP waits on this semaphore before reading the tensor buffer.
    // Vulkan must have signalled it at the end of the preprocess
    // vkQueueSubmit2 with stageMask=COMPUTE_SHADER.
    bool waitSemaphore(std::uint64_t hipSem, std::string& error);

    // Release imported semaphore handle.
    bool releaseSemaphore(std::uint64_t hipSem, std::string& error);

    // ── Diagnostics ─────────────────────────────────────────────
    // Returns true if the interop path is in use (vs CPU copy fallback).
    bool isUsingGpuInterop() const;

    // Log the current interop configuration and extension status.
    void logConfig() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ave
