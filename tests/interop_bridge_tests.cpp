#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

#include "ave/interop_bridge.hpp"

namespace {

void check(const bool condition, const char* message) {
    if (condition) {
        return;
    }
    std::cerr << "interop_bridge_tests failed: " << message << '\n';
    std::abort();
}

void testImportMemoryRejectsInvalidHandle() {
    ave::InteropBridge bridge;
    void* hipPtr = nullptr;
    std::string error;
    ave::VulkanMemoryExport invalidExport;
    invalidExport.fd = -1;
    invalidExport.sizeBytes = 4096u;

    check(!bridge.importMemory(invalidExport, &hipPtr, error),
          "invalid Vulkan memory exports must be rejected");
    check(hipPtr == nullptr,
          "invalid Vulkan memory imports must not populate a HIP pointer");
    check(!error.empty(),
          "invalid Vulkan memory imports must explain the failure");
}

void testImportSemaphoreRejectsInvalidHandle() {
    ave::InteropBridge bridge;
    std::uint64_t hipSemaphore = 0u;
    std::string error;
    ave::VulkanSemaphoreExport invalidExport;
    invalidExport.fd = -1;

    check(!bridge.importSemaphore(invalidExport, &hipSemaphore, error),
          "invalid Vulkan semaphore exports must be rejected");
    check(hipSemaphore == 0u,
          "invalid semaphore imports must not populate a HIP semaphore handle");
    check(!error.empty(),
          "invalid semaphore imports must explain the failure");
}

}  // namespace

int main() {
    testImportMemoryRejectsInvalidHandle();
    testImportSemaphoreRejectsInvalidHandle();
    return 0;
}
