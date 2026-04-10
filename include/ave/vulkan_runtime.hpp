#pragma once

#include <memory>
#include <string>

extern "C" {
struct AVBufferRef;
}

#ifdef AVE_HAVE_VULKAN
#include <vulkan/vulkan.h>
#endif

namespace ave {

class VulkanRuntime {
public:
    VulkanRuntime();
    ~VulkanRuntime();

    bool initialize(std::string& error);
    bool initialize(const AVBufferRef* ffmpegHwDeviceContext, std::string& error);
    bool isAvailable(std::string& reason) const;

#ifdef AVE_HAVE_VULKAN
    VkInstance getInstance() const;
    VkPhysicalDevice getPhysicalDevice() const;
    VkDevice getDevice() const;
    VkQueue getComputeQueue() const;
    uint32_t getComputeQueueFamilyIndex() const;

    // Allocate a buffer that can be exported as an FD
    // Note: This must be synchronised properly using VK_KHR_synchronization2
    // and VkImageMemoryBarrier2 in the pipeline submission.
    bool allocateExternalBuffer(std::size_t sizeBytes,
                                VkBuffer& bufferOut,
                                VkDeviceMemory& memoryOut,
                                int& fdOut,
                                void** mappedPtrOut,
                                std::string& error);

    void freeExternalBuffer(VkBuffer buffer, VkDeviceMemory memory);
#endif

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ave
