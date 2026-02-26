#include "ave/vulkan_runtime.hpp"

#include <iostream>
#include <vector>

#ifdef AVE_HAVE_VULKAN
#include <vulkan/vulkan.h>
#endif

namespace ave {

struct VulkanRuntime::Impl {
    bool available = false;
#ifdef AVE_HAVE_VULKAN
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue computeQueue = VK_NULL_HANDLE;
    uint32_t computeQueueFamilyIndex = 0;
#endif
};

VulkanRuntime::VulkanRuntime() : impl_(std::make_unique<Impl>()) {}

VulkanRuntime::~VulkanRuntime() {
#ifdef AVE_HAVE_VULKAN
    if (impl_->device != VK_NULL_HANDLE) {
        vkDestroyDevice(impl_->device, nullptr);
    }
    if (impl_->instance != VK_NULL_HANDLE) {
        vkDestroyInstance(impl_->instance, nullptr);
    }
#endif
}

bool VulkanRuntime::isAvailable(std::string& reason) const {
#ifdef AVE_HAVE_VULKAN
    if (impl_->available) {
        return true;
    }
    reason = "VulkanRuntime not initialized.";
    return false;
#else
    reason = "Vulkan support not compiled in.";
    return false;
#endif
}

bool VulkanRuntime::initialize(std::string& error) {
#ifdef AVE_HAVE_VULKAN
    if (impl_->available) {
        return true;
    }

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "AMD Video Enhancer";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "AVE";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_3;

    std::vector<const char*> extensions = {
        VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME
    };

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    if (vkCreateInstance(&createInfo, nullptr, &impl_->instance) != VK_SUCCESS) {
        error = "Failed to create Vulkan instance.";
        return false;
    }

    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(impl_->instance, &deviceCount, nullptr);
    if (deviceCount == 0) {
        error = "Failed to find GPUs with Vulkan support.";
        return false;
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(impl_->instance, &deviceCount, devices.data());

    // Pick the first device for now
    impl_->physicalDevice = devices[0];

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(impl_->physicalDevice, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(impl_->physicalDevice, &queueFamilyCount, queueFamilies.data());

    bool foundCompute = false;
    for (uint32_t i = 0; i < queueFamilyCount; i++) {
        if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            impl_->computeQueueFamilyIndex = i;
            foundCompute = true;
            break;
        }
    }

    if (!foundCompute) {
        error = "Failed to find a compute queue family.";
        return false;
    }

    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueCreateInfo{};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = impl_->computeQueueFamilyIndex;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    std::vector<const char*> deviceExtensions = {
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
        VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME
    };

    VkPhysicalDeviceSynchronization2FeaturesKHR sync2Features{};
    sync2Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR;
    sync2Features.synchronization2 = VK_TRUE;

    VkDeviceCreateInfo deviceCreateInfo{};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
    deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.data();
    deviceCreateInfo.pNext = &sync2Features;

    if (vkCreateDevice(impl_->physicalDevice, &deviceCreateInfo, nullptr, &impl_->device) != VK_SUCCESS) {
        error = "Failed to create logical device.";
        return false;
    }

    vkGetDeviceQueue(impl_->device, impl_->computeQueueFamilyIndex, 0, &impl_->computeQueue);

    impl_->available = true;
    return true;
#else
    error = "Vulkan support not compiled in.";
    return false;
#endif
}

#ifdef AVE_HAVE_VULKAN
VkInstance VulkanRuntime::getInstance() const { return impl_->instance; }
VkPhysicalDevice VulkanRuntime::getPhysicalDevice() const { return impl_->physicalDevice; }
VkDevice VulkanRuntime::getDevice() const { return impl_->device; }
VkQueue VulkanRuntime::getComputeQueue() const { return impl_->computeQueue; }
uint32_t VulkanRuntime::getComputeQueueFamilyIndex() const { return impl_->computeQueueFamilyIndex; }

bool VulkanRuntime::allocateExternalBuffer(std::size_t sizeBytes,
                                           VkBuffer& bufferOut,
                                           VkDeviceMemory& memoryOut,
                                           int& fdOut,
                                           void** mappedPtrOut,
                                           std::string& error) {
    if (!impl_->available) {
        error = "VulkanRuntime not initialized.";
        return false;
    }

    VkExternalMemoryBufferCreateInfo extBufInfo{};
    extBufInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    extBufInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkBufferCreateInfo bufInfo{};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.pNext = &extBufInfo;
    bufInfo.size = sizeBytes;
    bufInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(impl_->device, &bufInfo, nullptr, &bufferOut) != VK_SUCCESS) {
        error = "Failed to create external Vulkan buffer.";
        return false;
    }

    VkMemoryRequirements memReqs;
    vkGetBufferMemoryRequirements(impl_->device, bufferOut, &memReqs);

    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(impl_->physicalDevice, &memProps);

    uint32_t memTypeIndex = 0;
    bool foundMemType = false;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((memReqs.memoryTypeBits & (1 << i)) &&
            (memProps.memoryTypes[i].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))) {
            memTypeIndex = i;
            foundMemType = true;
            break;
        }
    }

    if (!foundMemType) {
        vkDestroyBuffer(impl_->device, bufferOut, nullptr);
        error = "Failed to find suitable memory type for external buffer.";
        return false;
    }

    VkExportMemoryAllocateInfo exportAllocInfo{};
    exportAllocInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
    exportAllocInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.pNext = &exportAllocInfo;
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = memTypeIndex;

    if (vkAllocateMemory(impl_->device, &allocInfo, nullptr, &memoryOut) != VK_SUCCESS) {
        vkDestroyBuffer(impl_->device, bufferOut, nullptr);
        error = "Failed to allocate external Vulkan memory.";
        return false;
    }

    if (vkBindBufferMemory(impl_->device, bufferOut, memoryOut, 0) != VK_SUCCESS) {
        vkFreeMemory(impl_->device, memoryOut, nullptr);
        vkDestroyBuffer(impl_->device, bufferOut, nullptr);
        error = "Failed to bind external Vulkan memory.";
        return false;
    }

    VkMemoryGetFdInfoKHR getFdInfo{};
    getFdInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    getFdInfo.memory = memoryOut;
    getFdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    auto vkGetMemoryFdKHR = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(impl_->device, "vkGetMemoryFdKHR");
    if (!vkGetMemoryFdKHR) {
        vkFreeMemory(impl_->device, memoryOut, nullptr);
        vkDestroyBuffer(impl_->device, bufferOut, nullptr);
        error = "vkGetMemoryFdKHR not found.";
        return false;
    }

    if (vkGetMemoryFdKHR(impl_->device, &getFdInfo, &fdOut) != VK_SUCCESS) {
        vkFreeMemory(impl_->device, memoryOut, nullptr);
        vkDestroyBuffer(impl_->device, bufferOut, nullptr);
        error = "Failed to get FD for external Vulkan memory.";
        return false;
    }

    if (vkMapMemory(impl_->device, memoryOut, 0, sizeBytes, 0, mappedPtrOut) != VK_SUCCESS) {
        // We don't strictly need to fail if mapping fails, but for our degraded mode we do.
        vkFreeMemory(impl_->device, memoryOut, nullptr);
        vkDestroyBuffer(impl_->device, bufferOut, nullptr);
        error = "Failed to map external Vulkan memory.";
        return false;
    }

    return true;
}

void VulkanRuntime::freeExternalBuffer(VkBuffer buffer, VkDeviceMemory memory) {
    if (!impl_->available) return;
    if (memory != VK_NULL_HANDLE) {
        vkUnmapMemory(impl_->device, memory);
        vkFreeMemory(impl_->device, memory, nullptr);
    }
    if (buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(impl_->device, buffer, nullptr);
    }
}
#endif

} // namespace ave
