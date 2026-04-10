#include "ave/vulkan_runtime.hpp"

#ifdef AVE_HAVE_VULKAN
extern "C" {
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_vulkan.h>
}
#endif

#include <algorithm>
#include <array>
#include <iostream>
#include <limits>
#include <vector>

#if defined(__linux__)
#  include <unistd.h>
#endif

#ifdef AVE_HAVE_VULKAN
#include <vulkan/vulkan.h>
#endif

struct ave::VulkanRuntime::Impl {
    bool available = false;
    bool ownsContext = true;
#ifdef AVE_HAVE_VULKAN
    AVBufferRef* adoptedHwDeviceContext = nullptr;
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue computeQueue = VK_NULL_HANDLE;
    uint32_t computeQueueFamilyIndex = 0;
#endif
};

namespace {

void closeFdIfValid(const int fd) {
#if defined(__linux__)
    if (fd >= 0) {
        (void)::close(fd);
    }
#else
    (void)fd;
#endif
}

bool hasRequiredDeviceExtensions(VkPhysicalDevice device) {
    uint32_t extensionCount = 0;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount,
                                         availableExtensions.data());
    const std::array<const char*, 3> requiredExtensions = {{
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
        VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
    }};
    return std::all_of(requiredExtensions.begin(), requiredExtensions.end(),
                       [&](const char* requiredName) {
                           return std::any_of(
                               availableExtensions.begin(),
                               availableExtensions.end(),
                               [&](const VkExtensionProperties& property) {
                                   return std::string(property.extensionName) == requiredName;
                               });
                       });
}

uint32_t selectComputeQueueFamilyIndex(const AVVulkanDeviceContext* vkContext) {
    if (vkContext == nullptr) {
        return std::numeric_limits<uint32_t>::max();
    }

    for (int i = 0; i < vkContext->nb_qf; ++i) {
        if ((vkContext->qf[i].flags & VK_QUEUE_COMPUTE_BIT) != 0 &&
            vkContext->qf[i].idx >= 0) {
            return static_cast<uint32_t>(vkContext->qf[i].idx);
        }
    }

    for (int i = 0; i < vkContext->nb_qf; ++i) {
        if (vkContext->qf[i].idx >= 0) {
            return static_cast<uint32_t>(vkContext->qf[i].idx);
        }
    }

    return std::numeric_limits<uint32_t>::max();
}

}  // namespace

ave::VulkanRuntime::VulkanRuntime() : impl_(std::make_unique<Impl>()) {}

ave::VulkanRuntime::~VulkanRuntime() {
#ifdef AVE_HAVE_VULKAN
    if (impl_->adoptedHwDeviceContext != nullptr) {
        av_buffer_unref(&impl_->adoptedHwDeviceContext);
    }
    if (impl_->ownsContext) {
        if (impl_->device != VK_NULL_HANDLE) {
            vkDestroyDevice(impl_->device, nullptr);
        }
        if (impl_->instance != VK_NULL_HANDLE) {
            vkDestroyInstance(impl_->instance, nullptr);
        }
    }
#endif
}

bool ave::VulkanRuntime::isAvailable(std::string& reason) const {
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

bool ave::VulkanRuntime::initialize(std::string& error) {
#ifdef AVE_HAVE_VULKAN
    if (impl_->available) {
        return true;
    }
    auto resetRuntimeState = [this]() {
        if (impl_->adoptedHwDeviceContext != nullptr) {
            av_buffer_unref(&impl_->adoptedHwDeviceContext);
        }
        if (impl_->ownsContext) {
            if (impl_->device != VK_NULL_HANDLE) {
                vkDestroyDevice(impl_->device, nullptr);
            }
            if (impl_->instance != VK_NULL_HANDLE) {
                vkDestroyInstance(impl_->instance, nullptr);
            }
        }
        impl_->available = false;
        impl_->ownsContext = true;
        impl_->instance = VK_NULL_HANDLE;
        impl_->physicalDevice = VK_NULL_HANDLE;
        impl_->device = VK_NULL_HANDLE;
        impl_->computeQueue = VK_NULL_HANDLE;
        impl_->computeQueueFamilyIndex = 0;
    };
    resetRuntimeState();

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
        resetRuntimeState();
        error = "Failed to find GPUs with Vulkan support.";
        return false;
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(impl_->instance, &deviceCount, devices.data());

    VkPhysicalDevice selectedDevice = VK_NULL_HANDLE;
    int selectedScore = std::numeric_limits<int>::min();
    for (const auto device : devices) {
        if (!hasRequiredDeviceExtensions(device)) {
            continue;
        }
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(device, &properties);
        int score = 0;
        if (properties.vendorID == 0x1002u) {
            score += 100;
        }
        if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            score += 50;
        } else if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
            score += 10;
        }
        if (score > selectedScore) {
            selectedScore = score;
            selectedDevice = device;
        }
    }

    if (selectedDevice == VK_NULL_HANDLE) {
        resetRuntimeState();
        error = "Failed to find a Vulkan device with the required external-memory extensions.";
        return false;
    }

    impl_->physicalDevice = selectedDevice;

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
        resetRuntimeState();
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
        // Context7 explicitly mandates VK_KHR_synchronization2 to prevent cross-API stalls
    // Submissions must use vkCmdPipelineBarrier2 with appropriate VkImageMemoryBarrier2 / VkBufferMemoryBarrier2 structures.
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
        resetRuntimeState();
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

bool ave::VulkanRuntime::initialize(const AVBufferRef* ffmpegHwDeviceContext, std::string& error) {
#ifdef AVE_HAVE_VULKAN
    if (impl_->available) {
        return true;
    }

    auto resetRuntimeState = [this]() {
        if (impl_->adoptedHwDeviceContext != nullptr) {
            av_buffer_unref(&impl_->adoptedHwDeviceContext);
        }
        if (impl_->ownsContext) {
            if (impl_->device != VK_NULL_HANDLE) {
                vkDestroyDevice(impl_->device, nullptr);
            }
            if (impl_->instance != VK_NULL_HANDLE) {
                vkDestroyInstance(impl_->instance, nullptr);
            }
        }
        impl_->available = false;
        impl_->ownsContext = true;
        impl_->instance = VK_NULL_HANDLE;
        impl_->physicalDevice = VK_NULL_HANDLE;
        impl_->device = VK_NULL_HANDLE;
        impl_->computeQueue = VK_NULL_HANDLE;
        impl_->computeQueueFamilyIndex = 0;
    };

    resetRuntimeState();
    if (ffmpegHwDeviceContext == nullptr || ffmpegHwDeviceContext->data == nullptr) {
        error = "Failed to adopt FFmpeg Vulkan device context: context is null.";
        return false;
    }

    const auto* hwDeviceContext =
        reinterpret_cast<const AVHWDeviceContext*>(ffmpegHwDeviceContext->data);
    if (hwDeviceContext == nullptr || hwDeviceContext->hwctx == nullptr) {
        error = "Failed to adopt FFmpeg Vulkan device context: hardware context is missing.";
        return false;
    }
    if (hwDeviceContext->type != AV_HWDEVICE_TYPE_VULKAN) {
        error = "Failed to adopt FFmpeg Vulkan device context: device type is not Vulkan.";
        return false;
    }

    const auto* vkContext =
        reinterpret_cast<const AVVulkanDeviceContext*>(hwDeviceContext->hwctx);
    if (vkContext == nullptr || vkContext->inst == VK_NULL_HANDLE ||
        vkContext->phys_dev == VK_NULL_HANDLE || vkContext->act_dev == VK_NULL_HANDLE) {
        error = "Failed to adopt FFmpeg Vulkan device context: Vulkan handles are incomplete.";
        return false;
    }

    if (!hasRequiredDeviceExtensions(vkContext->phys_dev)) {
        error =
            "Failed to adopt FFmpeg Vulkan device context: required Vulkan device "
            "extensions are unavailable on the supplied device.";
        return false;
    }

    const uint32_t computeQueueFamilyIndex = selectComputeQueueFamilyIndex(vkContext);
    if (computeQueueFamilyIndex == std::numeric_limits<uint32_t>::max()) {
        error = "Failed to adopt FFmpeg Vulkan device context: no compute queue family is available.";
        return false;
    }

    const auto vkGetMemoryFd =
        reinterpret_cast<PFN_vkGetMemoryFdKHR>(
            vkGetDeviceProcAddr(vkContext->act_dev, "vkGetMemoryFdKHR"));
    const auto vkGetSemaphoreFd =
        reinterpret_cast<PFN_vkGetSemaphoreFdKHR>(
            vkGetDeviceProcAddr(vkContext->act_dev, "vkGetSemaphoreFdKHR"));
    if (vkGetMemoryFd == nullptr || vkGetSemaphoreFd == nullptr) {
        error =
            "Failed to adopt FFmpeg Vulkan device context: external-memory/semaphore "
            "export entry points are unavailable on the supplied device.";
        return false;
    }

    AVBufferRef* adoptedRef = av_buffer_ref(const_cast<AVBufferRef*>(ffmpegHwDeviceContext));
    if (adoptedRef == nullptr) {
        error = "Failed to retain FFmpeg Vulkan device context.";
        return false;
    }

    impl_->adoptedHwDeviceContext = adoptedRef;
    impl_->instance = vkContext->inst;
    impl_->physicalDevice = vkContext->phys_dev;
    impl_->device = vkContext->act_dev;
    impl_->computeQueueFamilyIndex = computeQueueFamilyIndex;
    vkGetDeviceQueue(impl_->device, impl_->computeQueueFamilyIndex, 0, &impl_->computeQueue);
    impl_->available = true;
    impl_->ownsContext = false;
    return true;
#else
    (void)ffmpegHwDeviceContext;
    error = "Vulkan support not compiled in.";
    return false;
#endif
}

#ifdef AVE_HAVE_VULKAN
VkInstance ave::VulkanRuntime::getInstance() const { return impl_->instance; }
VkPhysicalDevice ave::VulkanRuntime::getPhysicalDevice() const { return impl_->physicalDevice; }
VkDevice ave::VulkanRuntime::getDevice() const { return impl_->device; }
VkQueue ave::VulkanRuntime::getComputeQueue() const { return impl_->computeQueue; }
uint32_t ave::VulkanRuntime::getComputeQueueFamilyIndex() const { return impl_->computeQueueFamilyIndex; }

bool ave::VulkanRuntime::allocateExternalBuffer(std::size_t sizeBytes,
                                                VkBuffer& bufferOut,
                                                VkDeviceMemory& memoryOut,
                                                int& fdOut,
                                                void** mappedPtrOut,
                                                std::string& error) {
    bufferOut = VK_NULL_HANDLE;
    memoryOut = VK_NULL_HANDLE;
    fdOut = -1;
    if (mappedPtrOut != nullptr) {
        *mappedPtrOut = nullptr;
    }
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

    VkBuffer buffer = VK_NULL_HANDLE;
    if (vkCreateBuffer(impl_->device, &bufInfo, nullptr, &buffer) != VK_SUCCESS) {
        error = "Failed to create external Vulkan buffer.";
        return false;
    }

    VkMemoryRequirements memReqs;
    vkGetBufferMemoryRequirements(impl_->device, buffer, &memReqs);

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
        vkDestroyBuffer(impl_->device, buffer, nullptr);
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

    VkDeviceMemory memory = VK_NULL_HANDLE;
    if (vkAllocateMemory(impl_->device, &allocInfo, nullptr, &memory) != VK_SUCCESS) {
        vkDestroyBuffer(impl_->device, buffer, nullptr);
        error = "Failed to allocate external Vulkan memory.";
        return false;
    }

    if (vkBindBufferMemory(impl_->device, buffer, memory, 0) != VK_SUCCESS) {
        vkFreeMemory(impl_->device, memory, nullptr);
        vkDestroyBuffer(impl_->device, buffer, nullptr);
        error = "Failed to bind external Vulkan memory.";
        return false;
    }

    VkMemoryGetFdInfoKHR getFdInfo{};
    getFdInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    getFdInfo.memory = memory;
    getFdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    auto vkGetMemoryFdKHR = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(impl_->device, "vkGetMemoryFdKHR");
    if (!vkGetMemoryFdKHR) {
        vkFreeMemory(impl_->device, memory, nullptr);
        vkDestroyBuffer(impl_->device, buffer, nullptr);
        error = "vkGetMemoryFdKHR not found.";
        return false;
    }

    int exportedFd = -1;
    if (vkGetMemoryFdKHR(impl_->device, &getFdInfo, &exportedFd) != VK_SUCCESS) {
        vkFreeMemory(impl_->device, memory, nullptr);
        vkDestroyBuffer(impl_->device, buffer, nullptr);
        error = "Failed to get FD for external Vulkan memory.";
        return false;
    }

    void* mappedPtr = nullptr;
    if (vkMapMemory(impl_->device, memory, 0, sizeBytes, 0, &mappedPtr) != VK_SUCCESS) {
        closeFdIfValid(exportedFd);
        vkFreeMemory(impl_->device, memory, nullptr);
        vkDestroyBuffer(impl_->device, buffer, nullptr);
        error = "Failed to map external Vulkan memory.";
        return false;
    }

    bufferOut = buffer;
    memoryOut = memory;
    fdOut = exportedFd;
    if (mappedPtrOut != nullptr) {
        *mappedPtrOut = mappedPtr;
    }
    return true;
}

void ave::VulkanRuntime::freeExternalBuffer(VkBuffer buffer, VkDeviceMemory memory) {
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
