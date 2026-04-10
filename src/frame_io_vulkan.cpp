#include "ave/frame_io.hpp"
#include "ave/vulkan_runtime.hpp"

extern "C" {
#include <libavutil/imgutils.h>
#include <libavutil/hwcontext_vulkan.h>
}

#include <algorithm>
#include <array>
#include <iostream>
#include <limits>
#if defined(__linux__)
#  include <unistd.h>
#endif

namespace ave {
namespace frame_io {

namespace {

#ifdef AVE_HAVE_VULKAN
struct VulkanPlaneInfo {
    uint32_t width = 0;
    uint32_t height = 0;
    VkImage image = VK_NULL_HANDLE;
    VkImageAspectFlags aspectMask = 0;
    VkImageLayout originalLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkAccessFlags originalAccess = 0;
    VkSemaphore semaphore = VK_NULL_HANDLE;
    uint64_t waitValue = 0;
    uint64_t signalValue = 0;
    std::size_t bufferOffset = 0u;
};

struct VulkanFrameStageInfo {
    AVPixelFormat swFormat = AV_PIX_FMT_NONE;
    int width = 0;
    int height = 0;
    std::array<VulkanPlaneInfo, AV_NUM_DATA_POINTERS> planes{};
    int planeCount = 0;
    std::size_t totalBytes = 0u;
};

uint32_t planeWidth(const AVPixFmtDescriptor* descriptor,
                    const int width,
                    const int planeIndex) {
    if (descriptor == nullptr || planeIndex <= 0) {
        return static_cast<uint32_t>(std::max(width, 0));
    }
    const int shiftedWidth =
        (width + (1 << descriptor->log2_chroma_w) - 1) >> descriptor->log2_chroma_w;
    return static_cast<uint32_t>(std::max(shiftedWidth, 0));
}

uint32_t planeHeight(const AVPixFmtDescriptor* descriptor,
                     const int height,
                     const int planeIndex) {
    if (descriptor == nullptr || planeIndex <= 0) {
        return static_cast<uint32_t>(std::max(height, 0));
    }
    const int shiftedHeight =
        (height + (1 << descriptor->log2_chroma_h) - 1) >> descriptor->log2_chroma_h;
    return static_cast<uint32_t>(std::max(shiftedHeight, 0));
}

bool isSupportedVulkanStageFormat(const AVPixelFormat format) {
    switch (format) {
        case AV_PIX_FMT_RGB24:
        case AV_PIX_FMT_BGR24:
        case AV_PIX_FMT_RGBA:
        case AV_PIX_FMT_BGRA:
        case AV_PIX_FMT_ARGB:
        case AV_PIX_FMT_ABGR:
        case AV_PIX_FMT_YUV420P:
        case AV_PIX_FMT_YUVJ420P:
        case AV_PIX_FMT_NV12:
        case AV_PIX_FMT_P010LE:
            return true;
        default:
            return false;
    }
}

bool usesMultiPlaneImageAspects(const AVPixelFormat format) {
    switch (format) {
        case AV_PIX_FMT_YUV420P:
        case AV_PIX_FMT_YUVJ420P:
        case AV_PIX_FMT_NV12:
        case AV_PIX_FMT_P010LE:
            return true;
        default:
            return false;
    }
}

VkImageAspectFlags aspectMaskForPlane(const AVPixelFormat format,
                                      const int planeIndex,
                                      const bool separateImages) {
    if (separateImages || !usesMultiPlaneImageAspects(format)) {
        return VK_IMAGE_ASPECT_COLOR_BIT;
    }
    switch (planeIndex) {
        case 0: return VK_IMAGE_ASPECT_PLANE_0_BIT;
        case 1: return VK_IMAGE_ASPECT_PLANE_1_BIT;
        case 2: return VK_IMAGE_ASPECT_PLANE_2_BIT;
        default: return 0;
    }
}

bool buildVulkanStageInfo(const AVFrame* hardwareFrame,
                          VulkanFrameStageInfo& info,
                          std::string& error) {
    if (hardwareFrame == nullptr || hardwareFrame->format != AV_PIX_FMT_VULKAN) {
        error = "Vulkan frame staging requires an AV_PIX_FMT_VULKAN frame.";
        return false;
    }
    if (hardwareFrame->hw_frames_ctx == nullptr ||
        hardwareFrame->hw_frames_ctx->data == nullptr) {
        error = "Vulkan frame staging requires a valid hw_frames_ctx.";
        return false;
    }

    const auto* hwFramesContext =
        reinterpret_cast<const AVHWFramesContext*>(hardwareFrame->hw_frames_ctx->data);
    if (hwFramesContext == nullptr || hwFramesContext->hwctx == nullptr) {
        error = "Vulkan frame staging requires a valid hardware-frames context.";
        return false;
    }

    const auto* vkFramesContext =
        reinterpret_cast<const AVVulkanFramesContext*>(hwFramesContext->hwctx);
    const auto* vkFrame = reinterpret_cast<const AVVkFrame*>(hardwareFrame->data[0]);
    if (vkFramesContext == nullptr || vkFrame == nullptr) {
        error = "Vulkan frame staging requires AVVkFrame metadata.";
        return false;
    }

    const AVPixelFormat swFormat = hwFramesContext->sw_format;
    if (!isSupportedVulkanStageFormat(swFormat)) {
        error = "Vulkan frame staging does not support software format "
              + std::to_string(static_cast<int>(swFormat)) + ".";
        return false;
    }

    const auto* descriptor = av_pix_fmt_desc_get(swFormat);
    if (descriptor == nullptr) {
        error = "Vulkan frame staging could not describe the software pixel format.";
        return false;
    }

    std::array<int, AV_NUM_DATA_POINTERS> rawLinesizes{};
    if (av_image_fill_linesizes(rawLinesizes.data(), swFormat, hardwareFrame->width) < 0) {
        error = "Vulkan frame staging failed to compute software linesizes.";
        return false;
    }
    std::array<std::ptrdiff_t, AV_NUM_DATA_POINTERS> linesizes{};
    for (std::size_t i = 0; i < linesizes.size(); ++i) {
        linesizes[i] = rawLinesizes[i];
    }
    std::array<size_t, AV_NUM_DATA_POINTERS> planeSizes{};
    if (av_image_fill_plane_sizes(planeSizes.data(),
                                  swFormat,
                                  hardwareFrame->height,
                                  linesizes.data()) < 0) {
        error = "Vulkan frame staging failed to compute software plane sizes.";
        return false;
    }

    const bool separateImages =
        vkFrame->img[1] != VK_NULL_HANDLE || vkFrame->img[2] != VK_NULL_HANDLE;

    info = {};
    info.swFormat = swFormat;
    info.width = hardwareFrame->width;
    info.height = hardwareFrame->height;

    std::size_t runningOffset = 0u;
    for (int planeIndex = 0; planeIndex < AV_NUM_DATA_POINTERS; ++planeIndex) {
        const auto planeSlot = static_cast<std::size_t>(planeIndex);
        if (planeSizes[planeSlot] == 0u) {
            continue;
        }
        VkImage image = vkFrame->img[planeIndex];
        if (image == VK_NULL_HANDLE) {
            image = vkFrame->img[0];
        }
        if (image == VK_NULL_HANDLE) {
            error = "Vulkan frame staging requires a valid source image for every plane.";
            return false;
        }

        auto& plane = info.planes[static_cast<std::size_t>(info.planeCount++)];
        plane.width = planeWidth(descriptor, hardwareFrame->width, planeIndex);
        plane.height = planeHeight(descriptor, hardwareFrame->height, planeIndex);
        plane.image = image;
        plane.aspectMask = aspectMaskForPlane(swFormat, planeIndex, separateImages);
        plane.originalLayout = vkFrame->layout[planeIndex];
        plane.originalAccess = vkFrame->access[planeIndex];
        plane.semaphore = vkFrame->sem[planeIndex];
        plane.waitValue = vkFrame->sem_value[planeIndex];
        plane.signalValue = plane.waitValue + 1u;
        plane.bufferOffset = runningOffset;
        runningOffset += planeSizes[planeSlot];
    }
    info.totalBytes = runningOffset;

    if (info.planeCount <= 0 || info.totalBytes == 0u) {
        error = "Vulkan frame staging found no populated planes to stage.";
        return false;
    }
    return true;
}

bool submitVulkanStageCopy(const VulkanFrameStageInfo& info,
                           AVFrame* mutableFrame,
                           VulkanRuntime& runtime,
                           VkCommandPool commandPool,
                           VkCommandBuffer commandBuffer,
                           VkBuffer dstBuffer,
                           std::string& error) {
    auto* hwFramesContext =
        reinterpret_cast<AVHWFramesContext*>(mutableFrame->hw_frames_ctx->data);
    auto* vkFramesContext =
        reinterpret_cast<AVVulkanFramesContext*>(hwFramesContext->hwctx);
    auto* vkFrame = reinterpret_cast<AVVkFrame*>(mutableFrame->data[0]);
    if (vkFramesContext == nullptr || vkFrame == nullptr) {
        error = "Vulkan frame staging lost AVVkFrame metadata during submission.";
        return false;
    }

    if (vkFramesContext->lock_frame != nullptr) {
        vkFramesContext->lock_frame(hwFramesContext, vkFrame);
    }

    const auto unlockFrame = [&]() {
        if (vkFramesContext->unlock_frame != nullptr) {
            vkFramesContext->unlock_frame(hwFramesContext, vkFrame);
        }
    };

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
        unlockFrame();
        error = "Vulkan frame staging failed to begin the command buffer.";
        return false;
    }

    for (int planeIndex = 0; planeIndex < info.planeCount; ++planeIndex) {
        const auto& plane = info.planes[static_cast<std::size_t>(planeIndex)];

        VkImageMemoryBarrier toTransfer{};
        toTransfer.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        toTransfer.srcAccessMask = plane.originalAccess;
        toTransfer.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        toTransfer.oldLayout = plane.originalLayout;
        toTransfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        toTransfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toTransfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toTransfer.image = plane.image;
        toTransfer.subresourceRange.aspectMask = plane.aspectMask;
        toTransfer.subresourceRange.baseMipLevel = 0;
        toTransfer.subresourceRange.levelCount = 1;
        toTransfer.subresourceRange.baseArrayLayer = 0;
        toTransfer.subresourceRange.layerCount = 1;
        vkCmdPipelineBarrier(commandBuffer,
                             VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0,
                             0, nullptr,
                             0, nullptr,
                             1, &toTransfer);

        VkBufferImageCopy copyRegion{};
        copyRegion.bufferOffset = plane.bufferOffset;
        copyRegion.bufferRowLength = 0;
        copyRegion.bufferImageHeight = 0;
        copyRegion.imageSubresource.aspectMask = plane.aspectMask;
        copyRegion.imageSubresource.mipLevel = 0;
        copyRegion.imageSubresource.baseArrayLayer = 0;
        copyRegion.imageSubresource.layerCount = 1;
        copyRegion.imageOffset = {0, 0, 0};
        copyRegion.imageExtent = {plane.width, plane.height, 1};
        vkCmdCopyImageToBuffer(commandBuffer,
                               plane.image,
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               dstBuffer,
                               1,
                               &copyRegion);

        VkImageMemoryBarrier restoreBarrier{};
        restoreBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        restoreBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        restoreBarrier.dstAccessMask = plane.originalAccess;
        restoreBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        restoreBarrier.newLayout = plane.originalLayout;
        restoreBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        restoreBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        restoreBarrier.image = plane.image;
        restoreBarrier.subresourceRange.aspectMask = plane.aspectMask;
        restoreBarrier.subresourceRange.baseMipLevel = 0;
        restoreBarrier.subresourceRange.levelCount = 1;
        restoreBarrier.subresourceRange.baseArrayLayer = 0;
        restoreBarrier.subresourceRange.layerCount = 1;
        vkCmdPipelineBarrier(commandBuffer,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             0,
                             0, nullptr,
                             0, nullptr,
                             1, &restoreBarrier);
    }

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        unlockFrame();
        error = "Vulkan frame staging failed to end the command buffer.";
        return false;
    }

    std::vector<VkSemaphore> waitSemaphores;
    std::vector<uint64_t> waitValues;
    std::vector<VkSemaphore> signalSemaphores;
    std::vector<uint64_t> signalValues;
    waitSemaphores.reserve(static_cast<std::size_t>(info.planeCount));
    waitValues.reserve(static_cast<std::size_t>(info.planeCount));
    signalSemaphores.reserve(static_cast<std::size_t>(info.planeCount));
    signalValues.reserve(static_cast<std::size_t>(info.planeCount));
    for (int planeIndex = 0; planeIndex < info.planeCount; ++planeIndex) {
        const auto& plane = info.planes[static_cast<std::size_t>(planeIndex)];
        if (plane.semaphore == VK_NULL_HANDLE) {
            continue;
        }
        const auto existingIt =
            std::find(waitSemaphores.begin(), waitSemaphores.end(), plane.semaphore);
        if (existingIt != waitSemaphores.end()) {
            continue;
        }
        waitSemaphores.push_back(plane.semaphore);
        waitValues.push_back(plane.waitValue);
        signalSemaphores.push_back(plane.semaphore);
        signalValues.push_back(plane.signalValue);
    }

    VkTimelineSemaphoreSubmitInfo timelineInfo{};
    timelineInfo.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    timelineInfo.waitSemaphoreValueCount =
        static_cast<uint32_t>(waitValues.size());
    timelineInfo.pWaitSemaphoreValues = waitValues.data();
    timelineInfo.signalSemaphoreValueCount =
        static_cast<uint32_t>(signalValues.size());
    timelineInfo.pSignalSemaphoreValues = signalValues.data();

    std::vector<VkPipelineStageFlags> waitStages(waitSemaphores.size(),
                                                 VK_PIPELINE_STAGE_TRANSFER_BIT);
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.pNext = waitSemaphores.empty() ? nullptr : &timelineInfo;
    submitInfo.waitSemaphoreCount =
        static_cast<uint32_t>(waitSemaphores.size());
    submitInfo.pWaitSemaphores = waitSemaphores.data();
    submitInfo.pWaitDstStageMask = waitStages.data();
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    submitInfo.signalSemaphoreCount =
        static_cast<uint32_t>(signalSemaphores.size());
    submitInfo.pSignalSemaphores = signalSemaphores.data();

    if (vkQueueSubmit(runtime.getComputeQueue(), 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
        unlockFrame();
        error = "Vulkan frame staging failed to submit the copy workload.";
        return false;
    }
    if (vkQueueWaitIdle(runtime.getComputeQueue()) != VK_SUCCESS) {
        unlockFrame();
        error = "Vulkan frame staging failed to wait for the copy workload.";
        return false;
    }

    for (int planeIndex = 0; planeIndex < info.planeCount; ++planeIndex) {
        const auto planeSlot = static_cast<std::size_t>(planeIndex);
        if (info.planes[planeSlot].semaphore != VK_NULL_HANDLE) {
            vkFrame->sem_value[planeIndex] = info.planes[planeSlot].signalValue;
        }
    }

    unlockFrame();
    if (vkResetCommandPool(runtime.getDevice(), commandPool, 0) != VK_SUCCESS) {
        error = "Vulkan frame staging failed to reset the command pool.";
        return false;
    }
    return true;
}
#endif

}  // namespace

bool isVulkanHardwareFrame(const AVFrame* frame) {
    return frame != nullptr && frame->format == AV_PIX_FMT_VULKAN;
}

struct VulkanHardwareFrameStager::Impl {
    AVFrame* softwareFrame = nullptr;
#ifdef AVE_HAVE_VULKAN
    VkDevice device = VK_NULL_HANDLE;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mappedPtr = nullptr;
#endif
    std::size_t capacityBytes = 0u;

    ~Impl() {
        if (softwareFrame != nullptr) {
            av_frame_free(&softwareFrame);
        }
    }
};

VulkanHardwareFrameStager::VulkanHardwareFrameStager()
    : impl_(std::make_unique<Impl>()) {
    impl_->softwareFrame = av_frame_alloc();
}

VulkanHardwareFrameStager::~VulkanHardwareFrameStager() {
    reset();
}

void VulkanHardwareFrameStager::reset() {
    if (!impl_) {
        return;
    }
#ifdef AVE_HAVE_VULKAN
    if (impl_->device != VK_NULL_HANDLE) {
        if (impl_->buffer != VK_NULL_HANDLE || impl_->memory != VK_NULL_HANDLE) {
            if (impl_->memory != VK_NULL_HANDLE) {
                vkUnmapMemory(impl_->device, impl_->memory);
                vkFreeMemory(impl_->device, impl_->memory, nullptr);
            }
            if (impl_->buffer != VK_NULL_HANDLE) {
                vkDestroyBuffer(impl_->device, impl_->buffer, nullptr);
            }
        }
        if (impl_->commandPool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(impl_->device, impl_->commandPool, nullptr);
        }
    }
    impl_->device = VK_NULL_HANDLE;
    impl_->commandPool = VK_NULL_HANDLE;
    impl_->commandBuffer = VK_NULL_HANDLE;
    impl_->buffer = VK_NULL_HANDLE;
    impl_->memory = VK_NULL_HANDLE;
    impl_->mappedPtr = nullptr;
#endif
    impl_->capacityBytes = 0u;
    if (impl_->softwareFrame != nullptr) {
        av_frame_unref(impl_->softwareFrame);
    }
}

bool VulkanHardwareFrameStager::stageToSoftwareFrame(const AVFrame* hardwareFrame,
                                                     VulkanRuntime& runtime,
                                                     AVFrame*& frameOut,
                                                     std::string& error) {
    frameOut = nullptr;
    if (hardwareFrame == nullptr) {
        error = "Vulkan frame staging requires a non-null source frame.";
        return false;
    }
    if (!isVulkanHardwareFrame(hardwareFrame)) {
        error = "Vulkan frame staging requires an AV_PIX_FMT_VULKAN source frame.";
        return false;
    }
    if (impl_ == nullptr || impl_->softwareFrame == nullptr) {
        error = "Vulkan frame staging scratch buffers are unavailable.";
        return false;
    }

#ifndef AVE_HAVE_VULKAN
    (void)runtime;
    error = "Vulkan support was not compiled in.";
    return false;
#else
    VulkanFrameStageInfo stageInfo;
    if (!buildVulkanStageInfo(hardwareFrame, stageInfo, error)) {
        return false;
    }

    if (runtime.getDevice() == VK_NULL_HANDLE) {
        error = "Vulkan frame staging requires an initialized Vulkan runtime.";
        return false;
    }

    if (impl_->device != runtime.getDevice()) {
        reset();
        impl_->device = runtime.getDevice();

        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = runtime.getComputeQueueFamilyIndex();
        if (vkCreateCommandPool(impl_->device, &poolInfo, nullptr, &impl_->commandPool) != VK_SUCCESS) {
            error = "Vulkan frame staging failed to create the command pool.";
            return false;
        }

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = impl_->commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(impl_->device, &allocInfo, &impl_->commandBuffer) != VK_SUCCESS) {
            error = "Vulkan frame staging failed to allocate the command buffer.";
            return false;
        }
    }

    if (impl_->capacityBytes < stageInfo.totalBytes ||
        impl_->buffer == VK_NULL_HANDLE ||
        impl_->memory == VK_NULL_HANDLE ||
        impl_->mappedPtr == nullptr) {
        if (impl_->buffer != VK_NULL_HANDLE || impl_->memory != VK_NULL_HANDLE) {
            runtime.freeExternalBuffer(impl_->buffer, impl_->memory);
            impl_->buffer = VK_NULL_HANDLE;
            impl_->memory = VK_NULL_HANDLE;
            impl_->mappedPtr = nullptr;
            impl_->capacityBytes = 0u;
        }

        int exportedFd = -1;
        if (!runtime.allocateExternalBuffer(stageInfo.totalBytes,
                                            impl_->buffer,
                                            impl_->memory,
                                            exportedFd,
                                            &impl_->mappedPtr,
                                            error)) {
            return false;
        }
#if defined(__linux__)
        if (exportedFd >= 0) {
            (void)::close(exportedFd);
        }
#else
        (void)exportedFd;
#endif
        impl_->capacityBytes = stageInfo.totalBytes;
    }

    if (!submitVulkanStageCopy(stageInfo,
                               const_cast<AVFrame*>(hardwareFrame),
                               runtime,
                               impl_->commandPool,
                               impl_->commandBuffer,
                               impl_->buffer,
                               error)) {
        return false;
    }

    av_frame_unref(impl_->softwareFrame);
    impl_->softwareFrame->format = stageInfo.swFormat;
    impl_->softwareFrame->width = stageInfo.width;
    impl_->softwareFrame->height = stageInfo.height;
    if (av_image_fill_arrays(impl_->softwareFrame->data,
                             impl_->softwareFrame->linesize,
                             reinterpret_cast<std::uint8_t*>(impl_->mappedPtr),
                             stageInfo.swFormat,
                             stageInfo.width,
                             stageInfo.height,
                             1) < 0) {
        error = "Vulkan frame staging failed to create a software view over the staged buffer.";
        return false;
    }
    impl_->softwareFrame->extended_data = impl_->softwareFrame->data;
    frameOut = impl_->softwareFrame;
    return true;
#endif
}

struct VulkanVideoReader::Impl {
    AVFormatContext* fmt_ctx = nullptr;
    AVCodecContext* codec_ctx = nullptr;
    AVBufferRef* hw_device_ctx = nullptr;
    int video_stream_idx = -1;
    AVPacket* pkt = nullptr;
    AVFrame* frame = nullptr;
    AVFrame* sw_frame = nullptr;
    bool decoderDraining = false;
};

static enum AVPixelFormat get_hw_format(AVCodecContext */*ctx*/, const enum AVPixelFormat *pix_fmts) {
    const enum AVPixelFormat *p;
    for (p = pix_fmts; *p != -1; p++) {
        if (*p == AV_PIX_FMT_VULKAN)
            return *p;
    }
    std::cerr << "Failed to get HW surface format." << std::endl;
    return AV_PIX_FMT_NONE;
}

VulkanVideoReader::VulkanVideoReader() : impl_(std::make_unique<Impl>()) {
    impl_->pkt = av_packet_alloc();
    impl_->frame = av_frame_alloc();
    impl_->sw_frame = av_frame_alloc();
}

VulkanVideoReader::~VulkanVideoReader() {
    close();
}

bool VulkanVideoReader::open(const std::string& path, std::string& error) {
    if (avformat_open_input(&impl_->fmt_ctx, path.c_str(), nullptr, nullptr) < 0) {
        error = "Could not open input file: " + path;
        return false;
    }

    if (avformat_find_stream_info(impl_->fmt_ctx, nullptr) < 0) {
        error = "Could not find stream information";
        return false;
    }

    const AVCodec* codec = nullptr;
    impl_->video_stream_idx = av_find_best_stream(impl_->fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, &codec, 0);
    if (impl_->video_stream_idx < 0) {
        error = "Could not find video stream";
        return false;
    }

    if (av_hwdevice_ctx_create(&impl_->hw_device_ctx, AV_HWDEVICE_TYPE_VULKAN, nullptr, nullptr, 0) < 0) {
        error = "Failed to create Vulkan hardware device context";
        return false;
    }

    impl_->codec_ctx = avcodec_alloc_context3(codec);
    if (!impl_->codec_ctx) {
        error = "Failed to allocate codec context";
        return false;
    }

    avcodec_parameters_to_context(impl_->codec_ctx, impl_->fmt_ctx->streams[impl_->video_stream_idx]->codecpar);
    impl_->codec_ctx->hw_device_ctx = av_buffer_ref(impl_->hw_device_ctx);
    impl_->codec_ctx->get_format = get_hw_format;

    if (avcodec_open2(impl_->codec_ctx, codec, nullptr) < 0) {
        error = "Failed to open codec";
        return false;
    }

    return true;
}

bool VulkanVideoReader::readFrame(AVFrame*& outFrame, std::string& error) {
    return readFrame(outFrame, VulkanFrameReadMode::TransferToHost, error);
}

bool VulkanVideoReader::readFrame(AVFrame*& outFrame,
                                  VulkanFrameReadMode mode,
                                  std::string& error) {
    auto resolveReturnedFrame = [&](AVFrame*& resolvedFrame) -> bool {
        if (resolvedFrame == nullptr) {
            outFrame = nullptr;
            return true;
        }

        if (!isVulkanHardwareFrame(resolvedFrame) ||
            mode == VulkanFrameReadMode::PreserveHardware) {
            outFrame = resolvedFrame;
            return true;
        }

        if (av_hwframe_transfer_data(impl_->sw_frame, resolvedFrame, 0) < 0) {
            error = "Error transferring frame from Vulkan to software memory";
            return false;
        }
        outFrame = impl_->sw_frame;
        return true;
    };

    while (true) {
        int ret = 0;
        if (!impl_->decoderDraining) {
            ret = av_read_frame(impl_->fmt_ctx, impl_->pkt);
            if (ret < 0) {
                if (ret != AVERROR_EOF) {
                    error = "Error reading frame";
                    return false;
                }
                impl_->decoderDraining = true;
                av_packet_unref(impl_->pkt);
                ret = avcodec_send_packet(impl_->codec_ctx, nullptr);
                if (ret < 0) {
                    error = "Error draining decoder";
                    return false;
                }
            }
        } else {
            ret = AVERROR_EOF;
        }

        if (!impl_->decoderDraining && impl_->pkt->stream_index == impl_->video_stream_idx) {
            ret = avcodec_send_packet(impl_->codec_ctx, impl_->pkt);
            if (ret < 0) {
                error = "Error sending packet to decoder";
                av_packet_unref(impl_->pkt);
                return false;
            }

            while (ret >= 0) {
                av_frame_unref(impl_->frame);
                av_frame_unref(impl_->sw_frame);
                ret = avcodec_receive_frame(impl_->codec_ctx, impl_->frame);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                    break;
                } else if (ret < 0) {
                    error = "Error receiving frame from decoder";
                    av_packet_unref(impl_->pkt);
                    return false;
                }

                if (!resolveReturnedFrame(impl_->frame)) {
                    av_packet_unref(impl_->pkt);
                    return false;
                }
                av_packet_unref(impl_->pkt);
                return true;
            }
        } else if (impl_->decoderDraining) {
            av_frame_unref(impl_->frame);
            av_frame_unref(impl_->sw_frame);
            ret = avcodec_receive_frame(impl_->codec_ctx, impl_->frame);
            if (ret == AVERROR_EOF || ret == AVERROR(EAGAIN)) {
                outFrame = nullptr;
                return true;
            }
            if (ret < 0) {
                error = "Error receiving drained frame from decoder";
                return false;
            }

            if (!resolveReturnedFrame(impl_->frame)) {
                if (isVulkanHardwareFrame(impl_->frame)) {
                    error = "Error transferring drained frame from Vulkan to software memory";
                }
                return false;
            }
            return true;
        }
        av_packet_unref(impl_->pkt);
    }
}

void VulkanVideoReader::close() {
    if (impl_->codec_ctx) {
        avcodec_free_context(&impl_->codec_ctx);
        impl_->codec_ctx = nullptr;
    }
    if (impl_->fmt_ctx) {
        avformat_close_input(&impl_->fmt_ctx);
        impl_->fmt_ctx = nullptr;
    }
    if (impl_->hw_device_ctx) {
        av_buffer_unref(&impl_->hw_device_ctx);
        impl_->hw_device_ctx = nullptr;
    }
    if (impl_->pkt) {
        av_packet_free(&impl_->pkt);
        impl_->pkt = nullptr;
    }
    if (impl_->frame) {
        av_frame_free(&impl_->frame);
        impl_->frame = nullptr;
    }
    if (impl_->sw_frame) {
        av_frame_free(&impl_->sw_frame);
        impl_->sw_frame = nullptr;
    }
    impl_->video_stream_idx = -1;
    impl_->decoderDraining = false;
}

int VulkanVideoReader::width() const { return impl_->codec_ctx ? impl_->codec_ctx->width : 0; }
int VulkanVideoReader::height() const { return impl_->codec_ctx ? impl_->codec_ctx->height : 0; }
AVRational VulkanVideoReader::frameRate() const { return impl_->fmt_ctx ? impl_->fmt_ctx->streams[impl_->video_stream_idx]->avg_frame_rate : AVRational{0, 1}; }
const AVBufferRef* VulkanVideoReader::hwDeviceContext() const { return impl_->hw_device_ctx; }

const AVVulkanDeviceContext* VulkanVideoReader::vulkanDeviceContext() const {
    if (impl_->hw_device_ctx == nullptr || impl_->hw_device_ctx->data == nullptr) {
        return nullptr;
    }
    const auto* hwDeviceContext =
        reinterpret_cast<const AVHWDeviceContext*>(impl_->hw_device_ctx->data);
    if (hwDeviceContext == nullptr || hwDeviceContext->hwctx == nullptr) {
        return nullptr;
    }
    return reinterpret_cast<const AVVulkanDeviceContext*>(hwDeviceContext->hwctx);
}

struct VulkanVideoWriter::Impl {
    AVFormatContext* fmt_ctx = nullptr;
    AVCodecContext* codec_ctx = nullptr;
    AVStream* video_stream = nullptr;
    AVPacket* pkt = nullptr;
    AVFrame* sw_frame = nullptr;
    int frame_count = 0;
    bool header_written = false;
};

VulkanVideoWriter::VulkanVideoWriter() : impl_(std::make_unique<Impl>()) {
    impl_->pkt = av_packet_alloc();
    impl_->sw_frame = av_frame_alloc();
}

VulkanVideoWriter::~VulkanVideoWriter() {
    close();
}

bool VulkanVideoWriter::open(const std::string& path, int width, int height, AVRational fps, std::string& error) {
    close();

    if (impl_->pkt == nullptr) {
        impl_->pkt = av_packet_alloc();
        if (impl_->pkt == nullptr) {
            error = "Failed to allocate packet scratch buffer";
            return false;
        }
    }
    if (impl_->sw_frame == nullptr) {
        impl_->sw_frame = av_frame_alloc();
        if (impl_->sw_frame == nullptr) {
            error = "Failed to allocate frame scratch buffer";
            return false;
        }
    }
    impl_->frame_count = 0;
    impl_->header_written = false;
    impl_->video_stream = nullptr;

    if (avformat_alloc_output_context2(&impl_->fmt_ctx, nullptr, nullptr, path.c_str()) < 0) {
        error = "Could not allocate output context for " + path;
        return false;
    }

    AVCodecID codecId = AV_CODEC_ID_H264;
    const auto extension = std::filesystem::path(path).extension().string();
    if (extension == ".mkv" || extension == ".avi") {
        codecId = AV_CODEC_ID_FFV1;
    }

    const AVCodec* codec = avcodec_find_encoder(codecId);
    if (!codec) {
        error = "Requested video encoder not found";
        return false;
    }

    impl_->video_stream = avformat_new_stream(impl_->fmt_ctx, codec);
    if (!impl_->video_stream) {
        error = "Failed to create new stream";
        return false;
    }

    impl_->codec_ctx = avcodec_alloc_context3(codec);
    if (!impl_->codec_ctx) {
        error = "Failed to allocate codec context";
        return false;
    }

    impl_->codec_ctx->width = width;
    impl_->codec_ctx->height = height;
    const AVRational safeFps = (fps.num > 0 && fps.den > 0) ? fps : AVRational{30, 1};
    impl_->codec_ctx->time_base = av_inv_q(safeFps);
    impl_->codec_ctx->framerate = safeFps;
    impl_->codec_ctx->pix_fmt = AV_PIX_FMT_YUV420P; // Fallback to software encoding for now
    impl_->video_stream->time_base = impl_->codec_ctx->time_base;

    if (impl_->fmt_ctx->oformat->flags & AVFMT_GLOBALHEADER) {
        impl_->codec_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }

    if (avcodec_open2(impl_->codec_ctx, codec, nullptr) < 0) {
        error = "Failed to open codec";
        return false;
    }

    if (avcodec_parameters_from_context(impl_->video_stream->codecpar, impl_->codec_ctx) < 0) {
        error = "Failed to copy codec parameters";
        return false;
    }

    if (!(impl_->fmt_ctx->oformat->flags & AVFMT_NOFILE)) {
        if (avio_open(&impl_->fmt_ctx->pb, path.c_str(), AVIO_FLAG_WRITE) < 0) {
            error = "Could not open output file " + path;
            return false;
        }
    }

    if (avformat_write_header(impl_->fmt_ctx, nullptr) < 0) {
        error = "Error occurred when opening output file";
        return false;
    }
    impl_->header_written = true;

    impl_->sw_frame->format = impl_->codec_ctx->pix_fmt;
    impl_->sw_frame->width = impl_->codec_ctx->width;
    impl_->sw_frame->height = impl_->codec_ctx->height;
    if (av_frame_get_buffer(impl_->sw_frame, 0) < 0) {
        error = "Could not allocate frame data";
        return false;
    }

    return true;
}

bool VulkanVideoWriter::writeFrame(AVFrame* frame, std::string& error) {
    AVFrame* encode_frame = frame;

    // If frame is Vulkan hardware frame, transfer to software frame
    if (frame->format == AV_PIX_FMT_VULKAN) {
        if (av_hwframe_transfer_data(impl_->sw_frame, frame, 0) < 0) {
            error = "Error transferring frame from Vulkan";
            return false;
        }
        encode_frame = impl_->sw_frame;
    }

    encode_frame->pts = impl_->frame_count++;

    int ret = avcodec_send_frame(impl_->codec_ctx, encode_frame);
    if (ret < 0) {
        error = "Error sending a frame for encoding";
        return false;
    }

    while (ret >= 0) {
        ret = avcodec_receive_packet(impl_->codec_ctx, impl_->pkt);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            break;
        } else if (ret < 0) {
            error = "Error during encoding";
            return false;
        }

        av_packet_rescale_ts(impl_->pkt, impl_->codec_ctx->time_base, impl_->video_stream->time_base);
        impl_->pkt->stream_index = impl_->video_stream->index;

        ret = av_interleaved_write_frame(impl_->fmt_ctx, impl_->pkt);
        av_packet_unref(impl_->pkt);
        if (ret < 0) {
            error = "Error while writing output packet";
            return false;
        }
    }

    return true;
}

void VulkanVideoWriter::close() {
    if (impl_->codec_ctx != nullptr && impl_->fmt_ctx != nullptr && impl_->header_written) {
        avcodec_send_frame(impl_->codec_ctx, nullptr);
        while (true) {
            const int ret = avcodec_receive_packet(impl_->codec_ctx, impl_->pkt);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                break;
            }
            if (ret < 0) {
                break;
            }
            av_packet_rescale_ts(impl_->pkt, impl_->codec_ctx->time_base, impl_->video_stream->time_base);
            impl_->pkt->stream_index = impl_->video_stream->index;
            av_interleaved_write_frame(impl_->fmt_ctx, impl_->pkt);
            av_packet_unref(impl_->pkt);
        }
    }
    if (impl_->fmt_ctx) {
        if (impl_->header_written) {
            av_write_trailer(impl_->fmt_ctx);
        }
        if (!(impl_->fmt_ctx->oformat->flags & AVFMT_NOFILE)) {
            avio_closep(&impl_->fmt_ctx->pb);
        }
        avformat_free_context(impl_->fmt_ctx);
        impl_->fmt_ctx = nullptr;
    }
    if (impl_->codec_ctx) {
        avcodec_free_context(&impl_->codec_ctx);
        impl_->codec_ctx = nullptr;
    }
    if (impl_->pkt) {
        av_packet_free(&impl_->pkt);
        impl_->pkt = nullptr;
    }
    if (impl_->sw_frame) {
        av_frame_free(&impl_->sw_frame);
        impl_->sw_frame = nullptr;
    }
    impl_->video_stream = nullptr;
    impl_->frame_count = 0;
    impl_->header_written = false;
}

} // namespace frame_io
} // namespace ave
