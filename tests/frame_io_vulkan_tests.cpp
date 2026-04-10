#include <cstdlib>
#include <iostream>

#include "ave/frame_io.hpp"
#include "ave/vulkan_runtime.hpp"

namespace {

void check(const bool condition, const char* message) {
    if (condition) {
        return;
    }
    std::cerr << "frame_io_vulkan_tests failed: " << message << '\n';
    std::abort();
}

void testHardwareFrameDetection() {
    check(!ave::frame_io::isVulkanHardwareFrame(nullptr),
          "null frames must not be treated as Vulkan hardware frames");

    AVFrame* softwareFrame = av_frame_alloc();
    check(softwareFrame != nullptr, "software frame allocation should succeed");
    softwareFrame->format = AV_PIX_FMT_RGB24;
    check(!ave::frame_io::isVulkanHardwareFrame(softwareFrame),
          "software frames must not be treated as Vulkan hardware frames");
    av_frame_free(&softwareFrame);

    AVFrame* hardwareFrame = av_frame_alloc();
    check(hardwareFrame != nullptr, "hardware frame allocation should succeed");
    hardwareFrame->format = AV_PIX_FMT_VULKAN;
    check(ave::frame_io::isVulkanHardwareFrame(hardwareFrame),
          "AV_PIX_FMT_VULKAN frames must be detected as hardware frames");
    av_frame_free(&hardwareFrame);
}

void testReadModeEnumValuesRemainDistinct() {
    check(ave::frame_io::VulkanFrameReadMode::TransferToHost !=
              ave::frame_io::VulkanFrameReadMode::PreserveHardware,
          "Vulkan frame read modes must remain distinct");
}

void testVideoFramePacketRgbHelperUsesEmbeddedRgbBytes() {
    ave::frame_io::VideoFramePacket packet;
    packet.width = 1;
    packet.height = 1;
    packet.sourceMode = ave::frame_io::RgbVideoSourceMode::RawPipe;
    packet.rgb24 = {1u, 2u, 3u};

    check(packet.hasRgb24(), "packet should report embedded RGB bytes");
    check(!packet.hasFrame(), "packet should not report an AVFrame when unset");
    check(!packet.isHardwareFrame(), "RGB-only packets must not report hardware frames");

    std::vector<std::uint8_t> rgb;
    std::string error;
    check(ave::frame_io::videoFramePacketToRgb24(packet, rgb, error),
          "packet RGB helper should succeed when RGB bytes are already embedded");
    check(rgb == packet.rgb24,
          "packet RGB helper should return the embedded RGB bytes unchanged");
}

void testPacketMaterializerReusesEmbeddedRgbBytes() {
    ave::frame_io::VideoFramePacket packet;
    packet.width = 1;
    packet.height = 1;
    packet.rgb24 = {9u, 8u, 7u};

    ave::frame_io::VideoFramePacketMaterializer materializer;
    const std::vector<std::uint8_t>* rgbView = nullptr;
    std::string error;
    check(materializer.resolveRgb24(packet, rgbView, error),
          "materializer should resolve embedded RGB bytes");
    check(rgbView == &packet.rgb24,
          "materializer should reuse embedded RGB bytes without copying");
    check(materializer.lastSoftwareFrameResolveMode() ==
              ave::frame_io::SoftwareFrameResolveMode::None,
          "RGB-only packet resolution should not report a software-frame access mode");
}

void testPacketMaterializerResolvesSoftwareRgbFrame() {
    AVFrame* frame = av_frame_alloc();
    check(frame != nullptr, "software RGB frame allocation should succeed");
    frame->format = AV_PIX_FMT_RGB24;
    frame->width = 2;
    frame->height = 1;
    check(av_frame_get_buffer(frame, 1) >= 0,
          "software RGB frame buffer allocation should succeed");
    frame->data[0][0] = 10u;
    frame->data[0][1] = 20u;
    frame->data[0][2] = 30u;
    frame->data[0][3] = 40u;
    frame->data[0][4] = 50u;
    frame->data[0][5] = 60u;

    ave::frame_io::VideoFramePacket packet;
    packet.width = 2;
    packet.height = 1;
    packet.frame.reset(frame);

    ave::frame_io::VideoFramePacketMaterializer materializer;
    AVFrame* resolvedFrame = nullptr;
    std::string error;
    check(materializer.resolveSoftwareFrame(packet, resolvedFrame, error),
          "materializer should resolve software AVFrame payloads directly");
    check(resolvedFrame == packet.frame.get(),
          "software AVFrame resolution should reuse the packet frame");
    check(materializer.lastSoftwareFrameResolveMode() ==
              ave::frame_io::SoftwareFrameResolveMode::PacketFrame,
          "software AVFrame resolution should report direct packet-frame reuse");

    const std::vector<std::uint8_t>* rgbView = nullptr;
    check(materializer.resolveRgb24(packet, rgbView, error),
          "materializer should convert software AVFrame payloads to RGB bytes");
    check(rgbView != nullptr, "materializer should return an RGB view for software AVFrames");
    check(rgbView->size() == 6u, "resolved RGB view should have the expected byte count");
    check((*rgbView)[0] == 10u && (*rgbView)[1] == 20u && (*rgbView)[2] == 30u
              && (*rgbView)[3] == 40u && (*rgbView)[4] == 50u && (*rgbView)[5] == 60u,
          "resolved RGB view should preserve software RGB frame bytes");
}

void testVulkanHardwareFrameStagerRejectsInvalidInputs() {
    ave::frame_io::VulkanHardwareFrameStager stager;
    ave::VulkanRuntime runtime;
    AVFrame* resolvedFrame = nullptr;
    std::string error;

    check(!stager.stageToSoftwareFrame(nullptr, runtime, resolvedFrame, error),
          "staging should reject null frames");
    check(!error.empty(), "null-frame staging failures should explain the problem");

    error.clear();
    AVFrame* softwareFrame = av_frame_alloc();
    check(softwareFrame != nullptr, "software frame allocation should succeed");
    softwareFrame->format = AV_PIX_FMT_RGB24;
    check(!stager.stageToSoftwareFrame(softwareFrame, runtime, resolvedFrame, error),
          "staging should reject non-Vulkan frames");
    check(!error.empty(), "non-Vulkan staging failures should explain the problem");
    av_frame_free(&softwareFrame);
}

}  // namespace

int main() {
    testHardwareFrameDetection();
    testReadModeEnumValuesRemainDistinct();
    testVideoFramePacketRgbHelperUsesEmbeddedRgbBytes();
    testPacketMaterializerReusesEmbeddedRgbBytes();
    testPacketMaterializerResolvesSoftwareRgbFrame();
    testVulkanHardwareFrameStagerRejectsInvalidInputs();
    return 0;
}
