#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "ave/backend.hpp"
#include "ave/frame_io.hpp"

namespace ave {

struct RgbVideoLoopOptions {
    std::string inputVideo;
    std::string outputVideo;
    int inputWidth = 0;
    int inputHeight = 0;
    int outputWidth = 0;
    int outputHeight = 0;
    double fps = 0.0;
    double fallbackFps = 30.0;
    std::int64_t totalFrames = 0;
    std::string backendTag;
    std::string progressLabel;
    std::string noFramesError;
    frame_io::RgbVideoSourceMode preferredSourceMode = frame_io::RgbVideoSourceMode::RawPipe;
    bool allowSourceFallback = true;
    std::size_t sourceStdioBufferBytes = 0u;
    int sourcePipeBytes = 0;
    ProcessVideoOptions processOptions;
};

struct RgbVideoLoopResult {
    StageResult stageResult = StageResult::Processed;
    int frameCount = 0;
};

struct RgbVideoSourceLoopIteration {
    const std::uint8_t* previewRgb = nullptr;
    int previewWidth = 0;
    int previewHeight = 0;
    std::string progressMessage;
};

struct RgbVideoSourceLoopResult {
    StageResult stageResult = StageResult::Processed;
    int frameCount = 0;
    std::chrono::nanoseconds readTime{0};
    bool sourceOpened = false;
    frame_io::RgbVideoSourceMode activeSourceMode = frame_io::RgbVideoSourceMode::RawPipe;
};

using RgbVideoSourceProcessor = std::function<bool(const std::vector<std::uint8_t>& inputRgb,
                                                   int frameIdx,
                                                   RgbVideoSourceLoopIteration& iteration,
                                                   std::string& error)>;

using VideoFrameSourceProcessor = std::function<bool(
    const frame_io::VideoFramePacket& inputFrame,
    int frameIdx,
    RgbVideoSourceLoopIteration& iteration,
    std::string& error)>;

using VideoFrameEncodeProcessor = std::function<bool(
    const frame_io::VideoFramePacket& inputFrame,
    std::vector<std::uint8_t>& outputRgb,
    int frameIdx,
    RgbVideoSourceLoopIteration& iteration,
    std::string& error)>;

using RgbVideoFrameProcessor = std::function<bool(const std::vector<std::uint8_t>& inputRgb,
                                                  std::vector<std::uint8_t>& outputRgb,
                                                  int frameIdx,
                                                  std::string& error)>;

RgbVideoSourceLoopResult runRgbVideoSourceLoop(const RgbVideoLoopOptions& options,
                                               const RgbVideoSourceProcessor& processor,
                                               const FrameProgressCb& progressCb,
                                               std::string& error);

RgbVideoSourceLoopResult runVideoFrameSourceLoop(const RgbVideoLoopOptions& options,
                                                 const VideoFrameSourceProcessor& processor,
                                                 const FrameProgressCb& progressCb,
                                                 std::string& error);

RgbVideoSourceLoopResult runVideoFrameEncodeLoop(const RgbVideoLoopOptions& options,
                                                 frame_io::IRgbVideoFrameSink& sink,
                                                 const VideoFrameEncodeProcessor& processor,
                                                 const FrameProgressCb& progressCb,
                                                 std::string& error);

RgbVideoLoopResult runRgbVideoLoop(const RgbVideoLoopOptions& options,
                                   const RgbVideoFrameProcessor& processor,
                                   const FrameProgressCb& progressCb,
                                   std::string& error);

}  // namespace ave
