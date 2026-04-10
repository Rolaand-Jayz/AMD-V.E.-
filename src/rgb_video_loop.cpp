#include "ave/rgb_video_loop.hpp"

#include <iostream>
#include <string>

#include "ave/process_loop.hpp"

namespace ave {
namespace {

bool openDecodeSource(const RgbVideoLoopOptions& options,
                      frame_io::VideoFrameSource& decodeSource,
                      std::string& error,
                      RgbVideoSourceLoopResult& result) {
    frame_io::RgbVideoFrameSourceOptions decodeOptions;
    decodeOptions.preferredMode = options.preferredSourceMode;
    decodeOptions.allowFallback = options.allowSourceFallback;
    decodeOptions.pipe.durationSec = options.processOptions.previewDurationSec;
    decodeOptions.pipe.stdioBufferBytes = options.sourceStdioBufferBytes;
    decodeOptions.pipe.pipeBytes = options.sourcePipeBytes;
    if (!decodeSource.open(options.inputVideo,
                           options.inputWidth,
                           options.inputHeight,
                           decodeOptions,
                           error)) {
        result.stageResult = StageResult::Error;
        return false;
    }

    result.sourceOpened = true;
    result.activeSourceMode = decodeSource.activeMode();
    if (options.preferredSourceMode == frame_io::RgbVideoSourceMode::RawPipe) {
        return true;
    }

    if (result.activeSourceMode == options.preferredSourceMode) {
        if (result.activeSourceMode == frame_io::RgbVideoSourceMode::VulkanTransfer) {
            std::cout << '[' << options.backendTag
                      << "] using Vulkan-transfer frame source for host RGB staging."
                      << std::endl;
        } else if (result.activeSourceMode == frame_io::RgbVideoSourceMode::VulkanHardware) {
            std::cout << '[' << options.backendTag
                      << "] using preserved Vulkan hardware frames at the backend boundary."
                      << std::endl;
        }
    } else {
        std::cout << '[' << options.backendTag
                  << "] requested alternate frame source, but fell back to raw FFmpeg pipe decode."
                  << std::endl;
    }
    return true;
}

std::int64_t effectiveTotalFrameCount(const RgbVideoLoopOptions& options,
                                      const std::int64_t maxFrames) {
    return maxFrames > 0 ? maxFrames : options.totalFrames;
}

void maybeEmitLoopProgress(const RgbVideoLoopOptions& options,
                           const RgbVideoSourceLoopIteration& iteration,
                           const FrameProgressCb& progressCb,
                           const int frameIdx,
                           const std::int64_t effectiveTotal) {
    if (!progressCb) {
        return;
    }

    std::string progressMessage = iteration.progressMessage;
    if (progressMessage.empty()) {
        progressMessage = "frame " + std::to_string(frameIdx);
        if (effectiveTotal > 0) {
            progressMessage += "/" + std::to_string(effectiveTotal);
        }
    }
    progressCb(computeProcessingProgress(frameIdx, effectiveTotal),
               options.progressLabel + ": " + progressMessage);
}

void maybeEmitLoopPreview(const RgbVideoLoopOptions& options,
                          const RgbVideoSourceLoopIteration& iteration,
                          const int frameIdx) {
    if (iteration.previewRgb == nullptr || options.processOptions.framePreviewCb == nullptr ||
        iteration.previewWidth <= 0 || iteration.previewHeight <= 0) {
        return;
    }

    const int pvInterval = options.processOptions.previewFrameInterval > 0
        ? options.processOptions.previewFrameInterval
        : 15;
    if (pvInterval == 1 || frameIdx % pvInterval == 1) {
        options.processOptions.framePreviewCb(
            iteration.previewRgb, iteration.previewWidth, iteration.previewHeight);
    }
}

void maybeLogFrameCount(const RgbVideoLoopOptions& options, const int frameIdx) {
    if (frameIdx % 30 != 0) {
        return;
    }

    std::cout << '[' << options.backendTag << "] Processed " << frameIdx
              << " frames"
              << (options.totalFrames > 0
                      ? " / " + std::to_string(options.totalFrames)
                      : "")
              << std::endl;
}

}  // namespace

RgbVideoSourceLoopResult runVideoFrameSourceLoop(const RgbVideoLoopOptions& options,
                                                 const VideoFrameSourceProcessor& processor,
                                                 const FrameProgressCb& progressCb,
                                                 std::string& error) {
    RgbVideoSourceLoopResult result;
    if (!processor) {
        error = "Video frame source loop requires a frame processor callback.";
        result.stageResult = StageResult::Error;
        return result;
    }

    frame_io::VideoFrameSource decodeSource;
    if (!openDecodeSource(options, decodeSource, error, result)) {
        return result;
    }
    const auto inputFrameBytes = static_cast<std::size_t>(options.inputWidth) *
                                 static_cast<std::size_t>(options.inputHeight) * 3u;
    const auto maxFrames =
        computePreviewFrameLimit(options.processOptions.previewDurationSec,
                                 options.fps,
                                 options.fallbackFps);

    frame_io::VideoFramePacket inputFrame;
    inputFrame.rgb24.reserve(inputFrameBytes);
    bool cancelled = false;
    int frameIdx = 0;

    while (true) {
        if (maxFrames > 0 && static_cast<std::int64_t>(frameIdx) >= maxFrames) {
            break;
        }
        if (waitForProcessingResumeOrCancel(options.processOptions,
                                            options.backendTag,
                                            frameIdx)) {
            cancelled = true;
            break;
        }

        const auto readStart = std::chrono::steady_clock::now();
        const auto readStatus = decodeSource.readFrame(inputFrame, error);
        result.readTime += std::chrono::steady_clock::now() - readStart;
        if (readStatus == frame_io::RgbVideoFrameReadStatus::Error) {
            decodeSource.close();
            result.stageResult = StageResult::Error;
            return result;
        }
        if (readStatus == frame_io::RgbVideoFrameReadStatus::EndOfStream) {
            break;
        }

        RgbVideoSourceLoopIteration iteration;
        if (!processor(inputFrame, frameIdx, iteration, error)) {
            decodeSource.close();
            result.stageResult = StageResult::Error;
            return result;
        }

        ++frameIdx;
        const auto effectiveTotal = effectiveTotalFrameCount(options, maxFrames);
        maybeEmitLoopPreview(options, iteration, frameIdx);
        maybeEmitLoopProgress(options, iteration, progressCb, frameIdx, effectiveTotal);
        maybeLogFrameCount(options, frameIdx);
    }

    decodeSource.close();

    if (cancelled) {
        error = "Processing cancelled by user at frame " + std::to_string(frameIdx);
        result.stageResult = StageResult::Cancelled;
        return result;
    }
    if (frameIdx == 0) {
        error = !options.noFramesError.empty()
            ? options.noFramesError
            : "No frames decoded from " + options.inputVideo;
        result.stageResult = StageResult::Error;
        return result;
    }

    result.stageResult = StageResult::Processed;
    result.frameCount = frameIdx;
    return result;
}

RgbVideoSourceLoopResult runVideoFrameEncodeLoop(const RgbVideoLoopOptions& options,
                                                 frame_io::IRgbVideoFrameSink& sink,
                                                 const VideoFrameEncodeProcessor& processor,
                                                 const FrameProgressCb& progressCb,
                                                 std::string& error) {
    RgbVideoSourceLoopResult result;
    if (!processor) {
        error = "Video frame encode loop requires a frame processor callback.";
        result.stageResult = StageResult::Error;
        return result;
    }

    frame_io::VideoFrameSource decodeSource;
    if (!openDecodeSource(options, decodeSource, error, result)) {
        return result;
    }

    const auto maxFrames =
        computePreviewFrameLimit(options.processOptions.previewDurationSec,
                                 options.fps,
                                 options.fallbackFps);

    frame_io::VideoFramePacket inputFrame;
    std::vector<std::uint8_t> outputFrame;
    bool cancelled = false;
    int frameIdx = 0;

    while (true) {
        if (maxFrames > 0 && static_cast<std::int64_t>(frameIdx) >= maxFrames) {
            break;
        }
        if (waitForProcessingResumeOrCancel(options.processOptions,
                                            options.backendTag,
                                            frameIdx)) {
            cancelled = true;
            break;
        }

        const auto readStart = std::chrono::steady_clock::now();
        const auto readStatus = decodeSource.readFrame(inputFrame, error);
        result.readTime += std::chrono::steady_clock::now() - readStart;
        if (readStatus == frame_io::RgbVideoFrameReadStatus::Error) {
            decodeSource.close();
            result.stageResult = StageResult::Error;
            return result;
        }
        if (readStatus == frame_io::RgbVideoFrameReadStatus::EndOfStream) {
            break;
        }

        if (!sink.acquireBuffer(outputFrame, error)) {
            decodeSource.close();
            result.stageResult = StageResult::Error;
            return result;
        }

        RgbVideoSourceLoopIteration iteration;
        if (!processor(inputFrame, outputFrame, frameIdx, iteration, error)) {
            decodeSource.close();
            result.stageResult = StageResult::Error;
            return result;
        }

        const int completedFrame = frameIdx + 1;
        const auto effectiveTotal = effectiveTotalFrameCount(options, maxFrames);
        maybeEmitLoopPreview(options, iteration, completedFrame);

        if (!sink.submitFrame(frameIdx, std::move(outputFrame), error)) {
            decodeSource.close();
            result.stageResult = StageResult::Error;
            return result;
        }

        frameIdx = completedFrame;
        maybeEmitLoopProgress(options, iteration, progressCb, frameIdx, effectiveTotal);
        maybeLogFrameCount(options, frameIdx);
    }

    decodeSource.close();

    if (cancelled) {
        error = "Processing cancelled by user at frame " + std::to_string(frameIdx);
        result.stageResult = StageResult::Cancelled;
        return result;
    }
    if (frameIdx == 0) {
        error = !options.noFramesError.empty()
            ? options.noFramesError
            : "No frames decoded from " + options.inputVideo;
        result.stageResult = StageResult::Error;
        return result;
    }

    result.stageResult = StageResult::Processed;
    result.frameCount = frameIdx;
    return result;
}

RgbVideoSourceLoopResult runRgbVideoSourceLoop(const RgbVideoLoopOptions& options,
                                               const RgbVideoSourceProcessor& processor,
                                               const FrameProgressCb& progressCb,
                                               std::string& error) {
    if (!processor) {
        error = "RGB video source loop requires a frame processor callback.";
        return RgbVideoSourceLoopResult{StageResult::Error, 0};
    }

    frame_io::VideoFramePacketMaterializer materializer;
    return runVideoFrameSourceLoop(
        options,
        [&](const frame_io::VideoFramePacket& inputFrame,
            const int frameIdx,
            RgbVideoSourceLoopIteration& iteration,
            std::string& loopError) {
            const std::vector<std::uint8_t>* rgbFrame = nullptr;
            if (!materializer.resolveRgb24(inputFrame, rgbFrame, loopError) || rgbFrame == nullptr) {
                return false;
            }
            return processor(*rgbFrame, frameIdx, iteration, loopError);
        },
        progressCb,
        error);
}

RgbVideoLoopResult runRgbVideoLoop(const RgbVideoLoopOptions& options,
                                   const RgbVideoFrameProcessor& processor,
                                   const FrameProgressCb& progressCb,
                                   std::string& error) {
    RgbVideoLoopResult result;
    if (!processor) {
        error = "RGB video loop requires a frame processor callback.";
        result.stageResult = StageResult::Error;
        return result;
    }

    frame_io::RgbVideoFrameWriter writer;
    if (!writer.open(options.outputVideo,
                     options.outputWidth,
                     options.outputHeight,
                     AVRational{
                         static_cast<int>(
                             options.fps > 0.0 ? options.fps + 0.5 : options.fallbackFps + 0.5),
                         1},
                     error)) {
        result.stageResult = StageResult::Error;
        return result;
    }

    const auto outputFrameBytes = static_cast<std::size_t>(options.outputWidth) *
                                  static_cast<std::size_t>(options.outputHeight) * 3u;
    frame_io::SyncRgbVideoFrameSink sink(writer, outputFrameBytes);
    frame_io::VideoFramePacketMaterializer materializer;

    const auto sourceResult = runVideoFrameEncodeLoop(
        options,
        sink,
        [&](const frame_io::VideoFramePacket& inputFrame,
            std::vector<std::uint8_t>& rgbOut,
            const int frameIdx,
            RgbVideoSourceLoopIteration& iteration,
            std::string& loopError) {
            const std::vector<std::uint8_t>* rgbFrame = nullptr;
            if (!materializer.resolveRgb24(inputFrame, rgbFrame, loopError) || rgbFrame == nullptr) {
                return false;
            }
            if (rgbOut.size() != outputFrameBytes) {
                rgbOut.resize(outputFrameBytes);
            }
            if (!processor(*rgbFrame, rgbOut, frameIdx, loopError)) {
                return false;
            }
            iteration.previewRgb = rgbOut.data();
            iteration.previewWidth = options.outputWidth;
            iteration.previewHeight = options.outputHeight;
            return true;
        },
        progressCb,
        error);

    writer.close();

    result.stageResult = sourceResult.stageResult;
    result.frameCount = sourceResult.frameCount;
    if (result.stageResult == StageResult::Processed && progressCb) {
        progressCb(1.0f,
                   options.progressLabel + " complete — "
                       + std::to_string(result.frameCount) + " frames.");
    }
    return result;
}

}  // namespace ave
