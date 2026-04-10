#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "ave/rgb_video_loop.hpp"
#include "ave/video_probe.hpp"

namespace {

void check(const bool condition, const char* message) {
    if (condition) {
        return;
    }
    std::cerr << "rgb_video_loop_tests failed: " << message << '\n';
    std::abort();
}

bool commandSucceeds(const std::string& command) {
    return std::system(command.c_str()) == 0;
}

std::filesystem::path prepareTestClip() {
    if (!commandSucceeds("command -v ffmpeg >/dev/null 2>&1")) {
        return {};
    }

    const auto tempDir =
        std::filesystem::temp_directory_path() / "ave_rgb_video_loop_tests";
    std::error_code ec;
    std::filesystem::remove_all(tempDir, ec);
    std::filesystem::create_directories(tempDir, ec);
    const auto videoPath = tempDir / "sample.mp4";

    const std::string createVideoCommand =
        "ffmpeg -hide_banner -loglevel error -y "
        "-f lavfi -i testsrc=size=4x2:rate=2 "
        "-frames:v 2 -pix_fmt yuv420p \"" + videoPath.string() + "\"";
    check(commandSucceeds(createVideoCommand),
          "ffmpeg should create the RGB video loop test clip");
    return videoPath;
}

void cleanupTestClip() {
    const auto tempDir =
        std::filesystem::temp_directory_path() / "ave_rgb_video_loop_tests";
    std::error_code ec;
    std::filesystem::remove_all(tempDir, ec);
}

void testRgbVideoLoopRoundTrip() {
    const auto inputPath = prepareTestClip();
    if (inputPath.empty()) {
        return;
    }

    const auto tempDir =
        std::filesystem::temp_directory_path() / "ave_rgb_video_loop_tests";
    const auto outputPath = tempDir / "loop_output.mkv";

    ave::RgbVideoLoopOptions options;
    options.inputVideo = inputPath.string();
    options.outputVideo = outputPath.string();
    options.inputWidth = 4;
    options.inputHeight = 2;
    options.outputWidth = 4;
    options.outputHeight = 2;
    options.fps = 2.0;
    options.backendTag = "rgb-loop-test";
    options.progressLabel = "RGB loop";

    std::string error;
    const auto result = ave::runRgbVideoLoop(
        options,
        [](const std::vector<std::uint8_t>& inputRgb,
           std::vector<std::uint8_t>& outputRgb,
           int /*frameIdx*/,
           std::string& /*error*/) {
            outputRgb = inputRgb;
            return true;
        },
        {},
        error);

    check(result.stageResult == ave::StageResult::Processed,
          "RGB video loop should process a simple identity transform");
    check(result.frameCount == 2,
          "RGB video loop should report the processed frame count");

    const auto probe = ave::probeVideoStream(outputPath.string(), error);
    check(probe.has_value(), "RGB video loop output should be probeable");
    check(probe->width == 4 && probe->height == 2,
          "RGB video loop output should preserve dimensions");
    check(probe->frameCount.has_value() && *probe->frameCount == 2,
          "RGB video loop output should preserve the frame count");

    cleanupTestClip();
}

void testRgbVideoSourceLoopCountsFrames() {
    const auto inputPath = prepareTestClip();
    if (inputPath.empty()) {
        return;
    }

    ave::RgbVideoLoopOptions options;
    options.inputVideo = inputPath.string();
    options.inputWidth = 4;
    options.inputHeight = 2;
    options.outputWidth = 4;
    options.outputHeight = 2;
    options.fps = 2.0;
    options.backendTag = "rgb-source-loop-test";
    options.progressLabel = "RGB source loop";

    int seenFrames = 0;
    std::string error;
    const auto result = ave::runRgbVideoSourceLoop(
        options,
        [&](const std::vector<std::uint8_t>& inputRgb,
            const int frameIdx,
            ave::RgbVideoSourceLoopIteration& iteration,
            std::string& /*loopError*/) {
            check(inputRgb.size() == 24, "source loop should deliver full RGB frames");
            check(frameIdx == seenFrames, "source loop frame index should be sequential");
            iteration.progressMessage = "counted frame " + std::to_string(frameIdx + 1);
            ++seenFrames;
            return true;
        },
        {},
        error);

    check(result.stageResult == ave::StageResult::Processed,
          "RGB video source loop should process a simple counting callback");
    check(result.frameCount == 2,
          "RGB video source loop should report the processed frame count");
    check(seenFrames == 2,
          "RGB video source loop should invoke the callback for every decoded frame");
    check(result.readTime.count() >= 0,
          "RGB video source loop should expose accumulated read timing");

    cleanupTestClip();
}

void testVideoFrameSourceLoopCountsFramesWithPacketInput() {
    const auto inputPath = prepareTestClip();
    if (inputPath.empty()) {
        return;
    }

    ave::RgbVideoLoopOptions options;
    options.inputVideo = inputPath.string();
    options.inputWidth = 4;
    options.inputHeight = 2;
    options.outputWidth = 4;
    options.outputHeight = 2;
    options.fps = 2.0;
    options.backendTag = "video-frame-source-loop-test";
    options.progressLabel = "Video frame source loop";

    int seenFrames = 0;
    std::string error;
    const auto result = ave::runVideoFrameSourceLoop(
        options,
        [&](const ave::frame_io::VideoFramePacket& inputFrame,
            const int frameIdx,
            ave::RgbVideoSourceLoopIteration& iteration,
            std::string& /*loopError*/) {
            check(inputFrame.hasRgb24(),
                  "raw pipe packet loop should expose embedded RGB bytes");
            check(!inputFrame.hasFrame(),
                  "raw pipe packet loop should not expose an AVFrame payload");
            check(inputFrame.rgb24.size() == 24u,
                  "packet loop should expose the full RGB frame");
            check(frameIdx == seenFrames, "packet loop frame index should be sequential");
            iteration.progressMessage = "packet frame " + std::to_string(frameIdx + 1);
            ++seenFrames;
            return true;
        },
        {},
        error);

    check(result.stageResult == ave::StageResult::Processed,
          "video frame source loop should process a simple counting callback");
    check(result.frameCount == 2,
          "video frame source loop should report the processed frame count");
    check(seenFrames == 2,
          "video frame source loop should invoke the callback for every decoded frame");

    cleanupTestClip();
}

void testVideoFrameEncodeLoopWritesProbeableOutput() {
    const auto inputPath = prepareTestClip();
    if (inputPath.empty()) {
        return;
    }

    const auto tempDir =
        std::filesystem::temp_directory_path() / "ave_rgb_video_loop_tests";
    const auto outputPath = tempDir / "packet_encode_output.mkv";

    ave::RgbVideoLoopOptions options;
    options.inputVideo = inputPath.string();
    options.outputVideo = outputPath.string();
    options.inputWidth = 4;
    options.inputHeight = 2;
    options.outputWidth = 4;
    options.outputHeight = 2;
    options.fps = 2.0;
    options.backendTag = "video-frame-encode-loop-test";
    options.progressLabel = "Video frame encode loop";

    ave::frame_io::RgbVideoFrameWriter writer;
    std::string error;
    check(writer.open(outputPath.string(), 4, 2, AVRational{2, 1}, error),
          "packet encode loop writer should open an output video");
    ave::frame_io::SyncRgbVideoFrameSink sink(writer, 24u);

    const auto result = ave::runVideoFrameEncodeLoop(
        options,
        sink,
        [](const ave::frame_io::VideoFramePacket& inputFrame,
           std::vector<std::uint8_t>& outputRgb,
           int /*frameIdx*/,
           ave::RgbVideoSourceLoopIteration& iteration,
           std::string& /*loopError*/) {
            check(inputFrame.hasRgb24(),
                  "packet encode loop should see embedded RGB bytes on the raw pipe path");
            outputRgb = inputFrame.rgb24;
            iteration.previewRgb = outputRgb.data();
            iteration.previewWidth = 4;
            iteration.previewHeight = 2;
            return true;
        },
        {},
        error);

    writer.close();

    check(result.stageResult == ave::StageResult::Processed,
          "video frame encode loop should complete a simple packet identity encode");
    check(result.frameCount == 2,
          "video frame encode loop should report the processed frame count");

    const auto probe = ave::probeVideoStream(outputPath.string(), error);
    check(probe.has_value(), "packet encode loop output should be probeable");
    check(probe->width == 4 && probe->height == 2,
          "packet encode loop output should preserve dimensions");
    check(probe->frameCount.has_value() && *probe->frameCount == 2,
          "packet encode loop output should preserve the frame count");

    cleanupTestClip();
}

}  // namespace

int main() {
    testRgbVideoLoopRoundTrip();
    testRgbVideoSourceLoopCountsFrames();
    testVideoFrameSourceLoopCountsFramesWithPacketInput();
    testVideoFrameEncodeLoopWritesProbeableOutput();
    return 0;
}
