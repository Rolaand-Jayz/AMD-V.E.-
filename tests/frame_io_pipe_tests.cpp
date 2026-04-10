#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "ave/frame_io.hpp"
#include "ave/video_probe.hpp"

namespace {

void check(const bool condition, const char* message) {
    if (condition) {
        return;
    }
    std::cerr << "frame_io_pipe_tests failed: " << message << '\n';
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
        std::filesystem::temp_directory_path() / "ave_frame_io_pipe_tests";
    std::error_code ec;
    std::filesystem::remove_all(tempDir, ec);
    std::filesystem::create_directories(tempDir, ec);
    const auto videoPath = tempDir / "sample.mp4";

    const std::string createVideoCommand =
        "ffmpeg -hide_banner -loglevel error -y "
        "-f lavfi -i testsrc=size=4x2:rate=2 "
        "-frames:v 2 -pix_fmt yuv420p \"" + videoPath.string() + "\"";
    check(commandSucceeds(createVideoCommand),
          "ffmpeg should create the temporary test clip");
    return videoPath;
}

void cleanupTestClip() {
    const auto tempDir =
        std::filesystem::temp_directory_path() / "ave_frame_io_pipe_tests";
    std::error_code ec;
    std::filesystem::remove_all(tempDir, ec);
}

void testRgbPipeSourceReadsFramesAndSignalsEndOfStream() {
    const auto videoPath = prepareTestClip();
    if (videoPath.empty()) {
        return;
    }

    ave::frame_io::RgbVideoPipeSource source;
    ave::frame_io::RgbVideoPipeSourceOptions options;
    std::string error;
    check(source.open(videoPath.string(), 4, 2, options, error),
          "RgbVideoPipeSource should open a generated clip");
    check(source.frameBytes() == 24u,
          "frame byte count should match width * height * 3");

    std::vector<std::uint8_t> frame;
    check(source.readFrame(frame, error) ==
              ave::frame_io::RgbVideoFrameReadStatus::FrameReady,
          "first frame should decode successfully");
    check(frame.size() == 24u, "decoded frame size should match the RGB frame size");

    check(source.readFrame(frame, error) ==
              ave::frame_io::RgbVideoFrameReadStatus::FrameReady,
          "second frame should decode successfully");
    check(frame.size() == 24u, "second decoded frame size should remain stable");

    check(source.readFrame(frame, error) ==
              ave::frame_io::RgbVideoFrameReadStatus::EndOfStream,
          "reader should report end-of-stream after the final frame");

    source.close();
    cleanupTestClip();
}

void testRgbFrameSourceDefaultsToRawPipeAndReadsFrames() {
    const auto videoPath = prepareTestClip();
    if (videoPath.empty()) {
        return;
    }

    ave::frame_io::RgbVideoFrameSource source;
    ave::frame_io::RgbVideoFrameSourceOptions options;
    std::string error;
    check(source.open(videoPath.string(), 4, 2, options, error),
          "RgbVideoFrameSource should open a generated clip");
    check(source.activeMode() == ave::frame_io::RgbVideoSourceMode::RawPipe,
          "default frame source mode should preserve raw pipe behavior");

    std::vector<std::uint8_t> frame;
    check(source.readFrame(frame, error) ==
              ave::frame_io::RgbVideoFrameReadStatus::FrameReady,
          "high-level frame source should decode the first frame");
    check(frame.size() == 24u, "high-level frame source should return RGB24 bytes");

    source.close();
    cleanupTestClip();
}

void testVideoFrameSourceDefaultsToRawPipeAndReturnsRgbPacket() {
    const auto videoPath = prepareTestClip();
    if (videoPath.empty()) {
        return;
    }

    ave::frame_io::VideoFrameSource source;
    ave::frame_io::RgbVideoFrameSourceOptions options;
    std::string error;
    check(source.open(videoPath.string(), 4, 2, options, error),
          "VideoFrameSource should open a generated clip");
    check(source.activeMode() == ave::frame_io::RgbVideoSourceMode::RawPipe,
          "VideoFrameSource should default to the raw pipe mode");

    ave::frame_io::VideoFramePacket frame;
    check(source.readFrame(frame, error) ==
              ave::frame_io::RgbVideoFrameReadStatus::FrameReady,
          "VideoFrameSource should decode the first frame");
    check(frame.hasRgb24(), "raw pipe packets should expose RGB24 bytes directly");
    check(!frame.hasFrame(), "raw pipe packets should not carry an AVFrame payload");
    check(frame.rgb24.size() == 24u, "raw pipe packets should expose the full RGB frame");

    source.close();
    cleanupTestClip();
}

void testAsyncRgbVideoPipeEncoderWritesProbeableOutput() {
    const auto videoPath = prepareTestClip();
    if (videoPath.empty()) {
        return;
    }

    const auto tempDir =
        std::filesystem::temp_directory_path() / "ave_frame_io_pipe_tests";
    const auto outputPath = tempDir / "encoded.mkv";
    const auto logPath = tempDir / "encoder.log";

    ave::frame_io::AsyncRgbVideoPipeEncoder encoder;
    ave::frame_io::RgbVideoPipeEncoderOptions options;
    options.inputVideo = videoPath.string();
    options.outputVideo = outputPath.string();
    options.width = 4;
    options.height = 2;
    options.fps = 2.0;
    options.directOutputEncode = false;
    options.queueDepth = 2u;
    options.stderrLogPath = logPath;

    std::string error;
    check(encoder.open(options, error),
          "AsyncRgbVideoPipeEncoder should open a raw RGB encode session");

    std::vector<std::uint8_t> frame;
    check(encoder.acquireBuffer(frame, error),
          "encoder should provide a free output buffer");
    check(frame.size() == 24u, "encoder buffer size should match the RGB frame size");
    std::fill(frame.begin(), frame.end(), 0u);
    check(encoder.submitFrame(0, std::move(frame), error),
          "encoder should accept the first frame");

    check(encoder.acquireBuffer(frame, error),
          "encoder should provide a second free output buffer");
    std::fill(frame.begin(), frame.end(), 255u);
    check(encoder.submitFrame(1, std::move(frame), error),
          "encoder should accept the second frame");

    check(encoder.finish(false) == 0,
          "encoder should finish without FFmpeg pipe errors");
    check(!encoder.writerFailed(),
          "encoder should not report asynchronous writer failures");
    check(std::filesystem::exists(outputPath),
          "encoder should create the output video file");

    const auto probe = ave::probeVideoStream(outputPath.string(), error);
    check(probe.has_value(), "encoded output should be probeable");
    check(probe->width == 4 && probe->height == 2,
          "encoded output should preserve the requested dimensions");
    check(probe->frameCount.has_value() && *probe->frameCount == 2,
          "encoded output should contain the submitted frame count");

    cleanupTestClip();
}

void testRgbVideoFrameWriterWritesProbeableOutput() {
    const auto videoPath = prepareTestClip();
    if (videoPath.empty()) {
        return;
    }

    const auto tempDir =
        std::filesystem::temp_directory_path() / "ave_frame_io_pipe_tests";
    const auto outputPath = tempDir / "writer_output.mkv";

    ave::frame_io::RgbVideoFrameWriter writer;
    std::string error;
    check(writer.open(outputPath.string(), 4, 2, AVRational{2, 1}, error),
          "RgbVideoFrameWriter should open an output video");

    std::vector<std::uint8_t> frame(24u, 0u);
    check(writer.writeFrame(frame.data(), error),
          "RgbVideoFrameWriter should accept the first RGB frame");

    std::fill(frame.begin(), frame.end(), 200u);
    check(writer.writeFrame(frame.data(), error),
          "RgbVideoFrameWriter should accept the second RGB frame");

    writer.close();

    check(std::filesystem::exists(outputPath),
          "RgbVideoFrameWriter should create an output video file");

    const auto probe = ave::probeVideoStream(outputPath.string(), error);
    check(probe.has_value(), "RgbVideoFrameWriter output should be probeable");
    check(probe->width == 4 && probe->height == 2,
          "RgbVideoFrameWriter output should preserve dimensions");
    check(probe->frameCount.has_value() && *probe->frameCount == 2,
          "RgbVideoFrameWriter output should contain the submitted frame count");

    cleanupTestClip();
}

void testRgbVideoSessionReadsAndWritesProbeableOutput() {
    const auto videoPath = prepareTestClip();
    if (videoPath.empty()) {
        return;
    }

    const auto tempDir =
        std::filesystem::temp_directory_path() / "ave_frame_io_pipe_tests";
    const auto outputPath = tempDir / "session_output.mkv";

    ave::frame_io::RgbVideoSession session;
    ave::frame_io::RgbVideoSessionOptions options;
    options.inputVideo = videoPath.string();
    options.outputVideo = outputPath.string();
    options.inputWidth = 4;
    options.inputHeight = 2;
    options.outputWidth = 4;
    options.outputHeight = 2;
    options.fps = AVRational{2, 1};

    std::string error;
    check(session.open(options, error),
          "RgbVideoSession should open a combined RGB decode/write session");
    check(session.activeSourceMode() == ave::frame_io::RgbVideoSourceMode::RawPipe,
          "RgbVideoSession should default to the raw pipe source mode");

    std::vector<std::uint8_t> frame;
    check(session.readFrame(frame, error) ==
              ave::frame_io::RgbVideoFrameReadStatus::FrameReady,
          "RgbVideoSession should decode the first frame");
    check(session.writeFrame(frame.data(), error),
          "RgbVideoSession should write the first decoded frame");

    check(session.readFrame(frame, error) ==
              ave::frame_io::RgbVideoFrameReadStatus::FrameReady,
          "RgbVideoSession should decode the second frame");
    check(session.writeFrame(frame.data(), error),
          "RgbVideoSession should write the second decoded frame");

    check(session.readFrame(frame, error) ==
              ave::frame_io::RgbVideoFrameReadStatus::EndOfStream,
          "RgbVideoSession should signal end-of-stream after the source frames");
    session.close();

    const auto probe = ave::probeVideoStream(outputPath.string(), error);
    check(probe.has_value(), "RgbVideoSession output should be probeable");
    check(probe->width == 4 && probe->height == 2,
          "RgbVideoSession output should preserve dimensions");
    check(probe->frameCount.has_value() && *probe->frameCount == 2,
          "RgbVideoSession output should preserve the decoded frame count");

    cleanupTestClip();
}

}  // namespace

int main() {
    testRgbPipeSourceReadsFramesAndSignalsEndOfStream();
    testRgbFrameSourceDefaultsToRawPipeAndReadsFrames();
    testVideoFrameSourceDefaultsToRawPipeAndReturnsRgbPacket();
    testAsyncRgbVideoPipeEncoderWritesProbeableOutput();
    testRgbVideoFrameWriterWritesProbeableOutput();
    testRgbVideoSessionReadsAndWritesProbeableOutput();
    return 0;
}
