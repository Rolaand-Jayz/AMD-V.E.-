#include <cstdlib>
#include <iostream>
#include <string>

#include "ave/ffmpeg_runner.hpp"
#include "ave/job.hpp"
#include "ave/types.hpp"

namespace {

using ave::EnhancementStage;
using ave::FfmpegRunner;
using ave::StageKind;
using ave::VideoJob;

void check(bool condition, const char* message) {
    if (condition) {
        return;
    }
    std::cerr << "ffmpeg_runner_tests failed: " << message << '\n';
    std::abort();
}

EnhancementStage stage(const StageKind kind) {
    EnhancementStage value;
    value.kind = kind;
    return value;
}

void testUnsafeEncodeTokensAreRejected() {
    FfmpegRunner runner;

    VideoJob job;
    job.inputPath = "input.mp4";
    job.outputPath = "output.mp4";
    job.encode.codec = "libx264;rm";

    std::string error;
    const bool ok = runner.encode(job, {}, nullptr, error);
    check(!ok, "unsafe ffmpeg codec tokens should be rejected before execution");
    check(error.find("Invalid FFmpeg codec token") != std::string::npos,
          "unsafe codec rejection should mention the codec token validation failure");
}

void testStereo3dRequiresAiBackend() {
    FfmpegRunner runner;
    std::string availabilityError;
    if (!runner.isAvailable(availabilityError)) {
        std::cout << "Skipping ffmpeg_runner_tests: " << availabilityError << '\n';
        return;
    }

    VideoJob job;
    job.inputPath = "missing-input.mp4";
    job.outputPath = "unused-output.mp4";

    std::string error;
    const bool ok = runner.encode(job, {stage(StageKind::Stereo3D)}, nullptr, error);
    check(!ok, "stereo_3d without an AI backend should fail");
    check(error.find("requires an AI backend with an available model") != std::string::npos,
          "stereo_3d failure should explain that FFmpeg fallback is unsupported");
}

}  // namespace

int main() {
    testUnsafeEncodeTokensAreRejected();
    testStereo3dRequiresAiBackend();
    return 0;
}
