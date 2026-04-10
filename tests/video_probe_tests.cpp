#include <cstdlib>
#include <iostream>

#include "ave/video_probe.hpp"

namespace {

void check(const bool condition, const char* message) {
    if (condition) {
        return;
    }
    std::cerr << "video_probe_tests failed: " << message << '\n';
    std::abort();
}

void testConstantFrameRateProbeParsing() {
    const std::string output =
        "width=1920\n"
        "height=1080\n"
        "avg_frame_rate=24000/1001\n"
        "r_frame_rate=24000/1001\n"
        "nb_frames=240\n"
        "duration=10.0\n";

    ave::VideoStreamProbe probe;
    std::string error;
    check(ave::parseVideoStreamProbeOutput(output, probe, error),
          "constant frame rate probe output should parse");
    check(!probe.variableFrameRate, "matching avg/r frame rates should not mark VFR");
    check(probe.preferredEncodeFrameRate() == "24000/1001",
          "constant rate probe should preserve the avg frame rate");
    check(probe.estimatedFrameCount() == 240,
          "probe should prefer explicit frame count when present");
}

void testVariableFrameRateProbeParsing() {
    const std::string output =
        "width=3840\n"
        "height=2160\n"
        "avg_frame_rate=24000/1001\n"
        "r_frame_rate=30000/1001\n"
        "duration=5.0\n";

    ave::VideoStreamProbe probe;
    std::string error;
    check(ave::parseVideoStreamProbeOutput(output, probe, error),
          "variable frame rate probe output should parse");
    check(probe.variableFrameRate, "differing avg/r frame rates should mark VFR");
    check(probe.preferredEncodeFrameRate().empty(),
          "VFR probe should not force an encode frame rate");
    check(probe.estimatedFrameCount() > 0,
          "duration plus avg frame rate should estimate frame count");
}

void testCountedFrameFallbackParsing() {
    const std::string output =
        "width=4\n"
        "height=2\n"
        "avg_frame_rate=2/1\n"
        "r_frame_rate=2/1\n"
        "nb_frames=N/A\n"
        "nb_read_frames=2\n"
        "duration=N/A\n";

    ave::VideoStreamProbe probe;
    std::string error;
    check(ave::parseVideoStreamProbeOutput(output, probe, error),
          "counted frame fallback probe output should parse");
    check(probe.frameCount.has_value() && *probe.frameCount == 2,
          "probe should fall back to counted frames when nb_frames is unavailable");
    check(probe.estimatedFrameCount() == 2,
          "estimated frame count should use the counted-frame fallback");
}

void testEffectiveFrameRateFallbacks() {
    ave::VideoStreamProbe probe;
    probe.avgFrameRate = "N/A";
    probe.nominalFrameRate = "60000/1001";
    check(probe.effectiveFrameRate() > 59.9 && probe.effectiveFrameRate() < 60.0,
          "effective frame rate should fall back to the nominal rate");

    probe.nominalFrameRate = "N/A";
    check(probe.effectiveFrameRate(48.0) == 48.0,
          "effective frame rate should fall back to the caller default");
}

void testMalformedProbeOutputFailsCleanly() {
    ave::VideoStreamProbe probe;
    std::string error;
    check(!ave::parseVideoStreamProbeOutput("avg_frame_rate=24000/1001\n", probe, error),
          "missing width/height should fail parsing");
    check(error.find("resolution") != std::string::npos,
          "malformed probe failure should mention resolution parsing");
}

}  // namespace

int main() {
    testConstantFrameRateProbeParsing();
    testVariableFrameRateProbeParsing();
    testCountedFrameFallbackParsing();
    testEffectiveFrameRateFallbacks();
    testMalformedProbeOutputFailsCleanly();
    return 0;
}
