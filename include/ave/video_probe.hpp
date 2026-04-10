#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace ave {

struct VideoStreamProbe {
    std::int64_t width = 0;
    std::int64_t height = 0;
    std::string avgFrameRate = "30/1";
    std::string nominalFrameRate = "30/1";
    std::optional<std::int64_t> frameCount;
    std::optional<double> durationSec;
    bool variableFrameRate = false;

    double effectiveFrameRate(double fallback = 30.0) const;
    std::string preferredEncodeFrameRate() const;
    std::int64_t estimatedFrameCount() const;
};

std::optional<double> parseFrameRateString(const std::string& value);
bool parseVideoStreamProbeOutput(const std::string& ffprobeOutput,
                                 VideoStreamProbe& probe,
                                 std::string& error);
std::optional<VideoStreamProbe> probeVideoStream(const std::string& inputPath,
                                                 std::string& error);

}  // namespace ave
