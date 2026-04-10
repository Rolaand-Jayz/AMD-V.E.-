#include "ave/video_probe.hpp"

#include <array>
#include <cmath>
#include <sstream>
#include <string>
#include <unordered_map>

#include "ave/process_observer.hpp"

namespace ave {
namespace {

bool frameRatesDifferMeaningfully(const std::optional<double> lhs,
                                  const std::optional<double> rhs) {
    if (!lhs.has_value() || !rhs.has_value()) {
        return false;
    }
    return std::abs(*lhs - *rhs) > 0.01;
}

}  // namespace

std::optional<double> parseFrameRateString(const std::string& value) {
    const std::string cleaned = process_observer::trimOutput(value);
    if (cleaned.empty() || cleaned == "N/A") {
        return std::nullopt;
    }

    try {
        const auto slash = cleaned.find('/');
        if (slash != std::string::npos) {
            const double numerator = std::stod(cleaned.substr(0, slash));
            const double denominator = std::stod(cleaned.substr(slash + 1));
            if (denominator == 0.0) {
                return std::nullopt;
            }
            return numerator / denominator;
        }
        return std::stod(cleaned);
    } catch (...) {
        return std::nullopt;
    }
}

bool parseVideoStreamProbeOutput(const std::string& ffprobeOutput,
                                 VideoStreamProbe& probe,
                                 std::string& error) {
    std::unordered_map<std::string, std::string> values;
    std::istringstream input(ffprobeOutput);
    for (std::string line; std::getline(input, line);) {
        const std::string cleaned = process_observer::trimOutput(line);
        if (cleaned.empty()) {
            continue;
        }
        const auto eq = cleaned.find('=');
        if (eq == std::string::npos || eq == 0) {
            continue;
        }
        values[cleaned.substr(0, eq)] = cleaned.substr(eq + 1);
    }

    try {
        probe.width = std::stoll(values.at("width"));
        probe.height = std::stoll(values.at("height"));
    } catch (...) {
        error = "Unable to parse input resolution from ffprobe output.";
        return false;
    }
    if (probe.width <= 0 || probe.height <= 0) {
        error = "ffprobe returned invalid input dimensions.";
        return false;
    }

    if (const auto it = values.find("avg_frame_rate"); it != values.end() && !it->second.empty()) {
        probe.avgFrameRate = it->second;
    }
    if (const auto it = values.find("r_frame_rate"); it != values.end() && !it->second.empty()) {
        probe.nominalFrameRate = it->second;
    }
    auto parseFrameCountField = [&](const char* key) {
        const auto it = values.find(key);
        if (it == values.end()) {
            return;
        }
        try {
            if (!it->second.empty() && it->second != "N/A") {
                probe.frameCount = std::stoll(it->second);
            }
        } catch (...) {
            if (!probe.frameCount.has_value()) {
                probe.frameCount = std::nullopt;
            }
        }
    };
    parseFrameCountField("nb_frames");
    if (!probe.frameCount.has_value()) {
        parseFrameCountField("nb_read_frames");
    }
    if (const auto it = values.find("duration"); it != values.end()) {
        try {
            if (!it->second.empty() && it->second != "N/A") {
                probe.durationSec = std::stod(it->second);
            }
        } catch (...) {
            probe.durationSec = std::nullopt;
        }
    }

    probe.variableFrameRate = frameRatesDifferMeaningfully(
        parseFrameRateString(probe.avgFrameRate),
        parseFrameRateString(probe.nominalFrameRate));
    error.clear();
    return true;
}

std::optional<VideoStreamProbe> probeVideoStream(const std::string& inputPath,
                                                 std::string& error) {
    const std::string cmd =
        "ffprobe -v error -count_frames -select_streams v:0 "
        "-show_entries stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,nb_read_frames:format=duration "
        "-of default=noprint_wrappers=1:nokey=0 "
        + process_observer::quoteShellArg(inputPath);

    int exitCode = 0;
    const auto output = process_observer::captureCommandStdout(cmd, exitCode);
    if (!output.has_value() || exitCode != 0) {
        error = "ffprobe failed while inspecting input video.";
        return std::nullopt;
    }

    VideoStreamProbe probe;
    if (!parseVideoStreamProbeOutput(*output, probe, error)) {
        return std::nullopt;
    }
    return probe;
}

std::string VideoStreamProbe::preferredEncodeFrameRate() const {
    if (variableFrameRate) {
        return {};
    }
    if (!avgFrameRate.empty() && avgFrameRate != "N/A") {
        return avgFrameRate;
    }
    if (!nominalFrameRate.empty() && nominalFrameRate != "N/A") {
        return nominalFrameRate;
    }
    return {};
}

double VideoStreamProbe::effectiveFrameRate(const double fallback) const {
    if (const auto avg = parseFrameRateString(avgFrameRate);
        avg.has_value() && *avg > 0.0) {
        return *avg;
    }
    if (const auto nominal = parseFrameRateString(nominalFrameRate);
        nominal.has_value() && *nominal > 0.0) {
        return *nominal;
    }
    return fallback;
}

std::int64_t VideoStreamProbe::estimatedFrameCount() const {
    if (frameCount.has_value() && *frameCount > 0) {
        return *frameCount;
    }
    const double fps = effectiveFrameRate(0.0);
    if (durationSec.has_value() && fps > 0.0) {
        return static_cast<std::int64_t>(*durationSec * fps + 0.5);
    }
    return 0;
}

}  // namespace ave
