#include "ave/ffmpeg_runner.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#ifdef __unix__
#  include <unistd.h>
#endif

#include "ave/backend.hpp"
#include "ave/stage.hpp"
#include "ave/types.hpp"

namespace ave {
namespace {

struct VideoProbeInfo {
    std::int64_t width = 0;
    std::int64_t height = 0;
    std::string fps = "30";
};

struct TempDirectoryGuard {
    std::filesystem::path path;

    ~TempDirectoryGuard() {
        if (path.empty()) {
            return;
        }
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
};

bool commandInPath(const std::string& command) {
    const char* pathEnv = std::getenv("PATH");
    if (pathEnv == nullptr) {
        return false;
    }

    const std::string path(pathEnv);
    std::size_t start = 0;
    while (start <= path.size()) {
        std::size_t end = path.find(':', start);
        if (end == std::string::npos) {
            end = path.size();
        }
        const std::string dir = path.substr(start, end - start);
        if (!dir.empty()) {
            std::error_code ec;
            const auto candidate = std::filesystem::path(dir) / command;
            if (std::filesystem::exists(candidate, ec)) {
                return true;
            }
        }
        if (end == path.size()) {
            break;
        }
        start = end + 1;
    }
    return false;
}

std::string quoteArg(const std::string& value) {
    std::string out = "\"";
    for (const char ch : value) {
        if (ch == '\\' || ch == '"') {
            out.push_back('\\');
        }
        out.push_back(ch);
    }
    out.push_back('"');
    return out;
}

std::string trim(const std::string& input) {
    std::size_t begin = 0;
    while (begin < input.size() && std::isspace(static_cast<unsigned char>(input[begin])) != 0) {
        ++begin;
    }

    std::size_t end = input.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(input[end - 1])) != 0) {
        --end;
    }

    return input.substr(begin, end - begin);
}

std::string toLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

std::optional<std::string> captureCommandStdout(const std::string& cmd, int& exitCode) {
    FILE* pipe = popen(cmd.c_str(), "r");
    if (pipe == nullptr) {
        exitCode = -1;
        return std::nullopt;
    }

    std::string output;
    std::array<char, 512> buffer{};
    while (true) {
        const std::size_t bytes = std::fread(buffer.data(), sizeof(char), buffer.size(), pipe);
        if (bytes == 0) {
            break;
        }
        output.append(buffer.data(), bytes);
    }

    exitCode = pclose(pipe);
    return output;
}

// ─── Progress-aware command runners ──────────────────────────────

// Return video frame count via fast metadata lookup (no packet scan).
std::int64_t countFrames(const std::string& inputPath) {
    // Try nb_frames tag first (instant, stored in container header)
    {
        const std::string cmd =
            "ffprobe -v error -select_streams v:0 "
            "-show_entries stream=nb_frames -of csv=p=0 " +
            quoteArg(inputPath) + " 2>/dev/null";
        int ec = 0;
        const auto out = captureCommandStdout(cmd, ec);
        if (out.has_value()) {
            const std::string s = trim(*out);
            if (!s.empty() && s != "N/A")
                try { return std::stoll(s); } catch (...) {}
        }
    }
    // Fallback: duration x frame rate (still instant, no decode)
    {
        const std::string cmd =
            "ffprobe -v error -select_streams v:0 "
            "-show_entries stream=duration,avg_frame_rate -of csv=p=0 " +
            quoteArg(inputPath) + " 2>/dev/null";
        int ec = 0;
        const auto out = captureCommandStdout(cmd, ec);
        if (out.has_value()) {
            std::istringstream ss(*out);
            std::string durStr, fpsStr;
            std::getline(ss, durStr, ',');
            std::getline(ss, fpsStr);
            durStr = trim(durStr); fpsStr = trim(fpsStr);
            try {
                if (!durStr.empty() && durStr != "N/A") {
                    const double dur = std::stod(durStr);
                    double fps = 30.0;
                    const auto sl = fpsStr.find('/');
                    if (sl != std::string::npos) {
                        const double n = std::stod(fpsStr.substr(0, sl));
                        const double d = std::stod(fpsStr.substr(sl + 1));
                        if (d > 0.0) fps = n / d;
                    } else if (!fpsStr.empty() && fpsStr != "N/A") {
                        fps = std::stod(fpsStr);
                    }
                    if (fps > 0.0)
                        return static_cast<std::int64_t>(dur * fps + 0.5);
                }
            } catch (...) {}
        }
    }
    return 0; // unknown; progress bar will still show indeterminate
}

// Run an ffmpeg command that includes "-progress -" (writes key=value
// progress to stdout). stderr is captured to a temp file so full error
// details are preserved and surfaced when the command fails.
bool runFfmpegWithProgress(
        const std::string& cmd,
        const std::string& stepName,
        std::int64_t totalFrames,
        const std::function<void(float, const std::string&)>& taskCb,
        std::string& error) {
    // Capture stderr to a temp file so we can report the actual error.
    const std::string errTmp = "/tmp/ave_ffmpeg_"
        + std::to_string(static_cast<long>(getpid())) + ".err";
    const std::string fullCmd = cmd + " 2>" + quoteArg(errTmp);
    std::cout << "[cmd] " << stepName << ": " << cmd << std::endl;

    FILE* pipe = popen(fullCmd.c_str(), "r");
    if (!pipe) {
        error = stepName + " could not be started (popen failed).";
        return false;
    }
    char buf[512];
    std::int64_t lastReportedFrame = -100;
    constexpr std::int64_t kReportInterval = 5;
    while (fgets(buf, sizeof(buf), pipe)) {
        std::string line = trim(std::string(buf));
        if (line.rfind("frame=", 0) == 0) {
            try {
                const std::int64_t frame = std::stoll(line.substr(6));
                if (totalFrames > 0 && (frame - lastReportedFrame) >= kReportInterval) {
                    lastReportedFrame = frame;
                    const float frac = std::min(1.0f,
                        static_cast<float>(frame) / static_cast<float>(totalFrames));
                    if (taskCb)
                        taskCb(frac, stepName + " \u2013 frame " +
                               std::to_string(frame) + "/" + std::to_string(totalFrames));
                }
            } catch (...) {}
        } else if (line == "progress=end") {
            if (taskCb) taskCb(1.0f, stepName + " \u2013 complete");
        }
    }
    const int ret = pclose(pipe);

    // Read captured stderr for diagnostics.
    std::string ffmpegStderr;
    {
        std::ifstream efile(errTmp);
        ffmpegStderr.assign((std::istreambuf_iterator<char>(efile)),
                             std::istreambuf_iterator<char>());
        ffmpegStderr = trim(ffmpegStderr);
    }
    std::error_code rmEc;
    std::filesystem::remove(errTmp, rmEc);

    if (ret != 0) {
        error = stepName + " failed";
        if (!ffmpegStderr.empty())
            error += ": " + ffmpegStderr;
        else
            error += " (exit " + std::to_string(ret) + ", no stderr captured).";
        return false;
    }
    if (!ffmpegStderr.empty())
        std::cerr << "[ffmpeg-warn] " << stepName << ": " << ffmpegStderr << std::endl;
    return true;
}

// Run realesrgan-ncnn-vulkan, parse "N.NN%" percentage lines from combined stdout+stderr.
// All non-progress lines are collected so that silent GPU/Vulkan failures are
// visible to the user rather than silently producing blank output frames.
bool runRealEsrganWithProgress(
        const std::string& baseCmd,
        const std::function<void(float, const std::string&)>& taskCb,
        std::string& error) {
    const std::string cmd = baseCmd + " 2>&1";
    std::cout << "[cmd] ai-upscale: " << baseCmd << std::endl;
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        error = "ai-upscale failed to start (popen failed).";
        return false;
    }
    char buf[512];
    float lastPct = -2.0f;
    std::vector<std::string> diagnosticLines;
    while (fgets(buf, sizeof(buf), pipe)) {
        std::string line = trim(std::string(buf));
        if (line.empty()) continue;

        // Progress lines end with '%', e.g. "12.34%"
        if (line.back() == '%') {
            try {
                const float pct = std::stof(line.substr(0, line.size() - 1));
                if (pct - lastPct >= 1.0f) {
                    lastPct = pct;
                    if (taskCb) taskCb(pct / 100.0f, "AI upscale \u2013 " + line);
                }
                continue;
            } catch (...) {}
        }

        // Capture non-progress output for diagnostics (limit to avoid OOM).
        if (diagnosticLines.size() < 50) {
            diagnosticLines.push_back(line);
        }
        std::cerr << "[realesrgan] " << line << '\n';
    }
    const int ret = pclose(pipe);
    if (ret != 0) {
        error = "ai-upscale failed (exit " + std::to_string(ret) + ")";
        if (!diagnosticLines.empty()) {
            error += ":\n";
            for (const auto& dl : diagnosticLines) error += "  " + dl + "\n";
        } else {
            error += " — no output captured.";
        }
        return false;
    }
    // Exited 0 but no progress seen — warn so the user can check GPU access.
    if (lastPct < 0.0f && !diagnosticLines.empty())
        std::cerr << "[realesrgan] WARNING: tool exited 0 but no progress lines seen; "
                     "check GPU/Vulkan access.\n";
    return true;
}

std::optional<VideoProbeInfo> probeVideo(const std::string& inputPath, std::string& error) {
    const std::string cmd =
        "ffprobe -v error -select_streams v:0 -show_entries stream=width,height,avg_frame_rate "
        "-of default=noprint_wrappers=1:nokey=1 " +
        quoteArg(inputPath);

    int exitCode = 0;
    const std::optional<std::string> output = captureCommandStdout(cmd, exitCode);
    if (!output.has_value() || exitCode != 0) {
        error = "ffprobe failed while inspecting input video.";
        return std::nullopt;
    }

    std::istringstream input(*output);
    std::vector<std::string> lines;
    for (std::string line; std::getline(input, line);) {
        const std::string cleaned = trim(line);
        if (!cleaned.empty()) {
            lines.push_back(cleaned);
        }
    }

    if (lines.size() < 3) {
        error = "ffprobe returned incomplete stream metadata.";
        return std::nullopt;
    }

    VideoProbeInfo info;
    try {
        info.width = std::stoll(lines[0]);
        info.height = std::stoll(lines[1]);
    } catch (...) {
        error = "Unable to parse input resolution from ffprobe output.";
        return std::nullopt;
    }

    info.fps = lines[2];
    if (info.fps.empty()) {
        info.fps = "30";
    }

    return info;
}

double clampDouble(const double value, const double minValue, const double maxValue) {
    if (value < minValue) {
        return minValue;
    }
    if (value > maxValue) {
        return maxValue;
    }
    return value;
}

std::int64_t clampInt(const std::int64_t value, const std::int64_t minValue, const std::int64_t maxValue) {
    if (value < minValue) {
        return minValue;
    }
    if (value > maxValue) {
        return maxValue;
    }
    return value;
}

std::string formatDouble(const double value) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(6) << value;
    return os.str();
}

double getDoubleParam(const EnhancementStage& stage, const std::string& key, const double defaultValue) {
    double value = defaultValue;
    if (!tryGetDouble(stage.params, key, value)) {
        return defaultValue;
    }
    if (!std::isfinite(value)) {
        return defaultValue;
    }
    return value;
}

std::optional<double> getOptionalDoubleParam(const EnhancementStage& stage, const std::string& key) {
    double value = 0.0;
    if (!tryGetDouble(stage.params, key, value) || !std::isfinite(value)) {
        return std::nullopt;
    }
    return value;
}

std::optional<std::int64_t> getOptionalIntParam(const EnhancementStage& stage, const std::string& key) {
    std::int64_t value = 0;
    if (!tryGetInt(stage.params, key, value)) {
        return std::nullopt;
    }
    return value;
}

std::optional<std::string> getOptionalStringParam(const EnhancementStage& stage, const std::string& key) {
    const auto it = stage.params.find(key);
    if (it == stage.params.end()) {
        return std::nullopt;
    }

    if (const auto* value = std::get_if<std::string>(&it->second)) {
        return *value;
    }
    if (const auto* value = std::get_if<std::int64_t>(&it->second)) {
        return std::to_string(*value);
    }
    if (const auto* value = std::get_if<double>(&it->second)) {
        return formatDouble(*value);
    }
    if (const auto* value = std::get_if<bool>(&it->second)) {
        return *value ? "true" : "false";
    }

    return std::nullopt;
}

std::optional<bool> getOptionalBoolParam(const EnhancementStage& stage, const std::string& key) {
    const auto it = stage.params.find(key);
    if (it == stage.params.end()) {
        return std::nullopt;
    }

    if (const auto* value = std::get_if<bool>(&it->second)) {
        return *value;
    }

    if (const auto* value = std::get_if<std::int64_t>(&it->second)) {
        return *value != 0;
    }

    if (const auto* value = std::get_if<double>(&it->second)) {
        return *value != 0.0;
    }

    if (const auto* value = std::get_if<std::string>(&it->second)) {
        const std::string normalized = toLower(trim(*value));
        if (normalized == "true" || normalized == "yes" || normalized == "on" || normalized == "1") {
            return true;
        }
        if (normalized == "false" || normalized == "no" || normalized == "off" || normalized == "0") {
            return false;
        }
    }

    return std::nullopt;
}

std::string joinFilters(const std::vector<std::string>& filters) {
    std::ostringstream chain;
    for (std::size_t i = 0; i < filters.size(); ++i) {
        if (i != 0) {
            chain << ',';
        }
        chain << filters[i];
    }
    return chain.str();
}

std::string buildEqFilter(const EnhancementStage& stage) {
    const double strength = clampDouble(getDoubleParam(stage, "strength", 0.5), 0.0, 2.0);
    const double contrast =
        clampDouble(getOptionalDoubleParam(stage, "contrast").value_or(1.0 + (0.06 * strength)), 0.1, 3.0);
    const double brightness = clampDouble(getOptionalDoubleParam(stage, "brightness").value_or(0.0), -1.0, 1.0);
    const double saturation =
        clampDouble(getOptionalDoubleParam(stage, "saturation").value_or(1.0 + (0.08 * strength)), 0.0, 3.0);
    const double gamma =
        clampDouble(getOptionalDoubleParam(stage, "gamma").value_or(1.0 + (0.02 * strength)), 0.1, 10.0);

    std::ostringstream os;
    os << "eq=contrast=" << formatDouble(contrast) << ":brightness=" << formatDouble(brightness)
       << ":saturation=" << formatDouble(saturation) << ":gamma=" << formatDouble(gamma);
    return os.str();
}

std::optional<std::string> buildVibranceFilter(const EnhancementStage& stage) {
    const double strength = clampDouble(getDoubleParam(stage, "strength", 0.5), 0.0, 2.0);
    const double intensity = clampDouble(getOptionalDoubleParam(stage, "vibrance").value_or(0.06 * strength), -2.0, 2.0);
    if (std::abs(intensity) < std::numeric_limits<double>::epsilon()) {
        return std::nullopt;
    }
    return "vibrance=intensity=" + formatDouble(intensity);
}

void appendFiltersForStage(const EnhancementStage& stage, std::vector<std::string>& filters, const bool includeUpscale) {
    // If the accelerator backend already performed AI inference for
    // this stage, do NOT add FFmpeg filters — they would either
    // duplicate the enhancement or overwrite the AI result with a
    // weaker signal-processing approximation.
    if (stage.backendProcessed) {
        return;
    }

    switch (stage.kind) {
        case StageKind::RestoreCompression: {
            const double strength = clampDouble(getDoubleParam(stage, "strength", 0.8), 0.0, 2.0);
            const double alpha = clampDouble(0.045 + (0.035 * strength), 0.005, 0.30);
            const double beta = clampDouble(0.030 + (0.020 * strength), 0.005, 0.30);
            const double gamma = clampDouble(0.020 + (0.025 * strength), 0.005, 0.30);
            const double delta = clampDouble(0.020 + (0.025 * strength), 0.005, 0.30);
            const double debandThr = clampDouble(0.010 + (0.012 * strength), 0.00003, 0.5);
            const std::int64_t range = clampInt(static_cast<std::int64_t>(std::llround(14.0 + (8.0 * strength))), 4, 64);
            const double lumaSpatial = clampDouble(0.6 + (1.8 * strength), 0.0, 32.0);
            const double chromaSpatial = clampDouble(0.4 + (1.2 * strength), 0.0, 32.0);
            const double lumaTmp = clampDouble(1.0 + (2.5 * strength), 0.0, 64.0);
            const double chromaTmp = clampDouble(0.8 + (1.8 * strength), 0.0, 64.0);

            filters.push_back("deblock=filter=strong:block=8:alpha=" + formatDouble(alpha) + ":beta=" + formatDouble(beta) +
                              ":gamma=" + formatDouble(gamma) + ":delta=" + formatDouble(delta));
            filters.push_back("deband=1thr=" + formatDouble(debandThr) + ":2thr=" + formatDouble(debandThr) +
                              ":3thr=" + formatDouble(debandThr) + ":range=" + std::to_string(range));
            filters.push_back("hqdn3d=luma_spatial=" + formatDouble(lumaSpatial) + ":chroma_spatial=" +
                              formatDouble(chromaSpatial) + ":luma_tmp=" + formatDouble(lumaTmp) +
                              ":chroma_tmp=" + formatDouble(chromaTmp));
            break;
        }
        case StageKind::RemoveArtifacts: {
            const double strength = clampDouble(getDoubleParam(stage, "strength", 0.7), 0.0, 2.0);
            const bool strong = strength >= 0.9;
            const double alpha = clampDouble(0.030 + (0.030 * strength), 0.005, 0.30);
            const double beta = clampDouble(0.020 + (0.020 * strength), 0.005, 0.30);
            const double gamma = clampDouble(0.018 + (0.018 * strength), 0.005, 0.30);
            const double delta = clampDouble(0.018 + (0.018 * strength), 0.005, 0.30);
            const double debandThr = clampDouble(0.008 + (0.012 * strength), 0.00003, 0.5);
            const std::int64_t range = clampInt(static_cast<std::int64_t>(std::llround(10.0 + (8.0 * strength))), 4, 64);

            filters.push_back(std::string("deblock=filter=") + (strong ? "strong" : "weak") + ":block=8:alpha=" +
                              formatDouble(alpha) + ":beta=" + formatDouble(beta) + ":gamma=" + formatDouble(gamma) +
                              ":delta=" + formatDouble(delta));
            filters.push_back("deband=1thr=" + formatDouble(debandThr) + ":2thr=" + formatDouble(debandThr) +
                              ":3thr=" + formatDouble(debandThr) + ":range=" + std::to_string(range));
            break;
        }
        case StageKind::Denoise: {
            const double strength = clampDouble(getDoubleParam(stage, "strength", 0.6), 0.0, 2.0);
            const double lumaSpatial = clampDouble(0.3 + (1.8 * strength), 0.0, 32.0);
            const double chromaSpatial = clampDouble(0.2 + (1.2 * strength), 0.0, 32.0);
            const double lumaTmp = clampDouble(0.8 + (3.0 * strength), 0.0, 64.0);
            const double chromaTmp = clampDouble(0.5 + (2.0 * strength), 0.0, 64.0);

            filters.push_back("hqdn3d=luma_spatial=" + formatDouble(lumaSpatial) + ":chroma_spatial=" +
                              formatDouble(chromaSpatial) + ":luma_tmp=" + formatDouble(lumaTmp) +
                              ":chroma_tmp=" + formatDouble(chromaTmp));
            break;
        }
        case StageKind::Deblur: {
            const double strength = clampDouble(getDoubleParam(stage, "strength", 0.55), 0.0, 2.0);
            const double casStrength = clampDouble(0.14 + (0.28 * strength), 0.0, 1.0);
            const double unsharpAmount = clampDouble(0.20 + (0.35 * strength), 0.0, 2.0);

            filters.push_back("cas=strength=" + formatDouble(casStrength));
            filters.push_back("unsharp=5:5:" + formatDouble(unsharpAmount) + ":3:3:0");
            break;
        }
        case StageKind::Dehalo: {
            const double strength = clampDouble(getDoubleParam(stage, "strength", 0.5), 0.0, 2.0);
            const double radius = clampDouble(0.5 + (0.7 * strength), 0.1, 5.0);
            const double blurStrength = clampDouble(0.18 + (0.24 * strength), -1.0, 1.0);
            const double casStrength = clampDouble(0.05 + (0.08 * strength), 0.0, 1.0);

            filters.push_back("smartblur=lr=" + formatDouble(radius) + ":ls=" + formatDouble(blurStrength) + ":lt=0");
            filters.push_back("cas=strength=" + formatDouble(casStrength));
            break;
        }
        case StageKind::ColorFix: {
            filters.push_back(buildEqFilter(stage));
            const std::optional<std::string> vibrance = buildVibranceFilter(stage);
            if (vibrance.has_value()) {
                filters.push_back(*vibrance);
            }
            break;
        }
        case StageKind::Upscale: {
            if (includeUpscale) {
                std::int64_t width = 0;
                std::int64_t height = 0;
                if (tryGetInt(stage.params, "width", width) && tryGetInt(stage.params, "height", height) && width > 0 &&
                    height > 0) {
                    filters.push_back("scale=" + std::to_string(width) + ":" + std::to_string(height));
                }
            }
            break;
        }
        case StageKind::Sharpen: {
            const double amount = clampDouble(getDoubleParam(stage, "amount", 0.6), 0.0, 2.0);
            const double casStrength = clampDouble(amount * 0.35, 0.0, 1.0);
            filters.push_back("cas=strength=" + formatDouble(casStrength));
            filters.push_back("unsharp=5:5:" + formatDouble(amount) + ":3:3:0");
            break;
        }
        case StageKind::Interpolate: {
            const double fps = clampDouble(getDoubleParam(stage, "fps", 60.0), 1.0, 240.0);
            std::ostringstream os;
            os << "minterpolate=fps=" << formatDouble(fps) << ":mi_mode=mci:mc_mode=aobmc:me_mode=bilat";
            filters.push_back(os.str());
            break;
        }
    }
}

bool shouldUseAiUpscale(const EnhancementStage& stage) {
    if (stage.kind != StageKind::Upscale) {
        return false;
    }

    const std::string engine = toLower(trim(getOptionalStringParam(stage, "engine").value_or("ai")));
    return !(engine == "ffmpeg" || engine == "basic" || engine == "none");
}

// Returns true if the stage has a "model_path" parameter injected by
// VideoProcessor::resolveModelPath, meaning a model file is actually
// available on disk for this stage.
bool hasModelPath(const EnhancementStage& stage) {
    const auto it = stage.params.find("model_path");
    if (it == stage.params.end()) return false;
    if (const auto* s = std::get_if<std::string>(&it->second))
        return !s->empty();
    return false;
}

// Returns true if the stage should be processed via AI inference
// (either via the backend's processFrameDir or via realesrgan).
bool isAiProcessable(const EnhancementStage& stage,
                     const IAcceleratorBackend* backend) {
    if (shouldUseAiUpscale(stage)) return true;
    if (backend && hasModelPath(stage) && !stage.backendProcessed) return true;
    return false;
}

// Counts regular PNG files in a directory whose size is >= minBytes.
std::size_t countValidFrames(const std::filesystem::path& dir,
                              std::uintmax_t minBytes = 1024) {
    std::error_code ec;
    std::size_t n = 0;
    for (const auto& e : std::filesystem::directory_iterator(dir, ec)) {
        if (ec || !e.is_regular_file()) continue;
        std::error_code szEc;
        if (e.file_size(szEc) >= minBytes && !szEc) ++n;
    }
    return n;
}

// Sort all PNG files in dir by name and rename them to %08d.png starting at 0.
// Handles any naming convention realesrgan uses and fills gaps caused by a
// partial run (so FFmpeg can always read a contiguous %08d sequence).
bool renumberFramesSequentially(const std::filesystem::path& dir, std::string& error) {
    std::error_code ec;
    std::vector<std::filesystem::path> files;
    for (const auto& e : std::filesystem::directory_iterator(dir, ec)) {
        if (!ec && e.is_regular_file() && e.path().extension() == ".png")
            files.push_back(e.path());
    }
    if (files.empty()) return true;
    std::sort(files.begin(), files.end());

    // Check whether renaming is actually needed.
    bool needsRename = false;
    for (std::size_t i = 0; i < files.size(); ++i) {
        char expected[32];
        snprintf(expected, sizeof(expected), "%08zu.png", i);
        if (files[i].filename().string() != expected) { needsRename = true; break; }
    }
    if (!needsRename) return true;

    // Two-pass rename: first to a guaranteed-unique temp name, then to final.
    for (std::size_t i = 0; i < files.size(); ++i) {
        std::filesystem::rename(files[i], dir / ("__t" + std::to_string(i) + ".png"), ec);
        if (ec) { error = "Failed to stage-rename AI frame: " + files[i].filename().string(); return false; }
    }
    for (std::size_t i = 0; i < files.size(); ++i) {
        char final_name[32];
        snprintf(final_name, sizeof(final_name), "%08zu.png", i);
        std::filesystem::rename(dir / ("__t" + std::to_string(i) + ".png"), dir / final_name, ec);
        if (ec) { error = std::string("Failed to final-rename AI frame to ") + final_name; return false; }
    }
    std::cout << "[ai] renamed " << files.size() << " frames to sequential %08d.png" << std::endl;
    return true;
}

std::filesystem::path createTempJobDir(const std::filesystem::path& outputPath) {
    const auto base = std::filesystem::path(outputPath).parent_path();
    const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
    std::filesystem::path path = base / ("ave_ai_job_" + std::to_string(tick));
    std::filesystem::create_directories(path);
    return path;
}

std::int64_t pickAiScale(const EnhancementStage& upscaleStage,
                         const std::optional<VideoProbeInfo>& probe,
                         const std::optional<std::int64_t>& targetWidth,
                         const std::optional<std::int64_t>& targetHeight) {
    const std::optional<std::int64_t> explicitScale = getOptionalIntParam(upscaleStage, "ai_scale");
    if (explicitScale.has_value()) {
        return clampInt(*explicitScale, 2, 4);
    }

    if (probe.has_value() && targetWidth.has_value() && targetHeight.has_value() && probe->width > 0 && probe->height > 0) {
        const double ratioW = static_cast<double>(*targetWidth) / static_cast<double>(probe->width);
        const double ratioH = static_cast<double>(*targetHeight) / static_cast<double>(probe->height);
        const double ratio = std::max(ratioW, ratioH);

        if (ratio <= 2.2) {
            return 2;
        }
        if (ratio <= 3.3) {
            return 3;
        }
        return 4;
    }

    return 4;
}

#if 0  // Superseded by encodeWithAiProcessing — kept for reference.

// Scales a single PNG to 1×1 and returns the average grey value (0–255).
// Returns -1 if ffmpeg is unavailable or the frame can't be read.
int sampleFrameLuma(const std::filesystem::path& framePath) {
    const std::string cmd =
        "ffmpeg -hide_banner -loglevel quiet -i " + quoteArg(framePath.string()) +
        " -vf \"scale=1:1,format=gray\" -frames:v 1 -f rawvideo pipe:1 2>/dev/null";
    FILE* p = popen(cmd.c_str(), "r");
    if (!p) return -1;
    unsigned char pixel = 0;
    [[maybe_unused]] const auto n = fread(&pixel, 1, 1, p);
    pclose(p);
    return static_cast<int>(pixel);
}

bool encodeWithAiUpscale(const VideoJob& job,
                         const std::vector<EnhancementStage>& orderedStages,
                         const std::size_t aiUpscaleIdx,
                         std::string& error) {
    if (!commandInPath("realesrgan-ncnn-vulkan")) {
        error = "AI upscale requested but realesrgan-ncnn-vulkan is not available in PATH.";
        return false;
    }

    std::string probeError;
    const std::optional<VideoProbeInfo> probe = probeVideo(job.inputPath, probeError);
    if (!probe.has_value()) {
        error = probeError;
        return false;
    }

    const EnhancementStage& upscaleStage = orderedStages[aiUpscaleIdx];

    std::int64_t width = 0;
    std::int64_t height = 0;
    const std::optional<std::int64_t> targetWidth =
        (tryGetInt(upscaleStage.params, "width", width) && width > 0) ? std::optional<std::int64_t>{width} : std::nullopt;
    const std::optional<std::int64_t> targetHeight =
        (tryGetInt(upscaleStage.params, "height", height) && height > 0) ? std::optional<std::int64_t>{height} : std::nullopt;

    const std::int64_t aiScale = pickAiScale(upscaleStage, probe, targetWidth, targetHeight);

    const std::string modelName = getOptionalStringParam(upscaleStage, "model").value_or("realesr-animevideov3");

    std::string modelDir = getOptionalStringParam(upscaleStage, "model_dir").value_or("");
    if (modelDir.empty()) {
        const std::filesystem::path defaultModelDir = "/usr/share/realesrgan-ncnn-vulkan/models";
        std::error_code ec;
        if (std::filesystem::exists(defaultModelDir, ec)) {
            modelDir = defaultModelDir.string();
        }
    }

    std::cout << "[ai] using realesrgan-ncnn-vulkan model=" << modelName << " scale=" << aiScale << std::endl;

    TempDirectoryGuard tempGuard;
    tempGuard.path = createTempJobDir(job.outputPath);

    const std::filesystem::path framesInputDir = tempGuard.path / "frames_input";
    const std::filesystem::path framesAiDir = tempGuard.path / "frames_ai";

    std::error_code ec;
    std::filesystem::create_directories(framesInputDir, ec);
    if (ec) {
        error = "Failed to create temporary input frame directory.";
        return false;
    }
    std::filesystem::create_directories(framesAiDir, ec);
    if (ec) {
        error = "Failed to create temporary AI frame directory.";
        return false;
    }

    std::vector<std::string> preFilters;
    std::vector<std::string> postFilters;

    for (std::size_t i = 0; i < orderedStages.size(); ++i) {
        if (i < aiUpscaleIdx) {
            appendFiltersForStage(orderedStages[i], preFilters, false);
            continue;
        }

        if (i == aiUpscaleIdx) {
            continue;
        }

        appendFiltersForStage(orderedStages[i], postFilters, true);
    }

    if (targetWidth.has_value() && targetHeight.has_value()) {
        postFilters.insert(postFilters.begin(), "scale=" + std::to_string(*targetWidth) + ":" + std::to_string(*targetHeight));
    }

    const auto& progressCb = job.progressCb;
    auto taskReport = [&progressCb](float frac, const std::string& msg) {
        if (progressCb) progressCb(0, static_cast<int>(frac * 100.0f + 0.5f), msg);
    };
    auto extractTaskCb = [&taskReport](float f, const std::string& msg) {
        taskReport(f * 0.25f, msg);         // 0–25 % of task bar
    };
    auto aiTaskCb = [&taskReport](float f, const std::string& msg) {
        taskReport(0.25f + f * 0.50f, msg); // 25–75 %
    };
    auto encodeTaskCb = [&taskReport](float f, const std::string& msg) {
        taskReport(0.75f + f * 0.25f, msg); // 75–100 %
    };

    if (progressCb) progressCb(0, 0, "Probing input video for frame count\u2026");
    const std::int64_t totalInputFrames = countFrames(job.inputPath);
    if (!preFilters.empty()) {
        const std::string preChain = joinFilters(preFilters);
        if (progressCb) progressCb(0, 0,
            std::to_string(preFilters.size()) + " pre-upscale filter(s): " + preChain);
        std::cout << "[ai] pre-upscale filters: " << preChain << std::endl;
    } else {
        if (progressCb) progressCb(0, 0, "No pre-upscale filters.");
    }

    const std::string inputPattern = (framesInputDir / "%08d.png").string();
    std::ostringstream extractCmd;
    extractCmd << "ffmpeg -y -hide_banner -loglevel error -progress - "
               << "-i " << quoteArg(job.inputPath) << ' ';
    if (!preFilters.empty()) {
        extractCmd << "-vf " << quoteArg(joinFilters(preFilters)) << ' ';
    }
    // Force RGB24 so realesrgan-ncnn-vulkan always receives a plain RGB PNG
    // (avoids alpha-channel or YUV-packed PNGs that confuse the tool).
    extractCmd << "-pix_fmt rgb24 -vsync 0 -start_number 0 " << quoteArg(inputPattern);

    if (!runFfmpegWithProgress(extractCmd.str(), "extract-frames", totalInputFrames, extractTaskCb, error)) {
        return false;
    }

    std::ostringstream aiCmd;
    aiCmd << "realesrgan-ncnn-vulkan"
          << " -i " << quoteArg(framesInputDir.string())
          << " -o " << quoteArg(framesAiDir.string())
          << " -s " << aiScale
          << " -n " << modelName
          << " -f png";

    if (!modelDir.empty()) {
        aiCmd << " -m " << quoteArg(modelDir);
    }

    if (const std::optional<std::int64_t> gpu = getOptionalIntParam(upscaleStage, "gpu_id"); gpu.has_value()) {
        aiCmd << " -g " << *gpu;
    }

    if (const std::optional<std::int64_t> tile = getOptionalIntParam(upscaleStage, "tile"); tile.has_value()) {
        aiCmd << " -t " << clampInt(*tile, 0, 4096);
    } else {
        // Default tile size: process frame in 256-px tiles so the entire super-
        // resolution frame is never loaded into GPU VRAM at once.  Without this
        // realesrgan easily OOMs on AMD GPUs, exits 0 and writes black PNG files
        // for unfinished frames — producing the "93 % hang + black output" bug.
        aiCmd << " -t 256";
    }

    if (const std::optional<std::string> jobs = getOptionalStringParam(upscaleStage, "jobs"); jobs.has_value() && !jobs->empty()) {
        aiCmd << " -j " << *jobs;
    }

    if (getOptionalBoolParam(upscaleStage, "tta").value_or(false)) {
        aiCmd << " -x";
    }

    if (progressCb) progressCb(0, 25, "Running AI upscale model " + modelName +
        " ×" + std::to_string(aiScale) + " on " + std::to_string(totalInputFrames) + " frames…");
    if (!runRealEsrganWithProgress(aiCmd.str(), aiTaskCb, error)) {
        return false;
    }

    // ── Post-realesrgan validation ──────────────────────────────────────
    // 1. Renumber output frames sequentially so FFmpeg always gets a clean
    //    %08d.png sequence regardless of how realesrgan names its output.
    {
        std::string renameErr;
        if (!renumberFramesSequentially(framesAiDir, renameErr)) {
            error = "Could not renumber AI output frames: " + renameErr;
            return false;
        }
    }

    // 2. Compare output frame count to input frame count.
    const std::size_t outputValidFrames = countValidFrames(framesAiDir);
    const std::size_t inputFrameCount   = countValidFrames(framesInputDir);
    if (outputValidFrames == 0) {
        error = "AI upscale produced no valid output frames.\n"
                "Check GPU/Vulkan access and that the model '" + modelName + "' is installed.";
        return false;
    }
    if (inputFrameCount > 0 && outputValidFrames < inputFrameCount * 8 / 10) {
        // More than 20% of frames missing: almost certainly a GPU OOM or crash.
        error = "AI upscale only completed " + std::to_string(outputValidFrames) +
                " of " + std::to_string(inputFrameCount) + " frames (" +
                std::to_string(outputValidFrames * 100 / inputFrameCount) + "%).\n"
                "This indicates the GPU ran out of memory.  Try adding a stage parameter\n"
                "  tile=128  (or tile=64 for large 4K sources) to reduce VRAM usage.";
        return false;
    }
    if (inputFrameCount > 0 && outputValidFrames < inputFrameCount) {
        std::cerr << "[ai] WARNING: " << (inputFrameCount - outputValidFrames)
                  << " frames missing from AI output (" << outputValidFrames
                  << "/" << inputFrameCount << "); those frames will be black.\n";
    }

    // 3. Sample-check the first, middle, and last output frames for black content.
    //    realesrgan can exit 0 and write valid-sized PNGs that are still all-black
    //    when a Vulkan compute shader silently fails.
    {
        std::vector<std::filesystem::path> sampleFiles;
        std::vector<std::filesystem::path> allFiles;
        std::error_code iterEc;
        for (const auto& e : std::filesystem::directory_iterator(framesAiDir, iterEc))
            if (!iterEc && e.is_regular_file() && e.path().extension() == ".png")
                allFiles.push_back(e.path());
        std::sort(allFiles.begin(), allFiles.end());
        if (!allFiles.empty()) {
            sampleFiles.push_back(allFiles.front());
            if (allFiles.size() > 2) sampleFiles.push_back(allFiles[allFiles.size() / 2]);
            if (allFiles.size() > 1) sampleFiles.push_back(allFiles.back());
        }
        int brightCount = 0;
        for (const auto& f : sampleFiles) {
            const int luma = sampleFrameLuma(f);
            if (luma > 8) ++brightCount;  // luma > 8/255 = not black
            std::cout << "[ai] brightness check " << f.filename().string()
                      << " luma=" << luma << std::endl;
        }
        if (!sampleFiles.empty() && brightCount == 0) {
            error = "AI upscale frames appear to be all-black (luma ≤ 8 across all sampled frames).\n"
                    "This is a Vulkan compute failure — the GPU driver processed the workload\n"
                    "but the shader produced zero output.  Common causes on AMD:\n"
                    "  • ROCm/AMDGPU-PRO driver version mismatch with the realesrgan Vulkan shaders\n"
                    "  • Insufficient VRAM — try adding  tile=128  to the upscale stage\n"
                    "  • Run 'vulkaninfo | grep deviceName' to confirm the GPU is being used";
            return false;
        }
    }

    const std::string aiPattern = (framesAiDir / "%08d.png").string();

    if (progressCb) progressCb(0, 75,
        "Encoding " + std::to_string(outputValidFrames) + " upscaled frames with FFmpeg…");
    std::ostringstream encodeCmd;
    encodeCmd << "ffmpeg -y -hide_banner -loglevel error -progress - "
              << "-framerate " << probe->fps << " -start_number 0 -i "
              << quoteArg(aiPattern) << " -i " << quoteArg(job.inputPath) << ' ';

    // Always force yuv420p so libx264/libx265 receive a compatible pixel format.
    // Without this, RGBA or RGB24 PNGs from realesrgan cause all-black output
    // because the encoder either rejects the format or silently mis-converts it.
    std::vector<std::string> finalPostFilters = postFilters;
    finalPostFilters.push_back("format=yuv420p");

    {
        const std::string chain = joinFilters(finalPostFilters);
        if (!postFilters.empty()) {
            std::cout << "[ai] post-upscale filters: " << chain << std::endl;
        }
        encodeCmd << "-vf " << quoteArg(chain) << ' ';
    }

    // -r sets the output frame rate explicitly; relying only on -framerate (input
    // demuxer option) can leave the encoder with an undefined/mismatched rate.
    encodeCmd << "-r " << probe->fps << ' '
              << "-map 0:v:0 -map 1:a? -c:v " << job.encode.codec << ' '
              << "-crf " << job.encode.crf << ' '
              << "-preset " << job.encode.preset << ' '
              << "-c:a copy -shortest "
              << quoteArg(job.outputPath);

    if (!runFfmpegWithProgress(encodeCmd.str(), "encode-video",
                               static_cast<std::int64_t>(outputValidFrames),
                               encodeTaskCb, error)) {
        return false;
    }

    return true;
}
#endif  // encodeWithAiUpscale disabled

// ─────────────────────────────────────────────────────────────────
// encodeWithAiProcessing — generalised AI frame-processing pipeline
//
// Handles ALL stages that benefit from AI inference, not just upscale.
// The pipeline:
//   1. Separate stages into pre-AI FFmpeg filters, AI stages, and
//      post-AI FFmpeg filters.
//   2. Extract frames from the input video (applying pre-AI filters).
//   3. For each AI stage:
//      a. AI upscale → realesrgan-ncnn-vulkan (existing path).
//      b. Other AI stage → backend->processFrameDir().
//      c. Between AI stages, apply any intermediate FFmpeg filters.
//   4. Re-encode the final processed frames with post-AI filters.
//
// This generalises encodeWithAiUpscale() to handle denoise, deblur,
// dehalo, restore-compression, remove-artifacts, etc.
// ─────────────────────────────────────────────────────────────────
bool encodeWithAiProcessing(const VideoJob& job,
                            const std::vector<EnhancementStage>& orderedStages,
                            IAcceleratorBackend* backend,
                            std::string& error) {
    std::string probeError;
    const std::optional<VideoProbeInfo> probe = probeVideo(job.inputPath, probeError);
    if (!probe.has_value()) {
        error = probeError;
        return false;
    }

    TempDirectoryGuard tempGuard;
    tempGuard.path = createTempJobDir(job.outputPath);

    const auto& progressCb = job.progressCb;
    const std::int64_t totalInputFrames = countFrames(job.inputPath);

    // ── Build the pipeline: categorise each stage ────────────────
    // We track which stages are AI-processable and which are FFmpeg-only.
    // Between AI stages, any FFmpeg-only stages become intermediate
    // filter passes applied to the frame directory.

    bool framesExtracted = false;
    std::filesystem::path currentFramesDir;
    int dirIndex = 0;

    auto makeFrameDir = [&]() -> std::filesystem::path {
        auto dir = tempGuard.path / ("frames_" + std::to_string(dirIndex++));
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        return dir;
    };

    // Pending FFmpeg filters that accumulate between AI stages.
    std::vector<std::string> pendingFilters;
    // Post-AI FFmpeg filters (stages after the last AI stage).
    std::vector<std::string> postFilters;

    // First pass: find the index of the last AI-processable stage.
    std::size_t lastAiIdx = 0;
    bool hasAnyAi = false;
    for (std::size_t i = 0; i < orderedStages.size(); ++i) {
        if (isAiProcessable(orderedStages[i], backend)) {
            lastAiIdx = i;
            hasAnyAi = true;
        }
    }
    if (!hasAnyAi) {
        // Shouldn't reach here (caller should check), but fallback gracefully.
        error = "No AI-processable stages found.";
        return false;
    }

    // Progress allocation:
    //   0–15%:   frame extraction
    //   15–85%:  AI processing (divided equally among AI stages)
    //   85–100%: final encoding
    int aiStageCount = 0;
    for (std::size_t i = 0; i < orderedStages.size(); ++i)
        if (isAiProcessable(orderedStages[i], backend)) ++aiStageCount;
    int aiStagesDone = 0;

    // ── Process each stage ───────────────────────────────────────
    for (std::size_t i = 0; i < orderedStages.size(); ++i) {
        const auto& stage = orderedStages[i];
        const bool isAi = isAiProcessable(stage, backend);

        if (i > lastAiIdx) {
            // Past the last AI stage → all remaining are post-AI filters.
            appendFiltersForStage(stage, postFilters, true);
            continue;
        }

        if (!isAi) {
            // FFmpeg-only stage before/between AI stages → accumulate filter.
            appendFiltersForStage(stage, pendingFilters, true);
            continue;
        }

        // ── This is an AI-processable stage ─────────────────────

        // Step 1: Extract frames if not yet done.
        if (!framesExtracted) {
            currentFramesDir = makeFrameDir();
            const std::string inputPattern =
                (currentFramesDir / "%08d.png").string();

            std::ostringstream extractCmd;
            extractCmd << "ffmpeg -y -hide_banner -loglevel error -progress - "
                       << "-i " << quoteArg(job.inputPath) << ' ';
            if (!pendingFilters.empty()) {
                const std::string chain = joinFilters(pendingFilters);
                extractCmd << "-vf " << quoteArg(chain) << ' ';
                std::cout << "[ai-pipeline] pre-AI filters: " << chain << std::endl;
            }
            extractCmd << "-pix_fmt rgb24 -vsync 0 -start_number 0 "
                       << quoteArg(inputPattern);

            auto extractCb = [&progressCb](float f, const std::string& msg) {
                if (progressCb) progressCb(0, static_cast<int>(f * 15.0f), msg);
            };

            if (progressCb) progressCb(0, 0, "Extracting frames for AI processing\u2026");
            if (!runFfmpegWithProgress(extractCmd.str(), "extract-frames",
                                       totalInputFrames, extractCb, error))
                return false;

            framesExtracted = true;
            pendingFilters.clear();
        }

        // Step 2: Apply any pending intermediate FFmpeg filters.
        if (!pendingFilters.empty()) {
            auto nextDir = makeFrameDir();
            const std::string inPat = (currentFramesDir / "%08d.png").string();
            const std::string outPat = (nextDir / "%08d.png").string();

            std::ostringstream filterCmd;
            filterCmd << "ffmpeg -y -hide_banner -loglevel error -progress - "
                      << "-framerate " << probe->fps << " -start_number 0 "
                      << "-i " << quoteArg(inPat) << ' '
                      << "-vf " << quoteArg(joinFilters(pendingFilters)) << ' '
                      << "-pix_fmt rgb24 -start_number 0 "
                      << quoteArg(outPat);

            std::cout << "[ai-pipeline] intermediate filters: "
                      << joinFilters(pendingFilters) << std::endl;

            if (!runFfmpegWithProgress(filterCmd.str(), "apply-intermediate-filters",
                                       totalInputFrames, nullptr, error))
                return false;

            currentFramesDir = nextDir;
            pendingFilters.clear();
        }

        // Step 3: Run AI inference on frames.
        auto aiOutputDir = makeFrameDir();

        if (shouldUseAiUpscale(stage)) {
            // ── AI Upscale via realesrgan-ncnn-vulkan ────────────
            if (!commandInPath("realesrgan-ncnn-vulkan")) {
                error = "AI upscale requested but realesrgan-ncnn-vulkan "
                        "is not available in PATH.";
                return false;
            }

            std::int64_t width = 0, height = 0;
            const auto tw = (tryGetInt(stage.params, "width", width) && width > 0)
                            ? std::optional<std::int64_t>{width} : std::nullopt;
            const auto th = (tryGetInt(stage.params, "height", height) && height > 0)
                            ? std::optional<std::int64_t>{height} : std::nullopt;

            const std::int64_t aiScale = pickAiScale(stage, probe, tw, th);
            const std::string modelName =
                getOptionalStringParam(stage, "model").value_or("realesr-animevideov3");

            std::string modelDir = getOptionalStringParam(stage, "model_dir").value_or("");
            if (modelDir.empty()) {
                const std::filesystem::path defDir =
                    "/usr/share/realesrgan-ncnn-vulkan/models";
                std::error_code ec;
                if (std::filesystem::exists(defDir, ec)) modelDir = defDir.string();
            }

            std::cout << "[ai-pipeline] realesrgan model=" << modelName
                      << " scale=" << aiScale << std::endl;

            std::ostringstream aiCmd;
            aiCmd << "realesrgan-ncnn-vulkan"
                  << " -i " << quoteArg(currentFramesDir.string())
                  << " -o " << quoteArg(aiOutputDir.string())
                  << " -s " << aiScale
                  << " -n " << modelName
                  << " -f png";
            if (!modelDir.empty())
                aiCmd << " -m " << quoteArg(modelDir);
            if (const auto gpu = getOptionalIntParam(stage, "gpu_id"); gpu.has_value())
                aiCmd << " -g " << *gpu;
            if (const auto tile = getOptionalIntParam(stage, "tile"); tile.has_value())
                aiCmd << " -t " << clampInt(*tile, 0, 4096);
            else
                aiCmd << " -t 256";
            if (const auto jobs = getOptionalStringParam(stage, "jobs");
                jobs.has_value() && !jobs->empty())
                aiCmd << " -j " << *jobs;
            if (getOptionalBoolParam(stage, "tta").value_or(false))
                aiCmd << " -x";

            const float aiBase = 15.0f + (70.0f * static_cast<float>(aiStagesDone) /
                                                   static_cast<float>(aiStageCount));
            const float aiSpan = 70.0f / static_cast<float>(aiStageCount);
            auto aiCb = [&progressCb, aiBase, aiSpan](float f, const std::string& msg) {
                if (progressCb) progressCb(0, static_cast<int>(aiBase + f * aiSpan), msg);
            };

            if (progressCb) progressCb(0, static_cast<int>(aiBase),
                "Running AI upscale " + modelName + " \u00d7" + std::to_string(aiScale) + "\u2026");
            if (!runRealEsrganWithProgress(aiCmd.str(), aiCb, error))
                return false;

            // Post-validation: renumber + count + brightness check.
            {
                std::string renErr;
                if (!renumberFramesSequentially(aiOutputDir, renErr)) {
                    error = "Could not renumber AI frames: " + renErr;
                    return false;
                }
            }
            const std::size_t outValid = countValidFrames(aiOutputDir);
            const std::size_t inValid  = countValidFrames(currentFramesDir);
            if (outValid == 0) {
                error = "AI upscale produced no valid output frames.";
                return false;
            }
            if (inValid > 0 && outValid < inValid * 8 / 10) {
                error = "AI upscale only completed " + std::to_string(outValid) +
                        " of " + std::to_string(inValid) + " frames.";
                return false;
            }

            // If target dimensions differ from realesrgan output, add a scale
            // filter as a pending filter for the next pass or final encode.
            if (tw.has_value() && th.has_value()) {
                postFilters.insert(postFilters.begin(),
                    "scale=" + std::to_string(*tw) + ":" + std::to_string(*th));
            }

        } else if (backend) {
            // ── Non-upscale AI stage via backend ────────────────
            const float aiBase = 15.0f + (70.0f * static_cast<float>(aiStagesDone) /
                                                   static_cast<float>(aiStageCount));
            const float aiSpan = 70.0f / static_cast<float>(aiStageCount);
            auto frameCb = [&progressCb, aiBase, aiSpan](float f, const std::string& msg) {
                if (progressCb) progressCb(0, static_cast<int>(aiBase + f * aiSpan), msg);
            };

            if (progressCb) progressCb(0, static_cast<int>(aiBase),
                "Running AI " + toString(stage.kind) + " via " + backend->name() + "\u2026");

            std::string backendError;
            const StageResult result = backend->processFrameDir(
                stage, currentFramesDir.string(), aiOutputDir.string(),
                frameCb, backendError);

            if (result == StageResult::Processed) {
                std::cout << "[ai-pipeline] " << toString(stage.kind)
                          << " → AI complete via " << backend->name() << std::endl;
            } else if (result == StageResult::Error) {
                error = "AI inference failed for " + toString(stage.kind) +
                        ": " + backendError;
                return false;
            } else {
                // Deferred: backend couldn't process. Apply FFmpeg filter instead.
                std::cout << "[ai-pipeline] " << toString(stage.kind)
                          << " → deferred to FFmpeg filter." << std::endl;
                // Apply the filter to the current frames via an FFmpeg pass.
                std::vector<std::string> fallbackFilter;
                appendFiltersForStage(stage, fallbackFilter, true);
                if (!fallbackFilter.empty()) {
                    const std::string inPat = (currentFramesDir / "%08d.png").string();
                    const std::string outPat = (aiOutputDir / "%08d.png").string();

                    std::ostringstream fbCmd;
                    fbCmd << "ffmpeg -y -hide_banner -loglevel error -progress - "
                          << "-framerate " << probe->fps << " -start_number 0 "
                          << "-i " << quoteArg(inPat) << ' '
                          << "-vf " << quoteArg(joinFilters(fallbackFilter)) << ' '
                          << "-pix_fmt rgb24 -start_number 0 "
                          << quoteArg(outPat);
                    if (!runFfmpegWithProgress(fbCmd.str(), "fallback-filter",
                                               totalInputFrames, nullptr, error))
                        return false;
                } else {
                    // No filter fallback and no AI: just copy frames.
                    aiOutputDir = currentFramesDir;
                }
            }
        }

        currentFramesDir = aiOutputDir;
        ++aiStagesDone;
    }

    // ── Final encode: frame directory → output video ─────────────
    // Always add format=yuv420p to ensure encoder compatibility.
    postFilters.push_back("format=yuv420p");

    if (!postFilters.empty()) {
        std::cout << "[ai-pipeline] post-AI filters: "
                  << joinFilters(postFilters) << std::endl;
    }

    const std::size_t finalFrameCount = countValidFrames(currentFramesDir);
    const std::string aiPattern = (currentFramesDir / "%08d.png").string();

    auto encodeCb = [&progressCb](float f, const std::string& msg) {
        if (progressCb) progressCb(0, static_cast<int>(85.0f + f * 15.0f), msg);
    };

    if (progressCb) progressCb(0, 85,
        "Encoding " + std::to_string(finalFrameCount) + " AI-processed frames\u2026");

    std::ostringstream encCmd;
    encCmd << "ffmpeg -y -hide_banner -loglevel error -progress - "
           << "-framerate " << probe->fps << " -start_number 0 -i "
           << quoteArg(aiPattern) << " -i " << quoteArg(job.inputPath) << ' ';

    const std::string postChain = joinFilters(postFilters);
    encCmd << "-vf " << quoteArg(postChain) << ' ';

    encCmd << "-r " << probe->fps << ' '
           << "-map 0:v:0 -map 1:a? -c:v " << job.encode.codec << ' '
           << "-crf " << job.encode.crf << ' '
           << "-preset " << job.encode.preset << ' '
           << "-c:a copy -shortest "
           << quoteArg(job.outputPath);

    return runFfmpegWithProgress(encCmd.str(), "encode-video",
                                  static_cast<std::int64_t>(finalFrameCount),
                                  encodeCb, error);
}

}  // namespace

bool FfmpegRunner::isAvailable(std::string& error) const {
    if (commandInPath("ffmpeg") && commandInPath("ffprobe")) {
        return true;
    }

    error = "ffmpeg/ffprobe binaries not found in PATH.";
    return false;
}

bool FfmpegRunner::encode(const VideoJob& job,
                          const std::vector<EnhancementStage>& orderedStages,
                          IAcceleratorBackend* backend,
                          std::string& error) const {
    std::string availabilityError;
    if (!isAvailable(availabilityError)) {
        error = availabilityError;
        return false;
    }

    // Check whether any stage should be processed via AI inference.
    bool hasAiStages = false;
    for (const auto& stage : orderedStages) {
        if (isAiProcessable(stage, backend)) {
            hasAiStages = true;
            break;
        }
    }

    if (hasAiStages) {
        return encodeWithAiProcessing(job, orderedStages, backend, error);
    }

    std::vector<std::string> filters;
    for (const EnhancementStage& stage : orderedStages) {
        appendFiltersForStage(stage, filters, true);
    }

    const auto& progressCb = job.progressCb;
    if (progressCb) progressCb(0, 0, "Probing input for frame count\u2026");
    const std::int64_t totalFrames = countFrames(job.inputPath);
    if (filters.empty()) {
        if (progressCb) progressCb(0, 0,
            "No enhancement filters active — encoding " + std::to_string(totalFrames) + " frames.");
    } else {
        const std::string chain = joinFilters(filters);
        if (progressCb) progressCb(0, 0,
            std::to_string(filters.size()) + " filter(s) active: " + chain);
        std::cout << "[ffmpeg] filter chain: " << chain << std::endl;
    }

    auto taskCb = [&progressCb](float frac, const std::string& msg) {
        if (progressCb) progressCb(0, static_cast<int>(frac * 100.0f + 0.5f), msg);
    };

    std::ostringstream cmd;
    cmd << "ffmpeg -y -hide_banner -loglevel error -progress - "
        << "-i " << quoteArg(job.inputPath) << ' ';

    if (!filters.empty()) {
        const std::string chain = joinFilters(filters);
        cmd << "-vf " << quoteArg(chain) << ' ';
    }

    cmd << "-map 0:v:0 -map 0:a? ";
    cmd << "-c:v " << job.encode.codec << ' ';
    cmd << "-crf " << job.encode.crf << ' ';
    cmd << "-preset " << job.encode.preset << ' ';
    cmd << "-c:a copy ";
    cmd << quoteArg(job.outputPath);

    return runFfmpegWithProgress(cmd.str(), "encode-video", totalFrames, taskCb, error);
}

}  // namespace ave
