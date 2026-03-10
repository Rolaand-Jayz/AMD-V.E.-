#include "ave/ffmpeg_runner.hpp"

#include <algorithm>
#include <array>
#include <atomic>
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
        std::string& error,
        std::atomic<bool>* cancelFlag = nullptr) {
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
    bool cancelled = false;
    while (fgets(buf, sizeof(buf), pipe)) {
        if (cancelFlag && cancelFlag->load(std::memory_order_relaxed)) {
            cancelled = true;
            break;
        }
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

    if (cancelled) {
        error = stepName + " cancelled by user.";
        // Remove temp stderr file
        std::error_code rmEc;
        std::filesystem::remove(errTmp, rmEc);
        return false;
    }

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

bool hasModelId(const EnhancementStage& stage) {
    const auto it = stage.params.find("model");
    if (it == stage.params.end()) return false;
    if (const auto* s = std::get_if<std::string>(&it->second))
        return !s->empty();
    return false;
}

bool isScriptedFilterBackend(const IAcceleratorBackend* backend) {
    if (backend == nullptr) {
        return false;
    }
    return backend->type() == BackendType::VapourSynth ||
           backend->type() == BackendType::GlslShader;
}

// Returns true if the stage should be processed via AI inference
// (either via the backend's processFrameDir or via realesrgan).
bool isAiProcessable(const EnhancementStage& stage,
                     const IAcceleratorBackend* backend) {
    if (stage.backendProcessed) {
        return false;
    }
    if (isScriptedFilterBackend(backend)) {
        return true;
    }
    if (shouldUseAiUpscale(stage)) return true;
    if (backend && (hasModelPath(stage) || hasModelId(stage))) {
        return true;
    }
    return false;
}

std::filesystem::path createTempJobDir(const std::filesystem::path& outputPath) {
    const auto base = std::filesystem::path(outputPath).parent_path();
    const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
    std::filesystem::path path = base / ("ave_ai_job_" + std::to_string(tick));
    std::filesystem::create_directories(path);
    return path;
}

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

    std::string currentVideoPath = job.inputPath;
    int fileIndex = 0;

    auto makeTempVideo = [&]() -> std::string {
        return (tempGuard.path / ("temp_" + std::to_string(fileIndex++) + ".mkv")).string();
    };

    // ── Preview: create a duration-limited clip if preview mode ──
    const bool isPreview = job.previewMode && job.previewDurationSec > 0.0;
    if (isPreview) {
        std::string previewClip = makeTempVideo();
        std::ostringstream pvCmd;
        pvCmd << "ffmpeg -y -hide_banner -loglevel error -progress - "
              << "-i " << quoteArg(currentVideoPath) << ' '
              << "-t " << job.previewDurationSec << ' '
              << "-c:v libx264 -crf 0 -preset ultrafast -c:a copy "
              << quoteArg(previewClip);
        if (!runFfmpegWithProgress(pvCmd.str(), "create-preview-clip",
                                   totalInputFrames, nullptr, error))
            return false;
        currentVideoPath = previewClip;
    }

    std::vector<std::string> pendingFilters;
    std::vector<std::string> postFilters;

    std::size_t lastAiIdx = 0;
    bool hasAnyAi = false;
    for (std::size_t i = 0; i < orderedStages.size(); ++i) {
        if (isAiProcessable(orderedStages[i], backend)) {
            lastAiIdx = i;
            hasAnyAi = true;
        }
    }
    if (!hasAnyAi) {
        error = "No AI-processable stages found.";
        return false;
    }

    int aiStageCount = 0;
    for (std::size_t i = 0; i < orderedStages.size(); ++i)
        if (isAiProcessable(orderedStages[i], backend)) ++aiStageCount;
    int aiStagesDone = 0;
    bool finalOutputReady = false;

    auto canDirectEncodeFinalFromBackend = [&](std::size_t aiIdx) {
        if (!backend || backend->type() != BackendType::MiGraphX) {
            return false;
        }
        if (aiIdx != lastAiIdx) {
            return false;
        }
        std::vector<std::string> trailingFilters;
        for (std::size_t j = aiIdx + 1; j < orderedStages.size(); ++j) {
            appendFiltersForStage(orderedStages[j], trailingFilters, true);
        }
        return trailingFilters.empty();
    };

    for (std::size_t i = 0; i < orderedStages.size(); ++i) {
        const auto& stage = orderedStages[i];
        const bool isAi = isAiProcessable(stage, backend);

        if (i > lastAiIdx) {
            appendFiltersForStage(stage, postFilters, true);
            continue;
        }

        if (!isAi) {
            appendFiltersForStage(stage, pendingFilters, true);
            continue;
        }

        // Apply pending filters before AI stage
        if (!pendingFilters.empty()) {
            std::string nextVideo = makeTempVideo();
            std::ostringstream filterCmd;
            filterCmd << "ffmpeg -y -hide_banner -loglevel error -progress - "
                      << "-i " << quoteArg(currentVideoPath) << ' '
                      << "-vf " << quoteArg(joinFilters(pendingFilters)) << ' '
                      << "-c:v libx264 -crf 0 -preset ultrafast -c:a copy "
                      << quoteArg(nextVideo);

            if (!runFfmpegWithProgress(filterCmd.str(), "apply-intermediate-filters",
                                       totalInputFrames, nullptr, error))
                return false;

            currentVideoPath = nextVideo;
            pendingFilters.clear();
        }

        // Run AI inference
        std::string aiOutputVideo = makeTempVideo();
        const bool directOutputEncode = canDirectEncodeFinalFromBackend(i);
        const std::string backendOutputVideo = directOutputEncode ? job.outputPath : aiOutputVideo;
        bool aiHandled = false;

        if (backend) {
            const float aiBase = 15.0f + (70.0f * static_cast<float>(aiStagesDone) /
                                                   static_cast<float>(aiStageCount));
            const float aiSpan = 70.0f / static_cast<float>(aiStageCount);
            auto frameCb = [&progressCb, aiBase, aiSpan](float f, const std::string& msg) {
                if (progressCb) progressCb(0, static_cast<int>(aiBase + f * aiSpan), msg);
            };

            if (progressCb) progressCb(0, static_cast<int>(aiBase),
                "Running AI " + toString(stage.kind) + " via " + backend->name() + "\u2026");

            // Build process options for preview support
            ProcessVideoOptions pvOpts;
            if (isPreview) {
                pvOpts.previewDurationSec = job.previewDurationSec;
            }
            pvOpts.framePreviewCb     = job.framePreviewCb;
            pvOpts.previewFrameInterval = job.previewFrameInterval;
            pvOpts.cancelFlag         = job.cancelFlag;
            pvOpts.pauseFlag          = job.pauseFlag;
            pvOpts.directOutputEncode = directOutputEncode;
            if (directOutputEncode) {
                pvOpts.outputCodec = job.encode.codec;
                pvOpts.outputProfile = job.encode.profile;
                pvOpts.outputCrf = job.encode.crf;
                pvOpts.outputPreset = job.encode.preset;
                pvOpts.outputThreads = job.encode.threads;
            }

            std::string backendError;
            const StageResult result = backend->processVideoFile(
                stage, currentVideoPath, backendOutputVideo,
                frameCb, backendError, pvOpts);

            if (result == StageResult::Processed) {
                std::cout << "[ai-pipeline] " << toString(stage.kind)
                          << " → AI complete via " << backend->name()
                          << (directOutputEncode ? " (direct final encode)" : "")
                          << std::endl;
                aiHandled = true;
                if (directOutputEncode) {
                    finalOutputReady = true;
                }
            } else if (result == StageResult::Cancelled) {
                error = "Processing cancelled by user.";
                return false;
            } else if (result == StageResult::Error) {
                error = "AI inference failed for " + toString(stage.kind) +
                        ": " + backendError;
                return false;
            } else {
                if (job.requestedBackend != BackendType::Auto) {
                    if (backendError.empty()) {
                        backendError = backend->name() + " deferred AI processing for stage '"
                                     + toString(stage.kind) + "'.";
                    }
                    error = "Explicit backend request cannot fall back for "
                          + toString(stage.kind) + ": " + backendError;
                    return false;
                }

                std::cout << "[ai-pipeline] " << toString(stage.kind)
                          << " → deferred to FFmpeg filter." << std::endl;
            }
        }

        // Fallback: backend is null or deferred — apply FFmpeg filter
        if (!aiHandled) {
            std::vector<std::string> fallbackFilter;
            appendFiltersForStage(stage, fallbackFilter, true);
            if (!fallbackFilter.empty()) {
                std::ostringstream fbCmd;
                fbCmd << "ffmpeg -y -hide_banner -loglevel error -progress - "
                      << "-i " << quoteArg(currentVideoPath) << ' '
                      << "-vf " << quoteArg(joinFilters(fallbackFilter)) << ' '
                      << "-c:v libx264 -crf 0 -preset ultrafast -c:a copy "
                      << quoteArg(aiOutputVideo);
                if (!runFfmpegWithProgress(fbCmd.str(), "fallback-filter",
                                           totalInputFrames, nullptr, error))
                    return false;
            } else {
                aiOutputVideo = currentVideoPath;
            }
        }

        currentVideoPath = aiOutputVideo;
        if (directOutputEncode && aiHandled) {
            currentVideoPath = job.outputPath;
        }
        ++aiStagesDone;
    }

    if (finalOutputReady) {
        std::cout << "[ai-pipeline] Final delivery encode completed inside "
                  << backend->name() << "; skipping redundant FFmpeg re-encode."
                  << std::endl;
        return true;
    }

    // Final encode
    postFilters.push_back("format=yuv420p");

    auto encodeCb = [&progressCb](float f, const std::string& msg) {
        if (progressCb) progressCb(0, static_cast<int>(85.0f + f * 15.0f), msg);
    };

    if (progressCb) progressCb(0, 85, "Encoding final video\u2026");

    std::ostringstream encCmd;
    encCmd << "ffmpeg -y -hide_banner -loglevel error -progress - "
           << "-i " << quoteArg(currentVideoPath) << ' ';

    if (currentVideoPath != job.inputPath) {
        encCmd << "-i " << quoteArg(job.inputPath) << ' ';
    }

    const std::string postChain = joinFilters(postFilters);
    encCmd << "-vf " << quoteArg(postChain) << ' ';

    encCmd << "-r " << probe->fps << ' ';
    
    if (currentVideoPath != job.inputPath) {
        encCmd << "-map 0:v:0 -map 1:a? ";
    } else {
        encCmd << "-map 0:v:0 -map 0:a? ";
    }
    
    encCmd << "-c:v " << job.encode.codec << ' '
           << "-crf " << job.encode.crf << ' '
           << "-preset " << job.encode.preset << ' ';
    if (!job.encode.profile.empty()) {
        encCmd << "-profile:v " << job.encode.profile << ' ';
    }
    if (job.encode.threads > 0) {
        encCmd << "-threads " << job.encode.threads << ' ';
    }
    encCmd << "-c:a copy -shortest "
           << quoteArg(job.outputPath);

    return runFfmpegWithProgress(encCmd.str(), "encode-video",
                                  totalInputFrames,
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

    // Limit duration for preview mode
    if (job.previewMode && job.previewDurationSec > 0.0) {
        cmd << "-t " << job.previewDurationSec << ' ';
    }

    if (!filters.empty()) {
        const std::string chain = joinFilters(filters);
        cmd << "-vf " << quoteArg(chain) << ' ';
    }

    cmd << "-map 0:v:0 -map 0:a? ";
    cmd << "-c:v " << job.encode.codec << ' ';
    cmd << "-crf " << job.encode.crf << ' ';
    cmd << "-preset " << job.encode.preset << ' ';
    if (job.encode.threads > 0) {
        cmd << "-threads " << job.encode.threads << ' ';
    }
    if (!job.encode.profile.empty()) {
        cmd << "-profile:v " << job.encode.profile << ' ';
    }
    cmd << "-c:a copy ";
    cmd << quoteArg(job.outputPath);

    return runFfmpegWithProgress(cmd.str(), "encode-video", totalFrames, taskCb, error,
                                 job.cancelFlag);
}

}  // namespace ave
