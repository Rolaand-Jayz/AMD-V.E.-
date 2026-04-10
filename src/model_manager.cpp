#include "ave/model_manager.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cctype>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
#if defined(__unix__) || defined(__APPLE__)
#  include <sys/wait.h>
#endif

#ifdef AVE_HAVE_CURL
#  include <curl/curl.h>
#endif

#include "ave/observability.hpp"
#include "ave/process_observer.hpp"
#include "ave/runtime_diagnostics.hpp"
#include "ave/runtime_paths.hpp"

#ifdef AVE_HAVE_HIP
#  include <hip/hip_runtime.h>
#endif

namespace ave {

// ─────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────
namespace {

// Small fixed-shape fallback used only for explicit "convert model" actions
// where no real video dimensions are available yet. Runtime auto-compile still
// waits for the actual input frame size before producing inference artifacts.
constexpr int kDefaultBaselineCompileWidth = 192;
constexpr int kDefaultBaselineCompileHeight = 192;
constexpr int kDefaultStereoDepthResolution = 384;
constexpr int kMinimumStereoDepthResolution = 224;
constexpr int kStereoDepthResolutionStride = 14;
// MiOpen kernel JIT is CPU-bound and dominates first-compile time.
// Use up to 16 parallel threads instead of the old conservative cap of 8.
constexpr int kDefaultMiopenCompileParallelCap = 16;
constexpr int kMinimumMiGraphXTileFallbackExtent = 64;
constexpr std::array<int, 4> kMiGraphXTileFallbackExtents = {192, 128, 96, 64};
constexpr char kArtifactManifestSchemaVersion[] = "2";

enum class MiGraphXCompileProfile {
    Fast,
    Balanced,
    Exhaustive,
};

struct MiGraphXDriverEnv {
    std::vector<std::pair<std::string, std::string>> effective;
    std::vector<std::pair<std::string, std::string>> overrides;
    MiGraphXCompileProfile profile = MiGraphXCompileProfile::Fast;
    std::string profileLabel;
    std::string disableMlir;
    std::string enableNhwc;
    std::string enableCk;
    std::string problemCachePath;
    std::string miopenUserDbPath;
    std::string miopenCustomCacheDir;
    std::string miopenFindMode;
    std::string miopenCompileParallelLevel;
    std::string visibleDevices;
    std::string runtimeFingerprint;
    // Cached system identity values — reused in buildArtifactManifestFields()
    // to avoid redundant subprocess launches and file reads.
    std::string rocmVersion;
    std::string gfxTarget;
};

std::mutex& processEnvMutex() {
    static std::mutex mutex;
    return mutex;
}

bool setProcessEnv(const std::string& name, const std::string& value) {
#if defined(_WIN32)
    return _putenv_s(name.c_str(), value.c_str()) == 0;
#else
    return ::setenv(name.c_str(), value.c_str(), 1) == 0;
#endif
}

bool unsetProcessEnv(const std::string& name) {
#if defined(_WIN32)
    return _putenv_s(name.c_str(), "") == 0;
#else
    return ::unsetenv(name.c_str()) == 0;
#endif
}

std::vector<std::pair<std::string, std::string>> mergeEnvOverrides(
        const std::vector<std::pair<std::string, std::string>>& base,
        const std::vector<std::pair<std::string, std::string>>& overrides) {
    std::vector<std::pair<std::string, std::string>> merged = base;
    for (const auto& [name, value] : overrides) {
        const auto existing = std::find_if(
            merged.begin(),
            merged.end(),
            [&](const auto& entry) { return entry.first == name; });
        if (existing != merged.end()) {
            existing->second = value;
        } else {
            merged.emplace_back(name, value);
        }
    }
    return merged;
}

class ScopedEnvOverrides {
  public:
    explicit ScopedEnvOverrides(const std::vector<std::pair<std::string, std::string>>& overrides)
        : lock_(processEnvMutex()) {
        previous_.reserve(overrides.size());
        for (const auto& [name, value] : overrides) {
            const char* current = std::getenv(name.c_str());
            previous_.emplace_back(name,
                                   current == nullptr
                                       ? std::optional<std::string>{}
                                       : std::make_optional(std::string(current)));
            if (!setProcessEnv(name, value)) {
                error_ = "Failed to set environment override " + name + "=" + value;
                restore();
                break;
            }
        }
    }

    ScopedEnvOverrides(const ScopedEnvOverrides&) = delete;
    ScopedEnvOverrides& operator=(const ScopedEnvOverrides&) = delete;

    ~ScopedEnvOverrides() { restore(); }

    bool ok() const { return error_.empty(); }

    const std::string& error() const { return error_; }

  private:
    void restore() {
        while (!previous_.empty()) {
            const auto previous = previous_.back();
            previous_.pop_back();
            if (previous.second.has_value()) {
                setProcessEnv(previous.first, *previous.second);
            } else {
                unsetProcessEnv(previous.first);
            }
        }
    }

    std::unique_lock<std::mutex> lock_;
    std::vector<std::pair<std::string, std::optional<std::string>>> previous_;
    std::string error_;
};

std::string defaultModelsDir() {
    return defaultWritableModelsDir().string();
}

bool ensureDir(const std::filesystem::path& p) {
    std::error_code ec;
    std::filesystem::create_directories(p, ec);
    return !ec;
}

bool fileExists(const std::filesystem::path& p) {
    std::error_code ec;
    return std::filesystem::exists(p, ec);
}

std::string trimCopy(const std::string& input) {
    std::size_t begin = 0;
    while (begin < input.size() &&
           std::isspace(static_cast<unsigned char>(input[begin])) != 0) {
        ++begin;
    }
    std::size_t end = input.size();
    while (end > begin &&
           std::isspace(static_cast<unsigned char>(input[end - 1])) != 0) {
        --end;
    }
    return input.substr(begin, end - begin);
}

std::optional<std::filesystem::path> resolveMiGraphXDriver() {
    if (const auto bundled = bundledMiGraphXDriverPath(); bundled.has_value()) {
        return bundled;
    }
    return process_observer::resolveCommandPath("migraphx-driver");
}

bool hasMxrExtension(const std::string& path) {
    if (path.size() < 4) { return false; }
    const std::string ext = path.substr(path.size() - 4);
    return ext == ".mxr" || ext == ".MXR";
}

std::string shellQuote(const std::string& value) {
    std::string out = "'";
    for (char ch : value) {
        if (ch == '\'') {
            out += "'\\''";
        } else {
            out.push_back(ch);
        }
    }
    out.push_back('\'');
    return out;
}

std::string trimLine(std::string line) {
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
        line.pop_back();
    }
    return line;
}

std::string toLowerCopy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](const unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

std::vector<std::string> splitCsv(const std::string& input) {
    std::vector<std::string> out;
    std::size_t start = 0;
    while (start <= input.size()) {
        const std::size_t end = input.find(',', start);
        const std::string token = trimCopy(input.substr(
            start, end == std::string::npos ? std::string::npos : end - start));
        if (!token.empty()) {
            out.push_back(token);
        }
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
    return out;
}

bool parseBoolValue(const std::string& input, bool& out) {
    const std::string normalized = toLowerCopy(trimCopy(input));
    if (normalized == "1" || normalized == "true" || normalized == "yes" || normalized == "on") {
        out = true;
        return true;
    }
    if (normalized == "0" || normalized == "false" || normalized == "no" || normalized == "off") {
        out = false;
        return true;
    }
    return false;
}

std::optional<ModelFormat> modelFormatFromString(const std::string& input) {
    const std::string normalized = toLowerCopy(trimCopy(input));
    if (normalized == "onnx") {
        return ModelFormat::Onnx;
    }
    if (normalized == "ncnn" || normalized == "ncnnbin" || normalized == "ncnn_bin") {
        return ModelFormat::NcnnBin;
    }
    if (normalized == "pytorch" || normalized == "pt" || normalized == "pth") {
        return ModelFormat::Pytorch;
    }
    return std::nullopt;
}

std::optional<ModelPrecision> modelPrecisionFromString(const std::string& input) {
    const std::string normalized = toLowerCopy(trimCopy(input));
    if (normalized == "fp32") {
        return ModelPrecision::Fp32;
    }
    if (normalized == "fp16") {
        return ModelPrecision::Fp16;
    }
    if (normalized == "int8") {
        return ModelPrecision::Int8;
    }
    return std::nullopt;
}

std::optional<ModelEntry> parseCustomModelManifest(const std::filesystem::path& manifestPath,
                                                   std::string& error) {
    std::ifstream in(manifestPath);
    if (!in.is_open()) {
        error = "Unable to open manifest: " + manifestPath.string();
        return std::nullopt;
    }

    std::unordered_map<std::string, std::string> kv;
    std::string line;
    int lineNo = 0;
    while (std::getline(in, line)) {
        ++lineNo;
        line = trimLine(std::move(line));
        const std::string trimmed = trimCopy(line);
        if (trimmed.empty() || trimmed[0] == '#') {
            continue;
        }
        const std::size_t pos = trimmed.find('=');
        if (pos == std::string::npos) {
            error = "Invalid manifest line " + std::to_string(lineNo) + " in "
                  + manifestPath.string() + ": expected key=value.";
            return std::nullopt;
        }
        kv[toLowerCopy(trimCopy(trimmed.substr(0, pos)))] = trimCopy(trimmed.substr(pos + 1));
    }

    ModelEntry entry;
    auto require = [&](const char* key) -> std::optional<std::string> {
        const auto it = kv.find(key);
        if (it == kv.end() || it->second.empty()) {
            error = "Missing required key '" + std::string(key) + "' in "
                  + manifestPath.string();
            return std::nullopt;
        }
        return it->second;
    };

    const auto id = require("id");
    const auto displayName = require("display_name");
    const auto stageValue = require("stage");
    const auto formatValue = require("source_format");
    if (!id.has_value() || !displayName.has_value() || !stageValue.has_value() ||
        !formatValue.has_value()) {
        return std::nullopt;
    }

    const auto stage = stageKindFromString(*stageValue);
    if (!stage.has_value()) {
        error = "Invalid stage '" + *stageValue + "' in " + manifestPath.string();
        return std::nullopt;
    }
    const auto format = modelFormatFromString(*formatValue);
    if (!format.has_value()) {
        error = "Invalid source_format '" + *formatValue + "' in " + manifestPath.string();
        return std::nullopt;
    }

    entry.id = *id;
    entry.displayName = *displayName;
    entry.stage = *stage;
    entry.sourceFormat = *format;
    entry.precision = ModelPrecision::Fp32;
    entry.scale = 1;
    entry.fpsMul = 1.0;
    entry.minVramMib = 512;

    if (const auto it = kv.find("precision"); it != kv.end() && !it->second.empty()) {
        const auto precision = modelPrecisionFromString(it->second);
        if (!precision.has_value()) {
            error = "Invalid precision '" + it->second + "' in " + manifestPath.string();
            return std::nullopt;
        }
        entry.precision = *precision;
    }
    if (const auto it = kv.find("scale"); it != kv.end() && !it->second.empty()) {
        try {
            entry.scale = std::stoi(it->second);
        } catch (...) {
            error = "Invalid scale '" + it->second + "' in " + manifestPath.string();
            return std::nullopt;
        }
    }
    if (const auto it = kv.find("fps_mul"); it != kv.end() && !it->second.empty()) {
        try {
            entry.fpsMul = std::stod(it->second);
        } catch (...) {
            error = "Invalid fps_mul '" + it->second + "' in " + manifestPath.string();
            return std::nullopt;
        }
    }
    if (const auto it = kv.find("min_vram_mib"); it != kv.end() && !it->second.empty()) {
        try {
            entry.minVramMib = std::stoi(it->second);
        } catch (...) {
            error = "Invalid min_vram_mib '" + it->second + "' in " + manifestPath.string();
            return std::nullopt;
        }
    }

    if (const auto it = kv.find("family_id"); it != kv.end()) {
        entry.familyId = it->second;
    }
    if (const auto it = kv.find("family_name"); it != kv.end()) {
        entry.familyName = it->second;
    }
    if (const auto it = kv.find("description"); it != kv.end()) {
        entry.description = it->second;
    }
    if (const auto it = kv.find("file"); it != kv.end()) {
        entry.filename = it->second;
    }
    if (const auto it = kv.find("file_aux"); it != kv.end()) {
        entry.filenameAux = it->second;
    }
    if (const auto it = kv.find("download_url"); it != kv.end()) {
        entry.downloadUrl = it->second;
    }
    if (const auto it = kv.find("download_url_aux"); it != kv.end()) {
        entry.downloadUrlAux = it->second;
    }
    if (const auto it = kv.find("default"); it != kv.end() && !it->second.empty()) {
        if (!parseBoolValue(it->second, entry.isDefault)) {
            error = "Invalid default flag '" + it->second + "' in " + manifestPath.string();
            return std::nullopt;
        }
    }
    if (const auto it = kv.find("fused"); it != kv.end() && !it->second.empty()) {
        if (!parseBoolValue(it->second, entry.supportsFusedExecution)) {
            error = "Invalid fused flag '" + it->second + "' in " + manifestPath.string();
            return std::nullopt;
        }
    }
    if (const auto it = kv.find("selective"); it != kv.end() && !it->second.empty()) {
        if (!parseBoolValue(it->second, entry.supportsSelectiveCapabilities)) {
            error = "Invalid selective flag '" + it->second + "' in " + manifestPath.string();
            return std::nullopt;
        }
    }
    if (const auto it = kv.find("capabilities"); it != kv.end() && !it->second.empty()) {
        for (const auto& token : splitCsv(it->second)) {
            const auto capability = stageKindFromString(token);
            if (!capability.has_value()) {
                error = "Invalid capability '" + token + "' in " + manifestPath.string();
                return std::nullopt;
            }
            entry.capabilities.push_back(*capability);
        }
    }
    if (entry.capabilities.empty()) {
        entry.capabilities.push_back(entry.stage);
    }

    for (const auto& [key, value] : kv) {
        constexpr std::string_view kControlPrefix = "control.";
        if (key.rfind(kControlPrefix.data(), 0) != 0) {
            continue;
        }
        const std::string inputName = trimCopy(key.substr(kControlPrefix.size()));
        if (inputName.empty()) {
            error = "Empty control input name in " + manifestPath.string();
            return std::nullopt;
        }
        if (value.empty()) {
            error = "Empty control binding for input '" + inputName + "' in "
                  + manifestPath.string();
            return std::nullopt;
        }
        entry.controlInputBindings[inputName] = value;
    }

    return entry;
}

std::string statePrefix(ModelState state) {
    switch (state) {
        case ModelState::NotDownloaded: return "[Not Downloaded]";
        case ModelState::Downloading:   return "[Downloading…]";
        case ModelState::Downloaded:    return "[Downloaded]";
        case ModelState::Converting:    return "[Converting…]";
        case ModelState::Converted:     return "[Compiled]";
        case ModelState::Error:         return "[Error]";
    }
    return "";
}

bool isSupportedMiGraphXCompilePrecision(ModelPrecision precision) {
    return precision == ModelPrecision::Fp32 ||
           precision == ModelPrecision::Fp16 ||
           precision == ModelPrecision::Int8;
}

bool isMiGraphXCompileTimeout(const std::string& error) {
    return error.find("migraphx-driver timed out after") != std::string::npos;
}

bool shouldRetryFp32AfterTimeout(ModelPrecision requestedPrecision,
                                 const std::string& error) {
    return requestedPrecision == ModelPrecision::Fp16 &&
           isMiGraphXCompileTimeout(error);
}

std::string compilePrecisionTag(ModelPrecision precision) {
    switch (precision) {
        case ModelPrecision::Fp32: return "fp32";
        case ModelPrecision::Fp16: return "fp16";
        case ModelPrecision::Int8: return "int8";
    }
    return "unknown";
}

std::string compiledArtifactStem(const std::string& modelId,
                                 ModelPrecision precision,
                                 std::optional<int> width = std::nullopt,
                                 std::optional<int> height = std::nullopt,
                                 int batch = 1) {
    std::ostringstream stem;
    stem << modelId;
    if (width.has_value() && height.has_value()) {
        stem << "_" << *width << "x" << *height;
    }
    if (batch > 1) {
        stem << "_b" << batch;
    }
    if (precision != ModelPrecision::Fp32) {
        stem << "_" << compilePrecisionTag(precision);
    }
    return stem.str();
}

std::filesystem::path compiledArtifactPath(const std::filesystem::path& dir,
                                           const std::string& modelId,
                                           ModelPrecision precision,
                                           std::optional<int> width = std::nullopt,
                                           std::optional<int> height = std::nullopt,
                                           int batch = 1) {
    return dir / (compiledArtifactStem(modelId, precision, width, height, batch) + ".mxr");
}

bool compiledArtifactBelongsToModel(const std::filesystem::path& artifactPath,
                                    const std::string& modelId) {
    if (artifactPath.extension() != ".mxr") {
        return false;
    }
    const std::string stem = artifactPath.stem().string();
    return stem == modelId || stem.rfind(modelId + "_", 0) == 0;
}

int compiledArtifactPrecisionRank(const std::filesystem::path& artifactPath) {
    const std::string stem = artifactPath.stem().string();
    if (stem.size() >= 5 && stem.compare(stem.size() - 5, 5, "_fp16") == 0) {
        return 0;
    }
    if (stem.size() >= 5 && stem.compare(stem.size() - 5, 5, "_int8") == 0) {
        return 1;
    }
    return 2;
}

std::vector<std::filesystem::path> findCompiledArtifactsForModel(
        const std::filesystem::path& dir,
        const std::string& modelId) {
    std::error_code ec;
    if (!std::filesystem::exists(dir, ec)) {
        return {};
    }

    std::vector<std::filesystem::path> matches;
    for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
        if (ec) {
            break;
        }
        if (!entry.is_regular_file(ec) || ec) {
            continue;
        }
        const auto& path = entry.path();
        if (!compiledArtifactBelongsToModel(path, modelId)) {
            continue;
        }
        matches.push_back(path);
    }

    std::stable_sort(matches.begin(), matches.end(),
                     [](const std::filesystem::path& lhs,
                        const std::filesystem::path& rhs) {
                         const int lhsRank = compiledArtifactPrecisionRank(lhs);
                         const int rhsRank = compiledArtifactPrecisionRank(rhs);
                         if (lhsRank != rhsRank) {
                             return lhsRank < rhsRank;
                         }
                         const auto lhsStem = lhs.stem().string();
                         const auto rhsStem = rhs.stem().string();
                         if (lhsStem.size() != rhsStem.size()) {
                             return lhsStem.size() > rhsStem.size();
                         }
                         return lhsStem < rhsStem;
                     });
    return matches;
}

struct ParsedCompiledArtifactDescriptor {
    ModelPrecision precision = ModelPrecision::Fp32;
};

std::optional<ParsedCompiledArtifactDescriptor> parseCompiledArtifactDescriptor(
        const std::filesystem::path& artifactPath,
        const std::string& modelId) {
    if (!compiledArtifactBelongsToModel(artifactPath, modelId)) {
        return std::nullopt;
    }

    ParsedCompiledArtifactDescriptor descriptor;
    std::string stem = artifactPath.stem().string();
    if (stem.size() >= 5 && stem.compare(stem.size() - 5, 5, "_fp16") == 0) {
        descriptor.precision = ModelPrecision::Fp16;
        stem.erase(stem.size() - 5);
    } else if (stem.size() >= 5 && stem.compare(stem.size() - 5, 5, "_int8") == 0) {
        descriptor.precision = ModelPrecision::Int8;
        stem.erase(stem.size() - 5);
    }

    const auto batchSeparator = stem.rfind("_b");
    if (batchSeparator != std::string::npos &&
        batchSeparator + 2 < stem.size()) {
        const bool allDigits = std::all_of(
            stem.begin() + static_cast<std::ptrdiff_t>(batchSeparator) + 2,
            stem.end(),
            [](unsigned char ch) { return std::isdigit(ch) != 0; });
        if (allDigits) {
            stem.erase(batchSeparator);
        }
    }

    const auto dimSeparator = stem.rfind('_');
    if (dimSeparator != std::string::npos &&
        dimSeparator + 1 < stem.size()) {
        const auto dimToken = stem.substr(dimSeparator + 1);
        const auto cross = dimToken.find('x');
        if (cross != std::string::npos &&
            cross > 0 &&
            cross + 1 < dimToken.size()) {
            const bool validDims =
                std::all_of(dimToken.begin(), dimToken.begin() + static_cast<std::ptrdiff_t>(cross),
                            [](unsigned char ch) { return std::isdigit(ch) != 0; }) &&
                std::all_of(dimToken.begin() + static_cast<std::ptrdiff_t>(cross) + 1, dimToken.end(),
                            [](unsigned char ch) { return std::isdigit(ch) != 0; });
            if (validDims) {
                stem.erase(dimSeparator);
            }
        }
    }

    if (stem != modelId) {
        return std::nullopt;
    }
    return descriptor;
}

int normaliseCompileBatch(int batch) {
    return std::clamp(batch, 1, 64);
}

int readPositiveIntEnv(const char* name, int defaultValue, int minimumValue, int maximumValue) {
    int value = defaultValue;
    if (const char* raw = std::getenv(name); raw != nullptr) {
        try {
            value = std::stoi(raw);
        } catch (...) {
            value = defaultValue;
        }
    }
    value = std::max(value, minimumValue);
    value = std::min(value, maximumValue);
    return value;
}

int baselineCompileWidth() {
    return readPositiveIntEnv("AVE_MIGRAPHX_BASELINE_WIDTH",
                              kDefaultBaselineCompileWidth, 32, 4096);
}

int baselineCompileHeight() {
    return readPositiveIntEnv("AVE_MIGRAPHX_BASELINE_HEIGHT",
                              kDefaultBaselineCompileHeight, 32, 4096);
}

bool hasFixedMiGraphXCompileDims(const ModelEntry& entry) {
    return entry.migraphxCompileWidth > 0 && entry.migraphxCompileHeight > 0;
}

int normalizeStereoDepthResolution(int resolution) {
    resolution = std::clamp(resolution, kMinimumStereoDepthResolution, 4096);
    const int remainder = resolution % kStereoDepthResolutionStride;
    if (remainder != 0) {
        resolution += (kStereoDepthResolutionStride - remainder);
    }
    return resolution;
}

std::optional<std::pair<int, int>> defaultMiGraphXCompileDims(const ModelEntry& entry) {
    if (hasFixedMiGraphXCompileDims(entry)) {
        return std::make_pair(entry.migraphxCompileWidth, entry.migraphxCompileHeight);
    }
    if (entry.stage == StageKind::Stereo3D) {
        const int extent = normalizeStereoDepthResolution(kDefaultStereoDepthResolution);
        return std::make_pair(extent, extent);
    }
    return std::nullopt;
}

std::pair<int, int> resolveMiGraphXCompileDims(const ModelEntry& entry,
                                               int requestedWidth,
                                               int requestedHeight) {
    if (hasFixedMiGraphXCompileDims(entry)) {
        return {entry.migraphxCompileWidth, entry.migraphxCompileHeight};
    }
    return {requestedWidth, requestedHeight};
}

std::filesystem::path preparedMiGraphXOnnxPath(const std::filesystem::path& dir,
                                               const std::string& modelId) {
    return dir / (modelId + "_migraphx.onnx");
}

bool preparedOnnxNeedsRefresh(const std::filesystem::path& sourcePath,
                              const std::filesystem::path& preparedPath) {
    if (!fileExists(preparedPath)) {
        return true;
    }
    std::error_code sourceEc;
    const auto sourceTime = std::filesystem::last_write_time(sourcePath, sourceEc);
    if (sourceEc) {
        return true;
    }
    std::error_code preparedEc;
    const auto preparedTime = std::filesystem::last_write_time(preparedPath, preparedEc);
    if (preparedEc) {
        return true;
    }
    return preparedTime < sourceTime;
}

bool rewriteOnnxResizeModeCubicToLinear(const std::filesystem::path& sourcePath,
                                        const std::filesystem::path& preparedPath,
                                        std::string& error) {
    if (!process_observer::commandInPath("python3")) {
        error = "python3 not found in PATH; cannot prepare ONNX model for MiGraphX compatibility.";
        return false;
    }

    const std::string pyScript =
        "import shutil, sys\n"
        "try:\n"
        "    import onnx\n"
        "except Exception as exc:\n"
        "    print(f'ERROR: unable to import onnx: {exc}', file=sys.stderr)\n"
        "    sys.exit(11)\n"
        "src, dst = sys.argv[1], sys.argv[2]\n"
        "model = onnx.load(src)\n"
        "changed = 0\n"
        "for node in model.graph.node:\n"
        "    if node.op_type != 'Resize':\n"
        "        continue\n"
        "    for attr in node.attribute:\n"
        "        if attr.name == 'mode' and attr.s == b'cubic':\n"
        "            attr.s = b'linear'\n"
        "            changed += 1\n"
        "if changed == 0:\n"
        "    shutil.copyfile(src, dst)\n"
        "else:\n"
        "    onnx.save(model, dst)\n";

    const auto scriptPath = std::filesystem::temp_directory_path()
                          / "ave_prepare_onnx_for_migraphx.py";
    {
        std::ofstream sf(scriptPath);
        if (!sf) {
            error = "Cannot write temporary ONNX preparation script to " + scriptPath.string();
            return false;
        }
        sf << pyScript;
    }

    std::ostringstream cmd;
    cmd << "python3 " << shellQuote(scriptPath.string())
        << ' ' << shellQuote(sourcePath.string())
        << ' ' << shellQuote(preparedPath.string());
    const int rc = std::system(cmd.str().c_str());

    std::error_code removeEc;
    std::filesystem::remove(scriptPath, removeEc);

    if (rc != 0) {
        error = "Failed to rewrite ONNX Resize(mode=cubic) to Resize(mode=linear) for MiGraphX. "
                "Ensure the Python 'onnx' package is installed.";
        return false;
    }
    if (!fileExists(preparedPath)) {
        error = "Prepared ONNX output not found after MiGraphX compatibility rewrite: "
              + preparedPath.string();
        return false;
    }
    return true;
}

bool prepareOnnxForMiGraphX(const ModelEntry& entry,
                            const std::filesystem::path& sourceOnnxPath,
                            const std::filesystem::path& manifestSourcePath,
                            const std::filesystem::path& preparedDir,
                            std::filesystem::path& effectiveOnnxPath,
                            std::filesystem::path& effectiveManifestSourcePath,
                            std::string& error) {
    effectiveOnnxPath = sourceOnnxPath;
    effectiveManifestSourcePath = manifestSourcePath;
    if (entry.migraphxOnnxTransform == MiGraphXOnnxTransform::None) {
        return true;
    }

    ensureDir(preparedDir);
    const auto preparedPath = preparedMiGraphXOnnxPath(preparedDir, entry.id);
    if (preparedOnnxNeedsRefresh(sourceOnnxPath, preparedPath)) {
        switch (entry.migraphxOnnxTransform) {
            case MiGraphXOnnxTransform::None:
                break;
            case MiGraphXOnnxTransform::ResizeCubicToLinear:
                if (!rewriteOnnxResizeModeCubicToLinear(sourceOnnxPath, preparedPath, error)) {
                    return false;
                }
                break;
        }
    }

    if (!fileExists(preparedPath)) {
        error = "Prepared MiGraphX ONNX file missing: " + preparedPath.string();
        return false;
    }

    effectiveOnnxPath = preparedPath;
    effectiveManifestSourcePath = preparedPath;
    return true;
}

std::optional<std::string> readNonEmptyEnv(const char* name) {
    if (const char* raw = std::getenv(name); raw != nullptr && *raw != '\0') {
        return std::string(raw);
    }
    return std::nullopt;
}

bool readBoolEnv(const char* name, bool defaultValue = false) {
    const auto raw = readNonEmptyEnv(name);
    if (!raw.has_value()) {
        return defaultValue;
    }

    std::string lowered = *raw;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    if (lowered == "1" || lowered == "true" || lowered == "yes" || lowered == "on") {
        return true;
    }
    if (lowered == "0" || lowered == "false" || lowered == "no" || lowered == "off") {
        return false;
    }
    return defaultValue;
}

bool migraphxOffloadCopyEnabled() {
    return readBoolEnv("AVE_MIGRAPHX_OFFLOAD_COPY", true);
}

std::optional<int> preferredAmdDeviceIndexFromSettings() {
    return ::ave::preferredAmdDeviceIndexFromEnv();
}

// Cached once per process — GFX target never changes while the app is running.
// Avoids spawning a rocminfo subprocess on every compilation/validation call.
std::string detectGfxTargetForIdentity() {
    return ::ave::detectAmdGpuArch();
}

// Cached once per process — ROCm version does not change at runtime.
std::string detectRocmVersion() {
    return ::ave::detectRocmVersion();
}

std::optional<int> autoSelectedAmdDeviceIndex() {
#ifdef AVE_HAVE_HIP
    static const std::optional<int> cached = []() -> std::optional<int> {
        int deviceCount = 0;
        if (hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount <= 0) {
            return std::nullopt;
        }

        int bestDevice = 0;
        std::uint64_t bestMemory = 0;
        bool foundDiscrete = false;
        for (int deviceIdx = 0; deviceIdx < deviceCount; ++deviceIdx) {
            hipDeviceProp_t props{};
            if (hipGetDeviceProperties(&props, deviceIdx) != hipSuccess) {
                continue;
            }
            const bool integrated = props.integrated != 0;
            const std::uint64_t totalMemory = static_cast<std::uint64_t>(props.totalGlobalMem);
            if (!integrated && (!foundDiscrete || totalMemory > bestMemory)) {
                bestDevice = deviceIdx;
                bestMemory = totalMemory;
                foundDiscrete = true;
            }
        }

        if (foundDiscrete) {
            return bestDevice;
        }
        return 0;
    }();
    return cached;
#else
    return std::nullopt;
#endif
}

std::string effectiveVisibleDeviceBinding() {
    if (const auto overrideDevices = readNonEmptyEnv("AVE_MIGRAPHX_VISIBLE_DEVICES");
        overrideDevices.has_value()) {
        return *overrideDevices;
    }
    if (const auto rocrVisible = readNonEmptyEnv("ROCR_VISIBLE_DEVICES"); rocrVisible.has_value()) {
        return *rocrVisible;
    }
    if (const auto hipVisible = readNonEmptyEnv("HIP_VISIBLE_DEVICES"); hipVisible.has_value()) {
        return *hipVisible;
    }
    if (const auto preferredDevice = preferredAmdDeviceIndexFromSettings(); preferredDevice.has_value()) {
        return std::to_string(*preferredDevice);
    }
    if (const auto autoDevice = autoSelectedAmdDeviceIndex(); autoDevice.has_value()) {
        return std::to_string(*autoDevice);
    }
    return "all";
}

std::uint64_t fnv1a64(const std::string& input) {
    std::uint64_t hash = 1469598103934665603ull;
    for (const char rawCh : input) {
        const auto ch = static_cast<unsigned char>(rawCh);
        hash ^= static_cast<std::uint64_t>(ch);
        hash *= 1099511628211ull;
    }
    return hash;
}

std::string hexFingerprint(std::uint64_t value) {
    std::ostringstream out;
    out << std::hex << value;
    return out.str();
}

std::string buildMiGraphXRuntimeFingerprint(const std::string& profileLabel,
                                            const std::string& rocmVersion,
                                            const std::string& gfxTarget,
                                            const std::string& visibleDevices,
                                            const std::string& disableMlir,
                                            const std::string& enableNhwc,
                                            const std::string& enableCk,
                                            const std::string& miopenFindMode,
                                            const std::string& miopenCompileParallelLevel) {
    std::ostringstream seed;
    seed << "profile=" << profileLabel
         << "|rocm=" << rocmVersion
         << "|gfx=" << gfxTarget
         << "|visible=" << visibleDevices
         << "|disable_mlir=" << disableMlir
         << "|enable_nhwc=" << enableNhwc
         << "|enable_ck=" << enableCk
         << "|miopen_find_mode=" << miopenFindMode
         << "|miopen_parallel=" << miopenCompileParallelLevel;
    return hexFingerprint(fnv1a64(seed.str()));
}

std::string defaultAveCacheDir() {
    const char* home = std::getenv("HOME");
    if (home != nullptr) {
        return std::string(home) + "/.cache/ave";
    }
    return "/tmp/ave_cache";
}

std::filesystem::path defaultMiGraphXProblemCachePath(const std::string& fingerprint) {
    return std::filesystem::path(defaultAveCacheDir()) / "migraphx" / "contexts" / fingerprint / "problem_cache.json";
}

std::filesystem::path defaultMiopenUserDbPath(const std::string& fingerprint) {
    return std::filesystem::path(defaultAveCacheDir()) / "migraphx" / "contexts" / fingerprint / "miopen_user_db";
}

std::filesystem::path defaultMiopenCacheDir(const std::string& fingerprint) {
    return std::filesystem::path(defaultAveCacheDir()) / "migraphx" / "contexts" / fingerprint / "miopen_cache";
}

int defaultMiopenCompileParallelLevel() {
    unsigned int hwThreads = std::thread::hardware_concurrency();
    if (hwThreads == 0u) {
        hwThreads = 4u;
    }
    // Use full available thread count — MiOpen JIT compilation is CPU-bound and
    // dominates the one-time compile phase; halving was unnecessarily conservative.
    return static_cast<int>(std::min(hwThreads,
                                     static_cast<unsigned int>(kDefaultMiopenCompileParallelCap)));
}

MiGraphXCompileProfile defaultMiGraphXCompileProfile() {
    const auto rawProfile = readNonEmptyEnv("AVE_MIGRAPHX_COMPILE_PROFILE");
    if (!rawProfile.has_value()) {
        return MiGraphXCompileProfile::Balanced;
    }

    std::string value = *rawProfile;
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    if (value == "balanced" || value == "default" || value == "dynamic_hybrid") {
        return MiGraphXCompileProfile::Balanced;
    }
    if (value == "exhaustive" || value == "full" || value == "normal") {
        return MiGraphXCompileProfile::Exhaustive;
    }
    return MiGraphXCompileProfile::Fast;
}

std::string miGraphXCompileProfileLabel(MiGraphXCompileProfile profile) {
    switch (profile) {
        case MiGraphXCompileProfile::Fast: return "fast";
        case MiGraphXCompileProfile::Balanced: return "balanced";
        case MiGraphXCompileProfile::Exhaustive: return "exhaustive";
    }
    return "fast";
}

std::string defaultMiopenFindMode(MiGraphXCompileProfile profile) {
    switch (profile) {
        case MiGraphXCompileProfile::Fast: return "FAST";
        case MiGraphXCompileProfile::Balanced: return "DYNAMIC_HYBRID";
        case MiGraphXCompileProfile::Exhaustive: return "NORMAL";
    }
    return "FAST";
}

std::string defaultMiGraphXEnableNhwc(MiGraphXCompileProfile profile) {
    switch (profile) {
        case MiGraphXCompileProfile::Balanced:
        case MiGraphXCompileProfile::Exhaustive:
            return "1";
        case MiGraphXCompileProfile::Fast:
            return "0";
    }
    return "0";
}

std::string defaultMiGraphXEnableCk(MiGraphXCompileProfile profile) {
    switch (profile) {
        case MiGraphXCompileProfile::Balanced:
        case MiGraphXCompileProfile::Exhaustive:
            return "1";
        case MiGraphXCompileProfile::Fast:
            return "0";
    }
    return "0";
}

void appendEffectiveEnv(MiGraphXDriverEnv& env,
                        const std::string& name,
                        const std::string& value) {
    env.effective.emplace_back(name, value);
    const char* current = std::getenv(name.c_str());
    if (current == nullptr || value != current) {
        env.overrides.emplace_back(name, value);
    }
}

MiGraphXDriverEnv buildMiGraphXDriverEnv() {
    MiGraphXDriverEnv env;
    env.profile = defaultMiGraphXCompileProfile();
    env.profileLabel = miGraphXCompileProfileLabel(env.profile);

    // Use process-wide cached values — avoids repeated file reads and
    // rocminfo subprocess launches on every compile/validate call.
    env.rocmVersion = detectRocmVersion();
    env.gfxTarget = detectGfxTargetForIdentity();
    const std::string rocmVersion = env.rocmVersion;
    const std::string gfxTarget = env.gfxTarget;
    env.disableMlir = readNonEmptyEnv("MIGRAPHX_DISABLE_MLIR").value_or("0");
    env.enableNhwc = readNonEmptyEnv("MIGRAPHX_ENABLE_NHWC")
        .value_or(defaultMiGraphXEnableNhwc(env.profile));
    env.enableCk = readNonEmptyEnv("MIGRAPHX_ENABLE_CK")
        .value_or(defaultMiGraphXEnableCk(env.profile));
    env.visibleDevices = effectiveVisibleDeviceBinding();

    const std::string defaultFindMode = readNonEmptyEnv("AVE_MIGRAPHX_MIOPEN_FIND_MODE")
        .value_or(readNonEmptyEnv("MIOPEN_FIND_MODE")
                      .value_or(defaultMiopenFindMode(env.profile)));
    const std::string defaultParallelLevel = readNonEmptyEnv("AVE_MIGRAPHX_MIOPEN_COMPILE_PARALLEL_LEVEL")
        .value_or(readNonEmptyEnv("MIOPEN_COMPILE_PARALLEL_LEVEL")
                      .value_or(std::to_string(defaultMiopenCompileParallelLevel())));
    env.runtimeFingerprint = buildMiGraphXRuntimeFingerprint(
        env.profileLabel, rocmVersion, gfxTarget, env.visibleDevices,
        env.disableMlir, env.enableNhwc, env.enableCk,
        defaultFindMode, defaultParallelLevel);

    const std::string problemCache = readNonEmptyEnv("AVE_MIGRAPHX_PROBLEM_CACHE")
        .value_or(readNonEmptyEnv("MIGRAPHX_PROBLEM_CACHE")
                      .value_or(defaultMiGraphXProblemCachePath(env.runtimeFingerprint).string()));
    const std::string miopenUserDb = readNonEmptyEnv("AVE_MIGRAPHX_MIOPEN_USER_DB_PATH")
        .value_or(readNonEmptyEnv("MIOPEN_USER_DB_PATH")
                      .value_or(defaultMiopenUserDbPath(env.runtimeFingerprint).string()));
    const std::string miopenCache = readNonEmptyEnv("AVE_MIGRAPHX_MIOPEN_CACHE_DIR")
        .value_or(readNonEmptyEnv("MIOPEN_CUSTOM_CACHE_DIR")
                      .value_or(defaultMiopenCacheDir(env.runtimeFingerprint).string()));
    const std::string miopenFindMode = readNonEmptyEnv("AVE_MIGRAPHX_MIOPEN_FIND_MODE")
        .value_or(readNonEmptyEnv("MIOPEN_FIND_MODE")
                      .value_or(defaultMiopenFindMode(env.profile)));
    const std::string miopenCompileParallel = readNonEmptyEnv("AVE_MIGRAPHX_MIOPEN_COMPILE_PARALLEL_LEVEL")
        .value_or(readNonEmptyEnv("MIOPEN_COMPILE_PARALLEL_LEVEL")
                      .value_or(std::to_string(defaultMiopenCompileParallelLevel())));

    env.problemCachePath = problemCache;
    env.miopenUserDbPath = miopenUserDb;
    env.miopenCustomCacheDir = miopenCache;
    env.miopenFindMode = miopenFindMode;
    env.miopenCompileParallelLevel = miopenCompileParallel;

    ensureDir(std::filesystem::path(problemCache).parent_path());
    ensureDir(std::filesystem::path(miopenUserDb));
    ensureDir(std::filesystem::path(miopenCache));

    appendEffectiveEnv(env, "MIGRAPHX_DISABLE_MLIR", env.disableMlir);
    appendEffectiveEnv(env, "MIGRAPHX_ENABLE_NHWC", env.enableNhwc);
    appendEffectiveEnv(env, "MIGRAPHX_ENABLE_CK", env.enableCk);
    appendEffectiveEnv(env, "MIGRAPHX_PROBLEM_CACHE", problemCache);
    appendEffectiveEnv(env, "MIOPEN_USER_DB_PATH", miopenUserDb);
    appendEffectiveEnv(env, "MIOPEN_CUSTOM_CACHE_DIR", miopenCache);
    appendEffectiveEnv(env, "MIOPEN_FIND_MODE", miopenFindMode);
    appendEffectiveEnv(env, "MIOPEN_COMPILE_PARALLEL_LEVEL", miopenCompileParallel);

    if (env.visibleDevices != "all") {
        appendEffectiveEnv(env, "ROCR_VISIBLE_DEVICES", env.visibleDevices);
        appendEffectiveEnv(env, "HIP_VISIBLE_DEVICES", env.visibleDevices);
    }

    return env;
}

obs::ArtifactManifestFields buildArtifactManifestFields(const std::filesystem::path& sourcePath,
                                                       ModelPrecision precision,
                                                       const MiGraphXDriverEnv& env,
                                                       const bool offloadCopy) {
    obs::ArtifactManifestFields fields;
    fields.manifestSchemaVersion = kArtifactManifestSchemaVersion;
    fields.migraphxVersion = readNonEmptyEnv("MIGRAPHX_VERSION").value_or("unknown");
    // Reuse the system identity already captured in env — no redundant subprocess/file reads.
    fields.rocmVersion = env.rocmVersion;
    fields.gpuGfxTarget = env.gfxTarget;
    std::error_code ec;
    if (std::filesystem::exists(sourcePath, ec)) {
        fields.onnxFileSizeStr = std::to_string(std::filesystem::file_size(sourcePath, ec));
        const auto mtime = std::filesystem::last_write_time(sourcePath, ec);
        const auto mtimeSec = std::chrono::duration_cast<std::chrono::seconds>(mtime.time_since_epoch()).count();
        fields.onnxMtimeStr = std::to_string(mtimeSec);
    } else {
        fields.onnxFileSizeStr = "0";
        fields.onnxMtimeStr = "0";
    }
    fields.sourceFingerprint = obs::buildArtifactSourceFingerprint(sourcePath.string());
    fields.offloadCopy = offloadCopy ? "1" : "0";
    fields.precision = compilePrecisionTag(precision);
    fields.compileProfile = env.profileLabel;
    fields.disableMlir = env.disableMlir;
    fields.enableNhwc = env.enableNhwc;
    fields.enableCk = env.enableCk;
    fields.problemCachePath = env.problemCachePath;
    fields.miopenUserDbPath = env.miopenUserDbPath;
    fields.miopenCustomCacheDir = env.miopenCustomCacheDir;
    fields.miopenFindMode = env.miopenFindMode;
    fields.miopenCompileParallelLevel = env.miopenCompileParallelLevel;
    fields.visibleDevices = env.visibleDevices;
    fields.runtimeFingerprint = env.runtimeFingerprint;
    return fields;
}

std::filesystem::path manifestPathForArtifact(const std::filesystem::path& artifactPath) {
    return artifactPath.string() + ".manifest";
}

std::optional<std::filesystem::path> validateCompiledArtifactCandidate(
        const ManagedModel& model,
        const std::filesystem::path& preparedDir,
        const std::filesystem::path& artifactPath) {
    if (!fileExists(artifactPath) ||
        model.downloadedPath.empty() ||
        model.downloadedPath == "(builtin)") {
        return std::nullopt;
    }

    const auto descriptor = parseCompiledArtifactDescriptor(artifactPath, model.entry.id);
    if (!descriptor.has_value()) {
        return std::nullopt;
    }

    const auto sourcePath = model.entry.migraphxOnnxTransform == MiGraphXOnnxTransform::None
        ? std::filesystem::path(model.downloadedPath)
        : preparedMiGraphXOnnxPath(preparedDir, model.entry.id);
    const MiGraphXDriverEnv driverEnv = buildMiGraphXDriverEnv();
    const auto expectedFields = buildArtifactManifestFields(
        sourcePath, descriptor->precision, driverEnv, migraphxOffloadCopyEnabled());
    std::string mismatchReason;
    if (!obs::validateArtifactManifest(manifestPathForArtifact(artifactPath).string(),
                                       expectedFields,
                                       mismatchReason)) {
        return std::nullopt;
    }
    return artifactPath;
}

std::string formatMiGraphXDriverEnv(const MiGraphXDriverEnv& env) {
    std::ostringstream out;
    out << "profile=" << miGraphXCompileProfileLabel(env.profile);
    for (const auto& [name, value] : env.effective) {
        out << "\n" << name << "=" << value;
    }
    return out.str();
}

std::string formatCompileDimensions(int width, int height) {
    return std::to_string(width) + "x" + std::to_string(height);
}

bool canUseMiGraphXTileFallbackLadder(int width, int height) {
    return width == height && width > kMinimumMiGraphXTileFallbackExtent;
}

std::vector<int> buildMiGraphXTileFallbackExtents(int requestedExtent) {
    std::vector<int> out;
    out.push_back(requestedExtent);
    for (const int candidate : kMiGraphXTileFallbackExtents) {
        if (candidate >= requestedExtent || candidate < kMinimumMiGraphXTileFallbackExtent) {
            continue;
        }
        if (candidate == out.back()) {
            continue;
        }
        out.push_back(candidate);
    }
    return out;
}

// ─── CURL download helper ─────────────────────────────────────────
#ifdef AVE_HAVE_CURL

struct CurlWriteCtx {
    std::ofstream*           file  = nullptr;
    std::atomic<bool>*       cancel = nullptr;
};

static std::size_t curlWrite(void* ptr, std::size_t size, std::size_t nmemb, void* userData) {
    auto* ctx = static_cast<CurlWriteCtx*>(userData);
    if (ctx->cancel && ctx->cancel->load()) { return 0; }
    const std::size_t bytes = size * nmemb;
    ctx->file->write(static_cast<const char*>(ptr), static_cast<std::streamsize>(bytes));
    return bytes;
}

struct CurlProgressCtx {
    ModelProgressCb          cb;
    std::string              modelId;
    std::atomic<bool>*       cancel = nullptr;
};

static int curlProgress(void* clientP, curl_off_t dltotal, curl_off_t dlnow,
                        curl_off_t /*ultotal*/, curl_off_t /*ulnow*/) {
    auto* ctx = static_cast<CurlProgressCtx*>(clientP);
    if (ctx->cancel && ctx->cancel->load()) { return 1; }
    if (dltotal > 0 && ctx->cb) {
        const float progress = static_cast<float>(dlnow) / static_cast<float>(dltotal);
        ctx->cb(ctx->modelId, progress, "Downloading…");
    }
    return 0;
}

bool curlDownload(const std::string& url, const std::filesystem::path& destPath,
                  const ModelProgressCb& progressCb, const std::string& modelId,
                  std::atomic<bool>& cancelFlag, std::string& error) {
    CURL* curl = curl_easy_init();
    if (curl == nullptr) {
        error = "Failed to initialise libcurl.";
        return false;
    }

    std::ofstream outFile(destPath, std::ios::binary | std::ios::trunc);
    if (!outFile.is_open()) {
        curl_easy_cleanup(curl);
        error = "Cannot open destination file for writing: " + destPath.string();
        return false;
    }

    CurlWriteCtx    writeCtx{&outFile, &cancelFlag};
    CurlProgressCtx progressCtx{progressCb, modelId, &cancelFlag};

    curl_easy_setopt(curl, CURLOPT_URL,              url.c_str());
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION,   1L);
    curl_easy_setopt(curl, CURLOPT_NOPROGRESS,       0L);
    curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, curlProgress);
    curl_easy_setopt(curl, CURLOPT_XFERINFODATA,     &progressCtx);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION,    curlWrite);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA,        &writeCtx);
    curl_easy_setopt(curl, CURLOPT_USERAGENT,        "AMD Video Enhancer/1.0");

    const CURLcode res = curl_easy_perform(curl);

    // ── Check HTTP status code before cleaning up ──────────────
    long httpCode = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &httpCode);

    curl_easy_cleanup(curl);
    outFile.close();

    if (cancelFlag.load()) {
        std::error_code ec;
        std::filesystem::remove(destPath, ec);
        error = "Download cancelled.";
        return false;
    }

    if (res != CURLE_OK) {
        std::error_code ec;
        std::filesystem::remove(destPath, ec);
        error = std::string("libcurl error: ") + curl_easy_strerror(res);
        return false;
    }

    // ── Validate HTTP response code ────────────────────────────
    if (httpCode != 200) {
        std::error_code ec;
        std::filesystem::remove(destPath, ec);
        error = "HTTP error " + std::to_string(httpCode) +
                " while downloading " + url;
        return false;
    }

    // ── Validate downloaded file size (models are always > 1 KB) ─
    {
        std::error_code ec;
        const auto fileSize = std::filesystem::file_size(destPath, ec);
        if (ec || fileSize < 1024) {
            std::filesystem::remove(destPath, ec);
            error = "Downloaded file is too small (" +
                    std::to_string(ec ? 0 : fileSize) +
                    " bytes); the URL may be invalid: " + url;
            return false;
        }
    }

    return true;
}

#else  // AVE_HAVE_CURL

bool curlDownload(const std::string& url, const std::filesystem::path& destPath,
                  const ModelProgressCb& progressCb, const std::string& modelId,
                  std::atomic<bool>& cancelFlag, std::string& error) {
    (void)progressCb; (void)modelId; (void)cancelFlag; (void)destPath;
    error = "libcurl is not compiled in.  Cannot download " + url +
            "\n\nPlease place the model file manually at:\n  " + destPath.string();
    return false;
}

#endif // AVE_HAVE_CURL

// ─── Zip archive extraction ──────────────────────────────────────
// Requires `unzip` on PATH (standard on all Linux distributions).
bool extractFromZip(const std::filesystem::path& zipPath,
                    const std::string& internalPath,
                    const std::filesystem::path& destFile,
                    std::string& error) {
    if (!process_observer::commandInPath("unzip")) {
        error = "'unzip' not found in PATH – cannot extract from archive.";
        return false;
    }

    // Extract with -j (junk paths) into a temp directory next to the zip.
    const std::filesystem::path tempDir = zipPath.parent_path() /
        (zipPath.stem().string() + "_extract_" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    std::error_code ec;
    std::filesystem::create_directories(tempDir, ec);

    std::ostringstream cmd;
    cmd << "unzip -o -j "
        << "\"" << zipPath.string() << "\" "
        << "\"" << internalPath     << "\" "
        << "-d \"" << tempDir.string() << "\"";

    const int rc = std::system(cmd.str().c_str());
    if (rc != 0) {
        std::filesystem::remove_all(tempDir, ec);
        error = "unzip failed (exit " + std::to_string(rc) +
                ") extracting '" + internalPath + "' from archive.";
        return false;
    }

    const auto extracted = tempDir /
        std::filesystem::path(internalPath).filename();
    if (!fileExists(extracted)) {
        std::filesystem::remove_all(tempDir, ec);
        error = "Extracted file not found after unzip: " + extracted.string();
        return false;
    }

    std::filesystem::rename(extracted, destFile, ec);
    if (ec) {
        // Fallback: copy then remove
        std::filesystem::copy_file(extracted, destFile,
            std::filesystem::copy_options::overwrite_existing, ec);
        std::filesystem::remove(extracted, ec);
    }
    std::filesystem::remove_all(tempDir, ec);
    return true;
}

// ─── Quantized-op pre-flight scanner ────────────────────────────
// MiGraphX cannot lower quantized operators (QLinearConv, etc.) and will
// call abort() — exit 134 (SIGABRT) — instead of returning a clean non-zero
// code.  We scan the ONNX binary for known quantized op-type strings BEFORE
// invoking migraphx-driver so we can surface a clear ModelIncompatible error.
//
// Detection is via raw byte search (no protobuf library required).  The
// op_type values are stored as plain ASCII bytes inside the protobuf encoding,
// so a substring scan is reliable for these distinctive long strings.
// Reads only the first kScanBytes of the file — op defs appear well within the
// first few MiB for all typical SR / interpolation / denoising models.
static const std::array<const char*, 9> kQuantizedOpMarkers = {{
    "QLinearConv",
    "QLinearMatMul",
    "QLinearAdd",
    "QuantizeLinear",
    "DequantizeLinear",
    "DynamicQuantizeLinear",
    "ConvInteger",
    "MatMulInteger",
    "QGemm",
}};

std::vector<std::string> scanOnnxQuantizedOps(
        const std::filesystem::path& onnxPath) {
    constexpr std::size_t kScanBytes = 4ULL * 1024 * 1024;  // 4 MiB
    std::vector<std::string> found;

    std::ifstream f(onnxPath, std::ios::binary);
    if (!f.is_open()) { return found; }

    std::vector<char> buf(kScanBytes);
    f.read(buf.data(), static_cast<std::streamsize>(buf.size()));
    const auto bytesRead = static_cast<std::size_t>(f.gcount());

    for (const char* op : kQuantizedOpMarkers) {
        std::string opStr(op);
        const auto it = std::search(
            buf.data(), buf.data() + bytesRead,
            opStr.data(), opStr.data() + opStr.size());
        if (it != buf.data() + bytesRead) {
            found.emplace_back(op);
        }
    }
    return found;
}

void appendDriverDimParamFallback(std::ostringstream& cmd,
                                  int compileWidth,
                                  int compileHeight,
                                  int compileBatch) {
    cmd << " --batch " << compileBatch
        << " --dim-param @batch_size " << compileBatch
        << " --dim-param @b " << compileBatch
        << " --dim-param @n " << compileBatch
        << " --dim-param @width " << compileWidth
        << " --dim-param @w " << compileWidth
        << " --dim-param @x " << compileWidth
        << " --dim-param @cols " << compileWidth
        << " --dim-param @height " << compileHeight
        << " --dim-param @h " << compileHeight
        << " --dim-param @y " << compileHeight
        << " --dim-param @rows " << compileHeight;

    if (compileWidth == compileHeight) {
        const std::string fixedDynDim =
            "{min:" + std::to_string(compileWidth) +
            ", max:" + std::to_string(compileWidth) +
            ", optimals:[" + std::to_string(compileWidth) + "]}";
        cmd << " --default-dyn-dim " << shellQuote(fixedDynDim);
    }
}

bool compileWithMigraphxDriver(const std::filesystem::path& onnxPath,
                               const std::filesystem::path& mxrPath,
                               const std::filesystem::path& manifestSourcePath,
                               int compileWidth,
                               int compileHeight,
                               int compileBatch,
                               ModelPrecision compilePrecision,
                               const ModelProgressCb& progressCb,
                               const std::string& modelId,
                               std::string& error) {
    const auto driverPath = resolveMiGraphXDriver();
    if (!driverPath.has_value()) {
        error = "migraphx-driver not found in PATH or bundled install. ROCm must be installed.";
        return false;
    }

    if (!isSupportedMiGraphXCompilePrecision(compilePrecision)) {
        error = "MiGraphX compile precision '" + compilePrecisionTag(compilePrecision)
              + "' is not supported by this app yet.";
        return false;
    }
    if (compilePrecision == ModelPrecision::Int8) {
        error = "MiGraphX int8 compilation is not available in the app runtime. "
                "Use an offline compiler workflow that can provide explicit calibration data.";
        return false;
    }

    const MiGraphXDriverEnv driverEnv = buildMiGraphXDriverEnv();
    const std::string driverEnvSummary = formatMiGraphXDriverEnv(driverEnv);
    const auto manifestFields = buildArtifactManifestFields(
        manifestSourcePath, compilePrecision, driverEnv, migraphxOffloadCopyEnabled());
    obs::logMiGraphXEnvironment(manifestFields, "compile", mxrPath.string(), "not-run");

    if (progressCb) {
        progressCb(modelId,
                   0.02f,
                   "Preparing symbolic input dimensions for migraphx-driver…");
    }

    std::ostringstream cmd;
    cmd << shellQuote(driverPath->string()) << " compile"
        << " --onnx " << shellQuote(onnxPath.string())
        << " --gpu"
        << " --output " << shellQuote(mxrPath.string());
    if (migraphxOffloadCopyEnabled()) {
        cmd << " --enable-offload-copy";
    }
    if (compilePrecision == ModelPrecision::Fp16) {
        cmd << " --fp16";
    }
    // Pass --exhaustive-tune to the driver for Exhaustive profile.
    // This was previously only applied in the C++ library fallback path,
    // leaving the driver path without exhaustive MiOpen solver search.
    if (driverEnv.profile == MiGraphXCompileProfile::Exhaustive) {
        cmd << " --exhaustive-tune";
    }

    appendDriverDimParamFallback(cmd, compileWidth, compileHeight, compileBatch);

    const char* timeoutEnv = std::getenv("AVE_MIGRAPHX_COMPILE_TIMEOUT_SEC");
    int timeoutSeconds = 45 * 60;
    if (timeoutEnv != nullptr) {
        try {
            timeoutSeconds = std::stoi(timeoutEnv);
        } catch (...) {
            timeoutSeconds = 45 * 60;
        }
    }
    if (timeoutSeconds < 60) { timeoutSeconds = 60; }

    const bool timeoutAvailable = process_observer::commandInPath("timeout");
    std::ostringstream wrappedCmd;
    if (timeoutAvailable) {
        wrappedCmd << "timeout --foreground " << timeoutSeconds << "s " << cmd.str();
    } else {
        wrappedCmd << cmd.str();
    }
    const auto compilerEnv = mergeEnvOverrides(
        bundledMiGraphXCompilerEnvOverrides(),
        driverEnv.overrides);
    const std::string cmdStr = wrappedCmd.str() + " 2>&1";

    if (progressCb) {
        progressCb(modelId, 0.03f, "Using symbolic dimension mapping for migraphx-driver…");
    }
    if (progressCb) {
        progressCb(modelId, 0.05f,
            "Launching migraphx-driver (" + miGraphXCompileProfileLabel(driverEnv.profile)
            + " compile profile)…");
    }

    std::atomic<bool> compileDone{false};
    std::atomic<int> compileExit{-1};
    std::string lastOutputLine;
    std::string capturedOutput;
    std::mutex outputMtx;

    std::thread compileThread([&]() {
        const int rc = process_observer::runObservedCommand(
            cmdStr,
            compilerEnv,
            [&](const std::string& line) {
                if (line.empty()) {
                    return;
                }
                std::lock_guard<std::mutex> lk(outputMtx);
                lastOutputLine = line;
                if (capturedOutput.size() < 16384) {
                    capturedOutput.append(line);
                    capturedOutput.push_back('\n');
                }
            });
        compileExit.store(rc);
        compileDone.store(true);
    });

    // Poll at 500 ms for fast completion detection; report progress every 2 s
    // to avoid flooding the GUI with updates during a 10-45 minute compile.
    int elapsedMs = 0;
    while (!compileDone.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        if (compileDone.load()) { break; }
        elapsedMs += 500;
        if (elapsedMs % 2000 != 0) { continue; }
        if (progressCb) {
            std::string line;
            {
                std::lock_guard<std::mutex> lk(outputMtx);
                line = lastOutputLine;
            }
            const int elapsedSec = elapsedMs / 1000;
            const int mins = elapsedSec / 60;
            const int secs = elapsedSec % 60;
            std::ostringstream status;
            status << "migraphx-driver compiling… "
                   << mins << "m " << secs << "s elapsed";
            if (!line.empty()) { status << " — " << line; }
            progressCb(modelId, -1.0f, status.str());
        }
    }
    compileThread.join();

    const int rawRc = compileExit.load();
    int exitCode = rawRc;
    int signalNum = 0;
#if defined(__unix__) || defined(__APPLE__)
    if (WIFEXITED(rawRc)) {
        exitCode = WEXITSTATUS(rawRc);
    } else if (WIFSIGNALED(rawRc)) {
        signalNum = WTERMSIG(rawRc);
        exitCode = 128 + signalNum;
    }
#endif

    if (exitCode != 0) {
        std::ostringstream msg;
        if (signalNum > 0) {
            msg << "migraphx-driver killed by signal " << signalNum;
            if (signalNum == 6) {
                msg << " (SIGABRT). This usually means MiGraphX hit an unsupported operator or internal assertion.";
            } else if (signalNum == 11) {
                msg << " (SIGSEGV).";
            } else {
                msg << '.';
            }
        } else if (timeoutAvailable && exitCode == 124) {
            msg << "migraphx-driver timed out after " << timeoutSeconds
                << " seconds. Set AVE_MIGRAPHX_COMPILE_TIMEOUT_SEC to a larger value if this model really needs longer.";
        } else if (rawRc == -1) {
            msg << "Failed to launch migraphx-driver.";
        } else {
            msg << "migraphx-driver exited with code " << exitCode << '.';
        }
        std::string driverOutput;
        {
            std::lock_guard<std::mutex> lk(outputMtx);
            driverOutput = capturedOutput.empty() ? lastOutputLine : capturedOutput;
        }
        msg << "\nCompile environment:\n" << driverEnvSummary;
        if (!driverOutput.empty()) {
            msg << "\nDriver output:\n" << driverOutput;
        }
        error = msg.str();
        if (progressCb) {
            progressCb(modelId, 0.0f,
                "Compilation failed (exit " + std::to_string(exitCode) + ")");
        }
        return false;
    }

    if (!fileExists(mxrPath)) {
        error = "migraphx-driver ran but output file not found: " + mxrPath.string();
        if (progressCb) { progressCb(modelId, 0.0f, "Compilation failed: output .mxr missing"); }
        return false;
    }

    std::string manifestError;
    if (!obs::writeArtifactManifest(manifestPathForArtifact(mxrPath).string(),
                                    manifestFields,
                                    manifestError)) {
        std::error_code removeEc;
        std::filesystem::remove(mxrPath, removeEc);
        error = "Compiled artifact could not be validated for reuse: " + manifestError;
        if (progressCb) { progressCb(modelId, 0.0f, "Compilation failed: manifest write"); }
        return false;
    }

    if (progressCb) { progressCb(modelId, 1.0f, "Compilation complete."); }
    return true;
}

// ─── MiGraphX compilation ─────────────────────────────────────────
// Uses the migraphx-driver command-line tool if present.
// The driver is shipped with ROCm and provides:
//   migraphx-driver compile --onnx <in.onnx> --output <out.mxr>
bool migraphxCompile(const std::filesystem::path& onnxPath,
                     const std::filesystem::path& mxrPath,
                     const std::filesystem::path& manifestSourcePath,
                     int compileWidth,
                     int compileHeight,
                     int compileBatch,
                     ModelPrecision compilePrecision,
                     std::optional<std::string> calibrationVideoPath,
                     const ModelProgressCb& progressCb,
                     const std::string& modelId,
                     std::string& error) {
    // ── Pre-flight: quantized-op scan ────────────────────────────
    // Run before compilation to detect quantized ops MiGraphX cannot lower.
    {
        const auto quantOps = scanOnnxQuantizedOps(onnxPath);
        if (!quantOps.empty()) {
            std::ostringstream msg;
            msg << "Model contains quantized operators not supported by MiGraphX: ";
            for (std::size_t i = 0; i < quantOps.size(); ++i) {
                if (i > 0) { msg << ", "; }
                msg << quantOps[i];
            }
            msg << ".\n"
                << "Action: use a non-quantized ONNX export before compiling with MiGraphX.";
            error = msg.str();
            if (progressCb) {
                progressCb(modelId, 0.0f,
                    "Unsupported: quantized model (" + quantOps[0] + " …)");
            }
            return false;
        }
    }

    if (compileWithMigraphxDriver(onnxPath, mxrPath, manifestSourcePath,
                                  compileWidth, compileHeight,
                                  compileBatch,
                                  compilePrecision,
                                  progressCb, modelId, error)) {
        return true;
    }

    (void)calibrationVideoPath;
    return false;
}

// ─────────────────────────────────────────────────────────────────
// torchExportToOnnx — convert a TorchScript .pt / .pth model to ONNX
// using the ambient Python + torch environment (torch-MiGraphX stack).
//
// Writes a dynamic-axes ONNX (opset 17, NCHW 1×3×H×W) to onnxDest.
// Tries torch.jit.load() first; falls back to torch.load() for plain
// state-dict / nn.Module checkpoints.
// ─────────────────────────────────────────────────────────────────
bool torchExportToOnnx(const std::filesystem::path& pthPath,
                       const std::filesystem::path& onnxDest,
                       std::string& error) {
    if (!process_observer::commandInPath("python3")) {
        error = "python3 not found in PATH; cannot convert PyTorch model. "
                "Install Python + torch: pip install torch";
        return false;
    }

    // Inline Python that handles both TorchScript and nn.Module checkpoints.
    const std::string pyScript =
        "import sys, torch\n"
        "pth, out = sys.argv[1], sys.argv[2]\n"
        "try:\n"
        "    model = torch.jit.load(pth, map_location='cpu')\n"
        "except Exception:\n"
        "    ckpt = torch.load(pth, map_location='cpu', weights_only=False)\n"
        "    model = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt\n"
        "if hasattr(model, 'eval'): model.eval()\n"
        "dummy = torch.zeros(1, 3, 256, 256)\n"
        "torch.onnx.export(\n"
        "    model, dummy, out,\n"
        "    input_names=['input'], output_names=['output'],\n"
        "    dynamic_axes={'input':{0:'b',2:'h',3:'w'},'output':{0:'b',2:'h',3:'w'}},\n"
        "    opset_version=17)\n"
        "print('OK: exported to', out)\n";

    const auto scriptPath = std::filesystem::temp_directory_path()
                          / "ave_torch_export.py";
    {
        std::ofstream sf(scriptPath);
        if (!sf) {
            error = "Cannot write temp export script to " + scriptPath.string();
            return false;
        }
        sf << pyScript;
    }

    std::ostringstream cmd;
    cmd << "python3 " << scriptPath.string()
        << " \"" << pthPath.string()  << "\""
        << " \"" << onnxDest.string() << "\"";
    const int rc = std::system(cmd.str().c_str());

    std::error_code ec2;
    std::filesystem::remove(scriptPath, ec2);

    if (rc != 0) {
        error = "torch.onnx.export() failed (exit " + std::to_string(rc)
              + ").  Ensure torch is installed: pip install torch torch_migraphx";
        return false;
    }
    if (!fileExists(onnxDest)) {
        error = "Export ran but ONNX output not found: " + onnxDest.string();
        return false;
    }
    return true;
}

}  // namespace

// ─────────────────────────────────────────────────────────────────
// std::string toString(ModelState)
// ─────────────────────────────────────────────────────────────────
std::string toString(ModelState state) {
    switch (state) {
        case ModelState::NotDownloaded: return "not_downloaded";
        case ModelState::Downloading:   return "downloading";
        case ModelState::Downloaded:    return "downloaded";
        case ModelState::Converting:    return "converting";
        case ModelState::Converted:     return "converted";
        case ModelState::Error:         return "error";
    }
    return "unknown";
}

// ─────────────────────────────────────────────────────────────────
// ModelManager::Impl
// ─────────────────────────────────────────────────────────────────
struct ModelManager::Impl {
    mutable std::mutex                             mtx;
    std::string                                    modelsDir;
    std::optional<std::filesystem::path>           bundledModelsDir = bundledModelsDirectory();
    std::unordered_map<std::string, ManagedModel>  records;   // keyed by modelId
    std::unordered_map<std::string, std::shared_ptr<std::atomic<bool>>> cancelFlags;

    ManagedModel& getOrCreate(const std::string& modelId) {
        auto it = records.find(modelId);
        if (it == records.end()) {
            const ModelEntry* entry = catalogEntryById(modelId);
            assert(entry != nullptr);
            ManagedModel m;
            m.entry = *entry;
            records.emplace(modelId, std::move(m));
        }
        return records.at(modelId);
    }

    std::filesystem::path downloadedDir()  const { return std::filesystem::path(modelsDir) / "downloaded"; }
    std::filesystem::path convertedDir()   const { return std::filesystem::path(modelsDir) / "migraphx"; }
    std::filesystem::path preparedDir()    const { return std::filesystem::path(modelsDir) / "prepared"; }

    std::filesystem::path bundledDownloadedDir() const {
        if (!bundledModelsDir.has_value()) {
            return {};
        }
        return *bundledModelsDir / "downloaded";
    }

    bool isBuiltinModelId(const std::string& modelId) const {
        return catalogEntryById(modelId) != nullptr;
    }

    std::unordered_map<std::string, ModelEntry> scanCustomModelEntries() const {
        std::unordered_map<std::string, ModelEntry> out;
        std::error_code ec;
        const auto dir = downloadedDir();
        if (!std::filesystem::exists(dir, ec)) {
            return out;
        }

        for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
            if (ec || !entry.is_regular_file(ec)) {
                continue;
            }
            const auto path = entry.path();
            if (path.extension() != ".avemodel") {
                continue;
            }

            std::string parseError;
            auto parsed = parseCustomModelManifest(path, parseError);
            if (!parsed.has_value()) {
                std::cerr << "[model] Skipping custom manifest '" << path.string()
                          << "': " << parseError << std::endl;
                continue;
            }
            out[parsed->id] = *parsed;
        }

        return out;
    }

    std::string resolveDownloadedPath(const std::string& filename) const {
        if (filename.empty()) {
            return {};
        }

        const std::filesystem::path candidate(filename);
        if (candidate.is_absolute() && fileExists(candidate)) {
            return candidate.string();
        }

        const auto localPath = downloadedDir() / filename;
        if (fileExists(localPath)) {
            return localPath.string();
        }

        const auto bundledDir = bundledDownloadedDir();
        const auto bundledPath = bundledDir / filename;
        if (!bundledDir.empty() && fileExists(bundledPath)) {
            return bundledPath.string();
        }

        return {};
    }

    void scanAndUpdate(ManagedModel& m) {
        const std::string& id = m.entry.id;

        // Primary downloaded file
        if (!m.entry.filename.empty()) {
            m.downloadedPath = resolveDownloadedPath(m.entry.filename);
        }

        // Auxiliary file (NCNN .bin)
        if (!m.entry.filenameAux.empty()) {
            m.downloadedPathAux = resolveDownloadedPath(m.entry.filenameAux);
        }

        // Compiled .mxr (regular)
        {
            m.convertedPath.clear();
            const std::array<ModelPrecision, 3> preferredPrecisions = {
                ModelPrecision::Fp16,
                ModelPrecision::Int8,
                ModelPrecision::Fp32,
            };
            const auto defaultDims = defaultMiGraphXCompileDims(m.entry);
            for (const auto precision : preferredPrecisions) {
                const auto p = defaultDims.has_value()
                    ? compiledArtifactPath(convertedDir(), id, precision,
                                           defaultDims->first,
                                           defaultDims->second,
                                           1)
                    : compiledArtifactPath(convertedDir(), id, precision);
                if (const auto validated = validateCompiledArtifactCandidate(
                        m, preparedDir(), p); validated.has_value()) {
                    m.convertedPath = validated->string();
                    break;
                }
            }
            if (m.convertedPath.empty()) {
                for (const auto& candidate : findCompiledArtifactsForModel(convertedDir(), id)) {
                    if (const auto validated = validateCompiledArtifactCandidate(
                            m, preparedDir(), candidate); validated.has_value()) {
                        m.convertedPath = validated->string();
                        break;
                    }
                }
            }
        }

        // Derive state (do not overwrite transient states like Downloading)
        if (m.state == ModelState::Downloading ||
            m.state == ModelState::Converting) {
            return;
        }

        if (!m.convertedPath.empty()) {
            m.state = ModelState::Converted;
        } else if (!m.downloadedPath.empty() && hasMxrExtension(m.downloadedPath)) {
            // Downloaded .mxr models are inference-ready without a separate convert step.
            m.state = ModelState::Converted;
        } else if (!m.downloadedPath.empty()) {
            m.state = ModelState::Downloaded;
        } else if (m.entry.filename.empty() && m.entry.downloadUrl.empty()) {
            // Built-in / parametric model – always "Downloaded"
            m.state = ModelState::Downloaded;
            m.downloadedPath = "(builtin)";
        } else {
            m.state = ModelState::NotDownloaded;
        }
    }
};

// ─────────────────────────────────────────────────────────────────
// ModelManager public interface
// ─────────────────────────────────────────────────────────────────
ModelManager::ModelManager() : impl_(std::make_shared<Impl>()) {
    impl_->modelsDir = defaultModelsDir();
    ensureDir(impl_->downloadedDir());
    ensureDir(impl_->convertedDir());
    ensureDir(impl_->preparedDir());

    for (const auto& entry : builtinModelCatalog()) {
        ManagedModel m;
        m.entry = entry;
        impl_->records.emplace(entry.id, std::move(m));
    }

    refresh();
}

ModelManager::~ModelManager() = default;

void ModelManager::setModelsDirectory(const std::string& dir) {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    impl_->modelsDir = dir;
    impl_->bundledModelsDir = bundledModelsDirectory();
    ensureDir(impl_->downloadedDir());
    ensureDir(impl_->convertedDir());
    ensureDir(impl_->preparedDir());
}

std::string ModelManager::modelsDirectory() const {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    return impl_->modelsDir;
}

void ModelManager::refresh() {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    const auto customEntries = impl_->scanCustomModelEntries();

    for (auto it = impl_->records.begin(); it != impl_->records.end();) {
        const bool builtin = impl_->isBuiltinModelId(it->first);
        if (!builtin && customEntries.find(it->first) == customEntries.end()) {
            it = impl_->records.erase(it);
            continue;
        }
        ++it;
    }

    for (const auto& [id, entry] : customEntries) {
        auto& record = impl_->records[id];
        record.entry = entry;
    }

    for (auto& [id, record] : impl_->records) {
        impl_->scanAndUpdate(record);
    }
}

std::vector<ManagedModel> ModelManager::allModels() const {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    std::vector<ManagedModel> out;
    out.reserve(impl_->records.size());
    for (const auto& [id, m] : impl_->records) {
        out.push_back(m);
    }
    // Stable order: preserve catalog order
    std::sort(out.begin(), out.end(), [](const ManagedModel& a, const ManagedModel& b) {
        const auto& cat = builtinModelCatalog();
        auto pos = [&](const std::string& id_) {
            for (std::size_t i = 0; i < cat.size(); ++i) {
                if (cat[i].id == id_) return static_cast<int>(i);
            }
            return static_cast<int>(cat.size());
        };
        return pos(a.entry.id) < pos(b.entry.id);
    });
    return out;
}

std::vector<ManagedModel> ModelManager::modelsForStage(StageKind kind) const {
    auto all = allModels();
    all.erase(std::remove_if(all.begin(), all.end(),
                             [kind](const ManagedModel& m) { return m.entry.stage != kind; }),
              all.end());
    return all;
}

std::optional<ManagedModel> ModelManager::findModel(const std::string& modelId) const {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    auto it = impl_->records.find(modelId);
    if (it == impl_->records.end()) { return std::nullopt; }
    return it->second;
}

std::optional<std::string> ModelManager::bestPathForModel(const std::string& modelId) const {
    std::string downloadedPath;
    {
        std::lock_guard<std::mutex> lock(impl_->mtx);
        auto it = impl_->records.find(modelId);
        if (it == impl_->records.end()) { return std::nullopt; }
        downloadedPath = it->second.downloadedPath;
    }

    const std::array<ModelPrecision, 3> preferredPrecisions = {
        ModelPrecision::Fp16,
        ModelPrecision::Int8,
        ModelPrecision::Fp32,
    };
    for (const auto precision : preferredPrecisions) {
        if (const auto validated = validatedCompiledArtifactPath(
                modelId, precision, std::nullopt, std::nullopt, 1, nullptr);
            validated.has_value()) {
            return validated;
        }
    }

    if (!downloadedPath.empty()) {
        return downloadedPath;
    }
    return std::nullopt;
}

std::optional<std::string> ModelManager::validatedCompiledArtifactPath(
        const std::string& modelId,
        ModelPrecision compilePrecision,
        std::optional<std::int64_t> inputWidth,
        std::optional<std::int64_t> inputHeight,
        int compileBatch,
        std::string* validationDetail) const {
    if ((inputWidth.has_value() && !inputHeight.has_value()) ||
        (!inputWidth.has_value() && inputHeight.has_value())) {
        if (validationDetail != nullptr) {
            *validationDetail = "Both inputWidth and inputHeight are required when validating a dimension-specific artifact.";
        }
        return std::nullopt;
    }

    const int batch = normaliseCompileBatch(compileBatch);
    std::filesystem::path artifactPath;
    std::filesystem::path sourcePath;
    {
        std::lock_guard<std::mutex> lock(impl_->mtx);
        auto it = impl_->records.find(modelId);
        if (it == impl_->records.end()) {
            if (validationDetail != nullptr) {
                *validationDetail = "Unknown model id: " + modelId;
            }
            return std::nullopt;
        }

        const auto& model = it->second;
        if (model.downloadedPath.empty() || model.downloadedPath == "(builtin)") {
            if (validationDetail != nullptr) {
                *validationDetail = "Model source is not available on disk for manifest validation.";
            }
            return std::nullopt;
        }

        std::optional<int> artifactWidth;
        std::optional<int> artifactHeight;
        if (const auto defaultDims = defaultMiGraphXCompileDims(model.entry);
            defaultDims.has_value() &&
            !inputWidth.has_value() && !inputHeight.has_value()) {
            artifactWidth = defaultDims->first;
            artifactHeight = defaultDims->second;
        } else if (inputWidth.has_value() && inputHeight.has_value()) {
            if (*inputWidth <= 0 || *inputHeight <= 0 ||
                *inputWidth > static_cast<std::int64_t>(std::numeric_limits<int>::max()) ||
                *inputHeight > static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
                if (validationDetail != nullptr) {
                    *validationDetail = "Invalid artifact dimensions requested for validation.";
                }
                return std::nullopt;
            }
            artifactWidth = static_cast<int>(*inputWidth);
            artifactHeight = static_cast<int>(*inputHeight);
        }

        sourcePath = model.entry.migraphxOnnxTransform == MiGraphXOnnxTransform::None
            ? std::filesystem::path(model.downloadedPath)
            : preparedMiGraphXOnnxPath(impl_->preparedDir(), model.entry.id);
        artifactPath = compiledArtifactPath(
            impl_->convertedDir(), modelId, compilePrecision,
            artifactWidth,
            artifactHeight,
            batch);
    }

    if (!fileExists(artifactPath)) {
        if (validationDetail != nullptr) {
            *validationDetail = "Compiled artifact not found: " + artifactPath.string();
        }
        return std::nullopt;
    }

    const MiGraphXDriverEnv driverEnv = buildMiGraphXDriverEnv();
    const auto expectedFields = buildArtifactManifestFields(
        sourcePath, compilePrecision, driverEnv, migraphxOffloadCopyEnabled());
    std::string mismatchReason;
    if (!obs::validateArtifactManifest(manifestPathForArtifact(artifactPath).string(),
                                       expectedFields,
                                       mismatchReason)) {
        if (validationDetail != nullptr) {
            *validationDetail = mismatchReason;
        }
        return std::nullopt;
    }

    if (validationDetail != nullptr) {
        validationDetail->clear();
    }
    return artifactPath.string();
}

std::vector<ModelManager::MiGraphXCompileProfile>
ModelManager::standardCompileProfilesForModel(const std::string& modelId) const {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    auto it = impl_->records.find(modelId);
    if (it == impl_->records.end()) {
        return {};
    }

    std::vector<MiGraphXCompileProfile> profiles;
    auto appendUniqueProfile = [&](const std::string& label,
                                   const int width,
                                   const int height,
                                   const int batch,
                                   const bool fixedShape) {
        if (width <= 0 || height <= 0 || batch <= 0) {
            return;
        }
        const auto duplicate = std::find_if(
            profiles.begin(), profiles.end(),
            [&](const MiGraphXCompileProfile& profile) {
                return profile.width == width &&
                       profile.height == height &&
                       profile.batch == batch;
            });
        if (duplicate != profiles.end()) {
            return;
        }
        profiles.push_back({label, width, height, batch, fixedShape});
    };

    const auto defaultDims = defaultMiGraphXCompileDims(it->second.entry);
    if (defaultDims.has_value()) {
        appendUniqueProfile("Default profile",
                            defaultDims->first,
                            defaultDims->second,
                            1,
                            true);
        return profiles;
    }

    appendUniqueProfile("1080p", 1920, 1080, 1, false);
    appendUniqueProfile("4K UHD", 3840, 2160, 1, false);
    return profiles;
}

bool ModelManager::prewarmStandardArtifacts(const std::string& modelId,
                                            const ModelProgressCb& progressCb,
                                            const ModelStateCb& stateCb,
                                            std::string& error,
                                            const ModelPrecision compilePrecision) {
    if (!isSupportedMiGraphXCompilePrecision(compilePrecision)) {
        error = "MiGraphX prewarm currently supports fp32 and fp16 exact-shape artifacts.";
        return false;
    }
    if (compilePrecision == ModelPrecision::Int8) {
        error = "MiGraphX prewarm does not support int8 because int8 requires calibration frames from real input video.";
        return false;
    }

    const auto profiles = standardCompileProfilesForModel(modelId);
    if (profiles.empty()) {
        error = "Unknown model id or no standard MiGraphX profiles defined: " + modelId;
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(impl_->mtx);
        auto it = impl_->records.find(modelId);
        if (it == impl_->records.end()) {
            error = "Unknown model id: " + modelId;
            return false;
        }
        const auto& model = it->second;
        if (model.downloadedPath.empty() || model.downloadedPath == "(builtin)") {
            error = "Model not yet downloaded – cannot prewarm MiGraphX artifacts.";
            return false;
        }
        if (model.entry.sourceFormat != ModelFormat::Onnx &&
            model.entry.sourceFormat != ModelFormat::Pytorch) {
            error = "MiGraphX prewarm requires an ONNX or PyTorch source model.";
            return false;
        }
        impl_->records[modelId].state = ModelState::Converting;
    }
    if (stateCb) {
        stateCb(modelId, ModelState::Converting);
    }

    bool ok = true;
    for (std::size_t i = 0; i < profiles.size(); ++i) {
        const auto& profile = profiles[i];
        const float baseProgress =
            static_cast<float>(i) / static_cast<float>(profiles.size());
        const float doneProgress =
            static_cast<float>(i + 1u) / static_cast<float>(profiles.size());
        const std::string profileLabel =
            profile.label + " (" + std::to_string(profile.width) + "x"
            + std::to_string(profile.height) + ")";

        if (progressCb) {
            progressCb(modelId, baseProgress,
                       "Checking MiGraphX cache for " + profileLabel + "…");
        }

        std::string validationDetail;
        if (validatedCompiledArtifactPath(modelId,
                                          compilePrecision,
                                          profile.width,
                                          profile.height,
                                          profile.batch,
                                          &validationDetail).has_value()) {
            if (progressCb) {
                progressCb(modelId, doneProgress,
                           "Reused cached MiGraphX artifact for " + profileLabel + ".");
            }
            continue;
        }

        std::string compileError;
        const auto artifact = autoCompileForInference(modelId,
                                                      compileError,
                                                      profile.width,
                                                      profile.height,
                                                      compilePrecision,
                                                      profile.batch);
        if (!artifact.has_value()) {
            error = "Failed to prewarm " + profileLabel + ": " + compileError;
            ok = false;
            break;
        }
        if (progressCb) {
            progressCb(modelId, doneProgress,
                       "Prepared MiGraphX artifact for " + profileLabel + ".");
        }
    }

    ModelState finalState = ModelState::Error;
    {
        std::lock_guard<std::mutex> lock(impl_->mtx);
        auto& model = impl_->records[modelId];
        if (ok) {
            model.state = ModelState::Downloaded;
            impl_->scanAndUpdate(model);
        } else {
            model.state = ModelState::Error;
            model.errorMessage = error;
        }
        finalState = model.state;
    }
    if (stateCb) {
        stateCb(modelId, finalState);
    }
    if (ok && progressCb) {
        progressCb(modelId, 1.0f, "MiGraphX standard artifact prewarm complete.");
    }
    return ok;
}

bool ModelManager::startDownload(const std::string& modelId,
                                  const ModelProgressCb& progressCb,
                                  const ModelStateCb&    stateCb,
                                  std::string&           error) {
    {
        std::lock_guard<std::mutex> lock(impl_->mtx);
        auto it = impl_->records.find(modelId);
        if (it == impl_->records.end()) {
            error = "Unknown model id: " + modelId;
            return false;
        }
        const auto& m = it->second;
        if (m.state == ModelState::Downloading) {
            error = "Already downloading.";
            return false;
        }
        if (m.state == ModelState::Downloaded ||
            m.state == ModelState::Converted) {
            error = "Model already available at: " + m.downloadedPath;
            return false;
        }
        if (m.entry.downloadUrl.empty()) {
            error = "Model has no download URL (built-in / parametric).";
            return false;
        }

        // Set transient state
        impl_->records[modelId].state = ModelState::Downloading;
        impl_->cancelFlags[modelId] = std::make_shared<std::atomic<bool>>(false);
    }

    if (stateCb) { stateCb(modelId, ModelState::Downloading); }

    // Capture copies for the thread
    const ModelEntry entry  = *catalogEntryById(modelId);
    const std::filesystem::path dlDir = impl_->downloadedDir();

    // Capture cancelFlag as shared_ptr so the thread holds a stable reference
    // even if the cancelFlags map rehashes.
    std::shared_ptr<std::atomic<bool>> cancelFlag;
    {
        std::lock_guard<std::mutex> lock(impl_->mtx);
        cancelFlag = impl_->cancelFlags.at(modelId);
    }

    // Capture impl_ as a shared_ptr so the background thread keeps
    // the Impl alive even if ModelManager is destroyed before the
    // download thread finishes.
    std::shared_ptr<Impl> implPtr = impl_;
    std::thread([implPtr, entry, dlDir, progressCb, stateCb, modelId, cancelFlag]() mutable {
        std::string err;

        const bool isZipArchive = !entry.archiveSubPath.empty();

        // For zip archives, download to a temporary file; for direct files,
        // download straight to the final destination.
        const auto destPath = isZipArchive
            ? dlDir / (entry.id + "_archive.zip")
            : dlDir / entry.filename;

        bool ok = curlDownload(entry.downloadUrl, destPath,
                               progressCb, modelId, *cancelFlag, err);

        if (ok && isZipArchive) {
            if (progressCb) { progressCb(entry.id, 0.95f, "Extracting from archive…"); }
            ok = extractFromZip(destPath, entry.archiveSubPath,
                                dlDir / entry.filename, err);
            if (ok && !entry.archiveSubPathAux.empty() && !entry.filenameAux.empty()) {
                ok = extractFromZip(destPath, entry.archiveSubPathAux,
                                    dlDir / entry.filenameAux, err);
            }
            // Remove the temporary archive regardless of extraction outcome.
            std::error_code removeEc;
            std::filesystem::remove(destPath, removeEc);
        } else if (ok && !entry.downloadUrlAux.empty()) {
            // Direct download of a second file (NCNN .bin without zip)
            const auto destAux = dlDir / entry.filenameAux;
            ok = curlDownload(entry.downloadUrlAux, destAux,
                              progressCb, modelId, *cancelFlag, err);
        }

        ModelState finalState = ModelState::Error;
        {
            std::lock_guard<std::mutex> lock(implPtr->mtx);
            auto& m = implPtr->records[modelId];
            if (ok) {
                // Clear the transient Downloading state before scanning so
                // scanAndUpdate() can derive the correct final state.
                m.state = ModelState::NotDownloaded;
                implPtr->scanAndUpdate(m);
            } else {
                m.state        = ModelState::Error;
                m.errorMessage = err;
            }
            finalState = m.state;
        }

        if (stateCb) { stateCb(modelId, finalState); }
        if (progressCb) {
            progressCb(modelId, 1.0f, ok ? "Download complete." : ("Download failed: " + err));
        }
    }).detach();

    return true;
}

void ModelManager::cancelDownload(const std::string& modelId) {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    auto it = impl_->cancelFlags.find(modelId);
    if (it != impl_->cancelFlags.end() && it->second) {
        it->second->store(true);
    }
}

bool ModelManager::convertToMiGraphX(const std::string& modelId,
                                      const ModelProgressCb& progressCb,
                                      const ModelStateCb&    stateCb,
                                      std::string&           error,
                                      ModelPrecision         compilePrecision) {
    if (!isSupportedMiGraphXCompilePrecision(compilePrecision)) {
        error = "MiGraphX compilation currently supports fp32, fp16, and int8 artifacts; requested "
              + compilePrecisionTag(compilePrecision) + ".";
        return false;
    }
    if (compilePrecision == ModelPrecision::Int8) {
        error = "Generic MiGraphX int8 conversion is not supported because int8 compilation "
                "requires calibration frames from a real input video. "
                "Use runtime auto-compile with AVE_MIGRAPHX_PRECISION=int8 instead.";
        return false;
    }

    std::string modelPath;
    ModelFormat sourceFormat = ModelFormat::Onnx;
    ModelEntry entry;
    {
        std::lock_guard<std::mutex> lock(impl_->mtx);
        auto it = impl_->records.find(modelId);
        if (it == impl_->records.end()) {
            error = "Unknown model id: " + modelId;
            return false;
        }
        const auto& m = it->second;
        if (m.downloadedPath.empty() || m.downloadedPath == "(builtin)") {
            error = "Model not yet downloaded – cannot compile.";
            return false;
        }
        if (m.entry.sourceFormat != ModelFormat::Onnx &&
            m.entry.sourceFormat != ModelFormat::Pytorch) {
            error = "MiGraphX compilation requires ONNX or PyTorch (.pth/.pt) format.";
            return false;
        }
        sourceFormat = m.entry.sourceFormat;
        modelPath    = m.downloadedPath;
        entry        = m.entry;
        impl_->records[modelId].state = ModelState::Converting;
    }
    if (stateCb) { stateCb(modelId, ModelState::Converting); }

    // If the model is a PyTorch checkpoint, export it to a temporary ONNX via
    // torch-MiGraphX (torch.onnx.export, opset 17) before calling migraphx-driver.
    std::filesystem::path onnxPath = modelPath;
    std::filesystem::path tempOnnxPath;
    if (sourceFormat == ModelFormat::Pytorch) {
        if (progressCb) { progressCb(modelId, 0.02f, "Exporting PyTorch model to ONNX…"); }
        tempOnnxPath = std::filesystem::temp_directory_path()
                     / (modelId + "_torch_export.onnx");
        std::string exportErr;
        if (!torchExportToOnnx(modelPath, tempOnnxPath, exportErr)) {
            error = "PyTorch → ONNX export failed: " + exportErr;
            ModelState finalState = ModelState::Error;
            {
                std::lock_guard<std::mutex> lockErr(impl_->mtx);
                auto& mErr        = impl_->records[modelId];
                mErr.state        = ModelState::Error;
                mErr.errorMessage = error;
                finalState        = mErr.state;
            }
            if (stateCb) { stateCb(modelId, finalState); }
            return false;
        }
        onnxPath = tempOnnxPath;
    }

    std::filesystem::path effectiveOnnxPath;
    std::filesystem::path manifestSourcePath = modelPath;
    if (progressCb && entry.migraphxOnnxTransform != MiGraphXOnnxTransform::None) {
        progressCb(modelId, 0.03f, "Preparing ONNX for MiGraphX compatibility…");
    }
    std::string prepareErr;
    if (!prepareOnnxForMiGraphX(entry, onnxPath, manifestSourcePath,
                                impl_->preparedDir(),
                                effectiveOnnxPath, manifestSourcePath,
                                prepareErr)) {
        error = "ONNX preparation failed: " + prepareErr;
        ModelState finalState = ModelState::Error;
        {
            std::lock_guard<std::mutex> lockErr(impl_->mtx);
            auto& mErr        = impl_->records[modelId];
            mErr.state        = ModelState::Error;
            mErr.errorMessage = error;
            finalState        = mErr.state;
        }
        if (stateCb) { stateCb(modelId, finalState); }
        return false;
    }

    const int baselineWidth = baselineCompileWidth();
    const int baselineHeight = baselineCompileHeight();
    int compileWidth = baselineWidth;
    int compileHeight = baselineHeight;
    const auto defaultDims = defaultMiGraphXCompileDims(entry);
    if (defaultDims.has_value()) {
        compileWidth = defaultDims->first;
        compileHeight = defaultDims->second;
    } else {
        std::tie(compileWidth, compileHeight) = resolveMiGraphXCompileDims(
            entry, baselineWidth, baselineHeight);
    }
    const auto mxrPath = defaultDims.has_value()
        ? compiledArtifactPath(impl_->convertedDir(), modelId, compilePrecision,
                               compileWidth, compileHeight, 1)
        : compiledArtifactPath(impl_->convertedDir(), modelId, compilePrecision);
    const bool allowTileFallback = !defaultDims.has_value() &&
        canUseMiGraphXTileFallbackLadder(compileWidth, compileHeight);
    bool ok = migraphxCompile(effectiveOnnxPath, mxrPath, manifestSourcePath,
                              compileWidth, compileHeight,
                              1,
                              compilePrecision,
                              std::nullopt,
                              progressCb, modelId, error);
    if (!ok && shouldRetryFp32AfterTimeout(compilePrecision, error)) {
        const auto fp32Path = defaultDims.has_value()
            ? compiledArtifactPath(impl_->convertedDir(), modelId, ModelPrecision::Fp32,
                                   compileWidth, compileHeight, 1)
            : compiledArtifactPath(impl_->convertedDir(), modelId, ModelPrecision::Fp32);
        if (progressCb) {
            progressCb(modelId, 0.02f,
                "fp16 compilation timed out; retrying with fp32 artifact…");
        }
        std::string validationDetail;
        if (validatedCompiledArtifactPath(modelId, ModelPrecision::Fp32,
                                          std::nullopt, std::nullopt, 1,
                                          &validationDetail).has_value()) {
            ok = true;
        } else {
            if (fileExists(fp32Path) && !validationDetail.empty()) {
                    std::cout << "[auto-compile] Ignoring stale fp32 fallback artifact '"
                              << fp32Path << "': " << validationDetail << std::endl;
            }
            std::string fp32Error;
            ok = migraphxCompile(effectiveOnnxPath, fp32Path, manifestSourcePath,
                                 compileWidth, compileHeight,
                                 1,
                                 ModelPrecision::Fp32,
                                 std::nullopt,
                                 progressCb, modelId, fp32Error);
            if (!ok) {
                error += "\n\nfp32 fallback also failed:\n" + fp32Error;
            }
        }
        if (ok) {
            error.clear();
        }
    }
    if (!ok && isMiGraphXCompileTimeout(error) && allowTileFallback) {
        const auto fallbackExtents = buildMiGraphXTileFallbackExtents(compileWidth);
        for (std::size_t i = 1; i < fallbackExtents.size(); ++i) {
            const int fallbackExtent = fallbackExtents[i];
            if (progressCb) {
                progressCb(modelId, 0.02f,
                    "Compilation timed out; retrying smaller fp32 tile at "
                    + formatCompileDimensions(fallbackExtent, fallbackExtent) + "…");
            }
            std::string fallbackError;
            std::string validationDetail;
            if (validatedCompiledArtifactPath(modelId, ModelPrecision::Fp32,
                                              fallbackExtent, fallbackExtent, 1,
                                              &validationDetail).has_value()) {
                ok = true;
            } else {
                if (fileExists(compiledArtifactPath(impl_->convertedDir(), modelId,
                                                   ModelPrecision::Fp32,
                                                   fallbackExtent, fallbackExtent, 1)) &&
                    !validationDetail.empty()) {
                    std::cout << "[auto-compile] Ignoring stale smaller-tile fallback artifact for '"
                              << modelId << "' at "
                              << formatCompileDimensions(fallbackExtent, fallbackExtent)
                              << ": " << validationDetail << std::endl;
                }
                ok = migraphxCompile(effectiveOnnxPath, mxrPath, manifestSourcePath,
                                     fallbackExtent, fallbackExtent,
                                     1,
                                     ModelPrecision::Fp32,
                                     std::nullopt,
                                     progressCb, modelId, fallbackError);
            }
            if (ok) {
                error.clear();
                break;
            }
            error += "\n\n"
                  + formatCompileDimensions(fallbackExtent, fallbackExtent)
                  + " fp32 fallback also failed:\n" + fallbackError;
            if (!isMiGraphXCompileTimeout(fallbackError)) {
                break;
            }
        }
    }

    ModelState finalState = ModelState::Error;
    {
        std::lock_guard<std::mutex> lock(impl_->mtx);
        auto& m = impl_->records[modelId];
        if (ok) {
            // Clear the transient Converting state before scanning so
            // scanAndUpdate() can derive the correct final state from
            // the files on disk (it early-returns for transient states).
            m.state = ModelState::Downloaded;
            impl_->scanAndUpdate(m);
        } else {
            m.state        = ModelState::Error;
            m.errorMessage = error;
        }
        finalState = m.state;
    }
    if (stateCb) { stateCb(modelId, finalState); }
    return ok;
}

// ─── autoCompileForInference ───────────────────────────────────────
// Automatically compile a model for inference if not already compiled.
std::optional<std::string> ModelManager::autoCompileForInference(
    const std::string& modelId,
    std::string& error,
    std::optional<std::int64_t> inputWidth,
    std::optional<std::int64_t> inputHeight,
    ModelPrecision compilePrecision,
    int compileBatch,
    std::optional<std::string> calibrationVideoPath) {

    if (!isSupportedMiGraphXCompilePrecision(compilePrecision)) {
        error = "MiGraphX auto-compile currently supports fp32, fp16, and int8 artifacts; requested "
              + compilePrecisionTag(compilePrecision) + ".";
        return std::nullopt;
    }

    const bool useCustomDims = inputWidth.has_value() || inputHeight.has_value();
    if (useCustomDims && (!inputWidth.has_value() || !inputHeight.has_value())) {
        error = "autoCompileForInference requires both inputWidth and inputHeight when either is provided.";
        return std::nullopt;
    }
    if (compileBatch <= 0) {
        error = "MiGraphX auto-compile requires a positive compile batch.";
        return std::nullopt;
    }
    const int requestedCompileBatch = normaliseCompileBatch(compileBatch);

    const ModelEntry* entry = catalogEntryById(modelId);
    if (entry == nullptr) {
        error = "Unknown model id: " + modelId;
        return std::nullopt;
    }
    const bool fixedCompileDims = hasFixedMiGraphXCompileDims(*entry);

    int compileWidth = baselineCompileWidth();
    int compileHeight = baselineCompileHeight();
    if (useCustomDims) {
        if (*inputWidth <= 0 || *inputHeight <= 0 ||
            *inputWidth > static_cast<std::int64_t>(std::numeric_limits<int>::max()) ||
            *inputHeight > static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
            error = "Invalid compile dimensions: " + std::to_string(*inputWidth) + "x" + std::to_string(*inputHeight);
            return std::nullopt;
        }
        compileWidth = static_cast<int>(*inputWidth);
        compileHeight = static_cast<int>(*inputHeight);
    }
    std::tie(compileWidth, compileHeight) =
        resolveMiGraphXCompileDims(*entry, compileWidth, compileHeight);

    const bool compileDimsResolved = fixedCompileDims || useCustomDims;
    const bool canReuseGenericBaselineArtifact =
        !fixedCompileDims &&
        compileWidth == baselineCompileWidth() &&
        compileHeight == baselineCompileHeight();
    const bool allowTileFallback = !fixedCompileDims &&
        canUseMiGraphXTileFallbackLadder(compileWidth, compileHeight);
    auto artifactPathFor = [&](ModelPrecision precision,
                               int width,
                               int height,
                               int batch) -> std::filesystem::path {
        return compiledArtifactPath(impl_->convertedDir(), modelId, precision, width, height, batch);
    };
    auto customMxrPath = [&](ModelPrecision precision, int batch) -> std::filesystem::path {
        return artifactPathFor(precision, compileWidth, compileHeight, batch);
    };
    auto logRejectedArtifact = [&](const std::filesystem::path& artifactPath,
                                   const std::string& detail) {
        if (fileExists(artifactPath) && !detail.empty()) {
            std::cout << "[auto-compile] Ignoring stale artifact '" << artifactPath
                      << "': " << detail << std::endl;
        }
    };
    auto reuseValidatedArtifact = [&](ModelPrecision precision,
                                      int width,
                                      int height,
                                      int batch,
                                      const std::string& prefix) -> std::optional<std::string> {
        std::string detail;
        const auto validated = validatedCompiledArtifactPath(
            modelId, precision, width, height, batch, &detail);
        if (validated.has_value()) {
            std::cout << prefix << *validated << std::endl;
            return validated;
        }
        logRejectedArtifact(artifactPathFor(precision, width, height, batch), detail);
        return std::nullopt;
    };
    auto reuseValidatedArtifactNoDims = [&](ModelPrecision precision,
                                            const std::string& prefix) -> std::optional<std::string> {
        std::string detail;
        const auto validated = validatedCompiledArtifactPath(
            modelId, precision, std::nullopt, std::nullopt, 1, &detail);
        if (validated.has_value()) {
            std::cout << prefix << *validated << std::endl;
            return validated;
        }
        if (const auto defaultDims = defaultMiGraphXCompileDims(*entry); defaultDims.has_value()) {
            logRejectedArtifact(compiledArtifactPath(impl_->convertedDir(), modelId, precision,
                                                     defaultDims->first, defaultDims->second, 1),
                                detail);
        } else {
            logRejectedArtifact(compiledArtifactPath(impl_->convertedDir(), modelId, precision), detail);
        }
        return std::nullopt;
    };

    if (compileDimsResolved) {
        if (const auto exactPath = reuseValidatedArtifact(
                compilePrecision, compileWidth, compileHeight, requestedCompileBatch,
                "[auto-compile] Reusing exact artifact: "); exactPath.has_value()) {
            return exactPath;
        }
        if (requestedCompileBatch > 1) {
            if (const auto batch1Path = reuseValidatedArtifact(
                    compilePrecision, compileWidth, compileHeight, 1,
                    "[auto-compile] Reusing validated batch-1 artifact: ");
                batch1Path.has_value()) {
                return batch1Path;
            }
        }
        if (compilePrecision == ModelPrecision::Fp16) {
            if (const auto fp32Path = reuseValidatedArtifact(
                    ModelPrecision::Fp32, compileWidth, compileHeight, requestedCompileBatch,
                    "[auto-compile] Reusing validated fp32 fallback artifact: ");
                fp32Path.has_value()) {
                return fp32Path;
            }
        }
        if (canReuseGenericBaselineArtifact) {
            if (const auto genericPath = reuseValidatedArtifactNoDims(
                    compilePrecision,
                    "[auto-compile] Reusing validated generic baseline artifact: ");
                genericPath.has_value()) {
                return genericPath;
            }
            if (compilePrecision == ModelPrecision::Fp16) {
                if (const auto genericFp32Path = reuseValidatedArtifactNoDims(
                        ModelPrecision::Fp32,
                        "[auto-compile] Reusing validated generic baseline fp32 fallback artifact: ");
                    genericFp32Path.has_value()) {
                    return genericFp32Path;
                }
            }
        }
        if (allowTileFallback) {
            const auto fallbackExtents = buildMiGraphXTileFallbackExtents(compileWidth);
            for (std::size_t i = 1; i < fallbackExtents.size(); ++i) {
                const int fallbackExtent = fallbackExtents[i];
                if (const auto fallbackFp32Path = reuseValidatedArtifact(
                        ModelPrecision::Fp32, fallbackExtent, fallbackExtent,
                        requestedCompileBatch,
                        "[auto-compile] Reusing validated smaller fp32 tile artifact: ");
                    fallbackFp32Path.has_value()) {
                    return fallbackFp32Path;
                }
            }
        }
    } else {
        {
            std::lock_guard<std::mutex> lock(impl_->mtx);
            auto it = impl_->records.find(modelId);
            if (it == impl_->records.end()) {
                error = "Unknown model id: " + modelId;
                return std::nullopt;
            }
            impl_->scanAndUpdate(it->second);
        }
        if (const auto exactPath = reuseValidatedArtifactNoDims(
                compilePrecision, "[auto-compile] Reusing validated compiled artifact: ");
            exactPath.has_value()) {
            return exactPath;
        }
        if (compilePrecision == ModelPrecision::Fp16) {
            if (const auto fp32Path = reuseValidatedArtifactNoDims(
                    ModelPrecision::Fp32,
                    "[auto-compile] Reusing validated default fp32 fallback artifact: ");
                fp32Path.has_value()) {
                return fp32Path;
            }
        }

        error = "[NeedsFrameSize] MiGraphX auto-compile requires actual input dimensions for the first compile of model '"
              + modelId + "'.";
        if (compilePrecision == ModelPrecision::Int8) {
            error += " Int8 also requires calibration frames from the real input video.";
        }
        error += " Deferring compile until the input video has been probed.";
        return std::nullopt;
    }

    if (compilePrecision == ModelPrecision::Int8 &&
        (!calibrationVideoPath.has_value() || calibrationVideoPath->empty())) {
        error = "MiGraphX int8 auto-compile requires a calibration video path.";
        return std::nullopt;
    }

    // Check if model is downloaded and capture source format/path.
    std::string modelPath;
    ModelFormat sourceFormat = ModelFormat::Onnx;
    {
        std::lock_guard<std::mutex> lock(impl_->mtx);
        auto it = impl_->records.find(modelId);
        if (it == impl_->records.end()) {
            error = "Unknown model id: " + modelId;
            return std::nullopt;
        }
        const auto& m = it->second;
        if (m.downloadedPath.empty() || m.downloadedPath == "(builtin)") {
            error = "Model not yet downloaded – cannot compile.";
            return std::nullopt;
        }
        if (m.entry.sourceFormat != ModelFormat::Onnx &&
            m.entry.sourceFormat != ModelFormat::Pytorch) {
            error = "MiGraphX compilation requires ONNX or PyTorch (.pth/.pt) format.";
            return std::nullopt;
        }
        modelPath = m.downloadedPath;
        sourceFormat = m.entry.sourceFormat;
    }

    // PyTorch checkpoints are exported to temporary ONNX before compile.
    std::filesystem::path onnxPath = modelPath;
    std::filesystem::path tempOnnxPath;
    if (sourceFormat == ModelFormat::Pytorch) {
        tempOnnxPath = std::filesystem::temp_directory_path()
                     / (modelId + "_torch_export.onnx");
        std::string exportErr;
        if (!torchExportToOnnx(modelPath, tempOnnxPath, exportErr)) {
            error = "PyTorch → ONNX export failed: " + exportErr;
            return std::nullopt;
        }
        onnxPath = tempOnnxPath;
    }

    std::filesystem::path effectiveOnnxPath;
    std::filesystem::path manifestSourcePath = modelPath;
    std::string prepareErr;
    if (!prepareOnnxForMiGraphX(*entry, onnxPath, manifestSourcePath,
                                impl_->preparedDir(),
                                effectiveOnnxPath, manifestSourcePath,
                                prepareErr)) {
        error = "ONNX preparation failed: " + prepareErr;
        return std::nullopt;
    }

    // Progress callback that logs to stdout
    auto progressCb = [](const std::string&, float progress, const std::string& msg) {
        std::cout << "[auto-compile] ";
        if (progress >= 0.0f) {
            std::cout << static_cast<int>(progress * 100) << "%: ";
        }
        std::cout << msg << std::endl;
    };

    if (compileDimsResolved) {
        auto compileForBatch = [&](int batch,
                                   std::string& batchError) -> std::optional<std::string> {
            const auto mxrPath = customMxrPath(compilePrecision, batch);
            if (const auto exactPath = reuseValidatedArtifact(
                    compilePrecision, compileWidth, compileHeight, batch,
                    "[auto-compile] Reusing exact artifact: "); exactPath.has_value()) {
                batchError.clear();
                return exactPath;
            }
            if (compilePrecision == ModelPrecision::Fp16) {
                if (const auto fp32Path = reuseValidatedArtifact(
                        ModelPrecision::Fp32, compileWidth, compileHeight, batch,
                        "[auto-compile] Reusing validated fp32 fallback artifact: ");
                    fp32Path.has_value()) {
                    batchError.clear();
                    return fp32Path;
                }
            }
            if (allowTileFallback) {
                const auto fallbackExtents = buildMiGraphXTileFallbackExtents(compileWidth);
                for (std::size_t i = 1; i < fallbackExtents.size(); ++i) {
                    const int fallbackExtent = fallbackExtents[i];
                    if (const auto fallbackFp32Path = reuseValidatedArtifact(
                            ModelPrecision::Fp32, fallbackExtent, fallbackExtent, batch,
                            "[auto-compile] Reusing validated smaller fp32 tile artifact: ");
                        fallbackFp32Path.has_value()) {
                        batchError.clear();
                        return fallbackFp32Path;
                    }
                }
            }
            std::cout << "[auto-compile] Compiling '" << modelId
                      << "' to " << compilePrecisionTag(compilePrecision)
                      << " at " << compileWidth << "x" << compileHeight
                      << " batch " << batch;
            if (compilePrecision != ModelPrecision::Int8) {
                std::cout << " (no int8 quantization)";
            }
            std::cout << "..." << std::endl;
            if (migraphxCompile(effectiveOnnxPath, mxrPath, manifestSourcePath,
                                compileWidth, compileHeight,
                                batch, compilePrecision,
                                calibrationVideoPath,
                                progressCb, modelId, batchError)) {
                std::cout << "[auto-compile] Successfully compiled to: " << mxrPath << std::endl;
                return mxrPath.string();
            }

            bool exactFp32Attempted = false;
            if (shouldRetryFp32AfterTimeout(compilePrecision, batchError)) {
                const auto fp32Path = customMxrPath(ModelPrecision::Fp32, batch);
                std::cout << "[auto-compile] fp16 compilation timed out; retrying fp32 at "
                          << compileWidth << "x" << compileHeight
                          << " batch " << batch << "..." << std::endl;
                exactFp32Attempted = true;
                if (const auto validatedFp32 = reuseValidatedArtifact(
                        ModelPrecision::Fp32, compileWidth, compileHeight, batch,
                        "[auto-compile] Reusing validated fp32 fallback artifact: ");
                    validatedFp32.has_value()) {
                    batchError.clear();
                    return validatedFp32;
                }
                std::string fp32Error;
                if (migraphxCompile(effectiveOnnxPath, fp32Path, manifestSourcePath,
                                    compileWidth, compileHeight,
                                    batch, ModelPrecision::Fp32,
                                    calibrationVideoPath,
                                    progressCb, modelId, fp32Error)) {
                    std::cout << "[auto-compile] Successfully compiled to: "
                              << fp32Path << std::endl;
                    batchError.clear();
                    return fp32Path.string();
                }
                batchError += "\n\nfp32 fallback also failed:\n" + fp32Error;
            }

            const bool shouldTrySmallerTiles =
                allowTileFallback &&
                ((compilePrecision == ModelPrecision::Fp32 && isMiGraphXCompileTimeout(batchError)) ||
                 (exactFp32Attempted && isMiGraphXCompileTimeout(batchError)));
            if (!shouldTrySmallerTiles) {
                return std::nullopt;
            }

            const auto fallbackExtents = buildMiGraphXTileFallbackExtents(compileWidth);
            for (std::size_t i = 1; i < fallbackExtents.size(); ++i) {
                const int fallbackExtent = fallbackExtents[i];
                const auto fallbackFp32Path =
                    artifactPathFor(ModelPrecision::Fp32, fallbackExtent, fallbackExtent, batch);
                std::cout << "[auto-compile] compile timed out; retrying smaller fp32 tile at "
                          << formatCompileDimensions(fallbackExtent, fallbackExtent)
                          << " batch " << batch << "..." << std::endl;
                if (const auto validatedFallback = reuseValidatedArtifact(
                        ModelPrecision::Fp32, fallbackExtent, fallbackExtent, batch,
                        "[auto-compile] Reusing validated smaller fp32 tile artifact: ");
                    validatedFallback.has_value()) {
                    batchError.clear();
                    return validatedFallback;
                }
                std::string fallbackError;
                if (migraphxCompile(effectiveOnnxPath, fallbackFp32Path, manifestSourcePath,
                                    fallbackExtent, fallbackExtent,
                                    batch, ModelPrecision::Fp32,
                                    calibrationVideoPath,
                                    progressCb, modelId, fallbackError)) {
                    std::cout << "[auto-compile] Successfully compiled to: "
                              << fallbackFp32Path << std::endl;
                    batchError.clear();
                    return fallbackFp32Path.string();
                }
                batchError += "\n\n"
                           + formatCompileDimensions(fallbackExtent, fallbackExtent)
                           + " fp32 fallback also failed:\n" + fallbackError;
                if (!isMiGraphXCompileTimeout(fallbackError)) {
                    break;
                }
            }
            return std::nullopt;
        };

        std::string primaryError;
        if (const auto primaryPath = compileForBatch(requestedCompileBatch, primaryError);
            primaryPath.has_value()) {
            error.clear();
            return primaryPath;
        }

        if (requestedCompileBatch > 1) {
            std::cout << "[auto-compile] Batch " << requestedCompileBatch
                      << " artifact failed; retrying batch 1 for compatibility."
                      << std::endl;
            std::string batch1Error;
            if (const auto batch1Path = compileForBatch(1, batch1Error);
                batch1Path.has_value()) {
                error.clear();
                return batch1Path;
            }
            error = primaryError;
            if (!error.empty()) {
                error += "\n\n";
            }
            error += "Batch-1 fallback also failed:\n" + batch1Error;
            return std::nullopt;
        }

        error = primaryError;
        return std::nullopt;
    }

    if (convertToMiGraphX(modelId, progressCb, ModelStateCb{}, error, compilePrecision)) {
        refresh();
        if (const auto exactPath = reuseValidatedArtifactNoDims(
                compilePrecision, "[auto-compile] Successfully compiled to: ");
            exactPath.has_value()) {
            return exactPath;
        }
        if (compilePrecision == ModelPrecision::Fp16) {
            if (const auto fp32Path = reuseValidatedArtifactNoDims(
                    ModelPrecision::Fp32,
                    "[auto-compile] Successfully compiled to validated fp32 fallback: ");
                fp32Path.has_value()) {
                return fp32Path;
            }
        }
    }

    if (error.empty()) {
        error = "Compilation failed: compiled .mxr was not found after convert step.";
    }
    return std::nullopt;
}

std::string ModelManager::modelDropdownLabel(const std::string& modelId) const {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    auto it = impl_->records.find(modelId);
    if (it == impl_->records.end()) { return modelId; }
    const auto& m = it->second;
    return statePrefix(m.state) + " " + m.entry.displayName;
}

std::vector<ModelManager::DropdownEntry>
ModelManager::dropdownEntriesForStage(StageKind kind) const {
    const auto models = modelsForStage(kind);
    std::vector<DropdownEntry> out;
    out.reserve(models.size());

    // Sort: inference-ready first (converted > downloaded), then by name
    for (const auto& m : models) {
        DropdownEntry de;
        de.modelId       = m.entry.id;
        de.label         = statePrefix(m.state) + " " + m.entry.displayName;
        de.inferenceReady = (m.state == ModelState::Downloaded  ||
                             m.state == ModelState::Converted);
        out.push_back(de);
    }

    std::stable_sort(out.begin(), out.end(), [](const DropdownEntry& a, const DropdownEntry& b) {
        if (a.inferenceReady != b.inferenceReady) { return a.inferenceReady > b.inferenceReady; }
        return false;
    });

    return out;
}

}  // namespace ave
