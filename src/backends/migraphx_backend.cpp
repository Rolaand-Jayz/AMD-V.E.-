// ─────────────────────────────────────────────────────────────────
// MiGraphX Backend — Gold-standard implementation
//
// Compliance with gold-standard requirements:
//   G1  ONNX opset ≤19 gate at parse time.
//   G2  Compile options explicit, logged, and encoded in cache key.
//   G3  Artifact manifest sidecar guards every cached .mxr;
//       mismatch → fail-fast recompile (ArtifactInvalid error).
//   G4  program::get_output_shapes() asserted at load and per frame.
//   G5  program::finish() called after every eval.
//   G6  Version tuple + MIGRAPHX_* env vars logged at initialize().
//   G7  TensorContracts built for input/output at load time; asserted
//       at inference entry (element-count gate).
//   G8  InteropBridge hook points documented for Vulkan↔HIP path;
//       current path uses host staging with an optional HIP-mapped
//       device alias path (logged when active).
//   G9  Structured InferenceError taxonomy throughout.
//   G10 ROCTx markers around compile, load, and eval for rocprof.
// ─────────────────────────────────────────────────────────────────
#include "ave/backends/migraphx_backend.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(__linux__)
#  include <unistd.h>
#endif

// F16C hardware intrinsics for fp16↔4 float conversion.
// All AMD Ryzen CPUs and Intel x86-64 from Ivy Bridge onward support F16C.
// -march=native (set in CMakeLists) exposes __F16C__ to the preprocessor.
#if defined(__F16C__)
#  include <immintrin.h>
#endif

#include "ave/error_taxonomy.hpp"
#include "ave/frame_io.hpp"
#include "ave/interop_bridge.hpp"
#include "ave/model_catalog.hpp"
#include "ave/model_manager.hpp"
#include "ave/observability.hpp"
#include "ave/process_observer.hpp"
#include "ave/rgb_video_loop.hpp"
#include "ave/runtime_diagnostics.hpp"
#include "ave/runtime_paths.hpp"
#include "ave/stage.hpp"
#include "ave/tensor_contract.hpp"
#include "ave/types.hpp"
#include "ave/video_probe.hpp"

#ifdef AVE_HAVE_MIGRAPHX
#  include <migraphx/migraphx.hpp>
#  ifdef MIGRAPHX_VERSION
#    define AVE_MIGRAPHX_VERSION_STR  MIGRAPHX_VERSION
#  else
#    define AVE_MIGRAPHX_VERSION_STR  "unknown"
#  endif
#endif

#ifdef AVE_HAVE_HIP
#  include <hip/hip_runtime.h>
#endif

namespace ave {
namespace {

std::string compilePrecisionTag(MiGraphXPrecision precision) {
    switch (precision) {
        case MiGraphXPrecision::Fp32: return "fp32";
        case MiGraphXPrecision::Fp16: return "fp16";
        case MiGraphXPrecision::Int8: return "int8";
    }
    return "unknown";
}

ModelPrecision modelCompilePrecision(MiGraphXPrecision precision) {
    switch (precision) {
        case MiGraphXPrecision::Fp32: return ModelPrecision::Fp32;
        case MiGraphXPrecision::Fp16: return ModelPrecision::Fp16;
        case MiGraphXPrecision::Int8: return ModelPrecision::Int8;
    }
    return ModelPrecision::Fp32;
}

std::optional<MiGraphXPrecision> parseCompilePrecisionValue(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    if (value == "fp32") {
        return MiGraphXPrecision::Fp32;
    }
    if (value == "fp16") {
        return MiGraphXPrecision::Fp16;
    }
    if (value == "int8") {
        return MiGraphXPrecision::Int8;
    }
    return std::nullopt;
}

std::optional<frame_io::RgbVideoSourceMode> parseFrameSourceModeValue(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    if (value == "pipe" || value == "raw" || value == "raw_pipe") {
        return frame_io::RgbVideoSourceMode::RawPipe;
    }
    if (value == "vulkan" || value == "vulkan_transfer" || value == "transfer") {
        return frame_io::RgbVideoSourceMode::VulkanTransfer;
    }
    if (value == "vulkan_hardware" || value == "vulkan_hw" ||
        value == "vulkan_preserve" || value == "hardware") {
        return frame_io::RgbVideoSourceMode::VulkanHardware;
    }
    return std::nullopt;
}

std::optional<int> readNonNegativeEnvInt(const char* name) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0') {
        return std::nullopt;
    }
    try {
        const int value = std::stoi(raw);
        if (value < 0) {
            return std::nullopt;
        }
        return value;
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<int> preferredAmdDeviceIndexFromSettings() {
    return ::ave::preferredAmdDeviceIndexFromEnv();
}

CompileOptions compileOptionsFromEnv() {
    CompileOptions opts;
    if (const char* rawOffloadCopy = std::getenv("AVE_MIGRAPHX_OFFLOAD_COPY");
        rawOffloadCopy != nullptr) {
        std::string value(rawOffloadCopy);
        if (value == "0" || value == "false" || value == "False" ||
            value == "off" || value == "OFF" || value == "no" || value == "NO") {
            opts.offloadCopy = false;
        } else if (value == "1" || value == "true" || value == "True" ||
                   value == "on" || value == "ON" || value == "yes" || value == "YES") {
            opts.offloadCopy = true;
        } else {
            std::cerr << "[migraphx] WARNING: unsupported AVE_MIGRAPHX_OFFLOAD_COPY='"
                      << rawOffloadCopy << "'; using default offload_copy=1." << std::endl;
        }
    }
    if (const char* rawPrecision = std::getenv("AVE_MIGRAPHX_PRECISION"); rawPrecision != nullptr) {
        if (const auto parsed = parseCompilePrecisionValue(rawPrecision); parsed.has_value()) {
            opts.precision = *parsed;
        } else {
            std::cerr << "[migraphx] WARNING: unsupported AVE_MIGRAPHX_PRECISION='"
                      << rawPrecision << "'; using default precision fp16." << std::endl;
        }
    }
#ifndef AVE_HAVE_HIP
    if (!opts.offloadCopy) {
        std::cerr << "[migraphx] WARNING: AVE_MIGRAPHX_OFFLOAD_COPY=0 requires HIP support; "
                     "falling back to offload_copy=1."
                  << std::endl;
        opts.offloadCopy = true;
    }
#endif
    return opts;
}

frame_io::RgbVideoSourceMode preferredFrameSourceModeFromEnv() {
    if (const char* rawMode = std::getenv("AVE_MIGRAPHX_FRAME_SOURCE");
        rawMode != nullptr) {
        if (const auto parsed = parseFrameSourceModeValue(rawMode); parsed.has_value()) {
            return *parsed;
        }
        std::cerr << "[migraphx] WARNING: unsupported AVE_MIGRAPHX_FRAME_SOURCE='"
                  << rawMode << "'; using raw RGB pipe decode." << std::endl;
    }
    return frame_io::RgbVideoSourceMode::RawPipe;
}

// ─────────────────────────────────────────────────────────────────
// System probe helpers
// ─────────────────────────────────────────────────────────────────

bool fileExists(const std::string& p) {
    std::error_code ec;
    return std::filesystem::exists(p, ec);
}

bool ensureDir(const std::filesystem::path& path) {
    if (path.empty()) {
        return false;
    }
    std::error_code ec;
    std::filesystem::create_directories(path, ec);
    return !ec;
}

bool hasAnyMiGraphXArtifact() {
    if (const auto bundledPrefix = bundledMiGraphXPrefix(); bundledPrefix.has_value()) {
        if (fileExists(((*bundledPrefix) / "bin" / "migraphx-driver").string())) {
            return true;
        }
        if (fileExists(((*bundledPrefix) / "lib" / "libmigraphx_c.so").string())) {
            return true;
        }
        if (fileExists(((*bundledPrefix) / "lib" / "migraphx" / "lib" / "libmigraphx.so").string())) {
            return true;
        }
    }

    for (const auto& c : {"/opt/rocm/lib/libmigraphx.so",
                           "/opt/rocm/lib64/libmigraphx.so",
                           "/usr/lib/libmigraphx.so",
                           "/usr/lib64/libmigraphx.so"}) {
        if (fileExists(c)) { return true; }
    }
    for (const auto& libDir : {"/opt/rocm/lib", "/opt/rocm/lib64",
                                "/usr/lib",     "/usr/lib64"}) {
        std::error_code ec;
        for (const auto& entry : std::filesystem::directory_iterator(libDir, ec)) {
            if (ec) { break; }
            const std::string fname = entry.path().filename().string();
            if (fname.rfind("libmigraphx", 0) == 0) { return true; }
        }
    }
    return process_observer::commandInPath("migraphx-driver")
        || bundledMiGraphXDriverPath().has_value();
}

bool hasAmdSignal() {
    return process_observer::commandInPath("rocminfo")
        || process_observer::commandInPath("rocm-smi")
        || fileExists("/opt/rocm");
}

// ─────────────────────────────────────────────────────────────────
// Version helpers (for manifest key construction — MiGraphX builds only)
// ─────────────────────────────────────────────────────────────────
#ifdef AVE_HAVE_MIGRAPHX

std::string getRocmVersion() {
    return ::ave::detectRocmVersion();
}

std::string getMiGraphXVersion() {
#ifdef AVE_MIGRAPHX_VERSION_STR
    return AVE_MIGRAPHX_VERSION_STR;
#else
    return "unknown";
#endif
}

std::string getGfxTarget() {
    return ::ave::detectAmdGpuArch();
}

std::string envOrDef(const char* name, const char* def) {
    const char* v = std::getenv(name);
    return v != nullptr ? std::string(v) : std::string(def);
}

std::optional<std::string> readNonEmptyEnvValue(const char* name) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0') {
        return std::nullopt;
    }
    return std::string(raw);
}

std::string currentCompileProfileLabel() {
    std::string value = envOrDef("AVE_MIGRAPHX_COMPILE_PROFILE", "balanced");
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    if (value == "fast") {
        return "fast";
    }
    if (value == "balanced" || value == "default" || value == "dynamic_hybrid") {
        return "balanced";
    }
    if (value == "exhaustive" || value == "full" || value == "normal") {
        return "exhaustive";
    }
    return "fast";
}

std::optional<int> autoSelectedVisibleDeviceIndex() {
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

std::string currentVisibleDeviceBinding() {
    const char* overrideVisible = std::getenv("AVE_MIGRAPHX_VISIBLE_DEVICES");
    if (overrideVisible != nullptr && *overrideVisible != '\0') {
        return overrideVisible;
    }
    const char* rocrVisible = std::getenv("ROCR_VISIBLE_DEVICES");
    if (rocrVisible != nullptr && *rocrVisible != '\0') {
        return rocrVisible;
    }
    const char* hipVisible = std::getenv("HIP_VISIBLE_DEVICES");
    if (hipVisible != nullptr && *hipVisible != '\0') {
        return hipVisible;
    }
    if (const auto preferredDevice = preferredAmdDeviceIndexFromSettings(); preferredDevice.has_value()) {
        return std::to_string(*preferredDevice);
    }
    if (const auto autoDevice = autoSelectedVisibleDeviceIndex(); autoDevice.has_value()) {
        return std::to_string(*autoDevice);
    }
    return "all";
}

#ifdef AVE_HAVE_HIP
int selectPreferredHipDeviceIndex(int deviceCount, std::string& detail) {
    if (const auto preferredDevice = preferredAmdDeviceIndexFromSettings(); preferredDevice.has_value()) {
        if (*preferredDevice < deviceCount) {
            detail = "Using explicit AMD device index " + std::to_string(*preferredDevice)
                   + " from environment override.";
            return *preferredDevice;
        }
        detail = "Requested AMD device index " + std::to_string(*preferredDevice)
               + " is out of range for " + std::to_string(deviceCount)
               + " visible HIP device(s); falling back to auto-selection.";
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
        detail = "Auto-selected discrete AMD device " + std::to_string(bestDevice) + ".";
        return bestDevice;
    }

    if (detail.empty()) {
        detail = "Using HIP device 0 (no discrete-only preference available).";
    }
    return 0;
}
#endif

std::string currentRuntimeFingerprint(const std::string& compileProfile,
                                      const std::string& rocmVersion,
                                      const std::string& gfxTarget,
                                      const std::string& visibleDevices,
                                      const std::string& disableMlir,
                                      const std::string& enableNhwc,
                                      const std::string& enableCk,
                                      const std::string& miopenFindMode,
                                      const std::string& miopenParallel) {
    const std::string seed = compileProfile + "|" + rocmVersion + "|"
        + gfxTarget + "|" + visibleDevices + "|" + disableMlir + "|"
        + enableNhwc + "|" + enableCk + "|"
        + miopenFindMode + "|" + miopenParallel;
    return std::to_string(std::hash<std::string>{}(seed));
}

std::string defaultAveCacheDir() {
    if (const char* home = std::getenv("HOME"); home != nullptr && *home != '\0') {
        return std::string(home) + "/.cache/ave";
    }
    return "/tmp/ave_cache";
}

std::filesystem::path defaultMiGraphXProblemCachePath(const std::string& fingerprint) {
    return std::filesystem::path(defaultAveCacheDir()) / "migraphx" / "contexts"
         / fingerprint / "problem_cache.json";
}

std::filesystem::path defaultMiopenUserDbPath(const std::string& fingerprint) {
    return std::filesystem::path(defaultAveCacheDir()) / "migraphx" / "contexts"
         / fingerprint / "miopen_user_db";
}

std::filesystem::path defaultMiopenCacheDir(const std::string& fingerprint) {
    return std::filesystem::path(defaultAveCacheDir()) / "migraphx" / "contexts"
         / fingerprint / "miopen_cache";
}

int defaultMiopenCompileParallelLevel() {
    unsigned workerCount = std::thread::hardware_concurrency();
    if (workerCount == 0u) {
        workerCount = 4u;
    }
    return static_cast<int>(std::min(workerCount, 16u));
}

std::string defaultMiopenFindModeForProfile(const std::string& profile) {
    if (profile == "exhaustive") {
        return "NORMAL";
    }
    if (profile == "balanced") {
        return "DYNAMIC_HYBRID";
    }
    return "FAST";
}

std::string defaultMiGraphXEnableNhwcForProfile(const std::string& profile) {
    if (profile == "balanced" || profile == "exhaustive") {
        return "1";
    }
    return "0";
}

std::string defaultMiGraphXEnableCkForProfile(const std::string& profile) {
    if (profile == "balanced" || profile == "exhaustive") {
        return "1";
    }
    return "0";
}

bool setProcessEnv(const std::string& name, const std::string& value);

bool applyDefaultMiGraphXProcessEnv(int deviceIdx, std::string& error) {
    const std::string profile = currentCompileProfileLabel();
    const std::array<std::pair<const char*, std::string>, 2> defaults = {{
        {"MIGRAPHX_ENABLE_NHWC", defaultMiGraphXEnableNhwcForProfile(profile)},
        {"MIGRAPHX_ENABLE_CK", defaultMiGraphXEnableCkForProfile(profile)},
    }};

    for (const auto& [name, value] : defaults) {
        if (std::getenv(name) != nullptr) {
            continue;
        }
        if (!setProcessEnv(name, value)) {
            error = std::string("Failed to set default MiGraphX environment ")
                  + name + "=" + value;
            return false;
        }
    }

    if (std::getenv("AVE_MIGRAPHX_VISIBLE_DEVICES") == nullptr &&
        std::getenv("ROCR_VISIBLE_DEVICES") == nullptr &&
        std::getenv("HIP_VISIBLE_DEVICES") == nullptr &&
        !preferredAmdDeviceIndexFromSettings().has_value()) {
        const std::string selectedDevice = std::to_string(deviceIdx);
        const std::array<std::pair<const char*, std::string>, 3> visibleDefaults = {{
            {"AVE_MIGRAPHX_VISIBLE_DEVICES", selectedDevice},
            {"ROCR_VISIBLE_DEVICES", selectedDevice},
            {"HIP_VISIBLE_DEVICES", selectedDevice},
        }};
        for (const auto& [name, value] : visibleDefaults) {
            if (!setProcessEnv(name, value)) {
                error = std::string("Failed to set default MiGraphX environment ")
                      + name + "=" + value;
                return false;
            }
        }
    }

    return true;
}

struct RuntimeEnvConfig {
    obs::ArtifactManifestFields fields;
    std::vector<std::pair<std::string, std::string>> overrides;
};

std::mutex& runtimeEnvMutex() {
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

class ScopedRuntimeEnvOverrides {
  public:
    explicit ScopedRuntimeEnvOverrides(
        const std::vector<std::pair<std::string, std::string>>& overrides)
        : lock_(runtimeEnvMutex()) {
        previous_.reserve(overrides.size());
        for (const auto& [name, value] : overrides) {
            const char* current = std::getenv(name.c_str());
            previous_.emplace_back(
                name,
                current == nullptr ? std::optional<std::string>{}
                                   : std::make_optional(std::string(current)));
            if (!setProcessEnv(name, value)) {
                error_ = "Failed to set MiGraphX runtime environment override "
                       + name + "=" + value;
                restore();
                break;
            }
        }
    }

    ScopedRuntimeEnvOverrides(const ScopedRuntimeEnvOverrides&) = delete;
    ScopedRuntimeEnvOverrides& operator=(const ScopedRuntimeEnvOverrides&) = delete;

    ~ScopedRuntimeEnvOverrides() { restore(); }

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

void appendRuntimeEnvOverride(RuntimeEnvConfig& env,
                              const std::string& name,
                              const std::string& value) {
    const char* current = std::getenv(name.c_str());
    if (current == nullptr || value != current) {
        env.overrides.emplace_back(name, value);
    }
}

RuntimeEnvConfig buildRuntimeEnvConfig(const std::string& sourcePath,
                                       const CompileOptions& opts);

#endif  // AVE_HAVE_MIGRAPHX

// ─────────────────────────────────────────────────────────────────
// Tensor-contract helpers
// ─────────────────────────────────────────────────────────────────

bool isInternalOutputParameterName(const std::string& name) {
    return name.find("#output") != std::string::npos;
}

bool toPositiveInt(std::int64_t value, const char* axis, int& out, std::string& error) {
    if (value <= 0 || value > static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
        std::ostringstream os;
        os << "Invalid " << axis << " dimension " << value;
        error = os.str();
        return false;
    }
    out = static_cast<int>(value);
    return true;
}

bool extractSpatialDims(const TensorContract& contract,
                        int&                  width,
                        int&                  height,
                        int&                  channels,
                        std::string&          error) {
    width = 0;
    height = 0;
    channels = 0;

    const auto& dims = contract.shape.dims;
    if (dims.size() == 4) {
        if (contract.layout == TensorLayout::NHWC) {
            return toPositiveInt(dims[2], "width", width, error)
                && toPositiveInt(dims[1], "height", height, error)
                && toPositiveInt(dims[3], "channels", channels, error);
        }
        return toPositiveInt(dims[3], "width", width, error)
            && toPositiveInt(dims[2], "height", height, error)
            && toPositiveInt(dims[1], "channels", channels, error);
    }
    if (dims.size() == 3) {
        if (contract.layout == TensorLayout::HWC) {
            return toPositiveInt(dims[1], "width", width, error)
                && toPositiveInt(dims[0], "height", height, error)
                && toPositiveInt(dims[2], "channels", channels, error);
        }
        return toPositiveInt(dims[2], "width", width, error)
            && toPositiveInt(dims[1], "height", height, error)
            && toPositiveInt(dims[0], "channels", channels, error);
    }

    std::ostringstream os;
    os << "Unsupported tensor rank " << dims.size()
       << " for contract '" << contract.name << "' (" << contract.shape.format() << ")";
    error = os.str();
    return false;
}

int extractBatchSize(const TensorContract& contract);

bool extractStereoDepthOutputDims(const TensorContract& contract,
                                  int                   expectedBatch,
                                  int&                  width,
                                  int&                  height,
                                  int&                  channels,
                                  int&                  batch,
                                  std::string&          error) {
    width = 0;
    height = 0;
    channels = 0;
    batch = 0;

    const auto& dims = contract.shape.dims;
    if (dims.size() == 3) {
        if (!toPositiveInt(dims[2], "width", width, error) ||
            !toPositiveInt(dims[1], "height", height, error)) {
            return false;
        }
        channels = 1;
        if (dims[0] == static_cast<std::int64_t>(expectedBatch) || dims[0] == 1) {
            return toPositiveInt(dims[0], "batch", batch, error);
        }
        error = "Stereo depth output rank-3 tensor does not expose the expected batch dimension.";
        return false;
    }

    if (!extractSpatialDims(contract, width, height, channels, error)) {
        return false;
    }
    batch = extractBatchSize(contract);
    return true;
}

int extractBatchSize(const TensorContract& contract) {
    const auto& dims = contract.shape.dims;
    if (dims.size() == 4 && dims[0] > 0 &&
        dims[0] <= static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
        return static_cast<int>(dims[0]);
    }
    return 1;
}

std::size_t elementsPerBatch(const TensorContract& contract, int batchSize) {
    const std::int64_t totalElements = contract.shape.elements();
    if (totalElements <= 0 || batchSize <= 0) {
        return 0;
    }
    const std::int64_t batchElements =
        totalElements / static_cast<std::int64_t>(batchSize);
    if (batchElements <= 0) {
        return 0;
    }
    return static_cast<std::size_t>(batchElements);
}

std::string loadFailureKey(const std::string& modelId,
                           const std::optional<int>& inputWidth,
                           const std::optional<int>& inputHeight) {
    if (inputWidth.has_value() && inputHeight.has_value()) {
        return modelId + "@" + std::to_string(*inputWidth) + "x" + std::to_string(*inputHeight);
    }
    return modelId + "@default";
}

[[maybe_unused]] const TensorContract* selectPrimaryInputContract(
        const std::vector<TensorContract>& contracts) {
    if (contracts.empty()) { return nullptr; }
    for (const auto& c : contracts) {
        if (c.name == "input") { return &c; }
    }
    return &contracts.front();
}

[[maybe_unused]] bool contractMatchesFrameDims(
        const TensorContract& contract, int width, int height) {
    int modelW = 0;
    int modelH = 0;
    int modelC = 0;
    std::string err;
    if (!extractSpatialDims(contract, modelW, modelH, modelC, err)) {
        return false;
    }
    return modelW == width && modelH == height;
}

// ─────────────────────────────────────────────────────────────────
// ONNX opset scanner (G1: opset ≤19 gate)
//
// Lightweight protobuf binary scan — no proto library required.
// Reads only enough bytes to find the opset_import fields.
//
// ONNX ModelProto field layout (protobuf wire format):
//   opset_import  = field 8, wire type 2 → tag byte 0x42
//   OpsetImport.domain  = field 1, wire type 2 → tag byte 0x0A
//   OpsetImport.version = field 2, wire type 0 → tag byte 0x10
//
// Returns the highest opset version found for the default ONNX domain.
// Returns 0 if no opset_import fields are found (very old model or parse error).
// Returns -1 on file open error.
// ─────────────────────────────────────────────────────────────────
[[maybe_unused]] static constexpr int kMaxSupportedOpset = 19;

namespace proto {

// Read a varint from a byte buffer.  Returns number of bytes consumed,
// or 0 on overflow.  value is set to the decoded uint64.
std::size_t readVarint(const std::uint8_t* buf, std::size_t len, std::uint64_t& value) {
    value = 0;
    for (std::size_t i = 0; i < len && i < 10; ++i) {
        const uint64_t b = buf[i];
        value |= (b & 0x7Fu) << (7u * i);
        if ((b & 0x80u) == 0) { return i + 1; }
    }
    return 0;  // error or overflow
}

// Skip a protobuf field of the given wire type.
// Returns number of bytes consumed, or 0 on error.
std::size_t skipField(const std::uint8_t* buf, std::size_t len, std::uint32_t wireType) {
    if (len == 0) { return 0; }
    if (wireType == 0) {  // varint
        std::uint64_t v;
        return readVarint(buf, len, v);
    }
    if (wireType == 1) {  // 64-bit
        return (len >= 8) ? 8u : 0u;
    }
    if (wireType == 2) {  // length-delimited
        std::uint64_t sz = 0;
        const std::size_t consumed = readVarint(buf, len, sz);
        if (consumed == 0) { return 0; }
        const std::size_t total = consumed + static_cast<std::size_t>(sz);
        return (total <= len) ? total : 0u;
    }
    if (wireType == 5) {  // 32-bit
        return (len >= 4) ? 4u : 0u;
    }
    return 0;  // unknown / unsupported wire type
}

}  // namespace proto

[[maybe_unused]] int extractOnnxMaxOpset(const std::string& onnxPath) {
    // Limit scan to first 256 KB — opset_import always precedes the graph
    // data in well-formed ONNX files.
    static constexpr std::size_t kScanLimit = 256 * 1024;

    std::ifstream f(onnxPath, std::ios::binary);
    if (!f.is_open()) { return -1; }

    std::vector<std::uint8_t> buf(kScanLimit);
    f.read(reinterpret_cast<char*>(buf.data()),
           static_cast<std::streamsize>(kScanLimit));
    const std::size_t bytesRead = static_cast<std::size_t>(f.gcount());

    int maxOpset = 0;
    std::size_t pos = 0;

    while (pos < bytesRead) {
        // Read outer field tag
        std::uint64_t tag = 0;
        const std::size_t tagConsumed = proto::readVarint(&buf[pos], bytesRead - pos, tag);
        if (tagConsumed == 0) { break; }
        pos += tagConsumed;

        const auto fieldNum  = static_cast<std::uint32_t>(tag >> 3u);
        const auto wireType  = static_cast<std::uint32_t>(tag & 0x7u);

        if (fieldNum == 8 && wireType == 2) {
            // opset_import: parse embedded OpsetImport message
            std::uint64_t msgLen = 0;
            const std::size_t lenConsumed = proto::readVarint(&buf[pos], bytesRead - pos, msgLen);
            if (lenConsumed == 0) { break; }
            pos += lenConsumed;

            const std::size_t msgStart = pos;
            const std::size_t msgEnd   = std::min(msgStart + static_cast<std::size_t>(msgLen),
                                                  bytesRead);
            std::size_t inner = msgStart;
            std::int64_t opsetVersion = 0;
            bool hasDomain = false;
            bool domainIsDefault = true;  // "" or "ai.onnx"

            while (inner < msgEnd) {
                std::uint64_t inner_tag = 0;
                const std::size_t it = proto::readVarint(&buf[inner], msgEnd - inner, inner_tag);
                if (it == 0) { break; }
                inner += it;

                const auto ifn = static_cast<std::uint32_t>(inner_tag >> 3u);
                const auto iwt = static_cast<std::uint32_t>(inner_tag & 0x7u);

                if (ifn == 1 && iwt == 2) {
                    // domain string
                    std::uint64_t slen = 0;
                    const std::size_t sc = proto::readVarint(&buf[inner], msgEnd - inner, slen);
                    if (sc == 0) { break; }
                    inner += sc;
                    const std::string domain(
                        reinterpret_cast<const char*>(&buf[inner]),
                        static_cast<std::size_t>(slen));
                    inner += static_cast<std::size_t>(slen);
                    hasDomain = true;
                    domainIsDefault = domain.empty() || domain == "ai.onnx";
                } else if (ifn == 2 && iwt == 0) {
                    // version
                    std::uint64_t v = 0;
                    const std::size_t vc = proto::readVarint(&buf[inner], msgEnd - inner, v);
                    if (vc == 0) { break; }
                    inner += vc;
                    opsetVersion = static_cast<std::int64_t>(v);
                } else {
                    const std::size_t skip = proto::skipField(&buf[inner], msgEnd - inner, iwt);
                    if (skip == 0) { break; }
                    inner += skip;
                }
            }
            pos = msgEnd;

            // Only count default ("ai.onnx" or "") domain entries
            if ((!hasDomain || domainIsDefault) && opsetVersion > maxOpset) {
                maxOpset = static_cast<int>(opsetVersion);
            }
        } else {
            // Skip this field
            if (wireType == 2) {
                std::uint64_t fl = 0;
                const std::size_t fc = proto::readVarint(&buf[pos], bytesRead - pos, fl);
                if (fc == 0) { break; }
                pos += fc + static_cast<std::size_t>(fl);
                if (pos > bytesRead) { break; }
            } else {
                const std::size_t skip = proto::skipField(&buf[pos], bytesRead - pos, wireType);
                if (skip == 0) { break; }
                pos += skip;
            }
        }
    }

    return maxOpset;
}

// ─────────────────────────────────────────────────────────────────
// Manifest key construction
// ─────────────────────────────────────────────────────────────────

#ifdef AVE_HAVE_MIGRAPHX
[[maybe_unused]] obs::ArtifactManifestFields buildManifestFields(
        const std::string& onnxPath,
        const CompileOptions& opts) {
    obs::ArtifactManifestFields f;
    f.manifestSchemaVersion = "2";
    f.migraphxVersion = getMiGraphXVersion();
    f.rocmVersion     = getRocmVersion();
    f.gpuGfxTarget    = getGfxTarget();

    // Model identity: size + mtime plus a deterministic content fingerprint.
    std::error_code ec;
    if (std::filesystem::exists(onnxPath, ec)) {
        f.onnxFileSizeStr = std::to_string(std::filesystem::file_size(onnxPath, ec));
        const auto mtime  = std::filesystem::last_write_time(onnxPath, ec);
        const auto mtimeSec = std::chrono::duration_cast<std::chrono::seconds>(
            mtime.time_since_epoch()).count();
        f.onnxMtimeStr = std::to_string(mtimeSec);
    } else {
        f.onnxFileSizeStr = "0";
        f.onnxMtimeStr    = "0";
    }
    f.sourceFingerprint = obs::buildArtifactSourceFingerprint(onnxPath);

    f.offloadCopy    = opts.offloadCopy ? "1" : "0";
    f.precision      = compilePrecisionTag(opts.precision);
    f.compileProfile = currentCompileProfileLabel();
    f.disableMlir    = readNonEmptyEnvValue("MIGRAPHX_DISABLE_MLIR").value_or("0");
    f.enableNhwc     = readNonEmptyEnvValue("MIGRAPHX_ENABLE_NHWC")
        .value_or(defaultMiGraphXEnableNhwcForProfile(f.compileProfile));
    f.enableCk       = readNonEmptyEnvValue("MIGRAPHX_ENABLE_CK")
        .value_or(defaultMiGraphXEnableCkForProfile(f.compileProfile));
    f.visibleDevices = currentVisibleDeviceBinding();

    f.miopenFindMode = readNonEmptyEnvValue("AVE_MIGRAPHX_MIOPEN_FIND_MODE")
        .value_or(readNonEmptyEnvValue("MIOPEN_FIND_MODE")
                      .value_or(defaultMiopenFindModeForProfile(f.compileProfile)));
    f.miopenCompileParallelLevel =
        readNonEmptyEnvValue("AVE_MIGRAPHX_MIOPEN_COMPILE_PARALLEL_LEVEL")
            .value_or(readNonEmptyEnvValue("MIOPEN_COMPILE_PARALLEL_LEVEL")
                          .value_or(std::to_string(defaultMiopenCompileParallelLevel())));
    f.runtimeFingerprint = currentRuntimeFingerprint(
        f.compileProfile, f.rocmVersion, f.gpuGfxTarget, f.visibleDevices,
        f.disableMlir, f.enableNhwc, f.enableCk,
        f.miopenFindMode, f.miopenCompileParallelLevel);
    f.problemCachePath = readNonEmptyEnvValue("AVE_MIGRAPHX_PROBLEM_CACHE")
        .value_or(readNonEmptyEnvValue("MIGRAPHX_PROBLEM_CACHE")
                      .value_or(defaultMiGraphXProblemCachePath(f.runtimeFingerprint).string()));
    f.miopenUserDbPath = readNonEmptyEnvValue("AVE_MIGRAPHX_MIOPEN_USER_DB_PATH")
        .value_or(readNonEmptyEnvValue("MIOPEN_USER_DB_PATH")
                      .value_or(defaultMiopenUserDbPath(f.runtimeFingerprint).string()));
    f.miopenCustomCacheDir = readNonEmptyEnvValue("AVE_MIGRAPHX_MIOPEN_CACHE_DIR")
        .value_or(readNonEmptyEnvValue("MIOPEN_CUSTOM_CACHE_DIR")
                      .value_or(defaultMiopenCacheDir(f.runtimeFingerprint).string()));
    return f;
}

RuntimeEnvConfig buildRuntimeEnvConfig(const std::string& sourcePath,
                                       const CompileOptions& opts) {
    RuntimeEnvConfig env;
    env.fields = buildManifestFields(sourcePath, opts);

    ensureDir(std::filesystem::path(env.fields.problemCachePath).parent_path());
    ensureDir(std::filesystem::path(env.fields.miopenUserDbPath));
    ensureDir(std::filesystem::path(env.fields.miopenCustomCacheDir));

    appendRuntimeEnvOverride(env, "MIGRAPHX_DISABLE_MLIR", env.fields.disableMlir);
    appendRuntimeEnvOverride(env, "MIGRAPHX_ENABLE_NHWC", env.fields.enableNhwc);
    appendRuntimeEnvOverride(env, "MIGRAPHX_ENABLE_CK", env.fields.enableCk);
    appendRuntimeEnvOverride(env, "MIGRAPHX_PROBLEM_CACHE", env.fields.problemCachePath);
    appendRuntimeEnvOverride(env, "MIOPEN_USER_DB_PATH", env.fields.miopenUserDbPath);
    appendRuntimeEnvOverride(env, "MIOPEN_CUSTOM_CACHE_DIR", env.fields.miopenCustomCacheDir);
    appendRuntimeEnvOverride(env, "MIOPEN_FIND_MODE", env.fields.miopenFindMode);
    appendRuntimeEnvOverride(
        env, "MIOPEN_COMPILE_PARALLEL_LEVEL", env.fields.miopenCompileParallelLevel);

    if (!env.fields.visibleDevices.empty() && env.fields.visibleDevices != "all") {
        appendRuntimeEnvOverride(env, "ROCR_VISIBLE_DEVICES", env.fields.visibleDevices);
        appendRuntimeEnvOverride(env, "HIP_VISIBLE_DEVICES", env.fields.visibleDevices);
    }

    return env;
}
#endif  // AVE_HAVE_MIGRAPHX

// ─────────────────────────────────────────────────────────────────
// Model/stage helpers
// ─────────────────────────────────────────────────────────────────

std::string defaultModelIdFor(StageKind kind) {
    const auto entries = catalogEntriesForStage(kind);
    for (const auto* e : entries) { if (e->isDefault) { return e->id; } }
    if (!entries.empty()) { return entries.front()->id; }
    return {};
}

std::string resolveModelId(const EnhancementStage& stage) {
    const auto pathIt = stage.params.find("model_path");
    const auto explicitIt = stage.params.find("model_path_explicit");
    const bool explicitModelPath =
        explicitIt != stage.params.end() &&
        std::holds_alternative<bool>(explicitIt->second) &&
        std::get<bool>(explicitIt->second);
    if (pathIt != stage.params.end()) {
        if (const auto* path = std::get_if<std::string>(&pathIt->second)) {
            const std::string inferred = inferModelIdFromPath(*path);
            if (explicitModelPath && !inferred.empty()) {
                return inferred;
            }
        }
    }
    const auto it = stage.params.find("model");
    if (it != stage.params.end()) {
        if (const auto* s = std::get_if<std::string>(&it->second)) {
            if (!s->empty()) { return *s; }
        }
    }
    if (pathIt != stage.params.end()) {
        if (const auto* path = std::get_if<std::string>(&pathIt->second)) {
            const std::string inferred = inferModelIdFromPath(*path);
            if (!inferred.empty()) {
                return inferred;
            }
        }
    }
    return defaultModelIdFor(stage.kind);
}

std::optional<std::string> stageModelPath(const EnhancementStage& stage) {
    const auto it = stage.params.find("model_path");
    if (it == stage.params.end()) { return std::nullopt; }
    if (const auto* s = std::get_if<std::string>(&it->second)) {
        if (!s->empty()) { return *s; }
    }
    return std::nullopt;
}

bool stageModelPathExplicit(const EnhancementStage& stage) {
    const auto it = stage.params.find("model_path_explicit");
    if (it == stage.params.end()) { return false; }
    if (const auto* b = std::get_if<bool>(&it->second)) { return *b; }
    return false;
}

std::string normalizeExtLower(const std::string& path) {
    std::string ext = std::filesystem::path(path).extension().string();
    for (char& ch : ext) { ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch))); }
    return ext;
}

std::filesystem::path makeTempLogPath(const std::string& prefix) {
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    const auto stamp = std::chrono::duration_cast<std::chrono::microseconds>(now).count();
    return std::filesystem::temp_directory_path()
         / (prefix + "_" + std::to_string(stamp) + ".log");
}

std::string readLogTail(const std::filesystem::path& path, std::size_t maxLines = 20) {
    std::ifstream in(path);
    if (!in.is_open()) { return {}; }

    std::deque<std::string> lines;
    std::string line;
    while (std::getline(in, line)) {
        lines.push_back(line);
        if (lines.size() > maxLines) {
            lines.pop_front();
        }
    }

    std::ostringstream out;
    bool firstLine = true;
    for (const auto& l : lines) {
        if (!firstLine) {
            out << '\n';
        }
        firstLine = false;
        out << l;
    }
    return out.str();
}

std::string formatProcessExit(int rawStatus) {
#if defined(__unix__) || defined(__APPLE__)
    if (WIFEXITED(rawStatus)) {
        return "exit code " + std::to_string(WEXITSTATUS(rawStatus));
    }
    if (WIFSIGNALED(rawStatus)) {
        return "signal " + std::to_string(WTERMSIG(rawStatus));
    }
#endif
    return "status " + std::to_string(rawStatus);
}

constexpr int kDefaultTileExtent = 192;
constexpr int kDefaultTileOverlap = 16;
constexpr int kDefaultTileBatch = 4;
constexpr int kDefaultStereoDepthResolution = 384;
constexpr int kMinimumStereoDepthResolution = 224;
constexpr int kStereoDepthResolutionStride = 14;
constexpr std::size_t kDefaultEncodeQueueDepth = 4u;
constexpr std::size_t kMaxEncodeQueueDepth = 8u;
constexpr std::size_t kMinPipeStdioBufferBytes = 1u << 20;
constexpr std::size_t kMaxPipeStdioBufferBytes = 8u << 20;
constexpr int kMinPipeCapacityBytes = 4 << 20;
constexpr int kMaxPipeCapacityBytes = 32 << 20;

int readTileEnvValue(const char* name, int defaultValue) {
    if (const char* raw = std::getenv(name); raw != nullptr) {
        try {
            return std::stoi(raw);
        } catch (...) {
            return defaultValue;
        }
    }
    return defaultValue;
}

bool hasNonEmptyEnv(const char* name) {
    const char* raw = std::getenv(name);
    return raw != nullptr && *raw != '\0';
}

std::size_t clampQueueDepth(std::size_t requested,
                            std::size_t defaultDepth,
                            std::size_t maxDepth) {
    if (requested == 0u) {
        return defaultDepth;
    }
    return std::clamp(requested, std::size_t{1u}, maxDepth);
}

std::size_t chooseFrameQueueDepth(std::size_t frameBytes,
                                  std::size_t defaultDepth,
                                  std::size_t maxDepth,
                                  const char* envName) {
    if (const auto overrideDepth = readNonNegativeEnvInt(envName); overrideDepth.has_value()) {
        return clampQueueDepth(static_cast<std::size_t>(*overrideDepth), defaultDepth, maxDepth);
    }

    std::size_t depth = defaultDepth;
    if (frameBytes >= (16u << 20)) {
        depth += 2u;
    }
    if (frameBytes >= (24u << 20)) {
        depth += 2u;
    }
    return clampQueueDepth(depth, defaultDepth, maxDepth);
}

std::size_t choosePipeStdioBufferBytes(std::size_t frameBytes) {
    if (const auto overrideMb = readNonNegativeEnvInt("AVE_MIGRAPHX_PIPE_STDIO_MB");
        overrideMb.has_value()) {
        const std::size_t requested =
            static_cast<std::size_t>(*overrideMb) * (1u << 20);
        return std::clamp(requested,
                          kMinPipeStdioBufferBytes,
                          kMaxPipeStdioBufferBytes);
    }

    const std::size_t target = std::max(frameBytes, kMinPipeStdioBufferBytes);
    return std::clamp(target, kMinPipeStdioBufferBytes, kMaxPipeStdioBufferBytes);
}

int choosePipeCapacityBytes(std::size_t frameBytes) {
    if (const auto overrideMb = readNonNegativeEnvInt("AVE_MIGRAPHX_PIPE_MB");
        overrideMb.has_value()) {
        const std::size_t requested =
            static_cast<std::size_t>(*overrideMb) * (1u << 20);
        return static_cast<int>(std::clamp(
            requested,
            static_cast<std::size_t>(kMinPipeCapacityBytes),
            static_cast<std::size_t>(kMaxPipeCapacityBytes)));
    }

    const std::size_t target = std::max(frameBytes * 2u,
                                        static_cast<std::size_t>(kMinPipeCapacityBytes));
    return static_cast<int>(std::min<std::size_t>(target,
                                                  static_cast<std::size_t>(kMaxPipeCapacityBytes)));
}

struct TileConfig {
    int width = kDefaultTileExtent;
    int height = kDefaultTileExtent;
    int overlap = kDefaultTileOverlap;
    int batch = kDefaultTileBatch;
    bool batchExplicit = false;
};

struct TileWindow {
    int begin = 0;
    int end = 0;
    int offset = 0;
};

struct TileDispatch {
    int tileX = 0;
    int tileY = 0;
    TileWindow srcXWindow;
    TileWindow srcYWindow;
};

template <typename Fn>
void parallelForRows(int rowCount, int workWidth, Fn&& fn) {
    if (rowCount <= 0 || workWidth <= 0) {
        return;
    }

    constexpr std::size_t kMinPixelsPerWorker = 262144u;
    unsigned workerCap = std::thread::hardware_concurrency();
    if (workerCap == 0u) {
        workerCap = 1u;
    }
    workerCap = std::min(workerCap, 8u);

    const std::size_t totalPixels =
        static_cast<std::size_t>(rowCount) * static_cast<std::size_t>(workWidth);
    const std::size_t maxWorkersByProblem =
        (totalPixels + kMinPixelsPerWorker - 1u) / kMinPixelsPerWorker;
    const unsigned workers = static_cast<unsigned>(
        std::min<std::size_t>(static_cast<std::size_t>(workerCap), maxWorkersByProblem));
    if (workers <= 1u) {
        fn(0, rowCount);
        return;
    }

    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(workers - 1u));
    for (unsigned worker = 1u; worker < workers; ++worker) {
        const int begin = static_cast<int>(
            static_cast<std::size_t>(rowCount) * static_cast<std::size_t>(worker)
            / static_cast<std::size_t>(workers));
        const int end = static_cast<int>(
            static_cast<std::size_t>(rowCount) * static_cast<std::size_t>(worker + 1u)
            / static_cast<std::size_t>(workers));
        threads.emplace_back([begin, end, &fn]() { fn(begin, end); });
    }

    fn(0, rowCount / static_cast<int>(workers));
    for (auto& thread : threads) {
        thread.join();
    }
}

std::vector<int> buildTileStarts(int fullExtent, int tileExtent, int overlap);

int chooseAdaptiveTileBatch(int frameWidth,
                            int frameHeight,
                            int tileWidth,
                            int tileHeight,
                            int tileOverlap) {
    const auto tileXs = buildTileStarts(frameWidth, tileWidth, tileOverlap);
    const auto tileYs = buildTileStarts(frameHeight, tileHeight, tileOverlap);
    const std::size_t estimatedTiles = tileXs.size() * tileYs.size();
    if (estimatedTiles <= 1u) {
        return 1;
    }

    const std::size_t tileArea =
        static_cast<std::size_t>(tileWidth) * static_cast<std::size_t>(tileHeight);
    int batch = 4;
    if (tileArea <= 96u * 96u) {
        batch = 16;
    } else if (tileArea <= 128u * 128u) {
        batch = 12;
    } else if (tileArea <= 192u * 192u) {
        batch = 8;
    } else if (tileArea <= 256u * 256u) {
        batch = 4;
    } else {
        batch = 2;
    }

    if (frameWidth * frameHeight <= 1280 * 720 && tileArea <= 192u * 192u) {
        batch = std::max(batch, 8);
    }

    batch = std::min(batch, 16);
    batch = std::min(batch, static_cast<int>(estimatedTiles));
    return std::max(batch, 1);
}

bool resolveTileConfig(const EnhancementStage& stage,
                       TileConfig& tileConfig,
                       std::string& error) {
    const int envTileSize = readTileEnvValue("AVE_MIGRAPHX_TILE_SIZE", kDefaultTileExtent);
    tileConfig.width = readTileEnvValue("AVE_MIGRAPHX_TILE_WIDTH", envTileSize);
    tileConfig.height = readTileEnvValue("AVE_MIGRAPHX_TILE_HEIGHT", envTileSize);
    tileConfig.overlap = readTileEnvValue("AVE_MIGRAPHX_TILE_OVERLAP", kDefaultTileOverlap);
    tileConfig.batch = readTileEnvValue("AVE_MIGRAPHX_TILE_BATCH", kDefaultTileBatch);
    tileConfig.batchExplicit = hasNonEmptyEnv("AVE_MIGRAPHX_TILE_BATCH");

    std::int64_t parsed = 0;
    if (tryGetInt(stage.params, "tile_size", parsed)) {
        tileConfig.width = static_cast<int>(parsed);
        tileConfig.height = static_cast<int>(parsed);
    }
    if (tryGetInt(stage.params, "tile_width", parsed)) {
        tileConfig.width = static_cast<int>(parsed);
    }
    if (tryGetInt(stage.params, "tile_height", parsed)) {
        tileConfig.height = static_cast<int>(parsed);
    }
    if (tryGetInt(stage.params, "tile_overlap", parsed)) {
        tileConfig.overlap = static_cast<int>(parsed);
    }
    if (tryGetInt(stage.params, "tile_batch", parsed)) {
        tileConfig.batch = static_cast<int>(parsed);
        tileConfig.batchExplicit = true;
    }

    if (tileConfig.width <= 0 || tileConfig.height <= 0) {
        error = "Tile dimensions must be positive.";
        return false;
    }
    if (tileConfig.width > 4096 || tileConfig.height > 4096) {
        error = "Tile dimensions are unreasonably large; keep them at or below 4096.";
        return false;
    }
    if (tileConfig.overlap < 0) {
        error = "Tile overlap cannot be negative.";
        return false;
    }
    if (tileConfig.batch <= 0) {
        error = "Tile batch must be positive.";
        return false;
    }
    if (tileConfig.batch > 64) {
        error = "Tile batch is unreasonably large; keep it at or below 64.";
        return false;
    }
    if (tileConfig.overlap * 2 >= tileConfig.width ||
        tileConfig.overlap * 2 >= tileConfig.height) {
        error = "Tile overlap must be less than half of the tile dimensions.";
        return false;
    }
    return true;
}

std::vector<int> buildTileStarts(int fullExtent, int tileExtent, int overlap) {
    std::vector<int> starts = {0};
    if (fullExtent <= tileExtent) {
        return starts;
    }

    const int step = std::max(1, tileExtent - overlap * 2);
    while (starts.back() + tileExtent < fullExtent) {
        const int next = std::min(starts.back() + step, fullExtent - tileExtent);
        if (next <= starts.back()) {
            break;
        }
        starts.push_back(next);
    }
    return starts;
}

TileWindow computeTileWindow(const std::vector<int>& starts,
                             std::size_t index,
                             int tileExtent,
                             int fullExtent) {
    const int tileStart = starts[index];
    const int tileEnd = tileStart + tileExtent;

    TileWindow window;
    if (index == 0u) {
        window.begin = 0;
    } else {
        const int prevEnd = starts[index - 1u] + tileExtent;
        window.begin = (prevEnd + tileStart) / 2;
    }

    if (index + 1u >= starts.size()) {
        window.end = fullExtent;
    } else {
        const int nextStart = starts[index + 1u];
        window.end = (tileEnd + nextStart) / 2;
    }

    window.begin = std::clamp(window.begin, tileStart, std::min(tileEnd, fullExtent));
    window.end = std::clamp(window.end, window.begin, std::min(tileEnd, fullExtent));
    window.offset = window.begin - tileStart;
    return window;
}

inline std::uint8_t floatToRgbByteLocal(float value) {
    value = std::clamp(value * 255.0f, 0.0f, 255.0f);
    return static_cast<std::uint8_t>(value + 0.5f);
}

// fp16 ⇔ float conversion.
// When -march=native enables F16C, delegate to single-cycle hardware
// instructions (_cvtss_sh, _cvtsh_ss).  Fall back to IEEE-conformant
// SW implementation otherwise (e.g. cross-compile, old CPUs).
#if defined(__F16C__)
inline std::uint16_t floatToHalfBitsLocal(float value) {
    return static_cast<std::uint16_t>(_cvtss_sh(value, _MM_FROUND_TO_NEAREST_INT));
}
inline float halfBitsToFloatLocal(std::uint16_t bits) {
    return _cvtsh_ss(bits);
}
#else
inline std::uint16_t floatToHalfBitsLocal(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));

    const std::uint32_t sign = (bits >> 16u) & 0x8000u;
    std::uint32_t mantissa = bits & 0x007fffffu;
    int exponent = static_cast<int>((bits >> 23u) & 0xffu) - 127 + 15;

    if (exponent <= 0) {
        if (exponent < -10) {
            return static_cast<std::uint16_t>(sign);
        }
        mantissa = (mantissa | 0x00800000u) >> static_cast<unsigned>(1 - exponent);
        return static_cast<std::uint16_t>(sign | ((mantissa + 0x00001000u) >> 13u));
    }

    if (exponent >= 31) {
        if (((bits >> 23u) & 0xffu) == 0xffu && mantissa != 0u) {
            return static_cast<std::uint16_t>(sign | 0x7c00u | (mantissa >> 13u) | 1u);
        }
        return static_cast<std::uint16_t>(sign | 0x7c00u);
    }

    return static_cast<std::uint16_t>(
        sign | (static_cast<std::uint32_t>(exponent) << 10u)
             | ((mantissa + 0x00001000u) >> 13u));
}

inline float halfBitsToFloatLocal(std::uint16_t bits) {
    const std::uint32_t sign = (static_cast<std::uint32_t>(bits & 0x8000u)) << 16u;
    std::uint32_t exponent = (static_cast<std::uint32_t>(bits) >> 10u) & 0x1fu;
    std::uint32_t mantissa = static_cast<std::uint32_t>(bits & 0x03ffu);
    std::uint32_t outBits = 0;

    if (exponent == 0u) {
        if (mantissa == 0u) {
            outBits = sign;
        } else {
            exponent = 127u - 15u + 1u;
            while ((mantissa & 0x0400u) == 0u) {
                mantissa <<= 1u;
                --exponent;
            }
            mantissa &= 0x03ffu;
            outBits = sign | (exponent << 23u) | (mantissa << 13u);
        }
    } else if (exponent == 0x1fu) {
        outBits = sign | 0x7f800000u | (mantissa << 13u);
    } else {
        exponent = exponent + (127u - 15u);
        outBits = sign | (exponent << 23u) | (mantissa << 13u);
    }

    float out = 0.0f;
    std::memcpy(&out, &outBits, sizeof(out));
    return out;
}
#endif  // __F16C__

void packRgbTileClampToNchwFp32(const std::uint8_t* source,
                                int sourceWidth,
                                int sourceHeight,
                                int tileX,
                                int tileY,
                                int tileWidth,
                                int tileHeight,
                                float* tensor) {
    const std::size_t hw = static_cast<std::size_t>(tileWidth) *
                           static_cast<std::size_t>(tileHeight);
    float* const rPlane = tensor;
    float* const gPlane = tensor + hw;
    float* const bPlane = tensor + hw * 2u;
    constexpr float kInv255 = 1.0f / 255.0f;

    // Fast path: tile is fully within the source image — skip per-pixel clamping.
    // This branch is taken for all interior tiles and enables auto-vectorization.
    if (tileX >= 0 && tileY >= 0 &&
        tileX + tileWidth  <= sourceWidth &&
        tileY + tileHeight <= sourceHeight) {
        parallelForRows(tileHeight, tileWidth, [&](int beginRow, int endRow) {
            for (int y = beginRow; y < endRow; ++y) {
                const std::size_t srcRowBase =
                    (static_cast<std::size_t>(tileY + y) * static_cast<std::size_t>(sourceWidth)
                     + static_cast<std::size_t>(tileX)) * 3u;
                const std::size_t dstRow = static_cast<std::size_t>(y) *
                                           static_cast<std::size_t>(tileWidth);
                for (int x = 0; x < tileWidth; ++x) {
                    const std::size_t srcOff = srcRowBase + static_cast<std::size_t>(x) * 3u;
                    const std::size_t dstOff = dstRow + static_cast<std::size_t>(x);
                    rPlane[dstOff] = static_cast<float>(source[srcOff + 0u]) * kInv255;
                    gPlane[dstOff] = static_cast<float>(source[srcOff + 1u]) * kInv255;
                    bPlane[dstOff] = static_cast<float>(source[srcOff + 2u]) * kInv255;
                }
            }
        });
        return;
    }

    // Slow path: border tile, clamp coordinates to image edges.
    parallelForRows(tileHeight, tileWidth, [&](int beginRow, int endRow) {
        for (int y = beginRow; y < endRow; ++y) {
            const int srcY = std::clamp(tileY + y, 0, sourceHeight - 1);
            const std::size_t dstRow = static_cast<std::size_t>(y) *
                                       static_cast<std::size_t>(tileWidth);
            for (int x = 0; x < tileWidth; ++x) {
                const int srcX = std::clamp(tileX + x, 0, sourceWidth - 1);
                const std::size_t srcOffset =
                    (static_cast<std::size_t>(srcY) * static_cast<std::size_t>(sourceWidth)
                     + static_cast<std::size_t>(srcX)) * 3u;
                const std::size_t dstOffset = dstRow + static_cast<std::size_t>(x);
                rPlane[dstOffset] = static_cast<float>(source[srcOffset + 0u]) * kInv255;
                gPlane[dstOffset] = static_cast<float>(source[srcOffset + 1u]) * kInv255;
                bPlane[dstOffset] = static_cast<float>(source[srcOffset + 2u]) * kInv255;
            }
        }
    });
}

void packRgbTileClampToNchwFp16(const std::uint8_t* source,
                                int sourceWidth,
                                int sourceHeight,
                                int tileX,
                                int tileY,
                                int tileWidth,
                                int tileHeight,
                                std::uint16_t* tensor) {
    const std::size_t hw = static_cast<std::size_t>(tileWidth) *
                           static_cast<std::size_t>(tileHeight);
    std::uint16_t* const rPlane = tensor;
    std::uint16_t* const gPlane = tensor + hw;
    std::uint16_t* const bPlane = tensor + hw * 2u;
    constexpr float kInv255 = 1.0f / 255.0f;

    // Fast path: tile is fully within the source image — skip per-pixel clamping.
    if (tileX >= 0 && tileY >= 0 &&
        tileX + tileWidth  <= sourceWidth &&
        tileY + tileHeight <= sourceHeight) {
        parallelForRows(tileHeight, tileWidth, [&](int beginRow, int endRow) {
            for (int y = beginRow; y < endRow; ++y) {
                const std::size_t srcRowBase =
                    (static_cast<std::size_t>(tileY + y) * static_cast<std::size_t>(sourceWidth)
                     + static_cast<std::size_t>(tileX)) * 3u;
                const std::size_t dstRow = static_cast<std::size_t>(y) *
                                           static_cast<std::size_t>(tileWidth);
                for (int x = 0; x < tileWidth; ++x) {
                    const std::size_t srcOff = srcRowBase + static_cast<std::size_t>(x) * 3u;
                    const std::size_t dstOff = dstRow + static_cast<std::size_t>(x);
                    rPlane[dstOff] = floatToHalfBitsLocal(static_cast<float>(source[srcOff + 0u]) * kInv255);
                    gPlane[dstOff] = floatToHalfBitsLocal(static_cast<float>(source[srcOff + 1u]) * kInv255);
                    bPlane[dstOff] = floatToHalfBitsLocal(static_cast<float>(source[srcOff + 2u]) * kInv255);
                }
            }
        });
        return;
    }

    // Slow path: border tile.
    parallelForRows(tileHeight, tileWidth, [&](int beginRow, int endRow) {
        for (int y = beginRow; y < endRow; ++y) {
            const int srcY = std::clamp(tileY + y, 0, sourceHeight - 1);
            const std::size_t dstRow = static_cast<std::size_t>(y) *
                                       static_cast<std::size_t>(tileWidth);
            for (int x = 0; x < tileWidth; ++x) {
                const int srcX = std::clamp(tileX + x, 0, sourceWidth - 1);
                const std::size_t srcOffset =
                    (static_cast<std::size_t>(srcY) * static_cast<std::size_t>(sourceWidth)
                     + static_cast<std::size_t>(srcX)) * 3u;
                const std::size_t dstOffset = dstRow + static_cast<std::size_t>(x);
                rPlane[dstOffset] = floatToHalfBitsLocal(static_cast<float>(source[srcOffset + 0u]) * kInv255);
                gPlane[dstOffset] = floatToHalfBitsLocal(static_cast<float>(source[srcOffset + 1u]) * kInv255);
                bPlane[dstOffset] = floatToHalfBitsLocal(static_cast<float>(source[srcOffset + 2u]) * kInv255);
            }
        }
    });
}

struct PackedRgbFrameView {
    const std::uint8_t* data = nullptr;
    int linesize = 0;
    int width = 0;
    int height = 0;
    int pixelStride = 0;
    int redOffset = 0;
    int greenOffset = 1;
    int blueOffset = 2;
};

bool makePackedRgbFrameView(const AVFrame* frame, PackedRgbFrameView& view) {
    view = {};
    if (frame == nullptr || frame->data[0] == nullptr || frame->width <= 0 || frame->height <= 0 ||
        frame->linesize[0] <= 0) {
        return false;
    }

    view.data = frame->data[0];
    view.linesize = frame->linesize[0];
    view.width = frame->width;
    view.height = frame->height;

    switch (static_cast<AVPixelFormat>(frame->format)) {
        case AV_PIX_FMT_RGB24:
            view.pixelStride = 3;
            view.redOffset = 0;
            view.greenOffset = 1;
            view.blueOffset = 2;
            return true;
        case AV_PIX_FMT_BGR24:
            view.pixelStride = 3;
            view.redOffset = 2;
            view.greenOffset = 1;
            view.blueOffset = 0;
            return true;
        case AV_PIX_FMT_RGBA:
            view.pixelStride = 4;
            view.redOffset = 0;
            view.greenOffset = 1;
            view.blueOffset = 2;
            return true;
        case AV_PIX_FMT_BGRA:
            view.pixelStride = 4;
            view.redOffset = 2;
            view.greenOffset = 1;
            view.blueOffset = 0;
            return true;
        case AV_PIX_FMT_ARGB:
            view.pixelStride = 4;
            view.redOffset = 1;
            view.greenOffset = 2;
            view.blueOffset = 3;
            return true;
        case AV_PIX_FMT_ABGR:
            view.pixelStride = 4;
            view.redOffset = 3;
            view.greenOffset = 2;
            view.blueOffset = 1;
            return true;
        default:
            return false;
    }
}

template <typename StorePixel>
void packPackedRgbFrameTileClampGeneric(const PackedRgbFrameView& view,
                                        int tileX,
                                        int tileY,
                                        int tileWidth,
                                        int tileHeight,
                                        StorePixel&& storePixel) {
    const std::size_t hw = static_cast<std::size_t>(tileWidth) *
                           static_cast<std::size_t>(tileHeight);
    parallelForRows(tileHeight, tileWidth, [&](int beginRow, int endRow) {
        for (int y = beginRow; y < endRow; ++y) {
            const int srcY = std::clamp(tileY + y, 0, view.height - 1);
            const auto* srcRow = view.data + static_cast<std::ptrdiff_t>(srcY) * view.linesize;
            const std::size_t dstRow = static_cast<std::size_t>(y) *
                                       static_cast<std::size_t>(tileWidth);
            for (int x = 0; x < tileWidth; ++x) {
                const int srcX = std::clamp(tileX + x, 0, view.width - 1);
                const auto* pixel = srcRow + static_cast<std::ptrdiff_t>(srcX) * view.pixelStride;
                const std::size_t dstOffset = dstRow + static_cast<std::size_t>(x);
                storePixel(dstOffset,
                           pixel[view.redOffset],
                           pixel[view.greenOffset],
                           pixel[view.blueOffset],
                           hw);
            }
        }
    });
}

void packPackedRgbFrameTileClampToNchwFp32(const PackedRgbFrameView& view,
                                           int tileX,
                                           int tileY,
                                           int tileWidth,
                                           int tileHeight,
                                           float* tensor) {
    constexpr float kInv255 = 1.0f / 255.0f;
    packPackedRgbFrameTileClampGeneric(
        view,
        tileX,
        tileY,
        tileWidth,
        tileHeight,
        [&](const std::size_t dstOffset,
            const std::uint8_t red,
            const std::uint8_t green,
            const std::uint8_t blue,
            const std::size_t hw) {
            tensor[dstOffset] = static_cast<float>(red) * kInv255;
            tensor[hw + dstOffset] = static_cast<float>(green) * kInv255;
            tensor[(2u * hw) + dstOffset] = static_cast<float>(blue) * kInv255;
        });
}

void packPackedRgbFrameTileClampToNchwFp16(const PackedRgbFrameView& view,
                                           int tileX,
                                           int tileY,
                                           int tileWidth,
                                           int tileHeight,
                                           std::uint16_t* tensor) {
    constexpr float kInv255 = 1.0f / 255.0f;
    packPackedRgbFrameTileClampGeneric(
        view,
        tileX,
        tileY,
        tileWidth,
        tileHeight,
        [&](const std::size_t dstOffset,
            const std::uint8_t red,
            const std::uint8_t green,
            const std::uint8_t blue,
            const std::size_t hw) {
            tensor[dstOffset] = floatToHalfBitsLocal(static_cast<float>(red) * kInv255);
            tensor[hw + dstOffset] = floatToHalfBitsLocal(static_cast<float>(green) * kInv255);
            tensor[(2u * hw) + dstOffset] = floatToHalfBitsLocal(static_cast<float>(blue) * kInv255);
        });
}

void blitNchwFp32TileRegionToRgb24(const float* tensor,
                                   int tileChannels,
                                   int tileWidth,
                                   int tileHeight,
                                   int srcX,
                                   int srcY,
                                   int copyWidth,
                                   int copyHeight,
                                   std::vector<std::uint8_t>& dst,
                                   int dstWidth,
                                   int dstX,
                                   int dstY) {
    const std::size_t hw = static_cast<std::size_t>(tileWidth) *
                           static_cast<std::size_t>(tileHeight);
    const float* const rPlane = tileChannels > 0 ? tensor : nullptr;
    const float* const gPlane = tileChannels > 1 ? tensor + hw : nullptr;
    const float* const bPlane = tileChannels > 2 ? tensor + (2u * hw) : nullptr;

    parallelForRows(copyHeight, copyWidth, [&](int beginRow, int endRow) {
        for (int y = beginRow; y < endRow; ++y) {
            const std::size_t srcRow =
                static_cast<std::size_t>(srcY + y) * static_cast<std::size_t>(tileWidth)
                + static_cast<std::size_t>(srcX);
            const std::size_t dstRow =
                (static_cast<std::size_t>(dstY + y) * static_cast<std::size_t>(dstWidth)
                 + static_cast<std::size_t>(dstX)) * 3u;
            for (int x = 0; x < copyWidth; ++x) {
                const std::size_t srcIdx = srcRow + static_cast<std::size_t>(x);
                const std::size_t dstIdx = dstRow + static_cast<std::size_t>(x) * 3u;
                dst[dstIdx + 0u] = floatToRgbByteLocal(rPlane[srcIdx]);
                dst[dstIdx + 1u] = floatToRgbByteLocal(gPlane[srcIdx]);
                dst[dstIdx + 2u] = floatToRgbByteLocal(bPlane[srcIdx]);
            }
        }
    });
}

void blitNchwFp16TileRegionToRgb24(const std::uint16_t* tensor,
                                   int tileChannels,
                                   int tileWidth,
                                   int tileHeight,
                                   int srcX,
                                   int srcY,
                                   int copyWidth,
                                   int copyHeight,
                                   std::vector<std::uint8_t>& dst,
                                   int dstWidth,
                                   int dstX,
                                   int dstY) {
    const std::size_t hw = static_cast<std::size_t>(tileWidth) *
                           static_cast<std::size_t>(tileHeight);
    const std::uint16_t* const rPlane = tileChannels > 0 ? tensor : nullptr;
    const std::uint16_t* const gPlane = tileChannels > 1 ? tensor + hw : nullptr;
    const std::uint16_t* const bPlane = tileChannels > 2 ? tensor + (2u * hw) : nullptr;

    parallelForRows(copyHeight, copyWidth, [&](int beginRow, int endRow) {
        for (int y = beginRow; y < endRow; ++y) {
            const std::size_t srcRow =
                static_cast<std::size_t>(srcY + y) * static_cast<std::size_t>(tileWidth)
                + static_cast<std::size_t>(srcX);
            const std::size_t dstRow =
                (static_cast<std::size_t>(dstY + y) * static_cast<std::size_t>(dstWidth)
                 + static_cast<std::size_t>(dstX)) * 3u;
            for (int x = 0; x < copyWidth; ++x) {
                const std::size_t srcIdx = srcRow + static_cast<std::size_t>(x);
                const std::size_t dstIdx = dstRow + static_cast<std::size_t>(x) * 3u;
                dst[dstIdx + 0u] = floatToRgbByteLocal(halfBitsToFloatLocal(rPlane[srcIdx]));
                dst[dstIdx + 1u] = floatToRgbByteLocal(halfBitsToFloatLocal(gPlane[srcIdx]));
                dst[dstIdx + 2u] = floatToRgbByteLocal(halfBitsToFloatLocal(bPlane[srcIdx]));
            }
        }
    });
}

float sampleFp32PlaneBilinear(const float* plane,
                              int width,
                              int height,
                              double x,
                              double y) {
    x = std::clamp(x, 0.0, static_cast<double>(std::max(width - 1, 0)));
    y = std::clamp(y, 0.0, static_cast<double>(std::max(height - 1, 0)));
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = std::min(x0 + 1, width - 1);
    const int y1 = std::min(y0 + 1, height - 1);
    const double fx = x - static_cast<double>(x0);
    const double fy = y - static_cast<double>(y0);

    auto load = [&](int sx, int sy) {
        return static_cast<double>(
            plane[static_cast<std::size_t>(sy) * static_cast<std::size_t>(width)
                + static_cast<std::size_t>(sx)]);
    };

    const double c00 = load(x0, y0);
    const double c10 = load(x1, y0);
    const double c01 = load(x0, y1);
    const double c11 = load(x1, y1);
    const double top = c00 * (1.0 - fx) + c10 * fx;
    const double bottom = c01 * (1.0 - fx) + c11 * fx;
    return static_cast<float>(top * (1.0 - fy) + bottom * fy);
}

float sampleFp16PlaneBilinear(const std::uint16_t* plane,
                              int width,
                              int height,
                              double x,
                              double y) {
    x = std::clamp(x, 0.0, static_cast<double>(std::max(width - 1, 0)));
    y = std::clamp(y, 0.0, static_cast<double>(std::max(height - 1, 0)));
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = std::min(x0 + 1, width - 1);
    const int y1 = std::min(y0 + 1, height - 1);
    const double fx = x - static_cast<double>(x0);
    const double fy = y - static_cast<double>(y0);

    auto load = [&](int sx, int sy) {
        return static_cast<double>(
            halfBitsToFloatLocal(
                plane[static_cast<std::size_t>(sy) * static_cast<std::size_t>(width)
                    + static_cast<std::size_t>(sx)]));
    };

    const double c00 = load(x0, y0);
    const double c10 = load(x1, y0);
    const double c01 = load(x0, y1);
    const double c11 = load(x1, y1);
    const double top = c00 * (1.0 - fx) + c10 * fx;
    const double bottom = c01 * (1.0 - fx) + c11 * fx;
    return static_cast<float>(top * (1.0 - fy) + bottom * fy);
}

void blitNchwFp32TileRegionChannel0ResampledToPlane(const float* tensor,
                                                    int tileWidth,
                                                    int tileHeight,
                                                    int inputTileWidth,
                                                    int inputTileHeight,
                                                    int srcX,
                                                    int srcY,
                                                    int copyWidth,
                                                    int copyHeight,
                                                    std::vector<float>& dst,
                                                    int dstWidth,
                                                    int dstX,
                                                    int dstY) {
    parallelForRows(copyHeight, copyWidth, [&](int beginRow, int endRow) {
        for (int y = beginRow; y < endRow; ++y) {
            const double sampleY =
                ((static_cast<double>(srcY + y) + 0.5) * static_cast<double>(tileHeight)
                 / static_cast<double>(inputTileHeight)) - 0.5;
            const std::size_t dstRow =
                static_cast<std::size_t>(dstY + y) * static_cast<std::size_t>(dstWidth)
                + static_cast<std::size_t>(dstX);
            for (int x = 0; x < copyWidth; ++x) {
                const double sampleX =
                    ((static_cast<double>(srcX + x) + 0.5) * static_cast<double>(tileWidth)
                     / static_cast<double>(inputTileWidth)) - 0.5;
                dst[dstRow + static_cast<std::size_t>(x)] =
                    sampleFp32PlaneBilinear(tensor, tileWidth, tileHeight, sampleX, sampleY);
            }
        }
    });
}

void blitNchwFp16TileRegionChannel0ResampledToPlane(const std::uint16_t* tensor,
                                                    int tileWidth,
                                                    int tileHeight,
                                                    int inputTileWidth,
                                                    int inputTileHeight,
                                                    int srcX,
                                                    int srcY,
                                                    int copyWidth,
                                                    int copyHeight,
                                                    std::vector<float>& dst,
                                                    int dstWidth,
                                                    int dstX,
                                                    int dstY) {
    parallelForRows(copyHeight, copyWidth, [&](int beginRow, int endRow) {
        for (int y = beginRow; y < endRow; ++y) {
            const double sampleY =
                ((static_cast<double>(srcY + y) + 0.5) * static_cast<double>(tileHeight)
                 / static_cast<double>(inputTileHeight)) - 0.5;
            const std::size_t dstRow =
                static_cast<std::size_t>(dstY + y) * static_cast<std::size_t>(dstWidth)
                + static_cast<std::size_t>(dstX);
            for (int x = 0; x < copyWidth; ++x) {
                const double sampleX =
                    ((static_cast<double>(srcX + x) + 0.5) * static_cast<double>(tileWidth)
                     / static_cast<double>(inputTileWidth)) - 0.5;
                dst[dstRow + static_cast<std::size_t>(x)] =
                    sampleFp16PlaneBilinear(tensor, tileWidth, tileHeight, sampleX, sampleY);
            }
        }
    });
}

std::string toLowerCopy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

std::string trimCopy(const std::string& value) {
    std::size_t begin = 0;
    while (begin < value.size() &&
           std::isspace(static_cast<unsigned char>(value[begin])) != 0) {
        ++begin;
    }
    std::size_t end = value.size();
    while (end > begin &&
           std::isspace(static_cast<unsigned char>(value[end - 1])) != 0) {
        --end;
    }
    return value.substr(begin, end - begin);
}

bool parseBoolValue(const std::string& value, bool& out) {
    const std::string normalized = toLowerCopy(trimCopy(value));
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

std::vector<std::string> splitString(const std::string& value, char delimiter) {
    std::vector<std::string> out;
    std::size_t begin = 0;
    while (begin <= value.size()) {
        std::size_t end = value.find(delimiter, begin);
        if (end == std::string::npos) {
            end = value.size();
        }
        out.push_back(value.substr(begin, end - begin));
        if (end == value.size()) {
            break;
        }
        begin = end + 1u;
    }
    return out;
}

double clamp01(double value) {
    return std::clamp(value, 0.0, 1.0);
}

bool getStageBoolParam(const EnhancementStage& stage,
                       const std::string& key,
                       bool fallback) {
    bool value = fallback;
    if (tryGetBool(stage, stage.kind, key, value)) {
        return value;
    }
    return fallback;
}

std::string getStageStringParam(const EnhancementStage& stage,
                                const std::string& key,
                                const std::string& fallback) {
    std::string value;
    if (tryGetString(stage, stage.kind, key, value)) {
        return value;
    }
    return fallback;
}

double getStageDoubleParam(const EnhancementStage& stage,
                           const std::string& key,
                           double fallback) {
    double value = fallback;
    if (tryGetDouble(stage, stage.kind, key, value)) {
        return value;
    }
    return fallback;
}

int getStageIntParam(const EnhancementStage& stage,
                     const std::string& key,
                     int fallback) {
    std::int64_t value = fallback;
    if (tryGetInt(stage, stage.kind, key, value) &&
        value >= static_cast<std::int64_t>(std::numeric_limits<int>::min()) &&
        value <= static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
        return static_cast<int>(value);
    }
    return fallback;
}

std::string normalizeControlToken(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) {
                       if (std::isalnum(ch) != 0) {
                           return static_cast<char>(std::tolower(ch));
                       }
                       return '_';
                   });
    std::string normalized;
    normalized.reserve(value.size());
    bool previousUnderscore = false;
    for (const char ch : value) {
        if (ch == '_') {
            if (!previousUnderscore) {
                normalized.push_back(ch);
            }
            previousUnderscore = true;
        } else {
            normalized.push_back(ch);
            previousUnderscore = false;
        }
    }
    while (!normalized.empty() && normalized.front() == '_') {
        normalized.erase(normalized.begin());
    }
    while (!normalized.empty() && normalized.back() == '_') {
        normalized.pop_back();
    }
    return normalized;
}

std::vector<StageKind> parseStageKindsCsvLocal(const std::string& value) {
    std::vector<StageKind> out;
    std::size_t begin = 0;
    while (begin <= value.size()) {
        std::size_t end = value.find(',', begin);
        if (end == std::string::npos) {
            end = value.size();
        }
        const std::string token = value.substr(begin, end - begin);
        if (!token.empty()) {
            if (const auto kind = stageKindFromString(token); kind.has_value()) {
                out.push_back(*kind);
            }
        }
        if (end == value.size()) {
            break;
        }
        begin = end + 1u;
    }
    return out;
}

std::unordered_map<std::string, std::string> parseControlBindingsMap(
    const std::string& value) {
    std::unordered_map<std::string, std::string> bindings;
    std::size_t begin = 0;
    while (begin <= value.size()) {
        std::size_t end = value.find(';', begin);
        if (end == std::string::npos) {
            end = value.size();
        }
        const std::string token = value.substr(begin, end - begin);
        const std::size_t equals = token.find('=');
        if (equals != std::string::npos) {
            const std::string inputName = normalizeControlToken(
                trimCopy(token.substr(0, equals)));
            const std::string binding = trimCopy(token.substr(equals + 1));
            if (!inputName.empty() && !binding.empty()) {
                bindings[inputName] = binding;
            }
        }
        if (end == value.size()) {
            break;
        }
        begin = end + 1u;
    }
    return bindings;
}

std::vector<StageKind> stageControlScopes(const EnhancementStage& stage) {
    std::vector<StageKind> scopes =
        parseStageKindsCsvLocal(getStageStringParam(stage, "model_capabilities", ""));
    if (std::find(scopes.begin(), scopes.end(), stage.kind) == scopes.end()) {
        scopes.push_back(stage.kind);
    }
    return scopes;
}

bool stageCapabilityRequested(const EnhancementStage& stage, const StageKind capability) {
    const auto requested =
        parseStageKindsCsvLocal(getStageStringParam(stage, "fused_requested_capabilities", ""));
    if (requested.empty()) {
        return capability == stage.kind;
    }
    return std::find(requested.begin(), requested.end(), capability) != requested.end();
}

bool matchScopedControlName(const std::string& normalizedInputName,
                            const StageKind scope,
                            std::string& outKey,
                            bool& outIsEnableFlag) {
    const std::string token = normalizeControlToken(toString(scope));
    if (normalizedInputName == token) {
        outKey = "enabled";
        outIsEnableFlag = true;
        return true;
    }

    const std::vector<std::string> enablePrefixes = {
        "enable_" + token,
        "enabled_" + token,
        "use_" + token,
        "apply_" + token,
        "capability_" + token,
    };
    for (const auto& prefix : enablePrefixes) {
        if (normalizedInputName == prefix) {
            outKey = "enabled";
            outIsEnableFlag = true;
            return true;
        }
    }

    const std::vector<std::string> enableSuffixes = {
        token + "_enable",
        token + "_enabled",
        token + "_use",
        token + "_apply",
    };
    for (const auto& suffix : enableSuffixes) {
        if (normalizedInputName == suffix) {
            outKey = "enabled";
            outIsEnableFlag = true;
            return true;
        }
    }

    const std::string scopedPrefix = token + "_";
    if (normalizedInputName.rfind(scopedPrefix, 0) == 0 &&
        normalizedInputName.size() > scopedPrefix.size()) {
        outKey = normalizedInputName.substr(scopedPrefix.size());
        outIsEnableFlag = false;
        return true;
    }
    return false;
}

bool resolveBindingExpression(const EnhancementStage& stage,
                             const std::string& binding,
                             double& outValue) {
    const std::string trimmed = trimCopy(binding);
    if (trimmed.empty()) {
        return false;
    }

    if (trimmed.rfind("literal:", 0) == 0) {
        const std::string literal = trimCopy(trimmed.substr(8));
        bool boolValue = false;
        if (parseBoolValue(literal, boolValue)) {
            outValue = boolValue ? 1.0 : 0.0;
            return true;
        }
        try {
            outValue = std::stod(literal);
            return true;
        } catch (...) {
            return false;
        }
    }

    const std::size_t dot = trimmed.find('.');
    if (dot == std::string::npos) {
        if (const auto kind = stageKindFromString(trimmed); kind.has_value()) {
            outValue = stageCapabilityRequested(stage, *kind) ? 1.0 : 0.0;
            return true;
        }

        double doubleValue = 0.0;
        std::int64_t intValue = 0;
        bool boolValue = false;
        if (tryGetDouble(stage, stage.kind, trimmed, doubleValue)) {
            outValue = doubleValue;
            return true;
        }
        if (tryGetInt(stage, stage.kind, trimmed, intValue)) {
            outValue = static_cast<double>(intValue);
            return true;
        }
        if (tryGetBool(stage, stage.kind, trimmed, boolValue)) {
            outValue = boolValue ? 1.0 : 0.0;
            return true;
        }
        return false;
    }

    const std::string stageToken = trimCopy(trimmed.substr(0, dot));
    const std::string keyToken = trimCopy(trimmed.substr(dot + 1));
    if (stageToken.empty() || keyToken.empty()) {
        return false;
    }
    const auto kind = stageKindFromString(stageToken);
    if (!kind.has_value()) {
        return false;
    }
    if (keyToken == "enabled") {
        outValue = stageCapabilityRequested(stage, *kind) ? 1.0 : 0.0;
        return true;
    }

    double doubleValue = 0.0;
    std::int64_t intValue = 0;
    bool boolValue = false;
    if (tryGetDouble(stage, *kind, keyToken, doubleValue)) {
        outValue = doubleValue;
        return true;
    }
    if (tryGetInt(stage, *kind, keyToken, intValue)) {
        outValue = static_cast<double>(intValue);
        return true;
    }
    if (tryGetBool(stage, *kind, keyToken, boolValue)) {
        outValue = boolValue ? 1.0 : 0.0;
        return true;
    }
    return false;
}

bool resolveAuxInputScalar(const EnhancementStage& stage,
                           const std::string& inputName,
                           double& outValue) {
    const std::string normalizedInputName = normalizeControlToken(inputName);
    const auto explicitBindings =
        parseControlBindingsMap(getStageStringParam(stage, "model_control_bindings", ""));
    if (const auto it = explicitBindings.find(normalizedInputName);
        it != explicitBindings.end()) {
        return resolveBindingExpression(stage, it->second, outValue);
    }

    const auto scopes = stageControlScopes(stage);

    for (const auto scope : scopes) {
        std::string scopedKey;
        bool enableFlag = false;
        if (!matchScopedControlName(normalizedInputName, scope, scopedKey, enableFlag)) {
            continue;
        }
        if (enableFlag) {
            outValue = stageCapabilityRequested(stage, scope) ? 1.0 : 0.0;
            return true;
        }

        double doubleValue = 0.0;
        std::int64_t intValue = 0;
        bool boolValue = false;
        if (tryGetDouble(stage, scope, scopedKey, doubleValue)) {
            outValue = doubleValue;
            return true;
        }
        if (tryGetInt(stage, scope, scopedKey, intValue)) {
            outValue = static_cast<double>(intValue);
            return true;
        }
        if (tryGetBool(stage, scope, scopedKey, boolValue)) {
            outValue = boolValue ? 1.0 : 0.0;
            return true;
        }
    }

    double doubleValue = 0.0;
    std::int64_t intValue = 0;
    bool boolValue = false;
    if (tryGetDouble(stage, stage.kind, normalizedInputName, doubleValue)) {
        outValue = doubleValue;
        return true;
    }
    if (tryGetInt(stage, stage.kind, normalizedInputName, intValue)) {
        outValue = static_cast<double>(intValue);
        return true;
    }
    if (tryGetBool(stage, stage.kind, normalizedInputName, boolValue)) {
        outValue = boolValue ? 1.0 : 0.0;
        return true;
    }

    for (const auto scope : scopes) {
        if (scope == stage.kind) {
            continue;
        }
        if (tryGetDouble(stage, scope, normalizedInputName, doubleValue)) {
            outValue = doubleValue;
            return true;
        }
        if (tryGetInt(stage, scope, normalizedInputName, intValue)) {
            outValue = static_cast<double>(intValue);
            return true;
        }
        if (tryGetBool(stage, scope, normalizedInputName, boolValue)) {
            outValue = boolValue ? 1.0 : 0.0;
            return true;
        }
    }

    return false;
}

struct AuxInputBuffer {
    TensorDtype dtype = TensorDtype::Unknown;
    std::vector<float> fp32;
    std::vector<std::uint16_t> fp16;
    std::vector<std::int8_t> int8;

    void* data() {
        switch (dtype) {
            case TensorDtype::Fp32: return fp32.data();
            case TensorDtype::Fp16: return fp16.data();
            case TensorDtype::Int8: return int8.data();
            default: return nullptr;
        }
    }
};

bool buildAuxInputBuffer(const EnhancementStage& stage,
                         const TensorContract& contract,
                         AuxInputBuffer& buffer,
                         std::string& error) {
    const std::size_t elementCount = static_cast<std::size_t>(contract.shape.elements());
    if (elementCount == 0u) {
        error = "Aux input '" + contract.name + "' has an empty tensor contract.";
        return false;
    }

    double scalarValue = 0.0;
    (void)resolveAuxInputScalar(stage, contract.name, scalarValue);

    buffer.dtype = contract.dtype;
    switch (contract.dtype) {
        case TensorDtype::Fp32:
            buffer.fp32.assign(elementCount, static_cast<float>(scalarValue));
            return true;
        case TensorDtype::Fp16:
            buffer.fp16.assign(elementCount, floatToHalfBitsLocal(static_cast<float>(scalarValue)));
            return true;
        case TensorDtype::Int8: {
            const auto clamped = static_cast<std::int8_t>(std::clamp(
                static_cast<long long>(std::llround(scalarValue)),
                static_cast<long long>(std::numeric_limits<std::int8_t>::min()),
                static_cast<long long>(std::numeric_limits<std::int8_t>::max())));
            buffer.int8.assign(elementCount, clamped);
            return true;
        }
        default:
            error = "Unsupported auxiliary input dtype for '" + contract.name +
                    "': " + toString(contract.dtype) + '.';
            return false;
    }
}

bool stagePreservesNativeResolution(const EnhancementStage& stage) {
    if (stage.kind != StageKind::Deblur) {
        return false;
    }
    if (getStageBoolParam(stage, "allow_scale_output", false) ||
        getStageBoolParam(stage, "keep_model_scale", false)) {
        return false;
    }
    return getStageBoolParam(
        stage,
        "preserve_source_resolution",
        getStageBoolParam(stage, "native_resolution", true));
}

int roundStereoDepthResolutionUp(int resolution) {
    resolution = std::clamp(resolution, kMinimumStereoDepthResolution, 4096);
    const int remainder = resolution % kStereoDepthResolutionStride;
    if (remainder != 0) {
        resolution += (kStereoDepthResolutionStride - remainder);
    }
    return resolution;
}

int resolveStereoDepthResolution(const EnhancementStage& stage,
                                 int sourceWidth,
                                 int sourceHeight,
                                 bool& limitedToSource,
                                 int& requestedResolution) {
    requestedResolution = getStageIntParam(
        stage, "resolution",
        getStageIntParam(stage, "depth_resolution", kDefaultStereoDepthResolution));
    requestedResolution = std::clamp(requestedResolution, kMinimumStereoDepthResolution, 4096);

    limitedToSource = false;
    if (getStageBoolParam(stage, "limit_resolution", false) &&
        sourceWidth > 0 && sourceHeight > 0) {
        const int sourceMinExtent = std::min(sourceWidth, sourceHeight);
        if (requestedResolution > sourceMinExtent) {
            int limitedExtent = sourceMinExtent;
            limitedExtent -= limitedExtent % kStereoDepthResolutionStride;
            if (limitedExtent < kMinimumStereoDepthResolution) {
                limitedExtent = kMinimumStereoDepthResolution;
            }
            limitedToSource = true;
            return limitedExtent;
        }
    }

    return roundStereoDepthResolutionUp(requestedResolution);
}

enum class StereoOutputFormat {
    FullSbs,
    HalfSbs,
    FullTb,
    HalfTb,
    CrossEyed,
    Anaglyph
};

enum class SyntheticViewMode {
    Both,
    Right,
    Left
};

enum class StereoPadMode {
    Tblr,
    Tb,
    Lr,
    Aspect16x9,
    Top
};

struct StereoRgbImage {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> pixels;
};

struct StereoStageConfig {
    StereoOutputFormat format = StereoOutputFormat::FullSbs;
    SyntheticViewMode syntheticView = SyntheticViewMode::Both;
    StereoPadMode padMode = StereoPadMode::Tblr;
    std::string method = "grid_sample";
    std::string mapper;
    std::string mapperType = "auto";
    double divergence = 2.0;
    double convergence = 0.5;
    double foregroundScale = 0.0;
    double ipdOffset = 0.0;
    double pad = 0.0;
    double emaDecay = 0.75;
    bool metricDepth = false;
    bool depthAa = true;
    bool emaNormalize = false;
    int edgeDilationX = 2;
    int edgeDilationY = 1;
};

StereoOutputFormat parseStereoOutputFormat(const std::string& raw) {
    const std::string value = toLowerCopy(raw);
    if (value == "half_sbs" || value == "half-sbs") {
        return StereoOutputFormat::HalfSbs;
    }
    if (value == "full_tb" || value == "full-tb" || value == "tb" || value == "top_bottom") {
        return StereoOutputFormat::FullTb;
    }
    if (value == "half_tb" || value == "half-tb") {
        return StereoOutputFormat::HalfTb;
    }
    if (value == "cross_eyed" || value == "cross-eyed" || value == "cross") {
        return StereoOutputFormat::CrossEyed;
    }
    if (value == "anaglyph" || value == "anaglyph_redcyan") {
        return StereoOutputFormat::Anaglyph;
    }
    return StereoOutputFormat::FullSbs;
}

SyntheticViewMode parseSyntheticViewMode(const std::string& raw) {
    const std::string value = toLowerCopy(raw);
    if (value == "right") {
        return SyntheticViewMode::Right;
    }
    if (value == "left") {
        return SyntheticViewMode::Left;
    }
    return SyntheticViewMode::Both;
}

StereoPadMode parseStereoPadMode(const std::string& raw) {
    const std::string value = toLowerCopy(raw);
    if (value == "tb") {
        return StereoPadMode::Tb;
    }
    if (value == "lr") {
        return StereoPadMode::Lr;
    }
    if (value == "16:9") {
        return StereoPadMode::Aspect16x9;
    }
    if (value == "top") {
        return StereoPadMode::Top;
    }
    return StereoPadMode::Tblr;
}

StereoStageConfig buildStereoStageConfig(const EnhancementStage& stage) {
    StereoStageConfig config;
    config.format = parseStereoOutputFormat(
        getStageStringParam(stage, "format", "full_sbs"));
    config.syntheticView = parseSyntheticViewMode(
        getStageStringParam(stage, "synthetic_view", "both"));
    config.padMode = parseStereoPadMode(
        getStageStringParam(stage, "pad_mode", "tblr"));
    config.method = toLowerCopy(getStageStringParam(stage, "method", "grid_sample"));
    config.mapper = getStageStringParam(stage, "mapper", "");
    config.mapperType = toLowerCopy(getStageStringParam(stage, "mapper_type", "auto"));
    config.divergence = getStageDoubleParam(stage, "divergence", 2.0);
    config.convergence = getStageDoubleParam(stage, "convergence", 0.5);
    config.foregroundScale = getStageDoubleParam(stage, "foreground_scale", 0.0);
    config.ipdOffset = getStageDoubleParam(stage, "ipd_offset", 0.0);
    config.pad = std::max(0.0, getStageDoubleParam(stage, "pad", 0.0));
    config.emaDecay = std::clamp(getStageDoubleParam(stage, "ema_decay", 0.75), 0.0, 0.99);
    config.metricDepth = getStageBoolParam(stage, "metric_depth", false);
    config.depthAa = getStageBoolParam(stage, "depth_aa", true);
    config.emaNormalize = getStageBoolParam(stage, "ema_normalize", false);
    config.edgeDilationX = std::max(0, getStageIntParam(stage, "edge_dilation_x", 2));
    config.edgeDilationY = std::max(0, getStageIntParam(stage, "edge_dilation_y", 1));
    return config;
}

struct ImageSize {
    int width = 0;
    int height = 0;
};

ImageSize applyStereoEyePaddingSize(ImageSize size,
                                    const StereoStageConfig& config) {
    int ipdPad = static_cast<int>(std::abs(config.ipdOffset) * 0.01 * static_cast<double>(std::max(size.width, size.height)));
    ipdPad -= ipdPad % 2;
    if (ipdPad > 0) {
        size.width += ipdPad * 3;
    }

    if (config.pad > 0.0 || config.padMode == StereoPadMode::Aspect16x9) {
        switch (config.padMode) {
            case StereoPadMode::Tblr: {
                const int padH = static_cast<int>(std::round(static_cast<double>(size.height) * config.pad)) / 2;
                const int padW = static_cast<int>(std::round(static_cast<double>(size.width) * config.pad)) / 2;
                size.width += padW * 2;
                size.height += padH * 2;
                break;
            }
            case StereoPadMode::Tb: {
                const int padH = static_cast<int>(std::round(static_cast<double>(size.height) * config.pad)) / 2;
                size.height += padH * 2;
                break;
            }
            case StereoPadMode::Lr: {
                const int padW = static_cast<int>(std::round(static_cast<double>(size.width) * config.pad)) / 2;
                size.width += padW * 2;
                break;
            }
            case StereoPadMode::Top: {
                const int padTop = static_cast<int>(std::round(static_cast<double>(size.height) * config.pad));
                size.height += padTop;
                break;
            }
            case StereoPadMode::Aspect16x9: {
                constexpr double kTargetRatio = 16.0 / 9.0;
                const double currentRatio =
                    size.height > 0 ? static_cast<double>(size.width) / static_cast<double>(size.height) : kTargetRatio;
                if (std::abs(currentRatio - kTargetRatio) > 1e-3) {
                    if (currentRatio > kTargetRatio) {
                        const int targetHeight = static_cast<int>(
                            std::round(static_cast<double>(size.width) / kTargetRatio));
                        const int padH = std::max(0, (targetHeight - size.height) / 2);
                        size.height += padH * 2;
                    } else {
                        const int targetWidth = static_cast<int>(
                            std::round(static_cast<double>(size.height) * kTargetRatio));
                        const int padW = std::max(0, (targetWidth - size.width) / 2);
                        size.width += padW * 2;
                    }
                }
                break;
            }
        }
    }

    switch (config.format) {
        case StereoOutputFormat::HalfSbs:
            size.width = std::max(1, size.width / 2);
            break;
        case StereoOutputFormat::HalfTb:
            size.height = std::max(1, size.height / 2);
            break;
        default:
            break;
    }

    return size;
}

ImageSize computeStereoOutputSize(int baseWidth,
                                  int baseHeight,
                                  const StereoStageConfig& config) {
    const ImageSize eye = applyStereoEyePaddingSize({baseWidth, baseHeight}, config);
    switch (config.format) {
        case StereoOutputFormat::FullTb:
        case StereoOutputFormat::HalfTb:
            return {eye.width, eye.height * 2};
        case StereoOutputFormat::Anaglyph:
            return eye;
        case StereoOutputFormat::CrossEyed:
        case StereoOutputFormat::FullSbs:
        case StereoOutputFormat::HalfSbs:
        default:
            return {eye.width * 2, eye.height};
    }
}

using MapperFn = std::function<double(double)>;

double softplus01Legacy(double x, double c = 6.0) {
    const double minV = std::log1p(std::exp(0.0 * 12.0 - c)) / (12.0 - c);
    const double maxV = std::log1p(std::exp(1.0 * 12.0 - c)) / (12.0 - c);
    const double v = std::log1p(std::exp(x * 12.0 - c)) / (12.0 - c);
    return (v - minV) / (maxV - minV);
}

double softplus01(double x, double bias, double scale) {
    const double minV = std::log1p(std::exp((0.0 - bias) * scale));
    const double maxV = std::log1p(std::exp((1.0 - bias) * scale));
    const double v = std::log1p(std::exp((x - bias) * scale));
    return (v - minV) / (maxV - minV);
}

double invSoftplus01(double x, double bias, double scale) {
    auto inv = [](double value) {
        return std::log(std::max(std::expm1(value), 1e-6));
    };
    const double minV = inv((0.0 - bias) * scale);
    const double maxV = inv((1.0 - bias) * scale);
    const double v = inv((x - bias) * scale);
    return (v - minV) / (maxV - minV);
}

double distanceToDisparity(double x, double c) {
    const double c1 = 1.0 + c;
    const double minV = c / c1;
    return ((c / (c1 - x)) - minV) / (1.0 - minV);
}

double shiftRelativeDepth(double x, double minDistance, double maxDistance = 16.0) {
    const double provisionalMaxDistance = minDistance + maxDistance;
    const double a = 1.0 / provisionalMaxDistance;
    const double b = (1.0 / minDistance) - (1.0 / provisionalMaxDistance);
    double distance = 1.0 / (a + b * x);
    distance = (1.0 - minDistance) + distance;
    const double newX = 1.0 / distance;
    const double minValue = 1.0 / (maxDistance + 1.0);
    const double valueRange = 1.0 - minValue;
    return (newX - minValue) / valueRange;
}

MapperFn resolveBaseMapperFunction(const std::string& rawName) {
    const std::string name = toLowerCopy(rawName);
    if (name == "pow2") {
        return [](double x) { return x * x; };
    }
    if (name == "softplus") {
        return [](double x) { return softplus01Legacy(x); };
    }
    if (name == "softplus2") {
        return [](double x) {
            const double value = softplus01Legacy(x);
            return value * value;
        };
    }
    if (name == "mul_1") {
        return [](double x) { return softplus01(x, 0.343, 12.0); };
    }
    if (name == "mul_2") {
        return [](double x) { return softplus01(x, 0.515, 12.0); };
    }
    if (name == "mul_3") {
        return [](double x) { return softplus01(x, 0.687, 12.0); };
    }
    if (name == "inv_mul_1") {
        return [](double x) { return invSoftplus01(x, -0.002102, 7.8788); };
    }
    if (name == "inv_mul_2") {
        return [](double x) { return invSoftplus01(x, -0.0003, 6.2626); };
    }
    if (name == "inv_mul_3") {
        return [](double x) { return invSoftplus01(x, -0.0001, 3.4343); };
    }
    if (name == "shift_30") {
        return [](double x) { return shiftRelativeDepth(x, 3.0); };
    }
    if (name == "shift_20") {
        return [](double x) { return shiftRelativeDepth(x, 2.0); };
    }
    if (name == "shift_14") {
        return [](double x) { return shiftRelativeDepth(x, 1.4); };
    }
    if (name == "shift_08") {
        return [](double x) { return shiftRelativeDepth(x, 0.8); };
    }
    if (name == "shift_06") {
        return [](double x) { return shiftRelativeDepth(x, 0.6); };
    }
    if (name == "shift_045") {
        return [](double x) { return shiftRelativeDepth(x, 0.45); };
    }
    if (name == "div_25") {
        return [](double x) { return distanceToDisparity(x, 2.5); };
    }
    if (name == "div_10") {
        return [](double x) { return distanceToDisparity(x, 1.0); };
    }
    if (name == "div_6") {
        return [](double x) { return distanceToDisparity(x, 0.6); };
    }
    if (name == "div_4") {
        return [](double x) { return distanceToDisparity(x, 0.4); };
    }
    if (name == "div_2") {
        return [](double x) { return distanceToDisparity(x, 0.2); };
    }
    if (name == "div_1") {
        return [](double x) { return distanceToDisparity(x, 0.1); };
    }
    return [](double x) { return x; };
}

MapperFn resolveMapperFunction(const std::string& mapperSpec) {
    if (mapperSpec.empty()) {
        return [](double x) { return x; };
    }

    MapperFn fn = [](double x) { return x; };
    for (const std::string& rawPart : splitString(mapperSpec, ':')) {
        const std::string part = toLowerCopy(rawPart);
        MapperFn partFn;
        const std::size_t plusPos = part.find('+');
        if (plusPos != std::string::npos) {
            const std::size_t eqPos = part.find('=');
            const std::string lhs = part.substr(0, plusPos);
            const std::string rhs = part.substr(plusPos + 1u,
                eqPos == std::string::npos ? std::string::npos : eqPos - plusPos - 1u);
            double weight = 0.5;
            if (eqPos != std::string::npos && eqPos + 1u < part.size()) {
                try {
                    weight = std::stod(part.substr(eqPos + 1u));
                } catch (...) {
                    weight = 0.5;
                }
            }
            weight = std::clamp(weight, 0.0, 1.0);
            const MapperFn a = resolveBaseMapperFunction(lhs);
            const MapperFn b = resolveBaseMapperFunction(rhs);
            partFn = [a, b, weight](double x) {
                return a(x) * (1.0 - weight) + b(x) * weight;
            };
        } else {
            partFn = resolveBaseMapperFunction(part);
        }

        const MapperFn previous = fn;
        fn = [previous, partFn](double x) {
            return partFn(previous(x));
        };
    }
    return fn;
}

std::string resolveStereoMapperName(const StereoStageConfig& config) {
    const std::string explicitMapper = toLowerCopy(config.mapper);
    if (!explicitMapper.empty()) {
        if (explicitMapper == "auto") {
            return config.metricDepth ? "div_6" : "none";
        }
        return explicitMapper;
    }

    const auto chooseLevels = [&]() {
        if (config.metricDepth || config.mapperType == "div") {
            return std::vector<std::string>{"none", "div_25", "div_10", "div_6", "div_4", "div_2", "div_1"};
        }
        if (config.mapperType == "shift") {
            return std::vector<std::string>{"shift_045", "shift_06", "shift_08", "none", "shift_14", "shift_20", "shift_30"};
        }
        return std::vector<std::string>{"inv_mul_3", "inv_mul_2", "inv_mul_1", "none", "mul_1", "mul_2", "mul_3"};
    };

    const std::vector<std::string> levels = chooseLevels();
    const double scale = std::clamp(config.foregroundScale, -3.0, 3.0);
    if (std::abs(scale - std::round(scale)) < 1e-6) {
        const int index = static_cast<int>(std::round(scale)) + 3;
        return levels[static_cast<std::size_t>(std::clamp(index, 0, 6))];
    }

    const double absScale = std::abs(scale);
    int a = static_cast<int>(std::floor(absScale));
    int b = static_cast<int>(std::ceil(absScale));
    double weight = absScale - static_cast<double>(a);
    if (scale < 0.0) {
        a = -a;
        b = -b;
    }
    const std::string mapperA = levels[static_cast<std::size_t>(std::clamp(a + 3, 0, 6))];
    const std::string mapperB = levels[static_cast<std::size_t>(std::clamp(b + 3, 0, 6))];
    std::ostringstream os;
    os << mapperA << '+' << mapperB << '=' << std::round(weight * 100.0) / 100.0;
    return os.str();
}

void normalizeDepthMap(std::vector<float>& depth,
                       bool emaNormalize,
                       double emaDecay,
                       std::optional<double>& emaMin,
                       std::optional<double>& emaMax) {
    double frameMin = std::numeric_limits<double>::infinity();
    double frameMax = -std::numeric_limits<double>::infinity();
    for (float value : depth) {
        if (!std::isfinite(value)) {
            continue;
        }
        frameMin = std::min(frameMin, static_cast<double>(value));
        frameMax = std::max(frameMax, static_cast<double>(value));
    }
    if (!std::isfinite(frameMin) || !std::isfinite(frameMax)) {
        std::fill(depth.begin(), depth.end(), 0.5f);
        return;
    }

    double minValue = frameMin;
    double maxValue = frameMax;
    if (emaNormalize) {
        if (!emaMin.has_value() || !emaMax.has_value()) {
            emaMin = frameMin;
            emaMax = frameMax;
        } else {
            *emaMin = emaDecay * *emaMin + (1.0 - emaDecay) * frameMin;
            *emaMax = emaDecay * *emaMax + (1.0 - emaDecay) * frameMax;
        }
        minValue = *emaMin;
        maxValue = *emaMax;
    }

    if (maxValue <= minValue + 1e-6) {
        maxValue = minValue + 1e-6;
    }
    const double invRange = 1.0 / (maxValue - minValue);
    for (float& value : depth) {
        if (!std::isfinite(value)) {
            value = 0.5f;
            continue;
        }
        value = static_cast<float>(clamp01((static_cast<double>(value) - minValue) * invRange));
    }
}

std::vector<float> gaussianBlurDepth(const std::vector<float>& input,
                                     int width,
                                     int height) {
    static constexpr int kKernel[3][3] = {
        {21, 31, 21},
        {31, 48, 31},
        {21, 31, 21},
    };
    std::vector<float> output(input.size(), 0.0f);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double accum = 0.0;
            for (int ky = -1; ky <= 1; ++ky) {
                const int sy = std::clamp(y + ky, 0, height - 1);
                for (int kx = -1; kx <= 1; ++kx) {
                    const int sx = std::clamp(x + kx, 0, width - 1);
                    accum += static_cast<double>(input[static_cast<std::size_t>(sy) * static_cast<std::size_t>(width)
                                                       + static_cast<std::size_t>(sx)])
                        * static_cast<double>(kKernel[ky + 1][kx + 1]);
                }
            }
            output[static_cast<std::size_t>(y) * static_cast<std::size_t>(width)
                   + static_cast<std::size_t>(x)] = static_cast<float>(accum / 256.0);
        }
    }
    return output;
}

std::vector<float> maxFilterDepth(const std::vector<float>& input,
                                  int width,
                                  int height,
                                  int radiusX,
                                  int radiusY) {
    std::vector<float> output(input.size(), 0.0f);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float value = 0.0f;
            for (int ky = -radiusY; ky <= radiusY; ++ky) {
                const int sy = std::clamp(y + ky, 0, height - 1);
                for (int kx = -radiusX; kx <= radiusX; ++kx) {
                    const int sx = std::clamp(x + kx, 0, width - 1);
                    value = std::max(
                        value,
                        input[static_cast<std::size_t>(sy) * static_cast<std::size_t>(width)
                              + static_cast<std::size_t>(sx)]);
                }
            }
            output[static_cast<std::size_t>(y) * static_cast<std::size_t>(width)
                   + static_cast<std::size_t>(x)] = value;
        }
    }
    return output;
}

std::vector<float> computeDepthEdgeWeights(const std::vector<float>& input,
                                           int width,
                                           int height) {
    std::vector<float> weights(input.size(), 0.0f);
    float globalMin = std::numeric_limits<float>::infinity();
    float globalMax = -std::numeric_limits<float>::infinity();
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float localMin = std::numeric_limits<float>::infinity();
            float localMax = -std::numeric_limits<float>::infinity();
            for (int ky = -1; ky <= 1; ++ky) {
                const int sy = std::clamp(y + ky, 0, height - 1);
                for (int kx = -1; kx <= 1; ++kx) {
                    const int sx = std::clamp(x + kx, 0, width - 1);
                    const float value =
                        input[static_cast<std::size_t>(sy) * static_cast<std::size_t>(width)
                              + static_cast<std::size_t>(sx)];
                    localMin = std::min(localMin, value);
                    localMax = std::max(localMax, value);
                }
            }
            const float range = localMax - localMin;
            weights[static_cast<std::size_t>(y) * static_cast<std::size_t>(width)
                    + static_cast<std::size_t>(x)] = range;
            globalMin = std::min(globalMin, range);
            globalMax = std::max(globalMax, range);
        }
    }
    const float denom = std::max(globalMax - globalMin, 1e-6f);
    for (float& value : weights) {
        value = std::clamp((value - globalMin) / denom, 0.0f, 1.0f);
    }
    return weights;
}

void applyDepthEdgeBlend(std::vector<float>& depth,
                         int width,
                         int height,
                         int radiusX,
                         int radiusY) {
    const std::vector<float> weights = computeDepthEdgeWeights(depth, width, height);
    const std::vector<float> blurred = gaussianBlurDepth(depth, width, height);
    const std::vector<float> dilated = maxFilterDepth(blurred, width, height, radiusX, radiusY);
    for (std::size_t i = 0; i < depth.size(); ++i) {
        depth[i] = std::clamp(depth[i] * (1.0f - weights[i]) + dilated[i] * weights[i], 0.0f, 1.0f);
    }
}

void postprocessDepthMap(std::vector<float>& depth,
                         int width,
                         int height,
                         const StereoStageConfig& config,
                         std::optional<double>& emaMin,
                         std::optional<double>& emaMax) {
    normalizeDepthMap(depth, config.emaNormalize, config.emaDecay, emaMin, emaMax);
    const std::string mapperName = resolveStereoMapperName(config);
    const MapperFn mapperFn = resolveMapperFunction(mapperName);
    for (float& value : depth) {
        value = static_cast<float>(clamp01(mapperFn(static_cast<double>(value))));
    }

    if (config.depthAa) {
        applyDepthEdgeBlend(depth, width, height, 1, 1);
    }

    const int sharedIterations = std::min(config.edgeDilationX, config.edgeDilationY);
    for (int i = 0; i < sharedIterations; ++i) {
        applyDepthEdgeBlend(depth, width, height, 1, 1);
    }
    for (int i = sharedIterations; i < config.edgeDilationX; ++i) {
        applyDepthEdgeBlend(depth, width, height, 1, 0);
    }
    for (int i = sharedIterations; i < config.edgeDilationY; ++i) {
        applyDepthEdgeBlend(depth, width, height, 0, 1);
    }
}

StereoRgbImage makeRgbImage(int width, int height) {
    StereoRgbImage image;
    image.width = width;
    image.height = height;
    image.pixels.resize(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3u);
    return image;
}

std::array<double, 3> sampleRgbBilinear(const std::uint8_t* pixels,
                                        int width,
                                        int height,
                                        double x,
                                        double y) {
    x = std::clamp(x, 0.0, static_cast<double>(std::max(width - 1, 0)));
    y = std::clamp(y, 0.0, static_cast<double>(std::max(height - 1, 0)));
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = std::min(x0 + 1, width - 1);
    const int y1 = std::min(y0 + 1, height - 1);
    const double fx = x - static_cast<double>(x0);
    const double fy = y - static_cast<double>(y0);

    auto load = [&](int sx, int sy, int channel) {
        return static_cast<double>(
            pixels[(static_cast<std::size_t>(sy) * static_cast<std::size_t>(width)
                    + static_cast<std::size_t>(sx)) * 3u
                   + static_cast<std::size_t>(channel)]);
    };

    std::array<double, 3> out{};
    for (int channel = 0; channel < 3; ++channel) {
        const double c00 = load(x0, y0, channel);
        const double c10 = load(x1, y0, channel);
        const double c01 = load(x0, y1, channel);
        const double c11 = load(x1, y1, channel);
        const double top = c00 * (1.0 - fx) + c10 * fx;
        const double bottom = c01 * (1.0 - fx) + c11 * fx;
        out[static_cast<std::size_t>(channel)] = top * (1.0 - fy) + bottom * fy;
    }
    return out;
}

StereoRgbImage resizeRgbImageBilinear(const StereoRgbImage& source,
                                      int newWidth,
                                      int newHeight) {
    if (source.width == newWidth && source.height == newHeight) {
        return source;
    }
    StereoRgbImage resized = makeRgbImage(newWidth, newHeight);
    for (int y = 0; y < newHeight; ++y) {
        const double srcY =
            (static_cast<double>(y) + 0.5) * static_cast<double>(source.height) / static_cast<double>(newHeight) - 0.5;
        for (int x = 0; x < newWidth; ++x) {
            const double srcX =
                (static_cast<double>(x) + 0.5) * static_cast<double>(source.width) / static_cast<double>(newWidth) - 0.5;
            const auto sample = sampleRgbBilinear(source.pixels.data(), source.width, source.height, srcX, srcY);
            const std::size_t dstIdx =
                (static_cast<std::size_t>(y) * static_cast<std::size_t>(newWidth)
                 + static_cast<std::size_t>(x)) * 3u;
            resized.pixels[dstIdx + 0u] = static_cast<std::uint8_t>(std::clamp(sample[0], 0.0, 255.0) + 0.5);
            resized.pixels[dstIdx + 1u] = static_cast<std::uint8_t>(std::clamp(sample[1], 0.0, 255.0) + 0.5);
            resized.pixels[dstIdx + 2u] = static_cast<std::uint8_t>(std::clamp(sample[2], 0.0, 255.0) + 0.5);
        }
    }
    return resized;
}

void resizeRgbBufferBilinear(const std::uint8_t* sourcePixels,
                             int sourceWidth,
                             int sourceHeight,
                             std::uint8_t* destPixels,
                             int destWidth,
                             int destHeight) {
    if (sourcePixels == nullptr || destPixels == nullptr ||
        sourceWidth <= 0 || sourceHeight <= 0 ||
        destWidth <= 0 || destHeight <= 0) {
        return;
    }

    if (sourceWidth == destWidth && sourceHeight == destHeight) {
        const std::size_t bytes = static_cast<std::size_t>(sourceWidth)
                                * static_cast<std::size_t>(sourceHeight) * 3u;
        std::memcpy(destPixels, sourcePixels, bytes);
        return;
    }

    parallelForRows(destHeight, destWidth, [&](int beginRow, int endRow) {
        for (int y = beginRow; y < endRow; ++y) {
            const double srcY =
                (static_cast<double>(y) + 0.5) * static_cast<double>(sourceHeight)
                / static_cast<double>(destHeight) - 0.5;
            for (int x = 0; x < destWidth; ++x) {
                const double srcX =
                    (static_cast<double>(x) + 0.5) * static_cast<double>(sourceWidth)
                    / static_cast<double>(destWidth) - 0.5;
                const auto sample = sampleRgbBilinear(
                    sourcePixels, sourceWidth, sourceHeight, srcX, srcY);
                const std::size_t dstIdx =
                    (static_cast<std::size_t>(y) * static_cast<std::size_t>(destWidth)
                     + static_cast<std::size_t>(x)) * 3u;
                destPixels[dstIdx + 0u] =
                    static_cast<std::uint8_t>(std::clamp(sample[0], 0.0, 255.0) + 0.5);
                destPixels[dstIdx + 1u] =
                    static_cast<std::uint8_t>(std::clamp(sample[1], 0.0, 255.0) + 0.5);
                destPixels[dstIdx + 2u] =
                    static_cast<std::uint8_t>(std::clamp(sample[2], 0.0, 255.0) + 0.5);
            }
        }
    });
}

StereoRgbImage padRgbImage(const StereoRgbImage& source,
                           int left,
                           int top,
                           int right,
                           int bottom) {
    if (left == 0 && top == 0 && right == 0 && bottom == 0) {
        return source;
    }
    StereoRgbImage padded = makeRgbImage(
        source.width + left + right,
        source.height + top + bottom);
    std::fill(padded.pixels.begin(), padded.pixels.end(), 0u);
    for (int y = 0; y < source.height; ++y) {
        for (int x = 0; x < source.width; ++x) {
            const std::size_t srcIdx =
                (static_cast<std::size_t>(y) * static_cast<std::size_t>(source.width)
                 + static_cast<std::size_t>(x)) * 3u;
            const std::size_t dstIdx =
                (static_cast<std::size_t>(y + top) * static_cast<std::size_t>(padded.width)
                 + static_cast<std::size_t>(x + left)) * 3u;
            padded.pixels[dstIdx + 0u] = source.pixels[srcIdx + 0u];
            padded.pixels[dstIdx + 1u] = source.pixels[srcIdx + 1u];
            padded.pixels[dstIdx + 2u] = source.pixels[srcIdx + 2u];
        }
    }
    return padded;
}

void postprocessStereoPadding(StereoRgbImage& leftEye,
                              StereoRgbImage& rightEye,
                              const StereoStageConfig& config) {
    const int ipdPad = [&]() {
        int value = static_cast<int>(std::abs(config.ipdOffset) * 0.01
            * static_cast<double>(std::max(leftEye.width, leftEye.height)));
        value -= value % 2;
        return value;
    }();
    if (ipdPad > 0) {
        const int outerPad = config.ipdOffset > 0.0 ? ipdPad * 2 : ipdPad;
        const int innerPad = config.ipdOffset > 0.0 ? ipdPad : ipdPad * 2;
        leftEye = padRgbImage(leftEye, outerPad, 0, innerPad, 0);
        rightEye = padRgbImage(rightEye, innerPad, 0, outerPad, 0);
    }

    if (config.pad > 0.0 || config.padMode == StereoPadMode::Aspect16x9) {
        switch (config.padMode) {
            case StereoPadMode::Tblr: {
                const int padH = static_cast<int>(std::round(static_cast<double>(leftEye.height) * config.pad)) / 2;
                const int padW = static_cast<int>(std::round(static_cast<double>(leftEye.width) * config.pad)) / 2;
                leftEye = padRgbImage(leftEye, padW, padH, padW, padH);
                rightEye = padRgbImage(rightEye, padW, padH, padW, padH);
                break;
            }
            case StereoPadMode::Tb: {
                const int padH = static_cast<int>(std::round(static_cast<double>(leftEye.height) * config.pad)) / 2;
                leftEye = padRgbImage(leftEye, 0, padH, 0, padH);
                rightEye = padRgbImage(rightEye, 0, padH, 0, padH);
                break;
            }
            case StereoPadMode::Lr: {
                const int padW = static_cast<int>(std::round(static_cast<double>(leftEye.width) * config.pad)) / 2;
                leftEye = padRgbImage(leftEye, padW, 0, padW, 0);
                rightEye = padRgbImage(rightEye, padW, 0, padW, 0);
                break;
            }
            case StereoPadMode::Top: {
                const int padTop = static_cast<int>(std::round(static_cast<double>(leftEye.height) * config.pad));
                leftEye = padRgbImage(leftEye, 0, padTop, 0, 0);
                rightEye = padRgbImage(rightEye, 0, padTop, 0, 0);
                break;
            }
            case StereoPadMode::Aspect16x9: {
                constexpr double kTargetRatio = 16.0 / 9.0;
                const double ratio = static_cast<double>(leftEye.width) / static_cast<double>(leftEye.height);
                if (std::abs(ratio - kTargetRatio) > 1e-3) {
                    if (ratio > kTargetRatio) {
                        const int targetHeight = static_cast<int>(
                            std::round(static_cast<double>(leftEye.width) / kTargetRatio));
                        const int padH = std::max(0, (targetHeight - leftEye.height) / 2);
                        leftEye = padRgbImage(leftEye, 0, padH, 0, padH);
                        rightEye = padRgbImage(rightEye, 0, padH, 0, padH);
                    } else {
                        const int targetWidth = static_cast<int>(
                            std::round(static_cast<double>(leftEye.height) * kTargetRatio));
                        const int padW = std::max(0, (targetWidth - leftEye.width) / 2);
                        leftEye = padRgbImage(leftEye, padW, 0, padW, 0);
                        rightEye = padRgbImage(rightEye, padW, 0, padW, 0);
                    }
                }
                break;
            }
        }
    }

    switch (config.format) {
        case StereoOutputFormat::HalfSbs:
            leftEye = resizeRgbImageBilinear(leftEye, std::max(1, leftEye.width / 2), leftEye.height);
            rightEye = resizeRgbImageBilinear(rightEye, std::max(1, rightEye.width / 2), rightEye.height);
            break;
        case StereoOutputFormat::HalfTb:
            leftEye = resizeRgbImageBilinear(leftEye, leftEye.width, std::max(1, leftEye.height / 2));
            rightEye = resizeRgbImageBilinear(rightEye, rightEye.width, std::max(1, rightEye.height / 2));
            break;
        default:
            break;
    }
}

StereoRgbImage composeSideBySide(const StereoRgbImage& leftEye,
                                 const StereoRgbImage& rightEye,
                                 bool reverse) {
    StereoRgbImage out = makeRgbImage(leftEye.width + rightEye.width, leftEye.height);
    const StereoRgbImage& a = reverse ? rightEye : leftEye;
    const StereoRgbImage& b = reverse ? leftEye : rightEye;
    for (int y = 0; y < out.height; ++y) {
        for (int x = 0; x < a.width; ++x) {
            const std::size_t srcIdx =
                (static_cast<std::size_t>(y) * static_cast<std::size_t>(a.width)
                 + static_cast<std::size_t>(x)) * 3u;
            const std::size_t dstIdx =
                (static_cast<std::size_t>(y) * static_cast<std::size_t>(out.width)
                 + static_cast<std::size_t>(x)) * 3u;
            out.pixels[dstIdx + 0u] = a.pixels[srcIdx + 0u];
            out.pixels[dstIdx + 1u] = a.pixels[srcIdx + 1u];
            out.pixels[dstIdx + 2u] = a.pixels[srcIdx + 2u];
        }
        for (int x = 0; x < b.width; ++x) {
            const std::size_t srcIdx =
                (static_cast<std::size_t>(y) * static_cast<std::size_t>(b.width)
                 + static_cast<std::size_t>(x)) * 3u;
            const std::size_t dstIdx =
                (static_cast<std::size_t>(y) * static_cast<std::size_t>(out.width)
                 + static_cast<std::size_t>(x + a.width)) * 3u;
            out.pixels[dstIdx + 0u] = b.pixels[srcIdx + 0u];
            out.pixels[dstIdx + 1u] = b.pixels[srcIdx + 1u];
            out.pixels[dstIdx + 2u] = b.pixels[srcIdx + 2u];
        }
    }
    return out;
}

StereoRgbImage composeTopBottom(const StereoRgbImage& topEye,
                                const StereoRgbImage& bottomEye) {
    StereoRgbImage out = makeRgbImage(topEye.width, topEye.height + bottomEye.height);
    for (int y = 0; y < topEye.height; ++y) {
        for (int x = 0; x < topEye.width; ++x) {
            const std::size_t srcIdx =
                (static_cast<std::size_t>(y) * static_cast<std::size_t>(topEye.width)
                 + static_cast<std::size_t>(x)) * 3u;
            const std::size_t dstIdx =
                (static_cast<std::size_t>(y) * static_cast<std::size_t>(out.width)
                 + static_cast<std::size_t>(x)) * 3u;
            out.pixels[dstIdx + 0u] = topEye.pixels[srcIdx + 0u];
            out.pixels[dstIdx + 1u] = topEye.pixels[srcIdx + 1u];
            out.pixels[dstIdx + 2u] = topEye.pixels[srcIdx + 2u];
        }
    }
    for (int y = 0; y < bottomEye.height; ++y) {
        for (int x = 0; x < bottomEye.width; ++x) {
            const std::size_t srcIdx =
                (static_cast<std::size_t>(y) * static_cast<std::size_t>(bottomEye.width)
                 + static_cast<std::size_t>(x)) * 3u;
            const std::size_t dstIdx =
                (static_cast<std::size_t>(y + topEye.height) * static_cast<std::size_t>(out.width)
                 + static_cast<std::size_t>(x)) * 3u;
            out.pixels[dstIdx + 0u] = bottomEye.pixels[srcIdx + 0u];
            out.pixels[dstIdx + 1u] = bottomEye.pixels[srcIdx + 1u];
            out.pixels[dstIdx + 2u] = bottomEye.pixels[srcIdx + 2u];
        }
    }
    return out;
}

StereoRgbImage composeAnaglyph(const StereoRgbImage& leftEye,
                               const StereoRgbImage& rightEye) {
    StereoRgbImage out = makeRgbImage(leftEye.width, leftEye.height);
    for (int y = 0; y < out.height; ++y) {
        for (int x = 0; x < out.width; ++x) {
            const std::size_t idx =
                (static_cast<std::size_t>(y) * static_cast<std::size_t>(out.width)
                 + static_cast<std::size_t>(x)) * 3u;
            out.pixels[idx + 0u] = leftEye.pixels[idx + 0u];
            out.pixels[idx + 1u] = rightEye.pixels[idx + 1u];
            out.pixels[idx + 2u] = rightEye.pixels[idx + 2u];
        }
    }
    return out;
}

StereoRgbImage synthesizeStereoEye(const StereoRgbImage& baseImage,
                                   const std::vector<float>& depth,
                                   double divergence,
                                   double convergence,
                                   bool passthrough,
                                   double sampleDirection) {
    if (passthrough) {
        return baseImage;
    }

    StereoRgbImage eye = makeRgbImage(baseImage.width, baseImage.height);
    const int baseSize = std::max(baseImage.width, baseImage.height);
    const double pixelShiftScale =
        divergence * 0.01
        * (static_cast<double>(baseSize) / static_cast<double>(baseImage.width))
        * (static_cast<double>(std::max(baseImage.width - 1, 0)) * 0.5);
    for (int y = 0; y < baseImage.height; ++y) {
        for (int x = 0; x < baseImage.width; ++x) {
            const std::size_t depthIdx =
                static_cast<std::size_t>(y) * static_cast<std::size_t>(baseImage.width)
                + static_cast<std::size_t>(x);
            const double shift =
                (static_cast<double>(depth[depthIdx]) - convergence) * pixelShiftScale;
            const double sampleX = static_cast<double>(x) + sampleDirection * shift;
            const auto sample = sampleRgbBilinear(
                baseImage.pixels.data(), baseImage.width, baseImage.height,
                sampleX, static_cast<double>(y));
            const std::size_t dstIdx = depthIdx * 3u;
            eye.pixels[dstIdx + 0u] = static_cast<std::uint8_t>(std::clamp(sample[0], 0.0, 255.0) + 0.5);
            eye.pixels[dstIdx + 1u] = static_cast<std::uint8_t>(std::clamp(sample[1], 0.0, 255.0) + 0.5);
            eye.pixels[dstIdx + 2u] = static_cast<std::uint8_t>(std::clamp(sample[2], 0.0, 255.0) + 0.5);
        }
    }
    return eye;
}

bool renderStereoFrame(const std::uint8_t* sourceRgb,
                       int sourceWidth,
                       int sourceHeight,
                       const std::vector<float>& depth,
                       int depthWidth,
                       int depthHeight,
                       const StereoStageConfig& config,
                       std::vector<std::uint8_t>& output) {
    StereoRgbImage baseImage;
    baseImage.width = sourceWidth;
    baseImage.height = sourceHeight;
    baseImage.pixels.assign(
        sourceRgb,
        sourceRgb + static_cast<std::ptrdiff_t>(sourceWidth) * static_cast<std::ptrdiff_t>(sourceHeight) * 3);
    if (sourceWidth != depthWidth || sourceHeight != depthHeight) {
        baseImage = resizeRgbImageBilinear(baseImage, depthWidth, depthHeight);
    }

    double effectiveDivergence = config.divergence;
    if (config.syntheticView != SyntheticViewMode::Both) {
        effectiveDivergence *= 2.0;
    }

    const bool leftPassthrough = config.syntheticView == SyntheticViewMode::Right;
    const bool rightPassthrough = config.syntheticView == SyntheticViewMode::Left;
    StereoRgbImage leftEye = synthesizeStereoEye(
        baseImage, depth, effectiveDivergence, config.convergence,
        leftPassthrough, -1.0);
    StereoRgbImage rightEye = synthesizeStereoEye(
        baseImage, depth, effectiveDivergence, config.convergence,
        rightPassthrough, 1.0);

    postprocessStereoPadding(leftEye, rightEye, config);

    StereoRgbImage composed;
    switch (config.format) {
        case StereoOutputFormat::FullTb:
        case StereoOutputFormat::HalfTb:
            composed = composeTopBottom(leftEye, rightEye);
            break;
        case StereoOutputFormat::CrossEyed:
            composed = composeSideBySide(leftEye, rightEye, true);
            break;
        case StereoOutputFormat::Anaglyph:
            composed = composeAnaglyph(leftEye, rightEye);
            break;
        case StereoOutputFormat::FullSbs:
        case StereoOutputFormat::HalfSbs:
        default:
            composed = composeSideBySide(leftEye, rightEye, false);
            break;
    }

    output = std::move(composed.pixels);
    return true;
}

// ─────────────────────────────────────────────────────────────────
// TensorDtype mapper (MiGraphX shape type → TensorDtype)
// ─────────────────────────────────────────────────────────────────
#ifdef AVE_HAVE_MIGRAPHX
TensorDtype mapMiGraphXType(migraphx_shape_datatype_t t) {
    switch (t) {
        case migraphx_shape_float_type:       return TensorDtype::Fp32;
        case migraphx_shape_half_type:        return TensorDtype::Fp16;
        case migraphx_shape_bf16_type:        return TensorDtype::Bf16;
        case migraphx_shape_int8_type:        return TensorDtype::Int8;
        case migraphx_shape_fp8e4m3fnuz_type: return TensorDtype::Fp8E4M3FNUZ;
        default:                              return TensorDtype::Unknown;
    }
}

std::size_t tensorDtypeSizeBytes(const TensorDtype dtype) {
    switch (dtype) {
        case TensorDtype::Fp32: return sizeof(float);
        case TensorDtype::Fp16:
        case TensorDtype::Bf16: return sizeof(std::uint16_t);
        case TensorDtype::Int8:
        case TensorDtype::Fp8E4M3FNUZ: return sizeof(std::uint8_t);
        case TensorDtype::Unknown: return 0u;
    }
    return 0u;
}

#ifdef AVE_HAVE_HIP
std::string hipErrorMessage(const hipError_t status) {
    const char* text = hipGetErrorString(status);
    if (text == nullptr || *text == '\0') {
        return "HIP error " + std::to_string(static_cast<int>(status));
    }
    return text;
}

bool isHipDeviceAccessiblePointer(const void* pointer) {
    if (pointer == nullptr) {
        return false;
    }
    hipPointerAttribute_t attributes{};
    const hipError_t status = hipPointerGetAttributes(&attributes, pointer);
    if (status != hipSuccess) {
        (void)hipGetLastError();
        return false;
    }
    return attributes.type == hipMemoryTypeDevice || attributes.isManaged != 0;
}

bool ensureHipBufferCapacity(void*& buffer,
                             std::size_t& capacityBytes,
                             const std::size_t requiredBytes,
                             const std::string& label,
                             std::string& error) {
    if (requiredBytes == 0u) {
        return true;
    }
    if (buffer != nullptr && capacityBytes >= requiredBytes) {
        return true;
    }
    if (buffer != nullptr) {
        const hipError_t freeStatus = hipFree(buffer);
        if (freeStatus != hipSuccess) {
            error = "Failed to release HIP buffer for " + label + ": "
                  + hipErrorMessage(freeStatus);
            return false;
        }
        buffer = nullptr;
        capacityBytes = 0u;
    }
    const hipError_t allocStatus = hipMalloc(&buffer, requiredBytes);
    if (allocStatus != hipSuccess) {
        error = "Failed to allocate HIP buffer for " + label + " (" +
              std::to_string(requiredBytes) + " bytes): " + hipErrorMessage(allocStatus);
        return false;
    }
    capacityBytes = requiredBytes;
    return true;
}

void releaseHipBuffer(void*& buffer, std::size_t& capacityBytes) {
    if (buffer != nullptr) {
        (void)hipFree(buffer);
        buffer = nullptr;
    }
    capacityBytes = 0u;
}

bool hipCopyAuto(void* dst,
                 const void* src,
                 const std::size_t bytes,
                 const std::string& label,
                 std::string& error) {
    if (bytes == 0u) {
        return true;
    }
    const hipError_t status = hipMemcpy(dst, src, bytes, hipMemcpyDefault);
    if (status != hipSuccess) {
        error = "HIP copy failed for " + label + ": " + hipErrorMessage(status);
        return false;
    }
    return true;
}

bool hipMappedHostStagingEnabled() {
    const char* raw = std::getenv("AVE_MIGRAPHX_MAPPED_STAGING");
    if (raw == nullptr || *raw == '\0') {
        return true;
    }
    std::string value(raw);
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    if (value == "1" || value == "true" || value == "on" || value == "yes") {
        return true;
    }
    if (value == "0" || value == "false" || value == "off" || value == "no") {
        return false;
    }
    std::cerr << "[migraphx] WARNING: unsupported AVE_MIGRAPHX_MAPPED_STAGING='"
              << raw << "'; using mapped host staging by default." << std::endl;
    return true;
}

struct HipMappedHostBuffer {
    void* hostPtr = nullptr;
    void* devicePtr = nullptr;
    std::size_t capacityBytes = 0u;

    HipMappedHostBuffer() = default;
    HipMappedHostBuffer(const HipMappedHostBuffer&) = delete;
    HipMappedHostBuffer& operator=(const HipMappedHostBuffer&) = delete;

    HipMappedHostBuffer(HipMappedHostBuffer&& other) noexcept
        : hostPtr(other.hostPtr),
          devicePtr(other.devicePtr),
          capacityBytes(other.capacityBytes) {
        other.hostPtr = nullptr;
        other.devicePtr = nullptr;
        other.capacityBytes = 0u;
    }

    HipMappedHostBuffer& operator=(HipMappedHostBuffer&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        release();
        hostPtr = other.hostPtr;
        devicePtr = other.devicePtr;
        capacityBytes = other.capacityBytes;
        other.hostPtr = nullptr;
        other.devicePtr = nullptr;
        other.capacityBytes = 0u;
        return *this;
    }

    ~HipMappedHostBuffer() { release(); }

    void release() {
        if (hostPtr != nullptr) {
            (void)hipHostFree(hostPtr);
        }
        hostPtr = nullptr;
        devicePtr = nullptr;
        capacityBytes = 0u;
    }
};

bool ensureHipMappedHostBufferCapacity(HipMappedHostBuffer& buffer,
                                       const std::size_t requiredBytes,
                                       const std::string& label,
                                       std::string& error) {
    if (requiredBytes == 0u) {
        return true;
    }
    if (buffer.hostPtr != nullptr && buffer.devicePtr != nullptr &&
        buffer.capacityBytes >= requiredBytes) {
        return true;
    }

    buffer.release();

    void* hostPtr = nullptr;
    const hipError_t allocStatus =
        hipHostMalloc(&hostPtr, requiredBytes, hipHostMallocMapped);
    if (allocStatus != hipSuccess) {
        error = "Failed to allocate HIP mapped host buffer for " + label + " ("
              + std::to_string(requiredBytes) + " bytes): "
              + hipErrorMessage(allocStatus);
        return false;
    }

    void* devicePtr = nullptr;
    const hipError_t mapStatus = hipHostGetDevicePointer(&devicePtr, hostPtr, 0);
    if (mapStatus != hipSuccess || devicePtr == nullptr) {
        error = "Failed to map HIP host staging buffer for " + label + ": "
              + hipErrorMessage(mapStatus);
        (void)hipHostFree(hostPtr);
        return false;
    }

    buffer.hostPtr = hostPtr;
    buffer.devicePtr = devicePtr;
    buffer.capacityBytes = requiredBytes;
    return true;
}
#endif

[[maybe_unused]] void setOnnxInputShapesForFrame(
        migraphx::onnx_options& options,
        const migraphx::program_parameter_shapes& shapes,
        int inputWidth,
        int inputHeight) {
    const std::size_t w = static_cast<std::size_t>(inputWidth);
    const std::size_t h = static_cast<std::size_t>(inputHeight);
    for (const char* rawName : shapes.names()) {
        if (rawName == nullptr) { continue; }
        const std::string name(rawName);
        if (name.empty() || isInternalOutputParameterName(name)) { continue; }

        auto dims = shapes[name.c_str()].lengths();
        if (dims.empty()) { continue; }

        if (dims.size() == 4) {
            dims[0] = 1; // enforce single-frame inference
            if (dims[1] <= 4) {
                // NCHW
                dims[2] = h;
                dims[3] = w;
            } else if (dims[3] <= 4) {
                // NHWC
                dims[1] = h;
                dims[2] = w;
            } else {
                // Default to NCHW when channel axis is ambiguous.
                dims[2] = h;
                dims[3] = w;
            }
            options.set_input_parameter_shape(name, dims);
            continue;
        }

        if (dims.size() == 3) {
            if (dims[0] <= 4) {
                // CHW
                dims[1] = h;
                dims[2] = w;
            } else if (dims[2] <= 4) {
                // HWC
                dims[0] = h;
                dims[1] = w;
            }
            options.set_input_parameter_shape(name, dims);
        }
    }
}
#endif

}  // namespace

// ─────────────────────────────────────────────────────────────────
// CompileOptions implementation
// ─────────────────────────────────────────────────────────────────

bool CompileOptions::validate(std::string& error) const {
    if (!offloadCopy) {
#ifdef AVE_HAVE_HIP
        error.clear();
#else
        error = "CompileOptions: offloadCopy=false requires HIP-enabled builds.";
        return false;
#endif
    }
    if (precision != MiGraphXPrecision::Fp32 &&
        precision != MiGraphXPrecision::Fp16 &&
        precision != MiGraphXPrecision::Int8) {
        error = "CompileOptions: only fp32, fp16, and int8 are supported.";
        return false;
    }
    error.clear();
    return true;
}

std::string CompileOptions::format() const {
    std::ostringstream os;
    os << "offload_copy=" << (offloadCopy ? "1" : "0")
       << " precision=" << compilePrecisionTag(precision);
    return os.str();
}

// ─────────────────────────────────────────────────────────────────
// Impl (compiled path: AVE_HAVE_MIGRAPHX)
// ─────────────────────────────────────────────────────────────────

#ifdef AVE_HAVE_MIGRAPHX

struct ModelProgram {
    migraphx::program          prog;
    std::optional<migraphx::arguments> lastResults;
    std::vector<migraphx::shape> inputShapes;
    std::vector<TensorContract> inputContracts;
    std::vector<TensorContract> outputContracts;
    std::string sourcePath;
    bool sourceIsMxr = false;
    int compiledInputWidth = 0;
    int compiledInputHeight = 0;
    std::mutex evalMutex;
    bool warmupComplete = false;
    obs::ArtifactManifestFields runtimeFields;
    std::vector<std::pair<std::string, std::string>> runtimeEnvOverrides;
#ifdef AVE_HAVE_HIP
    struct DeviceBuffer {
        void* ptr = nullptr;
        std::size_t capacityBytes = 0u;
    };
    DeviceBuffer inputDeviceBuffer;
    std::vector<DeviceBuffer> auxInputDeviceBuffers;
    HipMappedHostBuffer outputMappedHostBuffer;
    std::vector<std::uint8_t> outputHostBuffer;
    bool mappedOutputFallbackLogged = false;
#endif

    ~ModelProgram() {
#ifdef AVE_HAVE_HIP
        releaseHipBuffer(inputDeviceBuffer.ptr, inputDeviceBuffer.capacityBytes);
        for (auto& buffer : auxInputDeviceBuffers) {
            releaseHipBuffer(buffer.ptr, buffer.capacityBytes);
        }
#endif
    }
};

struct MiGraphXBackend::Impl {
    bool           initialised = false;
    int            deviceIdx   = 0;
    CompileOptions opts;
    std::mutex     mtx;
    std::unordered_map<std::string, std::shared_ptr<ModelProgram>> programs;
    std::unordered_map<std::string, std::string> loadFailures;
    ModelManager    modelManager;
    InteropBridge   interopBridge;

    // ── buildContracts ──────────────────────────────────────────
    // Construct TensorContracts from MiGraphX parameter/output shapes.
    static std::vector<TensorContract> buildContracts(
            const migraphx::program_parameter_shapes& shapes,
            const std::string& role) {
        std::vector<TensorContract> result;
        for (const char* rawName : shapes.names()) {
            const std::string name = rawName != nullptr ? std::string(rawName) : std::string();
            if (name.empty()) { continue; }
            // Skip internal output parameters
            if (isInternalOutputParameterName(name)) { continue; }
            const auto shape = shapes[name.c_str()];
            TensorContract c;
            c.name        = name;
            c.description = role + " parameter";
            c.dtype       = mapMiGraphXType(shape.type());
            // Build shape dims from MiGraphX lengths vector
            c.shape.dims.clear();
            for (const auto len : shape.lengths()) {
                c.shape.dims.push_back(static_cast<std::int64_t>(len));
            }
            c.layout = inferTensorLayout(c.shape.dims);
            if (c.layout == TensorLayout::Unknown) {
                c.layout = TensorLayout::NCHW;
            }
            result.push_back(std::move(c));
        }
        return result;
    }

    // ── loadProgram ──────────────────────────────────────────────
    // G1: ONNX opset gate  G3: manifest validation  G7: tensor contracts
    bool loadProgram(const std::string& modelId,
                     std::string& error,
                     std::optional<int> inputWidth = std::nullopt,
                     std::optional<int> inputHeight = std::nullopt,
                     std::optional<std::string> preferredPath = std::nullopt,
                     bool preferredPathExplicit = false,
                     std::optional<std::string> calibrationVideoPath = std::nullopt,
                     int compileBatch = 1) {
        std::string sourcePath;
        bool sourceIsMxr = true;
        const bool needFrameSpecificArtifact = inputWidth.has_value() && inputHeight.has_value();
        if (preferredPath.has_value() && !preferredPath->empty()) {
            const std::string ext = normalizeExtLower(*preferredPath);
            if (ext == ".mxr") {
                // For real video inference, ignore auto-selected generic .mxr
                // paths until we've resolved the frame-size-specific artifact.
                if (preferredPathExplicit || !needFrameSpecificArtifact) {
                    sourcePath = *preferredPath;
                }
            } else if (preferredPathExplicit) {
                const auto ie = InferenceError::modelIncompatible(
                    "MiGraphX backend requires a compiled .mxr artifact for inference.",
                    "model='" + modelId + "' explicit model_path=" + *preferredPath
                    + "\nAction: compile this model to .mxr in Model Manager.");
                error = ie.format();
                return false;
            }
        }

        auto tryResolveOrCompile = [&](std::optional<std::int64_t> iw,
                                       std::optional<std::int64_t> ih) -> bool {
            std::string compileError;
            const auto compiled = modelManager.autoCompileForInference(
                modelId, compileError, iw, ih, modelCompilePrecision(opts.precision),
                compileBatch, calibrationVideoPath);
            if (compiled.has_value() && normalizeExtLower(*compiled) == ".mxr") {
                sourcePath = *compiled;
                if (iw.has_value() && ih.has_value()) {
                    std::cout << "[migraphx] using frame-size specific .mxr for '" << modelId
                              << "': " << sourcePath << " (" << *iw << "x" << *ih << ")"
                              << std::endl;
                } else {
                    std::cout << "[migraphx] using compiled .mxr for '" << modelId
                              << "': " << sourcePath << std::endl;
                }
                return true;
            }
            if (!compileError.empty()) {
                const auto ie = InferenceError::compileFailure(
                    "Unable to compile model '" + modelId + "' for MiGraphX.",
                    compileError);
                error = ie.format();
                return false;
            }
            return true;
        };

        // For real video inference we always prefer a frame-size specific artifact.
        // This avoids loading a stale generic compile (for example 3840x2160) on
        // smaller sources and then silently mismatching tensor contracts.
        if (sourcePath.empty() && needFrameSpecificArtifact) {
            if (!tryResolveOrCompile(static_cast<std::int64_t>(*inputWidth),
                                     static_cast<std::int64_t>(*inputHeight))) {
                return false;
            }
        }

        if (sourcePath.empty() && !needFrameSpecificArtifact) {
            const auto bestPath = modelManager.bestPathForModel(modelId);
            if (bestPath.has_value() && normalizeExtLower(*bestPath) == ".mxr") {
                sourcePath = *bestPath;
            }
        }

        if (sourcePath.empty() && (!inputWidth.has_value() || !inputHeight.has_value())) {
            if (!tryResolveOrCompile(std::nullopt, std::nullopt)) {
                return false;
            }
        }

        if (sourcePath.empty()) {
            const auto ie = InferenceError::modelIncompatible(
                "No compiled .mxr artifact available for model '" + modelId + "'.",
                "Download/compile this model in Model Manager before running inference.");
            error = ie.format();
            return false;
        }

        const std::string key =
            loadFailureKey(modelId, inputWidth, inputHeight) + "|" + sourcePath;
        if (auto pit = programs.find(modelId); pit != programs.end()) {
            if (pit->second != nullptr &&
                pit->second->sourcePath == sourcePath &&
                pit->second->sourceIsMxr == sourceIsMxr) {
                return true;
            }
            programs.erase(pit);
        }

        if (const auto failIt = loadFailures.find(key); failIt != loadFailures.end()) {
            error = failIt->second;
            return false;
        }

        auto rememberFailure = [&](const std::string& msg) {
            error = msg;
            loadFailures[key] = error;
            return false;
        };

        AVE_ROCTX_RANGE("migraphx:load");
        try {
            auto mp = std::make_shared<ModelProgram>();
            mp->sourcePath = sourcePath;
            mp->sourceIsMxr = sourceIsMxr;
            const RuntimeEnvConfig runtimeEnv = buildRuntimeEnvConfig(sourcePath, opts);
            mp->runtimeFields = runtimeEnv.fields;
            mp->runtimeEnvOverrides = runtimeEnv.overrides;
            obs::logMiGraphXEnvironment(mp->runtimeFields, "runtime-load", sourcePath, "pending");

            ScopedRuntimeEnvOverrides scopedRuntimeEnv(mp->runtimeEnvOverrides);
            if (!scopedRuntimeEnv.ok()) {
                const auto ie = InferenceError::runtimeFailure(
                    "Unable to apply MiGraphX runtime environment overrides before model load.",
                    scopedRuntimeEnv.error());
                AVE_ROCTX_RANGE_END();
                return rememberFailure(ie.format());
            }

            mp->prog = migraphx::load(sourcePath.c_str());

            const auto outShapes = mp->prog.get_output_shapes();
            if (outShapes.empty()) {
                const auto ie = InferenceError::runtimeFailure(
                    "program::get_output_shapes() returned empty for model '"
                    + modelId + "'.",
                    "Source: " + sourcePath);
                std::cerr << ie.format() << std::endl;
                AVE_ROCTX_RANGE_END();
                return rememberFailure(ie.format());
            }

            const auto parameterShapes = mp->prog.get_parameter_shapes();
            mp->inputContracts  = buildContracts(parameterShapes, "input");
            if (mp->inputContracts.empty()) {
                const auto ie = InferenceError::modelIncompatible(
                    "Model '" + modelId + "' has no usable input tensors.",
                    "Internal '#output' placeholders were filtered from program parameters.");
                std::cerr << ie.format() << std::endl;
                AVE_ROCTX_RANGE_END();
                return rememberFailure(ie.format());
            }
            mp->inputShapes.reserve(mp->inputContracts.size());
            for (const auto& contract : mp->inputContracts) {
                mp->inputShapes.push_back(parameterShapes[contract.name.c_str()]);
            }

            mp->outputContracts.clear();
            for (std::size_t i = 0; i < outShapes.size(); ++i) {
                TensorContract oc;
                oc.name        = "output_" + std::to_string(i);
                oc.description = "output parameter";
                oc.dtype       = mapMiGraphXType(outShapes[i].type());
                for (const auto len : outShapes[i].lengths()) {
                    oc.shape.dims.push_back(static_cast<std::int64_t>(len));
                }
                oc.layout = inferTensorLayout(oc.shape.dims);
                if (oc.layout == TensorLayout::Unknown) {
                    oc.layout = TensorLayout::NCHW;
                }
                mp->outputContracts.push_back(std::move(oc));
            }

            std::cout << "[migraphx] loaded model='" << modelId
                      << "' source='" << sourcePath
                      << "' format=" << (sourceIsMxr ? "mxr" : "onnx") << "\n";
            for (const auto& c : mp->inputContracts) {
                std::cout << "  in:  " << c.format() << '\n';
            }
            for (const auto& c : mp->outputContracts) {
                std::cout << "  out: " << c.format() << '\n';
            }
            std::cout << "  compile_opts: " << opts.format() << '\n';
            std::cout << std::flush;

            programs.emplace(modelId, std::move(mp));
            loadFailures.erase(key);
        } catch (const std::exception& ex) {
            const auto ie = InferenceError::compileFailure(
                std::string("MiGraphX load/compile failed: ") + ex.what(),
                "Source: " + sourcePath);
            std::cerr << ie.format() << std::endl;
            AVE_ROCTX_RANGE_END();
            return rememberFailure(ie.format());
        }
        AVE_ROCTX_RANGE_END();
        return true;
    }

    // ── runInference ─────────────────────────────────────────────
    // G4: output shape assertion  G5: program::finish()
    // G7: element-count gate (TensorContract)
    // G8: InteropBridge hook documented
    bool runInference(const std::string& modelId,
                      const void*        inputData,
                      std::size_t        inputElements,
                      TensorDtype        inputDtype,
                      const void*&       outputData,
                      std::size_t&       outputElements,
                      TensorDtype&       outputDtype,
                      std::string&       error,
                      bool               finishAfterEval = true,
                      const EnhancementStage* stageContext = nullptr,
                      bool applyRuntimeEnvOverrides = true,
                      bool inputIsDeviceAccessible = false) {
        std::shared_ptr<ModelProgram> mp;
        {
            std::lock_guard<std::mutex> lk(mtx);
            const auto it = programs.find(modelId);
            if (it == programs.end() || it->second == nullptr) {
                error = InferenceError::runtimeFailure(
                    "Model not loaded: " + modelId).format();
                return false;
            }
            mp = it->second;
        }

        std::lock_guard<std::mutex> evalLock(mp->evalMutex);
        std::optional<ScopedRuntimeEnvOverrides> scopedRuntimeEnv;
        if (applyRuntimeEnvOverrides) {
            scopedRuntimeEnv.emplace(mp->runtimeEnvOverrides);
            if (!scopedRuntimeEnv->ok()) {
                error = InferenceError::runtimeFailure(
                    "Unable to apply MiGraphX runtime environment overrides before eval.",
                    scopedRuntimeEnv->error()).format();
                return false;
            }
        }

        if (mp->inputContracts.empty()) {
            error = InferenceError::runtimeFailure(
                "Model '" + modelId + "' has no input parameters.").format();
            return false;
        }

        std::size_t contractIdx = 0;
        bool matchedByElements = false;
        for (std::size_t i = 0; i < mp->inputContracts.size(); ++i) {
            const std::int64_t expected = mp->inputContracts[i].shape.elements();
            if (expected > 0 && static_cast<std::size_t>(expected) == inputElements) {
                contractIdx = i;
                matchedByElements = true;
                break;
            }
        }
        if (!matchedByElements) {
            for (std::size_t i = 0; i < mp->inputContracts.size(); ++i) {
                if (mp->inputContracts[i].name == "input") {
                    contractIdx = i;
                    break;
                }
            }
        }

        const auto& contract = mp->inputContracts[contractIdx];
        const auto& inName   = contract.name;

        if (contract.dtype != inputDtype) {
            error = InferenceError::runtimeFailure(
                "Input dtype mismatch for '" + modelId + "': expected "
                + toString(contract.dtype) + ", got " + toString(inputDtype) + '.').format();
            return false;
        }

        // ── G7: Element-count assertion ──────────────────────────
        {
            std::string contractError;
            if (!assertElementCount(contract, inputElements, contractError)) {
                obs::logTensorContractViolation(
                    "MiGraphXBackend::runInference (input gate)",
                    contract.format(),
                    "actual elements=" + std::to_string(inputElements));
                error = InferenceError::runtimeFailure(contractError).format();
                return false;
            }
        }

        // ── Build parameter map ─────────────────────────────────
        // Default path: offload_copy=1, so MiGraphX manages H2D/D2H copies
        // internally and eval() accepts host pointers directly.
        //
        // Experimental path: offload_copy=0, so we bind HIP device buffers
        // explicitly. Host callers still work because we upload/download
        // through reusable HIP staging buffers owned by the loaded program.
        try {
            const auto& inShape = mp->inputShapes[contractIdx];
            const std::size_t inputBytes =
                inputElements * tensorDtypeSizeBytes(inputDtype);
            if (inputBytes == 0u) {
                error = InferenceError::runtimeFailure(
                    "Unsupported input dtype size for '" + modelId + "': "
                    + toString(inputDtype) + '.').format();
                return false;
            }

            void* inputArgData = const_cast<void*>(inputData);
#ifdef AVE_HAVE_HIP
            if (!opts.offloadCopy &&
                !inputIsDeviceAccessible &&
                !isHipDeviceAccessiblePointer(inputData)) {
                std::string deviceError;
                if (!ensureHipBufferCapacity(mp->inputDeviceBuffer.ptr,
                                             mp->inputDeviceBuffer.capacityBytes,
                                             inputBytes,
                                             "primary input tensor",
                                             deviceError) ||
                    !hipCopyAuto(mp->inputDeviceBuffer.ptr,
                                 inputData,
                                 inputBytes,
                                 "primary input tensor upload",
                                 deviceError)) {
                    error = InferenceError::runtimeFailure(deviceError).format();
                    return false;
                }
                inputArgData = mp->inputDeviceBuffer.ptr;
            }
#endif

            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
            migraphx::argument inArg(inShape, inputArgData);

            migraphx::program_parameters pp;
            pp.add(inName.c_str(), inArg);

            std::vector<AuxInputBuffer> auxInputBuffers;
            auxInputBuffers.reserve(mp->inputContracts.size());
#ifdef AVE_HAVE_HIP
            if (mp->auxInputDeviceBuffers.size() < mp->inputContracts.size()) {
                mp->auxInputDeviceBuffers.resize(mp->inputContracts.size());
            }
#endif
            for (std::size_t i = 0; i < mp->inputContracts.size(); ++i) {
                if (i == contractIdx) {
                    continue;
                }
                const auto& auxContract = mp->inputContracts[i];
                EnhancementStage emptyStage;
                const EnhancementStage& auxStage =
                    stageContext != nullptr ? *stageContext : emptyStage;

                AuxInputBuffer buffer;
                std::string auxError;
                if (!buildAuxInputBuffer(auxStage, auxContract, buffer, auxError)) {
                    error = InferenceError::runtimeFailure(auxError).format();
                    return false;
                }

                const auto& auxShape = mp->inputShapes[i];
                void* auxArgData = buffer.data();
#ifdef AVE_HAVE_HIP
                if (!opts.offloadCopy) {
                    const std::size_t auxBytes =
                        static_cast<std::size_t>(auxContract.shape.elements())
                        * tensorDtypeSizeBytes(auxContract.dtype);
                    if (auxBytes == 0u) {
                        error = InferenceError::runtimeFailure(
                            "Unsupported auxiliary input dtype size for '" + auxContract.name
                            + "': " + toString(auxContract.dtype) + '.').format();
                        return false;
                    }
                    auto& deviceBuffer = mp->auxInputDeviceBuffers[i];
                    if (!ensureHipBufferCapacity(deviceBuffer.ptr,
                                                 deviceBuffer.capacityBytes,
                                                 auxBytes,
                                                 "auxiliary input '" + auxContract.name + "'",
                                                 auxError) ||
                        !hipCopyAuto(deviceBuffer.ptr,
                                     buffer.data(),
                                     auxBytes,
                                     "auxiliary input '" + auxContract.name + "' upload",
                                     auxError)) {
                        error = InferenceError::runtimeFailure(auxError).format();
                        return false;
                    }
                    auxArgData = deviceBuffer.ptr;
                }
#endif
                migraphx::argument auxArg(auxShape, auxArgData);
                pp.add(auxContract.name.c_str(), auxArg);
                auxInputBuffers.push_back(std::move(buffer));
            }

            // ── Eval ─────────────────────────────────────────────
            AVE_ROCTX_RANGE("migraphx:eval");
            const auto results = mp->prog.eval(pp);
            AVE_ROCTX_RANGE_END();
            if (finishAfterEval) {
                mp->prog.experimental_get_context().finish();
            }

            // ── G4: Assert output shapes per frame ───────────────
            if (results.empty()) {
                error = InferenceError::runtimeFailure(
                    "program::eval returned no outputs for '" + modelId + "'.").format();
                return false;
            }
            const auto& outShape = results[0].get_shape();
            if (!mp->outputContracts.empty()) {
                const auto expectedElems = static_cast<std::size_t>(
                    mp->outputContracts[0].shape.elements());
                if (outShape.elements() != expectedElems) {
                    obs::logTensorContractViolation(
                        "MiGraphXBackend::runInference (output gate)",
                        mp->outputContracts[0].format(),
                        "actual elements=" + std::to_string(outShape.elements()));
                    error = InferenceError::runtimeFailure(
                        "Output shape mismatch for '" + modelId + "': expected "
                        + std::to_string(expectedElems) + " elements, got "
                        + std::to_string(outShape.elements()) + ".").format();
                    return false;
                }
            }

            // ── Retrieve output ──────────────────────────────────
            // With offload_copy=1, MiGraphX returns CPU-accessible output.
            // With offload_copy=0, download once into a reusable host buffer
            // so existing postprocess/encode callers remain unchanged.
            outputElements = outShape.elements();
            outputDtype = mapMiGraphXType(outShape.type());
            mp->lastResults = std::move(results);
            const void* rawOutputData = (*mp->lastResults)[0].data();
            if (rawOutputData == nullptr) {
                error = InferenceError::runtimeFailure(
                    "MiGraphX returned a null output pointer for '" + modelId + "'.").format();
                return false;
            }
#ifdef AVE_HAVE_HIP
            if (!opts.offloadCopy) {
                const std::size_t outputBytes =
                    outputElements * tensorDtypeSizeBytes(outputDtype);
                if (outputBytes == 0u) {
                    error = InferenceError::runtimeFailure(
                        "Unsupported output dtype size for '" + modelId + "': "
                        + toString(outputDtype) + '.').format();
                    return false;
                }
                void* hostOutputPtr = nullptr;
                if (hipMappedHostStagingEnabled()) {
                    std::string mappedBufferError;
                    if (ensureHipMappedHostBufferCapacity(mp->outputMappedHostBuffer,
                                                          outputBytes,
                                                          "model output staging",
                                                          mappedBufferError)) {
                        hostOutputPtr = mp->outputMappedHostBuffer.hostPtr;
                    } else {
                        if (!mp->mappedOutputFallbackLogged) {
                            std::cerr << "[migraphx] WARNING: " << mappedBufferError
                                      << "\n  Falling back to pageable host output staging."
                                      << std::endl;
                            mp->mappedOutputFallbackLogged = true;
                        }
                        mp->outputMappedHostBuffer.release();
                    }
                }
                if (hostOutputPtr == nullptr) {
                    mp->outputHostBuffer.resize(outputBytes);
                    hostOutputPtr = mp->outputHostBuffer.data();
                }
                std::string copyError;
                if (!hipCopyAuto(hostOutputPtr,
                                 rawOutputData,
                                 outputBytes,
                                 "model output download",
                                 copyError)) {
                    error = InferenceError::runtimeFailure(copyError).format();
                    return false;
                }
                outputData = hostOutputPtr;
            } else {
                outputData = rawOutputData;
            }
#else
            outputData = rawOutputData;
#endif

        } catch (const std::exception& ex) {
            error = InferenceError::runtimeFailure(
                std::string("MiGraphX eval: ") + ex.what(),
                "model='" + modelId + "'").format();
            return false;
        }
        return true;
    }
};

#else  // !AVE_HAVE_MIGRAPHX

struct MiGraphXBackend::Impl {
    bool           initialised = false;
    int            deviceIdx   = 0;
    CompileOptions opts;
    std::mutex     mtx;
    std::unordered_map<std::string, bool>                    loaded;
    std::unordered_map<std::string, std::vector<TensorContract>> inputContracts_;
    std::unordered_map<std::string, std::vector<TensorContract>> outputContracts_;

    bool loadProgram(const std::string& modelId,
                     std::string& error,
                     std::optional<int> inputWidth = std::nullopt,
                     std::optional<int> inputHeight = std::nullopt,
                     std::optional<std::string> preferredPath = std::nullopt,
                     bool preferredPathExplicit = false,
                     std::optional<std::string> calibrationVideoPath = std::nullopt,
                     int compileBatch = 1) {
        (void)modelId;
        (void)inputWidth;
        (void)inputHeight;
        (void)preferredPath;
        (void)preferredPathExplicit;
        (void)calibrationVideoPath;
        (void)compileBatch;
        error = "MiGraphX hardware support was not compiled into this build (-DAVE_HAVE_MIGRAPHX=OFF).";
        return false;
    }
};

#endif  // AVE_HAVE_MIGRAPHX

// ─────────────────────────────────────────────────────────────────
// Public interface
// ─────────────────────────────────────────────────────────────────

MiGraphXBackend::MiGraphXBackend()  : impl_(std::make_unique<Impl>()) {
    std::string initEnvError;
    const int defaultDeviceIdx = autoSelectedVisibleDeviceIndex().value_or(0);
    if (!applyDefaultMiGraphXProcessEnv(defaultDeviceIdx, initEnvError)) {
        std::cerr << "[migraphx] WARNING: failed to prime default MiGraphX environment: "
                  << initEnvError << std::endl;
    }
    impl_->opts = compileOptionsFromEnv();
}
MiGraphXBackend::~MiGraphXBackend() = default;

BackendType MiGraphXBackend::type()  const { return BackendType::MiGraphX; }
std::string MiGraphXBackend::name()  const { return "MiGraphX (ROCm)"; }
bool MiGraphXBackend::supportsDirectOutputEncode() const { return true; }

bool MiGraphXBackend::isAvailable(std::string& reason) const {
#ifdef AVE_HAVE_MIGRAPHX
    if (!hasAmdSignal()) {
        reason = "ROCm tooling not detected (expected rocminfo/rocm-smi or /opt/rocm).";
        return false;
    }
    if (!hasAnyMiGraphXArtifact()) {
        reason = "MiGraphX runtime not found (system or bundled libmigraphx / migraphx-driver).";
        return false;
    }
    reason = "MiGraphX runtime detected.";
    return true;
#else
    reason = "MiGraphX hardware support was not compiled into this build (-DAVE_HAVE_MIGRAPHX=OFF).";
    return false;
#endif
}

bool MiGraphXBackend::initialize(std::string& error) {
    std::string reason;
    if (!isAvailable(reason)) {
        error = InferenceError::vulkanDevice(
            "MiGraphX init: " + reason).format();
        return false;
    }

    std::string optionsError;
    if (!impl_->opts.validate(optionsError)) {
        error = InferenceError::runtimeFailure(
            "MiGraphX init: invalid compile/runtime options.",
            optionsError).format();
        return false;
    }

#ifdef AVE_HAVE_HIP
    int devCount = 0;
    if (hipGetDeviceCount(&devCount) != hipSuccess || devCount == 0) {
        error = InferenceError::vulkanDevice(
            "HIP: no AMD GPU devices found.").format();
        return false;
    }
    std::string deviceSelectionDetail;
    impl_->deviceIdx = selectPreferredHipDeviceIndex(devCount, deviceSelectionDetail);
    if (hipSetDevice(impl_->deviceIdx) != hipSuccess) {
        error = InferenceError::vulkanDevice(
            "HIP: failed to select AMD device index " + std::to_string(impl_->deviceIdx) + ".").format();
        return false;
    }
    if (!deviceSelectionDetail.empty()) {
        std::cout << "[migraphx] " << deviceSelectionDetail << std::endl;
    }
    hipDeviceProp_t props{};
    if (hipGetDeviceProperties(&props, impl_->deviceIdx) == hipSuccess) {
        const auto totalMemoryMiB = static_cast<std::uint64_t>(props.totalGlobalMem) / (1024u * 1024u);
        std::cout << "[migraphx] HIP device " << impl_->deviceIdx
                  << ": " << props.name
                  << " arch=" << (props.gcnArchName[0] != '\0' ? props.gcnArchName : "unknown")
                  << " type=" << (props.integrated != 0 ? "integrated" : "discrete")
                  << " vram_mib=" << totalMemoryMiB
                  << std::endl;
    }
#endif

    impl_->initialised = true;

    if (!applyDefaultMiGraphXProcessEnv(impl_->deviceIdx, error)) {
        error = InferenceError::runtimeFailure(
            "MiGraphX init: unable to apply default runtime tuning environment.",
            error).format();
        return false;
    }

#ifdef AVE_HAVE_HIP
    if (devCount > 1 && std::getenv("HIP_VISIBLE_DEVICES") == nullptr &&
        std::getenv("ROCR_VISIBLE_DEVICES") == nullptr &&
        std::getenv("AVE_MIGRAPHX_VISIBLE_DEVICES") == nullptr &&
        !preferredAmdDeviceIndexFromSettings().has_value()) {
        std::cerr << "[migraphx] WARNING: HIP enumerated " << devCount
                  << " devices and HIP_VISIBLE_DEVICES is unset. "
                  << "AMD's ROCm install guidance warns that integrated graphics can destabilize ROCm; "
                  << "if MiGraphX compilation is flaky, pin the intended discrete GPU with HIP_VISIBLE_DEVICES."
                  << std::endl;
    }
#endif

    // ── G6: Log version tuple and MIGRAPHX_* env vars ───────────
    obs::logVersionTuple();
    obs::logMiGraphXEnvironment();
    impl_->interopBridge.logConfig();
    if (!impl_->opts.offloadCopy) {
        std::cout << "[migraphx] experimental device tensor path enabled "
                     "(offload_copy=0). Host callers still use explicit HIP staging "
                     "until Vulkan/HIP interop is wired into the frame loop."
                  << std::endl;
    }

    std::cout << "[backend] MiGraphX initialised on device " << impl_->deviceIdx
              << " — compile options: " << impl_->opts.format() << std::endl;
    return true;
}

void MiGraphXBackend::setCompileOptions(const CompileOptions& opts) {
    impl_->opts = opts;
}

CompileOptions MiGraphXBackend::compileOptions() const {
    return impl_->opts;
}

bool MiGraphXBackend::preloadModel(const std::string& modelId, std::string& error) {
    std::lock_guard<std::mutex> lk(impl_->mtx);
    return impl_->loadProgram(modelId, error);
}

void MiGraphXBackend::evictModel(const std::string& modelId) {
    std::lock_guard<std::mutex> lk(impl_->mtx);
#ifdef AVE_HAVE_MIGRAPHX
    impl_->programs.erase(modelId);
    const std::string prefix = modelId + "@";
    for (auto it = impl_->loadFailures.begin(); it != impl_->loadFailures.end();) {
        if (it->first.rfind(prefix, 0) == 0) {
            it = impl_->loadFailures.erase(it);
        } else {
            ++it;
        }
    }
#else
    impl_->loaded.erase(modelId);
    impl_->inputContracts_.erase(modelId);
    impl_->outputContracts_.erase(modelId);
#endif
}

void MiGraphXBackend::evictAll() {
    std::lock_guard<std::mutex> lk(impl_->mtx);
#ifdef AVE_HAVE_MIGRAPHX
    impl_->programs.clear();
    impl_->loadFailures.clear();
#else
    impl_->loaded.clear();
    impl_->inputContracts_.clear();
    impl_->outputContracts_.clear();
#endif
}

int MiGraphXBackend::deviceIndex() const { return impl_->deviceIdx; }

std::vector<TensorContract>
MiGraphXBackend::inputContracts(const std::string& modelId) const {
    std::lock_guard<std::mutex> lk(impl_->mtx);
#ifdef AVE_HAVE_MIGRAPHX
    const auto it = impl_->programs.find(modelId);
    if (it != impl_->programs.end() && it->second != nullptr) { return it->second->inputContracts; }
#else
    const auto it = impl_->inputContracts_.find(modelId);
    if (it != impl_->inputContracts_.end()) { return it->second; }
#endif
    return {};
}

std::vector<TensorContract>
MiGraphXBackend::outputContracts(const std::string& modelId) const {
    std::lock_guard<std::mutex> lk(impl_->mtx);
#ifdef AVE_HAVE_MIGRAPHX
    const auto it = impl_->programs.find(modelId);
    if (it != impl_->programs.end() && it->second != nullptr) { return it->second->outputContracts; }
#else
    const auto it = impl_->outputContracts_.find(modelId);
    if (it != impl_->outputContracts_.end()) { return it->second; }
#endif
    return {};
}

StageResult MiGraphXBackend::runStage(const EnhancementStage& stage, std::string& error) {
    const std::string modelId = resolveModelId(stage);
    const std::optional<std::string> selectedPath = stageModelPath(stage);
    const bool selectedPathExplicit = stageModelPathExplicit(stage);
    if (modelId.empty()) {
        std::cout << "[migraphx] no model configured for " << toString(stage.kind)
                  << " — deferring to FFmpeg filter chain." << std::endl;
        return StageResult::Deferred;
    }

    if (!selectedPathExplicit && selectedPath.has_value() &&
        normalizeExtLower(*selectedPath) == ".mxr") {
        std::cout << "[migraphx] stage '" << toString(stage.kind)
                  << "' deferring auto-selected .mxr preload until the actual frame size "
                     "is known; processVideoFile() will load the correct artifact."
                  << std::endl;
        return StageResult::Deferred;
    }

    {
        std::lock_guard<std::mutex> lk(impl_->mtx);
        if (!impl_->loadProgram(modelId, error, std::nullopt, std::nullopt,
                                selectedPath, selectedPathExplicit)) {
            if (error.find("[NeedsFrameSize]") != std::string::npos) {
                // Native ONNX path: we need actual frame dimensions from decode.
                error.clear();
                return StageResult::Deferred;
            }
            // Classify the error to decide fall-through vs hard-fail.
            // ModelIncompatible and ArtifactInvalid → warn + fallback.
            // SyncHazard → hard-fail (data integrity at risk).
            if (error.find("[SyncHazard]") != std::string::npos) {
                // SyncHazard is fail-fast per gold standard.
                return StageResult::Error;
            }
            std::cerr << "[migraphx] WARNING: model not ready for stage '"
                      << toString(stage.kind) << "':\n  " << error
                      << "\n  → Deferring to FFmpeg filter chain." << std::endl;
            error.clear();
            return StageResult::Deferred;
        }
    }

    // Model is loaded and verified. Actual per-frame AI inference runs in
    // processVideoFile() which is called by the FFmpeg encode pipeline.
    // Returning Deferred here tells the pipeline to call processVideoFile()
    // during the encode pass where real frame-by-frame AI processing happens.
    AVE_ROCTX_MARK("migraphx:stage-model-ready");
    std::cout << "[migraphx] model='" << modelId
              << "' loaded and verified for stage '" << toString(stage.kind)
              << "' | compile=" << impl_->opts.format()
              << "\n  AI inference will run via processVideoFile() during encode." << std::endl;
    return StageResult::Deferred;
}

// ─────────────────────────────────────────────────────────────────
// processFrameDir — per-frame AI inference on a directory of PNGs
//
// Gold-standard compliance:
//   G5  program::finish() called after every eval (via runInference).
//   G7  TensorContract element-count gate (via runInference).
//   G8  CPU staging logged as degraded mode.
//   G9  Structured InferenceError taxonomy.
//   G10 ROCTx markers around the processing loop.
// ─────────────────────────────────────────────────────────────────


// ─────────────────────────────────────────────────────────────────
// probeVideoDimensions — get width, height, fps via ffprobe
// ─────────────────────────────────────────────────────────────────
namespace {

}  // namespace

StageResult MiGraphXBackend::processVideoFile(
        const EnhancementStage& stage,
        const std::string& inputVideo,
        const std::string& outputVideo,
        const FrameProgressCb& progressCb,
        std::string& error,
        const ProcessVideoOptions& opts) {
#ifdef AVE_HAVE_MIGRAPHX
    auto deferToFfmpeg = [&](const std::string& detail) -> StageResult {
        std::cerr << "[migraphx] processVideoFile: " << detail
                  << "\n  → Deferring to FFmpeg." << std::endl;
        error.clear();
        return StageResult::Deferred;
    };

    const std::string modelId = resolveModelId(stage);
    const std::optional<std::string> selectedPath = stageModelPath(stage);
    const bool selectedPathExplicit = stageModelPathExplicit(stage);
    if (modelId.empty()) {
        std::cout << "[migraphx] processVideoFile: no model for "
                  << toString(stage.kind) << " — deferring." << std::endl;
        return StageResult::Deferred;
    }

    // ── Probe video dimensions via ffprobe (reliable, no Vulkan HW needed) ──
    const auto probe = probeVideoStream(inputVideo, error);
    if (!probe.has_value()) {
        return deferToFfmpeg("ffprobe failed: " + error);
    }

    const int inW = static_cast<int>(probe->width);
    const int inH = static_cast<int>(probe->height);
    const std::int64_t totalFrames = probe->estimatedFrameCount();
    const bool stereo3dStage = stage.kind == StageKind::Stereo3D;

    std::cout << "[migraphx] Input: " << inW << "x" << inH
              << " fps=" << probe->effectiveFrameRate(30.0)
              << " frames=" << totalFrames << std::endl;

    TileConfig tileConfig;
    if (!resolveTileConfig(stage, tileConfig, error)) {
        return deferToFfmpeg("Invalid tile configuration: " + error);
    }
    const auto stageHasTileParam = [&](const char* key) {
        return stage.params.find(key) != stage.params.end();
    };
    const bool tileGeometryExplicit =
        hasNonEmptyEnv("AVE_MIGRAPHX_TILE_SIZE") ||
        hasNonEmptyEnv("AVE_MIGRAPHX_TILE_WIDTH") ||
        hasNonEmptyEnv("AVE_MIGRAPHX_TILE_HEIGHT") ||
        stageHasTileParam("tile_size") ||
        stageHasTileParam("tile_width") ||
        stageHasTileParam("tile_height");
    if (!stereo3dStage && !selectedPathExplicit && !tileGeometryExplicit) {
        tileConfig.width = std::min(inW, 4096);
        tileConfig.height = std::min(inH, 4096);
        const int maxOverlap = std::max(0, (std::min(tileConfig.width, tileConfig.height) / 2) - 1);
        tileConfig.overlap = std::min(tileConfig.overlap, maxOverlap);
        std::cout << "[migraphx] full-frame compile preferred: requesting model input "
                  << tileConfig.width << "x" << tileConfig.height;
        if (tileConfig.width != inW || tileConfig.height != inH) {
            std::cout << " (clamped from source " << inW << "x" << inH << ")";
        }
        std::cout << std::endl;
    }
    if (stereo3dStage) {
        bool limitedToSource = false;
        int requestedStereoDepthResolution = kDefaultStereoDepthResolution;
        const int stereoDepthResolution = resolveStereoDepthResolution(
            stage, inW, inH, limitedToSource, requestedStereoDepthResolution);
        tileConfig.width = stereoDepthResolution;
        tileConfig.height = stereoDepthResolution;
        const int maxStereoOverlap = std::max(0, (stereoDepthResolution / 2) - 1);
        tileConfig.overlap = std::min(tileConfig.overlap, maxStereoOverlap);
        std::cout << "[migraphx] stereo depth resolution requested="
                  << requestedStereoDepthResolution
                  << " -> model input " << stereoDepthResolution << "x"
                  << stereoDepthResolution;
        if (limitedToSource) {
            std::cout << " (limited to source)";
        }
        std::cout << std::endl;
    }
    if (!tileConfig.batchExplicit) {
        const int desiredBatch = chooseAdaptiveTileBatch(
            inW, inH, tileConfig.width, tileConfig.height, tileConfig.overlap);
        tileConfig.batch = desiredBatch;
        const auto compilePrecision = modelCompilePrecision(impl_->opts.precision);
        std::vector<int> candidateBatches = {1, 2, 4, 8, 12, 16, desiredBatch};
        std::sort(candidateBatches.begin(), candidateBatches.end());
        candidateBatches.erase(
            std::unique(candidateBatches.begin(), candidateBatches.end()),
            candidateBatches.end());
        std::stable_sort(candidateBatches.begin(), candidateBatches.end(),
                         [desiredBatch](int lhs, int rhs) {
                             const int lhsDelta = std::abs(lhs - desiredBatch);
                             const int rhsDelta = std::abs(rhs - desiredBatch);
                             if (lhsDelta != rhsDelta) {
                                 return lhsDelta < rhsDelta;
                             }
                             return lhs > rhs;
                         });
        for (const int candidateBatch : candidateBatches) {
            std::string validationDetail;
            const auto validatedArtifact = impl_->modelManager.validatedCompiledArtifactPath(
                modelId, compilePrecision,
                static_cast<std::int64_t>(inW),
                static_cast<std::int64_t>(inH),
                candidateBatch,
                &validationDetail);
            if (validatedArtifact.has_value()) {
                tileConfig.batch = candidateBatch;
                if (candidateBatch != desiredBatch) {
                    std::cout << "[migraphx] reusing validated tile batch "
                              << candidateBatch << " instead of heuristic batch "
                              << desiredBatch << " for '" << modelId << "'." << std::endl;
                }
                break;
            }
        }
        std::cout << "[migraphx] adaptive tile batch selected: "
                  << tileConfig.batch << std::endl;
    }

    // Ensure a model program is ready for the requested tile size.
    {
        std::lock_guard<std::mutex> lk(impl_->mtx);
        if (!impl_->loadProgram(modelId, error, tileConfig.width, tileConfig.height,
                                selectedPath, selectedPathExplicit, inputVideo,
                                tileConfig.batch)) {
            std::cerr << "[migraphx] processVideoFile: model load failed: "
                      << error << "\n  → Deferring to FFmpeg." << std::endl;
            error.clear();
            return StageResult::Deferred;
        }
    }

    // Verify input/output contracts
    TensorContract inputContract;
    TensorContract outputContract;
    std::shared_ptr<ModelProgram> loadedProgram;
    {
        std::lock_guard<std::mutex> lk(impl_->mtx);
        const auto pit = impl_->programs.find(modelId);
        if (pit == impl_->programs.end() || pit->second == nullptr) {
            error = InferenceError::runtimeFailure(
                "Program unexpectedly missing after successful load: " + modelId).format();
            return StageResult::Error;
        }
        loadedProgram = pit->second;
        if (loadedProgram->inputContracts.empty()) {
            return deferToFfmpeg("Model has no input contracts.");
        }
        inputContract = loadedProgram->inputContracts.front();
        for (const auto& c : loadedProgram->inputContracts) {
            if (c.name == "input") { inputContract = c; break; }
        }
        if (loadedProgram->outputContracts.empty()) {
            return deferToFfmpeg("Model has no output contracts.");
        }
        outputContract = loadedProgram->outputContracts.front();
    }

    int modelInW = 0, modelInH = 0, modelInC = 0;
    int modelOutW = 0, modelOutH = 0, modelOutC = 0;
    const StereoStageConfig stereoConfig = buildStereoStageConfig(stage);
    std::string contractError;
    if (!extractSpatialDims(inputContract, modelInW, modelInH, modelInC, contractError)) {
        return deferToFfmpeg("Cannot derive model spatial dims: " + contractError);
    }
    int outputBatch = extractBatchSize(outputContract);
    if (stereo3dStage) {
        if (!extractStereoDepthOutputDims(outputContract, tileConfig.batch,
                                          modelOutW, modelOutH, modelOutC,
                                          outputBatch, contractError)) {
            return deferToFfmpeg("Cannot derive stereo depth output dims: " + contractError);
        }
    } else if (!extractSpatialDims(outputContract, modelOutW, modelOutH, modelOutC, contractError)) {
        return deferToFfmpeg("Cannot derive model output spatial dims: " + contractError);
    }
    if (modelInC != 3) {
        return deferToFfmpeg("Model expects " + std::to_string(modelInC) + " channels, need 3.");
    }
    if (!stereo3dStage && modelOutC < 3) {
        return deferToFfmpeg("Model outputs " + std::to_string(modelOutC)
                           + " channels; need at least 3.");
    }
    if (stereo3dStage && modelOutC <= 0) {
        return deferToFfmpeg("Stereo 3D depth model outputs no channels.");
    }
    if (inputContract.dtype != TensorDtype::Fp32 &&
        inputContract.dtype != TensorDtype::Fp16) {
        return deferToFfmpeg("Model input dtype " + toString(inputContract.dtype)
                           + " is not supported by the current host staging path.");
    }
    if (outputContract.dtype != TensorDtype::Fp32 &&
        outputContract.dtype != TensorDtype::Fp16) {
        return deferToFfmpeg("Model output dtype " + toString(outputContract.dtype)
                           + " is not supported by the current host staging path.");
    }

    if (modelInW <= 0 || modelInH <= 0 || modelOutW <= 0 || modelOutH <= 0) {
        return deferToFfmpeg("Model reported non-positive tensor dimensions.");
    }
    if (!stereo3dStage && (modelOutW % modelInW != 0 || modelOutH % modelInH != 0)) {
        return deferToFfmpeg("Model output dimensions are not integer multiples of the tile input size.");
    }

    const int compiledBatch = extractBatchSize(inputContract);
    if (compiledBatch <= 0) {
        return deferToFfmpeg("Model reported a non-positive batch dimension.");
    }
    if (outputBatch != compiledBatch) {
        return deferToFfmpeg("Model input/output batch dimensions do not match.");
    }
    if (compiledBatch != tileConfig.batch) {
        std::cout << "[migraphx] requested tile batch " << tileConfig.batch
                  << ", loaded artifact batch " << compiledBatch
                  << "; using the loaded artifact." << std::endl;
    }

    const int scaleX = stereo3dStage ? 1 : modelOutW / modelInW;
    const int scaleY = stereo3dStage ? 1 : modelOutH / modelInH;
    if (scaleX <= 0 || scaleY <= 0) {
        return deferToFfmpeg("Derived non-positive model scale factors.");
    }

    const bool preserveSourceResolution =
        !stereo3dStage && stagePreservesNativeResolution(stage) &&
        (scaleX != 1 || scaleY != 1);

    int tileOverlap = tileConfig.overlap;
    const int maxOverlap = std::max(0, std::min((modelInW - 1) / 2, (modelInH - 1) / 2));
    if (tileOverlap > maxOverlap) {
        std::cout << "[migraphx] reducing tile overlap from " << tileConfig.overlap
                  << " to " << maxOverlap << " to match the loaded tile artifact."
                  << std::endl;
        tileOverlap = maxOverlap;
    }

    const int outW = stereo3dStage ? inW : inW * scaleX;
    const int outH = stereo3dStage ? inH : inH * scaleY;
    ImageSize finalOutputSize = stereo3dStage
        ? computeStereoOutputSize(outW, outH, stereoConfig)
        : preserveSourceResolution ? ImageSize{inW, inH}
                                   : ImageSize{outW, outH};
    if (!stereo3dStage && stage.kind == StageKind::Upscale) {
        const int requestedWidth = getStageIntParam(stage, "width", 0);
        const int requestedHeight = getStageIntParam(stage, "height", 0);
        if (requestedWidth > 0 && requestedHeight > 0) {
            finalOutputSize = ImageSize{requestedWidth, requestedHeight};
        }
    }
    const int finalOutW = finalOutputSize.width;
    const int finalOutH = finalOutputSize.height;
    const bool resizeFinalOutput = !stereo3dStage &&
        (finalOutW != outW || finalOutH != outH);
    const std::size_t outFrameBytes = static_cast<std::size_t>(finalOutW) *
                                      static_cast<std::size_t>(finalOutH) * 3u;
    const std::size_t assembledFrameBytes =
        static_cast<std::size_t>(outW) * static_cast<std::size_t>(outH) * 3u;
    const auto tileXs = buildTileStarts(inW, modelInW, tileOverlap);
    const auto tileYs = buildTileStarts(inH, modelInH, tileOverlap);
    std::vector<TileWindow> tileXWindows;
    std::vector<TileWindow> tileYWindows;
    tileXWindows.reserve(tileXs.size());
    tileYWindows.reserve(tileYs.size());
    for (std::size_t i = 0; i < tileXs.size(); ++i) {
        tileXWindows.push_back(computeTileWindow(tileXs, i, modelInW, inW));
    }
    for (std::size_t i = 0; i < tileYs.size(); ++i) {
        tileYWindows.push_back(computeTileWindow(tileYs, i, modelInH, inH));
    }
    std::vector<TileDispatch> tileDispatches;
    tileDispatches.reserve(tileXs.size() * tileYs.size());
    for (std::size_t tileRow = 0; tileRow < tileYs.size(); ++tileRow) {
        for (std::size_t tileCol = 0; tileCol < tileXs.size(); ++tileCol) {
            tileDispatches.push_back(TileDispatch{
                tileXs[tileCol],
                tileYs[tileRow],
                tileXWindows[tileCol],
                tileYWindows[tileRow],
            });
        }
    }
    const std::size_t tilesPerFrame = tileXs.size() * tileYs.size();

    const frame_io::RgbVideoSourceMode preferredFrameSourceMode =
        preferredFrameSourceModeFromEnv();
    RgbVideoLoopOptions loopOptions;
    loopOptions.inputVideo = inputVideo;
    loopOptions.outputVideo = outputVideo;
    loopOptions.inputWidth = inW;
    loopOptions.inputHeight = inH;
    loopOptions.outputWidth = finalOutW;
    loopOptions.outputHeight = finalOutH;
    loopOptions.fps = probe->effectiveFrameRate(30.0);
    loopOptions.fallbackFps = 30.0;
    loopOptions.totalFrames = totalFrames;
    loopOptions.backendTag = "migraphx";
    loopOptions.progressLabel = "MiGraphX";
    loopOptions.noFramesError = "No frames were decoded from " + inputVideo;
    loopOptions.preferredSourceMode = preferredFrameSourceMode;
    loopOptions.allowSourceFallback = true;
    loopOptions.sourceStdioBufferBytes = choosePipeStdioBufferBytes(
        static_cast<std::size_t>(inW) * static_cast<std::size_t>(inH) * 3u);
    loopOptions.sourcePipeBytes = choosePipeCapacityBytes(
        static_cast<std::size_t>(inW) * static_cast<std::size_t>(inH) * 3u);
    loopOptions.processOptions = opts;

    const std::size_t encodeQueueDepth = chooseFrameQueueDepth(
        outFrameBytes,
        kDefaultEncodeQueueDepth,
        kMaxEncodeQueueDepth,
        "AVE_MIGRAPHX_ENCODE_QUEUE_DEPTH");

    // ── Open FFmpeg encode pipe (software encode — always works) ──
    const auto encodeLogPath = makeTempLogPath("ave_migraphx_encode");
    const bool directOutputEncode = opts.directOutputEncode;
    frame_io::AsyncRgbVideoPipeEncoder encodeSink;
    frame_io::RgbVideoPipeEncoderOptions encodeOptions;
    encodeOptions.inputVideo = inputVideo;
    encodeOptions.outputVideo = outputVideo;
    encodeOptions.width = finalOutW;
    encodeOptions.height = finalOutH;
    encodeOptions.fps = probe->effectiveFrameRate(30.0);
    encodeOptions.directOutputEncode = directOutputEncode;
    encodeOptions.outputCodec = opts.outputCodec;
    encodeOptions.outputProfile = opts.outputProfile;
    encodeOptions.outputPreset = opts.outputPreset;
    encodeOptions.outputCrf = opts.outputCrf;
    encodeOptions.outputThreads = opts.outputThreads;
    encodeOptions.stdioBufferBytes = choosePipeStdioBufferBytes(outFrameBytes);
    encodeOptions.pipeBytes = choosePipeCapacityBytes(outFrameBytes);
    encodeOptions.queueDepth = encodeQueueDepth;
    encodeOptions.stderrLogPath = encodeLogPath;
    if (!encodeSink.open(encodeOptions, error)) {
        return StageResult::Error;
    }

    auto cleanupEncodeLog = [&]() {
        std::error_code ec;
        std::filesystem::remove(encodeLogPath, ec);
    };
    auto formatEncodeFailure = [&](const std::string& prefix,
                                   std::optional<int> rawStatus = std::nullopt) {
        std::ostringstream os;
        os << prefix;
        if (rawStatus.has_value()) {
            os << " (" << formatProcessExit(*rawStatus) << ")";
        }
        const std::string ffmpegLog = readLogTail(encodeLogPath);
        if (!ffmpegLog.empty()) {
            os << "\nffmpeg stderr:\n" << ffmpegLog;
        }
        return os.str();
    };

    AVE_ROCTX_RANGE("migraphx:processVideoFile");
    const std::size_t batchesPerFrame =
        (tilesPerFrame + static_cast<std::size_t>(compiledBatch) - 1u)
        / static_cast<std::size_t>(compiledBatch);
    std::cout << "[migraphx] processVideoFile: model='" << modelId
              << "' input=" << inW << "x" << inH
              << " model_output=" << modelOutW << "x" << modelOutH
              << " assembled_output=" << outW << "x" << outH
              << " final_output=" << finalOutW << "x" << finalOutH
              << " tile=" << modelInW << "x" << modelInH
              << " overlap=" << tileOverlap
              << " batch=" << compiledBatch
              << " scale=" << scaleX << "x" << scaleY
              << " tiles/frame=" << tilesPerFrame
              << " batches/frame=" << batchesPerFrame
              << " stage=" << toString(stage.kind)
              << " preserve_source_resolution="
              << (preserveSourceResolution ? "on" : "off")
              << " direct_final_encode=" << (directOutputEncode ? "on" : "off")
              << std::endl;

    const std::int64_t totalInputElements = inputContract.shape.elements();
    const std::int64_t totalOutputElements = outputContract.shape.elements();
    if (totalInputElements <= 0 || totalOutputElements <= 0 ||
        totalInputElements % compiledBatch != 0 ||
        totalOutputElements % compiledBatch != 0) {
        return deferToFfmpeg("Model batch tensor sizes are not divisible by the loaded batch.");
    }
    const std::size_t tileTensorElements =
        elementsPerBatch(inputContract, compiledBatch);
    const std::size_t batchedInputTensorElements =
        tileTensorElements * static_cast<std::size_t>(compiledBatch);
    const std::size_t expectedTileOutputElements =
        elementsPerBatch(outputContract, compiledBatch);
    const std::size_t expectedBatchOutputElements =
        expectedTileOutputElements * static_cast<std::size_t>(compiledBatch);
    const std::size_t expectedHostTileInputElements =
        static_cast<std::size_t>(modelInW) * static_cast<std::size_t>(modelInH)
        * static_cast<std::size_t>(modelInC);
    const std::size_t expectedHostTileOutputElements =
        static_cast<std::size_t>(modelOutW) * static_cast<std::size_t>(modelOutH)
        * static_cast<std::size_t>(modelOutC);
    if (tileTensorElements != expectedHostTileInputElements ||
        expectedTileOutputElements != expectedHostTileOutputElements) {
        return deferToFfmpeg("Model tensor layout does not match the host NCHW staging path.");
    }
    std::vector<std::uint8_t> rgbOut;
    std::vector<std::uint8_t> assembledRgbOut;
    std::vector<float> stereoDepthFrame;
    std::vector<float> batchedInputTensorFp32;
    std::vector<std::uint16_t> batchedInputTensorFp16;
    bool useMappedHostInputStaging = false;
#ifdef AVE_HAVE_HIP
    HipMappedHostBuffer mappedInputTensor;
    if (!impl_->opts.offloadCopy && hipMappedHostStagingEnabled()) {
        const std::size_t mappedBytes =
            batchedInputTensorElements * tensorDtypeSizeBytes(inputContract.dtype);
        std::string mappedBufferError;
        if (mappedBytes > 0u &&
            ensureHipMappedHostBufferCapacity(mappedInputTensor,
                                             mappedBytes,
                                             "batched tile input staging",
                                             mappedBufferError)) {
            useMappedHostInputStaging = true;
            std::cout << "[migraphx] using HIP mapped host staging for input tiles "
                         "(device alias path, explicit upload bypassed)."
                      << std::endl;
        } else if (mappedBytes > 0u) {
            std::cerr << "[migraphx] WARNING: " << mappedBufferError
                      << "\n  Falling back to pageable host staging plus explicit HIP upload."
                      << std::endl;
        }
    }
#endif
    if (inputContract.dtype == TensorDtype::Fp32) {
        if (!useMappedHostInputStaging) {
            batchedInputTensorFp32.resize(batchedInputTensorElements);
        }
    } else if (!useMappedHostInputStaging) {
        batchedInputTensorFp16.resize(batchedInputTensorElements);
    }
    if (resizeFinalOutput) {
        assembledRgbOut.resize(assembledFrameBytes);
    }

    if (loadedProgram != nullptr && !loadedProgram->warmupComplete) {
        std::cout << "[migraphx] warming compiled program for '" << modelId
                  << "' before first frame." << std::endl;
        obs::logMiGraphXEnvironment(
            loadedProgram->runtimeFields,
            "runtime-warmup",
            loadedProgram->sourcePath,
            "start");

        std::optional<ScopedRuntimeEnvOverrides> sessionRuntimeEnv;
        sessionRuntimeEnv.emplace(loadedProgram->runtimeEnvOverrides);
        if (!sessionRuntimeEnv->ok()) {
            const int encodeStatus = encodeSink.finish(true);
            AVE_ROCTX_RANGE_END();
            error = formatEncodeFailure(
                InferenceError::runtimeFailure(
                    "Unable to apply MiGraphX runtime environment overrides for the encode session.",
                    sessionRuntimeEnv->error()).format(),
                encodeStatus);
            cleanupEncodeLog();
            return StageResult::Error;
        }

        const void* warmupOutputData = nullptr;
        std::size_t warmupOutputElements = 0;
        TensorDtype warmupOutputDtype = TensorDtype::Unknown;
        std::string warmupError;
        const void* warmupInputData = nullptr;
        std::size_t warmupInputElements = batchedInputTensorElements;
        if (inputContract.dtype == TensorDtype::Fp16) {
            if (useMappedHostInputStaging) {
#ifdef AVE_HAVE_HIP
                auto* mappedInput = static_cast<std::uint16_t*>(mappedInputTensor.hostPtr);
                std::fill(mappedInput, mappedInput + batchedInputTensorElements, 0u);
                warmupInputData = mappedInputTensor.devicePtr;
#endif
            } else {
                std::fill(batchedInputTensorFp16.begin(), batchedInputTensorFp16.end(), 0u);
                warmupInputData = batchedInputTensorFp16.data();
                warmupInputElements = batchedInputTensorFp16.size();
            }
        } else {
            if (useMappedHostInputStaging) {
#ifdef AVE_HAVE_HIP
                auto* mappedInput = static_cast<float*>(mappedInputTensor.hostPtr);
                std::fill(mappedInput, mappedInput + batchedInputTensorElements, 0.0f);
                warmupInputData = mappedInputTensor.devicePtr;
#endif
            } else {
                std::fill(batchedInputTensorFp32.begin(), batchedInputTensorFp32.end(), 0.0f);
                warmupInputData = batchedInputTensorFp32.data();
                warmupInputElements = batchedInputTensorFp32.size();
            }
        }

        if (!impl_->runInference(modelId, warmupInputData, warmupInputElements,
                                 inputContract.dtype, warmupOutputData,
                                 warmupOutputElements, warmupOutputDtype,
                                 warmupError, false, &stage, false,
                                 useMappedHostInputStaging)) {
            const int encodeStatus = encodeSink.finish(true);
            AVE_ROCTX_RANGE_END();
            error = formatEncodeFailure("MiGraphX warmup failed: " + warmupError, encodeStatus);
            cleanupEncodeLog();
            return StageResult::Error;
        }
        {
            std::lock_guard<std::mutex> evalLock(loadedProgram->evalMutex);
            loadedProgram->prog.experimental_get_context().finish();
            loadedProgram->warmupComplete = true;
        }
        obs::logMiGraphXEnvironment(
            loadedProgram->runtimeFields,
            "runtime-warmup",
            loadedProgram->sourcePath,
            "complete");
    }
    using Clock = std::chrono::steady_clock;
    std::chrono::nanoseconds readTime{0};
    std::chrono::nanoseconds preprocessTime{0};
    std::chrono::nanoseconds inferenceTime{0};
    std::chrono::nanoseconds postprocessTime{0};
    std::chrono::nanoseconds writeTime{0};
    auto finishEncodeSession = [&](const bool discardPending) {
        const int encodeStatus = encodeSink.finish(discardPending);
        writeTime += encodeSink.totalWriteTime();
        return encodeStatus;
    };
    auto combineEncodeFailure = [&](const std::string& detail,
                                    const int encodeStatus,
                                    const bool preferEncodeFormatting) {
        std::string combinedDetail = detail;
        const std::string encodeWriterError = encodeSink.writerFailure();
        if (!encodeWriterError.empty() && encodeWriterError != detail) {
            combinedDetail += "\n" + encodeWriterError;
        }
        if (preferEncodeFormatting || encodeSink.writerFailed() || encodeStatus != 0) {
            return formatEncodeFailure(combinedDetail, encodeStatus);
        }
        return combinedDetail;
    };

    std::optional<ScopedRuntimeEnvOverrides> sessionRuntimeEnv;
    if (loadedProgram != nullptr) {
        sessionRuntimeEnv.emplace(loadedProgram->runtimeEnvOverrides);
        if (!sessionRuntimeEnv->ok()) {
            const int encodeStatus = finishEncodeSession(true);
            AVE_ROCTX_RANGE_END();
            error = combineEncodeFailure(
                InferenceError::runtimeFailure(
                    "Unable to apply MiGraphX runtime environment overrides for the encode session.",
                    sessionRuntimeEnv->error()).format(),
                encodeStatus,
                true);
            cleanupEncodeLog();
            return StageResult::Error;
        }
    }
    std::optional<double> stereoEmaMin;
    std::optional<double> stereoEmaMax;
    frame_io::VideoFramePacketMaterializer inputFrameMaterializer;
    bool loggedHardwareSourceFrame = false;
    const auto loopStart = Clock::now();

    const auto loopResult = runVideoFrameEncodeLoop(
        loopOptions,
        encodeSink,
        [&](const frame_io::VideoFramePacket& inputFrame,
            std::vector<std::uint8_t>& rgbOut,
            const int frameIdx,
            RgbVideoSourceLoopIteration& iteration,
            std::string& loopError) {
            const std::vector<std::uint8_t>* rgbInput = nullptr;
            AVFrame* softwareFrame = nullptr;
            PackedRgbFrameView packedRgbView;
            bool usePackedRgbFrameView = false;
            if (inputFrame.hasRgb24()) {
                rgbInput = &inputFrame.rgb24;
            } else {
                const auto preprocessStart = Clock::now();
                if (!inputFrameMaterializer.resolveSoftwareFrame(inputFrame, softwareFrame, loopError)) {
                    return false;
                }
                usePackedRgbFrameView = makePackedRgbFrameView(softwareFrame, packedRgbView);
                if (!usePackedRgbFrameView) {
                    if (!inputFrameMaterializer.resolveRgb24(inputFrame, rgbInput, loopError) ||
                        rgbInput == nullptr) {
                        return false;
                    }
                }
                preprocessTime += Clock::now() - preprocessStart;
                if (inputFrame.isHardwareFrame() && !loggedHardwareSourceFrame) {
                    loggedHardwareSourceFrame = true;
                    std::cout << "[migraphx] preserved Vulkan hardware frames reached the "
                                 "MiGraphX backend boundary; packet preparation is now owned "
                                 "inside MiGraphX, with RGB fallback only for unsupported CPU "
                                 "frame layouts."
                              << std::endl;
                }
            }
            if (rgbOut.size() != outFrameBytes) {
                rgbOut.resize(outFrameBytes);
            }

            std::vector<std::uint8_t>* assembledOutput = &rgbOut;
            if (resizeFinalOutput) {
                if (assembledRgbOut.size() != assembledFrameBytes) {
                    assembledRgbOut.resize(assembledFrameBytes);
                }
                assembledOutput = &assembledRgbOut;
            }

            if (stereo3dStage &&
                stereoDepthFrame.size() != static_cast<std::size_t>(outW) * static_cast<std::size_t>(outH)) {
                stereoDepthFrame.resize(
                    static_cast<std::size_t>(outW) * static_cast<std::size_t>(outH));
            }

            for (std::size_t batchStart = 0; batchStart < tileDispatches.size();
                 batchStart += static_cast<std::size_t>(compiledBatch)) {
                const std::size_t activeTiles = std::min(
                    static_cast<std::size_t>(compiledBatch),
                    tileDispatches.size() - batchStart);

                const auto preprocessStart = Clock::now();
                const void* inputTensorData = nullptr;
                std::size_t inputTensorElements = batchedInputTensorElements;
                if (inputContract.dtype == TensorDtype::Fp16) {
                    std::uint16_t* batchTensorPtr = nullptr;
                    if (useMappedHostInputStaging) {
#ifdef AVE_HAVE_HIP
                        batchTensorPtr = static_cast<std::uint16_t*>(mappedInputTensor.hostPtr);
#endif
                    } else {
                        batchTensorPtr = batchedInputTensorFp16.data();
                    }
                    for (std::size_t slot = 0; slot < static_cast<std::size_t>(compiledBatch); ++slot) {
                        const std::size_t dispatchIndex =
                            batchStart + std::min(slot, activeTiles - 1u);
                        const TileDispatch& dispatch = tileDispatches[dispatchIndex];
                        if (usePackedRgbFrameView) {
                            packPackedRgbFrameTileClampToNchwFp16(
                                packedRgbView,
                                dispatch.tileX,
                                dispatch.tileY,
                                modelInW,
                                modelInH,
                                batchTensorPtr + slot * tileTensorElements);
                        } else {
                            packRgbTileClampToNchwFp16(
                                rgbInput->data(), inW, inH,
                                dispatch.tileX, dispatch.tileY,
                                modelInW, modelInH,
                                batchTensorPtr + slot * tileTensorElements);
                        }
                    }
                    inputTensorData = batchTensorPtr;
#ifdef AVE_HAVE_HIP
                    if (useMappedHostInputStaging) {
                        inputTensorData = mappedInputTensor.devicePtr;
                    }
#endif
                    inputTensorElements = useMappedHostInputStaging
                        ? batchedInputTensorElements
                        : batchedInputTensorFp16.size();
                } else {
                    float* batchTensorPtr = nullptr;
                    if (useMappedHostInputStaging) {
#ifdef AVE_HAVE_HIP
                        batchTensorPtr = static_cast<float*>(mappedInputTensor.hostPtr);
#endif
                    } else {
                        batchTensorPtr = batchedInputTensorFp32.data();
                    }
                    for (std::size_t slot = 0; slot < static_cast<std::size_t>(compiledBatch); ++slot) {
                        const std::size_t dispatchIndex =
                            batchStart + std::min(slot, activeTiles - 1u);
                        const TileDispatch& dispatch = tileDispatches[dispatchIndex];
                        if (usePackedRgbFrameView) {
                            packPackedRgbFrameTileClampToNchwFp32(
                                packedRgbView,
                                dispatch.tileX,
                                dispatch.tileY,
                                modelInW,
                                modelInH,
                                batchTensorPtr + slot * tileTensorElements);
                        } else {
                            packRgbTileClampToNchwFp32(
                                rgbInput->data(), inW, inH,
                                dispatch.tileX, dispatch.tileY,
                                modelInW, modelInH,
                                batchTensorPtr + slot * tileTensorElements);
                        }
                    }
                    inputTensorData = batchTensorPtr;
#ifdef AVE_HAVE_HIP
                    if (useMappedHostInputStaging) {
                        inputTensorData = mappedInputTensor.devicePtr;
                    }
#endif
                    inputTensorElements = useMappedHostInputStaging
                        ? batchedInputTensorElements
                        : batchedInputTensorFp32.size();
                }
                preprocessTime += Clock::now() - preprocessStart;

                const auto inferenceStart = Clock::now();
                const void* outputTensorData = nullptr;
                std::size_t outputTensorElements = 0;
                TensorDtype outputTensorDtype = TensorDtype::Unknown;
                if (!impl_->runInference(modelId, inputTensorData, inputTensorElements,
                                         inputContract.dtype, outputTensorData,
                                         outputTensorElements, outputTensorDtype, loopError,
                                         true, &stage, false,
                                         useMappedHostInputStaging)) {
                    std::cerr << "[migraphx] Frame " << frameIdx
                              << " batch starting at work item " << batchStart
                              << " inference FAILED: " << loopError << std::endl;
                    return false;
                }
                inferenceTime += Clock::now() - inferenceStart;

                const auto postprocessStart = Clock::now();
                if (outputTensorData == nullptr || outputTensorElements != expectedBatchOutputElements) {
                    std::ostringstream os;
                    os << "Output tensor pointer/size mismatch at frame " << frameIdx
                       << " batch starting at work item " << batchStart
                       << ": ptr=" << outputTensorData
                       << " elems=" << outputTensorElements
                       << " expected=" << expectedBatchOutputElements;
                    loopError = os.str();
                    return false;
                }

                for (std::size_t slot = 0; slot < activeTiles; ++slot) {
                    const TileDispatch& dispatch = tileDispatches[batchStart + slot];
                    if (outputTensorDtype == TensorDtype::Fp16) {
                        const auto* tileOutput =
                            static_cast<const std::uint16_t*>(outputTensorData)
                            + slot * expectedTileOutputElements;
                        if (stereo3dStage) {
                            const int copyWidth =
                                dispatch.srcXWindow.end - dispatch.srcXWindow.begin;
                            const int copyHeight =
                                dispatch.srcYWindow.end - dispatch.srcYWindow.begin;
                            const int dstX = dispatch.srcXWindow.begin;
                            const int dstY = dispatch.srcYWindow.begin;
                            blitNchwFp16TileRegionChannel0ResampledToPlane(
                                tileOutput, modelOutW, modelOutH,
                                modelInW, modelInH,
                                dispatch.srcXWindow.offset,
                                dispatch.srcYWindow.offset,
                                copyWidth, copyHeight,
                                stereoDepthFrame, outW, dstX, dstY);
                        } else {
                            const int copyWidth =
                                (dispatch.srcXWindow.end - dispatch.srcXWindow.begin) * scaleX;
                            const int copyHeight =
                                (dispatch.srcYWindow.end - dispatch.srcYWindow.begin) * scaleY;
                            const int srcX = dispatch.srcXWindow.offset * scaleX;
                            const int srcY = dispatch.srcYWindow.offset * scaleY;
                            const int dstX = dispatch.srcXWindow.begin * scaleX;
                            const int dstY = dispatch.srcYWindow.begin * scaleY;
                            blitNchwFp16TileRegionToRgb24(
                                tileOutput, modelOutC, modelOutW, modelOutH,
                                srcX, srcY,
                                copyWidth, copyHeight,
                                *assembledOutput, outW, dstX, dstY);
                        }
                    } else if (outputTensorDtype == TensorDtype::Fp32) {
                        const auto* tileOutput =
                            static_cast<const float*>(outputTensorData)
                            + slot * expectedTileOutputElements;
                        if (stereo3dStage) {
                            const int copyWidth =
                                dispatch.srcXWindow.end - dispatch.srcXWindow.begin;
                            const int copyHeight =
                                dispatch.srcYWindow.end - dispatch.srcYWindow.begin;
                            const int dstX = dispatch.srcXWindow.begin;
                            const int dstY = dispatch.srcYWindow.begin;
                            blitNchwFp32TileRegionChannel0ResampledToPlane(
                                tileOutput, modelOutW, modelOutH,
                                modelInW, modelInH,
                                dispatch.srcXWindow.offset,
                                dispatch.srcYWindow.offset,
                                copyWidth, copyHeight,
                                stereoDepthFrame, outW, dstX, dstY);
                        } else {
                            const int copyWidth =
                                (dispatch.srcXWindow.end - dispatch.srcXWindow.begin) * scaleX;
                            const int copyHeight =
                                (dispatch.srcYWindow.end - dispatch.srcYWindow.begin) * scaleY;
                            const int srcX = dispatch.srcXWindow.offset * scaleX;
                            const int srcY = dispatch.srcYWindow.offset * scaleY;
                            const int dstX = dispatch.srcXWindow.begin * scaleX;
                            const int dstY = dispatch.srcYWindow.begin * scaleY;
                            blitNchwFp32TileRegionToRgb24(
                                tileOutput, modelOutC, modelOutW, modelOutH,
                                srcX, srcY,
                                copyWidth, copyHeight,
                                *assembledOutput, outW, dstX, dstY);
                        }
                    } else {
                        std::ostringstream os;
                        os << "Unsupported output tensor dtype at frame " << frameIdx
                           << " batch starting at work item " << batchStart << ": "
                           << toString(outputTensorDtype);
                        loopError = os.str();
                        return false;
                    }
                }
                postprocessTime += Clock::now() - postprocessStart;
            }

            if (stereo3dStage) {
                const auto stereoPostprocessStart = Clock::now();
                if (rgbInput == nullptr &&
                    (!inputFrameMaterializer.resolveRgb24(inputFrame, rgbInput, loopError) ||
                     rgbInput == nullptr)) {
                    return false;
                }
                postprocessDepthMap(
                    stereoDepthFrame, outW, outH, stereoConfig,
                    stereoEmaMin, stereoEmaMax);
                if (!renderStereoFrame(
                        rgbInput->data(), inW, inH,
                        stereoDepthFrame, outW, outH,
                        stereoConfig, rgbOut)) {
                    loopError = "Stereo 3D frame synthesis failed.";
                    return false;
                }
                postprocessTime += Clock::now() - stereoPostprocessStart;
            } else if (resizeFinalOutput) {
                resizeRgbBufferBilinear(
                    assembledOutput->data(), outW, outH,
                    rgbOut.data(), finalOutW, finalOutH);
            }

            if (rgbOut.size() != outFrameBytes) {
                std::cerr << "[migraphx] Frame " << frameIdx
                          << " output size mismatch: " << rgbOut.size()
                          << " vs expected " << outFrameBytes << std::endl;
                std::ostringstream os;
                os << "Output tensor size mismatch at frame " << frameIdx;
                loopError = os.str();
                return false;
            }

            const int nextFrameIdx = frameIdx + 1;
            iteration.previewRgb = rgbOut.data();
            iteration.previewWidth = finalOutW;
            iteration.previewHeight = finalOutH;
            iteration.progressMessage =
                "processed frame " + std::to_string(nextFrameIdx)
                + (totalFrames > 0 ? "/" + std::to_string(totalFrames) : "");
            return true;
        },
        progressCb,
        error);

    readTime = loopResult.readTime;
    const int frameIdx = loopResult.frameCount;
    const int encodeRet = finishEncodeSession(loopResult.stageResult != StageResult::Processed);
    const bool encodeWriterFailed = encodeSink.writerFailed();
    const std::string encodeWriterError = encodeSink.writerFailure();

    AVE_ROCTX_RANGE_END();

    if (loopResult.stageResult == StageResult::Cancelled) {
        cleanupEncodeLog();
        return StageResult::Cancelled;
    }
    if (loopResult.stageResult == StageResult::Error) {
        if (encodeWriterFailed || encodeRet != 0) {
            error = combineEncodeFailure(error, encodeRet, false);
        }
        cleanupEncodeLog();
        return StageResult::Error;
    }

    if (encodeWriterFailed) {
        error = formatEncodeFailure(
            encodeWriterError.empty()
                ? "FFmpeg encode pipe exited after MiGraphX processing."
                : encodeWriterError,
            encodeRet);
        cleanupEncodeLog();
        return StageResult::Error;
    }

    if (encodeRet != 0) {
        error = formatEncodeFailure(
            "FFmpeg encode pipe exited after MiGraphX processing.",
            encodeRet);
        cleanupEncodeLog();
        return StageResult::Error;
    }

    cleanupEncodeLog();

    const auto totalElapsed = Clock::now() - loopStart;
    const double totalSeconds = std::chrono::duration<double>(totalElapsed).count();
    const auto avgMs = [frameIdx](std::chrono::nanoseconds totalNs) {
        if (frameIdx <= 0) {
            return 0.0;
        }
        return std::chrono::duration<double, std::milli>(totalNs).count()
             / static_cast<double>(frameIdx);
    };
    const double throughputFps = totalSeconds > 0.0
        ? static_cast<double>(frameIdx) / totalSeconds
        : 0.0;
    std::cout << "[migraphx] AI inference complete: " << frameIdx
              << " frames processed via MiGraphX for stage "
              << toString(stage.kind) << std::endl;
    std::cout << "[migraphx] timing: total=" << totalSeconds
              << "s fps=" << throughputFps
              << " avg_ms/frame read=" << avgMs(readTime)
              << " preprocess=" << avgMs(preprocessTime)
              << " infer=" << avgMs(inferenceTime)
              << " postprocess=" << avgMs(postprocessTime)
              << " write=" << avgMs(writeTime)
              << " direct_final_encode=" << (directOutputEncode ? "on" : "off")
              << std::endl;

    if (progressCb) {
        progressCb(1.0f, "MiGraphX inference complete — " + std::to_string(frameIdx) + " frames.");
    }

    return StageResult::Processed;
#else
    (void)stage;
    (void)inputVideo;
    (void)outputVideo;
    (void)progressCb;
    (void)opts;
    error = "MiGraphX backend not compiled (-DAVE_HAVE_MIGRAPHX=OFF).";
    return StageResult::Deferred;
#endif
}
}  // namespace ave
