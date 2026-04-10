// ─────────────────────────────────────────────────────────────────
// observability.cpp — Version logging, env-var audit, manifest I/O
// ─────────────────────────────────────────────────────────────────
#include "ave/observability.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>

#include "ave/process_observer.hpp"
#include "ave/runtime_diagnostics.hpp"

#ifdef AVE_HAVE_MIGRAPHX
#  include <migraphx/version.h>
#endif
#ifdef AVE_HAVE_HIP
#  include <hip/hip_runtime.h>
#endif

namespace ave {
namespace obs {

namespace {

constexpr std::uint64_t kFnv1a64Offset = 1469598103934665603ull;
constexpr std::uint64_t kFnv1a64Prime = 1099511628211ull;

// Run a shell command and capture its first line of stdout.
std::string captureFirstLine(const std::string& cmd) {
    int exitCode = 0;
    const auto output = process_observer::captureCommandStdout(cmd, exitCode);
    if (!output.has_value() || exitCode != 0) {
        return "?";
    }
    std::istringstream input(*output);
    for (std::string line; std::getline(input, line);) {
        const std::string trimmed = process_observer::trimOutput(line);
        if (!trimmed.empty()) {
            return trimmed;
        }
    }
    return "?";
}

void appendFnv1a64(std::uint64_t& hash, const std::string_view bytes) {
    for (const char rawCh : bytes) {
        const auto ch = static_cast<unsigned char>(rawCh);
        hash ^= static_cast<std::uint64_t>(ch);
        hash *= kFnv1a64Prime;
    }
}

std::string hexFingerprint(const std::uint64_t value) {
    std::ostringstream out;
    out << std::hex << value;
    return out.str();
}

} // namespace

// ─────────────────────────────────────────────────────────────────
// logVersionTuple
// ─────────────────────────────────────────────────────────────────
void logVersionTuple() {
    // ROCm version
    const std::string rocmVer = detectRocmVersion();

    // MiGraphX version
#ifdef AVE_HAVE_MIGRAPHX
    const std::string migraphxVer =
        std::to_string(MIGRAPHX_VERSION_MAJOR) + "."
        + std::to_string(MIGRAPHX_VERSION_MINOR) + "."
        + std::to_string(MIGRAPHX_VERSION_PATCH);
#else
    const std::string migraphxVer = "(not compiled in)";
#endif

    // Vulkan API + driver (via vulkaninfo, best-effort)
    const std::string vulkanInfo =
        captureFirstLine("vulkaninfo --summary 2>/dev/null"
                         " | grep -E 'apiVersion|driverVersion'"
                         " | head -1");

    // FFmpeg version
    const std::string ffmpegVer =
        captureFirstLine("ffmpeg -version 2>&1 | head -1");

    // GPU gfx target
    const std::string gfxTarget = detectAmdGpuArch();

    // Kernel version
    const std::string kernel = captureFirstLine("uname -r");

    std::cout << "[version] rocm=" << rocmVer << '\n'
              << "[version] migraphx=" << migraphxVer << '\n'
              << "[version] vulkan_info=" << vulkanInfo << '\n'
              << "[version] ffmpeg=" << ffmpegVer << '\n'
              << "[version] gpu_gfx_target=" << gfxTarget << '\n'
              << "[version] kernel=" << kernel << std::endl;
}

// ─────────────────────────────────────────────────────────────────
// logMiGraphXEnvironment
// ─────────────────────────────────────────────────────────────────
void logMiGraphXEnvironment() {
    // Gold standard: "log all options and relevant MIGRAPHX_* env var
    // values that affect compilation"
    static const char* vars[] = {
        "MIGRAPHX_TRACE_COMPILE",
        "MIGRAPHX_TRACE_PASSES",
        "MIGRAPHX_TIME_PASSES",
        "MIGRAPHX_TRACE_HIPRTC",
        "MIGRAPHX_GPU_DUMP_SRC",
        "MIGRAPHX_GPU_DUMP_ASM",
        "MIGRAPHX_DEBUG_SAVE_TEMP_DIR",
        "MIGRAPHX_GPU_DEBUG",
        "MIGRAPHX_DISABLE_MLIR",
        "MIGRAPHX_TRACE_MLIR",
        "MIGRAPHX_ENABLE_NHWC",
        "MIGRAPHX_ENABLE_CK",
        "MIGRAPHX_PROBLEM_CACHE",
        "MIOPEN_USER_DB_PATH",
        "MIOPEN_CUSTOM_CACHE_DIR",
        "MIOPEN_FIND_MODE",
        "MIOPEN_COMPILE_PARALLEL_LEVEL",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "AVE_MIGRAPHX_COMPILE_PROFILE",
        nullptr
    };
    std::cout << "[migraphx-env]";
    bool anySet = false;
    for (int i = 0; vars[i] != nullptr; ++i) {
        const char* val = std::getenv(vars[i]);
        if (val != nullptr) {
            std::cout << '\n' << "  " << vars[i] << "=" << val;
            anySet = true;
        }
    }
    if (!anySet) {
        std::cout << " (no MIGRAPHX_* env vars set – using defaults)";
    }
    std::cout << std::endl;
}

void logMiGraphXEnvironment(const ArtifactManifestFields& effective,
                            const std::string& phase,
                            const std::string& artifactPath,
                            const std::string& warmupStatus) {
    std::cout << "[migraphx-env] phase=" << phase << '\n'
              << "  migraphx_version=" << effective.migraphxVersion << '\n'
              << "  rocm_version=" << effective.rocmVersion << '\n'
              << "  gpu_gfx_target=" << effective.gpuGfxTarget << '\n'
              << "  precision=" << effective.precision << '\n'
              << "  compile_profile=" << effective.compileProfile << '\n'
              << "  offload_copy=" << effective.offloadCopy << '\n'
              << "  MIGRAPHX_DISABLE_MLIR=" << effective.disableMlir << '\n'
              << "  MIGRAPHX_ENABLE_NHWC=" << effective.enableNhwc << '\n'
              << "  MIGRAPHX_ENABLE_CK=" << effective.enableCk << '\n'
              << "  MIGRAPHX_PROBLEM_CACHE=" << effective.problemCachePath << '\n'
              << "  MIOPEN_USER_DB_PATH=" << effective.miopenUserDbPath << '\n'
              << "  MIOPEN_CUSTOM_CACHE_DIR=" << effective.miopenCustomCacheDir << '\n'
              << "  MIOPEN_FIND_MODE=" << effective.miopenFindMode << '\n'
              << "  MIOPEN_COMPILE_PARALLEL_LEVEL=" << effective.miopenCompileParallelLevel << '\n'
              << "  visible_devices=" << effective.visibleDevices << '\n'
              << "  runtime_fingerprint=" << effective.runtimeFingerprint;
    if (!artifactPath.empty()) {
        std::cout << '\n' << "  artifact_path=" << artifactPath;
    }
    if (!warmupStatus.empty()) {
        std::cout << '\n' << "  warmup_status=" << warmupStatus;
    }
    std::cout << std::endl;
}

// ─────────────────────────────────────────────────────────────────
// logTensorContractViolation
// ─────────────────────────────────────────────────────────────────
void logTensorContractViolation(const std::string& location,
                                const std::string& expected,
                                const std::string& actual) {
    std::cerr << "[tensor-contract-violation] at " << location << '\n'
              << "  expected: " << expected << '\n'
              << "  actual:   " << actual << std::endl;
}

// ─────────────────────────────────────────────────────────────────
// Artifact manifest I/O
// ─────────────────────────────────────────────────────────────────
// Plain-text INI-style format:
//   key=value\n
// All fields from ArtifactManifestFields, one per line, in order.

std::string buildArtifactSourceFingerprint(const std::string& sourcePath) {
    const std::filesystem::path path(sourcePath);
    std::error_code ec;
    if (!std::filesystem::exists(path, ec) || ec) {
        return "missing";
    }

    const auto fileSize = std::filesystem::file_size(path, ec);
    if (ec) {
        return "unreadable";
    }

    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        return "unreadable";
    }

    std::uint64_t hash = kFnv1a64Offset;
    std::array<char, 64 * 1024> buffer{};
    while (in.good()) {
        in.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto bytesRead = static_cast<std::size_t>(in.gcount());
        if (bytesRead == 0u) {
            break;
        }
        appendFnv1a64(hash, std::string_view(buffer.data(), bytesRead));
    }

    if (in.bad()) {
        return "unreadable";
    }

    std::ostringstream out;
    out << fileSize << ":" << hexFingerprint(hash);
    return out.str();
}

bool writeArtifactManifest(const std::string&            manifestPath,
                           const ArtifactManifestFields& f,
                           std::string&                  error) {
    std::ofstream out(manifestPath, std::ios::trunc);
    if (!out.is_open()) {
        error = "Cannot write manifest: " + manifestPath;
        return false;
    }
    out << "manifest_schema=" << f.manifestSchemaVersion << '\n'
        << "migraphx_version=" << f.migraphxVersion << '\n'
        << "rocm_version="     << f.rocmVersion     << '\n'
        << "gpu_gfx_target="   << f.gpuGfxTarget    << '\n'
        << "onnx_file_size="   << f.onnxFileSizeStr  << '\n'
        << "onnx_mtime="       << f.onnxMtimeStr     << '\n'
        << "source_fingerprint=" << f.sourceFingerprint << '\n'
        << "offload_copy="     << f.offloadCopy      << '\n'
        << "precision="        << f.precision        << '\n'
        << "compile_profile="  << f.compileProfile   << '\n'
        << "disable_mlir="     << f.disableMlir      << '\n'
        << "enable_nhwc="      << f.enableNhwc       << '\n'
        << "enable_ck="        << f.enableCk         << '\n'
        << "problem_cache_path=" << f.problemCachePath << '\n'
        << "miopen_user_db_path=" << f.miopenUserDbPath << '\n'
        << "miopen_custom_cache_dir=" << f.miopenCustomCacheDir << '\n'
        << "miopen_find_mode=" << f.miopenFindMode << '\n'
        << "miopen_compile_parallel_level=" << f.miopenCompileParallelLevel << '\n'
        << "visible_devices=" << f.visibleDevices << '\n'
        << "runtime_fingerprint=" << f.runtimeFingerprint << '\n';
    return true;
}

bool validateArtifactManifest(const std::string&            manifestPath,
                               const ArtifactManifestFields& expected,
                               std::string&                  mismatchReason) {
    std::ifstream in(manifestPath);
    if (!in.is_open()) {
        mismatchReason = "Manifest not found: " + manifestPath;
        return false;
    }

    // Parse into a simple map
    std::unordered_map<std::string, std::string> stored;
    std::string line;
    while (std::getline(in, line)) {
        const auto eq = line.find('=');
        if (eq == std::string::npos) { continue; }
        stored[line.substr(0, eq)] = line.substr(eq + 1);
    }

    // Check each field (fail on first mismatch – surface exactly what changed)
    auto check = [&](const std::string& key, const std::string& exp) -> bool {
        const auto it = stored.find(key);
        if (it == stored.end()) {
            mismatchReason = "Manifest missing field '" + key + "'.";
            return false;
        }
        if (it->second != exp) {
            mismatchReason = "Manifest field '" + key + "' mismatch: "
                           + "stored='" + it->second + "' expected='" + exp + "'.";
            return false;
        }
        return true;
    };

    return check("manifest_schema", expected.manifestSchemaVersion)
        && check("migraphx_version", expected.migraphxVersion)
        && check("rocm_version",     expected.rocmVersion)
        && check("gpu_gfx_target",   expected.gpuGfxTarget)
        && check("onnx_file_size",   expected.onnxFileSizeStr)
        && check("onnx_mtime",       expected.onnxMtimeStr)
        && check("source_fingerprint", expected.sourceFingerprint)
        && check("offload_copy",     expected.offloadCopy)
        && check("precision",        expected.precision)
        && check("compile_profile",  expected.compileProfile)
        && check("disable_mlir",     expected.disableMlir)
        && check("enable_nhwc",      expected.enableNhwc)
        && check("enable_ck",        expected.enableCk)
        && check("problem_cache_path", expected.problemCachePath)
        && check("miopen_user_db_path", expected.miopenUserDbPath)
        && check("miopen_custom_cache_dir", expected.miopenCustomCacheDir)
        && check("miopen_find_mode", expected.miopenFindMode)
        && check("miopen_compile_parallel_level", expected.miopenCompileParallelLevel)
        && check("visible_devices", expected.visibleDevices)
        && check("runtime_fingerprint", expected.runtimeFingerprint);
}

}  // namespace obs
}  // namespace ave
