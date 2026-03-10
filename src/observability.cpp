// ─────────────────────────────────────────────────────────────────
// observability.cpp — Version logging, env-var audit, manifest I/O
// ─────────────────────────────────────────────────────────────────
#include "ave/observability.hpp"

#include <array>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>

#ifdef AVE_HAVE_MIGRAPHX
#  include <migraphx/version.h>
#endif
#ifdef AVE_HAVE_HIP
#  include <hip/hip_runtime.h>
#endif

namespace ave {
namespace obs {

namespace {

// Run a shell command and capture its first line of stdout.
std::string captureFirstLine(const std::string& cmd) {
    FILE* p = popen(cmd.c_str(), "r");  // NOLINT(cert-env33-c)
    if (p == nullptr) { return "?"; }
    std::array<char, 256> buf{};
    std::string out;
    if (std::fgets(buf.data(), static_cast<int>(buf.size()), p) != nullptr) {
        out = std::string(buf.data());
        // strip trailing newline
        while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) {
            out.pop_back();
        }
    }
    pclose(p);
    return out.empty() ? "?" : out;
}

// Read the first non-empty line of a file, return "?" if missing.
std::string readFirstLine(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) { return "?"; }
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty()) { return line; }
    }
    return "?";
}

} // namespace

// ─────────────────────────────────────────────────────────────────
// logVersionTuple
// ─────────────────────────────────────────────────────────────────
void logVersionTuple() {
    // ROCm version
    const std::string rocmVer = readFirstLine("/opt/rocm/.info/version");

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
    std::string gfxTarget = "?";
#ifdef AVE_HAVE_HIP
    int device = 0;
    hipDeviceProp_t props{};
    if (hipGetDevice(&device) == hipSuccess &&
        hipGetDeviceProperties(&props, device) == hipSuccess &&
        props.gcnArchName[0] != '\0') {
        gfxTarget = props.gcnArchName;
    }
#endif
    if (gfxTarget == "?") {
        gfxTarget = captureFirstLine("rocminfo 2>/dev/null"
                                     " | grep -m1 'amdgcn-amd-amdhsa--gfx' | awk -F'--' '{print $2}'");
    }

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

bool writeArtifactManifest(const std::string&            manifestPath,
                           const ArtifactManifestFields& f,
                           std::string&                  error) {
    std::ofstream out(manifestPath, std::ios::trunc);
    if (!out.is_open()) {
        error = "Cannot write manifest: " + manifestPath;
        return false;
    }
    out << "migraphx_version=" << f.migraphxVersion << '\n'
        << "rocm_version="     << f.rocmVersion     << '\n'
        << "gpu_gfx_target="   << f.gpuGfxTarget    << '\n'
        << "onnx_file_size="   << f.onnxFileSizeStr  << '\n'
        << "onnx_mtime="       << f.onnxMtimeStr     << '\n'
        << "offload_copy="     << f.offloadCopy      << '\n'
        << "precision="        << f.precision        << '\n'
        << "disable_mlir="     << f.disableMlir      << '\n'
        << "enable_nhwc="      << f.enableNhwc       << '\n'
        << "enable_ck="        << f.enableCk         << '\n';
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

    return check("migraphx_version", expected.migraphxVersion)
        && check("rocm_version",     expected.rocmVersion)
        && check("gpu_gfx_target",   expected.gpuGfxTarget)
        && check("onnx_file_size",   expected.onnxFileSizeStr)
        && check("onnx_mtime",       expected.onnxMtimeStr)
        && check("offload_copy",     expected.offloadCopy)
        && check("precision",        expected.precision)
        && check("disable_mlir",     expected.disableMlir)
        && check("enable_nhwc",      expected.enableNhwc)
        && check("enable_ck",        expected.enableCk);
}

}  // namespace obs
}  // namespace ave
