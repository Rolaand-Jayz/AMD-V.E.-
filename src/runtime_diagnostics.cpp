#include "ave/runtime_diagnostics.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#if defined(__unix__)
#  include <unistd.h>
#endif

#ifdef AVE_HAVE_HIP
#  include <hip/hip_runtime.h>
#endif

#ifdef AVE_HAVE_ONNXRUNTIME_ROCM
#  include <onnxruntime/onnxruntime_cxx_api.h>
#endif

#include "ave/runtime_paths.hpp"
#include "ave/process_observer.hpp"

namespace ave {
namespace {

namespace fs = std::filesystem;

bool pathExists(const fs::path& path) {
    std::error_code ec;
    return fs::exists(path, ec);
}

bool anyPathExists(const std::vector<fs::path>& paths) {
    for (const auto& path : paths) {
        if (pathExists(path)) {
            return true;
        }
    }
    return false;
}

std::optional<std::string> readNonEmptyEnv(const char* name) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0') {
        return std::nullopt;
    }
    return std::string(raw);
}

std::optional<int> parseNonNegativeInt(const std::optional<std::string>& raw) {
    if (!raw.has_value()) {
        return std::nullopt;
    }
    try {
        const int value = std::stoi(*raw);
        if (value < 0) {
            return std::nullopt;
        }
        return value;
    } catch (...) {
        return std::nullopt;
    }
}

std::string summarizeFailures(const std::vector<RuntimeDiagnosticCheck>& checks) {
    std::ostringstream out;
    bool first = true;
    for (const auto& check : checks) {
        if (check.status == RuntimeDiagnosticStatus::Ok) {
            continue;
        }
        if (!first) {
            out << "; ";
        }
        out << check.title << ": " << check.detail;
        first = false;
    }
    return out.str();
}

std::string toLowerCopy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](const unsigned char ch) {
                       return static_cast<char>(std::tolower(ch));
                   });
    return value;
}

bool onnxRuntimeRocmProviderDetected() {
#ifdef AVE_HAVE_ONNXRUNTIME_ROCM
    try {
        const auto providers = Ort::GetAvailableProviders();
        return std::any_of(providers.begin(), providers.end(),
                           [](const std::string& provider) {
                               return toLowerCopy(provider).find("rocm") != std::string::npos;
                           });
    } catch (...) {
        return false;
    }
#else
    return false;
#endif
}

}  // namespace

std::optional<int> preferredAmdDeviceIndexFromEnv() {
    if (const auto specific = parseNonNegativeInt(readNonEmptyEnv("AVE_MIGRAPHX_DEVICE_INDEX"));
        specific.has_value()) {
        return specific;
    }
    return parseNonNegativeInt(readNonEmptyEnv("AVE_GPU_INDEX"));
}

std::string detectAmdGpuArch() {
    static const std::string cached = []() -> std::string {
        if (const auto overrideArch = readNonEmptyEnv("HSA_OVERRIDE_GFX_VERSION");
            overrideArch.has_value()) {
            return *overrideArch;
        }
        if (const auto rawTargets = readNonEmptyEnv("GPU_TARGETS");
            rawTargets.has_value()) {
            return *rawTargets;
        }
#ifdef AVE_HAVE_HIP
        hipDeviceProp_t props{};
        if (const auto preferredDevice = preferredAmdDeviceIndexFromEnv(); preferredDevice.has_value() &&
            hipGetDeviceProperties(&props, *preferredDevice) == hipSuccess &&
            props.gcnArchName[0] != '\0') {
            return std::string(props.gcnArchName);
        }
        int currentDevice = 0;
        if (hipGetDevice(&currentDevice) == hipSuccess &&
            hipGetDeviceProperties(&props, currentDevice) == hipSuccess &&
            props.gcnArchName[0] != '\0') {
            return std::string(props.gcnArchName);
        }
#endif
        int exitCode = 0;
        const auto rocminfoOutput = process_observer::captureCommandStdout(
            "rocminfo 2>/dev/null | grep -m1 'gfx[0-9]' | tr -s ' ' | cut -d' ' -f2",
            exitCode);
        if (rocminfoOutput.has_value() && exitCode == 0) {
            const std::string trimmed = process_observer::trimOutput(*rocminfoOutput);
            if (!trimmed.empty()) {
                return trimmed;
            }
        }
        return std::string("unknown");
    }();
    return cached;
}

std::string detectRocmVersion() {
    static const std::string cached = []() -> std::string {
        std::ifstream file("/opt/rocm/.info/version");
        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty()) {
                return line;
            }
        }
        return std::string("unknown");
    }();
    return cached;
}

std::string toString(const RuntimeDiagnosticStatus status) {
    switch (status) {
        case RuntimeDiagnosticStatus::Ok: return "ok";
        case RuntimeDiagnosticStatus::Warning: return "warning";
        case RuntimeDiagnosticStatus::Error: return "error";
    }
    return "unknown";
}

AmdRuntimeSnapshot probeAmdRuntimeSnapshot() {
    AmdRuntimeSnapshot snapshot;

    if (const char* envRocm = std::getenv("ROCM_PATH");
        envRocm != nullptr && *envRocm != '\0') {
        snapshot.rocmRoot = envRocm;
    }

    snapshot.rocmRootPresent = pathExists(snapshot.rocmRoot);
    snapshot.kfdDevicePresent = pathExists("/dev/kfd");
#if defined(__unix__)
    snapshot.kfdAccessible = snapshot.kfdDevicePresent &&
        (::access("/dev/kfd", R_OK | W_OK) == 0);
#else
    snapshot.kfdAccessible = snapshot.kfdDevicePresent;
#endif
    snapshot.rocminfoPresent = process_observer::commandInPath("rocminfo");
    snapshot.hipInfoPresent = process_observer::commandInPath("hipInfo")
        || process_observer::commandInPath("hipinfo");

    if (const auto bundledDriver = bundledMiGraphXDriverPath(); bundledDriver.has_value()) {
        snapshot.migraphxDriverPresent = true;
    } else {
        snapshot.migraphxDriverPresent = process_observer::commandInPath("migraphx-driver");
    }

    std::vector<fs::path> migraphxLibraryCandidates{
        fs::path(snapshot.rocmRoot) / "lib" / "libmigraphx.so",
        fs::path(snapshot.rocmRoot) / "lib64" / "libmigraphx.so",
        "/usr/lib/libmigraphx.so",
        "/usr/lib64/libmigraphx.so",
        "/usr/local/lib/libmigraphx.so",
    };
    if (const auto bundledPrefix = bundledMiGraphXPrefix(); bundledPrefix.has_value()) {
        migraphxLibraryCandidates.push_back(*bundledPrefix / "lib" / "libmigraphx.so");
        migraphxLibraryCandidates.push_back(*bundledPrefix / "lib" / "migraphx" / "lib" / "libmigraphx.so");
    }
    snapshot.migraphxLibraryPresent = anyPathExists(migraphxLibraryCandidates);
#ifdef AVE_HAVE_MIGRAPHX
    snapshot.migraphxLibraryPresent = true;
#endif

    snapshot.onnxruntimeLibraryPresent = anyPathExists({
        fs::path(snapshot.rocmRoot) / "lib" / "libonnxruntime.so",
        fs::path(snapshot.rocmRoot) / "lib64" / "libonnxruntime.so",
        "/usr/lib/libonnxruntime.so",
        "/usr/lib64/libonnxruntime.so",
        "/usr/local/lib/libonnxruntime.so",
        "/usr/local/lib64/libonnxruntime.so",
    });
#ifdef AVE_HAVE_ONNXRUNTIME_ROCM
    snapshot.onnxruntimeLibraryPresent = true;
#endif
    snapshot.onnxruntimeRocmProviderPresent = onnxRuntimeRocmProviderDetected();

    snapshot.hipRuntimePresent = anyPathExists({
        fs::path(snapshot.rocmRoot) / "lib" / "libamdhip64.so",
        fs::path(snapshot.rocmRoot) / "lib64" / "libamdhip64.so",
        "/usr/lib/libamdhip64.so",
        "/usr/lib64/libamdhip64.so",
    });
#ifdef AVE_HAVE_HIP
    snapshot.hipRuntimePresent = true;
#endif
    snapshot.vulkanLoaderPresent = anyPathExists({
        "/usr/lib/libvulkan.so",
        "/usr/lib64/libvulkan.so",
        "/usr/lib/libvulkan.so.1",
        "/usr/lib64/libvulkan.so.1",
    }) || process_observer::commandInPath("vulkaninfo");
    snapshot.ncnnRuntimePresent = snapshot.vulkanLoaderPresent;

    snapshot.gpuArch = detectAmdGpuArch();

    return snapshot;
}

RuntimeDiagnosticsReport buildRuntimeDiagnosticsReport(const AmdRuntimeSnapshot& snapshot) {
    RuntimeDiagnosticsReport report;
    report.checks.reserve(7);

    report.checks.push_back({
        "rocm-root",
        "ROCm root",
        snapshot.rocmRootPresent ? RuntimeDiagnosticStatus::Ok : RuntimeDiagnosticStatus::Error,
        snapshot.rocmRootPresent
            ? ("Detected at " + snapshot.rocmRoot)
            : ("Missing expected ROCm root at " + snapshot.rocmRoot),
        "Install ROCm or set ROCM_PATH to the active ROCm prefix."
    });

    RuntimeDiagnosticStatus kfdStatus = RuntimeDiagnosticStatus::Error;
    std::string kfdDetail;
    if (!snapshot.kfdDevicePresent) {
        kfdDetail = "/dev/kfd is missing.";
    } else if (!snapshot.kfdAccessible) {
        kfdStatus = RuntimeDiagnosticStatus::Warning;
        kfdDetail = "/dev/kfd exists but the current user cannot read/write it.";
    } else {
        kfdStatus = RuntimeDiagnosticStatus::Ok;
        kfdDetail = "Current user can access /dev/kfd.";
    }
    report.checks.push_back({
        "kfd-access",
        "KFD access",
        kfdStatus,
        kfdDetail,
        "Add the user to the render/video groups or fix udev permissions for /dev/kfd."
    });

    report.checks.push_back({
        "rocminfo",
        "ROCm visibility",
        snapshot.rocminfoPresent ? RuntimeDiagnosticStatus::Ok : RuntimeDiagnosticStatus::Warning,
        snapshot.rocminfoPresent
            ? "rocminfo is available for runtime probing."
            : "rocminfo is not on PATH.",
        "Install the ROCm runtime tools package so rocminfo can confirm device visibility."
    });

    report.checks.push_back({
        "hipinfo",
        "HIP visibility",
        snapshot.hipInfoPresent ? RuntimeDiagnosticStatus::Ok : RuntimeDiagnosticStatus::Warning,
        snapshot.hipInfoPresent
            ? "hipInfo/hipinfo is available for HIP runtime checks."
            : "hipInfo/hipinfo is not on PATH.",
        "Install HIP runtime utilities or ensure hipInfo is exported in PATH."
    });

    RuntimeDiagnosticStatus migraphxStatus = RuntimeDiagnosticStatus::Error;
    std::string migraphxDetail;
    if (snapshot.migraphxDriverPresent && snapshot.migraphxLibraryPresent) {
        migraphxStatus = RuntimeDiagnosticStatus::Ok;
        migraphxDetail = "Driver and shared library were detected.";
    } else if (snapshot.migraphxDriverPresent || snapshot.migraphxLibraryPresent) {
        migraphxStatus = RuntimeDiagnosticStatus::Warning;
        migraphxDetail = "MiGraphX is only partially installed.";
    } else {
        migraphxDetail = "MiGraphX driver/library could not be found.";
    }
    report.checks.push_back({
        "migraphx",
        "MiGraphX runtime",
        migraphxStatus,
        migraphxDetail,
        "Install migraphx-driver plus libmigraphx.so, or bundle the optimized runtime with the app."
    });

    RuntimeDiagnosticStatus rocmHipStatus = RuntimeDiagnosticStatus::Error;
    std::string rocmHipDetail;
    if (snapshot.onnxruntimeLibraryPresent && snapshot.onnxruntimeRocmProviderPresent) {
        rocmHipStatus = RuntimeDiagnosticStatus::Ok;
        rocmHipDetail = "ONNX Runtime and the ROCMExecutionProvider were detected.";
    } else if (snapshot.onnxruntimeLibraryPresent || snapshot.onnxruntimeRocmProviderPresent) {
        rocmHipStatus = RuntimeDiagnosticStatus::Warning;
        rocmHipDetail = "ONNX Runtime ROCm support is only partially available.";
    } else {
        rocmHipDetail = "ONNX Runtime ROCm execution provider could not be detected.";
    }
    report.checks.push_back({
        "onnxruntime-rocm",
        "ROCm/HIP fallback runtime",
        rocmHipStatus,
        rocmHipDetail,
        "Install an ONNX Runtime build that includes ROCMExecutionProvider support and keep libamdhip64 available at runtime."
    });

    RuntimeDiagnosticStatus vulkanStatus = RuntimeDiagnosticStatus::Error;
    std::string vulkanDetail;
    if (snapshot.vulkanLoaderPresent) {
        vulkanStatus = RuntimeDiagnosticStatus::Ok;
        vulkanDetail = "Vulkan loader/runtime detected.";
    } else {
        vulkanDetail = "Vulkan loader/runtime not detected.";
    }
    report.checks.push_back({
        "vulkan",
        "Vulkan fallback",
        vulkanStatus,
        vulkanDetail,
        "Install libvulkan and a working ICD to keep Vulkan/NCNN fallback paths available."
    });

    report.migraphxReady =
        snapshot.rocmRootPresent &&
        snapshot.kfdDevicePresent &&
        snapshot.kfdAccessible &&
        snapshot.migraphxDriverPresent &&
        snapshot.migraphxLibraryPresent &&
        snapshot.hipRuntimePresent;
    report.rocmHipFallbackReady =
        snapshot.rocmRootPresent &&
        snapshot.kfdDevicePresent &&
        snapshot.kfdAccessible &&
        snapshot.hipRuntimePresent &&
        snapshot.onnxruntimeLibraryPresent &&
        snapshot.onnxruntimeRocmProviderPresent;
    report.ncnnFallbackReady = snapshot.ncnnRuntimePresent;
    return report;
}

RuntimeDiagnosticsReport collectRuntimeDiagnostics() {
    return buildRuntimeDiagnosticsReport(probeAmdRuntimeSnapshot());
}

std::string RuntimeDiagnosticsReport::summary() const {
    if (migraphxReady) {
        return "ROCm and MiGraphX look ready for the preferred AMD execution path.";
    }
    if (rocmHipFallbackReady) {
        return "MiGraphX is not fully ready; ONNX Runtime ROCm/HIP fallback should cover supported ONNX image models.";
    }
    if (ncnnFallbackReady) {
        return "MiGraphX and ROCm/HIP fallback are not fully ready; Vulkan/NCNN fallback should remain available.";
    }
    return "MiGraphX, ROCm/HIP fallback, and Vulkan fallback all look incomplete.";
}

std::string RuntimeDiagnosticsReport::detailedText(const bool includeRemediation) const {
    std::ostringstream out;
    out << summary();
    for (const auto& check : checks) {
        out << '\n' << "- [" << toString(check.status) << "] "
            << check.title << ": " << check.detail;
        if (includeRemediation &&
            check.status != RuntimeDiagnosticStatus::Ok &&
            !check.remediation.empty()) {
            out << " Fix: " << check.remediation;
        }
    }
    const std::string failureSummary = summarizeFailures(checks);
    if (!failureSummary.empty()) {
        out << "\nFallback rationale: " << failureSummary;
    }
    return out.str();
}

}  // namespace ave
