#pragma once

#include <optional>
#include <string>
#include <vector>

namespace ave {

enum class RuntimeDiagnosticStatus {
    Ok,
    Warning,
    Error
};

std::string toString(RuntimeDiagnosticStatus status);

struct RuntimeDiagnosticCheck {
    std::string id;
    std::string title;
    RuntimeDiagnosticStatus status = RuntimeDiagnosticStatus::Error;
    std::string detail;
    std::string remediation;
};

struct AmdRuntimeSnapshot {
    std::string rocmRoot = "/opt/rocm";
    bool rocmRootPresent = false;
    bool kfdDevicePresent = false;
    bool kfdAccessible = false;
    bool rocminfoPresent = false;
    bool hipInfoPresent = false;
    bool migraphxDriverPresent = false;
    bool migraphxLibraryPresent = false;
    bool hipRuntimePresent = false;
    bool vulkanLoaderPresent = false;
    bool ncnnRuntimePresent = false;
    std::string gpuArch;
};

struct RuntimeDiagnosticsReport {
    bool migraphxReady = false;
    bool ncnnFallbackReady = false;
    std::vector<RuntimeDiagnosticCheck> checks;

    std::string summary() const;
    std::string detailedText(bool includeRemediation = true) const;
};

AmdRuntimeSnapshot probeAmdRuntimeSnapshot();
RuntimeDiagnosticsReport buildRuntimeDiagnosticsReport(const AmdRuntimeSnapshot& snapshot);
RuntimeDiagnosticsReport collectRuntimeDiagnostics();
std::optional<int> preferredAmdDeviceIndexFromEnv();
std::string detectAmdGpuArch();
std::string detectRocmVersion();

}  // namespace ave
