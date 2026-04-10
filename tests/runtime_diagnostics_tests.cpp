#include <cstdlib>
#include <iostream>

#include "ave/runtime_diagnostics.hpp"

namespace {

void check(const bool condition, const char* message) {
    if (condition) {
        return;
    }
    std::cerr << "runtime_diagnostics_tests failed: " << message << '\n';
    std::abort();
}

void testMigraphxReadyReport() {
    ave::AmdRuntimeSnapshot snapshot;
    snapshot.rocmRoot = "/opt/rocm";
    snapshot.rocmRootPresent = true;
    snapshot.kfdDevicePresent = true;
    snapshot.kfdAccessible = true;
    snapshot.rocminfoPresent = true;
    snapshot.hipInfoPresent = true;
    snapshot.migraphxDriverPresent = true;
    snapshot.migraphxLibraryPresent = true;
    snapshot.hipRuntimePresent = true;
    snapshot.vulkanLoaderPresent = true;
    snapshot.ncnnRuntimePresent = true;

    const auto report = ave::buildRuntimeDiagnosticsReport(snapshot);
    check(report.migraphxReady, "fully provisioned ROCm snapshot should be migraphx ready");
    check(report.ncnnFallbackReady, "vulkan-ready snapshot should also keep ncnn fallback ready");
    check(report.summary().find("preferred AMD execution path") != std::string::npos,
          "ready summary should mention preferred AMD execution path");
}

void testFallbackReportIncludesRemediation() {
    ave::AmdRuntimeSnapshot snapshot;
    snapshot.rocmRoot = "/missing/rocm";
    snapshot.rocmRootPresent = false;
    snapshot.kfdDevicePresent = true;
    snapshot.kfdAccessible = false;
    snapshot.rocminfoPresent = false;
    snapshot.hipInfoPresent = false;
    snapshot.migraphxDriverPresent = false;
    snapshot.migraphxLibraryPresent = false;
    snapshot.hipRuntimePresent = false;
    snapshot.vulkanLoaderPresent = true;
    snapshot.ncnnRuntimePresent = true;

    const auto report = ave::buildRuntimeDiagnosticsReport(snapshot);
    check(!report.migraphxReady, "incomplete snapshot should not mark migraphx ready");
    check(report.ncnnFallbackReady, "vulkan fallback should still be marked ready");
    const std::string detailed = report.detailedText();
    check(detailed.find("Fix: Install ROCm") != std::string::npos,
          "detailed report should include remediation text");
    check(detailed.find("Fallback rationale:") != std::string::npos,
          "detailed report should include fallback rationale");
}

void testStatusToString() {
    check(ave::toString(ave::RuntimeDiagnosticStatus::Ok) == "ok",
          "ok status string mismatch");
    check(ave::toString(ave::RuntimeDiagnosticStatus::Warning) == "warning",
          "warning status string mismatch");
    check(ave::toString(ave::RuntimeDiagnosticStatus::Error) == "error",
          "error status string mismatch");
}

void testSharedGpuArchProbeRespectsEnvironmentOverride() {
    setenv("HSA_OVERRIDE_GFX_VERSION", "gfx1101:test", 1);
    check(ave::detectAmdGpuArch() == "gfx1101:test",
          "shared GPU arch probe should honor the HSA override before probing runtime tools");
}

void testPreferredAmdDeviceIndexFromEnvironment() {
    unsetenv("AVE_MIGRAPHX_DEVICE_INDEX");
    setenv("AVE_GPU_INDEX", "3", 1);
    check(ave::preferredAmdDeviceIndexFromEnv().has_value(),
          "shared AMD device selection should parse AVE_GPU_INDEX");
    check(*ave::preferredAmdDeviceIndexFromEnv() == 3,
          "shared AMD device selection should use AVE_GPU_INDEX when no explicit MiGraphX index is set");
}

}  // namespace

int main() {
    testSharedGpuArchProbeRespectsEnvironmentOverride();
    testPreferredAmdDeviceIndexFromEnvironment();
    testMigraphxReadyReport();
    testFallbackReportIncludesRemediation();
    testStatusToString();
    return 0;
}
