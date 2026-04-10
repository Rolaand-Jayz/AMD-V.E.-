#pragma once

#include <optional>
#include <string>

namespace ave {

struct AmdTelemetryProbe {
    bool supported = false;
    std::string source;
    std::string detail;
    std::string remediation;

    std::string summary() const;
};

struct AmdTelemetrySnapshot {
    bool available = false;
    std::string source;
    std::string gpuId;
    std::optional<int> gpuUsePercent;
    std::optional<int> vramPercent;
    std::optional<int> memoryActivityPercent;
    std::optional<int> edgeTempC;
    std::optional<int> junctionTempC;
    std::optional<int> sclkMHz;
    std::optional<int> mclkMHz;
    std::string detail;

    std::string pressureHint() const;
    std::string summary() const;
};

AmdTelemetryProbe probeAmdTelemetrySupport();
std::optional<AmdTelemetrySnapshot> collectAmdTelemetry(std::string& error);
AmdTelemetrySnapshot parseAmdTelemetryJson(const std::string& json, std::string& error);

}  // namespace ave
