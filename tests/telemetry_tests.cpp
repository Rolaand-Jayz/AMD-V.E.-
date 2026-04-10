#include <cassert>
#include <cstdlib>
#include <string>

#include "ave/telemetry.hpp"

namespace {

void testTelemetryParserSelectsBusiestCard() {
    const std::string json =
        "{\"card0\": {\"GPU use (%)\": \"5\", "
        "\"GPU Memory Allocated (VRAM%)\": \"9\", "
        "\"Temperature (Sensor edge) (C)\": \"33.0\", "
        "\"sclk clock speed:\": \"(245Mhz)\"}, "
        "\"card1\": {\"GPU use (%)\": \"76\", "
        "\"GPU Memory Allocated (VRAM%)\": \"48\", "
        "\"Temperature (Sensor edge) (C)\": \"67.0\", "
        "\"Temperature (Sensor junction) (C)\": \"81.0\", "
        "\"sclk clock speed:\": \"(2220Mhz)\", "
        "\"mclk clock speed:\": \"(1249Mhz)\"}}";

    std::string error;
    const auto snapshot = ave::parseAmdTelemetryJson(json, error);
    assert(error.empty());
    assert(snapshot.available);
    assert(snapshot.gpuId == "card1");
    assert(snapshot.gpuUsePercent == 76);
    assert(snapshot.vramPercent == 48);
    assert(snapshot.junctionTempC == 81);
    assert(snapshot.sclkMHz == 2220);
}

void testTelemetryPressureHintAppearsInSummary() {
    ave::AmdTelemetrySnapshot snapshot;
    snapshot.available = true;
    snapshot.source = "rocm-smi";
    snapshot.gpuId = "card0";
    snapshot.gpuUsePercent = 97;
    snapshot.vramPercent = 92;
    snapshot.edgeTempC = 74;

    const std::string summary = snapshot.summary();
    assert(summary.find("VRAM 92%") != std::string::npos);
    assert(summary.find("VRAM pressure high") != std::string::npos);
}

void testTelemetryProbeSummaryWhenUnsupported() {
    const char* originalPath = std::getenv("PATH");
    if (originalPath != nullptr) {
        setenv("PATH", "", 1);
    }
    const auto probe = ave::probeAmdTelemetrySupport();
    assert(!probe.supported);
    assert(probe.summary().find("rocm-smi") != std::string::npos);
    if (originalPath != nullptr) {
        setenv("PATH", originalPath, 1);
    }
}

}  // namespace

int main() {
    testTelemetryParserSelectsBusiestCard();
    testTelemetryPressureHintAppearsInSummary();
    testTelemetryProbeSummaryWhenUnsupported();
    return 0;
}
