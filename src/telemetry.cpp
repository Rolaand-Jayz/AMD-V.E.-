#include "ave/telemetry.hpp"

#include <cctype>
#include <cstdlib>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#if defined(__unix__)
#  include <sys/wait.h>
#endif

#include "ave/process_observer.hpp"

namespace ave {
namespace {

struct ParsedCardTelemetry {
    std::string gpuId;
    std::optional<int> gpuUsePercent;
    std::optional<int> vramPercent;
    std::optional<int> memoryActivityPercent;
    std::optional<int> edgeTempC;
    std::optional<int> junctionTempC;
    std::optional<int> sclkMHz;
    std::optional<int> mclkMHz;
};

std::string buildTelemetryCommand() {
    std::ostringstream cmd;
    cmd << "LC_ALL=C ";
    if (process_observer::commandInPath("timeout")) {
        cmd << "timeout --foreground 3s ";
    }
    cmd << "rocm-smi --showuse --showmemuse --showtemp --showclocks --json 2>/dev/null";
    return cmd.str();
}

std::string escapeRegex(const std::string& input) {
    static const std::regex special{R"([-[\]{}()*+?.,\^$|#\s])"};
    return std::regex_replace(input, special, R"(\$&)");
}

std::optional<int> firstInteger(const std::string& input) {
    std::string digits;
    bool seenDigit = false;
    for (const char ch : input) {
        if (std::isdigit(static_cast<unsigned char>(ch)) != 0) {
            digits.push_back(ch);
            seenDigit = true;
        } else if (ch == '.' && seenDigit) {
            break;
        } else if (seenDigit) {
            break;
        }
    }
    if (digits.empty()) {
        return std::nullopt;
    }
    try {
        return std::stoi(digits);
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<int> extractMetric(const std::string& body, const std::string& key) {
    const std::regex pattern("\"" + escapeRegex(key) + "\"\\s*:\\s*\"([^\"]+)\"");
    std::smatch match;
    if (!std::regex_search(body, match, pattern) || match.size() < 2) {
        return std::nullopt;
    }
    return firstInteger(match[1].str());
}

std::size_t findMatchingBrace(const std::string& json, const std::size_t openPos) {
    int depth = 0;
    bool inString = false;
    bool escaped = false;
    for (std::size_t i = openPos; i < json.size(); ++i) {
        const char ch = json[i];
        if (inString) {
            if (escaped) {
                escaped = false;
            } else if (ch == '\\') {
                escaped = true;
            } else if (ch == '"') {
                inString = false;
            }
            continue;
        }

        if (ch == '"') {
            inString = true;
            continue;
        }
        if (ch == '{') {
            ++depth;
        } else if (ch == '}') {
            --depth;
            if (depth == 0) {
                return i;
            }
        }
    }
    return std::string::npos;
}

std::vector<ParsedCardTelemetry> parseCardTelemetry(const std::string& json) {
    std::vector<ParsedCardTelemetry> cards;
    std::size_t cursor = 0;
    while ((cursor = json.find("\"card", cursor)) != std::string::npos) {
        const std::size_t keyEnd = json.find('"', cursor + 1);
        if (keyEnd == std::string::npos) {
            break;
        }
        const std::size_t bodyStart = json.find('{', keyEnd);
        if (bodyStart == std::string::npos) {
            break;
        }
        const std::size_t bodyEnd = findMatchingBrace(json, bodyStart);
        if (bodyEnd == std::string::npos) {
            break;
        }

        ParsedCardTelemetry card;
        card.gpuId = json.substr(cursor + 1, keyEnd - cursor - 1);
        const std::string body = json.substr(bodyStart, bodyEnd - bodyStart + 1);
        card.gpuUsePercent = extractMetric(body, "GPU use (%)");
        card.vramPercent = extractMetric(body, "GPU Memory Allocated (VRAM%)");
        card.memoryActivityPercent = extractMetric(body, "GPU Memory Read/Write Activity (%)");
        card.edgeTempC = extractMetric(body, "Temperature (Sensor edge) (C)");
        card.junctionTempC = extractMetric(body, "Temperature (Sensor junction) (C)");
        card.sclkMHz = extractMetric(body, "sclk clock speed:");
        card.mclkMHz = extractMetric(body, "mclk clock speed:");
        cards.push_back(std::move(card));
        cursor = bodyEnd + 1;
    }
    return cards;
}

std::optional<std::string> preferredVisibleCard() {
    const char* visible = std::getenv("ROCR_VISIBLE_DEVICES");
    if (visible == nullptr || *visible == '\0') {
        visible = std::getenv("HIP_VISIBLE_DEVICES");
    }
    if (visible == nullptr || *visible == '\0') {
        return std::nullopt;
    }

    const std::string value(visible);
    if (value.find(',') != std::string::npos) {
        return std::nullopt;
    }

    const auto index = firstInteger(value);
    if (!index.has_value()) {
        return std::nullopt;
    }
    return "card" + std::to_string(*index);
}

int scoreCard(const ParsedCardTelemetry& card) {
    const int useScore = card.gpuUsePercent.value_or(0) * 10000;
    const int vramScore = card.vramPercent.value_or(0) * 100;
    const int tempScore = card.junctionTempC.value_or(card.edgeTempC.value_or(0));
    return useScore + vramScore + tempScore;
}

const ParsedCardTelemetry* selectPreferredCard(const std::vector<ParsedCardTelemetry>& cards) {
    if (cards.empty()) {
        return nullptr;
    }

    if (const auto preferred = preferredVisibleCard(); preferred.has_value()) {
        for (const auto& card : cards) {
            if (card.gpuId == *preferred) {
                return &card;
            }
        }
    }

    const ParsedCardTelemetry* best = &cards.front();
    int bestScore = scoreCard(*best);
    for (const auto& card : cards) {
        const int currentScore = scoreCard(card);
        if (currentScore > bestScore) {
            best = &card;
            bestScore = currentScore;
        }
    }
    return best;
}

void appendMetric(std::ostringstream& out,
                  bool& hasMetric,
                  const std::string& label,
                  const std::optional<int>& value,
                  const std::string& suffix = {}) {
    if (!value.has_value()) {
        return;
    }
    if (hasMetric) {
        out << " | ";
    }
    hasMetric = true;
    out << label << ' ' << *value << suffix;
}

}  // namespace

std::string AmdTelemetryProbe::summary() const {
    if (supported) {
        return detail.empty() ? "AMD telemetry ready." : detail;
    }
    if (!detail.empty()) {
        return detail;
    }
    return "AMD telemetry unavailable.";
}

std::string AmdTelemetrySnapshot::pressureHint() const {
    if (vramPercent.has_value() && *vramPercent >= 90) {
        return "VRAM pressure high; smaller tiles or a lighter backend may be safer.";
    }
    if (vramPercent.has_value() && *vramPercent >= 75) {
        return "VRAM pressure is elevated.";
    }
    return {};
}

std::string AmdTelemetrySnapshot::summary() const {
    if (!available) {
        return detail.empty() ? "AMD telemetry unavailable." : detail;
    }

    std::ostringstream out;
    out << source;
    if (!gpuId.empty()) {
        out << " " << gpuId;
    }

    bool hasMetric = false;
    if (gpuUsePercent.has_value() || vramPercent.has_value() ||
        edgeTempC.has_value() || junctionTempC.has_value() ||
        sclkMHz.has_value() || mclkMHz.has_value()) {
        out << " | ";
    }
    appendMetric(out, hasMetric, "GPU", gpuUsePercent, "%");
    appendMetric(out, hasMetric, "VRAM", vramPercent, "%");
    appendMetric(out, hasMetric, "Mem", memoryActivityPercent, "%");
    appendMetric(out, hasMetric, "Edge", edgeTempC, "C");
    appendMetric(out, hasMetric, "Junction", junctionTempC, "C");
    appendMetric(out, hasMetric, "SCLK", sclkMHz, "MHz");
    appendMetric(out, hasMetric, "MCLK", mclkMHz, "MHz");

    if (const std::string hint = pressureHint(); !hint.empty()) {
        out << " | " << hint;
    }
    return out.str();
}

AmdTelemetryProbe probeAmdTelemetrySupport() {
    AmdTelemetryProbe probe;
    probe.source = "rocm-smi";
    if (!process_observer::commandInPath("rocm-smi")) {
        probe.detail = "AMD telemetry unavailable: rocm-smi is not on PATH.";
        probe.remediation = "Install the rocm-smi package or add /opt/rocm/bin to PATH.";
        return probe;
    }

    probe.supported = true;
    probe.detail = "AMD telemetry ready via rocm-smi.";
    return probe;
}

AmdTelemetrySnapshot parseAmdTelemetryJson(const std::string& json, std::string& error) {
    AmdTelemetrySnapshot snapshot;
    const auto cards = parseCardTelemetry(json);
    const ParsedCardTelemetry* selected = selectPreferredCard(cards);
    if (selected == nullptr) {
        error = "rocm-smi did not return any GPU telemetry cards.";
        snapshot.detail = error;
        return snapshot;
    }

    snapshot.available = true;
    snapshot.source = "rocm-smi";
    snapshot.gpuId = selected->gpuId;
    snapshot.gpuUsePercent = selected->gpuUsePercent;
    snapshot.vramPercent = selected->vramPercent;
    snapshot.memoryActivityPercent = selected->memoryActivityPercent;
    snapshot.edgeTempC = selected->edgeTempC;
    snapshot.junctionTempC = selected->junctionTempC;
    snapshot.sclkMHz = selected->sclkMHz;
    snapshot.mclkMHz = selected->mclkMHz;
    return snapshot;
}

std::optional<AmdTelemetrySnapshot> collectAmdTelemetry(std::string& error) {
    const auto probe = probeAmdTelemetrySupport();
    if (!probe.supported) {
        error = probe.summary();
        return std::nullopt;
    }

    int exitCode = -1;
    const auto output = process_observer::captureCommandStdout(buildTelemetryCommand(), exitCode);
    if (exitCode == 124) {
        error = "rocm-smi timed out while sampling AMD telemetry.";
        return std::nullopt;
    }
    if (!output.has_value() || exitCode != 0) {
        error = "rocm-smi failed while sampling AMD telemetry.";
        return std::nullopt;
    }
    if (output->empty()) {
        error = "rocm-smi returned no telemetry output.";
        return std::nullopt;
    }

    AmdTelemetrySnapshot snapshot = parseAmdTelemetryJson(*output, error);
    if (!snapshot.available) {
        return std::nullopt;
    }
    error.clear();
    return snapshot;
}

}  // namespace ave
