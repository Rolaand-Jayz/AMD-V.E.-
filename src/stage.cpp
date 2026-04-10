#include "ave/stage.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <vector>

namespace ave {
namespace {

std::string trimStageValue(const std::string& input) {
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

std::string toLowerStageValue(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) {
                       return static_cast<char>(std::tolower(ch));
                   });
    return value;
}

std::vector<std::string> splitStageValue(const std::string& value,
                                         const char delimiter) {
    std::vector<std::string> parts;
    std::size_t start = 0;
    while (start <= value.size()) {
        std::size_t end = value.find(delimiter, start);
        if (end == std::string::npos) {
            end = value.size();
        }
        parts.push_back(value.substr(start, end - start));
        if (end == value.size()) {
            break;
        }
        start = end + 1;
    }
    return parts;
}

}  // namespace

std::string parameterValueToString(const ParameterValue& value) {
    if (std::holds_alternative<std::int64_t>(value)) {
        return std::to_string(std::get<std::int64_t>(value));
    }
    if (std::holds_alternative<double>(value)) {
        std::ostringstream os;
        os << std::setprecision(12) << std::get<double>(value);
        return os.str();
    }
    if (std::holds_alternative<bool>(value)) {
        return std::get<bool>(value) ? "true" : "false";
    }
    return std::get<std::string>(value);
}

ParameterValue parameterValueFromString(const std::string& value) {
    const std::string trimmed = trimStageValue(value);
    const std::string lowered = toLowerStageValue(trimmed);
    if (lowered == "true") {
        return true;
    }
    if (lowered == "false") {
        return false;
    }

    char* intEnd = nullptr;
    const long long intParsed = std::strtoll(trimmed.c_str(), &intEnd, 10);
    if (intEnd != trimmed.c_str() && *intEnd == '\0') {
        return static_cast<std::int64_t>(intParsed);
    }

    char* floatEnd = nullptr;
    const double floatParsed = std::strtod(trimmed.c_str(), &floatEnd);
    if (floatEnd != trimmed.c_str() && *floatEnd == '\0') {
        return floatParsed;
    }

    return trimmed;
}

std::optional<EnhancementStage> parseStageSpec(const std::string& spec,
                                               std::string& error) {
    const std::size_t colon = spec.find(':');
    const std::string kindToken =
        colon == std::string::npos ? spec : spec.substr(0, colon);
    const std::optional<StageKind> kind =
        stageKindFromString(trimStageValue(kindToken));
    if (!kind.has_value()) {
        error = "Unknown stage kind: " + kindToken;
        return std::nullopt;
    }

    EnhancementStage stage;
    stage.kind = *kind;

    if (colon == std::string::npos) {
        error.clear();
        return stage;
    }

    const std::string paramString = spec.substr(colon + 1);
    if (paramString.empty()) {
        error.clear();
        return stage;
    }

    const std::vector<std::string> assignments =
        splitStageValue(paramString, ',');
    for (const std::string& assignment : assignments) {
        const std::size_t eq = assignment.find('=');
        if (eq == std::string::npos) {
            error = "Malformed stage parameter in: " + assignment;
            return std::nullopt;
        }

        const std::string key = trimStageValue(assignment.substr(0, eq));
        const std::string rawValue = assignment.substr(eq + 1);
        if (key.empty()) {
            error = "Empty stage parameter key in: " + assignment;
            return std::nullopt;
        }

        stage.params[key] = parameterValueFromString(rawValue);
    }

    error.clear();
    return stage;
}

std::string scopedStageParamKey(const StageKind kind, const std::string& key) {
    return "fused." + toString(kind) + "." + key;
}

bool tryGetDouble(const std::unordered_map<std::string, ParameterValue>& params,
                  const std::string& key,
                  double& outValue) {
    const auto it = params.find(key);
    if (it == params.end()) {
        return false;
    }

    if (const auto* val = std::get_if<double>(&it->second)) {
        outValue = *val;
        return true;
    }

    if (const auto* val = std::get_if<std::int64_t>(&it->second)) {
        outValue = static_cast<double>(*val);
        return true;
    }

    if (const auto* val = std::get_if<std::string>(&it->second)) {
        char* end = nullptr;
        const double parsed = std::strtod(val->c_str(), &end);
        if (end != val->c_str() && *end == '\0') {
            outValue = parsed;
            return true;
        }
    }

    return false;
}

bool tryGetInt(const std::unordered_map<std::string, ParameterValue>& params,
               const std::string& key,
               std::int64_t& outValue) {
    const auto it = params.find(key);
    if (it == params.end()) {
        return false;
    }

    if (const auto* val = std::get_if<std::int64_t>(&it->second)) {
        outValue = *val;
        return true;
    }

    if (const auto* val = std::get_if<double>(&it->second)) {
        outValue = static_cast<std::int64_t>(*val);
        return true;
    }

    if (const auto* val = std::get_if<std::string>(&it->second)) {
        char* end = nullptr;
        const long long parsed = std::strtoll(val->c_str(), &end, 10);
        if (end != val->c_str() && *end == '\0') {
            outValue = static_cast<std::int64_t>(parsed);
            return true;
        }
    }

    return false;
}

bool tryGetDouble(const EnhancementStage& stage,
                  const StageKind scope,
                  const std::string& key,
                  double& outValue) {
    return tryGetDouble(stage.params, scopedStageParamKey(scope, key), outValue) ||
           tryGetDouble(stage.params, key, outValue);
}

bool tryGetInt(const EnhancementStage& stage,
               const StageKind scope,
               const std::string& key,
               std::int64_t& outValue) {
    return tryGetInt(stage.params, scopedStageParamKey(scope, key), outValue) ||
           tryGetInt(stage.params, key, outValue);
}

bool tryGetBool(const EnhancementStage& stage,
                const StageKind scope,
                const std::string& key,
                bool& outValue) {
    auto parseValue = [&](const ParameterValue& value) {
        if (const auto* booleanValue = std::get_if<bool>(&value)) {
            outValue = *booleanValue;
            return true;
        }
        if (const auto* intValue = std::get_if<std::int64_t>(&value)) {
            outValue = *intValue != 0;
            return true;
        }
        if (const auto* doubleValue = std::get_if<double>(&value)) {
            outValue = *doubleValue != 0.0;
            return true;
        }
        if (const auto* stringValue = std::get_if<std::string>(&value)) {
            if (*stringValue == "true" || *stringValue == "1" ||
                *stringValue == "yes" || *stringValue == "on") {
                outValue = true;
                return true;
            }
            if (*stringValue == "false" || *stringValue == "0" ||
                *stringValue == "no" || *stringValue == "off") {
                outValue = false;
                return true;
            }
        }
        return false;
    };

    const auto scopedIt = stage.params.find(scopedStageParamKey(scope, key));
    if (scopedIt != stage.params.end() && parseValue(scopedIt->second)) {
        return true;
    }

    const auto it = stage.params.find(key);
    if (it != stage.params.end() && parseValue(it->second)) {
        return true;
    }
    return false;
}

bool tryGetString(const EnhancementStage& stage,
                  const StageKind scope,
                  const std::string& key,
                  std::string& outValue) {
    auto parseValue = [&](const ParameterValue& value) {
        outValue = parameterValueToString(value);
        return true;
    };

    const auto scopedIt = stage.params.find(scopedStageParamKey(scope, key));
    if (scopedIt != stage.params.end() && parseValue(scopedIt->second)) {
        return true;
    }

    const auto it = stage.params.find(key);
    if (it != stage.params.end() && parseValue(it->second)) {
        return true;
    }
    return false;
}

}  // namespace ave
