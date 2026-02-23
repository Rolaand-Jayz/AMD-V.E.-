#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <variant>

#include "ave/types.hpp"

namespace ave {

using ParameterValue = std::variant<std::int64_t, double, bool, std::string>;

struct EnhancementStage {
    StageKind kind;
    std::unordered_map<std::string, ParameterValue> params;

    // Set to true by VideoProcessor when the accelerator backend has
    // already performed AI inference for this stage.  When true, the
    // FFmpeg encode pass should NOT re-apply its own filter for this
    // stage (otherwise the enhancement would be applied twice, or a
    // weaker FFmpeg filter would overwrite the AI result).
    bool backendProcessed = false;
};

std::string parameterValueToString(const ParameterValue& value);
bool tryGetDouble(const std::unordered_map<std::string, ParameterValue>& params,
                  const std::string& key,
                  double& outValue);
bool tryGetInt(const std::unordered_map<std::string, ParameterValue>& params,
               const std::string& key,
               std::int64_t& outValue);

}  // namespace ave
