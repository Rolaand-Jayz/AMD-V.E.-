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
ParameterValue parameterValueFromString(const std::string& value);
std::optional<EnhancementStage> parseStageSpec(const std::string& spec,
                                               std::string& error);
std::string scopedStageParamKey(StageKind kind, const std::string& key);
bool tryGetDouble(const std::unordered_map<std::string, ParameterValue>& params,
                  const std::string& key,
                  double& outValue);
bool tryGetInt(const std::unordered_map<std::string, ParameterValue>& params,
               const std::string& key,
               std::int64_t& outValue);
bool tryGetDouble(const EnhancementStage& stage,
                  StageKind scope,
                  const std::string& key,
                  double& outValue);
bool tryGetInt(const EnhancementStage& stage,
               StageKind scope,
               const std::string& key,
               std::int64_t& outValue);
bool tryGetBool(const EnhancementStage& stage,
                StageKind scope,
                const std::string& key,
                bool& outValue);
bool tryGetString(const EnhancementStage& stage,
                  StageKind scope,
                  const std::string& key,
                  std::string& outValue);

}  // namespace ave
